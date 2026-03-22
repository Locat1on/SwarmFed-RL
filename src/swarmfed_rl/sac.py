from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast
from torch.distributions import Normal

from .config import ExperimentConfig


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.ln1(h)
        h = self.act(h)
        h = self.fc2(h)
        h = self.ln2(h)
        return self.act(x + h)


class DeepMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, hidden_layers: int, residual: bool) -> None:
        super().__init__()
        if hidden_layers < 2:
            raise ValueError("hidden_layers must be >= 2")
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList()
        block_count = hidden_layers - 1
        if residual:
            for _ in range(block_count):
                self.blocks.append(ResidualBlock(hidden))
        else:
            for _ in range(block_count):
                self.blocks.append(
                    nn.Sequential(
                        nn.Linear(hidden, hidden),
                        nn.LayerNorm(hidden),
                        nn.ReLU(),
                    )
                )
        self.out_proj = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.out_proj(h)


class Radar1DEncoder(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.ReLU(),
        )

    def forward(self, radar: torch.Tensor) -> torch.Tensor:
        # radar: [B, 24]
        return self.net(radar.unsqueeze(1))


class RadarAttentionEncoder(nn.Module):
    def __init__(self, out_dim: int, model_dim: int, heads: int, layers: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(1, model_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 2,
            dropout=0.0,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.out_proj = nn.Sequential(
            nn.Linear(model_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, radar: torch.Tensor) -> torch.Tensor:
        # radar: [B, 24] -> tokens [B, 24, 1]
        tokens = radar.unsqueeze(-1)
        h = self.in_proj(tokens)
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.out_proj(pooled)


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int,
        hidden_layers: int,
        residual: bool,
        encoder_type: str,
        use_cnn_encoder: bool,
        attention_dim: int,
        attention_heads: int,
        attention_layers: int,
        log_std_min: float,
        log_std_max: float,
    ) -> None:
        super().__init__()
        self.encoder_type = encoder_type
        if self.encoder_type not in {"attention", "cnn", "mlp"}:
            raise ValueError(f"Unsupported actor encoder: {self.encoder_type}")
        if use_cnn_encoder and self.encoder_type == "mlp":
            self.encoder_type = "cnn"
        self.radar_dim = max(24, state_dim - 4)
        self.tail_dim = max(0, state_dim - self.radar_dim)
        if self.encoder_type in {"attention", "cnn"} and state_dim < self.radar_dim:
            raise ValueError("state_dim must be >= 24 when actor encoder uses radar split")
        if self.encoder_type == "attention":
            self.radar_encoder = RadarAttentionEncoder(
                out_dim=hidden // 2,
                model_dim=attention_dim,
                heads=attention_heads,
                layers=attention_layers,
            )
            self.tail_encoder = nn.Sequential(
                nn.Linear(self.tail_dim, hidden // 2),
                nn.ReLU(),
            )
            self.backbone = DeepMLP(hidden, hidden, hidden, hidden_layers, residual)
        elif self.encoder_type == "cnn":
            self.radar_encoder = Radar1DEncoder(out_dim=hidden // 2)
            self.tail_encoder = nn.Sequential(
                nn.Linear(self.tail_dim, hidden // 2),
                nn.ReLU(),
            )
            self.backbone = DeepMLP(hidden, hidden, hidden, hidden_layers, residual)
        else:
            self.backbone = DeepMLP(state_dim, hidden, hidden, hidden_layers, residual)
        self.mu = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.encoder_type in {"attention", "cnn"}:
            radar = state[..., : self.radar_dim]
            tail = state[..., self.radar_dim :]
            radar_feat = self.radar_encoder(radar)
            tail_feat = self.tail_encoder(tail)
            h = self.backbone(torch.cat([radar_feat, tail_feat], dim=-1))
        else:
            h = self.backbone(state)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int, hidden_layers: int, residual: bool) -> None:
        super().__init__()
        self.q = DeepMLP(state_dim + action_dim, 1, hidden, hidden_layers, residual)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([state, action], dim=-1))


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int) -> None:
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        i = self.ptr
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[idx], device=device),
            torch.as_tensor(self.actions[idx], device=device),
            torch.as_tensor(self.rewards[idx], device=device),
            torch.as_tensor(self.next_states[idx], device=device),
            torch.as_tensor(self.dones[idx], device=device),
        )


class GPUReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int, device: torch.device) -> None:
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.device = device
        self.states = torch.empty((capacity, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.empty((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.empty((capacity, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.empty((capacity, 1), dtype=torch.float32, device=device)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        i = self.ptr
        self.states[i].copy_(torch.as_tensor(state, dtype=torch.float32, device=self.device))
        self.actions[i].copy_(torch.as_tensor(action, dtype=torch.float32, device=self.device))
        self.rewards[i, 0] = float(reward)
        self.next_states[i].copy_(torch.as_tensor(next_state, dtype=torch.float32, device=self.device))
        self.dones[i, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states.index_select(0, idx),
            self.actions.index_select(0, idx),
            self.rewards.index_select(0, idx),
            self.next_states.index_select(0, idx),
            self.dones.index_select(0, idx),
        )


@dataclass
class SACAgent:
    cfg: ExperimentConfig
    device: torch.device

    def __post_init__(self) -> None:
        hidden = self.cfg.sac.hidden_size
        self.actor = Actor(
            self.cfg.state_dim,
            self.cfg.action_dim,
            hidden,
            self.cfg.sac.hidden_layers,
            self.cfg.sac.residual,
            self.cfg.sac.actor_encoder,
            self.cfg.sac.actor_use_cnn,
            self.cfg.sac.attention_dim,
            self.cfg.sac.attention_heads,
            self.cfg.sac.attention_layers,
            self.cfg.sac.log_std_min,
            self.cfg.sac.log_std_max,
        ).to(self.device)
        self.q1 = Critic(
            self.cfg.state_dim,
            self.cfg.action_dim,
            hidden,
            self.cfg.sac.hidden_layers,
            self.cfg.sac.residual,
        ).to(self.device)
        self.q2 = Critic(
            self.cfg.state_dim,
            self.cfg.action_dim,
            hidden,
            self.cfg.sac.hidden_layers,
            self.cfg.sac.residual,
        ).to(self.device)
        self.q1_target = Critic(
            self.cfg.state_dim,
            self.cfg.action_dim,
            hidden,
            self.cfg.sac.hidden_layers,
            self.cfg.sac.residual,
        ).to(self.device)
        self.q2_target = Critic(
            self.cfg.state_dim,
            self.cfg.action_dim,
            hidden,
            self.cfg.sac.hidden_layers,
            self.cfg.sac.residual,
        ).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        if self.cfg.sac.enable_torch_compile and hasattr(torch, "compile"):
            self.actor = torch.compile(self.actor, mode=self.cfg.sac.compile_mode)  # type: ignore[assignment]
            self.q1 = torch.compile(self.q1, mode=self.cfg.sac.compile_mode)  # type: ignore[assignment]
            self.q2 = torch.compile(self.q2, mode=self.cfg.sac.compile_mode)  # type: ignore[assignment]
            self.q1_target = torch.compile(self.q1_target, mode=self.cfg.sac.compile_mode)  # type: ignore[assignment]
            self.q2_target = torch.compile(self.q2_target, mode=self.cfg.sac.compile_mode)  # type: ignore[assignment]

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.sac.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=self.cfg.sac.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=self.cfg.sac.critic_lr)
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg.sac.alpha_lr)
        self.target_entropy = -float(self.cfg.action_dim)
        self.use_amp = bool(self.cfg.sac.use_amp and self.device.type == "cuda")
        amp_dtype_raw = str(self.cfg.sac.amp_dtype).lower()
        self.amp_dtype = torch.bfloat16 if amp_dtype_raw == "bf16" else torch.float16
        self.autocast_device_type = "cuda" if self.device.type == "cuda" else "cpu"
        self.scaler = GradScaler("cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)
        self._train_call_count = 0
        self.effective_update_every = max(1, self.cfg.sac.update_every if self.device.type == "cuda" else 1)
        self.effective_gradient_updates = max(
            1, self.cfg.sac.gradient_updates if self.device.type == "cuda" else 1
        )

        use_gpu_replay = bool(
            self.cfg.sac.use_gpu_replay and self.device.type == "cuda"
        )
        if use_gpu_replay:
            self.buffer = GPUReplayBuffer(
                self.cfg.state_dim,
                self.cfg.action_dim,
                self.cfg.sac.buffer_size,
                self.device,
            )
        else:
            self.buffer = ReplayBuffer(self.cfg.state_dim, self.cfg.action_dim, self.cfg.sac.buffer_size)

    @staticmethod
    def _safe_loss(loss: torch.Tensor, name: str) -> None:
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss encountered: {name}")

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        s = torch.as_tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mu, _ = self.actor(s)
                act = torch.tanh(mu)
            else:
                act, _ = self.actor.sample(s)
        action = act.squeeze(0).cpu().numpy()
        return self._rescale_action(action)

    def select_actions(self, states: np.ndarray, deterministic: bool = False) -> np.ndarray:
        s = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            if deterministic:
                mu, _ = self.actor(s)
                act = torch.tanh(mu)
            else:
                act, _ = self.actor.sample(s)
        actions = act.cpu().numpy()
        return self._rescale_actions(actions)

    def _rescale_action(self, tanh_action: np.ndarray) -> np.ndarray:
        low = np.asarray(self.cfg.action_low, dtype=np.float32)
        high = np.asarray(self.cfg.action_high, dtype=np.float32)
        return low + (tanh_action + 1.0) * (high - low) / 2.0

    def _rescale_actions(self, tanh_actions: np.ndarray) -> np.ndarray:
        low = np.asarray(self.cfg.action_low, dtype=np.float32)
        high = np.asarray(self.cfg.action_high, dtype=np.float32)
        return low + (tanh_actions + 1.0) * (high - low) / 2.0

    def train_step(self) -> dict[str, float]:
        self._train_call_count += 1
        if self.buffer.size < max(self.cfg.sac.batch_size, self.cfg.sac.update_after):
            return {}
        if self._train_call_count % self.effective_update_every != 0:
            return {}

        last_metrics: dict[str, float] = {}
        for _ in range(self.effective_gradient_updates):
            last_metrics = self._train_step_once()
        return last_metrics

    def _train_step_once(self) -> dict[str, float]:
        if self.buffer.size < max(self.cfg.sac.batch_size, self.cfg.sac.update_after):
            return {}

        state, action, reward, next_state, done = self.buffer.sample(self.cfg.sac.batch_size, self.device)

        with torch.no_grad():
            with autocast(self.autocast_device_type, enabled=self.use_amp, dtype=self.amp_dtype):
                next_a, next_logp = self.actor.sample(next_state)
                next_q = torch.min(
                    self.q1_target(next_state, next_a),
                    self.q2_target(next_state, next_a),
                ) - self.alpha * next_logp
                target_q = reward + (1.0 - done) * self.cfg.sac.gamma * next_q

        with autocast(self.autocast_device_type, enabled=self.use_amp, dtype=self.amp_dtype):
            q1_loss = F.mse_loss(self.q1(state, action), target_q)
            q2_loss = F.mse_loss(self.q2(state, action), target_q)
        self._safe_loss(q1_loss, "q1_loss")
        self._safe_loss(q2_loss, "q2_loss")
        self.q1_opt.zero_grad()
        self.scaler.scale(q1_loss).backward()
        self.scaler.unscale_(self.q1_opt)
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.cfg.sac.grad_clip_norm)
        self.scaler.step(self.q1_opt)
        self.q1_opt.zero_grad(set_to_none=True)
        self.q2_opt.zero_grad(set_to_none=True)
        self.scaler.scale(q2_loss).backward()
        self.scaler.unscale_(self.q2_opt)
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.cfg.sac.grad_clip_norm)
        self.scaler.step(self.q2_opt)

        # Delayed policy update: only update actor every N critic updates
        update_actor = (self._train_call_count % self.cfg.sac.actor_update_interval) == 0
        actor_loss_val = 0.0
        if update_actor:
            with autocast(self.autocast_device_type, enabled=self.use_amp, dtype=self.amp_dtype):
                pi, logp = self.actor.sample(state)
                q_pi = torch.min(self.q1(state, pi), self.q2(state, pi))
                actor_loss = (self.alpha * logp - q_pi).mean()
            self._safe_loss(actor_loss, "actor_loss")
            self.actor_opt.zero_grad()
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_opt)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.sac.grad_clip_norm)
            self.scaler.step(self.actor_opt)
            actor_loss_val = float(actor_loss.item())

            with autocast(self.autocast_device_type, enabled=self.use_amp, dtype=self.amp_dtype):
                pi_for_alpha, logp_for_alpha = self.actor.sample(state)
                alpha_loss = -(self.log_alpha * (logp_for_alpha + self.target_entropy).detach()).mean()
            self._safe_loss(alpha_loss, "alpha_loss")
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            torch.nn.utils.clip_grad_norm_([self.log_alpha], self.cfg.sac.grad_clip_norm)
            self.alpha_opt.step()

        self.scaler.update()

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": actor_loss_val,
            "alpha": float(self.alpha.item()),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        tau = self.cfg.sac.tau
        for p, t in zip(source.parameters(), target.parameters()):
            t.data.copy_(tau * p.data + (1.0 - tau) * t.data)

    def get_actor_state(self, *, cpu_clone: bool = True) -> dict[str, torch.Tensor]:
        if cpu_clone:
            return {k: v.detach().cpu().clone() for k, v in self.actor.state_dict().items()}
        return {k: v.detach().clone() for k, v in self.actor.state_dict().items()}

    def load_actor_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        isolated = {
            k: v.detach().to(self.device).clone()
            for k, v in state_dict.items()
        }
        self.actor.load_state_dict(isolated)
