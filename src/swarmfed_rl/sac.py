"""Soft Actor-Critic agent for multi-robot obstacle avoidance.

Simplified for convergence:
- Lighter networks (256-hidden, 2-layer MLP)
- No cadence gate — trains every call
- NaN protection on all losses
- Shared replay buffer support
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast
from torch.distributions import Normal

from .config import ExperimentConfig

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Network building blocks
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Simple MLP with LayerNorm and ReLU."""
    def __init__(self, in_dim: int, out_dim: int, hidden: int, num_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(prev, hidden), nn.LayerNorm(hidden), nn.ReLU()])
            prev = hidden
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Actor / Critic
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden: int, num_layers: int,
        log_std_min: float, log_std_max: float,
    ) -> None:
        super().__init__()
        self.backbone = MLP(state_dim, hidden, hidden, num_layers)
        self.mu = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        log_prob = (dist.log_prob(z) - torch.log1p(-action.pow(2) + 1e-6)).sum(dim=-1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int, num_layers: int) -> None:
        super().__init__()
        self.q = MLP(state_dim + action_dim, 1, hidden, num_layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([state, action], dim=-1))


# ---------------------------------------------------------------------------
# Replay buffers
# ---------------------------------------------------------------------------

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

    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
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

    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        i = self.ptr
        self.states[i] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.actions[i] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[i, 0] = reward
        self.next_states[i] = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
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


# ---------------------------------------------------------------------------
# SAC Agent
# ---------------------------------------------------------------------------

@dataclass
class SACAgent:
    cfg: ExperimentConfig
    device: torch.device
    buffer: ReplayBuffer | GPUReplayBuffer | None = None  # Allow external buffer

    def __post_init__(self) -> None:
        h = self.cfg.sac.hidden_size
        nl = self.cfg.sac.hidden_layers

        self.actor = Actor(
            self.cfg.state_dim, self.cfg.action_dim, h, nl,
            self.cfg.sac.log_std_min, self.cfg.sac.log_std_max,
        ).to(self.device)
        self.q1 = Critic(self.cfg.state_dim, self.cfg.action_dim, h, nl).to(self.device)
        self.q2 = Critic(self.cfg.state_dim, self.cfg.action_dim, h, nl).to(self.device)
        self.q1_target = Critic(self.cfg.state_dim, self.cfg.action_dim, h, nl).to(self.device)
        self.q2_target = Critic(self.cfg.state_dim, self.cfg.action_dim, h, nl).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.sac.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=self.cfg.sac.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=self.cfg.sac.critic_lr)
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg.sac.alpha_lr)
        self.target_entropy = -float(self.cfg.action_dim)

        self.use_amp = bool(self.cfg.sac.use_amp and self.device.type == "cuda")
        self.amp_dtype = torch.bfloat16 if str(self.cfg.sac.amp_dtype).lower() == "bf16" else torch.float16
        self.autocast_device_type = "cuda" if self.device.type == "cuda" else "cpu"
        self.scaler = GradScaler("cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)

        self._gradient_step_count = 0
        self._training_started = False

        # Create own buffer only if none provided externally (shared replay)
        if self.buffer is None:
            use_gpu = bool(self.cfg.sac.use_gpu_replay and self.device.type == "cuda")
            if use_gpu:
                self.buffer = GPUReplayBuffer(self.cfg.state_dim, self.cfg.action_dim, self.cfg.sac.buffer_size, self.device)
            else:
                self.buffer = ReplayBuffer(self.cfg.state_dim, self.cfg.action_dim, self.cfg.sac.buffer_size)

        # Pre-compute rescale constants
        self._act_low = np.asarray(self.cfg.action_low, dtype=np.float32)
        self._act_high = np.asarray(self.cfg.action_high, dtype=np.float32)
        self._act_scale = (self._act_high - self._act_low) / 2.0
        self._act_bias = (self._act_high + self._act_low) / 2.0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # -- Action selection ------------------------------------------------

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        s = torch.as_tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mu, _ = self.actor(s)
                act = torch.tanh(mu)
            else:
                act, _ = self.actor.sample(s)
        return self._rescale(act.squeeze(0).cpu().numpy())

    def select_actions(self, states: np.ndarray, deterministic: bool = False) -> np.ndarray:
        s = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            if deterministic:
                mu, _ = self.actor(s)
                act = torch.tanh(mu)
            else:
                act, _ = self.actor.sample(s)
        return self._rescale(act.cpu().numpy())

    def _rescale(self, tanh_action: np.ndarray) -> np.ndarray:
        return tanh_action * self._act_scale + self._act_bias

    # -- Training --------------------------------------------------------

    def train_step(self, num_updates: int = 1, pull_metrics: bool = False) -> dict[str, float]:
        buf_threshold = max(self.cfg.sac.batch_size, self.cfg.sac.update_after)
        if self.buffer.size < buf_threshold:
            return {}

        if not self._training_started:
            self._training_started = True
            print(
                f"  >>> [SAC] Training started: buffer={self.buffer.size} "
                f"threshold={buf_threshold} updates_per_call={num_updates} "
                f"batch={self.cfg.sac.batch_size}",
                flush=True,
            )

        metrics: dict[str, float] = {}
        for i in range(num_updates):
            m = self._train_step_once(return_metrics=(i == num_updates - 1) and pull_metrics)
            if m:
                metrics = m

        self.scaler.update()
        self._soft_update()
        return metrics

    def _train_step_once(self, return_metrics: bool = True) -> dict[str, float]:
        self._gradient_step_count += 1
        state, action, reward, next_state, done = self.buffer.sample(self.cfg.sac.batch_size, self.device)

        # -- Critic update --
        with torch.no_grad():
            with autocast(self.autocast_device_type, enabled=self.use_amp, dtype=self.amp_dtype):
                next_a, next_logp = self.actor.sample(next_state)
                target_q = reward + (1.0 - done) * self.cfg.sac.gamma * (
                    torch.min(self.q1_target(next_state, next_a), self.q2_target(next_state, next_a))
                    - self.alpha.detach() * next_logp
                )

        with autocast(self.autocast_device_type, enabled=self.use_amp, dtype=self.amp_dtype):
            q1_loss = F.mse_loss(self.q1(state, action), target_q)
            q2_loss = F.mse_loss(self.q2(state, action), target_q)

        # NaN protection
        if torch.isnan(q1_loss) or torch.isnan(q2_loss):
            _log.warning("NaN detected in critic loss, skipping update")
            return {}

        for opt, loss, params in [
            (self.q1_opt, q1_loss, self.q1.parameters()),
            (self.q2_opt, q2_loss, self.q2.parameters()),
        ]:
            opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, self.cfg.sac.grad_clip_norm)
            self.scaler.step(opt)

        # -- Actor + Alpha update --
        actor_loss_val = 0.0
        if self._gradient_step_count % self.cfg.sac.actor_update_interval == 0:
            with autocast(self.autocast_device_type, enabled=self.use_amp, dtype=self.amp_dtype):
                pi, logp = self.actor.sample(state)
                q_pi = torch.min(self.q1(state, pi), self.q2(state, pi))
                actor_loss = (self.alpha.detach() * logp - q_pi).mean()

            if torch.isnan(actor_loss):
                _log.warning("NaN detected in actor loss, skipping update")
            else:
                self.actor_opt.zero_grad(set_to_none=True)
                self.scaler.scale(actor_loss).backward()
                self.scaler.unscale_(self.actor_opt)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.sac.grad_clip_norm)
                self.scaler.step(self.actor_opt)

                if return_metrics:
                    actor_loss_val = float(actor_loss.item())

                # Alpha loss
                alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
                if not torch.isnan(alpha_loss):
                    self.alpha_opt.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    self.alpha_opt.step()
                else:
                    _log.warning("NaN detected in alpha loss, skipping update")

        if not return_metrics:
            return {}
        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": actor_loss_val,
            "alpha": float(self.alpha.item()),
        }

    def _soft_update(self) -> None:
        tau = self.cfg.sac.tau
        with torch.no_grad():
            for p, t in zip(self.q1.parameters(), self.q1_target.parameters()):
                t.lerp_(p, tau)
            for p, t in zip(self.q2.parameters(), self.q2_target.parameters()):
                t.lerp_(p, tau)

    # -- Weight I/O for P2P exchange ------------------------------------

    def get_actor_state(self, *, cpu_clone: bool = True) -> dict[str, torch.Tensor]:
        if cpu_clone:
            return {k: v.detach().cpu().clone() for k, v in self.actor.state_dict().items()}
        return {k: v.detach().clone() for k, v in self.actor.state_dict().items()}

    def load_actor_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        moved = {k: v.to(self.device) for k, v in state_dict.items()}
        self.actor.load_state_dict(moved)
