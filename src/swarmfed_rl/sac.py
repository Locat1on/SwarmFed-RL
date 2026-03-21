from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from .config import ExperimentConfig


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int, log_std_min: float, log_std_max: float) -> None:
        super().__init__()
        self.backbone = MLP(state_dim, hidden, hidden)
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
        log_prob = dist.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int) -> None:
        super().__init__()
        self.q = MLP(state_dim + action_dim, 1, hidden)

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
            self.cfg.sac.log_std_min,
            self.cfg.sac.log_std_max,
        ).to(self.device)
        self.q1 = Critic(self.cfg.state_dim, self.cfg.action_dim, hidden).to(self.device)
        self.q2 = Critic(self.cfg.state_dim, self.cfg.action_dim, hidden).to(self.device)
        self.q1_target = Critic(self.cfg.state_dim, self.cfg.action_dim, hidden).to(self.device)
        self.q2_target = Critic(self.cfg.state_dim, self.cfg.action_dim, hidden).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.sac.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=self.cfg.sac.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=self.cfg.sac.critic_lr)
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg.sac.alpha_lr)
        self.target_entropy = -float(self.cfg.action_dim)

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

    def _rescale_action(self, tanh_action: np.ndarray) -> np.ndarray:
        low = np.asarray(self.cfg.action_low, dtype=np.float32)
        high = np.asarray(self.cfg.action_high, dtype=np.float32)
        return low + (tanh_action + 1.0) * (high - low) / 2.0

    def train_step(self) -> dict[str, float]:
        if self.buffer.size < max(self.cfg.sac.batch_size, self.cfg.sac.update_after):
            return {}

        state, action, reward, next_state, done = self.buffer.sample(self.cfg.sac.batch_size, self.device)

        with torch.no_grad():
            next_a, next_logp = self.actor.sample(next_state)
            next_q = torch.min(
                self.q1_target(next_state, next_a),
                self.q2_target(next_state, next_a),
            ) - self.alpha * next_logp
            target_q = reward + (1.0 - done) * self.cfg.sac.gamma * next_q

        q1_loss = F.mse_loss(self.q1(state, action), target_q)
        q2_loss = F.mse_loss(self.q2(state, action), target_q)
        self._safe_loss(q1_loss, "q1_loss")
        self._safe_loss(q2_loss, "q2_loss")
        self.q1_opt.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.cfg.sac.grad_clip_norm)
        self.q1_opt.step()
        self.q2_opt.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.cfg.sac.grad_clip_norm)
        self.q2_opt.step()

        pi, logp = self.actor.sample(state)
        q_pi = torch.min(self.q1(state, pi), self.q2(state, pi))
        actor_loss = (self.alpha * logp - q_pi).mean()
        self._safe_loss(actor_loss, "actor_loss")
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.sac.grad_clip_norm)
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self._safe_loss(alpha_loss, "alpha_loss")
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([self.log_alpha], self.cfg.sac.grad_clip_norm)
        self.alpha_opt.step()

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        tau = self.cfg.sac.tau
        for p, t in zip(source.parameters(), target.parameters()):
            t.data.copy_(tau * p.data + (1.0 - tau) * t.data)

    def get_actor_state(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.actor.state_dict().items()}

    def load_actor_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.actor.load_state_dict(state_dict)
