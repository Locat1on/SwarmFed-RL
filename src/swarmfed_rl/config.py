from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SeedConfig:
    seed: int = 42
    deterministic_torch: bool = True


@dataclass(frozen=True)
class RewardConfig:
    goal_bonus: float = 200.0
    collision_penalty: float = -100.0
    progress_coeff: float = 10.0
    step_penalty: float = -0.5
    danger_zone_distance: float = 0.5
    proximity_penalty_coeff: float = 0.5
    action_smoothness_coeff: float = 0.1


@dataclass(frozen=True)
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 200_000
    hidden_size: int = 1024
    hidden_layers: int = 4
    residual: bool = True
    actor_encoder: str = "attention"  # attention | cnn | mlp
    actor_use_cnn: bool = True  # deprecated compatibility flag
    attention_dim: int = 128
    attention_heads: int = 4
    attention_layers: int = 3
    warmup_steps: int = 2_000
    update_after: int = 2_000
    update_every: int = 1
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    grad_clip_norm: float = 5.0


@dataclass(frozen=True)
class P2PConfig:
    comm_radius: float = 2.0
    cooldown_steps: int = 50
    beta: float = 0.7
    exchange_interval_steps: int = 20
    weight_std_threshold: float = 0.01


@dataclass(frozen=True)
class ExperimentConfig:
    state_dim: int = 28
    action_dim: int = 2
    action_low: tuple[float, float] = (0.0, -1.5)
    action_high: tuple[float, float] = (0.22, 1.5)
    max_timesteps: int = 20_000
    max_episode_steps: int = 400
    goal_threshold: float = 0.25
    seed: SeedConfig = field(default_factory=SeedConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    p2p: P2PConfig = field(default_factory=P2PConfig)


def build_config(
    *,
    seed: int | None = None,
    max_timesteps: int | None = None,
    comm_radius: float | None = None,
    cooldown_steps: int | None = None,
    exchange_interval_steps: int | None = None,
) -> ExperimentConfig:
    base = ExperimentConfig()
    seed_cfg = SeedConfig(
        seed=seed if seed is not None else base.seed.seed,
        deterministic_torch=base.seed.deterministic_torch,
    )
    p2p_cfg = P2PConfig(
        comm_radius=comm_radius if comm_radius is not None else base.p2p.comm_radius,
        cooldown_steps=cooldown_steps if cooldown_steps is not None else base.p2p.cooldown_steps,
        beta=base.p2p.beta,
        exchange_interval_steps=(
            exchange_interval_steps
            if exchange_interval_steps is not None
            else base.p2p.exchange_interval_steps
        ),
        weight_std_threshold=base.p2p.weight_std_threshold,
    )
    return ExperimentConfig(
        state_dim=base.state_dim,
        action_dim=base.action_dim,
        action_low=base.action_low,
        action_high=base.action_high,
        max_timesteps=max_timesteps if max_timesteps is not None else base.max_timesteps,
        max_episode_steps=base.max_episode_steps,
        goal_threshold=base.goal_threshold,
        seed=seed_cfg,
        reward=base.reward,
        sac=base.sac,
        p2p=p2p_cfg,
    )
