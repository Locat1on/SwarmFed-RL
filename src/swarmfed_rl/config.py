from __future__ import annotations

import math
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
    buffer_size: int = 100_000
    hidden_size: int = 256
    hidden_layers: int = 2
    residual: bool = False
    actor_encoder: str = "mlp"
    actor_use_cnn: bool = False
    attention_dim: int = 64
    attention_heads: int = 2
    attention_layers: int = 1
    warmup_steps: int = 500
    update_after: int = 256
    update_every: int = 1         # Train every step (no cadence gate)
    gradient_updates: int = 1     # 1 gradient update per train call
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    amp_dtype: str = "bf16"
    enable_tf32: bool = True
    enable_torch_compile: bool = False
    compile_mode: str = "reduce-overhead"
    use_gpu_replay: bool = True
    actor_update_interval: int = 1  # Update actor every critic update
    shared_replay: bool = True      # Share replay buffer across agents


@dataclass(frozen=True)
class P2PConfig:
    comm_radius: float = 3.0
    cooldown_steps: int = 100
    beta: float = 0.5               # Balanced blend: 50% local, 50% incoming
    exchange_interval_steps: int = 100
    weight_std_threshold: float = 0.0
    use_grid_index: bool = True
    grid_cell_size: float = 3.5
    use_fp16_comm: bool = False      # Disabled for stability
    layer_diff_threshold: float = 0.0  # Exchange all layers
    async_exchange: bool = False     # Synchronous to avoid race conditions
    dynamic_beta: bool = False       # Fixed beta for predictability
    beta_min: float = 0.3
    beta_max: float = 0.7
    beta_schedule: str = "constant"


@dataclass(frozen=True)
class ExperimentConfig:
    state_dim: int = 28
    frame_stack: int = 1
    action_dim: int = 2
    action_low: tuple[float, float] = (0.0, -1.5)
    action_high: tuple[float, float] = (0.22, 1.5)
    max_timesteps: int = 20_000
    max_epochs: int = 10
    steps_per_epoch: int = 2000
    max_episode_steps: int = 400
    goal_threshold: float = 0.25
    num_obstacles: int = 5
    obstacle_radius: float = 0.3
    seed: SeedConfig = field(default_factory=SeedConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    p2p: P2PConfig = field(default_factory=P2PConfig)


def build_config(
    *,
    seed: int | None = None,
    max_timesteps: int | None = None,
    max_epochs: int | None = None,
    steps_per_epoch: int | None = None,
    num_obstacles: int | None = None,
    obstacle_radius: float | None = None,
    max_episode_steps: int | None = None,
    goal_threshold: float | None = None,
    comm_radius: float | None = None,
    cooldown_steps: int | None = None,
    exchange_interval_steps: int | None = None,
    weight_std_threshold: float | None = None,
    frame_stack: int | None = None,
    use_gpu_replay: bool | None = None,
    use_grid_index: bool | None = None,
    grid_cell_size: float | None = None,
    actor_update_interval: int | None = None,
    use_fp16_comm: bool | None = None,
    layer_diff_threshold: float | None = None,
    async_exchange: bool | None = None,
    beta_schedule: str | None = None,
    sac_gradient_updates: int | None = None,
    sac_batch_size: int | None = None,
    shared_replay: bool | None = None,
) -> ExperimentConfig:
    base = ExperimentConfig()
    frame_stack_val = max(1, int(frame_stack if frame_stack is not None else base.frame_stack))
    state_dim_val = 24 * frame_stack_val + 4
    seed_cfg = SeedConfig(
        seed=seed if seed is not None else base.seed.seed,
        deterministic_torch=base.seed.deterministic_torch,
    )
    sac_cfg = SACConfig(
        gamma=base.sac.gamma,
        tau=base.sac.tau,
        actor_lr=base.sac.actor_lr,
        critic_lr=base.sac.critic_lr,
        alpha_lr=base.sac.alpha_lr,
        batch_size=sac_batch_size if sac_batch_size is not None else base.sac.batch_size,
        buffer_size=base.sac.buffer_size,
        hidden_size=base.sac.hidden_size,
        hidden_layers=base.sac.hidden_layers,
        residual=base.sac.residual,
        actor_encoder=base.sac.actor_encoder,
        actor_use_cnn=base.sac.actor_use_cnn,
        attention_dim=base.sac.attention_dim,
        attention_heads=base.sac.attention_heads,
        attention_layers=base.sac.attention_layers,
        warmup_steps=base.sac.warmup_steps,
        update_after=base.sac.update_after,
        update_every=base.sac.update_every,
        gradient_updates=sac_gradient_updates if sac_gradient_updates is not None else base.sac.gradient_updates,
        log_std_min=base.sac.log_std_min,
        log_std_max=base.sac.log_std_max,
        grad_clip_norm=base.sac.grad_clip_norm,
        use_amp=base.sac.use_amp,
        amp_dtype=base.sac.amp_dtype,
        enable_tf32=base.sac.enable_tf32,
        enable_torch_compile=base.sac.enable_torch_compile,
        compile_mode=base.sac.compile_mode,
        use_gpu_replay=base.sac.use_gpu_replay if use_gpu_replay is None else bool(use_gpu_replay),
        actor_update_interval=(
            actor_update_interval if actor_update_interval is not None else base.sac.actor_update_interval
        ),
        shared_replay=base.sac.shared_replay if shared_replay is None else bool(shared_replay),
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
        weight_std_threshold=(
            weight_std_threshold
            if weight_std_threshold is not None
            else base.p2p.weight_std_threshold
        ),
        use_grid_index=base.p2p.use_grid_index if use_grid_index is None else bool(use_grid_index),
        grid_cell_size=(
            float(base.p2p.grid_cell_size if grid_cell_size is None else grid_cell_size)
        ),
        use_fp16_comm=base.p2p.use_fp16_comm if use_fp16_comm is None else bool(use_fp16_comm),
        layer_diff_threshold=(
            layer_diff_threshold if layer_diff_threshold is not None else base.p2p.layer_diff_threshold
        ),
        async_exchange=base.p2p.async_exchange if async_exchange is None else bool(async_exchange),
        beta_schedule=beta_schedule if beta_schedule is not None else base.p2p.beta_schedule,
    )

    final_steps_per_epoch = steps_per_epoch if steps_per_epoch is not None else base.steps_per_epoch

    if max_timesteps is not None:
        calc_timesteps = max_timesteps
        final_max_epochs = max(1, math.ceil(calc_timesteps / final_steps_per_epoch))
    elif max_epochs is not None:
        final_max_epochs = max_epochs
        calc_timesteps = final_max_epochs * final_steps_per_epoch
    else:
        final_max_epochs = base.max_epochs
        calc_timesteps = base.max_timesteps

    return ExperimentConfig(
        state_dim=state_dim_val,
        frame_stack=frame_stack_val,
        action_dim=base.action_dim,
        action_low=base.action_low,
        action_high=base.action_high,
        max_timesteps=calc_timesteps,
        max_epochs=final_max_epochs,
        steps_per_epoch=final_steps_per_epoch,
        max_episode_steps=max_episode_steps if max_episode_steps is not None else base.max_episode_steps,
        goal_threshold=goal_threshold if goal_threshold is not None else base.goal_threshold,
        num_obstacles=num_obstacles if num_obstacles is not None else base.num_obstacles,
        obstacle_radius=obstacle_radius if obstacle_radius is not None else base.obstacle_radius,
        seed=seed_cfg,
        reward=base.reward,
        sac=sac_cfg,
        p2p=p2p_cfg,
    )
