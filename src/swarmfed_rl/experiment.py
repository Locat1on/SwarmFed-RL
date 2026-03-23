"""Main training loop for local / centralised / P2P experiments.

Key fixes for convergence:
- Train ALL agents every step (not round-robin 1/N)
- Shared replay buffer so all agents learn from collective experience
- Synchronous exchange to avoid race conditions
- Simplified timing and progress output
"""
from __future__ import annotations

import csv
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]

from .config import build_config
from .env import SimulatedROS2Env
from .p2p import CentralizedFedAvg, P2PAggregator
from .sac import SACAgent, GPUReplayBuffer, ReplayBuffer
from .utils import configure_torch_runtime, set_global_seed


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class ExperimentSummary:
    mode: str
    timesteps: int
    episodes: int
    successes: int
    collisions: int
    exchanges: int
    communication_bytes: int
    defense_accepted: int = 0
    defense_rejected: int = 0
    defense_rejected_malicious: int = 0
    defense_accepted_malicious: int = 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_experiment(
    *,
    mode: str,
    num_robots: int,
    max_timesteps: int | None = None,
    max_epochs: int | None = None,
    steps_per_epoch: int | None = None,
    num_obstacles: int | None = None,
    obstacle_radius: float | None = None,
    max_episode_steps: int | None = None,
    goal_threshold: float | None = None,
    seed: int,
    log_csv_path: str | None = None,
    checkpoint_dir: str | None = None,
    load_checkpoint_dir: str | None = None,
    defense_enabled: bool = False,
    malicious_nodes: set[int] | None = None,
    attack_type: str = "zero",
    calibration_steps: int = 0,
    attack_start_step: int = 0,
    defense_strategy: str = "cosine",
    defense_trim_ratio: float = 0.2,
    defense_krum_malicious: int = 1,
    tensorboard_log_dir: str | None = None,
    config_snapshot_path: str | None = None,
    progress_every: int | None = None,
    shared_agent: bool = False,
    env_step_workers: int = 0,
    weight_std_threshold: float | None = None,
    comm_radius: float | None = None,
    cooldown_steps: int | None = None,
    exchange_interval_steps: int | None = None,
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
) -> ExperimentSummary:
    if mode not in {"local", "centralized", "p2p"}:
        raise ValueError(f"Unsupported mode: {mode}")
    malicious_nodes = malicious_nodes or set()

    # -- Config -----------------------------------------------------------
    cfg = build_config(
        seed=seed, max_timesteps=max_timesteps, max_epochs=max_epochs,
        steps_per_epoch=steps_per_epoch, num_obstacles=num_obstacles,
        obstacle_radius=obstacle_radius, max_episode_steps=max_episode_steps,
        goal_threshold=goal_threshold, comm_radius=comm_radius,
        cooldown_steps=cooldown_steps, exchange_interval_steps=exchange_interval_steps,
        weight_std_threshold=weight_std_threshold, frame_stack=frame_stack,
        use_gpu_replay=use_gpu_replay, use_grid_index=use_grid_index,
        grid_cell_size=grid_cell_size, actor_update_interval=actor_update_interval,
        use_fp16_comm=use_fp16_comm, layer_diff_threshold=layer_diff_threshold,
        async_exchange=async_exchange, beta_schedule=beta_schedule,
        sac_gradient_updates=sac_gradient_updates, sac_batch_size=sac_batch_size,
        shared_replay=shared_replay,
    )
    set_global_seed(cfg.seed)
    configure_torch_runtime(enable_tf32=cfg.sac.enable_tf32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config_snapshot_path is not None:
        _save_config_snapshot(config_snapshot_path, {
            "mode": mode, "num_robots": num_robots, "seed": seed,
            "max_timesteps": cfg.max_timesteps, "max_epochs": cfg.max_epochs,
            "steps_per_epoch": cfg.steps_per_epoch, "num_obstacles": cfg.num_obstacles,
            "obstacle_radius": cfg.obstacle_radius,
            "defense_enabled": defense_enabled, "defense_strategy": defense_strategy,
            "shared_agent": shared_agent, "shared_replay": cfg.sac.shared_replay,
        })

    # -- Shared replay buffer --------------------------------------------
    shared_buffer: ReplayBuffer | GPUReplayBuffer | None = None
    if cfg.sac.shared_replay and not shared_agent:
        use_gpu = bool(cfg.sac.use_gpu_replay and device.type == "cuda")
        if use_gpu:
            shared_buffer = GPUReplayBuffer(cfg.state_dim, cfg.action_dim, cfg.sac.buffer_size, device)
        else:
            shared_buffer = ReplayBuffer(cfg.state_dim, cfg.action_dim, cfg.sac.buffer_size)

    # -- Agents & Envs ---------------------------------------------------
    envs = {rid: SimulatedROS2Env(cfg, rid) for rid in range(num_robots)}
    if shared_agent:
        shared = SACAgent(cfg, device)
        agents = {rid: shared for rid in range(num_robots)}
    else:
        agents = {rid: SACAgent(cfg, device, buffer=shared_buffer) for rid in range(num_robots)}
    states = {rid: envs[rid].reset() for rid in range(num_robots)}

    if load_checkpoint_dir:
        if shared_agent:
            _load_shared_actor_checkpoint(next(iter(agents.values())), load_checkpoint_dir)
        else:
            _load_actor_checkpoints(agents, load_checkpoint_dir)

    p2p = P2PAggregator(cfg.p2p)
    centralized = CentralizedFedAvg(interval_steps=cfg.p2p.exchange_interval_steps, beta=cfg.p2p.beta)

    # -- Counters --------------------------------------------------------
    episodes = successes = collisions = exchanges = 0
    cumulative_reward = 0.0
    current_ep_rewards = {rid: 0.0 for rid in range(num_robots)}
    completed_ep_returns: deque[float] = deque(maxlen=1000)
    episode_return_mean = 0.0

    # -- Logging ---------------------------------------------------------
    csv_file, csv_writer = _open_csv_writer(log_csv_path)
    log_epoch_path = log_csv_path.replace(".csv", "_epoch.csv") if log_csv_path else None
    epoch_file, epoch_writer = _open_epoch_csv_writer(log_epoch_path)
    tb_writer = None
    if tensorboard_log_dir:
        if SummaryWriter is None:
            raise RuntimeError("TensorBoard unavailable; install tensorboard.")
        tb_writer = SummaryWriter(log_dir=tensorboard_log_dir)

    # -- Timing ----------------------------------------------------------
    wall_start = time.time()
    last_log_time = wall_start
    steps_since_log = 0
    total_env_t = total_train_t = total_exch_t = 0.0

    # Pre-compute action normalisation constants
    act_low = np.asarray(cfg.action_low, dtype=np.float32)
    act_high = np.asarray(cfg.action_high, dtype=np.float32)
    act_range = act_high - act_low

    # Exchange arguments
    exch_kwargs: dict = dict(
        malicious_nodes=malicious_nodes, attack_type=attack_type,
        defense_enabled=defense_enabled, defense_strategy=defense_strategy,
        defense_trim_ratio=defense_trim_ratio, defense_krum_malicious=defense_krum_malicious,
        calibration_steps=calibration_steps, attack_start_step=attack_start_step,
    )

    try:
        for epoch in range(cfg.max_epochs):
            epoch_reward = 0.0
            epoch_successes = epoch_collisions = epoch_exchanges = 0

            for step_in_epoch in range(cfg.steps_per_epoch):
                step = epoch * cfg.steps_per_epoch + step_in_epoch
                positions: dict[int, np.ndarray] = {}
                step_reward = 0.0

                # ---- 1. Action selection ----
                t0 = time.time()
                if shared_agent:
                    sh = next(iter(agents.values()))
                    batch = np.stack([states[rid] for rid in range(num_robots)], axis=0).astype(np.float32)
                    acts = sh.select_actions(batch, deterministic=False)
                    actions_by_rid = {rid: acts[i] for i, rid in enumerate(range(num_robots))}
                else:
                    actions_by_rid = {rid: agents[rid].select_action(states[rid]) for rid in range(num_robots)}
                t_action = time.time() - t0

                # ---- 2. Environment step ----
                t0 = time.time()
                for rid in range(num_robots):
                    ns, rew, done, info = envs[rid].step(actions_by_rid[rid])
                    norm_act = 2.0 * (actions_by_rid[rid] - act_low) / act_range - 1.0
                    # Push to agent's buffer (shared or individual)
                    agents[rid].buffer.push(states[rid], norm_act.astype(np.float32), rew, ns, done)
                    states[rid] = envs[rid].reset() if done else ns
                    positions[rid] = envs[rid].get_position()
                    step_reward += rew
                    current_ep_rewards[rid] += rew
                    if done:
                        episodes += 1
                        completed_ep_returns.append(current_ep_rewards[rid])
                        current_ep_rewards[rid] = 0.0
                    if bool(info.get("success")):
                        successes += 1; epoch_successes += 1
                    if bool(info.get("collision")):
                        collisions += 1; epoch_collisions += 1
                _d = time.time() - t0
                total_env_t += _d

                # ---- 3. Training: ALL agents every step ----
                t0 = time.time()
                log_m = (step % 100 == 0)
                if shared_agent:
                    sh = next(iter(agents.values()))
                    sh.train_step(num_updates=cfg.sac.gradient_updates, pull_metrics=log_m)
                else:
                    # Train every agent every step
                    for rid in range(num_robots):
                        agents[rid].train_step(
                            num_updates=cfg.sac.gradient_updates,
                            pull_metrics=log_m and rid == 0,
                        )
                _d = time.time() - t0
                total_train_t += _d + t_action

                # ---- 4. P2P / Centralized exchange (synchronous) ----
                t0 = time.time()
                progress = step / cfg.max_timesteps
                comm_bytes = 0
                if step > 0 and step % cfg.p2p.exchange_interval_steps == 0:
                    if mode == "p2p":
                        res, merged = p2p.maybe_exchange(
                            step_idx=step, agents=agents, positions=positions,
                            progress=progress, **exch_kwargs,
                        )
                        exchanges += res; epoch_exchanges += res
                        _apply_merged_states(merged, agents, mode)
                        comm_bytes = p2p.bytes_transferred
                    elif mode == "centralized":
                        res, merged = centralized.maybe_aggregate(step_idx=step, agents=agents)
                        exchanges += res; epoch_exchanges += res
                        _apply_merged_states(merged, agents, mode)
                        comm_bytes = centralized.bytes_transferred
                else:
                    comm_bytes = p2p.bytes_transferred if mode == "p2p" else (centralized.bytes_transferred if mode == "centralized" else 0)
                _d = time.time() - t0
                total_exch_t += _d

                cumulative_reward += step_reward; epoch_reward += step_reward
                episode_return_mean = mean(completed_ep_returns) if completed_ep_returns else 0.0
                latest_ep_ret = completed_ep_returns[-1] if completed_ep_returns else None
                steps_since_log += 1

                # CSV / TB logging
                if step % 10 == 0 or step == cfg.max_timesteps - 1:
                    _write_step_row(csv_writer, csv_file, step + 1, mode, defense_enabled,
                                    step_reward, cumulative_reward, episodes, successes, collisions,
                                    exchanges, comm_bytes,
                                    p2p.defense_stats.accepted_updates, p2p.defense_stats.rejected_updates,
                                    p2p.defense_stats.rejected_malicious_updates, p2p.defense_stats.accepted_malicious_updates,
                                    episode_return_mean)
                    _write_tb(tb_writer, step + 1, step_reward, cumulative_reward, episodes,
                              successes, collisions, exchanges, comm_bytes,
                              p2p.defense_stats.accepted_updates, p2p.defense_stats.rejected_updates,
                              episode_return_mean, latest_ep_ret)

                # Progress output
                if progress_every and progress_every > 0:
                    cs = step + 1
                    if cs == 1 or cs % progress_every == 0 or cs == cfg.max_timesteps:
                        now = time.time()
                        dt = now - last_log_time
                        sps = steps_since_log / dt if dt > 0 else 0
                        et = now - wall_start
                        ep = total_env_t / et * 100 if et > 0 else 0
                        tp = total_train_t / et * 100 if et > 0 else 0
                        xp = total_exch_t / et * 100 if et > 0 else 0
                        print(
                            f"[{mode}] epoch={epoch+1}/{cfg.max_epochs} step={cs}/{cfg.max_timesteps} | "
                            f"eps={episodes} rew={cumulative_reward:.1f} succ={successes} coll={collisions} | "
                            f"ep_ret={episode_return_mean:.1f} | "
                            f"exch={exchanges} bytes={comm_bytes/1e6:.1f}MB | "
                            f"speed={sps:.1f}step/s | "
                            f"time: env={ep:.0f}% train={tp:.0f}% p2p={xp:.0f}%",
                            flush=True,
                        )
                        last_log_time = now; steps_since_log = 0

            # End of epoch
            if epoch_writer:
                _write_epoch_row(epoch_writer, epoch_file, epoch + 1, epoch_reward,
                                 epoch_successes, epoch_collisions, epoch_exchanges,
                                 episodes, episode_return_mean)
            if tb_writer:
                tb_writer.add_scalar("epoch/reward", epoch_reward, epoch + 1)
                tb_writer.add_scalar("epoch/success_rate", epoch_successes / cfg.steps_per_epoch, epoch + 1)
                tb_writer.add_scalar("epoch/collision_rate", epoch_collisions / cfg.steps_per_epoch, epoch + 1)

    finally:
        for f in (csv_file, epoch_file):
            if f is not None:
                f.close()
        if tb_writer:
            tb_writer.close()

    if checkpoint_dir:
        if shared_agent:
            _save_shared_actor_checkpoint(next(iter(agents.values())), checkpoint_dir)
        else:
            _save_actor_checkpoints(agents, checkpoint_dir)

    return ExperimentSummary(
        mode=mode, timesteps=cfg.max_timesteps, episodes=episodes,
        successes=successes, collisions=collisions, exchanges=exchanges,
        communication_bytes=p2p.bytes_transferred if mode == "p2p" else (centralized.bytes_transferred if mode == "centralized" else 0),
        defense_accepted=p2p.defense_stats.accepted_updates,
        defense_rejected=p2p.defense_stats.rejected_updates,
        defense_rejected_malicious=p2p.defense_stats.rejected_malicious_updates,
        defense_accepted_malicious=p2p.defense_stats.accepted_malicious_updates,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_merged_states(
    merged: dict[int, dict[str, torch.Tensor]],
    agents: dict[int, SACAgent],
    mode: str,
) -> None:
    if not merged:
        return
    if mode in {"p2p", "centralized"}:
        for rid, state in merged.items():
            agents[rid].load_actor_state(state)


def _average_actor_state(agents: dict[int, SACAgent], *, cpu_clone: bool = True) -> dict[str, torch.Tensor]:
    ids = sorted(agents.keys())
    states = [agents[rid].get_actor_state(cpu_clone=cpu_clone) for rid in ids]
    return {k: torch.stack([s[k] for s in states], dim=0).mean(dim=0) for k in states[0]}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _open_csv_writer(path: str | None):
    if path is None:
        return None, None
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    f = p.open("w", encoding="utf-8", newline="", buffering=1)
    w = csv.DictWriter(f, fieldnames=[
        "step", "mode", "defense_enabled", "step_reward", "cumulative_reward",
        "episodes", "successes", "collisions", "exchanges", "communication_bytes",
        "defense_accepted", "defense_rejected", "defense_rejected_malicious",
        "defense_accepted_malicious", "episode_return_mean",
    ])
    w.writeheader()
    return f, w


def _open_epoch_csv_writer(path: str | None):
    if path is None:
        return None, None
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    f = p.open("w", encoding="utf-8", newline="", buffering=1)
    w = csv.DictWriter(f, fieldnames=[
        "epoch", "epoch_reward", "epoch_successes", "epoch_collisions",
        "epoch_exchanges", "total_episodes", "episode_return_mean",
    ])
    w.writeheader()
    return f, w


def _save_config_snapshot(path: str, payload: dict) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_tb(writer, step, step_reward, cum_reward, episodes, successes, collisions,
              exchanges, comm_bytes, def_acc, def_rej, ep_ret_mean, latest_ep_ret):
    if writer is None:
        return
    writer.add_scalar("reward/step", step_reward, step)
    writer.add_scalar("reward/cumulative", cum_reward, step)
    writer.add_scalar("episode/count", episodes, step)
    writer.add_scalar("metrics/successes", successes, step)
    writer.add_scalar("metrics/collisions", collisions, step)
    writer.add_scalar("comm/exchanges", exchanges, step)
    writer.add_scalar("comm/bytes", comm_bytes, step)
    writer.add_scalar("defense/accepted", def_acc, step)
    writer.add_scalar("defense/rejected", def_rej, step)
    writer.add_scalar("episode/return_mean", ep_ret_mean, step)
    if latest_ep_ret is not None:
        writer.add_scalar("episode/return_latest", latest_ep_ret, step)


def _write_step_row(csv_writer, fh, step, mode, defense_enabled, step_reward,
                    cum_reward, episodes, successes, collisions, exchanges,
                    comm_bytes, def_acc, def_rej, def_rej_mal, def_acc_mal, ep_ret_mean):
    if csv_writer is None:
        return
    csv_writer.writerow({
        "step": step, "mode": mode, "defense_enabled": int(defense_enabled),
        "step_reward": step_reward, "cumulative_reward": cum_reward,
        "episodes": episodes, "successes": successes, "collisions": collisions,
        "exchanges": exchanges, "communication_bytes": comm_bytes,
        "defense_accepted": def_acc, "defense_rejected": def_rej,
        "defense_rejected_malicious": def_rej_mal, "defense_accepted_malicious": def_acc_mal,
        "episode_return_mean": ep_ret_mean,
    })
    if step % 10 == 0 and fh is not None:
        fh.flush()


def _write_epoch_row(ew, fh, epoch, reward, succ, coll, exch, eps, ep_ret_mean):
    if ew is None:
        return
    ew.writerow({
        "epoch": epoch, "epoch_reward": reward, "epoch_successes": succ,
        "epoch_collisions": coll, "epoch_exchanges": exch,
        "total_episodes": eps, "episode_return_mean": ep_ret_mean,
    })
    if fh is not None:
        fh.flush()


def _save_actor_checkpoints(agents: dict[int, SACAgent], out_dir: str) -> None:
    base = Path(out_dir); base.mkdir(parents=True, exist_ok=True)
    for rid, agent in agents.items():
        torch.save(agent.get_actor_state(), base / f"actor_robot_{rid}.pt")


def _save_shared_actor_checkpoint(agent: SACAgent, out_dir: str) -> None:
    base = Path(out_dir); base.mkdir(parents=True, exist_ok=True)
    torch.save(agent.get_actor_state(), base / "actor_robot_0.pt")


def _load_actor_checkpoints(agents: dict[int, SACAgent], in_dir: str) -> None:
    base = Path(in_dir)
    for rid, agent in agents.items():
        ckpt = base / f"actor_robot_{rid}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        agent.load_actor_state(torch.load(ckpt, map_location="cpu"))


def _load_shared_actor_checkpoint(agent: SACAgent, in_dir: str) -> None:
    base = Path(in_dir)
    ckpt = base / "actor_robot_0.pt"
    if not ckpt.exists():
        files = sorted(base.glob("actor_robot_*.pt"))
        if not files:
            raise FileNotFoundError(f"No actor checkpoints found under: {base}")
        ckpt = files[0]
    agent.load_actor_state(torch.load(ckpt, map_location="cpu"))
