from __future__ import annotations

import csv
import json
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
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
from .sac import SACAgent
from .utils import configure_torch_runtime, set_global_seed


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


def run_experiment(
    *,
    mode: str,
    num_robots: int,
    max_timesteps: int,
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
) -> ExperimentSummary:
    if mode not in {"local", "centralized", "p2p"}:
        raise ValueError(f"Unsupported mode: {mode}")
    malicious_nodes = malicious_nodes or set()

    cfg = build_config(
        seed=seed,
        max_timesteps=max_timesteps,
        comm_radius=comm_radius,
        cooldown_steps=cooldown_steps,
        exchange_interval_steps=exchange_interval_steps,
        weight_std_threshold=weight_std_threshold,
        frame_stack=frame_stack,
        use_gpu_replay=use_gpu_replay,
        use_grid_index=use_grid_index,
        grid_cell_size=grid_cell_size,
        actor_update_interval=actor_update_interval,
        use_fp16_comm=use_fp16_comm,
        layer_diff_threshold=layer_diff_threshold,
        async_exchange=async_exchange,
    )
    set_global_seed(cfg.seed)
    configure_torch_runtime(enable_tf32=cfg.sac.enable_tf32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config_snapshot_path is not None:
        _save_config_snapshot(
            path=config_snapshot_path,
            payload={
                "mode": mode,
                "num_robots": num_robots,
                "seed": seed,
                "max_timesteps": max_timesteps,
                "defense_enabled": defense_enabled,
                "defense_strategy": defense_strategy,
                "defense_trim_ratio": defense_trim_ratio,
                "defense_krum_malicious": defense_krum_malicious,
                "attack_type": attack_type,
                "calibration_steps": calibration_steps,
                "attack_start_step": attack_start_step,
                "shared_agent": shared_agent,
                "env_step_workers": env_step_workers,
                "weight_std_threshold": weight_std_threshold,
                "comm_radius": comm_radius,
                "cooldown_steps": cooldown_steps,
                "exchange_interval_steps": exchange_interval_steps,
                "frame_stack": frame_stack,
                "use_gpu_replay": use_gpu_replay,
                "use_grid_index": use_grid_index,
                "grid_cell_size": grid_cell_size,
                "actor_update_interval": actor_update_interval,
                "use_fp16_comm": use_fp16_comm,
                "layer_diff_threshold": layer_diff_threshold,
                "async_exchange": async_exchange,
            },
        )

    envs = {rid: SimulatedROS2Env(cfg, rid) for rid in range(num_robots)}
    if shared_agent:
        shared = SACAgent(cfg, device)
        # fast training path
        agents = {rid: shared for rid in range(num_robots)}
        # communication view for P2P/Centralized exchange
        shadow_agents = {rid: SACAgent(cfg, device) for rid in range(num_robots)}
    else:
        agents = {rid: SACAgent(cfg, device) for rid in range(num_robots)}
        shadow_agents = agents
    states = {rid: envs[rid].reset() for rid in range(num_robots)}

    if load_checkpoint_dir:
        if shared_agent:
            _load_shared_actor_checkpoint(next(iter(agents.values())), load_checkpoint_dir)
            shared_state = next(iter(agents.values())).get_actor_state()
            for sa in shadow_agents.values():
                sa.load_actor_state(shared_state)
        else:
            _load_actor_checkpoints(agents, load_checkpoint_dir)

    p2p = P2PAggregator(cfg.p2p)
    centralized = CentralizedFedAvg(interval_steps=cfg.p2p.exchange_interval_steps, beta=cfg.p2p.beta)

    episodes = 0
    successes = 0
    collisions = 0
    exchanges = 0
    cumulative_reward = 0.0
    current_ep_rewards = {rid: 0.0 for rid in range(num_robots)}
    completed_ep_returns: deque[float] = deque(maxlen=1000)
    csv_file, csv_writer = _open_csv_writer(log_csv_path)
    tb_writer = None
    if tensorboard_log_dir:
        if SummaryWriter is None:
            raise RuntimeError("TensorBoard support is unavailable; install tensorboard package.")
        tb_writer = SummaryWriter(log_dir=tensorboard_log_dir)
    step_executor = None
    if env_step_workers > 1:
        step_executor = ThreadPoolExecutor(max_workers=env_step_workers)
    
    exchange_executor: ThreadPoolExecutor | None = None
    exchange_future: Future[int] | None = None
    if cfg.p2p.async_exchange and mode in {"p2p", "centralized"}:
        exchange_executor = ThreadPoolExecutor(max_workers=1)

    # Performance tracking
    step_start_time = time.time()
    last_log_time = step_start_time
    steps_since_last_log = 0
    total_env_time = 0.0
    total_train_time = 0.0
    total_exchange_time = 0.0

    try:
        for step in range(cfg.max_timesteps):
            iter_start = time.time()
            positions: dict[int, np.ndarray] = {}
            step_reward = 0.0
            actions_by_rid: dict[int, np.ndarray] = {}
            # Ensure previous async exchange is done before touching agent networks
            if not shared_agent and exchange_future is not None and not exchange_future.done():
                exchanges += exchange_future.result()
                exchange_future = None
            if shared_agent:
                shared = next(iter(agents.values()))
                rid_list = list(range(num_robots))
                state_batch = np.stack([states[rid] for rid in rid_list], axis=0).astype(np.float32)
                action_batch = shared.select_actions(state_batch, deterministic=False)
                actions_by_rid = {rid: action_batch[i] for i, rid in enumerate(rid_list)}
            else:
                actions_by_rid = {
                    rid: agents[rid].select_action(states[rid], deterministic=False) for rid in range(num_robots)
                }

            # Environment stepping
            env_step_start = time.time()
            if step_executor is not None:
                step_futures = {
                    rid: step_executor.submit(
                        envs[rid].step,
                        actions_by_rid[rid],
                    )
                    for rid in range(num_robots)
                }
                step_outputs = {rid: fut.result() for rid, fut in step_futures.items()}
            else:
                step_outputs = {}

            for rid in range(num_robots):
                agent = agents[rid]
                env = envs[rid]
                state = states[rid]
                action = actions_by_rid[rid]
                if step_outputs:
                    next_state, reward, done, info = step_outputs[rid]
                else:
                    next_state, reward, done, info = env.step(action)
                normalized_action = _normalize_action(action, cfg.action_low, cfg.action_high)
                agent.buffer.push(state, normalized_action.astype(np.float32), reward, next_state, done)
                states[rid] = env.reset() if done else next_state
                positions[rid] = env.get_position()

                step_reward += reward
                current_ep_rewards[rid] += reward
                if done:
                    episodes += 1
                    completed_ep_returns.append(current_ep_rewards[rid])
                    current_ep_rewards[rid] = 0.0
                if bool(info["success"]):
                    successes += 1
                if bool(info["collision"]):
                    collisions += 1

            env_step_end = time.time()
            total_env_time += (env_step_end - env_step_start)

            # Training and exchange
            train_start = time.time()
            if not shared_agent:
                # Staggered training: each step only train one agent (round-robin).
                # Call train_step() num_robots times so _train_call_count grows
                # at the same rate as non-staggered mode, keeping update_every /
                # gradient_updates / update_after semantics unchanged.
                # Total gradient updates per step stays the same, but all on one
                # network → better GPU cache utilization.
                rid_to_train = step % num_robots
                for _ in range(num_robots):
                    agents[rid_to_train].train_step()
            if shared_agent:
                shared_ref = next(iter(agents.values()))
                shared_ref.train_step()
                # Only sync shadow agents on exchange steps to avoid per-step cloning overhead
                if mode in {"p2p", "centralized"} and step % cfg.p2p.exchange_interval_steps == 0:
                    shared_state = shared_ref.get_actor_state(cpu_clone=False)
                    for sa in shadow_agents.values():
                        sa.load_actor_state(shared_state)
                
            train_end = time.time()
            total_train_time += (train_end - train_start)
            
            # P2P exchange
            exchange_start = time.time()
            if shared_agent:
                if mode == "p2p":
                    if cfg.p2p.async_exchange:
                        # Wait for previous exchange to complete
                        if exchange_future is not None and not exchange_future.done():
                            exchanges += exchange_future.result()
                        # Submit new exchange asynchronously
                        exchange_future = exchange_executor.submit(
                            p2p.maybe_exchange,
                            step_idx=step,
                            agents=shadow_agents,
                            positions=positions,
                            malicious_nodes=malicious_nodes,
                            attack_type=attack_type,
                            defense_enabled=defense_enabled,
                            defense_strategy=defense_strategy,
                            defense_trim_ratio=defense_trim_ratio,
                            defense_krum_malicious=defense_krum_malicious,
                            calibration_steps=calibration_steps,
                            attack_start_step=attack_start_step,
                        )
                    else:
                        exchanges += p2p.maybe_exchange(
                            step_idx=step,
                            agents=shadow_agents,
                            positions=positions,
                            malicious_nodes=malicious_nodes,
                            attack_type=attack_type,
                            defense_enabled=defense_enabled,
                            defense_strategy=defense_strategy,
                            defense_trim_ratio=defense_trim_ratio,
                            defense_krum_malicious=defense_krum_malicious,
                            calibration_steps=calibration_steps,
                            attack_start_step=attack_start_step,
                        )
                    comm_bytes = p2p.bytes_transferred
                    shadow_mean = _average_actor_state(shadow_agents, cpu_clone=False)
                    shared_ref.load_actor_state(shadow_mean)
                elif mode == "centralized":
                    if cfg.p2p.async_exchange:
                        if exchange_future is not None and not exchange_future.done():
                            exchanges += exchange_future.result()
                        exchange_future = exchange_executor.submit(
                            centralized.maybe_aggregate,
                            step_idx=step,
                            agents=shadow_agents,
                        )
                    else:
                        exchanges += centralized.maybe_aggregate(step_idx=step, agents=shadow_agents)
                    comm_bytes = centralized.bytes_transferred
                    shadow_mean = _average_actor_state(shadow_agents, cpu_clone=False)
                    shared_ref.load_actor_state(shadow_mean)
                else:
                    comm_bytes = 0
            elif mode == "p2p":
                if cfg.p2p.async_exchange:
                    if exchange_future is not None and not exchange_future.done():
                        exchanges += exchange_future.result()
                    exchange_future = exchange_executor.submit(
                        p2p.maybe_exchange,
                        step_idx=step,
                        agents=agents,
                        positions=positions,
                        malicious_nodes=malicious_nodes,
                        attack_type=attack_type,
                        defense_enabled=defense_enabled,
                        defense_strategy=defense_strategy,
                        defense_trim_ratio=defense_trim_ratio,
                        defense_krum_malicious=defense_krum_malicious,
                        calibration_steps=calibration_steps,
                        attack_start_step=attack_start_step,
                    )
                else:
                    exchanges += p2p.maybe_exchange(
                        step_idx=step,
                        agents=agents,
                        positions=positions,
                        malicious_nodes=malicious_nodes,
                        attack_type=attack_type,
                        defense_enabled=defense_enabled,
                        defense_strategy=defense_strategy,
                        defense_trim_ratio=defense_trim_ratio,
                        defense_krum_malicious=defense_krum_malicious,
                        calibration_steps=calibration_steps,
                        attack_start_step=attack_start_step,
                    )
                comm_bytes = p2p.bytes_transferred
            elif mode == "centralized":
                if cfg.p2p.async_exchange:
                    if exchange_future is not None and not exchange_future.done():
                        exchanges += exchange_future.result()
                    exchange_future = exchange_executor.submit(
                        centralized.maybe_aggregate,
                        step_idx=step,
                        agents=agents,
                    )
                else:
                    exchanges += centralized.maybe_aggregate(step_idx=step, agents=agents)
                comm_bytes = centralized.bytes_transferred
            else:
                comm_bytes = 0
            
            exchange_end = time.time()
            total_exchange_time += (exchange_end - exchange_start)

            cumulative_reward += step_reward
            episode_return_mean = mean(completed_ep_returns) if completed_ep_returns else 0.0
            latest_episode_return = completed_ep_returns[-1] if completed_ep_returns else None
            
            # Performance metrics
            iter_end = time.time()
            iter_time = iter_end - iter_start
            steps_since_last_log += 1
            
            # Write metrics every 10 steps to reduce I/O overhead
            if step % 10 == 0 or step == cfg.max_timesteps - 1:
                _write_step_row(
                    csv_writer=csv_writer,
                    file_handle=csv_file,
                    step=step + 1,
                    mode=mode,
                    defense_enabled=defense_enabled,
                    step_reward=step_reward,
                    cumulative_reward=cumulative_reward,
                    episodes=episodes,
                    successes=successes,
                    collisions=collisions,
                    exchanges=exchanges,
                    communication_bytes=comm_bytes,
                    defense_accepted=p2p.defense_stats.accepted_updates,
                    defense_rejected=p2p.defense_stats.rejected_updates,
                    defense_rejected_malicious=p2p.defense_stats.rejected_malicious_updates,
                    defense_accepted_malicious=p2p.defense_stats.accepted_malicious_updates,
                    episode_return_mean=episode_return_mean,
                )
                _write_tb(
                    writer=tb_writer,
                    step=step + 1,
                    step_reward=step_reward,
                    cumulative_reward=cumulative_reward,
                    episodes=episodes,
                    successes=successes,
                    collisions=collisions,
                    exchanges=exchanges,
                    communication_bytes=comm_bytes,
                    defense_accepted=p2p.defense_stats.accepted_updates,
                    defense_rejected=p2p.defense_stats.rejected_updates,
                    episode_return_mean=episode_return_mean,
                    latest_episode_return=latest_episode_return,
                )
            if progress_every is not None and progress_every > 0:
                current_step = step + 1
                if current_step == 1 or current_step % progress_every == 0 or current_step == cfg.max_timesteps:
                    now = time.time()
                    elapsed_total = now - step_start_time
                    elapsed_since_last = now - last_log_time
                    steps_per_sec = steps_since_last_log / elapsed_since_last if elapsed_since_last > 0 else 0
                    
                    # Breakdown percentages
                    env_pct = (total_env_time / elapsed_total * 100) if elapsed_total > 0 else 0
                    train_pct = (total_train_time / elapsed_total * 100) if elapsed_total > 0 else 0
                    exchange_pct = (total_exchange_time / elapsed_total * 100) if elapsed_total > 0 else 0
                    
                    # Buffer stats
                    sample_agent = next(iter(agents.values()))
                    buffer_size = sample_agent.buffer.size
                    buffer_capacity = sample_agent.buffer.capacity if hasattr(sample_agent.buffer, 'capacity') else sample_agent.cfg.sac.buffer_size
                    buffer_usage_pct = (buffer_size / buffer_capacity * 100) if buffer_capacity > 0 else 0
                    
                    train_label = "train" if shared_agent else f"train(r{rid_to_train})"
                    print(
                        f"[{mode}] step={current_step}/{cfg.max_timesteps} | "
                        f"eps={episodes} rew={cumulative_reward:.2f} "
                        f"succ={successes} coll={collisions} | "
                        f"exch={exchanges} bytes={comm_bytes/1e6:.1f}MB | "
                        f"speed={steps_per_sec:.2f}step/s ({elapsed_since_last:.1f}s) | "
                        f"buf={buffer_usage_pct:.0f}% | "
                        f"time: env={env_pct:.0f}% {train_label}={train_pct:.0f}% p2p={exchange_pct:.0f}%",
                        flush=True,
                    )
                    last_log_time = now
                    steps_since_last_log = 0
        # Finalize any pending async exchanges
        if exchange_future is not None and not exchange_future.done():
            exchanges += exchange_future.result()
    finally:
        if csv_file is not None:
            csv_file.close()
        if tb_writer is not None:
            tb_writer.close()
        if step_executor is not None:
            step_executor.shutdown(wait=True)
        if exchange_executor is not None:
            exchange_executor.shutdown(wait=True)

    if checkpoint_dir:
        if shared_agent:
            _save_shared_actor_checkpoint(next(iter(agents.values())), checkpoint_dir)
        else:
            _save_actor_checkpoints(agents, checkpoint_dir)

    summary = ExperimentSummary(
        mode=mode,
        timesteps=cfg.max_timesteps,
        episodes=episodes,
        successes=successes,
        collisions=collisions,
        exchanges=exchanges,
        communication_bytes=(
            p2p.bytes_transferred
            if mode == "p2p"
            else centralized.bytes_transferred if mode == "centralized" else 0
        ),
        defense_accepted=p2p.defense_stats.accepted_updates,
        defense_rejected=p2p.defense_stats.rejected_updates,
        defense_rejected_malicious=p2p.defense_stats.rejected_malicious_updates,
        defense_accepted_malicious=p2p.defense_stats.accepted_malicious_updates,
    )
    return summary


def _normalize_action(
    action: np.ndarray,
    action_low: tuple[float, float],
    action_high: tuple[float, float],
) -> np.ndarray:
    low = np.asarray(action_low, dtype=np.float32)
    high = np.asarray(action_high, dtype=np.float32)
    return 2.0 * (action - low) / (high - low) - 1.0


def _open_csv_writer(path: str | None) -> tuple[object | None, csv.DictWriter | None]:
    if path is None:
        return None, None
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Using line buffering (buffering=1) to reduce data loss risk
    f = out.open("w", encoding="utf-8", newline="", buffering=1)
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "step",
            "mode",
            "defense_enabled",
            "step_reward",
            "cumulative_reward",
            "episodes",
            "successes",
            "collisions",
            "exchanges",
            "communication_bytes",
            "defense_accepted",
            "defense_rejected",
            "defense_rejected_malicious",
            "defense_accepted_malicious",
            "episode_return_mean",
        ],
    )
    writer.writeheader()
    return f, writer


def _save_config_snapshot(path: str, payload: dict[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_tb(
    *,
    writer: SummaryWriter | None,
    step: int,
    step_reward: float,
    cumulative_reward: float,
    episodes: int,
    successes: int,
    collisions: int,
    exchanges: int,
    communication_bytes: int,
    defense_accepted: int,
    defense_rejected: int,
    episode_return_mean: float,
    latest_episode_return: float | None,
) -> None:
    if writer is None:
        return
    writer.add_scalar("reward/step", step_reward, step)
    writer.add_scalar("reward/cumulative", cumulative_reward, step)
    writer.add_scalar("episode/count", episodes, step)
    writer.add_scalar("metrics/successes", successes, step)
    writer.add_scalar("metrics/collisions", collisions, step)
    writer.add_scalar("comm/exchanges", exchanges, step)
    writer.add_scalar("comm/bytes", communication_bytes, step)
    writer.add_scalar("defense/accepted", defense_accepted, step)
    writer.add_scalar("defense/rejected", defense_rejected, step)
    writer.add_scalar("episode/return_mean", episode_return_mean, step)
    if latest_episode_return is not None:
        writer.add_scalar("episode/return_latest", latest_episode_return, step)


def _write_step_row(
    *,
    csv_writer: csv.DictWriter | None,
    file_handle: object | None,
    step: int,
    mode: str,
    defense_enabled: bool,
    step_reward: float,
    cumulative_reward: float,
    episodes: int,
    successes: int,
    collisions: int,
    exchanges: int,
    communication_bytes: int,
    defense_accepted: int,
    defense_rejected: int,
    defense_rejected_malicious: int,
    defense_accepted_malicious: int,
    episode_return_mean: float,
) -> None:
    if csv_writer is None:
        return
    csv_writer.writerow(
        {
            "step": step,
            "mode": mode,
            "defense_enabled": int(defense_enabled),
            "step_reward": step_reward,
            "cumulative_reward": cumulative_reward,
            "episodes": episodes,
            "successes": successes,
            "collisions": collisions,
            "exchanges": exchanges,
            "communication_bytes": communication_bytes,
            "defense_accepted": defense_accepted,
            "defense_rejected": defense_rejected,
            "defense_rejected_malicious": defense_rejected_malicious,
            "defense_accepted_malicious": defense_accepted_malicious,
            "episode_return_mean": episode_return_mean,
        }
    )
    if step % 10 == 0 and file_handle is not None:
        file_handle.flush()


def _save_actor_checkpoints(agents: dict[int, SACAgent], out_dir: str) -> None:
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    for rid, agent in agents.items():
        torch.save(agent.get_actor_state(), base / f"actor_robot_{rid}.pt")


def _save_shared_actor_checkpoint(agent: SACAgent, out_dir: str) -> None:
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    torch.save(agent.get_actor_state(), base / "actor_robot_0.pt")


def _load_actor_checkpoints(agents: dict[int, SACAgent], in_dir: str) -> None:
    base = Path(in_dir)
    for rid, agent in agents.items():
        ckpt = base / f"actor_robot_{rid}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        state_dict = torch.load(ckpt, map_location="cpu")
        agent.load_actor_state(state_dict)


def _load_shared_actor_checkpoint(agent: SACAgent, in_dir: str) -> None:
    base = Path(in_dir)
    ckpt = base / "actor_robot_0.pt"
    if not ckpt.exists():
        files = sorted(base.glob("actor_robot_*.pt"))
        if not files:
            raise FileNotFoundError(f"No actor checkpoints found under: {base}")
        ckpt = files[0]
    state_dict = torch.load(ckpt, map_location="cpu")
    agent.load_actor_state(state_dict)


def _average_actor_state(
    agents: dict[int, SACAgent],
    *,
    cpu_clone: bool = True,
) -> dict[str, torch.Tensor]:
    ids = sorted(agents.keys())
    states = [agents[rid].get_actor_state(cpu_clone=cpu_clone) for rid in ids]
    out: dict[str, torch.Tensor] = {}
    for k in states[0]:
        out[k] = torch.stack([s[k] for s in states], dim=0).mean(dim=0)
    return out
