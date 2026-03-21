from __future__ import annotations

import csv
import json
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
from .utils import set_global_seed


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
) -> ExperimentSummary:
    if mode not in {"local", "centralized", "p2p"}:
        raise ValueError(f"Unsupported mode: {mode}")
    malicious_nodes = malicious_nodes or set()

    cfg = build_config(seed=seed, max_timesteps=max_timesteps)
    set_global_seed(cfg.seed)
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
            },
        )

    envs = {rid: SimulatedROS2Env(cfg, rid) for rid in range(num_robots)}
    agents = {rid: SACAgent(cfg, device) for rid in range(num_robots)}
    states = {rid: envs[rid].reset() for rid in range(num_robots)}

    if load_checkpoint_dir:
        _load_actor_checkpoints(agents, load_checkpoint_dir)

    p2p = P2PAggregator(cfg.p2p)
    centralized = CentralizedFedAvg(interval_steps=cfg.p2p.exchange_interval_steps)

    episodes = 0
    successes = 0
    collisions = 0
    exchanges = 0
    cumulative_reward = 0.0
    current_ep_rewards = {rid: 0.0 for rid in range(num_robots)}
    completed_ep_returns: list[float] = []
    csv_file, csv_writer = _open_csv_writer(log_csv_path)
    tb_writer = None
    if tensorboard_log_dir:
        if SummaryWriter is None:
            raise RuntimeError("TensorBoard support is unavailable; install tensorboard package.")
        tb_writer = SummaryWriter(log_dir=tensorboard_log_dir)

    try:
        for step in range(cfg.max_timesteps):
            positions: dict[int, np.ndarray] = {}
            step_reward = 0.0

            for rid in range(num_robots):
                agent = agents[rid]
                env = envs[rid]
                state = states[rid]
                action = agent.select_action(state, deterministic=False)
                next_state, reward, done, info = env.step(action)
                normalized_action = _normalize_action(action, cfg.action_low, cfg.action_high)
                agent.buffer.push(state, normalized_action.astype(np.float32), reward, next_state, done)
                agent.train_step()
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

            if mode == "p2p":
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
                exchanges += centralized.maybe_aggregate(step_idx=step, agents=agents)
                comm_bytes = centralized.bytes_transferred
            else:
                comm_bytes = 0

            cumulative_reward += step_reward
            episode_return_mean = mean(completed_ep_returns) if completed_ep_returns else 0.0
            latest_episode_return = completed_ep_returns[-1] if completed_ep_returns else None
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
                    print(
                        f"[{mode}] step={current_step}/{cfg.max_timesteps} "
                        f"episodes={episodes} reward={cumulative_reward:.2f} "
                        f"succ={successes} coll={collisions} "
                        f"exchanges={exchanges} bytes={comm_bytes}",
                        flush=True,
                    )
    finally:
        if csv_file is not None:
            csv_file.close()
        if tb_writer is not None:
            tb_writer.close()

    if checkpoint_dir:
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


def _load_actor_checkpoints(agents: dict[int, SACAgent], in_dir: str) -> None:
    base = Path(in_dir)
    for rid, agent in agents.items():
        ckpt = base / f"actor_robot_{rid}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        state_dict = torch.load(ckpt, map_location="cpu")
        agent.load_actor_state(state_dict)
