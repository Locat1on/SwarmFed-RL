from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from .experiment import run_experiment


@dataclass
class RunnerStats:
    timesteps: int = 0
    episodes: int = 0
    total_exchanges: int = 0


def run_phase1(
    num_robots: int,
    max_timesteps: int,
    seed: int,
    *,
    warmup_only: bool = False,
    checkpoint_dir: str | None = None,
    log_csv_path: str | None = None,
    progress_every: int | None = None,
) -> RunnerStats:
    mode = "local" if warmup_only else "p2p"
    summary = run_experiment(
        mode=mode,
        num_robots=num_robots,
        max_timesteps=max_timesteps,
        seed=seed,
        log_csv_path=log_csv_path,
        checkpoint_dir=checkpoint_dir,
        progress_every=progress_every,
    )
    stats = RunnerStats(
        timesteps=summary.timesteps,
        episodes=summary.episodes,
        total_exchanges=summary.exchanges,
    )
    print(
        "Phase-1 finished | "
        f"timesteps={stats.timesteps} episodes={stats.episodes} "
        f"exchanges={stats.total_exchanges} actor_bytes={summary.communication_bytes} "
        f"mode={mode}"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Run phase-1 core loop (SAC + P2P Actor)")
    parser.add_argument("--robots", type=int, default=3, help="number of robot agents")
    parser.add_argument("--timesteps", type=int, default=5_000, help="total timesteps")
    parser.add_argument("--seed", type=int, default=42, help="global random seed")
    parser.add_argument("--warmup-only", action="store_true", help="disable communication for local warm-up")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="save actor checkpoints to directory")
    parser.add_argument("--log-csv", type=str, default=None, help="write step metrics CSV to path")
    parser.add_argument("--progress-every", type=int, default=500, help="print progress every N timesteps")
    args = parser.parse_args()
    run_phase1(
        num_robots=args.robots,
        max_timesteps=args.timesteps,
        seed=args.seed,
        warmup_only=args.warmup_only,
        checkpoint_dir=args.checkpoint_dir,
        log_csv_path=args.log_csv,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
