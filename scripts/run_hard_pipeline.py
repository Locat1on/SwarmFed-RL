"""Hard-mode experiment pipeline.

Compared to the default pipeline:
- 5 robots (vs 3): denser P2P topology, more federation complexity
- 10 obstacles × 0.4m radius (vs 5 × 0.3m): ~3.5× obstacle area coverage
- 300 max episode steps (vs 400): tighter time budget per episode
- 0.15m goal threshold (vs 0.25m): requires more precise navigation
- 2.0m comm radius (vs 3.0m): sparser P2P exchange opportunities
- 200-step exchange interval/cooldown (vs 100): less frequent federation
- 40k timesteps: same training budget, harder environment exposes mode differences

Usage:
    python scripts/run_hard_pipeline.py
    python scripts/run_hard_pipeline.py --timesteps 40000 --seed 42
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def pipeline_dirs(base_dir: Path) -> dict[str, Path]:
    run_dir = base_dir / "pipeline_hard"
    return {
        "root": run_dir,
        "logs": run_dir / "logs",
        "plots": run_dir / "plots",
        "checkpoints": run_dir / "checkpoints" / "warmup",
        "configs": run_dir / "configs",
        "summary": run_dir / "summary.csv",
    }


def append_summary_row(summary_csv: Path, row: dict[str, str]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = summary_csv.exists()
    with summary_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_name", "seed", "mode", "log_csv", "config_snapshot"],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# ---- Hard-mode environment parameters (shared across all runs) ----
HARD_ENV_ARGS = [
    "--robots", "5",
    "--num-obstacles", "10",
    "--obstacle-radius", "0.4",
    "--max-episode-steps", "300",
    "--goal-threshold", "0.15",
]

# ---- Hard-mode P2P parameters ----
HARD_P2P_ARGS = [
    "--comm-radius", "2.0",
    "--exchange-interval-steps", "200",
    "--cooldown-steps", "200",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hard-mode experiment pipeline")
    parser.add_argument("--timesteps", type=int, default=40_000)
    parser.add_argument("--warmup-timesteps", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--base-dir", type=str, default="artifacts")
    args = parser.parse_args()

    seed = args.seed
    dirs = pipeline_dirs(Path(args.base_dir))
    for key in ["root", "logs", "plots", "checkpoints", "configs"]:
        dirs[key].mkdir(parents=True, exist_ok=True)

    run_cmd([sys.executable, "scripts/run_quality_checks.py"])

    # ---- Phase 1: Warmup (local training, no P2P) ----
    warmup_log = dirs["logs"] / f"warmup_seed{seed}.csv"
    run_cmd(
        [
            sys.executable, "scripts/run_warmup.py",
            *HARD_ENV_ARGS,
            "--timesteps", str(args.warmup_timesteps),
            "--seed", str(seed),
            "--checkpoint-dir", str(dirs["checkpoints"]),
            "--log-csv", str(warmup_log),
            "--progress-every", str(args.progress_every),
        ]
    )
    append_summary_row(
        dirs["summary"],
        {"run_name": f"warmup_seed{seed}", "seed": str(seed),
         "mode": "local_warmup", "log_csv": str(warmup_log), "config_snapshot": ""},
    )

    # ---- Phase 2: Three modes (local / centralized / p2p) ----
    for mode in ["local", "centralized", "p2p"]:
        log_csv = dirs["logs"] / f"{mode}_seed{seed}.csv"
        cmd = [
            sys.executable, "scripts/run_experiment.py",
            "--mode", mode,
            *HARD_ENV_ARGS,
            "--timesteps", str(args.timesteps),
            "--seed", str(seed),
            "--load-checkpoint-dir", str(dirs["checkpoints"]),
            "--log-csv", str(log_csv),
            "--progress-every", str(args.progress_every),
        ]
        if mode == "p2p":
            cmd.extend(HARD_P2P_ARGS)
        run_cmd(cmd)
        append_summary_row(
            dirs["summary"],
            {"run_name": f"{mode}_seed{seed}", "seed": str(seed),
             "mode": mode, "log_csv": str(log_csv), "config_snapshot": ""},
        )

    # ---- Phase 3: Generate comparison plots ----
    run_cmd(
        [
            sys.executable, "scripts/plot_metrics.py",
            "--compare",
            f"local={dirs['logs'] / f'local_seed{seed}.csv'}",
            f"centralized={dirs['logs'] / f'centralized_seed{seed}.csv'}",
            f"p2p={dirs['logs'] / f'p2p_seed{seed}.csv'}",
            "--out-dir", str(dirs["plots"] / "comparison"),
        ]
    )

    # Also generate per-mode individual plots
    for mode in ["local", "centralized", "p2p"]:
        log_csv = dirs["logs"] / f"{mode}_seed{seed}.csv"
        run_cmd(
            [
                sys.executable, "scripts/plot_metrics.py",
                "--csv", str(log_csv),
                "--out-dir", str(dirs["plots"] / mode),
            ]
        )

    print("\n" + "=" * 60)
    print("Hard-mode pipeline finished.")
    print(f"Summary CSV: {dirs['summary']}")
    print(f"Logs dir:    {dirs['logs']}")
    print(f"Plots dir:   {dirs['plots']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
