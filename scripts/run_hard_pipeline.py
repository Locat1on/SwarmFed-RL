"""Hard-mode experiment pipeline.

Compared to the default pipeline:
- 5 robots (vs 3): denser P2P topology, more federation complexity
- 10 obstacles × 0.4m radius (vs 5 × 0.3m): ~3.5× obstacle area coverage
- 300 max episode steps (vs 400): tighter time budget per episode
- 0.15m goal threshold (vs 0.25m): requires more precise navigation
- 2.0m comm radius (vs 3.0m): sparser P2P exchange opportunities
- 200-step exchange interval/cooldown (vs 100): less frequent federation
- 100k timesteps (vs 5k): longer training to handle increased difficulty

Usage:
    python scripts/run_hard_pipeline.py
    python scripts/run_hard_pipeline.py --timesteps 200000 --seeds 42,43,44
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
        "tb": run_dir / "tb",
        "configs": run_dir / "configs",
        "summary": run_dir / "summary.csv",
    }


def append_summary_row(summary_csv: Path, row: dict[str, str]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = summary_csv.exists()
    with summary_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name", "seed", "mode", "defense_strategy",
                "log_csv", "tb_dir", "config_snapshot",
            ],
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
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--warmup-timesteps", type=int, default=20_000)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--base-dir", type=str, default="artifacts")
    parser.add_argument("--attack-type", choices=["zero", "gaussian"], default="gaussian")
    parser.add_argument("--calibration-steps", type=int, default=1000)
    parser.add_argument("--attack-start-step", type=int, default=2000)
    parser.add_argument("--malicious-nodes", type=str, default="1,3")
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")

    dirs = pipeline_dirs(Path(args.base_dir))
    for key in ["root", "logs", "plots", "checkpoints", "tb", "configs"]:
        dirs[key].mkdir(parents=True, exist_ok=True)

    run_cmd([sys.executable, "scripts/run_quality_checks.py"])

    for seed in seeds:
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
             "mode": "local_warmup", "defense_strategy": "",
             "log_csv": str(warmup_log), "tb_dir": "", "config_snapshot": ""},
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
                 "mode": mode, "defense_strategy": "",
                 "log_csv": str(log_csv), "tb_dir": "", "config_snapshot": ""},
            )

        # ---- Phase 3: P2P + defense strategies under attack ----
        for defense_strategy in ["cosine", "trimmed_mean", "krum"]:
            name = f"p2p_{defense_strategy}_seed{seed}"
            log_csv = dirs["logs"] / f"{name}.csv"
            tb_dir = dirs["tb"] / name
            cfg = dirs["configs"] / f"{name}.json"
            run_cmd(
                [
                    sys.executable, "scripts/run_experiment.py",
                    "--mode", "p2p",
                    *HARD_ENV_ARGS,
                    *HARD_P2P_ARGS,
                    "--timesteps", str(args.timesteps),
                    "--seed", str(seed),
                    "--defense",
                    "--defense-strategy", defense_strategy,
                    "--malicious-nodes", args.malicious_nodes,
                    "--attack-type", args.attack_type,
                    "--calibration-steps", str(args.calibration_steps),
                    "--attack-start-step", str(args.attack_start_step),
                    "--load-checkpoint-dir", str(dirs["checkpoints"]),
                    "--tensorboard-log-dir", str(tb_dir),
                    "--config-snapshot", str(cfg),
                    "--log-csv", str(log_csv),
                    "--progress-every", str(args.progress_every),
                ]
            )
            append_summary_row(
                dirs["summary"],
                {"run_name": name, "seed": str(seed),
                 "mode": "p2p_defense", "defense_strategy": defense_strategy,
                 "log_csv": str(log_csv), "tb_dir": str(tb_dir),
                 "config_snapshot": str(cfg)},
            )

            run_cmd(
                [
                    sys.executable, "scripts/plot_metrics.py",
                    "--csv", str(log_csv),
                    "--out-dir", str(dirs["plots"] / name),
                ]
            )

        # ---- Phase 4: Generate comparison plots for this seed ----
        compare_args = [
            sys.executable, "scripts/plot_metrics.py",
            "--compare",
            f"local={dirs['logs'] / f'local_seed{seed}.csv'}",
            f"centralized={dirs['logs'] / f'centralized_seed{seed}.csv'}",
            f"p2p={dirs['logs'] / f'p2p_seed{seed}.csv'}",
            "--out-dir", str(dirs["plots"] / f"comparison_seed{seed}"),
        ]
        run_cmd(compare_args)

        # Defense comparison
        defense_compare_args = [
            sys.executable, "scripts/plot_metrics.py",
            "--compare",
            f"p2p_no_defense={dirs['logs'] / f'p2p_seed{seed}.csv'}",
            f"cosine={dirs['logs'] / f'p2p_cosine_seed{seed}.csv'}",
            f"trimmed_mean={dirs['logs'] / f'p2p_trimmed_mean_seed{seed}.csv'}",
            f"krum={dirs['logs'] / f'p2p_krum_seed{seed}.csv'}",
            "--out-dir", str(dirs["plots"] / f"defense_comparison_seed{seed}"),
        ]
        run_cmd(defense_compare_args)

    print("\n" + "=" * 60)
    print("Hard-mode pipeline finished.")
    print(f"Summary CSV: {dirs['summary']}")
    print(f"Logs dir:    {dirs['logs']}")
    print(f"Plots dir:   {dirs['plots']}")
    print(f"TensorBoard: {dirs['tb']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
