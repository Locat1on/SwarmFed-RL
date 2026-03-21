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
    run_dir = base_dir / "pipeline"
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
                "run_name",
                "seed",
                "mode",
                "defense_strategy",
                "log_csv",
                "tb_dir",
                "config_snapshot",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete experiment pipeline end-to-end")
    parser.add_argument("--robots", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--warmup-timesteps", type=int, default=5000)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--base-dir", type=str, default="artifacts")
    parser.add_argument("--attack-type", choices=["zero", "gaussian"], default="zero")
    parser.add_argument("--calibration-steps", type=int, default=500)
    parser.add_argument("--attack-start-step", type=int, default=500)
    parser.add_argument("--malicious-nodes", type=str, default="1")
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")

    dirs = pipeline_dirs(Path(args.base_dir))
    for key in ["root", "logs", "plots", "checkpoints", "tb", "configs"]:
        dirs[key].mkdir(parents=True, exist_ok=True)

    run_cmd([sys.executable, "scripts/run_quality_checks.py"])

    for seed in seeds:
        warmup_log = dirs["logs"] / f"warmup_seed{seed}.csv"
        run_cmd(
            [
                sys.executable,
                "scripts/run_warmup.py",
                "--robots",
                str(args.robots),
                "--timesteps",
                str(args.warmup_timesteps),
                "--seed",
                str(seed),
                "--checkpoint-dir",
                str(dirs["checkpoints"]),
                "--log-csv",
                str(warmup_log),
                "--progress-every",
                str(args.progress_every),
            ]
        )
        append_summary_row(
            dirs["summary"],
            {
                "run_name": f"warmup_seed{seed}",
                "seed": str(seed),
                "mode": "local_warmup",
                "defense_strategy": "",
                "log_csv": str(warmup_log),
                "tb_dir": "",
                "config_snapshot": "",
            },
        )

        for mode in ["local", "centralized", "p2p"]:
            log_csv = dirs["logs"] / f"{mode}_seed{seed}.csv"
            cmd = [
                sys.executable,
                "scripts/run_experiment.py",
                "--mode",
                mode,
                "--robots",
                str(args.robots),
                "--timesteps",
                str(args.timesteps),
                "--seed",
                str(seed),
                "--load-checkpoint-dir",
                str(dirs["checkpoints"]),
                "--log-csv",
                str(log_csv),
                "--progress-every",
                str(args.progress_every),
            ]
            run_cmd(cmd)
            append_summary_row(
                dirs["summary"],
                {
                    "run_name": f"{mode}_seed{seed}",
                    "seed": str(seed),
                    "mode": mode,
                    "defense_strategy": "",
                    "log_csv": str(log_csv),
                    "tb_dir": "",
                    "config_snapshot": "",
                },
            )

        for defense_strategy in ["cosine", "trimmed_mean", "krum"]:
            name = f"p2p_{defense_strategy}_seed{seed}"
            log_csv = dirs["logs"] / f"{name}.csv"
            tb_dir = dirs["tb"] / name
            cfg = dirs["configs"] / f"{name}.json"
            run_cmd(
                [
                    sys.executable,
                    "scripts/run_experiment.py",
                    "--mode",
                    "p2p",
                    "--robots",
                    str(args.robots),
                    "--timesteps",
                    str(args.timesteps),
                    "--seed",
                    str(seed),
                    "--defense",
                    "--defense-strategy",
                    defense_strategy,
                    "--malicious-nodes",
                    args.malicious_nodes,
                    "--attack-type",
                    args.attack_type,
                    "--calibration-steps",
                    str(args.calibration_steps),
                    "--attack-start-step",
                    str(args.attack_start_step),
                    "--load-checkpoint-dir",
                    str(dirs["checkpoints"]),
                    "--tensorboard-log-dir",
                    str(tb_dir),
                    "--config-snapshot",
                    str(cfg),
                    "--log-csv",
                    str(log_csv),
                    "--progress-every",
                    str(args.progress_every),
                ]
            )
            append_summary_row(
                dirs["summary"],
                {
                    "run_name": name,
                    "seed": str(seed),
                    "mode": "p2p_defense",
                    "defense_strategy": defense_strategy,
                    "log_csv": str(log_csv),
                    "tb_dir": str(tb_dir),
                    "config_snapshot": str(cfg),
                },
            )

            run_cmd(
                [
                    sys.executable,
                    "scripts/plot_metrics.py",
                    "--csv",
                    str(log_csv),
                    "--out-dir",
                    str(dirs["plots"] / name),
                ]
            )

    print("\nPipeline finished.")
    print(f"Summary CSV: {dirs['summary']}")
    print(f"Logs dir: {dirs['logs']}")
    print(f"Plots dir: {dirs['plots']}")
    print(f"TensorBoard root: {dirs['tb']}")


if __name__ == "__main__":
    main()
