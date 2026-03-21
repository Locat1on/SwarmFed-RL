import argparse
import sys

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swarmfed_rl.experiment import run_experiment


class _Tee:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def parse_malicious_nodes(raw: str) -> set[int]:
    if not raw.strip():
        return set()
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local/centralized/p2p experiments")
    parser.add_argument("--mode", choices=["local", "centralized", "p2p"], default="p2p")
    parser.add_argument("--robots", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-csv", type=str, default=None, help="Path to log CSV (default: auto-generated in artifacts/logs/)")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--load-checkpoint-dir", type=str, default=None)
    parser.add_argument("--defense", action="store_true", help="enable anomaly defense for p2p")
    parser.add_argument("--defense-strategy", choices=["cosine", "trimmed_mean", "krum"], default="cosine")
    parser.add_argument("--defense-trim-ratio", type=float, default=0.2)
    parser.add_argument("--defense-krum-malicious", type=int, default=1)
    parser.add_argument("--malicious-nodes", type=str, default="", help="comma-separated robot IDs")
    parser.add_argument("--attack-type", choices=["zero", "gaussian"], default="zero")
    parser.add_argument("--calibration-steps", type=int, default=500)
    parser.add_argument("--attack-start-step", type=int, default=500)
    parser.add_argument("--tensorboard-log-dir", type=str, default=None)
    parser.add_argument("--enable-tensorboard", action="store_true", help="Enable TensorBoard logging (disabled by default for speed)")
    parser.add_argument("--config-snapshot", type=str, default=None)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--shared-agent", action="store_true", help="Use one shared policy for all robots (faster throughput)")
    parser.add_argument("--env-step-workers", type=int, default=0, help="Parallel env.step workers (applies to both strict and shared modes)")
    parser.add_argument("--weight-std-threshold", type=float, default=0.01, help="Std threshold for counting/exchanging significant actor weights")
    parser.add_argument("--comm-radius", type=float, default=None, help="P2P communication radius override")
    parser.add_argument("--cooldown-steps", type=int, default=None, help="P2P cooldown steps override")
    parser.add_argument("--exchange-interval-steps", type=int, default=None, help="P2P exchange interval override")
    parser.add_argument("--frame-stack", type=int, default=1, help="Number of lidar frames to stack into state")
    parser.add_argument("--gpu-replay-buffer", action="store_true", help="Use GPU resident replay buffer when CUDA is available")
    parser.add_argument("--disable-grid-index", action="store_true", help="Disable P2P grid-based neighbor candidate indexing")
    parser.add_argument("--grid-cell-size", type=float, default=2.0, help="Grid cell size for P2P neighbor indexing")
    parser.add_argument("--run-name", type=str, default=None, help="Name of the run (default: auto-generated)")
    parser.add_argument("--artifact-root", type=str, default="artifacts", help="Root directory for artifacts")
    args = parser.parse_args()

    if args.defense and args.mode != "p2p":
        print(f"Warning: --defense requires --mode p2p. Forcing p2p mode.")
        args.mode = "p2p"
    if args.mode == "p2p" and args.shared_agent:
        print("Warning: Strict P2P experiment requires independent agents. Disabling --shared-agent for p2p.")
        args.shared_agent = False
    
    # --- Artifact Path Logic ---
    root = Path(args.artifact_root)
    
    # 1. Determine Run Name
    if args.run_name:
        run_name = args.run_name
    else:
        # If user provided a log_csv, try to derive run_name from it
        if args.log_csv:
            run_name = Path(args.log_csv).stem
        else:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{args.mode}_{args.robots}_{timestamp}"
            
    # 2. Construct Paths (allow overrides)
    # Structure: artifacts/{logs,plots,checkpoints,tb,configs}/{mode}/{run_name}/...
    
    # Log CSV
    if args.log_csv:
        log_csv_path = Path(args.log_csv)
    else:
        log_csv_path = root / "logs" / args.mode / f"{run_name}.csv"
        
    # Checkpoints
    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    else:
        checkpoint_dir = str(root / "checkpoints" / args.mode / run_name)
        
    # TensorBoard
    if args.enable_tensorboard and args.tensorboard_log_dir:
        tb_dir = args.tensorboard_log_dir
    elif args.enable_tensorboard:
        tb_dir = str(root / "tb" / args.mode / run_name)
    else:
        tb_dir = None
        
    # Config Snapshot
    if args.config_snapshot:
        config_path = args.config_snapshot
    else:
        config_path = str(root / "configs" / args.mode / f"{run_name}.json")
        
    # Plots (derived from log_csv location or standard location)
    # We prefer the standard location: artifacts/plots/{mode}/{run_name}
    plot_dir = root / "plots" / args.mode / run_name
    terminal_log_path = root / "logs" / args.mode / f"{run_name}.log"

    # Ensure parent directories exist where appropriate
    log_csv_path.parent.mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if tb_dir is not None:
        Path(tb_dir).mkdir(parents=True, exist_ok=True)
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    terminal_log_path.parent.mkdir(parents=True, exist_ok=True)

    with terminal_log_path.open("w", encoding="utf-8", buffering=1) as terminal_log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = _Tee(original_stdout, terminal_log_file)
        sys.stderr = _Tee(original_stderr, terminal_log_file)
        try:
            print(f"--- Experiment: {run_name} ---")
            print(f"Logs(CSV):   {log_csv_path}")
            print(f"Logs(Term):  {terminal_log_path}")
            print(f"Plots:       {plot_dir}")
            print(f"Checkpoints: {checkpoint_dir}")
            print(f"TensorBoard: {tb_dir if tb_dir is not None else 'disabled'}")
            print(f"Config:      {config_path}")

            # If using Krum defense and user didn't specify krum count, try to infer or validate
            if args.defense and args.defense_strategy == "krum":
                # In small swarms, Krum needs f < (N-2)/2.
                # With 3 robots, N=3, (3-2)/2 = 0.5 => f=0 is max possible (no defense).
                # We warn the user if configuration is mathematically impossible for standard Krum.
                max_f = (args.robots - 3) // 2
                if args.defense_krum_malicious > max_f and args.robots < 30:
                    print(
                        f"Warning: Krum with f={args.defense_krum_malicious} on N={args.robots} robots is theoretically unstable (N < 2f+3)."
                    )
                    print(f"  Standard Krum requires N >= {2*args.defense_krum_malicious + 3}.")
                    print("  Current implementation will fallback to 'best available' but performance may degrade to Local training.")

            summary = run_experiment(
                mode=args.mode,
                num_robots=args.robots,
                max_timesteps=args.timesteps,
                seed=args.seed,
                log_csv_path=str(log_csv_path),
                checkpoint_dir=checkpoint_dir,
                load_checkpoint_dir=args.load_checkpoint_dir,
                defense_enabled=args.defense,
                defense_strategy=args.defense_strategy,
                defense_trim_ratio=args.defense_trim_ratio,
                defense_krum_malicious=args.defense_krum_malicious,
                malicious_nodes=parse_malicious_nodes(args.malicious_nodes),
                attack_type=args.attack_type,
                calibration_steps=args.calibration_steps,
                attack_start_step=args.attack_start_step,
                tensorboard_log_dir=tb_dir,
                config_snapshot_path=config_path,
                progress_every=args.progress_every,
                shared_agent=args.shared_agent,
                env_step_workers=args.env_step_workers,
                weight_std_threshold=args.weight_std_threshold,
                comm_radius=args.comm_radius,
                cooldown_steps=args.cooldown_steps,
                exchange_interval_steps=args.exchange_interval_steps,
                frame_stack=args.frame_stack,
                use_gpu_replay=args.gpu_replay_buffer,
                use_grid_index=(not args.disable_grid_index),
                grid_cell_size=args.grid_cell_size,
            )
            print(
                "Experiment finished | "
                f"mode={summary.mode} timesteps={summary.timesteps} "
                f"episodes={summary.episodes} successes={summary.successes} collisions={summary.collisions} "
                f"exchanges={summary.exchanges} bytes={summary.communication_bytes} "
                f"defense_rejected={summary.defense_rejected} rejected_malicious={summary.defense_rejected_malicious}"
            )

            try:
                from swarmfed_rl.plotting import generate_plots

                print(f"Generating plots in: {plot_dir}")
                generate_plots(str(log_csv_path), str(plot_dir))
            except ImportError:
                print("Warning: Could not import plotting module. Skipping auto-plot.")
            except Exception as e:
                print(f"Warning: Plotting failed with error: {e}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
