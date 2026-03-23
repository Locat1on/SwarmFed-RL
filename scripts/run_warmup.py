import argparse

from swarmfed_rl.train_phase1 import run_phase1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local warm-up training (no P2P)")
    parser.add_argument("--robots", type=int, default=3, help="number of robot agents")
    parser.add_argument("--timesteps", type=int, default=5_000, help="total warm-up timesteps")
    parser.add_argument("--seed", type=int, default=42, help="global random seed")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="artifacts\\checkpoints\\warmup",
        help="directory to save actor checkpoints",
    )
    parser.add_argument(
        "--log-csv",
        type=str,
        default="artifacts\\logs\\warmup.csv",
        help="output CSV path",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="print progress every N timesteps",
    )
    parser.add_argument("--num-obstacles", type=int, default=None)
    parser.add_argument("--obstacle-radius", type=float, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--goal-threshold", type=float, default=None)
    args = parser.parse_args()
    run_phase1(
        num_robots=args.robots,
        max_timesteps=args.timesteps,
        seed=args.seed,
        warmup_only=True,
        checkpoint_dir=args.checkpoint_dir,
        log_csv_path=args.log_csv,
        progress_every=args.progress_every,
        num_obstacles=args.num_obstacles,
        obstacle_radius=args.obstacle_radius,
        max_episode_steps=args.max_episode_steps,
        goal_threshold=args.goal_threshold,
    )


if __name__ == "__main__":
    main()
