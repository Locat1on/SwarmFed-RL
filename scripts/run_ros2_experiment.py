import argparse
import datetime
import sys

from pathlib import Path
from swarmfed_rl.ros2_training import ROS2RunnerOptions, run_ros2_experiment


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


def parse_robot_ids(raw: str) -> list[int]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("robot_ids cannot be empty")
    return [int(x) for x in values]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ROS2/Gazebo connected RL experiment")
    parser.add_argument("--robot-ids", type=str, default="0,1,2", help="comma-separated robot IDs")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--ready-timeout-sec", type=float, default=15.0)
    parser.add_argument("--collision-scan-threshold", type=float, default=0.14)
    parser.add_argument("--topic-prefix", type=str, default="/tb3_")
    parser.add_argument("--weights-topic", type=str, default="/swarm/actor_weights")
    parser.add_argument("--max-chunk-payload", type=int, default=4096)
    parser.add_argument("--artifact-root", type=str, default="artifacts", help="Root directory for all artifacts")
    parser.add_argument("--run-name", type=str, default=None, help="Unique name for this run (default: auto-generated)")
    parser.add_argument("--no-gazebo-reset", action="store_true")
    parser.add_argument("--no-reset-on-done", action="store_true")
    parser.add_argument("--retransmit-count", type=int, default=1)
    args = parser.parse_args()

    if args.run_name:
        run_name = args.run_name
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"ros2_{len(parse_robot_ids(args.robot_ids))}_{timestamp}"

    # Construct artifact paths
    base_dir = Path(args.artifact_root)
    log_dir = base_dir / "logs" / "ros2"
    terminal_log_path = log_dir / f"{run_name}.log"
    ckpt_dir = base_dir / "checkpoints" / "ros2" / run_name
    tb_dir = base_dir / "tb" / "ros2" / run_name

    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    options = ROS2RunnerOptions(
        robot_ids=parse_robot_ids(args.robot_ids),
        max_timesteps=args.timesteps,
        seed=args.seed,
        control_hz=args.control_hz,
        wait_ready_timeout_sec=args.ready_timeout_sec,
        collision_scan_threshold=args.collision_scan_threshold,
        reset_on_done=not args.no_reset_on_done,
        use_gazebo_reset=not args.no_gazebo_reset,
        topic_prefix=args.topic_prefix,
        shared_weights_topic=args.weights_topic,
        max_chunk_payload=args.max_chunk_payload,
        retransmit_count=args.retransmit_count,
    )
    with terminal_log_path.open("w", encoding="utf-8", buffering=1) as terminal_log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = _Tee(original_stdout, terminal_log_file)
        sys.stderr = _Tee(original_stderr, terminal_log_file)
        try:
            print(f"--- ROS2 Experiment: {run_name} ---")
            print(f"Artifacts root: {base_dir}")
            print(f"Logs(Term):     {terminal_log_path}")
            print(f"Checkpoints:    {ckpt_dir}")
            print(f"TensorBoard:    {tb_dir}")
            summary = run_ros2_experiment(options)
            print(
                "ROS2 experiment finished | "
                f"timesteps={summary.timesteps} episodes={summary.episodes} "
                f"successes={summary.successes} collisions={summary.collisions} "
                f"exchanges={summary.exchanges} bytes={summary.communication_bytes}"
            )
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
