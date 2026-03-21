from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .config import ExperimentConfig, build_config
from .ros2_runtime import (
    GazeboResetManager,
    NeighborCooldownGate,
    ROS2RLNode,
    ReceivedWeights,
    ros2_available,
    sample_safe_xy,
)
from .sac import SACAgent
from .utils import set_global_seed

try:  # pragma: no cover
    import rclpy
except ImportError:  # pragma: no cover
    rclpy = None


@dataclass
class ROS2RunSummary:
    timesteps: int
    episodes: int
    successes: int
    collisions: int
    exchanges: int
    communication_bytes: int


@dataclass
class ROS2RunnerOptions:
    robot_ids: list[int]
    max_timesteps: int = 5_000
    seed: int = 42
    control_hz: float = 10.0
    wait_ready_timeout_sec: float = 15.0
    collision_scan_threshold: float = 0.14
    reset_on_done: bool = True
    use_gazebo_reset: bool = True
    topic_prefix: str = "/tb3_"
    shared_weights_topic: str = "/swarm/actor_weights"
    max_chunk_payload: int = 4096
    retransmit_count: int = 1


@dataclass
class _RobotContext:
    robot_id: int
    node: ROS2RLNode
    agent: SACAgent
    gate: NeighborCooldownGate
    goal_xy: np.ndarray
    prev_distance: float
    episode_steps: int
    reset_manager: GazeboResetManager | None
    last_action: np.ndarray
    received_bytes: int = 0


def run_ros2_experiment(options: ROS2RunnerOptions) -> ROS2RunSummary:  # pragma: no cover
    if not ros2_available() or rclpy is None:
        raise RuntimeError("ROS2 runtime unavailable: install and source ROS2 before running this script")
    if len(options.robot_ids) == 0:
        raise ValueError("At least one robot_id is required")

    cfg = build_config(seed=options.seed, max_timesteps=options.max_timesteps)
    set_global_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(options.seed)

    rclpy.init()
    contexts: dict[int, _RobotContext] = {}
    try:
        for rid in options.robot_ids:
            prefix = f"{options.topic_prefix}{rid}"
            node = ROS2RLNode(
                robot_id=rid,
                node_name=f"rl_control_node_{rid}",
                scan_topic=f"{prefix}/scan",
                odom_topic=f"{prefix}/odom",
                cmd_topic=f"{prefix}/cmd_vel",
                weights_topic=options.shared_weights_topic,
                qos_depth=10,
                max_chunk_payload=options.max_chunk_payload,
                retransmit_count=options.retransmit_count,
            )
            ok = node.wait_until_ready(timeout_sec=options.wait_ready_timeout_sec)
            if not ok:
                raise RuntimeError(f"Robot {rid} not ready: no /scan data within timeout")

            goal_xy = np.asarray(sample_safe_xy(rng=rng), dtype=np.float32)
            state = node.build_state(goal_xy=goal_xy)
            if state.shape[0] != cfg.state_dim:
                raise RuntimeError(f"State dim mismatch for robot {rid}: {state.shape[0]} != {cfg.state_dim}")

            reset_manager = None
            if options.use_gazebo_reset:
                reset_manager = GazeboResetManager(node=node, model_name=f"tb3_{rid}")

            contexts[rid] = _RobotContext(
                robot_id=rid,
                node=node,
                agent=SACAgent(cfg, device),
                gate=NeighborCooldownGate(cfg.p2p.comm_radius, cfg.p2p.cooldown_steps),
                goal_xy=goal_xy,
                prev_distance=_distance_to_goal(node.get_position_xy(), goal_xy),
                episode_steps=0,
                reset_manager=reset_manager,
                last_action=np.zeros(2, dtype=np.float32),
            )

        step_dt = 1.0 / max(options.control_hz, 1e-6)
        sim_clock = contexts[options.robot_ids[0]].node.get_clock()
        step_dt_ns = int(step_dt * 1e9)
        episodes = 0
        successes = 0
        collisions = 0
        exchanges = 0

        for step in range(cfg.max_timesteps):
            current_states: dict[int, np.ndarray] = {}
            actions: dict[int, np.ndarray] = {}
            for rid, ctx in contexts.items():
                rclpy.spin_once(ctx.node, timeout_sec=0.01)
                state = ctx.node.build_state(goal_xy=ctx.goal_xy)
                action = ctx.agent.select_action(state, deterministic=False)
                ctx.node.publish_action(float(action[0]), float(action[1]))
                current_states[rid] = state
                actions[rid] = action

            deadline_ns = sim_clock.now().nanoseconds + step_dt_ns
            while sim_clock.now().nanoseconds < deadline_ns:
                for ctx in contexts.values():
                    rclpy.spin_once(ctx.node, timeout_sec=0.001)

            for rid, ctx in contexts.items():
                next_state = ctx.node.build_state(goal_xy=ctx.goal_xy)
                reward, done, is_success, is_collision = _compute_reward_done(
                    cfg=cfg,
                    node=ctx.node,
                    prev_distance=ctx.prev_distance,
                    goal_xy=ctx.goal_xy,
                    episode_steps=ctx.episode_steps + 1,
                    collision_scan_threshold=options.collision_scan_threshold,
                    current_action=actions[rid],
                    last_action=ctx.last_action,
                )
                norm_action = _normalize_action(actions[rid], cfg.action_low, cfg.action_high)
                ctx.agent.buffer.push(current_states[rid], norm_action.astype(np.float32), reward, next_state, done)
                ctx.agent.train_step()

                ctx.episode_steps += 1
                ctx.last_action = actions[rid].astype(np.float32)
                ctx.prev_distance = _distance_to_goal(ctx.node.get_position_xy(), ctx.goal_xy)
                if done:
                    episodes += 1
                    successes += int(is_success)
                    collisions += int(is_collision)
                    _reset_robot_episode(
                        ctx=ctx,
                        cfg=cfg,
                        rng=rng,
                        do_reset=options.reset_on_done,
                    )

            if step % cfg.p2p.exchange_interval_steps == 0:
                exchanges += _exchange_over_ros2(step, contexts, cfg.p2p.beta)

        communication_bytes = sum(c.node.weights_bytes_sent + c.node.weights_bytes_received for c in contexts.values())
        return ROS2RunSummary(
            timesteps=cfg.max_timesteps,
            episodes=episodes,
            successes=successes,
            collisions=collisions,
            exchanges=exchanges,
            communication_bytes=communication_bytes,
        )
    finally:
        for ctx in contexts.values():
            ctx.node.destroy_node()
        rclpy.shutdown()


def _exchange_over_ros2(step: int, contexts: dict[int, _RobotContext], beta: float) -> int:
    for ctx in contexts.values():
        ctx.node.publish_actor_weights(ctx.agent.get_actor_state(), step_idx=step)

    for _ in range(3):
        for ctx in contexts.values():
            rclpy.spin_once(ctx.node, timeout_sec=0.01)

    exchanges = 0
    for rid, ctx in contexts.items():
        incoming = ctx.node.consume_incoming_weights()
        for packet in incoming:
            if packet.sender_id not in contexts:
                continue
            local_xy = ctx.node.get_position_xy()
            peer_xy = np.asarray(packet.sender_xy, dtype=np.float32)
            if not ctx.gate.should_exchange(
                peer_id=packet.sender_id,
                local_xy=local_xy,
                peer_xy=peer_xy,
                step_idx=step,
            ):
                continue
            _blend_actor(ctx.agent, packet, beta)
            exchanges += 1
            ctx.received_bytes += packet.payload_size_bytes
    return exchanges


def _blend_actor(agent: SACAgent, packet: ReceivedWeights, beta: float) -> None:
    local = agent.get_actor_state()
    merged: dict[str, torch.Tensor] = {}
    for k in local:
        merged[k] = beta * local[k] + (1.0 - beta) * packet.actor_state[k]
    agent.load_actor_state(merged)


def _compute_reward_done(
    *,
    cfg: ExperimentConfig,
    node: ROS2RLNode,
    prev_distance: float,
    goal_xy: np.ndarray,
    episode_steps: int,
    collision_scan_threshold: float,
    current_action: np.ndarray,
    last_action: np.ndarray,
) -> tuple[float, bool, bool, bool]:
    current_distance = _distance_to_goal(node.get_position_xy(), goal_xy)
    progress = prev_distance - current_distance
    reward = cfg.reward.progress_coeff * progress + cfg.reward.step_penalty
    min_scan = node.get_min_scan()
    if min_scan < cfg.reward.danger_zone_distance:
        danger_ratio = max(0.0, 1.0 - (min_scan / cfg.reward.danger_zone_distance))
        reward -= cfg.reward.proximity_penalty_coeff * danger_ratio
    reward -= cfg.reward.action_smoothness_coeff * float(np.linalg.norm(current_action - last_action))
    is_collision = min_scan <= collision_scan_threshold
    is_success = current_distance <= cfg.goal_threshold
    done = False
    if is_collision:
        reward += cfg.reward.collision_penalty
        done = True
    elif is_success:
        reward += cfg.reward.goal_bonus
        done = True
    elif episode_steps >= cfg.max_episode_steps:
        done = True
    return float(reward), done, is_success, is_collision


def _distance_to_goal(pos_xy: np.ndarray, goal_xy: np.ndarray) -> float:
    return float(np.linalg.norm(pos_xy - goal_xy))


def _normalize_action(
    action: np.ndarray,
    action_low: tuple[float, float],
    action_high: tuple[float, float],
) -> np.ndarray:
    low = np.asarray(action_low, dtype=np.float32)
    high = np.asarray(action_high, dtype=np.float32)
    return 2.0 * (action - low) / (high - low) - 1.0


def _reset_robot_episode(
    *,
    ctx: _RobotContext,
    cfg: ExperimentConfig,
    rng: np.random.Generator,
    do_reset: bool,
) -> None:
    if do_reset and ctx.reset_manager is not None:
        x, y = sample_safe_xy(rng=rng)
        yaw = float(rng.uniform(-math.pi, math.pi))
        ctx.reset_manager.reset_robot(x=x, y=y, yaw=yaw)
    ctx.goal_xy = np.asarray(sample_safe_xy(rng=rng), dtype=np.float32)
    ctx.prev_distance = _distance_to_goal(ctx.node.get_position_xy(), ctx.goal_xy)
    ctx.episode_steps = 0
    ctx.last_action = np.zeros(2, dtype=np.float32)
