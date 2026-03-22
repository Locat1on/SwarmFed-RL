from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import ExperimentConfig


@dataclass
class RobotState:
    x: float
    y: float
    yaw: float
    v: float
    omega: float


class SimulatedROS2Env:
    """
    A ROS2-compatible local simulator scaffold:
    - state: 24 lidar beams + [v, omega, distance_to_goal, heading_error]
    - action: [linear_velocity, angular_velocity]
    - reward follows README composite rule.
    - Random circular obstacles regenerated each episode.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        robot_id: int,
        world_size: float = 8.0,
    ) -> None:
        self.cfg = cfg
        self.robot_id = robot_id
        self.world_size = world_size
        self.num_obstacles = cfg.num_obstacles
        self.obstacle_radius = cfg.obstacle_radius
        self.rng = np.random.default_rng(cfg.seed.seed + robot_id * 31)
        self.state = RobotState(0.0, 0.0, 0.0, 0.0, 0.0)
        self.goal = np.zeros(2, dtype=np.float32)
        self.obstacles = np.zeros((self.num_obstacles, 2), dtype=np.float32)
        self.prev_distance = 0.0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.frame_stack = max(1, int(getattr(cfg, "frame_stack", 1)))
        self._lidar_history: deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self.episode_steps = 0
        self.collisions = 0
        self.success = 0
        # Precompute lidar beam angles (fixed across episodes)
        self._beam_count = 24
        self._max_range = 4.0
        self._beam_rel_angles = np.linspace(
            -math.pi, math.pi, self._beam_count, endpoint=False, dtype=np.float64
        )
        self.reset()

    def reset(self) -> np.ndarray:
        self.state = RobotState(
            x=float(self.rng.uniform(-2.0, 2.0)),
            y=float(self.rng.uniform(-2.0, 2.0)),
            yaw=float(self.rng.uniform(-math.pi, math.pi)),
            v=0.0,
            omega=0.0,
        )
        self.goal = np.array(
            [self.rng.uniform(-3.0, 3.0), self.rng.uniform(-3.0, 3.0)],
            dtype=np.float32,
        )
        self._place_obstacles()
        self.prev_distance = self._distance_to_goal()
        self.last_action = np.zeros(2, dtype=np.float32)
        self._lidar_history.clear()
        first_lidar = self._lidar_vectorized()
        for _ in range(self.frame_stack):
            self._lidar_history.append(first_lidar.copy())
        self.episode_steps = 0
        return self._build_state()

    def _place_obstacles(self) -> None:
        """Place random circular obstacles avoiding robot spawn and goal."""
        margin = self.world_size / 2.0 - self.obstacle_radius - 0.1
        robot_pos = np.array([self.state.x, self.state.y])
        min_dist_from_entities = self.obstacle_radius + 0.5  # safe clearance
        placed = 0
        attempts = 0
        while placed < self.num_obstacles and attempts < self.num_obstacles * 20:
            cx = float(self.rng.uniform(-margin, margin))
            cy = float(self.rng.uniform(-margin, margin))
            pos = np.array([cx, cy])
            # Don't overlap with robot or goal
            if np.linalg.norm(pos - robot_pos) < min_dist_from_entities:
                attempts += 1
                continue
            if np.linalg.norm(pos - self.goal) < min_dist_from_entities:
                attempts += 1
                continue
            # Don't overlap with other obstacles
            if placed > 0:
                dists = np.linalg.norm(self.obstacles[:placed] - pos, axis=1)
                if np.any(dists < self.obstacle_radius * 2.5):
                    attempts += 1
                    continue
            self.obstacles[placed] = pos
            placed += 1
            attempts += 1
        # If couldn't place all, zero out remaining
        if placed < self.num_obstacles:
            self.obstacles[placed:] = np.inf

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        self.episode_steps += 1
        v = float(np.clip(action[0], self.cfg.action_low[0], self.cfg.action_high[0]))
        omega = float(np.clip(action[1], self.cfg.action_low[1], self.cfg.action_high[1]))
        dt = 0.1
        self.state.v = v
        self.state.omega = omega
        self.state.yaw += omega * dt
        self.state.x += v * math.cos(self.state.yaw) * dt
        self.state.y += v * math.sin(self.state.yaw) * dt

        current_distance = self._distance_to_goal()
        progress = self.prev_distance - current_distance
        reward = self.cfg.reward.progress_coeff * progress + self.cfg.reward.step_penalty
        danger_zone_distance = float(getattr(self.cfg.reward, "danger_zone_distance", 0.5))
        proximity_penalty_coeff = float(getattr(self.cfg.reward, "proximity_penalty_coeff", 0.5))
        action_smoothness_coeff = float(getattr(self.cfg.reward, "action_smoothness_coeff", 0.1))
        lidar = self._lidar_vectorized()
        self._lidar_history.append(lidar.copy())
        min_laser_dist = float(np.min(lidar))
        if min_laser_dist < danger_zone_distance:
            danger_ratio = max(0.0, 1.0 - (min_laser_dist / danger_zone_distance))
            reward -= proximity_penalty_coeff * danger_ratio
        action_diff = float(np.linalg.norm(np.asarray([v, omega], dtype=np.float32) - self.last_action))
        reward -= action_smoothness_coeff * action_diff
        self.last_action = np.asarray([v, omega], dtype=np.float32)
        done = False
        info: dict[str, float | bool] = {"collision": False, "success": False}

        if self._check_collision():
            reward += self.cfg.reward.collision_penalty
            done = True
            self.collisions += 1
            info["collision"] = True
        elif current_distance <= self.cfg.goal_threshold:
            reward += self.cfg.reward.goal_bonus
            done = True
            self.success += 1
            info["success"] = True
        elif self.episode_steps >= self.cfg.max_episode_steps:
            done = True

        self.prev_distance = current_distance
        tail = self._build_tail()
        stacked_lidar = self._stacked_lidar(lidar)
        next_state = np.concatenate([stacked_lidar, tail], axis=0).astype(np.float32)
        return next_state, float(reward), done, info

    def get_position(self) -> np.ndarray:
        return np.array([self.state.x, self.state.y], dtype=np.float32)

    def _distance_to_goal(self) -> float:
        return float(np.linalg.norm(np.array([self.state.x, self.state.y], dtype=np.float32) - self.goal))

    def _check_collision(self) -> bool:
        margin = self.world_size / 2.0
        # Wall collision
        if (
            self.state.x <= -margin
            or self.state.x >= margin
            or self.state.y <= -margin
            or self.state.y >= margin
        ):
            return True
        # Obstacle collision
        robot_pos = np.array([self.state.x, self.state.y])
        dists = np.linalg.norm(self.obstacles - robot_pos, axis=1)
        if np.any(dists <= self.obstacle_radius):
            return True
        return False

    def _build_state(self) -> np.ndarray:
        lidar = self._lidar_vectorized()
        self._lidar_history.append(lidar.copy())
        tail = self._build_tail()
        return np.concatenate([self._stacked_lidar(lidar), tail], axis=0).astype(np.float32)

    def _build_tail(self) -> np.ndarray:
        goal_dx = float(self.goal[0] - self.state.x)
        goal_dy = float(self.goal[1] - self.state.y)
        dist = math.sqrt(goal_dx * goal_dx + goal_dy * goal_dy)
        goal_heading = math.atan2(goal_dy, goal_dx)
        heading_error = self._normalize_angle(goal_heading - self.state.yaw)
        return np.array([self.state.v, self.state.omega, dist, heading_error], dtype=np.float32)

    def _lidar_vectorized(self) -> np.ndarray:
        """Vectorized lidar: walls + circular obstacles, no Python loops."""
        x, y = self.state.x, self.state.y
        margin = self.world_size / 2.0
        max_range = self._max_range
        beam_count = self._beam_count

        angles = self.state.yaw + self._beam_rel_angles  # (24,)
        dx = np.cos(angles)  # (24,)
        dy = np.sin(angles)  # (24,)

        distances = np.full(beam_count, max_range, dtype=np.float64)

        # --- Wall intersections (vectorized) ---
        # Left wall: x = -margin
        with np.errstate(divide="ignore", invalid="ignore"):
            t_left = (-margin - x) / dx
            y_left = y + t_left * dy
            valid = (t_left > 0) & (np.abs(y_left) <= margin)
            distances = np.where(valid, np.minimum(distances, t_left), distances)

            # Right wall: x = +margin
            t_right = (margin - x) / dx
            y_right = y + t_right * dy
            valid = (t_right > 0) & (np.abs(y_right) <= margin)
            distances = np.where(valid, np.minimum(distances, t_right), distances)

            # Bottom wall: y = -margin
            t_down = (-margin - y) / dy
            x_down = x + t_down * dx
            valid = (t_down > 0) & (np.abs(x_down) <= margin)
            distances = np.where(valid, np.minimum(distances, t_down), distances)

            # Top wall: y = +margin
            t_up = (margin - y) / dy
            x_up = x + t_up * dx
            valid = (t_up > 0) & (np.abs(x_up) <= margin)
            distances = np.where(valid, np.minimum(distances, t_up), distances)

        # --- Obstacle intersections (vectorized over beams × obstacles) ---
        if self.num_obstacles > 0:
            r = self.obstacle_radius
            # ox, oy: offset from ray origin to obstacle center
            # obstacles: (N, 2), dx/dy: (24,)
            ox = self.obstacles[:, 0] - x  # (N,)
            oy = self.obstacles[:, 1] - y  # (N,)

            # Broadcast: (N, 1) vs (1, 24) -> (N, 24)
            # a = dx² + dy² = 1 for unit vectors, but keep general
            # b = 2(ox·dx + oy·dy), c = ox² + oy² - r²
            dot = ox[:, None] * dx[None, :] + oy[:, None] * dy[None, :]  # (N, 24)
            dist_sq = ox ** 2 + oy ** 2  # (N,)
            disc = dot ** 2 - (dist_sq[:, None] - r ** 2)  # (N, 24)

            hit_mask = disc >= 0  # (N, 24)
            sqrt_disc = np.where(hit_mask, np.sqrt(np.maximum(disc, 0.0)), 0.0)
            t_hit = dot - sqrt_disc  # nearest intersection (N, 24)
            t_hit = np.where(hit_mask & (t_hit > 1e-6), t_hit, max_range + 1)

            t_min_obs = np.min(t_hit, axis=0)  # (24,)
            distances = np.minimum(distances, t_min_obs)

        distances = np.clip(distances, 0.0, max_range)
        noise = self.rng.normal(0.0, 0.02, size=beam_count)
        return np.clip(distances + noise, 0.02, max_range).astype(np.float32)

    def _stacked_lidar(self, latest_lidar: np.ndarray) -> np.ndarray:
        if not self._lidar_history:
            return latest_lidar
        while len(self._lidar_history) < self.frame_stack:
            self._lidar_history.appendleft(self._lidar_history[0].copy())
        return np.concatenate(list(self._lidar_history), axis=0).astype(np.float32)

    @staticmethod
    def _normalize_angle(theta: float) -> float:
        return float((theta + math.pi) % (2.0 * math.pi) - math.pi)
