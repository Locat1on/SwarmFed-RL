from __future__ import annotations

import io
import math
import struct
import time
import zlib
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import torch

try:  # pragma: no cover
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import ByteMultiArray
except ImportError:  # pragma: no cover
    rclpy = None
    Node = object  # type: ignore[assignment]
    Parameter = None  # type: ignore[assignment]
    qos_profile_sensor_data = None  # type: ignore[assignment]
    LaserScan = object  # type: ignore[assignment]
    Odometry = object  # type: ignore[assignment]
    Twist = object  # type: ignore[assignment]
    ByteMultiArray = object  # type: ignore[assignment]

try:  # pragma: no cover
    from gazebo_msgs.msg import EntityState, ModelState
    from gazebo_msgs.srv import SetEntityState, SetModelState
except ImportError:  # pragma: no cover
    EntityState = None  # type: ignore[assignment]
    ModelState = None  # type: ignore[assignment]
    SetEntityState = None  # type: ignore[assignment]
    SetModelState = None  # type: ignore[assignment]


def ros2_available() -> bool:
    return rclpy is not None


@dataclass
class ReceivedWeights:
    sender_id: int
    step_idx: int
    sender_xy: tuple[float, float]
    actor_state: dict[str, torch.Tensor]
    payload_size_bytes: int


@dataclass(frozen=True)
class ChunkHeader:
    sender_id: int
    step_idx: int
    sender_xy: tuple[float, float]
    flags: int
    chunk_index: int
    chunk_total: int
    payload_len: int
    payload_crc32: int


class ChunkReassembler:
    def __init__(self, ttl_sec: float = 2.0) -> None:
        self.ttl_sec = ttl_sec
        self._pending: dict[
            tuple[int, int, int, int, int],
            dict[str, object],
        ] = {}

    def add_chunk(self, raw: bytes) -> tuple[ChunkHeader, bytes, int] | None:
        header, chunk_payload = unpack_weights_chunk(raw)
        key = (
            header.sender_id,
            header.step_idx,
            header.flags,
            header.payload_len,
            header.payload_crc32,
        )
        now = time.time()
        self._cleanup(now)

        bucket = self._pending.get(key)
        if bucket is None:
            bucket = {
                "created_at": now,
                "chunks": {},
                "total": header.chunk_total,
                "raw_bytes": 0,
                "sender_xy": header.sender_xy,
            }
            self._pending[key] = bucket

        chunks: dict[int, bytes] = bucket["chunks"]  # type: ignore[assignment]
        chunks[header.chunk_index] = chunk_payload
        bucket["raw_bytes"] = int(bucket["raw_bytes"]) + len(raw)

        if len(chunks) < int(bucket["total"]):
            return None

        merged = b"".join(chunks[i] for i in range(int(bucket["total"])))
        if len(merged) != header.payload_len:
            self._pending.pop(key, None)
            raise ValueError(f"Reassembled payload length mismatch: {len(merged)} != {header.payload_len}")
        if crc32_u32(merged) != header.payload_crc32:
            self._pending.pop(key, None)
            raise ValueError("CRC mismatch for reassembled actor payload")

        self._pending.pop(key, None)
        return header, merged, int(bucket["raw_bytes"])

    def _cleanup(self, now: float) -> None:
        stale = []
        for k, v in self._pending.items():
            if now - float(v["created_at"]) > self.ttl_sec:
                stale.append(k)
        for k in stale:
            self._pending.pop(k, None)


class ROS2StateAdapter:
    @staticmethod
    def downsample_scan(
        ranges: list[float] | tuple[float, ...] | np.ndarray,
        bins: int = 24,
        min_range: float = 0.02,
        max_range: float = 4.0,
    ) -> np.ndarray:
        arr = np.asarray(ranges, dtype=np.float32)
        if arr.size == 0:
            return np.full((bins,), max_range, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=max_range, posinf=max_range, neginf=min_range)
        arr = np.clip(arr, min_range, max_range)
        src = np.linspace(0, arr.size - 1, num=arr.size, dtype=np.float32)
        dst = np.linspace(0, arr.size - 1, num=bins, dtype=np.float32)
        sampled = np.interp(dst, src, arr).astype(np.float32)
        return sampled

    @staticmethod
    def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(math.atan2(siny_cosp, cosy_cosp))

    @staticmethod
    def build_state_vector(
        *,
        scan_ranges: list[float] | tuple[float, ...] | np.ndarray,
        linear_v: float,
        angular_v: float,
        robot_xy: tuple[float, float] | np.ndarray,
        robot_yaw: float,
        goal_xy: tuple[float, float] | np.ndarray,
        lidar_bins: int = 24,
        lidar_max_range: float = 4.0,
    ) -> np.ndarray:
        lidar = ROS2StateAdapter.downsample_scan(
            scan_ranges,
            bins=lidar_bins,
            min_range=0.02,
            max_range=lidar_max_range,
        )
        rx, ry = float(robot_xy[0]), float(robot_xy[1])
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        dx, dy = gx - rx, gy - ry
        dist = math.sqrt(dx * dx + dy * dy)
        heading = math.atan2(dy, dx)
        err = normalize_angle(heading - robot_yaw)
        tail = np.array([linear_v, angular_v, dist, err], dtype=np.float32)
        return np.concatenate([lidar, tail], axis=0).astype(np.float32)


def normalize_angle(theta: float) -> float:
    while theta > math.pi:
        theta -= 2 * math.pi
    while theta < -math.pi:
        theta += 2 * math.pi
    return theta


class NeighborCooldownGate:
    def __init__(self, comm_radius: float, cooldown_steps: int) -> None:
        self.comm_radius = comm_radius
        self.cooldown_steps = cooldown_steps
        self._last_exchange: dict[int, int] = defaultdict(lambda: -10**9)

    def should_exchange(
        self,
        *,
        peer_id: int,
        local_xy: np.ndarray,
        peer_xy: np.ndarray,
        step_idx: int,
    ) -> bool:
        if float(np.linalg.norm(local_xy - peer_xy)) >= self.comm_radius:
            return False
        if step_idx - self._last_exchange[peer_id] < self.cooldown_steps:
            return False
        self._last_exchange[peer_id] = step_idx
        return True


def encode_actor_state(actor_state: dict[str, torch.Tensor]) -> bytes:
    buf = io.BytesIO()
    cpu_state = {k: v.detach().cpu() for k, v in actor_state.items()}
    torch.save(cpu_state, buf)
    return buf.getvalue()


def decode_actor_state(raw: bytes) -> dict[str, torch.Tensor]:
    buf = io.BytesIO(raw)
    state = torch.load(buf, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError("Decoded actor payload is not a state dict")
    return state


_MSG_HEADER = struct.Struct("<IIffI")
_CHUNK_MAGIC = b"SFRL"
_CHUNK_VERSION = 1
_FLAG_COMPRESSED = 1
_CHUNK_HEADER = struct.Struct("<4sBBIIffHHII")


def pack_weights_message(
    *,
    sender_id: int,
    step_idx: int,
    sender_xy: tuple[float, float],
    payload: bytes,
) -> bytes:
    x, y = float(sender_xy[0]), float(sender_xy[1])
    head = _MSG_HEADER.pack(sender_id, step_idx, x, y, len(payload))
    return head + payload


def unpack_weights_message(raw: bytes) -> tuple[int, int, tuple[float, float], bytes]:
    if len(raw) < _MSG_HEADER.size:
        raise ValueError("Payload too small for actor-weights message")
    sender_id, step_idx, x, y, payload_len = _MSG_HEADER.unpack_from(raw, 0)
    payload = raw[_MSG_HEADER.size :]
    if len(payload) != payload_len:
        raise ValueError(f"Payload length mismatch: expected {payload_len}, got {len(payload)}")
    return sender_id, step_idx, (x, y), payload


def crc32_u32(raw: bytes) -> int:
    return int(zlib.crc32(raw) & 0xFFFFFFFF)


def _maybe_compress(raw: bytes, enabled: bool = True, min_size: int = 4096) -> tuple[bytes, int]:
    if (not enabled) or len(raw) < min_size:
        return raw, 0
    compressed = zlib.compress(raw, level=6)
    if len(compressed) >= len(raw):
        return raw, 0
    return compressed, _FLAG_COMPRESSED


def _maybe_decompress(raw: bytes, flags: int) -> bytes:
    if flags & _FLAG_COMPRESSED:
        return zlib.decompress(raw)
    return raw


def pack_weights_chunks(
    *,
    sender_id: int,
    step_idx: int,
    sender_xy: tuple[float, float],
    payload: bytes,
    max_chunk_payload: int = 4096,
    enable_compression: bool = True,
) -> list[bytes]:
    if max_chunk_payload <= 0:
        raise ValueError("max_chunk_payload must be > 0")
    framed_payload, flags = _maybe_compress(payload, enabled=enable_compression)
    payload_crc = crc32_u32(framed_payload)
    total = max(1, int(math.ceil(len(framed_payload) / max_chunk_payload)))
    chunks: list[bytes] = []

    sx, sy = float(sender_xy[0]), float(sender_xy[1])
    for idx in range(total):
        start = idx * max_chunk_payload
        end = min((idx + 1) * max_chunk_payload, len(framed_payload))
        part = framed_payload[start:end]
        header = _CHUNK_HEADER.pack(
            _CHUNK_MAGIC,
            _CHUNK_VERSION,
            flags,
            int(sender_id),
            int(step_idx),
            sx,
            sy,
            int(idx),
            int(total),
            int(len(framed_payload)),
            int(payload_crc),
        )
        chunks.append(header + part)
    return chunks


def unpack_weights_chunk(raw: bytes) -> tuple[ChunkHeader, bytes]:
    if len(raw) < _CHUNK_HEADER.size:
        raise ValueError("Chunk too small")
    magic, version, flags, sender_id, step_idx, x, y, idx, total, payload_len, payload_crc = _CHUNK_HEADER.unpack_from(
        raw, 0
    )
    if magic != _CHUNK_MAGIC:
        raise ValueError("Invalid chunk magic")
    if version != _CHUNK_VERSION:
        raise ValueError(f"Unsupported chunk version: {version}")
    if total <= 0:
        raise ValueError("Invalid total chunk count")
    if idx >= total:
        raise ValueError("Chunk index out of range")
    part = raw[_CHUNK_HEADER.size :]
    header = ChunkHeader(
        sender_id=int(sender_id),
        step_idx=int(step_idx),
        sender_xy=(float(x), float(y)),
        flags=int(flags),
        chunk_index=int(idx),
        chunk_total=int(total),
        payload_len=int(payload_len),
        payload_crc32=int(payload_crc),
    )
    return header, part


class ROS2RLNode(Node):  # pragma: no cover
    def __init__(
        self,
        *,
        robot_id: int,
        node_name: str = "rl_control_node",
        scan_topic: str = "/scan",
        odom_topic: str = "/odom",
        cmd_topic: str = "/cmd_vel",
        weights_topic: str = "/actor_weights",
        qos_depth: int = 10,
        max_chunk_payload: int = 4096,
        retransmit_count: int = 1,
    ) -> None:
        if rclpy is None:
            raise RuntimeError("rclpy is not available in this environment")
        super().__init__(node_name)
        # Keep control loop aligned with Gazebo /clock time when available.
        self.set_parameters([Parameter("use_sim_time", value=True)])
        self.robot_id = robot_id
        self._scan_ranges: list[float] | None = None
        self._linear_v = 0.0
        self._angular_v = 0.0
        self._position_xy = np.zeros(2, dtype=np.float32)
        self._yaw = 0.0
        self._incoming_weights: Deque[ReceivedWeights] = deque()
        self.weights_bytes_sent = 0
        self.weights_bytes_received = 0
        self.max_chunk_payload = max_chunk_payload
        self.retransmit_count = max(1, retransmit_count)
        self._reassembler = ChunkReassembler(ttl_sec=2.0)

        self.scan_sub = self.create_subscription(LaserScan, scan_topic, self._on_scan, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self._on_odom, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, qos_depth)
        self.weights_pub = self.create_publisher(ByteMultiArray, weights_topic, qos_depth)
        self.weights_sub = self.create_subscription(ByteMultiArray, weights_topic, self._on_weights, qos_depth)

    def _on_scan(self, msg: LaserScan) -> None:
        self._scan_ranges = list(msg.ranges)

    def _on_odom(self, msg: Odometry) -> None:
        self._position_xy[0] = float(msg.pose.pose.position.x)
        self._position_xy[1] = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self._yaw = ROS2StateAdapter.quaternion_to_yaw(float(q.x), float(q.y), float(q.z), float(q.w))
        self._linear_v = float(msg.twist.twist.linear.x)
        self._angular_v = float(msg.twist.twist.angular.z)

    def _on_weights(self, msg: ByteMultiArray) -> None:
        raw = bytes(msg.data)
        self.weights_bytes_received += len(raw)
        try:
            assembled = self._reassembler.add_chunk(raw)
            if assembled is None:
                return
            header, framed_payload, raw_size = assembled
            if header.sender_id == self.robot_id:
                return
            payload = _maybe_decompress(framed_payload, header.flags)
            actor_state = decode_actor_state(payload)
            self._incoming_weights.append(
                ReceivedWeights(
                    sender_id=header.sender_id,
                    step_idx=header.step_idx,
                    sender_xy=header.sender_xy,
                    actor_state=actor_state,
                    payload_size_bytes=raw_size,
                )
            )
            return
        except ValueError:
            if raw.startswith(_CHUNK_MAGIC):
                return

        # Backward compatibility for legacy single-frame payloads.
        sender_id, step_idx, sender_xy, payload = unpack_weights_message(raw)
        if sender_id == self.robot_id:
            return
        actor_state = decode_actor_state(payload)
        self._incoming_weights.append(
            ReceivedWeights(
                sender_id=sender_id,
                step_idx=step_idx,
                sender_xy=sender_xy,
                actor_state=actor_state,
                payload_size_bytes=len(raw),
            )
        )

    def publish_action(self, linear_v: float, angular_w: float) -> None:
        cmd = Twist()
        cmd.linear.x = float(linear_v)
        cmd.angular.z = float(angular_w)
        self.cmd_pub.publish(cmd)

    def ready(self) -> bool:
        return self._scan_ranges is not None

    def wait_until_ready(self, timeout_sec: float = 10.0, spin_interval_sec: float = 0.05) -> bool:
        if rclpy is None:
            return False
        start = time.time()
        while (time.time() - start) < timeout_sec:
            if self.ready():
                return True
            rclpy.spin_once(self, timeout_sec=spin_interval_sec)
        return False

    def build_state(self, goal_xy: tuple[float, float] | np.ndarray, lidar_bins: int = 24) -> np.ndarray:
        if self._scan_ranges is None:
            raise RuntimeError("Scan data not ready; call wait_until_ready first")
        return ROS2StateAdapter.build_state_vector(
            scan_ranges=self._scan_ranges,
            linear_v=self._linear_v,
            angular_v=self._angular_v,
            robot_xy=self._position_xy,
            robot_yaw=self._yaw,
            goal_xy=goal_xy,
            lidar_bins=lidar_bins,
        )

    def get_position_xy(self) -> np.ndarray:
        return self._position_xy.copy()

    def get_min_scan(self) -> float:
        if self._scan_ranges is None or len(self._scan_ranges) == 0:
            return float("inf")
        return float(np.nanmin(np.asarray(self._scan_ranges, dtype=np.float32)))

    def publish_actor_weights(self, actor_state: dict[str, torch.Tensor], step_idx: int) -> int:
        payload = encode_actor_state(actor_state)
        chunks = pack_weights_chunks(
            sender_id=self.robot_id,
            step_idx=step_idx,
            sender_xy=(float(self._position_xy[0]), float(self._position_xy[1])),
            payload=payload,
            max_chunk_payload=self.max_chunk_payload,
            enable_compression=True,
        )
        total_bytes = 0
        for _ in range(self.retransmit_count):
            for raw in chunks:
                msg = ByteMultiArray()
                msg.data = list(raw)
                self.weights_pub.publish(msg)
                total_bytes += len(raw)
        self.weights_bytes_sent += total_bytes
        return total_bytes

    def consume_incoming_weights(self) -> list[ReceivedWeights]:
        packets = list(self._incoming_weights)
        self._incoming_weights.clear()
        return packets


class GazeboResetManager:  # pragma: no cover
    def __init__(self, node: Node, model_name: str) -> None:
        if rclpy is None:
            raise RuntimeError("rclpy is not available in this environment")
        self.node = node
        self.model_name = model_name
        self._client = None
        self._service_type = None
        self._select_service_client()

    def _select_service_client(self) -> None:
        if SetEntityState is not None:
            self._client = self.node.create_client(SetEntityState, "/gazebo/set_entity_state")
            self._service_type = "entity"
            return
        if SetModelState is not None:
            self._client = self.node.create_client(SetModelState, "/gazebo/set_model_state")
            self._service_type = "model"
            return
        raise RuntimeError("gazebo_msgs service types are unavailable; install gazebo_ros_pkgs")

    def reset_robot(self, x: float, y: float, yaw: float, timeout_sec: float = 2.0) -> bool:
        if self._client is None:
            raise RuntimeError("Gazebo reset client is not initialized")
        if not self._client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError("Gazebo reset service is unavailable")

        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        if self._service_type == "entity":
            req = SetEntityState.Request()
            req.state = EntityState()
            req.state.name = self.model_name
            req.state.pose.position.x = float(x)
            req.state.pose.position.y = float(y)
            req.state.pose.orientation.z = float(qz)
            req.state.pose.orientation.w = float(qw)
        else:
            req = SetModelState.Request()
            req.model_state = ModelState()
            req.model_state.model_name = self.model_name
            req.model_state.pose.position.x = float(x)
            req.model_state.pose.position.y = float(y)
            req.model_state.pose.orientation.z = float(qz)
            req.model_state.pose.orientation.w = float(qw)

        future = self._client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout_sec)
        if not future.done():
            raise RuntimeError("Gazebo reset request timed out")
        result = future.result()
        if result is None:
            raise RuntimeError("Gazebo reset service returned no result")
        if hasattr(result, "success") and not bool(result.success):
            msg = getattr(result, "status_message", "unknown error")
            raise RuntimeError(f"Gazebo reset failed: {msg}")
        return True


def sample_safe_xy(
    *,
    rng: np.random.Generator,
    x_min: float = -2.0,
    x_max: float = 2.0,
    y_min: float = -2.0,
    y_max: float = 2.0,
) -> tuple[float, float]:
    x = float(rng.uniform(x_min, x_max))
    y = float(rng.uniform(y_min, y_max))
    return x, y
