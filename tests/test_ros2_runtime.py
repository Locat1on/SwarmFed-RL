import unittest

import numpy as np
import torch

from swarmfed_rl.config import build_config
from swarmfed_rl.ros2_runtime import (
    NeighborCooldownGate,
    ROS2StateAdapter,
    decode_actor_state,
    encode_actor_state,
    pack_weights_chunks,
    unpack_weights_chunk,
    _maybe_decompress,
)
from swarmfed_rl.p2p import aggregate_state_dicts_trimmed_mean, krum_select_index
from swarmfed_rl.sac import SACAgent


class TestROS2Runtime(unittest.TestCase):
    def test_state_adapter_output_shape(self) -> None:
        scan = np.linspace(0.1, 3.8, num=360, dtype=np.float32)
        state = ROS2StateAdapter.build_state_vector(
            scan_ranges=scan,
            linear_v=0.2,
            angular_v=-0.1,
            robot_xy=(1.0, 2.0),
            robot_yaw=0.3,
            goal_xy=(2.0, 2.5),
        )
        self.assertEqual(state.shape[0], 28)
        self.assertTrue(np.isfinite(state).all())

    def test_chunk_pack_unpack_and_decode(self) -> None:
        cfg = build_config(seed=7, max_timesteps=10)
        agent = SACAgent(cfg, torch.device("cpu"))
        payload = encode_actor_state(agent.get_actor_state())
        chunks = pack_weights_chunks(
            sender_id=2,
            step_idx=33,
            sender_xy=(1.2, -0.8),
            payload=payload,
            max_chunk_payload=512,
            enable_compression=True,
        )
        self.assertGreater(len(chunks), 1)

        parts: dict[int, bytes] = {}
        header_ref = None
        for raw in chunks:
            header, part = unpack_weights_chunk(raw)
            header_ref = header
            parts[header.chunk_index] = part

        assert header_ref is not None
        merged = b"".join(parts[i] for i in range(header_ref.chunk_total))
        decoded_payload = _maybe_decompress(merged, header_ref.flags)
        actor_state = decode_actor_state(decoded_payload)
        self.assertEqual(set(actor_state.keys()), set(agent.get_actor_state().keys()))

    def test_neighbor_cooldown_gate(self) -> None:
        gate = NeighborCooldownGate(comm_radius=2.0, cooldown_steps=5)
        local = np.array([0.0, 0.0], dtype=np.float32)
        peer = np.array([1.0, 0.0], dtype=np.float32)
        self.assertTrue(gate.should_exchange(peer_id=5, local_xy=local, peer_xy=peer, step_idx=10))
        self.assertFalse(gate.should_exchange(peer_id=5, local_xy=local, peer_xy=peer, step_idx=11))
        self.assertTrue(gate.should_exchange(peer_id=5, local_xy=local, peer_xy=peer, step_idx=15))

    def test_trimmed_mean_and_krum(self) -> None:
        base = {"w": torch.tensor([1.0, 1.1, 0.9])}
        near = {"w": torch.tensor([1.05, 1.0, 0.95])}
        outlier = {"w": torch.tensor([50.0, -40.0, 30.0])}
        agg = aggregate_state_dicts_trimmed_mean([base, near, outlier], trim_ratio=0.33)
        self.assertTrue(torch.all(torch.isfinite(agg["w"])))
        vectors = [base["w"], near["w"], outlier["w"]]
        idx = krum_select_index(vectors, malicious_count=1)
        self.assertIn(idx, [0, 1])


if __name__ == "__main__":
    unittest.main()
