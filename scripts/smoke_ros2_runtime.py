import argparse

from swarmfed_rl.ros2_runtime import (
    ROS2StateAdapter,
    ChunkReassembler,
    decode_actor_state,
    encode_actor_state,
    pack_weights_chunks,
    pack_weights_message,
    unpack_weights_chunk,
    unpack_weights_message,
)
from swarmfed_rl.sac import SACAgent
from swarmfed_rl.config import build_config

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for ROS2 adapter and payload codec")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = build_config(seed=args.seed, max_timesteps=10)
    device = torch.device("cpu")
    agent = SACAgent(cfg, device)

    scan = np.linspace(0.1, 3.5, num=360, dtype=np.float32)
    state = ROS2StateAdapter.build_state_vector(
        scan_ranges=scan,
        linear_v=0.1,
        angular_v=0.0,
        robot_xy=(0.0, 0.0),
        robot_yaw=0.0,
        goal_xy=(1.0, 1.0),
    )
    if state.shape[0] != cfg.state_dim:
        raise RuntimeError(f"State shape mismatch: {state.shape[0]} != {cfg.state_dim}")

    actor_state = agent.get_actor_state()
    payload = encode_actor_state(actor_state)
    msg = pack_weights_message(sender_id=1, step_idx=5, sender_xy=(1.5, -0.5), payload=payload)
    sender_id, step_idx, sender_xy, raw = unpack_weights_message(msg)
    decoded = decode_actor_state(raw)
    if sender_id != 1 or step_idx != 5:
        raise RuntimeError("Message header decode mismatch")
    if abs(sender_xy[0] - 1.5) > 1e-6 or abs(sender_xy[1] + 0.5) > 1e-6:
        raise RuntimeError("Sender position decode mismatch")
    if set(decoded.keys()) != set(actor_state.keys()):
        raise RuntimeError("Decoded actor keys mismatch")

    chunked = pack_weights_chunks(
        sender_id=2,
        step_idx=8,
        sender_xy=(0.2, 0.3),
        payload=payload,
        max_chunk_payload=512,
        enable_compression=True,
    )
    reassembler = ChunkReassembler()
    merged_payload = None
    for raw in chunked:
        out = reassembler.add_chunk(raw)
        if out is not None:
            header, packed_payload, _ = out
            header_check, _ = unpack_weights_chunk(raw)
            if header.sender_id != 2 or header.step_idx != 8:
                raise RuntimeError("Chunk header mismatch")
            if header_check.sender_id != header.sender_id:
                raise RuntimeError("Chunk header parse mismatch")
            if header.flags & 1:
                import zlib

                merged_payload = zlib.decompress(packed_payload)
            else:
                merged_payload = packed_payload
    if merged_payload is None:
        raise RuntimeError("Chunk reassembly did not complete")
    decoded2 = decode_actor_state(merged_payload)
    if set(decoded2.keys()) != set(actor_state.keys()):
        raise RuntimeError("Chunk decoded actor keys mismatch")

    print("ROS2 runtime smoke passed")


if __name__ == "__main__":
    main()
