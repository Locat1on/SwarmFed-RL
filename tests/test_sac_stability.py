import unittest

import numpy as np
import torch

from swarmfed_rl.config import build_config
from swarmfed_rl.sac import SACAgent


class TestSACStability(unittest.TestCase):
    def test_train_step_finite_losses(self) -> None:
        cfg = build_config(seed=11, max_timesteps=20)
        agent = SACAgent(cfg, torch.device("cpu"))
        target_size = max(cfg.sac.batch_size, cfg.sac.update_after)
        for _ in range(target_size):
            state = np.random.uniform(-1.0, 1.0, size=(cfg.state_dim,)).astype(np.float32)
            action = np.random.uniform(-1.0, 1.0, size=(cfg.action_dim,)).astype(np.float32)
            reward = float(np.random.uniform(-1.0, 1.0))
            next_state = np.random.uniform(-1.0, 1.0, size=(cfg.state_dim,)).astype(np.float32)
            done = bool(np.random.randint(0, 2))
            agent.buffer.push(state, action, reward, next_state, done)

        out = agent.train_step()
        self.assertIn("q1_loss", out)
        self.assertTrue(np.isfinite(out["q1_loss"]))
        self.assertTrue(np.isfinite(out["q2_loss"]))
        self.assertTrue(np.isfinite(out["actor_loss"]))
        self.assertTrue(np.isfinite(out["alpha"]))


if __name__ == "__main__":
    unittest.main()
