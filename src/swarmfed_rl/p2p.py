"""P2P federated aggregation with defence mechanisms.

Simplified for convergence:
- Synchronous exchange (no async races)
- Cleaner beta semantics: beta = weight on LOCAL model
- No FP16 quantization by default (stability)
- No layer filtering by default (exchange all)
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from .config import P2PConfig
from .sac import SACAgent

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def quantize_state_fp16(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.half() for k, v in state.items()}


def dequantize_state_fp16(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.float() for k, v in state.items()}


def selective_layer_filter(
    local: dict[str, torch.Tensor],
    incoming: dict[str, torch.Tensor],
    threshold: float = 0.001,
) -> dict[str, torch.Tensor]:
    filtered = {}
    for k in local:
        if k not in incoming:
            continue
        if float(torch.std(incoming[k] - local[k]).item()) >= threshold:
            filtered[k] = incoming[k]
    return filtered


def cosine_similarity_state_dict(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
) -> float:
    a_vec = torch.cat([state_a[k].float().reshape(-1) for k in state_a])
    b_vec = torch.cat([state_b[k].float().reshape(-1) for k in state_b])
    denom = torch.linalg.norm(a_vec) * torch.linalg.norm(b_vec)
    if denom == 0.0:
        return 0.0
    return float((torch.dot(a_vec, b_vec) / denom).item())


def state_dict_to_vector(state: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.reshape(-1).float() for v in state.values()], dim=0)


def aggregate_state_dicts_trimmed_mean(
    states: list[dict[str, torch.Tensor]],
    trim_ratio: float,
) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("states cannot be empty")
    if len(states) == 1:
        return states[0]
    trim_ratio = float(max(0.0, min(0.49, trim_ratio)))
    trim_k = int(len(states) * trim_ratio)
    out: dict[str, torch.Tensor] = {}
    for k in states[0]:
        stacked = torch.stack([s[k] for s in states], dim=0)
        if trim_k > 0 and (len(states) - 2 * trim_k) > 0:
            sorted_vals, _ = torch.sort(stacked, dim=0)
            out[k] = sorted_vals[trim_k:len(states) - trim_k].mean(dim=0)
        else:
            out[k] = stacked.mean(dim=0)
    return out


def krum_select_index(vectors: list[torch.Tensor], malicious_count: int) -> int:
    n = len(vectors)
    if n == 0:
        raise ValueError("vectors cannot be empty")
    if n == 1:
        return 0
    if n == 2:
        return 1 if malicious_count == 0 else 0
    f = max(0, malicious_count)
    neighbor_count = max(1, n - f - 2)
    stacked = torch.stack(vectors)
    distances = torch.cdist(stacked.unsqueeze(0), stacked.unsqueeze(0)).squeeze(0)
    distances.fill_diagonal_(float("inf"))
    k = min(neighbor_count, n - 1)
    nearest_dists, _ = torch.topk(distances, k, dim=1, largest=False)
    return int(torch.argmin(nearest_dists.sum(dim=1)).item())


# ---------------------------------------------------------------------------
# Stats / dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DefenseStats:
    accepted_updates: int = 0
    rejected_updates: int = 0
    rejected_malicious_updates: int = 0
    accepted_malicious_updates: int = 0
    potential_malicious_nodes: set[int] | None = None

    def __post_init__(self) -> None:
        if self.potential_malicious_nodes is None:
            self.potential_malicious_nodes = set()


@dataclass
class IncomingCandidate:
    sender_id: int
    state: dict[str, torch.Tensor]
    sender_is_malicious: bool
    distance: float = 0.0


class RunningStat:
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        d1 = x - self.mean
        self.mean += d1 / self.n
        d2 = x - self.mean
        self.m2 += d1 * d2

    @property
    def std(self) -> float:
        return float((self.m2 / (self.n - 1)) ** 0.5) if self.n >= 2 else 0.0


# ---------------------------------------------------------------------------
# P2P Aggregator
# ---------------------------------------------------------------------------

class P2PAggregator:
    """Synchronous actor-only aggregation with optional robust defences."""

    def __init__(self, cfg: P2PConfig) -> None:
        self.cfg = cfg
        self._last_exchange: dict[tuple[int, int], int] = defaultdict(lambda: -(10**9))
        self.bytes_transferred = 0
        self.defense_stats = DefenseStats()
        self._sim_stat = RunningStat()
        self.rejected_by_sender: dict[int, int] = defaultdict(int)
        self.last_similarity_threshold = -1.0
        self._lock = threading.Lock()

    def maybe_exchange(
        self,
        step_idx: int,
        agents: dict[int, SACAgent],
        positions: dict[int, np.ndarray],
        *,
        progress: float = 0.0,
        malicious_nodes: set[int] | None = None,
        attack_type: str = "zero",
        defense_enabled: bool = False,
        defense_strategy: str = "cosine",
        defense_trim_ratio: float = 0.2,
        defense_krum_malicious: int = 1,
        calibration_steps: int = 0,
        attack_start_step: int = 0,
    ) -> tuple[int, dict[int, dict[str, torch.Tensor]]]:
        if step_idx % self.cfg.exchange_interval_steps != 0:
            return 0, {}
        if defense_strategy not in {"cosine", "trimmed_mean", "krum"}:
            raise ValueError(f"Unsupported defense strategy: {defense_strategy}")

        malicious_nodes = malicious_nodes or set()
        ids = sorted(agents.keys())

        # 1. Extract states
        _t0 = time.monotonic()
        current_states = {rid: agents[rid].get_actor_state(cpu_clone=True) for rid in ids}
        if self.cfg.use_fp16_comm:
            current_states = {rid: quantize_state_fp16(s) for rid, s in current_states.items()}
        sample_bytes = sum(v.numel() * v.element_size() for v in current_states[ids[0]].values())

        # 2. Pair exchange
        incoming: dict[int, list[IncomingCandidate]] = defaultdict(list)
        exchanges = 0
        local_bytes = 0
        exchanged_pairs: list[tuple[int, int]] = []
        use_layer_filter = self.cfg.layer_diff_threshold > 0
        malicious_active = bool(malicious_nodes) and step_idx >= attack_start_step

        for a, b in self._candidate_pairs(ids, positions):
            if not self._can_exchange(step_idx, a, b, positions):
                continue
            local_bytes += sample_bytes * 2

            if malicious_active:
                inc_a = self._apply_attack(step_idx=step_idx, sender_id=b, original_state=current_states[b],
                                           malicious_nodes=malicious_nodes, attack_type=attack_type, attack_start_step=attack_start_step)
                inc_b = self._apply_attack(step_idx=step_idx, sender_id=a, original_state=current_states[a],
                                           malicious_nodes=malicious_nodes, attack_type=attack_type, attack_start_step=attack_start_step)
            else:
                inc_a, inc_b = current_states[b], current_states[a]

            if use_layer_filter:
                fa = selective_layer_filter(current_states[a], inc_a, self.cfg.layer_diff_threshold)
                fb = selective_layer_filter(current_states[b], inc_b, self.cfg.layer_diff_threshold)
                inc_a = {**current_states[a], **fa}
                inc_b = {**current_states[b], **fb}

            dist_ab = euclidean_distance(positions[a], positions[b])
            incoming[a].append(IncomingCandidate(b, inc_a, malicious_active and b in malicious_nodes, dist_ab))
            incoming[b].append(IncomingCandidate(a, inc_b, malicious_active and a in malicious_nodes, dist_ab))
            exchanged_pairs.append((a, b))
            exchanges += 1

        # 3. Record exchange history
        with self._lock:
            for a, b in exchanged_pairs:
                self._last_exchange[(a, b)] = step_idx
            self.bytes_transferred += local_bytes

        # 4. Merge
        merged_states: dict[int, dict[str, torch.Tensor]] = {}
        for rid in ids:
            if rid not in incoming:
                continue
            merged = self._merge_incoming(
                local_state=current_states[rid], candidates=incoming[rid], progress=progress,
                defense_enabled=defense_enabled, defense_strategy=defense_strategy,
                defense_trim_ratio=defense_trim_ratio, defense_krum_malicious=defense_krum_malicious,
                calibration_steps=calibration_steps, in_calibration=(step_idx < calibration_steps),
            )
            if self.cfg.use_fp16_comm:
                merged = dequantize_state_fp16(merged)
            merged_states[rid] = merged

        _t1 = time.monotonic()
        _log.debug(
            "exchange step=%d: total=%.3fs n_exch=%d n_merged=%d",
            step_idx, _t1 - _t0, exchanges, len(merged_states),
        )
        return exchanges, merged_states

    # -- Internals -------------------------------------------------------

    def _candidate_pairs(self, ids: list[int], positions: dict[int, np.ndarray]) -> list[tuple[int, int]]:
        if not self.cfg.use_grid_index:
            return [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]
        cell = max(1e-6, float(self.cfg.grid_cell_size))
        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for rid in ids:
            px, py = float(positions[rid][0]), float(positions[rid][1])
            buckets[(int(np.floor(px / cell)), int(np.floor(py / cell)))].append(rid)
        neigh = (-1, 0, 1)
        pair_set: set[tuple[int, int]] = set()
        for (cx, cy), members in buckets.items():
            ncs = [(cx + dx, cy + dy) for dx in neigh for dy in neigh]
            for rid in members:
                for nc in ncs:
                    for other in buckets.get(nc, []):
                        if rid < other:
                            pair_set.add((rid, other))
        return sorted(pair_set)

    def _can_exchange(self, step_idx: int, a: int, b: int, positions: dict[int, np.ndarray]) -> bool:
        if euclidean_distance(positions[a], positions[b]) >= self.cfg.comm_radius:
            return False
        return (step_idx - self._last_exchange[(a, b)]) >= self.cfg.cooldown_steps

    def _merge_incoming(
        self, *, local_state: dict[str, torch.Tensor], candidates: list[IncomingCandidate],
        progress: float, defense_enabled: bool, defense_strategy: str,
        defense_trim_ratio: float, defense_krum_malicious: int,
        calibration_steps: int, in_calibration: bool,
    ) -> dict[str, torch.Tensor]:
        if not candidates:
            return local_state

        if not defense_enabled:
            inc_mean = aggregate_state_dicts_trimmed_mean([c.state for c in candidates], 0.0)
            self._update_stats(candidates, set(range(len(candidates))))
            return self._blend(local_state, inc_mean, progress)

        if defense_strategy == "cosine":
            return self._merge_cosine(local_state, candidates, progress, calibration_steps, in_calibration)
        if defense_strategy == "trimmed_mean":
            inc_mean = aggregate_state_dicts_trimmed_mean([c.state for c in candidates], defense_trim_ratio)
            self._update_stats(candidates, set(range(len(candidates))))
            return self._blend(local_state, inc_mean, progress)
        # krum
        return self._merge_krum(local_state, candidates, progress, defense_krum_malicious)

    def _merge_cosine(
        self, local_state: dict[str, torch.Tensor], candidates: list[IncomingCandidate],
        progress: float, calibration_steps: int, in_calibration: bool,
    ) -> dict[str, torch.Tensor]:
        threshold = self._threshold(calibration_steps)
        self.last_similarity_threshold = threshold
        accepted: list[IncomingCandidate] = []
        for c in candidates:
            sim = cosine_similarity_state_dict(local_state, c.state)
            if in_calibration and not c.sender_is_malicious:
                self._sim_stat.update(sim)
            should_reject = (not in_calibration) and threshold > -1.0 and sim < threshold
            if should_reject:
                self.defense_stats.rejected_updates += 1
                if c.sender_is_malicious:
                    self.defense_stats.rejected_malicious_updates += 1
                self.defense_stats.potential_malicious_nodes.add(c.sender_id)
                self.rejected_by_sender[c.sender_id] += 1
            else:
                accepted.append(c)
                self.defense_stats.accepted_updates += 1
                if c.sender_is_malicious:
                    self.defense_stats.accepted_malicious_updates += 1
        if not accepted:
            return local_state
        inc_mean = aggregate_state_dicts_trimmed_mean([c.state for c in accepted], 0.0)
        return self._blend(local_state, inc_mean, progress)

    def _merge_krum(
        self, local_state: dict[str, torch.Tensor], candidates: list[IncomingCandidate],
        progress: float, krum_malicious: int,
    ) -> dict[str, torch.Tensor]:
        pool = [local_state] + [c.state for c in candidates]
        vectors = [state_dict_to_vector(s) for s in pool]
        sel = krum_select_index(vectors, krum_malicious)
        if sel == 0:
            self._update_stats(candidates, set())
            return local_state
        ci = sel - 1
        self._update_stats(candidates, {ci})
        return self._blend(local_state, candidates[ci].state, progress)

    def _update_stats(self, candidates: list[IncomingCandidate], used: set[int]) -> None:
        for idx, c in enumerate(candidates):
            if idx in used:
                self.defense_stats.accepted_updates += 1
                if c.sender_is_malicious:
                    self.defense_stats.accepted_malicious_updates += 1
            else:
                self.defense_stats.rejected_updates += 1
                if c.sender_is_malicious:
                    self.defense_stats.rejected_malicious_updates += 1
                    self.defense_stats.potential_malicious_nodes.add(c.sender_id)
                    self.rejected_by_sender[c.sender_id] += 1

    def _threshold(self, calibration_steps: int) -> float:
        if self._sim_stat.n < 2 or calibration_steps <= 0:
            return -1.0
        return self._sim_stat.mean - 3.0 * self._sim_stat.std

    def _blend(
        self, local_state: dict[str, torch.Tensor], incoming_state: dict[str, torch.Tensor],
        progress: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Blend local and incoming states.

        beta = weight on LOCAL model.
        beta=0.5 means equal blend. beta=0.7 means 70% local, 30% incoming.
        """
        beta = self.cfg.beta
        if self.cfg.beta_schedule == "linear":
            # Start trusting peers more, then reduce over time
            beta = self.cfg.beta_min + (self.cfg.beta_max - self.cfg.beta_min) * progress
        elif self.cfg.beta_schedule == "exponential" and self.cfg.beta_max > 0:
            beta = self.cfg.beta_min * ((self.cfg.beta_max / self.cfg.beta_min) ** progress)

        # torch.lerp(a, b, t) = a + t * (b - a) = (1-t)*a + t*b
        # We want: result = beta * local + (1-beta) * incoming
        # = torch.lerp(incoming, local, beta)
        return {k: torch.lerp(incoming_state[k], local_state[k], beta) for k in local_state}

    def _apply_attack(
        self, *, step_idx: int, sender_id: int, original_state: dict[str, torch.Tensor],
        malicious_nodes: set[int], attack_type: str, attack_start_step: int,
    ) -> dict[str, torch.Tensor]:
        if sender_id not in malicious_nodes or step_idx < attack_start_step:
            return original_state
        if attack_type == "zero":
            return {k: torch.zeros_like(v) for k, v in original_state.items()}
        if attack_type == "gaussian":
            return {k: torch.randn_like(v) for k, v in original_state.items()}
        raise ValueError(f"Unsupported attack_type: {attack_type}")


# ---------------------------------------------------------------------------
# Centralised baseline
# ---------------------------------------------------------------------------

class CentralizedFedAvg:
    """Baseline: average all actor weights at fixed intervals."""

    def __init__(self, interval_steps: int, beta: float) -> None:
        self.interval_steps = interval_steps
        self.beta = beta
        self.bytes_transferred = 0
        self._lock = threading.Lock()

    def maybe_aggregate(self, step_idx: int, agents: dict[int, SACAgent]) -> tuple[int, dict[int, dict[str, torch.Tensor]]]:
        if step_idx % self.interval_steps != 0 or not agents:
            return 0, {}
        ids = sorted(agents.keys())
        states = {rid: agents[rid].get_actor_state(cpu_clone=True) for rid in ids}
        merged: dict[str, torch.Tensor] = {}
        for k in states[ids[0]]:
            merged[k] = torch.stack([states[rid][k] for rid in ids], dim=0).mean(dim=0)
        payload = sum(v.numel() * v.element_size() for v in merged.values())
        with self._lock:
            self.bytes_transferred += payload * len(ids) * 2
        # beta = weight on local, (1-beta) = weight on global average
        return 1, {rid: {k: torch.lerp(merged[k], states[rid][k], self.beta) for k in merged} for rid in ids}
