from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from .config import P2PConfig
from .sac import SACAgent


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def quantize_state_fp16(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Quantize actor weights to FP16 for transmission."""
    return {k: v.half() for k, v in state.items()}


def dequantize_state_fp16(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Dequantize received FP16 weights back to FP32."""
    return {k: v.float() for k, v in state.items()}


def selective_layer_filter(
    local: dict[str, torch.Tensor],
    incoming: dict[str, torch.Tensor],
    threshold: float = 0.001,
) -> dict[str, torch.Tensor]:
    """
    Filter out layers where change is negligible.
    Returns a filtered incoming state_dict with only changed layers.
    """
    filtered = {}
    for k in local:
        if k not in incoming:
            continue
        layer_diff_std = float(torch.std(incoming[k] - local[k]).item())
        if layer_diff_std >= threshold:
            filtered[k] = incoming[k]
    return filtered


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
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return float((self.m2 / (self.n - 1)) ** 0.5)


def cosine_similarity_state_dict(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
) -> float:
    dot_product = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0
    for k in state_a:
        a_flat = state_a[k].float().view(-1)
        b_flat = state_b[k].float().view(-1)
        dot_product += torch.dot(a_flat, b_flat).item()
        norm_a_sq += torch.sum(a_flat ** 2).item()
        norm_b_sq += torch.sum(b_flat ** 2).item()
        
    denom = (norm_a_sq ** 0.5) * (norm_b_sq ** 0.5)
    if denom == 0.0:
        return 0.0
    return dot_product / denom


def state_dict_to_vector(state: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.reshape(-1).float() for v in state.values()], dim=0)


def aggregate_state_dicts_trimmed_mean(
    states: list[dict[str, torch.Tensor]],
    trim_ratio: float,
) -> dict[str, torch.Tensor]:
    if len(states) == 0:
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
            trimmed = sorted_vals[trim_k : len(states) - trim_k]
            out[k] = trimmed.mean(dim=0)
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
        # If there's only local model and 1 neighbor, and no malicious nodes expected,
        # we can blindly trust the neighbor to allow learning. Otherwise, trust local.
        return 1 if malicious_count == 0 else 0
    
    f = max(0, malicious_count)
    # Ensure we select at least 1 neighbor distance
    neighbor_count = max(1, n - f - 2)
    
    # If f is too large relative to n, Krum is theoretically impossible.
    # E.g. n=3, f=1 => n - f - 2 = 0.
    # We force neighbor_count to be at least 1 to avoid crash, but defense is weak.
    if neighbor_count < 1:
        neighbor_count = 1

    stacked = torch.stack(vectors)  # (n, d)
    distances = torch.cdist(stacked.unsqueeze(0), stacked.unsqueeze(0)).squeeze(0)
    # Mask diagonal (self-distance = 0) with inf so it's excluded from topk
    distances.fill_diagonal_(float("inf"))
    # Use topk to find nearest neighbors (faster than full sort)
    k = min(neighbor_count, n - 1)
    nearest_dists, _ = torch.topk(distances, k, dim=1, largest=False)
    scores = nearest_dists.sum(dim=1)
    return int(torch.argmin(scores).item())


class P2PAggregator:
    """
    Event-triggered actor-only aggregation with optional robust defenses.
    """

    def __init__(self, cfg: P2PConfig) -> None:
        self.cfg = cfg
        self._last_exchange: dict[tuple[int, int], int] = defaultdict(lambda: -10**9)
        self.bytes_transferred = 0
        self.defense_stats = DefenseStats()
        self._sim_stat = RunningStat()
        self.rejected_by_sender: dict[int, int] = defaultdict(int)
        self.last_similarity_threshold = -1.0

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
    ) -> int:
        if step_idx % self.cfg.exchange_interval_steps != 0:
            return 0
        if defense_strategy not in {"cosine", "trimmed_mean", "krum"}:
            raise ValueError(f"Unsupported defense strategy: {defense_strategy}")

        malicious_nodes = malicious_nodes or set()
        ids = sorted(agents.keys())
        current_states = {rid: agents[rid].get_actor_state(cpu_clone=False) for rid in ids}
        
        # FP16 quantization for communication
        if self.cfg.use_fp16_comm:
            current_states = {rid: quantize_state_fp16(s) for rid, s in current_states.items()}
            payload_bytes_cache = {rid: self._estimate_payload_bytes(current_states[rid]) for rid in ids}
        else:
            payload_bytes_cache = {rid: self._estimate_payload_bytes(current_states[rid]) for rid in ids}
        
        incoming: dict[int, list[IncomingCandidate]] = defaultdict(list)
        exchanges = 0

        for a, b in self._candidate_pairs(ids, positions):
                if not self._can_exchange(step_idx, a, b, positions):
                    continue
                payload_bytes = payload_bytes_cache[a]
                # Actual bytes after FP16 is already reflected in payload_bytes_cache
                self.bytes_transferred += payload_bytes * 2

                incoming_to_a = self._apply_attack(
                    step_idx=step_idx,
                    sender_id=b,
                    original_state=current_states[b],
                    malicious_nodes=malicious_nodes,
                    attack_type=attack_type,
                    attack_start_step=attack_start_step,
                )
                incoming_to_b = self._apply_attack(
                    step_idx=step_idx,
                    sender_id=a,
                    original_state=current_states[a],
                    malicious_nodes=malicious_nodes,
                    attack_type=attack_type,
                    attack_start_step=attack_start_step,
                )
                
                # Selective layer filtering if enabled
                if self.cfg.layer_diff_threshold > 0:
                    local_a_fp16 = current_states[a]
                    local_b_fp16 = current_states[b]
                    incoming_to_a_filtered = selective_layer_filter(local_a_fp16, incoming_to_a, self.cfg.layer_diff_threshold)
                    incoming_to_b_filtered = selective_layer_filter(local_b_fp16, incoming_to_b, self.cfg.layer_diff_threshold)
                    # Update local state with filtered incoming
                    incoming_to_a = {**local_a_fp16, **incoming_to_a_filtered}
                    incoming_to_b = {**local_b_fp16, **incoming_to_b_filtered}
                
                # Calculate distance for dynamic beta
                dist_ab = euclidean_distance(positions[a], positions[b])
                
                incoming[a].append(
                    IncomingCandidate(
                        sender_id=b,
                        state=incoming_to_a,
                        sender_is_malicious=(b in malicious_nodes and step_idx >= attack_start_step),
                        distance=dist_ab,
                    )
                )
                incoming[b].append(
                    IncomingCandidate(
                        sender_id=a,
                        state=incoming_to_b,
                        sender_is_malicious=(a in malicious_nodes and step_idx >= attack_start_step),
                        distance=dist_ab,
                    )
                )
                self._last_exchange[(a, b)] = step_idx
                exchanges += 1

        for rid in ids:
            if rid not in incoming:
                continue
            merged = self._merge_incoming(
                local_state=current_states[rid],
                candidates=incoming[rid],
                progress=progress,
                defense_enabled=defense_enabled,
                defense_strategy=defense_strategy,
                defense_trim_ratio=defense_trim_ratio,
                defense_krum_malicious=defense_krum_malicious,
                calibration_steps=calibration_steps,
                in_calibration=(step_idx < calibration_steps),
            )
            # Dequantize if using FP16 comm
            if self.cfg.use_fp16_comm:
                merged = dequantize_state_fp16(merged)
            agents[rid].load_actor_state(merged)
        return exchanges

    def _candidate_pairs(
        self,
        ids: list[int],
        positions: dict[int, np.ndarray],
    ) -> list[tuple[int, int]]:
        if not self.cfg.use_grid_index:
            return [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]
        cell = max(1e-6, float(self.cfg.grid_cell_size))
        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for rid in ids:
            px, py = float(positions[rid][0]), float(positions[rid][1])
            key = (int(np.floor(px / cell)), int(np.floor(py / cell)))
            buckets[key].append(rid)
        neigh = (-1, 0, 1)
        pair_set: set[tuple[int, int]] = set()
        for (cx, cy), members in buckets.items():
            local_cells = [(cx + dx, cy + dy) for dx in neigh for dy in neigh]
            for rid in members:
                for nc in local_cells:
                    for other in buckets.get(nc, []):
                        if rid >= other:
                            continue
                        pair_set.add((rid, other))
        return sorted(pair_set)

    def _estimate_payload_bytes(self, state: dict[str, torch.Tensor]) -> int:
        threshold = max(0.0, float(self.cfg.weight_std_threshold))
        estimated = 0
        for k, v in state.items():
            if float(torch.std(v).item()) > threshold:
                estimated += v.numel() * v.element_size()
        if estimated == 0:
            return sum(v.numel() * v.element_size() for v in state.values())
        return estimated

    def _can_exchange(
        self,
        step_idx: int,
        a: int,
        b: int,
        positions: dict[int, np.ndarray],
    ) -> bool:
        if euclidean_distance(positions[a], positions[b]) >= self.cfg.comm_radius:
            return False
        last = self._last_exchange[(a, b)]
        return (step_idx - last) >= self.cfg.cooldown_steps

    def _merge_incoming(
        self,
        *,
        local_state: dict[str, torch.Tensor],
        candidates: list[IncomingCandidate],
        progress: float,
        defense_enabled: bool,
        defense_strategy: str,
        defense_trim_ratio: float,
        defense_krum_malicious: int,
        calibration_steps: int,
        in_calibration: bool,
    ) -> dict[str, torch.Tensor]:
        if len(candidates) == 0:
            return local_state

        if not defense_enabled:
            # Average distance of all candidates for dynamic beta
            avg_dist = float(np.mean([c.distance for c in candidates]))
            incoming_mean = aggregate_state_dicts_trimmed_mean(
                [c.state for c in candidates], trim_ratio=0.0
            )
            self._update_accept_reject_stats(candidates, set(range(len(candidates))))
            return self._blend(local_state, incoming_mean, distance=avg_dist, progress=progress)

        if defense_strategy == "cosine":
            accepted: list[IncomingCandidate] = []
            threshold = self._threshold(calibration_steps=calibration_steps)
            self.last_similarity_threshold = threshold
            for c in candidates:
                similarity = cosine_similarity_state_dict(local_state, c.state)
                if in_calibration and (not c.sender_is_malicious):
                    self._sim_stat.update(similarity)
                should_reject = (
                    (not in_calibration)
                    and threshold > -1.0
                    and similarity < threshold
                )
                if should_reject:
                    self.defense_stats.rejected_updates += 1
                    if c.sender_is_malicious:
                        self.defense_stats.rejected_malicious_updates += 1
                    self.defense_stats.potential_malicious_nodes.add(c.sender_id)
                    self.rejected_by_sender[c.sender_id] += 1
                    continue
                accepted.append(c)
                self.defense_stats.accepted_updates += 1
                if c.sender_is_malicious:
                    self.defense_stats.accepted_malicious_updates += 1
            if len(accepted) == 0:
                return local_state
            
            avg_dist = float(np.mean([c.distance for c in accepted]))
            incoming_mean = aggregate_state_dicts_trimmed_mean(
                [c.state for c in accepted], trim_ratio=0.0
            )
            return self._blend(local_state, incoming_mean, distance=avg_dist, progress=progress)

        if defense_strategy == "trimmed_mean":
            avg_dist = float(np.mean([c.distance for c in candidates]))
            incoming_states = [c.state for c in candidates]
            incoming_mean = aggregate_state_dicts_trimmed_mean(
                incoming_states, trim_ratio=defense_trim_ratio
            )
            # Trimmed-mean is element-wise; a sender may be partly used and partly trimmed.
            # Record all senders as accepted at sender-level to avoid misleading "fully rejected" stats.
            self._update_accept_reject_stats(candidates, set(range(len(candidates))))
            return self._blend(local_state, incoming_mean, distance=avg_dist, progress=progress)

        # Krum: include local model as a candidate anchor for robust selection.
        krum_pool = [local_state] + [c.state for c in candidates]
        vectors = []
        for s in krum_pool:
            v = state_dict_to_vector(s)
            vectors.append(v)
            
        # Select the index of the "best" model in the pool (0 is local, 1..k are candidates)
        sel_idx = krum_select_index(vectors, malicious_count=defense_krum_malicious)
        
        # If sel_idx is 0, it means Krum chose our OWN local model as the best center.
        # This implies all external candidates were considered "worse" or "outliers".
        # So we reject all candidates and keep local state.
        if sel_idx == 0:
            self._update_accept_reject_stats(candidates, used_indices=set())
            return local_state
            
        # If sel_idx > 0, Krum chose candidate[sel_idx - 1]
        chosen_candidate_idx = sel_idx - 1
        used = {chosen_candidate_idx}
        self._update_accept_reject_stats(candidates, used)
        
        # Blend with the chosen candidate
        incoming_state = candidates[chosen_candidate_idx].state
        dist = candidates[chosen_candidate_idx].distance
        return self._blend(local_state, incoming_state, distance=dist, progress=progress)

    def _update_accept_reject_stats(self, candidates: list[IncomingCandidate], used_indices: set[int]) -> None:
        for idx, c in enumerate(candidates):
            if idx in used_indices:
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
        self,
        local_state: dict[str, torch.Tensor],
        incoming_state: dict[str, torch.Tensor],
        distance: float | None = None,
        progress: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        # 1. Base beta from schedule or default
        beta = self.cfg.beta
        if self.cfg.beta_schedule == "linear":
            beta = self.cfg.beta_max - (self.cfg.beta_max - self.cfg.beta_min) * progress
        elif self.cfg.beta_schedule == "exponential":
            if self.cfg.beta_max > 0:
                beta = self.cfg.beta_max * ((self.cfg.beta_min / self.cfg.beta_max) ** progress)
        
        # 2. Refine beta based on distance if enabled
        # If dynamic_beta is ON, distance-based logic scales the baseline beta
        if self.cfg.dynamic_beta and distance is not None:
            normalized_dist = min(distance / self.cfg.comm_radius, 1.0)
            # Use current scheduled beta as the 'local trust' baseline for nearby robots
            # and scale down to beta_min for far-away robots.
            b_high = max(beta, self.cfg.beta_max)
            b_low = self.cfg.beta_min
            beta = b_high - (b_high - b_low) * normalized_dist
        
        out: dict[str, torch.Tensor] = {}
        for k in local_state:
            out[k] = beta * local_state[k] + (1.0 - beta) * incoming_state[k]
        return out

    def _apply_attack(
        self,
        *,
        step_idx: int,
        sender_id: int,
        original_state: dict[str, torch.Tensor],
        malicious_nodes: set[int],
        attack_type: str,
        attack_start_step: int,
    ) -> dict[str, torch.Tensor]:
        if sender_id not in malicious_nodes or step_idx < attack_start_step:
            return original_state
        if attack_type not in {"zero", "gaussian"}:
            raise ValueError(f"Unsupported attack_type: {attack_type}")
        attacked: dict[str, torch.Tensor] = {}
        for k, v in original_state.items():
            if attack_type == "zero":
                attacked[k] = torch.zeros_like(v)
            else:
                attacked[k] = torch.randn_like(v)
        return attacked


class CentralizedFedAvg:
    """
    Baseline aggregator that averages all actor weights at fixed intervals.
    """

    def __init__(self, interval_steps: int, beta: float) -> None:
        self.interval_steps = interval_steps
        self.beta = beta
        self.bytes_transferred = 0

    def maybe_aggregate(self, step_idx: int, agents: dict[int, SACAgent]) -> int:
        if step_idx % self.interval_steps != 0:
            return 0
        if not agents:
            return 0

        ids = sorted(agents.keys())
        states = {rid: agents[rid].get_actor_state() for rid in ids}
        merged: dict[str, torch.Tensor] = {}

        for k in states[ids[0]]:
            stacked = torch.stack([states[rid][k] for rid in ids], dim=0)
            merged[k] = stacked.mean(dim=0)

        payload_one_way = 0
        for k in merged:
            payload_one_way += merged[k].numel() * merged[k].element_size()
        self.bytes_transferred += payload_one_way * (len(ids) * 2)

        for rid in ids:
            blended = self._blend(agents[rid].get_actor_state(), merged)
            agents[rid].load_actor_state(blended)
        return 1

    def _blend(
        self,
        local_state: dict[str, torch.Tensor],
        incoming_state: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for k in local_state:
            out[k] = self.beta * local_state[k] + (1.0 - self.beta) * incoming_state[k]
        return out
