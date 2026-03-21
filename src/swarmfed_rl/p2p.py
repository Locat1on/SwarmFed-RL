from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from .config import P2PConfig
from .sac import SACAgent


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


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
    a = torch.cat([v.reshape(-1).float() for v in state_a.values()], dim=0)
    b = torch.cat([v.reshape(-1).float() for v in state_b.values()], dim=0)
    denom = torch.norm(a, p=2) * torch.norm(b, p=2)
    if float(denom.item()) == 0.0:
        return 0.0
    return float(torch.dot(a, b).item() / denom.item())


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
    # For Krum to work, we need n >= 2*f + 3. 
    # If not enough neighbors, we can fallback to mean or just return "reject all" (0).
    # But returning 0 means "reject everything", which is why we saw 0 accepts.
    # In small swarms or sparse graphs, we should be lenient if n is small.
    if n <= 2:
        # Too few to vote, trust the first one (usually local model)
        return 0
    
    f = max(0, malicious_count)
    # Ensure we select at least 1 neighbor distance
    neighbor_count = max(1, n - f - 2)
    
    # If f is too large relative to n, Krum is theoretically impossible.
    # E.g. n=3, f=1 => n - f - 2 = 0.
    # We force neighbor_count to be at least 1 to avoid crash, but defense is weak.
    if neighbor_count < 1:
        neighbor_count = 1

    distances = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = torch.norm(vectors[i] - vectors[j], p=2)
            distances[i, j] = d
            distances[j, i] = d
    scores = []
    for i in range(n):
        others = torch.cat([distances[i, :i], distances[i, i + 1 :]], dim=0)
        nearest, _ = torch.sort(others)
        score = torch.sum(nearest[:neighbor_count])
        scores.append(float(score.item()))
    return int(np.argmin(scores))


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
        current_states = {rid: agents[rid].get_actor_state() for rid in ids}
        incoming: dict[int, list[IncomingCandidate]] = defaultdict(list)
        exchanges = 0

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if not self._can_exchange(step_idx, a, b, positions):
                    continue
                payload_bytes = self._estimate_payload_bytes(current_states[a])
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
                incoming[a].append(
                    IncomingCandidate(
                        sender_id=b,
                        state=incoming_to_a,
                        sender_is_malicious=(b in malicious_nodes and step_idx >= attack_start_step),
                    )
                )
                incoming[b].append(
                    IncomingCandidate(
                        sender_id=a,
                        state=incoming_to_b,
                        sender_is_malicious=(a in malicious_nodes and step_idx >= attack_start_step),
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
                defense_enabled=defense_enabled,
                defense_strategy=defense_strategy,
                defense_trim_ratio=defense_trim_ratio,
                defense_krum_malicious=defense_krum_malicious,
                calibration_steps=calibration_steps,
                in_calibration=(step_idx < calibration_steps),
            )
            agents[rid].load_actor_state(merged)
        return exchanges

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
            incoming_mean = aggregate_state_dicts_trimmed_mean(
                [c.state for c in candidates], trim_ratio=0.0
            )
            self._update_accept_reject_stats(candidates, set(range(len(candidates))))
            return self._blend(local_state, incoming_mean)

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
            incoming_mean = aggregate_state_dicts_trimmed_mean(
                [c.state for c in accepted], trim_ratio=0.0
            )
            return self._blend(local_state, incoming_mean)

        if defense_strategy == "trimmed_mean":
            incoming_states = [c.state for c in candidates]
            incoming_mean = aggregate_state_dicts_trimmed_mean(
                incoming_states, trim_ratio=defense_trim_ratio
            )
            # Trimmed-mean is element-wise; a sender may be partly used and partly trimmed.
            # Record all senders as accepted at sender-level to avoid misleading "fully rejected" stats.
            self._update_accept_reject_stats(candidates, set(range(len(candidates))))
            return self._blend(local_state, incoming_mean)

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
        return self._blend(local_state, incoming_state)

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
    ) -> dict[str, torch.Tensor]:
        beta = self.cfg.beta
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
