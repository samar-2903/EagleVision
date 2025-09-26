from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np


Direction = str  # one of "N","S","E","W"


@dataclass
class IntersectionState:
    queues: Dict[Direction, float]  # Q_i_dir(t)
    density: float  # rho_i(t)
    accident_flag: int  # A_i(t) in {0,1}


@dataclass
class Intersection:
    intersection_id: int
    service_rates_green: Dict[Direction, float]
    service_rates_red: Dict[Direction, float]
    saturation_queue: float = 1e3
    state: IntersectionState = field(default_factory=lambda: IntersectionState({"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0}, 0.0, 0))

    def service_rate(self, direction: Direction, action: Dict[Direction, int]) -> float:
        is_green = action.get(direction, 0) == 1
        return self.service_rates_green.get(direction, 0.0) if is_green else self.service_rates_red.get(direction, 0.0)

    def step_ode(self, dt: float, arrival_rates: Dict[Direction, float], action: Dict[Direction, int]) -> None:
        # dQ_i_dir/dt = lambda_i_dir(t) - mu_i_dir(a_t)
        for direction in ("N", "S", "E", "W"):
            lam = max(0.0, float(arrival_rates.get(direction, 0.0)))
            mu = max(0.0, self.service_rate(direction, action))
            dq = (lam - mu) * dt
            new_q = max(0.0, min(self.saturation_queue, self.state.queues.get(direction, 0.0) + dq))
            self.state.queues[direction] = new_q

    def step_discrete(self, dt: float, arrival_rates: Dict[Direction, float], action: Dict[Direction, int], blockage: float = 0.0, sat_flow: float | None = None, cycle_len: float = 60.0, green_alloc_s: float | None = None) -> None:
        # Q(t+dt) = Q(t) + (lambda - served) * dt
        # served = min(mu*dt, Q(t)); mu = s_id * (g/C) * (1 - b)
        s_id = float(sat_flow) if sat_flow is not None else 0.7
        g_ns = cycle_len / 2.0
        g_ew = cycle_len / 2.0
        if green_alloc_s is not None:
            g_ns = float(green_alloc_s)
            g_ew = float(max(0.0, cycle_len - g_ns))
        for direction in ("N", "S", "E", "W"):
            lam = max(0.0, float(arrival_rates.get(direction, 0.0)))
            is_ns = direction in ("N", "S")
            g = g_ns if is_ns else g_ew
            mu = max(0.0, s_id * (g / max(1e-6, cycle_len)) * (1.0 - max(0.0, min(1.0, blockage))))
            q_now = max(0.0, self.state.queues.get(direction, 0.0))
            served = min(mu * dt, q_now)
            dq = (lam * dt) - served
            self.state.queues[direction] = max(0.0, min(self.saturation_queue, q_now + dq))


class TrafficNetwork:
    def __init__(self, intersections: List[Intersection], connectivity: np.ndarray):
        # connectivity C_ij (nonnegative weights). Shape: num_intersections x num_intersections
        self.intersections = intersections
        self.C = np.array(connectivity, dtype=float)
        self.num_nodes = len(intersections)
        assert self.C.shape == (self.num_nodes, self.num_nodes), "Connectivity matrix shape mismatch"

        # base exogenous arrival rates per node and direction (can be updated externally)
        self.base_arrival: List[Dict[Direction, float]] = [
            {"N": 0.2, "S": 0.2, "E": 0.2, "W": 0.2} for _ in range(self.num_nodes)
        ]

        # history for simple spillover heuristic
        self.queue_history: List[List[float]] = [[] for _ in range(self.num_nodes)]

    def compute_arrival_rates_with_spillover(self, accident_flags: Optional[List[int]] = None, q_threshold: float = 150.0, kappa_accident: float = 0.3) -> List[Dict[Direction, float]]:
        # lambda_j_dir(t) = base + f(Q_i_dir(t), C_ij)
        arrivals: List[Dict[Direction, float]] = []
        upstream_queue = np.array([
            sum(max(0.0, self.intersections[i].state.queues[d]) for d in ("N", "S", "E", "W"))
            for i in range(self.num_nodes)
        ], dtype=float)

        # Normalize upstream queues to [0,1] by a soft cap
        norm_upstream = np.tanh(upstream_queue / (1.0 + np.mean(upstream_queue) + 1e-6))

        # Spillover contribution matrix S = C^T * norm_upstream
        spill = self.C.T @ norm_upstream
        # Accident-aware extra spillover when upstream exceeds threshold or accident active
        if accident_flags is None:
            accident_flags = [0] * self.num_nodes
        overload = (upstream_queue > q_threshold).astype(float) + np.array(accident_flags, dtype=float)
        if np.any(overload > 0):
            extra = self.C.T @ overload
            spill = spill + kappa_accident * extra

        for j in range(self.num_nodes):
            base = self.base_arrival[j]
            spill_term = float(spill[j])
            # Distribute spillover across directions proportionally
            arrivals.append({
                d: max(0.0, base.get(d, 0.0) + 0.25 * spill_term) for d in ("N", "S", "E", "W")
            })
        return arrivals

    def step(self, dt: float, actions: List[Dict[Direction, int]]) -> None:
        arrivals = self.compute_arrival_rates_with_spillover()
        for i, inter in enumerate(self.intersections):
            inter.step_ode(dt=dt, arrival_rates=arrivals[i], action=actions[i])

        # update history
        for i, inter in enumerate(self.intersections):
            total_q = sum(inter.state.queues.values())
            self.queue_history[i].append(total_q)

    def get_state_snapshot(self) -> List[IntersectionState]:
        return [inter.state for inter in self.intersections]


