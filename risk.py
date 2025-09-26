from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class RiskFeatures:
    weather_severity: float  # [0,1]
    congestion_growth_rate: float  # per second
    other: float = 0.0


def accident_probability(queue_sum: float, features: RiskFeatures) -> float:
    # Logistic model for accident risk
    x = 1.2 * (queue_sum / (50.0 + queue_sum)) + 0.8 * features.weather_severity + 0.6 * np.tanh(features.congestion_growth_rate)
    # Add 'other' as bias
    x += 0.2 * features.other
    return float(1.0 / (1.0 + np.exp(-4.0 * (x - 0.6))))


class AccidentMDP:
    def __init__(self, num_nodes: int, clearance_mean_min: float = 8.0, clearance_sigma_min: float = 0.6):
        # log-normal parameters (minutes) -> convert to seconds when sampling
        self.num_nodes = num_nodes
        self.clearance_mean_min = clearance_mean_min
        self.clearance_sigma_min = clearance_sigma_min
        self.active_accident: List[int] = [0] * num_nodes
        self.accident_start_time_s: List[float] = [-np.inf] * num_nodes
        self.accident_clear_until_s: List[float] = [0.0] * num_nodes
        self.last_clear_duration_s: List[float] = [0.0] * num_nodes

    def _sample_clearance_seconds(self) -> float:
        # Sample from log-normal with given mean and sigma in minutes, then convert to seconds
        # log-normal parameterization: mean_m = exp(mu + sigma^2/2)
        # choose sigma, derive mu to match mean
        sigma = max(0.1, float(self.clearance_sigma_min))
        mean_m = max(1.0, float(self.clearance_mean_min))
        mu = np.log(mean_m) - 0.5 * sigma * sigma
        sample_m = np.random.lognormal(mean=mu, sigma=sigma)
        return float(sample_m * 60.0)

    def step(self, t_s: float, queues_per_node: List[Dict[str, float]], features_per_node: List[RiskFeatures]) -> Tuple[List[int], List[Tuple[int, float]]]:
        accidents: List[int] = []
        cleared_events: List[Tuple[int, float]] = []  # (node_index, clearance_duration_s)
        for i in range(self.num_nodes):
            total_q = sum(max(0.0, v) for v in queues_per_node[i].values())
            p_acc = accident_probability(total_q, features_per_node[i])

            if self.active_accident[i] == 0:
                happened = np.random.rand() < p_acc
                if happened:
                    self.active_accident[i] = 1
                    self.accident_start_time_s[i] = t_s
                    self.accident_clear_until_s[i] = t_s + self._sample_clearance_seconds()
            else:
                if t_s >= self.accident_clear_until_s[i]:
                    self.active_accident[i] = 0
                    clear_dur = float(self.accident_clear_until_s[i] - self.accident_start_time_s[i])
                    self.last_clear_duration_s[i] = clear_dur
                    cleared_events.append((i, clear_dur))

            accidents.append(self.active_accident[i])
        return accidents, cleared_events

def compute_risk_score(lane_features):
    if not lane_features:
        return 0.0
    risk = 0.0
    for lf in lane_features.values():
        q = lf["queue_len"]
        d = lf["density"]
        g = lf["growth_rate"]
        v = lf["avg_speed"]
        lane_risk = 0.05 * q + 0.5 * d + 0.1 * max(0.0, g) + 0.05 * max(0.0, 2.0 - v)
        risk += lane_risk
    risk = risk / max(1, len(lane_features))
    return float(min(1.0, risk))


