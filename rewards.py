from __future__ import annotations

from typing import List


def compute_reward(total_queues: List[float], avg_delays: List[float], accidents: List[int], cluster_growths: List[float], alpha: float = 1.0, beta: float = 0.1, gamma: float = 5.0, delta: float = 0.2, Q_norm: float = 50.0, D_norm: float = 100.0, G_norm: float = 10.0) -> float:
    # r_t = -alpha * mean(Q_i/Q_norm) - beta * mean(Delay_i/D_norm) - gamma * mean(A_i) - delta * mean(max(0,g_Cj)/G_norm)
    N = max(1, len(total_queues))
    M = max(1, len(cluster_growths))
    term_q = sum(q / max(1e-6, Q_norm) for q in total_queues) / N
    term_d = sum(d / max(1e-6, D_norm) for d in avg_delays) / N
    term_a = sum(float(a) for a in accidents) / N
    term_g = sum(max(0.0, float(g)) / max(1e-6, G_norm) for g in cluster_growths) / M
    r = -alpha * term_q - beta * term_d - gamma * term_a - delta * term_g
    # clip to [-1000, 0]
    return float(max(-1000.0, min(0.0, r)))


