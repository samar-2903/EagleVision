# =============================================================================
# reward.py — Computes how good (or bad) the current traffic state is.
#
# WHY IS THE REWARD FUNCTION SO IMPORTANT?
# The reward IS the learning signal. If the reward is poorly designed:
#   - Too sparse (rare non-zero rewards): agent learns nothing for thousands of steps
#   - Too dense but wrong: agent learns to game the metric, not solve the problem
#   - Badly scaled: one term dominates and others are ignored
#
# Our reward has 4 terms, all normalized to roughly [-1, 0] range each,
# then weighted to reflect priority: safety > queues > delays > growth
# =============================================================================

from __future__ import annotations
from typing import List
import numpy as np
import config as cfg


def compute_reward(
    total_queues:    List[float],   # total vehicles queued per intersection
    avg_delays:      List[float],   # average delay (seconds) per intersection
    accidents:       List[int],     # accident flag (0 or 1) per intersection
    cluster_growths: List[float],   # cluster growth rate per intersection
) -> float:
    """
    r_t = -α * mean(Q_i / Q_norm)          ← penalize queue buildup
          -β * mean(Delay_i / D_norm)       ← penalize waiting time
          -γ * mean(A_i)                    ← heavily penalize accidents
          -δ * mean(max(0, g_Cj) / G_norm)  ← penalize growing clusters

    All terms are negative (we're minimizing badness).
    Clipped to [-1000, 0] to prevent extreme outliers from destabilizing training.

    WHY normalize each term?
    Without normalization, if queues go to 200 vehicles they dominate the
    reward and the agent ignores delays and accidents entirely.
    Dividing by Q_norm puts queues in [0, ~4] range, delays in [0, ~2], etc.
    Then weights α, β, γ, δ set the relative importance.
    """
    N = max(1, len(total_queues))
    M = max(1, len(cluster_growths))

    # Term 1: Queue length penalty
    # max(1e-6, ...) prevents division by zero
    term_q = sum(q / max(1e-6, cfg.Q_NORM) for q in total_queues) / N

    # Term 2: Delay penalty
    term_d = sum(d / max(1e-6, cfg.D_NORM) for d in avg_delays) / N

    # Term 3: Accident penalty
    # A_i ∈ {0, 1} so this term is either 0 or -γ
    # γ=5.0 means one accident is as bad as having 5x Q_norm queued vehicles
    term_a = sum(float(a) for a in accidents) / N

    # Term 4: Cluster growth penalty
    # Only penalize GROWING clusters (max with 0), not shrinking ones
    term_g = sum(max(0.0, float(g)) / max(1e-6, cfg.G_NORM) for g in cluster_growths) / M

    r = (
        -cfg.ALPHA    * term_q
        -cfg.BETA     * term_d
        -cfg.GAMMA_ACC * term_a
        -cfg.DELTA    * term_g
    )

    # Clip to prevent extreme values from causing huge gradient updates
    return float(max(-1000.0, min(0.0, r)))


def compute_delay_from_queue(queue: float, avg_service_rate: float = 0.35) -> float:
    """
    Estimate average delay using Little's Law: D ≈ Q / μ
    Little's Law: mean number in system = arrival rate × mean time in system
    Rearranged: mean time (delay) = queue / service_rate
    WHY: We don't always have direct delay measurements from SUMO, but we
    always have queue length. This gives a principled estimate.
    """
    return float(queue / max(1e-3, avg_service_rate))
