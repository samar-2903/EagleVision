from __future__ import annotations

from typing import Dict


class ServiceModel:
    def __init__(self, sat_flow: float = 0.7, dt: float = 1.0, kappa: float = 0.1, Q_cap: float = 50.0):
        self.sat_flow = float(sat_flow)
        self.dt = float(dt)
        self.kappa = float(kappa)
        self.Q_cap = float(Q_cap)

    def compute_mu(self, green_fraction: float, blockage: float) -> float:
        return max(0.0, self.sat_flow * max(0.0, min(1.0, green_fraction)) * (1.0 - max(0.0, min(1.0, blockage))))

    def compute_lambda(self, base_lambda: float, Q_upstream: float) -> float:
        spill = max(0.0, Q_upstream - self.Q_cap)
        return max(0.0, base_lambda + self.kappa * spill)

    def service_delta(self, mu: float) -> float:
        return max(0.0, mu * self.dt)


