from __future__ import annotations

import math
import random
from typing import Dict, Optional, Tuple, List


class AccidentManager:
    def __init__(self, mu_tc: float = 300.0, sigma_tc: float = 0.5, r_clear: float = 0.5, T_recovery: float = 60.0):
        self.mu_tc = float(mu_tc)
        self.sigma_tc = float(sigma_tc)
        self.r_clear = float(r_clear)
        self.T_recovery = float(T_recovery)
        # iid -> event state
        self.events: Dict[int, Dict[str, float]] = {}

    def trigger_accident(self, iid: int, t_s: float) -> Tuple[float, float]:
        T_clear = random.lognormvariate(math.log(max(1.0, self.mu_tc)), max(0.1, self.sigma_tc))
        self.events[iid] = {
            "start": float(t_s),
            "T_clear": float(T_clear),
            "acc": 1.0,
            "block": 1.0,
            "cleared": 0.0,
            "decay_end": float(t_s) + self.T_recovery,
        }
        print(f"[SIM] Accident triggered at {iid} t={t_s:.1f} T_clear={T_clear:.1f}")
        return float(T_clear), float(t_s)

    def get_acc_flag(self, iid: int) -> int:
        e = self.events.get(iid)
        return int(e["acc"]) if e else 0

    def get_blockage(self, iid: int) -> float:
        e = self.events.get(iid)
        return float(e["block"]) if e else 0.0

    def update(self, iid: int, t_s: float) -> Optional[Tuple[float, bool]]:
        e = self.events.get(iid)
        if not e:
            return None
        if e["acc"] == 1.0 and (t_s - e["start"]) >= e["T_clear"]:
            e["acc"] = 0.0
            e["cleared"] = 1.0
            e["decay_end"] = float(t_s) + self.T_recovery
            print(f"[SUMO] Accident cleared at {iid} (T_clear s={e['T_clear']:.1f})")
            return float(e["T_clear"]), True
        if e.get("cleared", 0.0) == 1.0:
            if t_s < e["decay_end"]:
                remaining = max(0.0, (e["decay_end"] - t_s) / max(1e-6, self.T_recovery))
                e["block"] = float(remaining)
            else:
                e["block"] = 0.0
        return None


