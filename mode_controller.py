from __future__ import annotations

from typing import List


class ModeController:
    def __init__(self, th_RL: float = 0.6, th_GNN: float = 0.5, min_mode_dur: float = 40.0):
        self.mode = "RL"
        self.last_switch_t = 0.0
        self.th_RL = float(th_RL)
        self.th_GNN = float(th_GNN)
        self.min_mode_dur = float(min_mode_dur)

    def decide_mode(self, t_s: float, RL_conf: float, GNN_conf: float) -> str:
        if (t_s - self.last_switch_t) < self.min_mode_dur:
            return self.mode
        prev = self.mode
        if RL_conf >= self.th_RL:
            self.mode = "RL"
        elif GNN_conf >= self.th_GNN:
            self.mode = "GNN"
        else:
            self.mode = "ADAPTIVE"
        if self.mode != prev:
            self.last_switch_t = t_s
            print(f"[SIM] mode={self.mode} at t={t_s:.1f} (RL_conf={RL_conf:.2f} GNN_conf={GNN_conf:.2f})")
        return self.mode


