from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import deque
import numpy as np
from forecast import ForecastModel, EnsembleForecaster


Action = Dict[str, int]  # {dir: 0/1}


@dataclass
class RLPolicy:
    theta1: float = 0.6  # confidence threshold to use RL
    Q_norm: float = 50.0
    G_norm: float = 10.0
    _reward_hist: deque = None
    _last_total_q: float = 0.0

    def __post_init__(self):
        if self._reward_hist is None:
            self._reward_hist = deque(maxlen=40)

    def _proxy_reward(self, total_q: float, growth: float, accident: int) -> float:
        q_term = -float(total_q) / max(1e-6, self.Q_norm)
        g_term = -0.2 * max(0.0, float(growth)) / max(1e-6, self.G_norm)
        a_term = -2.0 * float(accident)
        return float(q_term + g_term + a_term)

    def act(self, obs: Dict) -> Tuple[Action, float]:
        # Heuristic action: choose axis with larger total
        queues = obs.get("queues", {d: 0.0 for d in ("N","S","E","W")})
        ns = queues.get("N",0.0) + queues.get("S",0.0)
        ew = queues.get("E",0.0) + queues.get("W",0.0)
        action = {"N": int(ns>=ew), "S": int(ns>=ew), "E": int(ew>ns), "W": int(ew>ns)}

        # Confidence: stability (low variance of proxy reward), improving trend, and manageable load
        growth = float(obs.get("cluster_growth", 0.0))
        accident = int(obs.get("accident", 0))
        total_q = float(ns + ew)
        r_t = self._proxy_reward(total_q, growth, accident)
        self._reward_hist.append(r_t)
        var = float(np.var(self._reward_hist)) if len(self._reward_hist) >= 5 else 0.0
        c_var = float(np.exp(-0.08 * var))  # higher when more stable

        dq = float(self._last_total_q - total_q)
        self._last_total_q = total_q
        trend_bonus = 0.15 if dq > 0 else (-0.05 if dq < 0 else 0.0)

        load = float(1.0 - np.tanh(total_q / (2.0 * self.Q_norm)))  # high when load is reasonable
        imbalance = abs(ns - ew)
        imbalance_penalty = float(0.2 * np.tanh(imbalance / 15.0))

        confidence = c_var * 0.6 + load * 0.3 + trend_bonus
        confidence = max(0.0, min(1.0, confidence - imbalance_penalty))
        return action, confidence


@dataclass
class GNNForecaster:
    theta2: float = 0.5  # confidence threshold to use GNN when RL is low
    horizon_s: float = 20.0
    model: ForecastModel = None
    ensemble: EnsembleForecaster = None
    # last inference details for logging
    last_conf: float = 0.0
    last_alpha: float = 0.0
    last_c_lstm: float = 0.0
    last_c_st: float = 0.0

    def __post_init__(self):
        if self.model is None:
            self.model = ForecastModel()
        if self.ensemble is None:
            self.ensemble = EnsembleForecaster()

    def predict(self, obs: Dict) -> Tuple[Action, float]:
        # Use ensemble forecaster on total congestion proxy
        queues = obs.get("queues", {d: 0.0 for d in ("N","S","E","W")})
        total = float(sum(queues.values()))
        self.model.update(total)
        self.ensemble.update(total)
        neighbor_influence = float(obs.get("density", 0.0))
        meta = {
            "neighbor_influence": neighbor_influence,
            "cluster_growth": float(obs.get("cluster_growth", 0.0)),
            "accident": float(obs.get("accident", 0.0)),
        }
        q_ens, c_ens, alpha, c_lstm, c_st = self.ensemble.predict(meta)
        self.last_conf = float(c_ens)
        self.last_alpha = float(alpha)
        self.last_c_lstm = float(c_lstm)
        self.last_c_st = float(c_st)
        ns = queues.get("N",0.0) + queues.get("S",0.0)
        ew = queues.get("E",0.0) + queues.get("W",0.0)
        # use ensemble trend: if q_ens > total, prioritize heavier axis else lighter
        trend = np.tanh((q_ens - total) / 10.0)
        ns_adj = ns * (1.0 + 0.3 * trend)
        ew_adj = ew * (1.0 - 0.3 * trend)
        action = {"N": int(ns_adj>=ew_adj), "S": int(ns_adj>=ew_adj), "E": int(ew_adj>ns_adj), "W": int(ew_adj>ns_adj)}
        return action, self.last_conf


def adaptive_logic(queues: Dict[str,float], cluster_priority_axis: str | None = None) -> Action:
    ns = queues.get("N",0.0) + queues.get("S",0.0)
    ew = queues.get("E",0.0) + queues.get("W",0.0)
    if cluster_priority_axis == "NS":
        ns += 0.25 * (ns + ew)
    elif cluster_priority_axis == "EW":
        ew += 0.25 * (ns + ew)
    return {"N": int(ns>=ew), "S": int(ns>=ew), "E": int(ew>ns), "W": int(ew>ns)}


def fixed_cycle(t_s: float, cycle_s: float = 60.0) -> Action:
    # 30s NS, 30s EW
    phase = (t_s % cycle_s) < (cycle_s/2)
    return {"N": int(phase), "S": int(phase), "E": int(not phase), "W": int(not phase)}


class HierarchicalController:
    def __init__(self, rl: RLPolicy, gnn: GNNForecaster, degrade_threshold: int):
        self.rl = rl
        self.gnn = gnn
        self.degrade_threshold = degrade_threshold
        self.prolonged_failure_since: float | None = None
        self.mode_since_s: float | None = None
        self.last_mode: str | None = None
        self.min_mode_duration_s: float = 40.0

    def select_action_with_modes(self, t_s: float, obs_per_node: List[Dict], accident_cluster_size: int) -> Tuple[List[Action], List[str]]:
        actions: List[Action] = []
        modes: List[str] = []
        for obs in obs_per_node:
            queues = obs.get("queues", {})
            growth = float(obs.get("cluster_growth", 0.0))
            # Determine cluster-priority axis
            ns = queues.get("N",0.0) + queues.get("S",0.0)
            ew = queues.get("E",0.0) + queues.get("W",0.0)
            cluster_axis = None
            if accident_cluster_size >= self.degrade_threshold:
                cluster_axis = "NS" if ns >= ew else "EW"

            a_rl, c_rl = self.rl.act(obs)
            if c_rl >= self.rl.theta1:
                actions.append(a_rl)
                modes.append("RL")
                continue

            # Enforce GNN every 20s cadence: only trust confidence at multiples of horizon, otherwise reduce
            a_gnn, c_gnn = self.gnn.predict(obs)
            if (t_s % self.gnn.horizon_s) > 1e-6:
                c_gnn *= 0.8
            if c_gnn >= self.gnn.theta2:
                actions.append(a_gnn)
                modes.append("GNN")
            else:
                actions.append(adaptive_logic(queues, cluster_priority_axis=cluster_axis))
                modes.append("ADAPT")
        # apply hysteresis: if switching too soon, keep last mode
        if modes:
            chosen = modes[0]
            if self.last_mode is None:
                self.last_mode = chosen
                self.mode_since_s = float(t_s)
            else:
                if chosen != self.last_mode and self.mode_since_s is not None and (t_s - self.mode_since_s) < self.min_mode_duration_s:
                    # override to last mode
                    modes[0] = self.last_mode
                else:
                    if chosen != self.last_mode:
                        self.last_mode = chosen
                        self.mode_since_s = float(t_s)
        return actions, modes

    def select_action_with_modes_and_conf(self, t_s: float, obs_per_node: List[Dict], accident_cluster_size: int) -> Tuple[List[Action], List[str], List[Dict[str, float]]]:
        actions: List[Action] = []
        modes: List[str] = []
        confs: List[Dict[str, float]] = []
        for obs in obs_per_node:
            queues = obs.get("queues", {})
            ns = queues.get("N",0.0) + queues.get("S",0.0)
            ew = queues.get("E",0.0) + queues.get("W",0.0)
            cluster_axis = None
            if accident_cluster_size >= self.degrade_threshold:
                cluster_axis = "NS" if ns >= ew else "EW"

            a_rl, c_rl = self.rl.act(obs)
            if c_rl >= self.rl.theta1:
                actions.append(a_rl)
                modes.append("RL")
                confs.append({"c_rl": float(c_rl), "c_ens": 0.0, "c_lstm": 0.0, "c_st": 0.0, "alpha": 0.0})
                continue

            a_gnn, c_gnn = self.gnn.predict(obs)
            c_eff = float(c_gnn)
            if (t_s % self.gnn.horizon_s) > 1e-6:
                c_eff *= 0.8
            if c_eff >= self.gnn.theta2:
                actions.append(a_gnn)
                modes.append("GNN")
                confs.append({"c_rl": float(c_rl), "c_ens": float(c_gnn), "c_lstm": float(self.gnn.last_c_lstm), "c_st": float(self.gnn.last_c_st), "alpha": float(self.gnn.last_alpha)})
            else:
                a = adaptive_logic(queues, cluster_priority_axis=cluster_axis)
                actions.append(a)
                modes.append("ADAPT")
                confs.append({"c_rl": float(c_rl), "c_ens": float(c_eff), "c_lstm": float(self.gnn.last_c_lstm), "c_st": float(self.gnn.last_c_st), "alpha": float(self.gnn.last_alpha)})
        # hysteresis on first node's mode
        if modes:
            chosen = modes[0]
            if self.last_mode is None:
                self.last_mode = chosen
                self.mode_since_s = float(t_s)
            else:
                if chosen != self.last_mode and self.mode_since_s is not None and (t_s - self.mode_since_s) < self.min_mode_duration_s:
                    modes[0] = self.last_mode
                else:
                    if chosen != self.last_mode:
                        self.last_mode = chosen
                        self.mode_since_s = float(t_s)
        return actions, modes, confs

    def select_action(self, t_s: float, obs_per_node: List[Dict], accident_cluster_size: int) -> List[Action]:
        actions, _ = self.select_action_with_modes(t_s, obs_per_node, accident_cluster_size)
        return actions


