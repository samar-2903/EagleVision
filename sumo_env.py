from __future__ import annotations

import random
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np

import config as cfg
from accident_manager import AccidentManager
from clustering import ClusterFeatures, OpticsClustering
from reward import compute_delay_from_queue, compute_reward
from risk import RiskFeatures, accident_probability

try:
    import traci
except ImportError as exc:
    raise ImportError(
        "TraCI not found. Make sure SUMO is installed and SUMO_HOME is configured."
    ) from exc


class SumoEnv:
    """
    Multi-intersection SUMO environment with a shared-policy control interface.

    One DQN policy is shared across all controlled traffic lights. Each traffic
    light receives its own local state and produces its own discrete action.
    """

    DIRECTIONS = ("N", "S", "E", "W")

    def __init__(
        self,
        use_gui: bool = False,
        traffic_scale: float = 1.0,
        seed: int = cfg.SEED,
    ):
        self.use_gui = use_gui
        self.traffic_scale = float(traffic_scale)
        self.seed = int(seed)
        self.sumo_binary = "sumo-gui" if use_gui else cfg.SUMO_BINARY

        self.step_count = 0
        self.controlled_tls: List[str] = []
        self.current_phase: Dict[str, int] = {}
        self._num_phases: Dict[str, int] = {}
        self._tls_incoming_lanes: Dict[str, Dict[str, List[str]]] = {}
        self._lane_to_tls_dir: Dict[str, Tuple[str, str]] = {}
        self._arrival_history: Dict[str, Dict[str, deque]] = {}
        self._prev_vehicle_ids: set[str] = set()
        self._lane_speed_backup: Dict[str, float] = {}

        self.optics: Dict[str, OpticsClustering] = {}
        self.acc_mgr = AccidentManager()
        self._accident_lane: Dict[str, str] = {}

        self._jam_active = False
        self._demand_keep_prob = 1.0
        self._last_global_queue = 0.0
        self._removed_vehicle_count = 0

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, np.ndarray]:
        try:
            traci.close()
        except Exception:
            pass

        random.seed(self.seed)
        np.random.seed(self.seed)

        sumo_cmd = [
            self.sumo_binary,
            "-c", cfg.SUMO_CFG,
            "--no-step-log",
            "--waiting-time-memory", "100",
            "--seed", str(self.seed),
            "--scale", str(self.traffic_scale),
        ]
        if self.use_gui:
            sumo_cmd.extend(["--start", "--quit-on-end"])
        traci.start(sumo_cmd)

        self.step_count = 0
        self._prev_vehicle_ids = set()
        self._lane_speed_backup = {}
        self._accident_lane = {}
        self.optics = {}
        self.acc_mgr = AccidentManager()

        self._jam_active = False
        self._demand_keep_prob = 1.0
        self._last_global_queue = 0.0
        self._removed_vehicle_count = 0

        self._discover_controlled_tls()

        traci.simulationStep()
        self.step_count = 1
        self._update_arrival_history()

        observations = self._collect_observations()
        self._refresh_current_phases()
        return self._build_states(observations)

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(
        self, actions: Dict[str, int] | int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict]:
        return self._step_common(actions, duration_mode="queue_scaled")

    def step_fixed_time(
        self,
        actions: Dict[str, int] | int,
        fixed_duration: int = cfg.DEFAULT_GREEN,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict]:
        return self._step_common(
            actions,
            duration_mode="fixed",
            fixed_duration=fixed_duration,
        )

    def _step_common(
        self,
        actions: Dict[str, int] | int,
        duration_mode: str,
        fixed_duration: int = cfg.DEFAULT_GREEN,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict]:
        action_map = self._normalize_action_map(actions)

        current_obs = self._collect_observations()
        for tid in self.controlled_tls:
            self._apply_action_to_tls(
                tid,
                action_map[tid],
                current_obs[tid]["queues"],
                duration_mode=duration_mode,
                fixed_duration=fixed_duration,
            )

        for _ in range(cfg.CONTROL_INTERVAL):
            if traci.simulation.getMinExpectedNumber() <= 0 and self.step_count > 10:
                break
            traci.simulationStep()
            self.step_count += 1
            self._update_arrival_history()

        next_obs = self._collect_observations()
        accident_events, cleared_events = self._update_accidents(next_obs)
        self._refresh_current_phases()

        global_queue = float(sum(obs["total_queue"] for obs in next_obs.values()))
        jam_flags = self._update_jam_control(global_queue)
        states = self._build_states(next_obs)

        rewards = {}
        for tid, obs in next_obs.items():
            local_reward = compute_reward(
                [obs["total_queue"]],
                [obs["avg_delay"]],
                [obs["accident"]],
                [obs["cluster_growth"]],
            )
            shared_queue_penalty = cfg.GLOBAL_QUEUE_WEIGHT * (
                global_queue / max(1.0, len(self.controlled_tls) * cfg.Q_NORM)
            )
            jam_penalty = 0.0
            if jam_flags["jam_active"]:
                jam_severity = max(1.0, global_queue / max(1.0, cfg.JAM_QUEUE_THRESHOLD))
                jam_penalty = cfg.JAM_PENALTY * jam_severity
            rewards[tid] = float(local_reward - shared_queue_penalty - jam_penalty)

        network_empty = traci.simulation.getMinExpectedNumber() <= 0 and self.step_count > 10
        done = self.step_count >= cfg.SIMULATION_STEPS or network_empty

        info = {
            "tls": next_obs,
            "actions": action_map,
            "global_queue": global_queue,
            "global_avg_delay": float(
                np.mean([obs["avg_delay"] for obs in next_obs.values()]) if next_obs else 0.0
            ),
            "accident_events": accident_events,
            "cleared_events": cleared_events,
            "jam_detected": jam_flags["jam_detected"],
            "jam_mitigation_activated": jam_flags["jam_mitigation_activated"],
            "jam_mitigation_relaxed": jam_flags["jam_mitigation_relaxed"],
            "jam_active": jam_flags["jam_active"],
            "demand_keep_prob": self._demand_keep_prob,
            "vehicles_removed": self._removed_vehicle_count,
            "step": self.step_count,
            "sim_time": float(traci.simulation.getTime()),
            "network_empty": network_empty,
        }
        return states, rewards, done, info

    def _normalize_action_map(self, actions: Dict[str, int] | int) -> Dict[str, int]:
        if isinstance(actions, int):
            if len(self.controlled_tls) != 1:
                raise ValueError("Scalar action is only valid when controlling one traffic light.")
            return {self.controlled_tls[0]: int(actions)}
        return {tid: int(actions.get(tid, 2)) for tid in self.controlled_tls}

    # ------------------------------------------------------------------
    # SUMO control setup
    # ------------------------------------------------------------------
    def _discover_controlled_tls(self) -> None:
        tl_ids = list(traci.trafficlight.getIDList())
        if not tl_ids:
            raise RuntimeError("No traffic lights found in SUMO network.")

        if cfg.TLS_ID:
            selected = [cfg.TLS_ID]
        elif cfg.CONTROLLED_TLS_IDS:
            selected = list(cfg.CONTROLLED_TLS_IDS)
        else:
            selected = [
                tid
                for tid in tl_ids
                if len(traci.trafficlight.getAllProgramLogics(tid)[0].phases) >= 4
            ]
            if cfg.MAX_CONTROLLED_TLS is not None:
                selected = selected[: int(cfg.MAX_CONTROLLED_TLS)]

        if not selected:
            raise RuntimeError("No multi-phase traffic lights selected for RL control.")

        self.controlled_tls = selected
        self._num_phases = {}
        self.current_phase = {}
        self._tls_incoming_lanes = {}
        self._lane_to_tls_dir = {}
        self._arrival_history = {}

        for tid in self.controlled_tls:
            phases = traci.trafficlight.getAllProgramLogics(tid)[0].phases
            self._num_phases[tid] = len(phases)
            self.current_phase[tid] = traci.trafficlight.getPhase(tid)
            self.optics[tid] = OpticsClustering()

            lanes = []
            for lane in traci.trafficlight.getControlledLanes(tid):
                if lane.startswith(":"):
                    continue
                if lane not in lanes:
                    lanes.append(lane)

            dir_map = {d: [] for d in self.DIRECTIONS}
            for lane in lanes:
                direction = self._classify_lane_direction(tid, lane)
                dir_map[direction].append(lane)
                self._lane_to_tls_dir[lane] = (tid, direction)

            self._tls_incoming_lanes[tid] = dir_map
            self._arrival_history[tid] = {
                d: deque(maxlen=400) for d in self.DIRECTIONS
            }

        print(
            f"[ENV] Controlling {len(self.controlled_tls)} traffic lights: "
            + ", ".join(self.controlled_tls)
        )

    def _classify_lane_direction(self, tid: str, lane_id: str) -> str:
        junction_x, junction_y = traci.junction.getPosition(tid)
        shape = traci.lane.getShape(lane_id)
        if not shape:
            return "N"
        start_x, start_y = shape[0]
        dx = start_x - junction_x
        dy = start_y - junction_y
        if abs(dx) >= abs(dy):
            return "W" if dx < 0 else "E"
        return "S" if dy < 0 else "N"

    def _refresh_current_phases(self) -> None:
        for tid in self.controlled_tls:
            try:
                self.current_phase[tid] = traci.trafficlight.getPhase(tid)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Observations and state building
    # ------------------------------------------------------------------
    def _collect_observations(self) -> Dict[str, Dict]:
        observations: Dict[str, Dict] = {}
        for tid in self.controlled_tls:
            queues = self._get_tls_queue_dict(tid)
            speeds = self._get_tls_speed_dict(tid)
            arrivals = self._get_tls_arrival_dict(tid)
            cluster_f = self._get_tls_cluster_features(tid)
            total_q = float(sum(queues.values()))
            avg_delay = compute_delay_from_queue(total_q)
            risk = accident_probability(
                total_q,
                RiskFeatures(weather_severity=0.2, congestion_growth_rate=cluster_f.growth_rate),
            )
            accident_flag = float(self.acc_mgr.get_acc_flag(tid))
            observations[tid] = {
                "queues": queues,
                "speeds": speeds,
                "arrivals": arrivals,
                "cluster_growth": float(cluster_f.growth_rate),
                "cluster_count": int(cluster_f.num_accident_clusters),
                "cluster_size": int(cluster_f.accident_cluster_size),
                "accident": accident_flag,
                "risk_score": float(risk),
                "avg_delay": float(avg_delay),
                "total_queue": total_q,
                "phase": self.current_phase.get(tid, 0),
            }
        return observations

    def _build_states(self, observations: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        states: Dict[str, np.ndarray] = {}
        for tid, obs in observations.items():
            queues = obs["queues"]
            speeds = obs["speeds"]
            arrivals = obs["arrivals"]
            state = np.array(
                [
                    np.clip(queues["N"], 0, cfg.MAX_QUEUE) / cfg.MAX_QUEUE,
                    np.clip(queues["S"], 0, cfg.MAX_QUEUE) / cfg.MAX_QUEUE,
                    np.clip(queues["E"], 0, cfg.MAX_QUEUE) / cfg.MAX_QUEUE,
                    np.clip(queues["W"], 0, cfg.MAX_QUEUE) / cfg.MAX_QUEUE,
                    np.clip(speeds["N"], 0, cfg.MAX_SPEED) / cfg.MAX_SPEED,
                    np.clip(speeds["S"], 0, cfg.MAX_SPEED) / cfg.MAX_SPEED,
                    np.clip(speeds["E"], 0, cfg.MAX_SPEED) / cfg.MAX_SPEED,
                    np.clip(speeds["W"], 0, cfg.MAX_SPEED) / cfg.MAX_SPEED,
                    np.clip(arrivals["N"], 0, cfg.MAX_ARRIVAL) / cfg.MAX_ARRIVAL,
                    np.clip(arrivals["S"], 0, cfg.MAX_ARRIVAL) / cfg.MAX_ARRIVAL,
                    np.clip(arrivals["E"], 0, cfg.MAX_ARRIVAL) / cfg.MAX_ARRIVAL,
                    np.clip(arrivals["W"], 0, cfg.MAX_ARRIVAL) / cfg.MAX_ARRIVAL,
                    np.clip(obs["cluster_growth"], 0, cfg.MAX_GROWTH) / cfg.MAX_GROWTH,
                    float(obs["accident"]),
                    np.clip(obs["risk_score"], 0.0, 1.0),
                ],
                dtype=np.float32,
            )
            states[tid] = state
        return states

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------
    def _apply_action_to_tls(
        self,
        tid: str,
        action: int,
        queues: Dict[str, float],
        duration_mode: str = "queue_scaled",
        fixed_duration: int = cfg.DEFAULT_GREEN,
    ) -> None:
        ns = queues["N"] + queues["S"]
        ew = queues["E"] + queues["W"]
        ns_phase = 0
        ew_phase = 2 if self._num_phases.get(tid, 0) >= 3 else 0
        clamped_fixed_duration = int(
            np.clip(fixed_duration, cfg.MIN_GREEN, cfg.MAX_GREEN)
        )

        if action == 0:
            phase = ns_phase
            if duration_mode == "fixed":
                duration = clamped_fixed_duration
            else:
                duration = int(
                    np.clip(cfg.MIN_GREEN + ns * 0.5, cfg.MIN_GREEN, cfg.MAX_GREEN)
                )
            traci.trafficlight.setPhase(tid, phase)
            traci.trafficlight.setPhaseDuration(tid, duration)
        elif action == 1:
            phase = ew_phase
            if duration_mode == "fixed":
                duration = clamped_fixed_duration
            else:
                duration = int(
                    np.clip(cfg.MIN_GREEN + ew * 0.5, cfg.MIN_GREEN, cfg.MAX_GREEN)
                )
            traci.trafficlight.setPhase(tid, phase)
            traci.trafficlight.setPhaseDuration(tid, duration)
        elif action == 2:
            try:
                remaining = traci.trafficlight.getNextSwitch(tid) - traci.simulation.getTime()
                duration = int(np.clip(remaining + 10, cfg.MIN_GREEN, cfg.MAX_GREEN))
                traci.trafficlight.setPhaseDuration(tid, duration)
            except Exception:
                pass
        elif action == 3:
            phase = ns_phase if ns >= ew else ew_phase
            traci.trafficlight.setPhase(tid, phase)
            traci.trafficlight.setPhaseDuration(tid, cfg.MIN_GREEN)

    # ------------------------------------------------------------------
    # Local feature extraction
    # ------------------------------------------------------------------
    def _get_tls_queue_dict(self, tid: str) -> Dict[str, float]:
        queues = {d: 0.0 for d in self.DIRECTIONS}
        for direction, lanes in self._tls_incoming_lanes[tid].items():
            for lane in lanes:
                for vid in traci.lane.getLastStepVehicleIDs(lane):
                    if traci.vehicle.getSpeed(vid) < 0.5:
                        queues[direction] += 1.0
        return queues

    def _get_tls_speed_dict(self, tid: str) -> Dict[str, float]:
        speeds = {d: 0.0 for d in self.DIRECTIONS}
        counts = {d: 0 for d in self.DIRECTIONS}
        for direction, lanes in self._tls_incoming_lanes[tid].items():
            for lane in lanes:
                for vid in traci.lane.getLastStepVehicleIDs(lane):
                    speeds[direction] += traci.vehicle.getSpeed(vid)
                    counts[direction] += 1
        for direction in self.DIRECTIONS:
            if counts[direction] > 0:
                speeds[direction] /= counts[direction]
        return speeds

    def _update_arrival_history(self) -> None:
        try:
            current_ids = set(traci.vehicle.getIDList())
            sim_time = float(traci.simulation.getTime())
            for vid in current_ids - self._prev_vehicle_ids:
                try:
                    lane = traci.vehicle.getLaneID(vid)
                except Exception:
                    continue

                if lane not in self._lane_to_tls_dir:
                    continue

                if self._jam_active and np.random.rand() > self._demand_keep_prob:
                    try:
                        traci.vehicle.remove(vid)
                        self._removed_vehicle_count += 1
                    except Exception:
                        pass
                    continue

                tid, direction = self._lane_to_tls_dir[lane]
                self._arrival_history[tid][direction].append(sim_time)

            self._prev_vehicle_ids = set(traci.vehicle.getIDList())
        except Exception:
            pass

    def _get_tls_arrival_dict(self, tid: str) -> Dict[str, float]:
        sim_time = float(traci.simulation.getTime())
        window = 60.0
        rates = {}
        for direction in self.DIRECTIONS:
            recent = sum(
                1
                for t_arr in self._arrival_history[tid][direction]
                if sim_time - t_arr <= window
            )
            rates[direction] = recent / window
        return rates

    def _get_tls_cluster_features(self, tid: str) -> ClusterFeatures:
        points = []
        for lanes in self._tls_incoming_lanes[tid].values():
            for lane in lanes:
                for vid in traci.lane.getLastStepVehicleIDs(lane):
                    if traci.vehicle.getSpeed(vid) < 0.5:
                        x, y = traci.vehicle.getPosition(vid)
                        points.append([float(x), float(y)])
        pts = np.array(points, dtype=float) if points else np.zeros((0, 2))
        return self.optics[tid].run(pts)

    # ------------------------------------------------------------------
    # Accident modeling
    # ------------------------------------------------------------------
    def _update_accidents(self, observations: Dict[str, Dict]) -> Tuple[List[Dict], List[Dict]]:
        accident_events: List[Dict] = []
        cleared_events: List[Dict] = []
        sim_time = float(traci.simulation.getTime())

        for tid, obs in observations.items():
            cleared = self.acc_mgr.update(tid, sim_time)
            if cleared:
                lane = self._accident_lane.pop(tid, None)
                if lane and lane in self._lane_speed_backup:
                    try:
                        traci.lane.setMaxSpeed(lane, self._lane_speed_backup[lane])
                    except Exception:
                        pass
                if lane:
                    self._lane_speed_backup.pop(lane, None)
                obs["accident"] = 0.0
                cleared_events.append(
                    {
                        "tls_id": tid,
                        "clearance_seconds": float(cleared[0]),
                    }
                )

            if self.acc_mgr.get_acc_flag(tid):
                obs["accident"] = 1.0
                continue

            accident_prob = min(
                cfg.ACCIDENT_MAX_PROB_PER_STEP,
                max(0.0, obs["risk_score"]) * cfg.ACCIDENT_RATE_SCALE,
            )
            if np.random.rand() >= accident_prob:
                obs["accident"] = 0.0
                continue

            lane = self._choose_accident_lane(tid, obs["queues"])
            if lane:
                try:
                    original_speed = traci.lane.getMaxSpeed(lane)
                    traci.lane.setMaxSpeed(lane, cfg.ACCIDENT_BLOCK_SPEED)
                    self._lane_speed_backup[lane] = original_speed
                    self._accident_lane[tid] = lane
                except Exception:
                    lane = None

            clearance_seconds, _ = self.acc_mgr.trigger_accident(tid, sim_time)
            obs["accident"] = 1.0
            accident_events.append(
                {
                    "tls_id": tid,
                    "lane_id": lane,
                    "risk_score": float(obs["risk_score"]),
                    "clearance_seconds": float(clearance_seconds),
                }
            )

        return accident_events, cleared_events

    def _choose_accident_lane(self, tid: str, queues: Dict[str, float]) -> Optional[str]:
        sorted_dirs = sorted(self.DIRECTIONS, key=lambda d: queues[d], reverse=True)
        for direction in sorted_dirs:
            lanes = self._tls_incoming_lanes[tid][direction]
            if lanes:
                return lanes[0]
        return None

    # ------------------------------------------------------------------
    # Jam detection and dynamic demand control
    # ------------------------------------------------------------------
    def _update_jam_control(self, global_queue: float) -> Dict[str, bool]:
        queue_jump = global_queue - self._last_global_queue
        jam_detected = (
            global_queue >= cfg.JAM_QUEUE_THRESHOLD
            or queue_jump >= cfg.JAM_QUEUE_JUMP_THRESHOLD
        )
        mitigation_activated = False
        mitigation_relaxed = False

        if jam_detected:
            if not self._jam_active:
                self._jam_active = True
                mitigation_activated = True
                self._demand_keep_prob = min(self._demand_keep_prob, cfg.JAM_KEEP_PROB)
            elif global_queue >= cfg.JAM_QUEUE_THRESHOLD * 1.25:
                self._demand_keep_prob = max(
                    cfg.JAM_KEEP_PROB_MIN,
                    self._demand_keep_prob - cfg.JAM_KEEP_PROB_STEP_DOWN,
                )
        elif self._jam_active and global_queue <= cfg.JAM_RECOVERY_THRESHOLD:
            self._demand_keep_prob = min(1.0, self._demand_keep_prob + cfg.JAM_KEEP_PROB_RECOVER)
            mitigation_relaxed = True
            if self._demand_keep_prob >= 0.999:
                self._jam_active = False

        self._last_global_queue = global_queue
        return {
            "jam_detected": jam_detected,
            "jam_mitigation_activated": mitigation_activated,
            "jam_mitigation_relaxed": mitigation_relaxed,
            "jam_active": self._jam_active,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self) -> None:
        try:
            traci.close(wait=False)
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()
