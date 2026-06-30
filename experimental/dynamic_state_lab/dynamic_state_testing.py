from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations
from math import hypot
from typing import Deque, Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class VehicleObservation:
    vehicle_id: str
    intersection_id: str
    timestamp_s: float
    x: float
    y: float
    speed_mps: float
    in_cluster: bool


@dataclass
class VehicleSummary:
    vehicle_id: str
    intersection_id: str
    history_frames: int
    history_span_s: float
    mean_speed_mps: float
    speed_std_mps: float
    stable_speed: bool
    stalled_stable: bool
    recent_cluster_exit: bool
    recent_transition: bool
    collision_now: bool
    accident_candidate: bool
    same_intersection_candidate: bool
    entering_intersection_candidate: bool
    stable_frame_run: int
    time_since_exit_s: float | None


class DynamicStateVectorLab:
    """
    Standalone prototype for richer per-intersection state construction.

    The raw state can grow up to `max_raw_dim`, but the final policy-facing
    vector is compressed back to the current 15-D contract.
    """

    def __init__(
        self,
        max_raw_dim: int = 40,
        policy_dim: int = 15,
        history_window_s: float = 60.0,
        min_history_frames: int = 30,
        stable_speed_std_threshold: float = 0.35,
        stalled_speed_threshold: float = 1.0,
        collision_radius_m: float = 3.0,
        recent_exit_window_s: float = 20.0,
    ):
        self.max_raw_dim = int(max_raw_dim)
        self.policy_dim = int(policy_dim)
        self.history_window_s = float(history_window_s)
        self.min_history_frames = int(min_history_frames)
        self.stable_speed_std_threshold = float(stable_speed_std_threshold)
        self.stalled_speed_threshold = float(stalled_speed_threshold)
        self.collision_radius_m = float(collision_radius_m)
        self.recent_exit_window_s = float(recent_exit_window_s)

        self._vehicle_history: Dict[str, Deque[VehicleObservation]] = defaultdict(
            lambda: deque(maxlen=max(120, self.min_history_frames * 4))
        )
        self._last_cluster_state: Dict[str, bool] = {}
        self._last_exit_time_s: Dict[str, float] = {}
        self._last_intersection_id: Dict[str, str] = {}
        self._raw_maxima = np.array(
            [
                100.0, 100.0, 100.0, 100.0,
                15.0, 15.0, 15.0, 15.0,
                2.0, 2.0, 2.0, 2.0,
                5.0, 1.0, 1.0,
                60.0, 60.0, 30.0, 1.0, 3.0,
                3.0, 3.0, 60.0, 60.0, 60.0,
                60.0, 60.0, 60.0, 60.0, 1.0,
                60.0, 60.0, 60.0, 20.0, 1.0,
                1.0, 60.0, 120.0, 120.0, 1.0,
            ],
            dtype=np.float32,
        )

    def update_intersection_state(
        self,
        intersection_id: str,
        observations: Iterable[VehicleObservation],
        base_features: Dict[str, float] | None = None,
    ) -> Dict[str, object]:
        """
        Build one normalized 40-D raw vector and one compressed 15-D vector.

        `base_features` mirrors the live environment's fixed features:
        queues, speeds, arrivals, cluster_growth, accident_flag, risk_score.
        """
        all_observations = list(observations)
        for obs in all_observations:
            self._record_observation(obs)

        obs_list = [obs for obs in all_observations if obs.intersection_id == intersection_id]
        collision_flags, pair_count, mean_pair_distance = self._detect_collisions(obs_list)

        summaries: List[VehicleSummary] = []
        for obs in obs_list:
            summaries.append(
                self._summarize_vehicle(
                    obs=obs,
                    collision_now=collision_flags.get(obs.vehicle_id, False),
                )
            )

        raw_vector = self._build_raw_vector(
            intersection_id=intersection_id,
            summaries=summaries,
            pair_count=pair_count,
            mean_pair_distance=mean_pair_distance,
            base_features=base_features or {},
        )
        normalized_raw = self._normalize_raw_vector(raw_vector)
        policy_vector = self._compress_to_policy_vector(normalized_raw)

        return {
            "raw_vector": raw_vector,
            "normalized_raw_vector": normalized_raw,
            "policy_vector": policy_vector,
            "vehicle_summaries": summaries,
        }

    def _record_observation(self, obs: VehicleObservation) -> None:
        history = self._vehicle_history[obs.vehicle_id]
        history.append(obs)

        prev_cluster_state = self._last_cluster_state.get(obs.vehicle_id)
        if prev_cluster_state and not obs.in_cluster:
            self._last_exit_time_s[obs.vehicle_id] = obs.timestamp_s
        self._last_cluster_state[obs.vehicle_id] = obs.in_cluster

        self._last_intersection_id[obs.vehicle_id] = obs.intersection_id

    def _summarize_vehicle(self, obs: VehicleObservation, collision_now: bool) -> VehicleSummary:
        history = self._recent_history(obs.vehicle_id, obs.timestamp_s)
        speeds = np.array([item.speed_mps for item in history], dtype=np.float32)
        intersections = [item.intersection_id for item in history]

        mean_speed = float(np.mean(speeds)) if len(speeds) else 0.0
        speed_std = float(np.std(speeds)) if len(speeds) else 0.0
        history_span = float(history[-1].timestamp_s - history[0].timestamp_s) if len(history) > 1 else 0.0

        recent_transition = len(set(intersections)) > 1
        stable_speed = (
            len(history) >= self.min_history_frames
            and history_span >= min(30.0, self.history_window_s * 0.5)
            and speed_std <= self.stable_speed_std_threshold
        )
        stalled_stable = stable_speed and mean_speed <= self.stalled_speed_threshold
        stable_run = self._stable_frame_run(history)

        time_since_exit = None
        last_exit = self._last_exit_time_s.get(obs.vehicle_id)
        recent_cluster_exit = False
        if last_exit is not None:
            time_since_exit = max(0.0, obs.timestamp_s - last_exit)
            recent_cluster_exit = time_since_exit <= self.recent_exit_window_s

        entering_candidate = stable_speed and recent_transition and collision_now
        same_intersection_candidate = stable_speed and (not recent_transition) and collision_now
        accident_candidate = stalled_stable and (entering_candidate or same_intersection_candidate)

        return VehicleSummary(
            vehicle_id=obs.vehicle_id,
            intersection_id=obs.intersection_id,
            history_frames=len(history),
            history_span_s=history_span,
            mean_speed_mps=mean_speed,
            speed_std_mps=speed_std,
            stable_speed=stable_speed,
            stalled_stable=stalled_stable,
            recent_cluster_exit=recent_cluster_exit,
            recent_transition=recent_transition,
            collision_now=collision_now,
            accident_candidate=accident_candidate,
            same_intersection_candidate=same_intersection_candidate,
            entering_intersection_candidate=entering_candidate,
            stable_frame_run=stable_run,
            time_since_exit_s=time_since_exit,
        )

    def _recent_history(self, vehicle_id: str, now_s: float) -> List[VehicleObservation]:
        history = self._vehicle_history[vehicle_id]
        recent = [item for item in history if now_s - item.timestamp_s <= self.history_window_s]
        return recent if recent else list(history)

    def _stable_frame_run(self, history: List[VehicleObservation]) -> int:
        if not history:
            return 0
        run = 0
        tail_speeds: List[float] = []
        for item in reversed(history):
            tail_speeds.append(item.speed_mps)
            if len(tail_speeds) < 3:
                run += 1
                continue
            if float(np.std(np.array(tail_speeds, dtype=np.float32))) <= self.stable_speed_std_threshold:
                run += 1
            else:
                break
        return run

    def _detect_collisions(
        self, observations: List[VehicleObservation]
    ) -> Tuple[Dict[str, bool], int, float]:
        collision_flags = {obs.vehicle_id: False for obs in observations}
        pair_count = 0
        pair_distances: List[float] = []

        for left, right in combinations(observations, 2):
            distance = hypot(left.x - right.x, left.y - right.y)
            pair_distances.append(distance)
            if distance <= self.collision_radius_m:
                collision_flags[left.vehicle_id] = True
                collision_flags[right.vehicle_id] = True
                pair_count += 1

        mean_pair_distance = float(np.mean(pair_distances)) if pair_distances else 0.0
        return collision_flags, pair_count, mean_pair_distance

    def _build_raw_vector(
        self,
        intersection_id: str,
        summaries: List[VehicleSummary],
        pair_count: int,
        mean_pair_distance: float,
        base_features: Dict[str, float],
    ) -> np.ndarray:
        base = self._expand_base_features(base_features)
        vehicle_count = len(summaries)
        speed_stds = np.array([item.speed_std_mps for item in summaries], dtype=np.float32)
        stable_runs = np.array([item.stable_frame_run for item in summaries], dtype=np.float32)

        clustered_count = sum(1 for item in summaries if self._last_cluster_state.get(item.vehicle_id, False))
        recent_exit_count = sum(1 for item in summaries if item.recent_cluster_exit)
        stable_count = sum(1 for item in summaries if item.stable_speed)
        stalled_stable_count = sum(1 for item in summaries if item.stalled_stable)
        collision_vehicle_count = sum(1 for item in summaries if item.collision_now)
        transition_count = sum(1 for item in summaries if item.recent_transition)
        accident_candidate_count = sum(1 for item in summaries if item.accident_candidate)
        same_intersection_candidate_count = sum(
            1 for item in summaries if item.same_intersection_candidate
        )
        entering_intersection_candidate_count = sum(
            1 for item in summaries if item.entering_intersection_candidate
        )
        exit_with_collision_count = sum(
            1 for item in summaries if item.recent_cluster_exit and item.collision_now
        )

        exit_deltas = [
            item.time_since_exit_s
            for item in summaries
            if item.time_since_exit_s is not None and item.recent_cluster_exit
        ]

        accident_ratio = accident_candidate_count / max(1, vehicle_count)
        stopped_ratio = stalled_stable_count / max(1, vehicle_count)
        low_variance_ratio = stable_count / max(1, vehicle_count)

        extra = np.array(
            [
                float(vehicle_count),
                float(clustered_count),
                float(recent_exit_count),
                float(recent_exit_count / max(1.0, self.history_window_s)),
                float(np.mean(speed_stds)) if len(speed_stds) else 0.0,
                float(np.min(speed_stds)) if len(speed_stds) else 0.0,
                float(np.max(speed_stds)) if len(speed_stds) else 0.0,
                float(stable_count),
                float(stalled_stable_count),
                float(pair_count),
                float(collision_vehicle_count),
                float(transition_count),
                float(accident_candidate_count),
                float(accident_ratio),
                float(np.mean(exit_deltas)) if exit_deltas else 0.0,
                float(same_intersection_candidate_count),
                float(entering_intersection_candidate_count),
                float(mean_pair_distance),
                float(stopped_ratio),
                float(low_variance_ratio),
                float(exit_with_collision_count),
                float(np.max(stable_runs)) if len(stable_runs) else 0.0,
                float(np.mean(stable_runs)) if len(stable_runs) else 0.0,
                float(min(1.0, max((item.history_span_s for item in summaries), default=0.0) / self.history_window_s)),
                0.0,
            ],
            dtype=np.float32,
        )

        raw_vector = np.concatenate([base, extra], axis=0)
        if len(raw_vector) != self.max_raw_dim:
            raise ValueError(
                f"Expected {self.max_raw_dim} raw features, got {len(raw_vector)} for {intersection_id}."
            )
        return raw_vector

    def _expand_base_features(self, base_features: Dict[str, float]) -> np.ndarray:
        def fetch(key: str, default: float = 0.0) -> float:
            return float(base_features.get(key, default))

        return np.array(
            [
                fetch("queue_n"),
                fetch("queue_s"),
                fetch("queue_e"),
                fetch("queue_w"),
                fetch("speed_n"),
                fetch("speed_s"),
                fetch("speed_e"),
                fetch("speed_w"),
                fetch("arrival_n"),
                fetch("arrival_s"),
                fetch("arrival_e"),
                fetch("arrival_w"),
                fetch("cluster_growth"),
                fetch("accident_flag"),
                fetch("risk_score"),
            ],
            dtype=np.float32,
        )

    def _normalize_raw_vector(self, raw_vector: np.ndarray) -> np.ndarray:
        normalized = np.divide(
            np.clip(raw_vector, a_min=0.0, a_max=None),
            self._raw_maxima,
            out=np.zeros_like(raw_vector, dtype=np.float32),
            where=self._raw_maxima > 0,
        )
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    def _compress_to_policy_vector(self, normalized_raw: np.ndarray) -> np.ndarray:
        policy = np.zeros((self.policy_dim,), dtype=np.float32)

        policy[0:4] = normalized_raw[0:4]
        policy[4:8] = normalized_raw[4:8]
        policy[8:12] = normalized_raw[8:12]

        policy[12] = float(
            np.mean(
                normalized_raw[
                    [12, 16, 17, 18, 30, 34, 36]
                ]
            )
        )
        policy[13] = float(
            np.max(
                normalized_raw[
                    [13, 23, 24, 25, 27, 28, 31, 32]
                ]
            )
        )
        policy[14] = float(
            np.mean(
                normalized_raw[
                    [14, 19, 20, 21, 22, 26, 29, 33, 35, 37, 38, 39]
                ]
            )
        )

        return np.clip(policy, 0.0, 1.0)


def _demo_observations() -> Tuple[List[VehicleObservation], Dict[str, float]]:
    observations: List[VehicleObservation] = []

    for step in range(65):
        t = float(step)
        observations.append(
            VehicleObservation(
                vehicle_id="veh_a",
                intersection_id="J2" if step >= 35 else "J1",
                timestamp_s=t,
                x=10.0 + min(step, 35) * 0.25,
                y=5.0,
                speed_mps=0.45 + (0.01 if step % 2 == 0 else -0.01),
                in_cluster=step < 30,
            )
        )
        observations.append(
            VehicleObservation(
                vehicle_id="veh_b",
                intersection_id="J2",
                timestamp_s=t,
                x=18.5 if step < 55 else 18.1,
                y=5.0 if step < 55 else 5.2,
                speed_mps=0.42 + (0.02 if step % 3 == 0 else -0.01),
                in_cluster=step < 28,
            )
        )
        observations.append(
            VehicleObservation(
                vehicle_id="veh_c",
                intersection_id="J2",
                timestamp_s=t,
                x=40.0 + step * 0.5,
                y=16.0,
                speed_mps=5.5 + (0.4 if step % 5 == 0 else -0.3),
                in_cluster=False,
            )
        )

    base_features = {
        "queue_n": 12.0,
        "queue_s": 9.0,
        "queue_e": 17.0,
        "queue_w": 6.0,
        "speed_n": 1.2,
        "speed_s": 1.0,
        "speed_e": 0.8,
        "speed_w": 1.5,
        "arrival_n": 0.8,
        "arrival_s": 0.6,
        "arrival_e": 1.1,
        "arrival_w": 0.5,
        "cluster_growth": 1.8,
        "accident_flag": 0.0,
        "risk_score": 0.35,
    }
    return observations, base_features


def main() -> None:
    lab = DynamicStateVectorLab()
    observations, base_features = _demo_observations()

    latest_by_time: Dict[float, List[VehicleObservation]] = defaultdict(list)
    for obs in observations:
        latest_by_time[obs.timestamp_s].append(obs)

    result = {}
    for timestamp in sorted(latest_by_time):
        result = lab.update_intersection_state(
            intersection_id="J2",
            observations=latest_by_time[timestamp],
            base_features=base_features,
        )

    print("Normalized 40-D raw vector:")
    print(np.array2string(result["normalized_raw_vector"], precision=3, suppress_small=True))
    print()
    print("Compressed 15-D policy vector:")
    print(np.array2string(result["policy_vector"], precision=3, suppress_small=True))
    print()
    print("Vehicle summaries at final frame:")
    for summary in result["vehicle_summaries"]:
        print(
            f"{summary.vehicle_id}: "
            f"stable={summary.stable_speed} "
            f"stalled_stable={summary.stalled_stable} "
            f"recent_exit={summary.recent_cluster_exit} "
            f"transition={summary.recent_transition} "
            f"collision={summary.collision_now} "
            f"accident_candidate={summary.accident_candidate} "
            f"speed_std={summary.speed_std_mps:.3f}"
        )


if __name__ == "__main__":
    main()
