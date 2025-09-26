from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from sklearn.cluster import OPTICS


@dataclass
class ClusterFeatures:
    num_accident_clusters: int
    accident_cluster_size: int
    growth_rate: float


class OpticsClustering:
    def __init__(self, min_samples: int = 5, xi: float = 0.05, min_cluster_size: int = 0.05):
        self.model = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
        self.prev_labels: np.ndarray | None = None
        self.growth_buffer: list[float] = []
        self.h_window: int = 5
        self.G_high: float = 2.0
        self.G_low: float = 0.5

    def _compute_growth_rate(self, labels_prev: np.ndarray | None, labels_now: np.ndarray) -> float:
        if labels_prev is None:
            return 0.0
        def size_map(labels: np.ndarray) -> dict:
            m = {}
            for lbl in set(labels.tolist()):
                if lbl == -1:
                    continue
                m[lbl] = int(np.sum(labels == lbl))
            return m
        prev_map = size_map(labels_prev)
        now_map = size_map(labels_now)
        # Compare total clustered points size
        prev_total = sum(prev_map.values())
        now_total = sum(now_map.values())
        dt = 1.0
        return float((now_total - prev_total) / max(1.0, dt))

    def run(self, accident_points_xy: np.ndarray) -> ClusterFeatures:
        n = int(accident_points_xy.shape[0])
        if n == 0:
            return ClusterFeatures(0, 0, 0.0)
        # If too few samples for OPTICS, skip clustering gracefully
        if n < int(self.model.min_samples):
            labels = np.full((n,), -1, dtype=int)
            growth = self._compute_growth_rate(self.prev_labels, labels)
            self.prev_labels = labels
            return ClusterFeatures(0, 0, growth)

        labels = self.model.fit_predict(accident_points_xy)
        growth = self._compute_growth_rate(self.prev_labels, labels)
        self.prev_labels = labels
        # maintain growth hysteresis buffer
        self.growth_buffer.append(float(growth))
        if len(self.growth_buffer) > self.h_window:
            self.growth_buffer.pop(0)

        mask = labels != -1
        if not np.any(mask):
            return ClusterFeatures(0, 0, growth)
        num_clusters = len(set(labels[mask].tolist()))
        max_size = 0
        for lbl in set(labels[mask].tolist()):
            max_size = max(max_size, int(np.sum(labels == lbl)))
        return ClusterFeatures(num_clusters, max_size, growth)

    def growth_priority_axis(self, ns: float, ew: float) -> str | None:
        # If sustained high growth, prioritize heavier axis
        if len(self.growth_buffer) < self.h_window:
            return None
        if all(g > self.G_high for g in self.growth_buffer):
            return "NS" if ns >= ew else "EW"
        if all(g < self.G_low for g in self.growth_buffer):
            return None
        return None

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import config as cfg


def adaptive_dbscan(vehicle_data, prev_eps=cfg.DBSCAN_BASE_EPS):
    positions = np.array([v["position"] for v in vehicle_data])
    if len(positions) == 0:
        return [], positions, prev_eps

    if len(positions) > 1:
        nbrs = NearestNeighbors(n_neighbors=2).fit(positions)
        distances, _ = nbrs.kneighbors(positions)
        avg_distance = np.mean(distances[:, 1])
    else:
        avg_distance = prev_eps

    eps = prev_eps
    if avg_distance < prev_eps:
        eps += cfg.EPS_GROWTH_FACTOR * prev_eps
    else:
        eps -= cfg.EPS_DECAY_FACTOR * prev_eps
    eps = max(5, min(50, eps))

    min_samples = max(2, int(cfg.MIN_SAMPLES_SCALE * len(positions) / 5))

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
    return clustering.labels_, positions, eps


