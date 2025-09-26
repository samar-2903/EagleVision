from __future__ import annotations

import csv
from typing import List


class CsvRecorder:
    def __init__(self, path: str):
        self.path = path
        self._ensure_header()

    def _ensure_header(self) -> None:
        try:
            with open(self.path, "x", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "t_s",
                    "reward_t",
                    "episode_return",
                    "num_intersections",
                    "totalQ_list",
                    "avg_delay_list",
                    "accidents_list",
                    "num_accident_clusters",
                    "max_cluster_size",
                    "c_rl0",
                    "c_ens0",
                    "c_lstm0",
                    "c_st0",
                    "alpha0",
                    "avg_commute_s",
                ])
        except FileExistsError:
            pass

    def write_step(self, t_s: float, reward_t: float, episode_return: float, totalQ: List[float], avg_delays: List[float], accidents: List[int], num_clusters: int, max_cluster: int, c_rl0: float = 0.0, c_ens0: float = 0.0, c_lstm0: float = 0.0, c_st0: float = 0.0, alpha0: float = 0.0, avg_commute_s: float = 0.0) -> None:
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                float(t_s),
                float(reward_t),
                float(episode_return),
                int(len(totalQ)),
                ";".join(f"{x:.4f}" for x in totalQ),
                ";".join(f"{x:.4f}" for x in avg_delays),
                ";".join(str(a) for a in accidents),
                int(num_clusters),
                int(max_cluster),
                float(c_rl0),
                float(c_ens0),
                float(c_lstm0),
                float(c_st0),
                float(alpha0),
                float(avg_commute_s),
            ])


