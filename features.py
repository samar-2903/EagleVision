from collections import defaultdict, deque
import numpy as np

_lane_queue_history = defaultdict(lambda: deque(maxlen=5))


def assign_weights(vehicle_data):
    weights = {}
    for v in vehicle_data:
        vt_lower = v["type"].lower()
        if "ambulance" in vt_lower:
            weights[v["id"]] = 10
        elif "bus" in vt_lower:
            weights[v["id"]] = 5
        elif "car" in vt_lower:
            weights[v["id"]] = 1
        else:
            weights[v["id"]] = 0.5
    return weights


def compute_lane_features(vehicle_data, labels):
    lane_to_vehicle_ids = defaultdict(list)
    for v in vehicle_data:
        lane_to_vehicle_ids[v["lane"]].append(v["id"])

    lane_features = {}
    for lane, veh_ids in lane_to_vehicle_ids.items():
        lane_vehicles = [v for v in vehicle_data if v["id"] in veh_ids]
        speeds = [v["speed"] for v in lane_vehicles]
        waits = [v["wait"] for v in lane_vehicles]
        queue_len = sum(1 for v in lane_vehicles if v["speed"] < 0.5 or v["wait"] > 1.0)
        avg_wait = float(np.mean(waits)) if waits else 0.0
        avg_speed = float(np.mean(speeds)) if speeds else 0.0

        _lane_queue_history[lane].append(queue_len)
        hist = list(_lane_queue_history[lane])
        growth = 0.0
        if len(hist) >= 2:
            growth = float(hist[-1] - hist[0]) / max(1, len(hist) - 1)

        density = len(lane_vehicles) / 50.0

        lane_features[lane] = {
            "queue_len": queue_len,
            "avg_wait": avg_wait,
            "avg_speed": avg_speed,
            "growth_rate": growth,
            "density": density
        }
    return lane_features


def summarize_junction(vehicle_data, labels, weights, lane_features):
    speeds = [v["speed"] for v in vehicle_data]
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    max_cluster_size = max([list(labels).count(i) for i in set(labels) if i != -1], default=0)
    total_weighted = sum([weights[v["id"]] for v in vehicle_data])

    queue_len_total = int(sum(lf["queue_len"] for lf in lane_features.values()))
    avg_wait_total = float(np.mean([lf["avg_wait"] for lf in lane_features.values()])) if lane_features else 0.0
    avg_speed_overall = float(np.mean(speeds)) if speeds else 0.0

    return {
        "total_weighted_vehicles": total_weighted,
        "avg_speed": avg_speed_overall,
        "num_clusters": num_clusters,
        "max_cluster_size": max_cluster_size,
        "lane_features": lane_features,
        "queue_len_total": queue_len_total,
        "avg_wait_total": avg_wait_total
    }


