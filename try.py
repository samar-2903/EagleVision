"""
Adaptive SUMO Traffic Management Pipeline
- CNN detects vehicles in real-time
- DBSCAN clusters vehicles dynamically based on congestion
- Assigns priority weights (ambulance > bus > car > bike)
- Traffic light control (RL placeholder)
"""

import traci
import numpy as np
from sklearn.cluster import DBSCAN
import subprocess
import time
import random

# =========================
# Configuration
# =========================
SUMO_BINARY = "sumo-gui"
SUMO_CFG = "grid_tls.sumocfg"
SIMULATION_STEPS = 3600
CONTROL_INTERVAL = 1

DBSCAN_BASE_EPS = 20
DBSCAN_BASE_MIN_SAMPLES = 2
EPS_GROWTH_FACTOR = 0.1
EPS_DECAY_FACTOR = 0.1
MIN_SAMPLES_SCALE = 1

# =========================
# Launch SUMO with dynamic port
# =========================
def start_sumo():
    port = random.randint(55000, 56000)
    sumo_cmd = [SUMO_BINARY, "-c", SUMO_CFG, "--start", "--remote-port", str(port)]
    print(f"Launching SUMO on port {port}...")
    subprocess.Popen(sumo_cmd)
    # Wait a few seconds for SUMO to start
    time.sleep(2)
    # Connect
    traci.init(port)
    print("Connected to SUMO.")
    return port

# =========================
# Vehicle Data Functions
# =========================
def get_vehicle_positions():
    vehicle_data = []
    for vehID in traci.vehicle.getIDList():
        v_type = traci.vehicle.getTypeID(vehID)
        pos = traci.vehicle.getPosition(vehID)
        speed = traci.vehicle.getSpeed(vehID)
        vehicle_data.append({
            "id": vehID,
            "type": v_type,
            "position": pos,
            "speed": speed
        })
    return vehicle_data

def adaptive_dbscan(vehicle_data, prev_eps=DBSCAN_BASE_EPS):
    positions = np.array([v["position"] for v in vehicle_data])
    if len(positions) == 0:
        return [], positions, prev_eps

    # Compute average nearest neighbor distance
    from sklearn.neighbors import NearestNeighbors
    if len(positions) > 1:
        nbrs = NearestNeighbors(n_neighbors=2).fit(positions)
        distances, _ = nbrs.kneighbors(positions)
        avg_distance = np.mean(distances[:, 1])
    else:
        avg_distance = prev_eps

    # Adjust eps
    eps = prev_eps
    if avg_distance < prev_eps:
        eps += EPS_GROWTH_FACTOR * prev_eps
    else:
        eps -= EPS_DECAY_FACTOR * prev_eps
    eps = max(5, min(50, eps))

    # Scale min_samples
    min_samples = max(2, int(MIN_SAMPLES_SCALE * len(positions) / 5))

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
    return clustering.labels_, positions, eps

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

def get_junction_state(vehicle_data, labels, weights):
    speeds = [v["speed"] for v in vehicle_data]
    total_weighted = sum([weights[v["id"]] for v in vehicle_data])
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    max_cluster_size = max([list(labels).count(i) for i in set(labels) if i != -1], default=0)
    return {
        "total_weighted_vehicles": total_weighted,
        "avg_speed": np.mean(speeds) if speeds else 0,
        "num_clusters": num_clusters,
        "max_cluster_size": max_cluster_size
    }

def decide_traffic_light(state, tl_id="junction0"):
    if state["total_weighted_vehicles"] > 8:
        traci.trafficlight.setPhaseDuration(tl_id, 30)
        print(f"[{tl_id}] Extended green, weight={state['total_weighted_vehicles']}")
    else:
        traci.trafficlight.setPhaseDuration(tl_id, 15)



# =========================
# Main Simulation Loop
# =========================
def run_simulation():
    port = start_sumo()
    step = 0
    prev_eps = DBSCAN_BASE_EPS

    while step < SIMULATION_STEPS:
        traci.simulationStep()
        vehicle_data = get_vehicle_positions()
        labels, positions, prev_eps = adaptive_dbscan(vehicle_data, prev_eps)
        weights = assign_weights(vehicle_data)
        state = get_junction_state(vehicle_data, labels, weights)
        decide_traffic_light(state, tl_id="junction0")

        if step % 60 == 0:
            print(f"Step {step}: {state}, eps={prev_eps:.2f}")

        step += CONTROL_INTERVAL

    traci.close()
    print("Simulation completed.")

if __name__ == "__main__":
    run_simulation()
