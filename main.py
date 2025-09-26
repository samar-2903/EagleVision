import traci
import numpy as np
import random
import config as cfg
from sumo_utils import start_sumo
from data import get_vehicle_positions, get_vehicle_features, LaneArrivalCounter
from control import RLPolicy, GNNForecaster, HierarchicalController
from clustering import OpticsClustering
from risk import AccidentMDP, RiskFeatures, accident_probability
from accident_manager import AccidentManager
from service_model import ServiceModel


def run_simulation():
    port = start_sumo()
    step = 0

    # Controllers and models
    optics = OpticsClustering()
    ctrl = HierarchicalController(RLPolicy(), GNNForecaster(), degrade_threshold=5)
    accident_mdp = AccidentMDP(num_nodes=1)
    acc_mgr = AccidentManager()
    service = ServiceModel(dt=float(cfg.CONTROL_INTERVAL) if hasattr(cfg, "CONTROL_INTERVAL") else 1.0)

    lane_counter = LaneArrivalCounter(window_s=60.0)
    # seeds for reproducibility across runs
    try:
        random.seed(getattr(cfg, "SEED", 1234))
        np.random.seed(getattr(cfg, "SEED", 1234))
    except Exception:
        pass

    # commute tracking
    depart_time = {}
    trip_times = []

    while step < cfg.SIMULATION_STEPS:
        traci.simulationStep()
        t_s = float(step)
        # commute tracking via SUMO departed/arrived lists
        for vid in traci.simulation.getDepartedIDList():
            depart_time[vid] = t_s
        for vid in traci.simulation.getArrivedIDList():
            t0 = depart_time.pop(vid, None)
            if t0 is not None:
                trip_times.append(float(t_s - t0))
        vehicle_data = get_vehicle_positions()
        lane_counter.update(t_s)
        vfeats = get_vehicle_features(t_s)

        # Build queues per direction using lane id hints (N,S,E,W) and speed threshold
        queues = {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0}
        avg_speed = 0.0
        if vehicle_data:
            avg_speed = float(np.mean([max(0.0, v["speed"]) for v in vehicle_data]))
        s_th = 0.5
        for v in vfeats:
            lane = (v.get("lane") or "").upper()
            is_stopped = int(v.get("stopped", 0))
            incr = 1.0 if is_stopped else 0.2
            if "N" in lane:
                queues["N"] += incr
            elif "S" in lane:
                queues["S"] += incr
            elif "E" in lane:
                queues["E"] += incr
            elif "W" in lane:
                queues["W"] += incr
            else:
                # unknown orientation → add to largest current queue to be conservative
                dmax = max(queues, key=lambda d: queues[d])
                queues[dmax] += 0.2

        # Accident clustering from positions of stopped/slow vehicles
        points = []
        for v in vfeats:
            if int(v.get("stopped", 0)) == 1:
                x, y = v.get("position", (0.0, 0.0))
                points.append([float(x), float(y)])
        pts = np.array(points, dtype=float) if points else np.zeros((0, 2))
        cluster_feats = optics.run(pts)

        # Risk features and accident state
        # Simple congestion growth: difference from last loop via controller memory not kept here; use cluster growth
        risk_feats = [RiskFeatures(weather_severity=0.2, congestion_growth_rate=cluster_feats.growth_rate)]
        accidents, cleared = accident_mdp.step(float(step), [queues], risk_feats)
        if accidents[0] == 1 and acc_mgr.get_acc_flag(0) == 0:
            acc_mgr.trigger_accident(0, float(step))
            # Log approximate accident location using centroid of stopped vehicle points
            if pts.shape[0] > 0:
                cx = float(np.mean(pts[:, 0]))
                cy = float(np.mean(pts[:, 1]))
                print(f"[ALERT] Accident detected near ({cx:.1f}, {cy:.1f}) at t={step}s")
            else:
                print(f"[ALERT] Accident detected at t={step}s (location unknown)")
        upd = acc_mgr.update(0, float(step))
        if upd and upd[1]:
            for d in ("N","S","E","W"):
                queues[d] = max(0.0, queues[d] * acc_mgr.r_clear)

        # Prepare observation for controller
        # lane-level aggregate speeds and arrival rates
        lane_avg_speed = {d: 0.0 for d in ("N","S","E","W")}
        lane_counts = {d: 0 for d in ("N","S","E","W")}
        for v in vfeats:
            lane = (v.get("lane") or "").upper()
            sp = float(v.get("speed_est", 0.0))
            for d in ("N","S","E","W"):
                if d in lane:
                    lane_avg_speed[d] += sp
                    lane_counts[d] += 1
        for d in lane_avg_speed:
            lane_avg_speed[d] = float(lane_avg_speed[d] / lane_counts[d]) if lane_counts[d] > 0 else 0.0
        lane_arrival = {d: 0.0 for d in ("N","S","E","W")}
        for d in lane_arrival:
            # approximate by mapping any lane id substring with d
            rates = []
            for v in vfeats:
                ln = v.get("lane") or ""
                if d in ln.upper():
                    rates.append(lane_counter.rate(ln))
            lane_arrival[d] = float(max(rates) if rates else 0.0)

        obs = [{
            "queues": queues,
            "density": 0.0,
            "accident": int(accidents[0]),
            "cluster_growth": cluster_feats.growth_rate,
            "avg_speed": lane_avg_speed,
            "arrival_rate": lane_arrival,
        }]

        # Select action and apply to first available traffic light, with confidences
        actions, modes, confs = ctrl.select_action_with_modes_and_conf(float(step), obs, cluster_feats.accident_cluster_size)
        tl_ids = list(traci.trafficlight.getIDList())
        if tl_ids:
            tl_id = cfg.TLS_ID if getattr(cfg, "TLS_ID", None) else tl_ids[0]
            action = actions[0]
            # Map action to phase dynamically and avoid stuck phases
            ns_green = int(action.get("N", 0)) or int(action.get("S", 0))
            phase_index = 0 if ns_green else 2
            try:
                current_phase = traci.trafficlight.getPhase(tl_id)
            except Exception:
                current_phase = phase_index
            if current_phase == phase_index:
                # keep running but allow duration to adjust rather than being stuck
                pass
            try:
                traci.trafficlight.setPhase(tl_id, phase_index)
            except Exception:
                pass
            # Duration scaled by queue imbalance (5..60s) with cluster priority and blockage-aware service
            ns = queues["N"] + queues["S"]
            ew = queues["E"] + queues["W"]
            imbalance = max(0.0, abs(ns - ew))
            dur = int(min(60, max(5, 10 + 2 * imbalance)))
            traci.trafficlight.setPhaseDuration(tl_id, dur)

            if step % 20 == 0:
                c = confs[0] if confs else {"c_rl":0.0,"c_ens":0.0,"c_lstm":0.0,"c_st":0.0,"alpha":0.0}
                # Accident probability proxy from queues and growth
                qsum = sum(queues.values())
                pacc = accident_probability(qsum, RiskFeatures(weather_severity=0.2, congestion_growth_rate=cluster_feats.growth_rate))
                avg_commute = (sum(trip_times)/len(trip_times)) if trip_times else 0.0
                print(f"Step {step}: mode={modes[0]} tl={tl_id} phase={phase_index} dur={dur}s Q={queues} acc={accidents[0]} p_acc={pacc:.3f} clusters={cluster_feats.num_accident_clusters}/{cluster_feats.accident_cluster_size} g={cluster_feats.growth_rate:.2f} cRL={c['c_rl']:.2f} cENS={c['c_ens']:.2f} (alpha={c['alpha']:.2f} cLSTM={c['c_lstm']:.2f} cST={c['c_st']:.2f}) avg_commute={avg_commute:.1f}s n_trips={len(trip_times)}")
                # also write commute avg to CSV if the offline recorder is used in SUMO mode in future
        else:
            if step % 60 == 0:
                print("No traffic lights found in SUMO network.")

        step += cfg.CONTROL_INTERVAL

    traci.close()
    print("Simulation completed.")


if __name__ == "__main__":
    run_simulation()


