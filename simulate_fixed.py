# =============================================================================
# simulate_fixed.py — Run a FIXED-TIMING controller as the baseline.
#
# WHY DO WE NEED A BASELINE?
# "Our model achieves reward -250" means nothing on its own.
# "Our model achieves -250 vs fixed timing's -480" = 48% improvement. That means something.
# The baseline is what every paper/project compares against.
#
# Fixed-cycle timing is the real-world default at most intersections:
# 30s NS green → 5s yellow → 30s EW green → 5s yellow → repeat
# No adaptation, no intelligence, just a timer.
# =============================================================================

import traci
import csv
import os
import time
import numpy as np
from typing import List, Dict
from collections import defaultdict, deque

import config as cfg
from reward import compute_reward, compute_delay_from_queue
from clustering import OpticsClustering
from risk import RiskFeatures, accident_probability
from accident_manager import AccidentManager


def run_fixed(output_csv: str = "logs/fixed_results.csv", use_gui: bool = False):
    print("=" * 50)
    print("  EagleVision — Fixed Timing Baseline")
    print("=" * 50)

    sumo_binary = "sumo-gui" if use_gui else "sumo"
    sumo_cmd = [
        sumo_binary,
        "-c", cfg.SUMO_CFG,
        "--no-step-log",
        "--waiting-time-memory", "100",
        "--seed", str(cfg.SEED),
    ]

    try:
        traci.close()
    except Exception:
        pass

    traci.start(sumo_cmd)

    # Get traffic light
    tl_ids = list(traci.trafficlight.getIDList())
    if not tl_ids:
        raise RuntimeError("No traffic lights found!")
    tl_id = cfg.TLS_ID if cfg.TLS_ID else tl_ids[0]

    # Set up fixed cycle: alternating 30s NS / 30s EW
    # Phase 0 = NS green, Phase 2 = EW green (standard SUMO 4-way)
    CYCLE_NS  = cfg.DEFAULT_GREEN   # 30s
    CYCLE_EW  = cfg.DEFAULT_GREEN   # 30s

    optics  = OpticsClustering()
    acc_mgr = AccidentManager()

    os.makedirs("logs", exist_ok=True)
    results = []

    step = 0
    total_reward = 0.0
    cycle_timer  = 0       # counts steps since last phase switch
    ns_phase     = True    # True = NS green, False = EW green

    # Set initial phase
    traci.trafficlight.setPhase(tl_id, 0)
    traci.trafficlight.setPhaseDuration(tl_id, CYCLE_NS)

    while step < cfg.SIMULATION_STEPS:
        traci.simulationStep()
        step += 1
        cycle_timer += 1

        # ── Fixed timing logic ─────────────────────────────────────────────
        # Switch phase when current cycle duration expires
        if ns_phase and cycle_timer >= CYCLE_NS:
            traci.trafficlight.setPhase(tl_id, 2)    # EW green
            traci.trafficlight.setPhaseDuration(tl_id, CYCLE_EW)
            ns_phase    = False
            cycle_timer = 0

        elif not ns_phase and cycle_timer >= CYCLE_EW:
            traci.trafficlight.setPhase(tl_id, 0)    # NS green
            traci.trafficlight.setPhaseDuration(tl_id, CYCLE_NS)
            ns_phase    = True
            cycle_timer = 0

        # ── Collect metrics every CONTROL_INTERVAL steps ──────────────────
        if step % cfg.CONTROL_INTERVAL == 0:
            # Queue counts
            queues = {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0}
            try:
                for vid in traci.vehicle.getIDList():
                    spd  = traci.vehicle.getSpeed(vid)
                    lane = traci.vehicle.getLaneID(vid).upper()
                    if spd < 0.5:
                        for d in ("N", "S", "E", "W"):
                            if d in lane:
                                queues[d] += 1.0
                                break
            except Exception:
                pass

            total_q = sum(queues.values())

            # Clustering
            pts_list = []
            try:
                for vid in traci.vehicle.getIDList():
                    if traci.vehicle.getSpeed(vid) < 0.5:
                        x, y = traci.vehicle.getPosition(vid)
                        pts_list.append([float(x), float(y)])
            except Exception:
                pass

            import numpy as _np
            pts = _np.array(pts_list, dtype=float) if pts_list else _np.zeros((0, 2))
            cf  = optics.run(pts)

            # Accident
            acc_mgr.update(0, float(step))
            accident = acc_mgr.get_acc_flag(0)

            # Reward
            avg_delay = compute_delay_from_queue(total_q)
            reward = compute_reward(
                [total_q], [avg_delay], [accident], [cf.growth_rate]
            )
            total_reward += reward

            results.append({
                "step":           step,
                "reward":         reward,
                "total_queue":    total_q,
                "q_N":            queues["N"],
                "q_S":            queues["S"],
                "q_E":            queues["E"],
                "q_W":            queues["W"],
                "accident":       accident,
                "cluster_growth": cf.growth_rate,
                "mode":           "FIXED",
            })

    traci.close()

    # Write CSV
    if results:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        avg_q     = np.mean([r["total_queue"] for r in results])
        total_acc = sum(r["accident"] for r in results)
        print(f"✅ Done | Steps={step} | TotalReward={total_reward:.1f} | AvgQueue={avg_q:.1f} | Accidents={total_acc}")
        print(f"   Results → {output_csv}")

    return results


if __name__ == "__main__":
    run_fixed(use_gui=False)
