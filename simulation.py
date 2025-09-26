from __future__ import annotations

import numpy as np
from typing import Dict, List

from traffic import TrafficNetwork, Intersection, IntersectionState
from risk import AccidentMDP, RiskFeatures
from clustering import OpticsClustering
from control import RLPolicy, GNNForecaster, HierarchicalController, fixed_cycle
from rewards import compute_reward
from data_logging import CsvRecorder
from accident_manager import AccidentManager
from service_model import ServiceModel


def build_sample_network(n: int) -> TrafficNetwork:
    inters: List[Intersection] = []
    for i in range(n):
        inters.append(
            Intersection(
                intersection_id=i,
                service_rates_green={"N": 0.7, "S": 0.7, "E": 0.7, "W": 0.7},
                service_rates_red={"N": 0.05, "S": 0.05, "E": 0.05, "W": 0.05},
                saturation_queue=300.0,
                state=IntersectionState({"N": 5.0, "S": 4.0, "E": 6.0, "W": 3.0}, density=0.2, accident_flag=0),
            )
        )

    C = np.zeros((n, n))
    for i in range(n-1):
        C[i, i+1] = 1.0
        C[i+1, i] = 0.4
    return TrafficNetwork(inters, C)


def run_simulation(steps: int = 300, dt_s: float = 1.0, degrade_threshold: int = 5, log_path: str | None = "simulation_log.csv") -> None:
    net = build_sample_network(4)
    risk = AccidentMDP(num_nodes=len(net.intersections))
    optics = OpticsClustering()
    ctrl = HierarchicalController(RLPolicy(), GNNForecaster(), degrade_threshold=degrade_threshold)
    recorder = CsvRecorder(log_path) if log_path else None
    acc_mgr = AccidentManager()
    service = ServiceModel(dt=dt_s)

    t_s = 0.0
    prolonged_failure_timer: float = 0.0
    prolonged_failure_limit: float = 300.0  # 5 minutes

    episode_return = 0.0
    # commute tracking (fixed vs model comparable)
    depart_time = [{} for _ in net.intersections]
    trip_times: list[float] = []

    for step in range(steps):
        # Risk features per node
        features = []
        for i in range(len(net.intersections)):
            qhist = net.queue_history[i]
            growth = 0.0
            if len(qhist) >= 2:
                growth = (qhist[-1] - qhist[-2]) / max(1.0, dt_s)
            features.append(RiskFeatures(weather_severity=0.2, congestion_growth_rate=growth))

        # Accident step
        accidents, cleared = risk.step(t_s, [inter.state.queues for inter in net.intersections], features)
        for i, inter in enumerate(net.intersections):
            # Only AccidentManager should toggle; mirror for state readout
            if accidents[i] == 1 and acc_mgr.get_acc_flag(i) == 0:
                acc_mgr.trigger_accident(i, t_s)
            inter.state.accident_flag = acc_mgr.get_acc_flag(i)
            upd = acc_mgr.update(i, t_s)
            if upd and upd[1]:
                # apply queue reduction on clearance
                for d in ("N","S","E","W"):
                    inter.state.queues[d] = max(0.0, inter.state.queues[d] * acc_mgr.r_clear)
        # Reset queues upon clearance
        if cleared:
            for i, dur in cleared:
                for d in ("N","S","E","W"):
                    net.intersections[i].state.queues[d] = max(0.0, net.intersections[i].state.queues[d] * 0.5)
            print(f"Accident cleared at nodes {[i for i,_ in cleared]} (T_clear s={[round(d,1) for _,d in cleared]})")

        # Accident points for clustering (toy: put a point at (i, queue))
        pts = []
        for i, inter in enumerate(net.intersections):
            if inter.state.accident_flag:
                pts.append([float(i), sum(inter.state.queues.values())])
        pts_arr = np.array(pts, dtype=float) if len(pts) > 0 else np.zeros((0, 2))
        cluster_feats = optics.run(pts_arr)

        # Build observations for controller
        obs = []
        for inter in net.intersections:
            obs.append({
                "queues": inter.state.queues,
                "density": inter.state.density,
                "accident": inter.state.accident_flag,
                "cluster_growth": cluster_feats.growth_rate,
            })

        # Determine if degraded and prolonged failure; log fallback
        # Always use RL/GNN switching (no degrade/fixed fallback)
        prolonged_failure_timer = 0.0
        actions, modes, confs = ctrl.select_action_with_modes_and_conf(t_s, obs, cluster_feats.accident_cluster_size)

        # Apply traffic step with blockage-aware service: override via service model
        arrivals = net.compute_arrival_rates_with_spillover(accident_flags=[acc_mgr.get_acc_flag(i) for i in range(len(net.intersections))])
        for i, inter in enumerate(net.intersections):
            action = actions[i]
            # convert action to green allocation seconds in a cycle
            ns_green = float(1 if action.get("N",0) or action.get("S",0) else 0)
            ew_green = float(1 if action.get("E",0) or action.get("W",0) else 0)
            # simplistic mapping: if NS green then g_ns=40,g_ew=20 else opposite
            C = 60.0
            g_ns = 40.0 if ns_green >= ew_green else 20.0
            block = acc_mgr.get_blockage(i)
            # use discrete served-min update per lane
            inter.step_discrete(dt=dt_s, arrival_rates=arrivals[i], action=action, blockage=block, sat_flow=service.sat_flow, cycle_len=C, green_alloc_s=g_ns)
        # update queue histories
        for i, inter in enumerate(net.intersections):
            total_q = sum(inter.state.queues.values())
            net.queue_history[i].append(total_q)
        t_s += dt_s

        # Estimate delays as queue length proxy / service rate avg (toy)
        avg_delays = []
        for inter in net.intersections:
            q = sum(inter.state.queues.values())
            avg_mu = 0.35  # rough average service
            avg_delays.append(q / max(1e-3, avg_mu))
        total_q = [sum(inter.state.queues.values()) for inter in net.intersections]
        r_t = compute_reward(total_q, avg_delays, accidents, [cluster_feats.growth_rate])
        episode_return += r_t

        if recorder:
            # log confidences for node 0
            c_rl0 = confs[0]["c_rl"] if confs else 0.0
            c_ens0 = confs[0]["c_ens"] if confs else 0.0
            c_lstm0 = confs[0]["c_lstm"] if confs else 0.0
            c_st0 = confs[0]["c_st"] if confs else 0.0
            alpha0 = confs[0]["alpha"] if confs else 0.0
            recorder.write_step(
                t_s=t_s,
                reward_t=r_t,
                episode_return=episode_return,
                totalQ=total_q,
                avg_delays=avg_delays,
                accidents=accidents,
                num_clusters=cluster_feats.num_accident_clusters,
                max_cluster=cluster_feats.accident_cluster_size,
                c_rl0=c_rl0,
                c_ens0=c_ens0,
                c_lstm0=c_lstm0,
                c_st0=c_st0,
                alpha0=alpha0,
            )

        if step % 20 == 0:
            # Risk score proxy per node based on queues and growth
            risk_scores = []
            for i in range(len(net.intersections)):
                q = total_q[i]
                growth = 0.0
                qhist = net.queue_history[i]
                if len(qhist) >= 2:
                    growth = (qhist[-1] - qhist[-2]) / max(1.0, dt_s)
                # logistic accident probability proxy aligned with risk.accident_probability
                from risk import RiskFeatures, accident_probability
                pacc = accident_probability(q, RiskFeatures(weather_severity=0.2, congestion_growth_rate=growth))
                risk_scores.append(float(0.5*q/50.0 + 0.5*max(0.0, growth)))
            # aggregate confidences (node 0 shown for brevity)
            if confs:
                c_rl0 = confs[0]["c_rl"]
                c_ens0 = confs[0]["c_ens"]
                c_lstm0 = confs[0]["c_lstm"]
                c_st0 = confs[0]["c_st"]
                alpha0 = confs[0]["alpha"]
            else:
                c_rl0 = c_ens0 = c_lstm0 = c_st0 = alpha0 = 0.0
            # proxy commute time from Little's Law D ~ Q/mu_avg
            avg_mu = 0.35
            commute_proxy = [q / max(1e-3, avg_mu) for q in total_q]
            print(f"t={t_s:.0f}s R={r_t:.2f} G={episode_return:.2f} Q={total_q} acc={accidents} clusters={cluster_feats.num_accident_clusters}/{cluster_feats.accident_cluster_size} g={cluster_feats.growth_rate:.2f} risk~={ [round(r,3) for r in risk_scores] } cRL={c_rl0:.2f} cENS={c_ens0:.2f} (alpha={alpha0:.2f} cLSTM={c_lstm0:.2f} cST={c_st0:.2f}) commute_proxy={ [round(x,1) for x in commute_proxy] }")


if __name__ == "__main__":
    run_simulation()


