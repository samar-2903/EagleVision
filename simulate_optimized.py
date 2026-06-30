import csv
import os

import numpy as np

import config as cfg
from dqn_agent import DQNAgent
from sumo_env import SumoEnv


def run_optimized(
    output_csv: str = "logs/optimized_results.csv", use_gui: bool = False
):
    print("=" * 60)
    print("  EagleVision - Optimized Multi-Signal DQN Simulation")
    print("=" * 60)

    env = SumoEnv(use_gui=use_gui)
    agent = DQNAgent()
    agent.load(cfg.MODEL_SAVE_PATH)
    agent.epsilon = 0.0

    os.makedirs("logs", exist_ok=True)
    results = []

    states = env.reset()
    tls_ids = list(states.keys())
    done = False
    step = 0
    total_reward = 0.0

    while not done:
        actions = {tid: agent.select_action(states[tid]) for tid in tls_ids}
        next_states, rewards, done, info = env.step(actions)
        step_reward = float(sum(rewards.values()))
        total_reward += step_reward
        step += 1

        results.append(
            {
                "step": step,
                "reward": step_reward,
                "total_queue": info["global_queue"],
                "q_N": sum(metrics["queues"]["N"] for metrics in info["tls"].values()),
                "q_S": sum(metrics["queues"]["S"] for metrics in info["tls"].values()),
                "q_E": sum(metrics["queues"]["E"] for metrics in info["tls"].values()),
                "q_W": sum(metrics["queues"]["W"] for metrics in info["tls"].values()),
                "accident": len(info["accident_events"]),
                "cluster_growth": float(
                    np.mean([metrics["cluster_growth"] for metrics in info["tls"].values()])
                    if info["tls"]
                    else 0.0
                ),
                "mode": "DQN",
            }
        )

        states = next_states

    env.close()

    if results:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        avg_q = np.mean([r["total_queue"] for r in results])
        total_acc = sum(r["accident"] for r in results)
        print(
            f"Done | Steps={step} | TotalReward={total_reward:.1f} | "
            f"AvgQueue={avg_q:.1f} | AccidentEvents={total_acc}"
        )
        print(f"Results -> {output_csv}")
    return results


if __name__ == "__main__":
    run_optimized(use_gui=False)
