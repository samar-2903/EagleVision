import argparse
import csv
import os
import time
from collections import deque
from typing import Deque, List, Tuple

import numpy as np

import config as cfg
from dqn_agent import DQNAgent
from sumo_env import SumoEnv


ACTION_LABELS = {
    0: "NS_GREEN",
    1: "EW_GREEN",
    2: "EXTEND",
    3: "SHORT_PRI",
}

Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


def _format_worst_tls(step_info: dict, top_k: int = 3) -> str:
    tls_rows = sorted(
        step_info["tls"].items(),
        key=lambda item: item[1]["total_queue"],
        reverse=True,
    )
    return ", ".join(
        f"{tid}:{metrics['total_queue']:.0f}" for tid, metrics in tls_rows[:top_k]
    )


def train(
    use_gui: bool = cfg.TRAIN_USE_GUI,
    step_log: bool = cfg.TRAIN_STEP_LOG,
    step_log_interval: int = cfg.TRAIN_STEP_LOG_INTERVAL,
    step_sleep_s: float = cfg.TRAIN_STEP_SLEEP_S,
    num_episodes: int = cfg.NUM_EPISODES,
):
    print("=" * 72)
    print("  EagleVision - Shared-Policy Multi-Signal DQN Training")
    print("=" * 72)
    print(f"  Episodes:             {num_episodes}")
    print(f"  Steps/episode:        {cfg.SIMULATION_STEPS}")
    print(f"  Control interval:     {cfg.CONTROL_INTERVAL}")
    print(f"  State dim:            {cfg.STATE_DIM}")
    print(f"  Actions per TLS:      {cfg.NUM_ACTIONS}")
    print(f"  Batch size:           {cfg.BATCH_SIZE}")
    print(f"  Replay cap:           {cfg.REPLAY_CAPACITY}")
    print(f"  Learn updates/step:   {cfg.LEARN_UPDATES_PER_STEP}")
    print(f"  GUI mode:             {use_gui}")
    print(f"  Step log:             {step_log}")
    print("=" * 72)

    env = SumoEnv(use_gui=use_gui)
    agent = DQNAgent()
    agent.load(cfg.MODEL_SAVE_PATH)

    os.makedirs("logs", exist_ok=True)
    log_path = "logs/training_log.csv"
    log_file = open(log_path, "w", newline="")

    try:
        writer = csv.writer(log_file)
        writer.writerow([
            "episode",
            "total_reward",
            "avg_reward_10",
            "epsilon",
            "control_steps",
            "avg_global_queue",
            "accident_events",
            "loss_avg",
            "controlled_tls",
        ])

        episode_rewards: List[float] = []

        for ep in range(1, num_episodes + 1):
            ep_start = time.time()
            states = env.reset()
            tls_ids = list(states.keys())
            recent_batches: Deque[List[Transition]] = deque(
                maxlen=cfg.PRE_JAM_TRANSITION_WINDOW
            )

            ep_reward = 0.0
            ep_losses: List[float] = []
            ep_global_queues: List[float] = []
            ep_accident_events = 0
            step = 0

            print(
                f"\n[EP {ep:03d}] reset complete | epsilon={agent.epsilon:.4f} "
                f"| controlled_tls={len(tls_ids)}"
            )

            done = False
            while not done:
                actions = {tid: agent.select_action(states[tid]) for tid in tls_ids}
                next_states, rewards, done, info = env.step(actions)

                batch: List[Transition] = []
                step_reward = 0.0
                for tid in tls_ids:
                    reward = rewards[tid]
                    transition = (states[tid], actions[tid], reward, next_states[tid], done)
                    agent.remember(*transition)
                    batch.append(transition)
                    step_reward += reward

                recent_batches.append(batch)

                if info["jam_detected"] and recent_batches:
                    for _ in range(cfg.PRE_JAM_REPLAY_MULTIPLIER):
                        for old_batch in recent_batches:
                            for transition in old_batch:
                                agent.remember(*transition)

                for _ in range(cfg.LEARN_UPDATES_PER_STEP):
                    loss = agent.learn()
                    if loss > 0:
                        ep_losses.append(loss)

                agent.decay_epsilon()

                ep_reward += step_reward
                ep_global_queues.append(info["global_queue"])
                ep_accident_events += len(info["accident_events"])
                step += 1

                if info["accident_events"]:
                    for event in info["accident_events"]:
                        print(
                            f"[EP {ep:03d} | step {step:04d}] ACCIDENT "
                            f"tls={event['tls_id']} lane={event['lane_id']} "
                            f"risk={event['risk_score']:.3f} "
                            f"clear_in={event['clearance_seconds']:.1f}s"
                        )

                if info["cleared_events"]:
                    for event in info["cleared_events"]:
                        print(
                            f"[EP {ep:03d} | step {step:04d}] ACCIDENT CLEARED "
                            f"tls={event['tls_id']} "
                            f"duration={event['clearance_seconds']:.1f}s"
                        )

                if info["jam_mitigation_activated"]:
                    print(
                        f"[EP {ep:03d} | step {step:04d}] JAM MITIGATION ON "
                        f"global_queue={info['global_queue']:.1f} "
                        f"keep_prob={info['demand_keep_prob']:.2f}"
                    )

                if info["jam_mitigation_relaxed"]:
                    print(
                        f"[EP {ep:03d} | step {step:04d}] JAM MITIGATION RELAXED "
                        f"global_queue={info['global_queue']:.1f} "
                        f"keep_prob={info['demand_keep_prob']:.2f}"
                    )

                if step_log and (step % max(1, step_log_interval) == 0):
                    print(
                        f"[EP {ep:03d} | step {step:04d} | sim {info['sim_time']:6.1f}s] "
                        f"reward={step_reward:9.3f} "
                        f"global_queue={info['global_queue']:7.1f} "
                        f"avg_delay={info['global_avg_delay']:8.2f} "
                        f"jam={int(info['jam_active'])} "
                        f"keep={info['demand_keep_prob']:.2f} "
                        f"removed={info['vehicles_removed']:4d} "
                        f"eps={agent.epsilon:.4f} "
                        f"worst=[{_format_worst_tls(info)}]"
                    )
                    if info["network_empty"]:
                        print(
                            f"[EP {ep:03d}] network empty; ending episode early "
                            f"at sim t={info['sim_time']:.1f}s"
                        )
                    if step_sleep_s > 0:
                        time.sleep(step_sleep_s)

                states = next_states

            episode_rewards.append(ep_reward)
            ep_duration = time.time() - ep_start
            avg_queue = float(np.mean(ep_global_queues)) if ep_global_queues else 0.0
            avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
            avg_reward_10 = float(np.mean(episode_rewards[-10:]))

            writer.writerow([
                ep,
                round(ep_reward, 2),
                round(avg_reward_10, 2),
                round(agent.epsilon, 4),
                step,
                round(avg_queue, 2),
                ep_accident_events,
                round(avg_loss, 5),
                len(tls_ids),
            ])
            log_file.flush()

            print(
                f"[EP {ep:03d}] done | total_reward={ep_reward:10.2f} | "
                f"avg10={avg_reward_10:10.2f} | avg_global_queue={avg_queue:7.2f} | "
                f"accident_events={ep_accident_events} | loss={avg_loss:.5f} | "
                f"t={ep_duration:.1f}s"
            )

            if ep % cfg.LOG_FREQ == 0:
                print(
                    f"Summary {ep:4d}/{num_episodes} | "
                    f"R={ep_reward:10.1f} | Ravg10={avg_reward_10:10.1f} | "
                    f"eps={agent.epsilon:.3f} | Qavg={avg_queue:.1f} | "
                    f"acc_events={ep_accident_events} | loss={avg_loss:.5f}"
                )

            if ep % cfg.SAVE_FREQ == 0:
                agent.save(cfg.MODEL_SAVE_PATH)

        agent.save(cfg.MODEL_SAVE_PATH)
        print("\nTraining complete.")
        print(f"Best episode reward: {max(episode_rewards):.1f}")
        print(f"Final epsilon: {agent.epsilon:.4f}")
        print(f"Log saved to: {log_path}")
        print(f"Model saved to: {cfg.MODEL_SAVE_PATH}")
    finally:
        env.close()
        log_file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the shared-policy multi-signal DQN controller."
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run training with the SUMO GUI enabled.",
    )
    parser.add_argument(
        "--step-log",
        action="store_true",
        help="Print one log line per control step.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=None,
        help="Pause this many seconds after each logged control step.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=cfg.NUM_EPISODES,
        help="Override the number of training episodes.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Convenience mode: enable GUI, step logs, and a short sleep.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    use_gui = cfg.TRAIN_USE_GUI
    step_log = cfg.TRAIN_STEP_LOG
    step_sleep_s = cfg.TRAIN_STEP_SLEEP_S

    if args.watch:
        use_gui = True
        step_log = True
        step_sleep_s = 0.10 if args.sleep is None else args.sleep
    else:
        if args.gui:
            use_gui = True
        if args.step_log:
            step_log = True
        if args.sleep is not None:
            step_sleep_s = args.sleep

    train(
        use_gui=use_gui,
        step_log=step_log,
        step_sleep_s=step_sleep_s,
        num_episodes=args.episodes,
    )
