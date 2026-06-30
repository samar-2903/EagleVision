import argparse
import csv
import io
import os
import time
from collections import deque
from contextlib import redirect_stdout
from typing import Deque, List, Tuple

import numpy as np

import config as cfg
from dqn_agent import DQNAgent
from sumo_env import SumoEnv


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


def _quiet_call(func, *args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _heartbeat(
    start_time: float,
    episode: int,
    total_episodes: int,
    control_step: int,
    max_control_steps: int,
    agent: DQNAgent,
    replay_size: int,
    episode_reward: float,
    episode_avg_queue: float,
) -> None:
    elapsed = time.monotonic() - start_time
    print(
        f"[{elapsed:7.1f}s] working=yes "
        f"episode={episode}/{total_episodes} "
        f"step={control_step}/{max_control_steps} "
        f"epsilon={agent.epsilon:.4f} "
        f"replay={replay_size} "
        f"reward={episode_reward:.1f} "
        f"avg_queue={episode_avg_queue:.1f}",
        flush=True,
    )


def train_quiet(
    episodes: int = 10,
    traffic_scale: float = 1.0,
    heartbeat_s: float = 30.0,
    final_checkpoint: str = cfg.MODEL_SAVE_PATH,
) -> None:
    checkpoint_dir = os.path.dirname(final_checkpoint) or "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = SumoEnv(use_gui=False, traffic_scale=traffic_scale)
    agent = _quiet_call(DQNAgent)

    log_path = os.path.join("logs", "alternativet_training_log.csv")
    max_control_steps = int(np.ceil(cfg.SIMULATION_STEPS / max(1, cfg.CONTROL_INTERVAL)))
    start_time = time.monotonic()
    last_heartbeat = start_time

    print(
        f"training_start device={agent.device} episodes={episodes} "
        f"sim_steps={cfg.SIMULATION_STEPS} scale={traffic_scale}",
        flush=True,
    )

    try:
        with open(log_path, "w", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(
                [
                    "episode",
                    "control_steps",
                    "total_reward",
                    "avg_global_queue",
                    "accident_events",
                    "avg_loss",
                    "epsilon",
                    "checkpoint_path",
                ]
            )

            for episode in range(1, episodes + 1):
                states = _quiet_call(env.reset)
                tls_ids = list(states.keys())
                recent_batches: Deque[List[Transition]] = deque(
                    maxlen=cfg.PRE_JAM_TRANSITION_WINDOW
                )

                done = False
                control_step = 0
                episode_reward = 0.0
                episode_losses: List[float] = []
                episode_global_queues: List[float] = []
                episode_accident_events = 0

                while not done:
                    actions = {tid: agent.select_action(states[tid]) for tid in tls_ids}
                    next_states, rewards, done, info = _quiet_call(env.step, actions)

                    batch: List[Transition] = []
                    step_reward = 0.0
                    for tid in tls_ids:
                        reward = float(rewards[tid])
                        transition = (
                            states[tid],
                            actions[tid],
                            reward,
                            next_states[tid],
                            done,
                        )
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
                        if loss > 0.0:
                            episode_losses.append(loss)

                    agent.decay_epsilon()

                    control_step += 1
                    episode_reward += step_reward
                    episode_global_queues.append(float(info["global_queue"]))
                    episode_accident_events += len(info["accident_events"])
                    states = next_states

                    now = time.monotonic()
                    if now - last_heartbeat >= heartbeat_s:
                        avg_queue = (
                            float(np.mean(episode_global_queues))
                            if episode_global_queues
                            else 0.0
                        )
                        _heartbeat(
                            start_time=start_time,
                            episode=episode,
                            total_episodes=episodes,
                            control_step=control_step,
                            max_control_steps=max_control_steps,
                            agent=agent,
                            replay_size=len(agent.memory),
                            episode_reward=episode_reward,
                            episode_avg_queue=avg_queue,
                        )
                        last_heartbeat = now

                avg_queue = float(np.mean(episode_global_queues)) if episode_global_queues else 0.0
                avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
                episode_checkpoint = os.path.join(
                    checkpoint_dir,
                    f"dqn_checkpoint_ep{episode:02d}.pt",
                )

                _quiet_call(agent.save, episode_checkpoint)
                writer.writerow(
                    [
                        episode,
                        control_step,
                        round(episode_reward, 2),
                        round(avg_queue, 2),
                        episode_accident_events,
                        round(avg_loss, 5),
                        round(agent.epsilon, 4),
                        episode_checkpoint,
                    ]
                )
                log_file.flush()

        _quiet_call(agent.save, final_checkpoint)
    finally:
        env.close()

    total_elapsed = time.monotonic() - start_time
    print(
        f"training_complete working=yes episodes={episodes}/{episodes} "
        f"elapsed={total_elapsed:.1f}s final_checkpoint={final_checkpoint} "
        f"log=logs/alternativet_training_log.csv",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quiet GPU-friendly trainer for EagleVision."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to train. Default: 10",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="SUMO traffic scale. Default: 1.0",
    )
    parser.add_argument(
        "--heartbeat",
        type=float,
        default=30.0,
        help="Seconds between progress heartbeats. Default: 30",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=cfg.MODEL_SAVE_PATH,
        help="Final checkpoint output path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_quiet(
        episodes=args.episodes,
        traffic_scale=args.scale,
        heartbeat_s=args.heartbeat,
        final_checkpoint=args.checkpoint,
    )
