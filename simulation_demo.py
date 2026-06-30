import argparse
import csv
import io
import os
import time
from contextlib import redirect_stdout
from typing import Dict, List, Tuple

import numpy as np
import torch

import config as cfg
from dqn_agent import DQNAgent
from sumo_env import SumoEnv


FEATURE_NAMES = [
    "Q_N",
    "Q_S",
    "Q_E",
    "Q_W",
    "V_N",
    "V_S",
    "V_E",
    "V_W",
    "A_N",
    "A_S",
    "A_E",
    "A_W",
    "cluster_growth",
    "accident_flag",
    "risk_score",
]

ACTION_LABELS = {
    0: "NS_GREEN",
    1: "EW_GREEN",
    2: "EXTEND",
    3: "SHORT_PRI",
}


def _quiet_call(func, *args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _scale_tag(scale: float) -> str:
    return str(scale).replace(".", "p")


def _fixed_action_for_step(step_idx: int) -> int:
    block = max(1, int(round(cfg.DEFAULT_GREEN / max(1, cfg.CONTROL_INTERVAL))))
    return 0 if (step_idx // block) % 2 == 0 else 1


def _model_action_with_attribution(
    agent: DQNAgent,
    state: np.ndarray,
    top_k: int = 3,
) -> Tuple[int, np.ndarray, float, List[Tuple[str, float, float]]]:
    state_t = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    state_t.requires_grad_(True)

    agent.online_net.zero_grad(set_to_none=True)
    q_values = agent.online_net(state_t)
    action = int(torch.argmax(q_values, dim=1).item())
    chosen_q = q_values[0, action]
    chosen_q.backward()

    grad = state_t.grad.detach().cpu().numpy()[0]
    values = state_t.detach().cpu().numpy()[0]
    salience = np.abs(grad * values)

    top_idx = np.argsort(salience)[-top_k:][::-1]
    features = [
        (FEATURE_NAMES[idx], float(values[idx]), float(salience[idx]))
        for idx in top_idx
    ]

    sorted_q = np.sort(q_values.detach().cpu().numpy()[0])[::-1]
    q_margin = float(sorted_q[0] - sorted_q[1]) if len(sorted_q) > 1 else float(sorted_q[0])
    return action, q_values.detach().cpu().numpy()[0], q_margin, features


def _format_features(features: List[Tuple[str, float, float]]) -> str:
    return ", ".join(f"{name}={value:.2f}" for name, value, _ in features)


def _summarize_run(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return {
            "control_steps": 0,
            "avg_global_queue": 0.0,
            "max_global_queue": 0.0,
            "avg_global_delay": 0.0,
            "total_reward": 0.0,
            "accident_events": 0.0,
            "active_accident_steps": 0.0,
            "jam_steps": 0.0,
        }

    return {
        "control_steps": len(rows),
        "avg_global_queue": float(np.mean([row["global_queue"] for row in rows])),
        "max_global_queue": float(np.max([row["global_queue"] for row in rows])),
        "avg_global_delay": float(np.mean([row["global_avg_delay"] for row in rows])),
        "total_reward": float(np.sum([row["reward"] for row in rows])),
        "accident_events": float(np.sum([row["accident_events"] for row in rows])),
        "active_accident_steps": float(np.sum([row["active_accidents"] for row in rows])),
        "jam_steps": float(np.sum([row["jam_active"] for row in rows])),
    }


def _write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _extract_key_changes(
    fixed_tls_rows: List[Dict],
    model_tls_rows: List[Dict],
    top_k: int,
) -> List[Dict]:
    fixed_index = {(row["step"], row["tls_id"]): row for row in fixed_tls_rows}
    key_changes: List[Dict] = []

    for model_row in model_tls_rows:
        key = (model_row["step"], model_row["tls_id"])
        fixed_row = fixed_index.get(key)
        if fixed_row is None:
            continue

        queue_adv = float(fixed_row["local_queue_after"] - model_row["local_queue_after"])
        risk_adv = float(fixed_row["risk_after"] - model_row["risk_after"])
        reward_adv = float(model_row["reward"] - fixed_row["reward"])
        accident_adv = float(fixed_row["accident_after"] - model_row["accident_after"])
        clear_bonus = 1.0 if model_row["accident_cleared"] else 0.0

        score = (
            queue_adv
            + 15.0 * max(0.0, risk_adv)
            + 20.0 * max(0.0, accident_adv)
            + 10.0 * clear_bonus
            + 0.5 * max(0.0, reward_adv)
        )
        if score <= 0.0:
            continue

        entry = dict(model_row)
        entry["impact_score"] = score
        entry["fixed_queue_after"] = float(fixed_row["local_queue_after"])
        entry["fixed_risk_after"] = float(fixed_row["risk_after"])
        entry["fixed_reward"] = float(fixed_row["reward"])
        key_changes.append(entry)

    key_changes.sort(key=lambda row: row["impact_score"], reverse=True)
    return key_changes[:top_k]


def run_controller(
    mode: str,
    traffic_scale: float,
    use_gui: bool,
    checkpoint_path: str,
    fixed_green: int,
    allow_untrained: bool,
    sim_steps: int,
    step_pause_s: float = 0.0,
    explain_model: bool = True,
    explain_top_tls: int | None = None,
) -> Tuple[List[Dict], List[Dict], Dict[str, float]]:
    env = SumoEnv(use_gui=use_gui, traffic_scale=traffic_scale)
    agent = None
    original_sim_steps = cfg.SIMULATION_STEPS
    cfg.SIMULATION_STEPS = int(sim_steps)

    if mode == "model":
        if not os.path.exists(checkpoint_path):
            if not allow_untrained:
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_path}. "
                    "Train first or rerun with --allow-untrained."
                )
        agent = _quiet_call(DQNAgent)
        if os.path.exists(checkpoint_path):
            _quiet_call(agent.load, checkpoint_path)
        else:
            print("[DEMO] No checkpoint found. Running with an untrained network.")
        agent.epsilon = 0.0

    global_rows: List[Dict] = []
    tls_rows: List[Dict] = []

    try:
        states = _quiet_call(env.reset)
        tls_ids = list(states.keys())
        done = False
        step_idx = 0

        while not done:
            step_idx += 1
            before_obs = env._collect_observations()

            if mode == "fixed":
                fixed_action = _fixed_action_for_step(step_idx - 1)
                actions = {tid: fixed_action for tid in tls_ids}
                analyses = {tid: {} for tid in tls_ids}
                next_states, rewards, done, info = _quiet_call(
                    env.step_fixed_time,
                    actions,
                    fixed_duration=fixed_green,
                )
            else:
                actions = {}
                analyses = {}
                explain_tls_ids = set(tls_ids)
                if explain_model and explain_top_tls is not None:
                    ranked = sorted(
                        tls_ids,
                        key=lambda tid: before_obs[tid]["total_queue"],
                        reverse=True,
                    )
                    explain_tls_ids = set(ranked[: max(1, explain_top_tls)])
                for tid in tls_ids:
                    if explain_model and tid in explain_tls_ids:
                        action, q_values, q_margin, features = _model_action_with_attribution(
                            agent,
                            states[tid],
                        )
                    else:
                        action = agent.select_action(states[tid])
                        q_values = np.array([], dtype=np.float32)
                        q_margin = 0.0
                        features = []
                    actions[tid] = action
                    analyses[tid] = {
                        "q_values": q_values,
                        "q_margin": q_margin,
                        "salient_features": features,
                    }
                next_states, rewards, done, info = _quiet_call(env.step, actions)

            accident_triggered = {event["tls_id"] for event in info["accident_events"]}
            accident_cleared = {event["tls_id"] for event in info["cleared_events"]}
            step_reward = float(sum(rewards.values()))

            global_rows.append(
                {
                    "mode": mode.upper(),
                    "step": step_idx,
                    "sim_time": float(info["sim_time"]),
                    "reward": step_reward,
                    "global_queue": float(info["global_queue"]),
                    "global_avg_delay": float(info["global_avg_delay"]),
                    "accident_events": len(info["accident_events"]),
                    "active_accidents": int(sum(metrics["accident"] for metrics in info["tls"].values())),
                    "jam_active": int(info["jam_active"]),
                    "demand_keep_prob": float(info["demand_keep_prob"]),
                    "vehicles_removed": int(info["vehicles_removed"]),
                }
            )

            for tid in tls_ids:
                before = before_obs[tid]
                after = info["tls"][tid]
                analysis = analyses.get(tid, {})
                salient_features = analysis.get("salient_features", [])
                tls_rows.append(
                    {
                        "mode": mode.upper(),
                        "step": step_idx,
                        "sim_time": float(info["sim_time"]),
                        "tls_id": tid,
                        "action": int(actions[tid]),
                        "action_label": ACTION_LABELS[int(actions[tid])],
                        "reward": float(rewards[tid]),
                        "local_queue_before": float(before["total_queue"]),
                        "local_queue_after": float(after["total_queue"]),
                        "risk_before": float(before["risk_score"]),
                        "risk_after": float(after["risk_score"]),
                        "accident_after": float(after["accident"]),
                        "accident_triggered": int(tid in accident_triggered),
                        "accident_cleared": int(tid in accident_cleared),
                        "q_margin": float(analysis.get("q_margin", 0.0)),
                        "top_features": _format_features(salient_features),
                    }
                )

            states = next_states
            if step_pause_s > 0:
                time.sleep(step_pause_s)

    finally:
        env.close()
        cfg.SIMULATION_STEPS = original_sim_steps

    return global_rows, tls_rows, _summarize_run(global_rows)


def print_summary(fixed_summary: Dict[str, float], model_summary: Dict[str, float]) -> None:
    print("\n" + "=" * 72)
    print("Simulation Summary")
    print("=" * 72)
    print(
        f"{'Metric':<24}{'Fixed':>14}{'Model':>14}{'Delta':>14}"
    )
    print("-" * 72)

    rows = [
        ("Control steps", fixed_summary["control_steps"], model_summary["control_steps"], False),
        ("Avg global queue", fixed_summary["avg_global_queue"], model_summary["avg_global_queue"], True),
        ("Max global queue", fixed_summary["max_global_queue"], model_summary["max_global_queue"], True),
        ("Avg global delay", fixed_summary["avg_global_delay"], model_summary["avg_global_delay"], True),
        ("Total reward", fixed_summary["total_reward"], model_summary["total_reward"], False),
        ("Accident events", fixed_summary["accident_events"], model_summary["accident_events"], True),
        ("Active accident steps", fixed_summary["active_accident_steps"], model_summary["active_accident_steps"], True),
        ("Jam steps", fixed_summary["jam_steps"], model_summary["jam_steps"], True),
    ]

    for name, fixed_value, model_value, lower_is_better in rows:
        delta = model_value - fixed_value
        if lower_is_better:
            status = "better" if model_value < fixed_value else "worse"
        else:
            status = "better" if model_value > fixed_value else "worse"
        print(f"{name:<24}{fixed_value:>14.2f}{model_value:>14.2f}{delta:>11.2f}  {status}")


def print_key_changes(key_changes: List[Dict]) -> None:
    print("\n" + "=" * 72)
    print("Highest-Impact Model Decisions")
    print("=" * 72)

    if not key_changes:
        print("No positive model-vs-fixed trajectory changes were detected.")
        return

    for row in key_changes:
        accident_note = ""
        if row["accident_cleared"]:
            accident_note = " | accident cleared"

        print(
            f"[step {int(row['step']):03d} | t={row['sim_time']:.1f}s] "
            f"tls={row['tls_id']} action={row['action_label']} "
            f"queue {row['local_queue_before']:.0f}->{row['local_queue_after']:.0f} "
            f"(fixed {row['fixed_queue_after']:.0f}) "
            f"risk {row['risk_before']:.2f}->{row['risk_after']:.2f} "
            f"(fixed {row['fixed_risk_after']:.2f}) "
            f"reward {row['reward']:.2f} "
            f"margin={row['q_margin']:.3f}{accident_note}"
        )
        print(f"  model drivers: {row['top_features']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a side-by-side EagleVision demo: fixed schedule vs trained DQN."
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="SUMO demand scaling factor. Default: 0.5",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run the simulation with the SUMO GUI enabled.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=cfg.MODEL_SAVE_PATH,
        help="Path to the trained checkpoint.",
    )
    parser.add_argument(
        "--fixed-green",
        type=int,
        default=cfg.DEFAULT_GREEN,
        help="Fixed green duration in seconds for the non-model baseline.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of high-impact model decisions to print.",
    )
    parser.add_argument(
        "--allow-untrained",
        action="store_true",
        help="Run the model path with random weights if no checkpoint exists.",
    )
    parser.add_argument(
        "--sim-steps",
        type=int,
        default=600,
        help="Simulated seconds per run. Default: 600",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 72)
    print("EagleVision Demonstration")
    print("=" * 72)
    print(f"Traffic scale:   {args.scale}")
    print(f"Sim steps:       {args.sim_steps}")
    print(f"Fixed baseline:  {args.fixed_green}s alternating NS/EW")
    print(f"Checkpoint:      {args.checkpoint}")

    fixed_rows, fixed_tls_rows, fixed_summary = run_controller(
        mode="fixed",
        traffic_scale=args.scale,
        use_gui=args.gui,
        checkpoint_path=args.checkpoint,
        fixed_green=args.fixed_green,
        allow_untrained=args.allow_untrained,
        sim_steps=args.sim_steps,
    )

    model_rows, model_tls_rows, model_summary = run_controller(
        mode="model",
        traffic_scale=args.scale,
        use_gui=args.gui,
        checkpoint_path=args.checkpoint,
        fixed_green=args.fixed_green,
        allow_untrained=args.allow_untrained,
        sim_steps=args.sim_steps,
    )

    tag = _scale_tag(args.scale)
    _write_csv(f"logs/simulation_demo_fixed_scale_{tag}.csv", fixed_rows)
    _write_csv(f"logs/simulation_demo_model_scale_{tag}.csv", model_rows)

    print_summary(fixed_summary, model_summary)
    key_changes = _extract_key_changes(fixed_tls_rows, model_tls_rows, top_k=args.top_k)
    print_key_changes(key_changes)

    print("\nSaved:")
    print(f"  logs/simulation_demo_fixed_scale_{tag}.csv")
    print(f"  logs/simulation_demo_model_scale_{tag}.csv")


if __name__ == "__main__":
    main()
