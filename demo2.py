import argparse
import time

import config as cfg
from simulation_demo import (
    _extract_key_changes,
    _scale_tag,
    _write_csv,
    print_key_changes,
    print_summary,
    run_controller,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GUI presentation demo for EagleVision: fixed baseline vs trained model."
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="SUMO demand scaling factor. Default: 0.5",
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
        help="Fixed green duration in seconds for the baseline.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of high-impact model decisions to print.",
    )
    parser.add_argument(
        "--sim-steps",
        type=int,
        default=600,
        help="Simulated seconds per run. Default: 600",
    )
    parser.add_argument(
        "--step-pause",
        type=float,
        default=0.10,
        help="Real seconds to pause after each control step so the GUI is watchable.",
    )
    parser.add_argument(
        "--allow-untrained",
        action="store_true",
        help="Run the model path with random weights if no checkpoint exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 72)
    print("EagleVision GUI Demonstration")
    print("=" * 72)
    print("SUMO GUI:        enabled")
    print(f"Traffic scale:   {args.scale}")
    print(f"Sim steps:       {args.sim_steps}")
    print(f"Step pause:      {args.step_pause}s")
    print(f"Fixed baseline:  {args.fixed_green}s alternating NS/EW")
    print(f"Checkpoint:      {args.checkpoint}")

    fixed_rows, fixed_tls_rows, fixed_summary = run_controller(
        mode="fixed",
        traffic_scale=args.scale,
        use_gui=True,
        checkpoint_path=args.checkpoint,
        fixed_green=args.fixed_green,
        allow_untrained=args.allow_untrained,
        sim_steps=args.sim_steps,
        step_pause_s=args.step_pause,
    )

    print("\n[DEMO2] Fixed run finished, restarting SUMO GUI for the model pass...\n")
    time.sleep(2.5)

    model_rows, model_tls_rows, model_summary = run_controller(
        mode="model",
        traffic_scale=args.scale,
        use_gui=True,
        checkpoint_path=args.checkpoint,
        fixed_green=args.fixed_green,
        allow_untrained=args.allow_untrained,
        sim_steps=args.sim_steps,
        step_pause_s=args.step_pause,
        explain_model=True,
        explain_top_tls=1,
    )

    tag = _scale_tag(args.scale)
    _write_csv(f"logs/demo2_fixed_scale_{tag}.csv", fixed_rows)
    _write_csv(f"logs/demo2_model_scale_{tag}.csv", model_rows)

    print_summary(fixed_summary, model_summary)
    key_changes = _extract_key_changes(fixed_tls_rows, model_tls_rows, top_k=args.top_k)
    print_key_changes(key_changes)

    print("\nSaved:")
    print(f"  logs/demo2_fixed_scale_{tag}.csv")
    print(f"  logs/demo2_model_scale_{tag}.csv")


if __name__ == "__main__":
    main()
