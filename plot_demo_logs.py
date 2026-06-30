import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NUMERIC_FIELDS = [
    "step",
    "sim_time",
    "reward",
    "global_queue",
    "global_avg_delay",
    "accident_events",
    "active_accidents",
    "jam_active",
    "demand_keep_prob",
    "vehicles_removed",
]


def load_rows(path: Path) -> List[Dict[str, float | str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")

    parsed: List[Dict[str, float | str]] = []
    for row in rows:
        item: Dict[str, float | str] = dict(row)
        for field in NUMERIC_FIELDS:
            item[field] = float(row[field])
        parsed.append(item)
    return parsed


def column(rows: List[Dict[str, float | str]], key: str) -> np.ndarray:
    return np.array([float(row[key]) for row in rows], dtype=float)


def summarize(rows: List[Dict[str, float | str]]) -> Dict[str, float]:
    return {
        "avg_queue": float(np.mean(column(rows, "global_queue"))),
        "max_queue": float(np.max(column(rows, "global_queue"))),
        "avg_delay": float(np.mean(column(rows, "global_avg_delay"))),
        "total_reward": float(np.sum(column(rows, "reward"))),
        "accident_events": float(np.sum(column(rows, "accident_events"))),
        "active_accident_steps": float(np.sum(column(rows, "active_accidents"))),
        "jam_steps": float(np.sum(column(rows, "jam_active"))),
        "vehicles_removed": float(np.max(column(rows, "vehicles_removed"))),
    }


def default_output_path(fixed_csv: Path, model_csv: Path) -> Path:
    fixed_name = fixed_csv.stem.replace("demo2_", "").replace("fixed_", "").strip("_")
    model_name = model_csv.stem.replace("demo2_", "").replace("model_", "").strip("_")
    suffix = fixed_name if fixed_name == model_name else "comparison"
    return fixed_csv.parent / f"demo2_plots_{suffix}.png"


def plot_logs(fixed_csv: Path, model_csv: Path, output: Path) -> Path:
    fixed = load_rows(fixed_csv)
    model = load_rows(model_csv)

    fixed_steps = column(fixed, "step")
    model_steps = column(model, "step")

    fixed_summary = summarize(fixed)
    model_summary = summarize(model)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle("EagleVision Demo2: Fixed Baseline vs Model", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(fixed_steps, column(fixed, "global_queue"), label="Fixed", color="#c65d3b", linewidth=2)
    ax.plot(model_steps, column(model, "global_queue"), label="Model", color="#1f5b8f", linewidth=2)
    ax.set_title("Global Queue")
    ax.set_xlabel("Control Step")
    ax.set_ylabel("Vehicles")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(fixed_steps, column(fixed, "global_avg_delay"), label="Fixed", color="#c65d3b", linewidth=2)
    ax.plot(model_steps, column(model, "global_avg_delay"), label="Model", color="#1f5b8f", linewidth=2)
    ax.set_title("Average Delay")
    ax.set_xlabel("Control Step")
    ax.set_ylabel("Seconds")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[0, 2]
    ax.plot(fixed_steps, np.cumsum(column(fixed, "reward")), label="Fixed", color="#c65d3b", linewidth=2)
    ax.plot(model_steps, np.cumsum(column(model, "reward")), label="Model", color="#1f5b8f", linewidth=2)
    ax.set_title("Cumulative Reward")
    ax.set_xlabel("Control Step")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(fixed_steps, column(fixed, "active_accidents"), label="Fixed active", color="#c65d3b", linewidth=2)
    ax.plot(model_steps, column(model, "active_accidents"), label="Model active", color="#1f5b8f", linewidth=2)
    ax.plot(fixed_steps, column(fixed, "accident_events"), label="Fixed new", color="#e3a58b", linestyle="--")
    ax.plot(model_steps, column(model, "accident_events"), label="Model new", color="#7aa6c9", linestyle="--")
    ax.set_title("Accident Activity")
    ax.set_xlabel("Control Step")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(fixed_steps, column(fixed, "demand_keep_prob"), label="Fixed keep prob", color="#c65d3b", linewidth=2)
    ax.plot(model_steps, column(model, "demand_keep_prob"), label="Model keep prob", color="#1f5b8f", linewidth=2)
    ax.fill_between(model_steps, 0, column(model, "jam_active"), color="#1f5b8f", alpha=0.15, label="Model jam active")
    ax.fill_between(fixed_steps, 0, column(fixed, "jam_active"), color="#c65d3b", alpha=0.12, label="Fixed jam active")
    ax.set_title("Jam Mitigation")
    ax.set_xlabel("Control Step")
    ax.set_ylabel("Keep Probability / Jam Flag")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 2]
    labels = ["Avg queue", "Avg delay", "Total reward", "Accident events", "Jam steps", "Vehicles removed"]
    fixed_values = [
        fixed_summary["avg_queue"],
        fixed_summary["avg_delay"],
        fixed_summary["total_reward"],
        fixed_summary["accident_events"],
        fixed_summary["jam_steps"],
        fixed_summary["vehicles_removed"],
    ]
    model_values = [
        model_summary["avg_queue"],
        model_summary["avg_delay"],
        model_summary["total_reward"],
        model_summary["accident_events"],
        model_summary["jam_steps"],
        model_summary["vehicles_removed"],
    ]
    x = np.arange(len(labels))
    width = 0.38
    ax.bar(x - width / 2, fixed_values, width, label="Fixed", color="#c65d3b")
    ax.bar(x + width / 2, model_values, width, label="Model", color="#1f5b8f")
    ax.set_title("Run Summary")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend()

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot EagleVision demo2 CSV logs.")
    parser.add_argument(
        "--fixed-csv",
        type=Path,
        default=Path("logs/demo2_fixed_scale_0p5.csv"),
        help="Path to the fixed-baseline demo2 CSV.",
    )
    parser.add_argument(
        "--model-csv",
        type=Path,
        default=Path("logs/demo2_model_scale_0p5.csv"),
        help="Path to the model demo2 CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or default_output_path(args.fixed_csv, args.model_csv)
    saved = plot_logs(args.fixed_csv, args.model_csv, output)
    print(f"Saved plot: {saved}")


if __name__ == "__main__":
    main()
