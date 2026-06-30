# =============================================================================
# compare_results.py — Load both CSVs and generate comparison plots.
#
# Run AFTER both simulate_fixed.py and simulate_optimized.py have been run.
# =============================================================================

import csv
import os
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not installed — will print stats only, no plots")


def load_csv(path: str) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def compare(
    fixed_csv: str = "logs/fixed_results.csv",
    opt_csv:   str = "logs/optimized_results.csv",
    out_dir:   str = "logs/",
):
    if not os.path.exists(fixed_csv):
        print(f"Missing: {fixed_csv} — run simulate_fixed.py first")
        return
    if not os.path.exists(opt_csv):
        print(f"Missing: {opt_csv} — run simulate_optimized.py first")
        return

    fixed = load_csv(fixed_csv)
    opt   = load_csv(opt_csv)

    def col(rows, key):
        return [float(r[key]) for r in rows]

    # Extract columns
    f_q   = col(fixed, "total_queue")
    o_q   = col(opt,   "total_queue")
    f_r   = col(fixed, "reward")
    o_r   = col(opt,   "reward")
    f_acc = col(fixed, "accident")
    o_acc = col(opt,   "accident")
    f_g   = col(fixed, "cluster_growth")
    o_g   = col(opt,   "cluster_growth")

    # ── Print summary table ────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  {'Metric':<28} {'Fixed':>10} {'DQN':>10}")
    print("=" * 55)

    def row(name, fv, ov, fmt=".2f", lower_is_better=True):
        arrow = "✅" if (ov < fv) == lower_is_better else "❌"
        print(f"  {name:<28} {fv:>10{fmt}} {ov:>10{fmt}} {arrow}")

    row("Avg Queue Length",     np.mean(f_q), np.mean(o_q))
    row("Max Queue Length",     np.max(f_q),  np.max(o_q))
    row("Total Reward",         sum(f_r),     sum(o_r),   fmt=".1f", lower_is_better=False)
    row("Avg Reward/Step",      np.mean(f_r), np.mean(o_r), lower_is_better=False)
    row("Accident Steps",       sum(f_acc),   sum(o_acc))
    row("Avg Cluster Growth",   np.mean(f_g), np.mean(o_g))
    row("Queue Std Dev",        np.std(f_q),  np.std(o_q))
    print("=" * 55)

    # Improvement percentages
    q_improvement  = (np.mean(f_q) - np.mean(o_q)) / max(1e-6, np.mean(f_q)) * 100
    r_improvement  = (sum(o_r) - sum(f_r)) / max(1e-6, abs(sum(f_r))) * 100
    acc_improvement = (sum(f_acc) - sum(o_acc)) / max(1, sum(f_acc)) * 100
    print(f"\n  Queue reduction:    {q_improvement:+.1f}%")
    print(f"  Reward improvement: {r_improvement:+.1f}%")
    print(f"  Accident reduction: {acc_improvement:+.1f}%\n")

    # ── Plots ──────────────────────────────────────────────────────────────
    if not HAS_MPL:
        return

    steps_f = list(range(len(f_q)))
    steps_o = list(range(len(o_q)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("EagleVision: DQN vs Fixed-Timing Baseline", fontsize=14, fontweight="bold")

    # Plot 1: Queue length over time
    ax = axes[0, 0]
    ax.plot(steps_f, f_q, label="Fixed Timing", alpha=0.7, color="coral")
    ax.plot(steps_o, o_q, label="DQN Control",  alpha=0.7, color="steelblue")
    ax.set_title("Total Queue Length Over Time")
    ax.set_xlabel("Control Step")
    ax.set_ylabel("Vehicles Queued")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Cumulative reward
    ax = axes[0, 1]
    ax.plot(steps_f, np.cumsum(f_r), label="Fixed Timing", alpha=0.7, color="coral")
    ax.plot(steps_o, np.cumsum(o_r), label="DQN Control",  alpha=0.7, color="steelblue")
    ax.set_title("Cumulative Reward")
    ax.set_xlabel("Control Step")
    ax.set_ylabel("Cumulative Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Accident steps
    ax = axes[1, 0]
    ax.bar(["Fixed Timing", "DQN Control"], [sum(f_acc), sum(o_acc)],
           color=["coral", "steelblue"], alpha=0.8)
    ax.set_title("Total Accident Steps")
    ax.set_ylabel("Steps with Active Accident")
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 4: Queue distribution (histogram)
    ax = axes[1, 1]
    ax.hist(f_q, bins=30, alpha=0.6, label="Fixed Timing", color="coral",   density=True)
    ax.hist(o_q, bins=30, alpha=0.6, label="DQN Control",  color="steelblue", density=True)
    ax.set_title("Queue Length Distribution")
    ax.set_xlabel("Queue Length")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Plot saved → {out_path}")


if __name__ == "__main__":
    compare()
