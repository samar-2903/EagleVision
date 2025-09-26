from __future__ import annotations

from simulation import run_simulation


if __name__ == "__main__":
    # Run 1000 seconds with logging enabled
    run_simulation(steps=1000, dt_s=1.0, degrade_threshold=5, log_path="simulation_log.csv")


