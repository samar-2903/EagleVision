This folder is isolated from the live training code.

It does not modify `sumo_env.py`, `train.py`, `config.py`, or any active model path.

Files:

- `dynamic_state_testing.py`: standalone prototype for building a richer raw traffic-state vector up to 40 dimensions and compressing it back to the current 15-D policy shape

What it prototypes:

- per-vehicle speed-variance tracking over a rolling time window
- cluster-exit tracking per vehicle
- basic proximity-based collision checks
- accident candidate detection from:
  - low speed variance over many frames
  - low sustained speed
  - either a recent intersection transition or a same-intersection collision condition
- normalization of the expanded raw vector
- dimensional reduction from 40 raw dimensions back to 15 policy dimensions

Run:

```powershell
python experimental/dynamic_state_lab/dynamic_state_testing.py
```

Expected output:

- one normalized 40-D raw vector
- one compressed 15-D policy vector
- a few interpreted accident/cluster signals from the synthetic demo sequence

Integration note:

If this prototype behaves the way you want, the same logic can later be copied into a new environment variant without touching the current one.
