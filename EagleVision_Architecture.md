# EagleVision v2 - Architecture & Algorithm Reference
### Smart AI-Based Hybrid Traffic Signal Optimization System
*Personal reference document - Samar, 2025*

---

## What This Document Is

This document explains the system EagleVision v2 actually is today, not just the system
it was originally imagined to become.

It is written as a practical architecture note: what the code does, why those pieces
exist, how they connect, what has changed over time, and where the project is already
more ambitious than a simple single-junction DQN.

The tone here is deliberate. This is not meant to read like a paper. It is meant to be
read straight through so that if you come back to this project in a month, or present it
to someone else, you can recover the full mental model quickly.

---

## 1. The Big Picture

EagleVision is an adaptive traffic signal control system built on top of SUMO. At the
highest level, it does one thing:

It looks at what traffic is doing right now, decides how to control the lights, watches
what happened next, scores that decision, and repeats until the controller learns a policy
that is better than a fixed timer.

The original mental model was a single intelligent intersection. The current codebase has
gone beyond that. It now operates as a **shared-policy multi-intersection controller**:
multiple traffic lights are observed at once, each traffic light receives its own local
state, and the same DQN policy is reused across all controlled intersections.

So the system now has five layers rather than four:

1. **The SUMO Environment** - simulates traffic and exposes the network through TraCI
2. **The Observation Layer** - converts raw traffic state into normalized features
3. **The Shared Control Layer** - routes one policy across many intersections
4. **The Agent** - a DQN network that selects one of four signal actions
5. **The Training Loop** - stores experience, updates the network, and saves checkpoints

That shared control layer is where the project starts to become MARL-like, and later in
this document I explain exactly what that means and what it does not mean.

---

## 2. The Environment - SUMO and TraCI

SUMO (Simulation of Urban Mobility) is the traffic world EagleVision learns inside.
Vehicles spawn from a route file, move through a signalized grid, queue, accelerate,
slow down, and leave the network. Python does not simulate the physics itself. SUMO does.

Python talks to SUMO through TraCI, the Traffic Control Interface. TraCI is effectively
the system's nervous system:

- Python asks questions such as which vehicles are stopped, what speed they are moving,
  which lane they occupy, and which signal phase is active.
- Python issues commands such as changing the current traffic light phase, setting a green
  duration, or reducing lane speed when an accident is simulated.

In EagleVision, SUMO advances one simulation second at a time, but the controller acts
every `CONTROL_INTERVAL = 5` seconds. That 5-second decision interval is a compromise:
fast enough to react, slow enough to remain physically plausible.

The environment automatically discovers traffic lights with at least four phases and
controls all such signalized intersections by default. In the current network this means
the agent is not just acting on one junction. It is acting across the grid.

Two practical additions matter here:

- **Traffic scaling:** the environment can now run with a SUMO `--scale` factor, which
  makes it easy to demonstrate behavior under lighter traffic such as `0.5` demand.
- **Fixed-time stepping inside the same environment:** the same environment can now run
  either the learned controller or a fixed baseline controller, which makes the side-by-side
  demo much cleaner because both modes use the same network, seed, and state machinery.

---

## 3. The Observation Layer - Turning Raw Simulation State into a Learning State

The neural network does not receive images, maps, or symbolic traffic engineering rules.
It receives a 15-dimensional numeric state vector for each controlled intersection:

```text
[
  Q_N, Q_S, Q_E, Q_W,
  V_N, V_S, V_E, V_W,
  A_N, A_S, A_E, A_W,
  cluster_growth,
  accident_flag,
  risk_score
]
```

These 15 numbers are the compressed operational picture of one traffic light's local
neighborhood.

### Queue lengths

A vehicle is considered queued if its speed is below `0.5 m/s`. The environment counts
stopped vehicles separately for north, south, east, and west incoming approaches.

### Average speeds

For each direction, the environment averages the speeds of all vehicles currently on the
incoming lanes associated with that direction.

### Arrival rates

The environment tracks newly appearing vehicles and records their arrival times in a
direction-specific rolling history. The arrival feature is the number of arrivals during
the last 60 seconds divided by 60.

### Cluster growth

Stopped vehicle positions are passed to the OPTICS clustering module. The system tracks
how the size of clustered stopped groups changes over time. A growing cluster is treated
as an early warning signal for a congestion cascade.

### Accident flag

This is a binary feature from the accident manager indicating whether an accident is
currently active at that intersection.

### Risk score

This is a logistic-style congestion risk estimate computed from queue size, weather
severity, and cluster growth dynamics.

### Normalization

Every feature is clipped and normalized before entering the network. Queue, speed,
arrival, and growth are all scaled by configured upper bounds. This matters because
neural networks train more reliably when the inputs live on comparable ranges instead of
mixing raw values like 80 queued vehicles with 0.3 accident flags and 6 m/s speeds.

---

## 4. OPTICS Clustering - Reading the Shape of a Jam, Not Just Its Size

Queue length tells you how bad traffic already is. Cluster growth tells you whether the
 jam is still forming.

EagleVision uses OPTICS, a density-based clustering algorithm, on the positions of
stopped vehicles. Each stopped vehicle is a point in `(x, y)` space. OPTICS groups points
that are densely packed and identifies congestion structure without requiring a fixed
number of clusters up front.

The key output used by the controller is not a fancy cluster label. It is the **growth
rate** of clustered stopped vehicles over time. That growth rate is valuable because it
acts like an early warning term:

- long queue + growing clusters = congestion is still worsening
- long queue + shrinking clusters = the signal policy may already be recovering the state

That feature feeds both the state vector and the reward function, so the model not only
sees congestion forming but is penalized for allowing it to keep growing.

---

## 5. Risk Scoring and Accident Management

### The Risk Scorer

The risk score is a logistic transform of congestion-related features. In plain language:
as queues rise, and as congestion growth becomes more aggressive, the probability-like
risk score rises toward 1.

This is not meant to be a perfect crash model. It is a practical safety signal that
allows the controller to treat a dangerous traffic state differently from a merely slow one.

### The Accident Manager

When an accident is triggered, EagleVision does more than flip a binary flag.

- An accident event is attached to a traffic light
- A likely clearance duration is sampled from a lognormal distribution
- A lane can be partially blocked by reducing its maximum speed
- The event remains active until its sampled clearance time is reached
- After clearance, the blockage decays rather than disappearing instantaneously

This gives the agent a real reason to learn safety-aware control behavior. A bad traffic
state is not just "a bit more queue." It can turn into a multi-step disruption that
damages reward for a long period.

---

## 6. The DQN Agent - What the Network Is Learning

The learning core is a Deep Q-Network.

For each local traffic state, the network outputs four Q-values:

```text
Q(s, 0), Q(s, 1), Q(s, 2), Q(s, 3)
```

Each Q-value estimates how good that action is expected to be when future reward is also
taken into account.

### Network architecture

The current network is a compact MLP:

- input: 15 features
- hidden: 128 ReLU units
- hidden: 128 ReLU units
- output: 4 action values

This is intentionally small. The state is structured, low-dimensional, and already
engineered. A giant model would add cost without adding much intelligence.

### Target network

The system maintains:

- `online_net` - updated continuously
- `target_net` - updated periodically from the online network

This is standard DQN stabilization. Without it, the network would try to learn from a
target that is moving at the same time as the prediction.

### Epsilon-greedy action selection

During training, the controller explores with epsilon-greedy behavior:

- with probability `epsilon`, act randomly
- otherwise, act greedily using the largest Q-value

That keeps the system from prematurely locking onto a poor policy.

---

## 7. The Action Space - What the Controller Can Actually Do

Each controlled traffic light receives one of four discrete actions:

- **Action 0 - NS Green:** force a north-south green phase
- **Action 1 - EW Green:** force an east-west green phase
- **Action 2 - Extend:** extend the current green
- **Action 3 - Short Priority Green:** give a short green to the more congested axis

For actions 0 and 1, the duration is queue-scaled in the learned-control path:

```text
duration = clip(MIN_GREEN + 0.5 * queue_on_axis, MIN_GREEN, MAX_GREEN)
```

That means the agent is not only selecting a direction. It is indirectly selecting a
direction plus a congestion-responsive dwell time.

For demonstrations and fixed-baseline comparison, the environment can also run in a pure
fixed-duration mode where NS and EW alternate with the same configured green time.

---

## 8. Experience Replay and Learning Updates

After every decision, the system stores a transition:

```text
(state, action, reward, next_state, done)
```

These transitions are placed into a replay buffer with capacity `50,000`. Learning does
not begin immediately. The system waits until at least `1,000` transitions are present,
then samples random batches of size `64`.

This solves one of the central problems in reinforcement learning on time-series systems:
consecutive states are highly correlated. Random replay breaks that short-term correlation
and makes gradient updates far more stable.

The current trainer performs **4 learning updates per control step** once replay is warm.
So the model is not just learning once per traffic decision. It is learning several times
from memory after each decision.

There is also a jam-oriented replay bias: when a jam event is detected, a short window of
recent transitions is replayed again. This tells the learner, in effect, that the seconds
right before severe congestion matter disproportionately.

---

## 9. Reward Design - What the System Is Actually Optimizing

EagleVision uses a negative penalty-style reward. In simplified form:

```text
r_t =
  - alpha * queue_penalty
  - beta  * delay_penalty
  - gamma * accident_penalty
  - delta * cluster_growth_penalty
```

In the current configuration the priorities are:

- queue length matters
- delay matters, but less than queue
- accidents matter a lot
- growing stopped-vehicle clusters matter

The current code also adds two system-level penalties beyond the local reward:

### Global queue penalty

Each local reward is reduced by a penalty proportional to **global network queue**, not
just that intersection's queue. This matters because once multiple signals are controlled
at once, a purely local controller can improve one junction by dumping congestion into the
next one. The global penalty pushes against that behavior.

### Jam penalty

When the network enters a jam regime, an extra penalty is applied. Severe jam states are
treated as qualitatively worse than ordinary congestion.

Together, these terms push the policy away from greedy local improvements that damage the
rest of the network.

---

## 10. Jam Detection and Dynamic Demand Throttling

This is one of the more interesting practical additions in EagleVision v2.

The environment monitors global queue growth and detects when the system enters a jam
state using thresholds on:

- absolute global queue
- sudden queue jumps

When a jam is detected, the environment can temporarily lower the probability that newly
appearing vehicles are kept in the network. In other words, demand is dynamically throttled
under severe overload.

This is not traditional traffic signal control alone. It is a resilience mechanism.

It serves two purposes:

1. It prevents catastrophic meltdown during training runs
2. It marks jam onset explicitly so the replay logic and reward function can treat these
   moments as especially important

---

## 11. The Shared-Policy MARL Layer

This is the part that deserves the "we wrote a MARL after" note, with one important
qualification:

**EagleVision is not a fully separate-agent MARL implementation.**

It is better described as a **shared-policy multi-agent traffic control system**.

Here is what that means in practice:

- there are many controlled traffic lights
- each traffic light has its own local observation
- each traffic light receives its own action at each decision step
- the same DQN network is reused for all of them
- the rewards include both local and shared network effects

So yes, the system now has a multi-agent structure: multiple agents act simultaneously in
multiple locations. But they are not independently parameterized agents with separate
networks, separate replay buffers, and explicit inter-agent messaging.

Instead, this is a **parameter-sharing MARL design**:

- one policy
- many intersections
- decentralized local observations
- shared learning signal components

This is often a very sensible first MARL step because it gives you most of the scalability
benefits without exploding the number of trainable parameters.

### Why this matters

A single-junction controller can only solve traffic locally. A shared-policy multi-agent
controller can start learning reusable signal behavior patterns that generalize across the
grid:

- when a queue builds on one axis, how aggressively should that junction respond
- when the network is starting to jam, how should each local controller behave
- when accidents or spillbacks appear, which signal should hold green longer and which
  should stop feeding a bad downstream state

### What it is not yet

It is not yet:

- a graph-neural MARL controller
- a communication-aware agent team
- a centralized critic with decentralized actors
- a true learned coordination policy over explicit neighborhood embeddings

That would be the next stage. The current stage is the bridge between single-agent RL and
full network-level MARL.

---

## 12. The Training Loop - What Happens During a Real Run

A training episode works like this:

1. SUMO is reset
2. all controllable traffic lights are discovered
3. each traffic light gets its local 15-feature state
4. the shared DQN chooses one action per traffic light
5. SUMO advances for 5 simulated seconds
6. the next state and reward are computed
7. all resulting transitions are added to replay
8. the network learns from replay batches
9. epsilon decays
10. the process repeats until the episode ends

The episode limit is currently:

- `SIMULATION_STEPS = 3600`
- `CONTROL_INTERVAL = 5`

So a full episode is up to about `720` control decisions, unless the network empties early.

The default long-run trainer in `train.py` is still the full-featured training entry point.
It supports GUI mode, step logging, and long training schedules.

The new quiet trainer in `alternativet_training.py` is a practical execution path for
actually generating checkpoints without flooding the console:

- headless SUMO
- GPU used automatically if CUDA is available
- heartbeat every 30 seconds
- 10 episodes by default
- final checkpoint always saved
- per-episode checkpoints saved as well

That second trainer exists because in practice the project needed a reliable way to
produce a model artifact for demonstration, not just a research-style verbose trainer.

---

## 13. Checkpoints - How the Model Is Preserved

The DQN checkpoint stores:

- online network weights
- target network weights
- optimizer state
- current epsilon
- number of learning steps already completed

That means training can resume rather than starting from scratch.

There are now two important checkpoint workflows:

### Standard training path

`train.py` saves checkpoints periodically according to `SAVE_FREQ` and again at the end of
training if the run completes normally.

### Quiet training path

`alternativet_training.py` saves:

- `checkpoints/dqn_checkpoint_ep01.pt` through `...ep10.pt`
- `checkpoints/dqn_checkpoint.pt` as the final consolidated checkpoint

This matters for presentation because the demonstration path expects a real trained model
file to exist.

---

## 14. Evaluation and Demonstration

There are now three evaluation styles in the repository.

### Fixed baseline

`simulate_fixed.py` runs a non-learning fixed schedule controller.

### Trained DQN evaluation

`simulate_optimized.py` loads a checkpoint and runs the learned policy greedily.

### Side-by-side demonstration

`simulation_demo.py` was added specifically for presentation use.

It runs:

1. a fixed-time baseline pass
2. a model-driven pass

on the same environment setup, with optional traffic scaling such as `--scale 0.5`.

The point of that file is not just to print average reward. It is to show **why** the
model's trajectory diverged:

- which `tls_id` made the important decision
- what signal action was chosen
- how local queue and risk changed
- whether an accident was cleared or avoided
- which state features had the strongest influence on that decision

That gives you something much more explainable during a live demonstration than a bare
CSV comparison.

---

## 15. What Is Implemented, What Is Present, and What Is Still Aspirational

This codebase contains some ideas at different stages of maturity.

### Implemented and active in the main path

- shared-policy multi-intersection DQN control
- queue, speed, arrival, cluster growth, accident, and risk state features
- replay-based DQN training
- target network updates
- accident simulation
- jam detection and mitigation
- fixed-vs-model evaluation
- quiet checkpoint-producing training
- presentation-oriented simulation demo

### Present but not central to the active training loop

`forecast.py` contains a simple forecasting path and stubs for more advanced temporal
models such as LSTM and ST-GCN style components. These are not driving the current DQN
training loop.

### Still the natural next step

If the project continues to grow, the obvious next architectural upgrades are:

- true graph-based multi-intersection coordination
- stronger forecasting integrated directly into state or reward shaping
- explicit MARL coordination rather than parameter-sharing alone
- a more formal incident response policy

---

## Summary

EagleVision v2 began as a reinforcement learning traffic signal controller and has evolved
into a shared-policy multi-intersection traffic control system with clear MARL structure.

The current system:

- runs inside SUMO
- builds a 15-feature local state per traffic light
- uses one DQN policy across many intersections
- penalizes both local congestion and network-wide overload
- models accidents and jam conditions explicitly
- can train quietly on GPU and save usable checkpoints
- can demonstrate fixed vs model behavior at `0.5` traffic scale with interpretable
  decision-level output

That is the right way to think about EagleVision now:

not just as "a DQN on one junction," but as the first serious network-level version of an
adaptive, safety-aware, shared-policy traffic control architecture.

---
*EagleVision v2 - Samar, 2025*
