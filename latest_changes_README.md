# EagleVision v2 - Latest Changes

This document summarizes what has been built so far in EagleVision v2 and records the
main mathematical equations used by the project.

EagleVision v2 is a SUMO-based adaptive traffic signal control system. The current
version trains a Deep Q-Network (DQN) controller that observes traffic around signalized
intersections, selects signal actions, and learns from queue, delay, accident, risk, and
cluster-growth feedback.

## Current Project State

The project has moved beyond a single-intersection prototype. The current environment
automatically discovers controllable traffic lights in the SUMO network and applies one
shared DQN policy across multiple intersections.

The main pieces are:

- `sumo_env.py`: SUMO/TraCI environment, multi-signal control, observation building,
  accident handling, jam control, rewards, and action execution.
- `dqn_agent.py`: DQN model, epsilon-greedy action selection, target network, replay
  learning, checkpoint save/load.
- `replay_buffer.py`: circular experience replay buffer.
- `reward.py`: normalized penalty reward for queues, delay, accidents, and growing jams.
- `risk.py`: logistic accident-risk model and lane-level risk scoring helper.
- `clustering.py`: OPTICS-based clustering of stopped vehicles.
- `accident_manager.py`: accident lifecycle, clearance time, and recovery decay.
- `train.py`: full training loop for shared-policy multi-signal control.
- `alternativet_training.py`: quieter trainer with heartbeat logging and per-episode
  checkpoints.
- `simulation_demo.py` and `demo2.py`: fixed-baseline vs model-controlled demonstrations.
- `compare_results.py` and `plot_demo_logs.py`: result comparison and visualization.

## What We Have Built So Far

1. Built the SUMO traffic-control environment around TraCI.
2. Added a 15-feature local state vector for each traffic light.
3. Implemented DQN control with a compact neural network.
4. Added experience replay and target-network stabilization.
5. Added queue-aware traffic-light actions instead of only fixed actions.
6. Added OPTICS clustering to detect growth of stopped-vehicle clusters.
7. Added a risk score that reacts to queue, weather severity, and congestion growth.
8. Added accident simulation with lane speed blocking, clearance time, and recovery.
9. Added jam detection and demand throttling to prevent runaway gridlock.
10. Added fixed-timing baseline simulations for comparison.
11. Added side-by-side demo scripts that compare fixed control against trained DQN control.
12. Added logging, CSV output, checkpointing, and plotting utilities.

## State Representation

Each controlled traffic light receives this local state:

```text
s_t = [
  Q_N, Q_S, Q_E, Q_W,
  V_N, V_S, V_E, V_W,
  A_N, A_S, A_E, A_W,
  G,
  accident_flag,
  risk_score
]
```

Where:

- `Q_d` is the queue length in direction `d`.
- `V_d` is the average speed in direction `d`.
- `A_d` is the arrival rate in direction `d`.
- `G` is stopped-cluster growth.
- `accident_flag` is `1` when an accident is active and `0` otherwise.
- `risk_score` is a probability-like accident/congestion risk value.

The state dimension is:

```text
STATE_DIM = 15
```

## Queue Equation

A vehicle is counted as queued when its speed is below `0.5 m/s`.

```text
Q_d = sum_{i in lanes(d)} 1[v_i < 0.5]
```

The total local queue at one traffic light is:

```text
Q_total = Q_N + Q_S + Q_E + Q_W
```

## Average Speed Equation

For each direction:

```text
V_d = (1 / n_d) * sum_{i in vehicles(d)} v_i
```

If there are no vehicles in that direction, the implementation safely returns `0`.

## Arrival Rate Equation

The environment keeps a rolling 60-second arrival window:

```text
A_d(t) = arrivals_d(t - 60, t) / 60
```

This estimates how many vehicles per second are entering from direction `d`.

## Normalization

State values are clipped and normalized before entering the neural network:

```text
Q_d_norm = clip(Q_d, 0, MAX_QUEUE) / MAX_QUEUE
V_d_norm = clip(V_d, 0, MAX_SPEED) / MAX_SPEED
A_d_norm = clip(A_d, 0, MAX_ARRIVAL) / MAX_ARRIVAL
G_norm   = clip(G, 0, MAX_GROWTH) / MAX_GROWTH
```

Current normalization constants:

```text
MAX_QUEUE = 100
MAX_SPEED = 15
MAX_ARRIVAL = 2
MAX_GROWTH = 5
```

## OPTICS Cluster Growth

Stopped vehicles are represented as `(x, y)` points and clustered with OPTICS.

The growth feature is computed from the total number of clustered stopped vehicles:

```text
G_t = (C_t - C_{t-1}) / dt
```

In the current implementation:

```text
dt = 1
```

So:

```text
G_t = C_t - C_{t-1}
```

Positive growth means clustered stopped traffic is expanding. Negative growth means the
clustered jam is shrinking.

## Risk Equation

The accident/congestion risk model uses a logistic function:

```text
x = 1.2 * (Q_total / (50 + Q_total))
  + 0.8 * weather_severity
  + 0.6 * tanh(congestion_growth_rate)
  + 0.2 * other

risk = 1 / (1 + exp(-4 * (x - 0.6)))
```

This keeps the risk score in the range:

```text
0 <= risk <= 1
```

In the SUMO environment, weather severity is currently fixed at `0.2`, and congestion
growth comes from the OPTICS cluster-growth feature.

## Accident Probability

The environment converts risk into a per-control-step accident trigger probability:

```text
p_accident = min(ACCIDENT_MAX_PROB_PER_STEP, risk_score * ACCIDENT_RATE_SCALE)
```

Current values:

```text
ACCIDENT_RATE_SCALE = 0.015
ACCIDENT_MAX_PROB_PER_STEP = 0.10
```

When an accident is triggered, the affected lane has its max speed reduced:

```text
blocked_speed = ACCIDENT_BLOCK_SPEED
```

Current value:

```text
ACCIDENT_BLOCK_SPEED = 0.20
```

## Accident Clearance

Accident clearance time is sampled from a lognormal distribution:

```text
T_clear ~ LogNormal(log(mu_tc), sigma_tc)
```

Current manager defaults:

```text
mu_tc = 300 seconds
sigma_tc = 0.5
T_recovery = 60 seconds
```

After clearance, blockage decays linearly during recovery:

```text
blockage(t) = (decay_end - t) / T_recovery
```

The blockage reaches `0` after the recovery window ends.

## Delay Estimate

The reward module estimates delay from queue length using a Little's-Law-style
relationship:

```text
D = Q / mu
```

Where:

```text
mu = average service rate
```

The default service rate is:

```text
mu = 0.35
```

## Action Space

The DQN outputs one Q-value for each of four actions:

```text
Q(s, 0), Q(s, 1), Q(s, 2), Q(s, 3)
```

The actions are:

```text
0 = North-South green
1 = East-West green
2 = Extend current green
3 = Short priority green for the more congested axis
```

For actions `0` and `1`, green duration is queue-scaled:

```text
duration = clip(MIN_GREEN + 0.5 * queue_on_axis, MIN_GREEN, MAX_GREEN)
```

Current bounds:

```text
MIN_GREEN = 5
MAX_GREEN = 60
DEFAULT_GREEN = 30
```

For action `2`, the current green is extended:

```text
duration = clip(remaining_duration + 10, MIN_GREEN, MAX_GREEN)
```

For action `3`, the environment chooses the heavier axis:

```text
axis = NS if (Q_N + Q_S) >= (Q_E + Q_W) else EW
duration = MIN_GREEN
```

## Reward Function

The local reward is a normalized negative-penalty score:

```text
r_t =
  - alpha * mean(Q_i / Q_norm)
  - beta  * mean(D_i / D_norm)
  - gamma * mean(A_i)
  - delta * mean(max(0, G_i) / G_norm)
```

Where:

- `Q_i` is queue length.
- `D_i` is delay.
- `A_i` is the accident flag.
- `G_i` is cluster growth.
- `max(0, G_i)` penalizes only growing clusters.

Current reward constants:

```text
alpha = 1.0
beta = 0.1
gamma = 5.0
delta = 0.2
Q_norm = 50
D_norm = 100
G_norm = 10
```

The reward is clipped:

```text
-1000 <= r_t <= 0
```

The multi-signal environment also adds a shared global queue penalty:

```text
shared_queue_penalty =
  GLOBAL_QUEUE_WEIGHT * (global_queue / (num_controlled_tls * Q_norm))
```

And a jam penalty when global jam mitigation is active:

```text
jam_severity = max(1, global_queue / JAM_QUEUE_THRESHOLD)
jam_penalty = JAM_PENALTY * jam_severity
```

The final reward per traffic light is:

```text
r_final = r_local - shared_queue_penalty - jam_penalty
```

## Jam Detection and Demand Control

The environment marks a jam if either the global queue is too high or the queue jumps too
quickly:

```text
jam_detected =
  global_queue >= JAM_QUEUE_THRESHOLD
  or
  (global_queue - previous_global_queue) >= JAM_QUEUE_JUMP_THRESHOLD
```

Current thresholds:

```text
JAM_QUEUE_THRESHOLD = 180
JAM_QUEUE_JUMP_THRESHOLD = 35
JAM_RECOVERY_THRESHOLD = 90
```

When a jam is active, vehicle arrivals can be throttled using a keep probability:

```text
keep_vehicle ~ Bernoulli(demand_keep_prob)
```

The keep probability is reduced during heavy jams and relaxed as the network recovers.

## DQN Model

The neural network is a small multilayer perceptron:

```text
input:  15
hidden: 128 ReLU
hidden: 128 ReLU
output: 4
```

It approximates:

```text
Q(s, a; theta)
```

Where `theta` represents the online network weights.

## Bellman Target

The DQN target is:

```text
y = r + gamma * max_{a'} Q_target(s', a') * (1 - done)
```

If the episode has ended:

```text
done = 1
y = r
```

Current discount factor:

```text
gamma = 0.95
```

## Training Loss

The predicted Q-value is:

```text
Q_pred = Q_online(s, a)
```

The training loss uses Huber loss:

```text
L = SmoothL1Loss(Q_pred, y)
```

The target network is synchronized periodically:

```text
theta_target <- theta_online
```

Current sync frequency:

```text
TARGET_UPDATE_FREQ = 200
```

## Epsilon-Greedy Exploration

During training:

```text
action =
  random_action,              with probability epsilon
  argmax_a Q(s, a; theta),    with probability 1 - epsilon
```

Epsilon decays as:

```text
epsilon <- max(EPS_END, epsilon * EPS_DECAY)
```

Current values:

```text
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.9995
```

## Experience Replay

Each transition is stored as:

```text
(s_t, a_t, r_t, s_{t+1}, done)
```

The agent learns from random mini-batches instead of only the most recent step:

```text
batch ~ Uniform(replay_buffer)
```

Current replay settings:

```text
REPLAY_CAPACITY = 50000
MIN_REPLAY_SIZE = 1000
BATCH_SIZE = 64
LEARN_UPDATES_PER_STEP = 4
```

## Training Pipeline

The active training loop is:

```text
reset SUMO
observe local states for all controlled traffic lights
select one action per traffic light
apply actions through TraCI
advance SUMO by CONTROL_INTERVAL
compute next states and rewards
store transitions in replay memory
sample replay batches
update DQN
decay epsilon
save checkpoints
```

Current simulation settings:

```text
SIMULATION_STEPS = 1500
CONTROL_INTERVAL = 5
NUM_EPISODES = 500
```

A full episode gives approximately:

```text
1500 / 5 = 300 control decisions
```

## Demo and Evaluation Work

The project now includes a cleaner demo path:

- Run a fixed-timing baseline.
- Reset SUMO.
- Load a DQN checkpoint.
- Run model-controlled traffic signals.
- Compare both runs on the same network setup.

The demo records:

- Average global queue.
- Maximum global queue.
- Average global delay.
- Total reward.
- Accident events.
- Active accident steps.
- Jam-active steps.
- High-impact model decisions.
- Model decision drivers such as queue pressure, risk score, and cluster growth.

The presentation demo defaults to:

```text
traffic_scale = 0.5
sim_steps = 600
control decisions per pass = 600 / 5 = 120
```

## Summary

EagleVision v2 now has the core components of an adaptive traffic-signal controller:
SUMO simulation, multi-intersection observations, DQN learning, replay memory, safety
risk, accident modeling, clustering-based congestion growth, jam handling, fixed-baseline
comparison, and presentation-ready demos.

The most important mathematical pieces are the normalized state vector, the logistic risk
score, the queue/delay/accident/growth reward, the Bellman target, and the epsilon-greedy
DQN update loop. Together, these define how the system observes traffic, decides signal
actions, evaluates outcomes, and improves over repeated simulation.
