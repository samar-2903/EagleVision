# EagleVision v2

EagleVision v2 is a SUMO-based reinforcement learning project for adaptive traffic signal control.

The core idea is simple:

1. Run a traffic simulation in SUMO.
2. Observe the current traffic condition around one signalized intersection.
3. Let a DQN agent choose a signal-control action.
4. Score that action using a reward function based on congestion, delay, accidents, and cluster growth.
5. Repeat this many times so the neural network learns which actions tend to lead to better future traffic conditions.

This README explains the project from scratch and ties each concept to the actual files in this repository.

## What Problem This Model Is Solving

The project tries to control a traffic light better than a fixed-timing controller.

A fixed-timing controller follows a schedule like:

- keep North-South green for some seconds
- switch to East-West green for some seconds
- repeat forever

That is easy to implement, but it does not react to real congestion. If one direction is overloaded and the other is empty, fixed timing still wastes green time.

The DQN agent tries to do better by learning a policy:

- if the traffic state looks like this, action A is better
- if the traffic state looks like that, action B is better

Instead of hard-coding traffic rules, the project learns them from repeated simulation.

## High-Level Architecture

The active pipeline is:

- `train.py`: main training loop
- `sumo_env.py`: wraps SUMO as an RL environment
- `dqn_agent.py`: neural network, epsilon-greedy policy, DQN update
- `replay_buffer.py`: stores past transitions for experience replay
- `reward.py`: converts traffic conditions into a scalar reward
- `clustering.py`: computes stopped-vehicle cluster features with OPTICS
- `risk.py`: computes a risk score from congestion-related features
- `accident_manager.py`: tracks accident state if an accident event exists
- `simulate_fixed.py`: fixed-timing baseline
- `simulate_optimized.py`: evaluate trained DQN greedily
- `compare_results.py`: compare fixed vs DQN outputs
- `config.py`: all tunable constants

## What Happens During Training

Training happens in `train.py`.

For each episode:

1. `env.reset()` starts a fresh SUMO simulation.
2. The environment builds the current state vector.
3. The agent chooses one of 4 traffic-light actions.
4. SUMO advances for a short interval.
5. The environment computes reward and returns the next state.
6. The transition `(state, action, reward, next_state, done)` is stored.
7. The DQN samples a random batch from replay memory and updates the neural network.
8. Exploration is reduced gradually by epsilon decay.

This is the exact RL loop:

```text
state -> action -> next_state, reward -> store transition -> learn
```

## The Environment

`sumo_env.py` is the bridge between the RL agent and SUMO.

It does five main jobs:

1. launch and reset SUMO through TraCI
2. read queues, speeds, arrivals, and vehicle positions
3. convert raw simulation values into a normalized state vector
4. translate the agent's action into a traffic-light command
5. compute reward after the simulation advances

An episode is one SUMO run, up to `SIMULATION_STEPS` simulation ticks in `config.py`.

The agent does not act every single simulation tick. Instead, SUMO advances by `CONTROL_INTERVAL` steps per decision. Right now:

- `SIMULATION_STEPS = 1500`
- `CONTROL_INTERVAL = 5`

So one episode is up to 1500 simulation seconds, and the agent makes roughly `1500 / 5 = 300` decisions per full episode.

Episodes can also end earlier if the network becomes empty.

## The State Vector

The model does not look at raw images. It receives a 15-dimensional numeric feature vector.

The state is built in `sumo_env.py` as:

```text
[
  Q_N, Q_S, Q_E, Q_W,
  V_N, V_S, V_E, V_W,
  A_N, A_S, A_E, A_W,
  G,
  accident_flag,
  risk_score
]
```

Where:

- `Q_*`: queue length per direction
- `V_*`: average speed per direction
- `A_*`: arrival rate per direction
- `G`: cluster growth rate
- `accident_flag`: whether an accident is currently active
- `risk_score`: probability-like risk score in `[0, 1]`

### 1. Queue length

A vehicle is treated as queued if:

```text
speed < 0.5 m/s
```

The environment counts stopped vehicles separately for `N`, `S`, `E`, and `W`.

### 2. Speed

Average speed is computed per direction by averaging vehicle speeds for vehicles whose lane names contain `N`, `S`, `E`, or `W`.

### 3. Arrival rate

The environment tracks new vehicles entering the network and computes direction-wise arrival rate over a 60-second window:

```text
lambda_d = arrivals_in_last_60_seconds / 60
```

### 4. Cluster growth

Stopped vehicle positions are clustered with OPTICS in `clustering.py`.

The growth feature is based on the change in total clustered stopped vehicles:

```text
growth = (clustered_points_now - clustered_points_prev) / dt
```

In this implementation, `dt` is treated as 1 for the feature computation.

### 5. Accident flag

`accident_manager.py` stores accident event state per intersection id. In the current training path, the environment reads this flag and includes it in the state.

### 6. Risk score

The risk score comes from `risk.py` and is designed as a logistic function of congestion and related features:

```text
x = 1.2 * (queue_sum / (50 + queue_sum))
  + 0.8 * weather_severity
  + 0.6 * tanh(congestion_growth_rate)
  + 0.2 * other

risk = 1 / (1 + exp(-4 * (x - 0.6)))
```

This keeps the output in `[0, 1]`.

### Normalization

All state features are clipped and normalized before being passed to the neural network.

Examples:

- queue values are divided by `MAX_QUEUE`
- speed values are divided by `MAX_SPEED`
- arrival values are divided by `MAX_ARRIVAL`
- cluster growth is divided by `MAX_GROWTH`

Why normalize?

Because neural networks train more reliably when inputs are on comparable scales.

## The Action Space

The agent chooses one of 4 discrete actions:

- `0`: give green to North-South
- `1`: give green to East-West
- `2`: extend the current green phase
- `3`: give a short green to the more congested axis

Action execution happens in `sumo_env.py`.

For actions `0` and `1`, green time is scaled by observed congestion:

```text
duration = clip(MIN_GREEN + 0.5 * queue_on_axis, MIN_GREEN, MAX_GREEN)
```

So bigger queues get longer greens, but only within safe bounds.

## The Reward Function

The reward is the learning signal. It is implemented in `reward.py`.

The project uses a negative penalty-style reward:

```text
r_t = -alpha * mean(Q_i / Q_norm)
      -beta  * mean(Delay_i / D_norm)
      -gamma * mean(A_i)
      -delta * mean(max(0, cluster_growth_i) / G_norm)
```

Interpretation:

- larger queues are bad
- larger delays are bad
- accidents are very bad
- growing stopped-vehicle clusters are bad

The reward is clipped to `[-1000, 0]`.

### Delay estimate

Delay is estimated from queue length using a Little's-Law-style approximation:

```text
Delay ~= Queue / service_rate
```

In code:

```text
D ~= Q / mu
```

This is not a precise microscopic delay model. It is a practical proxy so the agent always has a delay-related signal even when direct delay is not explicitly queried from SUMO.

## The DQN Model

The neural network lives in `dqn_agent.py`.

It is a small multilayer perceptron:

- input: 15 state features
- hidden layer: 128 units + ReLU
- hidden layer: 128 units + ReLU
- output: 4 Q-values, one for each action

So for a given state `s`, the network outputs:

```text
Q(s, 0), Q(s, 1), Q(s, 2), Q(s, 3)
```

Each value answers:

"If I take this action now, how much future reward do I expect?"

## What Q-Learning Means Here

The main Q-learning idea is:

```text
Q(s, a) = r + gamma * max_a' Q(s', a')
```

Meaning:

- `r` is the immediate reward after action `a`
- `s'` is the next state
- `max_a' Q(s', a')` is the best future value available from the next state
- `gamma` discounts distant future reward

In practice the code uses a target network, so the training target becomes:

```text
target = r + gamma * max_a' Q_target(s', a') * (1 - done)
```

If `done = 1`, there is no future term.

The current network prediction is:

```text
pred = Q_online(s, a)
```

The network is trained so `pred` gets closer to `target`.

## Why This Is Called DQN

DQN stands for Deep Q-Network.

"Q-learning" by itself usually assumes a lookup table:

- state 1, action 1 -> value
- state 1, action 2 -> value
- ...

That breaks down when the state is continuous and multi-dimensional.

This project uses a neural network to approximate the Q-function instead of storing a table. That is the "deep" part.

## Epsilon-Greedy Exploration

Early in training, the model does not know what works. If it only used the current network predictions, it would exploit nonsense.

So action selection uses epsilon-greedy:

- with probability `epsilon`, choose a random action
- otherwise, choose the action with the highest predicted Q-value

This is implemented in `DQNAgent.select_action`.

Current defaults in `config.py`:

- `EPS_START = 1.0`
- `EPS_END = 0.05`
- `EPS_DECAY = 0.9995`

That means:

- early training is highly exploratory
- later training becomes mostly greedy, but still keeps some randomness

## Experience Replay

The replay buffer is implemented in `replay_buffer.py`.

Why not train on each transition immediately?

Because consecutive traffic states are highly correlated. If the network only learns from the latest step, optimization becomes unstable and overly myopic.

So the code stores many past transitions:

```text
(state, action, reward, next_state, done)
```

Then each learning step samples a random batch from memory.

This has two main benefits:

- breaks short-term correlation
- lowers gradient variance

## Target Network

The code maintains two networks:

- `online_net`: the one being optimized every step
- `target_net`: a frozen copy used to compute stable TD targets

Why?

If the same rapidly-changing network were used for both prediction and target construction, training would chase a moving target and become unstable.

So the code copies `online_net` into `target_net` only every `TARGET_UPDATE_FREQ` learning steps.

## The Loss Function

The project uses Huber loss (`smooth_l1_loss`) instead of raw MSE.

Why?

- MSE penalizes large errors quadratically
- that can create huge gradients if a target is far away
- Huber loss behaves more gently for large errors

That usually improves stability for RL.

The code also clips gradient norm to 10 to reduce exploding updates.

## What the Agent Is Actually Learning

The agent is not learning direct traffic engineering formulas.

It is learning a value approximation:

- states that tend to lead to lower congestion and better rewards should get higher Q-values
- actions that create bad downstream traffic patterns should get lower Q-values

Over time, the network should learn patterns like:

- if NS queues are high and EW is light, NS green is often better
- if the current phase is working and queues are still draining, extending may be better than switching
- if a cluster is growing quickly, starving that axis is risky

Those patterns are not manually programmed. They are supposed to emerge from repeated reward-driven learning.

## Training vs Evaluation

There are three different runtime modes in this repository.

### 1. Training

Run:

```bash
python train.py
```

This:

- explores with epsilon-greedy
- stores transitions
- updates the DQN
- saves checkpoints

### 2. Fixed baseline

Run:

```bash
python simulate_fixed.py
```

This does not learn anything. It just runs a fixed cycle and writes results.

### 3. Trained DQN evaluation

Run:

```bash
python simulate_optimized.py
```

This loads the checkpoint, forces `epsilon = 0.0`, and runs the learned policy greedily.

## How Results Are Compared

`compare_results.py` loads:

- `logs/fixed_results.csv`
- `logs/optimized_results.csv`

Then it compares:

- average queue length
- max queue length
- total reward
- average reward per step
- accident steps
- average cluster growth
- queue standard deviation

If `matplotlib` is installed, it also plots:

- queue over time
- cumulative reward
- total accident steps
- queue distribution histogram

## Current Default Behavior

Right now `config.py` is set up for interactive inspection rather than fast training:

- `TRAIN_USE_GUI = True`
- `TRAIN_STEP_LOG = True`
- `TRAIN_STEP_SLEEP_S = 0.10`

That means running `train.py` will:

- open the SUMO GUI
- print one line per control decision
- intentionally pause a bit so you can watch what is happening

This is useful for debugging, but it is much slower than headless training.

For faster training, you would usually change:

```python
TRAIN_USE_GUI = False
TRAIN_STEP_LOG = False
TRAIN_STEP_SLEEP_S = 0.0
```

## Important Files and What They Mean

- `config.py`: all hyperparameters, reward weights, and runtime settings
- `train.py`: the actual DQN training loop
- `sumo_env.py`: state construction, action execution, reward computation
- `dqn_agent.py`: Q-network, target network, replay learning
- `replay_buffer.py`: experience replay memory
- `reward.py`: reward equation and delay approximation
- `risk.py`: congestion-to-risk mapping
- `clustering.py`: OPTICS-based stopped-vehicle clustering
- `simulate_fixed.py`: baseline controller
- `simulate_optimized.py`: greedy evaluation of the trained controller
- `compare_results.py`: numerical and visual comparison
- `nets/grid_tls.sumocfg`: SUMO configuration
- `nets/grid_tls.net.xml`: road network
- `nets/grid_tls_routes.rou.xml`: vehicle demand / trips

## Modules Present But Not Fully Active in the Main Training Loop

Some modules exist but are not central to the current `train.py` pipeline.

### `forecast.py`

This file contains:

- a simple exponential smoothing forecaster
- an `LSTMStub`
- an `STGCNStub`
- an ensemble gate

These are currently helper or prototype components. They are not wired into the main DQN training loop.

### `risk.py::AccidentMDP`

There is an `AccidentMDP` class in `risk.py`, but the main environment currently uses `accident_probability(...)` and `AccidentManager`, not the full `AccidentMDP` class as the active driver of events.

## Limitations and Practical Notes

This repository is a useful RL traffic-control prototype, but there are important caveats.

### 1. Single-intersection control focus

The state and action design are centered on controlling one traffic light at a time.

### 2. Reward engineering matters a lot

If the reward weights are poorly tuned, the agent may optimize the wrong thing.

For example:

- over-penalize delay and it may oscillate phases
- under-penalize accidents and it may ignore risky situations

### 3. Queue-based delay is approximate

`Delay ~= Queue / service_rate` is a practical proxy, not a full traffic-theory delay model.

### 4. Training stability is not guaranteed

DQN can be noisy. Improvement is usually judged from trends over many episodes, not a few individual runs.

### 5. GUI mode is for understanding, not throughput

If you train with GUI and step logs on, wall-clock training time becomes much larger.

## Installation and Requirements

Python dependencies in `requirements.txt`:

- `torch`
- `numpy`
- `scikit-learn`
- `matplotlib`

You also need SUMO installed locally. `traci` comes from SUMO, not from pip.

Typical flow:

1. install SUMO
2. make sure SUMO binaries are on `PATH`
3. make sure Python can import `traci`
4. install Python dependencies

## Minimal Run Order

If you want the full workflow:

1. Train the model:

```bash
python train.py
```

2. Run the fixed baseline:

```bash
python simulate_fixed.py
```

3. Run the trained DQN greedily:

```bash
python simulate_optimized.py
```

4. Compare both:

```bash
python compare_results.py
```

## Mental Model Summary

If you want the shortest accurate summary, it is this:

- SUMO simulates cars and intersections.
- `sumo_env.py` turns that simulation into a 15-number state.
- `dqn_agent.py` predicts which of 4 traffic-light actions is best.
- `reward.py` tells the agent whether the result was good or bad.
- `replay_buffer.py` stores experiences so learning is more stable.
- `train.py` repeats this loop over many episodes until the Q-network becomes a useful traffic-control policy.

That is the model from scratch in this repository.
