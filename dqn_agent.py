# =============================================================================
# dqn_agent.py — The brain. Neural network + DQN learning logic.
#
# WHAT IS DQN?
# Q-learning asks: "What is the total future reward I'll get if I'm in state S
# and take action A, then act optimally forever after?"
# We call this Q(S, A) — the "action-value function."
#
# The Bellman equation tells us how Q should relate across timesteps:
#   Q(s, a) = r + γ * max_a'[ Q(s', a') ]
#   "The value of (s,a) = immediate reward + discounted best value of next state"
#
# DQN approximates Q with a neural network instead of a table.
# Two key tricks that make it work:
#   1. Experience Replay (replay_buffer.py) — breaks correlation in training data
#   2. Target Network — a FROZEN copy of the network used to compute Q(s', a')
#      WHY freeze it? If you use the same network for both prediction AND target,
#      you're chasing a moving target — training becomes unstable.
#      Solution: update the target network only every N steps.
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from typing import Tuple

import config as cfg
from replay_buffer import ReplayBuffer


# -----------------------------------------------------------------------------
# The Q-Network: a simple 3-layer MLP
# Input: state vector (STATE_DIM floats)
# Output: Q-value for each action (NUM_ACTIONS floats)
# WHY MLP and not CNN/RNN? Our state is a flat feature vector, not an image
# or sequence. MLP is the right tool here.
# -----------------------------------------------------------------------------
class QNetwork(nn.Module):

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int):
        super().__init__()

        # Three fully-connected layers with ReLU activations
        # WHY ReLU? It's simple, avoids vanishing gradients, trains fast.
        # WHY 3 layers? Enough to learn non-linear Q-value landscapes
        # without being overkill for a 14-feature input.
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),   # 14 → 128
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # 128 → 128
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions), # 128 → 4  (one Q-value per action)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, state_dim)
        # output shape: (batch_size, num_actions)
        return self.net(x)


# -----------------------------------------------------------------------------
# The DQN Agent: wraps two networks + replay buffer + epsilon-greedy policy
# -----------------------------------------------------------------------------
class DQNAgent:

    def __init__(self):
        # Use GPU if available, otherwise CPU
        # WHY GPU? Matrix multiplications are ~100x faster on GPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQN] Using device: {self.device}")

        # Online network: the one we're actively training
        self.online_net = QNetwork(cfg.STATE_DIM, cfg.NUM_ACTIONS, cfg.HIDDEN_DIM).to(self.device)

        # Target network: frozen copy, used only for computing TD targets
        # Initialized as an EXACT copy of online_net
        self.target_net = QNetwork(cfg.STATE_DIM, cfg.NUM_ACTIONS, cfg.HIDDEN_DIM).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # Freeze: don't compute gradients for this network

        # Adam optimizer — works well for non-stationary RL objectives
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=cfg.LEARNING_RATE)

        # Experience replay buffer
        self.memory = ReplayBuffer(cfg.REPLAY_CAPACITY)

        # Epsilon for epsilon-greedy exploration
        # Starts at 1.0 (fully random) and decays toward EPS_END
        self.epsilon = cfg.EPS_START

        # Step counter (used to trigger target network updates)
        self.steps_done = 0

    # -------------------------------------------------------------------------
    # ACTION SELECTION: epsilon-greedy policy
    # -------------------------------------------------------------------------
    def select_action(self, state: np.ndarray) -> int:
        """
        With probability epsilon: pick a RANDOM action (explore)
        Otherwise: pick the action with highest Q-value (exploit)

        WHY epsilon-greedy?
        Early in training the Q-values are garbage (random network weights),
        so we should mostly explore. As training progresses and Q-values
        get better, we should mostly exploit. Epsilon decay handles this.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(cfg.NUM_ACTIONS)   # Random exploration

        # Convert state to tensor, add batch dimension (1, STATE_DIM)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # No gradient needed for inference — saves memory and is faster
        with torch.no_grad():
            q_values = self.online_net(state_t)  # shape: (1, NUM_ACTIONS)

        # Pick action with highest Q-value
        # .item() converts a 1-element tensor to a Python int
        return int(q_values.argmax(dim=1).item())

    def decay_epsilon(self):
        """Call once per step to slowly reduce exploration."""
        self.epsilon = max(cfg.EPS_END, self.epsilon * cfg.EPS_DECAY)

    # -------------------------------------------------------------------------
    # LEARNING: one gradient update step
    # -------------------------------------------------------------------------
    def learn(self) -> float:
        """
        Sample a batch from memory, compute TD loss, backpropagate.
        Returns the loss value (for logging).

        THE MATH:
        For each experience (s, a, r, s', done) in the batch:
          - Prediction: Q_online(s)[a]   ← what the network currently thinks
          - Target:     r + γ * max_a'[Q_target(s')]  ← what it SHOULD be
                        (if done=True, target = r only, no future)
          - Loss: mean squared error between prediction and target
          - Backprop: nudge the online network's weights toward the target
        """
        if not self.memory.ready(cfg.MIN_REPLAY_SIZE):
            return 0.0  # Not enough memories yet

        # Sample random batch
        states, actions, rewards, next_states, dones = self.memory.sample(cfg.BATCH_SIZE)

        # Convert numpy arrays to tensors and move to device
        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # --- Compute predictions ---
        # Q_online(s) gives Q-values for ALL actions
        # We only care about Q-value of the action that was actually taken
        # .gather(1, ...) selects the Q-value at the chosen action index
        all_q = self.online_net(states_t)                    # (B, NUM_ACTIONS)
        pred_q = all_q.gather(1, actions_t.unsqueeze(1))     # (B, 1)
        pred_q = pred_q.squeeze(1)                           # (B,)

        # --- Compute targets ---
        # target_net has no_grad because we don't want gradients flowing through it
        with torch.no_grad():
            next_q = self.target_net(next_states_t)          # (B, NUM_ACTIONS)
            max_next_q = next_q.max(dim=1).values            # (B,)  best action's Q
            # If done=1, there's no future — multiply future term by (1 - done)
            target_q = rewards_t + cfg.GAMMA * max_next_q * (1.0 - dones_t)

        # --- Compute loss and backprop ---
        # Huber loss (smooth_l1) instead of MSE:
        # WHY Huber? For large errors, MSE grows quadratically (can cause huge
        # gradients and destabilize training). Huber is linear for large errors.
        loss = F.smooth_l1_loss(pred_q, target_q)

        self.optimizer.zero_grad()   # Clear gradients from last step
        loss.backward()              # Compute gradients via backprop

        # Gradient clipping: cap gradient norm at 10
        # WHY? Prevents the "exploding gradient" problem where one bad batch
        # causes a huge weight update that destroys the network
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)

        self.optimizer.step()        # Update weights

        # --- Periodically sync target network ---
        self.steps_done += 1
        if self.steps_done % cfg.TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    # -------------------------------------------------------------------------
    # MEMORY: push one experience
    # -------------------------------------------------------------------------
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    # -------------------------------------------------------------------------
    # SAVE / LOAD checkpoints
    # -------------------------------------------------------------------------
    def save(self, path: str = cfg.MODEL_SAVE_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "online_net":  self.online_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "epsilon":     self.epsilon,
            "steps_done":  self.steps_done,
        }, path)
        print(f"[DQN] Checkpoint saved → {path}")

    def load(self, path: str = cfg.MODEL_SAVE_PATH):
        if not os.path.exists(path):
            print(f"[DQN] No checkpoint found at {path}, starting fresh.")
            return
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon    = ckpt["epsilon"]
        self.steps_done = ckpt["steps_done"]
        print(f"[DQN] Checkpoint loaded ← {path}  (ε={self.epsilon:.3f}, steps={self.steps_done})")
