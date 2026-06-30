# =============================================================================
# replay_buffer.py — The agent's memory.
#
# WHY DO WE NEED THIS?
# If you train the agent on each experience immediately and throw it away,
# two problems happen:
#   1. Consecutive experiences are highly correlated (t=1 and t=2 look almost
#      identical), so the neural net just memorizes recent situations and
#      "forgets" earlier ones. This is called catastrophic forgetting.
#   2. The learning signal is very noisy (one experience at a time = high variance).
#
# Solution: store ALL experiences in a big buffer, then at each learning step
# sample a RANDOM BATCH from it. This breaks correlation and stabilizes training.
# This technique is called "Experience Replay" — it's one of the two key
# innovations in the original DQN paper (DeepMind, 2015).
# =============================================================================

import random
import numpy as np
from collections import deque
from typing import Tuple


class ReplayBuffer:
    """
    A circular buffer that stores (state, action, reward, next_state, done) tuples.
    'Circular' means when it's full, the oldest entry is automatically overwritten.
    """

    def __init__(self, capacity: int):
        # deque with maxlen = circular buffer behavior for free
        # When full and you append, the leftmost (oldest) element is dropped
        self.buffer = deque(maxlen=capacity)

    def push(self,
             state:      np.ndarray,
             action:     int,
             reward:     float,
             next_state: np.ndarray,
             done:       bool) -> None:
        """
        Store one experience tuple.
        'done' = True means the episode ended at this step.
        WHY track done? Because Q(s', a') should be 0 if s' is terminal —
        there is no future reward after the episode ends.
        """
        self.buffer.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done)
        ))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Randomly sample a batch of experiences.
        Returns 5 numpy arrays, one per field, each of length batch_size.
        WHY return numpy arrays? Faster to convert to tensors than lists.
        """
        batch = random.sample(self.buffer, batch_size)

        # Unzip: list of tuples → tuple of lists → numpy arrays
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.stack(states),                          # shape: (B, STATE_DIM)
            np.array(actions,  dtype=np.int64),        # shape: (B,)
            np.array(rewards,  dtype=np.float32),      # shape: (B,)
            np.stack(next_states),                     # shape: (B, STATE_DIM)
            np.array(dones,    dtype=np.float32),      # shape: (B,)  0.0 or 1.0
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def ready(self, min_size: int) -> bool:
        """Returns True once we have enough memories to start learning."""
        return len(self.buffer) >= min_size
