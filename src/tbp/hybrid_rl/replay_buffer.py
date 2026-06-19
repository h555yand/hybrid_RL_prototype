"""
Replay Buffer for P-SAC training.
Stores transitions in P-SAC format (type + continuous params).
Supports warm-start from BC data.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from .experience_extractor import PSACTransition


class ReplayBuffer:

    def __init__(
        self,
        capacity: int = 100_000,
        state_dim: int = 15,
        max_params: int = 3,
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.max_params = max_params

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action_types = np.zeros(capacity, dtype=np.int64)
        self.action_params = np.zeros((capacity, max_params), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.pos = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action_type: int,
        action_params: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        idx = self.pos % self.capacity

        self.states[idx] = state
        self.action_types[idx] = action_type
        self.action_params[idx, :len(action_params)] = action_params
        self.action_params[idx, len(action_params):] = 0.0
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)

        self.pos += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "states": self.states[indices],
            "action_types": self.action_types[indices],
            "action_params": self.action_params[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
        }

    def load_bc_data(self, transitions: List[PSACTransition]):
        count = 0
        for i, tr in enumerate(transitions):
            if tr.next_state is None:
                continue
            self.add(
                state=tr.state,
                action_type=tr.action_type,
                action_params=tr.action_params,
                reward=tr.reward,
                next_state=tr.next_state,
                done=tr.done,
            )
            count += 1
        print(f"ReplayBuffer: loaded {count} BC transitions (size={self.size})")

    def __len__(self):
        return self.size