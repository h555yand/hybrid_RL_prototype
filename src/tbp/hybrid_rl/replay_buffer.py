# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Replay Buffer for P-SAC training.
Stores transitions in P-SAC format (type + continuous params).
Supports warm-start from BC data with protected BC reservoir.
"""

from typing import Dict, List
import logging

import numpy as np

from .experience_extractor import PSACTransition

logger = logging.getLogger(__name__)


class ReplayBuffer:

    def __init__(
        self,
        capacity: int = 100_000,
        state_dim: int = 15,
        max_params: int = 3,
        bc_reserve_fraction: float = 0.15,
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.max_params = max_params

        self.bc_reserve = int(capacity * bc_reserve_fraction)
        self.bc_size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action_types = np.zeros(capacity, dtype=np.int64)
        self.action_params = np.zeros((capacity, max_params), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.online_pos = 0
        self.online_capacity = capacity
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
        idx = self.bc_size + (self.online_pos % self.online_capacity)

        self.states[idx] = state
        self.action_types[idx] = action_type
        self.action_params[idx, :len(action_params)] = action_params
        self.action_params[idx, len(action_params):] = 0.0
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)

        self.online_pos += 1
        self.size = min(self.bc_size + self.online_pos, self.capacity)

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
        for tr in transitions:
            if tr.next_state is None:
                continue
            if count >= self.bc_reserve:
                break

            idx = count
            self.states[idx] = tr.state
            self.action_types[idx] = tr.action_type
            self.action_params[idx, :len(tr.action_params)] = tr.action_params
            self.action_params[idx, len(tr.action_params):] = 0.0
            self.rewards[idx] = tr.reward
            self.next_states[idx] = tr.next_state
            self.dones[idx] = float(tr.done)
            count += 1

        self.bc_size = count
        self.online_capacity = self.capacity - self.bc_size
        self.size = self.bc_size
        logger.info(
            f"ReplayBuffer: loaded {count} BC transitions (protected), "
            f"online_capacity={self.online_capacity}"
        )

    def __len__(self):
        return self.size
