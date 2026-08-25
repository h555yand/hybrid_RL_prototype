# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Replay Buffer for P-SAC training.

ObjectPartitionedBuffer: per-object buckets with balanced sampling.
Supports warm-start from BC data with protected BC reservoir per object.
"""

from typing import Any, Dict, List, Optional
import logging

import numpy as np

from .experience_extractor import ExperienceExtractor, PSACTransition

logger = logging.getLogger(__name__)


class ObjectBucket:
    """FIFO circular buffer for a single object."""

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        max_params: int,
    ):
        self.capacity = capacity
        self.states = np.zeros(
            (capacity, state_dim), dtype=np.float32
        )
        self.action_types = np.zeros(
            capacity, dtype=np.int64
        )
        self.action_params = np.zeros(
            (capacity, max_params), dtype=np.float32
        )
        self.rewards = np.zeros(
            capacity, dtype=np.float32
        )
        self.next_states = np.zeros(
            (capacity, state_dim), dtype=np.float32
        )
        self.dones = np.zeros(
            capacity, dtype=np.float32
        )
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
        p_len = len(action_params)
        self.action_params[idx, :p_len] = action_params
        self.action_params[idx, p_len:] = 0.0
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        self.pos += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, n: int
    ) -> Dict[str, np.ndarray]:
        indices = np.random.randint(
            0, self.size, size=n
        )
        return {
            "states": self.states[indices],
            "action_types": self.action_types[indices],
            "action_params": self.action_params[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
        }

    def __len__(self):
        return self.size


class ObjectPartitionedBuffer:
    """Replay buffer partitioned by object (mesh).

    Each object gets its own FIFO bucket + optional elite bucket.
    BC data stored in separate protected buckets per object.
    Supports balanced sampling across objects during training.

    Drop-in replacement for ReplayBuffer — same add/sample API
    plus mesh-aware methods.
    """

    def __init__(
        self,
        capacity_per_object: int = 50_000,
        state_dim: int = 22,
        max_params: int = 3,
        bc_capacity_per_object: int = 5_000,
        elite_capacity_per_object: int = 5_000,
    ):
        self.capacity_per_object = capacity_per_object
        self.state_dim = state_dim
        self.max_params = max_params
        self.bc_capacity_per_object = bc_capacity_per_object
        self.elite_capacity_per_object = (
            elite_capacity_per_object
        )

        # Per-object online buckets
        self.buckets: Dict[str, ObjectBucket] = {}
        # Per-object BC buckets (protected)
        self.bc_buckets: Dict[str, ObjectBucket] = {}
        # Per-object elite buckets (best episodes)
        self.elite_buckets: Dict[str, ObjectBucket] = {}

        # Current mesh for default add()
        self._current_mesh: str = ""

        # Mesh ID to name mapping
        self._mesh_id_to_name = {
            v: k
            for k, v in (
                ExperienceExtractor
                .MESH_NAME_TO_ID.items()
            )
        }

    def _ensure_bucket(self, mesh_name: str):
        """Create buckets for a mesh if they don't exist."""
        if mesh_name not in self.buckets:
            self.buckets[mesh_name] = ObjectBucket(
                self.capacity_per_object,
                self.state_dim,
                self.max_params,
            )
        if mesh_name not in self.bc_buckets:
            self.bc_buckets[mesh_name] = ObjectBucket(
                self.bc_capacity_per_object,
                self.state_dim,
                self.max_params,
            )
        if mesh_name not in self.elite_buckets:
            self.elite_buckets[mesh_name] = ObjectBucket(
                self.elite_capacity_per_object,
                self.state_dim,
                self.max_params,
            )

    def set_current_mesh(self, mesh_name: str):
        """Set current mesh for default add() calls."""
        self._current_mesh = mesh_name
        self._ensure_bucket(mesh_name)

    def add(
        self,
        state: np.ndarray,
        action_type: int,
        action_params: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        mesh_name: Optional[str] = None,
    ):
        """Add transition to the appropriate object bucket.

        Args:
            state: Normalized state.
            action_type: Action type index.
            action_params: Normalized action parameters.
            reward: Reward value.
            next_state: Normalized next state.
            done: Episode done flag.
            mesh_name: Object name. If None, uses current mesh.
        """
        name = mesh_name or self._current_mesh
        if not name:
            name = "unknown"
        self._ensure_bucket(name)
        self.buckets[name].add(
            state, action_type, action_params,
            reward, next_state, done,
        )

    def add_elite(
        self,
        state: np.ndarray,
        action_type: int,
        action_params: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        mesh_name: Optional[str] = None,
    ):
        """Add transition to elite bucket (successful episodes)."""
        name = mesh_name or self._current_mesh
        if not name:
            name = "unknown"
        self._ensure_bucket(name)
        self.elite_buckets[name].add(
            state, action_type, action_params,
            reward, next_state, done,
        )

    def sample(
        self, batch_size: int
    ) -> Dict[str, np.ndarray]:
        """Uniform sample from all data (backward compatible).

        Samples uniformly from all online + BC + elite buckets.
        Used when no mesh-aware sampling is needed.
        """
        all_sizes = []
        all_buckets = []

        for name in self.buckets:
            for bucket_dict in (
                self.buckets,
                self.bc_buckets,
                self.elite_buckets,
            ):
                b = bucket_dict.get(name)
                if b is not None and b.size > 0:
                    all_sizes.append(b.size)
                    all_buckets.append(b)

        if not all_buckets:
            # Empty buffer — return zeros
            return {
                "states": np.zeros(
                    (batch_size, self.state_dim),
                    dtype=np.float32,
                ),
                "action_types": np.zeros(
                    batch_size, dtype=np.int64
                ),
                "action_params": np.zeros(
                    (batch_size, self.max_params),
                    dtype=np.float32,
                ),
                "rewards": np.zeros(
                    batch_size, dtype=np.float32
                ),
                "next_states": np.zeros(
                    (batch_size, self.state_dim),
                    dtype=np.float32,
                ),
                "dones": np.zeros(
                    batch_size, dtype=np.float32
                ),
            }

        total = sum(all_sizes)
        probs = [s / total for s in all_sizes]

        # Allocate samples per bucket proportionally
        counts = np.random.multinomial(
            batch_size, probs
        )

        batches = []
        for bucket, count in zip(
            all_buckets, counts
        ):
            if count > 0:
                batches.append(bucket.sample(count))

        return self._merge_batches(batches)

    def sample_balanced(
        self,
        batch_size: int,
        current_mesh: str,
        current_ratio: float = 0.5,
        bc_ratio: float = 0.1,
        elite_ratio: float = 0.1,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Balanced sampling across objects.

        Args:
            batch_size: Total batch size.
            current_mesh: Current training mesh name.
            current_ratio: Fraction from current mesh online bucket.
            bc_ratio: Fraction from BC buckets (all objects).
            elite_ratio: Fraction from elite buckets (all objects).

        Remaining fraction split equally among other objects'
        online buckets.

        Returns:
            Merged batch dict.
        """
        n_current = int(batch_size * current_ratio)
        n_bc = int(batch_size * bc_ratio)
        n_elite = int(batch_size * elite_ratio)
        n_others = batch_size - n_current - n_bc - n_elite

        batches = []

        # 1. Current mesh online bucket
        current_bucket = self.buckets.get(current_mesh)
        if (
            current_bucket is not None
            and current_bucket.size > 0
            and n_current > 0
        ):
            batches.append(
                current_bucket.sample(n_current)
            )
        else:
            n_others += n_current

        # 2. BC buckets (all objects, uniform)
        if n_bc > 0:
            bc_batch = self._sample_from_bucket_dict(
                self.bc_buckets, n_bc
            )
            if bc_batch is not None:
                batches.append(bc_batch)
            else:
                n_others += n_bc

        # 3. Elite buckets (all objects, uniform)
        if n_elite > 0:
            elite_batch = self._sample_from_bucket_dict(
                self.elite_buckets, n_elite
            )
            if elite_batch is not None:
                batches.append(elite_batch)
            else:
                n_others += n_elite

        # 4. Other objects' online buckets (uniform)
        if n_others > 0:
            other_buckets = {
                name: b
                for name, b in self.buckets.items()
                if name != current_mesh and b.size > 0
            }
            if other_buckets:
                other_batch = (
                    self._sample_from_bucket_dict(
                        other_buckets, n_others
                    )
                )
                if other_batch is not None:
                    batches.append(other_batch)
            elif (
                current_bucket is not None
                and current_bucket.size > 0
            ):
                # No other objects yet — sample more
                # from current
                batches.append(
                    current_bucket.sample(n_others)
                )

        if not batches:
            return self.sample(batch_size)

        return self._merge_batches(batches)

    def _sample_from_bucket_dict(
        self,
        bucket_dict: Dict[str, ObjectBucket],
        n: int,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Sample uniformly from all non-empty buckets."""
        non_empty = {
            name: b
            for name, b in bucket_dict.items()
            if b.size > 0
        }
        if not non_empty:
            return None

        names = list(non_empty.keys())
        sizes = [non_empty[name].size for name in names]
        total = sum(sizes)
        probs = [s / total for s in sizes]

        counts = np.random.multinomial(n, probs)

        batches = []
        for name, count in zip(names, counts):
            if count > 0:
                batches.append(
                    non_empty[name].sample(count)
                )

        return self._merge_batches(batches)

    @staticmethod
    def _merge_batches(
        batches: List[Dict[str, np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        """Merge multiple batch dicts into one."""
        if len(batches) == 1:
            return batches[0]

        merged = {}
        for key in batches[0]:
            merged[key] = np.concatenate(
                [b[key] for b in batches], axis=0
            )

        # Shuffle
        n = len(merged["rewards"])
        perm = np.random.permutation(n)
        return {k: v[perm] for k, v in merged.items()}

    def load_bc_data(
        self, transitions: List[PSACTransition]
    ):
        """Load BC transitions into per-object BC buckets.

        Args:
            transitions: Normalized BC transitions with mesh_id.
        """
        count = 0
        mesh_counts: Dict[str, int] = {}

        for tr in transitions:
            if tr.next_state is None:
                continue

            mesh_name = self._mesh_id_to_name.get(
                tr.mesh_id, "unknown"
            )
            self._ensure_bucket(mesh_name)

            bc_bucket = self.bc_buckets[mesh_name]
            if bc_bucket.size >= bc_bucket.capacity:
                continue

            bc_bucket.add(
                tr.state,
                tr.action_type,
                tr.action_params,
                tr.reward,
                tr.next_state,
                float(tr.done),
            )
            count += 1
            mesh_counts[mesh_name] = (
                mesh_counts.get(mesh_name, 0) + 1
            )

        logger.info(
            "ObjectPartitionedBuffer: loaded %d BC "
            "transitions across %d objects: %s",
            count,
            len(mesh_counts),
            {
                k: v
                for k, v in sorted(
                    mesh_counts.items()
                )
            },
        )

    @property
    def size(self) -> int:
        """Total transitions across all buckets."""
        total = 0
        for bucket_dict in (
            self.buckets,
            self.bc_buckets,
            self.elite_buckets,
        ):
            for b in bucket_dict.values():
                total += b.size
        return total

    def __len__(self):
        return self.size

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        stats = {
            "total_size": self.size,
            "objects": {},
        }
        for name in sorted(
            set(self.buckets.keys())
            | set(self.bc_buckets.keys())
            | set(self.elite_buckets.keys())
        ):
            stats["objects"][name] = {
                "online": (
                    self.buckets[name].size
                    if name in self.buckets
                    else 0
                ),
                "bc": (
                    self.bc_buckets[name].size
                    if name in self.bc_buckets
                    else 0
                ),
                "elite": (
                    self.elite_buckets[name].size
                    if name in self.elite_buckets
                    else 0
                ),
            }
        return stats


# Backward compatibility alias
ReplayBuffer = ObjectPartitionedBuffer
