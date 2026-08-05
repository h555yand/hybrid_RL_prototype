# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Memory-based store for strategic transition decisions.

Stores outcomes of past phase transitions (detach, direction switch)
and recommends whether to transition in similar states.

Uses HNSW for fast KNN lookup in compact state space.
Unlike full Q-store, stores single outcome value per point,
not Q-values for multiple actions.

Two instances used:
    transition_detach: Should agent detach from surface?
    transition_direction: Should agent switch from bypass to direct flight?
"""

import logging
import pathlib
from collections import deque
from typing import Dict, Optional, Tuple

import hnswlib
import numpy as np

logger = logging.getLogger(__name__)


class TransitionMemory:
    """Memory-based store for strategic transition decisions.

    Stores (compact_state, outcome) pairs from past transitions.
    On query, finds similar past states and returns weighted
    average outcome with confidence estimate.

    Outcome convention:
        +1.0 = transition led to goal_reached
        -1.0 = transition led to collision
        -0.3 = transition led to timeout
         0.0 = no data

    Args:
        state_dim: Dimensionality of compact transition state.
        max_points: Maximum stored experiences.
        k_neighbors: Neighbors for KNN interpolation.
        insert_threshold: L2 distance below which update existing point.
        name: Store name for logging.
    """

    def __init__(
        self,
        state_dim: int = 5,
        max_points: int = 10000,
        k_neighbors: int = 5,
        insert_threshold: float = 0.3,
        name: str = "transition",
    ):
        self.state_dim = state_dim
        self.max_points = max_points
        self.k_neighbors = k_neighbors
        self.insert_threshold = insert_threshold
        self.name = name

        self._index = hnswlib.Index(space="l2", dim=state_dim)
        self._index.init_index(
            max_elements=max_points,
            ef_construction=100,
            M=16,
            allow_replace_deleted=True,
        )
        self._index.set_ef(30)

        self.points: Dict[int, dict] = {}
        self.next_id: int = 0

        # Normalization
        self._state_mean = np.zeros(state_dim)
        self._state_std = np.ones(state_dim)
        self._state_buffer: deque = deque(maxlen=2000)
        self._norm_frozen = False
        self._norm_min_samples = 50
        self._norm_update_interval = 50
        self._norm_warmup = 500

    def record(
        self,
        state: np.ndarray,
        outcome: float,
        alpha: float = 0.3,
    ):
        """Record outcome of a transition decision.

        If similar state exists (within insert_threshold), updates
        its outcome via exponential moving average. Otherwise inserts
        new point.

        Args:
            state: Compact transition state vector.
            outcome: Result of transition (+1 success, -1 collision).
            alpha: Learning rate for updating existing points.
        """
        self._update_normalization(state)
        norm_state = self._normalize(state)

        if self.next_id > 0:
            k = min(self.k_neighbors, self.next_id)
            labels, distances = self._index.knn_query(
                norm_state.reshape(1, -1), k=k
            )

            if distances[0][0] < self.insert_threshold ** 2:
                # Update existing point
                point = self.points[labels[0][0]]
                point["outcome"] += alpha * (
                    outcome - point["outcome"]
                )
                point["count"] += 1
                logger.debug(
                    f"TRANSITION_MEMORY({self.name}): updated "
                    f"point {labels[0][0]}, "
                    f"outcome={point['outcome']:.2f}, "
                    f"count={point['count']}"
                )
                return

        # Insert new point
        if len(self.points) >= self.max_points:
            self._evict_oldest()

        point_id = self.next_id
        self._index.add_items(
            norm_state.reshape(1, -1),
            np.array([point_id]),
        )
        self.points[point_id] = {
            "raw_state": state.copy(),
            "norm_state": norm_state.copy(),
            "outcome": outcome,
            "count": 1,
        }
        self.next_id += 1

        logger.debug(
            f"TRANSITION_MEMORY({self.name}): inserted "
            f"point {point_id}, outcome={outcome:.2f}, "
            f"total={self.next_id}"
        )

    def query(self, state: np.ndarray) -> Tuple[float, float]:
        """Query whether transition is recommended in this state.

        Finds K nearest neighbors, computes Gaussian-weighted
        average of their outcomes, and estimates confidence
        based on weight sum and total experience count.

        Args:
            state: Compact transition state vector.

        Returns:
            (recommendation, confidence)
            recommendation: Weighted outcome, positive = do transition.
            confidence: 0.0 = no relevant data, 1.0 = very confident.
        """
        if self.next_id == 0:
            return 0.0, 0.0

        norm_state = self._normalize(state)
        k = min(self.k_neighbors, self.next_id)
        labels, distances = self._index.knn_query(
            norm_state.reshape(1, -1), k=k
        )

        # Gaussian kernel weights
        actual_dists = np.sqrt(np.maximum(distances[0], 0))
        median_dist = max(np.median(actual_dists), 0.1)
        sigma = max(median_dist * 0.5, 0.1)
        weights = np.exp(-distances[0] / (2.0 * sigma ** 2))
        weight_sum = weights.sum()

        if weight_sum < 0.1:
            return 0.0, 0.0

        weights /= weight_sum

        # Weighted average of outcomes
        weighted_outcome = 0.0
        total_count = 0
        for i, label in enumerate(labels[0]):
            point = self.points[label]
            weighted_outcome += weights[i] * point["outcome"]
            total_count += point["count"]

        # Confidence based on weight quality and data quantity
        confidence = (
            min(weight_sum, 1.0)
            * min(total_count / 10.0, 1.0)
        )

        logger.debug(
            f"TRANSITION_MEMORY({self.name}): query "
            f"recommendation={weighted_outcome:.2f}, "
            f"confidence={confidence:.2f}, "
            f"nearest_dist={actual_dists[0]:.3f}, "
            f"weight_sum={weight_sum:.3f}"
        )

        return weighted_outcome, confidence

    def _evict_oldest(self):
        """Remove oldest points when at capacity."""
        if not self.points:
            return
        n_evict = max(1, len(self.points) // 10)
        sorted_ids = sorted(
            self.points.keys(),
            key=lambda pid: self.points[pid]["count"],
        )
        for pid in sorted_ids[:n_evict]:
            self._index.mark_deleted(pid)
            del self.points[pid]

    def _normalize(self, state: np.ndarray) -> np.ndarray:
        return (state - self._state_mean) / (self._state_std + 1e-8)

    def _update_normalization(self, state: np.ndarray):
        self._state_buffer.append(state.copy())
        if self._norm_frozen:
            return
        n = len(self._state_buffer)
        if n < self._norm_min_samples:
            return
        if n % self._norm_update_interval != 0:
            return
        buf = np.array(self._state_buffer)
        self._state_mean = buf.mean(axis=0)
        self._state_std = np.maximum(buf.std(axis=0), 1e-4)
        if n >= self._norm_warmup:
            self._norm_frozen = True
            logger.info(
                f"TransitionMemory({self.name}): "
                f"normalization frozen on {n} samples"
            )

    def get_stats(self) -> dict:
        """Return diagnostic statistics."""
        if not self.points:
            return {
                "name": self.name,
                "num_points": 0,
                "total_recorded": self.next_id,
            }

        outcomes = [p["outcome"] for p in self.points.values()]
        counts = [p["count"] for p in self.points.values()]

        return {
            "name": self.name,
            "num_points": len(self.points),
            "total_recorded": self.next_id,
            "outcome_mean": float(np.mean(outcomes)),
            "outcome_positive_ratio": float(
                np.mean([1 if o > 0 else 0 for o in outcomes])
            ),
            "count_mean": float(np.mean(counts)),
            "count_max": int(np.max(counts)),
        }

    def save(self, filepath: str):
        """Save to disk."""
        pathlib.Path(filepath).parent.mkdir(
            parents=True, exist_ok=True
        )

        if not self.points:
            np.savez(
                filepath,
                state_dim=np.array([self.state_dim]),
                next_id=np.array([self.next_id]),
                state_mean=self._state_mean,
                state_std=self._state_std,
            )
            return

        ids = sorted(self.points.keys())
        raw_states = np.array(
            [self.points[i]["raw_state"] for i in ids]
        )
        outcomes = np.array(
            [self.points[i]["outcome"] for i in ids]
        )
        counts = np.array(
            [self.points[i]["count"] for i in ids]
        )

        np.savez(
            filepath,
            state_dim=np.array([self.state_dim]),
            next_id=np.array([self.next_id]),
            state_mean=self._state_mean,
            state_std=self._state_std,
            point_ids=np.array(ids),
            raw_states=raw_states,
            outcomes=outcomes,
            counts=counts,
        )
        logger.info(
            f"TransitionMemory({self.name}): saved "
            f"{len(ids)} points to {filepath}"
        )

    @classmethod
    def load(cls, filepath: str, name: str = "transition") -> "TransitionMemory":
        """Load from disk."""
        data = np.load(filepath, allow_pickle=False)

        state_dim = int(data["state_dim"][0])
        store = cls(state_dim=state_dim, name=name)
        store._state_mean = data["state_mean"]
        store._state_std = data["state_std"]
        store._norm_frozen = True

        if "raw_states" in data:
            raw_states = data["raw_states"]
            outcomes = data["outcomes"]
            counts = data["counts"]
            point_ids = (
                data["point_ids"]
                if "point_ids" in data
                else np.arange(len(raw_states))
            )

            for i in range(len(raw_states)):
                pid = int(point_ids[i])
                norm_state = store._normalize(raw_states[i])
                store._index.add_items(
                    norm_state.reshape(1, -1),
                    np.array([pid]),
                )
                store.points[pid] = {
                    "raw_state": raw_states[i].copy(),
                    "norm_state": norm_state.copy(),
                    "outcome": float(outcomes[i]),
                    "count": int(counts[i]),
                }

            store.next_id = int(data["next_id"][0])

        logger.info(
            f"TransitionMemory({name}): loaded "
            f"{len(store.points)} points from {filepath}"
        )
        return store