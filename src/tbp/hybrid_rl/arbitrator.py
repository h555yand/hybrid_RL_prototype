"""
Arbitrator: decides which action source to use per step.
Switches between Q-store (episodic memory), SAC (skill),
and heuristic (fallback).
"""

import numpy as np
import torch
import logging
from typing import Dict, Any, Optional, Tuple

from .rl_goal_approach_controller import RLGoalApproachController
from .sac_actor import SACActorNetwork
from .experience_extractor import ExperienceExtractor

logger = logging.getLogger(__name__)


class Arbitrator:

    def __init__(
        self,
        controller: RLGoalApproachController,
        sac_actor: Optional[SACActorNetwork] = None,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
        param_mean: Optional[np.ndarray] = None,
        param_std: Optional[np.ndarray] = None,
        q_confidence_threshold: float = 0.5,
        sac_confidence_threshold: float = 0.7,
    ):
        self.controller = controller
        self.sac_actor = sac_actor
        self.state_mean = state_mean
        self.state_std = state_std
        self.param_mean = param_mean
        self.param_std = param_std
        self.q_confidence_threshold = q_confidence_threshold
        self.sac_confidence_threshold = sac_confidence_threshold

        self.stats = {
            "q_store_chosen": 0,
            "sac_chosen": 0,
            "heuristic_chosen": 0,
            "total_decisions": 0,
        }

    def decide(
        self,
        state: np.ndarray,
        current_pose: np.ndarray,
        sensor_data: Dict[str, Any],
    ) -> Tuple[int, str]:
        self.stats["total_decisions"] += 1

        q_action, q_confidence = self._get_q_action(state)
        sac_action, sac_params, sac_confidence = self._get_sac_action(state)

        if q_confidence > self.q_confidence_threshold:
            if self.sac_actor is not None and sac_confidence > self.sac_confidence_threshold:
                store = self.controller._select_store(state)
                q_values = store.get_q_values(state)
                q_value_best = float(np.max(q_values))
                sac_q = float(q_values[sac_action]) if sac_action < len(q_values) else 0.0

                if q_value_best > sac_q:
                    self.stats["q_store_chosen"] += 1
                    return q_action, "q_store"
                else:
                    self.stats["sac_chosen"] += 1
                    return sac_action, "sac"
            else:
                self.stats["q_store_chosen"] += 1
                return q_action, "q_store"

        if self.sac_actor is not None and sac_confidence > self.sac_confidence_threshold:
            self.stats["sac_chosen"] += 1
            return sac_action, "sac"

        heuristic_action = self._get_heuristic_action(state, current_pose, sensor_data)
        self.stats["heuristic_chosen"] += 1
        return heuristic_action, "heuristic"

    def _get_q_action(self, state: np.ndarray) -> Tuple[int, float]:
        store = self.controller._select_store(state)

        if store.next_id == 0:
            return 0, 0.0

        norm_state = store._normalize(state)
        k = min(store.k_neighbors, store.next_id)

        labels, distances = store._index.knn_query(
            norm_state.reshape(1, -1), k=k
        )

        sigma = store._get_sigma(distances[0])
        weights = store._gaussian_kernel(distances[0], sigma)
        weight_sum = float(weights.sum())
        q_confidence = min(weight_sum / k, 1.0)

        q_values = store.get_q_values(state)
        q_action = int(np.argmax(q_values))

        return q_action, q_confidence

    def _get_sac_action(self, state: np.ndarray) -> Tuple[int, np.ndarray, float]:
        if self.sac_actor is None:
            return 0, np.zeros(3), 0.0

        state_norm = state
        if self.state_mean is not None:
            state_norm = (state - self.state_mean) / (self.state_std + 1e-8)

        self.sac_actor.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state_norm.astype(np.float32)).unsqueeze(0)
            type_logits, param_mus, _ = self.sac_actor(state_t)

            type_probs = torch.softmax(type_logits, dim=-1)[0]
            action_type = int(torch.argmax(type_probs).item())

            entropy = -float((type_probs * torch.log(type_probs + 1e-8)).sum())
            max_entropy = float(np.log(self.sac_actor.num_types))
            sac_confidence = 1.0 - entropy / max_entropy

            if action_type in param_mus:
                action_params = param_mus[action_type][0].numpy()
                if self.param_mean is not None:
                    param_dim = len(action_params)
                    action_params = action_params * self.param_std[:param_dim] + self.param_mean[:param_dim]
            else:
                action_params = np.zeros(3)

        return action_type, action_params, sac_confidence

    def _get_heuristic_action(
        self,
        state: np.ndarray,
        current_pose: np.ndarray,
        sensor_data: Dict[str, Any],
    ) -> int:
        heuristic, _ = self.controller._compute_heuristic_bias(
            state=state,
            current_pose=current_pose,
            sensor_data=sensor_data,
            prev_action=self.controller._last_action,
        )
        return int(np.argmax(heuristic))

    def get_stats(self) -> Dict[str, Any]:
        total = max(self.stats["total_decisions"], 1)
        return {
            **self.stats,
            "q_store_rate": self.stats["q_store_chosen"] / total,
            "sac_rate": self.stats["sac_chosen"] / total,
            "heuristic_rate": self.stats["heuristic_chosen"] / total,
        }
