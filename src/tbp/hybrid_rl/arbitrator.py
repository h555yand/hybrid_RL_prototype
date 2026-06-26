"""
Arbitrator: decides which action source to use per step.
Switches between Q-store (episodic memory), SAC (skill),
and heuristic (fallback).

Logic:
  1. Q-store confident AND distinguishes actions → Q-store (familiar territory)
  2. SAC confident → SAC (generalizes to unfamiliar states)
  3. Q-store has some data → Q-store (weak fallback)
  4. Heuristic (last resort)
"""

import numpy as np
import torch
import logging
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict

from .rl_goal_approach_controller import RLGoalApproachController
from .sac_actor import SACActorNetwork
from .experience_extractor import ExperienceExtractor

logger = logging.getLogger(__name__)


def sac_to_discrete(action_type: int, action_params: np.ndarray) -> int:
    if action_type == 0:
        angle = float(action_params[0]) % 360.0
        directions = [0, 45, 90, 135, 180, 225, 270, 315]
        best_idx = 0
        best_diff = 360.0
        for i, d in enumerate(directions):
            diff = abs(angle - d)
            diff = min(diff, 360 - diff)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        return best_idx

    elif action_type == 1:
        return 8 if float(action_params[0]) >= 0 else 9

    elif action_type == 2:
        return 10 if float(action_params[0]) >= 0 else 11

    elif action_type == 3:
        return 12 if float(action_params[0]) >= 0 else 13

    elif action_type == 4:
        return 14 if float(action_params[0]) >= 0 else 15

    elif action_type == 5:
        return 16

    elif action_type == 6:
        return 17

    elif action_type == 7:
        return 18

    return 0


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
        q_spread_threshold: float = 1.0,
        sac_confidence_threshold: float = 0.3,
        q_weak_threshold: float = 0.2,
    ):
        self.controller = controller
        self.sac_actor = sac_actor
        self.state_mean = state_mean
        self.state_std = state_std
        self.param_mean = param_mean
        self.param_std = param_std
        self.q_confidence_threshold = q_confidence_threshold
        self.q_spread_threshold = q_spread_threshold
        self.sac_confidence_threshold = sac_confidence_threshold
        self.q_weak_threshold = q_weak_threshold

        self.stats = {
            "q_store_chosen": 0,
            "q_store_weak_chosen": 0,
            "sac_chosen": 0,
            "heuristic_chosen": 0,
            "total_decisions": 0,
        }

        # Статистика предложенных действий (что каждый источник хотел)
        self.q_proposed_actions = defaultdict(int)
        self.sac_proposed_actions = defaultdict(int)

        # Статистика выбранных действий (что реально выполнилось)
        self.q_chosen_actions = defaultdict(int)
        self.sac_chosen_actions = defaultdict(int)
        self.heuristic_chosen_actions = defaultdict(int)

        # Согласованность: как часто Q и SAC предлагают одно и то же
        self.agreement_count = 0
        self.both_proposed_count = 0

        # Q-store диагностика
        self.q_confidence_history = []
        self.q_spread_history = []
        self.sac_confidence_history = []

    def decide(
        self,
        state: np.ndarray,
        current_pose: np.ndarray,
        sensor_data: Dict[str, Any],
    ) -> Tuple[int, str]:
        self.stats["total_decisions"] += 1

        q_action, q_confidence, q_spread = self._get_q_action(state)
        sac_action, sac_confidence = self._get_sac_action_discrete(state)

        action_space = self.controller.action_space
        q_name = action_space.get_info(q_action).name
        sac_name = action_space.get_info(sac_action).name

        # Записываем что каждый предложил
        self.q_proposed_actions[q_name] += 1
        if self.sac_actor is not None:
            self.sac_proposed_actions[sac_name] += 1

        # Согласованность
        if self.sac_actor is not None:
            self.both_proposed_count += 1
            if q_action == sac_action:
                self.agreement_count += 1

        # Диагностика (храним последние 1000)
        if len(self.q_confidence_history) < 1000:
            self.q_confidence_history.append(q_confidence)
            self.q_spread_history.append(q_spread)
            if self.sac_actor is not None:
                self.sac_confidence_history.append(sac_confidence)

        # 1. Q-store уверен И различает действия
        if q_confidence > self.q_confidence_threshold and q_spread > self.q_spread_threshold:
            self.stats["q_store_chosen"] += 1
            self.q_chosen_actions[q_name] += 1
            return q_action, "q_store"

        # 2. SAC уверен
        if self.sac_actor is not None and sac_confidence > self.sac_confidence_threshold:
            self.stats["sac_chosen"] += 1
            self.sac_chosen_actions[sac_name] += 1
            return sac_action, "sac"

        # 3. Q-store слабый fallback
        if q_confidence > self.q_weak_threshold:
            self.stats["q_store_weak_chosen"] += 1
            self.q_chosen_actions[q_name] += 1
            return q_action, "q_store_weak"

        # 4. Heuristic
        heuristic_action = self._get_heuristic_action(state, current_pose, sensor_data)
        h_name = action_space.get_info(heuristic_action).name
        self.stats["heuristic_chosen"] += 1
        self.heuristic_chosen_actions[h_name] += 1
        return heuristic_action, "heuristic"

    def _get_q_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        store = self.controller._select_store(state)

        if store.next_id == 0:
            return 0, 0.0, 0.0

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
        q_spread = float(np.max(q_values) - np.min(q_values))

        return q_action, q_confidence, q_spread

    def _get_sac_action_discrete(self, state: np.ndarray) -> Tuple[int, float]:
        if self.sac_actor is None:
            return 0, 0.0

        state_norm = state
        if self.state_mean is not None:
            state_norm = (state - self.state_mean) / (self.state_std + 1e-8)

        self.sac_actor.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state_norm.astype(np.float32)).unsqueeze(0)

            action_type_t, action_params_t, _, type_probs = self.sac_actor.sample(state_t)

            action_type = action_type_t[0].item()
            action_params_norm = action_params_t[0].numpy()

            param_dims = ExperienceExtractor.get_param_dims()
            dim = param_dims.get(action_type, 0)
            if dim > 0 and self.param_mean is not None:
                action_params = action_params_norm[:dim] * self.param_std[:dim] + self.param_mean[:dim]
            else:
                action_params = np.zeros(3)

            discrete_action = sac_to_discrete(action_type, action_params)

            sac_confidence = float(type_probs[0].max().item())

        return discrete_action, sac_confidence

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

        agreement_rate = (
            self.agreement_count / max(self.both_proposed_count, 1)
        )

        q_conf_mean = float(np.mean(self.q_confidence_history)) if self.q_confidence_history else 0.0
        q_spread_mean = float(np.mean(self.q_spread_history)) if self.q_spread_history else 0.0
        sac_conf_mean = float(np.mean(self.sac_confidence_history)) if self.sac_confidence_history else 0.0

        def _top_actions(action_dict, n=5):
            sorted_actions = sorted(action_dict.items(), key=lambda x: -x[1])
            total_actions = max(sum(action_dict.values()), 1)
            return {
                name: {"count": count, "rate": round(count / total_actions, 3)}
                for name, count in sorted_actions[:n]
            }

        return {
            # Выбор источника
            "total_decisions": self.stats["total_decisions"],
            "q_store_rate": round(self.stats["q_store_chosen"] / total, 3),
            "q_store_weak_rate": round(self.stats["q_store_weak_chosen"] / total, 3),
            "sac_rate": round(self.stats["sac_chosen"] / total, 3),
            "heuristic_rate": round(self.stats["heuristic_chosen"] / total, 3),

            # Согласованность Q и SAC
            "agreement_rate": round(agreement_rate, 3),

            # Средние confidence/spread
            "q_confidence_mean": round(q_conf_mean, 3),
            "q_spread_mean": round(q_spread_mean, 3),
            "sac_confidence_mean": round(sac_conf_mean, 3),

            # Топ предложенных действий
            "q_proposed_top": _top_actions(self.q_proposed_actions),
            "sac_proposed_top": _top_actions(self.sac_proposed_actions),

            # Топ выбранных действий
            "q_chosen_top": _top_actions(self.q_chosen_actions),
            "sac_chosen_top": _top_actions(self.sac_chosen_actions),
        }
