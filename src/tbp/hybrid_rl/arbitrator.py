# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Arbitrator: decides which action source to use per step.
Switches between Q-store (episodic memory), SAC (skill),
and heuristic (fallback).

Logic:
  1. Q-store confident AND distinguishes actions → Q-store (familiar territory)
  2. SAC confident → SAC (generalizes to unfamiliar states)
  3. Q-store has some data → Q-store (weak fallback)
  4. Heuristic (last resort)
"""

import logging
from collections import defaultdict, deque
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from collections import Counter

from .experience_extractor import ExperienceExtractor
from .rl_goal_approach_controller import RLGoalApproachController
from .sac_actor import SACActorNetwork

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
        step = float(action_params[0])
        if step < 0:
            return 9   # free_backward
        elif abs(step) <= 3.0:
            return 19  # free_forward_small (was 20)
        else:
            return 8   # free_forward

    elif action_type == 2:  # Turn
        angle = float(action_params[0])
        if abs(angle) > 10.0:
            return 22 if angle >= 0 else 23  # turn_left/right_big (was 23/24)
        return 10 if angle >= 0 else 11      # turn_left/right

    elif action_type == 3:  # Look
        angle = float(action_params[0])
        if abs(angle) > 10.0:
            return 20 if angle >= 0 else 21  # look_up/down_big (was 21/22)
        return 12 if angle >= 0 else 13      # look_up/down

    elif action_type == 4:
        return 14 if float(action_params[0]) >= 0 else 15

    elif action_type == 5:
        return 16

    elif action_type == 6:
        return 17

    elif action_type == 7:
        return 18  # detach

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
        source_success_window: int = 100,
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
            "sac_chosen": 0,
            "blend_chosen": 0,
            "heuristic_chosen": 0,
            "total_decisions": 0,
        }

        self.q_proposed_actions = defaultdict(int)
        self.sac_proposed_actions = defaultdict(int)
        self.q_chosen_actions = defaultdict(int)
        self.sac_chosen_actions = defaultdict(int)
        self.blend_chosen_actions = defaultdict(int)
        self.heuristic_chosen_actions = defaultdict(int)

        self.agreement_count = 0
        self.both_proposed_count = 0

        self.q_confidence_history = deque(maxlen=1000)
        self.q_spread_history = deque(maxlen=1000)
        self.sac_confidence_history = deque(maxlen=1000)

        self._current_episode_sources = []
        self._q_episode_results = deque(maxlen=source_success_window)
        self._sac_episode_results = deque(maxlen=source_success_window)
        self._heuristic_episode_results = deque(maxlen=source_success_window)

        self._sac_strategic_detach: Optional[Any] = None
        self._sac_strategic_direction: Optional[Any] = None

        self._param_dims = ExperienceExtractor.get_param_dims()
        self._type_names = ExperienceExtractor.get_type_names()

    def decide(
        self,
        state: np.ndarray,
        current_pose: np.ndarray,
        sensor_data: Dict[str, Any],
    ) -> Tuple[int, np.ndarray, str]:
        """Choose action using Q-type + SAC-params synergy.

        Strategy:
        1. Q confident → Q chooses type, SAC provides params
        2. Q not confident → SAC chooses both type and params
        3. No SAC → Q discrete action converted to (type, params)
        4. No Q data → heuristic

        Returns:
            Tuple of (action_type, action_params, source_name).
            action_params are denormalized, ready for ActionInterpreter.
        """
        self.stats["total_decisions"] += 1

        # Get Q-store action + confidence
        q_action_discrete, q_confidence, q_spread = (
            self._get_q_action(state)
        )
        q_action_type, q_params_fn = (
            ExperienceExtractor.DISCRETE_TO_PSAC[q_action_discrete]
        )

        # Track Q proposal
        q_name = self._type_names.get(q_action_type, f"type_{q_action_type}")
        self.q_proposed_actions[q_name] += 1
        self.q_confidence_history.append(q_confidence)
        self.q_spread_history.append(q_spread)

        has_sac = self.sac_actor is not None
        has_q_data = q_confidence > 0.01

        # Get SAC action if available
        sac_type = None
        sac_params = None
        if has_sac:
            sac_type, sac_params = self._get_sac_action_continuous(state)
            sac_name = self._type_names.get(sac_type, f"type_{sac_type}")
            self.sac_proposed_actions[sac_name] += 1
            self.both_proposed_count += 1
            if q_action_type == sac_type:
                self.agreement_count += 1

        # === Decision logic ===

        # Case 1: Q confident — Q chooses type, SAC provides params
        if (
            has_q_data
            and q_confidence > self.q_confidence_threshold
            and q_spread > self.q_spread_threshold
        ):
            if has_sac:
                if q_action_type == sac_type:
                    # Agreement: use SAC params (more precise)
                    source = "blend"
                    self.stats["blend_chosen"] += 1
                    self.blend_chosen_actions[q_name] += 1
                    self._current_episode_sources.append("q_store")
                    return q_action_type, sac_params, source
                else:
                    # Disagreement: Q type, SAC params for that type
                    forced_params = self._get_sac_params_for_type(
                        state, q_action_type
                    )
                    source = "q_type_sac_params"
                    self.stats["q_store_chosen"] += 1
                    self.q_chosen_actions[q_name] += 1
                    self._current_episode_sources.append("q_store")
                    return q_action_type, forced_params, source
            else:
                # No SAC: Q discrete → convert to (type, params)
                q_params = self._discrete_to_params(q_action_discrete)
                source = "q_store"
                self.stats["q_store_chosen"] += 1
                self.q_chosen_actions[q_name] += 1
                self._current_episode_sources.append("q_store")
                return q_action_type, q_params, source

        # Case 2: Q not confident, SAC available → SAC decides
        if has_sac:
            source = "sac"
            sac_name = self._type_names.get(sac_type, f"type_{sac_type}")
            self.stats["sac_chosen"] += 1
            self.sac_chosen_actions[sac_name] += 1
            self._current_episode_sources.append("sac")
            return sac_type, sac_params, source

        # Case 3: Q has some data but low confidence → Q fallback
        if has_q_data:
            q_params = self._discrete_to_params(q_action_discrete)
            source = "q_store_weak"
            self.stats["q_store_chosen"] += 1
            self.q_chosen_actions[q_name] += 1
            self._current_episode_sources.append("q_store")
            return q_action_type, q_params, source

        # Case 4: Nothing available → heuristic
        heuristic_action = self._get_heuristic_action(
            state, current_pose, sensor_data
        )
        h_type, _ = ExperienceExtractor.DISCRETE_TO_PSAC[heuristic_action]
        h_params = self._discrete_to_params(heuristic_action)
        h_name = self._type_names.get(h_type, f"type_{h_type}")
        self.stats["heuristic_chosen"] += 1
        self.heuristic_chosen_actions[h_name] += 1
        self._current_episode_sources.append("heuristic")
        return h_type, h_params, "heuristic"

    def _discrete_to_params(self, discrete_action: int) -> np.ndarray:
        """Convert discrete Q-store action to denormalized params.

        Args:
            discrete_action: Discrete action index (0-23).

        Returns:
            Denormalized action params ready for ActionInterpreter.
        """
        _, params_fn = ExperienceExtractor.DISCRETE_TO_PSAC[discrete_action]
        raw_params = params_fn(self.controller.config)
        padded = np.zeros(3, dtype=np.float32)
        padded[:len(raw_params)] = raw_params
        return padded

    def _get_sac_action_continuous(
        self, state: np.ndarray
    ) -> Tuple[int, np.ndarray]:
        """Get SAC action as (type, denormalized_params).

        Physical action masks applied (same as Q-store):
        - No MoveTangentially in air
        - No MoveLinear on surface
        - No Detach in air
        - No Detach spam (3x in a row)
        """
        state_norm = state
        if self.state_mean is not None:
            state_norm = (state - self.state_mean) / (self.state_std + 1e-8)

        on_object = state[11] > 0.5

        self.sac_actor.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(
                state_norm.astype(np.float32)
            ).unsqueeze(0)

            logits, param_mus, param_log_stds = self.sac_actor(state_t)

            # Physical action masks (same as Q-store)
            if not on_object:  # in air
                logits[0, 0] = -1e9  # no MoveTangentially
                logits[0, 7] = -1e9  # no Detach (already in air)
            if on_object:  # on surface
                logits[0, 1] = -1e9  # no MoveLinear
            # Anti-spam
            if self.controller._consecutive_detach_count >= 3:
                logits[0, 7] = -1e9

            # Sample from masked distribution
            temperature = 0.3
            type_probs = torch.softmax(logits / temperature, dim=-1)
            type_probs = type_probs.clamp(min=1e-8)
            type_probs = type_probs / type_probs.sum(dim=-1, keepdim=True)
            type_dist = torch.distributions.Categorical(type_probs)
            action_type_t = type_dist.sample()

            action_type = action_type_t[0].item()

            # Get params for chosen type
            dim = self._param_dims.get(action_type, 0)
            if dim > 0 and action_type in param_mus and self.param_mean is not None:
                mu = param_mus[action_type][0].numpy()
                params = mu[:dim] * self.param_std[:dim] + self.param_mean[:dim]
                padded = np.zeros(3, dtype=np.float32)
                padded[:dim] = params
            else:
                padded = np.zeros(3, dtype=np.float32)

        return action_type, padded

    def _get_sac_params_for_type(
        self, state: np.ndarray, forced_type: int
    ) -> np.ndarray:
        """Get SAC params for a specific action type.

        When Q chooses the type but SAC chose a different type,
        we can still get SAC's param prediction for Q's type
        because SAC has per-type param heads.

        Args:
            state: Raw state vector.
            forced_type: Action type to get params for.

        Returns:
            Denormalized action params ready for ActionInterpreter.
        """
        state_norm = state
        if self.state_mean is not None:
            state_norm = (state - self.state_mean) / (self.state_std + 1e-8)

        self.sac_actor.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(
                state_norm.astype(np.float32)
            ).unsqueeze(0)

            _, param_mus, _ = self.sac_actor(state_t)

            dim = self._param_dims.get(forced_type, 0)
            if dim > 0 and forced_type in param_mus and self.param_mean is not None:
                raw = param_mus[forced_type][0].numpy()
                params = raw[:dim] * self.param_std[:dim] + self.param_mean[:dim]
                padded = np.zeros(3, dtype=np.float32)
                padded[:dim] = params
                return padded
            else:
                # No params for this type (e.g. Detach)
                # Fall back to discrete params
                return np.zeros(3, dtype=np.float32)

    def _get_q_action(
        self, state: np.ndarray
    ) -> Tuple[int, float, float]:
        """Get action from Q-store with strategic override.

        Same as before — no changes needed.
        """
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

        # Strategic override for detach
        on_object = state[11] > 0.5
        if on_object:
            sensor_proxy = self._get_sensor_proxy()
            if sensor_proxy is not None:
                same_side = sensor_proxy.get("same_side", True)
                path_blocked = sensor_proxy.get("path_blocked", False)
                if (
                    (not same_side or path_blocked)
                    and self.controller._can_detach(state)
                ):
                    t_state = self.controller._compute_detach_transition_state(
                        state, sensor_proxy,
                        movement_efficiency=(
                            self.controller._compute_movement_efficiency(window=20)
                        ),
                    )
                    s_q = self.controller.strategic_detach.get_q_values(t_state)
                    if (
                        self.controller.strategic_detach.next_id > 0
                        and s_q[1] > s_q[0]
                    ):
                        detach_idx = self.controller.action_space.IDX_DETACH
                        q_values[detach_idx] = np.max(q_values) + 1.0

        if self.controller._consecutive_detach_count >= 3:
            q_values[self.controller.action_space.IDX_DETACH] = -1e9

        q_action = int(np.argmax(q_values))
        q_spread = float(np.max(q_values) - np.min(q_values))

        return q_action, q_confidence, q_spread

    def _get_sensor_proxy(self) -> Optional[dict]:
        if self.controller._prev_sensor_data is not None:
            return self.controller._prev_sensor_data
        return None

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

    def on_episode_end(self, success):
        if not self._current_episode_sources:
            self._current_episode_sources = []
            return

        counts = Counter(self._current_episode_sources)
        total = len(self._current_episode_sources)

        min_participation = 0.1
        for source, count in counts.items():
            participation = count / total
            if participation >= min_participation:
                if source == "q_store":
                    self._q_episode_results.append(success)
                elif source == "sac":
                    self._sac_episode_results.append(success)
                elif source == "heuristic":
                    self._heuristic_episode_results.append(success)

        self._current_episode_sources = []

    @property
    def q_success_rate(self) -> float:
        if len(self._q_episode_results) == 0:
            return 0.0
        return sum(self._q_episode_results) / len(self._q_episode_results)

    @property
    def sac_success_rate(self) -> float:
        if len(self._sac_episode_results) == 0:
            return 0.0
        return sum(self._sac_episode_results) / len(self._sac_episode_results)

    @property
    def heuristic_success_rate(self) -> float:
        if len(self._heuristic_episode_results) == 0:
            return 0.0
        return sum(self._heuristic_episode_results) / len(self._heuristic_episode_results)

    def get_stats(self) -> Dict[str, Any]:
        total = max(self.stats["total_decisions"], 1)

        agreement_rate = (
            self.agreement_count / max(self.both_proposed_count, 1)
        )

        q_conf_mean = float(np.mean(self.q_confidence_history)) if self.q_confidence_history else 0.0
        q_spread_mean = float(np.mean(self.q_spread_history)) if self.q_spread_history else 0.0

        def _top_actions(action_dict, n=15):
            sorted_actions = sorted(action_dict.items(), key=lambda x: -x[1])
            total_actions = max(sum(action_dict.values()), 1)
            return {
                name: {"count": count, "rate": round(count / total_actions, 3)}
                for name, count in sorted_actions[:n]
            }

        return {
            "total_decisions": self.stats["total_decisions"],
            "q_store_rate": round(self.stats["q_store_chosen"] / total, 3),
            "sac_rate": round(self.stats["sac_chosen"] / total, 3),
            "blend_rate": round(self.stats.get("blend_chosen", 0) / total, 3),
            "heuristic_rate": round(self.stats["heuristic_chosen"] / total, 3),
            "q_success_rate": round(self.q_success_rate, 3),
            "sac_success_rate": round(self.sac_success_rate, 3),
            "heuristic_success_rate": round(self.heuristic_success_rate, 3),
            "q_episodes": len(self._q_episode_results),
            "sac_episodes": len(self._sac_episode_results),
            "agreement_rate": round(agreement_rate, 3),
            "q_confidence_mean": round(q_conf_mean, 3),
            "q_spread_mean": round(q_spread_mean, 3),
            "q_proposed_top": _top_actions(self.q_proposed_actions),
            "sac_proposed_top": _top_actions(self.sac_proposed_actions),
            "q_chosen_top": _top_actions(self.q_chosen_actions),
            "sac_chosen_top": _top_actions(self.sac_chosen_actions),
            "blend_chosen_top": _top_actions(self.blend_chosen_actions),
        }
