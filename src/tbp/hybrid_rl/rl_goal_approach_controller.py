# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""RL Goal Approach Controller.

Receives goal_pose, uses current proprioceptive state + sensor data to choose
actions, and learns from dense reward (distance reduction to goal).
"""

import json
import logging
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from .action_space import ActionSpace
from .config import DEFAULT_CONFIG
from .hnsw_state_store import HNSWStateStore

logger = logging.getLogger(__name__)


class RLGoalApproachController:
    """Q-learning controller that moves agent toward goal pose.

    State vector (15D):
        local_pos_error  [3D]: direction to goal in agent's local frame
        rot_error        [3D]: orientation error (normalized angles)
        local_normal     [3D]: surface normal in agent's local frame
        k1               [1D]: principal curvature (max absolute)
        k2               [1D]: principal curvature (min absolute)
        on_object        [1D]: whether sensor sees object surface
        alignment        [1D]: dot(goal_direction, point_normal)
        distance         [1D]: Euclidean distance to goal
        norm_depth       [1D]: normalized depth to nearest surface
    """

    def __init__(self, agent_id: str, config: Optional[Dict] = None):
        # Merge config with defaults
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.agent_id = agent_id
        # Mode
        self.mode = self.config.get("mode", "auto")
        self.eval_epsilon = self.config.get("eval_epsilon", 0.02)
        self.eval_alpha_multiplier = self.config.get("eval_alpha_multiplier", 0.1)
        self.auto_train_threshold = self.config.get("auto_train_threshold", 100)

        # Action space
        self.action_space = ActionSpace(
            agent_id=agent_id,
            surface_step=self.config["surface_step"],
            free_step=self.config["free_step"],
            rotation_step=self.config["rotation_step"],
            free_step_small=self.config.get("free_step_small", 2.0),
            rotation_step_big=self.config.get("rotation_step_big", 15.0),
            free_step_backward=self.config.get("free_step_backward", 2.0),
        )
        self.num_actions = self.action_space.NUM_ACTIONS

        # HNSW Q-store
        self.q_store_free = HNSWStateStore(config=self.config, name="free")
        self.q_store_surface = HNSWStateStore(config=self.config, name="surface")

        # Q-learning parameters
        self.gamma = self.config["gamma"]
        self.alpha = self.config["alpha"]
        self.epsilon = self.config["epsilon_start"]
        self.epsilon_min = self.config["epsilon_min"]
        self.epsilon_decay = self.config["epsilon_decay"]
        self.success_backup_enabled = bool(
            self.config.get("success_backup_enabled", True)
        )
        self.success_backup_steps = int(
            self.config.get(
                "success_backup_steps",
                self.config.get("max_steps_per_goal", -1),
            )
        )
        self.success_backup_lambda = float(
            self.config.get("success_backup_lambda", 0.9)
        )
        self.success_backup_alpha_multiplier = float(
            self.config.get("success_backup_alpha_multiplier", 0.5)
        )
        if self.mode == "train_adapt_epsilon":
            total_episodes = max(1, int(self.config.get("num_episodes", 1)))
            eps_start = float(self.config.get("epsilon_start", self.epsilon))
            eps_min = float(self.config.get("epsilon_min", self.epsilon_min))
            if eps_start > eps_min > 0.0:
                self.epsilon_decay = (eps_min / eps_start) ** (
                    1.0 / total_episodes
                )

        # Episode state (reset each new goal)
        self._prev_state: Optional[np.ndarray] = None
        self._prev_sensor_data: Optional[Dict] = None
        self._last_action: Optional[int] = None
        self._prev_action: Optional[int] = None
        self._steps: int = 0
        self._current_goal: Optional[np.ndarray] = None
        self._episode_reward: float = 0.0
        self._episode_transitions: List[Dict[str, Any]] = []
        self.success_trails = []
        self.start_pos: Optional[np.ndarray] = None

        # Distance tracking for flyby detection
        self._distance_history: List[float] = []
        # No-effect action tracking for orientation cooldown
        self._action_no_effect_count: Dict[int, int] = {}

        # Lifetime stats
        self._total_episodes: int = 0
        self._total_steps: int = 0
        self._total_goals_reached: int = 0
        self._termination_counts: Dict[str, int] = {
            "goal_reached": 0,
            "timeout": 0,
            "collision_surface_violation": 0,
            "collision_lost_object": 0,
            "collision_other": 0,
        }
        self.temperature_override = None
        self._collision_stats = {}
        self._last_detach_sub_steps = 1
        self._consecutive_detach_count = 0
        self._global_action_counts: Dict[str, int] = {}
        self._flyby_count = 0
        self._cached_fly_direction = None
        self._fly_direction_age = 0
        self._air_phase = "DIRECT"

        logger.info(
            f"RLGoalApproachController initialized: "
            f"{self.num_actions} actions, "
            f"state_dim={self.config['state_dim']}, "
            f"epsilon={self.epsilon}"
        )

    def _select_store(self, state: np.ndarray) -> HNSWStateStore:
        return self.q_store_surface if state[11] > 0.5 else self.q_store_free

    # ══════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════
    @property
    def is_training(self):
        if self.mode in ("train", "train_adapt_epsilon"):
            return True
        if self.mode == "eval":
            return False
        return self.q_store_free.next_id < self.auto_train_threshold

    def set_new_goal(self, goal_pose: np.ndarray, start_pos: np.ndarray):
        self._current_goal = goal_pose.copy()
        self._prev_state = None
        self._prev_sensor_data = None
        self._last_action = None
        self._prev_action = None
        self._steps = 0
        self._episode_reward = 0.0
        self._episode_transitions = []
        self._distance_history = []
        self._action_no_effect_count = {}
        self._total_episodes += 1
        self.start_pos = start_pos.copy()
        self._last_detach_sub_steps = 1
        self._consecutive_detach_count = 0
        self._flyby_count = 0
        self._fly_direction_age = 0
        self._air_phase = "DIRECT"
        self._cached_fly_direction = None
        if self.is_training:
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay,
            )
        logger.debug(
            f"New goal set (episode {self._total_episodes}): "
            f"pos={goal_pose[:3]}, rot={np.degrees(goal_pose[3:])}°"
        )

    def step(
        self,
        current_pose: np.ndarray,
        sensor_data: Dict[str, Any],
    ) -> Tuple[Optional[str], Optional[Dict]]:
        if self._current_goal is None:
            logger.warning(
                "step() called without goal. Call set_new_goal() first."
            )
            return None, None

        self._last_detach_sub_steps = sensor_data.get("detach_sub_steps", 1)
        self._steps += 1
        self._total_steps += 1

        state = self._compute_state(current_pose, sensor_data)
        logger.debug(
            f"STEP {self._steps}: action={self._last_action}, "
            f"dist={state[13]:.1f}, on={state[11]:.0f}, "
            f"depth={sensor_data.get('depth',0):.1f}, "
            f"pos={current_pose[:3]}"
        )

        # Cache fly direction while on surface — used after detach
        # when point_normal may be None
        if sensor_data.get("on_object", False):
            if not sensor_data.get("same_side", True):
                cached = self._compute_detach_fly_direction(
                    current_pose, sensor_data
                )
                if cached is not None:
                    self._cached_fly_direction = cached
                    logger.debug(
                        f"CACHE_UPDATE: step={self._steps}, "
                        f"pos={[round(x,1) for x in current_pose[:3].tolist()]}, "
                        f"cached_dir={[round(x,3) for x in cached.tolist()]}, "
                        f"same_side=False, "
                        f"on_horizontal={self._is_on_horizontal_surface(sensor_data)}"
                    )

        collision = self._detect_collision(sensor_data)
        if collision:
            logger.debug(f"COLLISION: {collision} at step {self._steps}")

        done = False

        if self._prev_state is not None:
            prev_store = self._select_store(self._prev_state)
        next_store = self._select_store(state)

        if self._prev_state is not None:
            reward, done, termination_reason = self._compute_reward(
                state, self._prev_state, self._last_action, collision
            )
            self._episode_reward += reward
            self._episode_transitions.append(
                {
                    "state": self._prev_state.copy(),
                    "action": int(self._last_action),
                    "reward": float(reward),
                }
            )

            if done:
                td_target = reward
            else:
                next_q = next_store.get_q_values(state)
                td_target = reward + self.gamma * np.max(next_q)

            prev_store.update_q_value(
                self._prev_state,
                self._last_action,
                td_target,
                self._get_learning_rate(),
            )

            if done:
                if termination_reason == "goal_reached":
                    if self.is_training:
                        self._apply_success_backup_updates()
                    self.success_trails = self._episode_transitions.copy()
                self._on_episode_done(state, termination_reason)
                return None, None

        action_index, explanation = self._choose_action(
            state=state,
            current_pose=current_pose,
            sensor_data=sensor_data,
            explain=True,
        )

        self._prev_state = state
        self._prev_sensor_data = sensor_data
        self._prev_action = self._last_action
        self._last_action = action_index
        action_name = self.action_space.get_info(action_index).name
        self._global_action_counts[action_name] = (
            self._global_action_counts.get(action_name, 0) + 1
        )

        if not self.is_training:
            self.epsilon = self.eval_epsilon

        action_info = self.action_space.get_info(action_index)
        action = action_info.name
        dist_to_goal = np.linalg.norm(state[0:3])
        logger.debug(
            f"Step: {self._steps}: "
            f"Action: {action_info}, "
            f"dist={dist_to_goal:.1f}mm, eps={self.epsilon:.3f}"
        )
        return action, explanation

    @property
    def is_active(self) -> bool:
        return self._current_goal is not None

    # ══════════════════════════════════════════════════════════
    # STATE COMPUTATION
    # ══════════════════════════════════════════════════════════
    def _compute_state(self, current_pose, sensor_data):
        goal = self._current_goal

        pos_error_world = goal[:3] - current_pose[:3]
        local_pos_error = self._world_to_local(pos_error_world, current_pose)

        rot_error_deg = self._normalize_angles_deg(
            goal[3:] - current_pose[3:]
        )

        raw_normal = sensor_data.get("point_normal", None)
        if raw_normal is not None:
            local_normal = self._world_to_local(
                np.array(raw_normal), current_pose
            )
        else:
            local_normal = np.zeros(3)

        on_object = float(sensor_data.get("on_object", False))

        distance = np.linalg.norm(local_pos_error)

        normal_len = np.linalg.norm(local_normal)
        if distance > 1e-8 and normal_len > 1e-8:
            goal_dir = local_pos_error / distance
            alignment = np.dot(goal_dir, local_normal)
        else:
            alignment = 0.0

        depth = sensor_data.get("depth", self.config["max_sensor_range"])
        norm_depth = min(depth / self.config["max_sensor_range"], 1.0)

        # Principal curvatures
        k1_raw = float(sensor_data.get("k1", 0.0))
        k2_raw = float(sensor_data.get("k2", 0.0))
        # Convention: |k1| >= |k2|
        if abs(k1_raw) < abs(k2_raw):
            k1, k2 = k2_raw, k1_raw
        else:
            k1, k2 = k1_raw, k2_raw
        # In air, curvature is undefined
        if on_object < 0.5:
            k1, k2 = 0.0, 0.0

        state = np.concatenate(
            [
                local_pos_error,  # [0:3]   3D
                rot_error_deg,    # [3:6]   3D
                local_normal,     # [6:9]   3D
                [k1],             # [9]     1D  principal curvature max
                [k2],             # [10]    1D  principal curvature min
                [on_object],      # [11]    1D
                [alignment],      # [12]    1D
                [distance],       # [13]    1D
                [norm_depth],     # [14]    1D
            ]
        )

        # Track distance for flyby detection (used by heuristic)
        self._distance_history.append(distance)

        # Track no-effect orientation actions for cooldown
        if (self._last_action is not None
                and len(self._distance_history) >= 2):
            dist_change = abs(
                self._distance_history[-1] - self._distance_history[-2]
            )
            action = self._last_action
            orientation_actions = {
                self.action_space.IDX_LOOK_UP,
                self.action_space.IDX_LOOK_DOWN,
                self.action_space.IDX_LOOK_UP_BIG,
                self.action_space.IDX_LOOK_DOWN_BIG,
                self.action_space.IDX_TURN_LEFT,
                self.action_space.IDX_TURN_RIGHT,
                self.action_space.IDX_TURN_LEFT_BIG,
                self.action_space.IDX_TURN_RIGHT_BIG,
                self.action_space.IDX_ORIENT_HOR,
                self.action_space.IDX_ORIENT_VERT,
                self.action_space.IDX_ROTATE_POS,
                self.action_space.IDX_ROTATE_NEG,
            }
            # Only count no-effect on surface; in air turns are needed
            if (
                action in orientation_actions
                and dist_change < 0.1
                and on_object > 0.5
            ):
                self._action_no_effect_count[action] = (
                    self._action_no_effect_count.get(action, 0) + 1
                )
            else:
                self._action_no_effect_count[action] = 0

        return state

    # ══════════════════════════════════════════════════════════
    # COLLISION DETECTION
    # ══════════════════════════════════════════════════════════
    def _detect_collision(self, sensor_data):
        if sensor_data.get("passed_through", False):
            return "surface_violation"

        if sensor_data.get("detach_had_collision", False):
            return "detach_collision"

        depth = sensor_data.get("depth", self.config["max_sensor_range"])

        was_on = (
            self._prev_sensor_data is not None
            and self._prev_sensor_data.get("on_object", False)
        )
        now_on = sensor_data.get("on_object", False)

        prev_depth = (
            self._prev_sensor_data.get(
                "depth", self.config["max_sensor_range"]
            )
            if self._prev_sensor_data is not None
            else self.config["max_sensor_range"]
        )

        logger.debug(
            f"COLLISION_CHECK: was_on={was_on}, now_on={now_on}, "
            f"depth={depth:.3f}, prev_depth={prev_depth:.3f}"
        )

        if (
            was_on
            and prev_depth > 1.5
            and depth < self.config["min_valid_depth"]
        ):
            return "surface_violation"

        if self._prev_sensor_data is not None and was_on and now_on:
            prev_normal = self._prev_sensor_data.get("point_normal")
            curr_normal = sensor_data.get("point_normal")
            if prev_normal is not None and curr_normal is not None:
                dot = np.dot(
                    np.array(prev_normal), np.array(curr_normal)
                )
                if dot < self.config["normal_flip_threshold"]:
                    return "surface_violation"

        if self._prev_sensor_data is not None:
            if was_on and not now_on:
                return "lost_object"

        return None

    # ══════════════════════════════════════════════════════════
    # REWARD
    # ══════════════════════════════════════════════════════════
    def _compute_reward(self, state, prev_state, action, collision):
        cfg = self.config
        reward = 0.0
        done = False
        termination_reason = None

        distance = state[13]
        prev_distance = prev_state[13]
        on_object = state[11]
        prev_alignment = prev_state[12]
        prev_on_object = prev_state[11]

        surface_step = self.action_space.surface_step

        # ═══ 1. Progress toward goal ═══
        progress_raw = prev_distance - distance
        progress = progress_raw
        detour_mode = (
            prev_alignment < cfg["detour_alignment_threshold"]
            and (prev_on_object > 0.5 or collision == "lost_object")
        )
        if detour_mode and progress_raw < 0.0:
            min_progress = (
                -surface_step * cfg["detour_negative_progress_clip_steps"]
            )
            progress = max(progress_raw, min_progress)

        reward += progress / surface_step * cfg["reward_progress"]

        # ═══ 1.5 Subgoal shaping (potential-based, Ng 1999) ═══
        phi_current = self._subgoal_potential(state)
        phi_prev = self._subgoal_potential(prev_state)
        subgoal_shaping = self.gamma * phi_current - phi_prev
        reward += subgoal_shaping

        # ═══ 2. Goal reached ═══
        if distance < cfg["goal_threshold"]:
            reward += cfg["reward_goal_reached"]
            done = True
            termination_reason = "goal_reached"

        # ═══ 3. Step penalty ═══
        reward += cfg["reward_step_penalty"]

        # ═══ 3.5 Risky free actions on surface ═══
        if (
            action == self.action_space.IDX_FREE_FORWARD
            and prev_on_object > 0.5
        ):
            reward += -2.0
        if (
            action == self.action_space.IDX_FREE_BACKWARD
            and prev_on_object > 0.5
        ):
            reward += -2.0
        if (
            action == self.action_space.IDX_FREE_FORWARD_SMALL
            and prev_on_object > 0.5
        ):
            reward += -2.0

        # ═══ 4. Collisions ═══
        if collision == "surface_violation":
            reward += cfg["reward_surface_violation"]
            done = True
            termination_reason = "collision_surface_violation"
            action_name = (
                self.action_space.get_info(action).name
                if action is not None
                else "unknown"
            )
            self._collision_stats[action_name] = (
                self._collision_stats.get(action_name, 0) + 1
            )
            logger.debug(
                f"COLLISION_DETAIL: action={action_name}, "
                f"depth={state[14]*100:.1f}mm, "
                f"on_object={state[11]:.0f}, "
                f"alignment={state[12]:.3f}, "
                f"distance={state[13]:.1f}, "
            )
        elif collision == "detach_collision":
            reward += cfg["reward_surface_violation"]
            done = True
            termination_reason = "collision_surface_violation"
            action_name = (
                self.action_space.get_info(action).name
                if action is not None
                else "unknown"
            )
            self._collision_stats[action_name] = (
                self._collision_stats.get(action_name, 0) + 1
            )
            logger.debug(
                f"DETACH_COLLISION: action={action_name}, "
                f"distance={state[13]:.1f}, progress={progress_raw:.2f}"
            )
        elif collision == "lost_object":
            if action != self.action_space.IDX_DETACH:
                reward += cfg["reward_drifted_away"]

        # ═══ 4.6 Detach while not on surface ═══
        if action == self.action_space.IDX_DETACH:
            if prev_on_object < 0.5:
                reward += -5.0

        # ═══ 5. Near goal on surface ═══
        near_radius = surface_step * 3
        if distance < near_radius and on_object > 0.5:
            reward += cfg["reward_near_goal_on_surface"]

        # ═══ 6. Timeout ═══
        if self._steps >= cfg["max_steps_per_goal"]:
            reward += cfg["reward_timeout"]
            done = True
            if termination_reason is None:
                termination_reason = "timeout"

        return reward, done, termination_reason

    def _subgoal_potential(self, state: np.ndarray) -> float:
        """Potential function for subgoal reward shaping.

        Encourages agent to move toward object edge (alignment -> 0)
        when goal is behind the object (alignment < 0).

        Based on Ng et al. (1999): potential-based shaping preserves
        optimal policy while accelerating learning.
        """
        alignment = float(state[12])
        on_object = float(state[11])

        if alignment >= 0.0:
            return 0.0

        SHAPING_SCALE = 2.0

        if on_object > 0.5:
            # On surface: potential grows as alignment approaches 0 (edge)
            potential = (1.0 + alignment) * SHAPING_SCALE
            return max(potential, 0.0)
        else:
            # In air: smaller potential for flying toward edge
            potential = (1.0 + alignment) * SHAPING_SCALE * 0.3
            return max(potential, 0.0)

    # ══════════════════════════════════════════════════════════
    # ACTION SELECTION
    # ══════════════════════════════════════════════════════════
    def _get_learning_rate(self) -> float:
        if self.is_training:
            return self.alpha
        return self.alpha * self.eval_alpha_multiplier

    def _apply_success_backup_updates(self) -> None:
        if not self.success_backup_enabled:
            return
        if not self._episode_transitions:
            return

        if self.success_backup_steps > 0:
            k = min(
                self.success_backup_steps, len(self._episode_transitions)
            )
        else:
            k = len(self._episode_transitions)
        tail = self._episode_transitions[-k:]
        base_alpha = (
            self._get_learning_rate()
            * self.success_backup_alpha_multiplier
        )
        if base_alpha <= 0.0:
            return

        g_return = 0.0
        for depth, tr in enumerate(reversed(tail)):
            g_return = tr["reward"] + self.gamma * g_return
            lr = base_alpha * (self.success_backup_lambda**depth)
            store = self._select_store(tr["state"])
            store.update_q_value(
                tr["state"],
                tr["action"],
                g_return,
                lr,
                count_visit=False,
            )

        logger.debug(
            "Applied success backup updates: k=%d, base_alpha=%.4f, "
            "lambda=%.3f",
            k,
            base_alpha,
            self.success_backup_lambda,
        )

    def _get_current_epsilon(self):
        if self.is_training:
            return self.epsilon
        return self.eval_epsilon

    def _generate_choice_interpretation(
        self,
        action_index,
        is_random,
        q_recommends,
        h_recommends,
        dominant_heuristic,
        eps,
        confidence,
        blend,
        is_heuristic_override,
    ) -> str:
        action_name = self.action_space.get_info(action_index).name
        q_action_name = self.action_space.get_info(q_recommends).name
        h_action_name = self.action_space.get_info(h_recommends).name

        if is_random:
            return (
                f"##### Random: {action_name} - {action_index} "
                f"epsilon {eps}."
            )
        if is_heuristic_override:
            return (
                f"##### Heuristics: {action_name} - {action_index} "
                f"epsilon {eps}."
            )

        return (
            f"##### Softmax; {action_name}, confidence {confidence:.0%}). "
            f"blend: {blend}. "
            f"Q recommends: {q_action_name} - {q_recommends}; "
            f"Heuristics: {h_action_name} - {h_recommends}. "
        )

    def _choose_action(
        self,
        state: np.ndarray,
        current_pose: np.ndarray,
        sensor_data: Dict[str, Any],
        explain: bool = False,
    ):
        store = self._select_store(state)
        q_values = store.get_q_values(state)

        heuristic, heuristic_components = self._compute_heuristic_bias(
            state=state,
            current_pose=current_pose,
            sensor_data=sensor_data,
            prev_action=self._last_action,
        )

        q_norm = self._normalize_values(q_values)
        h_norm = self._normalize_values(heuristic)

        best_q_action = int(np.argmax(q_values))
        best_h_action = int(np.argmax(heuristic))

        eps = self._get_current_epsilon()

        has_q_data = (
            store.next_id > 0 and np.max(np.abs(q_values)) > 1e-6
        )

        if has_q_data:
            combined = (1 - eps) * q_norm + eps * h_norm
            temperature = np.clip(0.5 * eps, 0.01, 0.5)
        else:
            combined = h_norm.copy()
            temperature = 0.02

        if self.temperature_override is not None:
            temperature = self.temperature_override

        # ═══ ACTION MASK ═══
        if state[11] < 0.5:
            combined[self.action_space.IDX_DETACH] = -1e9
        if self._consecutive_detach_count >= 3:
            combined[self.action_space.IDX_DETACH] = -1e9
        if state[11] > 0.5:  # on surface
            combined[self.action_space.IDX_FREE_FORWARD] = -1e9
            combined[self.action_space.IDX_FREE_FORWARD_SMALL] = -1e9
            combined[self.action_space.IDX_FREE_BACKWARD] = -1e9

        is_random_override = False
        is_heuristic_override = False
        action_index = None

        v = combined / temperature
        v = v - np.max(v)
        exp_v = np.exp(v)
        probs = exp_v / exp_v.sum()

        p_random = 0.02 * eps
        if np.random.random() < p_random:
            is_random_override = True
            valid_mask = np.ones(self.num_actions, dtype=bool)
            if state[11] < 0.5:
                valid_mask[self.action_space.IDX_DETACH] = False
                # Don't randomly do surface actions in air
                for idx in range(8):
                    valid_mask[idx] = False
            if self._consecutive_detach_count >= 3:
                valid_mask[self.action_space.IDX_DETACH] = False
            if state[11] > 0.5:
                valid_mask[self.action_space.IDX_FREE_FORWARD] = False
                valid_mask[self.action_space.IDX_FREE_FORWARD_SMALL] = False
                valid_mask[self.action_space.IDX_FREE_BACKWARD] = False
            # Don't randomly detach when close to goal on surface
            if (
                state[11] > 0.5
                and state[13] < 5.0 * self.action_space.surface_step
            ):
                valid_mask[self.action_space.IDX_DETACH] = False

            valid_indices = np.where(valid_mask)[0]
            action_index = int(np.random.choice(valid_indices))
            probs = np.zeros(self.num_actions)
            probs[action_index] = 1.0
        else:
            action_index = int(np.random.choice(len(probs), p=probs))

        if not explain:
            return action_index, None

        contributions = {
            name: float(np.max(bias))
            for name, bias in heuristic_components.items()
        }
        dominant_heuristic = (
            max(contributions, key=contributions.get)
            if contributions
            else "none"
        )

        confidence = float(probs[action_index])

        explanation = {
            "chosen_action": {
                "index": action_index,
                "name": self.action_space.get_info(action_index).name,
                "probability": confidence,
            },
            "sampling_method": (
                "random_exploration"
                if is_random_override
                else "softmax_sampling"
            ),
            "temperature": temperature,
            "epsilon": eps,
            "has_q_data": has_q_data,
            "blend": (
                f"{(1-eps)*100:.0f}% Q + {eps*100:.0f}% heuristic"
                if has_q_data
                else "100% heuristic"
            ),
            "is_random_override": is_random_override,
            "action_probabilities": {
                self.action_space.get_info(i).name: float(probs[i])
                for i in range(self.num_actions)
            },
            "advice": {
                "q_recommends": {
                    "index": best_q_action,
                    "name": self.action_space.get_info(
                        best_q_action
                    ).name,
                },
                "heuristic_recommends": {
                    "index": best_h_action,
                    "name": self.action_space.get_info(
                        best_h_action
                    ).name,
                },
            },
            "dominant_heuristic": dominant_heuristic,
            "heuristic_contributions": contributions,
            "confidence": confidence,
            "is_confident": confidence > 0.7,
            "interpretation": self._generate_choice_interpretation(
                action_index=action_index,
                is_random=is_random_override,
                q_recommends=best_q_action,
                h_recommends=best_h_action,
                dominant_heuristic=dominant_heuristic,
                eps=eps,
                confidence=confidence,
                blend=(
                    f"{(1-eps)*100:.0f}% Q + {eps*100:.0f}% heuristic"
                    if has_q_data
                    else "100% heuristic"
                ),
                is_heuristic_override=is_heuristic_override,
            ),
        }

        chosen_name = self.action_space.get_info(action_index).name
        if chosen_name == "detach":
            self._consecutive_detach_count += 1
        else:
            self._consecutive_detach_count = 0

        if self._consecutive_detach_count > 2:
            h_best = self.action_space.get_info(best_h_action).name
            q_best = self.action_space.get_info(best_q_action).name
            logger.warning(
                f"DETACH_SPAM x{self._consecutive_detach_count}: "
                f"step={self._steps}, "
                f"chosen={chosen_name}, "
                f"q_recommends={q_best}, h_recommends={h_best}, "
                f"eps={eps:.3f}, temp={temperature:.4f}, "
                f"has_q={has_q_data}, "
                f"random={is_random_override}, "
                f"transitions={len(self._episode_transitions)}, "
                f"last_action={self._last_action}, "
                f"on_object={state[11]:.0f}, "
                f"distance={state[13]:.1f}, "
                f"alignment={state[12]:.3f}, "
                f"depth={state[14]*100:.1f}"
            )
        return action_index, explanation

    # ══════════════════════════════════════════════════════════
    # HEURISTIC BIAS
    # ══════════════════════════════════════════════════════════
    def _compute_heuristic_bias(
        self,
        state: np.ndarray,
        current_pose: np.ndarray,
        sensor_data: dict,
        prev_action: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:

        bias = np.zeros(self.num_actions, dtype=float)
        components: Dict[str, np.ndarray] = {}

        local_pos_error = state[0:3]
        rot_error_deg = state[3:6]
        local_normal = state[6:9]
        on_object = float(state[11])
        alignment = float(state[12])
        distance = float(state[13])
        norm_depth = float(state[14])
        point_normal = sensor_data.get("point_normal")

        eps = 1e-8

        rot = R.from_euler("xyz", current_pose[3:6], degrees=True)

        max_curvature = max(abs(float(state[9])), abs(float(state[10])))
        DETACH_ALIGN_THR = -0.3 + min(max_curvature * 5.0, 0.2)
        SURFACE_STRENGTH = 2.0

        close_to_goal = (
            distance < 3.0 * self.action_space.surface_step
            and alignment > DETACH_ALIGN_THR
        )

        # Reset flyby count when on surface
        if on_object > 0.5:
            self._flyby_count = 0

        if on_object > 0.5 and point_normal is None:
            logger.debug(
                f"BLIND_ON_SURFACE: step={self._steps}, "
                f"on_object={on_object}, point_normal=None, "
                f"depth={sensor_data.get('depth', -1):.2f}, "
                f"alignment={state[12]:.3f}, "
                f"distance={state[13]:.1f}"
            )
        if self._steps == 1:
            logger.debug(
                f"FIRST_STEP_ON_SURFACE: step={self._steps}, "
                f"on_object={on_object}, point_normal={point_normal}, "
                f"depth={sensor_data.get('depth', -1):.2f}, "
                f"alignment={state[12]:.3f}, "
                f"distance={state[13]:.1f}"
            )

        # ── Air phase management ──
        if on_object < 0.5:
            path_blocked_now = sensor_data.get("path_blocked", False)
            same_side_now = sensor_data.get("same_side", True)

            # Initialize at detach
            if self._last_action == self.action_space.IDX_DETACH:
                # Don't overwrite cache — keep direction from on-surface step
                self._fly_direction_age = 0
                self._air_phase = (
                    "FLY"
                    if self._cached_fly_direction is not None
                    else "DIRECT"
                )
                logger.debug(
                    f"DETACH_INIT: step={self._steps}, "
                    f"air_phase={self._air_phase}, "
                    f"cached={self._cached_fly_direction is not None}, "
                    f"cached_dir={[round(x,3) for x in self._cached_fly_direction] if self._cached_fly_direction is not None else None}, "
                    f"same_side={sensor_data.get('same_side', True)}, "
                    f"path_blocked={sensor_data.get('path_blocked', False)}, "
                    f"pos={[round(x,1) for x in current_pose[:3].tolist()]}"
                )
            # Track age of current fly direction
            self._fly_direction_age = getattr(
                self, '_fly_direction_age', 0
            ) + 1

            # State transitions
            if self._air_phase == "FLY":
                if same_side_now and not path_blocked_now:
                    self._air_phase = "DIRECT"
                    self._cached_fly_direction = None
                    logger.debug(
                        f"FLY_TO_DIRECT: step={self._steps}, "
                        f"reason=same_side_clear, "
                        f"same_side={same_side_now}, "
                        f"path_blocked={path_blocked_now}"
                    )
                else:
                    # Safety: abort FLY if agent flew too far from object
                    center_raw = sensor_data.get("object_center")
                    extents_raw = sensor_data.get("object_extents")
                    if center_raw is not None and extents_raw is not None:
                        center = np.asarray(center_raw, dtype=float)
                        dist_from_center = float(np.linalg.norm(
                            current_pose[:3] - center
                        ))
                        max_extent = float(max(extents_raw))
                        if dist_from_center > max_extent * 2.0:
                            logger.debug(
                                f"FLY_SAFETY_ABORT: "
                                f"dist_from_center={dist_from_center:.1f}, "
                                f"max_extent={max_extent:.1f}, "
                                f"threshold={max_extent * 2.0:.1f}"
                            )
                            logger.debug(
                                f"FLY_TO_DIRECT: step={self._steps}, "
                                f"reason=safety_abort, "
                                f"dist_from_center={dist_from_center:.1f}, "
                                f"threshold={max_extent * 2.0:.1f}"
                            )
                            self._air_phase = "DIRECT"
                            self._cached_fly_direction = None
            elif not path_blocked_now:
                self._air_phase = "DIRECT"
                self._cached_fly_direction = None
        else:
            self._air_phase = "DIRECT"

        # ────────────────────────────────────────────────────
        # 0) SUPPRESS
        # ────────────────────────────────────────────────────
        suppress = np.zeros(self.num_actions, dtype=float)
        suppress[self.action_space.IDX_ROTATE_POS] -= 2.0
        suppress[self.action_space.IDX_ROTATE_NEG] -= 2.0

        if not (on_object > 0.5 and close_to_goal):
            suppress[self.action_space.IDX_ORIENT_HOR] -= 2.0
            suppress[self.action_space.IDX_ORIENT_VERT] -= 2.0

        if on_object < 0.5:
            suppress[self.action_space.IDX_DETACH] -= 5.0

        if self._last_action == self.action_space.IDX_DETACH:
            suppress[self.action_space.IDX_DETACH] -= 5.0

        recent_detach = sum(
            1
            for tr in self._episode_transitions[-3:]
            if tr["action"] == self.action_space.IDX_DETACH
        )
        if recent_detach >= 1:
            suppress[self.action_space.IDX_DETACH] -= 5.0

        bias += suppress
        components["suppress"] = suppress

        # ────────────────────────────────────────────────────
        # 1) SURFACE MOVE
        # ────────────────────────────────────────────────────
        surface_move = np.zeros(self.num_actions, dtype=float)

        if on_object > 0.5:
            n_world = sensor_data.get("point_normal")

            if n_world is not None:
                n_world = np.asarray(n_world, dtype=float)
                n_len = float(np.linalg.norm(n_world))
                if n_len > eps:
                    n_hat = n_world / n_len
                    e_world = rot.apply(local_pos_error)
                    step = float(self.action_space.surface_step)

                    # Compute crawl direction: geodesic-aware
                    goal_normal_raw = sensor_data.get("goal_normal")
                    use_geodesic = False

                    if goal_normal_raw is not None:
                        gn = np.asarray(goal_normal_raw, dtype=float)
                        gn_len = float(np.linalg.norm(gn))
                        if gn_len > 1e-8:
                            gn = gn / gn_len
                            gc_axis = np.cross(n_hat, gn)
                            gc_len = float(np.linalg.norm(gc_axis))

                            if gc_len > 0.01:
                                gc_axis /= gc_len
                                geodesic_dir = np.cross(gc_axis, n_hat)
                                geo_len = float(np.linalg.norm(geodesic_dir))
                                if geo_len > 1e-8:
                                    geodesic_dir /= geo_len
                                    e_t_flat = (
                                        e_world
                                        - np.dot(e_world, n_hat) * n_hat
                                    )
                                    # Fix sign for concave surfaces:
                                    # cross product may invert direction
                                    if np.dot(geodesic_dir, e_t_flat) < 0:
                                        geodesic_dir = -geodesic_dir
                                    e_t = geodesic_dir * float(
                                        np.linalg.norm(e_t_flat)
                                    )
                                    use_geodesic = True

                    if not use_geodesic:
                        e_t = e_world - np.dot(e_world, n_hat) * n_hat

                    tangential_dist = float(np.linalg.norm(e_t))
                    normal_dist = abs(float(np.dot(e_world, n_hat)))
                    should_crawl = not (
                        normal_dist > tangential_dist * 2.0
                        and distance > 3 * step
                    )

                    if should_crawl:
                        right_world = rot.apply([1.0, 0.0, 0.0])
                        tb1 = (
                            right_world
                            - np.dot(right_world, n_hat) * n_hat
                        )
                        tb1_norm = np.linalg.norm(tb1)
                        if tb1_norm < 1e-8:
                            up_world = rot.apply([0.0, 1.0, 0.0])
                            tb1 = (
                                up_world
                                - np.dot(up_world, n_hat) * n_hat
                            )
                            tb1_norm = np.linalg.norm(tb1)
                        if tb1_norm < 1e-8:
                            tmp = np.array([0.0, 1.0, 0.0])
                            if abs(np.dot(tmp, n_hat)) > 0.9:
                                tmp = np.array([0.0, 0.0, 1.0])
                            tb1 = np.cross(n_hat, tmp)
                            tb1_norm = np.linalg.norm(tb1)
                        tb1 /= tb1_norm + 1e-12
                        tb2 = np.cross(n_hat, tb1)
                        tb2 /= np.linalg.norm(tb2) + 1e-12

                        best = None
                        best_score = -1e18
                        scores = np.full(8, -1e18, dtype=float)
                        for i, deg in enumerate(
                            self.action_space.SURFACE_DIRECTIONS
                        ):
                            a = np.radians(deg)
                            v_world = np.cos(a) * tb1 + np.sin(a) * tb2
                            v_norm = float(np.linalg.norm(v_world))
                            if v_norm < 1e-8:
                                continue
                            v_world /= v_norm
                            new_e = e_t - step * v_world
                            score = float(
                                np.dot(e_t, e_t)
                                - np.dot(new_e, new_e)
                            )
                            scores[i] = score
                            if score > best_score:
                                best_score = score
                                best = i

                        if distance < 3.0 * step:
                            HYST_ABS = 1.0
                        else:
                            HYST_ABS = 0.25
                        if (
                            prev_action is not None
                            and 0 <= prev_action < 8
                            and best is not None
                        ):
                            prev_score = scores[int(prev_action)]
                            if (best_score - prev_score) < HYST_ABS:
                                best = int(prev_action)

                        if best is not None:
                            surface_move[best] = SURFACE_STRENGTH

        bias += surface_move
        components["surface_move"] = surface_move

        # ────────────────────────────────────────────────────
        # 2) STAGNATION (global — triple window)
        # ────────────────────────────────────────────────────
        stagnation = np.zeros(self.num_actions, dtype=float)

        if on_object > 0.5:
            near_goal = distance < 5.0 * self.action_space.surface_step

            # Short window (5 steps) — fast stuck detection
            if len(self._episode_transitions) >= 5 and not near_goal:
                recent_dists = [
                    float(tr["state"][13])
                    for tr in self._episode_transitions[-5:]
                ]
                dist_reduction_5 = recent_dists[0] - recent_dists[-1]
                if dist_reduction_5 < self.action_space.surface_step * 0.5:
                    stagnation[self.action_space.IDX_DETACH] += 3.0
                    for idx in range(8):
                        stagnation[idx] -= 1.0

            # Medium window (10 steps) — catches oscillations faster
            if len(self._distance_history) >= 10 and not near_goal:
                dist_10_ago = self._distance_history[-10]
                dist_progress_10 = dist_10_ago - distance
                min_expected_10 = 3.0 * self.action_space.surface_step

                if dist_progress_10 < min_expected_10:
                    recent_detach_med = sum(
                        1
                        for tr in self._episode_transitions[-8:]
                        if tr["action"] == self.action_space.IDX_DETACH
                    )
                    last_was_detach = (
                        self._last_action == self.action_space.IDX_DETACH
                    )

                    if recent_detach_med < 1 and not last_was_detach:
                        stag_strength_med = min(2.0 + distance / 60.0, 6.0)
                        stagnation[
                            self.action_space.IDX_DETACH
                        ] += stag_strength_med
                        for idx in range(8):
                            stagnation[idx] -= stag_strength_med * 0.3

            # Long window (20 steps) — robust confirmation
            if len(self._distance_history) >= 20 and not near_goal:
                dist_20_ago = self._distance_history[-20]
                dist_progress_20 = dist_20_ago - distance
                min_expected_20 = 5.0 * self.action_space.surface_step

                if dist_progress_20 < min_expected_20:
                    recent_detach_long = sum(
                        1
                        for tr in self._episode_transitions[-10:]
                        if tr["action"] == self.action_space.IDX_DETACH
                    )
                    last_was_detach = (
                        self._last_action == self.action_space.IDX_DETACH
                    )

                    if recent_detach_long < 1 and not last_was_detach:
                        stag_strength = min(3.0 + distance / 50.0, 8.0)
                        stagnation[
                            self.action_space.IDX_DETACH
                        ] += stag_strength
                        for idx in range(8):
                            stagnation[idx] -= stag_strength * 0.3

        bias += stagnation
        components["stagnation"] = stagnation
        # ────────────────────────────────────────────────────
        # 3) DETACH
        # ────────────────────────────────────────────────────
        detach = np.zeros(self.num_actions, dtype=float)
        if on_object > 0.5:
            need_detach = False
            n_world = sensor_data.get("point_normal")
            path_blocked = sensor_data.get("path_blocked", False)

            if n_world is not None:
                n_world = np.asarray(n_world, dtype=float)
                n_len = float(np.linalg.norm(n_world))
                if n_len > eps:
                    n_hat = n_world / n_len
                    e_world = rot.apply(local_pos_error)
                    e_t = e_world - np.dot(e_world, n_hat) * n_hat
                    tangential_dist = float(np.linalg.norm(e_t))
                    normal_dist = abs(float(np.dot(e_world, n_hat)))

                    on_horizontal = self._is_on_horizontal_surface(
                        sensor_data
                    )
                    if on_horizontal:
                        # Bottom/top face: only detach if agent and goal
                        # on different sides (same_side=False).
                        # Uses nearest.on_surface (view-independent),
                        # more reliable than path_blocked on horizontal
                        # surfaces where ray-based normal is unreliable.
                        same_side_val = sensor_data.get("same_side", True)
                        if not same_side_val:
                            need_detach = True
                    elif path_blocked:
                        need_detach = True
                    elif alignment < DETACH_ALIGN_THR:
                        if alignment < -0.5:
                            need_detach = True
                        elif (
                            distance
                            > 15.0 * self.action_space.surface_step
                        ):
                            need_detach = True
                        else:
                            need_detach = False
                    elif (
                        normal_dist > tangential_dist * 2.0
                        and distance
                        > 15.0 * self.action_space.surface_step
                    ):
                        need_detach = True

                    logger.debug(
                        f"DETACH_SECTION3_DEBUG: step={self._steps}, "
                        f"path_blocked={path_blocked}, "
                        f"alignment={alignment:.3f}, "
                        f"DETACH_ALIGN_THR={DETACH_ALIGN_THR:.3f}, "
                        f"normal_dist={normal_dist:.1f}, "
                        f"tangential_dist={tangential_dist:.1f}, "
                        f"distance={distance:.1f}, "
                        f"threshold_15={15.0 * self.action_space.surface_step:.1f}, "
                        f"need_detach={need_detach}"
                    )
            if need_detach:
                recent_detach_count = sum(
                    1
                    for tr in self._episode_transitions[-5:]
                    if tr["action"] == self.action_space.IDX_DETACH
                )
                last_was_detach = (
                    self._last_action == self.action_space.IDX_DETACH
                )

                if (
                    recent_detach_count < 1
                    and not last_was_detach
                    and not close_to_goal
                ):
                    detach[self.action_space.IDX_DETACH] += 8.0
                    for idx in range(8):
                        detach[idx] -= 2.0
                    detach[self.action_space.IDX_FREE_FORWARD] -= 2.0
                    detach[self.action_space.IDX_FREE_BACKWARD] -= 2.0
                    detach[self.action_space.IDX_LOOK_UP] -= 1.0
                    detach[self.action_space.IDX_LOOK_DOWN] -= 1.0

        bias += detach
        components["detach"] = detach

        # ────────────────────────────────────────────────────
        # 4) STEER IN AIR
        # ────────────────────────────────────────────────────
        steer = np.zeros(self.num_actions, dtype=float)
        if on_object <= 0.5:
            STEER_STRENGTH = 8.0
            rotation_step = self.action_space.rotation_step

            # ── Determine effective goal direction ──
            goal_dir_world = self._current_goal[:3] - current_pose[:3]
            goal_dist_world = np.linalg.norm(goal_dir_world)
            if goal_dist_world > 1e-8:
                goal_dir_world /= goal_dist_world

            rot_current = R.from_euler(
                "xyz", current_pose[3:6], degrees=True
            )
            forward_current = rot_current.apply([0, 0, -1])

            air_phase = getattr(self, '_air_phase', 'DIRECT')

            if air_phase == "FLY" and self._cached_fly_direction is not None:
                effective_goal = self._cached_fly_direction
            elif air_phase == "REORIENT":
                effective_goal = goal_dir_world
            else:
                effective_goal = goal_dir_world

            dot_current = float(np.dot(forward_current, effective_goal))
            angle_to_goal = np.degrees(
                np.arccos(np.clip(dot_current, -1, 1))
            )

            logger.debug(
                f"STEER_DEBUG: step={self._steps}, "
                f"air_phase={air_phase}, "
                f"effective_goal={[round(x,3) for x in effective_goal.tolist()]}, "
                f"forward={[round(x,3) for x in forward_current.tolist()]}, "
                f"angle_to_goal={angle_to_goal:.1f}"
            )

            pose_angles = current_pose[3:6]

            # ── Compute improvement for each rotation direction ──
            forward_up = R.from_euler(
                "xyz",
                pose_angles + np.array([rotation_step, 0, 0]),
                degrees=True,
            ).apply([0, 0, -1])
            improvement_up = float(
                np.dot(forward_up, effective_goal) - dot_current
            )

            forward_down = R.from_euler(
                "xyz",
                pose_angles + np.array([-rotation_step, 0, 0]),
                degrees=True,
            ).apply([0, 0, -1])
            improvement_down = float(
                np.dot(forward_down, effective_goal) - dot_current
            )

            forward_left = R.from_euler(
                "xyz",
                pose_angles + np.array([0, rotation_step, 0]),
                degrees=True,
            ).apply([0, 0, -1])
            improvement_left = float(
                np.dot(forward_left, effective_goal) - dot_current
            )

            forward_right = R.from_euler(
                "xyz",
                pose_angles + np.array([0, -rotation_step, 0]),
                degrees=True,
            ).apply([0, 0, -1])
            improvement_right = float(
                np.dot(forward_right, effective_goal) - dot_current
            )

            # Zero out improvement if rotation didn't change forward
            if np.linalg.norm(forward_up - forward_current) < 1e-4:
                improvement_up = 0.0
            if np.linalg.norm(forward_down - forward_current) < 1e-4:
                improvement_down = 0.0
            if np.linalg.norm(forward_left - forward_current) < 1e-4:
                improvement_left = 0.0
            if np.linalg.norm(forward_right - forward_current) < 1e-4:
                improvement_right = 0.0

            # ── Thresholds based on action space ──
            TURN_ONLY = 3.0 * self.action_space.rotation_step_big
            FLY_THR = 4.0 * self.action_space.rotation_step

            # ── REORIENT phase: only turn, no forward ──
            if air_phase == "REORIENT":
                steer[self.action_space.IDX_FREE_FORWARD] -= STEER_STRENGTH
                steer[
                    self.action_space.IDX_FREE_FORWARD_SMALL
                ] -= STEER_STRENGTH

                best_pitch = max(improvement_up, improvement_down)
                best_yaw = max(improvement_left, improvement_right)

                if best_pitch >= best_yaw and best_pitch > 0.001:
                    if improvement_up > improvement_down:
                        steer[self.action_space.IDX_LOOK_UP] += STEER_STRENGTH
                    else:
                        steer[
                            self.action_space.IDX_LOOK_DOWN
                        ] += STEER_STRENGTH
                if best_yaw >= best_pitch and best_yaw > 0.001:
                    if improvement_left > improvement_right:
                        steer[
                            self.action_space.IDX_TURN_LEFT
                        ] += STEER_STRENGTH
                    else:
                        steer[
                            self.action_space.IDX_TURN_RIGHT
                        ] += STEER_STRENGTH

            # ── Phase 1: FAR FROM ALIGNED (angle > 45°) ──
            elif angle_to_goal > TURN_ONLY:
                steer[self.action_space.IDX_FREE_FORWARD] -= STEER_STRENGTH
                steer[
                    self.action_space.IDX_FREE_FORWARD_SMALL
                ] -= STEER_STRENGTH

                best_pitch = max(improvement_up, improvement_down)
                best_yaw = max(improvement_left, improvement_right)

                if best_pitch >= best_yaw and best_pitch > 0.001:
                    if improvement_up > improvement_down:
                        steer[self.action_space.IDX_LOOK_UP] += STEER_STRENGTH
                        steer[
                            self.action_space.IDX_LOOK_UP_BIG
                        ] += STEER_STRENGTH
                    else:
                        steer[
                            self.action_space.IDX_LOOK_DOWN
                        ] += STEER_STRENGTH
                        steer[
                            self.action_space.IDX_LOOK_DOWN_BIG
                        ] += STEER_STRENGTH

                if best_yaw >= best_pitch and best_yaw > 0.001:
                    if improvement_left > improvement_right:
                        steer[
                            self.action_space.IDX_TURN_LEFT
                        ] += STEER_STRENGTH
                        steer[
                            self.action_space.IDX_TURN_LEFT_BIG
                        ] += STEER_STRENGTH
                    else:
                        steer[
                            self.action_space.IDX_TURN_RIGHT
                        ] += STEER_STRENGTH
                        steer[
                            self.action_space.IDX_TURN_RIGHT_BIG
                        ] += STEER_STRENGTH

            # ── Phase 2: PARTIALLY ALIGNED (20° < angle <= 45°) ──
            elif angle_to_goal > FLY_THR:
                steer[
                    self.action_space.IDX_FREE_FORWARD_SMALL
                ] += STEER_STRENGTH * 0.5
                steer[
                    self.action_space.IDX_FREE_FORWARD
                ] -= STEER_STRENGTH * 0.5

                turn_strength = STEER_STRENGTH * 0.7

                best_pitch = max(improvement_up, improvement_down)
                best_yaw = max(improvement_left, improvement_right)

                if best_pitch > 0.001:
                    if improvement_up > improvement_down:
                        steer[
                            self.action_space.IDX_LOOK_UP
                        ] += turn_strength
                    else:
                        steer[
                            self.action_space.IDX_LOOK_DOWN
                        ] += turn_strength

                if best_yaw > 0.001:
                    if improvement_left > improvement_right:
                        steer[
                            self.action_space.IDX_TURN_LEFT
                        ] += turn_strength
                    else:
                        steer[
                            self.action_space.IDX_TURN_RIGHT
                        ] += turn_strength

            # ── Phase 3: WELL ALIGNED (angle <= 20°) ──
            else:
                if distance > 8.0 * self.action_space.free_step:
                    steer[
                        self.action_space.IDX_FREE_FORWARD
                    ] += STEER_STRENGTH
                    steer[
                        self.action_space.IDX_FREE_FORWARD_SMALL
                    ] += STEER_STRENGTH * 0.5
                else:
                    steer[
                        self.action_space.IDX_FREE_FORWARD_SMALL
                    ] += STEER_STRENGTH
                    steer[
                        self.action_space.IDX_FREE_FORWARD
                    ] += STEER_STRENGTH * 0.3

                correction = STEER_STRENGTH * 0.3

                best_pitch = max(improvement_up, improvement_down)
                best_yaw = max(improvement_left, improvement_right)

                if best_pitch > 0.01:
                    if improvement_up > improvement_down:
                        steer[
                            self.action_space.IDX_LOOK_UP
                        ] += correction
                    else:
                        steer[
                            self.action_space.IDX_LOOK_DOWN
                        ] += correction

                if best_yaw > 0.01:
                    if improvement_left > improvement_right:
                        steer[
                            self.action_space.IDX_TURN_LEFT
                        ] += correction
                    else:
                        steer[
                            self.action_space.IDX_TURN_RIGHT
                        ] += correction

            # ── Always suppress backward and surface/utility actions ──
            steer[self.action_space.IDX_FREE_BACKWARD] -= 8.0

            for idx in range(8):
                steer[idx] -= 8.0
            steer[self.action_space.IDX_ORIENT_HOR] -= 8.0
            steer[self.action_space.IDX_ORIENT_VERT] -= 8.0
            steer[self.action_space.IDX_ROTATE_POS] -= 8.0
            steer[self.action_space.IDX_ROTATE_NEG] -= 8.0

        bias += steer
        components["steer_in_air"] = steer
        # ────────────────────────────────────────────────────
        # 5) DAMP FREE
        # ────────────────────────────────────────────────────
        damp_free = np.zeros(self.num_actions, dtype=float)
        if on_object > 0.5:
            # Free movement dangerous on surface (also masked)
            damp_free[self.action_space.IDX_FREE_FORWARD] -= 8.0
            damp_free[self.action_space.IDX_FREE_FORWARD_SMALL] -= 8.0
            damp_free[self.action_space.IDX_FREE_BACKWARD] -= 8.0
            # Big orientations less useful on surface
            damp_free[self.action_space.IDX_LOOK_UP_BIG] -= 4.0
            damp_free[self.action_space.IDX_LOOK_DOWN_BIG] -= 4.0
            damp_free[self.action_space.IDX_TURN_LEFT_BIG] -= 4.0
            damp_free[self.action_space.IDX_TURN_RIGHT_BIG] -= 4.0
            # Utility actions less useful on surface
            damp_free[self.action_space.IDX_ORIENT_HOR] -= 4.0
            damp_free[self.action_space.IDX_ORIENT_VERT] -= 4.0
            damp_free[self.action_space.IDX_ROTATE_POS] -= 4.0
            damp_free[self.action_space.IDX_ROTATE_NEG] -= 4.0
        bias += damp_free
        components["damp_free_on_surface"] = damp_free

        # ────────────────────────────────────────────────────
        # 6) SUBGOAL GUIDANCE
        # ────────────────────────────────────────────────────
        subgoal_bias = np.zeros(self.num_actions, dtype=float)

        need_subgoal = (
            alignment < -0.1
            and distance > 3.0 * self.action_space.surface_step
        )

        if need_subgoal:
            subgoal_strength = min(abs(alignment) * 4.0 + 1.5, 3.0)

            subgoal_dir = self._compute_subgoal_direction(
                current_pose, sensor_data
            )

            if on_object > 0.5:
                on_horizontal = self._is_on_horizontal_surface(
                    sensor_data
                )
                should_boost_detach = (
                    not on_horizontal
                    and (
                        alignment < -0.5
                        or distance
                        > 15.0 * self.action_space.surface_step
                    )
                )

                if should_boost_detach:
                    recent_detach_count = sum(
                        1
                        for tr in self._episode_transitions[-5:]
                        if tr["action"] == self.action_space.IDX_DETACH
                    )
                    last_was_detach = (
                        self._last_action
                        == self.action_space.IDX_DETACH
                    )

                    if (
                        recent_detach_count < 1
                        and not last_was_detach
                    ):
                        subgoal_bias[
                            self.action_space.IDX_DETACH
                        ] += subgoal_strength
                    for idx in range(8):
                        subgoal_bias[idx] -= subgoal_strength * 0.3

            elif on_object < 0.5 and subgoal_dir is not None:
                # In air with goal behind object: steer toward edge
                forward_current = rot.apply([0, 0, -1])
                dot_to_edge = float(
                    np.dot(forward_current, subgoal_dir)
                )

                if dot_to_edge < 0.5:
                    right_vec = rot.apply([1, 0, 0])
                    up_vec = rot.apply([0, 1, 0])

                    yaw_component = float(
                        np.dot(subgoal_dir, right_vec)
                    )
                    pitch_component = float(
                        np.dot(subgoal_dir, up_vec)
                    )

                    edge_steer = subgoal_strength * 0.5

                    if abs(yaw_component) > abs(pitch_component):
                        if yaw_component > 0:
                            subgoal_bias[
                                self.action_space.IDX_TURN_RIGHT
                            ] += edge_steer
                        else:
                            subgoal_bias[
                                self.action_space.IDX_TURN_LEFT
                            ] += edge_steer
                    else:
                        if pitch_component > 0:
                            subgoal_bias[
                                self.action_space.IDX_LOOK_UP
                            ] += edge_steer
                        else:
                            subgoal_bias[
                                self.action_space.IDX_LOOK_DOWN
                            ] += edge_steer

        bias += subgoal_bias
        components["subgoal_guidance"] = subgoal_bias

        # ────────────────────────────────────────────────────
        # 7) FLYBY CORRECTION (improved)
        # ────────────────────────────────────────────────────
        flyby_bias = np.zeros(self.num_actions, dtype=float)

        if len(self._distance_history) >= 3:
            dist_trend_3 = (
                self._distance_history[-1] - self._distance_history[-3]
            )
            dist_trend_2 = (
                (self._distance_history[-1] - self._distance_history[-2])
                if len(self._distance_history) >= 2
                else 0.0
            )

            if on_object < 0.5:
                # IN AIR: flyby correction
                forward_current = rot.apply([0, 0, -1])
                goal_dir_world_fb = (
                    self._current_goal[:3] - current_pose[:3]
                )
                goal_dist_world_fb = float(
                    np.linalg.norm(goal_dir_world_fb)
                )
                if goal_dist_world_fb > 1e-8:
                    goal_dir_norm = goal_dir_world_fb / goal_dist_world_fb
                    approach_cos = float(
                        np.dot(forward_current, goal_dir_norm)
                    )
                else:
                    approach_cos = 1.0

                flyby_count = getattr(self, '_flyby_count', 0)
                flyby_triggered = False
                FLYBY_STRENGTH = 4.0

                # Lowered thresholds
                if dist_trend_2 > 2.0:
                    flyby_triggered = True
                    FLYBY_STRENGTH = 5.0 + flyby_count * 2.0
                elif dist_trend_3 > 3.0 and approach_cos < 0.5:
                    flyby_triggered = True
                    FLYBY_STRENGTH = 4.0 + flyby_count * 2.0

                # Detect via min distance: were closer, now moving away
                if len(self._distance_history) >= 5:
                    min_recent_5 = min(self._distance_history[-5:])
                    if distance > min_recent_5 + 2.0 * self.action_space.free_step:
                        if not flyby_triggered:
                            flyby_triggered = True
                            FLYBY_STRENGTH = max(
                                5.0 + flyby_count * 2.0,
                                FLYBY_STRENGTH,
                            )

                if flyby_triggered:
                    self._flyby_count = flyby_count + 1

                    flyby_bias[
                        self.action_space.IDX_FREE_FORWARD
                    ] -= FLYBY_STRENGTH
                    flyby_bias[
                        self.action_space.IDX_FREE_FORWARD_SMALL
                    ] -= FLYBY_STRENGTH * 0.8

                    # After 2+ flybys: even more aggressive
                    if self._flyby_count >= 2:
                        flyby_bias[
                            self.action_space.IDX_FREE_FORWARD
                        ] -= FLYBY_STRENGTH

                    pose_angles_fb = current_pose[3:6]
                    rotation_step_fb = self.action_space.rotation_step

                    fwd_up = R.from_euler(
                        "xyz",
                        pose_angles_fb
                        + np.array([rotation_step_fb, 0, 0]),
                        degrees=True,
                    ).apply([0, 0, -1])
                    fwd_down = R.from_euler(
                        "xyz",
                        pose_angles_fb
                        + np.array([-rotation_step_fb, 0, 0]),
                        degrees=True,
                    ).apply([0, 0, -1])
                    fwd_left = R.from_euler(
                        "xyz",
                        pose_angles_fb
                        + np.array([0, rotation_step_fb, 0]),
                        degrees=True,
                    ).apply([0, 0, -1])
                    fwd_right = R.from_euler(
                        "xyz",
                        pose_angles_fb
                        + np.array([0, -rotation_step_fb, 0]),
                        degrees=True,
                    ).apply([0, 0, -1])

                    imp_up = float(
                        np.dot(fwd_up, goal_dir_norm) - approach_cos
                    )
                    imp_down = float(
                        np.dot(fwd_down, goal_dir_norm) - approach_cos
                    )
                    imp_left = float(
                        np.dot(fwd_left, goal_dir_norm) - approach_cos
                    )
                    imp_right = float(
                        np.dot(fwd_right, goal_dir_norm) - approach_cos
                    )

                    best_pitch = max(imp_up, imp_down)
                    best_yaw = max(imp_left, imp_right)

                    if best_pitch > 0.001:
                        if imp_up > imp_down:
                            flyby_bias[
                                self.action_space.IDX_LOOK_UP
                            ] += FLYBY_STRENGTH
                        else:
                            flyby_bias[
                                self.action_space.IDX_LOOK_DOWN
                            ] += FLYBY_STRENGTH

                    if best_yaw > 0.001:
                        if imp_left > imp_right:
                            flyby_bias[
                                self.action_space.IDX_TURN_LEFT
                            ] += FLYBY_STRENGTH
                        else:
                            flyby_bias[
                                self.action_space.IDX_TURN_RIGHT
                            ] += FLYBY_STRENGTH

            elif on_object > 0.5:
                # ON SURFACE: if distance growing steadily, boost detach
                if (
                    len(self._distance_history) >= 6
                    and dist_trend_3 > 3.0
                ):
                    trend_5 = (
                        self._distance_history[-1]
                        - self._distance_history[-6]
                    )
                    if trend_5 > 4.0:
                        recent_detach = sum(
                            1
                            for tr in self._episode_transitions[-5:]
                            if tr["action"]
                            == self.action_space.IDX_DETACH
                        )
                        if recent_detach < 1:
                            flyby_bias[
                                self.action_space.IDX_DETACH
                            ] += 3.0
                            for idx in range(8):
                                flyby_bias[idx] -= 1.0

        bias += flyby_bias
        components["flyby_correction"] = flyby_bias

        # ────────────────────────────────────────────────────
        # 8) ORIENTATION COOLDOWN (suppress no-effect orientations)
        # ────────────────────────────────────────────────────
        cooldown_bias = np.zeros(self.num_actions, dtype=float)

        if on_object > 0.5:
            for action_idx, count in self._action_no_effect_count.items():
                if count >= 3:
                    penalty = min(count * 3.0, 12.0)
                    cooldown_bias[action_idx] -= penalty

        bias += cooldown_bias
        components["orientation_cooldown"] = cooldown_bias

        # ────────────────────────────────────────────────────
        # 9) LANDING — approach surface near goal, stop flyby
        # ────────────────────────────────────────────────────
        landing_bias = np.zeros(self.num_actions, dtype=float)

        if on_object < 0.5 and getattr(self, '_air_phase', 'DIRECT') != "FLY":
        # if on_object < 0.5:
            LANDING_THRESHOLD = 8.0 * self.action_space.free_step
            path_blocked_now = sensor_data.get("path_blocked", False)

            # Compute angle to goal
            goal_dir_land = self._current_goal[:3] - current_pose[:3]
            goal_dist_land = np.linalg.norm(goal_dir_land)
            if goal_dist_land > 1e-8:
                goal_dir_land /= goal_dist_land
            forward_land = rot.apply([0, 0, -1])
            dot_land = np.dot(forward_land, goal_dir_land)
            angle_to_goal_land = np.degrees(
                np.arccos(np.clip(dot_land, -1, 1))
            )

            can_land = (
                not path_blocked_now
                and angle_to_goal_land < 60.0
            )

            CLOSE_LANDING = 3.0 * self.action_space.free_step

            if distance < CLOSE_LANDING:
                # VERY CLOSE to goal — priority is to land on surface
                
                if angle_to_goal_land < 30.0:
                    # Aimed at goal — dive to surface
                    landing_bias[
                        self.action_space.IDX_FREE_FORWARD_SMALL
                    ] += 10.0
                    landing_bias[
                        self.action_space.IDX_FREE_FORWARD
                    ] -= 5.0  # only small steps
                    # Suppress turns — don't lose aim
                    for idx in [
                        self.action_space.IDX_LOOK_UP,
                        self.action_space.IDX_LOOK_DOWN,
                        self.action_space.IDX_TURN_LEFT,
                        self.action_space.IDX_TURN_RIGHT,
                        self.action_space.IDX_LOOK_UP_BIG,
                        self.action_space.IDX_LOOK_DOWN_BIG,
                        self.action_space.IDX_TURN_LEFT_BIG,
                        self.action_space.IDX_TURN_RIGHT_BIG,
                    ]:
                        landing_bias[idx] -= 5.0
                else:
                    # Not aimed at goal — STOP and turn toward it
                    landing_bias[
                        self.action_space.IDX_FREE_FORWARD
                    ] -= 15.0
                    landing_bias[
                        self.action_space.IDX_FREE_FORWARD_SMALL
                    ] -= 10.0
                    # steer_in_air turn biases will handle rotation

            elif can_land and distance < LANDING_THRESHOLD:
                landing_urgency = 1.0 - (distance / LANDING_THRESHOLD)

                # Approaching: slow down, prefer small forward
                landing_bias[
                    self.action_space.IDX_FREE_FORWARD
                ] -= 4.0 * landing_urgency
                landing_bias[
                    self.action_space.IDX_FREE_FORWARD_SMALL
                ] += 3.0 * landing_urgency

                # If we see surface nearby — go for it
                if norm_depth < 0.3:
                    landing_bias[
                        self.action_space.IDX_FREE_FORWARD_SMALL
                    ] += 4.0 * landing_urgency

                # Suppress big turns
                if distance < 4.0 * self.action_space.free_step:
                    for idx in [
                        self.action_space.IDX_LOOK_UP_BIG,
                        self.action_space.IDX_LOOK_DOWN_BIG,
                        self.action_space.IDX_TURN_LEFT_BIG,
                        self.action_space.IDX_TURN_RIGHT_BIG,
                    ]:
                        landing_bias[idx] -= 3.0

            # Flyby detection: path clear + was close + now moving away
            if not path_blocked_now and len(self._distance_history) >= 10:
                min_recent_10 = min(self._distance_history[-10:])
                overshoot = distance - min_recent_10

                if (
                    overshoot > 1.0 * self.action_space.free_step
                    and min_recent_10 < LANDING_THRESHOLD
                ):
                    # Flew past goal — STOP, let steer turn back
                    landing_bias[
                        self.action_space.IDX_FREE_FORWARD
                    ] -= 20.0
                    landing_bias[
                        self.action_space.IDX_FREE_FORWARD_SMALL
                    ] -= 15.0

        bias += landing_bias
        components["landing"] = landing_bias
        return bias, components

    # ══════════════════════════════════════════════════════════
    # SUBGOAL HELPERS
    # ══════════════════════════════════════════════════════════
    def _compute_detach_fly_direction(
        self,
        current_pose: np.ndarray,
        sensor_data: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        """Compute fly direction for obstacle avoidance after detach.

        When same_side=False (agent and goal on opposite sides of wall):
        fly toward open edge (rim) using up_direction projected onto
        tangent plane. For horizontal surfaces, fly away from center.

        When same_side=True: original tangent projection toward goal.
        Works for all solid objects and same-side hollow cases.

        Returns:
            Unit vector in world space, or None if no normal available.
        """
        goal_pos = self._current_goal[:3]
        goal_dir = goal_pos - current_pose[:3]
        goal_dist = np.linalg.norm(goal_dir)
        if goal_dist < 1e-8:
            return None
        goal_dir /= goal_dist

        normal = sensor_data.get("point_normal")
        if normal is None:
            logger.debug(
                "DETACH_FLY_DIR: point_normal is None, "
                f"agent_pos={current_pose[:3].tolist()}"
            )
            return None
        n = np.asarray(normal, dtype=float)
        n_len = np.linalg.norm(n)
        if n_len < 1e-8:
            return None
        n /= n_len

        same_side = sensor_data.get("same_side", True)

        logger.debug(
            f"DETACH_FLY_DIR: "
            f"same_side={same_side}, "
            f"normal={[round(x, 3) for x in n.tolist()]}, "
            f"goal_dir={[round(x, 3) for x in goal_dir.tolist()]}, "
            f"agent_pos={[round(x, 1) for x in current_pose[:3].tolist()]}, "
            f"goal_pos={[round(x, 1) for x in goal_pos.tolist()]}, "
            f"distance={goal_dist:.1f}"
        )

        if not same_side:
            # Different sides of wall — fly toward open edge (rim)
            up_raw = sensor_data.get("up_direction")
            up = (
                np.asarray(up_raw, dtype=float)
                if up_raw is not None
                else np.array([0.0, 0.0, 1.0])
            )

            # Project up onto tangent plane
            up_tangent = up - np.dot(up, n) * n
            up_tangent_len = np.linalg.norm(up_tangent)

            if up_tangent_len > 0.3:
                # Wall: fly along surface toward rim + away from surface
                #up_tangent /= up_tangent_len
                #fly_dir = up_tangent * 0.8 + n * 0.4
                #fly_dir /= (np.linalg.norm(fly_dir) + 1e-12)
                # Wall: fly along surface toward rim (straight up)
                up_tangent /= up_tangent_len
                fly_dir = up_tangent
                logger.debug(
                    f"DETACH_FLY_DIR: opposite sides, wall, "
                    f"up_tangent={[round(x, 3) for x in up_tangent.tolist()]}, "
                    f"fly_dir={[round(x, 3) for x in fly_dir.tolist()]}"
                )
                return fly_dir
            else:
                # Horizontal surface (bottom/top): fly away from center
                center_raw = sensor_data.get("object_center")
                if center_raw is not None:
                    center = np.asarray(center_raw, dtype=float)
                    away = current_pose[:3] - center
                    away_t = away - np.dot(away, n) * n
                    away_len = np.linalg.norm(away_t)
                    if away_len > 1e-8:
                        away_t /= away_len
                        logger.debug(
                            f"DETACH_FLY_DIR: opposite sides, horizontal, "
                            f"fly_dir={[round(x, 3) for x in away_t.tolist()]}"
                        )
                        return away_t

                logger.debug(
                    "DETACH_FLY_DIR: opposite sides, fallback to normal"
                )
                return n.copy()

        # Same side — original tangent logic
        tangent = goal_dir - np.dot(goal_dir, n) * n
        tangent_len = np.linalg.norm(tangent)

        up_raw = sensor_data.get("up_direction")
        if up_raw is not None:
            up = np.asarray(up_raw, dtype=float)
            normal_horizontality = 1.0 - abs(float(np.dot(n, up)))
        else:
            normal_horizontality = 1.0

        if tangent_len < 1e-8:
            fly_dir = n.copy()
            logger.debug(
                f"DETACH_FLY_DIR: same_side, tangent degenerate, "
                f"fly_dir=normal={[round(x, 3) for x in fly_dir.tolist()]}"
            )
            return fly_dir

        tangent /= tangent_len
        fly_dir = tangent * 0.9 + n * 0.3 * normal_horizontality
        fly_dir /= (np.linalg.norm(fly_dir) + 1e-12)
        logger.debug(
            f"DETACH_FLY_DIR: same_side, "
            f"tangent={[round(x, 3) for x in tangent.tolist()]}, "
            f"normal_horizontality={normal_horizontality:.2f}, "
            f"fly_dir={[round(x, 3) for x in fly_dir.tolist()]}"
        )
        return fly_dir
                
    def _compute_subgoal_direction(
        self,
        current_pose: np.ndarray,
        sensor_data: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        """Compute world-space direction toward object edge."""
        normal = sensor_data.get("point_normal")
        if normal is None:
            return None

        n_world = np.asarray(normal, dtype=float)
        n_len = float(np.linalg.norm(n_world))
        if n_len < 1e-8:
            return None
        n_hat = n_world / n_len

        to_goal = self._current_goal[:3] - current_pose[:3]
        tangent_to_goal = to_goal - np.dot(to_goal, n_hat) * n_hat
        tangent_len = float(np.linalg.norm(tangent_to_goal))

        if tangent_len < 1e-8:
            return None

        return tangent_to_goal / tangent_len

    def _estimate_edge_distance(
        self,
        alignment: float,
        distance: float,
    ) -> float:
        """Estimate surface distance to object edge."""
        if alignment >= 0.0:
            return 0.0

        estimated_radius = max(distance / 2.0, 5.0)
        theta = np.arccos(np.clip(alignment, -1.0, 1.0))
        arc = estimated_radius * (theta - np.pi / 2.0)

        return max(arc, 0.0)

    def _best_tangential_toward(
        self,
        target_dir_world: np.ndarray,
        current_pose: np.ndarray,
        sensor_data: Dict[str, Any],
    ) -> Optional[int]:
        """Find tangential action index closest to target direction."""
        normal = sensor_data.get("point_normal")
        if normal is None:
            return None

        n_world = np.asarray(normal, dtype=float)
        n_len = float(np.linalg.norm(n_world))
        if n_len < 1e-8:
            return None
        n_hat = n_world / n_len

        rot = R.from_euler("xyz", current_pose[3:6], degrees=True)
        right_world = rot.apply([1.0, 0.0, 0.0])

        tb1 = right_world - np.dot(right_world, n_hat) * n_hat
        tb1_norm = float(np.linalg.norm(tb1))
        if tb1_norm < 1e-8:
            up_world = rot.apply([0.0, 1.0, 0.0])
            tb1 = up_world - np.dot(up_world, n_hat) * n_hat
            tb1_norm = float(np.linalg.norm(tb1))
        if tb1_norm < 1e-8:
            return None
        tb1 = tb1 / tb1_norm
        tb2 = np.cross(n_hat, tb1)
        tb2_len = float(np.linalg.norm(tb2))
        if tb2_len < 1e-8:
            return None
        tb2 = tb2 / tb2_len

        best_idx = None
        best_dot = -1e18

        for i, deg in enumerate(self.action_space.SURFACE_DIRECTIONS):
            a = np.radians(deg)
            v = np.cos(a) * tb1 + np.sin(a) * tb2
            v_len = float(np.linalg.norm(v))
            if v_len < 1e-8:
                continue
            v = v / v_len
            dot = float(np.dot(v, target_dir_world))
            if dot > best_dot:
                best_dot = dot
                best_idx = i

        return best_idx

    def _is_on_horizontal_surface(self, sensor_data: dict) -> bool:
        """Check if agent is on a horizontal surface (bottom/top face).

        Used to suppress detach on horizontal surfaces where the agent
        can crawl to the edge and transition to a vertical wall.

        Returns:
            True if surface normal is within ~30° of up_direction.
        """
        point_normal = sensor_data.get("point_normal")
        if point_normal is None:
            return False
        n = np.asarray(point_normal, dtype=float)
        n_len = np.linalg.norm(n)
        if n_len < 1e-8:
            return False
        up_dir = np.asarray(
            sensor_data.get("up_direction", [0, 0, 1]),
            dtype=float,
        )
        return abs(float(np.dot(n / n_len, up_dir))) > 0.85

    # ══════════════════════════════════════════════════════════
    # COORDINATE TRANSFORMS
    # ══════════════════════════════════════════════════════════
    def _world_to_local(
        self,
        vector_world: np.ndarray,
        pose: np.ndarray,
    ) -> np.ndarray:
        rotation = R.from_euler("xyz", pose[3:6], degrees=True)
        return rotation.inv().apply(vector_world)

    @staticmethod
    def _normalize_angles(angles: np.ndarray) -> np.ndarray:
        return (angles + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _normalize_angles_deg(angles_deg: np.ndarray) -> np.ndarray:
        return (angles_deg + 180.0) % 360.0 - 180.0

    @staticmethod
    def _normalize_values(values: np.ndarray) -> np.ndarray:
        v_min = np.min(values)
        v_max = np.max(values)
        v_range = v_max - v_min
        if v_range < 1e-8:
            return np.zeros_like(values)
        return 2.0 * (values - v_min) / v_range - 1.0

    @staticmethod
    def _softmax_sample(values: np.ndarray, temperature: float) -> int:
        v = values / temperature
        v = v - np.max(v)
        exp_v = np.exp(v)
        probs = exp_v / exp_v.sum()
        return int(np.random.choice(len(values), p=probs))

    # ══════════════════════════════════════════════════════════
    # EPISODE LIFECYCLE
    # ══════════════════════════════════════════════════════════
    def _on_episode_done(self, final_state, termination_reason):
        distance = np.linalg.norm(final_state[0:3])
        goal_reached = termination_reason == "goal_reached"
        start_distance = np.linalg.norm(
            self._current_goal[0:3] - self.start_pos
        )

        if goal_reached:
            self._total_goals_reached += 1

        if goal_reached:
            reason = "GOAL_REACHED!!!"
            reason_key = "goal_reached"
            logger.info(
                f"Episode {self._total_episodes} DONE: {reason}, "
                f"start_dist={start_distance:.1f}mm, "
                f"{self._steps} steps, "
                f"reward={self._episode_reward:.1f}, "
                f"final_dist={distance:.1f}mm, "
                f"epsilon={self.epsilon:.3f}, "
                f"success_rate="
                f"{self._total_goals_reached}/{self._total_episodes}"
            )
        elif termination_reason == "collision_surface_violation":
            reason = "collision_surface_violation"
            reason_key = "collision_surface_violation"
        elif termination_reason == "collision_lost_object":
            reason = "collision_lost_object"
            reason_key = "collision_lost_object"
        elif termination_reason == "collision_other":
            reason = "collision_other"
            reason_key = "collision_other"
        elif termination_reason == "timeout":
            reason = "timeout"
            reason_key = "timeout"
        else:
            reason = "unknown"
            reason_key = "collision_other"

        self._termination_counts[reason_key] = (
            self._termination_counts.get(reason_key, 0) + 1
        )

        logger.debug(
            f"Episode {self._total_episodes} DONE: {reason}, "
            f"{self._steps} steps, "
            f"reward={self._episode_reward:.1f}, "
            f"final_dist={distance:.1f}mm, "
            f"epsilon={self.epsilon:.3f}, "
            f"success_rate="
            f"{self._total_goals_reached}/{self._total_episodes}"
        )

        self._current_goal = None
        self._prev_state = None
        self._prev_sensor_data = None
        self._last_action = None
        self._prev_action = None
        self._steps = 0
        self._episode_reward = 0.0
        self._episode_transitions = []
        self._distance_history = []
        self._flyby_count = 0
        self._cached_fly_direction = None
        self._fly_direction_age = 0
        self._air_phase = "DIRECT"

    # ══════════════════════════════════════════════════════════
    # DIAGNOSTICS
    # ══════════════════════════════════════════════════════════
    def get_stats(self) -> Dict[str, Any]:
        success_rate = self._total_goals_reached / max(
            self._total_episodes, 1
        )
        episodes = max(self._total_episodes, 1)
        termination_rates = {
            k: float(v) / episodes
            for k, v in self._termination_counts.items()
        }

        collision_rate_per_action = {}
        for action_name, collision_count in self._collision_stats.items():
            total_calls = self._global_action_counts.get(action_name, 0)
            collision_rate_per_action[action_name] = float(
                collision_count
            ) / max(total_calls, 1)

        steps_per_success = float(self._total_steps) / max(
            self._total_goals_reached, 1
        )

        surface_actions = {
            "move_tangentially",
            "orient_horizontal",
            "orient_vertical",
        }
        air_actions = {
            "free_forward",
            "free_forward_small",
            "free_backward",
            "look_up",
            "look_down",
            "turn_left",
            "turn_right",
        }
        surface_steps = sum(
            self._global_action_counts.get(a, 0) for a in surface_actions
        )
        air_steps = sum(
            self._global_action_counts.get(a, 0) for a in air_actions
        )
        total_nav_steps = max(surface_steps + air_steps, 1)

        stats = {
            "total_episodes": self._total_episodes,
            "total_steps": self._total_steps,
            "total_goals_reached": self._total_goals_reached,
            "success_rate": success_rate,
            "epsilon": self.epsilon,
            "current_episode_steps": self._steps,
            "current_episode_reward": self._episode_reward,
            "termination_counts": dict(self._termination_counts),
            "termination_rates": termination_rates,
            "q_store_free": self.q_store_free.get_stats(),
            "q_store_surface": self.q_store_surface.get_stats(),
            "collision_stats": dict(self._collision_stats),
            "global_action_counts": dict(self._global_action_counts),
            "collision_rate_per_action": collision_rate_per_action,
            "steps_per_success": steps_per_success,
            "surface_air_ratio": {
                "surface_steps": surface_steps,
                "air_steps": air_steps,
                "surface_ratio": float(surface_steps) / total_nav_steps,
                "air_ratio": float(air_steps) / total_nav_steps,
            },
        }

        return stats

    def update_only(self, current_pose, sensor_data, action_index):
        if self._current_goal is None:
            return None, True

        self._steps += 1
        self._total_steps += 1
        self._last_detach_sub_steps = sensor_data.get(
            "detach_sub_steps", 1
        )

        state = self._compute_state(current_pose, sensor_data)
        collision = self._detect_collision(sensor_data)
        done = False

        if self._prev_state is not None:
            prev_store = self._select_store(self._prev_state)
            next_store = self._select_store(state)

            reward, done, termination_reason = self._compute_reward(
                state, self._prev_state, self._last_action, collision
            )
            self._episode_reward += reward
            self._episode_transitions.append(
                {
                    "state": self._prev_state.copy(),
                    "action": int(self._last_action),
                    "reward": float(reward),
                }
            )

            if done:
                td_target = reward
            else:
                next_q = next_store.get_q_values(state)
                td_target = reward + self.gamma * np.max(next_q)

            prev_store.update_q_value(
                self._prev_state,
                self._last_action,
                td_target,
                self._get_learning_rate(),
            )

            if done:
                if termination_reason == "goal_reached":
                    if self.is_training:
                        self._apply_success_backup_updates()
                    self.success_trails = (
                        self._episode_transitions.copy()
                    )
                self._on_episode_done(state, termination_reason)
                return state, True

        self._prev_state = state
        self._prev_sensor_data = sensor_data
        self._prev_action = self._last_action
        self._last_action = action_index

        if not self.is_training:
            self.epsilon = self.eval_epsilon

        return state, False

    # ══════════════════════════════════════════════════════════
    # PERSISTENCE
    # ══════════════════════════════════════════════════════════
    def save(self, dirpath: str):
        pathlib.Path(dirpath).mkdir(exist_ok=True, parents=True)

        self.q_store_free.save_with_index(
            os.path.join(dirpath, "q_store_free")
        )
        self.q_store_surface.save_with_index(
            os.path.join(dirpath, "q_store_surface")
        )

        controller_state = {
            "epsilon": self.epsilon,
            "total_episodes": self._total_episodes,
            "total_steps": self._total_steps,
            "total_goals_reached": self._total_goals_reached,
            "mode": self.mode,
            "config": self.config,
        }

        np.savez(
            os.path.join(dirpath, "controller_state.npz"),
            **{
                k: np.array(v)
                if not isinstance(v, dict)
                else np.array([0])
                for k, v in controller_state.items()
            },
        )

        with pathlib.Path(
            os.path.join(dirpath, "config.json")
        ).open("w") as f:
            json.dump(self.config, f, indent=2)

        logger.info("Controller saved to %s", dirpath)

    @classmethod
    def load(
        cls, dirpath: str, agent_id: str, config=None
    ) -> "RLGoalApproachController":
        with pathlib.Path(
            os.path.join(dirpath, "config.json")
        ).open() as f:
            saved_config = json.load(f)
        cfg = {**saved_config, **(config or {})}

        controller = cls(agent_id=agent_id, config=cfg)

        controller.q_store_free = HNSWStateStore.load_with_index(
            os.path.join(dirpath, "q_store_free"), extra_cfg=config
        )
        controller.q_store_surface = HNSWStateStore.load_with_index(
            os.path.join(dirpath, "q_store_surface"), extra_cfg=config
        )

        state_data = np.load(
            os.path.join(dirpath, "controller_state.npz"),
            allow_pickle=False,
        )
        loaded_epsilon = float(state_data["epsilon"])
        loaded_total_episodes = int(state_data["total_episodes"])
        loaded_total_steps = int(state_data["total_steps"])
        loaded_total_goals_reached = int(
            state_data["total_goals_reached"]
        )

        logger.info(
            f"Controller loaded from {dirpath}: "
            f"{loaded_total_episodes} loaded_total_episodes, "
            f"{loaded_total_steps} loaded_total_steps, "
            f"{loaded_total_goals_reached} loaded_total_goals_reached, "
            f"loaded_epsilon={loaded_epsilon:.3f}, "
            f"loaded Q-store surface="
            f"{controller.q_store_surface.get_stats()['num_points']} points, "
            f"loaded Q-store free="
            f"{controller.q_store_free.get_stats()['num_points']} points"
        )

        return controller
