"""
RL Goal Approach Controller.

Receives goal_pose, uses current proprioceptive state + sensor data to choose actions,
and learns from dense reward (distance reduction to goal).
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple, List
from scipy.spatial.transform import Rotation as R
import os
import json

from tbp.monty.frameworks.actions.actions import Action

from .hnsw_state_store import HNSWStateStore
from .action_space import ActionSpace
from .config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


class RLGoalApproachController:
    """Q-learning controller that moves agent toward goal pose.

    Main loop (called each step by motor policy):
        1. Compute state from current pose, goal pose, sensor data
        2. Detect sensory collisions
        3. Compute reward and update Q-values (if not first step)
        4. Choose action via heuristic-guided exploration
        5. Return Action object

    State vector (13D):
        local_pos_error  [3D]: direction to goal in agent's local frame
        rot_error        [3D]: orientation error (normalized angles)
        local_normal     [3D]: surface normal in agent's local frame
        on_object        [1D]: whether sensor sees object surface
        alignment        [1D]: dot(goal_direction, point_normal)
        distance         [1D]: Euclidean distance to goal
        norm_depth       [1D]: normalized depth to nearest surface

    Args:
        agent_id: agent identifier.
        config: Dictionary with optional overrides for all parameters.
    """

    def __init__(self, agent_id: str, config: Optional[Dict] = None):
        # Merge config with defaults
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.agent_id = agent_id
        # Режим работы
        self.mode = self.config.get("mode", "auto")
        # "train"    — всегда обучение
        # "train_adapt_epsilon"    — обучение с адаптивным epsilon
        # "eval"     — всегда inference
        # "auto"     — определяем автоматически
        
        # Epsilon для eval режима
        self.eval_epsilon = self.config.get("eval_epsilon", 0.02)
        self.eval_alpha_multiplier = self.config.get("eval_alpha_multiplier", 0.1)
        
        # Порог для auto режима
        self.auto_train_threshold = self.config.get(
            "auto_train_threshold", 100
        )

        # Action space
        self.action_space = ActionSpace(
            agent_id=agent_id,
            surface_step=self.config["surface_step"],
            free_step=self.config["free_step"],
            rotation_step=self.config["rotation_step"],
        )
        self.num_actions = self.action_space.NUM_ACTIONS

        # HNSW Q-store
        # self.q_store = HNSWStateStore(config=self.config)
        self.q_store_free = HNSWStateStore(config=self.config, name="free")
        self.q_store_surface = HNSWStateStore(config=self.config, name="surface")

        # Q-learning parameters
        self.gamma = self.config["gamma"]
        self.alpha = self.config["alpha"]
        self.epsilon = self.config["epsilon_start"]
        self.epsilon_min = self.config["epsilon_min"]
        self.epsilon_decay = self.config["epsilon_decay"]
        # Optional successful-episode backup updates.
        # Keep defaults local for backward compatibility with old configs.
        self.success_backup_enabled = bool(
            self.config.get("success_backup_enabled", True)
        )
        # Default: cover the full episode horizon so detour steps get credit.
        # -1 means "all transitions in the episode" (no hard cap).
        self.success_backup_steps = int(
            self.config.get(
                "success_backup_steps",
                self.config.get("max_steps_per_goal", -1),
            )
        )
        # λ=0.9 → depth-20 weight ≈ 0.12, depth-40 ≈ 0.015.
        # Higher than 0.8 because we need detour credit to survive 20-40 steps.
        self.success_backup_lambda = float(
            self.config.get("success_backup_lambda", 0.9)
        )
        self.success_backup_alpha_multiplier = float(
            self.config.get("success_backup_alpha_multiplier", 0.5)
        )
        if self.mode == "train_adapt_epsilon":
            # Choose multiplicative decay so that after all planned training
            # steps epsilon goes from epsilon_start to epsilon_min:
            #   eps_T = eps_0 * decay^T = eps_min
            #   decay = (eps_min / eps_0)^(1 / T)
            total_steps = max(
                1,
                int(self.config.get("num_episodes", 1))
                * int(self.config.get("max_steps_per_goal", 1)),
            )
            eps_start = float(self.config.get("epsilon_start", self.epsilon))
            eps_min = float(self.config.get("epsilon_min", self.epsilon_min))
            if eps_start > eps_min > 0.0:
                self.epsilon_decay = (eps_min / eps_start) ** (1.0 / total_steps)

        # Episode state (reset each new goal)
        self._prev_state: Optional[np.ndarray] = None
        self._prev_sensor_data: Optional[Dict] = None
        self._last_action: Optional[int] = None
        self._prev_action: Optional[int] = None  # action before _last_action
        self._steps: int = 0
        self._current_goal: Optional[np.ndarray] = None
        self._episode_reward: float = 0.0
        self._episode_transitions: List[Dict[str, Any]] = []
        self.success_trails = []
        self.start_pos: Optional[np.ndarray] = None

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
        """Определяем режим."""
        if self.mode in ("train", "train_adapt_epsilon"):
            return True
        elif self.mode == "eval":
            return False
        else:
            # AUTO: если мало опыта → train, иначе → eval
            return self.q_store.next_id < self.auto_train_threshold

    def set_new_goal(self, goal_pose: np.ndarray, start_pos: np.ndarray):
        """Called when LM provides a new goal state.

        Resets episode state but preserves learned Q-values.

        Args:
            goal_pose: Target pose [x, y, z, roll, pitch, yaw].
                Position in mm, rotation in radians.
        """
        self._current_goal = goal_pose.copy()
        self._prev_state = None
        self._prev_sensor_data = None
        self._last_action = None
        self._prev_action = None
        self._steps = 0
        self._episode_reward = 0.0
        self._episode_transitions = []
        self._total_episodes += 1
        self.start_pos = start_pos.copy()

        logger.debug(
            f"New goal set (episode {self._total_episodes}): "
            f"pos={goal_pose[:3]}, rot={np.degrees(goal_pose[3:])}°"
        )

    def step(
        self,
        current_pose: np.ndarray,
        sensor_data: Dict[str, Any],
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Execute one step of the RL controller.

        This is the main method called by motor policy each timestep.

        Flow:
            1. Build state vector from current pose + sensors
            2. If we have previous state: compute reward, update Q
            3. Check if episode is done (goal reached / collision / timeout)
            4. Choose action via heuristic-guided exploration
            5. Return Monty Action

        Args:
            current_pose: Current agent pose [x, y, z, roll, pitch, yaw].
            sensor_data: Dict with keys:
                'point_normal': [nx, ny, nz] or None
                'principal_curvatures': [k1, k2] or None
                'on_object': bool
                'depth': float (mm)

        Returns:
            Monty Action object, or None if episode is done
            (goal reached, collision, or timeout).
        """
        if self._current_goal is None:
            logger.warning("step() called without goal. Call set_new_goal() first.")
            return None, None

        self._steps += 1
        self._total_steps += 1

        # Build state
        state = self._compute_state(current_pose, sensor_data)
        logger.debug(
            f"STEP {self._steps}: action={self._last_action}, "
            f"dist={state[13]:.1f}, on={state[11]:.0f}, "
            f"depth={sensor_data.get('depth',0):.1f}, "
            f"pos={current_pose[:3]}"
        )

        # Detect collision
        collision = self._detect_collision(sensor_data)
        if collision:
            logger.debug(f"COLLISION: {collision} at step {self._steps}")

        # Learn from previous step
        done = False

        # Two stores select
        if self._prev_state is not None:
            prev_store = self._select_store(self._prev_state)
        next_store = self._select_store(state)
        
        if self._prev_state is not None:
            reward, done, termination_reason = self._compute_reward(
                state, self._prev_state, self._last_action, collision
            )
            self._episode_reward += reward
            self._episode_transitions.append({
                "state": self._prev_state.copy(),
                "action": int(self._last_action),
                "reward": float(reward),
            })

            # Q-learning update
            if done:
                td_target = reward
            else:
                # next_q = self.q_store.get_q_values(state)
                next_q = next_store.get_q_values(state)
                td_target = reward + self.gamma * np.max(next_q)

            #self.q_store.update_q_value(
            #    self._prev_state, self._last_action, td_target, self.alpha
            #)
            # Обучение: обновляем Q-values
            #self.q_store.update_q_value(
            #    self._prev_state,
            #    self._last_action,
            #    td_target,
            #    self._get_learning_rate(),
            #)
            prev_store.update_q_value(self._prev_state, self._last_action, td_target, self._get_learning_rate())

            if done:
                if termination_reason == "goal_reached":
                    if self.is_training:
                        self._apply_success_backup_updates()
                    self.success_trails = self._episode_transitions.copy()
                self._on_episode_done(state, termination_reason)
                return None, None

        # Choose action
        action_index, explanation = self._choose_action(
            state=state,
            current_pose=current_pose,
            sensor_data=sensor_data,
            explain=True
        )

        # Save for next step
        self._prev_state = state
        self._prev_sensor_data = sensor_data
        self._prev_action = self._last_action
        self._last_action = action_index

        # Decay epsilon только в training
        if self.is_training:
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay,
            )
        else:
            self.epsilon = self.eval_epsilon

        # Convert to Action
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
        """Whether controller has an active goal."""
        return self._current_goal is not None

    # ══════════════════════════════════════════════════════════
    # STATE COMPUTATION
    # ══════════════════════════════════════════════════════════
    def _compute_state(self, current_pose, sensor_data):
        goal = self._current_goal

        pos_error_world = goal[:3] - current_pose[:3]
        local_pos_error = self._world_to_local(pos_error_world, current_pose)

        rot_error_deg = self._normalize_angles_deg(goal[3:] - current_pose[3:])

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

        depth = sensor_data.get(
            "depth", self.config["max_sensor_range"]
        )
        norm_depth = min(depth / self.config["max_sensor_range"], 1.0)

        curvatures = sensor_data.get("principal_curvatures", [0.0, 0.0])
        mean_curv = float(curvatures[0])
        gauss_curv = float(curvatures[1])

        state = np.concatenate([
            local_pos_error,        # [0:3]   3D
            rot_error_deg,          # [3:6]   3D
            local_normal,           # [6:9]   3D
            [mean_curv],            # [9]     1D
            [gauss_curv],           # [10]    1D
            [on_object],            # [11]    1D
            [alignment],            # [12]    1D
            [distance],             # [13]    1D
            [norm_depth],           # [14]    1D
        ])

        return state
    # ══════════════════════════════════════════════════════════
    # COLLISION DETECTION
    # ══════════════════════════════════════════════════════════
    def _detect_collision(self, sensor_data):
        if sensor_data.get("passed_through", False):
            return "surface_violation"

        depth = sensor_data.get("depth", self.config["max_sensor_range"])

        was_on = (self._prev_sensor_data is not None
                  and self._prev_sensor_data.get("on_object", False))
        now_on = sensor_data.get("on_object", False)

        prev_depth = (self._prev_sensor_data.get("depth", self.config["max_sensor_range"])
                      if self._prev_sensor_data is not None
                      else self.config["max_sensor_range"])

        logger.debug(
            f"COLLISION_CHECK: was_on={was_on}, now_on={now_on}, "
            f"depth={depth:.3f}, prev_depth={prev_depth:.3f}"
        )

        if was_on and prev_depth > 1.5 and depth < self.config["min_valid_depth"]:
            return "surface_violation"

        if (self._prev_sensor_data is not None
                and was_on
                and now_on):
            prev_normal = self._prev_sensor_data.get("point_normal")
            curr_normal = sensor_data.get("point_normal")
            if prev_normal is not None and curr_normal is not None:
                dot = np.dot(np.array(prev_normal), np.array(curr_normal))
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
            min_progress = -surface_step * cfg["detour_negative_progress_clip_steps"]
            progress = max(progress_raw, min_progress)

        if action == self.action_space.IDX_DETACH or action == self.action_space.IDX_DETACH_EDGE:
            sub_steps = max(getattr(self, '_last_detach_sub_steps', 1), 1)
            reward += progress / (self.action_space.free_step * sub_steps) * cfg["reward_progress"]
        else:
            reward += progress / surface_step * cfg["reward_progress"]

        # ═══ 2. Goal reached ═══
        if distance < cfg["goal_threshold"]:
            reward += cfg["reward_goal_reached"]
            done = True
            termination_reason = "goal_reached"

        # ═══ 3. Step penalty ═══
        reward += cfg["reward_step_penalty"]
        if action == self.action_space.IDX_DETACH or action == self.action_space.IDX_DETACH_EDGE:
            sub_steps = max(getattr(self, '_last_detach_sub_steps', 1), 1)
            reward += cfg["reward_step_penalty"] * (sub_steps - 1)

        # ═══ 4. Collisions ═══
        if collision == "surface_violation":
            reward += cfg["reward_surface_violation"]
            done = True
            termination_reason = "collision_surface_violation"
            action_name = self.action_space.get_info(action).name if action is not None else "unknown"
            self._collision_stats[action_name] = self._collision_stats.get(action_name, 0) + 1
            logger.info(
                f"COLLISION_DETAIL: action={action_name}, "
                f"depth={state[14]*100:.1f}mm, "
                f"on_object={state[11]:.0f}, "
                f"alignment={state[12]:.3f}, "
                f"distance={state[13]:.1f}, "
            )
        elif collision == "lost_object":
            if action != self.action_space.IDX_DETACH and action != self.action_space.IDX_DETACH_EDGE:
                reward += cfg["reward_drifted_away"]

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

    # ══════════════════════════════════════════════════════════
    # ACTION SELECTION
    # ══════════════════════════════════════════════════════════
    def _get_learning_rate(self) -> float:
        """Return effective learning rate for current mode."""
        if self.is_training:
            return self.alpha
        return self.alpha * self.eval_alpha_multiplier

    def _apply_success_backup_updates(self) -> None:
        """Apply backward updates over the whole successful trajectory.

        Uses a truncated Monte Carlo return with exponentially decaying
        learning rate (lambda decay) so that:
          - The terminal goal reward propagates all the way back to step 0.
          - Actions taken far from the goal receive a much smaller update
            than the final steps, avoiding high-variance over-correction.

        This is the right approach when episodes are bounded by
        max_steps_per_goal and the reward is partly sparse (goal_reached
        bonus): 1-step TD alone would need many revisits to propagate the
        terminal signal back through a 30-step detour.

        success_backup_steps <= 0 or missing  →  update the full episode.
        """
        if not self.success_backup_enabled:
            return
        if not self._episode_transitions:
            return

        # steps <= 0  means "all" (full episode bounded by max_steps_per_goal)
        if self.success_backup_steps > 0:
            k = min(self.success_backup_steps, len(self._episode_transitions))
        else:
            k = len(self._episode_transitions)
        tail = self._episode_transitions[-k:]
        base_alpha = self._get_learning_rate() * self.success_backup_alpha_multiplier
        if base_alpha <= 0.0:
            return

        # Reverse pass: truncated Monte Carlo return from terminal success.
        # count_visit=False: we are refining Q-values for already-visited
        # states, not counting new organic visits. This prevents visit_count
        # inflation that would distort eviction scores (visit_count × recency).
        # last_step is still refreshed inside update_q_value so successful-path
        # points remain eviction-fresh.
        g_return = 0.0
        for depth, tr in enumerate(reversed(tail)):
            g_return = tr["reward"] + self.gamma * g_return
            lr = base_alpha * (self.success_backup_lambda ** depth)
            store = self._select_store(tr["state"])
            store.update_q_value(
                tr["state"],
                tr["action"],
                g_return,
                lr,
                count_visit=False,
            )

        logger.debug(
            "Applied success backup updates: k=%d, base_alpha=%.4f, lambda=%.3f",
            k,
            base_alpha,
            self.success_backup_lambda,
        )

    def _get_current_epsilon(self):
        """Epsilon зависит от режима."""
        if self.is_training:
            return self.epsilon  # decaying
        else:
            return self.eval_epsilon  # фиксированный, маленький
    
    def _generate_choice_interpretation(
        self,
        action_index: int,
        is_random: bool,
        q_recommends: int,
        h_recommends: int,
        dominant_heuristic: str,
        eps: float,
        confidence: float,
        blend: str,
        is_heuristic_override: bool
    ) -> str:
        """Generate natural language interpretation of action choice."""
        action_name = self.action_space.get_info(action_index).name
        q_action_name = self.action_space.get_info(q_recommends).name
        h_action_name = self.action_space.get_info(h_recommends).name

        if is_random:
            return (
                f"##### Случайное действие: {action_name} - {action_index} "
                f"epsilon {eps}."
            )
        elif is_heuristic_override:
            return (
                f"##### Эвристика действие: {action_name} - {action_index} "
                f"epsilon {eps}."
            )

        return (
            f"##### Softmax действие; {action_name}, уверенность {confidence:.0%}). "
            f"blend: {blend}. "
            f"Q рекомендует: {q_action_name} - {q_recommends}; Эвристика: {h_action_name} - {h_recommends}. "
        )
    
    def _choose_action(self,
        state: np.ndarray,
        current_pose: np.ndarray,
        sensor_data: Dict[str, Any],
        explain: bool = False
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

        has_q_data = store.next_id > 0 and np.max(np.abs(q_values)) > 1e-6

        if has_q_data:
            combined = (1 - eps) * q_norm + eps * h_norm
            temperature = np.clip(0.5 * eps, 0.01, 0.5)
        else:
            combined = h_norm.copy()
            temperature = 0.05

        if self.temperature_override is not None:
            temperature = self.temperature_override

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
            action_index = np.random.randint(self.num_actions)
            probs = np.zeros(self.num_actions)
            probs[action_index] = 1.0
        else:
            action_index = int(np.random.choice(len(probs), p=probs))

        if not explain:
            return action_index, None

        contributions = {
            name: float(np.max(bias)) for name, bias in heuristic_components.items()
        }
        dominant_heuristic = max(contributions, key=contributions.get) if contributions else "none"

        confidence = float(probs[action_index])

        explanation = {
            "chosen_action": {
                "index": action_index,
                "name": self.action_space.get_info(action_index).name,
                "probability": confidence,
            },
            "sampling_method": "random_exploration" if is_random_override else "softmax_sampling",
            "temperature": temperature,
            "epsilon": eps,
            "has_q_data": has_q_data,
            "blend": f"{(1-eps)*100:.0f}% Q + {eps*100:.0f}% heuristic" if has_q_data else "100% heuristic",
            "is_random_override": is_random_override,
            "action_probabilities": {
                self.action_space.get_info(i).name: float(probs[i])
                for i in range(self.num_actions)
            },
            "advice": {
                "q_recommends": {
                    "index": best_q_action,
                    "name": self.action_space.get_info(best_q_action).name,
                },
                "heuristic_recommends": {
                    "index": best_h_action,
                    "name": self.action_space.get_info(best_h_action).name,
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
                blend=f"{(1-eps)*100:.0f}% Q + {eps*100:.0f}% heuristic" if has_q_data else "100% heuristic",
                is_heuristic_override=is_heuristic_override,
            ),
        }
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

        eps = 1e-8

        rot = R.from_euler("xyz", current_pose[3:6], degrees=True)

        # Адаптивный порог detach: на кривых поверхностях alignment
        # быстрее становится отрицательным, порог мягче
        curvature = abs(float(state[9])) + abs(float(state[10]))
        DETACH_ALIGN_THR = -0.3 + min(curvature * 5.0, 0.2)
        SURFACE_STRENGTH = 2.0

        # Близко к цели: distance < 9mm И цель доступна по поверхности
        # Используется для разрешения orient действий и блокировки detach
        close_to_goal = (
            distance < 3.0 * self.action_space.surface_step
            and alignment > DETACH_ALIGN_THR
        )

        # ────────────────────────────────────────────────────
        # 0) SUPPRESS: подавить бесполезные действия
        # Ситуация: всегда
        # Принцип: rotate_sensor не помогает навигации,
        # orient_hor/vert полезны только рядом с целью на поверхности
        # ────────────────────────────────────────────────────
        suppress = np.zeros(self.num_actions, dtype=float)
        suppress[self.action_space.IDX_ROTATE_POS] -= 2.0
        suppress[self.action_space.IDX_ROTATE_NEG] -= 2.0

        if not (on_object > 0.5 and close_to_goal):
            suppress[self.action_space.IDX_ORIENT_HOR] -= 2.0
            suppress[self.action_space.IDX_ORIENT_VERT] -= 2.0

        bias += suppress
        components["suppress"] = suppress

        # ────────────────────────────────────────────────────
        # 1) SURFACE MOVE: ползти по поверхности к цели
        # Ситуация: агент на поверхности, цель доступна (alignment >= порог)
        # Принцип: проецируем направление на цель на касательную плоскость,
        # выбираем из 8 tangential направлений то, которое максимально
        # уменьшает тангенциальную ошибку
        # ────────────────────────────────────────────────────
        surface_move = np.zeros(self.num_actions, dtype=float)

        if on_object > 0.5 and alignment >= DETACH_ALIGN_THR:
            n_world = sensor_data.get("point_normal", None)

            if n_world is not None:
                n_world = np.asarray(n_world, dtype=float)
                n_len = float(np.linalg.norm(n_world))
                if n_len > eps:
                    n_hat = n_world / n_len

                    e_world = rot.apply(local_pos_error)
                    e_t = e_world - np.dot(e_world, n_hat) * n_hat
                    step = float(self.action_space.surface_step)

                    tangential_dist = float(np.linalg.norm(e_t))
                    normal_dist = abs(float(np.dot(e_world, n_hat)))

                    # should_crawl: ползти если цель больше "вдоль" чем "за" поверхностью
                    # Если normal_dist >> tangential_dist — detach эвристика решит
                    should_crawl = not (normal_dist > tangential_dist * 2.0 and distance > 3 * step)

                    if should_crawl:
                        right_world = rot.apply([1.0, 0.0, 0.0])
                        tb1 = right_world - np.dot(right_world, n_hat) * n_hat
                        tb1_norm = np.linalg.norm(tb1)

                        if tb1_norm < 1e-8:
                            up_world = rot.apply([0.0, 1.0, 0.0])
                            tb1 = up_world - np.dot(up_world, n_hat) * n_hat
                            tb1_norm = np.linalg.norm(tb1)

                        if tb1_norm < 1e-8:
                            tmp = np.array([0.0, 1.0, 0.0])
                            if abs(np.dot(tmp, n_hat)) > 0.9:
                                tmp = np.array([0.0, 0.0, 1.0])
                            tb1 = np.cross(n_hat, tmp)
                            tb1_norm = np.linalg.norm(tb1)

                        tb1 /= (tb1_norm + 1e-12)
                        tb2 = np.cross(n_hat, tb1)
                        tb2 /= (np.linalg.norm(tb2) + 1e-12)

                        best = None
                        best_score = -1e18
                        scores = np.full(8, -1e18, dtype=float)

                        for i, deg in enumerate(self.action_space.SURFACE_DIRECTIONS):
                            a = np.radians(deg)
                            v_world = np.cos(a) * tb1 + np.sin(a) * tb2
                            v_norm = float(np.linalg.norm(v_world))
                            if v_norm < 1e-8:
                                continue
                            v_world /= v_norm

                            new_e = e_t - step * v_world
                            score = float(np.dot(e_t, e_t) - np.dot(new_e, new_e))
                            scores[i] = score
                            if score > best_score:
                                best_score = score
                                best = i

                        HYST_ABS = 0.25
                        if prev_action is not None and 0 <= prev_action < 8 and best is not None:
                            prev_score = scores[int(prev_action)]
                            if (best_score - prev_score) < HYST_ABS:
                                best = int(prev_action)

                        if best is not None:
                            surface_move[best] = SURFACE_STRENGTH

        bias += surface_move
        components["surface_move"] = surface_move

        # ────────────────────────────────────────────────────
        # 2) STAGNATION: застрял на поверхности — вызвать detach
        # Ситуация: агент на поверхности, alignment нормальный,
        # но за 5 шагов distance не уменьшилась (застрял у ребра, в вогнутости)
        # Принцип: если tangential ползание не помогает — detach перелетит
        # ────────────────────────────────────────────────────
        stagnation_override = np.zeros(self.num_actions, dtype=float)
        if on_object > 0.5 and alignment >= DETACH_ALIGN_THR:
            if len(self._episode_transitions) >= 5:
                recent_dists = [
                    float(tr["state"][13])
                    for tr in self._episode_transitions[-5:]
                ]
                dist_reduction = recent_dists[0] - recent_dists[-1]
                if dist_reduction < self.action_space.surface_step * 0.5:
                    stagnation_override[self.action_space.IDX_DETACH] += 3.0
                    for idx in range(8):
                        stagnation_override[idx] -= 2.0
        bias += stagnation_override
        components["stagnation_override"] = stagnation_override

        # ────────────────────────────────────────────────────
        # 3) DETACH: цель недоступна по поверхности — оторваться и перелететь
        # Два типа:
        #   detach_goal: цель "за" поверхностью (alignment < порог или
        #     normal_dist >> tangential_dist). Отлёт по нормали → полёт к цели.
        #   detach_edge: цель через тонкую стенку (нормали агента и цели
        #     противоположны). Облёт грани вверх → разворот к цели.
        # Ограничения: max 1 detach за 5 шагов, не рядом с целью
        # ────────────────────────────────────────────────────
        detach = np.zeros(self.num_actions, dtype=float)
        if on_object > 0.5:
            need_detach = False
            need_edge_detach = False

            n_world = sensor_data.get("point_normal", None)
            goal_normal = sensor_data.get("goal_normal", None)

            if n_world is not None:
                n_world = np.asarray(n_world, dtype=float)
                n_len = float(np.linalg.norm(n_world))
                if n_len > eps:
                    n_hat = n_world / n_len
                    e_world = rot.apply(local_pos_error)
                    e_t = e_world - np.dot(e_world, n_hat) * n_hat
                    tangential_dist = float(np.linalg.norm(e_t))
                    normal_dist = abs(float(np.dot(e_world, n_hat)))

                    if normal_dist > tangential_dist * 2.0 and distance > 3.0 * self.action_space.surface_step:
                        need_detach = True

                    if (alignment < DETACH_ALIGN_THR
                            and goal_normal is not None):
                        goal_n = np.array(goal_normal, dtype=float)
                        goal_n /= (np.linalg.norm(goal_n) + 1e-12)
                        normals_dot = float(np.dot(n_hat, goal_n))

                        if normals_dot < -0.5 and tangential_dist > normal_dist * 3.0:
                            need_edge_detach = True
                            need_detach = False

            if alignment < DETACH_ALIGN_THR and not need_edge_detach:
                need_detach = True

            if need_detach or need_edge_detach:
                recent_detach_count = sum(
                    1 for tr in self._episode_transitions[-5:]
                    if tr["action"] in (self.action_space.IDX_DETACH, self.action_space.IDX_DETACH_EDGE)
                )
                last_was_detach = self._last_action in (self.action_space.IDX_DETACH, self.action_space.IDX_DETACH_EDGE)

                if recent_detach_count < 1 and not last_was_detach and not close_to_goal:
                    if need_edge_detach:
                        detach[self.action_space.IDX_DETACH_EDGE] += 5.0
                    else:
                        detach[self.action_space.IDX_DETACH] += 5.0
                    for idx in range(8):
                        detach[idx] -= 2.0
                    detach[self.action_space.IDX_FREE_FORWARD] -= 2.0
                    detach[self.action_space.IDX_FREE_BACKWARD] -= 2.0
                    detach[self.action_space.IDX_LOOK_UP] -= 1.0
                    detach[self.action_space.IDX_LOOK_DOWN] -= 1.0
        bias += detach
        components["detach"] = detach

        # ────────────────────────────────────────────────────
        # 4) STEER IN AIR: навигация в воздухе к цели
        # Ситуация: агент не на поверхности (после detach или соскользнул)
        # Принцип: вычисляем yaw и pitch к позиции цели в локальной СК,
        # поворачиваемся последовательно (yaw → pitch → forward)
        # ────────────────────────────────────────────────────
        steer = np.zeros(self.num_actions, dtype=float)
        if on_object <= 0.5:
            goal_dir_local = local_pos_error.copy()
            goal_dist = np.linalg.norm(goal_dir_local)
            if goal_dist > 1e-8:
                goal_dir_local /= goal_dist

            gx, gy, gz = goal_dir_local

            yaw_to_goal = np.degrees(np.arctan2(-gx, -gz))
            horiz_dist = np.sqrt(gx**2 + gz**2)
            pitch_to_goal = np.degrees(np.arctan2(gy, horiz_dist))

            yaw_thr = 5.0
            pitch_thr = 5.0

            if abs(yaw_to_goal) > yaw_thr:
                if yaw_to_goal > 0:
                    steer[self.action_space.IDX_TURN_LEFT] += 2.0
                else:
                    steer[self.action_space.IDX_TURN_RIGHT] += 2.0
            elif abs(pitch_to_goal) > pitch_thr:
                if pitch_to_goal > 0:
                    steer[self.action_space.IDX_LOOK_UP] += 2.0
                else:
                    steer[self.action_space.IDX_LOOK_DOWN] += 2.0
            else:
                steer[self.action_space.IDX_FREE_FORWARD] += 2.0

            steer[self.action_space.IDX_FREE_BACKWARD] -= 0.5

        bias += steer
        components["steer_in_air"] = steer

        # ────────────────────────────────────────────────────
        # 5) DAMP FREE: на поверхности подавить free_forward/backward
        # Ситуация: агент на поверхности, цель доступна по поверхности
        # Принцип: tangential ползание эффективнее чем free_forward
        # на поверхности, free_forward может вызвать коллизию со стенкой
        # ────────────────────────────────────────────────────
        damp_free = np.zeros(self.num_actions, dtype=float)
        if on_object > 0.5:
            damp_free[self.action_space.IDX_FREE_FORWARD] -= 3.0
            damp_free[self.action_space.IDX_FREE_BACKWARD] -= 3.0
        bias += damp_free
        components["damp_free_on_surface"] = damp_free

        return bias, components
    
    # ══════════════════════════════════════════════════════════
    # COORDINATE TRANSFORMS
    # ══════════════════════════════════════════════════════════
    def _world_to_local(
        self,
        vector_world: np.ndarray,
        pose: np.ndarray,
    ) -> np.ndarray:
        """Transform vector from world frame to agent's local frame.

        Uses inverse of agent's rotation to convert world-frame
        vectors into the agent's reference frame.

        Args:
            vector_world: 3D vector in world coordinates.
            pose: Agent pose [x, y, z, [pitch,yaw,roll].

        Returns:
            3D vector in agent's local coordinates.
        """
        rotation = R.from_euler("xyz", pose[3:6], degrees=True)
        return rotation.inv().apply(vector_world)

    @staticmethod
    def _normalize_angles(angles: np.ndarray) -> np.ndarray:
        """Normalize angles to [-π, π] range.

        Handles wrap-around: e.g. goal_yaw=350°, current_yaw=10°
        → error should be -20°, not +340°.

        Args:
            angles: Array of angles in radians.

        Returns:
            Angles wrapped to [-π, π].
        """
        return (angles + np.pi) % (2.0 * np.pi) - np.pi

    # ══════════════════════════════════════════════════════════
    # MATH UTILITIES
    # ══════════════════════════════════════════════════════════
    @staticmethod
    def _normalize_angles_deg(angles_deg: np.ndarray) -> np.ndarray:
        return (angles_deg + 180.0) % 360.0 - 180.0

    @staticmethod
    def _normalize_values(values: np.ndarray) -> np.ndarray:
        """Normalize array to [-1, +1] range for fair blending.

        Maps min → -1, max → +1. Returns zeros if all values equal.

        Args:
            values: Array of any shape.

        Returns:
            Normalized array same shape as input.
        """
        v_min = np.min(values)
        v_max = np.max(values)
        v_range = v_max - v_min

        if v_range < 1e-8:
            return np.zeros_like(values)

        return 2.0 * (values - v_min) / v_range - 1.0

    @staticmethod
    def _softmax_sample(values: np.ndarray, temperature: float) -> int:
        """Sample action index from softmax distribution.

        Args:
            values: Preference scores for each action.
            temperature: Controls randomness.
                High (>1): more uniform (exploratory).
                Low (<0.5): more greedy (exploitative).

        Returns:
            Sampled action index.
        """
        v = values / temperature
        v = v - np.max(v)  # numerical stability
        exp_v = np.exp(v)
        probs = exp_v / exp_v.sum()
        return int(np.random.choice(len(values), p=probs))

    # ══════════════════════════════════════════════════════════
    # EPISODE LIFECYCLE
    # ══════════════════════════════════════════════════════════

    def _on_episode_done(
        self,
        final_state: np.ndarray,
        termination_reason: Optional[str],
    ):
        """Handle episode completion.

        Logs statistics, updates counters, resets episode state.

        Args:
            final_state: Last state of the episode.
            termination_reason: Canonical reason from _compute_reward.
        """
        distance = np.linalg.norm(final_state[0:3])
        goal_reached = termination_reason == "goal_reached"
        start_distance = np.linalg.norm(self._current_goal[0:3] - self.start_pos)

        if goal_reached:
            self._total_goals_reached += 1

        # Determine termination reason
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

        self._termination_counts[reason_key] = self._termination_counts.get(reason_key, 0) + 1

        logger.debug(
            f"Episode {self._total_episodes} DONE: {reason}, "
            f"{self._steps} steps, "
            f"reward={self._episode_reward:.1f}, "
            f"final_dist={distance:.1f}mm, "
            f"epsilon={self.epsilon:.3f}, "
            f"success_rate="
            f"{self._total_goals_reached}/{self._total_episodes}"
        )

        # Reset episode state (keep learned Q-values)
        self._current_goal = None
        self._prev_state = None
        self._prev_sensor_data = None
        self._last_action = None
        self._prev_action = None
        self._steps = 0
        self._episode_reward = 0.0
        self._episode_transitions = []

# ══════════════════════════════════════════════════════════
    # DIAGNOSTICS
    # ══════════════════════════════════════════════════════════

    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics for monitoring.

        Returns:
            Dict with episode counts, success rate, Q-store stats.
        """
        success_rate = (
            self._total_goals_reached / max(self._total_episodes, 1)
        )
        episodes = max(self._total_episodes, 1)
        termination_rates = {
            k: float(v) / episodes for k, v in self._termination_counts.items()
        }

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
            # "q_store": self.q_store.get_stats(),
            "q_store_free": self.q_store_free.get_stats(),
            "q_store_surface": self.q_store_surface.get_stats(),
            "collision_stats": dict(self._collision_stats),
        }

        return stats

    # ══════════════════════════════════════════════════════════
    # PERSISTENCE
    # ══════════════════════════════════════════════════════════

    def save(self, dirpath: str):
        """Save controller state to directory.

        Saves:
            - Q-store (HNSW index + points)
            - Controller parameters (epsilon, counters)

        Args:
            dirpath: Directory path. Created if doesn't exist.
        """
        import os
        os.makedirs(dirpath, exist_ok=True)

        # Save Q-store (with native HNSW index for fast reload)
        self.q_store_free.save_with_index(os.path.join(dirpath, "q_store_free"))
        self.q_store_surface.save_with_index(os.path.join(dirpath, "q_store_surface"))

        # Сохраняем текущий epsilon
        controller_state = {
            "epsilon": self.epsilon,
            "total_episodes": self._total_episodes,
            "total_steps": self._total_steps,
            "total_goals_reached": self._total_goals_reached,
            "mode": self.mode,
            "config": self.config,
        }
        # Save controller state
        #controller_state = {
        #    "epsilon": self.epsilon,
        #    "total_episodes": self._total_episodes,
        #    "total_steps": self._total_steps,
        #    "total_goals_reached": self._total_goals_reached,
        #    "config": self.config,
        #}
        np.savez(
            os.path.join(dirpath, "controller_state.npz"),
            **{k: np.array(v) if not isinstance(v, dict) else np.array([0])
               for k, v in controller_state.items()},
        )

        # Save config separately as readable format
        import json
        with open(os.path.join(dirpath, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"Controller saved to {dirpath}")

    @classmethod
    def load(cls, dirpath: str, agent_id: str, config=None) -> "RLGoalApproachController":
        """Load controller state from directory.

        Args:
            dirpath: Directory with saved state.
            agent_id: Monty agent identifier.

        Returns:
            Restored controller with learned Q-values.
        """
        # Load config
        with open(os.path.join(dirpath, "config.json"), "r") as f:
            saved_config = json.load(f)
        cfg = {**saved_config, **(config or {})}

        # Create controller
        controller = cls(agent_id=agent_id, config=cfg)
        
        # Load Q-store (fast path with native index, fallback to rebuild)
        controller.q_store_free = HNSWStateStore.load_with_index(
            os.path.join(dirpath, "q_store_free"), extra_cfg=config
        )
        controller.q_store_surface = HNSWStateStore.load_with_index(
            os.path.join(dirpath, "q_store_surface"), extra_cfg=config
        )

        # Load controller state
        state_data = np.load(
            os.path.join(dirpath, "controller_state.npz"),
            allow_pickle=False,
        )
        loaded_epsilon = float(state_data["epsilon"])
        loaded_total_episodes = int(state_data["total_episodes"])
        loaded_total_steps = int(state_data["total_steps"])
        loaded_total_goals_reached = int(state_data["total_goals_reached"])

        logger.info(
            f"Controller loaded from {dirpath}: "
            f"{loaded_total_episodes} loaded_total_episodes, "
            f"{loaded_total_steps} loaded_total_steps, "
            f"{loaded_total_goals_reached} loaded_total_goals_reached, "
            f"loaded_epsilon={loaded_epsilon:.3f}, "
            f"loaded Q-store={controller.q_store_surface.get_stats()['num_points']} points"
            f"loaded Q-store={controller.q_store_free.get_stats()['num_points']} points"
        )

        return controller
