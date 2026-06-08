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
        self.q_store = HNSWStateStore(
            config=self.config
        )

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

        logger.info(
            f"RLGoalApproachController initialized: "
            f"{self.num_actions} actions, "
            f"state_dim={self.config['state_dim']}, "
            f"epsilon={self.epsilon}"
        )

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

    def set_new_goal(self, goal_pose: np.ndarray):
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

        logger.debug(
            f"New goal set (episode {self._total_episodes}): "
            f"pos={goal_pose[:3]}, rot={np.degrees(goal_pose[3:])}°"
        )

    def step(
        self,
        current_pose: np.ndarray,
        sensor_data: Dict[str, Any],
    ) -> Optional[Action]:
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
            return None

        self._steps += 1
        self._total_steps += 1

        # Build state
        state = self._compute_state(current_pose, sensor_data)

        # Detect collision
        collision = self._detect_collision(sensor_data)

        # Learn from previous step
        done = False
        
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
                next_q = self.q_store.get_q_values(state)
                td_target = reward + self.gamma * np.max(next_q)

            #self.q_store.update_q_value(
            #    self._prev_state, self._last_action, td_target, self.alpha
            #)
            # Обучение: обновляем Q-values
            self.q_store.update_q_value(
                self._prev_state,
                self._last_action,
                td_target,
                self._get_learning_rate(),
            )

            if done:
                if self.is_training and termination_reason == "goal_reached":
                    self._apply_success_backup_updates()
                    self.success_trails = self._episode_transitions.copy()
                self._on_episode_done(state, termination_reason)
                return None

        # Choose action
        action_index = self._choose_action(state)

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

        return action

    @property
    def is_active(self) -> bool:
        """Whether controller has an active goal."""
        return self._current_goal is not None

    # ══════════════════════════════════════════════════════════
    # STATE COMPUTATION
    # ══════════════════════════════════════════════════════════
    def _compute_state(self, current_pose, sensor_data):
        """Build 13D state vector from pose and sensor data.

        All spatial quantities are converted to agent's LOCAL coordinate
        frame. This is critical because:
            - MoveTangentially(90°) always means "right relative to me"
            - State must be consistent with action semantics
            - Same relative situation → same state → same Q-values
              regardless of absolute position/orientation

        Args:
            current_pose: [x, y, z, roll, pitch, yaw]
            sensor_data: Sensor observations dict.

        Returns:
            State vector [13D].
        """
        goal = self._current_goal

        # ── Position error (local frame, мм) ──
        pos_error_world = goal[:3] - current_pose[:3]
        local_pos_error = self._world_to_local(pos_error_world, current_pose)

        # ── Rotation error (градусы) ──
        rot_error_rad = self._normalize_angles(goal[3:] - current_pose[3:])
        rot_error_deg = np.degrees(rot_error_rad)

        # ── Surface normal (local frame) ──
        raw_normal = sensor_data.get("point_normal", None)
        if raw_normal is not None:
            local_normal = self._world_to_local(
                np.array(raw_normal), current_pose
            )
        else:
            local_normal = np.zeros(3)

        # ── On object ──
        on_object = float(sensor_data.get("on_object", False))

        # ── Distance ──
        distance = np.linalg.norm(local_pos_error)

        # ── Alignment ──
        normal_len = np.linalg.norm(local_normal)
        if distance > 1e-8 and normal_len > 1e-8:
            goal_dir = local_pos_error / distance
            alignment = np.dot(goal_dir, local_normal)
        else:
            alignment = 0.0

        # ── Depth ──
        depth = sensor_data.get(
            "depth", self.config["max_sensor_range"]
        )
        norm_depth = min(depth / self.config["max_sensor_range"], 1.0)

        # ── Assemble (13D) ──
        state = np.concatenate([
            local_pos_error,        # [0:3]   3D  мм
            rot_error_deg,          # [3:6]   3D  градусы
            local_normal,           # [6:9]   3D  unitless
            [on_object],            # [9]     1D  binary
            [alignment],            # [10]    1D  [-1, +1]
            [distance],             # [11]    1D  мм
            [norm_depth],           # [12]    1D  [0, 1]
        ])

        return state  # 13D

# ══════════════════════════════════════════════════════════
    # COLLISION DETECTION
    # ══════════════════════════════════════════════════════════
    def _detect_collision(self, sensor_data):
        """Detect sensory collisions from depth/normal data.

        Three types of sensory collision:
            'inside_object': depth too small (clipped through surface)
            'lost_object': was on object, now lost contact
            'passed_through': surface normal flipped 180°

        These are detected from sensor data, not Habitat physics.
        Monty uses depth camera rendering, not rigid body collision.

        Args:
            sensor_data: Current sensor observations.

        Returns:
            Collision type string, or None if no collision.
        """

        # Объединяем в один тип: "surface_violation"
        
        # Проверка 1: внутри объекта
        depth = sensor_data.get("depth", self.config["max_sensor_range"])
        if depth < self.config["min_valid_depth"]:
            return "surface_violation"
        
        # Проверка 2: пролетели сквозь
        if (self._prev_sensor_data is not None
                and self._prev_sensor_data.get("on_object", False)
                and sensor_data.get("on_object", False)):
            
            prev_normal = self._prev_sensor_data.get("point_normal")
            curr_normal = sensor_data.get("point_normal")
            
            if prev_normal is not None and curr_normal is not None:
                dot = np.dot(np.array(prev_normal), np.array(curr_normal))
                if dot < self.config["normal_flip_threshold"]:
                    return "surface_violation"
        
        # Проверка 3: потеряли объект
        if self._prev_sensor_data is not None:
            was_on = self._prev_sensor_data.get("on_object", False)
            now_on = sensor_data.get("on_object", False)
            if was_on and not now_on:
                return "lost_object"
        
        return None

    # ══════════════════════════════════════════════════════════
    # REWARD
    # ══════════════════════════════════════════════════════════
    
    def _compute_reward(self, state, prev_state, action, collision):
        """Compute reward and done flag.

        Dense reward based on:
            - Progress toward goal (main signal)
            - Goal reached bonus
            - Collision penalties
            - Efficiency incentives (step penalty, anti-oscillation)

        Args:
            state: Current state [18D].
            prev_state: Previous state [18D].
            action: Action taken.
            collision: Collision type or None.

        Returns:
            Tuple of (reward, done, termination_reason).
            termination_reason ∈ {
                "goal_reached",
                "collision_surface_violation",
                "timeout",
                None,
            }
        """
        cfg = self.config
        reward = 0.0
        done = False
        termination_reason = None
        
        distance = state[11]       # обновлённый индекс (13D state)
        prev_distance = prev_state[11]
        on_object = state[9]
        prev_alignment = prev_state[10]
        prev_on_object = prev_state[9]
        
        surface_step = self.action_space.surface_step
        
        # ═══ 1. Progress toward goal ═══
        # Нормализован по step size: всегда ~5.0 за идеальный шаг.
        # В detour-режиме (цель через поверхность) клипуем отрицательный прогресс,
        # чтобы не наказывать за вынужденный обход между гранями.
        progress_raw = prev_distance - distance
        progress = progress_raw
        detour_mode = (
            prev_alignment < cfg["detour_alignment_threshold"]
            and (prev_on_object > 0.5 or collision == "lost_object")
        )
        if detour_mode and progress_raw < 0.0:
            min_progress = -surface_step * cfg["detour_negative_progress_clip_steps"]
            progress = max(progress_raw, min_progress)
        reward += progress / surface_step * cfg["reward_progress"]
        
        # ═══ 2. Goal reached ═══
        if distance < cfg["goal_threshold"]:
            reward += cfg["reward_goal_reached"]
            done = True
            termination_reason = "goal_reached"
        
        # ═══ 3. Step penalty ═══
        reward += cfg["reward_step_penalty"]
        
        # ═══ 4. Collisions ═══
        # Reward: один штраф вместо двух
        if collision == "surface_violation":
            reward += cfg["reward_surface_violation"]  # -5.0
            done = True
            termination_reason = "collision_surface_violation"
        elif collision == "lost_object":
            if prev_alignment < -0.3 and progress > 0:
                # Smart detach: цель через поверхность, приблизились
                reward += cfg["reward_smart_detach"]   # БОНУС за умное решение
            #elif progress > 0:
            #    # Приблизились, но причина неясна
            #    reward += cfg[""]
            else:
                # Улетели от цели
                reward += cfg["reward_drifted_away"]
        
        # ═══ 5. Near goal on surface ═══
        near_radius = surface_step * 3  # 3 шага от цели
        if distance < near_radius and on_object > 0.5:
            reward += cfg["reward_near_goal_on_surface"]
        
        # ═══ 6. Oscillation (safety net) ═══
        # Пока убирвем, соишком грубо
        #if (self._prev_action is not None
        #        and self.action_space.are_opposite(action, self._prev_action)):
        #    reward += cfg["reward_oscillation"]
        
        # ═══ 7. Timeout ═══
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
            self.q_store.update_q_value(
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
    
    def _choose_action(self, state):
        q_values = self.q_store.get_q_values(state)
        heuristic = self._compute_heuristic_bias(state)
        
        eps = self._get_current_epsilon()
        
        q_norm = self._normalize_values(q_values)
        h_norm = self._normalize_values(heuristic)
        
        combined = (1 - eps) * q_norm + eps * h_norm
        
        # Чисто случайное действие
        if np.random.random() < eps * 0.1:
            return np.random.randint(self.num_actions)
        
        temperature = max(0.1, eps)
        return self._softmax_sample(combined, temperature)

    # ══════════════════════════════════════════════════════════
    # HEURISTIC BIAS
    # ══════════════════════════════════════════════════════════
    def _compute_heuristic_bias(self, state):
        """Эвристические предпочтения действий.
        
        Все вычисления из геометрии и action space параметров.
        Нет магических чисел — всё выводится.
        
        Args:
            state: [13D] — обновлённый state vector
            
        Returns:
            bias: [18] — предпочтение для каждого действия
                положительное = предпочтительно
                отрицательное = нежелательно
                ноль = нейтрально
        """
        bias = np.zeros(self.num_actions)
        # Пока не работают просто возвращаем нули
        # return bias
        
        # Распаковка state (13D)
        local_pos_error = state[0:3]
        # rot_error = state[3:6]  # не используем в эвристике
        local_normal = state[6:9]
        on_object = state[9]
        alignment = state[10]
        distance = state[11]
        norm_depth = state[12]
        
        # Параметры action space
        surface_step = self.action_space.surface_step
        free_step = self.action_space.free_step
        normal_len = np.linalg.norm(local_normal)
        
        # ═══════════════════════════════════════════════════
        # Эвристика 1: ДВИГАЙСЯ К ЦЕЛИ
        # Источник: tangent_movement()
        #
        # Проецируем направление к цели на касательную
        # плоскость и находим ближайшее surface действие.
        # Чистая геометрия, нет порогов.
        # ═══════════════════════════════════════════════════
        
        goal_dir = local_pos_error / (distance + 1e-8)
        
        if on_object > 0.5 and normal_len > 1e-8:
            # На поверхности: проецируем на касательную плоскость
            n_hat = local_normal / normal_len
            tangent = goal_dir - np.dot(goal_dir, n_hat) * n_hat
            tangent_len = np.linalg.norm(tangent)
            if tangent_len > 1e-8:
                tangent = tangent / tangent_len
            else:
                # Цель точно по нормали — tangent неопределён
                tangent = goal_dir
        else:
            # В воздухе: просто направление к цели
            tangent = goal_dir
        
        # Угол tangent → cosine similarity для каждого surface действия
        tangent_angle = np.degrees(np.arctan2(tangent[1], tangent[0])) % 360
        bias += self.action_space.surface_direction_similarity(tangent_angle)
        
        # ═══════════════════════════════════════════════════
        # Эвристика 2: ДАЛЕКО → ЛЕТИ, БЛИЗКО → ПОЛЗИ
        # Источник: surface_crawl vs free movement
        #
        # Порог выводится из step sizes:
        # crossover = точка где лететь и ползти одинаково
        # ═══════════════════════════════════════════════════
        
        steps_needed = distance / surface_step
        crossover = 2.0 + free_step / surface_step
        # crossover при surface=5, free=10:
        #   2.0 + 10/5 = 4.0 шага
        #   Меньше 4 шагов → ползти
        #   Больше 4 шагов → лететь
        
        # Плавный переход через tanh
        fly_preference = np.tanh(steps_needed / crossover - 1.0)
        # steps=1  → tanh(-0.75) = -0.64 → ползти
        # steps=4  → tanh(0.0)   =  0.0  → нейтрально
        # steps=8  → tanh(1.0)   =  0.76 → лететь
        # steps=20 → tanh(4.0)   =  1.0  → точно лететь
        
        if fly_preference > 0:
            # Далеко: предпочитаем свободный полёт
            bias[self.action_space.IDX_FREE_FORWARD] += fly_preference
            bias[self.action_space.IDX_FREE_BACKWARD] -= 0.5
        else:
            # Близко: усиливаем surface, ослабляем free
            surface_mask = self.action_space.get_category_mask("surface")
            free_mask = self.action_space.get_category_mask("free")
            bias[surface_mask] *= (1.0 + abs(fly_preference))
            bias[free_mask] -= abs(fly_preference)  # penalize free movement near goal
        
        # ═══════════════════════════════════════════════════
        # Эвристика 3: ЦЕЛЬ ЧЕРЕЗ ПОВЕРХНОСТЬ → ОТОРВИСЬ
        # Источник: геометрия (alignment < 0)
        #
        # Если цель по ту сторону объекта, нужно
        # подняться над поверхностью. Сила пропорциональна
        # тому насколько "глубоко" цель за поверхностью.
        # ═══════════════════════════════════════════════════
        
        if alignment < 0 and on_object > 0.5:
            # alignment = -1.0 → цель прямо за поверхностью → сильный отрыв
            # alignment = -0.1 → цель почти вдоль поверхности → слабый отрыв
            detach_urgency = abs(alignment)  # [0, 1]
            bias[self.action_space.IDX_LOOK_UP] += detach_urgency
        
        # ═══════════════════════════════════════════════════
        # Эвристика 4: ОРИЕНТАЦИЯ СЕНСОРА
        # Источник: orient_to_surface()
        #
        # Сенсор должен смотреть на поверхность.
        # Чем сильнее отклонение, тем больше bias.
        # ═══════════════════════════════════════════════════
        
        sensor_forward = np.array([0.0, 0.0, -1.0])
        if on_object > 0.5 and normal_len > 1e-8:
            sensor_alignment = np.dot(sensor_forward, local_normal)
            # sensor_alignment = 1.0 → идеально (смотрит на поверхность)
            # sensor_alignment = 0.0 → перпендикулярно
            # sensor_alignment = -1.0 → отвернулся
            
            orient_need = max(0.0, 1.0 - sensor_alignment)
            # × 0.5: ориентация менее приоритетна чем движение
            # Если мы в режиме "отрыва", ориентация менее важна
            if alignment < 0:
                orient_need *= 0.3  # снижаем приоритет стабилизации
            else:
                orient_need *= 0.5  # обычный приоритет
            
            if local_normal[2] < 0:
                bias[self.action_space.IDX_LOOK_UP] += orient_need
            else:
                bias[self.action_space.IDX_LOOK_DOWN] += orient_need
        
        return bias

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
            pose: Agent pose [x, y, z, roll, pitch, yaw].

        Returns:
            3D vector in agent's local coordinates.
        """
        rotation = R.from_euler("xyz", pose[3:6])
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

        if goal_reached:
            self._total_goals_reached += 1

        # Determine termination reason
        if goal_reached:
            reason = "GOAL_REACHED!!!"
            reason_key = "goal_reached"
            logger.info(
                f"Episode {self._total_episodes} DONE: {reason}, "
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
            "q_store": self.q_store.get_stats(),
        }

        return stats

    def explain_action(
        self,
        current_pose: np.ndarray,
        sensor_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Explain WHY the controller would choose a particular action.

        Useful for debugging and understanding learned behavior.
        Does NOT execute the action or update any state.

        Args:
            current_pose: Current agent pose.
            sensor_data: Current sensor observations.

        Returns:
            Dict with state, Q-values, heuristic bias, combined scores,
            chosen action, and nearest neighbor info.
        """
        if self._current_goal is None:
            return {"error": "no active goal"}

        state = self._compute_state(current_pose, sensor_data)
        q_values = self.q_store.get_q_values(state)
        heuristic = self._compute_heuristic_bias(state)

        q_norm = self._normalize_values(q_values)
        h_norm = self._normalize_values(heuristic)
        combined = (1.0 - self.epsilon) * q_norm + self.epsilon * h_norm

        temperature = max(0.1, self.epsilon)
        v = combined / temperature
        v = v - np.max(v)
        exp_v = np.exp(v)
        probs = exp_v / exp_v.sum()

        best_action = int(np.argmax(combined))
        neighbors = self.q_store.get_nearest_points_info(state, k=5)

        collision = self._detect_collision(sensor_data)
        progress_raw_since_prev = None
        progress_used_since_prev = None
        progress_clipped_since_prev = None
        detour_mode_active = False
        if self._prev_state is not None:
            prev_distance = float(self._prev_state[11])
            distance = float(state[11])
            prev_alignment = float(self._prev_state[10])
            prev_on_object = float(self._prev_state[9])
            progress_raw_since_prev = prev_distance - distance
            progress_used_since_prev = progress_raw_since_prev
            detour_mode_active = (
                prev_alignment < self.config["detour_alignment_threshold"]
                and (prev_on_object > 0.5 or collision == "lost_object")
            )
            if detour_mode_active and progress_raw_since_prev < 0.0:
                min_progress = (
                    -self.action_space.surface_step
                    * self.config["detour_negative_progress_clip_steps"]
                )
                progress_used_since_prev = max(progress_raw_since_prev, min_progress)
            progress_clipped_since_prev = (
                progress_used_since_prev != progress_raw_since_prev
            )

        return {
            "state": {
                "local_pos_error": state[0:3].tolist(),
                "rot_error": state[3:6].tolist(),
                "local_normal": state[6:9].tolist(),
                "on_object": float(state[9]),
                "alignment": float(state[10]),
                "distance": float(state[11]),
                "norm_depth": float(state[12]),
            },
            "q_values": {
                self.action_space.get_info(i).name: float(q_values[i])
                for i in range(self.num_actions)
            },
            "heuristic_bias": {
                self.action_space.get_info(i).name: float(heuristic[i])
                for i in range(self.num_actions)
            },
            "action_probabilities": {
                self.action_space.get_info(i).name: float(probs[i])
                for i in range(self.num_actions)
            },
            "best_action": {
                "index": best_action,
                "name": self.action_space.get_info(best_action).name,
                "q_value": float(q_values[best_action]),
                "heuristic": float(heuristic[best_action]),
                "probability": float(probs[best_action]),
            },
            "epsilon": self.epsilon,
            "blend": f"{(1-self.epsilon)*100:.0f}% Q + {self.epsilon*100:.0f}% heuristic",
            "reward_diagnostics": {
                "detour_mode_active": detour_mode_active,
                "progress_raw_since_prev": progress_raw_since_prev,
                "progress_used_since_prev": progress_used_since_prev,
                "progress_clipped_since_prev": progress_clipped_since_prev,
                "collision_now": collision,
            },
            "nearest_neighbors": neighbors,
        }

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
        self.q_store.save_with_index(os.path.join(dirpath, "q_store"))

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
        controller.q_store = HNSWStateStore.load_with_index(
            os.path.join(dirpath, "q_store"), extra_cfg=config
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
            f"loaded Q-store={controller.q_store.get_stats()['num_points']} points"
        )

        return controller
