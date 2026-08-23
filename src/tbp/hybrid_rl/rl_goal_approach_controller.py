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
from .strategic_sac import StrategicSAC
from .experience_extractor import ExperienceExtractor

logger = logging.getLogger(__name__)


class RLGoalApproachController:
    """Q-learning controller that moves agent toward goal pose.

    State vector (18D):
        local_pos_error  [3D]: direction to goal in agent's local frame
        rot_error        [3D]: orientation error (normalized angles)
        local_normal     [3D]: surface normal in agent's local frame
        k1               [1D]: principal curvature (max absolute)
        k2               [1D]: principal curvature (min absolute)
        on_object        [1D]: whether sensor sees object surface
        alignment        [1D]: dot(goal_direction, point_normal)
        distance         [1D]: Euclidean distance to goal
        norm_depth       [1D]: normalized depth to nearest surface
        goal_normal_local[3D]: goal surface normal in agent's local frame
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

        # HNSW Q-store with per-store feature weights
        free_weights = self.config.get(
            "feature_weights_free", None
        )
        surface_weights = self.config.get(
            "feature_weights_surface", None
        )

        self.q_store_free = HNSWStateStore(
            config={
                **self.config,
                "feature_weights": free_weights,
            },
            name="free",
        )
        self.q_store_surface = HNSWStateStore(
            config={
                **self.config,
                "feature_weights": surface_weights,
            },
            name="surface",
        )

        # Strategic Q-stores (2 actions: stay=0, switch=1)
        strategic_detach_config = {
            **self.config,
            "state_dim": 5,
            "num_actions": 2,
            "max_points": self.config.get("transition_max_points", 10000),
            "k_neighbors": self.config.get("transition_k_neighbors", 5),
            "insert_threshold": self.config.get(
                "transition_insert_threshold", 0.5
            ),
        }
        self.strategic_detach = HNSWStateStore(
            config=strategic_detach_config, name="strategic_detach"
        )

        strategic_direction_config = {
            **self.config,
            "state_dim": 5,
            "num_actions": 2,
            "max_points": self.config.get(
                "transition_max_points", 10000
            ),
            "k_neighbors": self.config.get(
                "transition_k_neighbors", 5
            ),
            "insert_threshold": self.config.get(
                "transition_insert_threshold", 0.5
            ) * 0.5,
        }
        self.strategic_direction = HNSWStateStore(
            config=strategic_direction_config, name="strategic_direction"
        )

        # Strategic epsilon
        self.strategic_epsilon = float(
            self.config.get("strategic_epsilon_start", 1.0)
        )
        self.strategic_epsilon_min = float(
            self.config.get("strategic_epsilon_min", 0.3)
        )
        self.strategic_epsilon_decay = None

        # Strategic SAC (optional, created when available)
        self.strategic_sac: Optional[StrategicSAC] = None
        self._strategic_sac_pending: List[Dict[str, Any]] = []

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
            warmup_episodes = int(self.config.get("warmup_episodes", 0))
            
            # Считаем уровни из конфига
            curriculum_levels = self.config.get("curriculum_levels", [])
            num_levels = max(1, len(curriculum_levels))
            total_warmup = warmup_episodes * num_levels
            
            effective_episodes = max(1, total_episodes - total_warmup)
            
            eps_start = float(self.config.get("epsilon_start", self.epsilon))
            eps_min = float(self.config.get("epsilon_min", self.epsilon_min))
            if eps_start > eps_min > 0.0:
                self.epsilon_decay = (eps_min / eps_start) ** (1.0 / effective_episodes)

            s_start = float(self.config.get("strategic_epsilon_start", 1.0))
            s_min = float(self.config.get("strategic_epsilon_min", 0.3))
            if s_start > s_min > 0.0:
                self.strategic_epsilon_decay = (s_min / s_start) ** (1.0 / effective_episodes)
                
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
        self._current_max_extent: Optional[float] = None
        self._prev_phase: Optional[str] = None
        self._pending_strategic_detach: List[Dict[str, Any]] = []
        self._pending_strategic_direction: Optional[Dict[str, Any]] = None
        self._path_clear_streak: int = 0

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
        self._cached_orbit_direction = None
        self._orbit_direction_age = 0
        self._current_phase = "CRAWL_TO_GOAL"
        self._current_source = None
        self._strategic_stats = {
            # Detach decisions
            "detach_memory_triggered": 0,
            "detach_memory_suppressed": 0,
            "detach_heuristic_fallback": 0,
            "detach_total": 0,
            # Direction decisions
            "direction_memory_to_goal": 0,
            "direction_memory_keep_edge": 0,
            "direction_heuristic_fallback": 0,
            # Outcomes
            "detach_led_to_success": 0,
            "detach_led_to_collision": 0,
            "detach_led_to_timeout": 0,
            # Phase counts per episode (reset each episode)
            "phase_counts": {},
        }
        self._episode_phase_counts: Dict[str, int] = {}

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
        if self.mode in ("train", "train_adapt_epsilon", "adaptive"):
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
        self._current_max_extent = None
        self._last_detach_sub_steps = 1
        self._consecutive_detach_count = 0
        self._flyby_count = 0
        self._fly_direction_age = 0
        self._cached_fly_direction = None
        self._cached_orbit_direction = None
        self._orbit_direction_age = 0
        self._current_phase = "CRAWL_TO_GOAL"
        self._current_source = None
        self._prev_phase = None
        self._pending_strategic_detach = []
        self._pending_strategic_direction = None
        self._path_clear_streak = 0

        if self.is_training and self.mode != "adaptive":
            is_level_warmup = getattr(
                self, "_force_warmup", False
            )
            if not is_level_warmup:
                self.epsilon = max(
                    self.epsilon_min,
                    self.epsilon
                    * self.epsilon_decay,
                )
                if (
                    self.strategic_epsilon_decay
                    is not None
                ):
                    self.strategic_epsilon = max(
                        self.strategic_epsilon_min,
                        self.strategic_epsilon
                        * self.strategic_epsilon_decay,
                    )

        self._episode_phase_counts = {}
        self._strategic_sac_pending = []
        self._prev_subgoal_alignment = None
        self._prev_goal_alignment = None

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
        if sensor_data.get("on_object", False):
            if not sensor_data.get("same_side", True):
                cached = self._compute_detach_fly_direction(
                    current_pose, sensor_data
                )
                if cached is not None:
                    self._cached_fly_direction = cached

        collision = self._detect_collision(sensor_data)
        if collision:
            logger.debug(f"COLLISION: {collision} at step {self._steps}")

        done = False

        if self._prev_state is not None:
            prev_store = self._select_store(self._prev_state)
        next_store = self._select_store(state)

        if self._prev_state is not None:
            reward, done, termination_reason = self._compute_reward(
                state, self._prev_state, self._last_action, collision,
                sensor_data=sensor_data,
                prev_sensor_data=self._prev_sensor_data,
                current_pose=current_pose,
            )
            self._episode_reward += reward
            self._episode_transitions.append(
                {
                    "state": self._prev_state.copy(),
                    "action": int(self._last_action),
                    "reward": float(reward),
                    "path_blocked": sensor_data.get("path_blocked", False),
                    "same_side": sensor_data.get("same_side", True),
                     "movement_efficiency": self._compute_movement_efficiency(window=20),  # NEW
                }
            )
            if self.is_training:
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
            self.strategic_epsilon = self.config.get("strategic_eval_epsilon", 0.3)

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
        if abs(k1_raw) < abs(k2_raw):
            k1, k2 = k2_raw, k1_raw
        else:
            k1, k2 = k1_raw, k2_raw
        if on_object < 0.5:
            k1, k2 = 0.0, 0.0

        # Goal normal in agent's local frame
        raw_goal_normal = sensor_data.get("goal_normal", None)
        if raw_goal_normal is not None:
            goal_normal_local = self._world_to_local(
                np.array(raw_goal_normal), current_pose
            )
        else:
            goal_normal_local = np.zeros(3)

        # Path blocked
        path_blocked_val = float(
            sensor_data.get("path_blocked", False)
        )

        # Movement efficiency
        movement_efficiency = (
            self._compute_movement_efficiency(window=20)
        )
        # Projected goal direction on tangent plane
        if (
            on_object > 0.5
            and np.linalg.norm(local_normal) > 1e-8
            and distance > 1e-8
        ):
            # Goal direction in world frame
            rot_mat = R.from_euler(
                "xyz", current_pose[3:6],
                degrees=True,
            )
            goal_dir_world = rot_mat.apply(
                local_pos_error
            )
            normal_world = rot_mat.apply(
                local_normal
            )
            n_len = np.linalg.norm(normal_world)
            if n_len > 1e-8:
                n_hat = normal_world / n_len
                # Project onto tangent plane
                projected = (
                    goal_dir_world
                    - np.dot(goal_dir_world, n_hat)
                    * n_hat
                )
                proj_len = np.linalg.norm(projected)
                if proj_len > 1e-8:
                    projected /= proj_len
                # Back to local frame
                projected_local = (
                    rot_mat.inv().apply(projected)
                )
                # Take 2D (x, y components)
                projected_2d = projected_local[:2]
            else:
                projected_2d = np.zeros(2)
        else:
            # In air or no normal: use
            # local_pos_error direction
            if distance > 1e-8:
                projected_2d = (
                    local_pos_error[:2]
                    / distance
                )
            else:
                projected_2d = np.zeros(2)

        state = np.concatenate(
            [
                local_pos_error,      # [0:3]   3D
                rot_error_deg,        # [3:6]   3D
                local_normal,         # [6:9]   3D
                [k1],                 # [9]     1D
                [k2],                 # [10]    1D
                [on_object],          # [11]    1D
                [alignment],          # [12]    1D
                [distance],           # [13]    1D
                [norm_depth],         # [14]    1D
                goal_normal_local,    # [15:18] 3D
                [path_blocked_val],   # [18]    1D
                [movement_efficiency],# [19]    1D
                projected_2d,         # [20:22] 2D NEW
            ]
        )

        # Cache object extent for retrospective computations
        if self._current_max_extent is None:
            extents = sensor_data.get("object_extents", [84, 84, 84])
            self._current_max_extent = float(max(extents))

        # Track distance for flyby detection
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

    def _compute_detach_transition_state(
        self,
        state: np.ndarray,
        sensor_data: Dict[str, Any],
        movement_efficiency: float = 0.5,
    ) -> np.ndarray:
        """Compact state for detach decision (5D)."""
        agent_normal = state[6:9]
        goal_normal = state[15:18]

        an_len = np.linalg.norm(agent_normal)
        gn_len = np.linalg.norm(goal_normal)
        if an_len > 1e-8 and gn_len > 1e-8:
            normal_agreement = float(
                np.dot(agent_normal / an_len, goal_normal / gn_len)
            )
        else:
            normal_agreement = 0.0

        alignment = float(state[12])

        extents = sensor_data.get("object_extents", [84, 84, 84])
        max_extent = float(max(extents))
        norm_distance = float(state[13]) / max(max_extent, 1.0)

        path_blocked = float(
            sensor_data.get("path_blocked", False)
        )

        return np.array([
            normal_agreement,
            alignment,
            norm_distance,
            path_blocked,
            movement_efficiency,
        ], dtype=float)

    def _compute_direction_transition_state(
        self,
        state: np.ndarray,
        sensor_data: Dict[str, Any],
        current_pose: np.ndarray,
    ) -> np.ndarray:
        """Compact state for direction decision in air (5D).

        Features:
            lateral_deviation: how far off-axis the goal is (0=ahead, 1=side)
            alignment: dot(goal_dir, agent_normal)
            norm_distance: distance / object_extent
            angle_to_goal: dot(forward, goal_dir)
            path_blocked: is direct path to goal blocked
        """
        alignment = float(state[12])

        extents = sensor_data.get("object_extents", [84, 84, 84])
        max_extent = float(max(extents))
        norm_distance = float(state[13]) / max(max_extent, 1.0)

        rot = R.from_euler("xyz", current_pose[3:6], degrees=True)
        forward = rot.apply([0, 0, -1])
        goal_dir = self._current_goal[:3] - current_pose[:3]
        goal_dist = np.linalg.norm(goal_dir)
        if goal_dist > 1e-8:
            goal_dir /= goal_dist
        angle_to_goal = float(np.dot(forward, goal_dir))

        # Lateral deviation: how far goal is from forward axis
        # 0.0 = goal directly ahead, 1.0 = goal to the side/behind
        lateral_deviation = float(np.sqrt(max(0.0, 1.0 - angle_to_goal ** 2)))

        path_blocked = float(sensor_data.get("path_blocked", False))

        return np.array([
            lateral_deviation,
            alignment,
            norm_distance,
            angle_to_goal,
            path_blocked,
        ], dtype=float)

    def _compute_direction_state_from_full(
        self,
        full_state: np.ndarray,
        path_blocked: Optional[float] = None,
    ) -> np.ndarray:
        """Compute direction state from stored 18D state.

        Generates the same 5 features as
        _compute_direction_transition_state() but using
        surrogate computations from the stored full state.

        Args:
            full_state: Stored 18D state vector.
            path_blocked: Real path_blocked from transition dict.
                If None, uses heuristic fallback.

        Returns:
            5D direction state vector.
        """
        alignment = float(full_state[12])
        max_extent = getattr(self, "_current_max_extent", None) or 84.0
        norm_distance = float(full_state[13]) / max(max_extent, 1.0)

        local_pos = full_state[0:3]
        pos_len = np.linalg.norm(local_pos)

        # angle_to_goal surrogate: forward component of goal direction
        # in local frame. local_pos = goal - agent in agent frame,
        # forward = -Z, so dot(forward, goal_dir) ≈ -local_pos[2]/pos_len
        if pos_len > 1e-8:
            angle_to_goal = float(-local_pos[2] / pos_len)
        else:
            angle_to_goal = 1.0

        # lateral_deviation surrogate: how far goal is from forward axis
        # sqrt(local_pos[0]² + local_pos[1]²) / pos_len
        if pos_len > 1e-8:
            lateral_deviation = float(
                np.sqrt(local_pos[0] ** 2 + local_pos[1] ** 2) / pos_len
            )
        else:
            lateral_deviation = 0.0

        # path_blocked: use real value from transition if available
        if path_blocked is None:
            path_blocked = 1.0 if alignment < -0.3 else 0.0
        else:
            path_blocked = float(path_blocked)

        return np.array([
            lateral_deviation,
            alignment,
            norm_distance,
            angle_to_goal,
            path_blocked,
        ], dtype=float)

    def _compute_detach_state_from_full(
        self,
        full_state: np.ndarray,
        path_blocked: float = 1.0,
        movement_efficiency=0.5,
    ) -> np.ndarray:
        """Compute detach transition state from stored full state.

        Note: path_blocked not available from stored state,
        defaults to 1.0 (assume blocked — conservative).
        """
        agent_normal = full_state[6:9]
        goal_normal = full_state[15:18]

        an_len = np.linalg.norm(agent_normal)
        gn_len = np.linalg.norm(goal_normal)
        if an_len > 1e-8 and gn_len > 1e-8:
            normal_agreement = float(
                np.dot(agent_normal / an_len, goal_normal / gn_len)
            )
        else:
            normal_agreement = 0.0

        alignment = float(full_state[12])
        max_extent = getattr(self, "_current_max_extent", None) or 84.0
        norm_distance = float(full_state[13]) / max(max_extent, 1.0)

        return np.array([
            normal_agreement,
            alignment,
            norm_distance,
            path_blocked,
            movement_efficiency,
        ], dtype=float)
    
    def _compute_strategic_heuristic(
        self,
        current_phase: str,
        on_object: bool,
    ) -> np.ndarray:
        """Heuristic bias for strategic decision [stay, switch].

        On surface: stay=crawl, switch=detach
        In air: stay=FLY_TO_EDGE, switch=FLY_TO_GOAL

        Args:
            current_phase: Current phase from _determine_phase.
            on_object: Whether agent is on surface.

        Returns:
            Array [2]: [H_stay, H_switch].
        """
        h = np.zeros(2, dtype=float)

        if on_object:
            if current_phase == "DETACH_NEEDED":
                h[1] += 5.0
                h[0] -= 2.0
            elif current_phase == "CRAWL_TO_GOAL":
                h[0] += 3.0
                h[1] -= 3.0
            elif current_phase == "CRAWL_TO_EDGE":
                h[0] += 3.0   # stay — crawling to edge is productive
                h[1] -= 3.0   # don't detach while making progress to edge
        else:
            if current_phase == "FLY_TO_GOAL" or current_phase == "LAND":
                h[1] += 3.0
                h[0] -= 1.0
            elif current_phase == "FLY_TO_EDGE":
                h[0] += 3.0
                h[1] -= 1.0

        return h
    
    def _compute_strategic_state(
        self,
        state: np.ndarray,
        sensor_data: Dict[str, Any],
    ) -> np.ndarray:
        """Compact state for Strategic SAC (6D).

        Used by Strategic SAC to decide phase transitions.
        Same features visible to both TransitionMemory and
        Strategic SAC, enabling fair arbitration.

        Features:
            normal_agreement: dot(agent_normal, goal_normal)
            alignment: dot(goal_dir, agent_normal)
            norm_distance: distance / object_extent
            path_blocked: is direct path blocked (0/1)
            on_object: on surface or in air (0/1)
            norm_depth: normalized depth to surface

        Args:
            state: Full 18D state vector.
            sensor_data: Current sensor readings.

        Returns:
            6D strategic state vector.
        """
        agent_normal = state[6:9]
        goal_normal = state[15:18]

        an_len = np.linalg.norm(agent_normal)
        gn_len = np.linalg.norm(goal_normal)
        if an_len > 1e-8 and gn_len > 1e-8:
            normal_agreement = float(
                np.dot(agent_normal / an_len, goal_normal / gn_len)
            )
        else:
            normal_agreement = 0.0

        alignment = float(state[12])

        extents = sensor_data.get("object_extents", [84, 84, 84])
        max_extent = float(max(extents))
        norm_distance = float(state[13]) / max(max_extent, 1.0)

        path_blocked = float(sensor_data.get("path_blocked", False))
        on_object = float(state[11])
        norm_depth = float(state[14])

        return np.array([
            normal_agreement,
            alignment,
            norm_distance,
            path_blocked,
            on_object,
            norm_depth,
        ], dtype=float)    

    def _determine_phase(
        self,
        state: np.ndarray,
        sensor_data: Dict[str, Any],
        current_pose: np.ndarray,
    ) -> Tuple[str, Optional[np.ndarray], str]:
        on_object = float(state[11]) > 0.5
        same_side = sensor_data.get("same_side", True)
        path_blocked = sensor_data.get("path_blocked", False)
        distance = float(state[13])
        depth = sensor_data.get("depth", 100.0)

        goal_pos = self._current_goal[:3]

        # ═══ ON SURFACE ═══
        if on_object:
            self._cached_orbit_direction = None
            self._orbit_direction_age = 0

            if not same_side or path_blocked:
                stuck_threshold = self.config.get(
                    "stuck_threshold", 0.15
                )
                eff = (
                    self
                    ._compute_movement_efficiency(
                        window=20
                    )
                )
                if eff < stuck_threshold:
                    fly_dir = (
                        self
                        ._compute_detach_fly_direction(
                            current_pose,
                            sensor_data,
                        )
                    )
                    return (
                        "DETACH_NEEDED",
                        fly_dir,
                        f"stuck, consider detach "
                        f"(dist={distance:.0f}, "
                        f"eff={eff:.2f}, "
                        f"ss={same_side}, "
                        f"pb={path_blocked})",
                    )

                edge_dir = (
                    self
                    ._compute_crawl_to_edge_direction(
                        current_pose,
                        sensor_data,
                    )
                )
                return (
                    "CRAWL_TO_EDGE",
                    edge_dir,
                    f"crawling to edge "
                    f"(dist={distance:.0f}, "
                    f"ss={same_side}, "
                    f"pb={path_blocked})",
                )
            return (
                "CRAWL_TO_GOAL",
                None,
                f"crawl to goal "
                f"(dist={distance:.0f}, "
                f"ss={same_side}, "
                f"pb={path_blocked})",
            )
                
        # ═══ IN AIR ═══
        else:
            landing_threshold = (
                8.0 * self.action_space.free_step
            )

            # EMERGENCY LANDING
            if depth < 5.0:
                return (
                    "LAND",
                    None,
                    f"emergency landing, "
                    f"depth={depth:.1f}mm "
                    f"(dist={distance:.0f})",
                )

            # Close to goal + path clear → land
            if (
                distance < landing_threshold
                and not path_blocked
            ):
                self._cached_orbit_direction = None
                self._orbit_direction_age = 0
                return (
                    "LAND",
                    None,
                    f"near goal, landing "
                    f"(dist={distance:.0f})",
                )

            # Path blocked → must bypass obstacle
            if path_blocked:
                fly_dir = getattr(
                    self, "_cached_fly_direction", None
                )

                extents = sensor_data.get(
                    "object_extents", [84, 84, 84]
                )
                max_extent = float(max(extents))

                center_raw = sensor_data.get(
                    "object_center"
                )
                too_far = False
                dist_from_center = 0.0
                if center_raw is not None:
                    center = np.asarray(
                        center_raw, dtype=float
                    )
                    dist_from_center = float(
                        np.linalg.norm(
                            current_pose[:3] - center
                        )
                    )
                    too_far = (
                        dist_from_center
                        > max_extent * 1.5
                    )

                # Fallback: clear stale cache
                if fly_dir is not None and not too_far:
                    depth_now = sensor_data.get(
                        "depth", 100.0
                    )
                    if (
                        depth_now >= 100.0
                        and len(self._distance_history)
                        > 30
                    ):
                        recent_min = min(
                            self._distance_history[-30:]
                        )
                        if distance > recent_min + 10.0:
                            logger.debug(
                                f"FLY_CACHE_STALE: "
                                f"depth=100, "
                                f"dist={distance:.0f}, "
                                f"recent_min="
                                f"{recent_min:.0f}, "
                                f"clearing cache"
                            )
                            self._cached_fly_direction = (
                                None
                            )
                            self._cached_orbit_direction = (
                                None
                            )
                            fly_dir = None

                if fly_dir is not None and not too_far:
                    return (
                        "FLY_TO_EDGE",
                        fly_dir.copy(),
                        f"bypassing, cached dir "
                        f"(dist={distance:.0f}, "
                        f"from_center="
                        f"{dist_from_center:.0f})",
                    )
                else:
                    orbit_age = getattr(
                        self,
                        "_orbit_direction_age",
                        0,
                    )
                    cached_orbit = getattr(
                        self,
                        "_cached_orbit_direction",
                        None,
                    )

                    if (
                        cached_orbit is None
                        or orbit_age > 10
                    ):
                        orbit_dir = (
                            self._compute_orbit_direction(
                                current_pose, sensor_data
                            )
                        )
                        if orbit_dir is not None:
                            self._cached_orbit_direction = (
                                orbit_dir
                            )
                            self._orbit_direction_age = 0
                            self._cached_fly_direction = (
                                None
                            )
                        else:
                            orbit_dir = cached_orbit
                    else:
                        orbit_dir = cached_orbit
                        self._orbit_direction_age = (
                            orbit_age + 1
                        )

                    if orbit_dir is not None:
                        return (
                            "FLY_TO_EDGE",
                            orbit_dir.copy(),
                            f"orbiting, age="
                            f"{self._orbit_direction_age}"
                            f" (dist={distance:.0f}, "
                            f"from_center="
                            f"{dist_from_center:.0f})",
                        )

                    return (
                        "FLY_TO_GOAL",
                        None,
                        f"bypass fallback "
                        f"(dist={distance:.0f})",
                    )

            # ═══ Path NOT blocked — hysteresis ═══
            path_clear_streak = getattr(
                self, "_path_clear_streak", 0
            )

            if path_clear_streak >= 3:
                self._cached_orbit_direction = None
                self._orbit_direction_age = 0
                return (
                    "FLY_TO_GOAL",
                    None,
                    f"path clear "
                    f"{path_clear_streak} steps "
                    f"(dist={distance:.0f})",
                )

            prev_phase = getattr(
                self, "_prev_phase", None
            )
            if prev_phase == "FLY_TO_EDGE":
                cached_dir = getattr(
                    self,
                    "_cached_fly_direction",
                    None,
                )
                if cached_dir is None:
                    cached_dir = getattr(
                        self,
                        "_cached_orbit_direction",
                        None,
                    )
                return (
                    "FLY_TO_EDGE",
                    cached_dir,
                    f"path clear but hysteresis "
                    f"(streak={path_clear_streak}, "
                    f"dist={distance:.0f})",
                )

            # Default: fly to goal
            self._cached_orbit_direction = None
            self._orbit_direction_age = 0
            return (
                "FLY_TO_GOAL",
                None,
                f"path clear, fly to goal "
                f"(dist={distance:.0f})",
            )
                
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
            if sensor_data.get("edge_traversed", False):
                pass  # Skip normal flip check after edge traversal
            else:
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
    def compute_common_reward(
        self,
        state: np.ndarray,
        prev_state: np.ndarray,
        action_type: int,
        collision: Optional[str] = None,
        sensor_data: Optional[Dict] = None,
        prev_sensor_data: Optional[Dict] = None,
        current_pose: Optional[np.ndarray] = None,
    ) -> Tuple[float, bool, Optional[str]]:
        """Common reward function for Q-store and SAC.

        Same logic as _compute_reward but accepts
        action_type (0-7) instead of discrete action
        index (0-23).

        Args:
            state: Current raw state vector.
            prev_state: Previous raw state vector.
            action_type: PSAC action type (0-7).
            collision: Collision string or None.
            sensor_data: Current sensor readings.
            prev_sensor_data: Previous sensor readings.
            current_pose: Current agent pose.

        Returns:
            (reward, done, termination_reason)
        """
        cfg = self.config
        reward = 0.0
        done = False
        termination_reason = None

        distance = state[13]
        prev_distance = prev_state[13]
        on_object = state[11]
        prev_on_object = prev_state[11]
        prev_alignment = prev_state[12]

        surface_step = self.action_space.surface_step

        curr_same_side = (
            sensor_data.get("same_side", True)
            if sensor_data is not None
            else True
        )

        phase = getattr(
            self, "_current_phase",
            "CRAWL_TO_GOAL",
        )

        # ═══ 1. Progress ═══
        progress_raw = prev_distance - distance
        progress = progress_raw

        if (
            on_object < 0.5
            and phase == "FLY_TO_EDGE"
            and progress_raw < 0
        ):
            progress = progress_raw * 0.2

        if phase in (
            "CRAWL_TO_EDGE", "DETACH_NEEDED"
        ):
            progress = progress_raw * 0.1

        detour_mode = (
            prev_alignment
            < cfg["detour_alignment_threshold"]
            and (
                prev_on_object > 0.5
                or collision == "lost_object"
            )
        )
        if detour_mode and progress_raw < 0.0:
            min_progress = (
                -surface_step
                * cfg[
                    "detour_negative_progress_clip_steps"
                ]
            )
            progress = max(
                progress_raw, min_progress
            )

        reward += (
            progress
            / surface_step
            * cfg["reward_progress"]
        )

        if on_object > 0.5:
            efficiency = (
                self._compute_movement_efficiency(
                    window=20
                )
            )
            if efficiency < 0.1:
                reward += -0.3

        # ═══ 1.5 Subgoal shaping ═══
        phi_current = self._subgoal_potential(state)
        phi_prev = self._subgoal_potential(prev_state)
        subgoal_shaping = (
            self.gamma * phi_current - phi_prev
        )
        subgoal_shaping = float(
            np.clip(subgoal_shaping, -3.0, 3.0)
        )
        reward += subgoal_shaping

        # ═══ 2. Goal reached ═══
        if distance < cfg["goal_threshold"]:
            reward += cfg["reward_goal_reached"]
            done = True
            termination_reason = "goal_reached"

        # ═══ 3. Step penalty ═══
        reward += cfg["reward_step_penalty"]

        # ═══ 3.5 Risky free on surface ═══
        # action_type=1 is MoveLinear
        if (
            action_type == 1
            and prev_on_object > 0.5
        ):
            reward += -2.0

        # ═══ 3.6 Flying too far ═══
        if (
            on_object < 0.5
            and sensor_data is not None
        ):
            extents = sensor_data.get(
                "object_extents", [84, 84, 84]
            )
            max_extent = float(max(extents))
            if distance > max_extent * 1.5:
                reward += -2.0

        # ═══ 4. Collision ═══
        if collision == "surface_violation":
            reward += cfg[
                "reward_surface_violation"
            ]
            done = True
            termination_reason = (
                "collision_surface_violation"
            )
        elif collision == "detach_collision":
            reward += cfg[
                "reward_surface_violation"
            ]
            done = True
            termination_reason = (
                "collision_surface_violation"
            )
        elif collision == "lost_object":
            if action_type != 7:  # not Detach
                reward += cfg[
                    "reward_drifted_away"
                ]

        # ═══ 4.6 Detach in air ═══
        if (
            action_type == 7
            and prev_on_object < 0.5
        ):
            reward += -5.0

        # ═══ 5. Near goal on surface ═══
        near_radius = surface_step * 3
        if (
            distance < near_radius
            and on_object > 0.5
        ):
            reward += cfg[
                "reward_near_goal_on_surface"
            ]

        # ═══ 6.2 Successful landing ═══
        successful_landing = (
            prev_on_object < 0.5
            and on_object > 0.5
            and collision is None
            and curr_same_side
        )
        if successful_landing:
            landing_radius = 8.0 * surface_step
            landing_quality = max(
                0.0,
                1.0 - distance / landing_radius,
            )
            reward += 8.0 * landing_quality

        # ═══ 6.5 Correct crawl bonus ═══
        if (
            on_object > 0.5
            and phase == "CRAWL_TO_GOAL"
            and progress_raw > 0.1
        ):
            reward += 0.2

        # ═══ 6.6/6.7 Fly alignment ═══
        if (
            on_object < 0.5
            and current_pose is not None
        ):
            from scipy.spatial.transform import (
                Rotation as R,
            )

            if (
                phase == "FLY_TO_EDGE"
            ):
                subgoal_dir = getattr(
                    self,
                    "_current_subgoal_dir",
                    None,
                )
                if subgoal_dir is not None:
                    rot = R.from_euler(
                        "xyz",
                        current_pose[3:6],
                        degrees=True,
                    )
                    forward = rot.apply(
                        [0, 0, -1]
                    )
                    curr_a = float(
                        np.dot(
                            forward, subgoal_dir
                        )
                    )
                    prev_a = getattr(
                        self,
                        "_prev_subgoal_alignment",
                        None,
                    )
                    if prev_a is not None:
                        reward += (
                            (curr_a - prev_a)
                            * 2.0
                        )
                    self._prev_subgoal_alignment = (
                        curr_a
                    )
                else:
                    self._prev_subgoal_alignment = (
                        None
                    )

            elif phase in (
                "FLY_TO_GOAL", "LAND"
            ):
                goal = self._current_goal
                if goal is not None:
                    goal_dir = (
                        goal[:3]
                        - current_pose[:3]
                    )
                    goal_dist = np.linalg.norm(
                        goal_dir
                    )
                    if goal_dist > 1e-8:
                        goal_dir /= goal_dist
                        rot = R.from_euler(
                            "xyz",
                            current_pose[3:6],
                            degrees=True,
                        )
                        forward = rot.apply(
                            [0, 0, -1]
                        )
                        curr_a = float(
                            np.dot(
                                forward,
                                goal_dir,
                            )
                        )
                        prev_a = getattr(
                            self,
                            "_prev_goal_alignment",
                            None,
                        )
                        if prev_a is not None:
                            reward += (
                                (curr_a - prev_a)
                                * 2.0
                            )
                        self._prev_goal_alignment = (
                            curr_a
                        )
                    else:
                        self._prev_goal_alignment = (
                            None
                        )
                self._prev_subgoal_alignment = (
                    None
                )
            else:
                self._prev_subgoal_alignment = (
                    None
                )
                self._prev_goal_alignment = None

        # ═══ 7. Timeout ═══
        if self._steps >= cfg["max_steps_per_goal"]:
            reward += cfg["reward_timeout"]
            done = True
            if termination_reason is None:
                termination_reason = "timeout"

        return reward, done, termination_reason
    
    def _compute_reward(
        self, state, prev_state, action,
        collision, sensor_data=None,
        prev_sensor_data=None,
        current_pose=None,
    ):
        """Q-store reward = common + Q-specific."""
        # Map discrete action to action_type
        action_type, _ = (
            ExperienceExtractor.DISCRETE_TO_PSAC
            .get(action, (0, lambda c: []))
        )
        # Вызвать только тип, не params
        action_type = (
            ExperienceExtractor
            .DISCRETE_TO_PSAC[action][0]
        )

        reward, done, termination_reason = (
            self.compute_common_reward(
                state=state,
                prev_state=prev_state,
                action_type=action_type,
                collision=collision,
                sensor_data=sensor_data,
                prev_sensor_data=prev_sensor_data,
                current_pose=current_pose,
            )
        )

        # ═══ Q-specific: collision stats ═══
        if collision in (
            "surface_violation",
            "detach_collision",
        ):
            action_name = (
                self.action_space.get_info(action)
                .name
                if action is not None
                else "unknown"
            )
            self._collision_stats[action_name] = (
                self._collision_stats.get(
                    action_name, 0
                )
                + 1
            )

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
    def _choose_action_heuristic(self, state, current_pose, sensor_data):
        heuristic, _ = self._compute_heuristic_bias(
            state=state,
            current_pose=current_pose,
            sensor_data=sensor_data,
            prev_action=self._last_action,
        )
        combined = self.apply_action_mask(heuristic, state)
        action_index = None
        current_phase = getattr(self, "_current_phase", "CRAWL_TO_GOAL")
        on_object = state[11] > 0.5
        self._episode_phase_counts[current_phase] = self._episode_phase_counts.get(current_phase, 0) + 1

        if on_object:
            same_side = sensor_data.get("same_side", True)
            path_blocked = sensor_data.get("path_blocked", False)
            strategic_h = self._compute_strategic_heuristic(current_phase, on_object=True)
            if (
                strategic_h[1] > strategic_h[0]
                and self._can_detach(state)
                and (not same_side or path_blocked)
            ):
                action_index = self.action_space.IDX_DETACH
                self._strategic_stats["detach_memory_triggered"] += 1
                t_state = self._compute_detach_transition_state(
                    state, sensor_data,
                    movement_efficiency=self._compute_movement_efficiency(window=20),
                )
                self._pending_strategic_detach.append({
                    "state": t_state.copy(),
                    "step": len(self._episode_transitions),
                    "phase_was_detach_needed": current_phase == "DETACH_NEEDED",
                    "same_side_before": same_side,
                    "path_blocked_before": path_blocked,
                    "dist_before": float(state[13]),
                })
            else:
                if not same_side or path_blocked:
                    self._strategic_stats["detach_heuristic_fallback"] += 1

        elif not on_object:
            path_blocked = sensor_data.get("path_blocked", False)
            if path_blocked:
                if self._current_phase != "FLY_TO_EDGE":
                    self._current_phase = "FLY_TO_EDGE"
                self._strategic_stats["direction_memory_keep_edge"] += 1
            else:
                if self._current_phase == "FLY_TO_EDGE":
                    self._current_phase = "FLY_TO_GOAL"
                    self._cached_fly_direction = None
                self._strategic_stats["direction_memory_to_goal"] += 1
            if not path_blocked:
                self._path_clear_streak += 1
            else:
                self._path_clear_streak = 0

        self._prev_phase = current_phase

        if action_index is None:
            action_index = int(np.argmax(combined))

        if action_index == self.action_space.IDX_DETACH and on_object:
            already_recorded = (
                self._pending_strategic_detach
                and self._pending_strategic_detach[-1]["step"] == len(self._episode_transitions)
            )
            if not already_recorded:
                same_side = sensor_data.get("same_side", True)
                path_blocked = sensor_data.get("path_blocked", False)
                t_state = self._compute_detach_transition_state(
                    state, sensor_data,
                    movement_efficiency=self._compute_movement_efficiency(window=20),
                )
                self._pending_strategic_detach.append({
                    "state": t_state.copy(),
                    "step": len(self._episode_transitions),
                    "phase_was_detach_needed": current_phase == "DETACH_NEEDED",
                    "same_side_before": same_side,
                    "path_blocked_before": path_blocked,
                    "dist_before": float(state[13]),
                })

        chosen_name = self.action_space.get_info(action_index).name
        if chosen_name == "detach":
            self._consecutive_detach_count += 1
        else:
            self._consecutive_detach_count = 0

        return action_index

    def apply_action_mask(
        self, values: np.ndarray, state: np.ndarray
    ) -> np.ndarray:
        """Apply physical action masks."""
        masked = values.copy()
        on_object = state[11] > 0.5

        if on_object:
            masked[self.action_space.IDX_FREE_FORWARD] = -1e9
            masked[self.action_space.IDX_FREE_FORWARD_SMALL] = -1e9
            masked[self.action_space.IDX_FREE_BACKWARD] = -1e9
            masked[self.action_space.IDX_DETACH] = -1e9
        else:
            for idx in range(8):
                masked[idx] = -1e9
            masked[self.action_space.IDX_DETACH] = -1e9

        if self._consecutive_detach_count >= 3:
            masked[self.action_space.IDX_DETACH] = -1e9

        return masked
    
    def _get_learning_rate(self) -> float:
        if self.is_training:
            return self.alpha
        return self.alpha * self.eval_alpha_multiplier

    def _apply_success_backup_updates(self) -> None:
        """Apply backward updates along successful trajectory.

        Lambda-return: propagates discounted return backward
        through the trajectory with exponential decay.
        
        Critical Action Bonus REMOVED — credit assignment for detach
        is handled by strategic level in _on_episode_done.
        """
        if not self.success_backup_enabled:
            return
        if not self._episode_transitions:
            return

        # ═══ Standard lambda-return backup ═══
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
            lr = base_alpha * (self.success_backup_lambda ** depth)
            store = self._select_store(tr["state"])
            store.update_q_value(
                tr["state"],
                tr["action"],
                g_return,
                lr,
                count_visit=False,
            )

        # ═══ Critical Action Bonus — REMOVED ═══
        # Was: directly boosting Q-values for detach actions.
        # Now handled by strategic_detach store in _on_episode_done
        # with proper same_side/path_blocked based rewards.

        logger.debug(
            "Applied success backup: k=%d, base_alpha=%.4f, "
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
        is_strategic_override,
    ) -> str:
        action_name = self.action_space.get_info(action_index).name
        q_action_name = self.action_space.get_info(q_recommends).name
        h_action_name = self.action_space.get_info(h_recommends).name

        if is_random:
            return (
                f"##### Random: {action_name} - {action_index} "
                f"epsilon {eps}."
            )
        if is_strategic_override:
            return (
                f"##### Strategic: {action_name} - {action_index} "
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

        heuristic, heuristic_components = (
            self._compute_heuristic_bias(
                state=state,
                current_pose=current_pose,
                sensor_data=sensor_data,
                prev_action=self._last_action,
            )
        )

        q_norm = self._normalize_values(q_values)
        h_norm = self._normalize_values(heuristic)

        best_q_action = int(np.argmax(q_values))
        best_h_action = int(np.argmax(heuristic))

        eps = self._get_current_epsilon()

        has_q_data = (
            store.next_id > 0
            and np.max(np.abs(q_values)) > 1e-6
        )

        if has_q_data:
            combined = (1 - eps) * q_norm + eps * h_norm
            temperature = np.clip(0.5 * eps, 0.01, 0.5)
        else:
            combined = h_norm.copy()
            temperature = 0.02

        # Warmup: pure heuristic with greedy selection
        warmup_episodes = int(
            self.config.get("warmup_episodes", 0)
        )
        if (
            warmup_episodes > 0
            and self._total_episodes <= warmup_episodes
        ):
            combined = h_norm.copy()
            temperature = 0.001

        if self.temperature_override is not None:
            temperature = self.temperature_override

        # ═══ ACTION MASK ═══
        combined = self.apply_action_mask(combined, state)

        is_random_override = False
        is_strategic_override = False
        action_index = None
        strategic_source = None

        # ═══ STRATEGIC LEVEL ═══
        current_phase = getattr(
            self, "_current_phase", "CRAWL_TO_GOAL"
        )
        on_object = state[11] > 0.5

        self._episode_phase_counts[current_phase] = (
            self._episode_phase_counts.get(current_phase, 0) + 1
        )

        s_eps = self.strategic_epsilon
        cfg = self.config

        if on_object:
            same_side = sensor_data.get("same_side", True)
            path_blocked = sensor_data.get("path_blocked", False)

            t_state = self._compute_detach_transition_state(
                state, sensor_data,
                movement_efficiency=self._compute_movement_efficiency(window=20),
            )
            strategic_q = self.strategic_detach.get_q_values(t_state)
            strategic_h = self._compute_strategic_heuristic(
                current_phase, on_object=True
            )

            has_data = (
                self.strategic_detach.next_id > 0
                and np.max(np.abs(strategic_q)) > 1e-6
            )
            if has_data:
                sq_norm = self._normalize_values(strategic_q)
                sh_norm = self._normalize_values(strategic_h)
                strategic_combined = (
                    (1 - s_eps) * sq_norm + s_eps * sh_norm
                )
            else:
                strategic_combined = strategic_h.copy()

            should_switch = (
                strategic_combined[1] > strategic_combined[0]
            )

            if (
                should_switch
                and self._can_detach(state)
                and (not same_side or path_blocked)
            ):
                action_index = (
                    self.action_space.IDX_DETACH
                )
                strategic_source = (
                    f"detach_switch("
                    f"q=[{strategic_q[0]:.2f},"
                    f"{strategic_q[1]:.2f}],"
                    f"h=[{strategic_h[0]:.1f},"
                    f"{strategic_h[1]:.1f}],"
                    f"eps={s_eps:.2f})"
                )
                self._strategic_stats["detach_memory_triggered"] += 1

                # ═══ CHANGED: сохраняем контекст для ретроспективного обучения ═══
                self._pending_strategic_detach.append({
                    "state": t_state.copy(),
                    "step": len(self._episode_transitions),
                    "phase_was_detach_needed": (
                        current_phase == "DETACH_NEEDED"
                    ),
                    "same_side_before": same_side,
                    "path_blocked_before": path_blocked,
                    "dist_before": float(state[13]),
                })
            else:
                if not same_side or path_blocked:
                    strategic_source = (
                        f"detach_stay("
                        f"q=[{strategic_q[0]:.2f},"
                        f"{strategic_q[1]:.2f}],"
                        f"eps={s_eps:.2f})"
                    )
                    self._strategic_stats[
                        "detach_heuristic_fallback"
                    ] += 1

        elif not on_object:
            d_state = self._compute_direction_transition_state(
                state, sensor_data, current_pose
            )
            strategic_q = self.strategic_direction.get_q_values(
                d_state
            )

            # Fixed semantics: action=0=fly_to_goal, action=1=bypass
            path_blocked = sensor_data.get("path_blocked", False)

            strategic_h = np.zeros(2)
            if path_blocked:
                strategic_h[1] += 3.0
                strategic_h[0] -= 1.0
            else:
                strategic_h[0] += 3.0
                strategic_h[1] -= 3.0

            has_data = (
                self.strategic_direction.next_id > 0
                and np.max(np.abs(strategic_q)) > 1e-6
            )
            if has_data:
                sq_norm = self._normalize_values(strategic_q)
                sh_norm = self._normalize_values(strategic_h)
                dir_combined = (
                    (1 - s_eps) * sq_norm + s_eps * sh_norm
                )
            else:
                dir_combined = strategic_h.copy()

            should_bypass = dir_combined[1] > dir_combined[0]

            if should_bypass:
                if self._current_phase != "FLY_TO_EDGE":
                    self._current_phase = "FLY_TO_EDGE"
                strategic_source = (
                    f"direction_bypass("
                    f"q=[{strategic_q[0]:.2f},"
                    f"{strategic_q[1]:.2f}],"
                    f"h=[{strategic_h[0]:.1f},"
                    f"{strategic_h[1]:.1f}],"
                    f"eps={s_eps:.2f})"
                )
                self._strategic_stats[
                    "direction_memory_keep_edge"
                ] += 1
            else:
                if self._current_phase == "FLY_TO_EDGE":
                    self._current_phase = "FLY_TO_GOAL"
                    self._cached_fly_direction = None
                strategic_source = (
                    f"direction_goal("
                    f"q=[{strategic_q[0]:.2f},"
                    f"{strategic_q[1]:.2f}],"
                    f"h=[{strategic_h[0]:.1f},"
                    f"{strategic_h[1]:.1f}],"
                    f"eps={s_eps:.2f})"
                )
                self._strategic_stats[
                    "direction_memory_to_goal"
                ] += 1

            if not path_blocked:
                self._path_clear_streak += 1
            else:
                self._path_clear_streak = 0

        self._prev_phase = current_phase

        # ═══ SOFTMAX SAMPLING (tactical level) ═══
        if action_index is None:
            v = combined / temperature
            v = v - np.max(v)
            exp_v = np.exp(v)
            probs = exp_v / exp_v.sum()

            p_random = 0.02 * eps
            if np.random.random() < p_random:
                is_random_override = True
                valid_mask = np.ones(
                    self.num_actions, dtype=bool
                )
                # Apply same mask logic
                masked_check = self.apply_action_mask(
                    np.zeros(self.num_actions), state
                )
                valid_mask[masked_check < -1e8] = False
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
                action_index = int(
                    np.random.choice(len(probs), p=probs)
                )
        else:
            probs = np.zeros(self.num_actions)
            probs[action_index] = 1.0

        # ═══ RECORD PENDING TRANSITIONS ═══
        if (
            action_index == self.action_space.IDX_DETACH
            and on_object
        ):
            already_recorded = (
                self._pending_strategic_detach
                and self._pending_strategic_detach[-1]["step"]
                == len(self._episode_transitions)
            )
            if not already_recorded:
                same_side = sensor_data.get("same_side", True)
                path_blocked = sensor_data.get("path_blocked", False)
                t_state = self._compute_detach_transition_state(
                    state, sensor_data,
                    movement_efficiency=self._compute_movement_efficiency(window=20),
                )
                self._pending_strategic_detach.append({
                    "state": t_state.copy(),
                    "step": len(self._episode_transitions),
                    "phase_was_detach_needed": (
                        current_phase == "DETACH_NEEDED"
                    ),
                    "same_side_before": same_side,           # NEW
                    "path_blocked_before": path_blocked,     # NEW
                    "dist_before": float(state[13]),         # NEW
                })

        self._prev_phase = current_phase

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
                "strategic_override"
                if is_strategic_override
                else "random_exploration"
                if is_random_override
                else "softmax_sampling"
            ),
            "strategic_source": strategic_source,
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
                blend=(
                    f"{(1-eps)*100:.0f}% Q + {eps*100:.0f}% heuristic"
                    if has_q_data
                    else "100% heuristic"
                ),
                is_strategic_override=is_strategic_override,
            ),
        }

        chosen_name = self.action_space.get_info(action_index).name
        if chosen_name == "detach":
            self._consecutive_detach_count += 1
        else:
            self._consecutive_detach_count = 0

        explanation["interpretation"] = (
            f"[phase={current_phase}] "
            + explanation["interpretation"]
        )
        if strategic_source:
            explanation["interpretation"] = (
                f"[strategic={strategic_source}] "
                + explanation["interpretation"]
            )

        same_side = sensor_data.get("same_side", True)
        path_blocked = sensor_data.get("path_blocked", False)
        depth = sensor_data.get("depth", 100.0)

        subgoal_str = ""
        if getattr(self, "_current_subgoal_dir", None) is not None:
            sd = self._current_subgoal_dir
            subgoal_str = f"|sd=[{sd[0]:.2f},{sd[1]:.2f},{sd[2]:.2f}]"

        normal_str = ""
        pn = sensor_data.get("point_normal")
        if pn is not None:
            normal_str = f"|n=[{pn[0]:.2f},{pn[1]:.2f},{pn[2]:.2f}]"

        pos_str = (
            f"|pos=[{current_pose[0]:.1f},"
            f"{current_pose[1]:.1f},"
            f"{current_pose[2]:.1f}]"
        )

        debug_prefix = (
            f"[phase={current_phase}|"
            f"ss={int(same_side)}|"
            f"pb={int(path_blocked)}|"
            f"d={depth:.1f}"
            f"|al={float(state[12]):.2f}"
            f"{subgoal_str}"
            f"{normal_str}"
            f"{pos_str}]"
        )

        if strategic_source:
            debug_prefix += f"[strat={strategic_source}]"

        explanation["interpretation"] = (
            debug_prefix + " " + explanation["interpretation"]
        )
        return action_index, explanation
    
    def _can_detach(self, state: np.ndarray) -> bool:
        """Check if detach is allowed (anti-spam guards)."""
        recent_detach = sum(
            1
            for tr in self._episode_transitions[-5:]
            if tr["action"] == self.action_space.IDX_DETACH
        )
        last_was_detach = (
            self._last_action == self.action_space.IDX_DETACH
        )
        close_to_goal = (
            float(state[13]) < 3.0 * self.action_space.surface_step
        )
        return (
            recent_detach < 1
            and not last_was_detach
            and not close_to_goal
        )
    
    def _is_making_orbit_progress(self, window: int = 5) -> bool:
        """Check if agent is making progress while orbiting.

        Uses distance history: if min distance in last N steps
        is less than min distance in N steps before that,
        the orbit is productive.

        Args:
            window: Number of steps to check.

        Returns:
            True if orbit is productive.
        """
        if len(self._distance_history) < window * 2:
            return False  # Not enough data, don't assume progress

        recent = self._distance_history[-window:]
        previous = self._distance_history[-window * 2:-window]

        min_recent = min(recent)
        min_previous = min(previous)

        # Require real improvement, no tolerance
        return min_recent < min_previous
    
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
        on_object = float(state[11])
        alignment = float(state[12])
        distance = float(state[13])
        norm_depth = float(state[14])
        point_normal = sensor_data.get("point_normal")

        eps = 1e-8
        rot = R.from_euler(
            "xyz", current_pose[3:6], degrees=True
        )

        # ═══ Determine phase ═══
        phase, subgoal_dir, phase_desc = (
            self._determine_phase(
                state, sensor_data, current_pose
            )
        )

        self._current_phase = phase
        self._current_subgoal_dir = subgoal_dir
        # Store for logging
        self._current_phase_desc = phase_desc

        logger.debug(
            f"PHASE: step={self._steps}, "
            f"phase={phase}, "
            f"desc={phase_desc}, "
            f"on_object={on_object:.0f}, "
            f"distance={distance:.1f}, "
            f"alignment={alignment:.3f}"
        )

        if on_object > 0.5:
            self._flyby_count = 0

        if on_object > 0.5 and phase == "DETACH_NEEDED":
            cached = (
                self._compute_detach_fly_direction(
                    current_pose, sensor_data
                )
            )
            if cached is not None:
                self._cached_fly_direction = cached
                self._fly_direction_age = 0

        if on_object < 0.5:
            self._fly_direction_age = getattr(
                self, "_fly_direction_age", 0
            ) + 1

            if phase == "FLY_TO_EDGE":
                center_raw = sensor_data.get(
                    "object_center"
                )
                extents_raw = sensor_data.get(
                    "object_extents"
                )
                if (
                    center_raw is not None
                    and extents_raw is not None
                ):
                    center = np.asarray(
                        center_raw, dtype=float
                    )
                    dist_from_center = float(
                        np.linalg.norm(
                            current_pose[:3] - center
                        )
                    )
                    max_ext = float(max(extents_raw))
                    if dist_from_center > max_ext * 2.0:
                        self._cached_fly_direction = None
                        logger.debug(
                            f"FLY_SAFETY_ABORT: "
                            f"dist_from_center="
                            f"{dist_from_center:.1f}"
                        )

        close_to_goal = (
            distance
            < 3.0 * self.action_space.surface_step
        )

        # ────────────────────────────────────────
        # 0) SUPPRESS — always-on suppressions
        # ────────────────────────────────────────
        suppress = np.zeros(
            self.num_actions, dtype=float
        )
        suppress[self.action_space.IDX_ROTATE_POS] -= 2.0
        suppress[self.action_space.IDX_ROTATE_NEG] -= 2.0

        if not (on_object > 0.5 and close_to_goal):
            suppress[
                self.action_space.IDX_ORIENT_HOR
            ] -= 2.0
            suppress[
                self.action_space.IDX_ORIENT_VERT
            ] -= 2.0

        # supress as detach is strategic decision
        suppress[self.action_space.IDX_DETACH] -= 10.0

        if (
            self._last_action
            == self.action_space.IDX_DETACH
        ):
            suppress[
                self.action_space.IDX_DETACH
            ] -= 5.0

        recent_detach = sum(
            1
            for tr in self._episode_transitions[-3:]
            if tr["action"]
            == self.action_space.IDX_DETACH
        )
        if recent_detach >= 1:
            suppress[
                self.action_space.IDX_DETACH
            ] -= 5.0

        if (
            on_object > 0.5
            and phase == "CRAWL_TO_GOAL"
            and distance
            < 5.0 * self.action_space.surface_step
        ):
            suppress[
                self.action_space.IDX_DETACH
            ] -= 8.0

        bias += suppress
        components["suppress"] = suppress

        # ────────────────────────────────────────
        # 1) SURFACE MOVE — phase-aware
        # ────────────────────────────────────────
        surface_move = np.zeros(
            self.num_actions, dtype=float
        )
        if on_object > 0.5 and phase == "CRAWL_TO_EDGE":
            SURFACE_STRENGTH = 8.0  # было 5

            n_world = sensor_data.get("point_normal")
            if n_world is not None:
                n_world = np.asarray(n_world, dtype=float)
                n_len = float(np.linalg.norm(n_world))
                if n_len > eps:
                    n_hat = n_world / n_len
                    step = float(self.action_space.surface_step)

                    if subgoal_dir is not None:
                        e_t = subgoal_dir * step * 3.0
                    else:
                        e_t = None

                    if e_t is not None:
                        tangential_dist = float(np.linalg.norm(e_t))
                        if tangential_dist > 0.1:
                            # === Reuse existing scoring logic ===
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
                                    np.dot(e_t, e_t) - np.dot(new_e, new_e)
                                )
                                scores[i] = score
                                if score > best_score:
                                    best_score = score
                                    best = i

                            if best is not None:
                                surface_move[best] = SURFACE_STRENGTH

        if on_object > 0.5 and phase == "CRAWL_TO_GOAL":
            SURFACE_STRENGTH = 4.0

            path_blocked_now = sensor_data.get(
                "path_blocked", False
            )

            n_world = sensor_data.get("point_normal")
            if n_world is not None:
                n_world = np.asarray(
                    n_world, dtype=float
                )
                n_len = float(np.linalg.norm(n_world))
                if n_len > eps:
                    n_hat = n_world / n_len
                    step = float(
                        self.action_space.surface_step
                    )

                    if path_blocked_now:
                        # Path blocked — crawl toward
                        # edge using fly direction
                        edge_dir = (
                            self._compute_detach_fly_direction(
                                current_pose,
                                sensor_data,
                            )
                        )
                        if edge_dir is not None:
                            e_t = (
                                edge_dir
                                - np.dot(
                                    edge_dir, n_hat
                                )
                                * n_hat
                            )
                            e_t_len = np.linalg.norm(
                                e_t
                            )
                            if e_t_len > 1e-8:
                                e_t = (
                                    e_t
                                    / e_t_len
                                    * step
                                    * 3.0
                                )
                            else:
                                e_t = None
                        else:
                            e_t = None
                    else:
                        # Path clear — crawl toward
                        # goal with geodesic
                        e_world = rot.apply(
                            local_pos_error
                        )

                        goal_normal_raw = (
                            sensor_data.get(
                                "goal_normal"
                            )
                        )
                        use_geodesic = False

                        if (
                            goal_normal_raw
                            is not None
                        ):
                            gn = np.asarray(
                                goal_normal_raw,
                                dtype=float,
                            )
                            gn_len = float(
                                np.linalg.norm(gn)
                            )
                            if gn_len > 1e-8:
                                gn = gn / gn_len
                                gc_axis = np.cross(
                                    n_hat, gn
                                )
                                gc_len = float(
                                    np.linalg.norm(
                                        gc_axis
                                    )
                                )
                                if gc_len > 0.01:
                                    gc_axis /= (
                                        gc_len
                                    )
                                    geodesic_dir = (
                                        np.cross(
                                            gc_axis,
                                            n_hat,
                                        )
                                    )
                                    geo_len = float(
                                        np.linalg.norm(
                                            geodesic_dir
                                        )
                                    )
                                    if (
                                        geo_len
                                        > 1e-8
                                    ):
                                        geodesic_dir /= geo_len
                                        e_t_flat = (
                                            e_world
                                            - np.dot(
                                                e_world,
                                                n_hat,
                                            )
                                            * n_hat
                                        )
                                        if (
                                            np.dot(
                                                geodesic_dir,
                                                e_t_flat,
                                            )
                                            < 0
                                        ):
                                            geodesic_dir = -geodesic_dir
                                        e_t = (
                                            geodesic_dir
                                            * float(
                                                np.linalg.norm(
                                                    e_t_flat
                                                )
                                            )
                                        )
                                        use_geodesic = True

                        if not use_geodesic:
                            e_t = (
                                e_world
                                - np.dot(
                                    e_world, n_hat
                                )
                                * n_hat
                            )

                    # Score directions (shared logic)
                    if e_t is not None:
                        tangential_dist = float(
                            np.linalg.norm(e_t)
                        )

                        if tangential_dist > 0.1:
                            right_world = rot.apply(
                                [1.0, 0.0, 0.0]
                            )
                            tb1 = (
                                right_world
                                - np.dot(
                                    right_world,
                                    n_hat,
                                )
                                * n_hat
                            )
                            tb1_norm = (
                                np.linalg.norm(tb1)
                            )
                            if tb1_norm < 1e-8:
                                up_world = (
                                    rot.apply(
                                        [
                                            0.0,
                                            1.0,
                                            0.0,
                                        ]
                                    )
                                )
                                tb1 = (
                                    up_world
                                    - np.dot(
                                        up_world,
                                        n_hat,
                                    )
                                    * n_hat
                                )
                                tb1_norm = (
                                    np.linalg.norm(
                                        tb1
                                    )
                                )
                            if tb1_norm < 1e-8:
                                tmp = np.array(
                                    [0.0, 1.0, 0.0]
                                )
                                if (
                                    abs(
                                        np.dot(
                                            tmp,
                                            n_hat,
                                        )
                                    )
                                    > 0.9
                                ):
                                    tmp = np.array(
                                        [
                                            0.0,
                                            0.0,
                                            1.0,
                                        ]
                                    )
                                tb1 = np.cross(
                                    n_hat, tmp
                                )
                                tb1_norm = (
                                    np.linalg.norm(
                                        tb1
                                    )
                                )
                            tb1 /= (
                                tb1_norm + 1e-12
                            )
                            tb2 = np.cross(
                                n_hat, tb1
                            )
                            tb2 /= (
                                np.linalg.norm(tb2)
                                + 1e-12
                            )

                            best = None
                            best_score = -1e18
                            scores = np.full(
                                8,
                                -1e18,
                                dtype=float,
                            )
                            for (
                                i,
                                deg,
                            ) in enumerate(
                                self.action_space.SURFACE_DIRECTIONS
                            ):
                                a = np.radians(deg)
                                v_world = (
                                    np.cos(a) * tb1
                                    + np.sin(a)
                                    * tb2
                                )
                                v_norm = float(
                                    np.linalg.norm(
                                        v_world
                                    )
                                )
                                if v_norm < 1e-8:
                                    continue
                                v_world /= v_norm
                                new_e = (
                                    e_t
                                    - step * v_world
                                )
                                score = float(
                                    np.dot(e_t, e_t)
                                    - np.dot(
                                        new_e,
                                        new_e,
                                    )
                                )
                                scores[i] = score
                                if (
                                    score
                                    > best_score
                                ):
                                    best_score = (
                                        score
                                    )
                                    best = i

                            if (
                                not path_blocked_now
                            ):
                                if (
                                    distance
                                    < 3.0 * step
                                ):
                                    HYST_ABS = 1.0
                                else:
                                    HYST_ABS = 0.25
                                if (
                                    prev_action
                                    is not None
                                    and 0
                                    <= prev_action
                                    < 8
                                    and best
                                    is not None
                                ):
                                    prev_score = (
                                        scores[
                                            int(
                                                prev_action
                                            )
                                        ]
                                    )
                                    if (
                                        best_score
                                        - prev_score
                                    ) < HYST_ABS:
                                        best = int(
                                            prev_action
                                        )

                            if best is not None:
                                surface_move[
                                    best
                                ] = (
                                    SURFACE_STRENGTH
                                )

        bias += surface_move
        components["surface_move"] = surface_move

        # ────────────────────────────────────────
        # 2) STAGNATION — crawl direction only
        # ────────────────────────────────────────
        stagnation = np.zeros(
            self.num_actions, dtype=float
        )

        if on_object > 0.5 and phase  in ("CRAWL_TO_GOAL", "CRAWL_TO_EDGE"):
            if len(self._distance_history) >= 10:
                dist_10_ago = self._distance_history[-10]
                dist_progress_10 = dist_10_ago - distance
                if (
                    dist_progress_10
                    < self.action_space.surface_step
                    * 0.5
                ):
                    if (
                        prev_action is not None
                        and 0 <= prev_action < 8
                    ):
                        stagnation[prev_action] -= 2.0
                        perp1 = (prev_action + 2) % 8
                        perp2 = (prev_action + 6) % 8
                        stagnation[perp1] += 2.0
                        stagnation[perp2] += 2.0
                    if (
                        prev_action is not None
                        and 0 <= prev_action < 8
                    ):
                        opposite = (
                            prev_action + 4
                        ) % 8
                        stagnation[opposite] += 1.0

        bias += stagnation
        components["stagnation"] = stagnation

        # ────────────────────────────────────────
        # 3) STEER IN AIR — phase-driven
        # ────────────────────────────────────────
        steer = np.zeros(
            self.num_actions, dtype=float
        )

        if on_object <= 0.5:
            STEER_STRENGTH = 8.0
            rotation_step = (
                self.action_space.rotation_step
            )

            goal_dir_world = (
                self._current_goal[:3]
                - current_pose[:3]
            )
            goal_dist_world = np.linalg.norm(
                goal_dir_world
            )
            if goal_dist_world > 1e-8:
                goal_dir_world /= goal_dist_world

            if (
                phase == "FLY_TO_EDGE"
                and subgoal_dir is not None
            ):
                effective_goal = subgoal_dir
            else:
                effective_goal = goal_dir_world

            rot_current = R.from_euler(
                "xyz", current_pose[3:6], degrees=True
            )
            forward_current = rot_current.apply(
                [0, 0, -1]
            )
            dot_current = float(
                np.dot(forward_current, effective_goal)
            )
            angle_to_goal = np.degrees(
                np.arccos(np.clip(dot_current, -1, 1))
            )

            pose_angles = current_pose[3:6]

            forward_up = R.from_euler(
                "xyz",
                pose_angles
                + np.array([rotation_step, 0, 0]),
                degrees=True,
            ).apply([0, 0, -1])
            improvement_up = float(
                np.dot(forward_up, effective_goal)
                - dot_current
            )

            forward_down = R.from_euler(
                "xyz",
                pose_angles
                + np.array([-rotation_step, 0, 0]),
                degrees=True,
            ).apply([0, 0, -1])
            improvement_down = float(
                np.dot(forward_down, effective_goal)
                - dot_current
            )

            forward_left = R.from_euler(
                "xyz",
                pose_angles
                + np.array([0, rotation_step, 0]),
                degrees=True,
            ).apply([0, 0, -1])
            improvement_left = float(
                np.dot(forward_left, effective_goal)
                - dot_current
            )

            forward_right = R.from_euler(
                "xyz",
                pose_angles
                + np.array([0, -rotation_step, 0]),
                degrees=True,
            ).apply([0, 0, -1])
            improvement_right = float(
                np.dot(forward_right, effective_goal)
                - dot_current
            )

            if (
                np.linalg.norm(
                    forward_up - forward_current
                )
                < 1e-4
            ):
                improvement_up = 0.0
            if (
                np.linalg.norm(
                    forward_down - forward_current
                )
                < 1e-4
            ):
                improvement_down = 0.0
            if (
                np.linalg.norm(
                    forward_left - forward_current
                )
                < 1e-4
            ):
                improvement_left = 0.0
            if (
                np.linalg.norm(
                    forward_right - forward_current
                )
                < 1e-4
            ):
                improvement_right = 0.0

            TURN_ONLY = (
                3.0
                * self.action_space.rotation_step_big
            )
            FLY_THR = (
                4.0 * self.action_space.rotation_step
            )

            if angle_to_goal > TURN_ONLY:
                steer[
                    self.action_space.IDX_FREE_FORWARD
                ] -= STEER_STRENGTH
                steer[
                    self.action_space.IDX_FREE_FORWARD_SMALL
                ] -= STEER_STRENGTH

                best_pitch = max(
                    improvement_up, improvement_down
                )
                best_yaw = max(
                    improvement_left, improvement_right
                )
                big_multiplier = min(
                    angle_to_goal / 45.0, 2.0
                )

                if (
                    best_pitch >= best_yaw
                    and best_pitch > 0.001
                ):
                    if (
                        improvement_up
                        > improvement_down
                    ):
                        steer[
                            self.action_space.IDX_LOOK_UP
                        ] += STEER_STRENGTH
                        steer[
                            self.action_space.IDX_LOOK_UP_BIG
                        ] += (
                            STEER_STRENGTH
                            * big_multiplier
                        )
                    else:
                        steer[
                            self.action_space.IDX_LOOK_DOWN
                        ] += STEER_STRENGTH
                        steer[
                            self.action_space.IDX_LOOK_DOWN_BIG
                        ] += (
                            STEER_STRENGTH
                            * big_multiplier
                        )

                if (
                    best_yaw >= best_pitch
                    and best_yaw > 0.001
                ):
                    if (
                        improvement_left
                        > improvement_right
                    ):
                        steer[
                            self.action_space.IDX_TURN_LEFT
                        ] += STEER_STRENGTH
                        steer[
                            self.action_space.IDX_TURN_LEFT_BIG
                        ] += (
                            STEER_STRENGTH
                            * big_multiplier
                        )
                    else:
                        steer[
                            self.action_space.IDX_TURN_RIGHT
                        ] += STEER_STRENGTH
                        steer[
                            self.action_space.IDX_TURN_RIGHT_BIG
                        ] += (
                            STEER_STRENGTH
                            * big_multiplier
                        )

            elif angle_to_goal > FLY_THR:
                steer[
                    self.action_space.IDX_FREE_FORWARD_SMALL
                ] += (STEER_STRENGTH * 0.5)
                steer[
                    self.action_space.IDX_FREE_FORWARD
                ] -= (STEER_STRENGTH * 0.5)

                turn_strength = STEER_STRENGTH * 0.7
                best_pitch = max(
                    improvement_up, improvement_down
                )
                best_yaw = max(
                    improvement_left, improvement_right
                )

                if best_pitch > 0.001:
                    if (
                        improvement_up
                        > improvement_down
                    ):
                        steer[
                            self.action_space.IDX_LOOK_UP
                        ] += turn_strength
                    else:
                        steer[
                            self.action_space.IDX_LOOK_DOWN
                        ] += turn_strength
                steer[
                    self.action_space.IDX_LOOK_UP_BIG
                ] -= 3.0
                steer[
                    self.action_space.IDX_LOOK_DOWN_BIG
                ] -= 3.0

                if best_yaw > 0.001:
                    if (
                        improvement_left
                        > improvement_right
                    ):
                        steer[
                            self.action_space.IDX_TURN_LEFT
                        ] += turn_strength
                    else:
                        steer[
                            self.action_space.IDX_TURN_RIGHT
                        ] += turn_strength
                steer[
                    self.action_space.IDX_TURN_LEFT_BIG
                ] -= 3.0
                steer[
                    self.action_space.IDX_TURN_RIGHT_BIG
                ] -= 3.0

            else:
                if (
                    distance
                    > 8.0 * self.action_space.free_step
                ):
                    steer[
                        self.action_space.IDX_FREE_FORWARD
                    ] += STEER_STRENGTH
                    steer[
                        self.action_space.IDX_FREE_FORWARD_SMALL
                    ] += (STEER_STRENGTH * 0.5)
                else:
                    steer[
                        self.action_space.IDX_FREE_FORWARD_SMALL
                    ] += STEER_STRENGTH
                    steer[
                        self.action_space.IDX_FREE_FORWARD
                    ] += (STEER_STRENGTH * 0.3)

                correction = STEER_STRENGTH * 0.3
                best_pitch = max(
                    improvement_up, improvement_down
                )
                best_yaw = max(
                    improvement_left, improvement_right
                )

                if best_pitch > 0.01:
                    if (
                        improvement_up
                        > improvement_down
                    ):
                        steer[
                            self.action_space.IDX_LOOK_UP
                        ] += correction
                    else:
                        steer[
                            self.action_space.IDX_LOOK_DOWN
                        ] += correction

                if best_yaw > 0.01:
                    if (
                        improvement_left
                        > improvement_right
                    ):
                        steer[
                            self.action_space.IDX_TURN_LEFT
                        ] += correction
                    else:
                        steer[
                            self.action_space.IDX_TURN_RIGHT
                        ] += correction

            if (
                phase == "FLY_TO_EDGE"
                and sensor_data.get(
                    "path_blocked", False
                )
                and norm_depth < 0.2
            ):
                TRAPPED_BOOST = 4.0
                steer[
                    self.action_space.IDX_LOOK_UP
                ] += TRAPPED_BOOST
                steer[
                    self.action_space.IDX_LOOK_UP_BIG
                ] += TRAPPED_BOOST
                steer[
                    self.action_space.IDX_TURN_LEFT
                ] -= (TRAPPED_BOOST * 0.5)
                steer[
                    self.action_space.IDX_TURN_RIGHT
                ] -= (TRAPPED_BOOST * 0.5)
                steer[
                    self.action_space.IDX_TURN_LEFT_BIG
                ] -= (TRAPPED_BOOST * 0.5)
                steer[
                    self.action_space.IDX_TURN_RIGHT_BIG
                ] -= (TRAPPED_BOOST * 0.5)

            # Suppress surface/utility actions in air
            steer[
                self.action_space.IDX_FREE_BACKWARD
            ] -= 8.0
            for idx in range(8):
                steer[idx] -= 8.0
            steer[
                self.action_space.IDX_ORIENT_HOR
            ] -= 8.0
            steer[
                self.action_space.IDX_ORIENT_VERT
            ] -= 8.0
            steer[
                self.action_space.IDX_ROTATE_POS
            ] -= 8.0
            steer[
                self.action_space.IDX_ROTATE_NEG
            ] -= 8.0

        bias += steer
        components["steer_in_air"] = steer

        # ────────────────────────────────────────
        # 4) DAMP FREE ON SURFACE
        # ────────────────────────────────────────
        damp_free = np.zeros(
            self.num_actions, dtype=float
        )
        if on_object > 0.5:
            damp_free[
                self.action_space.IDX_FREE_FORWARD
            ] -= 8.0
            damp_free[
                self.action_space.IDX_FREE_FORWARD_SMALL
            ] -= 8.0
            damp_free[
                self.action_space.IDX_FREE_BACKWARD
            ] -= 8.0
            damp_free[
                self.action_space.IDX_LOOK_UP_BIG
            ] -= 4.0
            damp_free[
                self.action_space.IDX_LOOK_DOWN_BIG
            ] -= 4.0
            damp_free[
                self.action_space.IDX_TURN_LEFT_BIG
            ] -= 4.0
            damp_free[
                self.action_space.IDX_TURN_RIGHT_BIG
            ] -= 4.0
            damp_free[
                self.action_space.IDX_ORIENT_HOR
            ] -= 4.0
            damp_free[
                self.action_space.IDX_ORIENT_VERT
            ] -= 4.0
            damp_free[
                self.action_space.IDX_ROTATE_POS
            ] -= 4.0
            damp_free[
                self.action_space.IDX_ROTATE_NEG
            ] -= 4.0
        bias += damp_free
        components["damp_free_on_surface"] = damp_free

        # ────────────────────────────────────────
        # 5) FLYBY CORRECTION
        # ────────────────────────────────────────
        flyby_bias = np.zeros(
            self.num_actions, dtype=float
        )

        if len(self._distance_history) >= 3:
            dist_trend_2 = (
                (
                    self._distance_history[-1]
                    - self._distance_history[-2]
                )
                if len(self._distance_history) >= 2
                else 0.0
            )
            dist_trend_3 = (
                self._distance_history[-1]
                - self._distance_history[-3]
            )

            if (
                on_object < 0.5
                and phase != "FLY_TO_EDGE"
            ):
                forward_current = rot.apply([0, 0, -1])
                goal_dir_world_fb = (
                    self._current_goal[:3]
                    - current_pose[:3]
                )
                goal_dist_world_fb = float(
                    np.linalg.norm(goal_dir_world_fb)
                )
                if goal_dist_world_fb > 1e-8:
                    goal_dir_norm = (
                        goal_dir_world_fb
                        / goal_dist_world_fb
                    )
                    approach_cos = float(
                        np.dot(
                            forward_current,
                            goal_dir_norm,
                        )
                    )
                else:
                    approach_cos = 1.0

                flyby_count = getattr(
                    self, "_flyby_count", 0
                )
                flyby_triggered = False
                FLYBY_STRENGTH = 4.0

                if dist_trend_2 > 2.0:
                    flyby_triggered = True
                    FLYBY_STRENGTH = (
                        5.0 + flyby_count * 2.0
                    )
                elif (
                    dist_trend_3 > 3.0
                    and approach_cos < 0.5
                ):
                    flyby_triggered = True
                    FLYBY_STRENGTH = (
                        4.0 + flyby_count * 2.0
                    )

                if len(self._distance_history) >= 5:
                    min_recent_5 = min(
                        self._distance_history[-5:]
                    )
                    if (
                        distance
                        > min_recent_5
                        + 2.0
                        * self.action_space.free_step
                    ):
                        if not flyby_triggered:
                            flyby_triggered = True
                            FLYBY_STRENGTH = max(
                                5.0
                                + flyby_count * 2.0,
                                FLYBY_STRENGTH,
                            )

                if flyby_triggered:
                    self._flyby_count = (
                        flyby_count + 1
                    )

                    flyby_bias[
                        self.action_space.IDX_FREE_FORWARD
                    ] -= FLYBY_STRENGTH
                    flyby_bias[
                        self.action_space.IDX_FREE_FORWARD_SMALL
                    ] -= (FLYBY_STRENGTH * 0.8)

                    if self._flyby_count >= 2:
                        flyby_bias[
                            self.action_space.IDX_FREE_FORWARD
                        ] -= FLYBY_STRENGTH

                    pose_angles_fb = current_pose[3:6]
                    rotation_step_fb = (
                        self.action_space.rotation_step
                    )

                    fwd_up = R.from_euler(
                        "xyz",
                        pose_angles_fb
                        + np.array(
                            [rotation_step_fb, 0, 0]
                        ),
                        degrees=True,
                    ).apply([0, 0, -1])
                    fwd_down = R.from_euler(
                        "xyz",
                        pose_angles_fb
                        + np.array(
                            [-rotation_step_fb, 0, 0]
                        ),
                        degrees=True,
                    ).apply([0, 0, -1])
                    fwd_left = R.from_euler(
                        "xyz",
                        pose_angles_fb
                        + np.array(
                            [0, rotation_step_fb, 0]
                        ),
                        degrees=True,
                    ).apply([0, 0, -1])
                    fwd_right = R.from_euler(
                        "xyz",
                        pose_angles_fb
                        + np.array(
                            [0, -rotation_step_fb, 0]
                        ),
                        degrees=True,
                    ).apply([0, 0, -1])

                    imp_up = float(
                        np.dot(fwd_up, goal_dir_norm)
                        - approach_cos
                    )
                    imp_down = float(
                        np.dot(
                            fwd_down, goal_dir_norm
                        )
                        - approach_cos
                    )
                    imp_left = float(
                        np.dot(
                            fwd_left, goal_dir_norm
                        )
                        - approach_cos
                    )
                    imp_right = float(
                        np.dot(
                            fwd_right, goal_dir_norm
                        )
                        - approach_cos
                    )

                    best_pitch = max(
                        imp_up, imp_down
                    )
                    best_yaw = max(
                        imp_left, imp_right
                    )

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

        bias += flyby_bias
        components["flyby_correction"] = flyby_bias

        # ────────────────────────────────────────
        # 6) ORIENTATION COOLDOWN
        # ────────────────────────────────────────
        cooldown_bias = np.zeros(
            self.num_actions, dtype=float
        )

        if on_object > 0.5:
            for (
                action_idx,
                count,
            ) in self._action_no_effect_count.items():
                if count >= 3:
                    penalty = min(count * 3.0, 12.0)
                    cooldown_bias[action_idx] -= penalty

        bias += cooldown_bias
        components["orientation_cooldown"] = (
            cooldown_bias
        )

        # ────────────────────────────────────────
        # 7) LANDING
        # ────────────────────────────────────────
        landing_bias = np.zeros(
            self.num_actions, dtype=float
        )

        if on_object < 0.5 and phase in (
            "LAND",
            "FLY_TO_GOAL",
            "FLY_TO_EDGE",
        ):
            LANDING_THRESHOLD = (
                8.0 * self.action_space.free_step
            )
            CLOSE_LANDING = (
                3.0 * self.action_space.free_step
            )

            goal_dir_land = (
                self._current_goal[:3]
                - current_pose[:3]
            )
            goal_dist_land = np.linalg.norm(
                goal_dir_land
            )
            if goal_dist_land > 1e-8:
                goal_dir_land /= goal_dist_land
            forward_land = rot.apply([0, 0, -1])
            dot_land = float(
                np.dot(forward_land, goal_dir_land)
            )
            angle_to_goal_land = np.degrees(
                np.arccos(np.clip(dot_land, -1, 1))
            )

            depth = sensor_data.get("depth", 100.0)

            if depth < 5.0:
                landing_bias[
                    self.action_space.IDX_FREE_FORWARD
                ] -= 15.0
                landing_bias[
                    self.action_space.IDX_FREE_FORWARD_SMALL
                ] += 8.0
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
                    landing_bias[idx] -= 8.0

            elif phase == "LAND":
                if distance < CLOSE_LANDING:
                    if angle_to_goal_land < 30.0:
                        landing_bias[
                            self.action_space.IDX_FREE_FORWARD_SMALL
                        ] += 10.0
                        landing_bias[
                            self.action_space.IDX_FREE_FORWARD
                        ] -= 5.0
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
                        landing_bias[
                            self.action_space.IDX_FREE_FORWARD
                        ] -= 15.0
                        landing_bias[
                            self.action_space.IDX_FREE_FORWARD_SMALL
                        ] -= 10.0
                else:
                    landing_urgency = max(
                        0.0,
                        1.0
                        - (
                            distance
                            / LANDING_THRESHOLD
                        ),
                    )
                    landing_bias[
                        self.action_space.IDX_FREE_FORWARD
                    ] -= (4.0 * landing_urgency)
                    landing_bias[
                        self.action_space.IDX_FREE_FORWARD_SMALL
                    ] += (3.0 * landing_urgency)

                    if depth < 10.0:
                        landing_bias[
                            self.action_space.IDX_FREE_FORWARD_SMALL
                        ] += 4.0
                        landing_bias[
                            self.action_space.IDX_FREE_FORWARD
                        ] -= 4.0

                    if (
                        distance
                        < 4.0
                        * self.action_space.free_step
                    ):
                        for idx in [
                            self.action_space.IDX_LOOK_UP_BIG,
                            self.action_space.IDX_LOOK_DOWN_BIG,
                            self.action_space.IDX_TURN_LEFT_BIG,
                            self.action_space.IDX_TURN_RIGHT_BIG,
                        ]:
                            landing_bias[idx] -= 3.0

            if not sensor_data.get(
                "path_blocked", False
            ):
                if len(self._distance_history) >= 10:
                    min_recent_10 = min(
                        self._distance_history[-10:]
                    )
                    overshoot = (
                        distance - min_recent_10
                    )
                    if (
                        overshoot
                        > 1.0
                        * self.action_space.free_step
                        and min_recent_10
                        < LANDING_THRESHOLD
                    ):
                        landing_bias[
                            self.action_space.IDX_FREE_FORWARD
                        ] -= 20.0
                        landing_bias[
                            self.action_space.IDX_FREE_FORWARD_SMALL
                        ] -= 15.0

        if phase == "LAND" and on_object < 0.5:
            depth = sensor_data.get("depth", 100.0)
            if (
                depth
                < 3.0 * self.action_space.free_step
            ):
                landing_bias[
                    self.action_space.IDX_FREE_FORWARD
                ] -= 20.0
                landing_bias[
                    self.action_space.IDX_FREE_FORWARD_SMALL
                ] += 5.0
            elif (
                depth
                < 6.0 * self.action_space.free_step
            ):
                landing_bias[
                    self.action_space.IDX_FREE_FORWARD
                ] -= 10.0
                landing_bias[
                    self.action_space.IDX_FREE_FORWARD_SMALL
                ] += 3.0

        bias += landing_bias
        components["landing"] = landing_bias

        return bias, components
            
    # ══════════════════════════════════════════════════════════
    # SUBGOAL HELPERS
    # ══════════════════════════════════════════════════════════
    def _compute_orbit_direction(
        self,
        current_pose: np.ndarray,
        sensor_data: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        """Compute direction to orbit/bypass around object toward goal.

        Strategy depends on relative positions:
        - Goal farther from center: orbit around with outward radial
        - Goal closer to center: orbit around with inward radial
        (to fly over edge and descend inside)

        Args:
            current_pose: Agent pose [x, y, z, rx, ry, rz].
            sensor_data: Current sensor readings.

        Returns:
            Unit vector in world space, or None if cannot compute.
        """
        center_raw = sensor_data.get("object_center")
        if center_raw is None:
            return None

        center = np.asarray(center_raw, dtype=float)
        goal_pos = self._current_goal[:3]

        up_dir_raw = sensor_data.get("up_direction", [0, 0, 1])
        up_dir = np.asarray(up_dir_raw, dtype=float)
        height_axis = int(np.argmax(np.abs(up_dir)))

        # Horizontal vectors
        center_to_agent = current_pose[:3] - center
        center_to_agent[height_axis] = 0.0
        ca_len = np.linalg.norm(center_to_agent)
        if ca_len < 1e-8:
            return None
        radial_outward = center_to_agent / ca_len

        center_to_goal = goal_pos - center
        center_to_goal[height_axis] = 0.0
        cg_len = np.linalg.norm(center_to_goal)

        # Determine if goal is inside or outside relative to agent
        goal_is_inside = cg_len < ca_len * 0.8

        # Radial direction: outward or inward depending on goal
        if goal_is_inside:
            radial = -radial_outward
        else:
            radial = radial_outward

        # Two tangent directions
        tangent1 = np.cross(up_dir, center_to_agent)
        t1_len = np.linalg.norm(tangent1)
        if t1_len < 1e-8:
            return radial
        tangent1 /= t1_len
        tangent2 = -tangent1

        # Pick tangent closer to goal direction
        if cg_len > 1e-8:
            dot1 = float(np.dot(tangent1, center_to_goal))
            dot2 = float(np.dot(tangent2, center_to_goal))
            tangent = tangent1 if dot1 >= dot2 else tangent2
        else:
            tangent = tangent1

        # Blend: tangent + radial
        radial_dot_goal = float(np.dot(radial_outward, center_to_goal))
        goal_alignment = abs(radial_dot_goal) / (cg_len + 1e-8)

        if goal_is_inside:
            # Flying inward — more radial to cross over edge
            orbit_dir = tangent * 0.5 + radial * 0.7
        elif goal_alignment > 0.7:
            # Goal directly behind — need strong orbit
            orbit_dir = tangent * 0.8 + radial * 0.3
        else:
            # Goal to the side — balanced
            orbit_dir = tangent * 0.7 + radial * 0.5

        orbit_len = np.linalg.norm(orbit_dir)
        if orbit_len < 1e-8:
            return radial
        orbit_dir /= orbit_len

        logger.debug(
            f"ORBIT_DIR: "
            f"goal_is_inside={goal_is_inside}, "
            f"ca_len={ca_len:.1f}, cg_len={cg_len:.1f}, "
            f"radial={[round(x, 3) for x in radial.tolist()]}, "
            f"tangent={[round(x, 3) for x in tangent.tolist()]}, "
            f"orbit_dir={[round(x, 3) for x in orbit_dir.tolist()]}"
        )

        # ═══ Vertical escape when orbit stalls ═══
        orbit_age = getattr(self, "_orbit_direction_age", 0)
        if orbit_age > 15:
            # Orbit has been going too long — add vertical
            # component to escape over/under obstacle
            up = np.asarray(up_dir_raw, dtype=float)
            
            # Check if goal is above or below agent
            goal_height = goal_pos[height_axis]
            agent_height = current_pose[height_axis]
            
            if goal_height > agent_height:
                vertical = up * 0.5
            else:
                vertical = -up * 0.5
            
            orbit_dir = orbit_dir + vertical
            orbit_len = np.linalg.norm(orbit_dir)
            if orbit_len > 1e-8:
                orbit_dir /= orbit_len
            
            logger.debug(
                f"ORBIT_VERTICAL_ESCAPE: age={orbit_age}, "
                f"added vertical component"
            )

        return orbit_dir

    def _compute_crawl_to_edge_direction(
        self,
        current_pose: np.ndarray,
        sensor_data: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        """Compute surface-tangent direction toward nearest edge/rim.

        Mirrors the logic of _compute_detach_fly_direction but returns
        a direction projected onto the tangent plane (for crawling,
        not flying).

        Strategy:
        - not same_side + vertical wall: crawl up toward rim
        (same as detach_fly's up_tangent logic)
        - not same_side + horizontal surface: crawl away from center
        toward edge (same as detach_fly's away_from_center logic)
        - same_side + path_blocked: crawl toward goal projection
        on tangent plane (toward nearest corner/edge that leads
        to goal)

        Returns:
            Unit vector in world space (tangent to surface), or None.
        """
        normal = sensor_data.get("point_normal")
        if normal is None:
            return None
        n = np.asarray(normal, dtype=float)
        n_len = np.linalg.norm(n)
        if n_len < 1e-8:
            return None
        n /= n_len

        up_dir = np.asarray(
            sensor_data.get("up_direction", [0, 0, 1]), dtype=float
        )
        same_side = sensor_data.get("same_side", True)

        if not same_side:
            # ═══ Opposite sides — mirror _compute_detach_fly_direction ═══

            # Project up onto tangent plane
            up_tangent = up_dir - np.dot(up_dir, n) * n
            up_tangent_len = np.linalg.norm(up_tangent)

            if up_tangent_len > 0.3:
                # Vertical wall: crawl up toward rim
                crawl_dir = up_tangent / up_tangent_len
            else:
                # Horizontal surface (bottom/top):
                # crawl away from center toward edge
                center = np.asarray(
                    sensor_data.get("object_center", [0, 0, 0]),
                    dtype=float,
                )
                away = current_pose[:3] - center
                away_t = away - np.dot(away, n) * n
                away_len = np.linalg.norm(away_t)
                if away_len > 1e-8:
                    crawl_dir = away_t / away_len
                else:
                    # Fallback: any direction away from center
                    crawl_dir = up_dir - np.dot(up_dir, n) * n
                    cd_len = np.linalg.norm(crawl_dir)
                    if cd_len < 1e-8:
                        return None
                    crawl_dir /= cd_len
        else:
            # ═══ Same side, path blocked — crawl toward goal ═══
            # Project goal direction onto tangent plane
            # This naturally points toward the nearest edge/corner
            # that leads to the goal
            goal_dir = self._current_goal[:3] - current_pose[:3]
            crawl_dir = goal_dir - np.dot(goal_dir, n) * n

            crawl_len = np.linalg.norm(crawl_dir)
            if crawl_len < 1e-8:
                # Goal directly behind normal — crawl away from center
                center = np.asarray(
                    sensor_data.get("object_center", [0, 0, 0]),
                    dtype=float,
                )
                away = current_pose[:3] - center
                crawl_dir = away - np.dot(away, n) * n
                crawl_len = np.linalg.norm(crawl_dir)
                if crawl_len < 1e-8:
                    return None

        crawl_len = np.linalg.norm(crawl_dir)
        if crawl_len < 1e-8:
            return None
        return crawl_dir / crawl_len

    def _compute_detach_fly_direction(
        self,
        current_pose: np.ndarray,
        sensor_data: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        """Compute fly direction for obstacle avoidance after detach.

        When same_side=False (agent and goal on opposite sides of wall):
        fly toward open edge (rim) using up_direction projected onto
        tangent plane. For horizontal surfaces, fly away from center
        with vertical component toward rim.

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
            f"agent_pos="
            f"{[round(x, 1) for x in current_pose[:3].tolist()]}, "
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
                # Wall: fly along surface toward rim (straight up)
                up_tangent /= up_tangent_len
                fly_dir = up_tangent
                logger.debug(
                    f"DETACH_FLY_DIR: opposite sides, wall, "
                    f"up_tangent="
                    f"{[round(x, 3) for x in up_tangent.tolist()]}, "
                    f"fly_dir="
                    f"{[round(x, 3) for x in fly_dir.tolist()]}"
                )
                return fly_dir
            else:
                # Horizontal surface (bottom/top):
                # fly away from center + toward open edge (rim)
                center_raw = sensor_data.get("object_center")
                if center_raw is not None:
                    center = np.asarray(center_raw, dtype=float)
                    away = current_pose[:3] - center
                    away_t = away - np.dot(away, n) * n
                    away_len = np.linalg.norm(away_t)
                    if away_len > 1e-8:
                        away_t /= away_len
                        # Add vertical component toward open edge
                        fly_dir = away_t * 0.5 + up * 0.8
                        fly_dir /= (
                            np.linalg.norm(fly_dir) + 1e-12
                        )
                        logger.debug(
                            f"DETACH_FLY_DIR: opposite sides, "
                            f"horizontal, "
                            f"away={[round(x, 3) for x in away_t.tolist()]}, "
                            f"up={[round(x, 3) for x in up.tolist()]}, "
                            f"fly_dir="
                            f"{[round(x, 3) for x in fly_dir.tolist()]}"
                        )
                        return fly_dir

                # Fallback: fly toward rim (up direction)
                logger.debug(
                    "DETACH_FLY_DIR: opposite sides, "
                    "horizontal fallback to up"
                )
                return up.copy()

        # Same side — original tangent logic
        tangent = goal_dir - np.dot(goal_dir, n) * n
        tangent_len = np.linalg.norm(tangent)

        up_raw = sensor_data.get("up_direction")
        if up_raw is not None:
            up = np.asarray(up_raw, dtype=float)
            normal_horizontality = 1.0 - abs(
                float(np.dot(n, up))
            )
        else:
            normal_horizontality = 1.0

        if tangent_len < 1e-8:
            fly_dir = n.copy()
            logger.debug(
                f"DETACH_FLY_DIR: same_side, tangent degenerate, "
                f"fly_dir=normal="
                f"{[round(x, 3) for x in fly_dir.tolist()]}"
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

    def _is_crawl_to_edge_stuck(self, window: int = 30) -> bool:
        """Check if agent is stuck while crawling to edge.

        Cannot use distance-to-goal because agent may correctly
        move away from goal while approaching edge (e.g. crawling
        up outer wall of mug toward rim).

        Instead checks if agent position is oscillating in place:
        if the range of distances over the window is small,
        agent is not making meaningful movement.

        Args:
            window: Number of recent steps to check.

        Returns:
            True if agent appears stuck (oscillating in place).
        """
        if len(self._distance_history) < window:
            return False

        recent = self._distance_history[-window:]
        dist_range = max(recent) - min(recent)

        # Metric 1: not moving at all
        if dist_range < self.action_space.surface_step * 2.0:
            return True

        return False

    def _compute_movement_efficiency(self, window: int = 20) -> float:
        """Ratio of net displacement to total movement.
        
        Measures whether agent is making directional progress
        (toward goal OR toward edge) vs oscillating in place.
        Independent of direction — works for both CRAWL_TO_GOAL
        and CRAWL_TO_EDGE.
        
        Returns:
            Value in [0, 1]:
            ~0.0 = oscillating (stuck)
            ~1.0 = moving in consistent direction
        """
        if len(self._distance_history) < window:
            return 0.5
        
        recent = self._distance_history[-window:]
        
        net_change = abs(recent[-1] - recent[0])
        
        total_variation = sum(
            abs(recent[i+1] - recent[i]) 
            for i in range(len(recent) - 1)
        )
        
        if total_variation < 1e-8:
            return 0.0
        
        return float(np.clip(net_change / total_variation, 0.0, 1.0))

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

        # ═══ Retrospective strategic detach update ═══
        cfg = self.config
        switch_alpha = (
            self._get_learning_rate()
            * cfg.get("strategic_alpha_switch_multiplier", 1.0)
        )
        had_detach = len(self._pending_strategic_detach) > 0
        stuck_threshold = self.config.get("stuck_threshold", 0.05)

        # ── Episodes WITH detach ──
        if had_detach:
            for pending in self._pending_strategic_detach:
                detach_step = pending["step"]
                t_state = pending["state"]

                # Check: did detach actually execute?
                if detach_step + 1 < len(self._episode_transitions):
                    next_s = self._episode_transitions[
                        detach_step + 1
                    ]["state"]
                    if next_s[11] > 0.5:
                        continue

                self._strategic_stats["detach_total"] += 1

                # ═══ CHANGED: reward based on geometric outcome ═══
                same_side_before = pending.get("same_side_before", False)
                path_blocked_before = pending.get("path_blocked_before", True)

                # Find state after landing
                same_side_after = None
                path_blocked_after = None
                landed = False
                air_steps_after = 0
                for j in range(
                    detach_step + 1, len(self._episode_transitions)
                ):
                    future_state = self._episode_transitions[j]["state"]
                    if future_state[11] > 0.5:  # landed
                        landed = True
                        same_side_after = self._episode_transitions[j].get(
                            "same_side", True
                        )
                        path_blocked_after = self._episode_transitions[j].get(
                            "path_blocked", False
                        )
                        break
                    air_steps_after += 1

                if goal_reached:
                    if not landed:
                        # Never landed but somehow reached goal — unlikely, neutral
                        switch_target = 0.0
                    elif not same_side_before and same_side_after:
                        # Detach solved the same_side problem — full reward
                        switch_target = 1.0
                        self._strategic_stats["detach_led_to_success"] += 1
                    elif path_blocked_before and not path_blocked_after:
                        # Detach unblocked the path — good reward
                        switch_target = 0.8
                        self._strategic_stats["detach_led_to_success"] += 1
                    elif same_side_before and same_side_after:
                        # Was on correct side, stayed on correct side
                        # Detach was unnecessary but episode succeeded
                        switch_target = 0.0  # neutral
                        self._strategic_stats["detach_led_to_success"] += 1
                    else:
                        # Landed on wrong side but still reached goal
                        switch_target = 0.1
                        self._strategic_stats["detach_led_to_success"] += 1

                elif termination_reason == "collision_surface_violation":
                    switch_target = -0.5
                    self._strategic_stats["detach_led_to_collision"] += 1

                elif termination_reason == "timeout":
                    # ═══ CHANGED: strong penalty for timeout ═══
                    switch_target = -1.0  # was -0.3
                    self._strategic_stats["detach_led_to_timeout"] += 1

                else:
                    switch_target = 0.0

                self.strategic_detach.update_q_value(
                    t_state, action=1,
                    td_target=switch_target,
                    alpha=switch_alpha,
                )

                # ═══ CHANGED: retrospective stuck states ═══
                # Only update if detach was clearly good or bad
                stuck_states = []
                for i in range(detach_step - 1, -1, -1):
                    tr = self._episode_transitions[i]
                    s = tr["state"]
                    if s[11] < 0.5:
                        break
                    stuck_states.append(tr)

                if switch_target > 0.5:
                    # Detach was clearly good — teach stuck states that switch helps
                    for tr in stuck_states:
                        pb = float(tr.get("path_blocked", False))
                        eff = tr.get("movement_efficiency", 0.5)
                        t_retro = self._compute_detach_state_from_full(
                            tr["state"], path_blocked=pb, movement_efficiency=eff,
                        )
                        self.strategic_detach.update_q_value(
                            t_retro, action=1,
                            td_target=switch_target * 0.5,
                            alpha=switch_alpha * 0.3,
                            count_visit=False,
                        )
                elif switch_target < -0.5:
                    # Detach was clearly bad — teach stuck states that stay is better
                    for tr in stuck_states:
                        pb = float(tr.get("path_blocked", False))
                        eff = tr.get("movement_efficiency", 0.5)
                        t_retro = self._compute_detach_state_from_full(
                            tr["state"], path_blocked=pb, movement_efficiency=eff,
                        )
                        self.strategic_detach.update_q_value(
                            t_retro, action=0,
                            td_target=0.3,
                            alpha=switch_alpha * 0.2,
                            count_visit=False,
                        )

                # ═══ KEPT: long air penalty ═══
                max_reasonable_air = 50
                if (
                    air_steps_after > max_reasonable_air
                    and not landed
                ):
                    self.strategic_detach.update_q_value(
                        t_state, action=1,
                        td_target=cfg.get(
                            "strategic_reward_switch_long_air", -0.5
                        ),
                        alpha=switch_alpha * 0.5,
                    )

        # ── Episodes WITHOUT detach ──
        if not had_detach:
            if goal_reached:
                # ═══ CHANGED: stay reward only for "decision zone" states ═══
                for tr in self._episode_transitions:
                    s = tr["state"]
                    if s[11] < 0.5:  # in air — skip
                        continue

                    ss = tr.get("same_side", True)
                    pb = float(tr.get("path_blocked", False))
                    eff = tr.get("movement_efficiency", 1.0)

                    # Only update states where detach could have been considered
                    could_have_detached = (
                        (not ss or pb > 0.5)
                        and eff < stuck_threshold
                    )

                    if could_have_detached:
                        t_state = self._compute_detach_state_from_full(
                            s, path_blocked=pb, movement_efficiency=eff,
                        )
                        self.strategic_detach.update_q_value(
                            t_state, action=0,  # stay
                            td_target=1.0,      # was 0.5
                            alpha=switch_alpha,  # was switch_alpha * 0.3
                        )

            elif termination_reason == "timeout":
                # ═══ CHANGED: only penalize stay, do NOT reward switch ═══
                for tr in self._episode_transitions:
                    s = tr["state"]
                    if s[11] < 0.5:
                        continue

                    ss = tr.get("same_side", True)
                    pb = float(tr.get("path_blocked", False))
                    eff = tr.get("movement_efficiency", 1.0)

                    could_have_detached = (
                        (not ss or pb > 0.5)
                        and eff < stuck_threshold
                    )

                    if could_have_detached:
                        t_state = self._compute_detach_state_from_full(
                            s, path_blocked=pb, movement_efficiency=eff,
                        )
                        self.strategic_detach.update_q_value(
                            t_state, action=0,  # stay was bad
                            td_target=-0.3,
                            alpha=switch_alpha * 0.5,
                            count_visit=False,
                        )
                        # ═══ REMOVED: no counterfactual switch reward ═══
                        # Was:
                        # self.strategic_detach.update_q_value(
                        #     t_state, action=1, td_target=0.3, ...
                        # )

        # ═══ Retrospective direction update ═══
        air_transitions = []
        for tr in self._episode_transitions:
            s = tr["state"]
            if s[11] > 0.5:  # on_object — skip surface
                continue
            air_transitions.append(tr)

        if air_transitions:
            dir_alpha = switch_alpha * 0.2

            # Sample representative transitions for insert
            # (first, last, every 10th) to populate store
            # without flooding it with redundant nearby points
            insert_indices = set()
            insert_indices.add(0)
            insert_indices.add(len(air_transitions) - 1)
            for i in range(0, len(air_transitions), 10):
                insert_indices.add(i)

            if goal_reached:
                for i, tr in enumerate(air_transitions):
                    d_state = self._compute_direction_state_from_full(
                        tr["state"],
                        path_blocked=float(
                            tr.get("path_blocked", False)
                        ),
                    )
                    pb = d_state[4]
                    allow_insert = i in insert_indices
                    if pb > 0.5:
                        self.strategic_direction.update_q_value(
                            d_state, action=1, td_target=0.5,
                            alpha=dir_alpha,
                            count_visit=allow_insert,
                        )
                    else:
                        self.strategic_direction.update_q_value(
                            d_state, action=0, td_target=0.5,
                            alpha=dir_alpha,
                            count_visit=allow_insert,
                        )

            elif termination_reason == "timeout":
                for i, tr in enumerate(air_transitions):
                    d_state = self._compute_direction_state_from_full(
                        tr["state"],
                        path_blocked=float(
                            tr.get("path_blocked", False)
                        ),
                    )
                    pb = d_state[4]
                    allow_insert = i in insert_indices
                    if pb > 0.5:
                        self.strategic_direction.update_q_value(
                            d_state, action=1, td_target=-0.2,
                            alpha=dir_alpha * 0.5,
                            count_visit=allow_insert,
                        )
                    else:
                        self.strategic_direction.update_q_value(
                            d_state, action=0, td_target=-0.2,
                            alpha=dir_alpha * 0.5,
                            count_visit=allow_insert,
                        )

            elif termination_reason in (
                "collision_surface_violation",
                "collision_other",
            ):
                for i, tr in enumerate(air_transitions):
                    d_state = self._compute_direction_state_from_full(
                        tr["state"],
                        path_blocked=float(
                            tr.get("path_blocked", False)
                        ),
                    )
                    pb = d_state[4]
                    allow_insert = i in insert_indices
                    if pb > 0.5:
                        self.strategic_direction.update_q_value(
                            d_state, action=1, td_target=-0.3,
                            alpha=dir_alpha * 0.3,
                            count_visit=allow_insert,
                        )
                    else:
                        self.strategic_direction.update_q_value(
                            d_state, action=0, td_target=-0.1,
                            alpha=dir_alpha * 0.3,
                            count_visit=allow_insert,
                        )

        # ═══ Update Strategic SAC ═══
        if (
            self.strategic_sac is not None
            and self._strategic_sac_pending
        ):
            for pending in self._strategic_sac_pending:
                s_state = pending["state"]
                action = pending["action"]
                reward = 0.0  # default

                if pending["phase"] == "detach":
                    if action == 1.0:
                        if goal_reached:
                            reward = 1.0
                        elif (
                            termination_reason
                            == "collision_surface_violation"
                        ):
                            reward = -0.5
                        else:
                            reward = -0.2
                    else:
                        if goal_reached:
                            reward = 0.5
                        else:
                            reward = -0.3

                next_s_state = s_state
                self.strategic_sac.add_transition(
                    state=s_state,
                    action=action,
                    reward=reward,
                    next_state=next_s_state,
                    done=True,
                )

            self._strategic_sac_pending = []

        # ═══ Episode done logging ═══
        if goal_reached:
            reason = "GOAL_REACHED!!!"
            reason_key = "goal_reached"
            source_str = f", source={self._current_source}" if self._current_source else ""
            logger.info(
                f"Episode {self._total_episodes} DONE: {reason}, "
                f"start_dist={start_distance:.1f}mm, "
                f"{self._steps} steps, "
                f"reward={self._episode_reward:.1f}, "
                f"final_dist={distance:.1f}mm, "
                f"epsilon={self.epsilon:.3f}, "
                f"strategic_eps={self.strategic_epsilon:.3f}, "
                f"success_rate="
                f"{self._total_goals_reached}/{self._total_episodes}"
                f"{source_str}"
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

        # ═══ Reset ═══
        self._current_goal = None
        self._current_max_extent = None
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
        self._cached_orbit_direction = None
        self._orbit_direction_age = 0
        self._fly_direction_age = 0
        self._current_phase = "CRAWL_TO_GOAL"
        self._prev_phase = None
        self._pending_strategic_detach = []
        self._path_clear_streak = 0
        self._strategic_sac_pending = []
        self._prev_subgoal_alignment = None
        self._prev_goal_alignment = None

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
            "strategic_epsilon": self.strategic_epsilon,
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

        stats["strategic_detach"] = self.strategic_detach.get_stats()
        stats["strategic_direction"] = self.strategic_direction.get_stats()

        # Strategic Q-store diagnostic
        if self.strategic_detach.next_id > 0:
            detach_diag = {"zones": {}}
            
            # Analyze Q-values by normal_agreement zones
            zones = [
                ("opposite", -1.0, -0.3),    # same_side=False
                ("perpendicular", -0.3, 0.3), # edge/corner
                ("same", 0.3, 1.0),           # same_side=True
            ]
            
            for zone_name, na_min, na_max in zones:
                zone_points = []
                for pid, point in self.strategic_detach.points.items():
                    na = point.raw_state[0]  # normal_agreement
                    if na_min <= na < na_max:
                        zone_points.append(point)
                
                if zone_points:
                    q_stays = [p.q_values[0] for p in zone_points]
                    q_switches = [p.q_values[1] for p in zone_points]
                    switch_preferred = sum(
                        1 for p in zone_points
                        if p.q_values[1] > p.q_values[0]
                    )
                    
                    detach_diag["zones"][zone_name] = {
                        "count": len(zone_points),
                        "q_stay_mean": round(float(np.mean(q_stays)), 4),
                        "q_switch_mean": round(float(np.mean(q_switches)), 4),
                        "switch_preferred_ratio": round(
                            switch_preferred / len(zone_points), 3
                        ),
                        "na_range": [na_min, na_max],
                    }
            
            stats["strategic_detach_diagnostic"] = detach_diag

        if self.strategic_direction.next_id > 0:
            direction_diag = {"zones": {}}

            # By path_blocked
            for pb_name, pb_val in [("blocked", 1.0), ("clear", 0.0)]:
                zone_points = []
                for pid, point in self.strategic_direction.points.items():
                    # path_blocked is index 4 in direction state (5D)
                    if len(point.raw_state) > 4:
                        pb = point.raw_state[4]
                        if abs(pb - pb_val) < 0.5:
                            zone_points.append(point)

                if zone_points:
                    q_stays = [p.q_values[0] for p in zone_points]
                    q_switches = [p.q_values[1] for p in zone_points]
                    switch_preferred = sum(
                        1 for p in zone_points
                        if p.q_values[1] > p.q_values[0]
                    )

                    direction_diag["zones"][pb_name] = {
                        "count": len(zone_points),
                        "q_stay_mean": round(float(np.mean(q_stays)), 4),
                        "q_switch_mean": round(float(np.mean(q_switches)), 4),
                        "switch_preferred_ratio": round(
                            switch_preferred / len(zone_points), 3
                        ),
                        # ═══ NEW: Q_stay breakdown ═══
                        "q_stay_positive_count": sum(
                            1 for q in q_stays if q > 0.01
                        ),
                        "q_stay_negative_count": sum(
                            1 for q in q_stays if q < -0.01
                        ),
                        "q_stay_zero_count": sum(
                            1 for q in q_stays
                            if abs(q) <= 0.01
                        ),
                    }

            # By angle_to_goal zones
            angle_zones = [
                ("away", -1.0, 0.0),      # flying away from goal
                ("sideways", 0.0, 0.5),    # flying sideways
                ("toward", 0.5, 1.0),      # flying toward goal
            ]

            for zone_name, angle_min, angle_max in angle_zones:
                zone_points = []
                for pid, point in self.strategic_direction.points.items():
                    # angle_to_goal is index 3 in direction state (5D)
                    if len(point.raw_state) > 3:
                        angle = point.raw_state[3]
                        if angle_min <= angle < angle_max:
                            zone_points.append(point)

                if zone_points:
                    q_stays = [p.q_values[0] for p in zone_points]
                    q_switches = [p.q_values[1] for p in zone_points]
                    switch_preferred = sum(
                        1 for p in zone_points
                        if p.q_values[1] > p.q_values[0]
                    )

                    direction_diag["zones"][f"angle_{zone_name}"] = {
                        "count": len(zone_points),
                        "q_stay_mean": round(float(np.mean(q_stays)), 4),
                        "q_switch_mean": round(float(np.mean(q_switches)), 4),
                        "switch_preferred_ratio": round(
                            switch_preferred / len(zone_points), 3
                        ),
                        # ═══ NEW: Q_stay breakdown ═══
                        "q_stay_positive_count": sum(
                            1 for q in q_stays if q > 0.01
                        ),
                        "q_stay_negative_count": sum(
                            1 for q in q_stays if q < -0.01
                        ),
                        "q_stay_zero_count": sum(
                            1 for q in q_stays
                            if abs(q) <= 0.01
                        ),
                    }

            stats["strategic_direction_diagnostic"] = direction_diag

        # Strategic level stats
        total_detach_decisions = max(
            self._strategic_stats["detach_memory_triggered"]
            + self._strategic_stats["detach_memory_suppressed"]
            + self._strategic_stats["detach_heuristic_fallback"],
            1,
        )
        total_direction_decisions = max(
            self._strategic_stats["direction_memory_to_goal"]
            + self._strategic_stats["direction_memory_keep_edge"]
            + self._strategic_stats["direction_heuristic_fallback"],
            1,
        )
        total_detach_outcomes = max(
            self._strategic_stats["detach_total"], 1
        )

        stats["strategic"] = {
            "detach_decisions": {
                "memory_triggered": self._strategic_stats[
                    "detach_memory_triggered"
                ],
                "memory_suppressed": self._strategic_stats[
                    "detach_memory_suppressed"
                ],
                "heuristic_fallback": self._strategic_stats[
                    "detach_heuristic_fallback"
                ],
                "memory_usage_rate": round(
                    (
                        self._strategic_stats["detach_memory_triggered"]
                        + self._strategic_stats["detach_memory_suppressed"]
                    )
                    / total_detach_decisions,
                    3,
                ),
            },
            "detach_outcomes": {
                "total": self._strategic_stats["detach_total"],
                "success": self._strategic_stats[
                    "detach_led_to_success"
                ],
                "collision": self._strategic_stats[
                    "detach_led_to_collision"
                ],
                "timeout": self._strategic_stats[
                    "detach_led_to_timeout"
                ],
                "success_rate": round(
                    self._strategic_stats["detach_led_to_success"]
                    / total_detach_outcomes,
                    3,
                ),
            },
            "direction_decisions": {
                "memory_to_goal": self._strategic_stats[
                    "direction_memory_to_goal"
                ],
                "memory_keep_edge": self._strategic_stats[
                    "direction_memory_keep_edge"
                ],
                "heuristic_fallback": self._strategic_stats[
                    "direction_heuristic_fallback"
                ],
                "memory_usage_rate": round(
                    (
                        self._strategic_stats[
                            "direction_memory_to_goal"
                        ]
                        + self._strategic_stats[
                            "direction_memory_keep_edge"
                        ]
                    )
                    / total_direction_decisions,
                    3,
                ),
            },
            "phase_distribution": self._strategic_stats[
                "phase_counts"
            ],
            "strategic_detach_store": (
                self.strategic_detach.get_stats()
            ),
            "strategic_direction_store": (
                self.strategic_direction.get_stats()
            ),
        }

        if self.strategic_sac is not None:
            stats["strategic_sac"] = self.strategic_sac.get_stats()
            
        # Action source summary: strategic vs tactical
        strategic_actions = (
            self._strategic_stats[
                "detach_memory_triggered"
            ]
            + self._strategic_stats[
                "direction_memory_to_goal"
            ]
            + self._strategic_stats[
                "direction_memory_keep_edge"
            ]
        )
        tactical_actions = max(
            self._total_steps - strategic_actions, 0
        )
        total_decisions = max(
            strategic_actions + tactical_actions, 1
        )
        stats["action_source_summary"] = {
            "strategic_total": strategic_actions,
            "strategic_detach": self._strategic_stats[
                "detach_memory_triggered"
            ],
            "strategic_direction": (
                self._strategic_stats[
                    "direction_memory_to_goal"
                ]
                + self._strategic_stats[
                    "direction_memory_keep_edge"
                ]
            ),
            "tactical_total": tactical_actions,
            "strategic_rate": round(
                strategic_actions / total_decisions,
                4,
            ),
            "heuristic_fallback_total": (
                self._strategic_stats[
                    "detach_heuristic_fallback"
                ]
                + self._strategic_stats[
                    "direction_heuristic_fallback"
                ]
            ),
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
                state, self._prev_state, self._last_action, collision,
                sensor_data=sensor_data,
                prev_sensor_data=self._prev_sensor_data,
                current_pose=current_pose,
            )
            self._episode_reward += reward
            self._episode_transitions.append(
                {
                    "state": self._prev_state.copy(),
                    "action": int(self._last_action),
                    "reward": float(reward),
                    "path_blocked": sensor_data.get("path_blocked", False),
                    "same_side": sensor_data.get("same_side", True),
                    "movement_efficiency": self._compute_movement_efficiency(window=20),
                }
            )

            if self.is_training:
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
            self.strategic_epsilon = self.config.get("strategic_eval_epsilon", 0.3)

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

        # Save transition memories
        self.strategic_detach.save_with_index(
            os.path.join(dirpath, "strategic_detach")
        )
        self.strategic_direction.save_with_index(
            os.path.join(dirpath, "strategic_direction")
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

        if self.strategic_sac is not None:
            self.strategic_sac.save(
                os.path.join(dirpath, "strategic_sac")
            )

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

        free_cfg = {
            **(config or {}),
            "feature_weights": (config or {}).get(
                "feature_weights_free", None
            ),
        }
        controller.q_store_free = (
            HNSWStateStore.load_with_index(
                os.path.join(
                    dirpath, "q_store_free"
                ),
                extra_cfg=free_cfg,
            )
        )

        surface_cfg = {
            **(config or {}),
            "feature_weights": (config or {}).get(
                "feature_weights_surface", None
            ),
        }
        controller.q_store_surface = (
            HNSWStateStore.load_with_index(
                os.path.join(
                    dirpath, "q_store_surface"
                ),
                extra_cfg=surface_cfg,
            )
        )

        # Load transition memories
        detach_base = os.path.join(dirpath, "strategic_detach")
        if pathlib.Path(detach_base + ".npz").exists():
            s_detach_cfg = {
                **(config or {}),
                "state_dim": 5,  # CHANGED: was 4
                "num_actions": 2,
                "max_points": cfg.get("transition_max_points", 10000),
                "k_neighbors": cfg.get("transition_k_neighbors", 5),
            }
            controller.strategic_detach = HNSWStateStore.load_with_index(
                detach_base, extra_cfg=s_detach_cfg
            )

        direction_base = os.path.join(dirpath, "strategic_direction")
        if pathlib.Path(direction_base + ".npz").exists():
            s_dir_cfg = {
                **(config or {}),
                "state_dim": 5,
                "num_actions": 2,
                "max_points": cfg.get("transition_max_points", 10000),
                "k_neighbors": cfg.get("transition_k_neighbors", 5),
            }
            controller.strategic_direction = HNSWStateStore.load_with_index(
                direction_base, extra_cfg=s_dir_cfg
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
        strategic_sac_path = os.path.join(
            dirpath, "strategic_sac"
        )

        if (
            pathlib.Path(strategic_sac_path).exists()
            and (
                pathlib.Path(strategic_sac_path)
                / "strategic_actor.pt"
            ).exists()
        ):
            controller.strategic_sac = StrategicSAC.load(
                strategic_sac_path
            )

        logger.info(
            f"Controller loaded from {dirpath}: "
            f"{loaded_total_episodes} episodes, "
            f"{loaded_total_steps} steps, "
            f"{loaded_total_goals_reached} goals, "
            f"epsilon={loaded_epsilon:.3f}, "
            f"strategic_detach="
            f"{controller.strategic_detach.get_stats()}, "
            f"strategic_direction="
            f"{controller.strategic_direction.get_stats()}"
        )

        return controller
