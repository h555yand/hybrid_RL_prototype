"""RLGoalPolicy: drop-in replacement for JumpToGoal using RL navigation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import quaternion as qt
from scipy.spatial.transform import Rotation as Rot

from tbp.monty.cmp import Goal, Message
from tbp.monty.context import RuntimeContext
from tbp.monty.experiment.motor_system import ExperimentMotorSystem
from tbp.monty.frameworks.actions.actions import SetAgentPose, SetSensorRotation
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_policies import (
    MotorPolicy,
    MotorPolicyResult,
    PolicyStatus,
)
from tbp.monty.frameworks.models.motor_system_state import MotorSystemState
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.memento import Memento

from tbp.hybrid_rl.arbitrator import sac_to_discrete
from tbp.hybrid_rl.experience_extractor import ExperienceExtractor
from tbp.hybrid_rl.monty_rl_bridge import MontyRLBridge
from tbp.hybrid_rl.rl_goal_approach_controller import RLGoalApproachController

if TYPE_CHECKING:
    from tbp.hybrid_rl.adaptive_manager import AdaptiveTrainingManager
    from tbp.hybrid_rl.mujoco_env_adapter import MuJoCoEnvAdapter

logger = logging.getLogger(__name__)


class RLGoalPolicy(MotorPolicy):

    def __init__(
        self,
        agent_id: AgentID,
        sensor_id: SensorID,
        model_path: str,
        rl_config: Dict[str, Any],
        mujoco_adapter: MuJoCoEnvAdapter,
        mesh_path: Optional[str] = None,
        max_nav_steps: int = 200,
        enable_online_learning: bool = True,
    ):
        self._agent_id = agent_id
        self._sensor_id = sensor_id
        self._max_nav_steps = max_nav_steps
        self._enable_learning = enable_online_learning

        self._observe_every_n_steps: int = rl_config.get("observe_every_n_steps", 0)
        # 0 = only final observation (Phase 1 behavior)
        # 1 = every step (full directed exploration, slowest)
        # 5 = every 5th step (balanced)
        # 10 = every 10th step (fast)

        self._bridge = MontyRLBridge(agent_id, rl_config, mujoco_adapter)

        self._controller = RLGoalApproachController.load(
            model_path,
            agent_id=str(agent_id),
            config={**rl_config, "mode": "adaptive"},
        )
        self._controller.mode = "adaptive"

        self._manager: Optional[AdaptiveTrainingManager] = None
        if enable_online_learning:
            from tbp.hybrid_rl.adaptive_manager import AdaptiveTrainingManager

            if mesh_path:
                from tbp.hybrid_rl.lightweight_env import LightweightEnv
                trimesh_env = LightweightEnv(mesh_path)
            else:
                trimesh_env = None

            self._manager = AdaptiveTrainingManager(
                controller=self._controller,
                env=trimesh_env,
                config=rl_config,
                runs_dir=str(Path(model_path).parent),
                mesh_path=mesh_path,
            )

            if not mesh_path:
                logger.info(
                    "AdaptiveManager: inference-only mode "
                    "(no mesh -> no offline retrain, SAC inference OK)"
                )

        self._nav_active: bool = False
        self._nav_steps: int = 0
        self._current_goal: Optional[Goal] = None
        self._last_termination: Optional[str] = None
        self._prev_goals_reached: int = 0
        self._monty_goal_pose_mm: Optional[np.ndarray] = None
        self._rl_goal_pose_mm: Optional[np.ndarray] = None
        # Directed exploration state
        self._last_observe_pos: Optional[np.ndarray] = None
        self._last_observe_step: int = 0
        self._last_observe_normal: Optional[np.ndarray] = None
        self._last_observe_k1: Optional[float] = None
        self._stuck_counter: int = 0

        # Logging
        self._log_dir = Path(model_path).parent / "monty_integration_logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._episode_log: list[dict] = []
        self._total_nav_episodes: int = 0
        self._current_poses: list = []
        self._action_explanations: list[str] = []
        self._action_short_labels: list[str] = []
        self._start_distance: float = 0.0
        self._monty_start_pos_mm: Optional[np.ndarray] = None
        self._current_object_name: str = "unknown"
        self._experiment_ref = None

        # Visualizer
        self._visualizer = None
        try:
            from tbp.hybrid_rl.visualize_env import EpisodeVisualizer
            self._visualizer = EpisodeVisualizer(
                output_dir=self._log_dir,
                mesh_name="monty_nav",
                stage="monty_integration",
                max_per_type_per_level=10,
                num_levels=3,
                visualize_mode="pictures",
            )
        except Exception:
            pass

        logger.info(
            "RLGoalPolicy initialized: agent=%s, model=%s, "
            "max_steps=%d, online_learning=%s",
            agent_id, model_path, max_nav_steps, enable_online_learning,
        )

    def __call__(self, ctx, observations, state, percept, goal):
        if self._nav_active and not self._controller.is_active:
            return self._finish_navigation()

        if goal is not None and not self._nav_active:
            if goal.location is None:
                return MotorPolicyResult([])
            return self._start_navigation(state, percept, goal)

        if self._nav_active:
            return self._navigation_step(state, percept)

        return MotorPolicyResult([])

    def fixme_provide_motor_system(self, motor_system):
        pass

    def reset(self):
        self._nav_active = False
        self._nav_steps = 0
        self._current_goal = None
        self._last_termination = None
        self._prev_goals_reached = self._controller._total_goals_reached
        self._monty_goal_pose_mm = None
        self._rl_goal_pose_mm = None
        self._monty_start_euler = None
        self._last_observe_pos = None
        self._last_observe_step = 0
        self._last_observe_normal = None
        self._last_observe_k1 = None
        self._stuck_counter = 0
        # ═══ NEW: Ensure GSG is re-enabled on reset ═══
        self._set_gsg_navigation_active(False)

    def state_dict(self):
        return {"nav_active": self._nav_active, "nav_steps": self._nav_steps}

    def load_state_dict(self, memento):
        self._nav_active = memento.get("nav_active", False)
        self._nav_steps = memento.get("nav_steps", 0)

    @property
    def is_navigating(self):
        return self._nav_active

    def get_stats(self):
        stats = {
            "nav_active": self._nav_active,
            "nav_steps": self._nav_steps,
            "last_termination": self._last_termination,
            "controller": self._controller.get_stats(),
        }
        if self._manager:
            stats["adaptive"] = self._manager.get_stats()
            stats["arbitrator"] = self._manager.arbitrator.get_stats()
        return stats

    def set_object_name(self, name: str):
        """Set current object name for logging."""
        self._current_object_name = name

    def set_experiment_ref(self, experiment):
        """Store reference to Monty experiment for object name lookup."""
        self._experiment_ref = experiment

    def _get_object_name_from_experiment(self) -> str:
        """Get current object name from Monty experiment."""
        exp = self._experiment_ref
        if exp is None:
            return ""

        try:
            return str(exp.logger_args["target"]["object"])
        except (AttributeError, KeyError, TypeError):
            return ""
                
    def _start_navigation(self, state, percept, goal):
        self._current_goal = goal
        self._nav_active = True
        self._nav_steps = 0
        self._last_termination = None
        self._prev_goals_reached = self._controller._total_goals_reached
        
        # ═══ Auto-detect object name from experiment ═══
        if self._experiment_ref is not None:
            try:
                obj_name = self._get_object_name_from_experiment()
                if obj_name:
                    self._current_object_name = obj_name
            except Exception:
                pass

        self._bridge.set_goal(goal)

        # ═══ NEW: Suppress GSG goal generation during RL navigation ═══
        self._set_gsg_navigation_active(True)

        # ═══ 1) Save original Monty goal and start position ═══
        goal_pose_mm = self._bridge.goal_to_pose_mm(goal)
        self._monty_goal_pose_mm = goal_pose_mm.copy()
        self._monty_start_pos_mm = np.array(
            state[self._agent_id].position, dtype=float
        ) * 1000.0
        self._monty_start_euler = self._bridge._adapter._get_euler_deg().copy()

        # Update adapter center for shared sim
        # ═══ CHANGED: Always re-discover object geometry ═══
        # Objects change between episodes (master_chef_can → cracker_box),
        # but adapter keeps center/extents from previous object.
        # Old logic skipped discover if goal was near old center,
        # causing wrong center for new object.
        #
        # Fix: always discover, using midpoint between agent and goal
        # as hint (closer to true center than goal on object edge).
        if not self._bridge._adapter._owns_sim:
            agent_pos_mm = self._bridge._adapter._get_pos_mj_mm()
            goal_hint = (goal_pose_mm[:3] + agent_pos_mm) / 2.0

            self._bridge._adapter.discover_object(hint_pos_mm=goal_hint)

            logger.info(
                "  object props: center=%s extents=%s",
                self._bridge._adapter._mj_center_mm.round(1).tolist(),
                self._bridge._adapter._mj_extents_mm.round(1).tolist(),
            )

        # ═══ 2) Snap agent to surface ═══
        original_snap = self._bridge._adapter._snap_max_dist
        self._bridge._adapter._snap_max_dist = 35.0
        snap_ok = self._bridge._adapter._snap_to_surface()
        self._bridge._adapter._snap_max_dist = original_snap

        start_pos_mm = self._bridge._adapter._get_pos_mj_mm()
        start_sensor = self._bridge._adapter.get_sensor_data()

        logger.info(
            "  agent snap: ok=%s depth=%.1f on_object=%s",
            snap_ok,
            start_sensor.get("depth", -1),
            start_sensor.get("on_object", False),
        )
        
        # ═══ NEW: Validate snap — agent must be on surface ═══
        if not start_sensor.get("on_object", False):
            logger.info(
                "  snap validation FAILED: depth=%.1f, on_object=%s. "
                "Re-snapping toward object center.",
                start_sensor.get("depth", -1),
                start_sensor.get("on_object", False),
            )

            center = self._bridge._adapter._mj_center_mm
            pos = self._bridge._adapter._get_pos_mj_mm()
            to_center = center - pos
            to_center_len = float(np.linalg.norm(to_center))

            if to_center_len > 1e-8:
                to_center_dir = to_center / to_center_len
                euler = self._bridge._adapter._look_at_direction(to_center_dir)
                self._bridge._adapter._set_pose_mj_mm(pos, euler)

                self._bridge._adapter._snap_max_dist = 50.0
                snap_ok = self._bridge._adapter._snap_to_surface()
                self._bridge._adapter._snap_max_dist = original_snap

                start_pos_mm = self._bridge._adapter._get_pos_mj_mm()
                start_sensor = self._bridge._adapter.get_sensor_data()

                logger.info(
                    "  re-snap result: ok=%s depth=%.1f on_object=%s",
                    snap_ok,
                    start_sensor.get("depth", -1),
                    start_sensor.get("on_object", False),
                )

        # ═══ 3) Snap goal to surface ═══
        goal_pos = goal_pose_mm[:3].copy()

        agent_pos_backup = self._bridge._adapter._get_pos_mj_mm().copy()
        agent_euler_backup = self._bridge._adapter._get_euler_deg().copy()

        obj_center_monty = np.array([0.0, 1500.0, 0.0])
        to_obj = obj_center_monty - goal_pos
        to_obj_len = float(np.linalg.norm(to_obj))
        if to_obj_len > 1e-8:
            to_obj_dir = to_obj / to_obj_len
        else:
            to_obj_dir = np.array([0, 0, -1.0])

        goal_euler = self._bridge._adapter._look_at_direction(to_obj_dir)
        self._bridge._adapter._set_pose_mj_mm(goal_pos, goal_euler)

        self._bridge._adapter._snap_max_dist = 200.0
        goal_snap_ok = self._bridge._adapter._snap_to_surface()
        self._bridge._adapter._snap_max_dist = original_snap

        if goal_snap_ok:
            goal_surface_pos = self._bridge._adapter._get_pos_mj_mm().copy()
            goal_surface_euler = self._bridge._adapter._get_euler_deg().copy()
        else:
            goal_surface_pos = goal_pos
            goal_surface_euler = goal_euler

        goal_pose_mm_surface = goal_pose_mm.copy()
        goal_pose_mm_surface[:3] = goal_surface_pos
        goal_pose_mm_surface[3:6] = goal_surface_euler

        self._bridge._adapter._set_pose_mj_mm(agent_pos_backup, agent_euler_backup)

        self._bridge._adapter._snap_max_dist = 35.0
        self._bridge._adapter._snap_to_surface()
        self._bridge._adapter._snap_max_dist = original_snap

        start_pos_mm = self._bridge._adapter._get_pos_mj_mm()
        start_sensor = self._bridge._adapter.get_sensor_data()

        self._rl_goal_pose_mm = goal_pose_mm_surface.copy()

        logger.info(
            "  goal snap: ok=%s monty=[%.1f,%.1f,%.1f] surface=[%.1f,%.1f,%.1f]",
            goal_snap_ok, *goal_pos, *goal_surface_pos,
        )

        # ═══ 4) Set RL goal (on surface) and start navigation ═══
        self._controller.set_new_goal(goal_pose_mm_surface, start_pos_mm)
        self._bridge._adapter.set_goal(goal_pose_mm_surface)

        if self._manager:
            self._manager.arbitrator.start_episode(level=0)

        dist = float(np.linalg.norm(goal_pose_mm_surface[:3] - start_pos_mm))

        self._current_poses = [self._bridge._adapter.get_pose().copy()]
        self._action_explanations = []
        self._action_short_labels = []
        self._start_distance = dist

        logger.info(
            "RLGoalPolicy: navigation started\n"
            "  monty_start=[%.1f, %.1f, %.1f]mm\n"
            "  rl_start   =[%.1f, %.1f, %.1f]mm (depth=%.1f, on_obj=%s)\n"
            "  monty_goal =[%.1f, %.1f, %.1f]mm\n"
            "  rl_goal    =[%.1f, %.1f, %.1f]mm\n"
            "  dist=%.1fmm",
            *self._monty_start_pos_mm,
            *start_pos_mm, start_sensor.get("depth", -1), start_sensor.get("on_object", False),
            *self._monty_goal_pose_mm[:3],
            *goal_pose_mm_surface[:3],
            dist,
        )
        # Reset directed exploration state
        self._last_observe_pos = None
        self._last_observe_step = 0
        self._last_observe_normal = None
        self._last_observe_k1 = None
        self._stuck_counter = 0

        return self._navigation_step(state, percept)

    def _navigation_step(self, state, percept):
        # ═══ Restore after observation lift ═══
        if getattr(self, '_lifted_for_observation', False):
            self._bridge._adapter._set_pose_mj_mm(
                self._pre_lift_pos, self._pre_lift_euler
            )
            self._lifted_for_observation = False

        self._nav_steps += 1

        current_pose_mm = self._bridge._adapter.get_pose()
        sensor_data = self._bridge._adapter.get_sensor_data()

        rl_state = self._controller._compute_state(current_pose_mm, sensor_data)

        if self._manager:
            debug_info = self._controller.get_state_debug_info(
                rl_state, current_pose_mm, sensor_data,
            )
            action_type, action_params, source = self._manager.get_action(
                rl_state, current_pose_mm, sensor_data,
            )
            action_idx = sac_to_discrete(action_type, action_params)

            post_sensor_data = self._bridge.execute_continuous_action(
                action_type, action_params,
            )

            # Per-step logging
            type_names = ExperienceExtractor.get_type_names()
            act_name = type_names.get(action_type, f"type_{action_type}")

            adapter_pos = self._bridge._adapter.get_pose()
            self._current_poses.append(adapter_pos.copy())

            rl_goal = self._rl_goal_pose_mm
            dist_to_goal = float(np.linalg.norm(
                rl_goal[:3] - adapter_pos[:3]
            ))

            self._action_explanations.append(
                f"{act_name} {debug_info} | {source}"
            )
            self._action_short_labels.append(
                f"Step {self._nav_steps:03d} | dist={dist_to_goal:.1f}mm | {act_name}"
            )

            if self._enable_learning:
                post_pose_mm = self._bridge._adapter.get_pose()
                _, done = self._controller.update_only(
                    post_pose_mm, post_sensor_data, action_idx,
                )
                if done:
                    return self._finish_navigation()

        else:
            action_name, _ = self._controller.step(
                current_pose_mm, sensor_data,
            )
            if action_name is None:
                return self._finish_navigation()

            action_idx = self._controller._last_action
            post_sensor_data = self._bridge.execute_discrete_action(
                action_idx, self._controller.action_space,
            )

        if self._nav_steps >= self._max_nav_steps:
            if self._controller.is_active:
                post_pose = self._bridge._adapter.get_pose()
                self._controller._on_episode_done(
                    self._controller._compute_state(post_pose, post_sensor_data),
                    "timeout",
                )
            return self._finish_navigation()

        # ═══ Directed exploration: lift for Monty observation ═══
        if self._observe_every_n_steps > 0 and self._should_observe():
            self._pre_lift_pos = self._bridge._adapter._get_pos_mj_mm().copy()
            self._pre_lift_euler = self._bridge._adapter._get_euler_deg().copy()

            rot_mat = Rot.from_euler("xyz", self._pre_lift_euler, degrees=True)
            backward = -rot_mat.apply([0, 0, -1])
            lifted_pos = self._pre_lift_pos + backward * 23.0
            self._bridge._adapter._set_pose_mj_mm(lifted_pos, self._pre_lift_euler)
            self._lifted_for_observation = True

            actual_pos = self._bridge._adapter._get_pos_mj_mm()
            logger.info(
                "DIRECTED_EXPLORATION: nav_step=%d, lifted_pos=[%.1f,%.1f,%.1f], "
                "actual_mj_pos=[%.1f,%.1f,%.1f], trigger=adaptive",
                self._nav_steps, *lifted_pos, *actual_pos,
            )

            return MotorPolicyResult(
                actions=[],
                motor_only_step=False,
                status=PolicyStatus.IN_PROGRESS,
            )

        return MotorPolicyResult(
            actions=[],
            motor_only_step=True,
            status=PolicyStatus.IN_PROGRESS,
        )
    
    def _should_observe(self) -> bool:
        """Decide whether to send observation to LM on this step.

        Adaptive strategy:
        - Filter: off-object, bad depth, stuck agent
        - Trigger: surface normal change (new face = high information)
        - Trigger: curvature change (different surface region)
        - Adaptive interval: more frequent near goal
        """
        
        if self._rl_goal_pose_mm is None:
            return False

        sensor_data = self._bridge._adapter.get_sensor_data()
        current_pos = self._bridge._adapter._get_pos_mj_mm()

        # ═══ Filter 1: Must be on object ═══
        if not sensor_data.get("on_object", False):
            logger.debug("OBSERVE_SKIP: not on object")
            return False

        # ═══ Filter 2: Depth must be reasonable ═══
        depth = sensor_data.get("depth", -1)
        if depth < 0 or depth > 15:
            logger.debug("OBSERVE_SKIP: bad depth=%.1f", depth)
            return False

        # ═══ Filter 3: Must have moved enough ═══
        if self._last_observe_pos is not None:
            moved = float(np.linalg.norm(
                current_pos[:3] - self._last_observe_pos[:3]
            ))
            if moved < 3.0:
                self._stuck_counter += 1
                if self._stuck_counter > 3:
                    logger.debug(
                        "OBSERVE_SKIP: stuck (moved=%.1fmm, count=%d)",
                        moved, self._stuck_counter,
                    )
                return False
            else:
                self._stuck_counter = 0

        # ═══ Get current normal ═══
        raw_normal = sensor_data.get("point_normal", None)
        current_normal = None
        if raw_normal is not None:
            current_normal = np.array(raw_normal, dtype=float)
            n_len = np.linalg.norm(current_normal)
            if n_len > 0.1:
                current_normal = current_normal / n_len
            else:
                current_normal = None

        # ═══ Trigger 1: Normal changed (new face) ═══
        if current_normal is not None and self._last_observe_normal is not None:
            dot = float(np.clip(
                np.dot(current_normal, self._last_observe_normal), -1, 1
            ))
            normal_change = 1.0 - abs(dot)
            if normal_change > 0.3:
                logger.info(
                    "OBSERVE_TRIGGER: normal changed %.2f at step %d",
                    normal_change, self._nav_steps,
                )
                self._update_observe_state(
                    current_pos, current_normal, sensor_data
                )
                return True

        # ═══ Trigger 2: Curvature changed ═══
        current_k1 = sensor_data.get("k1", None)
        if (
            current_k1 is not None
            and self._last_observe_k1 is not None
            and current_k1 != 0
        ):
            k1_change = abs(current_k1 - self._last_observe_k1)
            if k1_change > 0.5:
                logger.info(
                    "OBSERVE_TRIGGER: curvature changed %.2f at step %d",
                    k1_change, self._nav_steps,
                )
                self._update_observe_state(
                    current_pos, current_normal, sensor_data
                )
                return True

        # ═══ Adaptive interval by distance to goal ═══
        dist_to_goal = float(np.linalg.norm(
            self._rl_goal_pose_mm[:3] - current_pos[:3]
        ))

        if dist_to_goal > 80:
            interval = 10
        elif dist_to_goal > 40:
            interval = 7
        elif dist_to_goal > 15:
            interval = 4
        else:
            interval = 2

        steps_since_last = self._nav_steps - self._last_observe_step
        if steps_since_last >= interval:
            logger.info(
                "OBSERVE_TRIGGER: interval=%d, dist=%.1fmm at step %d",
                interval, dist_to_goal, self._nav_steps,
            )
            self._update_observe_state(
                current_pos, current_normal, sensor_data
            )
            return True

        return False

    def _update_observe_state(
        self,
        pos: np.ndarray,
        normal: Optional[np.ndarray],
        sensor_data: dict,
    ) -> None:
        """Update tracking state after deciding to observe."""
        self._last_observe_pos = pos.copy()
        self._last_observe_step = self._nav_steps
        if normal is not None:
            self._last_observe_normal = normal.copy()
        k1 = sensor_data.get("k1", None)
        if k1 is not None:
            self._last_observe_k1 = k1

    def _finish_navigation(self):
        success = False
        termination = "unknown"
        self._total_nav_episodes += 1

        prev_goals = self._prev_goals_reached
        current_goals = self._controller._total_goals_reached
        if current_goals > prev_goals:
            success = True
            termination = "goal_reached"
        elif self._nav_steps >= self._max_nav_steps:
            termination = "timeout"
        elif not self._controller.is_active:
            termination = "collision"
        self._prev_goals_reached = current_goals

        if self._manager:
            self._manager.on_episode_complete(success=success, transitions=[])
            self._manager.arbitrator.on_episode_end(success)

        # Compute final distance
        rl_goal = (
            self._rl_goal_pose_mm
            if self._rl_goal_pose_mm is not None
            else np.zeros(6)
        )
        final_pos = self._bridge._adapter.get_pose()[:3]
        final_dist = float(np.linalg.norm(rl_goal[:3] - final_pos))

        logger.info(
            "RLGoalPolicy ep %d [%s]: %s after %d steps, "
            "dist %.1f→%.1fmm",
            self._total_nav_episodes, self._current_object_name,
            termination, self._nav_steps,
            self._start_distance, final_dist,
        )

        # ═══ Build enriched header for actions.txt ═══
        actions_header_lines = [
            f"source: monty_integration",
            f"Object: {self._current_object_name}",
            f"Result: {termination}",
            f"Steps: {self._nav_steps}",
            f"RL Goal (surface): {rl_goal.tolist()}",
        ]
        if self._monty_goal_pose_mm is not None:
            actions_header_lines.append(
                f"Monty Goal (original): "
                f"{self._monty_goal_pose_mm.tolist()}"
            )
        actions_header_lines.extend([
            f"Start distance: {self._start_distance:.1f}mm",
            f"End distance: {final_dist:.1f}mm",
            f"",
            f"=== Coordinate Debug ===",
            f"Adapter object center: "
            f"{self._bridge._adapter._mj_center_mm.round(1).tolist()}",
            f"Adapter object extents: "
            f"{self._bridge._adapter._mj_extents_mm.round(1).tolist()}",
            f"Adapter agent final pos: "
            f"{self._bridge._adapter._get_pos_mj_mm().round(1).tolist()}",
        ])
        final_sd = self._bridge._adapter.get_sensor_data()
        actions_header_lines.extend([
            f"Adapter agent final depth: "
            f"{final_sd.get('depth', -1):.1f}mm",
            f"Adapter agent on_object: "
            f"{final_sd.get('on_object', False)}",
        ])
        if len(self._current_poses) > 0:
            actions_header_lines.append(
                f"First adapter pos: "
                f"{np.array(self._current_poses[0][:3]).round(1).tolist()}"
            )
        if len(self._current_poses) > 1:
            actions_header_lines.append(
                f"Last adapter pos: "
                f"{np.array(self._current_poses[-1][:3]).round(1).tolist()}"
            )
        if self._monty_start_pos_mm is not None:
            actions_header_lines.extend([
                f"",
                f"=== Monty Original ===",
                f"Monty agent pos (mm): "
                f"{self._monty_start_pos_mm.round(1).tolist()}",
            ])
        if self._monty_goal_pose_mm is not None:
            actions_header_lines.append(
                f"Monty goal pos (mm): "
                f"{self._monty_goal_pose_mm[:3].round(1).tolist()}"
            )
        actions_header = "\n".join(actions_header_lines)

        # Episode log
        ep_entry = {
            "episode": self._total_nav_episodes,
            "object": self._current_object_name,
            "success": success,
            "termination": termination,
            "steps": self._nav_steps,
            "start_distance": round(self._start_distance, 1),
            "final_distance": round(final_dist, 1),
        }
        self._episode_log.append(ep_entry)

        log_path = self._log_dir / "episode_log.json"
        with log_path.open("w") as f:
            json.dump(self._episode_log, f, indent=2)

        # Visualizer (single source of truth for actions.txt)
        if self._visualizer is not None and self._current_goal is not None:
            try:
                viz_result = {
                    "goal_reached": "success",
                    "timeout": "timeout",
                    "collision": "collision",
                    "unknown": "timeout",
                }.get(termination, "timeout")

                self._visualizer.save_episode(
                    env=self._bridge._adapter,
                    episode=self._total_nav_episodes - 1,
                    level=0,
                    result=viz_result,
                    goal_pose=rl_goal,
                    poses=self._current_poses,
                    actions=self._action_explanations,
                    actions_short=self._action_short_labels,
                    extra_info={
                        "source": "monty_integration",
                        "object": self._current_object_name,
                    },
                    actions_header=actions_header,
                )
            except Exception as e:
                logger.warning("Visualizer failed: %s", e, exc_info=True)

        # ═══ Return agent to Monty position ═══
        if success:
            # Success: move agent to Monty goal position (30mm from surface)
            final_pos_mm = self._monty_goal_pose_mm[:3]
            final_euler = self._monty_goal_pose_mm[3:6]
            observe = True
        else:
            # Failure: agent is still on surface at some point.
            # Instead of returning to start silently, observe at current
            # position — it's still a valid surface observation.
            current_pos = self._bridge._adapter._get_pos_mj_mm()
            current_euler = self._bridge._adapter._get_euler_deg()
            current_sensor = self._bridge._adapter.get_sensor_data()

            if current_sensor.get("on_object", False) and current_sensor.get("depth", 100) < 15:
                # Agent is on surface — lift and observe here
                rot_mat = Rot.from_euler("xyz", current_euler, degrees=True)
                backward = -rot_mat.apply([0, 0, -1])
                lifted_pos = current_pos + backward * 23.0

                final_pos_mm = lifted_pos
                final_euler = current_euler
                observe = True

                logger.info(
                    "RLGoalPolicy: failed navigation, observing at "
                    "current pos=[%.1f,%.1f,%.1f] (depth=%.1f)",
                    *current_pos, current_sensor.get("depth", -1),
                )
            else:
                # Agent is off surface — return to start, no observation
                final_pos_mm = self._monty_start_pos_mm
                final_euler = self._monty_start_euler
                observe = False

                logger.info(
                    "RLGoalPolicy: failed navigation, off surface, "
                    "returning to start",
                )

        rot = Rot.from_euler("xyz", final_euler, degrees=True)
        q = rot.as_quat()
        final_quat = qt.quaternion(q[3], q[0], q[1], q[2])

        actions = [
            SetAgentPose(
                agent_id=self._agent_id,
                location=tuple(final_pos_mm / 1000.0),
                rotation_quat=final_quat,
            ),
            SetSensorRotation(
                agent_id=self._agent_id,
                rotation_quat=qt.one,
            ),
        ]
        
        # ═══ NEW: Re-enable GSG goal generation ═══
        self._set_gsg_navigation_active(False)

        self._nav_active = False
        self._current_goal = None
        self._last_termination = termination
        self._monty_goal_pose_mm = None
        self._rl_goal_pose_mm = None

        return MotorPolicyResult(
            actions=actions,
            motor_only_step=not observe,
            status=PolicyStatus.READY,
        )

    def _set_gsg_navigation_active(self, active: bool) -> None:
        """Set GSG navigation_active flag to suppress/enable goal generation.

        During RL navigation, GSG should not generate new hypothesis-testing
        goals because:
        1. They would be ignored by RLPolicySelector (navigation in progress)
        2. They would be logged as "attempted" with incorrect achieved status
        3. Evidence updates from directed exploration still happen normally

        Args:
            active: True to suppress GSG goals, False to re-enable.
        """
        if self._experiment_ref is None:
            return
        try:
            for lm in self._experiment_ref.model.learning_modules:
                if hasattr(lm, 'gsg') and lm.gsg is not None:
                    lm.gsg.navigation_active = active
                    logger.debug(
                        "GSG navigation_active=%s for LM %s",
                        active, lm.learning_module_id,
                    )
        except (AttributeError, TypeError) as e:
            logger.debug("Could not set GSG navigation_active: %s", e)
