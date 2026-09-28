# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

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

        self._bridge = MontyRLBridge(agent_id, rl_config, mujoco_adapter)

        self._controller = RLGoalApproachController.load(
            model_path,
            agent_id=str(agent_id),
            config={**rl_config, "mode": "adaptive"},
        )
        self._controller.mode = "adaptive"

        self._manager: Optional[AdaptiveTrainingManager] = None
        if enable_online_learning and mesh_path:
            from tbp.hybrid_rl.adaptive_manager import AdaptiveTrainingManager
            from tbp.hybrid_rl.lightweight_env import LightweightEnv

            trimesh_env = LightweightEnv(mesh_path)
            self._manager = AdaptiveTrainingManager(
                controller=self._controller,
                env=trimesh_env,
                config=rl_config,
                runs_dir=str(Path(model_path).parent),
                mesh_path=mesh_path,
            )

        self._nav_active: bool = False
        self._nav_steps: int = 0
        self._current_goal: Optional[Goal] = None
        self._last_termination: Optional[str] = None
        self._prev_goals_reached: int = 0
        self._monty_goal_pose_mm: Optional[np.ndarray] = None
        self._rl_goal_pose_mm: Optional[np.ndarray] = None

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

    def _start_navigation(self, state, percept, goal):
        self._current_goal = goal
        self._nav_active = True
        self._nav_steps = 0
        self._last_termination = None
        self._prev_goals_reached = self._controller._total_goals_reached

        self._bridge.set_goal(goal)

        # ═══ 1) Save original Monty goal and start position ═══
        goal_pose_mm = self._bridge.goal_to_pose_mm(goal)
        self._monty_goal_pose_mm = goal_pose_mm.copy()
        self._monty_start_pos_mm = np.array(
            state[self._agent_id].position, dtype=float
        ) * 1000.0
        self._monty_start_euler = self._bridge._adapter._get_euler_deg().copy()

        # Update adapter center for shared sim
        if not self._bridge._adapter._owns_sim:
            self._bridge._adapter._mj_center_mm = np.array([0.0, 1500.0, 0.0])

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

        return self._navigation_step(state, percept)

    def _navigation_step(self, state, percept):
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

            # Use RL surface goal for distance, not Monty goal
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

        return MotorPolicyResult(
            actions=[],
            motor_only_step=True,
            status=PolicyStatus.IN_PROGRESS,
        )

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
        rl_goal = self._rl_goal_pose_mm if self._rl_goal_pose_mm is not None else np.zeros(6)
        final_pos = self._bridge._adapter.get_pose()[:3]
        final_dist = float(np.linalg.norm(rl_goal[:3] - final_pos))

        logger.info(
            "RLGoalPolicy ep %d: %s after %d steps, "
            "dist %.1f→%.1fmm",
            self._total_nav_episodes, termination,
            self._nav_steps, self._start_distance, final_dist,
        )

        # Save actions.txt
        actions_path = (
            self._log_dir
            / f"actions_ep_{self._total_nav_episodes:05d}_{termination}.txt"
        )
        with actions_path.open("w") as f:
            f.write(f"source: monty_integration\n")
            f.write(f"Result: {termination}\n")
            f.write(f"Steps: {self._nav_steps}\n")
            f.write(f"RL Goal (surface): {rl_goal.tolist()}\n")
            if self._monty_goal_pose_mm is not None:
                f.write(f"Monty Goal (original): {self._monty_goal_pose_mm.tolist()}\n")
            f.write(f"Start distance: {self._start_distance:.1f}mm\n")
            f.write(f"End distance: {final_dist:.1f}mm\n")
            f.write(f"\n")
            f.write(f"=== Coordinate Debug ===\n")
            f.write(f"Adapter object center: "
                    f"{self._bridge._adapter._mj_center_mm.round(1).tolist()}\n")
            f.write(f"Adapter object extents: "
                    f"{self._bridge._adapter._mj_extents_mm.round(1).tolist()}\n")
            f.write(f"Adapter agent final pos: "
                    f"{self._bridge._adapter._get_pos_mj_mm().round(1).tolist()}\n")
            final_sd = self._bridge._adapter.get_sensor_data()
            f.write(f"Adapter agent final depth: "
                    f"{final_sd.get('depth', -1):.1f}mm\n")
            f.write(f"Adapter agent on_object: "
                    f"{final_sd.get('on_object', False)}\n")
            if len(self._current_poses) > 0:
                f.write(f"First adapter pos: "
                        f"{np.array(self._current_poses[0][:3]).round(1).tolist()}\n")
            if len(self._current_poses) > 1:
                f.write(f"Last adapter pos: "
                        f"{np.array(self._current_poses[-1][:3]).round(1).tolist()}\n")
            if self._monty_start_pos_mm is not None:
                f.write(f"\n=== Monty Original ===\n")
                f.write(f"Monty agent pos (mm): "
                        f"{self._monty_start_pos_mm.round(1).tolist()}\n")
            if self._monty_goal_pose_mm is not None:
                f.write(f"Monty goal pos (mm): "
                        f"{self._monty_goal_pose_mm[:3].round(1).tolist()}\n")
            f.write(f"\n")
            f.write("=" * 100 + "\n")
            for i, explanation in enumerate(self._action_explanations):
                dist_label = (
                    self._action_short_labels[i]
                    if i < len(self._action_short_labels) else ""
                )
                f.write(f"{dist_label}: {explanation}\n")

        # Episode log
        ep_entry = {
            "episode": self._total_nav_episodes,
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

        # Visualizer
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
                    extra_info={"source": "monty_integration"},
                )
            except Exception as e:
                logger.warning("Visualizer failed: %s", e, exc_info=True)

        # ═══ Return agent to Monty position ═══
        if success:
            # Success: move agent to Monty goal position (30mm from surface)
            final_pos_mm = self._monty_goal_pose_mm[:3]
            final_euler = self._monty_goal_pose_mm[3:6]
        else:
            # Failure: return agent to pre-navigation position (25mm from surface)
            final_pos_mm = self._monty_start_pos_mm
            final_euler = self._monty_start_euler

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

        self._nav_active = False
        self._current_goal = None
        self._last_termination = termination
        self._monty_goal_pose_mm = None
        self._rl_goal_pose_mm = None

        return MotorPolicyResult(
            actions=actions,
            motor_only_step=not success,
            status=PolicyStatus.READY,
        )
