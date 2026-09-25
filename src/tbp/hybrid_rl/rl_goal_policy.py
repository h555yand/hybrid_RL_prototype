# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""RLGoalPolicy: drop-in replacement for JumpToGoal using RL navigation.

Instead of teleporting to goal via SetAgentPose, navigates incrementally
using learned Q-store + SAC + heuristic policies through the Adaptive
Arbitrage system.

Integration path:
    GSG generates Goal → RLPolicySelector routes to RLGoalPolicy →
    RLGoalPolicy navigates step-by-step → returns motor_only_step=True
    with status=IN_PROGRESS until goal reached or timeout.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from tbp.monty.cmp import Goal, Message
from tbp.monty.context import RuntimeContext
from tbp.monty.experiment.motor_system import ExperimentMotorSystem
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
import quaternion as qt
from tbp.monty.frameworks.actions.actions import (
    SetAgentPose,
    SetSensorRotation,
)

if TYPE_CHECKING:
    from tbp.hybrid_rl.adaptive_manager import AdaptiveTrainingManager
    from tbp.hybrid_rl.mujoco_env_adapter import MuJoCoEnvAdapter

logger = logging.getLogger(__name__)


class RLGoalPolicy(MotorPolicy):
    """RL-based goal approach policy replacing JumpToGoal.

    Navigates incrementally toward GSG goals using:
    - Adaptive Arbitrage (Q-store + SAC + heuristic per-step selection)
    - Online Q-learning (updates Q-store from each transition)
    - Phase-aware navigation (crawl/fly/land/detach)

    Each __call__ = one RL step. Returns IN_PROGRESS until navigation
    completes (goal reached, timeout, or collision).

    Default mode: motor_only_step=True — LM does not process
    intermediate observations (same contract as JumpToGoal).
    """

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
        """Initialize RL goal policy.

        Args:
            agent_id: Monty agent ID.
            sensor_id: Sensor ID for depth checks.
            model_path: Path to pretrained Q-store + SAC weights.
            rl_config: RL controller configuration dict.
            mujoco_adapter: Initialized MuJoCoEnvAdapter.
            mesh_path: Path to object mesh (mm STL) for offline retrain.
                None disables offline retrain.
            max_nav_steps: Maximum steps per navigation episode.
            enable_online_learning: Whether to update Q-store online.
        """
        self._agent_id = agent_id
        self._sensor_id = sensor_id
        self._max_nav_steps = max_nav_steps
        self._enable_learning = enable_online_learning

        # Bridge: Monty ↔ RL data conversion
        self._bridge = MontyRLBridge(agent_id, rl_config, mujoco_adapter)

        # RL controller (loads pretrained Q-stores)
        self._controller = RLGoalApproachController.load(
            model_path,
            agent_id=str(agent_id),
            config={**rl_config, "mode": "adaptive"},
        )
        self._controller.mode = "adaptive"

        # Adaptive training manager (optional)
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

        # Navigation state
        self._nav_active: bool = False
        self._nav_steps: int = 0
        self._current_goal: Optional[Goal] = None
        self._last_termination: Optional[str] = None

        logger.info(
            "RLGoalPolicy initialized: agent=%s, model=%s, "
            "max_steps=%d, online_learning=%s",
            agent_id, model_path, max_nav_steps, enable_online_learning,
        )

    # ══════════════════════════════════════════════════════════
    # MotorPolicy protocol
    # ══════════════════════════════════════════════════════════

    def __call__(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        state: MotorSystemState,
        percept: Message,
        goal: Optional[Goal],
    ) -> MotorPolicyResult:
        """Execute one RL navigation step.

        Called by RLPolicySelector on each Monty step.

        Flow:
            1. If navigating and controller says done → finish
            2. If new goal and not navigating → start navigation
            3. If navigating → take one step
            4. If no goal and not navigating → return empty

        Args:
            ctx: Runtime context.
            observations: Environment observations.
            state: Motor system state (agent poses).
            percept: CMP Message from first sensor module.
            goal: Goal from GSG, or None.

        Returns:
            MotorPolicyResult with:
            - actions=[] (adapter already executed the action)
            - motor_only_step=True (LM skips intermediate observations)
            - status=IN_PROGRESS while navigating, READY when done
        """
        # ═══ Check if previous navigation step completed ═══
        if self._nav_active and not self._controller.is_active:
            return self._finish_navigation()

        # ═══ New goal → start navigation ═══
        if goal is not None and not self._nav_active:
            if goal.location is None:
                logger.warning("RLGoalPolicy: goal.location is None, skipping")
                return MotorPolicyResult([])
            return self._start_navigation(state, percept, goal)

        # ═══ Continue navigation ═══
        if self._nav_active:
            return self._navigation_step(state, percept)

        # ═══ No goal, not navigating ═══
        return MotorPolicyResult([])

    def fixme_provide_motor_system(
        self, motor_system: ExperimentMotorSystem
    ) -> None:
        pass

    def reset(self) -> None:
        self._nav_active = False
        self._nav_steps = 0
        self._current_goal = None
        self._last_termination = None
        self._prev_goals_reached = self._controller._total_goals_reached

    def state_dict(self) -> Memento:
        return {
            "nav_active": self._nav_active,
            "nav_steps": self._nav_steps,
        }

    def load_state_dict(self, memento: Memento) -> None:
        self._nav_active = memento.get("nav_active", False)
        self._nav_steps = memento.get("nav_steps", 0)

    # ══════════════════════════════════════════════════════════
    # NAVIGATION LIFECYCLE
    # ══════════════════════════════════════════════════════════

    def _start_navigation(
        self,
        state: MotorSystemState,
        percept: Message,
        goal: Goal,
    ) -> MotorPolicyResult:
        """Initialize RL navigation toward goal.

        Sets goal in bridge (for geometric queries) and controller
        (for state computation and reward).

        Args:
            state: Current motor system state.
            percept: Current sensor percept.
            goal: Target goal from GSG.

        Returns:
            Result of first navigation step.
        """
        self._current_goal = goal
        self._nav_active = True
        self._nav_steps = 0
        self._last_termination = None
        self._prev_goals_reached = self._controller._total_goals_reached

        # Set goal in bridge/adapter (computes goal_normal, enables same_side)
        self._bridge.set_goal(goal)

        # Set goal in RL controller
        goal_pose_mm = self._bridge.goal_to_pose_mm(goal)
        start_pos_mm = self._bridge.agent_position_mm(state)
        self._controller.set_new_goal(goal_pose_mm, start_pos_mm)

        # Start arbitrator episode tracking
        if self._manager:
            self._manager.arbitrator.start_episode(level=0)

        logger.debug(
            "RLGoalPolicy: navigation started, "
            "goal=[%.1f, %.1f, %.1f]mm, "
            "start=[%.1f, %.1f, %.1f]mm, "
            "dist=%.1fmm",
            *goal_pose_mm[:3], *start_pos_mm,
            float(np.linalg.norm(goal_pose_mm[:3] - start_pos_mm)),
        )

        # Take first step immediately
        return self._navigation_step(state, percept)

    def _navigation_step(self, state, percept):
        self._nav_steps += 1

        current_pose_mm = self._bridge.agent_pose_mm(state)
        sensor_data = self._bridge.percept_to_sensor_data(
            percept, state, self._current_goal,
        )

        rl_state = self._controller._compute_state(current_pose_mm, sensor_data)

        if self._manager:
            debug_info = self._controller.get_state_debug_info(
                rl_state, current_pose_mm, sensor_data,
            )
            action_type, action_params, source = self._manager.get_action(
                rl_state, current_pose_mm, sensor_data,
            )
            action_idx = sac_to_discrete(action_type, action_params)

            # Execute continuous action through adapter
            post_sensor_data = self._bridge.execute_continuous_action(
                action_type, action_params,
            )

            # Q-learning update
            if self._enable_learning:
                post_pose_mm = self._bridge.agent_pose_mm(state)
                _, done = self._controller.update_only(
                    post_pose_mm, post_sensor_data, action_idx,
                )
                if done:
                    return self._finish_navigation()

        else:
            # Direct controller mode: controller chooses AND learns in one call
            action_name, _ = self._controller.step(
                current_pose_mm, sensor_data,
            )
            if action_name is None:
                return self._finish_navigation()

            action_idx = self._controller._last_action

            # Execute discrete action through adapter
            post_sensor_data = self._bridge.execute_discrete_action(
                action_idx, self._controller.action_space,
            )

        # Timeout check
        if self._nav_steps >= self._max_nav_steps:
            if self._controller.is_active:
                post_pose = self._bridge.agent_pose_mm(state)
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
    
    def _finish_navigation(self) -> MotorPolicyResult:
        """Clean up after navigation ends.

        On success: returns SetAgentPose to sync Monty's simulator
        to the final position reached by RL adapter.
        This is equivalent to what JumpToGoal does — teleport + observe.

        On failure: returns empty actions, Monty agent stays where it was.
        """
        from scipy.spatial.transform import Rotation as Rot

        # Determine success from controller
        success = False
        termination = "unknown"

        # Check if controller ended the episode
        # _total_goals_reached increments on goal_reached
        prev_goals = getattr(self, '_prev_goals_reached', 0)
        current_goals = self._controller._total_goals_reached
        if current_goals > prev_goals:
            success = True
            termination = "goal_reached"
        elif self._nav_steps >= self._max_nav_steps:
            termination = "timeout"
        elif not self._controller.is_active:
            # Controller ended for other reason (collision, etc.)
            termination = "collision"
        self._prev_goals_reached = current_goals

        # Update adaptive manager
        if self._manager:
            self._manager.on_episode_complete(
                success=success,
                transitions=[],
            )
            self._manager.arbitrator.on_episode_end(success)

        logger.info(
            "RLGoalPolicy: navigation %s after %d steps (%s)",
            "SUCCESS" if success else "FAILED",
            self._nav_steps,
            termination,
        )

        # ═══ Sync Monty agent to adapter's final position ═══
        actions = []
        if success:
            final_pos_mm = self._bridge._adapter._get_pos_mj_mm()
            final_euler = self._bridge._adapter._get_euler_deg()

            rot = Rot.from_euler("xyz", final_euler, degrees=True)
            q = rot.as_quat()  # scipy returns xyzw
            final_quat = qt.quaternion(q[3], q[0], q[1], q[2])

            actions = [
                SetAgentPose(
                    agent_id=self._agent_id,
                    location=tuple(final_pos_mm / 1000.0),  # mm → meters
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

        return MotorPolicyResult(
            actions=actions,
            motor_only_step=not success,  # LM processes obs only on success
            status=PolicyStatus.READY,
        )

    # ══════════════════════════════════════════════════════════
    # DIAGNOSTICS
    # ══════════════════════════════════════════════════════════

    @property
    def is_navigating(self) -> bool:
        """Whether RL navigation is currently active."""
        return self._nav_active

    def get_stats(self) -> Dict[str, Any]:
        """Get RL navigation statistics."""
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
