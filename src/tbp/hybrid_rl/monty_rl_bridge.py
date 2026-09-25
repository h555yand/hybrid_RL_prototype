# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Bridge between Monty CMP protocol and RL Goal Approach Controller.

Handles:
- Unit conversion: Monty meters/quaternions ↔ RL mm/euler degrees
- Data mapping: Monty Message/Goal → RL sensor_data/goal_pose
- Action conversion: RL action_type+params → Monty Action objects
- Geometric queries: same_side, path_blocked via MuJoCo ray cast

Uses MuJoCoEnvAdapter for ray casting, object metadata, and action execution.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import quaternion as qt
from scipy.spatial.transform import Rotation as Rot

from tbp.monty.cmp import Goal, Message
from tbp.monty.frameworks.actions.actions import (
    Action,
    LookDown,
    LookUp,
    MoveForward,
    MoveTangentially,
    OrientHorizontal,
    OrientVertical,
    SetAgentPose,
    SetSensorRotation,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.motor_system_state import MotorSystemState

if TYPE_CHECKING:
    from tbp.hybrid_rl.mujoco_env_adapter import MuJoCoEnvAdapter

logger = logging.getLogger(__name__)

M_TO_MM = 1000.0


def _quat_to_euler_deg(q) -> np.ndarray:
    """Convert numpy-quaternion (WXYZ) to euler XYZ degrees.

    Args:
        q: numpy quaternion object (w, x, y, z).

    Returns:
        Euler angles [pitch, yaw, roll] in degrees, normalized to [-180, 180].
    """
    w, x, y, z = qt.as_float_array(q)
    euler = Rot.from_quat([x, y, z, w]).as_euler("xyz", degrees=True)
    return (euler + 180.0) % 360.0 - 180.0


def _euler_deg_to_quat(euler_deg: np.ndarray):
    """Convert euler XYZ degrees to numpy-quaternion (WXYZ).

    Args:
        euler_deg: [pitch, yaw, roll] in degrees.

    Returns:
        numpy quaternion object.
    """
    r = Rot.from_euler("xyz", euler_deg, degrees=True)
    x, y, z, w = r.as_quat()  # scipy returns xyzw
    return qt.quaternion(w, x, y, z)


class MontyRLBridge:
    """Thin bridge between Monty CMP and RL controller.

    Delegates geometric queries (same_side, path_blocked, goal_normal)
    to MuJoCoEnvAdapter which has ray cast and CAD mesh access.

    Extracts sensor features from Monty percept Message instead of
    re-rendering — avoids duplicate computation.

    All RL-side data is in MuJoCo frame, millimeters, euler degrees.
    All Monty-side data is in meters, quaternions.
    """

    def __init__(
        self,
        agent_id: AgentID,
        rl_config: Dict[str, Any],
        mujoco_adapter: MuJoCoEnvAdapter,
    ):
        """Initialize bridge.

        Args:
            agent_id: Monty agent ID.
            rl_config: RL controller configuration dict.
            mujoco_adapter: Initialized MuJoCoEnvAdapter with loaded object.
        """
        self._agent_id = agent_id
        self._config = rl_config
        self._adapter = mujoco_adapter

    # ══════════════════════════════════════════════════════════
    # COORDINATE CONVERSION
    # ══════════════════════════════════════════════════════════

    def goal_to_pose_mm(self, goal: Goal) -> np.ndarray:
        """Convert Monty Goal to RL goal pose [x,y,z,rx,ry,rz] in mm/degrees.

        Uses the same conversion as JumpToGoal._derive_set_agent_pose_from_goal:
        - location: meters → mm
        - pose_vectors[0]: agent direction vector → pitch/yaw euler angles

        Args:
            goal: Monty Goal from GSG.

        Returns:
            6D pose [x, y, z, pitch, yaw, roll] in mm and degrees.
        """
        loc_mm = np.array(goal.location, dtype=float) * M_TO_MM

        pose_vec = goal.morphological_features["pose_vectors"][0]
        # Same math as JumpToGoal._derive_set_agent_pose_from_goal
        yaw = math.degrees(math.atan2(-pose_vec[0], -pose_vec[2]))
        pitch = math.degrees(math.asin(np.clip(float(pose_vec[1]), -1.0, 1.0)))

        return np.array([loc_mm[0], loc_mm[1], loc_mm[2], pitch, yaw, 0.0])

    def agent_pose_mm(self, state: MotorSystemState) -> np.ndarray:
        """Extract agent pose from MotorSystemState in RL format.

        Args:
            state: Monty motor system state.

        Returns:
            6D pose [x, y, z, pitch, yaw, roll] in mm and degrees.
        """
        agent = state[self._agent_id]
        pos_mm = np.array(agent.position, dtype=float) * M_TO_MM
        euler = _quat_to_euler_deg(agent.rotation)
        return np.concatenate([pos_mm, euler])

    def agent_position_mm(self, state: MotorSystemState) -> np.ndarray:
        """Extract agent position in mm.

        Args:
            state: Monty motor system state.

        Returns:
            3D position [x, y, z] in mm.
        """
        return np.array(state[self._agent_id].position, dtype=float) * M_TO_MM

    # ══════════════════════════════════════════════════════════
    # GOAL MANAGEMENT
    # ══════════════════════════════════════════════════════════

    def set_goal(self, goal: Goal) -> None:
        """Set goal in adapter for geometric queries.

        Computes goal_normal from CAD mesh, enables same_side
        and path_blocked computation.

        Args:
            goal: Monty Goal from GSG.
        """
        goal_pose_mm = self.goal_to_pose_mm(goal)
        self._adapter.set_goal(goal_pose_mm)

    # ══════════════════════════════════════════════════════════
    # SENSOR DATA EXTRACTION
    # ══════════════════════════════════════════════════════════

    def percept_to_sensor_data(
        self,
        percept: Message,
        state: MotorSystemState,
        goal: Optional[Goal],
    ) -> Dict[str, Any]:
        """Convert Monty percept Message to RL sensor_data dict.

        Features from percept (already computed by Monty sensor pipeline):
            point_normal, on_object, depth, k1, k2

        Features from MuJoCoEnvAdapter (ray cast + CAD metadata):
            same_side, path_blocked, goal_normal, object_center,
            object_extents, up_direction, open_edge_height

        Collision flags from adapter (updated during action execution):
            passed_through, detach_had_collision, edge_traversed

        Args:
            percept: CMP Message from first sensor module.
            state: Current motor system state.
            goal: Current Goal (for path_blocked/same_side), or None.

        Returns:
            sensor_data dict compatible with RLGoalApproachController.
        """
        sd: Dict[str, Any] = {}

        # ═══ From Monty percept ═══
        # Surface normal
        try:
            normal = percept.get_surface_normal()
            sd["point_normal"] = (
                normal.tolist() if normal is not None else None
            )
        except (ValueError, AttributeError):
            sd["point_normal"] = None

        # On object flag
        sd["on_object"] = percept.get_on_object()

        # Depth (meters → mm)
        try:
            depth_m = percept.get_feature_by_name("min_depth")
            sd["depth"] = float(depth_m) * M_TO_MM
        except (ValueError, KeyError):
            sd["depth"] = 100.0

        # Principal curvatures
        try:
            pcs = percept.get_feature_by_name("principal_curvatures")
            sd["k1"] = float(pcs[0])
            sd["k2"] = float(pcs[1])
        except (ValueError, KeyError, TypeError):
            sd["k1"] = 0.0
            sd["k2"] = 0.0

        # ═══ From MuJoCoEnvAdapter (geometric queries) ═══
        if goal is not None and goal.location is not None:
            goal_pos_mm = np.array(goal.location, dtype=float) * M_TO_MM
            sd["path_blocked"] = self._adapter._check_path_blocked(goal_pos_mm)
            sd["same_side"] = self._adapter._compute_same_side(
                sd["point_normal"]
            )
        else:
            sd["path_blocked"] = False
            sd["same_side"] = True

        # Object metadata (cached in adapter at init time)
        sd["goal_normal"] = self._adapter._goal_normal_mj
        sd["object_center"] = self._adapter._mj_center_mm.tolist()
        sd["object_extents"] = self._adapter._mj_extents_mm.tolist()
        sd["up_direction"] = self._adapter.up_direction.tolist()
        sd["open_edge_height"] = self._adapter.open_edge_height

        # Collision flags (managed by adapter during action execution)
        sd["passed_through"] = self._adapter._passed_through
        sd["detach_had_collision"] = self._adapter._detach_had_collision
        sd["edge_traversed"] = self._adapter._edge_traversed
        sd["detach_sub_steps"] = self._adapter._last_detach_sub_steps

        return sd

    # ══════════════════════════════════════════════════════════
    # ACTION EXECUTION (through adapter)
    # ══════════════════════════════════════════════════════════

    def execute_continuous_action(
        self,
        action_type: int,
        action_params: np.ndarray,
    ) -> Dict[str, Any]:
        """Execute RL continuous action through MuJoCoEnvAdapter.

        Adapter handles tangential snap, collision detection,
        edge traversal, detach macro — all tested and working.

        Args:
            action_type: PSAC action type (0-7).
            action_params: Continuous parameters [3].

        Returns:
            Post-action sensor_data dict from adapter.
        """
        return self._adapter.step_continuous(action_type, action_params)

    def execute_discrete_action(
        self,
        action_idx: int,
        action_space,
    ) -> Dict[str, Any]:
        """Execute RL discrete action through MuJoCoEnvAdapter.

        Args:
            action_idx: Discrete action index (0-23).
            action_space: ActionSpace instance.

        Returns:
            Post-action sensor_data dict from adapter.
        """
        return self._adapter.step_discrete(action_idx, action_space)
