"""Bridge between Monty CMP protocol and RL Goal Approach Controller.

Handles:
- Unit conversion: Monty meters/quaternions ↔ RL mm/euler degrees
- Data mapping: Monty Message/Goal → RL sensor_data/goal_pose
- Geometric queries via MuJoCoEnvAdapter (same_side, path_blocked, ray cast)

With shared simulator: no coordinate offset needed.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import quaternion as qt
from scipy.spatial.transform import Rotation as Rot

from tbp.monty.cmp import Goal, Message
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.motor_system_state import MotorSystemState

if TYPE_CHECKING:
    from tbp.hybrid_rl.mujoco_env_adapter import MuJoCoEnvAdapter

logger = logging.getLogger(__name__)

M_TO_MM = 1000.0


def _quat_to_euler_deg(q) -> np.ndarray:
    w, x, y, z = qt.as_float_array(q)
    euler = Rot.from_quat([x, y, z, w]).as_euler("xyz", degrees=True)
    return (euler + 180.0) % 360.0 - 180.0


def _euler_deg_to_quat(euler_deg: np.ndarray):
    r = Rot.from_euler("xyz", euler_deg, degrees=True)
    x, y, z, w = r.as_quat()
    return qt.quaternion(w, x, y, z)


class MontyRLBridge:
    """Thin bridge between Monty CMP and RL controller.

    With shared simulator (external_sim), no coordinate offset needed.
    All positions are in the same MuJoCo world frame.
    """

    def __init__(
        self,
        agent_id: AgentID,
        rl_config: Dict[str, Any],
        mujoco_adapter: MuJoCoEnvAdapter,
    ):
        self._agent_id = agent_id
        self._config = rl_config
        self._adapter = mujoco_adapter

    def goal_to_pose_mm(self, goal: Goal) -> np.ndarray:
        loc_mm = np.array(goal.location, dtype=float) * M_TO_MM
        pose_vec = goal.morphological_features["pose_vectors"][0]
        yaw = math.degrees(math.atan2(-pose_vec[0], -pose_vec[2]))
        pitch = math.degrees(math.asin(np.clip(float(pose_vec[1]), -1.0, 1.0)))
        return np.array([loc_mm[0], loc_mm[1], loc_mm[2], pitch, yaw, 0.0])

    def agent_pose_mm(self, state: MotorSystemState) -> np.ndarray:
        agent = state[self._agent_id]
        pos_mm = np.array(agent.position, dtype=float) * M_TO_MM
        euler = _quat_to_euler_deg(agent.rotation)
        return np.concatenate([pos_mm, euler])

    def agent_position_mm(self, state: MotorSystemState) -> np.ndarray:
        return np.array(state[self._agent_id].position, dtype=float) * M_TO_MM

    def set_goal(self, goal: Goal) -> None:
        goal_pose_mm = self.goal_to_pose_mm(goal)
        self._adapter.set_goal(goal_pose_mm)

    def percept_to_sensor_data(
        self,
        percept: Message,
        state: MotorSystemState,
        goal: Optional[Goal],
    ) -> Dict[str, Any]:
        sd: Dict[str, Any] = {}

        try:
            normal = percept.get_surface_normal()
            sd["point_normal"] = normal.tolist() if normal is not None else None
        except (ValueError, AttributeError):
            sd["point_normal"] = None

        sd["on_object"] = percept.get_on_object()

        try:
            depth_m = percept.get_feature_by_name("min_depth")
            sd["depth"] = float(depth_m) * M_TO_MM
        except (ValueError, KeyError):
            sd["depth"] = 100.0

        try:
            pcs = percept.get_feature_by_name("principal_curvatures")
            sd["k1"] = float(pcs[0])
            sd["k2"] = float(pcs[1])
        except (ValueError, KeyError, TypeError):
            sd["k1"] = 0.0
            sd["k2"] = 0.0

        if goal is not None and goal.location is not None:
            goal_pos_mm = np.array(goal.location, dtype=float) * M_TO_MM
            sd["path_blocked"] = self._adapter._check_path_blocked(goal_pos_mm)
            sd["same_side"] = self._adapter._compute_same_side(sd["point_normal"])
        else:
            sd["path_blocked"] = False
            sd["same_side"] = True

        sd["goal_normal"] = self._adapter._goal_normal_mj
        sd["object_center"] = self._adapter._mj_center_mm.tolist()
        sd["object_extents"] = self._adapter._mj_extents_mm.tolist()
        sd["up_direction"] = self._adapter.up_direction.tolist()
        sd["open_edge_height"] = self._adapter.open_edge_height

        sd["passed_through"] = self._adapter._passed_through
        sd["detach_had_collision"] = self._adapter._detach_had_collision
        sd["edge_traversed"] = self._adapter._edge_traversed
        sd["detach_sub_steps"] = self._adapter._last_detach_sub_steps

        return sd

    def execute_continuous_action(
        self, action_type: int, action_params: np.ndarray,
    ) -> Dict[str, Any]:
        return self._adapter.step_continuous(action_type, action_params)

    def execute_discrete_action(
        self, action_idx: int, action_space,
    ) -> Dict[str, Any]:
        return self._adapter.step_discrete(action_idx, action_space)
    