"""MuJoCo Environment Adapter for RL Goal Approach Controller.

Wraps Monty's MuJoCoSimulator to expose the LightweightEnv interface,
enabling the RL agent trained on trimesh to operate in MuJoCo with
YCB objects.

All sensory observations are extracted from MuJoCo depth rendering
using Monty's own feature extraction pipeline (DepthTo3DLocations,
surface_normal_total_least_squares, principal_curvatures).

Object metadata (up_direction, centroid, extents) is loaded once
from the mesh file — analogous to having a CAD model on a real robot.

Architecture:
    MuJoCo Renderer → depth map → DepthTo3DLocations → point cloud
    → surface_normal_TLS → normal
    → principal_curvatures → k1, k2
    → center pixel depth → depth (scalar)
    → semantic map → on_object

Usage:
    adapter = MuJoCoEnvAdapter(
        mesh_path="path/to/textured.obj",
        data_path="path/to/ycb_objects",
        object_name="mug",
    )
    sensor_data = adapter.reset()
    adapter.set_goal(goal_pose_mm)
    sensor_data = adapter.step(action_index, action_space)
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import quaternion as qt
import trimesh
from scipy.spatial.transform import Rotation as R

from tbp.monty.frameworks.actions.actions import (
    LookDown,
    LookUp,
    MoveForward,
    MoveTangentially,
    OrientHorizontal,
    OrientVertical,
    SetAgentPose,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environment_utils.transforms import (
    DepthTo3DLocations,
    MissingToMaxDepth,
    TransformContext,
)
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    ProprioceptiveState,
)
from tbp.monty.frameworks.sensors import Resolution2D, SensorConfig, SensorID
from tbp.monty.simulators.mujoco.agents import SurfaceAgent
from tbp.monty.simulators.mujoco.simulator import MuJoCoSimulator

# Surface geometry functions from Monty
from tbp.monty.frameworks.utils.spatial_arithmetics import normalize
from tbp.monty.frameworks.environment_utils.graph_utils import (
    surface_normal_total_least_squares,
    principal_curvatures,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════
MM_PER_M = 1000.0
AGENT_ID = AgentID("rl_agent")
SENSOR_ID = SensorID("depth_camera")
DEFAULT_HFOV = 90.0
DEFAULT_ZOOM = 10.0  # Monty surface agent uses zoom=10
NO_SURFACE_DEPTH_MM = 100.0
ON_OBJECT_THRESHOLD_MM = 3.0
ZFAR_THRESHOLD_M = 10.0  # Beyond this, consider "no surface"


class MuJoCoEnvAdapter:
    """Wraps MuJoCoSimulator to expose the LightweightEnv interface.

    Sensory data pipeline:
        MuJoCo depth map → DepthTo3DLocations (Monty) → point cloud
        → surface_normal_total_least_squares (Monty) → point_normal
        → principal_curvatures (Monty) → k1, k2

    Object metadata (static, loaded once):
        trimesh.load(mesh) → up_direction, centroid, extents, goal_normal, same_side
    """

    def __init__(
        self,
        mesh_path: str,
        data_path: str | None = None,
        object_name: str = "mug",
        sensor_resolution: Resolution2D | None = None,
        zoom: float = DEFAULT_ZOOM,
        hfov: float = DEFAULT_HFOV,
        seed: int | None = None,
    ):
        if sensor_resolution is None:
            sensor_resolution = Resolution2D(width=64, height=64)

        self._sensor_resolution = sensor_resolution
        self._zoom = zoom
        self._hfov = hfov
        self._object_name = object_name

        # ═══ Object metadata (trimesh — loaded once, like CAD model) ═══
        self._object_mesh = trimesh.load(mesh_path)
        self._mesh_centroid = np.array(self._object_mesh.centroid, dtype=float)
        self._compute_up_direction()

        # ═══ MuJoCo simulator ═══
        sensor_configs = {
            SENSOR_ID: SensorConfig(
                resolution=sensor_resolution,
                zoom=zoom,
                semantic=True,
            )
        }

        agent_factory = partial(
            SurfaceAgent,
            agent_id=AGENT_ID,
            sensor_configs=sensor_configs,
        )

        self._sim = MuJoCoSimulator(
            agents=[agent_factory],
            data_path=data_path,
        )
        self._sim.add_object(object_name)

        # ═══ Monty transforms (reused without modification) ═══
        self._missing_to_max = MissingToMaxDepth(
            agent_id=AGENT_ID,
            max_depth=1.0,
            threshold=0.0,
        )
        self._depth_to_3d = DepthTo3DLocations(
            agent_id=AGENT_ID,
            sensor_ids=[SENSOR_ID],
            resolutions=[(sensor_resolution.height, sensor_resolution.width)],
            zooms=[zoom],
            hfov=[hfov],
            world_coord=True,
            get_all_points=True,
            use_semantic_sensor=False,
        )

        # ═══ Agent state (mm / euler degrees — LightweightEnv convention) ═══
        self.agent_pos = np.zeros(3)
        self.agent_rot = np.zeros(3)

        # ═══ Episode state ═══
        self._current_goal = None
        self._passed_through = False
        self._detach_had_collision = False
        self._edge_traversed = False
        self._last_detach_sub_steps = 1
        self._wrong_side_outward = None

        # ═══ Cache for curvature (avoid recomputing every call) ═══
        self._last_point_cloud = None
        self._last_cam_to_world = None

        if seed is not None:
            np.random.seed(seed)

    # ═══════════════════════════════════════════════════
    # Unit conversion
    # ═══════════════════════════════════════════════════

    @staticmethod
    def _mm_to_m(pos_mm: np.ndarray) -> tuple:
        arr = np.asarray(pos_mm, dtype=float) / MM_PER_M
        return tuple(arr)

    @staticmethod
    def _m_to_mm(pos_m) -> np.ndarray:
        return np.asarray(pos_m, dtype=float) * MM_PER_M

    @staticmethod
    def _euler_to_quat_wxyz(euler_xyz_deg: np.ndarray) -> tuple:
        """Euler XYZ degrees → quaternion (W, X, Y, Z)."""
        rot = R.from_euler("xyz", euler_xyz_deg, degrees=True)
        q_xyzw = rot.as_quat()  # scipy: [x, y, z, w]
        return (float(q_xyzw[3]), float(q_xyzw[0]),
                float(q_xyzw[1]), float(q_xyzw[2]))

    @staticmethod
    def _quat_wxyz_to_euler(quat_wxyz) -> np.ndarray:
        """Quaternion (W, X, Y, Z) → Euler XYZ degrees."""
        w, x, y, z = quat_wxyz
        rot = R.from_quat([x, y, z, w])  # scipy expects [x, y, z, w]
        return rot.as_euler("xyz", degrees=True)

    @staticmethod
    def _normalize_euler(angles):
        return (np.array(angles, dtype=float) + 180.0) % 360.0 - 180.0

    # ═══════════════════════════════════════════════════
    # State synchronization: adapter ↔ MuJoCo
    # ═══════════════════════════════════════════════════

    def _push_state_to_mujoco(self):
        """Push adapter agent_pos/agent_rot → MuJoCo SurfaceAgent."""
        pos_m = self._mm_to_m(self.agent_pos)
        quat_wxyz = self._euler_to_quat_wxyz(self.agent_rot)
        action = SetAgentPose(
            agent_id=AGENT_ID,
            location=pos_m,
            rotation_quat=quat_wxyz,
        )
        self._sim.step([action])

    def _pull_state_from_mujoco(self):
        """Pull MuJoCo SurfaceAgent state → adapter agent_pos/agent_rot."""
        agent = self._sim._agents[AGENT_ID]
        pos_m = np.array(agent._embodiment.position)
        quat_wxyz = agent._embodiment.rotation
        self.agent_pos = self._m_to_mm(pos_m)
        self.agent_rot = self._quat_wxyz_to_euler(quat_wxyz)
        self.agent_rot = self._normalize_euler(self.agent_rot)

    # ═══════════════════════════════════════════════════
    # Observation extraction from MuJoCo
    # ═══════════════════════════════════════════════════

    def _get_proprioceptive_state(self) -> ProprioceptiveState:
        """Build ProprioceptiveState for Monty transforms."""
        return self._sim.states

    def _render_observations(self) -> dict:
        """Get raw MuJoCo observations and apply Monty transforms.

        Returns:
            Transformed observations dict containing:
                - depth: (H, W) float
                - rgba: (H, W, 4) uint8
                - semantic_3d: (N, 4) [x, y, z, sem_id] world coords
                - sensor_frame_data: (N, 4) camera coords
                - cam_to_world: (4, 4) transform matrix
        """
        # Get raw observations from MuJoCo
        obs = self._sim.observations
        state = self._sim.states

        # Build transform context
        ctx = TransformContext(
            rng=np.random.RandomState(),
            state=state,
        )

        # Apply Monty transforms
        obs = self._missing_to_max(obs, ctx)
        obs = self._depth_to_3d(obs, ctx)

        return obs[AGENT_ID][SENSOR_ID]

    def _extract_normal_from_point_cloud(
        self, sensor_obs: dict
    ) -> tuple[list[float] | None, bool]:
        """Extract surface normal at center pixel using Monty's TLS method.

        Args:
            sensor_obs: Transformed observation dict with semantic_3d,
                        sensor_frame_data, cam_to_world.

        Returns:
            (point_normal, valid): normal as [nx, ny, nz] or None, validity flag.
        """
        semantic_3d = sensor_obs.get("semantic_3d")
        cam_to_world = sensor_obs.get("cam_to_world")

        if semantic_3d is None or cam_to_world is None:
            return None, False

        h = self._sensor_resolution.height
        w = self._sensor_resolution.width
        center_id = (h // 2) * w + (w // 2)

        # Check if center pixel is on object
        if center_id >= len(semantic_3d) or semantic_3d[center_id, 3] <= 0:
            return None, False

        # View direction = 3rd column of cam_to_world rotation
        view_dir = cam_to_world[:3, 2]

        try:
            normal, valid = surface_normal_total_least_squares(
                semantic_3d, center_id, view_dir
            )
        except Exception:
            logger.debug("Surface normal extraction failed", exc_info=True)
            return None, False

        if not valid:
            return None, False

        return normal.tolist(), True

    def _extract_curvature_from_point_cloud(
        self, sensor_obs: dict, normal: np.ndarray
    ) -> dict[str, float]:
        """Extract principal curvatures using Monty's quadratic regression.

        Args:
            sensor_obs: Transformed observation dict with semantic_3d.
            normal: Surface normal at center pixel.

        Returns:
            dict with 'k1' and 'k2' in 1/mm.
        """
        semantic_3d = sensor_obs.get("semantic_3d")
        if semantic_3d is None:
            return {"k1": 0.0, "k2": 0.0}

        h = self._sensor_resolution.height
        w = self._sensor_resolution.width
        center_id = (h // 2) * w + (w // 2)

        if center_id >= len(semantic_3d) or semantic_3d[center_id, 3] <= 0:
            return {"k1": 0.0, "k2": 0.0}

        try:
            k1, k2, _, _, valid = principal_curvatures(
                semantic_3d, center_id, np.array(normal)
            )
        except Exception:
            logger.debug("Curvature extraction failed", exc_info=True)
            return {"k1": 0.0, "k2": 0.0}

        if not valid:
            return {"k1": 0.0, "k2": 0.0}

        # Convert from 1/m to 1/mm
        k1_mm = float(k1) / MM_PER_M
        k2_mm = float(k2) / MM_PER_M

        # Convention: |k1| >= |k2|
        if abs(k1_mm) < abs(k2_mm):
            k1_mm, k2_mm = k2_mm, k1_mm

        return {"k1": k1_mm, "k2": k2_mm}

    def _extract_depth_scalar(self, depth_map: np.ndarray) -> float:
        """Extract scalar depth at center pixel in mm.

        Args:
            depth_map: (H, W) depth in meters.

        Returns:
            Depth in mm, or NO_SURFACE_DEPTH_MM if no surface.
        """
        h, w = depth_map.shape
        center_depth_m = float(depth_map[h // 2, w // 2])

        if center_depth_m >= 1.0:  # MissingToMaxDepth sets background to 1.0
            return NO_SURFACE_DEPTH_MM

        return center_depth_m * MM_PER_M

    def _check_on_object_from_sensor(
        self, depth_mm: float, sensor_obs: dict
    ) -> bool:
        """Determine if agent is on object surface.

        Uses both depth threshold and semantic segmentation.

        Args:
            depth_mm: Scalar depth in mm.
            sensor_obs: Observation dict (may contain semantic info).

        Returns:
            True if agent is on object surface.
        """
        # Primary: depth threshold (same as LightweightEnv)
        if depth_mm < ON_OBJECT_THRESHOLD_MM:
            return True

        # Fallback: check semantic at center pixel
        semantic_3d = sensor_obs.get("semantic_3d")
        if semantic_3d is not None:
            h = self._sensor_resolution.height
            w = self._sensor_resolution.width
            center_id = (h // 2) * w + (w // 2)
            if center_id < len(semantic_3d) and semantic_3d[center_id, 3] > 0:
                # On object but depth > threshold — close but not touching
                return depth_mm < ON_OBJECT_THRESHOLD_MM

        return False

    def _check_path_blocked_via_render(self, goal_pos_mm: np.ndarray) -> bool:
        """Check if direct path to goal is blocked by rendering toward goal.

        Temporarily orients agent toward goal, renders depth, checks if
        surface is closer than goal distance.

        Args:
            goal_pos_mm: Goal position in mm.

        Returns:
            True if path is blocked.
        """
        direction = goal_pos_mm - self.agent_pos
        dist_to_goal = float(np.linalg.norm(direction))
        if dist_to_goal < 1e-8:
            return False

        # Save current state
        saved_rot = self.agent_rot.copy()

        # Look toward goal
        self.agent_rot = self._look_at_direction(direction / dist_to_goal)
        self._push_state_to_mujoco()

        # Render and get center depth
        obs = self._sim.observations
        depth_map = obs[AGENT_ID][SENSOR_ID].depth
        h, w = depth_map.shape
        center_depth_m = float(depth_map[h // 2, w // 2])
        center_depth_mm = center_depth_m * MM_PER_M

        # Restore orientation
        self.agent_rot = saved_rot
        self._push_state_to_mujoco()

        # Blocked if surface is closer than goal (with margin)
        if center_depth_m >= 1.0:  # No surface in view
            return False

        return center_depth_mm < (dist_to_goal - 2.0)

    def _check_passed_through_via_render(
        self, old_pos_mm: np.ndarray, direction: np.ndarray, step_size_mm: float
    ) -> bool:
        """Check if agent passed through object during a move.

        Renders depth from old position in movement direction.

        Args:
            old_pos_mm: Position before move.
            direction: Normalized movement direction.
            step_size_mm: Step size in mm.

        Returns:
            True if agent passed through object.
        """
        # Save current state
        saved_pos = self.agent_pos.copy()
        saved_rot = self.agent_rot.copy()

        # Move to old position, look in movement direction
        self.agent_pos = old_pos_mm.copy()
        self.agent_rot = self._look_at_direction(direction)
        self._push_state_to_mujoco()

        # Render depth
        obs = self._sim.observations
        depth_map = obs[AGENT_ID][SENSOR_ID].depth
        h, w = depth_map.shape
        center_depth_m = float(depth_map[h // 2, w // 2])
        center_depth_mm = center_depth_m * MM_PER_M

        # Restore state
        self.agent_pos = saved_pos
        self.agent_rot = saved_rot
        self._push_state_to_mujoco()

        if center_depth_m >= 1.0:
            return False

        return center_depth_mm < step_size_mm

    def _detect_edge_traversal(
        self, normal_before: list[float] | None, normal_after: list[float] | None
    ) -> bool:
        """Detect if agent traversed an edge by comparing normals.

        Args:
            normal_before: Surface normal before step.
            normal_after: Surface normal after step.

        Returns:
            True if normals differ by > 45 degrees.
        """
        if normal_before is None or normal_after is None:
            return False

        n1 = np.array(normal_before)
        n2 = np.array(normal_after)
        dot = float(np.dot(n1, n2))
        # cos(45°) ≈ 0.707
        return dot < 0.707

    # ═══════════════════════════════════════════════════
    # Object metadata (from mesh, loaded once)
    # ═══════════════════════════════════════════════════

    def _get_goal_normal(self, goal_pos_mm: np.ndarray) -> list[float] | None:
        """Get surface normal at goal position from object mesh."""
        _, _, face_id = self._object_mesh.nearest.on_surface([goal_pos_mm])
        return self._object_mesh.face_normals[face_id[0]].tolist()

    def _is_reachable_by_surface(
        self, start_pos: np.ndarray, goal_pos: np.ndarray
    ) -> bool:
        """Check if start and goal are on the same side of the object."""
        from tbp.hybrid_rl.lightweight_env import _is_reachable_by_surface

        return _is_reachable_by_surface(self, start_pos, goal_pos)

    def _compute_up_direction(self):
        """Compute up direction from object mesh (same as LightweightEnv)."""
        from tbp.hybrid_rl.lightweight_env import LightweightEnv

        temp = object.__new__(LightweightEnv)
        temp.mesh = self._object_mesh
        temp._compute_up_direction()
        self.height_axis = temp.height_axis
        self.up_sign = temp.up_sign
        self.up_direction = temp.up_direction
        self.open_edge_height = temp.open_edge_height

    @property
    def mesh(self):
        """Expose mesh for _is_reachable_by_surface compatibility."""
        return self._object_mesh

    # ═══════════════════════════════════════════════════
    # Core interface (matches LightweightEnv)
    # ═══════════════════════════════════════════════════

    def reset(self, position=None, rotation=None):
        """Place the agent. Same interface as LightweightEnv.reset()."""
        self._passed_through = False
        self._detach_had_collision = False
        self._current_goal = None
        self._wrong_side_outward = None
        self._edge_traversed = False

        if position is not None:
            self.agent_pos = np.array(position, dtype=float)
        else:
            points, face_ids = self._object_mesh.sample(1, return_index=True)
            normal = self._object_mesh.face_normals[face_ids[0]]
            self.agent_pos = points[0] + normal * 2.0

            if rotation is None:
                self.agent_rot = self._look_at_direction(-normal)

        if rotation is not None:
            self.agent_rot = np.array(rotation, dtype=float)
        elif position is not None:
            self.agent_rot = np.zeros(3)

        self.agent_rot = self._normalize_euler(self.agent_rot)
        self._push_state_to_mujoco()

        return self.get_sensor_data()

    def set_goal(self, goal_pose):
        """Set goal pose [x,y,z,rx,ry,rz] in mm/degrees."""
        self._current_goal = np.array(goal_pose, dtype=float)

    def step(self, action_index, action_space):
        """Execute a discrete action. Same interface as LightweightEnv.step()."""
        self._detach_had_collision = False
        self._edge_traversed = False

        action_info = action_space.get_info(action_index)

        # Get normal before step (for edge detection)
        sensor_before = self.get_sensor_data()
        normal_before = sensor_before.get("point_normal")

        # Save position before step
        old_pos = self.agent_pos.copy()
        old_rot = self.agent_rot.copy()

        # Translate and execute action
        monty_actions = self._translate_action(action_info, action_space)

        for monty_action in monty_actions:
            self._sim.step([monty_action])

        # Pull updated state from MuJoCo
        self._pull_state_from_mujoco()

        # Post-step checks
        if action_info.name in ("free_forward", "free_backward", "free_forward_small"):
            rot = R.from_euler("xyz", old_rot, degrees=True)
            forward = rot.apply([0, 0, -1])
            step_size = self._get_step_size(action_info.name, action_space)
            self._passed_through = self._check_passed_through_via_render(
                old_pos, forward * np.sign(step_size), abs(step_size)
            )

            # Additional proximity check
            sensor_after = self._render_observations()
            depth_after = self._extract_depth_scalar(sensor_after["depth"])
            proximity_threshold = min(1.0, abs(step_size) * 0.25)
            if depth_after < proximity_threshold:
                self._passed_through = True
        else:
            self._passed_through = False

        # Edge traversal detection
        sensor_after_data = self.get_sensor_data()
        normal_after = sensor_after_data.get("point_normal")
        self._edge_traversed = self._detect_edge_traversal(normal_before, normal_after)

        # Update sensor_data with edge_traversed
        sensor_after_data["edge_traversed"] = self._edge_traversed
        sensor_after_data["passed_through"] = self._passed_through

        return sensor_after_data

    def get_pose(self):
        """Return [x, y, z, rx, ry, rz] in mm/degrees."""
        return np.concatenate([self.agent_pos, self.agent_rot])

    def get_sensor_data(self):
        """Extract sensor_data dict compatible with RLGoalApproachController.

        All sensory data from MuJoCo rendering + Monty feature extraction.
        Object metadata from mesh (loaded once).
        """
        # ═══ Render and extract features from MuJoCo ═══
        sensor_obs = self._render_observations()
        depth_map = sensor_obs["depth"]

        # Depth (scalar, mm)
        depth_mm = self._extract_depth_scalar(depth_map)

        # On object
        on_object = self._check_on_object_from_sensor(depth_mm, sensor_obs)

        # Surface normal (from Monty TLS)
        point_normal, normal_valid = self._extract_normal_from_point_cloud(sensor_obs)

        # Curvature (from Monty quadratic regression)
        if normal_valid and point_normal is not None:
            curvature = self._extract_curvature_from_point_cloud(
                sensor_obs, point_normal
            )
        else:
            curvature = {"k1": 0.0, "k2": 0.0}

        # ═══ Goal-dependent computations ═══
        goal_normal = None
        path_blocked = False
        same_side = True

        if self._current_goal is not None:
            goal_pos = self._current_goal[:3]
            goal_normal = self._get_goal_normal(goal_pos)
            path_blocked = self._check_path_blocked_via_render(goal_pos)
            same_side = self._is_reachable_by_surface(self.agent_pos, goal_pos)

        return {
            "point_normal": point_normal,
            "k1": curvature["k1"],
            "k2": curvature["k2"],
            "principal_curvatures": [curvature["k1"], curvature["k2"]],
            "on_object": on_object,
            "depth": depth_mm,
            "passed_through": getattr(self, "_passed_through", False),
            "goal_normal": goal_normal,
            "detach_had_collision": getattr(self, "_detach_had_collision", False),
            "detach_sub_steps": getattr(self, "_last_detach_sub_steps", 1),
            "path_blocked": path_blocked,
            "up_direction": self.up_direction.tolist(),
            "object_center": self._object_mesh.centroid.tolist(),
            "same_side": same_side,
            "object_extents": (
                self._object_mesh.bounds[1] - self._object_mesh.bounds[0]
            ).tolist(),
            "edge_traversed": getattr(self, "_edge_traversed", False),
        }

    def get_random_surface_point(self, **kwargs) -> np.ndarray:
        """Random point on surface. Same as LightweightEnv."""
        from tbp.hybrid_rl.lightweight_env import LightweightEnv

        temp = object.__new__(LightweightEnv)
        temp.mesh = self._object_mesh
        temp.up_direction = self.up_direction
        temp.height_axis = self.height_axis
        temp.up_sign = self.up_sign
        temp.open_edge_height = self.open_edge_height
        temp._look_at_direction = self._look_at_direction
        return temp.get_random_surface_point(**kwargs)

    # ═══════════════════════════════════════════════════
    # Action translation
    # ═══════════════════════════════════════════════════

    def _get_step_size(self, action_name: str, action_space) -> float:
        """Get step size in mm for a given action name."""
        if action_name == "free_forward":
            return action_space.free_step
        elif action_name == "free_backward":
            return -action_space.free_step_backward
        elif action_name == "free_forward_small":
            return action_space.free_step_small
        return 0.0

    def _translate_action(self, action_info, action_space) -> list:
        """Convert discrete action → list of Monty Action objects."""
        name = action_info.name

        if name == "move_tangentially":
            angle_rad = np.radians(action_info.direction_degrees)
            local_dir = np.array([
                np.sin(angle_rad), 0.0, -np.cos(angle_rad)
            ])
            local_dir /= (np.linalg.norm(local_dir) + 1e-12)
            distance_m = action_space.surface_step / MM_PER_M
            return [MoveTangentially(
                agent_id=AGENT_ID,
                distance=distance_m,
                direction=tuple(local_dir),
            )]

        elif name == "free_forward":
            return [MoveForward(
                agent_id=AGENT_ID,
                distance=action_space.free_step / MM_PER_M,
            )]

        elif name == "free_backward":
            return [MoveForward(
                agent_id=AGENT_ID,
                distance=-action_space.free_step_backward / MM_PER_M,
            )]

        elif name == "free_forward_small":
            return [MoveForward(
                agent_id=AGENT_ID,
                distance=action_space.free_step_small / MM_PER_M,
            )]

        elif name == "look_up":
            return [LookUp(
                agent_id=AGENT_ID,
                rotation_degrees=action_space.rotation_step,
            )]

        elif name == "look_down":
            return [LookDown(
                agent_id=AGENT_ID,
                rotation_degrees=action_space.rotation_step,
            )]

        elif name == "look_up_big":
            return [LookUp(
                agent_id=AGENT_ID,
                rotation_degrees=action_space.rotation_step_big,
            )]

        elif name == "look_down_big":
            return [LookDown(
                agent_id=AGENT_ID,
                rotation_degrees=action_space.rotation_step_big,
            )]

        elif name == "turn_left":
            return [TurnLeft(
                agent_id=AGENT_ID,
                rotation_degrees=action_space.rotation_step,
            )]

        elif name == "turn_right":
            return [TurnRight(
                agent_id=AGENT_ID,
                rotation_degrees=action_space.rotation_step,
            )]

        elif name == "turn_left_big":
            return [TurnLeft(
                agent_id=AGENT_ID,
                rotation_degrees=action_space.rotation_step_big,
            )]

        elif name == "turn_right_big":
            return [TurnRight(
                agent_id=AGENT_ID,
                rotation_degrees=action_space.rotation_step_big,
            )]

        elif name == "orient_horizontal":
            return [OrientHorizontal(
                agent_id=AGENT_ID,
                rotation_degrees=action_info.rotation_degrees,
                forward_distance=action_info.forward_distance / MM_PER_M,
                left_distance=action_info.left_distance / MM_PER_M,
            )]

        elif name == "orient_vertical":
            return [OrientVertical(
                agent_id=AGENT_ID,
                rotation_degrees=action_info.rotation_degrees,
                forward_distance=action_info.forward_distance / MM_PER_M,
                down_distance=action_info.down_distance / MM_PER_M,
            )]

        elif name in ("rotate_sensor_+", "rotate_sensor_-"):
            sign = 1.0 if name == "rotate_sensor_+" else -1.0
            self.agent_rot[2] += sign * action_space.rotation_step
            self.agent_rot = self._normalize_euler(self.agent_rot)
            self._push_state_to_mujoco()
            return []

        elif name == "detach":
            return self._handle_detach(action_space)

        else:
            logger.warning(f"Unknown action: {name}, skipping")
            return []

    def _handle_detach(self, action_space) -> list:
        """Handle detach macro-action via direct state manipulation.

        Detach: move along surface normal, then orient toward goal.
        Collision detection via MuJoCo depth rendering.
        """
        self._detach_had_collision = False
        self._last_detach_sub_steps = 1

        if self._current_goal is None:
            return []

        # Get current normal from MuJoCo rendering
        sensor_obs = self._render_observations()
        point_normal, valid = self._extract_normal_from_point_cloud(sensor_obs)

        if not valid or point_normal is None:
            logger.debug("DETACH: no valid normal, aborting")
            return []

        normal = np.array(point_normal, dtype=float)
        normal /= (np.linalg.norm(normal) + 1e-12)

        detach_distance = action_space.free_step * 3  # mm

        # Check collision: render depth in normal direction from current pos
        old_pos = self.agent_pos.copy()
        collision = self._check_passed_through_via_render(
            old_pos, normal, detach_distance
        )

        if collision:
            self._detach_had_collision = True
            logger.debug("DETACH: collision detected, aborting")
            return []

        # Move along normal
        new_pos_mm = old_pos + normal * detach_distance

        # Orient toward goal
        goal_pos = self._current_goal[:3]
        goal_dir = goal_pos - new_pos_mm
        goal_dist = np.linalg.norm(goal_dir)

        if goal_dist > 1e-8:
            goal_dir /= goal_dist
            dot_goal_normal = float(np.dot(goal_dir, normal))

            if dot_goal_normal < -0.2:
                # Goal behind surface — fly sideways
                tangent = goal_dir - dot_goal_normal * normal
                t_len = np.linalg.norm(tangent)
                if t_len > 1e-8:
                    tangent /= t_len
                    fly_dir = normal * 0.7 + tangent * 0.7
                else:
                    fly_dir = normal
                fly_dir /= (np.linalg.norm(fly_dir) + 1e-12)
            else:
                fly_dir = goal_dir + normal * 0.3
                fly_dir /= (np.linalg.norm(fly_dir) + 1e-12)

            new_rot = self._look_at_direction(fly_dir)
        else:
            new_rot = self.agent_rot.copy()

        # Apply via SetAgentPose
        self.agent_pos = new_pos_mm
        self.agent_rot = self._normalize_euler(new_rot)

        pos_m = self._mm_to_m(self.agent_pos)
        quat_wxyz = self._euler_to_quat_wxyz(self.agent_rot)

        return [SetAgentPose(
            agent_id=AGENT_ID,
            location=pos_m,
            rotation_quat=quat_wxyz,
        )]

    # ═══════════════════════════════════════════════════
    # Utility methods
    # ═══════════════════════════════════════════════════

    def _look_at_direction(self, direction) -> np.ndarray:
        """Return euler angles [rx, ry, rz] in degrees for looking in direction."""
        d = np.asarray(direction, dtype=float)
        d /= (np.linalg.norm(d) + 1e-12)
        forward = np.array([0.0, 0.0, -1.0])
        rot, _ = R.align_vectors([d], [forward])
        return rot.as_euler("xyz", degrees=True)

    def get_mujoco_render(self) -> dict:
        """Get raw MuJoCo rendered observations (for debugging)."""
        obs = self._sim.observations
        sensor_obs = obs[AGENT_ID][SENSOR_ID]
        return {
            "depth": sensor_obs.depth,
            "rgba": sensor_obs.rgba,
            "semantic": getattr(sensor_obs, "semantic", None),
        }

    def close(self):
        """Clean up MuJoCo resources."""
        self._sim.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
