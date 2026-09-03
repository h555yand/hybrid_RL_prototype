"""MuJoCo Environment Adapter for RL Goal Approach Controller.

MuJoCo-first environment: all movement and sensing through MuJoCo.
trimesh (CAD model) used only at load time for static metadata.

Architecture:
    Movement: MuJoCo Actions (MoveForward, MoveTangentially, OrientH/V, SetAgentPose)
    Sensing:  MuJoCo Rendering → Monty transforms → normal, curvature, depth
    Metadata: trimesh CAD (up_direction, extents, goal_normal — loaded once)
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot
from tbp.hybrid_rl.ablation_runner import _maybe_save_visualization

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════
MM_PER_M = 1000.0
NO_SURFACE_DEPTH_MM = 100.0
ON_OBJECT_DEPTH_MM = 3.0
SNAP_TARGET_DEPTH_MM = 2.0  # target distance from surface after snap

# ═══════════════════════════════════════════════════
# Monty imports
# ═══════════════════════════════════════════════════
from tbp.monty.frameworks.actions.actions import (
    MoveForward,
    MoveTangentially,
    OrientHorizontal,
    OrientVertical,
    SetAgentPose,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environment_utils.transforms import (
    DepthTo3DLocations,
    MissingToMaxDepth,
    TransformContext,
)
from tbp.monty.frameworks.models.motor_system_state import ProprioceptiveState
from tbp.monty.frameworks.sensors import Resolution2D, SensorConfig, SensorID
from tbp.monty.frameworks.utils.sensor_processing import (
    principal_curvatures,
    surface_normal_total_least_squares,
)
from tbp.monty.simulators.mujoco.agents import SurfaceAgent
from tbp.monty.simulators.mujoco.simulator import MuJoCoSimulator

_AGENT_ID = AgentID("rl_agent")
_SENSOR_ID = SensorID("depth_camera")
_DEFAULT_ZOOM = 10.0
_DEFAULT_HFOV = 90.0


class MuJoCoEnvAdapter:
    """MuJoCo environment with LightweightEnv-compatible interface.

    All movement through MuJoCo Actions.
    All sensing through MuJoCo rendering + Monty feature extraction.
    CAD model (trimesh) used only for static metadata at load time.
    """

    def __init__(
        self,
        mesh_path_mm: str,
        mujoco_object_name: str,
        mujoco_data_path: str,
        sensor_resolution: Tuple[int, int] = (64, 64),
        zoom: float = _DEFAULT_ZOOM,
        hfov: float = _DEFAULT_HFOV,
        seed: Optional[int] = None,
    ):
        """Initialize MuJoCo environment adapter.

        Args:
            mesh_path_mm: Path to mesh in mm units (for CAD metadata only).
            mujoco_object_name: Object name matching directory in mujoco_data_path.
                Directory must contain textured.obj + texture_map.png.
            mujoco_data_path: Path to directory containing object folders.
            sensor_resolution: (width, height) for MuJoCo camera.
            zoom: Camera zoom factor.
            hfov: Horizontal field of view in degrees.
            seed: Random seed.
        """
        self._sensor_w, self._sensor_h = sensor_resolution
        self._zoom = zoom
        self._hfov = hfov

        if seed is not None:
            np.random.seed(seed)

        # ═══ CAD model (trimesh, mm) — loaded once ═══
        self.mesh = trimesh.load(mesh_path_mm, force="mesh")
        if isinstance(self.mesh, trimesh.Scene):
            self.mesh = trimesh.util.concatenate(
                list(self.mesh.geometry.values())
            )
        self._cad_center_mm = np.array(self.mesh.centroid, dtype=float)
        self._cad_extents_mm = (
            self.mesh.bounds[1] - self.mesh.bounds[0]
        ).astype(float)
        self._compute_up_direction()

        # ═══ MuJoCo simulator ═══
        res = Resolution2D(width=self._sensor_w, height=self._sensor_h)
        sensor_configs = {
            _SENSOR_ID: SensorConfig(
                resolution=res,
                zoom=zoom,
                semantic=True,
            )
        }
        agent_factory = partial(
            SurfaceAgent,
            agent_id=_AGENT_ID,
            sensor_configs=sensor_configs,
        )
        self._sim = MuJoCoSimulator(
            agents=[agent_factory],
            data_path=mujoco_data_path,
        )
        self._sim.add_object(mujoco_object_name)
        logger.info("MuJoCo initialized: object='%s'", mujoco_object_name)

        # ═══ Monty transforms ═══
        self._missing_to_max = MissingToMaxDepth(
            agent_id=_AGENT_ID,
            max_depth=1.0,
            threshold=0.0,
        )
        self._depth_to_3d = DepthTo3DLocations(
            agent_id=_AGENT_ID,
            sensor_ids=[_SENSOR_ID],
            resolutions=[(self._sensor_h, self._sensor_w)],
            zooms=[zoom],
            hfov=[hfov],
            world_coord=True,
            get_all_points=True,
            use_semantic_sensor=False,
        )

        # ═══ Episode state ═══
        self._current_goal: Optional[np.ndarray] = None
        self._goal_normal_mm: Optional[List[float]] = None
        self._passed_through = False
        self._detach_had_collision = False
        self._edge_traversed = False
        self._last_detach_sub_steps = 1
        self._prev_normal: Optional[List[float]] = None

    # ═══════════════════════════════════════════════════
    # Unit conversion
    # ═══════════════════════════════════════════════════

    @staticmethod
    def _normalize_euler(angles):
        return (np.array(angles, dtype=float) + 180.0) % 360.0 - 180.0

    @staticmethod
    def _euler_to_quat_wxyz(euler_xyz_deg: np.ndarray) -> tuple:
        """Euler XYZ degrees → (W, X, Y, Z)."""
        r = Rot.from_euler("xyz", euler_xyz_deg, degrees=True)
        q = r.as_quat()  # scipy: [x, y, z, w]
        return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))

    @staticmethod
    def _quat_wxyz_to_euler(qw, qx, qy, qz) -> np.ndarray:
        """(W, X, Y, Z) → Euler XYZ degrees."""
        r = Rot.from_quat([qx, qy, qz, qw])
        return r.as_euler("xyz", degrees=True)

    def _look_at_direction(self, direction) -> np.ndarray:
        """Euler angles so forward (-Z) aligns with direction."""
        d = np.asarray(direction, dtype=float)
        d /= (np.linalg.norm(d) + 1e-12)
        r, _ = Rot.align_vectors([d], [[0, 0, -1]])
        return r.as_euler("xyz", degrees=True)

    # ═══════════════════════════════════════════════════
    # MuJoCo state access
    # ═══════════════════════════════════════════════════

    @property
    def _embodiment(self):
        return self._sim._agents[_AGENT_ID]._embodiment

    def _get_pos_m(self) -> np.ndarray:
        return np.array(self._embodiment.position, dtype=float)

    def _get_rot_wxyz(self) -> tuple:
        return self._embodiment.rotation

    def _get_pos_mm(self) -> np.ndarray:
        return self._get_pos_m() * MM_PER_M

    def _get_euler_deg(self) -> np.ndarray:
        w, x, y, z = self._get_rot_wxyz()
        return self._normalize_euler(self._quat_wxyz_to_euler(w, x, y, z))

    def _set_pose_mm(self, pos_mm: np.ndarray, euler_deg: np.ndarray):
        """Set agent pose via SetAgentPose action."""
        pos_m = tuple(pos_mm / MM_PER_M)
        quat = self._euler_to_quat_wxyz(euler_deg)
        action = SetAgentPose(
            agent_id=_AGENT_ID, location=pos_m, rotation_quat=quat
        )
        self._sim.step([action])

    def _apply_rotation_delta(self, axis: str, degrees: float):
        """Apply rotation delta via SetAgentPose (for look/turn in air)."""
        pos_m = self._get_pos_m()
        current = Rot.from_quat([
            self._get_rot_wxyz()[1], self._get_rot_wxyz()[2],
            self._get_rot_wxyz()[3], self._get_rot_wxyz()[0],
        ])
        if axis == "x":  # pitch (look up/down)
            delta = Rot.from_euler("x", degrees, degrees=True)
            new_rot = current * delta  # local frame
        elif axis == "y":  # yaw (turn left/right)
            delta = Rot.from_euler("y", degrees, degrees=True)
            new_rot = current * delta
        elif axis == "z":  # roll (rotate sensor)
            delta = Rot.from_euler("z", degrees, degrees=True)
            new_rot = current * delta
        else:
            return

        q = new_rot.as_quat()  # [x, y, z, w]
        quat_wxyz = (float(q[3]), float(q[0]), float(q[1]), float(q[2]))
        action = SetAgentPose(
            agent_id=_AGENT_ID,
            location=tuple(pos_m),
            rotation_quat=quat_wxyz,
        )
        self._sim.step([action])

    # ═══════════════════════════════════════════════════
    # MuJoCo rendering → Monty pipeline
    # ═══════════════════════════════════════════════════

    def _render_and_extract(self) -> dict:
        """Render MuJoCo scene and apply Monty transforms.

        Returns dict with keys:
            depth_mm, point_normal, k1_mm, k2_mm, on_object,
            semantic_3d, cam_to_world
        """
        from mujoco import mj_forward
        mj_forward(self._sim.model, self._sim.data)

        obs = self._sim.observations
        state = self._sim.states

        ctx = TransformContext(rng=np.random.RandomState(), state=state)
        obs = self._missing_to_max(obs, ctx)
        obs = self._depth_to_3d(obs, ctx)

        sensor_obs = obs[_AGENT_ID][_SENSOR_ID]
        depth_map = sensor_obs["depth"]  # (H, W), after MissingToMaxDepth
        semantic_3d = sensor_obs.get("semantic_3d")  # (N, 4) or None
        cam_to_world = sensor_obs.get("cam_to_world")  # (4, 4) or None

        # ═══ Center pixel depth → mm ═══
        cy, cx = self._sensor_h // 2, self._sensor_w // 2
        center_depth_raw = float(depth_map[cy, cx])

        if center_depth_raw >= 1.0:
            # Background (MissingToMaxDepth sets to 1.0)
            depth_mm = NO_SURFACE_DEPTH_MM
        else:
            # DepthTo3DLocations works with these depth values directly.
            # The raw depth from MuJoCo after MissingToMaxDepth is in
            # normalized units. Convert to meters using the z-buffer formula.
            # But actually, Monty's pipeline already handles this in
            # DepthTo3DLocations via inv_k projection. The depth values
            # after MissingToMaxDepth are the actual metric depths from
            # MuJoCo's renderer (not z-buffer — MuJoCo returns linear depth).
            depth_mm = center_depth_raw * MM_PER_M

        on_object = depth_mm < ON_OBJECT_DEPTH_MM

        # ═══ Normal and curvature from Monty pipeline ═══
        point_normal = None
        k1_mm, k2_mm = 0.0, 0.0

        if semantic_3d is not None and cam_to_world is not None:
            center_id = cy * self._sensor_w + cx

            if (center_id < len(semantic_3d)
                    and semantic_3d[center_id, 3] > 0):
                view_dir = cam_to_world[:3, 2]

                try:
                    normal, valid_sn = surface_normal_total_least_squares(
                        semantic_3d, center_id, view_dir
                    )
                    if valid_sn:
                        point_normal = normal.tolist()

                        try:
                            k1, k2, _, _, valid_pc = principal_curvatures(
                                semantic_3d, center_id, normal
                            )
                            if valid_pc:
                                # Monty returns curvature in 1/m (world coords)
                                # Controller expects 1/mm
                                k1_mm = float(k1) / MM_PER_M
                                k2_mm = float(k2) / MM_PER_M
                                # Convention: |k1| >= |k2|
                                if abs(k1_mm) < abs(k2_mm):
                                    k1_mm, k2_mm = k2_mm, k1_mm
                        except Exception:
                            logger.debug("Curvature extraction failed",
                                         exc_info=True)
                except Exception:
                    logger.debug("Normal extraction failed", exc_info=True)

        return {
            "depth_mm": depth_mm,
            "point_normal": point_normal,
            "k1_mm": k1_mm,
            "k2_mm": k2_mm,
            "on_object": on_object,
            "semantic_3d": semantic_3d,
            "cam_to_world": cam_to_world,
        }

    def _render_depth_in_direction(
        self, pos_mm: np.ndarray, direction: np.ndarray
    ) -> float:
        """Temporarily orient agent, render depth, return depth_mm.

        Used for path_blocked and collision checks.
        Does NOT restore pose — caller must handle that.
        """
        euler = self._look_at_direction(direction)
        self._set_pose_mm(pos_mm, euler)

        from mujoco import mj_forward
        mj_forward(self._sim.model, self._sim.data)

        agent = self._sim._agents[_AGENT_ID]
        sensor_obs = agent.observations[_SENSOR_ID]
        depth_map = sensor_obs["depth"]

        cy, cx = self._sensor_h // 2, self._sensor_w // 2
        center_raw = float(depth_map[cy, cx])

        if center_raw >= 1.0 - 1e-6:
            return NO_SURFACE_DEPTH_MM

        # MuJoCo renderer returns linear depth after enable_depth_rendering
        return center_raw * MM_PER_M

    # ═══════════════════════════════════════════════════
    # Snap-to-surface via MuJoCo depth
    # ═══════════════════════════════════════════════════

    def _snap_to_surface(self):
        """After tangential move: check depth, approach if needed, re-orient.

        Like a robot finger sliding on surface: move, check distance,
        correct position, re-orient along normal.
        """
        rendered = self._render_and_extract()
        depth_mm = rendered["depth_mm"]
        normal = rendered["point_normal"]

        if depth_mm >= NO_SURFACE_DEPTH_MM or normal is None:
            # Lost surface — don't snap
            return

        if depth_mm > ON_OBJECT_DEPTH_MM:
            # Too far from surface — approach
            approach_dist_m = (depth_mm - SNAP_TARGET_DEPTH_MM) / MM_PER_M
            if approach_dist_m > 0.0001:
                action = MoveForward(
                    agent_id=_AGENT_ID, distance=approach_dist_m
                )
                self._sim.step([action])

        # Re-orient to face surface (look along -normal)
        normal_arr = np.array(normal, dtype=float)
        n_len = np.linalg.norm(normal_arr)
        if n_len > 1e-8:
            normal_arr /= n_len
            new_euler = self._look_at_direction(-normal_arr)
            pos_mm = self._get_pos_mm()
            self._set_pose_mm(pos_mm, new_euler)

    # ═══════════════════════════════════════════════════
    # same_side via MuJoCo normal + CAD goal_normal
    # ═══════════════════════════════════════════════════

    def _compute_same_side(
        self, agent_normal: Optional[List[float]]
    ) -> bool:
        """Check if agent and goal are on same side of object.

        Uses current agent normal (from MuJoCo render) and
        goal normal (from CAD, computed once at set_goal).
        """
        if self._goal_normal_mm is None or agent_normal is None:
            return True

        center_mm = self._cad_center_mm
        agent_pos_mm = self._get_pos_mm()
        goal_pos_mm = self._current_goal[:3]

        height_axis = self.height_axis
        up = self.up_direction

        # Agent side
        an = np.array(agent_normal, dtype=float)
        an_h = an.copy()
        an_h[height_axis] = 0.0
        agent_from_center = agent_pos_mm - center_mm
        agent_from_center[height_axis] = 0.0

        if np.linalg.norm(an_h) >= 0.3:
            agent_outward = np.dot(an_h, agent_from_center) > 0
        else:
            agent_outward = np.dot(an, up) < 0

        # Goal side
        gn = np.array(self._goal_normal_mm, dtype=float)
        gn_h = gn.copy()
        gn_h[height_axis] = 0.0
        goal_from_center = goal_pos_mm - center_mm
        goal_from_center[height_axis] = 0.0

        if np.linalg.norm(gn_h) >= 0.3:
            goal_outward = np.dot(gn_h, goal_from_center) > 0
        else:
            goal_outward = np.dot(gn, up) < 0

        return agent_outward == goal_outward

    # ═══════════════════════════════════════════════════
    # path_blocked via MuJoCo depth render
    # ═══════════════════════════════════════════════════

    def _check_path_blocked(self, goal_pos_mm: np.ndarray) -> bool:
        """Check if direct path to goal is blocked by rendering toward goal."""
        agent_pos_mm = self._get_pos_mm()
        direction = goal_pos_mm - agent_pos_mm
        dist_to_goal = float(np.linalg.norm(direction))
        if dist_to_goal < 1e-8:
            return False

        direction_norm = direction / dist_to_goal

        # Save current pose
        saved_pos = agent_pos_mm.copy()
        saved_euler = self._get_euler_deg().copy()

        # Render toward goal
        depth_mm = self._render_depth_in_direction(agent_pos_mm, direction_norm)

        # Restore pose
        self._set_pose_mm(saved_pos, saved_euler)

        if depth_mm >= NO_SURFACE_DEPTH_MM:
            return False

        return depth_mm < (dist_to_goal - 2.0)

    # ═══════════════════════════════════════════════════
    # CAD metadata (trimesh, loaded once)
    # ═══════════════════════════════════════════════════

    def _compute_up_direction(self):
        """Compute up direction from CAD mesh."""
        from tbp.hybrid_rl.lightweight_env import LightweightEnv
        temp = object.__new__(LightweightEnv)
        temp.mesh = self.mesh
        temp._compute_up_direction()
        self.height_axis = temp.height_axis
        self.up_sign = temp.up_sign
        self.up_direction = temp.up_direction
        self.open_edge_height = temp.open_edge_height

    def _get_goal_normal_from_cad(self, goal_pos_mm: np.ndarray) -> List[float]:
        """Get surface normal at goal from CAD model."""
        _, _, face_id = self.mesh.nearest.on_surface([goal_pos_mm])
        return self.mesh.face_normals[face_id[0]].tolist()

    def get_random_surface_point(self, **kwargs) -> np.ndarray:
        """Random surface point from CAD model (for episode planning)."""
        from tbp.hybrid_rl.lightweight_env import LightweightEnv
        temp = object.__new__(LightweightEnv)
        temp.mesh = self.mesh
        temp.up_direction = self.up_direction
        temp.height_axis = self.height_axis
        temp.up_sign = self.up_sign
        temp.open_edge_height = self.open_edge_height
        temp._look_at_direction = self._look_at_direction
        return temp.get_random_surface_point(**kwargs)

    # ═══════════════════════════════════════════════════
    # Core interface (LightweightEnv compatible)
    # ═══════════════════════════════════════════════════

    def reset(self, position=None, rotation=None):
        """Place agent. Returns sensor_data."""
        self._passed_through = False
        self._detach_had_collision = False
        self._edge_traversed = False
        self._current_goal = None
        self._goal_normal_mm = None
        self._prev_normal = None
        self._last_detach_sub_steps = 1

        if position is not None:
            pos_mm = np.array(position, dtype=float)
            if rotation is not None:
                euler = np.array(rotation, dtype=float)
            else:
                euler = np.zeros(3)
        else:
            # Random point on CAD surface
            points, face_ids = self.mesh.sample(1, return_index=True)
            normal = self.mesh.face_normals[face_ids[0]]
            pos_mm = points[0] + normal * 2.0
            euler = self._look_at_direction(-normal)
            if rotation is not None:
                euler = np.array(rotation, dtype=float)

        euler = self._normalize_euler(euler)
        self._set_pose_mm(pos_mm, euler)
        return self.get_sensor_data()

    def set_goal(self, goal_pose):
        """Set goal [x,y,z,rx,ry,rz] in mm/degrees."""
        self._current_goal = np.array(goal_pose, dtype=float)
        self._goal_normal_mm = self._get_goal_normal_from_cad(
            self._current_goal[:3]
        )

    def get_pose(self) -> np.ndarray:
        """Return [x,y,z,rx,ry,rz] in mm/degrees."""
        pos_mm = self._get_pos_mm()
        euler = self._get_euler_deg()
        return np.concatenate([pos_mm, euler])

    def get_sensor_data(self) -> dict:
        """Build sensor_data dict for RLGoalApproachController."""
        rendered = self._render_and_extract()

        goal_normal = self._goal_normal_mm
        path_blocked = False
        same_side = True

        if self._current_goal is not None:
            goal_pos = self._current_goal[:3]
            path_blocked = self._check_path_blocked(goal_pos)
            same_side = self._compute_same_side(rendered["point_normal"])

        return {
            "point_normal": rendered["point_normal"],
            "k1": rendered["k1_mm"],
            "k2": rendered["k2_mm"],
            "principal_curvatures": [rendered["k1_mm"], rendered["k2_mm"]],
            "on_object": rendered["on_object"],
            "depth": rendered["depth_mm"],
            "passed_through": self._passed_through,
            "goal_normal": goal_normal,
            "detach_had_collision": self._detach_had_collision,
            "detach_sub_steps": self._last_detach_sub_steps,
            "path_blocked": path_blocked,
            "up_direction": self.up_direction.tolist(),
            "object_center": self._cad_center_mm.tolist(),
            "same_side": same_side,
            "object_extents": self._cad_extents_mm.tolist(),
            "edge_traversed": self._edge_traversed,
        }

    # ═══════════════════════════════════════════════════
    # Action execution
    # ═══════════════════════════════════════════════════

    def step(self, action_index, action_space):
        """Execute discrete action via MuJoCo. Returns sensor_data."""
        self._detach_had_collision = False
        self._edge_traversed = False
        self._passed_through = False

        # Save normal before step (for edge detection)
        self._prev_normal = None
        pre_render = self._render_and_extract()
        self._prev_normal = pre_render["point_normal"]

        action_info = action_space.get_info(action_index)
        name = action_info.name

        # ═══ Dispatch to MuJoCo actions ═══
        if name == "move_tangentially":
            self._do_move_tangentially(
                action_info.direction_degrees,
                action_space.surface_step,
            )
        elif name == "free_forward":
            self._do_move_forward(action_space.free_step)
        elif name == "free_backward":
            self._do_move_forward(-action_space.free_step_backward)
        elif name == "free_forward_small":
            self._do_move_forward(action_space.free_step_small)
        elif name == "look_up":
            self._apply_rotation_delta("x", action_space.rotation_step)
        elif name == "look_down":
            self._apply_rotation_delta("x", -action_space.rotation_step)
        elif name == "look_up_big":
            self._apply_rotation_delta("x", action_space.rotation_step_big)
        elif name == "look_down_big":
            self._apply_rotation_delta("x", -action_space.rotation_step_big)
        elif name == "turn_left":
            self._apply_rotation_delta("y", action_space.rotation_step)
        elif name == "turn_right":
            self._apply_rotation_delta("y", -action_space.rotation_step)
        elif name == "turn_left_big":
            self._apply_rotation_delta("y", action_space.rotation_step_big)
        elif name == "turn_right_big":
            self._apply_rotation_delta("y", -action_space.rotation_step_big)
        elif name == "rotate_sensor_+":
            self._apply_rotation_delta("z", action_space.rotation_step)
        elif name == "rotate_sensor_-":
            self._apply_rotation_delta("z", -action_space.rotation_step)
        elif name == "orient_horizontal":
            self._do_orient_horizontal(
                action_info.rotation_degrees,
                action_info.forward_distance,
                action_info.left_distance,
            )
        elif name == "orient_vertical":
            self._do_orient_vertical(
                action_info.rotation_degrees,
                action_info.forward_distance,
                action_info.down_distance,
            )
        elif name == "detach":
            if self._current_goal is not None:
                self._do_detach(
                    self._current_goal,
                    action_space.free_step * 3,
                )
        else:
            logger.warning("Unknown action: %s", name)

        # ═══ Edge traversal detection ═══
        post_render = self._render_and_extract()
        post_normal = post_render["point_normal"]
        if self._prev_normal is not None and post_normal is not None:
            dot = float(np.dot(
                np.array(self._prev_normal),
                np.array(post_normal),
            ))
            if dot < 0.707:  # > 45°
                self._edge_traversed = True

        return self.get_sensor_data()

    # ═══════════════════════════════════════════════════
    # Movement implementations
    # ═══════════════════════════════════════════════════

    def _do_move_tangentially(self, direction_degrees: float, step_mm: float):
        """Move tangentially via MuJoCo + snap to surface."""
        angle_rad = np.radians(direction_degrees)
        local_dir = (
            float(np.sin(angle_rad)),
            0.0,
            float(-np.cos(angle_rad)),
        )
        distance_m = step_mm / MM_PER_M

        action = MoveTangentially(
            agent_id=_AGENT_ID,
            distance=distance_m,
            direction=local_dir,
        )
        self._sim.step([action])

        # Snap to surface (robot finger sliding)
        self._snap_to_surface()

    def _do_move_forward(self, step_mm: float):
        """Move forward via MuJoCo + collision check."""
        # Save position before move
        old_pos_mm = self._get_pos_mm().copy()

        distance_m = step_mm / MM_PER_M
        action = MoveForward(agent_id=_AGENT_ID, distance=distance_m)
        self._sim.step([action])

        self._passed_through = False

        if abs(step_mm) > 0.5:
            # Check collision: render depth at new position
            new_render = self._render_and_extract()
            new_depth = new_render["depth_mm"]

            # If very close to surface after move → likely passed through
            proximity_threshold = min(1.0, abs(step_mm) * 0.25)
            if new_depth < proximity_threshold:
                self._passed_through = True

            # Also check: did we cross a surface?
            # Render from old position in movement direction
            new_pos_mm = self._get_pos_mm()
            move_dir = new_pos_mm - old_pos_mm
            move_len = np.linalg.norm(move_dir)
            if move_len > 1e-8:
                saved_pos = new_pos_mm.copy()
                saved_euler = self._get_euler_deg().copy()

                check_depth = self._render_depth_in_direction(
                    old_pos_mm, move_dir / move_len
                )
                if check_depth < abs(step_mm):
                    self._passed_through = True

                # Restore
                self._set_pose_mm(saved_pos, saved_euler)

    def _do_orient_horizontal(
        self, rotation_deg: float, forward_mm: float, left_mm: float
    ):
        """OrientHorizontal via MuJoCo (natively supported)."""
        # SurfaceAgent.actuate_orient_horizontal:
        #   move_along_local_axis(-left_distance, X)
        #   yaw(-rotation_degrees)  ← clockwise convention
        #   move_along_local_axis(-forward_distance, Z)
        #
        # LightweightEnv._orient_horizontal:
        #   agent_rot[1] += rotation_degrees  ← anticlockwise
        #
        # To match LightweightEnv: negate rotation_degrees
        action = OrientHorizontal(
            agent_id=_AGENT_ID,
            rotation_degrees=-rotation_deg,  # invert for clockwise convention
            left_distance=left_mm / MM_PER_M,
            forward_distance=forward_mm / MM_PER_M,
        )
        self._sim.step([action])

    def _do_orient_vertical(
        self, rotation_deg: float, forward_mm: float, down_mm: float
    ):
        """OrientVertical via MuJoCo (natively supported)."""
        # SurfaceAgent.actuate_orient_vertical:
        #   move_along_local_axis(-down_distance, Y)
        #   pitch(rotation_degrees)  ← same sign as LightweightEnv
        #   move_along_local_axis(-forward_distance, Z)
        action = OrientVertical(
            agent_id=_AGENT_ID,
            rotation_degrees=rotation_deg,
            down_distance=down_mm / MM_PER_M,
            forward_distance=forward_mm / MM_PER_M,
        )
        self._sim.step([action])

    def _do_detach(self, goal_pose: np.ndarray, detach_distance_mm: float):
        """Detach macro-action via SetAgentPose."""
        self._detach_had_collision = False
        self._last_detach_sub_steps = 1

        rendered = self._render_and_extract()
        normal = rendered["point_normal"]
        if normal is None:
            return

        normal_arr = np.array(normal, dtype=float)
        normal_arr /= (np.linalg.norm(normal_arr) + 1e-12)

        old_pos_mm = self._get_pos_mm().copy()
        old_euler = self._get_euler_deg().copy()

        # Collision check: render depth along normal
        check_depth = self._render_depth_in_direction(old_pos_mm, normal_arr)
        if check_depth < detach_distance_mm:
            self._detach_had_collision = True
            # Restore pose
            self._set_pose_mm(old_pos_mm, old_euler)
            return

        # Compute new position
        new_pos_mm = old_pos_mm + normal_arr * detach_distance_mm

        # Orient toward goal
        goal_pos = goal_pose[:3]
        goal_dir = goal_pos - new_pos_mm
        goal_dist = np.linalg.norm(goal_dir)

        if goal_dist > 1e-8:
            goal_dir /= goal_dist
            dot_goal_normal = float(np.dot(goal_dir, normal_arr))

            if dot_goal_normal < -0.2:
                tangent = goal_dir - dot_goal_normal * normal_arr
                t_len = float(np.linalg.norm(tangent))
                if t_len > 1e-8:
                    tangent /= t_len
                    fly_dir = normal_arr * 0.7 + tangent * 0.7
                else:
                    fly_dir = normal_arr
                fly_dir /= (np.linalg.norm(fly_dir) + 1e-12)
            else:
                fly_dir = goal_dir + normal_arr * 0.3
                fly_dir /= (np.linalg.norm(fly_dir) + 1e-12)

            new_euler = self._look_at_direction(fly_dir)
        else:
            new_euler = old_euler

        new_euler = self._normalize_euler(new_euler)
        self._set_pose_mm(new_pos_mm, new_euler)

    # ═══════════════════════════════════════════════════
    # Debug visualization
    # ═══════════════════════════════════════════════════

    def save_mujoco_frame(self, filepath: str, save_depth: bool = False):
        """Save MuJoCo camera view (agent's POV)."""
        from mujoco import mj_forward
        from PIL import Image

        mj_forward(self._sim.model, self._sim.data)

        cam_name = f"{_AGENT_ID}.{_SENSOR_ID}"
        res = Resolution2D(width=self._sensor_w, height=self._sensor_h)
        renderer = self._sim.renderer_for_res(res)

        renderer.update_scene(self._sim.data, camera=cam_name)
        rgb = renderer.render()
        img = Image.fromarray(rgb)
        img = img.resize((256, 256), Image.NEAREST)

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        img.save(filepath)

        if save_depth:
            renderer.enable_depth_rendering()
            depth = renderer.render()
            renderer.disable_depth_rendering()

            d_min, d_max = depth.min(), depth.max()
            if d_max > d_min:
                depth_norm = (
                    (depth - d_min) / (d_max - d_min) * 255
                ).astype(np.uint8)
            else:
                depth_norm = np.zeros_like(depth, dtype=np.uint8)
            depth_img = Image.fromarray(depth_norm, mode="L")
            depth_img = depth_img.resize((256, 256), Image.NEAREST)
            depth_img.save(filepath.replace(".png", "_depth.png"))
                        
    # ═══════════════════════════════════════════════════
    # Cleanup
    # ═══════════════════════════════════════════════════
    def close(self):
        if self._sim is not None:
            self._sim.close()
            self._sim = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
