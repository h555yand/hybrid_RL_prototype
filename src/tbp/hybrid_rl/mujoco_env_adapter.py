"""MuJoCo Environment Adapter for RL Goal Approach Controller.

MuJoCo-first environment. All runtime data in MuJoCo frame (mm).
CAD model (trimesh) used only at load time, converted to MuJoCo frame.

Coordinate frames:
    CAD frame: original mesh coordinates (mm), used by trimesh
    MuJoCo frame: after refpos/refquat/scale transform (mm for positions)
    
    All public API (get_pose, set_goal, sensor_data) uses MuJoCo frame in mm.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════
MM_PER_M = 1000.0
NO_SURFACE_DEPTH_MM = 100.0
ON_OBJECT_DEPTH_MM = 5.0
SNAP_TARGET_DEPTH_MM = 2.0

# ═══════════════════════════════════════════════════
# Monty imports
# ═══════════════════════════════════════════════════
from tbp.monty.frameworks.actions.actions import (
    MoveForward, MoveTangentially, OrientHorizontal,
    OrientVertical, SetAgentPose,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environment_utils.transforms import (
    DepthTo3DLocations, MissingToMaxDepth, TransformContext,
)
from tbp.monty.frameworks.sensors import Resolution2D, SensorConfig, SensorID
from tbp.monty.frameworks.utils.sensor_processing import (
    principal_curvatures, surface_normal_total_least_squares,
)
from tbp.monty.simulators.mujoco.agents import SurfaceAgent
from tbp.monty.simulators.mujoco.simulator import MuJoCoSimulator

_AGENT_ID = AgentID("rl_agent")
_SENSOR_ID = SensorID("depth_camera")
_DEFAULT_ZOOM = 10.0
_DEFAULT_HFOV = 90.0


class MuJoCoEnvAdapter:

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
        self._sensor_w, self._sensor_h = sensor_resolution
        self._zoom = zoom
        self._hfov = hfov

        if seed is not None:
            np.random.seed(seed)

        # ═══ Load MuJoCo metadata ═══
        from tbp.monty.simulators.mujoco.objects import load_object_metadata, ObjectMetadata
        metadata_path = Path(mujoco_data_path) / mujoco_object_name / "metadata.json"
        if metadata_path.exists():
            metadata = load_object_metadata(metadata_path, mujoco_object_name)
        else:
            metadata = ObjectMetadata()

        self._mj_refpos = np.array(metadata.refpos, dtype=float)
        self._mj_refquat = np.array(metadata.refquat, dtype=float)  # WXYZ
        self._mj_scale = np.array(metadata.scale, dtype=float)

        # Pre-compute rotation objects
        rw, rx, ry, rz = self._mj_refquat
        self._ref_rot = Rot.from_quat([rx, ry, rz, rw])          # refquat
        self._ref_rot_inv = Rot.from_quat([-rx, -ry, -rz, rw])   # conjugate

        logger.info(
            "MuJoCo metadata: refpos=%s, refquat=%s, scale=%s",
            self._mj_refpos.tolist(), self._mj_refquat.tolist(),
            self._mj_scale.tolist(),
        )

        # ═══ CAD model (trimesh, mm) — for surface sampling only ═══
        self._cad_mesh = trimesh.load(mesh_path_mm, force="mesh")
        if isinstance(self._cad_mesh, trimesh.Scene):
            self._cad_mesh = trimesh.util.concatenate(
                list(self._cad_mesh.geometry.values())
            )

        # ═══ Convert CAD metadata to MuJoCo frame (mm) ═══
        cad_center_mm = np.array(self._cad_mesh.centroid, dtype=float)
        self._mj_center_mm = self._pos_cad_to_mj_mm(cad_center_mm)

        cad_bounds_min = self._cad_mesh.bounds[0]
        cad_bounds_max = self._cad_mesh.bounds[1]
        mj_corners = np.array([
            self._pos_cad_to_mj_mm(np.array([x, y, z]))
            for x in [cad_bounds_min[0], cad_bounds_max[0]]
            for y in [cad_bounds_min[1], cad_bounds_max[1]]
            for z in [cad_bounds_min[2], cad_bounds_max[2]]
        ])
        self._mj_extents_mm = (mj_corners.max(axis=0) - mj_corners.min(axis=0))

        # Up direction: compute in CAD, convert to MuJoCo
        self._compute_up_direction_mj()

        # Expose mesh for get_random_surface_point compatibility
        self.mesh = self._cad_mesh

        # ═══ MuJoCo simulator ═══
        res = Resolution2D(width=self._sensor_w, height=self._sensor_h)
        sensor_configs = {
            _SENSOR_ID: SensorConfig(resolution=res, zoom=zoom, semantic=True)
        }
        agent_factory = partial(
            SurfaceAgent, agent_id=_AGENT_ID, sensor_configs=sensor_configs,
        )
        self._sim = MuJoCoSimulator(
            agents=[agent_factory], data_path=mujoco_data_path,
        )
        self._sim.add_object(mujoco_object_name)
        # ═══ Increase offscreen framebuffer for scene rendering ═══
        self._sim.model.vis.global_.offwidth = 256
        self._sim.model.vis.global_.offheight = 256

        logger.info("MuJoCo initialized: object='%s'", mujoco_object_name)

        # ═══ Monty transforms ═══
        self._missing_to_max = MissingToMaxDepth(
            agent_id=_AGENT_ID, max_depth=1.0, threshold=0.0,
        )
        self._depth_to_3d = DepthTo3DLocations(
            agent_id=_AGENT_ID, sensor_ids=[_SENSOR_ID],
            resolutions=[(self._sensor_h, self._sensor_w)],
            zooms=[zoom], hfov=[hfov],
            world_coord=True, get_all_points=True, use_semantic_sensor=False,
        )

        # ═══ Episode state ═══
        self._current_goal: Optional[np.ndarray] = None
        self._current_goal_cad: Optional[np.ndarray] = None
        self._goal_normal_mj: Optional[List[float]] = None
        self._passed_through = False
        self._detach_had_collision = False
        self._edge_traversed = False
        self._last_detach_sub_steps = 1
        self._prev_normal: Optional[List[float]] = None

        # ═══ Scene renderer (for visualization, separate from sensor) ═══
        self._scene_renderer = None
        self._scene_res = (256, 256)

    # ═══════════════════════════════════════════════════
    # Coordinate conversion (computed once, used everywhere)
    # ═══════════════════════════════════════════════════

    def _pos_cad_to_mj_m(self, pos_cad_mm: np.ndarray) -> np.ndarray:
        """CAD mm → MuJoCo meters."""
        pos_m = pos_cad_mm / MM_PER_M
        pos_shifted = pos_m - self._mj_refpos
        pos_rotated = self._ref_rot_inv.apply(pos_shifted)
        return pos_rotated * self._mj_scale

    def _pos_cad_to_mj_mm(self, pos_cad_mm: np.ndarray) -> np.ndarray:
        """CAD mm → MuJoCo mm."""
        return self._pos_cad_to_mj_m(pos_cad_mm) * MM_PER_M

    def _dir_cad_to_mj(self, dir_cad: np.ndarray) -> np.ndarray:
        """Direction CAD → MuJoCo (rotation only, no translation)."""
        return self._ref_rot_inv.apply(dir_cad)

    def _dir_mj_to_cad(self, dir_mj: np.ndarray) -> np.ndarray:
        """Direction MuJoCo → CAD."""
        return self._ref_rot.apply(dir_mj)

    # ═══════════════════════════════════════════════════
    # Up direction (CAD → MuJoCo)
    # ═══════════════════════════════════════════════════

    def _compute_up_direction_mj(self):
        """Compute up direction in MuJoCo frame."""
        from tbp.hybrid_rl.lightweight_env import LightweightEnv
        temp = object.__new__(LightweightEnv)
        temp.mesh = self._cad_mesh
        temp._compute_up_direction()

        # Convert to MuJoCo frame
        self.up_direction = self._dir_cad_to_mj(temp.up_direction)
        self.up_direction /= (np.linalg.norm(self.up_direction) + 1e-12)
        self.height_axis = int(np.argmax(np.abs(self.up_direction)))
        self.up_sign = float(np.sign(self.up_direction[self.height_axis]))

        cad_edge = temp.open_edge_height
        # Convert edge height: it's along the CAD height axis
        edge_point_cad = np.zeros(3)
        edge_point_cad[temp.height_axis] = cad_edge
        edge_point_mj = self._pos_cad_to_mj_mm(edge_point_cad)
        self.open_edge_height = edge_point_mj[self.height_axis]

    # ═══════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════
    def _mj_ray_cast(
        self, origin_mj_mm: np.ndarray, direction: np.ndarray
    ) -> float:
        """MuJoCo ray cast. Returns hit distance in mm, or -1 if no hit."""
        import mujoco

        origin_m = origin_mj_mm / MM_PER_M
        direction = np.array(direction, dtype=np.float64)
        d_len = np.linalg.norm(direction)
        if d_len < 1e-12:
            return -1.0
        direction = direction / d_len

        geomid = np.array([-1], dtype=np.int32)
        hit_dist = mujoco.mj_ray(
            self._sim.model,
            self._sim.data,
            origin_m,
            direction,
            None,       # geomgroup
            1,          # flg_static
            -1,         # bodyexclude
            geomid,
        )

        if hit_dist < 0:
            return -1.0
        return float(hit_dist) * MM_PER_M

    @staticmethod
    def _normalize_euler(angles):
        return (np.array(angles, dtype=float) + 180.0) % 360.0 - 180.0

    @staticmethod
    def _quat_wxyz_to_euler(qw, qx, qy, qz) -> np.ndarray:
        return Rot.from_quat([qx, qy, qz, qw]).as_euler("xyz", degrees=True)

    def _look_at_direction(self, direction) -> np.ndarray:
        """Euler angles so forward (-Z) aligns with direction. MuJoCo frame."""
        d = np.asarray(direction, dtype=float)
        d /= (np.linalg.norm(d) + 1e-12)
        r, _ = Rot.align_vectors([d], [[0, 0, -1]])
        return r.as_euler("xyz", degrees=True)

    # ═══════════════════════════════════════════════════
    # MuJoCo state access (all in MuJoCo frame)
    # ═══════════════════════════════════════════════════

    @property
    def _embodiment(self):
        return self._sim._agents[_AGENT_ID]._embodiment

    def _get_pos_mj_mm(self) -> np.ndarray:
        """Agent position in MuJoCo mm."""
        return np.array(self._embodiment.position, dtype=float) * MM_PER_M

    def _get_euler_deg(self) -> np.ndarray:
        """Agent euler in MuJoCo frame."""
        w, x, y, z = self._embodiment.rotation
        return self._normalize_euler(self._quat_wxyz_to_euler(w, x, y, z))

    def _set_pose_mj_mm(self, pos_mj_mm: np.ndarray, euler_deg: np.ndarray):
        """Set agent pose in MuJoCo frame (mm, degrees)."""
        pos_m = tuple(pos_mj_mm / MM_PER_M)
        rot = Rot.from_euler("xyz", euler_deg, degrees=True)
        q = rot.as_quat()  # [x,y,z,w]
        quat_wxyz = (float(q[3]), float(q[0]), float(q[1]), float(q[2]))
        action = SetAgentPose(
            agent_id=_AGENT_ID, location=pos_m, rotation_quat=quat_wxyz,
        )
        self._sim.step([action])

    def _apply_rotation_delta(self, axis: str, degrees: float):
        """Apply rotation delta in MuJoCo frame."""
        pos_m = np.array(self._embodiment.position, dtype=float)
        w, x, y, z = self._embodiment.rotation
        current = Rot.from_quat([x, y, z, w])

        delta = Rot.from_euler(axis, degrees, degrees=True)
        new_rot = current * delta  # local frame

        q = new_rot.as_quat()
        quat_wxyz = (float(q[3]), float(q[0]), float(q[1]), float(q[2]))
        action = SetAgentPose(
            agent_id=_AGENT_ID, location=tuple(pos_m),
            rotation_quat=quat_wxyz,
        )
        self._sim.step([action])

    # ═══════════════════════════════════════════════════
    # Rendering (returns data in MuJoCo frame)
    # ═══════════════════════════════════════════════════

    def _render_and_extract(self) -> dict:
        """Render and extract features. All in MuJoCo frame."""
        from mujoco import mj_forward
        mj_forward(self._sim.model, self._sim.data)

        obs = self._sim.observations
        state = self._sim.states
        ctx = TransformContext(rng=np.random.RandomState(), state=state)
        obs = self._missing_to_max(obs, ctx)
        obs = self._depth_to_3d(obs, ctx)

        sensor_obs = obs[_AGENT_ID][_SENSOR_ID]
        depth_map = sensor_obs["depth"]
        semantic_3d = sensor_obs.get("semantic_3d")
        cam_to_world = sensor_obs.get("cam_to_world")

        cy, cx = self._sensor_h // 2, self._sensor_w // 2
        center_depth_raw = float(depth_map[cy, cx])

        if center_depth_raw >= 1.0:
            depth_mm = NO_SURFACE_DEPTH_MM
        else:
            depth_mm = center_depth_raw * MM_PER_M

        on_object = depth_mm < ON_OBJECT_DEPTH_MM

        # Normal and curvature (already in MuJoCo world frame from Monty)
        point_normal = None
        k1_mm, k2_mm = 0.0, 0.0

        if semantic_3d is not None and cam_to_world is not None:
            center_id = cy * self._sensor_w + cx
            if center_id < len(semantic_3d) and semantic_3d[center_id, 3] > 0:
                view_dir = cam_to_world[:3, 2]
                try:
                    normal, valid_sn = surface_normal_total_least_squares(
                        semantic_3d, center_id, view_dir
                    )
                    if valid_sn:
                        point_normal = normal.tolist()  # MuJoCo frame
                        try:
                            k1, k2, _, _, valid_pc = principal_curvatures(
                                semantic_3d, center_id, normal
                            )
                            if valid_pc:
                                k1_mm = float(k1) / MM_PER_M
                                k2_mm = float(k2) / MM_PER_M
                                if abs(k1_mm) < abs(k2_mm):
                                    k1_mm, k2_mm = k2_mm, k1_mm
                        except Exception:
                            pass
                except Exception:
                    pass

        return {
            "depth_mm": depth_mm, "point_normal": point_normal,
            "k1_mm": k1_mm, "k2_mm": k2_mm, "on_object": on_object,
        }

    # ═══════════════════════════════════════════════════
    # same_side (MuJoCo frame)
    # ═══════════════════════════════════════════════════

    def _compute_same_side(self, agent_normal: Optional[List[float]]) -> bool:
        if self._goal_normal_mj is None or agent_normal is None:
            return True

        center = self._mj_center_mm
        agent_pos = self._get_pos_mj_mm()
        goal_pos = self._current_goal[:3]
        h = self.height_axis
        up = self.up_direction

        an = np.array(agent_normal, dtype=float)
        an_h = an.copy(); an_h[h] = 0.0
        afc = agent_pos - center; afc[h] = 0.0
        agent_out = np.dot(an_h, afc) > 0 if np.linalg.norm(an_h) >= 0.3 else np.dot(an, up) < 0

        gn = np.array(self._goal_normal_mj, dtype=float)
        gn_h = gn.copy(); gn_h[h] = 0.0
        gfc = goal_pos - center; gfc[h] = 0.0
        goal_out = np.dot(gn_h, gfc) > 0 if np.linalg.norm(gn_h) >= 0.3 else np.dot(gn, up) < 0

        return agent_out == goal_out

    # ═══════════════════════════════════════════════════
    # path_blocked (MuJoCo frame)
    # ═══════════════════════════════════════════════════

    def _check_path_blocked(self, goal_pos_mj_mm: np.ndarray) -> bool:
        """Check path blocked via MuJoCo ray cast."""
        agent_pos = self._get_pos_mj_mm()
        direction = goal_pos_mj_mm - agent_pos
        dist_to_goal = float(np.linalg.norm(direction))
        if dist_to_goal < 1e-8:
            return False

        hit_dist = self._mj_ray_cast(agent_pos, direction)
        if hit_dist < 0:
            return False
        return hit_dist < (dist_to_goal - 2.0)
        
    # ═══════════════════════════════════════════════════
    # Surface point generation (CAD → MuJoCo)
    # ═══════════════════════════════════════════════════
    def _pos_mj_mm_to_cad_mm(self, pos_mj_mm: np.ndarray) -> np.ndarray:
        """MuJoCo mm → CAD mm (inverse of _pos_cad_to_mj_mm)."""
        pos_mj_m = pos_mj_mm / MM_PER_M
        pos_unscaled = pos_mj_m / self._mj_scale
        pos_unrotated = self._ref_rot.apply(pos_unscaled)
        pos_cad_m = pos_unrotated + self._mj_refpos
        return pos_cad_m * MM_PER_M

    def get_random_surface_point(self, **kwargs) -> np.ndarray:
        """Generate random surface point. Returns 6D pose in MuJoCo mm."""
        from tbp.hybrid_rl.lightweight_env import LightweightEnv

        # Convert reference_pos from MuJoCo mm to CAD mm
        if "reference_pos" in kwargs and kwargs["reference_pos"] is not None:
            ref_mj_mm = np.array(kwargs["reference_pos"], dtype=float)
            kwargs["reference_pos"] = self._pos_mj_mm_to_cad_mm(ref_mj_mm)

        # Build temp LightweightEnv for CAD-based sampling
        temp = object.__new__(LightweightEnv)
        temp.mesh = self._cad_mesh
        up_cad = self._dir_mj_to_cad(self.up_direction)
        temp.up_direction = up_cad
        temp.height_axis = int(np.argmax(np.abs(up_cad)))
        temp.up_sign = float(np.sign(up_cad[temp.height_axis]))
        temp.open_edge_height = (
            self._cad_mesh.bounds[1][temp.height_axis]
            if temp.up_sign > 0
            else self._cad_mesh.bounds[0][temp.height_axis]
        )

        def cad_look_at(direction):
            d = np.asarray(direction, dtype=float)
            d /= (np.linalg.norm(d) + 1e-12)
            r, _ = Rot.align_vectors([d], [[0, 0, -1]])
            return r.as_euler("xyz", degrees=True)

        temp._look_at_direction = cad_look_at

        # Generate in CAD frame
        cad_pose = temp.get_random_surface_point(**kwargs)

        # Convert position CAD mm → MuJoCo mm
        pos_mj_mm = self._pos_cad_to_mj_mm(cad_pose[:3])

        # Convert rotation: forward direction CAD → MuJoCo
        cad_rot = Rot.from_euler("xyz", cad_pose[3:], degrees=True)
        forward_cad = cad_rot.apply([0, 0, -1])
        forward_mj = self._dir_cad_to_mj(forward_cad)
        euler_mj = self._look_at_direction(forward_mj)

        return np.concatenate([pos_mj_mm, euler_mj])

    def set_goal(self, goal_pose):
        """Set goal in MuJoCo mm."""
        self._current_goal = np.array(goal_pose, dtype=float)
        goal_pos_cad_mm = self._pos_mj_mm_to_cad_mm(goal_pose[:3])
        self._goal_normal_mj = self._get_goal_normal_mj(goal_pos_cad_mm)

    def _get_goal_normal_mj(self, goal_pos_cad_mm: np.ndarray) -> List[float]:
        """Get goal normal from CAD, convert to MuJoCo frame."""
        _, _, face_id = self._cad_mesh.nearest.on_surface([goal_pos_cad_mm])
        normal_cad = self._cad_mesh.face_normals[face_id[0]]
        normal_mj = self._dir_cad_to_mj(normal_cad)
        normal_mj /= (np.linalg.norm(normal_mj) + 1e-12)
        return normal_mj.tolist()

    # ═══════════════════════════════════════════════════
    # Core interface
    # ═══════════════════════════════════════════════════

    def reset(self, position=None, rotation=None):
        self._passed_through = False
        self._detach_had_collision = False
        self._edge_traversed = False
        self._current_goal = None
        self._current_goal_cad = None
        self._goal_normal_mj = None
        self._prev_normal = None
        self._last_detach_sub_steps = 1

        if position is not None:
            # position is in CAD mm (from episode pool)
            pos_mj_mm = self._pos_cad_to_mj_mm(np.array(position, dtype=float))
            if rotation is not None:
                # rotation is CAD euler — convert forward direction
                cad_rot = Rot.from_euler("xyz", rotation, degrees=True)
                fwd_cad = cad_rot.apply([0, 0, -1])
                fwd_mj = self._dir_cad_to_mj(fwd_cad)
                euler = self._look_at_direction(fwd_mj)
            else:
                euler = np.zeros(3)
        else:
            # Random point on CAD surface → convert to MuJoCo
            points, face_ids = self._cad_mesh.sample(1, return_index=True)
            normal_cad = self._cad_mesh.face_normals[face_ids[0]]
            pos_cad_mm = points[0] + normal_cad * 2.0

            pos_mj_mm = self._pos_cad_to_mj_mm(pos_cad_mm)
            normal_mj = self._dir_cad_to_mj(-normal_cad)  # look at surface
            euler = self._look_at_direction(normal_mj)

            if rotation is not None:
                cad_rot = Rot.from_euler("xyz", rotation, degrees=True)
                fwd_cad = cad_rot.apply([0, 0, -1])
                fwd_mj = self._dir_cad_to_mj(fwd_cad)
                euler = self._look_at_direction(fwd_mj)

        euler = self._normalize_euler(euler)
        self._set_pose_mj_mm(pos_mj_mm, euler)
        return self.get_sensor_data()

    def get_pose(self) -> np.ndarray:
        """Return [x,y,z,rx,ry,rz] in MuJoCo mm / degrees."""
        return np.concatenate([self._get_pos_mj_mm(), self._get_euler_deg()])

    def get_sensor_data(self) -> dict:
        rendered = self._render_and_extract()

        goal_normal = self._goal_normal_mj
        path_blocked = False
        same_side = True

        if self._current_goal is not None:
            goal_pos = self._current_goal[:3]
            same_side = self._compute_same_side(rendered["point_normal"])
            path_blocked = self._check_path_blocked(goal_pos)

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
            "object_center": self._mj_center_mm.tolist(),
            "same_side": same_side,
            "object_extents": self._mj_extents_mm.tolist(),
            "edge_traversed": self._edge_traversed,
        }

    # ═══════════════════════════════════════════════════
    # Actions
    # ═══════════════════════════════════════════════════

    def step(self, action_index, action_space):
        self._detach_had_collision = False
        self._edge_traversed = False
        self._passed_through = False

        pre = self._render_and_extract()
        self._prev_normal = pre["point_normal"]

        info = action_space.get_info(action_index)
        name = info.name

        if name == "move_tangentially":
            self._do_move_tangentially(info.direction_degrees, action_space.surface_step)
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
            self._do_orient_horizontal(info.rotation_degrees, info.forward_distance, info.left_distance)
        elif name == "orient_vertical":
            self._do_orient_vertical(info.rotation_degrees, info.forward_distance, info.down_distance)
        elif name == "detach":
            if self._current_goal is not None:
                self._do_detach(self._current_goal, action_space.free_step * 3)

        # Edge detection
        post = self._render_and_extract()
        if self._prev_normal is not None and post["point_normal"] is not None:
            dot = float(np.dot(np.array(self._prev_normal), np.array(post["point_normal"])))
            if dot < 0.707:
                self._edge_traversed = True

        return self.get_sensor_data()

    def _do_move_tangentially(self, direction_degrees: float, step_mm: float):
        pos = self._get_pos_mj_mm()
        euler = self._get_euler_deg()
        rot = Rot.from_euler("xyz", euler, degrees=True)

        angle_rad = np.radians(direction_degrees)
        local_dir = np.array([np.sin(angle_rad), 0.0, -np.cos(angle_rad)])
        world_dir = rot.apply(local_dir)

        # Project onto tangent plane using current normal
        rendered = self._render_and_extract()
        normal = rendered["point_normal"]
        if normal is not None:
            n = np.array(normal, dtype=float)
            n_len = np.linalg.norm(n)
            if n_len > 1e-8:
                n = n / n_len
                # Remove normal component
                world_dir = world_dir - np.dot(world_dir, n) * n
                w_len = np.linalg.norm(world_dir)
                if w_len > 1e-8:
                    world_dir = world_dir / w_len

        new_pos = pos + world_dir * step_mm
        self._set_pose_mj_mm(new_pos, euler)
        self._snap_to_surface()

    def _snap_to_surface(self):
        """After tangential move: approach to 2mm + re-orient along normal."""
        rendered = self._render_and_extract()
        depth_mm = rendered["depth_mm"]
        normal = rendered["point_normal"]

        if depth_mm >= 10.0 or normal is None:
            return

        normal_arr = np.array(normal, dtype=float)
        n_len = np.linalg.norm(normal_arr)
        if n_len < 1e-8:
            return
        normal_arr /= n_len

        pos = self._get_pos_mj_mm()

        # Approach to SNAP_TARGET_DEPTH_MM
        if abs(depth_mm - SNAP_TARGET_DEPTH_MM) > 0.3:
            approach = depth_mm - SNAP_TARGET_DEPTH_MM
            pos = pos - normal_arr * approach

        # Re-orient to face surface (like LightweightEnv)
        new_euler = self._look_at_direction(-normal_arr)
        self._set_pose_mj_mm(pos, new_euler)

    def pose_mj_to_cad(self, pose_mj: np.ndarray) -> np.ndarray:
        """Convert 6D pose from MuJoCo frame to CAD frame for visualization."""
        pos_cad = self._pos_mj_mm_to_cad_mm(pose_mj[:3])
        
        mj_rot = Rot.from_euler("xyz", pose_mj[3:], degrees=True)
        fwd_mj = mj_rot.apply([0, 0, -1])
        fwd_cad = self._dir_mj_to_cad(fwd_mj)
        d = fwd_cad / (np.linalg.norm(fwd_cad) + 1e-12)
        r, _ = Rot.align_vectors([d], [[0, 0, -1]])
        euler_cad = r.as_euler("xyz", degrees=True)
        
        return np.concatenate([pos_cad, euler_cad])

    def _do_move_forward(self, step_mm: float):
        """Move forward via MuJoCo + collision check via ray cast."""
        old_pos = self._get_pos_mj_mm().copy()

        # Get forward direction before move
        w, x, y, z = self._embodiment.rotation
        rot = Rot.from_quat([x, y, z, w])
        forward = rot.apply([0, 0, -1])

        distance_m = step_mm / MM_PER_M
        action = MoveForward(agent_id=_AGENT_ID, distance=distance_m)
        self._sim.step([action])

        self._passed_through = False

        if abs(step_mm) > 0.5:
            move_dir = forward * np.sign(step_mm)
            hit_dist = self._mj_ray_cast(old_pos, move_dir)
            if hit_dist > 0 and hit_dist < abs(step_mm):
                self._passed_through = True

    def _do_orient_horizontal(self, rotation_deg, forward_mm, left_mm):
        action = OrientHorizontal(
            agent_id=_AGENT_ID, rotation_degrees=-rotation_deg,
            left_distance=left_mm / MM_PER_M, forward_distance=forward_mm / MM_PER_M,
        )
        self._sim.step([action])

    def _do_orient_vertical(self, rotation_deg, forward_mm, down_mm):
        action = OrientVertical(
            agent_id=_AGENT_ID, rotation_degrees=rotation_deg,
            down_distance=down_mm / MM_PER_M, forward_distance=forward_mm / MM_PER_M,
        )
        self._sim.step([action])

    def _do_detach(self, goal_pose, detach_distance_mm):
        """Detach macro-action via SetAgentPose + ray cast collision."""
        self._detach_had_collision = False
        self._last_detach_sub_steps = 1

        rendered = self._render_and_extract()
        normal = rendered["point_normal"]
        if normal is None:
            return

        normal_arr = np.array(normal, dtype=float)
        normal_arr /= (np.linalg.norm(normal_arr) + 1e-12)

        old_pos = self._get_pos_mj_mm().copy()

        # Collision check via ray cast along normal
        hit_dist = self._mj_ray_cast(old_pos, normal_arr)
        if hit_dist > 0 and hit_dist < detach_distance_mm:
            self._detach_had_collision = True
            return

        # Compute new position
        new_pos = old_pos + normal_arr * detach_distance_mm

        # Orient toward goal
        goal_pos = goal_pose[:3]
        goal_dir = goal_pos - new_pos
        goal_dist = np.linalg.norm(goal_dir)

        if goal_dist > 1e-8:
            goal_dir /= goal_dist
            dot = float(np.dot(goal_dir, normal_arr))

            if dot < -0.2:
                tangent = goal_dir - dot * normal_arr
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
            new_euler = self._get_euler_deg()

        self._set_pose_mj_mm(new_pos, self._normalize_euler(new_euler))

    # ═══════════════════════════════════════════════════
    # Debug
    # ═══════════════════════════════════════════════════

    def save_mujoco_frame(self, filepath: str, save_depth: bool = False):
        from mujoco import mj_forward
        from PIL import Image
        mj_forward(self._sim.model, self._sim.data)
        cam_name = f"{_AGENT_ID}.{_SENSOR_ID}"
        res = Resolution2D(width=self._sensor_w, height=self._sensor_h)
        renderer = self._sim.renderer_for_res(res)
        renderer.update_scene(self._sim.data, camera=cam_name)
        rgb = renderer.render()
        img = Image.fromarray(rgb).resize((256, 256), Image.NEAREST)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        img.save(filepath)
        if save_depth:
            renderer.enable_depth_rendering()
            depth = renderer.render()
            renderer.disable_depth_rendering()
            d_min, d_max = depth.min(), depth.max()
            if d_max > d_min:
                dn = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            else:
                dn = np.zeros_like(depth, dtype=np.uint8)
            Image.fromarray(dn, "L").resize((256, 256), Image.NEAREST).save(
                filepath.replace(".png", "_depth.png")
            )

    def save_mujoco_scene(
        self,
        filepath: str,
        agent_pos_cad_mm: np.ndarray,
        goal_pos_cad_mm: np.ndarray,
        trail_positions_mj_mm: list = None,
    ):
        """Render MuJoCo scene with agent/goal/trail at 256x256."""
        from mujoco import mj_forward, Renderer
        from PIL import Image, ImageDraw
        import mujoco

        mj_forward(self._sim.model, self._sim.data)

        # Create scene renderer once
        if self._scene_renderer is None:
            self._scene_renderer = Renderer(
                self._sim.model,
                height=self._scene_res[1],
                width=self._scene_res[0],
            )

        agent_m = np.array(self._embodiment.position, dtype=float)
        goal_m = (
            self._current_goal[:3] / MM_PER_M
            if self._current_goal is not None
            else None
        )

        max_ext = float(max(self._mj_extents_mm)) / MM_PER_M
        sphere_size = max_ext * 0.03
        trail_size = sphere_size * 0.25

        # Camera
        if goal_m is not None:
            midpoint = (agent_m + goal_m) / 2
        else:
            midpoint = agent_m

        cam_dist = max_ext * 1.3
        to_agent = agent_m.copy()
        to_agent[2] = 0
        ta_len = np.linalg.norm(to_agent)
        azimuth = (
            np.degrees(np.arctan2(to_agent[1], to_agent[0]))
            if ta_len > 1e-5 else 135
        )

        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = midpoint
        camera.distance = cam_dist
        camera.azimuth = azimuth + 90
        camera.elevation = -25

        # Build scene
        max_trail = 50
        scene = mujoco.MjvScene(
            self._sim.model, maxgeom=max_trail + 50
        )
        mujoco.mjv_updateScene(
            self._sim.model, self._sim.data,
            mujoco.MjvOption(), None,
            camera, mujoco.mjtCatBit.mjCAT_ALL, scene,
        )

        # Trail (orange)
        if trail_positions_mj_mm:
            trail = trail_positions_mj_mm
            if len(trail) > max_trail:
                step = len(trail) // max_trail
                trail = trail[::step]
            for t_pos_mm in trail:
                if scene.ngeom >= scene.maxgeom:
                    break
                t_m = np.array(t_pos_mm[:3], dtype=float) / MM_PER_M
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    [trail_size, 0, 0], t_m,
                    np.eye(3).flatten(),
                    [1.0, 0.65, 0.0, 0.8],
                )
                scene.ngeom += 1

        # Agent (blue)
        if scene.ngeom < scene.maxgeom:
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                [sphere_size, 0, 0], agent_m,
                np.eye(3).flatten(),
                [0.2, 0.2, 1.0, 1.0],
            )
            scene.ngeom += 1

        # Goal (green)
        if goal_m is not None and scene.ngeom < scene.maxgeom:
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                [sphere_size * 1.3, 0, 0], goal_m,
                np.eye(3).flatten(),
                [0.2, 1.0, 0.2, 1.0],
            )
            scene.ngeom += 1

        # Gaze (red)
        if scene.ngeom < scene.maxgeom:
            w, x, y, z = self._embodiment.rotation
            rot = Rot.from_quat([x, y, z, w])
            fwd = rot.apply([0, 0, -1])
            gaze_end = agent_m + fwd * sphere_size * 5
            midpt = (agent_m + gaze_end) / 2
            d = gaze_end - agent_m
            length = float(np.linalg.norm(d))
            if length > 1e-8:
                d /= length
                za = np.array([0, 0, 1.0])
                v = np.cross(za, d)
                c = float(np.dot(za, d))
                if abs(c + 1) < 1e-6:
                    rm = np.diag([-1.0, -1.0, 1.0])
                elif np.linalg.norm(v) < 1e-8:
                    rm = np.eye(3)
                else:
                    vx = np.array([
                        [0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0],
                    ])
                    rm = np.eye(3) + vx + vx @ vx / (1 + c)
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    [sphere_size * 0.15, length / 2, 0],
                    midpt, rm.flatten(),
                    [1.0, 0.0, 0.0, 1.0],
                )
                scene.ngeom += 1

        # Render at scene resolution
        w, h = self._scene_res
        mujoco.mjr_render(
            mujoco.MjrRect(0, 0, w, h),
            scene, self._scene_renderer._mjr_context,
        )
        rgb = np.empty((h, w, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(
            rgb, None,
            mujoco.MjrRect(0, 0, w, h),
            self._scene_renderer._mjr_context,
        )
        rgb = np.flipud(rgb)

        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        dist = float(np.linalg.norm(
            agent_pos_cad_mm - goal_pos_cad_mm
        ))
        draw.text((10, 10), f"dist={dist:.1f}mm", fill="white")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        img.save(filepath)

    def close(self):
        if self._scene_renderer is not None:
            try:
                self._scene_renderer.close()
            except Exception:
                pass
            self._scene_renderer = None
        if self._sim is not None:
            self._sim.close()
            self._sim = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
