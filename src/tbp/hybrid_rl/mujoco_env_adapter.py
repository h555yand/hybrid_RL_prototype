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
ON_OBJECT_DEPTH_MM = 3.0
SNAP_TARGET_DEPTH_MM = 2.0
SNAP_MAX_DIST_DEFAULT = 15.0  # 5x surface_step, covers edge traversal

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
        external_sim=None,
        agent_id: str = "rl_agent",
        snap_max_dist: float = SNAP_MAX_DIST_DEFAULT,  # NEW
    ):
        self._agent_id_local = AgentID(agent_id)    
        self._snap_max_dist = snap_max_dist    

        # Determine sensor ID
        if external_sim is not None:
            # Monty surface agent uses "patch" as primary sensor
            self._sensor_id_local = SensorID("patch")
            # self._sensor_id_local = SensorID("view_finder")
        else:
            self._sensor_id_local = _SENSOR_ID

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
        if external_sim is not None:
            self._sim = external_sim
            self._owns_sim = False
        else:
            res = Resolution2D(width=self._sensor_w, height=self._sensor_h)
            sensor_configs = {
                _SENSOR_ID: SensorConfig(resolution=res, zoom=zoom, semantic=True)
            }
            agent_factory = partial(
                SurfaceAgent, agent_id=self._agent_id_local, sensor_configs=sensor_configs,
            )
            self._sim = MuJoCoSimulator(
                agents=[agent_factory], data_path=mujoco_data_path,
            )
            self._sim.add_object(mujoco_object_name)
            self._sim.model.vis.global_.offwidth = 256
            self._sim.model.vis.global_.offheight = 256
            self._owns_sim = True

        logger.info("MuJoCo initialized: object='%s'", mujoco_object_name)

        # ═══ Monty transforms ═══
        self._missing_to_max = MissingToMaxDepth(
            agent_id=self._agent_id_local, max_depth=1.0, threshold=0.0,
        )
        self._depth_to_3d = DepthTo3DLocations(
            agent_id=self._agent_id_local, sensor_ids=[self._sensor_id_local],
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
        edge_point_cad = np.zeros(3)
        edge_point_cad[temp.height_axis] = cad_edge
        edge_point_mj = self._pos_cad_to_mj_mm(edge_point_cad)
        self.open_edge_height = edge_point_mj[self.height_axis]

        # ═══ FIX 6: Cache bottom height ═══
        cad_bounds = self._cad_mesh.bounds  # [min, max]
        mj_corners = np.array([
            self._pos_cad_to_mj_mm(np.array([x, y, z]))
            for x in [cad_bounds[0][0], cad_bounds[1][0]]
            for y in [cad_bounds[0][1], cad_bounds[1][1]]
            for z in [cad_bounds[0][2], cad_bounds[1][2]]
        ])
        h = self.height_axis
        if self.up_sign > 0:
            self._bottom_height_mj = float(mj_corners.min(axis=0)[h])
        else:
            self._bottom_height_mj = float(mj_corners.max(axis=0)[h])

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
        return self._sim._agents[self._agent_id_local]._embodiment

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
            agent_id=self._agent_id_local, location=pos_m, rotation_quat=quat_wxyz,
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
            agent_id=self._agent_id_local, location=tuple(pos_m),
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

        sensor_obs = obs[self._agent_id_local][self._sensor_id_local]
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
        
        # Ray cast depth override for shared simulator
        if not self._owns_sim:
            pos = self._get_pos_mj_mm()
            euler = self._get_euler_deg()
            rot_check = Rot.from_euler("xyz", euler, degrees=True)
            fwd = rot_check.apply([0, 0, -1])
            ray_depth = self._mj_ray_cast(pos, fwd)
            if ray_depth > 0:
                depth_mm = ray_depth
                on_object = ray_depth < ON_OBJECT_DEPTH_MM
                if point_normal is None and on_object:
                    point_normal = (-fwd).tolist()

        return {
            "depth_mm": depth_mm, "point_normal": point_normal,
            "k1_mm": k1_mm, "k2_mm": k2_mm, "on_object": on_object,
        }

    # ═══════════════════════════════════════════════════
    # same_side (MuJoCo frame)
    # ═══════════════════════════════════════════════════

    def _compute_same_side(self, agent_normal: Optional[List[float]]) -> bool:
        if self._goal_normal_mj is None:
            return True
        
        center = self._mj_center_mm
        agent_pos = self._get_pos_mj_mm()
        goal_pos = self._current_goal[:3]
        h = self.height_axis
        up = self.up_direction
        
        # ═══ Agent side (position-based, no camera dependency) ═══
        agent_inside = self._is_point_inside_mj(agent_pos)
        agent_outward = not agent_inside
        
        # ═══ Goal side (on surface — reliable) ═══
        gn = np.array(self._goal_normal_mj, dtype=float)
        gn_h = gn.copy(); gn_h[h] = 0.0
        gfc = goal_pos - center; gfc[h] = 0.0
        
        if np.linalg.norm(gn_h) >= 0.3:
            goal_outward = np.dot(gn_h, gfc) > 0
        else:
            goal_outward = np.dot(gn, up) < 0
        
        return agent_outward == goal_outward

    def _is_point_inside_mj(self, pos_mj_mm):
        """Position-based inside/outside test via horizontal ray cast."""
        center = self._mj_center_mm
        h = self.height_axis

        # Above rim = outside
        point_height = pos_mj_mm[h]
        rim_height = self.open_edge_height
        if (point_height - rim_height) * self.up_sign > 0:
            return False

        # Below bottom = outside
        if (self._bottom_height_mj - point_height) * self.up_sign > 5.0:
            return False

        from_center = pos_mj_mm - center
        from_center_horiz = from_center.copy()
        from_center_horiz[h] = 0.0
        horiz_dist = float(np.linalg.norm(from_center_horiz))

        if horiz_dist < 1e-8:
            return True

        direction = -from_center_horiz / horiz_dist
        hit = self._mj_ray_cast(pos_mj_mm, direction)

        if hit > 0 and hit < horiz_dist + 5.0:
            return False

        return True
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
    def _get_goal_normal_runtime(self, goal_pos_mm):
        """Get approximate goal normal via ray cast toward object center."""
        center = self._mj_center_mm
        to_center = center - goal_pos_mm
        to_center_len = float(np.linalg.norm(to_center))
        if to_center_len < 1e-8:
            return [0.0, 1.0, 0.0]
        to_center_dir = to_center / to_center_len
        
        hit = self._mj_ray_cast(goal_pos_mm, to_center_dir)
        if hit > 0:
            return (-to_center_dir).tolist()
        return [0.0, 1.0, 0.0]
    
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
        if self._owns_sim:
            goal_pos_cad_mm = self._pos_mj_mm_to_cad_mm(goal_pose[:3])
            self._goal_normal_mj = self._get_goal_normal_mj(goal_pos_cad_mm)
        else:
            self._goal_normal_mj = self._get_goal_normal_runtime(goal_pose[:3])

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
            "open_edge_height": self.open_edge_height,
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

    # ═══════════════════════════════════════════════════
    # RLEnvironment protocol implementation
    # ═══════════════════════════════════════════════════

    def step_discrete(self, action_idx: int, action_space) -> dict:
        """Execute discrete action. Delegates to existing step().

        Args:
            action_idx: Discrete action index (0-23).
            action_space: ActionSpace instance.

        Returns:
            Sensor data after action.
        """
        return self.step(action_idx, action_space)

    def step_continuous(
        self, action_type: int, action_params: np.ndarray
    ) -> dict:
        """Execute continuous SAC action in MuJoCo.

        Maps (type, params) to MuJoCo primitives with the same
        semantics as ActionInterpreter.execute() for trimesh.
        This is the key method that enables SAC continuous params
        to work in MuJoCo (previously lost via sac_to_discrete).

        Args:
            action_type: PSAC action type (0-7).
            action_params: Continuous parameters array [3].

        Returns:
            Sensor data after action.
        """
        self._detach_had_collision = False
        self._edge_traversed = False
        self._passed_through = False

        pre = self._render_and_extract()
        self._prev_normal = pre["point_normal"]

        if action_type == 0:
            # Tangential surface move: [sin_angle, cos_angle, distance]
            sin_a = float(action_params[0])
            cos_a = float(action_params[1])
            angle_deg = float(np.degrees(np.arctan2(sin_a, cos_a)))
            distance = float(np.clip(action_params[2], 0.5, 15.0))
            self._do_move_tangentially(angle_deg, distance)

        elif action_type == 1:
            # Linear move: [distance, -, -]
            distance = float(np.clip(action_params[0], -25.0, 25.0))
            self._do_move_forward(distance)

        elif action_type == 2:
            # Yaw (turn left/right): [sin_angle, cos_angle, -]
            sin_a = float(action_params[0])
            cos_a = float(action_params[1])
            rotation = float(np.degrees(np.arctan2(sin_a, cos_a)))
            rotation = float(np.clip(rotation, -45.0, 45.0))
            self._apply_rotation_delta("y", rotation)

        elif action_type == 3:
            # Pitch (look up/down): [sin_angle, cos_angle, -]
            sin_a = float(action_params[0])
            cos_a = float(action_params[1])
            rotation = float(np.degrees(np.arctan2(sin_a, cos_a)))
            rotation = float(np.clip(rotation, -45.0, 45.0))
            self._apply_rotation_delta("x", rotation)

        elif action_type == 4:
            # Roll (tilt): [sin_angle, cos_angle, -]
            sin_a = float(action_params[0])
            cos_a = float(action_params[1])
            rotation = float(np.degrees(np.arctan2(sin_a, cos_a)))
            rotation = float(np.clip(rotation, -45.0, 45.0))
            self._apply_rotation_delta("z", rotation)

        elif action_type == 5:
            # Orient horizontal: [rotation_deg, left_dist, fwd_dist]
            rotation = float(action_params[0])
            left_dist = float(action_params[1])
            fwd_dist = float(action_params[2])
            self._do_orient_horizontal(rotation, fwd_dist, left_dist)

        elif action_type == 6:
            # Orient vertical: [rotation_deg, down_dist, fwd_dist]
            rotation = float(action_params[0])
            down_dist = float(action_params[1])
            fwd_dist = float(action_params[2])
            self._do_orient_vertical(rotation, fwd_dist, down_dist)

        elif action_type == 7:
            # Detach: params ignored, uses current goal
            if self._current_goal is not None:
                # Detach distance matches discrete version:
                # action_space.free_step * 3 = 8.0 * 3 = 24.0
                self._do_detach(self._current_goal, 24.0)

        # Edge detection (same logic as step())
        post = self._render_and_extract()
        if (
            self._prev_normal is not None
            and post["point_normal"] is not None
        ):
            dot = float(
                np.dot(
                    np.array(self._prev_normal),
                    np.array(post["point_normal"]),
                )
            )
            if dot < 0.707:
                self._edge_traversed = True

        return self.get_sensor_data()

    @property
    def supports_continuous(self) -> bool:
        """MuJoCo supports continuous actions."""
        return True

    @property
    def supports_offline_retrain(self) -> bool:
        """MuJoCo is too slow for offline retrain.

        Adaptive manager should use trimesh (LightweightEnv)
        for offline Q-store and SAC retraining.
        """
        return False
    
    def _do_move_tangentially(self, direction_degrees, step_mm):
        """Tangential surface move with snap and rollback.

        Mirrors LightweightEnv._move_tangentially:
        1. Build tangent basis from current normal
        2. Compute world direction from direction_degrees
        3. Move agent along tangent plane
        4. Snap to surface with normal consistency
        5. If snap fails, try half-step for edge traversal
        6. If all fails, rollback to original position
        """
        pos = self._get_pos_mj_mm()
        euler = self._get_euler_deg()
        rot = Rot.from_euler("xyz", euler, degrees=True)

        # Get current normal BEFORE move
        rendered = self._render_and_extract()
        normal = rendered["point_normal"]

        if normal is None:
            # No surface — move in local direction (like LightweightEnv)
            angle_rad = np.radians(direction_degrees)
            local_dir = np.array(
                [np.sin(angle_rad), 0.0, -np.cos(angle_rad)]
            )
            local_dir /= (np.linalg.norm(local_dir) + 1e-12)
            world_dir = rot.apply(local_dir)
            new_pos = pos + world_dir * step_mm
            self._set_pose_mj_mm(new_pos, euler)
            return

        n = np.array(normal, dtype=float)
        n /= (np.linalg.norm(n) + 1e-12)

        # ═══ Build tangent basis (same as LightweightEnv) ═══
        right_world = rot.apply([1.0, 0.0, 0.0])
        t1 = right_world - np.dot(right_world, n) * n
        t1_norm = np.linalg.norm(t1)

        if t1_norm < 1e-8:
            up_world = rot.apply([0.0, 1.0, 0.0])
            t1 = up_world - np.dot(up_world, n) * n
            t1_norm = np.linalg.norm(t1)

        if t1_norm < 1e-8:
            tmp = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(tmp, n)) > 0.9:
                tmp = np.array([0.0, 0.0, 1.0])
            t1 = np.cross(n, tmp)
            t1_norm = np.linalg.norm(t1)

        t1 /= (t1_norm + 1e-12)
        t2 = np.cross(n, t1)
        t2 /= (np.linalg.norm(t2) + 1e-12)

        # ═══ Compute tangent direction ═══
        a = np.radians(direction_degrees)
        world_dir = np.cos(a) * t1 + np.sin(a) * t2
        world_dir /= (np.linalg.norm(world_dir) + 1e-12)

        # ═══ Save old state for rollback ═══
        old_pos = pos.copy()
        old_euler = euler.copy()

        # ═══ Move ═══
        new_pos = pos + world_dir * step_mm
        self._set_pose_mj_mm(new_pos, euler)

        # ═══ Snap with normal consistency ═══
        snap_ok = self._snap_to_surface(prev_normal=normal)

        if snap_ok:
            return  # snap_to_surface verified ray cast and normal — trust it

        # ═══ Snap failed — try half-step (edge traversal) ═══
        half_pos = old_pos + world_dir * step_mm * 0.5
        self._set_pose_mj_mm(half_pos, old_euler)

        snap_half = self._snap_to_surface(prev_normal=normal)

        if snap_half:
            half_rendered = self._render_and_extract()
            half_normal = half_rendered["point_normal"]
            pos_after_half = self._get_pos_mj_mm()
            euler_after_half = self._get_euler_deg()

            full_pos = pos_after_half + world_dir * step_mm * 0.5
            self._set_pose_mj_mm(full_pos, euler_after_half)

            snap_full = self._snap_to_surface(prev_normal=half_normal)

            if snap_full:
                self._edge_traversed = True
                return

        # ═══ All attempts failed — rollback ═══
        self._set_pose_mj_mm(old_pos, old_euler)

    def _snap_to_surface(self, prev_normal=None):
        """Snap agent to nearest surface after tangential move.

        Emulates trimesh nearest.on_surface logic:
            closest, _, face_id = mesh.nearest.on_surface([pos])
            hit_n = mesh.face_normals[face_id]
            agent_pos = closest + hit_n * 2.0
            agent_rot = look_at(-hit_n)

        In MuJoCo we use ray casts instead of nearest.on_surface:
        1. Cast rays in multiple directions to find surface
        2. Move to SNAP_TARGET_DEPTH_MM from surface
        3. Orient camera toward surface (best_dir) so render can see it
        4. Render to get actual surface normal
        5. Re-orient by actual normal (-normal_arr)
        6. Validate final orientation; fallback to best_dir if needed

        Args:
            prev_normal: Surface normal before the move (for consistency
                check). None if no previous normal available.

        Returns:
            True if snapped successfully (agent on surface, camera
            oriented toward it). False if no surface found (caller
            should rollback).
        """
        pos = self._get_pos_mj_mm()
        euler = self._get_euler_deg()
        rot = Rot.from_euler("xyz", euler, degrees=True)
        forward = rot.apply([0, 0, -1])
        pos_after_move = pos.copy()

        # ═══ Step 1: Find surface via ray casts ═══
        candidates = []

        # 1a. Forward (current camera direction)
        hit = self._mj_ray_cast(pos, forward)
        if 0 < hit < self._snap_max_dist:
            candidates.append((forward.copy(), hit))

        # 1b. Toward previous surface (-prev_normal)
        if prev_normal is not None:
            prev_n = np.array(prev_normal, dtype=float)
            prev_n /= (np.linalg.norm(prev_n) + 1e-12)
            hit = self._mj_ray_cast(pos, -prev_n)
            if 0 < hit < self._snap_max_dist:
                candidates.append((-prev_n.copy(), hit))

        # 1c. Multi-probe cone around forward
        if not candidates:
            hit, probe_dir = self._multi_probe_ray_cast(
                pos, forward, rot,
                max_dist=self._snap_max_dist, surface_step=3.0,
            )
            if 0 < hit < self._snap_max_dist:
                candidates.append((probe_dir.copy(), hit))

        # 1d. Multi-probe cone around -prev_normal
        if not candidates and prev_normal is not None:
            prev_n = np.array(prev_normal, dtype=float)
            prev_n /= (np.linalg.norm(prev_n) + 1e-12)
            neg_n = -prev_n
            try:
                align_rot, _ = Rot.align_vectors([neg_n], [[0, 0, -1]])
            except Exception:
                align_rot = rot
            hit, probe_dir = self._multi_probe_ray_cast(
                pos, neg_n, align_rot,
                max_dist=self._snap_max_dist, surface_step=3.0,
            )
            if 0 < hit < self._snap_max_dist:
                candidates.append((probe_dir.copy(), hit))

        # 1e. Toward object center (last resort for edges/rims)
        if not candidates:
            to_center = self._mj_center_mm - pos
            to_center_dist = np.linalg.norm(to_center)
            if to_center_dist > 1e-8:
                to_center_dir = to_center / to_center_dist
                hit = self._mj_ray_cast(pos, to_center_dir)
                if 0 < hit < self._snap_max_dist:
                    candidates.append((to_center_dir.copy(), hit))

        if not candidates:
            return False

        # ═══ Step 2: Compute snap position ═══
        candidates.sort(key=lambda c: c[1])
        best_dir, best_dist = candidates[0]

        approach = best_dist - SNAP_TARGET_DEPTH_MM
        if abs(approach) > 0.3:
            snap_pos = pos + best_dir * approach
        else:
            snap_pos = pos.copy()

        # ═══ Step 3: Orient toward found surface ═══
        # Key fix: orient camera toward surface BEFORE rendering.
        # Old code kept old euler → camera could miss surface on edges.
        # This mirrors trimesh: agent_rot = look_at(-hit_n)
        # where best_dir ≈ -hit_n (points from agent toward surface).
        snap_euler = self._look_at_direction(best_dir)
        self._set_pose_mj_mm(snap_pos, snap_euler)

        # ═══ Step 4: Render to get actual surface normal ═══
        rendered = self._render_and_extract()

        if rendered["depth_mm"] >= NO_SURFACE_DEPTH_MM:
            # Even looking toward best_dir we don't see surface.
            # Ray cast found a hit but render disagrees — snap failed.
            return False

        new_normal = rendered["point_normal"]

        if new_normal is None:
            # Surface visible (depth < 100) but normal not computed
            # at center pixel. Agent is near surface with camera
            # pointing at it → depth < 3mm → on_object = True.
            # Keep best_dir orientation — it's our best estimate.
            return True

        normal_arr = np.array(new_normal, dtype=float)
        n_len = np.linalg.norm(normal_arr)
        if n_len < 1e-8:
            # Degenerate normal — keep best_dir orientation.
            return True

        normal_arr /= n_len

        # ═══ Step 5: Normal consistency check ═══
        # Mirrors trimesh logic:
        #   if np.dot(hit_n, n) < 0: hit_n = -hit_n
        #   can_transition = np.dot(hit_n, n) > -0.1
        if prev_normal is not None:
            prev_n = np.array(prev_normal, dtype=float)
            prev_n /= (np.linalg.norm(prev_n) + 1e-12)
            dot = float(np.dot(normal_arr, prev_n))

            if dot < -0.1:
                # Normal flipped >~95° — try to stay on same side.
                # Mirrors trimesh edge traversal: try alt position
                # along prev_normal direction.
                prev_hit = self._mj_ray_cast(pos_after_move, -prev_n)
                if 0 < prev_hit < 5.0:
                    approach2 = prev_hit - SNAP_TARGET_DEPTH_MM
                    alt_pos = pos_after_move - prev_n * approach2

                    # Collision check for alternative position
                    alt_vec = alt_pos - pos_after_move
                    alt_dist = float(np.linalg.norm(alt_vec))
                    safe = True
                    if alt_dist > 0.5:
                        alt_check_dir = alt_vec / alt_dist
                        alt_hit = self._mj_ray_cast(
                            pos_after_move, alt_check_dir
                        )
                        if 0 < alt_hit < alt_dist - 0.5:
                            safe = False

                    if safe:
                        alt_euler = self._look_at_direction(-prev_n)
                        self._set_pose_mj_mm(alt_pos, alt_euler)
                        check = self._render_and_extract()
                        if check["depth_mm"] < NO_SURFACE_DEPTH_MM:
                            return True

                # Could not stay on same side — accept edge traversal.
                # This is legitimate (e.g. mug rim transition).
                self._edge_traversed = True
                # normal_arr is the measured normal on the new side.
                # Don't flip it — it's correct for the new surface.

            elif dot < 0:
                # Slight flip (between -0.1 and 0) — correct sign.
                normal_arr = -normal_arr

        # ═══ Step 6: Final orientation by actual normal ═══
        # Mirrors trimesh: agent_rot = look_at(-hit_n)
        final_euler = self._look_at_direction(-normal_arr)
        self._set_pose_mj_mm(snap_pos, final_euler)

        # Validate: does camera still see surface after reorientation?
        # On curved surfaces, -normal may point slightly away from
        # where the surface actually is relative to snap_pos.
        final_check = self._render_and_extract()
        if final_check["depth_mm"] >= NO_SURFACE_DEPTH_MM:
            # Normal-based orientation lost the surface.
            # Fall back to snap_euler (best_dir) which we verified
            # in Step 4 — it definitely sees the surface.
            self._set_pose_mj_mm(snap_pos, snap_euler)

        return True

    def _estimate_extents_runtime(self):
        """Estimate object extents via ray cast from outside toward center."""
        center = self._mj_center_mm
        half_extents = np.zeros(3)
        far_dist = 500.0

        for axis in range(3):
            for sign in [1.0, -1.0]:
                direction = np.zeros(3)
                direction[axis] = -sign

                origin = center.copy()
                origin[axis] += sign * far_dist

                hit = self._mj_ray_cast(origin, direction)
                if 0 < hit < far_dist:
                    half_extents[axis] = max(half_extents[axis], far_dist - hit)

        # Also probe at offsets to catch angled surfaces
        for axis in range(3):
            other_axes = [i for i in range(3) if i != axis]
            for offset in [-20.0, 0.0, 20.0]:
                for sign in [1.0, -1.0]:
                    direction = np.zeros(3)
                    direction[axis] = -sign

                    origin = center.copy()
                    origin[axis] += sign * far_dist
                    origin[other_axes[0]] += offset

                    hit = self._mj_ray_cast(origin, direction)
                    if 0 < hit < far_dist:
                        half_extents[axis] = max(half_extents[axis], far_dist - hit)

        extents = half_extents * 2.0

        for axis in range(3):
            if extents[axis] < 1.0:
                extents[axis] = 100.0

        return extents
            
    def _snap_to_surface_old1(self, prev_normal=None):
        """Find nearest surface and snap agent to it.

        Strategy: cast rays in multiple directions, pick closest hit.
        Mimics trimesh nearest.on_surface but via directional ray casts.

        Order:
            1. Forward (camera direction)
            2. -prev_normal (toward surface we came from)
            3. Multi-probe cone around forward
            4. Multi-probe cone around -prev_normal

        Returns True if snapped, False if no surface found (→ rollback).
        """
        pos = self._get_pos_mj_mm()
        euler = self._get_euler_deg()
        rot = Rot.from_euler("xyz", euler, degrees=True)
        forward = rot.apply([0, 0, -1])
        pos_after_move = pos.copy()

        # ═══ Collect candidates (direction, distance) ═══
        candidates = []

        # 1. Forward
        hit = self._mj_ray_cast(pos, forward)
        if 0 < hit < self._snap_max_dist:
            candidates.append((forward.copy(), hit))

        # 2. -prev_normal (toward surface we came from)
        if prev_normal is not None:
            prev_n = np.array(prev_normal, dtype=float)
            prev_n /= (np.linalg.norm(prev_n) + 1e-12)
            hit = self._mj_ray_cast(pos, -prev_n)
            if 0 < hit < self._snap_max_dist:
                candidates.append((-prev_n.copy(), hit))

        # 3. Multi-probe around forward
        if not candidates:
            hit, best_dir = self._multi_probe_ray_cast(
                pos, forward, rot,
                max_dist=self._snap_max_dist,
                surface_step=3.0,
            )
            if 0 < hit < self._snap_max_dist:
                candidates.append((best_dir.copy(), hit))

        # 4. Multi-probe around -prev_normal
        if not candidates and prev_normal is not None:
            prev_n = np.array(prev_normal, dtype=float)
            prev_n /= (np.linalg.norm(prev_n) + 1e-12)
            neg_n = -prev_n
            try:
                align_rot, _ = Rot.align_vectors(
                    [neg_n], [[0, 0, -1]]
                )
            except Exception:
                align_rot = rot
            hit, best_dir = self._multi_probe_ray_cast(
                pos, neg_n, align_rot,
                max_dist=self._snap_max_dist,
                surface_step=3.0,
            )
            if 0 < hit < self._snap_max_dist:
                candidates.append((best_dir.copy(), hit))

        if not candidates:
            return False

        # ═══ Pick closest candidate ═══
        candidates.sort(key=lambda c: c[1])
        best_dir, best_dist = candidates[0]

        # ═══ Compute snap position ═══
        approach = best_dist - SNAP_TARGET_DEPTH_MM
        if abs(approach) > 0.3:
            snap_pos = pos + best_dir * approach
        else:
            snap_pos = pos.copy()

        # ═══ Apply snap position ═══
        self._set_pose_mj_mm(snap_pos, euler)

        # ═══ Get normal from render ═══
        rendered = self._render_and_extract()

        if rendered["depth_mm"] >= NO_SURFACE_DEPTH_MM:
            return False

        new_normal = rendered["point_normal"]
        if new_normal is None:
            if prev_normal is not None:
                new_euler = self._look_at_direction(-np.array(prev_normal))
                self._set_pose_mj_mm(snap_pos, new_euler)
                return True
            return False

        normal_arr = np.array(new_normal, dtype=float)
        n_len = np.linalg.norm(normal_arr)
        if n_len < 1e-8:
            return False
        normal_arr /= n_len

        # ═══ Normal consistency (matches trimesh: dot > -0.1) ═══
        if prev_normal is not None:
            prev_n = np.array(prev_normal, dtype=float)
            prev_n /= (np.linalg.norm(prev_n) + 1e-12)
            dot = float(np.dot(normal_arr, prev_n))

            if dot < -0.1:
                # Normal flipped — try to stay on same side
                prev_hit = self._mj_ray_cast(pos_after_move, -prev_n)
                if 0 < prev_hit < 5.0:
                    approach2 = prev_hit - SNAP_TARGET_DEPTH_MM
                    alt_pos = pos_after_move - prev_n * approach2

                    # Collision check for alternative position
                    alt_vec = alt_pos - pos_after_move
                    alt_dist = float(np.linalg.norm(alt_vec))
                    safe = True
                    if alt_dist > 0.5:
                        alt_dir = alt_vec / alt_dist
                        alt_hit = self._mj_ray_cast(
                            pos_after_move, alt_dir
                        )
                        if 0 < alt_hit < alt_dist - 0.5:
                            safe = False

                    if safe:
                        new_euler = self._look_at_direction(-prev_n)
                        self._set_pose_mj_mm(alt_pos, new_euler)
                        rendered = self._render_and_extract()
                        if rendered["depth_mm"] < NO_SURFACE_DEPTH_MM:
                            return True

                # Could not stay on same side — accept edge traversal
                self._edge_traversed = True
                normal_arr = -normal_arr

            elif dot < 0:
                normal_arr = -normal_arr

        new_euler = self._look_at_direction(-normal_arr)
        self._set_pose_mj_mm(snap_pos, new_euler)
        return True

    def _multi_probe_ray_cast(
        self,
        pos: np.ndarray,
        forward: np.ndarray,
        rot,
        max_dist: float = 10.0,
        surface_step: float = 3.0,
    ) -> tuple:
        """Cast rays in a cone around forward direction.

        Probes at 15°, 30°, 45°, 60° from forward in 8 directions.
        Returns closest hit within distance limits.

        Args:
            pos: Ray origin in MuJoCo mm.
            forward: Central direction of cone.
            rot: Rotation for computing right/up axes.
            max_dist: Maximum hit distance to accept.
            surface_step: Base step size for distance scaling.

        Returns:
            (distance, direction) of closest hit, or (-1.0, forward).
        """
        right = rot.apply([1, 0, 0])
        up = rot.apply([0, 1, 0])

        probe_angles = [15, 30, 45, 60]

        probe_dirs = [
            up, -up, right, -right,
            (up + right) / np.sqrt(2),
            (up - right) / np.sqrt(2),
            (-up + right) / np.sqrt(2),
            (-up - right) / np.sqrt(2),
        ]

        # Collect ALL hits, pick closest globally
        best_dist = max_dist
        best_direction = None

        for angle_deg in probe_angles:
            angle_rad = np.radians(angle_deg)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            for probe_dir in probe_dirs:
                probed = forward * cos_a + probe_dir * sin_a
                probed /= (np.linalg.norm(probed) + 1e-12)

                hit = self._mj_ray_cast(pos, probed)
                if 0 < hit < best_dist:
                    best_dist = hit
                    best_direction = probed

        if best_direction is not None:
            return best_dist, best_direction

        return -1.0, forward
    
    def _snap_to_surface_old(self, prev_normal=None):
        pos = self._get_pos_mj_mm()  # позиция после tangential move
        euler = self._get_euler_deg()
        rot = Rot.from_euler("xyz", euler, degrees=True)
        forward = rot.apply([0, 0, -1])

        # Save position before any approach
        pos_after_move = pos.copy()

        hit_dist = self._mj_ray_cast(pos, forward)

        if hit_dist < 0 or hit_dist >= 10.0:
            # Surface not found forward — fallback
            if prev_normal is not None:
                prev_n = np.array(prev_normal, dtype=float)
                prev_n /= (np.linalg.norm(prev_n) + 1e-12)
                prev_hit = self._mj_ray_cast(pos, -prev_n)
                if prev_hit > 0 and prev_hit < 10.0:
                    approach = prev_hit - SNAP_TARGET_DEPTH_MM
                    new_pos = pos - prev_n * approach
                    new_euler = self._look_at_direction(-prev_n)
                    self._set_pose_mj_mm(new_pos, new_euler)
                    return True
            return False

        # Approach to target depth
        if abs(hit_dist - SNAP_TARGET_DEPTH_MM) > 0.3:
            approach = hit_dist - SNAP_TARGET_DEPTH_MM
            pos = pos + forward * approach
            self._set_pose_mj_mm(pos, euler)

        # Get normal from render
        rendered = self._render_and_extract()
        new_normal = rendered["point_normal"]

        if new_normal is None:
            if prev_normal is not None:
                new_euler = self._look_at_direction(-np.array(prev_normal))
                self._set_pose_mj_mm(pos, new_euler)
                return True
            return False

        normal_arr = np.array(new_normal, dtype=float)
        n_len = np.linalg.norm(normal_arr)
        if n_len < 1e-8:
            return False
        normal_arr /= n_len

        if prev_normal is not None:
            prev_n = np.array(prev_normal, dtype=float)
            prev_n /= (np.linalg.norm(prev_n) + 1e-12)
            dot = float(np.dot(normal_arr, prev_n))

            if dot < -0.1:
                # Normal flipped — use pos_after_move (NOT pos after approach)
                # Ray cast from MOVED position in prev_normal direction
                prev_hit = self._mj_ray_cast(pos_after_move, -prev_n)
                if prev_hit > 0 and prev_hit < 5.0:
                    approach = prev_hit - SNAP_TARGET_DEPTH_MM
                    new_pos = pos_after_move - prev_n * approach
                    new_euler = self._look_at_direction(-prev_n)
                    self._set_pose_mj_mm(new_pos, new_euler)
                    return True
                else:
                    # Can't find surface — accept flipped normal
                    # (legitimate edge transition)
                    normal_arr = -normal_arr

            elif dot < 0:
                normal_arr = -normal_arr

        new_euler = self._look_at_direction(-normal_arr)
        self._set_pose_mj_mm(pos, new_euler)
        return True

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
        action = MoveForward(agent_id=self._agent_id_local, distance=distance_m)
        self._sim.step([action])

        self._passed_through = False

        if abs(step_mm) > 0.5:
            move_dir = forward * np.sign(step_mm)
            hit_dist = self._mj_ray_cast(old_pos, move_dir)
            if hit_dist > 0 and hit_dist < abs(step_mm):
                self._passed_through = True

    def _do_orient_horizontal(self, rotation_deg, forward_mm, left_mm):
        action = OrientHorizontal(
            agent_id=self._agent_id_local, rotation_degrees=-rotation_deg,
            left_distance=left_mm / MM_PER_M, forward_distance=forward_mm / MM_PER_M,
        )
        self._sim.step([action])

    def _do_orient_vertical(self, rotation_deg, forward_mm, down_mm):
        action = OrientVertical(
            agent_id=self._agent_id_local, rotation_degrees=rotation_deg,
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
    def render_episode_frame(
        self,
        agent_pose: np.ndarray,
        goal_pose: np.ndarray,
        filepath: str,
        trail_poses: list[np.ndarray] | None = None,
        text: str = "",
        step_num: int = 0,
        distance: float = 0.0,
        result: str = "",
    ) -> None:
        """Render split-view frame: MuJoCo solid + MuJoCo x-ray."""
        import mujoco
        from mujoco import mj_forward
        from PIL import Image as _Image, ImageDraw as _ImageDraw, ImageFont as _ImageFont
        from pathlib import Path as _Path
        from .visualize_env import add_text_overlay

        mj_forward(self._sim.model, self._sim.data)
        _Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Render at framebuffer size
        fb_w = self._sim.model.vis.global_.offwidth
        fb_h = self._sim.model.vis.global_.offheight
        render_half_w = min(fb_w, 256)
        render_h = min(fb_h, 256)

        # Target output size (upscale if needed)
        target_half_w = 256
        target_h = 256
        target_w = target_half_w * 2

        # Renderer
        if self._scene_renderer is None:
            self._scene_renderer = mujoco.Renderer(
                self._sim.model, height=render_h, width=render_half_w
            )

        # Camera setup
        agent_m = np.array(self._embodiment.position, dtype=float)
        goal_m = goal_pose[:3] / MM_PER_M

        max_ext = float(max(self._mj_extents_mm)) / MM_PER_M
        sphere_size = max_ext * 0.03
        trail_size = sphere_size * 0.25

        midpoint = (agent_m + goal_m) / 2
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

        def _build_scene_geoms(scene):
            agent_m_local = agent_pose[:3] / MM_PER_M

            # Trail
            if trail_poses:
                trail = trail_poses[-50:]
                for t_pos in trail:
                    if scene.ngeom >= scene.maxgeom:
                        break
                    t_m = np.array(t_pos[:3], dtype=float) / MM_PER_M
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        [trail_size, 0, 0], t_m, np.eye(3).flatten(),
                        [1.0, 0.65, 0.0, 0.8],
                    )
                    scene.ngeom += 1

            # Agent (blue)
            if scene.ngeom < scene.maxgeom:
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                    [sphere_size, 0, 0], agent_m_local, np.eye(3).flatten(),
                    [0.2, 0.2, 1.0, 1.0],
                )
                scene.ngeom += 1

            # Goal (green)
            if scene.ngeom < scene.maxgeom:
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                    [sphere_size * 1.3, 0, 0], goal_m, np.eye(3).flatten(),
                    [0.2, 1.0, 0.2, 1.0],
                )
                scene.ngeom += 1

            # Gaze (red)
            if scene.ngeom < scene.maxgeom:
                rot = Rot.from_euler("xyz", agent_pose[3:], degrees=True)
                fwd = rot.apply([0, 0, -1])
                gaze_end = agent_m_local + fwd * sphere_size * 2.5
                midpt = (agent_m_local + gaze_end) / 2
                d = gaze_end - agent_m_local
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
                        scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE,
                        [sphere_size * 0.15, length / 2, 0], midpt, rm.flatten(),
                        [1.0, 0.0, 0.0, 1.0],
                    )
                    scene.ngeom += 1

        def _render_view(transparent=False):
            model = self._sim.model
            max_geom = 100
            scene = mujoco.MjvScene(model, maxgeom=max_geom)

            opt = mujoco.MjvOption()
            if transparent:
                opt.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = False
                scene_flags_wireframe = True
            else:
                scene_flags_wireframe = False

            mujoco.mjv_updateScene(
                model, self._sim.data, opt, None,
                camera, mujoco.mjtCatBit.mjCAT_ALL, scene,
            )

            if scene_flags_wireframe:
                scene.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = True

            _build_scene_geoms(scene)

            mujoco.mjr_render(
                mujoco.MjrRect(0, 0, render_half_w, render_h),
                scene, self._scene_renderer._mjr_context,
            )
            rgb = np.empty((render_h, render_half_w, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(
                rgb, None, mujoco.MjrRect(0, 0, render_half_w, render_h),
                self._scene_renderer._mjr_context,
            )
            return np.flipud(rgb)

        try:
            rgb_solid = _render_view(transparent=False)
            rgb_xray = _render_view(transparent=True)

            img_left = _Image.fromarray(rgb_solid)
            img_right = _Image.fromarray(rgb_xray)

            # Upscale to target size
            img_left = img_left.resize((target_half_w, target_h), _Image.LANCZOS)
            img_right = img_right.resize((target_half_w, target_h), _Image.LANCZOS)

            merged = _Image.new("RGB", (target_w, target_h))
            merged.paste(img_left, (0, 0))
            merged.paste(img_right, (target_half_w, 0))

            # Adaptive font size based on image dimensions
            font_size = max(target_h // 25, 8)
            try:
                font = _ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size
                )
            except OSError:
                font = _ImageFont.load_default()

            if text:
                # Draw text with adaptive font
                draw = _ImageDraw.Draw(merged)
                # Wrap text to fit image width
                max_chars = target_w // (font_size * 0.6)
                lines = []
                for line in text.split("\n"):
                    while len(line) > max_chars:
                        lines.append(line[:int(max_chars)])
                        line = line[int(max_chars):]
                    lines.append(line)

                y_pos = 5
                for line in lines[:6]:  # max 6 lines
                    draw.text((5, y_pos), line, fill="white", font=font)
                    y_pos += font_size + 2

            # Labels
            draw = _ImageDraw.Draw(merged)
            label_font_size = max(font_size - 2, 8)
            try:
                label_font = _ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", label_font_size
                )
            except OSError:
                label_font = font
            draw.text((5, target_h - label_font_size - 5), "SOLID", fill="white", font=label_font)
            draw.text((target_half_w + 5, target_h - label_font_size - 5), "X-RAY", fill="white", font=label_font)

            merged.save(filepath, format="PNG")

        except Exception:
            logger.debug("MuJoCo render failed for %s", filepath, exc_info=True)

    def save_mujoco_frame(self, filepath: str, save_depth: bool = False):
        from mujoco import mj_forward
        from PIL import Image
        mj_forward(self._sim.model, self._sim.data)
        cam_name = f"{self._agent_id_local}.{self._sensor_id_local}"
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
        if self._owns_sim and self._sim is not None:
            self._sim.close()
            self._sim = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
