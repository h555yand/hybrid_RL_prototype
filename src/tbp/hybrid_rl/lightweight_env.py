# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


class LightweightEnv:
    """A lightweight simulator for training/testing RL without Habitat.
    Uses trimesh for:
    - Mesh loading (STL, OBJ, PLY)
    - Ray casting (depth, normal)
    - Collision detection
    Does not use:
    - Habitat
    - GPU
    """

    def __init__(self, mesh_path: str, seed=None):
        self.mesh = trimesh.load(mesh_path)
        # all numbers are mm
        # z+ is out of the page, x+ is "right", and y+ is "up"
        self.agent_pos = np.zeros(3)
        # euler angles
        self.agent_rot = np.zeros(3)
        # Pitch - turn up/down - rotation around the X axis
        # Yaw - turn left/right - rotation around the Y axis
        # Roll - turn head to shoulder - rotation around the Z axis
        # self.agent_rot = [Pitch, Yaw, Roll]
        # self.agent_rot = [X-angle, Y-angle, Z-angle]
        # Note: seed is set globally in train() => np.random.seed()
        # trimesh.sample() uses global np.random
        self._passed_through = False
        self._detach_had_collision = False
        self._compute_up_direction()
        self._mesh_centroid = np.array(self.mesh.centroid, dtype=float)
        self._wrong_side_outward = None

    @staticmethod
    def _normalize_euler(angles):
        return (np.array(angles, dtype=float) + 180.0) % 360.0 - 180.0

    def reset(self, position=None, rotation=None):
        """Place the agent in the starting position."""
        self._passed_through = False
        self._detach_had_collision = False
        self._wrong_side_outward = None
        if position is not None:
            self.agent_pos = np.array(position, dtype=float)
        else:
            # Random point on surface (using controlled RNG)
            points, face_ids = self.mesh.sample(1, return_index=True)

            normal = self.mesh.face_normals[face_ids[0]]
            self.agent_pos = points[0] + normal * 2.0  # 2 mm from the surface outwards

            # If rotation is not specified, look towards the surface
            if rotation is None:
                self.agent_rot = self._look_at_direction(-normal)

        if rotation is not None:
            self.agent_rot = np.array(rotation, dtype=float)
        elif position is not None:
            # If the position was set manually, but the rotation was not, you can leave it at 0
            self.agent_rot = np.zeros(3)

        self.agent_rot = self._normalize_euler(self.agent_rot)
        return self.get_sensor_data()

    def set_goal(self, goal_pose):
        self._current_goal = goal_pose

    def step(self, action_index, action_space):
        """Perform an action, return new sensor data.

        Args:
            action_index: action index (0-23)
            action_space: ActionSpace

        Returns:
            sensor_data: point_normal, on_object, depth
        """
        self._detach_had_collision = False
        self._edge_traversed = False

        action_info = action_space.get_info(action_index)

        if action_info.name in ("look_up", "look_down", "look_up_big", "look_down_big"):
            rot_before = R.from_euler("xyz", self.agent_rot, degrees=True)
            forward_before = rot_before.apply([0, 0, -1])

        if action_info.name == "move_tangentially":
            self._move_tangentially(
                action_info.direction_degrees,
                action_space.surface_step,
            )
        elif action_info.name == "free_forward":
            self._move_forward(action_space.free_step)
        elif action_info.name == "free_backward":
            self._move_forward(-action_space.free_step_backward)
        elif action_info.name == "look_up":
            self.agent_rot[0] += action_space.rotation_step
        elif action_info.name == "look_down":
            self.agent_rot[0] -= action_space.rotation_step
        elif action_info.name == "look_up_big":
            self.agent_rot[0] += action_space.rotation_step_big
        elif action_info.name == "look_down_big":
            self.agent_rot[0] -= action_space.rotation_step_big
        elif action_info.name == "rotate_sensor_+":
            self.agent_rot[2] += action_space.rotation_step
        elif action_info.name == "rotate_sensor_-":
            self.agent_rot[2] -= action_space.rotation_step
        elif action_info.name == "turn_left":
            self.agent_rot[1] += action_space.rotation_step
        elif action_info.name == "turn_right":
            self.agent_rot[1] -= action_space.rotation_step
        elif action_info.name == "turn_left_big":
            self.agent_rot[1] += action_space.rotation_step_big
        elif action_info.name == "turn_right_big":
            self.agent_rot[1] -= action_space.rotation_step_big
        elif action_info.name == "orient_horizontal":
            self._orient_horizontal(
                rotation_degrees=action_info.rotation_degrees,
                forward_distance=action_info.forward_distance,
                left_distance=action_info.left_distance,
            )
        elif action_info.name == "orient_vertical":
            self._orient_vertical(
                rotation_degrees=action_info.rotation_degrees,
                forward_distance=action_info.forward_distance,
                down_distance=action_info.down_distance,
            )
        elif action_info.name == "detach":
            if hasattr(self, "_current_goal") and self._current_goal is not None:
                self._detach_simple(
                    goal_pose=self._current_goal,
                    detach_distance=action_space.free_step * 3,
                )
        elif action_info.name == "free_forward_small":
            self._move_forward(action_space.free_step_small)

        if action_info.name in ("look_up", "look_down", "look_up_big", "look_down_big"):
            rot_after = R.from_euler("xyz", self.agent_rot, degrees=True)
            forward_after = rot_after.apply([0, 0, -1])
            height_axis = getattr(self, "height_axis", 2)
            logger.debug(
                f"LOOK_DEBUG: action={action_info.name}, "
                f"forward_z_before={forward_before[height_axis]:.3f}, "
                f"forward_z_after={forward_after[height_axis]:.3f}, "
                f"agent_rot={self.agent_rot}"
            )
        self.agent_rot = self._normalize_euler(self.agent_rot)
        return self.get_sensor_data()

    def get_sensor_data(self):
        rot = R.from_euler("xyz", self.agent_rot, degrees=True)
        ray_direction = rot.apply([0, 0, -1])
        locations, index_ray, index_tri = self.mesh.ray.intersects_location(
            ray_origins=[self.agent_pos],
            ray_directions=[ray_direction],
        )

        if len(locations) > 0:
            distances = np.linalg.norm(locations - self.agent_pos, axis=1)
            nearest_idx = np.argmin(distances)

            depth = distances[nearest_idx]
            face_idx = index_tri[nearest_idx]
            point_normal = self.mesh.face_normals[face_idx].tolist()
            on_object = bool(depth < 3.0)
        else:
            depth = 100.0
            point_normal = None
            on_object = False

        goal_normal = None
        if hasattr(self, "_current_goal") and self._current_goal is not None:
            goal_pos = self._current_goal[:3]
            _, _, gf_id = self.mesh.nearest.on_surface([goal_pos])
            goal_normal = self.mesh.face_normals[gf_id[0]].tolist()

        # ═══ path_blocked (ray cast) ═══
        path_blocked = False
        if hasattr(self, "_current_goal") and self._current_goal is not None:
            goal_pos = self._current_goal[:3]
            direction = goal_pos - self.agent_pos
            dist_to_goal = np.linalg.norm(direction)

            if dist_to_goal > 1e-8:
                direction_norm = direction / dist_to_goal
                bl_locations, _, _ = self.mesh.ray.intersects_location(
                    ray_origins=[self.agent_pos],
                    ray_directions=[direction_norm],
                )
                if len(bl_locations) > 0:
                    hit_distances = np.linalg.norm(
                        bl_locations - self.agent_pos, axis=1
                    )
                    # Blocked if any intersection is closer than goal
                    # margin 2.0mm to avoid counting the goal surface itself
                    path_blocked = bool(np.min(hit_distances) < dist_to_goal - 2.0)

        # ═══ Curvature ═══
        curvature_data = self._estimate_curvature()

        same_side = True
        if hasattr(self, "_current_goal") and self._current_goal is not None:
            same_side = _is_reachable_by_surface(
                self, self.agent_pos, self._current_goal[:3]
            )

        return {
            "point_normal": point_normal,
            "k1": curvature_data["k1"],
            "k2": curvature_data["k2"],
            "principal_curvatures": [curvature_data["k1"], curvature_data["k2"]],
            "on_object": on_object,
            "depth": depth,
            "passed_through": getattr(self, "_passed_through", False),
            "goal_normal": goal_normal,
            "detach_had_collision": getattr(self, "_detach_had_collision", False),
            "detach_sub_steps": getattr(self, "_last_detach_sub_steps", 1),
            "path_blocked": path_blocked,
            "up_direction": self.up_direction.tolist(),
            "object_center": self.mesh.centroid.tolist(),
            "same_side": same_side,
            "object_extents": (self.mesh.bounds[1] - self.mesh.bounds[0]).tolist(),
            "edge_traversed": getattr(self, "_edge_traversed", False),
        }
    
    def get_pose(self):
        """The agent's current pose."""
        return np.concatenate([self.agent_pos, self.agent_rot])

    def get_random_surface_point(
        self,
        reference_pos: np.ndarray = None,
        min_dist: float = None,
        max_dist: float = None,
        max_attempts: int = 200,
        mesh_sample=False,
        same_cube_side=False
    ) -> np.ndarray:
        """A random point on the surface with an optional distance limit.

        Args:
        reference_pos: The reference point (the agent's position). If None, no distance filter is applied.
        min_dist: Minimum distance from reference_pos (mm, inclusive).
        max_dist: Maximum distance from reference_pos (mm, inclusive).
        max_attempts: Limit of rejection sampling attempts. If exhausted,
        returns any point without filtering (fallback).

        Returns:
        pose [6D]: [x, y, z, rx, ry, rz] points on the surface.
        """
        use_filter = (
            reference_pos is not None
            and (min_dist is not None or max_dist is not None)
        )

        if mesh_sample:
            points, face_ids = self.mesh.sample(max_attempts, return_index=True)
            for i in range(max_attempts):
                normal = self.mesh.face_normals[face_ids[i]]
                position = points[i] + normal * 2.0

                if use_filter:
                    dist = float(np.linalg.norm(position - reference_pos))
                    if min_dist is not None and dist < min_dist:
                        continue
                    if max_dist is not None and dist > max_dist:
                        continue
                    if same_cube_side == True:
                        same_cube_side_check = is_on_same_cube_side(position, reference_pos)
                        if same_cube_side_check == False:
                            continue

                rotation = self._look_at_direction(-normal)
                return np.concatenate([position, rotation])
        else:
            for _ in range(max_attempts):
                points, face_ids = self.mesh.sample(1, return_index=True)
                normal = self.mesh.face_normals[face_ids[0]]
                position = points[0] + normal * 2.0

                if use_filter:
                    dist = float(np.linalg.norm(position - reference_pos))
                    if min_dist is not None and dist < min_dist:
                        continue
                    if max_dist is not None and dist > max_dist:
                        continue

                rotation = self._look_at_direction(-normal)
                return np.concatenate([position, rotation])

        # Fallback: no point was within the range for max_attempts.
        # Return a random point.
        points, face_ids = self.mesh.sample(1, return_index=True)
        normal = self.mesh.face_normals[face_ids[0]]
        position = points[0] + normal * 2.0
        rotation = self._look_at_direction(-normal)
        return np.concatenate([position, rotation])

    def _estimate_curvature(self):
        """Estimate principal curvatures k1, k2 at the point the agent is looking at.

        Uses discrete mean and Gaussian curvature to recover principal curvatures:
            k1 = H + sqrt(H² - K)
            k2 = H - sqrt(H² - K)
        where H = mean curvature, K = Gaussian curvature.

        Convention: |k1| >= |k2| (k1 is the curvature of maximum bending).

        Returns:
            dict with 'k1' and 'k2' (floats), or {'k1': 0.0, 'k2': 0.0}
            if no surface is visible.
        """
        rot = R.from_euler("xyz", self.agent_rot, degrees=True)
        ray_direction = rot.apply([0, 0, -1])
        locations, index_ray, index_tri = self.mesh.ray.intersects_location(
            ray_origins=[self.agent_pos],
            ray_directions=[ray_direction],
        )
        if len(locations) == 0:
            return {"k1": 0.0, "k2": 0.0}

        distances = np.linalg.norm(locations - self.agent_pos, axis=1)
        nearest_idx = np.argmin(distances)
        face_idx = index_tri[nearest_idx]
        vertex_indices = self.mesh.faces[face_idx]

        if not hasattr(self, "_vertex_mean_curvature"):
            self._vertex_mean_curvature = (
                trimesh.curvature.discrete_mean_curvature_measure(
                    self.mesh, self.mesh.vertices, radius=5.0
                )
            )
            self._vertex_gaussian_curvature = (
                trimesh.curvature.discrete_gaussian_curvature_measure(
                    self.mesh, self.mesh.vertices, radius=5.0
                )
            )

        H = float(np.mean(self._vertex_mean_curvature[vertex_indices]))
        K = float(np.mean(self._vertex_gaussian_curvature[vertex_indices]))

        # Principal curvatures from H and K:
        # k1, k2 are roots of: t² - 2Ht + K = 0
        # k1 = H + sqrt(H² - K), k2 = H - sqrt(H² - K)
        discriminant = H * H - K
        if discriminant < 0:
            # Numerical noise — clamp to zero
            discriminant = 0.0

        sqrt_disc = np.sqrt(discriminant)
        k1 = H + sqrt_disc
        k2 = H - sqrt_disc

        # Convention: |k1| >= |k2|
        if abs(k1) < abs(k2):
            k1, k2 = k2, k1

        return {"k1": float(k1), "k2": float(k2)}

    def _estimate_curvature_mean_gaus(self):
        rot = R.from_euler("xyz", self.agent_rot, degrees=True)
        ray_direction = rot.apply([0, 0, -1])
        locations, index_ray, index_tri = self.mesh.ray.intersects_location(
            ray_origins=[self.agent_pos],
            ray_directions=[ray_direction],
        )
        if len(locations) == 0:
            return [0.0, 0.0]

        distances = np.linalg.norm(locations - self.agent_pos, axis=1)
        nearest_idx = np.argmin(distances)
        face_idx = index_tri[nearest_idx]
        vertex_indices = self.mesh.faces[face_idx]

        if not hasattr(self, "_vertex_mean_curvature"):
            self._vertex_mean_curvature = trimesh.curvature.discrete_mean_curvature_measure(
                self.mesh, self.mesh.vertices, radius=5.0
            )
            self._vertex_gaussian_curvature = trimesh.curvature.discrete_gaussian_curvature_measure(
                self.mesh, self.mesh.vertices, radius=5.0
            )

        mean_curv = float(np.mean(self._vertex_mean_curvature[vertex_indices]))
        gauss_curv = float(np.mean(self._vertex_gaussian_curvature[vertex_indices]))

        return [mean_curv, gauss_curv]

    def _look_at_direction(self, direction):
        """Return euler angles [rx, ry, rz] IN DEGREES such that:
        R.from_euler("xyz", euler, degrees=True).apply([0, 0, -1]) ≈ direction
        Convention: forward is -Z when euler = [0,0,0].
        """
        d = np.asarray(direction, dtype=float)
        d /= (np.linalg.norm(d) + 1e-12)

        forward = np.array([0.0, 0.0, -1.0])

        # align_vectors: find R such that R @ forward ≈ d
        # The first argument is the target vectors, the second is the source vectors
        rot, _ = R.align_vectors([d], [forward])

        return rot.as_euler("xyz", degrees=True)

    def _look_at_direction_complex(self, direction):
        """Calculates euler angles for looking in a direction.
        Builds a full rotation matrix so that -Z (forward) looks in
        the direction, and Y (up) remains as vertical as possible.
        Resistant to gimbal lock (vertical normals).
        """
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        # forward = -Z, so we need -Z → direction, i.e. Z → -direction
        forward = -direction
        # We select the up vector, avoiding parallelism with the forward vector.
        world_up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(forward, world_up)) > 0.99:
            # The normal is almost vertical - we take X as a fallback up
            world_up = np.array([1.0, 0.0, 0.0])
        right = np.cross(world_up, forward)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(forward, right)
        # Rotation matrix: columns = right, up, forward (= -Z direction)
        rot_matrix = np.column_stack([right, up, forward])
        return R.from_matrix(rot_matrix).as_euler("xyz")

    def _move_tangentially(self, direction_degrees, step_size, snap_to_surface: bool = True):
        rot = R.from_euler("xyz", self.agent_rot, degrees=True)

        sensor_data = self.get_sensor_data()
        if sensor_data["point_normal"] is None:
            angle_rad = np.radians(direction_degrees)
            local_dir = np.array([np.sin(angle_rad), 0.0, -np.cos(angle_rad)], dtype=float)
            local_dir /= (np.linalg.norm(local_dir) + 1e-12)
            world_dir = rot.apply(local_dir)
            self.agent_pos += world_dir * step_size
            return

        n = np.array(sensor_data["point_normal"], dtype=float)
        n /= (np.linalg.norm(n) + 1e-12)

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

        a = np.radians(direction_degrees)
        world_dir = np.cos(a) * t1 + np.sin(a) * t2
        world_dir /= (np.linalg.norm(world_dir) + 1e-12)

        old_pos = self.agent_pos.copy()
        self.agent_pos += world_dir * step_size

        # Check if current normal is near-horizontal (rim/top/bottom).
        is_from_horizontal = (
            abs(float(np.dot(n, self.up_direction))) > 0.85
        )

        # ═══ Cache wrong-side wall info ═══
        # When agent is on wrong side (same_side=False) and on
        # a wall (not horizontal), remember whether wall faces
        # outward or inward. Used later for rim transitions.
        same_side = sensor_data.get("same_side", True)
        if not same_side and not is_from_horizontal:
            height_ax = self.height_axis
            n_h = n.copy()
            n_h[height_ax] = 0.0
            from_center = old_pos - self._mesh_centroid
            from_center[height_ax] = 0.0
            n_h_len = np.linalg.norm(n_h)
            fc_len = np.linalg.norm(from_center)
            if n_h_len > 1e-8 and fc_len > 1e-8:
                self._wrong_side_outward = bool(
                    np.dot(n_h, from_center) > 0
                )

        if snap_to_surface:
            closest, dist_to_mesh, face_id = self.mesh.nearest.on_surface(
                [self.agent_pos]
            )
            hit_n = self.mesh.face_normals[face_id[0]]
            hit_n = hit_n / (np.linalg.norm(hit_n) + 1e-12)

            # Standard normal alignment
            if np.dot(hit_n, n) < 0:
                if is_from_horizontal:
                    pass  # trust mesh normal for rim transitions
                else:
                    hit_n = -hit_n

            # ═══ Rim-to-wall side correction ═══
            new_is_horizontal = (
                abs(float(np.dot(hit_n, self.up_direction))) > 0.85
            )

            if (
                is_from_horizontal
                and not new_is_horizontal
                and hasattr(self, '_wrong_side_outward')
            ):
                height_ax = self.height_axis

                # Check if new wall is same type as old wrong-side wall
                hit_n_h = hit_n.copy()
                hit_n_h[height_ax] = 0.0
                from_center_new = closest[0] - self._mesh_centroid
                from_center_new[height_ax] = 0.0
                hit_h_len = np.linalg.norm(hit_n_h)
                fc_new_len = np.linalg.norm(from_center_new)

                if hit_h_len > 1e-8 and fc_new_len > 1e-8:
                    new_outward = bool(
                        np.dot(hit_n_h, from_center_new) > 0
                    )

                    if new_outward == self._wrong_side_outward:
                        # Same type of wall as before rim — wrong side.
                        # Search for opposite wall by going through.
                        opposite_pos = closest[0] - hit_n * 5.0
                        closest_opp, _, face_id_opp = (
                            self.mesh.nearest.on_surface([opposite_pos])
                        )
                        hit_n_opp = self.mesh.face_normals[face_id_opp[0]]
                        hit_n_opp = hit_n_opp / (
                            np.linalg.norm(hit_n_opp) + 1e-12
                        )

                        opp_is_horizontal = (
                            abs(
                                float(
                                    np.dot(
                                        hit_n_opp,
                                        self.up_direction,
                                    )
                                )
                            )
                            > 0.85
                        )

                        if not opp_is_horizontal:
                            # Check opposite wall is actually opposite
                            hit_n_opp_h = hit_n_opp.copy()
                            hit_n_opp_h[height_ax] = 0.0
                            from_center_opp = (
                                closest_opp[0] - self._mesh_centroid
                            )
                            from_center_opp[height_ax] = 0.0
                            opp_h_len = np.linalg.norm(hit_n_opp_h)
                            fc_opp_len = np.linalg.norm(from_center_opp)

                            if opp_h_len > 1e-8 and fc_opp_len > 1e-8:
                                opp_outward = bool(
                                    np.dot(hit_n_opp_h, from_center_opp)
                                    > 0
                                )

                                if opp_outward != self._wrong_side_outward:
                                    # Opposite wall found — use it
                                    hit_n = hit_n_opp
                                    closest = closest_opp
                                    self._wrong_side_outward = None

            # For rim transitions, allow larger normal change
            if is_from_horizontal and not new_is_horizontal:
                can_transition = True
            else:
                can_transition = np.dot(hit_n, n) > -0.1

            if can_transition:
                self.agent_pos = closest[0] + hit_n * 2.0
                self.agent_rot = self._look_at_direction(-hit_n)
            else:
                # ═══ Edge traversal via intermediate steps ═══
                edge_traversed = False

                ray_blocked = False
                locations, _, _ = self.mesh.ray.intersects_location(
                    ray_origins=[old_pos],
                    ray_directions=[world_dir],
                )
                if len(locations) > 0:
                    hit_dist = np.min(
                        np.linalg.norm(locations - old_pos, axis=1)
                    )
                    if hit_dist < step_size * 0.5:
                        ray_blocked = True

                if not ray_blocked:
                    half_pos = old_pos + world_dir * step_size * 0.5
                    closest_half, dist_half, face_half = (
                        self.mesh.nearest.on_surface([half_pos])
                    )
                    hit_n_half = self.mesh.face_normals[face_half[0]]
                    hit_n_half = hit_n_half / (
                        np.linalg.norm(hit_n_half) + 1e-12
                    )

                    if np.dot(hit_n_half, n) < 0:
                        if is_from_horizontal:
                            pass  # trust mesh normal
                        else:
                            hit_n_half = -hit_n_half

                    can_half = np.dot(hit_n_half, n) > -0.1

                    if can_half and dist_half[0] < step_size * 2.0:
                        intermediate_pos = (
                            closest_half[0] + hit_n_half * 2.0
                        )
                        full_pos = (
                            intermediate_pos
                            + world_dir * step_size * 0.5
                        )
                        closest_full, dist_full, face_full = (
                            self.mesh.nearest.on_surface([full_pos])
                        )
                        hit_n_full = self.mesh.face_normals[face_full[0]]
                        hit_n_full = hit_n_full / (
                            np.linalg.norm(hit_n_full) + 1e-12
                        )

                        if np.dot(hit_n_full, hit_n_half) < 0:
                            hit_n_full = -hit_n_full

                        can_full = (
                            np.dot(hit_n_full, hit_n_half) > -0.1
                        )

                        if (
                            can_full
                            and dist_full[0] < step_size * 2.0
                        ):
                            self.agent_pos = (
                                closest_full[0] + hit_n_full * 2.0
                            )
                            self.agent_rot = (
                                self._look_at_direction(-hit_n_full)
                            )
                            edge_traversed = True
                            self._edge_traversed = True

                if not edge_traversed:
                    self.agent_pos = old_pos
                    
    def _move_tangentially_old(self, direction_degrees, step_size, snap_to_surface: bool = True):
        rot = R.from_euler("xyz", self.agent_rot, degrees=True)

        sensor_data = self.get_sensor_data()
        if sensor_data["point_normal"] is None:
            angle_rad = np.radians(direction_degrees)
            local_dir = np.array([np.sin(angle_rad), 0.0, -np.cos(angle_rad)], dtype=float)
            local_dir /= (np.linalg.norm(local_dir) + 1e-12)
            world_dir = rot.apply(local_dir)
            self.agent_pos += world_dir * step_size
            return

        n = np.array(sensor_data["point_normal"], dtype=float)
        n /= (np.linalg.norm(n) + 1e-12)

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

        a = np.radians(direction_degrees)
        world_dir = np.cos(a) * t1 + np.sin(a) * t2
        world_dir /= (np.linalg.norm(world_dir) + 1e-12)

        old_pos = self.agent_pos.copy()
        self.agent_pos += world_dir * step_size

        if snap_to_surface:
            closest, dist_to_mesh, face_id = self.mesh.nearest.on_surface([self.agent_pos])
            hit_n = self.mesh.face_normals[face_id[0]]
            hit_n = hit_n / (np.linalg.norm(hit_n) + 1e-12)
            if np.dot(hit_n, n) < 0:
                hit_n = -hit_n

            can_transition = np.dot(hit_n, n) > -0.1

            if can_transition:
                self.agent_pos = closest[0] + hit_n * 2.0
                self.agent_rot = self._look_at_direction(-hit_n)
            else:
                # ═══ Edge traversal via intermediate steps ═══
                # When normal flips >95° in one step (e.g. mug rim),
                # try two half-steps to traverse edge gradually:
                # step1: old → edge (normal turns ~90°)
                # step2: edge → other side (normal turns another ~90°)
                edge_traversed = False

                # Safety: ray cast to check we won't pass through mesh
                ray_blocked = False
                locations, _, _ = self.mesh.ray.intersects_location(
                    ray_origins=[old_pos],
                    ray_directions=[world_dir],
                )
                if len(locations) > 0:
                    hit_dist = np.min(
                        np.linalg.norm(locations - old_pos, axis=1)
                    )
                    if hit_dist < step_size * 0.5:
                        ray_blocked = True

                if not ray_blocked:
                    # Try half-step
                    half_pos = old_pos + world_dir * step_size * 0.5
                    closest_half, dist_half, face_half = (
                        self.mesh.nearest.on_surface([half_pos])
                    )
                    hit_n_half = self.mesh.face_normals[face_half[0]]
                    hit_n_half = hit_n_half / (
                        np.linalg.norm(hit_n_half) + 1e-12
                    )
                    if np.dot(hit_n_half, n) < 0:
                        hit_n_half = -hit_n_half

                    can_half = np.dot(hit_n_half, n) > -0.1

                    if can_half and dist_half[0] < step_size * 2.0:
                        # Half-step landed on compatible surface
                        # Now try second half from intermediate position
                        intermediate_pos = (
                            closest_half[0] + hit_n_half * 2.0
                        )
                        full_pos = (
                            intermediate_pos
                            + world_dir * step_size * 0.5
                        )
                        closest_full, dist_full, face_full = (
                            self.mesh.nearest.on_surface([full_pos])
                        )
                        hit_n_full = self.mesh.face_normals[face_full[0]]
                        hit_n_full = hit_n_full / (
                            np.linalg.norm(hit_n_full) + 1e-12
                        )
                        if np.dot(hit_n_full, hit_n_half) < 0:
                            hit_n_full = -hit_n_full

                        can_full = (
                            np.dot(hit_n_full, hit_n_half) > -0.1
                        )

                        if (
                            can_full
                            and dist_full[0] < step_size * 2.0
                        ):
                            self.agent_pos = (
                                closest_full[0] + hit_n_full * 2.0
                            )
                            self.agent_rot = (
                                self._look_at_direction(-hit_n_full)
                            )
                            edge_traversed = True
                            self._edge_traversed = True

                if not edge_traversed:
                    self.agent_pos = old_pos

    def _move_forward(self, step_size):
        rot = R.from_euler("xyz", self.agent_rot, degrees=True)
        forward = rot.apply([0, 0, -1])
        old_pos = self.agent_pos.copy()
        self.agent_pos += forward * step_size
        self._passed_through = False

        if abs(step_size) > 0.5:
            locations, _, _ = self.mesh.ray.intersects_location(
                ray_origins=[old_pos],
                ray_directions=[forward * np.sign(step_size)],
            )
            if len(locations) > 0:
                distances = np.linalg.norm(locations - old_pos, axis=1)
                if np.min(distances) < abs(step_size):
                    self._passed_through = True

            closest, dist_to_mesh, _ = self.mesh.nearest.on_surface([self.agent_pos])
            # Adaptive proximity threshold based on step size
            proximity_threshold = min(1.0, abs(step_size) * 0.25)
            if dist_to_mesh[0] < proximity_threshold:
                self._passed_through = True
                
    def _orient_horizontal(self, rotation_degrees, forward_distance, left_distance):
        """Moves the agent forward and sideways (left/right), projects onto surface, and yaws.
        
        Args:
            rotation_degrees: additional yaw rotation (e.g. +45 for right turn)
            forward_distance: mm to move forward (+) or backward (-)
            left_distance: mm to move left (+) or right (-)
        """
        # Local movement direction
        local_dir = np.array([
            -left_distance,        # +left → -X (left), -left → +X (right)
            0.0,
            -forward_distance      # +forward → -Z, -forward → +Z
        ])

        if np.linalg.norm(local_dir) < 1e-8:
            local_dir = np.array([0.0, 0.0, -1.0])  # default: forward
        else:
            local_dir = local_dir / np.linalg.norm(local_dir)

        # Rotate to world
        rot = R.from_euler("xyz", self.agent_rot, degrees=True)
        world_dir = rot.apply(local_dir)

        # Project onto tangent plane
        sensor_data = self.get_sensor_data()
        if sensor_data["point_normal"] is not None:
            normal = np.array(sensor_data["point_normal"])
            world_dir = world_dir - np.dot(world_dir, normal) * normal
            norm = np.linalg.norm(world_dir)
            if norm > 1e-8:
                world_dir /= norm

        # Total step distance in mm
        step = np.linalg.norm([forward_distance, left_distance])
        self.agent_pos += world_dir * step

        # Apply yaw rotation
        self.agent_rot[1] += rotation_degrees  # Y-axis = yaw

    def _orient_vertical(self, rotation_degrees, forward_distance, down_distance):
        """Moves the agent forward and down/up, projects onto surface, and pitches.
        
        Args:
            rotation_degrees: additional pitch rotation (e.g. +10 for look down)
            forward_distance: mm to move forward (+) or backward (-)
            down_distance: mm to move down (+) or up (-)
        """
        # Local movement: X=0, Y=-down, Z=-forward
        local_dir = np.array([
            0.0,
            -down_distance,        # +down → -Y, -down → +Y (up)
            -forward_distance      # +forward → -Z
        ])

        if np.linalg.norm(local_dir) < 1e-8:
            local_dir = np.array([0.0, 0.0, -1.0])  # default: forward
        else:
            local_dir = local_dir / np.linalg.norm(local_dir)

        # Rotate to world
        rot = R.from_euler("xyz", self.agent_rot, degrees=True)
        world_dir = rot.apply(local_dir)

        # Project onto tangent plane
        sensor_data = self.get_sensor_data()
        if sensor_data["point_normal"] is not None:
            normal = np.array(sensor_data["point_normal"])
            world_dir = world_dir - np.dot(world_dir, normal) * normal
            norm = np.linalg.norm(world_dir)
            if norm > 1e-8:
                world_dir /= norm

        # Total step distance
        step = np.linalg.norm([forward_distance, down_distance])
        self.agent_pos += world_dir * step

        # Apply pitch rotation (around X axis)
        self.agent_rot[0] += rotation_degrees  # X-axis = pitch

    def _compute_up_direction(self):
        bbox_min = self.mesh.bounds[0]
        bbox_max = self.mesh.bounds[1]
        extents = bbox_max - bbox_min

        # use centroid - it is resistant to handles and other protrusions
        center = np.array(self.mesh.centroid, dtype=float)

        n_probes = 25
        best_axis = 0
        best_asymmetry = -1
        best_up_sign = 1.0
        axis_info = {}

        for axis in range(3):
            horiz_axes = [i for i in range(3) if i != axis]

            probe_points = []
            for i in range(5):
                for j in range(5):
                    pt = center.copy()
                    r0 = min(extents[horiz_axes[0]], extents[horiz_axes[1]]) * 0.3
                    pt[horiz_axes[0]] = center[horiz_axes[0]] + r0 * (i / 4 - 0.5)
                    pt[horiz_axes[1]] = center[horiz_axes[1]] + r0 * (j / 4 - 0.5)
                    probe_points.append(pt)
            probe_points = np.array(probe_points)

            # Rays in the + direction
            dir_plus = np.zeros(3)
            dir_plus[axis] = 1.0
            origins_plus = probe_points.copy()
            origins_plus[:, axis] = center[axis] + 1.0
            hits_plus, _, _ = self.mesh.ray.intersects_location(
                ray_origins=origins_plus,
                ray_directions=np.tile(dir_plus, (n_probes, 1)),
            )

            # Rays in the - direction
            dir_minus = np.zeros(3)
            dir_minus[axis] = -1.0
            origins_minus = probe_points.copy()
            origins_minus[:, axis] = center[axis] - 1.0
            hits_minus, _, _ = self.mesh.ray.intersects_location(
                ray_origins=origins_minus,
                ray_directions=np.tile(dir_minus, (n_probes, 1)),
            )

            n_plus = len(hits_plus)
            n_minus = len(hits_minus)
            asymmetry = abs(n_plus - n_minus)

            axis_info[axis] = {
                "plus": n_plus,
                "minus": n_minus,
                "asymmetry": asymmetry,
            }

            if asymmetry > best_asymmetry:
                best_asymmetry = asymmetry
                best_axis = axis
                if n_plus < n_minus:
                    best_up_sign = 1.0
                elif n_minus < n_plus:
                    best_up_sign = -1.0
                else:
                    best_up_sign = 1.0

        self.height_axis = best_axis
        self.up_sign = best_up_sign
        self.up_direction = np.zeros(3)
        self.up_direction[self.height_axis] = self.up_sign
        self.open_edge_height = (
            bbox_max[self.height_axis] if self.up_sign > 0
            else bbox_min[self.height_axis]
        )

        logger.debug(
            f"_compute_up_direction: "
            f"extents={extents.tolist()}, "
            f"centroid={center.tolist()}, "
            f"axis_info={axis_info}, "
            f"height_axis={self.height_axis}, "
            f"up_sign={self.up_sign}, "
            f"up_direction={self.up_direction.tolist()}, "
            f"open_edge_height={self.open_edge_height}"
        )

    def _detach_simple(self, goal_pose, detach_distance=8.0):
        """Detach from surface along normal, then look toward goal.

        Simple macro action:
        1. Move along surface normal by detach_distance
        2. Orient gaze toward goal position

        After detach, Q-store/SAC decide what to do next
        (turn, fly forward, detach again, etc.)

        Args:
            goal_pose: Target pose [x, y, z, rx, ry, rz].
            detach_distance: How far to fly from surface (mm).
        """
        self._detach_had_collision = False
        self._last_detach_sub_steps = 1

        sensor = self.get_sensor_data()
        if sensor.get("point_normal") is None:
            logger.debug("DETACH_SIMPLE_ABORT: no point_normal")
            return self.get_sensor_data()

        normal = np.array(sensor["point_normal"], dtype=float)
        normal /= (np.linalg.norm(normal) + 1e-12)

        # Step 1: Move along normal (away from surface)
        old_pos = self.agent_pos.copy()
        self.agent_pos += normal * detach_distance

        # Check for collision during detach
        locations, _, _ = self.mesh.ray.intersects_location(
            ray_origins=[old_pos],
            ray_directions=[normal],
        )
        if len(locations) > 0:
            distances = np.linalg.norm(locations - old_pos, axis=1)
            if np.min(distances) < detach_distance:
                self.agent_pos = old_pos
                self._passed_through = False
                self._detach_had_collision = True
                logger.debug("DETACH_SIMPLE_COLLISION: hit object during detach")
                return self.get_sensor_data()

        # Step 2: Orient gaze toward goal
        # Adapt fly direction based on whether goal is behind surface
        goal_pos = goal_pose[:3]
        goal_dir = goal_pos - self.agent_pos
        goal_dist = np.linalg.norm(goal_dir)
        if goal_dist > 1e-8:
            goal_dir /= goal_dist
            dot_goal_normal = float(np.dot(goal_dir, normal))

            if dot_goal_normal < -0.2:
                # Goal is behind surface — fly sideways to go around
                # Project goal_dir onto tangent plane
                tangent_to_goal = goal_dir - dot_goal_normal * normal
                tangent_len = float(np.linalg.norm(tangent_to_goal))

                if tangent_len > 1e-8:
                    tangent_to_goal /= tangent_len
                    # Fly: away from surface + sideways toward goal
                    fly_direction = normal * 0.7 + tangent_to_goal * 0.7
                else:
                    # Goal exactly behind normal — fly away from surface
                    fly_direction = normal

                fly_direction /= (np.linalg.norm(fly_direction) + 1e-12)
            else:
                # Goal is in front of or beside surface — standard blend
                fly_direction = goal_dir + normal * 0.3
                fly_direction /= (np.linalg.norm(fly_direction) + 1e-12)

            self.agent_rot = self._look_at_direction(fly_direction)
            
        logger.debug(
            f"DETACH_SIMPLE_DONE: "
            f"old_pos={old_pos.tolist()}, "
            f"new_pos={self.agent_pos.tolist()}, "
            f"normal={normal.tolist()}, "
            f"goal_dir={goal_dir.tolist()}, "
            f"dist_to_goal={goal_dist:.1f}"
        )

        return self.get_sensor_data()

    def _detach_and_fly_to_edge(self, goal_pose, rotation_step=5.0, free_step=8.0, max_sub_steps=3):
        detach_fly_step = 8.0
        safe_step = 3.0
        sub_steps = 0
        detach_step = 8.0

        self._detach_had_collision = False

        logger.debug(
            f"DETACH_EDGE_START: "
            f"agent_pos={self.agent_pos.tolist()}, "
            f"agent_rot={self.agent_rot.tolist()}, "
            f"goal_pos={goal_pose[:3].tolist()}, "
            f"height_axis={self.height_axis}, "
            f"up_direction={self.up_direction.tolist()}, "
            f"open_edge_height={self.open_edge_height:.1f}, "
            f"bbox={self.mesh.bounds.tolist()}"
        )

        sensor = self.get_sensor_data()
        if sensor.get("point_normal") is None:
            logger.debug("DETACH_EDGE_ABORT: no point_normal")
            self._last_detach_sub_steps = sub_steps
            return self.get_sensor_data()

        normal = np.array(sensor["point_normal"], dtype=float)
        normal /= (np.linalg.norm(normal) + 1e-12)

        # ═══ Phase 1: Normal Breakaway ═══
        self.agent_rot = self._look_at_direction(normal)

        logger.debug(
            f"DETACH_EDGE_PHASE1_START: "
            f"normal={normal.tolist()}, "
            f"agent_rot={self.agent_rot.tolist()}, "
            f"detach_step={detach_step}"
        )

        flown = 0.0
        while flown < detach_step:
            old_pos = self.agent_pos.copy()
            step = min(safe_step, detach_step - flown)
            sensor_check = self.get_sensor_data()
            depth = sensor_check.get("depth", 100.0)
            if depth < step + 1.0:
                step = max(depth - 1.0, 0.5)
            self._move_forward(step)
            flown += step
            sub_steps += 1
            if self._passed_through:
                logger.debug(
                    f"DETACH_EDGE_COLLISION: "
                    f"phase=1, sub_step={sub_steps}, "
                    f"pos_before={old_pos.tolist()}, "
                    f"pos_after={self.agent_pos.tolist()}, "
                    f"normal={normal.tolist()}, "
                    f"step={step:.1f}, "
                    f"depth={depth:.1f}, "
                    f"flown={flown:.1f}"
                )
                self.agent_pos = old_pos
                self._passed_through = False
                self._detach_had_collision = True
                self._last_detach_sub_steps = sub_steps
                return self.get_sensor_data()

        logger.debug(
            f"DETACH_EDGE_PHASE1_DONE: "
            f"pos={self.agent_pos.tolist()}, "
            f"flown={flown:.1f}, "
            f"sub_steps={sub_steps}"
        )

        # ═══ Phase 2: Fly to the open edge ═══
        up_dir = self.up_direction
        height_axis = self.height_axis
        open_edge_height = self.open_edge_height

        center = np.array(self.mesh.centroid, dtype=float)

        # Inside or outside?
        agent_to_center = center - self.agent_pos
        agent_to_center[height_axis] = 0.0
        dot_normal_to_center = np.dot(normal, agent_to_center)
        is_inside = dot_normal_to_center > 0

        if is_inside:
            to_center_horiz = agent_to_center.copy()
            to_center_len = np.linalg.norm(to_center_horiz)
            if to_center_len > 1e-8:
                to_center_horiz /= to_center_len
            fly_dir = up_dir * 1.0 + to_center_horiz * 0.2
            overshoot = detach_fly_step
        else:
            fly_dir = normal * 1.0 + up_dir * 0.5
            overshoot = detach_fly_step

        fly_dir /= (np.linalg.norm(fly_dir) + 1e-12)
        self.agent_rot = self._look_at_direction(fly_dir)

        target_height = open_edge_height + overshoot * self.up_sign

        logger.debug(
            f"DETACH_EDGE_PHASE2_START: "
            f"is_inside={is_inside}, "
            f"normal={normal.tolist()}, "
            f"center={center.tolist()}, "
            f"agent_to_center_horiz={agent_to_center.tolist()}, "
            f"dot_normal_to_center={dot_normal_to_center:.3f}, "
            f"fly_dir={fly_dir.tolist()}, "
            f"up_dir={up_dir.tolist()}, "
            f"target_height={target_height:.1f}, "
            f"current_height={self.agent_pos[height_axis]:.1f}, "
            f"open_edge_height={open_edge_height:.1f}, "
            f"agent_rot={self.agent_rot.tolist()}"
        )

        max_fly_steps = 20
        for fly_step_idx in range(max_fly_steps):
            sub_steps += 1
            old_pos = self.agent_pos.copy()

            sensor_check = self.get_sensor_data()
            depth = sensor_check.get("depth", 100.0)
            step = detach_fly_step
            if depth < step + 1.0:
                step = max(depth - 1.0, 0.5)

            self._move_forward(step)

            if self._passed_through:
                logger.debug(
                    f"DETACH_EDGE_COLLISION: "
                    f"phase=2, fly_step_idx={fly_step_idx}, sub_step={sub_steps}, "
                    f"pos_before={old_pos.tolist()}, "
                    f"pos_after={self.agent_pos.tolist()}, "
                    f"fly_dir={fly_dir.tolist()}, "
                    f"step={step:.1f}, "
                    f"depth={depth:.1f}, "
                    f"current_height={old_pos[height_axis]:.1f}, "
                    f"target_height={target_height:.1f}"
                )
                self.agent_pos = old_pos
                self._passed_through = False
                self._detach_had_collision = True
                break

            if self.up_sign > 0:
                if self.agent_pos[height_axis] > target_height:
                    logger.debug(
                        f"DETACH_EDGE_PHASE2_REACHED: "
                        f"fly_step_idx={fly_step_idx}, "
                        f"height={self.agent_pos[height_axis]:.1f}, "
                        f"target={target_height:.1f}"
                    )
                    break
            elif self.agent_pos[height_axis] < target_height:
                logger.debug(
                    f"DETACH_EDGE_PHASE2_REACHED: "
                    f"fly_step_idx={fly_step_idx}, "
                    f"height={self.agent_pos[height_axis]:.1f}, "
                    f"target={target_height:.1f}"
                )
                break

        logger.debug(
            f"DETACH_EDGE_PHASE2_DONE: "
            f"pos={self.agent_pos.tolist()}, "
            f"height={self.agent_pos[height_axis]:.1f}, "
            f"had_collision={self._detach_had_collision}, "
            f"sub_steps={sub_steps}"
        )

        # If there was a collision in phase 2, do not continue phase 3
        if self._detach_had_collision:
            self._last_detach_sub_steps = sub_steps
            return self.get_sensor_data()

        # ═══ Phase 3: Horizontal flight towards the target ═══
        goal_pos = goal_pose[:3]

        direction_to_goal = goal_pos - self.agent_pos
        direction_horiz = direction_to_goal.copy()
        direction_horiz[height_axis] = 0.0

        horiz_dist = np.linalg.norm(direction_horiz)
        if horiz_dist > 1e-8:
            direction_horiz /= horiz_dist
        else:
            direction_horiz = -normal
            direction_horiz[height_axis] = 0.0
            direction_horiz /= (np.linalg.norm(direction_horiz) + 1e-12)

        self.agent_rot = self._look_at_direction(direction_horiz)

        total_fly = int(detach_fly_step * max_sub_steps * 1.5)

        logger.debug(
            f"DETACH_EDGE_PHASE3_START: "
            f"pos={self.agent_pos.tolist()}, "
            f"goal_pos={goal_pos.tolist()}, "
            f"direction_horiz={direction_horiz.tolist()}, "
            f"horiz_dist_to_goal={horiz_dist:.1f}, "
            f"total_fly={total_fly:.1f}, "
            f"agent_rot={self.agent_rot.tolist()}"
        )

        flown = 0.0
        phase3_step_idx = 0
        while flown < total_fly:
            old_pos = self.agent_pos.copy()
            step = min(detach_fly_step, total_fly - flown)
            self._move_forward(step)
            flown += step
            sub_steps += 1
            phase3_step_idx += 1

            if self._passed_through:
                logger.debug(
                    f"DETACH_EDGE_COLLISION: "
                    f"phase=3, phase3_step_idx={phase3_step_idx}, sub_step={sub_steps}, "
                    f"pos_before={old_pos.tolist()}, "
                    f"pos_after={self.agent_pos.tolist()}, "
                    f"direction_horiz={direction_horiz.tolist()}, "
                    f"step={step:.1f}, "
                    f"flown={flown:.1f}, "
                    f"total_fly={total_fly:.1f}"
                )
                self.agent_pos = old_pos
                self._passed_through = False
                self._detach_had_collision = True
                break

        logger.debug(
            f"DETACH_EDGE_DONE: "
            f"sub_steps={sub_steps}, "
            f"had_collision={self._detach_had_collision}, "
            f"final_pos={self.agent_pos.tolist()}, "
            f"final_height={self.agent_pos[height_axis]:.1f}, "
            f"goal_pos={goal_pose[:3].tolist()}, "
            f"final_dist_to_goal={np.linalg.norm(self.agent_pos - goal_pose[:3]):.1f}"
        )

        self._last_detach_sub_steps = sub_steps

        # direct agent gaze downwards
        down_direction = -self.up_direction
        self.agent_rot = self._look_at_direction(down_direction)

        return self.get_sensor_data()

    def _detach_and_fly_to_goal(self, goal_pose, rotation_step=5.0, free_step=8.0, max_sub_steps=3):
        detach_fly_step = 8.0
        sub_steps = 1
        self._detach_had_collision = False

        sensor = self.get_sensor_data()
        normal = None
        if sensor.get("point_normal") is not None:
            normal = np.array(sensor["point_normal"], dtype=float)
            normal /= (np.linalg.norm(normal) + 1e-12)
            self.agent_rot = self._look_at_direction(normal)

        total_fly = detach_fly_step * max_sub_steps
        flown = 0.0
        while flown < total_fly:
            old_pos = self.agent_pos.copy()
            sensor = self.get_sensor_data()
            depth = sensor.get("depth", 100.0)
            step = min(detach_fly_step, total_fly - flown)
            if depth < step:
                step = max(depth - 2.0, 0.5)
            self._move_forward(step)
            flown += step

            if self._passed_through:
                self.agent_pos = old_pos
                self._passed_through = False
                self._detach_had_collision = True
                break

        goal_pos = goal_pose[:3]
        direction = goal_pos - self.agent_pos
        dist = np.linalg.norm(direction)
        if dist > 1e-8:
            direction /= dist

        if normal is not None:
            fly_direction = direction + normal * 0.5
            fly_direction /= (np.linalg.norm(fly_direction) + 1e-12)
        else:
            fly_direction = direction

        self.agent_rot = self._look_at_direction(fly_direction)

        for _ in range(max_sub_steps):
            sub_steps += 1
            old_pos = self.agent_pos.copy()

            sensor = self.get_sensor_data()
            depth = sensor.get("depth", 100.0)

            if depth < detach_fly_step:
                step = max(depth - 2.0, 0.5)
                self._move_forward(step)
            else:
                self._move_forward(detach_fly_step)

            if self._passed_through:
                self.agent_pos = old_pos
                self._passed_through = False
                self._detach_had_collision = True
                break

            sensor = self.get_sensor_data()
            if sensor.get("on_object", False):
                break

        self._last_detach_sub_steps = sub_steps
        return self.get_sensor_data()

def is_on_same_cube_side(pos_a, pos_b, cube_side=42.0, atol=1e-5):
    """Checks whether two points lie on the same side of a cube (e.g., both at +X when x = +42).

    Args:
        pos_a: position of point A [x, y, z]
        pos_b: position of point B [x, y, z]
        cube_side: cube side coordinate value (e.g., 42 mm)
        atol: numerical error tolerance

    Returns:
        bool: True if the points are on the same side of the cube
    """
    pos_a = np.array(pos_a)
    pos_b = np.array(pos_b)

    # check each axis
    for i in range(3):
        # Both points must be on the edge (coordinate ≈ ±cube_side)
        if (abs(abs(pos_a[i]) - cube_side) <= atol and
            abs(abs(pos_b[i]) - cube_side) <= atol):

            # And the sign must match (both +42 or both -42)
            if np.sign(pos_a[i]) == np.sign(pos_b[i]):
                return True

    return False


def _is_reachable_by_surface_old(
    env: LightweightEnv,
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
) -> bool:
    """Check if start and goal are on the same side of the object.

    Uses surface normal directions relative to object centroid.
    For vertical surfaces (walls): normal pointing outward from center
    means external side, inward means internal side.
    For horizontal surfaces (bottom/top): normal pointing against
    up_direction means external (bottom exterior), normal pointing
    with up_direction means internal (bottom interior).

    Args:
        env: Environment with mesh.
        start_pos: Start position [x, y, z].
        goal_pos: Goal position [x, y, z].

    Returns:
        True if both points are on the same side.
    """
    center = np.array(env.mesh.centroid, dtype=float)
    height_axis = env.height_axis
    up = env.up_direction

    # Get normals at nearest surface points
    _, _, start_face = env.mesh.nearest.on_surface([start_pos])
    _, _, goal_face = env.mesh.nearest.on_surface([goal_pos])

    start_normal = env.mesh.face_normals[start_face[0]]
    goal_normal = env.mesh.face_normals[goal_face[0]]

    # Horizontal components (ignore height axis)
    start_n = start_normal.copy()
    start_n[height_axis] = 0.0

    goal_n = goal_normal.copy()
    goal_n[height_axis] = 0.0

    # Direction from center to each point (horizontal)
    start_from_center = start_pos - center
    start_from_center[height_axis] = 0.0

    goal_from_center = goal_pos - center
    goal_from_center[height_axis] = 0.0

    # Determine side for each point
    if np.linalg.norm(start_n) >= 0.3:
        # Wall: outward = normal points same direction as center→point
        start_outward = np.dot(start_n, start_from_center) > 0
    else:
        # Horizontal surface (bottom/top): use vertical normal component
        # Normal against up = outside (bottom exterior)
        # Normal with up = inside (bottom interior)
        start_outward = np.dot(start_normal, up) < 0

    if np.linalg.norm(goal_n) >= 0.3:
        goal_outward = np.dot(goal_n, goal_from_center) > 0
    else:
        goal_outward = np.dot(goal_normal, up) < 0

    result = start_outward == goal_outward
    logger.info(
        f"REACHABLE_CHECK: "
        f"start={[round(x,1) for x in start_pos.tolist()]}, "
        f"goal={[round(x,1) for x in goal_pos.tolist()]}, "
        f"start_normal={[round(x,3) for x in start_normal.tolist()]}, "
        f"goal_normal={[round(x,3) for x in goal_normal.tolist()]}, "
        f"start_outward={start_outward}, "
        f"goal_outward={goal_outward}, "
        f"result={result}"
    )
    return result

def _is_reachable_by_surface(
    env: LightweightEnv,
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
) -> bool:
    """Check if start and goal are on the same side of the object.

    Uses surface normal directions relative to object centroid.
    For vertical surfaces (walls): normal pointing outward from center
    means external side, inward means internal side.
    For horizontal surfaces (bottom/top): normal pointing against
    up_direction means external (bottom exterior), normal pointing
    with up_direction means internal (bottom interior).
    For points far from surface: considered outside (nearest.on_surface
    is unreliable when far from object).

    Args:
        env: Environment with mesh.
        start_pos: Start position [x, y, z].
        goal_pos: Goal position [x, y, z].

    Returns:
        True if both points are on the same side.
    """
    center = np.array(env.mesh.centroid, dtype=float)
    height_axis = env.height_axis
    up = env.up_direction

    # Max wall thickness — points farther than this from surface
    # are reliably outside the object
    FAR_THRESHOLD = 10.0

    # Get normals at nearest surface points
    _, start_dist, start_face = env.mesh.nearest.on_surface([start_pos])
    _, goal_dist, goal_face = env.mesh.nearest.on_surface([goal_pos])

    start_normal = env.mesh.face_normals[start_face[0]]
    goal_normal = env.mesh.face_normals[goal_face[0]]

    # Horizontal components (ignore height axis)
    start_n = start_normal.copy()
    start_n[height_axis] = 0.0

    goal_n = goal_normal.copy()
    goal_n[height_axis] = 0.0

    # Direction from center to each point (horizontal)
    start_from_center = start_pos - center
    start_from_center[height_axis] = 0.0

    goal_from_center = goal_pos - center
    goal_from_center[height_axis] = 0.0

    # Determine side for start point
    if start_dist[0] > FAR_THRESHOLD:
        # Far from surface — reliably outside
        start_outward = True
    elif np.linalg.norm(start_n) >= 0.3:
        # Wall: outward = normal points same direction as center→point
        start_outward = np.dot(start_n, start_from_center) > 0
    else:
        # Horizontal surface (bottom/top): use vertical normal component
        start_outward = np.dot(start_normal, up) < 0

    # Determine side for goal point
    if goal_dist[0] > FAR_THRESHOLD:
        goal_outward = True
    elif np.linalg.norm(goal_n) >= 0.3:
        goal_outward = np.dot(goal_n, goal_from_center) > 0
    else:
        goal_outward = np.dot(goal_normal, up) < 0

    result = start_outward == goal_outward
    logger.debug(
        f"REACHABLE_CHECK: "
        f"start={[round(x,1) for x in start_pos.tolist()]}, "
        f"goal={[round(x,1) for x in goal_pos.tolist()]}, "
        f"start_normal={[round(x,3) for x in start_normal.tolist()]}, "
        f"goal_normal={[round(x,3) for x in goal_normal.tolist()]}, "
        f"start_outward={start_outward}, "
        f"goal_outward={goal_outward}, "
        f"result={result}"
    )
    return result
