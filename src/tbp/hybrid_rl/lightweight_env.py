# Новый файл: lightweight_env.py
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R
import logging

logger = logging.getLogger(__name__)


class LightweightEnv:
    """
    A lightweight simulator for training/testing RL without Habitat.
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
        #self.agent_rot = [X-angle, Y-angle, Z-angle]

        # Note: seed is set globally in train() => np.random.seed()
        # trimesh.sample() uses global np.random
    
    def reset(self, position=None, rotation=None):
        """Place the agent in the starting position."""
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

        return self.get_sensor_data()    
    
    def step(self, action_index, action_space):
        """
        Perform an action, return new sensor data.
        
        Args:
            action_index: action index (0-17)
            action_space: MontyActionSpace
            
        Returns:
            sensor_data: point_normal, on_object, depth
        """

        action_info = action_space.get_info(action_index)
        
        if action_info.name == "move_tangentially":
            self._move_tangentially(
                action_info.direction_degrees,
                action_space.surface_step,
            )
        elif action_info.name == "free_forward":
            self._move_forward(action_space.free_step)
        elif action_info.name == "free_backward":
            self._move_forward(-action_space.free_step)
        elif action_info.name == "look_up":
            self.agent_rot[0] += action_space.rotation_step
        elif action_info.name == "look_down":
            self.agent_rot[0] -= action_space.rotation_step
        elif action_info.name == "rotate_sensor_+":
            self.agent_rot[2] += action_space.rotation_step
        elif action_info.name == "rotate_sensor_-":
            self.agent_rot[2] -= action_space.rotation_step
        elif action_info.name == "turn_left":
            self.agent_rot[1] += action_space.rotation_step
        elif action_info.name == "turn_right":
            self.agent_rot[1] -= action_space.rotation_step
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
        
        return self.get_sensor_data()
    
    def get_sensor_data(self):
        """Get sensory data from the current position."""
        # Ray casting
        rot = R.from_euler("xyz", self.agent_rot, degrees=True)
        ray_direction = rot.apply([0, 0, -1])
        # Intersect ray with mesh
        locations, index_ray, index_tri = self.mesh.ray.intersects_location(
            ray_origins=[self.agent_pos],
            ray_directions=[ray_direction],
        )
        
        if len(locations) > 0:
            # Nearest intersection
            distances = np.linalg.norm(locations - self.agent_pos, axis=1)
            nearest_idx = np.argmin(distances)
            
            depth = distances[nearest_idx]
            face_idx = index_tri[nearest_idx]
            point_normal = self.mesh.face_normals[face_idx].tolist()
            on_object = bool(depth < 3.0)  # было 10mm
        else:
            depth = 100.0
            point_normal = None
            on_object = False
        
        return {
            "point_normal": point_normal,
            "principal_curvatures": self._estimate_curvature(),
            "on_object": on_object,
            "depth": depth,
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
        """
        A random point on the surface with an optional distance limit.

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
                # points, face_ids = self.mesh.sample(1, return_index=True)
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

        if not hasattr(self, '_vertex_mean_curvature'):
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
        """
        Return euler angles [rx, ry, rz] IN DEGREES such that:
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
        """
        Calculates euler angles for looking in a direction.
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

        if snap_to_surface:
            closest, dist_to_mesh, face_id = self.mesh.nearest.on_surface([self.agent_pos])
            hit_n = self.mesh.face_normals[face_id[0]]
            hit_n = hit_n / (np.linalg.norm(hit_n) + 1e-12)
            if np.dot(hit_n, n) < 0:
                hit_n = -hit_n

            same_face = np.dot(hit_n, n) > 0.5

            if same_face:
                self.agent_pos = closest[0] + hit_n * 2.0
                self.agent_rot = self._look_at_direction(-hit_n)
            else:
                self.agent_pos = old_pos

    def _move_tangentially_old(self, direction_degrees, step_size, snap_to_surface: bool = True):
        """
        Movement tangent to the surface using a tangent basis (t1, t2).

        This fixes degeneracy where local XZ directions, after rotation+projection,
        collapse to ~1D on some faces/poses.

        direction_degrees:
            angle in the tangent plane:
            0°   -> +t1
            90°  -> +t2
            180° -> -t1
            270° -> -t2
        """
        rot = R.from_euler("xyz", self.agent_rot, degrees=True)

        sensor_data = self.get_sensor_data()
        if sensor_data["point_normal"] is None:
            # Fallback: keep old behavior if no normal
            angle_rad = np.radians(direction_degrees)
            local_dir = np.array([np.sin(angle_rad), 0.0, -np.cos(angle_rad)], dtype=float)
            local_dir /= (np.linalg.norm(local_dir) + 1e-12)
            world_dir = rot.apply(local_dir)
            self.agent_pos += world_dir * step_size
            return

        n = np.array(sensor_data["point_normal"], dtype=float)
        n /= (np.linalg.norm(n) + 1e-12)

        # Build tangent basis.
        # Try sensor's right axis as primary tangent direction.
        right_world = rot.apply([1.0, 0.0, 0.0])
        t1 = right_world - np.dot(right_world, n) * n
        t1_norm = np.linalg.norm(t1)

        if t1_norm < 1e-8:
            # If right is parallel to normal, try sensor up axis
            up_world = rot.apply([0.0, 1.0, 0.0])
            t1 = up_world - np.dot(up_world, n) * n
            t1_norm = np.linalg.norm(t1)

        if t1_norm < 1e-8:
            # Pathological fallback: pick any vector not parallel to n
            tmp = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(tmp, n)) > 0.9:
                tmp = np.array([0.0, 0.0, 1.0])
            t1 = np.cross(n, tmp)
            t1_norm = np.linalg.norm(t1)

        t1 /= (t1_norm + 1e-12)

        # Second tangent axis
        t2 = np.cross(n, t1)
        t2 /= (np.linalg.norm(t2) + 1e-12)

        a = np.radians(direction_degrees)
        world_dir = np.cos(a) * t1 + np.sin(a) * t2
        world_dir /= (np.linalg.norm(world_dir) + 1e-12)

        # Move
        self.agent_pos += world_dir * step_size

        # Optional: snap back to surface at +2mm along normal for stability
        if snap_to_surface:
            closest, dist_to_mesh, face_id = self.mesh.nearest.on_surface([self.agent_pos])
            hit_n = self.mesh.face_normals[face_id[0]]
            hit_n = hit_n / (np.linalg.norm(hit_n) + 1e-12)
            if np.dot(hit_n, n) < 0:
                hit_n = -hit_n
            self.agent_pos = closest[0] + hit_n * 2.0
            # ← ДОБАВИТЬ ЭТУ СТРОКУ:
            self.agent_rot = self._look_at_direction(-hit_n)
        else:
            logger.warning(f"SNAP FAILED: ray_origin={self.agent_pos}, n={n}")

    def _move_forward(self, step_size):
        """Forward movement (where the sensor is looking)."""
        rot = R.from_euler("xyz", self.agent_rot, degrees=True)
        forward = rot.apply([0, 0, -1])
        self.agent_pos += forward * step_size
    
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


def is_on_same_cube_side(pos_a, pos_b, cube_side=42.0, atol=1e-5):
    """
    Проверяет, лежат ли две точки на одной стороне куба (например, обе на +X при x = +42).
    
    Args:
        pos_a: положение точки A [x, y, z]
        pos_b: положение точки B [x, y, z]
        cube_side: значение координаты грани куба (например, 42 мм)
        atol: допуск на численную погрешность
    
    Returns:
        bool: True, если точки на одной стороне куба
    """
    pos_a = np.array(pos_a)
    pos_b = np.array(pos_b)
    
    # Проверяем каждую ось
    for i in range(3):
        # Обе точки должны быть на грани (координата ≈ ±cube_side)
        if (abs(abs(pos_a[i]) - cube_side) <= atol and 
            abs(abs(pos_b[i]) - cube_side) <= atol):
            
            # И знак должен совпадать (обе +42 или обе -42)
            if np.sign(pos_a[i]) == np.sign(pos_b[i]):
                return True
    
    return False
