# Новый файл: lightweight_env.py
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R


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
            on_object = bool(depth < 10.0)  # 10mm
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
        """Rough estimate of curvature."""
        return [0.0, 0.0]  # simplification for start
    
    def _look_at_direction_simple(self, direction):
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

    def _look_at_direction(self, direction):
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

    def _move_tangentially(self, direction_degrees, step_size):
        """Movement tangent to the surface.
        0°  = forward    → -Z
        90° = right      → +X
        180° = backward  → +Z
        270° = left      → -X
        """
        rot = R.from_euler("xyz", self.agent_rot, degrees=True)
        
        angle_rad = np.radians(direction_degrees)
        local_dir = np.array([
            np.sin(angle_rad),   # X: + = right
            0.0,                 # Y: ignore
            -np.cos(angle_rad)   # Z: -1 at 0° = forward
        ])
        
        # Normalize in case (though it's unit)
        local_dir /= (np.linalg.norm(local_dir) + 1e-12)
        
        # Transform to world
        world_dir = rot.apply(local_dir)
        
        # Project onto tangent plane if normal is available
        sensor_data = self.get_sensor_data()
        if sensor_data["point_normal"] is not None:
            normal = np.array(sensor_data["point_normal"])
            world_dir -= np.dot(world_dir, normal) * normal
            norm = np.linalg.norm(world_dir)
            if norm > 1e-8:
                world_dir /= norm
        
        self.agent_pos += world_dir * step_size

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

"""
Да, симулятора хватает для всех 13 компонент
Проверим каждую компоненту — откуда берётся:

Разбор state vector по источникам
scss
Копировать код
State [13D]          Откуда              Источник в LightweightEnv
─────────────────────────────────────────────────────────────────
local_pos_error [3D] pose + goal         ✅ env.get_pose() + goal
rot_error       [3D] pose + goal         ✅ env.get_pose() + goal  
local_normal    [3D] sensor_data         ✅ env.get_sensor_data()["point_normal"]
on_object       [1D] sensor_data         ✅ env.get_sensor_data()["on_object"]
alignment       [1D] вычисляется         ✅ dot(goal_dir, normal)
distance        [1D] pose + goal         ✅ norm(pos - goal_pos)
norm_depth      [1D] sensor_data         ✅ env.get_sensor_data()["depth"]
Что нужно от симулятора
Данные	Метод	Есть?
Позиция агента [x,y,z]	env.get_pose()[:3]	✅
Ориентация агента [rx,ry,rz]	env.get_pose()[3:6]	✅
Нормаль поверхности	env.get_sensor_data()["point_normal"]	✅
Глубина	env.get_sensor_data()["depth"]	✅
Флаг на объекте	env.get_sensor_data()["on_object"]	✅
Точка цели	env.get_random_surface_point()	✅
Вычисление state в контроллере
python
Копировать код
def _build_state(self, pose, sensor_data, goal):
    pos = pose[:3]
    rot_euler = pose[3:6]
    
    rot = R.from_euler("xyz", rot_euler, degrees=True)
    rot_inv = rot.inv()
    
    # [0:3] local_pos_error — направление к цели в локальной СК агента
    goal_vec = goal[:3] - pos
    local_pos_error = rot_inv.apply(goal_vec)
    
    # [3:6] rot_error — разница ориентации
    goal_rot = R.from_euler("xyz", goal[3:6], degrees=True)
    rot_diff = (rot.inv() * goal_rot).as_rotvec() / np.pi  # normalized
    
    # [6:9] local_normal — нормаль в локальной СК
    normal = np.array(sensor_data["point_normal"] or [0, 0, 0])
    local_normal = rot_inv.apply(normal)
    
    # [9] on_object
    on_object = 1.0 if sensor_data["on_object"] else 0.0
    
    # [10] alignment — dot(направление к цели, нормаль)
    dist = np.linalg.norm(goal_vec) + 1e-12
    alignment = np.dot(goal_vec / dist, normal)
    
    # [11] distance
    distance = dist
    
    # [12] norm_depth — нормализованная глубина
    norm_depth = min(sensor_data["depth"] / 100.0, 1.0)
    
    return np.array([
        *local_pos_error,   # 3
        *rot_diff,          # 3
        *local_normal,      # 3
        on_object,          # 1
        alignment,          # 1
        distance,           # 1
        norm_depth,         # 1
    ])                      # = 13
Вывод
Habitat не нужен для обучения. Всё что нужно для 13D state:

scss
Копировать код
LightweightEnv (trimesh)
    ├── get_pose()              → позиция + ориентация
    ├── get_sensor_data()       → нормаль + глубина + on_object
    └── get_random_surface_point() → цель

Контроллер сам вычисляет:
    ├── local_pos_error         из pose + goal
    ├── rot_error               из pose + goal
    ├── local_normal            из normal + pose
    ├── alignment               из goal_dir + normal
    ├── distance                из pose + goal
    └── norm_depth              из depth
Habitat понадобится только для inference в реальной среде Monty, но обучение полностью standalone.
"""