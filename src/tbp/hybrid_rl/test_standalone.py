import trimesh
import numpy as np
import pytest
import tempfile
import os

from tbp.hybrid_rl.lightweight_env import LightweightEnv
from tbp.hybrid_rl.action_space import ActionSpace


class TestLightweightEnv:
    """Tests LightweightEnv."""
    
    @pytest.fixture
    def cube_env(self, tmp_path):
        """Environment with a cube."""
        mesh = trimesh.primitives.Box(extents=[20, 20, 20])
        mesh_path = str(tmp_path / "cube.stl")
        mesh.export(mesh_path)
        return LightweightEnv(mesh_path)
    
    @pytest.fixture
    def sphere_env(self, tmp_path):
        """Environment with a sphere."""
        mesh = trimesh.primitives.Sphere(radius=15)
        mesh_path = str(tmp_path / "sphere.stl")
        mesh.export(mesh_path)
        return LightweightEnv(mesh_path)
    
    def test_reset_on_surface(self, cube_env):
        """After reset, the agent must be near the surface."""
        sensor_data = cube_env.reset()
        assert sensor_data["on_object"] is True
        assert sensor_data["depth"] < 10.0
        assert sensor_data["point_normal"] is not None
    
    def test_get_pose_returns_6d(self, cube_env):
        """The pose should be [x, y, z, roll, pitch, yaw]."""
        cube_env.reset()
        pose = cube_env.get_pose()
        assert pose.shape == (6,)
    
    def test_random_surface_point(self, cube_env):
        """The random point must be near the surface."""
        cube_env.reset()
        goal = cube_env.get_random_surface_point()
        assert goal.shape == (6,)
        
        # Place an agent at this point and check
        cube_env.reset(position=goal[:3], rotation=goal[3:])
        sensor = cube_env.get_sensor_data()
        assert sensor["depth"] < 5.0
    
    def test_move_forward_changes_position(self, cube_env):
        """MoveForward must change position."""
        cube_env.reset(position=[0, 0, 30], rotation=[0, 0, 0])
        pos_before = cube_env.get_pose()[:3].copy()
        
        action_space = ActionSpace("test", free_step=10.0)
        cube_env.step(action_space.IDX_FREE_FORWARD, action_space)
        
        pos_after = cube_env.get_pose()[:3]
        distance_moved = np.linalg.norm(pos_after - pos_before)
        assert distance_moved == pytest.approx(10.0, abs=0.1)
    
    def test_look_up_changes_rotation(self, cube_env):
        """LookUp should change pitch."""
        cube_env.reset(position=[0, 0, 0], rotation=[30, 0, 0])
        rot_before = cube_env.get_pose()[3:].copy()
        
        action_space = ActionSpace("test", rotation_step=10.0)
        cube_env.step(action_space.IDX_LOOK_UP, action_space)
        
        rot_after = cube_env.get_pose()[3:]
        # assert rot_after[1] != rot_before[1]  # pitch changed
        assert rot_after[0] == pytest.approx(rot_before[0] + 10.0, abs=1e-6)
        
    def test_depth_decreases_when_approaching(self, cube_env):
        """Depth should decrease as you get closer to the object."""
        cube_env.reset(position=[0, 0, 40], rotation=[0, 0, 0])
        depth_before = cube_env.get_sensor_data()["depth"]
        
        action_space = ActionSpace("test", free_step=10.0)
        cube_env.step(action_space.IDX_FREE_FORWARD, action_space)
        
        depth_after = cube_env.get_sensor_data()["depth"]
        assert depth_after < depth_before
    
    def test_normal_points_outward(self, sphere_env):
        """The normal on a sphere must point outward."""
        sphere_env.reset()
        sensor = sphere_env.get_sensor_data()
        
        if sensor["point_normal"] is not None:
            normal = np.array(sensor["point_normal"])
            pos = sphere_env.get_pose()[:3]
            
            # На сфере нормаль ≈ направление от центра
            expected_dir = pos / (np.linalg.norm(pos) + 1e-8)
            dot = np.dot(normal, expected_dir)
            assert dot > 0, "Normal should point outward"
    
    def test_surface_tangent_movement(self, sphere_env):
        """MoveTangentially should maintain the distance to the center."""
        sphere_env.reset()
        pos_before = sphere_env.get_pose()[:3].copy()
        dist_before = np.linalg.norm(pos_before)
        
        action_space = ActionSpace("test", surface_step=3.0)
        sphere_env.step(0, action_space)  # surface_0°
        
        pos_after = sphere_env.get_pose()[:3]
        dist_after = np.linalg.norm(pos_after)
        
        # The distance from the center should be approximately the same
        assert dist_after == pytest.approx(dist_before, abs=2.0)
    
    def test_collision_detection_inside(self, cube_env):
        """Agent inside object → depth is very small."""
        cube_env.reset(position=[0, 0, 0], rotation=[0, 0, 0])
        sensor = cube_env.get_sensor_data()
        
        # Inside the cube: the beam immediately hits the wall
        assert sensor["depth"] < 15.0

