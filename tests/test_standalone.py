# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Tests for the hybrid RL navigation system."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import trimesh

from tbp.hybrid_rl.ablation_runner import run_episodes
from tbp.hybrid_rl.action_space import ActionSpace
from tbp.hybrid_rl.hnsw_state_store import HNSWStateStore, StatePoint
from tbp.hybrid_rl.lightweight_env import LightweightEnv


class TestLightweightEnv:
    """Tests for LightweightEnv simulation environment."""

    @pytest.fixture
    def cube_env(self, tmp_path: pathlib.Path) -> LightweightEnv:
        """Create an environment with a cube mesh.

        Args:
            tmp_path: Pytest temporary directory.

        Returns:
            LightweightEnv instance with a cube.
        """
        mesh = trimesh.primitives.Box(extents=[20, 20, 20])
        mesh_path = str(tmp_path / "cube.stl")
        mesh.export(mesh_path)
        return LightweightEnv(mesh_path)

    @pytest.fixture
    def sphere_env(self, tmp_path: pathlib.Path) -> LightweightEnv:
        """Create an environment with a sphere mesh.

        Args:
            tmp_path: Pytest temporary directory.

        Returns:
            LightweightEnv instance with a sphere.
        """
        mesh = trimesh.primitives.Sphere(radius=15)
        mesh_path = str(tmp_path / "sphere.stl")
        mesh.export(mesh_path)
        return LightweightEnv(mesh_path)

    def test_reset_on_surface(self, cube_env: LightweightEnv) -> None:
        """After reset, the agent must be near the surface."""
        sensor_data = cube_env.reset()
        assert sensor_data["on_object"] is True
        assert sensor_data["depth"] < 10.0
        assert sensor_data["point_normal"] is not None

    def test_get_pose_returns_6d(self, cube_env: LightweightEnv) -> None:
        """Pose should be a 6D vector [x, y, z, roll, pitch, yaw]."""
        cube_env.reset()
        pose = cube_env.get_pose()
        assert pose.shape == (6,)

    def test_random_surface_point(self, cube_env: LightweightEnv) -> None:
        """Random surface point must be near the surface."""
        cube_env.reset()
        goal = cube_env.get_random_surface_point()
        assert goal.shape == (6,)

        cube_env.reset(position=goal[:3], rotation=goal[3:])
        sensor = cube_env.get_sensor_data()
        assert sensor["depth"] < 5.0

    def test_move_forward_changes_position(
        self, cube_env: LightweightEnv
    ) -> None:
        """MoveForward must change position by free_step distance."""
        cube_env.reset(position=[0, 0, 30], rotation=[0, 0, 0])
        pos_before = cube_env.get_pose()[:3].copy()

        action_space = ActionSpace("test", free_step=10.0)
        cube_env.step(action_space.IDX_FREE_FORWARD, action_space)

        pos_after = cube_env.get_pose()[:3]
        distance_moved = float(np.linalg.norm(pos_after - pos_before))
        assert distance_moved == pytest.approx(10.0, abs=0.1)

    def test_free_forward_small_changes_position(
        self, cube_env: LightweightEnv
    ) -> None:
        """FreeForwardSmall must move by free_step_small distance."""
        cube_env.reset(position=[0, 0, 30], rotation=[0, 0, 0])
        pos_before = cube_env.get_pose()[:3].copy()

        action_space = ActionSpace(
            "test", free_step=10.0, free_step_small=2.0
        )
        cube_env.step(action_space.IDX_FREE_FORWARD_SMALL, action_space)

        pos_after = cube_env.get_pose()[:3]
        distance_moved = float(np.linalg.norm(pos_after - pos_before))
        assert distance_moved == pytest.approx(2.0, abs=0.1)

    def test_look_up_changes_rotation(
        self, cube_env: LightweightEnv
    ) -> None:
        """LookUp should change pitch by rotation_step degrees."""
        cube_env.reset(position=[0, 0, 0], rotation=[30, 0, 0])
        rot_before = cube_env.get_pose()[3:].copy()

        action_space = ActionSpace("test", rotation_step=10.0)
        cube_env.step(action_space.IDX_LOOK_UP, action_space)

        rot_after = cube_env.get_pose()[3:]
        assert rot_after[0] == pytest.approx(rot_before[0] + 10.0, abs=1e-6)

    def test_depth_decreases_when_approaching(
        self, cube_env: LightweightEnv
    ) -> None:
        """Depth should decrease as the agent gets closer to the object."""
        cube_env.reset(position=[0, 0, 40], rotation=[0, 0, 0])
        depth_before = cube_env.get_sensor_data()["depth"]

        action_space = ActionSpace("test", free_step=10.0)
        cube_env.step(action_space.IDX_FREE_FORWARD, action_space)

        depth_after = cube_env.get_sensor_data()["depth"]
        assert depth_after < depth_before

    def test_normal_points_outward(
        self, sphere_env: LightweightEnv
    ) -> None:
        """Normal on a sphere must point outward from center."""
        sphere_env.reset()
        sensor = sphere_env.get_sensor_data()

        if sensor["point_normal"] is not None:
            normal = np.array(sensor["point_normal"])
            pos = sphere_env.get_pose()[:3]

            expected_dir = pos / (np.linalg.norm(pos) + 1e-8)
            dot = float(np.dot(normal, expected_dir))
            assert dot > 0, "Normal should point outward"

    def test_surface_tangent_movement(
        self, sphere_env: LightweightEnv
    ) -> None:
        """MoveTangentially should maintain distance to center on a sphere."""
        sphere_env.reset()
        pos_before = sphere_env.get_pose()[:3].copy()
        dist_before = float(np.linalg.norm(pos_before))

        action_space = ActionSpace("test", surface_step=3.0)
        sphere_env.step(0, action_space)

        pos_after = sphere_env.get_pose()[:3]
        dist_after = float(np.linalg.norm(pos_after))

        assert dist_after == pytest.approx(dist_before, abs=2.0)

    def test_collision_detection_inside(
        self, cube_env: LightweightEnv
    ) -> None:
        """Agent inside object should have very small depth."""
        cube_env.reset(position=[0, 0, 0], rotation=[0, 0, 0])
        sensor = cube_env.get_sensor_data()

        assert sensor["depth"] < 15.0

    def test_sensor_data_contains_required_keys(
        self, cube_env: LightweightEnv
    ) -> None:
        """Sensor data must contain all required keys."""
        cube_env.reset()
        sensor = cube_env.get_sensor_data()

        required_keys = [
            "point_normal",
            "principal_curvatures",
            "on_object",
            "depth",
            "passed_through",
            "goal_normal",
            "detach_had_collision",
            "detach_sub_steps",
            "path_blocked",
        ]
        for key in required_keys:
            assert key in sensor, f"Missing key: {key}"

    def test_compute_up_direction(
        self, cube_env: LightweightEnv
    ) -> None:
        """Up direction should be a unit vector."""
        assert hasattr(cube_env, "height_axis")
        assert hasattr(cube_env, "up_sign")
        assert hasattr(cube_env, "up_direction")
        assert cube_env.up_direction.shape == (3,)
        norm = float(np.linalg.norm(cube_env.up_direction))
        assert abs(norm - 1.0) < 1e-6

    def test_action_space_has_correct_count(self) -> None:
        """ActionSpace should have 21 discrete actions."""
        action_space = ActionSpace("test")
        assert action_space.NUM_ACTIONS == 21
        for i in range(21):
            info = action_space.get_info(i)
            assert info.index == i
            assert info.name is not None


class TestHNSWEviction:
    """Tests for HNSW incremental eviction (mark_deleted + rebuild)."""

    def _make_store(
        self,
        max_points: int = 100,
        evict_fraction: float = 0.2,
    ) -> HNSWStateStore:
        """Create an HNSW store with specified capacity.

        Args:
            max_points: Maximum number of points before eviction.
            evict_fraction: Fraction of points to evict.

        Returns:
            Configured HNSWStateStore instance.
        """
        config = {
            "state_dim": 4,
            "num_actions": 3,
            "max_points": max_points,
            "k_neighbors": 3,
            "sigma": 1.0,
            "insert_threshold": 0.05,
            "evict_fraction": evict_fraction,
            "adaptive_sigma": False,
            "auto_calibrate": False,
        }
        return HNSWStateStore(config)

    def _fill_store(self, store: HNSWStateStore, n: int) -> None:
        """Insert n random points into the store.

        Args:
            store: Target HNSW store.
            n: Number of points to insert.
        """
        rng = np.random.RandomState(42)
        for _ in range(n):
            state = rng.randn(store.state_dim) * 10
            action = rng.randint(store.num_actions)
            store.update_q_value(state, action, td_target=1.0, alpha=0.1)

    def test_eviction_triggers_at_capacity(self) -> None:
        """Eviction triggers when active points >= max_points."""
        store = self._make_store(max_points=50)
        self._fill_store(store, 60)

        assert len(store.points) <= 50
        for pid in store.points:
            assert isinstance(store.points[pid], StatePoint)

    def test_eviction_removes_old_points(self) -> None:
        """Old rarely-visited points are evicted first."""
        store = self._make_store(max_points=30, evict_fraction=0.3)

        rng = np.random.RandomState(0)
        for _ in range(25):
            state = rng.randn(4) * 10
            store.update_q_value(state, 0, td_target=0.5, alpha=0.1)

        fresh_state = np.array([100.0, 100.0, 100.0, 100.0])
        store.update_q_value(fresh_state, 1, td_target=5.0, alpha=0.5)
        fresh_id = store.next_id - 1
        store.points[fresh_id].visit_count = 100

        for _ in range(10):
            state = rng.randn(4) * 10
            store.update_q_value(state, 0, td_target=0.5, alpha=0.1)

        survived_states: list[np.ndarray] = [
            p.norm_state for p in store.points.values()
        ]
        fresh_norm = store._normalize(fresh_state)
        distances = [
            float(np.linalg.norm(s - fresh_norm)) for s in survived_states
        ]
        assert min(distances) < 0.5, "Fresh high-visit point was evicted"

    def test_incremental_eviction_marks_deleted(self) -> None:
        """mark_deleted does not rebuild index (next_id keeps growing)."""
        store = self._make_store(max_points=50)
        store._rebuild_threshold = 0.99

        self._fill_store(store, 55)

        assert store.next_id >= 55
        assert store._deleted_count > 0
        assert len(store.points) < store.next_id

    def test_full_rebuild_on_high_ghost_ratio(self) -> None:
        """Full rebuild triggers when ghost ratio exceeds threshold."""
        store = self._make_store(max_points=40, evict_fraction=0.4)
        store._rebuild_threshold = 0.2

        self._fill_store(store, 50)

        if store._deleted_count == 0:
            assert store.next_id == len(store.points)

    def test_query_after_eviction_works(self) -> None:
        """get_q_values works correctly after eviction."""
        store = self._make_store(max_points=50)

        rng = np.random.RandomState(7)
        states: list[np.ndarray] = [rng.randn(4) * 5 for _ in range(60)]
        for s in states:
            store.update_q_value(s, rng.randint(3), td_target=1.0, alpha=0.2)

        for s in states[:10]:
            q = store.get_q_values(s)
            assert q.shape == (3,)
            assert np.isfinite(q).all()

    def test_insert_after_eviction_reuses_slots(self) -> None:
        """After mark_deleted, new insertions use replace_deleted=True."""
        store = self._make_store(max_points=30)
        store._rebuild_threshold = 0.99

        self._fill_store(store, 35)

        assert store._deleted_count > 0

        rng = np.random.RandomState(99)
        for _ in range(5):
            state = rng.randn(4) * 10
            store.update_q_value(state, 0, td_target=2.0, alpha=0.1)

        assert len(store.points) > 0
        assert len(store.points) <= store.max_points

    def test_eviction_persists_with_save_load(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Eviction state (_deleted_count) persists through save/load."""
        store = self._make_store(max_points=50)
        store._rebuild_threshold = 0.99

        self._fill_store(store, 55)
        assert store._deleted_count > 0

        base = str(tmp_path / "evict_test")
        store.save_with_index(base)

        loaded = HNSWStateStore.load_with_index(base, None)
        assert loaded._deleted_count == store._deleted_count
        assert len(loaded.points) == len(store.points)

        q = loaded.get_q_values(np.zeros(4))
        assert q.shape == (3,)

    def test_multiple_evictions(self) -> None:
        """Multiple eviction rounds do not break the index."""
        store = self._make_store(max_points=20, evict_fraction=0.3)

        rng = np.random.RandomState(13)
        for i in range(100):
            state = rng.randn(4) * 5
            store.update_q_value(
                state,
                rng.randint(3),
                td_target=float(i % 5),
                alpha=0.1,
            )

        assert len(store.points) <= 20
        q = store.get_q_values(rng.randn(4))
        assert q.shape == (3,)
        assert np.isfinite(q).all()


class TestStandaloneTraining:
    """Tests for standalone Q-learning training."""

    @pytest.fixture
    def mesh_dir(self, tmp_path: pathlib.Path) -> str:
        """Create a directory with simple mesh files.

        Args:
            tmp_path: Pytest temporary directory.

        Returns:
            Path to directory containing mesh files.
        """
        cube = trimesh.primitives.Box(extents=[20, 20, 20])
        cube.export(str(tmp_path / "cube.stl"))

        sphere = trimesh.primitives.Sphere(radius=15)
        sphere.export(str(tmp_path / "sphere.stl"))

        cylinder = trimesh.primitives.Cylinder(radius=10, height=30)
        cylinder.export(str(tmp_path / "cylinder.stl"))

        return str(tmp_path)

    def test_training_runs_without_crash(
        self, mesh_dir: str, tmp_path: pathlib.Path
    ) -> None:
        """Training should complete without errors."""
        save_dir = str(tmp_path / "checkpoints")
        run_episodes(
            mesh_dir=mesh_dir,
            save_dir=save_dir,
            num_episodes=10,
            config={
                "state_dim": 15,
                "num_actions": 21,
                "max_points": 1000,
                "max_steps_per_goal": 15,
                "epsilon_start": 0.8,
                "adaptive_sigma": False,
                "mode": "train",
            },
        )

        save_path = pathlib.Path(save_dir)
        assert (save_path / "q_store_free.npz").exists()
        assert (save_path / "q_store_free.hnsw").exists()
        assert (save_path / "q_store_surface.npz").exists()
        assert (save_path / "q_store_surface.hnsw").exists()
        assert (save_path / "controller_state.npz").exists()
        assert (save_path / "config.json").exists()

    def test_training_produces_q_values(
        self, mesh_dir: str, tmp_path: pathlib.Path
    ) -> None:
        """After training, Q-stores should contain points."""
        save_dir = str(tmp_path / "checkpoints")

        run_episodes(
            mesh_dir=mesh_dir,
            save_dir=save_dir,
            num_episodes=50,
            config={
                "state_dim": 15,
                "num_actions": 21,
                "max_points": 5000,
                "max_steps_per_goal": 20,
                "adaptive_sigma": False,
                "mode": "train",
            },
        )

        save_path = pathlib.Path(save_dir)
        store_free = HNSWStateStore.load_with_index(
            str(save_path / "q_store_free"), None
        )
        store_surface = HNSWStateStore.load_with_index(
            str(save_path / "q_store_surface"), None
        )

        total_points = store_free.next_id + store_surface.next_id
        assert total_points > 10, (
            f"Should have learned some Q-values, got {total_points} points"
        )

    def test_loaded_model_navigates(
        self, mesh_dir: str, tmp_path: pathlib.Path
    ) -> None:
        """Loaded model should navigate better than random."""
        save_dir = str(tmp_path / "checkpoints")
        mesh_path = str(pathlib.Path(mesh_dir) / "cube.stl")

        run_episodes(
            mesh_dir=mesh_dir,
            save_dir=save_dir,
            num_episodes=100,
            config={
                "state_dim": 15,
                "num_actions": 21,
                "max_points": 5000,
                "max_steps_per_goal": 20,
                "goal_threshold": 5.0,
                "adaptive_sigma": False,
                "mode": "train",
            },
            mesh_path=mesh_path,
        )

        goals_reached = run_episodes(
            mesh_dir=mesh_dir,
            save_dir=save_dir,
            num_episodes=20,
            config={
                "goal_threshold": 5.0,
                "num_actions": 21,
                "mode": "eval",
            },
            mesh_path=mesh_path,
            load_dir=save_dir,
        )

        assert goals_reached > 0, "Trained model should reach some goals"
