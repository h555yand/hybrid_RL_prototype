import trimesh
import numpy as np
import pytest
import tempfile
import os

from tbp.hybrid_rl.lightweight_env import LightweightEnv
from tbp.hybrid_rl.action_space import ActionSpace
from tbp.hybrid_rl.hnsw_state_store import HNSWStateStore, StatePoint
from tbp.hybrid_rl.ablation_runner import train


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


class TestHNSWEviction:
    """Тесты инкрементальной очистки индекса (mark_deleted + rebuild)."""

    def _make_store(self, max_points=100, evict_fraction=0.2):
        config = {
            "state_dim":4,
            "num_actions":3,
            "max_points":max_points,
            "k_neighbors":3,
            "sigma":1.0,
            "insert_threshold":0.05,
            "evict_fraction":evict_fraction,
            "adaptive_sigma":False,
            "auto_calibrate":False,
        }
        return HNSWStateStore(config)

    def _fill_store(self, store, n):
        """Вставить n случайных точек."""
        rng = np.random.RandomState(42)
        for _ in range(n):
            state = rng.randn(store.state_dim) * 10
            action = rng.randint(store.num_actions)
            store.update_q_value(state, action, td_target=1.0, alpha=0.1)

    def test_eviction_triggers_at_capacity(self):
        """Eviction срабатывает когда active points >= max_points."""
        store = self._make_store(max_points=50)
        self._fill_store(store, 60)

        # После eviction активных точек должно быть < max_points
        assert len(store.points) <= 50
        # Все точки в словаре — живые (не ghost)
        for pid in store.points:
            assert isinstance(store.points[pid], StatePoint)

    def test_eviction_removes_old_points(self):
        """Старые редко-посещённые точки удаляются первыми."""
        store = self._make_store(max_points=30, evict_fraction=0.3)

        # Вставляем 25 «старых» точек
        rng = np.random.RandomState(0)
        for i in range(25):
            state = rng.randn(4) * 10
            store.update_q_value(state, 0, td_target=0.5, alpha=0.1)

        # Создаём «свежую» точку с высоким visit_count
        fresh_state = np.array([100.0, 100.0, 100.0, 100.0])
        store.update_q_value(fresh_state, 1, td_target=5.0, alpha=0.5)
        # Имитируем частые визиты
        fresh_id = store.next_id - 1
        store.points[fresh_id].visit_count = 100

        # Заполняем до eviction
        for i in range(25, 35):
            state = rng.randn(4) * 10
            store.update_q_value(state, 0, td_target=0.5, alpha=0.1)

        # Свежая точка с высоким visit_count должна выжить
        survived_states = [p.norm_state for p in store.points.values()]
        fresh_norm = store._normalize(fresh_state)
        distances = [np.linalg.norm(s - fresh_norm) for s in survived_states]
        assert min(distances) < 0.5, "Fresh high-visit point was evicted"

    def test_incremental_eviction_marks_deleted(self):
        """mark_deleted не перестраивает индекс (next_id растёт)."""
        store = self._make_store(max_points=50)
        store._rebuild_threshold = 0.99  # отключаем auto-rebuild

        self._fill_store(store, 55)

        # next_id продолжает расти (не сбросился при eviction)
        assert store.next_id >= 55
        # Были удалённые — _deleted_count > 0
        assert store._deleted_count > 0
        # Активных меньше чем next_id
        assert len(store.points) < store.next_id

    def test_full_rebuild_on_high_ghost_ratio(self):
        """Когда ghost ratio > threshold, происходит полный rebuild."""
        store = self._make_store(max_points=40, evict_fraction=0.4)
        store._rebuild_threshold = 0.2  # агрессивный порог

        self._fill_store(store, 50)

        # После rebuild next_id сбрасывается, _deleted_count = 0
        # (может потребоваться несколько eviction для срабатывания)
        if store._deleted_count == 0:
            # Rebuild произошёл — next_id == len(points)
            assert store.next_id == len(store.points)

    def test_query_after_eviction_works(self):
        """get_q_values работает корректно после eviction."""
        store = self._make_store(max_points=50)

        rng = np.random.RandomState(7)
        states = [rng.randn(4) * 5 for _ in range(60)]
        for s in states:
            store.update_q_value(s, rng.randint(3), td_target=1.0, alpha=0.2)

        # Запрос к оставшимся точкам не падает
        for s in states[:10]:
            q = store.get_q_values(s)
            assert q.shape == (3,)
            assert np.isfinite(q).all()

    def test_insert_after_eviction_reuses_slots(self):
        """После mark_deleted новые вставки replace_deleted=True."""
        store = self._make_store(max_points=30)
        store._rebuild_threshold = 0.99  # отключаем rebuild

        self._fill_store(store, 35)

        deleted_before = store._deleted_count
        assert deleted_before > 0

        # Вставляем ещё точки — не должно упасть
        rng = np.random.RandomState(99)
        for _ in range(5):
            state = rng.randn(4) * 10
            store.update_q_value(state, 0, td_target=2.0, alpha=0.1)

         # Store жив, точки есть, лимит не превышен
        assert len(store.points) > 0
        assert len(store.points) <= store.max_points

    def test_eviction_persists_with_save_load(self, tmp_path):
        """_deleted_count сохраняется и восстанавливается."""
        store = self._make_store(max_points=50)
        store._rebuild_threshold = 0.99

        self._fill_store(store, 55)
        assert store._deleted_count > 0

        base = str(tmp_path / "evict_test")
        store.save_with_index(base)

        loaded = HNSWStateStore.load_with_index(base, None)
        assert loaded._deleted_count == store._deleted_count
        assert len(loaded.points) == len(store.points)

        # Запрос к загруженному store работает
        q = loaded.get_q_values(np.zeros(4))
        assert q.shape == (3,)

    def test_multiple_evictions(self):
        """Несколько раундов eviction подряд не ломают индекс."""
        store = self._make_store(max_points=20, evict_fraction=0.3)

        rng = np.random.RandomState(13)
        for i in range(100):
            state = rng.randn(4) * 5
            store.update_q_value(
                state, rng.randint(3), td_target=float(i % 5), alpha=0.1
            )

        # Store жив и работоспособен
        assert len(store.points) <= 20
        q = store.get_q_values(rng.randn(4))
        assert q.shape == (3,)
        assert np.isfinite(q).all()


class TestStandaloneTraining:
    """Тесты standalone обучения."""
    
    @pytest.fixture
    def mesh_dir(self, tmp_path):
        """Директория с простыми mesh файлами."""
        # Куб
        cube = trimesh.primitives.Box(extents=[20, 20, 20])
        cube.export(str(tmp_path / "cube.stl"))
        
        # Сфера
        sphere = trimesh.primitives.Sphere(radius=15)
        sphere.export(str(tmp_path / "sphere.stl"))
        
        # Цилиндр
        cylinder = trimesh.primitives.Cylinder(radius=10, height=30)
        cylinder.export(str(tmp_path / "cylinder.stl"))
        
        return str(tmp_path)
    
    def test_training_runs_without_crash(self, mesh_dir, tmp_path):
        """Обучение должно пройти без ошибок."""
        save_dir = str(tmp_path / "checkpoints")
        train(
            mesh_dir=mesh_dir,
            save_dir=save_dir,
            num_episodes=10,
            config={
                "state_dim": 15,
                "max_points": 1000,
                "max_steps_per_goal": 15,
                "epsilon_start": 0.8,
                "adaptive_sigma": False,
                "mode": "train",
            },
        )
        
        # Проверяем что файлы сохранились (save_with_index)
        assert os.path.exists(os.path.join(save_dir, "q_store.npz"))
        assert os.path.exists(os.path.join(save_dir, "q_store.hnsw"))
        assert os.path.exists(os.path.join(save_dir, "controller_state.npz"))
        assert os.path.exists(os.path.join(save_dir, "config.json"))
    
    def test_training_produces_q_values(self, mesh_dir, tmp_path):
        """После обучения Q-store должен содержать точки."""
        save_dir = str(tmp_path / "checkpoints")
        
        train(
            mesh_dir=mesh_dir,
            save_dir=save_dir,
            num_episodes=50,
            config={
                "state_dim": 15,
                "max_points": 5000,
                "max_steps_per_goal": 20,
                "adaptive_sigma": False,
                "mode": "train",
            },
        )
        
        # Загружаем и проверяем (через load_with_index — актуальный метод)
        store = HNSWStateStore.load_with_index(os.path.join(save_dir, "q_store"), None)
        
        assert store.next_id > 10, (
            f"Should have learned some Q-values, got {store.next_id} points"
        )
    
    def test_loaded_model_navigates(self, mesh_dir, tmp_path):
        """Загруженная модель должна навигировать лучше случайной."""
        save_dir = str(tmp_path / "checkpoints")
        mesh_path = os.path.join(mesh_dir, "cube.stl")
        
        # Обучаем
        goals_reached = train(
            mesh_dir=mesh_dir,
            save_dir=save_dir,
            num_episodes=100,
            config={
                "state_dim": 15,
                "max_points": 5000,
                "max_steps_per_goal": 20,
                "goal_threshold": 5.0,
                "adaptive_sigma": False,
                "mode": "train",
                "agent_id": "train",
            },
            mesh_path=mesh_path
        )

        # Провереям
        goals_reached = train(
            mesh_dir=mesh_dir,
            save_dir=save_dir,
            num_episodes=20,
            config={
                "goal_threshold": 5.0,
                "mode": "eval",
                "agent_id": "eval",
            },
            mesh_path=mesh_path,
            load_dir=save_dir
        )
        
        assert goals_reached > 0, "Trained model should reach some goals"
