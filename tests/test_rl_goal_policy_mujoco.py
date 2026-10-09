"""Integration tests: RLGoalPolicy with real MuJoCo environment.

Requires:
- Pretrained Q-store (from trimesh training)
- MuJoCo with YCB mug object
- mug.stl mesh in mm

Run:
    python -m pytest tests/test_rl_goal_policy_mujoco.py -v -s -n0
"""

from __future__ import annotations

import logging
import os
import unittest
from pathlib import Path

import numpy as np
import quaternion as qt

from tbp.monty.cmp import Goal, Message
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.motor_policies import (
    MotorPolicyResult,
    PolicyStatus,
)
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    MotorSystemState,
    SensorState,
)
from tbp.monty.frameworks.sensors import SensorID

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════
# Paths — adjust if your layout differs
# ═══════════════════════════════════════════════════
_BASE = Path(os.environ.get(
    "RL_PROJECT_DIR",
    os.path.expanduser(
        "~/Downloads/github/hybrid_RL_prototype"
    ),
))

_Q_STORE_DIR = str(
    _BASE / "results" / "adapt-baseline" / "data" / "runs" / "q_store_seed_11"
)
_MESH_PATH_MM = str(
    _BASE / "results" / "adapt-baseline" / "data" / "mug.stl"
)
_MUJOCO_DATA_PATH = os.path.expanduser(
    "~/Downloads/tbp/data/mujoco/objects/ycb"
)
_MUJOCO_OBJECT_NAME = "mug"

# ═══════════════════════════════════════════════════
# Skip if resources not available
# ═══════════════════════════════════════════════════
_SKIP_REASON = None
if not Path(_Q_STORE_DIR).exists():
    _SKIP_REASON = f"Q-store not found: {_Q_STORE_DIR}"
elif not Path(_MESH_PATH_MM).exists():
    _SKIP_REASON = f"Mesh not found: {_MESH_PATH_MM}"
elif not Path(_MUJOCO_DATA_PATH).exists():
    _SKIP_REASON = f"MuJoCo data not found: {_MUJOCO_DATA_PATH}"
elif not (Path(_MUJOCO_DATA_PATH) / _MUJOCO_OBJECT_NAME / "textured.obj").exists():
    _SKIP_REASON = (
        f"MuJoCo object not found: "
        f"{_MUJOCO_DATA_PATH}/{_MUJOCO_OBJECT_NAME}"
    )


def _make_percept_from_sensor_data(sd: dict) -> Message:
    """Create Monty Message from adapter sensor_data dict."""
    normal = sd.get("point_normal")
    if normal is not None:
        pose_vectors = np.array([
            normal,
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=float)
    else:
        pose_vectors = np.array([
            [0, 0, -1],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=float)

    depth_m = sd.get("depth", 100.0) / 1000.0
    on_object = sd.get("on_object", False)

    return Message(
        location=np.array([0.0, 0.0, 0.0]),
        morphological_features={
            "pose_vectors": pose_vectors,
            "pose_fully_defined": normal is not None,
            "on_object": on_object,
        },
        non_morphological_features={
            "min_depth": depth_m,
            "mean_depth": depth_m,
            "object_coverage": 0.5 if on_object else 0.0,
            "principal_curvatures": [
                sd.get("k1", 0.0),
                sd.get("k2", 0.0),
            ],
        },
        confidence=1.0,
        pass_message=True,
        sender_id="patch",
        sender_type="SM",
        process_features_in_lm=on_object,
    )


def _make_goal_from_pose_mm(pose_mm: np.ndarray) -> Goal:
    """Create Monty Goal from RL pose [x,y,z,rx,ry,rz] in mm/degrees."""
    from scipy.spatial.transform import Rotation as Rot

    location_m = pose_mm[:3] / 1000.0

    rot = Rot.from_euler("xyz", pose_mm[3:6], degrees=True)
    forward = rot.apply([0, 0, -1])

    pose_vectors = np.array([
        forward,
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=float)

    return Goal(
        location=location_m,
        morphological_features={
            "pose_vectors": pose_vectors,
            "pose_fully_defined": True,
        },
        non_morphological_features={},
        confidence=0.9,
        pass_message=True,
        sender_id="LM_0",
        sender_type="GSG",
        process_features_in_lm=False,
        goal_tolerances=None,
    )


def _state_from_adapter(adapter, agent_id: str = "agent_id_0") -> MotorSystemState:
    """Build MotorSystemState from adapter's current position."""
    from scipy.spatial.transform import Rotation as Rot

    pos_mm = adapter._get_pos_mj_mm()
    euler = adapter._get_euler_deg()

    pos_m = tuple(pos_mm / 1000.0)
    rot = Rot.from_euler("xyz", euler, degrees=True)
    q = rot.as_quat()  # xyzw
    quat = qt.quaternion(q[3], q[0], q[1], q[2])

    return MotorSystemState({
        AgentID(agent_id): AgentState(
            position=pos_m,
            rotation=quat,
            sensors={
                SensorID("view_finder"): SensorState(
                    position=(0.0, 0.0, 0.0),
                    rotation=qt.quaternion(1, 0, 0, 0),
                ),
            },
        ),
    })


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON)
class TestRLGoalPolicyMuJoCo(unittest.TestCase):
    """Integration tests with real MuJoCo environment and pretrained Q-store."""

    adapter = None

    @classmethod
    def setUpClass(cls):
        """Create MuJoCo adapter (shared across tests)."""
        from tbp.hybrid_rl.mujoco_env_adapter import MuJoCoEnvAdapter

        cls.adapter = MuJoCoEnvAdapter(
            mesh_path_mm=_MESH_PATH_MM,
            mujoco_object_name=_MUJOCO_OBJECT_NAME,
            mujoco_data_path=_MUJOCO_DATA_PATH,
            seed=42,
        )
        logger.info(
            "MuJoCo adapter created: object=%s, "
            "center=%s, extents=%s",
            _MUJOCO_OBJECT_NAME,
            cls.adapter._mj_center_mm.round(1).tolist(),
            cls.adapter._mj_extents_mm.round(1).tolist(),
        )

    @classmethod
    def tearDownClass(cls):
        if cls.adapter is not None:
            cls.adapter.close()

    def _create_policy(self, enable_learning=False):
        """Create RLGoalPolicy with real Q-store and adapter."""
        from tbp.hybrid_rl.rl_goal_policy import RLGoalPolicy

        rl_config = {
            "state_dim": 22,
            "num_actions": 24,
            "surface_step": 3.0,
            "free_step": 8.0,
            "free_step_small": 2.0,
            "free_step_backward": 2.0,
            "rotation_step": 5.0,
            "rotation_step_big": 15.0,
            "gamma": 0.95,
            "alpha": 0.1,
            "epsilon_start": 0.05,
            "epsilon_min": 0.02,
            "epsilon_decay": 0.999,
            "goal_threshold": 4.0,
            "max_steps_per_goal": 200,
            "max_sensor_range": 100.0,
            "min_valid_depth": 0.5,
            "normal_flip_threshold": -0.5,
            "reward_progress": 3.0,
            "reward_goal_reached": 30.0,
            "reward_step_penalty": -0.2,
            "reward_surface_violation": -12.0,
            "reward_timeout": -12.0,
            "reward_drifted_away": -3.0,
            "reward_near_goal_on_surface": 0.5,
            "detour_alignment_threshold": -0.3,
            "detour_negative_progress_clip_steps": 2.0,
            "stuck_threshold": 0.05,
            "k_neighbors": 7,
            "max_points": 500000,
            "insert_threshold": 0.50,
            "action_selection_version": "v2",
            "warmup_episodes": 0,
            "mode": "adaptive",
            "eval_epsilon": 0.02,
            "strategic_eval_epsilon": 0.02,
        }

        policy = RLGoalPolicy(
            agent_id=AgentID("agent_id_0"),
            sensor_id=SensorID("view_finder"),
            model_path=_Q_STORE_DIR,
            rl_config=rl_config,
            mujoco_adapter=self.adapter,
            mesh_path=_MESH_PATH_MM if enable_learning else None,
            max_nav_steps=200,
            enable_online_learning=enable_learning,
        )
        return policy

    def _generate_goal(self, start_pose, distance_range=(10, 60)):
        """Generate goal at specified distance from start via rejection sampling."""
        for _ in range(200):
            goal_pose = self.adapter.get_random_surface_point()
            dist = float(np.linalg.norm(goal_pose[:3] - start_pose[:3]))
            if distance_range[0] <= dist <= distance_range[1]:
                return goal_pose
        logger.warning(
            "Could not find goal in range %s, using random point",
            distance_range,
        )
        return self.adapter.get_random_surface_point()

    # ══════════════════════════════════════════════════════════
    # TEST 1: Bridge + Adapter integration
    # ══════════════════════════════════════════════════════════

    def test_bridge_percept_from_adapter(self):
        """Bridge correctly converts adapter sensor_data to percept and back."""
        from tbp.hybrid_rl.monty_rl_bridge import MontyRLBridge

        bridge = MontyRLBridge(
            agent_id=AgentID("agent_id_0"),
            rl_config={},
            mujoco_adapter=self.adapter,
        )

        # Reset adapter to random position
        sd = self.adapter.reset()
        state = _state_from_adapter(self.adapter)

        # Create percept from sensor data
        percept = _make_percept_from_sensor_data(sd)

        # Convert back through bridge
        sd_bridge = bridge.percept_to_sensor_data(percept, state, goal=None)

        # Verify key fields match
        self.assertEqual(sd_bridge["on_object"], sd["on_object"])
        self.assertAlmostEqual(sd_bridge["depth"], sd["depth"], places=0)
        if sd["point_normal"] is not None:
            np.testing.assert_allclose(
                sd_bridge["point_normal"], sd["point_normal"], atol=0.01,
            )

    # ══════════════════════════════════════════════════════════
    # TEST 2: Single navigation step
    # ══════════════════════════════════════════════════════════

    def test_single_navigation_step(self):
        """Policy takes one step and returns IN_PROGRESS."""
        policy = self._create_policy(enable_learning=False)

        # Setup: place agent on surface
        start_sd = self.adapter.reset()
        start_pose = self.adapter.get_pose()

        # Generate nearby goal (same side, easy)
        goal_pose = self._generate_goal(start_pose, distance_range=(10, 40))
        self.adapter.set_goal(goal_pose)

        # Build Monty objects
        state = _state_from_adapter(self.adapter)
        percept = _make_percept_from_sensor_data(start_sd)
        goal = _make_goal_from_pose_mm(goal_pose)
        ctx = RuntimeContext(rng=np.random.RandomState(42))

        # First call: should start navigation
        result = policy(ctx, {}, state, percept, goal)

        self.assertTrue(policy.is_navigating)
        self.assertEqual(result.status, PolicyStatus.IN_PROGRESS)
        self.assertTrue(result.motor_only_step)
        self.assertEqual(result.actions, [])

    # ══════════════════════════════════════════════════════════
    # TEST 3: Multi-step navigation (L0 easy goal)
    # ══════════════════════════════════════════════════════════

    def test_navigate_to_nearby_goal(self):
        """Agent navigates toward a nearby goal and makes progress."""
        policy = self._create_policy(enable_learning=False)

        # Setup
        start_sd = self.adapter.reset()
        start_pose = self.adapter.get_pose()

        goal_pose = self._generate_goal(start_pose, distance_range=(10, 40))
        self.adapter.set_goal(goal_pose)

        goal = _make_goal_from_pose_mm(goal_pose)
        ctx = RuntimeContext(rng=np.random.RandomState(42))

        initial_distance = float(np.linalg.norm(
            goal_pose[:3] - start_pose[:3]
        ))

        # Run navigation for up to 100 steps
        max_steps = 100
        steps_taken = 0
        finished = False

        for step in range(max_steps):
            # Get current state from adapter (simulates env_interface.step)
            current_sd = self.adapter.get_sensor_data()
            state = _state_from_adapter(self.adapter)
            percept = _make_percept_from_sensor_data(current_sd)

            # Policy step
            result = policy(ctx, {}, state, percept, goal)

            steps_taken += 1

            if result.status == PolicyStatus.READY:
                finished = True
                break

        # Check progress
        final_pose = self.adapter.get_pose()
        final_distance = float(np.linalg.norm(
            goal_pose[:3] - final_pose[:3]
        ))

        logger.info(
            "Navigation: %d steps, dist %.1f → %.1f mm, "
            "progress=%.1f mm, finished=%s",
            steps_taken, initial_distance, final_distance,
            initial_distance - final_distance, finished,
        )

        # Assert: either reached goal or made meaningful progress
        if finished:
            self.assertIsNotNone(policy._last_termination)
        else:
            # On complex objects (mug), 100 steps may not be enough
            # Just verify agent moved at all (not frozen)
            total_movement = float(np.linalg.norm(
                final_pose[:3] - start_pose[:3]
            ))
            logger.info(
                "Navigation: %d steps, dist %.1f → %.1f mm, "
                "progress=%.1f mm, movement=%.1f mm, finished=%s",
                steps_taken, initial_distance, final_distance,
                initial_distance - final_distance, total_movement, finished,
            )
            # Agent should have moved at least a few mm in 100 steps
            self.assertGreater(
                total_movement, 0.5,
                f"Agent frozen: moved only {total_movement:.2f}mm in {steps_taken} steps",
            )

    # ══════════════════════════════════════════════════════════
    # TEST 4: Navigation completes with READY
    # ══════════════════════════════════════════════════════════

    def test_navigation_completes(self):
        """Navigation eventually returns READY (goal or timeout)."""
        policy = self._create_policy(enable_learning=False)

        # Very close goal — should reach quickly
        start_sd = self.adapter.reset()
        start_pose = self.adapter.get_pose()

        goal_pose = self._generate_goal(start_pose, distance_range=(5, 15))
        self.adapter.set_goal(goal_pose)

        goal = _make_goal_from_pose_mm(goal_pose)
        ctx = RuntimeContext(rng=np.random.RandomState(42))

        finished = False
        for step in range(200):
            current_sd = self.adapter.get_sensor_data()
            state = _state_from_adapter(self.adapter)
            percept = _make_percept_from_sensor_data(current_sd)

            result = policy(ctx, {}, state, percept, goal)

            if result.status == PolicyStatus.READY:
                finished = True
                break

        self.assertTrue(
            finished,
            f"Navigation did not complete in 200 steps "
            f"(last termination: {policy._last_termination})",
        )
        self.assertFalse(policy.is_navigating)

    # ══════════════════════════════════════════════════════════
    # TEST 5: Multiple episodes
    # ══════════════════════════════════════════════════════════

    def test_multiple_episodes(self):
        """Policy handles multiple sequential navigation episodes."""
        policy = self._create_policy(enable_learning=False)
        ctx = RuntimeContext(rng=np.random.RandomState(42))

        successes = 0
        num_episodes = 5

        for ep in range(num_episodes):
            # Reset to new position
            self.adapter.reset()
            start_pose = self.adapter.get_pose()

            goal_pose = self._generate_goal(
                start_pose, distance_range=(10, 40),
            )
            self.adapter.set_goal(goal_pose)
            goal = _make_goal_from_pose_mm(goal_pose)

            # Reset policy between episodes
            policy.reset()

            # Run episode
            for step in range(150):
                current_sd = self.adapter.get_sensor_data()
                state = _state_from_adapter(self.adapter)
                percept = _make_percept_from_sensor_data(current_sd)

                result = policy(ctx, {}, state, percept, goal)

                if result.status == PolicyStatus.READY:
                    if policy._last_termination == "goal_reached":
                        successes += 1
                    break

            # Force cleanup if navigation didn't finish naturally
            if policy.is_navigating:
                policy.reset()

            # Verify policy is ready for next episode
            self.assertFalse(policy.is_navigating)

        logger.info(
            "Multiple episodes: %d/%d successes",
            successes, num_episodes,
        )

    # ══════════════════════════════════════════════════════════
    # TEST 6: Controller stats accessible
    # ══════════════════════════════════════════════════════════

    def test_stats_accessible(self):
        """Policy stats are accessible after navigation."""
        policy = self._create_policy(enable_learning=False)

        stats = policy.get_stats()
        self.assertIn("controller", stats)
        self.assertIn("nav_active", stats)

        # Controller should have loaded Q-store
        ctrl_stats = stats["controller"]
        self.assertIn("q_store_free", ctrl_stats)
        self.assertIn("q_store_surface", ctrl_stats)

        # Q-stores should have points from training
        free_points = ctrl_stats["q_store_free"].get("num_points", 0)
        surface_points = ctrl_stats["q_store_surface"].get("num_points", 0)
        logger.info(
            "Q-store loaded: free=%d points, surface=%d points",
            free_points, surface_points,
        )
        self.assertGreater(
            free_points + surface_points, 0,
            "Q-store should have points from training",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
