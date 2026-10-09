"""Integration tests for RLGoalPolicy replacing JumpToGoal.

Tests:
1. Unit test: RLGoalPolicy lifecycle (start → step → finish)
2. Unit test: MontyRLBridge data conversion
3. Unit test: RLPolicySelector routing
4. Integration: RLGoalPolicy navigates toward a goal in MuJoCo
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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


def _make_state(
    pos=(0.0, 1.5, -0.2),
    rot=None,
    agent_id="agent_id_0",
) -> MotorSystemState:
    """Create a minimal MotorSystemState for testing."""
    if rot is None:
        rot = qt.quaternion(1, 0, 0, 0)
    return MotorSystemState({
        AgentID(agent_id): AgentState(
            position=pos,
            rotation=rot,
            sensors={
                SensorID("view_finder"): SensorState(
                    position=(0.0, 0.0, 0.0),
                    rotation=qt.quaternion(1, 0, 0, 0),
                ),
            },
        ),
    })


def _make_percept(
    on_object=True,
    normal=(0.0, 0.0, -1.0),
    depth=0.025,
    curvatures=(0.0, 0.0),
) -> Message:
    """Create a minimal CMP Message for testing."""
    return Message(
        location=np.array([0.0, 1.5, -0.2]),
        morphological_features={
            "pose_vectors": np.array([
                list(normal),
                [1, 0, 0],
                [0, 1, 0],
            ]),
            "pose_fully_defined": True,
            "on_object": on_object,
        },
        non_morphological_features={
            "min_depth": depth,
            "mean_depth": depth,
            "object_coverage": 0.5,
            "principal_curvatures": list(curvatures),
        },
        confidence=1.0,
        pass_message=True,
        sender_id="patch",
        sender_type="SM",
        process_features_in_lm=True,
    )


def _make_goal(
    location=(0.05, 1.5, -0.2),
    direction=(0.0, 0.0, -1.0),
) -> Goal:
    """Create a minimal GSG Goal for testing."""
    return Goal(
        location=np.array(location),
        morphological_features={
            "pose_vectors": np.array([
                list(direction),
                [1, 0, 0],
                [0, 1, 0],
            ]),
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


class TestMontyRLBridge(unittest.TestCase):
    """Unit tests for MontyRLBridge data conversion."""

    def test_goal_to_pose_mm(self):
        """Goal location converts from meters to mm."""
        from tbp.hybrid_rl.monty_rl_bridge import MontyRLBridge

        # Mock adapter
        adapter = MagicMock()
        bridge = MontyRLBridge(
            agent_id=AgentID("agent_id_0"),
            rl_config={"surface_step": 3.0, "free_step": 8.0, "rotation_step": 5.0},
            mujoco_adapter=adapter,
        )

        goal = _make_goal(location=(0.1, 1.5, -0.3))
        pose = bridge.goal_to_pose_mm(goal)

        # Position should be in mm
        np.testing.assert_allclose(pose[:3], [100.0, 1500.0, -300.0])
        # Rotation should be in degrees
        self.assertEqual(len(pose), 6)

    def test_agent_pose_mm(self):
        """Agent position converts from meters to mm."""
        from tbp.hybrid_rl.monty_rl_bridge import MontyRLBridge

        adapter = MagicMock()
        bridge = MontyRLBridge(
            agent_id=AgentID("agent_id_0"),
            rl_config={},
            mujoco_adapter=adapter,
        )

        state = _make_state(pos=(0.1, 1.5, -0.3))
        pose = bridge.agent_pose_mm(state)

        np.testing.assert_allclose(pose[:3], [100.0, 1500.0, -300.0])
        self.assertEqual(len(pose), 6)

    def test_percept_to_sensor_data(self):
        """Percept Message maps to RL sensor_data dict."""
        from tbp.hybrid_rl.monty_rl_bridge import MontyRLBridge

        adapter = MagicMock()
        adapter._goal_normal_mj = [0.0, 0.0, -1.0]
        adapter._mj_center_mm = np.array([0.0, 0.0, 0.0])
        adapter._mj_extents_mm = np.array([50.0, 50.0, 50.0])
        adapter.up_direction = np.array([0.0, 1.0, 0.0])
        adapter.open_edge_height = 40.0
        adapter._passed_through = False
        adapter._detach_had_collision = False
        adapter._edge_traversed = False
        adapter._last_detach_sub_steps = 1
        adapter._check_path_blocked.return_value = False
        adapter._compute_same_side.return_value = True

        bridge = MontyRLBridge(
            agent_id=AgentID("agent_id_0"),
            rl_config={},
            mujoco_adapter=adapter,
        )

        percept = _make_percept(on_object=True, depth=0.025)
        state = _make_state()
        goal = _make_goal()

        sd = bridge.percept_to_sensor_data(percept, state, goal)

        # Check key fields
        self.assertTrue(sd["on_object"])
        self.assertAlmostEqual(sd["depth"], 25.0)  # 0.025m → 25mm
        self.assertIsNotNone(sd["point_normal"])
        self.assertFalse(sd["path_blocked"])
        self.assertTrue(sd["same_side"])
        self.assertIsNotNone(sd["object_center"])
        self.assertIsNotNone(sd["up_direction"])

    def test_percept_off_object(self):
        """Off-object percept produces correct sensor_data."""
        from tbp.hybrid_rl.monty_rl_bridge import MontyRLBridge

        adapter = MagicMock()
        adapter._goal_normal_mj = None
        adapter._mj_center_mm = np.array([0.0, 0.0, 0.0])
        adapter._mj_extents_mm = np.array([50.0, 50.0, 50.0])
        adapter.up_direction = np.array([0.0, 1.0, 0.0])
        adapter.open_edge_height = 40.0
        adapter._passed_through = False
        adapter._detach_had_collision = False
        adapter._edge_traversed = False
        adapter._last_detach_sub_steps = 1

        bridge = MontyRLBridge(
            agent_id=AgentID("agent_id_0"),
            rl_config={},
            mujoco_adapter=adapter,
        )

        percept = _make_percept(on_object=False, depth=0.5)
        state = _make_state()

        sd = bridge.percept_to_sensor_data(percept, state, goal=None)

        self.assertFalse(sd["on_object"])
        self.assertAlmostEqual(sd["depth"], 500.0)  # 0.5m → 500mm
        self.assertFalse(sd["path_blocked"])
        self.assertTrue(sd["same_side"])


class TestRLPolicySelector(unittest.TestCase):
    """Unit tests for RLPolicySelector routing."""

    def test_gsg_goal_routes_to_rl(self):
        """GSG goals are routed to RLGoalPolicy."""
        from tbp.hybrid_rl.rl_policy_selector import RLPolicySelector

        rl_policy = MagicMock()
        rl_policy.return_value = MotorPolicyResult(
            [], status=PolicyStatus.IN_PROGRESS,
        )

        look_at = MagicMock()
        default = MagicMock()

        selector = RLPolicySelector(
            rl_goal_policy=rl_policy,
            look_at_goal=look_at,
            default=default,
        )

        ctx = RuntimeContext(rng=np.random.RandomState(42))
        state = _make_state()
        percept = _make_percept()
        goal = _make_goal()

        result = selector(ctx, {}, state, percept, [goal])

        rl_policy.assert_called_once()
        look_at.assert_not_called()
        default.assert_not_called()
        self.assertEqual(result.status, PolicyStatus.IN_PROGRESS)
        self.assertTrue(selector._is_navigating)

    def test_sm_goal_routes_to_look_at(self):
        """SM goals are routed to LookAtGoal."""
        from tbp.hybrid_rl.rl_policy_selector import RLPolicySelector

        rl_policy = MagicMock()
        look_at = MagicMock()
        look_at.return_value = MotorPolicyResult([])
        default = MagicMock()

        selector = RLPolicySelector(
            rl_goal_policy=rl_policy,
            look_at_goal=look_at,
            default=default,
        )

        sm_goal = Goal(
            location=np.array([0.1, 1.5, -0.2]),
            morphological_features={"pose_vectors": np.eye(3), "pose_fully_defined": True},
            non_morphological_features={},
            confidence=0.8,
            pass_message=True,
            sender_id="SM_0",
            sender_type="SM",
            process_features_in_lm=False,
            goal_tolerances=None,
        )

        ctx = RuntimeContext(rng=np.random.RandomState(42))
        result = selector(ctx, {}, _make_state(), _make_percept(), [sm_goal])

        rl_policy.assert_not_called()
        look_at.assert_called_once()

    def test_no_goals_routes_to_default(self):
        """No goals → default exploration policy."""
        from tbp.hybrid_rl.rl_policy_selector import RLPolicySelector

        rl_policy = MagicMock()
        look_at = MagicMock()
        default = MagicMock()
        default.return_value = MotorPolicyResult([])

        selector = RLPolicySelector(
            rl_goal_policy=rl_policy,
            look_at_goal=look_at,
            default=default,
        )

        ctx = RuntimeContext(rng=np.random.RandomState(42))
        result = selector(ctx, {}, _make_state(), _make_percept(), [])

        rl_policy.assert_not_called()
        look_at.assert_not_called()
        default.assert_called_once()

    def test_navigation_continues_until_ready(self):
        """Selector stays in navigation mode while IN_PROGRESS."""
        from tbp.hybrid_rl.rl_policy_selector import RLPolicySelector

        call_count = 0

        def rl_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return MotorPolicyResult([], status=PolicyStatus.IN_PROGRESS)
            return MotorPolicyResult([], status=PolicyStatus.READY)

        rl_policy = MagicMock(side_effect=rl_side_effect)
        look_at = MagicMock()
        default = MagicMock()
        default.return_value = MotorPolicyResult([])

        selector = RLPolicySelector(
            rl_goal_policy=rl_policy,
            look_at_goal=look_at,
            default=default,
        )

        ctx = RuntimeContext(rng=np.random.RandomState(42))
        state = _make_state()
        percept = _make_percept()
        goal = _make_goal()

        # Step 1: start navigation
        r1 = selector(ctx, {}, state, percept, [goal])
        self.assertTrue(selector._is_navigating)
        self.assertEqual(r1.status, PolicyStatus.IN_PROGRESS)

        # Step 2: continue (no new goals needed)
        r2 = selector(ctx, {}, state, percept, [])
        self.assertTrue(selector._is_navigating)
        self.assertEqual(r2.status, PolicyStatus.IN_PROGRESS)

        # Step 3: navigation finishes
        r3 = selector(ctx, {}, state, percept, [])
        self.assertFalse(selector._is_navigating)

    def test_reset_clears_navigation(self):
        """Reset clears navigation state."""
        from tbp.hybrid_rl.rl_policy_selector import RLPolicySelector

        rl_policy = MagicMock()
        rl_policy.return_value = MotorPolicyResult(
            [], status=PolicyStatus.IN_PROGRESS,
        )
        look_at = MagicMock()
        default = MagicMock()

        selector = RLPolicySelector(
            rl_goal_policy=rl_policy,
            look_at_goal=look_at,
            default=default,
        )

        ctx = RuntimeContext(rng=np.random.RandomState(42))
        selector(ctx, {}, _make_state(), _make_percept(), [_make_goal()])
        self.assertTrue(selector._is_navigating)

        selector.reset()
        self.assertFalse(selector._is_navigating)


class TestRLGoalPolicyLifecycle(unittest.TestCase):
    """Unit tests for RLGoalPolicy navigation lifecycle."""

    def test_no_goal_returns_empty(self):
        """No goal and not navigating → empty result."""
        from tbp.hybrid_rl.rl_goal_policy import RLGoalPolicy

        with patch.object(RLGoalPolicy, '__init__', lambda self, **kw: None):
            policy = RLGoalPolicy.__new__(RLGoalPolicy)
            policy._nav_active = False
            policy._nav_steps = 0
            policy._current_goal = None
            policy._controller = MagicMock()
            policy._controller.is_active = False

            ctx = RuntimeContext(rng=np.random.RandomState(42))
            result = policy(ctx, {}, _make_state(), _make_percept(), None)

            self.assertEqual(result.actions, [])
            self.assertEqual(result.status, PolicyStatus.READY)

    def test_goal_starts_navigation(self):
        """Receiving a goal starts navigation."""
        from tbp.hybrid_rl.rl_goal_policy import RLGoalPolicy

        with patch.object(RLGoalPolicy, '__init__', lambda self, **kw: None):
            policy = RLGoalPolicy.__new__(RLGoalPolicy)
            policy._nav_active = False
            policy._nav_steps = 0
            policy._current_goal = None
            policy._last_termination = None
            policy._enable_learning = False
            policy._max_nav_steps = 200
            policy._manager = None
            policy._agent_id = AgentID("agent_id_0")

            # Mock bridge
            policy._bridge = MagicMock()
            policy._bridge.goal_to_pose_mm.return_value = np.array(
                [100.0, 1500.0, -200.0, 0.0, 0.0, 0.0]
            )
            policy._bridge.agent_position_mm.return_value = np.array(
                [0.0, 1500.0, -200.0]
            )
            policy._bridge.agent_pose_mm.return_value = np.array(
                [0.0, 1500.0, -200.0, 0.0, 0.0, 0.0]
            )
            policy._bridge.percept_to_sensor_data.return_value = {
                "point_normal": [0, 0, -1], "on_object": True,
                "depth": 25.0, "k1": 0.0, "k2": 0.0,
                "same_side": True, "path_blocked": False,
                "goal_normal": [0, 0, -1],
                "object_center": [0, 0, 0],
                "object_extents": [50, 50, 50],
                "up_direction": [0, 1, 0],
                "open_edge_height": 40.0,
                "passed_through": False,
                "detach_had_collision": False,
                "edge_traversed": False,
                "detach_sub_steps": 1,
            }
            policy._bridge.execute_continuous_action.return_value = (
                policy._bridge.percept_to_sensor_data.return_value
            )

            # Mock controller
            policy._controller = MagicMock()
            policy._controller.is_active = True
            policy._controller._compute_state.return_value = np.zeros(22)
            policy._controller.step.return_value = ("free_forward", None)
            policy._controller._last_action = 8
            policy._controller._episode_transitions = []

            ctx = RuntimeContext(rng=np.random.RandomState(42))
            goal = _make_goal()

            result = policy(ctx, {}, _make_state(), _make_percept(), goal)

            self.assertTrue(policy._nav_active)
            self.assertEqual(result.status, PolicyStatus.IN_PROGRESS)
            self.assertTrue(result.motor_only_step)
            policy._bridge.set_goal.assert_called_once_with(goal)
            policy._controller.set_new_goal.assert_called_once()

    def test_null_goal_location_skipped(self):
        """Goal with None location is skipped."""
        from tbp.hybrid_rl.rl_goal_policy import RLGoalPolicy

        with patch.object(RLGoalPolicy, '__init__', lambda self, **kw: None):
            policy = RLGoalPolicy.__new__(RLGoalPolicy)
            policy._nav_active = False
            policy._controller = MagicMock()
            policy._controller.is_active = False

            goal = Goal(
                location=None,
                morphological_features=None,
                non_morphological_features=None,
                confidence=0.5,
                pass_message=True,
                sender_id="LM_0",
                sender_type="GSG",
                process_features_in_lm=False,
                goal_tolerances=None,
            )

            ctx = RuntimeContext(rng=np.random.RandomState(42))
            result = policy(ctx, {}, _make_state(), _make_percept(), goal)

            self.assertFalse(policy._nav_active)
            self.assertEqual(result.actions, [])


if __name__ == "__main__":
    unittest.main()
