"""Unit tests for MuJoCoEnvAdapter.

Tests that the adapter correctly:
1. Initializes MuJoCo simulator with YCB object
2. Converts units (mm ↔ m, euler ↔ quaternion)
3. Extracts sensor_data from MuJoCo rendering
4. Translates discrete actions to Monty Actions
5. Maintains state synchronization
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.spatial.transform import Rotation as R

# Test unit conversion without MuJoCo dependency
from tbp.hybrid_rl.mujoco_env_adapter import MuJoCoEnvAdapter, MM_PER_M


class TestUnitConversion(unittest.TestCase):
    """Test unit conversion methods (no MuJoCo needed)."""

    def test_mm_to_m(self):
        pos_mm = np.array([100.0, 200.0, 300.0])
        pos_m = MuJoCoEnvAdapter._mm_to_m(pos_mm)
        np.testing.assert_allclose(pos_m, (0.1, 0.2, 0.3))

    def test_m_to_mm(self):
        pos_m = (0.1, 0.2, 0.3)
        pos_mm = MuJoCoEnvAdapter._m_to_mm(pos_m)
        np.testing.assert_allclose(pos_mm, [100.0, 200.0, 300.0])

    def test_mm_m_roundtrip(self):
        original = np.array([42.5, -17.3, 100.0])
        result = MuJoCoEnvAdapter._m_to_mm(MuJoCoEnvAdapter._mm_to_m(original))
        np.testing.assert_allclose(result, original, atol=1e-10)

    def test_euler_to_quat_identity(self):
        euler = np.array([0.0, 0.0, 0.0])
        quat = MuJoCoEnvAdapter._euler_to_quat_wxyz(euler)
        # Identity quaternion: (1, 0, 0, 0)
        np.testing.assert_allclose(quat, (1.0, 0.0, 0.0, 0.0), atol=1e-10)

    def test_quat_to_euler_identity(self):
        quat = (1.0, 0.0, 0.0, 0.0)
        euler = MuJoCoEnvAdapter._quat_wxyz_to_euler(quat)
        np.testing.assert_allclose(euler, [0.0, 0.0, 0.0], atol=1e-10)

    def test_euler_quat_roundtrip(self):
        euler_original = np.array([30.0, -45.0, 60.0])
        quat = MuJoCoEnvAdapter._euler_to_quat_wxyz(euler_original)
        euler_back = MuJoCoEnvAdapter._quat_wxyz_to_euler(quat)
        np.testing.assert_allclose(euler_back, euler_original, atol=1e-6)

    def test_euler_quat_roundtrip_extreme(self):
        """Test near-gimbal-lock angles."""
        euler_original = np.array([89.0, 0.0, 0.0])
        quat = MuJoCoEnvAdapter._euler_to_quat_wxyz(euler_original)
        euler_back = MuJoCoEnvAdapter._quat_wxyz_to_euler(quat)
        np.testing.assert_allclose(euler_back, euler_original, atol=1e-4)

    def test_normalize_euler(self):
        angles = np.array([200.0, -200.0, 370.0])
        normalized = MuJoCoEnvAdapter._normalize_euler(angles)
        for a in normalized:
            self.assertGreaterEqual(a, -180.0)
            self.assertLess(a, 180.0)

    def test_normalize_euler_identity(self):
        angles = np.array([45.0, -90.0, 0.0])
        normalized = MuJoCoEnvAdapter._normalize_euler(angles)
        np.testing.assert_allclose(normalized, angles)


class TestEdgeDetection(unittest.TestCase):
    """Test edge traversal detection."""

    def test_no_edge_same_normal(self):
        adapter = object.__new__(MuJoCoEnvAdapter)
        result = adapter._detect_edge_traversal([0, 0, 1], [0, 0, 1])
        self.assertFalse(result)

    def test_edge_perpendicular_normals(self):
        adapter = object.__new__(MuJoCoEnvAdapter)
        result = adapter._detect_edge_traversal([0, 0, 1], [1, 0, 0])
        self.assertTrue(result)

    def test_no_edge_small_angle(self):
        adapter = object.__new__(MuJoCoEnvAdapter)
        # ~20 degrees apart
        n1 = [0, 0, 1]
        n2 = [0, np.sin(np.radians(20)), np.cos(np.radians(20))]
        result = adapter._detect_edge_traversal(n1, n2)
        self.assertFalse(result)

    def test_edge_with_none(self):
        adapter = object.__new__(MuJoCoEnvAdapter)
        self.assertFalse(adapter._detect_edge_traversal(None, [0, 0, 1]))
        self.assertFalse(adapter._detect_edge_traversal([0, 0, 1], None))
        self.assertFalse(adapter._detect_edge_traversal(None, None))


class TestLookAtDirection(unittest.TestCase):
    """Test _look_at_direction utility."""

    def test_look_forward(self):
        adapter = object.__new__(MuJoCoEnvAdapter)
        # Looking along -Z should give zero rotation
        euler = adapter._look_at_direction([0, 0, -1])
        rot = R.from_euler("xyz", euler, degrees=True)
        forward = rot.apply([0, 0, -1])
        np.testing.assert_allclose(forward, [0, 0, -1], atol=1e-6)

    def test_look_right(self):
        adapter = object.__new__(MuJoCoEnvAdapter)
        euler = adapter._look_at_direction([1, 0, 0])
        rot = R.from_euler("xyz", euler, degrees=True)
        forward = rot.apply([0, 0, -1])
        np.testing.assert_allclose(forward, [1, 0, 0], atol=1e-6)

    def test_look_up(self):
        adapter = object.__new__(MuJoCoEnvAdapter)
        euler = adapter._look_at_direction([0, 1, 0])
        rot = R.from_euler("xyz", euler, degrees=True)
        forward = rot.apply([0, 0, -1])
        np.testing.assert_allclose(forward, [0, 1, 0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
