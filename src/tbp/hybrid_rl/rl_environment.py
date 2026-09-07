# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""RLEnvironment Protocol — unified interface for all RL environments.

All environments (trimesh, MuJoCo, Habitat, robot) implement this protocol.
The RL controller and adaptive loop work identically regardless of backend.

Key invariant: state computation is frame-independent.
All spatial data (poses, normals, positions) must be in a single
consistent coordinate frame per environment. Which frame doesn't matter —
only internal consistency.

Coordinate conventions:
    - Positions in mm
    - Rotations in degrees (euler xyz)
    - Normals as unit vectors in world frame

Sensor data contract (required keys):
    point_normal:       Optional[List[float]]  — surface normal at gaze point
    k1, k2:             float                  — principal curvatures
    on_object:          bool                   — sensor sees surface
    depth:              float                  — depth to surface (mm)
    passed_through:     bool                   — passed through mesh
    goal_normal:        Optional[List[float]]  — goal surface normal
    detach_had_collision: bool                 — detach hit obstacle
    detach_sub_steps:   int                    — detach sub-step count
    path_blocked:       bool                   — direct path to goal blocked
    up_direction:       List[float]            — object up direction
    object_center:      List[float]            — object centroid
    same_side:          bool                   — agent and goal on same side
    object_extents:     List[float]            — object bounding box extents
    edge_traversed:     bool                   — crossed surface edge this step
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class RLEnvironment(Protocol):
    """Unified environment interface for RL goal approach.

    Implementations:
        LightweightEnv      — trimesh, fast, supports offline retrain
        MuJoCoEnvAdapter    — MuJoCo physics + rendering
        (future) HabitatEnvAdapter
        (future) RobotEnvAdapter
    """

    # ═══════════════════════════════════════════════════
    # Core: state observation
    # ═══════════════════════════════════════════════════

    def reset(self, **kwargs: Any) -> dict:
        """Reset environment to initial state.

        Optional kwargs:
            position: np.ndarray — start position (env frame, mm)
            rotation: np.ndarray — start rotation (env frame, degrees)

        Returns:
            Sensor data dict.
        """
        ...

    def get_pose(self) -> np.ndarray:
        """Current agent pose [x, y, z, rx, ry, rz].

        Position in mm, rotation in degrees (euler xyz).
        All in environment's consistent coordinate frame.
        """
        ...

    def get_sensor_data(self) -> dict:
        """Current sensor readings.

        Returns dict with all keys from sensor data contract.
        All spatial vectors (normals, directions, positions)
        in environment's coordinate frame.
        """
        ...

    # ═══════════════════════════════════════════════════
    # Core: goal management
    # ═══════════════════════════════════════════════════

    def set_goal(self, goal_pose: np.ndarray) -> None:
        """Set navigation goal.

        Args:
            goal_pose: [x, y, z, rx, ry, rz] in env frame.
        """
        ...

    def get_random_surface_point(self, **kwargs: Any) -> np.ndarray:
        """Sample random surface point for goal generation.

        Optional kwargs:
            reference_pos: np.ndarray — reference position for distance filter
            min_dist: float           — minimum distance from reference (mm)
            max_dist: float           — maximum distance from reference (mm)
            max_attempts: int         — rejection sampling limit
            mesh_sample: bool         — batch sample mode

        Returns:
            Pose [x, y, z, rx, ry, rz] in env frame.
        """
        ...

    # ═══════════════════════════════════════════════════
    # Core: action execution
    # ═══════════════════════════════════════════════════

    def step_discrete(
        self, action_idx: int, action_space: Any
    ) -> dict:
        """Execute discrete action (from Q-store / heuristic).

        Args:
            action_idx: Discrete action index (0-23).
            action_space: ActionSpace instance with step sizes.

        Returns:
            Sensor data after action.
        """
        ...

    def step_continuous(
        self, action_type: int, action_params: np.ndarray
    ) -> dict:
        """Execute continuous action (from SAC / arbitrator).

        Action types and param semantics (same as ExperienceExtractor):
            0 — tangential: [sin_angle, cos_angle, distance]
            1 — linear:     [distance, -, -]
            2 — yaw:        [sin_angle, cos_angle, -]
            3 — pitch:      [sin_angle, cos_angle, -]
            4 — roll:       [sin_angle, cos_angle, -]
            5 — orient_h:   [rotation_deg, left_dist, fwd_dist]
            6 — orient_v:   [rotation_deg, down_dist, fwd_dist]
            7 — detach:     [-, -, -]  (params ignored)

        Args:
            action_type: PSAC action type (0-7).
            action_params: Continuous parameters [3].

        Returns:
            Sensor data after action.
        """
        ...

    # ═══════════════════════════════════════════════════
    # Capabilities (used by adaptive loop)
    # ═══════════════════════════════════════════════════

    @property
    def supports_continuous(self) -> bool:
        """Whether step_continuous() is implemented.

        If False, adaptive loop falls back to
        sac_to_discrete + step_discrete.
        """
        ...

    @property
    def supports_offline_retrain(self) -> bool:
        """Whether env is fast enough for offline retrain.

        True for trimesh (~1000 episodes/sec).
        False for MuJoCo, Habitat, robot.
        Adaptive manager uses this to choose retrain env.
        """
        ...

    # ═══════════════════════════════════════════════════
    # Optional: visualization
    # ═══════════════════════════════════════════════════

    def render_episode_frame(
        self,
        agent_pose: np.ndarray,
        goal_pose: np.ndarray,
        filepath: str,
        trail_poses: Optional[List[np.ndarray]] = None,
        text: str = "",
        step_num: int = 0,
        distance: float = 0.0,
        result: str = "",
    ) -> None:
        """Render episode frame for visualization.

        Optional — environments without rendering silently skip.
        """
        ...

    # ═══════════════════════════════════════════════════
    # Optional: lifecycle
    # ═══════════════════════════════════════════════════

    def close(self) -> None:
        """Release resources (renderers, simulators, connections)."""
        ...