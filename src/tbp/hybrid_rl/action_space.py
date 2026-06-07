import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple, NewType

from scipy.spatial.transform import Rotation as R

AgentID = NewType("AgentID", str)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActionInfo:
    """Metadata about a discrete action for logging and heuristics.
    Attributes:
        index: Action index in Q-table (0 to num_actions-1).
        name: Human-readable name.
        direction_degrees: For surface actions, tangential direction.
            None for non-surface actions.
    """
    index: int
    name: str
    category: str
    opposite_index: Optional[int] = None
    direction_degrees: Optional[float] = None
    rotation_degrees: float = 5.0
    forward_distance: float = 0.05
    left_distance: float = 0.02
    down_distance: float = 0.02


class ActionSpace:
    """Maps discrete action indices to Monty Action objects.

    Action layout (18 actions total):

        SURFACE (8): Crawl along object surface
            0: MoveTangentially 0°   (forward on surface)
            1: MoveTangentially 45°  (forward-right)
            2: MoveTangentially 90°  (right)
            3: MoveTangentially 135° (backward-right)
            4: MoveTangentially 180° (backward on surface)
            5: MoveTangentially 225° (backward-left)
            6: MoveTangentially 270° (left)
            7: MoveTangentially 315° (forward-left)

            8:  MoveForward  (forward, where sensor points)
            9:  MoveForward backward

            10: TurnLeft 
            11: TurnRight
            12: LookUp
            13: LookDown
            14: SetSensorRotation + (clockwise)
            15: SetSensorRotation - (counter-clockwise)
            16: OrientHorizontal
            17: OrientVertical

    Args:
        agent_id: Monty agent identifier for Action objects.
        surface_step: Step size for MoveTangentially (mm).
        free_step: Step size for MoveForward (mm).
        rotation_step: Step size for LookUp/Down/Rotation (degrees).
    """

    NUM_ACTIONS = 18

    # Surface action directions (8 evenly spaced)
    SURFACE_DIRECTIONS = [0, 45, 90, 135, 180, 225, 270, 315]
    # Index ranges
    IDX_SURFACE_START = 0
    IDX_SURFACE_END = 8      # exclusive
    IDX_FREE_FORWARD = 8
    IDX_FREE_BACKWARD = 9
    IDX_TURN_LEFT = 10
    IDX_TURN_RIGHT = 11
    IDX_LOOK_UP = 12
    IDX_LOOK_DOWN = 13
    IDX_ROTATE_POS = 14
    IDX_ROTATE_NEG = 15
    IDX_ORIENT_HOR = 16
    IDX_ORIENT_VERT = 17

    def __init__(
        self,
        agent_id: str,
        surface_step: float = 5.0,
        free_step: float = 10.0,
        rotation_step: float = 10.0,
    ):
        self.agent_id = agent_id
        self.surface_step = surface_step
        self.free_step = free_step
        self.rotation_step = rotation_step

        # Build action info table
        self._action_info = self._build_action_info()

        # Opposite pairs for anti-oscillation detection
        self._opposite_pairs = self._build_opposite_pairs()

        logger.info(
            f"ActionSpace initialized: {self.NUM_ACTIONS} actions, "
            f"surface_step={surface_step}mm, free_step={free_step}mm, "
            f"rotation_step={rotation_step}°"
        )

    def _build_action_info(self) -> List[ActionInfo]:
        """Build metadata table for all actions."""
        info = []

        # Surface actions (0-7)
        for i, deg in enumerate(self.SURFACE_DIRECTIONS):
            opposite_i = (i + 4) % 8  # 180° opposite
            info.append(ActionInfo(
                index=i,
                name="move_tangentially",
                direction_degrees=float(deg),
                opposite_index=opposite_i,
                category="surface",
            ))

        info.append(ActionInfo(
            index=self.IDX_FREE_FORWARD,
            name="free_forward",
            opposite_index=self.IDX_FREE_BACKWARD,
            category="free",
        ))

        info.append(ActionInfo(
            index=self.IDX_FREE_BACKWARD,
            name="free_backward",
            opposite_index=self.IDX_FREE_FORWARD,
            category="free",
        ))

        info.append(ActionInfo(
            index=self.IDX_TURN_LEFT,
            name="turn_left",
            opposite_index=self.IDX_TURN_RIGHT,
            category="orient",
        ))

        info.append(ActionInfo(
            index=self.IDX_TURN_RIGHT,
            name="turn_right",
            opposite_index=self.IDX_TURN_LEFT,
            category="orient",
        ))

        info.append(ActionInfo(
            index=self.IDX_LOOK_UP,
            name="look_up",
            opposite_index=self.IDX_LOOK_DOWN,
            category="orient",
        ))

        info.append(ActionInfo(
            index=self.IDX_LOOK_DOWN,
            name="look_down",
            opposite_index=self.IDX_LOOK_UP,
            category="orient",
        ))

        info.append(ActionInfo(
            index=self.IDX_ROTATE_POS,
            name="rotate_sensor_+",
            opposite_index=self.IDX_ROTATE_NEG,
            category="orient",
        ))

        info.append(ActionInfo(
            index=self.IDX_ROTATE_NEG,
            name="rotate_sensor_-",
            opposite_index=self.IDX_ROTATE_POS,
            category="orient",
        ))

        info.append(ActionInfo(
            index=self.IDX_ORIENT_HOR,
            name="orient_horizontal",
            rotation_degrees=5.0,
            forward_distance=0.05,
            left_distance=0.02,
            category="surface",
        ))

        info.append(ActionInfo(
            index=self.IDX_ORIENT_VERT,
            name="orient_vertical",
            rotation_degrees=5.0,
            forward_distance=0.05,
            down_distance=0.02,
            category="surface",
        ))

        return info

    def get_info(self, action_index: int) -> ActionInfo:
        """Get metadata for an action.

        Args:
            action_index: Action index.

        Returns:
            ActionInfo dataclass with name, category, etc.
        """
        return self._action_info[action_index]
    

    # ══════════════════════════════════════════════════════════
    # HEURISTIC SUPPORT
    # ══════════════════════════════════════════════════════════

    def best_surface_action_toward(
        self,
        target_angle_degrees: float,
    ) -> Tuple[int, float]:
        """Find surface action whose direction is closest to target angle.

        Used by heuristic bias to prefer moving toward the goal
        along the surface.

        Args:
            target_angle_degrees: Desired movement direction in degrees
                (0=forward, 90=right, 180=backward, 270=left).

        Returns:
            Tuple of (action_index, angle_difference_degrees).
        """
        target = target_angle_degrees % 360
        best_idx = 0
        best_diff = 360.0

        for i, direction in enumerate(self.SURFACE_DIRECTIONS):
            diff = abs(target - direction)
            diff = min(diff, 360 - diff)  # wrap around
            if diff < best_diff:
                best_diff = diff
                best_idx = i

        return best_idx, best_diff


    def _build_opposite_pairs(self) -> set:
        """Build set of (a1, a2) pairs that are opposite actions."""
        pairs = set()
        for ai in self._action_info:
            if ai.opposite_index is not None:
                pair = (min(ai.index, ai.opposite_index),
                        max(ai.index, ai.opposite_index))
                pairs.add(pair)
        return pairs
    
    def are_opposite(self, action_a: int, action_b: int) -> bool:
        """Check if two actions are opposites (for anti-oscillation).

        Examples:
            surface_0° and surface_180° → True
            free_forward and free_backward → True
            look_up and look_down → True
            surface_0° and look_up → False

        Args:
            action_a: First action index.
            action_b: Second action index.

        Returns:
            True if actions are opposite.
        """
        if action_a is None or action_b is None:
            return False
        pair = (min(action_a, action_b), max(action_a, action_b))
        return pair in self._opposite_pairs

    def surface_direction_similarity(
        self,
        target_angle_degrees: float,
    ) -> np.ndarray:
        """Cosine similarity between each surface action and target direction.

        Returns values in [-1, +1] for each surface action.
        Non-surface actions get 0.

        Used by heuristic to create smooth bias toward goal direction.

        Args:
            target_angle_degrees: Desired direction in degrees.

        Returns:
            Array [NUM_ACTIONS] with similarity scores.
        """
        similarity = np.zeros(self.NUM_ACTIONS)
        target_rad = np.radians(target_angle_degrees)

        for i, direction in enumerate(self.SURFACE_DIRECTIONS):
            action_rad = np.radians(direction)
            angle_diff = target_rad - action_rad
            similarity[i] = np.cos(angle_diff)

        return similarity

    def get_category_mask(self, category: str) -> np.ndarray:
        """Get boolean mask for actions of a given category.

        Useful for heuristic bias: boost all surface actions,
        suppress all orient actions, etc.

        Args:
            category: One of 'surface', 'free', 'orient', 'meta'.

        Returns:
            Boolean array [NUM_ACTIONS] where True = action belongs
            to category.
        """
        return np.array([
            ai.category == category for ai in self._action_info
        ])

