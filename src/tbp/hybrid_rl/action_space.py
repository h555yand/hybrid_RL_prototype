import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple, NewType

from scipy.spatial.transform import Rotation as R

AgentID = NewType("AgentID", str)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# ACTION CATEGORIES
# ══════════════════════════════════════════════════════════════

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

        logger.info(
            f"ActionSpace initialized: {self.NUM_ACTIONS} actions, "
            f"surface_step={surface_step}mm, free_step={free_step}mm, "
            f"rotation_step={rotation_step}°"
        )

    # ══════════════════════════════════════════════════════════
    # ACTION INFO
    # ══════════════════════════════════════════════════════════

    def _build_action_info(self) -> List[ActionInfo]:
        """Build metadata table for all actions."""
        info = []

        # Surface actions (0-7)
        for i, deg in enumerate(self.SURFACE_DIRECTIONS):
            info.append(ActionInfo(
                index=i,
                name="move_tangentially",
                direction_degrees=float(deg),
            ))

        info.append(ActionInfo(
            index=self.IDX_FREE_FORWARD,
            name="free_forward",
        ))

        info.append(ActionInfo(
            index=self.IDX_FREE_BACKWARD,
            name="free_backward",
        ))

        info.append(ActionInfo(
            index=self.IDX_TURN_LEFT,
            name="turn_left",
        ))

        info.append(ActionInfo(
            index=self.IDX_TURN_RIGHT,
            name="turn_right",
        ))

        info.append(ActionInfo(
            index=self.IDX_LOOK_UP,
            name="look_up",
        ))

        info.append(ActionInfo(
            index=self.IDX_LOOK_DOWN,
            name="look_down",
        ))

        info.append(ActionInfo(
            index=self.IDX_ROTATE_POS,
            name="rotate_sensor_+",
        ))

        info.append(ActionInfo(
            index=self.IDX_ROTATE_NEG,
            name="rotate_sensor_-",
        ))

        info.append(ActionInfo(
            index=self.IDX_ORIENT_HOR,
            name="orient_horizontal",
            rotation_degrees=5.0,
            forward_distance=0.05,
            left_distance=0.02,
        ))

        info.append(ActionInfo(
            index=self.IDX_ORIENT_VERT,
            name="orient_vertical",
            rotation_degrees=5.0,
            forward_distance=0.05,
            down_distance=0.02,
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
