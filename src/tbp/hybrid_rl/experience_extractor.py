# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""ExperienceExtractor: converts Q-learning discrete trajectories
to P-SAC parameterized format for Behavioral Cloning.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class PSACTransition:
    state: np.ndarray
    action_type: int
    action_params: np.ndarray
    reward: float
    next_state: Optional[np.ndarray] = None
    done: bool = False
    mesh_id: int = 0
    level: int = 0  # curriculum level


class ExperienceExtractor:

    MAX_PARAMS = 3

    # 24-action space (no detach_edge)
    DISCRETE_TO_PSAC = {
        0:  (0, lambda cfg: [0.0, cfg["surface_step"]]),
        1:  (0, lambda cfg: [45.0, cfg["surface_step"]]),
        2:  (0, lambda cfg: [90.0, cfg["surface_step"]]),
        3:  (0, lambda cfg: [135.0, cfg["surface_step"]]),
        4:  (0, lambda cfg: [180.0, cfg["surface_step"]]),
        5:  (0, lambda cfg: [225.0, cfg["surface_step"]]),
        6:  (0, lambda cfg: [270.0, cfg["surface_step"]]),
        7:  (0, lambda cfg: [315.0, cfg["surface_step"]]),
        8:  (1, lambda cfg: [cfg["free_step"]]),
        9:  (1, lambda cfg: [-cfg["free_step"]]),
        10: (2, lambda cfg: [cfg["rotation_step"]]),
        11: (2, lambda cfg: [-cfg["rotation_step"]]),
        12: (3, lambda cfg: [cfg["rotation_step"]]),
        13: (3, lambda cfg: [-cfg["rotation_step"]]),
        14: (4, lambda cfg: [cfg["rotation_step"]]),
        15: (4, lambda cfg: [-cfg["rotation_step"]]),
        16: (5, lambda cfg: [cfg["rotation_step"],
                             cfg.get("orient_left_distance", 0.02),
                             cfg.get("orient_forward_distance", 0.05)]),
        17: (6, lambda cfg: [cfg["rotation_step"],
                             cfg.get("orient_down_distance", 0.02),
                             cfg.get("orient_forward_distance", 0.05)]),
        18: (7, lambda cfg: []),                                     # detach
        19: (1, lambda cfg: [cfg.get("free_step_small", 2.0)]),      # free_forward_small
        20: (3, lambda cfg: [cfg.get("rotation_step_big", 15.0)]),   # look_up_big
        21: (3, lambda cfg: [-cfg.get("rotation_step_big", 15.0)]),  # look_down_big
        22: (2, lambda cfg: [cfg.get("rotation_step_big", 15.0)]),   # turn_left_big
        23: (2, lambda cfg: [-cfg.get("rotation_step_big", 15.0)]),  # turn_right_big
    }

    MESH_NAME_TO_ID = {
        "cube": 0,
        "cylinder": 1,
        "mug": 2,
        "cup": 3,
        "vase": 4,
        "thin_cylinder": 5,
        "flat_square": 6,
        "sphere": 7,
        "cone": 8,
    }

    def __init__(self, config: Dict[str, Any], mesh_name: str = ""):
        self.config = config
        self.mesh_name = mesh_name
        self.mesh_id = self.MESH_NAME_TO_ID.get(mesh_name, -1)

    def convert_action(self, discrete_action: int):
        action_type, params_fn = self.DISCRETE_TO_PSAC[discrete_action]
        raw_params = params_fn(self.config)
        padded_params = np.zeros(self.MAX_PARAMS, dtype=np.float32)
        padded_params[:len(raw_params)] = raw_params
        return action_type, padded_params

    def convert_trajectory(
        self,
        transitions: List[Dict[str, Any]],
    ) -> List[PSACTransition]:
        result = []
        for i, tr in enumerate(transitions):
            action_type, action_params = self.convert_action(tr["action"])
            next_state = (
                transitions[i + 1]["state"]
                if i + 1 < len(transitions)
                else None
            )
            done = (i == len(transitions) - 1)
            result.append(PSACTransition(
                state=np.array(tr["state"], dtype=np.float32),
                action_type=action_type,
                action_params=action_params,
                reward=float(tr["reward"]),
                next_state=(
                    np.array(next_state, dtype=np.float32)
                    if next_state is not None
                    else None
                ),
                done=done,
                mesh_id=self.mesh_id,
            ))
        return result

    def convert_all_trajectories(
        self,
        all_trails: List[List[Dict[str, Any]]],
    ) -> List[PSACTransition]:
        all_transitions = []
        for trail in all_trails:
            all_transitions.extend(self.convert_trajectory(trail))
        return all_transitions

    @staticmethod
    def get_param_dims() -> Dict[int, int]:
        return {
            0: 2,  # MoveTangentially: angle, distance
            1: 1,  # MoveLinear: distance
            2: 1,  # Turn: angle
            3: 1,  # Look: angle
            4: 1,  # SensorRotate: angle
            5: 3,  # OrientHorizontal: rotation, left, forward
            6: 3,  # OrientVertical: rotation, down, forward
            7: 0,  # Detach: no params
        }

    @staticmethod
    def get_type_names() -> Dict[int, str]:
        return {
            0: "MoveTangentially",
            1: "MoveLinear",
            2: "Turn",
            3: "Look",
            4: "SensorRotate",
            5: "OrientHorizontal",
            6: "OrientVertical",
            7: "Detach",
        }
