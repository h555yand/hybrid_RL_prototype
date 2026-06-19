"""
ExperienceExtractor: converts Q-learning 18D discrete trajectories
to P-SAC 8-type parameterized format for Behavioral Cloning.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from .rl_goal_approach_controller import RLGoalApproachController
from .lightweight_env import LightweightEnv


@dataclass
class PSACTransition:
    state: np.ndarray
    action_type: int
    action_params: np.ndarray
    reward: float
    next_state: Optional[np.ndarray] = None
    done: bool = False


class ExperienceExtractor:

    MAX_PARAMS = 3

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
        16: (5, lambda cfg: [cfg["rotation_step"], cfg.get("orient_left_distance", 0.02), cfg.get("orient_forward_distance", 0.05)]),
        17: (6, lambda cfg: [cfg["rotation_step"], cfg.get("orient_down_distance", 0.02), cfg.get("orient_forward_distance", 0.05)]),
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config

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
            next_state = transitions[i + 1]["state"] if i + 1 < len(transitions) else None
            done = (i == len(transitions) - 1)
            result.append(PSACTransition(
                state=np.array(tr["state"], dtype=np.float32),
                action_type=action_type,
                action_params=action_params,
                reward=float(tr["reward"]),
                next_state=np.array(next_state, dtype=np.float32) if next_state is not None else None,
                done=done,
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
            0: 2,
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: 3,
            6: 3,
            7: 0,
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
            7: "NoOp",
        }
