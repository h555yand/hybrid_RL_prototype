"""
ActionInterpreter: converts P-SAC (type, params) to LightweightEnv commands.
"""

import numpy as np
from typing import Tuple
from .experience_extractor import ExperienceExtractor


class ActionInterpreter:

    def __init__(self, env):
        self.env = env
        self.type_names = ExperienceExtractor.get_type_names()
        self.param_dims = ExperienceExtractor.get_param_dims()

    def execute(self, action_type: int, action_params: np.ndarray) -> dict:
        if action_type == 0:
            angle_deg = float(action_params[0])
            distance = float(np.clip(action_params[1], 0.5, 15.0))
            self.env._move_tangentially(angle_deg, distance)

        elif action_type == 1:
            distance = float(np.clip(action_params[0], -25.0, 25.0))
            self.env._move_forward(distance)

        elif action_type == 2:
            rotation = float(np.clip(action_params[0], -45.0, 45.0))
            self.env.agent_rot[1] += rotation

        elif action_type == 3:
            rotation = float(np.clip(action_params[0], -45.0, 45.0))
            self.env.agent_rot[0] += rotation

        elif action_type == 4:
            rotation = float(np.clip(action_params[0], -45.0, 45.0))
            self.env.agent_rot[2] += rotation

        elif action_type == 5:
            rotation = float(action_params[0])
            left_dist = float(action_params[1])
            fwd_dist = float(action_params[2])
            self.env._orient_horizontal(rotation, fwd_dist, left_dist)

        elif action_type == 6:
            rotation = float(action_params[0])
            down_dist = float(action_params[1])
            fwd_dist = float(action_params[2])
            self.env._orient_vertical(rotation, fwd_dist, down_dist)

        elif action_type == 7:
            if hasattr(self.env, '_current_goal') and self.env._current_goal is not None:
                self.env._detach_and_fly_to_goal(
                    goal_pose=self.env._current_goal,
                )

        return self.env.get_sensor_data()
