"""
Twin Critic (Clipped Double Q) for P-SAC.
Input: state + action_type_onehot + action_params → Q-value.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class QNetwork(nn.Module):

    def __init__(self, state_dim: int = 15, num_types: int = 8, max_params: int = 3):
        super().__init__()
        input_dim = state_dim + num_types + max_params

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        action_type: torch.Tensor,
        action_params: torch.Tensor,
        num_types: int = 8,
    ) -> torch.Tensor:
        type_onehot = F.one_hot(action_type.long(), num_classes=num_types).float()
        x = torch.cat([state, type_onehot, action_params], dim=-1)
        return self.net(x).squeeze(-1)


class TwinCritic(nn.Module):

    def __init__(self, state_dim: int = 15, num_types: int = 8, max_params: int = 3):
        super().__init__()
        self.num_types = num_types
        self.q1 = QNetwork(state_dim, num_types, max_params)
        self.q2 = QNetwork(state_dim, num_types, max_params)

    def forward(
        self,
        state: torch.Tensor,
        action_type: torch.Tensor,
        action_params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q1_val = self.q1(state, action_type, action_params, self.num_types)
        q2_val = self.q2(state, action_type, action_params, self.num_types)
        return q1_val, q2_val

    def min_q(
        self,
        state: torch.Tensor,
        action_type: torch.Tensor,
        action_params: torch.Tensor,
    ) -> torch.Tensor:
        q1_val, q2_val = self.forward(state, action_type, action_params)
        return torch.min(q1_val, q2_val)
