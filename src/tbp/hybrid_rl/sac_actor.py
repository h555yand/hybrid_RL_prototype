"""SAC Actor Network for Parameterized Action Space.
Stochastic policy: Categorical(types) + Squashed Gaussian(params|type).
Warm-started from BC Actor weights.
"""

from typing import Dict, Tuple
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .experience_extractor import ExperienceExtractor

logger = logging.getLogger(__name__)


# Action parameter bounds (in real/denormalized space)
# Format: {type_id: [(low, high), ...] per param}
ACTION_PARAM_BOUNDS = {
    0: [(-1.0, 1.0), (-1.0, 1.0), (0.5, 15.0)],   # MoveTang: sin, cos, step
    1: [(-25.0, 25.0)],                              # MoveLinear: dist
    2: [(-1.0, 1.0), (-1.0, 1.0)],                   # Turn: sin, cos (of [-45,45])
    3: [(-1.0, 1.0), (-1.0, 1.0)],                   # Look: sin, cos (of [-45,45])
    4: [(-1.0, 1.0), (-1.0, 1.0)],                   # SensorRotate: sin, cos
    5: [(-45.0, 45.0), (-5.0, 5.0), (-5.0, 5.0)],    # OrientHoriz: rot, left, fwd
    6: [(-45.0, 45.0), (-5.0, 5.0), (-5.0, 5.0)],    # OrientVert: rot, down, fwd
    7: [],                                             # Detach: no params
}


def _compute_scales():
    """Compute center and scale for each action type's params.

    tanh(x) ∈ [-1, 1]
    action = tanh(x) * scale + center
    → action ∈ [center - scale, center + scale] = [low, high]
    """
    centers = {}
    scales = {}
    for type_id, bounds in ACTION_PARAM_BOUNDS.items():
        if not bounds:
            continue
        c = []
        s = []
        for low, high in bounds:
            c.append((high + low) / 2.0)
            s.append((high - low) / 2.0)
        centers[type_id] = torch.FloatTensor(c)
        scales[type_id] = torch.FloatTensor(s)
    return centers, scales


class SACActorNetwork(nn.Module):

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(
        self, state_dim: int = 15, num_types: int = 8
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_types = num_types
        self.param_dims = (
            ExperienceExtractor.get_param_dims()
        )

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.type_head = nn.Linear(128, num_types)

        self.param_mu_heads = nn.ModuleDict()
        self.param_log_std_heads = nn.ModuleDict()
        for type_id in range(num_types):
            dim = self.param_dims[type_id]
            if dim > 0:
                self.param_mu_heads[
                    str(type_id)
                ] = nn.Linear(128, dim)
                self.param_log_std_heads[
                    str(type_id)
                ] = nn.Linear(128, dim)

        # Precompute action bounds
        self._centers, self._scales = (
            _compute_scales()
        )

    def _get_scale_center(self, type_id: int):
        """Get scale and center tensors for a type."""
        if type_id in self._scales:
            return (
                self._scales[type_id],
                self._centers[type_id],
            )
        return None, None

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        Dict[int, torch.Tensor],
        Dict[int, torch.Tensor],
    ]:
        x = self.encoder(state)
        type_logits = self.type_head(x)

        param_mus = {}
        param_log_stds = {}
        for type_id in range(self.num_types):
            key = str(type_id)
            if key in self.param_mu_heads:
                param_mus[type_id] = (
                    self.param_mu_heads[key](x)
                )
                log_std = (
                    self.param_log_std_heads[key](x)
                )
                param_log_stds[type_id] = (
                    torch.clamp(
                        log_std,
                        self.LOG_STD_MIN,
                        self.LOG_STD_MAX,
                    )
                )

        return type_logits, param_mus, param_log_stds

    def sample(
        self, state: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        type_logits, param_mus, param_log_stds = (
            self.forward(state)
        )

        type_logits = type_logits - type_logits.max(
            dim=-1, keepdim=True
        )[0]
        type_probs = F.softmax(type_logits, dim=-1)
        type_probs = type_probs.clamp(min=1e-8)
        type_probs = type_probs / type_probs.sum(
            dim=-1, keepdim=True
        )

        type_dist = torch.distributions.Categorical(
            type_probs
        )
        action_type = type_dist.sample()
        type_log_prob = type_dist.log_prob(action_type)

        batch_size = state.shape[0]
        max_params = 3
        action_params = torch.zeros(
            batch_size, max_params
        )
        param_log_prob = torch.zeros(batch_size)

        for type_id in range(self.num_types):
            mask = action_type == type_id
            if mask.sum() == 0:
                continue
            if type_id not in param_mus:
                continue

            dim = self.param_dims[type_id]
            mu = param_mus[type_id][mask]
            log_std = param_log_stds[type_id][mask]
            std = log_std.exp().clamp(min=1e-6)

            normal = torch.distributions.Normal(
                mu, std
            )
            raw_sample = normal.rsample()

            # ═══ Squash through tanh ═══
            squashed = torch.tanh(raw_sample)

            # ═══ Log prob with tanh correction ═══
            log_p = normal.log_prob(raw_sample)
            log_p -= torch.log(
                1 - squashed.pow(2) + 1e-6
            )
            log_p = log_p.sum(dim=-1)

            # ═══ Scale to action bounds ═══
            scale, center = self._get_scale_center(
                type_id
            )
            if scale is not None:
                scaled = (
                    squashed * scale[:dim] + center[:dim]
                )
            else:
                scaled = squashed

            action_params[mask, :dim] = scaled
            param_log_prob[mask] = log_p

        total_log_prob = type_log_prob + param_log_prob

        return (
            action_type,
            action_params,
            total_log_prob,
            type_probs,
        )

    def sample_eval(
        self,
        state: torch.Tensor,
        temperature: float = 0.3,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Low-noise sampling for evaluation."""
        type_logits, param_mus, param_log_stds = (
            self.forward(state)
        )

        type_probs = F.softmax(
            type_logits / temperature, dim=-1
        )
        type_probs = type_probs.clamp(min=1e-8)
        type_probs = type_probs / type_probs.sum(
            dim=-1, keepdim=True
        )
        type_dist = torch.distributions.Categorical(
            type_probs
        )
        action_type = type_dist.sample()

        batch_size = state.shape[0]
        max_params = 3
        action_params = torch.zeros(
            batch_size, max_params
        )
        param_log_prob = torch.zeros(batch_size)

        for type_id in range(self.num_types):
            mask = action_type == type_id
            if (
                mask.sum() == 0
                or type_id not in param_mus
            ):
                continue
            dim = self.param_dims[type_id]
            mu = param_mus[type_id][mask]
            log_std = param_log_stds[type_id][mask]
            std = (
                log_std.exp() * temperature
            ).clamp(min=1e-6)

            normal = torch.distributions.Normal(
                mu, std
            )
            raw_sample = normal.rsample()

            # ═══ Squash + scale ═══
            squashed = torch.tanh(raw_sample)
            scale, center = self._get_scale_center(
                type_id
            )
            if scale is not None:
                scaled = (
                    squashed * scale[:dim] + center[:dim]
                )
            else:
                scaled = squashed

            action_params[mask, :dim] = scaled

        type_log_prob = type_dist.log_prob(action_type)
        total_log_prob = type_log_prob + param_log_prob

        return (
            action_type,
            action_params,
            total_log_prob,
            type_probs,
        )

    def load_bc_weights(
        self, bc_actor_state_dict: dict
    ):
        own_state = self.state_dict()
        loaded = 0
        skipped = 0

        for name, param in (
            bc_actor_state_dict.items()
        ):
            if (
                name in own_state
                and own_state[name].shape
                == param.shape
            ):
                own_state[name].copy_(param)
                loaded += 1
            elif name.startswith("param_heads."):
                parts = name.split(".")
                type_id = parts[1]
                mu_name = (
                    f"param_mu_heads.{type_id}"
                    f".{parts[2]}"
                )
                if (
                    mu_name in own_state
                    and own_state[mu_name].shape
                    == param.shape
                ):
                    own_state[mu_name].copy_(param)
                    loaded += 1
                else:
                    skipped += 1
            else:
                skipped += 1

        self.load_state_dict(own_state)
        logger.info(
            "BC weights loaded: %d transferred, "
            "%d skipped",
            loaded,
            skipped,
        )
