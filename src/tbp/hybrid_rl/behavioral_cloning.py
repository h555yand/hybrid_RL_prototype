# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""BC Actor Network for Parameterized SAC.
Behavioral Cloning: learns to copy Q-learning expert from successful trajectories.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .experience_extractor import ExperienceExtractor, PSACTransition

logger = logging.getLogger(__name__)


class BCActorNetwork(nn.Module):

    def __init__(self, state_dim: int = 15, num_types: int = 9):
        super().__init__()
        self.state_dim = state_dim
        self.num_types = num_types
        self.param_dims = ExperienceExtractor.get_param_dims()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.type_head = nn.Linear(128, num_types)

        self.param_heads = nn.ModuleDict()
        for type_id in range(num_types):
            dim = self.param_dims[type_id]
            if dim > 0:
                self.param_heads[str(type_id)] = nn.Linear(128, dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        x = self.encoder(state)
        type_logits = self.type_head(x)
        params = {}
        for type_id in range(self.num_types):
            if str(type_id) in self.param_heads:
                params[type_id] = self.param_heads[str(type_id)](x)
        return type_logits, params

    def predict(self, state: np.ndarray) -> Tuple[int, np.ndarray]:
        self.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            type_logits, params = self.forward(state_t)
            action_type = int(torch.argmax(type_logits, dim=1).item())
            if action_type in params:
                action_params = params[action_type][0].numpy()
            else:
                action_params = np.zeros(0)
        return action_type, action_params


class BCTrainer:

    def __init__(
        self,
        state_dim: int = 15,
        num_types: int = 8,
        lr: float = 3e-4,
        batch_size: int = 64,
        param_loss_weight: float = 1.0,
        val_split: float = 0.1,
        patience: int = 20,
    ):
        self.state_dim = state_dim
        self.num_types = num_types
        self.lr = lr
        self.batch_size = batch_size
        self.param_loss_weight = param_loss_weight
        self.val_split = val_split
        self.patience = patience
        self.param_dims = ExperienceExtractor.get_param_dims()

        self.actor = BCActorNetwork(state_dim=state_dim, num_types=num_types)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.state_mean = None
        self.state_std = None

    def prepare_data(self, transitions: List[PSACTransition]):
        states = np.array([tr.state for tr in transitions], dtype=np.float32)
        types = np.array([tr.action_type for tr in transitions], dtype=np.int64)
        params = np.array([tr.action_params for tr in transitions], dtype=np.float32)

        self.state_mean = states.mean(axis=0)
        self.state_std = np.maximum(states.std(axis=0), 1e-6)
        states = (states - self.state_mean) / self.state_std

        self.param_mean = params.mean(axis=0)
        self.param_std = np.maximum(params.std(axis=0), 1e-6)
        params = (params - self.param_mean) / self.param_std

        n = len(states)
        n_val = max(1, int(n * self.val_split))
        indices = np.random.permutation(n)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        self.train_states = torch.FloatTensor(states[train_idx])
        self.train_types = torch.LongTensor(types[train_idx])
        self.train_params = torch.FloatTensor(params[train_idx])

        self.val_states = torch.FloatTensor(states[val_idx])
        self.val_types = torch.LongTensor(types[val_idx])
        self.val_params = torch.FloatTensor(params[val_idx])

        type_counts = {}
        for t in types:
            name = ExperienceExtractor.get_type_names()[t]
            type_counts[name] = type_counts.get(name, 0) + 1

        logger.info(
            f"BC data: {len(train_idx)} train, {len(val_idx)} val, "
            f"types: {type_counts}"
        )

    def _compute_loss(self, states, types, params):
        type_logits, predicted_params = self.actor(states)

        type_loss = F.cross_entropy(type_logits, types)

        param_loss = torch.tensor(0.0)
        param_count = 0

        for type_id in range(self.num_types):
            dim = self.param_dims[type_id]
            if dim == 0:
                continue
            mask = (types == type_id)
            if mask.sum() == 0:
                continue
            pred = predicted_params[type_id][mask]
            target = params[mask, :dim]
            param_loss = param_loss + F.mse_loss(pred, target) * mask.sum()
            param_count += mask.sum().item()

        if param_count > 0:
            param_loss = param_loss / param_count

        loss = type_loss + self.param_loss_weight * param_loss

        with torch.no_grad():
            predicted_types = torch.argmax(type_logits, dim=1)
            accuracy = (predicted_types == types).float().mean().item()

        return loss, type_loss.item(), param_loss.item(), accuracy

    def train(self, transitions: List[PSACTransition], num_epochs: int = 200):
        self.prepare_data(transitions)

        best_val_loss = float("inf")
        best_state_dict = None
        epochs_without_improvement = 0

        n_train = len(self.train_states)

        for epoch in range(num_epochs):
            self.actor.train()
            indices = torch.randperm(n_train)
            epoch_loss = 0.0
            epoch_type_loss = 0.0
            epoch_param_loss = 0.0
            epoch_accuracy = 0.0
            n_batches = 0

            for start in range(0, n_train, self.batch_size):
                end = min(start + self.batch_size, n_train)
                batch_idx = indices[start:end]

                batch_states = self.train_states[batch_idx]
                batch_types = self.train_types[batch_idx]
                batch_params = self.train_params[batch_idx]

                loss, tl, pl, acc = self._compute_loss(
                    batch_states, batch_types, batch_params
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_type_loss += tl
                epoch_param_loss += pl
                epoch_accuracy += acc
                n_batches += 1

            epoch_loss /= n_batches
            epoch_type_loss /= n_batches
            epoch_param_loss /= n_batches
            epoch_accuracy /= n_batches

            self.actor.eval()
            with torch.no_grad():
                val_loss, val_tl, val_pl, val_acc = self._compute_loss(
                    self.val_states, self.val_types, self.val_params
                )
                val_loss = val_loss.item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = {
                    k: v.clone() for k, v in self.actor.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs}: "
                    f"train_loss={epoch_loss:.4f} (type={epoch_type_loss:.4f}, param={epoch_param_loss:.4f}) "
                    f"train_acc={epoch_accuracy:.3f} | "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
                )

            if epochs_without_improvement >= self.patience:
                logger.info(
                    f"Early stopping at epoch {epoch+1}, "
                    f"best_val_loss={best_val_loss:.4f}"
                )
                break

        if best_state_dict is not None:
            self.actor.load_state_dict(best_state_dict)

        logger.info(
            f"BC training complete: best_val_loss={best_val_loss:.4f}"
        )

    def save(self, dirpath: str):
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)

        torch.save(self.actor.state_dict(), dirpath / "bc_actor.pt")
        np.savez(
            dirpath / "bc_normalization.npz",
            state_mean=self.state_mean,
            state_std=self.state_std,
            param_mean=self.param_mean,
            param_std=self.param_std,
        )
        logger.info("BC model saved to %s", dirpath)

    def load(self, dirpath: str):
        dirpath = Path(dirpath)

        self.actor.load_state_dict(
            torch.load(dirpath / "bc_actor.pt", weights_only=True)
        )
        norm = np.load(dirpath / "bc_normalization.npz")
        self.state_mean = norm["state_mean"]
        self.state_std = norm["state_std"]
        self.param_mean = norm["param_mean"]
        self.param_std = norm["param_std"]
        logger.info("BC model loaded from %s", dirpath)

    def predict(self, state: np.ndarray) -> Tuple[int, np.ndarray]:
        state_norm = (state - self.state_mean) / self.state_std
        action_type, action_params_norm = self.actor.predict(state_norm.astype(np.float32))
        param_dim = len(action_params_norm)
        action_params = action_params_norm * self.param_std[:param_dim] + self.param_mean[:param_dim]
        return action_type, action_params
