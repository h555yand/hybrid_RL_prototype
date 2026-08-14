# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Strategic SAC for phase transition decisions.

Small SAC network that learns when to switch phases:
- On surface: detach or keep crawling
- In air: switch to FLY_TO_GOAL or keep FLY_TO_EDGE

Uses compact strategic state (6D) instead of full 18D state.
Trained on outcomes of phase transitions collected by
TransitionMemory.
"""

import logging
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class StrategicActor(nn.Module):
    """Small actor for binary switch decision.

    Input: strategic state (6D)
    Output: switch probability (sigmoid)
    """

    def __init__(self, state_dim: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return logit for switch probability."""
        return self.net(state).squeeze(-1)

    def get_switch_prob(self, state: torch.Tensor) -> torch.Tensor:
        """Return switch probability (0-1)."""
        return torch.sigmoid(self.forward(state))

    def sample(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample binary action and return log probability.

        Returns:
            (action, log_prob) where action is 0 (stay) or 1 (switch)
        """
        logit = self.forward(state)
        prob = torch.sigmoid(logit)
        prob = prob.clamp(1e-6, 1 - 1e-6)

        dist = torch.distributions.Bernoulli(prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob


class StrategicCritic(nn.Module):
    """Twin critic for strategic decisions.

    Input: strategic state (6D) + action (1D binary)
    Output: Q-value
    """

    def __init__(self, state_dim: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action.unsqueeze(-1)], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)

    def min_q(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class StrategicSAC:
    """Strategic SAC for phase transition decisions.

    Learns when to switch phases (detach, direction change)
    from outcomes of past transitions. Works alongside
    TransitionMemory — TransitionMemory provides fast episodic
    recall, Strategic SAC provides generalization.

    Strategic state (6D):
        normal_agreement: dot(agent_normal, goal_normal)
        alignment: dot(goal_dir, agent_normal)
        norm_distance: distance / object_extent
        path_blocked: is direct path blocked (0/1)
        on_object: on surface or in air (0/1)
        norm_depth: normalized depth to surface

    Args:
        state_dim: Strategic state dimensionality.
        hidden_dim: Hidden layer size.
        gamma: Discount factor.
        tau: Soft update coefficient.
        lr: Learning rate for all networks.
        alpha_init: Initial entropy coefficient.
        buffer_capacity: Replay buffer size.
        batch_size: Training batch size.
    """

    def __init__(
        self,
        state_dim: int = 6,
        hidden_dim: int = 64,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        alpha_init: float = 0.2,
        buffer_capacity: int = 50000,
        batch_size: int = 128,
    ):
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.actor = StrategicActor(state_dim, hidden_dim)
        self.critic = StrategicCritic(state_dim, hidden_dim)
        self.critic_target = deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr
        )

        self.log_alpha = torch.tensor(
            np.log(alpha_init), requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=lr
        )
        self.target_entropy = -0.5  # binary action

        # Simple replay buffer
        self.buffer_capacity = buffer_capacity
        self.buffer_states = np.zeros(
            (buffer_capacity, state_dim), dtype=np.float32
        )
        self.buffer_actions = np.zeros(
            buffer_capacity, dtype=np.float32
        )
        self.buffer_rewards = np.zeros(
            buffer_capacity, dtype=np.float32
        )
        self.buffer_next_states = np.zeros(
            (buffer_capacity, state_dim), dtype=np.float32
        )
        self.buffer_dones = np.zeros(
            buffer_capacity, dtype=np.float32
        )
        self.buffer_size = 0
        self.buffer_ptr = 0

        # Normalization
        self._state_mean = np.zeros(state_dim)
        self._state_std = np.ones(state_dim)
        self._state_buffer = deque(maxlen=5000)
        self._norm_frozen = False

        # Stats
        self.total_updates = 0
        self._critic_losses = deque(maxlen=500)
        self._actor_losses = deque(maxlen=500)

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().item()

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return (state - self._state_mean) / (self._state_std + 1e-8)

    def _update_normalization(self, state: np.ndarray):
        self._state_buffer.append(state.copy())
        if self._norm_frozen:
            return
        n = len(self._state_buffer)
        if n < 100 or n % 100 != 0:
            return
        buf = np.array(self._state_buffer)
        self._state_mean = buf.mean(axis=0)
        self._state_std = np.maximum(buf.std(axis=0), 1e-4)
        if n >= 2000:
            self._norm_frozen = True

    def add_transition(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add strategic transition to buffer.

        Args:
            state: Strategic state (6D).
            action: 1.0 = switch, 0.0 = stay.
            reward: Outcome reward.
            next_state: Next strategic state.
            done: Episode ended.
        """
        self._update_normalization(state)
        norm_state = self.normalize_state(state)
        norm_next = self.normalize_state(next_state)

        idx = self.buffer_ptr % self.buffer_capacity
        self.buffer_states[idx] = norm_state
        self.buffer_actions[idx] = action
        self.buffer_rewards[idx] = reward
        self.buffer_next_states[idx] = norm_next
        self.buffer_dones[idx] = float(done)

        self.buffer_ptr += 1
        self.buffer_size = min(
            self.buffer_size + 1, self.buffer_capacity
        )

    def predict(self, state: np.ndarray) -> Tuple[bool, float]:
        """Predict whether to switch phase.

        Args:
            state: Strategic state (6D, raw).

        Returns:
            (should_switch, confidence)
        """
        norm_state = self.normalize_state(state)
        state_t = torch.FloatTensor(norm_state).unsqueeze(0)

        with torch.no_grad():
            prob = self.actor.get_switch_prob(state_t).item()

        should_switch = prob > 0.5
        confidence = abs(prob - 0.5) * 2  # 0 = uncertain, 1 = certain

        return should_switch, confidence

    def update(self, num_steps: int = 10) -> dict:
        """Run training updates.

        Args:
            num_steps: Number of gradient steps.

        Returns:
            Training stats dict.
        """
        if self.buffer_size < self.batch_size:
            return {"skipped": True, "buffer_size": self.buffer_size}

        for _ in range(num_steps):
            indices = np.random.randint(
                0, self.buffer_size, self.batch_size
            )

            states = torch.FloatTensor(self.buffer_states[indices])
            actions = torch.FloatTensor(self.buffer_actions[indices])
            rewards = torch.FloatTensor(self.buffer_rewards[indices])
            next_states = torch.FloatTensor(
                self.buffer_next_states[indices]
            )
            dones = torch.FloatTensor(self.buffer_dones[indices])

            # Update critic
            with torch.no_grad():
                next_action, next_log_prob = self.actor.sample(
                    next_states
                )
                next_q = self.critic_target.min_q(
                    next_states, next_action
                )
                target_q = rewards + self.gamma * (1 - dones) * (
                    next_q - self.alpha * next_log_prob
                )

            q1, q2 = self.critic(states, actions)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(
                q2, target_q
            )

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            self._critic_losses.append(critic_loss.item())

            # Update actor
            new_action, log_prob = self.actor.sample(states)
            q_val = self.critic.min_q(states, new_action)
            actor_loss = (self.alpha * log_prob - q_val).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self._actor_losses.append(actor_loss.item())

            # Update alpha
            alpha_loss = -(
                self.log_alpha
                * (log_prob.detach().mean() + self.target_entropy)
            )
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # Soft update target
            for param, target_param in zip(
                self.critic.parameters(),
                self.critic_target.parameters(),
            ):
                target_param.data.copy_(
                    self.tau * param.data
                    + (1 - self.tau) * target_param.data
                )

            self.total_updates += 1

        return {
            "updates": num_steps,
            "critic_loss": float(np.mean(self._critic_losses))
            if self._critic_losses else 0,
            "actor_loss": float(np.mean(self._actor_losses))
            if self._actor_losses else 0,
            "alpha": self.alpha,
            "buffer_size": self.buffer_size,
        }
    
    def warm_start_from_q_store(
        self,
        transition_store,
    ):
        """Populate buffer from one HNSWStateStore.

        Uses raw_state (5D) directly.

        Args:
            transition_store: HNSWStateStore
                (strategic_detach or strategic_direction).
        """
        for point in transition_store.points.values():
            raw = point.raw_state.copy().astype(
                np.float32
            )
            outcome = float(
                point.q_values[1] - point.q_values[0]
            )

            if outcome > 0.3:
                self.add_transition(
                    raw, 1.0, 1.0, raw, True
                )
                self.add_transition(
                    raw, 0.0, -0.3, raw, True
                )
            elif outcome < -0.1:
                self.add_transition(
                    raw, 1.0, -0.5, raw, True
                )
            else:
                self.add_transition(
                    raw, 1.0, 0.0, raw, True
                )

        logger.info(
            "Strategic SAC warm start: %d buffer entries "
            "from %d store points",
            self.buffer_size,
            len(transition_store.points),
        )

    def warm_start_from_transition_memory(
        self,
        transition_detach,
        transition_direction,
    ):
        """Legacy wrapper — kept for backward compatibility."""
        logger.warning(
            "warm_start_from_transition_memory is "
            "deprecated, use warm_start_from_q_store"
        )
        self.warm_start_from_q_store(transition_detach)

    def get_stats(self) -> dict:
        return {
            "total_updates": self.total_updates,
            "buffer_size": self.buffer_size,
            "alpha": round(self.alpha, 4),
            "critic_loss": round(
                float(np.mean(self._critic_losses))
                if self._critic_losses else 0, 4
            ),
            "actor_loss": round(
                float(np.mean(self._actor_losses))
                if self._actor_losses else 0, 4
            ),
            "norm_frozen": self._norm_frozen,
        }

    def save(self, filepath: str):
        dirpath = Path(filepath)
        dirpath.mkdir(parents=True, exist_ok=True)

        torch.save(
            self.actor.state_dict(),
            dirpath / "strategic_actor.pt",
        )
        torch.save(
            self.critic.state_dict(),
            dirpath / "strategic_critic.pt",
        )
        torch.save(
            self.critic_target.state_dict(),
            dirpath / "strategic_critic_target.pt",
        )
        np.savez(
            dirpath / "strategic_state.npz",
            state_mean=self._state_mean,
            state_std=self._state_std,
            total_updates=self.total_updates,
            log_alpha=self.log_alpha.detach().numpy(),
            state_dim=np.array([self.state_dim]),
        )
        # Save buffer
        np.savez(
            dirpath / "strategic_buffer.npz",
            states=(
                self.buffer_states[:self.buffer_size]
            ),
            actions=(
                self.buffer_actions[:self.buffer_size]
            ),
            rewards=(
                self.buffer_rewards[:self.buffer_size]
            ),
            next_states=(
                self.buffer_next_states[
                    :self.buffer_size
                ]
            ),
            dones=(
                self.buffer_dones[:self.buffer_size]
            ),
        )
        logger.info(
            "Strategic SAC saved to %s "
            "(%d buffer entries)",
            dirpath,
            self.buffer_size,
        )
        
    @classmethod
    def load(cls, filepath: str) -> "StrategicSAC":
        dirpath = Path(filepath)

        data = np.load(
            dirpath / "strategic_state.npz"
        )

        # Restore state_dim
        state_dim = (
            int(data["state_dim"][0])
            if "state_dim" in data
            else None
        )
        if state_dim is None:
            # Infer from saved actor weights
            actor_weights = torch.load(
                dirpath / "strategic_actor.pt",
                weights_only=True,
            )
            first_layer_key = "net.0.weight"
            if first_layer_key in actor_weights:
                state_dim = (
                    actor_weights[first_layer_key]
                    .shape[1]
                )
            else:
                state_dim = 6

        sac = cls(state_dim=state_dim)

        sac.actor.load_state_dict(
            torch.load(
                dirpath / "strategic_actor.pt",
                weights_only=True,
            )
        )
        sac.critic.load_state_dict(
            torch.load(
                dirpath / "strategic_critic.pt",
                weights_only=True,
            )
        )
        sac.critic_target.load_state_dict(
            torch.load(
                dirpath / "strategic_critic_target.pt",
                weights_only=True,
            )
        )

        sac._state_mean = data["state_mean"]
        sac._state_std = data["state_std"]
        sac.total_updates = int(
            data["total_updates"]
        )
        sac.log_alpha = torch.tensor(
            float(data["log_alpha"]),
            requires_grad=True,
        )
        sac._norm_frozen = True

        # Load buffer
        buf_path = dirpath / "strategic_buffer.npz"
        if buf_path.exists():
            buf = np.load(buf_path)
            n = len(buf["states"])
            sac.buffer_states[:n] = buf["states"]
            sac.buffer_actions[:n] = buf["actions"]
            sac.buffer_rewards[:n] = buf["rewards"]
            sac.buffer_next_states[:n] = (
                buf["next_states"]
            )
            sac.buffer_dones[:n] = buf["dones"]
            sac.buffer_size = n
            sac.buffer_ptr = n

        logger.info(
            "Strategic SAC loaded from %s "
            "(state_dim=%d, %d buffer entries)",
            dirpath,
            state_dim,
            sac.buffer_size,
        )
        return sac

class StrategicBCTrainer:
    """Behavioral Cloning for Strategic SAC warm start.

    Trains ONE strategic actor from ONE HNSWStateStore.

    Input: strategic state (5D)
    Output: switch probability (should detach / should switch direction)
    """

    def __init__(
        self,
        state_dim: int = 5,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        batch_size: int = 64,
        val_split: float = 0.1,
        patience: int = 20,
    ):
        self.state_dim = state_dim
        self.lr = lr
        self.batch_size = batch_size
        self.val_split = val_split
        self.patience = patience

        self.actor = StrategicActor(state_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr
        )

        self.state_mean = None
        self.state_std = None

    def prepare_data_from_q_store(
        self,
        transition_store,
    ) -> bool:
        """Convert one HNSWStateStore to BC training data.

        Uses raw_state directly (5D) from the store.
        Outcome = q_values[1] - q_values[0] (switch - stay).

        Positive outcome → label=1 (switch was good)
        Negative outcome → label=0 (switch was bad, should stay)

        Args:
            transition_store: HNSWStateStore (strategic_detach
                or strategic_direction).

        Returns:
            True if data was prepared, False if no data.
        """
        states = []
        labels = []

        for point in transition_store.points.values():
            raw = point.raw_state
            outcome = float(
                point.q_values[1] - point.q_values[0]
            )

            s_state = raw.copy().astype(np.float32)

            if outcome > 0.3:
                # Switch was good
                states.append(s_state)
                labels.append(1.0)
                # Also: staying would be bad
                states.append(s_state.copy())
                labels.append(0.0)
            elif outcome < -0.1:
                # Switch was bad
                states.append(s_state)
                labels.append(0.0)
            else:
                # Neutral
                states.append(s_state)
                labels.append(0.5)

        if not states:
            logger.warning(
                "No data for Strategic BC training"
            )
            return False

        states = np.array(states, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)

        # Normalize
        self.state_mean = states.mean(axis=0)
        self.state_std = np.maximum(
            states.std(axis=0), 1e-6
        )
        states = (
            (states - self.state_mean) / self.state_std
        )

        # Split
        n = len(states)
        n_val = max(1, int(n * self.val_split))
        indices = np.random.permutation(n)

        self.train_states = torch.FloatTensor(
            states[indices[n_val:]]
        )
        self.train_labels = torch.FloatTensor(
            labels[indices[n_val:]]
        )
        self.val_states = torch.FloatTensor(
            states[indices[:n_val]]
        )
        self.val_labels = torch.FloatTensor(
            labels[indices[:n_val]]
        )

        logger.info(
            "Strategic BC data: %d train, %d val "
            "(%d points from store)",
            len(self.train_states),
            len(self.val_states),
            len(transition_store.points),
        )
        return True

    def prepare_data_from_transition_memory(
        self,
        transition_detach,
        transition_direction,
    ):
        """Legacy wrapper — kept for backward compatibility.

        For new code use prepare_data_from_q_store() with
        separate StrategicBCTrainer instances.
        """
        logger.warning(
            "prepare_data_from_transition_memory is "
            "deprecated, use prepare_data_from_q_store"
        )
        # Use detach store as primary
        return self.prepare_data_from_q_store(
            transition_detach
        )
    
    def train(self, num_epochs: int = 100):
        """Train strategic actor with BCE loss."""
        if not hasattr(self, "train_states"):
            logger.warning("No training data prepared")
            return

        best_val_loss = float("inf")
        best_state_dict = None
        epochs_without_improvement = 0
        n_train = len(self.train_states)

        for epoch in range(num_epochs):
            self.actor.train()
            indices = torch.randperm(n_train)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_train, self.batch_size):
                end = min(start + self.batch_size, n_train)
                batch_idx = indices[start:end]

                batch_states = self.train_states[batch_idx]
                batch_labels = self.train_labels[batch_idx]

                logits = self.actor(batch_states)
                loss = F.binary_cross_entropy_with_logits(
                    logits, batch_labels
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            epoch_loss /= max(n_batches, 1)

            # Validation
            self.actor.eval()
            with torch.no_grad():
                val_logits = self.actor(self.val_states)
                val_loss = F.binary_cross_entropy_with_logits(
                    val_logits, self.val_labels
                ).item()
                val_preds = (torch.sigmoid(val_logits) > 0.5).float()
                val_labels_binary = (self.val_labels > 0.5).float()
                val_acc = (
                    (val_preds == val_labels_binary).float().mean().item()
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = {
                    k: v.clone()
                    for k, v in self.actor.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (epoch + 1) % 20 == 0:
                logger.info(
                    "Strategic BC epoch %d/%d: "
                    "train_loss=%.4f, val_loss=%.4f, val_acc=%.3f",
                    epoch + 1, num_epochs,
                    epoch_loss, val_loss, val_acc,
                )

            if epochs_without_improvement >= self.patience:
                logger.info(
                    "Strategic BC early stopping at epoch %d",
                    epoch + 1,
                )
                break

        if best_state_dict is not None:
            self.actor.load_state_dict(best_state_dict)

        logger.info(
            "Strategic BC complete: best_val_loss=%.4f",
            best_val_loss,
        )

    def get_actor_weights(self):
        """Return trained actor state dict for loading into Strategic SAC."""
        return self.actor.state_dict()

    def get_normalization(self):
        """Return normalization stats."""
        return self.state_mean, self.state_std
