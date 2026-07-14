# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""P-SAC Training Loop.

Combines Actor, Twin Critic, Replay Buffer, ActionInterpreter
for Parameterized Soft Actor-Critic training with BC warm-start.
"""

from __future__ import annotations

import logging
import pickle
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .action_interpreter import ActionInterpreter
from .experience_extractor import ExperienceExtractor, PSACTransition
from .lightweight_env import LightweightEnv
from .replay_buffer import ReplayBuffer
from .rl_goal_approach_controller import RLGoalApproachController
from .sac_actor import SACActorNetwork
from .twin_critic import TwinCritic

logger = logging.getLogger(__name__)


class PSACTrainer:
    """Parameterized SAC trainer with BC regularization."""

    MIN_LOG_ALPHA_TYPE = -2.0
    MAX_LOG_ALPHA_TYPE = 1.0
    MIN_LOG_ALPHA_PARAM = -2.0
    MAX_LOG_ALPHA_PARAM = 0.0

    def __init__(
        self,
        state_dim: int = 15,
        num_types: int = 9,
        max_params: int = 3,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        alpha_type_init: float = 0.2,
        alpha_param_init: float = 0.2,
        batch_size: int = 256,
        buffer_capacity: int = 100_000,
        bc_lambda_init: float = 1.0,
        bc_lambda_decay: float = 0.9999,
        max_steps_per_goal: int = 150,
        goal_threshold: float = 5.0,
        eval_interval: int = 200,
        eval_episodes: int = 100,
        eval_seed: int = 12345,
    ) -> None:
        self.state_dim = state_dim
        self.num_types = num_types
        self.max_params = max_params
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_steps_per_goal = max_steps_per_goal
        self.goal_threshold = goal_threshold
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.eval_seed = eval_seed

        self.actor = SACActorNetwork(state_dim, num_types)
        self.critic = TwinCritic(state_dim, num_types, max_params)
        self.critic_target = deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr_critic
        )

        self.log_alpha_type = torch.tensor(
            np.log(alpha_type_init), requires_grad=True
        )
        self.log_alpha_param = torch.tensor(
            np.log(alpha_param_init), requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha_type, self.log_alpha_param], lr=lr_alpha
        )

        self.target_entropy_type = -np.log(1.0 / num_types) * 0.5
        self.target_entropy_param = -max_params * 0.5

        self.buffer = ReplayBuffer(
            buffer_capacity, state_dim, max_params,
            bc_reserve_fraction=0.15,
        )

        self.bc_lambda = bc_lambda_init
        self.bc_lambda_decay = bc_lambda_decay
        self.bc_data = None

        self.state_mean = None
        self.state_std = None
        self.param_mean = None
        self.param_std = None

        self.total_steps = 0
        self.total_episodes = 0
        self.total_goals_reached = 0

        # Training metrics
        self._action_type_counts: dict[int, int] = {}
        self._collision_counts: dict[int, int] = {}
        self._episode_rewards: list[float] = []
        self._episode_steps: list[int] = []
        self._critic_losses: deque[float] = deque(maxlen=1000)
        self._actor_losses: deque[float] = deque(maxlen=1000)
        self._mesh_stats: dict[str, dict[str, Any]] = {}
        self._current_mesh: str = ""

    @property
    def alpha_type(self) -> float:
        """Current type entropy coefficient."""
        return self.log_alpha_type.exp().item()

    @property
    def alpha_param(self) -> float:
        """Current parameter entropy coefficient."""
        return self.log_alpha_param.exp().item()

    def load_bc(self, bc_model_dir: str, bc_data_path: str) -> None:
        """Load BC actor weights, normalization, and data into buffer.

        Args:
            bc_model_dir: Directory with bc_actor.pt and bc_normalization.npz.
            bc_data_path: Path to bc_data.pkl file.
        """
        bc_state_dict = torch.load(
            Path(bc_model_dir) / "bc_actor.pt", weights_only=True
        )
        self.actor.load_bc_weights(bc_state_dict)

        norm = np.load(Path(bc_model_dir) / "bc_normalization.npz")
        self.state_mean = norm["state_mean"]
        self.state_std = norm["state_std"]
        self.param_mean = norm["param_mean"]
        self.param_std = norm["param_std"]

        with Path(bc_data_path).open("rb") as f:
            bc_transitions = pickle.load(f)  # noqa: S301

        bc_normalized = self._normalize_bc_transitions(bc_transitions)
        self.buffer.load_bc_data(bc_normalized)
        self.bc_data = bc_transitions

        logger.info(
            "Loaded BC: actor weights, normalization, %d transitions",
            len(bc_transitions),
        )

    def _normalize_bc_transitions(
        self, transitions: list[PSACTransition]
    ) -> list[PSACTransition]:
        normalized = []
        for tr in transitions:
            norm_state = self.normalize_state(tr.state)
            norm_next = (
                self.normalize_state(tr.next_state)
                if tr.next_state is not None
                else None
            )
            norm_params = (
                tr.action_params
                - self.param_mean[:len(tr.action_params)]
            ) / (self.param_std[:len(tr.action_params)] + 1e-8)
            padded_params = np.zeros(self.max_params, dtype=np.float32)
            padded_params[:len(norm_params)] = norm_params
            normalized.append(PSACTransition(
                state=norm_state.astype(np.float32),
                action_type=tr.action_type,
                action_params=padded_params,
                reward=tr.reward,
                next_state=(
                    norm_next.astype(np.float32)
                    if norm_next is not None
                    else None
                ),
                done=tr.done,
            ))
        return normalized

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using BC normalization statistics.

        Args:
            state: Raw state vector.

        Returns:
            Normalized state vector.
        """
        if self.state_mean is not None:
            return (state - self.state_mean) / (self.state_std + 1e-8)
        return state

    def compute_state(
        self,
        env: LightweightEnv,
        controller: RLGoalApproachController,
        current_pose: np.ndarray,
        sensor_data: dict[str, Any],
    ) -> np.ndarray:
        """Compute state vector using controller's state computation.

        Args:
            env: Environment instance.
            controller: RL controller for state computation.
            current_pose: Current agent pose.
            sensor_data: Current sensor readings.

        Returns:
            State vector.
        """
        return controller._compute_state(current_pose, sensor_data)  # noqa: SLF001

    def compute_reward(
        self,
        state: np.ndarray,
        prev_state: np.ndarray,
        distance: float,
        prev_distance: float,
        collision: str | None,
        steps: int,
        config: dict[str, Any] | None = None,
        sensor_data: dict[str, Any] | None = None,
    ) -> tuple[float, bool]:
        """Compute reward for a SAC step.

        Uses SMDP step penalty proportional to sub_steps for detach actions.

        Args:
            state: Current state vector.
            prev_state: Previous state vector.
            distance: Current distance to goal.
            prev_distance: Previous distance to goal.
            collision: Collision type string or None.
            steps: Current step number in episode.
            config: Optional config overrides for reward parameters.
            sensor_data: Sensor data for sub_steps info.

        Returns:
            Tuple of (reward, done).
        """
        cfg = config or {}
        surface_step = cfg.get("surface_step", 3.0)
        reward_progress = cfg.get("reward_progress", 3.0)
        reward_goal = cfg.get("reward_goal_reached", 60.0)
        reward_step = cfg.get("reward_step_penalty", -0.5)
        reward_collision = cfg.get("reward_surface_violation", -12.0)
        reward_timeout = cfg.get("reward_timeout", -12.0)
        max_steps = cfg.get(
            "max_steps_per_goal", self.max_steps_per_goal
        )

        reward = 0.0
        done = False

        progress = prev_distance - distance
        reward += progress / surface_step * reward_progress

        if distance < self.goal_threshold:
            reward += reward_goal
            done = True

        # SMDP: step penalty proportional to sub_steps
        sub_steps = 1
        if sensor_data is not None:
            sub_steps = max(
                sensor_data.get("detach_sub_steps", 1), 1
            )
        reward += reward_step * sub_steps

        if collision == "surface_violation":
            reward += reward_collision
            done = True

        if steps >= max_steps:
            reward += reward_timeout
            done = True

        return reward, done

    def update_critic(
        self, batch: dict[str, np.ndarray]
    ) -> float:
        """Update twin critic networks.

        Args:
            batch: Batch from replay buffer.

        Returns:
            Critic loss value.
        """
        states = torch.FloatTensor(batch["states"])
        action_types = torch.LongTensor(batch["action_types"])
        action_params = torch.FloatTensor(batch["action_params"])
        rewards = torch.FloatTensor(batch["rewards"])
        next_states = torch.FloatTensor(batch["next_states"])
        dones = torch.FloatTensor(batch["dones"])

        with torch.no_grad():
            next_type, next_params, next_log_prob, _ = (
                self.actor.sample(next_states)
            )
            next_q = self.critic_target.min_q(
                next_states, next_type, next_params
            )
            alpha = self.alpha_type + self.alpha_param
            target_q = rewards + self.gamma * (1 - dones) * (
                next_q - alpha * next_log_prob
            )

        q1, q2 = self.critic(states, action_types, action_params)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(
            q2, target_q
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), 1.0
        )
        self.critic_optimizer.step()

        return critic_loss.item()

    def update_actor(
        self, batch: dict[str, np.ndarray]
    ) -> tuple[float, float]:
        """Update actor network with SAC + BC regularization.

        Args:
            batch: Batch from replay buffer.

        Returns:
            Tuple of (sac_loss, bc_loss).
        """
        states = torch.FloatTensor(batch["states"])

        action_type, action_params, log_prob, type_probs = (
            self.actor.sample(states)
        )
        q_val = self.critic.min_q(states, action_type, action_params)

        sac_loss = (
            (self.alpha_type + self.alpha_param) * log_prob - q_val
        ).mean()

        bc_loss = torch.tensor(0.0)
        if self.bc_lambda > 0.01 and self.bc_data is not None:
            bc_batch_size = min(64, len(self.bc_data))
            bc_indices = np.random.randint(
                0, len(self.bc_data), bc_batch_size
            )
            bc_states = torch.FloatTensor(np.array([
                self.normalize_state(self.bc_data[i].state)
                for i in bc_indices
            ]))
            bc_types = torch.LongTensor([
                self.bc_data[i].action_type for i in bc_indices
            ])
            bc_params_raw = np.array([
                self.bc_data[i].action_params for i in bc_indices
            ])
            bc_params = torch.FloatTensor(
                (bc_params_raw - self.param_mean)
                / (self.param_std + 1e-8)
            )

            type_logits, param_mus, _ = self.actor(bc_states)
            type_loss = F.cross_entropy(type_logits, bc_types)

            param_loss = torch.tensor(0.0)
            param_dims = ExperienceExtractor.get_param_dims()
            for type_id in range(self.num_types):
                dim = param_dims[type_id]
                if dim == 0:
                    continue
                mask = bc_types == type_id
                if mask.sum() == 0:
                    continue
                pred = param_mus[type_id][mask]
                target = bc_params[mask, :dim]
                param_loss = param_loss + F.mse_loss(pred, target)

            bc_loss = type_loss + param_loss

        actor_loss = sac_loss + self.bc_lambda * bc_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), 0.5
        )
        self.actor_optimizer.step()

        self.bc_lambda *= self.bc_lambda_decay

        return sac_loss.item(), bc_loss.item()

    def update_alpha(
        self, batch: dict[str, np.ndarray]
    ) -> None:
        """Update entropy temperature coefficients.

        Args:
            batch: Batch from replay buffer.
        """
        states = torch.FloatTensor(batch["states"])

        with torch.no_grad():
            _, _, log_prob, type_probs = self.actor.sample(states)
            type_entropy = -(
                type_probs * torch.log(type_probs + 1e-8)
            ).sum(dim=-1).mean()

        alpha_type_loss = -(
            self.log_alpha_type
            * (type_entropy - self.target_entropy_type).detach()
        )
        alpha_param_loss = -(
            self.log_alpha_param
            * (-log_prob.mean() - self.target_entropy_param).detach()
        )
        alpha_loss = alpha_type_loss + alpha_param_loss

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            self.log_alpha_type.clamp_(
                min=self.MIN_LOG_ALPHA_TYPE,
                max=self.MAX_LOG_ALPHA_TYPE,
            )
            self.log_alpha_param.clamp_(
                min=self.MIN_LOG_ALPHA_PARAM,
                max=self.MAX_LOG_ALPHA_PARAM,
            )

    def soft_update_target(self) -> None:
        """Soft update target critic networks."""
        for param, target_param in zip(
            self.critic.parameters(),
            self.critic_target.parameters(),
        ):
            target_param.data.copy_(
                self.tau * param.data
                + (1 - self.tau) * target_param.data
            )

    def _run_eval_during_training(
        self,
        env: LightweightEnv,
        controller: RLGoalApproachController,
        interpreter: ActionInterpreter,
        curriculum_levels: list[tuple[float, float]] | None = None,
        num_episodes: int = 100,
    ) -> float:
        """Run evaluation during training with fixed seed.

        Args:
            env: Environment instance.
            controller: RL controller for state computation.
            interpreter: Action interpreter.
            curriculum_levels: Optional curriculum levels.
            num_episodes: Number of eval episodes.

        Returns:
            Success rate.
        """
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()

        np.random.seed(self.eval_seed)  # noqa: NPY002
        torch.manual_seed(self.eval_seed)

        successes = 0
        max_level = (
            len(curriculum_levels) - 1 if curriculum_levels else 0
        )

        self.actor.eval()

        for _ep in range(num_episodes):
            if curriculum_levels:
                min_dist, max_dist = curriculum_levels[max_level]
            else:
                min_dist, max_dist = 10.0, 120.0

            env.reset()
            start_pos = env.get_pose()[:3]
            goal_pose = env.get_random_surface_point(
                reference_pos=start_pos,
                min_dist=min_dist,
                max_dist=max_dist,
                max_attempts=2000,
                mesh_sample=True,
            )
            controller.set_new_goal(goal_pose, start_pos)
            env.set_goal(goal_pose)

            for _step in range(self.max_steps_per_goal):
                current_pose = env.get_pose()
                sensor_data = env.get_sensor_data()
                state_raw = self.compute_state(
                    env, controller, current_pose, sensor_data
                )
                state = self.normalize_state(state_raw)

                state_t = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    at, ap, _, _ = self.actor.sample_eval(state_t)
                action_type = at[0].item()
                action_params = (
                    ap[0].numpy() * self.param_std
                    + self.param_mean
                )

                sensor_data = interpreter.execute(
                    action_type, action_params
                )
                current_pose = env.get_pose()
                distance = float(
                    np.linalg.norm(goal_pose[:3] - current_pose[:3])
                )

                if distance < self.goal_threshold:
                    successes += 1
                    break

                depth = sensor_data.get("depth", 100.0)
                if depth < 0.5:
                    break

        self.actor.train()

        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)

        return successes / max(num_episodes, 1)

    def _get_episode_data(
        self,
        episode_pools: dict[str, Any] | None,
        curr_level: int,
        episode: int,
    ) -> dict[str, Any] | None:
        if episode_pools is None:
            return None
        levels = episode_pools.get("levels", [])
        if curr_level >= len(levels):
            return None
        level_pool = levels[curr_level]
        ep_idx = episode % len(level_pool)
        return level_pool[ep_idx]

    def start_mesh_tracking(self, mesh_name: str) -> None:
        """Save current stats and start tracking for a new mesh.

        Args:
            mesh_name: Name of the new mesh being trained on.
        """
        if self._current_mesh and self.total_episodes > 0:
            self._mesh_stats[self._current_mesh] = {
                "episodes": self.total_episodes,
                "goals": self.total_goals_reached,
                "success_rate": (
                    self.total_goals_reached
                    / max(self.total_episodes, 1)
                ),
                "action_distribution": dict(self._action_type_counts),
                "collision_counts": dict(self._collision_counts),
            }
        self._current_mesh = mesh_name

    def get_training_stats(self) -> dict[str, Any]:
        """Get comprehensive training statistics.

        Returns:
            Dict with action distribution, collision rates,
            efficiency metrics, and loss history.
        """
        type_names = ExperienceExtractor.get_type_names()

        # Action type distribution
        total_actions = max(
            sum(self._action_type_counts.values()), 1
        )
        action_distribution = {
            type_names.get(k, f"type_{k}"): {
                "count": v,
                "rate": round(v / total_actions, 4),
            }
            for k, v in sorted(self._action_type_counts.items())
        }

        # Collision rate per type
        collision_rate_per_type = {}
        for type_id, collision_count in self._collision_counts.items():
            total_calls = self._action_type_counts.get(type_id, 0)
            name = type_names.get(type_id, f"type_{type_id}")
            collision_rate_per_type[name] = {
                "collisions": collision_count,
                "total_calls": total_calls,
                "rate": round(
                    collision_count / max(total_calls, 1), 4
                ),
            }

        # Efficiency
        steps_per_success = (
            float(self.total_steps)
            / max(self.total_goals_reached, 1)
        )

        # Episode stats (last 100)
        recent_rewards = (
            self._episode_rewards[-100:]
            if self._episode_rewards
            else [0]
        )
        recent_steps = (
            self._episode_steps[-100:]
            if self._episode_steps
            else [0]
        )

        # Finalize current mesh
        if self._current_mesh:
            self._mesh_stats[self._current_mesh] = {
                "episodes": self.total_episodes,
                "goals": self.total_goals_reached,
                "success_rate": (
                    self.total_goals_reached
                    / max(self.total_episodes, 1)
                ),
            }

        return {
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "total_goals_reached": self.total_goals_reached,
            "success_rate": (
                self.total_goals_reached
                / max(self.total_episodes, 1)
            ),
            "steps_per_success": round(steps_per_success, 1),
            "action_distribution": action_distribution,
            "collision_stats": {
                type_names.get(k, f"type_{k}"): v
                for k, v in self._collision_counts.items()
            },
            "collision_rate_per_type": collision_rate_per_type,
            "mean_episode_reward": round(
                float(np.mean(recent_rewards)), 2
            ),
            "mean_episode_steps": round(
                float(np.mean(recent_steps)), 1
            ),
            "critic_loss_avg": round(
                float(np.mean(self._critic_losses))
                if self._critic_losses
                else 0,
                4,
            ),
            "actor_loss_avg": round(
                float(np.mean(self._actor_losses))
                if self._actor_losses
                else 0,
                4,
            ),
            "alpha_type": round(self.alpha_type, 4),
            "alpha_param": round(self.alpha_param, 4),
            "bc_lambda": round(self.bc_lambda, 6),
            "buffer_size": len(self.buffer),
            "mesh_stats": dict(self._mesh_stats),
        }

    def train(  # noqa: PLR0913, C901, PLR0912, PLR0915
        self,
        env: LightweightEnv,
        controller: RLGoalApproachController,
        num_episodes: int = 5000,
        update_every: int = 1,
        updates_per_step: int = 1,
        warmup_steps: int = 1000,
        log_interval: int = 100,
        save_dir: str | None = None,
        curriculum_levels: list[tuple[float, float]] | None = None,
        promote_threshold: float = 0.5,
        promote_window: int = 100,
        episode_pools: dict[str, Any] | None = None,
    ) -> None:
        """Run SAC training loop.

        Args:
            env: Environment instance.
            controller: RL controller for state computation.
            num_episodes: Number of training episodes.
            update_every: Update networks every N steps.
            updates_per_step: Number of gradient updates per step.
            warmup_steps: Steps before starting updates.
            log_interval: Log every N episodes.
            save_dir: Directory to save best model.
            curriculum_levels: Optional curriculum levels.
            promote_threshold: Success rate threshold for promotion.
            promote_window: Window size for promotion check.
            episode_pools: Optional fixed episode pools.
        """
        interpreter = ActionInterpreter(env)

        curr_level = 0
        success_window: list[bool] = []
        rolling_history: deque[bool] = deque(maxlen=200)
        best_rolling_rate = 0.0
        best_eval_rate = 0.0
        best_state_dict = None
        best_critic_dict = None
        best_critic_target_dict = None
        best_extra = None

        for episode in range(num_episodes):
            if curriculum_levels:
                min_dist, max_dist = curriculum_levels[curr_level]
            else:
                min_dist, max_dist = 10.0, 120.0

            ep_data = self._get_episode_data(
                episode_pools, curr_level, episode
            )

            if ep_data is not None:
                start_pos = np.array(ep_data["start_pos"])
                start_rot = np.array(ep_data["start_rot"])
                env.reset(position=start_pos, rotation=start_rot)
                goal_pose = np.concatenate([
                    np.array(ep_data["goal_pos"]),
                    np.array(ep_data["goal_rot"]),
                ])
            else:
                env.reset()
                start_pos = env.get_pose()[:3]
                goal_pose = env.get_random_surface_point(
                    reference_pos=start_pos,
                    min_dist=min_dist,
                    max_dist=max_dist,
                    max_attempts=2000,
                    mesh_sample=True,
                )

            controller.set_new_goal(goal_pose, env.get_pose()[:3])
            env.set_goal(goal_pose)

            current_pose = env.get_pose()
            sensor_data = env.get_sensor_data()
            state_raw = self.compute_state(
                env, controller, current_pose, sensor_data
            )
            state = self.normalize_state(state_raw)
            prev_distance = float(
                np.linalg.norm(goal_pose[:3] - current_pose[:3])
            )

            episode_reward = 0.0
            episode_success = False

            for step in range(self.max_steps_per_goal):
                self.total_steps += 1

                state_t = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    at, ap, _, _ = self.actor.sample(state_t)
                action_type = at[0].item()
                action_params = (
                    ap[0].numpy() * self.param_std + self.param_mean
                )

                # Track action
                self._action_type_counts[action_type] = (
                    self._action_type_counts.get(action_type, 0) + 1
                )

                sensor_data = interpreter.execute(
                    action_type, action_params
                )
                current_pose = env.get_pose()
                next_state_raw = self.compute_state(
                    env, controller, current_pose, sensor_data
                )
                next_state = self.normalize_state(next_state_raw)

                distance = float(
                    np.linalg.norm(goal_pose[:3] - current_pose[:3])
                )

                collision = None
                depth = sensor_data.get("depth", 100.0)
                if depth < 0.5:
                    collision = "surface_violation"
                    # Track collision
                    self._collision_counts[action_type] = (
                        self._collision_counts.get(action_type, 0)
                        + 1
                    )

                reward, done = self.compute_reward(
                    next_state, state, distance, prev_distance,
                    collision, step + 1,
                    sensor_data=sensor_data,
                )

                action_params_norm = (
                    (action_params - self.param_mean)
                    / (self.param_std + 1e-8)
                )
                self.buffer.add(
                    state, action_type, action_params_norm,
                    reward, next_state, done,
                )

                episode_reward += reward
                state = next_state
                prev_distance = distance

                if (
                    self.total_steps >= warmup_steps
                    and self.total_steps % update_every == 0
                    and len(self.buffer) >= self.batch_size
                ):
                    for _ in range(updates_per_step):
                        batch = self.buffer.sample(self.batch_size)
                        critic_loss = self.update_critic(batch)
                        self._critic_losses.append(critic_loss)
                        if self.total_steps % 10 == 0:
                            sac_loss, _bc_loss = self.update_actor(
                                batch
                            )
                            self._actor_losses.append(sac_loss)
                            self.update_alpha(batch)
                        self.soft_update_target()

                if done:
                    if distance < self.goal_threshold:
                        self.total_goals_reached += 1
                        episode_success = True
                    break

            self.total_episodes += 1
            self._episode_rewards.append(episode_reward)
            self._episode_steps.append(step + 1)

            rolling_history.append(episode_success)
            if len(rolling_history) >= 50:
                rolling_rate = (
                    sum(rolling_history) / len(rolling_history)
                )
                best_rolling_rate = max(
                    best_rolling_rate, rolling_rate
                )
            else:
                rolling_rate = 0.0

            if (
                (episode + 1) % self.eval_interval == 0
                and self.total_steps >= warmup_steps
            ):
                eval_rate = self._run_eval_during_training(
                    env=env,
                    controller=controller,
                    interpreter=interpreter,
                    curriculum_levels=curriculum_levels,
                    num_episodes=self.eval_episodes,
                )
                logger.info(
                    "  EVAL at episode %d: success_rate=%.3f "
                    "(best=%.3f)",
                    episode + 1,
                    eval_rate,
                    best_eval_rate,
                )
                if eval_rate > best_eval_rate:
                    best_eval_rate = eval_rate
                    best_state_dict = {
                        k: v.clone()
                        for k, v in self.actor.state_dict().items()
                    }
                    best_critic_dict = {
                        k: v.clone()
                        for k, v in self.critic.state_dict().items()
                    }
                    best_critic_target_dict = {
                        k: v.clone()
                        for k, v in (
                            self.critic_target.state_dict().items()
                        )
                    }
                    best_extra = {
                        "total_steps": self.total_steps,
                        "total_episodes": self.total_episodes,
                        "total_goals_reached": (
                            self.total_goals_reached
                        ),
                        "bc_lambda": self.bc_lambda,
                        "log_alpha_type": (
                            self.log_alpha_type.detach().clone()
                        ),
                        "log_alpha_param": (
                            self.log_alpha_param.detach().clone()
                        ),
                        "eval_rate": eval_rate,
                    }
                    logger.info(
                        "  New best eval model! rate=%.3f",
                        eval_rate,
                    )

            if curriculum_levels:
                success_window.append(episode_success)
                if len(success_window) > promote_window:
                    success_window.pop(0)
                if (
                    len(success_window) == promote_window
                    and curr_level < len(curriculum_levels) - 1
                ):
                    rate = sum(success_window) / promote_window
                    if rate >= promote_threshold:
                        curr_level += 1
                        success_window = []
                        logger.info(
                            "SAC Curriculum: promoted to level %d "
                            "(%smm)",
                            curr_level,
                            curriculum_levels[curr_level],
                        )

            if (episode + 1) % log_interval == 0:
                success_rate = (
                    self.total_goals_reached
                    / max(self.total_episodes, 1)
                )
                level_info = (
                    f", level={curr_level}"
                    if curriculum_levels
                    else ""
                )
                logger.info(
                    "Episode %d/%d: reward=%.1f, steps=%d, "
                    "rate=%d/%d (%.3f), rolling=%.3f, "
                    "best_rolling=%.3f, best_eval=%.3f, "
                    "bc_lambda=%.4f, alpha_type=%.3f, "
                    "alpha_param=%.3f, buffer=%d%s",
                    episode + 1,
                    num_episodes,
                    episode_reward,
                    step + 1,
                    self.total_goals_reached,
                    self.total_episodes,
                    success_rate,
                    rolling_rate,
                    best_rolling_rate,
                    best_eval_rate,
                    self.bc_lambda,
                    self.alpha_type,
                    self.alpha_param,
                    len(self.buffer),
                    level_info,
                )

        if best_state_dict is not None:
            self.actor.load_state_dict(best_state_dict)
            self.critic.load_state_dict(best_critic_dict)
            self.critic_target.load_state_dict(
                best_critic_target_dict
            )
            self.total_steps = best_extra["total_steps"]
            self.total_episodes = best_extra["total_episodes"]
            self.total_goals_reached = best_extra[
                "total_goals_reached"
            ]
            self.bc_lambda = best_extra["bc_lambda"]
            self.log_alpha_type = torch.tensor(
                best_extra["log_alpha_type"].item(),
                requires_grad=True,
            )
            self.log_alpha_param = torch.tensor(
                best_extra["log_alpha_param"].item(),
                requires_grad=True,
            )
            logger.info(
                "Restored best model (eval_rate=%.3f)",
                best_extra["eval_rate"],
            )
        else:
            logger.info("No eval checkpoint was saved")

        if save_dir:
            self.save(save_dir)

    def save(self, dirpath: str) -> None:
        """Save SAC model to directory.

        Args:
            dirpath: Directory path.
        """
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)

        torch.save(
            self.actor.state_dict(), dirpath / "sac_actor.pt"
        )
        torch.save(
            self.critic.state_dict(), dirpath / "sac_critic.pt"
        )
        torch.save(
            self.critic_target.state_dict(),
            dirpath / "sac_critic_target.pt",
        )

        np.savez(
            dirpath / "sac_state.npz",
            state_mean=(
                self.state_mean
                if self.state_mean is not None
                else np.zeros(self.state_dim)
            ),
            state_std=(
                self.state_std
                if self.state_std is not None
                else np.ones(self.state_dim)
            ),
            param_mean=(
                self.param_mean
                if self.param_mean is not None
                else np.zeros(3)
            ),
            param_std=(
                self.param_std
                if self.param_std is not None
                else np.ones(3)
            ),
            total_steps=self.total_steps,
            total_episodes=self.total_episodes,
            total_goals_reached=self.total_goals_reached,
            bc_lambda=self.bc_lambda,
            log_alpha_type=self.log_alpha_type.detach().numpy(),
            log_alpha_param=self.log_alpha_param.detach().numpy(),
        )

        logger.info("P-SAC model saved to %s", dirpath)

    def load(self, dirpath: str) -> None:
        """Load SAC model from directory.

        Args:
            dirpath: Directory path.
        """
        dirpath = Path(dirpath)

        self.actor.load_state_dict(
            torch.load(
                dirpath / "sac_actor.pt", weights_only=True
            )
        )
        self.critic.load_state_dict(
            torch.load(
                dirpath / "sac_critic.pt", weights_only=True
            )
        )
        self.critic_target.load_state_dict(
            torch.load(
                dirpath / "sac_critic_target.pt",
                weights_only=True,
            )
        )

        data = np.load(dirpath / "sac_state.npz")
        self.state_mean = data["state_mean"]
        self.state_std = data["state_std"]
        self.total_steps = int(data["total_steps"])
        self.total_episodes = int(data["total_episodes"])
        self.total_goals_reached = int(data["total_goals_reached"])
        self.bc_lambda = float(data["bc_lambda"])
        self.log_alpha_type = torch.tensor(
            float(data["log_alpha_type"]), requires_grad=True
        )
        self.log_alpha_param = torch.tensor(
            float(data["log_alpha_param"]), requires_grad=True
        )
        self.param_mean = (
            data["param_mean"]
            if "param_mean" in data
            else np.zeros(3)
        )
        self.param_std = (
            data["param_std"]
            if "param_std" in data
            else np.ones(3)
        )

        logger.info("P-SAC model loaded from %s", dirpath)
