"""
P-SAC Training Loop.
Combines Actor, Twin Critic, Replay Buffer, ActionInterpreter.
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List
from copy import deepcopy

from .sac_actor import SACActorNetwork
from .twin_critic import TwinCritic
from .replay_buffer import ReplayBuffer
from .action_interpreter import ActionInterpreter
from .experience_extractor import ExperienceExtractor
from .lightweight_env import LightweightEnv
from .rl_goal_approach_controller import RLGoalApproachController

logger = logging.getLogger(__name__)


class PSACTrainer:

    def __init__(
        self,
        state_dim: int = 15,
        num_types: int = 8,
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
    ):
        self.state_dim = state_dim
        self.num_types = num_types
        self.max_params = max_params
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_steps_per_goal = max_steps_per_goal
        self.goal_threshold = goal_threshold

        self.actor = SACActorNetwork(state_dim, num_types)
        self.critic = TwinCritic(state_dim, num_types, max_params)
        self.critic_target = deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.log_alpha_type = torch.tensor(np.log(alpha_type_init), requires_grad=True)
        self.log_alpha_param = torch.tensor(np.log(alpha_param_init), requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha_type, self.log_alpha_param], lr=lr_alpha
        )

        self.target_entropy_type = -np.log(1.0 / num_types) * 0.5
        self.target_entropy_param = -max_params * 0.5

        self.buffer = ReplayBuffer(buffer_capacity, state_dim, max_params)

        self.bc_lambda = bc_lambda_init
        self.bc_lambda_decay = bc_lambda_decay
        self.bc_data = None

        self.state_mean = None
        self.state_std = None

        self.total_steps = 0
        self.total_episodes = 0
        self.total_goals_reached = 0

    @property
    def alpha_type(self):
        return self.log_alpha_type.exp().item()

    @property
    def alpha_param(self):
        return self.log_alpha_param.exp().item()

    def load_bc(self, bc_model_dir: str, bc_data_path: str):
        bc_state_dict = torch.load(
            Path(bc_model_dir) / "bc_actor.pt", weights_only=True
        )
        self.actor.load_bc_weights(bc_state_dict)

        norm = np.load(Path(bc_model_dir) / "bc_normalization.npz")
        self.state_mean = norm["state_mean"]
        self.state_std = norm["state_std"]

        with open(bc_data_path, "rb") as f:
            bc_transitions = pickle.load(f)
        self.buffer.load_bc_data(bc_transitions)
        self.bc_data = bc_transitions

        logger.info(
            f"Loaded BC: actor weights, normalization, "
            f"{len(bc_transitions)} transitions"
        )

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        if self.state_mean is not None:
            return (state - self.state_mean) / (self.state_std + 1e-8)
        return state

    def compute_state(self, env, controller, current_pose, sensor_data):
        return controller._compute_state(current_pose, sensor_data)

    def compute_reward(self, state, prev_state, distance, prev_distance, collision, steps):
        reward = 0.0
        done = False

        surface_step = 3.0
        progress = prev_distance - distance
        reward += progress / surface_step * 3.0

        if distance < self.goal_threshold:
            reward += 60.0
            done = True

        reward += -0.2

        if collision == "surface_violation":
            reward += -5.0
            done = True

        if steps >= self.max_steps_per_goal:
            reward += -8.0
            done = True

        return reward, done

    def update_critic(self, batch: Dict[str, np.ndarray]):
        states = torch.FloatTensor(batch["states"])
        action_types = torch.LongTensor(batch["action_types"])
        action_params = torch.FloatTensor(batch["action_params"])
        rewards = torch.FloatTensor(batch["rewards"])
        next_states = torch.FloatTensor(batch["next_states"])
        dones = torch.FloatTensor(batch["dones"])

        with torch.no_grad():
            next_type, next_params, next_log_prob, _ = self.actor.sample(next_states)
            next_q = self.critic_target.min_q(next_states, next_type, next_params)
            alpha = self.alpha_type + self.alpha_param
            target_q = rewards + self.gamma * (1 - dones) * (next_q - alpha * next_log_prob)

        q1, q2 = self.critic(states, action_types, action_params)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()

    def update_actor(self, batch: Dict[str, np.ndarray]):
        states = torch.FloatTensor(batch["states"])

        action_type, action_params, log_prob, type_probs = self.actor.sample(states)
        q_val = self.critic.min_q(states, action_type, action_params)

        sac_loss = (
            (self.alpha_type + self.alpha_param) * log_prob - q_val
        ).mean()

        bc_loss = torch.tensor(0.0)
        if self.bc_lambda > 0.01 and self.bc_data is not None:
            bc_batch_size = min(64, len(self.bc_data))
            bc_indices = np.random.randint(0, len(self.bc_data), bc_batch_size)
            bc_states = torch.FloatTensor(
                np.array([
                    self.normalize_state(self.bc_data[i].state)
                    for i in bc_indices
                ])
            )
            bc_types = torch.LongTensor(
                [self.bc_data[i].action_type for i in bc_indices]
            )
            bc_params = torch.FloatTensor(
                np.array([self.bc_data[i].action_params for i in bc_indices])
            )

            type_logits, param_mus, _ = self.actor(bc_states)
            type_loss = F.cross_entropy(type_logits, bc_types)

            param_loss = torch.tensor(0.0)
            param_dims = ExperienceExtractor.get_param_dims()
            for type_id in range(self.num_types):
                dim = param_dims[type_id]
                if dim == 0:
                    continue
                mask = (bc_types == type_id)
                if mask.sum() == 0:
                    continue
                pred = param_mus[type_id][mask]
                target = bc_params[mask, :dim]
                param_loss = param_loss + F.mse_loss(pred, target)

            bc_loss = type_loss + param_loss

        actor_loss = sac_loss + self.bc_lambda * bc_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.bc_lambda *= self.bc_lambda_decay

        return sac_loss.item(), bc_loss.item()

    def update_alpha(self, batch: Dict[str, np.ndarray]):
        states = torch.FloatTensor(batch["states"])

        with torch.no_grad():
            _, _, log_prob, type_probs = self.actor.sample(states)
            type_entropy = -(type_probs * torch.log(type_probs + 1e-8)).sum(dim=-1).mean()

        alpha_type_loss = -(
            self.log_alpha_type * (type_entropy - self.target_entropy_type).detach()
        )
        alpha_param_loss = -(
            self.log_alpha_param * (-log_prob.mean() - self.target_entropy_param).detach()
        )
        alpha_loss = alpha_type_loss + alpha_param_loss

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

    def soft_update_target(self):
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def train(
        self,
        env: LightweightEnv,
        controller: RLGoalApproachController,
        num_episodes: int = 5000,
        update_every: int = 1,
        updates_per_step: int = 1,
        warmup_steps: int = 1000,
        log_interval: int = 100,
        save_dir: Optional[str] = None,
    ):
        interpreter = ActionInterpreter(env)

        for episode in range(num_episodes):
            env.reset()
            start_pos = env.get_pose()[:3]
            goal_pose = env.get_random_surface_point(
                reference_pos=start_pos,
                min_dist=10.0,
                max_dist=120.0,
                max_attempts=2000,
                mesh_sample=True,
            )
            controller.set_new_goal(goal_pose, start_pos)

            current_pose = env.get_pose()
            sensor_data = env.get_sensor_data()
            state_raw = self.compute_state(env, controller, current_pose, sensor_data)
            state = self.normalize_state(state_raw)
            prev_distance = float(np.linalg.norm(goal_pose[:3] - current_pose[:3]))

            episode_reward = 0.0

            for step in range(self.max_steps_per_goal):
                self.total_steps += 1

                if self.total_steps < warmup_steps:
                    action_type = np.random.randint(0, self.num_types)
                    action_params = np.random.randn(self.max_params).astype(np.float32) * 5.0
                else:
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    with torch.no_grad():
                        at, ap, _, _ = self.actor.sample(state_t)
                    action_type = at[0].item()
                    action_params = ap[0].numpy()

                sensor_data = interpreter.execute(action_type, action_params)
                current_pose = env.get_pose()
                next_state_raw = self.compute_state(
                    env, controller, current_pose, sensor_data
                )
                next_state = self.normalize_state(next_state_raw)

                distance = float(np.linalg.norm(goal_pose[:3] - current_pose[:3]))

                collision = None
                depth = sensor_data.get("depth", 100.0)
                if depth < 0.5:
                    collision = "surface_violation"

                reward, done = self.compute_reward(
                    next_state, state, distance, prev_distance, collision, step + 1
                )

                self.buffer.add(state, action_type, action_params, reward, next_state, done)

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
                        sac_loss, bc_loss = self.update_actor(batch)
                        self.update_alpha(batch)
                        self.soft_update_target()

                if done:
                    if distance < self.goal_threshold:
                        self.total_goals_reached += 1
                    break

            self.total_episodes += 1

            if (episode + 1) % log_interval == 0:
                success_rate = self.total_goals_reached / max(self.total_episodes, 1)
                logger.info(
                    f"Episode {episode+1}/{num_episodes}: "
                    f"reward={episode_reward:.1f}, "
                    f"steps={step+1}, "
                    f"success_rate={self.total_goals_reached}/{self.total_episodes} "
                    f"({success_rate:.3f}), "
                    f"bc_lambda={self.bc_lambda:.4f}, "
                    f"alpha_type={self.alpha_type:.3f}, "
                    f"alpha_param={self.alpha_param:.3f}, "
                    f"buffer={len(self.buffer)}"
                )

        if save_dir:
            self.save(save_dir)

    def save(self, dirpath: str):
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)

        torch.save(self.actor.state_dict(), dirpath / "sac_actor.pt")
        torch.save(self.critic.state_dict(), dirpath / "sac_critic.pt")
        torch.save(self.critic_target.state_dict(), dirpath / "sac_critic_target.pt")

        np.savez(
            dirpath / "sac_state.npz",
            state_mean=self.state_mean if self.state_mean is not None else np.zeros(self.state_dim),
            state_std=self.state_std if self.state_std is not None else np.ones(self.state_dim),
            total_steps=self.total_steps,
            total_episodes=self.total_episodes,
            total_goals_reached=self.total_goals_reached,
            bc_lambda=self.bc_lambda,
            log_alpha_type=self.log_alpha_type.detach().numpy(),
            log_alpha_param=self.log_alpha_param.detach().numpy(),
        )

        logger.info(f"P-SAC model saved to {dirpath}")

    def load(self, dirpath: str):
        dirpath = Path(dirpath)

        self.actor.load_state_dict(
            torch.load(dirpath / "sac_actor.pt", weights_only=True)
        )
        self.critic.load_state_dict(
            torch.load(dirpath / "sac_critic.pt", weights_only=True)
        )
        self.critic_target.load_state_dict(
            torch.load(dirpath / "sac_critic_target.pt", weights_only=True)
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

        logger.info(f"P-SAC model loaded from {dirpath}")
