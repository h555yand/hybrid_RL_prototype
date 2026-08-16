# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""AdaptiveTrainingManager: monitors performance and decides
when/how to train for new objects.

Modes:
  - inference_only: success_rate > 80%, no training needed
  - online: 60-80%, update Q-store + accumulate for SAC
  - offline: < 60%, selective retraining:
      - Q-store: always retrain with decreasing epsilon (0.6 → 0.3 → 0.1 → 0.05)
      - SAC: retrain only if SAC success rate is below threshold
"""

import logging
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .ablation_runner import run_episodes
from .arbitrator import Arbitrator
from .behavioral_cloning import BCTrainer
from .experience_extractor import ExperienceExtractor
from .lightweight_env import LightweightEnv
from .rl_goal_approach_controller import RLGoalApproachController
from .sac_trainer import PSACTrainer

logger = logging.getLogger(__name__)


class AdaptiveTrainingManager:

    OFFLINE_Q_EPSILON_SCHEDULE = [0.6, 0.3, 0.1, 0.05]

    def __init__(
        self,
        controller: RLGoalApproachController,
        env: LightweightEnv,
        config: Dict[str, Any],
        runs_dir: str,
        mesh_path: str,
        online_threshold: float = 0.80,
        offline_threshold: float = 0.40,
        sac_offline_threshold: float = 0.50,
        monitor_window: int = 100,
        online_sac_update_every: int = 500,
        online_sac_update_steps: int = 20,
        online_bc_update_every: int = 2000,
        offline_q_episodes: int = 5000,
        offline_sac_episodes: int = 2000,
        post_offline_cooldown: int = 200,
    ):
        self.controller = controller
        self.env = env
        self.config = config
        self.runs_dir = Path(runs_dir)
        self.mesh_path = mesh_path

        self.online_threshold = online_threshold
        self.offline_threshold = offline_threshold
        self.sac_offline_threshold = sac_offline_threshold
        self.monitor_window = monitor_window
        self.online_sac_update_every = online_sac_update_every
        self.online_sac_update_steps = online_sac_update_steps
        self.online_bc_update_every = online_bc_update_every
        self.offline_q_episodes = offline_q_episodes
        self.offline_sac_episodes = offline_sac_episodes
        self.post_offline_cooldown = post_offline_cooldown

        self.success_history = deque(maxlen=monitor_window)
        self.online_transitions_raw = []
        self.online_episodes_since_sac_update = 0
        self.online_episodes_since_bc_update = 0
        self.total_episodes = 0
        self.total_sac_updates = 0
        self.total_bc_updates = 0
        self.total_offline_iterations = 0
        self._episodes_since_offline = 0
        self.mode = "inference_only"

        self.sac_trainer = None
        self.action_space = controller.action_space
        self.extractor = ExperienceExtractor(config=config)
        self.arbitrator = Arbitrator(controller=controller)
        self.bc_transitions = []  # накопитель BC данных от всех offline итераций
        mesh_name = Path(mesh_path).stem
        self.extractor = ExperienceExtractor(config=config, mesh_name=mesh_name)

    @property
    def success_rate(self):
        if len(self.success_history) == 0:
            return 0.0
        return sum(self.success_history) / len(self.success_history)

    def get_action(self, state, current_pose, sensor_data):
        """Get action from arbitrator.

        Args:
            state: Current state vector.
            current_pose: Current agent pose.
            sensor_data: Current sensor readings.

        Returns:
            Tuple of (action_index, source_name).
        """
        if self.sac_trainer is not None:
            self.arbitrator.sac_actor = (
                self.sac_trainer.actor
            )
            self.arbitrator.state_mean = (
                self.sac_trainer.state_mean
            )
            self.arbitrator.state_std = (
                self.sac_trainer.state_std
            )
            self.arbitrator.param_mean = (
                self.sac_trainer.param_mean
            )
            self.arbitrator.param_std = (
                self.sac_trainer.param_std
            )

        return self.arbitrator.decide(
            state, current_pose, sensor_data
        )

    def decide_mode(self):
        """Decide operating mode based on performance.

        Modes:
        - online: Q-store updates every step, SAC updates
          periodically. Default mode — always learning.
        - inference_only: no updates. Only when consistently
          high success rate — system has mastered the object.
        - offline: full retraining. Only when both sources
          fail for extended period — emergency mode.

        Returns:
            Mode string: "online", "inference_only", or "offline".
        """
        if len(self.success_history) < self.monitor_window:
            return "online"

        if (
            self._episodes_since_offline
            < self.post_offline_cooldown
        ):
            return "online"

        rate = self.success_rate

        # Mastered the object — stop learning
        if rate >= self.online_threshold:
            return "inference_only"

        # Both sources failing for extended period
        if rate < self.offline_threshold:
            q_rate = self.arbitrator.q_success_rate
            sac_rate = self.arbitrator.sac_success_rate
            episodes_stuck = (
                self._count_episodes_below_threshold()
            )

            min_stuck = self.monitor_window * 3
            if (
                q_rate < self.offline_threshold
                and sac_rate < self.offline_threshold
                and episodes_stuck > min_stuck
            ):
                return "offline"

        # Default: keep learning online
        return "online"
    
    def _count_episodes_below_threshold(self) -> int:
        """Count recent consecutive failures in success history.

        Returns:
            Number of consecutive failures from the end.
        """
        count = 0
        for success in reversed(self.success_history):
            if not success:
                count += 1
            else:
                break
        return count
    
    def on_episode_complete(self, success: bool, transitions: List[Dict[str, Any]]):
        self.success_history.append(success)
        self.total_episodes += 1
        self._episodes_since_offline += 1
        self.online_episodes_since_sac_update += 1
        self.online_episodes_since_bc_update += 1

        old_mode = self.mode
        self.mode = self.decide_mode()

        if self.mode != old_mode:
            logger.info(
                f"AdaptiveTraining: mode changed {old_mode} → {self.mode} "
                f"(success_rate={self.success_rate:.3f}, "
                f"episodes={self.total_episodes}, "
                f"cooldown={self._episodes_since_offline}/{self.post_offline_cooldown})"
            )

        if self.mode == "inference_only":
            self.controller.mode = "eval"
            self.controller.epsilon = self.controller.eval_epsilon
            return

        if self.mode == "online":
            self.controller.mode = "train"
            self.controller.epsilon = self.config.get(
                "epsilon_start", 0.2
            )
            self.controller.strategic_epsilon = (
                self.config.get(
                    "strategic_epsilon_start", 0.2
                )
            )
            self._collect_transitions(success, transitions)
            self._maybe_online_sac_update()
            # Update Strategic SAC
            if (
                self.controller.strategic_sac is not None
                and self.controller.strategic_sac.buffer_size
                >= self.controller.strategic_sac.batch_size
                and self.total_episodes % 100 == 0
            ):
                stats = self.controller.strategic_sac.update(
                    num_steps=20
                )
                logger.debug(
                    "Strategic SAC update: %s", stats
                )

        if self.mode == "offline":
            self._trigger_offline()
            self.controller.mode = "train"
            self.controller.mode = "train"
            self.controller.epsilon = self.config.get(
                "epsilon_start", 0.2
            )
            self.controller.strategic_epsilon = (
                self.config.get(
                    "strategic_epsilon_start", 0.2
                )
            )
            self.mode = "online"

    def _collect_transitions(
        self,
        success: bool,
        transitions: List[Dict[str, Any]],
    ):
        """Collect transitions for SAC update.

        All transitions → online_transitions_raw
            (for buffer, critic needs negative examples)
        Successful only → bc_transitions
            (for BC regularization)

        Args:
            success: Whether the episode succeeded.
            transitions: Episode transitions.
        """
        if not transitions:
            return

        psac_transitions = (
            self.extractor.convert_trajectory(
                transitions
            )
        )

        # All transitions for buffer
        self.online_transitions_raw.extend(
            psac_transitions
        )

        # Only successful for BC regularization
        if success:
            self.bc_transitions.extend(
                psac_transitions
            )

    def _maybe_online_sac_update(self):
        """Online SAC update with CQL support.

        Every online_sac_update_every episodes:
        - Add ALL transitions to buffer (critic
          needs negative examples)
        - Update BC data with successful only
        - Update critic (CQL if enabled)
        - Update actor (only after warmup)
        """
        if self.sac_trainer is None:
            return

        if (
            self.online_episodes_since_sac_update
            < self.online_sac_update_every
        ):
            return

        if len(self.online_transitions_raw) < 100:
            logger.info(
                "SAC update skipped: only %d "
                "transitions",
                len(self.online_transitions_raw),
            )
            self.online_episodes_since_sac_update = 0
            return

        # Add ALL transitions to buffer
        all_normalized = (
            self.sac_trainer
            ._normalize_bc_transitions(
                self.online_transitions_raw
            )
        )
        added = 0
        for tr in all_normalized:
            if tr.next_state is not None:
                self.sac_trainer.buffer.add(
                    state=tr.state,
                    action_type=tr.action_type,
                    action_params=tr.action_params,
                    reward=tr.reward,
                    next_state=tr.next_state,
                    done=tr.done,
                )
                added += 1

        # Update BC data with successful only
        if self.bc_transitions:
            self.sac_trainer.bc_data = (
                self.bc_transitions.copy()
            )

        # Clear raw transitions
        self.online_transitions_raw = []

        if (
            len(self.sac_trainer.buffer)
            < self.sac_trainer.batch_size
        ):
            logger.info(
                "SAC update skipped: buffer too "
                "small (%d)",
                len(self.sac_trainer.buffer),
            )
            self.online_episodes_since_sac_update = 0
            return

        # Conservative update
        old_bc_lambda = self.sac_trainer.bc_lambda
        self.sac_trainer.bc_lambda = max(
            old_bc_lambda,
            self.sac_trainer.bc_lambda_min,
        )

        # Actor warmup: first 200 episodes
        # only critic
        actor_ready = self.total_episodes > 200

        for _ in range(
            self.online_sac_update_steps
        ):
            batch = self.sac_trainer.buffer.sample(
                self.sac_trainer.batch_size
            )

            # CQL or standard critic
            if self.sac_trainer.use_cql:
                self.sac_trainer.update_critic_cql(
                    batch
                )
            else:
                self.sac_trainer.update_critic(
                    batch
                )

            # Actor: only after warmup
            if actor_ready:
                self.sac_trainer.update_actor(
                    batch
                )

            self.sac_trainer.soft_update_target()

        self.sac_trainer.bc_lambda = old_bc_lambda
        self.total_sac_updates += 1
        self.online_episodes_since_sac_update = 0

        logger.info(
            "Online SAC update: added=%d, "
            "bc_data=%d, steps=%d, cql=%s, "
            "actor=%s, bc_lambda=%.2f, "
            "total_updates=%d",
            added,
            len(self.bc_transitions),
            self.online_sac_update_steps,
            self.sac_trainer.use_cql,
            actor_ready,
            self.sac_trainer.bc_lambda,
            self.total_sac_updates,
        )

    def _maybe_online_bc_update(self):
        if self.sac_trainer is None:
            return

        if self.online_episodes_since_bc_update < self.online_bc_update_every:
            return

        if self.sac_trainer.bc_data is not None and len(self.sac_trainer.bc_data) > 100:
            for _ in range(20):
                batch = self.sac_trainer.buffer.sample(self.sac_trainer.batch_size)
                self.sac_trainer.update_actor(batch)

            self.total_bc_updates += 1
            self.online_episodes_since_bc_update = 0

            logger.info(
                f"AdaptiveTraining: online BC update #{self.total_bc_updates} "
                f"(bc_data={len(self.sac_trainer.bc_data)})"
            )

    def _trigger_offline(self):
        """Trigger offline retraining when success rate is too low."""
        self.total_offline_iterations += 1

        eps_idx = min(
            self.total_offline_iterations - 1,
            len(self.OFFLINE_Q_EPSILON_SCHEDULE) - 1,
        )
        q_epsilon = self.OFFLINE_Q_EPSILON_SCHEDULE[eps_idx]

        logger.info(
            "AdaptiveTraining: triggering OFFLINE "
            "iteration #%d (success_rate=%.3f, "
            "q_epsilon=%.3f)",
            self.total_offline_iterations,
            self.success_rate,
            q_epsilon,
        )

        # === Q-learning: save current, then retrain ===
        q_save_dir = str(self.runs_dir / "adaptive_q")
        self.controller.save(q_save_dir)

        logger.info(
            "AdaptiveTraining: OFFLINE Q-learning "
            "(episodes=%d, epsilon=%.3f, load=%s)",
            self.offline_q_episodes,
            q_epsilon,
            q_save_dir,
        )

        from .ablation_runner import run_episodes

        train_result = run_episodes(
            mesh_dir=str(self.runs_dir.parent),
            save_dir=q_save_dir,
            load_dir=q_save_dir,
            num_episodes=self.offline_q_episodes,
            config={
                **self.config,
                "mode": "train_adapt_epsilon",
                "epsilon_start": q_epsilon,
            },
            mesh_path=self.mesh_path,
            seed=42,
            return_metrics=True,
        )

        # Reload updated Q-store into controller
        self.controller = RLGoalApproachController.load(
            q_save_dir,
            agent_id=self.controller.agent_id,
            config={**self.config, "mode": "eval"},
        )

        # Collect BC transitions from successful episodes
        success_trails = train_result.get(
            "success_trails", []
        )
        if success_trails:
            new_transitions = (
                self.extractor.convert_all_trajectories(
                    success_trails
                )
            )
            self.bc_transitions.extend(new_transitions)
            logger.info(
                "AdaptiveTraining: collected %d BC "
                "transitions (total=%d)",
                len(new_transitions),
                len(self.bc_transitions),
            )

        q_success_rate = train_result.get(
            "success_rate", 0.0
        )
        logger.info(
            "AdaptiveTraining: OFFLINE Q-learning "
            "complete (success_rate=%.3f)",
            q_success_rate,
        )

        # === SAC: fine-tune existing, not from scratch ===
        sac_sr = self.arbitrator.sac_success_rate
        need_sac_retrain = (
            self.sac_trainer is not None
            and sac_sr < self.sac_offline_threshold
            and self.total_offline_iterations > 2
        )

        if need_sac_retrain:
            logger.info(
                "AdaptiveTraining: OFFLINE SAC "
                "CQL retrain "
                "(sac_rate=%.3f, "
                "all_transitions=%d, "
                "bc_transitions=%d)",
                sac_sr,
                len(self.online_transitions_raw),
                len(self.bc_transitions),
            )

            # Add ALL accumulated transitions
            # to buffer
            if self.online_transitions_raw:
                all_normalized = (
                    self.sac_trainer
                    ._normalize_bc_transitions(
                        self.online_transitions_raw
                    )
                )
                for tr in all_normalized:
                    if tr.next_state is not None:
                        self.sac_trainer.buffer.add(
                            state=tr.state,
                            action_type=(
                                tr.action_type
                            ),
                            action_params=(
                                tr.action_params
                            ),
                            reward=tr.reward,
                            next_state=(
                                tr.next_state
                            ),
                            done=tr.done,
                        )

            # Update BC data with successful
            if self.bc_transitions:
                self.sac_trainer.bc_data = (
                    self.bc_transitions.copy()
                )

            # CQL offline retrain on buffer
            old_bc_lambda = (
                self.sac_trainer.bc_lambda
            )
            self.sac_trainer.bc_lambda = max(
                old_bc_lambda,
                self.sac_trainer.bc_lambda_min,
            )

            offline_steps = min(
                self.offline_sac_episodes * 10,
                5000,
            )
            warmup_steps = offline_steps // 2

            for step in range(offline_steps):
                if (
                    len(self.sac_trainer.buffer)
                    < self.sac_trainer.batch_size
                ):
                    break

                batch = (
                    self.sac_trainer.buffer.sample(
                        self.sac_trainer.batch_size
                    )
                )

                if self.sac_trainer.use_cql:
                    self.sac_trainer \
                        .update_critic_cql(batch)
                else:
                    self.sac_trainer \
                        .update_critic(batch)

                if step >= warmup_steps:
                    self.sac_trainer \
                        .update_actor(batch)

                self.sac_trainer \
                    .soft_update_target()

            self.sac_trainer.bc_lambda = (
                old_bc_lambda
            )

            # Save
            sac_save_dir = str(
                self.runs_dir / "adaptive_sac"
            )
            self.sac_trainer.save(sac_save_dir)

            # Clear raw transitions
            self.online_transitions_raw = []

            logger.info(
                "AdaptiveTraining: OFFLINE SAC "
                "CQL retrain complete "
                "(steps=%d, warmup=%d)",
                offline_steps,
                warmup_steps,
            )
        else:
            reason = (
                "sac_trainer is None"
                if self.sac_trainer is None
                else (
                    f"sac_rate {sac_sr:.3f} >= "
                    f"threshold "
                    f"{self.sac_offline_threshold}"
                    if sac_sr
                    >= self.sac_offline_threshold
                    else (
                        f"only "
                        f"{self.total_offline_iterations}"
                        " offline iterations "
                        "(need > 2)"
                    )
                )
            )
            logger.info(
                "AdaptiveTraining: OFFLINE SAC "
                "skipped — %s",
                reason,
            )
        
    def get_stats(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "success_rate": self.success_rate,
            "total_episodes": self.total_episodes,
            "history_size": len(self.success_history),
            "online_transitions_pending": len(self.online_transitions_raw),
            "sac_loaded": self.sac_trainer is not None,
            "total_sac_updates": self.total_sac_updates,
            "total_bc_updates": self.total_bc_updates,
            "total_offline_iterations": self.total_offline_iterations,
            "episodes_since_sac_update": self.online_episodes_since_sac_update,
            "episodes_since_bc_update": self.online_episodes_since_bc_update,
            "episodes_since_offline": self._episodes_since_offline,
        }
