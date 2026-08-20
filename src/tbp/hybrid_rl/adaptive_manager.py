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
  - mastered: success_rate > 80%, light tuning (Q alpha*0.1, eps=0.02)
  - online: 40-80%, full Q learning + periodic SAC updates,
    adaptive epsilon based on success rate
  - offline: < 40% for extended period, Q retrain (500 ep, eps 1→0.3)
    + SAC CQL retrain, then switch to online
"""

import logging
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .ablation_runner import run_episodes
from .arbitrator import Arbitrator
from .experience_extractor import ExperienceExtractor
from .lightweight_env import LightweightEnv
from .rl_goal_approach_controller import RLGoalApproachController
from .sac_trainer import PSACTrainer

logger = logging.getLogger(__name__)


class AdaptiveTrainingManager:

    def __init__(
        self,
        controller: RLGoalApproachController,
        env: LightweightEnv,
        config: Dict[str, Any],
        runs_dir: str,
        mesh_path: str,
        mastered_threshold: float = 0.80,
        offline_threshold: float = 0.50,
        monitor_window: int = 100,
        online_sac_update_every: int = 200,
        online_sac_update_steps: int = 20,
        offline_q_episodes: int = 500,
        offline_sac_steps: int = 2000,
        post_offline_cooldown: int = 200,
        max_bc_transitions: int = 5000,
        epsilon_warmup_episodes: int = 20,
        epsilon_default: float = 0.15,
    ):
        self.controller = controller
        self.env = env
        self.config = config
        self.runs_dir = Path(runs_dir)
        self.mesh_path = mesh_path

        # Mode thresholds
        self.mastered_threshold = mastered_threshold
        self.offline_threshold = offline_threshold
        self.monitor_window = monitor_window

        # Online SAC updates
        self.online_sac_update_every = online_sac_update_every
        self.online_sac_update_steps = online_sac_update_steps

        # Offline retrain params
        self.offline_q_episodes = offline_q_episodes
        self.offline_q_epsilon_start = 1.0
        self.offline_q_epsilon_min = 0.3
        self.offline_sac_steps = offline_sac_steps
        self.post_offline_cooldown = post_offline_cooldown

        # BC data management
        self.max_bc_transitions = max_bc_transitions
        self.bc_transitions: List = []

        # Adaptive epsilon
        self.epsilon_warmup_episodes = epsilon_warmup_episodes
        self.epsilon_default = epsilon_default

        # Performance tracking
        self.success_history: deque = deque(maxlen=monitor_window)
        self.online_transitions_raw: List = []
        self.online_episodes_since_sac_update = 0
        self.total_episodes = 0
        self.total_sac_updates = 0
        self.total_offline_iterations = 0
        self._episodes_since_offline = 0
        self.mode = "online"

        # Components
        self.sac_trainer: PSACTrainer | None = None
        self.action_space = controller.action_space
        mesh_name = Path(mesh_path).stem
        self.extractor = ExperienceExtractor(
            config=config, mesh_name=mesh_name
        )
        self.arbitrator = Arbitrator(controller=controller)

    @property
    def success_rate(self) -> float:
        if len(self.success_history) == 0:
            return 0.0
        return sum(self.success_history) / len(
            self.success_history
        )

    def _get_adaptive_epsilon(self) -> float:
        """Compute epsilon based on rolling success rate.

        Returns moderate default during warmup, then adapts:
        high success → low epsilon (exploit),
        low success → high epsilon (explore).
        """
        if len(self.success_history) < self.epsilon_warmup_episodes:
            return self.epsilon_default
        rate = self.success_rate
        return 0.05 + 0.3 * (1.0 - rate)

    def get_action(self, state, current_pose, sensor_data):
        """Get action from arbitrator.

        Returns:
            Tuple of (action_type, action_params, source_name).
        """
        if self.sac_trainer is not None:
            self.arbitrator.sac_actor = self.sac_trainer.actor
            self.arbitrator.state_mean = self.sac_trainer.state_mean
            self.arbitrator.state_std = self.sac_trainer.state_std
            self.arbitrator.param_mean = self.sac_trainer.param_mean
            self.arbitrator.param_std = self.sac_trainer.param_std

        return self.arbitrator.decide(
            state, current_pose, sensor_data
        )

    def decide_mode(self) -> str:
        """Decide operating mode based on performance."""
        if len(self.success_history) < self.monitor_window:
            return "online"

        if self._episodes_since_offline < self.post_offline_cooldown:
            return "online"

        rate = self.success_rate

        if rate >= self.mastered_threshold:
            return "mastered"

        if rate < self.offline_threshold:
            return "offline"
    
    def _count_episodes_below_threshold(self) -> int:
        """Count recent consecutive failures from end of history."""
        count = 0
        for success in reversed(self.success_history):
            if not success:
                count += 1
            else:
                break
        return count

    def on_episode_complete(
        self,
        success: bool,
        transitions: List[Dict[str, Any]],
    ) -> None:
        """Process episode completion: update mode, collect data, trigger updates.

        Args:
            success: Whether episode reached goal.
            transitions: Episode transitions for SAC training.
        """
        self.success_history.append(success)
        self.total_episodes += 1
        self._episodes_since_offline += 1
        self.online_episodes_since_sac_update += 1

        old_mode = self.mode
        self.mode = self.decide_mode()

        if self.mode != old_mode:
            logger.info(
                "AdaptiveTraining: mode %s → %s "
                "(rate=%.3f, episodes=%d)",
                old_mode,
                self.mode,
                self.success_rate,
                self.total_episodes,
            )

        if self.mode == "mastered":
            # Light tuning: Q updates with reduced alpha (eval mode)
            self.controller.mode = "adaptive"
            self.controller.epsilon = 0.02
            self.controller.strategic_epsilon = 0.02
            return

        if self.mode == "online":
            self.controller.mode = "adaptive"
            eps = self._get_adaptive_epsilon()
            self.controller.epsilon = eps
            self.controller.strategic_epsilon = eps
            self._collect_transitions(success, transitions)
            self._maybe_online_sac_update()
            return

        if self.mode == "offline":
            self._trigger_offline()
            self.controller.mode = "train"
            eps = self._get_adaptive_epsilon()
            self.controller.epsilon = eps
            self.controller.strategic_epsilon = eps
            self.mode = "online"

    def _collect_transitions(
        self,
        success: bool,
        transitions: List[Dict[str, Any]],
    ) -> None:
        """Collect transitions for SAC training.

        Success transitions always collected (for buffer + BC).
        Failure transitions collected with 30% probability
        (critic needs some negatives, but not too many).
        """
        if not transitions:
            return

        psac_transitions = self.extractor.convert_trajectory(
            transitions
        )

        if success:
            # Always collect successful transitions
            self.online_transitions_raw.extend(psac_transitions)
            self.bc_transitions.extend(psac_transitions)
            if len(self.bc_transitions) > self.max_bc_transitions:
                self.bc_transitions = self.bc_transitions[
                    -self.max_bc_transitions :
                ]
        else:
            # Collect 30% of failure transitions (critic needs some negatives)
            if np.random.random() < 0.3:
                self.online_transitions_raw.extend(psac_transitions)

    def _maybe_online_sac_update(self) -> None:
        """Periodic online SAC update: critic only.

        Actor is NOT updated online to prevent catastrophic forgetting.
        Actor updates happen only in offline mode with curated BC data
        and full training protections (warmup, CQL, strong BC lambda).
        """
        if self.sac_trainer is None:
            return

        if (
            self.online_episodes_since_sac_update
            < self.online_sac_update_every
        ):
            return

        if len(self.online_transitions_raw) < 100:
            self.online_episodes_since_sac_update = 0
            return

        # Add transitions to buffer
        all_normalized = (
            self.sac_trainer._normalize_bc_transitions(
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

        # Update BC data
        if self.bc_transitions:
            self.sac_trainer.bc_data = self.bc_transitions.copy()

        self.online_transitions_raw = []

        if len(self.sac_trainer.buffer) < self.sac_trainer.batch_size:
            self.online_episodes_since_sac_update = 0
            return

        # Online: critic only (safe, no catastrophic forgetting)
        for _ in range(self.online_sac_update_steps):
            batch = self.sac_trainer.buffer.sample(
                self.sac_trainer.batch_size
            )
            self.sac_trainer.update_critic_cql(batch)
            self.sac_trainer.soft_update_target()

        self.total_sac_updates += 1
        self.online_episodes_since_sac_update = 0

        logger.info(
            "Online SAC update (critic only): added=%d, "
            "bc=%d, steps=%d, total=%d",
            added,
            len(self.bc_transitions),
            self.online_sac_update_steps,
            self.total_sac_updates,
        )

    def _trigger_offline(self) -> None:
        """Offline retraining: Q 500 ep + SAC with training protections."""
        self.total_offline_iterations += 1
        self._episodes_since_offline = 0

        logger.info(
            "AdaptiveTraining: OFFLINE #%d (rate=%.3f)",
            self.total_offline_iterations,
            self.success_rate,
        )

        # === Phase 1: Q-learning retrain ===
        q_save_dir = str(self.runs_dir / "adaptive_q")
        self.controller.save(q_save_dir)

        logger.info(
            "OFFLINE Q: %d episodes, eps %.1f→%.1f",
            self.offline_q_episodes,
            self.offline_q_epsilon_start,
            self.offline_q_epsilon_min,
        )

        train_result = run_episodes(
            mesh_dir=str(self.runs_dir.parent),
            save_dir=q_save_dir,
            load_dir=q_save_dir,
            num_episodes=self.offline_q_episodes,
            config={
                **self.config,
                "mode": "train_adapt_epsilon",
                "epsilon_start": self.offline_q_epsilon_start,
                "epsilon_min": self.offline_q_epsilon_min,
                "num_episodes": self.offline_q_episodes,
            },
            mesh_path=self.mesh_path,
            seed=42,
            return_metrics=True,
        )

        # Reload controller and fix all references
        self.controller = RLGoalApproachController.load(
            q_save_dir,
            agent_id=self.controller.agent_id,
            config={**self.config, "mode": "adaptive"},
        )
        self.arbitrator.controller = self.controller

        # Unfreeze normalization for new object
        for store in [
            self.controller.q_store_free,
            self.controller.q_store_surface,
            self.controller.strategic_detach,
            self.controller.strategic_direction,
        ]:
            store._norm_frozen = False
            store._freeze_done = False
            store._state_buffer.clear()

        # Collect BC transitions
        success_trails = train_result.get("success_trails", [])
        if success_trails:
            new_transitions = (
                self.extractor.convert_all_trajectories(
                    success_trails
                )
            )
            self.bc_transitions.extend(new_transitions)
            if len(self.bc_transitions) > self.max_bc_transitions:
                self.bc_transitions = self.bc_transitions[
                    -self.max_bc_transitions :
                ]

        q_rate = train_result.get("success_rate", 0.0)
        logger.info("OFFLINE Q complete: rate=%.3f", q_rate)

        # === Phase 2: SAC retrain with training protections ===
        if self.sac_trainer is not None and self.bc_transitions:
            logger.info(
                "OFFLINE SAC: %d steps, bc=%d",
                self.offline_sac_steps,
                len(self.bc_transitions),
            )

            # Add transitions to buffer
            all_normalized = (
                self.sac_trainer._normalize_bc_transitions(
                    self.bc_transitions
                )
            )
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

            self.sac_trainer.bc_data = self.bc_transitions.copy()

            # Reset BC lambda to init (strong regularization, like training)
            old_bc_lambda = self.sac_trainer.bc_lambda
            self.sac_trainer.bc_lambda = self.sac_trainer.bc_lambda_init

            warmup = self.offline_sac_steps // 2

            for step_i in range(self.offline_sac_steps):
                if (
                    len(self.sac_trainer.buffer)
                    < self.sac_trainer.batch_size
                ):
                    break

                batch = self.sac_trainer.buffer.sample(
                    self.sac_trainer.batch_size
                )

                # Always CQL (conservative)
                self.sac_trainer.update_critic_cql(batch)

                # Actor: after warmup, every 5th step (like training)
                if step_i >= warmup and step_i % 5 == 0:
                    self.sac_trainer.update_actor(batch)

                # Alpha update (like training)
                if step_i >= warmup:
                    self.sac_trainer.update_alpha(batch)

                self.sac_trainer.soft_update_target()

            # Restore BC lambda to min (for future online)
            self.sac_trainer.bc_lambda = max(
                old_bc_lambda, self.sac_trainer.bc_lambda_min
            )

            sac_save_dir = str(self.runs_dir / "adaptive_sac")
            self.sac_trainer.save(sac_save_dir)
            self.online_transitions_raw = []

            logger.info(
                "OFFLINE SAC complete: %d steps, "
                "warmup=%d, bc_lambda=%.1f→%.1f",
                self.offline_sac_steps,
                warmup,
                self.sac_trainer.bc_lambda_init,
                self.sac_trainer.bc_lambda,
            )

        # Reset for fresh evaluation
        self.success_history.clear()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "success_rate": round(self.success_rate, 3),
            "total_episodes": self.total_episodes,
            "history_size": len(self.success_history),
            "online_transitions_pending": len(
                self.online_transitions_raw
            ),
            "bc_transitions": len(self.bc_transitions),
            "sac_loaded": self.sac_trainer is not None,
            "total_sac_updates": self.total_sac_updates,
            "total_offline_iterations": self.total_offline_iterations,
            "episodes_since_sac_update": (
                self.online_episodes_since_sac_update
            ),
            "episodes_since_offline": self._episodes_since_offline,
            "current_epsilon": round(
                self._get_adaptive_epsilon(), 3
            ),
        }
