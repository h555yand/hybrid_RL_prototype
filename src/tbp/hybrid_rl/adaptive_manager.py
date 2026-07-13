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
        offline_threshold: float = 0.60,
        sac_offline_threshold: float = 0.60,
        monitor_window: int = 100,
        online_sac_update_every: int = 200,
        online_sac_update_steps: int = 50,
        online_bc_update_every: int = 2000,
        offline_q_episodes: int = 5000,
        offline_sac_episodes: int = 2000,
        post_offline_cooldown: int = 200,
        new_object_sac_priority_episodes: int = 200,
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
        self.new_object_sac_priority_episodes = new_object_sac_priority_episodes

        self.success_history = deque(maxlen=monitor_window)
        self.online_transitions_raw = []
        self.online_episodes_since_sac_update = 0
        self.online_episodes_since_bc_update = 0
        self.total_episodes = 0
        self.total_sac_updates = 0
        self.total_bc_updates = 0
        self.total_offline_iterations = 0
        self._episodes_since_offline = 0
        self._sac_priority_active = True
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
        if self.sac_trainer is not None:
            self.arbitrator.sac_actor = self.sac_trainer.actor
            self.arbitrator.state_mean = self.sac_trainer.state_mean
            self.arbitrator.state_std = self.sac_trainer.state_std
            self.arbitrator.param_mean = self.sac_trainer.param_mean
            self.arbitrator.param_std = self.sac_trainer.param_std

        # SAC priority at the new facility
        self.arbitrator.boost_sac = (
            self._sac_priority_active
            and self.total_episodes < self.new_object_sac_priority_episodes
        )

        action_index, source = self.arbitrator.decide(state, current_pose, sensor_data)
        return action_index, source

    def decide_mode(self):
        if len(self.success_history) < self.monitor_window:
            return "online"

        # Cooldown afer offline
        if self._episodes_since_offline < self.post_offline_cooldown:
            return "online"

        rate = self.success_rate
        if rate >= self.online_threshold:
            return "inference_only"
        if rate >= self.offline_threshold:
            return "online"
        return "offline"

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
            self.controller.epsilon = 0.1
            self._collect_transitions(success, transitions)
            self._maybe_online_sac_update()
            self._maybe_online_bc_update()

        if self.mode == "offline":
            self._trigger_offline()
            self.controller.mode = "train"
            self.controller.epsilon = 0.1
            self.mode = "online"

    def _collect_transitions(self, success: bool, transitions: List[Dict[str, Any]]):
        if transitions:
            psac_transitions = self.extractor.convert_trajectory(transitions)
            self.online_transitions_raw.extend(psac_transitions)

    def _maybe_online_sac_update(self):
        if self.sac_trainer is None:
            return

        if self.online_episodes_since_sac_update < self.online_sac_update_every:
            return

        if len(self.online_transitions_raw) < self.sac_trainer.batch_size:
            return

        added = 0
        for tr in self.online_transitions_raw:
            if tr.next_state is None:
                continue

            norm_state = self.sac_trainer.normalize_state(tr.state)
            norm_next = self.sac_trainer.normalize_state(tr.next_state)

            norm_params = (
                tr.action_params - self.sac_trainer.param_mean[:len(tr.action_params)]
            ) / (self.sac_trainer.param_std[:len(tr.action_params)] + 1e-8)
            padded_params = np.zeros(self.sac_trainer.max_params, dtype=np.float32)
            padded_params[:len(norm_params)] = norm_params

            self.sac_trainer.buffer.add(
                state=norm_state.astype(np.float32),
                action_type=tr.action_type,
                action_params=padded_params,
                reward=tr.reward,
                next_state=norm_next.astype(np.float32),
                done=tr.done,
            )
            added += 1

        if len(self.sac_trainer.buffer) >= self.sac_trainer.batch_size:
            for _ in range(self.online_sac_update_steps):
                batch = self.sac_trainer.buffer.sample(self.sac_trainer.batch_size)
                self.sac_trainer.update_critic(batch)
                self.sac_trainer.update_actor(batch)
                self.sac_trainer.soft_update_target()

        self.total_sac_updates += 1
        self.online_episodes_since_sac_update = 0
        self.online_transitions_raw.clear()

        logger.info(
            f"AdaptiveTraining: online SAC update #{self.total_sac_updates} "
            f"(added={added}, buffer={len(self.sac_trainer.buffer)}, "
            f"steps={self.online_sac_update_steps})"
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
        self.total_offline_iterations += 1

        eps_idx = min(
            self.total_offline_iterations - 1,
            len(self.OFFLINE_Q_EPSILON_SCHEDULE) - 1
        )
        q_epsilon = self.OFFLINE_Q_EPSILON_SCHEDULE[eps_idx]

        logger.info(
            f"AdaptiveTraining: triggering OFFLINE iteration #{self.total_offline_iterations} "
            f"(success_rate={self.success_rate:.3f}, q_epsilon={q_epsilon})"
        )

        # === Q-learning: always train ===
        q_save_dir = str(self.runs_dir / "adaptive_q")
        q_load_dir = q_save_dir if (Path(q_save_dir) / "config.json").exists() else None

        logger.info(
            f"AdaptiveTraining: OFFLINE Q-learning "
            f"(episodes={self.offline_q_episodes}, epsilon={q_epsilon}, "
            f"load={'yes' if q_load_dir else 'no'})"
        )

        train_result = run_episodes(
            mesh_dir=str(self.runs_dir.parent),
            save_dir=q_save_dir,
            num_episodes=self.offline_q_episodes,
            config={
                **self.config,
                "mode": "train_adapt_epsilon",
                "epsilon_start": q_epsilon,
            },
            mesh_path=self.mesh_path,
            load_dir=q_load_dir,
            seed=42,
            return_metrics=True,
        )

        success_trails = train_result.get("success_trails", [])
        if success_trails:
            new_transitions = self.extractor.convert_all_trajectories(success_trails)
            self.bc_transitions.extend(new_transitions)
            logger.info(
                f"AdaptiveTraining: collected {len(new_transitions)} BC transitions "
                f"(total={len(self.bc_transitions)})"
            )

        q_success_rate = train_result.get("success_rate", 0.0)
        logger.info(
            f"AdaptiveTraining: OFFLINE Q-learning complete "
            f"(success_rate={q_success_rate:.3f})"
        )

        # === SAC: train only if the SAC success rate is below the threshold. ===
        sac_sr = self.arbitrator.sac_success_rate
        need_sac_retrain = (
            (self.sac_trainer is None
            or sac_sr < self.sac_offline_threshold)
            and self.total_offline_iterations > 2  # только после двух обучений Q
        )

        if need_sac_retrain:
            logger.info(
                f"AdaptiveTraining: OFFLINE SAC retraining "
                f"(sac_success_rate={sac_sr:.3f}, "
                f"threshold={self.sac_offline_threshold})"
            )

            if len(self.bc_transitions) > 100:
                num_action_types = len(ExperienceExtractor.get_type_names())
                bc_trainer = BCTrainer(
                    state_dim=self.config.get("state_dim", 15),
                    num_types=num_action_types,
                )
                bc_trainer.train(self.bc_transitions, num_epochs=200)
                bc_model_dir = str(self.runs_dir / "adaptive_bc")
                bc_trainer.save(bc_model_dir)

                import pickle
                bc_data_path = str(self.runs_dir / "adaptive_bc_data.pkl")
                with Path(bc_data_path).open("wb") as f:
                    pickle.dump(self.bc_transitions, f)

                sac_trainer = PSACTrainer(
                    state_dim=self.config.get("state_dim", 15),
                    num_types=num_action_types,
                    lr_actor=1e-5,
                    bc_lambda_init=5.0,
                    bc_lambda_decay=0.999999,
                )
                sac_trainer.load_bc(bc_model_dir, bc_data_path)

                sac_trainer.train(
                    env=self.env,
                    controller=self.controller,
                    num_episodes=self.offline_sac_episodes,
                    save_dir=str(self.runs_dir / "adaptive_sac"),
                    curriculum_levels=[
                        (10.0, 40.0),
                        (20.0, 80.0),
                        (40.0, 120.0),
                    ],
                )

                self.sac_trainer = sac_trainer
                logger.info("AdaptiveTraining: OFFLINE SAC retraining complete")
            else:
                logger.info(
                    f"AdaptiveTraining: OFFLINE SAC skipped — "
                    f"insufficient BC data ({len(self.bc_transitions)} transitions)"
                )
        else:
            logger.info(
                f"AdaptiveTraining: OFFLINE SAC skipped — "
                f"sac_success_rate {sac_sr:.3f} >= threshold {self.sac_offline_threshold}"
            )

        # Reset state after offline
        self.success_history.clear()
        self._episodes_since_offline = 0
        self._sac_priority_active = False  # после offline Q-store адаптирован

        logger.info(
            f"AdaptiveTraining: OFFLINE iteration #{self.total_offline_iterations} complete, "
            f"history cleared, cooldown={self.post_offline_cooldown} episodes"
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
            "sac_priority_active": self._sac_priority_active,
        }
