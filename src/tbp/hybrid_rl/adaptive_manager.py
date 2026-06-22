"""
AdaptiveTrainingManager: monitors performance and decides
when/how to train for new objects.

Modes:
  - inference_only: success_rate > 80%, no training needed
  - online: 60-80%, update Q-store + accumulate for SAC
  - offline: < 60%, full Q-learning → BC → SAC cycle
"""

import numpy as np
import logging
from collections import deque
from pathlib import Path
from typing import Dict, Any, Optional, List

from .rl_goal_approach_controller import RLGoalApproachController
from .lightweight_env import LightweightEnv
from .action_space import ActionSpace
from .experience_extractor import ExperienceExtractor
from .ablation_runner import train
from .behavioral_cloning import BCTrainer
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
        online_threshold: float = 0.80,
        offline_threshold: float = 0.60,
        monitor_window: int = 100,
        online_sac_update_every: int = 100,
        online_bc_update_every: int = 1000,
        offline_q_episodes: int = 5000,
        offline_sac_episodes: int = 10000,
    ):
        self.controller = controller
        self.env = env
        self.config = config
        self.runs_dir = Path(runs_dir)
        self.mesh_path = mesh_path

        self.online_threshold = online_threshold
        self.offline_threshold = offline_threshold
        self.monitor_window = monitor_window
        self.online_sac_update_every = online_sac_update_every
        self.online_bc_update_every = online_bc_update_every
        self.offline_q_episodes = offline_q_episodes
        self.offline_sac_episodes = offline_sac_episodes

        self.success_history = deque(maxlen=monitor_window)
        self.online_transitions = []
        self.total_episodes = 0
        self.mode = "inference_only"

        self.sac_trainer = None
        self.action_space = controller.action_space
        self.extractor = ExperienceExtractor(config=config)

    @property
    def success_rate(self):
        if len(self.success_history) == 0:
            return 0.0
        return sum(self.success_history) / len(self.success_history)

    def decide_mode(self):
        if len(self.success_history) < self.monitor_window:
            return "online"

        rate = self.success_rate
        if rate >= self.online_threshold:
            return "inference_only"
        elif rate >= self.offline_threshold:
            return "online"
        else:
            return "offline"

    def on_episode_complete(self, success: bool, transitions: List[Dict[str, Any]]):
        self.success_history.append(success)
        self.total_episodes += 1

        old_mode = self.mode
        self.mode = self.decide_mode()

        if self.mode != old_mode:
            logger.info(
                f"AdaptiveTraining: mode changed {old_mode} → {self.mode} "
                f"(success_rate={self.success_rate:.3f}, "
                f"episodes={self.total_episodes})"
            )

        if self.mode == "inference_only":
            return

        if self.mode == "online":
            self._online_update(success, transitions)

        if self.mode == "offline":
            self._trigger_offline()
            self.mode = "online"

    def _online_update(self, success: bool, transitions: List[Dict[str, Any]]):
        if success and transitions:
            psac_transitions = self.extractor.convert_trajectory(transitions)
            self.online_transitions.extend(psac_transitions)

        if (self.sac_trainer is not None
                and len(self.online_transitions) >= self.online_sac_update_every):
            for tr in self.online_transitions:
                if tr.next_state is not None:
                    self.sac_trainer.buffer.add(
                        state=tr.state,
                        action_type=tr.action_type,
                        action_params=tr.action_params,
                        reward=tr.reward,
                        next_state=tr.next_state,
                        done=tr.done,
                    )

            if len(self.sac_trainer.buffer) >= self.sac_trainer.batch_size:
                for _ in range(10):
                    batch = self.sac_trainer.buffer.sample(self.sac_trainer.batch_size)
                    self.sac_trainer.update_critic(batch)
                    self.sac_trainer.update_actor(batch)
                    self.sac_trainer.soft_update_target()

            self.online_transitions.clear()
            logger.info(
                f"AdaptiveTraining: online SAC update "
                f"(buffer={len(self.sac_trainer.buffer)})"
            )

    def _trigger_offline(self):
        logger.info(
            f"AdaptiveTraining: triggering OFFLINE training "
            f"(success_rate={self.success_rate:.3f})"
        )

        q_save_dir = str(self.runs_dir / "adaptive_q")
        train_result = train(
            mesh_dir=str(self.runs_dir.parent),
            save_dir=q_save_dir,
            num_episodes=self.offline_q_episodes,
            config={**self.config, "mode": "train_adapt_epsilon", "epsilon_start": 0.3},
            mesh_path=self.mesh_path,
            load_dir=q_save_dir if (Path(q_save_dir) / "config.json").exists() else None,
            seed=42,
            return_metrics=True,
        )

        success_trails = train_result.get("success_trails", [])
        if success_trails:
            bc_transitions = self.extractor.convert_all_trajectories(success_trails)

            if len(bc_transitions) > 100:
                bc_trainer = BCTrainer(state_dim=self.config.get("state_dim", 15))
                bc_trainer.train(bc_transitions, num_epochs=200)
                bc_model_dir = str(self.runs_dir / "adaptive_bc")
                bc_trainer.save(bc_model_dir)

                import pickle
                bc_data_path = str(self.runs_dir / "adaptive_bc_data.pkl")
                with open(bc_data_path, "wb") as f:
                    pickle.dump(bc_transitions, f)

                sac_trainer = PSACTrainer(
                    state_dim=self.config.get("state_dim", 15),
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
                    curriculum_levels=[(10.0, 40.0), (20.0, 80.0), (40.0, 120.0)],
                )

                self.sac_trainer = sac_trainer

        logger.info("AdaptiveTraining: offline training complete")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "success_rate": self.success_rate,
            "total_episodes": self.total_episodes,
            "history_size": len(self.success_history),
            "online_transitions_pending": len(self.online_transitions),
            "sac_loaded": self.sac_trainer is not None,
        }
