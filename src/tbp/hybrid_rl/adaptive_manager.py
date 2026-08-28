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
  - online: 50-80%, full Q learning + periodic SAC updates
    (critic CQL + actor every 10th step with strong BC),
    adaptive epsilon based on success rate
  - offline: < 50% sustained, Q retrain (500 ep, eps 1→0.3)
    + SAC retrain via PSACTrainer.train(300 ep)
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
from .sac_trainer import PSACTrainer
from .rl_goal_approach_controller import (
    RLGoalApproachController,
    RunningQStats,
)

logger = logging.getLogger(__name__)


class AdaptiveTrainingManager:

    def __init__(
        self,
        controller: RLGoalApproachController,
        env: LightweightEnv,
        config: Dict[str, Any],
        runs_dir: str,
        mesh_path: str,
        mastered_threshold: float = 0.95,
        offline_threshold: float = 0.40,
        monitor_window: int = 100,
        online_sac_update_every: int = 200,
        online_sac_update_steps: int = 20,
        offline_q_episodes: int = 500,
        offline_sac_episodes: int = 300,
        post_offline_cooldown: int = 200,
        min_online_before_offline: int = 300,
        max_offline_iterations: int = 2,
        max_bc_transitions: int = 5000,
        epsilon_warmup_episodes: int = 20,
        epsilon_default: float = 0.15,
        q_save_dir: str = None,
        sac_save_dir: str = None,
        offline_check_window: int = 50,
        promote_threshold: float = 0.70,
        promote_window: int = 100,
    ):
        self.controller = controller
        self.env = env
        self.config = config
        self.runs_dir = Path(runs_dir)
        self.mesh_path = mesh_path
        self.q_save_dir = q_save_dir or str(Path(runs_dir) / "adaptive_q")
        self.sac_save_dir = sac_save_dir or str(Path(runs_dir) / "adaptive_sac")

        # Mode thresholds
        self.mastered_threshold = mastered_threshold
        self.offline_threshold = offline_threshold
        self.monitor_window = monitor_window
        self.offline_check_window = offline_check_window
        self.promote_threshold = promote_threshold
        self.promote_window = promote_window

        # Online SAC updates
        self.online_sac_update_every = online_sac_update_every
        self.online_sac_update_steps = online_sac_update_steps

        # Offline retrain params
        self.offline_q_episodes = offline_q_episodes
        self.offline_q_epsilon_start = 1.0
        self.offline_q_epsilon_min = 0.3
        self.offline_sac_episodes = offline_sac_episodes
        self.post_offline_cooldown = post_offline_cooldown
        self.min_online_before_offline = min_online_before_offline
        self.max_offline_iterations = max_offline_iterations

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
        self.mode_changes: List[Dict[str, Any]] = []
        self._level_success_history: Dict[int, deque] = {}
        self._offline_just_completed = False

    @property
    def success_rate(self) -> float:
        if len(self.success_history) == 0:
            return 0.0
        return sum(self.success_history) / len(
            self.success_history
        )

    def _get_level_success_rate(self, level: int) -> float:
        history = self._level_success_history.get(level)
        if history is None or len(history) == 0:
            return 0.0
        return sum(history) / len(history)

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
        current_level = self.arbitrator._current_level
        level_history = self._level_success_history.get(current_level)
        if level_history is None or len(level_history) < self.monitor_window:
            return "online"
        if self._episodes_since_offline < self.post_offline_cooldown:
            return "online"

        rate = sum(level_history) / len(level_history)

        if rate >= self.mastered_threshold:
            return "mastered"

        if self._episodes_since_offline < self.offline_check_window:
            return "online"

        h_track = self.arbitrator._get_track(
            self.arbitrator._level_heuristic_results[current_level]
        )
        h_track = max(h_track, 0.1)

        q_track = self.arbitrator._get_track(
            self.arbitrator._level_q_results[current_level]
        )
        sac_track = self.arbitrator._get_track(
            self.arbitrator._level_sac_results[current_level]
        )
        best_ml_track = max(q_track, sac_track)

        if best_ml_track < h_track * 0.5:
            if (
                self.total_offline_iterations == 0
                and self.total_episodes < self.min_online_before_offline
            ):
                return "online"
            if self.total_offline_iterations >= self.max_offline_iterations:
                return "online"
            return "offline"

        return "online"
    
    def on_episode_complete(
        self,
        success: bool,
        transitions: List[Dict[str, Any]],
    ) -> None:
        self.success_history.append(success)
        current_level = self.arbitrator._current_level
        if current_level not in self._level_success_history:
            self._level_success_history[current_level] = deque(maxlen=self.monitor_window)
        self._level_success_history[current_level].append(success)
        self.total_episodes += 1
        self._episodes_since_offline += 1
        self.online_episodes_since_sac_update += 1

        old_mode = self.mode
        self.mode = self.decide_mode()

        if self.mode != old_mode:
            self.mode_changes.append({
                "episode": self.total_episodes,
                "from_mode": old_mode,
                "to_mode": self.mode,
                "success_rate": round(self.success_rate, 3),
                "offline_iterations": self.total_offline_iterations,
                "sac_updates": self.total_sac_updates,
            })
            logger.info(
                "AdaptiveTraining: mode %s → %s "
                "(rate=%.3f, episodes=%d)",
                old_mode,
                self.mode,
                self.success_rate,
                self.total_episodes,
            )

        if self.mode == "mastered":
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
            self._offline_just_completed = True
            self.controller.mode = "adaptive"
            eps = self._get_adaptive_epsilon()
            self.controller.epsilon = eps
            self.controller.strategic_epsilon = eps
            self.mode = "online"
            self.mode_changes.append({
                "episode": self.total_episodes,
                "from_mode": "offline",
                "to_mode": "online",
                "success_rate": 0.0,
                "offline_iterations": self.total_offline_iterations,
                "sac_updates": self.total_sac_updates,
                "note": "post_offline_reset",
            })

    def _collect_transitions(
        self,
        success: bool,
        transitions: List[Dict[str, Any]],
    ) -> None:
        """Collect transitions for SAC training.

        Success transitions always collected (buffer + BC).
        Failure transitions collected with 30% probability
        (critic needs some negatives, but not too many).
        """
        if not transitions:
            return

        psac_transitions = self.extractor.convert_trajectory(
            transitions
        )

        if success:
            self.online_transitions_raw.extend(psac_transitions)
            self.bc_transitions.extend(psac_transitions)
            if len(self.bc_transitions) > self.max_bc_transitions:
                self.bc_transitions = self.bc_transitions[
                    -self.max_bc_transitions :
                ]
        else:
            if np.random.random() < 0.3:
                self.online_transitions_raw.extend(psac_transitions)

    def _maybe_online_sac_update(self) -> None:
        """Periodic online SAC update.

        Critic: CQL every step.
        Actor: every 10th step with strong BC lambda (like training).
        No interaction with environment — gradient steps on buffer.
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
        mesh_name = Path(self.mesh_path).stem
        for tr in all_normalized:
            if tr.next_state is not None:
                self.sac_trainer.buffer.add(
                    state=tr.state,
                    action_type=tr.action_type,
                    action_params=tr.action_params,
                    reward=tr.reward,
                    next_state=tr.next_state,
                    done=tr.done,
                    mesh_name=mesh_name,
                )
                added += 1

        # Update BC data (successful only)
        if self.bc_transitions:
            self.sac_trainer.bc_data = self.bc_transitions.copy()

        self.online_transitions_raw = []

        if len(self.sac_trainer.buffer) < self.sac_trainer.batch_size:
            self.online_episodes_since_sac_update = 0
            return

        # Strong BC for online (prevent forgetting, like training init)
        old_bc_lambda = self.sac_trainer.bc_lambda
        self.sac_trainer.bc_lambda = self.sac_trainer.bc_lambda_init

        mesh_name = Path(self.mesh_path).stem
        for step_i in range(self.online_sac_update_steps):
            batch = self.sac_trainer.buffer.sample_balanced(
                self.sac_trainer.batch_size,
                current_mesh=mesh_name,
                current_ratio=0.5,
                bc_ratio=0.1,
                elite_ratio=0.1,
            )

            # CQL critic every step
            self.sac_trainer.update_critic_cql(batch)

            # Actor every 10th step (same as training)
            if step_i % 10 == 0 and self.bc_transitions:
                self.sac_trainer.update_actor(batch)

            self.sac_trainer.soft_update_target()

        self.sac_trainer.bc_lambda = old_bc_lambda
        self.total_sac_updates += 1
        self.online_episodes_since_sac_update = 0

        logger.info(
            "Online SAC update: added=%d, bc=%d, "
            "steps=%d, bc_lambda=%.1f, total=%d",
            added,
            len(self.bc_transitions),
            self.online_sac_update_steps,
            self.sac_trainer.bc_lambda_init,
            self.total_sac_updates,
        )

    def _trigger_offline(self):
        self.total_offline_iterations += 1
        self._episodes_since_offline = 0

        logger.info(
            "AdaptiveTraining: OFFLINE #%d (rate=%.3f)",
            self.total_offline_iterations,
            self.success_rate,
        )

        # Log pre-offline track records
        current_level = self.arbitrator._current_level
        for source_name, results_dict in [
            ("Q", self.arbitrator._level_q_results),
            ("SAC", self.arbitrator._level_sac_results),
            ("Heuristic", self.arbitrator._level_heuristic_results),
        ]:
            for level, results in results_dict.items():
                if results:
                    logger.info(
                        "Pre-offline %s track L%d: %.3f "
                        "(%d episodes)",
                        source_name,
                        level,
                        sum(results) / len(results),
                        len(results),
                    )

        current_level = self.arbitrator._current_level
        all_levels = list(
            self.config.get("curriculum_levels", [[10, 120]])
        )
        all_filters = list(
            self.config.get("curriculum_filters", [{}])
        )
        if current_level >= len(all_levels):
            remaining_levels = [all_levels[-1]] if all_levels else [[10, 120]]
            remaining_filters = [all_filters[-1]] if all_filters else [{}]
        else:
            remaining_levels = all_levels[current_level:]
            remaining_filters = all_filters[current_level:]

        if not remaining_levels:
            remaining_levels = [[10, 120]]
        if not remaining_filters:
            remaining_filters = [{}]

        q_save_dir = self.q_save_dir
        self.controller.save(q_save_dir)

        q_warmup = self.offline_q_episodes // 2

        logger.info(
            "OFFLINE Q: %d episodes (warmup=%d), eps %.1f→%.1f, "
            "from level %d (%d levels)",
            self.offline_q_episodes,
            q_warmup,
            self.offline_q_epsilon_start,
            self.offline_q_epsilon_min,
            current_level,
            len(remaining_levels),
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
                "warmup_episodes": q_warmup,
                "curriculum_levels": remaining_levels,
                "curriculum_filters": remaining_filters,
                "unfreeze_normalization": True,
            },
            mesh_path=self.mesh_path,
            seed=42,
            return_metrics=True,
            curriculum_config={
                "levels": remaining_levels,
                "promote_threshold": self.promote_threshold,
                "promote_window": self.promote_window,
            },
        )

        self.controller = RLGoalApproachController.load(
            q_save_dir,
            agent_id=self.controller.agent_id,
            config={**self.config, "mode": "adaptive"},
        )
        self.arbitrator.controller = self.controller
        # Re-warmup RunningQStats after offline retrain
        self.arbitrator._running_q_stats_free = RunningQStats(warmup=200)
        self.arbitrator._running_q_stats_surface = RunningQStats(warmup=200)
        self.arbitrator._warmup_running_stats()

        self.arbitrator._level_q_results.clear()
        self.arbitrator._level_sac_results.clear()
        self.arbitrator._level_heuristic_results.clear()
        self.arbitrator._calibration_counter = 0
        self.arbitrator._is_calibrating = True
        self.arbitrator._episodes_on_level = 0

        q_rate = train_result.get("success_rate", 0.0)
        q_retrain_result = {
            "episodes": self.offline_q_episodes,
            "success_rate": q_rate,
            "curriculum_stats": train_result.get(
                "curriculum_stats", {}
            ),
            "stats": {
                k: v
                for k, v in train_result.get(
                    "stats", {}
                ).items()
                if k in (
                    "total_episodes",
                    "total_steps",
                    "total_goals_reached",
                    "success_rate",
                    "termination_counts",
                    "termination_rates",
                    "q_store_free",
                    "q_store_surface",
                    "collision_stats",
                    "steps_per_success",
                )
            },
            "phase_metrics": train_result.get(
                "phase_metrics", {}
            ),
        }

        q_retrain_path = (
            Path(self.q_save_dir)
            / "offline_retrain_result.json"
        )
        q_retrain_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        try:
            import json
            with q_retrain_path.open("w") as f:
                json.dump(
                    q_retrain_result, f, indent=2
                )
        except Exception as exc:
            logger.warning(
                "Could not save Q retrain result: %s",
                exc,
            )

        logger.info(
            "OFFLINE Q complete: rate=%.3f, "
            "saved to %s",
            q_rate,
            q_retrain_path,
        )
        
        success_trails = train_result.get("success_trails", [])

        if (
            self.sac_trainer is not None
            and success_trails
            and len(success_trails) >= 5
        ):
            all_bc = self.extractor.convert_all_trajectories(
                success_trails
            )

            logger.info(
                "OFFLINE SAC: %d ep, %d bc from %d trails, "
                "from level %d, CQL=True",
                self.offline_sac_episodes,
                len(all_bc),
                len(success_trails),
                current_level,
            )

            self.sac_trainer.bc_data = all_bc
            self.sac_trainer.bc_lambda = self.sac_trainer.bc_lambda_init

            old_use_cql = self.sac_trainer.use_cql
            old_cql_alpha = self.sac_trainer.cql_alpha
            self.sac_trainer.use_cql = True
            self.sac_trainer.cql_alpha = 1.0
            old_eval_interval = self.sac_trainer.eval_interval
            self.sac_trainer.eval_interval = 100
            self.sac_trainer.buffer.set_current_mesh(
                Path(self.mesh_path).stem
            )
            self.sac_trainer.train(
                env=self.env,
                controller=self.controller,
                num_episodes=self.offline_sac_episodes,
                warmup_steps=500,
                update_every=1,
                updates_per_step=1,
                log_interval=50,
                save_dir=self.sac_save_dir,
                curriculum_levels=remaining_levels,
                curriculum_filters=remaining_filters,
                mesh_name=Path(self.mesh_path).stem,
            )

            self.sac_trainer.eval_interval = old_eval_interval
            self.sac_trainer.use_cql = old_use_cql
            self.sac_trainer.cql_alpha = old_cql_alpha

            self.online_transitions_raw = []

            # Log SAC retrain results
            sac_stats = self.sac_trainer.get_stats() if hasattr(self.sac_trainer, 'get_stats') else {}
            sac_retrain_result = {
                "episodes": self.offline_sac_episodes,
                "sac_stats": sac_stats,
                "bc_data_size": len(all_bc),
                "success_trails_used": len(success_trails),
            }
            
            sac_retrain_path = (
                Path(self.sac_save_dir)
                / "offline_retrain_result.json"
            )
            sac_retrain_path.parent.mkdir(
                parents=True, exist_ok=True
            )
            try:
                import json
                with sac_retrain_path.open("w") as f:
                    json.dump(
                        sac_retrain_result, f, indent=2
                    )
            except Exception as exc:
                logger.warning(
                    "Could not save SAC retrain result: %s",
                    exc,
                )

            logger.info(
                "OFFLINE SAC complete: %d episodes, "
                "bc=%d, trails=%d, saved to %s",
                self.offline_sac_episodes,
                len(all_bc),
                len(success_trails),
                sac_retrain_path,
            )
        else:
            logger.info(
                "OFFLINE SAC skipped: insufficient trails (%d)",
                len(success_trails) if success_trails else 0,
            )

        self.success_history.clear()
        self._level_success_history.clear()
        
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
