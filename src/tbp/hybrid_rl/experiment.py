# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""RL Goal Approach Experiment — Hydra-compatible experiment class.

Orchestrates the full RL navigation pipeline: Q-learning training,
evaluation, Behavioral Cloning, SAC training, and adaptive mode.
Configured via Hydra YAML.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from typing_extensions import Self

from tbp.hybrid_rl.ablation_runner import (
    GOAL_THRESHOLD_PER_LEVEL,
    run_episodes,
    run_eval_per_seed,
)
from tbp.hybrid_rl.action_interpreter import ActionInterpreter
from tbp.hybrid_rl.adaptive_manager import AdaptiveTrainingManager
from tbp.hybrid_rl.behavioral_cloning import BCTrainer
from tbp.hybrid_rl.config import DEFAULT_CONFIG
from tbp.hybrid_rl.episode_pools import get_or_generate_pools
from tbp.hybrid_rl.experience_extractor import ExperienceExtractor
from tbp.hybrid_rl.lightweight_env import LightweightEnv
from tbp.hybrid_rl.mesh_factory import prepare_demo_meshes
from tbp.hybrid_rl.rl_goal_approach_controller import RLGoalApproachController
from tbp.hybrid_rl.sac_trainer import PSACTrainer

logger = logging.getLogger(__name__)

_ADAPTIVE_MIN_DIST = 10.0
_ADAPTIVE_MAX_DIST = 120.0
_ADAPTIVE_MAX_STEPS = 150
_ADAPTIVE_LOG_INTERVAL = 100
_SAC_WARMUP_STEPS = 5000
_BC_NUM_EPOCHS = 200
_EVAL_EPSILON = 0.02
_COLLISION_DEPTH_THRESHOLD = 0.5


class RLGoalApproachExperiment:
    """Hydra-compatible experiment for RL goal approach training pipeline.

    Supports stages: Q-learning train, eval, BC train, SAC train,
    SAC eval, and adaptive mode. All parameters are configured via
    Hydra YAML config.

    Usage::

        python run.py experiment=rl_goal_approach
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize experiment from Hydra config.

        Args:
            config: Hydra DictConfig with experiment parameters.
        """
        self.config: dict[str, Any] = (
            OmegaConf.to_object(config)
            if isinstance(config, DictConfig)
            else config
        )

        self.seed = self.config.get("seed", 42)
        self.output_dir = Path(self.config["logging"]["output_dir"])
        self.run_name = self.config["logging"]["run_name"]

        self.visualise = self.config.get("visualise", False)

        self.data_dir = self.output_dir / "data"
        self.runs_dir = self.data_dir / "runs"
        self.scripts_dir = self.data_dir / "episode_scripts"

        self.do_train = self.config.get("do_train", True)
        self.do_eval = self.config.get("do_eval", False)
        self.do_bc_train = self.config.get("do_bc_train", False)
        self.do_sac_train = self.config.get("do_sac_train", False)
        self.do_sac_eval = self.config.get("do_sac_eval", False)
        self.do_adaptive = self.config.get("do_adaptive", False)

        self.training_stages: list[dict[str, Any]] = self.config.get(
            "training_stages", []
        )
        self.curriculum_levels: list[tuple[float, float]] = [
            tuple(level)
            for level in self.config.get(
                "curriculum_levels",
                [[10.0, 40.0], [20.0, 80.0], [40.0, 120.0]],
            )
        ]
        self.train_seeds: list[int] = self.config.get(
            "train_seeds", [11]
        )
        self.eval_seeds: list[int] = self.config.get(
            "eval_seeds", [44]
        )
        self.sac_eval_seeds: list[int] = self.config.get(
            "sac_eval_seeds", [77, 88, 99]
        )
        self.eval_episodes_per_level: int = self.config.get(
            "eval_episodes_per_level", 500
        )
        self.sac_eval_episodes_per_level: int = self.config.get(
            "sac_eval_episodes_per_level", 500
        )
        self.regenerate_scripts: bool = self.config.get(
            "regenerate_scripts", True
        )

        self.rl_config: dict[str, Any] = {
            **DEFAULT_CONFIG,
            **self.config.get("rl_config", {}),
        }

        self.promote_threshold: float = self.config.get(
            "promote_threshold", 0.55
        )
        self.promote_window: int = self.config.get(
            "promote_window", 100
        )

        self.sac_config: dict[str, Any] = self.config.get(
            "sac_config", {}
        )
        self.sac_meshes: list[str] = self.config.get(
            "sac_meshes", ["cube", "cylinder", "mug", "cup"]
        )
        self.sac_episodes_per_mesh: dict[str, int] = self.config.get(
            "sac_episodes_per_mesh",
            {
                "cube": 1000,
                "cylinder": 1500,
                "mug": 2000,
                "cup": 2000,
            },
        )

        self.adaptive_mesh: str = self.config.get(
            "adaptive_mesh", "cup"
        )
        self.adaptive_episodes: int = self.config.get(
            "adaptive_episodes", 3000
        )

        self.eval_meshes: list[str] = self.config.get(
            "eval_meshes", ["cube", "cylinder", "mug", "cup"]
        )

        self.unified_save_dir = str(self.runs_dir / "unified_q")

    def __enter__(self) -> Self:
        """Set up experiment directories and demo meshes.

        Returns:
            Self for use in with-statement.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        prepare_demo_meshes(self.data_dir)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: types.TracebackType | None,
    ) -> bool:
        """Clean up on context exit.

        Returns:
            False to not suppress exceptions.
        """
        return False

    def run(self) -> None:
        """Run all enabled pipeline stages."""
        start_time = time.time()

        if self.do_train:
            self._run_train()
        if self.do_eval:
            self._run_eval()
        if self.do_bc_train:
            self._run_bc_train()
        if self.do_sac_train:
            self._run_sac_train()
        if self.do_sac_eval:
            self._run_sac_eval()
        if self.do_adaptive:
            self._run_adaptive()

        elapsed = time.time() - start_time
        logger.info("Experiment complete in %.1fs", elapsed)

    def _run_train(self) -> None:
        """Run Q-learning training across all configured stages."""
        logger.info("=" * 60)
        logger.info("Q-Learning Training (all stages)")
        logger.info("=" * 60)

        for stage_idx, stage in enumerate(self.training_stages):
            mesh_name = stage["mesh"]
            mesh_path = str(self.data_dir / f"{mesh_name}.stl")
            episodes = stage["episodes"]
            epsilon_start = stage["epsilon_start"]
            is_load = stage["is_load"]

            logger.info(
                "STAGE %d/%d: %s, episodes=%d, eps=%.2f",
                stage_idx + 1,
                len(self.training_stages),
                mesh_name,
                episodes,
                epsilon_start,
            )

            stage_cfg = {
                **self.rl_config,
                "epsilon_start": epsilon_start,
                "epsilon_min": stage.get("epsilon_min", 0.05),
                "num_episodes": episodes,
                "unfreeze_normalization": is_load,
            }

            train_pools = get_or_generate_pools(
                mesh_path=mesh_path,
                seeds=self.train_seeds,
                episodes_per_level=episodes,
                scripts_dir=self.scripts_dir,
                curriculum_levels=self.curriculum_levels,
                regenerate=self.regenerate_scripts,
                prefix=f"train_{mesh_name}",
            )

            for seed in self.train_seeds:
                seed_save_dir = (
                    f"{self.unified_save_dir}_seed_{seed}"
                )
                load_dir = seed_save_dir if is_load else None

                seed_pools = train_pools.get(seed)
                episode_pools = (
                    seed_pools.get("levels")
                    if seed_pools
                    else None
                )

                curriculum_config = {
                    "levels": self.curriculum_levels,
                    "promote_threshold": self.promote_threshold,
                    "promote_window": self.promote_window,
                }

                run_result = run_episodes(
                    mesh_dir=str(self.data_dir),
                    save_dir=seed_save_dir,
                    load_dir=load_dir,
                    num_episodes=episodes,
                    config=stage_cfg,
                    mesh_path=mesh_path,
                    seed=seed,
                    return_metrics=True,
                    curriculum_config=curriculum_config,
                    episode_pools=episode_pools,
                    visualise=self.visualise,
                )

                stage_output = (
                    self.data_dir
                    / f"train_result_{mesh_name}_seed_{seed}.json"
                )
                stage_data = {
                    "stage": stage_idx,
                    "mesh": mesh_name,
                    "seed": seed,
                    "success_rate": run_result.get("success_rate"),
                    "stats": run_result.get("stats"),
                    "curriculum_stats": run_result.get(
                        "curriculum_stats"
                    ),
                }
                with stage_output.open("w") as f:
                    json.dump(stage_data, f, indent=2)
                logger.info(
                    "Stage result: success_rate=%.4f",
                    run_result.get("success_rate", 0),
                )

    def _run_eval(self) -> None:
        """Run evaluation on all meshes and collect BC data."""
        logger.info("=" * 60)
        logger.info("Eval on all meshes")
        logger.info("=" * 60)

        eval_cfg = {
            **self.rl_config,
            "mode": "eval",
            "eval_epsilon": _EVAL_EPSILON,
            "goal_threshold": GOAL_THRESHOLD_PER_LEVEL[0],
        }

        all_bc_transitions: list[Any] = []
        all_eval_results: dict[str, Any] = {}

        for mesh_name in self.eval_meshes:
            mesh_path = str(self.data_dir / f"{mesh_name}.stl")
            logger.info("Evaluating: %s", mesh_name)

            eval_pools = get_or_generate_pools(
                mesh_path=mesh_path,
                seeds=self.eval_seeds,
                episodes_per_level=self.eval_episodes_per_level,
                scripts_dir=self.scripts_dir,
                curriculum_levels=self.curriculum_levels,
                regenerate=self.regenerate_scripts,
                prefix=f"eval_{mesh_name}",
            )

            eval_results, bc_transitions = run_eval_per_seed(
                data_dir=self.data_dir,
                runs_dir=self.runs_dir,
                mesh_path=mesh_path,
                train_seeds=self.train_seeds,
                eval_seeds=self.eval_seeds,
                variant="unified_q",
                eval_cfg=eval_cfg,
                eval_pools=eval_pools,
                collect_bc=True,
                episodes_per_level=self.eval_episodes_per_level,
                mesh_name=mesh_name,
            )

            all_eval_results[mesh_name] = eval_results
            all_bc_transitions.extend(bc_transitions)

            mesh_output = (
                self.data_dir / f"eval_result_{mesh_name}.json"
            )
            with mesh_output.open("w") as f:
                json.dump(eval_results, f, indent=2)

        eval_output = self.data_dir / "eval_result_all.json"
        with eval_output.open("w") as f:
            json.dump(all_eval_results, f, indent=2)

        if all_bc_transitions:
            # Filter for quality and balance
            all_bc_transitions = self._filter_bc_transitions(
                all_bc_transitions
            )

            bc_output = self.data_dir / "bc_data.pkl"
            with bc_output.open("wb") as f:
                pickle.dump(all_bc_transitions, f)  # noqa: S301
            logger.info(
                "BC data: %d transitions",
                len(all_bc_transitions),
            )

    def _run_bc_train(self) -> None:
        """Train Behavioral Cloning model from collected data."""
        logger.info("=" * 60)
        logger.info("BC Training")
        logger.info("=" * 60)

        bc_path = self.data_dir / "bc_data.pkl"
        with bc_path.open("rb") as f:
            bc_transitions = pickle.load(f)  # noqa: S301

        logger.info(
            "Loaded %d BC transitions", len(bc_transitions)
        )

        num_types = len(ExperienceExtractor.get_type_names())
        trainer = BCTrainer(
            state_dim=self.rl_config.get("state_dim", 15),
            num_types=num_types,
        )
        trainer.train(bc_transitions, num_epochs=_BC_NUM_EPOCHS)
        trainer.save(str(self.runs_dir / "bc_model"))

        # Save BC training stats
        bc_stats = trainer.get_training_stats(bc_transitions)
        bc_stats_path = self.data_dir / "bc_train_result.json"
        with bc_stats_path.open("w") as f:
            json.dump(bc_stats, f, indent=2)
        logger.info(
            "BC training complete: val_acc=%.4f, "
            "transitions=%d, saved to %s",
            bc_stats["val_accuracy"],
            bc_stats["total_transitions"],
            bc_stats_path,
        )

    def _run_sac_train(self) -> None:
        """Train P-SAC model sequentially on all meshes."""
        logger.info("=" * 60)
        logger.info("P-SAC Training")
        logger.info("=" * 60)

        num_types = len(ExperienceExtractor.get_type_names())
        sac_seed = self.train_seeds[0]

        trainer = PSACTrainer(
            state_dim=self.rl_config.get("state_dim", 15),
            num_types=num_types,
            **self.sac_config,
        )
        trainer.load_bc(
            bc_model_dir=str(self.runs_dir / "bc_model"),
            bc_data_path=str(self.data_dir / "bc_data.pkl"),
        )

        sac_model_dir = str(self.runs_dir / "sac_model")

        for mesh_name in self.sac_meshes:
            mesh_path = str(self.data_dir / f"{mesh_name}.stl")
            num_episodes = self.sac_episodes_per_mesh.get(
                mesh_name, 2000
            )

            logger.info(
                "SAC Training: %s, %d episodes",
                mesh_name,
                num_episodes,
            )

            env = LightweightEnv(mesh_path)
            controller = RLGoalApproachController(
                agent_id=f"sac_{mesh_name}",
                config={**self.rl_config, "mode": "eval"},
            )

            sac_pools = get_or_generate_pools(
                mesh_path=mesh_path,
                seeds=[sac_seed],
                episodes_per_level=num_episodes,
                scripts_dir=self.scripts_dir,
                curriculum_levels=self.curriculum_levels,
                regenerate=self.regenerate_scripts,
                prefix=f"sac_train_{mesh_name}",
            )

            trainer.start_mesh_tracking(mesh_name)

            is_first_mesh = mesh_name == self.sac_meshes[0]
            trainer.train(
                env=env,
                controller=controller,
                num_episodes=num_episodes,
                warmup_steps=(
                    _SAC_WARMUP_STEPS if is_first_mesh else 0
                ),
                save_dir=sac_model_dir,
                curriculum_levels=self.curriculum_levels,
                episode_pools=sac_pools[sac_seed],
            )

            # Save per-mesh training stats
            sac_stats = trainer.get_training_stats()
            result_path = (
                self.data_dir
                / f"sac_train_result_{mesh_name}.json"
            )
            with result_path.open("w") as f:
                json.dump(sac_stats, f, indent=2)
            logger.info(
                "SAC %s: rate=%.3f, steps_per_success=%.1f",
                mesh_name,
                sac_stats["success_rate"],
                sac_stats["steps_per_success"],
            )

        # Save final combined stats
        final_stats = trainer.get_training_stats()
        final_path = self.data_dir / "sac_train_result_all.json"
        with final_path.open("w") as f:
            json.dump(final_stats, f, indent=2)
        logger.info("SAC training complete. Saved to %s", final_path)

    def _run_sac_eval(self) -> None:
        """Evaluate P-SAC model on all meshes."""
        logger.info("=" * 60)
        logger.info("P-SAC Eval")
        logger.info("=" * 60)

        num_types = len(ExperienceExtractor.get_type_names())
        sac_trainer = PSACTrainer(
            state_dim=self.rl_config.get("state_dim", 15),
            num_types=num_types,
        )
        sac_trainer.load(str(self.runs_dir / "sac_model"))

        all_sac_eval_results: dict[str, Any] = {}

        for mesh_name in self.eval_meshes:
            mesh_path = str(self.data_dir / f"{mesh_name}.stl")
            logger.info("SAC Eval: %s", mesh_name)

            sac_eval_pools = get_or_generate_pools(
                mesh_path=mesh_path,
                seeds=self.sac_eval_seeds,
                episodes_per_level=(
                    self.sac_eval_episodes_per_level
                ),
                scripts_dir=self.scripts_dir,
                curriculum_levels=self.curriculum_levels,
                regenerate=self.regenerate_scripts,
                prefix=f"sac_eval_{mesh_name}",
            )

            results = self._eval_sac_on_pools(
                sac_trainer=sac_trainer,
                sac_eval_pools=sac_eval_pools,
                mesh_path=mesh_path,
            )
            all_sac_eval_results[mesh_name] = results

            for key in sorted(results.keys()):
                data = results[key]
                logger.info(
                    "  %s (%smm): success=%.4f, timeout=%.4f, "
                    "collision=%.4f",
                    key,
                    data.get("bounds_mm", []),
                    data["success_rate"],
                    data["timeout_rate"],
                    data["collision_rate"],
                )

            eval_output = (
                self.data_dir
                / f"sac_eval_result_{mesh_name}.json"
            )
            with eval_output.open("w") as f:
                json.dump(results, f, indent=2)

        eval_output = self.data_dir / "sac_eval_result_all.json"
        with eval_output.open("w") as f:
            json.dump(all_sac_eval_results, f, indent=2)
        logger.info(
            "Saved all SAC eval results to %s", eval_output
        )

    def _eval_sac_on_pools(
        self,
        sac_trainer: PSACTrainer,
        sac_eval_pools: dict[int, dict[str, Any]],
        mesh_path: str,
    ) -> dict[str, Any]:
        """Run SAC evaluation on episode pools.

        Args:
            sac_trainer: Trained PSACTrainer instance.
            sac_eval_pools: Episode pools keyed by seed.
            mesh_path: Path to the mesh file.

        Returns:
            Dict with per-level evaluation results including
            action distribution and collision stats.
        """
        type_names = ExperienceExtractor.get_type_names()
        results_per_level: dict[str, Any] = {}
        sample_seed = self.sac_eval_seeds[0]
        num_levels = len(
            sac_eval_pools[sample_seed].get("levels", [])
        )

        for level_idx in range(num_levels):
            level_successes = 0
            level_timeouts = 0
            level_collisions = 0
            level_total = 0
            level_action_counts: dict[int, int] = {}
            level_collision_counts: dict[int, int] = {}
            level_total_steps = 0

            for eval_seed in self.sac_eval_seeds:
                level_pool = sac_eval_pools[eval_seed]["levels"][
                    level_idx
                ]

                np.random.seed(eval_seed)  # noqa: NPY002
                torch.manual_seed(eval_seed)
                env = LightweightEnv(mesh_path, seed=eval_seed)
                controller = RLGoalApproachController(
                    agent_id=(
                        f"sac_eval_L{level_idx}_{eval_seed}"
                    ),
                    config={**self.rl_config, "mode": "eval"},
                )
                interpreter = ActionInterpreter(env)

                for ep_data in level_pool:
                    start_pos = np.array(ep_data["start_pos"])
                    start_rot = np.array(ep_data["start_rot"])
                    env.reset(
                        position=start_pos, rotation=start_rot
                    )
                    goal_pose = np.concatenate([
                        np.array(ep_data["goal_pos"]),
                        np.array(ep_data["goal_rot"]),
                    ])
                    controller.set_new_goal(
                        goal_pose, start_pos
                    )
                    env.set_goal(goal_pose)

                    success = False
                    collision = False
                    ep_steps = 0

                    for _ in range(
                        sac_trainer.max_steps_per_goal
                    ):
                        current_pose = env.get_pose()
                        sensor_data = env.get_sensor_data()
                        state_raw = (
                            controller._compute_state(  # noqa: SLF001
                                current_pose, sensor_data
                            )
                        )
                        state = sac_trainer.normalize_state(
                            state_raw
                        )

                        state_t = torch.FloatTensor(
                            state.astype(np.float32)
                        ).unsqueeze(0)
                        with torch.no_grad():
                            at, ap, _, _ = (
                                sac_trainer.actor.sample_eval(
                                    state_t
                                )
                            )
                        action_type = at[0].item()
                        action_params = (
                            ap[0].numpy()
                            * sac_trainer.param_std
                            + sac_trainer.param_mean
                        )

                        # Track action
                        level_action_counts[action_type] = (
                            level_action_counts.get(
                                action_type, 0
                            )
                            + 1
                        )
                        level_total_steps += 1
                        ep_steps += 1

                        sensor_data = interpreter.execute(
                            action_type, action_params
                        )

                        current_pose = env.get_pose()
                        distance = float(
                            np.linalg.norm(
                                goal_pose[:3]
                                - current_pose[:3]
                            )
                        )

                        if distance < sac_trainer.goal_threshold:
                            success = True
                            break

                        depth = sensor_data.get("depth", 100.0)
                        if depth < _COLLISION_DEPTH_THRESHOLD:
                            collision = True
                            # Track collision
                            level_collision_counts[
                                action_type
                            ] = (
                                level_collision_counts.get(
                                    action_type, 0
                                )
                                + 1
                            )
                            break

                    level_total += 1
                    if success:
                        level_successes += 1
                    elif collision:
                        level_collisions += 1
                    else:
                        level_timeouts += 1

            count = max(level_total, 1)
            bounds = self.curriculum_levels[level_idx]

            # Action distribution
            total_actions = max(
                sum(level_action_counts.values()), 1
            )
            action_distribution = {
                type_names.get(k, f"type_{k}"): {
                    "count": v,
                    "rate": round(v / total_actions, 4),
                }
                for k, v in sorted(
                    level_action_counts.items()
                )
            }

            # Collision rate per type
            collision_rate_per_type = {}
            for (
                type_id,
                col_count,
            ) in level_collision_counts.items():
                total_calls = level_action_counts.get(
                    type_id, 0
                )
                name = type_names.get(
                    type_id, f"type_{type_id}"
                )
                collision_rate_per_type[name] = {
                    "collisions": col_count,
                    "total_calls": total_calls,
                    "rate": round(
                        col_count / max(total_calls, 1), 4
                    ),
                }

            results_per_level[f"level_{level_idx}"] = {
                "bounds_mm": list(bounds),
                "success_rate": level_successes / count,
                "timeout_rate": level_timeouts / count,
                "collision_rate": level_collisions / count,
                "action_distribution": action_distribution,
                "collision_stats": {
                    type_names.get(k, f"type_{k}"): v
                    for k, v in level_collision_counts.items()
                },
                "collision_rate_per_type": (
                    collision_rate_per_type
                ),
                "steps_per_success": round(
                    level_total_steps
                    / max(level_successes, 1),
                    1,
                ),
                "mean_episode_steps": round(
                    level_total_steps / max(level_total, 1),
                    1,
                ),
            }

        return results_per_level

    def _run_adaptive(self) -> None:
        """Run adaptive mode with Q-store + SAC arbitration."""
        logger.info("=" * 60)
        logger.info("Adaptive: %s", self.adaptive_mesh)
        logger.info("=" * 60)

        mesh_path = str(
            self.data_dir / f"{self.adaptive_mesh}.stl"
        )
        env = LightweightEnv(mesh_path)
        num_types = len(ExperienceExtractor.get_type_names())

        q_load_dir = (
            f"{self.unified_save_dir}"
            f"_seed_{self.train_seeds[0]}"
        )
        controller = RLGoalApproachController.load(
            q_load_dir,
            agent_id=f"{self.adaptive_mesh}_adaptive",
            config={**self.rl_config, "mode": "eval"},
        )

        sac_trainer = PSACTrainer(
            state_dim=self.rl_config.get("state_dim", 15),
            num_types=num_types,
        )
        sac_trainer.load(str(self.runs_dir / "sac_model"))

        manager = AdaptiveTrainingManager(
            controller=controller,
            env=env,
            config=self.rl_config,
            runs_dir=str(self.runs_dir),
            mesh_path=mesh_path,
        )
        manager.sac_trainer = sac_trainer

        for episode in range(self.adaptive_episodes):
            env.reset()
            start_pos = env.get_pose()[:3]
            goal_pose = env.get_random_surface_point(
                reference_pos=start_pos,
                min_dist=_ADAPTIVE_MIN_DIST,
                max_dist=_ADAPTIVE_MAX_DIST,
                max_attempts=2000,
                mesh_sample=True,
            )
            controller.set_new_goal(goal_pose, start_pos)
            env.set_goal(goal_pose)

            goals_before = (
                controller._total_goals_reached  # noqa: SLF001
            )
            for _ in range(_ADAPTIVE_MAX_STEPS):
                current_pose = env.get_pose()
                sensor_data = env.get_sensor_data()
                state = (
                    controller._compute_state(  # noqa: SLF001
                        current_pose, sensor_data
                    )
                )
                action_index, _source = manager.get_action(
                    state, current_pose, sensor_data
                )
                _state, done = controller.update_only(
                    current_pose, sensor_data, action_index
                )
                if done:
                    break
                env.step(action_index, controller.action_space)

            success = (
                controller._total_goals_reached  # noqa: SLF001
                > goals_before
            )
            transitions = (
                controller.success_trails.copy()
                if success
                else []
            )
            manager.on_episode_complete(
                success=success, transitions=transitions
            )
            manager.arbitrator.on_episode_end(success)

            if (episode + 1) % _ADAPTIVE_LOG_INTERVAL == 0:
                logger.info(
                    "Adaptive episode %d: %s",
                    episode + 1,
                    manager.get_stats(),
                )

        controller.save(str(self.runs_dir / "adaptive_q"))
        if manager.sac_trainer:
            manager.sac_trainer.save(
                str(self.runs_dir / "adaptive_sac")
            )
        logger.info(
            "Adaptive complete: %s", manager.get_stats()
        )

    @staticmethod
    def _filter_bc_transitions(
        transitions: list[Any],
        max_per_mesh: int = 10000,
    ) -> list[Any]:
        """Filter BC transitions for quality and balance.

        Keeps shorter trajectories (more efficient demonstrations)
        and balances across meshes.

        Args:
            transitions: All collected BC transitions.
            max_per_mesh: Maximum transitions per mesh.

        Returns:
            Filtered list of transitions.
        """
        mesh_id_to_name = {
            v: k
            for k, v in ExperienceExtractor.MESH_NAME_TO_ID.items()
        }

        # Group by mesh
        by_mesh: dict[int, list[Any]] = {}
        for tr in transitions:
            mesh_id = getattr(tr, "mesh_id", -1)
            if mesh_id not in by_mesh:
                by_mesh[mesh_id] = []
            by_mesh[mesh_id].append(tr)

        # Cap per mesh
        filtered: list[Any] = []
        for mesh_id, mesh_transitions in by_mesh.items():
            mesh_name = mesh_id_to_name.get(
                mesh_id, f"unknown_{mesh_id}"
            )
            if len(mesh_transitions) > max_per_mesh:
                # Keep random subset
                indices = np.random.permutation(
                    len(mesh_transitions)
                )[:max_per_mesh]
                mesh_transitions = [
                    mesh_transitions[i] for i in indices
                ]
                logger.info(
                    "BC filter: %s capped to %d transitions",
                    mesh_name,
                    max_per_mesh,
                )
            filtered.extend(mesh_transitions)

        logger.info(
            "BC filter: %d → %d transitions",
            len(transitions),
            len(filtered),
        )
        return filtered