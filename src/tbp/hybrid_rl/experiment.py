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

import datetime
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
from tbp.hybrid_rl.rl_goal_approach_controller import (
    RLGoalApproachController,
)
from tbp.hybrid_rl.sac_trainer import PSACTrainer
from tbp.hybrid_rl.ablation_runner import _maybe_save_visualization, visualize_agent_goal
from tbp.hybrid_rl.strategic_sac import StrategicSAC, StrategicBCTrainer
from tbp.hybrid_rl.episode_pools import _is_reachable_by_surface
from .arbitrator import sac_to_discrete
from .action_interpreter import ActionInterpreter

logger = logging.getLogger(__name__)

_ADAPTIVE_MIN_DIST = 10.0
_ADAPTIVE_MAX_DIST = 120.0
_ADAPTIVE_LOG_INTERVAL = 100
_SAC_WARMUP_STEPS = 5000
_BC_NUM_EPOCHS = 200
# _EVAL_EPSILON = 0.02
_COLLISION_DEPTH_THRESHOLD = 0.5


class RLGoalApproachExperiment:
    """Hydra-compatible experiment for RL goal approach training.

    Supports stages: Q-learning train, eval, BC train, SAC train,
    SAC eval, and adaptive mode. All parameters configured via YAML.

    Model storage convention::

        runs/
        ├── q_store_seed_11/     # Q-learning model
        ├── sac_seed_11/         # SAC model
        ├── bc_model/            # BC model (no seed)
        ├── adaptive_q_seed_11/  # Adaptive Q-store
        ├── adaptive_sac_seed_11/# Adaptive SAC
        └── meta/                # Training metadata

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
        self.output_dir = Path(
            self.config["logging"]["output_dir"]
        )
        self.run_name = self.config["logging"]["run_name"]
        self.visualise = self.config.get("visualise", False)

        # Directories
        self.data_dir = self.output_dir / "data"
        self.runs_dir = self.data_dir / "runs"
        self.scripts_dir = self.data_dir / "episode_scripts"

        # Pipeline stages
        self.do_train = self.config.get("do_train", False)
        self.do_eval = self.config.get("do_eval", False)
        self.do_heur_eval = self.config.get("do_heur_eval", False)
        self.do_bc_train = self.config.get("do_bc_train", False)
        self.do_sac_train = self.config.get(
            "do_sac_train", False
        )
        self.do_sac_eval = self.config.get(
            "do_sac_eval", False
        )
        self.do_adaptive = self.config.get(
            "do_adaptive", False
        )

        # Training config
        self.training_stages: list[dict[str, Any]] = (
            self.config.get("training_stages", [])
        )
        self.curriculum_levels: list[tuple[float, float]] = [
            tuple(level)
            for level in self.config.get(
                "curriculum_levels",
                [[10.0, 40.0], [20.0, 80.0], [40.0, 120.0]],
            )
        ]
        self.curriculum_filters: list[dict[str, Any]] = (
            self.config.get(
                "curriculum_filters", []
            )
        )
        self.train_seeds: list[int] = self.config.get(
            "train_seeds", [11]
        )
        self.eval_seeds: list[int] = self.config.get(
            "eval_seeds", [44]
        )
        self.sac_eval_seeds: list[int] = self.config.get(
            "sac_eval_seeds", [77, 88, 99]
        )
        self.sac_seed: int = self.config.get(
            "sac_seed", self.train_seeds[0]
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

        # RL config
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

        # SAC config
        self.sac_config: dict[str, Any] = self.config.get(
            "sac_config", {}
        )
        self.sac_meshes: list[str] = self.config.get(
            "sac_meshes",
            ["cube", "cylinder", "mug", "cup"],
        )
        self.sac_episodes_per_mesh: dict[str, int] = (
            self.config.get(
                "sac_episodes_per_mesh",
                {
                    "cube": 1000,
                    "cylinder": 1500,
                    "mug": 2000,
                    "cup": 2000,
                },
            )
        )

        # Adaptive config
        self.adaptive_mesh: str = self.config.get(
            "adaptive_mesh", "cup"
        )
        self.adaptive_episodes: int = self.config.get(
            "adaptive_episodes", 3000
        )

        # Eval meshes
        self.eval_meshes: list[str] = self.config.get(
            "eval_meshes",
            ["cube", "cylinder", "mug", "cup"],
        )

    # ══════════════════════════════════════════════════════
    # Model path helpers
    # ══════════════════════════════════════════════════════

    def _q_model_dir(self, seed: int) -> str:
        """Get Q-store directory path for a seed.

        Args:
            seed: Training seed.

        Returns:
            Path string to Q-store directory.
        """
        return str(self.runs_dir / f"q_store_seed_{seed}")

    def _sac_model_dir(self, seed: int) -> str:
        """Get SAC model directory path for a seed.

        Args:
            seed: Training seed.

        Returns:
            Path string to SAC model directory.
        """
        return str(self.runs_dir / f"sac_seed_{seed}")

    def _resolve_load_dir(
        self,
        load_mode: str | None,
        model_type: str,
        seed: int,
    ) -> str | None:
        """Resolve load_mode to actual directory path.

        Args:
            load_mode: null (from scratch), "auto" (standard path),
                or explicit path string.
            model_type: "q_store" or "sac".
            seed: Training seed.

        Returns:
            Directory path or None.
        """
        if load_mode is None:
            return None
        if load_mode == "auto":
            if model_type == "q_store":
                return self._q_model_dir(seed)
            if model_type == "sac":
                return self._sac_model_dir(seed)
            return None
        return str(load_mode)

    def _save_meta(
        self,
        model_type: str,
        seed: int | None,
        stats: dict[str, Any],
        meshes: list[str],
    ) -> None:
        """Save training metadata to meta/ directory.

        Args:
            model_type: "q_store", "sac", or "bc".
            seed: Training seed (None for BC).
            stats: Training statistics.
            meshes: List of meshes trained on.
        """
        meta_dir = self.runs_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        meta_name = (
            f"{model_type}_seed_{seed}.json"
            if seed is not None
            else f"{model_type}.json"
        )
        meta = {
            "model_type": model_type,
            "seed": seed,
            "created": datetime.datetime.now(
                tz=datetime.timezone.utc
            ).isoformat(),
            "meshes": meshes,
            "stats": stats,
        }
        meta_path = meta_dir / meta_name
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Meta saved to %s", meta_path)

    @staticmethod
    def _balance_by_action_type(
        transitions: list,
        min_pct: float = 0.01,
        min_absolute: int = 100,
    ) -> list:
        """Ensure minimum representation of each
        action type.

        Each type gets at least max(min_absolute,
        total * min_pct) examples.
        Only oversamples types that already have
        >= min_absolute examples.

        Args:
            transitions: BC transitions.
            min_pct: Minimum fraction of total (0.01=1%).
            min_absolute: Minimum raw examples to
                qualify for oversampling.

        Returns:
            Balanced transitions.
        """
        type_names = (
            ExperienceExtractor.get_type_names()
        )

        by_type: dict[int, list] = {}
        for tr in transitions:
            tid = tr.action_type
            if tid not in by_type:
                by_type[tid] = []
            by_type[tid].append(tr)

        total = len(transitions)
        min_target = max(
            min_absolute,
            int(total * min_pct),
        )

        logger.info(
            "Action type balance: total=%d, "
            "min_target=%d (%.1f%% or %d)",
            total,
            min_target,
            min_pct * 100,
            min_absolute,
        )
        logger.info("  BEFORE:")
        for tid in sorted(by_type.keys()):
            name = type_names.get(
                tid, f"type_{tid}"
            )
            logger.info(
                "    %s: %d (%.2f%%)",
                name,
                len(by_type[tid]),
                100.0 * len(by_type[tid])
                / max(total, 1),
            )

        balanced = list(transitions)
        for tid, trs in by_type.items():
            name = type_names.get(
                tid, f"type_{tid}"
            )
            if (
                len(trs) < min_target
                and len(trs) >= min_absolute
            ):
                need = min_target - len(trs)
                indices = np.random.choice(
                    len(trs), need, replace=True
                )
                extra = [trs[i] for i in indices]
                balanced.extend(extra)
                logger.info(
                    "    %s: oversampled %d → %d",
                    name,
                    len(trs),
                    min_target,
                )
            elif len(trs) < min_absolute:
                logger.info(
                    "    %s: skipped (%d < %d "
                    "min_absolute)",
                    name,
                    len(trs),
                    min_absolute,
                )

        logger.info(
            "  Action type balance: %d → %d",
            total,
            len(balanced),
        )

        return balanced
    # ══════════════════════════════════════════════════════
    # Context manager
    # ══════════════════════════════════════════════════════

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
        if self.do_heur_eval:
            self._run_heuristic_eval()
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

    # ══════════════════════════════════════════════════════
    # Q-learning training
    # ══════════════════════════════════════════════════════

    def _run_train(self) -> None:
        """Run Q-learning training across all stages."""
        logger.info("=" * 60)
        logger.info("Q-Learning Training (all stages)")
        logger.info("=" * 60)

        trained_meshes: list[str] = []

        for stage_idx, stage in enumerate(
            self.training_stages
        ):
            mesh_name = stage["mesh"]
            mesh_path = str(
                self.data_dir / f"{mesh_name}.stl"
            )
            episodes = stage["episodes"]
            epsilon_start = stage["epsilon_start"]
            load_mode = stage.get("load_mode")

            logger.info(
                "STAGE %d/%d: %s, episodes=%d, eps=%.2f, "
                "load_mode=%s",
                stage_idx + 1,
                len(self.training_stages),
                mesh_name,
                episodes,
                epsilon_start,
                load_mode,
            )

            stage_cfg = {
                **self.rl_config,
                "epsilon_start": epsilon_start,
                "epsilon_min": stage.get(
                    "epsilon_min", 0.05
                ),
                "num_episodes": episodes,
                "unfreeze_normalization": (
                    load_mode is not None
                ),
            }

            train_pools = get_or_generate_pools(
                mesh_path=mesh_path,
                seeds=self.train_seeds,
                episodes_per_level=episodes,
                scripts_dir=self.scripts_dir,
                curriculum_levels=self.curriculum_levels,
                regenerate=self.regenerate_scripts,
                prefix=f"train_{mesh_name}",
                curriculum_filters=self.curriculum_filters,
            )

            for seed in self.train_seeds:
                save_dir = self._q_model_dir(seed)
                load_dir = self._resolve_load_dir(
                    load_mode, "q_store", seed
                )

                seed_pools = train_pools.get(seed)
                episode_pools = (
                    seed_pools.get("levels")
                    if seed_pools
                    else None
                )

                curriculum_config = {
                    "levels": self.curriculum_levels,
                    "filters": self.curriculum_filters,
                    "promote_threshold": (
                        self.promote_threshold
                    ),
                    "promote_window": self.promote_window,
                }

                run_result = run_episodes(
                    mesh_dir=str(self.data_dir),
                    save_dir=save_dir,
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
                    / f"train_result_{mesh_name}"
                    f"_seed_{seed}"
                    f"_stage_{stage_idx}.json"
                )
                stage_data = {
                    "stage": stage_idx,
                    "mesh": mesh_name,
                    "seed": seed,
                    "epsilon_start": epsilon_start,
                    "epsilon_min": stage.get(
                        "epsilon_min", 0.05
                    ),
                    "load_mode": load_mode,
                    "episodes": episodes,
                    "success_rate": run_result.get(
                        "success_rate"
                    ),
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

            trained_meshes.append(mesh_name)

        # Save meta
        for seed in self.train_seeds:
            self._save_meta(
                "q_store",
                seed,
                {"trained_meshes": trained_meshes},
                trained_meshes,
            )

    # ══════════════════════════════════════════════════════
    # Q-learning evaluation
    # ══════════════════════════════════════════════════════

    def _run_eval(self) -> None:
        """Run Q-learning eval on all meshes, collect BC data."""
        logger.info("=" * 60)
        logger.info("Eval on all meshes")
        logger.info("=" * 60)

        eval_cfg = {
            **self.rl_config,
            "mode": "eval",
        }

        all_bc_transitions: list[Any] = []
        all_eval_results: dict[str, Any] = {}

        for mesh_name in self.eval_meshes:
            mesh_path = str(
                self.data_dir / f"{mesh_name}.stl"
            )
            logger.info("Evaluating: %s", mesh_name)

            eval_pools = get_or_generate_pools(
                mesh_path=mesh_path,
                seeds=self.eval_seeds,
                episodes_per_level=(
                    self.eval_episodes_per_level
                ),
                scripts_dir=self.scripts_dir,
                curriculum_levels=self.curriculum_levels,
                regenerate=self.regenerate_scripts,
                prefix=f"eval_{mesh_name}",
                curriculum_filters=self.curriculum_filters,
            )

            eval_results, bc_transitions = (
                run_eval_per_seed(
                    data_dir=self.data_dir,
                    runs_dir=self.runs_dir,
                    mesh_path=mesh_path,
                    train_seeds=self.train_seeds,
                    eval_seeds=self.eval_seeds,
                    variant="q_store",
                    eval_cfg=eval_cfg,
                    eval_pools=eval_pools,
                    collect_bc=True,
                    episodes_per_level=(
                        self.eval_episodes_per_level
                    ),
                    mesh_name=mesh_name,
                    visualise=self.visualise
                )
            )

            all_eval_results[mesh_name] = eval_results
            all_bc_transitions.extend(bc_transitions)

            mesh_output = (
                self.data_dir
                / f"eval_result_{mesh_name}.json"
            )
            with mesh_output.open("w") as f:
                json.dump(eval_results, f, indent=2)

        eval_output = self.data_dir / "eval_result_all.json"
        with eval_output.open("w") as f:
            json.dump(all_eval_results, f, indent=2)

        if all_bc_transitions:
            bc_output = self.data_dir / "bc_data.pkl"
            with bc_output.open("wb") as f:
                pickle.dump(  # noqa: S301
                    all_bc_transitions, f
                )
            logger.info(
                "BC data: %d transitions",
                len(all_bc_transitions),
            )

    # ══════════════════════════════════════════════════════
    # Behavioral Cloning
    # ══════════════════════════════════════════════════════

    def _run_heuristic_eval(self) -> list[Any]:
        """Collect BC data using pure heuristic policy.

        Runs evaluation with epsilon=1.0 (100% heuristic) on all
        eval meshes to collect expert demonstrations with natural
        action distribution.

        Returns:
            List of PSACTransition with level tags.
        """
        logger.info("=" * 60)
        logger.info("Heuristic Expert Data Collection")
        logger.info("=" * 60)

        heuristic_cfg = {
            **self.rl_config,
            "mode": "eval",
            "eval_epsilon": 1.0,
            "temperature_override": 0.01,
            "strategic_eval_epsilon": 1.0,
            "air_start_enabled": True,
            "air_start_in_eval": True,
        }

        all_heuristic_transitions: list[Any] = []

        for mesh_name in self.eval_meshes:
            mesh_path = str(
                self.data_dir / f"{mesh_name}.stl"
            )
            logger.info(
                "Heuristic eval: %s", mesh_name
            )

            heuristic_pools = get_or_generate_pools(
                mesh_path=mesh_path,
                seeds=self.eval_seeds,
                episodes_per_level=(
                    self.eval_episodes_per_level
                ),
                scripts_dir=self.scripts_dir,
                curriculum_levels=self.curriculum_levels,
                regenerate=self.regenerate_scripts,
                prefix=f"heuristic_{mesh_name}",
                curriculum_filters=self.curriculum_filters,
            )

            _, heuristic_transitions = (
                run_eval_per_seed(
                    data_dir=self.data_dir,
                    runs_dir=self.runs_dir,
                    mesh_path=mesh_path,
                    train_seeds=self.train_seeds,
                    eval_seeds=self.eval_seeds,
                    variant="q_store",
                    eval_cfg=heuristic_cfg,
                    eval_pools=heuristic_pools,
                    collect_bc=True,
                    episodes_per_level=(
                        self.eval_episodes_per_level
                    ),
                    mesh_name=mesh_name,
                    visualise=self.visualise,
                    log_prefix="heuristic",
                )
            )

            all_heuristic_transitions.extend(
                heuristic_transitions
            )
            logger.info(
                "Heuristic %s: %d transitions",
                mesh_name,
                len(heuristic_transitions),
            )

        heuristic_cache_path = self.data_dir / "bc_data_heuristic.pkl"
        with heuristic_cache_path.open("wb") as f:
            pickle.dump(all_heuristic_transitions, f)
            logger.info(
                "Saved %d heuristic transitions to %s",
                len(heuristic_transitions),
                heuristic_cache_path,
            )
        logger.info(
            "Total heuristic transitions: %d",
            len(all_heuristic_transitions),
        )
        return all_heuristic_transitions

    def _run_bc_train(self) -> None:
        """Train BC model from collected data.

        Combines Q-store eval data with heuristic expert data,
        balances by (mesh, level), and trains BC model.
        """
        logger.info("=" * 60)
        logger.info("BC Training")
        logger.info("=" * 60)

        # Collect heuristic expert data (with caching)
        bc_data_combined_path = (
            self.data_dir / "bc_data_combined.pkl"
        )
        if bc_data_combined_path.exists():
            logger.info(
                "Loading cached bc_data_combined "
                "data from %s",
                bc_data_combined_path,
            )
            with bc_data_combined_path.open("rb") as f:
                bc_transitions = pickle.load(f)
            logger.info(
                "Loaded %d cached bc_data_combined "
                "transitions",
                len(bc_transitions),
            )
        else:
            # ═══ Load heuristic data (primary source) ═══
            heuristic_cache_path = (
                self.data_dir / "bc_data_heuristic.pkl"
            )
            if heuristic_cache_path.exists():
                logger.info(
                    "Loading cached heuristic data "
                    "from %s",
                    heuristic_cache_path,
                )
                with heuristic_cache_path.open(
                    "rb"
                ) as f:
                    heuristic_transitions = (
                        pickle.load(f)
                    )
                logger.info(
                    "Loaded %d cached heuristic "
                    "transitions",
                    len(heuristic_transitions),
                )
            else:
                heuristic_transitions = (
                    self._run_heuristic_eval()
                )
                logger.info(
                    "Collected %d heuristic "
                    "transitions",
                    len(heuristic_transitions),
                )

            # ═══ Optional: add small fraction of
            #     Q-store eval data ═══
            q_store_fraction = float(
                self.config.get(
                    "bc_q_store_fraction", 0.0
                )
            )
            q_store_transitions_used = []

            if q_store_fraction > 0.0:
                bc_path = (
                    self.data_dir / "bc_data.pkl"
                )
                if bc_path.exists():
                    with bc_path.open("rb") as f:
                        q_store_transitions = (
                            pickle.load(f)
                        )
                    max_q = int(
                        len(heuristic_transitions)
                        * q_store_fraction
                    )
                    if (
                        len(q_store_transitions)
                        > max_q
                    ):
                        indices = (
                            np.random.permutation(
                                len(
                                    q_store_transitions
                                )
                            )[:max_q]
                        )
                        q_store_transitions_used = [
                            q_store_transitions[i]
                            for i in indices
                        ]
                    else:
                        q_store_transitions_used = (
                            q_store_transitions
                        )
                    logger.info(
                        "Q-store data: %d → %d "
                        "(fraction=%.1f%%)",
                        len(q_store_transitions),
                        len(
                            q_store_transitions_used
                        ),
                        q_store_fraction * 100,
                    )

            # ═══ Combine ═══
            bc_transitions = (
                heuristic_transitions
                + q_store_transitions_used
            )
            logger.info(
                "BC data: %d heuristic + %d Q-store "
                "= %d total",
                len(heuristic_transitions),
                len(q_store_transitions_used),
                len(bc_transitions),
            )

            # ═══ Balance by (mesh, level) ═══
            bc_total_target = self.config.get(
                "bc_total_target", None
            )
            if bc_total_target is not None:
                bc_transitions = (
                    self._filter_bc_transitions(
                        bc_transitions,
                        total_target=int(
                            bc_total_target
                        ),
                    )
                )

            # ═══ Balance rare action types ═══
            bc_transitions = (
                self._balance_by_action_type(
                    bc_transitions,
                    min_pct=float(
                        self.config.get(
                            "bc_action_min_pct", 0.01
                        )
                    ),
                    min_absolute=int(
                        self.config.get(
                            "bc_action_min_absolute",
                            100,
                        )
                    ),
                )
            )
            
            np.random.shuffle(bc_transitions)

            logger.info(
                "BC data: %d total after balancing",
                len(bc_transitions),
            )

            # Save combined
            with bc_data_combined_path.open(
                "wb"
            ) as f:
                pickle.dump(bc_transitions, f)

        # Train BC
        num_types = len(
            ExperienceExtractor.get_type_names()
        )
        trainer = BCTrainer(
            state_dim=self.rl_config.get("state_dim", 20),
            num_types=num_types,
        )
        trainer.train(
            bc_transitions, num_epochs=_BC_NUM_EPOCHS
        )
        trainer.save(str(self.runs_dir / "bc_model"))

        bc_stats = trainer.get_training_stats(
            bc_transitions
        )
        bc_stats_path = (
            self.data_dir / "bc_train_result.json"
        )
        with bc_stats_path.open("w") as f:
            json.dump(bc_stats, f, indent=2)

        # ═══ Strategic BC Training ═══
        logger.info("=" * 40)
        logger.info("Strategic BC Training (two separate)")
        logger.info("=" * 40)

        for seed in self.train_seeds:
            q_dir = self._q_model_dir(seed)
            if not (
                Path(q_dir) / "config.json"
            ).exists():
                logger.warning(
                    "Q-store not found at %s, "
                    "skipping Strategic BC",
                    q_dir,
                )
                continue

            controller = RLGoalApproachController.load(
                q_dir,
                agent_id=f"strategic_bc_seed_{seed}",
                config={
                    **self.rl_config,
                    "mode": "eval",
                },
            )

            strategic_bc_dir = str(
                self.runs_dir / "strategic_bc_model"
            )
            Path(strategic_bc_dir).mkdir(
                parents=True, exist_ok=True
            )

            # 1. Detach BC
            detach_bc = StrategicBCTrainer(state_dim=5)
            has_detach = (
                detach_bc.prepare_data_from_q_store(
                    controller.strategic_detach,
                )
            )
            if has_detach:
                detach_bc.train(num_epochs=100)
                torch.save(
                    detach_bc.get_actor_weights(),
                    Path(strategic_bc_dir)
                    / "strategic_detach_actor.pt",
                )
                s_mean, s_std = (
                    detach_bc.get_normalization()
                )
                np.savez(
                    Path(strategic_bc_dir)
                    / "strategic_detach_norm.npz",
                    state_mean=s_mean,
                    state_std=s_std,
                )
                logger.info(
                    "Strategic Detach BC saved to %s",
                    strategic_bc_dir,
                )
            else:
                logger.warning(
                    "No data for Strategic Detach BC"
                )

            # 2. Direction BC
            direction_bc = StrategicBCTrainer(
                state_dim=5
            )
            has_direction = (
                direction_bc.prepare_data_from_q_store(
                    controller.strategic_direction,
                )
            )
            if has_direction:
                direction_bc.train(num_epochs=100)
                torch.save(
                    direction_bc.get_actor_weights(),
                    Path(strategic_bc_dir)
                    / "strategic_direction_actor.pt",
                )
                s_mean, s_std = (
                    direction_bc.get_normalization()
                )
                np.savez(
                    Path(strategic_bc_dir)
                    / "strategic_direction_norm.npz",
                    state_mean=s_mean,
                    state_std=s_std,
                )
                logger.info(
                    "Strategic Direction BC saved "
                    "to %s",
                    strategic_bc_dir,
                )
            else:
                logger.warning(
                    "No data for Strategic "
                    "Direction BC"
                )

            # Save strategic BC stats
            strategic_bc_stats = {
                "seed": seed,
                "detach": {
                    "has_data": has_detach,
                    "store_points": len(
                        controller.strategic_detach
                        .points
                    ),
                    "train_size": (
                        len(detach_bc.train_states)
                        if has_detach
                        and hasattr(
                            detach_bc, "train_states"
                        )
                        else 0
                    ),
                    "val_size": (
                        len(detach_bc.val_states)
                        if has_detach
                        and hasattr(
                            detach_bc, "val_states"
                        )
                        else 0
                    ),
                },
                "direction": {
                    "has_data": has_direction,
                    "store_points": len(
                        controller.strategic_direction
                        .points
                    ),
                    "train_size": (
                        len(
                            direction_bc.train_states
                        )
                        if has_direction
                        and hasattr(
                            direction_bc,
                            "train_states",
                        )
                        else 0
                    ),
                    "val_size": (
                        len(direction_bc.val_states)
                        if has_direction
                        and hasattr(
                            direction_bc,
                            "val_states",
                        )
                        else 0
                    ),
                },
            }
            strategic_stats_path = (
                self.data_dir
                / "strategic_bc_train_result.json"
            )
            with strategic_stats_path.open("w") as f:
                json.dump(
                    strategic_bc_stats, f, indent=2
                )
            logger.info(
                "Strategic BC stats saved to %s",
                strategic_stats_path,
            )
        self._save_meta(
            "bc",
            None,
            {
                **bc_stats,
                "heuristic_transitions": len(
                    bc_transitions
                ),
                "q_store_fraction": float(
                    self.config.get(
                        "bc_q_store_fraction", 0.0
                    )
                ),
            },
            list(self.eval_meshes),
        )
        logger.info(
            "BC complete: val_acc=%.4f, "
            "transitions=%d",
            bc_stats["val_accuracy"],
            bc_stats["total_transitions"],
        )

    # ══════════════════════════════════════════════════════
    # SAC training
    # ══════════════════════════════════════════════════════

    def _run_sac_train(self) -> None:
        """Train P-SAC model sequentially on all meshes."""
        logger.info("=" * 60)
        logger.info("P-SAC Training (seed=%d)", self.sac_seed)
        logger.info("=" * 60)

        num_types = len(
            ExperienceExtractor.get_type_names()
        )

        sac_cfg = {
            k: v
            for k, v in self.sac_config.items()
            if k != "load_mode"
        }
        trainer = PSACTrainer(
            state_dim=self.rl_config.get("state_dim", 20),
            num_types=num_types,
            **sac_cfg,
        )

        load_mode = self.sac_config.get("load_mode")
        load_dir = self._resolve_load_dir(
            load_mode, "sac", self.sac_seed
        )

        if load_dir:
            trainer.load(load_dir)
            bc_combined_path = (
                self.data_dir
                / "bc_data_combined.pkl"
            )
            bc_path = (
                bc_combined_path
                if bc_combined_path.exists()
                else self.data_dir / "bc_data.pkl"
            )
            if bc_path.exists():
                with bc_path.open("rb") as f:
                    bc_data = pickle.load(f)
                
                bc_norm = (
                    trainer._normalize_bc_transitions(  # noqa: SLF001
                        bc_data
                    )
                )
                trainer.buffer.load_bc_data(bc_norm)
                trainer.bc_data = bc_data
            logger.info(
                "SAC loaded from %s for fine-tuning",
                load_dir,
            )
        else:
            # Use combined BC data (heuristic + optional Q-store)
            bc_combined_path = (
                self.data_dir / "bc_data_combined.pkl"
            )
            bc_data_path = (
                str(bc_combined_path)
                if bc_combined_path.exists()
                else str(
                    self.data_dir / "bc_data.pkl"
                )
            )
            trainer.load_bc(
                bc_model_dir=str(
                    self.runs_dir / "bc_model"
                ),
                bc_data_path=bc_data_path,
            )
            
        sac_model_dir = self._sac_model_dir(self.sac_seed)

        # ═══ Load Strategic SACs (only if enabled) ═══
        use_strategic = self.sac_config.get(
            "use_strategic_override", False
        )
        if use_strategic:
            # Load Strategic BC into controller's Strategic SAC
            # ═══ Load Strategic SACs from BC ═══
            strategic_bc_dir = str(
                self.runs_dir / "strategic_bc_model"
            )

            strategic_detach_sac = None
            strategic_direction_sac = None

            # Detach SAC
            detach_actor_path = (
                Path(strategic_bc_dir)
                / "strategic_detach_actor.pt"
            )
            if detach_actor_path.exists():
                strategic_detach_sac = StrategicSAC(
                    state_dim=5
                )
                strategic_detach_sac.actor.load_state_dict(
                    torch.load(
                        detach_actor_path,
                        weights_only=True,
                    )
                )
                norm = np.load(
                    Path(strategic_bc_dir)
                    / "strategic_detach_norm.npz"
                )
                strategic_detach_sac._state_mean = (
                    norm["state_mean"]
                )
                strategic_detach_sac._state_std = (
                    norm["state_std"]
                )
                strategic_detach_sac._norm_frozen = True

                # Warm start buffer from Q-store
                for seed in self.train_seeds:
                    q_dir = self._q_model_dir(seed)
                    if (
                        Path(q_dir) / "config.json"
                    ).exists():
                        q_ctrl = (
                            RLGoalApproachController.load(
                                q_dir,
                                agent_id="strat_loader",
                                config={
                                    **self.rl_config,
                                    "mode": "eval",
                                },
                            )
                        )
                        strategic_detach_sac.warm_start_from_q_store(
                            q_ctrl.strategic_detach
                        )
                        break

                logger.info(
                    "Strategic Detach SAC loaded "
                    "(buf=%d)",
                    strategic_detach_sac.buffer_size,
                )

            # Direction SAC
            direction_actor_path = (
                Path(strategic_bc_dir)
                / "strategic_direction_actor.pt"
            )
            if direction_actor_path.exists():
                strategic_direction_sac = StrategicSAC(
                    state_dim=5
                )
                strategic_direction_sac.actor.load_state_dict(
                    torch.load(
                        direction_actor_path,
                        weights_only=True,
                    )
                )
                norm = np.load(
                    Path(strategic_bc_dir)
                    / "strategic_direction_norm.npz"
                )
                strategic_direction_sac._state_mean = (
                    norm["state_mean"]
                )
                strategic_direction_sac._state_std = (
                    norm["state_std"]
                )
                strategic_direction_sac._norm_frozen = True

                # Warm start buffer
                for seed in self.train_seeds:
                    q_dir = self._q_model_dir(seed)
                    if (
                        Path(q_dir) / "config.json"
                    ).exists():
                        q_ctrl = (
                            RLGoalApproachController.load(
                                q_dir,
                                agent_id="strat_loader2",
                                config={
                                    **self.rl_config,
                                    "mode": "eval",
                                },
                            )
                        )
                        strategic_direction_sac.warm_start_from_q_store(
                            q_ctrl.strategic_direction
                        )
                        break

                logger.info(
                    "Strategic Direction SAC loaded "
                    "(buf=%d)",
                    strategic_direction_sac.buffer_size,
                )

            # Pass to trainer
            trainer.load_strategic(
                strategic_detach_sac=strategic_detach_sac,
                strategic_direction_sac=(
                    strategic_direction_sac
                ),
            )

        for mesh_name in self.sac_meshes:
            mesh_path = str(
                self.data_dir / f"{mesh_name}.stl"
            )
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
                seeds=[self.sac_seed],
                episodes_per_level=num_episodes,
                scripts_dir=self.scripts_dir,
                curriculum_levels=self.curriculum_levels,
                regenerate=self.regenerate_scripts,
                prefix=f"sac_train_{mesh_name}",
                curriculum_filters=self.curriculum_filters,
            )

            trainer.start_mesh_tracking(mesh_name)

            is_first = mesh_name == self.sac_meshes[0]
            trainer.train(
                env=env,
                controller=controller,
                num_episodes=num_episodes,
                warmup_steps=(
                    _SAC_WARMUP_STEPS
                    if is_first and not load_dir
                    else 0
                ),
                save_dir=sac_model_dir,
                curriculum_levels=self.curriculum_levels,
                promote_threshold=self.promote_threshold,
                promote_window=self.promote_window,
                episode_pools=sac_pools[self.sac_seed],
                visualise=self.visualise,
                mesh_name=mesh_name,
            )

            sac_stats = trainer.get_training_stats()
            result_path = (
                self.data_dir
                / f"sac_train_result_{mesh_name}.json"
            )
            with result_path.open("w") as f:
                json.dump(sac_stats, f, indent=2)
            mesh_stat = sac_stats["mesh_stats"].get(
                mesh_name, {}
            )
            logger.info(
                "SAC %s: rate=%.3f",
                mesh_name,
                mesh_stat.get("success_rate", 0),
            )

        final_stats = trainer.get_training_stats()
        final_path = (
            self.data_dir / "sac_train_result_all.json"
        )
        with final_path.open("w") as f:
            json.dump(final_stats, f, indent=2)

        self._save_meta(
            "sac",
            self.sac_seed,
            final_stats,
            list(self.sac_meshes),
        )
        logger.info(
            "SAC training complete. Saved to %s",
            sac_model_dir,
        )

    # ══════════════════════════════════════════════════════
    # SAC evaluation
    # ══════════════════════════════════════════════════════

    def _run_sac_eval(self) -> None:
        """Evaluate P-SAC model on all meshes."""
        logger.info("=" * 60)
        logger.info("P-SAC Eval (seed=%d)", self.sac_seed)
        logger.info("=" * 60)

        num_types = len(
            ExperienceExtractor.get_type_names()
        )
        sac_trainer = PSACTrainer(
            state_dim=self.rl_config.get("state_dim", 20),
            num_types=num_types,
        )
        sac_trainer.load(
            self._sac_model_dir(self.sac_seed)
        )

        all_results: dict[str, Any] = {}

        for mesh_name in self.eval_meshes:
            mesh_path = str(
                self.data_dir / f"{mesh_name}.stl"
            )
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
                curriculum_filters=self.curriculum_filters,
            )

            results = self._eval_sac_on_pools(
                sac_trainer=sac_trainer,
                sac_eval_pools=sac_eval_pools,
                mesh_path=mesh_path,
            )
            all_results[mesh_name] = results

            for key in sorted(results.keys()):
                data = results[key]
                logger.info(
                    "  %s (%smm): success=%.4f, "
                    "timeout=%.4f, collision=%.4f",
                    key,
                    data.get("bounds_mm", []),
                    data["success_rate"],
                    data["timeout_rate"],
                    data["collision_rate"],
                )

            eval_out = (
                self.data_dir
                / f"sac_eval_result_{mesh_name}.json"
            )
            with eval_out.open("w") as f:
                json.dump(results, f, indent=2)

        all_out = (
            self.data_dir / "sac_eval_result_all.json"
        )
        with all_out.open("w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(
            "Saved all SAC eval results to %s", all_out
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
            Dict with per-level evaluation results.
        """
        type_names = ExperienceExtractor.get_type_names()
        results: dict[str, Any] = {}
        sample_seed = self.sac_eval_seeds[0]
        num_levels = len(
            sac_eval_pools[sample_seed].get("levels", [])
        )

        for level_idx in range(num_levels):
            successes = 0
            timeouts = 0
            collisions = 0
            total = 0
            action_counts: dict[int, int] = {}
            collision_counts: dict[int, int] = {}
            total_steps = 0

            for eval_seed in self.sac_eval_seeds:
                pool = sac_eval_pools[eval_seed][
                    "levels"
                ][level_idx]

                np.random.seed(eval_seed)  # noqa: NPY002
                torch.manual_seed(eval_seed)
                env = LightweightEnv(
                    mesh_path, seed=eval_seed
                )
                controller = RLGoalApproachController(
                    agent_id=(
                        f"sac_eval_L{level_idx}"
                        f"_{eval_seed}"
                    ),
                    config={
                        **self.rl_config,
                        "mode": "eval",
                    },
                )
                interpreter = ActionInterpreter(env)

                for ep_data in pool:
                    start_pos = np.array(
                        ep_data["start_pos"]
                    )
                    start_rot = np.array(
                        ep_data["start_rot"]
                    )
                    env.reset(
                        position=start_pos,
                        rotation=start_rot,
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

                    for _ in range(
                        sac_trainer.max_steps_per_goal
                    ):
                        pose = env.get_pose()
                        sensor = env.get_sensor_data()
                        state_raw = (
                            controller._compute_state(
                                pose, sensor
                            )
                        )
                        state = (
                            sac_trainer.normalize_state(
                                state_raw
                            )
                        )

                        # Strategic override (if enabled)
                        strategic_type = None
                        if (
                            sac_trainer
                            .use_strategic_override
                        ):
                            strategic_type, _ = (
                                sac_trainer
                                ._strategic_detach_check(
                                    state_raw,
                                    controller,
                                    sensor,
                                )
                            )

                        if (
                            strategic_type
                            is not None
                        ):
                            atype = strategic_type
                            aparams = np.zeros(
                                3,
                                dtype=np.float32,
                            )
                        else:
                            if (
                                sac_trainer
                                .use_strategic_override
                            ):
                                dir_phase = (
                                    sac_trainer
                                    ._strategic_direction_check(
                                        state_raw,
                                        controller,
                                        sensor,
                                        pose,
                                    )
                                )
                                if (
                                    dir_phase
                                    is not None
                                ):
                                    controller._current_phase = (
                                        dir_phase
                                    )

                            state_t = (
                                torch.FloatTensor(
                                    state.astype(
                                        np.float32
                                    )
                                ).unsqueeze(0)
                            )
                            with torch.no_grad():
                                at, ap, _, _ = (
                                    sac_trainer
                                    .actor
                                    .sample_eval(
                                        state_t
                                    )
                                )
                            atype = at[0].item()
                            aparams = (
                                ap[0].numpy()
                                * sac_trainer
                                .param_std
                                + sac_trainer
                                .param_mean
                            )

                            # Action masks
                            atype, aparams = (
                                sac_trainer
                                ._apply_action_masks(
                                    atype,
                                    aparams,
                                    state_raw,
                                    state,
                                    controller,
                                )
                            )

                        action_counts[atype] = (
                            action_counts.get(
                                atype, 0
                            )
                            + 1
                        )
                        total_steps += 1

                        sensor = (
                            interpreter.execute(
                                atype, aparams
                            )
                        )
                        pose = env.get_pose()

                        # Update tracking
                        next_raw = (
                            controller
                            ._compute_state(
                                pose, sensor
                            )
                        )
                        sac_trainer._update_controller_tracking(
                            controller,
                            next_raw,
                            sensor,
                            pose,
                            atype,
                        )

                        dist = float(
                            np.linalg.norm(
                                goal_pose[:3]
                                - pose[:3]
                            )
                        )

                        if (
                            dist
                            < sac_trainer
                            .goal_threshold
                        ):
                            success = True
                            break

                        depth = sensor.get(
                            "depth", 100.0
                        )
                        if (
                            depth
                            < _COLLISION_DEPTH_THRESHOLD
                        ):
                            collision = True
                            collision_counts[
                                atype
                            ] = (
                                collision_counts.get(
                                    atype, 0
                                )
                                + 1
                            )
                            break

                    total += 1
                    if success:
                        successes += 1
                    elif collision:
                        collisions += 1
                    else:
                        timeouts += 1

            count = max(total, 1)
            bounds = self.curriculum_levels[level_idx]
            total_act = max(
                sum(action_counts.values()), 1
            )

            collision_rate_per_type = {}
            for tid, cc in collision_counts.items():
                tc = action_counts.get(tid, 0)
                name = type_names.get(
                    tid, f"type_{tid}"
                )
                collision_rate_per_type[name] = {
                    "collisions": cc,
                    "total_calls": tc,
                    "rate": round(
                        cc / max(tc, 1), 4
                    ),
                }

            results[f"level_{level_idx}"] = {
                "bounds_mm": list(bounds),
                "total_episodes": total,
                "success_count": successes,
                "timeout_count": timeouts,
                "collision_count": collisions,
                "success_rate": successes / count,
                "timeout_rate": timeouts / count,
                "collision_rate": collisions / count,
                "action_distribution": {
                    type_names.get(
                        k, f"type_{k}"
                    ): {
                        "count": v,
                        "rate": round(
                            v / total_act, 4
                        ),
                    }
                    for k, v in sorted(
                        action_counts.items()
                    )
                },
                "collision_stats": {
                    type_names.get(
                        k, f"type_{k}"
                    ): v
                    for k, v in (
                        collision_counts.items()
                    )
                },
                "collision_rate_per_type": (
                    collision_rate_per_type
                ),
                "steps_per_success": round(
                    total_steps / max(successes, 1),
                    1,
                ),
                "mean_episode_steps": round(
                    total_steps / max(total, 1), 1
                ),
            }

        return results

    # ══════════════════════════════════════════════════════
    # Adaptive
    # ══════════════════════════════════════════════════════
    def _run_adaptive(self) -> None:
        """Run adaptive mode with Q-store + SAC arbitration.
        
        Synergy: Q-store chooses action type (familiar states),
        SAC provides continuous params. When Q is not confident,
        SAC decides both type and params.
        """
        logger.info("=" * 60)
        logger.info("Adaptive: %s", self.adaptive_mesh)
        logger.info("=" * 60)

        mesh_path = str(
            self.data_dir / f"{self.adaptive_mesh}.stl"
        )
        env = LightweightEnv(mesh_path)
        num_types = len(
            ExperienceExtractor.get_type_names()
        )
        adapt_seed = self.train_seeds[0]

        # Load Q-store: adaptive if exists, otherwise training
        adaptive_q_dir = str(
            self.runs_dir
            / f"adaptive_q_seed_{adapt_seed}"
        )
        if (
            Path(adaptive_q_dir) / "config.json"
        ).exists():
            q_load_dir = adaptive_q_dir
            logger.info(
                "Loading Q-store from adaptive: %s",
                q_load_dir,
            )
        else:
            q_load_dir = self._q_model_dir(adapt_seed)
            logger.info(
                "Loading Q-store from training: %s",
                q_load_dir,
            )

        adaptive_cfg = {
            **self.rl_config,
            "mode": "adaptive",
        }
        controller = RLGoalApproachController.load(
            q_load_dir,
            agent_id=f"{self.adaptive_mesh}_adaptive",
            config=adaptive_cfg,
        )
        # Unfreeze Q-store normalization for new object
        controller.q_store_free._norm_frozen = False
        controller.q_store_free._freeze_done = False
        controller.q_store_free._state_buffer.clear()
        controller.q_store_surface._norm_frozen = False
        controller.q_store_surface._freeze_done = False
        controller.q_store_surface._state_buffer.clear()
        controller.strategic_detach._norm_frozen = False
        controller.strategic_detach._freeze_done = False
        controller.strategic_detach._state_buffer.clear()
        controller.strategic_direction._norm_frozen = False
        controller.strategic_direction._freeze_done = False
        controller.strategic_direction._state_buffer.clear()
        logger.info("Q-store normalization unfrozen for adaptive mode")

        controller._collision_stats = {}

        # Load SAC: adaptive if exists, otherwise training
        adaptive_sac_dir = str(
            self.runs_dir
            / f"adaptive_sac_seed_{self.sac_seed}"
        )
        sac_trainer = PSACTrainer(
            state_dim=self.rl_config.get("state_dim", 20),
            num_types=num_types,
        )
        if (
            Path(adaptive_sac_dir) / "sac_actor.pt"
        ).exists():
            sac_trainer.load(adaptive_sac_dir)
            logger.info(
                "Loading SAC from adaptive: %s",
                adaptive_sac_dir,
            )
        else:
            sac_trainer.load(
                self._sac_model_dir(self.sac_seed)
            )
            logger.info(
                "Loading SAC from training: %s",
                self._sac_model_dir(self.sac_seed),
            )

        # Strategic SAC
        controller.strategic_sac = None

        adapt_q_dir = str(
            self.runs_dir
            / f"adaptive_q_seed_{adapt_seed}"
        )
        adapt_sac_dir = str(
            self.runs_dir
            / f"adaptive_sac_seed_{self.sac_seed}"
        )
        manager = AdaptiveTrainingManager(
            controller=controller,
            env=env,
            config=adaptive_cfg,
            runs_dir=str(self.runs_dir),
            mesh_path=mesh_path,
            q_save_dir=adapt_q_dir,
            sac_save_dir=adapt_sac_dir,
            offline_check_window=self.config.get("offline_check_window", 50),
        )
        manager.sac_trainer = sac_trainer

        if sac_trainer.strategic_detach_sac is not None:
            manager.arbitrator._sac_strategic_detach = (
                sac_trainer.strategic_detach_sac
            )
        if (
            sac_trainer.strategic_direction_sac
            is not None
        ):
            manager.arbitrator._sac_strategic_direction = (
                sac_trainer.strategic_direction_sac
            )

        # Metrics tracking
        episode_log: list[dict[str, Any]] = []
        snapshot_log: list[dict[str, Any]] = []
        action_counts: dict[str, int] = {}
        collision_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {
            "q_store": 0,
            "sac": 0,
            "blend": 0,
            "heuristic": 0,
        }
        total_steps_adaptive = 0
        rolling_successes: list[bool] = []

        adaptive_log_dir = (
            self.data_dir
            / f"adaptive_logs_{self.adaptive_mesh}"
        )
        adaptive_log_dir.mkdir(parents=True, exist_ok=True)

        q_terminations: dict[str, int] = {
            "success": 0, "collision": 0, "timeout": 0,
        }
        sac_terminations: dict[str, int] = {
            "success": 0, "collision": 0, "timeout": 0,
        }
        q_final_distances: list[float] = []
        sac_final_distances: list[float] = []
        q_episode_steps: list[int] = []
        sac_episode_steps: list[int] = []
        adaptive_max_steps = self.rl_config.get(
            "max_steps_per_goal", 400
        )

        collision_stats_before = dict(
            controller._collision_stats
        )

        # Curriculum for adaptive
        adaptive_curriculum = list(self.curriculum_levels)
        adaptive_level = 0
        adaptive_promote_window: list[bool] = []
        adaptive_promote_threshold = self.promote_threshold
        adaptive_promote_window_size = self.promote_window

        ep_successes = []

        # Create interpreter for continuous action execution
        interpreter = ActionInterpreter(env)
        type_names = ExperienceExtractor.get_type_names()

        for episode in range(self.adaptive_episodes):
            env.reset()
            start_pos = env.get_pose()[:3]

            # Curriculum goal generation with filters
            min_dist, max_dist = adaptive_curriculum[
                adaptive_level
            ]

            level_filter = (
                self.curriculum_filters[adaptive_level]
                if adaptive_level
                < len(self.curriculum_filters)
                else {}
            )
            require_same_side = level_filter.get(
                "same_side", None
            )
            require_path_blocked = level_filter.get(
                "path_blocked", None
            )

            max_goal_attempts = (
                50
                if (
                    require_same_side is not None
                    or require_path_blocked is not None
                )
                else 1
            )

            goal_pose = None
            for _attempt in range(max_goal_attempts):
                candidate = (
                    env.get_random_surface_point(
                        reference_pos=start_pos,
                        min_dist=min_dist,
                        max_dist=max_dist,
                        max_attempts=2000,
                        mesh_sample=True,
                    )
                )

                if require_same_side is not None:
                    same_side = _is_reachable_by_surface(
                        env, start_pos, candidate[:3],
                    )
                    if same_side != require_same_side:
                        continue

                if require_path_blocked is not None:
                    env._current_goal = np.concatenate([
                        candidate[:3], candidate[3:],
                    ])
                    sensor = env.get_sensor_data()
                    pb = sensor.get("path_blocked", False)
                    if pb != require_path_blocked:
                        continue

                goal_pose = candidate
                break

            if goal_pose is None:
                goal_pose = candidate

            controller.set_new_goal(goal_pose, start_pos)
            env.set_goal(goal_pose)

            # Tell arbitrator about new episode and level
            manager.arbitrator.start_episode(level=adaptive_level)

            goals_before = controller._total_goals_reached
            ep_steps = 0
            ep_sources: list[str] = []
            ep_actions: list[int] = []
            current_poses = []
            action_explanations = []

            # Save transitions before episode
            collision_stats_before = dict(
                controller._collision_stats
            )

            for step in range(adaptive_max_steps):
                pose = env.get_pose()
                current_poses.append(pose.copy())
                sensor = env.get_sensor_data()
                state = controller._compute_state(pose, sensor)

                # Arbitrator returns (type, params, source)
                action_type, action_params, source = (
                    manager.get_action(state, pose, sensor)
                )
                controller._current_source = source

                # Convert to discrete for Q-store learning
                discrete_idx = sac_to_discrete(
                    action_type, action_params
                )

                # Save transitions before update_only clears them
                last_transitions = (
                    controller._episode_transitions.copy()
                )

                # Execute continuous action in environment
                sensor_after = interpreter.execute(
                    action_type, action_params
                )

                # Q-store learns from discrete action
                pose_after = env.get_pose()
                _st, done = controller.update_only(
                    pose_after, sensor_after, discrete_idx
                )

                ep_steps += 1
                total_steps_adaptive += 1
                ep_sources.append(source)
                ep_actions.append(discrete_idx)

                # Track action type
                act_name = type_names.get(
                    action_type, f"type_{action_type}"
                )
                action_counts[act_name] = (
                    action_counts.get(act_name, 0) + 1
                )
                action_explanations.append(
                    f"source: {source}, type: {act_name}, "
                    f"params: [{action_params[0]:.2f}, "
                    f"{action_params[1]:.2f}, "
                    f"{action_params[2]:.2f}]"
                )

                # Track source
                if source.startswith("blend"):
                    source_key = "blend"
                elif source.startswith("q_"):
                    source_key = "q_store"
                elif source.startswith("sac"):
                    source_key = "sac"
                elif source.startswith("heuristic"):
                    source_key = "heuristic"
                else:
                    source_key = "other"

                source_counts[source_key] = (
                    source_counts.get(source_key, 0) + 1
                )

                if done:
                    break

            success = (
                controller._total_goals_reached
                > goals_before
            )

            ep_successes.append(success)
            total_episodes_rate = (
                sum(ep_successes) / len(ep_successes)
            )

            # Determine termination
            if success:
                termination = "success"
            elif step == (adaptive_max_steps - 1):
                termination = "timeout"
            else:
                termination = "collision"

            if termination == "collision":
                for act_name, count in (
                    controller._collision_stats.items()
                ):
                    prev = collision_stats_before.get(
                        act_name, 0
                    )
                    if count > prev:
                        collision_counts[act_name] = (
                            collision_counts.get(act_name, 0)
                            + (count - prev)
                        )

            collision_stats_before = dict(
                controller._collision_stats
            )

            transitions = last_transitions

            manager.on_episode_complete(
                success=success,
                transitions=transitions,
            )

            # Write mode changes to JSON
            while manager.mode_changes:
                change = manager.mode_changes.pop(0)
                change["recent_episodes"] = episode_log[-10:]
                change_idx = len([
                    f for f in adaptive_log_dir.iterdir()
                    if f.name.startswith("mode_change_")
                ])
                change_path = (
                    adaptive_log_dir
                    / f"mode_change_{change_idx:04d}_ep_{change['episode']:05d}.json"
                )
                with change_path.open("w") as f:
                    json.dump(change, f, indent=2)
                logger.info(
                    "Mode change saved: %s → %s at ep %d to %s",
                    change["from_mode"],
                    change["to_mode"],
                    change["episode"],
                    change_path,
                )

            manager.arbitrator.on_episode_end(success)

            # Curriculum promote
            adaptive_promote_window.append(success)
            if (
                len(adaptive_promote_window)
                > adaptive_promote_window_size
            ):
                adaptive_promote_window.pop(0)
            if (
                len(adaptive_promote_window)
                == adaptive_promote_window_size
                and adaptive_level
                < len(adaptive_curriculum) - 1
            ):
                promote_rate = (
                    sum(adaptive_promote_window)
                    / adaptive_promote_window_size
                )
                if promote_rate >= adaptive_promote_threshold:
                    adaptive_level += 1
                    adaptive_promote_window = []
                    logger.info(
                        "Adaptive curriculum: promoted to "
                        "level %d (%s mm) at ep %d "
                        "(rate=%.3f)",
                        adaptive_level,
                        adaptive_curriculum[adaptive_level],
                        episode + 1,
                        promote_rate,
                    )

            rolling_successes.append(success)
            if len(rolling_successes) > 100:
                rolling_successes.pop(0)
            rolling_rate = (
                sum(rolling_successes)
                / len(rolling_successes)
            )

            # Dominant source for this episode
            from collections import Counter

            source_counter = Counter(ep_sources)
            dominant_source = (
                source_counter.most_common(1)[0][0]
                if ep_sources
                else "none"
            )
            # Normalize dominant source for tracking
            if dominant_source in (
                "blend", "q_type_sac_params",
            ):
                dominant_source_key = "q_store"
            elif dominant_source.startswith("q_"):
                dominant_source_key = "q_store"
            elif dominant_source == "sac":
                dominant_source_key = "sac"
            else:
                dominant_source_key = "heuristic"

            # Per-episode log
            start_dist = float(
                np.linalg.norm(goal_pose[:3] - start_pos)
            )
            final_pose = env.get_pose()
            final_dist = float(
                np.linalg.norm(
                    goal_pose[:3] - final_pose[:3]
                )
            )

            episode_log.append({
                "episode": episode,
                "success": success,
                "termination": termination,
                "steps": ep_steps,
                "start_distance": round(start_dist, 1),
                "final_distance": round(final_dist, 1),
                "dominant_source": dominant_source,
                "rolling_success_rate": round(
                    rolling_rate, 3
                ),
                "mode": manager.mode,
                "curriculum_level": adaptive_level,
            })

            # Per-source termination and distance tracking
            if dominant_source_key == "q_store":
                q_terminations[termination] = (
                    q_terminations.get(termination, 0) + 1
                )
                q_final_distances.append(
                    round(final_dist, 1)
                )
                q_episode_steps.append(ep_steps)
            elif dominant_source_key == "sac":
                sac_terminations[termination] = (
                    sac_terminations.get(termination, 0) + 1
                )
                sac_final_distances.append(
                    round(final_dist, 1)
                )
                sac_episode_steps.append(ep_steps)

            # Snapshot every N episodes
            if (episode + 1) % _ADAPTIVE_LOG_INTERVAL == 0:
                if self.visualise:
                    vis_dir = (
                        Path(adaptive_log_dir)
                        / "visualizations"
                    )
                    _maybe_save_visualization(
                        controller=controller,
                        env=env,
                        episode=episode,
                        ep_result=termination,
                        goal_pose=goal_pose,
                        current_poses=current_poses,
                        action_explanations=action_explanations,
                        vis_dir=vis_dir,
                        vis_filter=None,
                        vis_counts=None,
                    )

                stats = manager.get_stats()
                arb_stats = manager.arbitrator.get_stats()

                total_src = max(
                    sum(source_counts.values()), 1
                )
                total_act = max(
                    sum(action_counts.values()), 1
                )

                snapshot = {
                    "episode": episode + 1,
                    "rolling_success_rate": round(
                        rolling_rate, 3
                    ),
                    "total_episodes_success_rate": round(
                        total_episodes_rate, 3
                    ),
                    "curriculum_level": adaptive_level,
                    "mode": stats["mode"],
                    "total_episodes": episode + 1,
                    "total_steps": total_steps_adaptive,
                    "success_count": sum(
                        1
                        for e in episode_log
                        if e["success"]
                    ),
                    "collision_count": sum(
                        1
                        for e in episode_log
                        if e["termination"] == "collision"
                    ),
                    "timeout_count": sum(
                        1
                        for e in episode_log
                        if e["termination"] == "timeout"
                    ),
                    "source_distribution": {
                        k: round(v / total_src, 3)
                        for k, v in source_counts.items()
                    },
                    "action_distribution": {
                        k: {
                            "count": v,
                            "rate": round(
                                v / total_act, 4
                            ),
                        }
                        for k, v in sorted(
                            action_counts.items(),
                            key=lambda x: -x[1],
                        )
                    },
                    "collision_stats": dict(
                        collision_counts
                    ),
                    "steps_per_success": round(
                        total_steps_adaptive
                        / max(
                            sum(
                                1
                                for e in episode_log
                                if e["success"]
                            ),
                            1,
                        ),
                        1,
                    ),
                    "arbitrator": {
                        "q_store_rate": arb_stats.get(
                            "q_store_rate", 0
                        ),
                        "sac_rate": arb_stats.get(
                            "sac_rate", 0
                        ),
                        "blend_rate": arb_stats.get(
                            "blend_rate", 0
                        ),
                        "heuristic_rate": arb_stats.get(
                            "heuristic_rate", 0
                        ),
                        "agreement_rate": arb_stats.get(
                            "agreement_rate", 0
                        ),
                        "q_spread_mean": arb_stats.get(
                            "q_spread_mean", 0
                        ),
                        "q_confidence_mean": arb_stats.get(
                            "q_confidence_mean", 0
                        ),
                        "q_success_rate": arb_stats.get(
                            "q_success_rate", 0
                        ),
                        "sac_success_rate": arb_stats.get(
                            "sac_success_rate", 0
                        ),
                    },
                    "manager": {
                        "sac_updates": stats.get(
                            "total_sac_updates", 0
                        ),
                        "offline_iterations": stats.get(
                            "total_offline_iterations",
                            0,
                        ),
                    },
                    "per_source_analysis": {
                        "q_store": {
                            "terminations": dict(
                                q_terminations
                            ),
                            "chosen_actions": (
                                arb_stats.get(
                                    "q_chosen_top", {}
                                )
                            ),
                            "proposed_actions": (
                                arb_stats.get(
                                    "q_proposed_top", {}
                                )
                            ),
                            "mean_final_distance": round(
                                float(
                                    np.mean(
                                        q_final_distances
                                    )
                                )
                                if q_final_distances
                                else 0,
                                1,
                            ),
                            "near_miss_count": sum(
                                1
                                for d in q_final_distances
                                if 2.0 < d <= 5.0
                            ),
                            "mean_episode_steps": round(
                                float(
                                    np.mean(
                                        q_episode_steps
                                    )
                                )
                                if q_episode_steps
                                else 0,
                                1,
                            ),
                        },
                        "sac": {
                            "terminations": dict(
                                sac_terminations
                            ),
                            "chosen_actions": (
                                arb_stats.get(
                                    "sac_chosen_top", {}
                                )
                            ),
                            "proposed_actions": (
                                arb_stats.get(
                                    "sac_proposed_top", {}
                                )
                            ),
                            "mean_final_distance": round(
                                float(
                                    np.mean(
                                        sac_final_distances
                                    )
                                )
                                if sac_final_distances
                                else 0,
                                1,
                            ),
                            "near_miss_count": sum(
                                1
                                for d in sac_final_distances
                                if 2.0 < d <= 5.0
                            ),
                            "mean_episode_steps": round(
                                float(
                                    np.mean(
                                        sac_episode_steps
                                    )
                                )
                                if sac_episode_steps
                                else 0,
                                1,
                            ),
                        },
                    },
                }
                snapshot_log.append(snapshot)

                snap_path = (
                    adaptive_log_dir
                    / f"snapshot_ep_{episode + 1:05d}.json"
                )
                with snap_path.open("w") as f:
                    json.dump(snapshot, f, indent=2)

                logger.info(
                    "Adaptive ep %d: rate=%.3f, "
                    "mode=%s, saved to %s",
                    episode + 1,
                    rolling_rate,
                    stats["mode"],
                    snap_path,
                )

        # Save adaptive models
        controller.save(adapt_q_dir)
        logger.info(
            "Adaptive Q-store saved to %s", adapt_q_dir
        )

        if manager.sac_trainer:
            manager.sac_trainer.save(adapt_sac_dir)
            logger.info(
                "Adaptive SAC saved to %s", adapt_sac_dir
            )

        # Save comprehensive results
        total_src = max(sum(source_counts.values()), 1)
        total_act = max(sum(action_counts.values()), 1)
        successes = sum(
            1 for e in episode_log if e["success"]
        )
        collisions = sum(
            1
            for e in episode_log
            if e["termination"] == "collision"
        )
        timeouts = sum(
            1
            for e in episode_log
            if e["termination"] == "timeout"
        )
        total_ep = len(episode_log)

        final_results = {
            "mesh": self.adaptive_mesh,
            "total_episodes": total_ep,
            "total_steps": total_steps_adaptive,
            "success_rate": round(
                successes / max(total_ep, 1), 4
            ),
            "collision_rate": round(
                collisions / max(total_ep, 1), 4
            ),
            "timeout_rate": round(
                timeouts / max(total_ep, 1), 4
            ),
            "steps_per_success": round(
                total_steps_adaptive
                / max(successes, 1),
                1,
            ),
            "source_distribution": {
                k: round(v / total_src, 4)
                for k, v in source_counts.items()
            },
            "action_distribution": {
                k: {
                    "count": v,
                    "rate": round(v / total_act, 4),
                }
                for k, v in sorted(
                    action_counts.items(),
                    key=lambda x: -x[1],
                )
            },
            "collision_stats": dict(collision_counts),
            "snapshots": snapshot_log,
            "episode_log": episode_log,
            "per_source_analysis": {
                "q_store": {
                    "terminations": dict(q_terminations),
                    "mean_final_distance": round(
                        float(np.mean(q_final_distances))
                        if q_final_distances
                        else 0,
                        1,
                    ),
                    "near_miss_count": sum(
                        1
                        for d in q_final_distances
                        if 2.0 < d <= 5.0
                    ),
                    "near_miss_rate": round(
                        sum(
                            1
                            for d in q_final_distances
                            if 2.0 < d <= 5.0
                        )
                        / max(len(q_final_distances), 1),
                        3,
                    ),
                    "mean_episode_steps": round(
                        float(np.mean(q_episode_steps))
                        if q_episode_steps
                        else 0,
                        1,
                    ),
                    "total_episodes": len(
                        q_final_distances
                    ),
                },
                "sac": {
                    "terminations": dict(sac_terminations),
                    "mean_final_distance": round(
                        float(np.mean(sac_final_distances))
                        if sac_final_distances
                        else 0,
                        1,
                    ),
                    "near_miss_count": sum(
                        1
                        for d in sac_final_distances
                        if 2.0 < d <= 5.0
                    ),
                    "near_miss_rate": round(
                        sum(
                            1
                            for d in sac_final_distances
                            if 2.0 < d <= 5.0
                        )
                        / max(
                            len(sac_final_distances), 1
                        ),
                        3,
                    ),
                    "mean_episode_steps": round(
                        float(
                            np.mean(sac_episode_steps)
                        )
                        if sac_episode_steps
                        else 0,
                        1,
                    ),
                    "total_episodes": len(
                        sac_final_distances
                    ),
                },
            },
        }

        results_path = (
            self.data_dir
            / f"adaptive_result_{self.adaptive_mesh}.json"
        )
        with results_path.open("w") as f:
            json.dump(final_results, f, indent=2)
        logger.info(
            "Adaptive results saved to %s", results_path
        )

        self._save_meta(
            f"adaptive_{self.adaptive_mesh}",
            adapt_seed,
            {
                "success_rate": final_results[
                    "success_rate"
                ],
                "total_episodes": total_ep,
                "source_distribution": final_results[
                    "source_distribution"
                ],
            },
            [self.adaptive_mesh],
        )

        logger.info(
            "Adaptive complete: success=%.3f, "
            "collision=%.3f, timeout=%.3f",
            final_results["success_rate"],
            final_results["collision_rate"],
            final_results["timeout_rate"],
        )
    
    # ═════════════════════════════════════════════════════
    # Utilities
    # ═════════════════════════════════════════════════════

    @staticmethod
    def _filter_bc_transitions(
        transitions: list,
        total_target: int = 100000,
        min_samples_to_keep: int = 50,
    ) -> list:
        """Balance BC transitions by (mesh, level).

        Distributes total_target equally across groups.
        Groups with fewer samples are oversampled,
        groups with more are undersampled.

        Args:
            transitions: All BC transitions.
            total_target: Total desired transitions.
            min_samples_to_keep: Minimum for inclusion.

        Returns:
            Balanced transitions.
        """
        type_names = (
            ExperienceExtractor.get_type_names()
        )
        mesh_id_to_name = {
            v: k
            for k, v in (
                ExperienceExtractor
                .MESH_NAME_TO_ID.items()
            )
        }

        # Group by (mesh_id, level)
        groups: dict[tuple[int, int], list] = {}
        for tr in transitions:
            key = (
                getattr(tr, "mesh_id", -1),
                getattr(tr, "level", 0),
            )
            if key not in groups:
                groups[key] = []
            groups[key].append(tr)

        # Calculate target per group
        num_groups = max(len(groups), 1)
        target_per_group = total_target // num_groups

        # Log before
        logger.info("BC balance BEFORE:")
        for (mid, level), trs in sorted(
            groups.items()
        ):
            mname = mesh_id_to_name.get(
                mid, f"mesh_{mid}"
            )
            type_counts: dict[str, int] = {}
            for tr in trs:
                tname = type_names.get(
                    tr.action_type,
                    f"type_{tr.action_type}",
                )
                type_counts[tname] = (
                    type_counts.get(tname, 0) + 1
                )
            logger.info(
                "  %s L%d: %d transitions, "
                "actions: %s",
                mname,
                level,
                len(trs),
                type_counts,
            )

        # Balance
        balanced: list = []
        for (mid, level), trs in groups.items():
            mname = mesh_id_to_name.get(
                mid, f"mesh_{mid}"
            )

            if len(trs) < min_samples_to_keep:
                balanced.extend(trs)
                logger.info(
                    "  %s L%d: kept all %d "
                    "(too few)",
                    mname,
                    level,
                    len(trs),
                )
            elif len(trs) >= target_per_group:
                indices = (
                    np.random.permutation(
                        len(trs)
                    )[:target_per_group]
                )
                balanced.extend(
                    [trs[i] for i in indices]
                )
                logger.info(
                    "  %s L%d: undersampled "
                    "%d → %d",
                    mname,
                    level,
                    len(trs),
                    target_per_group,
                )
            else:
                indices = np.random.choice(
                    len(trs),
                    target_per_group,
                    replace=True,
                )
                balanced.extend(
                    [trs[i] for i in indices]
                )
                logger.info(
                    "  %s L%d: oversampled "
                    "%d → %d",
                    mname,
                    level,
                    len(trs),
                    target_per_group,
                )

        np.random.shuffle(balanced)

        # Log after
        after_groups: dict[str, int] = {}
        for tr in balanced:
            mname = mesh_id_to_name.get(
                getattr(tr, "mesh_id", -1),
                "unknown",
            )
            key = (
                f"{mname}_L"
                f"{getattr(tr, 'level', 0)}"
            )
            after_groups[key] = (
                after_groups.get(key, 0) + 1
            )

        logger.info("BC balance AFTER:")
        for key in sorted(after_groups.keys()):
            logger.info(
                "  %s: %d transitions (%.1f%%)",
                key,
                after_groups[key],
                100.0 * after_groups[key]
                / max(len(balanced), 1),
            )

        logger.info(
            "BC filter: %d → %d transitions",
            len(transitions),
            len(balanced),
        )
        return balanced