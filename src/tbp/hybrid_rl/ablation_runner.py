# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Q-learning training orchestrator with curriculum support.

Provides the main ``run_episodes()`` function that runs Q-learning episodes
with heuristic-guided exploration, curriculum levels, and optional
visualization of episode trajectories.
"""

from __future__ import annotations

import collections
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

from tbp.hybrid_rl.config import DEFAULT_CONFIG
from tbp.hybrid_rl.experience_extractor import ExperienceExtractor
from tbp.hybrid_rl.lightweight_env import LightweightEnv
from tbp.hybrid_rl.rl_goal_approach_controller import RLGoalApproachController
from tbp.hybrid_rl.visualize_env import save_episode_frames, visualize_agent_goal

_LOG_INTERVAL = 100

logger = logging.getLogger(__name__)

CURRICULUM_LEVELS = [
    (10.0, 40.0),
    (20.0, 80.0),
    (40.0, 120.0),
]


def run_eval_per_seed(  # noqa: PLR0913
    data_dir: Path,
    runs_dir: Path,
    mesh_path: str,
    train_seeds: list[int],
    eval_seeds: list[int],
    variant: str,
    eval_cfg: dict[str, Any],
    eval_pools: dict[int, dict[str, Any]],
    collect_bc: bool = False,
    episodes_per_level: int | None = None,
    mesh_name: str = "",
    visualise=False
) -> tuple[dict[str, Any], list[Any]]:
    """Run evaluation across seeds and curriculum levels.

    Args:
        data_dir: Base data directory.
        runs_dir: Directory containing trained models.
        mesh_path: Path to the mesh file.
        train_seeds: List of training seeds.
        eval_seeds: List of evaluation seeds.
        variant: Model variant name (used for directory lookup).
        eval_cfg: Evaluation configuration dict.
        eval_pools: Episode pools keyed by seed.
        collect_bc: Whether to collect BC transitions.
        episodes_per_level: Max episodes per level.
        mesh_name: Mesh name for BC data tagging.

    Returns:
        Tuple of (results_per_level dict, bc_transitions list).
    """
    results_per_seed: dict[str, Any] = {}
    bc_transitions: list[Any] = []

    sample_seed = eval_seeds[0]
    num_levels = len(eval_pools[sample_seed].get("levels", []))

    for train_seed, eval_seed in zip(train_seeds, eval_seeds):
        seed_results: dict[str, Any] = {}
        load_dir = str(
            runs_dir / f"{variant.lower()}_seed_{train_seed}"
        )

        for level_idx in range(num_levels):
            level_pool = eval_pools[eval_seed]["levels"][level_idx]

            metrics = run_episodes(
                mesh_dir=str(data_dir),
                save_dir=load_dir,
                num_episodes=episodes_per_level,
                config=eval_cfg,
                mesh_path=mesh_path,
                load_dir=load_dir,
                seed=eval_seed,
                return_metrics=True,
                agent_id=(
                    f"eval_L{level_idx}_{variant}"
                    f"_t{train_seed}_e{eval_seed}"
                ),
                episode_script=level_pool,
                episodes_per_level=episodes_per_level,
                visualise=visualise,
                level_idx=level_idx
            )

            if collect_bc:
                trails = metrics.get("success_trails", [])
                if trails:
                    extractor = ExperienceExtractor(
                        config=eval_cfg, mesh_name=mesh_name
                    )
                    for trail in trails:
                        transitions = extractor.convert_trajectory(trail)
                        for tr in transitions:
                            tr.level = level_idx
                        bc_transitions.extend(transitions)

            rates = metrics.get("stats", {}).get(
                "termination_rates", {}
            )
            stats = metrics.get("stats", {})
            seed_results[f"level_{level_idx}"] = {
                "success_rate": float(
                    metrics.get("success_rate", 0.0)
                ),
                "timeout_rate": float(rates.get("timeout", 0.0)),
                "collision_rate": float(
                    rates.get("collision_surface_violation", 0.0)
                ),
                "collision_stats": stats.get("collision_stats", {}),
                "global_action_counts": stats.get(
                    "global_action_counts", {}
                ),
                "collision_rate_per_action": stats.get(
                    "collision_rate_per_action", {}
                ),
                "steps_per_success": stats.get("steps_per_success", 0),
                "surface_air_ratio": stats.get(
                    "surface_air_ratio", {}
                ),
                # ═══ NEW ═══
                "phase_metrics": metrics.get(
                    "phase_metrics", {}
                ),
                "strategic": {
                    "strategic_detach_diagnostic": stats.get(
                        "strategic_detach_diagnostic", {}
                    ),
                    "strategic_direction_diagnostic": stats.get(
                        "strategic_direction_diagnostic", {}
                    ),
                    "strategic_decisions": stats.get(
                        "strategic", {}
                    ),
                },
                 "action_source": stats.get(
                    "action_source_summary", {}
                ),
            }
        seed_key = f"train_{train_seed}_eval_{eval_seed}"
        results_per_seed[seed_key] = seed_results

        seed_output = (
            data_dir
            / f"eval_result_{mesh_name}_seed_{train_seed}_{eval_seed}.json"
        )
        with seed_output.open("w", encoding="utf-8") as f:
            json.dump(seed_results, f, indent=2)

    results_per_level = _aggregate_level_results(
        results_per_seed, num_levels
    )
    return results_per_level, bc_transitions


def _aggregate_level_results(
    results_per_seed: dict[str, Any],
    num_levels: int,
) -> dict[str, Any]:
    """Aggregate per-seed results into per-level summary.

    Args:
        results_per_seed: Results keyed by seed pair string.
        num_levels: Number of curriculum levels.

    Returns:
        Dict with per-level and overall aggregated metrics.
    """
    results_per_level: dict[str, Any] = {}

    for level_idx in range(num_levels):
        level_results = []
        for seed_data in results_per_seed.values():
            level_key = f"level_{level_idx}"
            if level_key in seed_data:
                level_results.append(seed_data[level_key])

        count = max(len(level_results), 1)
        bounds = (
            CURRICULUM_LEVELS[level_idx]
            if level_idx < len(CURRICULUM_LEVELS)
            else (0, 0)
        )
        results_per_level[f"level_{level_idx}"] = {
            "bounds_mm": list(bounds),
            "per_seed": level_results,
            "mean_success_rate": (
                sum(r["success_rate"] for r in level_results) / count
            ),
            "mean_timeout_rate": (
                sum(r["timeout_rate"] for r in level_results) / count
            ),
            "mean_collision_rate": (
                sum(r["collision_rate"] for r in level_results) / count
            ),
        }

    all_success = []
    all_timeout = []
    all_collision = []
    for key, level_data in results_per_level.items():
        if key != "overall":
            all_success.append(level_data["mean_success_rate"])
            all_timeout.append(level_data["mean_timeout_rate"])
            all_collision.append(level_data["mean_collision_rate"])

    n = max(len(all_success), 1)
    results_per_level["overall"] = {
        "mean_success_rate": sum(all_success) / n,
        "mean_timeout_rate": sum(all_timeout) / n,
        "mean_collision_rate": sum(all_collision) / n,
    }

    return results_per_level


class _CurriculumTracker:
    """Tracks curriculum level progression during training.

    Monitors rolling success rate and promotes to the next level
    when the rate exceeds the threshold.
    """

    def __init__(
        self,
        levels: list[tuple[float, float]],
        promote_threshold: float = 0.20,
        promote_window: int = 50,
    ) -> None:
        self.levels = list(levels)
        self.promote_threshold = promote_threshold
        self.promote_window = promote_window

        self.level_idx = 0
        self.window: collections.deque[bool] = collections.deque(
            maxlen=promote_window
        )
        self.level_episodes = 0
        self.level_successes = 0
        self.stats: dict[str, Any] = {
            "levels_reached": 1,
            "episodes_per_level": [],
            "successes_per_level": [],
            "success_rate_per_level": [],
            "fallback_episodes": 0,
        }

    @property
    def current_bounds(self) -> tuple[float, float]:
        """Return (min_dist, max_dist) for current level.

        Returns:
            Tuple of minimum and maximum distance bounds.
        """
        return tuple(self.levels[self.level_idx])

    def on_episode_end(
        self,
        success: bool,
        controller: RLGoalApproachController,
        episode: int,
    ) -> None:
        """Update curriculum state after an episode.

        Args:
            success: Whether the episode was successful.
            controller: Controller to update goal_threshold on promotion.
            episode: Current episode number (for logging).
        """
        self.window.append(success)
        self.level_episodes += 1
        if success:
            self.level_successes += 1

        window_full = len(self.window) == self.promote_window
        not_last_level = self.level_idx < len(self.levels) - 1

        if window_full and not_last_level:
            rolling_rate = sum(self.window) / self.promote_window
            if rolling_rate >= self.promote_threshold:
                self._promote(controller, episode, rolling_rate)

    def _promote(
        self,
        controller: RLGoalApproachController,
        episode: int,
        rolling_rate: float,
    ) -> None:
        self.stats["episodes_per_level"].append(self.level_episodes)
        self.stats["successes_per_level"].append(self.level_successes)
        self.stats["success_rate_per_level"].append(
            self.level_successes / max(self.level_episodes, 1)
        )

        self.level_idx += 1
        # goal_threshold stays fixed from config — no per-level override

        self.window = collections.deque(maxlen=self.promote_window)
        self.level_episodes = 0
        self.level_successes = 0
        self.stats["levels_reached"] = self.level_idx + 1

        new_min, new_max = self.levels[self.level_idx]
        logger.info(
            "  [Curriculum] ep=%d: promoted to level %d: "
            "dist [%s, %s] mm (rolling_rate=%.3f)(epsilon=%.3f)",
            episode + 1,
            self.level_idx,
            new_min,
            new_max,
            rolling_rate,
            controller.epsilon,
        )
        
    def finalize(self) -> dict[str, Any]:
        """Finalize and return curriculum statistics.

        Returns:
            Dictionary with curriculum training statistics.
        """
        self.stats["episodes_per_level"].append(self.level_episodes)
        self.stats["successes_per_level"].append(self.level_successes)
        self.stats["success_rate_per_level"].append(
            self.level_successes / max(self.level_episodes, 1)
        )
        self.stats["final_level"] = self.level_idx
        return self.stats


def _init_controller(
    cfg: dict[str, Any],
    load_dir: str | None,
    agent_id: str,
    config_overrides: dict[str, Any] | None,
) -> RLGoalApproachController:
    """Initialize or load the RL controller.

    Args:
        cfg: Merged configuration dictionary.
        load_dir: Directory to load from, or None for fresh start.
        agent_id: Agent identifier.
        config_overrides: Original config overrides for load.

    Returns:
        Initialized RLGoalApproachController.
    """
    if load_dir is None:
        return RLGoalApproachController(agent_id=agent_id, config=cfg)

    controller = RLGoalApproachController.load(
        load_dir, agent_id=agent_id, config=config_overrides
    )
    if cfg.get("unfreeze_normalization", False):
        controller.q_store_free._norm_frozen = False
        controller.q_store_free._freeze_done = False
        controller.q_store_surface._norm_frozen = False
        controller.q_store_surface._freeze_done = False
        logger.info("Normalization unfrozen for transfer learning")
    return controller


def _setup_episode_from_data(
    env: LightweightEnv,
    ep_data: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Set up an episode from a fixed data entry.

    Args:
        env: Environment instance.
        ep_data: Episode data with start_pos, start_rot, goal_pos, goal_rot.

    Returns:
        Tuple of (start_pos, goal_pose).
    """
    start_pos = np.array(ep_data["start_pos"])
    start_rot = np.array(ep_data["start_rot"])
    env.reset(position=start_pos, rotation=start_rot)
    goal_pose = np.concatenate([
        np.array(ep_data["goal_pos"]),
        np.array(ep_data["goal_rot"]),
    ])
    return start_pos, goal_pose


def _find_mesh_files(mesh_dir: str) -> list[str]:
    """Find all mesh files in a directory.

    Args:
        mesh_dir: Directory to search for mesh files.

    Returns:
        List of mesh file paths as strings.

    Raises:
        FileNotFoundError: If no mesh files found.
    """
    mesh_dir_path = Path(mesh_dir)
    mesh_files = [
        str(p)
        for ext in ("*.obj", "*.stl", "*.ply")
        for p in mesh_dir_path.glob(ext)
    ]
    if not mesh_files:
        msg = f"No mesh files in {mesh_dir}"
        raise FileNotFoundError(msg)
    return mesh_files


def _maybe_save_visualization(  # noqa: PLR0913
    controller: RLGoalApproachController,
    env: LightweightEnv,
    episode: int,
    ep_result: str,
    goal_pose: np.ndarray,
    current_poses: list[np.ndarray],
    action_explanations: list[str],
    vis_dir: Path,
    vis_filter: dict[str, Any]=None,
    vis_counts: dict[str, int]=None,
) -> None:
    """Save episode visualization if it matches the filter criteria.

    Args:
        controller: RL controller (for termination counts).
        env: Environment instance.
        episode: Current episode number.
        episode_success: Whether the episode was successful.
        goal_pose: Goal pose array.
        current_poses: List of agent poses during episode.
        action_explanations: List of action description strings.
        vis_dir: Directory for saving visualizations.
        vis_filter: Filter configuration dict.
        vis_counts: Mutable counter dict for saved episodes per result.
    """
    if vis_filter and vis_counts:
        filter_actions = vis_filter.get("actions", [])
        has_any_actions = (
            any(
                any(act_name in expl for expl in action_explanations)
                for act_name in filter_actions
            )
            if filter_actions
            else True
        )
        max_count = vis_filter.get(f"max_{ep_result}", 5)
        under_limit = vis_counts[ep_result] < max_count
    else:
        has_any_actions = True
        under_limit = True

    if has_any_actions and under_limit:
        episode_id = f"ep_{episode+1:05d}_{ep_result}"
        save_episode_frames(
            env=env,
            goal_pose=goal_pose,
            episode_poses=current_poses,
            episode_actions=action_explanations,
            output_dir=vis_dir,
            episode_id=episode_id,
            result=ep_result,
        )
        if vis_counts:
            vis_counts[ep_result] += 1

def run_episodes(  # noqa: PLR0913, C901, PLR0912, PLR0915
    mesh_dir: str,
    save_dir: str,
    num_episodes: int | None = None,
    config: dict[str, Any] | None = None,
    mesh_path: str | None = None,
    load_dir: str | None = None,
    agent_id: str = "standalone",
    seed: int | None = None,
    return_metrics: bool = False,
    curriculum_config: dict[str, Any] | None = None,
    episode_script: list[dict[str, Any]] | None = None,
    episode_pools: list[list[dict[str, Any]]] | None = None,
    visualise: bool = False,
    episodes_per_level: int | None = None,
    level_idx=0,
) -> int | dict[str, Any]:
    """Run Q-learning episodes in train or eval mode.

    Executes episodes where an RL agent navigates on 3D object surfaces
    toward goal positions. The mode is determined by ``config["mode"]``:
    - ``"train"`` / ``"train_adapt_epsilon"``: agent learns via Q-updates,
      epsilon decays over episodes, curriculum promotes to harder levels.
    - ``"eval"``: agent uses learned Q-values with fixed low epsilon,
      no Q-updates, collects success trajectories for BC data.

    Episode sourcing:
    - **Fixed script**: a flat list of (start, goal) pairs executed
      sequentially. Used for eval on specific episode pools.
    - **Fixed pools**: per-level lists of episodes. Curriculum tracker
      selects the current level and advances when success rate exceeds
      the promotion threshold.
    - **Random**: start and goal are sampled from the mesh surface.
      Optional curriculum constrains goal distance range per level.

    Args:
        mesh_dir: Directory containing mesh files (.obj, .stl, .ply).
        save_dir: Directory to save the trained controller.
        num_episodes: Number of episodes to run.
        config: Configuration overrides merged with DEFAULT_CONFIG.
        mesh_path: Path to a specific mesh file. If None, a random
            mesh from mesh_dir is used each episode.
        load_dir: Directory to load a pretrained controller from.
        agent_id: Agent identifier string.
        seed: Random seed for reproducibility.
        return_metrics: If True, return a dict with detailed metrics
            instead of just the goal count.
        curriculum_config: Curriculum configuration dict with keys
            ``levels`` (list of (min_dist, max_dist) tuples),
            ``promote_threshold`` (float), ``promote_window`` (int).
        episode_script: Fixed list of episode dicts, each containing
            ``start_pos``, ``start_rot``, ``goal_pos``, ``goal_rot``.
        episode_pools: List of per-level episode lists for curriculum.
        visualise: If True, save PNG frames for filtered episodes.
        episodes_per_level: Cap on episodes per curriculum level.

    Returns:
        Number of goals reached if return_metrics is False,
        otherwise a dict with keys: ``goals_reached``, ``num_episodes``,
        ``success_rate``, ``stats``, ``curriculum_stats``,
        ``success_trails``, ``success_actions``.

    Raises:
        ValueError: If both episode_script and episode_pools are provided,
            or if num_episodes is missing when using episode_pools.
    """
    if seed is not None:
        np.random.seed(seed)  # noqa: NPY002
        random.seed(seed)

    if num_episodes is not None:
        if config is None:
            config = {"num_episodes": int(num_episodes)}
        else:
            config["num_episodes"] = int(num_episodes)
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # Visualization setup
    visualizer = None
    if visualise:
        from .visualize_env import EpisodeVisualizer

        mesh_label = Path(mesh_path).stem if mesh_path else "multi"
        stage_label = "eval" if cfg.get("mode") == "eval" else "train"
        visualizer = EpisodeVisualizer(
            output_dir=Path(save_dir),
            mesh_name=mesh_label,
            stage=stage_label,
        )

    # Validate episode sources
    use_script = episode_script is not None
    use_pools = episode_pools is not None
    if use_script and use_pools:
        msg = "Provide either episode_script or episode_pools, not both"
        raise ValueError(msg)

    if use_script:
        num_episodes = len(episode_script)
        cfg["num_episodes"] = num_episodes
    elif use_pools:
        if num_episodes is None:
            msg = "num_episodes is required when using episode_pools"
            raise ValueError(msg)
        cfg["num_episodes"] = int(num_episodes)

    mesh_files = _find_mesh_files(mesh_dir)
    controller = _init_controller(cfg, load_dir, agent_id, config)
    action_space = controller.action_space

    # Curriculum setup
    use_curriculum = curriculum_config is not None
    curriculum: _CurriculumTracker | None = None
    if use_curriculum and "train" in cfg["mode"]:
        curriculum = _CurriculumTracker(
            levels=curriculum_config["levels"],
            promote_threshold=float(
                curriculum_config.get("promote_threshold", 0.20)
            ),
            promote_window=int(
                curriculum_config.get("promote_window", 50)
            ),
        )

    pool_indices = [0] * len(episode_pools) if use_pools else []

    goals_reached = 0
    success_trails: list[Any] = []
    success_actions: list[list[str]] = []

    # ═══ Aggregate tracking across episodes ═══
    _all_phase_counts: dict[str, int] = {}
    _phase_transitions: dict[str, int] = {}
    _orbit_ages: list[int] = []
    _detach_outcomes: list[dict[str, Any]] = []
    _collision_per_phase: dict[str, int] = {}

    for episode in range(num_episodes):
        if episodes_per_level is not None and episode >= episodes_per_level:
            break

        ep_mesh_path = mesh_path or np.random.choice(mesh_files)  # noqa: NPY002
        env = LightweightEnv(ep_mesh_path, seed=seed)
        goals_before = controller._total_goals_reached

        # Setup episode
        if use_script:
            start_pos, goal_pose = _setup_episode_from_data(
                env, episode_script[episode]
            )
        elif use_pools:
            level_idx = (
                curriculum.level_idx if curriculum is not None else 0
            )
            if level_idx >= len(episode_pools):
                msg = (
                    f"Missing episode pool for level {level_idx}; "
                    f"available={len(episode_pools)}"
                )
                raise ValueError(msg)
            pool = episode_pools[level_idx]
            if pool_indices[level_idx] >= len(pool):
                msg = (
                    f"Episode pool exhausted for level {level_idx}: "
                    f"index {pool_indices[level_idx]}, size {len(pool)}"
                )
                raise ValueError(msg)
            start_pos, goal_pose = _setup_episode_from_data(
                env, pool[pool_indices[level_idx]]
            )
            pool_indices[level_idx] += 1
        else:
            env.reset()
            start_pos = env.get_pose()[:3]
            if curriculum is not None:
                min_d, max_d = curriculum.current_bounds
                goal_pose = env.get_random_surface_point(
                    reference_pos=start_pos,
                    min_dist=min_d,
                    max_dist=max_d,
                    max_attempts=2000,
                )
                goal_dist = float(
                    np.linalg.norm(goal_pose[:3] - start_pos)
                )
                if goal_dist < min_d or goal_dist > max_d:
                    curriculum.stats["fallback_episodes"] += 1
            else:
                goal_pose = env.get_random_surface_point()

        controller.set_new_goal(goal_pose, start_pos)
        env.set_goal(goal_pose)

        # ═══ Air start: every 3rd training episode ═══
        is_training = "train" in cfg.get("mode", "")
        air_start_enabled = cfg.get(
            "air_start_enabled", False
        )
        air_start_in_eval = cfg.get(
            "air_start_in_eval", False
        )

        if (
            (is_training or air_start_in_eval)
            and air_start_enabled
            and episode % 3 == 2
        ):
            sensor = env.get_sensor_data()
            normal = sensor.get("point_normal")
            if normal is not None:
                n = np.array(normal, dtype=float)
                n_len = np.linalg.norm(n)
                if n_len > 1e-8:
                    n /= n_len
                    detach_dist = (
                        action_space.free_step * 3
                    )
                    air_pos = (
                        env.agent_pos + n * detach_dist
                    )

                    goal_dir = goal_pose[:3] - air_pos
                    goal_dist = np.linalg.norm(goal_dir)
                    if goal_dist > 1e-8:
                        air_rot = (
                            env._look_at_direction(
                                goal_dir / goal_dist
                            )
                        )
                    else:
                        air_rot = env.agent_rot.copy()

                    env.agent_pos = air_pos
                    env.agent_rot = air_rot
                    start_pos = air_pos.copy()
                    controller.set_new_goal(
                        goal_pose, start_pos
                    )
                    controller._total_episodes -= 1
        controller.temperature_override = cfg.get("temperature_override")

        # Per-episode tracking
        action_explanations: list[str] = []
        current_poses: list[np.ndarray] = []
        ep_phase_counts: dict[str, int] = {}
        ep_prev_phase: str | None = None
        ep_detach_start_step: int | None = None
        ep_detach_start_dist: float | None = None

        # Run episode steps
        for _step in range(controller.config["max_steps_per_goal"]):

            # ═══ 1. Check landing from previous detach ═══
            if ep_detach_start_step is not None:
                land_sensor = env.get_sensor_data()
                if land_sensor.get("on_object", False):
                    air_steps = _step - ep_detach_start_step
                    land_pose = env.get_pose()
                    land_dist = float(
                        np.linalg.norm(
                            goal_pose[:3] - land_pose[:3]
                        )
                    )
                    dist_impr = ep_detach_start_dist - land_dist
                    _detach_outcomes.append({
                        "air_steps": air_steps,
                        "dist_improvement": round(dist_impr, 1),
                        "landed": True,
                    })
                    ep_detach_start_step = None
                    ep_detach_start_dist = None

            # ═══ 2. Controller decides action ═══
            current_pose = env.get_pose()
            sensor_data = env.get_sensor_data()
            _, explanation = controller.step(
                current_pose, sensor_data
            )
            if explanation is not None:
                action_explanations.append(
                    explanation["interpretation"]
                )
            current_poses.append(env.get_pose())

            # ═══ 3. Track phase ═══
            cur_phase = getattr(
                controller, "_current_phase", "UNKNOWN"
            )
            ep_phase_counts[cur_phase] = (
                ep_phase_counts.get(cur_phase, 0) + 1
            )
            if (
                ep_prev_phase is not None
                and ep_prev_phase != cur_phase
            ):
                t_key = f"{ep_prev_phase}\u2192{cur_phase}"
                _phase_transitions[t_key] = (
                    _phase_transitions.get(t_key, 0) + 1
                )
                # Track orbit age when leaving FLY_TO_EDGE
                if ep_prev_phase == "FLY_TO_EDGE":
                    o_age = getattr(
                        controller, "_orbit_direction_age", 0
                    )
                    _orbit_ages.append(o_age)
            ep_prev_phase = cur_phase

            # ═══ 4. Check if episode ended inside controller ═══
            if controller._current_goal is None:
                goals_reached = max(
                    goals_reached,
                    controller._total_goals_reached,
                )
                break

            # ═══ 5. Execute action in environment ═══
            env.step(
                controller._last_action,
                action_space,
            )

            # ═══ 6. Record detach AFTER env executed it ═══
            if (
                controller._last_action is not None
                and controller.action_space.get_info(
                    controller._last_action
                ).name == "detach"
            ):
                post_detach_sensor = env.get_sensor_data()
                # Only record if agent actually left surface
                if not post_detach_sensor.get("on_object", True):
                    post_detach_pose = env.get_pose()
                    ep_detach_start_step = _step
                    ep_detach_start_dist = float(
                        np.linalg.norm(
                            goal_pose[:3] - post_detach_pose[:3]
                        )
                    )

        # ═══ Post-episode processing ═══
        episode_success = (
            controller._total_goals_reached > goals_before
        )

        if episode_success:
            success_trails.append(controller.success_trails)
            success_actions.append(action_explanations)
            ep_result = "success"
        else:
            if _step == (
                controller.config["max_steps_per_goal"] - 1
            ):
                ep_result = "timeout"
            else:
                ep_result = "collision"

        # Aggregate phase counts
        for phase, count in ep_phase_counts.items():
            _all_phase_counts[phase] = (
                _all_phase_counts.get(phase, 0) + count
            )

        # Track collision phase
        if ep_result == "collision" and ep_prev_phase:
            _collision_per_phase[ep_prev_phase] = (
                _collision_per_phase.get(ep_prev_phase, 0) + 1
            )

        # Handle unresolved detach (episode ended in air)
        if ep_detach_start_step is not None:
            air_steps = _step - ep_detach_start_step
            _detach_outcomes.append({
                "air_steps": air_steps,
                "dist_improvement": 0.0,
                "landed": False,
            })

        if visualizer:
            level = (
                curriculum.level_idx if curriculum else level_idx
            )
            if ((episode + 1) % _LOG_INTERVAL == 0) and ((episode + 1) >= 100):
                visualizer.save_episode(
                    env=env,
                    episode=episode,
                    level=level,
                    result=ep_result,
                    goal_pose=goal_pose,
                    poses=current_poses,
                    actions=action_explanations,
                )

        if curriculum is not None:
            warmup_episodes = int(
                cfg.get("warmup_episodes", 0)
            )
            is_warmup = (
                warmup_episodes > 0
                and controller._total_episodes
                <= warmup_episodes
            )
            if not is_warmup:
                curriculum.on_episode_end(
                    episode_success, controller, episode
                )

        if (episode + 1) % _LOG_INTERVAL == 0:
            logger.info(
                "Episode %d/%d: stats=%s",
                episode + 1,
                num_episodes,
                controller.get_stats(),
            )

    controller.save(save_dir)
    logger.info("Saved to %s", save_dir)
    logger.info(
        "Final: %d/%d goals reached", goals_reached, num_episodes
    )

    if not return_metrics:
        return goals_reached

    stats = controller.get_stats()
    success_rate = goals_reached / max(num_episodes, 1)
    curriculum_stats = (
        curriculum.finalize() if curriculum is not None else None
    )
    if curriculum is None and use_curriculum:
        curriculum_stats = {
            "levels_reached": 1,
            "episodes_per_level": [num_episodes],
            "successes_per_level": [goals_reached],
            "success_rate_per_level": [success_rate],
            "final_level": 0,
            "fallback_rate": 0.0,
        }

    # ═══ Build extended metrics ═══
    phase_metrics = {
        "phase_distribution": _all_phase_counts,
        "phase_transitions": _phase_transitions,
        "collision_per_phase": _collision_per_phase,
    }

    if _orbit_ages:
        phase_metrics["orbit_stats"] = {
            "mean_orbit_age": round(
                float(np.mean(_orbit_ages)), 1
            ),
            "median_orbit_age": round(
                float(np.median(_orbit_ages)), 1
            ),
            "max_orbit_age": int(np.max(_orbit_ages)),
            "orbits_over_15": sum(
                1 for a in _orbit_ages if a > 15
            ),
            "total_orbits": len(_orbit_ages),
        }

    if _detach_outcomes:
        air_steps_list = [
            d["air_steps"] for d in _detach_outcomes
        ]
        landed = [
            d for d in _detach_outcomes if d["landed"]
        ]
        phase_metrics["detach_analysis"] = {
            "total_detaches": len(_detach_outcomes),
            "mean_air_steps": round(
                float(np.mean(air_steps_list)), 1
            ),
            "median_air_steps": round(
                float(np.median(air_steps_list)), 1
            ),
            "max_air_steps": int(np.max(air_steps_list)),
            "landed_count": len(landed),
            "landed_rate": round(
                len(landed)
                / max(len(_detach_outcomes), 1),
                3,
            ),
            "mean_dist_improvement": (
                round(
                    float(
                        np.mean(
                            [
                                d["dist_improvement"]
                                for d in landed
                            ]
                        )
                    ),
                    1,
                )
                if landed
                else 0
            ),
        }

    return {
        "goals_reached": goals_reached,
        "num_episodes": num_episodes,
        "success_rate": success_rate,
        "stats": stats,
        "curriculum_stats": curriculum_stats,
        "success_trails": success_trails,
        "success_actions": success_actions,
        "phase_metrics": phase_metrics,
    }
