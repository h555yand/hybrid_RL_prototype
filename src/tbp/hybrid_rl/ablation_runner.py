# ablation_runner.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import json
from pathlib import Path
import trimesh
import logging
import collections
import numpy as np
import random
import glob

from tbp.hybrid_rl.rl_goal_approach_controller import RLGoalApproachController
from tbp.hybrid_rl.lightweight_env import LightweightEnv
from tbp.hybrid_rl.config import DEFAULT_CONFIG
from tbp.hybrid_rl.visualize_env import visualize_agent_goal

GOAL_THRESHOLD_PER_LEVEL = [5.0, 3.0, 2.0]

logger = logging.getLogger(__name__)


def train(
    mesh_dir,
    save_dir,
    num_episodes=None,
    config=None,
    mesh_path=None,
    load_dir=None,
    agent_id="standalone",
    seed=None,
    return_metrics=False,
    curriculum_config=None,
    episode_script: Optional[List[Dict[str, Any]]] = None,
    episode_pools: Optional[List[List[Dict[str, Any]]]] = None,
    visualise=False,
    EPISODES_PER_LEVEL=None
):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if num_episodes is not None:
        if config is None:
            config = {"num_episodes": int(num_episodes)}
        else:
            config["num_episodes"] = int(num_episodes)
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    use_fixed_episode_script = episode_script is not None
    use_fixed_episode_pools = episode_pools is not None
    if use_fixed_episode_script and use_fixed_episode_pools:
        raise ValueError("Provide either episode_script or episode_pools, not both")

    if use_fixed_episode_script:
        num_episodes = len(episode_script)
        cfg["num_episodes"] = num_episodes
        print(f"[Train] Using fixed episode script with {num_episodes} episodes")
    elif use_fixed_episode_pools:
        if num_episodes is None:
            raise ValueError("num_episodes is required when using episode_pools")
        cfg["num_episodes"] = int(num_episodes)
        print(
            f"[Train] Using fixed episode pools with {len(episode_pools)} levels "
            f"for {num_episodes} scheduled episodes"
        )

    mesh_files = (
        glob.glob(f"{mesh_dir}/*.obj")
        + glob.glob(f"{mesh_dir}/*.stl")
        + glob.glob(f"{mesh_dir}/*.ply")
    )

    if not mesh_files:
        raise FileNotFoundError(f"No mesh files in {mesh_dir}")

    print(f"Train Mode: {cfg['mode']}")

    if load_dir is None:
        controller = RLGoalApproachController(
            agent_id=agent_id,
            config=cfg,
        )
    else:
        controller = RLGoalApproachController.load(load_dir, agent_id=agent_id, config=config)

    action_space = controller.action_space

    _use_curriculum = curriculum_config is not None
    _pool_indices: List[int] = []
    if _use_curriculum and "train" in cfg["mode"]:
        _curr_level_idx = 0
        _curr_window: collections.deque = collections.deque()
        _curr_level_episodes = 0
        _curr_level_successes = 0
        _curriculum_stats: dict = {
            "levels_reached": 1,
            "episodes_per_level": [],
            "successes_per_level": [],
            "success_rate_per_level": [],
            "fallback_episodes": 0,
        }
        _curr_levels = list(curriculum_config["levels"])
        _promote_threshold = float(curriculum_config.get("promote_threshold", 0.20))
        _promote_window = int(curriculum_config.get("promote_window", 50))
        _curr_window = collections.deque(maxlen=_promote_window)
        print(
            f"[Curriculum] Starting level 0: "
            f"dist [{_curr_levels[0][0]}, {_curr_levels[0][1]}] mm, "
            f"promote_threshold={_promote_threshold}, window={_promote_window}"
        )

    if use_fixed_episode_pools:
        _pool_indices = [0 for _ in range(len(episode_pools))]

    goals_reached = 0
    success_trails = []
    success_actions = []

    for episode in range(num_episodes):
        if EPISODES_PER_LEVEL is not None and episode >= EPISODES_PER_LEVEL:
            break
        episode_mesh_path = mesh_path
        if episode_mesh_path is None:
            episode_mesh_path = np.random.choice(mesh_files)
        env = LightweightEnv(episode_mesh_path, seed=seed)

        _goals_before_episode = controller._total_goals_reached

        if use_fixed_episode_script:
            ep_data = episode_script[episode]
            start_pos = np.array(ep_data["start_pos"])
            start_rot = np.array(ep_data["start_rot"])
            env.reset(position=start_pos, rotation=start_rot)
            goal_pose = np.concatenate([
                np.array(ep_data["goal_pos"]),
                np.array(ep_data["goal_rot"]),
            ])
        elif use_fixed_episode_pools:
            pool_level_idx = _curr_level_idx if (_use_curriculum and "train" in cfg["mode"]) else 0
            if pool_level_idx >= len(episode_pools):
                raise ValueError(
                    f"Missing episode pool for level {pool_level_idx}; available={len(episode_pools)}"
                )
            level_pool = episode_pools[pool_level_idx]
            pool_index = _pool_indices[pool_level_idx]
            if pool_index >= len(level_pool):
                raise ValueError(
                    f"Episode pool exhausted for level {pool_level_idx}: "
                    f"requested index {pool_index}, pool size {len(level_pool)}"
                )
            ep_data = level_pool[pool_index]
            _pool_indices[pool_level_idx] += 1
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
            start_rot = env.get_pose()[3:]

            if _use_curriculum and "train" in cfg["mode"]:
                _min_d, _max_d = _curr_levels[_curr_level_idx]
                goal_pose = env.get_random_surface_point(
                    reference_pos=start_pos,
                    min_dist=_min_d,
                    max_dist=_max_d,
                    max_attempts=2000,
                )
                _goal_dist = float(np.linalg.norm(goal_pose[:3] - start_pos))
                if _goal_dist < _min_d or _goal_dist > _max_d:
                    _curriculum_stats["fallback_episodes"] += 1
            else:
                goal_pose = env.get_random_surface_point()

        controller.set_new_goal(goal_pose, start_pos)
        env.set_goal(goal_pose)

        retry_strategies = [
            {"eval_epsilon": cfg.get("eval_epsilon", 0.02), "temperature_override": None},
            {"eval_epsilon": 1.0, "temperature_override": 0.01},
            {"eval_epsilon": 0.5, "temperature_override": 0.01},
        ]

        max_retries = 3 if cfg.get("mode") == "eval_retries" else 1

        action_explanations = []
        current_poses = []

        for retry in range(max_retries):
            if retry > 0:
                if use_fixed_episode_script or use_fixed_episode_pools:
                    env.reset(position=start_pos, rotation=start_rot)
                else:
                    env.reset(position=start_pos)
                controller.set_new_goal(goal_pose, start_pos)
                env.set_goal(goal_pose)
                strategy = retry_strategies[min(retry, len(retry_strategies)-1)]
                controller.eval_epsilon = strategy["eval_epsilon"]
                controller.temperature_override = strategy["temperature_override"]
                action_explanations = []
                current_poses = []
            else:
                if controller.eval_epsilon == 1.0:
                    controller.temperature_override = 0.01

            for step in range(controller.config["max_steps_per_goal"]):
                current_pose = env.get_pose()
                sensor_data = env.get_sensor_data()

                action, explanation = controller.step(current_pose, sensor_data)
                if explanation is not None:
                    action_explanations.append(explanation["interpretation"])
                current_poses.append(env.get_pose())

                if controller._current_goal is None:
                    if controller._total_goals_reached > goals_reached:
                        goals_reached = controller._total_goals_reached
                    break

                action_index = controller._last_action
                env.step(action_index, action_space)

            _episode_success = controller._total_goals_reached > _goals_before_episode
            if _episode_success:
                break

        if max_retries > 1:
            controller.eval_epsilon = cfg.get("eval_epsilon", 0.02)
            controller.temperature_override = None

        start_distance = float(np.linalg.norm(goal_pose[:3] - start_pos))

        if _episode_success:
            success_trails.append(controller.success_trails)
            success_actions.append(action_explanations)
            logger.debug(f"SUCCESS, start_distance {start_distance}, explain_action_info: {action_explanations}")
            #if visualise:
            #    visualize_agent_goal(env, np.concatenate([start_pos, start_rot]), goal_pose)
            #    for pose in current_poses:
            #        visualize_agent_goal(env, pose, goal_pose)
        else:
            logger.debug(f"ERROR, start_distance {start_distance}, explain_action_info: {action_explanations}")

        if _use_curriculum and "train" in cfg["mode"]:
            _episode_success = controller._total_goals_reached > _goals_before_episode
            _curr_window.append(_episode_success)
            _curr_level_episodes += 1
            if _episode_success:
                _curr_level_successes += 1
            window_full = len(_curr_window) == _promote_window
            not_last_level = _curr_level_idx < len(_curr_levels) - 1
            if window_full and not_last_level:
                _rolling_rate = sum(_curr_window) / _promote_window
                if _rolling_rate >= _promote_threshold:
                    _curriculum_stats["episodes_per_level"].append(_curr_level_episodes)
                    _curriculum_stats["successes_per_level"].append(_curr_level_successes)
                    _curriculum_stats["success_rate_per_level"].append(
                        _curr_level_successes / max(_curr_level_episodes, 1)
                    )
                    _curr_level_idx += 1
                    if _curr_level_idx < len(GOAL_THRESHOLD_PER_LEVEL):
                        new_threshold = GOAL_THRESHOLD_PER_LEVEL[_curr_level_idx]
                        controller.config["goal_threshold"] = new_threshold
                        logger.info(f"  [Curriculum] goal_threshold → {new_threshold}mm")
                    _curr_window = collections.deque(maxlen=_promote_window)
                    _curr_level_episodes = 0
                    _curr_level_successes = 0
                    _curriculum_stats["levels_reached"] = _curr_level_idx + 1
                    _new_min, _new_max = _curr_levels[_curr_level_idx]
                    print(
                        f"  [Curriculum] ep={episode+1}: promoted to level "
                        f"{_curr_level_idx}: dist [{_new_min}, {_new_max}] mm "
                        f"(rolling_rate={_rolling_rate:.3f})"
                        f"(epsilon={controller.epsilon:.3f})"
                    )

        if (episode + 1) % 1000 == 0:
            stats = controller.get_stats()
            print(
                f"Episode {episode+1}/{num_episodes}: "
                f"stats={stats}"
            )

    controller.save(save_dir)
    print(f"Saved to {save_dir}")
    print(f"Final: {goals_reached}/{num_episodes} goals reached")

    if return_metrics:
        stats = controller.get_stats()
        success_rate = goals_reached / max(num_episodes, 1)
        if _use_curriculum and "train" in cfg["mode"]:
            _curriculum_stats["episodes_per_level"].append(_curr_level_episodes)
            _curriculum_stats["successes_per_level"].append(_curr_level_successes)
            _curriculum_stats["success_rate_per_level"].append(
                _curr_level_successes / max(_curr_level_episodes, 1)
            )
            _curriculum_stats["final_level"] = _curr_level_idx
            _curriculum_stats["fallback_rate"] = (
                _curriculum_stats["fallback_episodes"] / max(num_episodes, 1)
            )
        return {
            "goals_reached": goals_reached,
            "num_episodes": num_episodes,
            "success_rate": success_rate,
            "stats": stats,
            "curriculum_stats": _curriculum_stats if _use_curriculum else None,
            "success_trails": success_trails,
            "success_actions": success_actions,
        }

    return goals_reached

@dataclass
class AblationSummary:
    variant: str
    success_rate: float
    update_hit_rate_free: float
    active_to_created_ratio_free: float
    points_per_update_ratio_free: float
    update_hit_rate_surface: float
    active_to_created_ratio_surface: float
    points_per_update_ratio_surface: float
    timeout_rate: float = 0.0
    collision_surface_violation_rate: float = 0.0
    levels_reached: float = 0.0
    fallback_rate: float = 0.0
    collision_stats: Optional[Dict[str, Any]] = None


class RLAblationRunner:
    """Run A/B/C/D ablations across shared seeds and analyze results."""

    def __init__(
        self,
        mesh_dir: str,
        mesh_path: str,
        save_root_dir: str,
        num_episodes: int,
        base_config: Optional[Dict[str, Any]] = None,
        seeds: Optional[List[int]] = None,
        episode_pools_by_seed: Optional[Dict[int, Dict[str, Any]]] = None,
        is_load: bool = False
    ):
        self.mesh_dir = mesh_dir
        self.mesh_path = mesh_path
        self.save_root_dir = save_root_dir
        self.num_episodes = num_episodes
        self.base_config = base_config or {}
        self.seeds = seeds or [11, 22, 33]
        self.episode_pools_by_seed = episode_pools_by_seed or {}
        self.is_load = is_load

    def default_variants(self) -> Dict[str, Dict[str, Any]]:
        return {
            "A": {"insert_threshold": 0.50, "auto_calibrate": False}, # A was best
            "B": {"insert_threshold": 0.60, "auto_calibrate": False},
            "C": {"insert_threshold": 0.50, "k_neighbors": 11, "auto_calibrate": False},
            "D": {
                "insert_threshold": 0.50,
                "auto_calibrate": False,
                "mode": "auto",
                "auto_train_threshold": 3500,
                "eval_epsilon": 0.02,
            },
        }

    def goal_reward_variants(self) -> Dict[str, Dict[str, Any]]:
        """Stage 1: tune goal/reward pipeline with fixed max_steps_per_goal=20."""
        return {
            "G0": {
                "reward_goal_reached": 50.0,
                #"goal_threshold": 4.0,
                "reward_timeout": -10.0,
            },
            "G1": {
                "reward_goal_reached": 60.0,
                #"goal_threshold": 4.0,
                "reward_timeout": -8.0,
            },
            "G2": {
                "reward_goal_reached": 70.0,
                #"goal_threshold": 5.0,
                "reward_timeout": -6.0,
            },
        }

    def max_steps_variants(self, reward_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """Stage 2: compare max_steps_per_goal with fixed reward settings."""
        reward_overrides = reward_overrides or {}
        base = {
            "insert_threshold": 0.50,
            "auto_calibrate": False,
            **reward_overrides,
        }
        return {
            "M20": {**base, "max_steps_per_goal": 20},
            "M30": {**base, "max_steps_per_goal": 30},
            "M40": {**base, "max_steps_per_goal": 40},
        }

    def curriculum_variants(
        self,
        reward_overrides: Optional[Dict[str, Any]] = None,
        levels: Optional[List[Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Stage 3: compare curriculum schedules with fixed reward settings."""
        reward_overrides = reward_overrides or {}
        levels = levels or [(10.0, 30.0), (10.0, 70.0), (10.0, 120.0)]
        base = {
            "insert_threshold": 0.50,
            "auto_calibrate": False,
            **reward_overrides,
        }
        return {
            #"CL1": {
            #    **base,
            #    "curriculum_config": {
            #        "levels": levels,
            #        "promote_threshold": 0.1,
            #        "promote_window": 50,
            #    },
            #},
            #"CL2": {
            #    **base,
            #    "curriculum_config": {
            #        "levels": levels,
            #        "promote_threshold": 0.15,
            #        "promote_window": 50,
            #    },
            #},
            "CL3": {
                **base,
                "curriculum_config": {
                    "levels": levels,
                    "promote_threshold": 0.8,
                    "promote_window": 100,
                },
            },
        }

    def run(self, variants: Optional[Dict[str, Dict[str, Any]]] = None, visualise=False) -> Dict[str, Any]:
        variants = variants or self.default_variants()
        raw_results: Dict[str, List[Dict[str, Any]]] = {name: [] for name in variants}

        for seed in self.seeds:
            for variant_name, overrides in variants.items():
                curriculum_config = overrides.get("curriculum_config")
                cfg_overrides = {
                    k: v for k, v in overrides.items() if k != "curriculum_config"
                }
                cfg = {**self.base_config, **cfg_overrides}
                save_dir = f"{self.save_root_dir}/{variant_name.lower()}_seed_{seed}"
                seed_episode_pools = self.episode_pools_by_seed.get(seed)
                episode_pools = None
                if seed_episode_pools is not None:
                    episode_pools = seed_episode_pools.get("levels")
                    pool_sizes = [len(pool) for pool in episode_pools]
                    print(
                        f"[Ablation] Using fixed episode pools for seed={seed} "
                        f"(levels={pool_sizes})"
                    )
                if self.is_load:
                    load_dir = save_dir
                else:
                    load_dir = None

                run_result = train(
                    mesh_dir=self.mesh_dir,
                    save_dir=save_dir,
                    load_dir=load_dir,
                    num_episodes=self.num_episodes,
                    config=cfg,
                    mesh_path=self.mesh_path,
                    seed=seed,
                    return_metrics=True,
                    curriculum_config=curriculum_config,
                    episode_pools=episode_pools,
                    visualise=visualise
                )
                run_result["seed"] = seed
                raw_results[variant_name].append(run_result)

        summaries = {
            name: self._aggregate_variant(name, runs) for name, runs in raw_results.items()
        }
        best_variant = self._pick_best_variant(summaries)

        return {
            "seeds": self.seeds,
            "raw_results": raw_results,
            "summaries": {k: v.__dict__ for k, v in summaries.items()},
            "best_variant": best_variant,
        }

    @staticmethod
    def _pick_best_variant(summaries: Dict[str, AblationSummary]) -> str:
        ranked = sorted(
            summaries.values(),
            key=lambda s: (
                -s.success_rate,
                s.timeout_rate,
                s.collision_surface_violation_rate,
            ),
        )
        return ranked[0].variant

    def _aggregate_variant(self, variant: str, runs: List[Dict[str, Any]]) -> AblationSummary:
        def _metric(run: Dict[str, Any], key: str, default: float = 0.0, name="not defined") -> float:
            stats = run.get("stats", {})
            # Current train() returns controller stats with Q-store metrics nested under q_store.
            # Keep top-level fallback for compatibility with older synthetic tests.
            q_store_stats = stats.get(name, {})
            if key in q_store_stats:
                return float(q_store_stats.get(key, default))
            return float(stats.get(key, default))

        def _termination_rate(run: Dict[str, Any], key: str, default: float = 0.0) -> float:
            stats = run.get("stats", {})
            rates = stats.get("termination_rates", {})
            return float(rates.get(key, default))

        success_rate = float(np.mean([float(r["success_rate"]) for r in runs]))

        update_hit_rate_free = float(np.mean([_metric(r, "update_hit_rate", name="q_store_free") for r in runs]))
        active_to_created_ratio_free = float(
            np.mean([_metric(r, "active_to_created_ratio", name="q_store_free") for r in runs])
        )
        points_per_update_ratio_free = float(
            np.mean([_metric(r, "points_per_update_ratio", name="q_store_free") for r in runs])
        )

        update_hit_rate_surface = float(np.mean([_metric(r, "update_hit_rate", name="q_store_surface") for r in runs]))
        active_to_created_ratio_surface = float(
            np.mean([_metric(r, "active_to_created_ratio", name="q_store_surface") for r in runs])
        )
        points_per_update_ratio_surface = float(
            np.mean([_metric(r, "points_per_update_ratio", name="q_store_surface") for r in runs])
        )

        timeout_rate = float(np.mean([_termination_rate(r, "timeout") for r in runs]))
        collision_surface_violation_rate = float(
            np.mean([_termination_rate(r, "collision_surface_violation") for r in runs])
        )
        levels_reached = float(
            np.mean(
                [
                    float((r.get("curriculum_stats") or {}).get("levels_reached", 0.0))
                    for r in runs
                ]
            )
        )
        fallback_rate = float(
            np.mean(
                [
                    float((r.get("curriculum_stats") or {}).get("fallback_rate", 0.0))
                    for r in runs
                ]
            )
        )
        # ─── ДОБАВЛЕНО: агрегация collision_stats ───
        merged_collision_stats: Dict[str, float] = {}
        for r in runs:
            cs = r.get("stats", {}).get("collision_stats", {})
            for action_name, count in cs.items():
                merged_collision_stats[action_name] = (
                    merged_collision_stats.get(action_name, 0.0) + float(count)
                )
        # Усредняем по количеству seed'ов
        n_runs = max(len(runs), 1)
        averaged_collision_stats = {
            k: v / n_runs for k, v in merged_collision_stats.items()
        }

        return AblationSummary(
            variant=variant,
            success_rate=success_rate,
            update_hit_rate_free=update_hit_rate_free,
            active_to_created_ratio_free=active_to_created_ratio_free,
            points_per_update_ratio_free=points_per_update_ratio_free,
            update_hit_rate_surface=update_hit_rate_surface,
            active_to_created_ratio_surface=active_to_created_ratio_surface,
            points_per_update_ratio_surface=points_per_update_ratio_surface,
            timeout_rate=timeout_rate,
            collision_surface_violation_rate=collision_surface_violation_rate,
            levels_reached=levels_reached,
            fallback_rate=fallback_rate,
            collision_stats=averaged_collision_stats,  # ← ДОБАВЛЕНО
        )

    @staticmethod
    def choose_next_debug_direction(
        run_a: AblationSummary,
        run_b: AblationSummary,
        run_c: AblationSummary,
        run_d: AblationSummary,
        hit_rate_tol: float = 0.03,
        q_store_tol: float = 0.05,
    ) -> str:
        if (
            run_b.success_rate > run_a.success_rate
            and run_b.update_hit_rate > run_a.update_hit_rate
        ):
            return "threshold_reuse"

        if (
            run_c.success_rate > max(run_a.success_rate, run_b.success_rate, run_d.success_rate)
            and abs(run_c.update_hit_rate - run_a.update_hit_rate) <= hit_rate_tol
        ):
            return "kernel_smoothing"

        if (
            run_d.success_rate > max(run_a.success_rate, run_b.success_rate, run_c.success_rate)
            and abs(run_d.points_per_update_ratio - run_a.points_per_update_ratio)
            <= q_store_tol
        ):
            return "epsilon_reward_dynamics"

        store_good = (
            run_a.update_hit_rate >= 0.50
            and run_a.active_to_created_ratio >= 0.80
            and 0.25 <= run_a.points_per_update_ratio <= 0.50
        )
        no_success_help = (
            run_b.success_rate <= run_a.success_rate
            and run_c.success_rate <= run_a.success_rate
            and run_d.success_rate <= run_a.success_rate
        )
        if store_good and no_success_help:
            return "goal_reward_pipeline"

        return "no_clear_winner"

    @staticmethod
    def suggest_next_changes(direction: str) -> Dict[str, Any]:
        recommendations = {
            "threshold_reuse": {
                "focus": "raise state reuse",
                "changes": {
                    "insert_threshold": [0.65, 0.70],
                    "auto_calibrate": False,
                },
                "target": {
                    "update_hit_rate": ">= 0.55",
                    "points_per_update_ratio": "<= 0.45",
                },
            },
            "kernel_smoothing": {
                "focus": "improve interpolation smoothness",
                "changes": {
                    "k_neighbors": [11, 13],
                    "sigma": [1.1, 1.3],
                    "adaptive_sigma": True,
                },
                "target": {
                    "success_rate": "increase without hit_rate drop",
                },
            },
            "epsilon_reward_dynamics": {
                "focus": "exploration schedule",
                "changes": {
                    "mode": "auto",
                    "auto_train_threshold": [3000, 4000],
                    "eval_epsilon": [0.02, 0.05],
                    "epsilon_decay": [0.9998, 0.9999],
                },
                "target": {
                    "success_rate": "increase with similar q_store stats",
                },
            },
            "goal_reward_pipeline": {
                "focus": "goal/reward/termination path",
                "changes": {
                    "reward_goal_reached": [60.0, 80.0],
                    "goal_threshold": [3.0, 5.0],
                    "reward_timeout": [-8.0, -5.0],
                },
                "target": {
                    "success_rate": "increase while store metrics remain stable",
                },
            },
            "no_clear_winner": {
                "focus": "collect stronger evidence",
                "changes": {
                    "seeds": [11, 22, 33, 44, 55],
                    "num_episodes": "same or +20%",
                },
                "target": {
                    "confidence": "lower run-to-run variance",
                },
            },
        }
        return recommendations.get(direction, recommendations["no_clear_winner"])
