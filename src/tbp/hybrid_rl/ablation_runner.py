# ablation_runner.py
from typing import Any, Dict, List, Optional

import logging
import collections
import numpy as np
import random
import glob
from pathlib import Path


from tbp.hybrid_rl.rl_goal_approach_controller import RLGoalApproachController
from tbp.hybrid_rl.lightweight_env import LightweightEnv
from tbp.hybrid_rl.config import DEFAULT_CONFIG

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
    EPISODES_PER_LEVEL=None,
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

    # Визуализация: сбор эпизодов
    _vis_counts = {"success": 0, "collision": 0, "timeout": 0}
    _vis_dir = None
    _vis_filter = None
    if visualise:
        _vis_dir = Path(save_dir) / "visualizations"
        _vis_filter = (config or {}).get("visualise_filter", {
            "actions": ["detach", "detach_edge"],
            "max_success": 5,
            "max_collision": 5,
            "max_timeout": 5,
        })

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
        if cfg.get("unfreeze_normalization", False):
            controller.q_store_free._norm_frozen = False
            controller.q_store_free._freeze_done = False
            controller.q_store_surface._norm_frozen = False
            controller.q_store_surface._freeze_done = False
            logger.info("Normalization unfrozen for transfer learning")

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
                strategy = retry_strategies[min(retry, len(retry_strategies) - 1)]
                controller.eval_epsilon = strategy["eval_epsilon"]
                controller.temperature_override = strategy["temperature_override"]
                action_explanations = []
                current_poses = []
            else:
                controller.temperature_override = cfg.get("temperature_override", None)

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
            logger.debug(f"SUCCESS, start_distance {start_distance}")
        else:
            logger.debug(f"ERROR, start_distance {start_distance}")

        # Визуализация: сохранить эпизод если подходит под фильтр
        if visualise and _vis_dir and _vis_filter:
            from tbp.hybrid_rl.visualize_env import save_episode_frames

            # Определить результат
            last_termination = max(
                controller._termination_counts.items(), key=lambda x: x[1]
            )
            if _episode_success:
                ep_result = "success"
            elif "collision" in str(last_termination[0]):
                ep_result = "collision"
            else:
                ep_result = "timeout"

            # Проверить фильтр: ВСЕ указанные действия должны присутствовать
            filter_actions = _vis_filter.get("actions", [])
            has_all_actions = all(
                any(act_name in expl for expl in action_explanations)
                for act_name in filter_actions
            ) if filter_actions else True

            # Проверить лимит
            max_count = _vis_filter.get(f"max_{ep_result}", 5)
            under_limit = _vis_counts[ep_result] < max_count

            if has_all_actions and under_limit:
                episode_id = f"ep_{episode:05d}_{ep_result}"
                save_episode_frames(
                    env=env,
                    goal_pose=goal_pose,
                    episode_poses=current_poses,
                    episode_actions=action_explanations,
                    output_dir=_vis_dir,
                    episode_id=episode_id,
                    result=ep_result,
                )
                _vis_counts[ep_result] += 1
                print(
                    f"[Vis] Saved {ep_result} episode: {episode_id} "
                    f"(counts: {_vis_counts})"
                )

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
                        f"  [Curriculum] ep={episode + 1}: promoted to level "
                        f"{_curr_level_idx}: dist [{_new_min}, {_new_max}] mm "
                        f"(rolling_rate={_rolling_rate:.3f})"
                        f"(epsilon={controller.epsilon:.3f})"
                    )

        if (episode + 1) % 1000 == 0:
            stats = controller.get_stats()
            print(
                f"Episode {episode + 1}/{num_episodes}: "
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
