# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Episode pool generation and management.

Provides functions to generate, save, load, and cache episode pools
for reproducible training and evaluation.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

from tbp.hybrid_rl.lightweight_env import LightweightEnv, _is_reachable_by_surface

logger = logging.getLogger(__name__)


def generate_episode_pools(
    mesh_path: str,
    episodes_per_level: int,
    seed: int,
    curriculum_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate episode pools with fixed start/goal
    positions.

    Supports filtering by same_side and path_blocked
    per curriculum level.

    Args:
        mesh_path: Path to the mesh file.
        episodes_per_level: Number of episodes per
            curriculum level.
        seed: Random seed for reproducibility.
        curriculum_config: Optional curriculum config
            with "levels" and "filters" keys.

    Returns:
        Dict with seed, episodes_per_level,
        curriculum_levels, and levels.
    """
    np.random.seed(seed)
    random.seed(seed)
    env = LightweightEnv(mesh_path, seed=seed)
    curr_levels = list(
        (curriculum_config or {}).get("levels", [])
    )
    curr_filters = list(
        (curriculum_config or {}).get("filters", [])
    )
    use_curriculum = bool(
        curriculum_config is not None
        and curr_levels
    )
    level_specs = (
        curr_levels
        if use_curriculum
        else [(None, None)]
    )
    levels: list[list[dict[str, Any]]] = []

    for level_idx, bounds in enumerate(level_specs):
        min_d, max_d = bounds
        level_pool: list[dict[str, Any]] = []

        # Get filter for this level
        level_filter = (
            curr_filters[level_idx]
            if level_idx < len(curr_filters)
            else {}
        )
        require_same_side = level_filter.get(
            "same_side", None
        )
        require_path_blocked = level_filter.get(
            "path_blocked", None
        )

        max_attempts = (
            100
            if (
                require_same_side is not None
                or require_path_blocked
                is not None
            )
            else 1
        )

        for item_idx in range(episodes_per_level):
            goal_pose = None
            final_start_pose = None

            for _attempt in range(max_attempts):
                env.reset()
                start_pose = env.get_pose()
                start_pos = start_pose[:3]

                candidate = (
                    env.get_random_surface_point(
                        reference_pos=start_pos,
                        min_dist=min_d,
                        max_dist=max_d,
                        max_attempts=2000,
                        mesh_sample=True,
                    )
                )

                # Check same_side
                same_side = bool(
                    _is_reachable_by_surface(
                        env,
                        start_pos,
                        candidate[:3],
                    )
                )
                if (
                    require_same_side is not None
                    and same_side
                    != require_same_side
                ):
                    continue

                # Check path_blocked
                if (
                    require_path_blocked
                    is not None
                ):
                    env._current_goal = (
                        np.concatenate([
                            candidate[:3],
                            candidate[3:],
                        ])
                    )
                    sensor = env.get_sensor_data()
                    path_blocked = sensor.get(
                        "path_blocked", False
                    )
                    if (
                        path_blocked
                        != require_path_blocked
                    ):
                        continue

                goal_pose = candidate
                final_start_pose = start_pose
                break

            if goal_pose is None:
                goal_pose = candidate
                final_start_pose = start_pose
                logger.warning(
                    "Level %d ep %d: filter not "
                    "satisfied in %d attempts, "
                    "filter=%s",
                    level_idx,
                    item_idx,
                    max_attempts,
                    level_filter,
                )

            start_pos = final_start_pose[:3]

            same_side = bool(
                _is_reachable_by_surface(
                    env,
                    start_pos,
                    goal_pose[:3],
                )
            )
            env._current_goal = np.concatenate([
                goal_pose[:3],
                goal_pose[3:],
            ])
            sensor = env.get_sensor_data()
            path_blocked = bool(
                sensor.get(
                    "path_blocked", False
                )
            )

            target_distance = float(
                np.linalg.norm(
                    goal_pose[:3] - start_pos
                )
            )

            entry: dict[str, Any] = {
                "pool_index": int(item_idx),
                "start_pos": (
                    final_start_pose[:3].tolist()
                ),
                "start_rot": (
                    final_start_pose[3:].tolist()
                ),
                "goal_pos": (
                    goal_pose[:3].tolist()
                ),
                "goal_rot": (
                    goal_pose[3:].tolist()
                ),
                "target_distance": target_distance,
                "same_side": same_side,
                "path_blocked": path_blocked,
            }
            if use_curriculum:
                entry["curriculum_level"] = int(
                    level_idx
                )
                entry["curriculum_bounds"] = [
                    float(min_d), float(max_d)
                ]
            level_pool.append(entry)

        ss_count = sum(
            1
            for e in level_pool
            if e["same_side"]
        )
        pb_count = sum(
            1
            for e in level_pool
            if e.get("path_blocked", False)
        )
        logger.info(
            "[Pools] Level %d: %d episodes, "
            "same_side=%d (%.0f%%), "
            "path_blocked=%d (%.0f%%), "
            "filter=%s",
            level_idx,
            len(level_pool),
            ss_count,
            100.0 * ss_count
            / max(len(level_pool), 1),
            pb_count,
            100.0 * pb_count
            / max(len(level_pool), 1),
            level_filter,
        )

        levels.append(level_pool)

    return {
        "seed": int(seed),
        "episodes_per_level": int(
            episodes_per_level
        ),
        "curriculum_levels": [
            list(map(float, lvl))
            for lvl in curr_levels
        ],
        "levels": levels,
    }

def get_or_generate_pools(
    mesh_path: str,
    seeds: list[int],
    episodes_per_level: int,
    scripts_dir: Path,
    curriculum_levels: list[tuple[float, float]],
    regenerate: bool,
    prefix: str = "train",
    curriculum_filters: list[dict] | None = None,
) -> dict[int, dict[str, Any]]:
    """Get episode pools from cache or generate new ones.

    Args:
        mesh_path: Path to the mesh file.
        seeds: List of random seeds.
        episodes_per_level: Number of episodes per level.
        scripts_dir: Directory for saving/loading pool files.
        curriculum_levels: List of (min_dist, max_dist) tuples.
        regenerate: If True, always regenerate pools.
        prefix: Filename prefix for pool files.

    Returns:
        Dict mapping seed to pool data.
    """
    if not regenerate:
        loaded = _load_pools(
            seeds, scripts_dir, prefix
        )
        if loaded is not None:
            return loaded
        logger.info(
            "[Pools] Some %s files missing — "
            "regenerating...",
            prefix,
        )
    return _generate_and_save_pools(
        mesh_path=mesh_path,
        seeds=seeds,
        episodes_per_level=episodes_per_level,
        scripts_dir=scripts_dir,
        curriculum_levels=curriculum_levels,
        prefix=prefix,
        curriculum_filters=curriculum_filters,
    )


def _generate_and_save_pools(
    mesh_path: str,
    seeds: list[int],
    episodes_per_level: int,
    scripts_dir: Path,
    curriculum_levels: list[tuple[float, float]],
    prefix: str = "train",
    curriculum_filters: list[dict] | None = None,
) -> dict[int, dict[str, Any]]:
    scripts_dir.mkdir(parents=True, exist_ok=True)
    curriculum_config = (
        {
            "levels": curriculum_levels,
            "filters": curriculum_filters or [],
        }
        if curriculum_levels
        else None
    )
    pools: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        logger.info(
            "[Pools] Generating %s pools for seed=%d...", prefix, seed
        )
        pool = generate_episode_pools(
            mesh_path=mesh_path,
            episodes_per_level=episodes_per_level,
            seed=seed,
            curriculum_config=curriculum_config,
        )
        pools[seed] = pool
        pool_file = (
            scripts_dir / f"{prefix}_episode_pools_seed_{seed}.json"
        )
        with pool_file.open("w", encoding="utf-8") as f:
            json.dump(pool, f, indent=2)
        logger.info("[Pools] Saved to %s", pool_file)
    return pools


def _load_pools(
    seeds: list[int],
    scripts_dir: Path,
    prefix: str = "train",
) -> dict[int, dict[str, Any]] | None:
    pools_by_seed: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        pool_file = (
            scripts_dir / f"{prefix}_episode_pools_seed_{seed}.json"
        )
        if not pool_file.exists():
            logger.info("[Pools] File not found: %s", pool_file)
            return None
        with pool_file.open(encoding="utf-8") as f:
            pools_by_seed[seed] = json.load(f)
        level_sizes = [
            len(level)
            for level in pools_by_seed[seed].get("levels", [])
        ]
        logger.info(
            "[Pools] Loaded %s seed=%d levels=%s",
            prefix,
            seed,
            level_sizes,
        )
    return pools_by_seed
