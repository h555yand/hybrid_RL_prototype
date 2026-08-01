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

from tbp.hybrid_rl.lightweight_env import LightweightEnv

logger = logging.getLogger(__name__)


def _is_reachable_by_surface(
    env: LightweightEnv,
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
) -> bool:
    """Check if start and goal are on the same side of the object.

    Uses surface normal directions relative to object centroid.
    Both normals pointing outward = external side.
    Both normals pointing inward = internal side.
    Mixed = different sides, path blocked by wall.

    Args:
        env: Environment with mesh.
        start_pos: Start position [x, y, z].
        goal_pos: Goal position [x, y, z].

    Returns:
        True if both points are on the same side.
    """
    center = np.array(env.mesh.centroid, dtype=float)
    height_axis = env.height_axis

    # Get normals at nearest surface points
    _, _, start_face = env.mesh.nearest.on_surface([start_pos])
    _, _, goal_face = env.mesh.nearest.on_surface([goal_pos])

    start_normal = env.mesh.face_normals[start_face[0]]
    goal_normal = env.mesh.face_normals[goal_face[0]]

    # Horizontal components (ignore height axis)
    start_n = start_normal.copy()
    start_n[height_axis] = 0.0

    goal_n = goal_normal.copy()
    goal_n[height_axis] = 0.0

    # If either normal is mostly vertical (top/bottom surface) — allow
    if np.linalg.norm(start_n) < 0.3 or np.linalg.norm(goal_n) < 0.3:
        return True

    # Direction from center to each point (horizontal)
    start_from_center = start_pos - center
    start_from_center[height_axis] = 0.0

    goal_from_center = goal_pos - center
    goal_from_center[height_axis] = 0.0

    # Outward = normal points same direction as center→point
    start_outward = np.dot(start_n, start_from_center) > 0
    goal_outward = np.dot(goal_n, goal_from_center) > 0

    return start_outward == goal_outward


def generate_episode_pools(
    mesh_path: str,
    episodes_per_level: int,
    seed: int,
    curriculum_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate episode pools with fixed start/goal positions.

    For Level 0, ensures start and goal are on the same side
    of the object (reachable by surface crawling without detach).

    Args:
        mesh_path: Path to the mesh file.
        episodes_per_level: Number of episodes per curriculum level.
        seed: Random seed for reproducibility.
        curriculum_config: Optional curriculum config with "levels" key.

    Returns:
        Dict with seed, episodes_per_level, curriculum_levels, and levels.
    """
    np.random.seed(seed)  # noqa: NPY002
    random.seed(seed)
    env = LightweightEnv(mesh_path, seed=seed)
    curr_levels = list(
        (curriculum_config or {}).get("levels", [])
    )
    use_curriculum = bool(
        curriculum_config is not None and curr_levels
    )
    level_specs = (
        curr_levels if use_curriculum else [(None, None)]
    )
    levels: list[list[dict[str, Any]]] = []

    for level_idx, bounds in enumerate(level_specs):
        min_d, max_d = bounds
        level_pool: list[dict[str, Any]] = []
        for item_idx in range(episodes_per_level):
            env.reset()
            start_pose = env.get_pose()
            start_pos = start_pose[:3]

            is_cube = "cube" in mesh_path.lower()
            require_same_side = (
                use_curriculum and level_idx == 0
            )
            require_same_side = False # делает еще сложнее, много примеров когда что-то на дне, надо его тоже исключать

            if use_curriculum:
                max_goal_attempts = (
                    50 if require_same_side else 1
                )
                goal_pose = None

                for _attempt in range(max_goal_attempts):
                    candidate = env.get_random_surface_point(
                        reference_pos=start_pos,
                        min_dist=min_d,
                        max_dist=max_d,
                        max_attempts=2000,
                        mesh_sample=True,
                        same_cube_side=bool(
                            is_cube and level_idx == 0
                        ),
                    )

                    if not require_same_side:
                        goal_pose = candidate
                        break

                    if _is_reachable_by_surface(
                        env, start_pos, candidate[:3]
                    ):
                        goal_pose = candidate
                        break

                if goal_pose is None:
                    goal_pose = candidate
                    logger.warning(
                        "Level %d ep %d: could not find "
                        "same-side goal in %d attempts, "
                        "using fallback",
                        level_idx,
                        item_idx,
                        max_goal_attempts,
                    )

                curriculum_bounds = [
                    float(min_d), float(max_d)
                ]
            else:
                goal_pose = env.get_random_surface_point()
                curriculum_bounds = None

            target_distance = float(
                np.linalg.norm(goal_pose[:3] - start_pos)
            )
            same_side = bool(_is_reachable_by_surface(
                env, start_pos, goal_pose[:3]
            ))

            entry: dict[str, Any] = {
                "pool_index": int(item_idx),
                "start_pos": start_pose[:3].tolist(),
                "start_rot": start_pose[3:].tolist(),
                "goal_pos": goal_pose[:3].tolist(),
                "goal_rot": goal_pose[3:].tolist(),
                "target_distance": target_distance,
                "same_side": same_side,
            }
            if use_curriculum:
                entry["curriculum_level"] = int(level_idx)
                entry["curriculum_bounds"] = curriculum_bounds
            level_pool.append(entry)
        levels.append(level_pool)

    return {
        "seed": int(seed),
        "episodes_per_level": int(episodes_per_level),
        "curriculum_levels": [
            list(map(float, lvl)) for lvl in curr_levels
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
        loaded = _load_pools(seeds, scripts_dir, prefix)
        if loaded is not None:
            return loaded
        logger.info(
            "[Pools] Some %s files missing — regenerating...", prefix
        )
    return _generate_and_save_pools(
        mesh_path=mesh_path,
        seeds=seeds,
        episodes_per_level=episodes_per_level,
        scripts_dir=scripts_dir,
        curriculum_levels=curriculum_levels,
        prefix=prefix,
    )


def _generate_and_save_pools(
    mesh_path: str,
    seeds: list[int],
    episodes_per_level: int,
    scripts_dir: Path,
    curriculum_levels: list[tuple[float, float]],
    prefix: str = "train",
) -> dict[int, dict[str, Any]]:
    scripts_dir.mkdir(parents=True, exist_ok=True)
    curriculum_config = (
        {"levels": curriculum_levels} if curriculum_levels else None
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
