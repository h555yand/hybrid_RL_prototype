import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import trimesh
import logging
import numpy as np
import random

from tbp.hybrid_rl.lightweight_env import LightweightEnv
from tbp.hybrid_rl.ablation_runner import RLAblationRunner, train
from tbp.hybrid_rl.config import DEFAULT_CONFIG

logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

CURRICULUM_LEVELS = [
    (10.0, 40.0),
    (20.0, 80.0),
    (40.0, 120.0),
]

TRAIN_SEEDS = [11, 22, 33]
EVAL_SEEDS = [44, 55, 66]


def generate_episode_pools(
    mesh_path: str,
    episodes_per_level: int,
    seed: int,
    curriculum_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    np.random.seed(seed)
    random.seed(seed)
    env = LightweightEnv(mesh_path, seed=seed)
    curr_levels = list((curriculum_config or {}).get("levels", []))
    use_curriculum = bool(curriculum_config is not None and curr_levels)
    level_specs = curr_levels if use_curriculum else [(None, None)]
    levels: List[List[Dict[str, Any]]] = []
    for level_idx, bounds in enumerate(level_specs):
        min_d, max_d = bounds
        level_pool: List[Dict[str, Any]] = []
        for item_idx in range(episodes_per_level):
            env.reset()
            start_pose = env.get_pose()
            start_pos = start_pose[:3]
            if use_curriculum:
                is_cube = "cube" in mesh_path.lower()
                goal_pose = env.get_random_surface_point(
                    reference_pos=start_pos,
                    min_dist=min_d,
                    max_dist=max_d,
                    max_attempts=2000,
                    mesh_sample=True,
                    same_cube_side=True if (is_cube and level_idx == 0) else False,
                )
                curriculum_bounds = [float(min_d), float(max_d)]
            else:
                goal_pose = env.get_random_surface_point()
                curriculum_bounds = None
            target_distance = float(np.linalg.norm(goal_pose[:3] - start_pos))
            entry: Dict[str, Any] = {
                "pool_index": int(item_idx),
                "start_pos": start_pose[:3].tolist(),
                "start_rot": start_pose[3:].tolist(),
                "goal_pos": goal_pose[:3].tolist(),
                "goal_rot": goal_pose[3:].tolist(),
                "target_distance": target_distance,
            }
            if use_curriculum:
                entry["curriculum_level"] = int(level_idx)
                entry["curriculum_bounds"] = curriculum_bounds
            level_pool.append(entry)
        levels.append(level_pool)
    return {
        "seed": int(seed),
        "episodes_per_level": int(episodes_per_level),
        "curriculum_levels": [list(map(float, lvl)) for lvl in curr_levels],
        "levels": levels,
    }


def _generate_and_save_pools(
    mesh_path: str,
    seeds: List[int],
    episodes_per_level: int,
    scripts_dir: Path,
    curriculum_levels: List[tuple],
    prefix: str = "train",
) -> Dict[int, Dict[str, Any]]:
    scripts_dir.mkdir(parents=True, exist_ok=True)
    curriculum_config = {"levels": curriculum_levels} if curriculum_levels else None
    pools = {}
    for seed in seeds:
        print(f"[Pools] Generating {prefix} pools for seed={seed}...")
        pool = generate_episode_pools(
            mesh_path=mesh_path,
            episodes_per_level=episodes_per_level,
            seed=seed,
            curriculum_config=curriculum_config,
        )
        pools[seed] = pool
        pool_file = scripts_dir / f"{prefix}_episode_pools_seed_{seed}.json"
        with open(pool_file, "w", encoding="utf-8") as f:
            json.dump(pool, f, indent=2)
        print(f"[Pools] Saved to {pool_file}")
    return pools


def _load_pools(
    seeds: List[int],
    scripts_dir: Path,
    prefix: str = "train",
) -> Optional[Dict[int, Dict[str, Any]]]:
    pools_by_seed: Dict[int, Dict[str, Any]] = {}
    for seed in seeds:
        pool_file = scripts_dir / f"{prefix}_episode_pools_seed_{seed}.json"
        if not pool_file.exists():
            print(f"[Pools] File not found: {pool_file}")
            return None
        with open(pool_file, encoding="utf-8") as f:
            pools_by_seed[seed] = json.load(f)
        level_sizes = [len(level) for level in pools_by_seed[seed].get("levels", [])]
        print(f"[Pools] Loaded {prefix} seed={seed} levels={level_sizes}")
    return pools_by_seed


def _get_or_generate_pools(
    mesh_path: str,
    seeds: List[int],
    episodes_per_level: int,
    scripts_dir: Path,
    curriculum_levels: List[tuple],
    regenerate: bool,
    prefix: str = "train",
) -> Dict[int, Dict[str, Any]]:
    if not regenerate:
        loaded = _load_pools(seeds, scripts_dir, prefix)
        if loaded is not None:
            return loaded
        print(f"[Pools] Some {prefix} files missing — regenerating...")
    return _generate_and_save_pools(
        mesh_path=mesh_path,
        seeds=seeds,
        episodes_per_level=episodes_per_level,
        scripts_dir=scripts_dir,
        curriculum_levels=curriculum_levels,
        prefix=prefix,
    )


def _prepare_demo_meshes(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    cube = trimesh.primitives.Box(extents=[80, 80, 80])
    cube.export(str(data_dir / "cube.stl"))
    sphere = trimesh.primitives.Sphere(radius=50)
    sphere.export(str(data_dir / "sphere.stl"))
    cylinder = trimesh.primitives.Cylinder(radius=35, height=100)
    cylinder.export(str(data_dir / "cylinder.stl"))


def _run_eval_per_level(
    data_dir: Path,
    runs_dir: Path,
    mesh_path: str,
    train_seeds: List[int],
    eval_seeds: List[int],
    variant: str,
    overrides: Dict[str, Any],
    eval_pools: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    results_per_level = {}
    sample_seed = eval_seeds[0]
    num_levels = len(eval_pools[sample_seed].get("levels", []))
    for level_idx in range(num_levels):
        level_results = []
        for train_seed, eval_seed in zip(train_seeds, eval_seeds):
            level_pool = eval_pools[eval_seed]["levels"][level_idx]
            load_dir = str(runs_dir / f"{variant.lower()}_seed_{train_seed}")
            eval_cfg = {
                **overrides,
                "mode": "eval",
                "eval_epsilon": 0.02,
                "goal_threshold": 2.0,
            }
            metrics = train(
                mesh_dir=str(data_dir),
                save_dir=load_dir,
                num_episodes=len(level_pool),
                config=eval_cfg,
                mesh_path=mesh_path,
                load_dir=load_dir,
                seed=eval_seed,
                return_metrics=True,
                agent_id=f"eval_L{level_idx}_{variant}_t{train_seed}_e{eval_seed}",
                episode_script=level_pool,
            )
            rates = metrics.get("stats", {}).get("termination_rates", {})
            level_results.append({
                "train_seed": train_seed,
                "eval_seed": eval_seed,
                "success_rate": float(metrics.get("success_rate", 0.0)),
                "timeout_rate": float(rates.get("timeout", 0.0)),
                "collision_rate": float(rates.get("collision_surface_violation", 0.0)),
            })
        count = max(len(level_results), 1)
        bounds = CURRICULUM_LEVELS[level_idx] if level_idx < len(CURRICULUM_LEVELS) else (0, 0)
        results_per_level[f"level_{level_idx}"] = {
            "bounds_mm": list(bounds),
            "per_seed": level_results,
            "mean_success_rate": sum(r["success_rate"] for r in level_results) / count,
            "mean_timeout_rate": sum(r["timeout_rate"] for r in level_results) / count,
            "mean_collision_rate": sum(r["collision_rate"] for r in level_results) / count,
        }
    all_success = []
    all_timeout = []
    all_collision = []
    for key, level_data in results_per_level.items():
        if key == "overall":
            continue
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


def _print_eval_results(eval_results: Dict[str, Any]) -> None:
    for key in sorted(eval_results.keys()):
        data = eval_results[key]
        if key == "overall":
            print(
                f"  OVERALL: success={data['mean_success_rate']:.4f}, "
                f"timeout={data['mean_timeout_rate']:.4f}, "
                f"collision={data['mean_collision_rate']:.4f}"
            )
        else:
            bounds = data.get("bounds_mm", [])
            print(
                f"  {key} ({bounds}mm): success={data['mean_success_rate']:.4f}, "
                f"timeout={data['mean_timeout_rate']:.4f}, "
                f"collision={data['mean_collision_rate']:.4f}"
            )


def main() -> None:
    TRAIN_EPISODES_PER_LEVEL = 5_000
    EVAL_EPISODES_PER_LEVEL = 500
    REGENERATE_SCRIPTS = False
    IS_LOAD = False
    RUN_TRAIN = True
    RUN_EVAL = False

    if IS_LOAD:
        epsilon_start = 0.15
    else:
        epsilon_start = 1.0

    base_config = {
        "mode": "train_adapt_epsilon",
        "goal_threshold": 5.0,
        "state_dim": 13,
        "max_points": 500_000,
        "k_neighbors": 7,
        # "max_steps_per_goal": 50, # cube
        "max_steps_per_goal": 150, # cylinder
        "adaptive_sigma": True,
        "insert_threshold": 0.50,
        "auto_calibrate": False,
        "epsilon_start": epsilon_start,
        "epsilon_min": 0.05,
        "reward_goal_reached": 60.0,
        "reward_timeout": -8.0,
        "surface_step": 3.0,
        "free_step": 8.0,
        "rotation_step": 5.0,
    }
    cfg = {**DEFAULT_CONFIG, **base_config}

    data_dir = Path(__file__).resolve().parent / "data"
    runs_dir = data_dir / "runs"
    scripts_dir = data_dir / "episode_scripts"
    runs_dir.mkdir(parents=True, exist_ok=True)

    _prepare_demo_meshes(data_dir)
    # mesh_path = str(data_dir / "cube.stl")
    mesh_path = str(data_dir / "cylinder.stl")

    print("\n" + "=" * 60)
    print("STEP 1: Prepare episode pools (train + eval)")
    print("=" * 60)

    train_pools = _get_or_generate_pools(
        mesh_path=mesh_path,
        seeds=TRAIN_SEEDS,
        episodes_per_level=TRAIN_EPISODES_PER_LEVEL,
        scripts_dir=scripts_dir,
        curriculum_levels=CURRICULUM_LEVELS,
        regenerate=REGENERATE_SCRIPTS,
        prefix="train",
    )
    eval_pools = _get_or_generate_pools(
        mesh_path=mesh_path,
        seeds=EVAL_SEEDS,
        episodes_per_level=EVAL_EPISODES_PER_LEVEL,
        scripts_dir=scripts_dir,
        curriculum_levels=CURRICULUM_LEVELS,
        regenerate=REGENERATE_SCRIPTS,
        prefix="eval",
    )

    print(f"\nTrain pools: {len(TRAIN_SEEDS)} seeds x {TRAIN_EPISODES_PER_LEVEL} episodes/level")
    print(f"Eval pools:  {len(EVAL_SEEDS)} seeds x {EVAL_EPISODES_PER_LEVEL} episodes/level")

    best_variant = "CL3"
    best_overrides = {}

    if RUN_TRAIN:
        print("\n" + "=" * 60)
        print("STEP 2: Train with curriculum")
        print("=" * 60 + "\n")

        runner = RLAblationRunner(
            mesh_dir=str(data_dir),
            mesh_path=mesh_path,
            save_root_dir=str(runs_dir),
            num_episodes=TRAIN_EPISODES_PER_LEVEL,
            base_config=cfg,
            seeds=TRAIN_SEEDS,
            episode_pools_by_seed=train_pools,
            is_load=IS_LOAD,
        )
        variants = runner.curriculum_variants(
            reward_overrides={},
            levels=CURRICULUM_LEVELS,
        )
        result = runner.run(variants=variants, visualise=True)
        best_variant = str(result["best_variant"])
        best_overrides = dict(variants.get(best_variant, {}))

        print("\n=== Train Summaries ===")
        for name, s in result["summaries"].items():
            print(
                f"  {name}: success={s['success_rate']:.4f}, "
                f"timeout={s.get('timeout_rate', 0):.4f}, "
                f"levels_reached={s.get('levels_reached', 0):.1f}"
            )

        train_output = data_dir / "train_result.json"
        result_to_save = {k: v for k, v in result.items() if k != "raw_results"}
        with open(train_output, "w", encoding="utf-8") as f:
            json.dump(result_to_save, f, indent=2)
        print(f"\nSaved train result to {train_output}")

    if RUN_EVAL:
        print("\n" + "=" * 60)
        print("STEP 3: Eval per level (separate eval pools)")
        print("=" * 60)
        
        eval_overrides = {**cfg, **best_overrides}
        eval_results = _run_eval_per_level(
            data_dir=data_dir,
            runs_dir=runs_dir,
            mesh_path=mesh_path,
            train_seeds=TRAIN_SEEDS,
            eval_seeds=EVAL_SEEDS,
            variant=best_variant,
            overrides=eval_overrides,
            eval_pools=eval_pools,
        )

        print(f"\n=== Eval Results (variant={best_variant}) ===")
        _print_eval_results(eval_results)

        eval_output = data_dir / "eval_result.json"
        with open(eval_output, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2)
        print(f"\nSaved eval result to {eval_output}")


if __name__ == "__main__":
    main()