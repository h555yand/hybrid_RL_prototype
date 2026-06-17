import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import trimesh
import logging
import collections
import numpy as np
import random
import glob

from tbp.hybrid_rl.lightweight_env import LightweightEnv
from tbp.hybrid_rl.ablation_runner import RLAblationRunner, train
from tbp.hybrid_rl.config import DEFAULT_CONFIG

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Curriculum levels: (min_dist_mm, max_dist_mm)
# Anchored to the mesh geometry below — all ranges fit within max surface-to-surface
# distances of the generated objects:
#   Cube  80×80×80 mm  → max ~139 mm (space diagonal)
#   Sphere r=50 mm     → max ~100 mm (diameter)
#   Cylinder r=35,h=100→ max ~117 mm (sqrt(70²+100²))
CURRICULUM_LEVELS = [
    (10.0,  40.0),   # Level 0 — easy:   goal is close
    (20.0,  80.0),   # Level 1 — medium
    (40.0, 120.0),   # Level 2 — hard:   near-random distance
]


def generate_episode_pools(
    mesh_path: str,
    episodes_per_level: int,
    seed: int,
    curriculum_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate reproducible episode pools organized by curriculum level."""
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
                goal_pose = env.get_random_surface_point(
                    reference_pos=start_pos,
                    min_dist=min_d,
                    max_dist=max_d,
                    max_attempts=2000,
                    mesh_sample=True,
                    same_cube_side=True if level_idx == 0 else False
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


def _generate_and_save_episode_pools(
    mesh_path: str,
    seeds: List[int],
    episodes_per_level: int,
    scripts_dir: Path,
    curriculum_levels: List[tuple] = None,
) -> Dict[int, Dict[str, Any]]:
    """Generate episode pools for train and eval, save to disk.
    
    Args:
        mesh_path: Path to mesh file.
        seeds: List of seeds for scripts generation.
        episodes_per_level: Number of pre-generated episodes for each level.
        scripts_dir: Directory to save scripts.
        curriculum_levels: Optional curriculum levels.
        
    Returns:
        Dict mapping seed -> episode pool payload.
    """
    scripts_dir.mkdir(parents=True, exist_ok=True)
    
    curriculum_config = None
    if curriculum_levels:
        curriculum_config = {
            "levels": curriculum_levels,
            "promote_threshold": 0.2,
            "promote_window": 50,
        }
    
    train_pools = {}
    
    for seed in seeds:
        print(f"\n[Pools] Generating episode pools for seed={seed}...")
        pools = generate_episode_pools(
            mesh_path=mesh_path,
            episodes_per_level=episodes_per_level,
            seed=seed,
            curriculum_config=curriculum_config,
        )
        train_pools[seed] = pools
        
        pool_file = scripts_dir / f"train_episode_pools_seed_{seed}.json"
        with open(pool_file, "w", encoding="utf-8") as f:
            json.dump(pools, f, indent=2)
        print(f"[Pools] Saved to {pool_file}")
    
    return train_pools


def _load_episode_pools(
    seeds: List[int],
    scripts_dir: Path,
) -> Optional[Dict[int, Dict[str, Any]]]:
    """Load previously saved episode pools from disk.

    Returns None if any required file is missing, so the caller
    knows to regenerate instead.
    """
    pools_by_seed: Dict[int, Dict[str, Any]] = {}
    for seed in seeds:
        pool_file = scripts_dir / f"train_episode_pools_seed_{seed}.json"
        if not pool_file.exists():
            print(f"[Pools] File not found: {pool_file}")
            return None
        with open(pool_file, encoding="utf-8") as f:
            pools_by_seed[seed] = json.load(f)
        level_sizes = [len(level) for level in pools_by_seed[seed].get("levels", [])]
        print(f"[Pools] Loaded seed={seed} levels={level_sizes} from {pool_file}")
    return pools_by_seed


def _get_or_generate_pools(
    mesh_path: str,
    seeds: List[int],
    episodes_per_level: int,
    scripts_dir: Path,
    curriculum_levels: Optional[List[tuple]],
    regenerate: bool,
) -> Dict[int, Dict[str, Any]]:
    """Load episode pools from disk or regenerate them.

    Args:
        regenerate: If True  — always regenerate and overwrite saved files.
                    If False — load from disk; regenerate only if files are missing.
    """
    if not regenerate:
        loaded = _load_episode_pools(seeds, scripts_dir)
        if loaded is not None:
            print(f"[Pools] Loaded {len(seeds)} episode pools from {scripts_dir}")
            return loaded
        print("[Pools] Some files missing — regenerating...")

    return _generate_and_save_episode_pools(
        mesh_path=mesh_path,
        seeds=seeds,
        episodes_per_level=episodes_per_level,
        scripts_dir=scripts_dir,
        curriculum_levels=curriculum_levels,
    )


def _select_script_range(
    script: List[Dict[str, Any]],
    mode: str,
    count: Optional[int] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Select a fixed range from an episode script.

    Modes:
      - "last": use last `count` episodes
      - "first": use first `count` episodes
      - "range": use Python slice [start:end]
    """
    if not script:
        return []

    n = len(script)
    if mode == "last":
        c = n if count is None else max(1, min(int(count), n))
        return script[-c:]

    if mode == "first":
        c = n if count is None else max(1, min(int(count), n))
        return script[:c]

    if mode == "range":
        s = 0 if start is None else max(0, int(start))
        e = n if end is None else max(0, min(int(end), n))
        if e <= s:
            raise ValueError(f"Invalid eval range: start={s}, end={e}, total={n}")
        return script[s:e]

    raise ValueError(f"Unsupported eval range mode: {mode}")


def _build_eval_scripts_from_pools(
    train_pools: Dict[int, Dict[str, Any]],
    seeds: List[int],
    level: int,
    mode: str,
    count: Optional[int] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> Dict[int, List[Dict[str, Any]]]:
    """Create eval scripts by slicing a selected level from saved pools."""
    eval_scripts: Dict[int, List[Dict[str, Any]]] = {}
    for seed in seeds:
        seed_pools = train_pools.get(seed, {})
        levels = seed_pools.get("levels", [])
        if not levels:
            raise ValueError(f"No levels found in episode pools for seed={seed}")
        level_idx = level if level >= 0 else len(levels) + level
        if level_idx < 0 or level_idx >= len(levels):
            raise ValueError(
                f"Invalid level index {level} for seed={seed}; available levels={len(levels)}"
            )
        source = levels[level_idx]
        selected = _select_script_range(
            script=source,
            mode=mode,
            count=count,
            start=start,
            end=end,
        )
        if not selected:
            raise ValueError(f"Eval script for seed={seed} is empty after slicing")
        eval_scripts[seed] = selected
    return eval_scripts


def _prepare_demo_meshes(data_dir: Path) -> None:
    """Create simple meshes for standalone ablation runs.

    Sizes are chosen so that all three curriculum levels fit inside the
    geometry without excessive rejection-sampling fallback:

        Object           Size           Max surface-to-surface dist
        ──────────────── ────────────── ───────────────────────────
        Cube (box)       80×80×80 mm    ~139 mm (space diagonal)
        Sphere           r = 50 mm      ~100 mm (diameter)
        Cylinder         r=35, h=100 mm ~117 mm (sqrt(70²+100²))
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    cube = trimesh.primitives.Box(extents=[80, 80, 80])
    cube.export(str(data_dir / "cube.stl"))

    sphere = trimesh.primitives.Sphere(radius=50)
    sphere.export(str(data_dir / "sphere.stl"))

    cylinder = trimesh.primitives.Cylinder(radius=35, height=100)
    cylinder.export(str(data_dir / "cylinder.stl"))


def _print_summaries(summaries: Dict[str, Dict[str, Any]]) -> None:
    for variant in sorted(summaries.keys()):
        s = summaries[variant]
        print(
            f"{variant}: "
            f"success_rate={s['success_rate']:.4f}, "
            f"timeout_rate={s.get('timeout_rate', 0.0):.4f}, "
            f"collision_surface_violation_rate="
            f"{s.get('collision_surface_violation_rate', 0.0):.4f}, "
            f"levels_reached={s.get('levels_reached', 0.0):.2f}, "
            f"fallback_rate={s.get('fallback_rate', 0.0):.4f}"
            f"update_hit_rate_free={s['update_hit_rate_free']:.4f}, "
            f"active_to_created_ratio_free={s['active_to_created_ratio_free']:.4f}, "
            f"points_per_update_ratio_free={s['points_per_update_ratio_free']:.4f}, "
            f"update_hit_rate_surface={s['update_hit_rate_surface']:.4f}, "
            f"active_to_created_ratio_surface={s['active_to_created_ratio_surface']:.4f}, "
            f"points_per_update_ratio_surface={s['points_per_update_ratio_surface']:.4f}, "
        )


def _print_curriculum_details(result: Dict[str, Any]) -> None:
    """Print per-level curriculum stats averaged across seeds for each variant."""
    raw = result.get("raw_results", {})
    for variant in sorted(raw.keys()):
        runs = raw.get(variant, [])
        per_level: Dict[int, Dict[str, list]] = {}

        for run in runs:
            cs = (run or {}).get("curriculum_stats") or {}
            episodes = cs.get("episodes_per_level", [])
            successes = cs.get("successes_per_level", [])
            rates = cs.get("success_rate_per_level", [])

            for lvl, (ep, suc, rate) in enumerate(zip(episodes, successes, rates)):
                if lvl not in per_level:
                    per_level[lvl] = {"episodes": [], "successes": [], "rates": []}
                per_level[lvl]["episodes"].append(float(ep))
                per_level[lvl]["successes"].append(float(suc))
                per_level[lvl]["rates"].append(float(rate))
                print(
                    f"Details:  level={lvl}: "
                    f"episodes={float(ep):.1f}, "
                    f"successes={float(suc):.1f}, "
                    f"success_rate={float(rate):.3f}"
                )

        if per_level:
            print(f"{variant}:")
            for lvl in sorted(per_level.keys()):
                d = per_level[lvl]
                count = max(len(d["episodes"]), 1)
                print(
                    f"Summary:  level={lvl}: "
                    f"episodes={sum(d['episodes']) / count:.1f}, "
                    f"successes={sum(d['successes']) / count:.1f}, "
                    f"success_rate={sum(d['rates']) / count:.3f}"
                )


def _run_eval_on_same_figure(
    data_dir: Path,
    runs_dir: Path,
    mesh_path: str,
    seeds: List[int],
    best_variant: str,
    best_overrides: Dict[str, Any],
    eval_episodes: int,
    eval_episode_scripts: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    visualise=False
) -> Dict[str, Any]:
    """Run deterministic eval for the best checkpoint on a fixed mesh."""
    per_seed = []

    for seed in seeds:
        episode_script = None
        run_eval_episodes = int(eval_episodes)
        if eval_episode_scripts is not None:
            episode_script = eval_episode_scripts.get(seed)
            if episode_script is None:
                raise ValueError(f"Missing eval script for seed={seed}")
            run_eval_episodes = len(episode_script)

        load_dir = str(runs_dir / f"{best_variant.lower()}_seed_{seed}")
        eval_cfg = {
            **best_overrides,
            "mode": "eval",
            "eval_epsilon": 0.02,
        }
        metrics = train(
            mesh_dir=str(data_dir),
            save_dir=load_dir,
            num_episodes=run_eval_episodes,
            config=eval_cfg,
            mesh_path=mesh_path,
            load_dir=load_dir,
            seed=seed,
            return_metrics=True,
            agent_id=f"eval_{best_variant.lower()}_{seed}",
            episode_script=episode_script,
            visualise=visualise
        )

        rates = metrics.get("stats", {}).get("termination_rates", {})
        per_seed.append(
            {
                "seed": seed,
                "success_rate": float(metrics.get("success_rate", 0.0)),
                "timeout_rate": float(rates.get("timeout", 0.0)),
                "collision_surface_violation_rate": float(
                    rates.get("collision_surface_violation", 0.0)
                ),
            }
        )

    count = max(len(per_seed), 1)
    summary = {
        "mesh_path": mesh_path,
        "eval_episodes": int(eval_episodes),
        "mean_success_rate": float(
            sum(item["success_rate"] for item in per_seed) / count
        ),
        "mean_timeout_rate": float(
            sum(item["timeout_rate"] for item in per_seed) / count
        ),
        "mean_collision_surface_violation_rate": float(
            sum(item["collision_surface_violation_rate"] for item in per_seed)
            / count
        ),
    }

    return {
        "best_variant": best_variant,
        "per_seed": per_seed,
        "summary": summary,
    }


def _assess_training_readiness(
    result: Dict[str, Any],
    success_threshold: float,
    seed_spread_threshold: float,
    timeout_baseline_variant: str,
) -> Dict[str, Any]:
    """Assess whether the model is trained enough on the same figure."""
    post_eval = result.get("post_eval_same_figure", {})
    eval_summary = post_eval.get("summary", {})
    per_seed = post_eval.get("per_seed", [])

    mean_success_rate = float(eval_summary.get("mean_success_rate", 0.0))
    mean_timeout_rate = float(eval_summary.get("mean_timeout_rate", 1.0))

    seed_success_rates = [float(item.get("success_rate", 0.0)) for item in per_seed]
    seed_spread = 0.0
    if seed_success_rates:
        seed_spread = max(seed_success_rates) - min(seed_success_rates)

    summaries = result.get("summaries", {})
    baseline_timeout = None
    if timeout_baseline_variant in summaries:
        baseline_timeout = float(
            summaries[timeout_baseline_variant].get("timeout_rate", 1.0)
        )

    checks = {
        "success_rate_at_least_threshold": mean_success_rate >= success_threshold,
        "seed_spread_within_threshold": seed_spread <= seed_spread_threshold,
        "timeout_better_than_baseline": (
            baseline_timeout is not None and mean_timeout_rate < baseline_timeout
        ),
    }

    passed = all(checks.values())
    return {
        "passed": passed,
        "thresholds": {
            "success_rate_min": success_threshold,
            "seed_spread_max": seed_spread_threshold,
            "timeout_baseline_variant": timeout_baseline_variant,
        },
        "observed": {
            "mean_success_rate": mean_success_rate,
            "mean_timeout_rate": mean_timeout_rate,
            "seed_success_rate_spread": seed_spread,
            "timeout_baseline_rate": baseline_timeout,
        },
        "checks": checks,
    }


def main() -> None:
    # Debug-first setup (edit these values directly in VS Code)
    # SEEDS = [11, 22, 33]
    SEEDS = [11]

    RUN_TRAIN = False
    REGENERATE_SCRIPTS = False  # True  — всегда пересчитывать и перезаписывать
                                # False — грузить с диска; пересчитать только если нет файлов
    IS_LOAD = True  # загружать сохраненный граф состояний
    if not IS_LOAD:
        epsilon_start = 1.0
    else:
        epsilon_start = 0.1
    PRESET = "curriculum"  # "default" | "goal_reward" | "max_steps" | "curriculum"
    NUM_EPISODES = 5000

    RUN_POST_EVAL = True
    EVAL_EPISODES = 1000
    EVAL_SOURCE_LEVEL = 0  # -1 = last curriculum level, 0 = first level, etc.
    EVAL_RANGE_MODE = "last"  # "last" | "first" | "range"
    EVAL_RANGE_COUNT = EVAL_EPISODES
    EVAL_RANGE_START = 500
    EVAL_RANGE_END = 650
    #Как получить последние 150:
    #Оставить:
    #EVAL_RANGE_MODE = "last"
    #EVAL_RANGE_COUNT = 150
    #Другие варианты:
    #Первые 150:
    #EVAL_RANGE_MODE = "first"
    #EVAL_RANGE_COUNT = 150
    #Произвольный срез:
    #EVAL_RANGE_MODE = "range"
    #EVAL_RANGE_START = 850
    #EVAL_RANGE_END = 1000

    # "train"    — всегда обучение
    # "train_adapt_epsilon"    — обучение с адаптивным epsilon
    # "eval"     — всегда inference
    # "auto"     — определяем автоматически
    base_config = {
        "mode": "train_adapt_epsilon",
        "goal_threshold": 5.0, # mm
        "state_dim": 13,
        "max_points": 500_000,
        "k_neighbors": 7,
        "max_steps_per_goal": 50,
        "adaptive_sigma": True,
        "insert_threshold": 0.50,
        "auto_calibrate": False,
        "epsilon_start": epsilon_start,      # initial exploration
        "epsilon_min": 0.05,       # minimum exploration
        "reward_goal_reached": 60.0,
        "reward_timeout": -8.0,
    }
    cfg = {**DEFAULT_CONFIG, **(base_config or {})}

    data_dir = Path(__file__).resolve().parent / "data"
    runs_dir = data_dir / "runs"
    scripts_dir = data_dir / "episode_scripts"
    train_output_json = data_dir / "train_result.json"
    eval_output_json = data_dir / "eval_result.json"

    _prepare_demo_meshes(data_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = str(data_dir / "cube.stl")

    # Load or generate episode pools.
    print("\n" + "="*60)
    action = "Regenerate" if REGENERATE_SCRIPTS else "Load or generate"
    print(f"STEP 1: {action} episode pools for train")
    print("="*60)

    curriculum_levels = CURRICULUM_LEVELS if PRESET == "curriculum" else None
    train_pools = _get_or_generate_pools(
        mesh_path=mesh_path,
        seeds=SEEDS,
        episodes_per_level=NUM_EPISODES,
        scripts_dir=scripts_dir,
        curriculum_levels=curriculum_levels,
        regenerate=REGENERATE_SCRIPTS,
    )

    print(f"[Main] Pools ready ({len(SEEDS)} seeds):")
    for seed in SEEDS:
        level_sizes = [len(level) for level in train_pools[seed].get("levels", [])]
        print(f"  seed={seed}: levels={level_sizes}")

        if NUM_EPISODES != level_sizes[0]:
            logger.warning("Не совпадает кол-во эпизодов!!!")
            return
    
    best_variant = None

    if RUN_TRAIN:
        print("\n" + "="*60)
        print("STEP 2: Train ablation variants with fixed episode pools")
        print("="*60 + "\n")
        
        runner = RLAblationRunner(
            mesh_dir=str(data_dir),
            mesh_path=mesh_path,
            save_root_dir=str(runs_dir),
            num_episodes=NUM_EPISODES,
            base_config=cfg,
            seeds=SEEDS,
            episode_pools_by_seed=train_pools,
            is_load=IS_LOAD
        )

        variants = None
        if PRESET == "goal_reward":
            variants = runner.goal_reward_variants()
        elif PRESET == "max_steps":
            best_overrides = {
                "reward_goal_reached": 60.0,
                # "goal_threshold": 5.0,
                "reward_timeout": -8.0,
            }
            variants = runner.max_steps_variants(reward_overrides=best_overrides)
        elif PRESET == "curriculum":
            best_overrides = {
            }
            variants = runner.curriculum_variants(
                reward_overrides=best_overrides,
                levels=CURRICULUM_LEVELS,
            )
        active_variants = variants or runner.default_variants()

        result = runner.run(variants=active_variants, visualise=False)
        best_variant = str(result["best_variant"])
        # success_trails = result["raw_results"]['CL3'][0]['success_trails']
        # logger.info("SUCCESS_trails", success_trails)

        print("\n=== Ablation Summaries ===")
        _print_summaries(result["summaries"])

        if PRESET == "curriculum":
            print("\n=== Curriculum Per-Level Details ===")
            _print_curriculum_details(result)

        # Add metadata about episode pools
        result["metadata"] = {
            "episode_pools_used": True,
            "pools_dir": str(scripts_dir),
            "seeds": SEEDS,
            "episodes_per_level": NUM_EPISODES,
            "curriculum_preset": PRESET,
        }
        
        if "raw_results" in result:
            del result["raw_results"]
        with open(train_output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=True)
        print(f"\nSaved full result to {train_output_json}")
        
        # Also save pool index for reference
        pool_index = {
            "episodes_per_level": NUM_EPISODES,
            "seeds": SEEDS,
            "pools": {
                seed: scripts_dir / f"train_episode_pools_seed_{seed}.json"
                for seed in SEEDS
            }
        }
        index_file = scripts_dir / "index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump({k: str(v) for k, v in pool_index["pools"].items()}, f, indent=2)
        print(f"Saved pool index to {index_file}")

    if RUN_POST_EVAL:
        print("\n" + "="*60)
        print("STEP 3: Eval using fixed range from train pools")
        print("="*60)

        eval_scripts = _build_eval_scripts_from_pools(
            train_pools=train_pools,
            seeds=SEEDS,
            level=EVAL_SOURCE_LEVEL,
            mode=EVAL_RANGE_MODE,
            count=EVAL_RANGE_COUNT,
            start=EVAL_RANGE_START,
            end=EVAL_RANGE_END,
        )
        print(
            f"[Eval] level={EVAL_SOURCE_LEVEL}, range_mode={EVAL_RANGE_MODE}, "
            f"episodes_per_seed={len(eval_scripts[SEEDS[0]])}"
        )

        if best_variant is None:
            best_variant = "CL3" # name must be real from training as folder with result
            best_overrides = {
            }
        else:
            best_overrides = dict(active_variants.get(best_variant, {}))
        cfg_mode = {"mode": "eval"}
        cfg = {**cfg, **best_overrides, **cfg_mode}
        post_eval = _run_eval_on_same_figure(
            data_dir=data_dir,
            runs_dir=runs_dir,
            mesh_path=mesh_path,
            seeds=SEEDS,
            best_variant=best_variant,
            best_overrides=cfg,
            eval_episodes=EVAL_EPISODES,
            eval_episode_scripts=eval_scripts,
            visualise=False
        )

        eval = post_eval["summary"]
        print("\n=== Post Eval (Same Figure) ===")
        print(
            f"best={best_variant}, "
            f"success_rate={eval['mean_success_rate']:.4f}, "
            f"timeout_rate={eval['mean_timeout_rate']:.4f}, "
            f"collision_surface_violation_rate="
            f"{eval['mean_collision_surface_violation_rate']:.4f}"
        )

        with open(eval_output_json, "w", encoding="utf-8") as f:
            json.dump(eval, f, indent=2, ensure_ascii=True)
        print(f"\nSaved full result to {eval_output_json}")


if __name__ == "__main__":
    main()
