# run_ablation.py
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import trimesh
import logging
import numpy as np
import random
import pickle
import torch

from tbp.hybrid_rl.lightweight_env import LightweightEnv
from tbp.hybrid_rl.ablation_runner import train, GOAL_THRESHOLD_PER_LEVEL
from tbp.hybrid_rl.config import DEFAULT_CONFIG
from tbp.hybrid_rl.behavioral_cloning import BCTrainer
from tbp.hybrid_rl.experience_extractor import ExperienceExtractor
from tbp.hybrid_rl.sac_trainer import PSACTrainer
from tbp.hybrid_rl.rl_goal_approach_controller import RLGoalApproachController
from tbp.hybrid_rl.action_interpreter import ActionInterpreter
from tbp.hybrid_rl.adaptive_manager import AdaptiveTrainingManager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CURRICULUM_LEVELS = [
    (10.0, 40.0),
    (20.0, 80.0),
    (40.0, 120.0),
]

TRAIN_SEEDS = [11]
EVAL_SEEDS = [44]
SAC_EVAL_SEEDS = [77, 88, 99]


def create_tea_cup(
    bottom_radius=25.0,
    top_radius=42.0,
    body_height=60.0,
    wall_thickness=2.0,
    bottom_thickness=2.5,
    handle_radius_major=14.0,
    handle_radius_minor=2.5,
    handle_angle_deg=140.0,
    handle_segments=20,
    body_segments=64,
    circle_points=8,
):
    angles = np.linspace(0, 2 * np.pi, body_segments, endpoint=False)

    bottom_z = -body_height / 2
    top_z = body_height / 2
    bottom_pts = np.column_stack([
        bottom_radius * np.cos(angles),
        bottom_radius * np.sin(angles),
        np.full(body_segments, bottom_z),
    ])
    top_pts = np.column_stack([
        top_radius * np.cos(angles),
        top_radius * np.sin(angles),
        np.full(body_segments, top_z),
    ])
    outer_pts = np.vstack([bottom_pts, top_pts])
    outer = trimesh.convex.convex_hull(outer_pts)

    inner_bottom_radius = bottom_radius - wall_thickness
    inner_top_radius = top_radius - wall_thickness
    inner_bottom_z = bottom_z + bottom_thickness
    inner_top_z = top_z

    inner_bottom_pts = np.column_stack([
        inner_bottom_radius * np.cos(angles),
        inner_bottom_radius * np.sin(angles),
        np.full(body_segments, inner_bottom_z),
    ])
    inner_top_pts = np.column_stack([
        inner_top_radius * np.cos(angles),
        inner_top_radius * np.sin(angles),
        np.full(body_segments, inner_top_z),
    ])
    inner_pts = np.vstack([inner_bottom_pts, inner_top_pts])
    inner = trimesh.convex.convex_hull(inner_pts)

    body = outer.difference(inner)

    handle_angles = np.linspace(
        -np.radians(handle_angle_deg) / 2,
        np.radians(handle_angle_deg) / 2,
        handle_segments,
    )

    handle_center_x = (bottom_radius + top_radius) / 2
    handle_center_z = 0.0

    vertices_all = []
    for i, angle in enumerate(handle_angles):
        center = np.array([
            handle_center_x + handle_radius_major * np.cos(angle),
            0.0,
            handle_center_z + handle_radius_major * np.sin(angle),
        ])

        radial = center - np.array([handle_center_x, 0.0, handle_center_z])
        radial_len = np.linalg.norm(radial)
        if radial_len > 1e-12:
            radial /= radial_len
        else:
            radial = np.array([1.0, 0.0, 0.0])

        tangent = np.array([
            -handle_radius_major * np.sin(angle),
            0.0,
            handle_radius_major * np.cos(angle),
        ])
        tangent /= np.linalg.norm(tangent) + 1e-12

        binormal = np.cross(tangent, radial)
        binormal /= np.linalg.norm(binormal) + 1e-12

        for j in range(circle_points):
            theta = 2.0 * np.pi * j / circle_points
            point = (
                center
                + handle_radius_minor * np.cos(theta) * radial
                + handle_radius_minor * np.sin(theta) * binormal
            )
            vertices_all.append(point)

    vertices_all = np.array(vertices_all)

    faces_all = []
    for i in range(handle_segments - 1):
        for j in range(circle_points):
            j_next = (j + 1) % circle_points
            v0 = i * circle_points + j
            v1 = i * circle_points + j_next
            v2 = (i + 1) * circle_points + j
            v3 = (i + 1) * circle_points + j_next
            faces_all.append([v0, v2, v1])
            faces_all.append([v1, v2, v3])

    faces_all = np.array(faces_all)
    handle = trimesh.Trimesh(vertices=vertices_all, faces=faces_all)
    handle.fix_normals()

    cup = trimesh.util.concatenate([body, handle])
    return cup


def create_mug(
    body_radius=30.0,
    body_height=80.0,
    wall_thickness=3.0,
    bottom_thickness=3.0,
    handle_radius_major=15.0,
    handle_radius_minor=4.0,
    handle_angle_deg=180.0,
    handle_segments=32,
    body_segments=64,
    circle_points=8,
):
    outer = trimesh.primitives.Cylinder(
        radius=body_radius,
        height=body_height,
        sections=body_segments,
    )

    inner_radius = body_radius - wall_thickness
    inner_height = body_height - bottom_thickness
    inner = trimesh.primitives.Cylinder(
        radius=inner_radius,
        height=inner_height,
        sections=body_segments,
    )
    inner_shift = bottom_thickness / 2.0
    inner.apply_translation([0, 0, inner_shift])

    body = outer.difference(inner)

    angles = np.linspace(
        -np.radians(handle_angle_deg) / 2,
        np.radians(handle_angle_deg) / 2,
        handle_segments,
    )

    handle_center_x = body_radius
    handle_center_z = 0.0

    vertices_all = []
    for i, angle in enumerate(angles):
        center = np.array([
            handle_center_x + handle_radius_major * np.cos(angle),
            0.0,
            handle_center_z + handle_radius_major * np.sin(angle),
        ])

        radial = center - np.array([handle_center_x, 0.0, handle_center_z])
        radial_len = np.linalg.norm(radial)
        if radial_len > 1e-12:
            radial /= radial_len
        else:
            radial = np.array([1.0, 0.0, 0.0])

        tangent = np.array([
            -handle_radius_major * np.sin(angle),
            0.0,
            handle_radius_major * np.cos(angle),
        ])
        tangent /= np.linalg.norm(tangent) + 1e-12

        binormal = np.cross(tangent, radial)
        binormal /= np.linalg.norm(binormal) + 1e-12

        for j in range(circle_points):
            theta = 2.0 * np.pi * j / circle_points
            point = (
                center
                + handle_radius_minor * np.cos(theta) * radial
                + handle_radius_minor * np.sin(theta) * binormal
            )
            vertices_all.append(point)

    vertices_all = np.array(vertices_all)

    faces_all = []
    for i in range(handle_segments - 1):
        for j in range(circle_points):
            j_next = (j + 1) % circle_points
            v0 = i * circle_points + j
            v1 = i * circle_points + j_next
            v2 = (i + 1) * circle_points + j
            v3 = (i + 1) * circle_points + j_next
            faces_all.append([v0, v2, v1])
            faces_all.append([v1, v2, v3])

    faces_all = np.array(faces_all)
    handle = trimesh.Trimesh(vertices=vertices_all, faces=faces_all)
    handle.fix_normals()

    mug = trimesh.util.concatenate([body, handle])
    return mug


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
    mug = create_mug()
    mug.export(str(data_dir / "mug.stl"))
    cup = create_tea_cup()
    cup.export(str(data_dir / "cup.stl"))


def _run_eval_per_seed(
    data_dir: Path,
    runs_dir: Path,
    mesh_path: str,
    train_seeds: List[int],
    eval_seeds: List[int],
    variant: str,
    eval_cfg: Dict[str, Any],
    eval_pools: Dict[int, Dict[str, Any]],
    collect_bc: bool = False,
    EPISODES_PER_LEVEL=None,
    mesh_name: str = "",
) -> Tuple[Dict[str, Any], List]:
    results_per_seed = {}
    results_per_level = {}
    bc_transitions = []

    sample_seed = eval_seeds[0]
    num_levels = len(eval_pools[sample_seed].get("levels", []))

    for train_seed, eval_seed in zip(train_seeds, eval_seeds):
        seed_results = {}
        load_dir = str(runs_dir / f"{variant.lower()}_seed_{train_seed}")

        for level_idx in range(num_levels):
            level_pool = eval_pools[eval_seed]["levels"][level_idx]

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
                EPISODES_PER_LEVEL=EPISODES_PER_LEVEL,
            )

            if collect_bc:
                trails = metrics.get("success_trails", [])
                if trails:
                    extractor = ExperienceExtractor(config=eval_cfg, mesh_name=mesh_name)
                    for trail in trails:
                        bc_transitions.extend(extractor.convert_trajectory(trail))

            rates = metrics.get("stats", {}).get("termination_rates", {})
            collision_stats = metrics.get("stats", {}).get("collision_stats", {})
            seed_results[f"level_{level_idx}"] = {
                "success_rate": float(metrics.get("success_rate", 0.0)),
                "timeout_rate": float(rates.get("timeout", 0.0)),
                "collision_rate": float(rates.get("collision_surface_violation", 0.0)),
                "collision_stats": collision_stats,
            }

        results_per_seed[f"train_{train_seed}_eval_{eval_seed}"] = seed_results

        seed_output = data_dir / f"eval_result_{mesh_name}_seed_{train_seed}_{eval_seed}.json"
        with open(seed_output, "w", encoding="utf-8") as f:
            json.dump(seed_results, f, indent=2)
        print(f"\nSaved seed eval to {seed_output}")

        print(f"\n  Seed train={train_seed} eval={eval_seed}:")
        for level_key, level_data in seed_results.items():
            print(
                f"    {level_key}: success={level_data['success_rate']:.4f}, "
                f"timeout={level_data['timeout_rate']:.4f}, "
                f"collision={level_data['collision_rate']:.4f}, "
                f"collision_stats={level_data.get('collision_stats', {})}"
            )

    for level_idx in range(num_levels):
        level_results = []
        for seed_key, seed_data in results_per_seed.items():
            level_key = f"level_{level_idx}"
            if level_key in seed_data:
                level_results.append(seed_data[level_key])

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

    return results_per_level, bc_transitions


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

    EVAL_EPISODES_PER_LEVEL = 500
    REGENERATE_SCRIPTS = True
    RUN_TRAIN = True
    RUN_EVAL = False
    RUN_BC_TRAIN = False
    RUN_SAC_TRAIN = False
    RUN_SAC_EVAL = False
    RUN_ADAPTIVE = False

    TRAINING_STAGES = [
        {"mesh": "mug", "episodes": 5000, "epsilon_start": 0.75, "epsilon_min": 0.08, "is_load": True},
        {"mesh": "cup", "episodes": 5000, "epsilon_start": 0.50, "epsilon_min": 0.05, "is_load": True},
    ]

    base_config = {
        "mode": "train_adapt_epsilon",
        "goal_threshold": GOAL_THRESHOLD_PER_LEVEL[0],
        "max_points": 500_000,
        "k_neighbors": 7,
        "max_steps_per_goal": 400,
        "adaptive_sigma": True,
        "insert_threshold": 0.50,
        "auto_calibrate": False,
        "epsilon_min": 0.05,
        "reward_goal_reached": 60.0,
        "reward_timeout": -12.0,
        "reward_surface_violation": -12.0,
        "reward_step_penalty": -0.5,
        "surface_step": 3.0,
        "free_step": 8.0,
        "free_step_small": 2.0,
        "rotation_step": 5.0,
    }
    cfg = {**DEFAULT_CONFIG, **base_config}

    data_dir = Path(__file__).resolve().parent / "data"
    runs_dir = data_dir / "runs"
    scripts_dir = data_dir / "episode_scripts"
    runs_dir.mkdir(parents=True, exist_ok=True)

    _prepare_demo_meshes(data_dir)

    unified_save_dir = str(runs_dir / "unified_q")

    # ═══════════════════════════════════════════════════════
    # TRAIN
    # ═══════════════════════════════════════════════════════
    if RUN_TRAIN:
        for stage_idx, stage in enumerate(TRAINING_STAGES):
            mesh_name = stage["mesh"]
            mesh_path = str(data_dir / f"{mesh_name}.stl")
            episodes_per_level = stage["episodes"]
            epsilon_start = stage["epsilon_start"]
            is_load = stage["is_load"]

            print("\n" + "=" * 60)
            print(f"STAGE {stage_idx + 1}/{len(TRAINING_STAGES)}: "
                  f"{mesh_name}, episodes={episodes_per_level}, "
                  f"eps={epsilon_start}, is_load={is_load}")
            print("=" * 60 + "\n")

            stage_cfg = {
                **cfg,
                "epsilon_start": epsilon_start,
                "epsilon_min": stage.get("epsilon_min", 0.05),
                "num_episodes": episodes_per_level,
                "unfreeze_normalization": is_load,
            }

            train_pools = _get_or_generate_pools(
                mesh_path=mesh_path,
                seeds=TRAIN_SEEDS,
                episodes_per_level=episodes_per_level,
                scripts_dir=scripts_dir,
                curriculum_levels=CURRICULUM_LEVELS,
                regenerate=REGENERATE_SCRIPTS,
                prefix=f"train_{mesh_name}",
            )

            for seed in TRAIN_SEEDS:
                seed_save_dir = f"{unified_save_dir}_seed_{seed}"

                if is_load:
                    load_dir = seed_save_dir
                else:
                    load_dir = None

                seed_pools = train_pools.get(seed)
                episode_pools = seed_pools.get("levels") if seed_pools else None

                print(f"\n[Stage {stage_idx + 1}] Training {mesh_name}, seed={seed}")

                curriculum_config = {
                    "levels": CURRICULUM_LEVELS,
                    "promote_threshold": 0.55,
                    "promote_window": 100,
                }

                run_result = train(
                    mesh_dir=str(data_dir),
                    save_dir=seed_save_dir,
                    load_dir=load_dir,
                    num_episodes=episodes_per_level,
                    config=stage_cfg,
                    mesh_path=mesh_path,
                    seed=seed,
                    return_metrics=True,
                    curriculum_config=curriculum_config,
                    episode_pools=episode_pools,
                )

                stage_output = data_dir / f"train_result_{mesh_name}_seed_{seed}.json"
                stage_data = {
                    "stage": stage_idx,
                    "mesh": mesh_name,
                    "seed": seed,
                    "epsilon_start": epsilon_start,
                    "is_load": is_load,
                    "goals_reached": run_result.get("goals_reached"),
                    "num_episodes": run_result.get("num_episodes"),
                    "success_rate": run_result.get("success_rate"),
                    "curriculum_stats": run_result.get("curriculum_stats"),
                    "stats": run_result.get("stats"),
                    "collision_stats": run_result.get("stats", {}).get("collision_stats", {}),
                }
                with open(stage_output, "w", encoding="utf-8") as f:
                    json.dump(stage_data, f, indent=2)
                print(f"Saved to {stage_output}")

                stats = run_result.get("stats", {})
                print(
                    f"  success_rate={run_result.get('success_rate', 0):.4f}, "
                    f"collision_rate={stats.get('termination_rates', {}).get('collision_surface_violation', 0):.4f}, "
                    f"collision_stats={stats.get('collision_stats', {})}, "
                    f"action_counts={stats.get('global_action_counts', {})}"
                )

    best_variant = "unified_q"

    # ═══════════════════════════════════════════════════════
    # EVAL (Q-store on all meshes)
    # ═══════════════════════════════════════════════════════
    if RUN_EVAL:
        print("\n" + "=" * 60)
        print("STEP 3: Eval on all meshes (separate eval pools)")
        print("=" * 60)

        EVAL_MESHES = ["cube", "cylinder", "mug", "cup"]

        eval_cfg = {
            "mode": "eval",
            "eval_epsilon": 0.02,
            "goal_threshold": GOAL_THRESHOLD_PER_LEVEL[0],
        }
        eval_overrides = {**cfg, **eval_cfg}

        all_bc_transitions = []
        all_eval_results = {}

        for mesh_name in EVAL_MESHES:
            mesh_path = str(data_dir / f"{mesh_name}.stl")

            print(f"\n{'─' * 40}")
            print(f"Evaluating: {mesh_name}")
            print(f"{'─' * 40}")

            eval_pools = _get_or_generate_pools(
                mesh_path=mesh_path,
                seeds=EVAL_SEEDS,
                episodes_per_level=EVAL_EPISODES_PER_LEVEL,
                scripts_dir=scripts_dir,
                curriculum_levels=CURRICULUM_LEVELS,
                regenerate=REGENERATE_SCRIPTS,
                prefix=f"eval_{mesh_name}",
            )

            eval_results, bc_transitions = _run_eval_per_seed(
                data_dir=data_dir,
                runs_dir=runs_dir,
                mesh_path=mesh_path,
                train_seeds=TRAIN_SEEDS,
                eval_seeds=EVAL_SEEDS,
                variant=best_variant,
                eval_cfg=eval_overrides,
                eval_pools=eval_pools,
                collect_bc=True,
                EPISODES_PER_LEVEL=EVAL_EPISODES_PER_LEVEL,
                mesh_name=mesh_name,
            )

            all_eval_results[mesh_name] = eval_results
            all_bc_transitions.extend(bc_transitions)

            mesh_eval_output = data_dir / f"eval_result_{mesh_name}.json"
            with open(mesh_eval_output, "w", encoding="utf-8") as f:
                json.dump(eval_results, f, indent=2)
            print(f"Saved eval result to {mesh_eval_output}")

            print(f"\n=== Eval Results ({mesh_name}) ===")
            _print_eval_results(eval_results)

        print(f"\n{'=' * 60}")
        print("EVAL SUMMARY — All Meshes")
        print(f"{'=' * 60}")
        print(f"{'Mesh':<12} {'Success':<10} {'Timeout':<10} {'Collision':<10}")
        print("-" * 42)
        for mesh_name in EVAL_MESHES:
            r = all_eval_results[mesh_name].get("overall", {})
            print(
                f"{mesh_name:<12} "
                f"{r.get('mean_success_rate', 0):.4f}    "
                f"{r.get('mean_timeout_rate', 0):.4f}    "
                f"{r.get('mean_collision_rate', 0):.4f}"
            )

        eval_output = data_dir / "eval_result_all.json"
        with open(eval_output, "w", encoding="utf-8") as f:
            json.dump(all_eval_results, f, indent=2)
        print(f"\nSaved all eval results to {eval_output}")

        if all_bc_transitions:
            bc_output = data_dir / "bc_data.pkl"
            with open(bc_output, "wb") as f:
                pickle.dump(all_bc_transitions, f)

            type_names = ExperienceExtractor.get_type_names()
            mesh_id_to_name = {v: k for k, v in ExperienceExtractor.MESH_NAME_TO_ID.items()}
            mesh_counts = {}
            type_counts = {}
            for tr in all_bc_transitions:
                mesh_label = mesh_id_to_name.get(tr.mesh_id, f"unknown_{tr.mesh_id}")
                mesh_counts[mesh_label] = mesh_counts.get(mesh_label, 0) + 1
                name = type_names.get(tr.action_type, f"type_{tr.action_type}")
                type_counts[name] = type_counts.get(name, 0) + 1

            print(f"\nBC data: {len(all_bc_transitions)} transitions")
            print(f"Per mesh: {mesh_counts}")
            print(f"Action distribution: {type_counts}")
            print(f"Saved to {bc_output}")

    # ═══════════════════════════════════════════════════════
    # BC TRAIN
    # ═══════════════════════════════════════════════════════
    if RUN_BC_TRAIN:
        print("\n" + "=" * 60)
        print("STEP 5: BC Training")
        print("=" * 60)

        bc_data_path = data_dir / "bc_data.pkl"
        with open(bc_data_path, "rb") as f:
            bc_transitions = pickle.load(f)

        print(f"Loaded {len(bc_transitions)} BC transitions")

        type_names = ExperienceExtractor.get_type_names()
        mesh_id_to_name = {v: k for k, v in ExperienceExtractor.MESH_NAME_TO_ID.items()}
        mesh_counts = {}
        type_counts = {}
        for tr in bc_transitions:
            mesh_label = mesh_id_to_name.get(tr.mesh_id, f"unknown_{tr.mesh_id}")
            mesh_counts[mesh_label] = mesh_counts.get(mesh_label, 0) + 1
            name = type_names.get(tr.action_type, f"type_{tr.action_type}")
            type_counts[name] = type_counts.get(name, 0) + 1
        print(f"Per mesh: {mesh_counts}")
        print(f"Action distribution: {type_counts}")

        num_action_types = len(type_names)

        trainer = BCTrainer(
            state_dim=cfg.get("state_dim", 15),
            num_types=num_action_types,
            lr=3e-4,
            batch_size=64,
            param_loss_weight=1.0,
            val_split=0.1,
            patience=20,
        )

        trainer.train(bc_transitions, num_epochs=200)

        bc_model_dir = str(runs_dir / "bc_model")
        trainer.save(bc_model_dir)
        print(f"BC model saved to {bc_model_dir}")

        print("\n=== BC Test Predictions ===")
        for i in range(min(10, len(bc_transitions))):
            tr = bc_transitions[i]
            action_type, action_params = trainer.predict(tr.state)
            mesh_label = mesh_id_to_name.get(tr.mesh_id, "?")
            print(
                f"  [{mesh_label}] Expert: {type_names[tr.action_type]} {tr.action_params} | "
                f"Predicted: {type_names[action_type]} {action_params}"
            )

    # ═══════════════════════════════════════════════════════
    # SAC TRAIN (all meshes)
    # ═══════════════════════════════════════════════════════
    if RUN_SAC_TRAIN:
        print("\n" + "=" * 60)
        print("STEP 6: P-SAC Training (all meshes)")
        print("=" * 60)

        SAC_MESHES = ["cube", "cylinder", "mug", "cup"]
        SAC_EPISODES_PER_MESH = {
            "cube": 1000,
            "cylinder": 1500,
            "mug": 2000,
            "cup": 2000,
        }
        SAC_SEED = TRAIN_SEEDS[0]

        num_action_types = len(ExperienceExtractor.get_type_names())

        first_mesh_path = str(data_dir / f"{SAC_MESHES[0]}.stl")
        env = LightweightEnv(first_mesh_path)
        controller = RLGoalApproachController(
            agent_id="sac_state_helper",
            config={**cfg, "mode": "eval"},
        )

        trainer = PSACTrainer(
            state_dim=cfg.get("state_dim", 15),
            num_types=num_action_types,
            max_params=3,
            gamma=0.99,
            tau=0.005,
            lr_actor=1e-5,
            lr_critic=3e-4,
            batch_size=256,
            buffer_capacity=500_000,
            bc_lambda_init=5.0,
            bc_lambda_decay=0.999999,
            max_steps_per_goal=150,
            goal_threshold=cfg.get("goal_threshold", 5.0),
            eval_interval=200,
            eval_episodes=100,
            eval_seed=12345,
        )

        trainer.load_bc(
            bc_model_dir=str(runs_dir / "bc_model"),
            bc_data_path=str(data_dir / "bc_data.pkl"),
        )

        sac_model_dir = str(runs_dir / "sac_model")

        for mesh_name in SAC_MESHES:
            mesh_path = str(data_dir / f"{mesh_name}.stl")
            num_episodes = SAC_EPISODES_PER_MESH[mesh_name]

            print(f"\n{'─' * 40}")
            print(f"SAC Training: {mesh_name}, {num_episodes} episodes")
            print(f"{'─' * 40}")

            env = LightweightEnv(mesh_path)
            controller = RLGoalApproachController(
                agent_id=f"sac_{mesh_name}",
                config={**cfg, "mode": "eval"},
            )

            sac_train_pools = _get_or_generate_pools(
                mesh_path=mesh_path,
                seeds=[SAC_SEED],
                episodes_per_level=num_episodes,
                scripts_dir=scripts_dir,
                curriculum_levels=CURRICULUM_LEVELS,
                regenerate=REGENERATE_SCRIPTS,
                prefix=f"sac_train_{mesh_name}",
            )

            trainer.train(
                env=env,
                controller=controller,
                num_episodes=num_episodes,
                warmup_steps=5000 if mesh_name == SAC_MESHES[0] else 0,
                log_interval=100,
                save_dir=sac_model_dir,
                curriculum_levels=CURRICULUM_LEVELS,
                episode_pools=sac_train_pools[SAC_SEED],
            )

            print(
                f"  {mesh_name} done: "
                f"episodes={trainer.total_episodes}, "
                f"goals={trainer.total_goals_reached}, "
                f"rate={trainer.total_goals_reached / max(trainer.total_episodes, 1):.3f}"
            )

        print(f"\nP-SAC training complete:")
        print(f"  Total episodes: {trainer.total_episodes}")
        print(f"  Total goals reached: {trainer.total_goals_reached}")
        print(f"  Overall success rate: {trainer.total_goals_reached / max(trainer.total_episodes, 1):.3f}")
        print(f"  Saved to {sac_model_dir}")

    # ═══════════════════════════════════════════════════════
    # SAC EVAL (all meshes)
    # ═══════════════════════════════════════════════════════
    if RUN_SAC_EVAL:
        print("\n" + "=" * 60)
        print("STEP 7: P-SAC Eval (all meshes)")
        print("=" * 60)

        SAC_EVAL_MESHES = ["cube", "cylinder", "mug", "cup"]
        num_action_types = len(ExperienceExtractor.get_type_names())

        sac_model_dir = str(runs_dir / "sac_model")
        sac_trainer = PSACTrainer(
            state_dim=cfg.get("state_dim", 15),
            num_types=num_action_types,
        )
        sac_trainer.load(sac_model_dir)

        def _run_sac_eval(sac_trainer, sac_eval_pools, mesh_path, cfg):
            results_per_level = {}
            sample_seed = SAC_EVAL_SEEDS[0]
            num_levels = len(sac_eval_pools[sample_seed].get("levels", []))

            for level_idx in range(num_levels):
                level_successes = 0
                level_timeouts = 0
                level_collisions = 0
                level_total = 0

                for eval_seed in SAC_EVAL_SEEDS:
                    level_pool = sac_eval_pools[eval_seed]["levels"][level_idx]

                    np.random.seed(eval_seed)
                    torch.manual_seed(eval_seed)
                    env = LightweightEnv(mesh_path, seed=eval_seed)
                    controller = RLGoalApproachController(
                        agent_id=f"sac_eval_L{level_idx}_{eval_seed}",
                        config={**cfg, "mode": "eval"},
                    )
                    interpreter = ActionInterpreter(env)

                    for ep_data in level_pool:
                        start_pos = np.array(ep_data["start_pos"])
                        start_rot = np.array(ep_data["start_rot"])
                        env.reset(position=start_pos, rotation=start_rot)
                        goal_pose = np.concatenate([
                            np.array(ep_data["goal_pos"]),
                            np.array(ep_data["goal_rot"]),
                        ])
                        controller.set_new_goal(goal_pose, start_pos)
                        env.set_goal(goal_pose)

                        success = False
                        collision = False

                        for step in range(sac_trainer.max_steps_per_goal):
                            current_pose = env.get_pose()
                            sensor_data = env.get_sensor_data()
                            state_raw = controller._compute_state(
                                current_pose, sensor_data
                            )
                            state = sac_trainer.normalize_state(state_raw)

                            state_t = torch.FloatTensor(
                                state.astype(np.float32)
                            ).unsqueeze(0)
                            with torch.no_grad():
                                at, ap, _, _ = sac_trainer.actor.sample_eval(state_t)
                            action_type = at[0].item()
                            action_params = (
                                ap[0].numpy() * sac_trainer.param_std
                                + sac_trainer.param_mean
                            )

                            sensor_data = interpreter.execute(
                                action_type, action_params
                            )

                            current_pose = env.get_pose()
                            distance = float(np.linalg.norm(
                                goal_pose[:3] - current_pose[:3]
                            ))

                            if distance < sac_trainer.goal_threshold:
                                success = True
                                break

                            depth = sensor_data.get("depth", 100.0)
                            if depth < 0.5:
                                collision = True
                                break

                        level_total += 1
                        if success:
                            level_successes += 1
                        elif collision:
                            level_collisions += 1
                        else:
                            level_timeouts += 1

                count = max(level_total, 1)
                bounds = CURRICULUM_LEVELS[level_idx]
                results_per_level[f"level_{level_idx}"] = {
                    "bounds_mm": list(bounds),
                    "success_rate": level_successes / count,
                    "timeout_rate": level_timeouts / count,
                    "collision_rate": level_collisions / count,
                }

            return results_per_level

        all_sac_eval_results = {}

        for mesh_name in SAC_EVAL_MESHES:
            mesh_path = str(data_dir / f"{mesh_name}.stl")

            print(f"\n{'─' * 40}")
            print(f"SAC Eval: {mesh_name}")
            print(f"{'─' * 40}")

            sac_eval_pools = _get_or_generate_pools(
                mesh_path=mesh_path,
                seeds=SAC_EVAL_SEEDS,
                episodes_per_level=EVAL_EPISODES_PER_LEVEL,
                scripts_dir=scripts_dir,
                curriculum_levels=CURRICULUM_LEVELS,
                regenerate=REGENERATE_SCRIPTS,
                prefix=f"sac_eval_{mesh_name}",
            )

            results = _run_sac_eval(
                sac_trainer, sac_eval_pools, mesh_path, cfg
            )
            all_sac_eval_results[mesh_name] = results

            print(f"\n=== P-SAC Eval Results ({mesh_name}) ===")
            for key in sorted(results.keys()):
                data = results[key]
                print(
                    f"  {key} ({data.get('bounds_mm', [])}mm): "
                    f"success={data['success_rate']:.4f}, "
                    f"timeout={data['timeout_rate']:.4f}, "
                    f"collision={data['collision_rate']:.4f}"
                )

            eval_output = data_dir / f"sac_eval_result_{mesh_name}.json"
            with open(eval_output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Saved to {eval_output}")

        print(f"\n{'=' * 60}")
        print("SAC EVAL SUMMARY — All Meshes")
        print(f"{'=' * 60}")
        print(f"{'Mesh':<12} {'Level':<12} {'Success':<10} {'Timeout':<10} {'Collision':<10}")
        print("-" * 54)
        for mesh_name in SAC_EVAL_MESHES:
            results = all_sac_eval_results[mesh_name]
            for key in sorted(results.keys()):
                data = results[key]
                print(
                    f"{mesh_name:<12} {key:<12} "
                    f"{data['success_rate']:.4f}    "
                    f"{data['timeout_rate']:.4f}    "
                    f"{data['collision_rate']:.4f}"
                )

        eval_output = data_dir / "sac_eval_result_all.json"
        with open(eval_output, "w", encoding="utf-8") as f:
            json.dump(all_sac_eval_results, f, indent=2)
        print(f"\nSaved all SAC eval results to {eval_output}")

    # ═══════════════════════════════════════════════════════
    # ADAPTIVE
    # ═══════════════════════════════════════════════════════
    if RUN_ADAPTIVE:
        print("\n" + "=" * 60)
        print("STEP 10: Adaptive (transfer + online learning)")
        print("=" * 60)

        ADAPTIVE_MESH = "cup"
        ADAPTIVE_EPISODES = 3000
        ADAPTIVE_MIN_DIST = 10.0
        ADAPTIVE_MAX_DIST = 120.0

        adaptive_mesh_path = str(data_dir / f"{ADAPTIVE_MESH}.stl")
        adaptive_env = LightweightEnv(adaptive_mesh_path)

        num_action_types = len(ExperienceExtractor.get_type_names())

        adaptive_q_dir = runs_dir / "adaptive_q"
        if (adaptive_q_dir / "config.json").exists():
            q_load_dir = str(adaptive_q_dir)
            print(f"[Adaptive] Loading Q-store from adaptive: {q_load_dir}")
        else:
            q_load_dir = f"{unified_save_dir}_seed_{TRAIN_SEEDS[0]}"
            print(f"[Adaptive] Loading Q-store from unified (transfer): {q_load_dir}")

        controller = RLGoalApproachController.load(
            q_load_dir,
            agent_id=f"{ADAPTIVE_MESH}_adaptive",
            config={**cfg, "mode": "eval"},
        )

        adaptive_sac_dir = runs_dir / "adaptive_sac"
        if (adaptive_sac_dir / "sac_actor.pt").exists():
            sac_load_dir = str(adaptive_sac_dir)
            print(f"[Adaptive] Loading SAC from adaptive: {sac_load_dir}")
        else:
            sac_load_dir = str(runs_dir / "sac_model")
            print(f"[Adaptive] Loading SAC from training (transfer): {sac_load_dir}")

        sac_trainer = PSACTrainer(
            state_dim=cfg.get("state_dim", 15),
            num_types=num_action_types,
        )
        sac_trainer.load(sac_load_dir)

        manager = AdaptiveTrainingManager(
            controller=controller,
            env=adaptive_env,
            config=cfg,
            runs_dir=str(runs_dir),
            mesh_path=adaptive_mesh_path,
        )
        manager.sac_trainer = sac_trainer

        for episode in range(ADAPTIVE_EPISODES):
            adaptive_env.reset()
            start_pos = adaptive_env.get_pose()[:3]
            goal_pose = adaptive_env.get_random_surface_point(
                reference_pos=start_pos,
                min_dist=ADAPTIVE_MIN_DIST,
                max_dist=ADAPTIVE_MAX_DIST,
                max_attempts=2000,
                mesh_sample=True,
            )
            controller.set_new_goal(goal_pose, start_pos)
            adaptive_env.set_goal(goal_pose)

            goals_before = controller._total_goals_reached
            episode_transitions = []

            for step in range(150):
                current_pose = adaptive_env.get_pose()
                sensor_data = adaptive_env.get_sensor_data()

                state = controller._compute_state(current_pose, sensor_data)
                action_index, source = manager.get_action(state, current_pose, sensor_data)

                state, done = controller.update_only(current_pose, sensor_data, action_index)

                if done:
                    episode_transitions = (
                        controller.success_trails.copy()
                        if controller._total_goals_reached > goals_before
                        else []
                    )
                    break

                adaptive_env.step(action_index, controller.action_space)

            success = controller._total_goals_reached > goals_before
            manager.on_episode_complete(
                success=success,
                transitions=episode_transitions,
            )
            manager.arbitrator.on_episode_end(success)

            if (episode + 1) % 100 == 0:
                stats = manager.get_stats()
                arb_stats = manager.arbitrator.get_stats()
                print(
                    f"\n  Episode {episode+1}: mode={stats['mode']}, "
                    f"success_rate={stats['success_rate']:.3f}, "
                    f"sac_updates={stats['total_sac_updates']}, "
                    f"offline_iters={stats['total_offline_iterations']}, "
                    f"sac_priority={stats['sac_priority_active']}"
                )
                print(
                    f"  Sources: q_store={arb_stats['q_store_rate']:.2f}, "
                    f"q_weak={arb_stats.get('q_store_weak_rate', 0):.2f}, "
                    f"sac={arb_stats['sac_rate']:.2f}, "
                    f"heuristic={arb_stats['heuristic_rate']:.2f}"
                )
                print(
                    f"  Source success: q={arb_stats['q_success_rate']:.2f} "
                    f"({arb_stats['q_episodes']} ep), "
                    f"sac={arb_stats['sac_success_rate']:.2f} "
                    f"({arb_stats['sac_episodes']} ep)"
                )
                print(
                    f"  Agreement: {arb_stats['agreement_rate']:.2f}, "
                    f"q_spread={arb_stats['q_spread_mean']:.2f}, "
                    f"sac_conf={arb_stats['sac_confidence_mean']:.2f}"
                )

        adaptive_q_save = str(runs_dir / "adaptive_q")
        controller.save(adaptive_q_save)
        print(f"[Adaptive] Q-store saved to {adaptive_q_save}")

        if manager.sac_trainer is not None:
            adaptive_sac_save = str(runs_dir / "adaptive_sac")
            manager.sac_trainer.save(adaptive_sac_save)
            print(f"[Adaptive] SAC saved to {adaptive_sac_save}")

        print(f"\n=== {ADAPTIVE_MESH} Adaptive Final Stats ===")
        print(f"  {manager.get_stats()}")
        print(f"  {manager.arbitrator.get_stats()}")


if __name__ == "__main__":
    main()
