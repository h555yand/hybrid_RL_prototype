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
from tbp.hybrid_rl.ablation_runner import RLAblationRunner, train, GOAL_THRESHOLD_PER_LEVEL
from tbp.hybrid_rl.config import DEFAULT_CONFIG
from tbp.hybrid_rl.behavioral_cloning import BCTrainer
from tbp.hybrid_rl.experience_extractor import ExperienceExtractor
from tbp.hybrid_rl.replay_buffer import ReplayBuffer
from tbp.hybrid_rl.sac_trainer import PSACTrainer
from tbp.hybrid_rl.rl_goal_approach_controller import RLGoalApproachController
from tbp.hybrid_rl.action_interpreter import ActionInterpreter
from tbp.hybrid_rl.adaptive_manager import AdaptiveTrainingManager



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
    # Внешний усечённый конус через convex_hull двух цилиндров
    angles = np.linspace(0, 2 * np.pi, body_segments, endpoint=False)

    # Нижнее кольцо
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

    # Внутренний усечённый конус (полость)
    inner_bottom_radius = bottom_radius - wall_thickness
    inner_top_radius = top_radius - wall_thickness
    inner_height = body_height - bottom_thickness
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

    # Ручка
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
    EPISODES_PER_LEVEL=None
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
                EPISODES_PER_LEVEL=EPISODES_PER_LEVEL
            )

            if collect_bc:
                trails = metrics.get("success_trails", [])
                if trails:
                    extractor = ExperienceExtractor(config=eval_cfg)
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

        seed_output = data_dir / f"eval_result_seed_{train_seed}_{eval_seed}.json"
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

    eval_output = data_dir / "eval_result.json"
    with open(eval_output, "w", encoding="utf-8") as f:
        json.dump({"per_seed": results_per_seed, "per_level": results_per_level}, f, indent=2)
    print(f"\nSaved eval result to {eval_output}")

    return results_per_level, bc_transitions

def _run_eval_per_level(
    data_dir: Path,
    runs_dir: Path,
    mesh_path: str,
    train_seeds: List[int],
    eval_seeds: List[int],
    variant: str,
    eval_cfg: Dict[str, Any],
    eval_pools: Dict[int, Dict[str, Any]],
    collect_bc: bool = False,
    EPISODES_PER_LEVEL=None
) -> Tuple[Dict[str, Any], List]:
    results_per_level = {}
    bc_transitions = []

    sample_seed = eval_seeds[0]
    num_levels = len(eval_pools[sample_seed].get("levels", []))

    for level_idx in range(num_levels):
        level_results = []

        for train_seed, eval_seed in zip(train_seeds, eval_seeds):
            level_pool = eval_pools[eval_seed]["levels"][level_idx]
            load_dir = str(runs_dir / f"{variant.lower()}_seed_{train_seed}")

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
                EPISODES_PER_LEVEL=EPISODES_PER_LEVEL
            )

            if collect_bc:
                trails = metrics.get("success_trails", [])
                if trails:
                    extractor = ExperienceExtractor(config=eval_cfg)
                    for trail in trails:
                        bc_transitions.extend(extractor.convert_trajectory(trail))

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

        eval_output = data_dir / f"eval_result_{level_idx}.json"
        with open(eval_output, "w", encoding="utf-8") as f:
            json.dump(results_per_level, f, indent=2)
        print(f"\nSaved eval result to {eval_output}")

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

    TRAIN_EPISODES_PER_LEVEL = 5_000
    EVAL_EPISODES_PER_LEVEL = 500
    REGENERATE_SCRIPTS = False
    IS_LOAD = True
    RUN_TRAIN = False
    RUN_EVAL = False
    RUN_BC_TRAIN = False
    RUN_SAC_TRAIN = False
    RUN_SAC_EVAL = False
    RUN_ADAPTIVE = True
    RUN_CUP_EVAL = False


    if IS_LOAD:
        epsilon_start = 0.15
    else:
        epsilon_start = 1.0

    base_config = {
        "mode": "train_adapt_epsilon",
        "goal_threshold": GOAL_THRESHOLD_PER_LEVEL[0],
        "max_points": 500_000,
        "k_neighbors": 7,
        "max_steps_per_goal": 150,
        "adaptive_sigma": True,
        "insert_threshold": 0.50,
        "auto_calibrate": False,
        "epsilon_start": epsilon_start,
        "epsilon_min": 0.05,
        "reward_goal_reached": 60.0,
        "reward_timeout": -8.0,
        "surface_step": 3.0,
        "free_step": 5.0,
        "rotation_step": 5.0,
    }
    cfg = {**DEFAULT_CONFIG, **base_config}

    data_dir = Path(__file__).resolve().parent / "data"
    runs_dir = data_dir / "runs"
    scripts_dir = data_dir / "episode_scripts"
    runs_dir.mkdir(parents=True, exist_ok=True)

    #####################################################
    # Choose Mesh
    _prepare_demo_meshes(data_dir)
    # mesh_path = str(data_dir / "cube.stl")
    # mesh_path = str(data_dir / "cylinder.stl")
    # mesh_path = str(data_dir / "mug.stl")
    mesh_path = str(data_dir / "cup.stl")

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
    sac_eval_pools = _get_or_generate_pools(
        mesh_path=mesh_path,
        seeds=SAC_EVAL_SEEDS,
        episodes_per_level=EVAL_EPISODES_PER_LEVEL,
        scripts_dir=scripts_dir,
        curriculum_levels=CURRICULUM_LEVELS,
        regenerate=REGENERATE_SCRIPTS,
        prefix="sac_eval",
    )

    print(f"\nTrain pools: {len(TRAIN_SEEDS)} seeds x {TRAIN_EPISODES_PER_LEVEL} episodes/level")
    print(f"Eval pools:  {len(EVAL_SEEDS)} seeds x {EVAL_EPISODES_PER_LEVEL} episodes/level")

    best_variant = "CL3"
    
    def _run_sac_eval(mode, sac_trainer, sac_eval_pools, mesh_path, cfg):
        results_per_level = {}
        sample_seed = SAC_EVAL_SEEDS[0]
        num_levels = len(sac_eval_pools[sample_seed].get("levels", []))

        for level_idx in range(num_levels):
            level_successes = 0
            level_timeouts = 0
            level_collisions = 0
            level_total = 0

            for eval_seed in [SAC_EVAL_SEEDS[0]]:
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

                        if mode == "sample":
                            state_t = torch.FloatTensor(
                                state.astype(np.float32)
                            ).unsqueeze(0)
                            with torch.no_grad():
                                at, ap, _, _ = sac_trainer.actor.sample(state_t)
                            action_type = at[0].item()
                            action_params = (
                                ap[0].numpy() * sac_trainer.param_std
                                + sac_trainer.param_mean
                            )
                        else:
                            action_type, action_params_norm = sac_trainer.actor.predict(
                                state.astype(np.float32)
                            )
                            param_dim = len(action_params_norm)
                            action_params = (
                                action_params_norm
                                * sac_trainer.param_std[:param_dim]
                                + sac_trainer.param_mean[:param_dim]
                            )

                        sensor_data = interpreter.execute(
                            action_type, action_params
                        )

                        current_pose = env.get_pose()
                        distance = float(np.linalg.norm(
                            goal_pose[:3] - current_pose[:3]
                        ))

                        if step < 3 and level_total < 5:
                            type_names = ExperienceExtractor.get_type_names()
                            print(
                                f"  ep={level_total} step={step}: "
                                f"type={type_names.get(action_type, action_type)}, "
                                f"params={action_params}, "
                                f"dist={distance:.1f}"
                            )

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
            logger.info(results_per_level[f"level_{level_idx}"])

        return results_per_level


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

        eval_cfg = {
            "mode": "eval",
            "eval_epsilon": 0.02,
            "goal_threshold": GOAL_THRESHOLD_PER_LEVEL[0]
        }
        eval_overrides = {**cfg, **eval_cfg}
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
            EPISODES_PER_LEVEL=EVAL_EPISODES_PER_LEVEL
        )

        print(f"\n=== Eval Results (variant={best_variant}) ===")
        _print_eval_results(eval_results)

        eval_output = data_dir / "eval_result.json"
        with open(eval_output, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2)
        print(f"\nSaved eval result to {eval_output}")

        if bc_transitions:
            bc_output = data_dir / "bc_data.pkl"
            with open(bc_output, "wb") as f:
                pickle.dump(bc_transitions, f)
            type_counts = {}
            for tr in bc_transitions:
                name = ExperienceExtractor.get_type_names()[tr.action_type]
                type_counts[name] = type_counts.get(name, 0) + 1
            print(f"\nBC data: {len(bc_transitions)} transitions")
            print(f"Action distribution: {type_counts}")
            print(f"Saved to {bc_output}")

            with open(data_dir / "bc_data.pkl", "rb") as f:
                bc_transitions = pickle.load(f)

            buffer = ReplayBuffer(
                capacity=100_000,
                state_dim=cfg.get("state_dim", 15),
                max_params=3,
            )

            buffer.load_bc_data(bc_transitions)

            batch = buffer.sample(batch_size=64)
            print(f"Batch states shape: {batch['states'].shape}")
            print(f"Batch types: {batch['action_types'][:5]}")
            print(f"Batch params: {batch['action_params'][:5]}")

    if RUN_BC_TRAIN:
        print("\n" + "=" * 60)
        print("STEP 5: BC Training")
        print("=" * 60)

        to_join = True
        if to_join:
            with open(data_dir / "bc_data.pkl", "rb") as f:
                bc_1 = pickle.load(f)

            with open(data_dir / "bc_data_002.pkl", "rb") as f:
                bc_2 = pickle.load(f)

            bc_combined = bc_1 + bc_2

            with open(data_dir / "bc_data.pkl", "wb") as f:
                pickle.dump(bc_combined, f)

            print(f"File 1: {len(bc_1)}, File 2: {len(bc_2)}, Combined: {len(bc_combined)}")

        bc_data_path = data_dir / "bc_data.pkl"
        with open(bc_data_path, "rb") as f:
            bc_transitions = pickle.load(f)

        print(f"Loaded {len(bc_transitions)} BC transitions")

        trainer = BCTrainer(
            state_dim=cfg.get("state_dim", 15),
            num_types=8,
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
        type_names = ExperienceExtractor.get_type_names()
        for i in range(min(5, len(bc_transitions))):
            tr = bc_transitions[i]
            action_type, action_params = trainer.predict(tr.state)
            print(
                f"  Expert: {type_names[tr.action_type]} {tr.action_params} | "
                f"Predicted: {type_names[action_type]} {action_params}"
            )

    if RUN_SAC_TRAIN:
        print("\n" + "=" * 60)
        print("STEP 6: P-SAC Training")
        print("=" * 60)

        env = LightweightEnv(mesh_path)
        controller = RLGoalApproachController(
            agent_id="sac_state_helper",
            config={**cfg, "mode": "eval"},
        )

        trainer = PSACTrainer(
            state_dim=cfg.get("state_dim", 15),
            num_types=8,
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

        trainer.train(
            env=env,
            controller=controller,
            num_episodes=2000,
            warmup_steps=5000,
            log_interval=100,
            save_dir=sac_model_dir,
            curriculum_levels=CURRICULUM_LEVELS,
            episode_pools=train_pools[TRAIN_SEEDS[1]],
        )

        print(f"\nP-SAC training complete:")
        print(f"  Episodes: {trainer.total_episodes}")
        print(f"  Goals reached: {trainer.total_goals_reached}")
        print(f"  Success rate: {trainer.total_goals_reached / max(trainer.total_episodes, 1):.3f}")
        print(f"  Saved to {sac_model_dir}")

    if RUN_SAC_EVAL:
        print("\n" + "=" * 60)
        print("STEP 7: P-SAC Eval")
        print("=" * 60)

        SAC_EVAL_MODE = "both"

        sac_model_dir = str(runs_dir / "sac_model")
        sac_trainer = PSACTrainer(state_dim=cfg.get("state_dim", 15))
        sac_trainer.load(sac_model_dir)

        modes = ["sample", "predict"] if SAC_EVAL_MODE == "both" else [SAC_EVAL_MODE]

        for mode in modes:
            print(f"\n--- Eval mode: {mode} ---")
            results_per_level = _run_sac_eval(
                mode, sac_trainer, sac_eval_pools, mesh_path, cfg
            )

            print(f"\n=== P-SAC Eval Results (mode={mode}) ===")
            for key in sorted(results_per_level.keys()):
                data = results_per_level[key]
                print(
                    f"  {key} ({data.get('bounds_mm', [])}mm): "
                    f"success={data['success_rate']:.4f}, "
                    f"timeout={data['timeout_rate']:.4f}, "
                    f"collision={data['collision_rate']:.4f}"
                )

            eval_output = data_dir / f"sac_eval_result_{mode}.json"
            with open(eval_output, "w", encoding="utf-8") as f:
                json.dump(results_per_level, f, indent=2)
            print(f"Saved to {eval_output}")

    if RUN_ADAPTIVE:
        print("\n" + "=" * 60)
        print("STEP 8: Adaptive Training")
        print("=" * 60)

        env = LightweightEnv(mesh_path)
        controller = RLGoalApproachController.load(
            str(runs_dir / f"{best_variant.lower()}_seed_{TRAIN_SEEDS[0]}"),
            agent_id="adaptive",
            config={**cfg, "mode": "eval"},
        )

        sac_trainer = PSACTrainer(state_dim=cfg.get("state_dim", 15))
        sac_trainer.load(str(runs_dir / "sac_model"))

        manager = AdaptiveTrainingManager(
            controller=controller,
            env=env,
            config=cfg,
            runs_dir=str(runs_dir),
            mesh_path=mesh_path,
            offline_threshold=0.60,
            online_sac_update_every=200,
            online_sac_update_steps=50,
            online_bc_update_every=2000,
        )

        manager.sac_trainer = sac_trainer

        for episode in range(1000):
            env.reset()
            start_pos = env.get_pose()[:3]
            goal_pose = env.get_random_surface_point(
                reference_pos=start_pos,
                min_dist=10.0,
                max_dist=120.0,
                max_attempts=2000,
                mesh_sample=True,
            )
            controller.set_new_goal(goal_pose, start_pos)
            env.set_goal(goal_pose)

            goals_before = controller._total_goals_reached
            episode_transitions = []

            for step in range(150):
                current_pose = env.get_pose()
                sensor_data = env.get_sensor_data()

                state = controller._compute_state(current_pose, sensor_data)
                action_index, source = manager.get_action(state, current_pose, sensor_data)

                state, done = controller.update_only(current_pose, sensor_data, action_index)

                if done:
                    episode_transitions = controller.success_trails.copy() if controller._total_goals_reached > goals_before else []
                    break

                env.step(action_index, controller.action_space)
                if action_index in (controller.action_space.IDX_DETACH, controller.action_space.IDX_DETACH_EDGE):
                    controller._last_detach_sub_steps = getattr(env, '_last_detach_sub_steps', 1)

            success = controller._total_goals_reached > goals_before
            manager.on_episode_complete(
                success=success,
                transitions=episode_transitions,
            )

            if (episode + 1) % 100 == 0:
                stats = manager.get_stats()
                arb_stats = manager.arbitrator.get_stats()
                print(
                    f"\n  Episode {episode+1}: mode={stats['mode']}, "
                    f"success_rate={stats['success_rate']:.3f}"
                )
                print(
                    f"  Sources: q_store={arb_stats['q_store_rate']:.2f}, "
                    f"q_weak={arb_stats['q_store_weak_rate']:.2f}, "
                    f"sac={arb_stats['sac_rate']:.2f}, "
                    f"heuristic={arb_stats['heuristic_rate']:.2f}"
                )
                print(
                    f"  Agreement Q↔SAC: {arb_stats['agreement_rate']:.2f}, "
                    f"q_conf={arb_stats['q_confidence_mean']:.2f}, "
                    f"q_spread={arb_stats['q_spread_mean']:.2f}, "
                    f"sac_conf={arb_stats['sac_confidence_mean']:.2f}"
                )
                print(f"  Q proposed:  {arb_stats['q_proposed_top']}")
                print(f"  SAC proposed: {arb_stats['sac_proposed_top']}")
                print(f"  Q chosen:    {arb_stats['q_chosen_top']}")
                print(f"  SAC chosen:   {arb_stats['sac_chosen_top']}")

        print(f"\n=== Adaptive Final Stats ===")
        print(f"  {manager.get_stats()}")
        print(f"  {manager.arbitrator.get_stats()}")

    if RUN_CUP_EVAL:
        print("\n" + "=" * 60)
        print("STEP 9: Transfer Eval — Cup")
        print("=" * 60)

        cup_mesh_path = str(data_dir / "cup.stl")

        # Генерируем eval пулы для чашки
        cup_eval_pools = _get_or_generate_pools(
            mesh_path=cup_mesh_path,
            seeds=SAC_EVAL_SEEDS,
            episodes_per_level=EVAL_EPISODES_PER_LEVEL,
            scripts_dir=scripts_dir,
            curriculum_levels=CURRICULUM_LEVELS,
            regenerate=REGENERATE_SCRIPTS,
            prefix="cup_eval",
        )

        # ─── 9A: SAC zero-shot ───
        print("\n--- 9A: SAC zero-shot on Cup ---")

        sac_model_dir = str(runs_dir / "sac_model")
        sac_trainer = PSACTrainer(state_dim=cfg.get("state_dim", 15))
        sac_trainer.load(sac_model_dir)

        sac_results = _run_sac_eval(
            mode="sample",
            sac_trainer=sac_trainer,
            sac_eval_pools=cup_eval_pools,
            mesh_path=cup_mesh_path,
            cfg=cfg,
        )

        print(f"\n=== Cup SAC zero-shot ===")
        for key in sorted(sac_results.keys()):
            data = sac_results[key]
            print(
                f"  {key} ({data.get('bounds_mm', [])}mm): "
                f"success={data['success_rate']:.4f}, "
                f"timeout={data['timeout_rate']:.4f}, "
                f"collision={data['collision_rate']:.4f}"
            )

        eval_output = data_dir / "cup_sac_eval_result.json"
        with open(eval_output, "w", encoding="utf-8") as f:
            json.dump(sac_results, f, indent=2)
        print(f"Saved to {eval_output}")

        # ─── 9B: Арбитраж zero-shot ───
        print("\n--- 9B: Arbitrage zero-shot on Cup ---")

        cup_env = LightweightEnv(cup_mesh_path)
        controller = RLGoalApproachController.load(
            str(runs_dir / f"{best_variant.lower()}_seed_{TRAIN_SEEDS[0]}"),
            agent_id="cup_eval_arb",
            config={**cfg, "mode": "eval"},
        )

        sac_trainer_arb = PSACTrainer(state_dim=cfg.get("state_dim", 15))
        sac_trainer_arb.load(sac_model_dir)

        manager = AdaptiveTrainingManager(
            controller=controller,
            env=cup_env,
            config=cfg,
            runs_dir=str(runs_dir),
            mesh_path=cup_mesh_path,
            offline_threshold=0.0,
            online_sac_update_every=999999,  # отключить online update
            online_bc_update_every=999999,
        )
        manager.sac_trainer = sac_trainer_arb

        arb_results_per_level = {}
        sample_seed = SAC_EVAL_SEEDS[0]
        num_levels = len(cup_eval_pools[sample_seed].get("levels", []))

        for level_idx in range(num_levels):
            level_successes = 0
            level_timeouts = 0
            level_collisions = 0
            level_total = 0

            for eval_seed in [SAC_EVAL_SEEDS[0]]:
                level_pool = cup_eval_pools[eval_seed]["levels"][level_idx]

                np.random.seed(eval_seed)
                torch.manual_seed(eval_seed)
                cup_env = LightweightEnv(cup_mesh_path, seed=eval_seed)

                controller_eval = RLGoalApproachController.load(
                    str(runs_dir / f"{best_variant.lower()}_seed_{TRAIN_SEEDS[0]}"),
                    agent_id=f"cup_arb_L{level_idx}_{eval_seed}",
                    config={**cfg, "mode": "eval"},
                )

                manager_eval = AdaptiveTrainingManager(
                    controller=controller_eval,
                    env=cup_env,
                    config=cfg,
                    runs_dir=str(runs_dir),
                    mesh_path=cup_mesh_path,
                    offline_threshold=0.0,
                    online_sac_update_every=999999,
                    online_bc_update_every=999999,
                )
                manager_eval.sac_trainer = sac_trainer_arb

                for ep_data in level_pool:
                    start_pos = np.array(ep_data["start_pos"])
                    start_rot = np.array(ep_data["start_rot"])
                    cup_env.reset(position=start_pos, rotation=start_rot)
                    goal_pose = np.concatenate([
                        np.array(ep_data["goal_pos"]),
                        np.array(ep_data["goal_rot"]),
                    ])
                    controller_eval.set_new_goal(goal_pose, start_pos)
                    cup_env.set_goal(goal_pose)

                    goals_before = controller_eval._total_goals_reached
                    success = False
                    collision = False

                    for step in range(150):
                        current_pose = cup_env.get_pose()
                        sensor_data = cup_env.get_sensor_data()

                        state = controller_eval._compute_state(current_pose, sensor_data)
                        action_index, source = manager_eval.get_action(
                            state, current_pose, sensor_data
                        )

                        state, done = controller_eval.update_only(
                            current_pose, sensor_data, action_index
                        )

                        if done:
                            if controller_eval._total_goals_reached > goals_before:
                                success = True
                            break

                        cup_env.step(action_index, controller_eval.action_space)
                        if action_index in (
                            controller_eval.action_space.IDX_DETACH,
                            controller_eval.action_space.IDX_DETACH_EDGE,
                        ):
                            controller_eval._last_detach_sub_steps = getattr(
                                cup_env, '_last_detach_sub_steps', 1
                            )

                    level_total += 1
                    if success:
                        level_successes += 1
                    else:
                        depth = cup_env.get_sensor_data().get("depth", 100.0)
                        if depth < 0.5:
                            level_collisions += 1
                        else:
                            level_timeouts += 1

            count = max(level_total, 1)
            bounds = CURRICULUM_LEVELS[level_idx]
            arb_results_per_level[f"level_{level_idx}"] = {
                "bounds_mm": list(bounds),
                "success_rate": level_successes / count,
                "timeout_rate": level_timeouts / count,
                "collision_rate": level_collisions / count,
            }

        print(f"\n=== Cup Arbitrage zero-shot ===")
        for key in sorted(arb_results_per_level.keys()):
            data = arb_results_per_level[key]
            print(
                f"  {key} ({data.get('bounds_mm', [])}mm): "
                f"success={data['success_rate']:.4f}, "
                f"timeout={data['timeout_rate']:.4f}, "
                f"collision={data['collision_rate']:.4f}"
            )

        arb_stats = manager_eval.arbitrator.get_stats()
        print(f"\n  Arbitrator stats:")
        print(
            f"  Sources: q_store={arb_stats['q_store_rate']:.2f}, "
            f"q_weak={arb_stats.get('q_store_weak_rate', 0):.2f}, "
            f"sac={arb_stats['sac_rate']:.2f}, "
            f"heuristic={arb_stats['heuristic_rate']:.2f}"
        )
        print(
            f"  Agreement: {arb_stats['agreement_rate']:.2f}, "
            f"q_spread={arb_stats['q_spread_mean']:.2f}"
        )

        eval_output = data_dir / "cup_arb_eval_result.json"
        with open(eval_output, "w", encoding="utf-8") as f:
            json.dump(arb_results_per_level, f, indent=2)
        print(f"Saved to {eval_output}")

        # ─── Сравнение ───
        print(f"\n=== Transfer Comparison: Mug → Cup ===")
        print(f"{'Level':<25} {'SAC zero-shot':<18} {'Arbitrage zero-shot':<18}")
        print("-" * 61)
        for level_idx in range(num_levels):
            key = f"level_{level_idx}"
            bounds = CURRICULUM_LEVELS[level_idx]
            sac_sr = sac_results[key]["success_rate"]
            arb_sr = arb_results_per_level[key]["success_rate"]
            diff = arb_sr - sac_sr
            print(
                f"  {key} ({bounds}mm)    "
                f"{sac_sr:.3f}             "
                f"{arb_sr:.3f}  ({diff:+.3f})"
            )

if __name__ == "__main__":
    main()
