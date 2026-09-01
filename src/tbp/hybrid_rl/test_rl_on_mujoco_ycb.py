"""Test RL agent on YCB objects via MuJoCo.

Usage:
    python -m tbp.hybrid_rl.test_rl_on_mujoco_ycb \
        --mesh_path ~/tbp/data/habitat/ycb/025_mug/textured.obj \
        --data_path ~/tbp/data/habitat/ycb \
        --object_name 025_mug \
        --model_dir ./trained_models/mug_seed_42 \
        --num_episodes 100

    # Multiple objects:
    python -m tbp.hybrid_rl.test_rl_on_mujoco_ycb \
        --mesh_path ~/tbp/data/habitat/ycb/025_mug/textured.obj \
                    ~/tbp/data/habitat/ycb/005_tomato_soup_can/textured.obj \
        --data_path ~/tbp/data/habitat/ycb \
        --object_name 025_mug 005_tomato_soup_can \
        --model_dir ./trained_models/mug_seed_42 \
        --num_episodes 50
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from tbp.hybrid_rl.mujoco_env_adapter import MuJoCoEnvAdapter
from tbp.hybrid_rl.rl_goal_approach_controller import RLGoalApproachController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_on_object(
    mesh_path: str,
    data_path: str,
    object_name: str,
    controller: RLGoalApproachController,
    num_episodes: int,
    max_steps: int,
    seed: int,
    min_goal_dist: float = 10.0,
    max_goal_dist: float = 80.0,
) -> dict:
    """Evaluate RL agent on a single YCB object in MuJoCo.

    Args:
        mesh_path: Path to textured.obj
        data_path: Path to YCB objects directory
        object_name: YCB object name (e.g. "025_mug")
        controller: Trained RL controller
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        seed: Random seed
        min_goal_dist: Minimum goal distance in mm
        max_goal_dist: Maximum goal distance in mm

    Returns:
        Dict with evaluation metrics
    """
    np.random.seed(seed)
    action_space = controller.action_space

    goals_reached = 0
    total_steps = 0
    episode_results = []

    with MuJoCoEnvAdapter(
        mesh_path=mesh_path,
        data_path=data_path,
        object_name=object_name,
        seed=seed,
    ) as env:

        for episode in range(num_episodes):
            ep_start = time.time()

            # Reset agent
            sensor_data = env.reset()
            start_pos = env.get_pose()[:3]

            # Random goal on surface
            goal_pose = env.get_random_surface_point(
                reference_pos=start_pos,
                min_dist=min_goal_dist,
                max_dist=max_goal_dist,
            )
            env.set_goal(goal_pose)

            goal_dist = float(np.linalg.norm(goal_pose[:3] - start_pos))
            controller.set_new_goal(goal_pose, start_pos)

            # Run episode
            episode_success = False
            ep_steps = 0

            for step in range(max_steps):
                pose = env.get_pose()
                sensor_data = env.get_sensor_data()

                _, explanation = controller.step(pose, sensor_data)

                if controller._current_goal is None:
                    episode_success = True
                    break

                env.step(controller._last_action, action_space)
                ep_steps += 1

            ep_time = time.time() - ep_start

            if episode_success:
                goals_reached += 1
                total_steps += ep_steps

            ep_result = {
                "episode": episode,
                "success": episode_success,
                "steps": ep_steps,
                "goal_distance": round(goal_dist, 1),
                "time_s": round(ep_time, 2),
            }
            episode_results.append(ep_result)

            if (episode + 1) % 10 == 0:
                rate = goals_reached / (episode + 1)
                logger.info(
                    f"[{object_name}] Episode {episode+1}/{num_episodes}: "
                    f"success_rate={rate:.3f} ({goals_reached}/{episode+1})"
                )

    success_rate = goals_reached / max(num_episodes, 1)
    avg_steps = total_steps / max(goals_reached, 1)

    return {
        "object_name": object_name,
        "num_episodes": num_episodes,
        "goals_reached": goals_reached,
        "success_rate": round(success_rate, 4),
        "avg_steps_per_success": round(avg_steps, 1),
        "episode_results": episode_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test RL agent on YCB objects via MuJoCo"
    )
    parser.add_argument(
        "--mesh_path", nargs="+", required=True,
        help="Path(s) to textured.obj file(s)"
    )
    parser.add_argument(
        "--data_path", required=True,
        help="Path to YCB objects directory"
    )
    parser.add_argument(
        "--object_name", nargs="+", required=True,
        help="YCB object name(s)"
    )
    parser.add_argument(
        "--model_dir", required=True,
        help="Directory with trained RL model"
    )
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_goal_dist", type=float, default=10.0)
    parser.add_argument("--max_goal_dist", type=float, default=80.0)
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save results JSON"
    )
    args = parser.parse_args()

    assert len(args.mesh_path) == len(args.object_name), (
        "Number of mesh_path and object_name must match"
    )

    # Load trained controller
    logger.info(f"Loading model from {args.model_dir}")
    controller = RLGoalApproachController.load(
        args.model_dir, agent_id="mujoco_eval"
    )

    # Evaluate on each object
    all_results = {}
    for mesh_path, obj_name in zip(args.mesh_path, args.object_name):
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating on: {obj_name}")
        logger.info(f"Mesh: {mesh_path}")
        logger.info(f"{'='*60}")

        result = evaluate_on_object(
            mesh_path=mesh_path,
            data_path=args.data_path,
            object_name=obj_name,
            controller=controller,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            min_goal_dist=args.min_goal_dist,
            max_goal_dist=args.max_goal_dist,
        )
        all_results[obj_name] = result

        logger.info(
            f"\n{obj_name} RESULTS:\n"
            f"  Success rate: {result['success_rate']:.3f}\n"
            f"  Goals reached: {result['goals_reached']}/{result['num_episodes']}\n"
            f"  Avg steps/success: {result['avg_steps_per_success']:.1f}"
        )

    # Overall summary
    total_goals = sum(r["goals_reached"] for r in all_results.values())
    total_episodes = sum(r["num_episodes"] for r in all_results.values())
    overall_rate = total_goals / max(total_episodes, 1)

    logger.info(f"\n{'='*60}")
    logger.info(f"OVERALL RESULTS")
    logger.info(f"  Objects tested: {len(all_results)}")
    logger.info(f"  Total episodes: {total_episodes}")
    logger.info(f"  Overall success rate: {overall_rate:.3f}")
    for obj_name, result in all_results.items():
        logger.info(f"  {obj_name}: {result['success_rate']:.3f}")
    logger.info(f"{'='*60}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove episode_results for compact JSON
        compact_results = {}
        for obj_name, result in all_results.items():
            compact_results[obj_name] = {
                k: v for k, v in result.items() if k != "episode_results"
            }
        compact_results["overall"] = {
            "success_rate": round(overall_rate, 4),
            "total_episodes": total_episodes,
            "total_goals": total_goals,
        }

        with output_path.open("w") as f:
            json.dump(compact_results, f, indent=2)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
