"""Benchmark: JumpToGoal vs RL Navigation vs RL + Directed Exploration.

Runs 3 experiments with identical config (objects, rotations, pretrained model),
only varying the motor policy. Saves results for comparison.

Run:
    python benchmark_rl_vs_baseline.py
"""
import os
import csv
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
import numpy as np

os.environ["MONTY_DATA"] = os.path.expanduser("~/Downloads/tbp/data")

from tbp.monty.frameworks.run_env import setup_env
setup_env()

import hydra
import tbp.monty
from tbp.monty.hydra import instantiate_experiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══ Paths ═══
MONTY_CONF = os.path.join(os.path.dirname(tbp.monty.__file__), "conf")
PRETRAINED = os.path.expanduser(
    "~/tbp/results/monty/projects/surf_agent_1lm_2obj_train_mujoco/pretrained"
)
Q_STORE = os.path.expanduser(
    "~/Downloads/github/hybrid_RL_prototype/results/adapt-baseline"
    "/data/runs/q_store_seed_11"
)
SAC_DIR = os.path.expanduser(
    "~/Downloads/github/hybrid_RL_prototype/results/adapt-baseline"
    "/data/runs/sac_seed_55"
)
RESULTS_DIR = Path(os.path.expanduser(
    "~/tbp/results/monty/projects/rl_benchmark"
)) / datetime.now().strftime("%Y%m%d_%H%M%S")

EVAL_OUTPUT = Path(os.path.expanduser(
    "~/tbp/results/monty/projects/surf_agent_2obj_eval_mujoco"
))

# ═══ Shared experiment config ═══
EXPERIMENT_OVERRIDES = [
    "experiment=tutorial/surf_agent_2obj_eval_mujoco",
    f"experiment.config.model_name_or_path={PRETRAINED}",
    "experiment.config.show_sensor_output=false",
    "experiment.config.eval_env_interface_args.object_names=[mug, banana]",
    "experiment.config.n_eval_epochs=3",
    "experiment.config.max_eval_steps=50",
]

# ═══ RL config (shared between rl_no_de and rl_de5) ═══
RL_CONFIG_BASE = {
    "state_dim": 22,
    "num_actions": 24,
    "surface_step": 3.0,
    "free_step": 8.0,
    "free_step_small": 2.0,
    "free_step_backward": 2.0,
    "rotation_step": 5.0,
    "rotation_step_big": 15.0,
    "gamma": 0.95,
    "alpha": 0.1,
    "epsilon_start": 0.05,
    "epsilon_min": 0.02,
    "epsilon_decay": 0.999,
    "goal_threshold": 4.0,
    "max_steps_per_goal": 500,
    "max_sensor_range": 100.0,
    "min_valid_depth": 0.5,
    "normal_flip_threshold": -0.5,
    "reward_progress": 3.0,
    "reward_goal_reached": 30.0,
    "reward_step_penalty": -0.2,
    "reward_surface_violation": -12.0,
    "reward_timeout": -12.0,
    "reward_drifted_away": -3.0,
    "reward_near_goal_on_surface": 0.5,
    "detour_alignment_threshold": -0.3,
    "detour_negative_progress_clip_steps": 2.0,
    "stuck_threshold": 0.05,
    "k_neighbors": 7,
    "max_points": 500000,
    "insert_threshold": 0.50,
    "action_selection_version": "v2",
    "warmup_episodes": 0,
    "mode": "adaptive",
    "eval_epsilon": 0.02,
    "strategic_eval_epsilon": 0.02,
}

MODES = {
    "baseline": {"description": "Standard Monty with JumpToGoal"},
    #"rl_no_de": {"description": "RL Navigation, no directed exploration",
    #             "observe_every_n_steps": 0},
    "rl_de5":   {"description": "RL Navigation + directed exploration every 5 steps",
                 "observe_every_n_steps": 5},
}


def setup_rl_motor_system(experiment, observe_every_n_steps):
    """Replace motor system with RLPolicySelector."""
    from tbp.monty.frameworks.agents import AgentID
    from tbp.monty.frameworks.sensors import SensorID
    from tbp.hybrid_rl.mujoco_env_adapter import MuJoCoEnvAdapter
    from tbp.hybrid_rl.rl_goal_policy import RLGoalPolicy
    from tbp.hybrid_rl.rl_policy_selector import RLPolicySelector
    from tbp.monty.frameworks.models.motor_system import MotorSystem

    adapter = MuJoCoEnvAdapter(
        external_sim=experiment.env,
        agent_id="agent_id_0",
        seed=42,
    )

    rl_config = {**RL_CONFIG_BASE, "observe_every_n_steps": observe_every_n_steps}

    rl_policy = RLGoalPolicy(
        agent_id=AgentID("agent_id_0"),
        sensor_id=SensorID("view_finder"),
        model_path=Q_STORE,
        rl_config=rl_config,
        mujoco_adapter=adapter,
        mesh_path=None,
        max_nav_steps=500,
        enable_online_learning=True,
    )
    rl_policy.set_experiment_ref(experiment)

    # Load SAC
    if os.path.exists(os.path.join(SAC_DIR, "sac_actor.pt")):
        from tbp.hybrid_rl.sac_trainer import PSACTrainer
        from tbp.hybrid_rl.experience_extractor import ExperienceExtractor
        num_types = len(ExperienceExtractor.get_type_names())
        sac_trainer = PSACTrainer(
            state_dim=rl_config["state_dim"], num_types=num_types,
        )
        sac_trainer.load(SAC_DIR)
        if rl_policy._manager is not None:
            rl_policy._manager.sac_trainer = sac_trainer

    original_selector = experiment.model.motor_system._policy_selector
    original_policy = original_selector._policy
    original_policy.use_goal_driven_actions = False

    rl_selector = RLPolicySelector(
        rl_goal_policy=rl_policy, default=original_policy,
    )
    experiment.model.motor_system = MotorSystem(policy_selector=rl_selector)

    return adapter


def run_experiment(mode_name):
    mode = MODES[mode_name]
    output_dir = RESULTS_DIR / mode_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Running: %s — %s", mode_name, mode["description"])
    logger.info("=" * 60)

    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()

    with hydra.initialize_config_dir(version_base=None, config_dir=MONTY_CONF):
        config = hydra.compose(
            config_name="experiment", overrides=EXPERIMENT_OVERRIDES,
        )

    experiment = instantiate_experiment(config.experiment)
    experiment.__enter__()

    adapter = None
    try:
        if mode_name != "baseline":
            adapter = setup_rl_motor_system(
                experiment, mode["observe_every_n_steps"],
            )

        # ═══ Collect metrics after each episode ═══
        episode_results = []
        original_post_episode = experiment.post_episode

        def patched_post_episode(steps):
            # Collect metrics BEFORE post_episode resets things
            model = experiment.model
            env_iface = experiment.env_interface

            result = {
                "mode": mode_name,
                "monty_matching_steps": model.matching_steps,
                "monty_steps": model.episode_steps,
            }

            # Target object
            try:
                target = env_iface.primary_target
                result["primary_target_object"] = target.get("object", "unknown")
                result["primary_target_rotation_euler"] = str(
                    target.get("euler_rotation", "")
                )
            except (AttributeError, TypeError):
                result["primary_target_object"] = "unknown"
                result["primary_target_rotation_euler"] = ""

            # LM results
            for lm in model.learning_modules:
                result["lm_id"] = lm.learning_module_id
                result["terminal_state"] = lm.terminal_state
                result["detected_object"] = lm.detected_object
                result["num_possible_matches"] = len(lm.get_possible_matches())

                # Evidence
                if hasattr(lm, 'current_mlh'):
                    mlh = lm.current_mlh
                    result["highest_evidence"] = mlh.get("evidence", 0)
                    result["most_likely_object"] = mlh.get("graph_id", "")
                else:
                    result["highest_evidence"] = 0
                    result["most_likely_object"] = ""

                # Rotation error
                if hasattr(lm, 'detected_rotation_r') and lm.detected_rotation_r is not None:
                    try:
                        from scipy.spatial.transform import Rotation as Rot
                        detected_euler = lm.detected_rotation_r.as_euler(
                            'xyz', degrees=True
                        )
                        result["detected_rotation"] = str(
                            np.round(detected_euler, 3).tolist()
                        )
                    except Exception:
                        result["detected_rotation"] = ""
                else:
                    result["detected_rotation"] = ""

                result["detected_pose"] = str(
                    np.round(lm.detected_pose, 4).tolist()
                    if lm.detected_pose[0] is not None else ""
                )
                result["symmetry_evidence"] = getattr(lm, 'symmetry_evidence', 0)

                # Performance
                if lm.terminal_state == "match":
                    if (lm.detected_object is not None 
                        and result["primary_target_object"] in str(lm.detected_object)):
                        result["primary_performance"] = "correct"
                    else:
                        result["primary_performance"] = "incorrect"
                elif lm.terminal_state == "no_match":
                    result["primary_performance"] = "no_match"
                elif lm.terminal_state == "time_out":
                    result["primary_performance"] = "time_out"
                else:
                    result["primary_performance"] = str(lm.terminal_state)

                # GSG goals
                result["goal_states_attempted"] = len(
                    lm.buffer.stats.get("goal_states", [])
                )
                result["goal_state_achieved"] = lm.buffer.stats.get(
                    "goal_state_achieved", 0
                )

                break  # only first LM

            episode_results.append(result)
            logger.info(
                "EPISODE RESULT [%s]: obj=%s perf=%s "
                "matching=%d monty=%d evidence=%.2f goals=%d",
                mode_name,
                result.get("primary_target_object"),
                result.get("primary_performance"),
                result.get("monty_matching_steps", 0),
                result.get("monty_steps", 0),
                result.get("highest_evidence", 0),
                result.get("goal_states_attempted", 0),
            )

            # Call original
            original_post_episode(steps)

        experiment.post_episode = patched_post_episode

        # ═══ Run ═══
        import time
        t0 = time.time()
        experiment.run()
        elapsed = time.time() - t0
        logger.info("Experiment %s completed in %.1f seconds", mode_name, elapsed)

        # ═══ Save results to CSV ═══
        if episode_results:
            dest = output_dir / "eval_stats.csv"
            fieldnames = episode_results[0].keys()
            with dest.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(episode_results)
            logger.info("Saved %d episodes → %s", len(episode_results), dest)

        # ═══ Copy RL logs ═══
        if mode_name != "baseline":
            rl_log_src = Path(Q_STORE).parent / "monty_integration_logs"
            if rl_log_src.exists():
                shutil.copytree(rl_log_src, output_dir / "rl_logs", dirs_exist_ok=True)

    except Exception as e:
        logger.error("Failed %s: %s", mode_name, e, exc_info=True)
    finally:
        if adapter:
            adapter.close()
        experiment.__exit__(None, None, None)

def compare_results():
    """Load all results and build comparison table."""
    all_rows = []
    for mode_name in MODES:
        csv_path = RESULTS_DIR / mode_name / "eval_stats.csv"
        if not csv_path.exists():
            logger.warning("Missing results for %s", mode_name)
            continue
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)

    if not all_rows:
        logger.error("No results to compare!")
        return

    # Save combined
    combined_path = RESULTS_DIR / "comparison.csv"
    with combined_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    # Print summary
    from collections import defaultdict
    summary = defaultdict(lambda: {
        "correct": 0, "total": 0,
        "matching_steps": [], "monty_steps": [],
        "goals_attempted": [], "evidence": [],
    })

    for row in all_rows:
        mode = row["mode"]
        s = summary[mode]
        s["total"] += 1
        if row.get("primary_performance") == "correct":
            s["correct"] += 1
        s["matching_steps"].append(int(row.get("monty_matching_steps", 0)))
        s["monty_steps"].append(int(row.get("monty_steps", 0)))
        s["goals_attempted"].append(int(row.get("goal_states_attempted", 0)))
        s["evidence"].append(float(row.get("highest_evidence", 0)))

    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 80)
    header = f"{'Mode':<15} {'Acc':>5} {'MatchSteps':>11} {'MontySteps':>11} {'Goals':>6} {'Evidence':>9}"
    logger.info(header)
    logger.info("-" * 80)

    for mode in MODES:
        if mode not in summary:
            continue
        s = summary[mode]
        acc = f"{s['correct']}/{s['total']}"
        ms = f"{np.mean(s['matching_steps']):.1f}±{np.std(s['matching_steps']):.1f}"
        es = f"{np.mean(s['monty_steps']):.1f}±{np.std(s['monty_steps']):.1f}"
        ga = f"{np.mean(s['goals_attempted']):.1f}"
        ev = f"{np.mean(s['evidence']):.1f}"
        logger.info(f"{mode:<15} {acc:>5} {ms:>11} {es:>11} {ga:>6} {ev:>9}")

    logger.info("=" * 80)
    logger.info("Results saved to: %s", RESULTS_DIR)


if __name__ == "__main__":

    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for mode_name in MODES:
        run_experiment(mode_name)

    compare_results()