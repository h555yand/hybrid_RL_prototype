"""End-to-end eval: RLGoalPolicy replaces JumpToGoal in Monty pipeline.

Uses shared MuJoCo simulator — one coordinate system, no offset.

Run:
    python eval_rl_e2e.py
"""
import os

os.environ["MONTY_DATA"] = os.path.expanduser("~/Downloads/tbp/data")

from tbp.monty.frameworks.run_env import setup_env

setup_env()

import logging
from pathlib import Path
import json

import hydra
import numpy as np
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
BANANA_STL = os.path.expanduser(
    "~/Downloads/github/hybrid_RL_prototype/results/adapt-baseline"
    "/data/ycb/ycb_banana.stl"
)
MUG_STL = os.path.expanduser(
    "~/Downloads/github/hybrid_RL_prototype/results/adapt-baseline/data/mug.stl"
)
MUJOCO_DATA = os.path.expanduser(
    "~/Downloads/tbp/data/mujoco/objects/ycb"
)

# ═══ Step 1: Create baseline experiment config ═══
logger.info("Loading baseline experiment config...")

with hydra.initialize_config_dir(version_base=None, config_dir=MONTY_CONF):
    #config = hydra.compose(
    #    config_name="experiment",
    #    overrides=[
    #        "experiment=tutorial/surf_agent_2obj_eval_mujoco",
    #        f"experiment.config.model_name_or_path={PRETRAINED}",
    #        "experiment.config.n_eval_epochs=1",
    #        "experiment.config.max_eval_steps=50",
    #        "experiment.config.show_sensor_output=false",
    #    ],
    #)
    config = hydra.compose(
        config_name="experiment",
        overrides=[
            "experiment=tutorial/surf_agent_2obj_eval_mujoco",
            f"experiment.config.model_name_or_path={PRETRAINED}",
            "experiment.config.show_sensor_output=false",
            # ═══ Control experiments ═══
            "experiment.config.eval_env_interface_args.object_names=[mug, banana]",  # только banana
            "experiment.config.n_eval_epochs=3",      # 1 ротации = 1 эпизода
            "experiment.config.max_eval_steps=50",    # max Monty steps per episode
        ],
    )
# ═══ Step 2: Instantiate and enter experiment context ═══
logger.info("Instantiating experiment...")
experiment = instantiate_experiment(config.experiment)

logger.info("Entering experiment context...")
experiment.__enter__()

# ═══ Step 3: Get Monty's simulator (shared!) ═══
monty_sim = experiment.env
logger.info("Monty simulator: %s", type(monty_sim).__name__)

for agent_id, agent in monty_sim._agents.items():
    emb = agent._embodiment
    logger.info(
        "  Agent %s: pos=%s rot=%s",
        agent_id, emb.position, emb.rotation,
    )

# ═══ Step 4: Create adapter using Monty's simulator ═══
logger.info("Creating MuJoCo adapter with shared simulator...")

from tbp.hybrid_rl.mujoco_env_adapter import MuJoCoEnvAdapter

adapter = MuJoCoEnvAdapter(
    external_sim=monty_sim,
    agent_id="agent_id_0",
    seed=42,
)

# ═══ Step 5: Create RLGoalPolicy ═══
logger.info("Creating RLGoalPolicy...")

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.sensors import SensorID
from tbp.hybrid_rl.rl_goal_policy import RLGoalPolicy
from tbp.hybrid_rl.rl_policy_selector import RLPolicySelector
from tbp.monty.frameworks.models.motor_system import MotorSystem

rl_config = {
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
    # "observe_every_n_steps": 0,   # Phase 1: no intermediate observations
    "observe_every_n_steps": 5, # Phase 2: directed exploration
}

rl_policy = RLGoalPolicy(
    agent_id=AgentID("agent_id_0"),
    sensor_id=SensorID("view_finder"),
    model_path=Q_STORE,
    rl_config=rl_config,
    mujoco_adapter=adapter,
    mesh_path=None,           # ← no CAD dependency
    max_nav_steps=500,
    enable_online_learning=True,
)

# Pass experiment reference for object name lookup
rl_policy.set_experiment_ref(experiment)

# ═══ Load SAC into adaptive manager ═══
if rl_policy._manager is not None and os.path.exists(
    os.path.join(SAC_DIR, "sac_actor.pt")
):
    from tbp.hybrid_rl.sac_trainer import PSACTrainer
    from tbp.hybrid_rl.experience_extractor import ExperienceExtractor

    num_types = len(ExperienceExtractor.get_type_names())
    sac_trainer = PSACTrainer(
        state_dim=rl_config.get("state_dim", 22),
        num_types=num_types,
    )
    sac_trainer.load(SAC_DIR)
    rl_policy._manager.sac_trainer = sac_trainer
    logger.info("SAC loaded from %s", SAC_DIR)

# ═══ Step 6: Replace motor system ═══
logger.info("Replacing motor system with RLPolicySelector...")

original_selector = experiment.model.motor_system._policy_selector
original_policy = original_selector._policy
original_policy.use_goal_driven_actions = False

rl_selector = RLPolicySelector(
    rl_goal_policy=rl_policy,
    default=original_policy,
)

new_motor_system = MotorSystem(policy_selector=rl_selector)
experiment.model.motor_system = new_motor_system

# ═══ Step 7: Run eval ═══
logger.info("=" * 60)
logger.info("Running eval with RLGoalPolicy (shared simulator)...")
logger.info("Objects: banana + mug (pretrained)")
logger.info("RL model: Q-store + SAC trained on trimesh primitives")
logger.info("=" * 60)

try:
    experiment.run()
    # ═══ Check Monty recognition result ═══
    log_dir = Path(Q_STORE).parent / "monty_integration_logs"
    
    # Read eval stats if available
    eval_output = Path(os.path.expanduser(
        "~/tbp/results/monty/projects/surf_agent_2obj_eval_mujoco"
    ))
    eval_csv = eval_output / "eval_stats.csv"
    if eval_csv.exists():
        import csv
        with eval_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                logger.info("MONTY RESULT: %s", dict(row))
                
                # Append to episode_log
                result_path = log_dir / "monty_results.txt"
                with result_path.open("a") as rf:
                    rf.write(f"{dict(row)}\n")
    
    logger.info("=" * 60)
    logger.info("END-TO-END EVAL COMPLETE!")
    
    # Print RL navigation summary
    rl_log = log_dir / "episode_log.json"
    if rl_log.exists():
        with rl_log.open() as f:
            episodes = json.load(f)
        successes = sum(1 for e in episodes if e["success"])
        logger.info(
            "RL Navigation: %d/%d goals reached",
            successes, len(episodes),
        )
    
    logger.info("=" * 60)

except Exception as e:
    logger.error("Eval failed: %s", e, exc_info=True)
finally:
    adapter.close()
    experiment.__exit__(None, None, None)
