"""End-to-end eval: RLGoalPolicy replaces JumpToGoal in Monty pipeline.

Steps:
1. Load pretrained Monty model (banana + mug graphs)
2. Create MuJoCo environment with banana
3. Create RLPolicySelector with RLGoalPolicy + SurfacePolicy
4. Run eval: LM recognizes object using RL navigation for hypothesis testing

Run:
    export MONTY_DATA=~/Downloads/tbp/data
    python eval_rl_e2e.py
"""
from tbp.monty.frameworks.run_env import setup_env

import logging
import os

os.environ.setdefault("MONTY_DATA", os.path.expanduser("~/Downloads/tbp/data"))

setup_env()

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
BANANA_STL = os.path.expanduser(
    "~/Downloads/github/hybrid_RL_prototype/results/adapt-baseline"
    "/data/ycb/ycb_banana.stl"
)
MUJOCO_DATA = os.path.expanduser(
    "~/Downloads/tbp/data/mujoco/objects/ycb"
)

# ═══ Step 1: Create baseline experiment config ═══
logger.info("Loading baseline experiment config...")

with hydra.initialize_config_dir(version_base=None, config_dir=MONTY_CONF):
    config = hydra.compose(
        config_name="experiment",
        overrides=[
            "experiment=tutorial/surf_agent_2obj_eval_mujoco",
            f"experiment.config.model_name_or_path={PRETRAINED}",
            "experiment.config.n_eval_epochs=1",
            "experiment.config.max_eval_steps=300",
            "experiment.config.show_sensor_output=false",
        ],
    )

# ═══ Step 2: Instantiate and enter experiment context ═══
logger.info("Instantiating experiment...")
experiment = instantiate_experiment(config.experiment)

logger.info("Entering experiment context (initializes model + environment)...")
experiment.__enter__()

# ═══ Step 3: Replace motor system with RLPolicySelector ═══
logger.info("Replacing motor system with RLPolicySelector...")

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.sensors import SensorID
from tbp.hybrid_rl.mujoco_env_adapter import MuJoCoEnvAdapter
from tbp.hybrid_rl.rl_goal_policy import RLGoalPolicy
from tbp.hybrid_rl.rl_policy_selector import RLPolicySelector
from tbp.monty.frameworks.models.motor_system import MotorSystem

# Get the original surface policy from the existing motor system
original_selector = experiment.model.motor_system._policy_selector
original_policy = original_selector._policy  # SurfacePolicyCurvatureInformed

# Disable goal-driven actions in surface policy (RL handles goals now)
original_policy.use_goal_driven_actions = False

# Create MuJoCo adapter for RL
logger.info("Creating MuJoCo adapter for RL...")
adapter = MuJoCoEnvAdapter(
    mesh_path_mm=BANANA_STL,
    mujoco_object_name="banana",
    mujoco_data_path=MUJOCO_DATA,
    seed=42,
)

# Create RLGoalPolicy
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
}

rl_policy = RLGoalPolicy(
    agent_id=AgentID("agent_id_0"),
    sensor_id=SensorID("view_finder"),
    model_path=Q_STORE,
    rl_config=rl_config,
    mujoco_adapter=adapter,
    mesh_path=BANANA_STL,
    max_nav_steps=500,
    enable_online_learning=True,
)

# Create RLPolicySelector (look_at_goal=None for surface agent)
rl_selector = RLPolicySelector(
    rl_goal_policy=rl_policy,
    default=original_policy,
)

# Replace motor system
new_motor_system = MotorSystem(policy_selector=rl_selector)
experiment.model.motor_system = new_motor_system

# ═══ Step 4: Run eval ═══
logger.info("=" * 60)
logger.info("Running eval with RLGoalPolicy...")
logger.info("Objects: banana + mug (pretrained)")
logger.info("RL model: Q-store trained on trimesh primitives")
logger.info("=" * 60)

try:
    experiment.run()
    logger.info("=" * 60)
    logger.info("END-TO-END EVAL COMPLETE!")
    logger.info("=" * 60)
except Exception as e:
    logger.error("Eval failed: %s", e, exc_info=True)
finally:
    adapter.close()
    experiment.__exit__(None, None, None)
