# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Default configuration for RL goal-approach controller."""

DEFAULT_CONFIG = {
    # State
    "state_dim": 20,
    # HNSW store
    "max_points": 100000,
    "k_neighbors": 7,
    "sigma": 1.0,
    "insert_threshold": 0.5,
    "adaptive_sigma": True,
    "evict_fraction": 0.1,
    "auto_calibrate": False,
    "calibration_percentile": 10.0,
    "min_calibration_samples": 100,
    "min_weight_threshold": 0.01,
    "norm_warmup_steps": 5000,          # how many raw states to accumulate for freezing
    "norm_min_std": 1e-4,               # zero dispersion protection
    "rebuild_on_freeze": True,          # rebuild index after freeze
    "feature_weights": None,
    # Actions
    "num_actions": 24,
    "free_step_small": 2.0,
    "surface_step": 3.0,   # mm
    "free_step": 8.0,     # mm
    "free_step_backward": 2.0,         # mm
    "rotation_step": 5.0,              # degrees
    "rotation_step_big": 15.0,         # degrees
    # Episode
    "goal_threshold": 2.0,  # mm
    "max_steps_per_goal": 200,
    # Collision detection
    "min_valid_depth": 0.5,    # mm — below = inside object
    "max_sensor_range": 100.0,  # mm — max depth reading
    "normal_flip_threshold": -0.5,  # dot product for pass-through
    # Reward weights
    "reward_progress": 3.0,
    "reward_goal_reached": 60.0,
    "reward_step_penalty": -0.5,
    "reward_surface_violation": -12.0,
    "reward_smart_detach": 1.5,
    "reward_drifted_away": -1.0,
    "reward_near_goal_on_surface": 0.5,
    "reward_oscillation": -0.5,
    "reward_timeout": -12.0,
    "reward_detach_collision": -3.0,
    # Strategic Q-store
    "strategic_epsilon_start": 1.0,
    "strategic_epsilon_min": 0.3,
    "strategic_eval_epsilon": 0.3,
    "strategic_reward_switch_success": 1.0,
    "strategic_reward_switch_wrong_side": -0.3,
    "strategic_reward_switch_collision": -0.5,
    "strategic_reward_stay_when_needed": -0.1,
    "strategic_reward_stay_correct": 0.3,
    "strategic_alpha_stay_multiplier": 0.1,
    "strategic_alpha_switch_multiplier": 1.0,
    "strategic_reward_stay_when_clear": -0.3,
    "strategic_reward_switch_when_clear": 0.3,
    "strategic_reward_stay_orbit_progress": 0.02,
    "strategic_reward_stay_orbit_stuck": -0.05,
    "strategic_reward_stay_should_have": 0.2,
    # Detour shaping: if goal is behind surface, clip negative progress penalty
    # to avoid over-penalizing necessary face-to-face transitions on polyhedra.
    "detour_alignment_threshold": -0.2,
    "detour_negative_progress_clip_steps": 0.3,
    # Q-learning
    "gamma": 0.95,             # discount factor
    "alpha": 0.1,              # learning rate
    "num_episodes": 1000,
    # Mode
    # "train"    — always train
    # "train_adapt_epsilon"    — train with adaptive epsilon
    # "eval"     — always inference
    # "auto"     — define autmatically
    "mode": "auto",
    "auto_train_threshold": 3500,  # points in the Q-store to go to eval
    # Training epsilon
    "epsilon_decay": 0.99977,    # per-step decay
    "epsilon_start": 1.0,      # initial exploration
    "epsilon_min": 0.05,       # minimum exploration
    # Eval epsilon
    "eval_epsilon": 0.02,
    # Eval learning rate multiplier
    "eval_alpha_multiplier": 0.1,  # alpha × 0.1 в eval mode
    "temperature_override": None,
    "warmup_episodes": 200,
    "air_start_enabled": True
}
