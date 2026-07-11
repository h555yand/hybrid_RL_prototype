# config.py
"""Default configuration for RL goal-approach controller."""

DEFAULT_CONFIG = {
    # State
    "state_dim": 15,
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
    "norm_warmup_steps": 5000,          # сколько raw states накопить для freeze
    "norm_min_std": 1e-4,               # защита от нулевой дисперсии
    "rebuild_on_freeze": True,          # rebuild индекса после freeze
    # Actions
    "num_actions": 21,
    "free_step_small": 2.0,
    "surface_step": 3.0,   # mm
    "free_step": 8.0,     # mm
    "rotation_step": 5.0, # degrees
    # Episode
    "goal_threshold": 2.0, # mm
    "max_steps_per_goal": 200,
    # Collision detection
    "min_valid_depth": 0.5,    # mm — below = inside object
    "max_sensor_range": 100.0, # mm — max depth reading
    "normal_flip_threshold": -0.5,  # dot product for pass-through
    # Reward weights
    "reward_progress": 3.0,
    "reward_goal_reached": 60.0,
    "reward_step_penalty": -0.2,
    "reward_surface_violation": -15.0,
    "reward_smart_detach": 1.5,
    "reward_drifted_away": -1.0,
    "reward_near_goal_on_surface": 0.5,
    "reward_oscillation": -0.5,
    "reward_timeout": -8.0,
    "reward_detach_collision": -3.0,    # ← ДОБАВИТЬ
    # Detour shaping: if goal is behind surface, clip negative progress penalty
    # to avoid over-penalizing necessary face-to-face transitions on polyhedra.
    "detour_alignment_threshold": -0.2,
    "detour_negative_progress_clip_steps": 0.3,
    # Q-learning
    "gamma": 0.95,             # discount factor
    "alpha": 0.1,              # learning rate
    "num_episodes": 1000,
    # Режим
    # "train"    — всегда обучение
    # "train_adapt_epsilon"    — обучение с адаптивным epsilon
    # "eval"     — всегда inference
    # "auto"     — определяем автоматически
    "mode": "auto",
    "auto_train_threshold": 3500,  # точек в Q-store для перехода в eval
    # Training epsilon
    "epsilon_decay": 0.99977,    # per-step decay
    "epsilon_start": 1.0,      # initial exploration
    "epsilon_min": 0.05,       # minimum exploration
    # Eval epsilon
    "eval_epsilon": 0.02,
    # Eval learning rate multiplier
    "eval_alpha_multiplier": 0.1,  # alpha × 0.1 в eval режиме
    "temperature_override": None
}
