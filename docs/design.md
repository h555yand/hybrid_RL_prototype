### 1. Модули

| Модуль | Файл | Назначение |
|--------|------|------------|
| **LightweightEnv** | `lightweight_env.py` | Среда симуляции на Trimesh. Позиция агента, raycast, нормали, depth, движение по поверхности |
| **ActionSpace** | `action_space.py` | 20 дискретных действий (8 surface + 2 free + 6 orient + 2 orient_surface + 2 macro). Метаданные, противоположные пары |
| **HNSWStateStore** | `hnsw_state_store.py` | Эпизодическая память. HNSW граф + kNN + Gaussian Kernel для хранения и интерполяции Q-values |
| **RLGoalApproachController** | `rl_goal_approach_controller.py` | Q-learning контроллер. State computation (15D), reward, collision detection, heuristic-guided exploration, Q-update |
| **ExperienceExtractor** | `experience_extractor.py` | Конвертер дискретных действий (0–19) в параметризованные (8 типов + params). Формат PSACTransition |
| **BCTrainer** | `behavioral_cloning.py` | Behavioral Cloning. Supervised learning: state → (action_type, action_params) |
| **SACActorNetwork** | `sac_actor.py` | Actor нейросеть. Encoder → type_head (Categorical) + param_heads (Gaussian per type) |
| **TwinCritic** | `twin_critic.py` | Два Q-network (state + action_onehot + params → Q-value). Clipped Double Q |
| **ReplayBuffer** | `replay_buffer.py` | Буфер опыта с защищённым BC-резервуаром (15% capacity) |
| **PSACTrainer** | `sac_trainer.py` | SAC тренировочный цикл. Critic/Actor/Alpha update, curriculum, eval-based checkpointing |
| **ActionInterpreter** | `action_interpreter.py` | Исполнитель SAC действий. Конвертирует (type, params) → вызовы env методов |
| **Arbitrator** | `arbitrator.py` | Арбитраж между Q-store, SAC и heuristic. Конвертация SAC→discrete через sac_to_discrete() |
| **AdaptiveTrainingManager** | `adaptive_manager.py` | Менеджер режимов (inference/online/offline). Online SAC update, мониторинг success rate |
| **RLAblationRunner** | `ablation_runner.py` | Оркестратор Q-learning тренировки. Curriculum variants, multi-seed, функция train() |
| **run_ablation.py (main)** | `run_ablation.py` | Точка входа. STEP 1–10, генерация пулов, mesh объекты, конфигурация |

### 2. Взаимодействие модулей

```
┌─────────────────────────────────────────────────────────────────┐
│                        run_ablation.py (main)                   │
│  Оркестрация STEP 1-10, конфигурация, генерация episode pools   │
└──────────┬──────────┬──────────┬──────────┬──────────┬──────────┘
           │          │          │          │          │
     STEP 2-3    STEP 4-5    STEP 6     STEP 7    STEP 8-10
           │          │          │          │          │
           ▼          ▼          ▼          ▼          ▼
┌──────────────┐ ┌─────────┐ ┌─────────┐ ┌────┐ ┌──────────────┐
│RLAblation    │ │BCTrainer│ │PSACTrain│ │Eval│ │Adaptive      │
│Runner        │ │         │ │er       │ │    │ │Manager       │
│  ↓           │ │         │ │         │ │    │ │  ↓           │
│ train()      │ │         │ │         │ │    │ │ Arbitrator   │
└──────┬───────┘ └────┬────┘ └────┬────┘ └──┬─┘ └──────┬───────┘
       │              │           │          │          │
       ▼              │           ▼          │          ▼
┌──────────────┐      │    ┌───────────┐     │   ┌───────────┐
│RLGoalApproach│      │    │SACActorNet│     │   │  Q-store   │
│Controller    │      │    │TwinCritic │     │   │  + SAC     │
│  ↓           │      │    │ReplayBuf  │     │   │  + Heurist │
│ HNSWState×2  │      │    └─────┬─────┘     │   └───────────┘
│ (free+surf)  │      │          │           │
└──────┬───────┘      │          ▼           │
       │              │   ┌────────────┐     │
       ▼              │   │Action      │     │
┌──────────────┐      │   │Interpreter │     │
│ActionSpace   │      │   └──────┬─────┘     │
│(20 actions)  │      │          │           │
└──────┬───────┘      │          ▼           │
       │              │   ┌────────────┐     │
       └──────────────┴──►│Lightweight │◄────┘
                          │Env         │
                          │(Trimesh)   │
                          └────────────┘
```

### Потоки данных между модулями

**Experience flow (Q-learning):**
```
LightweightEnv → sensor_data
  → Controller._compute_state() → state (15D)
  → Controller._choose_action() → action_index (0-19)
  → ActionSpace → env.step() → new sensor_data
  → (state, action, reward, next_state) → HNSWStateStore (Q-update)
```

**BC flow:**
```
Controller.success_trails
  → ExperienceExtractor.convert_trajectory()
  → PSACTransition (type 0-7, params)
  → BCTrainer.train()
  → bc_actor.pt, bc_normalization.npz
```

**SAC flow:**
```
BCTrainer weights → SACActorNetwork.load_bc_weights()
BC transitions → ReplayBuffer.load_bc_data() (protected zone)
PSACTrainer.train():
  → actor.sample() → ActionInterpreter.execute() → env
  → reward → buffer.add()
  → update_critic/actor/alpha → soft_update_target
```

**Arbitrage flow:**
```
state → Arbitrator.decide()
  → _get_q_action():
      HNSWStateStore.get_q_values() → (action, confidence, spread)
  → _get_sac_action_discrete():
      actor.sample() → sac_to_discrete() → (action, confidence)
  → выбор: Q-store (confident+spread) > SAC > Q-store_weak > heuristic
action_index → Controller.update_only() → Q-store update
```

### 3. Пайплайн тренировки и валидации

**STEP 1: Генерация episode pools**
```
├── train_pools:     3 seeds × 5000 episodes × 3 levels
├── eval_pools:      3 seeds × 500 episodes × 3 levels
└── sac_eval_pools:  3 seeds × 500 episodes × 3 levels
Фиксированные (start_pos, goal_pos) для воспроизводимости.
Curriculum levels: (10-40mm), (20-80mm), (40-120mm)
```

**STEP 2: Q-learning тренировка**
```
├── Input:  train_pools, mesh (mug.stl)
├── Method: RLAblationRunner → train() с curriculum
│   ├── Heuristic-guided exploration (ε: 0.15→0.05)
│   ├── Q-update через HNSWStateStore (2 stores: free + surface)
│   ├── Success backup (λ-return по успешным траекториям)
│   └── Curriculum promotion (rolling rate > threshold)
├── Output: Q-store (67K free + 38K surface points), config.json
└── Метрики: success_rate per level, collision_rate, timeout_rate
```

**STEP 3: Q-learning валидация + сбор BC данных**
```
├── Input:  eval_pools, обученный Q-store
├── Method: eval mode (ε=0.02), collect success_trails
├── Output: bc_data.pkl (~11K transitions), eval_result.json
└── Метрики: success_rate per level per seed
```

**STEP 5: Behavioral Cloning**
```
├── Input:  bc_data.pkl (PSACTransition)
├── Method: Supervised learning (CrossEntropy + MSE)
│   ├── type_head: predict action_type (0-7)
│   └── param_heads: predict params per type
├── Output: bc_actor.pt, bc_normalization.npz
└── Метрики: train/val loss, type accuracy
```

**STEP 6: P-SAC тренировка**
```
├── Input:  bc_actor.pt, bc_data.pkl, mesh
├── Method: PSACTrainer.train()
│   ├── BC warm-start (actor weights + buffer)
│   ├── ReplayBuffer с BC-резервуаром (15%)
│   ├── Curriculum levels
│   ├── Alpha auto-tuning (type: 0.135-2.7, param: 0.135-1.0)
│   ├── BC-regularization (λ: 5.0, decay: 0.999999)
│   ├── Eval каждые 200 эпизодов (фиксированный seed)
│   └── Best model по eval success rate
├── Output: sac_actor.pt, sac_critic.pt, sac_state.npz
└── Метрики: train rolling rate, eval rate, alpha, bc_lambda
```

**STEP 7: P-SAC валидация**
```
├── Input:  sac_model, sac_eval_pools
├── Method: sample (softmax) и predict (argmax)
├── Output: sac_eval_result_sample.json, sac_eval_result_predict.json
└── Метрики: success_rate per level, mode comparison
```

**STEP 8: Adaptive (арбитраж Q-store + SAC)**
```
├── Input:  Q-store, SAC model, mesh
├── Method: AdaptiveTrainingManager
│   ├── Arbitrator.decide() → Q-store / SAC / heuristic
│   ├── Q-store обновляется каждый шаг
│   ├── Online SAC update каждые 200 эпизодов (50 gradient steps)
│   └── Мониторинг success_rate → режим (inference/online/offline)
└── Метрики: success_rate, source distribution, agreement rate
```

**STEP 9: Transfer eval (zero-shot на новом объекте)**
```
├── Input:  SAC model + Q-store (от кружки), cup.stl
├── Method: SAC eval + Arbitrage eval без дообучения
└── Метрики: success_rate comparison (SAC vs arbitrage)
```

**STEP 10: Adaptive на новом объекте**
```
├── Input:  Q-store + SAC (от кружки), cup.stl
├── Method: Арбитраж с online дообучением
└── Метрики: success_rate динамика, adaptation speed


### On action space independence

> "Try and make the solution one that is action space independent"

**Status: Addressed.** The architecture is action-space independent:
- `ActionSpace` is a configurable input parameter with metadata (categories, opposites)
- `RLGoalApproachController` works with any discrete action set
- SAC uses parameterized action types that map to any underlying action space via `ActionInterpreter`
- Heuristics compute bias weights for each action based on geometric reasoning, not hardcoded action selection

#### Discrete Action Space (Q-learning, 20 actions)

| Index | Action | Category | Parameters | Description |
|-------|--------|----------|------------|-------------|
| 0–7 | MoveTangentially | surface | direction: 0°,45°,...,315°; distance: surface_step | Crawl along surface in 8 directions |
| 8 | MoveForward | free | distance: free_step | Move forward (where sensor points) |
| 9 | MoveBackward | free | distance: -free_step | Move backward |
| 10 | TurnLeft | orient | rotation: rotation_step | Rotate agent left (yaw) |
| 11 | TurnRight | orient | rotation: -rotation_step | Rotate agent right (yaw) |
| 12 | LookUp | orient | rotation: rotation_step | Tilt agent up (pitch) |
| 13 | LookDown | orient | rotation: -rotation_step | Tilt agent down (pitch) |
| 14 | RotateSensor+ | orient | rotation: rotation_step | Rotate sensor clockwise (roll) |
| 15 | RotateSensor- | orient | rotation: -rotation_step | Rotate sensor counter-clockwise |
| 16 | OrientHorizontal | surface | rotation, left_dist, fwd_dist | Correct horizontal orientation on surface |
| 17 | OrientVertical | surface | rotation, down_dist, fwd_dist | Correct vertical orientation on surface |
| 18 | Detach | macro | — | Detach from surface, fly toward goal, land |
| 19 | DetachEdge | macro | — | Detach, fly over edge, land on other side |

#### Parameterized Action Space (SAC, 8 types)

| Type | Name | Parameters | Maps to discrete |
|------|------|------------|-----------------|
| 0 | MoveTangentially | angle_deg (0–360°), distance (mm) | → idx 0–7 (nearest direction) |
| 1 | MoveLinear | distance (mm, signed) | → idx 8 (positive) or 9 (negative) |
| 2 | Turn | rotation (deg, signed) | → idx 10 (positive) or 11 (negative) |
| 3 | Look | rotation (deg, signed) | → idx 12 (positive) or 13 (negative) |
| 4 | SensorRotate | rotation (deg, signed) | → idx 14 (positive) or 15 (negative) |
| 5 | OrientHorizontal | rotation, left_dist, fwd_dist | → idx 16 |
| 6 | OrientVertical | rotation, down_dist, fwd_dist | → idx 17 |
| 7 | Detach | — | → idx 18 |

SAC collapses 20 discrete actions into 8 parameterized types with continuous parameters. This allows the policy to output precise values (e.g., angle=37°, distance=4.2mm) instead of choosing from fixed directions.

#### Heuristic Bias System

Heuristics do not select actions directly. Instead, they compute a **bias vector** — a weight for each of the 20 actions — based on geometric reasoning about the current situation. This bias is blended with Q-values to form the final action distribution:

```
combined = (1 - ε) × Q_normalized + ε × heuristic_bias_normalized
action = softmax_sample(combined, temperature)
```

Each heuristic component outputs a vector of 20 weights:

| Heuristic | What it does | Example bias |
|-----------|-------------|-------------|
| **surface_move** | Boost surface action closest to goal direction | idx 3: +2.0 (goal is at 135°) |
| **detach** | Boost detach when goal is behind surface | idx 18: +5.0, idx 0–7: -2.0 |
| **steer_in_air** | Boost turn/forward when flying toward goal | idx 10: +2.0 (need to turn left) |
| **damp_free** | Suppress free_forward on surface (collision risk) | idx 8,9: -3.0 |
| **suppress** | Suppress rarely useful actions | idx 14,15: -2.0 (sensor rotate) |
| **stagnation** | Boost detach when stuck on surface | idx 18: +3.0, idx 0–7: -2.0 |

The biases are **additive** — multiple heuristics contribute simultaneously. For example, on surface near goal: `surface_move` boosts idx 2 (+2.0), `damp_free` suppresses idx 8 (-3.0), `suppress` suppresses idx 14 (-2.0). The combined bias vector guides exploration toward geometrically reasonable actions while allowing Q-values to override when they have learned better strategies.