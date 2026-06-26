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
