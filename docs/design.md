### 1. Modules

| Module | File | Purpose |
|--------|------|---------|
| **LightweightEnv** | `lightweight_env.py` | Trimesh-based simulation environment. Agent position, raycast, normals, depth, surface movement |
| **ActionSpace** | `action_space.py` | 21 discrete actions (8 surface + 2 free + 1 free_small + 6 orient + 2 orient_surface + 2 macro). Metadata, opposite pairs |
| **HNSWStateStore** | `hnsw_state_store.py` | Episodic memory. HNSW graph + kNN + Gaussian Kernel for Q-value storage and interpolation |
| **RLGoalApproachController** | `rl_goal_approach_controller.py` | Q-learning controller. State computation (15D), reward, collision detection, heuristic-guided exploration, Q-update |
| **ExperienceExtractor** | `experience_extractor.py` | Discrete-to-parameterized action converter (0–20 → 9 types + params). PSACTransition format |
| **BCTrainer** | `behavioral_cloning.py` | Behavioral Cloning. Supervised learning: state → (action_type, action_params) |
| **SACActorNetwork** | `sac_actor.py` | Actor network. Encoder → type_head (Categorical) + param_heads (Gaussian per type) |
| **TwinCritic** | `twin_critic.py` | Twin Q-networks (state + action_onehot + params → Q-value). Clipped Double Q |
| **ReplayBuffer** | `replay_buffer.py` | Experience buffer with protected BC reservoir (15% capacity) |
| **PSACTrainer** | `sac_trainer.py` | SAC training loop. Critic/Actor/Alpha update, curriculum, eval-based checkpointing |
| **ActionInterpreter** | `action_interpreter.py` | SAC action executor. Converts (type, params) → env method calls |
| **Arbitrator** | `arbitrator.py` | Arbitration between Q-store, SAC, and heuristic. SAC→discrete conversion via sac_to_discrete() |
| **AdaptiveTrainingManager** | `adaptive_manager.py` | Mode manager (inference/online/offline). Online SAC update, success rate monitoring |
| **train()** | `ablation_runner.py` | Q-learning training orchestrator. Curriculum, multi-seed, episode pools |
| **RLGoalApproachExperiment** | `experiment.py` | Hydra-compatible experiment class. Full pipeline: train → eval → BC → SAC → adaptive |
| **run_ablation.py (main)** | `run_ablation.py` | Legacy entry point. Episode pool generation, mesh objects, configuration |

### 2. Module Interactions

```
┌─────────────────────────────────────────────────────────────────┐
│              RLGoalApproachExperiment (experiment.py)            │
│     Hydra-compatible orchestration of the full pipeline          │
│     train → eval → BC → SAC → adaptive                          │
└──────────┬──────────┬──────────┬──────────┬──────────┬──────────┘
           │          │          │          │          │
        Train      Eval      BC Train   SAC Train  Adaptive
           │          │          │          │          │
           ▼          ▼          ▼          ▼          ▼
┌──────────────┐ ┌─────────┐ ┌─────────┐ ┌────┐ ┌──────────────┐
│train()       │ │BCTrainer│ │PSACTrain│ │Eval│ │Adaptive      │
│(ablation_    │ │         │ │er       │ │    │ │Manager       │
│ runner.py)   │ │         │ │         │ │    │ │  ↓           │
│              │ │         │ │         │ │    │ │ Arbitrator   │
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
│(21 actions)  │      │          │           │
└──────┬───────┘      │          ▼           │
       │              │   ┌────────────┐     │
       └──────────────┴──►│Lightweight │◄────┘
                          │Env         │
                          │(Trimesh)   │
                          └────────────┘
```

### Data Flows Between Modules

**Experience flow (Q-learning):**
```
LightweightEnv → sensor_data
  → Controller._compute_state() → state (15D)
  → Controller._choose_action() → action_index (0-20)
  → ActionSpace → env.step() → new sensor_data
  → (state, action, reward, next_state) → HNSWStateStore (Q-update)
```

**BC flow:**
```
Controller.success_trails
  → ExperienceExtractor.convert_trajectory()
  → PSACTransition (type 0-8, params, mesh_id)
  → BCTrainer.train()
  → bc_actor.pt, bc_normalization.npz
```

**SAC flow:**
```
BCTrainer weights → SACActorNetwork.load_bc_weights()
BC transitions → ReplayBuffer.load_bc_data() (protected zone)
PSACTrainer.train():
  → actor.sample() → ActionInterpreter.execute() → env
  → reward (SMDP: step_penalty × sub_steps for detach)
  → buffer.add()
  → update_critic/actor/alpha → soft_update_target
```

**Arbitrage flow:**
```
state → Arbitrator.decide()
  → _get_q_action():
      HNSWStateStore.get_q_values() → (action, confidence, spread)
  → _get_sac_action_discrete():
      actor.sample() → sac_to_discrete() → (action, confidence)
  → selection: Q-store (confident+spread) > SAC > Q-store_weak > heuristic
action_index → Controller.update_only() → Q-store update
```

### 3. Training and Validation Pipeline

**Episode Pool Generation**
```
├── train_pools:     per seed × episodes × 3 levels
├── eval_pools:      per seed × episodes × 3 levels
└── sac_eval_pools:  per seed × episodes × 3 levels
Fixed (start_pos, goal_pos) for reproducibility.
Curriculum levels: (10-40mm), (20-80mm), (40-120mm)
```

**Q-learning Training (sequential multi-mesh)**
```
├── Input:  train_pools, meshes (cube → cylinder → mug → cup)
├── Method: train() with curriculum, unified Q-store across meshes
│   ├── Heuristic-guided exploration (ε per stage, decay per episode)
│   ├── Q-update via HNSWStateStore (2 stores: free + surface)
│   ├── Success backup (λ-return over successful trajectories)
│   ├── Curriculum promotion (rolling rate > threshold)
│   └── Normalization unfreezing on mesh transition
├── Output: unified Q-store (free + surface points), config.json
└── Metrics: success_rate per level, collision_rate, timeout_rate
```

**Q-learning Validation + BC Data Collection**
```
├── Input:  eval_pools, trained Q-store, all meshes
├── Method: eval mode (ε=0.02), collect success_trails per mesh
├── Output: bc_data.pkl (PSACTransition with mesh_id), eval_result_all.json
└── Metrics: success_rate per level per seed per mesh
```

**Behavioral Cloning**
```
├── Input:  bc_data.pkl (PSACTransition from all meshes)
├── Method: Supervised learning (CrossEntropy + MSE)
│   ├── type_head: predict action_type (0-8)
│   └── param_heads: predict params per type
├── Output: bc_actor.pt, bc_normalization.npz
└── Metrics: train/val loss, type accuracy
```

**P-SAC Training (sequential multi-mesh)**
```
├── Input:  bc_actor.pt, bc_data.pkl, meshes
├── Method: PSACTrainer.train()
│   ├── BC warm-start (actor weights + buffer)
│   ├── ReplayBuffer with BC reservoir (15%)
│   ├── Curriculum levels
│   ├── Alpha auto-tuning (type + param entropy)
│   ├── BC-regularization (λ: 5.0, decay: 0.999999)
│   ├── SMDP reward: step_penalty × sub_steps for detach actions
│   ├── Eval every N episodes (fixed seed, sample_eval mode)
│   └── Best model by eval success rate
├── Output: sac_actor.pt, sac_critic.pt, sac_state.npz
└── Metrics: train rolling rate, eval rate, alpha, bc_lambda
```

**P-SAC Validation**
```
├── Input:  sac_model, sac_eval_pools, all meshes
├── Method: sample_eval (low-temperature softmax + reduced param noise)
├── Output: sac_eval_result_all.json
└── Metrics: success_rate per level per mesh
```

**Adaptive (Q-store + SAC arbitrage)**
```
├── Input:  Q-store, SAC model, target mesh
├── Method: AdaptiveTrainingManager
│   ├── Arbitrator.decide() → Q-store / SAC / heuristic
│   ├── Q-store updated every step
│   ├── Online SAC update every N episodes (gradient steps)
│   └── Success rate monitoring → mode (inference/online/offline)
└── Metrics: success_rate, source distribution, agreement rate
```