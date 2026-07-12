# Report: Training an RL Agent for Object Surface Navigation

## 1. Initial Parameters

### Environment and Object
- **Object:** mug (mug.stl) — cylindrical body + toroidal handle
- **Environment:** LightweightEnv based on Trimesh (raycast, normals, depth)
- **Task:** navigate from an arbitrary surface point to a target point

### Q-learning Configuration
| Parameter | Initial Value | Description |
|-----------|--------------|-------------|
| `surface_step` | 3.0 mm | Surface crawling step |
| `free_step` | 5.0 mm | Aerial movement step |
| `rotation_step` | 5.0° | Rotation step |
| `goal_threshold` | 5.0 mm | Distance to goal for success |
| `max_steps_per_goal` | 150 | Maximum steps per episode |
| `epsilon_start` | 0.15 (with load) / 1.0 (from scratch) | Initial exploration rate |
| `epsilon_min` | 0.05 | Minimum exploration rate |
| `k_neighbors` | 7 | Number of neighbors for kNN interpolation |
| `gamma` | 0.9 | Discount factor |

### SAC Configuration
| Parameter | Initial Value | Description |
|-----------|--------------|-------------|
| `lr_actor` | 1e-5 | Actor learning rate |
| `lr_critic` | 3e-4 | Critic learning rate |
| `batch_size` | 256 | Batch size |
| `buffer_capacity` | 100,000 | Replay buffer size |
| `bc_lambda_init` | 5.0 | BC regularization weight |
| `bc_lambda_decay` | 0.999999 | BC regularization decay |
| `alpha_type_init` | 0.2 | Initial action type entropy |
| `alpha_param_init` | 0.2 | Initial parameter entropy |

### Curriculum Levels
| Level | Distance Range | Description |
|-------|---------------|-------------|
| Level 0 | 10–40 mm | Close targets |
| Level 1 | 20–80 mm | Medium targets |
| Level 2 | 40–120 mm | Far targets |

### Episode Pools
- **Train:** 3 seeds × 5000 episodes × 3 levels
- **Eval:** 3 seeds × 500 episodes × 3 levels
- **SAC Eval:** 3 seeds × 500 episodes × 3 levels

---

## 2. Metrics and Their Interpretation

### Core Metrics

| Metric | Description | Good | Bad |
|--------|-------------|------|-----|
| **success_rate** | Fraction of episodes where agent reached goal (distance < 5mm) | > 0.70 | < 0.50 |
| **timeout_rate** | Fraction of episodes where agent didn't reach goal within 150 steps | < 0.20 | > 0.50 |
| **collision_rate** | Fraction of episodes where agent passed through the surface | < 0.05 | > 0.10 |
| **rolling_rate** | Rolling success_rate over last 200 episodes | Stable or rising | Falling |

### SAC Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **alpha_type** | Entropy coefficient for action type | Controls type selection diversity. Too low (< 0.01) → cycling. Too high (> 2.0) → chaos |
| **alpha_param** | Entropy coefficient for parameters | Controls parameter noise. Too high (> 1.0) → imprecise movements |
| **bc_lambda** | BC regularization weight | Keeps policy close to expert demonstrations. Decays over time |
| **best_eval** | Best eval success rate during training | Model is saved based on this metric |

### Q-store Metrics (HNSWStateStore)

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **num_points** | Number of points in HNSW graph | Volume of accumulated experience. More → better state space coverage |
| **q_magnitude_mean** | Mean absolute Q-value | Strength of estimates. Low → little experience, high → confident estimates |
| **q_spread_mean** | Mean max-min Q-value difference per point | How well Q-store distinguishes actions. Low (< 1.0) → all actions equal, high (> 5.0) → clear preferences |
| **visits_mean** | Mean number of visits per point | How often agent visits similar states |
| **visits_median** | Median visits | More robust estimate. Low median with high mean → a few points visited very frequently, rest rarely |
| **update_hit_rate** | Fraction of updates to existing points vs new insertions | High (> 0.8) → agent in familiar regions. Low → constantly new states |
| **nn_distance_median** | Median distance to nearest neighbors | Coverage density. Low → dense, high → sparse |

### Arbitration Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **q_store_rate** | Fraction of decisions from Q-store | High → agent in familiar territory |
| **sac_rate** | Fraction of decisions from SAC | High → agent in unfamiliar states |
| **heuristic_rate** | Fraction of decisions from heuristics | High → both sources uncertain |
| **agreement_rate** | How often Q-store and SAC propose the same action | Low (< 0.3) → sources provide different information, arbitration is meaningful |

---

## 3. Training Iterations: Problems and Solutions

### Iteration 1: HNSW State Store Testing
Described in a separate document.

### Iteration 2: Q-learning with Curriculum

**What we did:** Trained a Q-learning controller with heuristic-guided exploration on the mug. Curriculum: 3 difficulty levels (10–40mm → 20–80mm → 40–120mm).

**Result:** Training
- success rate: 41%
- timeout_rate: 55%
- collision_surface_violation_rate: 4%

The agent learned to crawl along the surface toward the goal but got stuck when the goal was on the other side of the object, and episodes ended due to step limit. However, it maintained low collision rates.

**Problem: agent cannot fly to the other side of the object.**
When the goal is "behind" the surface (alignment < 0), the agent crawls into a dead end — the surface leads away from the goal, not toward it.

**Solution:** Added macro-action **Detach** — the agent lifts off the surface, flies through the air toward the goal, and lands. Two variants: `detach` (direct flight) and `detach_edge` (edge traversal).

**Effect:** Success rate increased from 40% to 60%, BUT collisions increased significantly.
- success_rate: 60%
- timeout_rate: 0%
- collision_surface_violation_rate: 40%

---

### Iteration 3: Reducing Collisions

**Problem: high collision_rate (40%).**

The agent frequently passed through the surface when using `free_forward` on the surface.

**Solution:**
1. Reduced `free_step` from 10mm to 5mm — smaller step, lower probability of passing through
2. Added reward penalty for `free_forward` on surface (`-2.0`) and (`-15.0`) for collision_surface_violation
3. Added `damp_free_on_surface` heuristic — suppression of free_forward/backward when agent is on surface (`-3.0`)

**Effect:** Collision rate decreased from 40% to 30% on initial run and down to 5% after several training iterations. Meanwhile success_rate reached 75%.
Details below:

TRAIN SEED = 11
start eps=1
- goal_reached: 0.6466
- timeout: 0.0476
- collision_surface_violation: 0.3058

start eps=0.3
- goal_reached: 0.6444
- timeout: 0.1062
- collision_surface_violation: 0.2494

start eps=0.15
- goal_reached: 0.6648
- timeout: 0.1532
- collision_surface_violation: 0.182

EVAL SEED 44
start eps=0.02
- goal_reached: 0.754
- timeout: 0.194
- collision_surface_violation: 0.052

**Problem:**
Q recommends detach 48% of steps — inflated. Softmax follows Q (98% blend). Q(detach) dominates due to large progress reward.

**Solution:**
75% with frequent detach — works. Detach as the primary "jump to goal" strategy is effective. Suboptimal in step count but reaches the goal. Sufficient for prototype. Frequency optimization — next stage.

---

### Iteration 4: BC Data Collection (DAgger Approach)

**Problem: BC data is imbalanced.**
With `eval_epsilon=0.02`, the agent almost always uses Q-store → data contains only what Q-store already knows.
Q recommends detach 48% of steps — inflated. Problem: Q(detach) dominates due to large progress reward.

**Solution:** Applied a balanced approach — added data with `eval_epsilon=1.0` without decay and temperature=0.01 (pure heuristics as expert). Success rate dropped noticeably to 48%, but heuristics provide more balanced data: detach when needed and without multiple consecutive repeats. Heuristic recommends detach 15% — more reasonable.

**Result:** Combined data from two modes:
- `eval_epsilon=0.02`: 5002 transitions (Q-store experience)
- `eval_epsilon=1.0`: 6349 transitions (heuristic experience)
- **Total: 11351 transitions**

Action distribution: MoveTangentially 23%, Detach 25%, Turn 38%, Look 6%, MoveLinear 8%.

---

### Iteration 5: Behavioral Cloning

**What we did:** Trained a BC model (supervised learning) on 11351 transitions.

**Result:**
- Train accuracy: 85.7%
- Val accuracy: 82.6%
- Early stopping at epoch 133 (val_loss=0.6493)
- Predictions close to expert: `Expert: 225° → Predicted: 209°`

BC model is used as warm-start for SAC.

---

### Iteration 6: SAC v1 — First Run and Overfitting

**What we did:** Launched SAC with BC warm-start, curriculum, 2000 episodes.

**Training result:**
- Train rolling rate: up to 0.715 (peak at episode ~1200)
- Degradation after episode 1200: rolling dropped from 0.715 to 0.560

**Eval result (predict/argmax):**

| Level | Success | Timeout | Collision |
|-------|---------|---------|-----------|
| Level 0 (10-40mm) | 33.2% | 66.2% | 0.6% |
| Level 1 (20-80mm) | 33.6% | 64.8% | 1.6% |
| Level 2 (40-120mm) | 30.4% | 68.6% | 1.0% |

**Catastrophic overfitting: train 71.5% vs eval 33%.**

**Diagnosis — causes:**

1. **Sample vs predict:** The 2.5× gap is explained by SAC being trained with softmax exploration. Training uses sample (softmax), Eval uses predict (argmax). Argmax cycles on one action, softmax alternates actions proportional to network confidence.

2. **Alpha_type collapse (0.199 → 0.007).** Policy became nearly deterministic. During training this was compensated by `sample()` stochasticity, but during eval with `predict()` (argmax) the agent cycled on one action.

3. **BC transitions evicted from buffer.** Buffer capacity = 100K. BC data (~10.7K) loaded first. By episode 1300 the buffer filled up, BC data began being evicted by new experience. The agent "forgot" expert behavior.

---

### Iteration 7: SAC Fixes

**Change 1: Created two eval options: softmax, argmax.**

**Change 2: BC data protection in buffer.**
Replay buffer rewritten with BC reservoir. 15% of buffer (at 500K = 75K slots) is protected — online data writes only to zone `[bc_size, capacity)`, BC zone is never overwritten.

**Change 3: Alpha floor constraint.**
`MIN_LOG_ALPHA_TYPE = -2.0` → alpha_type ≥ 0.135 (was 0.007). Policy maintains minimum stochasticity, doesn't collapse to determinism.

**Change 4: Buffer size increase.**
`buffer_capacity`: 100K → 500K. BC data preserved longer, more diversity.

**Change 5: Warmup via BC policy.**
First 5000 steps use BC policy (sample) instead of random actions. Buffer isn't polluted with garbage transitions.

---

### Iteration 8: SAC Final Run

**All fixes applied.** Training 2000 episodes on fixed pools.
**Eval on fixed pools (500 episodes × 3 levels, seed=77):**

| Level | sample (softmax) | predict (argmax) |
|-------|-----------------|-----------------|
| Level 0 (10-40mm) | **77.0%** | 32.4% |
| Level 1 (20-80mm) | **75.0%** | 33.8% |
| Level 2 (40-120mm) | **71.0%** | 30.8% |

**Sample vs predict:** Eval predictably works — sample (softmax) is used for inference. Problem solved.

**Training dynamics:**
```
alpha_type:  0.193 → 0.135 (stable at floor)
alpha_param: 0.207 → 1.000 (stable at ceiling)
rolling:     0.670 → 0.715 (stable, no degradation)
best_eval:   0.760 (episode 1400)
```
**Model doesn't degrade** — rolling rate 0.69–0.74 throughout training.

---

### Iteration 9: Q-store + SAC Arbitration

**What we did:** Combined Q-store (episodic memory) and SAC (skill) through an arbitrator. Implemented three operating modes with automatic switching.

**Arbitration logic:**
1. Q-store confident **AND** distinguishes actions (q_spread > 1.0) → Q-store
2. SAC confident → SAC
3. Q-store weakly confident → Q-store (fallback)
4. Nobody confident → Heuristic

#### AdaptiveTrainingManager Modes

The manager automatically switches mode based on rolling success_rate (window of 100 episodes):

| Mode | Condition | What Happens |
|------|-----------|--------------|
| **inference_only** | success_rate ≥ 80% | Work only, no training. Epsilon fixed at minimum |
| **online** | 60% ≤ success_rate < 80% | Work + background Q-store and SAC fine-tuning |
| **offline** | success_rate < 60% | Full retraining cycle: Q-learning → BC → SAC |

#### Result (1000 episodes on mug) with retraining modes disabled

| Metric | Value |
|--------|-------|
| Success rate | **78%** |
| Q-store decisions | 59% |
| SAC decisions | 41% |
| Heuristic decisions | 0% |
| Agreement Q↔SAC | 31.5% |
| Online SAC updates | 5 (every 200 episodes) |

**Action distribution:**

| Action | Q proposes | SAC proposes | Q chosen | SAC chosen |
|--------|-----------|-------------|----------|------------|
| detach | 39% | 58% | 29% | 74% |
| move_tangentially | 32% | 17% | 35% | 12% |
| turn_left | 10% | 6% | 15% | 2% |
| turn_right | 9% | 7% | 15% | 2% |
| free_forward | 5% | 8% | 4% | 5% |

**Observations:**
- SAC strongly prefers detach (58%), Q-store is more balanced (39%)
- Q-store more often selects surface crawl and turns — fine navigation
- Agreement 31.5% — sources provide different information, arbitration is meaningful

**"Proposes" vs "Chosen":**

At each step **both** sources propose an action, but the arbitrator selects only **one**:

```
Step 42:
  Q-store proposes: move_tangentially (→ recorded in "Q proposes")
  SAC proposes:     detach            (→ recorded in "SAC proposes")
  
  Arbitrator decides: Q-store confident + spread high → choose Q-store
  
  Executed: move_tangentially          (→ recorded in "Q chosen")
  SAC ignored                          (→ nothing recorded in "SAC chosen")
```

Therefore "proposes" is what each source **would like** to do (100% of steps for each), while "chosen" is what **actually executed** (only when the arbitrator selected that source).

**What the difference shows:**

| Action | Q proposes | Q chosen | Shift | Interpretation |
|--------|-----------|----------|-------|----------------|
| detach | 39% | 29% | −10% | Arbitrator often rejects Q-store's detach in favor of SAC |
| move_tangentially | 32% | 35% | +3% | Q-store more often wins arbitration when proposing crawling |
| turn_left/right | 19% | 30% | +11% | Q-store almost always wins on turns — SAC rarely proposes them |

| Action | SAC proposes | SAC chosen | Shift | Interpretation |
|--------|-------------|------------|-------|----------------|
| detach | 58% | 74% | +16% | SAC wins arbitration precisely when proposing detach |
| move_tangentially | 17% | 12% | −5% | When SAC proposes crawling, Q-store often overrides |
| turn_left/right | 13% | 4% | −9% | SAC almost never wins on turns |

**Conclusion:** Arbitration creates specialization — Q-store handles fine navigation (crawling, turns), SAC handles strategic decisions (detach, flights). Each source wins at what it does best.

Current arbitration thresholds:

| Threshold | Value | Set | Description |
|-----------|-------|-----|-------------|
| `q_confidence_threshold` | 0.5 | Manually | Minimum Q-store confidence for priority selection |
| `q_spread_threshold` | 1.0 | Manually | Minimum Q-value spread (does store distinguish actions) |
| `sac_confidence_threshold` | 0.3 | Manually | Minimum SAC confidence (max type probability) |
| `q_weak_threshold` | 0.2 | Manually | Threshold for weak Q-store fallback |

All thresholds were selected empirically based on observed confidence and spread distributions in mug experiments. When transitioning to other objects or changing network architecture, thresholds may require re-tuning.

In the future, arbitration can be made adaptive — for example, training a meta-controller that selects the source based on each source's current success rate, or using a multi-armed bandit approach for automatic trust distribution between Q-store, SAC, and heuristic.

#### Online Fine-tuning (Primary Mode)

In online mode, the system simultaneously operates and learns:

**Q-store updates every step:**
```
Every step → controller.update_only()
  → TD-update: Q(s,a) += α × (reward + γ × max Q(s') - Q(s,a))
  → If state is new → insertion into HNSW graph
  → If state is familiar → update existing point
```

**SAC updates every 200 episodes:**
```
1. Collected transitions are normalized (state, params)
2. Added to SAC replay buffer (online zone, BC reservoir untouched)
3. 50 gradient descent steps:
   - update_critic: train Q-function on batch from buffer
   - update_actor: train policy + BC regularization
   - soft_update_target: smooth target network update (τ=0.005)
```

**BC regularization every 2000 episodes:**

During extended online operation, the SAC actor gradually drifts from the originally trained policy — each `update_actor` shifts weights toward Q-value maximization, and over time the actor may "forget" basic skills from BC. Periodic BC regularization counteracts this:

```
1. Take 20 batches from replay buffer (including protected BC zone)
2. For each batch call update_actor, which contains:
   - SAC loss: Q-value maximization (primary objective)
   - BC loss: CrossEntropy(predicted_type, expert_type) 
            + MSE(predicted_params, expert_params)
   - Total: actor_loss = sac_loss + bc_lambda × bc_loss
3. BC buffer zone (15%) guarantees that each batch 
   contains expert demonstrations
```
This is a gentle reminder — 20 steps don't radically change the policy but prevent gradual degradation of basic skills (surface crawling, correct turns) that were established during the Behavioral Cloning stage.

BC data is **not updated**. BC data is a fixed set of expert demonstrations collected once during STEP 3.

The term "BC regularization" means not retraining the BC model, but using BC data as an **anchor** when updating the SAC actor:

```
Replay Buffer (500K):
┌─────────────────────────────────────────────────┐
│ BC zone (15%, protected)                        │
│ ~10.7K transitions from expert                  │
│ Never overwritten                               │
├─────────────────────────────────────────────────┤
│ Online zone (85%)                               │
│ New experience from arbitration                 │
│ Overwritten cyclically                          │
└─────────────────────────────────────────────────┘
```

When `buffer.sample(batch_size=256)` is called, the batch contains transitions from **both zones** — expert (BC) and fresh (online). The actor trains on this mixed batch:

- On expert examples: "don't forget how the expert did it" (BC loss)
- On fresh examples: "improve policy by Q-value" (SAC loss)

The ratio of BC to online data in the batch is determined by their proportions in the buffer. With 10.7K BC and 100K online — approximately 10% of the batch will be expert examples. This is sufficient to prevent actor drift from basic skills.

#### Offline Retraining (Emergency Mode)

Triggered when success_rate drops below 60% — the system cannot handle the current object:

```
1. Full Q-learning from scratch (5000 episodes)
   └── New controller, epsilon_start=0.3, curriculum levels
   
2. Collect successful trajectories → BC data
   └── ExperienceExtractor converts to PSACTransition

3. Train BC model (200 epochs)
   └── Supervised learning: state → (action_type, action_params)

4. SAC training with BC warm-start (10000 episodes)
   └── Full cycle: curriculum, alpha tuning, eval checkpointing

5. Replace current SAC with new one
   └── Arbitrator begins using updated model
```

---

## 6. Transfer: Mug → Cup

### Experiment Conditions
- **Training:** mug (mug.stl) — cylinder Ø60mm × 80mm + handle R=15mm
- **Test:** cup (cup.stl) — cylinder Ø56mm × 70mm + handle R=12mm
- **Fine-tuning:** none (zero-shot transfer)
- **Eval:** 500 episodes × 3 levels, seed=77

### Zero-shot Transfer Results

| Level | SAC zero-shot | Arbitrage zero-shot | Difference |
|-------|--------------|-------------------|------------|
| Level 0 (10-40mm) | **73.4%** | 59.4% | -14.0% |
| Level 1 (20-80mm) | **74.8%** | 56.6% | -18.2% |
| Level 2 (40-120mm) | **70.8%** | 51.6% | -19.2% |

### Arbitration Statistics on Cup

| Metric | Mug (familiar) | Cup (new) |
|--------|----------------|-----------|
| Q-store decisions | 59% | 45% |
| SAC decisions | 41% | 55% |
| Agreement Q↔SAC | 31.5% | 39% |
| Q-spread | 15.1 | 5.8 |

### Conclusions

**1. SAC generalizes better than Q-store.**
SAC zero-shot on cup: 70–75% — only 2–4% lower than on mug (71–77%). The neural network learned general navigation patterns (detach, crawl toward goal) that transfer to similar geometry.

**2. Q-store interferes on new objects.**
Arbitration (51–59%) is worse than pure SAC (70–75%) on the cup by 14–19%. Q-store from the mug confidently gives incorrect advice on the cup — it memorized specific mug states that don't match cup geometry.

**3. Arbitrator correctly shifts toward SAC.**
On the cup, SAC receives 55% of decisions (vs 41% on mug), Q-store — 45% (vs 59%). Q-spread dropped from 15.1 to 5.8 — Q-store is less confident on the unfamiliar object. The arbitrator reacts correctly but not aggressively enough.

**4. Agreement increased (31.5% → 39%).**
On the cup, Q-store and SAC agree more often — likely in simple situations (close goal, direct path) both give the correct answer, while disagreements arise in complex cases where Q-store errs.

### Recommendations for Improving Transfer

**1. Adaptive mode as the target solution.** Current fixed arbitration thresholds are an experimental proof of concept. The target solution is adaptive mode with online fine-tuning, where Q-store starts from scratch on a new object and gradually gains confidence through TD-update at each step. As experience accumulates, `q_confidence` and `q_spread` grow naturally, and the arbitrator automatically begins trusting Q-store more — without manual threshold tuning.

**2. SAC as initial skill on new objects.** At the start of working with a new object, Q-store is empty → `q_confidence ≈ 0` → arbitrator automatically selects SAC. SAC provides baseline success rate ~70% through generalized skills. As Q-store accumulates experience on the specific object, it begins intercepting decisions in familiar states, improving accuracy.

**3. Expected adaptation dynamics on a new object:**

| Phase | Episodes | Q-store | SAC | Expected success rate |
|-------|---------|---------|-----|----------------------|
| Start | 1–100 | Empty, not participating | 100% decisions | ~70% (zero-shot) |
| Accumulation | 100–500 | Gaining points, low spread | Dominates | 70–75% |
| Adaptation | 500–1000 | Confident in familiar regions | Supplements in new ones | 75–80% |
| Stability | 1000+ | Dominates on familiar | Fallback on unfamiliar | 80%+ |

**4. Approach validation.** Run STEP 10 (adaptive on cup with Q-store from mug + online fine-tuning). Q-store from the mug contains partially relevant experience — similar geometry but different proportions. Through TD-update at each step, Q-values will be corrected for the new object: points with incorrect Q-values will receive updates, new points will be inserted for cup-specific regions. Expected success rate growth from 55% (zero-shot) to 70–80% over 500–1000 episodes — faster than from an empty Q-store, thanks to transfer of partially applicable experience.

### Additional Run on Modified Cup (same dimensions as mug but different shape)
Cup SAC zero-shot:
- level_0 ([10.0, 40.0]mm): success=0.7680, timeout=0.1860, collision=0.0460
- level_1 ([20.0, 80.0]mm): success=0.7180, timeout=0.2280, collision=0.0540
- level_2 ([40.0, 120.0]mm): success=0.7220, timeout=0.2380, collision=0.0400

Transfer Comparison: Mug → Cup:

| Level | SAC zero-shot | Arbitrage zero-shot | Difference |
|-------|--------------|-------------------|------------|
| level_0 (10.0, 40.0mm) | 0.768 | 0.624 | -0.144 |
| level_1 (20.0, 80.0mm) | 0.718 | 0.484 | -0.234 |
| level_2 (40.0, 120.0mm) | 0.722 | 0.466 | -0.256 |