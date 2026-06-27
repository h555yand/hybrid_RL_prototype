# Summary

## This is a prototype to implement, test and proof ideas below.

Replace the `JumpToGoalState` mixin in Monty's motor system with a model-free reinforcement learning (RL) agent that learns to navigate incrementally toward goal states provided by Learning Modules. Instead of teleporting the sensor to a target pose, the RL agent selects from existing Monty actions to move step-by-step toward the goal, learning from dense reward signals based on distance reduction.

My idea is to take the best practices and bring them closer to how the brain works.
Most similar in a practical sense is a hybrid of solutions:
### Episodic Memory  
- **Situation**: 'I've been in a similar situation before. What did I do then? What happened?'  
- **Algorithm**: HNSW + kNN + Gaussian Kernel Interpolation  
- **When to use**: At the begining of learning, novelty, rare / important events  
- **Characteristics**: One-shot / few-shot learning, Fast activation by similarity, High locality, poor generalization.

### Habits / skills  
- **Situation**: 'I've done this action many times under these conditions — it usually works well.'  
- **Algorithm**: Soft Actor-Critic (SAC) — parametric policy/value  
- **When to use**: During routine, automated actions, stable conditions  
- **Characteristics**: Slow learning over many repetitions, Generalization is highly effective

### Algorithm of arbitration between systems: when to trust memory and when to trust the network  
This is not a separate policy, but a mechanism for switching between behavior modes.

# Motivation

## Current Problem

The hypothesis-testing policy in Monty's Evidence Learning Module generates goal states — target poses where the sensor should move to gather disambiguating evidence about object identity. Currently, these goals are enacted by the `JumpToGoalState` mixin, which teleports the agent instantaneously to the target pose using `SetAgentPose`.
This teleportation approach has fundamental limitations as instantaneous teleportation has no biological or robotic analog. A real agent must navigate through space incrementally with collision awareness.

## Why kernel-based Q-learning with episodic memory (HNSW + kNN + Gaussian Kernel Interpolation)

### Biological Plausibility

The chosen approach (kernel-based Q-learning with episodic memory) has parallels to hippocampal memory systems:

- **Episodic storage**: Individual experiences stored as state-value pairs (analogous to hippocampal episode encoding)
- **One-shot / few-shot learning**: It is enough for a person to do something once in order to act similarly in a similar situation
- **Pattern completion**: KNN retrieval from partial state matches (analogous to hippocampal pattern completion)
- **Kernel generalization**: Smooth interpolation across similar experiences (analogous to memory generalization during retrieval)
- **Non-parametric**: No fixed-size weight matrix; memory grows with experience (analogous to ongoing hippocampal neurogenesis)

This aligns with Monty's broader goal of biologically plausible computation.

### Theoretical publications
- Ormoneit & Sen's (2002) 'Kernel-Based Reinforcement Learning' proposed and analyzed a variant of Q-learning for large state spaces, in which the Q-function is approximated by kernel regression on the data rather than by a neural network or table. A key contribution is the conditions under which such a 'kernel Q-learning' scheme converges.
- Blundell et al. (2016) 'Model-Free Episodic Control'. The agent remembers its past successful actions and returns and, when faced with a similar state, simply reproduces the best of what has already worked.

### HNSW + kNN + Gaussian Kernel Interpolation

Hierarchical Navigable Small World (HNSW) is an approximate nearest neighbor search algorithm based on a layered graph data structure. It belongs to the family of proximity graphs, where nodes (vertices) are connected based on their proximity, typically measured by the Euclidean distance.  
HNSW is currently actively used in embedding databases for searching similar text by vectors. So, I decided to use it to store and find states.  [What is State](#state-vector-13d)  
During the learning process, the agent stores experience in a graph and then uses the weighted past experience in a similar situation. Thehnically it looks like: store point in HNSW graph, then find the K closest ones and mix them with Gaussian kernel weights.

### Why not only Deep Learing
I'm not opposed to deep learning. I agree that it's well suited for approximation, embedding, and many other tasks.  
I'm not suggesting replacing neural networks, I'm suggesting supplementing it and improving the learning process.

## You can read README_old.md for details and questions before POC was implemented
[LINK](README_old.md)

# Guide-level explanation

##  Main Proof of Concept Results

The prototype has been implemented and tested on the Lightweight Environment (Trimesh-based). Below are the key results that I hope validate the ideas from the RFC.

### Training Pipeline Validated

The full pipeline works end-to-end: **Q-learning → Behavioral Cloning → SAC → Arbitrage**

| Stage | Method | Success Rate | Object |
|-------|--------|-------------|--------|
| Q-learning (episodic memory) | HNSW + kNN + Heuristic-Guided Exploration | 65–70% | Mug |
| SAC (skill, softmax policy) | BC warm-start → SAC | 71–77% | Mug |
| Arbitrage (memory + skill) | Q-store + SAC + Heuristic | **78%** | Mug |
| SAC zero-shot transfer | No retraining | **73%** | Cup (new object) |

### Key Findings

**1. Episodic Memory (HNSW State Store) works as proposed.** One-shot/few-shot learning through Q-update is effective. Update hit rate 0.84–0.95 confirms that kNN + Gaussian Kernel interpolation correctly finds and updates similar states. The store grows organically — dense where the agent visits often, sparse where it rarely goes.

**2. Heuristic-Guided Exploration significantly outperforms ε-greedy.** This confirms the hypothesis from the RFC. Pure geometric heuristics (move toward goal, detach when stuck, steer in air) provide reasonable behavior from step 1, and Q-values gradually take over as experience accumulates.

**4. Arbitrage between episodic memory and skills works.** On familiar objects, Q-store handles 59% of decisions (precise, local knowledge), SAC handles 41% (general strategy). Agreement rate 31% shows they provide complementary information.

**5. On new objects transfer limitation discovered.** Q-store from one object can hurt performance on a new object (arbitrage zero-shot: 52% vs pure SAC: 73%).  This is expected — episodic memory is local by nature.  SAC generalizes better on new objects. This validates the skill-based approach — the neural network learns general navigation patterns that transfer across similar geometries. One more SAC advantage is continues parameters.
The solution can be:
- Start with pure SAC on new objects and let Q-store accumulate fresh experience.
- Use Online / Offline learning approach during working with new objects. If success rate below online threshold during 100 episodes to start online learning in parallel, if below offline threshold to start offline and after finish to continue working with new object.
---

## Answers to Open Questions

### Q1: Optimal State Dimensionality

I started with 13D and expanded to **15D** during testing. The additional features (mean curvature, Gaussian curvature) improved surface navigation.
The state vector is action-space independent — it describes the agent's situation relative to the goal, not the specific actions available.

### Q2: Hyperparameter Sensitivity

Extensive empirical testing was performed. Key findings:
- **Split the HNSW Q store** into two different graphs:
`q_store_free` — airborne state (on_object = 0). Navigation: turns, forward, landing
` q_store_surface` — surface state (on_object = 1). Crawling, detaching, orientation
The same position in space requires **opposite strategies** depending on whether the agent is touching the surface
- **Reward weights:** Increase penalty for collision from -5 to -15 
- **Action steps:** `surface_step=3mm`, `free_step=5mm`, `rotation_step=5°` — smaller steps reduce collisions but increase episode length.
- **SAC alpha bounds:** Critical for stability. `alpha_type ≥ 0.135` prevents policy collapse, `alpha_param ≤ 1.0` prevents parameter noise explosion.
- **Replay buffer:** BC-protected reservoir (15%) prevents catastrophic forgetting of expert demonstrations.
- **At the beginning I used 18 actions and then 2 macro actions were added**
| Index | Action | Category | Description |
|-------|--------|----------|-------------|
| 18 | Detach | macro | Detach from surface along normal, then fly toward goal (blending goal direction with surface normal to avoid collision). Multi-step action: lift off along normal → turn toward goal → series of forward steps with depth checking and collision avoidance → land on contact with surface. Each sub-step incurs a step penalty. Collisions during flight are penalized but do not terminate the episode — the agent is repositioned to pre-collision location and the macro action terminates early |
| 19 | DetachEdge | macro | Detach from surface along normal, then fly upward along the object's longest bounding-box axis until reaching the edge, turn toward goal and fly over to the other side. Used when the goal is through a thin wall (agent and goal normals are opposite). Same penalty and collision handling as Detach |

The Lightweight Environment (Trimesh) proved essential for rapid iteration — each experiment takes ~ 30-60 minutes on my laptop)

---

### On action space independence

> "Try and make the solution one that is action space independent"

**Status: Partially addressed.**
- `ActionSpace` is a configurable input parameter — adding or removing actions does not require architectural changes to Q-learning or SAC
- HNSW and SAC work with any number of discrete actions
- The system was tested with 18 actions (without detach) and 20 actions (with detach), confirming that the core learning pipeline adapts

**Limitation:** Heuristic biases currently reference specific action indices (e.g., `IDX_DETACH`, `IDX_FREE_FORWARD`). For a different action space, heuristics would need to be adapted. This is by design — heuristics encode domain-specific geometric reasoning that depends on what actions are available. A fully action-space independent heuristic system would require a mapping layer between geometric intentions (e.g., "move toward goal") and available actions.

### On the need for HNSW if good heuristics exist

> "If we have good heuristics to generate initial action sequences, what is the additional need for storing and retrieving them?"

**Answer from experiments:** Heuristics alone achieve ~48% success rate. 
Of course they can be improved, but it's continues effort to tune them for new cases.. HNSW stores the **learned corrections** — situations where heuristics were wrong and Q-learning motivated to discovery better actions. After training, Q-store + heuristics achieve 65–70%, and with SAC added, 78%.

The key insight: heuristics provide the **starting point**, episodic memory stores **exceptions and refinements**, and SAC learns **general patterns**. Each layer adds value.


Of course it's not enough to understand all things, I will prepare and share details later together with the source code. Now it works but it needs to review, comment, check one more time etc.
After that we can discuss details and next steps.

## Training strategy
### You can read files train.md in ./docs folder
[LINK train](docs/train.md)

## Architecture Overview

### You can read files design.md, HNSW_store.md in ./docs folder
[LINK design](docs/design.md)  
[LINK HNSW_store](docs/HNSW_store.md)

#### Below old materails that will be updated based on POC results

Evidence LM's Goal-State Generator proposes the goal-state from the hypothesis-testing policy.
**goal_pose** = [x, y, z, pitch, yaw, roll]  
   ↓  
**RLGoalApproachController** computes **State Vector** using **goal_pose** as well as sensory patch input and proprioceptive information.   
   ↓  
**At the begining for new states** it needs to learn before inference.  
   ↓  
**RL Q-leraning with HNSW + kNN + Gaussian Kernel Interpolation** working with discrete actions. [What is Action](#montyactionspace)    
Actions are selected using [Heuristic-Guided Exploration](#heuristic-guided-exploration) approach. Objective: To obtain smart behavior and training data for SAC.  
HNSW graph points collection (gathering experience). Copying of **successful traces into replay buffer**.
**HNSWStateStore** stores **State Vector**, actions, Q-values.
Upon reaching a certain threshold of successful validation operations moves to next step of learning.    
   ↓  
**Behavioral cloning (BC)** — method of imitation learning in which an agent learns to imitate the behavior of an expert by directly copying his actions based on data. BC is the simplest form of imitation learning: we teach the robot's policy as a supervised learning task—to predict the action of an expert based on observation. It can be used as independent approach but usually used as supplement before SAC or other off-policy RL algorithm.
Translation of HNSW 18D discrete action space from replay buffer into Parameterized SAC continuous action space. Replacing of 8 direction MoveTangentially with one action with two parameters: angle_deg, distance. Others actions are stays the same. Main defference will be during SAC training when paramters can be any value, not fixed config free_step, rotation_step, surface_step.
   ↓  
**Train a SAC Actor policy using supervised learning loss** to have a continuous policy that copying the behavior of a discrete policy.  
   ↓  
**Warm-start to run RL SAC training with Critic policy** as well using the Actor policy weights from the BC.
Train with a reward (progress toward the goal, penalty for collisions) in real or sim environment.
The SAC refines the policy: it makes movements smoother and more accurate, adapting to new scenes.
**Now the policy doesn't just copy of a discrete policy, it optimizes**.  
   ↓  
In future when SAC is trained we use it as **skills to propose continuous actions**  

## Advantages of this scheme:
- Quick start with Q-learning and discrete actions – no need to wait for the SAC to learn from scratch.
- Stability – Heuristic-Guided Exploration learns in new areas.
- Precision – then SAC makes movements smooth and optimal as skills.
- Biologically plausible – like human learning: first we copy, then we hone.


**Below is explanation of the main components**:

## State Vector (13D)
**Purpose**: To represent the current state of the agent relative to the goal. The state vector is a 13-dimensional vector that includes spatial and rotational errors, surface normal, and sensor information.
All spatial quantities are in the agent's local coordinate frame.

| Index | Feature | Description |
|----------|----------|----------|
| 0-2   | position_error [x, y, z]   | direction to goal in agent's local frame   |
| 3-5   | rotation_error [pitch, yaw, roll]   | orientation error (normalized angles)   |
| 6-8   | local_normal   | surface normal in agent's local frame   |
| 9   | on_object   | whether sensor on object surface   |
| 10   | alignment   | dot(goal_direction, surface_normal)   |
| 11   | distance   | Euclidean distance to goal   |
| 12   | norm_depth   | normalized depth to nearest surface   |


## HNSWStateStore
Update state → normalize → KNN search
→ if near existing point: update it
→ else: insert new point with interpolated init

Get state → normalize → KNN search → kernel interpolation → Q-values


## MontyActionSpace

To properly train hand, finger movement, it is suggested to use the combined action spaces (distant + surface) from the beginning.  When the finger is on the surface it plays like surface agent, when is on the air like distant. For me current split for distant and surface for finger looks a little bit artificial in RL.
Proposed action space is universal high level primitives for moving finger to the goal.
These primitives can be used for any other similar agents and tasks and they are not dependent from specific low level robotic actions (joints, speed, strength etc). 
Transfer primitives to low level robotic is a separate and well known task.
In the industry standard is a mix: the Neural Network chooses 'What to do' (high level primitives), and the classical algorithm (Inverse kinematics & Impedance) decides 'How exactly to move the motors'.

So I agree that action space is agent specific. In my vision agent is something real (arm, hand, finger) that has actions and skills.
Now we try to learn moving to target, then we can add new action and train new skill to grasp objects.

### How to use actions in RL step by step:
1. Q-learning and discrete actions  
The policy outputs an index from 0 to 17.  
Fixed directions, surface_step, free_step, rotation_step are used.  
2. Parameterized SAC (current proposal)  
The policy outputs: action index (0-11) and a continuous parameter instaed of fixed step.
Replacing of 8 direction MoveTangentially with one action with two parameters: angle_deg, distance. Others actions are stays the same.
3. Purely continuous SAC (future)  
The policy outputs a vector [Δx, Δy, Δz, Δθ, Δφ] and then interprets this as a combined motion.
4. Mathematical controller (Low-level / Inverse kinematics & Impedance)
This is 'spinal cord' that receives a command from the neural network (SAC) and instantly calculates the motor actions.

### Discrete action space 18D
| Index | Action               | Description                                                                 | Mode     | Parameters |
|--------|------------------------|-------------------------------------------------------------------------|-----------|-----------|
| 0–7    | MoveTangentially       | Movement tangent to the surface in 8 directions: 0°, 45°, ..., 315° | surface   | `distance: float`, `direction: VectorXYZ` |
| 8      | MoveForward            | Moving forward (in the direction the agent is looking)              | both       | `distance: float` |
| 9      | MoveForward (neg)      | Moving backward                                                          | both       | `distance: float` |
| 10     | TurnLeft               | Rotate the agent to the left (along the Y axis, yaw)                  | distant   | `rotation_degrees: float` |
| 11     | TurnRight              | Rotate the agent to the right                                            | distant   | `rotation_degrees: float` |
| 12     | LookUp                 | Tilt the agent/sensor up (pitch)                                      | distant   | `rotation_degrees: float` |
| 13     | LookDown               | Tilt the agent/sensor down                                              | distant   | `rotation_degrees: float` |
| 14     | SetSensorRotation (+)  | Rotate the sensor clockwise around the normal (yaw)                          | both       | `rotation_quat: Quaternion` |
| 15     | SetSensorRotation (-)  | Rotate the sensor counterclockwise                                          | both       | `rotation_quat: Quaternion` |
| 16     | OrientHorizontal       | Correction of position and orientation in the horizontal plane (with compensation) | surface   | `rotation_degrees: float`, `left_distance: float`, `forward_distance: float` |
| 17     | OrientVertical         | Correction of position and orientation in the vertical plane                | surface   | `rotation_degrees: float`, `down_distance: float`, `forward_distance: float` |



## RLGoalApproachController
Core RL logic — state computation, reward, collision detection, action selection with heuristic-guided exploration.

### Reward Function
Dense reward signal computed locally in the motor system (no LM or CMP involvement):
| Component | Reward | Done? | When |
|:----------|-------:|:-----:|:-----|
| Progress (per good step) | ~+3.0 | No | Every step; `(prev_dist - dist) / surface_step × 3.0` |
| Goal reached | +50.0 | Yes | `distance < goal_threshold (2mm)` |
| Step penalty | -0.2 | No | Every step |
| Surface violation | -5.0 | Yes | Agent passed through object (depth < 0.5mm or normal flipped) |
| Lost object (smart detach) | +0.5 | No | Lost surface but approaching goal with alignment < -0.3 |
| Lost object (drifted away) | -3.0 | No | Lost surface and moved away from goal |
| Near goal on surface | +0.5 | No | `distance < 3 × surface_step` AND `on_object = true` |
| Oscillation | -0.5 | No | Current action is opposite of previous action |
| Timeout | -10.0 | Yes | `steps >= max_steps_per_goal` |


### Heuristic-Guided Exploration

#### Problem with Standard ε-Greedy

Standard ε-greedy exploration selects random actions with probability ε. In a 18-action space, a random action has only small chance of being useful (moving toward the goal). This means the most of exploration steps are wasted, resulting in slow learning and poor initial behavior.

#### My Approach: Blending Q-Values with Heuristics

Instead of random exploration, blend learned Q-values with heuristic bias derived geometric reasoning:

```python
combined = (1 - ε) × Q_normalized + ε × heuristic_normalized
action = softmax_sample(combined, temperature=max(0.1, ε))
A small fraction (ε × 10%) of actions remain purely random to guarantee full action space coverage.
```
Transition Schedule

| Phase | Epsilon | Behavior |
|---|-----------|--------|
| Cold start | 1.0 → 0.5 | Nearly pure heuristic — reasonable from step 1 |
| Learning | 0.5 → 0.1 | Blend of Q-values and heuristic |
| inference | 0.1 → 0.05 | Nearly pure Q-values with light heuristic safety net |

###	Possible Heuristic examples that use pure geometry and action space parameters

| # | Heuristic | Description |
|---|-----------|--------|
| 1 | Move toward goal | surface - MoveTangentially, distant - MoveForward |
| 2 | Goal far → fly,  goal close → crawl | surface_crawl vs free |
| 3 | Goal through surface → detach | LookUp |
| 4 | In the air navigating by rot_error | TurnRight, TurnLeft, LookDown, LookUp | 



## RLMotorPolicy
Integration point with Monty. Extends existing motor policy, activates RL when LM sends goal, falls back to standard Monty behavior otherwise.
This means:
All existing Monty behavior is preserved: When no goal is active, the parent class handles exploration (surface crawl, curvature-informed steps, orient to surface, etc.)
No modifications to LMs: Goal states are read from LM attributes that already exist for JumpToGoalState
No modifications to CMP: Reward is computed locally from proprioceptive and sensor data
Graceful degradation: If RL fails (timeout/collision), control returns to standard Monty exploration

## Lightweight Enviroment
To fast test hypophesys and ideas we need to create relevant approximation of Habitat, especially for training policies based on haptics/active perception.
It should not simulate graphics, but it should accurately reproduces the key physics that are important for training: contact, normals, ray casting, movement on surfaces.
I suggest to use Trimesh python library for loading and using triangular meshes with an emphasis on watertight surfaces. https://github.com/mikedh/trimesh

### Obejscts
To train / test complete action space we can start with these primitives:
- Cube: trimesh.primitives.Box
- Cylinder: trimesh.creation.cylinder
- Sphere: trimesh.primitives.Sphere

Next step can be to train / test with more complex objects, cup as example.
It can be created from primitives or loaded from library. 

### Methods
To compute State 13D it needs to implement enviroment methods to get:
- Agent position
- Agent orientation
- Target point: random surface point
- Surface normal
- Depth
- Flag on object

### Actions
It needs to implement 18D action space

Trimesh provides many usefull methods so implementation of Lightweight Enviroment looks very realistic in reasonble time.


# Unresolved questions

## Open Questions
### Q1: Optimal State Dimensionality
Should we start with 13D or reduce by removing less informative features?
Is 13D enough to distinguish one state from another to train RL controller?

Several factors mitigate this:

1. **Effective dimensionality is lower**: Only 6-8 features strongly
   determine action choice. The remaining features provide refinement.

2. **States lie on trajectories**: Not randomly distributed in 13D,
   but along low-dimensional manifolds (movement paths).

3. **Feature weighting**: Critical features (distance, alignment,
   pos_error) can be upweighted in distance computation, effectively
   reducing dimensionality.

4. **Use Embeddings**: It needs a separate embedding module but we can increase dimensions without constaints risk   

### Q2: Hyperparameter and Config Sensitivity
How sensitive is the system to sigma, k_neighbors, and learning rate?
What are optimal parameters for reward weights, action steps, etc?

Current position: To be determined empirically during many tests.
Mitigation: It makes sense to develop Lightweight Enviroment with trimesh for mesh-based primitives simulation to standalone RL training without Habitat.


# Future possibilities

The current architecture is designed to support future addition of **model-based planning**. The HNSW Q-store serves as a single integration point — both model-free updates (from real experience) and model-based updates (from simulated planning) write to the same store.

World model planning would become valuable when Monty extends to:
- Object manipulation (irreversible actions)
- Active recognition (predicting sensor observations)
- Real robot deployment (expensive physical steps)
- Multi-agent coordination