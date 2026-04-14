# LifeEngine: Mathematical Reference

Comprehensive mathematical definitions of all reward, fitness, and calculation functions used in the learning-vs-evolution experiment.

---

## 1. Reward Function (Reinforcement Learning)

Used in **Condition A** to update neural network weights via REINFORCE with eligibility traces.

### Components

**Per-tick reward** $r_t$ is accumulated as:

$$r_t = r_{\text{food}} + r_{\text{decay}} + r_{\text{explore}}$$

Where:

- **Food reward**: When food is eaten:
  $$r_{\text{food}} = v_{\text{food}}$$
  - $v_{\text{food}} \in \{0.5, 1.0, 2.0\}$ (energy value depends on food type)

- **Decay penalty**: Each time energy decays (every 10 ticks):
  $$r_{\text{decay}} = -\text{DECAY\_PENALTY} = -0.1$$

- **Exploration bonus**: First visit to a grid cell:
  $$r_{\text{explore}} = +\text{EXPLORE\_BONUS} = +0.02$$
  - Only triggered once per unique cell location

### REINFORCE Algorithm

Weight update rule with eligibility traces (applied to `active_weights` only):

$$w_{idx} \leftarrow w_{idx} + \alpha \cdot r_t \cdot e_t[idx]$$

Where:
- $\alpha = \text{RL\_LR} = 0.02$ (learning rate: controls step size of weight updates)
- $r_t$ is the reward signal this tick (positive for good outcomes, negative for bad)
- $e_t[idx]$ is the eligibility trace for weight $idx$ (credit assignment: how much this weight contributed to the recent action)

**No Backpropagation**: This uses REINFORCE, a policy gradient method that updates weights **directly** from rewards without error backpropagation. The eligibility trace acts as a memory of which weights were responsible for recent actions, allowing learning even when rewards are delayed. This differs from supervised learning (which backpropagates errors) — here, the only signal is reward, and traces accumulate responsibility across the network.

### Eligibility Trace Update (No Backpropagation)

Traces accumulate responsibility for the taken action. They are updated via **forward-time propagation** during the forward pass and weight update, NOT via backward error propagation. They decay each tick, implementing a recency bias (recent actions matter more than distant ones).

For output layer weights (after computing action $a$):

$$e_{W2,k,j}^{(t)} = \gamma \cdot e_{W2,k,j}^{(t-1)} + \delta_k \cdot h_j$$

$$e_{b2,k}^{(t)} = \gamma \cdot e_{b2,k}^{(t-1)} + \delta_k$$

Where:
- $\gamma = \text{TRACE\_DECAY} = 0.90$: exponential decay factor (higher = longer memory)
- $\delta_k = \mathbb{1}[k = a] - \pi_k$ is the **temporal difference** (TD error). It's positive for the action taken (promoting it) and negative for alternative actions (discouraging them)
- $\pi_k$ is the output probability (softmax) — the policy's belief in action $k$
- $h_j$ is the hidden layer activation — cached during forward pass

For hidden and input layer weights:

$$e_{W1,j,i}^{(t)} = \gamma \cdot e_{W1,j,i}^{(t-1)} + \Delta_j \cdot x_i \cdot \mathbb{1}[h_j > 0]$$

$$e_{b1,j}^{(t)} = \gamma \cdot e_{b1,j}^{(t-1)} + \Delta_j \cdot \mathbb{1}[h_j > 0]$$

Where:
$$\Delta_j = \sum_{k=0}^{3} w_{W2,k,j} \cdot (\mathbb{1}[k = a] - \pi_k)$$

**How responsibility propagates**: $\Delta_j$ aggregates the TD error from all output neurons weighted by their connection to hidden neuron $j$. The indicator $\mathbb{1}[h_j > 0]$ reflects ReLU: only active hidden neurons (those that fired during forward pass, $h_j > 0$) can be credited or blamed. Dead neurons ($h_j \leq 0$) have zero trace. This is **not error backpropagation** — it's eligibility trace decay with forward-time credit assignment.

---

## 2. Fitness Function (Genetic Algorithm)

Used by GAManager to rank organisms for selection. **This is the primary selection signal steering evolution.**

$$\text{Fitness}_i = \text{cumulative\_food\_score}_i = \sum_{t=1}^{T_i} v_{\text{food}}(t)$$

Where:
- $T_i$ is the lifetime (in ticks) of organism $i$ (can vary due to energy depletion)
- $v_{\text{food}}(t)$ is the energy value of food eaten at tick $t$

**Why cumulative food score?** This metric directly measures an organism's ability to locate and consume food resources. It's independent of the RL reward signal (which guides learning within a lifetime), so we can compare Condition A and B fairly. Organisms that find more food survive longer and produce more offspring, allowing good food-seeking behaviors to spread through the population.

### GA Selection and Reproduction

**Top-5 Parent Selection**: After generation ends (all living organisms dead), sort all agents (founders + descendants) by fitness descending and select the top 5:

$$\{a_1^*, a_2^*, \ldots, a_5^*\} = \text{argsort}(\{\text{Fitness}(a_i)\})_{\text{descending}}, \quad \text{take first 5}$$

**Cross-generation reproduction**: For the next generation, create new offspring by uniform crossover between randomly chosen pairs from the top 5, applied to the entire genome (308 weights).

**Population Size**: 
$$\text{next\_generation\_size} = \text{POPULATION\_SIZE} = 1$$

Only 1 founding organism per generation (for controlled comparison). Within-generation asexual reproduction can increase numbers via the `food_collected` threshold.

---

## 3. Energy System

### Energy Decay

Energy decays over time, creating an ecological pressure to find food. The decay rate depends on the day/night cycle and the organism's location.

Base energy loss every $\Delta t = 10$ ticks:

$$e_t = e_{t-\Delta t} - d \cdot m$$

Where:
- $d = \text{ENERGY\_DECAY\_RATE} = 1$ (base loss per decay event)
- $m$ = **location and time-dependent multiplier**:

**Day-time (0-150 ticks in 225-tick cycle)**:
- Outside cave: $m = 1.0$ (normal exploration cost)
- Inside cave: $m = 2.0$ (caves are energy-draining during day, not a refuge)

**Night-time (150-225 ticks in cycle)**:
- Inside cave: $m = 0.0$ (caves provide safe sleep; energy frozen)
- Outside cave: $m = 2.0$ (dangerous to be outside at night)

**Ecological intuition**: The day/night-cave system creates temporal and spatial pressures. Organisms must learn to spend the day foraging and nights sleeping in caves. This rewards adaptive navigation and energy timing.

### Energy from Food

When food is eaten:

$$e_t \leftarrow \min(e_t + v_{\text{food}}, e_{\max})$$

$$\text{cumulative\_food\_score} \leftarrow \text{cumulative\_food\_score} + v_{\text{food}}$$

Where:
- $e_{\max} = 200$ (ENERGY_CAPACITY)
- $v_{\text{food}}$ depends on food type:
  - Low food: 0.5
  - Medium food: 1.0
  - Prestige food: 2.0

### Death Condition

Organism dies when:
$$e_t \leq 0 \quad \text{OR} \quad t \geq T_{\max}$$

Where $T_{\max} = 5000$ (MAX_LIFETIME)

---

## 4. Neural Network Policy

### Architecture

$$\text{Input} \to \text{ReLU}(W_1, b_1) \to \text{softmax}(W_2, b_2) \to \text{Action}$$

**Dimensions**:
- Input: $n_i = 33$ (4 eye directions × 8 percept types + 1 energy scalar)
- Hidden: $n_h = 8$ (ReLU)
- Output: $n_o = 4$ (up, right, down, left)

**Total weights**: $\text{GENOME\_SIZE} = 308$
- $W_1$: $33 \times 8 = 264$
- $b_1$: $8$
- $W_2$: $8 \times 4 = 32$
- $b_2$: $4$

### Activation Functions

**ReLU** (hidden layer): Introduces non-linearity, enabling the network to learn complex functions.
$$h_j = \max(0, z_j^{(1)}) = \max(0, \sum_{i=0}^{32} w_{i,j}^{(1)} x_i + b_j^{(1)})$$

ReLU zeros out negative activations, creating sparse representations (some neurons inactive). This is beneficial for RL because the eligibility traces only update active neurons.

**Softmax** (output layer): Converts raw logits into a valid probability distribution over actions.
$$\pi_k = \frac{e^{z_k^{(2)}}}{\sum_{k'=0}^{3} e^{z_{k'}^{(2)}}}$$

Where:
$$z_k^{(2)} = \sum_{j=0}^{7} w_{j,k}^{(2)} h_j + b_k^{(2)}$$

This ensures $\sum_{k=0}^{3} \pi_k = 1$ and $\pi_k \geq 0$, making $\pi$ interpretable as a policy (probability of each action).

### Action Selection (Epsilon-Greedy Exploration)

The network explores early in life and exploits learned behavior later, balancing discovery vs. execution.

**Greedy (exploitation)** (probability $1 - \epsilon$): Choose the highest-probability action
$$a = \arg \max_k \pi_k$$

**Exploratory (exploration)** (probability $\epsilon$): Sample randomly from the policy
$$a \sim \text{Categorical}(\pi)$$

Linear exploration decay over organism lifetime:
$$\epsilon(t) = \epsilon_{\text{start}} + (\epsilon_{\text{end}} - \epsilon_{\text{start}}) \cdot \frac{t}{T_{\max}}$$

With:
- $\epsilon_{\text{start}} = 0.2$ (20% random at birth — young organisms explore)
- $\epsilon_{\text{end}} = 0.05$ (5% random at death — older organisms mostly exploit learned behavior)
- $T_{\max} = 5000$ (decay happens over full lifespan)

**Effect**: Young organisms explore the world; as they age and have learned behavior, they increasingly rely on their learned policy.

---

## 5. Perception (Input State Vector)

### State Vector Construction

33-element input vector $x \in \mathbb{R}^{33}$:

**Eye observations** (positions 0-31): Scalar encoding per direction

For each of 4 cardinal directions $d \in \{\text{up, right, down, left}\}$:
$$x_{d \cdot 8 + c} = \text{scalar value for percept type } c$$

Where $c \in \{0..7\}$ (percept type indices):
- 0: empty / nothing (value: 0)
- 1: low food (energy 0.5)
- 2: medium food (energy 1.0)
- 3: prestige food (energy 2.0)
- 4: wall / obstacle (value: 0)
- 5: low food landmark (scalar representation)
- 6: medium food landmark
- 7: prestige food landmark

**Why scalar encoding?** Scalar representation is more efficient than one-hot and allows the network to learn continuous mappings between percept types. This is particularly useful for food types at different energy levels, where ordinal relationships (e.g., medium > low) might be learned by the network.

**Energy scalar** (position 32): Normalized energy level
$$x_{32} = \min\left(1, \max\left(0, \frac{e}{e_{\max}}\right)\right)$$

Where $e$ is current energy, $e_{\max} = 200$. Clamping to $[0, 1]$ normalizes the signal so the network can process it consistently regardless of energy scale.

---

## 6. Genetic Operators

### Uniform Crossover

Each weight independently inherited from parent A or B with equal probability. This is **robust for neural networks** because it avoids positional bias—every weight position has equal chance from either parent.

$$c_i = \begin{cases} a_i & \text{if } u_i < 0.5 \\ b_i & \text{otherwise} \end{cases}, \quad u_i \sim \text{Uniform}(0, 1)$$

Applied to entire genome (308 weights).

**Why uniform over single-point crossover?** With single-point crossover, nearby genes are always inherited together, which can propagate bad linkages. Uniform crossover decouples inheritance, allowing the GA to mix good building blocks from different parents more flexibly.

### Gaussian Mutation

**Between-generation** (disabled, $\text{MUT\_PROB} = 0$): Applied to all genomes entering the next generation after crossover. Introduces random variation to explore new regions of weight space.

$$w_i' = w_i + \begin{cases} \mathcal{N}(0, \sigma) & \text{if } u < p \\ 0 & \text{otherwise} \end{cases}$$

- $p = \text{MUT\_PROB} = 0.0$ (currently disabled)
- $\sigma = \text{MUT\_SIGMA} = 0.1$ (would be the perturbation magnitude if enabled)

**Within-generation** (asexual offspring, also disabled): Applied when asexual reproduction occurs mid-generation. Allows children to be slightly different from parents, but this is turned off to maximize learned behavior inheritance.

$$w_i' = w_i + \begin{cases} \mathcal{N}(0, \sigma) & \text{if } u < p \\ 0 & \text{otherwise} \end{cases}$$

- $p = \text{ASEXUAL\_MUT\_PROB} = 0.0$ (currently disabled)
- $\sigma = \text{ASEXUAL\_MUT\_SIGMA} = 0.1$ (unused when p=0)

**Note**: Both mutation rates are currently disabled so that all genetic variation comes from crossover and RL drift.

---

## 7. Metrics and Analysis Calculations

### Genetic Drift

Measures how much the learnable weights have diverged from the heritable genome due to within-lifetime RL.

$$\text{drift}_i = \frac{1}{M} \sum_{j=0}^{M-1} |w^{\text{active}}_{i,j} - w^{\text{genome}}_{i,j}|$$

Where $M = 308$ (genome size).

**Condition A** (RL-enabled): $\text{drift}_i > 0$ 
- Organisms learn (active weights drift from genome)
- RL updates accumulate over the lifetime
- But drift is not inherited (non-Lamarckian: genome stays frozen)

**Condition B** (RL-disabled): $\text{drift}_i = 0$
- `active_weights === genome_weights` (same reference, no separate copies)
- No learning, no drift
- Evolution must discover good policies through GA alone

**Interpretation**: High drift indicates strong learning during the organism's lifetime. Average drift per generation tells us how quickly organisms are learning and adapting within their lifetime.

### Genome L2-Norm

Scalar measure of weight magnitude:

$$\|w_i\|_2 = \sqrt{\sum_{j=0}^{M-1} w_{i,j}^2}$$

### Population Genome Variance

Measures genetic diversity in the population by checking how much weights vary across organisms.

**Average per-position variance** across all 308 weights:

$$\sigma^2_{\text{genome}} = \frac{1}{M} \sum_{j=0}^{M-1} \text{Var}\left(\{w_{i,j}^{\text{genome}}\}_{i=1}^{N}\right)$$

Where the variance at position $j$ is:
$$\text{Var}\left(\{w_{i,j}\}_i\right) = \frac{1}{N} \sum_{i=1}^{N} (w_{i,j} - \mu_j)^2$$

And the mean at position $j$ is:
$$\mu_j = \frac{1}{N} \sum_{i=1}^{N} w_{i,j}$$

**Interpretation**: 
- **High variance** ($\sigma^2 \gg 0$): Population is diverse; many different weight configurations 
- **Low variance** ($\sigma^2 \approx 0$): Population has converged; most organisms are similar (potential sign of premature convergence)
- **Trend over generations**: Variance should typically decrease as selection pressure narrows the population around fit solutions

**Computational note**: Sample up to 50 agents if population > 50 (expensive to compute across full population).

### Average Energy Metrics

**At 20% lifetime** (early sample):
$$\bar{e}_{\text{early}} = \frac{1}{N} \sum_{i=1}^{N} e_i(t=0.2T_i)$$

**At death/end**:
$$\bar{e}_{\text{end}} = \frac{1}{N} \sum_{i=1}^{N} e_i(T_i)$$

### Average Lifetime

$$\bar{T} = \frac{1}{N} \sum_{i=1}^{N} T_i$$

### Top-5 Fitness Average

$$\bar{f}_{\text{top5}} = \frac{1}{5} \sum_{i=1}^{5} f_i, \quad \text{where agents sorted by fitness descending}$$

---

## 8. Reproduction Thresholds

### Within-Lifetime (Asexual Reproduction)

Organisms can produce offspring _during_ their lifetime if they gather enough food. This increases the effective population size and creates a second form of selection (selection for reproduction ability).

**Offspring trigger condition**:

$$\text{food\_collected} \geq \text{body\_size} = |\text{anatomy.cells}|$$

Once this threshold is met, a single offspring is created, and `food_collected` is reset.

**Offspring spawning location** (deterministic, same for all offspring):
$$(\text{spawn\_col}, \text{spawn\_row}) = (c_{\text{GA}}, r_{\text{GA}})$$

Offspring spawn at the world center, the same location where all organisms begin. This keeps the experimental setup controlled and eliminates positional bias.

**Genome inheritance**: Non-Lamarckian
- Child inherits parent's `genome_weights` only
- Parent's RL drift (`active_weights`) is **discarded** — not passed to offspring
- Child's `genome_weights` may mutate slightly (mutation currently disabled: `ASEXUAL_MUT_PROB = 0`)

### Between-Generation (Sexual Crossover via GA)

Executed **only** when the entire lineage dies (no living organisms remain). This marks the end of a generation and seeds the next.

**Steps**:

1. **Collect all agents** that lived this generation (founders + all asexual descendants generated during lifetime)
2. **Sort by fitness** descending: $a_1^* \succ a_2^* \succ \cdots \succ a_N^*$ where $a_i^* = \arg \max_i \text{Fitness}(i)$
3. **Select top parents**: Keep the top $\min(5, N)$ parents for reproduction
4. **Generate next generation**: For each of the $\text{POPULATION\_SIZE} = 1$ founding slot:
   - Randomly sample two parents **with replacement** from the top-5: $p_A, p_B \sim \text{Uniform}(\text{top-5})$
   - **Uniform crossover**: Each weight has 50/50 chance of coming from $p_A$ or $p_B$
   - **Mutation**: Apply Gaussian perturbation to each weight (currently **disabled**: `MUT_PROB = 0`)
5. **Place seed**: New generation starts from `(spawn_col, spawn_row)` at the world center

**Why sexual crossover?** Crossing over good solutions from different parents can combine their strengths. This recombination is more powerful than random mutation alone.

---

## 9. Key Constants Summary

| Constant | Value | Meaning |
|----------|-------|---------|
| `MAX_LIFETIME` | 5000 | Max ticks per organism |
| `ENERGY_CAPACITY` | 200 | Max energy |
| `START_ENERGY` | 150 | Initial energy |
| `ENERGY_DECAY_RATE` | 1 | Base loss per decay event |
| `ENERGY_DECAY_INTERVAL` | 10 | Ticks between decay |
| `DECAY_PENALTY` | 0.1 | Negative reward per decay |
| `EXPLORE_BONUS` | 0.02 | Positive reward per new cell |
| `RL_LR` | 0.02 | REINFORCE learning rate |
| `TRACE_DECAY` | 0.90 | Eligibility trace decay |
| `EPSILON_START` | 0.2 | Exploration rate at birth |
| `EPSILON_END` | 0.05 | Exploration rate at death |
| `POPULATION_SIZE` | 1 | Founders per generation |
| `N_PARENTS` | 5 | Parents for crossover |
| `MUT_PROB` | 0.0 | Between-gen mutation (disabled) |
| `ASEXUAL_MUT_PROB` | 0.0 | Within-gen mutation (disabled) |
| `GENOME_SIZE` | 308 | Total NN weights |
| `STATE_SIZE` | 33 | NN input dimension |
| `HIDDEN_SIZE` | 8 | NN hidden layer size |
| `OUTPUT_SIZE` | 4 | NN output dimension |

---

## 10. Experiment Conditions

### Condition A: Learning-Enabled
- `rl_enabled = true`
- REINFORCE updates `active_weights` every tick
- Genome stays frozen, RL drift discarded at death
- Selection based on `genome_weights` only

### Condition B: Genetic Algorithm Only
- `rl_enabled = false`
- `active_weights === genome_weights` (same reference)
- No REINFORCE, no drift
- Selection based on `genome_weights` (no difference from active)

---

## References

- **REINFORCE**: Williams, R. J. (1992). "Simple statistical gradient-following algorithms for connectionist reinforcement learning."
- **Eligibility Traces**: Sutton & Barto (2018). "Reinforcement Learning: An Introduction."
- **Genetic Algorithms**: Mitchell, M. (1998). "An Introduction to Genetic Algorithms."
