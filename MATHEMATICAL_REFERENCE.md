# LifeEngine: Mathematical Reference

Mathematical definitions of the reward, fitness, learning, and energy functions used in the learning‑vs‑evolution‑vs‑pure‑RL experiments.

> **Status.** Updated to the **current** implementation (tournament GA, importance‑weighted REINFORCE with a baseline, 54→64→6 network, enabled mutation, energy/day‑night as coded). For experiment/run structure and the SLURM jobs, see `EXPERIMENTS.md`. Values here are the in‑code defaults in `ExperimentParams.js`, `NNBrain.js`, `GAManager.js`, `PureRLManager.js`, `AdvancedOrganism.js`; per‑run overrides are recorded in each run's `params.json`.

---

## 1. Reward Function (Reinforcement Learning)

Used in the **learning** and **pure‑RL** conditions to update network weights via REINFORCE. The reward is **not** the selection signal (fitness is — see §3); it is the within‑lifetime learning signal.

**Per‑tick reward** $r_t$:

$$r_t = r_{\text{food}} + r_{\text{explore}} + r_{\text{decay}} + r_{\text{predator}}$$

- **Food reward** — when food is eaten: $r_{\text{food}} = v_{\text{food}} \in \{0.5, 1.0, 2.0\}$ (low / medium / prestige; base food = 0.01). The **full** food value is booked even when the energy gain is clipped at `max_energy`, so reward and energy diverge for a full organism.
- **Exploration bonus** — first visit to a grid cell: $r_{\text{explore}} = +\text{EXPLORE\_BONUS}$, once per unique cell. **Default 0 — the bonus is OFF in every production run** (the explore‑bonus sweep found no fitness or cave‑usage effect); only the sweep scripts pass a non‑zero `--explore-bonus`. Credited to the *next* tick's reward, not the current one (`AdvancedOrganism._nnMove()`).
- **Decay penalty** — per unit of energy lost to decay this tick: $r_{\text{decay}} = -\text{DECAY\_PENALTY}\cdot(\text{energy lost}) = -0.05 \cdot \Delta e_{\text{decay}}$.
- **Predator penalty** — per drain contact: $r_{\text{predator}} = -\text{PREDATOR\_DRAIN\_PENALTY}\cdot d = -0.5\,d$, where $d = \text{PredatorHyperparameters.drainAmount}$. So the penalty **scales with the environment**: $-0.5$ at the default $d=1$, but $-2.5$ in the hard/roaming condition ($d=5$). `PredatorDrainCell` subtracts the full $d$ from `energy` (which may go negative before `die()` fires), so the penalty is exactly $0.5\times$ the energy decrement — on a fatal contact part of that decrement is notional.

### 1.1 Energy loss is two terms, not one

`this.energy` moves on exactly three code paths — decay (`AdvancedOrganism.js:299`), food (`:362`), and predator drain (`PredatorDrainCell.js:37`). Movement, rotation and reproduction are all **energy‑free** (a child is constructed with a fresh `START_ENERGY = 300` at no cost to the parent). So total energy lost is exhaustively $\Delta e^- = \Delta e_{\text{decay}} + \Delta e_{\text{drain}}$, and the reward can be written as one weighted energy‑loss term:

$$r_t = v_{\text{food}} - \sum_{s\in\{\text{decay},\,\text{drain}\}} \kappa_s\,\Delta e_s, \qquad \kappa_{\text{decay}} = 0.05,\ \ \kappa_{\text{drain}} = 0.5$$

**The two coefficients differ by 10×, so they must not be collapsed into a single $-\kappa\,\Delta e^-$.** One unit of energy lost to a predator is penalised ten times as hard as the same unit lost to decay: $\kappa_{\text{drain}}$ was calibrated against *food* (one drain unit $= -0.5 =$ one low‑food tile), $\kappa_{\text{decay}}$ was not.

The separation is also load‑bearing for learning, not just for bookkeeping. Decay is **dense, periodic and near‑constant** (one event every $\text{ENERGY\_DECAY\_INTERVAL}=10$ ticks), so the EMA baseline (§2.1) absorbs most of it — what survives $r_t - \bar b_{t-1}$ is only the *differential* between the $\times 1$ / $\times 2$ / $\times 0$ branches, which is precisely the day‑night/cave signal. Predator drain is **sparse and spiky** ($-2.5$ per contact at $d=5$, against a baseline of order $10^{-2}$), so it survives baseline subtraction almost intact and dominates the gradient on the ticks it fires. Same physical quantity, opposite statistical roles in the update.

---

## 2. REINFORCE with Baseline, Eligibility Traces, and Importance Weighting

Policy‑gradient update applied to `active_weights` only (`genome_weights` is never touched by RL). Implemented in `NNBrain.reinforce()`.

### 2.1 Running‑mean baseline (variance reduction)

$$\bar b_t = \beta\,\bar b_{t-1} + (1-\beta)\,r_t,\qquad \beta = \text{BASELINE\_DECAY} = 0.9$$

$$\tilde r_t = r_t - \bar b_t \quad\text{(baseline‑adjusted reward)}$$

**The baseline is updated *before* it is subtracted** (`NNBrain.reinforce()` advances `running_baseline` with $r_t$ on the same call that uses it), so substituting the first line into the second gives, exactly,

$$\tilde r_t = \beta\,\big(r_t - \bar b_{t-1}\big)$$

i.e. the textbook "subtract the *previous* baseline" form, scaled by $\beta$. Because that is an exact scalar multiple, it introduces **no directional bias** — it rescales the step. The practical consequence is that the **effective learning rate is $\alpha_{\text{eff}} = \beta\alpha = 0.9\,\alpha$**: 0.009 in the predator‑free baseline environment and 0.018 in the roaming‑predator environment, not the nominal 0.01 / 0.02 (§2.4). The action‑independence a REINFORCE baseline needs is not compromised by the collapse itself: $\bar b_{t-1}$ is a function of rewards up to $t-1$ only, so it is independent of $a_t$ and $\mathbb{E}[\bar b_{t-1}\nabla\log\pi(a_t)] = 0$. The residual caveat is the one every EMA baseline carries in an online trace method — $\bar b_{t-1}$ is correlated with the *older* actions still held in $e_t$ — which is bounded, standard, and identical across all conditions being compared.

$\bar b_0 = 0$ and the traces are zeroed at birth (`resetTraces()`), so each organism's baseline warms up over roughly the first $1/(1-\beta) = 10$ ticks of its life; nothing about the baseline or the traces is inherited.

---

### 2.2 Eligibility traces (forward‑time credit assignment, no backprop)

Traces accumulate the (importance‑weighted) score function $\rho\,\nabla\log\pi(a)$ and decay each tick with $\gamma = \text{TRACE\_DECAY} = 0.90$.

$\gamma$ is **not** a return discount — no return $G_t$ is ever formed. It is the credit‑assignment horizon: the trace is a geometrically weighted memory of the last $\approx 1/(1-\gamma) = 10$ ticks of score functions, which is the mechanism that pays a delayed food reward back to the movements that reached the food. This is the eligibility of Baxter & Bartlett's OLPOMDP — their trace parameter, which that paper also happens to write $\beta$ (no relation to `BASELINE_DECAY` above) — not the $\gamma$ of a discounted return.

**Output layer** ($k = 0,\dots,5$, i.e. $\text{OUTPUT\_SIZE}=6$):

$$\phi_k = \rho\big(\mathbb{1}[k=a]-\pi_k\big)$$
$$e^{(t)}_{W_2,k,j} = \gamma\,e^{(t-1)}_{W_2,k,j} + \phi_k\,h_j,\qquad e^{(t)}_{b_2,k} = \gamma\,e^{(t-1)}_{b_2,k} + \phi_k$$

**Hidden/input layer** (only for active ReLU units, $h_j>0$):

$$\Delta_j = \rho\sum_{k=0}^{5} w_{W_2,k,j}\big(\mathbb{1}[k=a]-\pi_k\big)$$
$$e^{(t)}_{W_1,j,i} = \gamma\,e^{(t-1)}_{W_1,j,i} + \Delta_j\,x_i,\qquad e^{(t)}_{b_1,j} = \gamma\,e^{(t-1)}_{b_1,j} + \Delta_j$$

Dead units ($h_j\le 0$) get zero trace (ReLU gating).

### 2.3 Importance ratio (off‑policy correction for ε‑exploration)

The action is drawn from the **behaviour** policy (§4): with prob. $\epsilon$ a uniform random action, else a sample from $\pi$. So

$$b(a) = (1-\epsilon)\,\pi(a) + \frac{\epsilon}{N},\qquad N = \text{OUTPUT\_SIZE}=6$$
$$\rho = \min\!\left(\frac{\pi(a)}{b(a)},\ 10\right)$$

With `epsilon_enabled = false` (the `--no-epsilon` runs), $\epsilon=0 \Rightarrow b=\pi \Rightarrow \rho = 1$ (pure on‑policy REINFORCE). $\rho$ is capped at 10 for stability.

### 2.4 Weight update

$$w_{idx} \leftarrow \operatorname{clip}\!\Big(w_{idx} + \alpha\,\tilde r_t\, e^{(t)}_{idx},\ [-1,1]\Big)$$

- $\alpha = \text{RL\_LR}$ = **0.01** (predator‑free baseline env) / **0.02** (roaming‑predator env). Tunable via `--learning-rate`. Because $\tilde r_t = \beta(r_t-\bar b_{t-1})$ (§2.1), the **realised** step size is $\alpha_{\text{eff}} = 0.9\alpha$ = 0.009 / 0.018.
- Weights are clipped to $[-1,1]$ after every update; the same clip bounds crossover and mutation (§6), so the whole search space is the cube $[-1,1]^{|\theta|}$.

**Update ordering (an intentional off‑by‑one).** `_nnMove()` reads `pending_reward` — the consequences of $a_{t-1}$ — then calls `decide()`, which runs the forward pass (sampling $a_t$) *before* `reinforce(r_t)`. The trace therefore already contains $\nabla\log\pi(a_t)$ when it is multiplied by a reward $a_t$ cannot have caused. Since $\mathbb{E}_{a\sim\pi}[\nabla\log\pi(a)] = 0$ and $a_t$ is conditionally independent of $r_t$ given the state, that term contributes **variance but not bias**; the credit for $r_t$ still lands on $a_{t-1}$ and earlier, discounted by $\gamma$.

**REINFORCE, not backprop:** weights move directly from the reward‑weighted eligibility traces; there is no error backpropagation. Traces are the forward‑time memory of which weights were responsible for recent actions.

**Verified against a finite difference.** Because the score function $\nabla\log\pi(a)$ is written into the trace by hand rather than obtained from an autodiff library, the implementation is checked numerically rather than trusted. With the trace zeroed beforehand, one call to `reinforce()` leaves exactly $\nabla\log\pi(a)$ in it ($e = \gamma\cdot 0 + \nabla\log\pi(a)$), and weight updates are frozen so the comparison is taken at the same point in weight space. That trace is compared against a **central finite difference** of $\log\pi(a)$ evaluated through an independently written forward pass, so a bug shared between the forward pass and the gradient cannot hide. Agreement is to **max relative error $\approx 2\times10^{-7}$** with **cosine similarity $1.000000000$** over a stride sweep of the full genome — i.e. limited by float32 rounding, not by the implementation. Coordinates whose ReLU pre‑activation falls within the finite‑difference step of the kink are a genuine non‑differentiability; they are skipped and counted rather than silently included. Run it with `node src/eval/validate.js --only grad`; the numbers land in `output/validation/verification_report.json`.

### 2.5 The `--no-epsilon` case (all production runs)

With `epsilon_enabled = false`, $\epsilon\equiv 0$: actions are sampled straight from $\pi$, $b=\pi$, and the importance ratio is identically $\rho = 1$. The importance‑sampling term is therefore a **no‑op** — it only ever mattered for the earlier epsilon LR‑sweep runs. The update stays fully **on‑policy** but **retains** the two standard on‑policy components — the running‑mean baseline and the eligibility traces — so it is REINFORCE‑*with‑baseline‑and‑traces*, not baseline‑free vanilla REINFORCE. The trace factor collapses from $\phi_k = \rho(\mathbb 1[k=a]-\pi_k)$ to $(\mathbb 1[k=a]-\pi_k)$, giving:

$$e^{(t)}_{idx} = \gamma\,e^{(t-1)}_{idx} + \big[\nabla\log\pi(a)\big]_{idx},\qquad w_{idx}\leftarrow \operatorname{clip}\!\big(w_{idx}+\alpha\,(r_t-\bar b_t)\,e^{(t)}_{idx},\ [-1,1]\big)$$

### 2.6 What to call this algorithm

The update above is **online, per‑tick and non‑episodic**: there is no episode boundary short of death, no return $G_t$, and no learned value function. That is precisely the **OLPOMDP** estimator of Baxter & Bartlett (2001) — $z_t = \beta z_{t-1} + \nabla\log\mu(a_t)$, $\theta \leftarrow \theta + \alpha r_t z_t$ — built on the REINFORCE score‑function estimator of Williams (1992), with Williams' *reinforcement baseline* supplied here as an EMA of past reward rather than a critic.

Describe it as: **an online, non‑episodic REINFORCE (Williams, 1992) in the OLPOMDP form (Baxter & Bartlett, 2001), with an eligibility trace over the score function and a running‑mean reward baseline.** Calling it "episodic REINFORCE" would be wrong — it invites the question of where $G_t$ is — and calling it actor–critic would also be wrong, since the baseline is a scalar reward average, not $\hat v(s)$.

---

## 3. Fitness Function (Genetic Algorithm)

Used by `GAManager` to rank organisms for selection — the **primary selection signal** in the GA conditions. Independent of the RL reward, so learning and evolution are compared on the same yardstick.

$$\text{Fitness}_i = \text{cumulative\_food\_score}_i = \sum_{t=1}^{T_i} v_{\text{food}}(t)$$

where $T_i$ is organism $i$'s lifetime (ticks) and $v_{\text{food}}(t)$ is the energy value of any food eaten at tick $t$.

### GA Selection and Reproduction (`GAManager.evolve()`)

Runs at the end of every generation (a fixed 10 000‑tick / 5‑map window — see §8):

1. **Eligible set = ALL agents** of the generation (the 100 founders + every descendant that lived). There is **no truncation / elitism** — every agent has a non‑zero selection probability.
2. *(optional)* natural‑disaster cull of the eligible set before selection (fitness‑blind; off by default).
3. **Tournament selection**, $K = \text{TOURNAMENT\_K} = 2$: for each offspring, draw two parents, each by sampling $K$ agents **uniformly from the whole eligible set** and keeping the fitter:
   $$p = \arg\max_{a \in \{a_1,\dots,a_K\}} \text{Fitness}(a),\quad a_i \sim \text{Uniform(all agents)}$$
   Selection pressure is controlled by $K$ alone, not by a fitness cutoff. With $K=2$, an agent of rank $i$ (1 = fittest) among $n$ is selected with probability $\approx 2(n-i)/(n(n-1))$.
4. **Uniform crossover** (§6) of the two parents' genomes.
5. **Gaussian mutation** (§6), $\text{MUT\_PROB}=0.03$, $\sigma=0.1$.
6. **No elitism** — all $\text{POPULATION\_SIZE}=100$ founders are fresh crossover children, spawned at the world centre.

> $\text{SELECTION\_PERCENT}=0.2$ is a **reporting** parameter only (the top‑20% cohort for `top20percent_fitness` and top‑cohort drift/variance); it plays **no role** in parent selection.

---

## 4. Neural Network Policy

### Architecture

$$\text{Input}(54) \to \text{ReLU}(W_1,b_1)\,(64) \to \text{softmax}(W_2,b_2)\,(6) \to \text{Action}$$

- Input: $n_i = \text{STATE\_SIZE} = 54$ (4 eye directions × 13 percept classes + 2 scalars)
- Hidden: $n_h = \text{HIDDEN\_SIZE} = 64$ (ReLU; tunable, resizes the genome)
- Output: $n_o = \text{OUTPUT\_SIZE} = 6$ (up, right, down, left, rotate‑left, rotate‑right)

**Total weights** $\text{GENOME\_SIZE} = 61\,n_h + 6 = 3910$ at $n_h=64$:
$W_1: 54\times 64 = 3456$, $b_1: 64$, $W_2: 64\times 6 = 384$, $b_2: 6$.

Each organism holds `genome_weights` (heritable) and `active_weights` (acted on; RL updates these). In GA‑only mode they are the **same reference** (no drift).

### Weight initialisation (Glorot/Xavier **uniform**)

Founder genomes with no parent are drawn by `NNBrain._initGenome()`:

$$w \sim \mathcal U(-L,\,+L),\qquad L = \sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}},\qquad b_1 = b_2 = 0$$

so $\operatorname{Var}(w) = \dfrac{2}{n_{\text{in}}+n_{\text{out}}}$ — the same second moment as the Gaussian Xavier variant, with bounded support and no tails. The limit is **per layer**:

| Block | fan‑in → fan‑out | $L$ at $n_h=64$ | $L$ at $n_h=128$ |
|-------|------------------|-----------------|------------------|
| $W_1$ | $54 \to n_h$ | 0.2255 | 0.1816 |
| $W_2$ | $n_h \to 6$ | 0.2928 | 0.2116 |

Both bias blocks start at exactly zero. Every limit is well inside the $[-1,1]$ clip, so a fresh brain is never clipped at birth. The draw uses JavaScript's `Math.random()` (V8 xorshift128+), which is **not** seeded by the run seed — founder genomes are not bit‑reproducible from `params.json`, only distributionally reproducible. The landscape probes' fresh‑brain anchor (`ll_common.xavier_genome()`) reproduces this exactly, including the zero biases and the per‑layer limits.

### Activations

$$h_j = \max\!\Big(0,\ \sum_{i=0}^{53} w^{(1)}_{i,j} x_i + b^{(1)}_j\Big)\qquad(\text{ReLU})$$
$$\pi_k = \frac{e^{z^{(2)}_k}}{\sum_{k'=0}^{5} e^{z^{(2)}_{k'}}},\qquad z^{(2)}_k = \sum_{j=0}^{63} w^{(2)}_{j,k} h_j + b^{(2)}_k\qquad(\text{softmax})$$

### Action selection (ε‑greedy, importance‑corrected)

$$a = \begin{cases} \text{uniform}\{0,\dots,5\} & \text{with prob. } \epsilon \\ a \sim \text{Categorical}(\pi) & \text{with prob. } 1-\epsilon \end{cases}$$

(The exploit branch **samples** from $\pi$; it is not argmax.) The behaviour probability $b(a)=(1-\epsilon)\pi(a)+\epsilon/N$ feeds the importance ratio $\rho$ (§2.3).

**Epsilon schedule** — **per organism**. Each organism's ε decays over its *own* age fraction $f\in[0,1]$,

$$f = \min\!\Big(1,\ \frac{t_{\text{age}}}{\text{TICKS\_PER\_GEN} - \text{birth\_tick}}\Big)$$

where $t_{\text{age}}$ is ticks since **that** organism was born and the denominator is its **maximum achievable lifespan** ($\text{\_max\_possible\_lifetime}$ = generation ticks remaining at birth), so $f\to 1$ exactly as the organism reaches the end of the generation — not a flat $\text{TICKS\_PER\_GEN}$ for everyone (which would leave late‑born organisms still exploratory at death). $f$ is then shaped by an exponent:

$$\epsilon(f) = \epsilon_{\text{start}} + (\epsilon_{\text{end}}-\epsilon_{\text{start}})\cdot f^{\,c},\qquad c \in \{0.5,\ 1,\ 2\}$$

- $c=0.5$ `sublinear` (√): drop fast early, flatten late.
- $c=1$ `linear`: constant rate.
- $c=2$ `quadratic` (default): hold high, drop late.
- $\epsilon_{\text{start}} = 0.3$ (default; swept over $\{0.2,0.3,0.5,0.7\}$), $\epsilon_{\text{end}}=0.05$.
- `epsilon_enabled = false` pins $\epsilon\equiv 0$ (no exploration) — the `--no-epsilon` runs.

ε‑greedy is applied **only in RL conditions**; the evolution condition always acts on‑policy.

---

## 5. Energy System

### Decay

Every $\Delta t = \text{ENERGY\_DECAY\_INTERVAL} = 10$ ticks:

$$e_t = e_{t-\Delta t} - d\cdot m,\qquad d = \text{ENERGY\_DECAY\_RATE} = 1$$

$m$ is a **location × time‑of‑day** multiplier. Day/night are equal‑length (`day_length = night_length = 300`, a **600‑tick** cycle; night when $\text{total\_ticks}\bmod 600 \ge 300$):

| | Outside cave | Inside cave |
|---|---|---|
| **Day** | $m=1$ | $m=2$ (costly by day) |
| **Night** | $m=2$ (exposed) | $m=0$ (safe sleep, energy frozen) |

### Food energy

$$e_t \leftarrow \min(e_t + v_{\text{food}},\ e_{\max}),\qquad \text{cumulative\_food\_score}\mathrel{+}= v_{\text{food}}$$

$e_{\max} = \text{ENERGY\_CAPACITY} = 500$; $v_{\text{food}} \in \{0.5,1.0,2.0\}$ (low/med/prestige), base food $0.01$.

### Death

$$e_t \le 0 \quad\text{OR}\quad t \ge T_{\max},\qquad T_{\max}=\text{MAX\_LIFETIME}=1{,}000{,}000$$

In practice lifespan is bounded by energy and by the 10 000‑tick generation window, not the (effectively unreachable) $T_{\max}$ cap.

---

## 6. Genetic Operators

### Uniform crossover

Each weight independently from parent A or B (50/50), clipped to $[-1,1]$; applied to the whole genome:

$$c_i = \begin{cases} a_i & u_i < 0.5\\ b_i & \text{else}\end{cases},\quad u_i\sim\text{Uniform}(0,1)$$

### Gaussian mutation (enabled)

$$w_i' = \operatorname{clip}\Big(w_i + \mathbb{1}[u<p]\,\mathcal N(0,\sigma),\ [-1,1]\Big)$$

- **Inter‑generation** (GA between‑gen; pure‑RL between‑episode & top‑up): $p=\text{MUT\_PROB}=0.03$, $\sigma=\text{MUT\_SIGMA}=0.1$ (from `ExperimentParams`, honours `--mut-prob/--mut-sigma`).
- **Intra‑generation** (asexual reproduction, all conditions): $p=\text{ASEXUAL\_MUT\_PROB}=0.05$, $\sigma=0.1$.

Both draws are **per weight and independent** — a Bernoulli trial for every one of the $61 n_h + 6$ entries, not one draw per genome — and the $[-1,1]$ clip is applied only to the weights that actually mutated.

### Expected number of mutated weights

The rate alone understates the within‑generation channel, because the 5% fires on **every birth** while the 3% fires once per genome per generation boundary:

| Event | Rate | h64 (3910 weights) | h128 (7814 weights) |
|---|---|---|---|
| Asexual birth (intra‑generation) | 0.05 | ~196 | **~391** |
| Generation boundary (inter‑generation) | 0.03 | ~117 | **~234** |

An organism reproduces every time it banks 3.5 energy (§9), so one lineage can pass through many mutation rounds inside a single 10 000‑tick generation, each stacking ~391 perturbations at $n_h=128$. Intra‑generational asexual mutation is therefore the **larger source of genetic variation**, not the smaller one. Reproduction also only succeeds with probability $\text{REPRODUCTION\_SUCCESS\_RATE}=0.8$, and the energy counter resets either way, so ~20% of reproduction events discard an already‑mutated genome.

Note the two channels differ in kind as well as rate: within a generation reproduction is **asexual** — a clone of the parent's `genome_weights` (never `active_weights`, so RL drift is not passed on) plus noise, with no crossover. Crossover happens only at the generation boundary.

### Tunability asymmetry (matters when reading a mutation sweep)

`MUT_PROB` / `MUT_SIGMA` are read from `ExperimentParams` and honour `--mut-prob` / `--mut-sigma`; **`ASEXUAL_MUT_PROB` / `ASEXUAL_MUT_SIGMA` are module constants** in `AdvancedOrganism.js:117-118` with no CLI override. A sweep over `--mut-prob` therefore moves only the inter‑generation 3% term while the intra‑generation 5% term — the larger contributor above — stays pinned. Such a sweep measures the sensitivity of *between‑generation disruption*, not of total genetic variation, and should be reported as such. (`PureRLManager`'s between‑episode mutation reads the same `mut_prob`/`mut_sigma`, so it moves with the GA term.)

(Both were disabled in the earlier design; they are now enabled and consistent across conditions.)

---

## 7. Perception (Input State Vector)

54‑element vector $x\in\mathbb R^{54}$: **4 eye directions × 13 one‑hot percept classes** (positions 0–51) + **2 scalars** (positions 52–53, incl. normalised energy $x = \operatorname{clip}(e/e_{\max},[0,1])$).

Percept classes (`PERCEPT_INDEX` in `NNBrain.js`):

| idx | percept | idx | percept |
|---|---|---|---|
| 0 | empty / out‑of‑bounds | 7 | prestige‑food landmark |
| 1 | low food (0.5) | 8 | cave |
| 2 | medium food (1.0) | 9 | base food (fallback) |
| 3 | prestige food (2.0) | 10 | prey organism body |
| 4 | wall / obstacle | 11 | roaming predator body |
| 5 | low‑food landmark | 12 | patrol predator body |
| 6 | medium‑food landmark | | |

### Perception range

Each of the 4 eye cells casts a ray and returns the **first non‑empty cell** within $\text{lookRange}$ (`Hyperparameters.js:27`):

$$\text{lookRange} = \mathbf{200}\ \text{cells}$$

This is a large sensory horizon relative to the world — **40% of a 500‑wide grid**, and it exceeds every predator radius (detection 31, give‑up 44 after the ×1.25 rescale at width 500). Prey therefore see a roaming predator long before that predator can detect *them*, which is what makes avoidance learnable rather than purely reactive. It also means the one‑hot percept saturates easily: on a populated map the nearest non‑empty cell along a ray is usually close, so the effective range only matters in sparse regions.

Distinguishing prey (10) from roaming (11) and patrol (12) predators lets avoidance be learned/evolved from **direct perception**, not just from the indirect energy‑loss of contact.

---

## 8. Generation / Episode Structure

A generation is a fixed **$\text{TICKS\_PER\_GEN}=\text{TICKS\_PER\_MAP}\times\text{MAPS\_PER\_GEN}=2000\times5=10{,}000$‑tick** window, shown as 5 different maps in sequence. The population **persists across maps within a generation** (same energy/position; only terrain changes and RL traces reset). At the window end the manager runs GA `evolve()` (or the pure‑RL window close). This replaces the old "generation ends when the lineage dies" rule.

---

## 9. Reproduction Thresholds

### Within‑lifetime (asexual)

Energy‑triggered: when energy gained since the last reproduction reaches a threshold, spawn one child (at the central spawn, with success prob. $\text{REPRODUCTION\_SUCCESS\_RATE}=0.8$):

$$\text{energyGainedSinceReproduction} \ge 3.5$$

**Inheritance:**
- **GA conditions (learning, evolution):** child inherits parent's `genome_weights` only (**non‑Lamarckian**) + asexual mutation. RL drift is discarded.
- **Pure RL:** parent's learned `active_weights` are synced into the genome first, so the child inherits the **learned** state (**Lamarckian**).

### Between‑generation

GA: tournament selection → uniform crossover → mutation → 100 founders (§3). Pure RL: no GA — the window close caps/tops‑up the population to 100 from a buffer of recently‑dead learned weights (see `EXPERIMENTS.md §3.3`).

---

## 10. Predators

Fixed environmental hazard (`PredatorHyperparameters.js`), not part of the evolving population.

- **Roaming:** fixed count (`--roaming-predators`, 60 in the experiments); detect prey within a radius, chase, and on contact drain `--predator-drain` energy/tick (5 in experiments) + apply the $-0.5d = -2.5$ RL penalty (§1). Never die/reproduce, and the count is held constant by respawn.

  Spatial radii are calibrated for $\text{referenceCols}=400$ and rescaled by $s = \text{cols}/400$ in `resolveForGrid()` — $s = 1.25$ at the production width of 500:

  | Parameter | Reference (400 cols) | **At width 500** | Scaled? |
  |---|---|---|---|
  | `detectionRadius` | 25 | **31** | yes |
  | `giveUpRadius` | 35 | **44** | yes |
  | `wanderRange` | 6 | 6 | no — a duration |
  | `moveInterval` | 1 | 1 | no |
  | `respawnDelay` | 50 ticks | 50 | no |

  Note the asymmetry with perception: prey see 200 cells (§7), a predator detects at 31.
- **Patrol:** guard prestige patches, leashed; disabled (`--predators-per-patch 0`) in the core experiments.
- **Caves are predator‑safe.**

---

## 11. Metrics and Analysis Calculations

### Mean absolute weight difference (`avg_learned_weight_diff`)

The **mean absolute difference between an organism's active and genome weight vectors** — equivalently, the $L_1$ distance between them normalised by the number of weights:

$$\overline{|\Delta w|}_i = \frac{1}{M}\sum_{j=0}^{M-1}\big|w^{\text{active}}_{i,j}-w^{\text{genome}}_{i,j}\big| = \frac{1}{M}\big\|\mathbf w^{\text{active}}_i - \mathbf w^{\text{genome}}_i\big\|_1,\qquad M=\text{GENOME\_SIZE}$$

The population figure `avg_learned_weight_diff` is the unweighted mean of $\overline{|\Delta w|}_i$ over every agent in the generation (`Logger.js:173-176`) — a mean of means, so a newborn that has barely drifted counts as much as one that lived the full window.

> **Naming.** This metric was previously abbreviated **"MAD"** throughout the code and figures. That abbreviation is abandoned: in statistics MAD denotes the **median absolute deviation** $\operatorname{med}(|x_i - \operatorname{med}(x)|)$, and sometimes the mean absolute deviation *from a central value*. This quantity is neither — there is no median, and it is not a deviation from a centre but a distance between two specific paired vectors. Call it the **mean absolute weight difference**. The CSV columns (`avg_learned_weight_diff`, `learned_weight_diff`) and the `mad` verification-check id keep their names, since those are data/interface identifiers.

Normalising per weight makes the value comparable across hidden sizes: a raw $L_1$ norm would grow with $M$, so h64 and h128 runs could not be compared. Typical values span $10^{-5}$ to $10^{-3}$.

- Learning condition: $>0$ (RL adapts active weights; not inherited — genome frozen).
- Evolution: $=0$ (`active === genome`).
- Pure RL: $>0$, and the drift **is** inherited (Lamarckian).

### RMS weight magnitude (`avg_network_weight_mag`)

$$\text{RMS}_i = \sqrt{\tfrac{1}{M}\textstyle\sum_j (w_{i,j})^2}$$

### Population genome variance (diversity)

$$\sigma^2_{\text{genome}} = \frac1M\sum_{j=0}^{M-1}\operatorname{Var}\big(\{w^{\text{genome}}_{i,j}\}_i\big)$$

(sampled over up to 50 agents for cost). High = diverse; low = converged.

### Other logged aggregates

$\bar f_{\text{top20\%}}$ (top‑20% mean fitness), $\bar f$ (population mean fitness), $\bar T$ (mean lifetime), early/end energy means, peak population, total agents. Per‑organism behaviours (`organisms.csv`): `cells_visited`, `predator_touches`, `drained_ticks`, `cave_entries{,_day,_night}`, `food_{low,medium,prestige,default}`, `death_cause`.

---

## 12. Key Constants Summary

| Constant | Value | Meaning |
|----------|-------|---------|
| `TICKS_PER_MAP`/`MAPS_PER_GEN`/`TICKS_PER_GEN` | 2000 / 5 / 10 000 | timing |
| `MAX_LIFETIME` | 1 000 000 | ceiling (energy/window bind first) |
| `ENERGY_CAPACITY` / `START_ENERGY` | 500 / 300 | energy cap / start |
| `ENERGY_DECAY_RATE` / `_INTERVAL` | 1 / 10 | base decay per 10 ticks |
| day / night length | 300 / 300 | 600‑tick cycle |
| decay multipliers | ×2 day‑cave, ×2 night‑outside, ×0 night‑cave | applied to `ENERGY_DECAY_RATE` |
| `lookRange` | **200** cells | eye raycast horizon (§7) |
| food values | 0.5 / 1.0 / 2.0 | low / medium / prestige |
| `EXPLORE_BONUS` | **0** (off in all production runs) | reward per new cell; swept only |
| `DECAY_PENALTY` | 0.05 | reward per unit energy decayed |
| `PREDATOR_DRAIN_PENALTY` | 0.5 | multiplier on drain: reward $=-0.5d$ per contact ($d=1$ default, $d=5$ hard) |
| `RL_LR` (α) | 0.01 baseline / 0.02 roaming | nominal REINFORCE step ($\alpha_{\text{eff}} = 0.9\alpha$, §2.1) |
| `TRACE_DECAY` (γ) | 0.90 | eligibility‑trace decay |
| `BASELINE_DECAY` (β) | 0.9 | reward‑baseline EMA; also scales the step (§2.1) |
| `EPSILON_START`/`END` | 0.3 / 0.05 | exploration (0 with `--no-epsilon`) |
| epsilon decay exponent | 0.5 / 1 / 2 | sublinear / linear / quadratic |
| importance ratio cap | 10 | $\rho=\min(\pi/b,10)$ |
| Xavier limit $L$ | $\sqrt{6/(n_{in}+n_{out})}$ | founder weight init, uniform; biases 0 |
| `POPULATION_SIZE` | 100 | founders per generation |
| `SELECTION_PERCENT` | 0.20 | top‑20% **reporting** cohort only — not a selection cutoff |
| `TOURNAMENT_K` | 2 | GA tournament size |
| `MUT_PROB` / `MUT_SIGMA` | 0.03 / 0.1 | inter‑generation mutation |
| `ASEXUAL_MUT_PROB` / `_SIGMA` | 0.05 / 0.1 | intra‑generation mutation |
| reproduction trigger | 3.5 energy | asexual reproduction |
| `REPRODUCTION_SUCCESS_RATE` | 0.8 | child spawn success |
| `GENOME_SIZE` | 3910 at H=64, **7814 at H=128** (=61·H+6) | NN weights |
| `STATE_SIZE` / `HIDDEN_SIZE` / `OUTPUT_SIZE` | 54 / 64 default (**128 in production**) / 6 | NN dims |
| roaming predators / drain | 60 / 5 | experiments (patrol disabled: `--predators-per-patch 0`) |
| predator `detectionRadius` / `giveUpRadius` | 25 / 35 ref → **31 / 44** at width 500 | rescaled by cols/400 (§10) |
| predator `referenceCols` | 400 | radius calibration width |
| `wanderRange` / `moveInterval` / `respawnDelay` | 6 / 1 / 50 | not spatial — unscaled |
| `COLLAPSE_BUFFER_SIZE` | 25 | pure‑RL weight buffer |

---

## 13. Experiment Conditions

| Condition | Flags | RL | GA | Heritable unit | Lamarckian |
|---|---|---|---|---|---|
| **Evolution** | `--condition evolution` | — | ✓ | genome | n/a |
| **Learning** | `--condition learning` | ✓ | ✓ | genome (RL drift discarded) | No |
| **Pure RL** | `--mode pure_rl` | ✓ | — | active (synced→genome) | Yes |

**Environments & operating point (going forward):** the learning/pure‑RL conditions run **no‑epsilon**, with $\alpha=0.01$ in the **predator‑free baseline** environment and $\alpha=0.02$ in the **roaming‑predator** environment (60 predators, drain 5). Each α is the optimum from that environment's learning‑rate sweep. See `EXPERIMENTS.md` for the SLURM job arrays.

---

## References

- **REINFORCE / score function / reinforcement baseline**: Williams, R. J. (1992). "Simple statistical gradient‑following algorithms for connectionist reinforcement learning." *Machine Learning*, 8(3–4), 229–256.
- **The online, non‑episodic form actually implemented (OLPOMDP)**: Baxter, J. & Bartlett, P. L. (2001). "Infinite‑horizon policy‑gradient estimation." *JAIR*, 15, 319–350. — the per‑step update with a discounted eligibility of $\nabla\log\pi$; their $\beta$ is this code's `TRACE_DECAY` (§2.2, §2.6).
- **Eligibility traces / REINFORCE‑with‑baseline (textbook)**: Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.) — Ch. 12 (traces), §13.4 (REINFORCE with baseline).
- *(optional, same estimator arrived at independently)* Kimura, H. & Kobayashi, S. (1998). "An analysis of actor/critic algorithms using eligibility traces." *ICML*.
- **Weight initialisation**: Glorot, X. & Bengio, Y. (2010). "Understanding the difficulty of training deep feedforward neural networks." *AISTATS* — the uniform variant, §4.
- **Genetic Algorithms / tournament selection**: Mitchell, M. (1998). "An Introduction to Genetic Algorithms."
