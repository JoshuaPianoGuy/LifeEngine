/**
 * NNBrain.js
 *
 * Neural network policy for investigating whether within-lifetime reinforcement
 * learning influences the rate at which effective food-seeking behaviours become
 * encoded in heritable neural network weights across generations.
 *
 * Architecture:  54 -> 32 -> 6
 *   Input  (54): 4 eye directions x 13 one-hot percept types  +  1 energy scalar + 1 rotation scalar
 *   Hidden (32): ReLU
 *   Output  (6): softmax -> up / right / down / left / rotate-left / rotate-right
 *   Genome: 54x32 + 32 + 32x6 + 6 = 1728 + 32 + 192 + 6 = 1958 weights
 *
 * Perception:
 *   Uses the existing EyeCell.look() raycast (lookRange = 30 by default).
 *   Each of the 4 directional eye cells returns the first non-empty cell within
 *   range. NNBrain reads those observations directly rather than going through
 *   the Brain.observe() / Brain.decide() pipeline.
 *
 *   Percept types (one-hot, 13 classes):
 *     0 - nothing / empty / out-of-bounds
 *     1 - food_red       (low food, energy 0.5)
 *     2 - food_orange    (medium food, energy 1.0)
 *     3 - food_teal      (prestige food, energy 2.0)
 *     4 - wall / obstacle
 *     5 - landmark_red   (low food nearby)
 *     6 - landmark_orange (medium food nearby)
 *     7 - landmark_teal  (prestige food nearby)
 *     8 - cave
 *     9 - base food (fallback)
 *    10 - prey organism body (mouth/producer/mover/killer/armor/eye)
 *    11 - roaming predator body (free-roaming hazard, chases anywhere)
 *    12 - patrol predator body  (patch-leashed hazard, guards prestige food)
 *
 * GA / inheritance (non-Lamarckian):
 *   genome_weights -- starting weights; what the GA reads and crossovers.
 *                     Never modified during an agent's lifetime.
 *   active_weights -- copy of genome at birth; RL updates go here only.
 *                     Discarded at death. The GA never sees these.
 *
 * Condition A (rl_enabled = true):
 *   active_weights drifts via REINFORCE + eligibility traces each tick.
 *   genome_weights stays frozen. GA selects on genome_weights.
 *
 * Condition B (rl_enabled = false):
 *   active_weights === genome_weights (same reference). No RL, no drift.
 *
 * Exploration:
 *   Epsilon-greedy decay over the agent's available lifetime (see
 *   AdvancedOrganism._max_possible_lifetime). The decay shape is tunable via
 *   ExperimentParams.epsilon_decay_shape: 'sublinear' (√), 'linear', or
 *   'quadratic' (², the default). Exploratory actions are
 *   importance-weighted (ρ = π(a)/behaviour(a), behaviour = the ε-greedy
 *   mixture) so off-policy noise doesn't bias the REINFORCE update.
 *   Set ExperimentParams.epsilon_enabled = false to pin epsilon to 0 and run
 *   pure on-policy REINFORCE (no exploration at all).
 */

'use strict';

const Directions = require('../Directions');
// Tunable hyperparameters (RL rate, epsilon, hidden size). Read once here at
// module load; the headless runner overrides ExperimentParams before this
// module is required so per-job values bake in. Browser uses the defaults.
const ExperimentParams = require('../../ExperimentParams');

// Percept type map: CellState name -> one-hot index
// Add entries here when you add new CellStates.
// Anything not listed falls back to 0 (nothing).
// Keys are CellState.name values (the string passed to super() in each class).
// These must match exactly — note spaces in the new food/landmark names.
//
// Index 10 ("prey organism body") vs index 11/12 ("predator body") is the
// deliberate differentiation point: prey can tell "another prey organism is
// here" apart from "a predator is here" via eye raycasts, rather than both
// collapsing into one ambiguous "occupied" signal. This is what makes
// learned/evolved avoidance possible from direct perception, not just from
// the indirect energy-loss consequence of contact (PredatorDrainCell).
//
// Roaming and patrol predators are built from the SAME cell-state names, so
// the table below maps every predator cell to index 11 by default. They are
// behaviourally distinct hazards, though (roaming = chases anywhere, patrol =
// leashed to a prestige patch), so buildStateVector() promotes a patrol
// predator to its own percept (index 12) using the owning organism's
// is_patrol flag — letting prey perceive, and thus learn/evolve, type-specific
// responses. A richer hazard structure is also intended to roughen the fitness
// landscape rather than collapse to one generic "flee" response.
const PERCEPT_INDEX = {
    'empty':                    0,
    'wall':                     4,
    'low food':                 1,   // CellStates.lowFood.name
    'medium food':              2,   // CellStates.mediumFood.name
    'prestige food':            3,   // CellStates.prestigeFood.name
    'low food landmark':        5,   // CellStates.lowFoodLandmark.name
    'medium food landmark':     6,   // CellStates.mediumFoodLandmark.name
    'prestige food landmark':   7,   // CellStates.prestigeFoodLandmark.name
    'food':                     9,   // base LifeEngine food -> medium tier as fallback
    'cave':                     8,   // cshelter during night cycle; important navigation target
    'mouth':                    10,
    'producer':                 10,
    'mover':                    10,
    'killer':                   10,
    'armor':                    10,
    'eye':                      10,
    'predator body':            11,
    'predator mover':           11,
    'predator eye':             11,
    'predator drain':           11,
};
const N_PERCEPT_TYPES = 13;

// Base predator percept (any predator cell) and its patrol-specific promotion.
// perceptIndex() returns PERCEPT_PREDATOR for every predator cell-state name;
// buildStateVector() upgrades it to PERCEPT_PATROL_PREDATOR when the cell's
// owning organism is a patrol predator (is_patrol).
const PERCEPT_PREDATOR        = 11;
const PERCEPT_PATROL_PREDATOR = 12;

// Fixed iteration order for the 4 eye directions
const EYE_DIRECTIONS = [
    Directions.up,
    Directions.right,
    Directions.down,
    Directions.left,
];
const N_EYE_DIRECTIONS = 4;

// Network topology
const N_SCALARS   = 2;   // energy + rotation
const STATE_SIZE  = N_EYE_DIRECTIONS * N_PERCEPT_TYPES + N_SCALARS;  // 54
const HIDDEN_SIZE = ExperimentParams.hidden_size;  // tunable (default 64)
const OUTPUT_SIZE = 6;   // up, right, down, left, rotate-left, rotate-right

const DEBUG_STATE_VECTOR = false;  // Set to true to log the 54-element input vector

const W1_SIZE     = STATE_SIZE  * HIDDEN_SIZE;   // 1728
const B1_SIZE     = HIDDEN_SIZE;                 //   32
const W2_SIZE     = HIDDEN_SIZE * OUTPUT_SIZE;   //  192
const B2_SIZE     = OUTPUT_SIZE;                 //    6
const GENOME_SIZE = W1_SIZE + B1_SIZE + W2_SIZE + B2_SIZE;  // 1958

// RL hyper-parameters
const RL_LR            = ExperimentParams.learning_rate;  // tunable (default 0.02)
const TRACE_DECAY      = 0.90;
const BASELINE_DECAY   = 0.9;   // exponential moving average decay for running mean baseline (0.9 * mean + 0.1 * reward)

// Exploration (tunable; defaults 0.5 -> 0.05)
const EPSILON_START = ExperimentParams.epsilon_start;
const EPSILON_END   = ExperimentParams.epsilon_end;
// Master switch: when false, epsilon is pinned to 0 (pure on-policy REINFORCE,
// no random-action injection). See ExperimentParams.epsilon_enabled.
const EPSILON_ENABLED = ExperimentParams.epsilon_enabled !== false;

// Exponent applied to lifetime_frac (0->1) to shape the epsilon decay. Named
// shapes keep experiment conditions/logs readable; unknown -> quadratic (the
// original hard-coded behaviour). See ExperimentParams.epsilon_decay_shape.
const EPSILON_DECAY_EXPONENTS = { sublinear: 0.5, linear: 1, quadratic: 2 };
const EPSILON_DECAY_EXPONENT  =
    EPSILON_DECAY_EXPONENTS[ExperimentParams.epsilon_decay_shape] != null
        ? EPSILON_DECAY_EXPONENTS[ExperimentParams.epsilon_decay_shape]
        : EPSILON_DECAY_EXPONENTS.quadratic;

// ── Debugging ─────────────────────────────────────────────────────────────────
const DEBUG_OBSERVATIONS = false;  // Set to true to log what each eye observes per tick

// ── Helpers ───────────────────────────────────────────────────────────────────

function xavierRandom(fan_in, fan_out) {
    const limit = Math.sqrt(6 / (fan_in + fan_out));
    return (Math.random() * 2 - 1) * limit;
}

function relu(x) { return x > 0 ? x : 0; }

function softmax(arr, out_probs) {
    let max = -Infinity;
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] > max) max = arr[i];
    }
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
        const e = Math.exp(arr[i] - max);
        out_probs[i] = e;
        sum += e;
    }
    for (let i = 0; i < arr.length; i++) {
        out_probs[i] /= sum;
    }
    return out_probs;
}

function perceptIndex(name) {
    const idx = PERCEPT_INDEX[name];
    return idx !== undefined ? idx : 0;
}

// ── Main class ────────────────────────────────────────────────────────────────

class NNBrain {
    constructor(owner, rl_enabled = true) {
        this.owner      = owner;
        this.rl_enabled = rl_enabled;
        this.is_nnbrain = true;
        this.freeze_updates = false;

        this.genome_weights = new Float32Array(GENOME_SIZE);
        this._initGenome();

        // Condition A: independent copy that RL drifts. genome_weights stays frozen.
        // Condition B: same reference — no drift, no overhead.
        this.active_weights = rl_enabled
            ? new Float32Array(this.genome_weights)
            : this.genome_weights;

        this.traces           = rl_enabled ? new Float32Array(GENOME_SIZE) : null;
        this.running_baseline = 0;       // running mean of rewards for variance reduction
        this._baseline_count  = 0;       // count of reward samples for baseline
        this._hidden          = new Float32Array(HIDDEN_SIZE);
        this._input           = new Float32Array(STATE_SIZE);
        this._logits          = new Float32Array(OUTPUT_SIZE);
        this._probs           = new Float32Array(OUTPUT_SIZE);
        this._obs_buffer      = new Array(4);
        this._last_probs      = null;
        this._last_action     = null;
        // Importance-sampling ratio π(a)/behaviour(a) for the last action.
        // 1 on-policy; <1 or >1 to correct for epsilon-greedy exploration noise.
        this._last_is_ratio   = 1;
    }

    _initGenome() {
        let i = 0;
        for (let j = 0; j < W1_SIZE; j++) this.genome_weights[i++] = xavierRandom(STATE_SIZE, HIDDEN_SIZE);
        for (let j = 0; j < B1_SIZE; j++) this.genome_weights[i++] = 0;
        for (let j = 0; j < W2_SIZE; j++) this.genome_weights[i++] = xavierRandom(HIDDEN_SIZE, OUTPUT_SIZE);
        for (let j = 0; j < B2_SIZE; j++) this.genome_weights[i++] = 0;
    }

    // Inherit from parent — genome_weights only, never active_weights.
    // Non-Lamarckian: RL drift in the parent is discarded.
    copy(parent_brain) {
        this.rl_enabled     = parent_brain.rl_enabled;
        this.is_nnbrain     = true;
        this.genome_weights = new Float32Array(parent_brain.genome_weights);
        if (this.rl_enabled) {
            this.active_weights = new Float32Array(this.genome_weights);
            this.traces         = new Float32Array(GENOME_SIZE);
        } else {
            this.active_weights = this.genome_weights;
            this.traces         = null;
        }
        this._hidden      = new Float32Array(HIDDEN_SIZE);
        this._input       = new Float32Array(STATE_SIZE);
        this._logits      = new Float32Array(OUTPUT_SIZE);
        this._probs       = new Float32Array(OUTPUT_SIZE);
        this._obs_buffer  = new Array(4);
        this._last_probs  = null;
        this._last_action = null;
        this._last_is_ratio = 1;
    }

    // ── Perception ────────────────────────────────────────────────────────────

    resetTraces() {
        if (this.traces) {
            this.traces.fill(0);
        }
        this._last_probs = null;
        this._last_action = null;
    }

    // Build 54-element state vector from 4 eye observations + energy + rotation.
    // observations: array of 4 Observation objects in [up, right, down, left] order.
    buildStateVector(observations, max_energy) {
        const state = this._input;
        state.fill(0);

        for (let d = 0; d < N_EYE_DIRECTIONS; d++) {
            const obs  = observations[d];
            const base = d * N_PERCEPT_TYPES;
            let type_idx = 0;
            if (obs && obs.cell && obs.cell.state) {
                type_idx = perceptIndex(obs.cell.state.name);
                // Roaming and patrol predators share cell-state names (so both
                // map to PERCEPT_PREDATOR above); split them here via the owning
                // organism so prey perceive the two hazard types distinctly.
                if (type_idx === PERCEPT_PREDATOR &&
                    obs.cell.owner && obs.cell.owner.is_patrol) {
                    type_idx = PERCEPT_PATROL_PREDATOR;
                }
            }
            state[base + type_idx] = 1;
        }

        const energy_norm = max_energy > 0
            ? Math.min(1, Math.max(0, (this.owner.energy || 0) / max_energy))
            : 0;
        state[N_EYE_DIRECTIONS * N_PERCEPT_TYPES] = energy_norm;

        // Add rotation as normalized scalar (0->0.0, 1->0.33, 2->0.67, 3->1.0)
        const rotation_norm = this.owner.rotation / 3;
        state[N_EYE_DIRECTIONS * N_PERCEPT_TYPES + 1] = rotation_norm;

        return state;
    }

    // ── Forward pass ──────────────────────────────────────────────────────────

    forward(input_state) {
        const w = this.active_weights;

        // Hidden layer
        const b1_off = W1_SIZE;
        for (let j = 0; j < HIDDEN_SIZE; j++) {
            let sum = w[b1_off + j];
            const base = j * STATE_SIZE;
            for (let i = 0; i < STATE_SIZE; i++) sum += w[base + i] * input_state[i];
            this._hidden[j] = relu(sum);
        }

        // Output layer
        const W2_off = W1_SIZE + B1_SIZE;
        const b2_off = W2_off + W2_SIZE;
        const logits = this._logits;
        for (let k = 0; k < OUTPUT_SIZE; k++) {
            let sum = w[b2_off + k];
            const base = W2_off + k * HIDDEN_SIZE;
            for (let j = 0; j < HIDDEN_SIZE; j++) sum += w[base + j] * this._hidden[j];
            logits[k] = sum;
        }

        this._last_probs  = softmax(logits, this._probs);
        this._last_action = this._sampleCategorical(this._last_probs);
        return this._last_action;
    }

    _sampleCategorical(probs) {
        let r = Math.random();
        for (let i = 0; i < probs.length; i++) {
            r -= probs[i];
            if (r <= 0) return i;
        }
        return probs.length - 1;
    }

    // ── RL update ─────────────────────────────────────────────────────────────

    // REINFORCE with eligibility traces and running mean baseline.
    // Only updates active_weights — genome_weights is never touched.
    // The baseline reduces variance by subtracting the running mean of past rewards.
    reinforce(reward) {
        if (!this.rl_enabled || !this.traces || !this._last_probs) return;

        // Update running baseline using exponential moving average
        this._baseline_count++;
        this.running_baseline = BASELINE_DECAY * this.running_baseline + (1 - BASELINE_DECAY) * reward;
        
        // Baseline-adjusted reward for variance reduction
        const adjusted_reward = reward - this.running_baseline;

        const w      = this.active_weights;
        const traces = this.traces;
        const action = this._last_action;
        const probs  = this._last_probs;
        // Importance weight π(a)/behaviour(a); applied once to every component
        // of this step's score function ∇log π(a) before it enters the traces.
        const rho    = this._last_is_ratio;

        const W2_off = W1_SIZE + B1_SIZE;
        const b2_off = W2_off + W2_SIZE;

        for (let k = 0; k < OUTPUT_SIZE; k++) {
            const factor = rho * (((k === action ? 1 : 0)) - probs[k]);
            const base   = W2_off + k * HIDDEN_SIZE;
            for (let j = 0; j < HIDDEN_SIZE; j++) {
                traces[base + j] = TRACE_DECAY * traces[base + j] + factor * this._hidden[j];
            }
            traces[b2_off + k] = TRACE_DECAY * traces[b2_off + k] + factor;
        }

        const b1_off = W1_SIZE;
        for (let j = 0; j < HIDDEN_SIZE; j++) {
            if (this._hidden[j] <= 0) continue;
            let delta = 0;
            for (let k = 0; k < OUTPUT_SIZE; k++) {
                delta += w[W2_off + k * HIDDEN_SIZE + j] * ((k === action ? 1 : 0) - probs[k]);
            }
            delta *= rho;
            traces[b1_off + j] = TRACE_DECAY * traces[b1_off + j] + delta;
            const base = j * STATE_SIZE;
            for (let i = 0; i < STATE_SIZE; i++) {
                traces[base + i] = TRACE_DECAY * traces[base + i] + delta * this._input[i];
            }
        }

        if (this.freeze_updates) return;

        const delta_w = RL_LR * adjusted_reward;
        for (let idx = 0; idx < GENOME_SIZE; idx++) {
            let val = w[idx] + delta_w * traces[idx];
            // Clip weights to [-1, 1]
            if (val > 1.0) val = 1.0;
            else if (val < -1.0) val = -1.0;
            w[idx] = val;
        }
    }

    // Apply a single policy-gradient update using the stored eligibility traces.
    // Intended for frozen-policy experiments where weights do not change within life.
    applyFrozenUpdate(total_reward) {
        if (!this.traces || !this.genome_weights) return;
        const delta_w = RL_LR * total_reward;
        for (let idx = 0; idx < GENOME_SIZE; idx++) {
            let val = this.genome_weights[idx] + delta_w * this.traces[idx];
            if (val > 1.0) val = 1.0;
            else if (val < -1.0) val = -1.0;
            this.genome_weights[idx] = val;
            if (this.active_weights && this.active_weights !== this.genome_weights) {
                this.active_weights[idx] = val;
            }
        }
        this.resetTraces();
        this.running_baseline = 0;
        this._baseline_count = 0;
    }

    // ── Main entry point ──────────────────────────────────────────────────────

    /**
     * Called once per tick by AdvancedOrganism.update().
     *
     * @param {number} reward        reward signal this tick
     * @param {number} max_energy    for normalising energy input
     * @param {number} lifetime_frac 0.0 at birth -> 1.0 at max lifespan, for epsilon decay
     * @returns {number}  Action index 0-5 (0-3: movement directions, 4-5: rotations)
     */
    decide(reward, max_energy, lifetime_frac) {
        const observations = this._collectObservations();
        const state        = this.buildStateVector(observations, max_energy);

        // Debug: log what each eye observes
        if (DEBUG_OBSERVATIONS) {
            const dirNames = ['UP', 'RIGHT', 'DOWN', 'LEFT'];
            let obsLog = `[Org #${this.owner.id} tick ${this.owner.lifetime}] Eyes: `;
            for (let d = 0; d < N_EYE_DIRECTIONS; d++) {
                const obs = observations[d];
                const dirName = dirNames[d];
                if (obs && obs.cell && obs.cell.state) {
                    const cellName = obs.cell.state.name || 'unknown';
                    const distance = obs.distance || '?';
                    const perceptIdx = perceptIndex(cellName);
                    obsLog += `${dirName}(${cellName}@${distance}🔍${perceptIdx}) `;
                } else {
                    obsLog += `${dirName}(empty) `;
                }
            }
            console.log(obsLog);
        }

        if (DEBUG_STATE_VECTOR) {
            const dirNames = ['UP', 'RIGHT', 'DOWN', 'LEFT'];
            const parts = [];
            for (let d = 0; d < N_EYE_DIRECTIONS; d++) {
                const base = d * N_PERCEPT_TYPES;
                const slice = Array.from(state.slice(base, base + N_PERCEPT_TYPES)).map(v => v.toFixed(3));
                const obs = observations[d];
                const cellName = obs && obs.cell && obs.cell.state ? obs.cell.state.name : 'empty';
                parts.push(`${dirNames[d]}:${cellName} [${slice.join(', ')}]`);
            }
            const energy = state[N_EYE_DIRECTIONS * N_PERCEPT_TYPES].toFixed(3);
            const rotation = state[N_EYE_DIRECTIONS * N_PERCEPT_TYPES + 1].toFixed(3);
            parts.push(`energy:${energy}`);
            parts.push(`rotation:${rotation}`);
            console.log(`[Org #${this.owner.id} tick ${this.owner.lifetime}] State: ${parts.join(' | ')}`);
        }

        // Epsilon-greedy exploration only applies in RL conditions. In GA-only
        // (rl_enabled=false) organisms act purely on their inherited policy —
        // random action injection would corrupt the GA baseline since organisms
        // would be spending up to 50% of their ticks on random behaviour
        // regardless of what the genome encodes.
        let action;
        if (this.rl_enabled) {
            // epsilon=0 makes the branch below pure on-policy: Math.random() < 0
            // is never true, and b_a collapses to pi_a so the importance ratio is 1.
            const epsilon = EPSILON_ENABLED
                ? EPSILON_START + (EPSILON_END - EPSILON_START) * (lifetime_frac ** EPSILON_DECAY_EXPONENT)
                : 0;
            if (Math.random() < epsilon) {
                // Explore: forward pass caches hidden activations AND the true
                // softmax π (left in this._last_probs); only the chosen action
                // is overridden with a uniform-random one. Crucially we do NOT
                // overwrite _last_probs with a uniform distribution — REINFORCE
                // needs the real π(a) both for the policy gradient and for the
                // importance ratio below.
                this.forward(state);
                action = Math.floor(Math.random() * OUTPUT_SIZE);
                this._last_action = action;
            } else {
                action = this.forward(state);
            }

            // Importance-sampling correction. The behaviour policy is the
            // epsilon-greedy mixture over the softmax, b(a) = (1-ε)π(a) + ε/N,
            // while the gradient targets π. ρ = π(a)/b(a) reweights the update
            // so exploratory actions don't bias learning. ρ → 1 as ε → 0, and
            // ρ is bounded in ~[0, 1/(1-ε)], but clamp defensively.
            const pi_a = this._last_probs[action];
            const b_a  = (1 - epsilon) * pi_a + epsilon / OUTPUT_SIZE;
            this._last_is_ratio = b_a > 1e-12 ? Math.min(pi_a / b_a, 10) : 1;

            this.reinforce(reward);
        } else {
            // Pure policy execution — no exploration noise, no weight updates.
            action = this.forward(state);
        }
        return action;
    }

    // Collect one Observation per cardinal direction from the organism's eye cells.
    // Expects the anatomy to have 4 eye cells, one per direction.
    // Slots without an eye cell return null (encoded as "nothing" in buildStateVector).
    _collectObservations() {
        const obs_by_dir = this._obs_buffer;
        obs_by_dir[0] = null;
        obs_by_dir[1] = null;
        obs_by_dir[2] = null;
        obs_by_dir[3] = null;
        
        const cells = this.owner.anatomy.cells;
        for (let i = 0; i < cells.length; i++) {
            const cell = cells[i];
            if (typeof cell.look === 'function') {
                const abs_dir = cell.getAbsoluteDirection();
                let slot = -1;
                if (abs_dir === Directions.up) slot = 0;
                else if (abs_dir === Directions.right) slot = 1;
                else if (abs_dir === Directions.down) slot = 2;
                else if (abs_dir === Directions.left) slot = 3;
                if (slot !== -1) obs_by_dir[slot] = cell.look();
            }
        }
        return obs_by_dir;
    }

    // ── GA interface ──────────────────────────────────────────────────────────

    // Returns genome_weights only. RL drift (active_weights) is never exposed.
    getGenome() {
        return this.genome_weights;
    }

    setGenome(new_weights) {
        if (new_weights.length !== GENOME_SIZE) {
            throw new Error(`NNBrain.setGenome: expected ${GENOME_SIZE}, got ${new_weights.length}`);
        }
        this.genome_weights = new Float32Array(new_weights);
        if (this.rl_enabled) {
            this.active_weights = new Float32Array(this.genome_weights);
            this.traces.fill(0);
        } else {
            this.active_weights = this.genome_weights;
        }
        this._hidden.fill(0);
        this._input.fill(0);
        this._last_probs  = null;
        this._last_action = null;
    }

    // Copy the current learned policy into the inherited genome so future
    // offspring continue from the updated weights.
    syncGenomeFromActive() {
        if (!this.active_weights || !this.genome_weights) return;
        this.genome_weights = new Float32Array(this.active_weights);
    }

    // ── Serialisation ─────────────────────────────────────────────────────────

    serialize() {
        return {
            type:           'NNBrain',
            rl_enabled:     this.rl_enabled,
            genome_weights: Array.from(this.genome_weights),
        };
    }

    loadSerialized(data) {
        if (data.type !== 'NNBrain') throw new Error('NNBrain.loadSerialized: wrong type tag');
        this.rl_enabled = data.rl_enabled;
        this.setGenome(new Float32Array(data.genome_weights));
    }

    // Stub methods for compatibility with Anatomy.js cell addition/removal tracking
    // NNBrain has fixed input architecture so no dynamic updates needed
    checkAddedCell(cell) {
        // No-op: NNBrain input layer is fixed (50 elements)
    }

    checkRemovedCell(cell) {
        // No-op: NNBrain input layer is fixed (50 elements)
    }

    // Stub method for compatibility with EditorController
    // NNBrain is a fixed-architecture neural network, not a state machine
    countCells() {
        // Count eye cells from the organism's anatomy
        this.eye_cell_count = 0;
        for (const cell of this.owner.anatomy.cells) {
            if (cell.state && cell.state.name === 'eye') {
                this.eye_cell_count++;
            }
        }
    }

    // Stub properties for EditorController compatibility
    // NNBrain doesn't use state machines like Brain does
    get num_states() {
        return 1;  // NNBrain is not a state machine
    }

    get independent_eye_decisions() {
        return false;  // NNBrain doesn't have independent eye decisions
    }

    setIndependentEyeDecisions(enabled) {
        // No-op: NNBrain doesn't support this feature
    }

    // Stub method for compatibility with EyeCell.performFunction()
    // NNBrain reads eye observations directly via buildStateVector(), not via observe() pipeline
    observe(observation) {
        // No-op: NNBrain bypasses the Brain.observe() / Brain.decide() pipeline
    }
}

NNBrain.GENOME_SIZE      = GENOME_SIZE;
NNBrain.STATE_SIZE       = STATE_SIZE;
NNBrain.HIDDEN_SIZE      = HIDDEN_SIZE;
NNBrain.OUTPUT_SIZE      = OUTPUT_SIZE;
NNBrain.N_PERCEPT_TYPES  = N_PERCEPT_TYPES;
NNBrain.N_EYE_DIRECTIONS = N_EYE_DIRECTIONS;
NNBrain.PERCEPT_INDEX    = PERCEPT_INDEX;
NNBrain.PERCEPT_PREDATOR        = PERCEPT_PREDATOR;
NNBrain.PERCEPT_PATROL_PREDATOR = PERCEPT_PATROL_PREDATOR;
NNBrain.EPSILON_START    = EPSILON_START;
NNBrain.EPSILON_END      = EPSILON_END;
NNBrain.EPSILON_DECAY_EXPONENT = EPSILON_DECAY_EXPONENT;
NNBrain.EPSILON_ENABLED  = EPSILON_ENABLED;

module.exports = NNBrain;