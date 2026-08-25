'use strict';

/**
 * src/eval/validate.js — the verification suite.
 *
 *   node src/eval/validate.js                     # everything, default scale
 *   node src/eval/validate.js --only grad,mut     # a subset
 *   node src/eval/validate.js --list              # what can be run
 *   node src/eval/validate.js --seed 2001 --width 250 --height 250
 *
 * Every check here is written to double as a REPORTABLE CLAIM, not just a
 * regression guard: each one prints the measured quantity next to its expected
 * value, and the whole run is dumped to output/validation/verification_report.json
 * so a number quoted in the write-up can be traced back to the run that produced
 * it. Exit status is 0 only if every check selected passed.
 *
 * ── The checks ───────────────────────────────────────────────────────────────
 *
 *  mad     Non-Lamarckian invariant. In the evolution condition an organism's
 *          active weights ARE its genome (same Float32Array object), so the mean
 *          absolute weight difference between them is not merely small but identically
 *          zero. Checked three ways: object identity at construction, the live
 *          per-generation metric over a real short run, and a scan of every
 *          evolution-condition generations.csv already on disk.
 *
 *  maps    Terrain identity across conditions. Digests map_pool_seed<N>.json and
 *          then — the part that actually matters — digests the CELL GRID each
 *          condition ends up simulating on, after generateWorld() has scaled the
 *          pool onto the runtime grid. Equal digests are the evidence for
 *          "conditions differ only in the adaptation mechanism".
 *
 *  grad    Gradient check on NNBrain.reinforce(). The eligibility trace laid
 *          down by one update, with the trace zeroed first, is exactly
 *          ∇log π(a) — no backprop library computes it, so it is checked against
 *          a central finite difference of log π(a) taken through an independent
 *          forward pass. This is the claim a reader is most likely to doubt.
 *
 *  mut     Mutation-kernel identity. "Verified identical" is made to mean
 *          something: the three inter-generation kernels are driven off one
 *          pinned PRNG stream and their outputs compared bit for bit.
 *
 *  cave    Cave-at-night energy rule. A decay event that fires while an organism
 *          is inside a cave at night must move its energy by exactly 0.0.
 *
 *  drain   Predation accounting. Energy lost to predators equals the number of
 *          drain CONTACTS times PredatorHyperparameters.drainAmount, exactly.
 *          Also reports how contacts relate to the de-duplicated drained_ticks
 *          metric, which is what generations.csv actually carries.
 *
 *  ledger  Energy conservation. start + food - decay - predation == energy, for
 *          every organism in a live run. Nothing else may move energy.
 *
 *  floor   Random-policy floor smoke test. Confirms --random-floor really does
 *          re-randomise founders every generation (genome variance stays at the
 *          Xavier level, fitness does not trend), so the anchor runs measure
 *          chance rather than a crippled GA.
 *
 * The sim modules read their configuration at require() time, so this file
 * calls applyConfig() before touching any of them and every check that needs a
 * different configuration runs in a FORKED child process (see runIsolated).
 */

const path = require('path');
const fs   = require('fs');
const crypto = require('crypto');
const { execFileSync } = require('child_process');

const REPO = path.join(__dirname, '..', '..');

// ── CLI ──────────────────────────────────────────────────────────────────────

function parseArgs(argv) {
    const o = {};
    for (let i = 0; i < argv.length; i++) {
        if (!argv[i].startsWith('--')) continue;
        const k = argv[i].slice(2);
        const n = argv[i + 1];
        if (n !== undefined && !n.startsWith('--')) { o[k] = n; i++; } else o[k] = true;
    }
    return o;
}
const opts = parseArgs(process.argv.slice(2));

const ALL_CHECKS = ['mad', 'maps', 'grad', 'mut', 'cave', 'drain', 'ledger', 'floor'];
if (opts.list) { console.log(ALL_CHECKS.join('\n')); process.exit(0); }

const SELECTED = opts.only
    ? String(opts.only).split(',').map(s => s.trim()).filter(Boolean)
    : ALL_CHECKS;
for (const c of SELECTED) {
    if (!ALL_CHECKS.includes(c)) {
        console.error(`Unknown check "${c}". Known: ${ALL_CHECKS.join(', ')}`);
        process.exit(2);
    }
}

// Scale knobs. The defaults are chosen so the whole suite finishes in a couple
// of minutes on a laptop while still exercising the real code paths; --width /
// --height / --generations push it towards production scale.
const SEED        = opts.seed        ? parseInt(opts.seed, 10)        : 2001;
const WIDTH       = opts.width       ? parseInt(opts.width, 10)       : 200;
const HEIGHT      = opts.height      ? parseInt(opts.height, 10)      : 200;
const GENERATIONS = opts.generations ? parseInt(opts.generations, 10) : 3;
const HIDDEN      = opts['hidden-size'] ? parseInt(opts['hidden-size'], 10) : 128;
const LOGS_ROOTS  = opts['logs'] ? [String(opts.logs)] : ['logs', 'logs_hard'];
// organisms.csv runs to ~100 MB per run, so the production cave scan streams a
// bounded sample rather than the whole 42 GB tree. 0 skips it.
const ORG_SCAN    = opts['org-scan'] ? parseInt(opts['org-scan'], 10) : 3;

// ── Result plumbing ──────────────────────────────────────────────────────────

const results = [];

/** Record one check. `facts` are the reportable numbers; they go in the JSON. */
function record(name, ok, headline, facts) {
    results.push({ check: name, passed: !!ok, headline, facts: facts || {} });
    const tag = ok ? '  ✓ PASS' : '  ✗ FAIL';
    console.log(`${tag}  ${headline}`);
    for (const [k, v] of Object.entries(facts || {})) {
        console.log(`         ${k}: ${typeof v === 'object' ? JSON.stringify(v) : v}`);
    }
}

function section(title) { console.log(`\n=== ${title} ===`); }

/**
 * Run a check in a fresh Node process. Needed because ExperimentParams /
 * WorldConfig are baked into the sim modules at require() time, so two checks
 * wanting different conditions cannot share one process.
 *
 * The child runs `--child <name>` and prints one line of JSON prefixed with
 * RESULT: — everything else it writes is passed through as sim chatter.
 */
function runIsolated(name, env) {
    const args = [__filename, '--child', name,
        '--seed', String(SEED), '--width', String(WIDTH), '--height', String(HEIGHT),
        '--generations', String(GENERATIONS), '--hidden-size', String(HIDDEN)];
    const out = execFileSync(process.execPath, args, {
        cwd: REPO, encoding: 'utf8', maxBuffer: 256 * 1024 * 1024,
        env: Object.assign({}, process.env, env || {}),
    });
    const line = out.split('\n').filter(l => l.startsWith('RESULT:')).pop();
    if (!line) throw new Error(`child "${name}" produced no RESULT line:\n${out.slice(-2000)}`);
    return JSON.parse(line.slice('RESULT:'.length));
}

function emit(obj) { console.log('RESULT:' + JSON.stringify(obj)); }

// ─────────────────────────────────────────────────────────────────────────────
// Shared sim boot
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Boot a real WorldEnvironment for one condition and step it for `gens`
 * generations, exactly as headless.js does. Returns the env plus the Logger's
 * in-memory generation log — which is the UNROUNDED source the CSV is written
 * from, so a metric can be checked at full precision rather than at the 6 d.p.
 * the file carries.
 */
function bootAndRun(condition, mode, gens, extra) {
    const { installBrowserStubs } = require('./config');
    installBrowserStubs();

    const ExperimentParams = require('../ExperimentParams');
    ExperimentParams.applyOverrides(Object.assign({
        hidden_size: HIDDEN,
        epsilon_enabled: false,
        learning_rate: 0.02,
        explore_bonus: 0,
    }, (extra && extra.params) || {}));

    const WorldConfig = require('../WorldConfig');
    WorldConfig.headless         = true;
    WorldConfig.auto_pause       = false;
    WorldConfig.auto_reset       = false;
    WorldConfig.MAP_SEED         = SEED;
    WorldConfig.MAP_COLS         = WIDTH;
    WorldConfig.MAP_ROWS         = HEIGHT;
    WorldConfig.learning_enabled = (condition === 'learning');
    WorldConfig.experiment_mode  = mode || 'standard';

    const PredatorHyperparameters = require('../Organism/PredatorHyperparameters');
    const pred = (extra && extra.predators) || {};
    PredatorHyperparameters.drainAmount              = pred.drain != null ? pred.drain : 5;
    PredatorHyperparameters.count                    = pred.roaming != null ? pred.roaming : 0;
    PredatorHyperparameters.patrol.predatorsPerPatch = pred.patrol != null ? pred.patrol : 0;

    const logger = require('../Logger');
    logger.setGenomeLogging(false);
    // Keep the suite from writing CSVs into the real logs tree.
    if (typeof logger.setAutoSave === 'function') logger.setAutoSave(false);
    logger.auto_save = false;

    const WorldEnvironment    = require('../Environments/WorldEnvironment');
    const GenerationConstants = require('../Organism/GenerationConstants');

    const engine = { fps: Infinity, last_fps: Infinity, running: true,
        stop() { this.running = false; }, start() {}, restart() {} };
    const env = new WorldEnvironment(engine, 2);
    env.generateWorld();
    env.ga_manager.spawnGeneration();
    env.predator_manager.spawnAll();

    // `stopAtTick` stops the loop at a raw tick count instead of a generation
    // boundary. The audit checks need it: evolve() is immediately followed by
    // spawnGeneration(), which clears all_agents, so a run stopped on a boundary
    // hands back a cohort that has not lived a single tick and every audit
    // counter reads 0 — a vacuously passing check. Stopping one tick short of
    // the boundary leaves the full generation, dead agents included, in place.
    const target = (extra && extra.stopAtTick)
        ? extra.stopAtTick
        : gens * GenerationConstants.TICKS_PER_GEN;
    while (env.total_ticks < target && engine.running) {
        env.update();
        if (!(extra && extra.stopAtTick) &&
            env.ga_manager.generation != null && env.ga_manager.generation > gens) break;
        if (extra && extra.onTick) extra.onTick(env);
    }
    return { env, logger, gen_log: logger.generation_log || [],
             ticks_per_gen: GenerationConstants.TICKS_PER_GEN };
}

/** sha256 of the runtime cell grid: the terrain the organisms actually see. */
function digestGrid(env) {
    const g = env.grid_map;
    const names = [];
    for (let r = 0; r < g.rows; r++) {
        for (let c = 0; c < g.cols; c++) {
            const cell = g.cellAt(c, r);
            names.push(cell && cell.state ? cell.state.name : '?');
        }
    }
    return crypto.createHash('sha256').update(names.join('|')).digest('hex');
}

// ─────────────────────────────────────────────────────────────────────────────
// CHECK: mad — the non-Lamarckian invariant
// ─────────────────────────────────────────────────────────────────────────────

/** Child half: run both conditions' founder construction + a live evolution run. */
function childMad() {
    const { env, gen_log } = bootAndRun('evolution', 'standard', GENERATIONS);

    // (a) object identity at construction — the mechanism, not the symptom.
    let founders = 0, aliased = 0, traces_null = 0, worst_mad = 0;
    for (const org of env.ga_manager.founders) {
        const b = org.brain;
        if (!b || !b.genome_weights) continue;
        founders++;
        if (b.active_weights === b.genome_weights) aliased++;
        if (b.traces === null || b.traces === undefined) traces_null++;
        let s = 0;
        for (let i = 0; i < b.genome_weights.length; i++) {
            s += Math.abs(b.active_weights[i] - b.genome_weights[i]);
        }
        worst_mad = Math.max(worst_mad, s / b.genome_weights.length);
    }

    // (b) the live per-generation metric, at full float precision.
    const mads = gen_log.map(e => parseFloat(e.avg_learned_weight_diff));
    emit({
        founders, aliased, traces_null, worst_mad,
        generations: mads.length,
        max_logged_mad: mads.length ? Math.max.apply(null, mads) : null,
        all_logged_zero: mads.every(m => m === 0),
    });
}

/** Scan every evolution-condition generations.csv on disk. */
function scanLoggedMad() {
    const rows = [];
    const walk = (dir) => {
        let entries;
        try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch (_) { return; }
        for (const e of entries) {
            const p = path.join(dir, e.name);
            if (e.isDirectory()) { walk(p); continue; }
            if (e.name !== 'params.json') continue;
            const gen = path.join(dir, 'generations.csv');
            if (!fs.existsSync(gen)) continue;
            let params;
            try { params = JSON.parse(fs.readFileSync(p, 'utf8')); } catch (_) { continue; }
            if (String(params.condition) !== 'evolution') continue;
            if (String(params.mode || 'standard') !== 'standard') continue;
            rows.push({ dir, gen, seed: params.map_seed });
        }
    };
    for (const root of LOGS_ROOTS) walk(path.join(REPO, root));

    let runs = 0, gens = 0, nonzero = 0, worst = 0;
    const offenders = [];
    for (const r of rows) {
        const text = fs.readFileSync(r.gen, 'utf8').split('\n');
        if (text.length < 2) continue;
        const header = text[0].split(',');
        const idx = header.indexOf('avg_learned_weight_diff');
        if (idx < 0) continue;
        runs++;
        for (let i = 1; i < text.length; i++) {
            if (!text[i]) continue;
            const v = parseFloat(text[i].split(',')[idx]);
            if (!isFinite(v)) continue;
            gens++;
            if (v !== 0) {
                nonzero++;
                worst = Math.max(worst, Math.abs(v));
                if (offenders.length < 5) offenders.push(`${path.basename(r.dir)} gen-row ${i}: ${v}`);
            }
        }
    }
    return { runs, generation_rows: gens, nonzero_rows: nonzero, worst_value: worst, offenders };
}

function checkMad() {
    section('mad — non-Lamarckian invariant: mean absolute weight difference is identically zero under evolution');
    const live = runIsolated('mad');
    const ok_live = live.founders > 0 &&
                    live.aliased === live.founders &&
                    live.traces_null === live.founders &&
                    live.worst_mad === 0 &&
                    live.all_logged_zero === true;
    record('mad_live', ok_live,
        `live evolution run: mean absolute weight difference == 0 on all ${live.generations} generations, ` +
        `active_weights aliases genome_weights for ${live.aliased}/${live.founders} founders`,
        {
            founders: live.founders,
            'active_weights === genome_weights': `${live.aliased}/${live.founders}`,
            'eligibility traces allocated': `${live.founders - live.traces_null}/${live.founders} (expected 0)`,
            'max per-founder mean absolute weight difference': live.worst_mad,
            'max logged avg_learned_weight_diff': live.max_logged_mad,
        });

    const scan = scanLoggedMad();
    const ok_scan = scan.runs > 0 && scan.nonzero_rows === 0;
    record('mad_logged_runs', ok_scan,
        scan.runs === 0
            ? 'no evolution-condition runs found on disk to scan'
            : `${scan.runs} evolution runs on disk: 0/${scan.generation_rows} generation rows ` +
              'carry a nonzero avg_learned_weight_diff',
        {
            'runs scanned': scan.runs,
            'generation rows': scan.generation_rows,
            'nonzero rows': scan.nonzero_rows,
            'worst value': scan.worst_value,
            'first offenders': scan.offenders,
            'roots searched': LOGS_ROOTS.join(', '),
        });
}

// ─────────────────────────────────────────────────────────────────────────────
// CHECK: maps — identical terrain across conditions
// ─────────────────────────────────────────────────────────────────────────────

function childMaps() {
    const cond = process.env.VALIDATE_CONDITION || 'evolution';
    const mode = process.env.VALIDATE_MODE || 'standard';
    const { installBrowserStubs } = require('./config');
    installBrowserStubs();
    const ExperimentParams = require('../ExperimentParams');
    ExperimentParams.applyOverrides({ hidden_size: HIDDEN, epsilon_enabled: false });
    const WorldConfig = require('../WorldConfig');
    WorldConfig.headless = true; WorldConfig.auto_pause = false; WorldConfig.auto_reset = false;
    WorldConfig.MAP_SEED = SEED; WorldConfig.MAP_COLS = WIDTH; WorldConfig.MAP_ROWS = HEIGHT;
    WorldConfig.learning_enabled = (cond === 'learning');
    WorldConfig.experiment_mode  = mode;
    const PH = require('../Organism/PredatorHyperparameters');
    PH.drainAmount = 5; PH.count = 0; PH.patrol.predatorsPerPatch = 0;
    const logger = require('../Logger');
    logger.setGenomeLogging(false); logger.auto_save = false;

    const WorldEnvironment = require('../Environments/WorldEnvironment');
    const engine = { fps: Infinity, running: true, stop() {}, start() {}, restart() {} };
    const env = new WorldEnvironment(engine, 2);

    // Digest every map in the pool sequence, not just the first: a condition
    // that differed only on map 7 would slip past a single-map check.
    const digests = [];
    const n = Math.min(env._map_sequence.length, 10);
    env._sequence_index = 0;
    for (let i = 0; i < n; i++) {
        env.generateWorld();
        digests.push(digestGrid(env));
    }
    emit({ condition: cond, mode, maps: n, digests,
           combined: crypto.createHash('sha256').update(digests.join('|')).digest('hex'),
           pool_size: env._map_pool ? env._map_pool.length : 0,
           map_seed: env.map_seed });
}

function checkMaps() {
    section('maps — the three conditions simulate byte-identical terrain');

    // (a) the pool file itself
    const poolFile = path.join(REPO, 'src', 'maps', `map_pool_seed${SEED}.json`);
    let fileDigest = null;
    if (fs.existsSync(poolFile)) {
        fileDigest = crypto.createHash('sha256').update(fs.readFileSync(poolFile)).digest('hex');
    }

    // (b) the grid each condition actually simulates on
    const arms = [
        { key: 'evolution', condition: 'evolution', mode: 'standard' },
        { key: 'learning',  condition: 'learning',  mode: 'standard' },
        { key: 'pure_rl',   condition: 'learning',  mode: 'pure_rl'  },
    ];
    const got = arms.map(a => Object.assign({ key: a.key }, runIsolated('maps', {
        VALIDATE_CONDITION: a.condition, VALIDATE_MODE: a.mode,
    })));

    const combined = got.map(g => g.combined);
    const ok = combined.every(d => d === combined[0]) && got[0].maps > 0;
    const facts = {
        'map_pool_seed file': poolFile.replace(REPO + path.sep, ''),
        'file sha256': fileDigest ? fileDigest.slice(0, 16) + '…' : '(not cached yet)',
        'pool size': got[0].pool_size,
        'maps digested per condition': got[0].maps,
        'grid': `${WIDTH}x${HEIGHT}`,
    };
    for (const g of got) facts[`${g.key} terrain sha256`] = g.combined.slice(0, 16) + '…';
    record('map_terrain_identity', ok,
        ok ? `all 3 conditions produce the same sha256 over the first ${got[0].maps} maps of seed ${SEED}`
           : 'TERRAIN DIFFERS between conditions', facts);
}

// ─────────────────────────────────────────────────────────────────────────────
// CHECK: grad — finite-difference check on reinforce()
// ─────────────────────────────────────────────────────────────────────────────

/**
 * NNBrain does policy gradient without an autodiff library: reinforce() writes
 * ∇log π(a) straight into the eligibility trace by hand. With the trace zeroed
 * beforehand, trace <- γ·0 + ∇log π(a) leaves exactly the gradient in it, which
 * can be compared against a central finite difference of log π(a) computed by an
 * independent forward pass.
 *
 * Two details make the comparison honest:
 *   - freeze_updates stops reinforce() from moving the weights it is being
 *     measured at, so the finite difference is taken at the same point.
 *   - the ReLU kink is a genuine non-differentiability, so a probe is only used
 *     where the pre-activation sits further than the FD step from zero. Skipped
 *     coordinates are counted and reported rather than quietly dropped.
 */
function childGrad() {
    const { installBrowserStubs } = require('./config');
    installBrowserStubs();
    const ExperimentParams = require('../ExperimentParams');
    // A small hidden layer keeps the O(n) finite differencing cheap; the code
    // path under test does not depend on the width.
    ExperimentParams.applyOverrides({ hidden_size: 16, epsilon_enabled: false, trace_decay: 0.9 });
    const NNBrain = require('../Organism/Perception/NNBrain');

    const S = NNBrain.STATE_SIZE, H = NNBrain.HIDDEN_SIZE, O = NNBrain.OUTPUT_SIZE;
    const W1 = S * H, B1 = H, W2 = H * O;
    const N = NNBrain.GENOME_SIZE;

    // A brain with a stub owner: forward() only reads owner.energy / owner.rotation
    // through buildStateVector, which is bypassed here by driving forward() directly.
    const owner = { energy: 250, rotation: 1, anatomy: { cells: [] }, id: 0, lifetime: 0 };
    const brain = new NNBrain(owner, true);

    // Deterministic weights and state so a failure is reproducible.
    let s = 12345;
    const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
    for (let i = 0; i < N; i++) brain.active_weights[i] = (rnd() * 2 - 1) * 0.4;

    // A realistic state vector: one-hot percept per eye plus the two scalars.
    // It is written INTO brain._input rather than into a fresh array, because
    // that is the aliasing production has: decide() passes the array returned by
    // buildStateVector(), which IS this._input, and reinforce() reads this._input
    // when it builds the W1 half of the trace. Handing forward() an unaliased
    // array would leave this._input at zero and silently zero the W1 gradient —
    // so the check would be measuring a configuration the simulator never runs in.
    const state = brain._input;
    state.fill(0);
    for (let d = 0; d < NNBrain.N_EYE_DIRECTIONS; d++) {
        state[d * NNBrain.N_PERCEPT_TYPES + Math.floor(rnd() * NNBrain.N_PERCEPT_TYPES)] = 1;
    }
    state[S - 2] = 0.63;
    state[S - 1] = 1 / 3;

    // Independent forward pass — deliberately NOT reusing brain.forward(), so a
    // bug shared between the forward pass and the gradient cannot hide.
    const logProb = (w, action) => {
        const hid = new Float64Array(H);
        for (let j = 0; j < H; j++) {
            let sum = w[W1 + j];
            for (let i = 0; i < S; i++) sum += w[j * S + i] * state[i];
            hid[j] = sum > 0 ? sum : 0;
        }
        const logits = new Float64Array(O);
        for (let k = 0; k < O; k++) {
            let sum = w[W1 + B1 + W2 + k];
            for (let j = 0; j < H; j++) sum += w[W1 + B1 + k * H + j] * hid[j];
            logits[k] = sum;
        }
        let mx = -Infinity;
        for (let k = 0; k < O; k++) if (logits[k] > mx) mx = logits[k];
        let se = 0;
        for (let k = 0; k < O; k++) se += Math.exp(logits[k] - mx);
        return logits[action] - mx - Math.log(se);
    };

    const preActivations = (w) => {
        const z = new Float64Array(H);
        for (let j = 0; j < H; j++) {
            let sum = w[W1 + j];
            for (let i = 0; i < S; i++) sum += w[j * S + i] * state[i];
            z[j] = sum;
        }
        return z;
    };

    // One forward pass to populate _hidden / _input / _last_probs, then pick the
    // action by hand so the check covers a chosen coordinate rather than a
    // sampled one, and so it is reproducible.
    brain.forward(state);
    const action = 3;
    brain._last_action  = action;
    brain._last_is_ratio = 1;          // on-policy: the production setting

    // Zero the trace so trace == gradient after one call, and freeze the weight
    // step so the finite difference is evaluated at the same weights.
    brain.traces.fill(0);
    brain.freeze_updates = true;
    brain.reinforce(1.0);
    const analytic = Float64Array.from(brain.traces);

    const w = brain.active_weights;
    const eps = 1e-3;               // float32 weights: smaller steps lose to rounding
    const z = preActivations(w);

    // Probe a spread of coordinates from each of the four parameter blocks.
    const picks = [];
    const push = (lo, hi, howMany) => {
        for (let t = 0; t < howMany; t++) picks.push(lo + Math.floor(rnd() * (hi - lo)));
    };
    push(0, W1, 120);                       // W1
    push(W1, W1 + B1, H);                   // b1 (all of it)
    push(W1 + B1, W1 + B1 + W2, 60);        // W2
    push(W1 + B1 + W2, N, O);               // b2 (all of it)

    let maxAbs = 0, maxRel = 0, worst = null, skipped = 0, tested = 0;
    for (const idx of picks) {
        // Skip coordinates whose ReLU unit sits within the FD step of its kink:
        // there the two-sided difference straddles a genuine non-differentiability.
        let unit = -1;
        if (idx < W1)                unit = Math.floor(idx / S);
        else if (idx < W1 + B1)      unit = idx - W1;
        if (unit >= 0) {
            const scale = (idx < W1) ? Math.abs(state[idx % S]) : 1;
            if (Math.abs(z[unit]) < 4 * eps * Math.max(scale, 1e-6)) { skipped++; continue; }
            // A dead unit contributes nothing either side — gradient 0, FD 0.
        }
        // Divide by the step the float32 array actually took, not by 2*eps:
        // w0 +/- eps is rounded on store, and using the nominal step would charge
        // that rounding to the gradient.
        const w0 = w[idx];
        w[idx] = w0 + eps; const wp = w[idx]; const lp = logProb(w, action);
        w[idx] = w0 - eps; const wm = w[idx]; const lm = logProb(w, action);
        w[idx] = w0;
        const fd = (lp - lm) / (wp - wm);
        const an = analytic[idx];
        const abs = Math.abs(fd - an);
        const rel = abs / Math.max(1e-8, Math.abs(fd), Math.abs(an));
        tested++;
        if (abs > maxAbs) maxAbs = abs;
        // Only rank the relative error where there is a gradient to be relative
        // to: a coordinate whose true derivative is 0 (an eye percept that is
        // off, a dead ReLU unit) divides a rounding-level absolute error by
        // nothing and would dominate the maximum with noise.
        if (Math.max(Math.abs(fd), Math.abs(an)) > 1e-4 && rel > maxRel) {
            maxRel = rel; worst = { idx, fd, an };
        }
    }

    // Cosine similarity over every coordinate the FD is worth taking on: a
    // single scalar that says "the update points the same way the gradient does",
    // which is the direction claim the Methodology section makes.
    let dot = 0, na = 0, nf = 0, n_full = 0;
    for (let idx = 0; idx < N; idx += 7) {          // stride: full FD over 7814 weights is slow
        let unit = -1;
        if (idx < W1) unit = Math.floor(idx / S);
        else if (idx < W1 + B1) unit = idx - W1;
        if (unit >= 0 && Math.abs(z[unit]) < 4 * eps) continue;
        const w0 = w[idx];
        w[idx] = w0 + eps; const wp = w[idx]; const lp = logProb(w, action);
        w[idx] = w0 - eps; const wm = w[idx]; const lm = logProb(w, action);
        w[idx] = w0;
        const fd = (lp - lm) / (wp - wm);
        const an = analytic[idx];
        dot += fd * an; na += an * an; nf += fd * fd; n_full++;
    }
    const cosine = (na > 0 && nf > 0) ? dot / Math.sqrt(na * nf) : null;

    emit({ tested, skipped_relu_kink: skipped, max_abs_err: maxAbs, max_rel_err: maxRel,
           worst, cosine, cosine_coords: n_full, eps, hidden_size: H, action,
           probs: Array.from(brain._last_probs) });
}

function checkGrad() {
    section('grad — reinforce() lays down ∇log π(a): finite-difference check');
    const r = runIsolated('grad');
    // float32 weights with a 1e-3 step: ~1e-4 absolute agreement is the floor
    // set by rounding, not by the implementation.
    const ok = r.tested > 100 && r.max_rel_err < 1e-3 && r.cosine > 0.999999;
    record('reinforce_gradient', ok,
        `eligibility trace matches the finite-difference ∇log π(a): max relative error ` +
        `${r.max_rel_err.toExponential(2)} over ${r.tested} coordinates, cosine similarity ` +
        `${r.cosine.toFixed(9)}`,
        {
            'coordinates finite-differenced': r.tested,
            'skipped (within FD step of a ReLU kink)': r.skipped_relu_kink,
            'max absolute error': r.max_abs_err.toExponential(3),
            'max relative error': r.max_rel_err.toExponential(3),
            'cosine similarity (stride-7 sweep)': r.cosine,
            'coordinates in cosine': r.cosine_coords,
            'FD step': r.eps,
            'hidden size used': r.hidden_size,
            'action scored': r.action,
        });
}

// ─────────────────────────────────────────────────────────────────────────────
// CHECK: mut — mutation kernels are identical across conditions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * EXPERIMENTS.md §3.4 claims the conditions share "the same two mutation
 * kernels (verified identical)". The inter-generation kernel is written out
 * THREE times — GAManager._mutate, PureRLManager._mutateWeights, and (at a
 * different rate) AdvancedOrganism._mutateGenome — so identity is a property
 * that has to be measured, not read off the call graph.
 *
 * The measurement: pin Math.random to one deterministic stream, run each kernel
 * over the same starting genome from the same stream position, and compare the
 * outputs bit for bit. Identical bytes means identical distribution AND
 * identical PRNG consumption, which is the stronger statement — two kernels
 * that sampled the same distribution but drew a different number of variates
 * would still fail this, and should.
 */
function childMut() {
    const { installBrowserStubs } = require('./config');
    installBrowserStubs();
    const ExperimentParams = require('../ExperimentParams');
    // VALIDATE_MUT_PROB lets the parent re-run this child with the inter-generation
    // rate set to the intra-generation one (0.05), which is the only way to compare
    // the two kernels' ALGORITHMS: at their production rates they must differ, and
    // a digest mismatch would say nothing.
    ExperimentParams.applyOverrides({
        hidden_size: 32,
        mut_prob: parseFloat(process.env.VALIDATE_MUT_PROB || '0.03'),
        mut_sigma: 0.1,
    });

    const GAManager      = require('../Organism/GAManager');
    const PureRLManager  = require('../Organism/PureRLManager');
    const AdvancedOrganism = require('../Organism/AdvancedOrganism');
    const NNBrain        = require('../Organism/Perception/NNBrain');

    const N = NNBrain.GENOME_SIZE;
    const base = new Float32Array(N);
    let s0 = 99991;
    const seedRnd = () => { s0 = (s0 * 1103515245 + 12345) & 0x7fffffff; return s0 / 0x7fffffff; };
    for (let i = 0; i < N; i++) base[i] = (seedRnd() * 2 - 1) * 0.5;

    const realRandom = Math.random;
    const pin = (seed) => {
        let s = seed >>> 0;
        Math.random = () => {
            s = (s * 1664525 + 1013904223) >>> 0;
            return s / 4294967296;
        };
    };
    const digest = (f32) =>
        crypto.createHash('sha256').update(Buffer.from(f32.buffer, f32.byteOffset, f32.byteLength))
              .digest('hex');

    const runKernel = (fn) => {
        const g = new Float32Array(base);
        pin(4242);
        fn(g);
        Math.random = realRandom;
        return digest(g);
    };

    // Inter-generation kernel, once per condition. GAManager is constructed
    // with rl_enabled true and false to prove no genetic operator branches on it.
    const gaLearning  = Object.create(GAManager.prototype);
    const gaEvolution = Object.create(GAManager.prototype);
    const purerl      = Object.create(PureRLManager.prototype);

    const inter = {
        'GAManager._mutate (learning, rl_enabled=true)':  runKernel(g => gaLearning._mutate(g)),
        'GAManager._mutate (evolution, rl_enabled=false)': runKernel(g => gaEvolution._mutate(g)),
        'PureRLManager._mutateWeights (pure_rl)':          runKernel(g => purerl._mutateWeights(g)),
    };

    // Intra-generation kernel: one implementation, shared by every condition,
    // reached through AdvancedOrganism.reproduce(). Run it off the prototype so
    // no organism has to be constructed.
    const org = Object.create(AdvancedOrganism.prototype);
    const intra = {
        'AdvancedOrganism._mutateGenome': runKernel(g => org._mutateGenome(g)),
    };

    // Function-reference identity: the same object on the same prototype for
    // every condition, so there is no per-condition dispatch at all.
    const same_fn_object =
        gaLearning._mutate === gaEvolution._mutate &&
        gaLearning._mutate === GAManager.prototype._mutate;

    // Source-level: no genetic operator may mention rl_enabled / condition.
    const srcOf = fn => fn.toString();
    const operators = {
        '_mutate': GAManager.prototype._mutate,
        '_uniformCrossover': GAManager.prototype._uniformCrossover,
        '_tournamentSelect': GAManager.prototype._tournamentSelect,
        '_gaussianSample': GAManager.prototype._gaussianSample,
        '_mutateWeights': PureRLManager.prototype._mutateWeights,
        '_mutateGenome': AdvancedOrganism.prototype._mutateGenome,
    };
    const branching = [];
    for (const [name, fn] of Object.entries(operators)) {
        const src = srcOf(fn);
        if (/rl_enabled|condition_label|learning_enabled|experiment_mode/.test(src)) branching.push(name);
    }

    emit({
        inter, intra, same_fn_object, condition_branching: branching,
        mut_prob: ExperimentParams.mut_prob, mut_sigma: ExperimentParams.mut_sigma,
        genome_size: N,
    });
}

function checkMut() {
    section('mut — mutation kernels are identical across the three conditions');
    const r = runIsolated('mut');
    const interDigests = Object.values(r.inter);
    const ok = interDigests.every(d => d === interDigests[0]) &&
               r.same_fn_object === true &&
               r.condition_branching.length === 0;
    const facts = {
        'inter-generation rate / sigma': `${r.mut_prob} / ${r.mut_sigma}`,
        'genome length': r.genome_size,
        'same function object across conditions': r.same_fn_object,
        'genetic operators mentioning the condition': r.condition_branching.length
            ? r.condition_branching.join(', ') : 'none',
    };
    for (const [k, v] of Object.entries(r.inter)) facts[k] = v.slice(0, 16) + '…';
    for (const [k, v] of Object.entries(r.intra)) facts[k] = v.slice(0, 16) + '…';
    record('mutation_kernel_identity', ok,
        ok ? 'all three inter-generation kernels produce a bit-identical genome from one ' +
             'pinned PRNG stream; no genetic operator reads the condition'
           : 'mutation kernels DIFFER between conditions', facts);

    // Second pass with the inter-generation rate dialled to the intra-generation
    // one. The two kernels are separately written (ExperimentParams-driven in
    // GAManager, hard-coded 0.05/0.1 in AdvancedOrganism), so this is what turns
    // "the same two kernels" into a measured statement rather than a reading of
    // the source: at a matched rate they must be bit-identical, which means same
    // Box-Muller ordering, same clip, same PRNG consumption.
    const m = runIsolated('mut', { VALIDATE_MUT_PROB: '0.05' });
    const interAt05 = m.inter['GAManager._mutate (evolution, rl_enabled=false)'];
    const intraAt05 = m.intra['AdvancedOrganism._mutateGenome'];
    record('mutation_kernel_algorithm_match', interAt05 === intraAt05,
        interAt05 === intraAt05
            ? 'inter- and intra-generation kernels are the same algorithm: bit-identical output ' +
              'once the rate is matched at 0.05 / 0.1'
            : 'the two kernels are NOT the same algorithm at a matched rate',
        {
            'rate / sigma used for this comparison': `${m.mut_prob} / ${m.mut_sigma}`,
            'GAManager._mutate @ 0.05': interAt05.slice(0, 16) + '…',
            'AdvancedOrganism._mutateGenome @ 0.05': intraAt05.slice(0, 16) + '…',
            'production rates': 'inter-generation 0.03 / 0.1, intra-generation 0.05 / 0.1',
        });
}

// ─────────────────────────────────────────────────────────────────────────────
// CHECK: cave — energy delta is exactly zero for cave-at-night ticks
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Two-part check. The direct part parks an organism on a cave cell, forces the
 * clock to night, and steps it through decay events, asserting the energy never
 * moves. The live part runs a real generation and reads the branch counters off
 * every organism: night-in-cave decay events must have happened (otherwise the
 * check is vacuous) and must have cost exactly 0.0 in total.
 */
function childCave() {
    const GC = require('../Organism/GenerationConstants');
    // Several generations, stopping one tick short of the last boundary. One
    // generation is not enough at 500x500: untrained founders spawn near the
    // centre and the cave band sits at 0.50-0.86 of the map radius, so whether
    // anyone is standing in a cave on a decay tick is close to a coin flip.
    // Later generations navigate, and the check stops depending on luck.
    const gens = Math.max(3, GENERATIONS);
    const { env } = bootAndRun('evolution', 'standard', gens,
        { stopAtTick: gens * GC.TICKS_PER_GEN - 1 });
    const CellStates = require('../Organism/Cell/CellStates');

    // ── direct: a controlled organism on a cave cell at night ────────────────
    // Find a cave cell in the world the run just generated.
    let cave = null;
    for (let r = 0; r < env.grid_map.rows && !cave; r++) {
        for (let c = 0; c < env.grid_map.cols; c++) {
            const cell = env.grid_map.cellAt(c, r);
            if (cell && cell.state === CellStates.cave) { cave = { c, r }; break; }
        }
    }
    const direct = { cave_found: !!cave };
    if (cave) {
        const AdvancedOrganism = require('../Organism/AdvancedOrganism');
        const org = new AdvancedOrganism(cave.c, cave.r, env, null, false, env.ga_manager);
        org.anatomy.addDefaultCell(CellStates.mouth, 0, 0);
        org.anatomy.checkTypeChange();
        org.c = cave.c; org.r = cave.r;

        // Pin the clock to night and stub out everything that could move energy
        // for a reason other than decay, so the decay branch is measured alone.
        const wasNight = env.isNight;
        env.isNight = () => true;
        org._isInCave = () => true;

        const before = org.energy;
        const N_EVENTS = 40;
        // Step only the decay arithmetic: drive lifetime to each decay boundary
        // and run the same branch update() runs.
        for (let e = 1; e <= N_EVENTS; e++) {
            org.lifetime = e * 10 - 1;
            org.update();
        }
        direct.energy_before = before;
        direct.energy_after  = org.energy;
        direct.delta         = org.energy - before;
        direct.decay_events_night_cave = org.audit_decay_events_night_cave;
        direct.decay_energy_night_cave = org.audit_decay_energy_night_cave;
        direct.food_gained   = org.audit_energy_from_food;
        env.isNight = wasNight;
    }

    // ── live: every organism in a real generation ────────────────────────────
    let orgs = 0, events = 0, energy = 0, worst = 0, with_events = 0;
    let day_cave = 0, night_open = 0, day_open = 0;
    // The edge-triggered cave counters the experiment already logs, summed over
    // the same population. They are the diagnostic for a vacuous live check:
    // the audit counters only fire on the 1-in-10 ticks a decay event lands on,
    // so a population that visits caves in short bursts can show many entries and
    // still no cave decay event.
    let entries = 0, entries_night = 0, entries_day = 0;
    for (const a of env.ga_manager.all_agents) {
        if (a.audit_decay_events_night_cave == null) continue;
        orgs++;
        events += a.audit_decay_events_night_cave;
        energy += a.audit_decay_energy_night_cave;
        worst = Math.max(worst, Math.abs(a.audit_decay_energy_night_cave));
        if (a.audit_decay_events_night_cave > 0) with_events++;
        day_cave   += a.audit_decay_events_day_cave;
        night_open += a.audit_decay_events_night_open;
        day_open   += a.audit_decay_events_day_open;
        entries       += a.cave_entry_count || 0;
        entries_night += a.cave_entries_night || 0;
        entries_day   += a.cave_entries_day || 0;
    }
    emit({ direct, live: { organisms: orgs, organisms_with_night_cave_events: with_events,
        night_cave_events: events, night_cave_energy: energy, worst_per_organism: worst,
        day_cave_events: day_cave, night_open_events: night_open, day_open_events: day_open,
        cave_entries: entries, cave_entries_night: entries_night,
        cave_entries_day: entries_day } });
}

function checkCave() {
    section('cave — a decay event inside a cave at night costs exactly 0 energy');
    const r = runIsolated('cave');
    const d = r.direct, l = r.live;
    const ok_direct = d.cave_found && d.delta === 0 && d.decay_events_night_cave >= 40 &&
                      d.decay_energy_night_cave === 0;
    record('cave_night_energy_direct', ok_direct,
        d.cave_found
            ? `organism parked in a cave at night: ${d.decay_events_night_cave} decay events, ` +
              `energy delta exactly ${d.delta}`
            : 'no cave cell found in the generated world (check is vacuous)',
        {
            'energy before': d.energy_before,
            'energy after': d.energy_after,
            'delta (must be 0)': d.delta,
            'decay events fired': d.decay_events_night_cave,
            'energy booked to those events': d.decay_energy_night_cave,
        });

    // Non-vacuity is part of the pass condition: a check that never observed the
    // branch it exists to test has not verified anything, so it reports FAIL with
    // the cave-entry counters attached rather than a pass on an untaken branch.
    const ok_live = l.organisms > 0 && l.night_cave_events > 0 && l.night_cave_energy === 0 &&
                    l.worst_per_organism === 0;
    record('cave_night_energy_live', ok_live,
        l.night_cave_events > 0
            ? `live run: ${l.night_cave_events} night-in-cave decay events across ` +
              `${l.organisms_with_night_cave_events} organisms, total energy cost ` +
              `${l.night_cave_energy}`
            : 'VACUOUS — no organism was inside a cave on a night decay tick, so the ' +
              'branch was never exercised. Raise --generations (untrained founders ' +
              'spawn near the centre and rarely reach the cave band).',
        {
            'organisms': l.organisms,
            'organisms that were in a cave at night': l.organisms_with_night_cave_events,
            'night-in-cave decay events': l.night_cave_events,
            'energy lost to them (must be 0)': l.night_cave_energy,
            'other branches (day-cave / night-open / day-open)':
                `${l.day_cave_events} / ${l.night_open_events} / ${l.day_open_events}`,
            'cave entries logged (edge-triggered: total / night / day)':
                `${l.cave_entries} / ${l.cave_entries_night} / ${l.cave_entries_day}`,
            'note': 'decay fires every 10th tick, so brief cave visits can raise the ' +
                    'entry counters without ever landing on a decay tick',
        });

    // The face-validity evidence that the mechanic matters in the real
    // experiment rather than only in this harness.
    if (ORG_SCAN > 0) {
        const prod = scanProductionCaveUse(ORG_SCAN);
        record('cave_use_in_production', prod.runs > 0 && prod.entries_night > 0,
            prod.runs > 0
                ? `${prod.runs} production runs sampled: ${prod.entries_night} night cave ` +
                  `entries by ${prod.organisms_with_night} organisms`
                : 'no production organisms.csv found to sample',
            {
                'runs sampled': prod.runs,
                'organism rows read': prod.rows,
                'cave entries (night)': prod.entries_night,
                'cave entries (day)': prod.entries_day,
                'organisms with >=1 night cave entry': prod.organisms_with_night,
                'runs scanned': prod.names,
            });
    }
}

/**
 * Stream a bounded sample of production organisms.csv files and total the
 * edge-triggered cave counters. organisms.csv is ~100 MB per run and the tree is
 * tens of GB, so this reads at most `limit` files and only the columns it needs.
 */
function scanProductionCaveUse(limit) {
    const candidates = [];
    const walk = (dir) => {
        let entries;
        try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch (_) { return; }
        for (const e of entries) {
            const p = path.join(dir, e.name);
            if (e.isDirectory()) { walk(p); continue; }
            if (e.name === 'organisms.csv') candidates.push(p);
        }
    };
    for (const root of LOGS_ROOTS) walk(path.join(REPO, root));
    candidates.sort();

    let runs = 0, rows = 0, night = 0, day = 0, withNight = 0;
    const names = [];
    for (const file of candidates.slice(0, limit)) {
        const text = fs.readFileSync(file, 'utf8');
        const nl = text.indexOf('\n');
        if (nl < 0) continue;
        const header = text.slice(0, nl).trim().split(',');
        const iN = header.indexOf('cave_entries_night');
        const iD = header.indexOf('cave_entries_day');
        if (iN < 0 || iD < 0) continue;
        runs++;
        names.push(path.basename(path.dirname(file)));
        let pos = nl + 1;
        while (pos < text.length) {
            let end = text.indexOf('\n', pos);
            if (end < 0) end = text.length;
            const line = text.slice(pos, end);
            pos = end + 1;
            if (!line) continue;
            const cols = line.split(',');
            const n = parseInt(cols[iN], 10);
            const d = parseInt(cols[iD], 10);
            if (!isFinite(n) && !isFinite(d)) continue;
            rows++;
            if (isFinite(n)) { night += n; if (n > 0) withNight++; }
            if (isFinite(d)) day += d;
        }
    }
    return { runs, rows, entries_night: night, entries_day: day,
             organisms_with_night: withNight, names };
}

// ─────────────────────────────────────────────────────────────────────────────
// CHECK: drain — predation energy accounting
// ─────────────────────────────────────────────────────────────────────────────

function childDrain() {
    const drain = parseFloat(process.env.VALIDATE_DRAIN || '5');
    const GC = require('../Organism/GenerationConstants');
    const { env } = bootAndRun('evolution', 'standard', 1, {
        predators: { drain, roaming: 60, patrol: 0 },
        stopAtTick: GC.TICKS_PER_GEN - 1,
    });
    let orgs = 0, drained = 0, contacts = 0, ticks = 0, energy = 0, worst = 0;
    let contacts_gt_ticks = 0;
    for (const a of env.ga_manager.all_agents) {
        if (a.audit_energy_lost_predation == null) continue;
        orgs++;
        if (a.audit_drain_contacts > 0) drained++;
        contacts += a.audit_drain_contacts;
        ticks    += a.drained_ticks;
        energy   += a.audit_energy_lost_predation;
        worst = Math.max(worst, Math.abs(a.audit_energy_lost_predation - a.audit_drain_contacts * drain));
        if (a.audit_drain_contacts > a.drained_ticks) contacts_gt_ticks++;
    }
    emit({ drain, organisms: orgs, organisms_drained: drained, contacts, drained_ticks: ticks,
           energy_lost: energy, expected: contacts * drain, worst_per_organism: worst,
           organisms_with_multi_drain_ticks: contacts_gt_ticks });
}

function checkDrain() {
    section('drain — energy lost to predation equals contacts x drainAmount');
    const r = runIsolated('drain', { VALIDATE_DRAIN: '5' });
    const ok = r.organisms_drained > 0 &&
               r.worst_per_organism < 1e-9 &&
               Math.abs(r.energy_lost - r.expected) < 1e-6;
    record('predation_energy_accounting', ok,
        `${r.organisms_drained}/${r.organisms} organisms were drained: ${r.energy_lost} energy ` +
        `lost against ${r.contacts} contacts x ${r.drain} = ${r.expected}`,
        {
            'drainAmount': r.drain,
            'drain contacts': r.contacts,
            'energy lost to predation': r.energy_lost,
            'contacts x drainAmount': r.expected,
            'worst per-organism discrepancy': r.worst_per_organism,
            'drained_ticks (the logged, de-duplicated metric)': r.drained_ticks,
            'organisms hit more than once in a tick': r.organisms_with_multi_drain_ticks,
            'note': r.contacts === r.drained_ticks
                ? 'contacts == drained_ticks here, so drained_ticks x drainAmount is also exact'
                : `contacts exceed drained_ticks by ${r.contacts - r.drained_ticks}: drained_ticks ` +
                  'de-duplicates simultaneous drains, so drained_ticks x drainAmount is a LOWER ' +
                  'BOUND on energy lost, not an equality',
        });
}

// ─────────────────────────────────────────────────────────────────────────────
// CHECK: ledger — energy conservation
// ─────────────────────────────────────────────────────────────────────────────

function childLedger() {
    const GC = require('../Organism/GenerationConstants');
    // Sampled DURING the run, not at the end. In the hard environment a small
    // test world can lose its whole population before the generation closes, and
    // die() is free to leave a dead organism's energy at or below zero — an
    // endpoint-only check would then either read 0 organisms (vacuous) or compare
    // against a post-mortem value. Walking the living population every
    // SAMPLE_EVERY ticks checks the identity thousands of times across the run,
    // while every term is still live.
    const SAMPLE_EVERY = 100;
    let samples = 0, orgs = 0, worst = 0, worstRel = 0, failures = 0;
    let food_seen = 0, decay_seen = 0, pred_seen = 0;
    const examples = [];

    const sample = (env) => {
        if (env.total_ticks % SAMPLE_EVERY !== 0) return;
        samples++;
        for (const a of env.ga_manager.living_agents) {
            if (a.audit_start_energy == null || !a.living) continue;
            orgs++;
            const expected = a.audit_start_energy + a.audit_energy_from_food -
                             a.audit_energy_lost_decay - a.audit_energy_lost_predation;
            const err = Math.abs(expected - a.energy);
            const rel = err / Math.max(1, Math.abs(a.energy));
            if (err > worst) worst = err;
            if (rel > worstRel) worstRel = rel;
            if (rel > 1e-9) {
                failures++;
                if (examples.length < 3) examples.push({
                    tick: env.total_ticks, energy: a.energy, expected,
                    food: a.audit_energy_from_food, decay: a.audit_energy_lost_decay,
                    predation: a.audit_energy_lost_predation });
            }
            if (a.audit_energy_from_food > 0)       food_seen++;
            if (a.audit_energy_lost_decay > 0)      decay_seen++;
            if (a.audit_energy_lost_predation > 0)  pred_seen++;
        }
    };

    // Learning condition with predators on, so all three terms of the identity —
    // food, decay and predation — are actually exercised.
    bootAndRun('learning', 'standard', 1, {
        predators: { drain: 5, roaming: 20, patrol: 0 },
        stopAtTick: GC.TICKS_PER_GEN - 1,
        onTick: sample,
    });

    emit({ samples, organism_checks: orgs, worst_abs_err: worst, worst_rel_err: worstRel,
           failures, examples,
           checks_with_food: food_seen, checks_with_decay: decay_seen,
           checks_with_predation: pred_seen });
}

function checkLedger() {
    section('ledger — start + food - decay - predation == energy, for every organism');
    const r = runIsolated('ledger');
    // Not bit-exact: the sim accumulates the four terms interleaved into one
    // float, the ledger sums them in four separate ones, and float addition is
    // not associative. A relative error at the 1e-9 level is that reordering; an
    // unaccounted energy path would show up far above it.
    //
    // The check is only meaningful if all three terms were exercised, so a
    // sample set with no food, no decay or no predation in it fails rather than
    // passing on an untested identity.
    const ok = r.organism_checks > 0 && r.failures === 0 &&
               r.checks_with_food > 0 && r.checks_with_decay > 0 && r.checks_with_predation > 0;
    record('energy_ledger_closes', ok,
        `${r.organism_checks} organism-checks over ${r.samples} samples of a live generation: ` +
        `worst relative error ${r.worst_rel_err.toExponential(2)} (float re-association only — ` +
        'no unaccounted energy path)',
        {
            'samples taken during the run': r.samples,
            'organism-checks': r.organism_checks,
            'worst absolute error': r.worst_abs_err.toExponential(3),
            'worst relative error': r.worst_rel_err.toExponential(3),
            'checks above the 1e-9 relative tolerance': r.failures,
            'checks where food had been eaten': r.checks_with_food,
            'checks where decay had fired': r.checks_with_decay,
            'checks where a predator had drained': r.checks_with_predation,
            'failing examples': r.examples,
        });
}

// ─────────────────────────────────────────────────────────────────────────────
// CHECK: floor — the random-policy anchor really is random
// ─────────────────────────────────────────────────────────────────────────────

/**
 * The floor is only an anchor if it genuinely carries nothing across a
 * generation boundary. Two things are checked: the gene pool is null at every
 * boundary (so spawnGeneration falls through to Xavier init), and successive
 * generations' founder genomes are uncorrelated — a leak would show up as a
 * correlation near 1 between the generation-1 and generation-2 centroids.
 */
function childFloor() {
    // At least 4 generations so there are at least 2 consecutive-centroid
    // correlations to look at; one comparison is too thin to call a leak.
    const gens = Math.max(4, GENERATIONS);
    const centroids = [];
    const poolStates = [];
    const { env, gen_log } = bootAndRun('evolution', 'standard', gens, {
        params: { random_floor: true },
        onTick: (e) => {
            const ga = e.ga_manager;
            if (ga.tick_count === 0 && ga.founders.length && centroids.length < ga.generation) {
                const N = ga.founders[0].brain.genome_weights.length;
                const c = new Float64Array(N);
                for (const f of ga.founders) {
                    for (let i = 0; i < N; i++) c[i] += f.brain.genome_weights[i];
                }
                for (let i = 0; i < N; i++) c[i] /= ga.founders.length;
                centroids.push(Array.from(c.slice(0, 2000)));
                poolStates.push(ga.gene_pool === null);
            }
        },
    });

    const corr = (a, b) => {
        const n = Math.min(a.length, b.length);
        let ma = 0, mb = 0;
        for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
        ma /= n; mb /= n;
        let num = 0, da = 0, db = 0;
        for (let i = 0; i < n; i++) {
            const x = a[i] - ma, y = b[i] - mb;
            num += x * y; da += x * x; db += y * y;
        }
        return (da > 0 && db > 0) ? num / Math.sqrt(da * db) : 0;
    };
    const corrs = [];
    for (let i = 1; i < centroids.length; i++) corrs.push(corr(centroids[i - 1], centroids[i]));

    emit({
        condition_label: env.ga_manager.condition_label,
        random_floor: env.ga_manager.random_floor === true,
        generations_observed: centroids.length,
        gene_pool_null_at_spawn: poolStates,
        consecutive_centroid_correlations: corrs,
        fitness_by_generation: gen_log.map(e => parseFloat(e.avg_fitness)),
        genome_variance_by_generation: gen_log.map(e => parseFloat(e.genome_variance)),
    });
}

function checkFloor() {
    section('floor — --random-floor carries nothing across a generation boundary');
    const r = runIsolated('floor');
    const maxCorr = r.consecutive_centroid_correlations.length
        ? Math.max.apply(null, r.consecutive_centroid_correlations.map(Math.abs)) : 1;
    const ok = r.random_floor === true &&
               r.condition_label === 'random_floor' &&
               r.gene_pool_null_at_spawn.every(Boolean) &&
               r.generations_observed >= 3 &&
               r.consecutive_centroid_correlations.length >= 2 &&
               maxCorr < 0.1;
    record('random_floor_is_random', ok,
        `${r.generations_observed} generations of freshly Xavier-random founders; strongest ` +
        `consecutive-centroid correlation |r| = ${maxCorr.toExponential(2)}`,
        {
            'ga.random_floor': r.random_floor,
            'condition label written to CSV': r.condition_label,
            'gene pool null at every spawn': r.gene_pool_null_at_spawn,
            'consecutive centroid correlations': r.consecutive_centroid_correlations
                .map(v => v.toFixed(4)).join(', '),
            'avg_fitness by generation': r.fitness_by_generation.join(', '),
            'genome_variance by generation': r.genome_variance_by_generation.join(', '),
        });
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

const CHILDREN = {
    mad: childMad, maps: childMaps, grad: childGrad, mut: childMut,
    cave: childCave, drain: childDrain, ledger: childLedger, floor: childFloor,
};

if (opts.child) {
    const fn = CHILDREN[opts.child];
    if (!fn) { console.error(`no child "${opts.child}"`); process.exit(2); }
    fn();
} else {
    const PARENTS = {
        mad: checkMad, maps: checkMaps, grad: checkGrad, mut: checkMut,
        cave: checkCave, drain: checkDrain, ledger: checkLedger, floor: checkFloor,
    };
    console.log('LifeEngine verification suite');
    console.log(`  seed=${SEED}  grid=${WIDTH}x${HEIGHT}  generations=${GENERATIONS}  ` +
                `hidden=${HIDDEN}  node=${process.version}`);
    console.log(`  checks: ${SELECTED.join(', ')}`);

    const started = Date.now();
    for (const name of SELECTED) {
        try {
            PARENTS[name]();
        } catch (e) {
            record(name, false, `check threw: ${e.message}`, { stack: String(e.stack).slice(0, 1200) });
        }
    }

    const passed = results.filter(r => r.passed).length;
    section('summary');
    for (const r of results) {
        console.log(`  ${r.passed ? '✓' : '✗'} ${r.check}`);
    }
    console.log(`\n${passed}/${results.length} checks passed in ` +
                `${((Date.now() - started) / 1000).toFixed(1)}s`);

    const outDir = path.join(REPO, 'output', 'validation');
    fs.mkdirSync(outDir, { recursive: true });
    const outFile = path.join(outDir, 'verification_report.json');
    fs.writeFileSync(outFile, JSON.stringify({
        generated_at: new Date().toISOString(),
        node_version: process.version,
        config: { seed: SEED, width: WIDTH, height: HEIGHT, generations: GENERATIONS,
                  hidden_size: HIDDEN, logs_roots: LOGS_ROOTS },
        passed, total: results.length, results,
    }, null, 2));
    console.log(`report written to ${outFile.replace(REPO + path.sep, '')}`);

    process.exit(passed === results.length ? 0 : 1);
}
