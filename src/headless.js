'use strict';

/**
 * src/headless.js — headless runner for the learning-vs-evolution experiment.
 *
 * Runs the SAME experiment the browser runs (GAManager / FrozenPolicyManager /
 * PureRLManager + map pool + predators + structured Logger), with no browser,
 * canvas, or display — for the CHPC (PBS) cluster. See "CHPC Guide.pdf".
 *
 * It boots exactly like Engine.js (construct env -> generateWorld ->
 * spawnGeneration -> predators.spawnAll), then steps env.update() until the
 * requested number of generations (or ticks) is reached. The Logger auto-saves
 * generations.csv / organisms.csv / events.csv to
 *   logs/<condition>/<mode>/auto-run/run_N/
 * and this runner drops a params.txt + params.json of the exact run config into
 * that same folder.
 *
 * The map terrain is seeded (--map-seed); GA mutation, RL exploration and
 * predator wandering are deliberately left unseeded (run-to-run chance).
 *
 *   node src/headless.js --map-seed 42 --condition learning --generations 50
 *
 * Run `node src/headless.js` with no args for the full flag reference.
 */

// ─── Mock browser globals ─────────────────────────────────────────────────────
// Stub DOM / jQuery so canvas-based modules load without errors. Drawing calls
// become no-ops; rendering is skipped via WorldConfig.headless.
const mockCtx = new Proxy({}, {
    get: (_, prop) => typeof prop === 'string' ? () => {} : undefined,
    set: () => true,
});
const mockEl = {
    getContext: () => mockCtx,
    addEventListener: () => {},
    removeEventListener: () => {},
    onwheel: null,
    width: 800,
    height: 600,
};
global.document = {
    getElementById: () => mockEl,
    querySelector:  () => mockEl,
    createElement:  () => mockEl,
};
global.window = global;
try { global.navigator = { userAgent: '' }; } catch (_) {}
global.confirm = () => false;
const jqChain = new Proxy({}, {
    get(_, prop) {
        const primitives = { height: 600, width: 800, length: 1 };
        if (prop in primitives) return () => primitives[prop];
        if (prop === 'is') return () => false;
        if (prop === '0')  return { click: () => {} };
        return function () { return jqChain; };
    }
});
global.$ = new Proxy(function () { return jqChain; }, {
    get(_, prop) {
        if (prop === 'fn') return {};
        return jqChain[prop] || function () { return jqChain; };
    }
});

// ─── CLI parsing ──────────────────────────────────────────────────────────────

function parseArgs(argv) {
    const opts = {};
    for (let i = 0; i < argv.length; i++) {
        if (argv[i].startsWith('--')) {
            const key = argv[i].slice(2);
            const next = argv[i + 1];
            if (next !== undefined && !next.startsWith('--')) { opts[key] = next; i++; }
            else opts[key] = true;
        }
    }
    return opts;
}

const opts = parseArgs(process.argv.slice(2));

function intOpt(name, def = null) {
    if (opts[name] === undefined) return def;
    const v = parseInt(opts[name], 10);
    if (isNaN(v)) { console.error(`ERROR: --${name} must be an integer.`); process.exit(1); }
    return v;
}
function floatOpt(name, def = null) {
    if (opts[name] === undefined) return def;
    const v = parseFloat(opts[name]);
    if (isNaN(v)) { console.error(`ERROR: --${name} must be a number.`); process.exit(1); }
    return v;
}
function strOpt(name, def = null, allowed = null) {
    if (opts[name] === undefined) return def;
    const v = String(opts[name]);
    if (allowed && !allowed.includes(v)) {
        console.error(`ERROR: --${name} must be one of: ${allowed.join(', ')}.`); process.exit(1);
    }
    return v;
}

function usageAndExit() {
    console.error(
        'Usage: node src/headless.js --map-seed <N> (--generations <N> | --max-ticks <N>)\n' +
        '\n  Run control:\n' +
        '    --map-seed   <N>   REQUIRED. Terrain seed (deterministic world).\n' +
        '    --generations<N>   Run this many generations (1 generation = ticks-per-map x maps-per-gen ticks).\n' +
        '    --max-ticks  <N>   Alternative stop condition in raw ticks (use one of generations/max-ticks).\n' +
        '    --condition  <s>   learning | evolution            (default learning)\n' +
        '    --mode       <s>   standard | frozen_pg | pure_rl  (default standard)\n' +
        '    --width      <N>   grid columns (default 400)\n' +
        '    --height     <N>   grid rows    (default 300)\n' +
        '    --cell-size  <N>   pixels per cell (default 2)\n' +
        '    --ticks-per-map <N>  override generation map length (default 2000)\n' +
        '    --maps-per-gen  <N>  override maps per generation   (default 5)\n' +
        '    --log-every  <N>   progress line every N ticks (default 10000, 0 to silence)\n' +
        '    --run-name   <s>   output folder name under logs/<cond>/<mode>/auto-run/\n' +
        '                       (default: seed<N>_job<PBS_JOBID> on the cluster, else run_N)\n' +
        '    --log-dir    <s>   base output dir (default: ./logs). Pass an absolute\n' +
        '                       path to write to scratch instead of the project tree.\n' +
        '    --log-genomes      also write genome.csv: per-gen centroid + fittest\n' +
        '                       founder genomes (all founders every 250 gens), full\n' +
        '                       float32 precision. Off by default (tens of MB/run —\n' +
        '                       enable only for final runs, not tuning sweeps).\n' +
        '\n  Tunable hyperparameters (default = current in-code value):\n' +
        '    --learning-rate <f>       RL learning rate              (0.02)\n' +
        '    --epsilon-start <f>       exploration epsilon at birth  (0.5)\n' +
        '    --epsilon-end   <f>       exploration epsilon at death  (0.05)\n' +
        '    --epsilon-decay-shape <s> sublinear|linear|quadratic    (quadratic)\n' +
        '    --no-epsilon              disable epsilon entirely (pure on-policy REINFORCE)\n' +
        '    --trace-decay   <f>       eligibility-trace decay gamma  (0.90)\n' +
        '    --explore-bonus <f>       reward per newly visited cell   (0.15)\n' +
        '    --hidden-size   <N>       NN hidden-layer width         (64)\n' +
        '    --population-size <N>     founders per generation       (100)\n' +
        '    --mut-prob      <f>       between-gen mutation rate      (0.03)\n' +
        '    --mut-sigma     <f>       between-gen mutation std-dev   (0.1)\n' +
        '    --disaster                enable natural-disaster culls  (off)\n' +
        '    --disaster-prob <f>       per-gen chance a disaster hits (0.1)\n' +
        '    --disaster-fraction <f>   fixed fraction culled per event (0.2)\n' +
        '    --disaster-cooldown <N>   min generations between disasters (0=none)\n' +
        '    --disaster-seed <N>       disaster PRNG seed, 0=unseeded  (0)\n' +
        '    --disaster-recovery-rate <f>  taper cull by this fraction/gen (0=once-off)\n' +
        '    --predator-drain <f>      energy drained per contact     (1.0)\n' +
        '    --predators-per-patch <N> patrol predators per prestige patch (2)\n' +
        '    --roaming-predators   <N> roaming predator count         (0)\n' +
        '    --food-shuffle-period <N> ticks between food-tier payoff reshuffles;\n' +
        '                              0=off. Non-stationary env forcing in-life learning (0)\n' +
        '    --food-shuffle-seed   <N> reshuffle-schedule PRNG seed, 0=unseeded (0)\n'
    );
    process.exit(1);
}

const MAP_SEED    = intOpt('map-seed');
const GENERATIONS = intOpt('generations');
const MAX_TICKS   = intOpt('max-ticks');
if (MAP_SEED == null || (GENERATIONS == null && MAX_TICKS == null)) usageAndExit();

const CONDITION = (opts.condition || 'learning');
const MODE      = (opts.mode || 'standard');
if (!['learning', 'evolution'].includes(CONDITION)) { console.error(`ERROR: --condition must be learning|evolution.`); process.exit(1); }
if (!['standard', 'frozen_pg', 'pure_rl'].includes(MODE)) { console.error(`ERROR: --mode must be standard|frozen_pg|pure_rl.`); process.exit(1); }

const WIDTH     = intOpt('width', 400);
const HEIGHT    = intOpt('height', 300);
const CELL_SIZE = intOpt('cell-size', 2);
const LOG_EVERY = intOpt('log-every', 10000);
// Founder-genome logging (genome.csv) is opt-in: off for tuning sweeps, on for
// the final runs that feed the fitness-landscape analysis. See Logger.setGenomeLogging.
const LOG_GENOMES = opts['log-genomes'] ? true : false;

// ─── Apply tunable params BEFORE any sim module loads ─────────────────────────
// NNBrain and GAManager read these at module load, so the override must happen
// before they are required (transitively, via WorldEnvironment below).
const ExperimentParams = require('./ExperimentParams');
const changed = ExperimentParams.applyOverrides({
    learning_rate:          floatOpt('learning-rate'),
    epsilon_start:          floatOpt('epsilon-start'),
    epsilon_end:            floatOpt('epsilon-end'),
    epsilon_decay_shape:    strOpt('epsilon-decay-shape', null, ['sublinear', 'linear', 'quadratic']),
    epsilon_enabled:        opts['no-epsilon'] ? false : null,
    trace_decay:            floatOpt('trace-decay'),
    explore_bonus:          floatOpt('explore-bonus'),
    hidden_size:            intOpt('hidden-size'),
    population_size:        intOpt('population-size'),
    mut_prob:               floatOpt('mut-prob'),
    mut_sigma:              floatOpt('mut-sigma'),
    disaster_enabled:       opts.disaster ? true : null,
    disaster_prob:          floatOpt('disaster-prob'),
    disaster_fraction:      floatOpt('disaster-fraction'),
    disaster_cooldown:      intOpt('disaster-cooldown'),
    disaster_seed:          intOpt('disaster-seed'),
    disaster_recovery_rate: floatOpt('disaster-recovery-rate'),
    predator_drain:         floatOpt('predator-drain'),
    predators_per_patch:    intOpt('predators-per-patch'),
    roaming_predator_count: intOpt('roaming-predators'),
    food_shuffle_period:    intOpt('food-shuffle-period'),
    food_shuffle_seed:      intOpt('food-shuffle-seed'),
});

const WorldConfig = require('./WorldConfig');
WorldConfig.headless        = true;
WorldConfig.auto_pause      = false;
WorldConfig.auto_reset      = false;
WorldConfig.MAP_SEED        = MAP_SEED;
WorldConfig.learning_enabled = (CONDITION === 'learning');
WorldConfig.experiment_mode = MODE;
WorldConfig.MAP_COLS        = WIDTH;
WorldConfig.MAP_ROWS        = HEIGHT;
if (opts['ticks-per-map'] !== undefined) WorldConfig.TICKS_PER_MAP = intOpt('ticks-per-map');
if (opts['maps-per-gen']  !== undefined) WorldConfig.MAPS_PER_GEN  = intOpt('maps-per-gen');

// Predator knobs live on a mutable object read at runtime — set them directly.
const PredatorHyperparameters = require('./Organism/PredatorHyperparameters');
PredatorHyperparameters.drainAmount               = ExperimentParams.predator_drain;
PredatorHyperparameters.count                     = ExperimentParams.roaming_predator_count;
PredatorHyperparameters.patrol.predatorsPerPatch  = ExperimentParams.predators_per_patch;

// ─── Load sim modules (now sized/configured per the overrides) ────────────────
const WorldEnvironment    = require('./Environments/WorldEnvironment');
const GenerationConstants = require('./Organism/GenerationConstants');
const logger              = require('./Logger');
const fs                  = require('fs');
const path                = require('path');

// Apply the genome-logging choice for this run (default off; --log-genomes to enable).
logger.setGenomeLogging(LOG_GENOMES);

const TICKS_PER_GEN = GenerationConstants.TICKS_PER_GEN;
const MAX_TICKS_EFF = (MAX_TICKS != null) ? MAX_TICKS : GENERATIONS * TICKS_PER_GEN;

console.log(`[headless] condition=${CONDITION} mode=${MODE} map_seed=${MAP_SEED} ` +
    `grid=${WIDTH}x${HEIGHT} gen=${TICKS_PER_GEN}t target=${MAX_TICKS_EFF}t`);
if (changed.length) console.log(`[headless] overrides: ${changed.map(k => `${k}=${ExperimentParams[k]}`).join(', ')}`);

// ─── Engine shim (env only references .running indirectly) ────────────────────
const engine = { fps: Infinity, last_fps: Infinity, running: true,
    stop() { this.running = false; }, start() {}, restart() {} };

// ─── Boot environment (mirrors Engine constructor, minus rendering/UI) ────────
const env = new WorldEnvironment(engine, CELL_SIZE);
env.generateWorld();
env.ga_manager.spawnGeneration();
env.predator_manager.spawnAll();

// Resolve the run directory so the per-generation CSV auto-saves and our params
// files share one folder. Make the folder name collision-proof for parallel
// jobs: prefer --run-name, else derive from PBS_JOBID (unique per cluster job),
// else fall back to the Logger's auto-incremented run_N (fine for serial/local).
const rl_enabled = (env.ga_manager.rl_enabled !== undefined)
    ? env.ga_manager.rl_enabled
    : (WorldConfig.learning_enabled || MODE === 'frozen_pg' || MODE === 'pure_rl');
let runName = (typeof opts['run-name'] === 'string') ? opts['run-name'] : null;
if (!runName && process.env.PBS_JOBID) {
    const jobnum = String(process.env.PBS_JOBID).split('.')[0];  // strip .sched1.chpc...
    runName = `seed${MAP_SEED}_job${jobnum}`;
}
if (runName) logger.setRunName(runName);
// Redirect output off the home-quota'd project tree onto scratch when asked
// (absolute path recommended). Must precede the first getRunDir().
if (typeof opts['log-dir'] === 'string' && opts['log-dir']) logger.setAutoSaveDir(opts['log-dir']);
// getRunDir creates the .../auto-run dir; if the output/scratch base isn't
// creatable this throws. Catch it here so a bad path fails with a clear message
// instead of an uncaught stack trace (and long before wasting a whole run).
let runDir;
try {
    runDir = logger.getRunDir(rl_enabled);
} catch (e) {
    console.error(`[headless] FATAL: cannot create output directory: ${e.message}`);
    console.error('[headless] The output path (--log-dir / LOG_DIR) is likely wrong or not ' +
        'writable. Fix it and resubmit.');
    process.exit(1);
}

// ─── Write the run's parameter manifest ───────────────────────────────────────
const params = {
    map_seed:       MAP_SEED,
    condition:      CONDITION,
    mode:           MODE,
    generations:    GENERATIONS,
    max_ticks:      MAX_TICKS_EFF,
    ticks_per_gen:  TICKS_PER_GEN,
    ticks_per_map:  GenerationConstants.TICKS_PER_MAP,
    maps_per_gen:   GenerationConstants.MAPS_PER_GEN,
    grid_cols:      WIDTH,
    grid_rows:      HEIGHT,
    cell_size:      CELL_SIZE,
    log_genomes:    LOG_GENOMES,
    ...ExperimentParams.snapshot(),
    started_at:     new Date().toISOString(),
    node_version:   process.version,
    pbs_job_id:     process.env.PBS_JOBID || null,
};
if (runDir) {
    try {
        fs.mkdirSync(runDir, { recursive: true });
        const txt = Object.entries(params)
            .map(([k, v]) => `${k.padEnd(22)} = ${v}`)
            .join('\n') + '\n';
        fs.writeFileSync(path.join(runDir, 'params.txt'), txt, 'utf8');
        fs.writeFileSync(path.join(runDir, 'params.json'), JSON.stringify(params, null, 2), 'utf8');
        console.log(`[headless] Params written to ${path.join(runDir, 'params.txt')}`);
    } catch (e) {
        // The run dir is where the CSVs go too. If we can't write here, the whole
        // run would otherwise complete but silently produce nothing (Logger's
        // _flush swallows the same error and disables auto-save). Fail loudly and
        // immediately instead so a bad --log-dir / scratch path is obvious.
        console.error(`[headless] FATAL: cannot write to run directory:\n  ${runDir}\n  ${e.message}`);
        console.error('[headless] The output path (--log-dir / LOG_DIR) is likely wrong or not ' +
            'writable. Fix it and resubmit — no CSVs would be produced otherwise.');
        process.exit(1);
    }
} else {
    console.warn('[headless] No run directory (fs auto-save disabled) — params not written.');
}

// ─── Simulation loop ──────────────────────────────────────────────────────────
const wall_start = Date.now();
let last_log_wall = wall_start;
let last_log_tick = 0;

console.log(`[headless] Starting simulation...`);
while (env.total_ticks < MAX_TICKS_EFF && engine.running) {
    env.update();

    // Generation-based stop (standard / frozen GA). spawnGeneration sets
    // generation=1 before the loop; it becomes G+1 once generation G completes.
    if (GENERATIONS != null && env.ga_manager.generation != null &&
        env.ga_manager.generation > GENERATIONS) {
        break;
    }

    if (LOG_EVERY > 0 && env.total_ticks % LOG_EVERY === 0) {
        const now = Date.now();
        const elapsed_s = (now - wall_start) / 1000;
        const interval_s = (now - last_log_wall) / 1000;
        const inst = interval_s > 0 ? Math.round((env.total_ticks - last_log_tick) / interval_s) : 0;
        const pop = env.ga_manager.living_agents ? env.ga_manager.living_agents.size : env.organisms.length;
        const rss_mb = process.memoryUsage().rss / (1024 * 1024);
        console.log(
            `[headless] tick=${env.total_ticks}/${MAX_TICKS_EFF}` +
            `  gen=${env.ga_manager.generation}` +
            `  pop=${pop}` +
            `  inst=${inst} t/s  rss=${rss_mb.toFixed(0)}MB  elapsed=${elapsed_s.toFixed(1)}s`
        );
        last_log_wall = now;
        last_log_tick = env.total_ticks;
    }
}

// ─── Finalise ─────────────────────────────────────────────────────────────────
logger.flush(rl_enabled);  // persist the final sub-interval generations

const wall_elapsed_s = (Date.now() - wall_start) / 1000;
console.log(
    `[headless] Done — ${env.total_ticks} ticks, generation ${env.ga_manager.generation}, ` +
    `in ${wall_elapsed_s.toFixed(1)}s (avg ${Math.round(env.total_ticks / Math.max(wall_elapsed_s, 1e-9))} t/s).`
);
if (runDir) console.log(`[headless] CSV output + params in ${runDir}`);
