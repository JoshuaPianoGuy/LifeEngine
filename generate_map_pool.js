/**
 * generate_map_pool.js — CLI front-end for the seed-based map-pool generator.
 *
 * Generation logic lives in src/maps/generateMapPool.js (a render-free Node
 * module shared with the future headless/HPC runner). This script just handles
 * the disk cache + active-pool selection so the browser bundle picks up the
 * right maps.
 *
 * Usage:
 *   node generate_map_pool.js --seed 42        # generate (or reuse cached) seed 42
 *   node generate_map_pool.js --seed 42 --force  # regenerate even if cached
 *   node generate_map_pool.js                   # default seed (1)
 *   node generate_map_pool.js --seed 42 --out path/to/maps   # custom output dir
 *
 * Behaviour:
 *   - Writes/reuses src/maps/map_pool_seed<seed>.json (the deterministic cache).
 *     If that file already exists it is reused as-is (skips regeneration) unless
 *     --force is given — large maps generated once, reused thereafter.
 *   - Copies the chosen pool to src/maps/map_pool.json (the ACTIVE pool the
 *     webpack bundle loads via require). Only this one map is bundled, keeping
 *     the bundle light even for very large maps.
 *
 * One seed == one pool, shared by all conditions in that run. Set the matching
 * seed in src/WorldConfig.js (MAP_SEED) so the runtime can sanity-check it.
 */

'use strict';

const path = require('path');
const fs   = require('fs');
const { generateMapPool } = require('./src/maps/generateMapPool');

// ── Parse CLI args ────────────────────────────────────────────────────────────
function parseArgs(argv) {
    const args = { seed: 1, force: false, outDir: null };
    for (let i = 2; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--seed' || a === '-s') {
            args.seed = parseInt(argv[++i], 10);
        } else if (a.startsWith('--seed=')) {
            args.seed = parseInt(a.slice('--seed='.length), 10);
        } else if (a === '--force' || a === '-f') {
            args.force = true;
        } else if (a === '--out' || a === '-o') {
            args.outDir = argv[++i];
        } else if (a.startsWith('--out=')) {
            args.outDir = a.slice('--out='.length);
        } else if (!a.startsWith('-') && args.outDir === null) {
            // Positional fallback: treat a bare path as the output dir
            args.outDir = a;
        } else {
            console.error(`Unknown argument: ${a}`);
            process.exit(1);
        }
    }
    return args;
}

const args = parseArgs(process.argv);
if (!Number.isFinite(args.seed)) {
    console.error('Invalid --seed (must be an integer).');
    process.exit(1);
}

const mapsDir    = args.outDir ? path.resolve(args.outDir) : path.join(__dirname, 'src', 'maps');
fs.mkdirSync(mapsDir, { recursive: true });
const seedPath   = path.join(mapsDir, `map_pool_seed${args.seed}.json`);
const activePath = path.join(mapsDir, 'map_pool.json');

// ── Generate or reuse cache ───────────────────────────────────────────────────
let pool;
if (fs.existsSync(seedPath) && !args.force) {
    console.log(`[gen] cache hit — reusing ${path.basename(seedPath)} (use --force to regenerate)`);
    pool = JSON.parse(fs.readFileSync(seedPath, 'utf8'));
} else {
    console.log(`[gen] generating pool for seed=${args.seed}...`);
    pool = generateMapPool(args.seed);
    fs.writeFileSync(seedPath, JSON.stringify(pool, null, 2));
    console.log(`[gen] wrote cache ${path.basename(seedPath)}`);
}

// ── Summary ───────────────────────────────────────────────────────────────────
console.log(`\nMap pool summary (seed=${pool.seed}, ${pool.maps.length} maps):`);
pool.maps.forEach((m, i) => {
    console.log(
        `  Map ${String(i + 1).padStart(2)}: profile=${m.profile.padEnd(13)} ` +
        `cells=${String(m.cell_count).padStart(5)} ` +
        `low=${String(m.tier_counts.low_food).padStart(4)} ` +
        `med=${String(m.tier_counts.medium_food).padStart(4)} ` +
        `pres=${String(m.tier_counts.prestige_food).padStart(4)} ` +
        `caves=${String(m.tier_counts.cave).padStart(4)}`
    );
});

// ── Activate (copy to map_pool.json so the webpack bundle loads it) ───────────
fs.writeFileSync(activePath, JSON.stringify(pool, null, 2));
console.log(`\n[gen] active ${path.basename(activePath)} -> seed=${pool.seed} (${pool.maps.length} maps)`);
console.log(`[gen] set MAP_SEED: ${pool.seed} in src/WorldConfig.js, then rebuild (npm run build).`);
