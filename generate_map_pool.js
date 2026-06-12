/**
 * generate_map_pool.js
 *
 * Run once to produce src/maps/map_pool.json.
 * Usage: node generate_map_pool.js [output_path]
 *
 * Generates 50 maps across 5 difficulty profiles (10 maps each).
 * Maps are stored as relative coordinates {c_rel, r_rel, name} so they
 * scale correctly to any grid size at runtime.
 *
 * Profile summary (patches guaranteed per map):
 *   1 sparse       : low ≥4, medium ≥2, prestige ≥2, caves 8-10
 *   2 low-medium   : low ≥6, medium ≥4, prestige ≥3, caves 6-8
 *   3 balanced     : low ≥8, medium ≥6, prestige ≥5, caves 5-7
 *   4 medium-rich  : low ≥4, medium ≥8, prestige ≥6, caves 3-5
 *   5 prestige-rich: low ≥3, medium ≥6, prestige ≥9, caves 2-4
 *
 * Each food patch has a landmark ring (matching tier) placed at radius
 * 10-14 cells from the patch centre so organisms can navigate toward
 * food before they can directly see it.
 *
 * Cycle order: one map from each profile per 5-map group, repeated 10×.
 * [p1[0], p2[0], p3[0], p4[0], p5[0], p1[1], p2[1], ..., p5[9]]
 */

'use strict';

const path = require('path');
const fs   = require('fs');

// ── Virtual grid ─────────────────────────────────────────────────────────────
// Maps are generated on a reference grid then normalised to relative coords.
// Reference size matches the typical desktop browser window at 1x zoom.
const GRID_COLS = 400;
const GRID_ROWS = 300;

// ── Seeded RNG (mulberry32) ───────────────────────────────────────────────────
function mulberry32(seed) {
    return function() {
        seed |= 0; seed = seed + 0x6D2B79F5 | 0;
        let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

// ── Profile definitions ───────────────────────────────────────────────────────
// Each profile: min guaranteed patch counts and patch size ranges per tier.
// Counts are minimums — generator may add more if space allows.
const PROFILES = [
    {
        name:    'sparse',
        low:     { min: 4,  max: 6,  size: [80,  130] },
        medium:  { min: 2,  max: 4,  size: [60,  100] },
        prestige:{ min: 2,  max: 3,  size: [40,  70]  },
        caves:   { min: 8,  max: 10 },
    },
    {
        name:    'low_medium',
        low:     { min: 6,  max: 8,  size: [100, 160] },
        medium:  { min: 4,  max: 6,  size: [80,  130] },
        prestige:{ min: 3,  max: 5,  size: [50,  80]  },
        caves:   { min: 6,  max: 8  },
    },
    {
        name:    'balanced',
        low:     { min: 8,  max: 10, size: [130, 200] },
        medium:  { min: 6,  max: 9,  size: [90,  150] },
        prestige:{ min: 5,  max: 8,  size: [60,  100] },
        caves:   { min: 5,  max: 7  },
    },
    {
        name:    'medium_rich',
        low:     { min: 4,  max: 6,  size: [100, 150] },
        medium:  { min: 8,  max: 11, size: [110, 170] },
        prestige:{ min: 6,  max: 9,  size: [70,  110] },
        caves:   { min: 3,  max: 5  },
    },
    {
        name:    'prestige_rich',
        low:     { min: 3,  max: 5,  size: [80,  120] },
        medium:  { min: 6,  max: 8,  size: [90,  140] },
        prestige:{ min: 9,  max: 12, size: [80,  120] },
        caves:   { min: 2,  max: 4  },
    },
];

// ── Map generator ─────────────────────────────────────────────────────────────

function generateMap(profile, seed) {
    const rng  = mulberry32(seed);
    const cols = GRID_COLS;
    const rows = GRID_ROWS;
    const cx   = Math.floor(cols / 2);
    const cy   = Math.floor(rows / 2);

    // Virtual grid: 'empty' or cell name string
    const grid = new Array(cols * rows).fill('empty');
    const idx  = (c, r) => r * cols + c;
    const inBounds = (c, r) => c >= 0 && c < cols && r >= 0 && r < rows;
    const getCell  = (c, r) => inBounds(c, r) ? grid[idx(c, r)] : null;
    const setCell  = (c, r, name) => {
        if (inBounds(c, r) && grid[idx(c, r)] === 'empty') {
            grid[idx(c, r)] = name;
        }
    };

    const cells = [];  // final output: {c_rel, r_rel, name}
    const placed_centres = [];

    // ── Helpers ───────────────────────────────────────────────────────────────

    function placeBlob(cx, cy, sigma, name, count) {
        let placed = 0;
        for (let attempt = 0; attempt < count * 8 && placed < count; attempt++) {
            const u1 = Math.max(1e-10, rng());
            const u2 = rng();
            const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
            const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);
            const c  = Math.round(cx + z0 * sigma);
            const r  = Math.round(cy + z1 * sigma);
            if (getCell(c, r) === 'empty') {
                setCell(c, r, name);
                placed++;
            }
        }
    }

    function placeRect(cx, cy, w, h, name) {
        for (let dc = 0; dc < w; dc++) {
            for (let dr = 0; dr < h; dr++) {
                setCell(cx + dc, cy + dr, name);
            }
        }
    }

    // Place landmark lines pointing toward a food patch.
    // Each line starts at a random offset from the patch and draws inward.
    // This matches the document description: "blurred lines indicating food nearby".
    // n_lines: how many lines to draw (typically 2-4 per patch).
    // line_len: length of each line in cells (typically 18-28).
    // offset_r: how far from the patch centre to start each line.
    function placeLandmarkLines(patch_cx, patch_cy, landmark_name, n_lines, line_len, offset_r) {
        for (let i = 0; i < n_lines; i++) {
            // Start point: random direction, offset_r cells from patch centre
            const angle     = rng() * 2 * Math.PI;
            const start_c   = Math.round(patch_cx + offset_r * Math.cos(angle));
            const start_r   = Math.round(patch_cy + offset_r * Math.sin(angle));

            // Direction: point TOWARD the patch centre (inward vector)
            const dx = patch_cx - start_c;
            const dy = patch_cy - start_r;
            const dist = Math.sqrt(dx * dx + dy * dy) || 1;
            const ux = dx / dist;  // unit vector toward patch
            const uy = dy / dist;

            // Slight angular jitter so lines aren't perfectly straight
            const jitter = (rng() - 0.5) * 0.4;
            const cos_j  = Math.cos(jitter);
            const sin_j  = Math.sin(jitter);
            const vx = ux * cos_j - uy * sin_j;
            const vy = ux * sin_j + uy * cos_j;

            // Draw the line
            for (let step = 0; step < line_len; step++) {
                const c = Math.round(start_c + vx * step);
                const r = Math.round(start_r + vy * step);
                if (getCell(c, r) === 'empty') {
                    setCell(c, r, landmark_name);
                }
            }
        }
    }

    function pickPosition(min_r, max_r, min_sep, attempts = 80) {
        // Try with original sep, then fall back with progressively looser constraints
        // so guaranteed-minimum patches are always placed.
        for (const sep of [min_sep, min_sep * 0.7, min_sep * 0.4, 15]) {
            for (let i = 0; i < attempts; i++) {
                const angle = rng() * 2 * Math.PI;
                const dist  = min_r + rng() * (max_r - min_r);
                const c     = Math.round(cx + dist * Math.cos(angle));
                const r     = Math.round(cy + dist * Math.sin(angle));
                if (c < 10 || r < 10 || c > cols - 10 || r > rows - 10) continue;
                if (placed_centres.some(p => Math.hypot(p.c - c, p.r - r) < sep)) continue;
                return { c, r };
            }
        }
        return null;
    }

    // ── Spawn food: tight cluster at centre ───────────────────────────────────
    placeBlob(cx, cy, 18, 'food', 150);
    placed_centres.push({ c: cx, r: cy });

    // ── Low food patches with landmark rings ──────────────────────────────────
    const low_count = profile.low.min + Math.floor(rng() * (profile.low.max - profile.low.min + 1));
    for (let i = 0; i < low_count; i++) {
        const pos = pickPosition(50, 200, 70);
        if (!pos) continue;
        placed_centres.push(pos);
        const size = profile.low.size[0] + Math.floor(rng() * (profile.low.size[1] - profile.low.size[0]));
        placeBlob(pos.c, pos.r, 15, 'low food', size);
        // 2-3 landmark lines pointing toward this low food patch
        const low_n_lines  = 2 + Math.floor(rng() * 2);
        const low_line_len = 18 + Math.floor(rng() * 8);
        const low_offset   = 22 + Math.floor(rng() * 10);
        placeLandmarkLines(pos.c, pos.r, 'low food landmark', low_n_lines, low_line_len, low_offset);
    }

    // ── Medium food patches with landmark rings ───────────────────────────────
    const med_count = profile.medium.min + Math.floor(rng() * (profile.medium.max - profile.medium.min + 1));
    for (let i = 0; i < med_count; i++) {
        const pos = pickPosition(40, 190, 65);
        if (!pos) continue;
        placed_centres.push(pos);
        const size = profile.medium.size[0] + Math.floor(rng() * (profile.medium.size[1] - profile.medium.size[0]));
        placeBlob(pos.c, pos.r, 14, 'medium food', size);
        // 2-3 landmark lines pointing toward this medium food patch
        const med_n_lines  = 2 + Math.floor(rng() * 2);
        const med_line_len = 20 + Math.floor(rng() * 9);
        const med_offset   = 25 + Math.floor(rng() * 10);
        placeLandmarkLines(pos.c, pos.r, 'medium food landmark', med_n_lines, med_line_len, med_offset);
    }

    // ── Prestige food patches with landmark rings ─────────────────────────────
    const pres_count = profile.prestige.min + Math.floor(rng() * (profile.prestige.max - profile.prestige.min + 1));
    for (let i = 0; i < pres_count; i++) {
        const pos = pickPosition(60, 210, 55);
        if (!pos) continue;
        placed_centres.push(pos);
        const size = profile.prestige.size[0] + Math.floor(rng() * (profile.prestige.size[1] - profile.prestige.size[0]));
        placeBlob(pos.c, pos.r, 12, 'prestige food', size);
        // Prestige gets more lines (3-4) at longer range as a stronger signal
        const pres_n_lines  = 3 + Math.floor(rng() * 2);
        const pres_line_len = 22 + Math.floor(rng() * 11);
        const pres_offset   = 28 + Math.floor(rng() * 12);
        placeLandmarkLines(pos.c, pos.r, 'prestige food landmark', pres_n_lines, pres_line_len, pres_offset);
        // Extra shorter lines at greater distance for early detection
        placeLandmarkLines(pos.c, pos.r, 'prestige food landmark', 2, 14, pres_offset + 18);
    }

    // ── Caves ─────────────────────────────────────────────────────────────────
    const cave_count = profile.caves.min + Math.floor(rng() * (profile.caves.max - profile.caves.min + 1));
    const cave_specs = [
        { min_r: 40,  max_r: 100 },
        { min_r: 100, max_r: 160 },
        { min_r: 160, max_r: 230 },
    ];
    for (let i = 0; i < cave_count; i++) {
        const spec = cave_specs[Math.floor(i / Math.ceil(cave_count / cave_specs.length))] || cave_specs[2];
        const pos  = pickPosition(spec.min_r, spec.max_r, 50);
        if (!pos) continue;
        placed_centres.push(pos);
        const w = 10 + Math.floor(rng() * 10);
        const h = 10 + Math.floor(rng() * 10);
        placeRect(pos.c - Math.floor(w / 2), pos.r - Math.floor(h / 2), w, h, 'cave');
    }

    // ── Normalise to relative coords ──────────────────────────────────────────
    for (let c = 0; c < cols; c++) {
        for (let r = 0; r < rows; r++) {
            const name = grid[idx(c, r)];
            if (name !== 'empty') {
                cells.push({
                    c_rel: c / cols,
                    r_rel: r / rows,
                    name,
                });
            }
        }
    }

    return {
        profile: profile.name,
        seed,
        cell_count: cells.length,
        tier_counts: {
            low_food:     cells.filter(x => x.name === 'low food').length,
            medium_food:  cells.filter(x => x.name === 'medium food').length,
            prestige_food:cells.filter(x => x.name === 'prestige food').length,
            cave:         cells.filter(x => x.name === 'cave').length,
        },
        cells,
    };
}

// ── Generate 50 maps (10 per profile) and interleave ─────────────────────────

const maps_per_profile = 10;
const profile_banks    = PROFILES.map((profile, pi) => {
    return Array.from({ length: maps_per_profile }, (_, mi) => {
        const seed = (pi + 1) * 1000 + mi;  // deterministic, human-readable seeds
        console.log(`  Generating profile=${profile.name} map=${mi + 1} seed=${seed}...`);
        return generateMap(profile, seed);
    });
});

// Interleave: [p1[0], p2[0], p3[0], p4[0], p5[0], p1[1], p2[1], ...]
const ordered_maps = [];
for (let mi = 0; mi < maps_per_profile; mi++) {
    for (let pi = 0; pi < PROFILES.length; pi++) {
        ordered_maps.push(profile_banks[pi][mi]);
    }
}

// Summary
console.log('\nMap pool summary:');
ordered_maps.forEach((m, i) => {
    console.log(
        `  Map ${String(i + 1).padStart(2)}: profile=${m.profile.padEnd(13)} ` +
        `cells=${String(m.cell_count).padStart(5)} ` +
        `low=${String(m.tier_counts.low_food).padStart(4)} ` +
        `med=${String(m.tier_counts.medium_food).padStart(4)} ` +
        `pres=${String(m.tier_counts.prestige_food).padStart(4)} ` +
        `caves=${String(m.tier_counts.cave).padStart(4)}`
    );
});

const pool = {
    version:       1,
    generated_at:  new Date().toISOString(),
    map_count:     ordered_maps.length,
    reference_grid:{ cols: GRID_COLS, rows: GRID_ROWS },
    cycle_order:   'interleaved_profiles',
    profiles:      PROFILES.map(p => p.name),
    maps:          ordered_maps,
};

const out_path = process.argv[2]
    || path.join(__dirname, 'src', 'maps', 'map_pool.json');

fs.mkdirSync(path.dirname(out_path), { recursive: true });
fs.writeFileSync(out_path, JSON.stringify(pool, null, 2));
console.log(`\nWrote ${ordered_maps.length} maps to ${out_path}`);