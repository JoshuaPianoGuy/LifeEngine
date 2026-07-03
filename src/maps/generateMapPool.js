/**
 * generateMapPool.js
 *
 * Pure (browser-free, render-free) map-pool generator. Importable from both the
 * offline CLI (generate_map_pool.js) and a future headless/HPC runner — it has
 * no DOM, canvas, or webpack dependencies, only Node-safe JS.
 *
 * generateMapPool(masterSeed) -> deterministic pool object for that seed.
 * The SAME masterSeed always yields byte-identical maps (the file carries no
 * timestamp), so a seed is a complete, reproducible identifier for a run's
 * terrain. Every per-map seed is derived from the master seed, so different
 * master seeds give entirely different (but still deterministic) pools.
 *
 * Generates 50 maps across 5 difficulty profiles (10 maps each).
 * Maps are stored as relative coordinates {c_rel, r_rel, name} so they
 * scale correctly to any grid size at runtime (incl. large 1000x1000 grids).
 *
 * LAYOUT — radial tiers (low -> high value, spawn outward):
 *   Food value increases with distance from the spawn point (map centre):
 *     starter food : tight cluster at the centre (bootstraps the founder)
 *     low food     : inner band   (f ≈ 0.12–0.46 of the usable radius)
 *     medium food  : middle band  (f ≈ 0.48–0.70)
 *     prestige food: outer band   (f ≈ 0.72–0.97) — guarded by patrol predators
 *   Reaching high-value food therefore requires travelling out into predator
 *   territory: the forage-vs-avoid pressure the experiment selects for.
 *
 * Profile summary (patches guaranteed per map — controls DENSITY, not layout):
 *   1 sparse       : low ≥4, medium ≥2, prestige ≥2, caves 8-10
 *   2 low-medium   : low ≥6, medium ≥4, prestige ≥3, caves 6-8
 *   3 balanced     : low ≥8, medium ≥6, prestige ≥5, caves 5-7
 *   4 medium-rich  : low ≥4, medium ≥8, prestige ≥6, caves 3-5
 *   5 prestige-rich: low ≥3, medium ≥6, prestige ≥9, caves 2-4
 *
 * Cycle order: one map from each profile per 5-map group, repeated 10×.
 * Each group runs richest -> sparsest so organisms get an easier (food-rich)
 * start and ease into sparse maps later:
 *   balanced, medium_rich, prestige_rich, low_medium, sparse   (see CYCLE_ORDER)
 * Group g, slot k -> profile_banks[CYCLE_ORDER[k]][g], for g in 0..9, k in 0..4.
 */

'use strict';

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

// Derive a per-map uint32 seed from the master seed + profile/map indices.
// Pure integer mixing (deterministic and platform-independent) so the same
// master seed reproduces the identical 50-map pool on any machine.
function deriveSeed(masterSeed, profileIndex, mapIndex) {
    let h = masterSeed | 0;
    h = Math.imul(h ^ (profileIndex + 1), 0x9E3779B1);
    h = Math.imul(h ^ (mapIndex + 1),     0x85EBCA77);
    h ^= h >>> 13;
    h = Math.imul(h, 0xC2B2AE3D);
    h ^= h >>> 16;
    return h >>> 0;
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

// Within-group profile order (richest -> sparsest). Values are indices into
// PROFILES, so the underlying maps/seeds are unchanged — only the cycle
// sequence differs. PROFILES is [sparse, low_medium, balanced, medium_rich,
// prestige_rich]; this yields balanced, medium_rich, prestige_rich,
// low_medium, sparse for each 5-map group.
const CYCLE_ORDER = [2, 3, 4, 1, 0];

// ── Map generator ─────────────────────────────────────────────────────────────

function generateMap(profile, seed) {
    const rng  = mulberry32(seed);
    const cols = GRID_COLS;
    const rows = GRID_ROWS;
    const cx   = Math.floor(cols / 2);
    const cy   = Math.floor(rows / 2);
    // Elliptical radius (matches the grid aspect) so a radial fraction f maps
    // into the rectangular map proportionally — f≈1 reaches near every edge,
    // filling the space instead of leaving the left/right thirds empty.
    const MARGIN = 12;
    const rx = cx - MARGIN;
    const ry = cy - MARGIN;

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

    // Radial placement: spread `n_slots` patches around an elliptical band
    // (f_lo..f_hi of the usable radius) by assigning each an angular sector,
    // with jitter in both angle and radius. Sectoring guarantees angular
    // spacing so patches don't clump; the min_sep check (with progressive
    // fallback so guaranteed-minimum patches always place) prevents overlap
    // with anything already placed, including other tiers' patches.
    function pickRadialPosition(f_lo, f_hi, n_slots, slot_i, min_sep, attempts = 50) {
        const sector = (2 * Math.PI) / Math.max(1, n_slots);
        for (const sep of [min_sep, min_sep * 0.7, min_sep * 0.45, 16]) {
            for (let i = 0; i < attempts; i++) {
                const angle = (slot_i + 0.5) * sector + (rng() - 0.5) * sector * 0.85;
                const f     = f_lo + rng() * (f_hi - f_lo);
                const c     = Math.round(cx + f * rx * Math.cos(angle));
                const r     = Math.round(cy + f * ry * Math.sin(angle));
                if (c < 10 || r < 10 || c > cols - 10 || r > rows - 10) continue;
                if (placed_centres.some(p => Math.hypot(p.c - c, p.r - r) < sep)) continue;
                return { c, r };
            }
        }
        return null;
    }

    // True iff every cell of the rectangle (expanded by `margin` on each side)
    // is in-bounds AND currently empty. getCell returns null out of bounds, so
    // that also fails the test — keeping caves wholly inside the grid.
    function isRectClear(left, top, w, h, margin) {
        for (let c = left - margin; c < left + w + margin; c++) {
            for (let r = top - margin; r < top + h + margin; r++) {
                if (getCell(c, r) !== 'empty') return false;
            }
        }
        return true;
    }

    // Radial position picker for caves that, unlike pickRadialPosition (which
    // only spaces patch CENTRES), requires the cave's whole w×h rectangle to be
    // clear of food and landmarks. A progressive margin fallback prefers a gap
    // around the cave but degrades to 0 so a cave can still place in crowded
    // maps — it may then touch food/landmarks but never intersect them.
    function pickClearCavePosition(f_lo, f_hi, n_slots, slot_i, w, h, base_margin, attempts = 60) {
        const sector = (2 * Math.PI) / Math.max(1, n_slots);
        const half_w = Math.floor(w / 2);
        const half_h = Math.floor(h / 2);
        for (const margin of [base_margin, Math.floor(base_margin / 2), 1, 0]) {
            for (let i = 0; i < attempts; i++) {
                const angle = (slot_i + 0.5) * sector + (rng() - 0.5) * sector * 0.85;
                const f     = f_lo + rng() * (f_hi - f_lo);
                const c     = Math.round(cx + f * rx * Math.cos(angle));
                const r     = Math.round(cy + f * ry * Math.sin(angle));
                if (isRectClear(c - half_w, r - half_h, w, h, margin)) {
                    return { c, r };
                }
            }
        }
        return null;
    }

    // Radial bands per tier (fractions of the usable radius). Gaps between
    // bands keep the low -> medium -> prestige gradient visually distinct.
    const BANDS = {
        low:      [0.12, 0.46],
        medium:   [0.48, 0.70],
        prestige: [0.72, 0.97],
        cave:     [0.50, 0.86],
    };

    // ── Spawn food: tight cluster at centre ───────────────────────────────────
    placeBlob(cx, cy, 18, 'food', 150);
    placed_centres.push({ c: cx, r: cy });

    // ── Low food patches with landmark rings (INNER band) ─────────────────────
    const low_count = profile.low.min + Math.floor(rng() * (profile.low.max - profile.low.min + 1));
    for (let i = 0; i < low_count; i++) {
        const pos = pickRadialPosition(BANDS.low[0], BANDS.low[1], low_count, i, 60);
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

    // ── Medium food patches with landmark rings (MIDDLE band) ─────────────────
    const med_count = profile.medium.min + Math.floor(rng() * (profile.medium.max - profile.medium.min + 1));
    for (let i = 0; i < med_count; i++) {
        const pos = pickRadialPosition(BANDS.medium[0], BANDS.medium[1], med_count, i, 64);
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

    // ── Prestige food patches with landmark rings (OUTER band) ────────────────
    const pres_count = profile.prestige.min + Math.floor(rng() * (profile.prestige.max - profile.prestige.min + 1));
    for (let i = 0; i < pres_count; i++) {
        const pos = pickRadialPosition(BANDS.prestige[0], BANDS.prestige[1], pres_count, i, 58);
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

    // ── Caves (mid/outer band — refuges along the route to prestige food) ─────
    // Sited between the medium and prestige rings so prey foraging the outer
    // ring have somewhere to break predator pursuit (predators can't enter
    // caves). Placed LAST so the clear-rectangle test sees all food/landmarks
    // already on the grid: each cave is positioned where its full rectangle
    // (plus a margin) is empty, so caves never intersect food or landmarks and
    // never end up with holes punched through them by earlier placement.
    const cave_count = profile.caves.min + Math.floor(rng() * (profile.caves.max - profile.caves.min + 1));
    for (let i = 0; i < cave_count; i++) {
        const w = 10 + Math.floor(rng() * 10);
        const h = 10 + Math.floor(rng() * 10);
        const pos = pickClearCavePosition(BANDS.cave[0], BANDS.cave[1], cave_count, i, w, h, 4);
        if (!pos) continue;
        placed_centres.push(pos);
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

/**
 * Build the full 50-map pool for a master seed.
 * Deterministic: same masterSeed -> identical maps (no timestamp in output,
 * so the file is content-addressable by seed).
 *
 * @param {number} masterSeed  integer seed for the whole run
 * @returns {object} pool { version, seed, map_count, reference_grid,
 *                          cycle_order, profiles, maps }
 */
function generateMapPool(masterSeed) {
    const seed = masterSeed | 0;
    const maps_per_profile = 10;

    const profile_banks = PROFILES.map((profile, pi) =>
        Array.from({ length: maps_per_profile }, (_, mi) =>
            generateMap(profile, deriveSeed(seed, pi, mi))));

    // Interleave: [bal[0], medr[0], presr[0], lowm[0], sparse[0], bal[1], ...]
    const ordered_maps = [];
    for (let mi = 0; mi < maps_per_profile; mi++) {
        for (const pi of CYCLE_ORDER) {
            ordered_maps.push(profile_banks[pi][mi]);
        }
    }

    return {
        version:        2,
        seed,
        map_count:      ordered_maps.length,
        reference_grid: { cols: GRID_COLS, rows: GRID_ROWS },
        cycle_order:    'interleaved_profiles_rich_to_sparse',
        profiles:       PROFILES.map(p => p.name),
        maps:           ordered_maps,
    };
}

module.exports = {
    generateMapPool,
    generateMap,
    deriveSeed,
    PROFILES,
    CYCLE_ORDER,
    GRID_COLS,
    GRID_ROWS,
};
