const WorldConfig = {
    headless: false,
    clear_walls_on_reset: false,
    auto_reset: true,
    auto_pause: false,
    brush_size: 2,
    learning_enabled: true,  // Set to true to run RL experiment with AdvancedOrganism
    experiment_mode: 'standard', // 'standard' | 'frozen_pg' | 'pure_rl'

    // ── Map seed ──────────────────────────────────────────────────────────
    // One seed per run, shared by all conditions: every condition runs on the
    // identical map pool so results are comparable across conditions.
    // The active map_pool.json must be generated for this seed first:
    //   node generate_map_pool.js --seed <MAP_SEED>
    // The runtime warns if the loaded pool's seed doesn't match this value.
    MAP_SEED: 1,

    // ── Hardcoded map dimensions ──────────────────────────────────────────
    // When set, these override the canvas-derived size in WorldEnvironment.
    // null = use canvas size (browser default). Set both to fix a specific map.
    // cell_size is always 2 — changing it rescales everything proportionally.
    // Example: 400×300 at cell_size=2 gives a 200×150 grid.
    MAP_COLS: null,   // e.g. 200 for a 200-column grid
    MAP_ROWS: null,   // e.g. 150 for a 150-row grid

    // ── Generation length scaling ─────────────────────────────────────────
    // TICKS_PER_MAP * MAPS_PER_GEN = total ticks per generation.
    // Resolved in src/Organism/GenerationConstants.js, the single source of
    // truth imported by all managers AND by AdvancedOrganism's epsilon-decay
    // window — so setting these here scales generation length everywhere at once
    // (and keeps the epsilon horizon in sync). Values are read once at load;
    // set them and rebuild. null = use the defaults (2000 / 5).
    TICKS_PER_MAP: null,   // e.g. 2000
    MAPS_PER_GEN:  null,   // e.g. 5
}

module.exports = WorldConfig;