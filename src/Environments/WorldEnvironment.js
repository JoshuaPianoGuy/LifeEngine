const Environment = require('./Environment');
const Renderer = require('../Rendering/Renderer');
const GridMap = require('../Grid/GridMap');
const Organism = require('../Organism/Organism');
const AdvancedOrganism = require('../Organism/AdvancedOrganism');
const GAManager = require('../Organism/GAManager');
const FrozenPolicyManager = require('../Organism/FrozenPolicyManager');
const PureRLManager = require('../Organism/PureRLManager');
const PredatorManager = require('../Organism/PredatorManager');
const PredatorHyperparameters = require('../Organism/PredatorHyperparameters');
const CellStates = require('../Organism/Cell/CellStates');
const EnvironmentController = require('../Controllers/EnvironmentController');
const Hyperparams = require('../Hyperparameters.js');
const FossilRecord = require('../Stats/FossilRecord');
const WorldConfig = require('../WorldConfig');
const ExperimentParams = require('../ExperimentParams');
const FoodShuffle = require('./FoodShuffle');
const logger = require('../Logger');
const SerializeHelper = require('../Utils/SerializeHelper');
const Species = require('../Stats/Species');

class WorldEnvironment extends Environment {
    constructor(engine, cell_size) {
        super();
        this.engine = engine;
        this.renderer = new Renderer('env-canvas', 'env', cell_size);
        this.controller = new EnvironmentController(this, this.renderer.canvas);
        this.num_rows = Math.ceil(this.renderer.height / cell_size);
        this.num_cols = Math.ceil(this.renderer.width / cell_size);

        // Override canvas-derived dimensions with hardcoded values if set.
        // This is the single place to change map size for HPC/headless runs
        // without touching canvas or browser layout.
        if (WorldConfig.MAP_COLS != null && WorldConfig.MAP_ROWS != null) {
            this.renderer.fillShape(WorldConfig.MAP_ROWS * cell_size, WorldConfig.MAP_COLS * cell_size);
            this.num_cols = WorldConfig.MAP_COLS;
            this.num_rows = WorldConfig.MAP_ROWS;
        }

        this.grid_map = new GridMap(this.num_cols, this.num_rows, cell_size);
        this.organisms = [];
        this.walls = [];
        this.total_mutability = 0;
        this.largest_cell_count = 0;
        this.reset_count = 0;
        this.total_ticks = 0;
        this.data_update_rate = 100;

        // Non-stationary shuffle environment: restore the base food-tier payoffs
        // and (re)seed the reshuffle schedule so this fresh world never inherits
        // a stale permutation from a previous run/probe (FOOD_ENERGY_VALUES is
        // process-global). No-op when food_shuffle_period is 0. See FoodShuffle.js.
        FoodShuffle.reset(ExperimentParams.food_shuffle_seed);

        // Day/night cycle — equal length
        this.day_length   = 300;
        this.night_length = 300;
        this.cycle_length = this.day_length + this.night_length;

        // Snapshot of the initial map layout — restored at the start of each
        // new generation so all generations face identical conditions.
        // Null until generateWorld() is called for the first time.
        this._world_snapshot = null;

        // ── Fixed map pool ────────────────────────────────────────────────
        // 50 pre-generated maps loaded from map_pool.json.
        // All conditions cycle through the same sequence for reproducibility.
        // ── Fixed map pool ────────────────────────────────────────────────
        this._map_pool       = null;
        this._map_reference_grid = null;        // {cols, rows} the maps were authored on
        this._sequence_index = 0;               // Tracks where we are in the sequence
        this._loadMapPool();
        // Fixed deterministic order through ALL pool maps: every run cycles the
        // identical sequence 0,1,...,N-1. The pool is already profile-interleaved
        // (sparse, low_medium, balanced, medium_rich, prestige_rich, repeating),
        // so each pass over the 50 maps is a balanced rotation of all 5 density
        // profiles. Reproducible across runs/conditions by construction.
        this._map_sequence = (this._map_pool && this._map_pool.length > 0)
            ? Array.from({ length: this._map_pool.length }, (_, i) => i)
            : [0];

        this.learning_enabled = WorldConfig.learning_enabled;
        const center    = this.grid_map.getCenter();
        const spawn_col = center[0] + 15;
        const spawn_row = center[1] + 15;
        const use_frozen_pg = WorldConfig.experiment_mode === 'frozen_pg';
        const use_pure_rl   = WorldConfig.experiment_mode === 'pure_rl';
        if (use_frozen_pg) {
            this.learning_enabled = true;
            this.ga_manager = new FrozenPolicyManager(this, spawn_col, spawn_row);
        } else if (use_pure_rl) {
            this.learning_enabled = true;
            this.ga_manager = new PureRLManager(this, spawn_col, spawn_row);
        } else {
            this.ga_manager = new GAManager(this, this.learning_enabled, spawn_col, spawn_row);
        }

        FossilRecord.setEnv(this);

        // ── Predation ────────────────────────────────────────────────────
        // Independent of ga_manager: predators are a fixed environmental
        // hazard, not part of the experimental population. They persist
        // across generation/map boundaries (see PredatorManager docs) and
        // are spawned once here; population count is enforced by
        // PredatorManager for the entire lifetime of the simulation.
        this.predator_manager = new PredatorManager(this);
    }

    update() {
        this.clearDeadOrganisms(true);
        if (Hyperparams.foodDropProb > 0) {
            this.generateFood();
        }
        this.total_ticks++;

        // Non-stationary shuffle environment: on schedule, permute which food
        // tier pays which energy value (in place, on the global env clock so it
        // changes within a lifetime). No-op when food_shuffle_period is 0.
        const shuffled = FoodShuffle.maybeShuffle(this.total_ticks, ExperimentParams.food_shuffle_period);
        if (shuffled) {
            logger.logEvent('FOOD', `Food payoffs reshuffled @tick ${this.total_ticks}`, shuffled);
        }

        // Delegate generation/map lifecycle check to the GA Manager if present
        if (this.ga_manager) {
            const status = this.ga_manager.tick();
            if (status === 'NEXT_MAP') {
                this.generateWorld();
                this.ga_manager.startNextMap();
                this.predator_manager.relocateAll();
                this.renderFull();
            } else if (status === 'NEXT_GENERATION') {
                this.ga_manager.evolve();
                this.generateWorld();
                this.ga_manager.spawnGeneration();
                this.predator_manager.relocateAll();
                this.renderFull();
                this.predator_manager.tick();
                return;
            } else if (status === 'NEXT_WINDOW') {
                // PureRLManager: log metrics and reset window counters but do
                // NOT reset the population — surviving organisms continue living.
                // closeWindow() also handles COLLAPSE (population hit zero during
                // the episode) by reseeding from its rolling weight buffer at
                // THIS boundary — per its design, collapse is recovered here, not
                // mid-episode. So after it returns the population is repopulated
                // either way, and we simply start the next window on a fresh map,
                // mirroring the NEXT_MAP path.
                this.ga_manager.closeWindow();
                this.generateWorld();
                this.ga_manager.startNextMap();
                this.predator_manager.relocateAll();
                this.renderFull();
            }
        }

        // Predators persist across generation/map boundaries and are ticked
        // every world update regardless of ga_manager state — they are not
        // part of the experimental population and have no generation concept
        // of their own. (Skipped above via early `return` only on the
        // NEXT_GENERATION branch, where it's called explicitly instead,
        // since that branch returns before reaching here.)
        this.predator_manager.tick();

        // Randomly respawn food periodically
        if (this.total_ticks % 1200 == 0) {
            this._restoreWorldSnapshotWithProbability(0.2);
            // _restoreWorldSnapshotWithProbability calls fillGrid(empty, ...)
            // first, which blanks every non-wall cell including whatever
            // square a predator currently occupies. Predators redraw
            // themselves every tick via their own updateGrid(), but a food/
            // landmark cell from the snapshot could land on a predator's
            // square in the same pass and overwrite it until that happens.
            // Re-drawing here closes that one-tick window.
            for (const org of this.predator_manager.predators) {
                if (org && org.living) org.updateGrid();
            }
        }
        
        if (this.total_ticks % this.data_update_rate == 0) {
            FossilRecord.updateData();
        }
    }

    render() {
        if (WorldConfig.headless) {
            this.renderer.cells_to_render.clear();
            return;
        }
        this.renderer.renderCells();
        this.renderer.renderHighlights();
    }

    renderFull() {
        this.renderer.renderFullGrid(this.grid_map.grid);
    }

    // removed removeOrganisms to favor in-place DOD methodology in clearDeadOrganisms

    OriginOfLife() {
        var center = this.grid_map.getCenter();
        var org = this.createExperimentOrganism(center[0], center[1], null);
        this.addOrganism(org);
        // Always register agent with GA manager for proper generation lifecycle management
        if (this.ga_manager) {
            this.ga_manager.registerAgent(org);
        }
        FossilRecord.addSpecies(org, null);
    }

    createExperimentOrganism(col, row, parent = null) {
        const org = new AdvancedOrganism(col, row, this, parent, this.learning_enabled, this.ga_manager);

        org.anatomy.addDefaultCell(CellStates.mouth, 0, 0);

        const eye_up = org.anatomy.addDefaultCell(CellStates.eye, 0, -1);
        if (eye_up) eye_up.direction = 0;

        const eye_right = org.anatomy.addDefaultCell(CellStates.eye, 1, 0);
        if (eye_right) eye_right.direction = 1;

        org.anatomy.addDefaultCell(CellStates.mover, 0, 1);

        const eye_left = org.anatomy.addDefaultCell(CellStates.eye, -1, 1);
        if (eye_left) eye_left.direction = 3;

        const eye_down = org.anatomy.addDefaultCell(CellStates.eye, 0, 2);
        if (eye_down) eye_down.direction = 2;

        org.anatomy.checkTypeChange();
        return org;
    }

    // ── World generation ──────────────────────────────────────────────────────
    //
    // Layout philosophy:
    //   Spawn area (radius ~20):  low density default food + sparse low food
    //                             so new organisms can survive long enough to learn
    //   Mid-range (radius 20-60): clusters of low food, landmarks pointing inward
    //   Outer ring (radius 60+):  prestige food clusters, ringed by landmarks
    //   Caves: clumps of 12-20 cells, scattered at navigable distances
    //   Obstacles: short wall segments that break up open space
    //
    // All positions are jittered so the map looks organic, not grid-like.

    generateWorld() {
        if (this._map_pool && this._map_pool.length > 0) {
            this._loadMapFromPool();
            return;
        }
        // Fallback if pool failed to load — random generation
        this.clearWalls();
        this.grid_map.fillGrid(CellStates.empty, false);
        this.walls = [];

        const cols = this.grid_map.cols;
        const rows = this.grid_map.rows;
        const cx   = Math.floor(cols / 2);
        const cy   = Math.floor(rows / 2);
        const rng  = () => Math.random();

        // Helper: pick a position at a given radius from centre, with minimum
        // separation from all already-placed patch centres.
        // Returns {c, r} or null if no valid position found after max attempts.
        const pickPosition = (min_r, max_r, existing, min_sep) => {
            for (let attempt = 0; attempt < 60; attempt++) {
                const angle = rng() * 2 * Math.PI;
                const r     = min_r + rng() * (max_r - min_r);
                const c     = Math.round(cx + r * Math.cos(angle));
                const row   = Math.round(cy + r * Math.sin(angle));
                if (c < 8 || row < 8 || c > cols - 8 || row > rows - 8) continue;
                const too_close = existing.some(p => Math.hypot(p.c - c, p.r - row) < min_sep);
                if (!too_close) return { c, r: row };
            }
            return null;
        };

        const placed_centres = [];

        // ── Default food: tight cluster at spawn ──────────────────────────────
        // Dense but small — just enough for the founding agent to survive a few
        // ticks and reproduce once. Not worth seeking after the first generation.
        this._placeBlob(cx, cy, 18, CellStates.food, 150, rng);
        placed_centres.push({ c: cx, r: cy });

        const pickRandomPosition = (existing, min_sep) => {
            for (let attempt = 0; attempt < 150; attempt++) {
                const c = 8 + Math.floor(rng() * (cols - 16));
                const r = 8 + Math.floor(rng() * (rows - 16));
                if (Math.hypot(c - cx, r - cy) < 40) continue;
                const too_close = existing.some(p => Math.hypot(p.c - c, p.r - r) < min_sep);
                if (!too_close) return { c, r };
            }
            return null;
        };

        // ── Low food: 8–10 distinct separated patches, randomly placed ───────────────
        const low_positions = [];
        const low_patch_count = 8 + Math.floor(rng() * 3);
        for (let i = 0; i < low_patch_count; i++) {
            const pos = pickRandomPosition(placed_centres, 80);
            if (!pos) continue;
            placed_centres.push(pos);
            low_positions.push(pos);

            const patch_size = 130 + Math.floor(rng() * 80);  // 130–210 cells
            this._placeBlob(pos.c, pos.r, 15, CellStates.lowFood, patch_size, rng);
            // Landmark lines pointing toward this low food patch
            this._placeLandmarkLines(pos.c, pos.r, CellStates.lowFoodLandmark,
                2 + Math.floor(rng() * 2), 18 + Math.floor(rng() * 8),
                22 + Math.floor(rng() * 10), rng);
        }

        // ── Prestige food: 6–8 tight patches, randomly placed ────────────────
        const prestige_patch_count = 6 + Math.floor(rng() * 3);
        const prestige_centres = [];
        for (let i = 0; i < prestige_patch_count; i++) {
            const pos = pickRandomPosition(placed_centres, 100);
            if (!pos) continue;
            placed_centres.push(pos);
            prestige_centres.push(pos);

            const patch_size = 60 + Math.floor(rng() * 40);  // 60–100 cells
            this._placeBlob(pos.c, pos.r, 12, CellStates.prestigeFood, patch_size, rng);
            // More prominent landmark lines for rarer prestige food
            this._placeLandmarkLines(pos.c, pos.r, CellStates.prestigeFoodLandmark,
                3 + Math.floor(rng() * 2), 22 + Math.floor(rng() * 11),
                28 + Math.floor(rng() * 12), rng);
            // Extra shorter outer lines for early detection
            this._placeLandmarkLines(pos.c, pos.r, CellStates.prestigeFoodLandmark,
                2, 14, 44 + Math.floor(rng() * 8), rng);
        }
        // Expose prestige centres so PredatorManager can place patrol predators
        // at each patch without a second grid scan.
        this.prestige_patch_centres = prestige_centres;

        // ── Medium food: scattered patches, randomly placed ─────────────────
        const med_patch_count = 7 + Math.floor(rng() * 4);
        for (let i = 0; i < med_patch_count; i++) {
            const pos = pickRandomPosition(placed_centres, 70);
            if (!pos) continue;
            placed_centres.push(pos);

            const patch_size = 90 + Math.floor(rng() * 60);  // 90–150 cells
            this._placeBlob(pos.c, pos.r, 14, CellStates.mediumFood, patch_size, rng);
            // Landmark lines pointing toward this medium food patch
            this._placeLandmarkLines(pos.c, pos.r, CellStates.mediumFoodLandmark,
                2 + Math.floor(rng() * 2), 20 + Math.floor(rng() * 9),
                25 + Math.floor(rng() * 10), rng);
        }

        // ── Caves: solid rectangles, well-separated ────────────────────────────
        // 10–15 cave rectangles at varied distances. Large enough to stand in (10×10
        // minimum). Placed with min_sep=60 from food patches so they don't
        // visually merge with food clusters.
        const cave_specs = [
            { min_r: 50, max_r: 100, count: 3 },
            { min_r: 100, max_r: 160, count: 4 },
            { min_r: 160, max_r: 240, count: 3 + Math.floor(rng() * 3) },
        ];
        for (const spec of cave_specs) {
            for (let i = 0; i < spec.count; i++) {
                const pos = pickPosition(spec.min_r, spec.max_r, placed_centres, 60);
                if (!pos) continue;
                placed_centres.push(pos);
                const w = 10 + Math.floor(rng() * 10);   // 10–20 wide
                const h = 10 + Math.floor(rng() * 10);   // 10–20 tall
                this._placeRect(pos.c - Math.floor(w/2), pos.r - Math.floor(h/2), w, h, CellStates.cave);
            }
        }

        // Obstacles removed — fixed maps use pool; fallback uses no walls.

        // Take a snapshot so all subsequent generations restore this exact layout.
        this._takeWorldSnapshot();
    }


    // ── Fixed map pool ────────────────────────────────────────────────────

    _loadMapPool() {
        // Headless/Node path: load the seed-specific pool file directly,
        // generating it on demand if missing. This lets parallel HPC jobs each
        // use their own map_pool_seed<N>.json without clobbering a shared active
        // file. eval('require') keeps fs / the generator out of the webpack bundle.
        const isNode = typeof process !== 'undefined' && !!(process.versions && process.versions.node);
        const want   = WorldConfig.MAP_SEED;
        if (isNode && want != null) {
            try {
                const req  = eval('require');
                const fs   = req('fs');
                const path = req('path');
                const file = path.join(__dirname, '..', 'maps', `map_pool_seed${want}.json`);
                let pool;
                if (fs.existsSync(file)) {
                    pool = JSON.parse(fs.readFileSync(file, 'utf8'));
                    console.log(`[WorldEnv] Loaded cached map pool ${path.basename(file)}`);
                } else {
                    const { generateMapPool } = req('../maps/generateMapPool');
                    pool = generateMapPool(want);
                    // Atomic write (temp + rename) so a parallel job that starts
                    // at the same time can't read a half-written pool file. The
                    // temp name is per-process; rename is atomic on the same fs.
                    const tmp = `${file}.tmp.${process.pid}`;
                    fs.writeFileSync(tmp, JSON.stringify(pool, null, 2));
                    fs.renameSync(tmp, file);
                    console.log(`[WorldEnv] Generated map pool for seed=${want} -> ${path.basename(file)}`);
                }
                this._map_pool = pool.maps;
                this._map_reference_grid = pool.reference_grid || null;
                this.map_seed  = (pool.seed != null) ? pool.seed : want;
                console.log(`[WorldEnv] Map pool ready: ${this._map_pool.length} maps (seed=${this.map_seed})`);
                return;
            } catch (e) {
                console.warn('[WorldEnv] Seed-specific map pool load failed, falling back:', e.message);
            }
        }

        // Browser path: the active map_pool.json is bundled at build time.
        try {
            const pool = require('../maps/map_pool.json');

            this._map_pool = pool.maps;
            this._map_reference_grid = pool.reference_grid || null;
            this.map_seed  = (pool.seed != null) ? pool.seed : null;
            console.log(`[WorldEnv] Loaded map pool: ${this._map_pool.length} maps (seed=${this.map_seed})`);

            // Sanity-check: the active pool must match the configured run seed.
            // A mismatch means map_pool.json wasn't regenerated for MAP_SEED.
            if (want != null && this.map_seed != null && this.map_seed !== want) {
                console.warn(
                    `[WorldEnv] SEED MISMATCH: active map_pool.json is seed=${this.map_seed} ` +
                    `but WorldConfig.MAP_SEED=${want}. Run: ` +
                    `node generate_map_pool.js --seed ${want}  (then rebuild).`
                );
            }
        } catch (e) {
            console.warn('[WorldEnv] map_pool.json not found — falling back to random generation:', e.message);
            this._map_pool = null;
        }
    }

    /**
     * Load the next map from the pool into the world.
     * Advances _map_pool_index (wraps at pool length).
     * Converts relative {c_rel, r_rel} coords to absolute for the current grid.
     */
    _loadMapFromPool() {
        // Get the target map index from your custom sequence array
        const target_map_index = this._map_sequence[this._sequence_index % this._map_sequence.length];
        this._sequence_index++; // Move to the next item in your sequence

        // Grab the actual map using that target index
        const map = this._map_pool[target_map_index];

        const cols = this.grid_map.cols;
        const rows = this.grid_map.rows;

        // Clear everything
        this.clearWalls();
        this.grid_map.fillGrid(CellStates.empty, false);
        this.walls = [];

        const stateMap = {
            'food':                    CellStates.food,
            'low food':                CellStates.lowFood,
            'medium food':             CellStates.mediumFood,
            'prestige food':           CellStates.prestigeFood,
            'cave':                    CellStates.cave,
            'low food landmark':       CellStates.lowFoodLandmark,
            'medium food landmark':    CellStates.mediumFoodLandmark,
            'prestige food landmark':  CellStates.prestigeFoodLandmark,
        };

        // Map cells are stored as individual relative points. When the runtime
        // grid is larger than the reference grid, Math.round(c_rel * cols) maps
        // adjacent reference cells onto non-adjacent runtime cells, leaving gaps
        // — caves look like a dotted grid and landmark lines break into sparse
        // cells. To keep caves solid and landmark lines continuous at any scale,
        // those two cell types are drawn by filling the whole runtime footprint
        // each reference cell scales to, closing the inter-cell gaps. Food stays
        // as single points (it's an organic, scattered blob by design).
        //
        // Placement order also guarantees caves never intersect food/landmarks:
        //   1. Caves first, as solid footprint blocks on the empty grid.
        //   2. Food, skipping any cell already occupied by a cave.
        //   3. Landmarks, footprint-filled but only over still-empty cells, so a
        //      line never overwrites a cave or food — it just connects its own
        //      cells through the empty gaps between them.
        // The generator already keeps caves clear of food/landmarks on the
        // reference grid (see generateMapPool.js); this preserves that on scale.
        const ref     = this._map_reference_grid;
        const scale_x = (ref && ref.cols) ? cols / ref.cols : 1;
        const scale_y = (ref && ref.rows) ? rows / ref.rows : 1;
        const fp_w    = Math.max(1, Math.ceil(scale_x));
        const fp_h    = Math.max(1, Math.ceil(scale_y));

        const landmarkStates = new Set([
            CellStates.lowFoodLandmark,
            CellStates.mediumFoodLandmark,
            CellStates.prestigeFoodLandmark,
        ]);

        // Fill a runtime footprint block for one reference cell. onlyEmpty=true
        // makes the fill yield to anything already placed (used for landmarks so
        // they never overwrite caves/food).
        const fillFootprint = (c, r, state, onlyEmpty) => {
            for (let dc = 0; dc < fp_w; dc++) {
                for (let dr = 0; dr < fp_h; dr++) {
                    const fc = c + dc;
                    const fr = r + dr;
                    if (fc < 0 || fc >= cols || fr < 0 || fr >= rows) continue;
                    if (onlyEmpty) {
                        const existing = this.grid_map.cellAt(fc, fr);
                        if (!existing || existing.state !== CellStates.empty) continue;
                    }
                    this.changeCell(fc, fr, state, null);
                }
            }
        };

        // Pass 1: caves — solid footprint blocks.
        for (const entry of map.cells) {
            if (stateMap[entry.name] !== CellStates.cave) continue;
            const c = Math.round(entry.c_rel * cols);
            const r = Math.round(entry.r_rel * rows);
            fillFootprint(c, r, CellStates.cave, false);
        }
        // Pass 2: food — single points, never over a cave.
        for (const entry of map.cells) {
            const state = stateMap[entry.name];
            if (!state || state === CellStates.cave || landmarkStates.has(state)) continue;
            const c = Math.round(entry.c_rel * cols);
            const r = Math.round(entry.r_rel * rows);
            if (c < 0 || c >= cols || r < 0 || r >= rows) continue;
            const existing = this.grid_map.cellAt(c, r);
            if (existing && existing.state === CellStates.cave) continue;
            this.changeCell(c, r, state, null);
        }
        // Pass 3: landmarks — footprint-filled over empty cells only, so lines
        // become continuous without intersecting caves or food.
        for (const entry of map.cells) {
            const state = stateMap[entry.name];
            if (!landmarkStates.has(state)) continue;
            const c = Math.round(entry.c_rel * cols);
            const r = Math.round(entry.r_rel * rows);
            fillFootprint(c, r, state, true);
        }

        // Snapshot so periodic food respawn works from this map's layout
        this._takeWorldSnapshot();

        // Derive prestige patch centres from the loaded map so patrol predators
        // can be repositioned onto the correct patches after each map swap.
        this.prestige_patch_centres = this._extractPrestigeCentres();

        console.log(`[WorldEnv] Map ${this._sequence_index}/${this._map_sequence.length} ` +
            `(pool idx ${target_map_index}) profile=${map.profile} cells=${map.cell_count} ` +
            `prestige_patches=${this.prestige_patch_centres.length}`);
    }

    // Place a food patch using 2D Gaussian spread — organic, irregular shape.
    // Keep sigma small (5–8) for distinct patches, larger (12+) for diffuse areas.

    /**
     * Cluster all prestige food cells on the current grid into patch centres
     * using greedy grouping: cells within 20 cells of an existing centre are
     * merged into it. Returns [{c, r}, ...] one per patch.
     * Used after loading from the map pool so patrol predators can home to
     * the correct prestige patches without a full re-generation.
     */
    _extractPrestigeCentres() {
        // Connected-components clustering: one centre per distinct prestige
        // patch. Two prestige cells belong to the same patch if they lie within
        // LINK cells of EACH OTHER (union-find), so a patch stays a single
        // component no matter how wide it is — the fixed-seed approach this
        // replaces fragmented any patch wider than its merge radius, spawning
        // several patrol homes (and therefore several times predatorsPerPatch
        // predators) on one visual patch. LINK only has to exceed the largest
        // internal gap of one blob while staying below inter-patch spacing,
        // which is a wide, shape-independent window.
        const PredatorHyperparams = require('../Organism/PredatorHyperparameters');
        const LINK = (PredatorHyperparams.patrol &&
                      PredatorHyperparams.patrol.patchLinkRadius) || 12;
        const MIN_CELLS = (PredatorHyperparams.patrol &&
                           PredatorHyperparams.patrol.minPatchCells) || 1;
        const LINK2 = LINK * LINK;
        const cols = this.grid_map.cols;
        const rows = this.grid_map.rows;

        // 1. Collect every prestige-food cell.
        const cells = [];
        for (let c = 0; c < cols; c++) {
            for (let r = 0; r < rows; r++) {
                const cell = this.grid_map.cellAt(c, r);
                if (cell && cell.state === CellStates.prestigeFood) cells.push([c, r]);
            }
        }
        if (cells.length === 0) return [];

        // 2. Bucket cells into LINK-sized bins so each cell only compares
        //    against neighbours in its own + adjacent bins — O(n) instead of
        //    O(n^2). Any pair within LINK lands in bins at most one apart.
        const bin = LINK > 0 ? LINK : 1;
        const buckets = new Map();
        const key = (bx, by) => bx + ',' + by;
        cells.forEach(([c, r], i) => {
            const k = key(Math.floor(c / bin), Math.floor(r / bin));
            let arr = buckets.get(k);
            if (!arr) { arr = []; buckets.set(k, arr); }
            arr.push(i);
        });

        // 3. Union-find: join cells within LINK of one another.
        const parent = cells.map((_, i) => i);
        const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
        const union = (a, b) => { const ra = find(a), rb = find(b); if (ra !== rb) parent[ra] = rb; };
        cells.forEach(([c, r], i) => {
            const bx = Math.floor(c / bin), by = Math.floor(r / bin);
            for (let dx = -1; dx <= 1; dx++) {
                for (let dy = -1; dy <= 1; dy++) {
                    const arr = buckets.get(key(bx + dx, by + dy));
                    if (!arr) continue;
                    for (const j of arr) {
                        if (j <= i) continue;
                        const ddc = c - cells[j][0], ddr = r - cells[j][1];
                        if (ddc * ddc + ddr * ddr <= LINK2) union(i, j);
                    }
                }
            }
        });

        // 4. Centroid of each component = patch centre. Components smaller than
        //    MIN_CELLS are stray tail fragments, not real patches — drop them.
        const groups = new Map();  // root -> { sum_c, sum_r, count }
        cells.forEach(([c, r], i) => {
            const root = find(i);
            let g = groups.get(root);
            if (!g) { g = { sum_c: 0, sum_r: 0, count: 0 }; groups.set(root, g); }
            g.sum_c += c; g.sum_r += r; g.count += 1;
        });
        const centres = [];
        for (const g of groups.values()) {
            if (g.count < MIN_CELLS) continue;
            centres.push({ c: Math.round(g.sum_c / g.count), r: Math.round(g.sum_r / g.count) });
        }
        return centres;
    }

    /**
     * Draw N landmark lines pointing inward toward a food patch centre.
     * Each line starts at offset_r cells from the patch and draws toward it,
     * matching the document description: directional "blurred lines" that
     * indicate a food type is nearby. The NN perceives each landmark type
     * as a distinct input (indices 5-7 in the 11-type percept vector).
     */
    _placeLandmarkLines(patch_cx, patch_cy, state, n_lines, line_len, offset_r, rng) {
        for (let i = 0; i < n_lines; i++) {
            const angle   = rng() * 2 * Math.PI;
            const start_c = Math.round(patch_cx + offset_r * Math.cos(angle));
            const start_r = Math.round(patch_cy + offset_r * Math.sin(angle));
            const dx   = patch_cx - start_c;
            const dy   = patch_cy - start_r;
            const dist = Math.sqrt(dx * dx + dy * dy) || 1;
            const ux   = dx / dist;
            const uy   = dy / dist;
            // Slight angular jitter so lines aren't perfectly straight
            const jitter = (rng() - 0.5) * 0.4;
            const vx = ux * Math.cos(jitter) - uy * Math.sin(jitter);
            const vy = ux * Math.sin(jitter) + uy * Math.cos(jitter);
            for (let step = 0; step < line_len; step++) {
                const c = Math.round(start_c + vx * step);
                const r = Math.round(start_r + vy * step);
                const cell = this.grid_map.cellAt(c, r);
                if (cell && cell.state === CellStates.empty) {
                    this.changeCell(c, r, state, null);
                }
            }
        }
    }

    _placeBlob(cx, cy, sigma, state, count, rng) {
        let placed = 0;
        for (let attempt = 0; attempt < count * 8 && placed < count; attempt++) {
            const u1 = Math.max(1e-10, rng());
            const u2 = rng();
            const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
            const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);
            const c  = Math.round(cx + z0 * sigma);
            const r  = Math.round(cy + z1 * sigma);
            const cell = this.grid_map.cellAt(c, r);
            if (cell && cell.state === CellStates.empty) {
                this.changeCell(c, r, state, null);
                placed++;
            }
        }
    }

    // Draw a straight diagonal line of cells from midpoint (cx, cy).
    _placeLine(cx, cy, direction, length, state) {
        const dx   = Math.cos(direction);
        const dy   = Math.sin(direction);
        const half = Math.floor(length / 2);
        for (let i = -half; i <= half; i++) {
            const c    = Math.round(cx + dx * i);
            const r    = Math.round(cy + dy * i);
            const cell = this.grid_map.cellAt(c, r);
            if (cell && cell.state === CellStates.empty) {
                this.changeCell(c, r, state, null);
            }
        }
    }

    // Place a solid filled rectangle centred approximately at (cx, cy).
    _placeRect(cx, cy, width, height, state) {
        for (let dc = 0; dc < width; dc++) {
            for (let dr = 0; dr < height; dr++) {
                const cell = this.grid_map.cellAt(cx + dc, cy + dr);
                if (cell && cell.state === CellStates.empty) {
                    this.changeCell(cx + dc, cy + dr, state, null);
                }
            }
        }
    }

    // Scatter food randomly within an annulus — kept for backward compat
    _scatterFood(cx, cy, min_r, max_r, state, count, rng) {
        let placed = 0;
        for (let attempt = 0; attempt < count * 4 && placed < count; attempt++) {
            const angle = rng() * 2 * Math.PI;
            const r     = min_r + rng() * (max_r - min_r);
            const c     = Math.round(cx + r * Math.cos(angle));
            const row   = Math.round(cy + r * Math.sin(angle));
            const cell  = this.grid_map.cellAt(c, row);
            if (cell && cell.state === CellStates.empty) {
                this.changeCell(c, row, state, null);
                placed++;
            }
        }
    }

    // Alias kept for any external callers
    _placeCluster(cx, cy, sigma, state, count, rng) {
        this._placeBlob(cx, cy, sigma, state, count, rng);
    }

    // ── World snapshot ───────────────────────────────────────────────────────
    // Called once after the initial generateWorld(). Stores a compact copy of
    // every non-empty, non-organism cell so the map can be restored identically
    // at the start of each new generation.
    //
    // Organism cells are excluded — only independent cells (food, wall, cave,
    // landmarks) are snapshotted. Organisms are always re-spawned by GAManager.

    _takeWorldSnapshot() {
        this._world_snapshot = [];
        for (let c = 0; c < this.grid_map.cols; c++) {
            for (let r = 0; r < this.grid_map.rows; r++) {
                const cell = this.grid_map.cellAt(c, r);
                if (!cell) continue;
                const name = cell.state.name;
                // Skip empty and organism-owned cells
                if (name === 'empty') continue;
                if (cell.owner != null) continue;
                this._world_snapshot.push({ c, r, name });
            }
        }
    }

    // Restore the world to the snapshotted layout.
    // Clears everything except walls (preserves permanent structure), then
    // replaces all cells to match the snapshot exactly.

    _restoreWorldSnapshot() {
        if (!this._world_snapshot) {
            // Fallback: regenerate if no snapshot exists
            this.grid_map.fillGrid(CellStates.empty, true);
            this.generateWorld();
            return;
        }

        // Clear all non-wall cells (ignore_walls = true keeps walls in place)
        this.grid_map.fillGrid(CellStates.empty, true);

        // Rebuild from snapshot
        const stateMap = {
            'food':                    CellStates.food,
            'low food':                CellStates.lowFood,
            'medium food':             CellStates.mediumFood,
            'prestige food':           CellStates.prestigeFood,
            'wall':                    CellStates.wall,
            'cave':                    CellStates.cave,
            'low food landmark':       CellStates.lowFoodLandmark,
            'medium food landmark':    CellStates.mediumFoodLandmark,
            'prestige food landmark':  CellStates.prestigeFoodLandmark,
        };

        for (const entry of this._world_snapshot) {
            const state = stateMap[entry.name];
            if (state) {
                this.changeCell(entry.c, entry.r, state, null);
            }
        }
    }

    // Restore world snapshot but only respawn food cells with a given probability.
    // Landmarks and caves are always restored. Walls are never touched (already excluded).
    // This limits resources while keeping structure intact.
    //
    // @param {number} food_respawn_prob - probability (0-1) that each food cell respawns
    _restoreWorldSnapshotWithProbability(food_respawn_prob) {
        if (!this._world_snapshot) {
            this.grid_map.fillGrid(CellStates.empty, true);
            this.generateWorld();
            return;
        }

        // Clear all non-wall cells
        this.grid_map.fillGrid(CellStates.empty, true);

        const stateMap = {
            'food':                    CellStates.food,
            'low food':                CellStates.lowFood,
            'medium food':             CellStates.mediumFood,
            'prestige food':           CellStates.prestigeFood,
            'wall':                    CellStates.wall,
            'cave':                    CellStates.cave,
            'low food landmark':       CellStates.lowFoodLandmark,
            'medium food landmark':    CellStates.mediumFoodLandmark,
            'prestige food landmark':  CellStates.prestigeFoodLandmark,
        };

        for (const entry of this._world_snapshot) {
            const state = stateMap[entry.name];
            if (!state) continue;

            // For food types, roll the dice; always restore landmarks and caves
            const name = entry.name;
            const is_food = name.includes('food') && !name.includes('landmark');
            if (is_food && Math.random() > food_respawn_prob) {
                continue; // Skip this food cell
            }

            this.changeCell(entry.c, entry.r, state, null);
        }
    }

    addOrganism(organism) {
        organism.updateGrid();
        this.total_mutability += organism.mutability;
        this.organisms.push(organism);
        if (organism.anatomy.cells.length > this.largest_cell_count)
            this.largest_cell_count = organism.anatomy.cells.length;
    }

    canAddOrganism() {
        return this.organisms.length < Hyperparams.maxOrganisms || Hyperparams.maxOrganisms < 0;
    }

    averageMutability() {
        if (this.organisms.length < 1) return 0;
        if (Hyperparams.useGlobalMutability) return Hyperparams.globalMutability;
        return this.total_mutability / this.organisms.length;
    }

    changeCell(c, r, state, owner) {
        super.changeCell(c, r, state, owner);
        this.renderer.addToRender(this.grid_map.cellAt(c, r));
        if (state == CellStates.wall)
            this.walls.push(this.grid_map.cellAt(c, r));
    }

    clearWalls() {
        for (var wall of this.walls) {
            let wcell = this.grid_map.cellAt(wall.col, wall.row);
            if (wcell && wcell.state == CellStates.wall)
                this.changeCell(wall.col, wall.row, CellStates.empty, null);
        }
    }

    clearOrganisms() {
        for (var org of this.organisms) org.die();
        this.organisms = [];
    }

    clearDeadOrganisms(check_update = false) {
        let write_idx = 0;
        const orgs = this.organisms;
        let dead_count = 0;
        let start_pop = orgs.length;
        let newborns = [];

        for (let i = 0; i < start_pop; i++) {
            var org = orgs[i];
            // If check_update is true, we run org.update() and check if it died during update
            if (org.living && (!check_update || org.update())) {
                orgs[write_idx++] = org;
            } else {
                this.total_mutability -= org.mutability;
                dead_count++;
            }
        }
        
        // Recover any organisms added to the end by `addOrganism` during `org.update()`
        if (orgs.length > start_pop) {
            for (let i = start_pop; i < orgs.length; i++) {
                newborns.push(orgs[i]);
            }
        }

        if (dead_count > 0 || newborns.length > 0) {
            // Append newborns cleanly back to the surviving list
            for (let i = 0; i < newborns.length; i++) {
                orgs[write_idx++] = newborns[i];
            }
            
            orgs.length = write_idx; // truncate array in-place
            
            if (this.organisms.length === 0 && start_pop > 0 && !this.ga_manager) {
                if (WorldConfig.auto_pause)
                    $('.pause-button')[0].click();
                else if (WorldConfig.auto_reset) {
                    this.reset_count++;
                    this.reset(false);
                }
            }
        }
    }

    generateFood() {
        var num_food = Math.max(Math.floor(this.grid_map.cols * this.grid_map.rows * Hyperparams.foodDropProb / 50000), 1);
        for (var i = 0; i < num_food; i++) {
            if (Math.random() <= Hyperparams.foodDropProb) {
                var c = Math.floor(Math.random() * this.grid_map.cols);
                var r = Math.floor(Math.random() * this.grid_map.rows);
                if (this.grid_map.cellAt(c, r).state == CellStates.empty) {
                    this.changeCell(c, r, CellStates.food, null);
                }
            }
        }
    }

    isNight() {
        const time_in_cycle = this.total_ticks % this.cycle_length;
        return time_in_cycle >= this.day_length;
    }

    reset(confirm_reset = true, reset_life = true) {
        if (confirm_reset && !confirm('The current environment will be lost. Proceed?'))
            return false;
        let restart = false;
        if (this.engine.running) {
            this.engine.last_fps = this.engine.fps;
            this.engine.stop();
            restart = true;
        }
        this.organisms = [];
        this.grid_map.fillGrid(CellStates.empty, !WorldConfig.clear_walls_on_reset);
        this.renderer.renderFullGrid(this.grid_map.grid);
        this.total_mutability = 0;
        this.total_ticks = 0;
        this.largest_cell_count = 0;
        // Restart the shuffle schedule from the base mapping alongside the tick
        // clock (no-op when food_shuffle_period is 0). See FoodShuffle.js.
        FoodShuffle.reset(ExperimentParams.food_shuffle_seed);
        FossilRecord.clear_record();
        this.generateWorld();  // generates fresh random layout + takes new snapshot
        if (reset_life) {
            this.ga_manager.spawnGeneration();
        }
        // Reset predators to a fresh fixed-count population on the new
        // layout. A manual reset is the one place where re-spawning (rather
        // than relocating) is appropriate, since the rest of the simulation
        // state (ticks, generation count) is also being reset here.
        this.predator_manager.spawnAll();
        if (restart) this.engine.start(this.engine.last_fps);
        return true;
    }

    resizeGridColRow(cell_size, cols, rows) {
        this.renderer.cell_size = cell_size;
        this.renderer.fillShape(rows * cell_size, cols * cell_size);
        this.grid_map.resize(cols, rows, cell_size);
    }

    resizeFillWindow(cell_size) {
        this.renderer.cell_size = cell_size;
        this.renderer.fillWindow('env');
        this.num_cols = Math.ceil(this.renderer.width / cell_size);
        this.num_rows = Math.ceil(this.renderer.height / cell_size);
        this.grid_map.resize(this.num_cols, this.num_rows, cell_size);
    }

    serialize() {
        this.clearDeadOrganisms();
        let env = SerializeHelper.copyNonObjects(this);
        env.grid = this.grid_map.serialize();
        env.organisms = [];
        for (let org of this.organisms) env.organisms.push(org.serialize());
        env.fossil_record = FossilRecord.serialize();
        env.controls = Hyperparams;
        return env;
    }

    loadRaw(env) {
        this.organisms = [];
        FossilRecord.clear_record();
        let cell_size = env.grid.cell_size ? env.grid.cell_size : this.grid_map.cell_size;
        this.resizeGridColRow(cell_size, env.grid.cols, env.grid.rows);
        this.grid_map.loadRaw(env.grid);
        for (let wall of env.grid.walls) {
            this.walls.push(this.grid_map.cellAt(wall.c, wall.r));
        }
        let species = {};
        for (let name in env.fossil_record.species) {
            let s = new Species(null, null, 0);
            SerializeHelper.overwriteNonObjects(env.fossil_record.species[name], s);
            species[name] = s;
        }
        for (let orgRaw of env.organisms) {
            let org = new Organism(orgRaw.col, orgRaw.row, this);
            org.loadRaw(orgRaw);
            this.addOrganism(org);
            let s = species[orgRaw.species_name];
            if (!s) {
                s = new Species(org.anatomy, null, env.total_ticks);
                species[orgRaw.species_name] = s;
            }
            if (!s.anatomy) {
                s.anatomy = org.anatomy;
                s.calcAnatomyDetails();
            }
            s.name = orgRaw.species_name;
            org.species = s;
        }
        for (let name in species) FossilRecord.addSpeciesObj(species[name]);
        FossilRecord.loadRaw(env.fossil_record);
        SerializeHelper.overwriteNonObjects(env, this);
        if ($('#override-controls').is(':checked'))
            Hyperparams.loadJsonObj(env.controls);
        this.renderer.renderFullGrid(this.grid_map.grid);
    }
}

module.exports = WorldEnvironment;