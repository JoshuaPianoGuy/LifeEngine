const Environment = require('./Environment');
const Renderer = require('../Rendering/Renderer');
const GridMap = require('../Grid/GridMap');
const Organism = require('../Organism/Organism');
const AdvancedOrganism = require('../Organism/AdvancedOrganism');
const GAManager = require('../Organism/GAManager');
const CellStates = require('../Organism/Cell/CellStates');
const EnvironmentController = require('../Controllers/EnvironmentController');
const Hyperparams = require('../Hyperparameters.js');
const FossilRecord = require('../Stats/FossilRecord');
const WorldConfig = require('../WorldConfig');
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
        this.grid_map = new GridMap(this.num_cols, this.num_rows, cell_size);
        this.organisms = [];
        this.walls = [];
        this.total_mutability = 0;
        this.largest_cell_count = 0;
        this.reset_count = 0;
        this.total_ticks = 0;
        this.data_update_rate = 100;

        // Day/night cycle — equal length
        this.day_length   = 150;
        this.night_length = 150;
        this.cycle_length = this.day_length + this.night_length;

        // Snapshot of the initial map layout — restored at the start of each
        // new generation so all generations face identical conditions.
        // Null until generateWorld() is called for the first time.
        this._world_snapshot = null;

        this.learning_enabled = WorldConfig.learning_enabled;
        const center    = this.grid_map.getCenter();
        const spawn_col = center[0] + 15;
        const spawn_row = center[1] + 15;
        this.ga_manager = new GAManager(this, true, spawn_col, spawn_row);

        FossilRecord.setEnv(this);
    }

    update() {
        var to_remove = [];
        for (var i in this.organisms) {
            var org = this.organisms[i];
            if (!org.living || !org.update()) {
                to_remove.push(i);
            }
        }
        this.removeOrganisms(to_remove);
        if (Hyperparams.foodDropProb > 0) {
            this.generateFood();
        }
        this.total_ticks++;
        // Regenerate food every 50,000 ticks with 20% spawn probability
        if (this.total_ticks % 50000 == 0) {
            this._restoreWorldSnapshotWithProbability(0.2);
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

    removeOrganisms(org_indeces) {
        let start_pop = this.organisms.length;
        for (var i of org_indeces.reverse()) {
            this.total_mutability -= this.organisms[i].mutability;
            this.organisms.splice(i, 1);
        }
        if (this.organisms.length === 0 && start_pop > 0) {
            if (this.learning_enabled && this.ga_manager) {
                const gen_ended = this.ga_manager.tick();
                if (gen_ended) {
                    this.ga_manager.evolve();
                    // Restore the fixed map snapshot — food replenished to exactly
                    // the same layout as generation 1 so all generations are comparable.
                    this._restoreWorldSnapshot();
                    this.renderFull();
                    this.ga_manager.spawnGeneration();
                    return;
                }
            }
            if (WorldConfig.auto_pause)
                $('.pause-button')[0].click();
            else if (WorldConfig.auto_reset) {
                this.reset_count++;
                this.reset(false);
            }
        }
    }

    OriginOfLife() {
        var center = this.grid_map.getCenter();
        var org = this.createExperimentOrganism(center[0], center[1], null);
        this.addOrganism(org);
        if (this.learning_enabled && this.ga_manager) {
            this.ga_manager.registerAgent(org);
            FossilRecord.addSpecies(org, null);
        } else {
            FossilRecord.addSpecies(org, null);
        }
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

        // ── Low food: 8–10 distinct separated patches, mid-range ───────────────
        // Increased count and size for larger world. Small sigma (15) keeps tight.
        // min_sep=80 ensures genuine empty corridors between patches.
        // Each patch has one diagonal landmark line placed 12–20 cells closer
        // to spawn, angled toward the patch.
        const low_positions = [];
        const low_patch_count = 8 + Math.floor(rng() * 3);
        for (let i = 0; i < low_patch_count; i++) {
            const pos = pickPosition(80, 150, placed_centres, 80);
            if (!pos) continue;
            placed_centres.push(pos);
            low_positions.push(pos);

            const patch_size = 130 + Math.floor(rng() * 80);  // 130–210 cells
            this._placeBlob(pos.c, pos.r, 15, CellStates.lowFood, patch_size, rng);

            // Landmark line: midpoint between spawn and patch, pointing at patch
            const dx    = pos.c - cx;
            const dy    = pos.r - cy;
            const dist  = Math.hypot(dx, dy);
            const frac  = (0.45 + rng() * 0.2);  // 45–65% of the way
            const lm_c  = Math.round(cx + dx * frac);
            const lm_r  = Math.round(cy + dy * frac);
            const angle = Math.atan2(dy, dx) + (rng() - 0.5) * 0.5;
            const len   = 18 + Math.floor(rng() * 14);
            this._placeLine(lm_c, lm_r, angle, len, CellStates.lowFoodLandmark);
        }

        // ── Prestige food: 6–8 tight patches, outer ring ────────────────
        // Moderate sigma (12), scaled patch sizes for larger world.
        // min_sep=100 from everything so they're clearly isolated.
        const prestige_patch_count = 6 + Math.floor(rng() * 3);
        for (let i = 0; i < prestige_patch_count; i++) {
            const pos = pickPosition(140, 220, placed_centres, 100);
            if (!pos) continue;
            placed_centres.push(pos);

            const patch_size = 60 + Math.floor(rng() * 40);  // 60–100 cells
            this._placeBlob(pos.c, pos.r, 12, CellStates.prestigeFood, patch_size, rng);

            // Landmark line: placed at 50–70% of distance from spawn to patch
            const dx   = pos.c - cx;
            const dy   = pos.r - cy;
            const frac = 0.50 + rng() * 0.20;
            const lm_c = Math.round(cx + dx * frac);
            const lm_r = Math.round(cy + dy * frac);
            const angle = Math.atan2(dy, dx) + (rng() - 0.5) * 0.4;
            const len   = 20 + Math.floor(rng() * 16);
            this._placeLine(lm_c, lm_r, angle, len, CellStates.prestigeFoodLandmark);
        }

        // ── Medium food: scattered patches, gap-filling ─────────────────
        // 7–10 medium-sized patches at mid-range (not outer), each well-separated.
        // These fill the navigational space between low and prestige food zones
        // so there's always a reward signal for exploring outward.
        // Each patch has a landmark line placed 40–60% of distance from spawn.
        const med_patch_count = 7 + Math.floor(rng() * 4);
        for (let i = 0; i < med_patch_count; i++) {
            const pos = pickPosition(70, 140, placed_centres, 70);
            if (!pos) continue;
            placed_centres.push(pos);

            const patch_size = 90 + Math.floor(rng() * 60);  // 90–150 cells
            this._placeBlob(pos.c, pos.r, 14, CellStates.mediumFood, patch_size, rng);

            // Landmark line: placed at 40–60% of distance from spawn to patch
            const dx   = pos.c - cx;
            const dy   = pos.r - cy;
            const frac = 0.40 + rng() * 0.20;
            const lm_c = Math.round(cx + dx * frac);
            const lm_r = Math.round(cy + dy * frac);
            const angle = Math.atan2(dy, dx) + (rng() - 0.5) * 0.4;
            const len   = 16 + Math.floor(rng() * 14);
            this._placeLine(lm_c, lm_r, angle, len, CellStates.mediumFoodLandmark);
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

        // ── Obstacles: wall rectangles, like reference image ──────────────────
        // 10–15 wall blocks placed at mid-range, well away from food patches.
        // Centred on their position like caves.
        const obstacle_count = 10 + Math.floor(rng() * 6);
        for (let i = 0; i < obstacle_count; i++) {
            const pos = pickPosition(50, 180, placed_centres, 50);
            if (!pos) continue;
            placed_centres.push(pos);
            const w = 8 + Math.floor(rng() * 10);
            const h = 8 + Math.floor(rng() * 10);
            this._placeRect(pos.c - Math.floor(w/2), pos.r - Math.floor(h/2), w, h, CellStates.wall);
        }

        // Take a snapshot so all subsequent generations restore this exact layout.
        this._takeWorldSnapshot();
    }

    // Place a food patch using 2D Gaussian spread — organic, irregular shape.
    // Keep sigma small (5–8) for distinct patches, larger (12+) for diffuse areas.
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

    clearDeadOrganisms() {
        let to_remove = [];
        for (let i in this.organisms) {
            if (!this.organisms[i].living) to_remove.push(i);
        }
        this.removeOrganisms(to_remove);
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
        FossilRecord.clear_record();
        this.generateWorld();  // generates fresh random layout + takes new snapshot
        if (reset_life) {
            this.ga_manager.spawnGeneration();
        }
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