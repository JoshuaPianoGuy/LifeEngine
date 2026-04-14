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

        // Day/night cycle
        this.day_length   = 150;
        this.night_length = 75;
        this.cycle_length = this.day_length + this.night_length;

        // Snapshot of the initial map layout — restored at the start of each
        // new generation so all generations face identical conditions.
        // Null until generateWorld() is called for the first time.
        this._world_snapshot = null;

        this.learning_enabled = WorldConfig.learning_enabled;
        const center    = this.grid_map.getCenter();
        const spawn_col = center[0];
        const spawn_row = center[1];
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

        // Mover in the middle
        org.anatomy.addDefaultCell(CellStates.mover, 0, 0);

        // Two mouth cells stacked on each edge
        // Up edge
        org.anatomy.addDefaultCell(CellStates.mouth, 0, -1);
        org.anatomy.addDefaultCell(CellStates.mouth, 0, -2);

        // Right edge
        org.anatomy.addDefaultCell(CellStates.mouth, 1, 0);
        org.anatomy.addDefaultCell(CellStates.mouth, 2, 0);

        // Down edge
        org.anatomy.addDefaultCell(CellStates.mouth, 0, 1);
        org.anatomy.addDefaultCell(CellStates.mouth, 0, 2);

        // Left edge
        org.anatomy.addDefaultCell(CellStates.mouth, -1, 0);
        org.anatomy.addDefaultCell(CellStates.mouth, -2, 0);

        // Eyes on the four corners between mouth cells
        const eye_top_left = org.anatomy.addDefaultCell(CellStates.eye, -1, -1);
        if (eye_top_left) eye_top_left.direction = 3;  // left direction

        const eye_top_right = org.anatomy.addDefaultCell(CellStates.eye, 1, -1);
        if (eye_top_right) eye_top_right.direction = 0;  // up direction

        const eye_bottom_right = org.anatomy.addDefaultCell(CellStates.eye, 1, 1);
        if (eye_bottom_right) eye_bottom_right.direction = 1;  // right direction

        const eye_bottom_left = org.anatomy.addDefaultCell(CellStates.eye, -1, 1);
        if (eye_bottom_left) eye_bottom_left.direction = 2;  // down direction

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
        const cols     = this.grid_map.cols;
        const rows     = this.grid_map.rows;
        const cx       = Math.floor(cols / 2);
        const cy       = Math.floor(rows / 2);
        const rng      = () => Math.random(); //FIX THIS; HARDCODE FOR NOW

        // ── Spawn-area food: low density default + sparse low food ─────────────
        // Default food (near-zero score) gives organisms a survival floor near
        // spawn so they don't starve before finding anything meaningful.
        // Low food is also placed here at low density as a gentle first reward.
        this._scatterFood(cx, cy,  0, 20, CellStates.food,    80,  rng);   // default food, dense near centre
        this._scatterFood(cx, cy, 10, 30, CellStates.lowFood, 40,  rng);   // low food, sparse inner ring

        // ── Mid-range: low food clusters + landmarks ───────────────────────────
        // 8 organic clusters at radius 35–65, each 25–45 cells, with landmark
        // trails pointing back toward them from slightly further out.
        const mid_angles = [0, 45, 90, 135, 180, 225, 270, 315].map(d => d * Math.PI / 180);
        for (const angle of mid_angles) {
            const r   = 40 + rng() * 25;          // radius 40–65
            const jit = (rng() - 0.5) * 15;       // ±7.5° jitter
            const ac  = angle + jit * Math.PI / 180;
            const cc  = Math.round(cx + r * Math.cos(ac));
            const cr  = Math.round(cy + r * Math.sin(ac));
            const sz  = 25 + Math.floor(rng() * 20);  // cluster size 25–45

            this._placeCluster(cc, cr, 14, CellStates.lowFood, sz, rng);

            // Landmark trail: 4–6 cells at radius+10 pointing back to this cluster
            const lm_r = r + 10 + rng() * 8;
            const lm_c = Math.round(cx + lm_r * Math.cos(ac));
            const lm_r2 = Math.round(cy + lm_r * Math.sin(ac));
            this._placeCluster(lm_c, lm_r2, 5, CellStates.lowFoodLandmark, 5, rng);
        }

        // ── Outer ring: prestige food clusters ────────────────────────────────
        // 6 clusters at radius 70–95, small (10–18 cells each), rare and valuable.
        // Each cluster has a landmark halo 12–18 cells outward so agents that
        // wander far can learn to recognise "prestige zone ahead".
        const prestige_count = 6;
        for (let i = 0; i < prestige_count; i++) {
            const angle = (i / prestige_count) * 2 * Math.PI + (rng() - 0.5) * 0.4;
            const r     = 70 + rng() * 25;
            const cc    = Math.round(cx + r * Math.cos(angle));
            const cr    = Math.round(cy + r * Math.sin(angle));
            const sz    = 10 + Math.floor(rng() * 9);

            this._placeCluster(cc, cr, 10, CellStates.prestigeFood, sz, rng);

            // Landmark ring slightly beyond the cluster
            const halo_count = 6 + Math.floor(rng() * 4);
            for (let h = 0; h < halo_count; h++) {
                const ha  = angle + (rng() - 0.5) * 1.2;
                const hr  = r + 12 + rng() * 8;
                const hc  = Math.round(cx + hr * Math.cos(ha));
                const hrr = Math.round(cy + hr * Math.sin(ha));
                const cell = this.grid_map.cellAt(hc, hrr);
                if (cell && cell.state === CellStates.empty) {
                    this.changeCell(hc, hrr, CellStates.prestigeFoodLandmark, null);
                }
            }
        }

        // ── Medium food: scattered across map as intermediate reward ───────────
        // Not clustered — individual cells dotted everywhere so there's always
        // something to find when wandering between clusters.
        const medium_count = 120;
        let placed = 0;
        for (let attempt = 0; attempt < medium_count * 4 && placed < medium_count; attempt++) {
            const c = 5 + Math.floor(rng() * (cols - 10));
            const r = 5 + Math.floor(rng() * (rows - 10));
            // Prefer mid-range distances (30–80) from centre
            const dist = Math.hypot(c - cx, r - cy);
            if (dist < 25 || dist > 90) continue;
            const cell = this.grid_map.cellAt(c, r);
            if (cell && cell.state === CellStates.empty) {
                this.changeCell(c, r, CellStates.mediumFood, null);
                placed++;
            }
        }

        // ── Caves: clumps of 12–20 cells ──────────────────────────────────────
        // Placed at navigable distances — not too close to spawn (agents would
        // just hide immediately) and not at the very edge.
        // 7 caves total: 2 near-mid, 3 mid, 2 outer.
        const cave_specs = [
            { min_r: 25, max_r: 40,  count: 2 },  // near-mid
            { min_r: 45, max_r: 65,  count: 3 },  // mid
            { min_r: 70, max_r: 90,  count: 2 },  // outer
        ];
        for (const spec of cave_specs) {
            for (let i = 0; i < spec.count; i++) {
                const angle = rng() * 2 * Math.PI;
                const r     = spec.min_r + rng() * (spec.max_r - spec.min_r);
                const cc    = Math.round(cx + r * Math.cos(angle));
                const cr    = Math.round(cy + r * Math.sin(angle));
                const sz    = 12 + Math.floor(rng() * 9);  // 12–20 cells per cave
                this._placeCluster(cc, cr, 5, CellStates.cave, sz, rng);
            }
        }

        // ── Obstacles: short wall segments ────────────────────────────────────
        // 12 small clusters of walls (3–6 cells each) to break up navigation
        // without creating impassable barriers.
        for (let i = 0; i < 12; i++) {
            const angle = rng() * 2 * Math.PI;
            const r     = 30 + rng() * 55;
            const cc    = Math.round(cx + r * Math.cos(angle));
            const cr    = Math.round(cy + r * Math.sin(angle));
            const sz    = 3 + Math.floor(rng() * 4);
            this._placeCluster(cc, cr, 4, CellStates.wall, sz, rng);
        }

        // Take a snapshot of this layout so all subsequent generations can
        // restore it exactly. Called once per run (or after a manual reset).
        this._takeWorldSnapshot();
    }

    // Scatter food randomly within an annulus [min_r, max_r] around (cx, cy)
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

    // Place a cluster of cells using 2D Gaussian spread around (cx, cy)
    // sigma controls how tightly packed the cluster is
    _placeCluster(cx, cy, sigma, state, count, rng) {
        let placed = 0;
        for (let attempt = 0; attempt < count * 6 && placed < count; attempt++) {
            // Box-Muller for Gaussian offset
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

    // Legacy helpers kept for backward compatibility
    _drawFoodRing(cx, cy, min_radius, max_radius, food_state, count) {
        this._scatterFood(cx, cy, min_radius, max_radius, food_state, count, Math.random.bind(Math));
    }
    _drawLandmarkRing(cx, cy, min_radius, max_radius, landmark_state, count) {
        this._scatterFood(cx, cy, min_radius, max_radius, landmark_state, count, Math.random.bind(Math));
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