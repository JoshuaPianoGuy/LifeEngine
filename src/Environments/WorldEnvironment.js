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

class WorldEnvironment extends Environment{
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

        // ── Day/night cycle ────────────────────────────────────────────────────
        // Day: 150 ticks, Night: 75 ticks (225 ticks per full cycle)
        // Organisms use isNight() to access current time of day
        this.day_length = 150;
        this.night_length = 75;
        this.cycle_length = this.day_length + this.night_length;  // 225

        // ── Genetic Algorithm (always used for generational evolution) ────────────
        // Whether learning is enabled or not, we need the GA for genetic diversity
        // and generation tracking. RL is controlled by the learning_enabled flag.
        this.learning_enabled = WorldConfig.learning_enabled;
        const center = this.grid_map.getCenter();
        // Spawn offset from center to avoid being trapped in food
        const spawn_col = center[0] + 15;  // offset right
        const spawn_row = center[1] + 15;  // offset down
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
        this.total_ticks ++;
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
        for (var i of org_indeces.reverse()){
            this.total_mutability -= this.organisms[i].mutability;
            this.organisms.splice(i, 1);
        }
        if (this.organisms.length === 0 && start_pop > 0) {
            // Check if this generation has ended (for genetic algorithm)
            if (this.learning_enabled && this.ga_manager) {
                const gen_ended = this.ga_manager.tick();
                if (gen_ended) {
                    // Lineage died — evolve to next generation
                    this.ga_manager.evolve();
                    
                    // Clear old food and regenerate world with fresh food layout
                    this.grid_map.fillGrid(CellStates.empty, true);
                    this.generateWorld();
                    this.renderFull();
                    
                    this.ga_manager.spawnGeneration();
                    return;  // New generation spawned; don't auto_pause or reset
                }
            }

            // Standard behavior (no GA or within same generation)
            if (WorldConfig.auto_pause)
                $('.pause-button')[0].click();
            else if(WorldConfig.auto_reset) {
                this.reset_count++;
                this.reset(false);
            }
        }
    }

    OriginOfLife() {
        var center = this.grid_map.getCenter();
        // Always use experiment organism (4 eyes + mover + mouth)
        var org = this.createExperimentOrganism(center[0], center[1], null);
        this.addOrganism(org);
        
        if (this.learning_enabled && this.ga_manager) {
            this.ga_manager.registerAgent(org);
            FossilRecord.addSpecies(org, null);
        } else {
            FossilRecord.addSpecies(org, null);
        }
    }

    // ── Experiment organism with proper anatomy ────────────────────────────────
    // Creates an organism with:
    //   - Center: mover + mouth
    //   - 4 directional eyes (up, right, down, left)
    // This is the standard setup for the learning vs evolution experiment.

    createExperimentOrganism(col, row, parent = null) {
        // Always use AdvancedOrganism (has integrated NNBrain)
        // Whether learning is enabled or not is a runtime flag, not an architecture choice
        // ga_manager is always passed for generational tracking and spawning
        const org = new AdvancedOrganism(col, row, this, parent, this.learning_enabled, this.ga_manager);

        // Center (0, 0): Mouth
        org.anatomy.addDefaultCell(CellStates.mouth, 0, 0);

        // above mouth at (0, -1): Eye pointing up
        const eye_up = org.anatomy.addDefaultCell(CellStates.eye, 0, -1);
        if (eye_up) eye_up.direction = 0;  // Directions.up

        // Right of mouth at (1, 0): Eye pointing right
        const eye_right = org.anatomy.addDefaultCell(CellStates.eye, 1, 0);
        if (eye_right) eye_right.direction = 1;  // Directions.right

        // Below mouth at (0, 1): Mover cell
        org.anatomy.addDefaultCell(CellStates.mover, 0, 1);

        // Left of mover at (-1, 1): Eye pointing left
        const eye_left = org.anatomy.addDefaultCell(CellStates.eye, -1, 1);
        if (eye_left) eye_left.direction = 3;  // Directions.left

        // Below mover at (0, 2): Eye pointing down
        const eye_down = org.anatomy.addDefaultCell(CellStates.eye, 0, 2);
        if (eye_down) eye_down.direction = 2;  // Directions.down

        // Update anatomy type flags (is_mover, has_eyes, is_producer)
        org.anatomy.checkTypeChange();

        return org;
    }

    // ── World generation with hardcoded food and obstacles ─────────────────────
    // Hardcoded world layout with concentric food rings:
    // - Blue (medium food) in center
    // - Red (low food) in middle ring
    // - Orange and Teal scattered in outer regions
    // Landmarks point toward each food type

    generateWorld() {
        const cols = this.grid_map.cols;
        const rows = this.grid_map.rows;
        const center_c = cols / 2;
        const center_r = rows / 2;

        // ── Layer 1: Red (Low food) ABUNDANT at spawn location ────────────────
        // Dense cluster around center for organisms to survive and learn
        // Radius: 30 cells from center with HIGH density (300 cells)
        this._drawFoodRing(center_c, center_r, 0, 30, CellStates.lowFood, 300);
        this._drawLandmarkRing(center_c, center_r, 35, 45, CellStates.lowFoodLandmark, 12);

        // ── Layer 2: Orange (Low food) scattered across entire map ─────────────
        // 12 clusters distributed evenly (was 8), each with 35 cells (was 40)
        const orange_positions = [
            { c: center_c - 70, r: center_c - 70 },  // top-left
            { c: center_c + 70, r: center_r - 70 },  // top-right
            { c: center_c - 70, r: center_r + 70 },  // bottom-left
            { c: center_c + 70, r: center_r + 70 },  // bottom-right
            { c: center_c,      r: center_r - 80 },  // top center
            { c: center_c,      r: center_r + 80 },  // bottom center
            { c: center_c - 80, r: center_r },       // left center
            { c: center_c + 80, r: center_r },       // right center
            { c: center_c - 50, r: center_r - 50 },  // top-left diagonal
            { c: center_c + 50, r: center_r - 50 },  // top-right diagonal
            { c: center_c - 50, r: center_r + 50 },  // bottom-left diagonal
            { c: center_c + 50, r: center_r + 50 },  // bottom-right diagonal
        ];
        for (const pos of orange_positions) {
            const c = Math.max(10, Math.min(cols - 10, pos.c));
            const r = Math.max(10, Math.min(rows - 10, pos.r));
            this._placeCluster(c, r, 25, CellStates.lowFood, 35);
        }

        // ── Layer 3: Teal (Prestige food) scattered and rare ────────────────
        // 6 clusters in outlying regions (increased from 5)
        const teal_positions = [
            { c: center_c - 60, r: center_r },      // left
            { c: center_c + 60, r: center_r },      // right
            { c: center_c,      r: center_r - 60 }, // top center
            { c: center_c,      r: center_r + 60 }, // bottom center
            { c: center_c - 55, r: center_r - 55 }, // top-left
            { c: center_c + 55, r: center_r + 55 }, // bottom-right
        ];
        for (const pos of teal_positions) {
            const c = Math.max(10, Math.min(cols - 10, pos.c));
            const r = Math.max(10, Math.min(rows - 10, pos.r));
            this._placeCluster(c, r, 20, CellStates.prestigeFood, 15);
            this._drawLandmarkRing(c, r, 23, 32, CellStates.prestigeFoodLandmark, 5);
        }

        // ── Caves: strategic locations for night refuge ────────────────────────
        const cave_positions = [
            { c: center_c,           r: center_r },           // center
            { c: center_c - 40,      r: center_r - 40 },     // top-left
            { c: center_c + 40,      r: center_r - 40 },     // top-right
            { c: center_c - 40,      r: center_r + 40 },     // bottom-left
            { c: center_c + 40,      r: center_r + 40 },     // bottom-right
        ];
        for (const pos of cave_positions) {
            const c = Math.max(10, Math.min(cols - 10, pos.c));
            const r = Math.max(10, Math.min(rows - 10, pos.r));
            this._placeCluster(c, r, 10, CellStates.cave, 6);
        }

        // ── Light obstacles: scattered walls for navigation ────────────────────
        // REMOVED for now to simplify debugging — focus on food eating first
    }

    // Draw food in a ring (annulus) at given distance from center
    _drawFoodRing(cx, cy, min_radius, max_radius, food_state, count) {
        let placed = 0;
        for (let i = 0; i < count * 2 && placed < count; i++) {
            const angle = Math.random() * 2 * Math.PI;
            const radius = min_radius + Math.random() * (max_radius - min_radius);
            const c = Math.floor(cx + radius * Math.cos(angle));
            const r = Math.floor(cy + radius * Math.sin(angle));

            const cell = this.grid_map.cellAt(c, r);
            if (cell && cell.state === CellStates.empty) {
                this.changeCell(c, r, food_state, null);
                placed++;
            }
        }
    }

    // Draw landmarks in a ring
    _drawLandmarkRing(cx, cy, min_radius, max_radius, landmark_state, count) {
        let placed = 0;
        for (let i = 0; i < count * 2 && placed < count; i++) {
            const angle = Math.random() * 2 * Math.PI;
            const radius = min_radius + Math.random() * (max_radius - min_radius);
            const c = Math.floor(cx + radius * Math.cos(angle));
            const r = Math.floor(cy + radius * Math.sin(angle));

            const cell = this.grid_map.cellAt(c, r);
            if (cell && cell.state === CellStates.empty) {
                this.changeCell(c, r, landmark_state, null);
                placed++;
            }
        }
    }

    // Helper: place random walls (obstacles)
    _placeWalls(cols, rows, probability) {
        for (let c = 0; c < cols; c++) {
            for (let r = 0; r < rows; r++) {
                if (Math.random() < probability) {
                    const cell = this.grid_map.cellAt(c, r);
                    if (cell && cell.state === CellStates.empty) {
                        this.changeCell(c, r, CellStates.wall, null);
                    }
                }
            }
        }
    }

    // Helper: place a cluster of cells around a center point
    _placeCluster(center_c, center_r, radius, cell_state, count) {
        let placed = 0;
        for (let i = 0; i < count * 2 && placed < count; i++) {
            const angle = Math.random() * 2 * Math.PI;
            const dist = Math.random() * radius;
            const c = Math.floor(center_c + dist * Math.cos(angle));
            const r = Math.floor(center_r + dist * Math.sin(angle));

            const cell = this.grid_map.cellAt(c, r);
            if (cell && cell.state === CellStates.empty) {
                this.changeCell(c, r, cell_state, null);
                placed++;
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
        if (this.organisms.length < 1)
            return 0;
        if (Hyperparams.useGlobalMutability) {
            return Hyperparams.globalMutability;
        }
        return this.total_mutability / this.organisms.length;
    }

    changeCell(c, r, state, owner) {
        super.changeCell(c, r, state, owner);
        this.renderer.addToRender(this.grid_map.cellAt(c, r));
        if(state == CellStates.wall)
            this.walls.push(this.grid_map.cellAt(c, r));
    }

    clearWalls() {
        for(var wall of this.walls){
            let wcell = this.grid_map.cellAt(wall.col, wall.row);
            if (wcell && wcell.state == CellStates.wall)
                this.changeCell(wall.col, wall.row, CellStates.empty, null);
        }
    }

    clearOrganisms() {
        for (var org of this.organisms)
            org.die();
        this.organisms = [];
    }
    
    clearDeadOrganisms() {
        let to_remove = [];
        for (let i in this.organisms) {
            let org = this.organisms[i];
            if (!org.living)
                to_remove.push(i);
        }
        this.removeOrganisms(to_remove);
    }

    generateFood() {
        var num_food = Math.max(Math.floor(this.grid_map.cols*this.grid_map.rows*Hyperparams.foodDropProb/50000), 1)
        var prob = Hyperparams.foodDropProb;
        for (var i=0; i<num_food; i++) {
            if (Math.random() <= prob){
                var c=Math.floor(Math.random() * this.grid_map.cols);
                var r=Math.floor(Math.random() * this.grid_map.rows);

                if (this.grid_map.cellAt(c, r).state == CellStates.empty){
                    this.changeCell(c, r, CellStates.food, null);
                }
            }
        }
    }

    // ── Day/night cycle ───────────────────────────────────────────────────────
    // Returns true if it's currently night time (second half of cycle)

    isNight() {
        const time_in_cycle = this.total_ticks % this.cycle_length;
        return time_in_cycle >= this.day_length;
    }

    reset(confirm_reset=true, reset_life=true) {
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
        
        // Generate world with food, obstacles, landmarks, caves
        this.generateWorld();
        
        if (reset_life) {
            // Always use GA manager for spawning (controls generation tracking and diversity)
            this.ga_manager.spawnGeneration();
        }
        if (restart)
            this.engine.start(this.engine.last_fps);
        return true;
    }

    resizeGridColRow(cell_size, cols, rows) {
        this.renderer.cell_size = cell_size;
        this.renderer.fillShape(rows*cell_size, cols*cell_size);
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
        for (let org of this.organisms){
            env.organisms.push(org.serialize());
        }
        env.fossil_record = FossilRecord.serialize();
        env.controls = Hyperparams;
        return env;
    }

    loadRaw(env) { // species name->stats map, evolution controls, 
        this.organisms = [];
        FossilRecord.clear_record();
        let cell_size = env.grid.cell_size ? env.grid.cell_size : this.grid_map.cell_size;
        this.resizeGridColRow(cell_size, env.grid.cols, env.grid.rows)
        this.grid_map.loadRaw(env.grid);
        for (let wall of env.grid.walls) {
            this.walls.push(this.grid_map.cellAt(wall.c, wall.r));
        }

        // create species map
        let species = {};
        for (let name in env.fossil_record.species) {
            let s = new Species(null, null, 0);
            SerializeHelper.overwriteNonObjects(env.fossil_record.species[name], s)
            species[name] = s; // the species needs an anatomy obj still
        }

        for (let orgRaw of env.organisms) {
            let org = new Organism(orgRaw.col, orgRaw.row, this);
            org.loadRaw(orgRaw);
            this.addOrganism(org);
            let s = species[orgRaw.species_name];
            if (!s){ // ideally, every organisms species should exists, but there is a bug that misses some species sometimes
                s = new Species(org.anatomy, null, env.total_ticks);
                species[orgRaw.species_name] = s;
            }
            if (!s.anatomy) {
                //if the species doesn't have anatomy we need to initialize it
                s.anatomy = org.anatomy;
                s.calcAnatomyDetails();
            }
            s.name = orgRaw.species_name;
            org.species = s;
        }
        for (let name in species)
            FossilRecord.addSpeciesObj(species[name]);
        FossilRecord.loadRaw(env.fossil_record);
        SerializeHelper.overwriteNonObjects(env, this);
        if ($('#override-controls').is(':checked'))
            Hyperparams.loadJsonObj(env.controls)
        this.renderer.renderFullGrid(this.grid_map.grid);
    }
}

module.exports = WorldEnvironment;

