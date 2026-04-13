const WorldEnvironment = require('./Environments/WorldEnvironment');
const ControlPanel = require('./Controllers/ControlPanel');
const OrganismEditor = require('./Environments/OrganismEditor');
const {ColorScheme} = require('./Rendering/ColorScheme');
const WorldConfig = require('./WorldConfig');

const MIN_TIMER_MS = 1;
let SAFE_STEPS_PER_TICK = 5;// soft upper limit, can be bypassed when we have extra time
const MAX_STEPS_PER_TICK = 100; // hard upper limit to prevent stalling the UI

class Engine {
    constructor(){
        this.fps = 60;
        this.env = new WorldEnvironment(this, 5);
        this.organism_editor = new OrganismEditor();
        this.organism_editor.engine = this;
        this.controlpanel = new ControlPanel(this);
        ColorScheme.setEnvironment(this.env, this.organism_editor);
        ColorScheme.loadColorScheme();
        
        // Generate initial world with food, obstacles, landmarks, caves
        this.env.generateWorld();
        
        // Initialize founding population using GA manager (always used for generation tracking)
        // Do this AFTER ColorScheme.loadColorScheme() so rendering is properly initialized
        this.env.ga_manager.spawnGeneration();
        
        // Render the organisms on the canvas
        this.env.renderFull();
        
        this.sim_last_update = Date.now();
        this.sim_delta_time = 0;

        this.ui_last_update = Date.now();
        this.ui_delta_time = 0;

        this.actual_fps = 0; // steps simulated in the last 1000 ms
        this.step_timestamps = []; // rolling window of timestamps
        this.step_head = 0; // index of oldest timestamp within the 1-second window
        this.running = false;

        // fixed-timestep variables
        this.step_ms   = 1000 / this.fps;   // duration of one sim step
        this.accum_ms  = 0;                 // un-simulated time
 
        this.setUiLoop();
    }

    start(fps = 60) {
        if (fps <= 0) fps = 1;
        this.fps      = fps;
        this.step_ms  = 1000 / this.fps;
        this.accum_ms = 0;
        this.sim_last_update = Date.now();

        this.step_timestamps.length = 0;
        this.step_head = 0;

        if (this.sim_loop) clearInterval(this.sim_loop);

        const interval_ms = Math.max(this.step_ms, MIN_TIMER_MS);

        this.sim_loop = setInterval(() => {
            let now     = Date.now();
            let elapsed = now - this.sim_last_update;
            this.sim_last_update = now;

            this.accum_ms += elapsed;
            let steps = 0;

            // Pack many sim steps between timeouts
            let start_time = Date.now();
            while (this.accum_ms >= this.step_ms) {
                this.environmentUpdate();
                this.accum_ms -= this.step_ms;
                steps++;

                // FPS calculation
                // Maintain running list of step timestamps in the past second
                // Use a rolling window rather than a queue to avoid expensive shift operations
                // Steps before step_head are outdated and are occasionally purged
                const ts = Date.now();
                this.step_timestamps.push(ts);
                while (this.step_head < this.step_timestamps.length && this.step_timestamps[this.step_head] < ts - 1000) {
                    this.step_head++;
                }
                this.actual_fps = this.step_timestamps.length - this.step_head;
                // trim when outdated region exceeds current target FPS (or a minimum of 10000)
                const purge_threshold = Math.max(this.fps, 10000);
                if (this.step_head > purge_threshold) {
                    this.step_timestamps = this.step_timestamps.slice(this.step_head);
                    this.step_head = 0;
                }

                // the below failsafes are based on experimentation
                // they try to catch cases that would cause the sim to stall, while also packing in as many updates as possible

                // at low actual fps, don't bother packing updates
                if ((Date.now() - start_time) / steps > 1000 / 60) {
                    this.accum_ms = 0;
                    break;
                }
                // otherwise we can try packing more steps, but only if we're not stalling
                if (steps > SAFE_STEPS_PER_TICK) {
                    // if steps have taken less time than expected, we can afford to take more steps
                    if (Date.now() - start_time < this.step_ms*steps && steps < MAX_STEPS_PER_TICK)
                        continue;

                    // otherwise we are stalling, so stop the loop
                    this.accum_ms = 0; // drop excess time debt
                    break;
                }
            }
        }, interval_ms);

        this.running = true;
    }
    
    stop() {
        clearInterval(this.sim_loop);
        this.sim_loop = null;
        this.running = false;
    }

    restart(fps) {
        clearInterval(this.sim_loop);
        this.sim_loop = null;
        this.start(fps);
    }

    setUiLoop() {
        if (!this.ui_loop) {
            this.ui_loop = setInterval(() => {
                this.updateUIDeltaTime();
                this.necessaryUpdate();
            }, 1000 / 60);
        }
    }

    updateUIDeltaTime() {
        this.ui_delta_time = Date.now() - this.ui_last_update;
        this.ui_last_update = Date.now();
    }

    environmentUpdate() {
        this.env.update(this.step_ms); // WorldEnvironment ignores the arg, but it’s harmless
    }

    necessaryUpdate() {
        this.env.render();
        this.controlpanel.update(this.ui_delta_time);
        this.organism_editor.update();
    }

    setSafeStepsPerTick(val) {
        SAFE_STEPS_PER_TICK = val;
    }
}

module.exports = Engine;
