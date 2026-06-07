'use strict';

const AdvancedOrganism = require('./AdvancedOrganism');
const NNBrain = require('./Perception/NNBrain');
const Directions = require('./Directions');
const Hyperparams = require('../Hyperparameters');

const REPRODUCTION_SUCCESS_RATE = 0.8;

class FrozenPolicyOrganism extends AdvancedOrganism {
    constructor(col, row, env, parent = null, rl_enabled = true, ga_manager = null) {
        super(col, row, env, parent, rl_enabled, ga_manager);
        if (this.brain && this.brain instanceof NNBrain) {
            this.brain.freeze_updates = true;
        }
    }

    // Override: clone parent genome exactly; no within-generation mutation.
    reproduce() {
        const child = new FrozenPolicyOrganism(
            0,
            0,
            this.env,
            this,
            this.rl_enabled,
            this.ga_manager
        );

        if (Hyperparams.rotationEnabled) {
            child.rotation = Directions.getRandomDirection();
        }

        const child_genome = new Float32Array(this.brain.getGenome());
        child.brain.setGenome(child_genome);

        let new_c;
        let new_r;
        if (this.ga_manager) {
            [new_c, new_r] = this.ga_manager._getRandomSpawnPosition();
        } else {
            new_c = 0;
            new_r = 0;
        }

        if (
            child.isClear(new_c, new_r, child.rotation, true) &&
            this.env.canAddOrganism() &&
            Math.random() < REPRODUCTION_SUCCESS_RATE
        ) {
            child.c = new_c;
            child.r = new_r;
            this.env.addOrganism(child);

            if (this.ga_manager) {
                this.ga_manager.registerAgent(child);
            }

            if (this.species) {
                child.species = this.species;
                this.species.addPop();
            }
        }

        this.energyGainedSinceReproduction = 0;
    }
}

module.exports = FrozenPolicyOrganism;
