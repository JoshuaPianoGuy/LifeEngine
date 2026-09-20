/**
 * FrozenPolicyOrganism.js
 *
 * UNUSED IN THE FINAL EXPERIMENTS. Only ever instantiated by
 * FrozenPolicyManager, i.e. under --mode frozen_pg, which no reported run
 * uses. See FrozenPolicyManager.js for the full note.
 *
 * An AdvancedOrganism with two changes that together hold the policy fixed
 * for the whole generation:
 *   1. NNBrain.freeze_updates is set at construction, so the per-tick
 *      REINFORCE update returns early and active_weights never drift.
 *   2. reproduce() clones the parent genome EXACTLY, with no
 *      within-generation Gaussian mutation, so an entire generation is
 *      genetically uniform and its fitness is a clean estimate of one policy.
 *
 * Note REPRODUCTION_SUCCESS_RATE below is a local constant, so this class does
 * not honour ExperimentParams.reproduction_success_rate the way
 * AdvancedOrganism does.
 */

'use strict';

const AdvancedOrganism = require('./AdvancedOrganism');
const NNBrain = require('./Perception/NNBrain');
const Directions = require('./Directions');
const Hyperparams = require('../Hyperparameters');

const REPRODUCTION_SUCCESS_RATE = 0.8;

/**
 * Organism whose policy cannot change during its lifetime: RL updates are
 * frozen and reproduction is exact cloning.
 *
 * Unused in the final experiments — see the file header.
 */
class FrozenPolicyOrganism extends AdvancedOrganism {
    /**
     * @param {number}              col         spawn column
     * @param {number}              row         spawn row
     * @param {WorldEnvironment}    env         owning environment
     * @param {FrozenPolicyOrganism} [parent]   parent to inherit the genome from
     * @param {boolean}             [rl_enabled] passed through to the brain
     * @param {FrozenPolicyManager} [ga_manager] manager that tracks this agent
     */
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
