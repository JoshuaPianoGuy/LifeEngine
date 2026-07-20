/**
 * FoodShuffle.js
 *
 * Non-stationary "shuffle" environment: periodically permutes which food TIER
 * pays which ENERGY VALUE, forcing WITHIN-LIFETIME learning.
 *
 * Baseline world: the three food colours the agent perceives on distinct input
 * channels — 'low food' (0.5), 'medium food' (1.0), 'prestige food' (2.0) — are
 * permanently locked to their payoff, so a fixed evolved policy ("always chase
 * teal/prestige") is optimal and no in-life learning is needed. This module
 * breaks that: every `food_shuffle_period` ticks it re-assigns the SAME multiset
 * of values {0.5, 1.0, 2.0} to the three tiers under a new permutation. The
 * spatial layout (food/landmark cells) is untouched — only the payoff each
 * colour delivers changes. An agent must taste a colour to discover its CURRENT
 * value; because the reward lands on the same tick as eating, the mapping is
 * learnable in-life, while a fixed genome that hard-codes a colour preference is
 * penalised after each shuffle.
 *
 * IMPLEMENTATION NOTE — GLOBAL MUTABLE STATE. The live value table is the single
 * shared object MouthCell.FOOD_ENERGY_VALUES (read by MouthCell on eat and by
 * AdvancedOrganism._foodValue). We mutate it IN PLACE so both readers see the
 * current permutation without any signature changes. Because it is process-global,
 * reset() MUST be called when a fresh world is constructed (WorldEnvironment
 * ctor) so a leftover permutation from a previous run/probe never leaks in and
 * the seeded schedule restarts from the base mapping. When food_shuffle_period is
 * 0 (the default — baseline/roam/all existing runs) this module is inert: reset()
 * just restores the base values and maybeShuffle() never fires.
 *
 * Reproducibility: the permutation schedule is driven by a dedicated mulberry32
 * PRNG seeded from food_shuffle_seed (0 = unseeded, use Math.random), mirroring
 * the natural-disaster design so it never disturbs the global unseeded RNG that
 * GA mutation / RL exploration rely on. A seeded schedule is COMMON-MODE across
 * fitness-landscape grid points (like terrain), so it cancels between points.
 */

'use strict';

const MouthCell = require('../Organism/Cell/BodyCells/MouthCell');

// The three tiers whose payoffs are permuted. Order is fixed; values move.
const TIERS = ['low food', 'medium food', 'prestige food'];

// Capture the base payoff multiset once, at module load, from the single source
// of truth. Frozen so an accidental in-place edit can't corrupt the baseline.
const BASE_VALUES = Object.freeze(TIERS.map((t) => MouthCell.FOOD_ENERGY_VALUES[t]));

// Deterministic mulberry32 PRNG factory — identical to GAManager._mulberry32, so
// the shuffle schedule is reproducible from food_shuffle_seed without touching
// the global unseeded RNG. Returns a Math.random-style function () -> [0,1).
function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
        a |= 0; a = (a + 0x6D2B79F5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// Module-level schedule PRNG. Re-seeded on every reset().
let _rng = Math.random;

const FoodShuffle = {
    TIERS,

    /** Write the base {low:0.5, medium:1.0, prestige:2.0} mapping into the live
     *  table and (re)seed the schedule PRNG. Call once per fresh world so runs /
     *  probes never inherit a stale permutation. seed 0 (or falsy) = unseeded. */
    reset(seed) {
        for (let i = 0; i < TIERS.length; i++) {
            MouthCell.FOOD_ENERGY_VALUES[TIERS[i]] = BASE_VALUES[i];
        }
        _rng = seed ? mulberry32(seed) : Math.random;
    },

    /** Current live payoff per tier — for logging. */
    currentMapping() {
        const out = {};
        for (const t of TIERS) out[t] = MouthCell.FOOD_ENERGY_VALUES[t];
        return out;
    },

    /**
     * If a shuffle is due this tick, permute the base values across the tiers
     * (guaranteed different from the current live mapping) and write it in place.
     * Returns the new mapping object when a shuffle happened, else null.
     *
     * @param {number} total_ticks  env.total_ticks (post-increment this tick)
     * @param {number} period       food_shuffle_period; <=0 disables (no-op)
     */
    maybeShuffle(total_ticks, period) {
        if (!period || period <= 0) return null;
        if (total_ticks <= 0 || total_ticks % period !== 0) return null;

        // Fisher–Yates over a copy of the base values, rerolled until the result
        // differs from the CURRENT live mapping so every shuffle visibly changes
        // at least one tier. With 3 distinct values, 5 of the 6 permutations
        // differ from any given one, so this terminates in ~1 reroll.
        const current = TIERS.map((t) => MouthCell.FOOD_ENERGY_VALUES[t]);
        let perm;
        do {
            perm = BASE_VALUES.slice();
            for (let i = perm.length - 1; i > 0; i--) {
                const j = Math.floor(_rng() * (i + 1));
                const tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
            }
        } while (perm.every((v, i) => v === current[i]));

        for (let i = 0; i < TIERS.length; i++) {
            MouthCell.FOOD_ENERGY_VALUES[TIERS[i]] = perm[i];
        }
        return this.currentMapping();
    },
};

module.exports = FoodShuffle;
