/**
 * GenerationConstants.js
 *
 * Single source of truth for generation length. Previously TICKS_PER_MAP /
 * MAPS_PER_GEN / TICKS_PER_GEN were redeclared independently in GAManager,
 * Purerlmanager, FrozenPolicyManager AND (as a hardcoded literal 10000) in
 * AdvancedOrganism, where it is the denominator of the per-organism epsilon
 * decay window. If those ever drifted apart, epsilon would decay over the
 * wrong horizon — reintroducing the exact per-generation-vs-per-lifetime bug.
 * Importing from here keeps every consumer in lockstep.
 *
 * Honors the WorldConfig overrides (null = use the defaults below), so
 * TICKS_PER_MAP / MAPS_PER_GEN in WorldConfig now actually take effect.
 * Values are resolved once at module load (set them in WorldConfig and rebuild).
 */

'use strict';

const WorldConfig = require('../WorldConfig');

const DEFAULT_TICKS_PER_MAP = 2000;
const DEFAULT_MAPS_PER_GEN  = 5;

const TICKS_PER_MAP = (WorldConfig.TICKS_PER_MAP != null)
    ? WorldConfig.TICKS_PER_MAP
    : DEFAULT_TICKS_PER_MAP;
const MAPS_PER_GEN = (WorldConfig.MAPS_PER_GEN != null)
    ? WorldConfig.MAPS_PER_GEN
    : DEFAULT_MAPS_PER_GEN;

// Total world ticks in one generation = the upper bound on any organism's
// lifetime, hence the epsilon-decay horizon in AdvancedOrganism.
const TICKS_PER_GEN = TICKS_PER_MAP * MAPS_PER_GEN;

module.exports = { TICKS_PER_MAP, MAPS_PER_GEN, TICKS_PER_GEN };
