/**
 * @neural-trader/example-evolutionary-game-theory
 *
 * Self-learning evolutionary game theory with multi-agent tournaments,
 * replicator dynamics, and ESS calculation
 *
 * @packageDocumentation
 */

// Types
export * from './types.js';

// Games
export * from './games.js';

// Strategies
export * from './strategies.js';

// Replicator Dynamics
export * from './replicator-dynamics.js';

// ESS Analysis
export * from './ess.js';

// Tournament System
export * from './tournament.js';

// Swarm Evolution
export * from './swarm-evolution.js';

// Re-export commonly used items
export {
  PRISONERS_DILEMMA,
  HAWK_DOVE,
  STAG_HUNT,
  PUBLIC_GOODS,
  ROCK_PAPER_SCISSORS,
  createHawkDoveGame,
  createPublicGoodsGame,
} from './games.js';

export {
  ALWAYS_COOPERATE,
  ALWAYS_DEFECT,
  TIT_FOR_TAT,
  TIT_FOR_TWO_TATS,
  GRIM_TRIGGER,
  PAVLOV,
  RANDOM,
  ADAPTIVE,
  GRADUAL,
  PROBE,
  createGenerousTitForTat,
  createLearningStrategy,
} from './strategies.js';

export { ReplicatorDynamics, MultiPopulationDynamics } from './replicator-dynamics.js';

export { ESSCalculator, findAllESS } from './ess.js';

export { Tournament, quickTournament } from './tournament.js';

export { SwarmEvolution, quickSwarmEvolution } from './swarm-evolution.js';
