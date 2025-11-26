/**
 * ReasoningBank - Adaptive Learning System for E2B Trading Swarms
 *
 * Complete exports for ReasoningBank integration
 *
 * @module reasoningbank
 */

const { ReasoningBankSwarmLearner, LearningMode, VerdictScore } = require('./swarm-learning');
const TrajectoryTracker = require('./trajectory-tracker');
const VerdictJudge = require('./verdict-judge');
const MemoryDistiller = require('./memory-distiller');
const PatternRecognizer = require('./pattern-recognizer');

module.exports = {
  // Main learner class
  ReasoningBankSwarmLearner,

  // Core components
  TrajectoryTracker,
  VerdictJudge,
  MemoryDistiller,
  PatternRecognizer,

  // Enums and constants
  LearningMode,
  VerdictScore,

  // Convenience factory function
  createLearner: (swarmId, config) => new ReasoningBankSwarmLearner(swarmId, config)
};
