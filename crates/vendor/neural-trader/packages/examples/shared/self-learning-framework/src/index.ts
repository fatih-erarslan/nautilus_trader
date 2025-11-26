/**
 * Self-Learning Framework
 *
 * Unified self-learning interface with AgentDB integration for
 * experience replay, pattern recognition, and adaptive parameter tuning.
 *
 * @packageDocumentation
 */

export { ExperienceReplay } from './experience-replay';
export type {
  Experience,
  ReplayConfig,
  ReplayBatch,
} from './experience-replay';

export { PatternLearner } from './pattern-learner';
export type {
  Pattern,
  PatternMatch,
  LearnerConfig,
} from './pattern-learner';

export { AdaptiveParameters } from './adaptive-params';
export type {
  Parameter,
  AdaptationConfig,
  PerformanceMetrics,
} from './adaptive-params';

/**
 * Factory function to create a complete self-learning setup
 */
export function createSelfLearningSystem(config: {
  replay: import('./experience-replay').ReplayConfig;
  learner: import('./pattern-learner').LearnerConfig;
  adaptation: import('./adaptive-params').AdaptationConfig;
}) {
  const replay = new (require('./experience-replay').ExperienceReplay)(config.replay);
  const learner = new (require('./pattern-learner').PatternLearner)(replay, config.learner);
  const adaptation = new (require('./adaptive-params').AdaptiveParameters)(
    replay,
    learner,
    config.adaptation
  );

  return {
    replay,
    learner,
    adaptation,

    /**
     * Record experience and trigger adaptation if needed
     */
    async learn(experience: import('./experience-replay').Experience): Promise<void> {
      await replay.store(experience);
      const adapted = await adaptation.recordExperience(experience);

      if (adapted) {
        await learner.learnPatterns();
      }
    },

    /**
     * Get current parameters for execution
     */
    getParameters(): Record<string, any> {
      return adaptation.getAllParameters();
    },

    /**
     * Export full system state
     */
    async exportState() {
      return {
        experiences: await replay.export(),
        patterns: learner.export(),
        parameters: adaptation.export(),
      };
    },

    /**
     * Import system state
     */
    async importState(state: any) {
      if (state.experiences) {
        await replay.import(state.experiences);
      }
      if (state.patterns) {
        await learner.import(state.patterns);
      }
      if (state.parameters) {
        adaptation.import(state.parameters);
      }
    },
  };
}
