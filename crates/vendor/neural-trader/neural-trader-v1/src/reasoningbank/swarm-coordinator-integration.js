/**
 * SwarmCoordinator Integration with ReasoningBank
 *
 * Extends SwarmCoordinator with learning capabilities
 *
 * @module reasoningbank/swarm-coordinator-integration
 */

const { ReasoningBankSwarmLearner, LearningMode } = require('./swarm-learning');

/**
 * Add ReasoningBank learning methods to SwarmCoordinator
 *
 * @param {SwarmCoordinator} coordinator - SwarmCoordinator instance
 * @param {Object} learningConfig - Learning configuration
 * @returns {SwarmCoordinator} Enhanced coordinator
 */
function addLearningCapabilities(coordinator, learningConfig = {}) {
  // Initialize ReasoningBank learner
  const learner = new ReasoningBankSwarmLearner(coordinator.swarmId, {
    learningMode: learningConfig.mode || LearningMode.ONLINE,
    quicUrl: learningConfig.quicUrl || coordinator.agentDBConfig?.quicUrl,
    ...learningConfig
  });

  // Attach learner to coordinator
  coordinator.reasoningBank = learner;

  /**
   * Initialize learning system
   */
  coordinator.initializeLearning = async function() {
    if (!this.reasoningBank) {
      throw new Error('ReasoningBank not attached');
    }

    await this.reasoningBank.initialize();

    // Start episode when swarm initializes
    this.reasoningBank.startEpisode({
      topology: this.topology,
      agentCount: this.maxAgents
    });

    console.log('🧠 ReasoningBank learning enabled for swarm');

    return { enabled: true, mode: this.reasoningBank.learningMode };
  };

  /**
   * Record trading decision
   */
  coordinator.recordDecision = async function(agentId, decision, metadata = {}) {
    if (!this.reasoningBank?.isInitialized) {
      return null;
    }

    const trajectory = await this.reasoningBank.recordTrajectory(
      {
        ...decision,
        agentId,
        timestamp: Date.now()
      },
      null, // Outcome will be added later
      {
        ...metadata,
        episodeId: this.reasoningBank.currentEpisode?.id
      }
    );

    return trajectory;
  };

  /**
   * Update decision with outcome
   */
  coordinator.recordOutcome = async function(trajectoryId, outcome) {
    if (!this.reasoningBank?.isInitialized) {
      return null;
    }

    // Update trajectory with outcome
    await this.reasoningBank.trajectoryTracker.updateOutcome(trajectoryId, outcome);

    // Judge verdict if in online mode
    if (this.reasoningBank.learningMode === LearningMode.ONLINE) {
      await this.reasoningBank.judgeVerdict(trajectoryId);
    }

    return { trajectoryId, outcome };
  };

  /**
   * Get learning recommendations for current state
   */
  coordinator.getRecommendations = async function(marketState, options = {}) {
    if (!this.reasoningBank?.isInitialized) {
      return [];
    }

    const similar = await this.reasoningBank.querySimilarDecisions(marketState, options);

    return similar.map(s => ({
      pattern: s.pattern,
      trajectory: s.trajectory,
      recommendation: s.recommendation,
      similarity: s.similarity
    }));
  };

  /**
   * Adapt agent based on learnings
   */
  coordinator.adaptAgent = async function(agentId) {
    if (!this.reasoningBank?.isInitialized) {
      return { adapted: false, reason: 'learning_disabled' };
    }

    // Get agent's trajectories
    const trajectories = await this.reasoningBank.trajectoryTracker.getAgentTrajectories(agentId);

    if (trajectories.length === 0) {
      return { adapted: false, reason: 'no_data' };
    }

    // Extract learnings from successful trajectories
    const learnings = trajectories.filter(t => t.verdict && t.verdict.score >= 0.7);

    if (learnings.length === 0) {
      return { adapted: false, reason: 'no_successful_trades' };
    }

    // Adapt strategy
    const result = await this.reasoningBank.adaptAgentStrategy(agentId, learnings);

    return result;
  };

  /**
   * End trading episode and learn
   */
  coordinator.endTradingEpisode = async function(episodeResult) {
    if (!this.reasoningBank?.isInitialized || !this.reasoningBank.currentEpisode) {
      return null;
    }

    const episode = await this.reasoningBank.endEpisode(episodeResult);

    return {
      episodeId: episode.id,
      duration: episode.duration,
      trajectoriesCount: episode.trajectoryIds.length,
      result: episode.result
    };
  };

  /**
   * Get learning statistics
   */
  coordinator.getLearningStats = function() {
    if (!this.reasoningBank?.isInitialized) {
      return { enabled: false };
    }

    return {
      enabled: true,
      ...this.reasoningBank.getStats(),
      trajectoryStats: this.reasoningBank.trajectoryTracker.getStats(),
      verdictStats: this.reasoningBank.verdictJudge.getStats(),
      distillerStats: this.reasoningBank.memoryDistiller.getStats(),
      recognizerStats: this.reasoningBank.patternRecognizer.getStats()
    };
  };

  /**
   * Override shutdown to cleanup learning
   */
  const originalShutdown = coordinator.shutdown.bind(coordinator);
  coordinator.shutdown = async function() {
    // Shutdown ReasoningBank
    if (this.reasoningBank?.isInitialized) {
      await this.reasoningBank.shutdown();
    }

    // Call original shutdown
    await originalShutdown();
  };

  return coordinator;
}

module.exports = {
  addLearningCapabilities
};
