/**
 * ReasoningBank Swarm Learning Integration
 *
 * Adaptive learning system for E2B trading swarms that learns from experience:
 * - Records all trading decisions with full context (Trajectory Tracking)
 * - Evaluates decision quality based on outcomes (Verdict Judgment)
 * - Learns from successful/failed patterns (Memory Distillation)
 * - Retrieves similar past decisions for informed trading (Pattern Recognition)
 * - Continuously adapts agent strategies based on learning (Strategy Adaptation)
 *
 * Uses AgentDB for 150x faster vector similarity search and distributed memory.
 *
 * @module reasoningbank/swarm-learning
 */

const EventEmitter = require('events');
const { AgentDBClient } = require('../coordination/agentdb-client');
const TrajectoryTracker = require('./trajectory-tracker');
const VerdictJudge = require('./verdict-judge');
const MemoryDistiller = require('./memory-distiller');
const PatternRecognizer = require('./pattern-recognizer');

/**
 * Learning modes for different scenarios
 */
const LearningMode = {
  ONLINE: 'online',           // Learn from every trade immediately
  BATCH: 'batch',             // Learn from batches of trades
  EPISODE: 'episode',         // Learn from complete trading episodes
  CONTINUOUS: 'continuous'    // Continuous background learning
};

/**
 * Verdict scores (0-1 scale)
 */
const VerdictScore = {
  EXCELLENT: 0.9,    // Outstanding decision
  GOOD: 0.7,         // Positive outcome
  NEUTRAL: 0.5,      // No clear outcome
  POOR: 0.3,         // Negative outcome
  TERRIBLE: 0.1      // Very bad outcome
};

/**
 * ReasoningBank Swarm Learner
 *
 * Main class that coordinates learning across trading swarms
 */
class ReasoningBankSwarmLearner extends EventEmitter {
  constructor(swarmId, agentDBConfig = {}) {
    super();

    this.swarmId = swarmId;
    this.learningMode = agentDBConfig.learningMode || LearningMode.ONLINE;

    // Core ReasoningBank components
    this.trajectoryTracker = new TrajectoryTracker(swarmId);
    this.verdictJudge = new VerdictJudge();
    this.memoryDistiller = new MemoryDistiller();
    this.patternRecognizer = new PatternRecognizer();

    // AgentDB for fast vector storage and retrieval
    this.agentDB = null;
    this.agentDBConfig = {
      quicUrl: agentDBConfig.quicUrl || 'quic://localhost:8443',
      sandboxId: swarmId,
      strategyType: 'reasoningbank_learner',
      enableQuantization: agentDBConfig.enableQuantization !== false,
      enableHNSW: agentDBConfig.enableHNSW !== false,
      ...agentDBConfig
    };

    // Learning state
    this.learningStats = {
      totalDecisions: 0,
      totalOutcomes: 0,
      avgVerdictScore: 0.5,
      patternsLearned: 0,
      adaptationEvents: 0,
      startTime: Date.now()
    };

    // Episode tracking
    this.currentEpisode = null;
    this.episodes = new Map();

    // Learning intervals
    this.batchInterval = null;
    this.distillationInterval = null;

    this.isInitialized = false;
  }

  /**
   * Initialize ReasoningBank learning system
   * @returns {Promise<Object>} Initialization result
   */
  async initialize() {
    console.log(`\n🧠 Initializing ReasoningBank for swarm: ${this.swarmId}`);

    try {
      // Initialize AgentDB connection
      await this.initializeAgentDB();

      // Initialize components
      await this.trajectoryTracker.initialize(this.agentDB);
      await this.verdictJudge.initialize();
      await this.memoryDistiller.initialize(this.agentDB);
      await this.patternRecognizer.initialize(this.agentDB);

      // Start background learning based on mode
      if (this.learningMode === LearningMode.BATCH) {
        this.startBatchLearning();
      } else if (this.learningMode === LearningMode.CONTINUOUS) {
        this.startContinuousLearning();
      }

      this.isInitialized = true;

      const result = {
        swarmId: this.swarmId,
        learningMode: this.learningMode,
        agentDBConnected: this.agentDB !== null,
        componentsReady: true,
        timestamp: new Date().toISOString()
      };

      this.emit('initialized', result);
      console.log('✅ ReasoningBank initialized successfully\n');

      return result;

    } catch (error) {
      console.error('❌ ReasoningBank initialization failed:', error.message);
      throw error;
    }
  }

  /**
   * Initialize AgentDB connection
   * @private
   */
  async initializeAgentDB() {
    console.log('🔗 Connecting to AgentDB for learning storage...');

    this.agentDB = new AgentDBClient(this.agentDBConfig);
    await this.agentDB.connect();

    console.log('✅ AgentDB connected with QUIC acceleration');
  }

  /**
   * Record a trading decision with full context (Trajectory)
   *
   * @param {Object} decision - Trading decision details
   * @param {Object} outcome - Trade outcome (can be null initially)
   * @param {Object} metadata - Additional context
   * @returns {Promise<Object>} Recorded trajectory
   */
  async recordTrajectory(decision, outcome = null, metadata = {}) {
    if (!this.isInitialized) {
      throw new Error('ReasoningBank not initialized. Call initialize() first.');
    }

    console.log(`\n📝 Recording trajectory for decision: ${decision.id || 'unknown'}`);

    const trajectory = await this.trajectoryTracker.recordTrajectory({
      decision,
      outcome,
      metadata: {
        ...metadata,
        swarmId: this.swarmId,
        timestamp: Date.now(),
        learningMode: this.learningMode
      }
    });

    this.learningStats.totalDecisions++;

    // If outcome is present, update stats
    if (outcome) {
      this.learningStats.totalOutcomes++;
    }

    // Online learning: evaluate immediately if outcome exists
    if (this.learningMode === LearningMode.ONLINE && outcome) {
      await this.judgeVerdict(trajectory.id);
    }

    this.emit('trajectory:recorded', trajectory);

    return trajectory;
  }

  /**
   * Evaluate decision quality and assign verdict score
   *
   * @param {string} trajectoryId - ID of trajectory to judge
   * @returns {Promise<Object>} Verdict result
   */
  async judgeVerdict(trajectoryId) {
    console.log(`\n⚖️  Judging verdict for trajectory: ${trajectoryId}`);

    const trajectory = await this.trajectoryTracker.getTrajectory(trajectoryId);

    if (!trajectory) {
      throw new Error(`Trajectory ${trajectoryId} not found`);
    }

    if (!trajectory.outcome) {
      throw new Error(`Trajectory ${trajectoryId} has no outcome yet`);
    }

    // Evaluate decision quality
    const verdict = await this.verdictJudge.evaluate(trajectory);

    // Update trajectory with verdict
    await this.trajectoryTracker.updateVerdict(trajectoryId, verdict);

    // Update learning stats
    const currentAvg = this.learningStats.avgVerdictScore;
    const totalOutcomes = this.learningStats.totalOutcomes;
    this.learningStats.avgVerdictScore =
      (currentAvg * (totalOutcomes - 1) + verdict.score) / totalOutcomes;

    this.emit('verdict:judged', { trajectoryId, verdict });

    console.log(`  ✅ Verdict: ${verdict.quality} (score: ${verdict.score.toFixed(3)})`);

    return verdict;
  }

  /**
   * Learn from a completed episode of trading
   *
   * @param {string} episodeId - Episode identifier
   * @returns {Promise<Object>} Learning results
   */
  async learnFromExperience(episodeId) {
    console.log(`\n📚 Learning from episode: ${episodeId}`);

    const episode = this.episodes.get(episodeId);

    if (!episode) {
      throw new Error(`Episode ${episodeId} not found`);
    }

    // Get all trajectories for this episode
    const trajectories = await this.trajectoryTracker.getEpisodeTrajectories(episodeId);

    // Filter trajectories with verdicts
    const judgedTrajectories = trajectories.filter(t => t.verdict);

    if (judgedTrajectories.length === 0) {
      console.log('  ⚠️  No judged trajectories to learn from');
      return { learned: 0, patterns: [] };
    }

    // Learn patterns from successful decisions
    const successfulTrajectories = judgedTrajectories.filter(
      t => t.verdict.score >= VerdictScore.GOOD
    );

    const patterns = await this.memoryDistiller.distillPatterns(
      successfulTrajectories,
      {
        minScore: VerdictScore.GOOD,
        includeContext: true
      }
    );

    // Store learned patterns in AgentDB
    for (const pattern of patterns) {
      await this.patternRecognizer.storePattern(pattern);
      this.learningStats.patternsLearned++;
    }

    const result = {
      episodeId,
      trajectoriesAnalyzed: judgedTrajectories.length,
      patternsLearned: patterns.length,
      avgScore: judgedTrajectories.reduce((sum, t) => sum + t.verdict.score, 0) / judgedTrajectories.length,
      timestamp: new Date().toISOString()
    };

    this.emit('episode:learned', result);

    console.log(`  ✅ Learned ${patterns.length} patterns from ${judgedTrajectories.length} trajectories`);

    return result;
  }

  /**
   * Query similar past decisions using vector similarity
   *
   * @param {Object} currentState - Current market/trading state
   * @param {Object} options - Query options
   * @returns {Promise<Array>} Similar past decisions
   */
  async querySimilarDecisions(currentState, options = {}) {
    console.log('\n🔍 Querying similar past decisions...');

    const similarPatterns = await this.patternRecognizer.findSimilar(currentState, {
      topK: options.topK || 5,
      minSimilarity: options.minSimilarity || 0.7,
      includeContext: true
    });

    // Enrich with trajectory details
    const enriched = await Promise.all(
      similarPatterns.map(async (pattern) => {
        const trajectory = await this.trajectoryTracker.getTrajectory(pattern.trajectoryId);
        return {
          ...pattern,
          trajectory,
          recommendation: this.generateRecommendation(pattern, trajectory)
        };
      })
    );

    console.log(`  ✅ Found ${enriched.length} similar decisions`);

    return enriched;
  }

  /**
   * Adapt agent strategy based on learned patterns
   *
   * @param {string} agentId - Agent to adapt
   * @param {Array} learnings - Learning insights
   * @returns {Promise<Object>} Adaptation result
   */
  async adaptAgentStrategy(agentId, learnings) {
    console.log(`\n🔄 Adapting strategy for agent: ${agentId}`);

    if (!learnings || learnings.length === 0) {
      console.log('  ⚠️  No learnings to apply');
      return { adapted: false, reason: 'no_learnings' };
    }

    // Aggregate learnings into strategy adjustments
    const adjustments = this.memoryDistiller.aggregateLearnings(learnings);

    // Apply adjustments to agent via AgentDB
    if (this.agentDB) {
      await this.agentDB.updateState({
        agent_id: agentId,
        strategy_adjustments: adjustments,
        learning_applied: true,
        timestamp: Date.now()
      });
    }

    this.learningStats.adaptationEvents++;

    const result = {
      agentId,
      adjustmentsApplied: adjustments.length,
      learningsProcessed: learnings.length,
      timestamp: new Date().toISOString()
    };

    this.emit('strategy:adapted', result);

    console.log(`  ✅ Applied ${adjustments.length} strategy adjustments`);

    return result;
  }

  /**
   * Start a new trading episode
   *
   * @param {Object} episodeConfig - Episode configuration
   * @returns {Object} Episode info
   */
  startEpisode(episodeConfig = {}) {
    const episodeId = `episode-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    const episode = {
      id: episodeId,
      swarmId: this.swarmId,
      startTime: Date.now(),
      config: episodeConfig,
      trajectoryIds: [],
      status: 'active'
    };

    this.episodes.set(episodeId, episode);
    this.currentEpisode = episode;

    console.log(`\n▶️  Started episode: ${episodeId}`);

    this.emit('episode:started', episode);

    return episode;
  }

  /**
   * End current episode and trigger learning
   *
   * @param {Object} episodeResult - Episode outcome
   * @returns {Promise<Object>} Episode summary
   */
  async endEpisode(episodeResult = {}) {
    if (!this.currentEpisode) {
      throw new Error('No active episode');
    }

    const episodeId = this.currentEpisode.id;
    const episode = this.episodes.get(episodeId);

    episode.endTime = Date.now();
    episode.duration = episode.endTime - episode.startTime;
    episode.result = episodeResult;
    episode.status = 'completed';

    console.log(`\n⏹️  Ended episode: ${episodeId} (duration: ${episode.duration}ms)`);

    // Learn from episode if in episode mode
    if (this.learningMode === LearningMode.EPISODE) {
      await this.learnFromExperience(episodeId);
    }

    this.currentEpisode = null;

    this.emit('episode:ended', episode);

    return episode;
  }

  /**
   * Get learning statistics
   *
   * @returns {Object} Learning stats
   */
  getStats() {
    const uptime = (Date.now() - this.learningStats.startTime) / 1000;

    return {
      ...this.learningStats,
      uptime: `${uptime.toFixed(2)}s`,
      learningRate: this.learningStats.patternsLearned / Math.max(1, uptime),
      decisionsPerSecond: this.learningStats.totalDecisions / Math.max(1, uptime),
      adaptationRate: this.learningStats.adaptationEvents / Math.max(1, uptime),
      currentEpisode: this.currentEpisode?.id || null,
      totalEpisodes: this.episodes.size
    };
  }

  /**
   * Generate recommendation from pattern
   * @private
   */
  generateRecommendation(pattern, trajectory) {
    if (!trajectory || !trajectory.verdict) {
      return { action: 'unknown', confidence: 0 };
    }

    return {
      action: trajectory.verdict.score >= VerdictScore.GOOD ? 'follow' : 'avoid',
      confidence: Math.abs(trajectory.verdict.score - 0.5) * 2,
      reason: trajectory.verdict.analysis?.keyFactors || []
    };
  }

  /**
   * Start batch learning process
   * @private
   */
  startBatchLearning() {
    console.log('📦 Starting batch learning mode');

    this.batchInterval = setInterval(async () => {
      try {
        await this.processBatchLearning();
      } catch (error) {
        this.emit('error', { type: 'batch_learning', error: error.message });
      }
    }, 60000); // Every minute
  }

  /**
   * Start continuous learning process
   * @private
   */
  startContinuousLearning() {
    console.log('🔄 Starting continuous learning mode');

    this.distillationInterval = setInterval(async () => {
      try {
        await this.continuousDistillation();
      } catch (error) {
        this.emit('error', { type: 'continuous_learning', error: error.message });
      }
    }, 30000); // Every 30 seconds
  }

  /**
   * Process batch learning
   * @private
   */
  async processBatchLearning() {
    const pendingTrajectories = await this.trajectoryTracker.getPendingTrajectories();

    if (pendingTrajectories.length === 0) {
      return;
    }

    console.log(`\n📦 Processing batch: ${pendingTrajectories.length} trajectories`);

    // Judge all pending trajectories
    for (const trajectory of pendingTrajectories) {
      if (trajectory.outcome) {
        await this.judgeVerdict(trajectory.id);
      }
    }
  }

  /**
   * Continuous pattern distillation
   * @private
   */
  async continuousDistillation() {
    const recentTrajectories = await this.trajectoryTracker.getRecentTrajectories(100);
    const judgedTrajectories = recentTrajectories.filter(t => t.verdict);

    if (judgedTrajectories.length < 10) {
      return; // Need minimum data
    }

    const patterns = await this.memoryDistiller.distillPatterns(
      judgedTrajectories.filter(t => t.verdict.score >= VerdictScore.NEUTRAL)
    );

    for (const pattern of patterns) {
      await this.patternRecognizer.storePattern(pattern);
      this.learningStats.patternsLearned++;
    }
  }

  /**
   * Shutdown ReasoningBank system
   *
   * @returns {Promise<void>}
   */
  async shutdown() {
    console.log(`\n🛑 Shutting down ReasoningBank for swarm ${this.swarmId}...`);

    // Stop intervals
    if (this.batchInterval) {
      clearInterval(this.batchInterval);
    }

    if (this.distillationInterval) {
      clearInterval(this.distillationInterval);
    }

    // Disconnect AgentDB
    if (this.agentDB) {
      await this.agentDB.disconnect();
    }

    this.isInitialized = false;

    console.log('✅ ReasoningBank shutdown complete\n');
    this.emit('shutdown', { swarmId: this.swarmId, timestamp: new Date().toISOString() });
  }
}

// Export class and constants
module.exports = {
  ReasoningBankSwarmLearner,
  LearningMode,
  VerdictScore
};
