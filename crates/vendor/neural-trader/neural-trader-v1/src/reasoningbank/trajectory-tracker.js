/**
 * Trajectory Tracker - Records Trading Decision Trajectories
 *
 * Tracks the complete lifecycle of trading decisions:
 * 1. Initial decision with context (market state, agent state, reasoning)
 * 2. Execution details (timing, order fills, slippage)
 * 3. Outcome measurement (P&L, risk metrics, timing accuracy)
 * 4. Verdict assignment (quality evaluation)
 *
 * Stores trajectories in AgentDB for fast retrieval and distributed access.
 *
 * @module reasoningbank/trajectory-tracker
 */

const EventEmitter = require('events');
const crypto = require('crypto');

/**
 * Trajectory status types
 */
const TrajectoryStatus = {
  PENDING: 'pending',           // Decision made, waiting for outcome
  EXECUTING: 'executing',       // Trade in progress
  COMPLETED: 'completed',       // Outcome recorded
  JUDGED: 'judged',            // Verdict assigned
  LEARNED: 'learned'           // Used for learning
};

/**
 * Trajectory Tracker Class
 */
class TrajectoryTracker extends EventEmitter {
  constructor(swarmId) {
    super();

    this.swarmId = swarmId;
    this.trajectories = new Map();
    this.episodeTrajectories = new Map(); // episodeId -> Set of trajectoryIds
    this.agentDB = null;

    this.stats = {
      total: 0,
      pending: 0,
      completed: 0,
      judged: 0,
      learned: 0
    };
  }

  /**
   * Initialize tracker with AgentDB
   *
   * @param {AgentDBClient} agentDB - AgentDB client instance
   * @returns {Promise<void>}
   */
  async initialize(agentDB) {
    this.agentDB = agentDB;

    // Create collection for trajectories if it doesn't exist
    if (this.agentDB) {
      await this.agentDB.updateState({
        collection: 'trajectories',
        swarmId: this.swarmId,
        initialized: true
      });
    }

    console.log('✅ Trajectory Tracker initialized');
  }

  /**
   * Record a new trajectory
   *
   * @param {Object} trajectoryData - Trajectory data
   * @returns {Promise<Object>} Recorded trajectory
   */
  async recordTrajectory(trajectoryData) {
    const { decision, outcome, metadata } = trajectoryData;

    // Generate unique trajectory ID
    const trajectoryId = this.generateTrajectoryId(decision);

    const trajectory = {
      id: trajectoryId,
      swarmId: this.swarmId,

      // Decision details
      decision: {
        id: decision.id,
        agentId: decision.agentId,
        type: decision.type || 'unknown',
        action: decision.action,
        symbol: decision.symbol,
        quantity: decision.quantity,
        price: decision.price,
        reasoning: decision.reasoning || {},
        marketState: decision.marketState || {},
        agentState: decision.agentState || {},
        timestamp: decision.timestamp || Date.now()
      },

      // Outcome details (if available)
      outcome: outcome ? {
        executed: outcome.executed !== false,
        fillPrice: outcome.fillPrice,
        fillQuantity: outcome.fillQuantity,
        slippage: outcome.slippage,
        executionTime: outcome.executionTime,
        pnl: outcome.pnl,
        pnlPercent: outcome.pnlPercent,
        riskAdjustedReturn: outcome.riskAdjustedReturn,
        timestamp: outcome.timestamp || Date.now()
      } : null,

      // Metadata
      metadata: {
        ...metadata,
        recordedAt: Date.now()
      },

      // Status tracking
      status: outcome ? TrajectoryStatus.COMPLETED : TrajectoryStatus.PENDING,
      verdict: null,
      episodeId: metadata.episodeId || null,

      // Timestamps
      createdAt: Date.now(),
      updatedAt: Date.now()
    };

    // Store locally
    this.trajectories.set(trajectoryId, trajectory);

    // Update episode mapping
    if (trajectory.episodeId) {
      if (!this.episodeTrajectories.has(trajectory.episodeId)) {
        this.episodeTrajectories.set(trajectory.episodeId, new Set());
      }
      this.episodeTrajectories.get(trajectory.episodeId).add(trajectoryId);
    }

    // Store in AgentDB for distributed access
    if (this.agentDB) {
      await this.agentDB.updateState({
        collection: 'trajectories',
        trajectoryId,
        data: trajectory
      });
    }

    // Update stats
    this.stats.total++;
    this.stats[trajectory.status]++;

    this.emit('trajectory:recorded', trajectory);

    return trajectory;
  }

  /**
   * Update trajectory with outcome
   *
   * @param {string} trajectoryId - Trajectory ID
   * @param {Object} outcome - Outcome data
   * @returns {Promise<Object>} Updated trajectory
   */
  async updateOutcome(trajectoryId, outcome) {
    const trajectory = this.trajectories.get(trajectoryId);

    if (!trajectory) {
      throw new Error(`Trajectory ${trajectoryId} not found`);
    }

    // Update outcome
    trajectory.outcome = {
      executed: outcome.executed !== false,
      fillPrice: outcome.fillPrice,
      fillQuantity: outcome.fillQuantity,
      slippage: outcome.slippage,
      executionTime: outcome.executionTime,
      pnl: outcome.pnl,
      pnlPercent: outcome.pnlPercent,
      riskAdjustedReturn: outcome.riskAdjustedReturn,
      timestamp: outcome.timestamp || Date.now()
    };

    // Update status
    const oldStatus = trajectory.status;
    trajectory.status = TrajectoryStatus.COMPLETED;
    trajectory.updatedAt = Date.now();

    // Update stats
    if (oldStatus !== trajectory.status) {
      this.stats[oldStatus]--;
      this.stats[trajectory.status]++;
    }

    // Sync to AgentDB
    if (this.agentDB) {
      await this.agentDB.updateState({
        collection: 'trajectories',
        trajectoryId,
        data: trajectory
      });
    }

    this.emit('trajectory:outcome_updated', trajectory);

    return trajectory;
  }

  /**
   * Update trajectory with verdict
   *
   * @param {string} trajectoryId - Trajectory ID
   * @param {Object} verdict - Verdict data
   * @returns {Promise<Object>} Updated trajectory
   */
  async updateVerdict(trajectoryId, verdict) {
    const trajectory = this.trajectories.get(trajectoryId);

    if (!trajectory) {
      throw new Error(`Trajectory ${trajectoryId} not found`);
    }

    // Update verdict
    trajectory.verdict = {
      score: verdict.score,
      quality: verdict.quality,
      analysis: verdict.analysis,
      timestamp: Date.now()
    };

    // Update status
    const oldStatus = trajectory.status;
    trajectory.status = TrajectoryStatus.JUDGED;
    trajectory.updatedAt = Date.now();

    // Update stats
    if (oldStatus !== trajectory.status) {
      this.stats[oldStatus]--;
      this.stats[trajectory.status]++;
    }

    // Sync to AgentDB
    if (this.agentDB) {
      await this.agentDB.updateState({
        collection: 'trajectories',
        trajectoryId,
        data: trajectory
      });
    }

    this.emit('trajectory:verdict_updated', trajectory);

    return trajectory;
  }

  /**
   * Mark trajectory as learned
   *
   * @param {string} trajectoryId - Trajectory ID
   * @returns {Promise<Object>} Updated trajectory
   */
  async markLearned(trajectoryId) {
    const trajectory = this.trajectories.get(trajectoryId);

    if (!trajectory) {
      throw new Error(`Trajectory ${trajectoryId} not found`);
    }

    const oldStatus = trajectory.status;
    trajectory.status = TrajectoryStatus.LEARNED;
    trajectory.updatedAt = Date.now();

    // Update stats
    if (oldStatus !== trajectory.status) {
      this.stats[oldStatus]--;
      this.stats[trajectory.status]++;
    }

    // Sync to AgentDB
    if (this.agentDB) {
      await this.agentDB.updateState({
        collection: 'trajectories',
        trajectoryId,
        learned: true
      });
    }

    return trajectory;
  }

  /**
   * Get trajectory by ID
   *
   * @param {string} trajectoryId - Trajectory ID
   * @returns {Promise<Object>} Trajectory
   */
  async getTrajectory(trajectoryId) {
    // Try local cache first
    let trajectory = this.trajectories.get(trajectoryId);

    // Fall back to AgentDB
    if (!trajectory && this.agentDB) {
      const state = await this.agentDB.getState();
      trajectory = state?.trajectories?.[trajectoryId];

      if (trajectory) {
        this.trajectories.set(trajectoryId, trajectory);
      }
    }

    return trajectory;
  }

  /**
   * Get all trajectories for an episode
   *
   * @param {string} episodeId - Episode ID
   * @returns {Promise<Array>} Array of trajectories
   */
  async getEpisodeTrajectories(episodeId) {
    const trajectoryIds = this.episodeTrajectories.get(episodeId);

    if (!trajectoryIds) {
      return [];
    }

    const trajectories = [];

    for (const trajectoryId of trajectoryIds) {
      const trajectory = await this.getTrajectory(trajectoryId);
      if (trajectory) {
        trajectories.push(trajectory);
      }
    }

    return trajectories;
  }

  /**
   * Get pending trajectories (no outcome yet)
   *
   * @returns {Promise<Array>} Pending trajectories
   */
  async getPendingTrajectories() {
    return Array.from(this.trajectories.values())
      .filter(t => t.status === TrajectoryStatus.PENDING);
  }

  /**
   * Get recent trajectories
   *
   * @param {number} limit - Maximum number to return
   * @returns {Promise<Array>} Recent trajectories
   */
  async getRecentTrajectories(limit = 100) {
    return Array.from(this.trajectories.values())
      .sort((a, b) => b.createdAt - a.createdAt)
      .slice(0, limit);
  }

  /**
   * Get trajectories by status
   *
   * @param {string} status - Status to filter by
   * @returns {Promise<Array>} Filtered trajectories
   */
  async getTrajectorysByStatus(status) {
    return Array.from(this.trajectories.values())
      .filter(t => t.status === status);
  }

  /**
   * Get trajectories by agent
   *
   * @param {string} agentId - Agent ID
   * @returns {Promise<Array>} Agent's trajectories
   */
  async getAgentTrajectories(agentId) {
    return Array.from(this.trajectories.values())
      .filter(t => t.decision.agentId === agentId);
  }

  /**
   * Get statistics
   *
   * @returns {Object} Tracker statistics
   */
  getStats() {
    return {
      ...this.stats,
      episodes: this.episodeTrajectories.size,
      avgTrajectoriesPerEpisode: this.stats.total / Math.max(1, this.episodeTrajectories.size)
    };
  }

  /**
   * Generate unique trajectory ID
   * @private
   */
  generateTrajectoryId(decision) {
    const data = `${this.swarmId}-${decision.agentId}-${decision.symbol}-${decision.timestamp || Date.now()}`;
    return `traj-${crypto.createHash('sha256').update(data).digest('hex').substring(0, 16)}`;
  }

  /**
   * Clear old trajectories (cleanup)
   *
   * @param {number} maxAge - Maximum age in milliseconds
   * @returns {number} Number of trajectories cleared
   */
  clearOldTrajectories(maxAge = 7 * 24 * 60 * 60 * 1000) { // 7 days default
    const now = Date.now();
    let cleared = 0;

    for (const [trajectoryId, trajectory] of this.trajectories.entries()) {
      if (now - trajectory.createdAt > maxAge && trajectory.status === TrajectoryStatus.LEARNED) {
        this.trajectories.delete(trajectoryId);
        this.stats[trajectory.status]--;
        cleared++;
      }
    }

    return cleared;
  }
}

module.exports = TrajectoryTracker;
module.exports.TrajectoryStatus = TrajectoryStatus;
