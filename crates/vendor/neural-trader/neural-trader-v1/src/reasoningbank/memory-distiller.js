/**
 * Memory Distiller - Compresses Learning Patterns
 *
 * Distills successful trading patterns from trajectories:
 * - Identifies common success factors across profitable trades
 * - Compresses redundant patterns into templates
 * - Creates reusable strategy templates
 * - Prunes low-value patterns
 * - Aggregates learnings for strategy adaptation
 *
 * @module reasoningbank/pattern-recognizer
 */

const EventEmitter = require('events');
const crypto = require('crypto');

/**
 * Pattern types
 */
const PatternType = {
  MARKET_CONDITION: 'market_condition',    // Market state pattern
  TIMING: 'timing',                        // Entry/exit timing pattern
  RISK_MANAGEMENT: 'risk_management',      // Risk control pattern
  STRATEGY: 'strategy',                    // Trading strategy pattern
  COMPOSITE: 'composite'                   // Multi-factor pattern
};

/**
 * Memory Distiller Class
 */
class MemoryDistiller extends EventEmitter {
  constructor(options = {}) {
    super();

    this.options = {
      minPatternOccurrence: options.minPatternOccurrence || 3,
      similarityThreshold: options.similarityThreshold || 0.8,
      pruneThreshold: options.pruneThreshold || 0.3,
      maxPatternsPerType: options.maxPatternsPerType || 50,
      ...options
    };

    this.patterns = new Map();
    this.agentDB = null;

    this.stats = {
      totalDistilled: 0,
      totalPruned: 0,
      patternsByType: {}
    };
  }

  /**
   * Initialize memory distiller
   *
   * @param {AgentDBClient} agentDB - AgentDB client instance
   * @returns {Promise<void>}
   */
  async initialize(agentDB) {
    this.agentDB = agentDB;

    // Initialize pattern types
    for (const type of Object.values(PatternType)) {
      this.stats.patternsByType[type] = 0;
    }

    console.log('✅ Memory Distiller initialized');
  }

  /**
   * Distill patterns from trajectories
   *
   * @param {Array} trajectories - Trajectories to learn from
   * @param {Object} options - Distillation options
   * @returns {Promise<Array>} Distilled patterns
   */
  async distillPatterns(trajectories, options = {}) {
    if (trajectories.length < this.options.minPatternOccurrence) {
      return [];
    }

    const minScore = options.minScore || 0.7;
    const includeContext = options.includeContext !== false;

    // Filter high-quality trajectories
    const quality Trajectories = trajectories.filter(
      t => t.verdict && t.verdict.score >= minScore
    );

    if (qualityTrajectories.length === 0) {
      return [];
    }

    console.log(`\n🔬 Distilling patterns from ${qualityTrajectories.length} trajectories...`);

    // Extract patterns by type
    const patternsByType = {
      [PatternType.MARKET_CONDITION]: this.extractMarketPatterns(qualityTrajectories),
      [PatternType.TIMING]: this.extractTimingPatterns(qualityTrajectories),
      [PatternType.RISK_MANAGEMENT]: this.extractRiskPatterns(qualityTrajectories),
      [PatternType.STRATEGY]: this.extractStrategyPatterns(qualityTrajectories),
      [PatternType.COMPOSITE]: this.extractCompositePatterns(qualityTrajectories)
    };

    // Merge and deduplicate
    const allPatterns = [];

    for (const [type, patterns] of Object.entries(patternsByType)) {
      for (const pattern of patterns) {
        // Check if similar pattern exists
        const existing = this.findSimilarPattern(pattern, allPatterns);

        if (existing) {
          // Merge with existing
          this.mergePatterns(existing, pattern);
        } else {
          // Add new pattern
          allPatterns.push({
            ...pattern,
            id: this.generatePatternId(pattern),
            type,
            occurrences: 1,
            avgScore: pattern.score || 0.7,
            createdAt: Date.now(),
            updatedAt: Date.now(),
            context: includeContext ? this.extractContext(trajectories, pattern) : null
          });
        }
      }
    }

    // Prune low-value patterns
    const pruned = this.prunePatterns(allPatterns);

    // Store patterns
    for (const pattern of pruned) {
      this.patterns.set(pattern.id, pattern);
      this.stats.totalDistilled++;
      this.stats.patternsByType[pattern.type]++;
    }

    console.log(`  ✅ Distilled ${pruned.length} unique patterns`);

    this.emit('patterns:distilled', { count: pruned.length, patterns: pruned });

    return pruned;
  }

  /**
   * Extract market condition patterns
   * @private
   */
  extractMarketPatterns(trajectories) {
    const patterns = [];
    const marketStates = trajectories.map(t => t.decision.marketState).filter(Boolean);

    if (marketStates.length === 0) return patterns;

    // Group by similar market conditions
    const grouped = this.groupBySimilarity(marketStates, (a, b) => {
      const volatilitySim = 1 - Math.abs((a.volatility || 0) - (b.volatility || 0));
      const trendSim = a.trend === b.trend ? 1 : 0.5;
      return (volatilitySim + trendSim) / 2;
    });

    for (const group of grouped) {
      if (group.length >= this.options.minPatternOccurrence) {
        const avgState = this.averageMarketState(group);
        patterns.push({
          name: 'market_condition_pattern',
          conditions: avgState,
          score: 0.8,
          confidence: group.length / trajectories.length
        });
      }
    }

    return patterns;
  }

  /**
   * Extract timing patterns
   * @private
   */
  extractTimingPatterns(trajectories) {
    const patterns = [];

    // Analyze execution timing
    const timings = trajectories
      .filter(t => t.outcome && t.outcome.executionTime)
      .map(t => ({
        time: new Date(t.decision.timestamp).getHours(),
        executionTime: t.outcome.executionTime,
        slippage: t.outcome.slippage,
        score: t.verdict.score
      }));

    if (timings.length < this.options.minPatternOccurrence) {
      return patterns;
    }

    // Group by hour of day
    const byHour = {};
    for (const timing of timings) {
      const hour = timing.time;
      if (!byHour[hour]) byHour[hour] = [];
      byHour[hour].push(timing);
    }

    // Identify optimal hours
    for (const [hour, entries] of Object.entries(byHour)) {
      if (entries.length >= this.options.minPatternOccurrence) {
        const avgScore = entries.reduce((sum, e) => sum + e.score, 0) / entries.length;
        const avgSlippage = entries.reduce((sum, e) => sum + (e.slippage || 0), 0) / entries.length;

        if (avgScore >= 0.7) {
          patterns.push({
            name: 'optimal_timing_pattern',
            hour: parseInt(hour),
            avgSlippage,
            score: avgScore,
            confidence: entries.length / timings.length
          });
        }
      }
    }

    return patterns;
  }

  /**
   * Extract risk management patterns
   * @private
   */
  extractRiskPatterns(trajectories) {
    const patterns = [];

    // Analyze position sizing
    const positions = trajectories
      .filter(t => t.decision.quantity && t.outcome)
      .map(t => ({
        quantity: t.decision.quantity,
        pnl: t.outcome.pnl,
        riskAdjusted: t.outcome.riskAdjustedReturn,
        score: t.verdict.score
      }));

    if (positions.length < this.options.minPatternOccurrence) {
      return patterns;
    }

    // Find optimal position sizing
    const sorted = positions.sort((a, b) => b.score - a.score);
    const topQuartile = sorted.slice(0, Math.ceil(sorted.length / 4));

    const avgQuantity = topQuartile.reduce((sum, p) => sum + p.quantity, 0) / topQuartile.length;
    const avgRiskAdjusted = topQuartile.reduce((sum, p) => sum + (p.riskAdjusted || 0), 0) / topQuartile.length;

    patterns.push({
      name: 'optimal_position_sizing',
      avgQuantity,
      avgRiskAdjusted,
      score: topQuartile[0].score,
      confidence: topQuartile.length / positions.length
    });

    return patterns;
  }

  /**
   * Extract strategy patterns
   * @private
   */
  extractStrategyPatterns(trajectories) {
    const patterns = [];

    // Group by decision type
    const byType = {};
    for (const traj of trajectories) {
      const type = traj.decision.type;
      if (!byType[type]) byType[type] = [];
      byType[type].push(traj);
    }

    for (const [type, trajs] of Object.entries(byType)) {
      if (trajs.length >= this.options.minPatternOccurrence) {
        const avgScore = trajs.reduce((sum, t) => sum + t.verdict.score, 0) / trajs.length;
        const successRate = trajs.filter(t => t.verdict.score >= 0.7).length / trajs.length;

        if (avgScore >= 0.6) {
          patterns.push({
            name: 'strategy_pattern',
            strategyType: type,
            avgScore,
            successRate,
            score: avgScore,
            confidence: trajs.length / trajectories.length
          });
        }
      }
    }

    return patterns;
  }

  /**
   * Extract composite patterns (multi-factor)
   * @private
   */
  extractCompositePatterns(trajectories) {
    const patterns = [];

    // Analyze combinations of factors
    const successful = trajectories.filter(t => t.verdict.score >= 0.8);

    if (successful.length >= this.options.minPatternOccurrence) {
      // Extract common characteristics
      const commonFactors = this.extractCommonFactors(successful);

      patterns.push({
        name: 'composite_success_pattern',
        factors: commonFactors,
        score: successful.reduce((sum, t) => sum + t.verdict.score, 0) / successful.length,
        confidence: successful.length / trajectories.length
      });
    }

    return patterns;
  }

  /**
   * Aggregate learnings for strategy adaptation
   *
   * @param {Array} learnings - Learning insights
   * @returns {Array} Strategy adjustments
   */
  aggregateLearnings(learnings) {
    const adjustments = [];

    // Group learnings by category
    const byCategory = this.groupLearningsByCategory(learnings);

    for (const [category, items] of Object.entries(byCategory)) {
      const avgConfidence = items.reduce((sum, i) => sum + (i.confidence || 0.5), 0) / items.length;

      if (avgConfidence >= 0.6) {
        adjustments.push({
          category,
          action: this.determineAction(items),
          confidence: avgConfidence,
          evidence: items.length,
          timestamp: Date.now()
        });
      }
    }

    return adjustments;
  }

  /**
   * Find similar pattern in existing patterns
   * @private
   */
  findSimilarPattern(pattern, patterns) {
    for (const existing of patterns) {
      if (existing.type === pattern.type) {
        const similarity = this.calculatePatternSimilarity(pattern, existing);
        if (similarity >= this.options.similarityThreshold) {
          return existing;
        }
      }
    }
    return null;
  }

  /**
   * Merge two similar patterns
   * @private
   */
  mergePatterns(existing, newPattern) {
    existing.occurrences++;
    const totalOccurrences = existing.occurrences;

    // Weighted average of scores
    existing.avgScore = (
      existing.avgScore * (totalOccurrences - 1) +
      (newPattern.score || 0.7)
    ) / totalOccurrences;

    existing.updatedAt = Date.now();
  }

  /**
   * Prune low-value patterns
   * @private
   */
  prunePatterns(patterns) {
    const pruned = patterns.filter(p =>
      p.score >= this.options.pruneThreshold &&
      p.confidence >= 0.3
    );

    this.stats.totalPruned += patterns.length - pruned.length;

    // Limit patterns per type
    const byType = {};
    for (const pattern of pruned) {
      if (!byType[pattern.type]) byType[pattern.type] = [];
      byType[pattern.type].push(pattern);
    }

    const limited = [];
    for (const [type, typePatterns] of Object.entries(byType)) {
      const sorted = typePatterns.sort((a, b) => b.score - a.score);
      limited.push(...sorted.slice(0, this.options.maxPatternsPerType));
    }

    return limited;
  }

  /**
   * Calculate pattern similarity
   * @private
   */
  calculatePatternSimilarity(p1, p2) {
    if (p1.type !== p2.type) return 0;

    // Simple field-based similarity
    let matches = 0;
    let total = 0;

    const fields = Object.keys(p1).filter(k => typeof p1[k] !== 'object');

    for (const field of fields) {
      total++;
      if (p1[field] === p2[field]) {
        matches++;
      } else if (typeof p1[field] === 'number' && typeof p2[field] === 'number') {
        const diff = Math.abs(p1[field] - p2[field]);
        const avg = (p1[field] + p2[field]) / 2;
        if (diff / avg < 0.2) matches += 0.8; // 20% tolerance
      }
    }

    return total > 0 ? matches / total : 0;
  }

  /**
   * Group items by similarity
   * @private
   */
  groupBySimilarity(items, similarityFn) {
    const groups = [];

    for (const item of items) {
      let added = false;

      for (const group of groups) {
        const representative = group[0];
        if (similarityFn(item, representative) >= this.options.similarityThreshold) {
          group.push(item);
          added = true;
          break;
        }
      }

      if (!added) {
        groups.push([item]);
      }
    }

    return groups;
  }

  /**
   * Average market states
   * @private
   */
  averageMarketState(states) {
    const avg = {
      volatility: 0,
      volume: 0,
      trend: null
    };

    let trendCounts = {};

    for (const state of states) {
      avg.volatility += state.volatility || 0;
      avg.volume += state.volume || 0;
      if (state.trend) {
        trendCounts[state.trend] = (trendCounts[state.trend] || 0) + 1;
      }
    }

    avg.volatility /= states.length;
    avg.volume /= states.length;

    // Most common trend
    const trends = Object.entries(trendCounts);
    if (trends.length > 0) {
      avg.trend = trends.sort((a, b) => b[1] - a[1])[0][0];
    }

    return avg;
  }

  /**
   * Extract common factors
   * @private
   */
  extractCommonFactors(trajectories) {
    const factors = [];

    // Check market conditions
    const marketConditions = trajectories.map(t => t.decision.marketState).filter(Boolean);
    if (marketConditions.length > trajectories.length * 0.8) {
      factors.push({ type: 'market_condition', importance: 'high' });
    }

    // Check timing
    const timings = trajectories.map(t => new Date(t.decision.timestamp).getHours());
    const hourCounts = {};
    for (const hour of timings) {
      hourCounts[hour] = (hourCounts[hour] || 0) + 1;
    }
    const dominantHour = Object.entries(hourCounts).sort((a, b) => b[1] - a[1])[0];
    if (dominantHour && dominantHour[1] > trajectories.length * 0.5) {
      factors.push({ type: 'timing', hour: parseInt(dominantHour[0]), importance: 'medium' });
    }

    return factors;
  }

  /**
   * Extract context from trajectories
   * @private
   */
  extractContext(trajectories, pattern) {
    return {
      sampleSize: trajectories.length,
      timeRange: {
        start: Math.min(...trajectories.map(t => t.createdAt)),
        end: Math.max(...trajectories.map(t => t.createdAt))
      },
      agents: [...new Set(trajectories.map(t => t.decision.agentId))].length
    };
  }

  /**
   * Group learnings by category
   * @private
   */
  groupLearningsByCategory(learnings) {
    const grouped = {};

    for (const learning of learnings) {
      const category = learning.type || learning.trajectory?.decision?.type || 'general';
      if (!grouped[category]) grouped[category] = [];
      grouped[category].push(learning);
    }

    return grouped;
  }

  /**
   * Determine action from learnings
   * @private
   */
  determineAction(items) {
    const avgScore = items.reduce((sum, i) => sum + (i.trajectory?.verdict?.score || 0.5), 0) / items.length;

    if (avgScore >= 0.8) {
      return 'increase_usage';
    } else if (avgScore >= 0.6) {
      return 'maintain';
    } else {
      return 'reduce_usage';
    }
  }

  /**
   * Generate pattern ID
   * @private
   */
  generatePatternId(pattern) {
    const data = JSON.stringify(pattern);
    return `pattern-${crypto.createHash('sha256').update(data).digest('hex').substring(0, 16)}`;
  }

  /**
   * Get statistics
   *
   * @returns {Object} Distiller statistics
   */
  getStats() {
    return {
      ...this.stats,
      totalPatterns: this.patterns.size,
      options: { ...this.options }
    };
  }
}

module.exports = MemoryDistiller;
module.exports.PatternType = PatternType;
