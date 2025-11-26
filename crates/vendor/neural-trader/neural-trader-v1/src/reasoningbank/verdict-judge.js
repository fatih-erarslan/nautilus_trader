/**
 * Verdict Judge - Evaluates Trading Decision Quality
 *
 * Multi-factor evaluation system that assigns quality scores to trading decisions:
 * - P&L outcome (profitability)
 * - Risk-adjusted returns (Sharpe ratio consideration)
 * - Timing accuracy (entry/exit quality)
 * - Market condition alignment (did conditions match expectations?)
 * - Reasoning quality (was the decision logic sound?)
 *
 * Outputs a verdict score (0-1) and quality classification.
 *
 * @module reasoningbank/verdict-judge
 */

const EventEmitter = require('events');

/**
 * Quality classifications
 */
const VerdictQuality = {
  EXCELLENT: 'excellent',    // Score >= 0.9
  GOOD: 'good',             // Score >= 0.7
  NEUTRAL: 'neutral',       // Score >= 0.4
  POOR: 'poor',             // Score >= 0.2
  TERRIBLE: 'terrible'      // Score < 0.2
};

/**
 * Evaluation weights for different factors
 */
const DefaultWeights = {
  pnl: 0.35,                    // Profit/loss outcome
  riskAdjusted: 0.25,           // Risk-adjusted performance
  timing: 0.20,                 // Entry/exit timing quality
  marketAlignment: 0.15,        // Market condition match
  reasoning: 0.05               // Decision logic quality
};

/**
 * Verdict Judge Class
 */
class VerdictJudge extends EventEmitter {
  constructor(options = {}) {
    super();

    this.weights = {
      ...DefaultWeights,
      ...options.weights
    };

    // Thresholds for quality classification
    this.thresholds = {
      excellent: 0.9,
      good: 0.7,
      neutral: 0.4,
      poor: 0.2
    };

    this.stats = {
      totalEvaluations: 0,
      avgScore: 0,
      qualityDistribution: {
        excellent: 0,
        good: 0,
        neutral: 0,
        poor: 0,
        terrible: 0
      }
    };
  }

  /**
   * Initialize verdict judge
   *
   * @returns {Promise<void>}
   */
  async initialize() {
    console.log('✅ Verdict Judge initialized');
  }

  /**
   * Evaluate a trajectory and assign verdict
   *
   * @param {Object} trajectory - Trajectory to evaluate
   * @returns {Promise<Object>} Verdict result
   */
  async evaluate(trajectory) {
    if (!trajectory.outcome) {
      throw new Error('Cannot evaluate trajectory without outcome');
    }

    const { decision, outcome } = trajectory;

    // Calculate individual factor scores
    const pnlScore = this.evaluatePnL(outcome);
    const riskAdjustedScore = this.evaluateRiskAdjusted(outcome);
    const timingScore = this.evaluateTiming(decision, outcome);
    const marketAlignmentScore = this.evaluateMarketAlignment(decision, outcome);
    const reasoningScore = this.evaluateReasoning(decision);

    // Calculate weighted total score
    const totalScore = (
      pnlScore * this.weights.pnl +
      riskAdjustedScore * this.weights.riskAdjusted +
      timingScore * this.weights.timing +
      marketAlignmentScore * this.weights.marketAlignment +
      reasoningScore * this.weights.reasoning
    );

    // Determine quality classification
    const quality = this.classifyQuality(totalScore);

    // Build detailed analysis
    const analysis = {
      factors: {
        pnl: { score: pnlScore, weight: this.weights.pnl, weighted: pnlScore * this.weights.pnl },
        riskAdjusted: { score: riskAdjustedScore, weight: this.weights.riskAdjusted, weighted: riskAdjustedScore * this.weights.riskAdjusted },
        timing: { score: timingScore, weight: this.weights.timing, weighted: timingScore * this.weights.timing },
        marketAlignment: { score: marketAlignmentScore, weight: this.weights.marketAlignment, weighted: marketAlignmentScore * this.weights.marketAlignment },
        reasoning: { score: reasoningScore, weight: this.weights.reasoning, weighted: reasoningScore * this.weights.reasoning }
      },
      keyFactors: this.identifyKeyFactors({
        pnl: pnlScore,
        riskAdjusted: riskAdjustedScore,
        timing: timingScore,
        marketAlignment: marketAlignmentScore,
        reasoning: reasoningScore
      }),
      strengths: this.identifyStrengths({
        pnl: pnlScore,
        riskAdjusted: riskAdjustedScore,
        timing: timingScore,
        marketAlignment: marketAlignmentScore,
        reasoning: reasoningScore
      }),
      weaknesses: this.identifyWeaknesses({
        pnl: pnlScore,
        riskAdjusted: riskAdjustedScore,
        timing: timingScore,
        marketAlignment: marketAlignmentScore,
        reasoning: reasoningScore
      })
    };

    const verdict = {
      score: totalScore,
      quality,
      analysis,
      timestamp: Date.now()
    };

    // Update stats
    this.updateStats(totalScore, quality);

    this.emit('verdict:evaluated', { trajectoryId: trajectory.id, verdict });

    return verdict;
  }

  /**
   * Evaluate P&L outcome
   * @private
   */
  evaluatePnL(outcome) {
    if (!outcome.executed) {
      return 0.5; // Neutral for non-executed trades
    }

    const pnlPercent = outcome.pnlPercent || 0;

    // Sigmoid-like scoring: exceptional gains get high scores, losses get low scores
    if (pnlPercent > 0) {
      // Gains: 0% = 0.6, 5% = 0.85, 10%+ = 1.0
      return Math.min(1.0, 0.6 + (pnlPercent / 10) * 0.4);
    } else {
      // Losses: 0% = 0.4, -5% = 0.15, -10%- = 0.0
      return Math.max(0.0, 0.4 + (pnlPercent / 10) * 0.4);
    }
  }

  /**
   * Evaluate risk-adjusted returns
   * @private
   */
  evaluateRiskAdjusted(outcome) {
    if (!outcome.riskAdjustedReturn) {
      return this.evaluatePnL(outcome) * 0.9; // Fallback to PnL with penalty
    }

    const rarPercent = outcome.riskAdjustedReturn || 0;

    // Risk-adjusted returns should be higher than raw returns for good decisions
    if (rarPercent > 0) {
      return Math.min(1.0, 0.6 + (rarPercent / 8) * 0.4);
    } else {
      return Math.max(0.0, 0.4 + (rarPercent / 8) * 0.4);
    }
  }

  /**
   * Evaluate timing quality
   * @private
   */
  evaluateTiming(decision, outcome) {
    if (!outcome.executed) {
      return 0.5;
    }

    // Calculate slippage impact
    const slippage = outcome.slippage || 0;

    // Good timing = low slippage
    // Slippage of 0% = 1.0, 0.5% = 0.75, 1%+ = 0.5
    let timingScore = 1.0 - Math.min(1.0, Math.abs(slippage) * 50);

    // Consider execution time if available
    if (outcome.executionTime && decision.expectedExecutionTime) {
      const timeDiff = Math.abs(outcome.executionTime - decision.expectedExecutionTime);
      const timeScore = 1.0 - Math.min(1.0, timeDiff / 5000); // 5s tolerance
      timingScore = (timingScore + timeScore) / 2;
    }

    return Math.max(0.0, timingScore);
  }

  /**
   * Evaluate market condition alignment
   * @private
   */
  evaluateMarketAlignment(decision, outcome) {
    const marketState = decision.marketState || {};

    // Check if market moved as expected
    let alignmentScore = 0.5; // Default neutral

    if (marketState.expectedDirection && outcome.pnlPercent !== undefined) {
      const directionMatch = (
        (marketState.expectedDirection === 'up' && outcome.pnlPercent > 0) ||
        (marketState.expectedDirection === 'down' && outcome.pnlPercent < 0)
      );

      alignmentScore = directionMatch ? 0.8 : 0.3;
    }

    // Adjust for volatility expectations
    if (marketState.expectedVolatility && marketState.actualVolatility) {
      const volDiff = Math.abs(marketState.expectedVolatility - marketState.actualVolatility);
      const volScore = 1.0 - Math.min(1.0, volDiff);
      alignmentScore = (alignmentScore + volScore) / 2;
    }

    return alignmentScore;
  }

  /**
   * Evaluate reasoning quality
   * @private
   */
  evaluateReasoning(decision) {
    const reasoning = decision.reasoning || {};

    let reasoningScore = 0.5; // Default neutral

    // Check for structured reasoning
    if (reasoning.factors && Array.isArray(reasoning.factors)) {
      reasoningScore += 0.2 * Math.min(1.0, reasoning.factors.length / 3);
    }

    // Check for confidence level
    if (reasoning.confidence !== undefined) {
      reasoningScore += 0.1 * reasoning.confidence;
    }

    // Check for risk assessment
    if (reasoning.riskLevel) {
      reasoningScore += 0.1;
    }

    // Check for alternative scenarios
    if (reasoning.alternatives && reasoning.alternatives.length > 0) {
      reasoningScore += 0.1;
    }

    return Math.min(1.0, reasoningScore);
  }

  /**
   * Classify quality based on score
   * @private
   */
  classifyQuality(score) {
    if (score >= this.thresholds.excellent) return VerdictQuality.EXCELLENT;
    if (score >= this.thresholds.good) return VerdictQuality.GOOD;
    if (score >= this.thresholds.neutral) return VerdictQuality.NEUTRAL;
    if (score >= this.thresholds.poor) return VerdictQuality.POOR;
    return VerdictQuality.TERRIBLE;
  }

  /**
   * Identify key factors that drove the verdict
   * @private
   */
  identifyKeyFactors(scores) {
    const factors = Object.entries(scores)
      .map(([name, score]) => ({ name, score, weight: this.weights[name] }))
      .sort((a, b) => (b.score * b.weight) - (a.score * a.weight));

    return factors.slice(0, 3).map(f => f.name);
  }

  /**
   * Identify strengths
   * @private
   */
  identifyStrengths(scores) {
    return Object.entries(scores)
      .filter(([_, score]) => score >= 0.7)
      .map(([name, score]) => ({ factor: name, score }));
  }

  /**
   * Identify weaknesses
   * @private
   */
  identifyWeaknesses(scores) {
    return Object.entries(scores)
      .filter(([_, score]) => score < 0.4)
      .map(([name, score]) => ({ factor: name, score }));
  }

  /**
   * Update statistics
   * @private
   */
  updateStats(score, quality) {
    const currentAvg = this.stats.avgScore;
    const currentTotal = this.stats.totalEvaluations;

    this.stats.totalEvaluations++;
    this.stats.avgScore = (currentAvg * currentTotal + score) / this.stats.totalEvaluations;
    this.stats.qualityDistribution[quality]++;
  }

  /**
   * Get statistics
   *
   * @returns {Object} Judge statistics
   */
  getStats() {
    return {
      ...this.stats,
      weights: { ...this.weights }
    };
  }
}

module.exports = VerdictJudge;
module.exports.VerdictQuality = VerdictQuality;
module.exports.DefaultWeights = DefaultWeights;
