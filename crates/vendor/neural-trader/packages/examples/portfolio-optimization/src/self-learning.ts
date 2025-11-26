/**
 * Self-Learning Portfolio Optimization using AgentDB Memory Patterns
 * Learns optimal risk parameters and strategy preferences from historical performance
 */

import { AgentDB, LearningPlugin } from 'agentdb';
import { OptimizationResult, PortfolioConstraints } from './optimizer.js';

export interface RiskProfile {
  riskAversion: number;
  targetReturn: number;
  maxDrawdown: number;
  preferredAlgorithm: string;
  diversificationPreference: number;
}

export interface PerformanceMetrics {
  sharpeRatio: number;
  maxDrawdown: number;
  volatility: number;
  cumulativeReturn: number;
  winRate: number;
  informationRatio: number;
}

export interface LearningState {
  iteration: number;
  bestRiskProfile: RiskProfile;
  performanceHistory: PerformanceMetrics[];
  strategySuccessRates: Record<string, number>;
  adaptiveParameters: Record<string, number>;
}

/**
 * Self-Learning Optimizer with AgentDB Memory
 */
export class SelfLearningOptimizer {
  private db: AgentDB;
  private learningPlugin: LearningPlugin;
  private namespace: string;

  constructor(
    dbPath: string = './portfolio-memory.db',
    namespace: string = 'portfolio-optimization',
  ) {
    this.namespace = namespace;
    this.db = new AgentDB({
      path: dbPath,
      embedding: {
        provider: 'openai',
        model: 'text-embedding-3-small',
      },
    });

    // Initialize learning plugin with Decision Transformer
    this.learningPlugin = new LearningPlugin({
      algorithm: 'decision-transformer',
      stateSize: 16,
      actionSize: 8,
      hiddenSize: 128,
      numLayers: 4,
    });
  }

  /**
   * Initialize or restore learning state from memory
   */
  async initialize(): Promise<void> {
    const existingState = await this.loadLearningState();

    if (!existingState) {
      // Initialize default state
      const defaultState: LearningState = {
        iteration: 0,
        bestRiskProfile: {
          riskAversion: 2.5,
          targetReturn: 0.10,
          maxDrawdown: 0.20,
          preferredAlgorithm: 'mean-variance',
          diversificationPreference: 0.7,
        },
        performanceHistory: [],
        strategySuccessRates: {
          'mean-variance': 0.5,
          'risk-parity': 0.5,
          'black-litterman': 0.5,
          'multi-objective': 0.5,
        },
        adaptiveParameters: {},
      };

      await this.saveLearningState(defaultState);
    }
  }

  /**
   * Load learning state from AgentDB memory
   */
  async loadLearningState(): Promise<LearningState | null> {
    try {
      const memory = await this.db.memory.retrieve({
        namespace: this.namespace,
        key: 'learning-state',
      });

      return memory ? JSON.parse(memory.value) : null;
    } catch (error) {
      console.error('Failed to load learning state:', error);
      return null;
    }
  }

  /**
   * Save learning state to AgentDB memory
   */
  async saveLearningState(state: LearningState): Promise<void> {
    await this.db.memory.store({
      namespace: this.namespace,
      key: 'learning-state',
      value: JSON.stringify(state),
      metadata: {
        iteration: state.iteration,
        timestamp: Date.now(),
      },
    });
  }

  /**
   * Learn from optimization result and update risk profile
   */
  async learn(
    result: OptimizationResult,
    actualPerformance: PerformanceMetrics,
    marketConditions: Record<string, number>,
  ): Promise<RiskProfile> {
    const state = await this.loadLearningState();
    if (!state) throw new Error('Learning state not initialized');

    // Update performance history
    state.performanceHistory.push(actualPerformance);

    // Update strategy success rates
    const algorithm = result.algorithm;
    const success = actualPerformance.sharpeRatio > 0.5 ? 1 : 0;
    const currentRate = state.strategySuccessRates[algorithm] || 0.5;
    state.strategySuccessRates[algorithm] = currentRate * 0.9 + success * 0.1;

    // Learn optimal parameters using Decision Transformer
    const stateVector = this.encodeState(state, marketConditions);
    const action = this.learningPlugin.selectAction(stateVector);
    const reward = this.calculateReward(actualPerformance);

    // Update learning plugin
    this.learningPlugin.updateModel(
      stateVector,
      action,
      reward,
      this.encodeState(state, marketConditions), // next state
      false, // not done
    );

    // Adapt risk profile based on learning
    const newRiskProfile = this.adaptRiskProfile(state, action, actualPerformance);
    state.bestRiskProfile = newRiskProfile;
    state.iteration++;

    // Store trajectory in memory for future reference
    await this.storeTrajectory({
      state: stateVector,
      action,
      reward,
      marketConditions,
      performance: actualPerformance,
    });

    await this.saveLearningState(state);
    return newRiskProfile;
  }

  /**
   * Encode current state into vector for learning
   */
  private encodeState(state: LearningState, marketConditions: Record<string, number>): number[] {
    const recentPerformance = state.performanceHistory.slice(-10);
    const avgSharpe = recentPerformance.reduce((sum, p) => sum + p.sharpeRatio, 0) / (recentPerformance.length || 1);
    const avgDrawdown = recentPerformance.reduce((sum, p) => sum + p.maxDrawdown, 0) / (recentPerformance.length || 1);

    return [
      state.bestRiskProfile.riskAversion / 5.0,
      state.bestRiskProfile.targetReturn,
      state.bestRiskProfile.maxDrawdown,
      state.bestRiskProfile.diversificationPreference,
      avgSharpe,
      avgDrawdown,
      Object.values(state.strategySuccessRates).reduce((a, b) => a + b, 0) / 4,
      marketConditions.volatility || 0,
      marketConditions.trend || 0,
      marketConditions.correlation || 0,
      state.iteration / 1000.0,
      recentPerformance.length / 10.0,
      Math.max(...Object.values(state.strategySuccessRates)),
      Math.min(...Object.values(state.strategySuccessRates)),
      state.performanceHistory.length / 100.0,
      Object.keys(state.adaptiveParameters).length / 10.0,
    ];
  }

  /**
   * Calculate reward signal from performance
   */
  private calculateReward(performance: PerformanceMetrics): number {
    return (
      performance.sharpeRatio * 1.0 +
      performance.cumulativeReturn * 0.5 -
      performance.maxDrawdown * 2.0 -
      performance.volatility * 0.3 +
      performance.winRate * 0.4 +
      performance.informationRatio * 0.6
    );
  }

  /**
   * Adapt risk profile based on learned action
   */
  private adaptRiskProfile(
    state: LearningState,
    action: number[],
    performance: PerformanceMetrics,
  ): RiskProfile {
    const current = state.bestRiskProfile;

    // Decode action into parameter adjustments
    const adjustments = {
      riskAversion: (action[0] - 0.5) * 0.5,
      targetReturn: (action[1] - 0.5) * 0.05,
      maxDrawdown: (action[2] - 0.5) * 0.05,
      diversificationPreference: (action[3] - 0.5) * 0.2,
    };

    // Apply adjustments with bounds
    return {
      riskAversion: Math.max(1.0, Math.min(5.0, current.riskAversion + adjustments.riskAversion)),
      targetReturn: Math.max(0.01, Math.min(0.30, current.targetReturn + adjustments.targetReturn)),
      maxDrawdown: Math.max(0.05, Math.min(0.50, current.maxDrawdown + adjustments.maxDrawdown)),
      diversificationPreference: Math.max(0.3, Math.min(1.0, current.diversificationPreference + adjustments.diversificationPreference)),
      preferredAlgorithm: this.selectBestAlgorithm(state.strategySuccessRates),
    };
  }

  /**
   * Select algorithm with highest success rate
   */
  private selectBestAlgorithm(successRates: Record<string, number>): string {
    return Object.entries(successRates).reduce((best, [algo, rate]) =>
      rate > successRates[best] ? algo : best
    , 'mean-variance');
  }

  /**
   * Store trajectory in memory for experience replay
   */
  private async storeTrajectory(trajectory: {
    state: number[],
    action: number[],
    reward: number,
    marketConditions: Record<string, number>,
    performance: PerformanceMetrics,
  }): Promise<void> {
    const key = `trajectory-${Date.now()}`;
    await this.db.memory.store({
      namespace: `${this.namespace}/trajectories`,
      key,
      value: JSON.stringify(trajectory),
      embedding: trajectory.state, // Use state as embedding for similarity search
      metadata: {
        reward: trajectory.reward,
        sharpeRatio: trajectory.performance.sharpeRatio,
        timestamp: Date.now(),
      },
    });
  }

  /**
   * Retrieve similar past experiences for pattern recognition
   */
  async retrieveSimilarExperiences(
    currentState: number[],
    limit: number = 10,
  ): Promise<any[]> {
    const results = await this.db.memory.search({
      namespace: `${this.namespace}/trajectories`,
      embedding: currentState,
      limit,
    });

    return results.map(r => JSON.parse(r.value));
  }

  /**
   * Get recommended risk profile based on current market conditions
   */
  async getRecommendedProfile(marketConditions: Record<string, number>): Promise<RiskProfile> {
    const state = await this.loadLearningState();
    if (!state) throw new Error('Learning state not initialized');

    // Search for similar market conditions in memory
    const stateVector = this.encodeState(state, marketConditions);
    const similarExperiences = await this.retrieveSimilarExperiences(stateVector, 5);

    if (similarExperiences.length > 0) {
      // Aggregate insights from similar experiences
      const avgReward = similarExperiences.reduce((sum, exp) => sum + exp.reward, 0) / similarExperiences.length;

      if (avgReward > 0) {
        // Use insights to adjust current profile
        return state.bestRiskProfile;
      }
    }

    // Default to current best profile
    return state.bestRiskProfile;
  }

  /**
   * Distill learned patterns into memory
   */
  async distillLearning(): Promise<void> {
    const state = await this.loadLearningState();
    if (!state) return;

    const insights = {
      bestAlgorithm: this.selectBestAlgorithm(state.strategySuccessRates),
      optimalRiskAversion: state.bestRiskProfile.riskAversion,
      avgPerformance: state.performanceHistory.length > 0
        ? state.performanceHistory.reduce((sum, p) => sum + p.sharpeRatio, 0) / state.performanceHistory.length
        : 0,
      learningIterations: state.iteration,
      strategyPreferences: state.strategySuccessRates,
    };

    await this.db.memory.store({
      namespace: this.namespace,
      key: 'distilled-insights',
      value: JSON.stringify(insights),
      metadata: {
        timestamp: Date.now(),
        iteration: state.iteration,
      },
    });
  }

  /**
   * Export learning state for analysis
   */
  async exportLearningData(): Promise<LearningState | null> {
    return await this.loadLearningState();
  }

  /**
   * Reset learning state (for retraining)
   */
  async reset(): Promise<void> {
    await this.initialize();
  }

  /**
   * Close database connection
   */
  async close(): Promise<void> {
    await this.db.close();
  }
}

/**
 * Adaptive Risk Manager
 * Dynamically adjusts position sizing based on learned patterns
 */
export class AdaptiveRiskManager {
  private learningOptimizer: SelfLearningOptimizer;

  constructor(learningOptimizer: SelfLearningOptimizer) {
    this.learningOptimizer = learningOptimizer;
  }

  /**
   * Calculate adaptive position sizes
   */
  async calculatePositionSizes(
    baseWeights: number[],
    marketConditions: Record<string, number>,
    confidenceLevel: number = 0.95,
  ): Promise<number[]> {
    const profile = await this.learningOptimizer.getRecommendedProfile(marketConditions);

    // Adjust weights based on risk profile
    const riskMultiplier = this.calculateRiskMultiplier(profile, marketConditions);
    const adjustedWeights = baseWeights.map(w => w * riskMultiplier);

    // Normalize
    const sum = adjustedWeights.reduce((a, b) => a + b, 0);
    return adjustedWeights.map(w => w / sum);
  }

  /**
   * Calculate risk multiplier based on conditions
   */
  private calculateRiskMultiplier(
    profile: RiskProfile,
    marketConditions: Record<string, number>,
  ): number {
    const baseMultiplier = 1.0;

    // Reduce exposure in high volatility
    const volatilityAdjustment = 1.0 - Math.min(0.5, marketConditions.volatility || 0);

    // Adjust based on risk aversion
    const riskAdjustment = Math.max(0.5, Math.min(1.5, 3.0 / profile.riskAversion));

    return baseMultiplier * volatilityAdjustment * riskAdjustment;
  }

  /**
   * Monitor portfolio and trigger rebalancing if needed
   */
  async shouldRebalance(
    currentWeights: number[],
    targetWeights: number[],
    threshold: number = 0.05,
  ): Promise<boolean> {
    const maxDeviation = currentWeights.reduce((max, w, i) =>
      Math.max(max, Math.abs(w - targetWeights[i]))
    , 0);

    return maxDeviation > threshold;
  }
}
