/**
 * Reinforcement learning system for adaptive strategy allocation
 * Implements Q-Learning with experience replay
 */

import { AgentDB } from 'agentdb';
import {
  LearningState,
  Experience,
  StrategyPerformance,
  RegimeDetection,
  PortfolioState
} from './types';

export interface LearnerConfig {
  learningRate: number;
  discountFactor: number;
  explorationRate: number;
  explorationDecay: number;
  minExplorationRate: number;
  experienceBufferSize: number;
  batchSize: number;
  updateFrequency: number;
}

export class StrategyLearner {
  private config: LearnerConfig;
  private state: LearningState;
  private agentDB: AgentDB;
  private memoryKey: string;

  constructor(config: Partial<LearnerConfig> = {}, memoryKey = 'strategy-learner') {
    this.config = {
      learningRate: 0.1,
      discountFactor: 0.95,
      explorationRate: 1.0,
      explorationDecay: 0.995,
      minExplorationRate: 0.01,
      experienceBufferSize: 10000,
      batchSize: 32,
      updateFrequency: 100,
      ...config
    };

    this.memoryKey = memoryKey;
    this.state = {
      episodeCount: 0,
      totalReward: 0,
      explorationRate: this.config.explorationRate,
      qTable: new Map(),
      experienceBuffer: [],
      bestPerformance: []
    };

    // Initialize AgentDB for persistent memory
    this.agentDB = new AgentDB({
      storageType: 'hybrid',
      persistence: true,
      persistencePath: './.agentdb/strategy-learner'
    });
  }

  /**
   * Initialize learner from saved state
   */
  async initialize(): Promise<void> {
    try {
      const savedState = await this.agentDB.get(this.memoryKey);
      if (savedState) {
        this.state = {
          ...savedState,
          qTable: new Map(Object.entries(savedState.qTable || {}))
            .set('default', new Map()),
          experienceBuffer: savedState.experienceBuffer || []
        };
        console.log(`📚 Loaded learning state: ${this.state.episodeCount} episodes`);
      }
    } catch (error) {
      console.log('🆕 Starting fresh learning state');
    }
  }

  /**
   * Learn optimal strategy allocation from backtest results
   */
  async learnFromBacktest(
    portfolioStates: PortfolioState[],
    performances: StrategyPerformance[]
  ): Promise<void> {
    console.log('\n🧠 Learning from backtest results...');

    for (let i = 1; i < portfolioStates.length; i++) {
      const prevState = portfolioStates[i - 1];
      const currentState = portfolioStates[i];

      // Encode state
      const stateKey = this.encodeState(prevState);
      const nextStateKey = this.encodeState(currentState);

      // Encode action (strategy allocation)
      const action = this.encodeAction(prevState.strategyWeights);

      // Calculate reward
      const reward = this.calculateReward(prevState, currentState, performances);

      // Store experience
      const experience: Experience = {
        state: stateKey,
        action,
        reward,
        nextState: nextStateKey,
        done: i === portfolioStates.length - 1,
        timestamp: currentState.timestamp
      };

      this.addExperience(experience);

      // Update Q-values
      if (this.state.experienceBuffer.length >= this.config.batchSize) {
        this.updateQValues();
      }
    }

    // Update episode stats
    this.state.episodeCount++;
    this.state.explorationRate = Math.max(
      this.config.minExplorationRate,
      this.state.explorationRate * this.config.explorationDecay
    );

    // Save best performances
    this.updateBestPerformances(performances);

    // Persist state
    await this.saveState();

    console.log(`✅ Episode ${this.state.episodeCount} complete`);
    console.log(`📊 Exploration rate: ${this.state.explorationRate.toFixed(4)}`);
    console.log(`💰 Total reward: ${this.state.totalReward.toFixed(2)}`);
  }

  /**
   * Get optimal strategy weights for current market conditions
   */
  getOptimalWeights(
    currentState: PortfolioState,
    availableStrategies: string[]
  ): Record<string, number> {
    const stateKey = this.encodeState(currentState);

    // Epsilon-greedy exploration
    if (Math.random() < this.state.explorationRate) {
      return this.randomWeights(availableStrategies);
    }

    // Get Q-values for all possible actions
    const stateActions = this.state.qTable.get(stateKey) || new Map();

    if (stateActions.size === 0) {
      return this.randomWeights(availableStrategies);
    }

    // Find best action
    let bestAction = '';
    let bestValue = -Infinity;

    for (const [action, value] of stateActions.entries()) {
      if (value > bestValue) {
        bestValue = value;
        bestAction = action;
      }
    }

    return this.decodeAction(bestAction, availableStrategies);
  }

  /**
   * Update Q-values using experience replay
   */
  private updateQValues(): void {
    // Sample batch from experience buffer
    const batch = this.sampleExperiences(this.config.batchSize);

    for (const exp of batch) {
      const stateActions = this.state.qTable.get(exp.state) || new Map();
      const currentQ = stateActions.get(exp.action) || 0;

      let maxNextQ = 0;
      if (!exp.done) {
        const nextStateActions = this.state.qTable.get(exp.nextState) || new Map();
        maxNextQ = Math.max(...Array.from(nextStateActions.values()), 0);
      }

      // Q-learning update
      const newQ = currentQ + this.config.learningRate * (
        exp.reward + this.config.discountFactor * maxNextQ - currentQ
      );

      stateActions.set(exp.action, newQ);
      this.state.qTable.set(exp.state, stateActions);
    }
  }

  /**
   * Encode portfolio state into discrete representation
   */
  private encodeState(state: PortfolioState): string {
    const equityLevel = this.discretize(state.equity / 1000000, 10); // Millions
    const cashRatio = this.discretize(state.cash / state.equity, 10);
    const positionCount = this.discretize(state.positions.length, 5);
    const regime = state.regime.regime;
    const regimeConfidence = this.discretize(state.regime.confidence, 5);
    const trend = this.discretize(state.regime.indicators.trend, 10);
    const volatility = this.discretize(state.regime.indicators.volatility, 10);

    return `${equityLevel}-${cashRatio}-${positionCount}-${regime}-${regimeConfidence}-${trend}-${volatility}`;
  }

  /**
   * Encode strategy weights into action representation
   */
  private encodeAction(weights: Record<string, number>): string {
    const sortedWeights = Object.entries(weights)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([_, weight]) => this.discretize(weight, 10));
    return sortedWeights.join('-');
  }

  /**
   * Decode action string into strategy weights
   */
  private decodeAction(action: string, strategies: string[]): Record<string, number> {
    const weights: Record<string, number> = {};
    const values = action.split('-').map(Number);

    strategies.forEach((strategy, idx) => {
      weights[strategy] = values[idx] || 0;
    });

    // Normalize weights
    const sum = Object.values(weights).reduce((a, b) => a + b, 0);
    if (sum > 0) {
      for (const key in weights) {
        weights[key] /= sum;
      }
    }

    return weights;
  }

  /**
   * Calculate reward based on portfolio performance
   */
  private calculateReward(
    prevState: PortfolioState,
    currentState: PortfolioState,
    performances: StrategyPerformance[]
  ): number {
    // Return-based reward
    const returns = (currentState.equity - prevState.equity) / prevState.equity;

    // Risk-adjusted reward (Sharpe-like)
    const avgSharpe = performances.reduce((sum, p) => sum + p.sharpeRatio, 0) / performances.length;

    // Diversification bonus
    const activeStrategies = Object.keys(currentState.strategyWeights).length;
    const diversificationBonus = Math.min(activeStrategies / 4, 1) * 0.1;

    // Combined reward
    const reward = returns + (avgSharpe * 0.01) + diversificationBonus;

    this.state.totalReward += reward;
    return reward;
  }

  /**
   * Discretize continuous value into bins
   */
  private discretize(value: number, bins: number): number {
    return Math.max(0, Math.min(bins - 1, Math.floor(value * bins)));
  }

  /**
   * Generate random strategy weights
   */
  private randomWeights(strategies: string[]): Record<string, number> {
    const weights: Record<string, number> = {};
    let sum = 0;

    strategies.forEach(strategy => {
      const weight = Math.random();
      weights[strategy] = weight;
      sum += weight;
    });

    // Normalize
    for (const key in weights) {
      weights[key] /= sum;
    }

    return weights;
  }

  /**
   * Add experience to buffer
   */
  private addExperience(exp: Experience): void {
    this.state.experienceBuffer.push(exp);

    // Limit buffer size
    if (this.state.experienceBuffer.length > this.config.experienceBufferSize) {
      this.state.experienceBuffer.shift();
    }
  }

  /**
   * Sample random batch from experience buffer
   */
  private sampleExperiences(batchSize: number): Experience[] {
    const batch: Experience[] = [];
    const bufferSize = this.state.experienceBuffer.length;

    for (let i = 0; i < Math.min(batchSize, bufferSize); i++) {
      const idx = Math.floor(Math.random() * bufferSize);
      batch.push(this.state.experienceBuffer[idx]);
    }

    return batch;
  }

  /**
   * Update best performances
   */
  private updateBestPerformances(performances: StrategyPerformance[]): void {
    for (const perf of performances) {
      const existing = this.state.bestPerformance.find(p => p.strategyName === perf.strategyName);

      if (!existing || perf.sharpeRatio > existing.sharpeRatio) {
        this.state.bestPerformance = this.state.bestPerformance.filter(
          p => p.strategyName !== perf.strategyName
        );
        this.state.bestPerformance.push(perf);
      }
    }
  }

  /**
   * Save learning state to persistent storage
   */
  private async saveState(): Promise<void> {
    const serializedState = {
      ...this.state,
      qTable: Object.fromEntries(
        Array.from(this.state.qTable.entries()).map(([key, map]) => [
          key,
          Object.fromEntries(map.entries())
        ])
      )
    };

    await this.agentDB.set(this.memoryKey, serializedState);
  }

  /**
   * Get learning statistics
   */
  getStats(): {
    episodes: number;
    totalReward: number;
    explorationRate: number;
    qTableSize: number;
    experienceCount: number;
    bestPerformances: StrategyPerformance[];
  } {
    return {
      episodes: this.state.episodeCount,
      totalReward: this.state.totalReward,
      explorationRate: this.state.explorationRate,
      qTableSize: this.state.qTable.size,
      experienceCount: this.state.experienceBuffer.length,
      bestPerformances: this.state.bestPerformance
    };
  }

  /**
   * Reset learning state
   */
  reset(): void {
    this.state = {
      episodeCount: 0,
      totalReward: 0,
      explorationRate: this.config.explorationRate,
      qTable: new Map(),
      experienceBuffer: [],
      bestPerformance: []
    };
  }
}
