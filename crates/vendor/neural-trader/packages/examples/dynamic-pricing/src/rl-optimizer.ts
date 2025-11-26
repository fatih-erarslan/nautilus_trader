/**
 * Reinforcement learning optimizer for dynamic pricing
 * Supports Q-Learning, DQN, PPO, SARSA, and Actor-Critic
 */

import { MarketContext, RLState, RLAction, RLExperience, RLConfig, RLAlgorithm } from './types';

export class RLOptimizer {
  private config: RLConfig;
  private qTable: Map<string, Map<number, number>>; // Q-Learning/SARSA
  private replayMemory: RLExperience[]; // DQN
  private policyNetwork: Map<string, number[]>; // PPO/Actor-Critic
  private valueNetwork: Map<string, number>; // Actor-Critic
  private actions: RLAction[];
  private step: number;

  constructor(config: Partial<RLConfig> = {}) {
    this.config = {
      algorithm: 'q-learning',
      learningRate: 0.1,
      discountFactor: 0.95,
      epsilon: 0.3,
      epsilonDecay: 0.995,
      minEpsilon: 0.05,
      batchSize: 32,
      memorySize: 10000,
      targetUpdateFreq: 100,
      ...config,
    };

    this.qTable = new Map();
    this.replayMemory = [];
    this.policyNetwork = new Map();
    this.valueNetwork = new Map();
    this.step = 0;

    this.initializeActions();
  }

  /**
   * Initialize discrete price actions
   */
  private initializeActions(): void {
    this.actions = [];

    // Create price multipliers from 0.7 to 1.3 (70% to 130% of base)
    const steps = 13;
    for (let i = 0; i < steps; i++) {
      const multiplier = 0.7 + (i * 0.6) / (steps - 1);
      this.actions.push({
        priceMultiplier: multiplier,
        index: i,
      });
    }
  }

  /**
   * Convert market context to RL state
   */
  private contextToState(context: MarketContext): RLState {
    const competitorAvg = context.competitorPrices.length > 0
      ? context.competitorPrices.reduce((a, b) => a + b, 0) / context.competitorPrices.length
      : 0;

    // Time features
    const hourNorm = context.hour / 24;
    const dayNorm = context.dayOfWeek / 7;

    // Normalize features to [0, 1]
    const normalized = [
      context.demand / 200, // Assume max demand = 200
      context.inventory / 500, // Assume max inventory = 500
      competitorAvg / 200, // Assume max price = 200
      hourNorm,
      dayNorm,
      context.seasonality,
      context.isPromotion ? 1 : 0,
      context.isHoliday ? 1 : 0,
    ];

    return {
      price: 0, // Will be set by action
      demand: context.demand,
      inventory: context.inventory,
      competitorAvgPrice: competitorAvg,
      timeFeatures: [hourNorm, dayNorm, context.seasonality],
      normalized,
    };
  }

  /**
   * Convert state to string key for Q-table
   */
  private stateToKey(state: RLState): string {
    // Discretize continuous state for Q-table
    const discretized = state.normalized.map(v => Math.floor(v * 10) / 10);
    return discretized.join(',');
  }

  /**
   * Select action using epsilon-greedy policy
   */
  selectAction(context: MarketContext, explore: boolean = true): RLAction {
    const state = this.contextToState(context);

    // Epsilon-greedy exploration
    if (explore && Math.random() < this.config.epsilon) {
      return this.actions[Math.floor(Math.random() * this.actions.length)];
    }

    // Exploitation: choose best action based on algorithm
    switch (this.config.algorithm) {
      case 'q-learning':
      case 'sarsa':
        return this.selectQLearningAction(state);
      case 'dqn':
        return this.selectDQNAction(state);
      case 'ppo':
      case 'actor-critic':
        return this.selectPolicyAction(state);
      default:
        return this.actions[Math.floor(this.actions.length / 2)]; // Middle action
    }
  }

  /**
   * Q-Learning action selection
   */
  private selectQLearningAction(state: RLState): RLAction {
    const stateKey = this.stateToKey(state);
    const qValues = this.qTable.get(stateKey);

    if (!qValues) {
      // Initialize Q-values for new state
      const newQValues = new Map<number, number>();
      this.actions.forEach(a => newQValues.set(a.index, 0));
      this.qTable.set(stateKey, newQValues);
      return this.actions[Math.floor(this.actions.length / 2)];
    }

    // Find action with max Q-value
    let maxQ = -Infinity;
    let bestAction = this.actions[0];

    for (const action of this.actions) {
      const q = qValues.get(action.index) || 0;
      if (q > maxQ) {
        maxQ = q;
        bestAction = action;
      }
    }

    return bestAction;
  }

  /**
   * DQN action selection
   */
  private selectDQNAction(state: RLState): RLAction {
    // Simplified DQN: use Q-table approximation
    return this.selectQLearningAction(state);
  }

  /**
   * Policy network action selection (PPO/Actor-Critic)
   */
  private selectPolicyAction(state: RLState): RLAction {
    const stateKey = this.stateToKey(state);
    const policy = this.policyNetwork.get(stateKey);

    if (!policy) {
      // Initialize uniform policy
      const uniformProb = 1.0 / this.actions.length;
      const newPolicy = new Array(this.actions.length).fill(uniformProb);
      this.policyNetwork.set(stateKey, newPolicy);
      return this.actions[Math.floor(this.actions.length / 2)];
    }

    // Sample action from policy distribution
    const random = Math.random();
    let cumProb = 0;

    for (let i = 0; i < policy.length; i++) {
      cumProb += policy[i];
      if (random <= cumProb) {
        return this.actions[i];
      }
    }

    return this.actions[policy.length - 1];
  }

  /**
   * Learn from experience
   */
  learn(
    context: MarketContext,
    action: RLAction,
    reward: number,
    nextContext: MarketContext
  ): void {
    const state = this.contextToState(context);
    const nextState = this.contextToState(nextContext);

    const experience: RLExperience = {
      state,
      action,
      reward,
      nextState,
      done: false,
    };

    // Add to replay memory for DQN
    if (this.config.algorithm === 'dqn') {
      this.replayMemory.push(experience);
      if (this.replayMemory.length > this.config.memorySize!) {
        this.replayMemory.shift();
      }
    }

    // Update based on algorithm
    switch (this.config.algorithm) {
      case 'q-learning':
        this.updateQLearning(experience);
        break;
      case 'sarsa':
        this.updateSARSA(experience, nextContext);
        break;
      case 'dqn':
        if (this.replayMemory.length >= this.config.batchSize!) {
          this.updateDQN();
        }
        break;
      case 'ppo':
        this.updatePPO(experience);
        break;
      case 'actor-critic':
        this.updateActorCritic(experience);
        break;
    }

    // Decay epsilon
    this.config.epsilon = Math.max(
      this.config.minEpsilon,
      this.config.epsilon * this.config.epsilonDecay
    );

    this.step++;
  }

  /**
   * Q-Learning update
   */
  private updateQLearning(experience: RLExperience): void {
    const stateKey = this.stateToKey(experience.state);
    const nextStateKey = this.stateToKey(experience.nextState);

    // Get or initialize Q-values
    if (!this.qTable.has(stateKey)) {
      const qValues = new Map<number, number>();
      this.actions.forEach(a => qValues.set(a.index, 0));
      this.qTable.set(stateKey, qValues);
    }

    const qValues = this.qTable.get(stateKey)!;
    const currentQ = qValues.get(experience.action.index) || 0;

    // Get max Q-value for next state
    let maxNextQ = 0;
    if (this.qTable.has(nextStateKey)) {
      const nextQValues = this.qTable.get(nextStateKey)!;
      maxNextQ = Math.max(...Array.from(nextQValues.values()));
    }

    // Q-Learning update: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
    const newQ = currentQ + this.config.learningRate * (
      experience.reward + this.config.discountFactor * maxNextQ - currentQ
    );

    qValues.set(experience.action.index, newQ);
  }

  /**
   * SARSA update
   */
  private updateSARSA(experience: RLExperience, nextContext: MarketContext): void {
    const stateKey = this.stateToKey(experience.state);
    const nextStateKey = this.stateToKey(experience.nextState);

    if (!this.qTable.has(stateKey)) {
      const qValues = new Map<number, number>();
      this.actions.forEach(a => qValues.set(a.index, 0));
      this.qTable.set(stateKey, qValues);
    }

    const qValues = this.qTable.get(stateKey)!;
    const currentQ = qValues.get(experience.action.index) || 0;

    // Get Q-value for next action (SARSA: on-policy)
    const nextAction = this.selectAction(nextContext, true);
    let nextQ = 0;
    if (this.qTable.has(nextStateKey)) {
      const nextQValues = this.qTable.get(nextStateKey)!;
      nextQ = nextQValues.get(nextAction.index) || 0;
    }

    // SARSA update: Q(s,a) = Q(s,a) + α[r + γ*Q(s',a') - Q(s,a)]
    const newQ = currentQ + this.config.learningRate * (
      experience.reward + this.config.discountFactor * nextQ - currentQ
    );

    qValues.set(experience.action.index, newQ);
  }

  /**
   * DQN update with experience replay
   */
  private updateDQN(): void {
    // Sample random batch from replay memory
    const batchSize = Math.min(this.config.batchSize!, this.replayMemory.length);
    const batch: RLExperience[] = [];

    for (let i = 0; i < batchSize; i++) {
      const idx = Math.floor(Math.random() * this.replayMemory.length);
      batch.push(this.replayMemory[idx]);
    }

    // Update Q-values for batch
    for (const exp of batch) {
      this.updateQLearning(exp);
    }
  }

  /**
   * PPO update
   */
  private updatePPO(experience: RLExperience): void {
    const stateKey = this.stateToKey(experience.state);

    // Get or initialize policy
    if (!this.policyNetwork.has(stateKey)) {
      const uniformProb = 1.0 / this.actions.length;
      this.policyNetwork.set(stateKey, new Array(this.actions.length).fill(uniformProb));
    }

    const policy = this.policyNetwork.get(stateKey)!;

    // Simple policy gradient update
    // Increase probability of good actions, decrease bad ones
    const actionIdx = experience.action.index;
    const advantage = experience.reward; // Simplified

    // Update policy with gradient ascent
    policy[actionIdx] += this.config.learningRate * advantage;

    // Normalize probabilities
    const sum = policy.reduce((a, b) => Math.max(a + b, 0.01), 0);
    for (let i = 0; i < policy.length; i++) {
      policy[i] = Math.max(policy[i], 0.01) / sum;
    }
  }

  /**
   * Actor-Critic update
   */
  private updateActorCritic(experience: RLExperience): void {
    const stateKey = this.stateToKey(experience.state);
    const nextStateKey = this.stateToKey(experience.nextState);

    // Update value function (critic)
    const currentValue = this.valueNetwork.get(stateKey) || 0;
    const nextValue = this.valueNetwork.get(nextStateKey) || 0;

    // TD error
    const tdError = experience.reward + this.config.discountFactor * nextValue - currentValue;

    // Update value
    this.valueNetwork.set(
      stateKey,
      currentValue + this.config.learningRate * tdError
    );

    // Update policy (actor) using TD error as advantage
    this.updatePPO({ ...experience, reward: tdError });
  }

  /**
   * Get performance metrics
   */
  getMetrics(): {
    epsilon: number;
    statesExplored: number;
    avgQValue: number;
    step: number;
  } {
    let totalQ = 0;
    let count = 0;

    for (const qValues of this.qTable.values()) {
      for (const q of qValues.values()) {
        totalQ += q;
        count++;
      }
    }

    return {
      epsilon: this.config.epsilon,
      statesExplored: this.qTable.size,
      avgQValue: count > 0 ? totalQ / count : 0,
      step: this.step,
    };
  }

  /**
   * Export learned policy
   */
  exportPolicy(): Map<string, RLAction> {
    const policy = new Map<string, RLAction>();

    for (const [stateKey, qValues] of this.qTable) {
      let maxQ = -Infinity;
      let bestAction = this.actions[0];

      for (const action of this.actions) {
        const q = qValues.get(action.index) || 0;
        if (q > maxQ) {
          maxQ = q;
          bestAction = action;
        }
      }

      policy.set(stateKey, bestAction);
    }

    return policy;
  }
}
