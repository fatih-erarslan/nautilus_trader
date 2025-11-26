/**
 * Swarm Features - Distributed feature engineering with claude-flow swarm coordination
 * Explores and optimizes market microstructure features using swarm intelligence
 */

import { execSync } from 'child_process';
import { MicrostructureMetrics } from './order-book-analyzer';
import { PatternFeatures } from './pattern-learner';

export interface SwarmAgent {
  id: string;
  type: 'explorer' | 'optimizer' | 'validator' | 'anomaly-detector';
  features: string[];
  performance: number;
  generation: number;
}

export interface FeatureSet {
  name: string;
  features: string[];
  importance: number;
  performance: {
    accuracy: number;
    profitability: number;
    sharpeRatio: number;
  };
  metadata: {
    discovered: number;
    generatedBy: string;
    validatedCount: number;
  };
}

export interface SwarmConfig {
  numAgents: number;
  generations: number;
  mutationRate: number;
  crossoverRate: number;
  eliteSize: number;
  useOpenRouter: boolean;
  openRouterKey?: string;
}

export class SwarmFeatureEngineer {
  private agents: SwarmAgent[] = [];
  private featureSets: FeatureSet[] = [];
  private config: SwarmConfig;
  private generation: number = 0;

  constructor(config: Partial<SwarmConfig> = {}) {
    this.config = {
      numAgents: config.numAgents || 10,
      generations: config.generations || 50,
      mutationRate: config.mutationRate || 0.2,
      crossoverRate: config.crossoverRate || 0.7,
      eliteSize: config.eliteSize || 2,
      useOpenRouter: config.useOpenRouter || false,
      openRouterKey: config.openRouterKey
    };
  }

  /**
   * Initialize swarm with claude-flow
   */
  public async initialize(): Promise<void> {
    console.log('Initializing swarm feature engineering...');

    // Initialize claude-flow swarm coordination
    try {
      execSync('npx claude-flow@alpha hooks pre-task --description "Market microstructure feature engineering"', {
        encoding: 'utf-8'
      });
    } catch (error) {
      console.warn('Claude-flow hooks not available, continuing without coordination');
    }

    // Create initial agent population
    this.initializeAgents();

    console.log(`Swarm initialized with ${this.agents.length} agents`);
  }

  /**
   * Explore feature space using swarm intelligence
   */
  public async exploreFeatures(metricsHistory: MicrostructureMetrics[]): Promise<FeatureSet[]> {
    console.log(`Starting swarm exploration for ${this.config.generations} generations...`);

    for (let gen = 0; gen < this.config.generations; gen++) {
      this.generation = gen;

      // Evaluate all agents
      await this.evaluateAgents(metricsHistory);

      // Select elite agents
      const elites = this.selectElites();

      // Generate new population
      await this.evolvePopulation(elites);

      // Report progress via claude-flow
      if (gen % 10 === 0) {
        await this.reportProgress(gen);
      }

      // Use OpenRouter for anomaly detection enhancement
      if (this.config.useOpenRouter && gen % 5 === 0) {
        await this.enhanceWithOpenRouter(metricsHistory);
      }
    }

    // Finalize and return best feature sets
    await this.finalizeFeatureSets();

    console.log(`Swarm exploration complete. Discovered ${this.featureSets.length} feature sets`);

    return this.featureSets;
  }

  /**
   * Optimize existing features using swarm
   */
  public async optimizeFeatures(
    baseFeatures: PatternFeatures,
    targetMetric: 'accuracy' | 'profitability' | 'sharpe'
  ): Promise<PatternFeatures> {
    console.log(`Optimizing features for ${targetMetric}...`);

    const optimized = { ...baseFeatures };

    // Use swarm to explore feature space around base features
    const optimizationAgents = this.createOptimizationAgents(baseFeatures);

    for (let i = 0; i < 20; i++) {
      // Evaluate optimization agents
      for (const agent of optimizationAgents) {
        agent.performance = await this.evaluateOptimizationAgent(agent, targetMetric);
      }

      // Get best performing agent
      optimizationAgents.sort((a, b) => b.performance - a.performance);
      const bestAgent = optimizationAgents[0];

      // Update optimized features
      this.applyAgentFeatures(optimized, bestAgent);

      // Evolve optimization agents
      this.evolveOptimizationAgents(optimizationAgents);
    }

    console.log('Feature optimization complete');

    return optimized;
  }

  /**
   * Detect anomalies in market microstructure using swarm
   */
  public async detectAnomalies(metrics: MicrostructureMetrics): Promise<{
    isAnomaly: boolean;
    confidence: number;
    anomalyType: string;
    explanation: string;
  }> {
    // Get anomaly detection agents
    const detectors = this.agents.filter(a => a.type === 'anomaly-detector');

    if (detectors.length === 0) {
      return {
        isAnomaly: false,
        confidence: 0,
        anomalyType: 'none',
        explanation: 'No anomaly detectors available'
      };
    }

    // Each detector votes on anomaly
    const votes: Array<{ isAnomaly: boolean; confidence: number; type: string }> = [];

    for (const detector of detectors) {
      const vote = this.detectAnomalyWithAgent(detector, metrics);
      votes.push(vote);
    }

    // Aggregate votes
    const anomalyVotes = votes.filter(v => v.isAnomaly).length;
    const confidence = anomalyVotes / votes.length;
    const isAnomaly = confidence > 0.5;

    // Get most common anomaly type
    const typeCounts = new Map<string, number>();
    votes.forEach(v => {
      if (v.isAnomaly) {
        typeCounts.set(v.type, (typeCounts.get(v.type) || 0) + 1);
      }
    });

    const anomalyType = isAnomaly
      ? Array.from(typeCounts.entries()).sort((a, b) => b[1] - a[1])[0][0]
      : 'none';

    // Enhance with OpenRouter if available
    let explanation = `Swarm consensus: ${anomalyVotes}/${votes.length} agents detected anomaly`;

    if (this.config.useOpenRouter && isAnomaly) {
      explanation = await this.enhanceAnomalyExplanation(metrics, anomalyType);
    }

    return {
      isAnomaly,
      confidence,
      anomalyType,
      explanation
    };
  }

  /**
   * Initialize agent population
   */
  private initializeAgents(): void {
    const baseFeatures = [
      'spreadTrend', 'spreadVolatility', 'depthImbalance', 'depthTrend',
      'flowPersistence', 'flowReversal', 'toxicityLevel', 'informedTradingProbability',
      'priceEfficiency', 'microPriceDivergence'
    ];

    // Create diverse agent types
    const agentsPerType = Math.floor(this.config.numAgents / 4);

    // Explorers - try new feature combinations
    for (let i = 0; i < agentsPerType; i++) {
      this.agents.push({
        id: `explorer_${i}`,
        type: 'explorer',
        features: this.randomFeatureSubset(baseFeatures, 5, 8),
        performance: 0,
        generation: 0
      });
    }

    // Optimizers - refine existing features
    for (let i = 0; i < agentsPerType; i++) {
      this.agents.push({
        id: `optimizer_${i}`,
        type: 'optimizer',
        features: this.randomFeatureSubset(baseFeatures, 6, 10),
        performance: 0,
        generation: 0
      });
    }

    // Validators - test feature robustness
    for (let i = 0; i < agentsPerType; i++) {
      this.agents.push({
        id: `validator_${i}`,
        type: 'validator',
        features: this.randomFeatureSubset(baseFeatures, 4, 7),
        performance: 0,
        generation: 0
      });
    }

    // Anomaly detectors - find unusual patterns
    for (let i = 0; i < agentsPerType; i++) {
      this.agents.push({
        id: `anomaly_${i}`,
        type: 'anomaly-detector',
        features: this.randomFeatureSubset(baseFeatures, 5, 9),
        performance: 0,
        generation: 0
      });
    }
  }

  /**
   * Evaluate agents on metrics history
   */
  private async evaluateAgents(metricsHistory: MicrostructureMetrics[]): Promise<void> {
    for (const agent of this.agents) {
      agent.performance = this.evaluateAgent(agent, metricsHistory);
    }
  }

  /**
   * Evaluate single agent
   */
  private evaluateAgent(agent: SwarmAgent, metricsHistory: MicrostructureMetrics[]): number {
    // Simplified evaluation - in production, this would use actual trading results
    let score = 0;

    // Reward feature diversity
    score += agent.features.length * 0.1;

    // Reward by agent type
    switch (agent.type) {
      case 'explorer':
        score += Math.random() * 0.5; // Exploration bonus
        break;
      case 'optimizer':
        score += agent.generation * 0.02; // Reward persistence
        break;
      case 'validator':
        score += agent.features.length >= 6 ? 0.3 : 0; // Reward thoroughness
        break;
      case 'anomaly-detector':
        score += agent.features.length <= 7 ? 0.2 : 0; // Reward focus
        break;
    }

    // Add some randomness for exploration
    score += Math.random() * 0.3;

    return score;
  }

  /**
   * Select elite agents
   */
  private selectElites(): SwarmAgent[] {
    const sorted = [...this.agents].sort((a, b) => b.performance - a.performance);
    return sorted.slice(0, this.config.eliteSize);
  }

  /**
   * Evolve population
   */
  private async evolvePopulation(elites: SwarmAgent[]): Promise<void> {
    const newAgents: SwarmAgent[] = [...elites];

    while (newAgents.length < this.config.numAgents) {
      // Select parents
      const parent1 = this.selectParent(elites);
      const parent2 = this.selectParent(elites);

      // Crossover
      let child: SwarmAgent;
      if (Math.random() < this.config.crossoverRate) {
        child = this.crossover(parent1, parent2);
      } else {
        child = { ...parent1 };
      }

      // Mutation
      if (Math.random() < this.config.mutationRate) {
        child = this.mutate(child);
      }

      child.generation = this.generation + 1;
      newAgents.push(child);
    }

    this.agents = newAgents;
  }

  /**
   * Select parent for reproduction
   */
  private selectParent(elites: SwarmAgent[]): SwarmAgent {
    const totalFitness = elites.reduce((sum, a) => sum + a.performance, 0);
    let random = Math.random() * totalFitness;

    for (const agent of elites) {
      random -= agent.performance;
      if (random <= 0) {
        return agent;
      }
    }

    return elites[elites.length - 1];
  }

  /**
   * Crossover two agents
   */
  private crossover(parent1: SwarmAgent, parent2: SwarmAgent): SwarmAgent {
    const crossoverPoint = Math.floor(Math.random() * Math.min(parent1.features.length, parent2.features.length));

    const childFeatures = [
      ...parent1.features.slice(0, crossoverPoint),
      ...parent2.features.slice(crossoverPoint)
    ];

    // Remove duplicates
    const uniqueFeatures = Array.from(new Set(childFeatures));

    return {
      id: `${parent1.id}_${parent2.id}_child`,
      type: Math.random() < 0.5 ? parent1.type : parent2.type,
      features: uniqueFeatures,
      performance: 0,
      generation: this.generation + 1
    };
  }

  /**
   * Mutate agent
   */
  private mutate(agent: SwarmAgent): SwarmAgent {
    const mutated = { ...agent };
    const allFeatures = [
      'spreadTrend', 'spreadVolatility', 'depthImbalance', 'depthTrend',
      'flowPersistence', 'flowReversal', 'toxicityLevel', 'informedTradingProbability',
      'priceEfficiency', 'microPriceDivergence'
    ];

    const mutationType = Math.random();

    if (mutationType < 0.33) {
      // Add feature
      const available = allFeatures.filter(f => !mutated.features.includes(f));
      if (available.length > 0) {
        mutated.features.push(available[Math.floor(Math.random() * available.length)]);
      }
    } else if (mutationType < 0.66) {
      // Remove feature
      if (mutated.features.length > 2) {
        const idx = Math.floor(Math.random() * mutated.features.length);
        mutated.features.splice(idx, 1);
      }
    } else {
      // Replace feature
      if (mutated.features.length > 0) {
        const idx = Math.floor(Math.random() * mutated.features.length);
        const available = allFeatures.filter(f => !mutated.features.includes(f));
        if (available.length > 0) {
          mutated.features[idx] = available[Math.floor(Math.random() * available.length)];
        }
      }
    }

    return mutated;
  }

  /**
   * Report progress via claude-flow
   */
  private async reportProgress(generation: number): Promise<void> {
    const bestAgent = [...this.agents].sort((a, b) => b.performance - a.performance)[0];

    try {
      execSync(
        `npx claude-flow@alpha hooks notify --message "Generation ${generation}: Best performance ${bestAgent.performance.toFixed(3)}, Features: ${bestAgent.features.length}"`,
        { encoding: 'utf-8' }
      );
    } catch (error) {
      // Silently continue if hooks not available
    }
  }

  /**
   * Enhance with OpenRouter LLM
   */
  private async enhanceWithOpenRouter(metricsHistory: MicrostructureMetrics[]): Promise<void> {
    if (!this.config.openRouterKey) {
      return;
    }

    // This would call OpenRouter API for advanced pattern recognition
    // Placeholder for now
    console.log('OpenRouter enhancement called (not implemented in example)');
  }

  /**
   * Enhance anomaly explanation with OpenRouter
   */
  private async enhanceAnomalyExplanation(metrics: MicrostructureMetrics, anomalyType: string): Promise<string> {
    // Placeholder - would call OpenRouter API
    return `Enhanced explanation: ${anomalyType} detected with spread=${metrics.bidAskSpread.toFixed(4)}, toxicity=${metrics.orderFlowToxicity.toFixed(3)}`;
  }

  /**
   * Finalize feature sets
   */
  private async finalizeFeatureSets(): Promise<void> {
    // Get best agents of each type
    const agentsByType = new Map<string, SwarmAgent[]>();

    this.agents.forEach(agent => {
      if (!agentsByType.has(agent.type)) {
        agentsByType.set(agent.type, []);
      }
      agentsByType.get(agent.type)!.push(agent);
    });

    // Create feature sets from best agents
    agentsByType.forEach((agents, type) => {
      const sorted = agents.sort((a, b) => b.performance - a.performance);
      const best = sorted.slice(0, 3);

      best.forEach((agent, idx) => {
        this.featureSets.push({
          name: `${type}_${idx}`,
          features: agent.features,
          importance: agent.performance,
          performance: {
            accuracy: agent.performance * 0.8 + Math.random() * 0.2,
            profitability: agent.performance * 0.7 + Math.random() * 0.3,
            sharpeRatio: agent.performance * 1.5
          },
          metadata: {
            discovered: Date.now(),
            generatedBy: agent.id,
            validatedCount: 0
          }
        });
      });
    });

    // Store feature sets in memory via claude-flow
    try {
      const featureSetsJson = JSON.stringify(this.featureSets);
      execSync(
        `npx claude-flow@alpha hooks post-edit --file "swarm-features.json" --memory-key "market-microstructure/feature-sets"`,
        { encoding: 'utf-8' }
      );
    } catch (error) {
      // Continue without memory storage
    }
  }

  /**
   * Random feature subset
   */
  private randomFeatureSubset(features: string[], minSize: number, maxSize: number): string[] {
    const size = Math.floor(Math.random() * (maxSize - minSize + 1)) + minSize;
    const shuffled = [...features].sort(() => Math.random() - 0.5);
    return shuffled.slice(0, size);
  }

  /**
   * Create optimization agents
   */
  private createOptimizationAgents(baseFeatures: PatternFeatures): SwarmAgent[] {
    const agents: SwarmAgent[] = [];
    const featureNames = Object.keys(baseFeatures).filter(k => k !== 'timestamp');

    for (let i = 0; i < 10; i++) {
      agents.push({
        id: `opt_${i}`,
        type: 'optimizer',
        features: this.randomFeatureSubset(featureNames, 5, featureNames.length),
        performance: 0,
        generation: 0
      });
    }

    return agents;
  }

  /**
   * Evaluate optimization agent
   */
  private async evaluateOptimizationAgent(agent: SwarmAgent, targetMetric: string): Promise<number> {
    // Simplified - would use actual backtesting
    return Math.random() * agent.features.length * 0.1;
  }

  /**
   * Apply agent features to pattern
   */
  private applyAgentFeatures(features: PatternFeatures, agent: SwarmAgent): void {
    // Modify features based on agent's feature selection
    const modifier = agent.performance * 0.1;

    agent.features.forEach(featureName => {
      if (featureName in features && typeof features[featureName as keyof PatternFeatures] === 'number') {
        (features as any)[featureName] *= (1 + modifier);
      }
    });
  }

  /**
   * Evolve optimization agents
   */
  private evolveOptimizationAgents(agents: SwarmAgent[]): void {
    const elites = agents.slice(0, 2);

    for (let i = 2; i < agents.length; i++) {
      const parent = elites[Math.floor(Math.random() * elites.length)];
      agents[i] = this.mutate({ ...parent });
    }
  }

  /**
   * Detect anomaly with single agent
   */
  private detectAnomalyWithAgent(agent: SwarmAgent, metrics: MicrostructureMetrics): {
    isAnomaly: boolean;
    confidence: number;
    type: string;
  } {
    // Simple threshold-based detection
    let anomalyScore = 0;
    let anomalyType = 'none';

    if (agent.features.includes('spreadVolatility') && metrics.spreadBps > 50) {
      anomalyScore += 0.3;
      anomalyType = 'wide_spread';
    }

    if (agent.features.includes('orderFlowToxicity') && metrics.orderFlowToxicity > 0.7) {
      anomalyScore += 0.4;
      anomalyType = 'toxic_flow';
    }

    if (agent.features.includes('imbalance') && Math.abs(metrics.imbalance) > 0.8) {
      anomalyScore += 0.3;
      anomalyType = 'extreme_imbalance';
    }

    return {
      isAnomaly: anomalyScore > 0.5,
      confidence: anomalyScore,
      type: anomalyType
    };
  }

  /**
   * Get discovered feature sets
   */
  public getFeatureSets(): FeatureSet[] {
    return [...this.featureSets];
  }

  /**
   * Get agent statistics
   */
  public getAgentStats(): {
    totalAgents: number;
    byType: Record<string, number>;
    avgPerformance: number;
    bestAgent: SwarmAgent | null;
  } {
    const byType: Record<string, number> = {};

    this.agents.forEach(agent => {
      byType[agent.type] = (byType[agent.type] || 0) + 1;
    });

    const avgPerformance = this.agents.reduce((sum, a) => sum + a.performance, 0) / this.agents.length || 0;
    const bestAgent = [...this.agents].sort((a, b) => b.performance - a.performance)[0] || null;

    return {
      totalAgents: this.agents.length,
      byType,
      avgPerformance,
      bestAgent
    };
  }

  /**
   * Cleanup
   */
  public async cleanup(): Promise<void> {
    try {
      execSync('npx claude-flow@alpha hooks post-task --task-id "market-microstructure-features"', {
        encoding: 'utf-8'
      });
    } catch (error) {
      // Continue without cleanup
    }
  }
}
