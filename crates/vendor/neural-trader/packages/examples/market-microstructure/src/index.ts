/**
 * @neural-trader/example-market-microstructure
 *
 * Self-learning market microstructure analysis with swarm-based feature engineering
 *
 * Key capabilities:
 * - Real-time order flow analysis
 * - Market impact modeling
 * - Price discovery patterns
 * - Liquidity provision optimization
 * - Self-learning bid-ask spread prediction
 * - Swarm-based feature engineering with claude-flow
 * - AgentDB pattern persistence
 */

export {
  OrderBookAnalyzer,
  OrderBook,
  OrderBookLevel,
  OrderFlow,
  MicrostructureMetrics
} from './order-book-analyzer';

export {
  PatternLearner,
  Pattern,
  PatternFeatures,
  LearningConfig
} from './pattern-learner';

export {
  SwarmFeatureEngineer,
  SwarmAgent,
  FeatureSet,
  SwarmConfig
} from './swarm-features';

import { OrderBookAnalyzer, OrderBook, OrderFlow } from './order-book-analyzer';
import { PatternLearner } from './pattern-learner';
import { SwarmFeatureEngineer } from './swarm-features';

/**
 * Main MarketMicrostructure class - orchestrates all components
 */
export class MarketMicrostructure {
  private analyzer: OrderBookAnalyzer;
  private learner: PatternLearner;
  private swarm?: SwarmFeatureEngineer;
  private initialized: boolean = false;

  constructor(
    private config: {
      agentDbPath?: string;
      useSwarm?: boolean;
      swarmConfig?: {
        numAgents?: number;
        generations?: number;
        useOpenRouter?: boolean;
        openRouterKey?: string;
      };
    } = {}
  ) {
    this.analyzer = new OrderBookAnalyzer();
    this.learner = new PatternLearner({
      agentDbPath: config.agentDbPath || './market-patterns.db'
    });

    if (config.useSwarm !== false) {
      this.swarm = new SwarmFeatureEngineer(config.swarmConfig || {});
    }
  }

  /**
   * Initialize all components
   */
  public async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }

    await this.learner.initialize();

    if (this.swarm) {
      await this.swarm.initialize();
    }

    this.initialized = true;
  }

  /**
   * Analyze order book and learn patterns
   */
  public async analyze(orderBook: OrderBook, recentTrades?: OrderFlow[]): Promise<{
    metrics: any;
    pattern: any;
    anomaly?: any;
  }> {
    if (!this.initialized) {
      throw new Error('MarketMicrostructure not initialized. Call initialize() first.');
    }

    // Analyze order book
    const metrics = this.analyzer.analyzeOrderBook(orderBook, recentTrades);

    // Extract features from metrics history
    const metricsHistory = this.analyzer.getMetricsHistory();
    let pattern = null;

    if (metricsHistory.length >= 2) {
      const features = this.learner.extractFeatures(metricsHistory);

      // Try to recognize existing pattern
      pattern = await this.learner.recognizePattern(features);

      // If no pattern recognized, predict outcome
      if (!pattern) {
        const prediction = await this.learner.predictOutcome(features);
        pattern = { features, prediction };
      }
    }

    // Check for anomalies using swarm
    let anomaly;
    if (this.swarm) {
      anomaly = await this.swarm.detectAnomalies(metrics);
    }

    return {
      metrics,
      pattern,
      anomaly
    };
  }

  /**
   * Learn from observed outcome
   */
  public async learn(
    outcome: {
      priceMove: number;
      spreadChange: number;
      liquidityChange: number;
      timeHorizon: number;
    },
    label?: string
  ): Promise<void> {
    const metricsHistory = this.analyzer.getMetricsHistory();

    if (metricsHistory.length < 2) {
      return;
    }

    const features = this.learner.extractFeatures(metricsHistory);
    await this.learner.learnPattern(features, outcome, label);
  }

  /**
   * Explore feature space using swarm
   */
  public async exploreFeatures(): Promise<any[]> {
    if (!this.swarm) {
      throw new Error('Swarm not enabled. Set useSwarm: true in constructor.');
    }

    const metricsHistory = this.analyzer.getMetricsHistory();

    if (metricsHistory.length < 10) {
      throw new Error('Need at least 10 metrics points for feature exploration');
    }

    return await this.swarm.exploreFeatures(metricsHistory);
  }

  /**
   * Optimize features for a specific metric
   */
  public async optimizeFeatures(targetMetric: 'accuracy' | 'profitability' | 'sharpe'): Promise<any> {
    if (!this.swarm) {
      throw new Error('Swarm not enabled. Set useSwarm: true in constructor.');
    }

    const metricsHistory = this.analyzer.getMetricsHistory();

    if (metricsHistory.length < 2) {
      throw new Error('Need at least 2 metrics points for optimization');
    }

    const baseFeatures = this.learner.extractFeatures(metricsHistory);
    return await this.swarm.optimizeFeatures(baseFeatures, targetMetric);
  }

  /**
   * Get comprehensive statistics
   */
  public getStatistics(): {
    analyzer: {
      metricsCount: number;
      orderFlowCount: number;
    };
    learner: any;
    swarm?: any;
  } {
    const stats: any = {
      analyzer: {
        metricsCount: this.analyzer.getMetricsHistory().length,
        orderFlowCount: this.analyzer.getOrderFlowHistory().length
      },
      learner: this.learner.getStatistics()
    };

    if (this.swarm) {
      stats.swarm = this.swarm.getAgentStats();
    }

    return stats;
  }

  /**
   * Get learned patterns
   */
  public getPatterns(): any[] {
    return this.learner.getPatterns();
  }

  /**
   * Get discovered feature sets
   */
  public getFeatureSets(): any[] {
    if (!this.swarm) {
      return [];
    }

    return this.swarm.getFeatureSets();
  }

  /**
   * Reset analyzer state
   */
  public reset(): void {
    this.analyzer.reset();
  }

  /**
   * Cleanup and close all components
   */
  public async close(): Promise<void> {
    await this.learner.close();

    if (this.swarm) {
      await this.swarm.cleanup();
    }

    this.initialized = false;
  }
}

/**
 * Convenience function to create and initialize MarketMicrostructure instance
 */
export async function createMarketMicrostructure(
  config?: ConstructorParameters<typeof MarketMicrostructure>[0]
): Promise<MarketMicrostructure> {
  const mm = new MarketMicrostructure(config);
  await mm.initialize();
  return mm;
}

// Default export
export default MarketMicrostructure;
