/**
 * Core dynamic pricing engine with multiple strategies
 */

import {
  MarketContext,
  PricingStrategy,
  PriceRecommendation,
  CustomerSegment,
  CompetitorAnalysis,
} from './types';
import { ElasticityLearner } from './elasticity-learner';
import { RLOptimizer } from './rl-optimizer';
import { CompetitiveAnalyzer } from './competitive-analyzer';

export class DynamicPricer {
  private strategies: Map<string, PricingStrategy>;
  private elasticityLearner: ElasticityLearner;
  private rlOptimizer: RLOptimizer;
  private competitiveAnalyzer: CompetitiveAnalyzer;
  private basePrice: number;
  private priceHistory: Array<{ price: number; demand: number; revenue: number; timestamp: number }>;

  constructor(
    basePrice: number,
    elasticityLearner: ElasticityLearner,
    rlOptimizer: RLOptimizer,
    competitiveAnalyzer: CompetitiveAnalyzer
  ) {
    this.basePrice = basePrice;
    this.elasticityLearner = elasticityLearner;
    this.rlOptimizer = rlOptimizer;
    this.competitiveAnalyzer = competitiveAnalyzer;
    this.strategies = new Map();
    this.priceHistory = [];

    this.initializeStrategies();
  }

  private initializeStrategies(): void {
    // Cost-plus pricing
    this.strategies.set('cost-plus', {
      name: 'cost-plus',
      calculate: (context: MarketContext, basePrice: number) => {
        const margin = 0.3; // 30% margin
        return basePrice * (1 + margin);
      },
    });

    // Value-based pricing
    this.strategies.set('value-based', {
      name: 'value-based',
      calculate: (context: MarketContext, basePrice: number) => {
        const valueMultiplier = context.isPromotion ? 0.9 : 1.1;
        const seasonalityAdj = 1 + context.seasonality * 0.2;
        return basePrice * valueMultiplier * seasonalityAdj;
      },
    });

    // Competition-based pricing
    this.strategies.set('competition-based', {
      name: 'competition-based',
      calculate: (context: MarketContext, basePrice: number) => {
        if (context.competitorPrices.length === 0) return basePrice;
        const avgCompetitorPrice =
          context.competitorPrices.reduce((a, b) => a + b, 0) / context.competitorPrices.length;
        const position = 0.95; // Slightly undercut competitors
        return avgCompetitorPrice * position;
      },
    });

    // Dynamic demand-based pricing
    this.strategies.set('dynamic-demand', {
      name: 'dynamic-demand',
      calculate: (context: MarketContext, basePrice: number) => {
        const demandFactor = Math.min(Math.max(context.demand / 100, 0.5), 2.0);
        const inventoryFactor = Math.max(0.7, Math.min(1.3, 1 - (context.inventory - 100) / 500));
        return basePrice * demandFactor * inventoryFactor;
      },
    });

    // Peak/Off-peak pricing
    this.strategies.set('time-based', {
      name: 'time-based',
      calculate: (context: MarketContext, basePrice: number) => {
        const isPeak = context.hour >= 9 && context.hour <= 17;
        const isWeekend = context.dayOfWeek === 0 || context.dayOfWeek === 6;

        let multiplier = 1.0;
        if (isPeak && !isWeekend) multiplier = 1.2;
        else if (isWeekend) multiplier = 1.15;
        else multiplier = 0.85;

        return basePrice * multiplier;
      },
    });

    // Elasticity-optimized pricing
    this.strategies.set('elasticity-optimized', {
      name: 'elasticity-optimized',
      calculate: (context: MarketContext, basePrice: number) => {
        const elasticity = this.elasticityLearner.getElasticity(context);

        // Optimal markup = 1 / (1 + elasticity)
        // More elastic = lower markup, less elastic = higher markup
        const optimalMarkup = 1 / (1 + Math.abs(elasticity.mean));
        return basePrice * (1 + optimalMarkup);
      },
    });

    // RL-optimized pricing
    this.strategies.set('rl-optimized', {
      name: 'rl-optimized',
      calculate: (context: MarketContext, basePrice: number) => {
        const action = this.rlOptimizer.selectAction(context, false);
        return basePrice * action.priceMultiplier;
      },
    });
  }

  /**
   * Get price recommendation using specified strategy
   */
  async recommendPrice(
    context: MarketContext,
    strategyName?: string,
    segment?: CustomerSegment
  ): Promise<PriceRecommendation> {
    // If no strategy specified, use ensemble approach
    if (!strategyName) {
      return this.ensembleRecommendation(context, segment);
    }

    const strategy = this.strategies.get(strategyName);
    if (!strategy) {
      throw new Error(`Unknown pricing strategy: ${strategyName}`);
    }

    const price = strategy.calculate(context, this.basePrice);

    // Estimate demand using elasticity
    const elasticity = this.elasticityLearner.getElasticity(context);
    const priceChange = (price - this.basePrice) / this.basePrice;
    const expectedDemandChange = elasticity.mean * priceChange;
    const expectedDemand = context.demand * (1 + expectedDemandChange);

    // Calculate expected revenue
    const expectedRevenue = price * Math.max(0, expectedDemand);

    // Get competitive analysis
    const competitiveAnalysis = this.competitiveAnalyzer.analyze(context.competitorPrices);
    const competitivePosition = this.getCompetitivePosition(price, competitiveAnalysis);

    // Calculate uncertainty bounds using elasticity std
    const demandUncertainty = elasticity.std * Math.abs(priceChange) * context.demand;
    const lowerBound = Math.max(0, expectedDemand - 1.96 * demandUncertainty);
    const upperBound = expectedDemand + 1.96 * demandUncertainty;

    return {
      price,
      strategy: strategyName,
      expectedRevenue,
      expectedDemand,
      confidence: elasticity.confidence,
      uncertaintyBounds: [lowerBound, upperBound],
      competitivePosition,
    };
  }

  /**
   * Ensemble recommendation combining multiple strategies
   */
  private async ensembleRecommendation(
    context: MarketContext,
    segment?: CustomerSegment
  ): Promise<PriceRecommendation> {
    const recommendations: PriceRecommendation[] = [];

    // Get recommendations from all strategies
    for (const [name, strategy] of this.strategies) {
      const rec = await this.recommendPrice(context, name, segment);
      recommendations.push(rec);
    }

    // Weight strategies based on recent performance
    const weights = this.calculateStrategyWeights();

    // Weighted average
    let weightedPrice = 0;
    let weightedRevenue = 0;
    let totalWeight = 0;

    for (const rec of recommendations) {
      const weight = weights.get(rec.strategy) || 1.0;
      weightedPrice += rec.price * weight;
      weightedRevenue += rec.expectedRevenue * weight;
      totalWeight += weight;
    }

    const avgPrice = weightedPrice / totalWeight;
    const avgRevenue = weightedRevenue / totalWeight;

    // Aggregate uncertainty
    const allBounds = recommendations.flatMap(r => r.uncertaintyBounds);
    const avgLower = allBounds.filter((_, i) => i % 2 === 0).reduce((a, b) => a + b, 0) / recommendations.length;
    const avgUpper = allBounds.filter((_, i) => i % 2 === 1).reduce((a, b) => a + b, 0) / recommendations.length;

    const competitiveAnalysis = this.competitiveAnalyzer.analyze(context.competitorPrices);

    return {
      price: avgPrice,
      strategy: 'ensemble',
      expectedRevenue: avgRevenue,
      expectedDemand: avgRevenue / avgPrice,
      confidence: 0.85,
      uncertaintyBounds: [avgLower, avgUpper],
      competitivePosition: this.getCompetitivePosition(avgPrice, competitiveAnalysis),
    };
  }

  /**
   * Calculate strategy weights based on historical performance
   */
  private calculateStrategyWeights(): Map<string, number> {
    const weights = new Map<string, number>();

    // Initialize with equal weights
    for (const name of this.strategies.keys()) {
      weights.set(name, 1.0);
    }

    // Adjust based on recent performance (last 100 observations)
    const recentHistory = this.priceHistory.slice(-100);
    if (recentHistory.length === 0) return weights;

    // Simple performance metric: revenue/price ratio
    const avgRevenue = recentHistory.reduce((sum, h) => sum + h.revenue, 0) / recentHistory.length;

    // Boost RL and elasticity strategies if they perform well
    if (avgRevenue > this.basePrice * 50) {
      weights.set('rl-optimized', 1.5);
      weights.set('elasticity-optimized', 1.3);
    }

    return weights;
  }

  /**
   * Determine competitive position
   */
  private getCompetitivePosition(price: number, analysis: CompetitorAnalysis): string {
    if (price < analysis.minPrice) return 'aggressive-low';
    if (price < analysis.avgPrice * 0.95) return 'competitive-low';
    if (price < analysis.avgPrice * 1.05) return 'market-rate';
    if (price < analysis.maxPrice) return 'premium';
    return 'ultra-premium';
  }

  /**
   * Record actual outcome for learning
   */
  recordOutcome(price: number, demand: number, context: MarketContext): void {
    const revenue = price * demand;

    this.priceHistory.push({
      price,
      demand,
      revenue,
      timestamp: context.timestamp,
    });

    // Update learners
    this.elasticityLearner.observe(price, demand, context);

    // Update RL optimizer
    const reward = this.calculateReward(revenue, demand, price, context);
    this.rlOptimizer.learn(context, { priceMultiplier: price / this.basePrice, index: 0 }, reward, context);
  }

  /**
   * Calculate reward for RL
   */
  private calculateReward(revenue: number, demand: number, price: number, context: MarketContext): number {
    // Multi-objective reward:
    // 1. Revenue (primary)
    // 2. Demand (secondary - avoid pricing out)
    // 3. Inventory management
    // 4. Competitive position

    const revenueScore = revenue / (this.basePrice * 100); // Normalize
    const demandScore = Math.min(demand / 100, 1.0); // Encourage demand
    const inventoryScore = context.inventory > 50 ? 0.1 : -0.1; // Penalize excess inventory

    return revenueScore * 0.6 + demandScore * 0.3 + inventoryScore * 0.1;
  }

  /**
   * Get strategy by name
   */
  getStrategy(name: string): PricingStrategy | undefined {
    return this.strategies.get(name);
  }

  /**
   * Add custom strategy
   */
  addStrategy(strategy: PricingStrategy): void {
    this.strategies.set(strategy.name, strategy);
  }

  /**
   * Get performance metrics
   */
  getPerformanceMetrics(): {
    avgRevenue: number;
    avgPrice: number;
    avgDemand: number;
    totalRevenue: number;
  } {
    if (this.priceHistory.length === 0) {
      return { avgRevenue: 0, avgPrice: 0, avgDemand: 0, totalRevenue: 0 };
    }

    const totalRevenue = this.priceHistory.reduce((sum, h) => sum + h.revenue, 0);
    const avgRevenue = totalRevenue / this.priceHistory.length;
    const avgPrice = this.priceHistory.reduce((sum, h) => sum + h.price, 0) / this.priceHistory.length;
    const avgDemand = this.priceHistory.reduce((sum, h) => sum + h.demand, 0) / this.priceHistory.length;

    return { avgRevenue, avgPrice, avgDemand, totalRevenue };
  }
}
