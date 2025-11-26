/**
 * Self-learning price elasticity estimation with AgentDB memory
 */

import { AgentDB } from 'agentdb';
import { MarketContext, ElasticityEstimate, Purchase } from './types';

export class ElasticityLearner {
  private db: AgentDB;
  private observations: Array<{
    price: number;
    demand: number;
    context: MarketContext;
    timestamp: number;
  }>;
  private segmentElasticities: Map<string, number[]>;

  constructor(dbPath: string = './data/elasticity.db') {
    this.db = new AgentDB({ path: dbPath });
    this.observations = [];
    this.segmentElasticities = new Map();
  }

  /**
   * Observe a price-demand pair with context
   */
  async observe(price: number, demand: number, context: MarketContext): Promise<void> {
    const observation = {
      price,
      demand,
      context,
      timestamp: Date.now(),
    };

    this.observations.push(observation);

    // Store in AgentDB for persistent memory
    await this.db.set(`observation_${observation.timestamp}`, observation);

    // Update rolling elasticity estimate
    if (this.observations.length >= 10) {
      await this.updateElasticity();
    }
  }

  /**
   * Update elasticity estimate using recent observations
   */
  private async updateElasticity(): Promise<void> {
    const recentObs = this.observations.slice(-50); // Last 50 observations

    if (recentObs.length < 2) return;

    // Calculate elasticity using log-log regression
    // elasticity = d(log(Q)) / d(log(P))
    const elasticities: number[] = [];

    for (let i = 1; i < recentObs.length; i++) {
      const prev = recentObs[i - 1];
      const curr = recentObs[i];

      if (prev.price === curr.price || prev.demand === 0 || curr.demand === 0) continue;

      const priceChange = Math.log(curr.price) - Math.log(prev.price);
      const demandChange = Math.log(curr.demand) - Math.log(prev.demand);

      if (Math.abs(priceChange) > 0.001) {
        const elasticity = demandChange / priceChange;
        elasticities.push(elasticity);
      }
    }

    if (elasticities.length > 0) {
      const estimate = {
        values: elasticities,
        mean: this.mean(elasticities),
        std: this.std(elasticities),
        timestamp: Date.now(),
      };

      await this.db.set('current_elasticity', estimate);
    }
  }

  /**
   * Get current elasticity estimate for a context
   */
  getElasticity(context: MarketContext): ElasticityEstimate {
    // Load from DB or use default
    const stored = this.db.get('current_elasticity') as any;

    if (!stored || !stored.values) {
      return {
        mean: -1.5, // Default assumption: somewhat elastic
        std: 0.5,
        confidence: 0.3,
        samples: 0,
      };
    }

    // Adjust elasticity based on context
    let adjustedMean = stored.mean;

    // Promotions increase elasticity (customers more price-sensitive)
    if (context.isPromotion) {
      adjustedMean *= 1.2;
    }

    // Peak times may reduce elasticity (customers less price-sensitive)
    if (context.hour >= 9 && context.hour <= 17) {
      adjustedMean *= 0.9;
    }

    // Low inventory should reduce elasticity (urgency to buy)
    if (context.inventory < 50) {
      adjustedMean *= 0.85;
    }

    const confidence = Math.min(stored.values.length / 50, 1.0);

    return {
      mean: adjustedMean,
      std: stored.std,
      confidence,
      samples: stored.values.length,
    };
  }

  /**
   * Get segment-specific elasticity
   */
  async getSegmentElasticity(segmentId: string): Promise<ElasticityEstimate> {
    const segmentObs = this.observations.filter(obs => {
      // In practice, you'd have segment info in the observation
      return true; // Placeholder
    });

    if (segmentObs.length < 5) {
      return this.getElasticity({} as MarketContext); // Fallback to general
    }

    // Calculate segment-specific elasticity
    const elasticities = this.segmentElasticities.get(segmentId) || [];

    return {
      mean: this.mean(elasticities),
      std: this.std(elasticities),
      confidence: Math.min(elasticities.length / 30, 1.0),
      samples: elasticities.length,
    };
  }

  /**
   * Learn from purchase behavior
   */
  async learnFromPurchase(purchase: Purchase, segmentId: string): Promise<void> {
    // Extract elasticity signal from purchase
    if (purchase.alternatives.length > 0 && purchase.converted) {
      const avgAlternative = purchase.alternatives.reduce((a, b) => a + b, 0) / purchase.alternatives.length;
      const priceRatio = purchase.price / avgAlternative;

      // If they bought despite higher price, lower elasticity
      // If they bought because of lower price, higher elasticity
      const elasticitySignal = priceRatio > 1.05 ? -0.5 : -2.0;

      const currentElasticities = this.segmentElasticities.get(segmentId) || [];
      currentElasticities.push(elasticitySignal);
      this.segmentElasticities.set(segmentId, currentElasticities.slice(-30)); // Keep last 30
    }

    await this.db.set(`segment_${segmentId}_purchase`, purchase);
  }

  /**
   * Predict demand at a given price
   */
  predictDemand(price: number, basePrice: number, baseDemand: number, context: MarketContext): {
    demand: number;
    lower: number;
    upper: number;
  } {
    const elasticity = this.getElasticity(context);
    const priceChange = (price - basePrice) / basePrice;

    // Q1 = Q0 * (1 + elasticity * priceChange)
    const demandChange = elasticity.mean * priceChange;
    const predictedDemand = baseDemand * (1 + demandChange);

    // Uncertainty bounds
    const uncertainty = Math.abs(elasticity.std * priceChange * baseDemand);

    return {
      demand: Math.max(0, predictedDemand),
      lower: Math.max(0, predictedDemand - 1.96 * uncertainty),
      upper: predictedDemand + 1.96 * uncertainty,
    };
  }

  /**
   * Get seasonality patterns
   */
  async learnSeasonality(): Promise<Map<number, number>> {
    const seasonalityMap = new Map<number, number>();

    // Group observations by day of week
    const byDay: { [key: number]: number[] } = {};

    for (const obs of this.observations) {
      const day = obs.context.dayOfWeek;
      if (!byDay[day]) byDay[day] = [];
      byDay[day].push(obs.demand);
    }

    // Calculate average demand per day
    const overallAvg = this.observations.reduce((sum, obs) => sum + obs.demand, 0) / this.observations.length;

    for (const [day, demands] of Object.entries(byDay)) {
      const dayAvg = demands.reduce((a, b) => a + b, 0) / demands.length;
      const seasonality = (dayAvg - overallAvg) / overallAvg;
      seasonalityMap.set(parseInt(day), seasonality);
    }

    await this.db.set('seasonality_patterns', Array.from(seasonalityMap.entries()));

    return seasonalityMap;
  }

  /**
   * Get promotion effect learning
   */
  async learnPromotionEffect(): Promise<number> {
    const promoObs = this.observations.filter(obs => obs.context.isPromotion);
    const normalObs = this.observations.filter(obs => !obs.context.isPromotion);

    if (promoObs.length === 0 || normalObs.length === 0) {
      return 1.3; // Default 30% lift
    }

    const promoAvgDemand = promoObs.reduce((sum, obs) => sum + obs.demand, 0) / promoObs.length;
    const normalAvgDemand = normalObs.reduce((sum, obs) => sum + obs.demand, 0) / normalObs.length;

    const effect = promoAvgDemand / normalAvgDemand;

    await this.db.set('promotion_effect', effect);

    return effect;
  }

  // Utility functions
  private mean(arr: number[]): number {
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }

  private std(arr: number[]): number {
    const avg = this.mean(arr);
    const squareDiffs = arr.map(value => Math.pow(value - avg, 2));
    return Math.sqrt(this.mean(squareDiffs));
  }

  /**
   * Export learned patterns for analysis
   */
  async exportPatterns(): Promise<{
    elasticity: ElasticityEstimate;
    seasonality: Map<number, number>;
    promotionEffect: number;
    observationCount: number;
  }> {
    const elasticity = this.getElasticity({} as MarketContext);
    const seasonality = await this.learnSeasonality();
    const promotionEffect = await this.learnPromotionEffect();

    return {
      elasticity,
      seasonality,
      promotionEffect,
      observationCount: this.observations.length,
    };
  }
}
