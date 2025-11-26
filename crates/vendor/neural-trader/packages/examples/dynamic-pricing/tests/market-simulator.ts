/**
 * Market simulator for testing dynamic pricing strategies
 */

import { MarketContext, CustomerSegment } from '../src/types';

export class MarketSimulator {
  private basePrice: number;
  private baseDemand: number;
  private trueElasticity: number;
  private competitorCount: number;
  private time: number;

  constructor(
    basePrice: number = 100,
    baseDemand: number = 100,
    trueElasticity: number = -1.5,
    competitorCount: number = 4
  ) {
    this.basePrice = basePrice;
    this.baseDemand = baseDemand;
    this.trueElasticity = trueElasticity;
    this.competitorCount = competitorCount;
    this.time = 0;
  }

  /**
   * Generate realistic market context
   */
  generateContext(): MarketContext {
    this.time++;

    const timestamp = Date.now() + this.time * 3600000; // Advance by hours
    const date = new Date(timestamp);
    const dayOfWeek = date.getDay();
    const hour = date.getHours();

    // Seasonality: weekly pattern
    const seasonality = Math.sin((dayOfWeek / 7) * 2 * Math.PI) * 0.2;

    // Random promotion
    const isPromotion = Math.random() < 0.1;

    // Holiday detection (simplified)
    const isHoliday = dayOfWeek === 0 || dayOfWeek === 6;

    // Generate competitor prices with some variance
    const competitorPrices: number[] = [];
    for (let i = 0; i < this.competitorCount; i++) {
      const variance = (Math.random() - 0.5) * 20;
      competitorPrices.push(Math.max(this.basePrice * 0.7, this.basePrice + variance));
    }

    // Inventory with random walk
    const inventory = 100 + Math.floor((Math.random() - 0.5) * 100);

    // Base demand with time-of-day pattern
    let demand = this.baseDemand;
    if (hour >= 9 && hour <= 17) demand *= 1.3; // Peak hours
    if (isPromotion) demand *= 1.4;
    if (isHoliday) demand *= 1.2;
    demand *= 1 + seasonality;

    // Add noise
    demand += (Math.random() - 0.5) * 20;

    return {
      timestamp,
      dayOfWeek,
      hour,
      isHoliday,
      isPromotion,
      seasonality,
      competitorPrices,
      inventory: Math.max(0, inventory),
      demand: Math.max(0, demand),
    };
  }

  /**
   * Simulate demand response to price
   */
  simulateDemand(price: number, context: MarketContext, noise: boolean = true): number {
    // Calculate price elasticity effect
    const priceChange = (price - this.basePrice) / this.basePrice;
    const elasticityEffect = this.trueElasticity * priceChange;

    // Start with context demand
    let demand = context.demand * (1 + elasticityEffect);

    // Competitive effect
    const avgCompetitorPrice = context.competitorPrices.reduce((a, b) => a + b, 0) / context.competitorPrices.length;
    if (price > avgCompetitorPrice * 1.1) {
      demand *= 0.8; // Lose customers to competition
    } else if (price < avgCompetitorPrice * 0.9) {
      demand *= 1.2; // Gain customers from competition
    }

    // Inventory constraint
    if (demand > context.inventory) {
      demand = context.inventory; // Can't sell more than inventory
    }

    // Add realistic noise
    if (noise) {
      const noiseLevel = demand * 0.15; // 15% noise
      demand += (Math.random() - 0.5) * noiseLevel;
    }

    return Math.max(0, demand);
  }

  /**
   * Simulate customer segments
   */
  generateSegments(count: number = 3): CustomerSegment[] {
    const segments: CustomerSegment[] = [];

    const elasticities = [-2.5, -1.5, -0.8]; // High, medium, low elasticity
    const names = ['price-sensitive', 'moderate', 'brand-loyal'];

    for (let i = 0; i < count; i++) {
      segments.push({
        id: `segment_${i}_${names[i]}`,
        priceElasticity: elasticities[i],
        valuePerception: 0.5 + i * 0.25,
        competitorAwareness: 1 - i * 0.3,
        purchaseHistory: [],
        sensitivity: Math.abs(elasticities[i]),
      });
    }

    return segments;
  }

  /**
   * Simulate purchase decision for a segment
   */
  simulatePurchaseDecision(
    segment: CustomerSegment,
    price: number,
    competitorPrices: number[]
  ): { purchased: boolean; quantity: number } {
    // Calculate willingness to pay
    const willingnessToPayMultiplier = 1 + segment.valuePerception * 0.5;
    const willingnessToPay = this.basePrice * willingnessToPayMultiplier;

    // Price sensitivity
    if (price > willingnessToPay * 1.2) {
      return { purchased: false, quantity: 0 };
    }

    // Compare with competitors
    const minCompetitorPrice = Math.min(...competitorPrices);
    const priceAdvantage = (minCompetitorPrice - price) / price;

    // Purchase probability
    let purchaseProbability = 0.7;

    if (priceAdvantage > 0.1) {
      purchaseProbability = 0.9; // Much cheaper
    } else if (priceAdvantage < -0.1) {
      purchaseProbability = 0.4 * (1 - segment.competitorAwareness); // More expensive
    }

    const purchased = Math.random() < purchaseProbability;

    // Quantity depends on elasticity
    let quantity = 1;
    if (purchased) {
      const priceRatio = price / this.basePrice;
      const quantityMultiplier = Math.pow(priceRatio, segment.priceElasticity / 2);
      quantity = Math.max(1, Math.floor(5 * quantityMultiplier));
    }

    return { purchased, quantity };
  }

  /**
   * Run simulation episode
   */
  runEpisode(
    pricePolicy: (context: MarketContext) => number,
    steps: number = 100
  ): {
    totalRevenue: number;
    avgPrice: number;
    avgDemand: number;
    history: Array<{ context: MarketContext; price: number; demand: number; revenue: number }>;
  } {
    let totalRevenue = 0;
    let totalPrice = 0;
    let totalDemand = 0;
    const history: Array<{ context: MarketContext; price: number; demand: number; revenue: number }> = [];

    for (let step = 0; step < steps; step++) {
      const context = this.generateContext();
      const price = pricePolicy(context);
      const demand = this.simulateDemand(price, context);
      const revenue = price * demand;

      totalRevenue += revenue;
      totalPrice += price;
      totalDemand += demand;

      history.push({ context, price, demand, revenue });
    }

    return {
      totalRevenue,
      avgPrice: totalPrice / steps,
      avgDemand: totalDemand / steps,
      history,
    };
  }

  /**
   * Compare strategies
   */
  compareStrategies(
    strategies: Map<string, (context: MarketContext) => number>,
    steps: number = 100
  ): Map<
    string,
    {
      totalRevenue: number;
      avgPrice: number;
      avgDemand: number;
      revenuePerUnit: number;
    }
  > {
    const results = new Map();

    for (const [name, policy] of strategies) {
      // Reset time for fair comparison
      this.time = 0;

      const episode = this.runEpisode(policy, steps);

      results.set(name, {
        totalRevenue: episode.totalRevenue,
        avgPrice: episode.avgPrice,
        avgDemand: episode.avgDemand,
        revenuePerUnit: episode.totalRevenue / episode.avgDemand,
      });
    }

    return results;
  }

  /**
   * A/B test two strategies
   */
  abTest(
    strategyA: (context: MarketContext) => number,
    strategyB: (context: MarketContext) => number,
    steps: number = 100,
    splitRatio: number = 0.5
  ): {
    strategyA: { revenue: number; samples: number };
    strategyB: { revenue: number; samples: number };
    winner: 'A' | 'B' | 'tie';
    pValue: number;
  } {
    const resultsA: number[] = [];
    const resultsB: number[] = [];

    for (let step = 0; step < steps; step++) {
      const context = this.generateContext();

      // Randomly assign to A or B
      if (Math.random() < splitRatio) {
        const price = strategyA(context);
        const demand = this.simulateDemand(price, context);
        resultsA.push(price * demand);
      } else {
        const price = strategyB(context);
        const demand = this.simulateDemand(price, context);
        resultsB.push(price * demand);
      }
    }

    const avgA = resultsA.reduce((a, b) => a + b, 0) / resultsA.length;
    const avgB = resultsB.reduce((a, b) => a + b, 0) / resultsB.length;

    // Simple t-test approximation
    const stdA = Math.sqrt(resultsA.reduce((sum, x) => sum + Math.pow(x - avgA, 2), 0) / resultsA.length);
    const stdB = Math.sqrt(resultsB.reduce((sum, x) => sum + Math.pow(x - avgB, 2), 0) / resultsB.length);

    const tStat = Math.abs(avgA - avgB) / Math.sqrt(stdA ** 2 / resultsA.length + stdB ** 2 / resultsB.length);
    const pValue = Math.max(0.01, 1 / (1 + tStat)); // Simplified p-value

    let winner: 'A' | 'B' | 'tie' = 'tie';
    if (pValue < 0.05) {
      winner = avgA > avgB ? 'A' : 'B';
    }

    return {
      strategyA: {
        revenue: avgA * resultsA.length,
        samples: resultsA.length,
      },
      strategyB: {
        revenue: avgB * resultsB.length,
        samples: resultsB.length,
      },
      winner,
      pValue,
    };
  }
}
