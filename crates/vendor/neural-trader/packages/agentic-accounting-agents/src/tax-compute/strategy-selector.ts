/**
 * Strategy Selector
 *
 * Intelligently selects optimal tax calculation method based on:
 * - User preference (if specified)
 * - Market conditions (rising/falling)
 * - Transaction history
 * - Tax optimization goals
 * - Jurisdiction requirements
 */

import { TaxMethod, TaxCalculationResult, Transaction, TaxLot } from './calculator-wrapper';

export interface MarketCondition {
  trend: 'rising' | 'falling' | 'sideways';
  volatility: 'low' | 'medium' | 'high';
  confidence: number; // 0-1
}

export interface TaxProfile {
  jurisdiction: string;
  taxBracket: 'low' | 'medium' | 'high';
  preferredMethod?: TaxMethod;
  optimizationGoal: 'minimize_current_tax' | 'maximize_carryforward' | 'balanced';
  allowedMethods?: TaxMethod[];
}

export interface MethodRecommendation {
  method: TaxMethod;
  score: number;
  rationale: string;
  estimatedSavings?: string;
  alternatives?: Array<{
    method: TaxMethod;
    score: number;
    reason: string;
  }>;
}

export class StrategySelector {
  /**
   * Select optimal method based on all factors
   */
  async selectOptimalMethod(
    sale: Transaction,
    lots: TaxLot[],
    profile: TaxProfile,
    marketCondition?: MarketCondition
  ): Promise<MethodRecommendation> {
    // If user has strong preference, respect it
    if (profile.preferredMethod && this.isMethodAllowed(profile.preferredMethod, profile)) {
      return {
        method: profile.preferredMethod,
        score: 1.0,
        rationale: 'User-specified preferred method',
      };
    }

    // Analyze market conditions
    const market = marketCondition || this.inferMarketCondition(lots, sale);

    // Score each method
    const scores = await this.scoreAllMethods(sale, lots, profile, market);

    // Sort by score
    const sorted = scores.sort((a, b) => b.score - a.score);

    return {
      method: sorted[0].method,
      score: sorted[0].score,
      rationale: sorted[0].reason,
      alternatives: sorted.slice(1, 3),
    };
  }

  /**
   * Score all available methods
   */
  private async scoreAllMethods(
    sale: Transaction,
    lots: TaxLot[],
    profile: TaxProfile,
    market: MarketCondition
  ): Promise<Array<{ method: TaxMethod; score: number; reason: string }>> {
    const methods: TaxMethod[] = profile.allowedMethods || [
      'FIFO',
      'LIFO',
      'HIFO',
      'SPECIFIC_ID',
      'AVERAGE_COST',
    ];

    const scores: Array<{ method: TaxMethod; score: number; reason: string }> = [];

    for (const method of methods) {
      if (!this.isMethodAllowed(method, profile)) {
        continue;
      }

      const score = this.scoreMethod(method, sale, lots, profile, market);
      scores.push(score);
    }

    return scores;
  }

  /**
   * Score individual method
   */
  private scoreMethod(
    method: TaxMethod,
    sale: Transaction,
    lots: TaxLot[],
    profile: TaxProfile,
    market: MarketCondition
  ): { method: TaxMethod; score: number; reason: string } {
    let score = 0.5; // Base score
    const reasons: string[] = [];

    // Calculate average lot price
    const avgLotPrice = this.calculateAverageLotPrice(lots);
    const salePrice = parseFloat(sale.price);
    const isProfit = salePrice > avgLotPrice;

    // FIFO scoring
    if (method === 'FIFO') {
      score += 0.1; // Default method bonus
      reasons.push('Standard accounting method');

      if (market.trend === 'rising' && profile.optimizationGoal === 'minimize_current_tax') {
        score += 0.2;
        reasons.push('Rising market favors FIFO for gains');
      }
    }

    // LIFO scoring
    if (method === 'LIFO') {
      if (market.trend === 'falling' && profile.optimizationGoal === 'minimize_current_tax') {
        score += 0.3;
        reasons.push('Falling market favors LIFO for loss realization');
      }

      if (profile.jurisdiction === 'US' && profile.taxBracket === 'high') {
        score += 0.1;
        reasons.push('LIFO can defer gains in high tax bracket');
      }
    }

    // HIFO scoring
    if (method === 'HIFO') {
      if (isProfit && profile.optimizationGoal === 'minimize_current_tax') {
        score += 0.4;
        reasons.push('HIFO minimizes gains on profitable sales');
      }

      if (profile.taxBracket === 'high') {
        score += 0.2;
        reasons.push('HIFO effective for high tax brackets');
      }
    }

    // Specific ID scoring
    if (method === 'SPECIFIC_ID') {
      if (lots.length > 10) {
        score += 0.3;
        reasons.push('Specific ID allows granular optimization with many lots');
      }

      if (profile.optimizationGoal === 'maximize_carryforward') {
        score += 0.2;
        reasons.push('Specific ID enables strategic loss selection');
      }
    }

    // Average Cost scoring
    if (method === 'AVERAGE_COST') {
      score += 0.05; // Simplicity bonus
      reasons.push('Simplest calculation method');

      if (lots.length < 5) {
        score += 0.1;
        reasons.push('Efficient for small lot counts');
      }

      if (market.volatility === 'high') {
        score += 0.15;
        reasons.push('Smooths volatility impact');
      }
    }

    // Jurisdiction adjustments
    if (profile.jurisdiction === 'US') {
      if (method === 'FIFO') {
        score += 0.05; // IRS default
      }
    } else if (profile.jurisdiction === 'UK') {
      if (method === 'AVERAGE_COST') {
        score += 0.1; // Share pooling rules
      }
    }

    // Cap score at 1.0
    score = Math.min(score, 1.0);

    return {
      method,
      score,
      reason: reasons.join('; '),
    };
  }

  /**
   * Check if method is allowed in jurisdiction
   */
  private isMethodAllowed(method: TaxMethod, profile: TaxProfile): boolean {
    if (profile.allowedMethods) {
      return profile.allowedMethods.includes(method);
    }

    // Default: all methods allowed except jurisdiction-specific restrictions
    if (profile.jurisdiction === 'UK' && method === 'LIFO') {
      return false; // UK doesn't allow LIFO
    }

    return true;
  }

  /**
   * Infer market conditions from lot history
   */
  private inferMarketCondition(lots: TaxLot[], sale: Transaction): MarketCondition {
    if (lots.length === 0) {
      return {
        trend: 'sideways',
        volatility: 'medium',
        confidence: 0.5,
      };
    }

    // Calculate price trend
    const prices = lots.map(lot =>
      parseFloat(lot.costBasis) / parseFloat(lot.quantity)
    );
    const salePrice = parseFloat(sale.price);

    const avgPrice = prices.reduce((a, b) => a + b, 0) / prices.length;
    const priceChange = (salePrice - avgPrice) / avgPrice;

    let trend: 'rising' | 'falling' | 'sideways';
    if (priceChange > 0.1) {
      trend = 'rising';
    } else if (priceChange < -0.1) {
      trend = 'falling';
    } else {
      trend = 'sideways';
    }

    // Calculate volatility (coefficient of variation)
    const variance = prices.reduce((sum, price) =>
      sum + Math.pow(price - avgPrice, 2), 0
    ) / prices.length;
    const stdDev = Math.sqrt(variance);
    const cv = stdDev / avgPrice;

    let volatility: 'low' | 'medium' | 'high';
    if (cv < 0.2) {
      volatility = 'low';
    } else if (cv < 0.5) {
      volatility = 'medium';
    } else {
      volatility = 'high';
    }

    return {
      trend,
      volatility,
      confidence: Math.min(lots.length / 10, 1.0), // More lots = higher confidence
    };
  }

  /**
   * Calculate average lot price
   */
  private calculateAverageLotPrice(lots: TaxLot[]): number {
    if (lots.length === 0) return 0;

    const totalCost = lots.reduce((sum, lot) =>
      sum + parseFloat(lot.costBasis), 0
    );
    const totalQty = lots.reduce((sum, lot) =>
      sum + parseFloat(lot.remainingQuantity), 0
    );

    return totalQty > 0 ? totalCost / totalQty : 0;
  }

  /**
   * Compare multiple calculation results
   */
  async compareResults(
    results: Map<TaxMethod, TaxCalculationResult>
  ): Promise<{
    best: TaxMethod;
    savings: string;
    comparison: Array<{
      method: TaxMethod;
      gain: string;
      tax: string;
      rank: number;
    }>;
  }> {
    const comparison: Array<{
      method: TaxMethod;
      gain: string;
      tax: string;
      taxNumber: number;
      rank: number;
    }> = [];

    for (const [method, result] of results.entries()) {
      // Simplified tax calculation (would use actual rates in production)
      const shortTermRate = 0.37; // Example high rate
      const longTermRate = 0.20; // Example long term rate

      const shortTermTax = parseFloat(result.shortTermGain) * shortTermRate;
      const longTermTax = parseFloat(result.longTermGain) * longTermRate;
      const totalTax = shortTermTax + longTermTax;

      comparison.push({
        method,
        gain: result.netGainLoss,
        tax: totalTax.toFixed(2),
        taxNumber: totalTax,
        rank: 0, // Will be set after sorting
      });
    }

    // Sort by tax (lowest first = best)
    comparison.sort((a, b) => a.taxNumber - b.taxNumber);

    // Assign ranks
    comparison.forEach((item, index) => {
      item.rank = index + 1;
    });

    const best = comparison[0];
    const worst = comparison[comparison.length - 1];
    const savings = (worst.taxNumber - best.taxNumber).toFixed(2);

    return {
      best: best.method,
      savings,
      comparison: comparison.map(({ method, gain, tax, rank }) => ({
        method,
        gain,
        tax,
        rank,
      })),
    };
  }
}
