/**
 * Tax-Loss Harvesting System
 * Identifies and executes tax optimization strategies
 */

import { Transaction, Position } from '@neural-trader/agentic-accounting-types';
import { logger } from '../utils/logger';
import Decimal from 'decimal.js';

export interface HarvestOpportunity {
  id: string;
  asset: string;
  position: Position;
  currentPrice: number;
  unrealizedLoss: Decimal;
  potentialTaxSavings: Decimal;
  washSaleRisk: boolean;
  recommendation: 'HARVEST' | 'WAIT' | 'REVIEW';
  expirationDate?: Date;
  metadata?: any;
}

export interface WashSaleCheck {
  asset: string;
  hasViolation: boolean;
  recentBuys: Transaction[];
  daysUntilSafe: number;
}

export class TaxLossHarvestingService {
  private washSalePeriod = 30; // 30 days before and after

  /**
   * Scan portfolio for harvesting opportunities
   * Performance target: Identify 95%+ harvestable losses
   */
  async scanOpportunities(
    positions: Position[],
    currentPrices: Map<string, number>,
    recentTransactions: Transaction[],
    taxRate: number = 0.35
  ): Promise<HarvestOpportunity[]> {
    logger.info(`Scanning ${positions.length} positions for harvesting opportunities`);

    const opportunities: HarvestOpportunity[] = [];

    for (const position of positions) {
      const currentPrice = currentPrices.get(position.asset);
      if (!currentPrice) {
        continue;
      }

      // Calculate unrealized loss
      const currentValue = position.quantity.mul(currentPrice);
      const unrealizedPnL = currentValue.sub(position.totalCost);

      // Only consider losses
      if (unrealizedPnL.gte(0)) {
        continue;
      }

      const unrealizedLoss = unrealizedPnL.abs();

      // Check for wash sale risk
      const washSaleCheck = await this.checkWashSale(position.asset, recentTransactions);

      // Calculate potential tax savings
      const potentialTaxSavings = unrealizedLoss.mul(taxRate);

      // Determine recommendation
      const recommendation = this.determineRecommendation(
        unrealizedLoss,
        washSaleCheck,
        potentialTaxSavings
      );

      opportunities.push({
        id: `harvest-${position.asset}-${Date.now()}`,
        asset: position.asset,
        position,
        currentPrice,
        unrealizedLoss,
        potentialTaxSavings,
        washSaleRisk: washSaleCheck.hasViolation,
        recommendation,
        expirationDate: this.calculateExpirationDate(washSaleCheck),
        metadata: {
          washSaleCheck,
          holdingPeriod: this.calculateHoldingPeriod(position)
        }
      });
    }

    // Sort by potential savings (highest first)
    opportunities.sort((a, b) => b.potentialTaxSavings.cmp(a.potentialTaxSavings));

    logger.info(`Found ${opportunities.length} harvesting opportunities`, {
      harvestable: opportunities.filter(o => o.recommendation === 'HARVEST').length,
      totalPotentialSavings: opportunities
        .reduce((sum, o) => sum.add(o.potentialTaxSavings), new Decimal(0))
        .toString()
    });

    return opportunities;
  }

  /**
   * Check for wash sale violations
   * Target: <1% wash-sale violations
   */
  async checkWashSale(asset: string, recentTransactions: Transaction[]): Promise<WashSaleCheck> {
    const now = new Date();
    const washSaleStart = new Date(now.getTime() - this.washSalePeriod * 24 * 60 * 60 * 1000);

    // Find recent buys of the same asset
    const recentBuys = recentTransactions.filter(
      tx =>
        tx.asset === asset &&
        tx.type === 'BUY' &&
        tx.timestamp >= washSaleStart &&
        tx.timestamp <= now
    );

    const hasViolation = recentBuys.length > 0;

    // Calculate days until safe to sell
    let daysUntilSafe = 0;
    if (hasViolation) {
      const mostRecentBuy = recentBuys.reduce((latest, tx) =>
        tx.timestamp > latest.timestamp ? tx : latest
      );
      const safeDate = new Date(
        mostRecentBuy.timestamp.getTime() + this.washSalePeriod * 24 * 60 * 60 * 1000
      );
      daysUntilSafe = Math.ceil((safeDate.getTime() - now.getTime()) / (24 * 60 * 60 * 1000));
    }

    return {
      asset,
      hasViolation,
      recentBuys,
      daysUntilSafe
    };
  }

  /**
   * Find correlated replacement assets
   */
  async findReplacementAssets(
    asset: string,
    correlationThreshold: number = 0.7
  ): Promise<string[]> {
    logger.info(`Finding replacement assets for ${asset}`);

    // In production, this would use historical price data to calculate correlations
    // For now, return placeholder logic

    const replacements: string[] = [];

    // Example: ETFs in same sector
    const sectorReplacements: Record<string, string[]> = {
      'BTC': ['ETH', 'SOL'], // Crypto alternatives
      'AAPL': ['MSFT', 'GOOGL'], // Tech alternatives
      'SPY': ['VOO', 'IVV'] // S&P 500 ETF alternatives
    };

    if (sectorReplacements[asset]) {
      replacements.push(...sectorReplacements[asset]);
    }

    return replacements;
  }

  /**
   * Rank opportunities by tax savings
   */
  rankOpportunities(opportunities: HarvestOpportunity[]): HarvestOpportunity[] {
    return opportunities.sort((a, b) => {
      // Primary: by potential savings
      const savingsDiff = b.potentialTaxSavings.cmp(a.potentialTaxSavings);
      if (savingsDiff !== 0) return savingsDiff;

      // Secondary: by wash sale risk (lower risk first)
      if (a.washSaleRisk && !b.washSaleRisk) return 1;
      if (!a.washSaleRisk && b.washSaleRisk) return -1;

      return 0;
    });
  }

  /**
   * Determine recommendation based on opportunity metrics
   */
  private determineRecommendation(
    unrealizedLoss: Decimal,
    washSaleCheck: WashSaleCheck,
    potentialTaxSavings: Decimal
  ): HarvestOpportunity['recommendation'] {
    // Don't harvest if wash sale violation
    if (washSaleCheck.hasViolation) {
      return 'WAIT';
    }

    // Harvest if significant savings (>$500)
    if (potentialTaxSavings.gte(500)) {
      return 'HARVEST';
    }

    // Review if moderate savings ($100-$500)
    if (potentialTaxSavings.gte(100)) {
      return 'REVIEW';
    }

    // Wait if minimal savings
    return 'WAIT';
  }

  /**
   * Calculate when wash sale period expires
   */
  private calculateExpirationDate(washSaleCheck: WashSaleCheck): Date | undefined {
    if (!washSaleCheck.hasViolation) {
      return undefined;
    }

    const mostRecentBuy = washSaleCheck.recentBuys.reduce((latest, tx) =>
      tx.timestamp > latest.timestamp ? tx : latest
    );

    return new Date(
      mostRecentBuy.timestamp.getTime() + this.washSalePeriod * 24 * 60 * 60 * 1000
    );
  }

  /**
   * Calculate average holding period for position
   */
  private calculateHoldingPeriod(position: Position): number {
    if (position.lots.length === 0) {
      return 0;
    }

    const now = Date.now();
    const totalDays = position.lots.reduce((sum, lot) => {
      const days = (now - lot.acquisitionDate.getTime()) / (24 * 60 * 60 * 1000);
      return sum + days;
    }, 0);

    return Math.floor(totalDays / position.lots.length);
  }

  /**
   * Generate harvest execution plan
   */
  async generateExecutionPlan(opportunities: HarvestOpportunity[]): Promise<any> {
    const harvestable = opportunities.filter(o => o.recommendation === 'HARVEST');

    const plan = {
      totalOpportunities: opportunities.length,
      recommendedHarvests: harvestable.length,
      totalPotentialSavings: harvestable
        .reduce((sum, o) => sum.add(o.potentialTaxSavings), new Decimal(0))
        .toFixed(2),
      actions: harvestable.map(o => ({
        asset: o.asset,
        action: 'SELL',
        quantity: o.position.quantity.toString(),
        expectedSavings: o.potentialTaxSavings.toFixed(2),
        replacements: [] // Would be filled by findReplacementAssets
      })),
      createdAt: new Date()
    };

    return plan;
  }
}
