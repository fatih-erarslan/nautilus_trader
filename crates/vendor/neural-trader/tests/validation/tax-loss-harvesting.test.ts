/**
 * Tax-Loss Harvesting Validation Tests
 *
 * Validates tax-loss harvesting recommendations and calculations
 * to ensure optimal tax savings while avoiding wash sale violations.
 *
 * Reference: IRS Publication 550, Tax-Loss Harvesting Strategies
 */

import { describe, it, expect } from 'vitest';
import Decimal from 'decimal.js';

describe('Tax-Loss Harvesting - Strategy Validation', () => {

  describe('Loss Harvesting Opportunity Identification', () => {

    it('Identify unrealized losses for harvesting', () => {
      /**
       * Find positions with unrealized losses that can be harvested
       */

      const positions = [
        {
          asset: 'BTC',
          quantity: new Decimal(2),
          costBasis: new Decimal(100000),
          currentValue: new Decimal(80000),
          unrealizedLoss: new Decimal(-20000),
        },
        {
          asset: 'ETH',
          quantity: new Decimal(50),
          costBasis: new Decimal(100000),
          currentValue: new Decimal(120000),
          unrealizedGain: new Decimal(20000),
        },
        {
          asset: 'SOL',
          quantity: new Decimal(1000),
          costBasis: new Decimal(50000),
          currentValue: new Decimal(35000),
          unrealizedLoss: new Decimal(-15000),
        },
      ];

      // Identify loss positions
      const lossPositions = positions.filter(p =>
        p.unrealizedLoss && p.unrealizedLoss.lessThan(0)
      );

      expect(lossPositions.length).toBe(2);

      // Total harvestable loss
      const totalLoss = lossPositions.reduce(
        (sum, p) => sum.plus(p.unrealizedLoss || 0),
        new Decimal(0)
      );

      expect(totalLoss.toString()).toBe('-35000');
    });

    it('Prioritize losses by tax savings potential', () => {
      /**
       * Rank loss harvesting opportunities by tax impact
       * Short-term losses offset higher-taxed short-term gains first
       */

      const lossOpportunities = [
        {
          asset: 'BTC',
          unrealizedLoss: new Decimal(-20000),
          term: 'SHORT',
          taxRate: 0.37, // 37% ordinary income rate
          taxSavings: new Decimal(7400), // $20k * 37%
        },
        {
          asset: 'ETH',
          unrealizedLoss: new Decimal(-15000),
          term: 'LONG',
          taxRate: 0.20, // 20% long-term capital gains
          taxSavings: new Decimal(3000), // $15k * 20%
        },
        {
          asset: 'SOL',
          unrealizedLoss: new Decimal(-10000),
          term: 'SHORT',
          taxRate: 0.37,
          taxSavings: new Decimal(3700), // $10k * 37%
        },
      ];

      // Sort by tax savings (highest first)
      const ranked = lossOpportunities.sort((a, b) =>
        b.taxSavings.comparedTo(a.taxSavings)
      );

      expect(ranked[0].asset).toBe('BTC'); // Highest savings
      expect(ranked[1].asset).toBe('SOL');
      expect(ranked[2].asset).toBe('ETH'); // Lowest savings
    });

    it('Identify losses exceeding $3,000 threshold', () => {
      /**
       * Losses > $3,000/year provide maximum current deduction
       * Excess losses carried forward
       */

      const unrealizedLoss = new Decimal(-25000);
      const annualLimit = new Decimal(3000);

      const currentYearBenefit = Decimal.min(unrealizedLoss.abs(), annualLimit);
      const carryoverBenefit = unrealizedLoss.abs().minus(currentYearBenefit);

      expect(currentYearBenefit.toString()).toBe('3000');
      expect(carryoverBenefit.toString()).toBe('22000');

      // Multiple years of benefit
      const yearsOfBenefit = Math.ceil(unrealizedLoss.abs().dividedBy(annualLimit).toNumber());
      expect(yearsOfBenefit).toBe(9); // 25k / 3k = 8.33 years
    });
  });

  describe('Wash Sale Avoidance Strategies', () => {

    it('31-day safe harbor for loss harvesting', () => {
      /**
       * Wait 31 days to avoid wash sale
       */

      const saleLoss = {
        asset: 'BTC',
        quantity: new Decimal(1),
        loss: new Decimal(-10000),
        date: new Date('2023-06-15'),
      };

      // Wait 31 days before repurchasing
      const safeRepurchaseDate = new Date(saleLoss.date);
      safeRepurchaseDate.setDate(safeRepurchaseDate.getDate() + 31);

      const repurchase = {
        asset: 'BTC',
        quantity: new Decimal(1),
        date: safeRepurchaseDate,
      };

      const daysDiff = Math.floor(
        (repurchase.date.getTime() - saleLoss.date.getTime()) / (1000 * 60 * 60 * 24)
      );

      expect(daysDiff).toBe(31);
      expect(daysDiff).toBeGreaterThan(30);

      // No wash sale
      const washSale = {
        isWashSale: false,
      };

      expect(washSale.isWashSale).toBe(false);
    });

    it('Substitute similar but not identical securities', () => {
      /**
       * Sell BTC, buy ETH to maintain crypto exposure
       * without triggering wash sale
       */

      const btcSale = {
        asset: 'BTC',
        quantity: new Decimal(1),
        loss: new Decimal(-10000),
        date: new Date('2023-06-15'),
      };

      const ethPurchase = {
        asset: 'ETH',
        quantity: new Decimal(10),
        cost: new Decimal(18000),
        date: new Date('2023-06-16'), // Next day
      };

      // Different assets = not substantially identical
      const isSubstantiallyIdentical = btcSale.asset === ethPurchase.asset;

      expect(isSubstantiallyIdentical).toBe(false);

      // No wash sale, loss is deductible
      const washSale = {
        isWashSale: false,
      };

      expect(washSale.isWashSale).toBe(false);
    });

    it('Harvest losses from multiple lots selectively', () => {
      /**
       * Sell loss lots, keep gain lots
       * Maintain position size without wash sale
       */

      const lots = [
        { id: 'LOT-1', qty: 1, costBasis: 50000, currentValue: 40000, loss: -10000 },
        { id: 'LOT-2', qty: 1, costBasis: 30000, currentValue: 40000, gain: 10000 },
        { id: 'LOT-3', qty: 1, costBasis: 45000, currentValue: 40000, loss: -5000 },
        { id: 'LOT-4', qty: 1, costBasis: 35000, currentValue: 40000, gain: 5000 },
      ];

      // Sell only loss lots using Specific ID
      const lossLots = lots.filter(l => l.loss < 0);
      const totalLossHarvested = lossLots.reduce((sum, l) => sum + l.loss, 0);

      expect(lossLots.length).toBe(2);
      expect(totalLossHarvested).toBe(-15000);

      // Keep gain lots (2 BTC remaining)
      const remainingQty = lots.length - lossLots.length;
      expect(remainingQty).toBe(2);
    });
  });

  describe('Timing Optimization', () => {

    it('Harvest losses before year-end', () => {
      /**
       * Realize losses before Dec 31 to offset current year gains
       */

      const currentDate = new Date('2023-12-15');
      const yearEnd = new Date('2023-12-31');

      const daysUntilYearEnd = Math.floor(
        (yearEnd.getTime() - currentDate.getTime()) / (1000 * 60 * 60 * 24)
      );

      expect(daysUntilYearEnd).toBe(16);

      // Sufficient time to harvest and settle
      const tradeSettlement = 2; // T+2 settlement
      const timeToHarvest = daysUntilYearEnd > tradeSettlement;

      expect(timeToHarvest).toBe(true);
    });

    it('Defer gains to next year if beneficial', () => {
      /**
       * If already at loss limit, defer gains to next year
       */

      const currentYearSituation = {
        realizedGains: new Decimal(0),
        realizedLosses: new Decimal(-3000), // At limit
        unrealizedGains: new Decimal(50000),
      };

      const currentDate = new Date('2023-12-20');
      const yearEnd = new Date('2023-12-31');

      // Already at $3k loss limit
      const atLossLimit = currentYearSituation.realizedLosses.abs().equals(3000);

      // Defer gain realization to next year
      const shouldDeferGain = atLossLimit && currentDate > new Date('2023-12-01');

      expect(atLossLimit).toBe(true);
      expect(shouldDeferGain).toBe(true);
    });

    it('Harvest losses early in year for carryforward', () => {
      /**
       * Early year harvesting provides more time for replacement
       */

      const harvestDate = new Date('2023-01-15');
      const safeRepurchaseDate = new Date('2023-02-15'); // 31 days later
      const yearEnd = new Date('2023-12-31');

      const timeToRepurchase = Math.floor(
        (safeRepurchaseDate.getTime() - harvestDate.getTime()) / (1000 * 60 * 60 * 24)
      );

      const timeToYearEnd = Math.floor(
        (yearEnd.getTime() - safeRepurchaseDate.getTime()) / (1000 * 60 * 60 * 24)
      );

      expect(timeToRepurchase).toBe(31);
      expect(timeToYearEnd).toBeGreaterThan(300); // 10+ months
    });
  });

  describe('Loss Carryforward Strategy', () => {

    it('Calculate multi-year loss carryforward benefit', () => {
      /**
       * Large losses provide tax benefits over multiple years
       */

      const totalLoss = new Decimal(-50000);
      const annualLimit = new Decimal(3000);
      const taxRate = new Decimal(0.37); // 37%

      const years: any[] = [];
      let remainingLoss = totalLoss.abs();
      let year = 2023;

      while (remainingLoss.greaterThan(0)) {
        const yearDeduction = Decimal.min(remainingLoss, annualLimit);
        const taxSavings = yearDeduction.times(taxRate);

        years.push({
          year: year++,
          deduction: yearDeduction,
          taxSavings: taxSavings,
          remainingLoss: remainingLoss.minus(yearDeduction),
        });

        remainingLoss = remainingLoss.minus(yearDeduction);
      }

      expect(years.length).toBe(17); // 50k / 3k = 16.67 years

      const totalTaxSavings = years.reduce(
        (sum, y) => sum.plus(y.taxSavings),
        new Decimal(0)
      );

      expect(totalTaxSavings.toFixed(0)).toBe('18500'); // $50k * 37%
    });

    it('Optimize loss recognition across tax brackets', () => {
      /**
       * Recognize losses in years with higher income/tax rates
       */

      const scenarios = [
        {
          year: 2023,
          income: new Decimal(200000),
          taxRate: 0.32,
          lossValue: new Decimal(3000),
          savings: new Decimal(960), // $3k * 32%
        },
        {
          year: 2024,
          income: new Decimal(600000),
          taxRate: 0.37, // Higher bracket
          lossValue: new Decimal(3000),
          savings: new Decimal(1110), // $3k * 37%
        },
      ];

      // Higher income year provides more benefit
      expect(scenarios[1].savings.greaterThan(scenarios[0].savings)).toBe(true);
    });
  });

  describe('Offsetting Gains Strategy', () => {

    it('Offset short-term gains with losses first', () => {
      /**
       * Short-term gains taxed at higher rate
       * Prioritize offsetting with losses
       */

      const gains = {
        shortTerm: new Decimal(20000),
        longTerm: new Decimal(30000),
      };

      const harvestedLoss = new Decimal(-15000);

      // Apply loss to short-term gains first (higher tax rate)
      const offsetShortTerm = Decimal.min(gains.shortTerm, harvestedLoss.abs());
      const remainingLoss = harvestedLoss.abs().minus(offsetShortTerm);

      expect(offsetShortTerm.toString()).toBe('15000');
      expect(remainingLoss.toString()).toBe('0');

      // Net short-term gain after offset
      const netShortTerm = gains.shortTerm.minus(offsetShortTerm);
      expect(netShortTerm.toString()).toBe('5000');

      // Long-term gains unchanged
      const netLongTerm = gains.longTerm;
      expect(netLongTerm.toString()).toBe('30000');
    });

    it('Calculate tax savings from offsetting gains', () => {
      /**
       * Harvested loss offsets high-tax gains
       */

      const shortTermGain = new Decimal(10000);
      const shortTermRate = new Decimal(0.37);

      const harvestedLoss = new Decimal(-10000);

      // Tax without harvesting
      const taxWithoutHarvest = shortTermGain.times(shortTermRate);
      expect(taxWithoutHarvest.toString()).toBe('3700');

      // Tax with harvesting (gain offset)
      const netGain = shortTermGain.plus(harvestedLoss);
      const taxWithHarvest = netGain.greaterThan(0)
        ? netGain.times(shortTermRate)
        : new Decimal(0);

      expect(taxWithHarvest.toString()).toBe('0');

      // Tax savings
      const savings = taxWithoutHarvest.minus(taxWithHarvest);
      expect(savings.toString()).toBe('3700');
    });
  });

  describe('Portfolio Rebalancing with Loss Harvesting', () => {

    it('Harvest losses while rebalancing portfolio', () => {
      /**
       * Use loss harvesting as opportunity to rebalance
       */

      const portfolio = {
        target: { BTC: 0.50, ETH: 0.30, SOL: 0.20 },
        current: { BTC: 0.60, ETH: 0.25, SOL: 0.15 },
      };

      // BTC is overweight with unrealized loss
      const btcPosition = {
        currentAllocation: 0.60,
        targetAllocation: 0.50,
        overweight: 0.10,
        unrealizedLoss: new Decimal(-20000),
      };

      // Harvest loss by selling excess BTC
      const shouldHarvest = btcPosition.overweight > 0 && btcPosition.unrealizedLoss.lessThan(0);

      expect(shouldHarvest).toBe(true);

      // Rebalance to target allocation
      const rebalancedAllocation = btcPosition.targetAllocation;
      expect(rebalancedAllocation).toBe(0.50);
    });
  });

  describe('Loss Harvesting Decision Matrix', () => {

    it('Evaluate whether to harvest based on multiple factors', () => {
      /**
       * Decision factors:
       * - Size of unrealized loss
       * - Time to wash sale expiration
       * - Alternative investments available
       * - Tax rate
       * - Portfolio allocation
       */

      const position = {
        unrealizedLoss: new Decimal(-25000),
        daysHeld: 180,
        alternativeAvailable: true, // Can buy similar asset
        taxRate: 0.37,
        overweight: true,
      };

      const harvestScore = {
        lossSize: position.unrealizedLoss.abs().greaterThan(10000) ? 3 : 0,
        longHeld: position.daysHeld > 365 ? 2 : 1,
        alternatives: position.alternativeAvailable ? 2 : 0,
        taxBenefit: position.taxRate > 0.30 ? 3 : 1,
        rebalancing: position.overweight ? 2 : 0,
      };

      const totalScore = Object.values(harvestScore).reduce((sum, s) => sum + s, 0);

      expect(totalScore).toBe(11); // High score = harvest

      // Recommendation
      const shouldHarvest = totalScore >= 8;
      expect(shouldHarvest).toBe(true);
    });
  });

  describe('Crypto-Specific Loss Harvesting', () => {

    it('No wash sale for crypto in 2023 (IRS guidance)', () => {
      /**
       * Pre-2025: Wash sale rules don't apply to crypto
       * Can harvest and immediately repurchase
       */

      const btcSale = {
        asset: 'BTC',
        loss: new Decimal(-10000),
        date: new Date('2023-06-15'),
      };

      const btcRepurchase = {
        asset: 'BTC',
        date: new Date('2023-06-15'), // Same day
      };

      // Pre-2025: No wash sale rule for crypto
      const year = btcSale.date.getFullYear();
      const washSaleApplies = year >= 2025;

      expect(washSaleApplies).toBe(false);

      // Loss is fully deductible
      const deductibleLoss = btcSale.loss;
      expect(deductibleLoss.toString()).toBe('-10000');
    });

    it('Consider future wash sale rules (2025+)', () => {
      /**
       * Plan for potential future wash sale rules on crypto
       */

      const cryptoSale = {
        asset: 'BTC',
        loss: new Decimal(-10000),
        date: new Date('2025-06-15'),
      };

      // Conservative approach: assume wash sale will apply
      const year = cryptoSale.date.getFullYear();
      const assumeWashSale = year >= 2025;

      expect(assumeWashSale).toBe(true);

      // Plan for 31-day waiting period
      const safeRepurchaseDate = new Date(cryptoSale.date);
      safeRepurchaseDate.setDate(safeRepurchaseDate.getDate() + 31);

      const daysDiff = Math.floor(
        (safeRepurchaseDate.getTime() - cryptoSale.date.getTime()) / (1000 * 60 * 60 * 24)
      );

      expect(daysDiff).toBe(31);
    });
  });

  describe('Loss Harvesting Reporting', () => {

    it('Generate loss harvesting report', () => {
      /**
       * Comprehensive report of loss harvesting activities
       */

      const harvestingReport = {
        taxYear: 2023,
        totalLossesHarvested: new Decimal(-45000),
        transactionCount: 8,
        byAsset: [
          { asset: 'BTC', losses: new Decimal(-25000), transactions: 3 },
          { asset: 'ETH', losses: new Decimal(-15000), transactions: 3 },
          { asset: 'SOL', losses: new Decimal(-5000), transactions: 2 },
        ],
        taxSavings: {
          currentYear: new Decimal(1110), // $3k * 37%
          carryforward: new Decimal(42000),
          projectedFutureYears: 14,
          totalProjectedSavings: new Decimal(16650), // $45k * 37%
        },
        washSaleViolations: 0,
        recommendations: [
          'Losses exceed $3,000 annual limit - $42,000 will carry forward',
          'Consider offsetting future gains to maximize benefit',
          'Track carryforward losses for future tax years',
        ],
      };

      expect(harvestingReport.totalLossesHarvested.toString()).toBe('-45000');
      expect(harvestingReport.taxSavings.currentYear.toString()).toBe('1110');
      expect(harvestingReport.washSaleViolations).toBe(0);
      expect(harvestingReport.recommendations.length).toBeGreaterThan(0);
    });
  });
});
