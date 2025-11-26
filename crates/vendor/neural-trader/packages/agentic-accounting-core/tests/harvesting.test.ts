/**
 * Tax-Loss Harvesting Tests
 * Coverage Target: 90%+
 */

import { TaxLossHarvestingService } from '../src/tax/harvesting';
import { Position, Transaction } from '@neural-trader/agentic-accounting-types';
import Decimal from 'decimal.js';

describe('TaxLossHarvestingService', () => {
  let service: TaxLossHarvestingService;

  beforeEach(() => {
    service = new TaxLossHarvestingService();
  });

  describe('scanOpportunities', () => {
    it('should identify loss positions', async () => {
      const positions: Position[] = [
        {
          id: 'pos-001',
          asset: 'BTC',
          quantity: new Decimal(1),
          averageCost: 50000,
          currentValue: 45000,
          unrealizedGainLoss: -5000,
          totalCost: new Decimal(50000),
          averageCostBasis: 50000,
          lots: [],
          lastUpdated: new Date(),
        },
      ];

      const currentPrices = new Map([['BTC', 45000]]);
      const recentTransactions: Transaction[] = [];

      const opportunities = await service.scanOpportunities(
        positions,
        currentPrices,
        recentTransactions,
        0.35
      );

      expect(opportunities.length).toBeGreaterThan(0);
      expect(opportunities[0].asset).toBe('BTC');
      expect(opportunities[0].unrealizedLoss.toNumber()).toBe(5000);
      expect(opportunities[0].potentialTaxSavings.toNumber()).toBe(1750); // 5000 * 0.35
    });

    it('should skip positions with gains', async () => {
      const positions: Position[] = [
        {
          id: 'pos-002',
          asset: 'ETH',
          quantity: new Decimal(10),
          averageCost: 2000,
          currentValue: 30000,
          unrealizedGainLoss: 10000,
          totalCost: new Decimal(20000),
          averageCostBasis: 2000,
          lots: [],
          lastUpdated: new Date(),
        },
      ];

      const currentPrices = new Map([['ETH', 3000]]);
      const recentTransactions: Transaction[] = [];

      const opportunities = await service.scanOpportunities(
        positions,
        currentPrices,
        recentTransactions
      );

      expect(opportunities).toHaveLength(0);
    });

    it('should skip positions without current price', async () => {
      const positions: Position[] = [
        {
          id: 'pos-003',
          asset: 'SOL',
          quantity: new Decimal(100),
          averageCost: 100,
          currentValue: 7500,
          unrealizedGainLoss: -2500,
          totalCost: new Decimal(10000),
          averageCostBasis: 100,
          lots: [],
          lastUpdated: new Date(),
        },
      ];

      const currentPrices = new Map(); // No price for SOL
      const recentTransactions: Transaction[] = [];

      const opportunities = await service.scanOpportunities(
        positions,
        currentPrices,
        recentTransactions
      );

      expect(opportunities).toHaveLength(0);
    });

    it('should detect wash sale risk', async () => {
      const now = new Date();
      const fifteenDaysAgo = new Date(now.getTime() - 15 * 24 * 60 * 60 * 1000);

      const positions: Position[] = [
        {
          id: 'pos-004',
          asset: 'BTC',
          quantity: new Decimal(1),
          averageCost: 50000,
          currentValue: 45000,
          unrealizedGainLoss: -5000,
          totalCost: new Decimal(50000),
          averageCostBasis: 50000,
          lots: [],
          lastUpdated: new Date(),
        },
      ];

      const currentPrices = new Map([['BTC', 45000]]);

      const recentBuy: Transaction = {
        id: 'buy-001',
        timestamp: fifteenDaysAgo,
        type: 'BUY',
        asset: 'BTC',
        quantity: 0.5,
        price: 50000,
        source: 'test',
      };

      const opportunities = await service.scanOpportunities(
        positions,
        currentPrices,
        [recentBuy]
      );

      expect(opportunities[0].washSaleRisk).toBe(true);
      expect(opportunities[0].recommendation).toBe('WAIT');
    });

    it('should sort opportunities by potential savings', async () => {
      const positions: Position[] = [
        {
          id: 'pos-small',
          asset: 'DOGE',
          quantity: new Decimal(1000),
          averageCost: 0.1,
          currentValue: 50,
          unrealizedGainLoss: -50,
          totalCost: new Decimal(100),
          averageCostBasis: 0.1,
          lots: [],
          lastUpdated: new Date(),
        },
        {
          id: 'pos-large',
          asset: 'BTC',
          quantity: new Decimal(1),
          averageCost: 60000,
          currentValue: 45000,
          unrealizedGainLoss: -15000,
          totalCost: new Decimal(60000),
          averageCostBasis: 60000,
          lots: [],
          lastUpdated: new Date(),
        },
      ];

      const currentPrices = new Map([
        ['DOGE', 0.05],
        ['BTC', 45000],
      ]);

      const opportunities = await service.scanOpportunities(
        positions,
        currentPrices,
        []
      );

      expect(opportunities[0].asset).toBe('BTC'); // Highest savings first
      expect(opportunities[1].asset).toBe('DOGE');
    });

    it('should handle custom tax rate', async () => {
      const positions: Position[] = [
        {
          id: 'pos-005',
          asset: 'BTC',
          quantity: new Decimal(1),
          averageCost: 50000,
          currentValue: 40000,
          unrealizedGainLoss: -10000,
          totalCost: new Decimal(50000),
          averageCostBasis: 50000,
          lots: [],
          lastUpdated: new Date(),
        },
      ];

      const currentPrices = new Map([['BTC', 40000]]);

      const opportunities = await service.scanOpportunities(
        positions,
        currentPrices,
        [],
        0.25 // 25% tax rate
      );

      expect(opportunities[0].potentialTaxSavings.toNumber()).toBe(2500); // 10000 * 0.25
    });
  });

  describe('checkWashSale', () => {
    it('should return no violation without recent buys', async () => {
      const result = await service.checkWashSale('BTC', []);

      expect(result.hasViolation).toBe(false);
      expect(result.recentBuys).toHaveLength(0);
      expect(result.daysUntilSafe).toBe(0);
    });

    it('should detect wash sale with recent buy', async () => {
      const now = new Date();
      const twentyDaysAgo = new Date(now.getTime() - 20 * 24 * 60 * 60 * 1000);

      const recentBuy: Transaction = {
        id: 'buy-001',
        timestamp: twentyDaysAgo,
        type: 'BUY',
        asset: 'BTC',
        quantity: 1,
        price: 50000,
        source: 'test',
      };

      const result = await service.checkWashSale('BTC', [recentBuy]);

      expect(result.hasViolation).toBe(true);
      expect(result.recentBuys).toHaveLength(1);
      expect(result.daysUntilSafe).toBeGreaterThan(0);
    });

    it('should not flag buys outside 30-day window', async () => {
      const now = new Date();
      const fortyDaysAgo = new Date(now.getTime() - 40 * 24 * 60 * 60 * 1000);

      const oldBuy: Transaction = {
        id: 'buy-001',
        timestamp: fortyDaysAgo,
        type: 'BUY',
        asset: 'BTC',
        quantity: 1,
        price: 50000,
        source: 'test',
      };

      const result = await service.checkWashSale('BTC', [oldBuy]);

      expect(result.hasViolation).toBe(false);
    });

    it('should ignore different assets', async () => {
      const now = new Date();
      const fifteenDaysAgo = new Date(now.getTime() - 15 * 24 * 60 * 60 * 1000);

      const recentBuy: Transaction = {
        id: 'buy-001',
        timestamp: fifteenDaysAgo,
        type: 'BUY',
        asset: 'ETH', // Different asset
        quantity: 10,
        price: 2500,
        source: 'test',
      };

      const result = await service.checkWashSale('BTC', [recentBuy]);

      expect(result.hasViolation).toBe(false);
    });

    it('should ignore sell transactions', async () => {
      const now = new Date();
      const fifteenDaysAgo = new Date(now.getTime() - 15 * 24 * 60 * 60 * 1000);

      const recentSell: Transaction = {
        id: 'sell-001',
        timestamp: fifteenDaysAgo,
        type: 'SELL',
        asset: 'BTC',
        quantity: 1,
        price: 45000,
        source: 'test',
      };

      const result = await service.checkWashSale('BTC', [recentSell]);

      expect(result.hasViolation).toBe(false);
    });

    it('should calculate correct days until safe', async () => {
      const now = new Date();
      const fifteenDaysAgo = new Date(now.getTime() - 15 * 24 * 60 * 60 * 1000);

      const recentBuy: Transaction = {
        id: 'buy-001',
        timestamp: fifteenDaysAgo,
        type: 'BUY',
        asset: 'BTC',
        quantity: 1,
        price: 50000,
        source: 'test',
      };

      const result = await service.checkWashSale('BTC', [recentBuy]);

      expect(result.daysUntilSafe).toBeGreaterThanOrEqual(14); // ~15 days remaining
      expect(result.daysUntilSafe).toBeLessThanOrEqual(16);
    });
  });

  describe('findReplacementAssets', () => {
    it('should return crypto alternatives for BTC', async () => {
      const replacements = await service.findReplacementAssets('BTC');

      expect(replacements).toContain('ETH');
      expect(replacements).toContain('SOL');
    });

    it('should return tech alternatives for AAPL', async () => {
      const replacements = await service.findReplacementAssets('AAPL');

      expect(replacements).toContain('MSFT');
      expect(replacements).toContain('GOOGL');
    });

    it('should return ETF alternatives for SPY', async () => {
      const replacements = await service.findReplacementAssets('SPY');

      expect(replacements).toContain('VOO');
      expect(replacements).toContain('IVV');
    });

    it('should return empty array for unknown asset', async () => {
      const replacements = await service.findReplacementAssets('UNKNOWN');
      expect(replacements).toHaveLength(0);
    });
  });

  describe('rankOpportunities', () => {
    it('should rank by potential savings', () => {
      const opportunities = [
        {
          id: '1',
          asset: 'SMALL',
          position: {} as Position,
          currentPrice: 100,
          unrealizedLoss: new Decimal(100),
          potentialTaxSavings: new Decimal(35),
          washSaleRisk: false,
          recommendation: 'HARVEST' as const,
        },
        {
          id: '2',
          asset: 'LARGE',
          position: {} as Position,
          currentPrice: 40000,
          unrealizedLoss: new Decimal(10000),
          potentialTaxSavings: new Decimal(3500),
          washSaleRisk: false,
          recommendation: 'HARVEST' as const,
        },
      ];

      const ranked = service.rankOpportunities(opportunities);

      expect(ranked[0].asset).toBe('LARGE');
      expect(ranked[1].asset).toBe('SMALL');
    });

    it('should prioritize lower wash sale risk for equal savings', () => {
      const opportunities = [
        {
          id: '1',
          asset: 'RISKY',
          position: {} as Position,
          currentPrice: 45000,
          unrealizedLoss: new Decimal(5000),
          potentialTaxSavings: new Decimal(1750),
          washSaleRisk: true,
          recommendation: 'WAIT' as const,
        },
        {
          id: '2',
          asset: 'SAFE',
          position: {} as Position,
          currentPrice: 45000,
          unrealizedLoss: new Decimal(5000),
          potentialTaxSavings: new Decimal(1750),
          washSaleRisk: false,
          recommendation: 'HARVEST' as const,
        },
      ];

      const ranked = service.rankOpportunities(opportunities);

      expect(ranked[0].asset).toBe('SAFE');
      expect(ranked[1].asset).toBe('RISKY');
    });
  });

  describe('generateExecutionPlan', () => {
    it('should generate plan for harvestable opportunities', async () => {
      const opportunities = [
        {
          id: '1',
          asset: 'BTC',
          position: {
            id: 'pos-001',
            asset: 'BTC',
            quantity: new Decimal(1),
            averageCost: 50000,
            currentValue: 45000,
            unrealizedGainLoss: -5000,
            totalCost: new Decimal(50000),
            averageCostBasis: 50000,
            lots: [],
            lastUpdated: new Date(),
          },
          currentPrice: 45000,
          unrealizedLoss: new Decimal(5000),
          potentialTaxSavings: new Decimal(1750),
          washSaleRisk: false,
          recommendation: 'HARVEST' as const,
        },
      ];

      const plan = await service.generateExecutionPlan(opportunities);

      expect(plan.totalOpportunities).toBe(1);
      expect(plan.recommendedHarvests).toBe(1);
      expect(plan.totalPotentialSavings).toBe('1750.00');
      expect(plan.actions).toHaveLength(1);
      expect(plan.actions[0].asset).toBe('BTC');
      expect(plan.actions[0].action).toBe('SELL');
    });

    it('should only include HARVEST recommendations in plan', async () => {
      const opportunities = [
        {
          id: '1',
          asset: 'BTC',
          position: {} as Position,
          currentPrice: 45000,
          unrealizedLoss: new Decimal(5000),
          potentialTaxSavings: new Decimal(1750),
          washSaleRisk: false,
          recommendation: 'HARVEST' as const,
        },
        {
          id: '2',
          asset: 'ETH',
          position: {} as Position,
          currentPrice: 2400,
          unrealizedLoss: new Decimal(1000),
          potentialTaxSavings: new Decimal(350),
          washSaleRisk: true,
          recommendation: 'WAIT' as const,
        },
      ];

      const plan = await service.generateExecutionPlan(opportunities);

      expect(plan.totalOpportunities).toBe(2);
      expect(plan.recommendedHarvests).toBe(1);
      expect(plan.actions).toHaveLength(1);
      expect(plan.actions[0].asset).toBe('BTC');
    });

    it('should handle empty opportunities', async () => {
      const plan = await service.generateExecutionPlan([]);

      expect(plan.totalOpportunities).toBe(0);
      expect(plan.recommendedHarvests).toBe(0);
      expect(plan.totalPotentialSavings).toBe('0.00');
      expect(plan.actions).toHaveLength(0);
    });
  });
});
