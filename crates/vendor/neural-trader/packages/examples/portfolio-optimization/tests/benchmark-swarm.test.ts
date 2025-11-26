/**
 * Tests for Benchmark Swarm
 */

import { describe, it, expect, beforeEach } from '@jest/globals';
import {
  PortfolioOptimizationSwarm,
  ParallelPortfolioExplorer,
  BenchmarkConfig,
} from '../src/benchmark-swarm.js';
import { Asset } from '../src/optimizer.js';

describe('Portfolio Optimization Swarm', () => {
  let swarm: PortfolioOptimizationSwarm;
  let config: BenchmarkConfig;

  beforeEach(() => {
    swarm = new PortfolioOptimizationSwarm();

    const assets: Asset[] = [
      { symbol: 'AAPL', expectedReturn: 0.12, volatility: 0.20 },
      { symbol: 'GOOGL', expectedReturn: 0.15, volatility: 0.25 },
      { symbol: 'MSFT', expectedReturn: 0.11, volatility: 0.18 },
    ];

    const correlationMatrix = [
      [1.00, 0.65, 0.70],
      [0.65, 1.00, 0.68],
      [0.70, 0.68, 1.00],
    ];

    config = {
      algorithms: ['mean-variance', 'risk-parity'],
      constraintVariations: [
        { minWeight: 0.10, maxWeight: 0.50 },
        { minWeight: 0.15, maxWeight: 0.45 },
      ],
      assets,
      correlationMatrix,
      marketCapWeights: [0.40, 0.35, 0.25],
    };
  });

  describe('Benchmark Execution', () => {
    it('should run benchmark across algorithms and constraints', async () => {
      const insights = await swarm.runBenchmark(config);

      expect(insights).toBeDefined();
      expect(insights.bestAlgorithm).toBeDefined();
      expect(insights.bestResult).toBeDefined();
      expect(insights.algorithmRankings).toHaveLength(2);
    });

    it('should rank algorithms by performance', async () => {
      const insights = await swarm.runBenchmark(config);

      const rankings = insights.algorithmRankings;
      expect(rankings.length).toBeGreaterThan(0);

      // Verify sorted by Sharpe ratio
      for (let i = 1; i < rankings.length; i++) {
        expect(rankings[i - 1].avgSharpe).toBeGreaterThanOrEqual(rankings[i].avgSharpe);
      }
    });

    it('should analyze constraint impact', async () => {
      const insights = await swarm.runBenchmark(config);

      expect(insights.constraintImpact).toBeDefined();
      expect(Object.keys(insights.constraintImpact).length).toBeGreaterThan(0);
    });

    it('should identify best result', async () => {
      const insights = await swarm.runBenchmark(config);

      expect(insights.bestResult).toBeDefined();
      expect(insights.bestResult.result.sharpeRatio).toBeGreaterThan(0);
      expect(insights.bestResult.executionTime).toBeGreaterThan(0);
    });
  });

  describe('Constraint Exploration', () => {
    it('should explore constraint space', async () => {
      const insights = await swarm.exploreConstraints(
        config,
        {
          minWeight: [0.05, 0.20],
          maxWeight: [0.40, 0.60],
          targetReturn: [0.10, 0.15],
        },
        10,
      );

      expect(insights).toBeDefined();
      expect(insights.bestResult).toBeDefined();
    });

    it('should test multiple constraint combinations', async () => {
      const insights = await swarm.exploreConstraints(
        config,
        {
          minWeight: [0.10, 0.20],
          maxWeight: [0.40, 0.50],
          targetReturn: [0.11, 0.14],
        },
        5,
      );

      expect(insights.algorithmRankings.length).toBeGreaterThan(0);
    });
  });

  describe('Market Regime Comparison', () => {
    it('should compare performance across regimes', async () => {
      const regimes = [
        { name: 'low-volatility', volatilityMultiplier: 0.5, returnMultiplier: 1.0 },
        { name: 'high-volatility', volatilityMultiplier: 2.0, returnMultiplier: 1.0 },
      ];

      const results = await swarm.compareMarketRegimes(config, regimes);

      expect(results['low-volatility']).toBeDefined();
      expect(results['high-volatility']).toBeDefined();
    });

    it('should adapt strategies to market conditions', async () => {
      const regimes = [
        { name: 'bull-market', volatilityMultiplier: 0.8, returnMultiplier: 1.5 },
        { name: 'bear-market', volatilityMultiplier: 1.5, returnMultiplier: 0.5 },
      ];

      const results = await swarm.compareMarketRegimes(config, regimes);

      const bullResult = results['bull-market'];
      const bearResult = results['bear-market'];

      expect(bullResult.bestResult.result.expectedReturn).toBeGreaterThan(
        bearResult.bestResult.result.expectedReturn,
      );
    });
  });

  describe('Report Generation', () => {
    it('should generate comprehensive report', async () => {
      const insights = await swarm.runBenchmark(config);
      const report = swarm.generateReport(insights);

      expect(report).toContain('PORTFOLIO OPTIMIZATION BENCHMARK REPORT');
      expect(report).toContain('Best Algorithm:');
      expect(report).toContain('Algorithm Rankings:');
    });

    it('should include key metrics in report', async () => {
      const insights = await swarm.runBenchmark(config);
      const report = swarm.generateReport(insights);

      expect(report).toContain('Sharpe Ratio');
      expect(report).toContain('Risk Level');
      expect(report).toContain('Expected Return');
      expect(report).toContain('Diversification Ratio');
    });
  });

  describe('OpenRouter Integration', () => {
    it('should work without OpenRouter API key', async () => {
      const swarmWithoutAPI = new PortfolioOptimizationSwarm();
      const insights = await swarmWithoutAPI.runBenchmark(config);

      expect(insights).toBeDefined();
      expect(insights.recommendations).toEqual([]);
    });

    it('should skip AI recommendations when unavailable', async () => {
      const insights = await swarm.runBenchmark(config);

      // Without API key, recommendations should be empty
      expect(Array.isArray(insights.recommendations)).toBe(true);
    });
  });

  describe('Error Handling', () => {
    it('should handle invalid algorithm gracefully', async () => {
      const invalidConfig = {
        ...config,
        algorithms: ['invalid-algorithm'],
      };

      const insights = await swarm.runBenchmark(invalidConfig);

      // Should still return insights with remaining valid algorithms
      expect(insights).toBeDefined();
    });

    it('should continue on partial failures', async () => {
      const mixedConfig = {
        ...config,
        algorithms: ['mean-variance', 'invalid-algo', 'risk-parity'],
      };

      const insights = await swarm.runBenchmark(mixedConfig);

      expect(insights).toBeDefined();
      expect(insights.algorithmRankings.length).toBeGreaterThan(0);
    });
  });
});

describe('Parallel Portfolio Explorer', () => {
  let explorer: ParallelPortfolioExplorer;

  beforeEach(() => {
    explorer = new ParallelPortfolioExplorer(2);
  });

  describe('Parallel Optimization', () => {
    it('should process optimizations in parallel', async () => {
      const assets: Asset[] = [
        { symbol: 'AAPL', expectedReturn: 0.12, volatility: 0.20 },
        { symbol: 'GOOGL', expectedReturn: 0.15, volatility: 0.25 },
      ];

      const correlationMatrix = [
        [1.00, 0.65],
        [0.65, 1.00],
      ];

      const configs = [
        {
          algorithm: 'mean-variance',
          config: {
            algorithms: ['mean-variance'],
            constraintVariations: [],
            assets,
            correlationMatrix,
          },
          constraints: { minWeight: 0.2, maxWeight: 0.8 },
        },
        {
          algorithm: 'risk-parity',
          config: {
            algorithms: ['risk-parity'],
            constraintVariations: [],
            assets,
            correlationMatrix,
          },
          constraints: { minWeight: 0.2, maxWeight: 0.8 },
        },
      ];

      const results = await explorer.optimizeInParallel(configs);

      expect(results).toHaveLength(2);
      expect(results[0].algorithm).toBe('mean-variance');
      expect(results[1].algorithm).toBe('risk-parity');
    });

    it('should handle batching correctly', async () => {
      const assets: Asset[] = [
        { symbol: 'AAPL', expectedReturn: 0.12, volatility: 0.20 },
      ];

      const correlationMatrix = [[1.00]];

      const configs = Array(5).fill(0).map(() => ({
        algorithm: 'mean-variance',
        config: {
          algorithms: ['mean-variance'],
          constraintVariations: [],
          assets,
          correlationMatrix,
        },
        constraints: {},
      }));

      const results = await explorer.optimizeInParallel(configs);

      expect(results).toHaveLength(5);
    });
  });
});
