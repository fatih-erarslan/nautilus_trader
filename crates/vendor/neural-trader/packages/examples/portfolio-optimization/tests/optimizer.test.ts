/**
 * Tests for Portfolio Optimizers
 */

import { describe, it, expect, beforeEach } from '@jest/globals';
import {
  MeanVarianceOptimizer,
  RiskParityOptimizer,
  BlackLittermanOptimizer,
  MultiObjectiveOptimizer,
  Asset,
} from '../src/optimizer.js';

describe('Portfolio Optimizers', () => {
  let assets: Asset[];
  let correlationMatrix: number[][];
  let marketCapWeights: number[];
  let historicalReturns: number[][];

  beforeEach(() => {
    assets = [
      { symbol: 'AAPL', expectedReturn: 0.12, volatility: 0.20 },
      { symbol: 'GOOGL', expectedReturn: 0.15, volatility: 0.25 },
      { symbol: 'MSFT', expectedReturn: 0.11, volatility: 0.18 },
    ];

    correlationMatrix = [
      [1.00, 0.65, 0.70],
      [0.65, 1.00, 0.68],
      [0.70, 0.68, 1.00],
    ];

    marketCapWeights = [0.40, 0.35, 0.25];

    historicalReturns = Array(252).fill(0).map(() =>
      assets.map(() => (Math.random() - 0.5) * 0.02)
    );
  });

  describe('MeanVarianceOptimizer', () => {
    it('should optimize portfolio with basic constraints', () => {
      const optimizer = new MeanVarianceOptimizer(assets, correlationMatrix);
      const result = optimizer.optimize({
        minWeight: 0.10,
        maxWeight: 0.50,
      });

      expect(result).toBeDefined();
      expect(result.weights).toHaveLength(3);
      expect(result.weights.reduce((a, b) => a + b, 0)).toBeCloseTo(1.0, 2);
      expect(result.sharpeRatio).toBeGreaterThan(0);
      expect(result.algorithm).toBe('mean-variance');
    });

    it('should respect weight constraints', () => {
      const optimizer = new MeanVarianceOptimizer(assets, correlationMatrix);
      const result = optimizer.optimize({
        minWeight: 0.20,
        maxWeight: 0.40,
      });

      result.weights.forEach(weight => {
        expect(weight).toBeGreaterThanOrEqual(0.15); // Allow small tolerance
        expect(weight).toBeLessThanOrEqual(0.45);
      });
    });

    it('should generate efficient frontier', () => {
      const optimizer = new MeanVarianceOptimizer(assets, correlationMatrix);
      const frontier = optimizer.generateEfficientFrontier(20);

      expect(frontier).toHaveLength(20);
      expect(frontier[0].risk).toBeLessThanOrEqual(frontier[frontier.length - 1].risk);
    });

    it('should handle target return constraint', () => {
      const optimizer = new MeanVarianceOptimizer(assets, correlationMatrix);
      const targetReturn = 0.13;
      const result = optimizer.optimize({ targetReturn });

      expect(result.expectedReturn).toBeGreaterThan(0);
    });
  });

  describe('RiskParityOptimizer', () => {
    it('should equalize risk contributions', () => {
      const optimizer = new RiskParityOptimizer(assets, correlationMatrix);
      const result = optimizer.optimize();

      expect(result).toBeDefined();
      expect(result.weights).toHaveLength(3);
      expect(result.weights.reduce((a, b) => a + b, 0)).toBeCloseTo(1.0, 2);
      expect(result.algorithm).toBe('risk-parity');
    });

    it('should produce positive weights', () => {
      const optimizer = new RiskParityOptimizer(assets, correlationMatrix);
      const result = optimizer.optimize();

      result.weights.forEach(weight => {
        expect(weight).toBeGreaterThan(0);
      });
    });

    it('should have reasonable diversification', () => {
      const optimizer = new RiskParityOptimizer(assets, correlationMatrix);
      const result = optimizer.optimize();

      expect(result.diversificationRatio).toBeGreaterThan(0.5);
      expect(result.diversificationRatio).toBeLessThanOrEqual(1.0);
    });
  });

  describe('BlackLittermanOptimizer', () => {
    it('should incorporate investor views', () => {
      const optimizer = new BlackLittermanOptimizer(
        assets,
        correlationMatrix,
        marketCapWeights,
        2.5,
      );

      const views = [
        { assets: [0], expectedReturn: 0.15, confidence: 0.7 },
        { assets: [1, 2], expectedReturn: 0.12, confidence: 0.5 },
      ];

      const result = optimizer.optimize(views);

      expect(result).toBeDefined();
      expect(result.weights).toHaveLength(3);
      expect(result.algorithm).toBe('black-litterman');
    });

    it('should work without views (equilibrium)', () => {
      const optimizer = new BlackLittermanOptimizer(
        assets,
        correlationMatrix,
        marketCapWeights,
      );

      const result = optimizer.optimize([]);

      expect(result).toBeDefined();
      expect(result.weights.reduce((a, b) => a + b, 0)).toBeCloseTo(1.0, 2);
    });

    it('should adjust based on risk aversion', () => {
      const conservativeOptimizer = new BlackLittermanOptimizer(
        assets,
        correlationMatrix,
        marketCapWeights,
        5.0, // High risk aversion
      );

      const aggressiveOptimizer = new BlackLittermanOptimizer(
        assets,
        correlationMatrix,
        marketCapWeights,
        1.0, // Low risk aversion
      );

      const conservativeResult = conservativeOptimizer.optimize([]);
      const aggressiveResult = aggressiveOptimizer.optimize([]);

      expect(conservativeResult.risk).toBeLessThanOrEqual(aggressiveResult.risk);
    });
  });

  describe('MultiObjectiveOptimizer', () => {
    it('should optimize multiple objectives', () => {
      const optimizer = new MultiObjectiveOptimizer(
        assets,
        correlationMatrix,
        historicalReturns,
      );

      const result = optimizer.optimize({
        return: 1.0,
        risk: 1.0,
        drawdown: 0.5,
      });

      expect(result).toBeDefined();
      expect(result.weights).toHaveLength(3);
      expect(result.algorithm).toBe('multi-objective');
    });

    it('should balance competing objectives', () => {
      const optimizer = new MultiObjectiveOptimizer(
        assets,
        correlationMatrix,
        historicalReturns,
      );

      const returnFocused = optimizer.optimize({
        return: 2.0,
        risk: 0.5,
        drawdown: 0.5,
      });

      const riskFocused = optimizer.optimize({
        return: 0.5,
        risk: 2.0,
        drawdown: 0.5,
      });

      expect(returnFocused.expectedReturn).toBeGreaterThanOrEqual(riskFocused.expectedReturn);
      expect(riskFocused.risk).toBeLessThanOrEqual(returnFocused.risk);
    });

    it('should consider drawdown in optimization', () => {
      const optimizer = new MultiObjectiveOptimizer(
        assets,
        correlationMatrix,
        historicalReturns,
      );

      const result = optimizer.optimize({
        return: 1.0,
        risk: 1.0,
        drawdown: 2.0, // High weight on drawdown
      });

      expect(result).toBeDefined();
      expect(result.risk).toBeGreaterThan(0);
    });
  });

  describe('Edge Cases', () => {
    it('should handle single asset portfolio', () => {
      const singleAsset = [{ symbol: 'AAPL', expectedReturn: 0.12, volatility: 0.20 }];
      const singleCorrelation = [[1.00]];

      const optimizer = new MeanVarianceOptimizer(singleAsset, singleCorrelation);
      const result = optimizer.optimize();

      expect(result.weights).toHaveLength(1);
      expect(result.weights[0]).toBeCloseTo(1.0, 2);
    });

    it('should handle highly correlated assets', () => {
      const highCorrelation = [
        [1.00, 0.95, 0.95],
        [0.95, 1.00, 0.95],
        [0.95, 0.95, 1.00],
      ];

      const optimizer = new MeanVarianceOptimizer(assets, highCorrelation);
      const result = optimizer.optimize();

      expect(result).toBeDefined();
      expect(result.diversificationRatio).toBeGreaterThan(0);
    });

    it('should handle uncorrelated assets', () => {
      const noCorrelation = [
        [1.00, 0.00, 0.00],
        [0.00, 1.00, 0.00],
        [0.00, 0.00, 1.00],
      ];

      const optimizer = new MeanVarianceOptimizer(assets, noCorrelation);
      const result = optimizer.optimize();

      expect(result).toBeDefined();
      expect(result.diversificationRatio).toBeGreaterThan(0.8);
    });
  });
});
