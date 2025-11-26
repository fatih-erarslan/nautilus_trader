/**
 * Comprehensive test suite for Backtester
 */

import { Backtester } from '../src/backtester';
import { MomentumStrategy } from '../src/strategies/momentum';
import { MeanReversionStrategy } from '../src/strategies/mean-reversion';
import { BacktestConfig, MarketData, Strategy } from '../src/types';

describe('Backtester', () => {
  let marketData: MarketData[];
  let config: BacktestConfig;
  let strategies: Strategy[];

  beforeEach(() => {
    // Generate synthetic market data
    marketData = generateMarketData(100, 100);

    config = {
      startDate: new Date(marketData[0].timestamp),
      endDate: new Date(marketData[marketData.length - 1].timestamp),
      initialCapital: 100000,
      symbols: ['TEST'],
      strategies: [
        {
          name: 'momentum',
          type: 'momentum',
          initialWeight: 0.5,
          parameters: { lookbackPeriod: 10 },
          enabled: true
        },
        {
          name: 'mean-reversion',
          type: 'mean-reversion',
          initialWeight: 0.5,
          parameters: { maPeriod: 10 },
          enabled: true
        }
      ],
      commission: 0.001,
      slippage: 0.0005,
      walkForwardPeriods: 2
    };

    strategies = [
      new MomentumStrategy({ lookbackPeriod: 10 }),
      new MeanReversionStrategy({ maPeriod: 10 })
    ];
  });

  describe('Constructor', () => {
    it('should initialize with correct configuration', () => {
      const backtester = new Backtester(config, strategies);
      expect(backtester).toBeDefined();
      expect(backtester.getFinalEquity()).toBe(config.initialCapital);
    });
  });

  describe('runBacktest', () => {
    it('should execute backtest and return performances', async () => {
      const backtester = new Backtester(config, strategies);
      const performances = await backtester.runBacktest(marketData);

      expect(performances).toBeDefined();
      expect(Array.isArray(performances)).toBe(true);
      expect(performances.length).toBeGreaterThan(0);
    });

    it('should track portfolio states', async () => {
      const backtester = new Backtester(config, strategies);
      await backtester.runBacktest(marketData);

      const portfolioStates = backtester.getPortfolioStates();
      expect(portfolioStates.length).toBeGreaterThan(0);
      expect(portfolioStates[0].equity).toBe(config.initialCapital);
    });

    it('should generate trades', async () => {
      const backtester = new Backtester(config, strategies);
      await backtester.runBacktest(marketData);

      const trades = backtester.getTrades();
      expect(Array.isArray(trades)).toBe(true);
    });

    it('should calculate performance metrics', async () => {
      const backtester = new Backtester(config, strategies);
      const performances = await backtester.runBacktest(marketData);

      for (const perf of performances) {
        expect(perf).toHaveProperty('strategyName');
        expect(perf).toHaveProperty('totalReturn');
        expect(perf).toHaveProperty('sharpeRatio');
        expect(perf).toHaveProperty('maxDrawdown');
        expect(perf).toHaveProperty('winRate');
        expect(perf).toHaveProperty('profitFactor');
      }
    });
  });

  describe('Walk-Forward Optimization', () => {
    it('should handle multiple walk-forward periods', async () => {
      config.walkForwardPeriods = 3;
      const backtester = new Backtester(config, strategies);
      const performances = await backtester.runBacktest(marketData);

      expect(performances.length).toBeGreaterThan(0);
    });

    it('should maintain equity across periods', async () => {
      config.walkForwardPeriods = 2;
      const backtester = new Backtester(config, strategies);
      await backtester.runBacktest(marketData);

      const finalEquity = backtester.getFinalEquity();
      expect(finalEquity).toBeGreaterThan(0);
    });
  });

  describe('Transaction Costs', () => {
    it('should apply commission costs', async () => {
      config.commission = 0.01; // 1%
      const backtester = new Backtester(config, strategies);
      await backtester.runBacktest(marketData);

      const trades = backtester.getTrades();
      for (const trade of trades) {
        expect(trade.commission).toBeGreaterThan(0);
      }
    });

    it('should apply slippage costs', async () => {
      config.slippage = 0.005; // 0.5%
      const backtester = new Backtester(config, strategies);
      await backtester.runBacktest(marketData);

      const trades = backtester.getTrades();
      for (const trade of trades) {
        expect(trade.slippage).toBeGreaterThan(0);
      }
    });
  });

  describe('Regime Detection', () => {
    it('should detect market regimes', async () => {
      const backtester = new Backtester(config, strategies);
      await backtester.runBacktest(marketData);

      const portfolioStates = backtester.getPortfolioStates();
      for (const state of portfolioStates) {
        expect(state.regime).toBeDefined();
        expect(state.regime.regime).toMatch(/bull|bear|sideways|high-volatility|low-volatility/);
        expect(state.regime.confidence).toBeGreaterThanOrEqual(0);
        expect(state.regime.confidence).toBeLessThanOrEqual(1);
      }
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty market data', async () => {
      const backtester = new Backtester(config, strategies);
      const performances = await backtester.runBacktest([]);

      expect(performances).toBeDefined();
      expect(performances.length).toBe(0);
    });

    it('should handle insufficient data', async () => {
      const smallData = marketData.slice(0, 5);
      const backtester = new Backtester(config, strategies);
      const performances = await backtester.runBacktest(smallData);

      expect(performances).toBeDefined();
    });

    it('should handle zero initial capital', async () => {
      config.initialCapital = 0;
      const backtester = new Backtester(config, strategies);
      await backtester.runBacktest(marketData);

      const trades = backtester.getTrades();
      expect(trades.length).toBe(0);
    });
  });

  describe('Performance Metrics', () => {
    it('should calculate Sharpe ratio correctly', async () => {
      const backtester = new Backtester(config, strategies);
      const performances = await backtester.runBacktest(marketData);

      for (const perf of performances) {
        expect(typeof perf.sharpeRatio).toBe('number');
        expect(isFinite(perf.sharpeRatio)).toBe(true);
      }
    });

    it('should calculate maximum drawdown correctly', async () => {
      const backtester = new Backtester(config, strategies);
      const performances = await backtester.runBacktest(marketData);

      for (const perf of performances) {
        expect(perf.maxDrawdown).toBeLessThanOrEqual(0);
      }
    });

    it('should calculate win rate correctly', async () => {
      const backtester = new Backtester(config, strategies);
      const performances = await backtester.runBacktest(marketData);

      for (const perf of performances) {
        expect(perf.winRate).toBeGreaterThanOrEqual(0);
        expect(perf.winRate).toBeLessThanOrEqual(1);
      }
    });
  });
});

// Helper function to generate synthetic market data
function generateMarketData(bars: number, startPrice: number): MarketData[] {
  const data: MarketData[] = [];
  let price = startPrice;
  const startTime = Date.now() - (bars * 24 * 60 * 60 * 1000);

  for (let i = 0; i < bars; i++) {
    const change = (Math.random() - 0.5) * 4;
    price *= (1 + change / 100);

    data.push({
      timestamp: startTime + (i * 24 * 60 * 60 * 1000),
      symbol: 'TEST',
      open: price * 0.99,
      high: price * 1.02,
      low: price * 0.98,
      close: price,
      volume: Math.floor(Math.random() * 1000000) + 500000
    });
  }

  return data;
}
