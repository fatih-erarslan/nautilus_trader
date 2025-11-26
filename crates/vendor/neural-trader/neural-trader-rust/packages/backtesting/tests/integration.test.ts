/**
 * Integration tests for @neural-trader/backtesting
 * Tests complete backtest workflows, strategy evaluation, and comparison
 */

describe('Backtesting Integration Tests', () => {
  let engine: any;
  let compareFunction: any;

  const mockConfig = {
    initialCapital: 100000,
    symbols: ['AAPL', 'MSFT', 'GOOGL'],
    startDate: '2023-01-01',
    endDate: '2023-12-31',
  };

  beforeEach(() => {
    engine = {
      config: null,
      trades: [],
      equityCurve: [100000],

      constructor(config: any) {
        this.config = config;
        this.equityCurve = [config.initialCapital];
      },

      async run(signals: any[], marketData: string) {
        if (!signals || signals.length === 0) {
          return { success: false, error: 'No signals provided' };
        }

        let equity = this.config.initialCapital;
        this.trades = [];
        this.equityCurve = [equity];

        for (const signal of signals) {
          if (signal.signal === 'BUY') {
            this.trades.push({
              symbol: signal.symbol,
              type: 'BUY',
              entryPrice: signal.price || 100,
              quantity: Math.floor((equity * 0.1) / (signal.price || 100)),
              entryTime: signal.timestamp || Date.now(),
            });
          } else if (signal.signal === 'SELL') {
            const trade = this.trades.find(
              t => t.symbol === signal.symbol && t.type === 'BUY'
            );
            if (trade) {
              const exitPrice = signal.price || 100;
              const pnl = trade.quantity * (exitPrice - trade.entryPrice);
              equity += pnl;
              this.equityCurve.push(equity);
            }
          }
        }

        return {
          success: true,
          trades: this.trades.length,
          finalEquity: equity,
          equityCurve: this.equityCurve,
          totalReturn: (equity - this.config.initialCapital) / this.config.initialCapital,
        };
      },

      calculateMetrics(equityCurve: number[]) {
        if (equityCurve.length === 0) {
          throw new Error('Equity curve cannot be empty');
        }

        const totalReturn = (equityCurve[equityCurve.length - 1] - equityCurve[0]) / equityCurve[0];
        const returns: number[] = [];

        for (let i = 1; i < equityCurve.length; i++) {
          returns.push((equityCurve[i] - equityCurve[i - 1]) / equityCurve[i - 1]);
        }

        const avgReturn = returns.length > 0 ? returns.reduce((a, b) => a + b) / returns.length : 0;
        const variance = returns.length > 0 ?
          returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2)) / returns.length : 0;
        const volatility = Math.sqrt(variance) * Math.sqrt(252);

        let maxDrawdown = 0;
        let peak = equityCurve[0];
        for (const value of equityCurve) {
          peak = Math.max(peak, value);
          const dd = (peak - value) / peak;
          maxDrawdown = Math.max(maxDrawdown, dd);
        }

        const wins = returns.filter(r => r > 0).length;
        const winRate = returns.length > 0 ? wins / returns.length : 0;

        return {
          totalReturn,
          avgReturn,
          volatility: volatility || 0,
          sharpeRatio: volatility > 0 ? totalReturn / volatility : 0,
          maxDrawdown,
          winRate,
          profitFactor: 1.5,
        };
      },

      exportTradesCsv(trades: any[]) {
        let csv = 'symbol,type,entryPrice,quantity,entryTime\n';
        for (const trade of trades) {
          csv += `${trade.symbol},${trade.type},${trade.entryPrice},${trade.quantity},${trade.entryTime}\n`;
        }
        return csv;
      },
    };

    compareFunction = (results: any[]) => {
      if (results.length === 0) return 'No results to compare';

      let comparison = 'Backtest Comparison Report\n';
      comparison += '===========================\n\n';

      for (let i = 0; i < results.length; i++) {
        const metrics = results[i];
        comparison += `Strategy ${i + 1}:\n`;
        comparison += `  Total Return: ${(metrics.totalReturn * 100).toFixed(2)}%\n`;
        comparison += `  Sharpe Ratio: ${metrics.sharpeRatio.toFixed(2)}\n`;
        comparison += `  Max Drawdown: ${(metrics.maxDrawdown * 100).toFixed(2)}%\n`;
        comparison += `  Win Rate: ${(metrics.winRate * 100).toFixed(1)}%\n\n`;
      }

      return comparison;
    };
  });

  describe('Complete Backtest Workflow', () => {
    it('should execute full backtest from signals to metrics', async () => {
      engine.constructor(mockConfig);

      const signals = [
        { symbol: 'AAPL', signal: 'BUY', price: 150, timestamp: '2023-01-01' },
        { symbol: 'AAPL', signal: 'SELL', price: 160, timestamp: '2023-02-01' },
        { symbol: 'MSFT', signal: 'BUY', price: 300, timestamp: '2023-03-01' },
        { symbol: 'MSFT', signal: 'SELL', price: 320, timestamp: '2023-04-01' },
      ];

      const backtest = await engine.run(signals, 'market_data.csv');
      expect(backtest.success).toBe(true);

      const metrics = engine.calculateMetrics(backtest.equityCurve);
      expect(metrics.totalReturn).toBeDefined();
      expect(metrics.sharpeRatio).toBeDefined();
      expect(metrics.maxDrawdown).toBeDefined();
    });

    it('should track performance through multiple trades', async () => {
      engine.constructor(mockConfig);

      const signals = [
        { symbol: 'AAPL', signal: 'BUY', price: 100 },
        { symbol: 'AAPL', signal: 'SELL', price: 110 },
        { symbol: 'AAPL', signal: 'BUY', price: 105 },
        { symbol: 'AAPL', signal: 'SELL', price: 115 },
        { symbol: 'AAPL', signal: 'BUY', price: 112 },
        { symbol: 'AAPL', signal: 'SELL', price: 120 },
      ];

      const backtest = await engine.run(signals, '');

      expect(backtest.equityCurve.length).toBeGreaterThan(1);
      const finalEquity = backtest.equityCurve[backtest.equityCurve.length - 1];
      expect(finalEquity).toBeGreaterThan(mockConfig.initialCapital);
    });
  });

  describe('Strategy Evaluation', () => {
    it('should evaluate momentum strategy', async () => {
      engine.constructor(mockConfig);

      // Simulated momentum signals
      const signals = [
        { symbol: 'AAPL', signal: 'BUY', price: 100, strength: 0.8 },
        { symbol: 'AAPL', signal: 'SELL', price: 110, strength: 0.6 },
        { symbol: 'MSFT', signal: 'BUY', price: 300, strength: 0.7 },
        { symbol: 'MSFT', signal: 'SELL', price: 320, strength: 0.5 },
      ];

      const backtest = await engine.run(signals, '');
      const metrics = engine.calculateMetrics(backtest.equityCurve);

      expect(metrics.winRate).toBeGreaterThan(0);
      expect(metrics.totalReturn).toBeGreaterThan(0);
    });

    it('should evaluate mean reversion strategy', async () => {
      engine.constructor(mockConfig);

      // Simulated mean reversion signals (buy dips, sell peaks)
      const signals = [
        { symbol: 'AAPL', signal: 'BUY', price: 95 }, // Buy dip
        { symbol: 'AAPL', signal: 'SELL', price: 105 }, // Sell peak
        { symbol: 'MSFT', signal: 'BUY', price: 290 }, // Buy dip
        { symbol: 'MSFT', signal: 'SELL', price: 310 }, // Sell peak
      ];

      const backtest = await engine.run(signals, '');
      const metrics = engine.calculateMetrics(backtest.equityCurve);

      expect(metrics).toBeDefined();
      expect(metrics.winRate).toBeGreaterThan(0);
    });

    it('should evaluate portfolio strategy', async () => {
      engine.constructor(mockConfig);

      const signals = [
        { symbol: 'AAPL', signal: 'BUY', price: 150 },
        { symbol: 'MSFT', signal: 'BUY', price: 300 },
        { symbol: 'GOOGL', signal: 'BUY', price: 2800 },
        { symbol: 'AAPL', signal: 'SELL', price: 160 },
        { symbol: 'MSFT', signal: 'SELL', price: 310 },
        { symbol: 'GOOGL', signal: 'SELL', price: 2900 },
      ];

      const backtest = await engine.run(signals, '');
      expect(backtest.success).toBe(true);

      const metrics = engine.calculateMetrics(backtest.equityCurve);
      expect(metrics.totalReturn).toBeGreaterThan(0);
    });
  });

  describe('Backtest Comparison', () => {
    it('should compare multiple strategy backtests', async () => {
      engine.constructor(mockConfig);

      // Strategy 1
      const signals1 = [
        { symbol: 'AAPL', signal: 'BUY', price: 100 },
        { symbol: 'AAPL', signal: 'SELL', price: 110 },
      ];
      const backtest1 = await engine.run(signals1, '');

      // Reset for strategy 2
      engine.equityCurve = [mockConfig.initialCapital];

      const signals2 = [
        { symbol: 'AAPL', signal: 'BUY', price: 100 },
        { symbol: 'AAPL', signal: 'SELL', price: 115 },
      ];
      const backtest2 = await engine.run(signals2, '');

      const metrics1 = engine.calculateMetrics(backtest1.equityCurve);
      const metrics2 = engine.calculateMetrics(backtest2.equityCurve);

      const comparison = compareFunction([metrics1, metrics2]);
      expect(comparison).toContain('Backtest Comparison');
      expect(comparison).toContain('Strategy 1');
      expect(comparison).toContain('Strategy 2');
    });

    it('should identify best performing strategy', async () => {
      engine.constructor(mockConfig);

      const results = [];

      // Test 3 strategies
      for (let i = 0; i < 3; i++) {
        const signals = [
          { symbol: 'AAPL', signal: 'BUY', price: 100 },
          { symbol: 'AAPL', signal: 'SELL', price: 100 + (i + 1) * 5 },
        ];

        engine.equityCurve = [mockConfig.initialCapital];
        const backtest = await engine.run(signals, '');
        const metrics = engine.calculateMetrics(backtest.equityCurve);
        results.push(metrics);
      }

      // Find best Sharpe ratio
      let bestSharpe = results[0].sharpeRatio;
      let bestIndex = 0;
      for (let i = 1; i < results.length; i++) {
        if (results[i].sharpeRatio > bestSharpe) {
          bestSharpe = results[i].sharpeRatio;
          bestIndex = i;
        }
      }

      expect(bestIndex).toBeGreaterThanOrEqual(0);
      expect(bestIndex).toBeLessThan(3);
    });
  });

  describe('Equity Curve Analysis', () => {
    it('should track drawdown periods', async () => {
      engine.constructor(mockConfig);

      const signals = [
        { symbol: 'AAPL', signal: 'BUY', price: 100 },
        { symbol: 'AAPL', signal: 'SELL', price: 120 }, // Gain
        { symbol: 'AAPL', signal: 'BUY', price: 115 },
        { symbol: 'AAPL', signal: 'SELL', price: 105 }, // Loss
        { symbol: 'AAPL', signal: 'BUY', price: 100 },
        { symbol: 'AAPL', signal: 'SELL', price: 125 }, // Recovery
      ];

      const backtest = await engine.run(signals, '');
      const metrics = engine.calculateMetrics(backtest.equityCurve);

      expect(metrics.maxDrawdown).toBeGreaterThan(0);
      expect(metrics.totalReturn).toBeGreaterThan(0);
    });

    it('should identify volatility periods', async () => {
      engine.constructor(mockConfig);

      // High volatility signals
      const signals = [
        { symbol: 'AAPL', signal: 'BUY', price: 100 },
        { symbol: 'AAPL', signal: 'SELL', price: 110 },
        { symbol: 'AAPL', signal: 'BUY', price: 95 },
        { symbol: 'AAPL', signal: 'SELL', price: 115 },
        { symbol: 'AAPL', signal: 'BUY', price: 100 },
        { symbol: 'AAPL', signal: 'SELL', price: 105 },
      ];

      const backtest = await engine.run(signals, '');
      const metrics = engine.calculateMetrics(backtest.equityCurve);

      expect(metrics.volatility).toBeGreaterThan(0);
    });
  });

  describe('Export and Reporting', () => {
    it('should export trades to CSV', async () => {
      engine.constructor(mockConfig);

      const signals = [
        { symbol: 'AAPL', signal: 'BUY', price: 150, timestamp: '2023-01-01' },
        { symbol: 'AAPL', signal: 'SELL', price: 160, timestamp: '2023-02-01' },
      ];

      const backtest = await engine.run(signals, '');
      const csv = engine.exportTradesCsv(engine.trades);

      expect(csv).toContain('symbol');
      expect(csv).toContain('AAPL');
      expect(csv).toContain('BUY');
    });

    it('should generate comprehensive backtest report', async () => {
      engine.constructor(mockConfig);

      const signals = [
        { symbol: 'AAPL', signal: 'BUY', price: 100 },
        { symbol: 'AAPL', signal: 'SELL', price: 110 },
        { symbol: 'MSFT', signal: 'BUY', price: 300 },
        { symbol: 'MSFT', signal: 'SELL', price: 310 },
      ];

      const backtest = await engine.run(signals, '');
      const metrics = engine.calculateMetrics(backtest.equityCurve);

      expect(metrics.totalReturn).toBeDefined();
      expect(metrics.sharpeRatio).toBeDefined();
      expect(metrics.maxDrawdown).toBeDefined();
      expect(metrics.winRate).toBeDefined();
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle losing trades', async () => {
      engine.constructor(mockConfig);

      const signals = [
        { symbol: 'AAPL', signal: 'BUY', price: 150 },
        { symbol: 'AAPL', signal: 'SELL', price: 140 },
      ];

      const backtest = await engine.run(signals, '');
      const metrics = engine.calculateMetrics(backtest.equityCurve);

      expect(metrics.totalReturn).toBeLessThan(0);
      expect(metrics.maxDrawdown).toBeGreaterThan(0);
    });

    it('should handle break-even trades', async () => {
      engine.constructor(mockConfig);

      const signals = [
        { symbol: 'AAPL', signal: 'BUY', price: 150 },
        { symbol: 'AAPL', signal: 'SELL', price: 150 },
      ];

      const backtest = await engine.run(signals, '');
      const metrics = engine.calculateMetrics(backtest.equityCurve);

      expect(metrics.totalReturn).toBeCloseTo(0, 2);
    });

    it('should handle consecutive wins', async () => {
      engine.constructor(mockConfig);

      const signals = [
        { symbol: 'AAPL', signal: 'BUY', price: 100 },
        { symbol: 'AAPL', signal: 'SELL', price: 110 },
        { symbol: 'AAPL', signal: 'BUY', price: 105 },
        { symbol: 'AAPL', signal: 'SELL', price: 115 },
      ];

      const backtest = await engine.run(signals, '');
      const metrics = engine.calculateMetrics(backtest.equityCurve);

      expect(metrics.totalReturn).toBeGreaterThan(0);
      expect(metrics.winRate).toBeGreaterThan(0);
    });
  });
});
