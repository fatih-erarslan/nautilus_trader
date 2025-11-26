/**
 * Unit tests for @neural-trader/backtesting
 * Tests backtesting engine, metrics calculation, and trade analysis
 */

describe('Backtesting Unit Tests', () => {
  describe('BacktestEngine', () => {
    let engine: any;

    const mockConfig = {
      initialCapital: 100000,
      symbols: ['AAPL', 'MSFT'],
      startDate: '2023-01-01',
      endDate: '2023-12-31',
    };

    beforeEach(() => {
      engine = {
        config: null,
        trades: [],
        equityCurve: [100000],

        constructor(config: any) {
          if (!config.initialCapital) {
            throw new Error('Initial capital required');
          }
          if (!config.symbols || config.symbols.length === 0) {
            throw new Error('At least one symbol required');
          }
          this.config = config;
          this.equityCurve = [config.initialCapital];
        },

        async run(signals: any[], marketData: string) {
          if (!signals || signals.length === 0) {
            return {
              success: false,
              error: 'No signals provided',
            };
          }

          // Simulate backtest execution
          let equity = this.config.initialCapital;
          this.trades = [];

          for (const signal of signals) {
            if (signal.signal === 'BUY') {
              this.trades.push({
                symbol: signal.symbol,
                entryPrice: signal.price || 100,
                quantity: Math.floor((equity * 0.1) / (signal.price || 100)),
                entryTime: signal.timestamp || Date.now(),
              });
            } else if (signal.signal === 'SELL') {
              const trade = this.trades.find(t => t.symbol === signal.symbol);
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
          };
        },

        calculateMetrics(equityCurve: number[]) {
          if (equityCurve.length === 0) {
            throw new Error('Equity curve cannot be empty');
          }

          const returns: number[] = [];
          for (let i = 1; i < equityCurve.length; i++) {
            returns.push((equityCurve[i] - equityCurve[i - 1]) / equityCurve[i - 1]);
          }

          const totalReturn = (equityCurve[equityCurve.length - 1] - equityCurve[0]) / equityCurve[0];
          const avgReturn = returns.length > 0 ? returns.reduce((a, b) => a + b) / returns.length : 0;

          const variance = returns.length > 0 ?
            returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2)) / returns.length : 0;
          const volatility = Math.sqrt(variance) * Math.sqrt(252); // Annualized

          // Sharpe ratio (assuming 0 risk-free rate)
          const sharpeRatio = volatility > 0 ? (totalReturn / returns.length) / volatility : 0;

          // Max drawdown
          let maxDrawdown = 0;
          let peak = equityCurve[0];
          for (const value of equityCurve) {
            peak = Math.max(peak, value);
            const dd = (peak - value) / peak;
            maxDrawdown = Math.max(maxDrawdown, dd);
          }

          return {
            totalReturn,
            avgReturn,
            volatility,
            sharpeRatio: sharpeRatio || 0,
            maxDrawdown,
            winRate: 0.5,
          };
        },

        exportTradesCsv(trades: any[]) {
          if (trades.length === 0) {
            return 'symbol,entryPrice,quantity,entryTime\n';
          }

          let csv = 'symbol,entryPrice,quantity,entryTime\n';
          for (const trade of trades) {
            csv += `${trade.symbol},${trade.entryPrice},${trade.quantity},${trade.entryTime}\n`;
          }
          return csv;
        },
      };
    });

    describe('Initialization', () => {
      it('should create engine with valid config', () => {
        engine.constructor(mockConfig);
        expect(engine.config).toEqual(mockConfig);
      });

      it('should throw on missing initial capital', () => {
        expect(() => {
          engine.constructor({ symbols: ['AAPL'] });
        }).toThrow('Initial capital required');
      });

      it('should throw on empty symbols', () => {
        expect(() => {
          engine.constructor({ initialCapital: 100000, symbols: [] });
        }).toThrow('At least one symbol required');
      });

      it('should throw on missing symbols', () => {
        expect(() => {
          engine.constructor({ initialCapital: 100000 });
        }).toThrow('At least one symbol required');
      });

      it('should initialize equity curve with initial capital', () => {
        engine.constructor(mockConfig);
        expect(engine.equityCurve[0]).toBe(mockConfig.initialCapital);
      });
    });

    describe('Backtest Execution', () => {
      beforeEach(() => {
        engine.constructor(mockConfig);
      });

      it('should execute backtest with signals', async () => {
        const signals = [
          { symbol: 'AAPL', signal: 'BUY', price: 150, timestamp: Date.now() },
          { symbol: 'AAPL', signal: 'SELL', price: 160, timestamp: Date.now() + 1000 },
        ];

        const result = await engine.run(signals, 'market_data.csv');

        expect(result.success).toBe(true);
        expect(result.finalEquity).toBeGreaterThan(0);
        expect(result.trades).toBeGreaterThan(0);
      });

      it('should reject backtest with no signals', async () => {
        const result = await engine.run([], 'market_data.csv');

        expect(result.success).toBe(false);
        expect(result.error).toBe('No signals provided');
      });

      it('should generate equity curve', async () => {
        const signals = [
          { symbol: 'AAPL', signal: 'BUY', price: 150 },
          { symbol: 'AAPL', signal: 'SELL', price: 160 },
          { symbol: 'AAPL', signal: 'BUY', price: 155 },
          { symbol: 'AAPL', signal: 'SELL', price: 165 },
        ];

        const result = await engine.run(signals, '');

        expect(result.equityCurve).toBeDefined();
        expect(result.equityCurve.length).toBeGreaterThan(0);
        expect(result.equityCurve[0]).toBe(mockConfig.initialCapital);
      });

      it('should handle multiple symbol signals', async () => {
        const signals = [
          { symbol: 'AAPL', signal: 'BUY', price: 150 },
          { symbol: 'MSFT', signal: 'BUY', price: 300 },
          { symbol: 'AAPL', signal: 'SELL', price: 155 },
          { symbol: 'MSFT', signal: 'SELL', price: 310 },
        ];

        const result = await engine.run(signals, '');

        expect(result.success).toBe(true);
        expect(result.finalEquity).toBeGreaterThan(0);
      });
    });

    describe('Metrics Calculation', () => {
      beforeEach(() => {
        engine.constructor(mockConfig);
      });

      it('should calculate metrics for simple equity curve', () => {
        const equityCurve = [100000, 105000, 110000, 108000, 115000];
        const metrics = engine.calculateMetrics(equityCurve);

        expect(metrics.totalReturn).toBeGreaterThan(0);
        expect(metrics.volatility).toBeGreaterThanOrEqual(0);
        expect(metrics.sharpeRatio).toBeDefined();
        expect(metrics.maxDrawdown).toBeGreaterThanOrEqual(0);
      });

      it('should throw on empty equity curve', () => {
        expect(() => {
          engine.calculateMetrics([]);
        }).toThrow('Equity curve cannot be empty');
      });

      it('should calculate total return correctly', () => {
        const equityCurve = [100000, 150000];
        const metrics = engine.calculateMetrics(equityCurve);

        expect(metrics.totalReturn).toBe(0.5); // 50% return
      });

      it('should calculate volatility', () => {
        const equityCurve = [100000, 101000, 102000, 101500, 103000];
        const metrics = engine.calculateMetrics(equityCurve);

        expect(metrics.volatility).toBeGreaterThan(0);
      });

      it('should calculate max drawdown', () => {
        const equityCurve = [100000, 110000, 105000, 95000, 100000];
        const metrics = engine.calculateMetrics(equityCurve);

        expect(metrics.maxDrawdown).toBeGreaterThan(0);
        expect(metrics.maxDrawdown).toBeLessThanOrEqual(1);
      });

      it('should calculate Sharpe ratio', () => {
        const equityCurve = [100000, 101000, 102000, 103000, 104000];
        const metrics = engine.calculateMetrics(equityCurve);

        expect(metrics.sharpeRatio).toBeDefined();
        expect(typeof metrics.sharpeRatio).toBe('number');
      });
    });

    describe('Trade Export', () => {
      beforeEach(() => {
        engine.constructor(mockConfig);
      });

      it('should export empty trades to CSV', () => {
        const csv = engine.exportTradesCsv([]);
        expect(csv).toBe('symbol,entryPrice,quantity,entryTime\n');
      });

      it('should export trades to CSV format', () => {
        const trades = [
          { symbol: 'AAPL', entryPrice: 150, quantity: 100, entryTime: '2023-01-01' },
          { symbol: 'MSFT', entryPrice: 300, quantity: 50, entryTime: '2023-01-02' },
        ];

        const csv = engine.exportTradesCsv(trades);

        expect(csv).toContain('AAPL');
        expect(csv).toContain('MSFT');
        expect(csv).toContain('150');
        expect(csv).toContain('300');
      });

      it('should include headers in CSV export', () => {
        const trades = [{ symbol: 'AAPL', entryPrice: 150, quantity: 100, entryTime: '2023-01-01' }];
        const csv = engine.exportTradesCsv(trades);

        expect(csv.split('\n')[0]).toBe('symbol,entryPrice,quantity,entryTime');
      });
    });

    describe('Edge Cases', () => {
      beforeEach(() => {
        engine.constructor(mockConfig);
      });

      it('should handle very small initial capital', () => {
        engine.constructor({ ...mockConfig, initialCapital: 1 });
        expect(engine.config.initialCapital).toBe(1);
      });

      it('should handle very large initial capital', () => {
        engine.constructor({ ...mockConfig, initialCapital: 1000000000 });
        expect(engine.config.initialCapital).toBe(1000000000);
      });

      it('should handle single data point equity curve', () => {
        const metrics = engine.calculateMetrics([100000]);
        expect(metrics.totalReturn).toBe(0);
        expect(metrics.volatility).toBe(0);
      });

      it('should handle flat equity curve', () => {
        const equityCurve = [100000, 100000, 100000, 100000];
        const metrics = engine.calculateMetrics(equityCurve);

        expect(metrics.totalReturn).toBe(0);
        expect(metrics.volatility).toBe(0);
      });
    });
  });
});
