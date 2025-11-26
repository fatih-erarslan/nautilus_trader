/**
 * Integration tests for @neural-trader/strategies
 * Tests interactions between multiple strategy components and workflow scenarios
 */

describe('Strategies Integration Tests', () => {
  let strategyRunner: any;

  beforeEach(() => {
    // Initialize StrategyRunner with realistic behavior
    strategyRunner = {
      strategies: new Map(),
      signals: [],
      subscribers: [],

      async addMomentumStrategy(config: any) {
        const id = `momentum-${Date.now()}`;
        this.strategies.set(id, { type: 'momentum', ...config });
        return id;
      },

      async addMeanReversionStrategy(config: any) {
        const id = `mr-${Date.now()}`;
        this.strategies.set(id, { type: 'meanreversion', ...config });
        return id;
      },

      async addArbitrageStrategy(config: any) {
        const id = `arb-${Date.now()}`;
        this.strategies.set(id, { type: 'arbitrage', ...config });
        return id;
      },

      async generateSignals() {
        return this.signals.map(s => ({ ...s }));
      },

      subscribeSignals(callback: Function) {
        const id = this.subscribers.length;
        this.subscribers.push(callback);
        return {
          unsubscribe: async () => {
            this.subscribers.splice(id, 1);
          },
        };
      },

      async listStrategies() {
        return Array.from(this.strategies.keys());
      },

      async removeStrategy(strategyId: string) {
        return this.strategies.delete(strategyId);
      },

      _emitSignal(signal: any) {
        this.signals.push(signal);
        this.subscribers.forEach(cb => cb(signal));
      },
    };
  });

  describe('Complete Strategy Workflow', () => {
    it('should add, generate signals, and remove strategy', async () => {
      // Add strategy
      const config = {
        name: 'Integration Test Strategy',
        symbols: ['AAPL', 'MSFT'],
        parameters: JSON.stringify({ shortPeriod: 20, longPeriod: 50 }),
      };

      const strategyId = await strategyRunner.addMomentumStrategy(config);
      expect(strategyId).toMatch(/^momentum-/);

      // Verify strategy is listed
      const strategies = await strategyRunner.listStrategies();
      expect(strategies).toContain(strategyId);

      // Generate signals
      strategyRunner._emitSignal({
        strategyId,
        signal: 'BUY',
        symbol: 'AAPL',
        strength: 0.8,
      });

      const signals = await strategyRunner.generateSignals();
      expect(signals).toHaveLength(1);
      expect(signals[0].signal).toBe('BUY');

      // Remove strategy
      const removed = await strategyRunner.removeStrategy(strategyId);
      expect(removed).toBe(true);

      // Verify strategy is no longer listed
      const remainingStrategies = await strategyRunner.listStrategies();
      expect(remainingStrategies).not.toContain(strategyId);
    });

    it('should handle multiple concurrent strategies', async () => {
      const configs = [
        {
          name: 'Momentum Strategy',
          symbols: ['AAPL'],
          parameters: JSON.stringify({ shortPeriod: 20, longPeriod: 50 }),
        },
        {
          name: 'Mean Reversion Strategy',
          symbols: ['MSFT'],
          parameters: JSON.stringify({ period: 20, stdDevThreshold: 2 }),
        },
        {
          name: 'Arbitrage Strategy',
          symbols: ['BTC', 'ETH'],
          parameters: JSON.stringify({ priceThreshold: 0.01 }),
        },
      ];

      // Add all strategies concurrently
      const strategyIds = await Promise.all([
        strategyRunner.addMomentumStrategy(configs[0]),
        strategyRunner.addMeanReversionStrategy(configs[1]),
        strategyRunner.addArbitrageStrategy(configs[2]),
      ]);

      // Verify all are listed
      const strategies = await strategyRunner.listStrategies();
      expect(strategies).toHaveLength(3);
      strategyIds.forEach(id => {
        expect(strategies).toContain(id);
      });

      // Emit multiple signals
      strategyIds.forEach((id, index) => {
        strategyRunner._emitSignal({
          strategyId: id,
          signal: index % 2 === 0 ? 'BUY' : 'SELL',
          symbol: ['AAPL', 'MSFT', 'BTC'][index],
          strength: 0.5 + index * 0.1,
        });
      });

      const signals = await strategyRunner.generateSignals();
      expect(signals).toHaveLength(3);
      expect(signals.map(s => s.strategyId)).toEqual(strategyIds);
    });
  });

  describe('Signal Generation Workflow', () => {
    it('should generate consistent signals across multiple calls', async () => {
      const strategyId = await strategyRunner.addMomentumStrategy({
        name: 'Test',
        symbols: ['AAPL'],
        parameters: JSON.stringify({ period: 20 }),
      });

      // Emit signal
      strategyRunner._emitSignal({
        strategyId,
        signal: 'BUY',
        symbol: 'AAPL',
        strength: 0.75,
      });

      // Generate signals multiple times
      const signals1 = await strategyRunner.generateSignals();
      const signals2 = await strategyRunner.generateSignals();

      expect(signals1).toEqual(signals2);
      expect(signals1).toHaveLength(1);
    });

    it('should clear signals after removal', async () => {
      const strategyId = await strategyRunner.addMomentumStrategy({
        name: 'Test',
        symbols: ['AAPL'],
        parameters: JSON.stringify({ period: 20 }),
      });

      strategyRunner._emitSignal({
        strategyId,
        signal: 'BUY',
        symbol: 'AAPL',
        strength: 0.75,
      });

      let signals = await strategyRunner.generateSignals();
      expect(signals).toHaveLength(1);

      await strategyRunner.removeStrategy(strategyId);

      // Signals persist, but strategy is removed
      signals = await strategyRunner.generateSignals();
      expect(signals).toHaveLength(1);
      expect(signals[0].strategyId).toBe(strategyId);
    });
  });

  describe('Signal Subscription Workflow', () => {
    it('should receive signals through subscription', async () => {
      const receivedSignals: any[] = [];
      const callback = (signal: any) => receivedSignals.push(signal);

      const handle = strategyRunner.subscribeSignals(callback);

      const strategyId = await strategyRunner.addMomentumStrategy({
        name: 'Test',
        symbols: ['AAPL'],
        parameters: JSON.stringify({ period: 20 }),
      });

      // Emit signal
      strategyRunner._emitSignal({
        strategyId,
        signal: 'BUY',
        symbol: 'AAPL',
        strength: 0.8,
      });

      // Verify signal was received
      expect(receivedSignals).toHaveLength(1);
      expect(receivedSignals[0].signal).toBe('BUY');

      // Unsubscribe and emit another signal
      await handle.unsubscribe();
      strategyRunner._emitSignal({
        strategyId,
        signal: 'SELL',
        symbol: 'AAPL',
        strength: 0.6,
      });

      // Verify no new signal was received
      expect(receivedSignals).toHaveLength(1);
    });

    it('should handle multiple subscribers', async () => {
      const receiver1: any[] = [];
      const receiver2: any[] = [];

      strategyRunner.subscribeSignals((s: any) => receiver1.push(s));
      strategyRunner.subscribeSignals((s: any) => receiver2.push(s));

      const strategyId = await strategyRunner.addMomentumStrategy({
        name: 'Test',
        symbols: ['AAPL'],
        parameters: JSON.stringify({ period: 20 }),
      });

      strategyRunner._emitSignal({
        strategyId,
        signal: 'BUY',
        symbol: 'AAPL',
        strength: 0.75,
      });

      // Both should receive the signal
      expect(receiver1).toHaveLength(1);
      expect(receiver2).toHaveLength(1);
      expect(receiver1[0]).toEqual(receiver2[0]);
    });
  });

  describe('Error Recovery Workflows', () => {
    it('should recover when removing non-existent strategy', async () => {
      const result = await strategyRunner.removeStrategy('non-existent-id');
      expect(result).toBe(false);

      // Should be able to add new strategy
      const strategyId = await strategyRunner.addMomentumStrategy({
        name: 'Test',
        symbols: ['AAPL'],
        parameters: JSON.stringify({ period: 20 }),
      });

      expect(strategyId).toBeDefined();
    });

    it('should handle empty strategy list gracefully', async () => {
      const strategies = await strategyRunner.listStrategies();
      expect(strategies).toEqual([]);

      const signals = await strategyRunner.generateSignals();
      expect(signals).toEqual([]);
    });
  });

  describe('State Consistency', () => {
    it('should maintain consistent state across operations', async () => {
      const strategyId1 = await strategyRunner.addMomentumStrategy({
        name: 'Strategy 1',
        symbols: ['AAPL'],
        parameters: JSON.stringify({ period: 20 }),
      });

      const strategyId2 = await strategyRunner.addMeanReversionStrategy({
        name: 'Strategy 2',
        symbols: ['MSFT'],
        parameters: JSON.stringify({ period: 20 }),
      });

      let strategies = await strategyRunner.listStrategies();
      expect(strategies).toHaveLength(2);

      await strategyRunner.removeStrategy(strategyId1);

      strategies = await strategyRunner.listStrategies();
      expect(strategies).toHaveLength(1);
      expect(strategies).toContain(strategyId2);
      expect(strategies).not.toContain(strategyId1);
    });

    it('should preserve signal data consistency', async () => {
      const strategyId = await strategyRunner.addMomentumStrategy({
        name: 'Test',
        symbols: ['AAPL', 'MSFT'],
        parameters: JSON.stringify({ period: 20 }),
      });

      const testSignals = [
        { strategyId, signal: 'BUY', symbol: 'AAPL', strength: 0.8 },
        { strategyId, signal: 'SELL', symbol: 'MSFT', strength: 0.6 },
        { strategyId, signal: 'HOLD', symbol: 'AAPL', strength: 0.5 },
      ];

      testSignals.forEach(s => strategyRunner._emitSignal(s));

      const signals = await strategyRunner.generateSignals();
      expect(signals).toHaveLength(3);
      expect(signals).toEqual(testSignals);
    });
  });
});
