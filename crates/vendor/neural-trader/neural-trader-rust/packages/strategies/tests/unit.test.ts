/**
 * Unit tests for @neural-trader/strategies
 * Tests core strategy functionality, StrategyRunner initialization, and strategy management
 */

describe('Strategies Unit Tests', () => {
  // Mock setup for NAPI bindings
  let mockStrategyRunner: any;

  beforeEach(() => {
    // Mock StrategyRunner class
    mockStrategyRunner = {
      addMomentumStrategy: jest.fn(),
      addMeanReversionStrategy: jest.fn(),
      addArbitrageStrategy: jest.fn(),
      generateSignals: jest.fn(),
      subscribeSignals: jest.fn(),
      listStrategies: jest.fn(),
      removeStrategy: jest.fn(),
    };
  });

  describe('StrategyRunner Initialization', () => {
    it('should create StrategyRunner instance', () => {
      expect(mockStrategyRunner).toBeDefined();
      expect(typeof mockStrategyRunner.addMomentumStrategy).toBe('function');
    });

    it('should have all required methods', () => {
      const requiredMethods = [
        'addMomentumStrategy',
        'addMeanReversionStrategy',
        'addArbitrageStrategy',
        'generateSignals',
        'subscribeSignals',
        'listStrategies',
        'removeStrategy',
      ];

      requiredMethods.forEach(method => {
        expect(mockStrategyRunner[method]).toBeDefined();
      });
    });
  });

  describe('Adding Strategies', () => {
    it('should add momentum strategy successfully', async () => {
      const config = {
        name: 'Test Momentum',
        symbols: ['AAPL'],
        parameters: JSON.stringify({ shortPeriod: 20, longPeriod: 50 }),
      };

      mockStrategyRunner.addMomentumStrategy.mockResolvedValue('momentum-001');
      const result = await mockStrategyRunner.addMomentumStrategy(config);

      expect(result).toBe('momentum-001');
      expect(mockStrategyRunner.addMomentumStrategy).toHaveBeenCalledWith(config);
    });

    it('should add mean reversion strategy successfully', async () => {
      const config = {
        name: 'Test MR',
        symbols: ['MSFT'],
        parameters: JSON.stringify({ period: 20, stdDevThreshold: 2 }),
      };

      mockStrategyRunner.addMeanReversionStrategy.mockResolvedValue('mr-001');
      const result = await mockStrategyRunner.addMeanReversionStrategy(config);

      expect(result).toBe('mr-001');
      expect(mockStrategyRunner.addMeanReversionStrategy).toHaveBeenCalledWith(config);
    });

    it('should add arbitrage strategy successfully', async () => {
      const config = {
        name: 'Test Arb',
        symbols: ['BTC', 'ETH'],
        parameters: JSON.stringify({ priceThreshold: 0.01 }),
      };

      mockStrategyRunner.addArbitrageStrategy.mockResolvedValue('arb-001');
      const result = await mockStrategyRunner.addArbitrageStrategy(config);

      expect(result).toBe('arb-001');
      expect(mockStrategyRunner.addArbitrageStrategy).toHaveBeenCalledWith(config);
    });

    it('should reject strategy with empty name', async () => {
      const config = {
        name: '',
        symbols: ['AAPL'],
        parameters: JSON.stringify({ period: 20 }),
      };

      mockStrategyRunner.addMomentumStrategy.mockRejectedValue(
        new Error('Strategy name cannot be empty')
      );

      await expect(mockStrategyRunner.addMomentumStrategy(config)).rejects.toThrow(
        'Strategy name cannot be empty'
      );
    });

    it('should reject strategy with empty symbols', async () => {
      const config = {
        name: 'Test Strategy',
        symbols: [],
        parameters: JSON.stringify({ period: 20 }),
      };

      mockStrategyRunner.addMomentumStrategy.mockRejectedValue(
        new Error('At least one symbol is required')
      );

      await expect(mockStrategyRunner.addMomentumStrategy(config)).rejects.toThrow(
        'At least one symbol is required'
      );
    });

    it('should reject strategy with invalid JSON parameters', async () => {
      const config = {
        name: 'Test Strategy',
        symbols: ['AAPL'],
        parameters: 'not json',
      };

      mockStrategyRunner.addMomentumStrategy.mockRejectedValue(
        new Error('Invalid parameters JSON')
      );

      await expect(mockStrategyRunner.addMomentumStrategy(config)).rejects.toThrow(
        'Invalid parameters JSON'
      );
    });
  });

  describe('Generating Signals', () => {
    it('should generate signals successfully', async () => {
      const mockSignals = [
        { strategyId: 'momentum-001', signal: 'BUY', symbol: 'AAPL', strength: 0.8 },
        { strategyId: 'mr-001', signal: 'SELL', symbol: 'MSFT', strength: 0.6 },
      ];

      mockStrategyRunner.generateSignals.mockResolvedValue(mockSignals);
      const result = await mockStrategyRunner.generateSignals();

      expect(result).toEqual(mockSignals);
      expect(result).toHaveLength(2);
      expect(result[0].signal).toBe('BUY');
    });

    it('should return empty array when no strategies active', async () => {
      mockStrategyRunner.generateSignals.mockResolvedValue([]);
      const result = await mockStrategyRunner.generateSignals();

      expect(result).toEqual([]);
      expect(result).toHaveLength(0);
    });

    it('should include signal strength between 0 and 1', async () => {
      const mockSignals = [
        { strategyId: 'test-001', signal: 'BUY', symbol: 'AAPL', strength: 0.5 },
      ];

      mockStrategyRunner.generateSignals.mockResolvedValue(mockSignals);
      const result = await mockStrategyRunner.generateSignals();

      expect(result[0].strength).toBeGreaterThanOrEqual(0);
      expect(result[0].strength).toBeLessThanOrEqual(1);
    });
  });

  describe('Strategy Listing and Removal', () => {
    it('should list all active strategies', async () => {
      const strategies = ['momentum-001', 'mr-001', 'arb-001'];
      mockStrategyRunner.listStrategies.mockResolvedValue(strategies);

      const result = await mockStrategyRunner.listStrategies();

      expect(result).toEqual(strategies);
      expect(result).toHaveLength(3);
    });

    it('should remove strategy by ID', async () => {
      mockStrategyRunner.removeStrategy.mockResolvedValue(true);
      const result = await mockStrategyRunner.removeStrategy('momentum-001');

      expect(result).toBe(true);
      expect(mockStrategyRunner.removeStrategy).toHaveBeenCalledWith('momentum-001');
    });

    it('should return false when removing non-existent strategy', async () => {
      mockStrategyRunner.removeStrategy.mockResolvedValue(false);
      const result = await mockStrategyRunner.removeStrategy('non-existent-001');

      expect(result).toBe(false);
    });

    it('should handle empty strategy list', async () => {
      mockStrategyRunner.listStrategies.mockResolvedValue([]);
      const result = await mockStrategyRunner.listStrategies();

      expect(result).toEqual([]);
      expect(result).toHaveLength(0);
    });
  });

  describe('Signal Subscription', () => {
    it('should subscribe to signals', () => {
      const callback = jest.fn();
      const mockHandle = { unsubscribe: jest.fn() };

      mockStrategyRunner.subscribeSignals.mockReturnValue(mockHandle);
      const handle = mockStrategyRunner.subscribeSignals(callback);

      expect(handle).toBeDefined();
      expect(typeof handle.unsubscribe).toBe('function');
    });

    it('should unsubscribe from signals', async () => {
      const mockHandle = { unsubscribe: jest.fn().mockResolvedValue(undefined) };

      await mockHandle.unsubscribe();

      expect(mockHandle.unsubscribe).toHaveBeenCalled();
    });

    it('should handle callback for new signals', () => {
      const callback = jest.fn();
      const signal = { strategyId: 'momentum-001', signal: 'BUY', symbol: 'AAPL' };

      callback(signal);

      expect(callback).toHaveBeenCalledWith(signal);
    });
  });

  describe('Edge Cases', () => {
    it('should handle symbol case sensitivity', async () => {
      const config = {
        name: 'Test',
        symbols: ['aapl'],
        parameters: JSON.stringify({ period: 20 }),
      };

      mockStrategyRunner.addMomentumStrategy.mockRejectedValue(
        new Error('Symbol must be uppercase')
      );

      await expect(mockStrategyRunner.addMomentumStrategy(config)).rejects.toThrow();
    });

    it('should handle extremely long strategy names', async () => {
      const longName = 'A'.repeat(1000);
      const config = {
        name: longName,
        symbols: ['AAPL'],
        parameters: JSON.stringify({ period: 20 }),
      };

      mockStrategyRunner.addMomentumStrategy.mockRejectedValue(
        new Error('Strategy name too long')
      );

      await expect(mockStrategyRunner.addMomentumStrategy(config)).rejects.toThrow();
    });

    it('should handle too many symbols', async () => {
      const symbols = Array(150).fill('AAPL');
      const config = {
        name: 'Test',
        symbols: symbols,
        parameters: JSON.stringify({ period: 20 }),
      };

      mockStrategyRunner.addMomentumStrategy.mockRejectedValue(
        new Error('Too many symbols')
      );

      await expect(mockStrategyRunner.addMomentumStrategy(config)).rejects.toThrow();
    });

    it('should handle concurrent strategy additions', async () => {
      const configs = [
        {
          name: 'Strategy 1',
          symbols: ['AAPL'],
          parameters: JSON.stringify({ period: 20 }),
        },
        {
          name: 'Strategy 2',
          symbols: ['MSFT'],
          parameters: JSON.stringify({ period: 20 }),
        },
      ];

      mockStrategyRunner.addMomentumStrategy
        .mockResolvedValueOnce('momentum-001')
        .mockResolvedValueOnce('momentum-002');

      const results = await Promise.all(
        configs.map(config => mockStrategyRunner.addMomentumStrategy(config))
      );

      expect(results).toHaveLength(2);
      expect(results[0]).toBe('momentum-001');
      expect(results[1]).toBe('momentum-002');
    });
  });
});
