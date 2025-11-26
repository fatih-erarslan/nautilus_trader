/**
 * Tests for strategy validation schemas
 */

import {
  strategyConfigSchema,
  momentumParametersSchema,
  meanReversionParametersSchema,
  arbitrageParametersSchema,
  pairsTradingParametersSchema,
  strategyIdSchema,
  validateStrategyConfig,
  validateMomentumParameters,
  validateMeanReversionParameters,
  validateArbitrageParameters,
  validatePairsTradingParameters,
  validateStrategyId,
  ValidationError,
} from './validation';

describe('Strategy Validation Schemas', () => {
  describe('strategyConfigSchema', () => {
    it('should validate valid strategy config', () => {
      const config = {
        name: 'Test Strategy',
        symbols: ['AAPL', 'MSFT'],
        parameters: JSON.stringify({ period: 20 }),
      };
      expect(strategyConfigSchema.parse(config)).toEqual(config);
    });

    it('should reject empty name', () => {
      const config = {
        name: '',
        symbols: ['AAPL'],
        parameters: JSON.stringify({ period: 20 }),
      };
      expect(() => strategyConfigSchema.parse(config)).toThrow();
    });

    it('should reject invalid symbol format', () => {
      const config = {
        name: 'Test',
        symbols: ['aapl', 'msft'],
        parameters: JSON.stringify({ period: 20 }),
      };
      expect(() => strategyConfigSchema.parse(config)).toThrow();
    });

    it('should reject empty symbols array', () => {
      const config = {
        name: 'Test',
        symbols: [],
        parameters: JSON.stringify({ period: 20 }),
      };
      expect(() => strategyConfigSchema.parse(config)).toThrow();
    });

    it('should reject invalid JSON parameters', () => {
      const config = {
        name: 'Test',
        symbols: ['AAPL'],
        parameters: 'not json',
      };
      expect(() => strategyConfigSchema.parse(config)).toThrow();
    });

    it('should reject too many symbols', () => {
      const config = {
        name: 'Test',
        symbols: Array(101).fill('AAPL'),
        parameters: JSON.stringify({ period: 20 }),
      };
      expect(() => strategyConfigSchema.parse(config)).toThrow();
    });
  });

  describe('momentumParametersSchema', () => {
    it('should validate valid momentum parameters', () => {
      const params = {
        shortPeriod: 20,
        longPeriod: 50,
        minVolume: 1000000,
      };
      expect(momentumParametersSchema.parse(params)).toEqual(params);
    });

    it('should reject when short period >= long period', () => {
      const params = {
        shortPeriod: 50,
        longPeriod: 50,
      };
      expect(() => momentumParametersSchema.parse(params)).toThrow();
    });

    it('should reject zero or negative periods', () => {
      expect(() =>
        momentumParametersSchema.parse({
          shortPeriod: 0,
          longPeriod: 50,
        })
      ).toThrow();
    });

    it('should reject periods > 200 and 500', () => {
      expect(() =>
        momentumParametersSchema.parse({
          shortPeriod: 201,
          longPeriod: 50,
        })
      ).toThrow();

      expect(() =>
        momentumParametersSchema.parse({
          shortPeriod: 20,
          longPeriod: 501,
        })
      ).toThrow();
    });
  });

  describe('meanReversionParametersSchema', () => {
    it('should validate valid mean reversion parameters', () => {
      const params = {
        period: 20,
        stdDevThreshold: 2,
        rsiThreshold: 30,
      };
      expect(meanReversionParametersSchema.parse(params)).toEqual(params);
    });

    it('should reject zero or negative period', () => {
      expect(() =>
        meanReversionParametersSchema.parse({
          period: 0,
          stdDevThreshold: 2,
        })
      ).toThrow();
    });

    it('should reject invalid std dev threshold', () => {
      expect(() =>
        meanReversionParametersSchema.parse({
          period: 20,
          stdDevThreshold: -1,
        })
      ).toThrow();

      expect(() =>
        meanReversionParametersSchema.parse({
          period: 20,
          stdDevThreshold: 6,
        })
      ).toThrow();
    });

    it('should reject invalid RSI threshold', () => {
      expect(() =>
        meanReversionParametersSchema.parse({
          period: 20,
          stdDevThreshold: 2,
          rsiThreshold: 101,
        })
      ).toThrow();
    });
  });

  describe('arbitrageParametersSchema', () => {
    it('should validate valid arbitrage parameters', () => {
      const params = {
        priceThreshold: 0.01,
        minProfit: 100,
        exchangePairs: [
          { exchange1: 'BINANCE', exchange2: 'KRAKEN' },
        ],
      };
      expect(arbitrageParametersSchema.parse(params)).toEqual(params);
    });

    it('should reject invalid price threshold', () => {
      expect(() =>
        arbitrageParametersSchema.parse({
          priceThreshold: 1.5,
          minProfit: 100,
          exchangePairs: [{ exchange1: 'A', exchange2: 'B' }],
        })
      ).toThrow();
    });

    it('should reject negative min profit', () => {
      expect(() =>
        arbitrageParametersSchema.parse({
          priceThreshold: 0.01,
          minProfit: -100,
          exchangePairs: [{ exchange1: 'A', exchange2: 'B' }],
        })
      ).toThrow();
    });

    it('should reject empty exchange pairs', () => {
      expect(() =>
        arbitrageParametersSchema.parse({
          priceThreshold: 0.01,
          minProfit: 100,
          exchangePairs: [],
        })
      ).toThrow();
    });
  });

  describe('pairsTradingParametersSchema', () => {
    it('should validate valid pairs trading parameters', () => {
      const params = {
        symbol1: 'AAPL',
        symbol2: 'MSFT',
        cointegrationThreshold: 0.95,
        spreadThreshold: 5,
      };
      expect(pairsTradingParametersSchema.parse(params)).toEqual(params);
    });

    it('should reject invalid cointegration threshold', () => {
      expect(() =>
        pairsTradingParametersSchema.parse({
          symbol1: 'AAPL',
          symbol2: 'MSFT',
          cointegrationThreshold: 1.5,
          spreadThreshold: 5,
        })
      ).toThrow();
    });

    it('should reject negative spread threshold', () => {
      expect(() =>
        pairsTradingParametersSchema.parse({
          symbol1: 'AAPL',
          symbol2: 'MSFT',
          cointegrationThreshold: 0.95,
          spreadThreshold: -5,
        })
      ).toThrow();
    });
  });

  describe('strategyIdSchema', () => {
    it('should validate valid strategy ID', () => {
      expect(strategyIdSchema.parse('strategy-123')).toBe('strategy-123');
      expect(strategyIdSchema.parse('strat_001')).toBe('strat_001');
    });

    it('should reject empty ID', () => {
      expect(() => strategyIdSchema.parse('')).toThrow();
    });

    it('should reject invalid characters', () => {
      expect(() => strategyIdSchema.parse('strategy@123')).toThrow();
      expect(() => strategyIdSchema.parse('strategy 123')).toThrow();
    });
  });

  describe('Validation helper functions', () => {
    describe('validateStrategyConfig', () => {
      it('should throw ValidationError on invalid config', () => {
        expect(() =>
          validateStrategyConfig({
            name: '',
            symbols: [],
          })
        ).toThrow(ValidationError);
      });

      it('should return config on valid input', () => {
        const config = {
          name: 'Test',
          symbols: ['AAPL'],
          parameters: JSON.stringify({ period: 20 }),
        };
        const result = validateStrategyConfig(config);
        expect(result).toEqual(config);
      });
    });

    describe('validateMomentumParameters', () => {
      it('should throw ValidationError on invalid params', () => {
        expect(() =>
          validateMomentumParameters({
            shortPeriod: 50,
            longPeriod: 50,
          })
        ).toThrow(ValidationError);
      });
    });

    describe('validateStrategyId', () => {
      it('should throw ValidationError on invalid ID', () => {
        expect(() => validateStrategyId('')).toThrow(ValidationError);
      });

      it('should return ID on valid input', () => {
        const result = validateStrategyId('strategy-123');
        expect(result).toBe('strategy-123');
      });
    });
  });

  describe('ValidationError class', () => {
    it('should create error with message', () => {
      const error = new ValidationError('Test error');
      expect(error.message).toBe('Test error');
      expect(error.name).toBe('ValidationError');
    });

    it('should capture original error', () => {
      const originalError = new Error('Original');
      const error = new ValidationError('Test', originalError);
      expect(error.originalError).toBe(originalError);
    });
  });
});
