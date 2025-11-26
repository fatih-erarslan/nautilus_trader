/**
 * Tests for portfolio validation schemas
 */

import {
  portfolioConfigSchema,
  symbolSchema,
  positionUpdateSchema,
  optimizerConfigSchema,
  optimizationInputSchema,
  validatePortfolioConfig,
  validatePositionUpdate,
  validateOptimizerConfig,
  validateOptimizationInput,
  validateSymbol,
  ValidationError,
} from './validation';

describe('Portfolio Validation Schemas', () => {
  describe('portfolioConfigSchema', () => {
    it('should validate valid portfolio config', () => {
      const config = {
        initialCash: 100000,
        baseCurrency: 'USD',
      };
      const result = portfolioConfigSchema.parse(config);
      expect(result.initialCash).toBe(100000);
      expect(result.baseCurrency).toBe('USD');
    });

    it('should set default values', () => {
      const config = { initialCash: 100000 };
      const result = portfolioConfigSchema.parse(config);
      expect(result.baseCurrency).toBe('USD');
    });

    it('should reject zero or negative initial cash', () => {
      expect(() =>
        portfolioConfigSchema.parse({
          initialCash: 0,
        })
      ).toThrow();

      expect(() =>
        portfolioConfigSchema.parse({
          initialCash: -100000,
        })
      ).toThrow();
    });

    it('should reject non-finite initial cash', () => {
      expect(() =>
        portfolioConfigSchema.parse({
          initialCash: Infinity,
        })
      ).toThrow();
    });

    it('should accept trading costs config', () => {
      const config = {
        initialCash: 100000,
        tradingCosts: {
          commissionPercentage: 0.002,
          slippagePercentage: 0.0005,
        },
      };
      const result = portfolioConfigSchema.parse(config);
      expect(result.tradingCosts).toBeDefined();
    });

    it('should reject invalid commission percentage', () => {
      expect(() =>
        portfolioConfigSchema.parse({
          initialCash: 100000,
          tradingCosts: {
            commissionPercentage: 1.5,
          },
        })
      ).toThrow();
    });
  });

  describe('symbolSchema', () => {
    it('should validate valid symbols', () => {
      expect(symbolSchema.parse('AAPL')).toBe('AAPL');
      expect(symbolSchema.parse('BRK-B')).toBe('BRK-B');
      expect(symbolSchema.parse('SPY')).toBe('SPY');
    });

    it('should reject lowercase symbols', () => {
      expect(() => symbolSchema.parse('aapl')).toThrow();
    });

    it('should reject symbols with invalid characters', () => {
      expect(() => symbolSchema.parse('AAPL@')).toThrow();
      expect(() => symbolSchema.parse('AAPL ')).toThrow();
    });

    it('should reject empty or too long symbols', () => {
      expect(() => symbolSchema.parse('')).toThrow();
      expect(() => symbolSchema.parse('A'.repeat(11))).toThrow();
    });
  });

  describe('positionUpdateSchema', () => {
    it('should validate valid position update', () => {
      const update = {
        symbol: 'AAPL',
        quantity: 100,
        price: 150.25,
      };
      expect(positionUpdateSchema.parse(update)).toEqual(update);
    });

    it('should reject zero quantity', () => {
      expect(() =>
        positionUpdateSchema.parse({
          symbol: 'AAPL',
          quantity: 0,
          price: 150,
        })
      ).toThrow();
    });

    it('should accept negative quantity (shorting)', () => {
      const update = {
        symbol: 'AAPL',
        quantity: -100,
        price: 150,
      };
      expect(positionUpdateSchema.parse(update)).toEqual(update);
    });

    it('should reject non-finite quantity or price', () => {
      expect(() =>
        positionUpdateSchema.parse({
          symbol: 'AAPL',
          quantity: Infinity,
          price: 150,
        })
      ).toThrow();

      expect(() =>
        positionUpdateSchema.parse({
          symbol: 'AAPL',
          quantity: 100,
          price: -150,
        })
      ).toThrow();
    });

    it('should accept optional fields', () => {
      const update = {
        symbol: 'AAPL',
        quantity: 100,
        price: 150,
        entryPrice: 145,
        timestamp: Date.now(),
      };
      expect(positionUpdateSchema.parse(update)).toEqual(update);
    });
  });

  describe('optimizerConfigSchema', () => {
    it('should validate valid optimizer config', () => {
      const config = {
        method: 'markowitz',
        constraints: {
          minAllocation: 0.05,
          maxAllocation: 0.3,
          targetVolatility: 0.15,
        },
      };
      expect(optimizerConfigSchema.parse(config)).toBeDefined();
    });

    it('should set default method', () => {
      const config = {
        constraints: {
          minAllocation: 0,
          maxAllocation: 1,
        },
      };
      const result = optimizerConfigSchema.parse(config);
      expect(result.method).toBe('markowitz');
    });

    it('should reject invalid optimization method', () => {
      expect(() =>
        optimizerConfigSchema.parse({
          method: 'invalid_method',
        })
      ).toThrow();
    });

    it('should reject invalid constraint values', () => {
      expect(() =>
        optimizerConfigSchema.parse({
          constraints: {
            minAllocation: 1.5,
          },
        })
      ).toThrow();
    });
  });

  describe('optimizationInputSchema', () => {
    it('should validate valid optimization input', () => {
      const input = {
        symbols: ['AAPL', 'MSFT'],
        returns: [0.1, 0.15],
        covariance: [[0.04, 0.006], [0.006, 0.09]],
      };
      expect(optimizationInputSchema.parse(input)).toEqual(input);
    });

    it('should reject < 2 symbols', () => {
      expect(() =>
        optimizationInputSchema.parse({
          symbols: ['AAPL'],
          returns: [0.1],
          covariance: [[0.04]],
        })
      ).toThrow();
    });

    it('should reject mismatched dimensions', () => {
      expect(() =>
        optimizationInputSchema.parse({
          symbols: ['AAPL', 'MSFT'],
          returns: [0.1],
          covariance: [[0.04, 0.006], [0.006, 0.09]],
        })
      ).toThrow();
    });

    it('should reject non-square covariance matrix', () => {
      expect(() =>
        optimizationInputSchema.parse({
          symbols: ['AAPL', 'MSFT'],
          returns: [0.1, 0.15],
          covariance: [[0.04, 0.006, 0.01], [0.006, 0.09, 0.01]],
        })
      ).toThrow();
    });

    it('should reject non-finite values', () => {
      expect(() =>
        optimizationInputSchema.parse({
          symbols: ['AAPL', 'MSFT'],
          returns: [0.1, Infinity],
          covariance: [[0.04, 0.006], [0.006, 0.09]],
        })
      ).toThrow();
    });
  });

  describe('Validation helper functions', () => {
    describe('validatePortfolioConfig', () => {
      it('should throw ValidationError on invalid config', () => {
        expect(() =>
          validatePortfolioConfig({
            initialCash: -100000,
          })
        ).toThrow(ValidationError);
      });

      it('should return config on valid input', () => {
        const config = { initialCash: 100000 };
        const result = validatePortfolioConfig(config);
        expect(result.initialCash).toBe(100000);
      });
    });

    describe('validateSymbol', () => {
      it('should throw ValidationError on invalid symbol', () => {
        expect(() => validateSymbol('aapl')).toThrow(ValidationError);
      });

      it('should return symbol on valid input', () => {
        const result = validateSymbol('AAPL');
        expect(result).toBe('AAPL');
      });
    });

    describe('validatePositionUpdate', () => {
      it('should throw ValidationError on zero quantity', () => {
        expect(() =>
          validatePositionUpdate({
            symbol: 'AAPL',
            quantity: 0,
            price: 150,
          })
        ).toThrow(ValidationError);
      });
    });

    describe('validateOptimizerConfig', () => {
      it('should throw ValidationError on invalid method', () => {
        expect(() =>
          validateOptimizerConfig({
            method: 'invalid',
          })
        ).toThrow(ValidationError);
      });
    });

    describe('validateOptimizationInput', () => {
      it('should throw ValidationError on mismatched dimensions', () => {
        expect(() =>
          validateOptimizationInput({
            symbols: ['AAPL'],
            returns: [0.1, 0.15],
            covariance: [[0.04]],
          })
        ).toThrow(ValidationError);
      });
    });
  });

  describe('ValidationError class', () => {
    it('should create error with message', () => {
      const error = new ValidationError('Test error');
      expect(error.message).toBe('Test error');
      expect(error.name).toBe('ValidationError');
    });
  });
});
