/**
 * Tests for risk validation schemas
 */

import {
  riskConfigSchema,
  returnsSchema,
  equityCurveSchema,
  varInputSchema,
  cvarInputSchema,
  kellyInputSchema,
  drawdownInputSchema,
  positionSizeInputSchema,
  validateRiskConfig,
  validateReturns,
  validateEquityCurve,
  validateVarInput,
  validateCvarInput,
  validateKellyInput,
  validateDrawdownInput,
  validatePositionSizeInput,
  ValidationError,
} from './validation';

describe('Risk Validation Schemas', () => {
  describe('riskConfigSchema', () => {
    it('should validate valid risk config', () => {
      const config = {
        confidenceLevel: 0.95,
        riskFreeRate: 0.02,
        lookbackPeriod: 252,
      };
      const result = riskConfigSchema.parse(config);
      expect(result.confidenceLevel).toBe(0.95);
      expect(result.riskFreeRate).toBe(0.02);
    });

    it('should set default values', () => {
      const config = {};
      const result = riskConfigSchema.parse(config);
      expect(result.confidenceLevel).toBe(0.95);
      expect(result.riskFreeRate).toBe(0.02);
      expect(result.lookbackPeriod).toBe(252);
      expect(result.maxDrawdown).toBe(0.1);
      expect(result.annualizationFactor).toBe(252);
    });

    it('should reject invalid confidence level', () => {
      expect(() =>
        riskConfigSchema.parse({
          confidenceLevel: 1.5,
        })
      ).toThrow();

      expect(() =>
        riskConfigSchema.parse({
          confidenceLevel: -0.1,
        })
      ).toThrow();
    });

    it('should reject invalid risk-free rate', () => {
      expect(() =>
        riskConfigSchema.parse({
          riskFreeRate: 1.5,
        })
      ).toThrow();
    });

    it('should reject invalid lookback period', () => {
      expect(() =>
        riskConfigSchema.parse({
          lookbackPeriod: 0,
        })
      ).toThrow();

      expect(() =>
        riskConfigSchema.parse({
          lookbackPeriod: 5001,
        })
      ).toThrow();
    });

    it('should reject invalid max leverage', () => {
      expect(() =>
        riskConfigSchema.parse({
          maxLeverage: 0.5,
        })
      ).toThrow();

      expect(() =>
        riskConfigSchema.parse({
          maxLeverage: 101,
        })
      ).toThrow();
    });
  });

  describe('returnsSchema', () => {
    it('should validate valid returns', () => {
      const returns = [0.01, 0.02, -0.01, 0.015];
      expect(returnsSchema.parse(returns)).toEqual(returns);
    });

    it('should reject < 2 values', () => {
      expect(() => returnsSchema.parse([0.01])).toThrow();
    });

    it('should reject non-finite values', () => {
      expect(() => returnsSchema.parse([0.01, Infinity])).toThrow();
      expect(() => returnsSchema.parse([0.01, NaN])).toThrow();
    });

    it('should accept negative returns', () => {
      const returns = [-0.05, -0.02, 0.01, -0.03];
      expect(returnsSchema.parse(returns)).toEqual(returns);
    });
  });

  describe('equityCurveSchema', () => {
    it('should validate valid equity curve', () => {
      const curve = [100000, 105000, 110000, 108000];
      expect(equityCurveSchema.parse(curve)).toEqual(curve);
    });

    it('should reject zero or negative values', () => {
      expect(() => equityCurveSchema.parse([100000, 0, 110000])).toThrow();
      expect(() => equityCurveSchema.parse([100000, -105000, 110000])).toThrow();
    });

    it('should reject non-finite values', () => {
      expect(() => equityCurveSchema.parse([100000, Infinity])).toThrow();
      expect(() => equityCurveSchema.parse([100000, NaN])).toThrow();
    });

    it('should reject < 2 values', () => {
      expect(() => equityCurveSchema.parse([100000])).toThrow();
    });
  });

  describe('varInputSchema', () => {
    it('should validate valid VAR input', () => {
      const input = {
        returns: [0.01, 0.02, -0.01],
        portfolioValue: 100000,
      };
      expect(varInputSchema.parse(input)).toEqual(input);
    });

    it('should reject non-positive portfolio value', () => {
      expect(() =>
        varInputSchema.parse({
          returns: [0.01, 0.02, -0.01],
          portfolioValue: 0,
        })
      ).toThrow();

      expect(() =>
        varInputSchema.parse({
          returns: [0.01, 0.02, -0.01],
          portfolioValue: -100000,
        })
      ).toThrow();
    });

    it('should accept optional confidence level', () => {
      const input = {
        returns: [0.01, 0.02, -0.01],
        portfolioValue: 100000,
        confidenceLevel: 0.99,
      };
      expect(varInputSchema.parse(input)).toEqual(input);
    });

    it('should reject invalid confidence level', () => {
      expect(() =>
        varInputSchema.parse({
          returns: [0.01, 0.02, -0.01],
          portfolioValue: 100000,
          confidenceLevel: 1.5,
        })
      ).toThrow();
    });
  });

  describe('cvarInputSchema', () => {
    it('should validate valid CVAR input', () => {
      const input = {
        returns: [0.01, 0.02, -0.01],
        portfolioValue: 100000,
      };
      expect(cvarInputSchema.parse(input)).toEqual(input);
    });

    it('should reject invalid returns', () => {
      expect(() =>
        cvarInputSchema.parse({
          returns: [0.01],
          portfolioValue: 100000,
        })
      ).toThrow();
    });
  });

  describe('kellyInputSchema', () => {
    it('should validate valid Kelly input', () => {
      const input = {
        winRate: 0.55,
        avgWin: 2,
        avgLoss: 1,
      };
      expect(kellyInputSchema.parse(input)).toEqual(input);
    });

    it('should reject invalid win rate', () => {
      expect(() =>
        kellyInputSchema.parse({
          winRate: 0,
          avgWin: 2,
          avgLoss: 1,
        })
      ).toThrow();

      expect(() =>
        kellyInputSchema.parse({
          winRate: 1,
          avgWin: 2,
          avgLoss: 1,
        })
      ).toThrow();

      expect(() =>
        kellyInputSchema.parse({
          winRate: 1.5,
          avgWin: 2,
          avgLoss: 1,
        })
      ).toThrow();
    });

    it('should reject non-positive avgWin', () => {
      expect(() =>
        kellyInputSchema.parse({
          winRate: 0.55,
          avgWin: 0,
          avgLoss: 1,
        })
      ).toThrow();
    });

    it('should reject non-positive avgLoss', () => {
      expect(() =>
        kellyInputSchema.parse({
          winRate: 0.55,
          avgWin: 2,
          avgLoss: 0,
        })
      ).toThrow();
    });
  });

  describe('drawdownInputSchema', () => {
    it('should validate valid drawdown input', () => {
      const input = {
        equityCurve: [100000, 105000, 102000, 110000],
      };
      expect(drawdownInputSchema.parse(input)).toEqual(input);
    });

    it('should reject invalid equity curve', () => {
      expect(() =>
        drawdownInputSchema.parse({
          equityCurve: [100000],
        })
      ).toThrow();
    });
  });

  describe('positionSizeInputSchema', () => {
    it('should validate valid position size input', () => {
      const input = {
        portfolioValue: 100000,
        pricePerShare: 150,
        riskPerTrade: 0.02,
        stopLossDistance: 5,
      };
      expect(positionSizeInputSchema.parse(input)).toEqual(input);
    });

    it('should reject non-positive portfolio value', () => {
      expect(() =>
        positionSizeInputSchema.parse({
          portfolioValue: 0,
          pricePerShare: 150,
          riskPerTrade: 0.02,
          stopLossDistance: 5,
        })
      ).toThrow();
    });

    it('should reject non-positive price per share', () => {
      expect(() =>
        positionSizeInputSchema.parse({
          portfolioValue: 100000,
          pricePerShare: 0,
          riskPerTrade: 0.02,
          stopLossDistance: 5,
        })
      ).toThrow();
    });

    it('should reject invalid risk per trade', () => {
      expect(() =>
        positionSizeInputSchema.parse({
          portfolioValue: 100000,
          pricePerShare: 150,
          riskPerTrade: -0.02,
          stopLossDistance: 5,
        })
      ).toThrow();

      expect(() =>
        positionSizeInputSchema.parse({
          portfolioValue: 100000,
          pricePerShare: 150,
          riskPerTrade: 1.5,
          stopLossDistance: 5,
        })
      ).toThrow();
    });

    it('should reject non-positive stop loss distance', () => {
      expect(() =>
        positionSizeInputSchema.parse({
          portfolioValue: 100000,
          pricePerShare: 150,
          riskPerTrade: 0.02,
          stopLossDistance: 0,
        })
      ).toThrow();
    });
  });

  describe('Validation helper functions', () => {
    describe('validateRiskConfig', () => {
      it('should throw ValidationError on invalid config', () => {
        expect(() =>
          validateRiskConfig({
            confidenceLevel: 1.5,
          })
        ).toThrow(ValidationError);
      });

      it('should return config on valid input', () => {
        const config = { confidenceLevel: 0.95 };
        const result = validateRiskConfig(config);
        expect(result.confidenceLevel).toBe(0.95);
      });
    });

    describe('validateReturns', () => {
      it('should throw ValidationError on invalid returns', () => {
        expect(() => validateReturns([0.01])).toThrow(ValidationError);
      });

      it('should return returns on valid input', () => {
        const returns = [0.01, 0.02, -0.01];
        const result = validateReturns(returns);
        expect(result).toEqual(returns);
      });
    });

    describe('validateEquityCurve', () => {
      it('should throw ValidationError on invalid curve', () => {
        expect(() => validateEquityCurve([100000])).toThrow(ValidationError);
      });

      it('should return curve on valid input', () => {
        const curve = [100000, 105000, 110000];
        const result = validateEquityCurve(curve);
        expect(result).toEqual(curve);
      });
    });

    describe('validateKellyInput', () => {
      it('should throw ValidationError on invalid Kelly input', () => {
        expect(() =>
          validateKellyInput({
            winRate: 0,
            avgWin: 2,
            avgLoss: 1,
          })
        ).toThrow(ValidationError);
      });

      it('should return input on valid Kelly input', () => {
        const input = {
          winRate: 0.55,
          avgWin: 2,
          avgLoss: 1,
        };
        const result = validateKellyInput(input);
        expect(result).toEqual(input);
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
