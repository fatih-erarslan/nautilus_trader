/**
 * Input validation schemas for @neural-trader/risk
 * Uses Zod for runtime validation of risk management configurations and calculations
 */

import { z } from 'zod';

// Risk manager configuration schema
export const riskConfigSchema = z.object({
  confidenceLevel: z.number()
    .min(0, 'Confidence level cannot be negative')
    .max(1, 'Confidence level should be between 0 and 1')
    .default(0.95),

  riskFreeRate: z.number()
    .min(0, 'Risk-free rate cannot be negative')
    .max(1, 'Risk-free rate should be between 0 and 1')
    .default(0.02),

  lookbackPeriod: z.number()
    .int('Lookback period must be an integer')
    .min(1, 'Lookback period must be at least 1')
    .max(5000, 'Lookback period too large')
    .default(252),

  maxDrawdown: z.number()
    .min(0, 'Max drawdown must be positive')
    .max(1, 'Max drawdown should be between 0 and 1')
    .default(0.1),

  maxPositionSize: z.number()
    .min(0, 'Max position size must be positive')
    .max(1, 'Max position size should be between 0 and 1')
    .default(0.1),

  maxLeverage: z.number()
    .min(1, 'Max leverage must be at least 1')
    .max(100, 'Max leverage too high')
    .default(2),

  annualizationFactor: z.number()
    .min(1, 'Annualization factor must be at least 1')
    .default(252),
});

// Returns data validation
export const returnsSchema = z.array(z.number()
  .finite('Returns must be finite numbers'))
  .min(2, 'At least 2 return values required')
  .refine((arr) => !arr.some((val) => Number.isNaN(val)), 'Returns cannot contain NaN values');

// Equity curve validation
export const equityCurveSchema = z.array(z.number()
  .positive('Equity values must be positive')
  .finite('Equity values must be finite'))
  .min(2, 'At least 2 equity values required')
  .refine((arr) => !arr.some((val) => Number.isNaN(val)), 'Equity curve cannot contain NaN values');

// VAR calculation input
export const varInputSchema = z.object({
  returns: returnsSchema,
  portfolioValue: z.number()
    .positive('Portfolio value must be positive')
    .finite('Portfolio value must be finite'),
  confidenceLevel: z.number()
    .min(0, 'Confidence level must be between 0 and 1')
    .max(1, 'Confidence level must be between 0 and 1')
    .optional(),
});

// CVAR calculation input
export const cvarInputSchema = z.object({
  returns: returnsSchema,
  portfolioValue: z.number()
    .positive('Portfolio value must be positive')
    .finite('Portfolio value must be finite'),
  confidenceLevel: z.number()
    .min(0, 'Confidence level must be between 0 and 1')
    .max(1, 'Confidence level must be between 0 and 1')
    .optional(),
});

// Kelly criterion calculation input
export const kellyInputSchema = z.object({
  winRate: z.number()
    .min(0, 'Win rate must be between 0 and 1')
    .max(1, 'Win rate must be between 0 and 1'),

  avgWin: z.number()
    .positive('Average win must be positive')
    .finite('Average win must be finite'),

  avgLoss: z.number()
    .positive('Average loss must be positive')
    .finite('Average loss must be finite'),
})
  .refine((data) => {
    // Verify win rate is reasonable given at least 1 trade
    return data.winRate > 0 && data.winRate < 1;
  }, {
    message: 'Invalid win rate for Kelly calculation',
    path: ['winRate'],
  });

// Drawdown calculation input
export const drawdownInputSchema = z.object({
  equityCurve: equityCurveSchema,
});

// Position sizing calculation input
export const positionSizeInputSchema = z.object({
  portfolioValue: z.number()
    .positive('Portfolio value must be positive')
    .finite('Portfolio value must be finite'),

  pricePerShare: z.number()
    .positive('Price per share must be positive')
    .finite('Price per share must be finite'),

  riskPerTrade: z.number()
    .min(0, 'Risk per trade must be between 0 and 1')
    .max(1, 'Risk per trade must be between 0 and 1'),

  stopLossDistance: z.number()
    .positive('Stop loss distance must be positive')
    .finite('Stop loss distance must be finite'),
});

// Position validation input
export const validatePositionInputSchema = z.object({
  positionSize: z.number()
    .positive('Position size must be positive')
    .finite('Position size must be finite'),

  portfolioValue: z.number()
    .positive('Portfolio value must be positive')
    .finite('Portfolio value must be finite'),

  maxPositionPercentage: z.number()
    .min(0, 'Max position percentage must be between 0 and 1')
    .max(1, 'Max position percentage must be between 0 and 1'),
})
  .refine((data) => {
    // Check if position size is within limits
    const positionPercentage = data.positionSize / data.portfolioValue;
    return positionPercentage <= data.maxPositionPercentage;
  }, {
    message: 'Position size exceeds maximum allowed percentage',
    path: ['positionSize'],
  });

// Sharpe ratio calculation input
export const sharpeRatioInputSchema = z.object({
  returns: returnsSchema,
  riskFreeRate: z.number()
    .min(0, 'Risk-free rate cannot be negative')
    .max(1, 'Risk-free rate should be between 0 and 1'),
  annualizationFactor: z.number()
    .min(1, 'Annualization factor must be at least 1')
    .default(252),
});

// Sortino ratio calculation input
export const sortinoRatioInputSchema = z.object({
  returns: returnsSchema,
  targetReturn: z.number()
    .min(-1, 'Target return must be valid')
    .max(1, 'Target return too high')
    .default(0),
  annualizationFactor: z.number()
    .min(1, 'Annualization factor must be at least 1')
    .default(252),
});

// Max leverage calculation input
export const maxLeverageInputSchema = z.object({
  portfolioValue: z.number()
    .positive('Portfolio value must be positive')
    .finite('Portfolio value must be finite'),

  volatility: z.number()
    .min(0, 'Volatility cannot be negative')
    .finite('Volatility must be finite'),

  maxVolatilityTarget: z.number()
    .min(0, 'Max volatility target must be positive')
    .max(1, 'Max volatility target should be between 0 and 1'),
});

// Validation helper functions
export function validateRiskConfig(config: unknown) {
  try {
    return riskConfigSchema.parse(config);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const messages = error.errors
        .map((err) => `${err.path.join('.')}: ${err.message}`)
        .join('; ');
      throw new ValidationError(`Invalid risk configuration: ${messages}`, error);
    }
    throw error;
  }
}

export function validateReturns(returns: unknown) {
  try {
    return returnsSchema.parse(returns);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const messages = error.errors
        .map((err) => `${err.path.join('.')}: ${err.message}`)
        .join('; ');
      throw new ValidationError(`Invalid returns data: ${messages}`, error);
    }
    throw error;
  }
}

export function validateEquityCurve(curve: unknown) {
  try {
    return equityCurveSchema.parse(curve);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const messages = error.errors
        .map((err) => `${err.path.join('.')}: ${err.message}`)
        .join('; ');
      throw new ValidationError(`Invalid equity curve: ${messages}`, error);
    }
    throw error;
  }
}

export function validateVarInput(input: unknown) {
  try {
    return varInputSchema.parse(input);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const messages = error.errors
        .map((err) => `${err.path.join('.')}: ${err.message}`)
        .join('; ');
      throw new ValidationError(`Invalid VAR input: ${messages}`, error);
    }
    throw error;
  }
}

export function validateCvarInput(input: unknown) {
  try {
    return cvarInputSchema.parse(input);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const messages = error.errors
        .map((err) => `${err.path.join('.')}: ${err.message}`)
        .join('; ');
      throw new ValidationError(`Invalid CVAR input: ${messages}`, error);
    }
    throw error;
  }
}

export function validateKellyInput(input: unknown) {
  try {
    return kellyInputSchema.parse(input);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const messages = error.errors
        .map((err) => `${err.path.join('.')}: ${err.message}`)
        .join('; ');
      throw new ValidationError(`Invalid Kelly input: ${messages}`, error);
    }
    throw error;
  }
}

export function validateDrawdownInput(input: unknown) {
  try {
    return drawdownInputSchema.parse(input);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const messages = error.errors
        .map((err) => `${err.path.join('.')}: ${err.message}`)
        .join('; ');
      throw new ValidationError(`Invalid drawdown input: ${messages}`, error);
    }
    throw error;
  }
}

export function validatePositionSizeInput(input: unknown) {
  try {
    return positionSizeInputSchema.parse(input);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const messages = error.errors
        .map((err) => `${err.path.join('.')}: ${err.message}`)
        .join('; ');
      throw new ValidationError(`Invalid position size input: ${messages}`, error);
    }
    throw error;
  }
}

export function validatePositionValue(input: unknown) {
  try {
    return validatePositionInputSchema.parse(input);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const messages = error.errors
        .map((err) => `${err.path.join('.')}: ${err.message}`)
        .join('; ');
      throw new ValidationError(`Invalid position: ${messages}`, error);
    }
    throw error;
  }
}

export function validateSharpeRatioInput(input: unknown) {
  try {
    return sharpeRatioInputSchema.parse(input);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const messages = error.errors
        .map((err) => `${err.path.join('.')}: ${err.message}`)
        .join('; ');
      throw new ValidationError(`Invalid Sharpe ratio input: ${messages}`, error);
    }
    throw error;
  }
}

export function validateSortinoRatioInput(input: unknown) {
  try {
    return sortinoRatioInputSchema.parse(input);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const messages = error.errors
        .map((err) => `${err.path.join('.')}: ${err.message}`)
        .join('; ');
      throw new ValidationError(`Invalid Sortino ratio input: ${messages}`, error);
    }
    throw error;
  }
}

export function validateMaxLeverageInput(input: unknown) {
  try {
    return maxLeverageInputSchema.parse(input);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const messages = error.errors
        .map((err) => `${err.path.join('.')}: ${err.message}`)
        .join('; ');
      throw new ValidationError(`Invalid max leverage input: ${messages}`, error);
    }
    throw error;
  }
}

// Custom validation error class
export class ValidationError extends Error {
  name = 'ValidationError';

  constructor(message: string, public originalError?: unknown) {
    super(message);
    Object.setPrototypeOf(this, ValidationError.prototype);
  }
}
