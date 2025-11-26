/**
 * Input validation schemas for @neural-trader/portfolio
 * Uses Zod for runtime validation of portfolio configurations and operations
 */

import { z } from 'zod';

// Portfolio manager initialization schema
export const portfolioConfigSchema = z.object({
  initialCash: z.number()
    .positive('Initial cash must be positive')
    .finite('Initial cash must be finite'),

  baseCurrency: z.string()
    .length(3, 'Currency code must be 3 characters')
    .regex(/^[A-Z]+$/, 'Currency code must be uppercase letters')
    .default('USD'),

  tradingCosts: z.object({
    commissionPercentage: z.number()
      .min(0, 'Commission percentage cannot be negative')
      .max(1, 'Commission percentage should be between 0 and 1')
      .default(0.001),

    slippagePercentage: z.number()
      .min(0, 'Slippage percentage cannot be negative')
      .max(1, 'Slippage percentage should be between 0 and 1')
      .default(0.0005),

    spreadPercentage: z.number()
      .min(0, 'Spread percentage cannot be negative')
      .max(1, 'Spread percentage should be between 0 and 1')
      .default(0.0001),
  }).optional(),

  riskManagement: z.object({
    maxPositionPercentage: z.number()
      .min(0, 'Max position percentage must be positive')
      .max(1, 'Max position percentage should be between 0 and 1')
      .default(0.1),

    maxDailyLossPercentage: z.number()
      .min(0, 'Max daily loss percentage must be positive')
      .max(1, 'Max daily loss percentage should be between 0 and 1')
      .default(0.05),

    minTradeSize: z.number()
      .min(0, 'Min trade size cannot be negative')
      .default(10),
  }).optional(),
});

// Position management schemas
export const symbolSchema = z.string()
  .min(1, 'Symbol cannot be empty')
  .max(10, 'Symbol too long')
  .regex(/^[A-Z0-9-]+$/, 'Invalid symbol format');

export const positionUpdateSchema = z.object({
  symbol: symbolSchema,

  quantity: z.number()
    .finite('Quantity must be finite'),

  price: z.number()
    .positive('Price must be positive')
    .finite('Price must be finite'),

  entryPrice: z.number()
    .positive('Entry price must be positive')
    .finite('Entry price must be finite')
    .optional(),

  timestamp: z.number()
    .int('Timestamp must be an integer')
    .positive('Timestamp must be positive')
    .optional(),
})
  .refine((data) => {
    // Quantity should be non-zero (either positive or negative)
    return data.quantity !== 0;
  }, {
    message: 'Quantity cannot be zero',
    path: ['quantity'],
  });

// Portfolio optimization configuration
export const optimizerConfigSchema = z.object({
  method: z.enum(['markowitz', 'risk_parity', 'min_variance', 'max_sharpe'], {
    errorMap: () => ({ message: 'Invalid optimization method' }),
  })
    .default('markowitz'),

  constraints: z.object({
    minAllocation: z.number()
      .min(0, 'Min allocation cannot be negative')
      .max(1, 'Min allocation should be between 0 and 1')
      .default(0),

    maxAllocation: z.number()
      .min(0, 'Max allocation must be positive')
      .max(1, 'Max allocation should be between 0 and 1')
      .default(1),

    targetVolatility: z.number()
      .min(0, 'Target volatility cannot be negative')
      .max(1, 'Target volatility should be between 0 and 1')
      .optional(),

    minReturn: z.number()
      .optional(),

    leverageLimit: z.number()
      .min(1, 'Leverage limit must be at least 1')
      .max(10, 'Leverage limit too high')
      .optional(),
  }).optional(),

  rebalancingPeriod: z.number()
    .int('Rebalancing period must be an integer')
    .min(1, 'Rebalancing period must be at least 1')
    .max(365, 'Rebalancing period too high')
    .optional(),
});

// Optimization input validation
export const optimizationInputSchema = z.object({
  symbols: z.array(symbolSchema)
    .min(2, 'At least 2 symbols required')
    .max(100, 'Too many symbols'),

  returns: z.array(z.number().finite('Returns must be finite numbers'))
    .min(2, 'At least 2 return values required')
    .refine((arr) => arr.length > 0, 'Returns array cannot be empty'),

  covariance: z.array(z.array(z.number().finite('Covariance values must be finite')))
    .refine((cov) => {
      // Covariance matrix should be square
      if (cov.length === 0) return false;
      const n = cov.length;
      return cov.every((row) => row.length === n);
    }, 'Covariance matrix must be square'),
})
  .refine((data) => {
    // Returns array length should match covariance matrix dimensions
    return data.returns.length === data.symbols.length &&
           data.covariance.length === data.symbols.length;
  }, {
    message: 'Returns and covariance dimensions must match symbols length',
    path: ['returns'],
  });

// Position size calculation schema
export const positionSizeInputSchema = z.object({
  portfolioValue: z.number()
    .positive('Portfolio value must be positive'),

  price: z.number()
    .positive('Price must be positive'),

  quantity: z.number()
    .positive('Quantity must be positive'),

  maxPositionPercentage: z.number()
    .min(0, 'Max position percentage must be positive')
    .max(1, 'Max position percentage should be between 0 and 1'),

  riskPerTrade: z.number()
    .min(0, 'Risk per trade must be non-negative')
    .max(1, 'Risk per trade should be between 0 and 1'),
});

// Risk metrics calculation schema
export const riskMetricsInputSchema = z.object({
  positions: z.record(
    symbolSchema,
    z.number().finite('Position value must be finite')
  ),

  prices: z.record(
    symbolSchema,
    z.number()
      .positive('Price must be positive')
      .finite('Price must be finite')
  ),

  returns: z.array(z.number().finite('Returns must be finite'))
    .optional(),
});

// Validation helper functions
export function validatePortfolioConfig(config: unknown) {
  try {
    return portfolioConfigSchema.parse(config);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const messages = error.errors
        .map((err) => `${err.path.join('.')}: ${err.message}`)
        .join('; ');
      throw new ValidationError(`Invalid portfolio configuration: ${messages}`, error);
    }
    throw error;
  }
}

export function validatePositionUpdate(update: unknown) {
  try {
    return positionUpdateSchema.parse(update);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const messages = error.errors
        .map((err) => `${err.path.join('.')}: ${err.message}`)
        .join('; ');
      throw new ValidationError(`Invalid position update: ${messages}`, error);
    }
    throw error;
  }
}

export function validateOptimizerConfig(config: unknown) {
  try {
    return optimizerConfigSchema.parse(config);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const messages = error.errors
        .map((err) => `${err.path.join('.')}: ${err.message}`)
        .join('; ');
      throw new ValidationError(`Invalid optimizer configuration: ${messages}`, error);
    }
    throw error;
  }
}

export function validateOptimizationInput(input: unknown) {
  try {
    return optimizationInputSchema.parse(input);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const messages = error.errors
        .map((err) => `${err.path.join('.')}: ${err.message}`)
        .join('; ');
      throw new ValidationError(`Invalid optimization input: ${messages}`, error);
    }
    throw error;
  }
}

export function validateSymbol(symbol: unknown) {
  try {
    return symbolSchema.parse(symbol);
  } catch (error) {
    if (error instanceof z.ZodError) {
      throw new ValidationError(`Invalid symbol: ${error.errors[0].message}`, error);
    }
    throw error;
  }
}

export function validateRiskMetricsInput(input: unknown) {
  try {
    return riskMetricsInputSchema.parse(input);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const messages = error.errors
        .map((err) => `${err.path.join('.')}: ${err.message}`)
        .join('; ');
      throw new ValidationError(`Invalid risk metrics input: ${messages}`, error);
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
