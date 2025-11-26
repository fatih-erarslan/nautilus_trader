/**
 * Input validation schemas for @neural-trader/strategies
 * Uses Zod for runtime validation of strategy configurations
 */

import { z } from 'zod';

// Base strategy configuration schema
export const strategyConfigSchema = z.object({
  name: z.string()
    .min(1, 'Strategy name cannot be empty')
    .max(255, 'Strategy name too long'),

  symbols: z.array(z.string()
    .min(1, 'Symbol cannot be empty')
    .max(10, 'Symbol too long')
    .regex(/^[A-Z0-9-]+$/, 'Invalid symbol format'))
    .min(1, 'At least one symbol required')
    .max(100, 'Too many symbols'),

  parameters: z.string()
    .transform((val) => {
      try {
        return JSON.parse(val);
      } catch {
        throw new Error('Parameters must be valid JSON');
      }
    })
    .refine((val) => typeof val === 'object' && val !== null, 'Parameters must be a valid JSON object'),
});

// Momentum strategy specific parameters
export const momentumParametersSchema = z.object({
  shortPeriod: z.number()
    .int('Period must be an integer')
    .min(1, 'Short period must be at least 1')
    .max(200, 'Short period too large'),

  longPeriod: z.number()
    .int('Period must be an integer')
    .min(2, 'Long period must be at least 2')
    .max(500, 'Long period too large'),

  minVolume: z.number()
    .min(0, 'Minimum volume cannot be negative')
    .optional(),
}).refine((data) => data.shortPeriod < data.longPeriod, {
  message: 'Short period must be less than long period',
  path: ['shortPeriod'],
});

// Mean reversion strategy specific parameters
export const meanReversionParametersSchema = z.object({
  period: z.number()
    .int('Period must be an integer')
    .min(1, 'Period must be at least 1')
    .max(500, 'Period too large'),

  stdDevThreshold: z.number()
    .min(0, 'Std dev threshold cannot be negative')
    .max(5, 'Std dev threshold too high'),

  rsiThreshold: z.number()
    .min(0, 'RSI threshold must be between 0 and 100')
    .max(100, 'RSI threshold must be between 0 and 100')
    .optional(),
});

// Arbitrage strategy specific parameters
export const arbitrageParametersSchema = z.object({
  priceThreshold: z.number()
    .min(0, 'Price threshold cannot be negative')
    .max(1, 'Price threshold should be between 0 and 1'),

  minProfit: z.number()
    .min(0, 'Minimum profit cannot be negative'),

  exchangePairs: z.array(z.object({
    exchange1: z.string().min(1, 'Exchange name required'),
    exchange2: z.string().min(1, 'Exchange name required'),
  }))
    .min(1, 'At least one exchange pair required'),
});

// Pairs trading strategy specific parameters
export const pairsTradingParametersSchema = z.object({
  symbol1: z.string()
    .min(1, 'First symbol required')
    .max(10, 'Symbol too long'),

  symbol2: z.string()
    .min(1, 'Second symbol required')
    .max(10, 'Symbol too long'),

  cointegrationThreshold: z.number()
    .min(0, 'Cointegration threshold must be between 0 and 1')
    .max(1, 'Cointegration threshold must be between 0 and 1'),

  spreadThreshold: z.number()
    .min(0, 'Spread threshold cannot be negative'),
});

// Schema for strategy IDs
export const strategyIdSchema = z.string()
  .min(1, 'Strategy ID cannot be empty')
  .max(100, 'Strategy ID too long')
  .regex(/^[a-zA-Z0-9_-]+$/, 'Invalid strategy ID format');

// Signal subscription callback signature
export const signalCallbackSchema = z.function()
  .args(z.object({
    symbol: z.string(),
    direction: z.enum(['BUY', 'SELL', 'HOLD']),
    confidence: z.number().min(0).max(1),
    entryPrice: z.number().positive('Entry price must be positive'),
    stopLoss: z.number().positive('Stop loss must be positive'),
    takeProfit: z.number().positive('Take profit must be positive'),
    reasoning: z.string(),
  }))
  .returns(z.any());

// Validation helper functions
export function validateStrategyConfig(config: unknown) {
  try {
    return strategyConfigSchema.parse(config);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const messages = error.errors
        .map((err) => `${err.path.join('.')}: ${err.message}`)
        .join('; ');
      throw new ValidationError(`Invalid strategy configuration: ${messages}`, error);
    }
    throw error;
  }
}

export function validateMomentumParameters(params: unknown) {
  try {
    return momentumParametersSchema.parse(params);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const messages = error.errors
        .map((err) => `${err.path.join('.')}: ${err.message}`)
        .join('; ');
      throw new ValidationError(`Invalid momentum parameters: ${messages}`, error);
    }
    throw error;
  }
}

export function validateMeanReversionParameters(params: unknown) {
  try {
    return meanReversionParametersSchema.parse(params);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const messages = error.errors
        .map((err) => `${err.path.join('.')}: ${err.message}`)
        .join('; ');
      throw new ValidationError(`Invalid mean reversion parameters: ${messages}`, error);
    }
    throw error;
  }
}

export function validateArbitrageParameters(params: unknown) {
  try {
    return arbitrageParametersSchema.parse(params);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const messages = error.errors
        .map((err) => `${err.path.join('.')}: ${err.message}`)
        .join('; ');
      throw new ValidationError(`Invalid arbitrage parameters: ${messages}`, error);
    }
    throw error;
  }
}

export function validatePairsTradingParameters(params: unknown) {
  try {
    return pairsTradingParametersSchema.parse(params);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const messages = error.errors
        .map((err) => `${err.path.join('.')}: ${err.message}`)
        .join('; ');
      throw new ValidationError(`Invalid pairs trading parameters: ${messages}`, error);
    }
    throw error;
  }
}

export function validateStrategyId(id: unknown) {
  try {
    return strategyIdSchema.parse(id);
  } catch (error) {
    if (error instanceof z.ZodError) {
      throw new ValidationError(`Invalid strategy ID: ${error.errors[0].message}`, error);
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
