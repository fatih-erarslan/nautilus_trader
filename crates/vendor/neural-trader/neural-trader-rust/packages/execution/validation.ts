/**
 * Input validation schemas for @neural-trader/execution
 * Uses Zod for runtime validation of trading configurations and orders
 */

import { z } from 'zod';

// Execution configuration schema
export const executionConfigSchema = z.object({
  brokerId: z.string()
    .min(1, 'Broker ID cannot be empty')
    .max(100, 'Broker ID too long'),

  apiKey: z.string()
    .min(1, 'API key cannot be empty')
    .max(500, 'API key too long'),

  apiSecret: z.string()
    .min(1, 'API secret cannot be empty')
    .max(500, 'API secret too long'),

  endpoint: z.string()
    .url('Must be a valid URL')
    .optional(),

  accountId: z.string()
    .min(1, 'Account ID cannot be empty')
    .max(100, 'Account ID too long')
    .optional(),

  maxSlippage: z.number()
    .min(0, 'Max slippage cannot be negative')
    .max(1, 'Max slippage should be between 0 and 1')
    .default(0.001),

  timeout: z.number()
    .int('Timeout must be an integer')
    .min(100, 'Timeout must be at least 100ms')
    .max(60000, 'Timeout too long')
    .default(5000),

  orderRetryCount: z.number()
    .int('Order retry count must be an integer')
    .min(0, 'Order retry count cannot be negative')
    .max(10, 'Order retry count too high')
    .default(3),
});

// Order schema for place orders
export const orderSchema = z.object({
  symbol: z.string()
    .min(1, 'Symbol cannot be empty')
    .max(10, 'Symbol too long')
    .regex(/^[A-Z0-9-]+$/, 'Invalid symbol format'),

  side: z.enum(['BUY', 'SELL'], {
    errorMap: () => ({ message: 'Side must be BUY or SELL' }),
  }),

  quantity: z.number()
    .positive('Quantity must be positive'),

  price: z.number()
    .positive('Price must be positive'),

  orderType: z.enum(['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT'], {
    errorMap: () => ({ message: 'Invalid order type' }),
  })
    .default('MARKET'),

  timeInForce: z.enum(['GTC', 'IOC', 'FOK', 'DAY'], {
    errorMap: () => ({ message: 'Invalid time in force' }),
  })
    .default('GTC'),

  // Advanced order parameters
  stopPrice: z.number()
    .positive('Stop price must be positive')
    .optional(),

  stopLoss: z.number()
    .positive('Stop loss must be positive')
    .optional(),

  takeProfit: z.number()
    .positive('Take profit must be positive')
    .optional(),

  trailingStopPercent: z.number()
    .min(0.01, 'Trailing stop percent must be at least 0.01')
    .max(50, 'Trailing stop percent too high')
    .optional(),

  // Advanced execution strategies
  executionStrategy: z.enum(['MARKET', 'TWAP', 'VWAP', 'ICEBERG', 'POV'], {
    errorMap: () => ({ message: 'Invalid execution strategy' }),
  })
    .default('MARKET'),

  // For TWAP/VWAP execution
  sliceDuration: z.number()
    .int('Slice duration must be an integer')
    .min(100, 'Slice duration at least 100ms')
    .max(3600000, 'Slice duration max 1 hour')
    .optional(),

  sliceCount: z.number()
    .int('Slice count must be an integer')
    .min(1, 'Slice count must be at least 1')
    .max(1000, 'Slice count too high')
    .optional(),

  // For Iceberg orders
  icebergQty: z.number()
    .positive('Iceberg quantity must be positive')
    .optional(),

  // Additional metadata
  clientOrderId: z.string()
    .min(1, 'Client order ID cannot be empty')
    .max(100, 'Client order ID too long')
    .optional(),

  comment: z.string()
    .max(500, 'Comment too long')
    .optional(),
})
  .refine((data) => {
    // If stop order, stopPrice is required
    if (data.orderType === 'STOP' && !data.stopPrice) {
      return false;
    }
    return true;
  }, {
    message: 'Stop price required for STOP orders',
    path: ['stopPrice'],
  })
  .refine((data) => {
    // If stop limit order, both stopPrice and price are required
    if (data.orderType === 'STOP_LIMIT' && !data.stopPrice) {
      return false;
    }
    return true;
  }, {
    message: 'Stop price required for STOP_LIMIT orders',
    path: ['stopPrice'],
  })
  .refine((data) => {
    // For TWAP/VWAP, need sliceDuration and sliceCount
    if (['TWAP', 'VWAP'].includes(data.executionStrategy)) {
      return data.sliceDuration !== undefined && data.sliceCount !== undefined;
    }
    return true;
  }, {
    message: 'Slice duration and slice count required for TWAP/VWAP',
    path: ['sliceDuration'],
  });

// Batch order validation
export const batchOrderSchema = z.array(orderSchema)
  .min(1, 'At least one order required')
  .max(100, 'Too many orders in batch');

// Order update schema (partial order for updates)
export const orderUpdateSchema = orderSchema.partial();

// Validation helper functions
export function validateExecutionConfig(config: unknown) {
  try {
    return executionConfigSchema.parse(config);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const messages = error.errors
        .map((err) => `${err.path.join('.')}: ${err.message}`)
        .join('; ');
      throw new ValidationError(`Invalid execution configuration: ${messages}`, error);
    }
    throw error;
  }
}

export function validateOrder(order: unknown) {
  try {
    return orderSchema.parse(order);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const messages = error.errors
        .map((err) => `${err.path.join('.')}: ${err.message}`)
        .join('; ');
      throw new ValidationError(`Invalid order: ${messages}`, error);
    }
    throw error;
  }
}

export function validateBatchOrders(orders: unknown) {
  try {
    return batchOrderSchema.parse(orders);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const messages = error.errors
        .map((err) => `${err.path.join('.')}: ${err.message}`)
        .join('; ');
      throw new ValidationError(`Invalid batch orders: ${messages}`, error);
    }
    throw error;
  }
}

export function validateOrderUpdate(update: unknown) {
  try {
    return orderUpdateSchema.parse(update);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const messages = error.errors
        .map((err) => `${err.path.join('.')}: ${err.message}`)
        .join('; ');
      throw new ValidationError(`Invalid order update: ${messages}`, error);
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
