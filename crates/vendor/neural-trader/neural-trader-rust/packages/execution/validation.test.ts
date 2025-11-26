/**
 * Tests for execution validation schemas
 */

import {
  executionConfigSchema,
  orderSchema,
  batchOrderSchema,
  validateExecutionConfig,
  validateOrder,
  validateBatchOrders,
  ValidationError,
} from './validation';

describe('Execution Validation Schemas', () => {
  describe('executionConfigSchema', () => {
    it('should validate valid execution config', () => {
      const config = {
        brokerId: 'BROKER-1',
        apiKey: 'key123',
        apiSecret: 'secret123',
        accountId: 'ACC-001',
        maxSlippage: 0.001,
      };
      const result = executionConfigSchema.parse(config);
      expect(result.brokerId).toBe('BROKER-1');
      expect(result.maxSlippage).toBe(0.001);
    });

    it('should set default values', () => {
      const config = {
        brokerId: 'BROKER-1',
        apiKey: 'key123',
        apiSecret: 'secret123',
      };
      const result = executionConfigSchema.parse(config);
      expect(result.maxSlippage).toBe(0.001);
      expect(result.timeout).toBe(5000);
      expect(result.orderRetryCount).toBe(3);
    });

    it('should reject invalid endpoint URL', () => {
      expect(() =>
        executionConfigSchema.parse({
          brokerId: 'BROKER-1',
          apiKey: 'key123',
          apiSecret: 'secret123',
          endpoint: 'not-a-url',
        })
      ).toThrow();
    });

    it('should reject invalid maxSlippage', () => {
      expect(() =>
        executionConfigSchema.parse({
          brokerId: 'BROKER-1',
          apiKey: 'key123',
          apiSecret: 'secret123',
          maxSlippage: 1.5,
        })
      ).toThrow();
    });

    it('should reject timeout < 100ms', () => {
      expect(() =>
        executionConfigSchema.parse({
          brokerId: 'BROKER-1',
          apiKey: 'key123',
          apiSecret: 'secret123',
          timeout: 50,
        })
      ).toThrow();
    });
  });

  describe('orderSchema', () => {
    it('should validate valid market order', () => {
      const order = {
        symbol: 'AAPL',
        side: 'BUY',
        quantity: 100,
        price: 150.25,
        orderType: 'MARKET',
      };
      expect(orderSchema.parse(order)).toEqual(order);
    });

    it('should validate valid limit order', () => {
      const order = {
        symbol: 'MSFT',
        side: 'SELL',
        quantity: 50,
        price: 300,
        orderType: 'LIMIT',
      };
      expect(orderSchema.parse(order)).toEqual(order);
    });

    it('should require stopPrice for STOP orders', () => {
      expect(() =>
        orderSchema.parse({
          symbol: 'AAPL',
          side: 'SELL',
          quantity: 100,
          price: 150,
          orderType: 'STOP',
        })
      ).toThrow();

      expect(
        orderSchema.parse({
          symbol: 'AAPL',
          side: 'SELL',
          quantity: 100,
          price: 150,
          orderType: 'STOP',
          stopPrice: 145,
        })
      ).toBeDefined();
    });

    it('should require stopPrice for STOP_LIMIT orders', () => {
      expect(() =>
        orderSchema.parse({
          symbol: 'AAPL',
          side: 'SELL',
          quantity: 100,
          price: 150,
          orderType: 'STOP_LIMIT',
        })
      ).toThrow();
    });

    it('should require sliceDuration and sliceCount for TWAP', () => {
      expect(() =>
        orderSchema.parse({
          symbol: 'AAPL',
          side: 'BUY',
          quantity: 1000,
          price: 150,
          executionStrategy: 'TWAP',
        })
      ).toThrow();

      expect(
        orderSchema.parse({
          symbol: 'AAPL',
          side: 'BUY',
          quantity: 1000,
          price: 150,
          executionStrategy: 'TWAP',
          sliceDuration: 3600000,
          sliceCount: 10,
        })
      ).toBeDefined();
    });

    it('should reject invalid symbol format', () => {
      expect(() =>
        orderSchema.parse({
          symbol: 'aapl',
          side: 'BUY',
          quantity: 100,
          price: 150,
        })
      ).toThrow();
    });

    it('should reject invalid side', () => {
      expect(() =>
        orderSchema.parse({
          symbol: 'AAPL',
          side: 'INVALID',
          quantity: 100,
          price: 150,
        })
      ).toThrow();
    });

    it('should reject negative or zero quantity', () => {
      expect(() =>
        orderSchema.parse({
          symbol: 'AAPL',
          side: 'BUY',
          quantity: 0,
          price: 150,
        })
      ).toThrow();

      expect(() =>
        orderSchema.parse({
          symbol: 'AAPL',
          side: 'BUY',
          quantity: -100,
          price: 150,
        })
      ).toThrow();
    });

    it('should reject negative or zero price', () => {
      expect(() =>
        orderSchema.parse({
          symbol: 'AAPL',
          side: 'BUY',
          quantity: 100,
          price: 0,
        })
      ).toThrow();

      expect(() =>
        orderSchema.parse({
          symbol: 'AAPL',
          side: 'BUY',
          quantity: 100,
          price: -150,
        })
      ).toThrow();
    });

    it('should accept optional fields', () => {
      const order = {
        symbol: 'AAPL',
        side: 'BUY',
        quantity: 100,
        price: 150,
        clientOrderId: 'ORDER-001',
        comment: 'Test order',
      };
      expect(orderSchema.parse(order)).toEqual(order);
    });
  });

  describe('batchOrderSchema', () => {
    it('should validate batch of valid orders', () => {
      const orders = [
        {
          symbol: 'AAPL',
          side: 'BUY',
          quantity: 100,
          price: 150,
        },
        {
          symbol: 'MSFT',
          side: 'SELL',
          quantity: 50,
          price: 300,
        },
      ];
      expect(batchOrderSchema.parse(orders)).toEqual(orders);
    });

    it('should reject empty batch', () => {
      expect(() => batchOrderSchema.parse([])).toThrow();
    });

    it('should reject batch with invalid order', () => {
      expect(() =>
        batchOrderSchema.parse([
          {
            symbol: 'AAPL',
            side: 'BUY',
            quantity: 0,
            price: 150,
          },
        ])
      ).toThrow();
    });

    it('should reject batch > 100 orders', () => {
      const orders = Array(101).fill({
        symbol: 'AAPL',
        side: 'BUY',
        quantity: 100,
        price: 150,
      });
      expect(() => batchOrderSchema.parse(orders)).toThrow();
    });
  });

  describe('Validation helper functions', () => {
    describe('validateExecutionConfig', () => {
      it('should throw ValidationError on invalid config', () => {
        expect(() =>
          validateExecutionConfig({
            brokerId: '',
            apiKey: 'key',
            apiSecret: 'secret',
          })
        ).toThrow(ValidationError);
      });

      it('should return config on valid input', () => {
        const config = {
          brokerId: 'BROKER-1',
          apiKey: 'key123',
          apiSecret: 'secret123',
        };
        const result = validateExecutionConfig(config);
        expect(result.brokerId).toBe('BROKER-1');
      });
    });

    describe('validateOrder', () => {
      it('should throw ValidationError on invalid order', () => {
        expect(() =>
          validateOrder({
            symbol: 'aapl',
            side: 'BUY',
            quantity: 100,
            price: 150,
          })
        ).toThrow(ValidationError);
      });

      it('should return order on valid input', () => {
        const order = {
          symbol: 'AAPL',
          side: 'BUY',
          quantity: 100,
          price: 150,
        };
        const result = validateOrder(order);
        expect(result.symbol).toBe('AAPL');
      });
    });

    describe('validateBatchOrders', () => {
      it('should throw ValidationError on invalid batch', () => {
        expect(() => validateBatchOrders([])).toThrow(ValidationError);
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
