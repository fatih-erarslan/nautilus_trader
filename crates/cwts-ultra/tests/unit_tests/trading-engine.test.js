const { TradingEngine } = require('../../quantum_trading/core/trading_engine');
const { OrderBook } = require('../../quantum_trading/core/order_book');
const { RiskManager } = require('../../quantum_trading/core/risk_manager');

describe('TradingEngine - Unit Tests (100% Coverage)', () => {
  let tradingEngine;
  let mockOrderBook;
  let mockRiskManager;

  beforeEach(() => {
    mockOrderBook = {
      addOrder: jest.fn(),
      removeOrder: jest.fn(),
      matchOrders: jest.fn(),
      getBestBid: jest.fn(),
      getBestOffer: jest.fn(),
      getOrderById: jest.fn(),
      clear: jest.fn()
    };

    mockRiskManager = {
      validateOrder: jest.fn(),
      checkPositionLimits: jest.fn(),
      calculateRisk: jest.fn(),
      enforceKillSwitch: jest.fn()
    };

    tradingEngine = new TradingEngine({
      orderBook: mockOrderBook,
      riskManager: mockRiskManager
    });
  });

  describe('Order Processing', () => {
    test('should process valid buy order successfully', async () => {
      const order = {
        id: 'ORD001',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00,
        timestamp: Date.now(),
        userId: 'USER001'
      };

      mockRiskManager.validateOrder.mockResolvedValue(true);
      mockOrderBook.addOrder.mockResolvedValue({ orderId: 'ORD001', status: 'accepted' });

      const result = await tradingEngine.processOrder(order);

      expect(result.success).toBe(true);
      expect(result.orderId).toBe('ORD001');
      expect(mockRiskManager.validateOrder).toHaveBeenCalledWith(order);
      expect(mockOrderBook.addOrder).toHaveBeenCalledWith(order);
    });

    test('should reject order with invalid price', async () => {
      const order = {
        id: 'ORD002',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: -50.00, // Invalid negative price
        timestamp: Date.now(),
        userId: 'USER001'
      };

      const result = await tradingEngine.processOrder(order);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Invalid price');
      expect(mockRiskManager.validateOrder).not.toHaveBeenCalled();
    });

    test('should reject order with zero quantity', async () => {
      const order = {
        id: 'ORD003',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 0, // Invalid zero quantity
        price: 150.00,
        timestamp: Date.now(),
        userId: 'USER001'
      };

      const result = await tradingEngine.processOrder(order);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Invalid quantity');
    });

    test('should handle risk manager rejection', async () => {
      const order = {
        id: 'ORD004',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00,
        timestamp: Date.now(),
        userId: 'USER001'
      };

      mockRiskManager.validateOrder.mockResolvedValue(false);

      const result = await tradingEngine.processOrder(order);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Risk validation failed');
      expect(mockOrderBook.addOrder).not.toHaveBeenCalled();
    });

    test('should handle extremely large quantities', async () => {
      const order = {
        id: 'ORD005',
        symbol: 'AAPL',
        side: 'buy',
        quantity: Number.MAX_SAFE_INTEGER,
        price: 150.00,
        timestamp: Date.now(),
        userId: 'USER001'
      };

      mockRiskManager.validateOrder.mockResolvedValue(false);

      const result = await tradingEngine.processOrder(order);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Quantity exceeds maximum allowed');
    });

    test('should handle precision issues with decimal prices', async () => {
      const order = {
        id: 'ORD006',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.123456789, // High precision price
        timestamp: Date.now(),
        userId: 'USER001'
      };

      mockRiskManager.validateOrder.mockResolvedValue(true);
      mockOrderBook.addOrder.mockResolvedValue({ orderId: 'ORD006', status: 'accepted' });

      const result = await tradingEngine.processOrder(order);

      expect(result.success).toBe(true);
      // Verify price is rounded to 4 decimal places
      const processedOrder = mockOrderBook.addOrder.mock.calls[0][0];
      expect(processedOrder.price).toBe(150.1235);
    });
  });

  describe('Order Cancellation', () => {
    test('should cancel existing order successfully', async () => {
      const orderId = 'ORD001';
      mockOrderBook.getOrderById.mockResolvedValue({
        id: orderId,
        status: 'active'
      });
      mockOrderBook.removeOrder.mockResolvedValue(true);

      const result = await tradingEngine.cancelOrder(orderId);

      expect(result.success).toBe(true);
      expect(mockOrderBook.removeOrder).toHaveBeenCalledWith(orderId);
    });

    test('should handle cancellation of non-existent order', async () => {
      const orderId = 'NONEXISTENT';
      mockOrderBook.getOrderById.mockResolvedValue(null);

      const result = await tradingEngine.cancelOrder(orderId);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Order not found');
    });

    test('should handle cancellation of already filled order', async () => {
      const orderId = 'ORD001';
      mockOrderBook.getOrderById.mockResolvedValue({
        id: orderId,
        status: 'filled'
      });

      const result = await tradingEngine.cancelOrder(orderId);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Cannot cancel filled order');
    });
  });

  describe('Performance Metrics', () => {
    test('should process orders within latency requirements', async () => {
      const order = {
        id: 'ORD007',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00,
        timestamp: Date.now(),
        userId: 'USER001'
      };

      mockRiskManager.validateOrder.mockResolvedValue(true);
      mockOrderBook.addOrder.mockResolvedValue({ orderId: 'ORD007', status: 'accepted' });

      const { result, duration } = await measurePerformance(
        () => tradingEngine.processOrder(order)
      );

      expect(result.success).toBe(true);
      expect(duration).toBeLessThan(10); // Must process within 10ms
    });

    test('should not leak memory during order processing', async () => {
      const orders = Array.from({ length: 1000 }, (_, i) => ({
        id: `ORD${i}`,
        symbol: 'AAPL',
        side: i % 2 === 0 ? 'buy' : 'sell',
        quantity: 100,
        price: 150.00 + (Math.random() - 0.5),
        timestamp: Date.now(),
        userId: 'USER001'
      }));

      mockRiskManager.validateOrder.mockResolvedValue(true);
      mockOrderBook.addOrder.mockResolvedValue({ status: 'accepted' });

      const { memoryDelta } = await measureMemory(async () => {
        for (const order of orders) {
          await tradingEngine.processOrder(order);
        }
      });

      expect(memoryDelta.heapUsed).toBeLessThan(MEMORY_LEAK_THRESHOLD_MB * 1024 * 1024);
    });
  });

  describe('Error Handling', () => {
    test('should handle OrderBook exceptions gracefully', async () => {
      const order = {
        id: 'ORD008',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00,
        timestamp: Date.now(),
        userId: 'USER001'
      };

      mockRiskManager.validateOrder.mockResolvedValue(true);
      mockOrderBook.addOrder.mockRejectedValue(new Error('OrderBook failure'));

      const result = await tradingEngine.processOrder(order);

      expect(result.success).toBe(false);
      expect(result.error).toContain('OrderBook failure');
    });

    test('should handle RiskManager timeout', async () => {
      const order = {
        id: 'ORD009',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00,
        timestamp: Date.now(),
        userId: 'USER001'
      };

      mockRiskManager.validateOrder.mockImplementation(
        () => new Promise(resolve => setTimeout(resolve, 5000))
      );

      const result = await tradingEngine.processOrder(order);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Risk validation timeout');
    });
  });

  describe('Edge Cases', () => {
    test('should handle null order input', async () => {
      const result = await tradingEngine.processOrder(null);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Order cannot be null');
    });

    test('should handle malformed order object', async () => {
      const malformedOrder = {
        // Missing required fields
        symbol: 'AAPL'
      };

      const result = await tradingEngine.processOrder(malformedOrder);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Missing required fields');
    });

    test('should handle concurrent order processing', async () => {
      const orders = Array.from({ length: 10 }, (_, i) => ({
        id: `ORD${i}`,
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00,
        timestamp: Date.now(),
        userId: 'USER001'
      }));

      mockRiskManager.validateOrder.mockResolvedValue(true);
      mockOrderBook.addOrder.mockResolvedValue({ status: 'accepted' });

      const promises = orders.map(order => tradingEngine.processOrder(order));
      const results = await Promise.all(promises);

      expect(results).toHaveLength(10);
      expect(results.every(r => r.success)).toBe(true);
    });
  });
});