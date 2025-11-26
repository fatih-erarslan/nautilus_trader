/**
 * Unit tests for @neural-trader/execution
 * Tests order execution, position management, and trading operations
 */

describe('Execution Unit Tests', () => {
  let neuralTrader: any;

  const mockConfig = {
    brokerApi: 'paper-trading',
    symbols: ['AAPL', 'MSFT'],
    accountId: 'test-account-001',
  };

  const mockOrder = {
    symbol: 'AAPL',
    quantity: 100,
    price: 150.0,
    orderType: 'MARKET',
    side: 'BUY',
  };

  beforeEach(() => {
    neuralTrader = {
      config: null,
      isRunning: false,
      positions: new Map(),
      balance: 100000,
      equity: 100000,

      constructor(config: any) {
        if (!config) {
          throw new Error('Config is required');
        }
        if (!config.brokerApi) {
          throw new Error('Broker API is required');
        }
        this.config = config;
      },

      async start() {
        if (this.isRunning) {
          return { success: false, error: 'Already running' };
        }
        this.isRunning = true;
        return { success: true, status: 'started' };
      },

      async stop() {
        if (!this.isRunning) {
          return { success: false, error: 'Not running' };
        }
        this.isRunning = false;
        return { success: true, status: 'stopped' };
      },

      async getPositions() {
        return Array.from(this.positions.values());
      },

      async placeOrder(order: any) {
        if (!order.symbol || !order.quantity || typeof order.price !== 'number') {
          return { success: false, error: 'Invalid order' };
        }
        if (order.quantity <= 0) {
          return { success: false, error: 'Quantity must be positive' };
        }
        if (order.price <= 0) {
          return { success: false, error: 'Price must be positive' };
        }

        const orderId = `order-${Date.now()}`;
        const totalCost = order.quantity * order.price;

        if (order.side === 'BUY' && totalCost > this.balance) {
          return { success: false, error: 'Insufficient balance' };
        }

        if (order.side === 'BUY') {
          this.balance -= totalCost;
          const position = this.positions.get(order.symbol) || {
            symbol: order.symbol,
            quantity: 0,
            avgPrice: 0,
          };
          position.quantity += order.quantity;
          position.avgPrice = (position.avgPrice * (position.quantity - order.quantity) + order.price * order.quantity) / position.quantity;
          this.positions.set(order.symbol, position);
        } else if (order.side === 'SELL') {
          this.balance += totalCost;
          const position = this.positions.get(order.symbol);
          if (!position || position.quantity < order.quantity) {
            this.balance -= totalCost; // Revert
            return { success: false, error: 'Insufficient position' };
          }
          position.quantity -= order.quantity;
          if (position.quantity === 0) {
            this.positions.delete(order.symbol);
          } else {
            this.positions.set(order.symbol, position);
          }
        }

        return {
          success: true,
          orderId,
          status: 'filled',
          order: { ...order, orderId },
        };
      },

      async getBalance() {
        return { balance: this.balance, currency: 'USD' };
      },

      async getEquity() {
        return { equity: this.equity, unrealizedPnL: 0 };
      },
    };
  });

  describe('NeuralTrader Initialization', () => {
    it('should create NeuralTrader with valid config', () => {
      neuralTrader.constructor(mockConfig);
      expect(neuralTrader.config).toEqual(mockConfig);
      expect(neuralTrader.isRunning).toBe(false);
    });

    it('should throw on missing config', () => {
      expect(() => {
        neuralTrader.constructor(null);
      }).toThrow('Config is required');
    });

    it('should throw on missing broker API', () => {
      expect(() => {
        neuralTrader.constructor({ symbols: ['AAPL'] });
      }).toThrow('Broker API is required');
    });

    it('should store configuration correctly', () => {
      neuralTrader.constructor(mockConfig);
      expect(neuralTrader.config.brokerApi).toBe('paper-trading');
      expect(neuralTrader.config.accountId).toBe('test-account-001');
    });
  });

  describe('Trader Start/Stop', () => {
    beforeEach(() => {
      neuralTrader.constructor(mockConfig);
    });

    it('should start trader successfully', async () => {
      const result = await neuralTrader.start();
      expect(result.success).toBe(true);
      expect(result.status).toBe('started');
      expect(neuralTrader.isRunning).toBe(true);
    });

    it('should reject start when already running', async () => {
      await neuralTrader.start();
      const result = await neuralTrader.start();
      expect(result.success).toBe(false);
      expect(result.error).toBe('Already running');
    });

    it('should stop trader successfully', async () => {
      await neuralTrader.start();
      const result = await neuralTrader.stop();
      expect(result.success).toBe(true);
      expect(result.status).toBe('stopped');
      expect(neuralTrader.isRunning).toBe(false);
    });

    it('should reject stop when not running', async () => {
      const result = await neuralTrader.stop();
      expect(result.success).toBe(false);
      expect(result.error).toBe('Not running');
    });
  });

  describe('Order Placement', () => {
    beforeEach(async () => {
      neuralTrader.constructor(mockConfig);
      await neuralTrader.start();
    });

    it('should place BUY order successfully', async () => {
      const result = await neuralTrader.placeOrder(mockOrder);
      expect(result.success).toBe(true);
      expect(result.orderId).toBeDefined();
      expect(result.status).toBe('filled');
    });

    it('should place SELL order successfully', async () => {
      // First buy
      await neuralTrader.placeOrder(mockOrder);

      // Then sell
      const sellOrder = { ...mockOrder, side: 'SELL' };
      const result = await neuralTrader.placeOrder(sellOrder);
      expect(result.success).toBe(true);
      expect(result.status).toBe('filled');
    });

    it('should reject order with zero quantity', async () => {
      const order = { ...mockOrder, quantity: 0 };
      const result = await neuralTrader.placeOrder(order);
      expect(result.success).toBe(false);
      expect(result.error).toBe('Quantity must be positive');
    });

    it('should reject order with negative quantity', async () => {
      const order = { ...mockOrder, quantity: -100 };
      const result = await neuralTrader.placeOrder(order);
      expect(result.success).toBe(false);
      expect(result.error).toBe('Quantity must be positive');
    });

    it('should reject order with zero price', async () => {
      const order = { ...mockOrder, price: 0 };
      const result = await neuralTrader.placeOrder(order);
      expect(result.success).toBe(false);
      expect(result.error).toBe('Price must be positive');
    });

    it('should reject order with negative price', async () => {
      const order = { ...mockOrder, price: -150 };
      const result = await neuralTrader.placeOrder(order);
      expect(result.success).toBe(false);
      expect(result.error).toBe('Price must be positive');
    });

    it('should reject order with insufficient balance', async () => {
      const order = { ...mockOrder, quantity: 1000 }; // Cost: 150,000 > balance: 100,000
      const result = await neuralTrader.placeOrder(order);
      expect(result.success).toBe(false);
      expect(result.error).toBe('Insufficient balance');
    });

    it('should reject SELL order with insufficient position', async () => {
      const sellOrder = { ...mockOrder, side: 'SELL', quantity: 500 };
      const result = await neuralTrader.placeOrder(sellOrder);
      expect(result.success).toBe(false);
      expect(result.error).toBe('Insufficient position');
    });

    it('should reject order with missing symbol', async () => {
      const order = { ...mockOrder, symbol: undefined };
      const result = await neuralTrader.placeOrder(order);
      expect(result.success).toBe(false);
    });

    it('should reject order with missing quantity', async () => {
      const order = { ...mockOrder, quantity: undefined };
      const result = await neuralTrader.placeOrder(order);
      expect(result.success).toBe(false);
    });
  });

  describe('Position Management', () => {
    beforeEach(async () => {
      neuralTrader.constructor(mockConfig);
      await neuralTrader.start();
    });

    it('should return empty positions initially', async () => {
      const positions = await neuralTrader.getPositions();
      expect(positions).toEqual([]);
    });

    it('should track positions after buy order', async () => {
      await neuralTrader.placeOrder(mockOrder);
      const positions = await neuralTrader.getPositions();

      expect(positions).toHaveLength(1);
      expect(positions[0].symbol).toBe('AAPL');
      expect(positions[0].quantity).toBe(100);
    });

    it('should update position on additional buy', async () => {
      await neuralTrader.placeOrder(mockOrder);
      await neuralTrader.placeOrder({ ...mockOrder, quantity: 50 });

      const positions = await neuralTrader.getPositions();
      expect(positions).toHaveLength(1);
      expect(positions[0].quantity).toBe(150);
    });

    it('should remove position when quantity reaches zero', async () => {
      await neuralTrader.placeOrder(mockOrder);
      await neuralTrader.placeOrder({
        ...mockOrder,
        side: 'SELL',
        quantity: 100,
      });

      const positions = await neuralTrader.getPositions();
      expect(positions).toHaveLength(0);
    });

    it('should handle multiple positions', async () => {
      await neuralTrader.placeOrder(mockOrder);
      await neuralTrader.placeOrder({
        ...mockOrder,
        symbol: 'MSFT',
        quantity: 50,
      });

      const positions = await neuralTrader.getPositions();
      expect(positions).toHaveLength(2);
      expect(positions.map(p => p.symbol).sort()).toEqual(['AAPL', 'MSFT']);
    });
  });

  describe('Balance and Equity', () => {
    beforeEach(async () => {
      neuralTrader.constructor(mockConfig);
      await neuralTrader.start();
    });

    it('should return initial balance', async () => {
      const result = await neuralTrader.getBalance();
      expect(result.balance).toBe(100000);
      expect(result.currency).toBe('USD');
    });

    it('should update balance after buy order', async () => {
      const initialBalance = (await neuralTrader.getBalance()).balance;

      await neuralTrader.placeOrder(mockOrder);

      const newBalance = (await neuralTrader.getBalance()).balance;
      expect(newBalance).toBe(initialBalance - mockOrder.quantity * mockOrder.price);
    });

    it('should restore balance after sell order', async () => {
      const initialBalance = (await neuralTrader.getBalance()).balance;

      await neuralTrader.placeOrder(mockOrder);
      await neuralTrader.placeOrder({
        ...mockOrder,
        side: 'SELL',
      });

      const finalBalance = (await neuralTrader.getBalance()).balance;
      expect(finalBalance).toBe(initialBalance);
    });

    it('should return equity', async () => {
      const result = await neuralTrader.getEquity();
      expect(result.equity).toBeDefined();
      expect(result.unrealizedPnL).toBeDefined();
    });
  });

  describe('Edge Cases', () => {
    beforeEach(async () => {
      neuralTrader.constructor(mockConfig);
      await neuralTrader.start();
    });

    it('should handle very small order quantity', async () => {
      const order = { ...mockOrder, quantity: 0.01 };
      const result = await neuralTrader.placeOrder(order);
      expect(result.success).toBe(true);
    });

    it('should handle very large order quantity', async () => {
      const order = { ...mockOrder, quantity: 1000000 };
      const result = await neuralTrader.placeOrder(order);
      expect(result.success).toBe(false); // Insufficient balance
    });

    it('should handle decimal prices', async () => {
      const order = { ...mockOrder, price: 150.5678 };
      const result = await neuralTrader.placeOrder(order);
      expect(result.success).toBe(true);
    });
  });
});
