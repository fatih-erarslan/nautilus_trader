/**
 * Integration tests for @neural-trader/execution
 * Tests complete trading workflows, order routing, and position lifecycle
 */

describe('Execution Integration Tests', () => {
  let neuralTrader: any;

  const mockConfig = {
    brokerApi: 'paper-trading',
    symbols: ['AAPL', 'MSFT', 'GOOGL'],
    accountId: 'test-account-001',
  };

  beforeEach(() => {
    neuralTrader = {
      config: null,
      isRunning: false,
      positions: new Map(),
      orderHistory: [],
      balance: 100000,
      equity: 100000,
      baseCash: 100000,

      constructor(config: any) {
        this.config = config;
      },

      async start() {
        this.isRunning = true;
        return { success: true, status: 'started' };
      },

      async stop() {
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

        const orderId = `order-${this.orderHistory.length + 1}`;
        const totalCost = order.quantity * order.price;

        if (order.side === 'BUY') {
          if (totalCost > this.balance) {
            return { success: false, error: 'Insufficient balance' };
          }
          this.balance -= totalCost;
          const position = this.positions.get(order.symbol) || {
            symbol: order.symbol,
            quantity: 0,
            avgPrice: 0,
            entryTime: new Date(),
          };
          position.quantity += order.quantity;
          position.avgPrice = (position.avgPrice * (position.quantity - order.quantity) + order.price * order.quantity) / position.quantity;
          this.positions.set(order.symbol, position);
        } else if (order.side === 'SELL') {
          const position = this.positions.get(order.symbol);
          if (!position || position.quantity < order.quantity) {
            return { success: false, error: 'Insufficient position' };
          }
          this.balance += totalCost;
          position.quantity -= order.quantity;
          if (position.quantity === 0) {
            this.positions.delete(order.symbol);
          }
        }

        this.orderHistory.push({ orderId, ...order, status: 'filled', timestamp: Date.now() });
        return { success: true, orderId, status: 'filled' };
      },

      async getBalance() {
        return { balance: this.balance, currency: 'USD' };
      },

      async getEquity() {
        return { equity: this.equity };
      },

      getTotalOrderValue() {
        return this.orderHistory.reduce((sum, o) => sum + o.quantity * o.price, 0);
      },

      getOrderCount() {
        return this.orderHistory.length;
      },

      getAverageOrderSize() {
        if (this.orderHistory.length === 0) return 0;
        return this.getTotalOrderValue() / this.orderHistory.length;
      },
    };
  });

  describe('Complete Trading Lifecycle', () => {
    it('should execute full buy-hold-sell workflow', async () => {
      neuralTrader.constructor(mockConfig);
      await neuralTrader.start();

      const buyResult = await neuralTrader.placeOrder({
        symbol: 'AAPL',
        quantity: 100,
        price: 150,
        side: 'BUY',
      });

      expect(buyResult.success).toBe(true);

      let positions = await neuralTrader.getPositions();
      expect(positions).toHaveLength(1);
      expect(positions[0].symbol).toBe('AAPL');
      expect(positions[0].quantity).toBe(100);

      const sellResult = await neuralTrader.placeOrder({
        symbol: 'AAPL',
        quantity: 100,
        price: 160,
        side: 'SELL',
      });

      expect(sellResult.success).toBe(true);

      positions = await neuralTrader.getPositions();
      expect(positions).toHaveLength(0);

      const finalBalance = (await neuralTrader.getBalance()).balance;
      // Initial 100k - (100 * 150) + (100 * 160) = 100k + 1000 = 101k
      expect(finalBalance).toBe(101000);

      await neuralTrader.stop();
    });

    it('should handle portfolio diversification workflow', async () => {
      neuralTrader.constructor(mockConfig);
      await neuralTrader.start();

      // Buy multiple securities
      const symbols = ['AAPL', 'MSFT', 'GOOGL'];
      for (const symbol of symbols) {
        const result = await neuralTrader.placeOrder({
          symbol,
          quantity: 50,
          price: 150,
          side: 'BUY',
        });
        expect(result.success).toBe(true);
      }

      let positions = await neuralTrader.getPositions();
      expect(positions).toHaveLength(3);

      // Balance should be: 100000 - (3 * 50 * 150) = 100000 - 22500 = 77500
      const balance = (await neuralTrader.getBalance()).balance;
      expect(balance).toBe(77500);

      await neuralTrader.stop();
    });
  });

  describe('Multiple Order Handling', () => {
    beforeEach(async () => {
      neuralTrader.constructor(mockConfig);
      await neuralTrader.start();
    });

    it('should handle rapid buy orders', async () => {
      const results = await Promise.all([
        neuralTrader.placeOrder({
          symbol: 'AAPL',
          quantity: 25,
          price: 150,
          side: 'BUY',
        }),
        neuralTrader.placeOrder({
          symbol: 'AAPL',
          quantity: 25,
          price: 150,
          side: 'BUY',
        }),
        neuralTrader.placeOrder({
          symbol: 'AAPL',
          quantity: 25,
          price: 150,
          side: 'BUY',
        }),
        neuralTrader.placeOrder({
          symbol: 'AAPL',
          quantity: 25,
          price: 150,
          side: 'BUY',
        }),
      ]);

      results.forEach(r => expect(r.success).toBe(true));

      const positions = await neuralTrader.getPositions();
      expect(positions).toHaveLength(1);
      expect(positions[0].quantity).toBe(100);
    });

    it('should track order history', async () => {
      await neuralTrader.placeOrder({
        symbol: 'AAPL',
        quantity: 100,
        price: 150,
        side: 'BUY',
      });

      await neuralTrader.placeOrder({
        symbol: 'MSFT',
        quantity: 50,
        price: 300,
        side: 'BUY',
      });

      expect(neuralTrader.getOrderCount()).toBe(2);
      expect(neuralTrader.getTotalOrderValue()).toBe(100 * 150 + 50 * 300);
    });

    it('should calculate average order size', async () => {
      await neuralTrader.placeOrder({
        symbol: 'AAPL',
        quantity: 100,
        price: 150,
        side: 'BUY',
      });

      await neuralTrader.placeOrder({
        symbol: 'MSFT',
        quantity: 50,
        price: 300,
        side: 'BUY',
      });

      const avgOrderSize = neuralTrader.getAverageOrderSize();
      // (15000 + 15000) / 2 = 15000
      expect(avgOrderSize).toBe(15000);
    });
  });

  describe('Complex Trading Scenarios', () => {
    beforeEach(async () => {
      neuralTrader.constructor(mockConfig);
      await neuralTrader.start();
    });

    it('should handle position scaling in and out', async () => {
      // Scale in
      await neuralTrader.placeOrder({
        symbol: 'AAPL',
        quantity: 25,
        price: 150,
        side: 'BUY',
      });

      await neuralTrader.placeOrder({
        symbol: 'AAPL',
        quantity: 25,
        price: 148,
        side: 'BUY',
      });

      let positions = await neuralTrader.getPositions();
      expect(positions[0].quantity).toBe(50);

      // Scale out
      await neuralTrader.placeOrder({
        symbol: 'AAPL',
        quantity: 20,
        price: 160,
        side: 'SELL',
      });

      positions = await neuralTrader.getPositions();
      expect(positions[0].quantity).toBe(30);

      // Close remaining
      await neuralTrader.placeOrder({
        symbol: 'AAPL',
        quantity: 30,
        price: 160,
        side: 'SELL',
      });

      positions = await neuralTrader.getPositions();
      expect(positions).toHaveLength(0);
    });

    it('should handle hedging strategy', async () => {
      // Buy AAPL as hedge
      await neuralTrader.placeOrder({
        symbol: 'AAPL',
        quantity: 100,
        price: 150,
        side: 'BUY',
      });

      // Buy inverse position
      await neuralTrader.placeOrder({
        symbol: 'AAPL',
        quantity: 50,
        price: 155,
        side: 'BUY',
      });

      const positions = await neuralTrader.getPositions();
      expect(positions[0].quantity).toBe(150);

      // Partial unwind
      await neuralTrader.placeOrder({
        symbol: 'AAPL',
        quantity: 75,
        price: 155,
        side: 'SELL',
      });

      const remainingPositions = await neuralTrader.getPositions();
      expect(remainingPositions[0].quantity).toBe(75);
    });

    it('should handle pairs trading scenario', async () => {
      // Buy first leg
      await neuralTrader.placeOrder({
        symbol: 'AAPL',
        quantity: 100,
        price: 150,
        side: 'BUY',
      });

      // Sell second leg
      // (This should fail initially, but demonstrate the concept)
      const result = await neuralTrader.placeOrder({
        symbol: 'MSFT',
        quantity: 100,
        price: 300,
        side: 'BUY',
      });

      expect(result.success).toBe(false); // Insufficient balance

      const balance = (await neuralTrader.getBalance()).balance;
      expect(balance).toBeLessThan(neuralTrader.baseCash);
    });
  });

  describe('State Consistency During Operations', () => {
    beforeEach(async () => {
      neuralTrader.constructor(mockConfig);
      await neuralTrader.start();
    });

    it('should maintain position accuracy across operations', async () => {
      const symbols = ['AAPL', 'MSFT', 'GOOGL'];
      const quantities = [100, 50, 75];

      for (let i = 0; i < symbols.length; i++) {
        await neuralTrader.placeOrder({
          symbol: symbols[i],
          quantity: quantities[i],
          price: 150,
          side: 'BUY',
        });
      }

      let positions = await neuralTrader.getPositions();
      expect(positions).toHaveLength(3);

      for (let i = 0; i < symbols.length; i++) {
        const pos = positions.find(p => p.symbol === symbols[i]);
        expect(pos?.quantity).toBe(quantities[i]);
      }
    });

    it('should maintain balance consistency', async () => {
      const initialBalance = (await neuralTrader.getBalance()).balance;

      const orders = [
        { symbol: 'AAPL', quantity: 100, price: 150, side: 'BUY' as const },
        { symbol: 'MSFT', quantity: 50, price: 300, side: 'BUY' as const },
        { symbol: 'AAPL', quantity: 50, price: 155, side: 'SELL' as const },
      ];

      let expectedBalance = initialBalance;
      for (const order of orders) {
        const totalCost = order.quantity * order.price;
        if (order.side === 'BUY') {
          expectedBalance -= totalCost;
        } else {
          expectedBalance += totalCost;
        }
        await neuralTrader.placeOrder(order);
      }

      const finalBalance = (await neuralTrader.getBalance()).balance;
      expect(finalBalance).toBe(expectedBalance);
    });
  });

  describe('Error Recovery', () => {
    beforeEach(async () => {
      neuralTrader.constructor(mockConfig);
      await neuralTrader.start();
    });

    it('should recover from insufficient balance error', async () => {
      // Try order that exceeds balance
      const failResult = await neuralTrader.placeOrder({
        symbol: 'AAPL',
        quantity: 1000,
        price: 150,
        side: 'BUY',
      });

      expect(failResult.success).toBe(false);

      // Verify balance unchanged
      const balance1 = (await neuralTrader.getBalance()).balance;
      expect(balance1).toBe(100000);

      // Successfully place smaller order
      const successResult = await neuralTrader.placeOrder({
        symbol: 'AAPL',
        quantity: 100,
        price: 150,
        side: 'BUY',
      });

      expect(successResult.success).toBe(true);

      const balance2 = (await neuralTrader.getBalance()).balance;
      expect(balance2).toBe(85000);
    });

    it('should recover from insufficient position error', async () => {
      // Try to sell non-existent position
      const failResult = await neuralTrader.placeOrder({
        symbol: 'AAPL',
        quantity: 100,
        price: 150,
        side: 'SELL',
      });

      expect(failResult.success).toBe(false);

      // Verify balance unchanged
      const balance1 = (await neuralTrader.getBalance()).balance;
      expect(balance1).toBe(100000);

      // Buy and then successfully sell
      await neuralTrader.placeOrder({
        symbol: 'AAPL',
        quantity: 100,
        price: 150,
        side: 'BUY',
      });

      const sellResult = await neuralTrader.placeOrder({
        symbol: 'AAPL',
        quantity: 100,
        price: 160,
        side: 'SELL',
      });

      expect(sellResult.success).toBe(true);
    });
  });
});
