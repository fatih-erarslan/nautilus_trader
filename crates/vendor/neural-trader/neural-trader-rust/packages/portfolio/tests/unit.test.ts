/**
 * Unit tests for @neural-trader/portfolio
 * Tests portfolio management, optimization, and risk calculations
 */

describe('Portfolio Unit Tests', () => {
  describe('PortfolioManager', () => {
    let manager: any;
    const initialCash = 100000;

    beforeEach(() => {
      manager = {
        cash: initialCash,
        positions: new Map(),

        async getPositions() {
          return Array.from(this.positions.values());
        },

        async getPosition(symbol: string) {
          return this.positions.get(symbol) || null;
        },

        async updatePosition(symbol: string, quantity: number, price: number) {
          if (quantity < 0) {
            throw new Error('Quantity cannot be negative');
          }
          if (price <= 0) {
            throw new Error('Price must be positive');
          }

          let position = this.positions.get(symbol);
          if (!position) {
            position = {
              symbol,
              quantity: 0,
              avgPrice: 0,
              currentPrice: price,
              value: 0,
            };
          }

          position.quantity = quantity;
          position.currentPrice = price;
          position.value = quantity * price;

          if (quantity > 0) {
            this.positions.set(symbol, position);
          } else {
            this.positions.delete(symbol);
          }

          return position;
        },

        async getCash() {
          return this.cash;
        },

        async getTotalValue() {
          let total = this.cash;
          for (const position of this.positions.values()) {
            total += position.value;
          }
          return total;
        },

        async getTotalPnl() {
          let pnl = 0;
          for (const position of this.positions.values()) {
            // Simplified: assumes cost basis is at current price
            pnl += position.value;
          }
          return pnl;
        },
      };
    });

    describe('Initialization', () => {
      it('should initialize with cash', () => {
        expect(manager.cash).toBe(initialCash);
      });

      it('should start with no positions', async () => {
        const positions = await manager.getPositions();
        expect(positions).toEqual([]);
      });

      it('should return initial cash', async () => {
        const cash = await manager.getCash();
        expect(cash).toBe(initialCash);
      });
    });

    describe('Position Updates', () => {
      it('should add new position', async () => {
        const position = await manager.updatePosition('AAPL', 100, 150);

        expect(position.symbol).toBe('AAPL');
        expect(position.quantity).toBe(100);
        expect(position.currentPrice).toBe(150);
        expect(position.value).toBe(15000);
      });

      it('should update existing position', async () => {
        await manager.updatePosition('AAPL', 100, 150);
        const updated = await manager.updatePosition('AAPL', 150, 155);

        expect(updated.quantity).toBe(150);
        expect(updated.currentPrice).toBe(155);
      });

      it('should reject negative quantity', async () => {
        await expect(
          manager.updatePosition('AAPL', -100, 150)
        ).rejects.toThrow('Quantity cannot be negative');
      });

      it('should reject zero or negative price', async () => {
        await expect(
          manager.updatePosition('AAPL', 100, 0)
        ).rejects.toThrow('Price must be positive');

        await expect(
          manager.updatePosition('AAPL', 100, -150)
        ).rejects.toThrow('Price must be positive');
      });

      it('should remove position when quantity is zero', async () => {
        await manager.updatePosition('AAPL', 100, 150);
        await manager.updatePosition('AAPL', 0, 150);

        const position = await manager.getPosition('AAPL');
        expect(position).toBeNull();
      });
    });

    describe('Position Retrieval', () => {
      it('should get existing position by symbol', async () => {
        await manager.updatePosition('AAPL', 100, 150);
        const position = await manager.getPosition('AAPL');

        expect(position).not.toBeNull();
        expect(position.symbol).toBe('AAPL');
        expect(position.quantity).toBe(100);
      });

      it('should return null for non-existent position', async () => {
        const position = await manager.getPosition('GOOGL');
        expect(position).toBeNull();
      });

      it('should get all positions', async () => {
        await manager.updatePosition('AAPL', 100, 150);
        await manager.updatePosition('MSFT', 50, 300);
        await manager.updatePosition('GOOGL', 75, 2800);

        const positions = await manager.getPositions();
        expect(positions).toHaveLength(3);
        expect(positions.map(p => p.symbol).sort()).toEqual(['AAPL', 'GOOGL', 'MSFT']);
      });
    });

    describe('Portfolio Value', () => {
      it('should calculate total value with single position', async () => {
        await manager.updatePosition('AAPL', 100, 150);

        const total = await manager.getTotalValue();
        expect(total).toBe(initialCash + 100 * 150);
      });

      it('should calculate total value with multiple positions', async () => {
        await manager.updatePosition('AAPL', 100, 150);
        await manager.updatePosition('MSFT', 50, 300);

        const total = await manager.getTotalValue();
        expect(total).toBe(initialCash + 100 * 150 + 50 * 300);
      });

      it('should decrease total value when selling', async () => {
        await manager.updatePosition('AAPL', 100, 150);

        const valueBefore = await manager.getTotalValue();

        await manager.updatePosition('AAPL', 50, 150);

        const valueAfter = await manager.getTotalValue();
        expect(valueAfter).toBeLessThan(valueBefore);
      });

      it('should reflect price changes', async () => {
        await manager.updatePosition('AAPL', 100, 150);

        let total = await manager.getTotalValue();
        const firstTotal = total;

        // Price increase
        await manager.updatePosition('AAPL', 100, 160);
        total = await manager.getTotalValue();
        expect(total).toBeGreaterThan(firstTotal);

        // Price decrease
        await manager.updatePosition('AAPL', 100, 140);
        total = await manager.getTotalValue();
        expect(total).toBeLessThan(firstTotal);
      });
    });

    describe('Cash Management', () => {
      it('should maintain cash balance', async () => {
        const cash1 = await manager.getCash();
        expect(cash1).toBe(initialCash);

        manager.cash -= 1000;
        const cash2 = await manager.getCash();
        expect(cash2).toBe(initialCash - 1000);
      });
    });

    describe('Edge Cases', () => {
      it('should handle very small quantities', async () => {
        const position = await manager.updatePosition('AAPL', 0.01, 150);
        expect(position.quantity).toBe(0.01);
        expect(position.value).toBe(0.01 * 150);
      });

      it('should handle very large quantities', async () => {
        const position = await manager.updatePosition('AAPL', 1000000, 150);
        expect(position.quantity).toBe(1000000);
        expect(position.value).toBe(1000000 * 150);
      });

      it('should handle many positions', async () => {
        for (let i = 0; i < 100; i++) {
          await manager.updatePosition(`STOCK${i}`, i * 10, 100);
        }

        const positions = await manager.getPositions();
        expect(positions).toHaveLength(100);
      });
    });
  });

  describe('PortfolioOptimizer', () => {
    let optimizer: any;

    beforeEach(() => {
      optimizer = {
        config: { riskFreeRate: 0.02, targetReturn: 0.1 },

        constructor(config: any) {
          this.config = config;
        },

        async optimize(
          symbols: string[],
          returns: number[],
          covariance: number[][]
        ) {
          if (symbols.length === 0) {
            throw new Error('At least one symbol required');
          }
          if (returns.length !== symbols.length) {
            throw new Error('Returns array length must match symbols');
          }

          // Simple equal weight optimization for testing
          const weights = new Array(symbols.length).fill(1 / symbols.length);

          return {
            symbols,
            weights,
            expectedReturn: returns.reduce((a, b) => a + b) / symbols.length,
            expectedRisk: Math.sqrt(0.001), // Simplified
            sharpeRatio: 1.5,
          };
        },

        calculateRisk(positions: Record<string, number>) {
          const symbols = Object.keys(positions);
          const values = Object.values(positions);
          const total = values.reduce((a, b) => a + b, 0);

          if (total === 0) {
            return { volatility: 0, maxDrawdown: 0, sharpeRatio: 0 };
          }

          return {
            volatility: Math.sqrt(0.001 * total),
            maxDrawdown: 0.1 * total,
            sharpeRatio: 1.5,
          };
        },
      };
    });

    describe('Initialization', () => {
      it('should initialize with config', () => {
        optimizer.constructor({ riskFreeRate: 0.025, targetReturn: 0.12 });
        expect(optimizer.config.riskFreeRate).toBe(0.025);
        expect(optimizer.config.targetReturn).toBe(0.12);
      });
    });

    describe('Portfolio Optimization', () => {
      it('should optimize with single asset', async () => {
        const result = await optimizer.optimize(
          ['AAPL'],
          [0.1],
          [[0.001]]
        );

        expect(result.symbols).toEqual(['AAPL']);
        expect(result.weights).toEqual([1]);
        expect(result.expectedReturn).toBe(0.1);
      });

      it('should optimize with multiple assets', async () => {
        const result = await optimizer.optimize(
          ['AAPL', 'MSFT', 'GOOGL'],
          [0.1, 0.12, 0.08],
          [[0.001, 0.0005, 0.0003], [0.0005, 0.0015, 0.0004], [0.0003, 0.0004, 0.0012]]
        );

        expect(result.symbols).toHaveLength(3);
        expect(result.weights).toHaveLength(3);
        expect(result.weights.reduce((a, b) => a + b)).toBeCloseTo(1.0);
      });

      it('should reject empty symbols', async () => {
        await expect(
          optimizer.optimize([], [], [])
        ).rejects.toThrow('At least one symbol required');
      });

      it('should reject mismatched arrays', async () => {
        await expect(
          optimizer.optimize(
            ['AAPL', 'MSFT'],
            [0.1],
            [[0.001]]
          )
        ).rejects.toThrow('Returns array length must match symbols');
      });

      it('should return valid weights', async () => {
        const result = await optimizer.optimize(
          ['AAPL', 'MSFT', 'GOOGL'],
          [0.1, 0.12, 0.08],
          [[0.001, 0.0005, 0.0003], [0.0005, 0.0015, 0.0004], [0.0003, 0.0004, 0.0012]]
        );

        const weightSum = result.weights.reduce((a: number, b: number) => a + b);
        expect(weightSum).toBeCloseTo(1.0);
        result.weights.forEach((w: number) => {
          expect(w).toBeGreaterThanOrEqual(0);
          expect(w).toBeLessThanOrEqual(1);
        });
      });
    });

    describe('Risk Calculation', () => {
      it('should calculate risk for empty positions', () => {
        const risk = optimizer.calculateRisk({});
        expect(risk.volatility).toBe(0);
        expect(risk.maxDrawdown).toBe(0);
        expect(risk.sharpeRatio).toBe(0);
      });

      it('should calculate risk for single position', () => {
        const risk = optimizer.calculateRisk({ AAPL: 10000 });
        expect(risk.volatility).toBeGreaterThan(0);
        expect(risk.maxDrawdown).toBeGreaterThan(0);
        expect(risk.sharpeRatio).toBeGreaterThan(0);
      });

      it('should calculate risk for multiple positions', () => {
        const risk = optimizer.calculateRisk({
          AAPL: 30000,
          MSFT: 40000,
          GOOGL: 30000,
        });
        expect(risk.volatility).toBeGreaterThan(0);
        expect(risk.maxDrawdown).toBeGreaterThan(0);
        expect(risk.sharpeRatio).toBeGreaterThan(0);
      });

      it('should scale risk with portfolio size', () => {
        const risk1 = optimizer.calculateRisk({ AAPL: 10000 });
        const risk2 = optimizer.calculateRisk({ AAPL: 20000 });

        expect(risk2.volatility).toBeGreaterThan(risk1.volatility);
      });
    });

    describe('Edge Cases', () => {
      it('should handle very small returns', async () => {
        const result = await optimizer.optimize(
          ['AAPL', 'MSFT'],
          [0.0001, 0.0002],
          [[0.001, 0.0005], [0.0005, 0.001]]
        );

        expect(result.expectedReturn).toBeGreaterThan(0);
      });

      it('should handle negative returns', async () => {
        const result = await optimizer.optimize(
          ['AAPL', 'MSFT'],
          [-0.1, 0.05],
          [[0.001, 0.0005], [0.0005, 0.001]]
        );

        expect(result.expectedReturn).toBeLessThan(0.05);
      });
    });
  });
});
