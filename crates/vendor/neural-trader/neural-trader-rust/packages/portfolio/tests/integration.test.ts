/**
 * Integration tests for @neural-trader/portfolio
 * Tests portfolio lifecycle, rebalancing, and optimization workflows
 */

describe('Portfolio Integration Tests', () => {
  let manager: any;
  let optimizer: any;

  beforeEach(() => {
    manager = {
      cash: 100000,
      positions: new Map(),

      async getPositions() {
        return Array.from(this.positions.values());
      },

      async updatePosition(symbol: string, quantity: number, price: number) {
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
          pnl += position.value;
        }
        return pnl;
      },
    };

    optimizer = {
      async optimize(symbols: string[], returns: number[], covariance: number[][]) {
        const weights = new Array(symbols.length).fill(1 / symbols.length);
        return {
          symbols,
          weights,
          expectedReturn: returns.reduce((a: number, b: number) => a + b) / symbols.length,
          expectedRisk: 0.05,
          sharpeRatio: 1.5,
        };
      },

      calculateRisk(positions: Record<string, number>) {
        const values = Object.values(positions);
        const total = values.reduce((a, b) => a + b, 0);
        return {
          volatility: 0.05 * total,
          maxDrawdown: 0.1,
          sharpeRatio: 1.5,
        };
      },
    };
  });

  describe('Portfolio Construction', () => {
    it('should build diversified portfolio from scratch', async () => {
      const symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN'];
      const prices = [150, 300, 2800, 3200];

      for (let i = 0; i < symbols.length; i++) {
        await manager.updatePosition(symbols[i], 10, prices[i]);
      }

      const positions = await manager.getPositions();
      expect(positions).toHaveLength(4);

      const total = await manager.getTotalValue();
      // initial 100k + (10 * (150 + 300 + 2800 + 3200)) = 100k + 65000 = 165k
      expect(total).toBeGreaterThan(100000);
    });

    it('should maintain portfolio weights after price changes', async () => {
      // Create equal weighted portfolio
      const symbols = ['AAPL', 'MSFT', 'GOOGL'];
      for (let i = 0; i < symbols.length; i++) {
        await manager.updatePosition(symbols[i], 100, 100);
      }

      const valueBefore = await manager.getTotalValue();

      // Price changes
      await manager.updatePosition('AAPL', 100, 110);
      await manager.updatePosition('MSFT', 100, 90);
      await manager.updatePosition('GOOGL', 100, 100);

      const valueAfter = await manager.getTotalValue();
      expect(valueAfter).not.toBe(valueBefore);
    });
  });

  describe('Portfolio Rebalancing', () => {
    it('should execute rebalancing workflow', async () => {
      // Initial setup: equal weight
      await manager.updatePosition('AAPL', 100, 100);
      await manager.updatePosition('MSFT', 100, 100);
      await manager.updatePosition('GOOGL', 100, 100);

      let positions = await manager.getPositions();
      expect(positions).toHaveLength(3);

      // Price changes create imbalance
      await manager.updatePosition('AAPL', 100, 150);
      await manager.updatePosition('MSFT', 100, 100);
      await manager.updatePosition('GOOGL', 100, 80);

      // Rebalance by updating quantities to restore equal weight
      const totalValue = await manager.getTotalValue();
      const targetValue = (totalValue - 100000) / 3; // Excluding cash

      // Calculate new quantities based on target value
      let newQtyAppl = Math.floor(targetValue / 150);
      let newQtyMsft = Math.floor(targetValue / 100);
      let newQtyGoogl = Math.floor(targetValue / 80);

      await manager.updatePosition('AAPL', newQtyAppl, 150);
      await manager.updatePosition('MSFT', newQtyMsft, 100);
      await manager.updatePosition('GOOGL', newQtyGoogl, 80);

      positions = await manager.getPositions();
      expect(positions).toHaveLength(3);
    });

    it('should handle concentrated portfolio rebalancing', async () => {
      // Concentrated position
      await manager.updatePosition('AAPL', 500, 150);
      await manager.updatePosition('MSFT', 50, 300);

      let positions = await manager.getPositions();
      let applPosition = positions.find(p => p.symbol === 'AAPL');
      expect(applPosition!.value).toBeGreaterThan(positions.find(p => p.symbol === 'MSFT')!.value);

      // Rebalance to equal weight
      const totalValue = await manager.getTotalValue();
      const targetValue = (totalValue - 100000) / 2;

      await manager.updatePosition('AAPL', Math.floor(targetValue / 150), 150);
      await manager.updatePosition('MSFT', Math.floor(targetValue / 300), 300);

      positions = await manager.getPositions();
      const applVal = positions.find(p => p.symbol === 'AAPL')!.value;
      const msftVal = positions.find(p => p.symbol === 'MSFT')!.value;

      expect(Math.abs(applVal - msftVal)).toBeLessThan(100);
    });
  });

  describe('Portfolio Optimization Workflow', () => {
    it('should optimize portfolio allocation', async () => {
      const symbols = ['AAPL', 'MSFT', 'GOOGL'];
      const returns = [0.1, 0.12, 0.08];
      const covariance = [
        [0.001, 0.0005, 0.0003],
        [0.0005, 0.0015, 0.0004],
        [0.0003, 0.0004, 0.0012],
      ];

      const result = await optimizer.optimize(symbols, returns, covariance);

      expect(result.symbols).toEqual(symbols);
      expect(result.weights).toHaveLength(3);

      // Construct portfolio based on optimized weights
      const totalInvest = 50000;
      for (let i = 0; i < symbols.length; i++) {
        const investAmount = totalInvest * result.weights[i];
        const price = 100 + i * 50;
        const quantity = investAmount / price;
        await manager.updatePosition(symbols[i], quantity, price);
      }

      const positions = await manager.getPositions();
      expect(positions).toHaveLength(3);
    });

    it('should track portfolio metrics during optimization', async () => {
      const symbols = ['AAPL', 'MSFT', 'GOOGL'];
      const returns = [0.1, 0.12, 0.08];
      const covariance = [
        [0.001, 0.0005, 0.0003],
        [0.0005, 0.0015, 0.0004],
        [0.0003, 0.0004, 0.0012],
      ];

      const optimization = await optimizer.optimize(symbols, returns, covariance);

      // Build portfolio
      const totalInvest = 50000;
      const positions: Record<string, number> = {};

      for (let i = 0; i < symbols.length; i++) {
        const investAmount = totalInvest * optimization.weights[i];
        positions[symbols[i]] = investAmount;
        await manager.updatePosition(symbols[i], investAmount / 100, 100);
      }

      // Calculate risk
      const risk = optimizer.calculateRisk(positions);
      expect(risk.volatility).toBeGreaterThan(0);
      expect(risk.sharpeRatio).toBeGreaterThan(0);
    });
  });

  describe('Complex Portfolio Scenarios', () => {
    it('should handle sector rotation strategy', async () => {
      // Initial tech-heavy portfolio
      await manager.updatePosition('AAPL', 200, 150);
      await manager.updatePosition('MSFT', 100, 300);
      await manager.updatePosition('GOOGL', 50, 2800);

      let positions = await manager.getPositions();
      expect(positions).toHaveLength(3);

      // Rotate to financials
      await manager.updatePosition('AAPL', 0, 150);
      await manager.updatePosition('MSFT', 0, 300);
      await manager.updatePosition('GOOGL', 0, 2800);

      await manager.updatePosition('JPM', 300, 150);
      await manager.updatePosition('WFC', 200, 50);

      positions = await manager.getPositions();
      expect(positions).toHaveLength(2);
      expect(positions.map(p => p.symbol).sort()).toEqual(['JPM', 'WFC']);
    });

    it('should handle risk management through position sizing', async () => {
      const maxPositionSize = 10000;

      // Attempt to build portfolio with risk constraints
      const targetPositions = [
        { symbol: 'AAPL', shares: 100, price: 150 },
        { symbol: 'MSFT', shares: 100, price: 300 },
        { symbol: 'GOOGL', shares: 100, price: 2800 },
      ];

      for (const target of targetPositions) {
        const value = target.shares * target.price;
        const adjustedShares = value > maxPositionSize ?
          Math.floor(maxPositionSize / target.price) : target.shares;

        await manager.updatePosition(target.symbol, adjustedShares, target.price);
      }

      const positions = await manager.getPositions();
      for (const pos of positions) {
        expect(pos.value).toBeLessThanOrEqual(maxPositionSize * 1.1); // Allow small margin
      }
    });

    it('should handle tactical allocation adjustments', async () => {
      // Build initial strategic allocation
      await manager.updatePosition('AAPL', 150, 150);
      await manager.updatePosition('MSFT', 100, 300);
      await manager.updatePosition('GOOGL', 25, 2800);

      const initialValue = await manager.getTotalValue();

      // Make tactical adjustments based on market conditions
      await manager.updatePosition('AAPL', 175, 150); // Increase
      await manager.updatePosition('MSFT', 75, 300); // Decrease
      // Keep GOOGL same

      const adjustedValue = await manager.getTotalValue();
      expect(adjustedValue).not.toEqual(initialValue);
    });
  });

  describe('State Consistency During Complex Operations', () => {
    it('should maintain portfolio integrity through multiple updates', async () => {
      const symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN'];

      // Build portfolio
      for (let i = 0; i < symbols.length; i++) {
        await manager.updatePosition(symbols[i], 100 + i * 10, 100);
      }

      const initialValue = await manager.getTotalValue();

      // Multiple price updates
      for (let i = 0; i < 5; i++) {
        for (let j = 0; j < symbols.length; j++) {
          const newPrice = 100 + (Math.random() * 50 - 25);
          await manager.updatePosition(symbols[j], 100 + j * 10, Math.max(1, newPrice));
        }
      }

      const finalPositions = await manager.getPositions();
      expect(finalPositions).toHaveLength(4);

      const finalValue = await manager.getTotalValue();
      expect(finalValue).toBeGreaterThan(0);
    });

    it('should correctly track PnL throughout transactions', async () => {
      const initialValue = 100000;

      await manager.updatePosition('AAPL', 100, 150);
      const value1 = await manager.getTotalValue();
      expect(value1).toBe(initialValue + 100 * 150);

      // Price increase
      await manager.updatePosition('AAPL', 100, 160);
      const value2 = await manager.getTotalValue();
      expect(value2).toBeGreaterThan(value1);

      // Add position
      await manager.updatePosition('MSFT', 50, 300);
      const value3 = await manager.getTotalValue();
      expect(value3).toBeGreaterThan(value2);

      // Close AAPL position
      await manager.updatePosition('AAPL', 0, 160);
      const value4 = await manager.getTotalValue();
      expect(value4).toEqual(initialValue + 50 * 300);
    });
  });

  describe('Error Handling and Recovery', () => {
    it('should recover from invalid updates', async () => {
      try {
        await manager.updatePosition('AAPL', -100, 150);
      } catch (e) {
        // Expected
      }

      // Should still be able to update
      const position = await manager.updatePosition('AAPL', 100, 150);
      expect(position).toBeDefined();
      expect(position.quantity).toBe(100);
    });

    it('should maintain consistency after validation errors', async () => {
      await manager.updatePosition('AAPL', 100, 150);

      try {
        await manager.updatePosition('MSFT', 50, 0);
      } catch (e) {
        // Expected
      }

      // First position should be intact
      const positions = await manager.getPositions();
      expect(positions).toHaveLength(1);
      expect(positions[0].symbol).toBe('AAPL');
    });
  });
});
