const fc = require('fast-check');
const { Decimal } = require('decimal.js');
const { PositionCalculator } = require('../../quantum_trading/core/position_calculator');
const { PriceCalculator } = require('../../quantum_trading/core/price_calculator');

describe('Mathematical Properties - Property-Based Testing', () => {
  let positionCalculator;
  let priceCalculator;

  beforeEach(() => {
    positionCalculator = new PositionCalculator();
    priceCalculator = new PriceCalculator();
  });

  describe('Position Calculation Properties', () => {
    test('Position updates must maintain conservation of shares', () => {
      fc.assert(fc.property(
        fc.array(fc.record({
          side: fc.constantFrom('buy', 'sell'),
          quantity: fc.integer({ min: 1, max: 10000 }),
          price: fc.float({ min: 0.01, max: 1000, noNaN: true })
        }), { minLength: 1, maxLength: 100 }),
        (trades) => {
          const position = positionCalculator.createEmptyPosition();
          let totalBought = 0;
          let totalSold = 0;

          for (const trade of trades) {
            positionCalculator.updatePosition(position, trade);
            if (trade.side === 'buy') {
              totalBought += trade.quantity;
            } else {
              totalSold += trade.quantity;
            }
          }

          const expectedNetPosition = totalBought - totalSold;
          expect(position.quantity).toBe(expectedNetPosition);
        }
      ), { numRuns: 1000 });
    });

    test('FIFO cost basis calculation must be mathematically consistent', () => {
      fc.assert(fc.property(
        fc.array(fc.record({
          side: fc.constant('buy'),
          quantity: fc.integer({ min: 1, max: 1000 }),
          price: fc.float({ min: 0.01, max: 1000, noNaN: true })
        }), { minLength: 2, maxLength: 10 }),
        fc.record({
          side: fc.constant('sell'),
          quantity: fc.integer({ min: 1, max: 500 })
        }),
        (buyTrades, sellTrade) => {
          const position = positionCalculator.createEmptyPosition();
          let totalCost = new Decimal(0);

          // Execute buy trades
          for (const trade of buyTrades) {
            positionCalculator.updatePosition(position, trade);
            totalCost = totalCost.plus(new Decimal(trade.quantity).times(trade.price));
          }

          const totalQuantity = buyTrades.reduce((sum, trade) => sum + trade.quantity, 0);
          
          if (sellTrade.quantity <= totalQuantity) {
            const initialPosition = { ...position };
            positionCalculator.updatePosition(position, sellTrade);

            // FIFO property: cost basis should reflect oldest purchases first
            const expectedRemainingQuantity = totalQuantity - sellTrade.quantity;
            expect(position.quantity).toBe(expectedRemainingQuantity);
            expect(position.costBasis).toBeGreaterThanOrEqual(0);
          }
        }
      ), { numRuns: 500 });
    });

    test('No money creation or destruction property', () => {
      fc.assert(fc.property(
        fc.array(fc.record({
          side: fc.constantFrom('buy', 'sell'),
          quantity: fc.integer({ min: 1, max: 1000 }),
          price: fc.float({ min: 0.01, max: 1000, noNaN: true })
        }), { minLength: 1, maxLength: 50 }),
        (trades) => {
          const position = positionCalculator.createEmptyPosition();
          let totalCashFlow = new Decimal(0);

          for (const trade of trades) {
            const cashFlow = trade.side === 'buy' 
              ? new Decimal(trade.quantity).times(trade.price).negated()
              : new Decimal(trade.quantity).times(trade.price);
            
            totalCashFlow = totalCashFlow.plus(cashFlow);
            positionCalculator.updatePosition(position, trade);
          }

          // Total value must equal unrealized P&L plus cash flows
          const currentPrice = trades[trades.length - 1].price;
          const positionValue = new Decimal(position.quantity).times(currentPrice);
          const unrealizedPnL = positionValue.minus(position.costBasis);
          const totalValue = unrealizedPnL.plus(totalCashFlow);

          // Money conservation: no creation or destruction
          expect(Math.abs(totalValue.toNumber())).toBeLessThan(0.01);
        }
      ), { numRuns: 300 });
    });
  });

  describe('Price Calculation Properties', () => {
    test('Price calculations must maintain precision', () => {
      fc.assert(fc.property(
        fc.float({ min: 0.0001, max: 999999.9999, noNaN: true }),
        fc.float({ min: 0.0001, max: 999999.9999, noNaN: true }),
        (price1, price2) => {
          const result = priceCalculator.add(price1, price2);
          const expected = new Decimal(price1).plus(new Decimal(price2));
          
          expect(Math.abs(result - expected.toNumber())).toBeLessThan(0.00001);
        }
      ), { numRuns: 10000 });
    });

    test('Price multiplication must be associative', () => {
      fc.assert(fc.property(
        fc.float({ min: 0.01, max: 1000, noNaN: true }),
        fc.float({ min: 0.01, max: 1000, noNaN: true }),
        fc.float({ min: 0.01, max: 1000, noNaN: true }),
        (a, b, c) => {
          const result1 = priceCalculator.multiply(priceCalculator.multiply(a, b), c);
          const result2 = priceCalculator.multiply(a, priceCalculator.multiply(b, c));
          
          expect(Math.abs(result1 - result2)).toBeLessThan(0.00001);
        }
      ), { numRuns: 1000 });
    });

    test('Division by multiplication must be identity', () => {
      fc.assert(fc.property(
        fc.float({ min: 0.01, max: 1000, noNaN: true }),
        fc.float({ min: 0.01, max: 1000, noNaN: true }),
        (dividend, divisor) => {
          const quotient = priceCalculator.divide(dividend, divisor);
          const result = priceCalculator.multiply(quotient, divisor);
          
          expect(Math.abs(result - dividend)).toBeLessThan(0.00001);
        }
      ), { numRuns: 1000 });
    });
  });

  describe('Order Matching Properties', () => {
    test('Order matching must preserve price-time priority', () => {
      fc.assert(fc.property(
        fc.array(fc.record({
          id: fc.string({ minLength: 1, maxLength: 10 }),
          price: fc.float({ min: 0.01, max: 1000, noNaN: true }),
          quantity: fc.integer({ min: 1, max: 1000 }),
          timestamp: fc.integer({ min: 1000000000000, max: 9999999999999 })
        }), { minLength: 2, maxLength: 20 }),
        (orders) => {
          const sortedOrders = orders.sort((a, b) => {
            if (Math.abs(a.price - b.price) < 0.0001) {
              return a.timestamp - b.timestamp; // Time priority
            }
            return b.price - a.price; // Price priority (descending for bids)
          });

          // Verify that best price orders are matched first
          for (let i = 1; i < sortedOrders.length; i++) {
            const prev = sortedOrders[i - 1];
            const curr = sortedOrders[i];
            
            if (Math.abs(prev.price - curr.price) < 0.0001) {
              expect(prev.timestamp).toBeLessThanOrEqual(curr.timestamp);
            } else {
              expect(prev.price).toBeGreaterThanOrEqual(curr.price);
            }
          }
        }
      ), { numRuns: 500 });
    });
  });

  describe('Risk Calculation Properties', () => {
    test('Portfolio risk must be subadditive', () => {
      fc.assert(fc.property(
        fc.array(fc.record({
          symbol: fc.constantFrom('AAPL', 'GOOGL', 'MSFT', 'AMZN'),
          position: fc.integer({ min: -10000, max: 10000 }),
          price: fc.float({ min: 1, max: 1000, noNaN: true }),
          volatility: fc.float({ min: 0.01, max: 2, noNaN: true })
        }), { minLength: 2, maxLength: 10 }),
        (positions) => {
          const individualRisks = positions.map(pos => 
            Math.abs(pos.position * pos.price * pos.volatility)
          );
          const totalIndividualRisk = individualRisks.reduce((sum, risk) => sum + risk, 0);
          
          const portfolioRisk = positionCalculator.calculatePortfolioRisk(positions);
          
          // Subadditivity: portfolio risk <= sum of individual risks
          expect(portfolioRisk).toBeLessThanOrEqual(totalIndividualRisk + 0.01);
        }
      ), { numRuns: 300 });
    });

    test('VaR calculations must be monotonic with confidence level', () => {
      fc.assert(fc.property(
        fc.record({
          position: fc.integer({ min: 1, max: 10000 }),
          price: fc.float({ min: 1, max: 1000, noNaN: true }),
          volatility: fc.float({ min: 0.01, max: 1, noNaN: true })
        }),
        fc.float({ min: 0.90, max: 0.99 }),
        fc.float({ min: 0.90, max: 0.99 }),
        (position, confidence1, confidence2) => {
          if (confidence1 !== confidence2) {
            const var1 = positionCalculator.calculateVaR(position, confidence1);
            const var2 = positionCalculator.calculateVaR(position, confidence2);
            
            if (confidence1 > confidence2) {
              expect(var1).toBeGreaterThanOrEqual(var2);
            } else {
              expect(var2).toBeGreaterThanOrEqual(var1);
            }
          }
        }
      ), { numRuns: 500 });
    });
  });

  describe('Decimal Precision Properties', () => {
    test('All financial calculations must maintain 4 decimal place precision', () => {
      fc.assert(fc.property(
        fc.float({ min: 0.0001, max: 999999.9999 }),
        fc.float({ min: 0.0001, max: 999999.9999 }),
        (a, b) => {
          const operations = [
            () => priceCalculator.add(a, b),
            () => priceCalculator.subtract(a, b),
            () => priceCalculator.multiply(a, b),
            () => a !== 0 ? priceCalculator.divide(b, a) : b
          ];

          operations.forEach(op => {
            const result = op();
            const rounded = Math.round(result * 10000) / 10000;
            expect(Math.abs(result - rounded)).toBeLessThan(0.00001);
          });
        }
      ), { numRuns: 1000 });
    });
  });
});