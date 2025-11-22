const { TradingEngine } = require('../../quantum_trading/core/trading_engine');
const { OrderBook } = require('../../quantum_trading/core/order_book');
const { RiskManager } = require('../../quantum_trading/core/risk_manager');
const { PositionManager } = require('../../quantum_trading/core/position_manager');
const { AuditLogger } = require('../../quantum_trading/compliance/audit_logger');
const { MarketDataFeed } = require('../../quantum_trading/data/market_data_feed');

describe('Component Integration Testing', () => {
  let tradingEngine;
  let orderBook;
  let riskManager;
  let positionManager;
  let auditLogger;
  let marketDataFeed;

  beforeEach(async () => {
    // Initialize all components with real implementations
    auditLogger = new AuditLogger();
    orderBook = new OrderBook({ auditLogger });
    riskManager = new RiskManager({ auditLogger });
    positionManager = new PositionManager({ auditLogger });
    marketDataFeed = new MarketDataFeed();
    
    tradingEngine = new TradingEngine({
      orderBook,
      riskManager,
      positionManager,
      auditLogger,
      marketDataFeed
    });

    await tradingEngine.initialize();
  });

  afterEach(async () => {
    await tradingEngine.shutdown();
  });

  describe('Order Processing Flow Integration', () => {
    test('should handle complete order lifecycle with all components', async () => {
      const userId = 'INTEGRATION_USER_001';
      
      // Set up user with credit limit
      await riskManager.setCreditLimit(userId, 1000000);
      await positionManager.initializeUserPosition(userId, 'AAPL');

      // Submit market buy order
      const buyOrder = {
        id: 'INTEGRATION_BUY_001',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        type: 'market',
        timestamp: Date.now(),
        userId
      };

      // Process order - should flow through all components
      const buyResult = await tradingEngine.processOrder(buyOrder);
      expect(buyResult.success).toBe(true);

      // Verify risk manager was consulted
      const riskLogs = await auditLogger.getLogsByType('RISK_CHECK');
      expect(riskLogs).toContainEqual(
        expect.objectContaining({
          orderId: 'INTEGRATION_BUY_001',
          userId,
          result: 'APPROVED'
        })
      );

      // Verify order was added to order book
      const orderBookState = await orderBook.getSnapshot();
      const order = await orderBook.getOrderById('INTEGRATION_BUY_001');
      expect(order).toBeDefined();
      expect(order.status).toBe('active');

      // Add a matching sell order
      const sellOrder = {
        id: 'INTEGRATION_SELL_001',
        symbol: 'AAPL',
        side: 'sell',
        quantity: 100,
        price: 150.00,
        type: 'limit',
        timestamp: Date.now() + 1000,
        userId: 'SELLER_001'
      };

      await riskManager.setCreditLimit('SELLER_001', 1000000);
      await positionManager.initializeUserPosition('SELLER_001', 'AAPL', { quantity: 100, avgCost: 140.00 });

      const sellResult = await tradingEngine.processOrder(sellOrder);
      expect(sellResult.success).toBe(true);

      // Wait for trade execution
      await new Promise(resolve => setTimeout(resolve, 100));

      // Verify trade execution and position updates
      const buyerPosition = await positionManager.getPosition(userId, 'AAPL');
      const sellerPosition = await positionManager.getPosition('SELLER_001', 'AAPL');

      expect(buyerPosition.quantity).toBe(100);
      expect(sellerPosition.quantity).toBe(0);

      // Verify audit trail completeness
      const auditTrail = await auditLogger.getCompleteAuditTrail();
      const orderEvents = auditTrail.filter(log => 
        log.orderId === 'INTEGRATION_BUY_001' || log.orderId === 'INTEGRATION_SELL_001'
      );
      
      expect(orderEvents).toContainEqual(expect.objectContaining({ event: 'ORDER_RECEIVED' }));
      expect(orderEvents).toContainEqual(expect.objectContaining({ event: 'RISK_CHECK_PASSED' }));
      expect(orderEvents).toContainEqual(expect.objectContaining({ event: 'ORDER_MATCHED' }));
      expect(orderEvents).toContainEqual(expect.objectContaining({ event: 'TRADE_EXECUTED' }));
      expect(orderEvents).toContainEqual(expect.objectContaining({ event: 'POSITION_UPDATED' }));
    });

    test('should handle order rejection flow across components', async () => {
      const userId = 'REJECTED_USER_001';
      
      // Set very low credit limit to trigger rejection
      await riskManager.setCreditLimit(userId, 100);

      const largeOrder = {
        id: 'LARGE_ORDER_001',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 10000,
        price: 150.00,
        type: 'limit',
        timestamp: Date.now(),
        userId
      };

      const result = await tradingEngine.processOrder(largeOrder);
      expect(result.success).toBe(false);
      expect(result.error).toContain('Credit limit exceeded');

      // Verify order was never added to order book
      const order = await orderBook.getOrderById('LARGE_ORDER_001');
      expect(order).toBeNull();

      // Verify position was not created
      const position = await positionManager.getPosition(userId, 'AAPL');
      expect(position.quantity).toBe(0);

      // Verify complete audit trail for rejection
      const rejectionLogs = await auditLogger.getLogsByOrderId('LARGE_ORDER_001');
      expect(rejectionLogs).toContainEqual(
        expect.objectContaining({
          event: 'ORDER_REJECTED',
          reason: 'CREDIT_LIMIT_EXCEEDED'
        })
      );
    });
  });

  describe('Market Data Integration', () => {
    test('should process market data updates across all components', async () => {
      const symbol = 'AAPL';
      
      // Subscribe to market data
      await marketDataFeed.subscribe(symbol);
      
      // Inject market data update
      const marketUpdate = {
        symbol,
        lastPrice: 155.50,
        bid: 155.45,
        ask: 155.55,
        bidSize: 1000,
        askSize: 800,
        timestamp: Date.now()
      };

      await marketDataFeed.publishUpdate(marketUpdate);
      
      // Wait for propagation
      await new Promise(resolve => setTimeout(resolve, 50));

      // Verify order book received update
      const bestBid = await orderBook.getBestBid(symbol);
      const bestOffer = await orderBook.getBestOffer(symbol);
      
      expect(bestBid?.price).toBe(155.45);
      expect(bestOffer?.price).toBe(155.55);

      // Verify risk manager has latest price
      const latestPrice = await riskManager.getLatestPrice(symbol);
      expect(latestPrice).toBe(155.50);

      // Verify position manager calculated mark-to-market
      const userId = 'MTM_USER_001';
      await positionManager.initializeUserPosition(userId, symbol, { quantity: 100, avgCost: 150.00 });
      
      const position = await positionManager.getPosition(userId, symbol);
      const expectedPnL = (155.50 - 150.00) * 100;
      expect(position.unrealizedPnL).toBeCloseTo(expectedPnL, 2);
    });

    test('should handle market data feed disconnection and recovery', async () => {
      const symbol = 'AAPL';
      await marketDataFeed.subscribe(symbol);

      // Simulate market data feed disconnection
      await marketDataFeed.simulateDisconnection();

      // Verify system enters degraded mode
      const systemStatus = await tradingEngine.getSystemStatus();
      expect(systemStatus.marketDataStatus).toBe('DEGRADED');

      // Orders should still be processed but with warnings
      const order = {
        id: 'DEGRADED_ORDER_001',
        symbol,
        side: 'buy',
        quantity: 100,
        price: 150.00,
        type: 'limit',
        timestamp: Date.now(),
        userId: 'DEGRADED_USER_001'
      };

      await riskManager.setCreditLimit('DEGRADED_USER_001', 100000);
      const result = await tradingEngine.processOrder(order);
      
      expect(result.success).toBe(true);
      expect(result.warnings).toContain('STALE_MARKET_DATA');

      // Simulate reconnection
      await marketDataFeed.reconnect();
      await new Promise(resolve => setTimeout(resolve, 100));

      // Verify system returns to normal
      const recoveredStatus = await tradingEngine.getSystemStatus();
      expect(recoveredStatus.marketDataStatus).toBe('ACTIVE');
    });
  });

  describe('Risk Management Integration', () => {
    test('should enforce position limits across multiple orders', async () => {
      const userId = 'POSITION_LIMIT_USER';
      const symbol = 'AAPL';
      
      // Set position limit
      await riskManager.setPositionLimit(userId, symbol, 500);
      await riskManager.setCreditLimit(userId, 1000000);

      // Place orders approaching limit
      const orders = Array.from({ length: 6 }, (_, i) => ({
        id: `POSITION_ORDER_${i}`,
        symbol,
        side: 'buy',
        quantity: 100,
        price: 150.00,
        type: 'limit',
        timestamp: Date.now() + i,
        userId
      }));

      const results = [];
      for (const order of orders) {
        const result = await tradingEngine.processOrder(order);
        results.push(result);
      }

      // First 5 orders should succeed
      expect(results.slice(0, 5).every(r => r.success)).toBe(true);
      
      // 6th order should be rejected due to position limit
      expect(results[5].success).toBe(false);
      expect(results[5].error).toContain('Position limit exceeded');

      // Verify position manager shows correct position
      const position = await positionManager.getPosition(userId, symbol);
      expect(position.quantity).toBe(500);

      // Verify audit trail shows limit enforcement
      const limitLogs = await auditLogger.getLogsByType('POSITION_LIMIT_CHECK');
      expect(limitLogs.length).toBeGreaterThan(0);
    });

    test('should handle real-time risk monitoring and alerts', async () => {
      const userId = 'RISK_MONITOR_USER';
      const symbol = 'AAPL';
      
      // Set risk thresholds
      await riskManager.setRiskThresholds(userId, {
        maxPositionValue: 100000,
        maxDailyLoss: 10000,
        maxLeverage: 2.0
      });

      // Track alerts
      const alerts = [];
      riskManager.on('riskAlert', (alert) => {
        alerts.push(alert);
      });

      // Place large orders
      const largeOrders = Array.from({ length: 3 }, (_, i) => ({
        id: `RISK_ORDER_${i}`,
        symbol,
        side: 'buy',
        quantity: 200,
        price: 150.00 + (i * 10), // Increasing prices
        type: 'limit',
        timestamp: Date.now() + (i * 1000),
        userId
      }));

      await riskManager.setCreditLimit(userId, 1000000);
      
      for (const order of largeOrders) {
        await tradingEngine.processOrder(order);
        // Wait for risk calculations
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      // Should have generated risk alerts
      expect(alerts.length).toBeGreaterThan(0);
      expect(alerts).toContainEqual(
        expect.objectContaining({
          type: 'POSITION_VALUE_WARNING',
          userId,
          threshold: 100000
        })
      );

      // Verify position manager and risk manager are in sync
      const position = await positionManager.getPosition(userId, symbol);
      const riskMetrics = await riskManager.getRealTimeMetrics(userId);
      
      expect(riskMetrics.currentPositionValue).toBeCloseTo(position.marketValue, 2);
    });
  });

  describe('Error Propagation and Recovery', () => {
    test('should handle component failures gracefully', async () => {
      const userId = 'FAILURE_TEST_USER';
      
      // Simulate position manager failure
      const originalUpdate = positionManager.updatePosition;
      positionManager.updatePosition = jest.fn().mockRejectedValue(new Error('Position update failed'));

      const order = {
        id: 'FAILURE_ORDER_001',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00,
        type: 'limit',
        timestamp: Date.now(),
        userId
      };

      await riskManager.setCreditLimit(userId, 100000);
      const result = await tradingEngine.processOrder(order);

      // Order should be processed but with error handling
      expect(result.success).toBe(false);
      expect(result.error).toContain('Position update failed');

      // Verify system maintains consistency
      const orderFromBook = await orderBook.getOrderById('FAILURE_ORDER_001');
      expect(orderFromBook).toBeNull(); // Order should not be in book if position update failed

      // Restore position manager
      positionManager.updatePosition = originalUpdate;

      // Verify system recovery
      const recoveryOrder = {
        id: 'RECOVERY_ORDER_001',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00,
        type: 'limit',
        timestamp: Date.now(),
        userId
      };

      const recoveryResult = await tradingEngine.processOrder(recoveryOrder);
      expect(recoveryResult.success).toBe(true);
    });

    test('should maintain data consistency across component boundaries', async () => {
      const userId = 'CONSISTENCY_USER';
      const symbol = 'AAPL';
      
      await riskManager.setCreditLimit(userId, 1000000);

      // Execute multiple operations concurrently
      const operations = [
        () => tradingEngine.processOrder({
          id: 'CONCURRENT_1',
          symbol,
          side: 'buy',
          quantity: 100,
          price: 150.00,
          type: 'limit',
          timestamp: Date.now(),
          userId
        }),
        () => tradingEngine.processOrder({
          id: 'CONCURRENT_2',
          symbol,
          side: 'sell',
          quantity: 50,
          price: 149.00,
          type: 'limit',
          timestamp: Date.now() + 1,
          userId: 'OTHER_USER'
        }),
        () => positionManager.getPosition(userId, symbol),
        () => riskManager.getRealTimeMetrics(userId),
        () => orderBook.getSnapshot()
      ];

      await riskManager.setCreditLimit('OTHER_USER', 1000000);
      await positionManager.initializeUserPosition('OTHER_USER', symbol, { quantity: 100, avgCost: 140.00 });

      const results = await Promise.all(operations.map(op => op()));

      // Verify all operations completed without corruption
      expect(results).toHaveLength(5);
      results.forEach(result => {
        expect(result).toBeDefined();
      });

      // Verify data consistency across components
      const finalPosition = await positionManager.getPosition(userId, symbol);
      const finalRiskMetrics = await riskManager.getRealTimeMetrics(userId);
      const finalOrderBook = await orderBook.getSnapshot();

      // Position value should match risk metrics
      expect(finalRiskMetrics.currentPositionValue).toBeCloseTo(finalPosition.marketValue, 2);

      // Order book should reflect actual orders
      const userOrders = finalOrderBook.bids.filter(order => order.userId === userId);
      expect(userOrders.length).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Performance Integration', () => {
    test('should maintain performance across integrated components', async () => {
      const userId = 'PERFORMANCE_USER';
      const symbol = 'AAPL';
      
      await riskManager.setCreditLimit(userId, 10000000);

      // Generate realistic order flow
      const orders = Array.from({ length: 1000 }, (_, i) => ({
        id: `PERF_ORDER_${i}`,
        symbol,
        side: i % 2 === 0 ? 'buy' : 'sell',
        quantity: Math.floor(Math.random() * 1000) + 100,
        price: 150.00 + (Math.random() - 0.5) * 10,
        type: 'limit',
        timestamp: Date.now() + i,
        userId: i % 2 === 0 ? userId : `OTHER_USER_${i % 10}`
      }));

      // Set up other users
      for (let i = 0; i < 10; i++) {
        await riskManager.setCreditLimit(`OTHER_USER_${i}`, 1000000);
        await positionManager.initializeUserPosition(`OTHER_USER_${i}`, symbol, { 
          quantity: 1000, 
          avgCost: 145.00 
        });
      }

      const startTime = performance.now();
      const results = [];

      // Process orders in batches
      for (let i = 0; i < orders.length; i += 50) {
        const batch = orders.slice(i, i + 50);
        const batchResults = await Promise.all(
          batch.map(order => tradingEngine.processOrder(order))
        );
        results.push(...batchResults);
      }

      const processingTime = performance.now() - startTime;
      const avgLatency = processingTime / orders.length;

      // Performance requirements
      expect(avgLatency).toBeLessThan(50); // <50ms average latency
      expect(processingTime).toBeLessThan(30000); // <30s total time

      // Verify system consistency after high load
      const systemHealth = await tradingEngine.healthCheck();
      expect(systemHealth.status).toBe('healthy');
      expect(systemHealth.componentStatus.orderBook).toBe('healthy');
      expect(systemHealth.componentStatus.riskManager).toBe('healthy');
      expect(systemHealth.componentStatus.positionManager).toBe('healthy');

      // Verify successful order rate
      const successfulOrders = results.filter(r => r.success).length;
      const successRate = successfulOrders / results.length;
      expect(successRate).toBeGreaterThan(0.95); // >95% success rate
    });
  });
});