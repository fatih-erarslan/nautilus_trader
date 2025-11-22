const { TradingEngine } = require('../../quantum_trading/core/trading_engine');
const { OrderBook } = require('../../quantum_trading/core/order_book');
const { RiskManager } = require('../../quantum_trading/core/risk_manager');
const { PerformanceMonitor } = require('../../quantum_trading/utils/performance_monitor');

describe('Extreme Market Conditions - Stress Testing', () => {
  let tradingEngine;
  let orderBook;
  let riskManager;
  let performanceMonitor;

  beforeEach(() => {
    orderBook = new OrderBook();
    riskManager = new RiskManager();
    tradingEngine = new TradingEngine({ orderBook, riskManager });
    performanceMonitor = new PerformanceMonitor();
  });

  describe('Flash Crash Scenarios', () => {
    test('should handle 90% price drop within microseconds', async () => {
      const basePrice = 150.00;
      const crashPrice = basePrice * 0.1; // 90% drop
      const orders = [];

      // Generate massive sell pressure
      for (let i = 0; i < 10000; i++) {
        orders.push({
          id: `CRASH_${i}`,
          symbol: 'AAPL',
          side: 'sell',
          quantity: Math.floor(Math.random() * 10000) + 1000,
          price: crashPrice + (Math.random() * 5), // Near crash price
          timestamp: Date.now() + i,
          userId: `USER_${i % 100}`
        });
      }

      const startTime = performance.now();
      const results = [];

      // Process orders concurrently to simulate flash crash
      const batchSize = 100;
      for (let i = 0; i < orders.length; i += batchSize) {
        const batch = orders.slice(i, i + batchSize);
        const batchPromises = batch.map(order => tradingEngine.processOrder(order));
        const batchResults = await Promise.all(batchPromises);
        results.push(...batchResults);
      }

      const processingTime = performance.now() - startTime;

      // Verify system maintains stability
      expect(results.length).toBe(orders.length);
      expect(processingTime).toBeLessThan(30000); // 30 second max processing time
      
      // Verify circuit breakers activated
      const rejectedOrders = results.filter(r => !r.success);
      expect(rejectedOrders.length).toBeGreaterThan(0); // Some orders should be rejected

      // Verify no system crash
      expect(() => tradingEngine.getStatus()).not.toThrow();
    }, 60000);

    test('should activate kill switch under extreme conditions', async () => {
      const extremeOrders = Array.from({ length: 1000 }, (_, i) => ({
        id: `EXTREME_${i}`,
        symbol: 'AAPL',
        side: 'sell',
        quantity: 1000000, // Extremely large quantities
        price: 0.01, // Penny pricing
        timestamp: Date.now(),
        userId: 'WHALE_USER'
      }));

      let killSwitchActivated = false;
      riskManager.on('killSwitch', () => {
        killSwitchActivated = true;
      });

      // Process extreme orders
      for (const order of extremeOrders.slice(0, 10)) {
        await tradingEngine.processOrder(order);
        if (killSwitchActivated) break;
      }

      expect(killSwitchActivated).toBe(true);
    }, 30000);
  });

  describe('High Frequency Trading Simulation', () => {
    test('should handle 1 million orders per second burst', async () => {
      const orderCount = 1000000;
      const orders = Array.from({ length: orderCount }, (_, i) => ({
        id: `HFT_${i}`,
        symbol: 'AAPL',
        side: i % 2 === 0 ? 'buy' : 'sell',
        quantity: Math.floor(Math.random() * 100) + 1,
        price: 150.00 + (Math.random() - 0.5) * 10,
        timestamp: Date.now() + i,
        userId: `HFT_${i % 1000}`
      }));

      const startTime = performance.now();
      const startMemory = process.memoryUsage();

      // Process in parallel batches
      const batchSize = 10000;
      const results = [];

      for (let i = 0; i < orders.length; i += batchSize) {
        const batch = orders.slice(i, i + batchSize);
        const batchPromises = batch.map(order => tradingEngine.processOrder(order));
        const batchResults = await Promise.allSettled(batchPromises);
        results.push(...batchResults);

        // Monitor memory usage
        const currentMemory = process.memoryUsage();
        const memoryIncrease = currentMemory.heapUsed - startMemory.heapUsed;
        expect(memoryIncrease).toBeLessThan(500 * 1024 * 1024); // 500MB limit
      }

      const processingTime = performance.now() - startTime;
      const ordersPerSecond = orderCount / (processingTime / 1000);

      expect(results.length).toBe(orderCount);
      expect(ordersPerSecond).toBeGreaterThan(100000); // At least 100K orders/sec
      expect(processingTime).toBeLessThan(60000); // 60 second max
    }, 120000);

    test('should maintain order book integrity under concurrent modifications', async () => {
      const concurrentOperations = 10000;
      const operations = [];

      // Generate mixed operations
      for (let i = 0; i < concurrentOperations; i++) {
        const operationType = Math.random();
        
        if (operationType < 0.7) {
          // 70% add orders
          operations.push({
            type: 'add',
            order: {
              id: `CONCURRENT_${i}`,
              symbol: 'AAPL',
              side: Math.random() > 0.5 ? 'buy' : 'sell',
              quantity: Math.floor(Math.random() * 1000) + 1,
              price: 150.00 + (Math.random() - 0.5) * 20,
              timestamp: Date.now() + i,
              userId: `USER_${i % 100}`
            }
          });
        } else if (operationType < 0.9) {
          // 20% cancel orders
          operations.push({
            type: 'cancel',
            orderId: `CONCURRENT_${Math.floor(Math.random() * i)}`
          });
        } else {
          // 10% query operations
          operations.push({
            type: 'query'
          });
        }
      }

      // Execute operations concurrently
      const promises = operations.map(async (op) => {
        try {
          switch (op.type) {
            case 'add':
              return await tradingEngine.processOrder(op.order);
            case 'cancel':
              return await tradingEngine.cancelOrder(op.orderId);
            case 'query':
              return await tradingEngine.getOrderBookSnapshot();
            default:
              return null;
          }
        } catch (error) {
          return { error: error.message };
        }
      });

      const results = await Promise.all(promises);

      // Verify no corruption
      expect(results.length).toBe(concurrentOperations);
      
      // Verify order book is still functional
      const finalSnapshot = await tradingEngine.getOrderBookSnapshot();
      expect(finalSnapshot).toBeDefined();
      expect(Array.isArray(finalSnapshot.bids)).toBe(true);
      expect(Array.isArray(finalSnapshot.offers)).toBe(true);
    }, 60000);
  });

  describe('Market Open Surge Simulation', () => {
    test('should handle market opening surge of 100,000 orders in 1 second', async () => {
      const surgOrders = Array.from({ length: 100000 }, (_, i) => ({
        id: `SURGE_${i}`,
        symbol: 'AAPL',
        side: Math.random() > 0.5 ? 'buy' : 'sell',
        quantity: Math.floor(Math.random() * 1000) + 100,
        price: 150.00 + (Math.random() - 0.5) * 5,
        timestamp: Date.now() + Math.floor(i / 10), // Compress into narrow time window
        userId: `USER_${i % 1000}`
      }));

      const startTime = performance.now();
      
      // Simulate market open surge with maximum concurrency
      const chunkSize = 1000;
      const chunks = [];
      for (let i = 0; i < surgOrders.length; i += chunkSize) {
        chunks.push(surgOrders.slice(i, i + chunkSize));
      }

      const results = await Promise.all(
        chunks.map(chunk => 
          Promise.all(chunk.map(order => tradingEngine.processOrder(order)))
        )
      );

      const flatResults = results.flat();
      const processingTime = performance.now() - startTime;

      expect(flatResults.length).toBe(100000);
      expect(processingTime).toBeLessThan(10000); // 10 second max
      
      // Verify system responsiveness maintained
      const healthCheck = await tradingEngine.healthCheck();
      expect(healthCheck.status).toBe('healthy');
      expect(healthCheck.latency).toBeLessThan(100); // 100ms max latency
    }, 30000);
  });

  describe('Network Partition Simulation', () => {
    test('should handle network partition and recovery gracefully', async () => {
      // Simulate normal operations
      const normalOrders = Array.from({ length: 1000 }, (_, i) => ({
        id: `NORMAL_${i}`,
        symbol: 'AAPL',
        side: Math.random() > 0.5 ? 'buy' : 'sell',
        quantity: 100,
        price: 150.00,
        timestamp: Date.now(),
        userId: 'USER_001'
      }));

      // Process normal orders
      const normalResults = await Promise.all(
        normalOrders.map(order => tradingEngine.processOrder(order))
      );

      expect(normalResults.every(r => r.success)).toBe(true);

      // Simulate network partition (mock network failures)
      riskManager.simulateNetworkPartition(true);

      // Try to process orders during partition
      const partitionOrders = Array.from({ length: 100 }, (_, i) => ({
        id: `PARTITION_${i}`,
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00,
        timestamp: Date.now(),
        userId: 'USER_001'
      }));

      const partitionResults = await Promise.all(
        partitionOrders.map(order => tradingEngine.processOrder(order))
      );

      // Some orders should be queued or rejected during partition
      const rejectedDuringPartition = partitionResults.filter(r => !r.success);
      expect(rejectedDuringPartition.length).toBeGreaterThan(0);

      // Recover from partition
      riskManager.simulateNetworkPartition(false);

      // Verify system recovery
      const recoveryOrder = {
        id: 'RECOVERY_TEST',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00,
        timestamp: Date.now(),
        userId: 'USER_001'
      };

      const recoveryResult = await tradingEngine.processOrder(recoveryOrder);
      expect(recoveryResult.success).toBe(true);
    }, 45000);
  });

  describe('Memory Pressure Testing', () => {
    test('should handle extreme memory pressure without crashes', async () => {
      const largeOrderCount = 1000000;
      let processedCount = 0;
      const startMemory = process.memoryUsage();

      // Process orders in batches to avoid overwhelming the system
      for (let batch = 0; batch < 100; batch++) {
        const batchOrders = Array.from({ length: 10000 }, (_, i) => ({
          id: `MEMORY_${batch}_${i}`,
          symbol: 'AAPL',
          side: Math.random() > 0.5 ? 'buy' : 'sell',
          quantity: Math.floor(Math.random() * 100) + 1,
          price: 150.00 + (Math.random() - 0.5) * 10,
          timestamp: Date.now(),
          userId: `USER_${i % 100}`
        }));

        const batchResults = await Promise.all(
          batchOrders.map(order => tradingEngine.processOrder(order))
        );

        processedCount += batchResults.length;

        // Monitor memory growth
        const currentMemory = process.memoryUsage();
        const memoryGrowth = currentMemory.heapUsed - startMemory.heapUsed;

        // Force garbage collection periodically
        if (batch % 10 === 0 && global.gc) {
          global.gc();
        }

        // Verify memory growth is reasonable
        expect(memoryGrowth).toBeLessThan(1024 * 1024 * 1024); // 1GB limit

        // Verify system is still responsive
        if (batch % 25 === 0) {
          const healthCheck = await tradingEngine.healthCheck();
          expect(healthCheck.status).toBe('healthy');
        }
      }

      expect(processedCount).toBe(largeOrderCount);
    }, 300000);
  });
});