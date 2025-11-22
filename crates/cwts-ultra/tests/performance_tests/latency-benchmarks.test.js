const { TradingEngine } = require('../../quantum_trading/core/trading_engine');
const { OrderBook } = require('../../quantum_trading/core/order_book');
const { RiskManager } = require('../../quantum_trading/core/risk_manager');
const { PerformanceProfiler } = require('../utils/performance_profiler');

describe('Performance Regression Testing', () => {
  let tradingEngine;
  let orderBook;
  let riskManager;
  let profiler;

  beforeEach(async () => {
    orderBook = new OrderBook();
    riskManager = new RiskManager();
    tradingEngine = new TradingEngine({ orderBook, riskManager });
    profiler = new PerformanceProfiler();
    
    await tradingEngine.initialize();
  });

  afterEach(async () => {
    await tradingEngine.shutdown();
  });

  describe('Order Processing Latency', () => {
    test('should process single order within 1ms (99th percentile)', async () => {
      const userId = 'LATENCY_USER_001';
      await riskManager.setCreditLimit(userId, 1000000);

      const order = {
        id: 'LATENCY_TEST_001',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00,
        type: 'limit',
        timestamp: Date.now(),
        userId
      };

      // Warmup
      for (let i = 0; i < 100; i++) {
        await tradingEngine.processOrder({
          ...order,
          id: `WARMUP_${i}`,
          timestamp: Date.now()
        });
      }

      // Measure latency for 1000 orders
      const latencies = [];
      for (let i = 0; i < 1000; i++) {
        const testOrder = {
          ...order,
          id: `LATENCY_${i}`,
          timestamp: Date.now()
        };

        const start = process.hrtime.bigint();
        await tradingEngine.processOrder(testOrder);
        const end = process.hrtime.bigint();
        
        const latencyNs = Number(end - start);
        const latencyMs = latencyNs / 1_000_000;
        latencies.push(latencyMs);
      }

      // Calculate percentiles
      latencies.sort((a, b) => a - b);
      const p50 = latencies[Math.floor(latencies.length * 0.50)];
      const p95 = latencies[Math.floor(latencies.length * 0.95)];
      const p99 = latencies[Math.floor(latencies.length * 0.99)];
      const p999 = latencies[Math.floor(latencies.length * 0.999)];

      console.log(`Latency Percentiles (ms):`);
      console.log(`P50: ${p50.toFixed(3)}`);
      console.log(`P95: ${p95.toFixed(3)}`);
      console.log(`P99: ${p99.toFixed(3)}`);
      console.log(`P99.9: ${p999.toFixed(3)}`);

      // Strict latency requirements for financial trading
      expect(p50).toBeLessThan(0.5);  // 500μs median
      expect(p95).toBeLessThan(1.0);  // 1ms 95th percentile
      expect(p99).toBeLessThan(2.0);  // 2ms 99th percentile
      expect(p999).toBeLessThan(5.0); // 5ms 99.9th percentile
    });

    test('should maintain latency under concurrent load', async () => {
      const numUsers = 100;
      const ordersPerUser = 10;
      
      // Setup users
      const users = Array.from({ length: numUsers }, (_, i) => `CONCURRENT_USER_${i}`);
      for (const userId of users) {
        await riskManager.setCreditLimit(userId, 1000000);
      }

      // Generate concurrent orders
      const orders = [];
      for (let i = 0; i < numUsers; i++) {
        for (let j = 0; j < ordersPerUser; j++) {
          orders.push({
            id: `CONCURRENT_${i}_${j}`,
            symbol: 'AAPL',
            side: Math.random() > 0.5 ? 'buy' : 'sell',
            quantity: 100,
            price: 150.00 + (Math.random() - 0.5) * 2,
            type: 'limit',
            timestamp: Date.now(),
            userId: users[i]
          });
        }
      }

      // Shuffle orders for realistic concurrent pattern
      for (let i = orders.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [orders[i], orders[j]] = [orders[j], orders[i]];
      }

      const startTime = process.hrtime.bigint();
      
      // Process orders in batches to simulate concurrency
      const batchSize = 50;
      const latencies = [];
      
      for (let i = 0; i < orders.length; i += batchSize) {
        const batch = orders.slice(i, i + batchSize);
        
        const batchPromises = batch.map(async (order) => {
          const orderStart = process.hrtime.bigint();
          const result = await tradingEngine.processOrder(order);
          const orderEnd = process.hrtime.bigint();
          
          const latencyMs = Number(orderEnd - orderStart) / 1_000_000;
          latencies.push(latencyMs);
          
          return result;
        });

        await Promise.all(batchPromises);
      }

      const totalTime = Number(process.hrtime.bigint() - startTime) / 1_000_000;
      const throughput = orders.length / (totalTime / 1000);

      // Calculate latency statistics under load
      latencies.sort((a, b) => a - b);
      const p99 = latencies[Math.floor(latencies.length * 0.99)];

      console.log(`Concurrent Load Performance:`);
      console.log(`Total Orders: ${orders.length}`);
      console.log(`Total Time: ${totalTime.toFixed(2)}ms`);
      console.log(`Throughput: ${throughput.toFixed(0)} orders/sec`);
      console.log(`P99 Latency: ${p99.toFixed(3)}ms`);

      // Performance requirements under load
      expect(throughput).toBeGreaterThan(50000); // >50K orders/sec
      expect(p99).toBeLessThan(10.0); // P99 < 10ms under load
    });
  });

  describe('Memory Performance', () => {
    test('should not leak memory during sustained operation', async () => {
      const userId = 'MEMORY_TEST_USER';
      await riskManager.setCreditLimit(userId, 100000000);

      const initialMemory = process.memoryUsage();
      console.log(`Initial Memory: ${JSON.stringify(initialMemory)}`);

      // Process many orders to stress memory
      for (let batch = 0; batch < 50; batch++) {
        const orders = Array.from({ length: 1000 }, (_, i) => ({
          id: `MEMORY_${batch}_${i}`,
          symbol: 'AAPL',
          side: Math.random() > 0.5 ? 'buy' : 'sell',
          quantity: Math.floor(Math.random() * 1000) + 100,
          price: 150.00 + (Math.random() - 0.5) * 10,
          type: 'limit',
          timestamp: Date.now() + i,
          userId
        }));

        // Process batch
        const promises = orders.map(order => tradingEngine.processOrder(order));
        await Promise.all(promises);

        // Force garbage collection every 10 batches
        if (batch % 10 === 0 && global.gc) {
          global.gc();
          
          const currentMemory = process.memoryUsage();
          const memoryIncrease = currentMemory.heapUsed - initialMemory.heapUsed;
          
          console.log(`Batch ${batch}: Memory increase: ${memoryIncrease / 1024 / 1024}MB`);
          
          // Memory increase should be reasonable (<500MB)
          expect(memoryIncrease).toBeLessThan(500 * 1024 * 1024);
        }
      }

      // Final memory check
      if (global.gc) global.gc();
      const finalMemory = process.memoryUsage();
      const totalIncrease = finalMemory.heapUsed - initialMemory.heapUsed;
      
      console.log(`Final Memory Increase: ${totalIncrease / 1024 / 1024}MB`);
      
      // Should not leak more than 100MB after processing 50K orders
      expect(totalIncrease).toBeLessThan(100 * 1024 * 1024);
    });

    test('should maintain performance with large order book', async () => {
      const userId = 'LARGE_BOOK_USER';
      await riskManager.setCreditLimit(userId, 100000000);

      // Build large order book
      console.log('Building large order book...');
      const bookSizes = [1000, 5000, 10000, 25000, 50000];
      const latencyResults = {};

      for (const size of bookSizes) {
        // Clear order book
        await orderBook.clear();

        // Add orders to build book
        for (let i = 0; i < size; i++) {
          const order = {
            id: `BOOK_BUILD_${i}`,
            symbol: 'AAPL',
            side: i % 2 === 0 ? 'buy' : 'sell',
            quantity: 100,
            price: 150.00 + (Math.random() - 0.5) * 20,
            type: 'limit',
            timestamp: Date.now() + i,
            userId
          };
          
          await tradingEngine.processOrder(order);
        }

        // Measure order processing latency with large book
        const testOrder = {
          id: `LARGE_BOOK_TEST_${size}`,
          symbol: 'AAPL',
          side: 'buy',
          quantity: 100,
          price: 150.00,
          type: 'limit',
          timestamp: Date.now(),
          userId
        };

        const latencies = [];
        for (let i = 0; i < 100; i++) {
          const start = process.hrtime.bigint();
          await tradingEngine.processOrder({
            ...testOrder,
            id: `${testOrder.id}_${i}`,
            timestamp: Date.now()
          });
          const end = process.hrtime.bigint();
          
          latencies.push(Number(end - start) / 1_000_000);
        }

        latencies.sort((a, b) => a - b);
        const p99 = latencies[Math.floor(latencies.length * 0.99)];
        latencyResults[size] = p99;

        console.log(`Book size ${size}: P99 latency ${p99.toFixed(3)}ms`);
        
        // Latency should not degrade significantly with book size
        expect(p99).toBeLessThan(5.0); // P99 < 5ms even with large book
      }

      // Verify latency doesn't increase exponentially with book size
      const maxLatency = Math.max(...Object.values(latencyResults));
      const minLatency = Math.min(...Object.values(latencyResults));
      const latencyRatio = maxLatency / minLatency;
      
      expect(latencyRatio).toBeLessThan(10); // Max 10x latency increase
    });
  });

  describe('Throughput Performance', () => {
    test('should handle high frequency trading workload', async () => {
      const numUsers = 50;
      const ordersPerSecond = 100000;
      const testDurationSeconds = 10;
      
      // Setup HFT users
      const users = Array.from({ length: numUsers }, (_, i) => `HFT_USER_${i}`);
      for (const userId of users) {
        await riskManager.setCreditLimit(userId, 10000000);
      }

      console.log(`Starting HFT simulation: ${ordersPerSecond} orders/sec for ${testDurationSeconds}s`);

      const startTime = Date.now();
      const endTime = startTime + (testDurationSeconds * 1000);
      let orderCount = 0;
      const results = [];

      // Generate orders at target rate
      while (Date.now() < endTime) {
        const batchStart = Date.now();
        const batchSize = Math.floor(ordersPerSecond / 10); // 100ms batches
        
        const batch = Array.from({ length: batchSize }, (_, i) => ({
          id: `HFT_${orderCount + i}`,
          symbol: 'AAPL',
          side: Math.random() > 0.5 ? 'buy' : 'sell',
          quantity: Math.floor(Math.random() * 1000) + 100,
          price: 150.00 + (Math.random() - 0.5) * 1,
          type: 'limit',
          timestamp: Date.now() + i,
          userId: users[i % numUsers]
        }));

        const batchPromises = batch.map(order => 
          tradingEngine.processOrder(order).catch(err => ({ success: false, error: err.message }))
        );
        
        const batchResults = await Promise.all(batchPromises);
        results.push(...batchResults);
        orderCount += batchSize;

        // Maintain target rate
        const batchDuration = Date.now() - batchStart;
        const targetBatchDuration = 100; // 100ms
        if (batchDuration < targetBatchDuration) {
          await new Promise(resolve => setTimeout(resolve, targetBatchDuration - batchDuration));
        }
      }

      const actualDuration = (Date.now() - startTime) / 1000;
      const actualThroughput = orderCount / actualDuration;
      const successRate = results.filter(r => r.success).length / results.length;

      console.log(`HFT Results:`);
      console.log(`Orders processed: ${orderCount}`);
      console.log(`Actual duration: ${actualDuration.toFixed(2)}s`);
      console.log(`Actual throughput: ${actualThroughput.toFixed(0)} orders/sec`);
      console.log(`Success rate: ${(successRate * 100).toFixed(2)}%`);

      // Performance requirements for HFT
      expect(actualThroughput).toBeGreaterThan(ordersPerSecond * 0.8); // 80% of target
      expect(successRate).toBeGreaterThan(0.95); // 95% success rate
    });

    test('should scale with multiple symbols', async () => {
      const symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX'];
      const userId = 'MULTI_SYMBOL_USER';
      await riskManager.setCreditLimit(userId, 50000000);

      const results = {};

      for (let symbolCount = 1; symbolCount <= symbols.length; symbolCount++) {
        const activeSymbols = symbols.slice(0, symbolCount);
        
        console.log(`Testing with ${symbolCount} symbols: ${activeSymbols.join(', ')}`);

        const orders = Array.from({ length: 1000 }, (_, i) => ({
          id: `MULTI_${symbolCount}_${i}`,
          symbol: activeSymbols[i % symbolCount],
          side: Math.random() > 0.5 ? 'buy' : 'sell',
          quantity: 100,
          price: 150.00 + (Math.random() - 0.5) * 10,
          type: 'limit',
          timestamp: Date.now() + i,
          userId
        }));

        const start = performance.now();
        const orderResults = await Promise.all(
          orders.map(order => tradingEngine.processOrder(order))
        );
        const duration = performance.now() - start;

        const throughput = orders.length / (duration / 1000);
        const successRate = orderResults.filter(r => r.success).length / orderResults.length;

        results[symbolCount] = { throughput, successRate, duration };

        console.log(`${symbolCount} symbols: ${throughput.toFixed(0)} orders/sec, ${(successRate * 100).toFixed(1)}% success`);

        // Performance should not degrade significantly with more symbols
        expect(throughput).toBeGreaterThan(5000); // Min 5K orders/sec
        expect(successRate).toBeGreaterThan(0.95); // 95% success rate
      }

      // Verify scaling characteristics
      const singleSymbolThroughput = results[1].throughput;
      const multiSymbolThroughput = results[symbols.length].throughput;
      const scalingRatio = singleSymbolThroughput / multiSymbolThroughput;
      
      expect(scalingRatio).toBeLessThan(2); // Less than 2x throughput degradation
    });
  });

  describe('CPU and Resource Utilization', () => {
    test('should maintain efficient CPU utilization', async () => {
      const userId = 'CPU_TEST_USER';
      await riskManager.setCreditLimit(userId, 10000000);

      // Start CPU monitoring
      const cpuMonitor = profiler.startCpuMonitoring();

      // Generate sustained load
      const orders = Array.from({ length: 10000 }, (_, i) => ({
        id: `CPU_ORDER_${i}`,
        symbol: 'AAPL',
        side: i % 2 === 0 ? 'buy' : 'sell',
        quantity: 100,
        price: 150.00 + (Math.random() - 0.5) * 5,
        type: 'limit',
        timestamp: Date.now() + i,
        userId
      }));

      const start = performance.now();
      
      // Process in batches to monitor CPU
      for (let i = 0; i < orders.length; i += 500) {
        const batch = orders.slice(i, i + 500);
        await Promise.all(batch.map(order => tradingEngine.processOrder(order)));
      }

      const duration = performance.now() - start;
      const cpuStats = profiler.stopCpuMonitoring(cpuMonitor);

      console.log(`CPU Utilization Stats:`);
      console.log(`Average CPU: ${cpuStats.averageCpuUsage.toFixed(2)}%`);
      console.log(`Peak CPU: ${cpuStats.peakCpuUsage.toFixed(2)}%`);
      console.log(`Orders per CPU second: ${(orders.length / (cpuStats.averageCpuUsage / 100)).toFixed(0)}`);

      // CPU utilization should be efficient
      expect(cpuStats.averageCpuUsage).toBeLessThan(80); // <80% average CPU
      expect(cpuStats.peakCpuUsage).toBeLessThan(95); // <95% peak CPU

      // Should process orders efficiently per CPU cycle
      const ordersPerCpuSecond = orders.length / (duration / 1000) / (cpuStats.averageCpuUsage / 100);
      expect(ordersPerCpuSecond).toBeGreaterThan(1000); // >1K orders per CPU second
    });
  });
});