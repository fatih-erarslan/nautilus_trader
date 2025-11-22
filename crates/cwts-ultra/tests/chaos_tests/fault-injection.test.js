const { TradingEngine } = require('../../quantum_trading/core/trading_engine');
const { OrderBook } = require('../../quantum_trading/core/order_book');
const { RiskManager } = require('../../quantum_trading/core/risk_manager');
const { FaultInjector } = require('../utils/fault_injector');
const { SystemMonitor } = require('../../quantum_trading/utils/system_monitor');

describe('Chaos Engineering - Fault Injection Testing', () => {
  let tradingEngine;
  let orderBook;
  let riskManager;
  let faultInjector;
  let systemMonitor;

  beforeEach(() => {
    orderBook = new OrderBook();
    riskManager = new RiskManager();
    tradingEngine = new TradingEngine({ orderBook, riskManager });
    faultInjector = new FaultInjector();
    systemMonitor = new SystemMonitor();
  });

  afterEach(async () => {
    await faultInjector.clearAllFaults();
    await systemMonitor.reset();
  });

  describe('Database Failure Scenarios', () => {
    test('should handle database connection loss during order processing', async () => {
      const orders = Array.from({ length: 100 }, (_, i) => ({
        id: `DB_FAULT_${i}`,
        symbol: 'AAPL',
        side: i % 2 === 0 ? 'buy' : 'sell',
        quantity: 100,
        price: 150.00,
        timestamp: Date.now(),
        userId: 'USER_001'
      }));

      // Process some orders normally
      const normalResults = [];
      for (let i = 0; i < 30; i++) {
        const result = await tradingEngine.processOrder(orders[i]);
        normalResults.push(result);
      }

      expect(normalResults.every(r => r.success)).toBe(true);

      // Inject database connection failure
      await faultInjector.injectDatabaseFault({
        type: 'CONNECTION_LOSS',
        duration: 5000, // 5 seconds
        recovery: 'automatic'
      });

      // Continue processing orders during fault
      const faultResults = [];
      for (let i = 30; i < 70; i++) {
        const result = await tradingEngine.processOrder(orders[i]);
        faultResults.push(result);
      }

      // Some orders should be queued or handled gracefully
      const gracefulHandling = faultResults.filter(r => 
        r.success || r.error.includes('queued') || r.error.includes('retry')
      );
      expect(gracefulHandling.length).toBeGreaterThan(faultResults.length * 0.8);

      // Wait for automatic recovery
      await new Promise(resolve => setTimeout(resolve, 6000));

      // Process remaining orders after recovery
      const recoveryResults = [];
      for (let i = 70; i < 100; i++) {
        const result = await tradingEngine.processOrder(orders[i]);
        recoveryResults.push(result);
      }

      expect(recoveryResults.every(r => r.success)).toBe(true);

      // Verify data consistency after recovery
      const systemState = await tradingEngine.getSystemState();
      expect(systemState.isConsistent).toBe(true);
      expect(systemState.lostOrders).toBe(0);
    });

    test('should handle database corruption and recovery', async () => {
      // Create initial state
      const setupOrders = Array.from({ length: 50 }, (_, i) => ({
        id: `SETUP_${i}`,
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00,
        timestamp: Date.now(),
        userId: 'USER_001'
      }));

      for (const order of setupOrders) {
        await tradingEngine.processOrder(order);
      }

      const preCorruptionState = await tradingEngine.getSystemState();

      // Inject data corruption
      await faultInjector.injectDatabaseFault({
        type: 'DATA_CORRUPTION',
        scope: 'partial',
        affectedTables: ['orders', 'positions']
      });

      // Attempt operations during corruption
      const corruptionOrder = {
        id: 'CORRUPTION_TEST',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00,
        timestamp: Date.now(),
        userId: 'USER_001'
      };

      const corruptionResult = await tradingEngine.processOrder(corruptionOrder);
      
      // System should detect corruption and enter safe mode
      expect(corruptionResult.success).toBe(false);
      expect(corruptionResult.error).toContain('data integrity');

      // Trigger recovery process
      const recoveryResult = await tradingEngine.recoverFromCorruption();
      expect(recoveryResult.success).toBe(true);
      expect(recoveryResult.restoredFromBackup).toBe(true);

      // Verify system is operational after recovery
      const postRecoveryOrder = {
        id: 'POST_RECOVERY_TEST',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00,
        timestamp: Date.now(),
        userId: 'USER_001'
      };

      const postRecoveryResult = await tradingEngine.processOrder(postRecoveryOrder);
      expect(postRecoveryResult.success).toBe(true);
    });
  });

  describe('Network Failure Scenarios', () => {
    test('should handle intermittent network failures gracefully', async () => {
      // Inject intermittent network faults
      await faultInjector.injectNetworkFault({
        type: 'INTERMITTENT_LOSS',
        probability: 0.3, // 30% packet loss
        duration: 10000,
        pattern: 'random'
      });

      const orders = Array.from({ length: 200 }, (_, i) => ({
        id: `NET_FAULT_${i}`,
        symbol: 'AAPL',
        side: i % 2 === 0 ? 'buy' : 'sell',
        quantity: 100,
        price: 150.00 + (Math.random() - 0.5),
        timestamp: Date.now() + i,
        userId: `USER_${i % 10}`
      }));

      const results = [];
      const retryResults = [];

      // Process orders with automatic retry logic
      for (const order of orders) {
        let result = await tradingEngine.processOrder(order);
        results.push(result);

        // If failed due to network, retry once
        if (!result.success && result.error.includes('network')) {
          await new Promise(resolve => setTimeout(resolve, 100));
          const retryResult = await tradingEngine.processOrder(order);
          retryResults.push(retryResult);
        }
      }

      // Analyze success rates
      const initialSuccessRate = results.filter(r => r.success).length / results.length;
      const retrySuccessRate = retryResults.filter(r => r.success).length / retryResults.length;

      expect(initialSuccessRate).toBeGreaterThan(0.6); // At least 60% success despite faults
      expect(retrySuccessRate).toBeGreaterThan(0.8); // Retries should improve success rate

      // Verify no order duplication occurred
      const allOrderIds = [...results, ...retryResults]
        .filter(r => r.success)
        .map(r => r.orderId);
      const uniqueOrderIds = new Set(allOrderIds);
      expect(uniqueOrderIds.size).toBe(allOrderIds.length);
    });

    test('should handle complete network partition between components', async () => {
      // Create orders before partition
      const prePartitionOrders = Array.from({ length: 50 }, (_, i) => ({
        id: `PRE_PARTITION_${i}`,
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00,
        timestamp: Date.now(),
        userId: 'USER_001'
      }));

      for (const order of prePartitionOrders) {
        const result = await tradingEngine.processOrder(order);
        expect(result.success).toBe(true);
      }

      // Inject complete network partition
      await faultInjector.injectNetworkFault({
        type: 'COMPLETE_PARTITION',
        components: ['trading_engine', 'risk_manager'],
        duration: 15000
      });

      // Attempt operations during partition
      const partitionOrders = Array.from({ length: 30 }, (_, i) => ({
        id: `PARTITION_${i}`,
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00,
        timestamp: Date.now(),
        userId: 'USER_001'
      }));

      const partitionResults = [];
      for (const order of partitionOrders) {
        const result = await tradingEngine.processOrder(order);
        partitionResults.push(result);
      }

      // Orders should be queued or rejected safely
      const safelyHandled = partitionResults.filter(r => 
        !r.success && (r.error.includes('queued') || r.error.includes('partition'))
      );
      expect(safelyHandled.length).toBe(partitionResults.length);

      // Wait for partition recovery
      await new Promise(resolve => setTimeout(resolve, 16000));

      // Process queued orders after recovery
      const recoveryResults = await tradingEngine.processQueuedOrders();
      expect(recoveryResults.processed).toBeGreaterThan(0);
      expect(recoveryResults.failures).toBe(0);

      // Verify system consistency after partition recovery
      const finalState = await tradingEngine.getSystemState();
      expect(finalState.isConsistent).toBe(true);
      expect(finalState.partitionRecoveryComplete).toBe(true);
    });
  });

  describe('Memory and Resource Exhaustion', () => {
    test('should handle memory exhaustion gracefully', async () => {
      // Monitor initial memory state
      const initialMemory = process.memoryUsage();

      // Inject memory pressure
      await faultInjector.injectResourceFault({
        type: 'MEMORY_EXHAUSTION',
        targetUsage: 0.95, // 95% memory usage
        gradual: true,
        duration: 30000
      });

      const orders = Array.from({ length: 10000 }, (_, i) => ({
        id: `MEM_FAULT_${i}`,
        symbol: 'AAPL',
        side: i % 2 === 0 ? 'buy' : 'sell',
        quantity: 100,
        price: 150.00,
        timestamp: Date.now(),
        userId: `USER_${i % 100}`
      }));

      let processedCount = 0;
      let rejectedCount = 0;
      let memoryErrors = 0;

      // Process orders under memory pressure
      for (let i = 0; i < orders.length; i += 100) {
        const batch = orders.slice(i, i + 100);
        
        try {
          const batchResults = await Promise.all(
            batch.map(order => tradingEngine.processOrder(order))
          );

          for (const result of batchResults) {
            if (result.success) {
              processedCount++;
            } else {
              rejectedCount++;
              if (result.error.includes('memory')) {
                memoryErrors++;
              }
            }
          }

          // Monitor memory usage
          const currentMemory = process.memoryUsage();
          if (currentMemory.heapUsed > initialMemory.heapUsed * 10) {
            // System should start rejecting orders to prevent crash
            expect(rejectedCount).toBeGreaterThan(0);
          }

        } catch (error) {
          if (error.message.includes('memory')) {
            memoryErrors++;
          }
        }

        // Force garbage collection if available
        if (global.gc && i % 500 === 0) {
          global.gc();
        }
      }

      // Verify system remained stable
      expect(memoryErrors).toBeGreaterThan(0); // Should detect memory issues
      expect(processedCount + rejectedCount).toBe(orders.length);

      // Verify system can recover after memory pressure is released
      await faultInjector.clearResourceFaults();
      await new Promise(resolve => setTimeout(resolve, 5000));

      if (global.gc) global.gc();

      const recoveryOrder = {
        id: 'MEMORY_RECOVERY_TEST',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00,
        timestamp: Date.now(),
        userId: 'USER_001'
      };

      const recoveryResult = await tradingEngine.processOrder(recoveryOrder);
      expect(recoveryResult.success).toBe(true);
    });

    test('should handle CPU exhaustion and maintain critical operations', async () => {
      // Inject CPU exhaustion
      await faultInjector.injectResourceFault({
        type: 'CPU_EXHAUSTION',
        targetUsage: 0.98, // 98% CPU usage
        duration: 20000
      });

      // Test critical operations under CPU pressure
      const criticalOperations = [
        () => riskManager.activateKillSwitch('TEST_USER', 'CPU_STRESS_TEST'),
        () => tradingEngine.getSystemHealth(),
        () => riskManager.checkCriticalRiskLimits(),
        () => tradingEngine.emergencyShutdown()
      ];

      const results = [];
      for (const operation of criticalOperations) {
        const startTime = performance.now();
        try {
          const result = await operation();
          const duration = performance.now() - startTime;
          results.push({ success: true, duration, result });
        } catch (error) {
          const duration = performance.now() - startTime;
          results.push({ success: false, duration, error: error.message });
        }
      }

      // Critical operations should still complete within reasonable time
      results.forEach((result, index) => {
        expect(result.duration).toBeLessThan(5000); // 5 second max for critical ops
        if (index === 0) { // Kill switch must always work
          expect(result.success).toBe(true);
        }
      });

      // Clear CPU fault
      await faultInjector.clearResourceFaults();
    });
  });

  describe('Byzantine Failure Scenarios', () => {
    test('should handle Byzantine failures in distributed components', async () => {
      // Simulate Byzantine behavior in risk manager
      await faultInjector.injectByzantineFault({
        component: 'risk_manager',
        behavior: 'MALICIOUS_RESPONSES',
        probability: 0.3,
        responses: {
          'creditCheck': { result: false, reason: 'MALICIOUS_REJECTION' },
          'positionCheck': { result: true, actualResult: false }
        }
      });

      const orders = Array.from({ length: 100 }, (_, i) => ({
        id: `BYZANTINE_${i}`,
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00,
        timestamp: Date.now(),
        userId: 'USER_001'
      }));

      const results = [];
      const detectedByzantine = [];

      for (const order of orders) {
        const result = await tradingEngine.processOrder(order);
        results.push(result);

        // Check for Byzantine detection
        const byzantineDetection = await systemMonitor.checkByzantineDetection();
        if (byzantineDetection.detected) {
          detectedByzantine.push(byzantineDetection);
        }
      }

      // System should detect Byzantine behavior
      expect(detectedByzantine.length).toBeGreaterThan(0);
      expect(detectedByzantine[0]).toMatchObject({
        detected: true,
        component: 'risk_manager',
        inconsistencies: expect.any(Number),
        confidence: expect.any(Number)
      });

      // System should take defensive action
      const systemStatus = await tradingEngine.getSystemStatus();
      expect(systemStatus.byzantineDefenseActive).toBe(true);

      // Clear Byzantine fault and verify recovery
      await faultInjector.clearByzantineFaults();
      
      const recoveryOrder = {
        id: 'BYZANTINE_RECOVERY_TEST',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00,
        timestamp: Date.now(),
        userId: 'USER_001'
      };

      const recoveryResult = await tradingEngine.processOrder(recoveryOrder);
      expect(recoveryResult.success).toBe(true);
    });
  });

  describe('Cascade Failure Prevention', () => {
    test('should prevent cascade failures across system components', async () => {
      // Inject multiple simultaneous faults
      await Promise.all([
        faultInjector.injectDatabaseFault({
          type: 'SLOW_QUERIES',
          delay: 2000,
          probability: 0.5
        }),
        faultInjector.injectNetworkFault({
          type: 'HIGH_LATENCY',
          delay: 1000,
          jitter: 500
        }),
        faultInjector.injectResourceFault({
          type: 'MEMORY_PRESSURE',
          targetUsage: 0.85
        })
      ]);

      // Monitor system behavior under multiple faults
      const systemHealth = await systemMonitor.startContinuousMonitoring({
        interval: 500,
        duration: 30000
      });

      // Process orders during multi-fault scenario
      const orders = Array.from({ length: 500 }, (_, i) => ({
        id: `CASCADE_${i}`,
        symbol: 'AAPL',
        side: i % 2 === 0 ? 'buy' : 'sell',
        quantity: 100,
        price: 150.00,
        timestamp: Date.now(),
        userId: `USER_${i % 20}`
      }));

      const results = [];
      let cascadeDetected = false;

      for (let i = 0; i < orders.length; i += 10) {
        const batch = orders.slice(i, i + 10);
        const batchResults = await Promise.allSettled(
          batch.map(order => tradingEngine.processOrder(order))
        );
        
        results.push(...batchResults);

        // Check for cascade failure indicators
        const healthCheck = await systemMonitor.checkCascadeRisk();
        if (healthCheck.cascadeRisk > 0.7) {
          cascadeDetected = true;
          // System should activate circuit breakers
          const circuitBreakerStatus = await tradingEngine.getCircuitBreakerStatus();
          expect(circuitBreakerStatus.active).toBe(true);
          break;
        }
      }

      // Verify circuit breakers prevented full cascade
      if (cascadeDetected) {
        const finalSystemState = await tradingEngine.getSystemState();
        expect(finalSystemState.components.trading_engine.status).not.toBe('FAILED');
        expect(finalSystemState.components.order_book.status).not.toBe('FAILED');
      }

      // Clear all faults and verify recovery
      await faultInjector.clearAllFaults();
      await new Promise(resolve => setTimeout(resolve, 5000));

      const recoveryStatus = await tradingEngine.healthCheck();
      expect(recoveryStatus.status).toBe('healthy');
      expect(recoveryStatus.allComponentsOperational).toBe(true);
    });
  });
});