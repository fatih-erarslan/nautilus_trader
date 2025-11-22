const { TradingEngine } = require('../../quantum_trading/core/trading_engine');
const { OrderBook } = require('../../quantum_trading/core/order_book');
const { RiskManager } = require('../../quantum_trading/core/risk_manager');
const { SecurityValidator } = require('../utils/security_validator');

describe('Security and Memory Safety Testing', () => {
  let tradingEngine;
  let orderBook;
  let riskManager;
  let securityValidator;

  beforeEach(async () => {
    orderBook = new OrderBook();
    riskManager = new RiskManager();
    tradingEngine = new TradingEngine({ orderBook, riskManager });
    securityValidator = new SecurityValidator();
    
    await tradingEngine.initialize();
  });

  afterEach(async () => {
    await tradingEngine.shutdown();
  });

  describe('Input Validation and Sanitization', () => {
    test('should reject orders with malicious payloads', async () => {
      const maliciousPayloads = [
        // SQL injection attempts
        { id: "'; DROP TABLE orders; --", symbol: 'AAPL' },
        { id: 'ORD001', symbol: "'; DELETE FROM positions; --" },
        
        // XSS attempts
        { id: '<script>alert("xss")</script>', symbol: 'AAPL' },
        { id: 'ORD002', symbol: '<img src=x onerror=alert(1)>' },
        
        // Path traversal
        { id: '../../etc/passwd', symbol: 'AAPL' },
        { id: 'ORD003', symbol: '../../../root/.ssh/id_rsa' },
        
        // Command injection
        { id: 'ORD004; rm -rf /', symbol: 'AAPL' },
        { id: 'ORD005', symbol: 'AAPL && cat /etc/passwd' },
        
        // Buffer overflow attempts
        { id: 'A'.repeat(10000), symbol: 'AAPL' },
        { id: 'ORD006', symbol: 'B'.repeat(1000000) },
        
        // Null byte injection
        { id: 'ORD007\x00malicious', symbol: 'AAPL' },
        { id: 'ORD008', symbol: 'AAPL\x00../../etc/passwd' }
      ];

      const results = [];
      for (const payload of maliciousPayloads) {
        const order = {
          ...payload,
          side: 'buy',
          quantity: 100,
          price: 150.00,
          type: 'limit',
          timestamp: Date.now(),
          userId: 'SECURITY_TEST_USER'
        };

        const result = await tradingEngine.processOrder(order);
        results.push(result);
      }

      // All malicious orders should be rejected
      expect(results.every(r => !r.success)).toBe(true);
      
      // Verify specific security error messages
      results.forEach(result => {
        expect(result.error).toMatch(/invalid|malicious|security|sanitization/i);
      });

      // Verify system remains stable after attacks
      const healthCheck = await tradingEngine.healthCheck();
      expect(healthCheck.status).toBe('healthy');
    });

    test('should validate all numeric inputs for overflow/underflow', async () => {
      const extremeValues = [
        // JavaScript number limits
        { quantity: Number.MAX_SAFE_INTEGER + 1 },
        { quantity: Number.MIN_SAFE_INTEGER - 1 },
        { price: Number.MAX_VALUE },
        { price: Number.MIN_VALUE },
        
        // Infinity values
        { quantity: Number.POSITIVE_INFINITY },
        { price: Number.NEGATIVE_INFINITY },
        
        // NaN values
        { quantity: NaN },
        { price: NaN },
        
        // Very large numbers that could cause precision loss
        { quantity: 999999999999999999999 },
        { price: 0.000000000000000001 },
        
        // Negative values where they shouldn't be allowed
        { quantity: -100 },
        { price: -150.00 },
        
        // Zero values
        { quantity: 0 },
        { price: 0 }
      ];

      const results = [];
      for (const extremeValue of extremeValues) {
        const order = {
          id: `EXTREME_${Date.now()}_${Math.random()}`,
          symbol: 'AAPL',
          side: 'buy',
          quantity: 100,
          price: 150.00,
          type: 'limit',
          timestamp: Date.now(),
          userId: 'EXTREME_TEST_USER',
          ...extremeValue
        };

        const result = await tradingEngine.processOrder(order);
        results.push(result);
      }

      // All extreme value orders should be rejected
      expect(results.every(r => !r.success)).toBe(true);
      
      // Verify appropriate error messages
      results.forEach(result => {
        expect(result.error).toMatch(/invalid|range|limit|overflow|underflow/i);
      });
    });

    test('should prevent integer overflow in calculations', async () => {
      const userId = 'OVERFLOW_TEST_USER';
      await riskManager.setCreditLimit(userId, Number.MAX_SAFE_INTEGER);

      // Attempt orders that could cause integer overflow
      const overflowOrders = [
        {
          id: 'OVERFLOW_001',
          symbol: 'AAPL',
          side: 'buy',
          quantity: Math.floor(Number.MAX_SAFE_INTEGER / 100),
          price: 150.00,
          type: 'limit',
          timestamp: Date.now(),
          userId
        },
        {
          id: 'OVERFLOW_002',
          symbol: 'AAPL',
          side: 'buy',
          quantity: 1000,
          price: Number.MAX_SAFE_INTEGER / 500,
          type: 'limit',
          timestamp: Date.now(),
          userId
        }
      ];

      const results = [];
      for (const order of overflowOrders) {
        const result = await tradingEngine.processOrder(order);
        results.push(result);
      }

      // Orders should be rejected or handled safely
      results.forEach(result => {
        if (result.success) {
          // If accepted, verify calculations are correct
          expect(result.orderValue).toBeDefined();
          expect(Number.isSafeInteger(result.orderValue)).toBe(true);
        } else {
          // If rejected, should have appropriate error
          expect(result.error).toMatch(/overflow|limit|calculation/i);
        }
      });
    });
  });

  describe('Memory Safety', () => {
    test('should prevent buffer overflows in string handling', async () => {
      const largeStrings = [
        'A'.repeat(1024 * 1024), // 1MB string
        'B'.repeat(10 * 1024 * 1024), // 10MB string
        '🚀'.repeat(1024 * 1024), // 1M Unicode characters
        '\x00'.repeat(1024 * 1024) // Null bytes
      ];

      const results = [];
      for (let i = 0; i < largeStrings.length; i++) {
        const order = {
          id: largeStrings[i],
          symbol: 'AAPL',
          side: 'buy',
          quantity: 100,
          price: 150.00,
          type: 'limit',
          timestamp: Date.now(),
          userId: 'BUFFER_TEST_USER',
          metadata: largeStrings[i] // Test metadata field too
        };

        const startMemory = process.memoryUsage();
        const result = await tradingEngine.processOrder(order);
        const endMemory = process.memoryUsage();
        
        results.push({
          result,
          memoryDelta: endMemory.heapUsed - startMemory.heapUsed
        });
      }

      // Large strings should be rejected or truncated safely
      results.forEach(({ result, memoryDelta }) => {
        if (result.success) {
          // If accepted, memory usage should be reasonable
          expect(memoryDelta).toBeLessThan(100 * 1024 * 1024); // <100MB
        } else {
          expect(result.error).toMatch(/size|length|memory|buffer/i);
        }
      });

      // System should remain stable
      const healthCheck = await tradingEngine.healthCheck();
      expect(healthCheck.status).toBe('healthy');
    });

    test('should handle memory pressure gracefully', async () => {
      const userId = 'MEMORY_PRESSURE_USER';
      await riskManager.setCreditLimit(userId, 100000000);

      // Create memory pressure by processing many large orders
      const largeOrders = Array.from({ length: 10000 }, (_, i) => ({
        id: `PRESSURE_${i}`,
        symbol: 'AAPL',
        side: i % 2 === 0 ? 'buy' : 'sell',
        quantity: 10000,
        price: 150.00 + (Math.random() - 0.5) * 10,
        type: 'limit',
        timestamp: Date.now() + i,
        userId,
        metadata: 'X'.repeat(1024) // 1KB metadata per order
      }));

      const initialMemory = process.memoryUsage();
      let processedCount = 0;
      let rejectedCount = 0;

      // Process orders in batches
      for (let i = 0; i < largeOrders.length; i += 100) {
        const batch = largeOrders.slice(i, i + 100);
        
        const batchResults = await Promise.all(
          batch.map(order => tradingEngine.processOrder(order))
        );

        batchResults.forEach(result => {
          if (result.success) {
            processedCount++;
          } else {
            rejectedCount++;
          }
        });

        // Check memory usage
        const currentMemory = process.memoryUsage();
        const memoryIncrease = currentMemory.heapUsed - initialMemory.heapUsed;

        // If memory usage is too high, system should start rejecting orders
        if (memoryIncrease > 500 * 1024 * 1024) { // 500MB
          expect(rejectedCount).toBeGreaterThan(0);
        }

        // Force GC periodically to simulate real conditions
        if (i % 500 === 0 && global.gc) {
          global.gc();
        }
      }

      console.log(`Memory Pressure Test: ${processedCount} processed, ${rejectedCount} rejected`);

      // System should handle memory pressure without crashing
      expect(processedCount + rejectedCount).toBe(largeOrders.length);
      
      const finalMemory = process.memoryUsage();
      console.log(`Memory usage: ${finalMemory.heapUsed / 1024 / 1024}MB`);
    });

    test('should prevent memory leaks in error conditions', async () => {
      const initialMemory = process.memoryUsage();
      
      // Generate orders that will cause various error conditions
      const errorOrders = Array.from({ length: 1000 }, (_, i) => ({
        id: `ERROR_${i}`,
        symbol: 'INVALID_SYMBOL_' + i,
        side: 'invalid_side',
        quantity: -100,
        price: 'not_a_number',
        type: 'invalid_type',
        timestamp: 'invalid_timestamp',
        userId: null
      }));

      // Process error orders
      const results = [];
      for (const order of errorOrders) {
        try {
          const result = await tradingEngine.processOrder(order);
          results.push(result);
        } catch (error) {
          results.push({ success: false, error: error.message });
        }
      }

      // All orders should fail gracefully
      expect(results.every(r => !r.success)).toBe(true);

      // Force garbage collection
      if (global.gc) global.gc();

      const finalMemory = process.memoryUsage();
      const memoryIncrease = finalMemory.heapUsed - initialMemory.heapUsed;

      // Memory should not increase significantly from error handling
      expect(memoryIncrease).toBeLessThan(50 * 1024 * 1024); // <50MB
    });
  });

  describe('Cryptographic Security', () => {
    test('should validate data integrity with cryptographic hashes', async () => {
      const userId = 'CRYPTO_TEST_USER';
      await riskManager.setCreditLimit(userId, 1000000);

      const order = {
        id: 'CRYPTO_ORDER_001',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00,
        type: 'limit',
        timestamp: Date.now(),
        userId
      };

      const result = await tradingEngine.processOrder(order);
      expect(result.success).toBe(true);

      // Verify order has cryptographic hash
      const storedOrder = await orderBook.getOrderById('CRYPTO_ORDER_001');
      expect(storedOrder.hash).toBeDefined();
      expect(storedOrder.hash).toMatch(/^[a-f0-9]{64}$/); // SHA-256 hash

      // Verify hash integrity
      const calculatedHash = securityValidator.calculateOrderHash(storedOrder);
      expect(calculatedHash).toBe(storedOrder.hash);

      // Verify tampering detection
      const tamperedOrder = { ...storedOrder, quantity: 200 };
      const tamperedHash = securityValidator.calculateOrderHash(tamperedOrder);
      expect(tamperedHash).not.toBe(storedOrder.hash);
    });

    test('should protect sensitive data with encryption', async () => {
      const sensitiveOrder = {
        id: 'SENSITIVE_ORDER_001',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00,
        type: 'limit',
        timestamp: Date.now(),
        userId: 'SENSITIVE_USER',
        clientInfo: {
          accountNumber: '1234567890',
          ssn: '123-45-6789',
          bankAccount: '9876543210'
        }
      };

      await riskManager.setCreditLimit('SENSITIVE_USER', 1000000);
      const result = await tradingEngine.processOrder(sensitiveOrder);
      expect(result.success).toBe(true);

      // Verify sensitive data is encrypted in storage
      const storedOrder = await orderBook.getOrderById('SENSITIVE_ORDER_001');
      expect(storedOrder.clientInfo.accountNumber).not.toBe('1234567890');
      expect(storedOrder.clientInfo.ssn).not.toBe('123-45-6789');
      expect(storedOrder.clientInfo.bankAccount).not.toBe('9876543210');

      // Verify data can be decrypted correctly
      const decryptedOrder = await securityValidator.decryptSensitiveData(storedOrder);
      expect(decryptedOrder.clientInfo.accountNumber).toBe('1234567890');
      expect(decryptedOrder.clientInfo.ssn).toBe('123-45-6789');
      expect(decryptedOrder.clientInfo.bankAccount).toBe('9876543210');
    });

    test('should implement secure random number generation', async () => {
      const randomNumbers = [];
      
      // Generate random numbers for order IDs
      for (let i = 0; i < 1000; i++) {
        const randomId = securityValidator.generateSecureRandomId();
        randomNumbers.push(randomId);
      }

      // Verify randomness properties
      const uniqueNumbers = new Set(randomNumbers);
      expect(uniqueNumbers.size).toBe(randomNumbers.length); // All unique

      // Verify entropy (no obvious patterns)
      const firstBits = randomNumbers.map(n => n.charCodeAt(0));
      const averageFirstBit = firstBits.reduce((a, b) => a + b, 0) / firstBits.length;
      expect(averageFirstBit).toBeCloseTo(128, 10); // Should be around middle of range

      // Verify cryptographic strength
      const entropy = securityValidator.calculateEntropy(randomNumbers.join(''));
      expect(entropy).toBeGreaterThan(7.0); // High entropy
    });
  });

  describe('Access Control and Authorization', () => {
    test('should enforce user isolation', async () => {
      const user1 = 'ISOLATED_USER_001';
      const user2 = 'ISOLATED_USER_002';
      
      await riskManager.setCreditLimit(user1, 1000000);
      await riskManager.setCreditLimit(user2, 1000000);

      // User 1 places order
      const order1 = {
        id: 'ISOLATION_ORDER_001',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00,
        type: 'limit',
        timestamp: Date.now(),
        userId: user1
      };

      const result1 = await tradingEngine.processOrder(order1);
      expect(result1.success).toBe(true);

      // User 2 tries to access User 1's order
      try {
        await tradingEngine.cancelOrder('ISOLATION_ORDER_001', user2);
        expect(true).toBe(false); // Should not reach here
      } catch (error) {
        expect(error.message).toContain('unauthorized');
      }

      // User 1 can access their own order
      const cancelResult = await tradingEngine.cancelOrder('ISOLATION_ORDER_001', user1);
      expect(cancelResult.success).toBe(true);
    });

    test('should validate session tokens', async () => {
      const validToken = securityValidator.generateSessionToken('VALID_USER');
      const invalidToken = 'invalid_token_123';
      const expiredToken = securityValidator.generateExpiredToken('EXPIRED_USER');

      const testCases = [
        { token: validToken, shouldSucceed: true },
        { token: invalidToken, shouldSucceed: false },
        { token: expiredToken, shouldSucceed: false },
        { token: null, shouldSucceed: false },
        { token: undefined, shouldSucceed: false }
      ];

      for (const testCase of testCases) {
        const order = {
          id: `TOKEN_TEST_${Date.now()}_${Math.random()}`,
          symbol: 'AAPL',
          side: 'buy',
          quantity: 100,
          price: 150.00,
          type: 'limit',
          timestamp: Date.now(),
          userId: 'TOKEN_USER',
          sessionToken: testCase.token
        };

        const result = await tradingEngine.processOrder(order);
        
        if (testCase.shouldSucceed) {
          expect(result.success).toBe(true);
        } else {
          expect(result.success).toBe(false);
          expect(result.error).toMatch(/token|session|auth/i);
        }
      }
    });

    test('should implement rate limiting per user', async () => {
      const userId = 'RATE_LIMITED_USER';
      await riskManager.setCreditLimit(userId, 1000000);
      await riskManager.setRateLimit(userId, 10, 1000); // 10 orders per second

      // Generate orders exceeding rate limit
      const orders = Array.from({ length: 25 }, (_, i) => ({
        id: `RATE_LIMIT_${i}`,
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00,
        type: 'limit',
        timestamp: Date.now() + i,
        userId
      }));

      const results = [];
      const startTime = Date.now();

      // Submit all orders rapidly
      for (const order of orders) {
        const result = await tradingEngine.processOrder(order);
        results.push(result);
      }

      const endTime = Date.now();
      const duration = endTime - startTime;

      // Some orders should be rate limited
      const acceptedOrders = results.filter(r => r.success);
      const rateLimitedOrders = results.filter(r => !r.success && r.error.includes('rate limit'));

      expect(rateLimitedOrders.length).toBeGreaterThan(0);
      expect(acceptedOrders.length).toBeLessThanOrEqual(Math.ceil(10 * duration / 1000) + 5); // Allow some tolerance

      console.log(`Rate limiting: ${acceptedOrders.length} accepted, ${rateLimitedOrders.length} rate limited`);
    });
  });

  describe('Audit and Compliance Security', () => {
    test('should maintain tamper-proof audit logs', async () => {
      const userId = 'AUDIT_SECURITY_USER';
      await riskManager.setCreditLimit(userId, 1000000);

      const order = {
        id: 'AUDIT_SECURITY_ORDER',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00,
        type: 'limit',
        timestamp: Date.now(),
        userId
      };

      const result = await tradingEngine.processOrder(order);
      expect(result.success).toBe(true);

      // Get audit logs
      const auditLogs = await tradingEngine.getAuditLogs('AUDIT_SECURITY_ORDER');
      expect(auditLogs.length).toBeGreaterThan(0);

      // Verify each log entry has integrity protection
      for (const log of auditLogs) {
        expect(log.hash).toBeDefined();
        expect(log.signature).toBeDefined();
        expect(log.timestamp).toBeDefined();

        // Verify hash integrity
        const calculatedHash = securityValidator.calculateLogHash(log);
        expect(calculatedHash).toBe(log.hash);

        // Verify digital signature
        const signatureValid = await securityValidator.verifySignature(log);
        expect(signatureValid).toBe(true);
      }

      // Attempt to tamper with log
      const tamperedLog = { ...auditLogs[0], quantity: 999 };
      const tamperedHash = securityValidator.calculateLogHash(tamperedLog);
      expect(tamperedHash).not.toBe(auditLogs[0].hash);

      // Verify tampering is detected
      const tamperDetected = await securityValidator.detectTampering(auditLogs);
      expect(tamperDetected).toBe(false); // Original logs should be intact
    });
  });
});