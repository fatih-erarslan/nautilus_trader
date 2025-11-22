const { TradingEngine } = require('../../quantum_trading/core/trading_engine');
const { RiskManager } = require('../../quantum_trading/core/risk_manager');
const { AuditLogger } = require('../../quantum_trading/compliance/audit_logger');
const { ComplianceValidator } = require('../../quantum_trading/compliance/compliance_validator');

describe('SEC Rule 15c3-5 Compliance Testing', () => {
  let tradingEngine;
  let riskManager;
  let auditLogger;
  let complianceValidator;

  beforeEach(() => {
    auditLogger = new AuditLogger();
    riskManager = new RiskManager({ auditLogger });
    complianceValidator = new ComplianceValidator();
    tradingEngine = new TradingEngine({ 
      riskManager, 
      auditLogger,
      complianceValidator 
    });
  });

  describe('Pre-Trade Risk Controls (15c3-5(c)(1)(i))', () => {
    test('must prevent orders exceeding appropriate pre-set credit or capital thresholds', async () => {
      const highRiskUser = 'HIGH_RISK_USER';
      const creditLimit = 1000000; // $1M credit limit

      // Set credit limit for user
      await riskManager.setCreditLimit(highRiskUser, creditLimit);

      // Attempt order exceeding credit limit
      const excessiveOrder = {
        id: 'EXCESSIVE_001',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 10000,
        price: 150.00, // $1.5M total value
        timestamp: Date.now(),
        userId: highRiskUser
      };

      const result = await tradingEngine.processOrder(excessiveOrder);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Credit limit exceeded');
      
      // Verify audit trail
      const auditLogs = await auditLogger.getLogsByOrderId('EXCESSIVE_001');
      expect(auditLogs).toContainEqual(
        expect.objectContaining({
          event: 'ORDER_REJECTED',
          reason: 'CREDIT_LIMIT_EXCEEDED',
          userId: highRiskUser,
          orderValue: 1500000,
          creditLimit: 1000000
        })
      );
    });

    test('must prevent orders for customers without appropriate authorization', async () => {
      const unauthorizedUser = 'UNAUTHORIZED_USER';

      // Do not set authorization for user
      const unauthorizedOrder = {
        id: 'UNAUTH_001',
        symbol: 'RESTRICTED_STOCK',
        side: 'buy',
        quantity: 100,
        price: 50.00,
        timestamp: Date.now(),
        userId: unauthorizedUser
      };

      const result = await tradingEngine.processOrder(unauthorizedOrder);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Insufficient authorization');
      
      // Verify compliance validation occurred
      const compliance = await complianceValidator.getLastValidation();
      expect(compliance.result).toBe('REJECTED');
      expect(compliance.reason).toBe('UNAUTHORIZED_USER');
    });

    test('must prevent orders in securities for which there is not a reasonable basis to believe market exists', async () => {
      const illiquidOrder = {
        id: 'ILLIQUID_001',
        symbol: 'ILLIQUID_STOCK',
        side: 'buy',
        quantity: 1000000, // Massive quantity for illiquid stock
        price: 0.01,
        timestamp: Date.now(),
        userId: 'USER_001'
      };

      // Mock market data showing no liquidity
      await riskManager.setMarketLiquidity('ILLIQUID_STOCK', {
        avgDailyVolume: 100,
        bidAskSpread: 50.0, // 50% spread indicates illiquidity
        lastTradeTime: Date.now() - (24 * 60 * 60 * 1000) // 24 hours ago
      });

      const result = await tradingEngine.processOrder(illiquidOrder);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Insufficient market liquidity');
      
      // Verify liquidity check was performed
      const liquidityCheck = await auditLogger.getLogsByType('LIQUIDITY_CHECK');
      expect(liquidityCheck).toContainEqual(
        expect.objectContaining({
          symbol: 'ILLIQUID_STOCK',
          result: 'INSUFFICIENT_LIQUIDITY',
          avgDailyVolume: 100,
          requestedQuantity: 1000000
        })
      );
    });

    test('must prevent erroneous orders due to problems with algorithmic trading systems', async () => {
      const algorithmicUser = 'ALGO_USER_001';
      
      // Set up algorithmic trading detection
      await riskManager.enableAlgorithmicMonitoring(algorithmicUser);

      // Send orders that appear algorithmic and erroneous
      const erroneousOrders = [
        { price: 150.00, quantity: 100 },
        { price: 150.01, quantity: 100 },
        { price: 150.02, quantity: 100 },
        { price: 1500.00, quantity: 100 }, // Price 10x higher - erroneous
        { price: 150.03, quantity: 100 }
      ].map((order, i) => ({
        id: `ALGO_${i}`,
        symbol: 'AAPL',
        side: 'buy',
        ...order,
        timestamp: Date.now() + i * 100, // Rapid succession
        userId: algorithmicUser
      }));

      const results = [];
      for (const order of erroneousOrders) {
        const result = await tradingEngine.processOrder(order);
        results.push(result);
      }

      // The erroneous order (index 3) should be rejected
      expect(results[3].success).toBe(false);
      expect(results[3].error).toContain('Erroneous order detected');

      // Verify algorithmic pattern detection
      const algoDetection = await auditLogger.getLogsByType('ALGO_DETECTION');
      expect(algoDetection).toContainEqual(
        expect.objectContaining({
          userId: algorithmicUser,
          pattern: 'PRICE_ANOMALY',
          orderId: 'ALGO_3'
        })
      );
    });
  });

  describe('Kill Switch Requirements (15c3-5(c)(1)(ii))', () => {
    test('must have kill switch functionality to immediately stop orders', async () => {
      const userId = 'KILL_SWITCH_USER';
      
      // Process normal orders first
      const normalOrder = {
        id: 'NORMAL_001',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00,
        timestamp: Date.now(),
        userId
      };

      const normalResult = await tradingEngine.processOrder(normalOrder);
      expect(normalResult.success).toBe(true);

      // Activate kill switch
      const killSwitchResult = await riskManager.activateKillSwitch(userId, 'MANUAL_INTERVENTION');
      expect(killSwitchResult.success).toBe(true);
      expect(killSwitchResult.timestamp).toBeDefined();

      // Attempt to process order after kill switch
      const blockedOrder = {
        id: 'BLOCKED_001',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00,
        timestamp: Date.now(),
        userId
      };

      const blockedResult = await tradingEngine.processOrder(blockedOrder);
      expect(blockedResult.success).toBe(false);
      expect(blockedResult.error).toContain('Kill switch activated');

      // Verify kill switch timing (must be immediate)
      const killSwitchLogs = await auditLogger.getLogsByType('KILL_SWITCH');
      const activationTime = killSwitchLogs[0].timestamp;
      const rejectionTime = await auditLogger.getLogsByOrderId('BLOCKED_001');
      const responseTime = rejectionTime[0].timestamp - activationTime;
      
      expect(responseTime).toBeLessThan(100); // Must respond within 100ms
    });

    test('must maintain kill switch audit trail with precise timing', async () => {
      const userId = 'AUDIT_USER';
      const reason = 'RISK_THRESHOLD_EXCEEDED';

      const killSwitchResult = await riskManager.activateKillSwitch(userId, reason);

      // Verify detailed audit trail
      const auditLogs = await auditLogger.getLogsByType('KILL_SWITCH');
      const killSwitchLog = auditLogs[0];

      expect(killSwitchLog).toMatchObject({
        event: 'KILL_SWITCH_ACTIVATED',
        userId,
        reason,
        timestamp: expect.any(Number),
        systemStatus: 'BLOCKED',
        activatedBy: expect.any(String),
        nanosecondPrecision: expect.any(Number)
      });

      // Verify timestamp precision (must be nanosecond accurate)
      expect(killSwitchLog.nanosecondPrecision).toBeGreaterThan(0);
      expect(killSwitchLog.timestamp).toBeValidTimestamp();
    });

    test('must allow selective kill switch by symbol or user', async () => {
      const user1 = 'USER_001';
      const user2 = 'USER_002';

      // Activate kill switch for specific symbol
      await riskManager.activateSymbolKillSwitch('AAPL', 'VOLATILITY_SPIKE');

      // Test orders for different symbols
      const appleOrder = {
        id: 'AAPL_001',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00,
        timestamp: Date.now(),
        userId: user1
      };

      const googleOrder = {
        id: 'GOOGL_001',
        symbol: 'GOOGL',
        side: 'buy',
        quantity: 100,
        price: 2500.00,
        timestamp: Date.now(),
        userId: user1
      };

      const appleResult = await tradingEngine.processOrder(appleOrder);
      const googleResult = await tradingEngine.processOrder(googleOrder);

      expect(appleResult.success).toBe(false);
      expect(appleResult.error).toContain('Symbol kill switch active');
      expect(googleResult.success).toBe(true);

      // Activate user-specific kill switch
      await riskManager.activateUserKillSwitch(user2, 'SUSPICIOUS_ACTIVITY');

      const user2Order = {
        id: 'USER2_001',
        symbol: 'MSFT',
        side: 'buy',
        quantity: 100,
        price: 300.00,
        timestamp: Date.now(),
        userId: user2
      };

      const user2Result = await tradingEngine.processOrder(user2Order);
      expect(user2Result.success).toBe(false);
      expect(user2Result.error).toContain('User kill switch active');
    });
  });

  describe('Audit Trail Requirements', () => {
    test('must maintain complete audit trail of all risk control actions', async () => {
      const userId = 'AUDIT_TEST_USER';
      
      // Perform various trading activities
      const activities = [
        { type: 'ORDER', order: { id: 'AUD_001', symbol: 'AAPL', side: 'buy', quantity: 100, price: 150.00, userId } },
        { type: 'CANCEL', orderId: 'AUD_001' },
        { type: 'CREDIT_CHECK', userId, amount: 50000 },
        { type: 'KILL_SWITCH', userId, reason: 'TEST' }
      ];

      for (const activity of activities) {
        switch (activity.type) {
          case 'ORDER':
            await tradingEngine.processOrder({ ...activity.order, timestamp: Date.now() });
            break;
          case 'CANCEL':
            await tradingEngine.cancelOrder(activity.orderId);
            break;
          case 'CREDIT_CHECK':
            await riskManager.checkCreditLimit(activity.userId, activity.amount);
            break;
          case 'KILL_SWITCH':
            await riskManager.activateKillSwitch(activity.userId, activity.reason);
            break;
        }
      }

      // Verify comprehensive audit trail
      const auditTrail = await auditLogger.getCompleteAuditTrail(userId);
      
      expect(auditTrail.length).toBeGreaterThanOrEqual(4);
      expect(auditTrail).toContainEqual(expect.objectContaining({ event: 'ORDER_RECEIVED' }));
      expect(auditTrail).toContainEqual(expect.objectContaining({ event: 'ORDER_CANCELLED' }));
      expect(auditTrail).toContainEqual(expect.objectContaining({ event: 'CREDIT_CHECK' }));
      expect(auditTrail).toContainEqual(expect.objectContaining({ event: 'KILL_SWITCH_ACTIVATED' }));

      // Verify audit trail immutability
      const originalHash = await auditLogger.calculateTrailHash(userId);
      
      // Attempt to modify audit trail (should fail)
      try {
        await auditLogger.modifyLog(auditTrail[0].id, { event: 'MODIFIED' });
        expect(true).toBe(false); // Should not reach here
      } catch (error) {
        expect(error.message).toContain('Audit trail is immutable');
      }

      const finalHash = await auditLogger.calculateTrailHash(userId);
      expect(finalHash).toBe(originalHash);
    });

    test('must ensure audit trail data integrity and tamper resistance', async () => {
      const criticalOrder = {
        id: 'CRITICAL_001',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 10000,
        price: 150.00,
        timestamp: Date.now(),
        userId: 'CRITICAL_USER'
      };

      await tradingEngine.processOrder(criticalOrder);

      // Get audit record
      const auditRecord = await auditLogger.getLogsByOrderId('CRITICAL_001');
      const originalRecord = auditRecord[0];

      // Verify cryptographic integrity
      expect(originalRecord.hash).toBeDefined();
      expect(originalRecord.signature).toBeDefined();
      expect(originalRecord.timestamp).toBeValidTimestamp();

      // Verify hash calculation
      const calculatedHash = auditLogger.calculateRecordHash(originalRecord);
      expect(calculatedHash).toBe(originalRecord.hash);

      // Verify digital signature
      const signatureValid = await auditLogger.verifySignature(originalRecord);
      expect(signatureValid).toBe(true);

      // Test tamper detection
      const tamperedRecord = { ...originalRecord, quantity: 20000 };
      const tamperedHash = auditLogger.calculateRecordHash(tamperedRecord);
      expect(tamperedHash).not.toBe(originalRecord.hash);
    });
  });

  describe('Real-time Monitoring Requirements', () => {
    test('must monitor trading in real-time for compliance violations', async () => {
      const userId = 'REALTIME_USER';
      
      // Set up real-time monitoring
      const monitoringResults = [];
      riskManager.on('complianceViolation', (violation) => {
        monitoringResults.push(violation);
      });

      // Generate pattern that should trigger monitoring
      const rapidOrders = Array.from({ length: 100 }, (_, i) => ({
        id: `RAPID_${i}`,
        symbol: 'AAPL',
        side: 'buy',
        quantity: 100,
        price: 150.00 + (i * 0.01), // Incrementing prices
        timestamp: Date.now() + (i * 10), // 10ms apart
        userId
      }));

      // Process orders rapidly
      for (const order of rapidOrders) {
        await tradingEngine.processOrder(order);
      }

      // Wait for real-time processing
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Verify monitoring detected patterns
      expect(monitoringResults.length).toBeGreaterThan(0);
      expect(monitoringResults).toContainEqual(
        expect.objectContaining({
          type: 'RAPID_TRADING_PATTERN',
          userId,
          orderCount: 100,
          timeWindow: expect.any(Number)
        })
      );

      // Verify real-time response (must be within seconds)
      const firstViolation = monitoringResults[0];
      const detectionDelay = firstViolation.detectedAt - firstViolation.firstOrderTime;
      expect(detectionDelay).toBeLessThan(5000); // 5 second max detection time
    });

    test('must provide real-time risk metrics and alerts', async () => {
      const userId = 'METRICS_USER';
      
      // Set risk thresholds
      await riskManager.setRiskThresholds(userId, {
        maxPositionValue: 1000000,
        maxDailyVolume: 10000,
        maxOrderRate: 10 // orders per minute
      });

      // Generate orders approaching risk limits
      const orders = Array.from({ length: 8 }, (_, i) => ({
        id: `RISK_${i}`,
        symbol: 'AAPL',
        side: 'buy',
        quantity: 1000,
        price: 150.00,
        timestamp: Date.now() + (i * 1000), // 1 second apart
        userId
      }));

      const alerts = [];
      riskManager.on('riskAlert', (alert) => {
        alerts.push(alert);
      });

      // Process orders
      for (const order of orders) {
        await tradingEngine.processOrder(order);
        
        // Get real-time metrics
        const metrics = await riskManager.getRealTimeMetrics(userId);
        expect(metrics).toMatchObject({
          currentPositionValue: expect.any(Number),
          dailyVolume: expect.any(Number),
          orderRatePerMinute: expect.any(Number),
          riskScore: expect.any(Number)
        });
      }

      // Verify alerts were generated
      expect(alerts.length).toBeGreaterThan(0);
      expect(alerts).toContainEqual(
        expect.objectContaining({
          type: 'POSITION_LIMIT_APPROACHING',
          userId,
          currentValue: expect.any(Number),
          limit: 1000000
        })
      );
    });
  });
});