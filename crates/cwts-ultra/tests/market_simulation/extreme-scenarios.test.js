const { TradingEngine } = require('../../quantum_trading/core/trading_engine');
const { OrderBook } = require('../../quantum_trading/core/order_book');
const { RiskManager } = require('../../quantum_trading/core/risk_manager');
const { MarketSimulator } = require('../utils/market_simulator');
const { HistoricalDataProvider } = require('../utils/historical_data_provider');

describe('Market Condition Simulation Testing', () => {
  let tradingEngine;
  let orderBook;
  let riskManager;
  let marketSimulator;
  let historicalData;

  beforeEach(async () => {
    orderBook = new OrderBook();
    riskManager = new RiskManager();
    tradingEngine = new TradingEngine({ orderBook, riskManager });
    marketSimulator = new MarketSimulator();
    historicalData = new HistoricalDataProvider();
    
    await tradingEngine.initialize();
  });

  afterEach(async () => {
    await tradingEngine.shutdown();
  });

  describe('Flash Crash Scenarios', () => {
    test('should handle May 6, 2010 Flash Crash simulation', async () => {
      // Replicate the 2010 Flash Crash conditions
      const flashCrashData = await historicalData.getFlashCrashData('2010-05-06');
      
      const precrashPrice = 1150.00; // S&P 500 E-mini futures
      const crashLowPrice = 1056.00; // Intraday low
      const recoveryPrice = 1128.15; // Close price

      // Setup initial market state
      await marketSimulator.setInitialState({
        symbol: 'ES', // E-mini S&P 500
        price: precrashPrice,
        bidSize: 5000,
        askSize: 5000,
        spread: 0.25
      });

      // Create users with various strategies
      const users = {
        'HFT_MAKER': { strategy: 'market_making', capital: 100000000 },
        'MOMENTUM_TRADER': { strategy: 'momentum', capital: 50000000 },
        'PENSION_FUND': { strategy: 'long_term', capital: 1000000000 },
        'HEDGE_FUND': { strategy: 'arbitrage', capital: 500000000 }
      };

      for (const [userId, config] of Object.entries(users)) {
        await riskManager.setCreditLimit(userId, config.capital);
        await riskManager.setUserStrategy(userId, config.strategy);
      }

      console.log('Starting Flash Crash simulation...');

      // Phase 1: Normal trading (09:30 - 14:42)
      const normalTradingResults = await marketSimulator.simulateNormalTrading({
        duration: 5 * 60 * 1000, // 5 minutes compressed
        priceRange: [precrashPrice - 5, precrashPrice + 5],
        volumeProfile: 'normal'
      });

      expect(normalTradingResults.averageSpread).toBeLessThan(1.0);
      expect(normalTradingResults.marketDepth).toBeGreaterThan(1000);

      // Phase 2: Initial selling pressure (14:42)
      const sellPressure = await marketSimulator.simulateSellPressure({
        initialVolume: 75000, // Large sell program
        duration: 20 * 1000, // 20 seconds
        aggressiveness: 'high'
      });

      expect(sellPressure.priceImpact).toBeGreaterThan(0.5);

      // Phase 3: Flash crash (14:42 - 14:47)
      const crashResults = await marketSimulator.simulateFlashCrash({
        triggerPrice: precrashPrice - 10,
        bottomPrice: crashLowPrice,
        duration: 5 * 60 * 1000, // 5 minutes
        liquidityDrain: 0.95 // 95% liquidity reduction
      });

      // Verify crash characteristics
      expect(crashResults.maxDrawdown).toBeGreaterThan(0.08); // >8% drop
      expect(crashResults.liquidityReduction).toBeGreaterThan(0.90);
      expect(crashResults.spreadWidening).toBeGreaterThan(10);

      // Phase 4: Recovery (14:47 - 15:00)
      const recoveryResults = await marketSimulator.simulateRecovery({
        fromPrice: crashLowPrice,
        toPrice: recoveryPrice,
        duration: 13 * 60 * 1000, // 13 minutes
        liquidityReturn: 'gradual'
      });

      expect(recoveryResults.priceRecovery).toBeGreaterThan(0.6); // 60% recovery
      expect(recoveryResults.liquidityRecovery).toBeGreaterThan(0.7);

      // Verify system stability throughout crash
      const systemMetrics = await tradingEngine.getSystemMetrics();
      expect(systemMetrics.uptime).toBe(100);
      expect(systemMetrics.errorRate).toBeLessThan(0.01);

      // Verify risk controls activated
      const riskEvents = await riskManager.getRiskEvents();
      expect(riskEvents.filter(e => e.type === 'CIRCUIT_BREAKER')).toHaveLength.toBeGreaterThan(0);
      expect(riskEvents.filter(e => e.type === 'KILL_SWITCH')).toHaveLength.toBeGreaterThan(0);

      console.log('Flash Crash simulation completed successfully');
    });

    test('should handle algorithmic selling spiral', async () => {
      const symbol = 'SPY';
      const initialPrice = 400.00;
      
      await marketSimulator.setInitialState({
        symbol,
        price: initialPrice,
        volume: 100000
      });

      // Setup algorithmic traders with stop-loss orders
      const algoTraders = Array.from({ length: 50 }, (_, i) => ({
        userId: `ALGO_${i}`,
        stopLoss: initialPrice * (0.95 - i * 0.001), // Cascading stops
        position: 10000,
        strategy: 'momentum_following'
      }));

      for (const trader of algoTraders) {
        await riskManager.setCreditLimit(trader.userId, 50000000);
        await marketSimulator.setPosition(trader.userId, symbol, trader.position);
        await marketSimulator.setStopLoss(trader.userId, symbol, trader.stopLoss);
      }

      // Trigger initial price drop
      const triggerOrder = {
        id: 'TRIGGER_SELL',
        symbol,
        side: 'sell',
        quantity: 1000000, // Large sell order
        type: 'market',
        timestamp: Date.now(),
        userId: 'TRIGGER_USER'
      };

      await riskManager.setCreditLimit('TRIGGER_USER', 1000000000);
      await marketSimulator.setPosition('TRIGGER_USER', symbol, 1000000);

      const triggerResult = await tradingEngine.processOrder(triggerOrder);
      expect(triggerResult.success).toBe(true);

      // Monitor cascade effect
      const cascadeResults = await marketSimulator.monitorCascade({
        duration: 2 * 60 * 1000, // 2 minutes
        priceThreshold: initialPrice * 0.85, // 15% drop threshold
        stopCascadeThreshold: 0.05 // Stop when cascade slows
      });

      // Verify cascade characteristics
      expect(cascadeResults.triggeredStops).toBeGreaterThan(10);
      expect(cascadeResults.maxPriceMove).toBeGreaterThan(0.10); // >10% move
      expect(cascadeResults.volumeSpike).toBeGreaterThan(5); // 5x normal volume

      // Verify circuit breakers activated
      const circuitBreakers = await riskManager.getCircuitBreakerEvents();
      expect(circuitBreakers.length).toBeGreaterThan(0);

      // Verify system remained operational
      const finalSystemState = await tradingEngine.getSystemState();
      expect(finalSystemState.status).toBe('operational');
    });
  });

  describe('High Volatility Events', () => {
    test('should handle earnings announcement volatility spike', async () => {
      const symbol = 'AAPL';
      const preEarningsPrice = 150.00;
      
      // Setup pre-earnings market state
      await marketSimulator.setInitialState({
        symbol,
        price: preEarningsPrice,
        impliedVolatility: 0.25, // 25% IV
        timeToEarnings: 300 // 5 minutes
      });

      // Create realistic option and equity positions
      const participants = [
        { userId: 'EARNINGS_LONG', strategy: 'long_straddle', position: 10000 },
        { userId: 'EARNINGS_SHORT', strategy: 'short_strangle', position: -5000 },
        { userId: 'MOMENTUM_TRADER', strategy: 'breakout', position: 0 },
        { userId: 'MARKET_MAKER', strategy: 'delta_neutral', position: 50000 }
      ];

      for (const participant of participants) {
        await riskManager.setCreditLimit(participant.userId, 100000000);
        if (participant.position !== 0) {
          await marketSimulator.setPosition(participant.userId, symbol, participant.position);
        }
      }

      // Simulate earnings announcement
      const earningsResults = await marketSimulator.simulateEarningsEvent({
        symbol,
        expectedMove: 0.08, // 8% expected move
        actualMove: 0.15, // 15% actual move (surprise)
        direction: 'up',
        volumeMultiplier: 10,
        durationMs: 30 * 1000 // 30 seconds of intense activity
      });

      // Verify volatility spike characteristics
      expect(earningsResults.realizedVolatility).toBeGreaterThan(2.0); // >200% annualized
      expect(earningsResults.volumeIncrease).toBeGreaterThan(8);
      expect(earningsResults.spreadWidening).toBeGreaterThan(5);

      // Verify system handled high message rate
      expect(earningsResults.messageRate).toBeGreaterThan(100000); // >100K messages/sec
      expect(earningsResults.systemLatency.p99).toBeLessThan(10); // <10ms P99 latency

      // Verify risk controls managed exposure
      const riskMetrics = await riskManager.getRealTimeMetrics();
      participants.forEach(p => {
        expect(riskMetrics[p.userId].varExceeded).toBe(false);
      });
    });

    test('should handle currency devaluation event', async () => {
      const symbol = 'USDTRY'; // Turkish Lira example
      const preEventRate = 8.5;
      
      await marketSimulator.setInitialState({
        symbol,
        price: preEventRate,
        liquidity: 'normal',
        spread: 0.001 // 0.1% spread
      });

      // Setup currency traders
      const traders = [
        { userId: 'CENTRAL_BANK', strategy: 'intervention', capital: 10000000000 },
        { userId: 'HEDGE_FUND_1', strategy: 'short_currency', capital: 1000000000 },
        { userId: 'RETAIL_BANK', strategy: 'flow_trading', capital: 500000000 },
        { userId: 'CARRY_TRADER', strategy: 'carry_trade', capital: 100000000 }
      ];

      for (const trader of traders) {
        await riskManager.setCreditLimit(trader.userId, trader.capital);
      }

      // Simulate sudden devaluation trigger
      const devaluationResults = await marketSimulator.simulateDevaluation({
        symbol,
        trigger: 'central_bank_policy',
        severity: 'major', // 20%+ move
        liquidityShock: 0.8, // 80% liquidity reduction
        duration: 60 * 1000 // 1 minute
      });

      // Verify devaluation characteristics
      expect(devaluationResults.priceMove).toBeGreaterThan(0.20); // >20% devaluation
      expect(devaluationResults.liquidityDrain).toBeGreaterThan(0.75);
      expect(devaluationResults.spreadExplosion).toBeGreaterThan(50); // 50x spread widening

      // Verify gap handling
      expect(devaluationResults.gapSize).toBeGreaterThan(0.05); // >5% gap
      expect(devaluationResults.gapOrders.filled).toBeLessThan(0.1); // <10% gap orders filled

      // Verify system stability during extreme moves
      const systemHealth = await tradingEngine.healthCheck();
      expect(systemHealth.status).toBe('healthy');
      expect(systemHealth.gapHandling).toBe('functional');
    });
  });

  describe('Market Structure Events', () => {
    test('should handle exchange connectivity loss', async () => {
      const symbol = 'SPY';
      
      // Setup multi-exchange trading
      const exchanges = ['NYSE', 'NASDAQ', 'ARCA', 'BATS'];
      await marketSimulator.setupMultiExchange(symbol, exchanges);

      // Setup market makers on each exchange
      const marketMakers = exchanges.map(exchange => ({
        userId: `MM_${exchange}`,
        exchange,
        bidSize: 10000,
        askSize: 10000,
        spread: 0.01
      }));

      for (const mm of marketMakers) {
        await riskManager.setCreditLimit(mm.userId, 100000000);
        await marketSimulator.addLiquidity(mm.userId, symbol, mm.exchange, mm.bidSize, mm.askSize);
      }

      // Simulate primary exchange outage
      const outageResults = await marketSimulator.simulateExchangeOutage({
        exchange: 'NYSE',
        duration: 5 * 60 * 1000, // 5 minutes
        failureType: 'connectivity_loss'
      });

      // Verify smart order routing adapted
      expect(outageResults.routingShift.NASDAQ).toBeGreaterThan(0.5); // >50% to NASDAQ
      expect(outageResults.routingShift.ARCA).toBeGreaterThan(0.3); // >30% to ARCA

      // Verify market quality maintained
      expect(outageResults.spreadImpact).toBeLessThan(2.0); // <2x spread increase
      expect(outageResults.liquidityImpact).toBeLessThan(0.5); // <50% liquidity reduction

      // Verify no trades broke through
      expect(outageResults.brokenTrades).toBe(0);

      // Simulate recovery
      const recoveryResults = await marketSimulator.simulateExchangeRecovery({
        exchange: 'NYSE',
        recoveryType: 'gradual',
        duration: 2 * 60 * 1000 // 2 minutes
      });

      expect(recoveryResults.routingNormalization).toBeGreaterThan(0.8); // 80% normalized
    });

    test('should handle market data feed corruption', async () => {
      const symbol = 'TSLA';
      const truePrice = 800.00;
      
      await marketSimulator.setInitialState({
        symbol,
        price: truePrice,
        volume: 50000
      });

      // Setup traders relying on market data
      const traders = [
        { userId: 'STAT_ARB', strategy: 'statistical_arbitrage' },
        { userId: 'MOMENTUM', strategy: 'momentum' },
        { userId: 'MEAN_REVERT', strategy: 'mean_reversion' }
      ];

      for (const trader of traders) {
        await riskManager.setCreditLimit(trader.userId, 50000000);
      }

      // Inject corrupted market data
      const corruptionResults = await marketSimulator.simulateDataCorruption({
        symbol,
        corruptionType: 'price_spike',
        corruptedPrice: truePrice * 2.5, // 150% spike
        duration: 30 * 1000, // 30 seconds
        affectedFeeds: ['primary', 'backup']
      });

      // Verify corruption detection
      expect(corruptionResults.detectionTime).toBeLessThan(5000); // <5 seconds
      expect(corruptionResults.corruptionDetected).toBe(true);

      // Verify circuit breakers activated
      const circuitBreakers = await riskManager.getCircuitBreakerEvents();
      expect(circuitBreakers.filter(cb => cb.trigger === 'DATA_CORRUPTION')).toHaveLength.toBeGreaterThan(0);

      // Verify orders rejected during corruption
      const corruptionOrder = {
        id: 'CORRUPTION_ORDER',
        symbol,
        side: 'buy',
        quantity: 1000,
        price: truePrice * 2.4, // Near corrupted price
        type: 'limit',
        timestamp: Date.now(),
        userId: 'STAT_ARB'
      };

      const result = await tradingEngine.processOrder(corruptionOrder);
      expect(result.success).toBe(false);
      expect(result.error).toContain('market data');

      // Verify recovery
      const recoveryResults = await marketSimulator.simulateDataRecovery({
        symbol,
        recoveryPrice: truePrice,
        confidenceThreshold: 0.95
      });

      expect(recoveryResults.priceNormalized).toBe(true);
      expect(recoveryResults.tradingResumed).toBe(true);
    });
  });

  describe('Systemic Risk Events', () => {
    test('should handle credit crisis scenario', async () => {
      const symbols = ['SPY', 'XLF', 'TLT', 'VIX']; // Equities, financials, bonds, volatility
      
      // Setup pre-crisis state
      for (const symbol of symbols) {
        await marketSimulator.setInitialState({
          symbol,
          price: symbol === 'VIX' ? 15.0 : 100.0,
          correlation: symbol === 'VIX' ? -0.8 : 0.7 // VIX negatively correlated
        });
      }

      // Setup institutional participants
      const institutions = [
        { userId: 'PENSION_FUND', assets: 50000000000, leverage: 1.2 },
        { userId: 'HEDGE_FUND', assets: 10000000000, leverage: 3.0 },
        { userId: 'PROP_DESK', assets: 5000000000, leverage: 5.0 },
        { userId: 'RETAIL_FUND', assets: 20000000000, leverage: 1.0 }
      ];

      for (const inst of institutions) {
        await riskManager.setCreditLimit(inst.userId, inst.assets * inst.leverage);
        await riskManager.setLeverageLimit(inst.userId, inst.leverage);
      }

      // Trigger credit event
      const creditCrisisResults = await marketSimulator.simulateCreditCrisis({
        trigger: 'financial_institution_failure',
        severity: 'severe',
        duration: 24 * 60 * 60 * 1000, // 24 hours compressed to minutes
        contagionRate: 0.8
      });

      // Verify crisis characteristics
      expect(creditCrisisResults.correlationBreakdown).toBe(true);
      expect(creditCrisisResults.liquidityDrain).toBeGreaterThan(0.6);
      expect(creditCrisisResults.volatilitySpike.VIX).toBeGreaterThan(3.0); // VIX tripled

      // Verify margin calls triggered
      const marginCalls = await riskManager.getMarginCallEvents();
      expect(marginCalls.length).toBeGreaterThan(0);

      // Verify leveraged participants deleveraged
      const finalLeverage = await Promise.all(
        institutions.map(inst => riskManager.getCurrentLeverage(inst.userId))
      );
      
      institutions.forEach((inst, i) => {
        if (inst.leverage > 2.0) {
          expect(finalLeverage[i]).toBeLessThan(inst.leverage * 0.8); // 20% deleveraging
        }
      });

      // Verify system maintained orderly market
      const systemMetrics = await tradingEngine.getSystemMetrics();
      expect(systemMetrics.halts).toBeGreaterThan(0); // Trading halts occurred
      expect(systemMetrics.brokenTrades).toBe(0); // No broken trades
    });

    test('should handle algorithmic feedback loop', async () => {
      const symbol = 'QQQ';
      const initialPrice = 300.00;
      
      await marketSimulator.setInitialState({
        symbol,
        price: initialPrice,
        volume: 1000000
      });

      // Setup momentum algorithms with similar triggers
      const momentumAlgos = Array.from({ length: 20 }, (_, i) => ({
        userId: `MOMENTUM_${i}`,
        trigger: 0.02 + i * 0.001, // 2% - 4% triggers
        position: 0,
        maxPosition: 1000000
      }));

      for (const algo of momentumAlgos) {
        await riskManager.setCreditLimit(algo.userId, 500000000);
        await marketSimulator.setMomentumStrategy(algo.userId, symbol, algo.trigger, algo.maxPosition);
      }

      // Trigger initial momentum
      const triggerOrder = {
        id: 'MOMENTUM_TRIGGER',
        symbol,
        side: 'buy',
        quantity: 500000,
        type: 'market',
        timestamp: Date.now(),
        userId: 'EXTERNAL_TRIGGER'
      };

      await riskManager.setCreditLimit('EXTERNAL_TRIGGER', 1000000000);
      await marketSimulator.setPosition('EXTERNAL_TRIGGER', symbol, 1000000);

      const triggerResult = await tradingEngine.processOrder(triggerOrder);
      expect(triggerResult.success).toBe(true);

      // Monitor feedback loop
      const feedbackResults = await marketSimulator.monitorFeedbackLoop({
        symbol,
        duration: 5 * 60 * 1000, // 5 minutes
        breakThreshold: 0.10, // 10% move threshold
        velocityThreshold: 0.05 // 5% per minute
      });

      // Verify feedback loop characteristics
      expect(feedbackResults.amplificationFactor).toBeGreaterThan(2.0);
      expect(feedbackResults.maxVelocity).toBeGreaterThan(0.03); // >3% per minute
      expect(feedbackResults.participatingAlgos).toBeGreaterThan(10);

      // Verify circuit breakers stopped runaway
      const breakerEvents = await riskManager.getCircuitBreakerEvents();
      expect(breakerEvents.filter(e => e.type === 'VELOCITY_BREAKER')).toHaveLength.toBeGreaterThan(0);

      // Verify algorithms adapted
      const adaptationMetrics = await marketSimulator.getAlgorithmAdaptation();
      expect(adaptationMetrics.triggersAdjusted).toBeGreaterThan(5);
      expect(adaptationMetrics.positionLimitsReduced).toBeGreaterThan(3);
    });
  });

  describe('Recovery and Resilience Testing', () => {
    test('should recover from complete system failure', async () => {
      const symbol = 'SPY';
      
      // Setup active trading state
      const activeUsers = Array.from({ length: 100 }, (_, i) => `USER_${i}`);
      for (const userId of activeUsers) {
        await riskManager.setCreditLimit(userId, 10000000);
      }

      // Place many active orders
      const activeOrders = Array.from({ length: 1000 }, (_, i) => ({
        id: `ACTIVE_${i}`,
        symbol,
        side: i % 2 === 0 ? 'buy' : 'sell',
        quantity: 100,
        price: 100.00 + (Math.random() - 0.5) * 10,
        type: 'limit',
        timestamp: Date.now() + i,
        userId: activeUsers[i % activeUsers.length]
      }));

      const orderResults = await Promise.all(
        activeOrders.map(order => tradingEngine.processOrder(order))
      );
      
      const successfulOrders = orderResults.filter(r => r.success).length;
      expect(successfulOrders).toBeGreaterThan(800); // >80% success

      // Simulate complete system failure
      console.log('Simulating complete system failure...');
      await tradingEngine.simulateSystemFailure({
        type: 'complete_outage',
        duration: 30 * 1000 // 30 seconds
      });

      // Verify failure state
      const failureState = await tradingEngine.getSystemState();
      expect(failureState.status).toBe('failed');

      // Initiate recovery
      console.log('Initiating system recovery...');
      const recoveryResults = await tradingEngine.initiateRecovery({
        method: 'cold_restart',
        dataValidation: true,
        stateReconstruction: true
      });

      expect(recoveryResults.success).toBe(true);
      expect(recoveryResults.dataIntegrity).toBe(true);

      // Verify recovered state
      const recoveredState = await tradingEngine.getSystemState();
      expect(recoveredState.status).toBe('operational');

      // Verify order book integrity
      const orderBookState = await orderBook.getSnapshot();
      expect(orderBookState.bids.length + orderBookState.offers.length).toBeGreaterThan(0);

      // Verify positions are correct
      for (const userId of activeUsers.slice(0, 10)) {
        const position = await riskManager.getPosition(userId, symbol);
        expect(position).toBeDefined();
      }

      // Test normal operation resumed
      const testOrder = {
        id: 'RECOVERY_TEST',
        symbol,
        side: 'buy',
        quantity: 100,
        price: 100.00,
        type: 'limit',
        timestamp: Date.now(),
        userId: 'USER_0'
      };

      const testResult = await tradingEngine.processOrder(testOrder);
      expect(testResult.success).toBe(true);

      console.log('System recovery completed successfully');
    });
  });
});