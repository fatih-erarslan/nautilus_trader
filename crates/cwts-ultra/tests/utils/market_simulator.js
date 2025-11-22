/**
 * Market Simulator for Testing Trading System Under Various Market Conditions
 * Provides realistic market scenario simulation capabilities
 */

const EventEmitter = require('events');

class MarketSimulator extends EventEmitter {
  constructor() {
    super();
    this.marketState = new Map();
    this.participants = new Map();
    this.orders = new Map();
    this.trades = [];
    this.isRunning = false;
    this.simulationSpeed = 1.0; // 1.0 = real time
  }

  /**
   * Set initial market state for a symbol
   */
  async setInitialState(config) {
    const {
      symbol,
      price,
      volume = 100000,
      bidSize = 5000,
      askSize = 5000,
      spread = 0.01,
      impliedVolatility = 0.20,
      liquidity = 'normal'
    } = config;

    this.marketState.set(symbol, {
      symbol,
      lastPrice: price,
      bid: price - spread / 2,
      ask: price + spread / 2,
      bidSize,
      askSize,
      volume,
      impliedVolatility,
      liquidity,
      timestamp: Date.now(),
      priceHistory: [price],
      volumeHistory: [volume],
      orderBook: {
        bids: [],
        offers: []
      }
    });

    this.emit('marketStateUpdated', { symbol, state: this.marketState.get(symbol) });
  }

  /**
   * Simulate normal trading activity
   */
  async simulateNormalTrading(config) {
    const {
      duration = 60000, // 1 minute
      priceRange = [95, 105],
      volumeProfile = 'normal',
      participantCount = 50
    } = config;

    console.log(`Simulating normal trading for ${duration}ms...`);

    const startTime = Date.now();
    const endTime = startTime + duration;
    const results = {
      tradesExecuted: 0,
      totalVolume: 0,
      averageSpread: 0,
      marketDepth: 0,
      priceMovements: []
    };

    // Generate market participants
    await this.generateParticipants(participantCount);

    while (Date.now() < endTime && this.isRunning !== false) {
      // Generate random market activity
      await this.generateMarketActivity({
        priceRange,
        volumeProfile,
        intensity: 'normal'
      });

      // Update market metrics
      const currentMetrics = await this.calculateMarketMetrics();
      results.averageSpread += currentMetrics.spread;
      results.marketDepth += currentMetrics.depth;

      // Wait for next iteration
      await this.sleep(100 / this.simulationSpeed);
    }

    // Calculate final results
    const iterations = Math.floor(duration / 100);
    results.averageSpread /= iterations;
    results.marketDepth /= iterations;
    results.tradesExecuted = this.trades.length;
    results.totalVolume = this.trades.reduce((sum, trade) => sum + trade.quantity, 0);

    return results;
  }

  /**
   * Simulate flash crash scenario
   */
  async simulateFlashCrash(config) {
    const {
      triggerPrice,
      bottomPrice,
      duration = 300000, // 5 minutes
      liquidityDrain = 0.9
    } = config;

    console.log(`Simulating flash crash: ${triggerPrice} -> ${bottomPrice}...`);

    const startTime = Date.now();
    const results = {
      maxDrawdown: 0,
      liquidityReduction: 0,
      spreadWidening: 0,
      recoveryTime: 0,
      circuitBreakersTriggered: 0
    };

    // Phase 1: Trigger event
    await this.triggerLargeSellOrder(triggerPrice * 0.98);

    // Phase 2: Cascade selling
    const cascadeResults = await this.simulateCascadeSelling({
      startPrice: triggerPrice,
      endPrice: bottomPrice,
      liquidityDrain,
      duration: duration * 0.6 // 60% of time for crash
    });

    results.maxDrawdown = cascadeResults.maxDrawdown;
    results.liquidityReduction = cascadeResults.liquidityReduction;
    results.spreadWidening = cascadeResults.spreadWidening;

    // Phase 3: Recovery
    const recoveryStart = Date.now();
    const recoveryResults = await this.simulateRecovery({
      fromPrice: bottomPrice,
      toPrice: bottomPrice * 1.6, // Partial recovery
      duration: duration * 0.4,
      liquidityReturn: 'gradual'
    });

    results.recoveryTime = Date.now() - recoveryStart;
    results.priceRecovery = recoveryResults.priceRecovery;

    return results;
  }

  /**
   * Simulate massive sell pressure
   */
  async simulateSellPressure(config) {
    const {
      initialVolume,
      duration,
      aggressiveness = 'medium'
    } = config;

    const aggressivenessMultipliers = {
      low: 0.5,
      medium: 1.0,
      high: 2.0,
      extreme: 5.0
    };

    const multiplier = aggressivenessMultipliers[aggressiveness] || 1.0;
    const volumePerSecond = (initialVolume * multiplier) / (duration / 1000);

    const results = {
      totalVolume: 0,
      priceImpact: 0,
      ordersPlaced: 0
    };

    const startPrice = this.marketState.values().next().value.lastPrice;
    let currentPrice = startPrice;

    const interval = setInterval(async () => {
      const orderSize = Math.floor(volumePerSecond / 10); // 10 orders per second
      
      for (let i = 0; i < 10; i++) {
        const sellOrder = {
          id: `SELL_PRESSURE_${Date.now()}_${i}`,
          side: 'sell',
          quantity: orderSize + Math.floor(Math.random() * orderSize * 0.5),
          price: currentPrice * (0.995 - Math.random() * 0.01), // Slightly below market
          type: 'limit',
          timestamp: Date.now(),
          participant: 'SELL_PRESSURE_BOT'
        };

        await this.placeOrder(sellOrder);
        results.ordersPlaced++;
        results.totalVolume += sellOrder.quantity;
      }

      // Update price based on selling pressure
      currentPrice *= (1 - 0.001 * multiplier);
    }, 100);

    await this.sleep(duration);
    clearInterval(interval);

    results.priceImpact = (startPrice - currentPrice) / startPrice;
    return results;
  }

  /**
   * Simulate cascade selling effect
   */
  async simulateCascadeSelling(config) {
    const {
      startPrice,
      endPrice,
      liquidityDrain,
      duration
    } = config;

    const results = {
      maxDrawdown: 0,
      liquidityReduction: 0,
      spreadWidening: 0,
      stopLossesTriggered: 0
    };

    let currentPrice = startPrice;
    const priceDecline = (startPrice - endPrice) / startPrice;
    const stepsPerSecond = 10;
    const totalSteps = (duration / 1000) * stepsPerSecond;
    const priceStepSize = (startPrice - endPrice) / totalSteps;

    for (let step = 0; step < totalSteps; step++) {
      currentPrice -= priceStepSize;
      
      // Trigger stop losses
      await this.triggerStopLosses(currentPrice);
      
      // Reduce liquidity
      await this.reduceLiquidity(liquidityDrain * (step / totalSteps));
      
      // Widen spreads
      await this.widenSpreads(1 + (step / totalSteps) * 10);
      
      // Update market state
      await this.updateMarketPrice(currentPrice);
      
      await this.sleep(100 / this.simulationSpeed);
    }

    results.maxDrawdown = priceDecline;
    results.liquidityReduction = liquidityDrain;
    results.spreadWidening = 10; // 10x spread widening

    return results;
  }

  /**
   * Simulate market recovery
   */
  async simulateRecovery(config) {
    const {
      fromPrice,
      toPrice,
      duration,
      liquidityReturn = 'gradual'
    } = config;

    const results = {
      priceRecovery: 0,
      liquidityRecovery: 0,
      timeToRecover: duration
    };

    let currentPrice = fromPrice;
    const priceIncrease = toPrice - fromPrice;
    const stepsPerSecond = 5; // Slower recovery
    const totalSteps = (duration / 1000) * stepsPerSecond;
    const priceStepSize = priceIncrease / totalSteps;

    for (let step = 0; step < totalSteps; step++) {
      currentPrice += priceStepSize;
      
      // Gradually restore liquidity
      if (liquidityReturn === 'gradual') {
        await this.restoreLiquidity(step / totalSteps);
      }
      
      // Tighten spreads
      await this.tightenSpreads(Math.max(1, 10 - (step / totalSteps) * 9));
      
      // Add buying interest
      await this.addBuyingInterest(currentPrice);
      
      await this.updateMarketPrice(currentPrice);
      await this.sleep(200 / this.simulationSpeed);
    }

    results.priceRecovery = (toPrice - fromPrice) / fromPrice;
    results.liquidityRecovery = 0.8; // 80% liquidity recovery

    return results;
  }

  /**
   * Simulate earnings announcement volatility
   */
  async simulateEarningsEvent(config) {
    const {
      symbol,
      expectedMove,
      actualMove,
      direction,
      volumeMultiplier = 10,
      durationMs = 30000
    } = config;

    const results = {
      realizedVolatility: 0,
      volumeIncrease: 0,
      spreadWidening: 0,
      messageRate: 0,
      systemLatency: { p50: 0, p95: 0, p99: 0 }
    };

    const marketState = this.marketState.get(symbol);
    const startPrice = marketState.lastPrice;
    const targetPrice = startPrice * (1 + (direction === 'up' ? actualMove : -actualMove));

    console.log(`Simulating earnings event: ${symbol} ${startPrice} -> ${targetPrice}`);

    // Increase message rate dramatically
    const normalMessageRate = 1000; // 1K messages/sec
    const earningsMessageRate = normalMessageRate * 100; // 100K messages/sec

    // Generate intense trading activity
    const startTime = Date.now();
    let messageCount = 0;

    const interval = setInterval(async () => {
      const messagesThisSecond = earningsMessageRate / 10; // 100ms intervals
      
      for (let i = 0; i < messagesThisSecond; i++) {
        await this.generateEarningsOrder(symbol, targetPrice, volumeMultiplier);
        messageCount++;
      }
    }, 100);

    await this.sleep(durationMs);
    clearInterval(interval);

    const endTime = Date.now();
    const actualDuration = endTime - startTime;

    results.messageRate = messageCount / (actualDuration / 1000);
    results.volumeIncrease = volumeMultiplier;
    results.spreadWidening = 5; // 5x spread widening
    results.realizedVolatility = Math.abs(actualMove) * Math.sqrt(252); // Annualized

    // Mock latency measurements
    results.systemLatency = {
      p50: 2.5,
      p95: 8.2,
      p99: 15.7
    };

    return results;
  }

  /**
   * Simulate currency devaluation
   */
  async simulateDevaluation(config) {
    const {
      symbol,
      trigger,
      severity,
      liquidityShock,
      duration
    } = config;

    const results = {
      priceMove: 0,
      liquidityDrain: 0,
      spreadExplosion: 0,
      gapSize: 0,
      gapOrders: { total: 0, filled: 0 }
    };

    const marketState = this.marketState.get(symbol);
    const startPrice = marketState.lastPrice;

    // Determine devaluation magnitude
    const devaluationMagnitudes = {
      minor: 0.05,   // 5%
      moderate: 0.15, // 15%
      major: 0.25,    // 25%
      extreme: 0.50   // 50%
    };

    const devaluationSize = devaluationMagnitudes[severity] || 0.20;
    const targetPrice = startPrice * (1 - devaluationSize);

    console.log(`Simulating ${severity} devaluation: ${symbol} ${startPrice} -> ${targetPrice}`);

    // Create price gap
    await this.createPriceGap(symbol, startPrice, targetPrice);
    results.gapSize = (startPrice - targetPrice) / startPrice;

    // Drain liquidity instantly
    await this.drainLiquidity(symbol, liquidityShock);
    results.liquidityDrain = liquidityShock;

    // Explode spreads
    const spreadMultiplier = 50;
    await this.explodeSpreads(symbol, spreadMultiplier);
    results.spreadExplosion = spreadMultiplier;

    // Process gap orders
    const gapOrderResults = await this.processGapOrders(symbol, startPrice, targetPrice);
    results.gapOrders = gapOrderResults;

    results.priceMove = devaluationSize;

    return results;
  }

  /**
   * Simulate exchange outage
   */
  async simulateExchangeOutage(config) {
    const {
      exchange,
      duration,
      failureType
    } = config;

    const results = {
      routingShift: {},
      spreadImpact: 0,
      liquidityImpact: 0,
      brokenTrades: 0
    };

    console.log(`Simulating ${exchange} outage for ${duration}ms`);

    // Disable exchange
    await this.disableExchange(exchange);

    // Reroute orders to other exchanges
    const reroutingResults = await this.rerouteOrders(exchange);
    results.routingShift = reroutingResults.routingShift;
    results.spreadImpact = reroutingResults.spreadImpact;
    results.liquidityImpact = reroutingResults.liquidityImpact;

    // Wait for outage duration
    await this.sleep(duration);

    // Monitor for broken trades
    results.brokenTrades = await this.checkForBrokenTrades();

    return results;
  }

  /**
   * Simulate exchange recovery
   */
  async simulateExchangeRecovery(config) {
    const {
      exchange,
      recoveryType,
      duration
    } = config;

    const results = {
      routingNormalization: 0,
      liquidityReturn: 0,
      spreadNormalization: 0
    };

    console.log(`Simulating ${exchange} recovery (${recoveryType})`);

    if (recoveryType === 'gradual') {
      const steps = 10;
      const stepDuration = duration / steps;

      for (let step = 0; step < steps; step++) {
        const completionRatio = (step + 1) / steps;
        await this.partiallyEnableExchange(exchange, completionRatio);
        await this.sleep(stepDuration);
      }
    } else {
      await this.enableExchange(exchange);
    }

    results.routingNormalization = 0.85; // 85% back to normal
    results.liquidityReturn = 0.90; // 90% liquidity returned
    results.spreadNormalization = 0.95; // 95% spread normalization

    return results;
  }

  // Helper methods for simulation

  async generateParticipants(count) {
    const strategies = ['market_making', 'momentum', 'mean_reversion', 'arbitrage'];
    
    for (let i = 0; i < count; i++) {
      const participant = {
        id: `PARTICIPANT_${i}`,
        strategy: strategies[i % strategies.length],
        capital: 1000000 + Math.random() * 10000000,
        active: true
      };
      
      this.participants.set(participant.id, participant);
    }
  }

  async generateMarketActivity(config) {
    const participants = Array.from(this.participants.values()).filter(p => p.active);
    
    for (let i = 0; i < 5; i++) { // 5 orders per cycle
      const participant = participants[Math.floor(Math.random() * participants.length)];
      const order = await this.generateOrderForParticipant(participant, config);
      await this.placeOrder(order);
    }
  }

  async generateOrderForParticipant(participant, config) {
    const symbol = Array.from(this.marketState.keys())[0];
    const marketState = this.marketState.get(symbol);
    const { priceRange } = config;

    return {
      id: `ORDER_${Date.now()}_${Math.random()}`,
      symbol,
      side: Math.random() > 0.5 ? 'buy' : 'sell',
      quantity: Math.floor(Math.random() * 1000) + 100,
      price: priceRange[0] + Math.random() * (priceRange[1] - priceRange[0]),
      type: 'limit',
      timestamp: Date.now(),
      participant: participant.id
    };
  }

  async placeOrder(order) {
    this.orders.set(order.id, order);
    
    // Simulate order matching
    if (Math.random() < 0.3) { // 30% fill rate
      await this.executeTrade(order);
    }
  }

  async executeTrade(order) {
    const trade = {
      id: `TRADE_${Date.now()}_${Math.random()}`,
      orderId: order.id,
      symbol: order.symbol,
      side: order.side,
      quantity: order.quantity,
      price: order.price,
      timestamp: Date.now()
    };

    this.trades.push(trade);
    
    // Update market state
    const marketState = this.marketState.get(order.symbol);
    marketState.lastPrice = order.price;
    marketState.volume += order.quantity;
    marketState.priceHistory.push(order.price);
    marketState.volumeHistory.push(order.quantity);
  }

  async calculateMarketMetrics() {
    const symbol = Array.from(this.marketState.keys())[0];
    const marketState = this.marketState.get(symbol);
    
    return {
      spread: marketState.ask - marketState.bid,
      depth: marketState.bidSize + marketState.askSize,
      price: marketState.lastPrice,
      volume: marketState.volume
    };
  }

  async triggerStopLosses(currentPrice) {
    // Mock stop loss triggering
    const triggeredCount = Math.floor(Math.random() * 10);
    for (let i = 0; i < triggeredCount; i++) {
      const stopLossOrder = {
        id: `STOP_LOSS_${Date.now()}_${i}`,
        side: 'sell',
        quantity: Math.floor(Math.random() * 1000) + 500,
        price: currentPrice * 0.999,
        type: 'market',
        timestamp: Date.now(),
        participant: 'STOP_LOSS_TRIGGER'
      };
      
      await this.placeOrder(stopLossOrder);
    }
  }

  async reduceLiquidity(drainRatio) {
    for (const [symbol, state] of this.marketState) {
      state.bidSize *= (1 - drainRatio);
      state.askSize *= (1 - drainRatio);
    }
  }

  async widenSpreads(multiplier) {
    for (const [symbol, state] of this.marketState) {
      const midPrice = (state.bid + state.ask) / 2;
      const currentSpread = state.ask - state.bid;
      const newSpread = currentSpread * multiplier;
      
      state.bid = midPrice - newSpread / 2;
      state.ask = midPrice + newSpread / 2;
    }
  }

  async updateMarketPrice(newPrice) {
    const symbol = Array.from(this.marketState.keys())[0];
    const state = this.marketState.get(symbol);
    state.lastPrice = newPrice;
    state.priceHistory.push(newPrice);
    
    this.emit('priceUpdate', { symbol, price: newPrice, timestamp: Date.now() });
  }

  async sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // Mock implementations for complex operations
  async setupMultiExchange(symbol, exchanges) {
    // Mock setup
  }

  async addLiquidity(userId, symbol, exchange, bidSize, askSize) {
    // Mock liquidity addition
  }

  async disableExchange(exchange) {
    // Mock exchange disable
  }

  async enableExchange(exchange) {
    // Mock exchange enable
  }

  async rerouteOrders(failedExchange) {
    return {
      routingShift: { NASDAQ: 0.6, ARCA: 0.4 },
      spreadImpact: 1.5,
      liquidityImpact: 0.3
    };
  }

  async checkForBrokenTrades() {
    return 0; // No broken trades
  }

  // Additional helper methods...
  async triggerLargeSellOrder(price) {
    const order = {
      id: `LARGE_SELL_${Date.now()}`,
      side: 'sell',
      quantity: 1000000,
      price,
      type: 'market',
      timestamp: Date.now(),
      participant: 'LARGE_SELLER'
    };
    
    await this.placeOrder(order);
  }

  async restoreLiquidity(ratio) {
    for (const [symbol, state] of this.marketState) {
      state.bidSize = 5000 * ratio;
      state.askSize = 5000 * ratio;
    }
  }

  async tightenSpreads(multiplier) {
    await this.widenSpreads(1 / multiplier);
  }

  async addBuyingInterest(price) {
    for (let i = 0; i < 3; i++) {
      const buyOrder = {
        id: `BUY_INTEREST_${Date.now()}_${i}`,
        side: 'buy',
        quantity: Math.floor(Math.random() * 500) + 200,
        price: price * (0.998 + Math.random() * 0.004),
        type: 'limit',
        timestamp: Date.now(),
        participant: 'RECOVERY_BUYER'
      };
      
      await this.placeOrder(buyOrder);
    }
  }

  async generateEarningsOrder(symbol, targetPrice, volumeMultiplier) {
    const order = {
      id: `EARNINGS_${Date.now()}_${Math.random()}`,
      symbol,
      side: Math.random() > 0.5 ? 'buy' : 'sell',
      quantity: Math.floor(Math.random() * 1000 * volumeMultiplier) + 100,
      price: targetPrice * (0.99 + Math.random() * 0.02),
      type: 'limit',
      timestamp: Date.now(),
      participant: 'EARNINGS_TRADER'
    };
    
    await this.placeOrder(order);
  }

  async createPriceGap(symbol, fromPrice, toPrice) {
    const state = this.marketState.get(symbol);
    state.lastPrice = toPrice;
    state.bid = toPrice * 0.995;
    state.ask = toPrice * 1.005;
  }

  async drainLiquidity(symbol, drainRatio) {
    const state = this.marketState.get(symbol);
    state.bidSize *= (1 - drainRatio);
    state.askSize *= (1 - drainRatio);
  }

  async explodeSpreads(symbol, multiplier) {
    const state = this.marketState.get(symbol);
    const midPrice = state.lastPrice;
    const newSpread = 0.01 * multiplier;
    
    state.bid = midPrice - newSpread / 2;
    state.ask = midPrice + newSpread / 2;
  }

  async processGapOrders(symbol, oldPrice, newPrice) {
    // Mock gap order processing
    return {
      total: 50,
      filled: 3
    };
  }

  async partiallyEnableExchange(exchange, ratio) {
    // Mock partial enablement
  }
}

module.exports = { MarketSimulator };