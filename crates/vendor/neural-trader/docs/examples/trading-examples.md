# Complete Trading Workflow Examples

Comprehensive, production-ready examples for the Neural Trader Backend API.

## Table of Contents

1. [Basic Trading Setup](#basic-trading-setup)
2. [Advanced Strategy Backtesting](#advanced-strategy-backtesting)
3. [Multi-Asset Portfolio Management](#multi-asset-portfolio-management)
4. [Risk-Managed Trading Bot](#risk-managed-trading-bot)
5. [Automated Market Making](#automated-market-making)
6. [Complete Production Trading System](#complete-production-trading-system)

---

## 1. Basic Trading Setup

Simple getting started example with market analysis and trade execution.

```javascript
const {
  initNeuralTrader,
  getSystemInfo,
  listStrategies,
  quickAnalysis,
  simulateTrade,
  executeTrade,
  getPortfolioStatus
} = require('@rUv/neural-trader-backend');

async function basicTradingWorkflow() {
  try {
    // Step 1: Initialize the system
    console.log('=== Initializing Neural Trader ===');
    const initResult = await initNeuralTrader();
    console.log(initResult);

    // Step 2: Get system information
    const sysInfo = getSystemInfo();
    console.log(`\nVersion: ${sysInfo.version}`);
    console.log(`Features: ${sysInfo.features.join(', ')}`);
    console.log(`Total Tools: ${sysInfo.totalTools}`);

    // Step 3: List available strategies
    console.log('\n=== Available Strategies ===');
    const strategies = await listStrategies();
    strategies.forEach(s => {
      console.log(`- ${s.name}: ${s.description} (GPU: ${s.gpuCapable})`);
    });

    // Step 4: Analyze market
    console.log('\n=== Market Analysis ===');
    const symbol = 'AAPL';
    const analysis = await quickAnalysis(symbol, true);
    console.log(`Symbol: ${analysis.symbol}`);
    console.log(`Trend: ${analysis.trend}`);
    console.log(`Volatility: ${analysis.volatility}`);
    console.log(`Volume Trend: ${analysis.volumeTrend}`);
    console.log(`Recommendation: ${analysis.recommendation}`);

    // Step 5: Simulate trade first
    console.log('\n=== Trade Simulation ===');
    if (analysis.recommendation === 'BUY') {
      const simulation = await simulateTrade('momentum', symbol, 'buy', true);
      console.log(`Expected Return: ${simulation.expectedReturn}%`);
      console.log(`Risk Score: ${simulation.riskScore}`);
      console.log(`Execution Time: ${simulation.executionTimeMs}ms`);

      // Step 6: Execute trade if simulation looks good
      if (simulation.expectedReturn > 5 && simulation.riskScore < 0.5) {
        console.log('\n=== Executing Trade ===');
        const trade = await executeTrade(
          'momentum',
          symbol,
          'buy',
          100  // 100 shares
        );
        console.log(`Order ID: ${trade.orderId}`);
        console.log(`Status: ${trade.status}`);
        console.log(`Fill Price: $${trade.fillPrice}`);
        console.log(`Quantity: ${trade.quantity} shares`);
      } else {
        console.log('\nSimulation results not favorable - skipping trade');
      }
    }

    // Step 7: Check portfolio status
    console.log('\n=== Portfolio Status ===');
    const portfolio = await getPortfolioStatus(true);
    console.log(`Total Value: $${portfolio.totalValue.toLocaleString()}`);
    console.log(`Cash: $${portfolio.cash.toLocaleString()}`);
    console.log(`Positions: ${portfolio.positions}`);
    console.log(`Daily P&L: $${portfolio.dailyPnl.toLocaleString()}`);
    console.log(`Total Return: ${portfolio.totalReturn}%`);

  } catch (error) {
    console.error('Error in trading workflow:', error.message);
  }
}

// Run the workflow
basicTradingWorkflow();
```

**Output Example:**
```
=== Initializing Neural Trader ===
Neural Trader initialized successfully

Version: 2.1.1
Features: trading, neural, sports-betting, syndicates, e2b-swarm
Total Tools: 70+

=== Available Strategies ===
- momentum: Momentum-based trend following (GPU: true)
- mean_reversion: Mean reversion strategy (GPU: true)
- pairs_trading: Statistical pairs trading (GPU: true)

=== Market Analysis ===
Symbol: AAPL
Trend: bullish
Volatility: 0.25
Volume Trend: increasing
Recommendation: BUY

=== Trade Simulation ===
Expected Return: 8.5%
Risk Score: 0.35
Execution Time: 125ms

=== Executing Trade ===
Order ID: ORD-20250115-001
Status: filled
Fill Price: $175.25
Quantity: 100 shares

=== Portfolio Status ===
Total Value: $125,000
Cash: $25,000
Positions: 15
Daily P&L: $2,500
Total Return: 18.5%
```

---

## 2. Advanced Strategy Backtesting

Comprehensive backtesting with multiple strategies and optimization.

```javascript
const {
  runBacktest,
  optimizeStrategy,
  getStrategyInfo
} = require('@rUv/neural-trader-backend');

async function comprehensiveBacktest() {
  const strategies = ['momentum', 'mean_reversion', 'pairs_trading'];
  const symbols = ['AAPL', 'GOOGL', 'MSFT'];
  const startDate = '2023-01-01';
  const endDate = '2024-01-01';

  console.log('=== Running Comprehensive Backtest ===\n');

  for (const strategy of strategies) {
    console.log(`\n--- Testing Strategy: ${strategy} ---`);

    for (const symbol of symbols) {
      try {
        // Run backtest with GPU acceleration
        const result = await runBacktest(
          strategy,
          symbol,
          startDate,
          endDate,
          true  // Use GPU
        );

        console.log(`\n${symbol}:`);
        console.log(`  Total Return: ${result.totalReturn.toFixed(2)}%`);
        console.log(`  Sharpe Ratio: ${result.sharpeRatio.toFixed(2)}`);
        console.log(`  Max Drawdown: ${result.maxDrawdown.toFixed(2)}%`);
        console.log(`  Win Rate: ${result.winRate.toFixed(2)}%`);
        console.log(`  Total Trades: ${result.totalTrades}`);

        // If Sharpe ratio is good, optimize parameters
        if (result.sharpeRatio > 1.0) {
          console.log(`  → Optimizing parameters...`);

          const paramRanges = JSON.stringify({
            lookback_period: [10, 20, 30, 50],
            threshold: [0.01, 0.02, 0.03, 0.05],
            stop_loss: [0.02, 0.03, 0.05, 0.07]
          });

          const optimization = await optimizeStrategy(
            strategy,
            symbol,
            paramRanges,
            true
          );

          const bestParams = JSON.parse(optimization.bestParams);
          console.log(`  → Best Sharpe: ${optimization.bestSharpe.toFixed(2)}`);
          console.log(`  → Best Parameters:`, bestParams);
          console.log(`  → Optimization Time: ${optimization.optimizationTimeMs}ms`);
        }

      } catch (error) {
        console.error(`  Error testing ${symbol}:`, error.message);
      }
    }
  }

  // Summary
  console.log('\n\n=== Backtest Summary ===');
  console.log('Completed testing across:');
  console.log(`- ${strategies.length} strategies`);
  console.log(`- ${symbols.length} symbols`);
  console.log(`- 1 year period (${startDate} to ${endDate})`);
}

comprehensiveBacktest();
```

---

## 3. Multi-Asset Portfolio Management

Complete portfolio management with rebalancing and risk analysis.

```javascript
const {
  getPortfolioStatus,
  riskAnalysis,
  correlationAnalysis,
  portfolioRebalance
} = require('@rUv/neural-trader-backend');

async function portfolioManagement() {
  console.log('=== Portfolio Management Workflow ===\n');

  // Step 1: Get current portfolio
  console.log('1. Current Portfolio Status:');
  const portfolio = await getPortfolioStatus(true);
  console.log(`   Total Value: $${portfolio.totalValue.toLocaleString()}`);
  console.log(`   Positions: ${portfolio.positions}`);
  console.log(`   Daily P&L: $${portfolio.dailyPnl.toLocaleString()}`);
  console.log(`   Total Return: ${portfolio.totalReturn}%`);

  // Step 2: Risk analysis
  console.log('\n2. Risk Analysis:');
  const portfolioData = JSON.stringify({
    positions: [
      { symbol: 'AAPL', quantity: 100, cost_basis: 150 },
      { symbol: 'GOOGL', quantity: 50, cost_basis: 2800 },
      { symbol: 'MSFT', quantity: 75, cost_basis: 380 },
      { symbol: 'AMZN', quantity: 40, cost_basis: 3200 },
      { symbol: 'TSLA', quantity: 60, cost_basis: 220 }
    ]
  });

  const risk = await riskAnalysis(portfolioData, true);
  console.log(`   VaR (95%): $${risk.var95.toLocaleString()}`);
  console.log(`   CVaR (95%): $${risk.cvar95.toLocaleString()}`);
  console.log(`   Sharpe Ratio: ${risk.sharpeRatio.toFixed(2)}`);
  console.log(`   Max Drawdown: ${risk.maxDrawdown.toFixed(2)}%`);
  console.log(`   Beta: ${risk.beta.toFixed(2)}`);

  // Step 3: Correlation analysis
  console.log('\n3. Asset Correlation Analysis:');
  const symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'];
  const correlation = await correlationAnalysis(symbols, true);

  console.log(`   Analysis Period: ${correlation.analysisPeriod}`);
  console.log('\n   Correlation Matrix:');
  console.log('        ' + correlation.symbols.join('   '));
  correlation.matrix.forEach((row, i) => {
    const rowStr = row.map(val => val.toFixed(2).padStart(5)).join(' ');
    console.log(`   ${correlation.symbols[i].padEnd(5)} ${rowStr}`);
  });

  // Step 4: Portfolio rebalancing
  console.log('\n4. Portfolio Rebalancing:');
  const targetAllocation = JSON.stringify({
    'AAPL': 0.25,
    'GOOGL': 0.20,
    'MSFT': 0.25,
    'AMZN': 0.15,
    'TSLA': 0.15
  });

  const currentAllocation = JSON.stringify({
    'AAPL': 0.30,
    'GOOGL': 0.18,
    'MSFT': 0.28,
    'AMZN': 0.12,
    'TSLA': 0.12
  });

  const rebalance = await portfolioRebalance(targetAllocation, currentAllocation);

  console.log(`   Target Achieved: ${rebalance.targetAchieved}`);
  console.log(`   Estimated Cost: $${rebalance.estimatedCost.toLocaleString()}`);
  console.log('\n   Trades Needed:');
  rebalance.tradesNeeded.forEach(trade => {
    const action = trade.action.toUpperCase();
    console.log(`   - ${action} ${trade.quantity} ${trade.symbol}`);
  });

  // Step 5: Risk-adjusted recommendations
  console.log('\n5. Risk-Adjusted Recommendations:');
  if (risk.sharpeRatio < 1.0) {
    console.log('   ⚠ Low Sharpe Ratio - Consider diversification');
  }
  if (risk.beta > 1.5) {
    console.log('   ⚠ High Beta - Portfolio is volatile');
  }
  if (risk.maxDrawdown > 20) {
    console.log('   ⚠ Large drawdown risk - Reduce position sizes');
  }

  // Find highly correlated pairs
  console.log('\n6. Correlation Warnings:');
  for (let i = 0; i < correlation.matrix.length; i++) {
    for (let j = i + 1; j < correlation.matrix[i].length; j++) {
      const corr = correlation.matrix[i][j];
      if (corr > 0.8) {
        console.log(`   ⚠ High correlation (${corr.toFixed(2)}) between ${symbols[i]} and ${symbols[j]}`);
        console.log(`      → Consider reducing exposure to one`);
      }
    }
  }
}

portfolioManagement();
```

---

## 4. Risk-Managed Trading Bot

Automated trading bot with comprehensive risk management.

```javascript
const {
  initNeuralTrader,
  quickAnalysis,
  simulateTrade,
  executeTrade,
  getPortfolioStatus,
  riskAnalysis
} = require('@rUv/neural-trader-backend');

class RiskManagedTradingBot {
  constructor(config) {
    this.config = {
      maxPositionSize: config.maxPositionSize || 10000,
      maxDailyLoss: config.maxDailyLoss || 5000,
      maxPortfolioRisk: config.maxPortfolioRisk || 0.5,
      minExpectedReturn: config.minExpectedReturn || 5,
      maxRiskScore: config.maxRiskScore || 0.5,
      symbols: config.symbols || ['AAPL', 'GOOGL', 'MSFT'],
      strategy: config.strategy || 'momentum'
    };
    this.dailyPnL = 0;
    this.isInitialized = false;
  }

  async initialize() {
    console.log('=== Initializing Risk-Managed Trading Bot ===');
    await initNeuralTrader();
    this.isInitialized = true;
    console.log('Bot initialized successfully\n');
  }

  async checkRiskLimits() {
    // Check daily loss limit
    if (Math.abs(this.dailyPnL) >= this.config.maxDailyLoss) {
      console.log(`⚠ Daily loss limit reached: $${this.dailyPnL}`);
      return false;
    }

    // Check portfolio risk
    const portfolio = await getPortfolioStatus(true);
    const portfolioData = JSON.stringify({
      positions: await this.getCurrentPositions()
    });

    const risk = await riskAnalysis(portfolioData, true);
    if (risk.riskScore > this.config.maxPortfolioRisk) {
      console.log(`⚠ Portfolio risk too high: ${risk.riskScore}`);
      return false;
    }

    return true;
  }

  async getCurrentPositions() {
    // Placeholder - would fetch from actual portfolio
    return [
      { symbol: 'AAPL', quantity: 100, cost_basis: 150 },
      { symbol: 'GOOGL', quantity: 50, cost_basis: 2800 }
    ];
  }

  async analyzeOpportunity(symbol) {
    console.log(`\nAnalyzing ${symbol}...`);

    // Market analysis
    const analysis = await quickAnalysis(symbol, true);
    console.log(`  Trend: ${analysis.trend}`);
    console.log(`  Recommendation: ${analysis.recommendation}`);

    if (analysis.recommendation !== 'BUY') {
      console.log(`  ✗ Not a buy signal`);
      return null;
    }

    // Simulate trade
    const simulation = await simulateTrade(
      this.config.strategy,
      symbol,
      'buy',
      true
    );
    console.log(`  Expected Return: ${simulation.expectedReturn}%`);
    console.log(`  Risk Score: ${simulation.riskScore}`);

    // Check if meets criteria
    if (simulation.expectedReturn < this.config.minExpectedReturn) {
      console.log(`  ✗ Expected return too low`);
      return null;
    }

    if (simulation.riskScore > this.config.maxRiskScore) {
      console.log(`  ✗ Risk score too high`);
      return null;
    }

    console.log(`  ✓ Opportunity meets criteria`);
    return simulation;
  }

  async executeTrade(symbol, quantity) {
    try {
      const trade = await executeTrade(
        this.config.strategy,
        symbol,
        'buy',
        quantity
      );

      console.log(`\n✓ Trade Executed:`);
      console.log(`  Order ID: ${trade.orderId}`);
      console.log(`  Symbol: ${trade.symbol}`);
      console.log(`  Quantity: ${trade.quantity}`);
      console.log(`  Fill Price: $${trade.fillPrice}`);
      console.log(`  Status: ${trade.status}`);

      return trade;
    } catch (error) {
      console.error(`✗ Trade failed: ${error.message}`);
      return null;
    }
  }

  async runTradingCycle() {
    if (!this.isInitialized) {
      await this.initialize();
    }

    console.log('\n=== Starting Trading Cycle ===');
    console.log(`Time: ${new Date().toISOString()}`);

    // Check risk limits
    const canTrade = await this.checkRiskLimits();
    if (!canTrade) {
      console.log('✗ Risk limits exceeded - stopping trading');
      return;
    }

    // Check portfolio status
    const portfolio = await getPortfolioStatus(true);
    console.log(`\nPortfolio Status:`);
    console.log(`  Total Value: $${portfolio.totalValue.toLocaleString()}`);
    console.log(`  Daily P&L: $${portfolio.dailyPnl.toLocaleString()}`);
    this.dailyPnL = portfolio.dailyPnl;

    // Analyze each symbol
    for (const symbol of this.config.symbols) {
      const opportunity = await this.analyzeOpportunity(symbol);

      if (opportunity) {
        // Calculate position size
        const positionSize = Math.min(
          this.config.maxPositionSize,
          portfolio.totalValue * 0.1  // Max 10% of portfolio
        );
        const quantity = Math.floor(positionSize / 175); // Estimate price

        // Execute trade
        await this.executeTrade(symbol, quantity);

        // Wait between trades
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }

    console.log('\n=== Trading Cycle Complete ===\n');
  }

  async startBot(intervalMinutes = 60) {
    console.log(`Starting bot with ${intervalMinutes} minute intervals...`);

    // Run initial cycle
    await this.runTradingCycle();

    // Set up recurring cycles
    setInterval(async () => {
      await this.runTradingCycle();
    }, intervalMinutes * 60 * 1000);
  }
}

// Usage
const bot = new RiskManagedTradingBot({
  maxPositionSize: 10000,
  maxDailyLoss: 5000,
  maxPortfolioRisk: 0.5,
  minExpectedReturn: 5,
  maxRiskScore: 0.5,
  symbols: ['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
  strategy: 'momentum'
});

// Run once
bot.runTradingCycle();

// Or start continuous trading (uncomment to run)
// bot.startBot(60);  // Run every 60 minutes
```

---

## 5. Automated Market Making

Market making strategy with bid-ask spread management.

```javascript
const {
  quickAnalysis,
  executeTrade,
  getPortfolioStatus
} = require('@rUv/neural-trader-backend');

class MarketMaker {
  constructor(symbol, spreadBps = 10) {
    this.symbol = symbol;
    this.spreadBps = spreadBps;  // Spread in basis points
    this.inventory = 0;
    this.maxInventory = 1000;
    this.targetPrice = null;
  }

  async updateMarketData() {
    const analysis = await quickAnalysis(this.symbol, true);
    // In production, would get real mid price from order book
    this.targetPrice = 175.0;  // Placeholder
    return analysis;
  }

  calculateSpread() {
    const spread = this.targetPrice * (this.spreadBps / 10000);
    const bidPrice = this.targetPrice - (spread / 2);
    const askPrice = this.targetPrice + (spread / 2);

    return { bidPrice, askPrice, spread };
  }

  async placeOrders() {
    const { bidPrice, askPrice, spread } = this.calculateSpread();

    console.log(`\n=== Market Making for ${this.symbol} ===`);
    console.log(`Mid Price: $${this.targetPrice.toFixed(2)}`);
    console.log(`Spread: $${spread.toFixed(2)} (${this.spreadBps} bps)`);
    console.log(`Bid: $${bidPrice.toFixed(2)}`);
    console.log(`Ask: $${askPrice.toFixed(2)}`);
    console.log(`Current Inventory: ${this.inventory} shares`);

    // Adjust quantities based on inventory
    let bidQty = 100;
    let askQty = 100;

    if (this.inventory > this.maxInventory * 0.7) {
      // Inventory too high - reduce bid, increase ask
      bidQty = 50;
      askQty = 150;
      console.log(`⚠ High inventory - adjusting quantities`);
    } else if (this.inventory < -this.maxInventory * 0.7) {
      // Inventory too low - increase bid, reduce ask
      bidQty = 150;
      askQty = 50;
      console.log(`⚠ Low inventory - adjusting quantities`);
    }

    try {
      // Place bid (buy) limit order
      const bidOrder = await executeTrade(
        'market_making',
        this.symbol,
        'buy',
        bidQty,
        'limit',
        bidPrice
      );
      console.log(`✓ Bid order placed: ${bidOrder.orderId}`);

      // Place ask (sell) limit order
      const askOrder = await executeTrade(
        'market_making',
        this.symbol,
        'sell',
        askQty,
        'limit',
        askPrice
      );
      console.log(`✓ Ask order placed: ${askOrder.orderId}`);

      return { bidOrder, askOrder };
    } catch (error) {
      console.error(`✗ Order placement failed: ${error.message}`);
      return null;
    }
  }

  async run() {
    console.log(`Starting market maker for ${this.symbol}...`);

    while (true) {
      await this.updateMarketData();
      await this.placeOrders();

      // Wait before next update
      await new Promise(resolve => setTimeout(resolve, 5000));
    }
  }
}

// Run market maker
const mm = new MarketMaker('AAPL', 10);  // 10 bps spread
mm.run();
```

---

## 6. Complete Production Trading System

Full-featured production trading system with all components.

```javascript
const {
  initNeuralTrader,
  initAuth,
  createApiKey,
  validateApiKey,
  initRateLimiter,
  checkRateLimit,
  initAuditLogger,
  logAuditEvent,
  quickAnalysis,
  simulateTrade,
  executeTrade,
  getPortfolioStatus,
  riskAnalysis,
  runBacktest
} = require('@rUv/neural-trader-backend');

class ProductionTradingSystem {
  constructor(config) {
    this.config = config;
    this.apiKey = null;
    this.isInitialized = false;
  }

  async initialize() {
    console.log('=== Initializing Production Trading System ===\n');

    // 1. Initialize core system
    console.log('1. Initializing neural trader...');
    await initNeuralTrader();

    // 2. Setup authentication
    console.log('2. Setting up authentication...');
    initAuth(this.config.jwtSecret);
    this.apiKey = createApiKey(
      this.config.username,
      'user',
      100,  // 100 req/min
      365   // 1 year expiration
    );
    console.log(`   API Key created (keep secure!)`);

    // 3. Initialize rate limiting
    console.log('3. Initializing rate limiter...');
    initRateLimiter({
      maxRequestsPerMinute: 100,
      burstSize: 20,
      windowDurationSecs: 60
    });

    // 4. Setup audit logging
    console.log('4. Setting up audit logging...');
    initAuditLogger(10000, true, true);

    // Log initialization
    logAuditEvent(
      'info',
      'system',
      'system_initialized',
      'success',
      null,
      this.config.username,
      null,
      'trading_system',
      JSON.stringify({ version: '2.1.1' })
    );

    this.isInitialized = true;
    console.log('\n✓ System initialized successfully\n');
  }

  async validateAccess() {
    // Validate API key
    try {
      const user = validateApiKey(this.apiKey);
      console.log(`✓ Access validated for user: ${user.username}`);

      // Check rate limit
      const allowed = checkRateLimit(user.userId);
      if (!allowed) {
        throw new Error('Rate limit exceeded');
      }

      return true;
    } catch (error) {
      console.error(`✗ Access validation failed: ${error.message}`);
      return false;
    }
  }

  async backtestStrategy() {
    console.log('=== Running Strategy Backtest ===\n');

    const strategies = ['momentum', 'mean_reversion'];
    const results = [];

    for (const strategy of strategies) {
      console.log(`Testing ${strategy}...`);

      const result = await runBacktest(
        strategy,
        'AAPL',
        '2023-01-01',
        '2024-01-01',
        true
      );

      console.log(`  Total Return: ${result.totalReturn.toFixed(2)}%`);
      console.log(`  Sharpe Ratio: ${result.sharpeRatio.toFixed(2)}`);
      console.log(`  Max Drawdown: ${result.maxDrawdown.toFixed(2)}%`);
      console.log(`  Win Rate: ${result.winRate.toFixed(2)}%\n`);

      results.push({ strategy, ...result });

      // Log backtest
      logAuditEvent(
        'info',
        'trading',
        'backtest_completed',
        'success',
        null,
        this.config.username,
        null,
        strategy,
        JSON.stringify({
          return: result.totalReturn,
          sharpe: result.sharpeRatio
        })
      );
    }

    // Select best strategy
    const best = results.reduce((a, b) =>
      a.sharpeRatio > b.sharpeRatio ? a : b
    );
    console.log(`✓ Best strategy: ${best.strategy} (Sharpe: ${best.sharpeRatio.toFixed(2)})\n`);

    return best.strategy;
  }

  async analyzeAndTrade(symbol, strategy) {
    console.log(`=== Analyzing ${symbol} with ${strategy} ===\n`);

    // 1. Market analysis
    console.log('1. Market Analysis:');
    const analysis = await quickAnalysis(symbol, true);
    console.log(`   Trend: ${analysis.trend}`);
    console.log(`   Volatility: ${analysis.volatility}`);
    console.log(`   Recommendation: ${analysis.recommendation}`);

    // 2. Risk check
    console.log('\n2. Risk Analysis:');
    const portfolio = await getPortfolioStatus(true);
    const portfolioData = JSON.stringify({
      positions: [
        { symbol: 'AAPL', quantity: 100, cost_basis: 150 },
        { symbol: 'GOOGL', quantity: 50, cost_basis: 2800 }
      ]
    });

    const risk = await riskAnalysis(portfolioData, true);
    console.log(`   VaR (95%): $${risk.var95.toLocaleString()}`);
    console.log(`   Sharpe Ratio: ${risk.sharpeRatio.toFixed(2)}`);
    console.log(`   Max Drawdown: ${risk.maxDrawdown.toFixed(2)}%`);

    // 3. Trade simulation
    console.log('\n3. Trade Simulation:');
    if (analysis.recommendation === 'BUY') {
      const simulation = await simulateTrade(strategy, symbol, 'buy', true);
      console.log(`   Expected Return: ${simulation.expectedReturn}%`);
      console.log(`   Risk Score: ${simulation.riskScore}`);

      // 4. Execute if criteria met
      if (simulation.expectedReturn > 5 && simulation.riskScore < 0.5) {
        console.log('\n4. Executing Trade:');

        const trade = await executeTrade(strategy, symbol, 'buy', 100);
        console.log(`   ✓ Trade executed: ${trade.orderId}`);
        console.log(`   Fill Price: $${trade.fillPrice}`);
        console.log(`   Status: ${trade.status}`);

        // Log trade
        logAuditEvent(
          'info',
          'trading',
          'trade_executed',
          'success',
          null,
          this.config.username,
          null,
          symbol,
          JSON.stringify({
            strategy,
            quantity: 100,
            price: trade.fillPrice
          })
        );

        return trade;
      } else {
        console.log('\n4. Trade Criteria Not Met');
        console.log('   Skipping execution');
      }
    } else {
      console.log('\n   Not a buy signal - skipping');
    }

    return null;
  }

  async monitorPortfolio() {
    console.log('\n=== Portfolio Monitoring ===\n');

    const portfolio = await getPortfolioStatus(true);

    console.log('Current Status:');
    console.log(`  Total Value: $${portfolio.totalValue.toLocaleString()}`);
    console.log(`  Cash: $${portfolio.cash.toLocaleString()}`);
    console.log(`  Positions: ${portfolio.positions}`);
    console.log(`  Daily P&L: $${portfolio.dailyPnl.toLocaleString()}`);
    console.log(`  Total Return: ${portfolio.totalReturn}%`);

    // Alerts
    if (portfolio.dailyPnl < -5000) {
      console.log('\n⚠ ALERT: Daily loss exceeds $5,000');
      logAuditEvent(
        'warning',
        'portfolio',
        'daily_loss_alert',
        'triggered',
        null,
        this.config.username,
        null,
        'portfolio',
        JSON.stringify({ pnl: portfolio.dailyPnl })
      );
    }

    if (portfolio.totalReturn < -10) {
      console.log('\n⚠ ALERT: Portfolio down more than 10%');
      logAuditEvent(
        'warning',
        'portfolio',
        'drawdown_alert',
        'triggered',
        null,
        this.config.username,
        null,
        'portfolio',
        JSON.stringify({ return: portfolio.totalReturn })
      );
    }
  }

  async run() {
    try {
      // Initialize system
      if (!this.isInitialized) {
        await this.initialize();
      }

      // Validate access
      const hasAccess = await this.validateAccess();
      if (!hasAccess) {
        throw new Error('Access validation failed');
      }

      // Backtest and select strategy
      const bestStrategy = await this.backtestStrategy();

      // Analyze and trade
      await this.analyzeAndTrade('AAPL', bestStrategy);

      // Monitor portfolio
      await this.monitorPortfolio();

      console.log('\n=== Trading System Run Complete ===\n');

    } catch (error) {
      console.error(`\n✗ System error: ${error.message}`);

      // Log error
      logAuditEvent(
        'error',
        'system',
        'system_error',
        'failure',
        null,
        this.config.username,
        null,
        'trading_system',
        JSON.stringify({ error: error.message })
      );
    }
  }
}

// Usage
const system = new ProductionTradingSystem({
  username: 'trader_001',
  jwtSecret: 'your-secret-key-here'
});

system.run();
```

---

## Testing Best Practices

1. **Always test in simulation first**
2. **Use small position sizes initially**
3. **Monitor risk metrics continuously**
4. **Implement circuit breakers**
5. **Log all trades for audit**
6. **Backtest thoroughly before live trading**
7. **Use rate limiting to prevent abuse**
8. **Validate all inputs**
9. **Handle errors gracefully**
10. **Monitor system health**

---

## Production Checklist

- [ ] Environment variables configured
- [ ] API keys securely stored
- [ ] Rate limiting enabled
- [ ] Audit logging active
- [ ] Risk limits configured
- [ ] Backup strategy in place
- [ ] Monitoring alerts set up
- [ ] Error handling tested
- [ ] Performance benchmarked
- [ ] Documentation complete

---

**Next Steps:**
- Review [Neural Network Examples](./neural-examples.md)
- Explore [Syndicate Examples](./syndicate-examples.md)
- Learn [E2B Swarm Deployment](./swarm-examples.md)
