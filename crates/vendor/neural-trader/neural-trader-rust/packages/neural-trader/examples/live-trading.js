/**
 * Neural Trader - Live Trading Example
 *
 * This example demonstrates:
 * - Real-time market data
 * - Live order execution
 * - Risk monitoring
 * - Performance tracking
 */

const {
  LiveTradingEngine,
  Strategy,
  TechnicalIndicators,
  RiskManager,
  MarketDataFeed,
  OrderExecutor
} = require('neural-trader');

/**
 * Adaptive Trading Strategy
 */
class AdaptiveTradingStrategy extends Strategy {
  constructor() {
    super('Adaptive Live Strategy');
    this.state = {
      lastSignal: null,
      confidence: 0,
      marketRegime: 'neutral'
    };
  }

  onBar(bar, portfolio, context) {
    // Detect market regime
    this.state.marketRegime = this.detectMarketRegime(context);

    // Adapt strategy based on regime
    if (this.state.marketRegime === 'trending') {
      return this.trendFollowingLogic(bar, portfolio, context);
    } else if (this.state.marketRegime === 'ranging') {
      return this.meanReversionLogic(bar, portfolio, context);
    }

    return null;
  }

  detectMarketRegime(context) {
    const prices = context.getPriceHistory(50);
    if (prices.length < 50) return 'neutral';

    const adx = TechnicalIndicators.adx(prices, 14);

    if (adx > 25) return 'trending';
    if (adx < 20) return 'ranging';
    return 'neutral';
  }

  trendFollowingLogic(bar, portfolio, context) {
    const prices = context.getPriceHistory(20);
    const sma20 = TechnicalIndicators.sma(prices, 20);
    const sma50 = TechnicalIndicators.sma(context.getPriceHistory(50), 50);

    if (bar.close > sma20 && sma20 > sma50 && !portfolio.hasPosition()) {
      this.state.confidence = 0.8;
      return {
        action: 'buy',
        quantity: Math.floor(portfolio.cash * 0.5 / bar.close),
        price: bar.close,
        confidence: this.state.confidence
      };
    }

    return null;
  }

  meanReversionLogic(bar, portfolio, context) {
    const prices = context.getPriceHistory(20);
    const { upper, lower } = TechnicalIndicators.bollingerBands(prices, 20, 2);

    if (bar.close < lower && !portfolio.hasPosition()) {
      this.state.confidence = 0.7;
      return {
        action: 'buy',
        quantity: Math.floor(portfolio.cash * 0.3 / bar.close),
        price: bar.close,
        confidence: this.state.confidence
      };
    }

    return null;
  }

  onOrderFilled(order, portfolio) {
    console.log(`✅ Order filled: ${order.action} ${order.quantity} @ $${order.price}`);
  }

  onOrderRejected(order, reason) {
    console.error(`❌ Order rejected: ${reason}`);
  }
}

/**
 * Set up live trading environment
 */
async function setupLiveTrading(mode = 'paper') {
  console.log(`🚀 Starting Live Trading (${mode} mode)\n`);

  // Initialize market data feed
  const dataFeed = new MarketDataFeed({
    provider: 'alpaca', // or 'interactive-brokers', 'binance', etc.
    apiKey: process.env.BROKER_API_KEY,
    apiSecret: process.env.BROKER_API_SECRET,
    mode: mode // 'paper' or 'live'
  });

  // Initialize order executor
  const executor = new OrderExecutor({
    broker: 'alpaca',
    apiKey: process.env.BROKER_API_KEY,
    apiSecret: process.env.BROKER_API_SECRET,
    mode: mode
  });

  // Create strategy
  const strategy = new AdaptiveTradingStrategy();

  // Set up risk management
  const riskManager = new RiskManager({
    maxPositionSize: 0.15,
    maxDailyLoss: 0.03,
    maxPortfolioRisk: 0.05,
    stopLoss: 0.05,
    takeProfit: 0.12,
    trailingStop: 0.03
  });

  // Create live trading engine
  const engine = new LiveTradingEngine({
    strategy,
    dataFeed,
    executor,
    riskManager,
    initialCapital: 100000,
    symbols: ['AAPL', 'GOOGL', 'MSFT']
  });

  return engine;
}

/**
 * Run live trading with monitoring
 */
async function runLiveTrading() {
  const mode = process.env.TRADING_MODE || 'paper';

  if (mode === 'live') {
    console.log('⚠️  LIVE TRADING MODE - REAL MONEY AT RISK!');
    console.log('Make sure you understand the risks involved.');
  }

  const engine = await setupLiveTrading(mode);

  // Set up event handlers
  engine.on('trade', (trade) => {
    console.log(`📊 Trade executed: ${trade.action} ${trade.quantity} @ $${trade.price}`);
  });

  engine.on('position_update', (position) => {
    console.log(`📈 Position update: ${position.symbol} - ${position.quantity} shares, P&L: ${position.unrealizedPnl.toFixed(2)}`);
  });

  engine.on('risk_alert', (alert) => {
    console.warn(`⚠️  Risk Alert: ${alert.message}`);
  });

  engine.on('error', (error) => {
    console.error(`❌ Error: ${error.message}`);
  });

  // Start trading
  console.log('🔄 Starting live trading...');
  await engine.start();

  // Monitor performance every 5 minutes
  setInterval(() => {
    const performance = engine.getPerformance();
    console.log('\n📊 Performance Update:');
    console.log(`  Portfolio Value: $${performance.portfolioValue.toFixed(2)}`);
    console.log(`  Daily P&L: $${performance.dailyPnl.toFixed(2)} (${performance.dailyReturn.toFixed(2)}%)`);
    console.log(`  Total Return: ${performance.totalReturn.toFixed(2)}%`);
    console.log(`  Open Positions: ${performance.openPositions}`);
    console.log(`  Win Rate: ${performance.winRate.toFixed(2)}%`);
  }, 5 * 60 * 1000);

  // Graceful shutdown
  process.on('SIGINT', async () => {
    console.log('\n🛑 Shutting down...');
    await engine.stop();
    console.log('✅ Shutdown complete');
    process.exit(0);
  });
}

/**
 * Paper trading simulation
 */
async function runPaperTrading() {
  console.log('📄 Starting Paper Trading Simulation\n');

  const engine = await setupLiveTrading('paper');

  console.log('💡 This is a risk-free simulation using real market data');
  console.log('💡 No real money is at risk\n');

  await engine.start();

  // Auto-stop after 1 hour for demo
  setTimeout(async () => {
    console.log('\n⏱  Demo period ended');
    const performance = engine.getPerformance();

    console.log('\n📊 Final Results:');
    console.log(`Total Trades: ${performance.totalTrades}`);
    console.log(`Win Rate: ${performance.winRate.toFixed(2)}%`);
    console.log(`Total Return: ${performance.totalReturn.toFixed(2)}%`);
    console.log(`Sharpe Ratio: ${performance.sharpeRatio.toFixed(2)}`);

    await engine.stop();
    process.exit(0);
  }, 60 * 60 * 1000); // 1 hour
}

/**
 * Get starter code for live trading
 */
function getStarterCode() {
  return `const { LiveTradingEngine, Strategy, MarketDataFeed } = require('neural-trader');

class MyLiveStrategy extends Strategy {
  constructor() {
    super('My Live Strategy');
  }

  onBar(bar, portfolio, context) {
    // Implement your real-time trading logic
    // This runs every time new market data arrives
    return null;
  }

  onOrderFilled(order, portfolio) {
    console.log('Order filled:', order);
  }
}

async function main() {
  // Set up data feed
  const dataFeed = new MarketDataFeed({
    provider: 'alpaca',
    apiKey: process.env.BROKER_API_KEY,
    apiSecret: process.env.BROKER_API_SECRET,
    mode: 'paper' // Use 'live' for real trading
  });

  // Create engine
  const engine = new LiveTradingEngine({
    strategy: new MyLiveStrategy(),
    dataFeed,
    initialCapital: 100000,
    symbols: ['AAPL']
  });

  // Start trading
  await engine.start();
}

main().catch(console.error);
`;
}

module.exports = {
  setupLiveTrading,
  runLiveTrading,
  runPaperTrading,
  getStarterCode,
  AdaptiveTradingStrategy
};

// Run if called directly
if (require.main === module) {
  const mode = process.argv[2] || 'paper';

  if (mode === 'paper') {
    runPaperTrading().catch(console.error);
  } else {
    runLiveTrading().catch(console.error);
  }
}
