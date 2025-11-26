/**
 * Neural Trader - Quick Start Example
 *
 * This example shows the basics of using Neural Trader:
 * - Loading market data
 * - Creating a simple strategy
 * - Running a backtest
 */

const {
  BacktestEngine,
  Strategy,
  TechnicalIndicators,
  RiskManager
} = require('neural-trader');

/**
 * Simple Moving Average Crossover Strategy
 */
class SimpleMAStrategy extends Strategy {
  constructor() {
    super('Simple MA Crossover');
    this.shortPeriod = 20;
    this.longPeriod = 50;
  }

  onBar(bar, portfolio) {
    // Calculate moving averages
    const shortMA = TechnicalIndicators.sma(bar.close, this.shortPeriod);
    const longMA = TechnicalIndicators.sma(bar.close, this.longPeriod);

    // Trading logic
    if (shortMA > longMA && !portfolio.hasPosition()) {
      // Buy signal
      return {
        action: 'buy',
        quantity: Math.floor(portfolio.cash / bar.close),
        price: bar.close
      };
    } else if (shortMA < longMA && portfolio.hasPosition()) {
      // Sell signal
      return {
        action: 'sell',
        quantity: portfolio.getPosition().quantity,
        price: bar.close
      };
    }

    return null; // No action
  }
}

/**
 * Run the backtest
 */
async function runQuickStart() {
  console.log('🚀 Neural Trader Quick Start\n');

  // Create strategy
  const strategy = new SimpleMAStrategy();

  // Create backtest engine
  const engine = new BacktestEngine({
    initialCapital: 100000,
    startDate: '2023-01-01',
    endDate: '2023-12-31',
    symbol: 'AAPL'
  });

  // Add risk management
  const riskManager = new RiskManager({
    maxPositionSize: 0.1, // Max 10% of portfolio per position
    stopLoss: 0.05,       // 5% stop loss
    takeProfit: 0.15      // 15% take profit
  });

  engine.setRiskManager(riskManager);

  // Run backtest
  console.log('📊 Running backtest...');
  const results = await engine.run(strategy);

  // Display results
  console.log('\n📈 Backtest Results:');
  console.log(`Total Return: ${results.totalReturn.toFixed(2)}%`);
  console.log(`Sharpe Ratio: ${results.sharpeRatio.toFixed(2)}`);
  console.log(`Max Drawdown: ${results.maxDrawdown.toFixed(2)}%`);
  console.log(`Win Rate: ${results.winRate.toFixed(2)}%`);
  console.log(`Total Trades: ${results.totalTrades}`);
  console.log(`Avg Trade Duration: ${results.avgTradeDuration} bars`);

  // Trade-by-trade analysis
  console.log('\n📋 Sample Trades:');
  results.trades.slice(0, 5).forEach((trade, idx) => {
    console.log(`${idx + 1}. ${trade.type} at $${trade.entryPrice.toFixed(2)} -> $${trade.exitPrice.toFixed(2)} (${trade.pnl > 0 ? '+' : ''}${trade.pnl.toFixed(2)}%)`);
  });

  return results;
}

/**
 * Get starter code for project initialization
 */
function getStarterCode() {
  return `const { BacktestEngine, Strategy } = require('neural-trader');

class MyStrategy extends Strategy {
  constructor() {
    super('My Trading Strategy');
  }

  onBar(bar, portfolio) {
    // Implement your trading logic here
    // Return { action: 'buy'|'sell', quantity, price } or null
    return null;
  }
}

async function main() {
  const strategy = new MyStrategy();
  const engine = new BacktestEngine({
    initialCapital: 100000,
    startDate: '2023-01-01',
    endDate: '2023-12-31',
    symbol: 'AAPL'
  });

  const results = await engine.run(strategy);
  console.log('Backtest completed:', results);
}

main().catch(console.error);
`;
}

// Export for use in CLI
module.exports = {
  runQuickStart,
  getStarterCode,
  SimpleMAStrategy
};

// Run if called directly
if (require.main === module) {
  runQuickStart().catch(console.error);
}
