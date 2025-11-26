/**
 * Neural Trader - Advanced Backtesting Example
 *
 * This example demonstrates:
 * - Multiple strategies
 * - Advanced risk management
 * - Performance analysis
 * - Strategy optimization
 */

const {
  BacktestEngine,
  Strategy,
  TechnicalIndicators,
  RiskManager,
  PerformanceAnalyzer
} = require('neural-trader');

/**
 * Advanced Mean Reversion Strategy
 */
class MeanReversionStrategy extends Strategy {
  constructor(options = {}) {
    super('Mean Reversion');
    this.lookback = options.lookback || 20;
    this.threshold = options.threshold || 2.0;
  }

  onBar(bar, portfolio, context) {
    const prices = context.getPriceHistory(this.lookback);

    if (prices.length < this.lookback) return null;

    // Calculate Bollinger Bands
    const { upper, middle, lower } = TechnicalIndicators.bollingerBands(
      prices,
      this.lookback,
      this.threshold
    );

    const currentPrice = bar.close;
    const position = portfolio.getPosition();

    // Mean reversion logic
    if (currentPrice < lower && !position) {
      // Price below lower band - buy signal
      return {
        action: 'buy',
        quantity: Math.floor(portfolio.cash * 0.95 / currentPrice),
        price: currentPrice,
        reason: 'Oversold condition'
      };
    } else if (currentPrice > middle && position) {
      // Price returned to mean - sell signal
      return {
        action: 'sell',
        quantity: position.quantity,
        price: currentPrice,
        reason: 'Mean reversion complete'
      };
    } else if (currentPrice > upper && position) {
      // Stop loss at upper band
      return {
        action: 'sell',
        quantity: position.quantity,
        price: currentPrice,
        reason: 'Stop loss - overbought'
      };
    }

    return null;
  }
}

/**
 * Momentum Strategy with RSI
 */
class MomentumStrategy extends Strategy {
  constructor(options = {}) {
    super('Momentum RSI');
    this.rsiPeriod = options.rsiPeriod || 14;
    this.oversold = options.oversold || 30;
    this.overbought = options.overbought || 70;
  }

  onBar(bar, portfolio, context) {
    const prices = context.getPriceHistory(this.rsiPeriod + 1);

    if (prices.length < this.rsiPeriod + 1) return null;

    const rsi = TechnicalIndicators.rsi(prices, this.rsiPeriod);
    const position = portfolio.getPosition();

    if (rsi < this.oversold && !position) {
      return {
        action: 'buy',
        quantity: Math.floor(portfolio.cash * 0.95 / bar.close),
        price: bar.close,
        reason: `RSI oversold: ${rsi.toFixed(2)}`
      };
    } else if (rsi > this.overbought && position) {
      return {
        action: 'sell',
        quantity: position.quantity,
        price: bar.close,
        reason: `RSI overbought: ${rsi.toFixed(2)}`
      };
    }

    return null;
  }
}

/**
 * Run advanced backtesting with multiple strategies
 */
async function runAdvancedBacktest() {
  console.log('🔬 Advanced Backtesting Example\n');

  const symbols = ['AAPL', 'GOOGL', 'MSFT'];
  const strategies = [
    new MeanReversionStrategy({ lookback: 20, threshold: 2.0 }),
    new MeanReversionStrategy({ lookback: 30, threshold: 2.5 }),
    new MomentumStrategy({ rsiPeriod: 14, oversold: 30, overbought: 70 }),
    new MomentumStrategy({ rsiPeriod: 21, oversold: 35, overbought: 65 })
  ];

  const results = [];

  // Test each strategy on each symbol
  for (const symbol of symbols) {
    console.log(`\n📊 Testing ${symbol}...`);

    for (const strategy of strategies) {
      const engine = new BacktestEngine({
        initialCapital: 100000,
        startDate: '2022-01-01',
        endDate: '2023-12-31',
        symbol,
        commission: 0.001, // 0.1% commission
        slippage: 0.0005   // 0.05% slippage
      });

      // Advanced risk management
      const riskManager = new RiskManager({
        maxPositionSize: 0.2,
        maxPortfolioRisk: 0.05,
        stopLoss: 0.05,
        takeProfit: 0.15,
        trailingStop: 0.03,
        maxDrawdown: 0.20
      });

      engine.setRiskManager(riskManager);

      const result = await engine.run(strategy);

      results.push({
        symbol,
        strategy: strategy.name,
        ...result
      });

      console.log(`  ${strategy.name}: Return ${result.totalReturn.toFixed(2)}%, Sharpe ${result.sharpeRatio.toFixed(2)}`);
    }
  }

  // Performance analysis
  console.log('\n📈 Performance Analysis\n');

  const analyzer = new PerformanceAnalyzer(results);

  // Best performers
  const bestByReturn = analyzer.rankByMetric('totalReturn');
  console.log('🏆 Top 3 by Total Return:');
  bestByReturn.slice(0, 3).forEach((r, idx) => {
    console.log(`${idx + 1}. ${r.strategy} on ${r.symbol}: ${r.totalReturn.toFixed(2)}%`);
  });

  const bestBySharpe = analyzer.rankByMetric('sharpeRatio');
  console.log('\n🏆 Top 3 by Sharpe Ratio:');
  bestBySharpe.slice(0, 3).forEach((r, idx) => {
    console.log(`${idx + 1}. ${r.strategy} on ${r.symbol}: ${r.sharpeRatio.toFixed(2)}`);
  });

  // Statistical analysis
  const stats = analyzer.getStatistics();
  console.log('\n📊 Overall Statistics:');
  console.log(`Average Return: ${stats.avgReturn.toFixed(2)}%`);
  console.log(`Std Deviation: ${stats.stdReturn.toFixed(2)}%`);
  console.log(`Best Strategy: ${stats.bestStrategy}`);
  console.log(`Most Consistent: ${stats.mostConsistent}`);

  return results;
}

/**
 * Strategy optimization example
 */
async function optimizeStrategy() {
  console.log('\n🔧 Strategy Optimization\n');

  const parameterRanges = {
    lookback: [10, 15, 20, 25, 30],
    threshold: [1.5, 2.0, 2.5, 3.0]
  };

  let bestResult = null;
  let bestParams = null;

  for (const lookback of parameterRanges.lookback) {
    for (const threshold of parameterRanges.threshold) {
      const strategy = new MeanReversionStrategy({ lookback, threshold });

      const engine = new BacktestEngine({
        initialCapital: 100000,
        startDate: '2023-01-01',
        endDate: '2023-12-31',
        symbol: 'AAPL'
      });

      const result = await engine.run(strategy);

      if (!bestResult || result.sharpeRatio > bestResult.sharpeRatio) {
        bestResult = result;
        bestParams = { lookback, threshold };
      }
    }
  }

  console.log('✅ Optimization Complete');
  console.log(`Best Parameters: lookback=${bestParams.lookback}, threshold=${bestParams.threshold}`);
  console.log(`Sharpe Ratio: ${bestResult.sharpeRatio.toFixed(2)}`);
  console.log(`Total Return: ${bestResult.totalReturn.toFixed(2)}%`);

  return { bestResult, bestParams };
}

/**
 * Get starter code for advanced backtesting
 */
function getStarterCode() {
  return `const { BacktestEngine, Strategy, TechnicalIndicators } = require('neural-trader');

class MyAdvancedStrategy extends Strategy {
  constructor(options = {}) {
    super('My Advanced Strategy');
    this.period = options.period || 20;
  }

  onBar(bar, portfolio, context) {
    const prices = context.getPriceHistory(this.period);
    if (prices.length < this.period) return null;

    // Add your advanced logic here
    const sma = TechnicalIndicators.sma(prices, this.period);
    const rsi = TechnicalIndicators.rsi(prices, 14);

    // Example: Combine multiple indicators
    // Return trading signal or null

    return null;
  }
}

async function main() {
  const strategy = new MyAdvancedStrategy({ period: 20 });
  const engine = new BacktestEngine({
    initialCapital: 100000,
    startDate: '2023-01-01',
    endDate: '2023-12-31',
    symbol: 'AAPL'
  });

  const results = await engine.run(strategy);
  console.log('Advanced backtest completed:', results);
}

main().catch(console.error);
`;
}

module.exports = {
  runAdvancedBacktest,
  optimizeStrategy,
  getStarterCode,
  MeanReversionStrategy,
  MomentumStrategy
};

// Run if called directly
if (require.main === module) {
  (async () => {
    await runAdvancedBacktest();
    await optimizeStrategy();
  })().catch(console.error);
}
