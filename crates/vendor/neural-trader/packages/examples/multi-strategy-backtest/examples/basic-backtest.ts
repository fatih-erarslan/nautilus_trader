/**
 * Basic backtesting example
 * Demonstrates simple multi-strategy backtest with synthetic data
 */

import { MultiStrategyBacktestSystem } from '../src';
import { BacktestConfig, MarketData } from '../src/types';

async function main() {
  console.log('🚀 Multi-Strategy Backtest Example\n');

  // Generate synthetic market data
  console.log('📊 Generating synthetic market data...');
  const marketData = generateSyntheticData(500, 100);
  console.log(`✅ Generated ${marketData.length} bars\n`);

  // Configure backtest
  const config: BacktestConfig = {
    startDate: new Date(marketData[0].timestamp),
    endDate: new Date(marketData[marketData.length - 1].timestamp),
    initialCapital: 100000,
    symbols: ['SYNTH'],
    strategies: [
      {
        name: 'momentum',
        type: 'momentum',
        initialWeight: 0.25,
        parameters: {
          lookbackPeriod: 20,
          entryThreshold: 0.02,
          exitThreshold: -0.01
        },
        enabled: true
      },
      {
        name: 'mean-reversion',
        type: 'mean-reversion',
        initialWeight: 0.25,
        parameters: {
          maPeriod: 20,
          stdDevMultiplier: 2.0,
          minDeviation: 0.015
        },
        enabled: true
      },
      {
        name: 'pairs-trading',
        type: 'pairs-trading',
        initialWeight: 0.25,
        parameters: {
          lookbackPeriod: 60,
          entryZScore: 2.0,
          exitZScore: 0.5
        },
        enabled: true
      },
      {
        name: 'market-making',
        type: 'market-making',
        initialWeight: 0.25,
        parameters: {
          spreadBps: 10,
          inventoryLimit: 1000
        },
        enabled: true
      }
    ],
    commission: 0.001, // 0.1%
    slippage: 0.0005, // 0.05%
    walkForwardPeriods: 3,
    rebalanceFrequency: 'weekly'
  };

  // Initialize system (without OpenRouter for this example)
  const system = new MultiStrategyBacktestSystem(config);
  await system.initialize();

  // Run complete workflow
  console.log('🎯 Starting complete backtest workflow...\n');
  const results = await system.runCompleteWorkflow(marketData);

  // Display results summary
  console.log('\n' + '='.repeat(80));
  console.log('📊 FINAL RESULTS SUMMARY');
  console.log('='.repeat(80));

  console.log('\n🏆 Best Strategy:');
  const bestStrategy = results.performances.reduce((best, curr) =>
    curr.sharpeRatio > best.sharpeRatio ? curr : best
  );
  console.log(`  ${bestStrategy.strategyName}`);
  console.log(`  Sharpe Ratio: ${bestStrategy.sharpeRatio.toFixed(2)}`);
  console.log(`  Total Return: ${(bestStrategy.totalReturn * 100).toFixed(2)}%`);
  console.log(`  Max Drawdown: ${(bestStrategy.maxDrawdown * 100).toFixed(2)}%`);

  console.log('\n📚 Learning Progress:');
  console.log(`  Episodes: ${results.learningStats.episodes}`);
  console.log(`  Total Reward: ${results.learningStats.totalReward.toFixed(2)}`);
  console.log(`  Experience Buffer: ${results.learningStats.experienceCount}`);

  console.log('\n🐝 Optimization Summary:');
  for (const [strategy, result] of results.optimizationResults.entries()) {
    console.log(`  ${strategy}:`);
    console.log(`    Score: ${result.bestScore.toFixed(4)}`);
    console.log(`    Evaluations: ${result.evaluations}`);
  }

  console.log('\n✅ Example complete!\n');
}

/**
 * Generate synthetic market data with realistic patterns
 */
function generateSyntheticData(bars: number, startPrice: number): MarketData[] {
  const data: MarketData[] = [];
  let price = startPrice;
  const startTime = Date.now() - (bars * 24 * 60 * 60 * 1000);

  // Add some regime changes
  let trend = 0.001; // Initial uptrend
  let volatility = 0.02;

  for (let i = 0; i < bars; i++) {
    // Change regime occasionally
    if (i % 100 === 0) {
      trend = (Math.random() - 0.5) * 0.002;
      volatility = 0.01 + (Math.random() * 0.03);
    }

    // Random walk with drift and volatility
    const change = trend + (Math.random() - 0.5) * volatility;
    price *= (1 + change);

    data.push({
      timestamp: startTime + (i * 24 * 60 * 60 * 1000),
      symbol: 'SYNTH',
      open: price * (1 + (Math.random() - 0.5) * 0.005),
      high: price * (1 + Math.random() * 0.01),
      low: price * (1 - Math.random() * 0.01),
      close: price,
      volume: Math.floor(Math.random() * 1000000) + 500000
    });
  }

  return data;
}

// Run example
if (require.main === module) {
  main().catch(console.error);
}
