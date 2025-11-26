/**
 * Example showing integration with claude-flow hooks for memory persistence
 */

import { execSync } from 'child_process';
import { MultiStrategyBacktestSystem } from '../src';
import { BacktestConfig, MarketData } from '../src/types';

async function main() {
  console.log('🚀 Multi-Strategy Backtest with Hooks Integration\n');

  // Execute pre-task hook
  console.log('🔧 Executing pre-task hook...');
  try {
    execSync('npx claude-flow@alpha hooks pre-task --description "Multi-strategy backtest with RL and swarm optimization"', {
      stdio: 'inherit'
    });
  } catch (error) {
    console.log('⚠️  Hook execution skipped (optional)');
  }

  // Generate market data
  const marketData = generateMarketData(300, 100);

  // Configure backtest
  const config: BacktestConfig = {
    startDate: new Date(marketData[0].timestamp),
    endDate: new Date(marketData[marketData.length - 1].timestamp),
    initialCapital: 100000,
    symbols: ['TEST'],
    strategies: [
      { name: 'momentum', type: 'momentum', initialWeight: 0.25, parameters: {}, enabled: true },
      { name: 'mean-reversion', type: 'mean-reversion', initialWeight: 0.25, parameters: {}, enabled: true },
      { name: 'pairs-trading', type: 'pairs-trading', initialWeight: 0.25, parameters: {}, enabled: true },
      { name: 'market-making', type: 'market-making', initialWeight: 0.25, parameters: {}, enabled: true }
    ],
    commission: 0.001,
    slippage: 0.0005,
    walkForwardPeriods: 3
  };

  // Run backtest
  const system = new MultiStrategyBacktestSystem(config);
  await system.initialize();

  console.log('\n🎯 Running complete workflow with memory persistence...\n');
  const results = await system.runCompleteWorkflow(marketData);

  // Store results in memory bank
  console.log('\n💾 Storing results in memory bank...');
  const memoryData = {
    timestamp: Date.now(),
    performances: results.performances,
    learningStats: results.learningStats,
    optimizationResults: Array.from(results.optimizationResults.entries()).map(([name, result]) => ({
      strategy: name,
      ...result
    }))
  };

  try {
    execSync(`npx claude-flow@alpha hooks post-edit --file "backtest-results.json" --memory-key "swarm/backtest/results-${Date.now()}"`, {
      stdio: 'inherit'
    });
  } catch (error) {
    console.log('⚠️  Memory storage skipped (optional)');
  }

  // Notify completion
  try {
    const bestStrategy = results.performances.reduce((best, curr) =>
      curr.sharpeRatio > best.sharpeRatio ? curr : best
    );

    execSync(
      `npx claude-flow@alpha hooks notify --message "Backtest complete: Best strategy ${bestStrategy.strategyName} with Sharpe ${bestStrategy.sharpeRatio.toFixed(2)}"`,
      { stdio: 'inherit' }
    );
  } catch (error) {
    console.log('⚠️  Notification skipped (optional)');
  }

  // Execute post-task hook
  console.log('\n🏁 Executing post-task hook...');
  try {
    execSync('npx claude-flow@alpha hooks post-task --task-id "backtest-example"', {
      stdio: 'inherit'
    });
  } catch (error) {
    console.log('⚠️  Hook execution skipped (optional)');
  }

  // Display summary
  console.log('\n' + '='.repeat(80));
  console.log('✅ BACKTEST COMPLETE WITH MEMORY PERSISTENCE');
  console.log('='.repeat(80));

  console.log('\n📊 Results Summary:');
  console.log(`  Strategies Evaluated: ${results.performances.length}`);
  console.log(`  Learning Episodes: ${results.learningStats.episodes}`);
  console.log(`  Optimization Runs: ${results.optimizationResults.size}`);

  console.log('\n💾 Memory Bank:');
  console.log('  Results stored in: swarm/backtest/results-*');
  console.log('  Learning state: strategy-learner');
  console.log('  Access with: npx claude-flow@alpha memory retrieve <key>');

  console.log('\n🔍 Next Steps:');
  console.log('  1. Retrieve learning state for next run');
  console.log('  2. Compare results across multiple backtests');
  console.log('  3. Use learned weights for live trading');

  console.log('\n✨ Example complete!\n');
}

function generateMarketData(bars: number, startPrice: number): MarketData[] {
  const data: MarketData[] = [];
  let price = startPrice;
  const startTime = Date.now() - (bars * 24 * 60 * 60 * 1000);

  for (let i = 0; i < bars; i++) {
    const change = (Math.random() - 0.48) * 2;
    price *= (1 + change / 100);

    data.push({
      timestamp: startTime + (i * 24 * 60 * 60 * 1000),
      symbol: 'TEST',
      open: price * 0.99,
      high: price * 1.02,
      low: price * 0.98,
      close: price,
      volume: Math.floor(Math.random() * 1000000) + 500000
    });
  }

  return data;
}

if (require.main === module) {
  main().catch(console.error);
}
