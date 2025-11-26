/**
 * Basic usage example for @neural-trader/example-market-microstructure
 */

import {
  createMarketMicrostructure,
  OrderBook,
  OrderFlow
} from '../src';

async function main() {
  console.log('=== Market Microstructure Analysis Example ===\n');

  // Initialize market microstructure analyzer
  const mm = await createMarketMicrostructure({
    agentDbPath: './examples/market-patterns.db',
    useSwarm: true,
    swarmConfig: {
      numAgents: 20,
      generations: 30,
      useOpenRouter: false
    }
  });

  console.log('✓ MarketMicrostructure initialized\n');

  // Simulate market data stream
  console.log('Analyzing market data stream...\n');

  for (let i = 0; i < 30; i++) {
    // Create mock order book
    const orderBook: OrderBook = {
      bids: [
        { price: 100.0 - i * 0.01, size: 1000 + Math.random() * 500, orders: 10 },
        { price: 99.9 - i * 0.01, size: 800 + Math.random() * 400, orders: 8 },
        { price: 99.8 - i * 0.01, size: 600 + Math.random() * 300, orders: 6 }
      ],
      asks: [
        { price: 100.1 - i * 0.01, size: 1000 + Math.random() * 500, orders: 10 },
        { price: 100.2 - i * 0.01, size: 800 + Math.random() * 400, orders: 8 },
        { price: 100.3 - i * 0.01, size: 600 + Math.random() * 300, orders: 6 }
      ],
      timestamp: Date.now() + i * 1000,
      symbol: 'BTCUSD'
    };

    // Create mock recent trades
    const trades: OrderFlow[] = [];
    for (let j = 0; j < 5; j++) {
      trades.push({
        type: Math.random() > 0.5 ? 'buy' : 'sell',
        price: 100 + (Math.random() - 0.5) * 0.5,
        size: Math.random() * 100 + 50,
        aggressor: Math.random() > 0.5 ? 'buyer' : 'seller',
        timestamp: Date.now() + i * 1000 + j * 200
      });
    }

    // Analyze order book
    const result = await mm.analyze(orderBook, trades);

    // Display metrics every 5 iterations
    if (i % 5 === 4) {
      console.log(`\nIteration ${i + 1}:`);
      console.log('Metrics:');
      console.log(`  Spread: ${result.metrics.bidAskSpread.toFixed(4)} (${result.metrics.spreadBps.toFixed(2)} bps)`);
      console.log(`  Imbalance: ${result.metrics.imbalance.toFixed(3)}`);
      console.log(`  Liquidity Score: ${result.metrics.liquidityScore.toFixed(3)}`);
      console.log(`  Order Flow Toxicity: ${result.metrics.orderFlowToxicity.toFixed(3)}`);
      console.log(`  VPIN: ${result.metrics.vpin.toFixed(3)}`);
      console.log(`  Net Flow: ${result.metrics.netFlow.toFixed(3)}`);

      if (result.pattern) {
        console.log('\nPattern Recognition:');
        if ('label' in result.pattern) {
          console.log(`  Recognized: ${result.pattern.label} (confidence: ${result.pattern.confidence.toFixed(2)})`);
        } else if ('prediction' in result.pattern) {
          console.log('  Prediction:');
          console.log(`    Price Move: ${result.pattern.prediction.priceMove.toFixed(4)}`);
          console.log(`    Spread Change: ${result.pattern.prediction.spreadChange.toFixed(4)}`);
          console.log(`    Liquidity Change: ${result.pattern.prediction.liquidityChange.toFixed(4)}`);
        }
      }

      if (result.anomaly) {
        console.log('\nAnomaly Detection:');
        console.log(`  Is Anomaly: ${result.anomaly.isAnomaly}`);
        console.log(`  Confidence: ${result.anomaly.confidence.toFixed(3)}`);
        console.log(`  Type: ${result.anomaly.anomalyType}`);
      }
    }

    // Simulate learning from outcomes
    if (i > 0 && i % 10 === 9) {
      const outcome = {
        priceMove: (Math.random() - 0.5) * 2,
        spreadChange: (Math.random() - 0.5) * 0.1,
        liquidityChange: (Math.random() - 0.5) * 0.3,
        timeHorizon: 5000
      };

      await mm.learn(outcome);
      console.log(`\n  ✓ Learned pattern from outcome`);
    }
  }

  // Explore features using swarm
  console.log('\n\nExploring feature space with swarm intelligence...');
  const featureSets = await mm.exploreFeatures();

  console.log(`\nDiscovered ${featureSets.length} feature sets:`);
  featureSets.slice(0, 5).forEach((fs, idx) => {
    console.log(`\n${idx + 1}. ${fs.name}:`);
    console.log(`   Features: ${fs.features.join(', ')}`);
    console.log(`   Importance: ${fs.importance.toFixed(3)}`);
    console.log(`   Performance:`);
    console.log(`     Accuracy: ${fs.performance.accuracy.toFixed(3)}`);
    console.log(`     Profitability: ${fs.performance.profitability.toFixed(3)}`);
    console.log(`     Sharpe Ratio: ${fs.performance.sharpeRatio.toFixed(3)}`);
  });

  // Optimize features for profitability
  console.log('\n\nOptimizing features for profitability...');
  const optimized = await mm.optimizeFeatures('profitability');

  console.log('\nOptimized Features:');
  console.log(`  Spread Trend: ${optimized.spreadTrend.toFixed(4)}`);
  console.log(`  Spread Volatility: ${optimized.spreadVolatility.toFixed(4)}`);
  console.log(`  Depth Imbalance: ${optimized.depthImbalance.toFixed(4)}`);
  console.log(`  Flow Persistence: ${optimized.flowPersistence.toFixed(4)}`);
  console.log(`  Toxicity Level: ${optimized.toxicityLevel.toFixed(4)}`);
  console.log(`  Price Efficiency: ${optimized.priceEfficiency.toFixed(4)}`);

  // Get final statistics
  console.log('\n\nFinal Statistics:');
  const stats = mm.getStatistics();

  console.log('\nAnalyzer:');
  console.log(`  Metrics Count: ${stats.analyzer.metricsCount}`);
  console.log(`  Order Flow Count: ${stats.analyzer.orderFlowCount}`);

  console.log('\nPattern Learner:');
  console.log(`  Total Patterns: ${stats.learner.totalPatterns}`);
  console.log(`  High Confidence: ${stats.learner.highConfidencePatterns}`);
  console.log(`  Average Confidence: ${stats.learner.avgConfidence.toFixed(3)}`);

  if (stats.learner.mostCommonLabels.length > 0) {
    console.log('\n  Most Common Patterns:');
    stats.learner.mostCommonLabels.slice(0, 5).forEach(({ label, count }) => {
      console.log(`    ${label}: ${count} occurrences`);
    });
  }

  console.log('\nSwarm:');
  console.log(`  Total Agents: ${stats.swarm.totalAgents}`);
  console.log(`  Average Performance: ${stats.swarm.avgPerformance.toFixed(3)}`);
  console.log('  Agents by Type:');
  Object.entries(stats.swarm.byType).forEach(([type, count]) => {
    console.log(`    ${type}: ${count}`);
  });

  if (stats.swarm.bestAgent) {
    console.log('\n  Best Agent:');
    console.log(`    ID: ${stats.swarm.bestAgent.id}`);
    console.log(`    Type: ${stats.swarm.bestAgent.type}`);
    console.log(`    Performance: ${stats.swarm.bestAgent.performance.toFixed(3)}`);
    console.log(`    Features: ${stats.swarm.bestAgent.features.slice(0, 5).join(', ')}...`);
  }

  // Get learned patterns
  const patterns = mm.getPatterns();
  console.log(`\n\nLearned ${patterns.length} patterns`);

  if (patterns.length > 0) {
    console.log('\nTop 5 Patterns:');
    patterns.slice(0, 5).forEach((pattern, idx) => {
      console.log(`\n${idx + 1}. ${pattern.label}:`);
      console.log(`   Confidence: ${pattern.confidence.toFixed(3)}`);
      console.log(`   Occurrences: ${pattern.metadata.occurrences}`);
      if (pattern.outcome) {
        console.log(`   Outcome:`);
        console.log(`     Price Move: ${pattern.outcome.priceMove.toFixed(4)}`);
        console.log(`     Spread Change: ${pattern.outcome.spreadChange.toFixed(4)}`);
        console.log(`     Liquidity Change: ${pattern.outcome.liquidityChange.toFixed(4)}`);
      }
    });
  }

  // Cleanup
  await mm.close();

  console.log('\n\n✓ Analysis complete!');
}

// Run example
main().catch(console.error);
