/**
 * Advanced anomaly detection example
 */

import {
  createMarketMicrostructure,
  OrderBook,
  OrderFlow
} from '../src';

async function main() {
  console.log('=== Advanced Anomaly Detection Example ===\n');

  const mm = await createMarketMicrostructure({
    agentDbPath: './examples/anomaly-patterns.db',
    useSwarm: true,
    swarmConfig: {
      numAgents: 30,
      generations: 20,
      useOpenRouter: false // Set to true with API key for enhanced detection
    }
  });

  console.log('✓ MarketMicrostructure initialized with swarm\n');

  // Simulate normal market conditions
  console.log('Phase 1: Normal Market Conditions\n');

  for (let i = 0; i < 20; i++) {
    const orderBook: OrderBook = {
      bids: [
        { price: 100.0, size: 1000, orders: 10 },
        { price: 99.9, size: 800, orders: 8 },
        { price: 99.8, size: 600, orders: 6 }
      ],
      asks: [
        { price: 100.1, size: 1000, orders: 10 },
        { price: 100.2, size: 800, orders: 8 },
        { price: 100.3, size: 600, orders: 6 }
      ],
      timestamp: Date.now() + i * 1000,
      symbol: 'BTCUSD'
    };

    const result = await mm.analyze(orderBook);

    if (i % 5 === 4) {
      console.log(`Iteration ${i + 1}: Normal conditions`);
      if (result.anomaly) {
        console.log(`  Anomaly: ${result.anomaly.isAnomaly} (confidence: ${result.anomaly.confidence.toFixed(3)})`);
      }
    }
  }

  // Inject anomaly: Wide spread
  console.log('\n\nPhase 2: Wide Spread Anomaly\n');

  for (let i = 0; i < 5; i++) {
    const orderBook: OrderBook = {
      bids: [
        { price: 100.0, size: 1000, orders: 10 },
        { price: 99.5, size: 800, orders: 8 },
        { price: 99.0, size: 600, orders: 6 }
      ],
      asks: [
        { price: 101.0, size: 1000, orders: 10 }, // Wide spread!
        { price: 101.5, size: 800, orders: 8 },
        { price: 102.0, size: 600, orders: 6 }
      ],
      timestamp: Date.now() + (20 + i) * 1000,
      symbol: 'BTCUSD'
    };

    const result = await mm.analyze(orderBook);

    console.log(`Iteration ${21 + i}:`);
    console.log(`  Spread: ${result.metrics.bidAskSpread.toFixed(4)} (${result.metrics.spreadBps.toFixed(2)} bps)`);
    if (result.anomaly) {
      console.log(`  Anomaly Detected: ${result.anomaly.isAnomaly}`);
      console.log(`  Confidence: ${result.anomaly.confidence.toFixed(3)}`);
      console.log(`  Type: ${result.anomaly.anomalyType}`);
      console.log(`  Explanation: ${result.anomaly.explanation}`);
    }
  }

  // Inject anomaly: Toxic flow
  console.log('\n\nPhase 3: Toxic Order Flow\n');

  for (let i = 0; i < 10; i++) {
    const orderBook: OrderBook = {
      bids: [
        { price: 99.9, size: 1000, orders: 10 },
        { price: 99.8, size: 800, orders: 8 },
        { price: 99.7, size: 600, orders: 6 }
      ],
      asks: [
        { price: 100.0, size: 1000, orders: 10 },
        { price: 100.1, size: 800, orders: 8 },
        { price: 100.2, size: 600, orders: 6 }
      ],
      timestamp: Date.now() + (25 + i) * 1000,
      symbol: 'BTCUSD'
    };

    // Create aggressive one-sided flow
    const trades: OrderFlow[] = [];
    for (let j = 0; j < 10; j++) {
      trades.push({
        type: 'buy',
        price: 100 + j * 0.01,
        size: 200,
        aggressor: 'buyer',
        timestamp: Date.now() + (25 + i) * 1000 + j * 100
      });
    }

    const result = await mm.analyze(orderBook, trades);

    if (i % 3 === 2) {
      console.log(`\nIteration ${26 + i}:`);
      console.log(`  Order Flow Toxicity: ${result.metrics.orderFlowToxicity.toFixed(3)}`);
      console.log(`  VPIN: ${result.metrics.vpin.toFixed(3)}`);
      console.log(`  Buy Pressure: ${result.metrics.buyPressure.toFixed(3)}`);
      if (result.anomaly) {
        console.log(`  Anomaly Detected: ${result.anomaly.isAnomaly}`);
        console.log(`  Confidence: ${result.anomaly.confidence.toFixed(3)}`);
        console.log(`  Type: ${result.anomaly.anomalyType}`);
      }
    }
  }

  // Inject anomaly: Extreme imbalance
  console.log('\n\nPhase 4: Extreme Order Book Imbalance\n');

  for (let i = 0; i < 5; i++) {
    const orderBook: OrderBook = {
      bids: [
        { price: 100.0, size: 5000, orders: 50 }, // Much larger bid side
        { price: 99.9, size: 4000, orders: 40 },
        { price: 99.8, size: 3000, orders: 30 }
      ],
      asks: [
        { price: 100.1, size: 200, orders: 2 }, // Thin ask side
        { price: 100.2, size: 150, orders: 2 },
        { price: 100.3, size: 100, orders: 1 }
      ],
      timestamp: Date.now() + (35 + i) * 1000,
      symbol: 'BTCUSD'
    };

    const result = await mm.analyze(orderBook);

    console.log(`\nIteration ${36 + i}:`);
    console.log(`  Bid Depth: ${result.metrics.bidDepth.toFixed(0)}`);
    console.log(`  Ask Depth: ${result.metrics.askDepth.toFixed(0)}`);
    console.log(`  Imbalance: ${result.metrics.imbalance.toFixed(3)}`);
    if (result.anomaly) {
      console.log(`  Anomaly Detected: ${result.anomaly.isAnomaly}`);
      console.log(`  Confidence: ${result.anomaly.confidence.toFixed(3)}`);
      console.log(`  Type: ${result.anomaly.anomalyType}`);
      console.log(`  Explanation: ${result.anomaly.explanation}`);
    }
  }

  // Return to normal
  console.log('\n\nPhase 5: Return to Normal\n');

  for (let i = 0; i < 10; i++) {
    const orderBook: OrderBook = {
      bids: [
        { price: 100.0, size: 1000, orders: 10 },
        { price: 99.9, size: 800, orders: 8 },
        { price: 99.8, size: 600, orders: 6 }
      ],
      asks: [
        { price: 100.1, size: 1000, orders: 10 },
        { price: 100.2, size: 800, orders: 8 },
        { price: 100.3, size: 600, orders: 6 }
      ],
      timestamp: Date.now() + (40 + i) * 1000,
      symbol: 'BTCUSD'
    };

    const result = await mm.analyze(orderBook);

    if (i % 3 === 2) {
      console.log(`Iteration ${41 + i}: Returning to normal`);
      if (result.anomaly) {
        console.log(`  Anomaly: ${result.anomaly.isAnomaly} (confidence: ${result.anomaly.confidence.toFixed(3)})`);
      }
    }
  }

  // Final statistics
  console.log('\n\nFinal Statistics:');
  const stats = mm.getStatistics();

  console.log(`\nTotal Metrics Analyzed: ${stats.analyzer.metricsCount}`);
  console.log(`Total Order Flow Events: ${stats.analyzer.orderFlowCount}`);
  console.log(`\nSwarm Agents: ${stats.swarm.totalAgents}`);
  console.log(`Average Agent Performance: ${stats.swarm.avgPerformance.toFixed(3)}`);

  await mm.close();

  console.log('\n✓ Anomaly detection analysis complete!');
}

main().catch(console.error);
