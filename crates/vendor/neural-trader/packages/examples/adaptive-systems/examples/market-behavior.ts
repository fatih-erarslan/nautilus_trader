/**
 * Market Behavior using Cellular Automata
 *
 * Demonstrates:
 * - Market sentiment propagation
 * - Herd behavior patterns
 * - Information cascades
 * - Market regime detection
 */

import {
  CellularAutomata,
  EmergenceDetector,
  type SystemState,
  type AutomatonRule
} from '../src';

/**
 * Market CA Rule:
 * States: 0 = neutral, 1 = bullish, 2 = bearish
 * Rules:
 * - Strong local sentiment influences neighbors
 * - Information cascade when critical mass reached
 * - Random noise for market uncertainty
 */
const MarketSentiment: AutomatonRule = {
  name: 'Market Sentiment',
  states: 3,
  neighborhoodType: 'moore',
  updateRule: (cell, neighbors) => {
    const bullish = neighbors.filter(n => n === 1).length;
    const bearish = neighbors.filter(n => n === 2).length;

    // Add noise (random events)
    const noise = Math.random();

    if (cell === 0) {
      // Neutral -> influenced by neighbors
      if (bullish > bearish + 2) return 1;
      if (bearish > bullish + 2) return 2;
      if (noise < 0.05) return Math.random() < 0.5 ? 1 : 2; // Random catalyst
      return 0;
    } else if (cell === 1) {
      // Bullish -> can turn bearish or neutral
      if (bearish > bullish + 1) return 2; // Sentiment shift
      if (bearish > bullish) return 0; // Profit taking
      if (noise < 0.02) return 2; // Black swan event
      return 1;
    } else {
      // Bearish -> can turn bullish or neutral
      if (bullish > bearish + 1) return 1; // Recovery
      if (bullish > bearish) return 0; // Bottom formation
      if (noise < 0.02) return 1; // Positive surprise
      return 2;
    }
  }
};

async function runMarketBehaviorSimulation() {
  console.log('📈 Market Behavior Simulation using Cellular Automata\n');

  // Create market grid (each cell is a trader/agent)
  const width = 100;
  const height = 100;

  const market = new CellularAutomata(
    { width, height, wrapEdges: true },
    MarketSentiment
  );

  // Initialize emergence detector
  const emergence = new EmergenceDetector(process.env.OPENAI_API_KEY);

  // Initialize with random sentiment
  console.log('Initializing market with random sentiment distribution...');
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const rand = Math.random();
      if (rand < 0.4) market.setCell(x, y, 0); // 40% neutral
      else if (rand < 0.7) market.setCell(x, y, 1); // 30% bullish
      else market.setCell(x, y, 2); // 30% bearish
    }
  }

  // Add some "information centers" (clusters of strong sentiment)
  console.log('Adding information centers (market makers)...');
  const centers = [
    { x: 20, y: 20, sentiment: 1, radius: 5 },
    { x: 80, y: 20, sentiment: 2, radius: 5 },
    { x: 50, y: 50, sentiment: 1, radius: 7 },
    { x: 20, y: 80, sentiment: 2, radius: 4 },
    { x: 80, y: 80, sentiment: 1, radius: 6 }
  ];

  for (const center of centers) {
    for (let dy = -center.radius; dy <= center.radius; dy++) {
      for (let dx = -center.radius; dx <= center.radius; dx++) {
        if (dx * dx + dy * dy <= center.radius * center.radius) {
          const x = center.x + dx;
          const y = center.y + dy;
          if (x >= 0 && x < width && y >= 0 && y < height) {
            market.setCell(x, y, center.sentiment);
          }
        }
      }
    }
  }

  // Run simulation
  console.log('\nRunning market simulation...');
  const generations = 200;
  const snapshotInterval = 20;

  let cascadeDetected = false;
  let regimeShiftDetected = false;

  for (let gen = 0; gen < generations; gen++) {
    await market.step();

    if (gen % snapshotInterval === 0) {
      const grid = market.getGrid();

      // Analyze market state
      const sentiment = analyzeMarketSentiment(grid);
      const clusters = detectClusters(grid);
      const stability = calculateStability(grid);

      console.log(`\nGeneration ${gen}:`);
      console.log(`  Bullish: ${sentiment.bullish}% | Neutral: ${sentiment.neutral}% | Bearish: ${sentiment.bearish}%`);
      console.log(`  Sentiment clusters: ${clusters.count}`);
      console.log(`  Average cluster size: ${clusters.avgSize.toFixed(1)}`);
      console.log(`  Market stability: ${(stability * 100).toFixed(1)}%`);

      // Detect market events
      if (!cascadeDetected && clusters.avgSize > 200) {
        console.log('\n  🌊 EMERGENCE: Information cascade detected!');
        cascadeDetected = true;
      }

      if (!regimeShiftDetected && (sentiment.bullish > 70 || sentiment.bearish > 70)) {
        const regime = sentiment.bullish > 70 ? 'BULL' : 'BEAR';
        console.log(`\n  📊 EMERGENCE: Market regime shift to ${regime} MARKET!`);
        regimeShiftDetected = true;
      }

      // Create system state for emergence detection
      const agents = [];
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          if (Math.random() < 0.1) { // Sample 10% of cells
            agents.push({
              id: `trader-${x}-${y}`,
              position: { x, y },
              state: { sentiment: grid[y][x] },
              neighbors: []
            });
          }
        }
      }

      const state: SystemState = {
        timestamp: Date.now(),
        agents,
        globalMetrics: {
          entropy: 1 - stability,
          order: stability,
          complexity: clusters.count / 50,
          connectivity: clusters.avgSize / 100
        }
      };

      await emergence.addState(state);
    }
  }

  // Final analysis
  console.log('\n' + '='.repeat(60));
  console.log('Final Market Analysis');
  console.log('='.repeat(60));

  const finalGrid = market.getGrid();
  const finalSentiment = analyzeMarketSentiment(finalGrid);

  console.log('\nFinal Market Sentiment:');
  console.log(`  Bullish traders: ${finalSentiment.bullish}%`);
  console.log(`  Neutral traders: ${finalSentiment.neutral}%`);
  console.log(`  Bearish traders: ${finalSentiment.bearish}%`);

  const marketBias = finalSentiment.bullish - finalSentiment.bearish;
  const biasDirection = marketBias > 0 ? 'BULLISH' : 'BEARISH';
  console.log(`\n  Overall market bias: ${biasDirection} (${Math.abs(marketBias).toFixed(1)}%)`);

  // Emergence metrics
  const metrics = emergence.getLatestMetrics();
  console.log('\nEmergence Metrics:');
  console.log(`  Self-Organization: ${(metrics.selfOrganization * 100).toFixed(1)}%`);
  console.log(`  Complexity: ${(metrics.complexity * 100).toFixed(1)}%`);
  console.log(`  Coherence: ${(metrics.coherence * 100).toFixed(1)}%`);
  console.log(`  Robustness: ${(metrics.robustness * 100).toFixed(1)}%`);
  console.log(`  Novelty: ${(metrics.novelty * 100).toFixed(1)}%`);

  const events = emergence.getEmergenceEvents();
  if (events.length > 0) {
    console.log('\nEmergence Events:');
    events.forEach((event, i) => {
      console.log(`\n${i + 1}. ${event.type} (confidence: ${(event.confidence * 100).toFixed(1)}%)`);
      console.log(`   ${event.description}`);
    });
  }

  console.log('\n✅ Simulation complete!');
  console.log('\n💡 Key Observations:');
  console.log('  - Sentiment propagates through the market like a wave');
  console.log('  - Information centers (market makers) influence local regions');
  console.log('  - Critical mass leads to information cascades (herd behavior)');
  console.log('  - Market can exhibit phase transitions (bull ↔ bear)');
  console.log('  - Random noise creates realistic market uncertainty');
}

/**
 * Analyze market sentiment distribution
 */
function analyzeMarketSentiment(grid: number[][]): {
  bullish: number;
  neutral: number;
  bearish: number;
} {
  let bullish = 0;
  let neutral = 0;
  let bearish = 0;
  let total = 0;

  for (const row of grid) {
    for (const cell of row) {
      if (cell === 0) neutral++;
      else if (cell === 1) bullish++;
      else if (cell === 2) bearish++;
      total++;
    }
  }

  return {
    bullish: (bullish / total) * 100,
    neutral: (neutral / total) * 100,
    bearish: (bearish / total) * 100
  };
}

/**
 * Detect sentiment clusters using flood fill
 */
function detectClusters(grid: number[][]): {
  count: number;
  avgSize: number;
} {
  const height = grid.length;
  const width = grid[0].length;
  const visited = Array(height).fill(0).map(() => Array(width).fill(false));
  const clusters: number[] = [];

  function floodFill(x: number, y: number, sentiment: number): number {
    if (x < 0 || x >= width || y < 0 || y >= height) return 0;
    if (visited[y][x] || grid[y][x] !== sentiment) return 0;

    visited[y][x] = true;
    let size = 1;

    // Check 4 neighbors
    size += floodFill(x + 1, y, sentiment);
    size += floodFill(x - 1, y, sentiment);
    size += floodFill(x, y + 1, sentiment);
    size += floodFill(x, y - 1, sentiment);

    return size;
  }

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (!visited[y][x] && grid[y][x] !== 0) {
        const size = floodFill(x, y, grid[y][x]);
        if (size > 5) { // Only count significant clusters
          clusters.push(size);
        }
      }
    }
  }

  const avgSize = clusters.length > 0
    ? clusters.reduce((sum, s) => sum + s, 0) / clusters.length
    : 0;

  return {
    count: clusters.length,
    avgSize
  };
}

/**
 * Calculate market stability (consistency over time)
 */
function calculateStability(grid: number[][]): number {
  const sentiment = analyzeMarketSentiment(grid);

  // Market is stable when one sentiment dominates
  const maxSentiment = Math.max(sentiment.bullish, sentiment.neutral, sentiment.bearish);

  return maxSentiment / 100;
}

// Run if called directly
if (require.main === module) {
  runMarketBehaviorSimulation().catch(console.error);
}

export { runMarketBehaviorSimulation };
