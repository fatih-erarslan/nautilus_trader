/**
 * Swarm-Based Portfolio Exploration Example
 * Demonstrates benchmark swarm, constraint exploration, and market regime testing
 */

import {
  PortfolioOptimizationSwarm,
  BenchmarkConfig,
} from '../src/benchmark-swarm.js';
import { SelfLearningOptimizer } from '../src/self-learning.js';
import { Asset } from '../src/optimizer.js';

async function main() {
  console.log('🚀 Neural Trader - Portfolio Optimization Swarm Example\n');
  console.log('='.repeat(70) + '\n');

  // Define portfolio assets
  const assets: Asset[] = [
    { symbol: 'AAPL', expectedReturn: 0.12, volatility: 0.20 },
    { symbol: 'GOOGL', expectedReturn: 0.15, volatility: 0.25 },
    { symbol: 'MSFT', expectedReturn: 0.11, volatility: 0.18 },
    { symbol: 'AMZN', expectedReturn: 0.16, volatility: 0.28 },
    { symbol: 'NVDA', expectedReturn: 0.18, volatility: 0.35 },
    { symbol: 'META', expectedReturn: 0.14, volatility: 0.30 },
  ];

  const correlationMatrix = [
    [1.00, 0.65, 0.70, 0.60, 0.55, 0.62],
    [0.65, 1.00, 0.68, 0.72, 0.58, 0.66],
    [0.70, 0.68, 1.00, 0.64, 0.52, 0.60],
    [0.60, 0.72, 0.64, 1.00, 0.60, 0.68],
    [0.55, 0.58, 0.52, 0.60, 1.00, 0.64],
    [0.62, 0.66, 0.60, 0.68, 0.64, 1.00],
  ];

  const marketCapWeights = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10];

  // Initialize self-learning optimizer
  console.log('🧠 Initializing Self-Learning Optimizer...');
  const learningOptimizer = new SelfLearningOptimizer(
    './swarm-portfolio-memory.db',
    'swarm-portfolio',
  );
  await learningOptimizer.initialize();
  console.log('✅ Learning optimizer ready\n');

  // Initialize swarm (without OpenRouter for this example)
  const swarm = new PortfolioOptimizationSwarm(undefined, learningOptimizer);

  // Example 1: Comprehensive Benchmark
  console.log('1️⃣  COMPREHENSIVE ALGORITHM BENCHMARK');
  console.log('-'.repeat(70));

  const benchmarkConfig: BenchmarkConfig = {
    algorithms: ['mean-variance', 'risk-parity', 'black-litterman', 'multi-objective'],
    constraintVariations: [
      { minWeight: 0.05, maxWeight: 0.40 },
      { minWeight: 0.10, maxWeight: 0.35 },
      { minWeight: 0.08, maxWeight: 0.45 },
    ],
    assets,
    correlationMatrix,
    marketCapWeights,
    historicalReturns: generateHistoricalReturns(assets.length, 252),
  };

  const benchmarkInsights = await swarm.runBenchmark(benchmarkConfig);

  console.log(swarm.generateReport(benchmarkInsights));

  // Example 2: Constraint Space Exploration
  console.log('\n2️⃣  CONSTRAINT SPACE EXPLORATION');
  console.log('-'.repeat(70));

  const constraintInsights = await swarm.exploreConstraints(
    benchmarkConfig,
    {
      minWeight: [0.02, 0.15],
      maxWeight: [0.30, 0.60],
      targetReturn: [0.10, 0.18],
    },
    15, // 15 different constraint combinations
  );

  console.log('🎯 Optimal Constraints Found:');
  console.log(`Best Algorithm: ${constraintInsights.bestAlgorithm}`);
  console.log(`Best Sharpe: ${constraintInsights.bestResult.result.sharpeRatio.toFixed(4)}`);
  console.log(`Optimal Weights:`);
  assets.forEach((asset, i) => {
    const weight = constraintInsights.bestResult.result.weights[i];
    console.log(`  ${asset.symbol}: ${(weight * 100).toFixed(2)}%`);
  });
  console.log();

  // Example 3: Market Regime Comparison
  console.log('3️⃣  MARKET REGIME COMPARISON');
  console.log('-'.repeat(70));

  const regimes = [
    { name: 'Bull Market', volatilityMultiplier: 0.7, returnMultiplier: 1.5 },
    { name: 'Normal Market', volatilityMultiplier: 1.0, returnMultiplier: 1.0 },
    { name: 'Bear Market', volatilityMultiplier: 1.8, returnMultiplier: 0.4 },
    { name: 'Crisis', volatilityMultiplier: 3.0, returnMultiplier: -0.5 },
  ];

  const regimeResults = await swarm.compareMarketRegimes(benchmarkConfig, regimes);

  console.log('📈 Performance Across Market Regimes:\n');

  for (const [regimeName, insights] of Object.entries(regimeResults)) {
    console.log(`${regimeName}:`);
    console.log(`  Best Algorithm: ${insights.bestAlgorithm}`);
    console.log(`  Sharpe Ratio: ${insights.bestResult.result.sharpeRatio.toFixed(4)}`);
    console.log(`  Expected Return: ${(insights.bestResult.result.expectedReturn * 100).toFixed(2)}%`);
    console.log(`  Risk: ${(insights.bestResult.result.risk * 100).toFixed(2)}%`);
    console.log();
  }

  // Example 4: Self-Learning Integration
  console.log('4️⃣  SELF-LEARNING OPTIMIZATION');
  console.log('-'.repeat(70));

  console.log('Training from benchmark results...');

  // Simulate learning from best results
  const bestResult = benchmarkInsights.bestResult.result;
  const mockPerformance = {
    sharpeRatio: bestResult.sharpeRatio * 1.1, // Slightly better actual performance
    maxDrawdown: 0.12,
    volatility: bestResult.risk,
    cumulativeReturn: bestResult.expectedReturn * 1.2,
    winRate: 0.65,
    informationRatio: bestResult.sharpeRatio * 0.85,
  };

  const marketConditions = {
    volatility: 0.22,
    trend: 1,
    correlation: 0.65,
  };

  const newProfile = await learningOptimizer.learn(
    bestResult,
    mockPerformance,
    marketConditions,
  );

  console.log('🧠 Updated Risk Profile:');
  console.log(`  Risk Aversion: ${newProfile.riskAversion.toFixed(2)}`);
  console.log(`  Target Return: ${(newProfile.targetReturn * 100).toFixed(2)}%`);
  console.log(`  Max Drawdown: ${(newProfile.maxDrawdown * 100).toFixed(2)}%`);
  console.log(`  Preferred Algorithm: ${newProfile.preferredAlgorithm}`);
  console.log(`  Diversification Preference: ${newProfile.diversificationPreference.toFixed(2)}`);
  console.log();

  // Get recommendation based on current conditions
  const recommendedProfile = await learningOptimizer.getRecommendedProfile(marketConditions);

  console.log('💡 Recommended Strategy:');
  console.log(`  Algorithm: ${recommendedProfile.preferredAlgorithm}`);
  console.log(`  Target Return: ${(recommendedProfile.targetReturn * 100).toFixed(2)}%`);
  console.log(`  Max Acceptable Drawdown: ${(recommendedProfile.maxDrawdown * 100).toFixed(2)}%`);
  console.log();

  // Export learning data
  const learningData = await learningOptimizer.exportLearningData();
  if (learningData) {
    console.log('📊 Learning Statistics:');
    console.log(`  Iterations: ${learningData.iteration}`);
    console.log(`  Performance History Length: ${learningData.performanceHistory.length}`);
    console.log(`  Strategy Success Rates:`);
    Object.entries(learningData.strategySuccessRates).forEach(([algo, rate]) => {
      console.log(`    ${algo}: ${(rate * 100).toFixed(1)}%`);
    });
  }
  console.log();

  // Distill insights
  await learningOptimizer.distillLearning();
  console.log('✅ Learning insights distilled to memory');
  console.log();

  // Example 5: Performance Summary
  console.log('5️⃣  SWARM PERFORMANCE SUMMARY');
  console.log('-'.repeat(70));

  const allResults = [
    { name: 'Initial Benchmark', insights: benchmarkInsights },
    { name: 'Constraint Exploration', insights: constraintInsights },
    { name: 'Bull Market', insights: regimeResults['Bull Market'] },
    { name: 'Bear Market', insights: regimeResults['Bear Market'] },
  ];

  console.log('Scenario'.padEnd(25) + 'Best Algo'.padEnd(20) + 'Sharpe'.padEnd(12) + 'Return'.padEnd(12) + 'Risk');
  console.log('-'.repeat(70));

  allResults.forEach(({ name, insights }) => {
    const result = insights.bestResult.result;
    console.log(
      name.padEnd(25) +
      insights.bestAlgorithm.padEnd(20) +
      result.sharpeRatio.toFixed(4).padEnd(12) +
      `${(result.expectedReturn * 100).toFixed(2)}%`.padEnd(12) +
      `${(result.risk * 100).toFixed(2)}%`
    );
  });

  console.log('\n✅ Swarm exploration complete!\n');
  console.log('📝 Key Insights:');
  console.log('  • Tested 4 algorithms across multiple constraint sets');
  console.log('  • Explored optimal constraint ranges');
  console.log('  • Compared performance across 4 market regimes');
  console.log('  • Self-learning system adapted risk parameters');
  console.log('  • Memory-based strategy persistence enabled\n');

  // Cleanup
  await learningOptimizer.close();
}

/**
 * Generate mock historical returns with realistic properties
 */
function generateHistoricalReturns(numAssets: number, numPeriods: number): number[][] {
  const returns: number[][] = [];
  const baseReturns = Array(numAssets).fill(0);

  for (let t = 0; t < numPeriods; t++) {
    const marketReturn = (Math.random() - 0.48) * 0.02; // Slight positive bias

    const periodReturns = Array(numAssets).fill(0).map((_, i) => {
      const beta = 0.7 + i * 0.1; // Different market exposures
      const idiosyncratic = (Math.random() - 0.5) * 0.015;
      const drift = 0.0003 + i * 0.0001;

      return drift + beta * marketReturn + idiosyncratic;
    });

    returns.push(periodReturns);
  }

  return returns;
}

// Run the example
main().catch(error => {
  console.error('❌ Error:', error);
  process.exit(1);
});
