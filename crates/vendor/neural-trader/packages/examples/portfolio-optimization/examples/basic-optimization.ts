/**
 * Basic Portfolio Optimization Example
 * Demonstrates all optimization algorithms
 */

import {
  MeanVarianceOptimizer,
  RiskParityOptimizer,
  BlackLittermanOptimizer,
  MultiObjectiveOptimizer,
  Asset,
} from '../src/optimizer.js';

async function main() {
  console.log('🎯 Neural Trader - Basic Portfolio Optimization Example\n');
  console.log('='.repeat(70) + '\n');

  // Define sample assets
  const assets: Asset[] = [
    { symbol: 'AAPL', expectedReturn: 0.12, volatility: 0.20 },
    { symbol: 'GOOGL', expectedReturn: 0.15, volatility: 0.25 },
    { symbol: 'MSFT', expectedReturn: 0.11, volatility: 0.18 },
    { symbol: 'AMZN', expectedReturn: 0.16, volatility: 0.28 },
    { symbol: 'NVDA', expectedReturn: 0.18, volatility: 0.35 },
  ];

  // Correlation matrix
  const correlationMatrix = [
    [1.00, 0.65, 0.70, 0.60, 0.55],
    [0.65, 1.00, 0.68, 0.72, 0.58],
    [0.70, 0.68, 1.00, 0.64, 0.52],
    [0.60, 0.72, 0.64, 1.00, 0.60],
    [0.55, 0.58, 0.52, 0.60, 1.00],
  ];

  console.log('📊 Portfolio Assets:');
  assets.forEach(asset => {
    console.log(
      `  ${asset.symbol}: Expected Return ${(asset.expectedReturn * 100).toFixed(2)}%, ` +
      `Volatility ${(asset.volatility * 100).toFixed(2)}%`
    );
  });
  console.log();

  // 1. Mean-Variance Optimization
  console.log('1️⃣  MEAN-VARIANCE OPTIMIZATION (Markowitz)');
  console.log('-'.repeat(70));

  const mvOptimizer = new MeanVarianceOptimizer(assets, correlationMatrix);
  const mvResult = mvOptimizer.optimize({
    minWeight: 0.05,
    maxWeight: 0.40,
    targetReturn: 0.14,
  });

  console.log('Optimal Portfolio Weights:');
  assets.forEach((asset, i) => {
    console.log(`  ${asset.symbol}: ${(mvResult.weights[i] * 100).toFixed(2)}%`);
  });
  console.log(`Expected Return: ${(mvResult.expectedReturn * 100).toFixed(2)}%`);
  console.log(`Risk (Volatility): ${(mvResult.risk * 100).toFixed(2)}%`);
  console.log(`Sharpe Ratio: ${mvResult.sharpeRatio.toFixed(4)}`);
  console.log(`Diversification: ${mvResult.diversificationRatio.toFixed(4)}`);
  console.log();

  // Generate efficient frontier
  console.log('📈 Generating Efficient Frontier (30 points)...');
  const frontier = mvOptimizer.generateEfficientFrontier(30);
  console.log(`Generated ${frontier.length} portfolio combinations`);
  console.log(`Min Risk: ${(frontier[0].risk * 100).toFixed(2)}%`);
  console.log(`Max Risk: ${(frontier[frontier.length - 1].risk * 100).toFixed(2)}%`);
  console.log();

  // 2. Risk Parity Optimization
  console.log('2️⃣  RISK PARITY OPTIMIZATION');
  console.log('-'.repeat(70));

  const rpOptimizer = new RiskParityOptimizer(assets, correlationMatrix);
  const rpResult = rpOptimizer.optimize();

  console.log('Risk-Balanced Portfolio Weights:');
  assets.forEach((asset, i) => {
    console.log(`  ${asset.symbol}: ${(rpResult.weights[i] * 100).toFixed(2)}%`);
  });
  console.log(`Expected Return: ${(rpResult.expectedReturn * 100).toFixed(2)}%`);
  console.log(`Risk (Volatility): ${(rpResult.risk * 100).toFixed(2)}%`);
  console.log(`Sharpe Ratio: ${rpResult.sharpeRatio.toFixed(4)}`);
  console.log(`Diversification: ${rpResult.diversificationRatio.toFixed(4)}`);
  console.log();

  // 3. Black-Litterman Optimization
  console.log('3️⃣  BLACK-LITTERMAN OPTIMIZATION');
  console.log('-'.repeat(70));

  const marketCapWeights = [0.30, 0.25, 0.20, 0.15, 0.10];
  const blOptimizer = new BlackLittermanOptimizer(
    assets,
    correlationMatrix,
    marketCapWeights,
    2.5, // Risk aversion
  );

  // Define investor views
  const views = [
    { assets: [0], expectedReturn: 0.15, confidence: 0.7 }, // Bullish on AAPL
    { assets: [4], expectedReturn: 0.20, confidence: 0.6 }, // Very bullish on NVDA
    { assets: [1, 2], expectedReturn: 0.12, confidence: 0.5 }, // Moderate on GOOGL & MSFT
  ];

  const blResult = blOptimizer.optimize(views);

  console.log('Black-Litterman Portfolio (with investor views):');
  assets.forEach((asset, i) => {
    console.log(`  ${asset.symbol}: ${(blResult.weights[i] * 100).toFixed(2)}%`);
  });
  console.log(`Expected Return: ${(blResult.expectedReturn * 100).toFixed(2)}%`);
  console.log(`Risk (Volatility): ${(blResult.risk * 100).toFixed(2)}%`);
  console.log(`Sharpe Ratio: ${blResult.sharpeRatio.toFixed(4)}`);
  console.log(`Diversification: ${blResult.diversificationRatio.toFixed(4)}`);
  console.log();

  // 4. Multi-Objective Optimization
  console.log('4️⃣  MULTI-OBJECTIVE OPTIMIZATION');
  console.log('-'.repeat(70));

  // Generate mock historical returns
  const historicalReturns = generateHistoricalReturns(assets.length, 252);

  const moOptimizer = new MultiObjectiveOptimizer(
    assets,
    correlationMatrix,
    historicalReturns,
  );

  const moResult = moOptimizer.optimize({
    return: 1.0,
    risk: 1.0,
    drawdown: 0.8,
  });

  console.log('Multi-Objective Portfolio (Return, Risk, Drawdown):');
  assets.forEach((asset, i) => {
    console.log(`  ${asset.symbol}: ${(moResult.weights[i] * 100).toFixed(2)}%`);
  });
  console.log(`Expected Return: ${(moResult.expectedReturn * 100).toFixed(2)}%`);
  console.log(`Risk (Volatility): ${(moResult.risk * 100).toFixed(2)}%`);
  console.log(`Sharpe Ratio: ${moResult.sharpeRatio.toFixed(4)}`);
  console.log(`Diversification: ${moResult.diversificationRatio.toFixed(4)}`);
  console.log();

  // Comparison Summary
  console.log('📊 ALGORITHM COMPARISON SUMMARY');
  console.log('='.repeat(70));

  const results = [
    { name: 'Mean-Variance', result: mvResult },
    { name: 'Risk Parity', result: rpResult },
    { name: 'Black-Litterman', result: blResult },
    { name: 'Multi-Objective', result: moResult },
  ];

  console.log('Algorithm'.padEnd(20) + 'Return'.padEnd(12) + 'Risk'.padEnd(12) + 'Sharpe'.padEnd(12) + 'Diversification');
  console.log('-'.repeat(70));

  results.forEach(({ name, result }) => {
    console.log(
      name.padEnd(20) +
      `${(result.expectedReturn * 100).toFixed(2)}%`.padEnd(12) +
      `${(result.risk * 100).toFixed(2)}%`.padEnd(12) +
      result.sharpeRatio.toFixed(4).padEnd(12) +
      result.diversificationRatio.toFixed(4)
    );
  });

  console.log('\n✅ Example complete!\n');
}

/**
 * Generate mock historical returns
 */
function generateHistoricalReturns(numAssets: number, numPeriods: number): number[][] {
  const returns: number[][] = [];

  for (let t = 0; t < numPeriods; t++) {
    const periodReturns = Array(numAssets).fill(0).map((_, i) => {
      const drift = 0.0003 + i * 0.0001; // Slight upward drift
      const volatility = 0.01 + i * 0.002;
      return drift + volatility * (Math.random() - 0.5) * 2;
    });
    returns.push(periodReturns);
  }

  return returns;
}

// Run the example
main().catch(console.error);
