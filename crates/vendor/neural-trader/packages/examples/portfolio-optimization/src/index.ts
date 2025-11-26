/**
 * @neural-trader/example-portfolio-optimization
 *
 * Self-learning portfolio optimization with benchmark swarms
 * Implements Mean-Variance, Risk Parity, Black-Litterman, and Multi-Objective optimization
 * Uses AgentDB for memory patterns and OpenRouter for AI-powered strategy suggestions
 */

// Export optimizers
export {
  MeanVarianceOptimizer,
  RiskParityOptimizer,
  BlackLittermanOptimizer,
  MultiObjectiveOptimizer,
  type Asset,
  type PortfolioConstraints,
  type OptimizationResult,
  type EfficientFrontierPoint,
} from './optimizer.js';

// Export self-learning components
export {
  SelfLearningOptimizer,
  AdaptiveRiskManager,
  type RiskProfile,
  type PerformanceMetrics,
  type LearningState,
} from './self-learning.js';

// Export benchmark swarm
export {
  PortfolioOptimizationSwarm,
  ParallelPortfolioExplorer,
  type BenchmarkConfig,
  type BenchmarkResult,
  type SwarmInsights,
} from './benchmark-swarm.js';

/**
 * Quick start example
 */
export async function quickStart() {
  console.log('🚀 Neural Trader Portfolio Optimization');
  console.log('=======================================\n');

  // Example assets
  const assets = [
    { symbol: 'AAPL', expectedReturn: 0.12, volatility: 0.20 },
    { symbol: 'GOOGL', expectedReturn: 0.15, volatility: 0.25 },
    { symbol: 'MSFT', expectedReturn: 0.11, volatility: 0.18 },
    { symbol: 'AMZN', expectedReturn: 0.16, volatility: 0.28 },
    { symbol: 'TSLA', expectedReturn: 0.20, volatility: 0.40 },
  ];

  // Correlation matrix (simplified)
  const correlationMatrix = [
    [1.00, 0.65, 0.70, 0.60, 0.45],
    [0.65, 1.00, 0.68, 0.72, 0.50],
    [0.70, 0.68, 1.00, 0.64, 0.48],
    [0.60, 0.72, 0.64, 1.00, 0.55],
    [0.45, 0.50, 0.48, 0.55, 1.00],
  ];

  console.log('📊 Initializing Mean-Variance Optimizer...');
  const { MeanVarianceOptimizer } = await import('./optimizer.js');
  const optimizer = new MeanVarianceOptimizer(assets, correlationMatrix);

  const result = optimizer.optimize({
    minWeight: 0.05,
    maxWeight: 0.40,
    targetReturn: 0.14,
  });

  console.log('\n✅ Optimization Complete!');
  console.log('Portfolio Weights:');
  assets.forEach((asset, i) => {
    console.log(`  ${asset.symbol}: ${(result.weights[i] * 100).toFixed(2)}%`);
  });
  console.log(`\nExpected Return: ${(result.expectedReturn * 100).toFixed(2)}%`);
  console.log(`Risk (Volatility): ${(result.risk * 100).toFixed(2)}%`);
  console.log(`Sharpe Ratio: ${result.sharpeRatio.toFixed(4)}`);
  console.log(`Diversification: ${result.diversificationRatio.toFixed(4)}`);

  return result;
}

// Export convenience function
export { quickStart as default };
