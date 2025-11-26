/**
 * @neural-trader/example-dynamic-pricing
 * Self-learning dynamic pricing with RL optimization and swarm strategy exploration
 */

export { DynamicPricer } from './pricer';
export { ElasticityLearner } from './elasticity-learner';
export { RLOptimizer } from './rl-optimizer';
export { CompetitiveAnalyzer } from './competitive-analyzer';
export { PricingSwarm } from './swarm';
export { ConformalPredictor } from './conformal-predictor';

export * from './types';

/**
 * Quick start example
 */
import { DynamicPricer } from './pricer';
import { ElasticityLearner } from './elasticity-learner';
import { RLOptimizer } from './rl-optimizer';
import { CompetitiveAnalyzer } from './competitive-analyzer';
import { PricingSwarm } from './swarm';
import { MarketContext } from './types';

export async function runExample() {
  console.log('🎯 Neural Trader Dynamic Pricing Example\n');

  // Initialize components
  const basePrice = 100;
  const elasticityLearner = new ElasticityLearner('./data/example_elasticity.db');
  const rlOptimizer = new RLOptimizer({
    algorithm: 'q-learning',
    learningRate: 0.1,
    epsilon: 0.2,
  });
  const competitiveAnalyzer = new CompetitiveAnalyzer();

  // Create pricer
  const pricer = new DynamicPricer(
    basePrice,
    elasticityLearner,
    rlOptimizer,
    competitiveAnalyzer
  );

  // Create swarm for strategy exploration
  const swarm = new PricingSwarm(
    {
      numAgents: 7,
      strategies: ['cost-plus', 'value-based', 'competition-based', 'dynamic-demand', 'time-based', 'elasticity-optimized', 'rl-optimized'],
      communicationTopology: 'mesh',
      consensusMechanism: 'weighted',
      explorationRate: 0.15,
    },
    basePrice,
    elasticityLearner,
    rlOptimizer,
    competitiveAnalyzer
  );

  // Simulate market context
  const context: MarketContext = {
    timestamp: Date.now(),
    dayOfWeek: 3, // Wednesday
    hour: 14,
    isHoliday: false,
    isPromotion: false,
    seasonality: 0.1,
    competitorPrices: [95, 98, 102, 105],
    inventory: 150,
    demand: 80,
  };

  console.log('📊 Market Context:');
  console.log(`  Day: ${['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][context.dayOfWeek]}`);
  console.log(`  Hour: ${context.hour}:00`);
  console.log(`  Current Demand: ${context.demand}`);
  console.log(`  Inventory: ${context.inventory}`);
  console.log(`  Competitor Prices: $${context.competitorPrices.join(', $')}\n`);

  // Test individual strategies
  console.log('💡 Individual Strategy Recommendations:\n');

  const strategies = ['cost-plus', 'value-based', 'competition-based', 'dynamic-demand', 'elasticity-optimized', 'rl-optimized'];

  for (const strategy of strategies) {
    const recommendation = await pricer.recommendPrice(context, strategy);
    console.log(`  ${strategy.padEnd(25)} → $${recommendation.price.toFixed(2)} (revenue: $${recommendation.expectedRevenue.toFixed(2)})`);
  }

  // Ensemble recommendation
  console.log('\n🎯 Ensemble Recommendation:');
  const ensembleRec = await pricer.recommendPrice(context);
  console.log(`  Price: $${ensembleRec.price.toFixed(2)}`);
  console.log(`  Expected Revenue: $${ensembleRec.expectedRevenue.toFixed(2)}`);
  console.log(`  Expected Demand: ${ensembleRec.expectedDemand.toFixed(1)} units`);
  console.log(`  Confidence: ${(ensembleRec.confidence * 100).toFixed(0)}%`);
  console.log(`  Competitive Position: ${ensembleRec.competitivePosition}\n`);

  // Swarm exploration
  console.log('🐝 Running Swarm Exploration (100 trials)...\n');
  const swarmResult = await swarm.explore(context, 100);

  console.log(`✨ Best Strategy: ${swarmResult.bestStrategy}`);
  console.log(`   Best Price: $${swarmResult.bestPrice.toFixed(2)}`);
  console.log(`   Avg Revenue: $${swarmResult.avgRevenue.toFixed(2)}\n`);

  console.log('📈 Strategy Performance:\n');
  for (const [strategy, performance] of swarmResult.results) {
    console.log(`  ${strategy.padEnd(25)} → Revenue: $${performance.totalRevenue.toFixed(0)}, Avg Demand: ${performance.avgDemand.toFixed(1)}`);
  }

  // Get competitive analysis
  console.log('\n🔍 Competitive Analysis:');
  const compAnalysis = competitiveAnalyzer.analyze(context.competitorPrices);
  console.log(`  Market Average: $${compAnalysis.avgPrice.toFixed(2)}`);
  console.log(`  Price Range: $${compAnalysis.minPrice.toFixed(2)} - $${compAnalysis.maxPrice.toFixed(2)}`);
  console.log(`  Dispersion: ${(compAnalysis.priceDispersion * 100).toFixed(1)}%`);
  console.log(`  Market Structure: ${compAnalysis.marketPosition}`);
  console.log(`  Recommendation: ${compAnalysis.recommendedPosition}\n`);

  // RL optimizer metrics
  console.log('🤖 RL Optimizer Metrics:');
  const rlMetrics = rlOptimizer.getMetrics();
  console.log(`  States Explored: ${rlMetrics.statesExplored}`);
  console.log(`  Exploration Rate: ${(rlMetrics.epsilon * 100).toFixed(1)}%`);
  console.log(`  Training Steps: ${rlMetrics.step}`);
  console.log(`  Avg Q-Value: ${rlMetrics.avgQValue.toFixed(4)}\n`);

  // Swarm statistics
  console.log('📊 Swarm Statistics:');
  const swarmStats = swarm.getStatistics();
  console.log(`  Active Agents: ${swarmStats.numAgents}`);
  console.log(`  Best Strategy: ${swarmStats.bestStrategy}`);
  console.log(`  Diversity Score: ${(swarmStats.diversityScore * 100).toFixed(0)}%\n`);

  console.log('✅ Example completed! Check the code for integration details.');
}

// Run if executed directly
if (require.main === module) {
  runExample().catch(console.error);
}
