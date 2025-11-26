/**
 * Basic usage example for logistics optimization
 */

import {
  LogisticsOptimizer,
  createSampleData,
  SwarmConfig
} from '../src';

async function main() {
  console.log('=== @neural-trader/example-logistics-optimization ===\n');

  // Create sample data
  console.log('Creating sample data...');
  const { customers, vehicles } = createSampleData(30, 5);
  console.log(`- ${customers.length} customers`);
  console.log(`- ${vehicles.length} vehicles\n`);

  // Single-agent optimization
  console.log('1. Single-Agent Genetic Algorithm:');
  const singleAgent = new LogisticsOptimizer(customers, vehicles, false);
  const solution1 = await singleAgent.optimize('genetic');
  console.log(`   - Fitness: ${solution1.fitness.toFixed(2)}`);
  console.log(`   - Total Cost: $${solution1.totalCost.toFixed(2)}`);
  console.log(`   - Routes: ${solution1.routes.length}`);
  console.log(`   - Compute Time: ${solution1.metadata.computeTime}ms\n`);

  // Swarm optimization
  console.log('2. Multi-Agent Swarm Optimization:');
  const swarmConfig: SwarmConfig = {
    numAgents: 6,
    topology: 'mesh',
    communicationStrategy: 'best-solution',
    convergenceCriteria: {
      maxIterations: 100,
      noImprovementSteps: 20
    }
  };

  const swarmOptimizer = new LogisticsOptimizer(customers, vehicles, true, swarmConfig);
  const solution2 = await swarmOptimizer.optimize();
  console.log(`   - Fitness: ${solution2.fitness.toFixed(2)}`);
  console.log(`   - Total Cost: $${solution2.totalCost.toFixed(2)}`);
  console.log(`   - Routes: ${solution2.routes.length}`);
  console.log(`   - Compute Time: ${solution2.metadata.computeTime}ms`);
  console.log(`   - Agent: ${solution2.metadata.agentId || 'N/A'}\n`);

  // Show improvement
  const improvement = ((solution1.fitness - solution2.fitness) / solution1.fitness) * 100;
  console.log(`3. Improvement:
   - Swarm vs Single-Agent: ${improvement.toFixed(1)}%
   - Speedup: ${(solution1.metadata.computeTime / solution2.metadata.computeTime).toFixed(2)}x\n`);

  // Learning statistics
  console.log('4. Learning Statistics:');
  const stats = swarmOptimizer.getStatistics();
  console.log(`   - Episodes: ${stats.totalEpisodes}`);
  console.log(`   - Avg Quality: ${stats.avgSolutionQuality.toFixed(2)}`);
  console.log(`   - Avg Compute Time: ${stats.avgComputeTime.toFixed(0)}ms`);
  console.log(`   - Traffic Patterns Learned: ${stats.trafficPatternsLearned}\n`);

  // Route details
  console.log('5. Best Solution Routes:');
  solution2.routes.forEach((route, i) => {
    console.log(`   Route ${i + 1} (${route.vehicleId}):`);
    console.log(`   - Customers: ${route.customers.length}`);
    console.log(`   - Distance: ${route.totalDistance.toFixed(2)} km`);
    console.log(`   - Time: ${route.totalTime.toFixed(0)} min`);
    console.log(`   - Cost: $${route.totalCost.toFixed(2)}`);
    console.log(`   - Utilization: ${(route.utilizationRate * 100).toFixed(1)}%`);
  });

  console.log('\n=== Optimization Complete ===');
}

main().catch(console.error);
