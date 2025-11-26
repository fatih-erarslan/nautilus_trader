/**
 * Advanced example demonstrating 10+ agent swarm coordination
 */

import {
  SwarmCoordinator,
  createSampleData,
  SwarmConfig
} from '../src';

async function main() {
  console.log('=== Multi-Agent Swarm Coordination Demo ===\n');

  // Create larger problem
  console.log('Creating large-scale problem...');
  const { customers, vehicles } = createSampleData(100, 10);
  console.log(`- ${customers.length} customers`);
  console.log(`- ${vehicles.length} vehicles\n`);

  // Configure 12-agent swarm
  const swarmConfig: SwarmConfig = {
    numAgents: 12,
    topology: 'mesh',
    communicationStrategy: 'best-solution',
    convergenceCriteria: {
      maxIterations: 200,
      noImprovementSteps: 30
    }
  };

  console.log('Initializing 12-agent swarm...');
  const coordinator = new SwarmCoordinator(swarmConfig, customers, vehicles);

  const agents = coordinator.getAgents();
  console.log(`\nAgent Configuration:`);
  agents.forEach((agent, i) => {
    console.log(`  ${i + 1}. ${agent.id}: ${agent.algorithm}`);
  });
  console.log();

  // Monitor optimization progress
  console.log('Starting swarm optimization...\n');

  const optimizationPromise = coordinator.optimize();

  // Monitor progress
  const monitorInterval = setInterval(() => {
    const status = coordinator.getStatus();
    console.log(`Iteration ${status.iteration}:
    - Active Agents: ${status.agentsWorking}
    - Completed Agents: ${status.agentsCompleted}
    - Best Fitness: ${status.globalBestFitness?.toFixed(2) || 'N/A'}
    - Convergence: ${(status.convergence * 100).toFixed(1)}%`);
  }, 2000);

  const solution = await optimizationPromise;
  clearInterval(monitorInterval);

  console.log('\n=== Optimization Complete ===\n');

  // Final results
  console.log('Final Solution:');
  console.log(`- Fitness: ${solution.fitness.toFixed(2)}`);
  console.log(`- Total Cost: $${solution.totalCost.toFixed(2)}`);
  console.log(`- Total Distance: ${solution.totalDistance.toFixed(2)} km`);
  console.log(`- Routes: ${solution.routes.length}`);
  console.log(`- Customers Served: ${solution.routes.reduce((sum, r) => sum + r.customers.length, 0)}`);
  console.log(`- Unassigned: ${solution.unassignedCustomers.length}`);
  console.log(`- Algorithm: ${solution.metadata.algorithm}`);
  console.log(`- Compute Time: ${solution.metadata.computeTime}ms`);
  console.log(`- Found by Agent: ${solution.metadata.agentId}\n`);

  // Agent performance summary
  console.log('Agent Performance:');
  const finalAgents = coordinator.getAgents();
  finalAgents
    .sort((a, b) => (a.bestSolution?.fitness || Infinity) - (b.bestSolution?.fitness || Infinity))
    .forEach((agent, i) => {
      console.log(`  ${i + 1}. ${agent.id} (${agent.algorithm}):
     - Best Fitness: ${agent.bestSolution?.fitness.toFixed(2) || 'N/A'}
     - Iterations: ${agent.iterations}
     - Status: ${agent.status}`);
    });

  // Get LLM recommendations
  console.log('\nGetting AI recommendations...');
  const recommendations = await coordinator.reasonAboutConstraints(solution);
  console.log('\nAI Analysis:');
  console.log(recommendations);

  console.log('\n=== Demo Complete ===');
}

main().catch(console.error);
