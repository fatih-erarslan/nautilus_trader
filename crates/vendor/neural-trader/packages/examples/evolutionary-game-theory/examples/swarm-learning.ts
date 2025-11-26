/**
 * Swarm-based evolutionary learning example
 * Demonstrates genetic algorithms, self-learning, and AgentDB integration
 */

import {
  SwarmEvolution,
  PRISONERS_DILEMMA,
  HAWK_DOVE,
  Tournament,
  createLearningStrategy,
  TIT_FOR_TAT,
  PAVLOV,
} from '../src/index.js';

console.log('=== Swarm Evolution Examples ===\n');

// Example 1: Basic Genetic Algorithm
console.log('1. Basic Genetic Algorithm Evolution');
console.log('-------------------------------------');

const swarm1 = new SwarmEvolution(PRISONERS_DILEMMA, {
  populationSize: 50,
  mutationRate: 0.1,
  crossoverRate: 0.7,
  elitismRate: 0.1,
  maxGenerations: 30,
});

console.log('Configuration:');
console.log('  Population: 50 strategies');
console.log('  Mutation rate: 10%');
console.log('  Crossover rate: 70%');
console.log('  Elitism: Top 10%');
console.log('  Generations: 30');
console.log('\nEvolving...\n');

const startTime = Date.now();

// Track progress
const evolutionHistory: any[] = [];
for (let i = 0; i < 30; i++) {
  const result = await swarm1.evolveGeneration();
  evolutionHistory.push(result);

  if (i % 5 === 0) {
    console.log(
      `Gen ${i}: Best Fitness = ${result.bestFitness.toFixed(3)}, ` +
      `Diversity = ${result.populationDiversity.toFixed(3)}`
    );
  }
}

const elapsedMs = Date.now() - startTime;
const finalStats = swarm1.getStatistics();

console.log(`\nEvolution complete in ${(elapsedMs / 1000).toFixed(2)}s`);
console.log(`\nFinal Statistics:`);
console.log(`  Generation: ${finalStats.generation}`);
console.log(`  Best Fitness: ${finalStats.averageFitness.toFixed(3)}`);
console.log(`  Fitness Variance: ${finalStats.fitnessVariance.toFixed(3)}`);
console.log('');

// Example 2: Convergence Analysis
console.log('2. Convergence Analysis');
console.log('-----------------------');

console.log('Fitness progression:');
console.log('Gen\tBest\t\tAvg\t\tDiversity');
console.log('---\t----\t\t---\t\t---------');

[0, 5, 10, 15, 20, 25, 29].forEach((gen) => {
  const result = evolutionHistory[gen];
  const avgFitness = result.convergenceHistory.reduce((a: number, b: number) => a + b, 0) / result.convergenceHistory.length;
  console.log(
    `${gen}\t${result.bestFitness.toFixed(3)}\t\t${avgFitness.toFixed(3)}\t\t${result.populationDiversity.toFixed(3)}`
  );
});

console.log('\nObservations:');
console.log('  - Fitness improves over generations');
console.log('  - Diversity decreases as population converges');
console.log('  - Best strategies are preserved through elitism');
console.log('');

// Example 3: Compare Evolved vs Classic Strategies
console.log('3. Evolved Strategy vs Classic Strategies');
console.log('------------------------------------------');

const evolvedStrategy = finalStats.bestStrategies[0];

const comparisonTournament = new Tournament({
  game: PRISONERS_DILEMMA,
  strategies: [
    evolvedStrategy,
    TIT_FOR_TAT,
    PAVLOV,
  ],
  roundsPerMatch: 200,
  tournamentStyle: 'round-robin',
  repeatMatches: 10,
});

console.log('Testing evolved strategy against:');
console.log('  - Tit-for-Tat');
console.log('  - Pavlov');
console.log('\nTournament results (200 rounds, 10 repeats):\n');

const compResult = comparisonTournament.run();

console.log('Rank\tStrategy\t\t\tScore\tWin Rate');
console.log('----\t--------\t\t\t-----\t--------');

compResult.rankings.forEach((player, index) => {
  const winRate = (player.wins + 0.5 * player.draws) / player.matches;
  const paddedName = player.strategy.name.padEnd(24);
  console.log(
    `${index + 1}\t${paddedName}\t${player.score.toFixed(1)}\t${(winRate * 100).toFixed(1)}%`
  );
});

console.log('');

// Example 4: Multi-Population Coevolution
console.log('4. Multi-Population Coevolution');
console.log('--------------------------------');

const swarm2 = new SwarmEvolution(PRISONERS_DILEMMA, {
  populationSize: 40,
  maxGenerations: 20,
});

const swarm3 = new SwarmEvolution(HAWK_DOVE, {
  populationSize: 40,
  maxGenerations: 20,
});

console.log('Evolving two populations in parallel:');
console.log('  Population A: Prisoner\'s Dilemma');
console.log('  Population B: Hawk-Dove Game');
console.log('');

const [resultPD, resultHD] = await Promise.all([
  swarm2.run(),
  swarm3.run(),
]);

console.log('Results:');
console.log('\nPrisoner\'s Dilemma Population:');
console.log(`  Generations: ${resultPD.generation}`);
console.log(`  Best Fitness: ${resultPD.bestFitness.toFixed(3)}`);
console.log(`  Final Diversity: ${resultPD.populationDiversity.toFixed(3)}`);

console.log('\nHawk-Dove Population:');
console.log(`  Generations: ${resultHD.generation}`);
console.log(`  Best Fitness: ${resultHD.bestFitness.toFixed(3)}`);
console.log(`  Final Diversity: ${resultHD.populationDiversity.toFixed(3)}`);
console.log('');

// Example 5: Fitness Landscape Exploration
console.log('5. Fitness Landscape Exploration');
console.log('--------------------------------');

const explorer = new SwarmEvolution(PRISONERS_DILEMMA, {
  populationSize: 30,
  maxGenerations: 10,
});

// Evolve a bit first
await explorer.run();

console.log('Sampling fitness landscape...\n');

const landscape = await explorer.exploreFitnessLandscape(10);

// Find peaks and valleys
const sortedPoints = [...landscape].sort((a, b) => b.fitness - a.fitness);

console.log('Fitness Landscape Analysis:');
console.log('  Total samples: ' + landscape.length);
console.log(`  Highest fitness: ${sortedPoints[0].fitness.toFixed(3)}`);
console.log(`  Lowest fitness: ${sortedPoints[sortedPoints.length - 1].fitness.toFixed(3)}`);
console.log(`  Mean fitness: ${(landscape.reduce((sum, p) => sum + p.fitness, 0) / landscape.length).toFixed(3)}`);

console.log('\nTop 5 Strategy Archetypes (by fitness):');
sortedPoints.slice(0, 5).forEach((point, i) => {
  const weights = point.strategy.slice(0, 3).map((w) => w.toFixed(2));
  console.log(`  ${i + 1}. Fitness: ${point.fitness.toFixed(3)}, Weights: [${weights.join(', ')}, ...]`);
});

console.log('');

// Example 6: Strategy Distribution Analysis
console.log('6. Strategy Distribution Analysis');
console.log('---------------------------------');

const swarm4 = new SwarmEvolution(PRISONERS_DILEMMA, {
  populationSize: 100,
  mutationRate: 0.15, // Higher mutation for diversity
  maxGenerations: 25,
});

const finalResult = await swarm4.run();

console.log('Final population distribution:');
console.log('\nTop strategy types:');

const sortedDistribution = Array.from(finalResult.strategyDistribution.entries())
  .sort((a, b) => b[1] - a[1])
  .slice(0, 10);

sortedDistribution.forEach(([name, count]) => {
  const percentage = (count / 100) * 100;
  const bar = '█'.repeat(Math.floor(percentage / 2));
  console.log(`  ${name.substring(0, 20).padEnd(20)}: ${bar} ${percentage.toFixed(1)}%`);
});

console.log('');

// Example 7: Performance Metrics
console.log('7. Performance Metrics');
console.log('----------------------');

console.log('Swarm Evolution Performance:');
console.log(`  Population size: 50-100 strategies`);
console.log(`  Generations: 20-30`);
console.log(`  Time per generation: ~${(elapsedMs / 30 / 1000).toFixed(2)}s`);
console.log(`  Total evolution time: ~${(elapsedMs / 1000).toFixed(2)}s`);
console.log(`  Strategies evaluated: ~${50 * 30 * 10} (pop × gen × opponents)`);
console.log('');

console.log('Genetic Algorithm Parameters Impact:');
console.log('  - High mutation (>15%): More exploration, slower convergence');
console.log('  - High crossover (>70%): Faster mixing of good traits');
console.log('  - High elitism (>10%): Faster convergence, less diversity');
console.log('  - Large population (>100): Better exploration, slower');
console.log('');

console.log('=== Swarm Evolution Examples Complete ===');
console.log('\nKey Insights:');
console.log('1. Genetic algorithms can discover competitive strategies');
console.log('2. Evolved strategies often match or exceed classic strategies');
console.log('3. Fitness landscapes have multiple peaks (different strategy types)');
console.log('4. Population diversity decreases as selection intensifies');
console.log('5. Different games produce different evolved strategies');
console.log('6. Coevolution can discover novel strategy interactions');
console.log('\nApplications:');
console.log('- Market competition modeling');
console.log('- Cooperation emergence studies');
console.log('- Social dynamics simulation');
console.log('- Mechanism design optimization');
console.log('- Multi-agent system coordination');
