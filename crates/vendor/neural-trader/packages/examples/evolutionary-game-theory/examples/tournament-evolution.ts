/**
 * Tournament evolution example
 * Demonstrates iterated games and strategy competition
 */

import {
  Tournament,
  PRISONERS_DILEMMA,
  ALWAYS_COOPERATE,
  ALWAYS_DEFECT,
  TIT_FOR_TAT,
  TIT_FOR_TWO_TATS,
  GRIM_TRIGGER,
  PAVLOV,
  ADAPTIVE,
  GRADUAL,
  createGenerousTitForTat,
} from '../src/index.js';

console.log('=== Tournament Evolution Examples ===\n');

// Example 1: Classic Tournament
console.log('1. Classic Iterated Prisoner\'s Dilemma Tournament');
console.log('--------------------------------------------------');

const classicStrategies = [
  ALWAYS_COOPERATE,
  ALWAYS_DEFECT,
  TIT_FOR_TAT,
  TIT_FOR_TWO_TATS,
  GRIM_TRIGGER,
  PAVLOV,
];

const tournament1 = new Tournament({
  game: PRISONERS_DILEMMA,
  strategies: classicStrategies,
  roundsPerMatch: 200,
  tournamentStyle: 'round-robin',
  repeatMatches: 5,
});

console.log('Strategies:', classicStrategies.map((s) => s.name).join(', '));
console.log('Match length: 200 rounds');
console.log('Tournament style: Round-robin with 5 repeats');
console.log('\nRunning tournament...\n');

const result1 = tournament1.run();

console.log('Final Rankings:');
console.log('Rank\tStrategy\t\t\tScore\tWins\tLosses\tDraws\tWin Rate');
console.log('----\t--------\t\t\t-----\t----\t------\t-----\t--------');

result1.rankings.forEach((player, index) => {
  const winRate = (player.wins + 0.5 * player.draws) / player.matches;
  const paddedName = player.strategy.name.padEnd(24);
  console.log(
    `${index + 1}\t${paddedName}\t${player.score.toFixed(1)}\t${player.wins}\t${player.losses}\t${player.draws}\t${(winRate * 100).toFixed(1)}%`
  );
});

console.log(`\nBest Strategy: ${result1.bestStrategy.name}`);
console.log(`Average Score: ${result1.averageScore.toFixed(2)}`);
console.log(`Diversity Index: ${result1.diversityIndex.toFixed(3)}`);
console.log('');

// Example 2: Strategy Analysis
console.log('2. Detailed Strategy Analysis');
console.log('------------------------------');

const tftAnalysis = tournament1.analyzeStrategy(TIT_FOR_TAT.id);
const pavlovAnalysis = tournament1.analyzeStrategy(PAVLOV.id);

console.log('Tit-for-Tat Performance:');
console.log(`  Average Score: ${tftAnalysis.averageScore.toFixed(2)}`);
console.log(`  Win Rate: ${(tftAnalysis.winRate * 100).toFixed(1)}%`);
console.log(`  Cooperation Rate: ${(tftAnalysis.cooperationRate * 100).toFixed(1)}%`);
console.log('  Performance by Opponent:');

tftAnalysis.performanceByOpponent.forEach((score, opponentId) => {
  const opponent = classicStrategies.find((s) => s.id === opponentId);
  if (opponent) {
    console.log(`    vs ${opponent.name.padEnd(20)}: ${score.toFixed(2)}`);
  }
});

console.log('\nPavlov (Win-Stay, Lose-Shift) Performance:');
console.log(`  Average Score: ${pavlovAnalysis.averageScore.toFixed(2)}`);
console.log(`  Win Rate: ${(pavlovAnalysis.winRate * 100).toFixed(1)}%`);
console.log(`  Cooperation Rate: ${(pavlovAnalysis.cooperationRate * 100).toFixed(1)}%`);
console.log('');

// Example 3: Noisy Environment
console.log('3. Tournament with Noisy Moves (1% error rate)');
console.log('----------------------------------------------');

const noisyTournament = new Tournament({
  game: PRISONERS_DILEMMA,
  strategies: [TIT_FOR_TAT, PAVLOV, createGenerousTitForTat(0.1), GRIM_TRIGGER],
  roundsPerMatch: 200,
  tournamentStyle: 'round-robin',
  noiseProbability: 0.01,
});

const noisyResult = noisyTournament.run();

console.log('Rankings with 1% move errors:');
console.log('Rank\tStrategy\t\t\tScore');
console.log('----\t--------\t\t\t-----');

noisyResult.rankings.forEach((player, index) => {
  const paddedName = player.strategy.name.padEnd(24);
  console.log(`${index + 1}\t${paddedName}\t${player.score.toFixed(1)}`);
});

console.log('\nNote: Forgiving strategies (Pavlov, Generous TFT) handle noise better');
console.log('      than unforgiving strategies (Grim Trigger).');
console.log('');

// Example 4: Extended Tournament with Learning
console.log('4. Extended Tournament with Adaptive Strategy');
console.log('----------------------------------------------');

const extendedStrategies = [
  ALWAYS_COOPERATE,
  ALWAYS_DEFECT,
  TIT_FOR_TAT,
  PAVLOV,
  ADAPTIVE,
  GRADUAL,
];

const tournament2 = new Tournament({
  game: PRISONERS_DILEMMA,
  strategies: extendedStrategies,
  roundsPerMatch: 300, // Longer matches for adaptation
  tournamentStyle: 'round-robin',
});

const result2 = tournament2.run();

console.log('Rankings (300 rounds/match):');
console.log('Rank\tStrategy\t\tScore\tCooperation');
console.log('----\t--------\t\t-----\t-----------');

result2.rankings.forEach((player, index) => {
  const coopRate = tournament2.getCooperationRate(player.id);
  const paddedName = player.strategy.name.padEnd(16);
  console.log(
    `${index + 1}\t${paddedName}\t${player.score.toFixed(1)}\t${(coopRate * 100).toFixed(1)}%`
  );
});

console.log('');

// Example 5: Ecological Simulation
console.log('5. Ecological Simulation (Multiple Tournaments)');
console.log('------------------------------------------------');

const ecology = {
  populations: new Map<string, number>(),
};

// Initialize populations
extendedStrategies.forEach((s) => {
  ecology.populations.set(s.id, 100); // Start with 100 of each
});

console.log('Simulating 10 generations of population dynamics...\n');
console.log('Gen\tAC\tAD\tTFT\tPavlov\tAdaptive\tGradual');
console.log('---\t--\t--\t---\t------\t--------\t-------');

for (let gen = 0; gen < 10; gen++) {
  // Run tournament
  const tourney = new Tournament({
    game: PRISONERS_DILEMMA,
    strategies: extendedStrategies,
    roundsPerMatch: 100,
    tournamentStyle: 'round-robin',
  });

  const result = tourney.run();

  // Update populations based on fitness (simplified)
  const totalScore = result.rankings.reduce((sum, p) => sum + p.score, 0);

  result.rankings.forEach((player) => {
    const currentPop = ecology.populations.get(player.id) || 0;
    const fitness = player.score / totalScore;
    const newPop = Math.round(currentPop * (1 + fitness * 0.5)); // 50% growth rate
    ecology.populations.set(player.id, newPop);
  });

  // Normalize to keep total population constant
  const totalPop = Array.from(ecology.populations.values()).reduce((a, b) => a + b, 0);
  ecology.populations.forEach((pop, id) => {
    ecology.populations.set(id, Math.round((pop / totalPop) * 600));
  });

  // Display generation
  const ac = ecology.populations.get(ALWAYS_COOPERATE.id) || 0;
  const ad = ecology.populations.get(ALWAYS_DEFECT.id) || 0;
  const tft = ecology.populations.get(TIT_FOR_TAT.id) || 0;
  const pav = ecology.populations.get(PAVLOV.id) || 0;
  const adapt = ecology.populations.get(ADAPTIVE.id) || 0;
  const grad = ecology.populations.get(GRADUAL.id) || 0;

  console.log(`${gen}\t${ac}\t${ad}\t${tft}\t${pav}\t${adapt}\t\t${grad}`);
}

console.log('\n=== Tournament Examples Complete ===');
console.log('\nKey Findings:');
console.log('1. TFT and Pavlov consistently perform well in tournaments');
console.log('2. Forgiveness helps in noisy environments');
console.log('3. Adaptive strategies can match opponent patterns');
console.log('4. Population dynamics favor strategies that do well against themselves');
console.log('5. No single strategy dominates all contexts - it depends on the ecology');
