/**
 * Basic evolutionary game theory examples
 */

import {
  PRISONERS_DILEMMA,
  HAWK_DOVE,
  STAG_HUNT,
  ReplicatorDynamics,
  ESSCalculator,
  findAllESS,
  ALWAYS_COOPERATE,
  ALWAYS_DEFECT,
  TIT_FOR_TAT,
} from '../src/index.js';

console.log('=== Evolutionary Game Theory Examples ===\n');

// Example 1: Prisoner's Dilemma Analysis
console.log('1. Prisoner\'s Dilemma Analysis');
console.log('--------------------------------');
console.log('Payoff Matrix:');
console.log('              Cooperate  Defect');
console.log('Cooperate     3          0');
console.log('Defect        5          1');
console.log('');

// Find Nash equilibria
const pdCalculator = new ESSCalculator(PRISONERS_DILEMMA);
const pdESS = findAllESS(PRISONERS_DILEMMA);

console.log('Pure Strategy ESS:', pdESS.pure.map((s) =>
  PRISONERS_DILEMMA.strategyNames[s]
));

console.log('Defect is ESS:', pdCalculator.isPureESS(1));
console.log('Cooperate is ESS:', pdCalculator.isPureESS(0));
console.log('');

// Example 2: Replicator Dynamics
console.log('2. Replicator Dynamics Simulation');
console.log('-----------------------------------');

const dynamics = new ReplicatorDynamics(PRISONERS_DILEMMA, [0.7, 0.3]);
console.log('Initial population: 70% Cooperate, 30% Defect');

const states = dynamics.simulate(20, 0.1);
console.log('\nEvolution over 20 generations:');
console.log('Gen\tCooperate\tDefect\t\tAvg Fitness');
console.log('---\t---------\t------\t\t-----------');

[0, 5, 10, 15, 20].forEach((gen) => {
  const state = states[gen] || dynamics.getState();
  console.log(
    `${gen}\t${state.frequencies[0].toFixed(4)}\t\t${state.frequencies[1].toFixed(4)}\t\t${(state.averageFitness || 0).toFixed(4)}`
  );
});

console.log('\nConvergence:');
const finalState = dynamics.simulateUntilConvergence();
console.log(
  `Final: ${(finalState.frequencies[0] * 100).toFixed(2)}% Cooperate, ` +
  `${(finalState.frequencies[1] * 100).toFixed(2)}% Defect`
);
console.log('');

// Example 3: Hawk-Dove Game
console.log('3. Hawk-Dove Game (V=4, C=6)');
console.log('-----------------------------');
console.log('Payoff Matrix:');
console.log('         Dove    Hawk');
console.log('Dove     2       0');
console.log('Hawk     4       -1');
console.log('');

const hawkDove = new ReplicatorDynamics(HAWK_DOVE, [0.5, 0.5]);
const hdStates = hawkDove.simulate(30, 0.05);

console.log('Evolution (starting from 50-50):');
console.log('Gen\tDove\tHawk');
console.log('---\t----\t----');

[0, 10, 20, 30].forEach((gen) => {
  const state = hdStates[gen] || hawkDove.getState();
  console.log(
    `${gen}\t${(state.frequencies[0] * 100).toFixed(1)}%\t${(state.frequencies[1] * 100).toFixed(1)}%`
  );
});

const hdESS = findAllESS(HAWK_DOVE);
console.log('\nESS Analysis:');
console.log('Pure ESS:', hdESS.pure.map((s) => HAWK_DOVE.strategyNames[s]));
console.log('Mixed ESS:', hdESS.mixed.map((ess) =>
  `${(ess.strategy[0] * 100).toFixed(1)}% Dove, ${(ess.strategy[1] * 100).toFixed(1)}% Hawk`
));
console.log('');

// Example 4: Stag Hunt Coordination
console.log('4. Stag Hunt Coordination Game');
console.log('-------------------------------');
console.log('Payoff Matrix:');
console.log('         Stag    Hare');
console.log('Stag     4       0');
console.log('Hare     3       3');
console.log('');

const stagHunt = new ReplicatorDynamics(STAG_HUNT, [0.8, 0.2]);
console.log('Starting with 80% Stag hunters, 20% Hare hunters');

const shStates = stagHunt.simulate(15, 0.1);
const shFinal = shStates[14];

console.log(`\nAfter 15 generations:`);
console.log(
  `${(shFinal.frequencies[0] * 100).toFixed(1)}% Stag, ` +
  `${(shFinal.frequencies[1] * 100).toFixed(1)}% Hare`
);

const shESSCalc = new ESSCalculator(STAG_HUNT);
console.log('\nESS Analysis:');
console.log('Stag is ESS:', shESSCalc.isPureESS(0));
console.log('Hare is ESS:', shESSCalc.isPureESS(1));
console.log('');

// Example 5: Invasion Analysis
console.log('5. Invasion Analysis');
console.log('--------------------');

const invCalc = new ESSCalculator(PRISONERS_DILEMMA);

// Try to invade all-defect with cooperators
const resident = [0, 1]; // All defect
const invader = [1, 0]; // All cooperate

const canInvade = invCalc.canInvade(resident, invader, 0.01);
const invasionFitness = invCalc.invasionFitness(resident, invader);

console.log('Resident: 100% Defect');
console.log('Invader: 100% Cooperate (1% frequency)');
console.log('Can invade:', canInvade);
console.log('Invasion fitness:', invasionFitness.toFixed(4));
console.log('');

// Try TFT invasion
const tftInvader = [0.5, 0.5]; // Assume TFT is mixed
const tftCanInvade = invCalc.canInvade(resident, tftInvader, 0.05);
const tftFitness = invCalc.invasionFitness(resident, tftInvader);

console.log('Invader: Mixed (simulating TFT) at 5% frequency');
console.log('Can invade:', tftCanInvade);
console.log('Invasion fitness:', tftFitness.toFixed(4));
console.log('');

// Example 6: Diversity Tracking
console.log('6. Population Diversity Over Time');
console.log('----------------------------------');

const divDynamics = new ReplicatorDynamics(PRISONERS_DILEMMA, [0.5, 0.5]);
const diversityHistory: number[] = [];

for (let i = 0; i < 20; i++) {
  diversityHistory.push(divDynamics.calculateDiversity());
  divDynamics.step(0.1);
}

console.log('Generation\tDiversity (Shannon entropy)');
console.log('----------\t---------------------------');

[0, 5, 10, 15, 19].forEach((gen) => {
  console.log(`${gen}\t\t${diversityHistory[gen].toFixed(4)}`);
});

console.log('\nDiversity decreases as population homogenizes toward defect.');
console.log('');

console.log('=== Examples Complete ===');
console.log('\nKey Insights:');
console.log('1. In Prisoner\'s Dilemma, defection dominates despite mutual cooperation being better');
console.log('2. Hawk-Dove game has a mixed ESS (evolutionary stable polymorphism)');
console.log('3. Stag Hunt shows coordination challenges - multiple equilibria');
console.log('4. Population diversity decreases as selection drives toward ESS');
console.log('5. Replicator dynamics captures frequency-dependent selection');
