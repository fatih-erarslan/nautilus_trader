/**
 * Classic game theory games
 */

import type { Game, Strategy, GameHistory } from './types.js';

/**
 * Prisoner's Dilemma
 *
 * Two prisoners can cooperate (stay silent) or defect (betray).
 * - Both cooperate: light sentence (3 points each)
 * - Both defect: medium sentence (1 point each)
 * - One defects: defector goes free (5 points), cooperator gets heavy sentence (0 points)
 */
export const PRISONERS_DILEMMA: Game = {
  id: 'prisoners-dilemma',
  name: "Prisoner's Dilemma",
  description: 'Classic dilemma of cooperation vs defection',
  payoffMatrix: [
    [3, 0], // Cooperate: (C,C)=3, (C,D)=0
    [5, 1], // Defect: (D,C)=5, (D,D)=1
  ],
  numStrategies: 2,
  strategyNames: ['Cooperate', 'Defect'],
  isSymmetric: true,
};

/**
 * Hawk-Dove (Chicken) Game
 *
 * Contest over a resource of value V.
 * - Both Dove: share resource (V/2 each)
 * - Both Hawk: fight with cost C (V/2 - C each)
 * - Hawk vs Dove: Hawk gets all, Dove gets nothing
 */
export function createHawkDoveGame(resourceValue: number = 4, fightCost: number = 6): Game {
  return {
    id: 'hawk-dove',
    name: 'Hawk-Dove Game',
    description: 'Contest over resources with fighting costs',
    payoffMatrix: [
      [resourceValue / 2, 0], // Dove: (D,D)=V/2, (D,H)=0
      [resourceValue, (resourceValue - fightCost) / 2], // Hawk: (H,D)=V, (H,H)=(V-C)/2
    ],
    numStrategies: 2,
    strategyNames: ['Dove', 'Hawk'],
    isSymmetric: true,
  };
}

export const HAWK_DOVE = createHawkDoveGame();

/**
 * Stag Hunt
 *
 * Cooperation for big reward vs safe defection.
 * - Both hunt stag: big payoff (4 each)
 * - Both hunt hare: small safe payoff (3 each)
 * - Stag alone: no payoff (0), hare hunter gets small payoff (3)
 */
export const STAG_HUNT: Game = {
  id: 'stag-hunt',
  name: 'Stag Hunt',
  description: 'Coordination game with risk',
  payoffMatrix: [
    [4, 0], // Stag: (S,S)=4, (S,H)=0
    [3, 3], // Hare: (H,S)=3, (H,H)=3
  ],
  numStrategies: 2,
  strategyNames: ['Stag', 'Hare'],
  isSymmetric: true,
};

/**
 * Public Goods Game
 *
 * N players can contribute to public good or free-ride.
 * Contributions are multiplied by factor r and shared equally.
 *
 * For 2-player approximation:
 * - Both contribute: net gain (r-1)
 * - One contributes: contributor loses 1-r/2, free-rider gains r/2
 * - Neither contributes: no change (0)
 */
export function createPublicGoodsGame(multiplicationFactor: number = 1.5): Game {
  const r = multiplicationFactor;
  return {
    id: 'public-goods',
    name: 'Public Goods Game',
    description: 'Contribution to public goods with free-rider problem',
    payoffMatrix: [
      [r - 1, -1 + r / 2], // Contribute: (C,C)=r-1, (C,D)=r/2-1
      [r / 2, 0], // Free-ride: (D,C)=r/2, (D,D)=0
    ],
    numStrategies: 2,
    strategyNames: ['Contribute', 'Free-ride'],
    isSymmetric: true,
  };
}

export const PUBLIC_GOODS = createPublicGoodsGame();

/**
 * Rock-Paper-Scissors
 *
 * Classic cyclic game with no pure strategy Nash equilibrium.
 */
export const ROCK_PAPER_SCISSORS: Game = {
  id: 'rock-paper-scissors',
  name: 'Rock-Paper-Scissors',
  description: 'Cyclic game with no pure Nash equilibrium',
  payoffMatrix: [
    [0, -1, 1], // Rock: tie with rock, loses to paper, beats scissors
    [1, 0, -1], // Paper: beats rock, ties with paper, loses to scissors
    [-1, 1, 0], // Scissors: loses to rock, beats paper, ties with scissors
  ],
  numStrategies: 3,
  strategyNames: ['Rock', 'Paper', 'Scissors'],
  isSymmetric: true,
};

/**
 * Battle of the Sexes
 *
 * Coordination game with conflicting preferences.
 * Couple wants to go out together but has different preferences.
 */
export const BATTLE_OF_SEXES: Game = {
  id: 'battle-of-sexes',
  name: 'Battle of the Sexes',
  description: 'Coordination with conflicting preferences',
  payoffMatrix: [
    [2, 0], // Opera lover: (O,O)=2, (O,F)=0
    [0, 1], // Football lover: (F,O)=0, (F,F)=1
  ],
  numStrategies: 2,
  strategyNames: ['Opera', 'Football'],
  isSymmetric: false,
};

/**
 * All available games
 */
export const ALL_GAMES = [
  PRISONERS_DILEMMA,
  HAWK_DOVE,
  STAG_HUNT,
  PUBLIC_GOODS,
  ROCK_PAPER_SCISSORS,
  BATTLE_OF_SEXES,
];

/**
 * Get game by ID
 */
export function getGame(gameId: string): Game | undefined {
  return ALL_GAMES.find((g) => g.id === gameId);
}

/**
 * Calculate expected payoff for a strategy against a population
 */
export function calculatePayoff(
  game: Game,
  strategy: number,
  population: number[]
): number {
  let payoff = 0;
  for (let opponent = 0; opponent < game.numStrategies; opponent++) {
    payoff += game.payoffMatrix[strategy][opponent] * population[opponent];
  }
  return payoff;
}

/**
 * Calculate all fitness values for current population
 */
export function calculateFitnessValues(
  game: Game,
  population: number[]
): number[] {
  const fitness: number[] = [];
  for (let i = 0; i < game.numStrategies; i++) {
    fitness[i] = calculatePayoff(game, i, population);
  }
  return fitness;
}

/**
 * Find Nash equilibrium (pure strategies)
 */
export function findPureNashEquilibria(game: Game): number[][] {
  const equilibria: number[][] = [];

  for (let i = 0; i < game.numStrategies; i++) {
    for (let j = 0; j < game.numStrategies; j++) {
      // Check if (i,j) is a Nash equilibrium
      let isNash = true;

      // Check if player 1 can improve by deviating
      for (let k = 0; k < game.numStrategies; k++) {
        if (game.payoffMatrix[k][j] > game.payoffMatrix[i][j]) {
          isNash = false;
          break;
        }
      }

      // Check if player 2 can improve by deviating
      if (isNash) {
        for (let k = 0; k < game.numStrategies; k++) {
          if (game.payoffMatrix[i][k] > game.payoffMatrix[i][j]) {
            isNash = false;
            break;
          }
        }
      }

      if (isNash) {
        equilibria.push([i, j]);
      }
    }
  }

  return equilibria;
}
