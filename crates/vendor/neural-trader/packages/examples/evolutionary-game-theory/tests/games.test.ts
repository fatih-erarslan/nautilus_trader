/**
 * Tests for game definitions
 */

import {
  PRISONERS_DILEMMA,
  HAWK_DOVE,
  STAG_HUNT,
  PUBLIC_GOODS,
  ROCK_PAPER_SCISSORS,
  calculatePayoff,
  calculateFitnessValues,
  findPureNashEquilibria,
  createHawkDoveGame,
  createPublicGoodsGame,
} from '../src/games.js';

describe('Game Definitions', () => {
  describe("Prisoner's Dilemma", () => {
    it('should have correct payoff matrix', () => {
      expect(PRISONERS_DILEMMA.payoffMatrix).toEqual([
        [3, 0],
        [5, 1],
      ]);
    });

    it('should be symmetric', () => {
      expect(PRISONERS_DILEMMA.isSymmetric).toBe(true);
    });

    it('should have defection as dominant strategy', () => {
      // Defect dominates cooperate
      expect(PRISONERS_DILEMMA.payoffMatrix[1][0]).toBeGreaterThan(
        PRISONERS_DILEMMA.payoffMatrix[0][0]
      ); // D > C when opponent cooperates
      expect(PRISONERS_DILEMMA.payoffMatrix[1][1]).toBeGreaterThan(
        PRISONERS_DILEMMA.payoffMatrix[0][1]
      ); // D > C when opponent defects
    });
  });

  describe('Hawk-Dove Game', () => {
    it('should have correct payoff structure', () => {
      const game = createHawkDoveGame(4, 6);
      expect(game.payoffMatrix[0][0]).toBe(2); // Both dove: V/2
      expect(game.payoffMatrix[1][1]).toBe(-1); // Both hawk: (V-C)/2
    });

    it('should favor dove when cost > value', () => {
      const game = createHawkDoveGame(4, 10);
      expect(game.payoffMatrix[1][1]).toBeLessThan(0); // Fighting is costly
    });
  });

  describe('Stag Hunt', () => {
    it('should have two pure Nash equilibria', () => {
      const equilibria = findPureNashEquilibria(STAG_HUNT);
      expect(equilibria.length).toBeGreaterThanOrEqual(2);
    });

    it('should favor coordination on stag', () => {
      expect(STAG_HUNT.payoffMatrix[0][0]).toBeGreaterThan(
        STAG_HUNT.payoffMatrix[1][1]
      ); // Stag-Stag > Hare-Hare
    });
  });

  describe('Public Goods Game', () => {
    it('should incentivize free-riding', () => {
      const game = createPublicGoodsGame(1.5);
      // Free-rider gets more when others contribute
      expect(game.payoffMatrix[1][0]).toBeGreaterThan(
        game.payoffMatrix[0][0]
      );
    });

    it('should have tragedy of commons', () => {
      const game = PUBLIC_GOODS;
      // Both contribute is better than both free-ride
      expect(game.payoffMatrix[0][0]).toBeGreaterThan(
        game.payoffMatrix[1][1]
      );
    });
  });

  describe('Rock-Paper-Scissors', () => {
    it('should have cyclic dominance', () => {
      const game = ROCK_PAPER_SCISSORS;
      // Rock beats Scissors
      expect(game.payoffMatrix[0][2]).toBeGreaterThan(game.payoffMatrix[0][0]);
      // Paper beats Rock
      expect(game.payoffMatrix[1][0]).toBeGreaterThan(game.payoffMatrix[1][1]);
      // Scissors beats Paper
      expect(game.payoffMatrix[2][1]).toBeGreaterThan(game.payoffMatrix[2][2]);
    });

    it('should have no pure Nash equilibrium', () => {
      const equilibria = findPureNashEquilibria(ROCK_PAPER_SCISSORS);
      expect(equilibria.length).toBe(0);
    });
  });
});

describe('Game Utilities', () => {
  describe('calculatePayoff', () => {
    it('should calculate correct payoff', () => {
      const population = [0.5, 0.5]; // Equal mix
      const cooperatePayoff = calculatePayoff(PRISONERS_DILEMMA, 0, population);
      const defectPayoff = calculatePayoff(PRISONERS_DILEMMA, 1, population);

      expect(cooperatePayoff).toBe(1.5); // 0.5*3 + 0.5*0
      expect(defectPayoff).toBe(3); // 0.5*5 + 0.5*1
    });

    it('should handle pure populations', () => {
      const allCooperate = [1, 0];
      const payoff = calculatePayoff(PRISONERS_DILEMMA, 0, allCooperate);
      expect(payoff).toBe(3); // C vs C
    });
  });

  describe('calculateFitnessValues', () => {
    it('should calculate fitness for all strategies', () => {
      const population = [0.7, 0.3];
      const fitness = calculateFitnessValues(PRISONERS_DILEMMA, population);

      expect(fitness).toHaveLength(2);
      expect(fitness[0]).toBeGreaterThanOrEqual(0);
      expect(fitness[1]).toBeGreaterThan(fitness[0]); // Defect does better
    });
  });

  describe('findPureNashEquilibria', () => {
    it('should find defect-defect in PD', () => {
      const equilibria = findPureNashEquilibria(PRISONERS_DILEMMA);
      expect(equilibria).toContainEqual([1, 1]); // (Defect, Defect)
    });

    it('should find both equilibria in Stag Hunt', () => {
      const equilibria = findPureNashEquilibria(STAG_HUNT);
      expect(equilibria.length).toBeGreaterThanOrEqual(2);
    });

    it('should handle mixed strategy games', () => {
      const equilibria = findPureNashEquilibria(ROCK_PAPER_SCISSORS);
      // RPS has no pure Nash equilibrium
      expect(equilibria.length).toBe(0);
    });
  });
});
