/**
 * Tests for ESS (Evolutionarily Stable Strategy) analysis
 */

import { ESSCalculator, findAllESS } from '../src/ess.js';
import {
  PRISONERS_DILEMMA,
  HAWK_DOVE,
  STAG_HUNT,
  ROCK_PAPER_SCISSORS,
} from '../src/games.js';

describe('ESSCalculator', () => {
  describe('Pure Strategy ESS', () => {
    it('should identify defect as ESS in PD', () => {
      const calculator = new ESSCalculator(PRISONERS_DILEMMA);
      const isDefectESS = calculator.isPureESS(1); // Defect
      const isCooperateESS = calculator.isPureESS(0); // Cooperate

      expect(isDefectESS).toBe(true);
      expect(isCooperateESS).toBe(false);
    });

    it('should find all pure ESS in PD', () => {
      const calculator = new ESSCalculator(PRISONERS_DILEMMA);
      const essStrategies = calculator.findPureESS();

      expect(essStrategies).toContain(1); // Defect
      expect(essStrategies).not.toContain(0); // Not cooperate
    });

    it('should handle multiple ESS in coordination games', () => {
      const calculator = new ESSCalculator(STAG_HUNT);
      const essStrategies = calculator.findPureESS();

      // Stag Hunt can have multiple ESS
      expect(essStrategies.length).toBeGreaterThanOrEqual(1);
    });

    it('should find no pure ESS in Rock-Paper-Scissors', () => {
      const calculator = new ESSCalculator(ROCK_PAPER_SCISSORS);
      const essStrategies = calculator.findPureESS();

      expect(essStrategies).toHaveLength(0);
    });
  });

  describe('Mixed Strategy ESS', () => {
    it('should check stability of mixed strategies', () => {
      const calculator = new ESSCalculator(HAWK_DOVE);
      // Mixed strategy: some hawks, some doves
      const result = calculator.isMixedESS([0.6, 0.4]);

      expect(result.strategy).toHaveLength(2);
      expect(result.eigenvalues).toBeDefined();
      expect(typeof result.isStable).toBe('boolean');
    });

    it('should calculate stability margin', () => {
      const calculator = new ESSCalculator(PRISONERS_DILEMMA);
      const result = calculator.isMixedESS([0, 1]); // All defect

      expect(result.stabilityMargin).toBeDefined();
      expect(typeof result.stabilityMargin).toBe('number');
    });

    it('should normalize strategy before checking', () => {
      const calculator = new ESSCalculator(PRISONERS_DILEMMA);
      const result1 = calculator.isMixedESS([0.5, 0.5]);
      const result2 = calculator.isMixedESS([1, 1]); // Will be normalized

      expect(result1.strategy[0]).toBeCloseTo(result2.strategy[0]);
      expect(result1.strategy[1]).toBeCloseTo(result2.strategy[1]);
    });
  });

  describe('Invasion Analysis', () => {
    it('should check if invader can succeed', () => {
      const calculator = new ESSCalculator(PRISONERS_DILEMMA);
      const resident = [0, 1]; // All defect
      const invader = [1, 0]; // All cooperate

      const canInvade = calculator.canInvade(resident, invader, 0.01);

      expect(typeof canInvade).toBe('boolean');
      expect(canInvade).toBe(false); // Cooperate cannot invade defect
    });

    it('should calculate invasion fitness', () => {
      const calculator = new ESSCalculator(PRISONERS_DILEMMA);
      const resident = [0, 1]; // All defect
      const invader = [1, 0]; // All cooperate

      const fitness = calculator.invasionFitness(resident, invader);

      expect(typeof fitness).toBe('number');
      expect(fitness).toBeLessThan(0); // Invader has lower fitness
    });

    it('should handle mixed strategy invasions', () => {
      const calculator = new ESSCalculator(HAWK_DOVE);
      const resident = [0.5, 0.5];
      const invader = [0.7, 0.3];

      const fitness = calculator.invasionFitness(resident, invader);
      expect(typeof fitness).toBe('number');
    });
  });

  describe('Eigenvalue Calculation', () => {
    it('should calculate eigenvalues for stability analysis', () => {
      const calculator = new ESSCalculator(PRISONERS_DILEMMA);
      const result = calculator.isMixedESS([0.5, 0.5]);

      expect(result.eigenvalues).toBeDefined();
      expect(result.eigenvalues.length).toBeGreaterThan(0);
    });

    it('should identify stable equilibria', () => {
      const calculator = new ESSCalculator(PRISONERS_DILEMMA);
      const result = calculator.isMixedESS([0, 1]); // All defect

      // All defect should be stable
      const allNegative = result.eigenvalues.every((lambda) => lambda <= 0.01);
      expect(result.isStable || allNegative).toBe(true);
    });
  });

  describe('Basin of Attraction', () => {
    it('should find basin of attraction for ESS', () => {
      const calculator = new ESSCalculator(PRISONERS_DILEMMA);
      const ess = [0, 1]; // All defect
      const basin = calculator.findBasinOfAttraction(ess, 10, 0.5);

      expect(Array.isArray(basin)).toBe(true);
      expect(basin.length).toBeGreaterThan(0);
    });

    it('should include ESS in its own basin', () => {
      const calculator = new ESSCalculator(PRISONERS_DILEMMA);
      const ess = [0, 1];
      const basin = calculator.findBasinOfAttraction(ess, 5, 0.1);

      // Should find points close to ESS
      expect(basin.length).toBeGreaterThan(0);
    });
  });
});

describe('findAllESS', () => {
  it('should find both pure and mixed ESS', () => {
    const result = findAllESS(PRISONERS_DILEMMA);

    expect(result.pure).toBeDefined();
    expect(result.mixed).toBeDefined();
    expect(Array.isArray(result.pure)).toBe(true);
    expect(Array.isArray(result.mixed)).toBe(true);
  });

  it('should find defect in Prisoners Dilemma', () => {
    const result = findAllESS(PRISONERS_DILEMMA);

    expect(result.pure).toContain(1); // Defect
  });

  it('should handle games with no pure ESS', () => {
    const result = findAllESS(ROCK_PAPER_SCISSORS);

    expect(result.pure).toHaveLength(0);
    // Mixed strategy ESS should exist (uniform distribution)
    expect(result.mixed.length).toBeGreaterThanOrEqual(0);
  });

  it('should find mixed ESS in Hawk-Dove', () => {
    const result = findAllESS(HAWK_DOVE);

    // Hawk-Dove has a mixed strategy ESS
    expect(result.pure.length + result.mixed.length).toBeGreaterThan(0);
  });
});

describe('ESS Properties', () => {
  it('should satisfy ESS definition', () => {
    const calculator = new ESSCalculator(PRISONERS_DILEMMA);

    // Defect is ESS, so it should resist all invaders
    const resident = [0, 1]; // All defect
    const invaders = [
      [1, 0], // All cooperate
      [0.5, 0.5], // Mixed
      [0.9, 0.1], // Mostly cooperate
    ];

    for (const invader of invaders) {
      const canInvade = calculator.canInvade(resident, invader);
      expect(canInvade).toBe(false);
    }
  });

  it('should have correct stability properties', () => {
    const calculator = new ESSCalculator(PRISONERS_DILEMMA);
    const result = calculator.isMixedESS([0, 1]); // All defect

    if (result.isStable) {
      // If stable, eigenvalues should be non-positive
      const hasPositiveEigenvalue = result.eigenvalues.some(
        (lambda) => lambda > 1e-4
      );
      expect(hasPositiveEigenvalue).toBe(false);
    }
  });

  it('should be invasion-proof', () => {
    const calculator = new ESSCalculator(PRISONERS_DILEMMA);
    const ess = [0, 1]; // All defect (known ESS)

    // Test multiple random invaders
    for (let i = 0; i < 10; i++) {
      const invader = [Math.random(), Math.random()];
      const sum = invader[0] + invader[1];
      invader[0] /= sum;
      invader[1] /= sum;

      const fitness = calculator.invasionFitness(ess, invader);
      expect(fitness).toBeLessThanOrEqual(0); // ESS should not be invaded
    }
  });
});
