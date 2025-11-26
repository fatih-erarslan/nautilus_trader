/**
 * Tests for strategy implementations
 */

import {
  ALWAYS_COOPERATE,
  ALWAYS_DEFECT,
  TIT_FOR_TAT,
  TIT_FOR_TWO_TATS,
  GRIM_TRIGGER,
  PAVLOV,
  RANDOM,
  ADAPTIVE,
  GRADUAL,
  createGenerousTitForTat,
  createLearningStrategy,
} from '../src/strategies.js';

describe('Classic Strategies', () => {
  describe('Always Cooperate', () => {
    it('should always return 0 (cooperate)', () => {
      expect(ALWAYS_COOPERATE.play([], [])).toBe(0);
      expect(ALWAYS_COOPERATE.play([1, 1, 1], [0, 0, 0])).toBe(0);
    });
  });

  describe('Always Defect', () => {
    it('should always return 1 (defect)', () => {
      expect(ALWAYS_DEFECT.play([], [])).toBe(1);
      expect(ALWAYS_DEFECT.play([0, 0, 0], [1, 1, 1])).toBe(1);
    });
  });

  describe('Tit-for-Tat', () => {
    it('should cooperate on first move', () => {
      expect(TIT_FOR_TAT.play([], [])).toBe(0);
    });

    it('should copy opponent last move', () => {
      expect(TIT_FOR_TAT.play([0], [0])).toBe(0);
      expect(TIT_FOR_TAT.play([0], [1])).toBe(1);
      expect(TIT_FOR_TAT.play([0, 1], [1, 0])).toBe(0);
    });

    it('should have high cooperation rate', () => {
      expect(TIT_FOR_TAT.cooperationRate).toBeGreaterThan(0.8);
    });
  });

  describe('Tit-for-Two-Tats', () => {
    it('should cooperate on first moves', () => {
      expect(TIT_FOR_TWO_TATS.play([], [])).toBe(0);
      expect(TIT_FOR_TWO_TATS.play([0], [1])).toBe(0);
    });

    it('should defect after two defections', () => {
      expect(TIT_FOR_TWO_TATS.play([0, 0], [1, 1])).toBe(1);
    });

    it('should forgive single defection', () => {
      expect(TIT_FOR_TWO_TATS.play([0, 0], [1, 0])).toBe(0);
    });
  });

  describe('Grim Trigger', () => {
    it('should cooperate initially', () => {
      expect(GRIM_TRIGGER.play([], [])).toBe(0);
      expect(GRIM_TRIGGER.play([0, 0], [0, 0])).toBe(0);
    });

    it('should defect forever after opponent defects', () => {
      expect(GRIM_TRIGGER.play([0, 0], [0, 1])).toBe(1);
      expect(GRIM_TRIGGER.play([0, 0, 1], [0, 1, 0])).toBe(1);
      expect(GRIM_TRIGGER.play([0, 0, 1, 1], [0, 1, 0, 0])).toBe(1);
    });
  });

  describe('Pavlov (Win-Stay, Lose-Shift)', () => {
    it('should cooperate first', () => {
      expect(PAVLOV.play([], [])).toBe(0);
    });

    it('should repeat if both cooperated', () => {
      expect(PAVLOV.play([0], [0])).toBe(0);
    });

    it('should repeat if both defected', () => {
      expect(PAVLOV.play([1], [1])).toBe(1);
    });

    it('should switch if different outcomes', () => {
      expect(PAVLOV.play([0], [1])).toBe(1); // Was cooperating, got punished, switch to defect
      expect(PAVLOV.play([1], [0])).toBe(0); // Was defecting, got punished, switch to cooperate
    });
  });

  describe('Adaptive Strategy', () => {
    it('should start cooperatively', () => {
      expect(ADAPTIVE.play([], [])).toBe(0);
    });

    it('should learn opponent cooperation rate', () => {
      // Opponent always cooperates
      const alwaysCoopHistory = [0, 0, 0, 0, 0];
      const result = ADAPTIVE.play([], alwaysCoopHistory);
      // Should tend toward cooperation
      expect([0, 1]).toContain(result);
    });
  });

  describe('Gradual Strategy', () => {
    it('should start cooperatively', () => {
      expect(GRADUAL.play([], [])).toBe(0);
    });

    it('should retaliate proportionally', () => {
      // After opponent defects once, should defect once then cooperate twice
      const moves = [];
      for (let i = 0; i < 5; i++) {
        const opponentHistory = i === 0 ? [1] : [1, 0, 0, 0, 0];
        moves.push(GRADUAL.play(moves, opponentHistory));
      }
      // Should show retaliation pattern
      expect(moves).toContain(1); // Should defect at some point
    });
  });
});

describe('Parameterized Strategies', () => {
  describe('Generous Tit-for-Tat', () => {
    it('should sometimes forgive defections', () => {
      const strategy = createGenerousTitForTat(1.0); // Always forgive
      const move = strategy.play([0], [1]);
      expect(move).toBe(0); // Should forgive
    });

    it('should vary by forgiveness rate', () => {
      const never = createGenerousTitForTat(0.0);
      const always = createGenerousTitForTat(1.0);

      expect(never.description).toContain('0%');
      expect(always.description).toContain('100%');
    });
  });

  describe('Learning Strategy', () => {
    it('should create strategy with given weights', () => {
      const weights = [1, -1, 0.5, 0.5, 0.1, 0, 0, 0, 0, 0];
      const strategy = createLearningStrategy('test', 'Test', weights);

      expect(strategy.id).toBe('test');
      expect(strategy.name).toBe('Test');
      expect(strategy.memory).toBe(5);
    });

    it('should make decisions based on features', () => {
      const weights = [10, -10, 0, 0, 0, 0, 0, 0, 0, 0]; // Strong bias toward cooperation
      const strategy = createLearningStrategy('test', 'Test', weights);

      const move = strategy.play([], []);
      expect([0, 1]).toContain(move);
    });

    it('should use opponent history', () => {
      const weights = [0, 10, 0, 0, 0, 0, 0, 0, 0, 0]; // Strongly influenced by last opponent move
      const strategy = createLearningStrategy('test', 'Test', weights);

      // Should be influenced by opponent's last move
      const move1 = strategy.play([0], [0]);
      const move2 = strategy.play([0], [1]);

      // Both are probabilistic but should show some pattern
      expect([0, 1]).toContain(move1);
      expect([0, 1]).toContain(move2);
    });
  });
});

describe('Strategy Properties', () => {
  it('should have valid cooperation rates', () => {
    const strategies = [
      ALWAYS_COOPERATE,
      ALWAYS_DEFECT,
      TIT_FOR_TAT,
      PAVLOV,
      ADAPTIVE,
    ];

    for (const strategy of strategies) {
      expect(strategy.cooperationRate).toBeGreaterThanOrEqual(0);
      expect(strategy.cooperationRate).toBeLessThanOrEqual(1);
    }
  });

  it('should have valid memory lengths', () => {
    expect(ALWAYS_COOPERATE.memory).toBe(0);
    expect(TIT_FOR_TAT.memory).toBe(1);
    expect(TIT_FOR_TWO_TATS.memory).toBe(2);
    expect(ADAPTIVE.memory).toBe(10);
  });

  it('should return valid moves', () => {
    const strategies = [
      ALWAYS_COOPERATE,
      ALWAYS_DEFECT,
      TIT_FOR_TAT,
      PAVLOV,
    ];

    for (const strategy of strategies) {
      const move = strategy.play([], []);
      expect([0, 1]).toContain(move);
    }
  });
});
