/**
 * Classic strategies for iterated games
 */

import type { Strategy, GameHistory } from './types.js';

/**
 * Always cooperate (strategy 0)
 */
export const ALWAYS_COOPERATE: Strategy = {
  id: 'always-cooperate',
  name: 'Always Cooperate',
  description: 'Always plays cooperate',
  cooperationRate: 1.0,
  memory: 0,
  play: () => 0,
};

/**
 * Always defect (strategy 1)
 */
export const ALWAYS_DEFECT: Strategy = {
  id: 'always-defect',
  name: 'Always Defect',
  description: 'Always plays defect',
  cooperationRate: 0.0,
  memory: 0,
  play: () => 1,
};

/**
 * Tit-for-Tat: Start with cooperation, then copy opponent's last move
 */
export const TIT_FOR_TAT: Strategy = {
  id: 'tit-for-tat',
  name: 'Tit-for-Tat',
  description: 'Cooperates first, then copies opponent',
  cooperationRate: 0.9,
  memory: 1,
  play: (history: GameHistory, opponentHistory: GameHistory) => {
    if (opponentHistory.length === 0) return 0; // Cooperate first
    return opponentHistory[opponentHistory.length - 1]; // Copy last move
  },
};

/**
 * Tit-for-Two-Tats: Only retaliate after two defections
 */
export const TIT_FOR_TWO_TATS: Strategy = {
  id: 'tit-for-two-tats',
  name: 'Tit-for-Two-Tats',
  description: 'Defects only after two consecutive opponent defections',
  cooperationRate: 0.95,
  memory: 2,
  play: (history: GameHistory, opponentHistory: GameHistory) => {
    if (opponentHistory.length < 2) return 0; // Cooperate first
    const lastTwo = opponentHistory.slice(-2);
    return lastTwo[0] === 1 && lastTwo[1] === 1 ? 1 : 0;
  },
};

/**
 * Grim Trigger: Cooperate until opponent defects once, then defect forever
 */
export const GRIM_TRIGGER: Strategy = {
  id: 'grim-trigger',
  name: 'Grim Trigger',
  description: 'Cooperates until first defection, then defects forever',
  cooperationRate: 0.7,
  memory: Infinity,
  play: (history: GameHistory, opponentHistory: GameHistory) => {
    if (opponentHistory.includes(1)) return 1; // Defect if opponent ever defected
    return 0; // Otherwise cooperate
  },
};

/**
 * Pavlov (Win-Stay, Lose-Shift): Repeat if won, change if lost
 */
export const PAVLOV: Strategy = {
  id: 'pavlov',
  name: 'Pavlov',
  description: 'Win-Stay, Lose-Shift strategy',
  cooperationRate: 0.85,
  memory: 1,
  play: (history: GameHistory, opponentHistory: GameHistory) => {
    if (history.length === 0) return 0; // Cooperate first

    const lastMove = history[history.length - 1];
    const lastOpponent = opponentHistory[opponentHistory.length - 1];

    // Win-stay: if both cooperated or both defected, repeat
    if (lastMove === lastOpponent) {
      return lastMove;
    }

    // Lose-shift: if different outcomes, switch
    return 1 - lastMove;
  },
};

/**
 * Random strategy
 */
export const RANDOM: Strategy = {
  id: 'random',
  name: 'Random',
  description: 'Plays randomly with 50% cooperation',
  cooperationRate: 0.5,
  memory: 0,
  play: () => Math.random() < 0.5 ? 0 : 1,
};

/**
 * Generous Tit-for-Tat: Like TFT but forgives with some probability
 */
export function createGenerousTitForTat(forgivenessRate: number = 0.1): Strategy {
  return {
    id: 'generous-tit-for-tat',
    name: 'Generous Tit-for-Tat',
    description: `TFT with ${forgivenessRate * 100}% forgiveness rate`,
    cooperationRate: 0.9,
    memory: 1,
    play: (history: GameHistory, opponentHistory: GameHistory) => {
      if (opponentHistory.length === 0) return 0;
      const lastOpponentMove = opponentHistory[opponentHistory.length - 1];

      // If opponent defected, forgive with some probability
      if (lastOpponentMove === 1 && Math.random() < forgivenessRate) {
        return 0; // Forgive
      }

      return lastOpponentMove; // Otherwise copy
    },
  };
}

/**
 * Adaptive strategy: Learns opponent's cooperation rate
 */
export const ADAPTIVE: Strategy = {
  id: 'adaptive',
  name: 'Adaptive',
  description: 'Learns and matches opponent cooperation rate',
  cooperationRate: 0.75,
  memory: 10,
  play: (history: GameHistory, opponentHistory: GameHistory) => {
    if (opponentHistory.length < 5) return 0; // Start cooperative

    // Calculate opponent's recent cooperation rate
    const recent = opponentHistory.slice(-10);
    const cooperationRate = recent.filter((m) => m === 0).length / recent.length;

    // Match opponent's rate with slight bias toward cooperation
    return Math.random() < cooperationRate + 0.1 ? 0 : 1;
  },
};

/**
 * Gradual: Increases retaliation with each defection
 */
export const GRADUAL: Strategy = {
  id: 'gradual',
  name: 'Gradual',
  description: 'Increases punishment for each defection',
  cooperationRate: 0.8,
  memory: Infinity,
  play: (history: GameHistory, opponentHistory: GameHistory) => {
    if (opponentHistory.length === 0) return 0;

    // Count opponent's total defections
    const defections = opponentHistory.filter((m) => m === 1).length;

    // Count our recent retaliation defections
    const recentRetaliations = history.slice(-defections).filter((m) => m === 1).length;

    // If we haven't retaliated enough, defect
    if (recentRetaliations < defections) {
      return 1;
    }

    // After retaliation, cooperate twice as a peace offering
    const lastN = history.slice(-2);
    if (lastN.length === 2 && lastN.every((m) => m === 0)) {
      return 0;
    }

    return 0; // Default to cooperation
  },
};

/**
 * Probe: Test opponent occasionally
 */
export const PROBE: Strategy = {
  id: 'probe',
  name: 'Probe',
  description: 'Occasionally tests opponent with defection',
  cooperationRate: 0.85,
  memory: 5,
  play: (history: GameHistory, opponentHistory: GameHistory) => {
    // Occasionally probe with defection (5% chance)
    if (Math.random() < 0.05) return 1;

    // Otherwise play Tit-for-Tat
    if (opponentHistory.length === 0) return 0;
    return opponentHistory[opponentHistory.length - 1];
  },
};

/**
 * All classic strategies
 */
export const CLASSIC_STRATEGIES: Strategy[] = [
  ALWAYS_COOPERATE,
  ALWAYS_DEFECT,
  TIT_FOR_TAT,
  TIT_FOR_TWO_TATS,
  GRIM_TRIGGER,
  PAVLOV,
  RANDOM,
  createGenerousTitForTat(0.1),
  ADAPTIVE,
  GRADUAL,
  PROBE,
];

/**
 * Get strategy by ID
 */
export function getStrategy(strategyId: string): Strategy | undefined {
  return CLASSIC_STRATEGIES.find((s) => s.id === strategyId);
}

/**
 * Create a custom strategy with learning
 */
export function createLearningStrategy(
  id: string,
  name: string,
  weights: number[]
): Strategy {
  return {
    id,
    name,
    description: 'Learned strategy with weighted features',
    cooperationRate: 0.5,
    memory: 5,
    play: (history: GameHistory, opponentHistory: GameHistory) => {
      if (opponentHistory.length === 0) return 0;

      // Feature extraction
      const features = [
        1.0, // Bias
        opponentHistory[opponentHistory.length - 1], // Last move
        opponentHistory.slice(-5).filter((m) => m === 0).length / Math.min(5, opponentHistory.length), // Recent cooperation
        opponentHistory.filter((m) => m === 0).length / opponentHistory.length, // Total cooperation
        history.length, // Game length
      ];

      // Weighted sum
      let score = 0;
      for (let i = 0; i < Math.min(weights.length, features.length); i++) {
        score += weights[i] * features[i];
      }

      // Sigmoid activation
      const probability = 1 / (1 + Math.exp(-score));
      return Math.random() < probability ? 0 : 1;
    },
  };
}
