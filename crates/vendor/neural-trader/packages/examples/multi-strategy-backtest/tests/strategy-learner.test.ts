/**
 * Test suite for StrategyLearner
 */

import { StrategyLearner } from '../src/strategy-learner';
import { PortfolioState, StrategyPerformance, RegimeDetection } from '../src/types';

describe('StrategyLearner', () => {
  let learner: StrategyLearner;

  beforeEach(async () => {
    learner = new StrategyLearner({
      learningRate: 0.1,
      discountFactor: 0.9,
      explorationRate: 0.5,
      experienceBufferSize: 100
    }, 'test-learner');

    await learner.initialize();
  });

  describe('Initialization', () => {
    it('should initialize with default config', () => {
      const stats = learner.getStats();
      expect(stats.episodes).toBe(0);
      expect(stats.explorationRate).toBe(0.5);
    });
  });

  describe('Learning', () => {
    it('should learn from backtest results', async () => {
      const portfolioStates = generatePortfolioStates(50);
      const performances = generatePerformances();

      await learner.learnFromBacktest(portfolioStates, performances);

      const stats = learner.getStats();
      expect(stats.episodes).toBe(1);
      expect(stats.experienceCount).toBeGreaterThan(0);
    });

    it('should update Q-table', async () => {
      const portfolioStates = generatePortfolioStates(50);
      const performances = generatePerformances();

      await learner.learnFromBacktest(portfolioStates, performances);

      const stats = learner.getStats();
      expect(stats.qTableSize).toBeGreaterThan(0);
    });

    it('should decay exploration rate', async () => {
      const portfolioStates = generatePortfolioStates(50);
      const performances = generatePerformances();

      const initialRate = learner.getStats().explorationRate;

      await learner.learnFromBacktest(portfolioStates, performances);

      const finalRate = learner.getStats().explorationRate;
      expect(finalRate).toBeLessThan(initialRate);
    });

    it('should accumulate total reward', async () => {
      const portfolioStates = generatePortfolioStates(50);
      const performances = generatePerformances();

      await learner.learnFromBacktest(portfolioStates, performances);

      const stats = learner.getStats();
      expect(typeof stats.totalReward).toBe('number');
    });
  });

  describe('Optimal Weight Selection', () => {
    it('should return optimal weights', () => {
      const currentState = generatePortfolioStates(1)[0];
      const strategies = ['momentum', 'mean-reversion'];

      const weights = learner.getOptimalWeights(currentState, strategies);

      expect(Object.keys(weights).length).toBe(strategies.length);

      const sum = Object.values(weights).reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 5);
    });

    it('should normalize weights to sum to 1', () => {
      const currentState = generatePortfolioStates(1)[0];
      const strategies = ['momentum', 'mean-reversion', 'pairs-trading'];

      const weights = learner.getOptimalWeights(currentState, strategies);

      const sum = Object.values(weights).reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 5);
    });
  });

  describe('Experience Replay', () => {
    it('should store experiences in buffer', async () => {
      const portfolioStates = generatePortfolioStates(20);
      const performances = generatePerformances();

      await learner.learnFromBacktest(portfolioStates, performances);

      const stats = learner.getStats();
      expect(stats.experienceCount).toBeGreaterThan(0);
    });

    it('should limit buffer size', async () => {
      learner = new StrategyLearner({
        experienceBufferSize: 10
      }, 'test-learner-small');

      const portfolioStates = generatePortfolioStates(50);
      const performances = generatePerformances();

      await learner.learnFromBacktest(portfolioStates, performances);

      const stats = learner.getStats();
      expect(stats.experienceCount).toBeLessThanOrEqual(10);
    });
  });

  describe('Best Performances Tracking', () => {
    it('should track best performances', async () => {
      const portfolioStates = generatePortfolioStates(50);
      const performances = generatePerformances();

      await learner.learnFromBacktest(portfolioStates, performances);

      const stats = learner.getStats();
      expect(stats.bestPerformances.length).toBeGreaterThan(0);
    });

    it('should update best performances when improved', async () => {
      const portfolioStates = generatePortfolioStates(50);

      // First run with mediocre performance
      const performances1 = generatePerformances(1.0);
      await learner.learnFromBacktest(portfolioStates, performances1);

      const stats1 = learner.getStats();
      const initialBestSharpe = stats1.bestPerformances[0]?.sharpeRatio || 0;

      // Second run with better performance
      const performances2 = generatePerformances(2.0);
      await learner.learnFromBacktest(portfolioStates, performances2);

      const stats2 = learner.getStats();
      const newBestSharpe = stats2.bestPerformances[0]?.sharpeRatio || 0;

      expect(newBestSharpe).toBeGreaterThanOrEqual(initialBestSharpe);
    });
  });

  describe('Reset', () => {
    it('should reset learning state', async () => {
      const portfolioStates = generatePortfolioStates(50);
      const performances = generatePerformances();

      await learner.learnFromBacktest(portfolioStates, performances);

      learner.reset();

      const stats = learner.getStats();
      expect(stats.episodes).toBe(0);
      expect(stats.qTableSize).toBe(0);
      expect(stats.experienceCount).toBe(0);
    });
  });
});

// Helper functions
function generatePortfolioStates(count: number): PortfolioState[] {
  const states: PortfolioState[] = [];
  let equity = 100000;

  for (let i = 0; i < count; i++) {
    equity *= (1 + (Math.random() - 0.48) / 100);

    const regime: RegimeDetection = {
      regime: ['bull', 'bear', 'sideways', 'high-volatility', 'low-volatility'][Math.floor(Math.random() * 5)] as any,
      confidence: Math.random(),
      timestamp: Date.now() + (i * 24 * 60 * 60 * 1000),
      indicators: {
        trend: (Math.random() - 0.5) * 0.1,
        volatility: Math.random() * 0.05,
        correlation: Math.random()
      }
    };

    states.push({
      timestamp: Date.now() + (i * 24 * 60 * 60 * 1000),
      equity,
      cash: equity * 0.3,
      positions: [],
      strategyWeights: {
        momentum: Math.random() * 0.5,
        'mean-reversion': Math.random() * 0.5
      },
      regime
    });
  }

  return states;
}

function generatePerformances(sharpeMultiplier: number = 1.0): StrategyPerformance[] {
  return [
    {
      strategyName: 'momentum',
      totalReturn: Math.random() * 0.2,
      sharpeRatio: Math.random() * 2 * sharpeMultiplier,
      maxDrawdown: -Math.random() * 0.15,
      winRate: 0.5 + Math.random() * 0.2,
      profitFactor: 1 + Math.random(),
      trades: Math.floor(Math.random() * 100) + 10,
      avgWin: Math.random() * 0.02,
      avgLoss: -Math.random() * 0.015,
      calmarRatio: Math.random() * 1.5,
      sortinoRatio: Math.random() * 2.5
    },
    {
      strategyName: 'mean-reversion',
      totalReturn: Math.random() * 0.15,
      sharpeRatio: Math.random() * 1.8 * sharpeMultiplier,
      maxDrawdown: -Math.random() * 0.12,
      winRate: 0.5 + Math.random() * 0.15,
      profitFactor: 1 + Math.random() * 0.8,
      trades: Math.floor(Math.random() * 100) + 10,
      avgWin: Math.random() * 0.018,
      avgLoss: -Math.random() * 0.014,
      calmarRatio: Math.random() * 1.3,
      sortinoRatio: Math.random() * 2.2
    }
  ];
}
