/**
 * Tests for dynamic pricing system
 */

import { DynamicPricer } from '../src/pricer';
import { ElasticityLearner } from '../src/elasticity-learner';
import { RLOptimizer } from '../src/rl-optimizer';
import { CompetitiveAnalyzer } from '../src/competitive-analyzer';
import { PricingSwarm } from '../src/swarm';
import { ConformalPredictor } from '../src/conformal-predictor';
import { MarketSimulator } from './market-simulator';
import { MarketContext } from '../src/types';

describe('DynamicPricer', () => {
  let pricer: DynamicPricer;
  let elasticityLearner: ElasticityLearner;
  let rlOptimizer: RLOptimizer;
  let competitiveAnalyzer: CompetitiveAnalyzer;
  let simulator: MarketSimulator;

  beforeEach(() => {
    elasticityLearner = new ElasticityLearner(':memory:');
    rlOptimizer = new RLOptimizer({ algorithm: 'q-learning', learningRate: 0.1 });
    competitiveAnalyzer = new CompetitiveAnalyzer();
    pricer = new DynamicPricer(100, elasticityLearner, rlOptimizer, competitiveAnalyzer);
    simulator = new MarketSimulator(100, 100, -1.5, 4);
  });

  test('should recommend prices for all strategies', async () => {
    const context = simulator.generateContext();
    const strategies = ['cost-plus', 'value-based', 'competition-based', 'dynamic-demand', 'time-based'];

    for (const strategy of strategies) {
      const recommendation = await pricer.recommendPrice(context, strategy);

      expect(recommendation).toBeDefined();
      expect(recommendation.price).toBeGreaterThan(0);
      expect(recommendation.strategy).toBe(strategy);
      expect(recommendation.expectedRevenue).toBeGreaterThanOrEqual(0);
    }
  });

  test('competition-based strategy should respond to competitor prices', async () => {
    const contextLowComp: MarketContext = {
      ...simulator.generateContext(),
      competitorPrices: [80, 85, 90, 95],
    };

    const contextHighComp: MarketContext = {
      ...simulator.generateContext(),
      competitorPrices: [120, 125, 130, 135],
    };

    const lowRec = await pricer.recommendPrice(contextLowComp, 'competition-based');
    const highRec = await pricer.recommendPrice(contextHighComp, 'competition-based');

    expect(lowRec.price).toBeLessThan(highRec.price);
  });

  test('dynamic-demand strategy should respond to demand changes', async () => {
    const contextLowDemand: MarketContext = {
      ...simulator.generateContext(),
      demand: 30,
    };

    const contextHighDemand: MarketContext = {
      ...simulator.generateContext(),
      demand: 150,
    };

    const lowRec = await pricer.recommendPrice(contextLowDemand, 'dynamic-demand');
    const highRec = await pricer.recommendPrice(contextHighDemand, 'dynamic-demand');

    expect(highRec.price).toBeGreaterThan(lowRec.price);
  });

  test('should learn from outcomes', async () => {
    const context = simulator.generateContext();

    // Record multiple outcomes
    for (let i = 0; i < 50; i++) {
      const ctx = simulator.generateContext();
      const price = 100 + (Math.random() - 0.5) * 20;
      const demand = simulator.simulateDemand(price, ctx);
      pricer.recordOutcome(price, demand, ctx);
    }

    const metrics = pricer.getPerformanceMetrics();
    expect(metrics.avgRevenue).toBeGreaterThan(0);
    expect(metrics.avgPrice).toBeGreaterThan(0);
    expect(metrics.totalRevenue).toBeGreaterThan(0);
  });

  test('ensemble recommendation should combine strategies', async () => {
    const context = simulator.generateContext();
    const ensemble = await pricer.recommendPrice(context);

    expect(ensemble.strategy).toBe('ensemble');
    expect(ensemble.price).toBeGreaterThan(0);
    expect(ensemble.confidence).toBeGreaterThanOrEqual(0);
    expect(ensemble.confidence).toBeLessThanOrEqual(1);
  });
});

describe('ElasticityLearner', () => {
  let learner: ElasticityLearner;
  let simulator: MarketSimulator;

  beforeEach(() => {
    learner = new ElasticityLearner(':memory:');
    simulator = new MarketSimulator(100, 100, -1.5, 4);
  });

  test('should estimate elasticity from observations', async () => {
    const context = simulator.generateContext();

    // Generate observations with known elasticity
    for (let i = 0; i < 30; i++) {
      const price = 90 + i * 2;
      const demand = simulator.simulateDemand(price, context, false);
      await learner.observe(price, demand, context);
    }

    const elasticity = learner.getElasticity(context);

    expect(elasticity.mean).toBeLessThan(0); // Should be negative
    expect(Math.abs(elasticity.mean)).toBeGreaterThan(0.5); // Should be somewhat elastic
    expect(elasticity.confidence).toBeGreaterThan(0);
  });

  test('should predict demand at different prices', async () => {
    const context = simulator.generateContext();

    // Train with observations
    for (let i = 0; i < 20; i++) {
      const price = 95 + Math.random() * 10;
      const demand = simulator.simulateDemand(price, context);
      await learner.observe(price, demand, context);
    }

    const prediction = learner.predictDemand(110, 100, 100, context);

    expect(prediction.demand).toBeGreaterThanOrEqual(0);
    expect(prediction.lower).toBeLessThanOrEqual(prediction.demand);
    expect(prediction.upper).toBeGreaterThanOrEqual(prediction.demand);
  });

  test('should learn seasonality patterns', async () => {
    // Generate week of observations
    for (let day = 0; day < 7; day++) {
      for (let hour = 0; hour < 24; hour++) {
        const context: MarketContext = {
          ...simulator.generateContext(),
          dayOfWeek: day,
          hour,
        };

        const price = 100;
        const demand = simulator.simulateDemand(price, context);
        await learner.observe(price, demand, context);
      }
    }

    const seasonality = await learner.learnSeasonality();
    expect(seasonality.size).toBeGreaterThan(0);
  });

  test('should learn promotion effects', async () => {
    const context = simulator.generateContext();

    // Record observations with and without promotions
    for (let i = 0; i < 30; i++) {
      const ctx = { ...context, isPromotion: i < 15 };
      const demand = simulator.simulateDemand(100, ctx);
      await learner.observe(100, demand, ctx);
    }

    const promoEffect = await learner.learnPromotionEffect();
    expect(promoEffect).toBeGreaterThan(1.0); // Promotions should increase demand
  });
});

describe('RLOptimizer', () => {
  let optimizer: RLOptimizer;
  let simulator: MarketSimulator;

  beforeEach(() => {
    optimizer = new RLOptimizer({
      algorithm: 'q-learning',
      learningRate: 0.2,
      epsilon: 0.3,
    });
    simulator = new MarketSimulator(100, 100, -1.5, 4);
  });

  test('should select actions', () => {
    const context = simulator.generateContext();
    const action = optimizer.selectAction(context, true);

    expect(action).toBeDefined();
    expect(action.priceMultiplier).toBeGreaterThan(0);
    expect(action.index).toBeGreaterThanOrEqual(0);
  });

  test('should learn from experiences', () => {
    const context = simulator.generateContext();
    const nextContext = simulator.generateContext();

    const action = optimizer.selectAction(context, false);
    const reward = 10;

    optimizer.learn(context, action, reward, nextContext);

    const metrics = optimizer.getMetrics();
    expect(metrics.step).toBe(1);
    expect(metrics.epsilon).toBeLessThanOrEqual(0.3);
  });

  test('should improve over time', () => {
    let totalReward = 0;

    // Training loop
    for (let episode = 0; episode < 100; episode++) {
      const context = simulator.generateContext();
      const action = optimizer.selectAction(context, true);
      const price = 100 * action.priceMultiplier;
      const demand = simulator.simulateDemand(price, context);
      const revenue = price * demand;

      const reward = revenue / 10000; // Normalize
      totalReward += reward;

      const nextContext = simulator.generateContext();
      optimizer.learn(context, action, reward, nextContext);
    }

    const metrics = optimizer.getMetrics();
    expect(metrics.statesExplored).toBeGreaterThan(0);
    expect(metrics.epsilon).toBeLessThan(0.3); // Should decay
  });

  test('should work with different algorithms', () => {
    const algorithms: Array<'q-learning' | 'dqn' | 'sarsa' | 'ppo' | 'actor-critic'> = [
      'q-learning',
      'dqn',
      'sarsa',
      'ppo',
      'actor-critic',
    ];

    for (const algo of algorithms) {
      const opt = new RLOptimizer({ algorithm: algo, learningRate: 0.1 });
      const context = simulator.generateContext();
      const action = opt.selectAction(context, false);

      expect(action).toBeDefined();
      expect(action.priceMultiplier).toBeGreaterThan(0);
    }
  });
});

describe('CompetitiveAnalyzer', () => {
  let analyzer: CompetitiveAnalyzer;

  beforeEach(() => {
    analyzer = new CompetitiveAnalyzer();
  });

  test('should analyze competitor prices', () => {
    const prices = [95, 100, 105, 110];
    const analysis = analyzer.analyze(prices);

    expect(analysis.avgPrice).toBeCloseTo(102.5, 1);
    expect(analysis.minPrice).toBe(95);
    expect(analysis.maxPrice).toBe(110);
    expect(analysis.priceDispersion).toBeGreaterThan(0);
  });

  test('should identify market structure', () => {
    // Low dispersion = commoditized
    const commoditized = [100, 101, 102, 103];
    const analysisComm = analyzer.analyze(commoditized);
    expect(analysisComm.priceDispersion).toBeLessThan(0.1);

    // High dispersion = differentiated
    const differentiated = [80, 100, 130, 160];
    const analysisDiff = analyzer.analyze(differentiated);
    expect(analysisDiff.priceDispersion).toBeGreaterThan(0.2);
  });

  test('should predict competitor response', () => {
    const currentPrices = [100, 105, 110];
    const myNewPrice = 85; // Aggressive undercut

    const response = analyzer.predictCompetitorResponse(myNewPrice, currentPrices);

    expect(response).toBeDefined();
    expect(response.expectedPrices).toHaveLength(3);
    expect(response.confidence).toBeGreaterThanOrEqual(0);
    expect(response.confidence).toBeLessThanOrEqual(1);
  });

  test('should find pricing gaps', () => {
    const prices = [80, 100, 150]; // Big gap between 100 and 150
    const gaps = analyzer.findPricingGaps(prices);

    expect(gaps.length).toBeGreaterThan(0);
    expect(gaps[0].size).toBeGreaterThan(0);
  });

  test('should track competitor behavior', () => {
    const competitorId = 'competitor_1';

    // Track price changes
    for (let i = 0; i < 10; i++) {
      analyzer.trackCompetitor(competitorId, 100 + i);
    }

    const behavior = analyzer.getCompetitorBehavior(competitorId);
    expect(behavior.trend).toBe('increasing');
    expect(behavior.avgPrice).toBeCloseTo(104.5, 1);
  });
});

describe('PricingSwarm', () => {
  let swarm: PricingSwarm;
  let elasticityLearner: ElasticityLearner;
  let rlOptimizer: RLOptimizer;
  let competitiveAnalyzer: CompetitiveAnalyzer;
  let simulator: MarketSimulator;

  beforeEach(() => {
    elasticityLearner = new ElasticityLearner(':memory:');
    rlOptimizer = new RLOptimizer({ algorithm: 'q-learning' });
    competitiveAnalyzer = new CompetitiveAnalyzer();
    simulator = new MarketSimulator();

    swarm = new PricingSwarm(
      {
        numAgents: 5,
        strategies: ['cost-plus', 'value-based', 'competition-based', 'dynamic-demand', 'time-based'],
        communicationTopology: 'mesh',
        consensusMechanism: 'weighted',
        explorationRate: 0.1,
      },
      100,
      elasticityLearner,
      rlOptimizer,
      competitiveAnalyzer
    );
  });

  test('should explore strategies in parallel', async () => {
    const context = simulator.generateContext();
    const result = await swarm.explore(context, 20); // Small trial count for speed

    expect(result.bestStrategy).toBeDefined();
    expect(result.bestPrice).toBeGreaterThan(0);
    expect(result.avgRevenue).toBeGreaterThan(0);
    expect(result.results.size).toBeGreaterThan(0);
  }, 10000);

  test('should reach consensus', async () => {
    const context = simulator.generateContext();
    const consensus = await swarm.getConsensusPrice(context);

    expect(consensus.strategy).toBe('swarm-consensus');
    expect(consensus.price).toBeGreaterThan(0);
    expect(consensus.confidence).toBeGreaterThan(0);
  });

  test('should provide swarm statistics', () => {
    const stats = swarm.getStatistics();

    expect(stats.numAgents).toBe(5);
    expect(stats.diversityScore).toBeGreaterThan(0);
    expect(stats.bestStrategy).toBeDefined();
  });
});

describe('ConformalPredictor', () => {
  let predictor: ConformalPredictor;

  beforeEach(() => {
    predictor = new ConformalPredictor(0.1); // 90% coverage
  });

  test('should calibrate from data', () => {
    const predictions = [100, 105, 95, 110, 90];
    const actuals = [102, 103, 97, 108, 92];

    predictor.calibrate(predictions, actuals);

    const pred = predictor.predict(100);
    expect(pred.point).toBe(100);
    expect(pred.lower).toBeLessThanOrEqual(100);
    expect(pred.upper).toBeGreaterThanOrEqual(100);
    expect(pred.coverage).toBeCloseTo(0.9, 1);
  });

  test('should provide prediction intervals', () => {
    // Calibrate with some data
    predictor.calibrate([100, 110, 90], [102, 108, 92]);

    const prediction = predictor.predict(105);

    expect(prediction.lower).toBeLessThan(prediction.point);
    expect(prediction.upper).toBeGreaterThan(prediction.point);
  });

  test('should validate predictions', () => {
    predictor.calibrate([100, 110, 90], [102, 108, 92]);

    const pred = predictor.predict(100);
    expect(predictor.isValid(pred, 100)).toBe(true);
    expect(predictor.isValid(pred, pred.lower - 1)).toBe(false);
    expect(predictor.isValid(pred, pred.upper + 1)).toBe(false);
  });

  test('should calculate empirical coverage', () => {
    predictor.calibrate([100, 110, 90, 95, 105], [102, 108, 92, 93, 107]);

    const predictions = [
      predictor.predict(100),
      predictor.predict(105),
      predictor.predict(95),
    ];

    const actuals = [101, 104, 96];

    const coverage = predictor.calculateCoverage(predictions, actuals);
    expect(coverage).toBeGreaterThanOrEqual(0);
    expect(coverage).toBeLessThanOrEqual(1);
  });
});

describe('Integration Tests', () => {
  test('end-to-end pricing workflow', async () => {
    const simulator = new MarketSimulator(100, 100, -1.5, 4);
    const elasticityLearner = new ElasticityLearner(':memory:');
    const rlOptimizer = new RLOptimizer({ algorithm: 'q-learning', learningRate: 0.1 });
    const competitiveAnalyzer = new CompetitiveAnalyzer();
    const pricer = new DynamicPricer(100, elasticityLearner, rlOptimizer, competitiveAnalyzer);

    // Simulate 50 time steps
    let totalRevenue = 0;

    for (let step = 0; step < 50; step++) {
      const context = simulator.generateContext();

      // Get recommendation
      const recommendation = await pricer.recommendPrice(context);

      // Simulate market response
      const actualDemand = simulator.simulateDemand(recommendation.price, context);
      const revenue = recommendation.price * actualDemand;
      totalRevenue += revenue;

      // Record outcome for learning
      pricer.recordOutcome(recommendation.price, actualDemand, context);
    }

    expect(totalRevenue).toBeGreaterThan(0);

    const metrics = pricer.getPerformanceMetrics();
    expect(metrics.avgRevenue).toBeGreaterThan(0);
    expect(metrics.totalRevenue).toBe(totalRevenue);
  }, 15000);

  test('strategy comparison via simulation', async () => {
    const simulator = new MarketSimulator(100, 100, -1.5, 4);
    const elasticityLearner = new ElasticityLearner(':memory:');
    const rlOptimizer = new RLOptimizer({ algorithm: 'q-learning' });
    const competitiveAnalyzer = new CompetitiveAnalyzer();
    const pricer = new DynamicPricer(100, elasticityLearner, rlOptimizer, competitiveAnalyzer);

    const strategies = new Map();
    strategies.set('cost-plus', (ctx: MarketContext) => 100 * 1.3);
    strategies.set('competition-based', (ctx: MarketContext) => {
      const avg = ctx.competitorPrices.reduce((a, b) => a + b, 0) / ctx.competitorPrices.length;
      return avg * 0.95;
    });

    const results = simulator.compareStrategies(strategies, 50);

    expect(results.size).toBe(2);
    expect(results.get('cost-plus')).toBeDefined();
    expect(results.get('competition-based')).toBeDefined();
  });
});
