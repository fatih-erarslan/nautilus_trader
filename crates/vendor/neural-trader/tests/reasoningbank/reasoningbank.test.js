/**
 * ReasoningBank Integration Tests
 *
 * Comprehensive tests for ReasoningBank learning system
 */

const { ReasoningBankSwarmLearner, LearningMode, VerdictScore } = require('../../src/reasoningbank');
const { SwarmCoordinator, TOPOLOGY } = require('../../src/e2b/swarm-coordinator');
const { addLearningCapabilities } = require('../../src/reasoningbank/swarm-coordinator-integration');
const { E2BMonitor } = require('../../src/e2b/monitor-and-scale');
const { addLearningMetrics } = require('../../src/reasoningbank/e2b-monitor-integration');

describe('ReasoningBank Swarm Learning', () => {
  let learner;
  const swarmId = 'test-swarm-001';

  beforeEach(() => {
    learner = new ReasoningBankSwarmLearner(swarmId, {
      learningMode: LearningMode.ONLINE
    });
  });

  afterEach(async () => {
    if (learner && learner.isInitialized) {
      await learner.shutdown();
    }
  });

  describe('Initialization', () => {
    test('should initialize successfully', async () => {
      const result = await learner.initialize();

      expect(result).toMatchObject({
        swarmId,
        learningMode: LearningMode.ONLINE,
        componentsReady: true
      });

      expect(learner.isInitialized).toBe(true);
    });

    test('should initialize all components', async () => {
      await learner.initialize();

      expect(learner.trajectoryTracker).toBeDefined();
      expect(learner.verdictJudge).toBeDefined();
      expect(learner.memoryDistiller).toBeDefined();
      expect(learner.patternRecognizer).toBeDefined();
    });
  });

  describe('Trajectory Recording', () => {
    beforeEach(async () => {
      await learner.initialize();
    });

    test('should record trading decision', async () => {
      const decision = {
        id: 'dec-001',
        agentId: 'agent-001',
        type: 'momentum',
        action: 'buy',
        symbol: 'AAPL',
        quantity: 100,
        price: 150.25,
        reasoning: {
          confidence: 0.8,
          factors: ['momentum', 'volume'],
          riskLevel: 'medium'
        },
        marketState: {
          volatility: 25,
          trend: 'up',
          volume: 1000000
        }
      };

      const trajectory = await learner.recordTrajectory(decision);

      expect(trajectory).toMatchObject({
        swarmId,
        decision: expect.objectContaining({
          id: 'dec-001',
          agentId: 'agent-001',
          action: 'buy'
        }),
        status: 'pending'
      });

      expect(learner.learningStats.totalDecisions).toBe(1);
    });

    test('should record trajectory with outcome', async () => {
      const decision = {
        id: 'dec-002',
        agentId: 'agent-001',
        action: 'sell',
        symbol: 'TSLA',
        quantity: 50,
        price: 200
      };

      const outcome = {
        executed: true,
        fillPrice: 202,
        fillQuantity: 50,
        slippage: 0.01,
        executionTime: 150,
        pnl: 100,
        pnlPercent: 1.0,
        riskAdjustedReturn: 0.8
      };

      const trajectory = await learner.recordTrajectory(decision, outcome);

      expect(trajectory).toMatchObject({
        status: 'completed',
        outcome: expect.objectContaining({
          executed: true,
          pnl: 100
        })
      });

      expect(learner.learningStats.totalOutcomes).toBe(1);
    });
  });

  describe('Verdict Judgment', () => {
    beforeEach(async () => {
      await learner.initialize();
    });

    test('should judge profitable trade', async () => {
      const decision = {
        id: 'dec-003',
        agentId: 'agent-001',
        action: 'buy',
        symbol: 'NVDA',
        quantity: 10,
        price: 500
      };

      const outcome = {
        executed: true,
        pnlPercent: 5.0,
        riskAdjustedReturn: 4.0,
        slippage: 0.1
      };

      const trajectory = await learner.recordTrajectory(decision, outcome);
      const verdict = await learner.judgeVerdict(trajectory.id);

      expect(verdict).toMatchObject({
        score: expect.any(Number),
        quality: expect.stringMatching(/excellent|good|neutral|poor|terrible/),
        analysis: expect.objectContaining({
          factors: expect.any(Object)
        })
      });

      expect(verdict.score).toBeGreaterThan(0.7); // Should be good/excellent for 5% profit
    });

    test('should judge losing trade', async () => {
      const decision = {
        id: 'dec-004',
        agentId: 'agent-001',
        action: 'sell',
        symbol: 'AMD',
        quantity: 20,
        price: 100
      };

      const outcome = {
        executed: true,
        pnlPercent: -3.0,
        riskAdjustedReturn: -2.5,
        slippage: 0.5
      };

      const trajectory = await learner.recordTrajectory(decision, outcome);
      const verdict = await learner.judgeVerdict(trajectory.id);

      expect(verdict.score).toBeLessThan(0.5); // Should be poor for -3% loss
    });
  });

  describe('Episode Learning', () => {
    beforeEach(async () => {
      await learner.initialize();
    });

    test('should start and end episode', async () => {
      const episode = learner.startEpisode({ strategy: 'momentum' });

      expect(episode).toMatchObject({
        swarmId,
        status: 'active',
        config: expect.objectContaining({ strategy: 'momentum' })
      });

      // Record some trajectories
      for (let i = 0; i < 5; i++) {
        await learner.recordTrajectory({
          id: `dec-ep-${i}`,
          agentId: 'agent-001',
          action: 'buy',
          symbol: 'SPY',
          quantity: 100,
          price: 400
        }, {
          executed: true,
          pnlPercent: Math.random() * 2,
          riskAdjustedReturn: Math.random() * 1.5
        });
      }

      const endedEpisode = await learner.endEpisode({ profitTotal: 500 });

      expect(endedEpisode).toMatchObject({
        status: 'completed',
        result: expect.objectContaining({ profitTotal: 500 })
      });
    });

    test('should learn from episode in episode mode', async () => {
      learner.learningMode = LearningMode.EPISODE;
      const episode = learner.startEpisode();

      // Record multiple trajectories
      const trajectories = [];
      for (let i = 0; i < 10; i++) {
        const traj = await learner.recordTrajectory({
          id: `dec-learn-${i}`,
          agentId: 'agent-001',
          action: 'buy',
          symbol: 'QQQ',
          quantity: 50,
          price: 300
        }, {
          executed: true,
          pnlPercent: Math.random() * 4 - 1, // -1% to +3%
          riskAdjustedReturn: Math.random() * 3
        });

        await learner.judgeVerdict(traj.id);
        trajectories.push(traj);
      }

      const endedEpisode = await learner.endEpisode();

      expect(learner.learningStats.patternsLearned).toBeGreaterThan(0);
    });
  });

  describe('Pattern Recognition', () => {
    beforeEach(async () => {
      await learner.initialize();
    });

    test('should find similar past decisions', async () => {
      // Record some successful trades
      for (let i = 0; i < 5; i++) {
        const traj = await learner.recordTrajectory({
          id: `dec-sim-${i}`,
          agentId: 'agent-001',
          action: 'buy',
          symbol: 'MSFT',
          quantity: 100,
          price: 350 + i,
          marketState: {
            volatility: 20 + i,
            trend: 'up',
            volume: 1000000
          }
        }, {
          executed: true,
          pnlPercent: 2.0 + i * 0.5,
          riskAdjustedReturn: 1.5 + i * 0.3
        });

        await learner.judgeVerdict(traj.id);
      }

      // Learn patterns
      await learner.learnFromExperience(learner.currentEpisode.id);

      // Query similar
      const similar = await learner.querySimilarDecisions({
        marketState: {
          volatility: 22,
          trend: 'up',
          volume: 1000000
        }
      }, {
        topK: 3,
        minSimilarity: 0.5
      });

      expect(similar).toBeInstanceOf(Array);
      expect(similar.length).toBeGreaterThan(0);
      expect(similar[0]).toHaveProperty('similarity');
      expect(similar[0]).toHaveProperty('recommendation');
    });
  });

  describe('Strategy Adaptation', () => {
    beforeEach(async () => {
      await learner.initialize();
    });

    test('should adapt agent strategy from learnings', async () => {
      const agentId = 'agent-adapt-001';

      // Record successful trades for this agent
      const learnings = [];
      for (let i = 0; i < 3; i++) {
        const traj = await learner.recordTrajectory({
          id: `dec-adapt-${i}`,
          agentId,
          action: 'buy',
          symbol: 'GOOG',
          quantity: 10,
          price: 100
        }, {
          executed: true,
          pnlPercent: 3.0,
          riskAdjustedReturn: 2.5
        });

        await learner.judgeVerdict(traj.id);
        learnings.push(traj);
      }

      const result = await learner.adaptAgentStrategy(agentId, learnings);

      expect(result).toMatchObject({
        agentId,
        adjustmentsApplied: expect.any(Number),
        learningsProcessed: 3
      });

      expect(learner.learningStats.adaptationEvents).toBe(1);
    });
  });

  describe('SwarmCoordinator Integration', () => {
    let coordinator;

    beforeEach(() => {
      coordinator = new SwarmCoordinator({
        swarmId: 'test-swarm-integration',
        topology: TOPOLOGY.MESH,
        maxAgents: 5
      });

      addLearningCapabilities(coordinator, {
        mode: LearningMode.ONLINE
      });
    });

    afterEach(async () => {
      if (coordinator) {
        await coordinator.shutdown();
      }
    });

    test('should add learning methods to coordinator', () => {
      expect(coordinator.reasoningBank).toBeDefined();
      expect(coordinator.initializeLearning).toBeInstanceOf(Function);
      expect(coordinator.recordDecision).toBeInstanceOf(Function);
      expect(coordinator.recordOutcome).toBeInstanceOf(Function);
      expect(coordinator.getRecommendations).toBeInstanceOf(Function);
      expect(coordinator.adaptAgent).toBeInstanceOf(Function);
      expect(coordinator.getLearningStats).toBeInstanceOf(Function);
    });

    test('should initialize learning system', async () => {
      const result = await coordinator.initializeLearning();

      expect(result).toMatchObject({
        enabled: true,
        mode: LearningMode.ONLINE
      });
    });

    test('should record and learn from decisions', async () => {
      await coordinator.initializeLearning();

      const trajectory = await coordinator.recordDecision('agent-001', {
        id: 'int-dec-001',
        action: 'buy',
        symbol: 'INTC',
        quantity: 100,
        price: 50
      });

      expect(trajectory).toBeDefined();
      expect(trajectory.id).toBeDefined();

      const outcome = await coordinator.recordOutcome(trajectory.id, {
        executed: true,
        pnlPercent: 2.0,
        riskAdjustedReturn: 1.5
      });

      expect(outcome).toMatchObject({
        trajectoryId: trajectory.id,
        outcome: expect.objectContaining({ executed: true })
      });
    });
  });

  describe('E2BMonitor Integration', () => {
    let monitor;
    let coordinator;

    beforeEach(() => {
      coordinator = new SwarmCoordinator({
        swarmId: 'test-monitor-integration',
        topology: TOPOLOGY.STAR,
        maxAgents: 3
      });

      addLearningCapabilities(coordinator);

      monitor = new E2BMonitor({
        monitorInterval: 10000
      });

      addLearningMetrics(monitor, coordinator);
    });

    afterEach(async () => {
      if (monitor.isMonitoring) {
        await monitor.stopMonitoring();
      }
      if (coordinator) {
        await coordinator.shutdown();
      }
    });

    test('should add learning metrics to monitor', () => {
      expect(monitor.checkLearningHealth).toBeInstanceOf(Function);
    });

    test('should check learning health', async () => {
      await coordinator.initializeLearning();

      const health = monitor.checkLearningHealth();

      expect(health).toMatchObject({
        status: expect.stringMatching(/disabled|healthy|degraded|critical/),
        issues: expect.any(Array)
      });
    });

    test('should generate health report with learning metrics', async () => {
      await coordinator.initializeLearning();

      // Record some activity
      await coordinator.recordDecision('agent-001', {
        id: 'mon-dec-001',
        action: 'buy',
        symbol: 'META',
        quantity: 10,
        price: 300
      });

      const report = await monitor.generateHealthReport();

      expect(report).toHaveProperty('learning');
      expect(report.learning).toMatchObject({
        enabled: true,
        stats: expect.objectContaining({
          totalDecisions: expect.any(Number)
        })
      });
    });
  });

  describe('Statistics and Metrics', () => {
    beforeEach(async () => {
      learner = new ReasoningBankSwarmLearner(swarmId);
      await learner.initialize();
    });

    test('should track learning statistics', async () => {
      // Record some activity
      for (let i = 0; i < 5; i++) {
        await learner.recordTrajectory({
          id: `stats-dec-${i}`,
          agentId: 'agent-001',
          action: 'buy',
          symbol: 'NFLX',
          quantity: 1,
          price: 400
        }, {
          executed: true,
          pnlPercent: Math.random() * 2,
          riskAdjustedReturn: Math.random() * 1.5
        });
      }

      const stats = learner.getStats();

      expect(stats).toMatchObject({
        totalDecisions: 5,
        totalOutcomes: 5,
        avgVerdictScore: expect.any(Number),
        uptime: expect.any(String),
        learningRate: expect.any(Number)
      });
    });
  });
});
