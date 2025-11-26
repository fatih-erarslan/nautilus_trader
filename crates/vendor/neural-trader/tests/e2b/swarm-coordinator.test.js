/**
 * SwarmCoordinator Test Suite
 *
 * Comprehensive tests for multi-agent coordination
 */

const { SwarmCoordinator, TOPOLOGY, DISTRIBUTION_STRATEGY, AGENT_STATE } = require('../../src/e2b/swarm-coordinator');

describe('SwarmCoordinator', () => {
  let coordinator;

  beforeEach(() => {
    coordinator = new SwarmCoordinator({
      swarmId: 'test-swarm',
      topology: TOPOLOGY.MESH,
      maxAgents: 5,
      e2bApiKey: process.env.E2B_API_KEY || 'test-key',
      quicEnabled: false // Disable for testing
    });
  });

  afterEach(async () => {
    if (coordinator.isInitialized) {
      await coordinator.shutdown();
    }
  });

  describe('Initialization', () => {
    test('should create coordinator with default configuration', () => {
      expect(coordinator.swarmId).toBe('test-swarm');
      expect(coordinator.topology).toBe(TOPOLOGY.MESH);
      expect(coordinator.maxAgents).toBe(5);
      expect(coordinator.isInitialized).toBe(false);
    });

    test('should initialize swarm with mesh topology', async () => {
      const mockConfig = {
        agents: [
          {
            name: 'test_agent_1',
            agent_type: 'momentum_trader',
            symbols: ['SPY'],
            resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
          }
        ]
      };

      // Mock deployer
      coordinator.deployer.createSandbox = jest.fn().mockResolvedValue({
        id: 'sandbox-1',
        status: 'running'
      });

      const result = await coordinator.initializeSwarm(mockConfig);

      expect(result.swarmId).toBe('test-swarm');
      expect(result.topology).toBe(TOPOLOGY.MESH);
      expect(result.status).toBe('initialized');
      expect(coordinator.isInitialized).toBe(true);
    });

    test('should establish hierarchical topology', async () => {
      coordinator.topology = TOPOLOGY.HIERARCHICAL;

      const mockConfig = {
        agents: [
          { name: 'coordinator', agent_type: 'risk_manager', symbols: ['ALL'], resources: {} },
          { name: 'worker1', agent_type: 'momentum_trader', symbols: ['SPY'], resources: {} },
          { name: 'worker2', agent_type: 'neural_forecaster', symbols: ['AAPL'], resources: {} }
        ]
      };

      coordinator.deployer.createSandbox = jest.fn().mockResolvedValue({
        id: 'sandbox',
        status: 'running'
      });

      await coordinator.initializeSwarm(mockConfig);

      const agents = Array.from(coordinator.agents.values());
      expect(agents.length).toBe(3);

      // Coordinator should be connected to all workers
      const coordinatorAgent = agents[0];
      expect(coordinatorAgent.connections.size).toBe(2);
    });
  });

  describe('Task Distribution', () => {
    beforeEach(async () => {
      coordinator.deployer.createSandbox = jest.fn().mockResolvedValue({
        id: 'sandbox',
        status: 'running'
      });

      await coordinator.initializeSwarm({
        agents: [
          { name: 'agent1', agent_type: 'momentum_trader', symbols: ['SPY'], resources: {} },
          { name: 'agent2', agent_type: 'neural_forecaster', symbols: ['AAPL'], resources: {} },
          { name: 'agent3', agent_type: 'risk_manager', symbols: ['ALL'], resources: {} }
        ]
      });

      // Set agents to ready
      for (const agent of coordinator.agents.values()) {
        agent.state = AGENT_STATE.READY;
      }
    });

    test('should distribute task using round-robin strategy', async () => {
      const task = {
        type: 'analyze',
        data: { symbol: 'SPY' }
      };

      const result = await coordinator.distributeTask(task, DISTRIBUTION_STRATEGY.ROUND_ROBIN);

      expect(result.assignedAgents).toHaveLength(1);
      expect(result.strategy).toBe(DISTRIBUTION_STRATEGY.ROUND_ROBIN);
      expect(coordinator.metrics.tasksDistributed).toBe(1);
    });

    test('should distribute task to least loaded agent', async () => {
      const agents = Array.from(coordinator.agents.values());
      agents[0].performance.load = 0.8;
      agents[1].performance.load = 0.2;
      agents[2].performance.load = 0.5;

      const task = {
        type: 'analyze',
        data: { symbol: 'AAPL' }
      };

      const result = await coordinator.distributeTask(task, DISTRIBUTION_STRATEGY.LEAST_LOADED);

      expect(result.assignedAgents[0].load).toBe(0.2);
    });

    test('should distribute task based on agent capabilities', async () => {
      const task = {
        type: 'forecast',
        requiredCapability: 'forecast',
        data: { symbol: 'TSLA' }
      };

      const result = await coordinator.distributeTask(task, DISTRIBUTION_STRATEGY.SPECIALIZED);

      const assignedAgent = coordinator.agents.get(result.assignedAgents[0].id);
      expect(assignedAgent.capabilities).toContain('forecast');
    });

    test('should distribute task for consensus decision', async () => {
      const task = {
        type: 'trade_decision',
        requireConsensus: true,
        data: { symbol: 'SPY', action: 'buy' }
      };

      const result = await coordinator.distributeTask(task, DISTRIBUTION_STRATEGY.CONSENSUS);

      expect(result.assignedAgents.length).toBeGreaterThanOrEqual(2);
    });

    test('should use adaptive strategy with scoring', async () => {
      const agents = Array.from(coordinator.agents.values());
      agents[0].performance.errorRate = 0.5;
      agents[1].performance.errorRate = 0.1;
      agents[2].performance.avgLatency = 1000;

      const task = {
        type: 'optimize',
        data: { portfolio: ['SPY', 'QQQ'] }
      };

      const result = await coordinator.distributeTask(task, DISTRIBUTION_STRATEGY.ADAPTIVE);

      expect(result.assignedAgents).toHaveLength(1);
      // Should select agent with lowest error rate
      const assignedAgent = coordinator.agents.get(result.assignedAgents[0].id);
      expect(assignedAgent.performance.errorRate).toBeLessThan(0.3);
    });
  });

  describe('Result Collection', () => {
    beforeEach(async () => {
      coordinator.deployer.createSandbox = jest.fn().mockResolvedValue({
        id: 'sandbox',
        status: 'running'
      });

      await coordinator.initializeSwarm({
        agents: [
          { name: 'agent1', agent_type: 'momentum_trader', symbols: ['SPY'], resources: {} },
          { name: 'agent2', agent_type: 'neural_forecaster', symbols: ['AAPL'], resources: {} }
        ]
      });

      for (const agent of coordinator.agents.values()) {
        agent.state = AGENT_STATE.READY;
      }
    });

    test('should collect results from agents', async () => {
      const task = { type: 'analyze', data: { symbol: 'SPY' } };
      const distResult = await coordinator.distributeTask(task);

      // Simulate agent completion
      for (const agentInfo of distResult.assignedAgents) {
        const key = `task:${distResult.taskId}:agent:${agentInfo.id}`;
        coordinator.sharedMemory.set(key, {
          completed: true,
          result: { prediction: 'bullish', confidence: 0.8 },
          timestamp: Date.now()
        });
      }

      const results = await coordinator.collectResults(distResult.taskId);

      expect(results.taskId).toBe(distResult.taskId);
      expect(results.agentCount).toBe(distResult.assignedAgents.length);
      expect(coordinator.metrics.tasksCompleted).toBe(1);
    });

    test('should check consensus with sufficient agreement', async () => {
      const agentResults = [
        { decision: 'buy', confidence: 0.9 },
        { decision: 'buy', confidence: 0.8 },
        { decision: 'hold', confidence: 0.6 }
      ];

      const consensus = await coordinator.checkConsensus(agentResults);

      expect(consensus.decision).toBe('buy');
      expect(consensus.agreement).toBeGreaterThan(0.5);
      expect(consensus.achieved).toBe(true);
    });
  });

  describe('State Synchronization', () => {
    beforeEach(async () => {
      coordinator.deployer.createSandbox = jest.fn().mockResolvedValue({
        id: 'sandbox',
        status: 'running'
      });

      await coordinator.initializeSwarm({
        agents: [
          { name: 'agent1', agent_type: 'momentum_trader', symbols: ['SPY'], resources: {} }
        ]
      });
    });

    test('should synchronize swarm state', async () => {
      const snapshot = await coordinator.synchronizeState();

      expect(snapshot.swarmId).toBe('test-swarm');
      expect(snapshot.agents).toHaveLength(1);
      expect(snapshot.metrics).toBeDefined();
      expect(coordinator.sharedMemory.has('swarm:state')).toBe(true);
    });

    test('should start periodic synchronization', () => {
      expect(coordinator.syncIntervalId).not.toBeNull();
    });
  });

  describe('Health Monitoring', () => {
    beforeEach(async () => {
      coordinator.deployer.createSandbox = jest.fn().mockResolvedValue({
        id: 'sandbox',
        status: 'running'
      });

      await coordinator.initializeSwarm({
        agents: [
          { name: 'agent1', agent_type: 'momentum_trader', symbols: ['SPY'], resources: {} }
        ]
      });
    });

    test('should detect unresponsive agents', async () => {
      const agent = Array.from(coordinator.agents.values())[0];
      agent.lastHeartbeat = Date.now() - 60000; // 60 seconds ago
      agent.state = AGENT_STATE.READY;

      await coordinator.performHealthCheck();

      expect(agent.state).toBe(AGENT_STATE.OFFLINE);
    });

    test('should start health monitoring', () => {
      expect(coordinator.healthCheckIntervalId).not.toBeNull();
    });
  });

  describe('Rebalancing', () => {
    beforeEach(async () => {
      coordinator.deployer.createSandbox = jest.fn().mockResolvedValue({
        id: 'sandbox',
        status: 'running'
      });

      await coordinator.initializeSwarm({
        agents: [
          { name: 'agent1', agent_type: 'momentum_trader', symbols: ['SPY'], resources: {} },
          { name: 'agent2', agent_type: 'neural_forecaster', symbols: ['AAPL'], resources: {} },
          { name: 'agent3', agent_type: 'risk_manager', symbols: ['ALL'], resources: {} }
        ]
      });

      for (const agent of coordinator.agents.values()) {
        agent.state = AGENT_STATE.READY;
      }
    });

    test('should rebalance when load is imbalanced', async () => {
      const agents = Array.from(coordinator.agents.values());
      agents[0].performance.load = 0.9;
      agents[1].performance.load = 0.1;
      agents[2].performance.load = 0.2;

      const result = await coordinator.rebalance();

      expect(result.rebalanced).toBe(true);
      expect(result.adjustedAgents).toBeGreaterThan(0);
      expect(coordinator.metrics.rebalanceEvents).toBe(1);
    });

    test('should not rebalance when load is balanced', async () => {
      const agents = Array.from(coordinator.agents.values());
      agents[0].performance.load = 0.5;
      agents[1].performance.load = 0.5;
      agents[2].performance.load = 0.5;

      const result = await coordinator.rebalance();

      expect(result.rebalanced).toBe(false);
      expect(result.reason).toBe('balanced');
    });
  });

  describe('Status and Metrics', () => {
    beforeEach(async () => {
      coordinator.deployer.createSandbox = jest.fn().mockResolvedValue({
        id: 'sandbox',
        status: 'running'
      });

      await coordinator.initializeSwarm({
        agents: [
          { name: 'agent1', agent_type: 'momentum_trader', symbols: ['SPY'], resources: {} }
        ]
      });
    });

    test('should return comprehensive status', () => {
      const status = coordinator.getStatus();

      expect(status.swarmId).toBe('test-swarm');
      expect(status.topology).toBe(TOPOLOGY.MESH);
      expect(status.isInitialized).toBe(true);
      expect(status.agents.total).toBe(1);
      expect(status.tasks).toBeDefined();
      expect(status.performance).toBeDefined();
      expect(status.coordination).toBeDefined();
    });

    test('should track metrics correctly', async () => {
      for (const agent of coordinator.agents.values()) {
        agent.state = AGENT_STATE.READY;
      }

      await coordinator.distributeTask({ type: 'test' });

      expect(coordinator.metrics.tasksDistributed).toBe(1);
    });
  });

  describe('Shutdown', () => {
    test('should shutdown cleanly', async () => {
      coordinator.deployer.createSandbox = jest.fn().mockResolvedValue({
        id: 'sandbox',
        status: 'running'
      });

      await coordinator.initializeSwarm({
        agents: [
          { name: 'agent1', agent_type: 'momentum_trader', symbols: ['SPY'], resources: {} }
        ]
      });

      await coordinator.shutdown();

      expect(coordinator.isInitialized).toBe(false);
      expect(coordinator.syncIntervalId).toBeNull();
      expect(coordinator.healthCheckIntervalId).toBeNull();

      for (const agent of coordinator.agents.values()) {
        expect(agent.state).toBe(AGENT_STATE.OFFLINE);
      }
    });
  });
});
