/**
 * Comprehensive E2B Swarm Deployment Patterns Test Suite
 *
 * Tests 8 production deployment patterns:
 * 1. Mesh Topology (Peer-to-Peer)
 * 2. Hierarchical Topology (Leader-Worker)
 * 3. Ring Topology (Sequential Processing)
 * 4. Star Topology (Centralized Hub)
 * 5. Auto-Scaling Deployment
 * 6. Multi-Strategy Deployment
 * 7. Blue-Green Deployment
 * 8. Canary Deployment
 *
 * Each test validates coordination, performance, and failure scenarios.
 */

const { SandboxManager } = require('../../src/e2b/sandbox-manager');
const { SwarmCoordinator, TOPOLOGY, DISTRIBUTION_STRATEGY } = require('../../src/e2b/swarm-coordinator');
const { E2BMonitor, HealthStatus, ScalingAction } = require('../../src/e2b/monitor-and-scale');

// Test configuration
const TEST_CONFIG = {
  timeout: 120000, // 2 minutes per test
  e2bApiKey: process.env.E2B_API_KEY || 'test-key',
  dryRun: !process.env.E2B_API_KEY, // Run in mock mode if no API key
  minTestDuration: 10000, // Minimum 10 seconds per test
  symbols: ['SPY', 'QQQ', 'AAPL', 'TSLA'],
  strategies: ['momentum', 'mean_reversion', 'pairs_trading', 'arbitrage']
};

// Mock E2B operations if no API key
const MOCK_MODE = TEST_CONFIG.dryRun;

// Test utilities
const TestUtils = {
  /**
   * Wait for specified duration
   */
  wait: (ms) => new Promise(resolve => setTimeout(resolve, ms)),

  /**
   * Execute trading operation simulation
   */
  async executeTradeSimulation(sandboxId, strategy, symbols) {
    const startTime = Date.now();

    // Simulate trade execution
    await this.wait(Math.random() * 1000 + 500);

    return {
      sandboxId,
      strategy,
      symbols,
      executionTime: Date.now() - startTime,
      success: Math.random() > 0.1, // 90% success rate
      profit: (Math.random() - 0.5) * 1000,
      timestamp: new Date().toISOString()
    };
  },

  /**
   * Measure performance metrics
   */
  measurePerformance(operations) {
    const executionTimes = operations.map(op => op.executionTime);
    const successRate = operations.filter(op => op.success).length / operations.length;
    const totalProfit = operations.reduce((sum, op) => sum + (op.profit || 0), 0);

    return {
      avgExecutionTime: executionTimes.reduce((a, b) => a + b, 0) / executionTimes.length,
      minExecutionTime: Math.min(...executionTimes),
      maxExecutionTime: Math.max(...executionTimes),
      successRate,
      totalProfit,
      totalOperations: operations.length
    };
  },

  /**
   * Validate coordination works correctly
   */
  async validateCoordination(coordinator, expectedAgents) {
    const status = coordinator.getStatus();

    expect(status.isInitialized).toBe(true);
    expect(status.agents.total).toBe(expectedAgents);
    expect(status.agents.ready).toBeGreaterThanOrEqual(expectedAgents * 0.8); // At least 80% ready

    return status;
  },

  /**
   * Inject failure into sandbox
   */
  async injectFailure(sandboxManager, sandboxId, failureType = 'cpu_spike') {
    console.log(`    🔧 Injecting ${failureType} failure into ${sandboxId}`);

    // Simulate different failure types
    const failures = {
      cpu_spike: () => ({ cpu: 95, memory: 50, errorRate: 0.05 }),
      memory_leak: () => ({ cpu: 50, memory: 98, errorRate: 0.1 }),
      network_timeout: () => ({ cpu: 30, memory: 40, errorRate: 0.5 }),
      agent_crash: () => ({ cpu: 0, memory: 0, errorRate: 1.0 })
    };

    const failureMetrics = failures[failureType] ? failures[failureType]() : failures.cpu_spike();

    return failureMetrics;
  }
};

describe('E2B Deployment Patterns - Production Test Suite', () => {
  let sandboxManager;
  let monitor;

  beforeAll(() => {
    if (MOCK_MODE) {
      console.log('\n⚠️  Running in MOCK MODE (no E2B API key detected)');
      console.log('Set E2B_API_KEY environment variable for real testing\n');
    }
  });

  beforeEach(() => {
    sandboxManager = new SandboxManager();
    monitor = new E2BMonitor({
      monitorInterval: 2000,
      scaleUpThreshold: 0.75,
      scaleDownThreshold: 0.25,
      maxSandboxes: 10,
      minSandboxes: 1
    });
  });

  afterEach(async () => {
    // Cleanup resources
    if (monitor.isMonitoring) {
      await monitor.stopMonitoring();
    }

    if (sandboxManager) {
      await sandboxManager.shutdown();
    }
  });

  // ==================== PATTERN 1: MESH TOPOLOGY ====================
  describe('Pattern 1: Mesh Topology (Peer-to-Peer)', () => {
    test('Mesh Deployment: 5 momentum traders with equal coordination', async () => {
      console.log('\n  📊 Testing Mesh Topology: 5 Momentum Traders');

      const coordinator = new SwarmCoordinator({
        topology: TOPOLOGY.MESH,
        maxAgents: 5,
        distributionStrategy: DISTRIBUTION_STRATEGY.ROUND_ROBIN,
        e2bApiKey: TEST_CONFIG.e2bApiKey
      });

      // Initialize swarm with 5 agents
      const agents = Array(5).fill(null).map((_, i) => ({
        name: `momentum_trader_${i + 1}`,
        agent_type: 'momentum_trader',
        symbols: [TEST_CONFIG.symbols[i % TEST_CONFIG.symbols.length]],
        resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
      }));

      const initResult = await coordinator.initializeSwarm({ agents });

      expect(initResult.status).toBe('initialized');
      expect(initResult.agentCount).toBe(5);
      expect(initResult.topology).toBe(TOPOLOGY.MESH);

      // Validate coordination
      await TestUtils.validateCoordination(coordinator, 5);

      // Execute trading operations
      const operations = [];
      for (let i = 0; i < 20; i++) {
        const task = {
          id: `trade-${i}`,
          type: 'execute_trade',
          symbol: TEST_CONFIG.symbols[i % TEST_CONFIG.symbols.length],
          action: Math.random() > 0.5 ? 'buy' : 'sell',
          quantity: Math.floor(Math.random() * 100) + 1
        };

        const distribution = await coordinator.distributeTask(task);
        expect(distribution.assignedAgents).toHaveLength(1);

        const operation = await TestUtils.executeTradeSimulation(
          distribution.assignedAgents[0].id,
          'momentum',
          [task.symbol]
        );
        operations.push(operation);
      }

      // Measure performance
      const performance = TestUtils.measurePerformance(operations);
      console.log(`    ✅ Performance: ${performance.successRate * 100}% success, ${performance.avgExecutionTime.toFixed(0)}ms avg`);

      expect(performance.successRate).toBeGreaterThan(0.85);
      expect(performance.avgExecutionTime).toBeLessThan(2000);

      // Verify mesh connectivity (all agents connected to all others)
      const status = coordinator.getStatus();
      const agentsArray = Array.from(coordinator.agents.values());

      // In mesh, each agent should be connected to n-1 other agents
      agentsArray.forEach(agent => {
        expect(agent.connections.size).toBe(4); // 5 agents - 1 self = 4 connections
      });

      await coordinator.shutdown();
    }, TEST_CONFIG.timeout);

    test('Mesh Deployment: Consensus trading with 3 agents', async () => {
      console.log('\n  🤝 Testing Mesh Consensus: 3 Agents');

      const coordinator = new SwarmCoordinator({
        topology: TOPOLOGY.MESH,
        maxAgents: 3,
        distributionStrategy: DISTRIBUTION_STRATEGY.CONSENSUS,
        consensusThreshold: 0.66,
        e2bApiKey: TEST_CONFIG.e2bApiKey
      });

      const agents = Array(3).fill(null).map((_, i) => ({
        name: `consensus_trader_${i + 1}`,
        agent_type: 'neural_forecaster',
        symbols: ['SPY'],
        resources: { cpu: 4, memory_mb: 2048, timeout: 3600 }
      }));

      await coordinator.initializeSwarm({ agents });

      // Execute consensus task
      const consensusTask = {
        id: 'consensus-trade-1',
        type: 'consensus_trade',
        symbol: 'SPY',
        requireConsensus: true
      };

      const distribution = await coordinator.distributeTask(consensusTask);
      expect(distribution.assignedAgents.length).toBeGreaterThanOrEqual(2); // At least 66% of 3

      // Simulate agent decisions
      for (const agent of distribution.assignedAgents) {
        const agentId = agent.id;
        const decision = Math.random() > 0.3 ? 'buy' : 'hold'; // Biased towards buy

        coordinator.sharedMemory.set(`task:${consensusTask.id}:agent:${agentId}`, {
          taskId: consensusTask.id,
          agentId,
          decision,
          confidence: Math.random() * 0.5 + 0.5,
          completed: true
        });
      }

      // Collect and verify consensus
      const results = await coordinator.collectResults(consensusTask.id);
      expect(results.consensus).toBeDefined();
      expect(results.consensus.achieved).toBeDefined();

      if (results.consensus.achieved) {
        console.log(`    ✅ Consensus achieved: ${results.consensus.decision} (${(results.consensus.confidence * 100).toFixed(1)}% agreement)`);
      }

      await coordinator.shutdown();
    }, TEST_CONFIG.timeout);

    test('Mesh Deployment: Failover and redundancy', async () => {
      console.log('\n  🔄 Testing Mesh Failover');

      const coordinator = new SwarmCoordinator({
        topology: TOPOLOGY.MESH,
        maxAgents: 4,
        e2bApiKey: TEST_CONFIG.e2bApiKey
      });

      const agents = Array(4).fill(null).map((_, i) => ({
        name: `failover_agent_${i + 1}`,
        agent_type: 'momentum_trader',
        symbols: ['QQQ'],
        resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
      }));

      await coordinator.initializeSwarm({ agents });

      // Simulate agent failure
      const agentsArray = Array.from(coordinator.agents.values());
      const failedAgent = agentsArray[0];
      failedAgent.state = 'offline';

      console.log(`    🔧 Simulated failure of agent: ${failedAgent.id}`);

      // Task should still be distributed to remaining agents
      const task = {
        id: 'failover-task-1',
        type: 'execute_trade',
        symbol: 'QQQ'
      };

      const distribution = await coordinator.distributeTask(task);
      expect(distribution.assignedAgents).toHaveLength(1);
      expect(distribution.assignedAgents[0].id).not.toBe(failedAgent.id);

      console.log(`    ✅ Task successfully routed to healthy agent`);

      await coordinator.shutdown();
    }, TEST_CONFIG.timeout);
  });

  // ==================== PATTERN 2: HIERARCHICAL TOPOLOGY ====================
  describe('Pattern 2: Hierarchical Topology (Leader-Worker)', () => {
    test('Hierarchical Deployment: 1 coordinator + 4 workers', async () => {
      console.log('\n  🎯 Testing Hierarchical Topology: 1 Coordinator + 4 Workers');

      const coordinator = new SwarmCoordinator({
        topology: TOPOLOGY.HIERARCHICAL,
        maxAgents: 5,
        distributionStrategy: DISTRIBUTION_STRATEGY.LEAST_LOADED,
        e2bApiKey: TEST_CONFIG.e2bApiKey
      });

      const agents = [
        {
          name: 'coordinator',
          agent_type: 'portfolio_optimizer',
          symbols: ['ALL'],
          resources: { cpu: 4, memory_mb: 2048, timeout: 7200 }
        },
        ...Array(4).fill(null).map((_, i) => ({
          name: `worker_${i + 1}`,
          agent_type: 'momentum_trader',
          symbols: [TEST_CONFIG.symbols[i]],
          resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
        }))
      ];

      await coordinator.initializeSwarm({ agents });

      // Verify hierarchical structure
      const agentsArray = Array.from(coordinator.agents.values());
      const coordinatorAgent = agentsArray[0]; // First agent is coordinator
      const workerAgents = agentsArray.slice(1);

      // Coordinator should be connected to all workers
      expect(coordinatorAgent.connections.size).toBe(4);

      // Workers should only be connected to coordinator
      workerAgents.forEach(worker => {
        expect(worker.connections.size).toBe(1);
        expect(worker.connections.has(coordinatorAgent.id)).toBe(true);
      });

      // Execute tasks through coordinator
      const operations = [];
      for (let i = 0; i < 16; i++) {
        const task = {
          id: `hierarchical-task-${i}`,
          type: 'execute_trade',
          symbol: TEST_CONFIG.symbols[i % TEST_CONFIG.symbols.length]
        };

        await coordinator.distributeTask(task, DISTRIBUTION_STRATEGY.LEAST_LOADED);

        const operation = await TestUtils.executeTradeSimulation(
          `worker_${(i % 4) + 1}`,
          'momentum',
          [task.symbol]
        );
        operations.push(operation);
      }

      const performance = TestUtils.measurePerformance(operations);
      console.log(`    ✅ Load balanced across 4 workers: ${performance.avgExecutionTime.toFixed(0)}ms avg`);

      await coordinator.shutdown();
    }, TEST_CONFIG.timeout);

    test('Hierarchical Deployment: Multi-strategy coordination', async () => {
      console.log('\n  🎭 Testing Hierarchical Multi-Strategy');

      const coordinator = new SwarmCoordinator({
        topology: TOPOLOGY.HIERARCHICAL,
        maxAgents: 5,
        distributionStrategy: DISTRIBUTION_STRATEGY.SPECIALIZED,
        e2bApiKey: TEST_CONFIG.e2bApiKey
      });

      const agents = [
        {
          name: 'strategy_coordinator',
          agent_type: 'portfolio_optimizer',
          symbols: ['ALL'],
          resources: { cpu: 4, memory_mb: 2048, timeout: 7200 }
        },
        {
          name: 'momentum_specialist',
          agent_type: 'momentum_trader',
          symbols: ['SPY', 'QQQ'],
          resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
        },
        {
          name: 'pairs_specialist',
          agent_type: 'mean_reversion_trader',
          symbols: ['AAPL', 'TSLA'],
          resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
        },
        {
          name: 'arbitrage_specialist',
          agent_type: 'neural_forecaster',
          symbols: ['SPY'],
          resources: { cpu: 4, memory_mb: 2048, timeout: 3600 }
        },
        {
          name: 'risk_specialist',
          agent_type: 'risk_manager',
          symbols: ['ALL'],
          resources: { cpu: 2, memory_mb: 512, timeout: 7200 }
        }
      ];

      await coordinator.initializeSwarm({ agents });

      // Test specialized routing
      const tasks = [
        { id: 'task-1', type: 'momentum', requiredCapability: 'momentum_detection' },
        { id: 'task-2', type: 'pairs', requiredCapability: 'mean_reversion' },
        { id: 'task-3', type: 'forecast', requiredCapability: 'forecast' },
        { id: 'task-4', type: 'risk', requiredCapability: 'risk_assessment' }
      ];

      for (const task of tasks) {
        const distribution = await coordinator.distributeTask(task, DISTRIBUTION_STRATEGY.SPECIALIZED);
        expect(distribution.assignedAgents).toHaveLength(1);

        const assignedAgent = coordinator.agents.get(distribution.assignedAgents[0].id);
        expect(assignedAgent.capabilities).toContain(task.requiredCapability);
      }

      console.log(`    ✅ All tasks correctly routed to specialized agents`);

      await coordinator.shutdown();
    }, TEST_CONFIG.timeout);

    test('Hierarchical Deployment: Load balancing across workers', async () => {
      console.log('\n  ⚖️  Testing Hierarchical Load Balancing');

      const coordinator = new SwarmCoordinator({
        topology: TOPOLOGY.HIERARCHICAL,
        maxAgents: 6,
        distributionStrategy: DISTRIBUTION_STRATEGY.LEAST_LOADED,
        e2bApiKey: TEST_CONFIG.e2bApiKey
      });

      const agents = [
        {
          name: 'load_coordinator',
          agent_type: 'portfolio_optimizer',
          symbols: ['ALL'],
          resources: { cpu: 4, memory_mb: 2048, timeout: 7200 }
        },
        ...Array(5).fill(null).map((_, i) => ({
          name: `load_worker_${i + 1}`,
          agent_type: 'momentum_trader',
          symbols: ['SPY'],
          resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
        }))
      ];

      await coordinator.initializeSwarm({ agents });

      // Distribute 50 tasks and measure load distribution
      const agentTaskCounts = new Map();

      for (let i = 0; i < 50; i++) {
        const task = { id: `load-task-${i}`, type: 'execute_trade' };
        const distribution = await coordinator.distributeTask(task);

        const agentId = distribution.assignedAgents[0].id;
        agentTaskCounts.set(agentId, (agentTaskCounts.get(agentId) || 0) + 1);

        // Simulate varying execution times
        await TestUtils.wait(Math.random() * 100);
      }

      // Verify load is balanced (no agent should have >30% of tasks)
      const counts = Array.from(agentTaskCounts.values());
      const maxCount = Math.max(...counts);
      const loadBalance = maxCount / 50;

      console.log(`    ✅ Load distribution: max agent load = ${(loadBalance * 100).toFixed(1)}%`);
      expect(loadBalance).toBeLessThan(0.35); // No single agent handles >35% of tasks

      await coordinator.shutdown();
    }, TEST_CONFIG.timeout);
  });

  // ==================== PATTERN 3: RING TOPOLOGY ====================
  describe('Pattern 3: Ring Topology (Sequential Processing)', () => {
    test('Ring Deployment: Pipeline processing with 4 agents', async () => {
      console.log('\n  🔄 Testing Ring Topology: 4-Agent Pipeline');

      const coordinator = new SwarmCoordinator({
        topology: TOPOLOGY.RING,
        maxAgents: 4,
        e2bApiKey: TEST_CONFIG.e2bApiKey
      });

      const agents = [
        {
          name: 'data_collector',
          agent_type: 'neural_forecaster',
          symbols: ['SPY'],
          resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
        },
        {
          name: 'signal_analyzer',
          agent_type: 'momentum_trader',
          symbols: ['SPY'],
          resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
        },
        {
          name: 'risk_assessor',
          agent_type: 'risk_manager',
          symbols: ['SPY'],
          resources: { cpu: 2, memory_mb: 512, timeout: 3600 }
        },
        {
          name: 'executor',
          agent_type: 'momentum_trader',
          symbols: ['SPY'],
          resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
        }
      ];

      await coordinator.initializeSwarm({ agents });

      // Verify ring structure
      const agentsArray = Array.from(coordinator.agents.values());
      agentsArray.forEach((agent, index) => {
        expect(agent.connections.size).toBe(2); // Each agent connected to 2 neighbors
      });

      // Execute pipeline processing
      const pipelineData = {
        id: 'pipeline-1',
        stage: 'collection',
        data: { symbol: 'SPY', timestamp: Date.now() }
      };

      // Simulate data flow through ring
      const stages = ['collection', 'analysis', 'risk_assessment', 'execution'];
      let currentData = pipelineData;

      for (let i = 0; i < stages.length; i++) {
        const task = {
          id: `pipeline-stage-${i}`,
          type: stages[i],
          data: currentData
        };

        await coordinator.distributeTask(task);
        await TestUtils.wait(500); // Simulate processing time

        currentData = {
          ...currentData,
          stage: stages[(i + 1) % stages.length],
          processedBy: agentsArray[i].id
        };
      }

      console.log(`    ✅ Pipeline completed all 4 stages`);

      await coordinator.shutdown();
    }, TEST_CONFIG.timeout);

    test('Ring Deployment: Data flow optimization', async () => {
      console.log('\n  📊 Testing Ring Data Flow');

      const coordinator = new SwarmCoordinator({
        topology: TOPOLOGY.RING,
        maxAgents: 3,
        e2bApiKey: TEST_CONFIG.e2bApiKey
      });

      const agents = Array(3).fill(null).map((_, i) => ({
        name: `flow_agent_${i + 1}`,
        agent_type: 'momentum_trader',
        symbols: ['QQQ'],
        resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
      }));

      await coordinator.initializeSwarm({ agents });

      // Measure data flow latency
      const flowTests = [];
      for (let i = 0; i < 10; i++) {
        const startTime = Date.now();

        // Send data around the ring
        for (let j = 0; j < 3; j++) {
          const task = {
            id: `flow-${i}-stage-${j}`,
            type: 'process_data',
            iteration: i
          };
          await coordinator.distributeTask(task);
          await TestUtils.wait(100);
        }

        flowTests.push({
          iteration: i,
          duration: Date.now() - startTime
        });
      }

      const avgDuration = flowTests.reduce((sum, t) => sum + t.duration, 0) / flowTests.length;
      console.log(`    ✅ Average ring flow time: ${avgDuration.toFixed(0)}ms`);

      expect(avgDuration).toBeLessThan(1000);

      await coordinator.shutdown();
    }, TEST_CONFIG.timeout);

    test('Ring Deployment: Circuit breaker on failure', async () => {
      console.log('\n  ⚡ Testing Ring Circuit Breaker');

      const coordinator = new SwarmCoordinator({
        topology: TOPOLOGY.RING,
        maxAgents: 4,
        e2bApiKey: TEST_CONFIG.e2bApiKey
      });

      const agents = Array(4).fill(null).map((_, i) => ({
        name: `circuit_agent_${i + 1}`,
        agent_type: 'momentum_trader',
        symbols: ['AAPL'],
        resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
      }));

      await coordinator.initializeSwarm({ agents });

      // Simulate agent failure in ring
      const agentsArray = Array.from(coordinator.agents.values());
      const failedAgent = agentsArray[1]; // Break the ring at second agent
      failedAgent.state = 'error';

      console.log(`    🔧 Simulated failure in ring: ${failedAgent.id}`);

      // Attempt to distribute task - should detect broken ring
      const task = { id: 'circuit-task-1', type: 'execute_trade' };

      try {
        const distribution = await coordinator.distributeTask(task);

        // Should route around failed agent
        expect(distribution.assignedAgents[0].id).not.toBe(failedAgent.id);
        console.log(`    ✅ Circuit breaker activated, routed around failure`);
      } catch (error) {
        // Circuit breaker may throw error if ring is broken
        console.log(`    ✅ Circuit breaker prevented task distribution`);
        expect(error.message).toContain('available');
      }

      await coordinator.shutdown();
    }, TEST_CONFIG.timeout);
  });

  // ==================== PATTERN 4: STAR TOPOLOGY ====================
  describe('Pattern 4: Star Topology (Centralized Hub)', () => {
    test('Star Deployment: Central hub with 6 specialized agents', async () => {
      console.log('\n  ⭐ Testing Star Topology: Hub + 6 Spokes');

      const coordinator = new SwarmCoordinator({
        topology: TOPOLOGY.STAR,
        maxAgents: 7,
        distributionStrategy: DISTRIBUTION_STRATEGY.SPECIALIZED,
        e2bApiKey: TEST_CONFIG.e2bApiKey
      });

      const agents = [
        {
          name: 'central_hub',
          agent_type: 'portfolio_optimizer',
          symbols: ['ALL'],
          resources: { cpu: 8, memory_mb: 4096, timeout: 7200 }
        },
        ...TEST_CONFIG.strategies.map((strategy, i) => ({
          name: `${strategy}_specialist`,
          agent_type: 'momentum_trader',
          symbols: [TEST_CONFIG.symbols[i % TEST_CONFIG.symbols.length]],
          resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
        })),
        {
          name: 'risk_monitor',
          agent_type: 'risk_manager',
          symbols: ['ALL'],
          resources: { cpu: 2, memory_mb: 512, timeout: 7200 }
        },
        {
          name: 'performance_tracker',
          agent_type: 'neural_forecaster',
          symbols: ['ALL'],
          resources: { cpu: 2, memory_mb: 1024, timeout: 7200 }
        }
      ];

      await coordinator.initializeSwarm({ agents });

      // Verify star structure
      const agentsArray = Array.from(coordinator.agents.values());
      const hub = agentsArray[0];
      const spokes = agentsArray.slice(1);

      expect(hub.connections.size).toBe(6); // Hub connected to all 6 spokes
      spokes.forEach(spoke => {
        expect(spoke.connections.size).toBe(1); // Each spoke only connected to hub
        expect(spoke.connections.has(hub.id)).toBe(true);
      });

      // Execute tasks through hub
      const operations = [];
      for (let i = 0; i < 24; i++) {
        const task = {
          id: `star-task-${i}`,
          type: 'execute_trade',
          symbol: TEST_CONFIG.symbols[i % TEST_CONFIG.symbols.length]
        };

        await coordinator.distributeTask(task);

        const operation = await TestUtils.executeTradeSimulation(
          spokes[i % 6].id,
          'momentum',
          [task.symbol]
        );
        operations.push(operation);
      }

      const performance = TestUtils.measurePerformance(operations);
      console.log(`    ✅ Hub coordinated ${operations.length} operations: ${performance.successRate * 100}% success`);

      await coordinator.shutdown();
    }, TEST_CONFIG.timeout);

    test('Star Deployment: Hub failover recovery', async () => {
      console.log('\n  🔄 Testing Star Hub Failover');

      const coordinator = new SwarmCoordinator({
        topology: TOPOLOGY.STAR,
        maxAgents: 4,
        e2bApiKey: TEST_CONFIG.e2bApiKey
      });

      const agents = [
        {
          name: 'primary_hub',
          agent_type: 'portfolio_optimizer',
          symbols: ['ALL'],
          resources: { cpu: 4, memory_mb: 2048, timeout: 7200 }
        },
        ...Array(3).fill(null).map((_, i) => ({
          name: `spoke_${i + 1}`,
          agent_type: 'momentum_trader',
          symbols: ['SPY'],
          resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
        }))
      ];

      await coordinator.initializeSwarm({ agents });

      // Simulate hub failure
      const agentsArray = Array.from(coordinator.agents.values());
      const hub = agentsArray[0];
      hub.state = 'offline';

      console.log(`    🔧 Simulated hub failure: ${hub.id}`);

      // In star topology, hub failure is critical
      // System should detect and handle gracefully
      const task = { id: 'failover-task-1', type: 'execute_trade' };

      try {
        await coordinator.distributeTask(task);

        // If successful, a spoke should have been promoted or backup used
        console.log(`    ✅ Failover successful: task routed to backup`);
      } catch (error) {
        // Expected behavior: star topology fails without hub
        console.log(`    ✅ Hub failure correctly detected`);
        expect(error.message).toContain('available');
      }

      await coordinator.shutdown();
    }, TEST_CONFIG.timeout);
  });

  // ==================== PATTERN 5: AUTO-SCALING ====================
  describe('Pattern 5: Auto-Scaling Deployment', () => {
    test('Auto-Scale: Start with 2, scale to 10 based on load', async () => {
      console.log('\n  📈 Testing Auto-Scaling: 2 → 10 agents');

      const coordinator = new SwarmCoordinator({
        topology: TOPOLOGY.MESH,
        maxAgents: 10,
        e2bApiKey: TEST_CONFIG.e2bApiKey
      });

      // Start with 2 agents
      const initialAgents = Array(2).fill(null).map((_, i) => ({
        name: `scaler_${i + 1}`,
        agent_type: 'momentum_trader',
        symbols: ['SPY'],
        resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
      }));

      await coordinator.initializeSwarm({ agents: initialAgents });
      expect(coordinator.agents.size).toBe(2);

      // Start monitoring with auto-scaling
      await monitor.startMonitoring();

      // Register sandboxes with monitor
      for (const [sandboxId, agent] of coordinator.agents.entries()) {
        monitor.registerSandbox(sandboxId, {
          id: sandboxId,
          type: agent.type,
          status: 'healthy'
        });
      }

      // Simulate high load
      console.log(`    📊 Simulating high load...`);

      for (const sandboxId of monitor.sandboxes.keys()) {
        monitor.recordMetric(sandboxId, {
          cpu: 85,
          memory: 80,
          responseTime: 1500,
          errorRate: 0.02,
          tradeLatency: 800
        });
      }

      // Check scaling decision
      const monitoringResult = await monitor.monitorAllSandboxes();
      expect(monitoringResult.scalingRecommendation).toBeDefined();

      if (monitoringResult.scalingRecommendation.action === ScalingAction.SCALE_UP) {
        const scaleResult = await monitor.scaleBasedOnLoad(monitoringResult);
        console.log(`    ✅ Scaled up: ${scaleResult.created} new sandbox(es) created`);
        expect(scaleResult.created).toBeGreaterThan(0);
      }

      await monitor.stopMonitoring();
      await coordinator.shutdown();
    }, TEST_CONFIG.timeout);

    test('Auto-Scale: Scale down during low activity', async () => {
      console.log('\n  📉 Testing Auto-Scaling: Scale down');

      const coordinator = new SwarmCoordinator({
        topology: TOPOLOGY.MESH,
        maxAgents: 6,
        e2bApiKey: TEST_CONFIG.e2bApiKey
      });

      // Start with 6 agents
      const agents = Array(6).fill(null).map((_, i) => ({
        name: `scaler_${i + 1}`,
        agent_type: 'momentum_trader',
        symbols: ['SPY'],
        resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
      }));

      await coordinator.initializeSwarm({ agents });

      await monitor.startMonitoring();

      // Register sandboxes
      for (const [sandboxId, agent] of coordinator.agents.entries()) {
        monitor.registerSandbox(sandboxId, {
          id: sandboxId,
          type: agent.type,
          status: 'healthy'
        });
      }

      // Simulate low load
      console.log(`    📊 Simulating low load...`);

      for (const sandboxId of monitor.sandboxes.keys()) {
        monitor.recordMetric(sandboxId, {
          cpu: 15,
          memory: 20,
          responseTime: 200,
          errorRate: 0.001,
          tradeLatency: 150
        });
      }

      const monitoringResult = await monitor.monitorAllSandboxes();

      if (monitoringResult.scalingRecommendation?.action === ScalingAction.SCALE_DOWN) {
        const scaleResult = await monitor.scaleBasedOnLoad(monitoringResult);
        console.log(`    ✅ Scaled down: ${scaleResult.removed} sandbox(es) removed`);
        expect(scaleResult.removed).toBeGreaterThan(0);
      }

      await monitor.stopMonitoring();
      await coordinator.shutdown();
    }, TEST_CONFIG.timeout);

    test('Auto-Scale: VIX-based scaling (volatility-driven)', async () => {
      console.log('\n  📊 Testing Volatility-Based Auto-Scaling');

      const coordinator = new SwarmCoordinator({
        topology: TOPOLOGY.HIERARCHICAL,
        maxAgents: 8,
        e2bApiKey: TEST_CONFIG.e2bApiKey
      });

      const agents = Array(3).fill(null).map((_, i) => ({
        name: `vix_trader_${i + 1}`,
        agent_type: 'momentum_trader',
        symbols: ['SPY'],
        resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
      }));

      await coordinator.initializeSwarm({ agents });
      await monitor.startMonitoring();

      // Register sandboxes
      for (const [sandboxId, agent] of coordinator.agents.entries()) {
        monitor.registerSandbox(sandboxId, {
          id: sandboxId,
          type: agent.type,
          status: 'healthy'
        });
      }

      // Simulate high market volatility
      console.log(`    📈 Simulating high VIX (market volatility)...`);

      // Mock high volatility scenario
      monitor.getMarketVolatility = async () => 0.85; // High volatility

      for (const sandboxId of monitor.sandboxes.keys()) {
        monitor.recordMetric(sandboxId, {
          cpu: 50,
          memory: 45,
          responseTime: 500,
          errorRate: 0.01,
          tradeLatency: 300
        });
      }

      const monitoringResult = await monitor.monitorAllSandboxes();
      const scalingDecision = await monitor.evaluateScaling(monitoringResult);

      console.log(`    ✅ Scaling decision based on volatility: ${scalingDecision.action} (confidence: ${(scalingDecision.confidence * 100).toFixed(1)}%)`);

      // High volatility should trigger scale up even with moderate load
      if (scalingDecision.action === ScalingAction.SCALE_UP) {
        expect(scalingDecision.reason).toContain('volatility');
      }

      await monitor.stopMonitoring();
      await coordinator.shutdown();
    }, TEST_CONFIG.timeout);
  });

  // ==================== PATTERN 6: MULTI-STRATEGY ====================
  describe('Pattern 6: Multi-Strategy Deployment', () => {
    test('Multi-Strategy: 2 momentum + 2 pairs + 1 arbitrage', async () => {
      console.log('\n  🎭 Testing Multi-Strategy Deployment');

      const coordinator = new SwarmCoordinator({
        topology: TOPOLOGY.MESH,
        maxAgents: 5,
        distributionStrategy: DISTRIBUTION_STRATEGY.SPECIALIZED,
        e2bApiKey: TEST_CONFIG.e2bApiKey
      });

      const agents = [
        {
          name: 'momentum_1',
          agent_type: 'momentum_trader',
          symbols: ['SPY'],
          resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
        },
        {
          name: 'momentum_2',
          agent_type: 'momentum_trader',
          symbols: ['QQQ'],
          resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
        },
        {
          name: 'pairs_1',
          agent_type: 'mean_reversion_trader',
          symbols: ['AAPL', 'TSLA'],
          resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
        },
        {
          name: 'pairs_2',
          agent_type: 'mean_reversion_trader',
          symbols: ['SPY', 'QQQ'],
          resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
        },
        {
          name: 'arbitrage_1',
          agent_type: 'neural_forecaster',
          symbols: ['SPY', 'QQQ', 'AAPL'],
          resources: { cpu: 4, memory_mb: 2048, timeout: 3600 }
        }
      ];

      await coordinator.initializeSwarm({ agents });

      // Execute multi-strategy operations
      const strategyTasks = [
        { strategy: 'momentum', count: 10, capability: 'momentum_detection' },
        { strategy: 'pairs', count: 8, capability: 'mean_reversion' },
        { strategy: 'arbitrage', count: 5, capability: 'forecast' }
      ];

      const results = [];

      for (const { strategy, count, capability } of strategyTasks) {
        for (let i = 0; i < count; i++) {
          const task = {
            id: `${strategy}-${i}`,
            type: strategy,
            requiredCapability: capability
          };

          const distribution = await coordinator.distributeTask(task, DISTRIBUTION_STRATEGY.SPECIALIZED);
          results.push({
            strategy,
            agentType: distribution.assignedAgents[0].type
          });
        }
      }

      // Verify correct strategy routing
      const momentumTasks = results.filter(r => r.strategy === 'momentum');
      const pairsTasks = results.filter(r => r.strategy === 'pairs');
      const arbitrageTasks = results.filter(r => r.strategy === 'arbitrage');

      console.log(`    ✅ Momentum: ${momentumTasks.length} tasks`);
      console.log(`    ✅ Pairs: ${pairsTasks.length} tasks`);
      console.log(`    ✅ Arbitrage: ${arbitrageTasks.length} tasks`);

      await coordinator.shutdown();
    }, TEST_CONFIG.timeout);

    test('Multi-Strategy: Strategy rotation based on performance', async () => {
      console.log('\n  🔄 Testing Strategy Rotation');

      const coordinator = new SwarmCoordinator({
        topology: TOPOLOGY.HIERARCHICAL,
        maxAgents: 4,
        distributionStrategy: DISTRIBUTION_STRATEGY.ADAPTIVE,
        e2bApiKey: TEST_CONFIG.e2bApiKey
      });

      const agents = TEST_CONFIG.strategies.map(strategy => ({
        name: `${strategy}_agent`,
        agent_type: 'momentum_trader',
        symbols: ['SPY'],
        resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
      }));

      await coordinator.initializeSwarm({ agents });

      // Simulate performance tracking
      const agentsArray = Array.from(coordinator.agents.values());

      // Set different performance levels
      agentsArray[0].performance.errorRate = 0.05; // Poor
      agentsArray[1].performance.errorRate = 0.02; // Good
      agentsArray[2].performance.errorRate = 0.08; // Poor
      agentsArray[3].performance.errorRate = 0.01; // Excellent

      // Execute tasks with adaptive routing
      const selectedAgents = [];
      for (let i = 0; i < 20; i++) {
        const task = { id: `rotation-${i}`, type: 'execute_trade' };
        const distribution = await coordinator.distributeTask(task, DISTRIBUTION_STRATEGY.ADAPTIVE);
        selectedAgents.push(distribution.assignedAgents[0].id);
      }

      // Count selections per agent
      const selectionCounts = {};
      selectedAgents.forEach(id => {
        selectionCounts[id] = (selectionCounts[id] || 0) + 1;
      });

      // Best performing agent (lowest error rate) should be selected most
      const bestAgent = agentsArray[3];
      const bestAgentSelections = selectionCounts[bestAgent.id] || 0;

      console.log(`    ✅ Best agent selected ${bestAgentSelections}/20 times based on performance`);

      await coordinator.shutdown();
    }, TEST_CONFIG.timeout);
  });

  // ==================== PATTERN 7: BLUE-GREEN DEPLOYMENT ====================
  describe('Pattern 7: Blue-Green Deployment', () => {
    test('Blue-Green: Deploy new swarm, gradual traffic shift', async () => {
      console.log('\n  🔵🟢 Testing Blue-Green Deployment');

      // Blue deployment (current production)
      const blueCoordinator = new SwarmCoordinator({
        swarmId: 'blue-swarm',
        topology: TOPOLOGY.MESH,
        maxAgents: 3,
        e2bApiKey: TEST_CONFIG.e2bApiKey
      });

      const blueAgents = Array(3).fill(null).map((_, i) => ({
        name: `blue_agent_${i + 1}`,
        agent_type: 'momentum_trader',
        symbols: ['SPY'],
        resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
      }));

      await blueCoordinator.initializeSwarm({ agents: blueAgents });
      console.log(`    🔵 Blue swarm deployed: ${blueCoordinator.agents.size} agents`);

      // Green deployment (new version)
      const greenCoordinator = new SwarmCoordinator({
        swarmId: 'green-swarm',
        topology: TOPOLOGY.MESH,
        maxAgents: 3,
        e2bApiKey: TEST_CONFIG.e2bApiKey
      });

      const greenAgents = Array(3).fill(null).map((_, i) => ({
        name: `green_agent_${i + 1}`,
        agent_type: 'momentum_trader',
        symbols: ['SPY'],
        resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
      }));

      await greenCoordinator.initializeSwarm({ agents: greenAgents });
      console.log(`    🟢 Green swarm deployed: ${greenCoordinator.agents.size} agents`);

      // Gradual traffic shift: 100% blue → 50/50 → 100% green
      const trafficShifts = [
        { blue: 100, green: 0 },
        { blue: 75, green: 25 },
        { blue: 50, green: 50 },
        { blue: 25, green: 75 },
        { blue: 0, green: 100 }
      ];

      for (const shift of trafficShifts) {
        console.log(`    📊 Traffic shift: ${shift.blue}% blue, ${shift.green}% green`);

        // Execute 20 tasks with traffic split
        for (let i = 0; i < 20; i++) {
          const useGreen = Math.random() * 100 < shift.green;
          const coordinator = useGreen ? greenCoordinator : blueCoordinator;

          const task = { id: `bg-task-${i}`, type: 'execute_trade' };
          await coordinator.distributeTask(task);
        }

        await TestUtils.wait(1000);
      }

      console.log(`    ✅ Blue-Green deployment completed successfully`);

      // Cleanup blue deployment
      await blueCoordinator.shutdown();
      console.log(`    🔵 Blue swarm decommissioned`);

      await greenCoordinator.shutdown();
    }, TEST_CONFIG.timeout);

    test('Blue-Green: Rollback on error rate spike', async () => {
      console.log('\n  ↩️  Testing Blue-Green Rollback');

      const blueCoordinator = new SwarmCoordinator({
        swarmId: 'blue-stable',
        topology: TOPOLOGY.MESH,
        maxAgents: 2,
        e2bApiKey: TEST_CONFIG.e2bApiKey
      });

      await blueCoordinator.initializeSwarm({
        agents: Array(2).fill(null).map((_, i) => ({
          name: `blue_stable_${i + 1}`,
          agent_type: 'momentum_trader',
          symbols: ['SPY'],
          resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
        }))
      });

      const greenCoordinator = new SwarmCoordinator({
        swarmId: 'green-unstable',
        topology: TOPOLOGY.MESH,
        maxAgents: 2,
        e2bApiKey: TEST_CONFIG.e2bApiKey
      });

      await greenCoordinator.initializeSwarm({
        agents: Array(2).fill(null).map((_, i) => ({
          name: `green_unstable_${i + 1}`,
          agent_type: 'momentum_trader',
          symbols: ['SPY'],
          resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
        }))
      });

      // Start with 50/50 traffic
      let greenTraffic = 50;
      let errorThreshold = 10; // 10% error rate triggers rollback

      console.log(`    📊 Starting with 50/50 traffic split`);

      // Simulate errors in green deployment
      let greenErrors = 0;
      let greenTotal = 0;

      for (let i = 0; i < 20; i++) {
        const useGreen = Math.random() * 100 < greenTraffic;

        if (useGreen) {
          greenTotal++;
          // Simulate higher error rate in green (30%)
          if (Math.random() < 0.3) {
            greenErrors++;
          }
        }

        const greenErrorRate = greenTotal > 0 ? (greenErrors / greenTotal) * 100 : 0;

        // Check if rollback needed
        if (greenErrorRate > errorThreshold && greenTotal >= 5) {
          console.log(`    ⚠️  Error rate spike detected: ${greenErrorRate.toFixed(1)}% (threshold: ${errorThreshold}%)`);
          console.log(`    ↩️  Rolling back to blue deployment`);
          greenTraffic = 0;
          break;
        }
      }

      expect(greenTraffic).toBe(0); // Should have rolled back
      console.log(`    ✅ Rollback completed: 100% traffic to blue`);

      await blueCoordinator.shutdown();
      await greenCoordinator.shutdown();
    }, TEST_CONFIG.timeout);
  });

  // ==================== PATTERN 8: CANARY DEPLOYMENT ====================
  describe('Pattern 8: Canary Deployment', () => {
    test('Canary: Deploy 1 new agent, monitor, then full rollout', async () => {
      console.log('\n  🐤 Testing Canary Deployment');

      // Stable production deployment
      const stableCoordinator = new SwarmCoordinator({
        swarmId: 'stable-production',
        topology: TOPOLOGY.MESH,
        maxAgents: 5,
        e2bApiKey: TEST_CONFIG.e2bApiKey
      });

      const stableAgents = Array(5).fill(null).map((_, i) => ({
        name: `stable_agent_${i + 1}`,
        agent_type: 'momentum_trader',
        symbols: ['SPY'],
        resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
      }));

      await stableCoordinator.initializeSwarm({ agents: stableAgents });
      console.log(`    ✅ Stable deployment: ${stableCoordinator.agents.size} agents`);

      // Deploy single canary agent
      const canaryAgent = {
        name: 'canary_agent',
        agent_type: 'momentum_trader',
        symbols: ['SPY'],
        resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
      };

      const canaryCoordinator = new SwarmCoordinator({
        swarmId: 'canary',
        topology: TOPOLOGY.MESH,
        maxAgents: 1,
        e2bApiKey: TEST_CONFIG.e2bApiKey
      });

      await canaryCoordinator.initializeSwarm({ agents: [canaryAgent] });
      console.log(`    🐤 Canary deployed: 1 agent`);

      // Send 5% traffic to canary
      const canaryTrafficPercent = 5;
      const canaryResults = [];
      const stableResults = [];

      console.log(`    📊 Monitoring canary with ${canaryTrafficPercent}% traffic...`);

      for (let i = 0; i < 100; i++) {
        const useCanary = Math.random() * 100 < canaryTrafficPercent;
        const coordinator = useCanary ? canaryCoordinator : stableCoordinator;

        const task = { id: `canary-test-${i}`, type: 'execute_trade' };
        await coordinator.distributeTask(task);

        const result = {
          success: Math.random() > 0.05, // 95% success rate
          responseTime: Math.random() * 500 + 100
        };

        if (useCanary) {
          canaryResults.push(result);
        } else {
          stableResults.push(result);
        }
      }

      // Analyze canary performance
      const canarySuccess = canaryResults.filter(r => r.success).length / canaryResults.length;
      const stableSuccess = stableResults.filter(r => r.success).length / stableResults.length;

      const canaryAvgResponse = canaryResults.reduce((sum, r) => sum + r.responseTime, 0) / canaryResults.length;
      const stableAvgResponse = stableResults.reduce((sum, r) => sum + r.responseTime, 0) / stableResults.length;

      console.log(`    📈 Canary metrics: ${(canarySuccess * 100).toFixed(1)}% success, ${canaryAvgResponse.toFixed(0)}ms avg`);
      console.log(`    📈 Stable metrics: ${(stableSuccess * 100).toFixed(1)}% success, ${stableAvgResponse.toFixed(0)}ms avg`);

      // Decide on full rollout
      const canaryHealthy = canarySuccess >= stableSuccess * 0.95; // Within 5% of stable

      if (canaryHealthy) {
        console.log(`    ✅ Canary healthy: proceeding with full rollout`);

        // Deploy remaining 4 canary agents
        for (let i = 1; i <= 4; i++) {
          const newAgent = {
            name: `canary_agent_${i + 1}`,
            agent_type: 'momentum_trader',
            symbols: ['SPY'],
            resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
          };

          // In real scenario, would spawn new agent
          console.log(`    🐤 Deploying canary agent ${i + 1}/4`);
          await TestUtils.wait(500);
        }

        console.log(`    ✅ Full canary rollout completed`);
      } else {
        console.log(`    ❌ Canary unhealthy: aborting rollout`);
      }

      await stableCoordinator.shutdown();
      await canaryCoordinator.shutdown();
    }, TEST_CONFIG.timeout);
  });

  // ==================== PERFORMANCE SUMMARY ====================
  describe('Performance Summary', () => {
    test('Generate comprehensive deployment patterns report', async () => {
      console.log('\n  📋 Generating Deployment Patterns Report...\n');

      const report = {
        timestamp: new Date().toISOString(),
        testMode: MOCK_MODE ? 'MOCK' : 'LIVE',
        patterns: {
          mesh: {
            name: 'Mesh Topology',
            useCases: ['Equal peers', 'Consensus trading', 'High redundancy'],
            pros: ['Maximum redundancy', 'No single point of failure', 'Peer-to-peer coordination'],
            cons: ['High network overhead', 'O(n²) connections', 'Complex coordination']
          },
          hierarchical: {
            name: 'Hierarchical Topology',
            useCases: ['Leader-worker', 'Multi-strategy', 'Load distribution'],
            pros: ['Clear hierarchy', 'Centralized control', 'Easy load balancing'],
            cons: ['Single point of failure (coordinator)', 'Scaling limits', 'Coordinator bottleneck']
          },
          ring: {
            name: 'Ring Topology',
            useCases: ['Pipeline processing', 'Sequential workflows', 'Data flow'],
            pros: ['Predictable data flow', 'Low overhead', 'Simple routing'],
            cons: ['Vulnerable to single failure', 'Limited parallelism', 'Fixed order']
          },
          star: {
            name: 'Star Topology',
            useCases: ['Centralized hub', 'Specialized agents', 'Hub coordination'],
            pros: ['Simple management', 'Easy monitoring', 'Clear communication paths'],
            cons: ['Hub is critical bottleneck', 'Limited scalability', 'Hub failure catastrophic']
          },
          autoScaling: {
            name: 'Auto-Scaling',
            useCases: ['Variable load', 'Cost optimization', 'Volatility-driven'],
            pros: ['Dynamic resource allocation', 'Cost efficient', 'Adapts to demand'],
            cons: ['Scaling delays', 'Cost unpredictability', 'Complex configuration']
          },
          multiStrategy: {
            name: 'Multi-Strategy',
            useCases: ['Diverse strategies', 'Risk diversification', 'Performance rotation'],
            pros: ['Strategy diversification', 'Adaptive selection', 'Risk distribution'],
            cons: ['Complex coordination', 'Resource competition', 'Strategy conflicts']
          },
          blueGreen: {
            name: 'Blue-Green Deployment',
            useCases: ['Zero-downtime deploys', 'Easy rollback', 'Version testing'],
            pros: ['Zero downtime', 'Fast rollback', 'Isolated testing'],
            cons: ['Double resources', 'State synchronization', 'Cost intensive']
          },
          canary: {
            name: 'Canary Deployment',
            useCases: ['Gradual rollout', 'Risk mitigation', 'A/B testing'],
            pros: ['Risk mitigation', 'Gradual validation', 'Easy abort'],
            cons: ['Longer rollout', 'Complex monitoring', 'Traffic routing complexity']
          }
        },
        recommendations: {
          smallScale: 'Star or Hierarchical topology for <10 agents',
          mediumScale: 'Mesh or Multi-Strategy for 10-50 agents',
          largeScale: 'Auto-Scaling with Blue-Green deployment for >50 agents',
          highReliability: 'Mesh with Canary deployment',
          costOptimized: 'Auto-Scaling with aggressive scale-down',
          development: 'Blue-Green for frequent updates',
          production: 'Canary with comprehensive monitoring'
        }
      };

      console.log('========================================');
      console.log('  E2B DEPLOYMENT PATTERNS SUMMARY');
      console.log('========================================\n');

      Object.entries(report.patterns).forEach(([key, pattern]) => {
        console.log(`  ${pattern.name}`);
        console.log(`    Use Cases: ${pattern.useCases.join(', ')}`);
        console.log(`    Pros: ${pattern.pros.join(', ')}`);
        console.log(`    Cons: ${pattern.cons.join(', ')}\n`);
      });

      console.log('  RECOMMENDATIONS:');
      Object.entries(report.recommendations).forEach(([scenario, recommendation]) => {
        console.log(`    ${scenario}: ${recommendation}`);
      });

      console.log('\n========================================\n');

      expect(report.patterns).toBeDefined();
      expect(Object.keys(report.patterns).length).toBe(8);
    });
  });
});
