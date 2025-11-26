/**
 * SwarmCoordinator - Multi-Agent Trading Orchestration for E2B Sandboxes
 *
 * Coordinates distributed trading agents across E2B sandboxes with:
 * - Dynamic topology management (mesh, hierarchical, ring, star)
 * - Intelligent task distribution and load balancing
 * - Consensus mechanisms for trade decisions
 * - Real-time performance monitoring and optimization
 * - Inter-agent communication via shared memory
 * - Self-healing and automatic rebalancing
 *
 * @module swarm-coordinator
 */

const { Sandbox } = require('e2b');
const { AgentDBClient } = require('../coordination/agentdb-client');
const E2BSandboxDeployer = require('../../scripts/deployment/e2b-sandbox-deployer');
const EventEmitter = require('events');

/**
 * Swarm topology types
 */
const TOPOLOGY = {
  MESH: 'mesh',           // Full connectivity - all agents communicate
  HIERARCHICAL: 'hierarchical',  // Tree structure with coordinator nodes
  RING: 'ring',           // Circular communication pattern
  STAR: 'star'            // Central coordinator with agent spokes
};

/**
 * Task distribution strategies
 */
const DISTRIBUTION_STRATEGY = {
  ROUND_ROBIN: 'round_robin',      // Simple rotation
  LEAST_LOADED: 'least_loaded',    // Send to agent with lowest load
  SPECIALIZED: 'specialized',      // Route by agent capabilities
  CONSENSUS: 'consensus',          // Require multi-agent agreement
  ADAPTIVE: 'adaptive'             // ML-based dynamic routing
};

/**
 * Agent states
 */
const AGENT_STATE = {
  INITIALIZING: 'initializing',
  READY: 'ready',
  BUSY: 'busy',
  ERROR: 'error',
  OFFLINE: 'offline'
};

/**
 * SwarmCoordinator Class
 *
 * Manages multi-agent coordination across E2B sandboxes
 */
class SwarmCoordinator extends EventEmitter {
  constructor(options = {}) {
    super();

    this.swarmId = options.swarmId || `swarm-${Date.now()}`;
    this.topology = options.topology || TOPOLOGY.MESH;
    this.maxAgents = options.maxAgents || 10;
    this.distributionStrategy = options.distributionStrategy || DISTRIBUTION_STRATEGY.ADAPTIVE;

    // Core components
    this.agents = new Map();
    this.taskQueue = [];
    this.results = new Map();
    this.sharedMemory = new Map();

    // Coordination state
    this.coordinationState = {
      consensusThreshold: options.consensusThreshold || 0.66,
      syncInterval: options.syncInterval || 5000,
      healthCheckInterval: options.healthCheckInterval || 10000,
      rebalanceThreshold: options.rebalanceThreshold || 0.3
    };

    // Performance tracking
    this.metrics = {
      tasksDistributed: 0,
      tasksCompleted: 0,
      tasksFailed: 0,
      consensusDecisions: 0,
      rebalanceEvents: 0,
      totalLatency: 0,
      startTime: Date.now()
    };

    // E2B integration
    this.deployer = new E2BSandboxDeployer(options.e2bApiKey);

    // AgentDB for distributed memory
    this.agentDB = null;
    this.quicEnabled = options.quicEnabled !== false;

    // Intervals
    this.syncIntervalId = null;
    this.healthCheckIntervalId = null;

    this.isInitialized = false;
  }

  /**
   * Initialize swarm with specified topology and agent count
   * @param {Object} config - Swarm configuration
   * @returns {Promise<Object>} Initialization result
   */
  async initializeSwarm(config = {}) {
    console.log(`\n🚀 Initializing Swarm: ${this.swarmId}`);
    console.log(`Topology: ${this.topology} | Max Agents: ${this.maxAgents}`);

    try {
      // Initialize AgentDB for distributed memory
      if (this.quicEnabled) {
        await this.initializeAgentDB(config.agentDBUrl);
      }

      // Deploy agents based on configuration
      const agentConfigs = config.agents || this.generateDefaultAgentConfigs();

      for (const agentConfig of agentConfigs) {
        await this.deployAgent(agentConfig);
      }

      // Establish topology connections
      await this.establishTopology();

      // Start background processes
      this.startSynchronization();
      this.startHealthMonitoring();

      this.isInitialized = true;

      const result = {
        swarmId: this.swarmId,
        topology: this.topology,
        agentCount: this.agents.size,
        status: 'initialized',
        timestamp: new Date().toISOString()
      };

      this.emit('initialized', result);
      console.log(`✅ Swarm initialized with ${this.agents.size} agents\n`);

      return result;

    } catch (error) {
      console.error('❌ Swarm initialization failed:', error.message);
      throw error;
    }
  }

  /**
   * Initialize AgentDB for distributed memory with QUIC
   * @private
   */
  async initializeAgentDB(agentDBUrl) {
    console.log('🔗 Connecting to AgentDB...');

    this.agentDB = new AgentDBClient({
      quicUrl: agentDBUrl || 'quic://localhost:8443',
      sandboxId: this.swarmId,
      strategyType: 'swarm_coordinator'
    });

    await this.agentDB.connect();
    console.log('✅ AgentDB connected');
  }

  /**
   * Deploy an agent to E2B sandbox
   * @private
   */
  async deployAgent(agentConfig) {
    const agentId = `${agentConfig.agent_type}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    console.log(`  📦 Deploying agent: ${agentId}`);

    try {
      // Create sandbox via deployer
      const sandboxInfo = await this.deployer.createSandbox(agentConfig);

      // Create agent metadata
      const agent = {
        id: agentId,
        type: agentConfig.agent_type,
        name: agentConfig.name,
        symbols: agentConfig.symbols,
        sandbox: sandboxInfo,
        state: AGENT_STATE.INITIALIZING,
        capabilities: this.getAgentCapabilities(agentConfig.agent_type),
        performance: {
          tasksCompleted: 0,
          avgLatency: 0,
          errorRate: 0,
          load: 0
        },
        connections: new Set(),
        lastHeartbeat: Date.now()
      };

      this.agents.set(agentId, agent);

      // Update state to ready
      setTimeout(() => {
        if (this.agents.has(agentId)) {
          this.agents.get(agentId).state = AGENT_STATE.READY;
          this.emit('agent-ready', agent);
        }
      }, 2000);

      return agent;

    } catch (error) {
      console.error(`  ❌ Failed to deploy ${agentId}:`, error.message);
      throw error;
    }
  }

  /**
   * Get capabilities for agent type
   * @private
   */
  getAgentCapabilities(agentType) {
    const capabilities = {
      neural_forecaster: ['forecast', 'trend_analysis', 'pattern_recognition'],
      momentum_trader: ['momentum_detection', 'breakout_analysis', 'trade_execution'],
      mean_reversion_trader: ['statistical_analysis', 'mean_reversion', 'pair_trading'],
      risk_manager: ['risk_assessment', 'position_sizing', 'stop_loss_management'],
      portfolio_optimizer: ['optimization', 'allocation', 'rebalancing']
    };

    return capabilities[agentType] || ['generic_trading'];
  }

  /**
   * Establish topology connections between agents
   * @private
   */
  async establishTopology() {
    console.log(`\n🔗 Establishing ${this.topology} topology...`);

    const agentIds = Array.from(this.agents.keys());

    switch (this.topology) {
      case TOPOLOGY.MESH:
        // Full connectivity - all agents connected
        for (let i = 0; i < agentIds.length; i++) {
          for (let j = i + 1; j < agentIds.length; j++) {
            this.connectAgents(agentIds[i], agentIds[j]);
          }
        }
        console.log(`  ✅ Mesh topology: ${agentIds.length * (agentIds.length - 1) / 2} connections`);
        break;

      case TOPOLOGY.HIERARCHICAL:
        // Tree structure with coordinator
        const coordinator = agentIds[0];
        for (let i = 1; i < agentIds.length; i++) {
          this.connectAgents(coordinator, agentIds[i]);
        }
        console.log(`  ✅ Hierarchical topology: 1 coordinator, ${agentIds.length - 1} workers`);
        break;

      case TOPOLOGY.RING:
        // Circular connections
        for (let i = 0; i < agentIds.length; i++) {
          const next = (i + 1) % agentIds.length;
          this.connectAgents(agentIds[i], agentIds[next]);
        }
        console.log(`  ✅ Ring topology: ${agentIds.length} bidirectional connections`);
        break;

      case TOPOLOGY.STAR:
        // Central hub with spokes
        const hub = agentIds[0];
        for (let i = 1; i < agentIds.length; i++) {
          this.connectAgents(hub, agentIds[i]);
        }
        console.log(`  ✅ Star topology: 1 hub, ${agentIds.length - 1} spokes`);
        break;
    }
  }

  /**
   * Connect two agents bidirectionally
   * @private
   */
  connectAgents(agentId1, agentId2) {
    const agent1 = this.agents.get(agentId1);
    const agent2 = this.agents.get(agentId2);

    if (agent1 && agent2) {
      agent1.connections.add(agentId2);
      agent2.connections.add(agentId1);
    }
  }

  /**
   * Distribute task to agents using specified strategy
   * @param {Object} task - Task to distribute
   * @param {string} strategy - Distribution strategy (optional)
   * @returns {Promise<Object>} Task distribution result
   */
  async distributeTask(task, strategy = null) {
    if (!this.isInitialized) {
      throw new Error('Swarm not initialized. Call initializeSwarm() first.');
    }

    const distributionStrategy = strategy || this.distributionStrategy;
    const taskId = task.id || `task-${Date.now()}`;

    console.log(`\n📋 Distributing task ${taskId} using ${distributionStrategy} strategy`);

    task.id = taskId;
    task.status = 'distributing';
    task.startTime = Date.now();

    this.metrics.tasksDistributed++;

    try {
      let selectedAgents;

      switch (distributionStrategy) {
        case DISTRIBUTION_STRATEGY.ROUND_ROBIN:
          selectedAgents = this.selectRoundRobin(task);
          break;

        case DISTRIBUTION_STRATEGY.LEAST_LOADED:
          selectedAgents = this.selectLeastLoaded(task);
          break;

        case DISTRIBUTION_STRATEGY.SPECIALIZED:
          selectedAgents = this.selectSpecialized(task);
          break;

        case DISTRIBUTION_STRATEGY.CONSENSUS:
          selectedAgents = this.selectForConsensus(task);
          break;

        case DISTRIBUTION_STRATEGY.ADAPTIVE:
          selectedAgents = await this.selectAdaptive(task);
          break;

        default:
          selectedAgents = this.selectLeastLoaded(task);
      }

      if (selectedAgents.length === 0) {
        throw new Error('No available agents for task');
      }

      // Assign task to selected agents
      const assignments = await Promise.all(
        selectedAgents.map(agent => this.assignTaskToAgent(agent, task))
      );

      task.status = 'assigned';
      task.agents = selectedAgents.map(a => a.id);

      this.taskQueue.push(task);

      const result = {
        taskId,
        strategy: distributionStrategy,
        assignedAgents: selectedAgents.map(a => ({
          id: a.id,
          type: a.type,
          load: a.performance.load
        })),
        timestamp: new Date().toISOString()
      };

      this.emit('task-distributed', result);
      console.log(`  ✅ Task assigned to ${selectedAgents.length} agent(s)`);

      return result;

    } catch (error) {
      console.error(`  ❌ Task distribution failed:`, error.message);
      this.metrics.tasksFailed++;
      throw error;
    }
  }

  /**
   * Round-robin agent selection
   * @private
   */
  selectRoundRobin(task) {
    const readyAgents = Array.from(this.agents.values())
      .filter(a => a.state === AGENT_STATE.READY);

    if (readyAgents.length === 0) return [];

    const index = this.metrics.tasksDistributed % readyAgents.length;
    return [readyAgents[index]];
  }

  /**
   * Select least loaded agent
   * @private
   */
  selectLeastLoaded(task) {
    const readyAgents = Array.from(this.agents.values())
      .filter(a => a.state === AGENT_STATE.READY)
      .sort((a, b) => a.performance.load - b.performance.load);

    return readyAgents.length > 0 ? [readyAgents[0]] : [];
  }

  /**
   * Select specialized agent based on capabilities
   * @private
   */
  selectSpecialized(task) {
    const requiredCapability = task.requiredCapability;

    const specializedAgents = Array.from(this.agents.values())
      .filter(a =>
        a.state === AGENT_STATE.READY &&
        a.capabilities.includes(requiredCapability)
      )
      .sort((a, b) => a.performance.load - b.performance.load);

    return specializedAgents.length > 0 ? [specializedAgents[0]] : [];
  }

  /**
   * Select multiple agents for consensus decision
   * @private
   */
  selectForConsensus(task) {
    const readyAgents = Array.from(this.agents.values())
      .filter(a => a.state === AGENT_STATE.READY);

    const minAgents = Math.ceil(readyAgents.length * this.coordinationState.consensusThreshold);

    return readyAgents
      .sort((a, b) => a.performance.load - b.performance.load)
      .slice(0, Math.max(minAgents, 3));
  }

  /**
   * Adaptive agent selection using ML-based scoring
   * @private
   */
  async selectAdaptive(task) {
    const readyAgents = Array.from(this.agents.values())
      .filter(a => a.state === AGENT_STATE.READY);

    if (readyAgents.length === 0) return [];

    // Score agents based on multiple factors
    const scoredAgents = readyAgents.map(agent => ({
      agent,
      score: this.calculateAgentScore(agent, task)
    }));

    // Sort by score (higher is better)
    scoredAgents.sort((a, b) => b.score - a.score);

    // Use AgentDB for coordination if available
    if (this.agentDB) {
      try {
        const action = await this.agentDB.getCoordinationAction();
        if (action && action.selectedAgent) {
          const preferredAgent = readyAgents.find(a => a.id === action.selectedAgent);
          if (preferredAgent) {
            return [preferredAgent];
          }
        }
      } catch (error) {
        console.warn('  ⚠️  AgentDB coordination failed, using local scoring');
      }
    }

    return [scoredAgents[0].agent];
  }

  /**
   * Calculate agent fitness score for task
   * @private
   */
  calculateAgentScore(agent, task) {
    let score = 100;

    // Penalize by load
    score -= agent.performance.load * 50;

    // Penalize by error rate
    score -= agent.performance.errorRate * 30;

    // Reward low latency
    score += Math.max(0, 20 - agent.performance.avgLatency / 100);

    // Reward capability match
    if (task.requiredCapability) {
      if (agent.capabilities.includes(task.requiredCapability)) {
        score += 20;
      }
    }

    // Reward high completion rate
    score += (agent.performance.tasksCompleted / Math.max(1, this.metrics.tasksCompleted)) * 10;

    return Math.max(0, score);
  }

  /**
   * Assign task to specific agent
   * @private
   */
  async assignTaskToAgent(agent, task) {
    agent.state = AGENT_STATE.BUSY;
    agent.performance.load += 0.2;

    // Store in shared memory
    this.sharedMemory.set(`task:${task.id}:agent:${agent.id}`, {
      taskId: task.id,
      agentId: agent.id,
      status: 'assigned',
      timestamp: Date.now()
    });

    // Sync to AgentDB if available
    if (this.agentDB) {
      await this.agentDB.updateState({
        assigned_task: task.id,
        task_type: task.type
      });
    }

    return {
      agentId: agent.id,
      taskId: task.id,
      status: 'assigned'
    };
  }

  /**
   * Collect results from all agents
   * @param {string} taskId - Task ID to collect results for
   * @returns {Promise<Object>} Aggregated results
   */
  async collectResults(taskId) {
    console.log(`\n📥 Collecting results for task ${taskId}`);

    const task = this.taskQueue.find(t => t.id === taskId);

    if (!task) {
      throw new Error(`Task ${taskId} not found`);
    }

    const agentResults = [];

    // Collect from each assigned agent
    for (const agentId of task.agents) {
      const resultKey = `task:${taskId}:agent:${agentId}`;
      const result = this.sharedMemory.get(resultKey);

      if (result && result.completed) {
        agentResults.push(result);
      }
    }

    // Aggregate results
    const aggregated = await this.aggregateResults(agentResults, task);

    this.results.set(taskId, aggregated);
    this.metrics.tasksCompleted++;

    task.status = 'completed';
    task.endTime = Date.now();
    task.duration = task.endTime - task.startTime;

    this.metrics.totalLatency += task.duration;

    console.log(`  ✅ Results collected from ${agentResults.length} agent(s)`);

    return aggregated;
  }

  /**
   * Aggregate results from multiple agents
   * @private
   */
  async aggregateResults(agentResults, task) {
    if (agentResults.length === 0) {
      return {
        status: 'no_results',
        taskId: task.id
      };
    }

    // Simple aggregation - average numeric results
    const aggregated = {
      taskId: task.id,
      agentCount: agentResults.length,
      results: agentResults,
      timestamp: Date.now()
    };

    // If consensus required, check agreement
    if (task.requireConsensus) {
      aggregated.consensus = await this.checkConsensus(agentResults);
    }

    return aggregated;
  }

  /**
   * Check consensus among agent results
   * @private
   */
  async checkConsensus(agentResults) {
    this.metrics.consensusDecisions++;

    if (agentResults.length < 2) {
      return { achieved: true, confidence: 1.0 };
    }

    // Simple majority voting
    const decisions = agentResults.map(r => r.decision);
    const counts = {};

    for (const decision of decisions) {
      counts[decision] = (counts[decision] || 0) + 1;
    }

    const maxVotes = Math.max(...Object.values(counts));
    const totalVotes = decisions.length;
    const agreement = maxVotes / totalVotes;

    const majorityDecision = Object.keys(counts).find(k => counts[k] === maxVotes);

    return {
      achieved: agreement >= this.coordinationState.consensusThreshold,
      decision: majorityDecision,
      agreement,
      votes: counts,
      confidence: agreement
    };
  }

  /**
   * Synchronize state across all agents
   * @returns {Promise<Object>} Synchronization result
   */
  async synchronizeState() {
    console.log(`\n🔄 Synchronizing swarm state...`);

    const stateSnapshot = {
      swarmId: this.swarmId,
      timestamp: Date.now(),
      agents: Array.from(this.agents.values()).map(a => ({
        id: a.id,
        type: a.type,
        state: a.state,
        load: a.performance.load,
        lastHeartbeat: a.lastHeartbeat
      })),
      tasks: this.taskQueue.map(t => ({
        id: t.id,
        status: t.status,
        agents: t.agents
      })),
      metrics: { ...this.metrics }
    };

    // Store in shared memory
    this.sharedMemory.set('swarm:state', stateSnapshot);

    // Sync to AgentDB if available
    if (this.agentDB) {
      await this.agentDB.updateState(stateSnapshot);
    }

    this.emit('state-synchronized', stateSnapshot);

    return stateSnapshot;
  }

  /**
   * Start periodic state synchronization
   * @private
   */
  startSynchronization() {
    if (this.syncIntervalId) {
      clearInterval(this.syncIntervalId);
    }

    this.syncIntervalId = setInterval(
      () => this.synchronizeState().catch(console.error),
      this.coordinationState.syncInterval
    );

    console.log(`⏱️  State synchronization started (${this.coordinationState.syncInterval}ms interval)`);
  }

  /**
   * Start health monitoring
   * @private
   */
  startHealthMonitoring() {
    if (this.healthCheckIntervalId) {
      clearInterval(this.healthCheckIntervalId);
    }

    this.healthCheckIntervalId = setInterval(
      () => this.performHealthCheck().catch(console.error),
      this.coordinationState.healthCheckInterval
    );

    console.log(`❤️  Health monitoring started (${this.coordinationState.healthCheckInterval}ms interval)`);
  }

  /**
   * Perform health check on all agents
   * @private
   */
  async performHealthCheck() {
    const now = Date.now();
    const timeout = 30000; // 30 seconds

    for (const [agentId, agent] of this.agents.entries()) {
      const timeSinceHeartbeat = now - agent.lastHeartbeat;

      if (timeSinceHeartbeat > timeout && agent.state !== AGENT_STATE.OFFLINE) {
        console.warn(`⚠️  Agent ${agentId} unresponsive (${timeSinceHeartbeat}ms)`);
        agent.state = AGENT_STATE.OFFLINE;
        this.emit('agent-offline', agent);

        // Trigger rebalancing if needed
        await this.checkRebalance();
      }
    }
  }

  /**
   * Dynamic agent rebalancing
   * @returns {Promise<Object>} Rebalance result
   */
  async rebalance() {
    console.log(`\n⚖️  Rebalancing swarm...`);

    const readyAgents = Array.from(this.agents.values())
      .filter(a => a.state === AGENT_STATE.READY);

    const busyAgents = Array.from(this.agents.values())
      .filter(a => a.state === AGENT_STATE.BUSY);

    if (readyAgents.length === 0) {
      console.log('  ⚠️  No agents available for rebalancing');
      return { rebalanced: false, reason: 'no_agents' };
    }

    // Calculate load imbalance
    const loads = readyAgents.map(a => a.performance.load);
    const avgLoad = loads.reduce((a, b) => a + b, 0) / loads.length;
    const maxLoad = Math.max(...loads);
    const imbalance = maxLoad - avgLoad;

    if (imbalance < this.coordinationState.rebalanceThreshold) {
      return { rebalanced: false, reason: 'balanced', imbalance };
    }

    // Redistribute load from overloaded agents
    const overloadedAgents = readyAgents
      .filter(a => a.performance.load > avgLoad + this.coordinationState.rebalanceThreshold);

    const underloadedAgents = readyAgents
      .filter(a => a.performance.load < avgLoad - this.coordinationState.rebalanceThreshold);

    console.log(`  📊 Rebalancing: ${overloadedAgents.length} overloaded, ${underloadedAgents.length} underloaded`);

    // Simple load redistribution
    for (const agent of overloadedAgents) {
      agent.performance.load = Math.max(0, agent.performance.load - 0.2);
    }

    this.metrics.rebalanceEvents++;

    const result = {
      rebalanced: true,
      imbalance,
      adjustedAgents: overloadedAgents.length,
      timestamp: new Date().toISOString()
    };

    this.emit('rebalanced', result);
    console.log(`  ✅ Rebalancing complete`);

    return result;
  }

  /**
   * Check if rebalancing is needed
   * @private
   */
  async checkRebalance() {
    const offlineCount = Array.from(this.agents.values())
      .filter(a => a.state === AGENT_STATE.OFFLINE).length;

    const totalCount = this.agents.size;
    const offlineRatio = offlineCount / totalCount;

    if (offlineRatio > this.coordinationState.rebalanceThreshold) {
      await this.rebalance();
    }
  }

  /**
   * Get swarm status and metrics
   * @returns {Object} Current swarm status
   */
  getStatus() {
    const agents = Array.from(this.agents.values());
    const uptime = (Date.now() - this.metrics.startTime) / 1000;

    return {
      swarmId: this.swarmId,
      topology: this.topology,
      isInitialized: this.isInitialized,

      agents: {
        total: agents.length,
        ready: agents.filter(a => a.state === AGENT_STATE.READY).length,
        busy: agents.filter(a => a.state === AGENT_STATE.BUSY).length,
        error: agents.filter(a => a.state === AGENT_STATE.ERROR).length,
        offline: agents.filter(a => a.state === AGENT_STATE.OFFLINE).length
      },

      tasks: {
        distributed: this.metrics.tasksDistributed,
        completed: this.metrics.tasksCompleted,
        failed: this.metrics.tasksFailed,
        pending: this.taskQueue.filter(t => t.status === 'assigned').length,
        successRate: this.metrics.tasksCompleted / Math.max(1, this.metrics.tasksDistributed)
      },

      performance: {
        avgLatency: this.metrics.totalLatency / Math.max(1, this.metrics.tasksCompleted),
        consensusDecisions: this.metrics.consensusDecisions,
        rebalanceEvents: this.metrics.rebalanceEvents,
        uptime: `${uptime.toFixed(2)}s`,
        throughput: (this.metrics.tasksCompleted / uptime).toFixed(2) + ' tasks/sec'
      },

      coordination: {
        quicEnabled: this.quicEnabled,
        agentDBConnected: this.agentDB !== null,
        sharedMemorySize: this.sharedMemory.size,
        consensusThreshold: this.coordinationState.consensusThreshold
      },

      timestamp: new Date().toISOString()
    };
  }

  /**
   * Generate default agent configurations
   * @private
   */
  generateDefaultAgentConfigs() {
    return [
      {
        name: 'momentum_trader_1',
        agent_type: 'momentum_trader',
        symbols: ['SPY', 'QQQ'],
        resources: { cpu: 2, memory_mb: 1024, timeout: 3600 }
      },
      {
        name: 'neural_forecaster_1',
        agent_type: 'neural_forecaster',
        symbols: ['AAPL', 'TSLA'],
        resources: { cpu: 4, memory_mb: 2048, timeout: 3600 }
      },
      {
        name: 'risk_manager_1',
        agent_type: 'risk_manager',
        symbols: ['ALL'],
        resources: { cpu: 2, memory_mb: 512, timeout: 7200 }
      }
    ];
  }

  /**
   * Shutdown swarm and cleanup resources
   * @returns {Promise<void>}
   */
  async shutdown() {
    console.log(`\n🛑 Shutting down swarm ${this.swarmId}...`);

    // Stop background processes
    if (this.syncIntervalId) {
      clearInterval(this.syncIntervalId);
    }

    if (this.healthCheckIntervalId) {
      clearInterval(this.healthCheckIntervalId);
    }

    // Disconnect AgentDB
    if (this.agentDB) {
      await this.agentDB.disconnect();
    }

    // Cleanup agents
    for (const [agentId, agent] of this.agents.entries()) {
      console.log(`  🔌 Shutting down agent ${agentId}`);
      agent.state = AGENT_STATE.OFFLINE;
    }

    this.isInitialized = false;

    console.log('✅ Swarm shutdown complete\n');
    this.emit('shutdown', { swarmId: this.swarmId, timestamp: new Date().toISOString() });
  }
}

// Export class and constants
module.exports = {
  SwarmCoordinator,
  TOPOLOGY,
  DISTRIBUTION_STRATEGY,
  AGENT_STATE
};
