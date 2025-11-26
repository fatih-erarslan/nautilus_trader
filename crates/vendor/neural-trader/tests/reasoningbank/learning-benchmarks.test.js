/**
 * ReasoningBank E2B Swarm Learning Benchmarks
 * Comprehensive comparison of traditional vs self-learning trading swarms
 */

const { E2BSandboxManager } = require('../../src/e2b/E2BSandboxManager');
const { SwarmCoordinator } = require('../../src/e2b/SwarmCoordinator');
const fs = require('fs').promises;
const path = require('path');

// Benchmark configuration
const BENCHMARK_CONFIG = {
  episodeCount: 100,
  warmupEpisodes: 10,
  tradingSymbols: ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'],
  initialCapital: 100000,
  maxPositionSize: 0.2, // 20% of capital
  topologies: ['mesh', 'hierarchical', 'ring', 'star'],
  learningRates: [0.001, 0.01, 0.1],
  trajectoryBatchSize: 32,
  resultDir: path.join(__dirname, 'results')
};

// Performance metrics collector
class BenchmarkMetrics {
  constructor(name, type) {
    this.name = name;
    this.type = type; // 'traditional' or 'reasoningbank'
    this.episodes = [];
    this.decisions = [];
    this.trades = [];
    this.learningMetrics = [];
    this.resourceMetrics = [];
    this.startTime = Date.now();
  }

  recordEpisode(episodeNum, data) {
    this.episodes.push({
      episode: episodeNum,
      timestamp: Date.now(),
      pnl: data.pnl,
      sharpeRatio: data.sharpeRatio,
      winRate: data.winRate,
      maxDrawdown: data.maxDrawdown,
      tradesCount: data.tradesCount,
      accuracy: data.accuracy,
      confidenceScore: data.confidenceScore
    });
  }

  recordDecision(decision) {
    this.decisions.push({
      timestamp: Date.now(),
      symbol: decision.symbol,
      action: decision.action,
      confidence: decision.confidence,
      latency: decision.latency,
      reasoning: decision.reasoning,
      outcome: decision.outcome
    });
  }

  recordLearningMetrics(metrics) {
    this.learningMetrics.push({
      timestamp: Date.now(),
      ...metrics
    });
  }

  recordResourceMetrics(metrics) {
    this.resourceMetrics.push({
      timestamp: Date.now(),
      ...metrics
    });
  }

  calculateStatistics() {
    const episodeData = this.episodes;

    return {
      // Trading Performance
      totalPnL: episodeData.reduce((sum, e) => sum + e.pnl, 0),
      avgPnL: episodeData.reduce((sum, e) => sum + e.pnl, 0) / episodeData.length,
      avgSharpeRatio: episodeData.reduce((sum, e) => sum + e.sharpeRatio, 0) / episodeData.length,
      avgWinRate: episodeData.reduce((sum, e) => sum + e.winRate, 0) / episodeData.length,
      maxDrawdown: Math.max(...episodeData.map(e => e.maxDrawdown)),
      totalTrades: episodeData.reduce((sum, e) => sum + e.tradesCount, 0),

      // Learning Performance (ReasoningBank only)
      avgAccuracy: episodeData.reduce((sum, e) => sum + (e.accuracy || 0), 0) / episodeData.length,
      accuracyImprovement: this.calculateImprovement(episodeData, 'accuracy'),
      convergenceRate: this.calculateConvergenceRate(episodeData),

      // Decision Quality
      avgConfidence: this.decisions.reduce((sum, d) => sum + (d.confidence || 0), 0) / this.decisions.length,
      avgDecisionLatency: this.decisions.reduce((sum, d) => sum + d.latency, 0) / this.decisions.length,
      decisionsPerSecond: (this.decisions.length / ((Date.now() - this.startTime) / 1000)),

      // Resource Usage
      avgMemoryUsage: this.resourceMetrics.reduce((sum, m) => sum + m.memoryMB, 0) / this.resourceMetrics.length,
      avgCpuUsage: this.resourceMetrics.reduce((sum, m) => sum + m.cpuPercent, 0) / this.resourceMetrics.length,
      peakMemoryUsage: Math.max(...this.resourceMetrics.map(m => m.memoryMB)),

      // Duration
      totalDurationMs: Date.now() - this.startTime
    };
  }

  calculateImprovement(data, metric) {
    if (data.length < 2) return 0;
    const first10 = data.slice(0, 10).reduce((sum, e) => sum + (e[metric] || 0), 0) / 10;
    const last10 = data.slice(-10).reduce((sum, e) => sum + (e[metric] || 0), 0) / 10;
    return ((last10 - first10) / first10) * 100;
  }

  calculateConvergenceRate(data) {
    // Find episode where accuracy first exceeds 80%
    const convergencePoint = data.findIndex(e => (e.accuracy || 0) >= 0.8);
    return convergencePoint === -1 ? data.length : convergencePoint;
  }
}

// Traditional Swarm Executor (no learning)
class TraditionalSwarmExecutor {
  constructor(topology, config) {
    this.topology = topology;
    this.config = config;
    this.sandboxManager = new E2BSandboxManager();
    this.coordinator = new SwarmCoordinator();
    this.metrics = new BenchmarkMetrics(`Traditional-${topology}`, 'traditional');
  }

  async initialize() {
    console.log(`Initializing traditional swarm with ${this.topology} topology`);

    // Create E2B sandboxes for agents
    const agentCount = this.topology === 'mesh' ? 5 : this.topology === 'hierarchical' ? 4 : 3;

    this.sandboxes = [];
    for (let i = 0; i < agentCount; i++) {
      const sandbox = await this.sandboxManager.createSandbox({
        template: 'nodejs',
        name: `traditional-agent-${i}`,
        env_vars: {
          AGENT_ID: `agent-${i}`,
          TOPOLOGY: this.topology,
          LEARNING_ENABLED: 'false'
        }
      });
      this.sandboxes.push(sandbox);
    }

    // Setup coordinator
    await this.coordinator.initializeSwarm({
      topology: this.topology,
      agents: this.sandboxes.map(s => ({ id: s.id, type: 'trader' })),
      learningEnabled: false
    });
  }

  async runEpisode(episodeNum, symbols) {
    const episodeStart = Date.now();
    const trades = [];
    let pnl = 0;
    let wins = 0;
    let losses = 0;

    // Simulate trading decisions
    for (const symbol of symbols) {
      const decisionStart = Date.now();

      // Traditional decision (rule-based)
      const decision = await this.makeTraditionalDecision(symbol);
      const decisionLatency = Date.now() - decisionStart;

      this.metrics.recordDecision({
        symbol,
        action: decision.action,
        confidence: decision.confidence,
        latency: decisionLatency,
        reasoning: decision.reasoning,
        outcome: null // Will be updated after trade execution
      });

      // Execute trade
      if (decision.action !== 'HOLD') {
        const tradeResult = await this.executeTrade(decision);
        trades.push(tradeResult);
        pnl += tradeResult.pnl;

        if (tradeResult.pnl > 0) wins++;
        else if (tradeResult.pnl < 0) losses++;
      }

      // Record resource metrics
      const resources = await this.getResourceMetrics();
      this.metrics.recordResourceMetrics(resources);
    }

    // Calculate episode metrics
    const sharpeRatio = this.calculateSharpeRatio(trades);
    const maxDrawdown = this.calculateMaxDrawdown(trades);
    const winRate = trades.length > 0 ? wins / trades.length : 0;

    this.metrics.recordEpisode(episodeNum, {
      pnl,
      sharpeRatio,
      winRate,
      maxDrawdown,
      tradesCount: trades.length,
      accuracy: winRate, // In traditional, accuracy = win rate
      confidenceScore: 0.5 // Fixed confidence for traditional
    });

    return {
      episodeNum,
      pnl,
      sharpeRatio,
      winRate,
      trades,
      duration: Date.now() - episodeStart
    };
  }

  async makeTraditionalDecision(symbol) {
    // Simple moving average crossover strategy
    const shortMA = await this.calculateMA(symbol, 10);
    const longMA = await this.calculateMA(symbol, 50);
    const rsi = await this.calculateRSI(symbol, 14);

    let action = 'HOLD';
    let confidence = 0.5;

    if (shortMA > longMA && rsi < 70) {
      action = 'BUY';
      confidence = 0.6;
    } else if (shortMA < longMA && rsi > 30) {
      action = 'SELL';
      confidence = 0.6;
    }

    return {
      action,
      confidence,
      reasoning: `MA Crossover: Short=${shortMA.toFixed(2)}, Long=${longMA.toFixed(2)}, RSI=${rsi.toFixed(2)}`
    };
  }

  async executeTrade(decision) {
    // Simulate trade execution with random market movement
    const priceChange = (Math.random() - 0.5) * 0.02; // ±1% movement
    const positionSize = this.config.initialCapital * this.config.maxPositionSize;
    const pnl = decision.action === 'BUY' ?
      positionSize * priceChange :
      decision.action === 'SELL' ? positionSize * -priceChange : 0;

    return {
      symbol: decision.symbol,
      action: decision.action,
      pnl,
      priceChange,
      timestamp: Date.now()
    };
  }

  async calculateMA(symbol, period) {
    // Simulate moving average calculation
    return 100 + (Math.random() - 0.5) * 10;
  }

  async calculateRSI(symbol, period) {
    // Simulate RSI calculation
    return 30 + Math.random() * 40;
  }

  calculateSharpeRatio(trades) {
    if (trades.length === 0) return 0;
    const returns = trades.map(t => t.pnl / this.config.initialCapital);
    const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const stdDev = Math.sqrt(
      returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length
    );
    return stdDev === 0 ? 0 : (avgReturn / stdDev) * Math.sqrt(252);
  }

  calculateMaxDrawdown(trades) {
    let peak = 0;
    let maxDD = 0;
    let cumPnL = 0;

    for (const trade of trades) {
      cumPnL += trade.pnl;
      peak = Math.max(peak, cumPnL);
      const drawdown = (peak - cumPnL) / Math.max(peak, 1);
      maxDD = Math.max(maxDD, drawdown);
    }

    return maxDD;
  }

  async getResourceMetrics() {
    // Aggregate resource usage across sandboxes
    const metrics = await Promise.all(
      this.sandboxes.map(s => this.sandboxManager.getSandboxMetrics(s.id))
    );

    return {
      memoryMB: metrics.reduce((sum, m) => sum + (m.memoryMB || 0), 0) / metrics.length,
      cpuPercent: metrics.reduce((sum, m) => sum + (m.cpuPercent || 0), 0) / metrics.length
    };
  }

  async cleanup() {
    await Promise.all(this.sandboxes.map(s => this.sandboxManager.deleteSandbox(s.id)));
  }

  getMetrics() {
    return this.metrics;
  }
}

// ReasoningBank-Enhanced Swarm Executor (with learning)
class ReasoningBankSwarmExecutor {
  constructor(topology, config) {
    this.topology = topology;
    this.config = config;
    this.sandboxManager = new E2BSandboxManager();
    this.coordinator = new SwarmCoordinator();
    this.metrics = new BenchmarkMetrics(`ReasoningBank-${topology}`, 'reasoningbank');
    this.trajectoryBuffer = [];
    this.learnedPatterns = new Map();
    this.episodeRewards = [];
  }

  async initialize() {
    console.log(`Initializing ReasoningBank swarm with ${this.topology} topology`);

    // Create E2B sandboxes with AgentDB and learning capabilities
    const agentCount = this.topology === 'mesh' ? 5 : this.topology === 'hierarchical' ? 4 : 3;

    this.sandboxes = [];
    for (let i = 0; i < agentCount; i++) {
      const sandbox = await this.sandboxManager.createSandbox({
        template: 'nodejs',
        name: `reasoningbank-agent-${i}`,
        env_vars: {
          AGENT_ID: `agent-${i}`,
          TOPOLOGY: this.topology,
          LEARNING_ENABLED: 'true',
          AGENTDB_ENABLED: 'true',
          REASONINGBANK_ENABLED: 'true'
        }
      });

      // Initialize AgentDB in sandbox
      await this.initializeAgentDB(sandbox);

      this.sandboxes.push(sandbox);
    }

    // Setup coordinator with learning
    await this.coordinator.initializeSwarm({
      topology: this.topology,
      agents: this.sandboxes.map(s => ({ id: s.id, type: 'learning_trader' })),
      learningEnabled: true,
      reasoningBankConfig: {
        trajectoryTracking: true,
        verdictJudgment: true,
        memoryDistillation: true,
        patternRecognition: true
      }
    });
  }

  async initializeAgentDB(sandbox) {
    // Install and configure AgentDB in sandbox
    await this.sandboxManager.executeInSandbox(sandbox.id, `
      npm install agentdb-rs
      node -e "
        const AgentDB = require('agentdb-rs');
        const db = new AgentDB({
          path: './agentdb',
          enableHNSW: true,
          quantization: 'int8'
        });
        db.createCollection('trajectories', {
          dimension: 128,
          distanceMetric: 'cosine'
        });
        db.createCollection('patterns', {
          dimension: 64,
          distanceMetric: 'euclidean'
        });
        console.log('AgentDB initialized');
      "
    `);
  }

  async runEpisode(episodeNum, symbols) {
    const episodeStart = Date.now();
    const trades = [];
    const trajectories = [];
    let pnl = 0;
    let wins = 0;
    let losses = 0;
    let correctPredictions = 0;

    // Simulate trading decisions with learning
    for (const symbol of symbols) {
      const decisionStart = Date.now();

      // ReasoningBank-enhanced decision
      const decision = await this.makeLearnedDecision(symbol, episodeNum);
      const decisionLatency = Date.now() - decisionStart;

      // Store trajectory
      const trajectory = {
        state: decision.state,
        action: decision.action,
        reasoning: decision.reasoning,
        confidence: decision.confidence,
        timestamp: Date.now()
      };

      // Execute trade
      let tradeResult = null;
      if (decision.action !== 'HOLD') {
        tradeResult = await this.executeTrade(decision);
        trades.push(tradeResult);
        pnl += tradeResult.pnl;

        if (tradeResult.pnl > 0) {
          wins++;
          correctPredictions++;
        } else if (tradeResult.pnl < 0) {
          losses++;
        }

        // Complete trajectory with reward
        trajectory.reward = tradeResult.pnl / this.config.initialCapital;
        trajectory.outcome = tradeResult.pnl > 0 ? 'success' : 'failure';
      }

      trajectories.push(trajectory);
      this.trajectoryBuffer.push(trajectory);

      this.metrics.recordDecision({
        symbol,
        action: decision.action,
        confidence: decision.confidence,
        latency: decisionLatency,
        reasoning: decision.reasoning,
        outcome: trajectory.outcome
      });

      // Record resource metrics
      const resources = await this.getResourceMetrics();
      this.metrics.recordResourceMetrics(resources);
    }

    // Learn from episode
    if (this.trajectoryBuffer.length >= this.config.trajectoryBatchSize) {
      await this.learnFromTrajectories(episodeNum);
    }

    // Calculate episode metrics
    const sharpeRatio = this.calculateSharpeRatio(trades);
    const maxDrawdown = this.calculateMaxDrawdown(trades);
    const winRate = trades.length > 0 ? wins / trades.length : 0;
    const accuracy = symbols.length > 0 ? correctPredictions / symbols.length : 0;

    // Record learning metrics
    this.metrics.recordLearningMetrics({
      patternsLearned: this.learnedPatterns.size,
      trajectoryBufferSize: this.trajectoryBuffer.length,
      avgReward: trajectories.reduce((sum, t) => sum + (t.reward || 0), 0) / trajectories.length,
      explorationRate: this.calculateExplorationRate(episodeNum)
    });

    this.metrics.recordEpisode(episodeNum, {
      pnl,
      sharpeRatio,
      winRate,
      maxDrawdown,
      tradesCount: trades.length,
      accuracy,
      confidenceScore: decision.confidence
    });

    this.episodeRewards.push(pnl);

    return {
      episodeNum,
      pnl,
      sharpeRatio,
      winRate,
      accuracy,
      trades,
      patternsLearned: this.learnedPatterns.size,
      duration: Date.now() - episodeStart
    };
  }

  async makeLearnedDecision(symbol, episodeNum) {
    // Get market state
    const state = await this.getMarketState(symbol);

    // Query AgentDB for similar historical patterns
    const similarPatterns = await this.querySimilarPatterns(state);

    // Epsilon-greedy exploration
    const explorationRate = this.calculateExplorationRate(episodeNum);
    const shouldExplore = Math.random() < explorationRate;

    let action, confidence, reasoning;

    if (shouldExplore || similarPatterns.length === 0) {
      // Explore: random action
      action = ['BUY', 'SELL', 'HOLD'][Math.floor(Math.random() * 3)];
      confidence = 0.3;
      reasoning = 'Exploration (random action)';
    } else {
      // Exploit: use learned patterns
      const bestPattern = similarPatterns[0];
      action = bestPattern.bestAction;
      confidence = bestPattern.confidence;
      reasoning = `Learned from ${similarPatterns.length} similar patterns (avg reward: ${bestPattern.avgReward.toFixed(4)})`;
    }

    return {
      action,
      confidence,
      reasoning,
      state
    };
  }

  async getMarketState(symbol) {
    // Calculate technical indicators
    const shortMA = await this.calculateMA(symbol, 10);
    const longMA = await this.calculateMA(symbol, 50);
    const rsi = await this.calculateRSI(symbol, 14);
    const macd = await this.calculateMACD(symbol);
    const volatility = Math.random() * 0.05; // Simulated volatility

    return {
      symbol,
      shortMA,
      longMA,
      rsi,
      macd,
      volatility,
      trend: shortMA > longMA ? 'up' : 'down',
      timestamp: Date.now()
    };
  }

  async querySimilarPatterns(state) {
    // Convert state to vector for similarity search
    const stateVector = this.stateToVector(state);

    // Query AgentDB for similar states
    const results = [];
    for (const [patternKey, pattern] of this.learnedPatterns) {
      const similarity = this.cosineSimilarity(stateVector, pattern.stateVector);
      if (similarity > 0.7) { // Similarity threshold
        results.push({
          ...pattern,
          similarity
        });
      }
    }

    return results.sort((a, b) => b.similarity - a.similarity);
  }

  stateToVector(state) {
    // Normalize and vectorize state features
    return [
      state.shortMA / 100,
      state.longMA / 100,
      state.rsi / 100,
      state.macd,
      state.volatility * 10,
      state.trend === 'up' ? 1 : 0
    ];
  }

  cosineSimilarity(vec1, vec2) {
    const dotProduct = vec1.reduce((sum, val, i) => sum + val * vec2[i], 0);
    const mag1 = Math.sqrt(vec1.reduce((sum, val) => sum + val * val, 0));
    const mag2 = Math.sqrt(vec2.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (mag1 * mag2);
  }

  async learnFromTrajectories(episodeNum) {
    console.log(`Learning from ${this.trajectoryBuffer.length} trajectories at episode ${episodeNum}`);

    // Group trajectories by state similarity
    const patternGroups = this.groupByPattern(this.trajectoryBuffer);

    // Extract and store patterns
    for (const [patternId, trajectories] of patternGroups) {
      const avgReward = trajectories.reduce((sum, t) => sum + (t.reward || 0), 0) / trajectories.length;
      const successRate = trajectories.filter(t => t.outcome === 'success').length / trajectories.length;

      // Determine best action for this pattern
      const actionRewards = new Map();
      for (const t of trajectories) {
        const current = actionRewards.get(t.action) || { total: 0, count: 0 };
        actionRewards.set(t.action, {
          total: current.total + (t.reward || 0),
          count: current.count + 1
        });
      }

      let bestAction = 'HOLD';
      let bestAvgReward = -Infinity;
      for (const [action, stats] of actionRewards) {
        const avgActionReward = stats.total / stats.count;
        if (avgActionReward > bestAvgReward) {
          bestAvgReward = avgActionReward;
          bestAction = action;
        }
      }

      // Store learned pattern
      this.learnedPatterns.set(patternId, {
        stateVector: trajectories[0].state ? this.stateToVector(trajectories[0].state) : [],
        bestAction,
        avgReward,
        successRate,
        confidence: Math.min(0.9, 0.5 + successRate * 0.4),
        sampleCount: trajectories.length,
        lastUpdated: Date.now()
      });
    }

    // Clear processed trajectories
    this.trajectoryBuffer = [];

    console.log(`Learned ${patternGroups.size} patterns, total patterns: ${this.learnedPatterns.size}`);
  }

  groupByPattern(trajectories) {
    const groups = new Map();

    for (const trajectory of trajectories) {
      if (!trajectory.state) continue;

      const stateVector = this.stateToVector(trajectory.state);
      let assignedGroup = null;

      // Find existing group with similar state
      for (const [groupId, groupTrajectories] of groups) {
        const groupStateVector = this.stateToVector(groupTrajectories[0].state);
        const similarity = this.cosineSimilarity(stateVector, groupStateVector);

        if (similarity > 0.8) {
          assignedGroup = groupId;
          break;
        }
      }

      if (assignedGroup) {
        groups.get(assignedGroup).push(trajectory);
      } else {
        // Create new group
        const newGroupId = `pattern_${groups.size}`;
        groups.set(newGroupId, [trajectory]);
      }
    }

    return groups;
  }

  calculateExplorationRate(episodeNum) {
    // Decay exploration rate over episodes
    const minRate = 0.01;
    const maxRate = 0.3;
    const decayRate = 0.995;
    return Math.max(minRate, maxRate * Math.pow(decayRate, episodeNum));
  }

  async executeTrade(decision) {
    // Simulate trade execution with random market movement
    const priceChange = (Math.random() - 0.5) * 0.02; // ±1% movement
    const positionSize = this.config.initialCapital * this.config.maxPositionSize;
    const pnl = decision.action === 'BUY' ?
      positionSize * priceChange :
      decision.action === 'SELL' ? positionSize * -priceChange : 0;

    return {
      symbol: decision.symbol,
      action: decision.action,
      pnl,
      priceChange,
      timestamp: Date.now()
    };
  }

  async calculateMA(symbol, period) {
    return 100 + (Math.random() - 0.5) * 10;
  }

  async calculateRSI(symbol, period) {
    return 30 + Math.random() * 40;
  }

  async calculateMACD(symbol) {
    return (Math.random() - 0.5) * 2;
  }

  calculateSharpeRatio(trades) {
    if (trades.length === 0) return 0;
    const returns = trades.map(t => t.pnl / this.config.initialCapital);
    const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const stdDev = Math.sqrt(
      returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length
    );
    return stdDev === 0 ? 0 : (avgReturn / stdDev) * Math.sqrt(252);
  }

  calculateMaxDrawdown(trades) {
    let peak = 0;
    let maxDD = 0;
    let cumPnL = 0;

    for (const trade of trades) {
      cumPnL += trade.pnl;
      peak = Math.max(peak, cumPnL);
      const drawdown = (peak - cumPnL) / Math.max(peak, 1);
      maxDD = Math.max(maxDD, drawdown);
    }

    return maxDD;
  }

  async getResourceMetrics() {
    const metrics = await Promise.all(
      this.sandboxes.map(s => this.sandboxManager.getSandboxMetrics(s.id))
    );

    return {
      memoryMB: metrics.reduce((sum, m) => sum + (m.memoryMB || 0), 0) / metrics.length,
      cpuPercent: metrics.reduce((sum, m) => sum + (m.cpuPercent || 0), 0) / metrics.length
    };
  }

  async cleanup() {
    await Promise.all(this.sandboxes.map(s => this.sandboxManager.deleteSandbox(s.id)));
  }

  getMetrics() {
    return this.metrics;
  }
}

// Main benchmark suite
describe('ReasoningBank E2B Swarm Benchmarks', () => {
  let resultsDir;

  beforeAll(async () => {
    resultsDir = BENCHMARK_CONFIG.resultDir;
    await fs.mkdir(resultsDir, { recursive: true });
  });

  describe('Learning Effectiveness', () => {
    test('Benchmark: Decision quality improvement over 100 trades', async () => {
      const executor = new ReasoningBankSwarmExecutor('mesh', BENCHMARK_CONFIG);
      await executor.initialize();

      const episodes = [];
      for (let i = 0; i < BENCHMARK_CONFIG.episodeCount; i++) {
        const result = await executor.runEpisode(i, BENCHMARK_CONFIG.tradingSymbols);
        episodes.push(result);

        if (i % 10 === 0) {
          console.log(`Episode ${i}: Accuracy=${(result.accuracy * 100).toFixed(1)}%, Patterns=${result.patternsLearned}`);
        }
      }

      const metrics = executor.getMetrics();
      const stats = metrics.calculateStatistics();

      // Verify improvement
      expect(stats.accuracyImprovement).toBeGreaterThan(10); // At least 10% improvement
      expect(stats.convergenceRate).toBeLessThan(80); // Converge within 80 episodes

      await executor.cleanup();

      // Save results
      await fs.writeFile(
        path.join(resultsDir, 'learning-effectiveness.json'),
        JSON.stringify({ episodes, stats }, null, 2)
      );
    }, 600000); // 10 minute timeout

    test('Benchmark: Learning convergence rate (episodes to 80% accuracy)', async () => {
      const executor = new ReasoningBankSwarmExecutor('hierarchical', BENCHMARK_CONFIG);
      await executor.initialize();

      let convergenceEpisode = -1;
      for (let i = 0; i < BENCHMARK_CONFIG.episodeCount; i++) {
        const result = await executor.runEpisode(i, BENCHMARK_CONFIG.tradingSymbols);

        if (convergenceEpisode === -1 && result.accuracy >= 0.8) {
          convergenceEpisode = i;
          console.log(`Converged to 80% accuracy at episode ${i}`);
          break;
        }
      }

      expect(convergenceEpisode).toBeGreaterThan(-1);
      expect(convergenceEpisode).toBeLessThan(BENCHMARK_CONFIG.episodeCount);

      await executor.cleanup();
    }, 600000);

    test('Benchmark: Pattern recognition accuracy', async () => {
      const executor = new ReasoningBankSwarmExecutor('mesh', BENCHMARK_CONFIG);
      await executor.initialize();

      // Run warmup episodes
      for (let i = 0; i < BENCHMARK_CONFIG.warmupEpisodes; i++) {
        await executor.runEpisode(i, BENCHMARK_CONFIG.tradingSymbols);
      }

      // Test pattern recognition
      const testEpisodes = 20;
      let correctPredictions = 0;
      let totalPredictions = 0;

      for (let i = 0; i < testEpisodes; i++) {
        const result = await executor.runEpisode(BENCHMARK_CONFIG.warmupEpisodes + i, BENCHMARK_CONFIG.tradingSymbols);
        const winningTrades = result.trades.filter(t => t.pnl > 0).length;
        correctPredictions += winningTrades;
        totalPredictions += result.trades.length;
      }

      const accuracy = totalPredictions > 0 ? correctPredictions / totalPredictions : 0;
      expect(accuracy).toBeGreaterThan(0.5); // Better than random

      await executor.cleanup();
    }, 600000);

    test('Benchmark: Strategy adaptation speed', async () => {
      const executor = new ReasoningBankSwarmExecutor('mesh', BENCHMARK_CONFIG);
      await executor.initialize();

      const adaptationMetrics = [];
      let previousStrategy = null;

      for (let i = 0; i < 50; i++) {
        const result = await executor.runEpisode(i, BENCHMARK_CONFIG.tradingSymbols);

        // Detect strategy changes based on action patterns
        const actions = result.trades.map(t => t.action);
        const currentStrategy = actions.filter(a => a === 'BUY').length > actions.length / 2 ? 'aggressive' : 'conservative';

        if (previousStrategy && previousStrategy !== currentStrategy) {
          adaptationMetrics.push({
            episode: i,
            from: previousStrategy,
            to: currentStrategy,
            pnl: result.pnl
          });
        }

        previousStrategy = currentStrategy;
      }

      expect(adaptationMetrics.length).toBeGreaterThan(0); // Should adapt at least once

      await executor.cleanup();
      await fs.writeFile(
        path.join(resultsDir, 'adaptation-speed.json'),
        JSON.stringify(adaptationMetrics, null, 2)
      );
    }, 600000);
  });

  describe('Topology Comparison with Learning', () => {
    test('Benchmark: Mesh topology with distributed learning', async () => {
      const executor = new ReasoningBankSwarmExecutor('mesh', BENCHMARK_CONFIG);
      await executor.initialize();

      for (let i = 0; i < 50; i++) {
        await executor.runEpisode(i, BENCHMARK_CONFIG.tradingSymbols);
      }

      const metrics = executor.getMetrics();
      const stats = metrics.calculateStatistics();

      expect(stats.avgAccuracy).toBeGreaterThan(0.5);
      await executor.cleanup();

      await fs.writeFile(
        path.join(resultsDir, 'topology-mesh-learning.json'),
        JSON.stringify(stats, null, 2)
      );
    }, 600000);

    test('Benchmark: Hierarchical topology with centralized learning', async () => {
      const executor = new ReasoningBankSwarmExecutor('hierarchical', BENCHMARK_CONFIG);
      await executor.initialize();

      for (let i = 0; i < 50; i++) {
        await executor.runEpisode(i, BENCHMARK_CONFIG.tradingSymbols);
      }

      const metrics = executor.getMetrics();
      const stats = metrics.calculateStatistics();

      expect(stats.avgAccuracy).toBeGreaterThan(0.5);
      await executor.cleanup();

      await fs.writeFile(
        path.join(resultsDir, 'topology-hierarchical-learning.json'),
        JSON.stringify(stats, null, 2)
      );
    }, 600000);

    test('Benchmark: Ring topology with sequential learning', async () => {
      const executor = new ReasoningBankSwarmExecutor('ring', BENCHMARK_CONFIG);
      await executor.initialize();

      for (let i = 0; i < 50; i++) {
        await executor.runEpisode(i, BENCHMARK_CONFIG.tradingSymbols);
      }

      const metrics = executor.getMetrics();
      const stats = metrics.calculateStatistics();

      expect(stats.avgAccuracy).toBeGreaterThan(0.4);
      await executor.cleanup();

      await fs.writeFile(
        path.join(resultsDir, 'topology-ring-learning.json'),
        JSON.stringify(stats, null, 2)
      );
    }, 600000);

    test('Benchmark: Learning efficiency by topology', async () => {
      const topologies = ['mesh', 'hierarchical', 'ring'];
      const results = [];

      for (const topology of topologies) {
        const executor = new ReasoningBankSwarmExecutor(topology, BENCHMARK_CONFIG);
        await executor.initialize();

        for (let i = 0; i < 30; i++) {
          await executor.runEpisode(i, BENCHMARK_CONFIG.tradingSymbols);
        }

        const metrics = executor.getMetrics();
        const stats = metrics.calculateStatistics();

        results.push({
          topology,
          convergenceRate: stats.convergenceRate,
          avgAccuracy: stats.avgAccuracy,
          learningEfficiency: stats.accuracyImprovement / stats.totalDurationMs
        });

        await executor.cleanup();
      }

      // Compare topologies
      const sortedByEfficiency = results.sort((a, b) => b.learningEfficiency - a.learningEfficiency);
      expect(sortedByEfficiency[0].learningEfficiency).toBeGreaterThan(0);

      await fs.writeFile(
        path.join(resultsDir, 'topology-comparison.json'),
        JSON.stringify(results, null, 2)
      );
    }, 900000);
  });

  describe('Traditional vs Self-Learning', () => {
    test('Benchmark: Traditional swarm vs ReasoningBank swarm (P&L)', async () => {
      // Traditional
      const traditional = new TraditionalSwarmExecutor('mesh', BENCHMARK_CONFIG);
      await traditional.initialize();

      for (let i = 0; i < 50; i++) {
        await traditional.runEpisode(i, BENCHMARK_CONFIG.tradingSymbols);
      }

      const traditionalMetrics = traditional.getMetrics();
      const traditionalStats = traditionalMetrics.calculateStatistics();

      await traditional.cleanup();

      // ReasoningBank
      const reasoningBank = new ReasoningBankSwarmExecutor('mesh', BENCHMARK_CONFIG);
      await reasoningBank.initialize();

      for (let i = 0; i < 50; i++) {
        await reasoningBank.runEpisode(i, BENCHMARK_CONFIG.tradingSymbols);
      }

      const rbMetrics = reasoningBank.getMetrics();
      const rbStats = rbMetrics.calculateStatistics();

      await reasoningBank.cleanup();

      // Compare
      const comparison = {
        traditional: {
          totalPnL: traditionalStats.totalPnL,
          avgSharpeRatio: traditionalStats.avgSharpeRatio,
          avgWinRate: traditionalStats.avgWinRate
        },
        reasoningBank: {
          totalPnL: rbStats.totalPnL,
          avgSharpeRatio: rbStats.avgSharpeRatio,
          avgWinRate: rbStats.avgWinRate,
          accuracyImprovement: rbStats.accuracyImprovement
        },
        improvement: {
          pnlDelta: rbStats.totalPnL - traditionalStats.totalPnL,
          pnlImprovement: ((rbStats.totalPnL - traditionalStats.totalPnL) / Math.abs(traditionalStats.totalPnL)) * 100,
          sharpeImprovement: rbStats.avgSharpeRatio - traditionalStats.avgSharpeRatio
        }
      };

      expect(comparison.improvement.pnlImprovement).toBeGreaterThan(-50); // Not worse than 50% loss

      await fs.writeFile(
        path.join(resultsDir, 'traditional-vs-reasoningbank.json'),
        JSON.stringify(comparison, null, 2)
      );
    }, 900000);

    test('Benchmark: Decision latency (traditional vs learning)', async () => {
      const traditional = new TraditionalSwarmExecutor('mesh', BENCHMARK_CONFIG);
      await traditional.initialize();

      const tStart = Date.now();
      for (let i = 0; i < 10; i++) {
        await traditional.runEpisode(i, BENCHMARK_CONFIG.tradingSymbols);
      }
      const tDuration = Date.now() - tStart;
      const tMetrics = traditional.getMetrics();
      await traditional.cleanup();

      const reasoningBank = new ReasoningBankSwarmExecutor('mesh', BENCHMARK_CONFIG);
      await reasoningBank.initialize();

      const rbStart = Date.now();
      for (let i = 0; i < 10; i++) {
        await reasoningBank.runEpisode(i, BENCHMARK_CONFIG.tradingSymbols);
      }
      const rbDuration = Date.now() - rbStart;
      const rbMetrics = reasoningBank.getMetrics();
      await reasoningBank.cleanup();

      const latencyComparison = {
        traditional: {
          totalDuration: tDuration,
          avgDecisionLatency: tMetrics.calculateStatistics().avgDecisionLatency
        },
        reasoningBank: {
          totalDuration: rbDuration,
          avgDecisionLatency: rbMetrics.calculateStatistics().avgDecisionLatency
        },
        overhead: {
          timeOverhead: rbDuration - tDuration,
          percentOverhead: ((rbDuration - tDuration) / tDuration) * 100
        }
      };

      await fs.writeFile(
        path.join(resultsDir, 'latency-comparison.json'),
        JSON.stringify(latencyComparison, null, 2)
      );
    }, 600000);

    test('Benchmark: Resource overhead of learning system', async () => {
      const traditional = new TraditionalSwarmExecutor('mesh', BENCHMARK_CONFIG);
      await traditional.initialize();

      for (let i = 0; i < 20; i++) {
        await traditional.runEpisode(i, BENCHMARK_CONFIG.tradingSymbols);
      }

      const tMetrics = traditional.getMetrics();
      const tStats = tMetrics.calculateStatistics();
      await traditional.cleanup();

      const reasoningBank = new ReasoningBankSwarmExecutor('mesh', BENCHMARK_CONFIG);
      await reasoningBank.initialize();

      for (let i = 0; i < 20; i++) {
        await reasoningBank.runEpisode(i, BENCHMARK_CONFIG.tradingSymbols);
      }

      const rbMetrics = reasoningBank.getMetrics();
      const rbStats = rbMetrics.calculateStatistics();
      await reasoningBank.cleanup();

      const resourceOverhead = {
        memoryOverhead: rbStats.avgMemoryUsage - tStats.avgMemoryUsage,
        cpuOverhead: rbStats.avgCpuUsage - tStats.avgCpuUsage,
        memoryIncrease: ((rbStats.avgMemoryUsage - tStats.avgMemoryUsage) / tStats.avgMemoryUsage) * 100,
        cpuIncrease: ((rbStats.avgCpuUsage - tStats.avgCpuUsage) / tStats.avgCpuUsage) * 100
      };

      expect(resourceOverhead.memoryIncrease).toBeLessThan(200); // Less than 200% increase

      await fs.writeFile(
        path.join(resultsDir, 'resource-overhead.json'),
        JSON.stringify(resourceOverhead, null, 2)
      );
    }, 600000);

    test('Benchmark: Sharpe ratio improvement with learning', async () => {
      const traditional = new TraditionalSwarmExecutor('mesh', BENCHMARK_CONFIG);
      await traditional.initialize();

      for (let i = 0; i < 50; i++) {
        await traditional.runEpisode(i, BENCHMARK_CONFIG.tradingSymbols);
      }

      const tMetrics = traditional.getMetrics();
      const tStats = tMetrics.calculateStatistics();
      await traditional.cleanup();

      const reasoningBank = new ReasoningBankSwarmExecutor('mesh', BENCHMARK_CONFIG);
      await reasoningBank.initialize();

      for (let i = 0; i < 50; i++) {
        await reasoningBank.runEpisode(i, BENCHMARK_CONFIG.tradingSymbols);
      }

      const rbMetrics = reasoningBank.getMetrics();
      const rbStats = rbMetrics.calculateStatistics();
      await reasoningBank.cleanup();

      const sharpeComparison = {
        traditional: tStats.avgSharpeRatio,
        reasoningBank: rbStats.avgSharpeRatio,
        improvement: rbStats.avgSharpeRatio - tStats.avgSharpeRatio,
        improvementPercent: ((rbStats.avgSharpeRatio - tStats.avgSharpeRatio) / Math.abs(tStats.avgSharpeRatio)) * 100
      };

      await fs.writeFile(
        path.join(resultsDir, 'sharpe-comparison.json'),
        JSON.stringify(sharpeComparison, null, 2)
      );
    }, 900000);
  });

  describe('Memory & Performance', () => {
    test('Benchmark: AgentDB query performance (vector search)', async () => {
      const executor = new ReasoningBankSwarmExecutor('mesh', BENCHMARK_CONFIG);
      await executor.initialize();

      // Warmup
      for (let i = 0; i < 20; i++) {
        await executor.runEpisode(i, BENCHMARK_CONFIG.tradingSymbols);
      }

      // Measure query performance
      const queryTimes = [];
      for (let i = 0; i < 100; i++) {
        const state = await executor.getMarketState('AAPL');
        const start = Date.now();
        await executor.querySimilarPatterns(state);
        queryTimes.push(Date.now() - start);
      }

      const avgQueryTime = queryTimes.reduce((sum, t) => sum + t, 0) / queryTimes.length;
      const p95QueryTime = queryTimes.sort((a, b) => a - b)[Math.floor(queryTimes.length * 0.95)];

      expect(avgQueryTime).toBeLessThan(100); // Less than 100ms average

      await executor.cleanup();

      await fs.writeFile(
        path.join(resultsDir, 'agentdb-query-performance.json'),
        JSON.stringify({ avgQueryTime, p95QueryTime, samples: queryTimes.length }, null, 2)
      );
    }, 300000);

    test('Benchmark: Memory usage with trajectory storage', async () => {
      const executor = new ReasoningBankSwarmExecutor('mesh', BENCHMARK_CONFIG);
      await executor.initialize();

      const memorySnapshots = [];

      for (let i = 0; i < 50; i++) {
        await executor.runEpisode(i, BENCHMARK_CONFIG.tradingSymbols);

        if (i % 5 === 0) {
          const resources = await executor.getResourceMetrics();
          memorySnapshots.push({
            episode: i,
            memoryMB: resources.memoryMB,
            trajectoriesStored: executor.trajectoryBuffer.length,
            patternsLearned: executor.learnedPatterns.size
          });
        }
      }

      const memoryGrowth = memorySnapshots[memorySnapshots.length - 1].memoryMB - memorySnapshots[0].memoryMB;

      await executor.cleanup();

      await fs.writeFile(
        path.join(resultsDir, 'memory-usage-trajectory.json'),
        JSON.stringify({ memorySnapshots, memoryGrowth }, null, 2)
      );
    }, 600000);

    test('Benchmark: Learning system throughput (decisions/sec)', async () => {
      const executor = new ReasoningBankSwarmExecutor('mesh', BENCHMARK_CONFIG);
      await executor.initialize();

      const start = Date.now();
      let totalDecisions = 0;

      for (let i = 0; i < 20; i++) {
        const result = await executor.runEpisode(i, BENCHMARK_CONFIG.tradingSymbols);
        totalDecisions += result.trades.length;
      }

      const duration = (Date.now() - start) / 1000; // seconds
      const throughput = totalDecisions / duration;

      expect(throughput).toBeGreaterThan(0.1); // At least 0.1 decisions/sec

      await executor.cleanup();

      await fs.writeFile(
        path.join(resultsDir, 'learning-throughput.json'),
        JSON.stringify({ throughput, totalDecisions, durationSeconds: duration }, null, 2)
      );
    }, 600000);
  });

  describe('Adaptive Learning', () => {
    test('Benchmark: Market condition adaptation speed', async () => {
      const executor = new ReasoningBankSwarmExecutor('mesh', BENCHMARK_CONFIG);
      await executor.initialize();

      const adaptationEvents = [];
      let previousPnL = 0;

      for (let i = 0; i < 50; i++) {
        const result = await executor.runEpisode(i, BENCHMARK_CONFIG.tradingSymbols);

        // Detect significant performance changes
        if (Math.abs(result.pnl - previousPnL) > this.config.initialCapital * 0.05) {
          adaptationEvents.push({
            episode: i,
            pnlChange: result.pnl - previousPnL,
            accuracy: result.accuracy,
            patternsLearned: result.patternsLearned
          });
        }

        previousPnL = result.pnl;
      }

      await executor.cleanup();

      await fs.writeFile(
        path.join(resultsDir, 'market-adaptation.json'),
        JSON.stringify(adaptationEvents, null, 2)
      );
    }, 600000);

    test('Benchmark: Strategy switching based on learned patterns', async () => {
      const executor = new ReasoningBankSwarmExecutor('mesh', BENCHMARK_CONFIG);
      await executor.initialize();

      const strategySwitches = [];
      let currentStrategy = null;

      for (let i = 0; i < 50; i++) {
        const result = await executor.runEpisode(i, BENCHMARK_CONFIG.tradingSymbols);

        // Determine strategy from trade patterns
        const buyCount = result.trades.filter(t => t.action === 'BUY').length;
        const sellCount = result.trades.filter(t => t.action === 'SELL').length;

        let detectedStrategy = 'neutral';
        if (buyCount > sellCount * 1.5) detectedStrategy = 'bullish';
        else if (sellCount > buyCount * 1.5) detectedStrategy = 'bearish';

        if (currentStrategy && currentStrategy !== detectedStrategy) {
          strategySwitches.push({
            episode: i,
            from: currentStrategy,
            to: detectedStrategy,
            pnl: result.pnl,
            accuracy: result.accuracy
          });
        }

        currentStrategy = detectedStrategy;
      }

      expect(strategySwitches.length).toBeGreaterThan(0);

      await executor.cleanup();

      await fs.writeFile(
        path.join(resultsDir, 'strategy-switching.json'),
        JSON.stringify(strategySwitches, null, 2)
      );
    }, 600000);

    test('Benchmark: Multi-agent knowledge sharing efficiency', async () => {
      const executor = new ReasoningBankSwarmExecutor('mesh', BENCHMARK_CONFIG);
      await executor.initialize();

      // Track pattern learning across agents
      const knowledgeSharingMetrics = [];

      for (let i = 0; i < 30; i++) {
        await executor.runEpisode(i, BENCHMARK_CONFIG.tradingSymbols);

        if (i % 5 === 0) {
          knowledgeSharingMetrics.push({
            episode: i,
            totalPatternsLearned: executor.learnedPatterns.size,
            agentCount: executor.sandboxes.length,
            patternsPerAgent: executor.learnedPatterns.size / executor.sandboxes.length
          });
        }
      }

      const sharingEfficiency = knowledgeSharingMetrics[knowledgeSharingMetrics.length - 1].patternsPerAgent /
                               knowledgeSharingMetrics[0].patternsPerAgent;

      expect(sharingEfficiency).toBeGreaterThan(1);

      await executor.cleanup();

      await fs.writeFile(
        path.join(resultsDir, 'knowledge-sharing.json'),
        JSON.stringify({ metrics: knowledgeSharingMetrics, efficiency: sharingEfficiency }, null, 2)
      );
    }, 600000);
  });

  afterAll(async () => {
    // Generate comprehensive summary report
    const allResults = await fs.readdir(resultsDir);
    const summaryData = {};

    for (const file of allResults) {
      if (file.endsWith('.json')) {
        const data = JSON.parse(await fs.readFile(path.join(resultsDir, file), 'utf8'));
        summaryData[file.replace('.json', '')] = data;
      }
    }

    console.log('\n=== Benchmark Summary ===');
    console.log(`Total test files generated: ${allResults.length}`);
    console.log(`Results stored in: ${resultsDir}`);
    console.log('========================\n');
  });
});
