/**
 * ReasoningBank Learning Integration - Complete Example
 *
 * Demonstrates full E2B trading swarm with adaptive learning
 */

const { SwarmCoordinator, TOPOLOGY } = require('../../src/e2b/swarm-coordinator');
const { E2BMonitor } = require('../../src/e2b/monitor-and-scale');
const { addLearningCapabilities } = require('../../src/reasoningbank/swarm-coordinator-integration');
const { addLearningMetrics } = require('../../src/reasoningbank/e2b-monitor-integration');
const { LearningMode } = require('../../src/reasoningbank');

async function runReasoningBankExample() {
  console.log('\n🧠 ReasoningBank Learning Integration Example\n');
  console.log('=' .repeat(60));

  // 1. Create SwarmCoordinator
  console.log('\n📋 Step 1: Creating SwarmCoordinator...');

  const coordinator = new SwarmCoordinator({
    swarmId: 'example-learning-swarm',
    topology: TOPOLOGY.MESH,
    maxAgents: 5,
    distributionStrategy: 'adaptive'
  });

  // 2. Add Learning Capabilities
  console.log('\n📋 Step 2: Adding ReasoningBank learning...');

  addLearningCapabilities(coordinator, {
    mode: LearningMode.ONLINE,      // Learn from every trade
    enableHNSW: true,                 // 150x faster vector search
    enableQuantization: true,         // 4-32x memory reduction
    minPatternOccurrence: 3,          // Require 3+ occurrences for patterns
    similarityThreshold: 0.8          // 80%+ similarity to merge patterns
  });

  // 3. Initialize Swarm and Learning
  console.log('\n📋 Step 3: Initializing swarm and learning...');

  await coordinator.initializeSwarm({
    agents: [
      {
        name: 'momentum_trader',
        agent_type: 'momentum_trader',
        symbols: ['AAPL', 'TSLA'],
        resources: { cpu: 2, memory_mb: 1024 }
      },
      {
        name: 'neural_forecaster',
        agent_type: 'neural_forecaster',
        symbols: ['NVDA', 'AMD'],
        resources: { cpu: 4, memory_mb: 2048 }
      },
      {
        name: 'risk_manager',
        agent_type: 'risk_manager',
        symbols: ['ALL'],
        resources: { cpu: 2, memory_mb: 512 }
      }
    ]
  });

  await coordinator.initializeLearning();

  // 4. Setup E2BMonitor with Learning Metrics
  console.log('\n📋 Step 4: Setting up monitoring...');

  const monitor = new E2BMonitor({
    monitorInterval: 5000,
    scaleUpThreshold: 0.8,
    scaleDownThreshold: 0.3
  });

  addLearningMetrics(monitor, coordinator);

  // Register sandboxes for monitoring
  for (const [agentId, agent] of coordinator.agents.entries()) {
    monitor.registerSandbox(agentId, agent);
  }

  await monitor.startMonitoring();

  // 5. Simulate Trading Episode
  console.log('\n📋 Step 5: Running trading episode...');

  // Start episode
  coordinator.reasoningBank.startEpisode({
    strategy: 'momentum_neural_combined',
    marketRegime: 'bullish',
    targetReturn: 0.05
  });

  // Simulate multiple trades
  const agents = Array.from(coordinator.agents.keys());
  const symbols = ['AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT'];

  console.log('\n  📊 Recording trading decisions...');

  for (let i = 0; i < 20; i++) {
    const agentId = agents[i % agents.length];
    const symbol = symbols[i % symbols.length];

    // Random market conditions
    const marketState = {
      volatility: 15 + Math.random() * 20,
      trend: Math.random() > 0.5 ? 'up' : 'down',
      volume: 1000000 + Math.random() * 2000000,
      expectedDirection: Math.random() > 0.5 ? 'up' : 'down',
      price: 100 + Math.random() * 200
    };

    // Make decision
    const decision = {
      id: `decision-${i}`,
      action: marketState.expectedDirection === 'up' ? 'buy' : 'sell',
      symbol,
      quantity: Math.floor(10 + Math.random() * 90),
      price: marketState.price,
      reasoning: {
        confidence: 0.6 + Math.random() * 0.3,
        factors: [
          marketState.trend === 'up' ? 'bullish_trend' : 'bearish_trend',
          marketState.volatility < 25 ? 'low_volatility' : 'high_volatility',
          'volume_confirmation'
        ],
        riskLevel: marketState.volatility > 25 ? 'high' : 'medium'
      },
      marketState,
      timestamp: Date.now()
    };

    // Record decision
    const trajectory = await coordinator.recordDecision(agentId, decision);

    // Simulate execution outcome
    const success = Math.random() > 0.4; // 60% success rate
    const pnlPercent = success
      ? 0.5 + Math.random() * 3.0      // +0.5% to +3.5% on success
      : -0.5 - Math.random() * 2.0;    // -0.5% to -2.5% on failure

    const outcome = {
      executed: true,
      fillPrice: decision.price * (1 + (Math.random() - 0.5) * 0.001),
      fillQuantity: decision.quantity,
      slippage: (Math.random() - 0.5) * 0.002,
      executionTime: 50 + Math.random() * 200,
      pnl: decision.quantity * decision.price * (pnlPercent / 100),
      pnlPercent,
      riskAdjustedReturn: pnlPercent * (1 - marketState.volatility / 100),
      timestamp: Date.now()
    };

    // Record outcome (verdict automatically judged in ONLINE mode)
    await coordinator.recordOutcome(trajectory.id, outcome);

    // Small delay between trades
    await new Promise(resolve => setTimeout(resolve, 100));
  }

  console.log('  ✅ Recorded 20 trading decisions\n');

  // 6. Query Similar Decisions
  console.log('📋 Step 6: Querying similar past decisions...');

  const currentMarketState = {
    volatility: 22,
    trend: 'up',
    volume: 1500000,
    expectedDirection: 'up'
  };

  const recommendations = await coordinator.getRecommendations(currentMarketState, {
    topK: 5,
    minSimilarity: 0.7
  });

  console.log(`\n  Found ${recommendations.length} similar decisions:`);
  recommendations.forEach((rec, idx) => {
    console.log(`  ${idx + 1}. Similarity: ${rec.similarity.toFixed(3)}, ` +
                `Recommendation: ${rec.recommendation.action}, ` +
                `Confidence: ${rec.recommendation.confidence.toFixed(3)}`);
  });

  // 7. Adapt Agents Based on Learnings
  console.log('\n📋 Step 7: Adapting agent strategies...');

  for (const agentId of agents) {
    const result = await coordinator.adaptAgent(agentId);
    console.log(`  Agent ${agentId}: ${result.adapted ? 'adapted' : result.reason}`);
  }

  // 8. End Episode and Learn
  console.log('\n📋 Step 8: Ending episode and learning patterns...');

  const episodeResult = await coordinator.endTradingEpisode({
    totalTrades: 20,
    profitableTrades: 12,
    totalPnL: 1250,
    avgPnLPercent: 1.2
  });

  console.log('  Episode Summary:');
  console.log(`    - Episode ID: ${episodeResult.episodeId}`);
  console.log(`    - Duration: ${episodeResult.duration}ms`);
  console.log(`    - Trajectories: ${episodeResult.trajectoriesCount}`);

  // 9. Generate Comprehensive Reports
  console.log('\n📋 Step 9: Generating reports...\n');

  // Learning statistics
  const learningStats = coordinator.getLearningStats();
  console.log('  📊 Learning Statistics:');
  console.log(`    - Total Decisions: ${learningStats.totalDecisions}`);
  console.log(`    - Total Outcomes: ${learningStats.totalOutcomes}`);
  console.log(`    - Avg Verdict Score: ${learningStats.avgVerdictScore.toFixed(3)}`);
  console.log(`    - Patterns Learned: ${learningStats.patternsLearned}`);
  console.log(`    - Adaptation Events: ${learningStats.adaptationEvents}`);
  console.log(`    - Learning Rate: ${learningStats.learningRate.toFixed(3)} patterns/sec`);

  // Trajectory statistics
  console.log('\n  📈 Trajectory Statistics:');
  console.log(`    - Total: ${learningStats.trajectoryStats.total}`);
  console.log(`    - Pending: ${learningStats.trajectoryStats.pending}`);
  console.log(`    - Completed: ${learningStats.trajectoryStats.completed}`);
  console.log(`    - Judged: ${learningStats.trajectoryStats.judged}`);
  console.log(`    - Learned: ${learningStats.trajectoryStats.learned}`);

  // Verdict statistics
  console.log('\n  ⚖️  Verdict Statistics:');
  console.log(`    - Total Evaluations: ${learningStats.verdictStats.totalEvaluations}`);
  console.log(`    - Avg Score: ${learningStats.verdictStats.avgScore.toFixed(3)}`);
  console.log('    - Quality Distribution:');
  const qualityDist = learningStats.verdictStats.qualityDistribution;
  for (const [quality, count] of Object.entries(qualityDist)) {
    console.log(`      - ${quality}: ${count}`);
  }

  // Pattern recognizer statistics
  console.log('\n  🔍 Pattern Recognizer Statistics:');
  console.log(`    - Total Patterns: ${learningStats.recognizerStats.totalPatterns}`);
  console.log(`    - Total Queries: ${learningStats.recognizerStats.totalQueries}`);
  console.log(`    - Avg Query Time: ${learningStats.recognizerStats.avgQueryTime.toFixed(2)}ms`);
  console.log(`    - Cache Hit Rate: ${(learningStats.recognizerStats.cacheHitRate * 100).toFixed(1)}%`);

  // Learning health
  console.log('\n  ❤️  Learning Health:');
  const learningHealth = monitor.checkLearningHealth();
  console.log(`    - Status: ${learningHealth.status}`);
  console.log(`    - Issues: ${learningHealth.issues.length}`);
  if (learningHealth.issues.length > 0) {
    learningHealth.issues.forEach(issue => {
      console.log(`      - [${issue.severity}] ${issue.type}: ${issue.message}`);
    });
  }

  // Full health report
  console.log('\n  📋 Full Health Report:');
  const healthReport = await monitor.generateHealthReport();
  console.log(`    - Total Sandboxes: ${healthReport.summary.totalSandboxes}`);
  console.log(`    - Healthy: ${healthReport.summary.status.healthy}`);
  console.log(`    - Learning Enabled: ${healthReport.learning.enabled}`);
  console.log(`    - Learning Mode: ${healthReport.learning.mode}`);

  if (healthReport.recommendations.length > 0) {
    console.log('\n  💡 Recommendations:');
    healthReport.recommendations.forEach(rec => {
      console.log(`    - [${rec.severity}] ${rec.type}: ${rec.message}`);
      console.log(`      Action: ${rec.action}`);
    });
  }

  // Swarm status
  console.log('\n  🐝 Swarm Status:');
  const swarmStatus = coordinator.getStatus();
  console.log(`    - Agents Total: ${swarmStatus.agents.total}`);
  console.log(`    - Agents Ready: ${swarmStatus.agents.ready}`);
  console.log(`    - Tasks Completed: ${swarmStatus.tasks.completed}`);
  console.log(`    - Success Rate: ${(swarmStatus.tasks.successRate * 100).toFixed(1)}%`);

  // 10. Cleanup
  console.log('\n📋 Step 10: Cleaning up...');

  await monitor.stopMonitoring();
  await coordinator.shutdown();

  console.log('\n✅ Example completed successfully!\n');
  console.log('=' .repeat(60));
  console.log('\n🎓 Key Takeaways:');
  console.log('  1. ReasoningBank learns from every trading decision');
  console.log('  2. Verdict scores help identify successful patterns');
  console.log('  3. Vector similarity enables fast pattern recognition');
  console.log('  4. Agents adapt based on learned patterns');
  console.log('  5. Continuous monitoring ensures learning health\n');
}

// Run example
if (require.main === module) {
  runReasoningBankExample()
    .then(() => process.exit(0))
    .catch(error => {
      console.error('\n❌ Error running example:', error);
      process.exit(1);
    });
}

module.exports = { runReasoningBankExample };
