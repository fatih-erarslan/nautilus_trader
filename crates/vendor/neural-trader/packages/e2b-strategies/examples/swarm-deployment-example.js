/**
 * E2B Swarm Deployment Example
 *
 * Demonstrates how to deploy multiple trading strategies concurrently
 * using agentic-jujutsu coordination and E2B sandboxes.
 *
 * This example shows:
 * - Multi-agent swarm deployment
 * - Self-learning from executions
 * - Pattern discovery
 * - AI-powered suggestions
 * - Performance metrics
 */

const { SwarmCoordinator } = require('@neural-trader/e2b-strategies/swarm');

async function main() {
    console.log('🚀 E2B Swarm Deployment Example\n');

    // Initialize swarm coordinator with learning enabled
    const coordinator = new SwarmCoordinator({
        maxAgents: 15,
        learningEnabled: true,
        autoOptimize: true,
        sandboxTimeout: 300000  // 5 minutes
    });

    // Register trading strategies
    coordinator.registerStrategy('momentum', {
        type: 'momentum',
        symbols: ['SPY', 'QQQ', 'IWM'],
        threshold: 0.02,
        positionSize: 10,
        interval: '1min'
    });

    coordinator.registerStrategy('mean-reversion', {
        type: 'mean-reversion',
        symbols: ['SPY', 'QQQ', 'IWM'],
        zScore: 2.0,
        lookback: 20,
        positionSize: 10
    });

    coordinator.registerStrategy('neural-forecast', {
        type: 'neural-forecast',
        symbols: ['SPY', 'QQQ', 'IWM'],
        model: 'lstm',
        forecastHorizon: 5,
        positionSize: 10
    });

    console.log('✅ Strategies registered\n');

    // Create swarm deployments
    const deployments = [
        // Momentum agents
        { strategyName: 'momentum', params: { symbol: 'SPY', threshold: 0.02 } },
        { strategyName: 'momentum', params: { symbol: 'QQQ', threshold: 0.025 } },
        { strategyName: 'momentum', params: { symbol: 'IWM', threshold: 0.03 } },

        // Mean reversion agents
        { strategyName: 'mean-reversion', params: { symbol: 'SPY', zScore: 2.0 } },
        { strategyName: 'mean-reversion', params: { symbol: 'QQQ', zScore: 2.5 } },

        // Neural forecast agents
        { strategyName: 'neural-forecast', params: { symbol: 'SPY', forecastHorizon: 5 } },
        { strategyName: 'neural-forecast', params: { symbol: 'QQQ', forecastHorizon: 10 } }
    ];

    console.log(`📊 Deploying ${deployments.length} agents concurrently...\n`);

    // Deploy swarm
    const startTime = Date.now();
    const results = await coordinator.deploySwarm(deployments);
    const duration = Date.now() - startTime;

    // Analyze results
    const successful = results.filter(r => r.status === 'fulfilled').length;
    const failed = results.filter(r => r.status === 'rejected').length;

    console.log('\n📈 Deployment Results:');
    console.log(`   Total Agents: ${deployments.length}`);
    console.log(`   ✅ Successful: ${successful}`);
    console.log(`   ❌ Failed: ${failed}`);
    console.log(`   Success Rate: ${((successful / deployments.length) * 100).toFixed(1)}%`);
    console.log(`   Total Duration: ${(duration / 1000).toFixed(2)}s`);
    console.log(`   Avg per Agent: ${(duration / deployments.length).toFixed(0)}ms\n`);

    // Get learning statistics
    const learningStats = coordinator.getLearningStats();
    if (learningStats) {
        console.log('🧠 Learning Statistics:');
        console.log(`   Total Trajectories: ${learningStats.totalTrajectories}`);
        console.log(`   Average Success: ${(learningStats.avgSuccessRate * 100).toFixed(1)}%`);
        console.log(`   Improvement Rate: ${(learningStats.improvementRate * 100).toFixed(1)}%`);
        console.log(`   Prediction Accuracy: ${(learningStats.predictionAccuracy * 100).toFixed(1)}%\n`);
    }

    // Get discovered patterns
    const patterns = coordinator.getPatterns();
    if (patterns.length > 0) {
        console.log(`🔍 Discovered ${patterns.length} Patterns:\n`);
        patterns.slice(0, 3).forEach((pattern, i) => {
            console.log(`   ${i + 1}. ${pattern.name || 'Unnamed Pattern'}`);
            console.log(`      Success Rate: ${(pattern.successRate * 100).toFixed(1)}%`);
            console.log(`      Observations: ${pattern.observationCount}`);
            console.log(`      Confidence: ${(pattern.confidence * 100).toFixed(1)}%\n`);
        });
    }

    // Get AI suggestion for next deployment
    console.log('💡 AI Suggestion for next deployment:\n');
    const suggestion = coordinator.getSuggestion('momentum', { symbol: 'SPY' });
    console.log(`   Confidence: ${(suggestion.confidence * 100).toFixed(1)}%`);
    console.log(`   Expected Success: ${(suggestion.expectedSuccessRate * 100).toFixed(1)}%`);
    if (suggestion.reasoning) {
        console.log(`   Reasoning: ${suggestion.reasoning}`);
    }

    // Print comprehensive report
    console.log(coordinator.generateReport());

    // Cleanup
    await coordinator.cleanup();

    console.log('✅ Example complete!\n');
}

// Run example
if (require.main === module) {
    main()
        .then(() => process.exit(0))
        .catch(error => {
            console.error('❌ Example failed:', error);
            process.exit(1);
        });
}

module.exports = { main };
