#!/usr/bin/env node

/**
 * E2B Swarm Deployment CLI
 *
 * Command-line tool for deploying and managing trading strategy swarms
 * with agentic-jujutsu coordination and E2B sandbox isolation.
 *
 * Usage:
 *   node scripts/deploy-swarm.js deploy --strategy momentum --agents 5
 *   node scripts/deploy-swarm.js benchmark --scenario all
 *   node scripts/deploy-swarm.js status
 *   node scripts/deploy-swarm.js patterns
 *
 * @module scripts/deploy-swarm
 */

'use strict';

const { SwarmCoordinator } = require('../swarm/coordinator');
const { E2BBenchmark } = require('../benchmark/e2b-benchmark');
const { program } = require('commander');
const fs = require('fs').promises;
const path = require('path');

// CLI Configuration
program
    .name('deploy-swarm')
    .description('Deploy and manage E2B trading strategy swarms')
    .version('1.1.0');

// Deploy command
program
    .command('deploy')
    .description('Deploy trading strategies to E2B sandboxes')
    .requiredOption('-s, --strategy <name>', 'Strategy name (momentum, mean-reversion, neural-forecast)')
    .option('-a, --agents <number>', 'Number of agents', '1')
    .option('-p, --params <json>', 'Strategy parameters as JSON', '{}')
    .option('--learning', 'Enable learning mode', true)
    .option('--encryption-key <key>', 'Encryption key for secure coordination')
    .action(async (options) => {
        try {
            console.log('\n🚀 Deploying Strategy Swarm\n');

            const agents = parseInt(options.agents, 10);
            const params = JSON.parse(options.params);

            // Initialize coordinator
            const coordinator = new SwarmCoordinator({
                maxAgents: Math.max(agents, 10),
                learningEnabled: options.learning,
                encryptionKey: options.encryptionKey || null
            });

            // Register strategy
            coordinator.registerStrategy(options.strategy, {
                type: options.strategy,
                symbols: params.symbols || ['SPY', 'QQQ', 'IWM'],
                interval: params.interval || '1min',
                threshold: params.threshold || 0.02,
                positionSize: params.positionSize || 10
            });

            // Create deployments
            const deployments = [];
            for (let i = 0; i < agents; i++) {
                deployments.push({
                    strategyName: options.strategy,
                    params: {
                        ...params,
                        agentId: i,
                        symbol: (params.symbols || ['SPY', 'QQQ', 'IWM'])[i % 3]
                    }
                });
            }

            // Deploy swarm
            console.log(`Deploying ${agents} agents with strategy: ${options.strategy}\n`);
            const results = await coordinator.deploySwarm(deployments);

            // Print results
            const successful = results.filter(r => r.status === 'fulfilled').length;
            const failed = results.filter(r => r.status === 'rejected').length;

            console.log('\n📊 Deployment Results:');
            console.log(`   ✅ Successful: ${successful}/${agents}`);
            console.log(`   ❌ Failed: ${failed}/${agents}`);
            console.log(`   Success Rate: ${((successful / agents) * 100).toFixed(1)}%\n`);

            // Print coordinator report
            console.log(coordinator.generateReport());

            // Cleanup
            await coordinator.cleanup();

            process.exit(0);

        } catch (error) {
            console.error('\n❌ Deployment failed:', error.message);
            process.exit(1);
        }
    });

// Benchmark command
program
    .command('benchmark')
    .description('Run performance benchmarks on E2B sandboxes')
    .option('-s, --scenario <name>', 'Scenario name (light-load, medium-load, heavy-load, stress-test, all)', 'all')
    .option('-o, --output <dir>', 'Output directory', './benchmark-results')
    .option('--strategies <list>', 'Comma-separated list of strategies', 'momentum,mean-reversion,neural-forecast')
    .action(async (options) => {
        try {
            console.log('\n📊 Starting Benchmark Suite\n');

            const strategies = options.strategies.split(',').map(s => s.trim());

            // Configure benchmark
            const benchmarkConfig = {
                outputDir: options.output,
                strategies: strategies
            };

            // Filter scenarios if specific one requested
            if (options.scenario !== 'all') {
                benchmarkConfig.scenarios = [
                    { name: options.scenario, agents: 5, iterations: 10 }
                ];
            }

            // Run benchmark
            const benchmark = new E2BBenchmark(benchmarkConfig);
            const report = await benchmark.run();

            console.log('\n✅ Benchmark complete!');
            console.log(`   Results saved to: ${options.output}`);
            console.log(`   Total executions: ${report.summary.totalExecutions}`);
            console.log(`   Success rate: ${(report.summary.successRate * 100).toFixed(2)}%\n`);

            process.exit(0);

        } catch (error) {
            console.error('\n❌ Benchmark failed:', error.message);
            process.exit(1);
        }
    });

// Status command
program
    .command('status')
    .description('Get coordinator and learning status')
    .option('--learning', 'Show learning statistics')
    .option('--patterns', 'Show discovered patterns')
    .action(async (options) => {
        try {
            console.log('\n📊 Swarm Status\n');

            const coordinator = new SwarmCoordinator({
                learningEnabled: true
            });

            // Get metrics
            const metrics = coordinator.getMetrics();

            console.log('Current Metrics:');
            console.log(`  Active Agents: ${metrics.activeAgents}`);
            console.log(`  Active Sandboxes: ${metrics.activeSandboxes}`);
            console.log(`  Total Executions: ${metrics.totalExecutions}`);
            console.log(`  Success Rate: ${(metrics.successRate * 100).toFixed(2)}%`);
            console.log(`  Average Duration: ${metrics.averageDuration.toFixed(2)}ms\n`);

            if (options.learning && metrics.learningStats) {
                console.log('Learning Statistics:');
                console.log(`  Total Trajectories: ${metrics.learningStats.totalTrajectories}`);
                console.log(`  Patterns Discovered: ${metrics.patternsDiscovered}`);
                console.log(`  Average Success: ${(metrics.learningStats.avgSuccessRate * 100).toFixed(2)}%`);
                console.log(`  Improvement Rate: ${(metrics.learningStats.improvementRate * 100).toFixed(2)}%`);
                console.log(`  Prediction Accuracy: ${(metrics.learningStats.predictionAccuracy * 100).toFixed(2)}%\n`);
            }

            if (options.patterns) {
                const patterns = coordinator.getPatterns();
                console.log(`Discovered Patterns (${patterns.length}):`);
                patterns.forEach((pattern, i) => {
                    console.log(`\n  ${i + 1}. ${pattern.name || 'Unnamed Pattern'}`);
                    console.log(`     Success Rate: ${(pattern.successRate * 100).toFixed(1)}%`);
                    console.log(`     Observations: ${pattern.observationCount}`);
                    console.log(`     Confidence: ${(pattern.confidence * 100).toFixed(1)}%`);
                    if (pattern.operationSequence) {
                        console.log(`     Operations: ${pattern.operationSequence.join(' → ')}`);
                    }
                });
                console.log();
            }

            await coordinator.cleanup();
            process.exit(0);

        } catch (error) {
            console.error('\n❌ Status check failed:', error.message);
            process.exit(1);
        }
    });

// Patterns command
program
    .command('patterns')
    .description('View discovered patterns and AI suggestions')
    .option('-t, --task <description>', 'Get AI suggestion for specific task')
    .action(async (options) => {
        try {
            console.log('\n🧠 Pattern Analysis\n');

            const coordinator = new SwarmCoordinator({
                learningEnabled: true
            });

            const patterns = coordinator.getPatterns();

            if (patterns.length === 0) {
                console.log('No patterns discovered yet. Run some strategies to build learning data.\n');
            } else {
                console.log(`Discovered ${patterns.length} patterns:\n`);

                patterns.forEach((pattern, i) => {
                    console.log(`${i + 1}. ${pattern.name || 'Unnamed Pattern'}`);
                    console.log(`   Success Rate: ${(pattern.successRate * 100).toFixed(1)}%`);
                    console.log(`   Observations: ${pattern.observationCount}`);
                    console.log(`   Confidence: ${(pattern.confidence * 100).toFixed(1)}%`);

                    if (pattern.operationSequence && pattern.operationSequence.length > 0) {
                        console.log(`   Operations:`);
                        pattern.operationSequence.forEach(op => {
                            console.log(`     - ${op}`);
                        });
                    }

                    console.log();
                });
            }

            // Get AI suggestion if task provided
            if (options.task) {
                console.log(`\n💡 AI Suggestion for: "${options.task}"\n`);

                const suggestion = coordinator.getSuggestion(options.task, {});

                console.log(`   Confidence: ${(suggestion.confidence * 100).toFixed(1)}%`);
                console.log(`   Expected Success: ${(suggestion.expectedSuccessRate * 100).toFixed(1)}%`);
                console.log(`   Reasoning: ${suggestion.reasoning || 'Not enough data'}`);

                if (suggestion.recommendedOperations && suggestion.recommendedOperations.length > 0) {
                    console.log(`\n   Recommended Operations:`);
                    suggestion.recommendedOperations.forEach((op, i) => {
                        console.log(`     ${i + 1}. ${op}`);
                    });
                }

                console.log();
            }

            await coordinator.cleanup();
            process.exit(0);

        } catch (error) {
            console.error('\n❌ Pattern analysis failed:', error.message);
            process.exit(1);
        }
    });

// Export command - Export coordinator state
program
    .command('export')
    .description('Export coordinator state and learning data')
    .option('-o, --output <file>', 'Output file', './swarm-state.json')
    .action(async (options) => {
        try {
            console.log('\n💾 Exporting Coordinator State\n');

            const coordinator = new SwarmCoordinator({
                learningEnabled: true
            });

            const state = coordinator.exportState();

            // Save to file
            await fs.writeFile(options.output, JSON.stringify(state, null, 2));

            console.log(`✅ State exported to: ${options.output}`);
            console.log(`   Total Trajectories: ${state.learningStats?.totalTrajectories || 0}`);
            console.log(`   Patterns Discovered: ${state.patterns.length}`);
            console.log(`   Registered Strategies: ${state.strategies.length}\n`);

            await coordinator.cleanup();
            process.exit(0);

        } catch (error) {
            console.error('\n❌ Export failed:', error.message);
            process.exit(1);
        }
    });

// Parse arguments
program.parse(process.argv);

// Show help if no command provided
if (!process.argv.slice(2).length) {
    program.outputHelp();
}
