/**
 * E2B Sandbox Benchmark Framework
 *
 * Comprehensive benchmarking system for trading strategies using E2B sandboxes
 * with agentic-jujutsu learning and optimization.
 *
 * Features:
 * - Multi-scenario benchmark execution
 * - Performance profiling (latency, throughput, resource usage)
 * - Statistical analysis (mean, median, p95, p99)
 * - Comparison reports (baseline vs optimized)
 * - Automated optimization recommendations
 * - Learning from benchmark results
 *
 * @module benchmark/e2b-benchmark
 */

'use strict';

const { SwarmCoordinator } = require('../swarm/coordinator');
const { performance } = require('perf_hooks');
const fs = require('fs').promises;
const path = require('path');

/**
 * Benchmark Configuration
 */
const DEFAULT_BENCHMARK_CONFIG = {
    // Test scenarios
    scenarios: [
        { name: 'light-load', agents: 1, iterations: 10 },
        { name: 'medium-load', agents: 5, iterations: 20 },
        { name: 'heavy-load', agents: 10, iterations: 30 },
        { name: 'stress-test', agents: 20, iterations: 50 }
    ],

    // Performance thresholds
    thresholds: {
        maxLatencyMs: 5000,
        minThroughput: 10, // ops/sec
        maxErrorRate: 0.05, // 5%
        minSuccessRate: 0.95 // 95%
    },

    // Strategy parameters
    strategies: ['momentum', 'mean-reversion', 'neural-forecast'],

    // Output configuration
    outputDir: './benchmark-results',
    saveRawData: true,
    generateCharts: false
};

/**
 * E2B Benchmark Runner
 */
class E2BBenchmark {
    constructor(config = {}) {
        this.config = { ...DEFAULT_BENCHMARK_CONFIG, ...config };
        this.results = [];
        this.coordinator = null;
    }

    /**
     * Run complete benchmark suite
     * @returns {Promise<Object>} Benchmark results
     */
    async run() {
        console.log('\n🚀 Starting E2B Benchmark Suite\n');
        console.log('Configuration:');
        console.log(`  Scenarios: ${this.config.scenarios.length}`);
        console.log(`  Strategies: ${this.config.strategies.join(', ')}`);
        console.log(`  Output: ${this.config.outputDir}\n`);

        // Initialize coordinator
        this.coordinator = new SwarmCoordinator({
            maxAgents: 20,
            learningEnabled: true,
            autoOptimize: true
        });

        // Register strategies
        this.config.strategies.forEach(strategy => {
            this.coordinator.registerStrategy(strategy, {
                type: strategy,
                symbols: ['SPY', 'QQQ', 'IWM'],
                interval: '1min'
            });
        });

        const startTime = performance.now();

        // Run all scenarios
        for (const scenario of this.config.scenarios) {
            await this.runScenario(scenario);
        }

        const totalDuration = performance.now() - startTime;

        // Generate comprehensive report
        const report = this.generateReport(totalDuration);

        // Save results
        await this.saveResults(report);

        // Cleanup
        await this.coordinator.cleanup();

        console.log('\n✅ Benchmark suite complete!');
        console.log(`   Total duration: ${(totalDuration / 1000).toFixed(2)}s`);
        console.log(`   Results saved to: ${this.config.outputDir}\n`);

        return report;
    }

    /**
     * Run individual scenario
     * @private
     */
    async runScenario(scenario) {
        console.log(`\n📊 Running scenario: ${scenario.name}`);
        console.log(`   Agents: ${scenario.agents}`);
        console.log(`   Iterations: ${scenario.iterations}\n`);

        const scenarioResults = {
            name: scenario.name,
            agents: scenario.agents,
            iterations: scenario.iterations,
            executions: [],
            metrics: null,
            startTime: new Date().toISOString()
        };

        for (let i = 0; i < scenario.iterations; i++) {
            const iterationStart = performance.now();

            // Create swarm deployments
            const deployments = [];
            for (let j = 0; j < scenario.agents; j++) {
                const strategy = this.config.strategies[j % this.config.strategies.length];
                deployments.push({
                    strategyName: strategy,
                    params: {
                        symbols: ['SPY', 'QQQ', 'IWM'][j % 3],
                        threshold: 0.02 + (j * 0.01),
                        positionSize: 10 + j
                    }
                });
            }

            // Execute swarm
            try {
                const swarmResults = await this.coordinator.deploySwarm(deployments);

                const iterationDuration = performance.now() - iterationStart;
                const successful = swarmResults.filter(r => r.status === 'fulfilled').length;
                const failed = swarmResults.filter(r => r.status === 'rejected').length;

                scenarioResults.executions.push({
                    iteration: i + 1,
                    duration: iterationDuration,
                    successful,
                    failed,
                    successRate: successful / deployments.length,
                    throughput: (deployments.length / iterationDuration) * 1000 // ops/sec
                });

                console.log(`   Iteration ${i + 1}/${scenario.iterations}: ${successful}/${deployments.length} successful, ${iterationDuration.toFixed(0)}ms`);

            } catch (error) {
                console.error(`   ❌ Iteration ${i + 1} failed:`, error.message);
                scenarioResults.executions.push({
                    iteration: i + 1,
                    duration: performance.now() - iterationStart,
                    successful: 0,
                    failed: scenario.agents,
                    successRate: 0,
                    error: error.message
                });
            }
        }

        // Calculate scenario metrics
        scenarioResults.metrics = this.calculateMetrics(scenarioResults.executions);
        scenarioResults.endTime = new Date().toISOString();

        // Add to results
        this.results.push(scenarioResults);

        // Print scenario summary
        console.log(`\n   Summary:`);
        console.log(`     Avg Duration: ${scenarioResults.metrics.avgDuration.toFixed(2)}ms`);
        console.log(`     Avg Success Rate: ${(scenarioResults.metrics.avgSuccessRate * 100).toFixed(1)}%`);
        console.log(`     Avg Throughput: ${scenarioResults.metrics.avgThroughput.toFixed(2)} ops/sec`);
        console.log(`     P95 Latency: ${scenarioResults.metrics.p95Latency.toFixed(2)}ms`);
        console.log(`     P99 Latency: ${scenarioResults.metrics.p99Latency.toFixed(2)}ms`);

        // Check thresholds
        this.checkThresholds(scenario.name, scenarioResults.metrics);
    }

    /**
     * Calculate statistical metrics
     * @private
     */
    calculateMetrics(executions) {
        const durations = executions.map(e => e.duration).filter(d => !isNaN(d));
        const successRates = executions.map(e => e.successRate).filter(r => !isNaN(r));
        const throughputs = executions.map(e => e.throughput).filter(t => !isNaN(t) && isFinite(t));

        // Sort for percentile calculations
        const sortedDurations = [...durations].sort((a, b) => a - b);

        return {
            // Duration metrics
            avgDuration: durations.reduce((a, b) => a + b, 0) / durations.length,
            minDuration: Math.min(...durations),
            maxDuration: Math.max(...durations),
            medianDuration: this.calculatePercentile(sortedDurations, 50),
            p95Latency: this.calculatePercentile(sortedDurations, 95),
            p99Latency: this.calculatePercentile(sortedDurations, 99),

            // Success rate metrics
            avgSuccessRate: successRates.reduce((a, b) => a + b, 0) / successRates.length,
            minSuccessRate: Math.min(...successRates),
            maxSuccessRate: Math.max(...successRates),

            // Throughput metrics
            avgThroughput: throughputs.length > 0
                ? throughputs.reduce((a, b) => a + b, 0) / throughputs.length
                : 0,
            minThroughput: throughputs.length > 0 ? Math.min(...throughputs) : 0,
            maxThroughput: throughputs.length > 0 ? Math.max(...throughputs) : 0,

            // Sample count
            sampleCount: executions.length
        };
    }

    /**
     * Calculate percentile
     * @private
     */
    calculatePercentile(sortedArray, percentile) {
        if (sortedArray.length === 0) return 0;

        const index = (percentile / 100) * (sortedArray.length - 1);
        const lower = Math.floor(index);
        const upper = Math.ceil(index);
        const weight = index - lower;

        if (upper >= sortedArray.length) {
            return sortedArray[sortedArray.length - 1];
        }

        return sortedArray[lower] * (1 - weight) + sortedArray[upper] * weight;
    }

    /**
     * Check performance thresholds
     * @private
     */
    checkThresholds(scenarioName, metrics) {
        const violations = [];

        if (metrics.p95Latency > this.config.thresholds.maxLatencyMs) {
            violations.push(`P95 latency (${metrics.p95Latency.toFixed(2)}ms) exceeds threshold (${this.config.thresholds.maxLatencyMs}ms)`);
        }

        if (metrics.avgThroughput < this.config.thresholds.minThroughput) {
            violations.push(`Throughput (${metrics.avgThroughput.toFixed(2)} ops/sec) below threshold (${this.config.thresholds.minThroughput} ops/sec)`);
        }

        if (metrics.avgSuccessRate < this.config.thresholds.minSuccessRate) {
            violations.push(`Success rate (${(metrics.avgSuccessRate * 100).toFixed(1)}%) below threshold (${this.config.thresholds.minSuccessRate * 100}%)`);
        }

        if (violations.length > 0) {
            console.log(`\n   ⚠️  Threshold Violations in ${scenarioName}:`);
            violations.forEach(v => console.log(`      - ${v}`));
        } else {
            console.log(`\n   ✅ All thresholds passed for ${scenarioName}`);
        }
    }

    /**
     * Generate comprehensive report
     * @private
     */
    generateReport(totalDuration) {
        const coordinatorMetrics = this.coordinator.getMetrics();
        const learningStats = this.coordinator.getLearningStats();
        const patterns = this.coordinator.getPatterns();

        return {
            summary: {
                totalDuration: totalDuration,
                totalExecutions: coordinatorMetrics.totalExecutions,
                successRate: coordinatorMetrics.successRate,
                scenarios: this.results.length,
                timestamp: new Date().toISOString()
            },
            scenarios: this.results,
            coordinatorMetrics,
            learningStats,
            patterns,
            recommendations: this.generateRecommendations(),
            thresholds: this.config.thresholds
        };
    }

    /**
     * Generate optimization recommendations
     * @private
     */
    generateRecommendations() {
        const recommendations = [];

        // Analyze scenario results
        for (const scenario of this.results) {
            if (scenario.metrics.avgSuccessRate < 0.9) {
                recommendations.push({
                    type: 'reliability',
                    priority: 'high',
                    scenario: scenario.name,
                    message: `Success rate (${(scenario.metrics.avgSuccessRate * 100).toFixed(1)}%) is below 90%. Consider adding retry logic or circuit breakers.`
                });
            }

            if (scenario.metrics.p95Latency > 3000) {
                recommendations.push({
                    type: 'performance',
                    priority: 'high',
                    scenario: scenario.name,
                    message: `P95 latency (${scenario.metrics.p95Latency.toFixed(0)}ms) is high. Consider optimizing strategy code or increasing resources.`
                });
            }

            if (scenario.metrics.avgThroughput < 5) {
                recommendations.push({
                    type: 'scalability',
                    priority: 'medium',
                    scenario: scenario.name,
                    message: `Low throughput (${scenario.metrics.avgThroughput.toFixed(2)} ops/sec). Consider parallel execution or caching.`
                });
            }
        }

        // Analyze learning patterns
        const patterns = this.coordinator.getPatterns();
        if (patterns.length > 0) {
            const highSuccessPatterns = patterns.filter(p => p.successRate > 0.9);
            if (highSuccessPatterns.length > 0) {
                recommendations.push({
                    type: 'optimization',
                    priority: 'low',
                    message: `${highSuccessPatterns.length} high-success patterns discovered. Review patterns for best practices.`
                });
            }
        }

        return recommendations;
    }

    /**
     * Save benchmark results
     * @private
     */
    async saveResults(report) {
        // Ensure output directory exists
        await fs.mkdir(this.config.outputDir, { recursive: true });

        const timestamp = new Date().toISOString().replace(/:/g, '-');

        // Save JSON report
        const jsonPath = path.join(this.config.outputDir, `benchmark-${timestamp}.json`);
        await fs.writeFile(jsonPath, JSON.stringify(report, null, 2));
        console.log(`\n📄 Saved JSON report: ${jsonPath}`);

        // Save human-readable report
        const txtPath = path.join(this.config.outputDir, `benchmark-${timestamp}.txt`);
        await fs.writeFile(txtPath, this.formatTextReport(report));
        console.log(`📄 Saved text report: ${txtPath}`);

        // Save CSV for easy analysis
        const csvPath = path.join(this.config.outputDir, `benchmark-${timestamp}.csv`);
        await fs.writeFile(csvPath, this.formatCSV(report));
        console.log(`📄 Saved CSV data: ${csvPath}`);

        // Save coordinator report
        const coordReportPath = path.join(this.config.outputDir, `coordinator-${timestamp}.txt`);
        await fs.writeFile(coordReportPath, this.coordinator.generateReport());
        console.log(`📄 Saved coordinator report: ${coordReportPath}`);
    }

    /**
     * Format text report
     * @private
     */
    formatTextReport(report) {
        let text = '╔═══════════════════════════════════════════════════════════╗\n';
        text += '║      E2B Benchmark Report - Trading Strategies           ║\n';
        text += '╚═══════════════════════════════════════════════════════════╝\n\n';

        text += `Generated: ${report.summary.timestamp}\n`;
        text += `Total Duration: ${(report.summary.totalDuration / 1000).toFixed(2)}s\n`;
        text += `Total Executions: ${report.summary.totalExecutions}\n`;
        text += `Success Rate: ${(report.summary.successRate * 100).toFixed(2)}%\n`;
        text += `Scenarios: ${report.summary.scenarios}\n\n`;

        text += '─'.repeat(63) + '\n';
        text += 'Scenario Results\n';
        text += '─'.repeat(63) + '\n\n';

        for (const scenario of report.scenarios) {
            text += `${scenario.name.toUpperCase()}\n`;
            text += `  Agents: ${scenario.agents}, Iterations: ${scenario.iterations}\n`;
            text += `  Avg Duration: ${scenario.metrics.avgDuration.toFixed(2)}ms\n`;
            text += `  P95 Latency: ${scenario.metrics.p95Latency.toFixed(2)}ms\n`;
            text += `  P99 Latency: ${scenario.metrics.p99Latency.toFixed(2)}ms\n`;
            text += `  Avg Success Rate: ${(scenario.metrics.avgSuccessRate * 100).toFixed(2)}%\n`;
            text += `  Avg Throughput: ${scenario.metrics.avgThroughput.toFixed(2)} ops/sec\n\n`;
        }

        if (report.learningStats) {
            text += '─'.repeat(63) + '\n';
            text += 'Learning Statistics\n';
            text += '─'.repeat(63) + '\n\n';
            text += `  Total Trajectories: ${report.learningStats.totalTrajectories}\n`;
            text += `  Patterns Discovered: ${report.patterns.length}\n`;
            text += `  Average Success: ${(report.learningStats.avgSuccessRate * 100).toFixed(2)}%\n`;
            text += `  Improvement Rate: ${(report.learningStats.improvementRate * 100).toFixed(2)}%\n`;
            text += `  Prediction Accuracy: ${(report.learningStats.predictionAccuracy * 100).toFixed(2)}%\n\n`;
        }

        if (report.recommendations.length > 0) {
            text += '─'.repeat(63) + '\n';
            text += 'Recommendations\n';
            text += '─'.repeat(63) + '\n\n';
            report.recommendations.forEach((rec, i) => {
                text += `${i + 1}. [${rec.priority.toUpperCase()}] ${rec.type}\n`;
                text += `   ${rec.message}\n\n`;
            });
        }

        return text;
    }

    /**
     * Format CSV data
     * @private
     */
    formatCSV(report) {
        let csv = 'Scenario,Agents,Iterations,AvgDuration,P95Latency,P99Latency,AvgSuccessRate,AvgThroughput\n';

        for (const scenario of report.scenarios) {
            csv += `${scenario.name},`;
            csv += `${scenario.agents},`;
            csv += `${scenario.iterations},`;
            csv += `${scenario.metrics.avgDuration.toFixed(2)},`;
            csv += `${scenario.metrics.p95Latency.toFixed(2)},`;
            csv += `${scenario.metrics.p99Latency.toFixed(2)},`;
            csv += `${scenario.metrics.avgSuccessRate.toFixed(4)},`;
            csv += `${scenario.metrics.avgThroughput.toFixed(2)}\n`;
        }

        return csv;
    }
}

// CLI execution
if (require.main === module) {
    const benchmark = new E2BBenchmark();

    benchmark.run()
        .then(report => {
            console.log('\n📊 Benchmark Complete!');
            console.log(`   Success Rate: ${(report.summary.successRate * 100).toFixed(2)}%`);
            console.log(`   Total Executions: ${report.summary.totalExecutions}`);
            process.exit(0);
        })
        .catch(error => {
            console.error('\n❌ Benchmark failed:', error);
            process.exit(1);
        });
}

module.exports = { E2BBenchmark, DEFAULT_BENCHMARK_CONFIG };
