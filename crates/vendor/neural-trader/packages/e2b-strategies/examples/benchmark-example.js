/**
 * E2B Benchmark Example
 *
 * Demonstrates comprehensive benchmarking of trading strategies
 * using E2B sandboxes with performance metrics and analysis.
 *
 * This example shows:
 * - Multiple benchmark scenarios
 * - Performance threshold checking
 * - Statistical analysis (mean, p95, p99)
 * - Optimization recommendations
 * - Report generation
 */

const { E2BBenchmark } = require('@neural-trader/e2b-strategies/benchmark');

async function main() {
    console.log('📊 E2B Benchmark Example\n');

    // Configure benchmark suite
    const benchmark = new E2BBenchmark({
        // Test scenarios
        scenarios: [
            {
                name: 'light-load',
                agents: 2,
                iterations: 5
            },
            {
                name: 'medium-load',
                agents: 5,
                iterations: 10
            },
            {
                name: 'heavy-load',
                agents: 10,
                iterations: 15
            }
        ],

        // Strategies to test
        strategies: ['momentum', 'mean-reversion', 'neural-forecast'],

        // Performance thresholds
        thresholds: {
            maxLatencyMs: 5000,           // 5s max latency
            minThroughput: 5,              // 5 ops/sec minimum
            maxErrorRate: 0.05,            // 5% max errors
            minSuccessRate: 0.90           // 90% min success
        },

        // Output configuration
        outputDir: './benchmark-results',
        saveRawData: true
    });

    console.log('Configuration:');
    console.log(`  Scenarios: ${benchmark.config.scenarios.length}`);
    console.log(`  Strategies: ${benchmark.config.strategies.join(', ')}`);
    console.log(`  Output: ${benchmark.config.outputDir}\n`);

    // Run benchmark suite
    console.log('🚀 Starting benchmark...\n');
    const startTime = Date.now();

    const report = await benchmark.run();

    const totalDuration = Date.now() - startTime;

    // Display summary
    console.log('\n' + '='.repeat(60));
    console.log('Benchmark Summary');
    console.log('='.repeat(60) + '\n');

    console.log('Overall Results:');
    console.log(`  Total Duration: ${(totalDuration / 1000).toFixed(2)}s`);
    console.log(`  Total Executions: ${report.summary.totalExecutions}`);
    console.log(`  Success Rate: ${(report.summary.successRate * 100).toFixed(2)}%`);
    console.log(`  Scenarios Completed: ${report.summary.scenarios}\n`);

    // Display scenario results
    console.log('Scenario Breakdown:\n');
    report.scenarios.forEach(scenario => {
        console.log(`  ${scenario.name.toUpperCase()}:`);
        console.log(`    Agents: ${scenario.agents}, Iterations: ${scenario.iterations}`);
        console.log(`    Avg Duration: ${scenario.metrics.avgDuration.toFixed(2)}ms`);
        console.log(`    P95 Latency: ${scenario.metrics.p95Latency.toFixed(2)}ms`);
        console.log(`    P99 Latency: ${scenario.metrics.p99Latency.toFixed(2)}ms`);
        console.log(`    Success Rate: ${(scenario.metrics.avgSuccessRate * 100).toFixed(2)}%`);
        console.log(`    Throughput: ${scenario.metrics.avgThroughput.toFixed(2)} ops/sec\n`);
    });

    // Display learning statistics
    if (report.learningStats) {
        console.log('Learning Statistics:');
        console.log(`  Total Trajectories: ${report.learningStats.totalTrajectories}`);
        console.log(`  Patterns Discovered: ${report.patterns.length}`);
        console.log(`  Average Success: ${(report.learningStats.avgSuccessRate * 100).toFixed(2)}%`);
        console.log(`  Improvement Rate: ${(report.learningStats.improvementRate * 100).toFixed(2)}%`);
        console.log(`  Prediction Accuracy: ${(report.learningStats.predictionAccuracy * 100).toFixed(2)}%\n`);
    }

    // Display recommendations
    if (report.recommendations.length > 0) {
        console.log('Optimization Recommendations:\n');
        report.recommendations.forEach((rec, i) => {
            console.log(`  ${i + 1}. [${rec.priority.toUpperCase()}] ${rec.type}`);
            console.log(`     ${rec.message}\n`);
        });
    }

    // Display threshold status
    console.log('Threshold Status:');
    let allPassed = true;

    report.scenarios.forEach(scenario => {
        const violations = [];

        if (scenario.metrics.p95Latency > report.thresholds.maxLatencyMs) {
            violations.push('P95 latency');
            allPassed = false;
        }

        if (scenario.metrics.avgThroughput < report.thresholds.minThroughput) {
            violations.push('Throughput');
            allPassed = false;
        }

        if (scenario.metrics.avgSuccessRate < report.thresholds.minSuccessRate) {
            violations.push('Success rate');
            allPassed = false;
        }

        if (violations.length > 0) {
            console.log(`  ❌ ${scenario.name}: ${violations.join(', ')} violations`);
        } else {
            console.log(`  ✅ ${scenario.name}: All thresholds passed`);
        }
    });

    console.log('\n' + '='.repeat(60) + '\n');

    if (allPassed) {
        console.log('🎉 All scenarios passed performance thresholds!\n');
    } else {
        console.log('⚠️  Some scenarios violated performance thresholds.\n');
        console.log('Review recommendations above for optimization guidance.\n');
    }

    console.log(`📁 Detailed reports saved to: ${benchmark.config.outputDir}\n`);
    console.log('Files generated:');
    console.log(`  - benchmark-*.json (complete data)`);
    console.log(`  - benchmark-*.txt (human-readable)`);
    console.log(`  - benchmark-*.csv (spreadsheet format)`);
    console.log(`  - coordinator-*.txt (learning data)\n`);

    console.log('✅ Benchmark complete!\n');
}

// Run example
if (require.main === module) {
    main()
        .then(() => process.exit(0))
        .catch(error => {
            console.error('❌ Benchmark failed:', error);
            process.exit(1);
        });
}

module.exports = { main };
