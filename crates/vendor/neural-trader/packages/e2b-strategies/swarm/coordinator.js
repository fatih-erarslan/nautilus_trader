/**
 * E2B Swarm Coordinator with Agentic Jujutsu Integration
 *
 * Provides production-grade multi-agent coordination for trading strategies
 * with self-learning capabilities, quantum-resistant security, and E2B sandbox isolation.
 *
 * Features:
 * - Multi-agent coordination with zero conflicts
 * - Self-learning from successful strategies (ReasoningBank)
 * - E2B sandbox isolation for safe strategy execution
 * - Pattern recognition and intelligent suggestions
 * - Distributed execution with automatic load balancing
 * - Quantum-resistant security (SHA3-512 + HQC-128)
 * - Real-time performance monitoring
 *
 * @module swarm/coordinator
 */

'use strict';

const { JjWrapper } = require('agentic-jujutsu');
const { Sandbox } = require('@e2b/sdk');
const EventEmitter = require('events');

/**
 * Swarm Coordinator for distributed trading strategy execution
 */
class SwarmCoordinator extends EventEmitter {
    constructor(options = {}) {
        super();

        this.options = {
            maxAgents: options.maxAgents || 10,
            sandboxTimeout: options.sandboxTimeout || 300000, // 5 minutes
            learningEnabled: options.learningEnabled !== false,
            encryptionKey: options.encryptionKey || null,
            autoOptimize: options.autoOptimize !== false,
            ...options
        };

        // Initialize agentic-jujutsu for coordination
        this.jj = new JjWrapper();

        // Enable encryption if key provided
        if (this.options.encryptionKey) {
            this.jj.enableEncryption(this.options.encryptionKey);
        }

        // Agent tracking
        this.agents = new Map();
        this.sandboxes = new Map();
        this.activeTrajectories = new Map();

        // Performance metrics
        this.metrics = {
            totalExecutions: 0,
            successfulExecutions: 0,
            failedExecutions: 0,
            totalDuration: 0,
            averageDuration: 0,
            patternsDiscovered: 0
        };

        // Strategy registry
        this.strategies = new Map();

        console.log('🚀 SwarmCoordinator initialized');
        console.log(`   Max agents: ${this.options.maxAgents}`);
        console.log(`   Learning: ${this.options.learningEnabled ? 'enabled' : 'disabled'}`);
        console.log(`   Encryption: ${this.options.encryptionKey ? 'enabled' : 'disabled'}`);
    }

    /**
     * Register a trading strategy
     * @param {string} name - Strategy name
     * @param {Object} config - Strategy configuration
     */
    registerStrategy(name, config) {
        this.strategies.set(name, {
            name,
            config,
            executions: 0,
            successRate: 0,
            avgDuration: 0,
            registeredAt: new Date()
        });

        console.log(`📝 Registered strategy: ${name}`);
        return this;
    }

    /**
     * Deploy strategy to E2B sandbox with self-learning
     * @param {string} strategyName - Strategy to deploy
     * @param {Object} params - Execution parameters
     * @returns {Promise<Object>} Execution result
     */
    async deployStrategy(strategyName, params = {}) {
        const strategy = this.strategies.get(strategyName);
        if (!strategy) {
            throw new Error(`Strategy not found: ${strategyName}`);
        }

        const agentId = `agent-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

        console.log(`\n🤖 Deploying strategy: ${strategyName}`);
        console.log(`   Agent ID: ${agentId}`);
        console.log(`   Params:`, params);

        // Start learning trajectory if enabled
        let trajectoryId = null;
        if (this.options.learningEnabled) {
            trajectoryId = this.jj.startTrajectory(
                `Deploy ${strategyName} with params: ${JSON.stringify(params)}`
            );
            this.activeTrajectories.set(agentId, trajectoryId);
            console.log(`   📊 Started learning trajectory: ${trajectoryId}`);
        }

        // Get AI suggestion based on past executions
        if (this.options.learningEnabled) {
            const suggestion = this.getSuggestion(strategyName, params);
            if (suggestion.confidence > 0.7) {
                console.log(`   💡 AI Suggestion (${(suggestion.confidence * 100).toFixed(1)}% confidence):`);
                console.log(`      ${suggestion.reasoning}`);
                console.log(`      Expected success: ${(suggestion.expectedSuccessRate * 100).toFixed(1)}%`);
            }
        }

        // Create E2B sandbox
        let sandbox = null;
        let result = null;
        const startTime = Date.now();

        try {
            console.log(`   🔧 Creating E2B sandbox...`);
            sandbox = await Sandbox.create({
                timeoutMs: this.options.sandboxTimeout
            });

            this.sandboxes.set(agentId, sandbox);
            this.agents.set(agentId, {
                id: agentId,
                strategy: strategyName,
                sandbox,
                startTime,
                status: 'running'
            });

            console.log(`   ✅ Sandbox created: ${sandbox.sandboxId}`);

            // Execute strategy in sandbox
            result = await this.executeInSandbox(sandbox, strategy, params);

            const duration = Date.now() - startTime;

            // Update metrics
            this.metrics.totalExecutions++;
            this.metrics.totalDuration += duration;
            this.metrics.averageDuration = this.metrics.totalDuration / this.metrics.totalExecutions;

            if (result.success) {
                this.metrics.successfulExecutions++;
                console.log(`   ✅ Strategy executed successfully in ${duration}ms`);
            } else {
                this.metrics.failedExecutions++;
                console.log(`   ❌ Strategy execution failed in ${duration}ms`);
            }

            // Finalize learning trajectory
            if (this.options.learningEnabled) {
                this.jj.addToTrajectory();

                const successScore = result.success ? 0.9 : 0.4;
                const critique = result.success
                    ? `Successful execution in ${duration}ms. Profit: ${result.profit || 'N/A'}`
                    : `Failed: ${result.error}. Duration: ${duration}ms`;

                this.jj.finalizeTrajectory(successScore, critique);
                console.log(`   📊 Finalized trajectory with score: ${successScore}`);
            }

            // Update strategy stats
            strategy.executions++;
            strategy.successRate = this.metrics.successfulExecutions / this.metrics.totalExecutions;
            strategy.avgDuration = this.metrics.averageDuration;

            return {
                agentId,
                strategyName,
                success: result.success,
                duration,
                result: result.data,
                sandboxId: sandbox.sandboxId,
                metrics: this.getMetrics()
            };

        } catch (error) {
            const duration = Date.now() - startTime;

            this.metrics.totalExecutions++;
            this.metrics.failedExecutions++;
            this.metrics.totalDuration += duration;
            this.metrics.averageDuration = this.metrics.totalDuration / this.metrics.totalExecutions;

            console.error(`   ❌ Deployment failed:`, error.message);

            // Record failure in learning trajectory
            if (this.options.learningEnabled && trajectoryId) {
                this.jj.addToTrajectory();
                this.jj.finalizeTrajectory(0.2, `Failed: ${error.message}`);
            }

            throw error;

        } finally {
            // Cleanup
            if (sandbox) {
                await this.cleanupAgent(agentId);
            }
        }
    }

    /**
     * Execute strategy in E2B sandbox
     * @private
     */
    async executeInSandbox(sandbox, strategy, params) {
        try {
            // Install dependencies in sandbox
            await sandbox.commands.run('npm install express @alpacahq/alpaca-trade-api node-cache opossum');

            // Upload strategy code
            const strategyCode = this.generateStrategyCode(strategy, params);
            await sandbox.files.write('/strategy.js', strategyCode);

            // Execute strategy
            const execution = await sandbox.commands.run('node /strategy.js', {
                timeout: this.options.sandboxTimeout - 10000 // 10s buffer
            });

            // Parse results
            const output = execution.stdout + execution.stderr;

            return {
                success: execution.exitCode === 0,
                data: this.parseStrategyOutput(output),
                error: execution.exitCode !== 0 ? execution.stderr : null,
                exitCode: execution.exitCode
            };

        } catch (error) {
            return {
                success: false,
                data: null,
                error: error.message,
                exitCode: -1
            };
        }
    }

    /**
     * Generate strategy execution code
     * @private
     */
    generateStrategyCode(strategy, params) {
        return `
const { performance } = require('perf_hooks');

async function executeStrategy() {
    const startTime = performance.now();

    try {
        console.log('Strategy: ${strategy.name}');
        console.log('Config:', ${JSON.stringify(strategy.config, null, 2)});
        console.log('Params:', ${JSON.stringify(params, null, 2)});

        // Simulate strategy execution
        // In production, this would be actual strategy logic
        const result = {
            strategy: '${strategy.name}',
            profit: Math.random() * 1000,
            trades: Math.floor(Math.random() * 10),
            duration: performance.now() - startTime,
            timestamp: new Date().toISOString()
        };

        console.log('RESULT:', JSON.stringify(result));
        process.exit(0);

    } catch (error) {
        console.error('ERROR:', error.message);
        process.exit(1);
    }
}

executeStrategy();
`;
    }

    /**
     * Parse strategy output
     * @private
     */
    parseStrategyOutput(output) {
        try {
            const match = output.match(/RESULT: ({.*})/);
            if (match) {
                return JSON.parse(match[1]);
            }
            return null;
        } catch {
            return null;
        }
    }

    /**
     * Deploy multiple strategies concurrently (swarm execution)
     * @param {Array<Object>} deployments - Array of {strategyName, params}
     * @returns {Promise<Array<Object>>} Results from all deployments
     */
    async deploySwarm(deployments) {
        if (deployments.length > this.options.maxAgents) {
            throw new Error(`Too many agents requested. Max: ${this.options.maxAgents}`);
        }

        console.log(`\n🐝 Deploying swarm with ${deployments.length} agents...`);

        const swarmId = `swarm-${Date.now()}`;
        const results = await Promise.allSettled(
            deployments.map(({ strategyName, params }) =>
                this.deployStrategy(strategyName, params)
            )
        );

        const successful = results.filter(r => r.status === 'fulfilled').length;
        const failed = results.filter(r => r.status === 'rejected').length;

        console.log(`\n🐝 Swarm execution complete:`);
        console.log(`   ✅ Successful: ${successful}/${deployments.length}`);
        console.log(`   ❌ Failed: ${failed}/${deployments.length}`);

        return results.map((r, i) => ({
            deployment: deployments[i],
            result: r.status === 'fulfilled' ? r.value : null,
            error: r.status === 'rejected' ? r.reason.message : null,
            status: r.status
        }));
    }

    /**
     * Get AI suggestion for strategy deployment
     * @param {string} strategyName - Strategy name
     * @param {Object} params - Parameters
     * @returns {Object} AI suggestion
     */
    getSuggestion(strategyName, params) {
        if (!this.options.learningEnabled) {
            return { confidence: 0, reasoning: 'Learning disabled' };
        }

        const task = `Deploy ${strategyName} with params: ${JSON.stringify(params)}`;
        const suggestionJson = this.jj.getSuggestion(task);

        try {
            return JSON.parse(suggestionJson);
        } catch {
            return { confidence: 0, reasoning: 'No learning data available' };
        }
    }

    /**
     * Get discovered patterns
     * @returns {Array<Object>} Learned patterns
     */
    getPatterns() {
        if (!this.options.learningEnabled) {
            return [];
        }

        try {
            const patternsJson = this.jj.getPatterns();
            const patterns = JSON.parse(patternsJson);
            this.metrics.patternsDiscovered = patterns.length;
            return patterns;
        } catch {
            return [];
        }
    }

    /**
     * Get learning statistics
     * @returns {Object} Learning stats
     */
    getLearningStats() {
        if (!this.options.learningEnabled) {
            return null;
        }

        try {
            return JSON.parse(this.jj.getLearningStats());
        } catch {
            return null;
        }
    }

    /**
     * Get current metrics
     * @returns {Object} Performance metrics
     */
    getMetrics() {
        return {
            ...this.metrics,
            successRate: this.metrics.totalExecutions > 0
                ? this.metrics.successfulExecutions / this.metrics.totalExecutions
                : 0,
            activeAgents: this.agents.size,
            activeSandboxes: this.sandboxes.size,
            registeredStrategies: this.strategies.size,
            learningStats: this.getLearningStats(),
            patterns: this.getPatterns()
        };
    }

    /**
     * Cleanup agent and sandbox
     * @private
     */
    async cleanupAgent(agentId) {
        const agent = this.agents.get(agentId);
        if (!agent) return;

        try {
            if (agent.sandbox) {
                await agent.sandbox.kill();
                console.log(`   🧹 Cleaned up sandbox: ${agent.sandbox.sandboxId}`);
            }
        } catch (error) {
            console.error(`   ⚠️  Cleanup error:`, error.message);
        } finally {
            this.agents.delete(agentId);
            this.sandboxes.delete(agentId);
            this.activeTrajectories.delete(agentId);
        }
    }

    /**
     * Cleanup all agents and sandboxes
     */
    async cleanup() {
        console.log('\n🧹 Cleaning up all agents and sandboxes...');

        const agentIds = Array.from(this.agents.keys());
        await Promise.all(agentIds.map(id => this.cleanupAgent(id)));

        console.log('✅ Cleanup complete');
    }

    /**
     * Export coordinator state for persistence
     * @returns {Object} Serializable state
     */
    exportState() {
        return {
            metrics: this.metrics,
            strategies: Array.from(this.strategies.entries()).map(([name, data]) => ({
                name,
                ...data
            })),
            learningStats: this.getLearningStats(),
            patterns: this.getPatterns(),
            timestamp: new Date().toISOString()
        };
    }

    /**
     * Generate performance report
     * @returns {string} Formatted report
     */
    generateReport() {
        const metrics = this.getMetrics();
        const learningStats = this.getLearningStats();

        let report = '\n📊 Swarm Coordinator Performance Report\n';
        report += '='.repeat(50) + '\n\n';

        report += 'Execution Metrics:\n';
        report += `  Total Executions: ${metrics.totalExecutions}\n`;
        report += `  Successful: ${metrics.successfulExecutions} (${(metrics.successRate * 100).toFixed(1)}%)\n`;
        report += `  Failed: ${metrics.failedExecutions} (${((1 - metrics.successRate) * 100).toFixed(1)}%)\n`;
        report += `  Average Duration: ${metrics.averageDuration.toFixed(2)}ms\n`;
        report += `  Active Agents: ${metrics.activeAgents}\n`;
        report += `  Active Sandboxes: ${metrics.activeSandboxes}\n\n`;

        report += 'Strategy Registry:\n';
        for (const [name, strategy] of this.strategies) {
            report += `  ${name}:\n`;
            report += `    Executions: ${strategy.executions}\n`;
            report += `    Success Rate: ${(strategy.successRate * 100).toFixed(1)}%\n`;
            report += `    Avg Duration: ${strategy.avgDuration.toFixed(2)}ms\n`;
        }
        report += '\n';

        if (learningStats) {
            report += 'Learning Statistics:\n';
            report += `  Total Trajectories: ${learningStats.totalTrajectories}\n`;
            report += `  Patterns Discovered: ${metrics.patternsDiscovered}\n`;
            report += `  Average Success: ${(learningStats.avgSuccessRate * 100).toFixed(1)}%\n`;
            report += `  Improvement Rate: ${(learningStats.improvementRate * 100).toFixed(1)}%\n`;
            report += `  Prediction Accuracy: ${(learningStats.predictionAccuracy * 100).toFixed(1)}%\n\n`;
        }

        report += '='.repeat(50) + '\n';

        return report;
    }
}

module.exports = { SwarmCoordinator };
