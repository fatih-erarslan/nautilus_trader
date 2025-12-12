/**
 * QKS Plugin - TypeScript Basic Usage Example
 *
 * Demonstrates all three cognitive layers:
 * - Layer 6: Consciousness (IIT + Global Workspace)
 * - Layer 7: Metacognition (Introspection)
 * - Layer 8: Integration (Cognitive Cycles)
 */

import { QKSPlugin, SensoryInput } from '../src';

async function main() {
    console.log('=== QKS Plugin - TypeScript Example ===\n');

    // Initialize plugin with GPU acceleration
    const plugin = new QKSPlugin({
        numQubits: 20,
        useGpu: true,
        enableConsciousness: true,
        enableMetacognition: true,
    });

    try {
        console.log(`QKS Version: ${QKSPlugin.version()}`);
        console.log(`Configuration:`, plugin.info(), '\n');

        // ================================================================
        // LAYER 6: CONSCIOUSNESS
        // ================================================================
        console.log('=== Layer 6: Consciousness ===');

        // IIT 3.0: Compute integrated information Φ
        const networkState = new Float64Array([0.8, 0.6, 0.9, 0.7, 0.5]);
        console.log('Network state:', Array.from(networkState));

        const phiResult = plugin.consciousness.computePhi(networkState);
        console.log(`\nΦ = ${phiResult.phi.toFixed(3)}`);
        console.log(`Conscious: ${phiResult.phi > 1.0}`);
        console.log(`Level: ${plugin.consciousness.getConsciousnessLevel(phiResult.phi)}`);
        console.log(`MIP size: ${phiResult.mipSize}`);
        console.log(`Computation time: ${phiResult.computationTimeMs.toFixed(2)}ms`);

        // Global Workspace: Broadcast content
        const broadcastResult = plugin.consciousness.broadcast({
            type: 'visual',
            scene: 'red triangle',
            salience: 0.9
        }, 0.8);
        console.log(`\nBroadcast successful: ${broadcastResult.success}`);
        console.log(`Recipients: ${broadcastResult.recipients}`);

        // ================================================================
        // LAYER 7: METACOGNITION
        // ================================================================
        console.log('\n=== Layer 7: Metacognition ===');

        // Add beliefs
        plugin.metacognition.addBelief(
            'The network exhibits consciousness',
            0.85,
            ['Phi > 1.0', 'Global integration present']
        );

        plugin.metacognition.addBelief(
            'Visual processing is active',
            0.92,
            ['High salience visual input']
        );

        // Add goals
        plugin.metacognition.addGoal('Optimize quantum circuit depth', 0.9);
        plugin.metacognition.addGoal('Maintain metacognitive accuracy', 0.7);

        // Introspection
        const report = plugin.metacognition.introspect();
        console.log('\nIntrospection Report:');
        console.log(`  Confidence: ${(report.confidence * 100).toFixed(1)}%`);
        console.log(`  Beliefs: ${report.beliefs.length}`);
        console.log(`  Goals: ${report.goals.length} (${report.goals.filter(g => g.active).length} active)`);
        console.log(`  Capabilities: ${report.capabilities.length}`);

        console.log('\nHigh-confidence beliefs:');
        const highConfBeliefs = plugin.metacognition.getHighConfidenceBeliefs(0.7);
        for (const belief of highConfBeliefs) {
            console.log(`  - ${belief.content} (${belief.confidence.toFixed(2)})`);
        }

        // Monitor performance
        plugin.metacognition.monitorPerformance({
            accuracy: 0.92,
            latencyMs: 145,
            memoryMb: 512,
        });

        // ================================================================
        // LAYER 8: INTEGRATION
        // ================================================================
        console.log('\n=== Layer 8: Integration ===');

        // Execute cognitive cycle
        const sensoryInput: SensoryInput = {
            modality: 'visual',
            data: {
                scene: 'moving red triangle',
                velocity: 2.5,
                salience: 0.9
            },
            priority: 0.8,
        };

        const output = plugin.integration.cognitiveCycle(sensoryInput);
        console.log('\nCognitive Cycle Output:');
        console.log(`  Action: ${output.action}`);
        console.log(`  Confidence: ${(output.confidence * 100).toFixed(1)}%`);
        console.log('\nReasoning trace:');
        for (const step of output.reasoning) {
            console.log(`  - ${step}`);
        }

        console.log('\nPhase timings (ms):');
        for (const [phase, timeMs] of Object.entries(output.phaseTimings)) {
            console.log(`  ${phase}: ${timeMs.toFixed(2)}ms`);
        }

        // Batch processing
        console.log('\n=== Batch Processing ===');
        const batchInputs: SensoryInput[] = [
            { modality: 'visual', data: { scene: 'cat' }, priority: 0.6 },
            { modality: 'audio', data: { sound: 'meow' }, priority: 0.7 },
            { modality: 'visual', data: { scene: 'dog' }, priority: 0.5 },
        ];

        const batchOutputs = plugin.integration.batchProcess(batchInputs);
        console.log(`Processed ${batchOutputs.length} inputs:`);
        batchOutputs.forEach((out, i) => {
            console.log(`  ${i + 1}. ${out.action} (confidence: ${out.confidence.toFixed(2)})`);
        });

        // Statistics
        const stats = plugin.integration.getStatistics();
        console.log('\nCognitive Cycle Statistics:');
        console.log(`  Total cycles: ${stats.totalCycles}`);
        console.log(`  Success rate: ${(plugin.integration.getSuccessRate() * 100).toFixed(1)}%`);
        console.log(`  Avg cycle time: ${stats.avgCycleTimeMs.toFixed(1)}ms`);

    } finally {
        // Cleanup
        plugin.destroy();
    }

    console.log('\n=== Example Complete ===');
}

main().catch(console.error);
