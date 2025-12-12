/**
 * Quantum Knowledge System - TypeScript Bindings
 *
 * Drop-in super pill of wisdom for cognitive computing applications.
 *
 * This package provides TypeScript bindings to the QKS Rust core, enabling:
 * - Integrated Information Theory (IIT 3.0) - Φ computation
 * - Global Workspace Theory - Broadcast mechanisms
 * - Metacognitive monitoring - Real-time introspection
 * - Full cognitive cycles - Perception-cognition-action
 *
 * @example
 * ```typescript
 * import { QKSPlugin } from 'qks-plugin';
 *
 * const plugin = new QKSPlugin({ numQubits: 20, useGpu: true });
 * const phi = await plugin.consciousness.computePhi(networkState);
 * console.log(`Consciousness level: ${phi.toFixed(3)}`);
 * ```
 *
 * @packageDocumentation
 */

export { QKSPlugin, QKSConfig } from './plugin';
export {
    Consciousness,
    PhiResult,
    PhiAlgorithm,
    BroadcastResult
} from './consciousness';
export {
    Metacognition,
    IntrospectionReport,
    BeliefState,
    GoalState,
    ConfidenceLevel
} from './metacognition';
export {
    Integration,
    SensoryInput,
    CognitiveOutput,
    CognitivePhase,
    CycleStatistics
} from './integration';

/** Package version */
export const VERSION = '0.1.0';
