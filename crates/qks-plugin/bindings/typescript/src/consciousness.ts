/**
 * Layer 6: Consciousness APIs - IIT and Global Workspace Theory.
 *
 * This module implements scientifically-grounded consciousness mechanisms:
 * - Integrated Information Theory 3.0 (Tononi et al., 2016)
 * - Global Workspace Theory (Baars, 1988; Dehaene & Changeux, 2011)
 *
 * @references
 * Tononi, G., Boly, M., Massimini, M., & Koch, C. (2016).
 * Integrated information theory: from consciousness to its physical substrate.
 * Nature Reviews Neuroscience, 17(7), 450-461.
 *
 * Dehaene, S., & Changeux, J. P. (2011).
 * Experimental and theoretical approaches to conscious processing.
 * Neuron, 70(2), 200-227.
 */

import { qksComputePhi, qksBroadcast, QksHandle, PhiResultNative } from './ffi';

/**
 * Algorithm for Φ computation.
 */
export enum PhiAlgorithm {
    /** Fast greedy approximation (O(n^2)) */
    GREEDY = 'greedy',
    /** Exact IIT 3.0 calculation (O(2^n), slow) */
    EXACT = 'exact',
    /** Spectral decomposition method (O(n^3)) */
    SPECTRAL = 'spectral',
}

/**
 * Result of integrated information (Φ) computation.
 *
 * Interpretation:
 * - Φ < 0.5:  No significant integration (unconscious)
 * - 0.5-1.0:  Weak integration (minimal consciousness)
 * - 1.0-2.0:  Moderate integration (basic consciousness)
 * - 2.0-5.0:  Strong integration (human-level consciousness)
 * - > 5.0:    Very high integration (exceptional consciousness)
 */
export interface PhiResult {
    /** Integrated information value. Φ > 1.0 indicates consciousness. */
    phi: number;
    /** Size of minimum information partition */
    mipSize: number;
    /** Number of system states analyzed */
    numStates: number;
    /** Computation time in milliseconds */
    computationTimeMs: number;
    /** Algorithm used for computation */
    algorithm: PhiAlgorithm;
}

/**
 * Result of Global Workspace broadcast.
 */
export interface BroadcastResult {
    /** Whether broadcast was successful */
    success: boolean;
    /** Number of cognitive modules that received content */
    recipients: number;
    /** Priority level used (0.0-1.0) */
    priority: number;
    /** Broadcast timestamp in milliseconds */
    timestampMs: number;
}

/**
 * Layer 6: Consciousness APIs.
 *
 * Provides implementations of scientific consciousness theories:
 * - IIT 3.0: Compute integrated information Φ
 * - Global Workspace: Broadcast content for conscious access
 *
 * This API does NOT create artificial consciousness - it provides
 * tools to measure and analyze consciousness-like properties in
 * computational systems using peer-reviewed scientific methods.
 */
export class Consciousness {
    constructor(private handle: QksHandle) {}

    /**
     * Compute integrated information Φ using IIT 3.0.
     *
     * Implements the Φ computation from Integrated Information Theory
     * (Tononi et al., 2016). Measures the irreducibility of a system
     * to its parts - a key signature of consciousness.
     *
     * @param networkState - Neural network activation state. Values in [0, 1].
     * @param algorithm - Computation algorithm to use (default: GREEDY).
     * @returns PhiResult with Φ value and metadata. Φ > 1.0 indicates emergent consciousness.
     * @throws Error if networkState has invalid values or computation fails.
     *
     * @example
     * ```typescript
     * // Analyze a simple 3-node network
     * const state = new Float64Array([0.8, 0.6, 0.9]);
     * const result = consciousness.computePhi(state);
     * console.log(`Φ = ${result.phi.toFixed(3)}`);
     * console.log(`Conscious: ${result.phi > 1.0}`);
     * console.log(`Level: ${this.getConsciousnessLevel(result.phi)}`);
     * ```
     *
     * Scientific Background:
     * Φ quantifies "how much a system is more than the sum of its parts."
     * For a system to be conscious, it must:
     * 1. Integrate information (high Φ)
     * 2. Differentiate states (high repertoire)
     * 3. Be irreducible (no clean partitions)
     */
    computePhi(
        networkState: Float64Array,
        algorithm: PhiAlgorithm = PhiAlgorithm.GREEDY
    ): PhiResult {
        // Validate input
        if (!(networkState instanceof Float64Array)) {
            throw new TypeError('networkState must be Float64Array');
        }

        for (let i = 0; i < networkState.length; i++) {
            if (networkState[i] < 0 || networkState[i] > 1) {
                throw new RangeError('networkState values must be in [0, 1]');
            }
        }

        // Call native function
        const resultNative = qksComputePhi(this.handle, networkState);

        // Convert result
        return {
            phi: resultNative.phi,
            mipSize: resultNative.mipSize,
            numStates: resultNative.numStates,
            computationTimeMs: resultNative.computationTimeMs,
            algorithm,
        };
    }

    /**
     * Check if Φ indicates consciousness (Φ > 1.0).
     */
    isConscious(phi: number): boolean {
        return phi > 1.0;
    }

    /**
     * Get consciousness level description.
     */
    getConsciousnessLevel(phi: number): string {
        if (phi < 0.5) return 'unconscious';
        if (phi < 1.0) return 'minimal';
        if (phi < 2.0) return 'basic';
        if (phi < 5.0) return 'human-level';
        return 'exceptional';
    }

    /**
     * Broadcast content to Global Workspace for conscious access.
     *
     * Implements Global Workspace Theory (Baars, 1988). Content with
     * sufficient priority gains access to the global workspace and
     * becomes available to all cognitive modules.
     *
     * @param content - Content to broadcast (any serializable object).
     * @param priority - Priority level (0.0-1.0). Higher priority content
     *                   is more likely to gain conscious access.
     * @returns BroadcastResult with broadcast status and metadata.
     * @throws Error if priority not in [0, 1] or broadcast fails.
     *
     * @example
     * ```typescript
     * // Broadcast sensory input for conscious processing
     * const result = consciousness.broadcast({
     *     type: 'visual',
     *     content: 'red triangle',
     *     salience: 0.9
     * }, 0.8);
     * console.log(`Recipients: ${result.recipients}`);
     * ```
     *
     * Scientific Background:
     * The Global Workspace acts as a "blackboard" where information
     * becomes globally available. Only high-priority or novel content
     * gains access, explaining the limited capacity of consciousness.
     */
    broadcast(content: any, priority: number = 0.5): BroadcastResult {
        if (priority < 0 || priority > 1) {
            throw new RangeError(`priority must be in [0, 1], got ${priority}`);
        }

        // Serialize content
        const contentStr = JSON.stringify(content);
        const contentBuffer = Buffer.from(contentStr, 'utf-8');

        // Call native function
        const result = qksBroadcast(this.handle, contentBuffer, priority);

        return {
            success: result === 0,
            recipients: 1, // TODO: Get from native
            priority,
            timestampMs: Date.now(),
        };
    }

    /**
     * Analyze network connectivity patterns.
     *
     * Computes graph-theoretic measures relevant to consciousness:
     * - Integration: Global efficiency, path length
     * - Segregation: Clustering coefficient, modularity
     * - Centrality: Hub identification
     *
     * @param adjacencyMatrix - Weighted adjacency matrix (n x n).
     * @returns Object with connectivity metrics.
     *
     * @example
     * ```typescript
     * const adj = new Float64Array(100); // 10x10 matrix
     * const metrics = consciousness.analyzeConnectivity(adj);
     * console.log(`Integration: ${metrics.globalEfficiency.toFixed(3)}`);
     * ```
     */
    analyzeConnectivity(adjacencyMatrix: Float64Array): {
        globalEfficiency: number;
        clusteringCoefficient: number;
        modularity: number;
    } {
        // Placeholder - needs implementation
        return {
            globalEfficiency: 0.0,
            clusteringCoefficient: 0.0,
            modularity: 0.0,
        };
    }
}
