/**
 * Main QKS Plugin class - Entry point for all cognitive operations.
 */

import {
    qksCreate,
    qksDestroy,
    qksVersion,
    QksHandle,
    QksConfigNative
} from './ffi';
import { Consciousness } from './consciousness';
import { Metacognition } from './metacognition';
import { Integration } from './integration';

/**
 * Configuration for QKS plugin.
 */
export interface QKSConfig {
    /** Number of qubits for quantum simulation (default: 10) */
    numQubits?: number;
    /** Enable GPU acceleration if available (default: true) */
    useGpu?: boolean;
    /** Enable consciousness APIs (default: true) */
    enableConsciousness?: boolean;
    /** Enable metacognition APIs (default: true) */
    enableMetacognition?: boolean;
}

/**
 * Quantum Knowledge System - Drop-in Cognitive Super Pill.
 *
 * The main interface to QKS cognitive capabilities across all layers:
 * - Layer 6: Consciousness (IIT, Global Workspace)
 * - Layer 7: Metacognition (Introspection, Self-monitoring)
 * - Layer 8: Integration (Full cognitive cycles)
 *
 * @example
 * ```typescript
 * const plugin = new QKSPlugin({ numQubits: 20, useGpu: true });
 *
 * // Layer 6: Consciousness
 * const phi = await plugin.consciousness.computePhi(networkState);
 * if (phi > 1.0) {
 *     console.log("System exhibits consciousness!");
 * }
 *
 * // Layer 7: Metacognition
 * const report = plugin.metacognition.introspect();
 * console.log(`Confidence: ${(report.confidence * 100).toFixed(1)}%`);
 *
 * // Layer 8: Integration
 * const output = plugin.integration.cognitiveCycle(sensoryInput);
 * console.log(`Action: ${output.action}`);
 * ```
 *
 * Architecture:
 * The plugin wraps a Rust core library that provides:
 * - GPU-accelerated quantum simulation (Metal on macOS)
 * - IIT 3.0 Φ computation (Tononi et al.)
 * - Global Workspace Theory implementation
 * - Real-time metacognitive monitoring
 * - Full perception-cognition-action loops
 */
export class QKSPlugin {
    private handle: QksHandle;
    private config: Required<QKSConfig>;
    private _consciousness: Consciousness;
    private _metacognition: Metacognition;
    private _integration: Integration;
    private destroyed = false;

    /**
     * Initialize QKS plugin.
     *
     * @param config - Plugin configuration. Uses defaults if not provided.
     * @throws Error if plugin initialization fails.
     */
    constructor(config: QKSConfig = {}) {
        // Apply defaults
        this.config = {
            numQubits: config.numQubits ?? 10,
            useGpu: config.useGpu ?? true,
            enableConsciousness: config.enableConsciousness ?? true,
            enableMetacognition: config.enableMetacognition ?? true,
        };

        // Convert to native config
        const nativeConfig: QksConfigNative = {
            numQubits: this.config.numQubits,
            useGpu: this.config.useGpu,
            enableConsciousness: this.config.enableConsciousness,
            enableMetacognition: this.config.enableMetacognition,
        };

        // Create native handle
        try {
            this.handle = qksCreate(nativeConfig);
        } catch (error) {
            throw new Error(`Failed to create QKS plugin: ${error}`);
        }

        // Initialize layer APIs
        this._consciousness = new Consciousness(this.handle);
        this._metacognition = new Metacognition(this.handle);
        this._integration = new Integration(this.handle);
    }

    /**
     * Access Layer 6: Consciousness APIs.
     *
     * @returns Consciousness API instance for IIT and Global Workspace operations.
     */
    get consciousness(): Consciousness {
        this.checkNotDestroyed();
        return this._consciousness;
    }

    /**
     * Access Layer 7: Metacognition APIs.
     *
     * @returns Metacognition API instance for introspection and self-monitoring.
     */
    get metacognition(): Metacognition {
        this.checkNotDestroyed();
        return this._metacognition;
    }

    /**
     * Access Layer 8: Integration APIs.
     *
     * @returns Integration API instance for full cognitive cycles.
     */
    get integration(): Integration {
        this.checkNotDestroyed();
        return this._integration;
    }

    /**
     * Get plugin configuration.
     *
     * @returns Current plugin configuration.
     */
    getConfig(): Readonly<Required<QKSConfig>> {
        return { ...this.config };
    }

    /**
     * Get QKS plugin version.
     *
     * @returns Version string (e.g., "0.1.0")
     */
    static version(): string {
        return qksVersion();
    }

    /**
     * Get plugin information and status.
     *
     * @returns Object with plugin metadata.
     */
    info(): {
        version: string;
        config: Required<QKSConfig>;
        capabilities: {
            consciousness: boolean;
            metacognition: boolean;
            quantumSimulation: boolean;
            gpuAcceleration: boolean;
        };
        status: string;
    } {
        this.checkNotDestroyed();

        return {
            version: QKSPlugin.version(),
            config: this.config,
            capabilities: {
                consciousness: this.config.enableConsciousness,
                metacognition: this.config.enableMetacognition,
                quantumSimulation: true,
                gpuAcceleration: this.config.useGpu,
            },
            status: 'initialized',
        };
    }

    /**
     * Cleanup native resources.
     * Call this when done with the plugin to free resources.
     */
    destroy(): void {
        if (!this.destroyed) {
            qksDestroy(this.handle);
            this.destroyed = true;
        }
    }

    /**
     * Check if plugin has been destroyed.
     * @throws Error if destroyed.
     */
    private checkNotDestroyed(): void {
        if (this.destroyed) {
            throw new Error('QKS plugin has been destroyed');
        }
    }

    /**
     * Automatic cleanup on garbage collection.
     * Note: Explicit destroy() is still recommended.
     */
    [Symbol.dispose](): void {
        this.destroy();
    }
}
