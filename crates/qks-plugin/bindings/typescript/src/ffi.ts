/**
 * Low-level FFI bindings to QKS Rust library using Node-API.
 *
 * DO NOT use these functions directly - use the high-level APIs instead.
 *
 * @internal
 */

import { createRequire } from 'module';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

// Determine library path
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let _native: any;

/**
 * Load the native QKS library.
 * Searches in order: release build, debug build, installed location.
 */
function loadNative(): any {
    if (_native) return _native;

    const searchPaths = [
        // Development paths (relative to bindings)
        join(__dirname, '../../../../target/release'),
        join(__dirname, '../../../../target/debug'),
        // Installed path (node_modules)
        join(__dirname, '../native'),
    ];

    let lastError: Error | null = null;

    for (const searchPath of searchPaths) {
        try {
            const require = createRequire(import.meta.url);
            _native = require(join(searchPath, 'qks.node'));
            return _native;
        } catch (e) {
            lastError = e as Error;
            continue;
        }
    }

    throw new Error(
        `QKS native library not found. Searched: ${searchPaths.join(', ')}\n` +
        `Last error: ${lastError?.message}\n` +
        'Please build the Rust library with: cargo build --release'
    );
}

// Initialize native module
_native = loadNative();

// ============================================================================
// Type Definitions
// ============================================================================

/** Opaque handle to QKS instance */
export type QksHandle = Buffer;

/** QKS configuration */
export interface QksConfigNative {
    numQubits: number;
    useGpu: boolean;
    enableConsciousness: boolean;
    enableMetacognition: boolean;
}

/** IIT Φ computation result (native) */
export interface PhiResultNative {
    phi: number;
    mipSize: number;
    numStates: number;
    computationTimeMs: number;
}

/** Metacognitive introspection report (native) */
export interface IntrospectionReportNative {
    numBeliefs: number;
    numGoals: number;
    numCapabilities: number;
    confidence: number;
    timestampMs: bigint;
}

// ============================================================================
// Core Functions
// ============================================================================

/**
 * Create QKS plugin instance.
 */
export function qksCreate(config: QksConfigNative): QksHandle {
    return _native.qks_create(config);
}

/**
 * Destroy QKS plugin instance.
 */
export function qksDestroy(handle: QksHandle): void {
    _native.qks_destroy(handle);
}

/**
 * Get QKS version string.
 */
export function qksVersion(): string {
    return _native.qks_version();
}

// ============================================================================
// Layer 6: Consciousness
// ============================================================================

/**
 * Compute integrated information Φ (IIT 3.0).
 */
export function qksComputePhi(
    handle: QksHandle,
    networkState: Float64Array
): PhiResultNative {
    return _native.qks_compute_phi(handle, networkState);
}

/**
 * Broadcast content to Global Workspace.
 */
export function qksBroadcast(
    handle: QksHandle,
    content: Buffer,
    priority: number
): number {
    return _native.qks_broadcast(handle, content, priority);
}

// ============================================================================
// Layer 7: Metacognition
// ============================================================================

/**
 * Perform introspection of cognitive state.
 */
export function qksIntrospect(handle: QksHandle): IntrospectionReportNative {
    return _native.qks_introspect(handle);
}

/**
 * Monitor performance metrics.
 */
export function qksMonitorPerformance(
    handle: QksHandle,
    metrics: Float64Array
): void {
    _native.qks_monitor_performance(handle, metrics);
}

// ============================================================================
// Layer 8: Integration
// ============================================================================

/**
 * Execute one cognitive cycle.
 */
export function qksCognitiveCycle(
    handle: QksHandle,
    sensoryInput: Buffer
): Buffer {
    return _native.qks_cognitive_cycle(handle, sensoryInput);
}

// ============================================================================
// Error Handling
// ============================================================================

/**
 * Get last error message from native library.
 */
export function qksGetLastError(): string | null {
    try {
        return _native.qks_get_last_error();
    } catch {
        return null;
    }
}

/**
 * Check result code and throw error if failed.
 */
export function checkError(result: number): void {
    if (result !== 0) {
        const error = qksGetLastError();
        throw new Error(error || `QKS error code: ${result}`);
    }
}
