/**
 * Low-level FFI bindings to QKS Rust library using C interop.
 *
 * DO NOT use these functions directly - use the high-level APIs instead.
 */

import Foundation

// MARK: - C Function Declarations

/// Opaque handle to QKS instance
public typealias QksHandle = OpaquePointer

/// QKS configuration (C struct)
struct QksConfigC {
    var numQubits: size_t
    var useGpu: Bool
    var enableConsciousness: Bool
    var enableMetacognition: Bool
}

/// IIT Φ computation result (C struct)
struct PhiResultC {
    var phi: Double
    var mipSize: size_t
    var numStates: size_t
    var computationTimeMs: Double
}

/// Metacognitive introspection report (C struct)
struct IntrospectionReportC {
    var numBeliefs: size_t
    var numGoals: size_t
    var numCapabilities: size_t
    var confidence: Double
    var timestampMs: UInt64
}

// MARK: - Core Functions

@_silgen_name("qks_create")
func qks_create(_ config: UnsafePointer<QksConfigC>?) -> QksHandle?

@_silgen_name("qks_destroy")
func qks_destroy(_ handle: QksHandle)

@_silgen_name("qks_version")
func qks_version() -> UnsafePointer<CChar>

// MARK: - Layer 6: Consciousness

@_silgen_name("qks_compute_phi")
func qks_compute_phi(
    _ handle: QksHandle,
    _ networkState: UnsafePointer<Double>,
    _ stateSize: size_t,
    _ result: UnsafeMutablePointer<PhiResultC>
) -> Int32

@_silgen_name("qks_broadcast")
func qks_broadcast(
    _ handle: QksHandle,
    _ content: UnsafeRawPointer,
    _ contentSize: size_t,
    _ priority: Double
) -> Int32

// MARK: - Layer 7: Metacognition

@_silgen_name("qks_introspect")
func qks_introspect(
    _ handle: QksHandle,
    _ report: UnsafeMutablePointer<IntrospectionReportC>
) -> Int32

@_silgen_name("qks_monitor_performance")
func qks_monitor_performance(
    _ handle: QksHandle,
    _ metrics: UnsafePointer<Double>,
    _ numMetrics: size_t
) -> Int32

// MARK: - Layer 8: Integration

@_silgen_name("qks_cognitive_cycle")
func qks_cognitive_cycle(
    _ handle: QksHandle,
    _ sensoryInput: UnsafeRawPointer,
    _ inputSize: size_t,
    _ outputBuffer: UnsafeMutableRawPointer,
    _ outputSize: size_t
) -> Int32

// MARK: - Error Handling

@_silgen_name("qks_get_last_error")
func qks_get_last_error() -> UnsafePointer<CChar>?

/// QKS error type
public enum QKSError: Error, CustomStringConvertible {
    case initializationFailed(String)
    case computationFailed(String)
    case invalidInput(String)
    case nativeError(String)

    public var description: String {
        switch self {
        case .initializationFailed(let msg):
            return "Initialization failed: \(msg)"
        case .computationFailed(let msg):
            return "Computation failed: \(msg)"
        case .invalidInput(let msg):
            return "Invalid input: \(msg)"
        case .nativeError(let msg):
            return "Native error: \(msg)"
        }
    }
}

/// Check FFI result code and throw error if failed
func checkError(_ result: Int32) throws {
    if result != 0 {
        var errorMessage = "QKS error code: \(result)"
        if let errorPtr = qks_get_last_error() {
            errorMessage = String(cString: errorPtr)
        }
        throw QKSError.nativeError(errorMessage)
    }
}

// MARK: - Library Loading

/// Load the QKS shared library
func loadQKSLibrary() -> Bool {
    // The library should be linked at compile time or loaded via dlopen
    // For now, assume it's available through proper linking
    return true
}
