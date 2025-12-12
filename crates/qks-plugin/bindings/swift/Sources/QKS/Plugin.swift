/**
 * Main QKS Plugin class - Entry point for all cognitive operations.
 */

import Foundation

public enum Plugin {

    /// Configuration for QKS plugin
    public struct QKSConfig {
        /// Number of qubits for quantum simulation (default: 10)
        public var numQubits: Int
        /// Enable GPU acceleration if available (default: true)
        public var useGpu: Bool
        /// Enable consciousness APIs (default: true)
        public var enableConsciousness: Bool
        /// Enable metacognition APIs (default: true)
        public var enableMetacognition: Bool

        public init(
            numQubits: Int = 10,
            useGpu: Bool = true,
            enableConsciousness: Bool = true,
            enableMetacognition: Bool = true
        ) {
            self.numQubits = numQubits
            self.useGpu = useGpu
            self.enableConsciousness = enableConsciousness
            self.enableMetacognition = enableMetacognition
        }
    }

    /// Quantum Knowledge System - Drop-in Cognitive Super Pill
    ///
    /// The main interface to QKS cognitive capabilities across all layers:
    /// - Layer 6: Consciousness (IIT, Global Workspace)
    /// - Layer 7: Metacognition (Introspection, Self-monitoring)
    /// - Layer 8: Integration (Full cognitive cycles)
    ///
    /// Example:
    /// ```swift
    /// let plugin = try QKSPlugin(config: QKSConfig(numQubits: 20, useGpu: true))
    ///
    /// // Layer 6: Consciousness
    /// let phi = try plugin.consciousness.computePhi(networkState)
    /// if phi > 1.0 {
    ///     print("System exhibits consciousness!")
    /// }
    ///
    /// // Layer 7: Metacognition
    /// let report = try plugin.metacognition.introspect()
    /// print("Confidence: \(String(format: "%.1f%%", report.confidence * 100))")
    ///
    /// // Layer 8: Integration
    /// let output = try plugin.integration.cognitiveCycle(sensoryInput)
    /// print("Action: \(output.action)")
    /// ```
    ///
    /// Architecture:
    /// The plugin wraps a Rust core library that provides:
    /// - GPU-accelerated quantum simulation (Metal on macOS)
    /// - IIT 3.0 Φ computation (Tononi et al.)
    /// - Global Workspace Theory implementation
    /// - Real-time metacognitive monitoring
    /// - Full perception-cognition-action loops
    public class QKSPlugin {
        private var handle: QksHandle?
        private let config: QKSConfig
        private var isDestroyed = false

        /// Layer 6: Consciousness APIs
        public private(set) lazy var consciousness: ConsciousnessAPI.Consciousness = {
            ConsciousnessAPI.Consciousness(handle: self.handle!)
        }()

        /// Layer 7: Metacognition APIs
        public private(set) lazy var metacognition: MetacognitionAPI.Metacognition = {
            MetacognitionAPI.Metacognition(handle: self.handle!)
        }()

        /// Initialize QKS plugin
        ///
        /// - Parameter config: Plugin configuration. Uses defaults if not provided.
        /// - Throws: QKSError if plugin initialization fails
        public init(config: QKSConfig = QKSConfig()) throws {
            self.config = config

            // Convert to C struct
            var configC = QksConfigC(
                numQubits: config.numQubits,
                useGpu: config.useGpu,
                enableConsciousness: config.enableConsciousness,
                enableMetacognition: config.enableMetacognition
            )

            // Create native handle
            guard let h = qks_create(&configC) else {
                throw QKSError.initializationFailed("Failed to create QKS plugin instance")
            }

            self.handle = h
        }

        deinit {
            destroy()
        }

        /// Get plugin configuration
        ///
        /// - Returns: Current plugin configuration
        public func getConfig() -> QKSConfig {
            return config
        }

        /// Get QKS plugin version
        ///
        /// - Returns: Version string (e.g., "0.1.0")
        public static func version() -> String {
            let versionPtr = qks_version()
            return String(cString: versionPtr)
        }

        /// Get plugin information and status
        ///
        /// - Returns: Dictionary with plugin metadata
        public func info() -> [String: Any] {
            checkNotDestroyed()

            return [
                "version": Self.version(),
                "config": [
                    "numQubits": config.numQubits,
                    "useGpu": config.useGpu,
                    "enableConsciousness": config.enableConsciousness,
                    "enableMetacognition": config.enableMetacognition,
                ],
                "capabilities": [
                    "consciousness": config.enableConsciousness,
                    "metacognition": config.enableMetacognition,
                    "quantumSimulation": true,
                    "gpuAcceleration": config.useGpu,
                ],
                "status": "initialized",
            ]
        }

        /// Cleanup native resources
        ///
        /// Call this when done with the plugin to free resources.
        /// This is also called automatically on deinit.
        public func destroy() {
            if !isDestroyed, let h = handle {
                qks_destroy(h)
                handle = nil
                isDestroyed = true
            }
        }

        /// Check if plugin has been destroyed
        private func checkNotDestroyed() {
            if isDestroyed {
                fatalError("QKS plugin has been destroyed")
            }
        }
    }
}
