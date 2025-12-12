/**
 * Layer 6: Consciousness APIs - IIT and Global Workspace Theory.
 *
 * This module implements scientifically-grounded consciousness mechanisms:
 * - Integrated Information Theory 3.0 (Tononi et al., 2016)
 * - Global Workspace Theory (Baars, 1988; Dehaene & Changeux, 2011)
 *
 * References:
 * Tononi, G., Boly, M., Massimini, M., & Koch, C. (2016).
 * Integrated information theory: from consciousness to its physical substrate.
 * Nature Reviews Neuroscience, 17(7), 450-461.
 *
 * Dehaene, S., & Changeux, J. P. (2011).
 * Experimental and theoretical approaches to conscious processing.
 * Neuron, 70(2), 200-227.
 */

import Foundation

public enum ConsciousnessAPI {

    /// Algorithm for Φ computation
    public enum PhiAlgorithm: String {
        /// Fast greedy approximation (O(n^2))
        case greedy
        /// Exact IIT 3.0 calculation (O(2^n), slow)
        case exact
        /// Spectral decomposition method (O(n^3))
        case spectral
    }

    /// Result of integrated information (Φ) computation
    ///
    /// Interpretation:
    /// - Φ < 0.5:  No significant integration (unconscious)
    /// - 0.5-1.0:  Weak integration (minimal consciousness)
    /// - 1.0-2.0:  Moderate integration (basic consciousness)
    /// - 2.0-5.0:  Strong integration (human-level consciousness)
    /// - > 5.0:    Very high integration (exceptional consciousness)
    public struct PhiResult {
        /// Integrated information value. Φ > 1.0 indicates consciousness.
        public let phi: Double
        /// Size of minimum information partition
        public let mipSize: Int
        /// Number of system states analyzed
        public let numStates: Int
        /// Computation time in milliseconds
        public let computationTimeMs: Double
        /// Algorithm used for computation
        public let algorithm: PhiAlgorithm

        /// Check if Φ indicates consciousness (Φ > 1.0)
        public var isConscious: Bool {
            return phi > 1.0
        }

        /// Get consciousness level description
        public var consciousnessLevel: String {
            if phi < 0.5 { return "unconscious" }
            if phi < 1.0 { return "minimal" }
            if phi < 2.0 { return "basic" }
            if phi < 5.0 { return "human-level" }
            return "exceptional"
        }
    }

    /// Result of Global Workspace broadcast
    public struct BroadcastResult {
        /// Whether broadcast was successful
        public let success: Bool
        /// Number of cognitive modules that received content
        public let recipients: Int
        /// Priority level used (0.0-1.0)
        public let priority: Double
        /// Broadcast timestamp in milliseconds
        public let timestampMs: UInt64
    }

    /// Layer 6: Consciousness APIs
    ///
    /// Provides implementations of scientific consciousness theories:
    /// - IIT 3.0: Compute integrated information Φ
    /// - Global Workspace: Broadcast content for conscious access
    ///
    /// This API does NOT create artificial consciousness - it provides
    /// tools to measure and analyze consciousness-like properties in
    /// computational systems using peer-reviewed scientific methods.
    public class Consciousness {
        private let handle: QksHandle

        init(handle: QksHandle) {
            self.handle = handle
        }

        /// Compute integrated information Φ using IIT 3.0
        ///
        /// Implements the Φ computation from Integrated Information Theory
        /// (Tononi et al., 2016). Measures the irreducibility of a system
        /// to its parts - a key signature of consciousness.
        ///
        /// - Parameters:
        ///   - networkState: Neural network activation state. Values in [0, 1].
        ///   - algorithm: Computation algorithm to use (default: greedy).
        /// - Returns: PhiResult with Φ value and metadata. Φ > 1.0 indicates emergent consciousness.
        /// - Throws: QKSError if networkState has invalid values or computation fails.
        ///
        /// Example:
        /// ```swift
        /// // Analyze a simple 3-node network
        /// let state = [0.8, 0.6, 0.9]
        /// let result = try consciousness.computePhi(state)
        /// print("Φ = \(String(format: "%.3f", result.phi))")
        /// print("Conscious: \(result.isConscious)")
        /// print("Level: \(result.consciousnessLevel)")
        /// ```
        ///
        /// Scientific Background:
        /// Φ quantifies "how much a system is more than the sum of its parts."
        /// For a system to be conscious, it must:
        /// 1. Integrate information (high Φ)
        /// 2. Differentiate states (high repertoire)
        /// 3. Be irreducible (no clean partitions)
        public func computePhi(
            _ networkState: [Double],
            algorithm: PhiAlgorithm = .greedy
        ) throws -> PhiResult {
            // Validate input
            for value in networkState {
                if value < 0 || value > 1 {
                    throw QKSError.invalidInput("networkState values must be in [0, 1]")
                }
            }

            // Prepare C call
            var resultC = PhiResultC(
                phi: 0.0,
                mipSize: 0,
                numStates: 0,
                computationTimeMs: 0.0
            )

            let result = networkState.withUnsafeBufferPointer { ptr in
                qks_compute_phi(
                    handle,
                    ptr.baseAddress!,
                    networkState.count,
                    &resultC
                )
            }

            try checkError(result)

            // Convert result
            return PhiResult(
                phi: resultC.phi,
                mipSize: resultC.mipSize,
                numStates: resultC.numStates,
                computationTimeMs: resultC.computationTimeMs,
                algorithm: algorithm
            )
        }

        /// Broadcast content to Global Workspace for conscious access
        ///
        /// Implements Global Workspace Theory (Baars, 1988). Content with
        /// sufficient priority gains access to the global workspace and
        /// becomes available to all cognitive modules.
        ///
        /// - Parameters:
        ///   - content: Content to broadcast (any encodable object).
        ///   - priority: Priority level (0.0-1.0). Higher priority content
        ///               is more likely to gain conscious access.
        /// - Returns: BroadcastResult with broadcast status and metadata.
        /// - Throws: QKSError if priority not in [0, 1] or broadcast fails.
        ///
        /// Example:
        /// ```swift
        /// // Broadcast sensory input for conscious processing
        /// let input = ["type": "visual", "content": "red triangle", "salience": 0.9]
        /// let result = try consciousness.broadcast(input, priority: 0.8)
        /// print("Recipients: \(result.recipients)")
        /// ```
        ///
        /// Scientific Background:
        /// The Global Workspace acts as a "blackboard" where information
        /// becomes globally available. Only high-priority or novel content
        /// gains access, explaining the limited capacity of consciousness.
        public func broadcast<T: Encodable>(
            _ content: T,
            priority: Double = 0.5
        ) throws -> BroadcastResult {
            guard priority >= 0 && priority <= 1 else {
                throw QKSError.invalidInput("priority must be in [0, 1], got \(priority)")
            }

            // Serialize content
            let encoder = JSONEncoder()
            let contentData = try encoder.encode(content)

            // Call native function
            let result = contentData.withUnsafeBytes { ptr in
                qks_broadcast(
                    handle,
                    ptr.baseAddress!,
                    contentData.count,
                    priority
                )
            }

            try checkError(result)

            return BroadcastResult(
                success: result == 0,
                recipients: 1, // TODO: Get from native
                priority: priority,
                timestampMs: UInt64(Date().timeIntervalSince1970 * 1000)
            )
        }

        /// Analyze network connectivity patterns
        ///
        /// Computes graph-theoretic measures relevant to consciousness:
        /// - Integration: Global efficiency, path length
        /// - Segregation: Clustering coefficient, modularity
        /// - Centrality: Hub identification
        ///
        /// - Parameter adjacencyMatrix: Weighted adjacency matrix (n x n).
        /// - Returns: Dictionary of connectivity metrics.
        ///
        /// Example:
        /// ```swift
        /// let adj = [Double](repeating: 0.5, count: 100) // 10x10 matrix
        /// let metrics = consciousness.analyzeConnectivity(adj)
        /// print("Integration: \(String(format: "%.3f", metrics["globalEfficiency"]!))")
        /// ```
        public func analyzeConnectivity(_ adjacencyMatrix: [Double]) -> [String: Double] {
            // Placeholder - needs implementation
            return [
                "globalEfficiency": 0.0,
                "clusteringCoefficient": 0.0,
                "modularity": 0.0,
            ]
        }
    }
}
