/**
 * Layer 7: Metacognition APIs - Introspection and Self-Monitoring.
 *
 * This module implements metacognitive capabilities - "thinking about thinking":
 * - Real-time introspection of cognitive state
 * - Performance monitoring and adjustment
 * - Belief and goal tracking
 * - Confidence estimation
 *
 * References:
 * Fleming, S. M., & Dolan, R. J. (2012).
 * The neural basis of metacognitive ability.
 * Philosophical Transactions of the Royal Society B, 367(1594), 1338-1349.
 *
 * Nelson, T. O., & Narens, L. (1990).
 * Metamemory: A theoretical framework and new findings.
 * Psychology of Learning and Motivation, 26, 125-173.
 */

import Foundation

public enum MetacognitionAPI {

    /// Metacognitive confidence levels
    public enum ConfidenceLevel: String {
        case veryLow = "very_low"     // < 20%
        case low = "low"               // 20-40%
        case moderate = "moderate"     // 40-60%
        case high = "high"             // 60-80%
        case veryHigh = "very_high"    // > 80%

        public init(confidence: Double) {
            if confidence < 0.2 { self = .veryLow }
            else if confidence < 0.4 { self = .low }
            else if confidence < 0.6 { self = .moderate }
            else if confidence < 0.8 { self = .high }
            else { self = .veryHigh }
        }
    }

    /// Represents a belief in the cognitive system
    public struct BeliefState {
        /// Belief content (proposition)
        public let content: String
        /// Confidence level (0.0-1.0)
        public var confidence: Double
        /// Supporting evidence
        public let evidence: [String]
        /// Creation timestamp
        public var timestamp: Date

        /// Get categorical confidence level
        public var confidenceLevel: ConfidenceLevel {
            return ConfidenceLevel(confidence: confidence)
        }

        public init(content: String, confidence: Double, evidence: [String] = []) {
            self.content = content
            self.confidence = confidence
            self.evidence = evidence
            self.timestamp = Date()
        }
    }

    /// Represents a goal in the cognitive system
    public struct GoalState {
        /// Goal description
        public let description: String
        /// Priority level (0.0-1.0)
        public let priority: Double
        /// Progress toward goal (0.0-1.0)
        public var progress: Double
        /// Whether goal is currently active
        public var active: Bool

        public init(description: String, priority: Double, progress: Double = 0.0) {
            self.description = description
            self.priority = priority
            self.progress = progress
            self.active = true
        }
    }

    /// Complete introspection report of cognitive state
    ///
    /// This report provides a snapshot of the system's current cognitive state,
    /// including beliefs, goals, capabilities, and overall confidence.
    public struct IntrospectionReport {
        /// Current belief states
        public let beliefs: [BeliefState]
        /// Active goals
        public let goals: [GoalState]
        /// Known capabilities
        public let capabilities: [String]
        /// Overall confidence level (0.0-1.0)
        public let confidence: Double
        /// Report generation timestamp
        public let timestampMs: UInt64
        /// Recent performance metrics
        public let performanceMetrics: [String: Double]

        /// Count active goals
        public var numActiveGoals: Int {
            return goals.filter { $0.active }.count
        }

        /// Get beliefs with high confidence (> 0.7)
        public var highConfidenceBeliefs: [BeliefState] {
            return beliefs.filter { $0.confidence > 0.7 }
        }
    }

    /// Layer 7: Metacognition APIs
    ///
    /// Provides tools for introspection and self-monitoring - the ability
    /// of the system to monitor and reason about its own cognitive processes.
    ///
    /// Key capabilities:
    /// - Real-time state introspection
    /// - Performance monitoring
    /// - Confidence estimation
    /// - Belief and goal management
    public class Metacognition {
        private let handle: QksHandle
        private var beliefStore: [BeliefState] = []
        private var goalStore: [GoalState] = []

        init(handle: QksHandle) {
            self.handle = handle
        }

        /// Perform real-time introspection of cognitive state
        ///
        /// Examines the current state of the cognitive system and generates
        /// a comprehensive report including beliefs, goals, capabilities,
        /// and confidence levels.
        ///
        /// - Returns: IntrospectionReport with complete cognitive state snapshot.
        /// - Throws: QKSError if introspection fails.
        ///
        /// Example:
        /// ```swift
        /// let report = try metacognition.introspect()
        /// print("Confidence: \(String(format: "%.1f%%", report.confidence * 100))")
        /// print("Active goals: \(report.numActiveGoals)")
        /// for belief in report.highConfidenceBeliefs {
        ///     print("- \(belief.content) (\(String(format: "%.2f", belief.confidence)))")
        /// }
        /// ```
        ///
        /// Scientific Background:
        /// Metacognition allows agents to monitor their own cognition,
        /// enabling self-correction, learning from mistakes, and
        /// explaining their reasoning. This is a key component of
        /// human-level intelligence.
        public func introspect() throws -> IntrospectionReport {
            // Get native introspection data
            var reportC = IntrospectionReportC(
                numBeliefs: 0,
                numGoals: 0,
                numCapabilities: 0,
                confidence: 0.0,
                timestampMs: 0
            )

            let result = qks_introspect(handle, &reportC)
            try checkError(result)

            // Calculate average belief confidence
            let avgBeliefConfidence = beliefStore.isEmpty
                ? 0.0
                : beliefStore.reduce(0.0) { $0 + $1.confidence } / Double(beliefStore.count)

            // Build report
            return IntrospectionReport(
                beliefs: beliefStore,
                goals: goalStore,
                capabilities: [
                    "consciousness_measurement",
                    "metacognitive_monitoring",
                    "cognitive_cycles",
                    "quantum_simulation",
                ],
                confidence: reportC.confidence,
                timestampMs: reportC.timestampMs,
                performanceMetrics: [
                    "numBeliefs": Double(beliefStore.count),
                    "numGoals": Double(goalStore.count),
                    "avgBeliefConfidence": avgBeliefConfidence,
                ]
            )
        }

        /// Add a new belief to the cognitive state
        ///
        /// - Parameters:
        ///   - content: Belief content (proposition).
        ///   - confidence: Confidence level (0.0-1.0).
        ///   - evidence: Supporting evidence items.
        /// - Returns: Created BeliefState.
        /// - Throws: QKSError if confidence not in [0, 1].
        ///
        /// Example:
        /// ```swift
        /// let belief = try metacognition.addBelief(
        ///     "The network exhibits consciousness",
        ///     confidence: 0.85,
        ///     evidence: ["Phi > 1.0", "Global integration present"]
        /// )
        /// ```
        public func addBelief(
            _ content: String,
            confidence: Double,
            evidence: [String] = []
        ) throws -> BeliefState {
            guard confidence >= 0 && confidence <= 1 else {
                throw QKSError.invalidInput("confidence must be in [0, 1], got \(confidence)")
            }

            let belief = BeliefState(content: content, confidence: confidence, evidence: evidence)
            beliefStore.append(belief)
            return belief
        }

        /// Update confidence of an existing belief
        ///
        /// - Parameters:
        ///   - content: Belief content to update.
        ///   - newConfidence: New confidence level (0.0-1.0).
        /// - Returns: Updated BeliefState if found, nil otherwise.
        @discardableResult
        public func updateBelief(_ content: String, newConfidence: Double) -> BeliefState? {
            guard let index = beliefStore.firstIndex(where: { $0.content == content }) else {
                return nil
            }

            beliefStore[index].confidence = newConfidence
            beliefStore[index].timestamp = Date()
            return beliefStore[index]
        }

        /// Add a new goal to the cognitive state
        ///
        /// - Parameters:
        ///   - description: Goal description.
        ///   - priority: Priority level (0.0-1.0).
        /// - Returns: Created GoalState.
        /// - Throws: QKSError if priority not in [0, 1].
        ///
        /// Example:
        /// ```swift
        /// let goal = try metacognition.addGoal(
        ///     "Optimize quantum circuit depth",
        ///     priority: 0.9
        /// )
        /// ```
        public func addGoal(_ description: String, priority: Double) throws -> GoalState {
            guard priority >= 0 && priority <= 1 else {
                throw QKSError.invalidInput("priority must be in [0, 1], got \(priority)")
            }

            let goal = GoalState(description: description, priority: priority)
            goalStore.append(goal)
            return goal
        }

        /// Update progress of an existing goal
        ///
        /// - Parameters:
        ///   - description: Goal description.
        ///   - progress: New progress value (0.0-1.0).
        /// - Returns: Updated GoalState if found, nil otherwise.
        @discardableResult
        public func updateGoalProgress(_ description: String, progress: Double) -> GoalState? {
            guard let index = goalStore.firstIndex(where: { $0.description == description }) else {
                return nil
            }

            goalStore[index].progress = progress
            if progress >= 1.0 {
                goalStore[index].active = false
            }
            return goalStore[index]
        }

        /// Monitor performance metrics
        ///
        /// Tracks performance over time and adjusts cognitive strategies
        /// based on observed performance.
        ///
        /// - Parameter metrics: Dictionary of metric_name -> value.
        /// - Throws: QKSError if monitoring fails.
        ///
        /// Example:
        /// ```swift
        /// try metacognition.monitorPerformance([
        ///     "accuracy": 0.92,
        ///     "latencyMs": 145,
        ///     "memoryMb": 512,
        /// ])
        /// ```
        public func monitorPerformance(_ metrics: [String: Double]) throws {
            let values = Array(metrics.values)
            let result = values.withUnsafeBufferPointer { ptr in
                qks_monitor_performance(handle, ptr.baseAddress!, values.count)
            }
            try checkError(result)
        }

        /// Estimate confidence for a given task
        ///
        /// Uses historical performance and current state to estimate
        /// confidence in successfully completing a task.
        ///
        /// - Parameter taskDescription: Description of the task.
        /// - Returns: Estimated confidence level (0.0-1.0).
        ///
        /// Example:
        /// ```swift
        /// let confidence = metacognition.estimateConfidence(
        ///     "Solve 20-qubit VQE problem"
        /// )
        /// print("Confidence: \(String(format: "%.1f%%", confidence * 100))")
        /// ```
        public func estimateConfidence(_ taskDescription: String) -> Double {
            // Placeholder - needs implementation with ML model
            return 0.5
        }

        /// Get beliefs with high confidence
        public func getHighConfidenceBeliefs(threshold: Double = 0.7) -> [BeliefState] {
            return beliefStore.filter { $0.confidence > threshold }
        }

        /// Get active goals
        public func getActiveGoals() -> [GoalState] {
            return goalStore.filter { $0.active }
        }
    }
}
