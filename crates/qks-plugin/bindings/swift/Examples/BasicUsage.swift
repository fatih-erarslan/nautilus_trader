/**
 * QKS Plugin - Swift Basic Usage Example
 *
 * Demonstrates all three cognitive layers:
 * - Layer 6: Consciousness (IIT + Global Workspace)
 * - Layer 7: Metacognition (Introspection)
 * - Layer 8: Integration (Cognitive Cycles)
 */

import Foundation
import QKS

func main() throws {
    print("=== QKS Plugin - Swift Example ===\n")

    // Initialize plugin with GPU acceleration
    let config = QKSConfig(
        numQubits: 20,
        useGpu: true,
        enableConsciousness: true,
        enableMetacognition: true
    )

    let plugin = try QKSPlugin(config: config)
    defer { plugin.destroy() }

    print("QKS Version: \(QKSPlugin.version())")
    print("Configuration: \(plugin.info())\n")

    // ====================================================================
    // LAYER 6: CONSCIOUSNESS
    // ====================================================================
    print("=== Layer 6: Consciousness ===")

    // IIT 3.0: Compute integrated information Φ
    let networkState = [0.8, 0.6, 0.9, 0.7, 0.5]
    print("Network state: \(networkState)")

    let phiResult = try plugin.consciousness.computePhi(networkState)
    print("\nΦ = \(String(format: "%.3f", phiResult.phi))")
    print("Conscious: \(phiResult.isConscious)")
    print("Level: \(phiResult.consciousnessLevel)")
    print("MIP size: \(phiResult.mipSize)")
    print("Computation time: \(String(format: "%.2f", phiResult.computationTimeMs))ms")

    // Global Workspace: Broadcast content
    let broadcastContent = [
        "type": "visual",
        "scene": "red triangle",
        "salience": 0.9
    ] as [String: Any]

    let broadcastResult = try plugin.consciousness.broadcast(broadcastContent, priority: 0.8)
    print("\nBroadcast successful: \(broadcastResult.success)")
    print("Recipients: \(broadcastResult.recipients)")

    // ====================================================================
    // LAYER 7: METACOGNITION
    // ====================================================================
    print("\n=== Layer 7: Metacognition ===")

    // Add beliefs
    _ = try plugin.metacognition.addBelief(
        "The network exhibits consciousness",
        confidence: 0.85,
        evidence: ["Phi > 1.0", "Global integration present"]
    )

    _ = try plugin.metacognition.addBelief(
        "Visual processing is active",
        confidence: 0.92,
        evidence: ["High salience visual input"]
    )

    // Add goals
    _ = try plugin.metacognition.addGoal("Optimize quantum circuit depth", priority: 0.9)
    _ = try plugin.metacognition.addGoal("Maintain metacognitive accuracy", priority: 0.7)

    // Introspection
    let report = try plugin.metacognition.introspect()
    print("\nIntrospection Report:")
    print("  Confidence: \(String(format: "%.1f%%", report.confidence * 100))")
    print("  Beliefs: \(report.beliefs.count)")
    print("  Goals: \(report.goals.count) (\(report.numActiveGoals) active)")
    print("  Capabilities: \(report.capabilities.count)")

    print("\nHigh-confidence beliefs:")
    for belief in report.highConfidenceBeliefs {
        print("  - \(belief.content) (\(String(format: "%.2f", belief.confidence)))")
    }

    // Monitor performance
    try plugin.metacognition.monitorPerformance([
        "accuracy": 0.92,
        "latencyMs": 145,
        "memoryMb": 512,
    ])

    // ====================================================================
    // LAYER 8: INTEGRATION (Placeholder - needs implementation)
    // ====================================================================
    print("\n=== Layer 8: Integration ===")
    print("(Integration layer not yet implemented in Swift bindings)")

    // Statistics would go here when implemented

    print("\n=== Example Complete ===")
}

// Run the example
do {
    try main()
} catch {
    print("Error: \(error)")
    exit(1)
}
