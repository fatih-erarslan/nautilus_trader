/**
 * Quantum Knowledge System - Swift Bindings
 *
 * Drop-in super pill of wisdom for cognitive computing applications.
 *
 * This package provides Swift bindings to the QKS Rust core, enabling:
 * - Integrated Information Theory (IIT 3.0) - Φ computation
 * - Global Workspace Theory - Broadcast mechanisms
 * - Metacognitive monitoring - Real-time introspection
 * - Full cognitive cycles - Perception-cognition-action
 *
 * Example:
 * ```swift
 * let plugin = try QKSPlugin(config: QKSConfig(numQubits: 20, useGpu: true))
 * let phi = try plugin.consciousness.computePhi(networkState)
 * print("Consciousness level: \(String(format: "%.3f", phi))")
 * ```
 */

import Foundation

/// Package version
public let QKS_VERSION = "0.1.0"

/// Re-export main types
public typealias QKSPlugin = Plugin.QKSPlugin
public typealias QKSConfig = Plugin.QKSConfig

// Layer 6: Consciousness
public typealias Consciousness = ConsciousnessAPI.Consciousness
public typealias PhiResult = ConsciousnessAPI.PhiResult
public typealias PhiAlgorithm = ConsciousnessAPI.PhiAlgorithm
public typealias BroadcastResult = ConsciousnessAPI.BroadcastResult

// Layer 7: Metacognition
public typealias Metacognition = MetacognitionAPI.Metacognition
public typealias IntrospectionReport = MetacognitionAPI.IntrospectionReport
public typealias BeliefState = MetacognitionAPI.BeliefState
public typealias GoalState = MetacognitionAPI.GoalState
public typealias ConfidenceLevel = MetacognitionAPI.ConfidenceLevel
