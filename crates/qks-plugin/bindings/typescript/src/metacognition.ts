/**
 * Layer 7: Metacognition APIs - Introspection and Self-Monitoring.
 *
 * This module implements metacognitive capabilities - "thinking about thinking":
 * - Real-time introspection of cognitive state
 * - Performance monitoring and adjustment
 * - Belief and goal tracking
 * - Confidence estimation
 *
 * @references
 * Fleming, S. M., & Dolan, R. J. (2012).
 * The neural basis of metacognitive ability.
 * Philosophical Transactions of the Royal Society B, 367(1594), 1338-1349.
 *
 * Nelson, T. O., & Narens, L. (1990).
 * Metamemory: A theoretical framework and new findings.
 * Psychology of Learning and Motivation, 26, 125-173.
 */

import { qksIntrospect, qksMonitorPerformance, QksHandle, IntrospectionReportNative } from './ffi';

/**
 * Metacognitive confidence levels.
 */
export enum ConfidenceLevel {
    VERY_LOW = 'very_low',   // < 20%
    LOW = 'low',             // 20-40%
    MODERATE = 'moderate',   // 40-60%
    HIGH = 'high',           // 60-80%
    VERY_HIGH = 'very_high', // > 80%
}

/**
 * Represents a belief in the cognitive system.
 */
export interface BeliefState {
    /** Belief content (proposition) */
    content: string;
    /** Confidence level (0.0-1.0) */
    confidence: number;
    /** Supporting evidence */
    evidence: string[];
    /** Creation timestamp */
    timestamp: number;
}

/**
 * Represents a goal in the cognitive system.
 */
export interface GoalState {
    /** Goal description */
    description: string;
    /** Priority level (0.0-1.0) */
    priority: number;
    /** Progress toward goal (0.0-1.0) */
    progress: number;
    /** Whether goal is currently active */
    active: boolean;
}

/**
 * Complete introspection report of cognitive state.
 *
 * This report provides a snapshot of the system's current cognitive state,
 * including beliefs, goals, capabilities, and overall confidence.
 */
export interface IntrospectionReport {
    /** Current belief states */
    beliefs: BeliefState[];
    /** Active goals */
    goals: GoalState[];
    /** Known capabilities */
    capabilities: string[];
    /** Overall confidence level (0.0-1.0) */
    confidence: number;
    /** Report generation timestamp */
    timestampMs: bigint;
    /** Recent performance metrics */
    performanceMetrics: Record<string, number>;
}

/**
 * Layer 7: Metacognition APIs.
 *
 * Provides tools for introspection and self-monitoring - the ability
 * of the system to monitor and reason about its own cognitive processes.
 *
 * Key capabilities:
 * - Real-time state introspection
 * - Performance monitoring
 * - Confidence estimation
 * - Belief and goal management
 */
export class Metacognition {
    private beliefStore: BeliefState[] = [];
    private goalStore: GoalState[] = [];

    constructor(private handle: QksHandle) {}

    /**
     * Perform real-time introspection of cognitive state.
     *
     * Examines the current state of the cognitive system and generates
     * a comprehensive report including beliefs, goals, capabilities,
     * and confidence levels.
     *
     * @returns IntrospectionReport with complete cognitive state snapshot.
     * @throws Error if introspection fails.
     *
     * @example
     * ```typescript
     * const report = metacognition.introspect();
     * console.log(`Confidence: ${(report.confidence * 100).toFixed(1)}%`);
     * console.log(`Active goals: ${report.goals.filter(g => g.active).length}`);
     * for (const belief of report.beliefs.filter(b => b.confidence > 0.7)) {
     *     console.log(`- ${belief.content} (${belief.confidence.toFixed(2)})`);
     * }
     * ```
     *
     * Scientific Background:
     * Metacognition allows agents to monitor their own cognition,
     * enabling self-correction, learning from mistakes, and
     * explaining their reasoning. This is a key component of
     * human-level intelligence.
     */
    introspect(): IntrospectionReport {
        // Get native introspection data
        const reportNative = qksIntrospect(this.handle);

        // Calculate average belief confidence
        const avgBeliefConfidence = this.beliefStore.length > 0
            ? this.beliefStore.reduce((sum, b) => sum + b.confidence, 0) / this.beliefStore.length
            : 0;

        // Build report
        return {
            beliefs: [...this.beliefStore],
            goals: [...this.goalStore],
            capabilities: [
                'consciousness_measurement',
                'metacognitive_monitoring',
                'cognitive_cycles',
                'quantum_simulation',
            ],
            confidence: reportNative.confidence,
            timestampMs: reportNative.timestampMs,
            performanceMetrics: {
                numBeliefs: this.beliefStore.length,
                numGoals: this.goalStore.length,
                avgBeliefConfidence,
            },
        };
    }

    /**
     * Add a new belief to the cognitive state.
     *
     * @param content - Belief content (proposition).
     * @param confidence - Confidence level (0.0-1.0).
     * @param evidence - Supporting evidence items.
     * @returns Created BeliefState.
     * @throws Error if confidence not in [0, 1].
     *
     * @example
     * ```typescript
     * const belief = metacognition.addBelief(
     *     "The network exhibits consciousness",
     *     0.85,
     *     ["Phi > 1.0", "Global integration present"]
     * );
     * ```
     */
    addBelief(content: string, confidence: number, evidence: string[] = []): BeliefState {
        if (confidence < 0 || confidence > 1) {
            throw new RangeError(`confidence must be in [0, 1], got ${confidence}`);
        }

        const belief: BeliefState = {
            content,
            confidence,
            evidence,
            timestamp: Date.now(),
        };

        this.beliefStore.push(belief);
        return belief;
    }

    /**
     * Update confidence of an existing belief.
     *
     * @param content - Belief content to update.
     * @param newConfidence - New confidence level (0.0-1.0).
     * @returns Updated BeliefState if found, undefined otherwise.
     */
    updateBelief(content: string, newConfidence: number): BeliefState | undefined {
        const belief = this.beliefStore.find(b => b.content === content);
        if (belief) {
            belief.confidence = newConfidence;
            belief.timestamp = Date.now();
        }
        return belief;
    }

    /**
     * Get confidence level category.
     */
    getConfidenceLevel(confidence: number): ConfidenceLevel {
        if (confidence < 0.2) return ConfidenceLevel.VERY_LOW;
        if (confidence < 0.4) return ConfidenceLevel.LOW;
        if (confidence < 0.6) return ConfidenceLevel.MODERATE;
        if (confidence < 0.8) return ConfidenceLevel.HIGH;
        return ConfidenceLevel.VERY_HIGH;
    }

    /**
     * Add a new goal to the cognitive state.
     *
     * @param description - Goal description.
     * @param priority - Priority level (0.0-1.0).
     * @returns Created GoalState.
     *
     * @example
     * ```typescript
     * const goal = metacognition.addGoal(
     *     "Optimize quantum circuit depth",
     *     0.9
     * );
     * ```
     */
    addGoal(description: string, priority: number): GoalState {
        if (priority < 0 || priority > 1) {
            throw new RangeError(`priority must be in [0, 1], got ${priority}`);
        }

        const goal: GoalState = {
            description,
            priority,
            progress: 0.0,
            active: true,
        };

        this.goalStore.push(goal);
        return goal;
    }

    /**
     * Update progress of an existing goal.
     *
     * @param description - Goal description.
     * @param progress - New progress value (0.0-1.0).
     * @returns Updated GoalState if found, undefined otherwise.
     */
    updateGoalProgress(description: string, progress: number): GoalState | undefined {
        const goal = this.goalStore.find(g => g.description === description);
        if (goal) {
            goal.progress = progress;
            if (progress >= 1.0) {
                goal.active = false;
            }
        }
        return goal;
    }

    /**
     * Monitor performance metrics.
     *
     * Tracks performance over time and adjusts cognitive strategies
     * based on observed performance.
     *
     * @param metrics - Object with metric_name -> value mappings.
     *
     * @example
     * ```typescript
     * metacognition.monitorPerformance({
     *     accuracy: 0.92,
     *     latencyMs: 145,
     *     memoryMb: 512,
     * });
     * ```
     */
    monitorPerformance(metrics: Record<string, number>): void {
        const values = Object.values(metrics);
        const metricsArray = new Float64Array(values);

        qksMonitorPerformance(this.handle, metricsArray);
    }

    /**
     * Estimate confidence for a given task.
     *
     * Uses historical performance and current state to estimate
     * confidence in successfully completing a task.
     *
     * @param taskDescription - Description of the task.
     * @returns Estimated confidence level (0.0-1.0).
     *
     * @example
     * ```typescript
     * const confidence = metacognition.estimateConfidence(
     *     "Solve 20-qubit VQE problem"
     * );
     * console.log(`Confidence: ${(confidence * 100).toFixed(1)}%`);
     * ```
     */
    estimateConfidence(taskDescription: string): number {
        // Placeholder - needs implementation with ML model
        return 0.5;
    }

    /**
     * Get beliefs with high confidence (> threshold).
     */
    getHighConfidenceBeliefs(threshold: number = 0.7): BeliefState[] {
        return this.beliefStore.filter(b => b.confidence > threshold);
    }

    /**
     * Get active goals.
     */
    getActiveGoals(): GoalState[] {
        return this.goalStore.filter(g => g.active);
    }
}
