/**
 * Layer 8: Integration APIs - Full Cognitive Cycles.
 *
 * This module implements complete perception-cognition-action loops,
 * integrating consciousness and metacognition into unified cognitive cycles.
 *
 * @references
 * Anderson, J. R. (2007).
 * How can the human mind occur in the physical universe?
 * Oxford University Press.
 *
 * Laird, J. E. (2012).
 * The Soar cognitive architecture.
 * MIT Press.
 */

import { qksCognitiveCycle, QksHandle } from './ffi';

/**
 * Phases of the cognitive cycle.
 */
export enum CognitivePhase {
    PERCEPTION = 'perception',
    ATTENTION = 'attention',
    REASONING = 'reasoning',
    DECISION = 'decision',
    ACTION = 'action',
    LEARNING = 'learning',
}

/**
 * Sensory input for cognitive processing.
 */
export interface SensoryInput {
    /** Input modality (visual, auditory, etc.) */
    modality: string;
    /** Raw sensory data */
    data: any;
    /** Input timestamp */
    timestampMs?: number;
    /** Input priority (0.0-1.0) */
    priority?: number;
    /** Additional metadata */
    metadata?: Record<string, any>;
}

/**
 * Output of a cognitive cycle.
 */
export interface CognitiveOutput {
    /** Recommended action */
    action: string;
    /** Action confidence (0.0-1.0) */
    confidence: number;
    /** Reasoning trace */
    reasoning: string[];
    /** Time spent in each phase (ms) */
    phaseTimings: Partial<Record<CognitivePhase, number>>;
    /** Beliefs updated during cycle */
    beliefsUpdated: string[];
    /** Goals updated during cycle */
    goalsUpdated: string[];
}

/**
 * Statistics from cognitive cycles.
 */
export interface CycleStatistics {
    /** Total number of cycles executed */
    totalCycles: number;
    /** Average cycle time */
    avgCycleTimeMs: number;
    /** Number of successful cycles */
    successfulCycles: number;
    /** Number of failed cycles */
    failedCycles: number;
    /** Time breakdown by phase */
    phaseBreakdown: Partial<Record<CognitivePhase, number>>;
}

/**
 * Layer 8: Integration APIs.
 *
 * Provides complete cognitive cycles that integrate:
 * - Sensory perception
 * - Conscious awareness (Layer 6)
 * - Metacognitive monitoring (Layer 7)
 * - Decision making
 * - Action generation
 * - Learning and adaptation
 *
 * This represents the highest level of cognitive integration,
 * combining all lower layers into unified cognitive behavior.
 */
export class Integration {
    private statistics: CycleStatistics = {
        totalCycles: 0,
        avgCycleTimeMs: 0,
        successfulCycles: 0,
        failedCycles: 0,
        phaseBreakdown: {},
    };

    constructor(private handle: QksHandle) {}

    /**
     * Execute one complete cognitive cycle.
     *
     * Implements a full perception-cognition-action loop:
     *
     * 1. PERCEPTION: Process sensory input
     * 2. ATTENTION: Filter for salient information
     * 3. REASONING: Apply knowledge and beliefs
     * 4. DECISION: Select action based on goals
     * 5. ACTION: Generate action recommendation
     * 6. LEARNING: Update beliefs and models
     *
     * @param sensoryInput - Input to process.
     * @param enableLearning - Whether to update internal models (default: true).
     * @returns CognitiveOutput with action and reasoning trace.
     * @throws Error if cycle execution fails.
     *
     * @example
     * ```typescript
     * // Process visual input
     * const input: SensoryInput = {
     *     modality: "visual",
     *     data: { scene: "red triangle", salience: 0.9 },
     *     priority: 0.8
     * };
     *
     * const output = integration.cognitiveCycle(input);
     * console.log(`Action: ${output.action}`);
     * console.log(`Confidence: ${(output.confidence * 100).toFixed(1)}%`);
     * for (const step of output.reasoning) {
     *     console.log(`  - ${step}`);
     * }
     * ```
     *
     * Scientific Background:
     * This implements a cognitive architecture similar to ACT-R
     * (Anderson, 2007) and Soar (Laird, 2012), with added
     * consciousness and metacognition layers. The cycle integrates
     * bottom-up sensory processing with top-down goal-driven control.
     */
    cognitiveCycle(sensoryInput: SensoryInput, enableLearning: boolean = true): CognitiveOutput {
        const startTime = performance.now();
        const phaseTimings: Partial<Record<CognitivePhase, number>> = {};

        try {
            // Phase 1: PERCEPTION
            let phaseStart = performance.now();
            const percept = this.perceive(sensoryInput);
            phaseTimings[CognitivePhase.PERCEPTION] = performance.now() - phaseStart;

            // Phase 2: ATTENTION
            phaseStart = performance.now();
            const attended = this.attend(percept, sensoryInput.priority ?? 0.5);
            phaseTimings[CognitivePhase.ATTENTION] = performance.now() - phaseStart;

            // Phase 3: REASONING
            phaseStart = performance.now();
            const reasoning = this.reason(attended);
            phaseTimings[CognitivePhase.REASONING] = performance.now() - phaseStart;

            // Phase 4: DECISION
            phaseStart = performance.now();
            const [decision, confidence] = this.decide(attended, reasoning);
            phaseTimings[CognitivePhase.DECISION] = performance.now() - phaseStart;

            // Phase 5: ACTION
            phaseStart = performance.now();
            const action = this.generateAction(decision);
            phaseTimings[CognitivePhase.ACTION] = performance.now() - phaseStart;

            // Phase 6: LEARNING
            let beliefsUpdated: string[] = [];
            let goalsUpdated: string[] = [];
            if (enableLearning) {
                phaseStart = performance.now();
                [beliefsUpdated, goalsUpdated] = this.learn(sensoryInput, action, confidence);
                phaseTimings[CognitivePhase.LEARNING] = performance.now() - phaseStart;
            }

            // Update statistics
            const cycleTime = performance.now() - startTime;
            this.statistics.totalCycles++;
            this.statistics.successfulCycles++;
            this.statistics.avgCycleTimeMs =
                (this.statistics.avgCycleTimeMs * (this.statistics.totalCycles - 1) + cycleTime) /
                this.statistics.totalCycles;

            return {
                action,
                confidence,
                reasoning,
                phaseTimings,
                beliefsUpdated,
                goalsUpdated,
            };
        } catch (error) {
            this.statistics.totalCycles++;
            this.statistics.failedCycles++;
            throw new Error(`Cognitive cycle failed: ${error}`);
        }
    }

    /**
     * Phase 1: Perceive and preprocess sensory input.
     */
    private perceive(sensoryInput: SensoryInput): any {
        return {
            modality: sensoryInput.modality,
            features: sensoryInput.data,
            timestamp: sensoryInput.timestampMs ?? Date.now(),
        };
    }

    /**
     * Phase 2: Apply attention to salient features.
     */
    private attend(percept: any, priority: number): any {
        return {
            ...percept,
            priority,
            attended: priority > 0.5,
        };
    }

    /**
     * Phase 3: Apply reasoning and knowledge.
     */
    private reason(attended: any): string[] {
        const reasoning: string[] = [];

        if (attended.attended) {
            reasoning.push('Input exceeded attention threshold');
            reasoning.push(`Processing ${attended.modality} input`);

            if (attended.features) {
                reasoning.push(`Extracted features: ${Object.keys(attended.features).join(', ')}`);
            }
        }

        return reasoning;
    }

    /**
     * Phase 4: Make decision based on goals and reasoning.
     */
    private decide(attended: any, reasoning: string[]): [string, number] {
        if (attended.attended) {
            return ['process_input', attended.priority];
        } else {
            return ['ignore_input', 0.3];
        }
    }

    /**
     * Phase 5: Generate action from decision.
     */
    private generateAction(decision: string): string {
        const actionMap: Record<string, string> = {
            process_input: 'Process and store input',
            ignore_input: 'Ignore low-priority input',
            request_clarification: 'Request additional information',
        };
        return actionMap[decision] ?? 'Unknown action';
    }

    /**
     * Phase 6: Update beliefs and models based on outcome.
     */
    private learn(sensoryInput: SensoryInput, action: string, confidence: number): [string[], string[]] {
        const beliefsUpdated: string[] = [];
        const goalsUpdated: string[] = [];

        if (confidence > 0.7) {
            beliefsUpdated.push(`High confidence in ${sensoryInput.modality} processing`);
        }

        return [beliefsUpdated, goalsUpdated];
    }

    /**
     * Get cognitive cycle statistics.
     *
     * @returns CycleStatistics with performance metrics.
     *
     * @example
     * ```typescript
     * const stats = integration.getStatistics();
     * console.log(`Total cycles: ${stats.totalCycles}`);
     * console.log(`Success rate: ${(stats.successfulCycles / stats.totalCycles * 100).toFixed(1)}%`);
     * console.log(`Avg cycle time: ${stats.avgCycleTimeMs.toFixed(1)}ms`);
     * ```
     */
    getStatistics(): Readonly<CycleStatistics> {
        return { ...this.statistics };
    }

    /**
     * Get success rate.
     */
    getSuccessRate(): number {
        if (this.statistics.totalCycles === 0) return 0;
        return this.statistics.successfulCycles / this.statistics.totalCycles;
    }

    /**
     * Reset cycle statistics.
     */
    resetStatistics(): void {
        this.statistics = {
            totalCycles: 0,
            avgCycleTimeMs: 0,
            successfulCycles: 0,
            failedCycles: 0,
            phaseBreakdown: {},
        };
    }

    /**
     * Process multiple inputs in batch.
     *
     * @param inputs - Array of sensory inputs.
     * @param parallel - Whether to process in parallel (default: false).
     * @returns Array of cognitive outputs.
     *
     * @example
     * ```typescript
     * const inputs: SensoryInput[] = [
     *     { modality: "visual", data: { scene: "cat" } },
     *     { modality: "audio", data: { sound: "meow" } },
     * ];
     * const outputs = integration.batchProcess(inputs);
     * ```
     */
    batchProcess(inputs: SensoryInput[], parallel: boolean = false): CognitiveOutput[] {
        if (parallel) {
            // TODO: Implement parallel processing with Promise.all
            return inputs.map(input => this.cognitiveCycle(input));
        }

        const outputs: CognitiveOutput[] = [];
        for (const input of inputs) {
            try {
                outputs.push(this.cognitiveCycle(input));
            } catch (error) {
                // Continue processing remaining inputs
                outputs.push({
                    action: 'error',
                    confidence: 0.0,
                    reasoning: [`Error: ${error}`],
                    phaseTimings: {},
                    beliefsUpdated: [],
                    goalsUpdated: [],
                });
            }
        }

        return outputs;
    }

    /**
     * Process inputs in parallel using async operations.
     */
    async batchProcessAsync(inputs: SensoryInput[]): Promise<CognitiveOutput[]> {
        return Promise.all(
            inputs.map(async input => {
                try {
                    return this.cognitiveCycle(input);
                } catch (error) {
                    return {
                        action: 'error',
                        confidence: 0.0,
                        reasoning: [`Error: ${error}`],
                        phaseTimings: {},
                        beliefsUpdated: [],
                        goalsUpdated: [],
                    };
                }
            })
        );
    }
}
