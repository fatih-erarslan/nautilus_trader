/* Auto-generated TypeScript definitions for HyperPhysics Cognition System */

/** Initialize tracing (call once at startup) */
export function initTracing(): void

/** Cognition system configuration */
export interface CognitionConfig {
  enableAttention: boolean
  enableLoops: boolean
  enableDream: boolean
  enableLearning: boolean
  enableIntegration: boolean
  defaultCurvature: number
  loopFrequency: number
  dreamThreshold: number
}

/** Get default cognition configuration */
export function defaultConfig(): CognitionConfig

/** Cognition phase enum */
export const enum CognitionPhase {
  Perceiving = 0,
  Cognizing = 1,
  Deliberating = 2,
  Intending = 3,
  Integrating = 4,
  Acting = 5
}

/** Get next phase in loop */
export function nextPhase(phase: CognitionPhase): CognitionPhase

/** Get phase name */
export function phaseName(phase: CognitionPhase): string

/** Bio-Digital Isomorphic Cognition System */
export class CognitionSystem {
  /**
   * Create new cognition system
   * @param config Configuration object
   */
  constructor(config: CognitionConfig)

  /**
   * Get current arousal level (0.0 = deep sleep, 1.0 = maximal arousal)
   */
  getArousal(): number

  /**
   * Set arousal level
   * @param level Arousal level in [0.0, 1.0]
   */
  setArousal(level: number): void

  /**
   * Get current cognitive load (0.0 = minimal, 1.0 = maximal)
   */
  getLoad(): number

  /**
   * Set cognitive load
   * @param load Cognitive load in [0.0, 1.0]
   */
  setLoad(load: number): void

  /**
   * Check if system is healthy
   */
  isHealthy(): boolean

  /**
   * Get configuration
   */
  config(): CognitionConfig
}

/** Get version string */
export function version(): string
