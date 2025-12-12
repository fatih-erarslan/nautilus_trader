/**
 * QKS MCP Server - Type Definitions
 *
 * Comprehensive TypeScript types for the 8-layer cognitive architecture
 */

// =============================================================================
// Common Types
// =============================================================================

export interface QksResult<T = any> {
  success: boolean;
  data?: T;
  error?: string;
}

export interface Complex {
  real: number;
  imag: number;
}

export type Vector = number[];
export type Matrix = number[][];
export type ComplexVector = Complex[];

// =============================================================================
// Layer 1: Thermodynamic Types
// =============================================================================

export interface ThermodynamicState {
  energy: number;
  temperature: number;
  entropy: number;
  free_energy: number;
}

export interface CriticalPoint {
  temperature: number;
  pressure?: number;
  density?: number;
}

export interface OperationCost {
  operation: string;
  landauer_cost: number;
  efficiency_factor: number;
  total_cost: number;
}

// =============================================================================
// Layer 2: Cognitive Types
// =============================================================================

export interface AttentionState {
  focus_weights: Vector;
  attention_mask: boolean[];
  entropy: number;
}

export interface MemoryState {
  working_memory: any[];
  episodic_memory: any[];
  semantic_memory: Map<string, any>;
  consolidation_factor: number;
}

export interface PatternRecognition {
  pattern: Vector;
  similarity: number;
  confidence: number;
}

// =============================================================================
// Layer 3: Decision Making Types
// =============================================================================

export interface BeliefState {
  beliefs: Vector;
  precision: Vector;
  confidence: number;
}

export interface Policy {
  actions: string[];
  expected_free_energy: number;
  pragmatic_value: number;
  epistemic_value: number;
}

export interface ActiveInferenceState {
  beliefs: BeliefState;
  policies: Policy[];
  selected_policy: number;
}

// =============================================================================
// Layer 4: Learning Types
// =============================================================================

export interface LearningState {
  parameters: Vector;
  learning_rate: number;
  momentum: number;
  gradient_history: Vector[];
}

export interface TransferLearningMetrics {
  source_task_performance: number;
  target_task_performance: number;
  transfer_efficiency: number;
}

export interface StdpWeightChange {
  delta_w: number;
  delta_t: number;
  rule: string; // "hebbian" | "fibonacci" | "exponential"
}

// =============================================================================
// Layer 5: Collective Intelligence Types
// =============================================================================

export interface AgentInfo {
  id: string;
  role: string;
  capabilities: string[];
  state: any;
}

export interface SwarmState {
  agents: AgentInfo[];
  topology: string; // "star" | "mesh" | "hyperbolic"
  consensus_level: number;
}

export interface ConsensusProposal {
  id: string;
  proposer: string;
  content: any;
  votes_for: number;
  votes_against: number;
  status: "pending" | "approved" | "rejected";
}

// =============================================================================
// Layer 6: Consciousness Types
// =============================================================================

export interface ConsciousnessMetrics {
  phi: number; // Integrated Information (IIT 3.0)
  global_workspace_coherence: number;
  phase_synchrony: number;
  complexity: number;
}

export interface GlobalWorkspaceState {
  broadcast_content: any;
  attending_modules: string[];
  priority: number;
  duration: number;
}

export interface IntegrationState {
  partitions: number[][];
  min_information: number;
  phi_value: number;
}

// =============================================================================
// Layer 7: Metacognition Types
// =============================================================================

export interface SelfModel {
  beliefs_about_self: Map<string, any>;
  goals: string[];
  capabilities: string[];
  limitations: string[];
  confidence: number;
}

export interface MetaLearningState {
  learning_strategy: string;
  task_distribution: any[];
  adaptation_speed: number;
  meta_parameters: Vector;
}

export interface IntrospectionResult {
  internal_state: any;
  certainty: number;
  coherence: number;
  conflicts: string[];
}

export interface ConfidenceCalibration {
  predicted_confidence: number;
  actual_performance: number;
  calibration_error: number;
}

// =============================================================================
// Layer 8: Integration Types
// =============================================================================

export interface SystemHealthMetrics {
  layer1_health: number;
  layer2_health: number;
  layer3_health: number;
  layer4_health: number;
  layer5_health: number;
  layer6_health: number;
  layer7_health: number;
  layer8_health: number;
  overall_health: number;
}

export interface CognitiveLoopState {
  perception: any;
  inference: any;
  action: any;
  prediction_error: number;
  loop_latency_ms: number;
}

export interface HomeostaticState {
  energy_level: number;
  temperature: number;
  stress_level: number;
  arousal: number;
  in_homeostasis: boolean;
}

export interface EmergentFeatures {
  self_organization_level: number;
  criticality: number;
  adaptability: number;
  autonomy: number;
}

// =============================================================================
// Bridge Types (for Rust FFI)
// =============================================================================

export interface RustBridge {
  // Layer 1
  thermo_compute_energy(state: any): Promise<number>;
  thermo_compute_temperature(state: any): Promise<number>;
  thermo_compute_entropy(state: any): Promise<number>;
  thermo_critical_point(system: string): Promise<CriticalPoint>;

  // Layer 2
  cognitive_attention_focus(inputs: Vector, weights: Vector): Promise<AttentionState>;
  cognitive_memory_consolidate(working: any[], strength: number): Promise<any[]>;

  // Layer 3
  decision_compute_efe(policy: Policy, beliefs: BeliefState): Promise<number>;
  decision_select_policy(policies: Policy[]): Promise<number>;

  // Layer 4
  learning_stdp_update(pre_time: number, post_time: number): Promise<number>;
  learning_consolidate_memory(episodes: any[]): Promise<any[]>;

  // Layer 5
  collective_swarm_coordinate(agents: AgentInfo[]): Promise<SwarmState>;
  collective_reach_consensus(proposal: ConsensusProposal): Promise<boolean>;

  // Layer 6
  consciousness_compute_phi(network: Matrix): Promise<number>;
  consciousness_broadcast_workspace(content: any): Promise<GlobalWorkspaceState>;

  // Layer 7
  meta_introspect(): Promise<IntrospectionResult>;
  meta_update_self_model(observations: any[]): Promise<SelfModel>;

  // Layer 8
  integration_system_health(): Promise<SystemHealthMetrics>;
  integration_cognitive_loop_step(input: any): Promise<CognitiveLoopState>;
}

// =============================================================================
// MCP Tool Context
// =============================================================================

export interface ToolContext {
  rustBridge?: RustBridge;
  executePython?: (code: string) => Promise<any>;
  config?: Record<string, any>;
}
