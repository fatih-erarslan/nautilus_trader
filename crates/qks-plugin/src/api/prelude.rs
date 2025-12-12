//! # Prelude - Convenient Imports
//!
//! Import everything you need for the QKS Cognitive API.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use qks_plugin::api::prelude::*;
//!
//! // Layer 1: Thermodynamic
//! set_temperature(ISING_CRITICAL_TEMP)?;
//!
//! // Layer 3: Decision
//! let action = select_action(&beliefs, &preferences)?;
//!
//! // Layer 6: Consciousness
//! let phi = compute_phi(&network)?;
//!
//! // Layer 8: Integration
//! let output = cognitive_cycle(&input)?;
//! ```

// ============================================================================
// Layer 1: Thermodynamic
// ============================================================================

pub use crate::api::thermodynamic::{
    boltzmann_sample, boltzmann_weight, critical_transition, criticality_measure, free_energy,
    get_energy, get_energy_state, get_phase, heat_capacity, is_critical, partition_function,
    set_field, set_temperature, EnergyState, Phase, BOLTZMANN_CONSTANT, DEFAULT_COUPLING,
    ISING_CRITICAL_TEMP,
};

// ============================================================================
// Layer 2: Cognitive
// ============================================================================

pub use crate::api::cognitive::{
    compute_salience, cosine_similarity, focus_attention, predict_next, prediction_error,
    recognize_pattern, retrieve_memory, store_memory, update_working_memory, AttentionState,
    MemoryItem, PatternMatch, WorkingMemory, ATTENTION_THRESHOLD, WORKING_MEMORY_CAPACITY,
};

// ============================================================================
// Layer 3: Decision (Active Inference)
// ============================================================================

pub use crate::api::decision::{
    compute_efe, infer_state, minimize_free_energy, select_action, variational_free_energy,
    Action, BeliefState, Policy, Preferences, MIN_PRECISION, POLICY_TEMPERATURE,
};

// ============================================================================
// Layer 4: Learning
// ============================================================================

pub use crate::api::learning::{
    apply_stdp, consolidate, eligibility_trace, meta_learn_rate, stdp_weight_change,
    transfer_knowledge, update_eligibility_traces, update_synapses, KnowledgeItem, ReasoningBank,
    SpikeEvent, Synapse, A_MINUS, A_PLUS, DEFAULT_LEARNING_RATE, FIBONACCI_TAU, STDP_WINDOW,
};

// ============================================================================
// Layer 5: Collective Intelligence
// ============================================================================

pub use crate::api::collective::{
    achieve_consensus, check_consensus, create_proposal, join_swarm, leave_swarm, receive_messages,
    send_message, swarm_cohesion, update_swarm_position, vote, Agent, AgentRole,
    ConsensusAlgorithm, ConsensusStatus, Proposal, SwarmState, Vote, BYZANTINE_THRESHOLD_RATIO,
    MIN_QUORUM_SIZE,
};

// ============================================================================
// Layer 6: Consciousness
// ============================================================================

pub use crate::api::consciousness::{
    access_workspace, broadcast_to_workspace, compete_for_workspace, compute_phi,
    consciousness_level, effective_information, integrate_information, is_conscious,
    GlobalWorkspace, NeuralState, Partition, PhiResult, WorkspaceContent, BROADCAST_TIMEOUT,
    MIN_CONSCIOUS_SIZE, PHI_THRESHOLD,
};

// ============================================================================
// Layer 7: Meta-Cognition
// ============================================================================

pub use crate::api::metacognition::{
    adapt_learning_strategy, assess_capability, calibrate_confidence, detect_uncertainty,
    explain_decision, get_confidence, get_self_model, introspect, is_well_calibrated, meta_learn,
    update_self_model, ConfidenceInterval, IntrospectionReport, LearningStrategy,
    PerformanceMetrics, ResourceUsage, SelfModel, CALIBRATION_TOLERANCE,
    HIGH_CONFIDENCE_THRESHOLD, META_LEARNING_RATE,
};

// ============================================================================
// Layer 8: Integration
// ============================================================================

pub use crate::api::integration::{
    cognitive_cycle, get_homeostasis, has_converged, orchestrate, regulate_homeostasis, restart,
    should_shutdown, system_health, CognitiveOutput, HealthReport, HomeostasisState,
    InternalState, SensoryInput, COGNITIVE_CYCLE_FREQ, DEFAULT_KD, DEFAULT_KI, DEFAULT_KP,
    MAX_ITERATIONS, N_HOMEOSTATIC_VARS,
};

// ============================================================================
// Common Types
// ============================================================================

pub use crate::{QksError, Result};

// ============================================================================
// Constants Overview
// ============================================================================

/// Scientific constants from all layers
pub mod constants {
    pub use super::ISING_CRITICAL_TEMP; // 2.269185
    pub use super::FIBONACCI_TAU; // 1.618034
    pub use super::PHI_THRESHOLD; // 1.0
    pub use super::WORKING_MEMORY_CAPACITY; // 7
    pub use super::N_HOMEOSTATIC_VARS; // 6
}

// ============================================================================
// Type Aliases for Convenience
// ============================================================================

/// Energy state (Layer 1)
pub type Energy = EnergyState;

/// Working memory (Layer 2)
pub type Memory = WorkingMemory;

/// Belief state (Layer 3)
pub type Beliefs = BeliefState;

/// Knowledge bank (Layer 4)
pub type Knowledge = ReasoningBank;

/// Swarm state (Layer 5)
pub type Swarm = SwarmState;

/// Consciousness measure (Layer 6)
pub type Phi = PhiResult;

/// Self-awareness (Layer 7)
pub type MetaState = SelfModel;

/// Homeostatic control (Layer 8)
pub type Homeostasis = HomeostasisState;
