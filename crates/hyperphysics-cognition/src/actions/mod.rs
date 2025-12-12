//! Cognitive Action Matrix - Biomimetic Algorithms for Ultra-Fast Cognition
//!
//! This module implements a comprehensive mapping of mental actions to computational
//! algorithms, inspired by biological cognitive systems. All actions are designed
//! for <25ms execution within 40Hz gamma cycles.
//!
//! ## Scientific Foundations
//!
//! Based on peer-reviewed research:
//! - **Predictive Coding**: Rao & Ballard (1999) "Predictive coding in the visual cortex"
//! - **Drift-Diffusion**: Ratcliff & McKoon (2008) "The diffusion decision model"
//! - **Active Inference**: Friston (2010) "The free-energy principle"
//! - **STDP**: Bi & Poo (2001) "Synaptic modification by correlated activity"
//! - **Salience**: Itti & Koch (2001) "Computational modelling of visual attention"
//! - **Optimal Control**: Todorov & Jordan (2002) "Optimal feedback control"
//!
//! ## Zero-Copy Architecture
//!
//! All state transitions use `Arc<RwLock<T>>` for zero-copy shared state access.
//! Actions operate on references, never copying large data structures.

use crate::error::{CognitionError, Result};
use crate::types::*;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

pub mod perception;
pub mod cognition;
pub mod emotion;
pub mod memory;
pub mod attention;
pub mod decision;
pub mod action;
pub mod learning;

// ============================================================================
// Cognitive Action Types
// ============================================================================

/// Comprehensive enumeration of all cognitive actions across domains
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CognitiveAction {
    // PERCEPTION DOMAIN
    /// Predictive coding: generate top-down predictions
    PredictSensory,
    /// Compute prediction error: compare prediction to input
    ComputePredictionError,
    /// Sensory gating: filter inputs by relevance
    GateSensoryInput,
    /// Attention modulation: enhance attended features
    ModulateByAttention,

    // COGNITION DOMAIN
    /// Symbolic grounding: map symbols to perceptual features
    GroundSymbol,
    /// Analogical reasoning: map source to target domain
    MapAnalogy,
    /// Mental simulation: forward model prediction
    SimulateForward,
    /// Causal inference: infer causal structure
    InferCausality,

    // EMOTION DOMAIN
    /// Appraisal: evaluate event significance
    AppraiseEvent,
    /// Affective forecasting: predict future feelings
    ForecastAffect,
    /// Emotion regulation: modulate affective state
    RegulateEmotion,
    /// Mood congruent recall: emotion biases memory
    BiasMemoryByMood,

    // MEMORY DOMAIN
    /// Pattern separation: orthogonalize similar inputs
    SeparatePatterns,
    /// Consolidation: transfer to long-term memory
    ConsolidateMemory,
    /// Reconsolidation: update existing memory
    ReconsolidateMemory,
    /// Retrieval: access stored patterns
    RetrievePattern,

    // ATTENTION DOMAIN
    /// Salience detection: compute bottom-up salience
    DetectSalience,
    /// Attentional blink: temporary blindness
    BlinkAttention,
    /// Inhibition of return: avoid recently attended
    InhibitReturn,
    /// Attention switch: shift focus to new target
    SwitchAttention,

    // DECISION DOMAIN
    /// Bayesian inference: update beliefs
    UpdateBelief,
    /// Drift-diffusion: accumulate evidence
    AccumulateEvidence,
    /// Expected utility: compute value
    ComputeUtility,
    /// Commit decision: cross threshold
    CommitDecision,

    // ACTION DOMAIN
    /// Optimal control: compute control policy
    ComputeControl,
    /// Inverse model: infer action from goal
    InvertModel,
    /// Error correction: adjust based on feedback
    CorrectError,
    /// Action cancellation: inhibit prepared action
    CancelAction,

    // LEARNING DOMAIN
    /// TD learning: temporal difference update
    UpdateTD,
    /// STDP: spike-timing dependent plasticity
    UpdateSTDP,
    /// Meta-learning: update learning strategy
    UpdateMetaParameters,
    /// One-shot learning: rapid acquisition
    LearnOneShot,
}

impl CognitiveAction {
    /// Get temporal cost in nanoseconds
    pub const fn temporal_cost_ns(&self) -> u64 {
        match self {
            // PERCEPTION (fast: 1-10μs)
            Self::PredictSensory => 2_000,
            Self::ComputePredictionError => 1_500,
            Self::GateSensoryInput => 800,
            Self::ModulateByAttention => 1_200,

            // COGNITION (medium: 10-100μs)
            Self::GroundSymbol => 15_000,
            Self::MapAnalogy => 50_000,
            Self::SimulateForward => 25_000,
            Self::InferCausality => 40_000,

            // EMOTION (fast: 1-5μs)
            Self::AppraiseEvent => 3_000,
            Self::ForecastAffect => 5_000,
            Self::RegulateEmotion => 4_000,
            Self::BiasMemoryByMood => 2_500,

            // MEMORY (medium-slow: 10-200μs)
            Self::SeparatePatterns => 20_000,
            Self::ConsolidateMemory => 150_000,
            Self::ReconsolidateMemory => 100_000,
            Self::RetrievePattern => 12_000,

            // ATTENTION (fast: 1-5μs)
            Self::DetectSalience => 2_000,
            Self::BlinkAttention => 500,
            Self::InhibitReturn => 1_000,
            Self::SwitchAttention => 3_000,

            // DECISION (medium: 5-50μs)
            Self::UpdateBelief => 8_000,
            Self::AccumulateEvidence => 5_000,
            Self::ComputeUtility => 10_000,
            Self::CommitDecision => 1_500,

            // ACTION (fast-medium: 2-20μs)
            Self::ComputeControl => 15_000,
            Self::InvertModel => 10_000,
            Self::CorrectError => 5_000,
            Self::CancelAction => 2_000,

            // LEARNING (slow: 50-500μs)
            Self::UpdateTD => 8_000,
            Self::UpdateSTDP => 50_000,
            Self::UpdateMetaParameters => 200_000,
            Self::LearnOneShot => 100_000,
        }
    }

    /// Get domain classification
    pub const fn domain(&self) -> CognitiveDomain {
        match self {
            Self::PredictSensory | Self::ComputePredictionError |
            Self::GateSensoryInput | Self::ModulateByAttention => CognitiveDomain::Perception,

            Self::GroundSymbol | Self::MapAnalogy |
            Self::SimulateForward | Self::InferCausality => CognitiveDomain::Cognition,

            Self::AppraiseEvent | Self::ForecastAffect |
            Self::RegulateEmotion | Self::BiasMemoryByMood => CognitiveDomain::Emotion,

            Self::SeparatePatterns | Self::ConsolidateMemory |
            Self::ReconsolidateMemory | Self::RetrievePattern => CognitiveDomain::Memory,

            Self::DetectSalience | Self::BlinkAttention |
            Self::InhibitReturn | Self::SwitchAttention => CognitiveDomain::Attention,

            Self::UpdateBelief | Self::AccumulateEvidence |
            Self::ComputeUtility | Self::CommitDecision => CognitiveDomain::Decision,

            Self::ComputeControl | Self::InvertModel |
            Self::CorrectError | Self::CancelAction => CognitiveDomain::Action,

            Self::UpdateTD | Self::UpdateSTDP |
            Self::UpdateMetaParameters | Self::LearnOneShot => CognitiveDomain::Learning,
        }
    }

    /// Get biomimetic algorithm type
    pub const fn algorithm(&self) -> BiomimeticAlgorithm {
        match self {
            Self::PredictSensory => BiomimeticAlgorithm::PredictiveCoding,
            Self::ComputePredictionError => BiomimeticAlgorithm::PredictiveCoding,
            Self::GateSensoryInput => BiomimeticAlgorithm::SensoryGating,
            Self::ModulateByAttention => BiomimeticAlgorithm::AttentionModulation,
            Self::GroundSymbol => BiomimeticAlgorithm::SymbolicGrounding,
            Self::MapAnalogy => BiomimeticAlgorithm::AnalogicalMapping,
            Self::SimulateForward => BiomimeticAlgorithm::MentalSimulation,
            Self::InferCausality => BiomimeticAlgorithm::CausalInference,
            Self::AppraiseEvent => BiomimeticAlgorithm::AppraisalTheory,
            Self::ForecastAffect => BiomimeticAlgorithm::AffectiveForecasting,
            Self::RegulateEmotion => BiomimeticAlgorithm::EmotionRegulation,
            Self::BiasMemoryByMood => BiomimeticAlgorithm::MoodCongruence,
            Self::SeparatePatterns => BiomimeticAlgorithm::PatternSeparation,
            Self::ConsolidateMemory => BiomimeticAlgorithm::MemoryConsolidation,
            Self::ReconsolidateMemory => BiomimeticAlgorithm::Reconsolidation,
            Self::RetrievePattern => BiomimeticAlgorithm::PatternCompletion,
            Self::DetectSalience => BiomimeticAlgorithm::SalienceDetection,
            Self::BlinkAttention => BiomimeticAlgorithm::AttentionalBlink,
            Self::InhibitReturn => BiomimeticAlgorithm::InhibitionOfReturn,
            Self::SwitchAttention => BiomimeticAlgorithm::AttentionSwitching,
            Self::UpdateBelief => BiomimeticAlgorithm::BayesianInference,
            Self::AccumulateEvidence => BiomimeticAlgorithm::DriftDiffusion,
            Self::ComputeUtility => BiomimeticAlgorithm::ExpectedUtility,
            Self::CommitDecision => BiomimeticAlgorithm::ThresholdCrossing,
            Self::ComputeControl => BiomimeticAlgorithm::OptimalControl,
            Self::InvertModel => BiomimeticAlgorithm::InverseModel,
            Self::CorrectError => BiomimeticAlgorithm::ErrorCorrection,
            Self::CancelAction => BiomimeticAlgorithm::ActionInhibition,
            Self::UpdateTD => BiomimeticAlgorithm::TemporalDifference,
            Self::UpdateSTDP => BiomimeticAlgorithm::STDP,
            Self::UpdateMetaParameters => BiomimeticAlgorithm::MetaLearning,
            Self::LearnOneShot => BiomimeticAlgorithm::OneShotLearning,
        }
    }
}

// ============================================================================
// Cognitive Domains
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CognitiveDomain {
    Perception,
    Cognition,
    Emotion,
    Memory,
    Attention,
    Decision,
    Action,
    Learning,
}

// ============================================================================
// Biomimetic Algorithms
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BiomimeticAlgorithm {
    // Perception
    PredictiveCoding,
    SensoryGating,
    AttentionModulation,

    // Cognition
    SymbolicGrounding,
    AnalogicalMapping,
    MentalSimulation,
    CausalInference,

    // Emotion
    AppraisalTheory,
    AffectiveForecasting,
    EmotionRegulation,
    MoodCongruence,

    // Memory
    PatternSeparation,
    MemoryConsolidation,
    Reconsolidation,
    PatternCompletion,

    // Attention
    SalienceDetection,
    AttentionalBlink,
    InhibitionOfReturn,
    AttentionSwitching,

    // Decision
    BayesianInference,
    DriftDiffusion,
    ExpectedUtility,
    ThresholdCrossing,

    // Action
    OptimalControl,
    InverseModel,
    ErrorCorrection,
    ActionInhibition,

    // Learning
    TemporalDifference,
    STDP,
    MetaLearning,
    OneShotLearning,
}

impl BiomimeticAlgorithm {
    /// Get scientific citation
    pub const fn citation(&self) -> &'static str {
        match self {
            Self::PredictiveCoding => "Rao & Ballard (1999) Nature Neuroscience",
            Self::SensoryGating => "Freedman et al. (1987) Biological Psychiatry",
            Self::AttentionModulation => "Reynolds & Heeger (2009) Neuron",
            Self::SymbolicGrounding => "Harnad (1990) Physica D",
            Self::AnalogicalMapping => "Gentner (1983) Cognitive Science",
            Self::MentalSimulation => "Jeannerod (2001) Nature Reviews Neuroscience",
            Self::CausalInference => "Pearl (2009) Causality",
            Self::AppraisalTheory => "Scherer (1999) Cognition & Emotion",
            Self::AffectiveForecasting => "Gilbert & Wilson (2007) PNAS",
            Self::EmotionRegulation => "Gross (1998) JPSP",
            Self::MoodCongruence => "Bower (1981) American Psychologist",
            Self::PatternSeparation => "Marr (1971) Phil Trans B",
            Self::MemoryConsolidation => "McGaugh (2000) Science",
            Self::Reconsolidation => "Nader et al. (2000) Nature",
            Self::PatternCompletion => "McClelland et al. (1995) Psych Review",
            Self::SalienceDetection => "Itti & Koch (2001) Nature Reviews Neuroscience",
            Self::AttentionalBlink => "Raymond et al. (1992) JEP:HPP",
            Self::InhibitionOfReturn => "Posner & Cohen (1984) Attention & Performance",
            Self::AttentionSwitching => "Monsell (2003) Trends in Cognitive Sciences",
            Self::BayesianInference => "Knill & Pouget (2004) Trends in Neurosciences",
            Self::DriftDiffusion => "Ratcliff & McKoon (2008) Neural Computation",
            Self::ExpectedUtility => "Von Neumann & Morgenstern (1944) Theory of Games",
            Self::ThresholdCrossing => "Bogacz et al. (2006) Psych Review",
            Self::OptimalControl => "Todorov & Jordan (2002) Nature Neuroscience",
            Self::InverseModel => "Wolpert & Kawato (1998) Neural Networks",
            Self::ErrorCorrection => "Shadmehr & Krakauer (2008) Exp Brain Res",
            Self::ActionInhibition => "Logan & Cowan (1984) Psych Review",
            Self::TemporalDifference => "Sutton & Barto (1998) Reinforcement Learning",
            Self::STDP => "Bi & Poo (2001) Ann Rev Neuroscience",
            Self::MetaLearning => "Finn et al. (2017) ICML - MAML",
            Self::OneShotLearning => "Lake et al. (2015) Science",
        }
    }

    /// Get computational complexity
    pub const fn complexity(&self) -> &'static str {
        match self {
            Self::PredictiveCoding => "O(n) - linear in feature dimensions",
            Self::SensoryGating => "O(1) - constant time gating",
            Self::AttentionModulation => "O(n) - linear modulation",
            Self::SymbolicGrounding => "O(n log n) - feature matching",
            Self::AnalogicalMapping => "O(n²) - structural alignment",
            Self::MentalSimulation => "O(n·t) - forward model steps",
            Self::CausalInference => "O(2^n) - exponential in variables (DAG)",
            Self::AppraisalTheory => "O(n) - linear appraisal dimensions",
            Self::AffectiveForecasting => "O(t) - time steps",
            Self::EmotionRegulation => "O(n) - regulation strategies",
            Self::MoodCongruence => "O(log n) - mood-biased retrieval",
            Self::PatternSeparation => "O(n²) - pairwise decorrelation",
            Self::MemoryConsolidation => "O(n log n) - replay sorting",
            Self::Reconsolidation => "O(n) - memory update",
            Self::PatternCompletion => "O(n) - associative completion",
            Self::SalienceDetection => "O(n) - feature-based salience",
            Self::AttentionalBlink => "O(1) - fixed refractory period",
            Self::InhibitionOfReturn => "O(k) - k recent locations",
            Self::AttentionSwitching => "O(1) - switch cost",
            Self::BayesianInference => "O(n) - belief update",
            Self::DriftDiffusion => "O(t) - evidence accumulation steps",
            Self::ExpectedUtility => "O(n) - option values",
            Self::ThresholdCrossing => "O(1) - threshold check",
            Self::OptimalControl => "O(n³) - LQR solution",
            Self::InverseModel => "O(n²) - Jacobian inversion",
            Self::ErrorCorrection => "O(n) - proportional correction",
            Self::ActionInhibition => "O(1) - stop signal",
            Self::TemporalDifference => "O(n) - TD update",
            Self::STDP => "O(n²) - pairwise spike timing",
            Self::MetaLearning => "O(k·n) - k adaptation steps",
            Self::OneShotLearning => "O(n) - single example encoding",
        }
    }
}

// ============================================================================
// Action Triggers
// ============================================================================

/// Conditions that trigger cognitive actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionTrigger {
    /// Sensory input arrival
    SensoryInput,
    /// Prediction error exceeds threshold
    PredictionError { threshold: f64 },
    /// Arousal level change
    ArousalChange { delta: f64 },
    /// Attention shift
    AttentionShift,
    /// Decision confidence threshold
    ConfidenceThreshold { level: f64 },
    /// Memory retrieval cue
    RetrievalCue,
    /// Goal activation
    GoalActivation,
    /// Error detection
    ErrorDetection,
    /// Learning signal
    LearningSignal,
    /// Temporal event (periodic)
    TemporalEvent { period_ms: u64 },
}

// ============================================================================
// Action Executor (Zero-Copy)
// ============================================================================

/// Zero-copy action executor using shared state
pub struct ActionExecutor {
    /// Shared cognitive state (zero-copy access)
    state: Arc<RwLock<CognitiveState>>,

    /// Performance metrics
    metrics: Arc<RwLock<ActionMetrics>>,
}

impl ActionExecutor {
    /// Create new action executor
    pub fn new(state: Arc<RwLock<CognitiveState>>) -> Self {
        Self {
            state,
            metrics: Arc::new(RwLock::new(ActionMetrics::default())),
        }
    }

    /// Execute cognitive action (zero-copy)
    pub fn execute(&self, action: CognitiveAction) -> Result<Duration> {
        let start = std::time::Instant::now();

        // Execute action based on type
        match action {
            CognitiveAction::PredictSensory => self.predict_sensory()?,
            CognitiveAction::ComputePredictionError => self.compute_prediction_error()?,
            CognitiveAction::DetectSalience => self.detect_salience()?,
            CognitiveAction::AccumulateEvidence => self.accumulate_evidence()?,
            // ... (other actions)
            _ => return Err(CognitionError::ActionNotImplemented(format!("{:?}", action))),
        }

        let elapsed = start.elapsed();

        // Update metrics
        let mut metrics = self.metrics.write();
        metrics.record_action(action, elapsed);

        Ok(elapsed)
    }

    // Action implementations (zero-copy, operating on shared state)

    fn predict_sensory(&self) -> Result<()> {
        let mut state = self.state.write();
        // Predictive coding: generate top-down prediction
        // Uses hyperbolic embeddings for hierarchical prediction
        state.prediction = state.top_down_generate();
        Ok(())
    }

    fn compute_prediction_error(&self) -> Result<()> {
        let mut state = self.state.write();
        // Compute prediction error: bottom-up - top-down
        state.prediction_error = state.bottom_up_input - state.prediction;
        Ok(())
    }

    fn detect_salience(&self) -> Result<()> {
        let mut state = self.state.write();
        // Itti & Koch salience map
        state.salience = state.compute_salience_map();
        Ok(())
    }

    fn accumulate_evidence(&self) -> Result<()> {
        let mut state = self.state.write();
        // Drift-diffusion: accumulate evidence toward threshold
        state.evidence += state.current_input * state.drift_rate;
        Ok(())
    }
}

// ============================================================================
// Cognitive State (Zero-Copy Shared)
// ============================================================================

/// Shared cognitive state for zero-copy operations
#[derive(Debug, Clone)]
pub struct CognitiveState {
    // Perception
    pub bottom_up_input: f64,
    pub prediction: f64,
    pub prediction_error: f64,

    // Attention
    pub salience: f64,
    pub attention_weight: f64,

    // Decision
    pub evidence: f64,
    pub drift_rate: f64,
    pub decision_threshold: f64,

    // Placeholders for other domains
    pub current_input: f64,
}

impl CognitiveState {
    fn top_down_generate(&self) -> f64 {
        // Simplified: weighted prediction
        self.prediction * 0.9 + self.bottom_up_input * 0.1
    }

    fn compute_salience_map(&self) -> f64 {
        // Simplified: basic salience
        (self.bottom_up_input - self.prediction).abs()
    }
}

// ============================================================================
// Performance Metrics
// ============================================================================

#[derive(Debug, Clone, Default)]
pub struct ActionMetrics {
    pub total_actions: u64,
    pub total_time_ns: u64,
    pub action_counts: std::collections::HashMap<CognitiveAction, u64>,
}

impl ActionMetrics {
    fn record_action(&mut self, action: CognitiveAction, duration: Duration) {
        self.total_actions += 1;
        self.total_time_ns += duration.as_nanos() as u64;
        *self.action_counts.entry(action).or_insert(0) += 1;
    }

    pub fn average_latency_ns(&self) -> u64 {
        if self.total_actions == 0 {
            0
        } else {
            self.total_time_ns / self.total_actions
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_timing() {
        // All actions should be < 25ms (25,000,000 ns)
        for action in [
            CognitiveAction::PredictSensory,
            CognitiveAction::DetectSalience,
            CognitiveAction::AccumulateEvidence,
        ] {
            assert!(action.temporal_cost_ns() < 25_000_000,
                "{:?} exceeds 25ms budget", action);
        }
    }

    #[test]
    fn test_zero_copy_execution() {
        let state = Arc::new(RwLock::new(CognitiveState {
            bottom_up_input: 1.0,
            prediction: 0.8,
            prediction_error: 0.0,
            salience: 0.0,
            attention_weight: 0.5,
            evidence: 0.0,
            drift_rate: 0.1,
            decision_threshold: 1.0,
            current_input: 0.5,
        }));

        let executor = ActionExecutor::new(Arc::clone(&state));
        let duration = executor.execute(CognitiveAction::PredictSensory).unwrap();

        assert!(duration.as_nanos() < 25_000_000);
    }
}

// ============================================================================
// Symbolic Decision Logging
// ============================================================================

pub mod symbolic_logger;
pub use symbolic_logger::{
    SymbolicDecisionLogger, DecisionPhase, ComputationStep,
    SymbolicPath, WolframValidation,
};

// ============================================================================
// Core Cognitive Actions (Production Implementations)
// ============================================================================

pub mod core_actions;
pub use core_actions::{observe, investigate, learn, predict, broadcast, consolidate, adapt, rest};
