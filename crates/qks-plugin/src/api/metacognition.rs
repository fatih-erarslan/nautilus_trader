//! # Layer 7: Meta-Cognition API
//!
//! Self-modeling, introspection, meta-learning, and cybernetic agency.
//!
//! ## Scientific Foundation
//!
//! **Meta-Cognition** - "Thinking about thinking":
//! - **Self-Model**: Internal representation of own capabilities and beliefs
//! - **Introspection**: Monitoring of internal cognitive states
//! - **Meta-Learning**: Learning how to learn (learning-to-learn)
//! - **Confidence Calibration**: Accurate self-assessment of uncertainty
//!
//! **Cybernetic Agency** (via HyperPhysics Bridge):
//! - **Free Energy Principle (FEP)**: F = Complexity - Accuracy (Friston, 2010)
//! - **Integrated Information Theory (IIT)**: Φ consciousness metric (Tononi, 2004)
//! - **Negentropy Framework**: N = S_max - S_actual, N ≥ 0.5 = "alive"
//! - **Bateson Learning Levels**: L0 (Reflex) → L4 (Evolution)
//! - **Scaffold Modes**: Autonomous, CollaborativeDialogue, GuidedExploration, CuriosityNudge
//!
//! ## Key Concepts
//!
//! ```text
//! Self-Model:
//!   M_self = {beliefs, goals, capabilities, limitations}
//!
//! Introspection:
//!   confidence = P(correct | internal_state)
//!
//! Meta-Learning:
//!   θ* = argmin_θ E_task[L(f_θ(D_train), D_test)]
//!   Learn optimal initialization θ across tasks
//!
//! Confidence Calibration:
//!   ECE = E[|P(correct) - accuracy|]
//!   Expected Calibration Error
//!
//! Free Energy Principle:
//!   F = D_KL[Q(s)||P(s)] - E_Q[log P(o|s)]
//!   Complexity (KL) - Accuracy (log likelihood)
//!
//! Negentropy (Schrödinger, 1944):
//!   N = S_max - S_actual = -Σ p_i log p_i / log(n)
//!   Agent is "alive" when N ≥ 0.5
//!
//! Bateson Learning Levels (1972):
//!   L0: Reflex (stimulus-response)
//!   L1: Conditioning (pattern learning)
//!   L2: Meta-learning (learning to learn)
//!   L3: Transformation (paradigm shifts)
//!   L4: Evolution (population adaptation)
//! ```
//!
//! ## References
//! - Flavell (1979). Metacognition and cognitive monitoring.
//! - Finn et al. (2017). Model-agnostic meta-learning (MAML).
//! - Friston, K. (2010). The free-energy principle. Nature Reviews Neuroscience.
//! - Tononi, G. (2004). Integrated information theory. BMC Neuroscience.
//! - Bateson, G. (1972). Steps to an ecology of mind.
//! - Schrödinger, E. (1944). What is Life?

use crate::{Result, QksError};
use std::collections::HashMap;

// Re-export Agency Bridge types from quantum_knowledge_core
pub use quantum_knowledge_core::metacognition::{
    // Core Agency Bridge
    AgencyBridge,
    AgencyConfig,
    PIDGainsConfig,
    // Free Energy Principle results
    FreeEnergyResult,
    // Belief updating
    BeliefUpdateResult,
    // Survival drive
    SurvivalDriveResult,
    // Homeostasis
    HomeostasisResult,
    // Consciousness metrics
    ConsciousnessMetrics,
    // Policy evaluation
    PolicyEvaluationResult,
    // Negentropy & Bateson levels
    NegentropyAssessment,
    // Factory functions
    create_agency_bridge,
    create_agency_bridge_with_config,
};

/// Minimum confidence for high-certainty decisions
pub const HIGH_CONFIDENCE_THRESHOLD: f64 = 0.9;

/// Calibration tolerance
pub const CALIBRATION_TOLERANCE: f64 = 0.05;

/// Meta-learning adaptation rate
pub const META_LEARNING_RATE: f64 = 0.001;

/// Negentropy threshold for "alive" state (Schrödinger, 1944)
pub const NEGENTROPY_ALIVE_THRESHOLD: f64 = 0.5;

/// Phi threshold for consciousness emergence (Tononi, 2004)
pub const PHI_CONSCIOUSNESS_THRESHOLD: f64 = 1.0;

/// Branching ratio for self-organized criticality (Beggs & Plenz, 2003)
pub const SOC_CRITICAL_BRANCHING_RATIO: f64 = 1.0;

/// Self-model: System's representation of itself
#[derive(Debug, Clone)]
pub struct SelfModel {
    /// Beliefs about the world (id -> probability)
    pub beliefs: HashMap<String, f64>,
    /// Goals and their priorities
    pub goals: HashMap<String, f64>,
    /// Known capabilities
    pub capabilities: Vec<String>,
    /// Known limitations
    pub limitations: Vec<String>,
    /// Self-efficacy (0-1)
    pub self_efficacy: f64,
    /// Model version/update count
    pub version: usize,
}

impl SelfModel {
    /// Create new self-model
    pub fn new() -> Self {
        Self {
            beliefs: HashMap::new(),
            goals: HashMap::new(),
            capabilities: vec![],
            limitations: vec![],
            self_efficacy: 0.5,
            version: 0,
        }
    }

    /// Update belief with new evidence
    pub fn update_belief(&mut self, belief_id: &str, probability: f64) {
        self.beliefs.insert(belief_id.to_string(), probability);
        self.version += 1;
    }

    /// Add goal with priority
    pub fn add_goal(&mut self, goal: &str, priority: f64) {
        self.goals.insert(goal.to_string(), priority);
    }

    /// Add capability
    pub fn add_capability(&mut self, capability: &str) {
        if !self.capabilities.contains(&capability.to_string()) {
            self.capabilities.push(capability.to_string());
        }
    }

    /// Add limitation
    pub fn add_limitation(&mut self, limitation: &str) {
        if !self.limitations.contains(&limitation.to_string()) {
            self.limitations.push(limitation.to_string());
        }
    }
}

impl Default for SelfModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Introspection report
#[derive(Debug, Clone)]
pub struct IntrospectionReport {
    /// Current cognitive state summary
    pub state_summary: String,
    /// Confidence in current state assessment
    pub confidence: f64,
    /// Active processes
    pub active_processes: Vec<String>,
    /// Resource utilization
    pub resource_usage: ResourceUsage,
    /// Detected issues
    pub issues: Vec<String>,
    /// Overall health (0-1)
    pub health: f64,
}

/// Resource utilization metrics
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Memory usage (0-1)
    pub memory: f64,
    /// Processing usage (0-1)
    pub processing: f64,
    /// Attention usage (0-1)
    pub attention: f64,
    /// Energy usage (0-1)
    pub energy: f64,
}

/// Confidence interval for beliefs
#[derive(Debug, Clone)]
pub struct ConfidenceInterval {
    /// Point estimate
    pub estimate: f64,
    /// Lower bound (95%)
    pub lower: f64,
    /// Upper bound (95%)
    pub upper: f64,
    /// Uncertainty (width of interval)
    pub uncertainty: f64,
}

/// Learning strategy
#[derive(Debug, Clone)]
pub struct LearningStrategy {
    /// Strategy name
    pub name: String,
    /// Learning rate
    pub learning_rate: f64,
    /// Exploration rate
    pub exploration_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Update frequency
    pub update_frequency: usize,
}

/// Performance metrics for meta-learning
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Accuracy (0-1)
    pub accuracy: f64,
    /// Loss value
    pub loss: f64,
    /// Convergence rate
    pub convergence_rate: f64,
    /// Sample efficiency
    pub sample_efficiency: f64,
}

/// Get current self-model
///
/// # Returns
/// System's self-model
///
/// # Example
/// ```rust,ignore
/// let model = get_self_model()?;
/// println!("Beliefs: {:?}", model.beliefs);
/// println!("Goals: {:?}", model.goals);
/// ```
pub fn get_self_model() -> Result<SelfModel> {
    // TODO: Interface with self-model storage
    Ok(SelfModel::new())
}

/// Update self-model with new information
///
/// # Arguments
/// * `model` - Updated self-model
///
/// # Example
/// ```rust,ignore
/// let mut model = get_self_model()?;
/// model.update_belief("sky_is_blue", 0.95);
/// update_self_model(model)?;
/// ```
pub fn update_self_model(model: SelfModel) -> Result<()> {
    // TODO: Persist self-model
    Ok(())
}

/// Introspect current cognitive state
///
/// # Returns
/// Detailed introspection report
///
/// # Example
/// ```rust,ignore
/// let report = introspect()?;
/// println!("System health: {}", report.health);
/// println!("Issues: {:?}", report.issues);
/// ```
pub fn introspect() -> Result<IntrospectionReport> {
    Ok(IntrospectionReport {
        state_summary: "Normal operation".to_string(),
        confidence: 0.8,
        active_processes: vec!["perception".to_string(), "planning".to_string()],
        resource_usage: ResourceUsage {
            memory: 0.6,
            processing: 0.4,
            attention: 0.5,
            energy: 0.7,
        },
        issues: vec![],
        health: 0.9,
    })
}

/// Get confidence for a specific belief
///
/// # Arguments
/// * `belief_id` - Belief identifier
///
/// # Returns
/// Confidence interval for belief
///
/// # Example
/// ```rust,ignore
/// let conf = get_confidence("hypothesis_A")?;
/// println!("Estimate: {} ± {}", conf.estimate, conf.uncertainty);
/// ```
pub fn get_confidence(belief_id: &str) -> Result<ConfidenceInterval> {
    // TODO: Compute confidence based on evidence
    Ok(ConfidenceInterval {
        estimate: 0.7,
        lower: 0.6,
        upper: 0.8,
        uncertainty: 0.1,
    })
}

/// Calibrate confidence to match actual accuracy
///
/// # Arguments
/// * `predictions` - Predicted probabilities
/// * `outcomes` - Actual outcomes (0 or 1)
///
/// # Returns
/// Calibrated confidence model
///
/// # Example
/// ```rust,ignore
/// let predictions = vec![0.9, 0.7, 0.5];
/// let outcomes = vec![1.0, 1.0, 0.0];
/// calibrate_confidence(&predictions, &outcomes)?;
/// ```
pub fn calibrate_confidence(predictions: &[f64], outcomes: &[f64]) -> Result<f64> {
    if predictions.len() != outcomes.len() || predictions.is_empty() {
        return Err(QksError::InvalidConfig(
            "Invalid predictions/outcomes".to_string()
        ));
    }

    // Compute Expected Calibration Error (ECE)
    let n_bins = 10;
    let mut bin_counts = vec![0_usize; n_bins];
    let mut bin_accuracies = vec![0.0; n_bins];
    let mut bin_confidences = vec![0.0; n_bins];

    for (&pred, &outcome) in predictions.iter().zip(outcomes.iter()) {
        let bin = ((pred * n_bins as f64).floor() as usize).min(n_bins - 1);
        bin_counts[bin] += 1;
        bin_accuracies[bin] += outcome;
        bin_confidences[bin] += pred;
    }

    let mut ece = 0.0;
    let total = predictions.len() as f64;

    for i in 0..n_bins {
        if bin_counts[i] > 0 {
            let bin_acc = bin_accuracies[i] / bin_counts[i] as f64;
            let bin_conf = bin_confidences[i] / bin_counts[i] as f64;
            let bin_weight = bin_counts[i] as f64 / total;

            ece += bin_weight * (bin_conf - bin_acc).abs();
        }
    }

    Ok(ece)
}

/// Adapt learning strategy based on performance
///
/// # Arguments
/// * `performance` - Recent performance metrics
///
/// # Returns
/// Updated learning strategy
///
/// # Example
/// ```rust,ignore
/// let metrics = PerformanceMetrics {
///     accuracy: 0.85,
///     loss: 0.2,
///     convergence_rate: 0.05,
///     sample_efficiency: 0.7,
/// };
///
/// let strategy = adapt_learning_strategy(&metrics)?;
/// println!("New learning rate: {}", strategy.learning_rate);
/// ```
pub fn adapt_learning_strategy(performance: &PerformanceMetrics) -> Result<LearningStrategy> {
    let mut strategy = LearningStrategy {
        name: "adaptive".to_string(),
        learning_rate: 0.01,
        exploration_rate: 0.1,
        batch_size: 32,
        update_frequency: 10,
    };

    // If accuracy is low, increase exploration
    if performance.accuracy < 0.7 {
        strategy.exploration_rate = 0.3;
    }

    // If converging slowly, increase learning rate
    if performance.convergence_rate < 0.01 {
        strategy.learning_rate *= 1.5;
    }

    // If sample inefficient, increase batch size
    if performance.sample_efficiency < 0.5 {
        strategy.batch_size *= 2;
    }

    Ok(strategy)
}

/// Meta-learn: Learn optimal learning strategy across tasks
///
/// # Arguments
/// * `task_performances` - Performance on different tasks
///
/// # Returns
/// Optimal learning strategy
pub fn meta_learn(task_performances: &[PerformanceMetrics]) -> Result<LearningStrategy> {
    if task_performances.is_empty() {
        return Err(QksError::InvalidConfig("Empty task performance".to_string()));
    }

    // Average performance across tasks
    let avg_accuracy: f64 = task_performances.iter().map(|p| p.accuracy).sum::<f64>()
        / task_performances.len() as f64;

    let avg_convergence: f64 = task_performances
        .iter()
        .map(|p| p.convergence_rate)
        .sum::<f64>()
        / task_performances.len() as f64;

    // Optimize strategy based on average performance
    let strategy = LearningStrategy {
        name: "meta_learned".to_string(),
        learning_rate: if avg_convergence < 0.02 {
            0.02
        } else {
            0.01
        },
        exploration_rate: if avg_accuracy < 0.8 { 0.2 } else { 0.1 },
        batch_size: 64,
        update_frequency: 20,
    };

    Ok(strategy)
}

/// Assess capability for a task
///
/// # Arguments
/// * `task_description` - Description of task
/// * `self_model` - Current self-model
///
/// # Returns
/// Confidence in ability to perform task (0-1)
pub fn assess_capability(task_description: &str, self_model: &SelfModel) -> f64 {
    // Check if task matches known capabilities
    let capability_match = self_model
        .capabilities
        .iter()
        .any(|cap| task_description.contains(cap));

    // Check if task conflicts with limitations
    let limitation_conflict = self_model
        .limitations
        .iter()
        .any(|lim| task_description.contains(lim));

    if limitation_conflict {
        0.2 // Low confidence if limitation applies
    } else if capability_match {
        0.9 // High confidence if capability matches
    } else {
        0.5 // Moderate confidence otherwise
    }
}

/// Detect uncertainty in current state
///
/// # Returns
/// Uncertainty level (0-1, higher = more uncertain)
pub fn detect_uncertainty() -> Result<f64> {
    // TODO: Analyze belief distributions, prediction variances
    Ok(0.3)
}

/// Check if system is well-calibrated
///
/// # Arguments
/// * `ece` - Expected Calibration Error
///
/// # Returns
/// `true` if well-calibrated
pub fn is_well_calibrated(ece: f64) -> bool {
    ece < CALIBRATION_TOLERANCE
}

/// Generate meta-cognitive explanation
///
/// # Arguments
/// * `decision` - Decision that was made
/// * `reasoning` - Internal reasoning process
///
/// # Returns
/// Human-readable explanation
pub fn explain_decision(decision: &str, reasoning: &[String]) -> String {
    format!(
        "Decision: {}\n\nReasoning:\n{}",
        decision,
        reasoning
            .iter()
            .enumerate()
            .map(|(i, r)| format!("{}. {}", i + 1, r))
            .collect::<Vec<_>>()
            .join("\n")
    )
}

// =============================================================================
// Agency Bridge API (HyperPhysics Integration)
// =============================================================================

/// Compute variational free energy F = Complexity - Accuracy
///
/// Implements Friston's Free Energy Principle for active inference.
/// F measures the "surprise" of observations given the agent's model.
///
/// # Arguments
/// * `observation` - Sensory observation vector
/// * `beliefs` - Current belief state (posterior distribution)
/// * `precision` - Belief precision (inverse variance)
///
/// # Returns
/// `FreeEnergyResult` containing F, complexity, accuracy, and prediction error
///
/// # Example
/// ```rust,ignore
/// let observation = vec![1.0, 0.5, 0.2];
/// let beliefs = vec![0.8, 0.6, 0.3];
/// let precision = vec![1.0, 1.0, 1.0];
///
/// let result = compute_free_energy(&observation, &beliefs, &precision)?;
/// println!("Free energy: {} nats", result.free_energy);
/// ```
///
/// # Reference
/// Friston, K. (2010). The free-energy principle: a unified brain theory?
/// Nature Reviews Neuroscience, 11(2), 127-138.
pub fn compute_free_energy(
    observation: &[f64],
    beliefs: &[f64],
    precision: &[f64],
) -> Result<FreeEnergyResult> {
    let bridge = create_agency_bridge();
    bridge.compute_free_energy(observation, beliefs, precision)
        .map_err(|e| QksError::ComputationError(format!("Free energy computation failed: {}", e)))
}

/// Assess negentropy and Bateson learning level
///
/// Computes negentropy N = S_max - S_actual and determines:
/// - Learning level (L0-L4 per Bateson, 1972)
/// - Scaffold mode (per Vygotsky's Zone of Proximal Development)
/// - Intrinsic motivation (Self-Determination Theory)
///
/// # Bateson Learning Levels
/// - **L0 (Reflex)**: Stimulus-response, no learning
/// - **L1 (Conditioning)**: Pattern learning, associative
/// - **L2 (Meta-learning)**: Learning to learn
/// - **L3 (Transformation)**: Paradigm shifts, deep restructuring
/// - **L4 (Evolution)**: Population-level adaptation
///
/// # Scaffold Modes
/// - **Autonomous**: Agent operates independently (N ≥ 0.9)
/// - **CollaborativeDialogue**: Partnership mode (0.7 ≤ N < 0.9)
/// - **GuidedExploration**: Supported discovery (0.5 ≤ N < 0.7)
/// - **CuriosityNudge**: Gentle prompting (N < 0.5)
///
/// # Arguments
/// * `beliefs` - Current belief state vector
/// * `precision` - Belief precision (inverse variance)
/// * `prediction_error` - Current prediction error magnitude
/// * `free_energy` - Current variational free energy
///
/// # Returns
/// `NegentropyAssessment` with negentropy, Bateson level, scaffold mode, and motivation
///
/// # Example
/// ```rust,ignore
/// let result = assess_negentropy(&beliefs, &precision, 0.1, 0.5)?;
/// println!("Negentropy: {}", result.negentropy);
/// println!("Bateson Level: L{}", result.bateson_level);
/// println!("Scaffold Mode: {}", result.scaffold_mode);
/// println!("Is Alive: {}", result.is_alive);
/// ```
///
/// # Reference
/// - Bateson, G. (1972). Steps to an ecology of mind.
/// - Schrödinger, E. (1944). What is Life?
pub fn assess_negentropy(
    beliefs: &[f64],
    precision: &[f64],
    prediction_error: f64,
    free_energy: f64,
) -> Result<NegentropyAssessment> {
    let bridge = create_agency_bridge();
    bridge.assess_negentropy(beliefs, precision, prediction_error, free_energy)
        .map_err(|e| QksError::ComputationError(format!("Negentropy assessment failed: {}", e)))
}

/// Compute survival drive from free energy and hyperbolic position
///
/// Returns urgency metric [0,1] based on:
/// - Current free energy (homeostatic error)
/// - Distance from safe region in H^11 hyperbolic space
///
/// Higher drive indicates greater need for corrective action.
///
/// # Arguments
/// * `free_energy` - Current variational free energy (nats)
/// * `position` - Position in H^11 hyperbolic space (12D Lorentz coordinates)
///
/// # Returns
/// `SurvivalDriveResult` with drive [0,1], threat level, and homeostatic status
///
/// # Example
/// ```rust,ignore
/// let position = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
/// let result = compute_survival_drive(0.5, &position)?;
/// println!("Survival drive: {}", result.drive);
/// println!("Threat level: {}", result.threat_level);
/// ```
pub fn compute_survival_drive(
    free_energy: f64,
    position: &[f64; 12],
) -> Result<SurvivalDriveResult> {
    let bridge = create_agency_bridge();
    bridge.compute_survival_drive(free_energy, position)
        .map_err(|e| QksError::ComputationError(format!("Survival drive computation failed: {}", e)))
}

/// Compute consciousness metrics via IIT Φ and SOC analysis
///
/// Returns integrated information and criticality markers:
/// - Φ > 1.0: Emergent consciousness (Tononi, 2004)
/// - σ ≈ 1.0: Self-organized criticality (Beggs & Plenz, 2003)
///
/// # Arguments
/// * `network_state` - Neural network activation vector
/// * `connectivity` - Connectivity matrix (adjacency)
///
/// # Returns
/// `ConsciousnessMetrics` with Φ, branching ratio, phase coherence, and consciousness flag
///
/// # Example
/// ```rust,ignore
/// let state = vec![1.0, 0.5, 0.3];
/// let connectivity = vec![
///     vec![0.0, 1.0, 0.0],
///     vec![1.0, 0.0, 1.0],
///     vec![0.0, 1.0, 0.0],
/// ];
///
/// let metrics = compute_consciousness_metrics(&state, &connectivity)?;
/// println!("Φ = {}", metrics.phi);
/// println!("Conscious: {}", metrics.conscious);
/// ```
///
/// # Reference
/// Tononi, G. (2004). An information integration theory of consciousness.
/// BMC Neuroscience, 5(1), 42.
pub fn compute_consciousness_metrics(
    network_state: &[f64],
    connectivity: &[Vec<f64>],
) -> Result<ConsciousnessMetrics> {
    let bridge = create_agency_bridge();
    bridge.compute_consciousness_metrics(network_state, connectivity)
        .map_err(|e| QksError::ComputationError(format!("Consciousness metrics computation failed: {}", e)))
}

/// Perform homeostatic regulation using PID control
///
/// Maintains system variables within optimal bounds using:
/// - PID feedback control
/// - Allostatic prediction (setpoint adjustment)
/// - Multi-sensor interoceptive fusion
///
/// # Arguments
/// * `current_state` - Current system state vector
/// * `target_state` - Target setpoints
/// * `sensor_readings` - Interoceptive sensor array
///
/// # Returns
/// `HomeostasisResult` with control signals, setpoint adjustments, and system health
///
/// # Example
/// ```rust,ignore
/// let current = vec![1.0, 2.0];
/// let target = vec![1.5, 1.8];
/// let sensors = vec![1.2, 1.9];
///
/// let result = regulate_homeostasis(&current, &target, &sensors)?;
/// println!("Control signals: {:?}", result.control_signals);
/// println!("System health: {}", result.system_health);
/// ```
pub fn regulate_homeostasis(
    current_state: &[f64],
    target_state: &[f64],
    sensor_readings: &[f64],
) -> Result<HomeostasisResult> {
    let bridge = create_agency_bridge();
    bridge.regulate_homeostasis(current_state, target_state, sensor_readings)
        .map_err(|e| QksError::ComputationError(format!("Homeostasis regulation failed: {}", e)))
}

/// Evaluate policies and select action via expected free energy
///
/// Minimizes EFE = Epistemic Value + Pragmatic Value where:
/// - Epistemic: Information gain (exploration)
/// - Pragmatic: Goal achievement (exploitation)
///
/// # Arguments
/// * `policies` - Available policy vectors
/// * `beliefs` - Current belief state
/// * `goal` - Preferred observation (goal state)
/// * `exploration_weight` - Balance exploration vs exploitation [0,1]
///
/// # Returns
/// `PolicyEvaluationResult` with selected policy index and EFE breakdown
///
/// # Example
/// ```rust,ignore
/// let policies = vec![vec![1.0, 0.5], vec![0.8, 0.6]];
/// let beliefs = vec![0.9, 0.7];
/// let goal = vec![1.0, 1.0];
///
/// let result = evaluate_policies(&policies, &beliefs, &goal, 0.5)?;
/// println!("Selected policy: {}", result.policy_index);
/// println!("Expected free energy: {}", result.expected_free_energy);
/// ```
pub fn evaluate_policies(
    policies: &[Vec<f64>],
    beliefs: &[f64],
    goal: &[f64],
    exploration_weight: f64,
) -> Result<PolicyEvaluationResult> {
    let bridge = create_agency_bridge();
    bridge.evaluate_policies(policies, beliefs, goal, exploration_weight)
        .map_err(|e| QksError::ComputationError(format!("Policy evaluation failed: {}", e)))
}

/// Update beliefs using precision-weighted prediction errors
///
/// Implements hierarchical Bayesian inference with optimal gain.
///
/// # Arguments
/// * `observation` - Current sensory observation
/// * `current_beliefs` - Prior belief state
/// * `precision` - Belief precision vector
///
/// # Returns
/// `BeliefUpdateResult` with updated beliefs, precision, and prediction errors
pub fn update_beliefs_fep(
    observation: &[f64],
    current_beliefs: &[f64],
    precision: &[f64],
) -> Result<BeliefUpdateResult> {
    let bridge = create_agency_bridge();
    bridge.update_beliefs(observation, current_beliefs, precision)
        .map_err(|e| QksError::ComputationError(format!("Belief update failed: {}", e)))
}

/// Check if system is conscious based on Φ threshold
///
/// Returns true if Φ exceeds PHI_CONSCIOUSNESS_THRESHOLD (1.0)
pub fn is_conscious(phi: f64) -> bool {
    phi > PHI_CONSCIOUSNESS_THRESHOLD
}

/// Check if agent is "alive" based on negentropy threshold
///
/// Returns true if negentropy ≥ NEGENTROPY_ALIVE_THRESHOLD (0.5)
/// Based on Schrödinger's (1944) concept of negentropy as life's organizing principle.
pub fn is_alive(negentropy: f64) -> bool {
    negentropy >= NEGENTROPY_ALIVE_THRESHOLD
}

/// Get Bateson learning level name
///
/// # Arguments
/// * `level` - Bateson level (0-4)
///
/// # Returns
/// Human-readable level name
pub fn bateson_level_name(level: u8) -> &'static str {
    match level {
        0 => "L0: Reflex (stimulus-response)",
        1 => "L1: Conditioning (pattern learning)",
        2 => "L2: Meta-learning (learning to learn)",
        3 => "L3: Transformation (paradigm shifts)",
        4 => "L4: Evolution (population adaptation)",
        _ => "Unknown level",
    }
}

/// Get scaffold mode description
///
/// # Arguments
/// * `mode` - Scaffold mode name
///
/// # Returns
/// Description of the scaffold mode and when to use it
pub fn scaffold_mode_description(mode: &str) -> &'static str {
    match mode {
        "Autonomous" => "Agent operates independently. Minimal intervention needed.",
        "CollaborativeDialogue" => "Partnership mode. Engage in collaborative problem-solving.",
        "GuidedExploration" => "Supported discovery. Provide structure but allow exploration.",
        "CuriosityNudge" => "Gentle prompting. Spark interest without overwhelming.",
        "DirectInstruction" => "Explicit teaching. Clear step-by-step guidance needed.",
        "Observation" => "Watch and learn mode. Minimize direct interaction.",
        _ => "Unknown scaffold mode",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_model() {
        let mut model = SelfModel::new();
        model.update_belief("test", 0.9);

        assert_eq!(model.beliefs.get("test"), Some(&0.9));
        assert_eq!(model.version, 1);
    }

    #[test]
    fn test_calibrate_confidence() {
        let predictions = vec![0.9, 0.8, 0.6, 0.3];
        let outcomes = vec![1.0, 1.0, 0.0, 0.0];

        let ece = calibrate_confidence(&predictions, &outcomes).unwrap();
        assert!(ece < 0.2); // Should be well-calibrated
    }

    #[test]
    fn test_is_well_calibrated() {
        assert!(is_well_calibrated(0.03));
        assert!(!is_well_calibrated(0.1));
    }

    #[test]
    fn test_assess_capability() {
        let mut model = SelfModel::new();
        model.add_capability("image_recognition");
        model.add_limitation("real_time_video");

        let conf = assess_capability("perform image_recognition", &model);
        assert!(conf > 0.8);

        let conf = assess_capability("real_time_video processing", &model);
        assert!(conf < 0.3);
    }

    #[test]
    fn test_explain_decision() {
        let reasoning = vec![
            "Observed high confidence signal".to_string(),
            "Matched known pattern".to_string(),
            "No conflicting evidence".to_string(),
        ];

        let explanation = explain_decision("accept_proposal", &reasoning);
        assert!(explanation.contains("Decision: accept_proposal"));
        assert!(explanation.contains("1. Observed high confidence signal"));
    }
}
