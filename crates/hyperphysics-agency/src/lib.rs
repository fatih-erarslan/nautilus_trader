//! # HyperPhysics Agency Framework
//!
//! Cybernetic agency implementation based on:
//! - **Free Energy Principle** (Karl Friston)
//! - **Integrated Information Theory** (Giulio Tononi)
//! - **Autopoiesis** (Maturana & Varela)
//! - **Hyperbolic Geometry** (pBRTCA architecture)
//!
//! ## Core Thesis
//!
//! **Agency emerges from:**
//! 1. Free energy minimization (survival)
//! 2. Information integration (consciousness)
//! 3. Hyperbolic embedding (capacity)
//! 4. Impermanence (adaptation)
//! 5. Post-quantum security (robustness)
//!
//! ## Example
//!
//! ```rust,no_run
//! use hyperphysics_agency::{CyberneticAgent, Observation, AgencyConfig};
//! use ndarray::Array1;
//!
//! let config = AgencyConfig::default();
//! let mut agent = CyberneticAgent::new(config);
//!
//! for step in 0..1000 {
//!     let observation = Observation {
//!         sensory: Array1::from_elem(32, 0.5),
//!         timestamp: step,
//!     };
//!     let action = agent.step(&observation);
//!
//!     // Monitor consciousness
//!     println!("Φ = {:.3}", agent.integrated_information());
//!     println!("F = {:.3}", agent.free_energy());
//! }
//! ```

use ndarray::Array1;
use rand::Rng;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// NOTE: This crate is self-contained to avoid cyclic dependencies.
// Integration types (PhiCalculator, LorentzPoint11, etc.) are provided
// by the hyperphysics-plugin crate which depends on this crate.

// ============================================================================
// Traits for External Integration
// ============================================================================

/// Trait for consciousness (Φ) calculation
/// Implementations provided by hyperphysics-consciousness or hyperphysics-plugin
pub trait PhiCalculatorTrait: Send + Sync {
    /// Compute integrated information Φ from network state
    fn compute_phi(&self, network_state: &Array1<f64>) -> f64;
}

pub mod free_energy;
pub mod active_inference;
pub mod survival;
pub mod homeostasis;
pub mod policy;
pub mod systems_dynamics;
pub mod negentropy;

pub use free_energy::FreeEnergyEngine;
pub use active_inference::ActiveInferenceEngine;
pub use survival::SurvivalDrive;
pub use homeostasis::HomeostaticController;
pub use policy::{Policy, PolicySelector};
pub use systems_dynamics::{
    AgencyDynamics, CriticalityMetrics, CriticalitySummary,
    TemporalStats, SpectralResult, DynamicsStats, Avalanche,
};
pub use negentropy::{
    NegentropyEngine, NegentropyConfig, BatesonLevel, ScaffoldMode,
    CognitiveRegulator, PedagogicScaffold, PrefrontalCortex,
    AnteriorCingulate, Insula, BasalGanglia, Hippocampus, Episode,
};

// ============================================================================
// Core Types
// ============================================================================

/// Observation from the environment
#[derive(Debug, Clone)]
pub struct Observation {
    pub sensory: Array1<f64>,
    pub timestamp: u64,
}

/// Action to execute in the environment
#[derive(Debug, Clone)]
pub struct Action {
    pub motor: Array1<f64>,
    pub timestamp: u64,
}

/// Internal state of the cybernetic agent
#[derive(Debug, Clone)]
pub struct AgentState {
    /// Position in hyperbolic H¹¹ space (12D Lorentz coordinates)
    /// Format: [time-like, space-like x 11], with ⟨x,x⟩_L = -1
    pub position: Array1<f64>,

    /// Current beliefs about hidden states
    pub beliefs: Array1<f64>,

    /// Precision (inverse variance) of beliefs
    pub precision: Array1<f64>,

    /// Prediction errors
    pub prediction_errors: VecDeque<f64>,

    /// Integrated information Φ
    pub phi: f64,

    /// Variational free energy F
    pub free_energy: f64,

    /// Control authority [0, 1]
    pub control: f64,

    /// Survival drive [0, 1]
    pub survival: f64,

    /// Model accuracy [0, 1]
    pub model_accuracy: f64,
}

impl Default for AgentState {
    fn default() -> Self {
        // Origin in Lorentz model: [1, 0, 0, ..., 0] (12D)
        let mut position = Array1::zeros(12);
        position[0] = 1.0; // Time-like coordinate

        Self {
            position,
            beliefs: Array1::from_elem(64, 0.0),
            precision: Array1::from_elem(64, 1.0),
            prediction_errors: VecDeque::with_capacity(100),
            phi: 0.1,
            free_energy: 1.0,
            control: 0.2,
            survival: 0.5,
            model_accuracy: 0.3,
        }
    }
}

/// Configuration for the cybernetic agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgencyConfig {
    /// Dimensionality of observation space
    pub observation_dim: usize,

    /// Dimensionality of action space
    pub action_dim: usize,

    /// Dimensionality of hidden state space
    pub hidden_dim: usize,

    /// Learning rate for belief updates
    pub learning_rate: f64,

    /// Free energy minimization rate
    pub fe_min_rate: f64,

    /// Survival drive strength
    pub survival_strength: f64,

    /// Impermanence rate (state change)
    pub impermanence_rate: f64,

    /// Branching ratio target (criticality)
    pub branching_target: f64,

    /// Enable Dilithium signatures
    pub use_dilithium: bool,
}

impl Default for AgencyConfig {
    fn default() -> Self {
        Self {
            observation_dim: 32,
            action_dim: 16,
            hidden_dim: 64,
            learning_rate: 0.01,
            fe_min_rate: 0.1,
            survival_strength: 1.0,
            impermanence_rate: 0.4,
            branching_target: 1.0,
            use_dilithium: false,
        }
    }
}

// ============================================================================
// Main Agent
// ============================================================================

/// Cybernetic agent with consciousness, agency, and survival drive
pub struct CyberneticAgent {
    /// Configuration
    pub config: AgencyConfig,

    /// Current state
    pub state: AgentState,

    /// Free energy engine
    pub free_energy: FreeEnergyEngine,

    /// Active inference engine
    pub active_inference: ActiveInferenceEngine,

    /// Survival drive controller
    pub survival: SurvivalDrive,

    /// Homeostatic regulator
    pub homeostasis: HomeostaticController,

    /// Policy selector
    pub policy_selector: PolicySelector,

    /// Consciousness calculator (optional - can be provided by plugin)
    pub phi_calculator: Option<Box<dyn PhiCalculatorTrait>>,

    /// Systems dynamics tracker
    pub dynamics: AgencyDynamics,

    /// Negentropy engine for pedagogic awareness
    pub negentropy_engine: NegentropyEngine,

    /// Step counter
    step_count: u64,
}

impl CyberneticAgent {
    /// Create new cybernetic agent with given configuration
    pub fn new(config: AgencyConfig) -> Self {
        let state = AgentState::default();

        Self {
            free_energy: FreeEnergyEngine::new(config.hidden_dim),
            active_inference: ActiveInferenceEngine::new(
                config.observation_dim,
                config.hidden_dim,
                config.action_dim,
            ),
            survival: SurvivalDrive::new(config.survival_strength),
            homeostasis: HomeostaticController::new(),
            policy_selector: PolicySelector::new(config.action_dim),
            phi_calculator: None, // Can be set with set_phi_calculator()
            dynamics: AgencyDynamics::new(),
            negentropy_engine: NegentropyEngine::new(),
            config,
            state,
            step_count: 0,
        }
    }

    /// Set the consciousness (Φ) calculator (provided by plugin)
    pub fn set_phi_calculator(&mut self, calculator: Box<dyn PhiCalculatorTrait>) {
        self.phi_calculator = Some(calculator);
    }

    /// Process one time step: observation → inference → action
    pub fn step(&mut self, observation: &Observation) -> Action {
        self.step_count += 1;

        // ===== PERCEPTION PHASE =====
        // Update beliefs given observation
        let prediction_error = self.active_inference
            .update_beliefs(&observation.sensory, &mut self.state.beliefs);

        self.state.prediction_errors.push_back(prediction_error);
        if self.state.prediction_errors.len() > 100 {
            self.state.prediction_errors.pop_front();
        }

        // ===== CONSCIOUSNESS PHASE =====
        // Compute integrated information Φ
        self.state.phi = self.compute_phi();

        // ===== FREE ENERGY PHASE =====
        // Compute variational free energy
        self.state.free_energy = self.free_energy.compute(
            &observation.sensory,
            &self.state.beliefs,
            &self.state.precision,
        );

        // ===== SURVIVAL PHASE =====
        // Compute survival drive based on free energy
        self.state.survival = self.survival.compute_drive(
            self.state.free_energy,
            &self.state.position,
        );

        // Homeostatic regulation
        self.homeostasis.regulate(&mut self.state);

        // ===== NEGENTROPY PHASE =====
        // Compute negentropy for pedagogic awareness (graceful scaffolding)
        let _negentropy = self.negentropy_engine.compute(
            &self.state.beliefs,
            &self.state.precision,
            prediction_error,
            self.state.free_energy,
        );

        // Get scaffolding intervention if needed (not punishment, but guidance)
        let intervention = self.negentropy_engine.get_intervention();

        // Boost curiosity for exploration when appropriate
        let curiosity_boost = self.negentropy_engine.get_curiosity_boost();

        // ===== CONTROL PHASE =====
        // Update control authority (modulated by negentropy and intervention)
        self.state.control = self.update_control() * (1.0 - intervention * 0.3)
            + curiosity_boost * 0.1;

        // ===== ACTION SELECTION PHASE =====
        // Select policy that minimizes expected free energy
        let policy = self.policy_selector.select(
            &self.state.beliefs,
            self.state.phi,
            self.state.survival,
            self.state.control,
        );

        // Generate action from policy
        let action = self.active_inference.generate_action(&policy, &self.state.beliefs);

        // ===== ADAPTATION PHASE =====
        // Structural plasticity (impermanence)
        self.adapt();

        // ===== DYNAMICS TRACKING =====
        self.dynamics.record_state(&self.state);

        Action {
            motor: action,
            timestamp: self.step_count,
        }
    }

    /// Compute integrated information Φ
    fn compute_phi(&self) -> f64 {
        // Simplified Φ calculation based on belief coherence
        // Full implementation would partition state space and compute EI
        let coherence = self.state.beliefs.iter()
            .zip(&self.state.precision)
            .map(|(b, p)| b.abs() * p)
            .sum::<f64>() / self.state.beliefs.len() as f64;

        coherence.max(0.0).min(10.0)
    }

    /// Update control authority
    fn update_control(&self) -> f64 {
        // Control emerges from Φ and model accuracy
        let base_control = self.state.phi * self.state.model_accuracy;

        // Modulated by survival drive
        let modulated = base_control * (1.0 + 0.5 * self.state.survival);

        modulated.max(0.0).min(1.0)
    }

    /// Structural adaptation (impermanence)
    fn adapt(&mut self) {
        // Apply impermanence: continuous state change
        let mut rng = rand::thread_rng();
        let noise_dist = Normal::new(0.0, self.config.impermanence_rate).unwrap();

        for belief in self.state.beliefs.iter_mut() {
            *belief += rng.sample(noise_dist);
            *belief = belief.clamp(-5.0, 5.0);
        }

        // Precision metaplasticity
        let avg_error = self.state.prediction_errors.iter()
            .sum::<f64>() / self.state.prediction_errors.len().max(1) as f64;

        for precision in self.state.precision.iter_mut() {
            *precision *= 1.0 + 0.01 * avg_error.abs();
            *precision = precision.clamp(0.1, 10.0);
        }

        // Update model accuracy
        self.state.model_accuracy = (-avg_error.abs()).exp();
    }

    // ===== PUBLIC API =====

    /// Get current integrated information
    pub fn integrated_information(&self) -> f64 {
        self.state.phi
    }

    /// Get current free energy
    pub fn free_energy(&self) -> f64 {
        self.state.free_energy
    }

    /// Get current survival drive
    pub fn survival_drive(&self) -> f64 {
        self.state.survival
    }

    /// Get current control authority
    pub fn control_authority(&self) -> f64 {
        self.state.control
    }

    /// Get model accuracy
    pub fn model_accuracy(&self) -> f64 {
        self.state.model_accuracy
    }

    /// Get current negentropy level [0, 1]
    /// N >= 0.5: Agent is "alive" (autonomous)
    /// N < 0.5: Agent needs pedagogic scaffolding
    pub fn negentropy(&self) -> f64 {
        self.negentropy_engine.negentropy()
    }

    /// Check if agent is "alive" (negentropy above threshold)
    pub fn is_alive(&self) -> bool {
        self.negentropy_engine.is_alive()
    }

    /// Get current Bateson learning level (L0-L3)
    pub fn learning_level(&self) -> BatesonLevel {
        self.negentropy_engine.learning_level()
    }

    /// Get current scaffolding mode
    pub fn scaffold_mode(&self) -> ScaffoldMode {
        self.negentropy_engine.scaffold_mode()
    }

    /// Get intrinsic motivation (from Self-Determination Theory)
    pub fn intrinsic_motivation(&self) -> f64 {
        self.negentropy_engine.intrinsic_motivation()
    }

    /// Check if agent is at criticality (branching ratio ≈ 1)
    pub fn at_criticality(&mut self) -> bool {
        self.dynamics.branching_ratio().map_or(false, |br| {
            (br - self.config.branching_target).abs() < 0.1
        })
    }

    /// Get full dynamics history
    pub fn dynamics(&self) -> &AgencyDynamics {
        &self.dynamics
    }

    /// Reset agent state
    pub fn reset(&mut self) {
        self.state = AgentState::default();
        self.step_count = 0;
        self.dynamics.clear();
        self.negentropy_engine.reset();
    }

    /// Get negentropy engine reference for advanced operations
    pub fn negentropy_engine(&self) -> &NegentropyEngine {
        &self.negentropy_engine
    }

    /// Get mutable negentropy engine for configuration
    pub fn negentropy_engine_mut(&mut self) -> &mut NegentropyEngine {
        &mut self.negentropy_engine
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_creation() {
        let config = AgencyConfig::default();
        let agent = CyberneticAgent::new(config);

        assert!(agent.state.phi >= 0.0);
        assert!(agent.state.free_energy >= 0.0);
    }

    #[test]
    fn test_agent_step() {
        let mut agent = CyberneticAgent::new(AgencyConfig::default());

        let obs = Observation {
            sensory: Array1::from_elem(32, 0.5),
            timestamp: 0,
        };

        let action = agent.step(&obs);
        assert_eq!(action.motor.len(), 16);
    }

    #[test]
    fn test_survival_drive_increases_with_high_fe() {
        let mut agent = CyberneticAgent::new(AgencyConfig::default());

        // Set high free energy (danger!)
        agent.state.free_energy = 5.0;

        let obs = Observation {
            sensory: Array1::from_elem(32, 0.0),
            timestamp: 0,
        };

        agent.step(&obs);

        // Survival drive should be elevated
        assert!(agent.survival_drive() > 0.3);
    }

    #[test]
    fn test_impermanence() {
        let mut agent = CyberneticAgent::new(AgencyConfig {
            impermanence_rate: 0.5,
            ..Default::default()
        });

        let initial_beliefs = agent.state.beliefs.clone();

        let obs = Observation {
            sensory: Array1::from_elem(32, 0.5),
            timestamp: 0,
        };

        agent.step(&obs);

        // Beliefs should have changed (impermanence)
        let change = (&agent.state.beliefs - &initial_beliefs).mapv(f64::abs).sum()
            / initial_beliefs.len() as f64;

        assert!(change > 0.1, "Impermanence should cause state change");
    }
}
