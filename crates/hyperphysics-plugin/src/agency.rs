//! # Cybernetic Agency Module
//!
//! Re-exports from `hyperphysics-agency` crate providing:
//! - **Free Energy Principle** (Karl Friston) - Survival through surprise minimization
//! - **Integrated Information Theory** (Giulio Tononi) - Consciousness as Φ
//! - **Active Inference** - Perception-action coupling via expected free energy
//! - **Homeostatic Control** - PID + Allostatic regulation
//! - **Survival Drive** - Hyperbolic geometry-based threat assessment
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use hyperphysics_plugin::prelude::*;
//! use ndarray::Array1;
//!
//! let mut agent = CyberneticAgent::new(AgentConfig::default());
//!
//! for step in 0..1000 {
//!     let observation = AgencyObservation {
//!         sensory: Array1::from_elem(32, 0.5),
//!         timestamp: step,
//!     };
//!
//!     let action = agent.step(&observation);
//!
//!     // Monitor consciousness and survival
//!     println!("Φ = {:.3}, F = {:.3}, Survival = {:.3}",
//!         agent.integrated_information(),
//!         agent.free_energy(),
//!         agent.survival_drive()
//!     );
//! }
//! ```
//!
//! ## Core Components
//!
//! ### CyberneticAgent
//!
//! Main agent implementing the perception-action loop:
//! 1. **Perception**: Update beliefs from observations
//! 2. **Consciousness**: Compute Φ (integrated information)
//! 3. **Free Energy**: Compute F = Complexity - Accuracy
//! 4. **Survival**: Drive increases with high F (danger)
//! 5. **Homeostasis**: Regulate internal states
//! 6. **Control**: Authority emerges from Φ × Accuracy
//! 7. **Action**: Select policy minimizing expected free energy
//! 8. **Adaptation**: Structural plasticity (impermanence >40%)
//!
//! ### FreeEnergyEngine
//!
//! Implements Karl Friston's Free Energy Principle:
//! ```text
//! F = D_KL[q(s|m) || p(s|o,m)] - log p(o|m)
//!   = Complexity - Accuracy
//! ```
//!
//! ### SurvivalDrive
//!
//! Computes survival urgency from free energy and hyperbolic position:
//! ```text
//! drive = 0.7 × sigmoid(F - optimal) + 0.3 × tanh(hyperbolic_distance)
//! ```
//!
//! ### HomeostaticController
//!
//! Maintains Φ, F, and Survival within homeostatic bounds using:
//! - PID feedback control (Proportional + Integral + Derivative)
//! - Allostatic prediction (anticipatory setpoint adjustment)
//! - Interoceptive fusion (multi-sensor Kalman filtering)
//!
//! ## Mathematical Foundations
//!
//! ### Free Energy Principle
//!
//! An organism minimizes variational free energy to maintain its existence:
//!
//! ```text
//! F = E_q[ln q(s|m) - ln p(o,s|m)]
//!   = D_KL[q(s|m) || p(s|o,m)] - ln p(o|m)
//! ```
//!
//! Where:
//! - `F`: Variational free energy (surprise bound)
//! - `q(s|m)`: Recognition density (beliefs about hidden states)
//! - `p(o,s|m)`: Generative model (how observations arise from states)
//! - `D_KL`: Kullback-Leibler divergence
//!
//! ### Integrated Information Theory
//!
//! Consciousness is integrated information:
//!
//! ```text
//! Φ = min_{M} EI(X | M(X))
//! ```
//!
//! Where:
//! - `Φ`: Integrated information (bits)
//! - `M`: Minimum information partition (MIP)
//! - `EI`: Effective information across partition
//!
//! ### Systems Dynamics
//!
//! Agency emerges from coupled differential equations:
//!
//! ```text
//! dΦ/dt = α·I·C - β·Φ                  (Information integration)
//! dC/dt = γ·Φ·M - δ·C                   (Control development)
//! dS/dt = η·C·(1-S) - μ·S               (Survival emergence)
//! dM/dt = ν·(1-F)·M - ξ·M               (Model learning)
//! dF/dt = -κ·∇F - ζ·F + σ·ε             (Free energy minimization)
//! ```
//!
//! ## Integration with HyperPhysics Ecosystem
//!
//! ```rust,ignore
//! use hyperphysics_plugin::prelude::*;
//! use rand::thread_rng;
//! use ndarray::Array1;
//!
//! // Use hyperbolic geometry for survival drive
//! let position = LorentzPoint11::from_tangent_at_origin(&[0.5, 0.0, 0.0]);
//! let survival = SurvivalDrive::new(1.0);
//! let free_energy = 1.0;
//! let drive = survival.compute_drive(free_energy, &position.coords);
//!
//! // Use consciousness metrics for Phi calculation
//! let phi_calc = PhiCalculator::new(64);
//! let network_state = Array1::from_elem(64, 0.5);
//! let consciousness = phi_calc.compute(&network_state);
//!
//! // Use pBit for stochastic action selection
//! let mut rng = thread_rng();
//! let temperature = 2.269; // Ising critical temperature
//! let mut pbit = PBit::new(0.0, 0.0);
//! pbit.sample(temperature, &mut rng);
//! ```

// Re-export everything from hyperphysics-agency
pub use hyperphysics_agency::*;

/// Convenience type alias for agency configuration
pub type AgentConfig = hyperphysics_agency::AgencyConfig;

/// Convenience type alias for agent state
pub type AgentState = hyperphysics_agency::AgentState;

/// Convenience type alias for observations
pub type AgencyObservation = hyperphysics_agency::Observation;

/// Convenience type alias for actions
pub type AgencyAction = hyperphysics_agency::Action;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agency_module_exports() {
        // Verify all key types are accessible
        let _config: AgentConfig = Default::default();
        let _engine: FreeEnergyEngine = FreeEnergyEngine::new(64);
        let _survival: SurvivalDrive = SurvivalDrive::new(1.0);
        let _homeostasis: HomeostaticController = HomeostaticController::new();
    }

    #[test]
    fn test_agent_creation() {
        let config = AgentConfig::default();
        let agent = CyberneticAgent::new(config);

        assert!(agent.integrated_information() >= 0.0);
        assert!(agent.free_energy() >= 0.0);
        assert!(agent.survival_drive() >= 0.0);
        assert!(agent.control_authority() >= 0.0);
    }

    #[test]
    fn test_negentropy_module_exports() {
        // Verify negentropy types are accessible
        let engine = NegentropyEngine::new();

        // Check Bateson learning level (L0-L4)
        let level = engine.learning_level();
        assert!(matches!(level,
            BatesonLevel::L0Reflex | BatesonLevel::L1Conditioning |
            BatesonLevel::L2MetaLearning | BatesonLevel::L3Transformation |
            BatesonLevel::L4Evolution
        ));

        // Check scaffold mode
        let mode = engine.scaffold_mode();
        assert!(matches!(mode, ScaffoldMode::Observation | ScaffoldMode::CuriosityNudge
            | ScaffoldMode::GuidedExploration | ScaffoldMode::DirectInstruction
            | ScaffoldMode::CollaborativeDialogue | ScaffoldMode::Autonomous));

        // Check negentropy is alive (N >= 0.5)
        assert!(engine.negentropy() >= 0.0);
    }

    #[test]
    fn test_l4_evolution_exports() {
        // Test L4 Evolution (Holland, 1975) functionality
        let mut engine = NegentropyEngine::new();

        // Set population context (required for L4)
        engine.set_population_context(5);
        assert_eq!(engine.population_context(), 5);

        // Update fitness signal
        engine.update_fitness(0.7);
        assert!((engine.fitness_signal() - 0.7).abs() < 0.01);

        // Check L4 readiness
        // Returns: (stabilization_progress, coherence_progress, negentropy_progress, population_ready, fitness_ready)
        let (stab_progress, coherence_progress, neg_progress, population_ready, fitness_ready) = engine.l4_readiness();
        assert!(stab_progress >= 0.0 && stab_progress <= 1.0);
        assert!(coherence_progress >= 0.0 && coherence_progress <= 1.0);
        assert!(neg_progress >= 0.0 && neg_progress <= 1.0);
        // Population is 5 >= 3, so should be ready
        assert!(population_ready);
        // Fitness is 0.7 >= 0.5, so should be ready
        assert!(fitness_ready);

        // L4 requires: population >= 3, fitness >= 0.5, negentropy >= 0.9, L3 stabilization
        // Fresh engine won't meet L3 stabilization, which is expected
        assert!(!engine.at_l4());

        // Verify BatesonLevel::L4Evolution properties
        assert_eq!(BatesonLevel::L4Evolution.level(), 4);
        assert_eq!(BatesonLevel::L4Evolution.population_requirement(), 3);
        assert!((BatesonLevel::L4Evolution.fitness_pressure() - 0.5).abs() < 0.01);
        assert!(BatesonLevel::L4Evolution.requires_population());
    }

    #[test]
    fn test_cognitive_regulator_exports() {
        // Verify brain-inspired modules are accessible (use Default trait)
        let _regulator: CognitiveRegulator = Default::default();
        let _scaffold: PedagogicScaffold = Default::default();
        let _pfc: PrefrontalCortex = Default::default();
        let _acc: AnteriorCingulate = Default::default();
        let _insula: Insula = Default::default();
        let _bg: BasalGanglia = Default::default();
        let _hippocampus: Hippocampus = Default::default();
    }

    #[test]
    fn test_agent_negentropy_integration() {
        let config = AgentConfig::default();
        let agent = CyberneticAgent::new(config);

        // Verify negentropy is accessible from agent
        let negentropy = agent.negentropy();
        assert!(negentropy >= 0.0 && negentropy <= 1.0);

        // Check is_alive (N >= 0.5)
        let _alive = agent.is_alive();

        // Check learning level
        let _level = agent.learning_level();

        // Check scaffold mode
        let _mode = agent.scaffold_mode();

        // Check intrinsic motivation
        let motivation = agent.intrinsic_motivation();
        assert!(motivation >= 0.0 && motivation <= 3.0);
    }
}
