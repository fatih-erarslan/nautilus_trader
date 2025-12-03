//! Bridge implementations connecting autopoiesis theory to HyperPhysics components
//!
//! Bridges provide high-level integration between autopoiesis concepts and
//! the HyperPhysics ecosystem, orchestrating multiple adapters and managing
//! bidirectional state synchronization.
//!
//! ## Bridge Architecture
//!
//! ```text
//! Autopoiesis Theory          Bridge Layer              HyperPhysics
//! ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
//! │ AutopoieticSystem│────│ AutopoieticBridge│────│ Multiple crates │
//! │ (operational    │    │ (coordinates     │    │ (thermo, risk,  │
//! │  closure)       │    │  adapters)       │    │  consciousness) │
//! └─────────────────┘    └──────────────────┘    └─────────────────┘
//! ```
//!
//! ## References
//! - Maturana & Varela (1980) "Autopoiesis and Cognition"
//! - Prigogine (1977) "Self-Organization in Nonequilibrium Systems"

use std::collections::HashMap;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::adapters::{ThermoAdapter, ConsciousnessAdapter, SyncAdapter};
use crate::error::{AutopoiesisError, Result};
use crate::{OPERATIONAL_CLOSURE_THRESHOLD, SYNTERGIC_UNITY_THRESHOLD};

/// State of an autopoietic system for bridge coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutopoieticState {
    /// System health (operational closure ratio)
    pub health: f64,
    /// Set of components being produced
    pub produced_components: Vec<String>,
    /// Set of components being consumed
    pub consumed_components: Vec<String>,
    /// Current cycle count
    pub cycle_count: u64,
    /// Timestamp of last update
    pub last_update: chrono::DateTime<chrono::Utc>,
}

impl Default for AutopoieticState {
    fn default() -> Self {
        Self {
            health: 1.0,
            produced_components: Vec::new(),
            consumed_components: Vec::new(),
            cycle_count: 0,
            last_update: chrono::Utc::now(),
        }
    }
}

/// Bridge connecting Maturana-Varela autopoiesis to HyperPhysics ecosystem
///
/// Coordinates operational closure verification with thermodynamic constraints,
/// ensuring that self-producing systems maintain energetic feasibility.
///
/// ## Theoretical Foundation
///
/// An autopoietic system must satisfy:
/// 1. **Operational Closure**: All consumed components are produced internally
/// 2. **Thermodynamic Feasibility**: Energy balance allows sustained operation
/// 3. **Structural Coupling**: Environment interactions don't violate closure
///
/// ## Integration Points
/// - hyperphysics-thermo: Energy constraints via ThermoAdapter
/// - hyperphysics-consciousness: Cognitive closure via ConsciousnessAdapter
/// - hyperphysics-risk: Network resilience via NetworkAdapter
#[derive(Debug)]
pub struct AutopoieticBridge {
    /// Current autopoietic state
    state: AutopoieticState,
    /// Thermodynamic adapter
    thermo: ThermoAdapter,
    /// Operational closure threshold
    closure_threshold: f64,
    /// Component production rates
    production_rates: HashMap<String, f64>,
    /// Component consumption rates
    consumption_rates: HashMap<String, f64>,
}

impl Default for AutopoieticBridge {
    fn default() -> Self {
        Self::new(OPERATIONAL_CLOSURE_THRESHOLD)
    }
}

impl AutopoieticBridge {
    /// Create new autopoietic bridge with closure threshold
    pub fn new(closure_threshold: f64) -> Self {
        Self {
            state: AutopoieticState::default(),
            thermo: ThermoAdapter::default(),
            closure_threshold,
            production_rates: HashMap::new(),
            consumption_rates: HashMap::new(),
        }
    }

    /// Register a component with its production and consumption rates
    pub fn register_component(&mut self, name: &str, production_rate: f64, consumption_rate: f64) {
        self.production_rates.insert(name.to_string(), production_rate);
        self.consumption_rates.insert(name.to_string(), consumption_rate);

        if production_rate > 0.0 {
            if !self.state.produced_components.contains(&name.to_string()) {
                self.state.produced_components.push(name.to_string());
            }
        }
        if consumption_rate > 0.0 {
            if !self.state.consumed_components.contains(&name.to_string()) {
                self.state.consumed_components.push(name.to_string());
            }
        }
    }

    /// Verify operational closure (all consumed components are produced)
    ///
    /// Returns closure ratio ∈ [0, 1] where 1 = perfect closure
    pub fn verify_operational_closure(&self) -> f64 {
        if self.state.consumed_components.is_empty() {
            return 1.0;
        }

        let consumed_set: std::collections::HashSet<_> =
            self.state.consumed_components.iter().collect();
        let produced_set: std::collections::HashSet<_> =
            self.state.produced_components.iter().collect();

        let internally_satisfied: usize = consumed_set
            .iter()
            .filter(|c| produced_set.contains(*c))
            .count();

        internally_satisfied as f64 / consumed_set.len() as f64
    }

    /// Execute one autopoietic cycle with thermodynamic validation
    ///
    /// Returns updated health and any violations detected
    pub fn execute_cycle(&mut self) -> Result<AutopoieticCycleResult> {
        self.state.cycle_count += 1;
        self.state.last_update = chrono::Utc::now();

        // Verify operational closure
        let closure = self.verify_operational_closure();

        // Check thermodynamic feasibility
        let mut fluxes = Vec::new();
        let mut forces = Vec::new();

        for (component, &production) in &self.production_rates {
            let consumption = self.consumption_rates.get(component).copied().unwrap_or(0.0);
            fluxes.push(production - consumption);
            forces.push(1.0); // Unit thermodynamic force
        }

        let entropy_production = if !fluxes.is_empty() {
            self.thermo.compute_entropy_production(&fluxes, &forces)?
        } else {
            0.0
        };

        // Update health based on closure and thermodynamics
        let thermo_factor = if entropy_production >= 0.0 {
            1.0 // Positive entropy production is thermodynamically valid
        } else {
            0.5 // Negative entropy production indicates issues
        };

        self.state.health = closure * thermo_factor;

        let result = AutopoieticCycleResult {
            health: self.state.health,
            closure_ratio: closure,
            entropy_production,
            cycle: self.state.cycle_count,
            closure_satisfied: closure >= self.closure_threshold,
        };

        if !result.closure_satisfied {
            return Err(AutopoiesisError::OperationalClosureViolation);
        }

        Ok(result)
    }

    /// Get current state
    pub fn state(&self) -> &AutopoieticState {
        &self.state
    }

    /// Get health metric
    pub fn health(&self) -> f64 {
        self.state.health
    }
}

/// Result of an autopoietic cycle execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutopoieticCycleResult {
    /// System health after cycle
    pub health: f64,
    /// Operational closure ratio
    pub closure_ratio: f64,
    /// Entropy production during cycle
    pub entropy_production: f64,
    /// Cycle number
    pub cycle: u64,
    /// Whether closure threshold was satisfied
    pub closure_satisfied: bool,
}

/// State for dissipative structure dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DissipativeState {
    /// Current entropy production rate
    pub entropy_production: f64,
    /// Control parameter value
    pub control_parameter: f64,
    /// Current regime (stable, bifurcation, chaotic)
    pub regime: DissipativeRegime,
    /// Order parameter value
    pub order_parameter: f64,
}

/// Regime classification for dissipative structures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DissipativeRegime {
    /// Near equilibrium, linear response
    Stable,
    /// Near bifurcation point, critical fluctuations
    Bifurcation,
    /// Far from equilibrium, sustained patterns
    Dissipative,
    /// Beyond stability, chaotic dynamics
    Chaotic,
}

impl Default for DissipativeState {
    fn default() -> Self {
        Self {
            entropy_production: 0.0,
            control_parameter: 0.0,
            regime: DissipativeRegime::Stable,
            order_parameter: 0.0,
        }
    }
}

/// Bridge connecting Prigogine dissipative structures to HyperPhysics
///
/// Models far-from-equilibrium thermodynamics with bifurcation detection
/// and entropy production tracking.
///
/// ## Theoretical Foundation
///
/// Dissipative structures emerge when:
/// 1. System is far from equilibrium (high entropy production)
/// 2. Nonlinear feedback mechanisms present
/// 3. Fluctuations can be amplified past bifurcation points
///
/// ## References
/// - Prigogine & Nicolis (1977) "Self-Organization in Nonequilibrium Systems"
/// - Cross & Hohenberg (1993) "Pattern formation outside of equilibrium"
#[derive(Debug)]
pub struct DissipativeBridge {
    /// Current dissipative state
    state: DissipativeState,
    /// Thermodynamic adapter
    thermo: ThermoAdapter,
    /// History of control parameter values
    control_history: Vec<f64>,
    /// History of order parameter values
    order_history: Vec<f64>,
    /// Bifurcation points detected
    bifurcation_points: Vec<f64>,
    /// Critical entropy threshold for regime transition
    critical_entropy: f64,
}

impl Default for DissipativeBridge {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl DissipativeBridge {
    /// Create new dissipative bridge with critical entropy threshold
    pub fn new(critical_entropy: f64) -> Self {
        Self {
            state: DissipativeState::default(),
            thermo: ThermoAdapter::default(),
            control_history: Vec::new(),
            order_history: Vec::new(),
            bifurcation_points: Vec::new(),
            critical_entropy,
        }
    }

    /// Update control parameter and detect regime changes
    pub fn update_control_parameter(&mut self, value: f64) -> Result<DissipativeRegime> {
        let previous_regime = self.state.regime;
        self.state.control_parameter = value;
        self.control_history.push(value);

        // Detect regime based on control parameter dynamics
        self.state.regime = self.classify_regime(value);

        // Check for bifurcation transition
        if previous_regime != self.state.regime {
            if self.state.regime == DissipativeRegime::Bifurcation {
                self.bifurcation_points.push(value);
            }
        }

        Ok(self.state.regime)
    }

    /// Classify regime based on control parameter value
    fn classify_regime(&self, control: f64) -> DissipativeRegime {
        // Normalized regime boundaries (can be configured)
        if control < 0.3 {
            DissipativeRegime::Stable
        } else if control < 0.5 {
            DissipativeRegime::Bifurcation
        } else if control < 0.8 {
            DissipativeRegime::Dissipative
        } else {
            DissipativeRegime::Chaotic
        }
    }

    /// Compute order parameter from system observables
    ///
    /// Order parameter captures spontaneous symmetry breaking
    pub fn compute_order_parameter(&mut self, observables: &[f64]) -> f64 {
        if observables.is_empty() {
            self.state.order_parameter = 0.0;
            return 0.0;
        }

        // Order parameter = deviation from mean (simplified)
        let mean: f64 = observables.iter().sum::<f64>() / observables.len() as f64;
        let variance: f64 = observables
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / observables.len() as f64;

        // Normalized order parameter
        self.state.order_parameter = variance.sqrt() / (mean.abs() + 1e-10);
        self.order_history.push(self.state.order_parameter);

        self.state.order_parameter
    }

    /// Update entropy production from thermodynamic fluxes
    pub fn update_entropy_production(&mut self, fluxes: &[f64], forces: &[f64]) -> Result<f64> {
        self.state.entropy_production = self.thermo.compute_entropy_production(fluxes, forces)?;

        // Check for instability
        if self.state.entropy_production > self.critical_entropy * 2.0 {
            return Err(AutopoiesisError::DissipativeInstability {
                entropy: self.state.entropy_production,
            });
        }

        Ok(self.state.entropy_production)
    }

    /// Check if system is near bifurcation
    pub fn near_bifurcation(&self) -> bool {
        self.state.regime == DissipativeRegime::Bifurcation
    }

    /// Get detected bifurcation points
    pub fn bifurcation_points(&self) -> &[f64] {
        &self.bifurcation_points
    }

    /// Get current state
    pub fn state(&self) -> &DissipativeState {
        &self.state
    }
}

/// State for syntergic field dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntergicState {
    /// Current coherence level
    pub coherence: f64,
    /// Integrated information Φ
    pub phi: f64,
    /// Unity achieved flag
    pub unity_achieved: bool,
    /// Field strength
    pub field_strength: f64,
}

impl Default for SyntergicState {
    fn default() -> Self {
        Self {
            coherence: 0.0,
            phi: 0.0,
            unity_achieved: false,
            field_strength: 0.0,
        }
    }
}

/// Bridge connecting Grinberg's syntergy to HyperPhysics consciousness
///
/// Integrates neuronal field coherence with integrated information theory,
/// providing a unified view of collective consciousness dynamics.
///
/// ## Theoretical Foundation
///
/// Syntergic unity occurs when:
/// 1. Neuronal field coherence exceeds threshold
/// 2. Information integration is maximized
/// 3. Observer-observed distinction collapses
///
/// ## References
/// - Grinberg-Zylberbaum (1995) "Syntergic Theory"
/// - Tononi & Edelman (1998) "Consciousness and Complexity"
#[derive(Debug)]
pub struct SyntergicBridge {
    /// Current syntergic state
    state: SyntergicState,
    /// Consciousness adapter
    consciousness: ConsciousnessAdapter,
    /// Synchronization adapter
    sync: SyncAdapter,
    /// Unity threshold
    unity_threshold: f64,
    /// Coherence history
    coherence_history: Vec<f64>,
}

impl Default for SyntergicBridge {
    fn default() -> Self {
        Self::new(SYNTERGIC_UNITY_THRESHOLD)
    }
}

impl SyntergicBridge {
    /// Create new syntergic bridge with unity threshold
    pub fn new(unity_threshold: f64) -> Self {
        Self {
            state: SyntergicState::default(),
            consciousness: ConsciousnessAdapter::new(unity_threshold),
            sync: SyncAdapter::default(),
            unity_threshold,
            coherence_history: Vec::new(),
        }
    }

    /// Update coherence from oscillator phases
    pub fn update_from_phases(&mut self, phases: &[f64]) -> f64 {
        let _order = self.sync.compute_order_parameter(phases);
        self.state.coherence = self.sync.order_to_coherence();
        self.coherence_history.push(self.state.coherence);

        self.check_unity();
        self.state.coherence
    }

    /// Update integrated information from partition matrix
    pub fn update_phi(&mut self, partition_mi: &DMatrix<f64>) -> Result<f64> {
        self.state.phi = self.consciousness.coherence_to_phi(self.state.coherence, partition_mi)?;
        Ok(self.state.phi)
    }

    /// Check if syntergic unity is achieved
    fn check_unity(&mut self) {
        self.state.unity_achieved = self.state.coherence >= self.unity_threshold;
    }

    /// Set field strength from external measurement
    pub fn set_field_strength(&mut self, strength: f64) {
        self.state.field_strength = strength;
    }

    /// Get probability of unity based on coherence dynamics
    pub fn unity_probability(&self) -> f64 {
        // Sigmoid function centered at threshold
        let x = (self.state.coherence - self.unity_threshold) * 10.0;
        1.0 / (1.0 + (-x).exp())
    }

    /// Check for coherence loss
    pub fn check_coherence_loss(&self) -> Result<()> {
        if self.state.coherence < self.unity_threshold * 0.5 {
            return Err(AutopoiesisError::SyntergicCoherenceLoss {
                coherence: self.state.coherence,
            });
        }
        Ok(())
    }

    /// Get current state
    pub fn state(&self) -> &SyntergicState {
        &self.state
    }

    /// Get coherence history
    pub fn coherence_history(&self) -> &[f64] {
        &self.coherence_history
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_autopoietic_bridge_closure() {
        let mut bridge = AutopoieticBridge::new(0.8);

        // Register components where all consumed are produced
        bridge.register_component("A", 1.0, 0.5);
        bridge.register_component("B", 0.8, 0.3);

        let closure = bridge.verify_operational_closure();
        assert_relative_eq!(closure, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_autopoietic_bridge_partial_closure() {
        let mut bridge = AutopoieticBridge::new(0.8);

        // A is produced and consumed, B is only consumed
        bridge.register_component("A", 1.0, 0.5);
        bridge.state.consumed_components.push("B".to_string()); // Consumed but not produced

        let closure = bridge.verify_operational_closure();
        assert_relative_eq!(closure, 0.5, epsilon = 1e-10); // 1 of 2 satisfied
    }

    #[test]
    fn test_dissipative_regime_classification() {
        let mut bridge = DissipativeBridge::new(1.0);

        bridge.update_control_parameter(0.1).unwrap();
        assert_eq!(bridge.state().regime, DissipativeRegime::Stable);

        bridge.update_control_parameter(0.4).unwrap();
        assert_eq!(bridge.state().regime, DissipativeRegime::Bifurcation);

        bridge.update_control_parameter(0.6).unwrap();
        assert_eq!(bridge.state().regime, DissipativeRegime::Dissipative);

        bridge.update_control_parameter(0.9).unwrap();
        assert_eq!(bridge.state().regime, DissipativeRegime::Chaotic);
    }

    #[test]
    fn test_syntergic_unity() {
        let mut bridge = SyntergicBridge::new(0.9);

        // High coherence phases (all aligned)
        let aligned = vec![0.0, 0.01, 0.02, 0.01];
        bridge.update_from_phases(&aligned);

        assert!(bridge.state().coherence > 0.9);
        assert!(bridge.state().unity_achieved);
    }

    #[test]
    fn test_syntergic_no_unity() {
        let mut bridge = SyntergicBridge::new(0.9);

        // Low coherence phases (spread out)
        let spread = vec![0.0, 1.57, 3.14, 4.71]; // 0, π/2, π, 3π/2
        bridge.update_from_phases(&spread);

        assert!(bridge.state().coherence < 0.1);
        assert!(!bridge.state().unity_achieved);
    }
}
