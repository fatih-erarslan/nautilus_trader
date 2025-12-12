//! # Layer 8: Integration & Orchestration API
//!
//! Full cognitive loop orchestration with homeostatic control.
//!
//! ## Scientific Foundation
//!
//! **Homeostasis** - Claude Bernard & Walter Cannon:
//! - Maintenance of stable internal conditions despite external changes
//! - Negative feedback loops for regulation
//! - PID control: Proportional-Integral-Derivative
//!
//! **Cognitive Architecture Integration**:
//! - Perception → Cognition → Decision → Action loop
//! - Multi-layer coordination (Layers 1-7)
//! - Emergent behavior from layer interactions
//!
//! ## Key Equations
//!
//! ```text
//! PID Control:
//!   u(t) = K_p·e(t) + K_i·∫e(τ)dτ + K_d·de/dt
//!
//!   where:
//!   e(t) = setpoint - measurement
//!   K_p = proportional gain
//!   K_i = integral gain
//!   K_d = derivative gain
//!
//! Homeostatic Variables (6):
//!   1. Energy level
//!   2. Temperature
//!   3. Prediction error
//!   4. Attention capacity
//!   5. Memory load
//!   6. Confidence level
//! ```

use crate::api::{
    cognitive::WorkingMemory,
    consciousness::PhiResult,
    decision::BeliefState,
    metacognition::{SelfModel, LearningStrategy},
    thermodynamic::EnergyState,
};
use crate::{Result, QksError};

/// Number of homeostatic variables
pub const N_HOMEOSTATIC_VARS: usize = 6;

/// Default PID gains
pub const DEFAULT_KP: f64 = 1.0;
pub const DEFAULT_KI: f64 = 0.1;
pub const DEFAULT_KD: f64 = 0.05;

/// Cognitive cycle frequency (Hz)
pub const COGNITIVE_CYCLE_FREQ: f64 = 10.0;

/// Maximum iterations for convergence
pub const MAX_ITERATIONS: usize = 100;

/// Sensory input to the system
#[derive(Debug, Clone)]
pub struct SensoryInput {
    /// Visual features
    pub visual: Vec<f64>,
    /// Auditory features
    pub auditory: Vec<f64>,
    /// Proprioceptive (internal) state
    pub proprioceptive: Vec<f64>,
    /// Timestamp
    pub timestamp: f64,
}

/// Cognitive output (action + internal state)
#[derive(Debug, Clone)]
pub struct CognitiveOutput {
    /// Selected action
    pub action: String,
    /// Action parameters
    pub parameters: Vec<f64>,
    /// Confidence in action
    pub confidence: f64,
    /// Internal state summary
    pub internal_state: InternalState,
}

/// Internal cognitive state
#[derive(Debug, Clone)]
pub struct InternalState {
    /// Energy level (0-1)
    pub energy: f64,
    /// Temperature
    pub temperature: f64,
    /// Prediction error
    pub prediction_error: f64,
    /// Attention capacity used (0-1)
    pub attention_used: f64,
    /// Memory load (0-1)
    pub memory_load: f64,
    /// Confidence level (0-1)
    pub confidence: f64,
}

/// Homeostatic state with PID controllers
#[derive(Debug, Clone)]
pub struct HomeostasisState {
    /// Current variable values [6]
    pub values: [f64; N_HOMEOSTATIC_VARS],
    /// Setpoints for each variable [6]
    pub setpoints: [f64; N_HOMEOSTATIC_VARS],
    /// PID errors [6]
    pub errors: [f64; N_HOMEOSTATIC_VARS],
    /// Integral terms [6]
    pub integrals: [f64; N_HOMEOSTATIC_VARS],
    /// Derivative terms [6]
    pub derivatives: [f64; N_HOMEOSTATIC_VARS],
    /// Control outputs [6]
    pub controls: [f64; N_HOMEOSTATIC_VARS],
    /// PID gains (Kp, Ki, Kd)
    pub gains: (f64, f64, f64),
}

impl HomeostasisState {
    /// Create new homeostasis state with default setpoints
    pub fn new() -> Self {
        Self {
            values: [0.5; N_HOMEOSTATIC_VARS],
            setpoints: [0.7, 2.269, 0.1, 0.5, 0.5, 0.8], // Energy, Temp, Error, Attention, Memory, Confidence
            errors: [0.0; N_HOMEOSTATIC_VARS],
            integrals: [0.0; N_HOMEOSTATIC_VARS],
            derivatives: [0.0; N_HOMEOSTATIC_VARS],
            controls: [0.0; N_HOMEOSTATIC_VARS],
            gains: (DEFAULT_KP, DEFAULT_KI, DEFAULT_KD),
        }
    }

    /// Update PID controllers
    pub fn update(&mut self, dt: f64) {
        let (kp, ki, kd) = self.gains;

        for i in 0..N_HOMEOSTATIC_VARS {
            // Compute error
            let error = self.setpoints[i] - self.values[i];

            // Integral term
            self.integrals[i] += error * dt;

            // Derivative term
            self.derivatives[i] = (error - self.errors[i]) / dt;

            // PID control
            self.controls[i] = kp * error + ki * self.integrals[i] + kd * self.derivatives[i];

            // Update error
            self.errors[i] = error;
        }
    }

    /// Apply control outputs to values
    pub fn apply_controls(&mut self, dt: f64) {
        for i in 0..N_HOMEOSTATIC_VARS {
            self.values[i] += self.controls[i] * dt;
            // Clamp to reasonable ranges
            self.values[i] = self.values[i].clamp(0.0, 2.0);
        }
    }

    /// Check if all variables are within tolerance
    pub fn is_stable(&self, tolerance: f64) -> bool {
        self.errors.iter().all(|&e| e.abs() < tolerance)
    }
}

impl Default for HomeostasisState {
    fn default() -> Self {
        Self::new()
    }
}

/// System health report
#[derive(Debug, Clone)]
pub struct HealthReport {
    /// Overall health score (0-1)
    pub health_score: f64,
    /// Homeostatic stability
    pub homeostatic_stability: f64,
    /// Consciousness level (Φ)
    pub phi: f64,
    /// Resource utilization
    pub resource_usage: f64,
    /// Active issues
    pub issues: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Run one full cognitive cycle
///
/// # Arguments
/// * `input` - Sensory input
///
/// # Returns
/// Cognitive output (action + state)
///
/// # Process Flow
/// ```text
/// 1. Perception (Layer 2): Process sensory input
/// 2. Thermodynamic (Layer 1): Update energy state
/// 3. Decision (Layer 3): Active inference for action selection
/// 4. Learning (Layer 4): Update synaptic weights
/// 5. Consciousness (Layer 6): Compute Φ, broadcast to workspace
/// 6. Meta-Cognition (Layer 7): Introspection and confidence
/// 7. Integration (Layer 8): Homeostatic regulation
/// ```
///
/// # Example
/// ```rust,ignore
/// let input = SensoryInput {
///     visual: vec![0.5, 0.3, 0.8],
///     auditory: vec![0.2],
///     proprioceptive: vec![0.7],
///     timestamp: 0.0,
/// };
///
/// let output = cognitive_cycle(&input)?;
/// println!("Action: {}", output.action);
/// println!("Confidence: {}", output.confidence);
/// ```
pub fn cognitive_cycle(input: &SensoryInput) -> Result<CognitiveOutput> {
    // 1. Perception (Layer 2)
    // TODO: Process input through attention and pattern recognition

    // 2. Thermodynamic (Layer 1)
    // TODO: Update energy state

    // 3. Decision (Layer 3)
    // TODO: Active inference for action selection
    let action = "explore".to_string();
    let confidence = 0.7;

    // 4. Learning (Layer 4)
    // TODO: STDP weight updates

    // 5. Consciousness (Layer 6)
    // TODO: Compute Φ and global workspace

    // 6. Meta-Cognition (Layer 7)
    // TODO: Introspection and calibration

    // 7. Integration (Layer 8)
    // TODO: Homeostatic regulation

    Ok(CognitiveOutput {
        action,
        parameters: vec![],
        confidence,
        internal_state: InternalState {
            energy: 0.7,
            temperature: 2.269,
            prediction_error: 0.1,
            attention_used: 0.5,
            memory_load: 0.4,
            confidence: 0.7,
        },
    })
}

/// Get current homeostatic state
///
/// # Returns
/// Homeostasis state with all 6 variables
///
/// # Example
/// ```rust,ignore
/// let state = get_homeostasis()?;
/// println!("Energy: {}", state.values[0]);
/// println!("Temperature: {}", state.values[1]);
/// ```
pub fn get_homeostasis() -> Result<HomeostasisState> {
    Ok(HomeostasisState::new())
}

/// Regulate homeostatic variables using PID control
///
/// # Arguments
/// * `state` - Current homeostatic state
/// * `dt` - Time step
///
/// # Returns
/// Updated homeostatic state
pub fn regulate_homeostasis(mut state: HomeostasisState, dt: f64) -> Result<HomeostasisState> {
    state.update(dt);
    state.apply_controls(dt);
    Ok(state)
}

/// Orchestrate all cognitive layers
///
/// # Arguments
/// * `inputs` - Sequence of sensory inputs
/// * `max_steps` - Maximum simulation steps
///
/// # Returns
/// Final cognitive state
pub fn orchestrate(inputs: &[SensoryInput], max_steps: usize) -> Result<CognitiveOutput> {
    let mut homeostasis = HomeostasisState::new();
    let dt = 1.0 / COGNITIVE_CYCLE_FREQ;

    let mut final_output = None;

    for (step, input) in inputs.iter().take(max_steps).enumerate() {
        // Run cognitive cycle
        let output = cognitive_cycle(input)?;

        // Update homeostasis
        homeostasis.values[0] = output.internal_state.energy;
        homeostasis.values[1] = output.internal_state.temperature;
        homeostasis.values[2] = output.internal_state.prediction_error;
        homeostasis.values[3] = output.internal_state.attention_used;
        homeostasis.values[4] = output.internal_state.memory_load;
        homeostasis.values[5] = output.internal_state.confidence;

        homeostasis = regulate_homeostasis(homeostasis, dt)?;

        final_output = Some(output);

        // Check for stability
        if homeostasis.is_stable(0.05) {
            break;
        }
    }

    final_output.ok_or_else(|| QksError::Internal("No output generated".to_string()))
}

/// Get system health report
///
/// # Returns
/// Comprehensive health assessment
///
/// # Example
/// ```rust,ignore
/// let health = system_health()?;
/// println!("Health score: {}", health.health_score);
/// println!("Φ: {}", health.phi);
/// ```
pub fn system_health() -> Result<HealthReport> {
    let homeostasis = get_homeostasis()?;

    let stability = if homeostasis.is_stable(0.1) {
        1.0
    } else {
        0.5
    };

    // TODO: Compute actual Φ
    let phi = 1.5;

    // TODO: Measure actual resource usage
    let resource_usage = 0.6;

    let health_score = (stability + f64::min(phi / 2.0, 1.0) + (1.0 - resource_usage)) / 3.0;

    let mut issues = Vec::new();
    let mut recommendations = Vec::new();

    if stability < 0.8 {
        issues.push("Homeostatic instability detected".to_string());
        recommendations.push("Reduce cognitive load".to_string());
    }

    if phi < 1.0 {
        issues.push("Low consciousness level".to_string());
        recommendations.push("Increase integration".to_string());
    }

    if resource_usage > 0.8 {
        issues.push("High resource utilization".to_string());
        recommendations.push("Optimize memory and processing".to_string());
    }

    Ok(HealthReport {
        health_score,
        homeostatic_stability: stability,
        phi,
        resource_usage,
        issues,
        recommendations,
    })
}

/// Convergence check for cognitive loop
///
/// # Arguments
/// * `history` - Recent cognitive outputs
/// * `tolerance` - Convergence tolerance
///
/// # Returns
/// `true` if converged
pub fn has_converged(history: &[CognitiveOutput], tolerance: f64) -> bool {
    if history.len() < 2 {
        return false;
    }

    // Check if recent outputs are similar
    let recent = &history[history.len() - 1];
    let previous = &history[history.len() - 2];

    (recent.confidence - previous.confidence).abs() < tolerance
}

/// Emergency shutdown if system is unstable
///
/// # Arguments
/// * `health` - Current health report
///
/// # Returns
/// `true` if shutdown recommended
pub fn should_shutdown(health: &HealthReport) -> bool {
    health.health_score < 0.3 || health.homeostatic_stability < 0.2
}

/// Restart cognitive system
///
/// # Returns
/// Fresh homeostatic state
pub fn restart() -> Result<HomeostasisState> {
    Ok(HomeostasisState::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_homeostasis_state() {
        let state = HomeostasisState::new();
        assert_eq!(state.values.len(), N_HOMEOSTATIC_VARS);
        assert_eq!(state.setpoints.len(), N_HOMEOSTATIC_VARS);
    }

    #[test]
    fn test_homeostasis_update() {
        let mut state = HomeostasisState::new();
        state.values = [0.5, 2.0, 0.2, 0.4, 0.6, 0.7];

        state.update(0.1);

        // Check that controls were computed
        assert!(state.controls.iter().any(|&c| c != 0.0));
    }

    #[test]
    fn test_homeostasis_stability() {
        let mut state = HomeostasisState::new();
        state.errors = [0.01, 0.02, 0.01, 0.03, 0.02, 0.01];

        assert!(state.is_stable(0.05));
        assert!(!state.is_stable(0.005));
    }

    #[test]
    fn test_cognitive_cycle() {
        let input = SensoryInput {
            visual: vec![0.5, 0.3],
            auditory: vec![0.2],
            proprioceptive: vec![0.7],
            timestamp: 0.0,
        };

        let output = cognitive_cycle(&input).unwrap();
        assert!(!output.action.is_empty());
        assert!(output.confidence >= 0.0 && output.confidence <= 1.0);
    }

    #[test]
    fn test_system_health() {
        let health = system_health().unwrap();
        assert!(health.health_score >= 0.0 && health.health_score <= 1.0);
        assert!(health.phi >= 0.0);
    }

    #[test]
    fn test_should_shutdown() {
        let good_health = HealthReport {
            health_score: 0.9,
            homeostatic_stability: 0.95,
            phi: 2.0,
            resource_usage: 0.5,
            issues: vec![],
            recommendations: vec![],
        };

        assert!(!should_shutdown(&good_health));

        let bad_health = HealthReport {
            health_score: 0.2,
            homeostatic_stability: 0.1,
            phi: 0.3,
            resource_usage: 0.95,
            issues: vec!["Critical".to_string()],
            recommendations: vec![],
        };

        assert!(should_shutdown(&bad_health));
    }
}
