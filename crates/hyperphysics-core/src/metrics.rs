//! Engine metrics and state tracking

use serde::{Deserialize, Serialize};

/// Complete simulation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineMetrics {
    /// Simulation state
    pub state: SimulationState,

    /// Energy metrics
    pub energy: f64,
    pub energy_per_pbit: f64,

    /// Entropy metrics
    pub entropy: f64,
    pub negentropy: f64,
    pub entropy_production_rate: f64,

    /// Consciousness metrics
    pub phi: Option<f64>,
    pub ci: Option<f64>,

    /// Network metrics
    pub magnetization: f64,
    pub causal_density: f64,
    pub clustering_coefficient: f64,

    /// Thermodynamic verification
    pub landauer_bound_satisfied: bool,
    pub second_law_satisfied: bool,

    /// Performance
    pub timestep: usize,
    pub simulation_time: f64,
}

/// Current simulation state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationState {
    /// Total pBits
    pub num_pbits: usize,

    /// Current states
    pub states: Vec<bool>,

    /// Probabilities
    pub probabilities: Vec<f64>,

    /// Time elapsed
    pub time: f64,

    /// Total events
    pub events: usize,
}

impl EngineMetrics {
    /// Create metrics with defaults
    pub fn new(num_pbits: usize) -> Self {
        Self {
            state: SimulationState {
                num_pbits,
                states: vec![false; num_pbits],
                probabilities: vec![0.5; num_pbits],
                time: 0.0,
                events: 0,
            },
            energy: 0.0,
            energy_per_pbit: 0.0,
            entropy: 0.0,
            negentropy: 0.0,
            entropy_production_rate: 0.0,
            phi: None,
            ci: None,
            magnetization: 0.0,
            causal_density: 0.0,
            clustering_coefficient: 0.0,
            landauer_bound_satisfied: true,
            second_law_satisfied: true,
            timestep: 0,
            simulation_time: 0.0,
        }
    }

    /// Summary string
    pub fn summary(&self) -> String {
        format!(
            "Step {}: E={:.2e} J, S={:.2e} J/K, M={:.3}, Φ={}, CI={}",
            self.timestep,
            self.energy,
            self.entropy,
            self.magnetization,
            self.phi.map_or("N/A".to_string(), |p| format!("{:.3}", p)),
            self.ci.map_or("N/A".to_string(), |c| format!("{:.3}", c)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = EngineMetrics::new(48);
        assert_eq!(metrics.state.num_pbits, 48);
        assert_eq!(metrics.timestep, 0);
    }

    #[test]
    fn test_summary() {
        let metrics = EngineMetrics::new(48);
        let summary = metrics.summary();
        assert!(summary.contains("Step 0"));
    }
}
