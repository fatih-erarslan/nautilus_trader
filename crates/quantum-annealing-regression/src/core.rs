//! Core types and structures for Quantum Annealing Regression

use crate::error::{QarError, QarResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for quantum annealing regression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumAnnealingConfig {
    /// Number of annealing steps
    pub num_steps: usize,
    /// Initial temperature
    pub initial_temperature: f64,
    /// Final temperature
    pub final_temperature: f64,
    /// Cooling rate
    pub cooling_rate: f64,
    /// Number of parallel chains
    pub num_chains: usize,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for QuantumAnnealingConfig {
    fn default() -> Self {
        Self {
            num_steps: 1000,
            initial_temperature: 10.0,
            final_temperature: 0.01,
            cooling_rate: 0.95,
            num_chains: 4,
            max_iterations: 10000,
            tolerance: 1e-6,
            seed: None,
        }
    }
}

impl QuantumAnnealingConfig {
    /// Validate configuration parameters
    pub fn validate(&self) -> QarResult<()> {
        if self.num_steps == 0 {
            return Err(QarError::InvalidInput("num_steps must be > 0".to_string()));
        }
        if self.initial_temperature <= 0.0 {
            return Err(QarError::InvalidInput("initial_temperature must be > 0".to_string()));
        }
        if self.final_temperature <= 0.0 {
            return Err(QarError::InvalidInput("final_temperature must be > 0".to_string()));
        }
        if self.final_temperature >= self.initial_temperature {
            return Err(QarError::InvalidInput("final_temperature must be < initial_temperature".to_string()));
        }
        if self.cooling_rate <= 0.0 || self.cooling_rate >= 1.0 {
            return Err(QarError::InvalidInput("cooling_rate must be in (0, 1)".to_string()));
        }
        if self.num_chains == 0 {
            return Err(QarError::InvalidInput("num_chains must be > 0".to_string()));
        }
        if self.tolerance <= 0.0 {
            return Err(QarError::InvalidInput("tolerance must be > 0".to_string()));
        }
        Ok(())
    }
}

/// Regression problem specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionProblem {
    /// Input features (X matrix)
    pub features: Vec<Vec<f64>>,
    /// Target values (y vector)
    pub targets: Vec<f64>,
    /// Feature names (optional)
    pub feature_names: Option<Vec<String>>,
    /// Problem metadata
    pub metadata: HashMap<String, String>,
}

impl RegressionProblem {
    /// Create a new regression problem
    pub fn new(features: Vec<Vec<f64>>, targets: Vec<f64>) -> QarResult<Self> {
        if features.is_empty() {
            return Err(QarError::InvalidInput("Features cannot be empty".to_string()));
        }
        if targets.is_empty() {
            return Err(QarError::InvalidInput("Targets cannot be empty".to_string()));
        }
        if features.len() != targets.len() {
            return Err(QarError::InvalidInput("Features and targets must have same length".to_string()));
        }

        // Validate all feature vectors have same dimension
        let feature_dim = features[0].len();
        if feature_dim == 0 {
            return Err(QarError::InvalidInput("Feature dimension cannot be zero".to_string()));
        }
        
        for (i, feature) in features.iter().enumerate() {
            if feature.len() != feature_dim {
                return Err(QarError::InvalidInput(
                    format!("Feature vector {} has different dimension", i)
                ));
            }
        }

        Ok(Self {
            features,
            targets,
            feature_names: None,
            metadata: HashMap::new(),
        })
    }

    /// Get number of samples
    pub fn num_samples(&self) -> usize {
        self.features.len()
    }

    /// Get feature dimension
    pub fn feature_dimension(&self) -> usize {
        if self.features.is_empty() {
            0
        } else {
            self.features[0].len()
        }
    }

    /// Add feature names
    pub fn with_feature_names(mut self, names: Vec<String>) -> QarResult<Self> {
        if names.len() != self.feature_dimension() {
            return Err(QarError::InvalidInput("Feature names length mismatch".to_string()));
        }
        self.feature_names = Some(names);
        Ok(self)
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Quantum annealing state
#[derive(Debug, Clone)]
pub struct AnnealingState {
    /// Current parameter values
    pub parameters: Vec<f64>,
    /// Current energy (cost function value)
    pub energy: f64,
    /// Current temperature
    pub temperature: f64,
    /// Iteration number
    pub iteration: usize,
    /// Acceptance history
    pub acceptance_history: Vec<bool>,
}

impl AnnealingState {
    /// Create a new annealing state
    pub fn new(parameters: Vec<f64>, energy: f64, temperature: f64) -> Self {
        Self {
            parameters,
            energy,
            temperature,
            iteration: 0,
            acceptance_history: Vec::new(),
        }
    }

    /// Update state with new parameters and energy
    pub fn update(&mut self, new_parameters: Vec<f64>, new_energy: f64, accepted: bool) {
        if accepted {
            self.parameters = new_parameters;
            self.energy = new_energy;
        }
        self.acceptance_history.push(accepted);
        self.iteration += 1;
    }

    /// Get acceptance rate
    pub fn acceptance_rate(&self) -> f64 {
        if self.acceptance_history.is_empty() {
            0.0
        } else {
            let accepted = self.acceptance_history.iter().filter(|&&x| x).count();
            accepted as f64 / self.acceptance_history.len() as f64
        }
    }
}

/// Annealing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnealingResult {
    /// Optimal parameters found
    pub optimal_parameters: Vec<f64>,
    /// Optimal energy (cost function value)
    pub optimal_energy: f64,
    /// Number of iterations performed
    pub iterations: usize,
    /// Convergence status
    pub converged: bool,
    /// Final acceptance rate
    pub acceptance_rate: f64,
    /// Computation time in seconds
    pub computation_time: f64,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

impl AnnealingResult {
    /// Create a new annealing result
    pub fn new(
        parameters: Vec<f64>,
        energy: f64,
        iterations: usize,
        converged: bool,
        acceptance_rate: f64,
        computation_time: f64,
    ) -> Self {
        Self {
            optimal_parameters: parameters,
            optimal_energy: energy,
            iterations,
            converged,
            acceptance_rate,
            computation_time,
            metrics: HashMap::new(),
        }
    }

    /// Add a metric
    pub fn with_metric(mut self, name: String, value: f64) -> Self {
        self.metrics.insert(name, value);
        self
    }

    /// Check if optimization was successful
    pub fn is_successful(&self) -> bool {
        self.converged && self.acceptance_rate > 0.1 && self.iterations > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_quantum_annealing_config_default() {
        let config = QuantumAnnealingConfig::default();
        assert_eq!(config.num_steps, 1000);
        assert_eq!(config.initial_temperature, 10.0);
        assert_eq!(config.final_temperature, 0.01);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_quantum_annealing_config_validation() {
        let mut config = QuantumAnnealingConfig::default();
        
        // Test invalid num_steps
        config.num_steps = 0;
        assert!(config.validate().is_err());
        
        // Test invalid temperature
        config = QuantumAnnealingConfig::default();
        config.initial_temperature = -1.0;
        assert!(config.validate().is_err());
        
        // Test invalid cooling rate
        config = QuantumAnnealingConfig::default();
        config.cooling_rate = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_regression_problem_creation() {
        let features = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        let targets = vec![1.0, 2.0, 3.0];
        
        let problem = RegressionProblem::new(features, targets);
        assert!(problem.is_ok());
        
        let problem = problem.unwrap();
        assert_eq!(problem.num_samples(), 3);
        assert_eq!(problem.feature_dimension(), 2);
    }

    #[test]
    fn test_regression_problem_validation() {
        // Empty features
        let result = RegressionProblem::new(vec![], vec![1.0]);
        assert!(result.is_err());
        
        // Mismatched lengths
        let features = vec![vec![1.0, 2.0]];
        let targets = vec![1.0, 2.0];
        let result = RegressionProblem::new(features, targets);
        assert!(result.is_err());
        
        // Different feature dimensions
        let features = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0, 5.0], // Different dimension
        ];
        let targets = vec![1.0, 2.0];
        let result = RegressionProblem::new(features, targets);
        assert!(result.is_err());
    }

    #[test]
    fn test_annealing_state() {
        let mut state = AnnealingState::new(vec![1.0, 2.0], 5.0, 1.0);
        assert_eq!(state.parameters, vec![1.0, 2.0]);
        assert_eq!(state.energy, 5.0);
        assert_eq!(state.iteration, 0);
        
        // Test update with acceptance
        state.update(vec![2.0, 3.0], 3.0, true);
        assert_eq!(state.parameters, vec![2.0, 3.0]);
        assert_eq!(state.energy, 3.0);
        assert_eq!(state.iteration, 1);
        assert_relative_eq!(state.acceptance_rate(), 1.0);
        
        // Test update with rejection
        state.update(vec![4.0, 5.0], 7.0, false);
        assert_eq!(state.parameters, vec![2.0, 3.0]); // Should not change
        assert_eq!(state.energy, 3.0); // Should not change
        assert_eq!(state.iteration, 2);
        assert_relative_eq!(state.acceptance_rate(), 0.5);
    }

    #[test]
    fn test_annealing_result() {
        let result = AnnealingResult::new(
            vec![1.0, 2.0],
            3.0,
            1000,
            true,
            0.5,
            10.0,
        ).with_metric("rmse".to_string(), 0.1);
        
        assert!(result.is_successful());
        assert_eq!(result.optimal_parameters, vec![1.0, 2.0]);
        assert_eq!(result.optimal_energy, 3.0);
        assert_eq!(result.metrics.get("rmse"), Some(&0.1));
    }
}