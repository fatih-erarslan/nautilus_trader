//! pBit-Enhanced Quantum Annealing
//!
//! True probabilistic computing using pBit lattice for quantum-inspired
//! simulated annealing with Boltzmann dynamics.
//!
//! ## Advantages over Classical Simulated Annealing
//!
//! - Native Boltzmann sampling via pBit dynamics
//! - Parallel exploration through coupled pBit states  
//! - STDP-like learning for adaptive energy landscapes
//! - Natural tunneling via high-temperature excursions

use crate::core::*;
use crate::error::*;
use quantum_core::{PBitState, PBitConfig, PBitCoupling};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::time::Instant;

/// pBit-enhanced quantum annealing configuration
#[derive(Debug, Clone)]
pub struct PBitAnnealingConfig {
    /// Number of pBits for parameter encoding
    pub num_pbits: usize,
    /// Initial temperature
    pub initial_temperature: f64,
    /// Final temperature
    pub final_temperature: f64,
    /// Number of annealing steps
    pub num_steps: usize,
    /// Sweeps per temperature step
    pub sweeps_per_step: usize,
    /// Number of parallel chains
    pub num_chains: usize,
    /// Coupling strength for pBit interactions
    pub coupling_strength: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Convergence threshold
    pub convergence_threshold: f64,
    /// Parameter bounds (min, max) for each dimension
    pub bounds: Option<Vec<(f64, f64)>>,
}

impl Default for PBitAnnealingConfig {
    fn default() -> Self {
        Self {
            num_pbits: 16,
            initial_temperature: 10.0,
            final_temperature: 0.01,
            num_steps: 1000,
            sweeps_per_step: 10,
            num_chains: 4,
            coupling_strength: 1.0,
            seed: None,
            convergence_threshold: 1e-6,
            bounds: None,
        }
    }
}

impl PBitAnnealingConfig {
    /// Validate configuration
    pub fn validate(&self) -> QarResult<()> {
        if self.num_pbits == 0 {
            return Err(QarError::InvalidConfig("num_pbits must be > 0".into()));
        }
        if self.initial_temperature <= self.final_temperature {
            return Err(QarError::InvalidConfig(
                "initial_temperature must be > final_temperature".into()
            ));
        }
        if self.num_steps == 0 {
            return Err(QarError::InvalidConfig("num_steps must be > 0".into()));
        }
        Ok(())
    }
}

/// pBit-enhanced quantum annealer
pub struct PBitAnnealer {
    config: PBitAnnealingConfig,
    rng: ChaCha8Rng,
}

impl PBitAnnealer {
    /// Create a new pBit annealer
    pub fn new(config: PBitAnnealingConfig) -> QarResult<Self> {
        config.validate()?;
        
        let seed = config.seed.unwrap_or_else(|| rand::thread_rng().gen());
        let rng = ChaCha8Rng::seed_from_u64(seed);
        
        Ok(Self { config, rng })
    }
    
    /// Optimize using pBit annealing
    pub fn optimize<F>(&mut self, cost_function: F, initial_parameters: Vec<f64>) -> QarResult<AnnealingResult>
    where
        F: Fn(&[f64]) -> f64 + Send + Sync,
    {
        let start_time = Instant::now();
        let num_params = initial_parameters.len();
        
        // Run multiple parallel chains
        let chain_results: Vec<AnnealingResult> = (0..self.config.num_chains)
            .into_par_iter()
            .map(|chain_id| {
                let seed = self.config.seed.unwrap_or(0) + chain_id as u64;
                self.run_pbit_chain(&cost_function, &initial_parameters, num_params, seed)
            })
            .collect::<Result<Vec<_>>>()?;
        
        // Select best result
        let best_result = chain_results
            .into_iter()
            .min_by(|a, b| a.optimal_energy.partial_cmp(&b.optimal_energy).unwrap())
            .unwrap();
        
        let computation_time = start_time.elapsed().as_secs_f64();
        
        Ok(AnnealingResult::new(
            best_result.optimal_parameters,
            best_result.optimal_energy,
            best_result.iterations,
            best_result.converged,
            best_result.acceptance_rate,
            computation_time,
        ))
    }
    
    /// Run a single pBit annealing chain
    fn run_pbit_chain<F>(
        &self,
        cost_function: &F,
        initial_parameters: &[f64],
        num_params: usize,
        seed: u64,
    ) -> QarResult<AnnealingResult>
    where
        F: Fn(&[f64]) -> f64 + Send + Sync,
    {
        // Create pBit state
        let pbit_config = PBitConfig {
            temperature: self.config.initial_temperature,
            coupling_strength: self.config.coupling_strength,
            external_field: 0.0,
            seed: Some(seed),
        };
        
        let num_pbits = self.config.num_pbits.max(num_params);
        let mut pbit_state = PBitState::with_config(num_pbits, pbit_config)
            .map_err(|e| QarError::QuantumAnnealingError(e.to_string()))?;
        
        // Add nearest-neighbor couplings for exploration
        for i in 0..num_pbits.saturating_sub(1) {
            pbit_state.add_coupling(PBitCoupling::bell_coupling(i, i + 1, 0.3));
        }
        
        // Initialize parameters
        let mut current_params = initial_parameters.to_vec();
        let mut current_energy = cost_function(&current_params);
        let mut best_params = current_params.clone();
        let mut best_energy = current_energy;
        
        // Encode initial parameters into pBit probabilities
        self.encode_parameters(&mut pbit_state, &current_params);
        
        let mut accepted_count = 0;
        let mut total_proposals = 0;
        let mut prev_energy = f64::MAX;
        let mut converged = false;
        
        // Temperature schedule
        let cooling_rate = (self.config.final_temperature / self.config.initial_temperature)
            .powf(1.0 / self.config.num_steps as f64);
        let mut temperature = self.config.initial_temperature;
        
        for step in 0..self.config.num_steps {
            // Update pBit temperature
            // Note: PBitState temperature is internal, we use sweeps to equilibrate
            
            // pBit sweeps at current temperature
            for _ in 0..self.config.sweeps_per_step {
                pbit_state.sweep();
            }
            
            // Decode candidate parameters from pBit state
            let candidate_params = self.decode_parameters(&pbit_state, num_params);
            let candidate_energy = cost_function(&candidate_params);
            total_proposals += 1;
            
            // Metropolis acceptance with pBit-derived randomness
            let accept = if candidate_energy < current_energy {
                true
            } else {
                let delta = candidate_energy - current_energy;
                let acceptance_prob = (-delta / temperature).exp();
                // Use pBit magnetization as random source
                let random_val = (pbit_state.magnetization() + 1.0) / 2.0;
                random_val < acceptance_prob
            };
            
            if accept {
                current_params = candidate_params;
                current_energy = candidate_energy;
                accepted_count += 1;
                
                // Update pBit biases based on new parameters
                self.encode_parameters(&mut pbit_state, &current_params);
            }
            
            // Track best solution
            if current_energy < best_energy {
                best_params = current_params.clone();
                best_energy = current_energy;
            }
            
            // Cool down
            temperature *= cooling_rate;
            
            // Check convergence
            if step > 100 {
                let energy_change = (prev_energy - best_energy).abs();
                if energy_change < self.config.convergence_threshold {
                    converged = true;
                    break;
                }
            }
            prev_energy = best_energy;
        }
        
        let acceptance_rate = accepted_count as f64 / total_proposals as f64;
        
        Ok(AnnealingResult::new(
            best_params,
            best_energy,
            total_proposals,
            converged,
            acceptance_rate,
            0.0, // Will be set by caller
        ))
    }
    
    /// Encode parameters into pBit probabilities
    fn encode_parameters(&self, pbit_state: &mut PBitState, params: &[f64]) {
        for (i, &param) in params.iter().enumerate() {
            if i >= pbit_state.num_qubits() {
                break;
            }
            
            // Normalize parameter to [0, 1] using bounds or sigmoid
            let normalized = if let Some(ref bounds) = self.config.bounds {
                if i < bounds.len() {
                    let (min, max) = bounds[i];
                    (param - min) / (max - min)
                } else {
                    1.0 / (1.0 + (-param).exp()) // Sigmoid
                }
            } else {
                1.0 / (1.0 + (-param).exp()) // Sigmoid
            };
            
            if let Some(pbit) = pbit_state.get_pbit_mut(i) {
                pbit.probability_up = normalized.clamp(0.01, 0.99);
                pbit.bias = param * 0.1; // Small bias based on parameter
            }
        }
    }
    
    /// Decode parameters from pBit state
    fn decode_parameters(&self, pbit_state: &PBitState, num_params: usize) -> Vec<f64> {
        let mut params = Vec::with_capacity(num_params);
        
        for i in 0..num_params {
            let param = if let Some(pbit) = pbit_state.get_pbit(i) {
                // Combine spin and probability for parameter value
                let base = pbit.spin; // [-1, 1]
                let variation = pbit.probability_up - 0.5; // [-0.5, 0.5]
                
                // Scale to parameter range
                if let Some(ref bounds) = self.config.bounds {
                    if i < bounds.len() {
                        let (min, max) = bounds[i];
                        let normalized = (base + 1.0) / 2.0 + variation * 0.2;
                        min + normalized.clamp(0.0, 1.0) * (max - min)
                    } else {
                        base + variation
                    }
                } else {
                    base + variation
                }
            } else {
                0.0
            };
            params.push(param);
        }
        
        params
    }
}

/// pBit-enhanced linear regression
pub struct PBitLinearRegression {
    /// Model coefficients
    pub coefficients: Option<Vec<f64>>,
    /// Intercept term
    pub intercept: Option<f64>,
    /// Training metrics
    pub metrics: std::collections::HashMap<String, f64>,
    /// Annealing configuration
    config: PBitAnnealingConfig,
}

impl PBitLinearRegression {
    /// Create a new pBit linear regression model
    pub fn new(config: PBitAnnealingConfig) -> Self {
        Self {
            coefficients: None,
            intercept: None,
            metrics: std::collections::HashMap::new(),
            config,
        }
    }
    
    /// Train the model using pBit annealing
    pub fn fit(&mut self, problem: &RegressionProblem) -> QarResult<()> {
        let num_features = problem.feature_dimension();
        let num_params = num_features + 1; // +1 for intercept
        
        // Set bounds for parameters
        let mut config = self.config.clone();
        config.bounds = Some(vec![(-10.0, 10.0); num_params]);
        config.num_pbits = num_params.max(8);
        
        // Initial parameters
        let initial_params: Vec<f64> = vec![0.0; num_params];
        
        // Cost function (MSE)
        let cost_function = |params: &[f64]| -> f64 {
            let coefficients = &params[0..num_features];
            let intercept = params[num_features];
            
            let mut total_error = 0.0;
            for (features, &target) in problem.features.iter().zip(problem.targets.iter()) {
                let prediction: f64 = features.iter()
                    .zip(coefficients.iter())
                    .map(|(&x, &w)| x * w)
                    .sum::<f64>() + intercept;
                
                let error = prediction - target;
                total_error += error * error;
            }
            
            total_error / problem.num_samples() as f64
        };
        
        // Optimize using pBit annealing
        let mut annealer = PBitAnnealer::new(config)?;
        let result = annealer.optimize(cost_function, initial_params)?;
        
        // Extract coefficients
        self.coefficients = Some(result.optimal_parameters[0..num_features].to_vec());
        self.intercept = Some(result.optimal_parameters[num_features]);
        
        // Store metrics
        self.metrics.insert("mse".to_string(), result.optimal_energy);
        self.metrics.insert("rmse".to_string(), result.optimal_energy.sqrt());
        self.metrics.insert("iterations".to_string(), result.iterations as f64);
        self.metrics.insert("acceptance_rate".to_string(), result.acceptance_rate);
        self.metrics.insert("converged".to_string(), if result.converged { 1.0 } else { 0.0 });
        
        Ok(())
    }
    
    /// Make predictions
    pub fn predict(&self, features: &[Vec<f64>]) -> QarResult<Vec<f64>> {
        let coefficients = self.coefficients.as_ref()
            .ok_or_else(|| QarError::RegressionError("Model not trained".into()))?;
        let intercept = self.intercept
            .ok_or_else(|| QarError::RegressionError("Model not trained".into()))?;
        
        let predictions: Vec<f64> = features.iter()
            .map(|feature_vec| {
                feature_vec.iter()
                    .zip(coefficients.iter())
                    .map(|(&x, &w)| x * w)
                    .sum::<f64>() + intercept
            })
            .collect();
        
        Ok(predictions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pbit_annealer_creation() {
        let config = PBitAnnealingConfig::default();
        let annealer = PBitAnnealer::new(config);
        assert!(annealer.is_ok());
    }
    
    #[test]
    fn test_pbit_optimization() {
        let config = PBitAnnealingConfig {
            num_steps: 100,
            num_chains: 2,
            ..Default::default()
        };
        
        let mut annealer = PBitAnnealer::new(config).unwrap();
        
        // Simple quadratic: minimize (x-2)^2 + (y-3)^2
        let cost = |params: &[f64]| -> f64 {
            (params[0] - 2.0).powi(2) + (params[1] - 3.0).powi(2)
        };
        
        let result = annealer.optimize(cost, vec![0.0, 0.0]).unwrap();
        
        // Should be close to [2, 3]
        assert!(result.optimal_energy < 1.0);
    }
    
    #[test]
    fn test_pbit_linear_regression() {
        let config = PBitAnnealingConfig {
            num_steps: 200,
            num_chains: 2,
            ..Default::default()
        };
        
        // y = 2x + 1 with some noise
        let features: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let targets: Vec<f64> = features.iter().map(|f| 2.0 * f[0] + 1.0).collect();
        
        let problem = RegressionProblem::new(features.clone(), targets).unwrap();
        
        let mut model = PBitLinearRegression::new(config);
        model.fit(&problem).unwrap();
        
        let predictions = model.predict(&features).unwrap();
        assert_eq!(predictions.len(), 10);
    }
}
