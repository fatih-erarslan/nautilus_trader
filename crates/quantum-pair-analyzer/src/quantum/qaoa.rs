//! QAOA (Quantum Approximate Optimization Algorithm) Implementation
//!
//! This module implements the Quantum Approximate Optimization Algorithm for 
//! pair selection optimization in trading systems.

use std::collections::HashMap;
use std::f64::consts::PI;
use anyhow::{Result, Context};
use tracing::{info, debug, warn};
use serde::{Deserialize, Serialize};
use nalgebra::{DMatrix, DVector};
use quantum_core::{
    QuantumCircuit, QuantumState, QuantumGate, CircuitBuilder, 
    ComplexAmplitude, QuantumResult, QuantumError
};

use crate::AnalyzerError;
use super::{QuantumConfig, QuantumProblem, QuantumProblemParameters, OptimizationObjective};

/// QAOA parameter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QAOAParameters {
    /// Beta parameters for mixer Hamiltonian
    pub beta: Vec<f64>,
    /// Gamma parameters for cost Hamiltonian
    pub gamma: Vec<f64>,
    /// Number of QAOA layers
    pub layers: usize,
    /// Parameter bounds for optimization
    pub bounds: ParameterBounds,
    /// Optimization step size
    pub step_size: f64,
    /// Convergence tolerance
    pub tolerance: f64,
}

/// Parameter bounds for QAOA optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterBounds {
    /// Lower bound for beta parameters
    pub beta_min: f64,
    /// Upper bound for beta parameters
    pub beta_max: f64,
    /// Lower bound for gamma parameters
    pub gamma_min: f64,
    /// Upper bound for gamma parameters
    pub gamma_max: f64,
}

impl Default for ParameterBounds {
    fn default() -> Self {
        Self {
            beta_min: 0.0,
            beta_max: PI,
            gamma_min: 0.0,
            gamma_max: 2.0 * PI,
        }
    }
}

/// QAOA optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QAOAResult {
    /// Optimal parameters found
    pub optimal_parameters: QAOAParameters,
    /// Final objective value
    pub objective_value: f64,
    /// Number of iterations performed
    pub iterations: usize,
    /// Convergence status
    pub converged: bool,
    /// Quantum state after optimization
    pub final_state: Vec<ComplexAmplitude>,
    /// Measurement probabilities
    pub probabilities: Vec<f64>,
    /// Classical optimization trace
    pub optimization_trace: Vec<OptimizationStep>,
    /// Quantum circuit used
    pub circuit_stats: CircuitStatistics,
}

/// Single optimization step information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStep {
    /// Step number
    pub step: usize,
    /// Current parameter values
    pub parameters: QAOAParameters,
    /// Objective value at this step
    pub objective_value: f64,
    /// Gradient norm
    pub gradient_norm: f64,
    /// Step size used
    pub step_size: f64,
    /// Timestamp of the step
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Circuit statistics for QAOA
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitStatistics {
    /// Total number of gates
    pub total_gates: usize,
    /// Circuit depth
    pub depth: usize,
    /// Gate type counts
    pub gate_counts: HashMap<String, usize>,
    /// Estimated execution time
    pub estimated_execution_time_ns: u64,
    /// Memory usage estimate
    pub memory_usage_bytes: usize,
}

/// QAOA Engine for quantum optimization
#[derive(Debug)]
pub struct QAOAEngine {
    config: QuantumConfig,
    current_parameters: Option<QAOAParameters>,
    optimization_history: Vec<OptimizationStep>,
    classical_optimizer: ClassicalOptimizer,
}

/// Classical optimizer for QAOA parameters
#[derive(Debug)]
pub struct ClassicalOptimizer {
    optimizer_type: super::ClassicalOptimizer,
    step_size: f64,
    momentum: f64,
    velocity: HashMap<String, f64>,
    learning_rate_schedule: LearningRateSchedule,
}

/// Learning rate schedule for parameter optimization
#[derive(Debug, Clone)]
pub struct LearningRateSchedule {
    initial_rate: f64,
    decay_rate: f64,
    decay_steps: usize,
    minimum_rate: f64,
}

impl Default for LearningRateSchedule {
    fn default() -> Self {
        Self {
            initial_rate: 0.1,
            decay_rate: 0.95,
            decay_steps: 10,
            minimum_rate: 0.001,
        }
    }
}

impl QAOAEngine {
    /// Create a new QAOA engine
    pub async fn new(config: QuantumConfig) -> Result<Self, AnalyzerError> {
        info!("Initializing QAOA engine with {} layers", config.qaoa_layers);
        
        let classical_optimizer = ClassicalOptimizer {
            optimizer_type: config.classical_optimizer,
            step_size: 0.1,
            momentum: 0.9,
            velocity: HashMap::new(),
            learning_rate_schedule: LearningRateSchedule::default(),
        };
        
        Ok(Self {
            config,
            current_parameters: None,
            optimization_history: Vec::new(),
            classical_optimizer,
        })
    }
    
    /// Optimize using QAOA algorithm
    pub async fn optimize(
        &mut self,
        circuit: &QuantumCircuit,
        problem: &QuantumProblemParameters,
    ) -> Result<QAOAResult, AnalyzerError> {
        let start_time = std::time::Instant::now();
        debug!("Starting QAOA optimization for {} qubits", problem.num_qubits);
        
        // Initialize parameters
        let mut parameters = self.initialize_parameters(problem).await?;
        
        // Build QAOA circuit
        let mut qaoa_circuit = self.build_qaoa_circuit(problem, &parameters).await?;
        
        // Optimization loop
        let mut best_objective = f64::NEG_INFINITY;
        let mut best_parameters = parameters.clone();
        let mut converged = false;
        
        for iteration in 0..self.config.optimization_iterations {
            // Evaluate current parameters
            let (objective_value, gradient) = self.evaluate_parameters(
                &qaoa_circuit, problem, &parameters
            ).await?;
            
            // Update best solution
            if objective_value > best_objective {
                best_objective = objective_value;
                best_parameters = parameters.clone();
            }
            
            // Calculate gradient norm
            let gradient_norm = self.calculate_gradient_norm(&gradient);
            
            // Record optimization step
            self.optimization_history.push(OptimizationStep {
                step: iteration,
                parameters: parameters.clone(),
                objective_value,
                gradient_norm,
                step_size: self.classical_optimizer.step_size,
                timestamp: chrono::Utc::now(),
            });
            
            // Check convergence
            if gradient_norm < self.config.convergence_threshold {
                converged = true;
                info!("QAOA converged after {} iterations", iteration + 1);
                break;
            }
            
            // Update parameters using classical optimizer
            parameters = self.classical_optimizer.update_parameters(
                parameters, gradient, iteration
            ).await?;
            
            // Rebuild circuit with new parameters
            qaoa_circuit = self.build_qaoa_circuit(problem, &parameters).await?;
            
            if iteration % 10 == 0 {
                debug!("QAOA iteration {}: objective = {:.6}, gradient_norm = {:.6}", 
                       iteration, objective_value, gradient_norm);
            }
        }
        
        // Execute final circuit to get quantum state
        let final_result = self.execute_qaoa_circuit(&qaoa_circuit).await?;
        
        // Calculate circuit statistics
        let circuit_stats = self.calculate_circuit_statistics(&qaoa_circuit);
        
        let duration = start_time.elapsed();
        info!("QAOA optimization completed in {:?} with objective value {:.6}", 
              duration, best_objective);
        
        Ok(QAOAResult {
            optimal_parameters: best_parameters,
            objective_value: best_objective,
            iterations: self.optimization_history.len(),
            converged,
            final_state: final_result.state.amplitudes().to_vec(),
            probabilities: final_result.probabilities,
            optimization_trace: self.optimization_history.clone(),
            circuit_stats,
        })
    }
    
    /// Initialize QAOA parameters
    async fn initialize_parameters(
        &self,
        problem: &QuantumProblemParameters,
    ) -> Result<QAOAParameters, AnalyzerError> {
        let layers = self.config.qaoa_layers;
        
        // Initialize with random values within bounds
        let bounds = ParameterBounds::default();
        let mut rng = rand::thread_rng();
        
        let beta: Vec<f64> = (0..layers)
            .map(|_| {
                use rand::Rng;
                rng.gen_range(bounds.beta_min..bounds.beta_max)
            })
            .collect();
        
        let gamma: Vec<f64> = (0..layers)
            .map(|_| {
                use rand::Rng;
                rng.gen_range(bounds.gamma_min..bounds.gamma_max)
            })
            .collect();
        
        Ok(QAOAParameters {
            beta,
            gamma,
            layers,
            bounds,
            step_size: 0.1,
            tolerance: self.config.convergence_threshold,
        })
    }
    
    /// Build QAOA circuit
    async fn build_qaoa_circuit(
        &self,
        problem: &QuantumProblemParameters,
        parameters: &QAOAParameters,
    ) -> Result<QuantumCircuit, AnalyzerError> {
        let mut circuit = QuantumCircuit::new(
            format!("QAOA_p{}", parameters.layers),
            problem.num_qubits,
        );
        
        // Initial superposition state
        for qubit in 0..problem.num_qubits {
            circuit.add_hadamard(qubit)
                .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        }
        
        // QAOA layers
        for layer in 0..parameters.layers {
            // Cost Hamiltonian (problem layer)
            self.add_cost_hamiltonian(&mut circuit, problem, parameters.gamma[layer]).await?;
            
            // Mixer Hamiltonian (driver layer)
            self.add_mixer_hamiltonian(&mut circuit, problem, parameters.beta[layer]).await?;
        }
        
        Ok(circuit)
    }
    
    /// Add cost Hamiltonian to circuit
    async fn add_cost_hamiltonian(
        &self,
        circuit: &mut QuantumCircuit,
        problem: &QuantumProblemParameters,
        gamma: f64,
    ) -> Result<(), AnalyzerError> {
        let num_qubits = problem.num_qubits;
        
        // Add diagonal terms (individual qubit costs)
        for i in 0..num_qubits {
            let cost = problem.cost_matrix[(i, i)];
            if cost.abs() > 1e-10 {
                circuit.add_gate(
                    QuantumGate::RZ { 
                        qubit: i, 
                        angle: 2.0 * gamma * cost 
                    },
                    vec![i],
                ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
            }
        }
        
        // Add off-diagonal terms (coupling between qubits)
        for i in 0..num_qubits {
            for j in (i + 1)..num_qubits {
                let coupling = problem.cost_matrix[(i, j)];
                if coupling.abs() > 1e-10 {
                    // ZZ interaction
                    circuit.add_cnot(i, j)
                        .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                    circuit.add_gate(
                        QuantumGate::RZ { 
                            qubit: j, 
                            angle: 2.0 * gamma * coupling 
                        },
                        vec![j],
                    ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                    circuit.add_cnot(i, j)
                        .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Add mixer Hamiltonian to circuit
    async fn add_mixer_hamiltonian(
        &self,
        circuit: &mut QuantumCircuit,
        problem: &QuantumProblemParameters,
        beta: f64,
    ) -> Result<(), AnalyzerError> {
        // Standard X mixer
        for qubit in 0..problem.num_qubits {
            circuit.add_gate(
                QuantumGate::RX { 
                    qubit, 
                    angle: 2.0 * beta 
                },
                vec![qubit],
            ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        }
        
        Ok(())
    }
    
    /// Evaluate parameters and compute gradient
    async fn evaluate_parameters(
        &self,
        circuit: &QuantumCircuit,
        problem: &QuantumProblemParameters,
        parameters: &QAOAParameters,
    ) -> Result<(f64, ParameterGradient), AnalyzerError> {
        // Execute circuit to get quantum state
        let result = self.execute_qaoa_circuit(circuit).await?;
        
        // Calculate objective value
        let objective_value = self.calculate_objective_value(&result, problem).await?;
        
        // Calculate gradient using parameter shift rule
        let gradient = self.calculate_gradient(circuit, problem, parameters).await?;
        
        Ok((objective_value, gradient))
    }
    
    /// Execute QAOA circuit
    async fn execute_qaoa_circuit(
        &self,
        circuit: &QuantumCircuit,
    ) -> Result<QuantumResult, AnalyzerError> {
        // Create quantum state
        let mut state = QuantumState::new(circuit.num_qubits)
            .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        
        // Execute circuit
        circuit.execute(&mut state)
            .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        
        // Perform measurements
        let mut probabilities = Vec::new();
        for i in 0..(1 << circuit.num_qubits) {
            let amplitude = state.get_amplitude(i)
                .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
            probabilities.push(amplitude.norm_sqr());
        }
        
        Ok(QuantumResult {
            state,
            probabilities,
            metadata: quantum_core::ComputationMetadata {
                num_qubits: circuit.num_qubits,
                gate_count: circuit.instructions.len(),
                circuit_depth: circuit.depth(),
                backend: "QAOA_Simulator".to_string(),
                error_correction: false,
            },
            fidelity: 0.99,
            execution_time_ns: 1000000,
        })
    }
    
    /// Calculate objective value from quantum result
    async fn calculate_objective_value(
        &self,
        result: &QuantumResult,
        problem: &QuantumProblemParameters,
    ) -> Result<f64, AnalyzerError> {
        let mut objective_value = 0.0;
        let num_qubits = problem.num_qubits;
        
        // Calculate expectation value of cost Hamiltonian
        for (bitstring, &probability) in result.probabilities.iter().enumerate() {
            if probability > 1e-10 {
                let cost = self.calculate_bitstring_cost(bitstring, problem).await?;
                objective_value += probability * cost;
            }
        }
        
        // Apply optimization objective
        match problem.optimization_objective {
            OptimizationObjective::MaximizeReturn => objective_value,
            OptimizationObjective::MinimizeRisk => -objective_value,
            OptimizationObjective::MaximizeRiskAdjustedReturn => objective_value,
            OptimizationObjective::MaximizeSharpeRatio => objective_value,
            OptimizationObjective::MaximizeDiversification => objective_value,
        }
        
        Ok(objective_value)
    }
    
    /// Calculate cost for a specific bitstring
    async fn calculate_bitstring_cost(
        &self,
        bitstring: usize,
        problem: &QuantumProblemParameters,
    ) -> Result<f64, AnalyzerError> {
        let mut cost = 0.0;
        let num_qubits = problem.num_qubits;
        
        // Convert bitstring to binary representation
        let bits: Vec<bool> = (0..num_qubits)
            .map(|i| (bitstring >> i) & 1 == 1)
            .collect();
        
        // Calculate cost from cost matrix
        for i in 0..num_qubits {
            for j in 0..num_qubits {
                if bits[i] && bits[j] {
                    cost += problem.cost_matrix[(i, j)];
                }
            }
        }
        
        // Apply constraint penalties
        for (k, constraint_matrix) in problem.constraint_matrices.iter().enumerate() {
            let constraint_value: f64 = (0..num_qubits)
                .map(|i| if bits[i] { constraint_matrix[(0, i)] } else { 0.0 })
                .sum();
            
            if k < problem.penalty_coefficients.len() {
                cost -= problem.penalty_coefficients[k] * constraint_value.powi(2);
            }
        }
        
        Ok(cost)
    }
    
    /// Calculate gradient using parameter shift rule
    async fn calculate_gradient(
        &self,
        circuit: &QuantumCircuit,
        problem: &QuantumProblemParameters,
        parameters: &QAOAParameters,
    ) -> Result<ParameterGradient, AnalyzerError> {
        let mut beta_gradient = Vec::new();
        let mut gamma_gradient = Vec::new();
        
        let shift = PI / 2.0;
        
        // Calculate gradient for beta parameters
        for i in 0..parameters.beta.len() {
            let mut params_plus = parameters.clone();
            let mut params_minus = parameters.clone();
            
            params_plus.beta[i] += shift;
            params_minus.beta[i] -= shift;
            
            let circuit_plus = self.build_qaoa_circuit(problem, &params_plus).await?;
            let circuit_minus = self.build_qaoa_circuit(problem, &params_minus).await?;
            
            let result_plus = self.execute_qaoa_circuit(&circuit_plus).await?;
            let result_minus = self.execute_qaoa_circuit(&circuit_minus).await?;
            
            let objective_plus = self.calculate_objective_value(&result_plus, problem).await?;
            let objective_minus = self.calculate_objective_value(&result_minus, problem).await?;
            
            let gradient = (objective_plus - objective_minus) / (2.0 * shift);
            beta_gradient.push(gradient);
        }
        
        // Calculate gradient for gamma parameters
        for i in 0..parameters.gamma.len() {
            let mut params_plus = parameters.clone();
            let mut params_minus = parameters.clone();
            
            params_plus.gamma[i] += shift;
            params_minus.gamma[i] -= shift;
            
            let circuit_plus = self.build_qaoa_circuit(problem, &params_plus).await?;
            let circuit_minus = self.build_qaoa_circuit(problem, &params_minus).await?;
            
            let result_plus = self.execute_qaoa_circuit(&circuit_plus).await?;
            let result_minus = self.execute_qaoa_circuit(&circuit_minus).await?;
            
            let objective_plus = self.calculate_objective_value(&result_plus, problem).await?;
            let objective_minus = self.calculate_objective_value(&result_minus, problem).await?;
            
            let gradient = (objective_plus - objective_minus) / (2.0 * shift);
            gamma_gradient.push(gradient);
        }
        
        Ok(ParameterGradient {
            beta_gradient,
            gamma_gradient,
        })
    }
    
    /// Calculate gradient norm
    fn calculate_gradient_norm(&self, gradient: &ParameterGradient) -> f64 {
        let beta_norm_sq: f64 = gradient.beta_gradient.iter().map(|g| g * g).sum();
        let gamma_norm_sq: f64 = gradient.gamma_gradient.iter().map(|g| g * g).sum();
        
        (beta_norm_sq + gamma_norm_sq).sqrt()
    }
    
    /// Calculate circuit statistics
    fn calculate_circuit_statistics(&self, circuit: &QuantumCircuit) -> CircuitStatistics {
        let gate_counts = circuit.gate_counts();
        let total_gates = circuit.instructions.len();
        let depth = circuit.depth();
        
        CircuitStatistics {
            total_gates,
            depth,
            gate_counts,
            estimated_execution_time_ns: (total_gates as u64) * 1000, // Rough estimate
            memory_usage_bytes: circuit.estimate_memory_usage_public(),
        }
    }
    
    /// Reset QAOA engine state
    pub async fn reset(&mut self) -> Result<(), AnalyzerError> {
        self.current_parameters = None;
        self.optimization_history.clear();
        Ok(())
    }
    
    /// Get optimization history
    pub fn get_optimization_history(&self) -> &[OptimizationStep] {
        &self.optimization_history
    }
    
    /// Get current parameters
    pub fn get_current_parameters(&self) -> Option<&QAOAParameters> {
        self.current_parameters.as_ref()
    }
}

/// Parameter gradient structure
#[derive(Debug, Clone)]
pub struct ParameterGradient {
    pub beta_gradient: Vec<f64>,
    pub gamma_gradient: Vec<f64>,
}

impl ClassicalOptimizer {
    /// Update parameters using selected optimization algorithm
    pub async fn update_parameters(
        &mut self,
        current_params: QAOAParameters,
        gradient: ParameterGradient,
        iteration: usize,
    ) -> Result<QAOAParameters, AnalyzerError> {
        match self.optimizer_type {
            super::ClassicalOptimizer::Adam => {
                self.adam_update(current_params, gradient, iteration).await
            }
            super::ClassicalOptimizer::BFGS => {
                self.bfgs_update(current_params, gradient, iteration).await
            }
            super::ClassicalOptimizer::NelderMead => {
                self.nelder_mead_update(current_params, gradient, iteration).await
            }
            super::ClassicalOptimizer::GradientDescent => {
                self.gradient_descent_update(current_params, gradient, iteration).await
            }
            super::ClassicalOptimizer::CobylA => {
                self.cobyla_update(current_params, gradient, iteration).await
            }
        }
    }
    
    /// Adam optimizer update
    async fn adam_update(
        &mut self,
        mut params: QAOAParameters,
        gradient: ParameterGradient,
        iteration: usize,
    ) -> Result<QAOAParameters, AnalyzerError> {
        let learning_rate = self.learning_rate_schedule.get_rate(iteration);
        let beta1 = 0.9;
        let beta2 = 0.999;
        let epsilon = 1e-8;
        
        // Update beta parameters
        for i in 0..params.beta.len() {
            let key = format!("beta_{}", i);
            let velocity = self.velocity.get(&key).unwrap_or(&0.0);
            let new_velocity = beta1 * velocity + (1.0 - beta1) * gradient.beta_gradient[i];
            self.velocity.insert(key.clone(), new_velocity);
            
            let bias_correction = 1.0 - beta1.powi(iteration as i32 + 1);
            let corrected_velocity = new_velocity / bias_correction;
            
            params.beta[i] -= learning_rate * corrected_velocity;
            params.beta[i] = params.beta[i].max(params.bounds.beta_min).min(params.bounds.beta_max);
        }
        
        // Update gamma parameters
        for i in 0..params.gamma.len() {
            let key = format!("gamma_{}", i);
            let velocity = self.velocity.get(&key).unwrap_or(&0.0);
            let new_velocity = beta1 * velocity + (1.0 - beta1) * gradient.gamma_gradient[i];
            self.velocity.insert(key.clone(), new_velocity);
            
            let bias_correction = 1.0 - beta1.powi(iteration as i32 + 1);
            let corrected_velocity = new_velocity / bias_correction;
            
            params.gamma[i] -= learning_rate * corrected_velocity;
            params.gamma[i] = params.gamma[i].max(params.bounds.gamma_min).min(params.bounds.gamma_max);
        }
        
        Ok(params)
    }
    
    /// Simple gradient descent update
    async fn gradient_descent_update(
        &mut self,
        mut params: QAOAParameters,
        gradient: ParameterGradient,
        iteration: usize,
    ) -> Result<QAOAParameters, AnalyzerError> {
        let learning_rate = self.learning_rate_schedule.get_rate(iteration);
        
        // Update beta parameters
        for i in 0..params.beta.len() {
            params.beta[i] -= learning_rate * gradient.beta_gradient[i];
            params.beta[i] = params.beta[i].max(params.bounds.beta_min).min(params.bounds.beta_max);
        }
        
        // Update gamma parameters
        for i in 0..params.gamma.len() {
            params.gamma[i] -= learning_rate * gradient.gamma_gradient[i];
            params.gamma[i] = params.gamma[i].max(params.bounds.gamma_min).min(params.bounds.gamma_max);
        }
        
        Ok(params)
    }
    
    /// BFGS update (simplified implementation)
    async fn bfgs_update(
        &mut self,
        params: QAOAParameters,
        gradient: ParameterGradient,
        iteration: usize,
    ) -> Result<QAOAParameters, AnalyzerError> {
        // Simplified BFGS - would need proper implementation in production
        self.gradient_descent_update(params, gradient, iteration).await
    }
    
    /// Nelder-Mead update (simplified implementation)
    async fn nelder_mead_update(
        &mut self,
        params: QAOAParameters,
        gradient: ParameterGradient,
        iteration: usize,
    ) -> Result<QAOAParameters, AnalyzerError> {
        // Simplified Nelder-Mead - would need proper implementation in production
        self.gradient_descent_update(params, gradient, iteration).await
    }
    
    /// COBYLA update (simplified implementation)
    async fn cobyla_update(
        &mut self,
        params: QAOAParameters,
        gradient: ParameterGradient,
        iteration: usize,
    ) -> Result<QAOAParameters, AnalyzerError> {
        // Simplified COBYLA - would need proper implementation in production
        self.gradient_descent_update(params, gradient, iteration).await
    }
}

impl LearningRateSchedule {
    /// Get learning rate for current iteration
    pub fn get_rate(&self, iteration: usize) -> f64 {
        let decay_steps = self.decay_steps.max(1);
        let decay_factor = self.decay_rate.powf((iteration / decay_steps) as f64);
        (self.initial_rate * decay_factor).max(self.minimum_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    
    #[tokio::test]
    async fn test_qaoa_engine_creation() {
        let config = QuantumConfig::default();
        let engine = QAOAEngine::new(config).await;
        assert!(engine.is_ok());
    }
    
    #[tokio::test]
    async fn test_qaoa_parameter_initialization() {
        let config = QuantumConfig::default();
        let engine = QAOAEngine::new(config).await.unwrap();
        
        let problem = QuantumProblemParameters {
            num_qubits: 4,
            cost_matrix: DMatrix::identity(4, 4),
            constraint_matrices: vec![],
            optimization_objective: OptimizationObjective::MaximizeReturn,
            penalty_coefficients: vec![],
        };
        
        let params = engine.initialize_parameters(&problem).await;
        assert!(params.is_ok());
        
        let params = params.unwrap();
        assert_eq!(params.beta.len(), engine.config.qaoa_layers);
        assert_eq!(params.gamma.len(), engine.config.qaoa_layers);
    }
    
    #[tokio::test]
    async fn test_qaoa_circuit_building() {
        let config = QuantumConfig::default();
        let engine = QAOAEngine::new(config).await.unwrap();
        
        let problem = QuantumProblemParameters {
            num_qubits: 2,
            cost_matrix: DMatrix::identity(2, 2),
            constraint_matrices: vec![],
            optimization_objective: OptimizationObjective::MaximizeReturn,
            penalty_coefficients: vec![],
        };
        
        let params = engine.initialize_parameters(&problem).await.unwrap();
        let circuit = engine.build_qaoa_circuit(&problem, &params).await;
        
        assert!(circuit.is_ok());
        let circuit = circuit.unwrap();
        assert_eq!(circuit.num_qubits, 2);
        assert!(circuit.instructions.len() > 0);
    }
    
    #[tokio::test]
    async fn test_qaoa_circuit_execution() {
        let config = QuantumConfig::default();
        let engine = QAOAEngine::new(config).await.unwrap();
        
        let mut circuit = QuantumCircuit::new("test_qaoa".to_string(), 2);
        circuit.add_hadamard(0).unwrap();
        circuit.add_hadamard(1).unwrap();
        circuit.add_cnot(0, 1).unwrap();
        
        let result = engine.execute_qaoa_circuit(&circuit).await;
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert_eq!(result.probabilities.len(), 4);
        assert!(result.fidelity > 0.9);
    }
    
    #[test]
    fn test_learning_rate_schedule() {
        let schedule = LearningRateSchedule::default();
        
        let rate0 = schedule.get_rate(0);
        let rate10 = schedule.get_rate(10);
        let rate100 = schedule.get_rate(100);
        
        assert!(rate0 > rate10);
        assert!(rate10 > rate100);
        assert!(rate100 >= schedule.minimum_rate);
    }
    
    #[tokio::test]
    async fn test_parameter_gradient_calculation() {
        let config = QuantumConfig::default();
        let engine = QAOAEngine::new(config).await.unwrap();
        
        let gradient = ParameterGradient {
            beta_gradient: vec![0.1, 0.2, 0.3],
            gamma_gradient: vec![0.4, 0.5, 0.6],
        };
        
        let norm = engine.calculate_gradient_norm(&gradient);
        assert!(norm > 0.0);
        
        let expected_norm = (0.1*0.1 + 0.2*0.2 + 0.3*0.3 + 0.4*0.4 + 0.5*0.5 + 0.6*0.6).sqrt();
        assert!((norm - expected_norm).abs() < 1e-10);
    }
    
    #[tokio::test]
    async fn test_classical_optimizer_update() {
        let mut optimizer = ClassicalOptimizer {
            optimizer_type: super::super::ClassicalOptimizer::Adam,
            step_size: 0.1,
            momentum: 0.9,
            velocity: HashMap::new(),
            learning_rate_schedule: LearningRateSchedule::default(),
        };
        
        let params = QAOAParameters {
            beta: vec![0.5, 0.5],
            gamma: vec![1.0, 1.0],
            layers: 2,
            bounds: ParameterBounds::default(),
            step_size: 0.1,
            tolerance: 1e-6,
        };
        
        let gradient = ParameterGradient {
            beta_gradient: vec![0.1, 0.2],
            gamma_gradient: vec![0.3, 0.4],
        };
        
        let updated_params = optimizer.update_parameters(params, gradient, 0).await;
        assert!(updated_params.is_ok());
        
        let updated_params = updated_params.unwrap();
        assert_eq!(updated_params.beta.len(), 2);
        assert_eq!(updated_params.gamma.len(), 2);
    }
}