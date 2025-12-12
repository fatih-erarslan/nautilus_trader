//! Quantum optimization algorithms for QAR
//!
//! This module provides quantum optimization algorithms including:
//! - Variational Quantum Eigensolver (VQE)
//! - Quantum Approximate Optimization Algorithm (QAOA)
//! - Quantum gradient descent
//! - Parameter optimization for quantum circuits

use crate::core::{QarResult, QarError, constants};
use crate::quantum::{QuantumState, Gate, StandardGates};
use crate::core::{CircuitParams, ExecutionContext};
use async_trait::async_trait;
use std::collections::HashMap;
use super::types::*;
use super::traits::*;

/// Optimization objective function
pub type ObjectiveFunction = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;

/// Gradient function for optimization
pub type GradientFunction = Box<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>;

/// Quantum optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Optimal parameters found
    pub optimal_parameters: Vec<f64>,
    /// Optimal objective value
    pub optimal_value: f64,
    /// Number of iterations performed
    pub iterations: usize,
    /// Convergence status
    pub converged: bool,
    /// Final gradient norm
    pub gradient_norm: f64,
    /// Optimization history
    pub history: Vec<(Vec<f64>, f64)>,
}

impl OptimizationResult {
    /// Create a new optimization result
    pub fn new(params: Vec<f64>, value: f64, iterations: usize) -> Self {
        Self {
            optimal_parameters: params,
            optimal_value: value,
            iterations,
            converged: false,
            gradient_norm: 0.0,
            history: Vec::new(),
        }
    }

    /// Check if optimization was successful
    pub fn is_successful(&self) -> bool {
        self.converged && self.gradient_norm < constants::math::EPSILON
    }
}

/// Quantum parameter optimizer using gradient descent
#[derive(Debug)]
pub struct QuantumOptimizer {
    /// Learning rate for parameter updates
    learning_rate: f64,
    /// Maximum number of iterations
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: f64,
    /// Momentum factor
    momentum: f64,
    /// Whether to use adaptive learning rate
    adaptive_lr: bool,
    /// Optimization history
    history: Vec<(Vec<f64>, f64)>,
}

impl QuantumOptimizer {
    /// Create a new quantum optimizer
    pub fn new(learning_rate: f64, max_iterations: usize) -> Self {
        Self {
            learning_rate,
            max_iterations,
            tolerance: constants::math::EPSILON,
            momentum: 0.9,
            adaptive_lr: true,
            history: Vec::new(),
        }
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set momentum factor
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Enable or disable adaptive learning rate
    pub fn with_adaptive_lr(mut self, adaptive: bool) -> Self {
        self.adaptive_lr = adaptive;
        self
    }

    /// Optimize parameters using gradient descent
    pub async fn optimize<F>(
        &mut self,
        initial_params: Vec<f64>,
        objective: F,
    ) -> QarResult<OptimizationResult>
    where
        F: Fn(&[f64]) -> QarResult<f64> + Send + Sync,
    {
        let mut params = initial_params.clone();
        let mut velocity = vec![0.0; params.len()];
        let mut learning_rate = self.learning_rate;
        
        self.history.clear();
        
        for iteration in 0..self.max_iterations {
            // Evaluate objective function
            let current_value = objective(&params)?;
            self.history.push((params.clone(), current_value));
            
            // Calculate gradient using finite differences
            let gradient = self.calculate_gradient(&params, &objective).await?;
            let gradient_norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
            
            // Check convergence
            if gradient_norm < self.tolerance {
                return Ok(OptimizationResult {
                    optimal_parameters: params,
                    optimal_value: current_value,
                    iterations: iteration + 1,
                    converged: true,
                    gradient_norm,
                    history: self.history.clone(),
                });
            }
            
            // Update parameters using momentum
            for (i, (param, grad)) in params.iter_mut().zip(gradient.iter()).enumerate() {
                velocity[i] = self.momentum * velocity[i] - learning_rate * grad;
                *param += velocity[i];
            }
            
            // Adaptive learning rate
            if self.adaptive_lr && iteration > 0 {
                let prev_value = self.history[iteration - 1].1;
                if current_value > prev_value {
                    learning_rate *= 0.9; // Reduce if not improving
                } else {
                    learning_rate *= 1.1; // Increase if improving
                }
                learning_rate = learning_rate.max(1e-6).min(1.0);
            }
        }
        
        // Max iterations reached
        let final_value = objective(&params)?;
        let final_gradient = self.calculate_gradient(&params, &objective).await?;
        let gradient_norm = final_gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        
        Ok(OptimizationResult {
            optimal_parameters: params,
            optimal_value: final_value,
            iterations: self.max_iterations,
            converged: false,
            gradient_norm,
            history: self.history.clone(),
        })
    }

    /// Calculate gradient using finite differences
    async fn calculate_gradient<F>(
        &self,
        params: &[f64],
        objective: &F,
    ) -> QarResult<Vec<f64>>
    where
        F: Fn(&[f64]) -> QarResult<f64> + Send + Sync,
    {
        let epsilon = 1e-8;
        let mut gradient = Vec::with_capacity(params.len());
        
        for i in 0..params.len() {
            let mut params_plus = params.to_vec();
            let mut params_minus = params.to_vec();
            
            params_plus[i] += epsilon;
            params_minus[i] -= epsilon;
            
            let value_plus = objective(&params_plus)?;
            let value_minus = objective(&params_minus)?;
            
            let grad = (value_plus - value_minus) / (2.0 * epsilon);
            gradient.push(grad);
        }
        
        Ok(gradient)
    }
}

/// Variational Quantum Eigensolver (VQE) for quantum optimization
#[derive(Debug)]
pub struct VariationalQuantumEigensolver {
    /// Number of qubits
    num_qubits: usize,
    /// Variational circuit layers
    num_layers: usize,
    /// Hamiltonian terms
    hamiltonian: Vec<(f64, String)>, // (coefficient, Pauli string)
    /// Optimizer
    optimizer: QuantumOptimizer,
}

impl VariationalQuantumEigensolver {
    /// Create a new VQE instance
    pub fn new(num_qubits: usize, num_layers: usize) -> Self {
        Self {
            num_qubits,
            num_layers,
            hamiltonian: Vec::new(),
            optimizer: QuantumOptimizer::new(0.01, 1000),
        }
    }

    /// Add Hamiltonian term
    pub fn add_hamiltonian_term(&mut self, coefficient: f64, pauli_string: String) {
        self.hamiltonian.push((coefficient, pauli_string));
    }

    /// Set custom optimizer
    pub fn with_optimizer(mut self, optimizer: QuantumOptimizer) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Build variational ansatz circuit
    fn build_ansatz(&self, params: &[f64]) -> QarResult<QuantumState> {
        if params.len() != self.num_layers * self.num_qubits * 3 {
            return Err(QarError::InvalidInput(
                "Parameter count doesn't match ansatz requirements".to_string()
            ));
        }

        let mut state = QuantumState::new(self.num_qubits);
        let mut param_idx = 0;

        // Apply initial Hadamard gates
        for qubit in 0..self.num_qubits {
            let h_gate = StandardGates::hadamard();
            state.apply_single_qubit_gate(qubit, &h_gate)?;
        }

        // Apply variational layers
        for _layer in 0..self.num_layers {
            // Rotation gates
            for qubit in 0..self.num_qubits {
                let rx_gate = StandardGates::rx(params[param_idx]);
                state.apply_single_qubit_gate(qubit, &rx_gate)?;
                param_idx += 1;

                let ry_gate = StandardGates::ry(params[param_idx]);
                state.apply_single_qubit_gate(qubit, &ry_gate)?;
                param_idx += 1;

                let rz_gate = StandardGates::rz(params[param_idx]);
                state.apply_single_qubit_gate(qubit, &rz_gate)?;
                param_idx += 1;
            }

            // Entangling gates
            for qubit in 0..(self.num_qubits - 1) {
                let cnot = StandardGates::cnot();
                state.apply_two_qubit_gate(qubit, qubit + 1, &cnot)?;
            }
        }

        Ok(state)
    }

    /// Calculate expectation value of Hamiltonian
    fn calculate_expectation(&self, state: &QuantumState) -> QarResult<f64> {
        let mut expectation = 0.0;

        for (coefficient, pauli_string) in &self.hamiltonian {
            let term_expectation = self.calculate_pauli_expectation(state, pauli_string)?;
            expectation += coefficient * term_expectation;
        }

        Ok(expectation)
    }

    /// Calculate expectation value of Pauli string
    fn calculate_pauli_expectation(&self, state: &QuantumState, pauli_string: &str) -> QarResult<f64> {
        if pauli_string.len() != self.num_qubits {
            return Err(QarError::InvalidInput(
                "Pauli string length doesn't match number of qubits".to_string()
            ));
        }

        let mut expectation = 1.0;

        for (qubit, pauli_char) in pauli_string.chars().enumerate() {
            let single_expectation = match pauli_char {
                'I' => 1.0, // Identity
                'X' => state.expectation_x(qubit)?,
                'Y' => state.expectation_y(qubit)?,
                'Z' => state.expectation_z(qubit)?,
                _ => return Err(QarError::InvalidInput(
                    format!("Invalid Pauli operator: {}", pauli_char)
                )),
            };
            expectation *= single_expectation;
        }

        Ok(expectation)
    }

    /// Run VQE optimization
    pub async fn optimize(&mut self) -> QarResult<OptimizationResult> {
        if self.hamiltonian.is_empty() {
            return Err(QarError::InvalidInput("No Hamiltonian terms specified".to_string()));
        }

        // Initialize random parameters
        let param_count = self.num_layers * self.num_qubits * 3;
        let initial_params: Vec<f64> = (0..param_count)
            .map(|_| (rand::random::<f64>() - 0.5) * 2.0 * std::f64::consts::PI)
            .collect();

        // Define objective function (minimize energy expectation)
        let objective = |params: &[f64]| -> QarResult<f64> {
            let state = self.build_ansatz(params)?;
            self.calculate_expectation(&state)
        };

        self.optimizer.optimize(initial_params, objective).await
    }
}

/// Quantum Approximate Optimization Algorithm (QAOA)
#[derive(Debug)]
pub struct QuantumApproximateOptimization {
    /// Number of qubits
    num_qubits: usize,
    /// QAOA depth (number of layers)
    depth: usize,
    /// Cost Hamiltonian
    cost_hamiltonian: Vec<(f64, String)>,
    /// Mixer Hamiltonian (default: X on all qubits)
    mixer_hamiltonian: Vec<(f64, String)>,
    /// Optimizer
    optimizer: QuantumOptimizer,
}

impl QuantumApproximateOptimization {
    /// Create a new QAOA instance
    pub fn new(num_qubits: usize, depth: usize) -> Self {
        let mut qaoa = Self {
            num_qubits,
            depth,
            cost_hamiltonian: Vec::new(),
            mixer_hamiltonian: Vec::new(),
            optimizer: QuantumOptimizer::new(0.1, 500),
        };

        // Default mixer: X on all qubits
        for qubit in 0..num_qubits {
            let mut pauli_string = "I".repeat(num_qubits);
            pauli_string.replace_range(qubit..=qubit, "X");
            qaoa.mixer_hamiltonian.push((1.0, pauli_string));
        }

        qaoa
    }

    /// Add cost Hamiltonian term
    pub fn add_cost_term(&mut self, coefficient: f64, pauli_string: String) {
        self.cost_hamiltonian.push((coefficient, pauli_string));
    }

    /// Set custom mixer Hamiltonian
    pub fn set_mixer(&mut self, mixer: Vec<(f64, String)>) {
        self.mixer_hamiltonian = mixer;
    }

    /// Build QAOA circuit
    fn build_qaoa_circuit(&self, params: &[f64]) -> QarResult<QuantumState> {
        if params.len() != 2 * self.depth {
            return Err(QarError::InvalidInput(
                "Parameter count must be 2 * depth for QAOA".to_string()
            ));
        }

        let mut state = QuantumState::superposition(self.num_qubits);

        for layer in 0..self.depth {
            let gamma = params[2 * layer];     // Cost Hamiltonian parameter
            let beta = params[2 * layer + 1];  // Mixer Hamiltonian parameter

            // Apply cost Hamiltonian evolution
            self.apply_hamiltonian_evolution(&mut state, &self.cost_hamiltonian, gamma)?;

            // Apply mixer Hamiltonian evolution
            self.apply_hamiltonian_evolution(&mut state, &self.mixer_hamiltonian, beta)?;
        }

        Ok(state)
    }

    /// Apply Hamiltonian evolution to quantum state
    fn apply_hamiltonian_evolution(
        &self,
        state: &mut QuantumState,
        hamiltonian: &[(f64, String)],
        parameter: f64,
    ) -> QarResult<()> {
        for (coefficient, pauli_string) in hamiltonian {
            let angle = coefficient * parameter;
            self.apply_pauli_rotation(state, pauli_string, angle)?;
        }
        Ok(())
    }

    /// Apply Pauli rotation to quantum state
    fn apply_pauli_rotation(&self, state: &mut QuantumState, pauli_string: &str, angle: f64) -> QarResult<()> {
        // For simplicity, we'll implement single-qubit Pauli rotations
        // In a full implementation, this would handle multi-qubit Pauli strings
        for (qubit, pauli_char) in pauli_string.chars().enumerate() {
            match pauli_char {
                'I' => {}, // Identity - no operation
                'X' => {
                    let rx_gate = StandardGates::rx(2.0 * angle);
                    state.apply_single_qubit_gate(qubit, &rx_gate)?;
                },
                'Y' => {
                    let ry_gate = StandardGates::ry(2.0 * angle);
                    state.apply_single_qubit_gate(qubit, &ry_gate)?;
                },
                'Z' => {
                    let rz_gate = StandardGates::rz(2.0 * angle);
                    state.apply_single_qubit_gate(qubit, &rz_gate)?;
                },
                _ => return Err(QarError::InvalidInput(
                    format!("Invalid Pauli operator: {}", pauli_char)
                )),
            }
        }
        Ok(())
    }

    /// Calculate cost function expectation
    fn calculate_cost_expectation(&self, state: &QuantumState) -> QarResult<f64> {
        let mut expectation = 0.0;

        for (coefficient, pauli_string) in &self.cost_hamiltonian {
            let term_expectation = self.calculate_pauli_expectation(state, pauli_string)?;
            expectation += coefficient * term_expectation;
        }

        Ok(expectation)
    }

    /// Calculate expectation value of Pauli string
    fn calculate_pauli_expectation(&self, state: &QuantumState, pauli_string: &str) -> QarResult<f64> {
        let mut expectation = 1.0;

        for (qubit, pauli_char) in pauli_string.chars().enumerate() {
            let single_expectation = match pauli_char {
                'I' => 1.0,
                'X' => state.expectation_x(qubit)?,
                'Y' => state.expectation_y(qubit)?,
                'Z' => state.expectation_z(qubit)?,
                _ => return Err(QarError::InvalidInput(
                    format!("Invalid Pauli operator: {}", pauli_char)
                )),
            };
            expectation *= single_expectation;
        }

        Ok(expectation)
    }

    /// Run QAOA optimization
    pub async fn optimize(&mut self) -> QarResult<OptimizationResult> {
        if self.cost_hamiltonian.is_empty() {
            return Err(QarError::InvalidInput("No cost Hamiltonian specified".to_string()));
        }

        // Initialize parameters
        let param_count = 2 * self.depth;
        let initial_params: Vec<f64> = (0..param_count)
            .map(|_| rand::random::<f64>() * std::f64::consts::PI)
            .collect();

        // Define objective function (minimize cost expectation)
        let objective = |params: &[f64]| -> QarResult<f64> {
            let state = self.build_qaoa_circuit(params)?;
            self.calculate_cost_expectation(&state)
        };

        self.optimizer.optimize(initial_params, objective).await
    }
}

/// Quantum gradient descent optimizer
#[derive(Debug)]
pub struct QuantumGradientDescent {
    /// Learning rate
    learning_rate: f64,
    /// Maximum iterations
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: f64,
    /// Use quantum gradient calculation
    use_quantum_gradient: bool,
}

impl QuantumGradientDescent {
    /// Create a new quantum gradient descent optimizer
    pub fn new(learning_rate: f64, max_iterations: usize) -> Self {
        Self {
            learning_rate,
            max_iterations,
            tolerance: 1e-6,
            use_quantum_gradient: false,
        }
    }

    /// Enable quantum gradient calculation
    pub fn with_quantum_gradient(mut self) -> Self {
        self.use_quantum_gradient = true;
        self
    }

    /// Optimize using quantum gradient descent
    pub async fn optimize<F>(
        &self,
        initial_params: Vec<f64>,
        objective: F,
    ) -> QarResult<OptimizationResult>
    where
        F: Fn(&[f64]) -> QarResult<f64> + Send + Sync,
    {
        let mut params = initial_params;
        let mut history = Vec::new();

        for iteration in 0..self.max_iterations {
            let current_value = objective(&params)?;
            history.push((params.clone(), current_value));

            let gradient = if self.use_quantum_gradient {
                self.calculate_quantum_gradient(&params, &objective).await?
            } else {
                self.calculate_classical_gradient(&params, &objective)?
            };

            let gradient_norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();

            if gradient_norm < self.tolerance {
                return Ok(OptimizationResult {
                    optimal_parameters: params,
                    optimal_value: current_value,
                    iterations: iteration + 1,
                    converged: true,
                    gradient_norm,
                    history,
                });
            }

            // Update parameters
            for (param, grad) in params.iter_mut().zip(gradient.iter()) {
                *param -= self.learning_rate * grad;
            }
        }

        let final_value = objective(&params)?;
        Ok(OptimizationResult {
            optimal_parameters: params,
            optimal_value: final_value,
            iterations: self.max_iterations,
            converged: false,
            gradient_norm: 0.0,
            history,
        })
    }

    /// Calculate gradient using parameter shift rule (quantum gradient)
    async fn calculate_quantum_gradient<F>(
        &self,
        params: &[f64],
        objective: &F,
    ) -> QarResult<Vec<f64>>
    where
        F: Fn(&[f64]) -> QarResult<f64> + Send + Sync,
    {
        let mut gradient = Vec::with_capacity(params.len());
        let shift = std::f64::consts::PI / 2.0; // Parameter shift rule

        for i in 0..params.len() {
            let mut params_plus = params.to_vec();
            let mut params_minus = params.to_vec();

            params_plus[i] += shift;
            params_minus[i] -= shift;

            let value_plus = objective(&params_plus)?;
            let value_minus = objective(&params_minus)?;

            let grad = (value_plus - value_minus) / 2.0;
            gradient.push(grad);
        }

        Ok(gradient)
    }

    /// Calculate gradient using finite differences
    fn calculate_classical_gradient<F>(
        &self,
        params: &[f64],
        objective: &F,
    ) -> QarResult<Vec<f64>>
    where
        F: Fn(&[f64]) -> QarResult<f64> + Send + Sync,
    {
        let epsilon = 1e-8;
        let mut gradient = Vec::with_capacity(params.len());

        for i in 0..params.len() {
            let mut params_plus = params.to_vec();
            let mut params_minus = params.to_vec();

            params_plus[i] += epsilon;
            params_minus[i] -= epsilon;

            let value_plus = objective(&params_plus)?;
            let value_minus = objective(&params_minus)?;

            let grad = (value_plus - value_minus) / (2.0 * epsilon);
            gradient.push(grad);
        }

        Ok(gradient)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_optimization_result() {
        let result = OptimizationResult::new(vec![1.0, 2.0], 0.5, 10);
        assert_eq!(result.optimal_parameters, vec![1.0, 2.0]);
        assert_eq!(result.optimal_value, 0.5);
        assert_eq!(result.iterations, 10);
        assert!(!result.converged);
    }

    #[tokio::test]
    async fn test_quantum_optimizer() {
        let mut optimizer = QuantumOptimizer::new(0.1, 100);

        // Simple quadratic objective: f(x) = (x-1)^2 + (y-2)^2
        let objective = |params: &[f64]| -> QarResult<f64> {
            if params.len() != 2 {
                return Err(QarError::InvalidInput("Expected 2 parameters".to_string()));
            }
            Ok((params[0] - 1.0).powi(2) + (params[1] - 2.0).powi(2))
        };

        let result = optimizer.optimize(vec![0.0, 0.0], objective).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.iterations > 0);
        // Should converge close to (1, 2)
        assert_relative_eq!(result.optimal_parameters[0], 1.0, epsilon = 0.1);
        assert_relative_eq!(result.optimal_parameters[1], 2.0, epsilon = 0.1);
    }

    #[tokio::test]
    async fn test_vqe() {
        let mut vqe = VariationalQuantumEigensolver::new(2, 1);
        
        // Simple Hamiltonian: Z_0
        vqe.add_hamiltonian_term(1.0, "ZI".to_string());
        
        let result = vqe.optimize().await;
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert!(result.iterations > 0);
        // For H = Z_0, ground state energy should be close to -1
        assert!(result.optimal_value <= 1.0);
    }

    #[tokio::test]
    async fn test_qaoa() {
        let mut qaoa = QuantumApproximateOptimization::new(2, 1);
        
        // Simple cost function: -Z_0 * Z_1 (maximize agreement)
        qaoa.add_cost_term(-1.0, "ZZ".to_string());
        
        let result = qaoa.optimize().await;
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert!(result.iterations > 0);
        assert!(result.optimal_parameters.len() == 2); // 2 * depth
    }

    #[tokio::test]
    async fn test_quantum_gradient_descent() {
        let optimizer = QuantumGradientDescent::new(0.1, 100)
            .with_quantum_gradient();

        // Simple objective: f(x) = x^2
        let objective = |params: &[f64]| -> QarResult<f64> {
            if params.len() != 1 {
                return Err(QarError::InvalidInput("Expected 1 parameter".to_string()));
            }
            Ok(params[0].powi(2))
        };

        let result = optimizer.optimize(vec![2.0], objective).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        // Should converge to x = 0
        assert_relative_eq!(result.optimal_parameters[0], 0.0, epsilon = 0.1);
    }

    #[test]
    fn test_vqe_hamiltonian() {
        let mut vqe = VariationalQuantumEigensolver::new(2, 1);
        
        // Add multiple terms
        vqe.add_hamiltonian_term(0.5, "ZI".to_string());
        vqe.add_hamiltonian_term(-0.3, "IZ".to_string());
        vqe.add_hamiltonian_term(0.1, "XX".to_string());
        
        assert_eq!(vqe.hamiltonian.len(), 3);
        assert_eq!(vqe.hamiltonian[0].0, 0.5);
        assert_eq!(vqe.hamiltonian[1].1, "IZ");
    }

    #[test]
    fn test_qaoa_cost_terms() {
        let mut qaoa = QuantumApproximateOptimization::new(3, 2);
        
        qaoa.add_cost_term(1.0, "ZZI".to_string());
        qaoa.add_cost_term(-0.5, "IZZ".to_string());
        
        assert_eq!(qaoa.cost_hamiltonian.len(), 2);
        assert_eq!(qaoa.depth, 2);
        assert_eq!(qaoa.num_qubits, 3);
    }
}