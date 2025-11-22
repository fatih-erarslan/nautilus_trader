//! Quantum-enhanced optimization algorithms
//!
//! This module provides optimization algorithms inspired by quantum computing,
//! including QAOA-inspired optimizers and variational quantum eigensolvers.

use crate::{Circuit, Result, QuantumError, Operator};
use serde::{Deserialize, Serialize};

/// Trait for quantum-enhanced optimizers
pub trait Optimizer {
    /// Optimize the given objective function
    fn optimize<F>(&mut self, objective: F, initial_params: &[f64]) -> Result<OptimizationResult>
    where
        F: Fn(&[f64]) -> f64;
        
    /// Get optimizer configuration
    fn config(&self) -> &OptimizerConfig;
}

/// Trait for variational quantum optimizers  
/// Note: Made object-safe by removing generic methods
pub trait VariationalOptimizer {
    /// Optimize the given objective function
    fn optimize(&mut self, objective: fn(&[f64]) -> f64, initial_params: &[f64]) -> Result<OptimizationResult>;
        
    /// Get optimizer configuration
    fn config(&self) -> &OptimizerConfig;
    /// Optimize a variational quantum circuit
    fn optimize_circuit(
        &mut self,
        circuit: &Circuit,
        cost_function: fn(&Circuit, &[f64]) -> Result<f64>,
        initial_params: &[f64],
    ) -> Result<OptimizationResult>;
    
    /// Optimize with gradient information
    fn optimize_with_gradients(
        &mut self,
        objective: fn(&[f64]) -> f64,
        gradient: fn(&[f64]) -> Vec<f64>,
        initial_params: &[f64],
    ) -> Result<OptimizationResult>;
}

/// Optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub learning_rate: f64,
    pub momentum: f64,
    pub adaptive_lr: bool,
    pub verbose: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-8,
            learning_rate: 0.01,
            momentum: 0.9,
            adaptive_lr: false,
            verbose: false,
        }
    }
}

/// Optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Optimal parameters found
    pub optimal_params: Vec<f64>,
    /// Optimal function value
    pub optimal_value: f64,
    /// Number of iterations performed
    pub iterations: usize,
    /// Function evaluations count
    pub function_evaluations: usize,
    /// Convergence status
    pub converged: bool,
    /// Optimization history
    pub history: OptimizationHistory,
}

/// Optimization history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationHistory {
    pub values: Vec<f64>,
    pub parameters: Vec<Vec<f64>>,
    pub gradients: Vec<Vec<f64>>,
    pub learning_rates: Vec<f64>,
}

impl OptimizationHistory {
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            parameters: Vec::new(),
            gradients: Vec::new(),
            learning_rates: Vec::new(),
        }
    }
    
    pub fn add_step(
        &mut self,
        value: f64,
        params: Vec<f64>,
        gradient: Vec<f64>,
        lr: f64,
    ) {
        self.values.push(value);
        self.parameters.push(params);
        self.gradients.push(gradient);
        self.learning_rates.push(lr);
    }
}

/// Quantum Approximate Optimization Algorithm (QAOA) inspired optimizer
pub struct QAOAOptimizer {
    config: OptimizerConfig,
    mixer_layers: usize,
    cost_layers: usize,
    beta_params: Vec<f64>,
    gamma_params: Vec<f64>,
}

impl QAOAOptimizer {
    /// Create a new QAOA optimizer
    pub fn new(config: OptimizerConfig, layers: usize) -> Self {
        Self {
            config,
            mixer_layers: layers,
            cost_layers: layers,
            beta_params: vec![0.1; layers],
            gamma_params: vec![0.1; layers],
        }
    }
    
    /// Set the QAOA parameters
    pub fn set_qaoa_params(&mut self, beta: Vec<f64>, gamma: Vec<f64>) -> Result<()> {
        if beta.len() != self.mixer_layers || gamma.len() != self.cost_layers {
            return Err(QuantumError::InvalidParameter(
                "QAOA parameter dimensions don't match number of layers".to_string()
            ));
        }
        self.beta_params = beta;
        self.gamma_params = gamma;
        Ok(())
    }
    
    /// Build QAOA ansatz circuit
    pub fn build_qaoa_circuit(&self, n_qubits: usize, cost_hamiltonian: &CostHamiltonian) -> Result<Circuit> {
        let mut circuit = Circuit::new(n_qubits);
        
        // Initialize with uniform superposition (Hadamard on all qubits)
        for qubit in 0..n_qubits {
            circuit.add_gate(Box::new(crate::gates::Hadamard::new(qubit)))?;
        }
        
        // QAOA layers
        for layer in 0..self.cost_layers {
            // Apply cost Hamiltonian evolution
            self.apply_cost_evolution(&mut circuit, cost_hamiltonian, self.gamma_params[layer])?;
            
            // Apply mixer Hamiltonian evolution (X rotations)
            for qubit in 0..n_qubits {
                circuit.add_gate(Box::new(crate::gates::RX::new(qubit, 2.0 * self.beta_params[layer])))?;
            }
        }
        
        Ok(circuit)
    }
    
    /// Apply cost Hamiltonian evolution to the circuit
    fn apply_cost_evolution(
        &self,
        circuit: &mut Circuit,
        cost_hamiltonian: &CostHamiltonian,
        gamma: f64,
    ) -> Result<()> {
        // Apply ZZ interactions for cost Hamiltonian
        for &(i, j, weight) in &cost_hamiltonian.interactions {
            // ZZ interaction: exp(-i*gamma*weight*Z_i*Z_j)
            // Can be decomposed as CNOT gates and RZ rotations
            circuit.add_gate(Box::new(crate::gates::CNOT::new(i, j)))?;
            circuit.add_gate(Box::new(crate::gates::RZ::new(j, 2.0 * gamma * weight)))?;
            circuit.add_gate(Box::new(crate::gates::CNOT::new(i, j)))?;
        }
        
        // Apply local Z fields
        for &(i, weight) in &cost_hamiltonian.local_fields {
            circuit.add_gate(Box::new(crate::gates::RZ::new(i, 2.0 * gamma * weight)))?;
        }
        
        Ok(())
    }
}

impl Optimizer for QAOAOptimizer {
    fn optimize<F>(&mut self, objective: F, initial_params: &[f64]) -> Result<OptimizationResult>
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut params = initial_params.to_vec();
        let mut history = OptimizationHistory::new();
        let mut best_value = objective(&params);
        let mut best_params = params.clone();
        let mut function_evaluations = 1;
        
        for iteration in 0..self.config.max_iterations {
            // Simple gradient-free optimization using random perturbations
            let mut improved = false;
            
            for i in 0..params.len() {
                let original_val = params[i];
                let perturbation = 0.1 * (2.0 * rand::random::<f64>() - 1.0);
                
                params[i] = original_val + perturbation;
                let new_value = objective(&params);
                function_evaluations += 1;
                
                if new_value < best_value {
                    best_value = new_value;
                    best_params = params.clone();
                    improved = true;
                } else {
                    params[i] = original_val; // Revert if no improvement
                }
            }
            
            // Record history
            history.add_step(
                best_value,
                best_params.clone(),
                vec![0.0; params.len()], // No gradient computation
                self.config.learning_rate,
            );
            
            if self.config.verbose && iteration % 10 == 0 {
                println!("QAOA Iteration {}: f = {:.6}", iteration, best_value);
            }
            
            // Convergence check
            if !improved && history.values.len() > 1 {
                let improvement = (history.values[history.values.len() - 2] - best_value).abs();
                if improvement < self.config.tolerance {
                    break;
                }
            }
        }
        
        Ok(OptimizationResult {
            optimal_params: best_params,
            optimal_value: best_value,
            iterations: history.values.len(),
            function_evaluations,
            converged: true,
            history,
        })
    }
    
    fn config(&self) -> &OptimizerConfig {
        &self.config
    }
}

/// Cost Hamiltonian definition for QAOA
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostHamiltonian {
    /// ZZ interaction terms: (qubit_i, qubit_j, weight)
    pub interactions: Vec<(usize, usize, f64)>,
    /// Local Z field terms: (qubit, weight)
    pub local_fields: Vec<(usize, f64)>,
}

impl CostHamiltonian {
    /// Create a new cost Hamiltonian
    pub fn new() -> Self {
        Self {
            interactions: Vec::new(),
            local_fields: Vec::new(),
        }
    }
    
    /// Add a ZZ interaction term
    pub fn add_zz_interaction(&mut self, i: usize, j: usize, weight: f64) {
        self.interactions.push((i, j, weight));
    }
    
    /// Add a local Z field term
    pub fn add_z_field(&mut self, qubit: usize, weight: f64) {
        self.local_fields.push((qubit, weight));
    }
    
    /// Create Max-Cut Hamiltonian for a graph
    pub fn max_cut(edges: &[(usize, usize)]) -> Self {
        let mut hamiltonian = Self::new();
        for &(i, j) in edges {
            hamiltonian.add_zz_interaction(i, j, -0.5); // Want to minimize, so negative weight
        }
        hamiltonian
    }
}

/// Variational Quantum Eigensolver (VQE) optimizer
pub struct VQEOptimizer {
    config: OptimizerConfig,
    hamiltonian: Operator,
}

impl VQEOptimizer {
    /// Create a new VQE optimizer
    pub fn new(config: OptimizerConfig, hamiltonian: Operator) -> Self {
        Self {
            config,
            hamiltonian,
        }
    }
    
    /// Compute expectation value of the Hamiltonian
    pub fn compute_expectation(&self, circuit: &Circuit, params: &[f64]) -> Result<f64> {
        circuit.expectation_value_with_parameters(&self.hamiltonian, params)
    }
}

impl Optimizer for VQEOptimizer {
    fn optimize<F>(&mut self, objective: F, initial_params: &[f64]) -> Result<OptimizationResult>
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut params = initial_params.to_vec();
        let mut history = OptimizationHistory::new();
        let mut velocity = vec![0.0; params.len()]; // For momentum
        let mut function_evaluations = 0;
        
        for iteration in 0..self.config.max_iterations {
            let current_value = objective(&params);
            function_evaluations += 1;
            
            // Numerical gradient computation
            let mut gradient = vec![0.0; params.len()];
            let h = 1e-8;
            
            for i in 0..params.len() {
                params[i] += h;
                let f_plus = objective(&params);
                params[i] -= 2.0 * h;
                let f_minus = objective(&params);
                params[i] += h; // Restore
                
                gradient[i] = (f_plus - f_minus) / (2.0 * h);
                function_evaluations += 2;
            }
            
            // Gradient descent with momentum
            for i in 0..params.len() {
                velocity[i] = self.config.momentum * velocity[i] - self.config.learning_rate * gradient[i];
                params[i] += velocity[i];
            }
            
            // Record history
            history.add_step(
                current_value,
                params.clone(),
                gradient.clone(),
                self.config.learning_rate,
            );
            
            if self.config.verbose && iteration % 10 == 0 {
                println!("VQE Iteration {}: E = {:.6}", iteration, current_value);
            }
            
            // Convergence check
            if gradient.iter().map(|g| g * g).sum::<f64>().sqrt() < self.config.tolerance {
                break;
            }
        }
        
        let final_value = objective(&params);
        function_evaluations += 1;
        
        Ok(OptimizationResult {
            optimal_params: params,
            optimal_value: final_value,
            iterations: history.values.len(),
            function_evaluations,
            converged: true,
            history,
        })
    }
    
    fn config(&self) -> &OptimizerConfig {
        &self.config
    }
}

impl VariationalOptimizer for VQEOptimizer {
    fn optimize(&mut self, objective: fn(&[f64]) -> f64, initial_params: &[f64]) -> Result<OptimizationResult> {
        let mut params = initial_params.to_vec();
        let mut history = OptimizationHistory::new();
        let mut velocity = vec![0.0; params.len()]; // For momentum
        let mut function_evaluations = 0;
        
        for iteration in 0..self.config.max_iterations {
            let current_value = objective(&params);
            function_evaluations += 1;
            
            // Numerical gradient computation
            let mut gradient = vec![0.0; params.len()];
            let h = 1e-8;
            
            for i in 0..params.len() {
                params[i] += h;
                let f_plus = objective(&params);
                params[i] -= 2.0 * h;
                let f_minus = objective(&params);
                params[i] += h; // Restore
                
                gradient[i] = (f_plus - f_minus) / (2.0 * h);
                function_evaluations += 2;
            }
            
            // Gradient descent with momentum
            for i in 0..params.len() {
                velocity[i] = self.config.momentum * velocity[i] - self.config.learning_rate * gradient[i];
                params[i] += velocity[i];
            }
            
            // Record history
            history.add_step(
                current_value,
                params.clone(),
                gradient.clone(),
                self.config.learning_rate,
            );
            
            if self.config.verbose && iteration % 10 == 0 {
                println!("VQE Iteration {}: E = {:.6}", iteration, current_value);
            }
            
            // Convergence check
            if gradient.iter().map(|g| g * g).sum::<f64>().sqrt() < self.config.tolerance {
                break;
            }
        }
        
        let final_value = objective(&params);
        function_evaluations += 1;
        
        Ok(OptimizationResult {
            optimal_params: params,
            optimal_value: final_value,
            iterations: history.values.len(),
            function_evaluations,
            converged: true,
            history,
        })
    }
    
    fn config(&self) -> &OptimizerConfig {
        &self.config
    }
    
    fn optimize_circuit(
        &mut self,
        circuit: &Circuit,
        cost_function: fn(&Circuit, &[f64]) -> Result<f64>,
        initial_params: &[f64],
    ) -> Result<OptimizationResult> {
        // Direct implementation instead of using the optimize method
        let mut params = initial_params.to_vec();
        let mut history = OptimizationHistory::new();
        let mut velocity = vec![0.0; params.len()];
        let mut function_evaluations = 0;
        
        for iteration in 0..self.config.max_iterations {
            let current_value = cost_function(circuit, &params).unwrap_or(f64::INFINITY);
            function_evaluations += 1;
            
            // Numerical gradient computation
            let mut gradient = vec![0.0; params.len()];
            let h = 1e-8;
            
            for i in 0..params.len() {
                params[i] += h;
                let f_plus = cost_function(circuit, &params).unwrap_or(f64::INFINITY);
                params[i] -= 2.0 * h;
                let f_minus = cost_function(circuit, &params).unwrap_or(f64::INFINITY);
                params[i] += h; // Restore
                
                gradient[i] = (f_plus - f_minus) / (2.0 * h);
                function_evaluations += 2;
            }
            
            // Gradient descent with momentum
            for i in 0..params.len() {
                velocity[i] = self.config.momentum * velocity[i] - self.config.learning_rate * gradient[i];
                params[i] += velocity[i];
            }
            
            // Record history
            history.add_step(
                current_value,
                params.clone(),
                gradient.clone(),
                self.config.learning_rate,
            );
            
            if self.config.verbose && iteration % 10 == 0 {
                println!("VQE Iteration {}: E = {:.6}", iteration, current_value);
            }
            
            // Convergence check
            if gradient.iter().map(|g| g * g).sum::<f64>().sqrt() < self.config.tolerance {
                break;
            }
        }
        
        let final_value = cost_function(circuit, &params).unwrap_or(f64::INFINITY);
        function_evaluations += 1;
        
        Ok(OptimizationResult {
            optimal_params: params,
            optimal_value: final_value,
            iterations: history.values.len(),
            function_evaluations,
            converged: true,
            history,
        })
    }
    
    fn optimize_with_gradients(
        &mut self,
        objective: fn(&[f64]) -> f64,
        gradient: fn(&[f64]) -> Vec<f64>,
        initial_params: &[f64],
    ) -> Result<OptimizationResult> {
        let mut params = initial_params.to_vec();
        let mut history = OptimizationHistory::new();
        let mut velocity = vec![0.0; params.len()];
        let mut function_evaluations = 0;
        
        for iteration in 0..self.config.max_iterations {
            let current_value = objective(&params);
            let grad = gradient(&params);
            function_evaluations += 1;
            
            // Gradient descent with momentum
            for i in 0..params.len() {
                velocity[i] = self.config.momentum * velocity[i] - self.config.learning_rate * grad[i];
                params[i] += velocity[i];
            }
            
            // Record history
            history.add_step(
                current_value,
                params.clone(),
                grad.clone(),
                self.config.learning_rate,
            );
            
            if self.config.verbose && iteration % 10 == 0 {
                println!("Iteration {}: f = {:.6}", iteration, current_value);
            }
            
            // Convergence check
            if grad.iter().map(|g| g * g).sum::<f64>().sqrt() < self.config.tolerance {
                break;
            }
        }
        
        let final_value = objective(&params);
        function_evaluations += 1;
        
        Ok(OptimizationResult {
            optimal_params: params,
            optimal_value: final_value,
            iterations: history.values.len(),
            function_evaluations,
            converged: true,
            history,
        })
    }
}

/// Adam optimizer for quantum-enhanced machine learning
pub struct AdamOptimizer {
    config: OptimizerConfig,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    m: Vec<f64>, // First moment estimate
    v: Vec<f64>, // Second moment estimate
    t: usize,    // Time step
}

impl AdamOptimizer {
    /// Create a new Adam optimizer
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }
    
    /// Set Adam hyperparameters
    pub fn set_adam_params(&mut self, beta1: f64, beta2: f64, epsilon: f64) {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self.epsilon = epsilon;
    }
    
    /// Initialize moment estimates
    fn initialize_moments(&mut self, n_params: usize) {
        if self.m.is_empty() {
            self.m = vec![0.0; n_params];
            self.v = vec![0.0; n_params];
        }
    }
}

impl Optimizer for AdamOptimizer {
    fn optimize<F>(&mut self, objective: F, initial_params: &[f64]) -> Result<OptimizationResult>
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut params = initial_params.to_vec();
        self.initialize_moments(params.len());
        let mut history = OptimizationHistory::new();
        let mut function_evaluations = 0;
        
        for iteration in 0..self.config.max_iterations {
            self.t += 1;
            let current_value = objective(&params);
            function_evaluations += 1;
            
            // Numerical gradient computation
            let mut gradient = vec![0.0; params.len()];
            let h = 1e-8;
            
            for i in 0..params.len() {
                params[i] += h;
                let f_plus = objective(&params);
                params[i] -= 2.0 * h;
                let f_minus = objective(&params);
                params[i] += h; // Restore
                
                gradient[i] = (f_plus - f_minus) / (2.0 * h);
                function_evaluations += 2;
            }
            
            // Adam update
            for i in 0..params.len() {
                // Update biased first moment estimate
                self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * gradient[i];
                
                // Update biased second raw moment estimate
                self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * gradient[i] * gradient[i];
                
                // Compute bias-corrected first moment estimate
                let m_hat = self.m[i] / (1.0 - self.beta1.powi(self.t as i32));
                
                // Compute bias-corrected second raw moment estimate
                let v_hat = self.v[i] / (1.0 - self.beta2.powi(self.t as i32));
                
                // Update parameters
                params[i] -= self.config.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
            }
            
            // Record history
            history.add_step(
                current_value,
                params.clone(),
                gradient.clone(),
                self.config.learning_rate,
            );
            
            if self.config.verbose && iteration % 10 == 0 {
                println!("Adam Iteration {}: f = {:.6}", iteration, current_value);
            }
            
            // Convergence check
            if gradient.iter().map(|g| g * g).sum::<f64>().sqrt() < self.config.tolerance {
                break;
            }
        }
        
        let final_value = objective(&params);
        function_evaluations += 1;
        
        Ok(OptimizationResult {
            optimal_params: params,
            optimal_value: final_value,
            iterations: history.values.len(),
            function_evaluations,
            converged: true,
            history,
        })
    }
    
    fn config(&self) -> &OptimizerConfig {
        &self.config
    }
}

impl VariationalOptimizer for AdamOptimizer {
    fn optimize(&mut self, objective: fn(&[f64]) -> f64, initial_params: &[f64]) -> Result<OptimizationResult> {
        let mut params = initial_params.to_vec();
        self.initialize_moments(params.len());
        let mut history = OptimizationHistory::new();
        let mut function_evaluations = 0;
        
        for iteration in 0..self.config.max_iterations {
            self.t += 1;
            let current_value = objective(&params);
            function_evaluations += 1;
            
            // Numerical gradient computation
            let mut gradient = vec![0.0; params.len()];
            let h = 1e-8;
            
            for i in 0..params.len() {
                params[i] += h;
                let f_plus = objective(&params);
                params[i] -= 2.0 * h;
                let f_minus = objective(&params);
                params[i] += h; // Restore
                
                gradient[i] = (f_plus - f_minus) / (2.0 * h);
                function_evaluations += 2;
            }
            
            // Adam update
            for i in 0..params.len() {
                // Update biased first moment estimate
                self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * gradient[i];
                
                // Update biased second raw moment estimate
                self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * gradient[i] * gradient[i];
                
                // Compute bias-corrected first moment estimate
                let m_hat = self.m[i] / (1.0 - self.beta1.powi(self.t as i32));
                
                // Compute bias-corrected second raw moment estimate
                let v_hat = self.v[i] / (1.0 - self.beta2.powi(self.t as i32));
                
                // Update parameters
                params[i] -= self.config.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
            }
            
            // Record history
            history.add_step(
                current_value,
                params.clone(),
                gradient.clone(),
                self.config.learning_rate,
            );
            
            if self.config.verbose && iteration % 10 == 0 {
                println!("Adam Iteration {}: f = {:.6}", iteration, current_value);
            }
            
            // Convergence check
            if gradient.iter().map(|g| g * g).sum::<f64>().sqrt() < self.config.tolerance {
                break;
            }
        }
        
        let final_value = objective(&params);
        function_evaluations += 1;
        
        Ok(OptimizationResult {
            optimal_params: params,
            optimal_value: final_value,
            iterations: history.values.len(),
            function_evaluations,
            converged: true,
            history,
        })
    }
    
    fn config(&self) -> &OptimizerConfig {
        &self.config
    }
    
    fn optimize_circuit(
        &mut self,
        circuit: &Circuit,
        cost_function: fn(&Circuit, &[f64]) -> Result<f64>,
        initial_params: &[f64],
    ) -> Result<OptimizationResult> {
        // Direct implementation instead of using the optimize method
        let mut params = initial_params.to_vec();
        let mut history = OptimizationHistory::new();
        let mut velocity = vec![0.0; params.len()];
        let mut function_evaluations = 0;
        
        for iteration in 0..self.config.max_iterations {
            let current_value = cost_function(circuit, &params).unwrap_or(f64::INFINITY);
            function_evaluations += 1;
            
            // Numerical gradient computation
            let mut gradient = vec![0.0; params.len()];
            let h = 1e-8;
            
            for i in 0..params.len() {
                params[i] += h;
                let f_plus = cost_function(circuit, &params).unwrap_or(f64::INFINITY);
                params[i] -= 2.0 * h;
                let f_minus = cost_function(circuit, &params).unwrap_or(f64::INFINITY);
                params[i] += h; // Restore
                
                gradient[i] = (f_plus - f_minus) / (2.0 * h);
                function_evaluations += 2;
            }
            
            // Gradient descent with momentum
            for i in 0..params.len() {
                velocity[i] = self.config.momentum * velocity[i] - self.config.learning_rate * gradient[i];
                params[i] += velocity[i];
            }
            
            // Record history
            history.add_step(
                current_value,
                params.clone(),
                gradient.clone(),
                self.config.learning_rate,
            );
            
            if self.config.verbose && iteration % 10 == 0 {
                println!("VQE Iteration {}: E = {:.6}", iteration, current_value);
            }
            
            // Convergence check
            if gradient.iter().map(|g| g * g).sum::<f64>().sqrt() < self.config.tolerance {
                break;
            }
        }
        
        let final_value = cost_function(circuit, &params).unwrap_or(f64::INFINITY);
        function_evaluations += 1;
        
        Ok(OptimizationResult {
            optimal_params: params,
            optimal_value: final_value,
            iterations: history.values.len(),
            function_evaluations,
            converged: true,
            history,
        })
    }
    
    fn optimize_with_gradients(
        &mut self,
        objective: fn(&[f64]) -> f64,
        gradient: fn(&[f64]) -> Vec<f64>,
        initial_params: &[f64],
    ) -> Result<OptimizationResult> {
        let mut params = initial_params.to_vec();
        let mut history = OptimizationHistory::new();
        let mut velocity = vec![0.0; params.len()];
        let mut function_evaluations = 0;
        
        for iteration in 0..self.config.max_iterations {
            let current_value = objective(&params);
            let grad = gradient(&params);
            function_evaluations += 1;
            
            // Gradient descent with momentum
            for i in 0..params.len() {
                velocity[i] = self.config.momentum * velocity[i] - self.config.learning_rate * grad[i];
                params[i] += velocity[i];
            }
            
            // Record history
            history.add_step(
                current_value,
                params.clone(),
                grad.clone(),
                self.config.learning_rate,
            );
            
            if self.config.verbose && iteration % 10 == 0 {
                println!("Iteration {}: f = {:.6}", iteration, current_value);
            }
            
            // Convergence check
            if grad.iter().map(|g| g * g).sum::<f64>().sqrt() < self.config.tolerance {
                break;
            }
        }
        
        let final_value = objective(&params);
        function_evaluations += 1;
        
        Ok(OptimizationResult {
            optimal_params: params,
            optimal_value: final_value,
            iterations: history.values.len(),
            function_evaluations,
            converged: true,
            history,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_qaoa_optimizer() {
        let config = OptimizerConfig {
            max_iterations: 10,
            tolerance: 1e-6,
            learning_rate: 0.1,
            ..Default::default()
        };
        
        let mut optimizer = QAOAOptimizer::new(config, 2);
        
        // Simple quadratic function
        let objective = |params: &[f64]| {
            params.iter().map(|x| x * x).sum::<f64>()
        };
        
        let initial_params = vec![1.0, 1.0];
        let result = Optimizer::optimize(&mut optimizer, objective, &initial_params).unwrap();
        
        // Should converge close to origin
        assert!(result.optimal_value < 0.1);
    }
    
    #[test]
    fn test_vqe_optimizer() {
        let config = OptimizerConfig {
            max_iterations: 50,
            tolerance: 1e-6,
            learning_rate: 0.01,
            ..Default::default()
        };
        
        let hamiltonian = constants::pauli_z();
        let mut optimizer = VQEOptimizer::new(config, hamiltonian);
        
        // Simple quadratic function
        let objective = |params: &[f64]| {
            params.iter().map(|x| (x - 0.5).powi(2)).sum::<f64>()
        };
        
        let initial_params = vec![0.0];
        let result = Optimizer::optimize(&mut optimizer, objective, &initial_params).unwrap();
        
        // Should converge close to 0.5
        assert_abs_diff_eq!(result.optimal_params[0], 0.5, epsilon = 0.1);
    }
    
    #[test]
    fn test_adam_optimizer() {
        let config = OptimizerConfig {
            max_iterations: 100,
            tolerance: 1e-6,
            learning_rate: 0.01,
            ..Default::default()
        };
        
        let mut optimizer = AdamOptimizer::new(config);
        
        // Rosenbrock function (modified for 2D)
        let objective = |params: &[f64]| {
            let x = params[0];
            let y = params[1];
            (1.0 - x).powi(2) + 100.0 * (y - x.powi(2)).powi(2)
        };
        
        let initial_params = vec![-1.0, 1.0];
        let result = Optimizer::optimize(&mut optimizer, objective, &initial_params).unwrap();
        
        // Should converge close to (1, 1)
        assert_abs_diff_eq!(result.optimal_params[0], 1.0, epsilon = 0.1);
        assert_abs_diff_eq!(result.optimal_params[1], 1.0, epsilon = 0.1);
    }
    
    #[test]
    fn test_cost_hamiltonian() {
        let mut hamiltonian = CostHamiltonian::new();
        hamiltonian.add_zz_interaction(0, 1, 1.0);
        hamiltonian.add_z_field(0, 0.5);
        
        assert_eq!(hamiltonian.interactions.len(), 1);
        assert_eq!(hamiltonian.local_fields.len(), 1);
        
        let edges = vec![(0, 1), (1, 2)];
        let max_cut_ham = CostHamiltonian::max_cut(&edges);
        assert_eq!(max_cut_ham.interactions.len(), 2);
    }
    
    #[test]
    fn test_optimization_history() {
        let mut history = OptimizationHistory::new();
        history.add_step(1.0, vec![0.1], vec![0.2], 0.01);
        history.add_step(0.5, vec![0.2], vec![0.1], 0.01);
        
        assert_eq!(history.values.len(), 2);
        assert_eq!(history.parameters.len(), 2);
        assert_eq!(history.gradients.len(), 2);
        assert_eq!(history.learning_rates.len(), 2);
    }
}