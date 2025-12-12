//! Quantum Nash Equilibrium Solver
//! 
//! Implements quantum algorithms for finding Nash equilibria in game theory,
//! optimized for high-frequency trading scenarios.

use crate::{error::Result, types::*, PennyLaneBackend};
use quantum_core::{QuantumCircuit, QuantumState, QuantumGate, ComplexAmplitude};
use ndarray::{Array1, Array2};
use std::sync::Arc;
use num_complex::Complex64;
use parking_lot::RwLock;
use tracing::{info, debug};

/// Nash equilibrium solution
#[derive(Debug, Clone)]
pub struct NashEquilibriumSolution {
    pub strategy_player1: Vec<f64>,
    pub strategy_player2: Vec<f64>,
    pub value: f64,
    pub iterations: usize,
    pub convergence_error: f64,
    pub quantum_advantage: f64,
}

/// Quantum Nash solver using variational quantum eigensolver approach
pub struct QuantumNashSolver {
    backend: Arc<PennyLaneBackend>,
    max_iterations: usize,
    convergence_threshold: f64,
    learning_rate: f64,
    cache: RwLock<SolverCache>,
}

/// Cache for solver optimizations
struct SolverCache {
    recent_solutions: Vec<(Array2<f64>, NashEquilibriumSolution)>,
    circuit_templates: Vec<QuantumCircuit>,
}

impl QuantumNashSolver {
    /// Create new quantum Nash solver
    pub async fn new(backend: Arc<PennyLaneBackend>) -> Result<Self> {
        Ok(Self {
            backend,
            max_iterations: 100,
            convergence_threshold: 1e-6,
            learning_rate: 0.1,
            cache: RwLock::new(SolverCache {
                recent_solutions: Vec::new(),
                circuit_templates: Vec::new(),
            }),
        })
    }
    
    /// Solve Nash equilibrium for given payoff matrix
    pub async fn solve(
        &self,
        payoff_matrix: &Array2<f64>,
        num_qubits: usize,
    ) -> Result<NashEquilibriumSolution> {
        info!("Solving Nash equilibrium with quantum algorithm");
        
        // Check cache first
        if let Some(cached) = self.check_cache(payoff_matrix) {
            debug!("Using cached solution");
            return Ok(cached);
        }
        
        let start = std::time::Instant::now();
        
        // Initialize quantum state
        let initial_params = self.initialize_parameters(payoff_matrix.nrows(), num_qubits);
        
        // Run variational optimization
        let mut params = initial_params;
        let mut best_value = f64::NEG_INFINITY;
        let mut best_strategies = (vec![], vec![]);
        let mut convergence_error = 1.0;
        
        for iteration in 0..self.max_iterations {
            // Construct variational circuit
            let circuit = self.construct_nash_circuit(&params, num_qubits)?;
            
            // Execute circuit
            let device = self.backend.select_optimal_device(&circuit).await?;
            let result = self.backend.execute_on_device(&circuit, device).await?;
            
            // Extract strategies from quantum state
            let (strategy1, strategy2) = self.extract_strategies(&result.state, payoff_matrix)?;
            
            // Calculate game value
            let value = self.calculate_game_value(payoff_matrix, &strategy1, &strategy2);
            
            // Update best solution
            if value > best_value {
                best_value = value;
                best_strategies = (strategy1.clone(), strategy2.clone());
            }
            
            // Calculate gradient
            let gradient = self.calculate_gradient(
                payoff_matrix,
                &params,
                &strategy1,
                &strategy2,
                value,
                num_qubits,
            ).await?;
            
            // Update parameters
            for i in 0..params.len() {
                params[i] += self.learning_rate * gradient[i];
            }
            
            // Check convergence
            convergence_error = gradient.iter().map(|g| g.abs()).sum::<f64>() / gradient.len() as f64;
            if convergence_error < self.convergence_threshold {
                info!("Converged after {} iterations", iteration + 1);
                break;
            }
        }
        
        // Calculate quantum advantage
        let classical_time = self.estimate_classical_time(payoff_matrix);
        let quantum_time = start.elapsed().as_secs_f64();
        let quantum_advantage = classical_time / quantum_time;
        
        let solution = NashEquilibriumSolution {
            strategy_player1: best_strategies.0,
            strategy_player2: best_strategies.1,
            value: best_value,
            iterations: self.max_iterations.min(convergence_error as usize),
            convergence_error,
            quantum_advantage,
        };
        
        // Cache solution
        self.cache_solution(payoff_matrix.clone(), solution.clone());
        
        Ok(solution)
    }
    
    /// Construct variational circuit for Nash equilibrium
    fn construct_nash_circuit(&self, params: &[f64], num_qubits: usize) -> Result<QuantumCircuit> {
        let mut circuit = QuantumCircuit::new(num_qubits);
        
        // Initial superposition
        for i in 0..num_qubits {
            circuit.add_gate(QuantumGate::hadamard(i));
        }
        
        // Variational layers
        let layers = 3;
        let mut param_idx = 0;
        
        for _ in 0..layers {
            // Rotation layer
            for i in 0..num_qubits {
                if param_idx < params.len() {
                    circuit.add_gate(QuantumGate::ry(i, params[param_idx]));
                    param_idx += 1;
                }
                if param_idx < params.len() {
                    circuit.add_gate(QuantumGate::rz(i, params[param_idx]));
                    param_idx += 1;
                }
            }
            
            // Entanglement layer
            for i in 0..num_qubits - 1 {
                circuit.add_gate(QuantumGate::cnot(i, i + 1));
            }
            // Wrap around
            if num_qubits > 2 {
                circuit.add_gate(QuantumGate::cnot(num_qubits - 1, 0));
            }
        }
        
        Ok(circuit)
    }
    
    /// Extract mixed strategies from quantum state
    fn extract_strategies(
        &self,
        state: &QuantumState,
        payoff_matrix: &Array2<f64>,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let n_actions1 = payoff_matrix.nrows();
        let n_actions2 = payoff_matrix.ncols();
        
        // Get probability distribution from quantum state
        let probs = state.probabilities();
        
        // Map quantum state to strategy spaces
        let mut strategy1 = vec![0.0; n_actions1];
        let mut strategy2 = vec![0.0; n_actions2];
        
        // Partition state space
        let states_per_action1 = probs.len() / n_actions1;
        let states_per_action2 = probs.len() / n_actions2;
        
        // Sum probabilities for each action
        for (i, &prob) in probs.iter().enumerate() {
            let action1 = i / states_per_action1;
            let action2 = i / states_per_action2;
            
            if action1 < n_actions1 {
                strategy1[action1] += prob;
            }
            if action2 < n_actions2 {
                strategy2[action2] += prob;
            }
        }
        
        // Normalize strategies
        let sum1: f64 = strategy1.iter().sum();
        let sum2: f64 = strategy2.iter().sum();
        
        if sum1 > 0.0 {
            strategy1.iter_mut().for_each(|s| *s /= sum1);
        }
        if sum2 > 0.0 {
            strategy2.iter_mut().for_each(|s| *s /= sum2);
        }
        
        Ok((strategy1, strategy2))
    }
    
    /// Calculate expected game value
    fn calculate_game_value(
        &self,
        payoff_matrix: &Array2<f64>,
        strategy1: &[f64],
        strategy2: &[f64],
    ) -> f64 {
        let mut value = 0.0;
        
        for i in 0..payoff_matrix.nrows() {
            for j in 0..payoff_matrix.ncols() {
                value += strategy1[i] * strategy2[j] * payoff_matrix[[i, j]];
            }
        }
        
        value
    }
    
    /// Calculate gradient using parameter shift rule
    async fn calculate_gradient(
        &self,
        payoff_matrix: &Array2<f64>,
        params: &[f64],
        strategy1: &[f64],
        strategy2: &[f64],
        current_value: f64,
        num_qubits: usize,
    ) -> Result<Vec<f64>> {
        let mut gradient = vec![0.0; params.len()];
        let shift = std::f64::consts::PI / 2.0;
        
        for i in 0..params.len() {
            // Forward shift
            let mut params_plus = params.to_vec();
            params_plus[i] += shift;
            let circuit_plus = self.construct_nash_circuit(&params_plus, num_qubits)?;
            let device = self.backend.select_optimal_device(&circuit_plus).await?;
            let result_plus = self.backend.execute_on_device(&circuit_plus, device).await?;
            let (strat1_plus, strat2_plus) = self.extract_strategies(&result_plus.state, payoff_matrix)?;
            let value_plus = self.calculate_game_value(payoff_matrix, &strat1_plus, &strat2_plus);
            
            // Backward shift
            let mut params_minus = params.to_vec();
            params_minus[i] -= shift;
            let circuit_minus = self.construct_nash_circuit(&params_minus, num_qubits)?;
            let device = self.backend.select_optimal_device(&circuit_minus).await?;
            let result_minus = self.backend.execute_on_device(&circuit_minus, device).await?;
            let (strat1_minus, strat2_minus) = self.extract_strategies(&result_minus.state, payoff_matrix)?;
            let value_minus = self.calculate_game_value(payoff_matrix, &strat1_minus, &strat2_minus);
            
            // Parameter shift rule
            gradient[i] = (value_plus - value_minus) / 2.0;
        }
        
        Ok(gradient)
    }
    
    /// Initialize variational parameters
    fn initialize_parameters(&self, matrix_size: usize, num_qubits: usize) -> Vec<f64> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Number of parameters: 2 rotations per qubit per layer
        let layers = 3;
        let num_params = num_qubits * 2 * layers;
        
        (0..num_params)
            .map(|_| rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI))
            .collect()
    }
    
    /// Estimate classical computation time
    fn estimate_classical_time(&self, payoff_matrix: &Array2<f64>) -> f64 {
        // Rough estimate based on matrix size
        let size = payoff_matrix.nrows().max(payoff_matrix.ncols());
        (size as f64).powi(3) * 1e-9 // Cubic complexity in nanoseconds
    }
    
    /// Check solution cache
    fn check_cache(&self, payoff_matrix: &Array2<f64>) -> Option<NashEquilibriumSolution> {
        let cache = self.cache.read();
        
        for (cached_matrix, solution) in &cache.recent_solutions {
            if matrices_equal(cached_matrix, payoff_matrix) {
                return Some(solution.clone());
            }
        }
        
        None
    }
    
    /// Cache computed solution
    fn cache_solution(&self, payoff_matrix: Array2<f64>, solution: NashEquilibriumSolution) {
        let mut cache = self.cache.write();
        
        // Keep only recent solutions
        if cache.recent_solutions.len() > 10 {
            cache.recent_solutions.remove(0);
        }
        
        cache.recent_solutions.push((payoff_matrix, solution));
    }
}

/// Check if two matrices are equal
fn matrices_equal(a: &Array2<f64>, b: &Array2<f64>) -> bool {
    a.shape() == b.shape() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-10)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_nash_solver() {
        let backend = Arc::new(PennyLaneBackend::new().await.unwrap());
        let solver = QuantumNashSolver::new(backend).await.unwrap();
        
        // Prisoner's dilemma payoff matrix
        let payoff = Array2::from_shape_vec(
            (2, 2),
            vec![3.0, 0.0, 5.0, 1.0]
        ).unwrap();
        
        let solution = solver.solve(&payoff, 4).await.unwrap();
        
        assert_eq!(solution.strategy_player1.len(), 2);
        assert_eq!(solution.strategy_player2.len(), 2);
        assert!((solution.strategy_player1.iter().sum::<f64>() - 1.0).abs() < 0.01);
    }
}