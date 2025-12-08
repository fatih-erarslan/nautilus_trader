//! VQE and QAOA Implementation for Market Optimization
//! 
//! Variational Quantum Eigensolver (VQE) and Quantum Approximate Optimization Algorithm (QAOA)
//! for solving complex optimization problems in trading.

use crate::{error::Result, types::*, PennyLaneBackend};
use quantum_core::{QuantumCircuit, QuantumGate, ComplexAmplitude};
use ndarray::{Array1, Array2};
use std::sync::Arc;
use parking_lot::RwLock;
use tracing::{info, debug};

/// Quantum Hamiltonian for optimization problems
#[derive(Debug, Clone)]
pub struct QuantumHamiltonian {
    /// Pauli string coefficients
    pub terms: Vec<(f64, Vec<PauliOperator>)>,
    /// Number of qubits
    pub num_qubits: usize,
}

/// Pauli operators
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PauliOperator {
    I,
    X,
    Y,
    Z,
}

/// Quantum ansatz for variational algorithms
#[derive(Debug, Clone)]
pub struct QuantumAnsatz {
    pub ansatz_type: AnsatzType,
    pub num_qubits: usize,
    pub depth: usize,
    pub entanglement: EntanglementType,
}

/// Ansatz types
#[derive(Debug, Clone)]
pub enum AnsatzType {
    HardwareEfficient,
    UCCSD,
    RealAmplitudes,
    EfficientSU2,
    Custom(String),
}

/// Entanglement patterns
#[derive(Debug, Clone)]
pub enum EntanglementType {
    Linear,
    Circular,
    Full,
    Ladder,
}

/// VQE optimization result
#[derive(Debug, Clone)]
pub struct VqeResult {
    pub energy: f64,
    pub optimal_params: Vec<f64>,
    pub state_vector: Vec<ComplexAmplitude>,
    pub iterations: usize,
    pub convergence_error: f64,
}

/// QAOA optimization result
#[derive(Debug, Clone)]
pub struct QaoaResult {
    pub cost: f64,
    pub optimal_params: (Vec<f64>, Vec<f64>), // (beta, gamma)
    pub solution_bitstring: String,
    pub probabilities: Vec<f64>,
    pub iterations: usize,
}

/// Combinatorial optimization problem
#[derive(Debug, Clone)]
pub struct CombOptProblem {
    pub problem_type: ProblemType,
    pub cost_matrix: Array2<f64>,
    pub constraints: Vec<Constraint>,
}

#[derive(Debug, Clone)]
pub enum ProblemType {
    MaxCut,
    PortfolioOptimization,
    VehicleRouting,
    AssetAllocation,
}

#[derive(Debug, Clone)]
pub struct Constraint {
    pub constraint_type: ConstraintType,
    pub value: f64,
}

#[derive(Debug, Clone)]
pub enum ConstraintType {
    Equality,
    Inequality,
    Budget,
}

/// VQE/QAOA engine
pub struct VqeQaoaEngine {
    backend: Arc<PennyLaneBackend>,
    optimizer_type: OptimizerType,
    max_iterations: usize,
    convergence_threshold: f64,
    cache: RwLock<AlgorithmCache>,
}

#[derive(Debug, Clone)]
enum OptimizerType {
    GradientDescent,
    COBYLA,
    SPSA,
    NelderMead,
}

struct AlgorithmCache {
    vqe_results: Vec<(QuantumHamiltonian, VqeResult)>,
    qaoa_results: Vec<(CombOptProblem, QaoaResult)>,
}

impl VqeQaoaEngine {
    /// Create new VQE/QAOA engine
    pub async fn new(backend: Arc<PennyLaneBackend>) -> Result<Self> {
        Ok(Self {
            backend,
            optimizer_type: OptimizerType::SPSA, // Good for noisy quantum devices
            max_iterations: 200,
            convergence_threshold: 1e-6,
            cache: RwLock::new(AlgorithmCache {
                vqe_results: Vec::new(),
                qaoa_results: Vec::new(),
            }),
        })
    }
    
    /// Run VQE optimization
    pub async fn optimize_vqe(
        &self,
        hamiltonian: &QuantumHamiltonian,
        ansatz: &QuantumAnsatz,
    ) -> Result<VqeResult> {
        info!("Starting VQE optimization for {} qubit system", hamiltonian.num_qubits);
        
        // Initialize parameters
        let num_params = self.count_ansatz_parameters(ansatz);
        let mut params = self.initialize_parameters(num_params);
        
        let mut best_energy = f64::INFINITY;
        let mut best_params = params.clone();
        let mut best_state = vec![];
        
        // Optimization loop
        for iteration in 0..self.max_iterations {
            // Construct circuit with current parameters
            let circuit = self.construct_vqe_circuit(ansatz, &params)?;
            
            // Calculate expectation value
            let (energy, state) = self.calculate_expectation_value(
                &circuit,
                hamiltonian,
            ).await?;
            
            // Update best solution
            if energy < best_energy {
                best_energy = energy;
                best_params = params.clone();
                best_state = state;
            }
            
            // Calculate gradient
            let gradient = self.calculate_vqe_gradient(
                hamiltonian,
                ansatz,
                &params,
            ).await?;
            
            // Update parameters
            params = self.update_parameters(&params, &gradient, iteration);
            
            // Check convergence
            let error = gradient.iter().map(|g| g.abs()).sum::<f64>() / gradient.len() as f64;
            if error < self.convergence_threshold {
                info!("VQE converged after {} iterations", iteration + 1);
                break;
            }
            
            if iteration % 10 == 0 {
                debug!("VQE iteration {}: energy = {:.6}", iteration, best_energy);
            }
        }
        
        Ok(VqeResult {
            energy: best_energy,
            optimal_params: best_params,
            state_vector: best_state,
            iterations: self.max_iterations.min(best_energy as usize),
            convergence_error: 0.0,
        })
    }
    
    /// Run QAOA optimization
    pub async fn optimize_qaoa(
        &self,
        problem: &CombOptProblem,
        layers: usize,
    ) -> Result<QaoaResult> {
        info!("Starting QAOA optimization with {} layers", layers);
        
        // Convert problem to QAOA Hamiltonian
        let (cost_hamiltonian, mixer_hamiltonian) = self.problem_to_hamiltonian(problem)?;
        
        // Initialize parameters (beta and gamma for each layer)
        let mut beta = vec![0.1; layers];
        let mut gamma = vec![0.1; layers];
        
        let mut best_cost = f64::INFINITY;
        let mut best_params = (beta.clone(), gamma.clone());
        let mut best_solution = String::new();
        let mut best_probs = vec![];
        
        // Optimization loop
        for iteration in 0..self.max_iterations {
            // Construct QAOA circuit
            let circuit = self.construct_qaoa_circuit(
                &cost_hamiltonian,
                &mixer_hamiltonian,
                &beta,
                &gamma,
            )?;
            
            // Execute circuit
            let device = self.backend.select_optimal_device(&circuit).await?;
            let result = self.backend.execute_on_device(&circuit, device).await?;
            
            // Calculate cost
            let (cost, solution) = self.evaluate_qaoa_cost(
                problem,
                &result.probabilities,
            )?;
            
            // Update best solution
            if cost < best_cost {
                best_cost = cost;
                best_params = (beta.clone(), gamma.clone());
                best_solution = solution;
                best_probs = result.probabilities.clone();
            }
            
            // Calculate gradients
            let (grad_beta, grad_gamma) = self.calculate_qaoa_gradient(
                &cost_hamiltonian,
                &mixer_hamiltonian,
                &beta,
                &gamma,
                problem,
            ).await?;
            
            // Update parameters
            for i in 0..layers {
                let lr = self.get_learning_rate(iteration);
                beta[i] -= lr * grad_beta[i];
                gamma[i] -= lr * grad_gamma[i];
            }
            
            // Check convergence
            let error = grad_beta.iter().chain(grad_gamma.iter())
                .map(|g| g.abs())
                .sum::<f64>() / (2.0 * layers as f64);
                
            if error < self.convergence_threshold {
                info!("QAOA converged after {} iterations", iteration + 1);
                break;
            }
            
            if iteration % 10 == 0 {
                debug!("QAOA iteration {}: cost = {:.6}", iteration, best_cost);
            }
        }
        
        Ok(QaoaResult {
            cost: best_cost,
            optimal_params: best_params,
            solution_bitstring: best_solution,
            probabilities: best_probs,
            iterations: self.max_iterations,
        })
    }
    
    /// Construct VQE circuit
    fn construct_vqe_circuit(
        &self,
        ansatz: &QuantumAnsatz,
        params: &[f64],
    ) -> Result<QuantumCircuit> {
        let mut circuit = QuantumCircuit::new(ansatz.num_qubits);
        let mut param_idx = 0;
        
        match &ansatz.ansatz_type {
            AnsatzType::HardwareEfficient => {
                for layer in 0..ansatz.depth {
                    // Single-qubit rotations
                    for q in 0..ansatz.num_qubits {
                        circuit.add_gate(QuantumGate::ry(q, params[param_idx]));
                        param_idx += 1;
                        circuit.add_gate(QuantumGate::rz(q, params[param_idx]));
                        param_idx += 1;
                    }
                    
                    // Entangling layer
                    self.add_entangling_layer(&mut circuit, &ansatz.entanglement)?;
                }
            }
            AnsatzType::RealAmplitudes => {
                // Initial rotation layer
                for q in 0..ansatz.num_qubits {
                    circuit.add_gate(QuantumGate::ry(q, params[param_idx]));
                    param_idx += 1;
                }
                
                // Alternating CNOT and rotation layers
                for _ in 0..ansatz.depth {
                    self.add_entangling_layer(&mut circuit, &ansatz.entanglement)?;
                    
                    for q in 0..ansatz.num_qubits {
                        circuit.add_gate(QuantumGate::ry(q, params[param_idx]));
                        param_idx += 1;
                    }
                }
            }
            _ => {
                // Other ansatz types would be implemented similarly
                return Err(anyhow::anyhow!("Ansatz type not yet implemented"));
            }
        }
        
        Ok(circuit)
    }
    
    /// Construct QAOA circuit
    fn construct_qaoa_circuit(
        &self,
        cost_hamiltonian: &QuantumHamiltonian,
        mixer_hamiltonian: &QuantumHamiltonian,
        beta: &[f64],
        gamma: &[f64],
    ) -> Result<QuantumCircuit> {
        let mut circuit = QuantumCircuit::new(cost_hamiltonian.num_qubits);
        
        // Initial superposition
        for q in 0..cost_hamiltonian.num_qubits {
            circuit.add_gate(QuantumGate::hadamard(q));
        }
        
        // QAOA layers
        for p in 0..beta.len() {
            // Cost Hamiltonian evolution
            self.add_hamiltonian_evolution(&mut circuit, cost_hamiltonian, gamma[p])?;
            
            // Mixer Hamiltonian evolution
            self.add_hamiltonian_evolution(&mut circuit, mixer_hamiltonian, beta[p])?;
        }
        
        Ok(circuit)
    }
    
    /// Add entangling layer based on pattern
    fn add_entangling_layer(
        &self,
        circuit: &mut QuantumCircuit,
        entanglement: &EntanglementType,
    ) -> Result<()> {
        let n = circuit.num_qubits();
        
        match entanglement {
            EntanglementType::Linear => {
                for i in 0..n-1 {
                    circuit.add_gate(QuantumGate::cnot(i, i + 1));
                }
            }
            EntanglementType::Circular => {
                for i in 0..n-1 {
                    circuit.add_gate(QuantumGate::cnot(i, i + 1));
                }
                circuit.add_gate(QuantumGate::cnot(n - 1, 0));
            }
            EntanglementType::Full => {
                for i in 0..n {
                    for j in i+1..n {
                        circuit.add_gate(QuantumGate::cnot(i, j));
                    }
                }
            }
            EntanglementType::Ladder => {
                for i in (0..n-1).step_by(2) {
                    circuit.add_gate(QuantumGate::cnot(i, i + 1));
                }
                for i in (1..n-1).step_by(2) {
                    circuit.add_gate(QuantumGate::cnot(i, i + 1));
                }
            }
        }
        
        Ok(())
    }
    
    /// Add Hamiltonian evolution to circuit
    fn add_hamiltonian_evolution(
        &self,
        circuit: &mut QuantumCircuit,
        hamiltonian: &QuantumHamiltonian,
        time: f64,
    ) -> Result<()> {
        for (coeff, pauli_string) in &hamiltonian.terms {
            let angle = 2.0 * coeff * time;
            
            // Apply basis rotations
            for (q, op) in pauli_string.iter().enumerate() {
                match op {
                    PauliOperator::X => circuit.add_gate(QuantumGate::hadamard(q)),
                    PauliOperator::Y => {
                        circuit.add_gate(QuantumGate::rx(q, std::f64::consts::PI / 2.0));
                    }
                    _ => {}
                }
            }
            
            // Apply controlled rotations
            let non_identity: Vec<_> = pauli_string.iter()
                .enumerate()
                .filter(|(_, op)| **op != PauliOperator::I)
                .map(|(q, _)| q)
                .collect();
                
            if non_identity.len() > 1 {
                // Multi-qubit Pauli string
                for i in 0..non_identity.len()-1 {
                    circuit.add_gate(QuantumGate::cnot(non_identity[i], non_identity[i+1]));
                }
                circuit.add_gate(QuantumGate::rz(non_identity.last().unwrap().clone(), angle));
                for i in (0..non_identity.len()-1).rev() {
                    circuit.add_gate(QuantumGate::cnot(non_identity[i], non_identity[i+1]));
                }
            } else if non_identity.len() == 1 {
                // Single-qubit Pauli
                circuit.add_gate(QuantumGate::rz(non_identity[0], angle));
            }
            
            // Undo basis rotations
            for (q, op) in pauli_string.iter().enumerate() {
                match op {
                    PauliOperator::X => circuit.add_gate(QuantumGate::hadamard(q)),
                    PauliOperator::Y => {
                        circuit.add_gate(QuantumGate::rx(q, -std::f64::consts::PI / 2.0));
                    }
                    _ => {}
                }
            }
        }
        
        Ok(())
    }
    
    /// Calculate expectation value of Hamiltonian
    async fn calculate_expectation_value(
        &self,
        circuit: &QuantumCircuit,
        hamiltonian: &QuantumHamiltonian,
    ) -> Result<(f64, Vec<ComplexAmplitude>)> {
        let device = self.backend.select_optimal_device(circuit).await?;
        let result = self.backend.execute_on_device(circuit, device).await?;
        
        // Calculate expectation value
        let mut expectation = 0.0;
        
        for (coeff, pauli_string) in &hamiltonian.terms {
            let pauli_expectation = self.measure_pauli_string(
                &result.state,
                pauli_string,
            )?;
            expectation += coeff * pauli_expectation;
        }
        
        Ok((expectation, result.state.amplitudes().to_vec()))
    }
    
    /// Measure Pauli string expectation
    fn measure_pauli_string(
        &self,
        state: &quantum_core::QuantumState,
        pauli_string: &[PauliOperator],
    ) -> Result<f64> {
        // This is a simplified implementation
        // Real implementation would apply Pauli operators and measure
        
        Ok(1.0) // Placeholder
    }
    
    /// Convert optimization problem to Hamiltonian
    fn problem_to_hamiltonian(
        &self,
        problem: &CombOptProblem,
    ) -> Result<(QuantumHamiltonian, QuantumHamiltonian)> {
        let num_qubits = problem.cost_matrix.nrows();
        
        // Cost Hamiltonian based on problem type
        let cost_hamiltonian = match &problem.problem_type {
            ProblemType::MaxCut => self.maxcut_hamiltonian(&problem.cost_matrix)?,
            ProblemType::PortfolioOptimization => self.portfolio_hamiltonian(&problem.cost_matrix)?,
            _ => return Err(anyhow::anyhow!("Problem type not yet implemented")),
        };
        
        // Standard mixer Hamiltonian (sum of Pauli-X)
        let mut mixer_terms = vec![];
        for i in 0..num_qubits {
            let mut pauli_string = vec![PauliOperator::I; num_qubits];
            pauli_string[i] = PauliOperator::X;
            mixer_terms.push((1.0, pauli_string));
        }
        
        let mixer_hamiltonian = QuantumHamiltonian {
            terms: mixer_terms,
            num_qubits,
        };
        
        Ok((cost_hamiltonian, mixer_hamiltonian))
    }
    
    /// MaxCut Hamiltonian
    fn maxcut_hamiltonian(&self, weights: &Array2<f64>) -> Result<QuantumHamiltonian> {
        let n = weights.nrows();
        let mut terms = vec![];
        
        for i in 0..n {
            for j in i+1..n {
                if weights[[i, j]] != 0.0 {
                    let mut pauli_string = vec![PauliOperator::I; n];
                    pauli_string[i] = PauliOperator::Z;
                    pauli_string[j] = PauliOperator::Z;
                    terms.push((0.5 * weights[[i, j]], pauli_string));
                }
            }
        }
        
        Ok(QuantumHamiltonian {
            terms,
            num_qubits: n,
        })
    }
    
    /// Portfolio optimization Hamiltonian
    fn portfolio_hamiltonian(&self, returns: &Array2<f64>) -> Result<QuantumHamiltonian> {
        // Simplified portfolio optimization
        // Real implementation would include risk, constraints, etc.
        
        let n = returns.nrows();
        let mut terms = vec![];
        
        // Linear terms (expected returns)
        for i in 0..n {
            let mut pauli_string = vec![PauliOperator::I; n];
            pauli_string[i] = PauliOperator::Z;
            terms.push((-returns[[i, i]], pauli_string));
        }
        
        Ok(QuantumHamiltonian {
            terms,
            num_qubits: n,
        })
    }
    
    /// Evaluate QAOA cost
    fn evaluate_qaoa_cost(
        &self,
        problem: &CombOptProblem,
        probabilities: &[f64],
    ) -> Result<(f64, String)> {
        let mut best_cost = f64::INFINITY;
        let mut best_solution = String::new();
        
        // Evaluate all possible solutions
        for (idx, &prob) in probabilities.iter().enumerate() {
            let bitstring = format!("{:0width$b}", idx, width = problem.cost_matrix.nrows());
            let cost = self.evaluate_solution(problem, &bitstring)?;
            
            if cost < best_cost {
                best_cost = cost;
                best_solution = bitstring;
            }
        }
        
        Ok((best_cost, best_solution))
    }
    
    /// Evaluate solution cost
    fn evaluate_solution(&self, problem: &CombOptProblem, bitstring: &str) -> Result<f64> {
        let bits: Vec<bool> = bitstring.chars().map(|c| c == '1').collect();
        let mut cost = 0.0;
        
        match &problem.problem_type {
            ProblemType::MaxCut => {
                for i in 0..bits.len() {
                    for j in i+1..bits.len() {
                        if bits[i] != bits[j] {
                            cost += problem.cost_matrix[[i, j]];
                        }
                    }
                }
                cost = -cost; // Maximization to minimization
            }
            _ => {}
        }
        
        Ok(cost)
    }
    
    /// Calculate VQE gradient
    async fn calculate_vqe_gradient(
        &self,
        hamiltonian: &QuantumHamiltonian,
        ansatz: &QuantumAnsatz,
        params: &[f64],
    ) -> Result<Vec<f64>> {
        let mut gradient = vec![0.0; params.len()];
        let shift = std::f64::consts::PI / 2.0;
        
        // Parameter shift rule
        for i in 0..params.len() {
            let mut params_plus = params.to_vec();
            params_plus[i] += shift;
            let circuit_plus = self.construct_vqe_circuit(ansatz, &params_plus)?;
            let (energy_plus, _) = self.calculate_expectation_value(&circuit_plus, hamiltonian).await?;
            
            let mut params_minus = params.to_vec();
            params_minus[i] -= shift;
            let circuit_minus = self.construct_vqe_circuit(ansatz, &params_minus)?;
            let (energy_minus, _) = self.calculate_expectation_value(&circuit_minus, hamiltonian).await?;
            
            gradient[i] = (energy_plus - energy_minus) / 2.0;
        }
        
        Ok(gradient)
    }
    
    /// Calculate QAOA gradient
    async fn calculate_qaoa_gradient(
        &self,
        cost_hamiltonian: &QuantumHamiltonian,
        mixer_hamiltonian: &QuantumHamiltonian,
        beta: &[f64],
        gamma: &[f64],
        problem: &CombOptProblem,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let shift = std::f64::consts::PI / 4.0;
        let mut grad_beta = vec![0.0; beta.len()];
        let mut grad_gamma = vec![0.0; gamma.len()];
        
        // Parameter shift for beta
        for i in 0..beta.len() {
            let mut beta_plus = beta.to_vec();
            beta_plus[i] += shift;
            let circuit_plus = self.construct_qaoa_circuit(
                cost_hamiltonian,
                mixer_hamiltonian,
                &beta_plus,
                gamma,
            )?;
            let device = self.backend.select_optimal_device(&circuit_plus).await?;
            let result_plus = self.backend.execute_on_device(&circuit_plus, device).await?;
            let (cost_plus, _) = self.evaluate_qaoa_cost(problem, &result_plus.probabilities)?;
            
            let mut beta_minus = beta.to_vec();
            beta_minus[i] -= shift;
            let circuit_minus = self.construct_qaoa_circuit(
                cost_hamiltonian,
                mixer_hamiltonian,
                &beta_minus,
                gamma,
            )?;
            let device = self.backend.select_optimal_device(&circuit_minus).await?;
            let result_minus = self.backend.execute_on_device(&circuit_minus, device).await?;
            let (cost_minus, _) = self.evaluate_qaoa_cost(problem, &result_minus.probabilities)?;
            
            grad_beta[i] = (cost_plus - cost_minus) / (2.0 * shift);
        }
        
        // Similar for gamma
        for i in 0..gamma.len() {
            let mut gamma_plus = gamma.to_vec();
            gamma_plus[i] += shift;
            let circuit_plus = self.construct_qaoa_circuit(
                cost_hamiltonian,
                mixer_hamiltonian,
                beta,
                &gamma_plus,
            )?;
            let device = self.backend.select_optimal_device(&circuit_plus).await?;
            let result_plus = self.backend.execute_on_device(&circuit_plus, device).await?;
            let (cost_plus, _) = self.evaluate_qaoa_cost(problem, &result_plus.probabilities)?;
            
            let mut gamma_minus = gamma.to_vec();
            gamma_minus[i] -= shift;
            let circuit_minus = self.construct_qaoa_circuit(
                cost_hamiltonian,
                mixer_hamiltonian,
                beta,
                &gamma_minus,
            )?;
            let device = self.backend.select_optimal_device(&circuit_minus).await?;
            let result_minus = self.backend.execute_on_device(&circuit_minus, device).await?;
            let (cost_minus, _) = self.evaluate_qaoa_cost(problem, &result_minus.probabilities)?;
            
            grad_gamma[i] = (cost_plus - cost_minus) / (2.0 * shift);
        }
        
        Ok((grad_beta, grad_gamma))
    }
    
    /// Count ansatz parameters
    fn count_ansatz_parameters(&self, ansatz: &QuantumAnsatz) -> usize {
        match &ansatz.ansatz_type {
            AnsatzType::HardwareEfficient => ansatz.num_qubits * 2 * ansatz.depth,
            AnsatzType::RealAmplitudes => ansatz.num_qubits * (ansatz.depth + 1),
            _ => ansatz.num_qubits * ansatz.depth, // Default
        }
    }
    
    /// Initialize optimization parameters
    fn initialize_parameters(&self, num_params: usize) -> Vec<f64> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        (0..num_params)
            .map(|_| rng.gen_range(0.0..2.0 * std::f64::consts::PI))
            .collect()
    }
    
    /// Update parameters based on gradient
    fn update_parameters(&self, params: &[f64], gradient: &[f64], iteration: usize) -> Vec<f64> {
        let learning_rate = self.get_learning_rate(iteration);
        
        params.iter()
            .zip(gradient.iter())
            .map(|(p, g)| p - learning_rate * g)
            .collect()
    }
    
    /// Get learning rate with decay
    fn get_learning_rate(&self, iteration: usize) -> f64 {
        let initial_lr = 0.1;
        let decay_rate = 0.95;
        initial_lr * decay_rate.powi(iteration as i32 / 10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_vqe_optimization() {
        let backend = Arc::new(PennyLaneBackend::new().await.unwrap());
        let engine = VqeQaoaEngine::new(backend).await.unwrap();
        
        // Simple H2 molecule Hamiltonian
        let hamiltonian = QuantumHamiltonian {
            terms: vec![
                (-1.0523, vec![PauliOperator::I, PauliOperator::I]),
                (0.3979, vec![PauliOperator::Z, PauliOperator::I]),
                (-0.3979, vec![PauliOperator::I, PauliOperator::Z]),
                (-0.0112, vec![PauliOperator::Z, PauliOperator::Z]),
                (0.1809, vec![PauliOperator::X, PauliOperator::X]),
            ],
            num_qubits: 2,
        };
        
        let ansatz = QuantumAnsatz {
            ansatz_type: AnsatzType::HardwareEfficient,
            num_qubits: 2,
            depth: 2,
            entanglement: EntanglementType::Linear,
        };
        
        let result = engine.optimize_vqe(&hamiltonian, &ansatz).await.unwrap();
        
        // Ground state energy should be around -1.85
        assert!((result.energy + 1.85).abs() < 0.1);
    }
    
    #[tokio::test]
    async fn test_qaoa_maxcut() {
        let backend = Arc::new(PennyLaneBackend::new().await.unwrap());
        let engine = VqeQaoaEngine::new(backend).await.unwrap();
        
        // Simple 4-node graph
        let cost_matrix = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 1.0, 1.0, 0.0,
                1.0, 0.0, 1.0, 1.0,
                1.0, 1.0, 0.0, 1.0,
                0.0, 1.0, 1.0, 0.0,
            ]
        ).unwrap();
        
        let problem = CombOptProblem {
            problem_type: ProblemType::MaxCut,
            cost_matrix,
            constraints: vec![],
        };
        
        let result = engine.optimize_qaoa(&problem, 2).await.unwrap();
        
        // Should find a valid cut
        assert!(!result.solution_bitstring.is_empty());
        assert!(result.cost < 0.0); // Negative because we minimize
    }
}