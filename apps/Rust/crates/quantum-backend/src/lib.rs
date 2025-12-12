//! Quantum Backend with PennyLane Integration
//! 
//! High-performance quantum computing backend with GPU acceleration
//! and PennyLane device hierarchy support.

pub mod pennylane_backend;
pub mod quantum_solver;
pub mod circuit_optimizer;
pub mod vqe_qaoa;
pub mod cuda_kernels;
pub mod error;
pub mod types;
pub mod utils;
pub mod hybrid_strategies;

pub use pennylane_backend::*;
pub use quantum_solver::*;
pub use circuit_optimizer::*;
pub use vqe_qaoa::*;
pub use cuda_kernels::*;
pub use error::*;
pub use types::*;
pub use hybrid_strategies::*;

use quantum_core::{QuantumState, QuantumCircuit, QuantumResult};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Main quantum backend coordinator
pub struct QuantumBackend {
    pennylane: Arc<PennyLaneBackend>,
    solver: Arc<QuantumNashSolver>,
    optimizer: Arc<CircuitOptimizer>,
    vqe_qaoa: Arc<VqeQaoaEngine>,
    cuda_accelerator: Arc<CudaAccelerator>,
}

impl QuantumBackend {
    /// Create new quantum backend with full GPU acceleration
    pub async fn new() -> Result<Self> {
        let pennylane = Arc::new(PennyLaneBackend::new().await?);
        let solver = Arc::new(QuantumNashSolver::new(pennylane.clone()).await?);
        let optimizer = Arc::new(CircuitOptimizer::new().await?);
        let vqe_qaoa = Arc::new(VqeQaoaEngine::new(pennylane.clone()).await?);
        let cuda_accelerator = Arc::new(CudaAccelerator::new().await?);
        
        Ok(Self {
            pennylane,
            solver,
            optimizer,
            vqe_qaoa,
            cuda_accelerator,
        })
    }
    
    /// Execute quantum circuit with optimal device selection
    pub async fn execute_circuit(&self, circuit: &QuantumCircuit) -> Result<QuantumResult> {
        // Optimize circuit first
        let optimized = self.optimizer.optimize(circuit).await?;
        
        // Select best device based on circuit properties
        let device = self.pennylane.select_optimal_device(&optimized).await?;
        
        // Execute with GPU acceleration if available
        if device.supports_gpu() {
            self.cuda_accelerator.execute_circuit(&optimized, device).await
        } else {
            self.pennylane.execute_on_device(&optimized, device).await
        }
    }
    
    /// Solve quantum Nash equilibrium
    pub async fn solve_nash_equilibrium(
        &self,
        payoff_matrix: &ndarray::Array2<f64>,
        num_qubits: usize,
    ) -> Result<NashEquilibriumSolution> {
        self.solver.solve(payoff_matrix, num_qubits).await
    }
    
    /// Run VQE optimization
    pub async fn run_vqe(
        &self,
        hamiltonian: &QuantumHamiltonian,
        ansatz: &QuantumAnsatz,
    ) -> Result<VqeResult> {
        self.vqe_qaoa.optimize_vqe(hamiltonian, ansatz).await
    }
    
    /// Run QAOA for combinatorial optimization
    pub async fn run_qaoa(
        &self,
        problem: &CombOptProblem,
        layers: usize,
    ) -> Result<QaoaResult> {
        self.vqe_qaoa.optimize_qaoa(problem, layers).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_backend_initialization() {
        let backend = QuantumBackend::new().await.unwrap();
        assert!(backend.pennylane.is_initialized().await);
    }
}