pub mod isolation_forest;

// CUDA-accelerated quantum operations
#[cfg(feature = "cuda")]
pub mod cuda;

pub use isolation_forest::{IsolationForest, IsolationTree, AnomalyScore};

// Re-export main types
pub use isolation_forest::{
    IsolationForestBuilder,
    FeatureImportance,
    IsolationForestConfig,
};

// Re-export CUDA quantum operations when available
#[cfg(feature = "cuda")]
pub use cuda::{
    QBMIACudaContext,
    QuantumState,
    QuantumCircuit,
    QuantumGate,
    NashEquilibrium,
    PortfolioOptimizer,
    CudaTensor,
    TensorEngine,
    KernelMetrics,
    OptimizationMetrics,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isolation_forest_creation() {
        let forest = IsolationForest::builder()
            .n_estimators(200)
            .max_samples(256)
            .contamination(0.05)
            .build();
        
        assert_eq!(forest.n_estimators(), 200);
    }
}