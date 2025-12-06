//! Type definitions for NQO

use serde::{Deserialize, Serialize};

/// Optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Optimized parameters
    pub params: Vec<f64>,
    /// Final objective value
    pub value: f64,
    /// Initial objective value
    pub initial_value: f64,
    /// Optimization history
    pub history: Vec<f64>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Confidence in result (0.0 to 1.0)
    pub confidence: f64,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
}

/// Trading parameter optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingParameters {
    /// Entry threshold (0.0 to 1.0)
    pub entry_threshold: f64,
    /// Stop loss percentage
    pub stop_loss: f64,
    /// Take profit percentage
    pub take_profit: f64,
    /// Confidence in parameters (0.0 to 1.0)
    pub confidence: f64,
}

/// Allocation optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationResult {
    /// Allocation percentage (0.0 to 1.0)
    pub allocation: f64,
    /// Confidence in allocation (0.0 to 1.0)
    pub confidence: f64,
}

/// Neural network weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralWeights {
    /// Input to hidden layer weights
    pub input_hidden: Vec<Vec<f64>>,
    /// Hidden to output layer weights
    pub hidden_output: Vec<Vec<f64>>,
    /// Recurrent connection weights
    pub recurrent: Vec<Vec<f64>>,
    /// Hidden layer biases
    pub hidden_biases: Vec<f64>,
    /// Output layer biases
    pub output_biases: Vec<f64>,
}

/// Configuration for NQO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NqoConfig {
    /// Number of neurons
    pub neurons: usize,
    /// Number of qubits for quantum circuits
    pub qubits: usize,
    /// Adaptivity parameter (0.0 to 1.0)
    pub adaptivity: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Maximum history size
    pub max_history: usize,
    /// Cache size
    pub cache_size: usize,
    /// Number of epochs for training
    pub epochs: usize,
    /// Use GPU acceleration
    pub use_gpu: bool,
    /// Use SIMD acceleration
    pub use_simd: bool,
    /// Number of shots for quantum measurements
    pub quantum_shots: Option<usize>,
    /// Enable fault tolerance
    pub enable_fault_tolerance: bool,
    /// Log level
    pub log_level: String,
}

impl Default for NqoConfig {
    fn default() -> Self {
        Self {
            neurons: 128,
            qubits: 4,
            adaptivity: 0.7,
            learning_rate: 0.01,
            max_history: 50,
            cache_size: 100,
            epochs: 10,
            use_gpu: true,
            use_simd: true,
            quantum_shots: None,
            enable_fault_tolerance: true,
            log_level: "INFO".to_string(),
        }
    }
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Mean improvement across optimizations
    pub mean_improvement: f64,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Number of samples
    pub sample_size: usize,
}

/// Execution statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ExecutionStats {
    /// Average execution time in ms
    pub avg_time_ms: f64,
    /// Minimum execution time in ms
    pub min_time_ms: f64,
    /// Maximum execution time in ms
    pub max_time_ms: f64,
    /// Number of executions
    pub count: usize,
}

/// Optimizer configuration (alias for NqoConfig)
pub type OptimizerConfig = NqoConfig;

/// Objective function type
pub type ObjectiveFunction = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;

/// Quantum circuit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCircuitConfig {
    /// Number of qubits
    pub num_qubits: usize,
    /// Circuit depth
    pub depth: usize,
    /// Use entanglement
    pub use_entanglement: bool,
}

/// Neural network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetworkConfig {
    /// Number of hidden neurons
    pub hidden_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Activation function
    pub activation: String,
}

/// Hardware configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Use GPU
    pub use_gpu: bool,
    /// Use SIMD
    pub use_simd: bool,
    /// Preferred device
    pub device: String,
}

/// Optimization problem definition
pub struct OptimizationProblem {
    /// Objective function to minimize
    pub objective: ObjectiveFunction,
    /// Initial parameters
    pub initial_params: Vec<f64>,
    /// Parameter bounds (min, max)
    pub bounds: Option<Vec<(f64, f64)>>,
    /// Problem dimension
    pub dimension: usize,
}