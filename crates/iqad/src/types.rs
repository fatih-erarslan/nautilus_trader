//! Type definitions for IQAD

use serde::{Deserialize, Serialize};

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyResult {
    /// Whether an anomaly was detected
    pub detected: bool,
    /// Anomaly score (0.0 to 1.0)
    pub score: f64,
    /// Detection threshold used
    pub threshold: f64,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Detector affinities for top detectors
    pub detector_affinities: Vec<f64>,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Time to event estimate (if anomaly detected)
    pub time_to_event: Option<TimeToEvent>,
}

/// Time to event estimate
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TimeToEvent {
    /// Event is imminent (high urgency)
    Imminent,
    /// Event is near-term (medium urgency)
    NearTerm,
    /// Event is potential (low urgency)
    Potential,
}

/// Anomaly type classification
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AnomalyType {
    /// Market crash pattern
    MarketCrash,
    /// Flash crash pattern
    FlashCrash,
    /// Volatility spike
    VolatilitySpike,
    /// Liquidity crisis
    LiquidityCrisis,
    /// Regime change
    RegimeChange,
    /// Unknown anomaly type
    Unknown,
}

/// Code features for governance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeFeatures {
    /// Synthetic data generation score (0.0 to 1.0)
    pub synthetic_data_score: f64,
    /// Code complexity score (0.0 to 1.0)
    pub complexity_score: f64,
    /// Hardcoded value ratio (0.0 to 1.0)
    pub hardcoded_ratio: f64,
    /// Mock usage score (0.0 to 1.0)
    pub mock_usage_score: f64,
    /// Loop nesting depth
    pub loop_depth: usize,
    /// Recursion depth estimate
    pub recursion_depth: usize,
    /// API call ratio
    pub api_call_ratio: f64,
    /// Resource usage score (0.0 to 1.0)
    pub resource_usage_score: f64,
}

/// Immune detector representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImmuneDetector {
    /// Detector pattern vector
    pub pattern: Vec<f64>,
    /// Affinity threshold
    pub affinity_threshold: f64,
    /// Creation timestamp
    pub created_at: std::time::SystemTime,
    /// Number of activations
    pub activation_count: usize,
}

/// Configuration for IQAD
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IqadConfig {
    /// Number of detectors
    pub num_detectors: usize,
    /// Quantum circuit dimension (number of qubits)
    pub quantum_dimension: usize,
    /// Detection sensitivity (0.0 to 1.0)
    pub sensitivity: f64,
    /// Negative selection threshold
    pub negative_selection_threshold: f64,
    /// Maximum self patterns to store
    pub max_self_patterns: usize,
    /// Maximum anomaly memory size
    pub max_anomaly_memory: usize,
    /// Cache size
    pub cache_size: usize,
    /// Use GPU acceleration
    pub use_gpu: bool,
    /// Use SIMD acceleration
    pub use_simd: bool,
    /// Number of shots for quantum measurements
    pub quantum_shots: Option<usize>,
    /// Mutation rate for detector evolution
    pub mutation_rate: f64,
    /// Enable fault tolerance
    pub enable_fault_tolerance: bool,
    /// Log level
    pub log_level: String,
}

impl Default for IqadConfig {
    fn default() -> Self {
        Self {
            num_detectors: 50,
            quantum_dimension: 4,
            sensitivity: 0.85,
            negative_selection_threshold: 0.7,
            max_self_patterns: 100,
            max_anomaly_memory: 50,
            cache_size: 100,
            use_gpu: true,
            use_simd: true,
            quantum_shots: None,
            mutation_rate: 0.05,
            enable_fault_tolerance: true,
            log_level: "INFO".to_string(),
        }
    }
}

/// Quantum state representation
pub type QuantumState = Vec<f64>;

/// Detector configuration (alias for IqadConfig)
pub type DetectorConfig = IqadConfig;

/// Anomaly score (simple f64 wrapper)
pub type AnomalyScore = f64;

/// Data point representation
pub type DataPoint = Vec<f64>;

/// Quantum circuit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCircuitConfig {
    /// Number of qubits
    pub num_qubits: usize,
    /// Circuit depth
    pub depth: usize,
    /// Number of shots
    pub shots: usize,
}

/// Immune system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImmuneSystemConfig {
    /// Number of detectors
    pub num_detectors: usize,
    /// Affinity threshold
    pub affinity_threshold: f64,
    /// Mutation rate
    pub mutation_rate: f64,
}

/// Hardware configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Use GPU
    pub use_gpu: bool,
    /// Use SIMD
    pub use_simd: bool,
    /// Number of threads
    pub num_threads: usize,
}