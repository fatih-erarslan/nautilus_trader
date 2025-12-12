//! # Quantum Uncertainty Configuration
//!
//! This module defines configuration structures for the quantum uncertainty system.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main configuration for quantum uncertainty quantification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumConfig {
    /// Number of qubits in quantum circuits
    pub n_qubits: usize,
    /// Number of layers in variational circuits
    pub n_layers: usize,
    /// Ensemble size for VQC ensemble
    pub ensemble_size: usize,
    /// Confidence level for conformal prediction
    pub confidence_level: f64,
    /// Maximum iterations for optimization
    pub max_iterations: usize,
    /// Learning rate for quantum optimization
    pub learning_rate: f64,
    /// Convergence threshold
    pub convergence_threshold: f64,
    /// Enable noise simulation
    pub enable_noise: bool,
    /// Noise parameters
    pub noise_config: NoiseConfig,
    /// Feature extraction configuration
    pub feature_config: FeatureConfig,
    /// Correlation analysis configuration
    pub correlation_config: CorrelationConfig,
    /// Conformal prediction configuration
    pub conformal_config: ConformalConfig,
    /// Measurement optimization configuration
    pub measurement_config: MeasurementConfig,
    /// Performance configuration
    pub performance_config: PerformanceConfig,
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            n_qubits: 8,
            n_layers: 3,
            ensemble_size: 5,
            confidence_level: 0.95,
            max_iterations: 1000,
            learning_rate: 0.01,
            convergence_threshold: 1e-6,
            enable_noise: false,
            noise_config: NoiseConfig::default(),
            feature_config: FeatureConfig::default(),
            correlation_config: CorrelationConfig::default(),
            conformal_config: ConformalConfig::default(),
            measurement_config: MeasurementConfig::default(),
            performance_config: PerformanceConfig::default(),
        }
    }
}

/// Noise simulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseConfig {
    /// Depolarizing noise rate
    pub depolarizing_rate: f64,
    /// Phase damping rate
    pub phase_damping_rate: f64,
    /// Amplitude damping rate
    pub amplitude_damping_rate: f64,
    /// Thermal noise temperature
    pub thermal_temperature: f64,
    /// Gate error rates
    pub gate_error_rates: GateErrorRates,
    /// Measurement error rate
    pub measurement_error_rate: f64,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            depolarizing_rate: 0.001,
            phase_damping_rate: 0.001,
            amplitude_damping_rate: 0.001,
            thermal_temperature: 0.01,
            gate_error_rates: GateErrorRates::default(),
            measurement_error_rate: 0.01,
        }
    }
}

/// Gate error rates for different quantum gates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateErrorRates {
    /// Single-qubit gate error rate
    pub single_qubit_error_rate: f64,
    /// Two-qubit gate error rate
    pub two_qubit_error_rate: f64,
    /// Hadamard gate error rate
    pub hadamard_error_rate: f64,
    /// CNOT gate error rate
    pub cnot_error_rate: f64,
    /// Rotation gate error rate
    pub rotation_error_rate: f64,
}

impl Default for GateErrorRates {
    fn default() -> Self {
        Self {
            single_qubit_error_rate: 0.001,
            two_qubit_error_rate: 0.01,
            hadamard_error_rate: 0.001,
            cnot_error_rate: 0.01,
            rotation_error_rate: 0.002,
        }
    }
}

/// Feature extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Enable superposition feature extraction
    pub enable_superposition: bool,
    /// Enable entanglement feature extraction
    pub enable_entanglement: bool,
    /// Enable interference feature extraction
    pub enable_interference: bool,
    /// Enable phase feature extraction
    pub enable_phase: bool,
    /// Enable amplitude feature extraction
    pub enable_amplitude: bool,
    /// Enable coherence feature extraction
    pub enable_coherence: bool,
    /// Feature normalization method
    pub normalization_method: String,
    /// Maximum number of features
    pub max_features: usize,
    /// Feature selection threshold
    pub selection_threshold: f64,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            enable_superposition: true,
            enable_entanglement: true,
            enable_interference: true,
            enable_phase: true,
            enable_amplitude: true,
            enable_coherence: true,
            normalization_method: "l2".to_string(),
            max_features: 100,
            selection_threshold: 0.01,
        }
    }
}

/// Correlation analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationConfig {
    /// Enable quantum mutual information
    pub enable_quantum_mutual_info: bool,
    /// Enable quantum discord
    pub enable_quantum_discord: bool,
    /// Enable entanglement measures
    pub enable_entanglement_measures: bool,
    /// Maximum correlation order
    pub max_correlation_order: usize,
    /// Correlation threshold
    pub correlation_threshold: f64,
    /// Number of correlation samples
    pub n_correlation_samples: usize,
}

impl Default for CorrelationConfig {
    fn default() -> Self {
        Self {
            enable_quantum_mutual_info: true,
            enable_quantum_discord: true,
            enable_entanglement_measures: true,
            max_correlation_order: 3,
            correlation_threshold: 0.1,
            n_correlation_samples: 1000,
        }
    }
}

/// Conformal prediction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformalConfig {
    /// Enable quantum calibration
    pub enable_quantum_calibration: bool,
    /// Calibration method
    pub calibration_method: String,
    /// Non-conformity measure type
    pub nonconformity_type: String,
    /// Interval optimization method
    pub interval_optimization: String,
    /// Coverage probability estimation method
    pub coverage_estimation: String,
    /// Number of bootstrap samples
    pub n_bootstrap_samples: usize,
    /// Efficiency optimization weight
    pub efficiency_weight: f64,
}

impl Default for ConformalConfig {
    fn default() -> Self {
        Self {
            enable_quantum_calibration: true,
            calibration_method: "quantum_variational".to_string(),
            nonconformity_type: "quantum_absolute".to_string(),
            interval_optimization: "quantum_efficiency".to_string(),
            coverage_estimation: "quantum_enhanced".to_string(),
            n_bootstrap_samples: 1000,
            efficiency_weight: 0.5,
        }
    }
}

/// Measurement optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementConfig {
    /// Enable measurement optimization
    pub enable_optimization: bool,
    /// Optimization method
    pub optimization_method: String,
    /// Information metric type
    pub information_metric: String,
    /// Maximum number of measurement operators
    pub max_operators: usize,
    /// Basis optimization method
    pub basis_optimization: String,
    /// Adaptive scheduling enabled
    pub enable_adaptive_scheduling: bool,
    /// Measurement precision
    pub measurement_precision: f64,
}

impl Default for MeasurementConfig {
    fn default() -> Self {
        Self {
            enable_optimization: true,
            optimization_method: "quantum_fisher_information".to_string(),
            information_metric: "mutual_information".to_string(),
            max_operators: 20,
            basis_optimization: "adaptive".to_string(),
            enable_adaptive_scheduling: true,
            measurement_precision: 1e-6,
        }
    }
}

/// Performance and computational configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable parallel processing
    pub enable_parallel: bool,
    /// Number of threads
    pub n_threads: usize,
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    /// GPU device ID
    pub gpu_device_id: usize,
    /// Memory limit (MB)
    pub memory_limit_mb: usize,
    /// Cache size (MB)
    pub cache_size_mb: usize,
    /// Enable result caching
    pub enable_caching: bool,
    /// Cache directory
    pub cache_directory: PathBuf,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    /// Profiling enabled
    pub enable_profiling: bool,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_parallel: true,
            n_threads: num_cpus::get(),
            enable_gpu: false,
            gpu_device_id: 0,
            memory_limit_mb: 8192,
            cache_size_mb: 1024,
            enable_caching: true,
            cache_directory: PathBuf::from("./cache/quantum"),
            enable_monitoring: true,
            enable_profiling: false,
        }
    }
}

/// Quantum algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmConfig {
    /// VQC ansatz type
    pub vqc_ansatz: String,
    /// Optimization algorithm
    pub optimizer: String,
    /// Gradient computation method
    pub gradient_method: String,
    /// Parameter initialization
    pub parameter_init: String,
    /// Circuit depth strategy
    pub depth_strategy: String,
    /// Entanglement strategy
    pub entanglement_strategy: String,
}

impl Default for AlgorithmConfig {
    fn default() -> Self {
        Self {
            vqc_ansatz: "hardware_efficient".to_string(),
            optimizer: "adam".to_string(),
            gradient_method: "finite_difference".to_string(),
            parameter_init: "random".to_string(),
            depth_strategy: "adaptive".to_string(),
            entanglement_strategy: "circular".to_string(),
        }
    }
}

/// Hardware configuration for quantum simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Simulation backend
    pub backend: String,
    /// Device connectivity
    pub connectivity: Option<Vec<(usize, usize)>>,
    /// Native gate set
    pub native_gates: Vec<String>,
    /// Coherence times (microseconds)
    pub coherence_times: CoherenceTimes,
    /// Gate times (nanoseconds)
    pub gate_times: GateTimes,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            backend: "quantum_simulator".to_string(),
            connectivity: None,
            native_gates: vec!["RZ".to_string(), "RX".to_string(), "CNOT".to_string()],
            coherence_times: CoherenceTimes::default(),
            gate_times: GateTimes::default(),
        }
    }
}

/// Coherence time parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceTimes {
    /// T1 relaxation time (microseconds)
    pub t1_relaxation: f64,
    /// T2 dephasing time (microseconds)
    pub t2_dephasing: f64,
    /// T2* coherence time (microseconds)
    pub t2_star: f64,
}

impl Default for CoherenceTimes {
    fn default() -> Self {
        Self {
            t1_relaxation: 100.0,
            t2_dephasing: 50.0,
            t2_star: 25.0,
        }
    }
}

/// Gate execution time parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateTimes {
    /// Single-qubit gate time (nanoseconds)
    pub single_qubit_gate: f64,
    /// Two-qubit gate time (nanoseconds)
    pub two_qubit_gate: f64,
    /// Readout time (nanoseconds)
    pub readout_time: f64,
    /// Reset time (nanoseconds)
    pub reset_time: f64,
}

impl Default for GateTimes {
    fn default() -> Self {
        Self {
            single_qubit_gate: 20.0,
            two_qubit_gate: 200.0,
            readout_time: 1000.0,
            reset_time: 5000.0,
        }
    }
}

impl QuantumConfig {
    /// Create configuration from file
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let config: QuantumConfig = toml::from_str(&contents)?;
        Ok(config)
    }

    /// Save configuration to file
    pub fn to_file(&self, path: impl AsRef<std::path::Path>) -> Result<(), Box<dyn std::error::Error>> {
        let contents = toml::to_string_pretty(self)?;
        std::fs::write(path, contents)?;
        Ok(())
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.n_qubits == 0 {
            return Err("Number of qubits must be greater than 0".to_string());
        }

        if self.n_qubits > 30 {
            return Err("Number of qubits exceeds reasonable simulation limit".to_string());
        }

        if self.confidence_level <= 0.0 || self.confidence_level >= 1.0 {
            return Err("Confidence level must be between 0 and 1".to_string());
        }

        if self.learning_rate <= 0.0 {
            return Err("Learning rate must be positive".to_string());
        }

        if self.convergence_threshold <= 0.0 {
            return Err("Convergence threshold must be positive".to_string());
        }

        if self.ensemble_size == 0 {
            return Err("Ensemble size must be greater than 0".to_string());
        }

        Ok(())
    }

    /// Get estimated memory requirements (MB)
    pub fn estimated_memory_mb(&self) -> usize {
        // Rough estimate based on quantum state size and ensemble
        let state_size = 2_usize.pow(self.n_qubits as u32);
        let memory_per_state = state_size * 16; // Complex64 = 16 bytes
        let total_memory = memory_per_state * self.ensemble_size;
        (total_memory / (1024 * 1024)).max(1)
    }

    /// Get estimated computational complexity
    pub fn estimated_complexity(&self) -> usize {
        // Rough estimate of computational operations
        let state_size = 2_usize.pow(self.n_qubits as u32);
        let operations_per_layer = state_size * self.n_qubits;
        operations_per_layer * self.n_layers * self.ensemble_size
    }

    /// Check if configuration is suitable for real-time processing
    pub fn is_real_time_capable(&self) -> bool {
        self.estimated_memory_mb() < 1024 && // Less than 1GB
        self.estimated_complexity() < 1_000_000 && // Reasonable complexity
        self.n_qubits <= 12 // Manageable qubit count
    }

    /// Get recommended batch size for processing
    pub fn recommended_batch_size(&self) -> usize {
        let complexity = self.estimated_complexity();
        if complexity < 10_000 {
            100
        } else if complexity < 100_000 {
            50
        } else if complexity < 1_000_000 {
            10
        } else {
            1
        }
    }

    /// Create a lightweight configuration for testing
    pub fn lightweight() -> Self {
        Self {
            n_qubits: 4,
            n_layers: 2,
            ensemble_size: 3,
            max_iterations: 100,
            ..Default::default()
        }
    }

    /// Create a high-performance configuration
    pub fn high_performance() -> Self {
        Self {
            n_qubits: 12,
            n_layers: 5,
            ensemble_size: 10,
            max_iterations: 5000,
            performance_config: PerformanceConfig {
                enable_parallel: true,
                n_threads: num_cpus::get(),
                enable_gpu: true,
                memory_limit_mb: 16384,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Create a research configuration with full features
    pub fn research() -> Self {
        Self {
            n_qubits: 8,
            n_layers: 4,
            ensemble_size: 7,
            enable_noise: true,
            noise_config: NoiseConfig {
                depolarizing_rate: 0.01,
                phase_damping_rate: 0.01,
                ..Default::default()
            },
            performance_config: PerformanceConfig {
                enable_profiling: true,
                enable_monitoring: true,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_default_config() {
        let config = QuantumConfig::default();
        assert_eq!(config.n_qubits, 8);
        assert_eq!(config.n_layers, 3);
        assert_eq!(config.ensemble_size, 5);
        assert_eq!(config.confidence_level, 0.95);
    }

    #[test]
    fn test_config_validation() {
        let mut config = QuantumConfig::default();
        assert!(config.validate().is_ok());

        config.n_qubits = 0;
        assert!(config.validate().is_err());

        config.n_qubits = 8;
        config.confidence_level = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_memory_estimation() {
        let config = QuantumConfig::default();
        let memory = config.estimated_memory_mb();
        assert!(memory > 0);
    }

    #[test]
    fn test_complexity_estimation() {
        let config = QuantumConfig::default();
        let complexity = config.estimated_complexity();
        assert!(complexity > 0);
    }

    #[test]
    fn test_real_time_capability() {
        let lightweight = QuantumConfig::lightweight();
        assert!(lightweight.is_real_time_capable());

        let heavy = QuantumConfig {
            n_qubits: 20,
            ..Default::default()
        };
        assert!(!heavy.is_real_time_capable());
    }

    #[test]
    fn test_config_serialization() {
        let config = QuantumConfig::default();
        let serialized = toml::to_string(&config).unwrap();
        let deserialized: QuantumConfig = toml::from_str(&serialized).unwrap();
        assert_eq!(config.n_qubits, deserialized.n_qubits);
    }

    #[test]
    fn test_config_file_operations() {
        let config = QuantumConfig::default();
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_config.toml");

        // Test saving
        config.to_file(&file_path).unwrap();
        assert!(file_path.exists());

        // Test loading
        let loaded_config = QuantumConfig::from_file(&file_path).unwrap();
        assert_eq!(config.n_qubits, loaded_config.n_qubits);
    }

    #[test]
    fn test_specialized_configs() {
        let lightweight = QuantumConfig::lightweight();
        assert_eq!(lightweight.n_qubits, 4);

        let high_perf = QuantumConfig::high_performance();
        assert_eq!(high_perf.n_qubits, 12);

        let research = QuantumConfig::research();
        assert!(research.enable_noise);
    }

    #[test]
    fn test_recommended_batch_size() {
        let lightweight = QuantumConfig::lightweight();
        let batch_size = lightweight.recommended_batch_size();
        assert!(batch_size > 0);
    }
}