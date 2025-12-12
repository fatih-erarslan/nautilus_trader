//! Constants and default values for Quantum Agentic Reasoning

use std::f64::consts::PI;

/// Number of standard factors used in QAR
pub const NUM_STANDARD_FACTORS: usize = 8;

/// Default decision threshold
pub const DEFAULT_DECISION_THRESHOLD: f64 = 0.3;

/// Default memory length for decision history
pub const DEFAULT_MEMORY_LENGTH: usize = 50;

/// Default quantum fallback threshold
pub const DEFAULT_QUANTUM_FALLBACK_THRESHOLD: usize = 3;

/// Default number of qubits for quantum circuits
pub const DEFAULT_NUM_QUBITS: usize = 8;

/// Maximum number of qubits supported
pub const MAX_NUM_QUBITS: usize = 32;

/// Default circuit cache size
pub const DEFAULT_CIRCUIT_CACHE_SIZE: usize = 100;

/// Default learning rate
pub const DEFAULT_LEARNING_RATE: f64 = 0.01;

/// Default adaptation rate
pub const DEFAULT_ADAPTATION_RATE: f64 = 0.05;

/// Quantum circuit execution timeout in milliseconds
pub const QUANTUM_EXECUTION_TIMEOUT_MS: u64 = 10000;

/// Classical fallback execution timeout in milliseconds
pub const CLASSICAL_FALLBACK_TIMEOUT_MS: u64 = 1000;

/// Default number of measurement shots
pub const DEFAULT_MEASUREMENT_SHOTS: usize = 1000;

/// Minimum confidence threshold for pattern matching
pub const MIN_PATTERN_CONFIDENCE: f64 = 0.5;

/// Maximum pattern storage limit
pub const MAX_PATTERN_STORAGE: usize = 1000;

/// Default pattern similarity threshold
pub const DEFAULT_PATTERN_SIMILARITY_THRESHOLD: f64 = 0.7;

/// Quantum Fourier Transform related constants
pub mod qft {
    use super::PI;
    
    /// QFT rotation angle multiplier
    pub const ROTATION_MULTIPLIER: f64 = 2.0 * PI;
    
    /// QFT precision threshold
    pub const PRECISION_THRESHOLD: f64 = 1e-10;
    
    /// Default QFT iterations
    pub const DEFAULT_ITERATIONS: usize = 10;
}

/// Decision optimization constants
pub mod decision {
    /// Amplitude amplification iterations
    pub const AMPLITUDE_AMPLIFICATION_ITERATIONS: usize = 5;
    
    /// Decision weight normalization factor
    pub const WEIGHT_NORMALIZATION_FACTOR: f64 = 0.5;
    
    /// Information gain threshold
    pub const INFORMATION_GAIN_THRESHOLD: f64 = 0.1;
    
    /// Minimum decision confidence
    pub const MIN_DECISION_CONFIDENCE: f64 = 0.2;
    
    /// Maximum decision confidence
    pub const MAX_DECISION_CONFIDENCE: f64 = 0.95;
}

/// Pattern recognition constants
pub mod pattern {
    /// Oracle threshold for pattern matching
    pub const ORACLE_THRESHOLD: f64 = 0.8;
    
    /// Pattern encoding precision
    pub const ENCODING_PRECISION: usize = 16;
    
    /// Maximum pattern features
    pub const MAX_PATTERN_FEATURES: usize = 64;
    
    /// Pattern matching timeout in milliseconds
    pub const PATTERN_MATCHING_TIMEOUT_MS: u64 = 5000;
}

/// Hardware acceleration constants
pub mod hardware {
    /// GPU memory threshold in MB
    pub const GPU_MEMORY_THRESHOLD_MB: usize = 1024;
    
    /// Quantum device connection timeout in milliseconds
    pub const QUANTUM_DEVICE_TIMEOUT_MS: u64 = 5000;
    
    /// Maximum concurrent quantum operations
    pub const MAX_CONCURRENT_QUANTUM_OPS: usize = 10;
    
    /// Hardware detection timeout in milliseconds
    pub const HARDWARE_DETECTION_TIMEOUT_MS: u64 = 3000;
}

/// Market analysis constants
pub mod market {
    /// Volatility regime thresholds
    pub const LOW_VOLATILITY_THRESHOLD: f64 = 0.15;
    pub const MEDIUM_VOLATILITY_THRESHOLD: f64 = 0.35;
    pub const HIGH_VOLATILITY_THRESHOLD: f64 = 0.55;
    
    /// Trend strength thresholds
    pub const WEAK_TREND_THRESHOLD: f64 = 0.3;
    pub const MODERATE_TREND_THRESHOLD: f64 = 0.6;
    pub const STRONG_TREND_THRESHOLD: f64 = 0.8;
    
    /// Market phase transition thresholds
    pub const PHASE_TRANSITION_THRESHOLD: f64 = 0.5;
    pub const PHASE_STABILITY_THRESHOLD: f64 = 0.7;
    
    /// Spectral analysis parameters
    pub const SPECTRAL_WINDOW_SIZE: usize = 64;
    pub const SPECTRAL_OVERLAP: f64 = 0.5;
    pub const SPECTRAL_POWER_THRESHOLD: f64 = 0.1;
}

/// Performance monitoring constants
pub mod performance {
    /// Metrics collection interval in milliseconds
    pub const METRICS_COLLECTION_INTERVAL_MS: u64 = 60000;
    
    /// Performance window size for averaging
    pub const PERFORMANCE_WINDOW_SIZE: usize = 100;
    
    /// Memory usage warning threshold in MB
    pub const MEMORY_WARNING_THRESHOLD_MB: usize = 1024;
    
    /// CPU usage warning threshold (0.0 to 1.0)
    pub const CPU_WARNING_THRESHOLD: f64 = 0.8;
    
    /// Quantum operation latency threshold in milliseconds
    pub const QUANTUM_LATENCY_THRESHOLD_MS: f64 = 100.0;
}

/// Cache management constants
pub mod cache {
    /// Default cache TTL in seconds
    pub const DEFAULT_CACHE_TTL_SECONDS: u64 = 3600;
    
    /// Cache cleanup interval in seconds
    pub const CACHE_CLEANUP_INTERVAL_SECONDS: u64 = 300;
    
    /// Maximum cache memory usage in MB
    pub const MAX_CACHE_MEMORY_MB: usize = 512;
    
    /// Cache hit ratio warning threshold
    pub const CACHE_HIT_RATIO_WARNING: f64 = 0.5;
}

/// Error handling constants
pub mod error {
    /// Maximum retry attempts
    pub const MAX_RETRY_ATTEMPTS: usize = 3;
    
    /// Retry delay in milliseconds
    pub const RETRY_DELAY_MS: u64 = 1000;
    
    /// Circuit execution retry multiplier
    pub const CIRCUIT_RETRY_MULTIPLIER: f64 = 1.5;
    
    /// Error recovery timeout in milliseconds
    pub const ERROR_RECOVERY_TIMEOUT_MS: u64 = 5000;
}

/// Serialization constants
pub mod serialization {
    /// JSON serialization buffer size
    pub const JSON_BUFFER_SIZE: usize = 8192;
    
    /// Binary serialization buffer size
    pub const BINARY_BUFFER_SIZE: usize = 4096;
    
    /// Compression threshold in bytes
    pub const COMPRESSION_THRESHOLD: usize = 1024;
    
    /// Serialization timeout in milliseconds
    pub const SERIALIZATION_TIMEOUT_MS: u64 = 1000;
}

/// Logging constants
pub mod logging {
    /// Default log buffer size
    pub const LOG_BUFFER_SIZE: usize = 1024;
    
    /// Log file rotation size in MB
    pub const LOG_FILE_ROTATION_SIZE_MB: usize = 100;
    
    /// Maximum log files to keep
    pub const MAX_LOG_FILES: usize = 5;
    
    /// Performance log interval in milliseconds
    pub const PERFORMANCE_LOG_INTERVAL_MS: u64 = 10000;
}

/// Mathematical constants
pub mod math {
    /// Machine epsilon for floating point comparisons
    pub const EPSILON: f64 = 1e-10;
    
    /// Golden ratio
    pub const GOLDEN_RATIO: f64 = 1.618033988749895;
    
    /// Euler's number
    pub const E: f64 = std::f64::consts::E;
    
    /// Pi
    pub const PI: f64 = std::f64::consts::PI;
    
    /// Tau (2 * Pi)
    pub const TAU: f64 = 2.0 * std::f64::consts::PI;
    
    /// Natural logarithm of 2
    pub const LN_2: f64 = std::f64::consts::LN_2;
    
    /// Square root of 2
    pub const SQRT_2: f64 = std::f64::consts::SQRT_2;
    
    /// Quantum phase gate precision
    pub const PHASE_PRECISION: f64 = 1e-8;
    
    /// Normalization epsilon
    pub const NORMALIZATION_EPSILON: f64 = 1e-12;
}

/// Quantum circuit gate parameters
pub mod gates {
    use super::PI;
    
    /// Hadamard gate angle
    pub const HADAMARD_ANGLE: f64 = PI / 2.0;
    
    /// Pauli-X rotation angle
    pub const PAULI_X_ANGLE: f64 = PI;
    
    /// Pauli-Y rotation angle
    pub const PAULI_Y_ANGLE: f64 = PI;
    
    /// Pauli-Z rotation angle
    pub const PAULI_Z_ANGLE: f64 = PI;
    
    /// Default rotation angle
    pub const DEFAULT_ROTATION_ANGLE: f64 = PI / 4.0;
    
    /// Phase gate angle
    pub const PHASE_GATE_ANGLE: f64 = PI / 2.0;
    
    /// Controlled gate threshold
    pub const CONTROLLED_GATE_THRESHOLD: f64 = 0.5;
}

/// Version and build information
pub mod version {
    /// Library version
    pub const VERSION: &str = env!("CARGO_PKG_VERSION");
    
    /// Build timestamp
    pub const BUILD_TIMESTAMP: &str = "unknown";
    
    /// Git commit hash  
    pub const GIT_COMMIT: &str = "unknown";
    
    /// Rust version used for compilation
    pub const RUSTC_VERSION: &str = "unknown";
}

/// Feature flags for conditional compilation
pub mod features {
    /// SIMD support available
    pub const SIMD_SUPPORT: bool = cfg!(feature = "simd");
    
    /// GPU support available
    pub const GPU_SUPPORT: bool = cfg!(feature = "gpu");
    
    /// Quantum hardware support available
    pub const QUANTUM_HARDWARE_SUPPORT: bool = cfg!(feature = "quantum-hardware");
    
    /// Performance monitoring enabled
    pub const PERFORMANCE_MONITORING: bool = cfg!(feature = "performance-monitoring");
    
    /// Experimental features enabled
    pub const EXPERIMENTAL_FEATURES: bool = cfg!(feature = "experimental");
}

/// Utility functions for constants
pub mod utils {
    /// Check if a value is within epsilon of zero
    pub fn is_zero(value: f64) -> bool {
        value.abs() < super::math::EPSILON
    }
    
    /// Clamp value between min and max
    pub fn clamp(value: f64, min: f64, max: f64) -> f64 {
        value.max(min).min(max)
    }
    
    /// Normalize angle to [0, 2π) range
    pub fn normalize_angle(angle: f64) -> f64 {
        angle.rem_euclid(super::math::TAU)
    }
    
    /// Convert degrees to radians
    pub fn deg_to_rad(degrees: f64) -> f64 {
        degrees * super::math::PI / 180.0
    }
    
    /// Convert radians to degrees
    pub fn rad_to_deg(radians: f64) -> f64 {
        radians * 180.0 / super::math::PI
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(NUM_STANDARD_FACTORS, 8);
        assert!(DEFAULT_DECISION_THRESHOLD > 0.0);
        assert!(DEFAULT_DECISION_THRESHOLD < 1.0);
        assert!(DEFAULT_MEMORY_LENGTH > 0);
        assert!(MAX_NUM_QUBITS > DEFAULT_NUM_QUBITS);
    }

    #[test]
    fn test_utility_functions() {
        assert!(utils::is_zero(0.0));
        assert!(utils::is_zero(1e-11));
        assert!(!utils::is_zero(0.1));
        
        assert_eq!(utils::clamp(0.5, 0.0, 1.0), 0.5);
        assert_eq!(utils::clamp(-0.5, 0.0, 1.0), 0.0);
        assert_eq!(utils::clamp(1.5, 0.0, 1.0), 1.0);
        
        let angle = utils::normalize_angle(3.0 * math::PI);
        assert!((angle - math::PI).abs() < math::EPSILON);
    }

    #[test]
    fn test_market_constants() {
        assert!(market::LOW_VOLATILITY_THRESHOLD < market::MEDIUM_VOLATILITY_THRESHOLD);
        assert!(market::MEDIUM_VOLATILITY_THRESHOLD < market::HIGH_VOLATILITY_THRESHOLD);
        assert!(market::WEAK_TREND_THRESHOLD < market::MODERATE_TREND_THRESHOLD);
        assert!(market::MODERATE_TREND_THRESHOLD < market::STRONG_TREND_THRESHOLD);
    }

    #[test]
    fn test_quantum_constants() {
        assert!(qft::ROTATION_MULTIPLIER > 0.0);
        assert!(qft::PRECISION_THRESHOLD > 0.0);
        assert!(qft::DEFAULT_ITERATIONS > 0);
        
        assert!(decision::AMPLITUDE_AMPLIFICATION_ITERATIONS > 0);
        assert!(decision::MIN_DECISION_CONFIDENCE < decision::MAX_DECISION_CONFIDENCE);
        
        assert!(pattern::ORACLE_THRESHOLD > 0.0);
        assert!(pattern::ORACLE_THRESHOLD <= 1.0);
        assert!(pattern::MAX_PATTERN_FEATURES > 0);
    }

    #[test]
    fn test_hardware_constants() {
        assert!(hardware::GPU_MEMORY_THRESHOLD_MB > 0);
        assert!(hardware::QUANTUM_DEVICE_TIMEOUT_MS > 0);
        assert!(hardware::MAX_CONCURRENT_QUANTUM_OPS > 0);
        assert!(hardware::HARDWARE_DETECTION_TIMEOUT_MS > 0);
    }
}