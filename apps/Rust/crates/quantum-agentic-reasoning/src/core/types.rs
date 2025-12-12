//! Core type definitions for Quantum Agentic Reasoning

use std::collections::HashMap;
use std::fmt::Debug;
use serde::{Deserialize, Serialize};

/// Quantum circuit execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumResult {
    /// Expectation values from quantum measurements
    pub expectation_values: Vec<f64>,
    /// Measurement probabilities
    pub probabilities: Vec<f64>,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Whether quantum hardware was used
    pub used_quantum: bool,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl QuantumResult {
    /// Create a new quantum result
    pub fn new(expectation_values: Vec<f64>, execution_time_ms: f64, used_quantum: bool) -> Self {
        Self {
            expectation_values,
            probabilities: Vec::new(),
            execution_time_ms,
            used_quantum,
            metadata: HashMap::new(),
        }
    }

    /// Add probability distribution
    pub fn with_probabilities(mut self, probabilities: Vec<f64>) -> Self {
        self.probabilities = probabilities;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Get the primary result (first expectation value)
    pub fn primary_result(&self) -> Option<f64> {
        self.expectation_values.first().copied()
    }

    /// Get all results as a vector
    pub fn results(&self) -> &[f64] {
        &self.expectation_values
    }
}

/// Pattern matching result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatch {
    /// Pattern identifier
    pub pattern_id: String,
    /// Similarity score (0.0 to 1.0)
    pub similarity: f64,
    /// Confidence in the match
    pub confidence: f64,
    /// Pattern metadata
    pub metadata: HashMap<String, String>,
}

impl PatternMatch {
    /// Create a new pattern match
    pub fn new(pattern_id: String, similarity: f64, confidence: f64) -> Self {
        Self {
            pattern_id,
            similarity,
            confidence,
            metadata: HashMap::new(),
        }
    }

    /// Check if this is a strong match
    pub fn is_strong_match(&self, threshold: f64) -> bool {
        self.similarity >= threshold && self.confidence >= threshold
    }
}

/// Market regime analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeAnalysis {
    /// Detected market phase
    pub phase: super::MarketPhase,
    /// Confidence in the phase detection
    pub confidence: f64,
    /// Regime strength indicator
    pub strength: f64,
    /// Volatility estimate
    pub volatility: f64,
    /// Noise level
    pub noise_level: f64,
    /// Spectral characteristics
    pub spectral_power: Vec<f64>,
    /// Phase coherence
    pub phase_coherence: f64,
}

impl RegimeAnalysis {
    /// Create a new regime analysis
    pub fn new(phase: super::MarketPhase, confidence: f64, strength: f64) -> Self {
        Self {
            phase,
            confidence,
            strength,
            volatility: 0.0,
            noise_level: 0.0,
            spectral_power: Vec::new(),
            phase_coherence: 0.0,
        }
    }

    /// Check if the regime is stable
    pub fn is_stable(&self) -> bool {
        self.phase.is_stable() && self.confidence > 0.7 && self.noise_level < 0.3
    }

    /// Check if the regime is in transition
    pub fn is_transitioning(&self) -> bool {
        self.confidence < 0.5 || self.noise_level > 0.5
    }
}

/// Decision optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionOptimization {
    /// Optimized decision weights
    pub weights: Vec<f64>,
    /// Decision confidence
    pub confidence: f64,
    /// Information gain
    pub information_gain: f64,
    /// Optimization metadata
    pub metadata: HashMap<String, f64>,
}

impl DecisionOptimization {
    /// Create a new decision optimization result
    pub fn new(weights: Vec<f64>, confidence: f64, information_gain: f64) -> Self {
        Self {
            weights,
            confidence,
            information_gain,
            metadata: HashMap::new(),
        }
    }

    /// Get the strongest weighted factor
    pub fn strongest_factor(&self) -> Option<(usize, f64)> {
        self.weights
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &w)| (i, w))
    }
}

/// Circuit execution parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitParams {
    /// Input parameters for the circuit
    pub parameters: Vec<f64>,
    /// Number of qubits to use
    pub num_qubits: usize,
    /// Number of measurement shots
    pub shots: Option<usize>,
    /// Circuit-specific options
    pub options: HashMap<String, f64>,
}

impl CircuitParams {
    /// Create new circuit parameters
    pub fn new(parameters: Vec<f64>, num_qubits: usize) -> Self {
        Self {
            parameters,
            num_qubits,
            shots: None,
            options: HashMap::new(),
        }
    }

    /// Set the number of shots
    pub fn with_shots(mut self, shots: usize) -> Self {
        self.shots = Some(shots);
        self
    }

    /// Add an option
    pub fn with_option(mut self, key: String, value: f64) -> Self {
        self.options.insert(key, value);
        self
    }

    /// Get an option value
    pub fn get_option(&self, key: &str) -> Option<f64> {
        self.options.get(key).copied()
    }
}

/// Hardware performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareMetrics {
    /// Quantum execution time (ms)
    pub quantum_time_ms: f64,
    /// Classical execution time (ms)
    pub classical_time_ms: f64,
    /// Memory usage (MB)
    pub memory_usage_mb: f64,
    /// GPU utilization (0.0 to 1.0)
    pub gpu_utilization: f64,
    /// Quantum gate count
    pub quantum_gates: usize,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
}

impl Default for HardwareMetrics {
    fn default() -> Self {
        Self {
            quantum_time_ms: 0.0,
            classical_time_ms: 0.0,
            memory_usage_mb: 0.0,
            gpu_utilization: 0.0,
            quantum_gates: 0,
            cache_hit_ratio: 0.0,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Cache size
    pub size: usize,
    /// Cache capacity
    pub capacity: usize,
}

impl CacheStats {
    /// Create new cache stats
    pub fn new(capacity: usize) -> Self {
        Self {
            hits: 0,
            misses: 0,
            size: 0,
            capacity,
        }
    }

    /// Record a cache hit
    pub fn record_hit(&mut self) {
        self.hits += 1;
    }

    /// Record a cache miss
    pub fn record_miss(&mut self) {
        self.misses += 1;
    }

    /// Get hit ratio
    pub fn hit_ratio(&self) -> f64 {
        if self.hits + self.misses == 0 {
            0.0
        } else {
            self.hits as f64 / (self.hits + self.misses) as f64
        }
    }

    /// Get total requests
    pub fn total_requests(&self) -> u64 {
        self.hits + self.misses
    }
}

/// Execution context for quantum operations
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Prefer quantum execution
    pub prefer_quantum: bool,
    /// Maximum execution time
    pub max_execution_time_ms: u64,
    /// Number of retries on failure
    pub max_retries: usize,
    /// Circuit cache enabled
    pub cache_enabled: bool,
    /// Performance monitoring enabled
    pub monitoring_enabled: bool,
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self {
            prefer_quantum: true,
            max_execution_time_ms: 10000, // 10 seconds
            max_retries: 3,
            cache_enabled: true,
            monitoring_enabled: true,
        }
    }
}

/// Error recovery information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorRecovery {
    /// Number of failures encountered
    pub failure_count: usize,
    /// Last error message
    pub last_error: String,
    /// Recovery strategy used
    pub recovery_strategy: String,
    /// Whether recovery was successful
    pub recovered: bool,
    /// Recovery timestamp
    pub recovery_time: chrono::DateTime<chrono::Utc>,
}

impl ErrorRecovery {
    /// Create a new error recovery record
    pub fn new(error: String, strategy: String) -> Self {
        Self {
            failure_count: 1,
            last_error: error,
            recovery_strategy: strategy,
            recovered: false,
            recovery_time: chrono::Utc::now(),
        }
    }

    /// Record successful recovery
    pub fn mark_recovered(&mut self) {
        self.recovered = true;
        self.recovery_time = chrono::Utc::now();
    }

    /// Increment failure count
    pub fn increment_failures(&mut self) {
        self.failure_count += 1;
    }
}

/// Decision outcome for feedback
#[derive(Debug, Clone)]
pub enum DecisionOutcome {
    /// Decision was successful
    Success { profit: f64, duration_ms: u64 },
    /// Decision was unsuccessful
    Failure { loss: f64, duration_ms: u64 },
    /// Decision is still pending
    Pending,
}

/// Decision engine metrics
#[derive(Debug, Clone)]
pub struct DecisionMetrics {
    /// Total decisions made
    pub total_decisions: u64,
    /// Successful decisions
    pub successful_decisions: u64,
    /// Average confidence
    pub average_confidence: f64,
    /// Average execution time
    pub average_execution_time_ms: f64,
    /// Last decision timestamp
    pub last_decision_time: chrono::DateTime<chrono::Utc>,
}

/// Market prediction result
#[derive(Debug, Clone)]
pub struct MarketPrediction {
    /// Predicted direction (-1.0 to 1.0)
    pub direction: f64,
    /// Prediction confidence (0.0 to 1.0)
    pub confidence: f64,
    /// Time horizon for prediction
    pub time_horizon_ms: u64,
    /// Supporting factors
    pub factors: HashMap<String, f64>,
}

/// Pattern data structure
#[derive(Debug, Clone)]
pub struct PatternData {
    /// Pattern identifier
    pub id: String,
    /// Pattern features
    pub features: Vec<f64>,
    /// Pattern timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Pattern metadata
    pub metadata: HashMap<String, String>,
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Number of stored decisions
    pub decisions_stored: usize,
    /// Number of stored patterns
    pub patterns_stored: usize,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Average execution time by operation
    pub execution_times: HashMap<String, f64>,
    /// Quantum usage percentage
    pub quantum_usage_percent: f64,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
    /// Total operations performed
    pub total_operations: u64,
    /// Operations per second
    pub ops_per_second: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_result() {
        let result = QuantumResult::new(vec![0.5, 0.3, 0.2], 100.0, true);
        assert_eq!(result.primary_result(), Some(0.5));
        assert_eq!(result.results().len(), 3);
        assert!(result.used_quantum);
    }

    #[test]
    fn test_pattern_match() {
        let pattern = PatternMatch::new("test_pattern".to_string(), 0.8, 0.9);
        assert!(pattern.is_strong_match(0.7));
        assert!(!pattern.is_strong_match(0.85));
    }

    #[test]
    fn test_regime_analysis() {
        let analysis = RegimeAnalysis::new(
            super::MarketPhase::Growth,
            0.8,
            0.7,
        );
        assert!(analysis.is_stable());
        assert!(!analysis.is_transitioning());
    }

    #[test]
    fn test_circuit_params() {
        let params = CircuitParams::new(vec![0.1, 0.2, 0.3], 4)
            .with_shots(1000)
            .with_option("temperature".to_string(), 0.5);
        
        assert_eq!(params.parameters.len(), 3);
        assert_eq!(params.num_qubits, 4);
        assert_eq!(params.shots, Some(1000));
        assert_eq!(params.get_option("temperature"), Some(0.5));
    }

    #[test]
    fn test_cache_stats() {
        let mut stats = CacheStats::new(100);
        stats.record_hit();
        stats.record_hit();
        stats.record_miss();
        
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_ratio(), 2.0 / 3.0);
        assert_eq!(stats.total_requests(), 3);
    }

    #[test]
    fn test_error_recovery() {
        let mut recovery = ErrorRecovery::new(
            "Quantum device error".to_string(),
            "fallback_to_classical".to_string(),
        );
        
        assert_eq!(recovery.failure_count, 1);
        assert!(!recovery.recovered);
        
        recovery.mark_recovered();
        assert!(recovery.recovered);
        
        recovery.increment_failures();
        assert_eq!(recovery.failure_count, 2);
    }
}