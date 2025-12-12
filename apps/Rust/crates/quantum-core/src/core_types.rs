//! Core Types and Re-exports for QAR Integration
//!
//! This module provides the missing types that the quantum-agentic-reasoning crate expects
//! from quantum-core, ensuring seamless integration between the two crates.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Re-export core types for QAR compatibility
pub use crate::traits::{
    ResourceLimits,
    CircuitComplexity,
    ResourceRequirements,
    CalibrationResult,
    QuantumJob,
    JobResult,
    JobStatus,
    JobPriority,
    QuantumData,
    MeasurementResult,
    PatternAnalysis,
    DetectedPattern,
    TrainingResult,
    PatternPrediction,
    ModelMetrics,
    AlgorithmInput,
    AlgorithmOutput,
    AlgorithmMetadata,
};

// Additional types needed by QAR that aren't in traits
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

/// Hardware metrics for performance monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareMetrics {
    /// CPU utilization percentage
    pub cpu_usage: f64,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// GPU utilization if available
    pub gpu_usage: Option<f64>,
    /// Quantum processor utilization if available
    pub qpu_usage: Option<f64>,
    /// Number of active quantum circuits
    pub active_circuits: usize,
    /// Average gate execution time in nanoseconds
    pub avg_gate_time_ns: f64,
    /// Error rates by gate type
    pub gate_error_rates: HashMap<String, f64>,
    /// Coherence time in microseconds
    pub coherence_time_us: f64,
    /// Measurement timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl Default for HardwareMetrics {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage_mb: 0.0,
            gpu_usage: None,
            qpu_usage: None,
            active_circuits: 0,
            avg_gate_time_ns: 1000.0,
            gate_error_rates: HashMap::new(),
            coherence_time_us: 100.0,
            timestamp: chrono::Utc::now(),
        }
    }
}

impl HardwareMetrics {
    /// Create new hardware metrics snapshot
    pub fn new() -> Self {
        Self::default()
    }

    /// Update CPU usage
    pub fn with_cpu_usage(mut self, usage: f64) -> Self {
        self.cpu_usage = usage.clamp(0.0, 100.0);
        self
    }

    /// Update memory usage
    pub fn with_memory_usage(mut self, usage_mb: f64) -> Self {
        self.memory_usage_mb = usage_mb.max(0.0);
        self
    }

    /// Add gate error rate
    pub fn add_gate_error_rate(mut self, gate: String, error_rate: f64) -> Self {
        self.gate_error_rates.insert(gate, error_rate.clamp(0.0, 1.0));
        self
    }

    /// Check if hardware is healthy
    pub fn is_healthy(&self) -> bool {
        self.cpu_usage < 90.0 
            && self.memory_usage_mb < 8192.0 // 8GB limit
            && self.avg_gate_time_ns < 10000.0 // 10μs limit
            && self.coherence_time_us > 10.0 // minimum coherence
    }
}

/// Hardware capabilities information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareCapabilities {
    /// Available quantum backends
    pub quantum_backends: Vec<String>,
    /// Classical simulation capabilities
    pub classical_backends: Vec<String>,
    /// Maximum qubits supported
    pub max_qubits: usize,
    /// Supported gate types
    pub supported_gates: Vec<String>,
    /// SIMD instruction sets available
    pub simd_support: Vec<String>,
    /// GPU compute capability
    pub gpu_compute: Option<String>,
    /// Available memory in MB
    pub available_memory_mb: u64,
    /// CPU core count
    pub cpu_cores: usize,
    /// Hardware features
    pub features: HashMap<String, bool>,
}

impl Default for HardwareCapabilities {
    fn default() -> Self {
        Self {
            quantum_backends: vec!["simulator".to_string()],
            classical_backends: vec!["cpu".to_string()],
            max_qubits: 32,
            supported_gates: vec![
                "H".to_string(),
                "X".to_string(),
                "Y".to_string(),
                "Z".to_string(),
                "CNOT".to_string(),
                "RZ".to_string(),
            ],
            simd_support: vec!["AVX2".to_string()],
            gpu_compute: None,
            available_memory_mb: 4096,
            cpu_cores: num_cpus::get(),
            features: HashMap::new(),
        }
    }
}

impl HardwareCapabilities {
    /// Detect current hardware capabilities
    pub fn detect() -> Self {
        let mut caps = Self::default();
        
        // Update with actual system information
        caps.available_memory_mb = 8192; // Placeholder - would use system API
        caps.cpu_cores = num_cpus::get();
        
        // Detect SIMD support
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                caps.simd_support.push("AVX2".to_string());
            }
            if is_x86_feature_detected!("avx512f") {
                caps.simd_support.push("AVX512".to_string());
            }
        }
        
        // Add quantum simulator by default
        caps.features.insert("quantum_simulator".to_string(), true);
        caps.features.insert("classical_fallback".to_string(), true);
        caps.features.insert("parallel_execution".to_string(), true);
        
        caps
    }

    /// Check if a specific gate is supported
    pub fn supports_gate(&self, gate: &str) -> bool {
        self.supported_gates.contains(&gate.to_string())
    }

    /// Check if quantum backend is available
    pub fn has_quantum_backend(&self, backend: &str) -> bool {
        self.quantum_backends.contains(&backend.to_string())
    }

    /// Get effective qubit limit for given backend
    pub fn effective_qubit_limit(&self, backend: &str) -> usize {
        match backend {
            "simulator" => self.max_qubits.min(24), // Memory limited
            "hardware" => self.max_qubits.min(16),  // Hardware limited
            _ => self.max_qubits.min(8),             // Conservative default
        }
    }
}

/// Analysis results for various market and quantum analysis operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Trend direction (-1.0 to 1.0)
    pub direction: f64,
    /// Trend strength (0.0 to 1.0)
    pub strength: f64,
    /// Confidence in analysis (0.0 to 1.0)
    pub confidence: f64,
    /// Time horizon for the trend
    pub time_horizon: chrono::Duration,
    /// Supporting indicators
    pub indicators: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketAnalysis {
    /// Market regime classification
    pub regime: String,
    /// Volatility estimate
    pub volatility: f64,
    /// Market efficiency score
    pub efficiency: f64,
    /// Liquidity indicators
    pub liquidity_metrics: HashMap<String, f64>,
    /// Risk metrics
    pub risk_metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumDecisionResult {
    /// Primary decision recommendation
    pub decision: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Alternative decisions with probabilities
    pub alternatives: Vec<(String, f64)>,
    /// Quantum advantage achieved
    pub quantum_advantage: f64,
    /// Execution metrics
    pub metrics: HashMap<String, f64>,
}

/// Decision optimization results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionOptimization {
    /// Optimized parameters
    pub parameters: HashMap<String, f64>,
    /// Expected utility
    pub expected_utility: f64,
    /// Risk-adjusted return
    pub risk_adjusted_return: f64,
    /// Optimization convergence
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
}

/// Decision execution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionMetrics {
    /// Decision accuracy over time
    pub accuracy: f64,
    /// Average execution time
    pub avg_execution_time_ms: f64,
    /// Success rate
    pub success_rate: f64,
    /// Resource utilization
    pub resource_utilization: HashMap<String, f64>,
}

/// Decision execution outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionOutcome {
    /// Outcome classification
    pub outcome: String,
    /// Realized utility
    pub realized_utility: f64,
    /// Prediction accuracy
    pub prediction_accuracy: f64,
    /// Execution timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}