//! Core types for quantum-enhanced pattern recognition

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use uuid::Uuid;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

/// Quantum signal containing collapsed pattern information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSignal {
    /// Signal ID
    pub id: Uuid,
    /// Signal strength (-1.0 to 1.0)
    pub strength: f64,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Pattern type detected
    pub pattern_type: QuantumPatternType,
    /// Quantum coherence level
    pub coherence: f64,
    /// Entanglement correlation map
    pub entanglement_map: HashMap<String, f64>,
    /// Frequency domain characteristics
    pub frequency_signature: Array1<f64>,
    /// Timestamp of pattern detection
    pub timestamp: DateTime<Utc>,
    /// Execution time in microseconds
    pub execution_time_us: u64,
    /// Affected assets/instruments
    pub affected_instruments: Vec<String>,
    /// Pattern metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Types of quantum patterns that can be detected
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum QuantumPatternType {
    /// Quantum superposition momentum pattern
    SuperpositionMomentum,
    /// Entangled cross-asset correlation
    EntangledCorrelation,
    /// Quantum interference pattern
    QuantumInterference,
    /// Coherent oscillation pattern
    CoherentOscillation,
    /// Quantum tunneling breakthrough
    QuantumTunneling,
    /// Quantum decoherence warning
    DecoherenceWarning,
    /// Phase transition pattern
    PhaseTransition,
    /// Quantum resonance pattern
    QuantumResonance,
}

/// Market data represented in quantum superposition space
#[derive(Debug, Clone)]
pub struct QuantumMarketData {
    /// Superposition states of price movements
    pub superposition_states: Array2<Complex64>,
    /// Quantum amplitudes for each state
    pub amplitudes: Array1<Complex64>,
    /// Entanglement matrix between instruments
    pub entanglement_matrix: Array2<Complex64>,
    /// Quantum phase information
    pub phase_matrix: Array2<f64>,
    /// Classical market data for reference
    pub classical_data: MarketData,
    /// Coherence time estimate
    pub coherence_time_ms: f64,
}

/// Classical market data input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    /// Price history for instruments
    pub price_history: HashMap<String, Vec<f64>>,
    /// Volume data
    pub volume_data: HashMap<String, Vec<f64>>,
    /// Timestamp series
    pub timestamps: Vec<DateTime<Utc>>,
    /// Market features (technical indicators)
    pub features: Array2<f64>,
    /// Market regime indicators
    pub regime_indicators: Array1<f64>,
}

/// Quantum entanglement correlation result
#[derive(Debug, Clone)]
pub struct EntanglementCorrelation {
    /// Entanglement strength (0.0 to 1.0)
    pub strength: f64,
    /// Entangled instrument pairs
    pub entangled_pairs: Vec<(String, String)>,
    /// Correlation matrix in quantum space
    pub correlation_matrix: Array2<Complex64>,
    /// Bell state coefficients
    pub bell_coefficients: Array1<Complex64>,
    /// Entanglement fidelity
    pub fidelity: f64,
    /// Decoherence rate
    pub decoherence_rate: f64,
}

/// Quantum Fourier Transform result
#[derive(Debug, Clone)]
pub struct QuantumFourierResult {
    /// Frequency domain representation
    pub frequency_amplitudes: Array1<Complex64>,
    /// Quantum phase information
    pub phase_spectrum: Array1<f64>,
    /// Dominant frequencies
    pub dominant_frequencies: Vec<f64>,
    /// Frequency-domain entanglement
    pub frequency_entanglement: Array2<Complex64>,
    /// Spectral coherence
    pub spectral_coherence: f64,
}

/// Configuration for quantum pattern detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumConfig {
    /// Maximum superposition states to consider
    pub max_superposition_states: usize,
    /// Quantum coherence threshold
    pub coherence_threshold: f64,
    /// Entanglement detection sensitivity
    pub entanglement_sensitivity: f64,
    /// Frequency resolution for QFT
    pub frequency_resolution: f64,
    /// Claude Flow integration settings
    pub claude_flow: ClaudeFlowQuantumConfig,
    /// Performance optimization settings
    pub performance: QuantumPerformanceConfig,
    /// SIMD acceleration settings
    pub simd: QuantumSimdConfig,
}

/// Claude Flow integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeFlowQuantumConfig {
    /// Enable swarm coordination
    pub enable_swarm: bool,
    /// Number of quantum agents
    pub quantum_agents: usize,
    /// Quantum memory namespace
    pub memory_namespace: String,
    /// Real-time pattern sharing
    pub real_time_sharing: bool,
}

/// Performance configuration for quantum operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumPerformanceConfig {
    /// Target latency for quantum operations (microseconds)
    pub target_latency_us: u64,
    /// Maximum concurrent quantum calculations
    pub max_concurrent_calculations: usize,
    /// Memory pool size for quantum states
    pub memory_pool_size_mb: usize,
    /// Enable GPU acceleration
    pub enable_gpu: bool,
}

/// SIMD optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSimdConfig {
    /// Enable SIMD operations
    pub enable_simd: bool,
    /// Vector width for SIMD operations
    pub vector_width: usize,
    /// Parallel processing threads
    pub parallel_threads: usize,
}

/// Performance metrics for quantum pattern detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumPerformanceMetrics {
    /// Average detection latency (microseconds)
    pub avg_detection_latency_us: f64,
    /// Pattern detection success rate
    pub detection_success_rate: f64,
    /// Quantum coherence preservation rate
    pub coherence_preservation_rate: f64,
    /// Memory utilization efficiency
    pub memory_efficiency: f64,
    /// CPU utilization for quantum calculations
    pub cpu_utilization: f64,
    /// GPU utilization (if enabled)
    pub gpu_utilization: Option<f64>,
    /// Patterns detected per second
    pub patterns_per_second: f64,
    /// Total quantum calculations performed
    pub total_calculations: u64,
    /// Error rate in quantum operations
    pub quantum_error_rate: f64,
}

/// Result of quantum pattern validation
#[derive(Debug, Clone)]
pub struct QuantumValidationResult {
    /// Whether the pattern is valid
    pub is_valid: bool,
    /// Validation confidence
    pub confidence: f64,
    /// Classical correlation with the quantum pattern
    pub classical_correlation: f64,
    /// Predicted persistence time
    pub persistence_time_ms: f64,
    /// Trading signal strength
    pub signal_strength: f64,
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            max_superposition_states: 1024,
            coherence_threshold: 0.7,
            entanglement_sensitivity: 0.5,
            frequency_resolution: 0.001,
            claude_flow: ClaudeFlowQuantumConfig {
                enable_swarm: true,
                quantum_agents: 4,
                memory_namespace: "quantum-patterns".to_string(),
                real_time_sharing: true,
            },
            performance: QuantumPerformanceConfig {
                target_latency_us: 100,
                max_concurrent_calculations: 8,
                memory_pool_size_mb: 512,
                enable_gpu: true,
            },
            simd: QuantumSimdConfig {
                enable_simd: true,
                vector_width: 8,
                parallel_threads: num_cpus::get(),
            },
        }
    }
}

impl QuantumSignal {
    /// Create a new quantum signal
    pub fn new(
        strength: f64,
        confidence: f64,
        pattern_type: QuantumPatternType,
        coherence: f64,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            strength,
            confidence,
            pattern_type,
            coherence,
            entanglement_map: HashMap::new(),
            frequency_signature: Array1::zeros(0),
            timestamp: Utc::now(),
            execution_time_us: 0,
            affected_instruments: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Check if the signal is strong enough for trading
    pub fn is_tradeable(&self, min_confidence: f64, min_coherence: f64) -> bool {
        self.confidence >= min_confidence && 
        self.coherence >= min_coherence &&
        self.strength.abs() > 0.1
    }

    /// Get the trading direction from the signal
    pub fn trading_direction(&self) -> Option<TradingDirection> {
        if self.strength > 0.1 {
            Some(TradingDirection::Long)
        } else if self.strength < -0.1 {
            Some(TradingDirection::Short)
        } else {
            None
        }
    }
}

/// Trading direction indicated by quantum signal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingDirection {
    Long,
    Short,
}

impl std::fmt::Display for QuantumPatternType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantumPatternType::SuperpositionMomentum => write!(f, "Superposition Momentum"),
            QuantumPatternType::EntangledCorrelation => write!(f, "Entangled Correlation"),
            QuantumPatternType::QuantumInterference => write!(f, "Quantum Interference"),
            QuantumPatternType::CoherentOscillation => write!(f, "Coherent Oscillation"),
            QuantumPatternType::QuantumTunneling => write!(f, "Quantum Tunneling"),
            QuantumPatternType::DecoherenceWarning => write!(f, "Decoherence Warning"),
            QuantumPatternType::PhaseTransition => write!(f, "Phase Transition"),
            QuantumPatternType::QuantumResonance => write!(f, "Quantum Resonance"),
        }
    }
}