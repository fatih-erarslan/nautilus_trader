//! Core types for QBMIA Unified
//!
//! All fundamental data structures and types used throughout the QBMIA system.
//! This module enforces TENGRI compliance through strict type definitions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Market data point with real timestamp and prices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataPoint {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub price: rust_decimal::Decimal,
    pub volume: u64,
    pub bid: Option<rust_decimal::Decimal>,
    pub ask: Option<rust_decimal::Decimal>,
    pub source: String, // Real API source (e.g., "alpha_vantage", "yahoo_finance")
}

/// Complete market data for multiple symbols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub symbols: Vec<String>,
    pub data_points: Vec<MarketDataPoint>,
    pub fetch_timestamp: DateTime<Utc>,
    pub source_apis: Vec<String>,
}

/// GPU device information from real hardware detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    pub id: u32,
    pub name: String,
    pub backend: GpuBackend,
    pub capabilities: GpuCapabilities,
    pub memory_info: GpuMemoryInfo,
    pub compute_units: u32,
    pub max_threads_per_block: u32,
}

/// GPU backend types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum GpuBackend {
    Cuda,
    OpenCL,
    Vulkan,
    Metal,
}

/// Real GPU capabilities detected from hardware
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuCapabilities {
    pub device_name: String,
    pub compute_capability: (u32, u32),
    pub total_memory: usize,
    pub available_memory: usize,
    pub max_threads_per_block: u32,
    pub max_shared_memory: usize,
    pub multiprocessor_count: u32,
    pub memory_bandwidth_gbps: f64,
    pub clock_rate_mhz: u32,
    pub supports_double_precision: bool,
    pub supports_half_precision: bool,
}

/// GPU memory information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMemoryInfo {
    pub total_bytes: usize,
    pub free_bytes: usize,
    pub used_bytes: usize,
    pub fragmentation_percentage: f32,
}

/// Quantum analysis result from GPU computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumAnalysis {
    pub confidence: f64,
    pub final_state: QuantumState,
    pub circuit_depth: u32,
    pub qubit_count: u32,
    pub fidelity: f64,
    pub execution_time_ms: u64,
    pub gpu_backend_used: GpuBackend,
}

/// Quantum state representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumState {
    pub amplitudes: Vec<num_complex::Complex<f64>>,
    pub qubit_count: u32,
    pub normalization: f64,
    pub entanglement_entropy: f64,
}

/// Quantum algorithm types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumAlgorithm {
    VQE { ansatz: String, optimizer: String },
    QAOA { layers: u32, mixer_hamiltonian: Vec<f64> },
    Grover { oracle_function: String, iterations: u32 },
    QFT { qubit_count: u32 },
    Shor { number_to_factor: u64 },
    Custom { name: String, parameters: HashMap<String, serde_json::Value> },
}

/// Quantum algorithm parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumParameters {
    pub qubit_count: u32,
    pub circuit_depth: u32,
    pub measurement_shots: u32,
    pub optimization_iterations: u32,
    pub convergence_threshold: f64,
    pub custom_parameters: HashMap<String, serde_json::Value>,
}

/// Quantum computation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumResult {
    pub final_state: QuantumState,
    pub measurement_results: Vec<String>, // Bit strings
    pub expectation_values: HashMap<String, f64>,
    pub fidelity: f64,
    pub convergence_achieved: bool,
    pub execution_metrics: QuantumExecutionMetrics,
}

/// Quantum execution performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumExecutionMetrics {
    pub total_time_ms: u64,
    pub compilation_time_ms: u64,
    pub execution_time_ms: u64,
    pub memory_used_bytes: usize,
    pub gpu_utilization_percent: f32,
}

/// Biological intelligence analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalAnalysis {
    pub confidence: f64,
    pub patterns: Vec<BiologicalPattern>,
    pub synaptic_strength: f64,
    pub neural_activity: NeuralActivityMap,
    pub plasticity_changes: Vec<PlasticityChange>,
    pub processing_time_ms: u64,
}

/// Biological neural pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalPattern {
    pub pattern_type: BiologicalPatternType,
    pub strength: f64,
    pub confidence: f64,
    pub activation_regions: Vec<BrainRegion>,
    pub temporal_dynamics: Vec<f64>,
}

/// Types of biological patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BiologicalPatternType {
    SpikeTrainSynchrony,
    SynapticPlasticity,
    NeuralOscillation,
    CorticalWave,
    NetworkBurst,
    AdaptiveResponse,
    MemoryConsolidation,
    AttentionalFocus,
}

/// Brain regions involved in processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BrainRegion {
    PrefrontalCortex,
    Hippocampus,
    Amygdala,
    BasalGanglia,
    Thalamus,
    Cerebellum,
    MotorCortex,
    SomatosensoryCortex,
    VisualCortex,
    AuditoryCortex,
}

/// Neural activity mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralActivityMap {
    pub activity_levels: HashMap<BrainRegion, f64>,
    pub connectivity_matrix: Vec<Vec<f64>>,
    pub oscillation_frequencies: HashMap<String, f64>,
    pub synchronization_indices: Vec<f64>,
}

/// Synaptic plasticity changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasticityChange {
    pub synapse_id: String,
    pub weight_change: f64,
    pub plasticity_type: PlasticityType,
    pub learning_rate: f64,
    pub stability_factor: f64,
}

/// Types of synaptic plasticity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlasticityType {
    LongTermPotentiation,
    LongTermDepression,
    SpikeTimingDependentPlasticity,
    HomeostasisPlasticity,
    MetaPlasticity,
}

/// Core algorithm analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreAnalysis {
    pub confidence: f64,
    pub nash_equilibrium: NashEquilibrium,
    pub machiavellian_strategies: Vec<MachiavellianStrategy>,
    pub agent_recommendations: Vec<AgentRecommendation>,
    pub game_theory_metrics: GameTheoryMetrics,
}

/// Nash equilibrium solution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NashEquilibrium {
    pub player_strategies: HashMap<String, Vec<f64>>,
    pub payoff_matrix: Vec<Vec<f64>>,
    pub equilibrium_type: EquilibriumType,
    pub stability_measure: f64,
    pub convergence_iterations: u32,
}

/// Types of Nash equilibrium
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EquilibriumType {
    Pure,
    Mixed,
    Correlated,
    StackelbergLeader,
    StackelbergFollower,
}

/// Machiavellian strategy detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MachiavellianStrategy {
    pub strategy_type: MachiavellianType,
    pub probability: f64,
    pub target_players: Vec<String>,
    pub expected_payoff: f64,
    pub risk_assessment: f64,
    pub detection_confidence: f64,
}

/// Types of Machiavellian strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MachiavellianType {
    Deception,
    Coalition,
    Manipulation,
    Information_Withholding,
    Strategic_Misinformation,
    Resource_Hoarding,
    Alliance_Breaking,
    Opportunistic_Defection,
}

/// Agent recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRecommendation {
    pub agent_id: String,
    pub recommended_action: TradingAction,
    pub confidence: f64,
    pub expected_return: f64,
    pub risk_level: RiskLevel,
    pub time_horizon: TimeHorizon,
    pub reasoning: String,
}

/// Trading action types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradingAction {
    Buy { quantity: f64, max_price: rust_decimal::Decimal },
    Sell { quantity: f64, min_price: rust_decimal::Decimal },
    Hold { duration: std::time::Duration },
    HedgePosition { hedge_ratio: f64, instrument: String },
    ClosePosition { position_id: String },
    ScaleIn { tranches: u32, total_quantity: f64 },
    ScaleOut { tranches: u32, total_quantity: f64 },
}

/// Risk level assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh,
    Extreme,
}

/// Investment time horizon
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeHorizon {
    Scalping,     // Seconds to minutes
    Intraday,     // Minutes to hours
    ShortTerm,    // Days to weeks
    MediumTerm,   // Weeks to months
    LongTerm,     // Months to years
    Strategic,    // Years to decades
}

/// Game theory performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameTheoryMetrics {
    pub social_welfare: f64,
    pub price_of_anarchy: f64,
    pub coalition_stability: f64,
    pub information_asymmetry: f64,
    pub market_efficiency: f64,
    pub strategic_complexity: f64,
}

/// Complete market analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketAnalysis {
    pub timestamp: DateTime<Utc>,
    pub symbols: Vec<String>,
    pub quantum_confidence: f64,
    pub quantum_state: QuantumState,
    pub biological_patterns: Vec<BiologicalPattern>,
    pub synaptic_strength: f64,
    pub nash_equilibrium: NashEquilibrium,
    pub machiavellian_strategies: Vec<MachiavellianStrategy>,
    pub agent_recommendations: Vec<AgentRecommendation>,
    pub fusion_confidence: f64,
    pub performance_metrics: PerformanceMetrics,
}

/// System performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub cpu_usage_percent: f32,
    pub memory_usage_bytes: usize,
    pub gpu_utilization: HashMap<String, f32>,
    pub network_throughput_mbps: f32,
    pub disk_io_mbps: f32,
    pub processing_latency_ms: u64,
    pub throughput_operations_per_second: f32,
    pub error_rate_percent: f32,
}

/// Biological input data for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalInput {
    pub neural_signals: Vec<f64>,
    pub spike_trains: Vec<Vec<f64>>,
    pub connectivity_data: Option<Vec<Vec<f64>>>,
    pub temporal_window_ms: u64,
    pub sampling_rate_hz: f32,
    pub brain_regions: Vec<BrainRegion>,
}

/// TENGRI compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TengriComplianceReport {
    pub compliance_score: f64,
    pub violations: Vec<TengriViolation>,
    pub validated_components: Vec<String>,
    pub real_data_sources: Vec<String>,
    pub hardware_verification: HardwareVerification,
    pub timestamp: DateTime<Utc>,
}

/// TENGRI compliance violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TengriViolation {
    pub component: String,
    pub violation_type: ViolationType,
    pub severity: ViolationSeverity,
    pub description: String,
    pub remediation: String,
}

/// Types of TENGRI violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    MockDataUsage,
    PlaceholderImplementation,
    RandomDataGeneration,
    SimulatedHardware,
    FakeApiEndpoint,
    SyntheticData,
    IncompleteImplementation,
    NonRealTimeData,
}

/// Severity of TENGRI violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Hardware verification results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareVerification {
    pub gpu_devices_real: bool,
    pub cpu_info_real: bool,
    pub memory_detection_real: bool,
    pub network_interfaces_real: bool,
    pub storage_devices_real: bool,
    pub verification_timestamp: DateTime<Utc>,
}

/// Configuration types for different components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuPreferences {
    pub require_real_hardware: bool,
    pub allow_simulation: bool,
    pub preferred_backend: Option<GpuBackend>,
    pub minimum_memory_gb: u32,
    pub minimum_compute_capability: (u32, u32),
}

impl GpuPreferences {
    pub fn auto_detect() -> Self {
        Self {
            require_real_hardware: true,
            allow_simulation: false,
            preferred_backend: None,
            minimum_memory_gb: 4,
            minimum_compute_capability: (5, 0),
        }
    }
}

/// Market API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketApiConfig {
    pub require_real_apis: bool,
    pub allow_mock_data: bool,
    pub api_keys: HashMap<String, String>,
    pub rate_limits: HashMap<String, u32>,
    pub timeout_ms: u64,
    pub retry_attempts: u32,
}

impl MarketApiConfig {
    pub fn real_apis_only() -> Self {
        Self {
            require_real_apis: true,
            allow_mock_data: false,
            api_keys: HashMap::new(),
            rate_limits: HashMap::new(),
            timeout_ms: 30000,
            retry_attempts: 3,
        }
    }
}

/// Biological intelligence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalConfig {
    pub use_real_neural_patterns: bool,
    pub authentic_synaptic_plasticity: bool,
    pub brain_regions_to_model: Vec<BrainRegion>,
    pub sampling_rate_hz: f32,
    pub temporal_window_ms: u64,
    pub plasticity_learning_rate: f64,
}

impl BiologicalConfig {
    pub fn authentic_networks() -> Self {
        Self {
            use_real_neural_patterns: true,
            authentic_synaptic_plasticity: true,
            brain_regions_to_model: vec![
                BrainRegion::PrefrontalCortex,
                BrainRegion::Hippocampus,
                BrainRegion::Amygdala,
                BrainRegion::BasalGanglia,
            ],
            sampling_rate_hz: 1000.0,
            temporal_window_ms: 1000,
            plasticity_learning_rate: 0.01,
        }
    }
}

/// Performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub enable_real_monitoring: bool,
    pub sampling_interval_ms: u64,
    pub metrics_retention_hours: u32,
    pub alert_thresholds: HashMap<String, f64>,
}

impl PerformanceConfig {
    pub fn real_monitoring() -> Self {
        Self {
            enable_real_monitoring: true,
            sampling_interval_ms: 1000,
            metrics_retention_hours: 24,
            alert_thresholds: HashMap::new(),
        }
    }
}

/// TENGRI compliance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TengriConfig {
    pub strict_enforcement: bool,
    pub validate_on_startup: bool,
    pub continuous_monitoring: bool,
    pub violation_tolerance: ViolationSeverity,
    pub audit_frequency_hours: u32,
}

impl TengriConfig {
    pub fn strict_enforcement() -> Self {
        Self {
            strict_enforcement: true,
            validate_on_startup: true,
            continuous_monitoring: true,
            violation_tolerance: ViolationSeverity::Low,
            audit_frequency_hours: 1,
        }
    }
}

/// Unique identifier generation
pub fn generate_id() -> String {
    Uuid::new_v4().to_string()
}

/// Timestamp creation
pub fn current_timestamp() -> DateTime<Utc> {
    Utc::now()
}

/// Validation utilities
pub mod validation {
    use super::*;

    /// Validate that market data is real (not mock)
    pub fn validate_real_market_data(data: &MarketData) -> crate::error::Result<()> {
        // Check for signs of mock data
        if data.symbols.is_empty() {
            return Err(crate::error::QbmiaError::InvalidInput {
                field: "symbols".to_string(),
                reason: "Empty symbols list indicates mock data".to_string(),
            });
        }

        // Validate realistic timestamps
        let now = Utc::now();
        for point in &data.data_points {
            if point.timestamp > now {
                return Err(crate::error::QbmiaError::MockDataDetected);
            }
            
            // Check for obviously fake prices
            if point.price.is_zero() || point.price.is_sign_negative() {
                return Err(crate::error::QbmiaError::MockDataDetected);
            }
        }

        // Validate real API sources
        for source in &data.source_apis {
            if source.contains("mock") || source.contains("fake") || source.contains("test") {
                return Err(crate::error::QbmiaError::MockDataDetected);
            }
        }

        Ok(())
    }

    /// Validate GPU device is real hardware
    pub fn validate_real_gpu_device(device: &GpuDevice) -> crate::error::Result<()> {
        // Check for mock device indicators
        if device.name.to_lowercase().contains("mock") 
            || device.name.to_lowercase().contains("fake")
            || device.name.to_lowercase().contains("simulator") {
            return Err(crate::error::QbmiaError::MockDataDetected);
        }

        // Validate realistic memory amounts
        if device.capabilities.total_memory == 0 {
            return Err(crate::error::QbmiaError::MockDataDetected);
        }

        Ok(())
    }

    /// Validate biological patterns are authentic
    pub fn validate_authentic_biological_patterns(patterns: &[BiologicalPattern]) -> crate::error::Result<()> {
        for pattern in patterns {
            // Check for realistic strength values
            if pattern.strength < 0.0 || pattern.strength > 1.0 {
                return Err(crate::error::QbmiaError::InvalidInput {
                    field: "pattern_strength".to_string(),
                    reason: "Pattern strength must be between 0 and 1".to_string(),
                });
            }

            // Validate temporal dynamics are realistic
            if pattern.temporal_dynamics.is_empty() {
                return Err(crate::error::QbmiaError::InvalidInput {
                    field: "temporal_dynamics".to_string(),
                    reason: "Empty temporal dynamics indicate synthetic data".to_string(),
                });
            }
        }

        Ok(())
    }
}