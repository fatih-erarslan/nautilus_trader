//! Configuration system for PADS
//!
//! This module provides comprehensive configuration management for all PADS components,
//! allowing for flexible and type-safe configuration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::error::{PadsError, PadsResult};

/// Main PADS configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PadsConfig {
    /// System name
    pub system_name: String,
    
    /// Agent configuration
    pub agent_config: AgentConfig,
    
    /// Board system configuration
    pub board_config: BoardConfig,
    
    /// Panarchy system configuration
    pub panarchy_config: PanarchyConfig,
    
    /// Risk management configuration
    pub risk_config: RiskConfig,
    
    /// Strategy configuration
    pub strategy_config: StrategyConfig,
    
    /// Analyzer configuration
    pub analyzer_config: AnalyzerConfig,
    
    /// Hardware configuration
    pub hardware_config: HardwareConfig,
    
    /// Memory configuration
    pub memory_config: MemoryConfig,
    
    /// Phase parameters for adaptive behavior
    pub phase_parameters: Option<HashMap<String, HashMap<String, serde_json::Value>>>,
    
    /// Performance configuration
    pub performance_config: PerformanceConfig,
    
    /// Logging configuration
    pub logging_config: LoggingConfig,
}

/// Agent system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Maximum number of agents
    pub max_agents: usize,
    
    /// QAR agent configuration
    pub qar_config: QarConfig,
    
    /// QERC agent configuration
    pub qerc_config: QercConfig,
    
    /// IQAD agent configuration
    pub iqad_config: IqadConfig,
    
    /// NQO agent configuration
    pub nqo_config: NqoConfig,
    
    /// QStar agent configuration
    pub qstar_config: QstarConfig,
    
    /// Narrative agent configuration
    pub narrative_config: NarrativeConfig,
    
    /// Agent coordination settings
    pub coordination_settings: CoordinationSettings,
}

/// QAR agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QarConfig {
    /// Number of factors
    pub num_factors: usize,
    
    /// Decision threshold
    pub decision_threshold: f64,
    
    /// Memory length
    pub memory_length: usize,
    
    /// Enable quantum features
    pub enable_quantum: bool,
    
    /// Factor weights
    pub factor_weights: HashMap<String, f64>,
}

/// QERC agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QercConfig {
    /// Reservoir size
    pub reservoir_size: usize,
    
    /// Spectral radius
    pub spectral_radius: f64,
    
    /// Input scaling
    pub input_scaling: f64,
    
    /// Leak rate
    pub leak_rate: f64,
}

/// IQAD agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IqadConfig {
    /// Number of detectors
    pub num_detectors: usize,
    
    /// Anomaly threshold
    pub anomaly_threshold: f64,
    
    /// Memory cells
    pub memory_cells: usize,
}

/// NQO agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NqoConfig {
    /// Number of neurons
    pub num_neurons: usize,
    
    /// Learning rate
    pub learning_rate: f64,
    
    /// Optimization steps
    pub optimization_steps: usize,
}

/// QStar agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QstarConfig {
    /// Use quantum representation
    pub use_quantum_representation: bool,
    
    /// Initial states
    pub initial_states: usize,
    
    /// Training episodes
    pub training_episodes: usize,
}

/// Narrative agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeConfig {
    /// LLM provider
    pub provider: String,
    
    /// API key
    pub api_key: Option<String>,
    
    /// Model name
    pub model: String,
    
    /// Temperature
    pub temperature: f64,
    
    /// Base URL
    pub base_url: Option<String>,
    
    /// Max tokens
    pub max_tokens: usize,
    
    /// Timeout
    pub timeout: f64,
    
    /// Retry attempts
    pub retry_attempts: usize,
    
    /// Cache duration
    pub cache_duration: u64,
}

/// Agent coordination settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationSettings {
    /// Enable parallel execution
    pub parallel_execution: bool,
    
    /// Timeout for agent responses
    pub agent_timeout_ms: u64,
    
    /// Maximum retries
    pub max_retries: usize,
}

/// Board system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoardConfig {
    /// Board size
    pub board_size: usize,
    
    /// LMSR configuration
    pub lmsr_config: LmsrConfig,
    
    /// Voting configuration
    pub voting_config: VotingConfig,
    
    /// Consensus configuration
    pub consensus_config: ConsensusConfig,
}

/// LMSR configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmsrConfig {
    /// Liquidity parameter
    pub liquidity_parameter: f64,
    
    /// Enable parallel processing
    pub enable_parallel: bool,
    
    /// Cache size
    pub cache_size: usize,
}

/// Voting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VotingConfig {
    /// Minimum quorum
    pub min_quorum: f64,
    
    /// Vote timeout
    pub vote_timeout_ms: u64,
    
    /// Enable weighted voting
    pub weighted_voting: bool,
}

/// Consensus configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfig {
    /// Consensus threshold
    pub consensus_threshold: f64,
    
    /// Maximum dissent allowed
    pub max_dissent: f64,
    
    /// Convergence timeout
    pub convergence_timeout_ms: u64,
}

/// Panarchy system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanarchyConfig {
    /// Enable adaptive cycles
    pub enable_adaptive_cycles: bool,
    
    /// Enable cross-scale interactions
    pub enable_cross_scale_interactions: bool,
    
    /// Enable resilience mechanisms
    pub enable_resilience_mechanisms: bool,
    
    /// Phase transition thresholds
    pub phase_transition_thresholds: HashMap<String, f64>,
    
    /// Regime detection settings
    pub regime_detection: RegimeDetectionConfig,
}

/// Regime detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeDetectionConfig {
    /// Window size for regime detection
    pub window_size: usize,
    
    /// Regime change threshold
    pub change_threshold: f64,
    
    /// Minimum regime duration
    pub min_regime_duration: u64,
}

/// Risk management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    /// Enable all risk components
    pub enable_all_components: bool,
    
    /// Via negativa filter settings
    pub via_negativa_config: ViaNegativaConfig,
    
    /// Luck vs skill analyzer settings
    pub luck_skill_config: LuckSkillConfig,
    
    /// Barbell allocator settings
    pub barbell_config: BarbellConfig,
    
    /// Reputation system settings
    pub reputation_config: ReputationConfig,
    
    /// Enhanced anomaly detector settings
    pub enhanced_anomaly_config: EnhancedAnomalyConfig,
    
    /// Antifragile risk manager settings
    pub antifragile_risk_config: AntifragileRiskConfig,
    
    /// Prospect theory manager settings
    pub prospect_theory_config: ProspectTheoryConfig,
}

/// Via negativa filter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViaNegativaConfig {
    /// Filter threshold
    pub filter_threshold: f64,
    
    /// Negative criteria weights
    pub negative_criteria: HashMap<String, f64>,
}

/// Luck vs skill analyzer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LuckSkillConfig {
    /// Analysis window
    pub analysis_window: usize,
    
    /// Skill threshold
    pub skill_threshold: f64,
}

/// Barbell allocator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarbellConfig {
    /// Safe allocation percentage
    pub safe_allocation: f64,
    
    /// Risk allocation percentage
    pub risk_allocation: f64,
}

/// Reputation system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationConfig {
    /// Initial reputation
    pub initial_reputation: f64,
    
    /// Decay rate
    pub decay_rate: f64,
    
    /// Update frequency
    pub update_frequency: u64,
}

/// Enhanced anomaly detector configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedAnomalyConfig {
    /// Anomaly threshold
    pub anomaly_threshold: f64,
    
    /// Detection window
    pub detection_window: usize,
    
    /// Sensitivity
    pub sensitivity: f64,
}

/// Antifragile risk manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntifragileRiskConfig {
    /// Antifragility threshold
    pub antifragility_threshold: f64,
    
    /// Adaptation rate
    pub adaptation_rate: f64,
}

/// Prospect theory manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProspectTheoryConfig {
    /// Loss aversion coefficient
    pub loss_aversion: f64,
    
    /// Reference point
    pub reference_point: f64,
    
    /// Probability weighting parameters
    pub probability_weighting: ProbabilityWeightingConfig,
}

/// Probability weighting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilityWeightingConfig {
    /// Alpha parameter
    pub alpha: f64,
    
    /// Beta parameter
    pub beta: f64,
}

/// Strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// Default strategy
    pub default_strategy: String,
    
    /// Strategy-specific settings
    pub strategy_settings: HashMap<String, StrategySettings>,
    
    /// Strategy selection criteria
    pub selection_criteria: StrategySelectionConfig,
}

/// Strategy settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategySettings {
    /// Confidence threshold
    pub confidence_threshold: f64,
    
    /// Risk tolerance
    pub risk_tolerance: f64,
    
    /// Custom parameters
    pub custom_parameters: HashMap<String, serde_json::Value>,
}

/// Strategy selection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategySelectionConfig {
    /// Selection method
    pub selection_method: String,
    
    /// Evaluation criteria
    pub evaluation_criteria: Vec<String>,
    
    /// Selection timeout
    pub selection_timeout_ms: u64,
}

/// Analyzer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzerConfig {
    /// Enable all analyzers
    pub enable_all_analyzers: bool,
    
    /// Whale detector settings
    pub whale_detector_config: WhaleDetectorConfig,
    
    /// Black swan detector settings
    pub black_swan_config: BlackSwanConfig,
    
    /// Antifragility analyzer settings
    pub antifragility_config: AntifragilityConfig,
    
    /// Fibonacci analyzer settings
    pub fibonacci_config: FibonacciConfig,
    
    /// SOC analyzer settings
    pub soc_config: SocConfig,
    
    /// Panarchy analyzer settings
    pub panarchy_analyzer_config: PanarchyAnalyzerConfig,
}

/// Whale detector configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleDetectorConfig {
    /// Volume threshold
    pub volume_threshold: f64,
    
    /// Detection window
    pub detection_window: usize,
    
    /// Confidence threshold
    pub confidence_threshold: f64,
}

/// Black swan detector configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackSwanConfig {
    /// Risk threshold
    pub risk_threshold: f64,
    
    /// Historical window
    pub historical_window: usize,
    
    /// Severity levels
    pub severity_levels: Vec<f64>,
}

/// Antifragility analyzer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntifragilityConfig {
    /// Use JIT compilation
    pub use_jit: bool,
    
    /// Cache size
    pub cache_size: usize,
    
    /// Analysis depth
    pub analysis_depth: usize,
}

/// Fibonacci analyzer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FibonacciConfig {
    /// Cache size
    pub cache_size: usize,
    
    /// Use JIT compilation
    pub use_jit: bool,
    
    /// Fibonacci levels
    pub fibonacci_levels: Vec<f64>,
}

/// SOC analyzer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocConfig {
    /// Criticality threshold
    pub criticality_threshold: f64,
    
    /// Analysis window
    pub analysis_window: usize,
}

/// Panarchy analyzer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanarchyAnalyzerConfig {
    /// Cycle detection sensitivity
    pub cycle_sensitivity: f64,
    
    /// Phase prediction horizon
    pub prediction_horizon: usize,
}

/// Hardware configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    
    /// Enable SIMD
    pub enable_simd: bool,
    
    /// Enable memory mapping
    pub enable_memory_mapping: bool,
    
    /// CPU thread count
    pub cpu_threads: Option<usize>,
    
    /// GPU device ID
    pub gpu_device_id: Option<usize>,
    
    /// Memory pool size
    pub memory_pool_size: usize,
}

/// Memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Decision history size
    pub history_size: usize,
    
    /// Cache size
    pub cache_size: usize,
    
    /// Enable persistent storage
    pub enable_persistent_storage: bool,
    
    /// Storage path
    pub storage_path: Option<String>,
    
    /// Memory optimization level
    pub optimization_level: MemoryOptimizationLevel,
}

/// Memory optimization levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryOptimizationLevel {
    /// Low optimization, more memory usage
    Low,
    
    /// Medium optimization
    Medium,
    
    /// High optimization, less memory usage
    High,
    
    /// Maximum optimization
    Maximum,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Target decision latency in nanoseconds
    pub target_latency_ns: u64,
    
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    
    /// Monitoring interval
    pub monitoring_interval_ms: u64,
    
    /// Performance alerts
    pub performance_alerts: PerformanceAlerts,
}

/// Performance alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlerts {
    /// Latency threshold for alerts
    pub latency_threshold_ns: u64,
    
    /// Memory usage threshold
    pub memory_threshold_mb: usize,
    
    /// CPU usage threshold
    pub cpu_threshold_percent: f64,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    pub level: String,
    
    /// Enable structured logging
    pub structured: bool,
    
    /// Log file path
    pub file_path: Option<String>,
    
    /// Enable console logging
    pub console: bool,
    
    /// Log rotation settings
    pub rotation: Option<LogRotationConfig>,
}

/// Log rotation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogRotationConfig {
    /// Max file size in MB
    pub max_size_mb: usize,
    
    /// Number of files to keep
    pub keep_files: usize,
    
    /// Rotation schedule
    pub schedule: String,
}

impl Default for PadsConfig {
    fn default() -> Self {
        Self {
            system_name: "PADS Unified".to_string(),
            agent_config: AgentConfig::default(),
            board_config: BoardConfig::default(),
            panarchy_config: PanarchyConfig::default(),
            risk_config: RiskConfig::default(),
            strategy_config: StrategyConfig::default(),
            analyzer_config: AnalyzerConfig::default(),
            hardware_config: HardwareConfig::default(),
            memory_config: MemoryConfig::default(),
            phase_parameters: None,
            performance_config: PerformanceConfig::default(),
            logging_config: LoggingConfig::default(),
        }
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_agents: 12,
            qar_config: QarConfig::default(),
            qerc_config: QercConfig::default(),
            iqad_config: IqadConfig::default(),
            nqo_config: NqoConfig::default(),
            qstar_config: QstarConfig::default(),
            narrative_config: NarrativeConfig::default(),
            coordination_settings: CoordinationSettings::default(),
        }
    }
}

impl Default for QarConfig {
    fn default() -> Self {
        Self {
            num_factors: 8,
            decision_threshold: 0.6,
            memory_length: 50,
            enable_quantum: true,
            factor_weights: HashMap::new(),
        }
    }
}

impl Default for QercConfig {
    fn default() -> Self {
        Self {
            reservoir_size: 100,
            spectral_radius: 0.95,
            input_scaling: 1.0,
            leak_rate: 0.3,
        }
    }
}

impl Default for IqadConfig {
    fn default() -> Self {
        Self {
            num_detectors: 10,
            anomaly_threshold: 0.8,
            memory_cells: 50,
        }
    }
}

impl Default for NqoConfig {
    fn default() -> Self {
        Self {
            num_neurons: 100,
            learning_rate: 0.01,
            optimization_steps: 1000,
        }
    }
}

impl Default for QstarConfig {
    fn default() -> Self {
        Self {
            use_quantum_representation: true,
            initial_states: 200,
            training_episodes: 100,
        }
    }
}

impl Default for NarrativeConfig {
    fn default() -> Self {
        Self {
            provider: "lmstudio".to_string(),
            api_key: None,
            model: "mistral".to_string(),
            temperature: 1.0,
            base_url: Some("http://localhost:1234/v1/chat/completions".to_string()),
            max_tokens: 4096,
            timeout: 60.0,
            retry_attempts: 3,
            cache_duration: 60,
        }
    }
}

impl Default for CoordinationSettings {
    fn default() -> Self {
        Self {
            parallel_execution: true,
            agent_timeout_ms: 5000,
            max_retries: 3,
        }
    }
}

impl Default for BoardConfig {
    fn default() -> Self {
        Self {
            board_size: 14,
            lmsr_config: LmsrConfig::default(),
            voting_config: VotingConfig::default(),
            consensus_config: ConsensusConfig::default(),
        }
    }
}

impl Default for LmsrConfig {
    fn default() -> Self {
        Self {
            liquidity_parameter: 50.0,
            enable_parallel: true,
            cache_size: 256,
        }
    }
}

impl Default for VotingConfig {
    fn default() -> Self {
        Self {
            min_quorum: 0.5,
            vote_timeout_ms: 1000,
            weighted_voting: true,
        }
    }
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            consensus_threshold: 0.7,
            max_dissent: 0.3,
            convergence_timeout_ms: 2000,
        }
    }
}

impl Default for PanarchyConfig {
    fn default() -> Self {
        Self {
            enable_adaptive_cycles: true,
            enable_cross_scale_interactions: true,
            enable_resilience_mechanisms: true,
            phase_transition_thresholds: [
                ("growth_to_conservation".to_string(), 0.8),
                ("conservation_to_release".to_string(), 0.7),
                ("release_to_reorganization".to_string(), 0.6),
                ("reorganization_to_growth".to_string(), 0.7),
            ].into_iter().collect(),
            regime_detection: RegimeDetectionConfig::default(),
        }
    }
}

impl Default for RegimeDetectionConfig {
    fn default() -> Self {
        Self {
            window_size: 50,
            change_threshold: 0.3,
            min_regime_duration: 10,
        }
    }
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            enable_all_components: true,
            via_negativa_config: ViaNegativaConfig::default(),
            luck_skill_config: LuckSkillConfig::default(),
            barbell_config: BarbellConfig::default(),
            reputation_config: ReputationConfig::default(),
            enhanced_anomaly_config: EnhancedAnomalyConfig::default(),
            antifragile_risk_config: AntifragileRiskConfig::default(),
            prospect_theory_config: ProspectTheoryConfig::default(),
        }
    }
}

impl Default for ViaNegativaConfig {
    fn default() -> Self {
        Self {
            filter_threshold: 0.5,
            negative_criteria: HashMap::new(),
        }
    }
}

impl Default for LuckSkillConfig {
    fn default() -> Self {
        Self {
            analysis_window: 100,
            skill_threshold: 0.6,
        }
    }
}

impl Default for BarbellConfig {
    fn default() -> Self {
        Self {
            safe_allocation: 0.8,
            risk_allocation: 0.2,
        }
    }
}

impl Default for ReputationConfig {
    fn default() -> Self {
        Self {
            initial_reputation: 0.5,
            decay_rate: 0.01,
            update_frequency: 10,
        }
    }
}

impl Default for EnhancedAnomalyConfig {
    fn default() -> Self {
        Self {
            anomaly_threshold: 0.8,
            detection_window: 50,
            sensitivity: 0.7,
        }
    }
}

impl Default for AntifragileRiskConfig {
    fn default() -> Self {
        Self {
            antifragility_threshold: 0.7,
            adaptation_rate: 0.1,
        }
    }
}

impl Default for ProspectTheoryConfig {
    fn default() -> Self {
        Self {
            loss_aversion: 2.25,
            reference_point: 0.0,
            probability_weighting: ProbabilityWeightingConfig::default(),
        }
    }
}

impl Default for ProbabilityWeightingConfig {
    fn default() -> Self {
        Self {
            alpha: 0.88,
            beta: 0.88,
        }
    }
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            default_strategy: "consensus".to_string(),
            strategy_settings: HashMap::new(),
            selection_criteria: StrategySelectionConfig::default(),
        }
    }
}

impl Default for StrategySelectionConfig {
    fn default() -> Self {
        Self {
            selection_method: "adaptive".to_string(),
            evaluation_criteria: vec![
                "volatility".to_string(),
                "trend_strength".to_string(),
                "black_swan_risk".to_string(),
            ],
            selection_timeout_ms: 100,
        }
    }
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            enable_all_analyzers: true,
            whale_detector_config: WhaleDetectorConfig::default(),
            black_swan_config: BlackSwanConfig::default(),
            antifragility_config: AntifragilityConfig::default(),
            fibonacci_config: FibonacciConfig::default(),
            soc_config: SocConfig::default(),
            panarchy_analyzer_config: PanarchyAnalyzerConfig::default(),
        }
    }
}

impl Default for WhaleDetectorConfig {
    fn default() -> Self {
        Self {
            volume_threshold: 1000000.0,
            detection_window: 20,
            confidence_threshold: 0.8,
        }
    }
}

impl Default for BlackSwanConfig {
    fn default() -> Self {
        Self {
            risk_threshold: 0.8,
            historical_window: 1000,
            severity_levels: vec![0.6, 0.8, 0.9],
        }
    }
}

impl Default for AntifragilityConfig {
    fn default() -> Self {
        Self {
            use_jit: true,
            cache_size: 100,
            analysis_depth: 10,
        }
    }
}

impl Default for FibonacciConfig {
    fn default() -> Self {
        Self {
            cache_size: 100,
            use_jit: true,
            fibonacci_levels: vec![0.236, 0.382, 0.5, 0.618, 0.786],
        }
    }
}

impl Default for SocConfig {
    fn default() -> Self {
        Self {
            criticality_threshold: 0.8,
            analysis_window: 100,
        }
    }
}

impl Default for PanarchyAnalyzerConfig {
    fn default() -> Self {
        Self {
            cycle_sensitivity: 0.7,
            prediction_horizon: 50,
        }
    }
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            enable_gpu: true,
            enable_simd: true,
            enable_memory_mapping: true,
            cpu_threads: None, // Auto-detect
            gpu_device_id: None, // Auto-select
            memory_pool_size: 1024 * 1024 * 100, // 100MB
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            history_size: 1000,
            cache_size: 10000,
            enable_persistent_storage: false,
            storage_path: None,
            optimization_level: MemoryOptimizationLevel::Medium,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            target_latency_ns: 10_000, // 10 microseconds
            enable_monitoring: true,
            monitoring_interval_ms: 1000,
            performance_alerts: PerformanceAlerts::default(),
        }
    }
}

impl Default for PerformanceAlerts {
    fn default() -> Self {
        Self {
            latency_threshold_ns: 50_000, // 50 microseconds
            memory_threshold_mb: 1000,
            cpu_threshold_percent: 80.0,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            structured: true,
            file_path: None,
            console: true,
            rotation: None,
        }
    }
}

impl PadsConfig {
    /// Load configuration from file
    pub fn from_file(path: &str) -> PadsResult<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| PadsError::configuration(format!("Failed to read config file {}: {}", path, e)))?;
        
        if path.ends_with(".toml") {
            toml::from_str(&content)
                .map_err(|e| PadsError::configuration(format!("Failed to parse TOML config: {}", e)))
        } else if path.ends_with(".json") {
            serde_json::from_str(&content)
                .map_err(|e| PadsError::configuration(format!("Failed to parse JSON config: {}", e)))
        } else if path.ends_with(".yaml") || path.ends_with(".yml") {
            serde_yaml::from_str(&content)
                .map_err(|e| PadsError::configuration(format!("Failed to parse YAML config: {}", e)))
        } else {
            Err(PadsError::configuration("Unsupported config file format"))
        }
    }
    
    /// Save configuration to file
    pub fn to_file(&self, path: &str) -> PadsResult<()> {
        let content = if path.ends_with(".toml") {
            toml::to_string_pretty(self)
                .map_err(|e| PadsError::configuration(format!("Failed to serialize to TOML: {}", e)))?
        } else if path.ends_with(".json") {
            serde_json::to_string_pretty(self)
                .map_err(|e| PadsError::configuration(format!("Failed to serialize to JSON: {}", e)))?
        } else if path.ends_with(".yaml") || path.ends_with(".yml") {
            serde_yaml::to_string(self)
                .map_err(|e| PadsError::configuration(format!("Failed to serialize to YAML: {}", e)))?
        } else {
            return Err(PadsError::configuration("Unsupported config file format"));
        };
        
        std::fs::write(path, content)
            .map_err(|e| PadsError::configuration(format!("Failed to write config file {}: {}", path, e)))?;
        
        Ok(())
    }
    
    /// Validate configuration
    pub fn validate(&self) -> PadsResult<()> {
        // Validate agent config
        if self.agent_config.max_agents == 0 {
            return Err(PadsError::configuration("max_agents must be greater than 0"));
        }
        
        if self.agent_config.qar_config.decision_threshold < 0.0 || self.agent_config.qar_config.decision_threshold > 1.0 {
            return Err(PadsError::configuration("QAR decision_threshold must be between 0.0 and 1.0"));
        }
        
        // Validate board config
        if self.board_config.board_size == 0 {
            return Err(PadsError::configuration("board_size must be greater than 0"));
        }
        
        if self.board_config.lmsr_config.liquidity_parameter <= 0.0 {
            return Err(PadsError::configuration("LMSR liquidity_parameter must be positive"));
        }
        
        // Validate performance config
        if self.performance_config.target_latency_ns == 0 {
            return Err(PadsError::configuration("target_latency_ns must be greater than 0"));
        }
        
        // Validate memory config
        if self.memory_config.history_size == 0 {
            return Err(PadsError::configuration("history_size must be greater than 0"));
        }
        
        Ok(())
    }
    
    /// Create a test configuration
    #[cfg(test)]
    pub fn test_config() -> Self {
        let mut config = Self::default();
        config.system_name = "Test PADS".to_string();
        config.performance_config.enable_monitoring = false;
        config.logging_config.level = "debug".to_string();
        config
    }
    
    /// Create a minimal configuration for testing
    pub fn minimal() -> Self {
        let mut config = Self::default();
        config.agent_config.max_agents = 3;
        config.board_config.board_size = 5;
        config.memory_config.history_size = 100;
        config.memory_config.cache_size = 1000;
        config.performance_config.enable_monitoring = false;
        config
    }
    
    /// Create a high-performance configuration
    pub fn high_performance() -> Self {
        let mut config = Self::default();
        config.hardware_config.enable_gpu = true;
        config.hardware_config.enable_simd = true;
        config.hardware_config.enable_memory_mapping = true;
        config.hardware_config.cpu_threads = Some(num_cpus::get());
        config.performance_config.target_latency_ns = 5_000; // 5 microseconds
        config.memory_config.optimization_level = MemoryOptimizationLevel::Maximum;
        config.agent_config.coordination_settings.parallel_execution = true;
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = PadsConfig::default();
        assert_eq!(config.system_name, "PADS Unified");
        assert_eq!(config.agent_config.max_agents, 12);
        assert_eq!(config.board_config.board_size, 14);
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = PadsConfig::default();
        assert!(config.validate().is_ok());
        
        config.agent_config.max_agents = 0;
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_config_serialization() {
        let config = PadsConfig::default();
        
        // Test JSON serialization
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: PadsConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.system_name, deserialized.system_name);
        
        // Test TOML serialization
        let toml = toml::to_string(&config).unwrap();
        let deserialized: PadsConfig = toml::from_str(&toml).unwrap();
        assert_eq!(config.system_name, deserialized.system_name);
    }
    
    #[test]
    fn test_minimal_config() {
        let config = PadsConfig::minimal();
        assert_eq!(config.agent_config.max_agents, 3);
        assert_eq!(config.board_config.board_size, 5);
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_high_performance_config() {
        let config = PadsConfig::high_performance();
        assert!(config.hardware_config.enable_gpu);
        assert!(config.hardware_config.enable_simd);
        assert_eq!(config.performance_config.target_latency_ns, 5_000);
        assert!(config.validate().is_ok());
    }
}