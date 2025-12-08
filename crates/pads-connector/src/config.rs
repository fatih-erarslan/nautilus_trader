//! PADS Configuration Module

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// PADS system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PadsConfig {
    /// Scale management configuration
    pub scale_config: ScaleConfig,
    
    /// Decision routing configuration
    pub routing_config: RoutingConfig,
    
    /// Communication configuration
    pub comm_config: CommunicationConfig,
    
    /// Resilience configuration
    pub resilience_config: ResilienceConfig,
    
    /// Monitoring configuration
    pub monitor_config: MonitorConfig,
    
    /// Performance tuning
    pub performance: PerformanceConfig,
}

/// Scale management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleConfig {
    /// Number of scale levels
    pub num_levels: usize,
    
    /// Micro scale parameters
    pub micro_scale: MicroScaleConfig,
    
    /// Meso scale parameters
    pub meso_scale: MesoScaleConfig,
    
    /// Macro scale parameters
    pub macro_scale: MacroScaleConfig,
    
    /// Scale transition thresholds
    pub transition_thresholds: TransitionThresholds,
    
    /// Adaptive cycle parameters
    pub adaptive_cycle: AdaptiveCycleConfig,
}

/// Micro scale configuration (exploitation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroScaleConfig {
    /// Time horizon (milliseconds)
    pub time_horizon_ms: u64,
    
    /// Decision frequency
    pub decision_frequency_hz: f64,
    
    /// Optimization focus (0-1, higher = more exploitation)
    pub exploitation_weight: f64,
    
    /// Local search radius
    pub search_radius: f64,
    
    /// Memory window size
    pub memory_window: usize,
}

/// Meso scale configuration (transition)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MesoScaleConfig {
    /// Time horizon (seconds)
    pub time_horizon_secs: u64,
    
    /// Balance between exploration and exploitation
    pub balance_factor: f64,
    
    /// Coordination window
    pub coordination_window: usize,
    
    /// Adaptation rate
    pub adaptation_rate: f64,
}

/// Macro scale configuration (exploration)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroScaleConfig {
    /// Time horizon (minutes)
    pub time_horizon_mins: u64,
    
    /// Strategic planning depth
    pub planning_depth: usize,
    
    /// Exploration weight (0-1, higher = more exploration)
    pub exploration_weight: f64,
    
    /// Innovation threshold
    pub innovation_threshold: f64,
}

/// Scale transition thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionThresholds {
    /// Micro to meso threshold
    pub micro_to_meso: f64,
    
    /// Meso to macro threshold
    pub meso_to_macro: f64,
    
    /// Macro to meso threshold
    pub macro_to_meso: f64,
    
    /// Meso to micro threshold
    pub meso_to_micro: f64,
    
    /// Hysteresis factor
    pub hysteresis: f64,
}

/// Adaptive cycle configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveCycleConfig {
    /// Growth phase duration
    pub growth_duration: Duration,
    
    /// Conservation phase duration
    pub conservation_duration: Duration,
    
    /// Release phase duration
    pub release_duration: Duration,
    
    /// Reorganization phase duration
    pub reorganization_duration: Duration,
    
    /// Phase transition smoothness
    pub transition_smoothness: f64,
}

/// Decision routing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConfig {
    /// Maximum queue size per scale
    pub max_queue_size: usize,
    
    /// Routing timeout
    pub routing_timeout: Duration,
    
    /// Priority weights
    pub priority_weights: PriorityWeights,
    
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    
    /// Batch processing size
    pub batch_size: usize,
}

/// Priority weights for decision routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityWeights {
    /// Urgency weight
    pub urgency: f64,
    
    /// Impact weight
    pub impact: f64,
    
    /// Confidence weight
    pub confidence: f64,
    
    /// Resource availability weight
    pub resource: f64,
}

/// Load balancing strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastLoaded,
    WeightedRandom,
    Adaptive,
}

/// Communication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationConfig {
    /// Channel buffer size
    pub channel_buffer_size: usize,
    
    /// Message timeout
    pub message_timeout: Duration,
    
    /// Retry policy
    pub retry_policy: RetryPolicy,
    
    /// Compression enabled
    pub enable_compression: bool,
    
    /// Encryption enabled
    pub enable_encryption: bool,
}

/// Retry policy for communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum retries
    pub max_retries: usize,
    
    /// Base delay between retries
    pub base_delay: Duration,
    
    /// Exponential backoff factor
    pub backoff_factor: f64,
    
    /// Maximum delay
    pub max_delay: Duration,
}

/// Resilience configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResilienceConfig {
    /// Circuit breaker configuration
    pub circuit_breaker: CircuitBreakerConfig,
    
    /// Fault tolerance settings
    pub fault_tolerance: FaultToleranceConfig,
    
    /// Recovery strategies
    pub recovery_strategies: RecoveryStrategies,
    
    /// Health check interval
    pub health_check_interval: Duration,
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Failure threshold
    pub failure_threshold: usize,
    
    /// Success threshold
    pub success_threshold: usize,
    
    /// Timeout duration
    pub timeout: Duration,
    
    /// Half-open test interval
    pub half_open_interval: Duration,
}

/// Fault tolerance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceConfig {
    /// Enable redundancy
    pub enable_redundancy: bool,
    
    /// Redundancy factor
    pub redundancy_factor: usize,
    
    /// Failover timeout
    pub failover_timeout: Duration,
    
    /// State persistence
    pub persist_state: bool,
}

/// Recovery strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryStrategies {
    /// Enable automatic recovery
    pub auto_recovery: bool,
    
    /// Recovery timeout
    pub recovery_timeout: Duration,
    
    /// Gradual recovery
    pub gradual_recovery: bool,
    
    /// Recovery rate
    pub recovery_rate: f64,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfig {
    /// Metrics collection interval
    pub metrics_interval: Duration,
    
    /// Enable detailed logging
    pub detailed_logging: bool,
    
    /// Metrics retention period
    pub retention_period: Duration,
    
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
}

/// Alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Latency threshold (ms)
    pub latency_ms: u64,
    
    /// Error rate threshold (%)
    pub error_rate_percent: f64,
    
    /// Memory usage threshold (%)
    pub memory_percent: f64,
    
    /// CPU usage threshold (%)
    pub cpu_percent: f64,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Thread pool size
    pub thread_pool_size: usize,
    
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    
    /// Cache size (MB)
    pub cache_size_mb: usize,
    
    /// Batch processing threshold
    pub batch_threshold: usize,
    
    /// Memory pool size (MB)
    pub memory_pool_mb: usize,
}

impl Default for PadsConfig {
    fn default() -> Self {
        Self {
            scale_config: ScaleConfig::default(),
            routing_config: RoutingConfig::default(),
            comm_config: CommunicationConfig::default(),
            resilience_config: ResilienceConfig::default(),
            monitor_config: MonitorConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

impl Default for ScaleConfig {
    fn default() -> Self {
        Self {
            num_levels: 3,
            micro_scale: MicroScaleConfig::default(),
            meso_scale: MesoScaleConfig::default(),
            macro_scale: MacroScaleConfig::default(),
            transition_thresholds: TransitionThresholds::default(),
            adaptive_cycle: AdaptiveCycleConfig::default(),
        }
    }
}

impl Default for MicroScaleConfig {
    fn default() -> Self {
        Self {
            time_horizon_ms: 100,
            decision_frequency_hz: 1000.0,
            exploitation_weight: 0.8,
            search_radius: 0.1,
            memory_window: 1000,
        }
    }
}

impl Default for MesoScaleConfig {
    fn default() -> Self {
        Self {
            time_horizon_secs: 60,
            balance_factor: 0.5,
            coordination_window: 100,
            adaptation_rate: 0.1,
        }
    }
}

impl Default for MacroScaleConfig {
    fn default() -> Self {
        Self {
            time_horizon_mins: 60,
            planning_depth: 10,
            exploration_weight: 0.8,
            innovation_threshold: 0.7,
        }
    }
}

impl Default for TransitionThresholds {
    fn default() -> Self {
        Self {
            micro_to_meso: 0.7,
            meso_to_macro: 0.8,
            macro_to_meso: 0.3,
            meso_to_micro: 0.2,
            hysteresis: 0.1,
        }
    }
}

impl Default for AdaptiveCycleConfig {
    fn default() -> Self {
        Self {
            growth_duration: Duration::from_secs(300),
            conservation_duration: Duration::from_secs(600),
            release_duration: Duration::from_secs(60),
            reorganization_duration: Duration::from_secs(120),
            transition_smoothness: 0.8,
        }
    }
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 10000,
            routing_timeout: Duration::from_millis(100),
            priority_weights: PriorityWeights::default(),
            load_balancing: LoadBalancingStrategy::Adaptive,
            batch_size: 100,
        }
    }
}

impl Default for PriorityWeights {
    fn default() -> Self {
        Self {
            urgency: 0.4,
            impact: 0.3,
            confidence: 0.2,
            resource: 0.1,
        }
    }
}

impl Default for CommunicationConfig {
    fn default() -> Self {
        Self {
            channel_buffer_size: 1000,
            message_timeout: Duration::from_millis(50),
            retry_policy: RetryPolicy::default(),
            enable_compression: true,
            enable_encryption: false,
        }
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(10),
            backoff_factor: 2.0,
            max_delay: Duration::from_secs(1),
        }
    }
}

impl Default for ResilienceConfig {
    fn default() -> Self {
        Self {
            circuit_breaker: CircuitBreakerConfig::default(),
            fault_tolerance: FaultToleranceConfig::default(),
            recovery_strategies: RecoveryStrategies::default(),
            health_check_interval: Duration::from_secs(10),
        }
    }
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            timeout: Duration::from_secs(30),
            half_open_interval: Duration::from_secs(10),
        }
    }
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            enable_redundancy: true,
            redundancy_factor: 2,
            failover_timeout: Duration::from_secs(5),
            persist_state: true,
        }
    }
}

impl Default for RecoveryStrategies {
    fn default() -> Self {
        Self {
            auto_recovery: true,
            recovery_timeout: Duration::from_secs(60),
            gradual_recovery: true,
            recovery_rate: 0.1,
        }
    }
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            metrics_interval: Duration::from_secs(1),
            detailed_logging: false,
            retention_period: Duration::from_secs(3600),
            alert_thresholds: AlertThresholds::default(),
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            latency_ms: 100,
            error_rate_percent: 5.0,
            memory_percent: 80.0,
            cpu_percent: 90.0,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            thread_pool_size: num_cpus::get(),
            enable_simd: true,
            cache_size_mb: 256,
            batch_threshold: 50,
            memory_pool_mb: 512,
        }
    }
}