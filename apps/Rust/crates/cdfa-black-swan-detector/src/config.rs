//! Configuration parameters for Black Swan detection

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for Black Swan detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackSwanConfig {
    /// Size of the rolling window for analysis
    pub window_size: usize,
    
    /// Tail threshold for extreme value analysis (0.0 to 1.0)
    pub tail_threshold: f64,
    
    /// Minimum number of points required for tail analysis
    pub min_tail_points: usize,
    
    /// Statistical significance level for hypothesis testing
    pub significance_level: f64,
    
    /// Hill estimator k parameter (number of order statistics)
    pub hill_estimator_k: usize,
    
    /// Z-score threshold for extreme event detection
    pub extreme_z_threshold: f64,
    
    /// Alpha parameter for volatility clustering GARCH model
    pub volatility_clustering_alpha: f64,
    
    /// Threshold for liquidity crisis detection
    pub liquidity_crisis_threshold: f64,
    
    /// Threshold for correlation breakdown detection
    pub correlation_breakdown_threshold: f64,
    
    /// Memory pool size for efficient allocations
    pub memory_pool_size: usize,
    
    /// Enable GPU acceleration
    pub use_gpu: bool,
    
    /// Enable SIMD optimizations
    pub use_simd: bool,
    
    /// Enable parallel processing
    pub parallel_processing: bool,
    
    /// Size of the computation cache
    pub cache_size: usize,
    
    /// Performance tuning parameters
    pub performance: PerformanceConfig,
    
    /// Risk model parameters
    pub risk_model: RiskModelConfig,
    
    /// Alert configuration
    pub alerts: AlertConfig,
}

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Target latency in nanoseconds
    pub target_latency_ns: u64,
    
    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,
    
    /// Number of worker threads
    pub num_threads: usize,
    
    /// Batch size for SIMD operations
    pub simd_batch_size: usize,
    
    /// Cache line size optimization
    pub cache_line_size: usize,
    
    /// Prefetch distance for memory access
    pub prefetch_distance: usize,
    
    /// Enable zero-copy operations
    pub zero_copy: bool,
    
    /// Memory allocation strategy
    pub allocation_strategy: AllocationStrategy,
}

/// Memory allocation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// Use system allocator
    System,
    /// Use memory pool
    Pool,
    /// Use custom allocator (jemalloc/mimalloc)
    Custom,
}

/// Risk model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskModelConfig {
    /// Component weights for probability calculation
    pub component_weights: ComponentWeights,
    
    /// Extreme Value Theory parameters
    pub evt_params: EVTConfig,
    
    /// Volatility model parameters
    pub volatility_params: VolatilityConfig,
    
    /// Liquidity model parameters
    pub liquidity_params: LiquidityConfig,
    
    /// Correlation model parameters
    pub correlation_params: CorrelationConfig,
}

/// Component weights for Black Swan probability calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentWeights {
    /// Weight for fat tail component
    pub fat_tail: f64,
    
    /// Weight for volatility clustering component
    pub volatility_clustering: f64,
    
    /// Weight for liquidity crisis component
    pub liquidity_crisis: f64,
    
    /// Weight for correlation breakdown component
    pub correlation_breakdown: f64,
    
    /// Weight for jump discontinuity component
    pub jump_discontinuity: f64,
    
    /// Weight for microstructure anomaly component
    pub microstructure_anomaly: f64,
}

/// Extreme Value Theory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EVTConfig {
    /// Hill estimator parameters
    pub hill_k_min: usize,
    pub hill_k_max: usize,
    pub hill_k_step: usize,
    
    /// Threshold selection method
    pub threshold_method: ThresholdMethod,
    
    /// Confidence intervals
    pub confidence_level: f64,
    
    /// Bootstrap parameters for uncertainty quantification
    pub bootstrap_samples: usize,
    
    /// Goodness-of-fit tests
    pub goodness_of_fit_tests: bool,
}

/// Threshold selection method for EVT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThresholdMethod {
    /// Fixed quantile threshold
    Quantile(f64),
    /// Mean excess function
    MeanExcess,
    /// Automated threshold selection
    Automated,
}

/// Volatility model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityConfig {
    /// GARCH model parameters
    pub garch_p: usize,
    pub garch_q: usize,
    
    /// Volatility clustering detection parameters
    pub clustering_window: usize,
    pub clustering_threshold: f64,
    
    /// Regime switching parameters
    pub regime_states: usize,
    pub regime_transition_prob: f64,
}

/// Liquidity model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityConfig {
    /// Volume-based liquidity measures
    pub volume_window: usize,
    pub volume_threshold: f64,
    
    /// Bid-ask spread parameters
    pub spread_window: usize,
    pub spread_threshold: f64,
    
    /// Market depth parameters
    pub depth_levels: usize,
    pub depth_threshold: f64,
}

/// Correlation model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationConfig {
    /// Correlation calculation window
    pub correlation_window: usize,
    
    /// Correlation breakdown threshold
    pub breakdown_threshold: f64,
    
    /// Dynamic correlation parameters
    pub dynamic_window: usize,
    pub dynamic_alpha: f64,
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Probability threshold for alerts
    pub probability_threshold: f64,
    
    /// Minimum time between alerts
    pub min_alert_interval: Duration,
    
    /// Alert severity levels
    pub severity_levels: Vec<f64>,
    
    /// Enable email alerts
    pub email_alerts: bool,
    
    /// Enable webhook alerts
    pub webhook_alerts: bool,
    
    /// Alert message templates
    pub message_templates: AlertTemplates,
}

/// Alert message templates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertTemplates {
    /// Low severity alert template
    pub low_severity: String,
    
    /// Medium severity alert template
    pub medium_severity: String,
    
    /// High severity alert template
    pub high_severity: String,
    
    /// Critical severity alert template
    pub critical_severity: String,
}

impl Default for BlackSwanConfig {
    fn default() -> Self {
        Self {
            window_size: 1000,
            tail_threshold: 0.95,
            min_tail_points: 50,
            significance_level: 0.01,
            hill_estimator_k: 100,
            extreme_z_threshold: 3.0,
            volatility_clustering_alpha: 0.1,
            liquidity_crisis_threshold: 0.3,
            correlation_breakdown_threshold: 0.5,
            memory_pool_size: 1024 * 1024,
            use_gpu: true,
            use_simd: true,
            parallel_processing: true,
            cache_size: 10000,
            performance: PerformanceConfig::default(),
            risk_model: RiskModelConfig::default(),
            alerts: AlertConfig::default(),
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            target_latency_ns: 500, // Sub-microsecond target
            max_memory_bytes: 64 * 1024 * 1024, // 64MB
            num_threads: num_cpus::get(),
            simd_batch_size: 8,
            cache_line_size: 64,
            prefetch_distance: 64,
            zero_copy: true,
            allocation_strategy: AllocationStrategy::Pool,
        }
    }
}

impl Default for RiskModelConfig {
    fn default() -> Self {
        Self {
            component_weights: ComponentWeights::default(),
            evt_params: EVTConfig::default(),
            volatility_params: VolatilityConfig::default(),
            liquidity_params: LiquidityConfig::default(),
            correlation_params: CorrelationConfig::default(),
        }
    }
}

impl Default for ComponentWeights {
    fn default() -> Self {
        Self {
            fat_tail: 0.3,
            volatility_clustering: 0.2,
            liquidity_crisis: 0.15,
            correlation_breakdown: 0.15,
            jump_discontinuity: 0.1,
            microstructure_anomaly: 0.1,
        }
    }
}

impl Default for EVTConfig {
    fn default() -> Self {
        Self {
            hill_k_min: 20,
            hill_k_max: 200,
            hill_k_step: 10,
            threshold_method: ThresholdMethod::Quantile(0.95),
            confidence_level: 0.05,
            bootstrap_samples: 1000,
            goodness_of_fit_tests: true,
        }
    }
}

impl Default for VolatilityConfig {
    fn default() -> Self {
        Self {
            garch_p: 1,
            garch_q: 1,
            clustering_window: 50,
            clustering_threshold: 0.5,
            regime_states: 2,
            regime_transition_prob: 0.05,
        }
    }
}

impl Default for LiquidityConfig {
    fn default() -> Self {
        Self {
            volume_window: 20,
            volume_threshold: 0.5,
            spread_window: 10,
            spread_threshold: 0.01,
            depth_levels: 5,
            depth_threshold: 0.1,
        }
    }
}

impl Default for CorrelationConfig {
    fn default() -> Self {
        Self {
            correlation_window: 30,
            breakdown_threshold: 0.5,
            dynamic_window: 100,
            dynamic_alpha: 0.1,
        }
    }
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            probability_threshold: 0.7,
            min_alert_interval: Duration::from_secs(60),
            severity_levels: vec![0.5, 0.7, 0.85, 0.95],
            email_alerts: false,
            webhook_alerts: false,
            message_templates: AlertTemplates::default(),
        }
    }
}

impl Default for AlertTemplates {
    fn default() -> Self {
        Self {
            low_severity: "Low Black Swan probability detected: {probability:.2}".to_string(),
            medium_severity: "Medium Black Swan probability detected: {probability:.2}".to_string(),
            high_severity: "High Black Swan probability detected: {probability:.2}".to_string(),
            critical_severity: "CRITICAL: Black Swan event imminent: {probability:.2}".to_string(),
        }
    }
}

impl ComponentWeights {
    /// Validate that weights sum to 1.0
    pub fn validate(&self) -> Result<(), String> {
        let sum = self.fat_tail + self.volatility_clustering + self.liquidity_crisis 
                + self.correlation_breakdown + self.jump_discontinuity + self.microstructure_anomaly;
        
        if (sum - 1.0).abs() > 1e-6 {
            return Err(format!("Component weights sum to {:.6}, expected 1.0", sum));
        }
        
        Ok(())
    }
    
    /// Normalize weights to sum to 1.0
    pub fn normalize(&mut self) {
        let sum = self.fat_tail + self.volatility_clustering + self.liquidity_crisis 
                + self.correlation_breakdown + self.jump_discontinuity + self.microstructure_anomaly;
        
        if sum > 1e-6 {
            self.fat_tail /= sum;
            self.volatility_clustering /= sum;
            self.liquidity_crisis /= sum;
            self.correlation_breakdown /= sum;
            self.jump_discontinuity /= sum;
            self.microstructure_anomaly /= sum;
        }
    }
}

impl BlackSwanConfig {
    /// Create a high-performance configuration for production use
    pub fn high_performance() -> Self {
        Self {
            window_size: 2000,
            tail_threshold: 0.99,
            min_tail_points: 100,
            significance_level: 0.001,
            hill_estimator_k: 200,
            extreme_z_threshold: 4.0,
            volatility_clustering_alpha: 0.05,
            liquidity_crisis_threshold: 0.2,
            correlation_breakdown_threshold: 0.3,
            memory_pool_size: 2 * 1024 * 1024,
            use_gpu: true,
            use_simd: true,
            parallel_processing: true,
            cache_size: 20000,
            performance: PerformanceConfig {
                target_latency_ns: 100, // 100ns target
                max_memory_bytes: 128 * 1024 * 1024, // 128MB
                num_threads: num_cpus::get() * 2,
                simd_batch_size: 16,
                cache_line_size: 64,
                prefetch_distance: 128,
                zero_copy: true,
                allocation_strategy: AllocationStrategy::Custom,
            },
            risk_model: RiskModelConfig::default(),
            alerts: AlertConfig::default(),
        }
    }
    
    /// Create a low-latency configuration for real-time trading
    pub fn low_latency() -> Self {
        Self {
            window_size: 500,
            tail_threshold: 0.95,
            min_tail_points: 25,
            significance_level: 0.05,
            hill_estimator_k: 50,
            extreme_z_threshold: 2.5,
            volatility_clustering_alpha: 0.2,
            liquidity_crisis_threshold: 0.4,
            correlation_breakdown_threshold: 0.6,
            memory_pool_size: 512 * 1024,
            use_gpu: false, // CPU-only for consistent latency
            use_simd: true,
            parallel_processing: false, // Single-threaded for latency
            cache_size: 5000,
            performance: PerformanceConfig {
                target_latency_ns: 50, // 50ns target
                max_memory_bytes: 32 * 1024 * 1024, // 32MB
                num_threads: 1,
                simd_batch_size: 4,
                cache_line_size: 64,
                prefetch_distance: 32,
                zero_copy: true,
                allocation_strategy: AllocationStrategy::Pool,
            },
            risk_model: RiskModelConfig::default(),
            alerts: AlertConfig::default(),
        }
    }
    
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.window_size < 10 {
            return Err("Window size must be at least 10".to_string());
        }
        
        if self.tail_threshold <= 0.0 || self.tail_threshold >= 1.0 {
            return Err("Tail threshold must be between 0 and 1".to_string());
        }
        
        if self.min_tail_points < 5 {
            return Err("Minimum tail points must be at least 5".to_string());
        }
        
        if self.significance_level <= 0.0 || self.significance_level >= 1.0 {
            return Err("Significance level must be between 0 and 1".to_string());
        }
        
        if self.hill_estimator_k < 5 {
            return Err("Hill estimator k must be at least 5".to_string());
        }
        
        if self.extreme_z_threshold <= 0.0 {
            return Err("Extreme z-score threshold must be positive".to_string());
        }
        
        self.risk_model.component_weights.validate()?;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_default_config() {
        let config = BlackSwanConfig::default();
        assert_eq!(config.window_size, 1000);
        assert_relative_eq!(config.tail_threshold, 0.95);
        assert!(config.use_gpu);
        assert!(config.use_simd);
        assert!(config.parallel_processing);
    }

    #[test]
    fn test_component_weights_validation() {
        let mut weights = ComponentWeights::default();
        assert!(weights.validate().is_ok());
        
        weights.fat_tail = 0.5;
        assert!(weights.validate().is_err());
        
        weights.normalize();
        assert!(weights.validate().is_ok());
    }

    #[test]
    fn test_high_performance_config() {
        let config = BlackSwanConfig::high_performance();
        assert_eq!(config.window_size, 2000);
        assert_relative_eq!(config.tail_threshold, 0.99);
        assert_eq!(config.performance.target_latency_ns, 100);
    }

    #[test]
    fn test_low_latency_config() {
        let config = BlackSwanConfig::low_latency();
        assert_eq!(config.window_size, 500);
        assert_eq!(config.performance.target_latency_ns, 50);
        assert_eq!(config.performance.num_threads, 1);
        assert!(!config.use_gpu);
        assert!(!config.parallel_processing);
    }

    #[test]
    fn test_config_validation() {
        let mut config = BlackSwanConfig::default();
        assert!(config.validate().is_ok());
        
        config.window_size = 5;
        assert!(config.validate().is_err());
        
        config.window_size = 1000;
        config.tail_threshold = 1.5;
        assert!(config.validate().is_err());
        
        config.tail_threshold = 0.95;
        config.significance_level = 0.0;
        assert!(config.validate().is_err());
    }
}