//! Configuration structures for the risk management system

use std::time::Duration;
use serde::{Deserialize, Serialize};
use crate::quantum_uncertainty::QuantumConfig;

/// Main risk management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    /// VaR calculation configuration
    pub var_config: VarConfig,
    /// Stress testing configuration
    pub stress_config: StressConfig,
    /// Position sizing configuration
    pub position_config: PositionConfig,
    /// Portfolio optimization configuration
    pub portfolio_config: PortfolioConfig,
    /// Real-time monitoring configuration
    pub monitoring_config: MonitoringConfig,
    /// Compliance configuration
    pub compliance_config: ComplianceConfig,
    /// Risk metrics configuration
    pub metrics_config: MetricsConfig,
    /// Correlation analysis configuration
    pub correlation_config: CorrelationConfig,
    /// GPU configuration
    pub gpu_config: GpuConfig,
    /// Quantum uncertainty configuration
    pub quantum_config: QuantumConfig,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            var_config: VarConfig::default(),
            stress_config: StressConfig::default(),
            position_config: PositionConfig::default(),
            portfolio_config: PortfolioConfig::default(),
            monitoring_config: MonitoringConfig::default(),
            compliance_config: ComplianceConfig::default(),
            metrics_config: MetricsConfig::default(),
            correlation_config: CorrelationConfig::default(),
            gpu_config: GpuConfig::default(),
            quantum_config: QuantumConfig::default(),
        }
    }
}

/// VaR calculation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarConfig {
    /// Confidence levels to calculate
    pub confidence_levels: Vec<f64>,
    /// Historical window size
    pub historical_window: usize,
    /// Monte Carlo simulation count
    pub monte_carlo_simulations: usize,
    /// VaR calculation method
    pub method: VarMethod,
    /// Enable quantum uncertainty
    pub enable_quantum: bool,
    /// Smoothing parameter
    pub smoothing_factor: f64,
}

impl Default for VarConfig {
    fn default() -> Self {
        Self {
            confidence_levels: vec![0.01, 0.05, 0.1],
            historical_window: 250,
            monte_carlo_simulations: 100_000,
            method: VarMethod::QuantumMonteCarlo,
            enable_quantum: true,
            smoothing_factor: 0.94,
        }
    }
}

/// VaR calculation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VarMethod {
    Historical,
    Parametric,
    MonteCarlo,
    QuantumMonteCarlo,
    Copula,
}

/// Stress testing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressConfig {
    /// Pre-defined stress scenarios
    pub scenarios: Vec<StressScenarioConfig>,
    /// Monte Carlo stress simulations
    pub monte_carlo_simulations: usize,
    /// Maximum stress loss threshold
    pub max_stress_loss: f64,
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    /// Tail risk scenarios
    pub tail_risk_scenarios: usize,
}

impl Default for StressConfig {
    fn default() -> Self {
        Self {
            scenarios: vec![
                StressScenarioConfig::market_crash(),
                StressScenarioConfig::volatility_spike(),
                StressScenarioConfig::liquidity_crisis(),
                StressScenarioConfig::correlation_breakdown(),
            ],
            monte_carlo_simulations: 1_000_000,
            max_stress_loss: 0.20,
            enable_gpu: true,
            tail_risk_scenarios: 10_000,
        }
    }
}

/// Stress scenario configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressScenarioConfig {
    pub name: String,
    pub description: String,
    pub market_shock: f64,
    pub volatility_multiplier: f64,
    pub correlation_shift: f64,
    pub liquidity_impact: f64,
    pub probability: f64,
}

impl StressScenarioConfig {
    pub fn market_crash() -> Self {
        Self {
            name: "Market Crash".to_string(),
            description: "Severe market downturn (-20% equity, +50% volatility)".to_string(),
            market_shock: -0.20,
            volatility_multiplier: 1.5,
            correlation_shift: 0.3,
            liquidity_impact: 0.4,
            probability: 0.01,
        }
    }
    
    pub fn volatility_spike() -> Self {
        Self {
            name: "Volatility Spike".to_string(),
            description: "Sudden volatility increase (+100% volatility)".to_string(),
            market_shock: 0.0,
            volatility_multiplier: 2.0,
            correlation_shift: 0.2,
            liquidity_impact: 0.2,
            probability: 0.05,
        }
    }
    
    pub fn liquidity_crisis() -> Self {
        Self {
            name: "Liquidity Crisis".to_string(),
            description: "Severe liquidity shortage".to_string(),
            market_shock: -0.10,
            volatility_multiplier: 1.3,
            correlation_shift: 0.4,
            liquidity_impact: 0.8,
            probability: 0.02,
        }
    }
    
    pub fn correlation_breakdown() -> Self {
        Self {
            name: "Correlation Breakdown".to_string(),
            description: "Breakdown of historical correlations".to_string(),
            market_shock: 0.0,
            volatility_multiplier: 1.2,
            correlation_shift: 0.8,
            liquidity_impact: 0.1,
            probability: 0.03,
        }
    }
}

/// Position sizing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionConfig {
    /// Maximum position size as fraction of portfolio
    pub max_position_size: f64,
    /// Maximum leverage
    pub max_leverage: f64,
    /// Kelly criterion configuration
    pub kelly_config: KellyConfig,
    /// Risk budget per position
    pub risk_budget_per_position: f64,
    /// Enable quantum optimization
    pub enable_quantum_optimization: bool,
}

impl Default for PositionConfig {
    fn default() -> Self {
        Self {
            max_position_size: 0.1,
            max_leverage: 3.0,
            kelly_config: KellyConfig::default(),
            risk_budget_per_position: 0.02,
            enable_quantum_optimization: true,
        }
    }
}

/// Kelly criterion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KellyConfig {
    /// Kelly fraction multiplier (safety factor)
    pub kelly_multiplier: f64,
    /// Minimum Kelly fraction
    pub min_kelly_fraction: f64,
    /// Maximum Kelly fraction
    pub max_kelly_fraction: f64,
    /// Lookback period for Kelly calculation
    pub lookback_period: usize,
}

impl Default for KellyConfig {
    fn default() -> Self {
        Self {
            kelly_multiplier: 0.25,
            min_kelly_fraction: 0.01,
            max_kelly_fraction: 0.20,
            lookback_period: 252,
        }
    }
}

/// Portfolio optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioConfig {
    /// Optimization objective
    pub objective: OptimizationObjective,
    /// Risk tolerance
    pub risk_tolerance: f64,
    /// Minimum and maximum weights
    pub min_weight: f64,
    pub max_weight: f64,
    /// Turnover constraints
    pub max_turnover: f64,
    /// Rebalancing frequency
    pub rebalancing_frequency: Duration,
    /// Enable quantum optimization
    pub enable_quantum_optimization: bool,
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        Self {
            objective: OptimizationObjective::MaxSharpe,
            risk_tolerance: 0.15,
            min_weight: 0.0,
            max_weight: 0.20,
            max_turnover: 0.50,
            rebalancing_frequency: Duration::from_secs(24 * 3600), // Daily
            enable_quantum_optimization: true,
        }
    }
}

/// Portfolio optimization objectives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationObjective {
    MaxReturn,
    MinRisk,
    MaxSharpe,
    MinCvar,
    MaxUtility,
    RiskParity,
    QuantumOptimal,
}

/// Real-time monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Update frequency
    pub update_frequency: Duration,
    /// Maximum latency for risk calculations
    pub max_latency: Duration,
    /// Risk limits
    pub risk_limits: RiskLimits,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
    /// Enable real-time VaR monitoring
    pub enable_realtime_var: bool,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            update_frequency: Duration::from_micros(100),
            max_latency: Duration::from_micros(10),
            risk_limits: RiskLimits::default(),
            alert_thresholds: AlertThresholds::default(),
            enable_realtime_var: true,
        }
    }
}

/// Risk limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    /// Maximum portfolio VaR
    pub max_portfolio_var: f64,
    /// Maximum position VaR
    pub max_position_var: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Maximum leverage
    pub max_leverage: f64,
    /// Maximum concentration
    pub max_concentration: f64,
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            max_portfolio_var: 0.05,
            max_position_var: 0.02,
            max_drawdown: 0.10,
            max_leverage: 3.0,
            max_concentration: 0.20,
        }
    }
}

/// Alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// VaR threshold for alerts
    pub var_alert_threshold: f64,
    /// Drawdown threshold for alerts
    pub drawdown_alert_threshold: f64,
    /// Leverage threshold for alerts
    pub leverage_alert_threshold: f64,
    /// Correlation threshold for alerts
    pub correlation_alert_threshold: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            var_alert_threshold: 0.03,
            drawdown_alert_threshold: 0.05,
            leverage_alert_threshold: 2.5,
            correlation_alert_threshold: 0.8,
        }
    }
}

/// Compliance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceConfig {
    /// Regulatory frameworks to comply with
    pub regulatory_frameworks: Vec<RegulatoryFramework>,
    /// Reporting requirements
    pub reporting_requirements: ReportingRequirements,
    /// Audit trail configuration
    pub audit_trail: AuditTrailConfig,
}

impl Default for ComplianceConfig {
    fn default() -> Self {
        Self {
            regulatory_frameworks: vec![
                RegulatoryFramework::Basel3,
                RegulatoryFramework::MiFID2,
                RegulatoryFramework::Dodd_Frank,
            ],
            reporting_requirements: ReportingRequirements::default(),
            audit_trail: AuditTrailConfig::default(),
        }
    }
}

/// Regulatory frameworks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegulatoryFramework {
    Basel3,
    MiFID2,
    Dodd_Frank,
    CFTC,
    SEC,
    ESMA,
}

/// Reporting requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingRequirements {
    pub daily_var_report: bool,
    pub stress_test_report: bool,
    pub position_limit_report: bool,
    pub large_exposure_report: bool,
    pub model_validation_report: bool,
}

impl Default for ReportingRequirements {
    fn default() -> Self {
        Self {
            daily_var_report: true,
            stress_test_report: true,
            position_limit_report: true,
            large_exposure_report: true,
            model_validation_report: true,
        }
    }
}

/// Audit trail configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditTrailConfig {
    pub enable_audit_trail: bool,
    pub retention_period: Duration,
    pub detailed_logging: bool,
}

impl Default for AuditTrailConfig {
    fn default() -> Self {
        Self {
            enable_audit_trail: true,
            retention_period: Duration::from_secs(7 * 365 * 24 * 3600), // 7 years
            detailed_logging: true,
        }
    }
}

/// Risk metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Metrics to calculate
    pub metrics: Vec<RiskMetric>,
    /// Risk-free rate for Sharpe ratio
    pub risk_free_rate: f64,
    /// Benchmark for relative metrics
    pub benchmark: Option<String>,
    /// Calculation frequency
    pub calculation_frequency: Duration,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            metrics: vec![
                RiskMetric::Sharpe,
                RiskMetric::Sortino,
                RiskMetric::Calmar,
                RiskMetric::MaxDrawdown,
                RiskMetric::VAR,
                RiskMetric::CVaR,
                RiskMetric::Beta,
                RiskMetric::Alpha,
                RiskMetric::InformationRatio,
                RiskMetric::TrackingError,
            ],
            risk_free_rate: 0.02,
            benchmark: Some("SPY".to_string()),
            calculation_frequency: Duration::from_secs(3600), // Hourly
        }
    }
}

/// Risk metrics to calculate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskMetric {
    Sharpe,
    Sortino,
    Calmar,
    MaxDrawdown,
    VAR,
    CVaR,
    Beta,
    Alpha,
    InformationRatio,
    TrackingError,
    TreynorRatio,
    JensenAlpha,
    Volatility,
    Skewness,
    Kurtosis,
}

/// Correlation analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationConfig {
    /// Correlation calculation method
    pub method: CorrelationMethod,
    /// Rolling window size
    pub window_size: usize,
    /// Enable quantum correlation modeling
    pub enable_quantum_correlation: bool,
    /// Copula models to use
    pub copula_models: Vec<CopulaModel>,
}

impl Default for CorrelationConfig {
    fn default() -> Self {
        Self {
            method: CorrelationMethod::QuantumEnhanced,
            window_size: 60,
            enable_quantum_correlation: true,
            copula_models: vec![
                CopulaModel::Gaussian,
                CopulaModel::Student,
                CopulaModel::Clayton,
                CopulaModel::Gumbel,
            ],
        }
    }
}

/// Correlation calculation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorrelationMethod {
    Pearson,
    Spearman,
    Kendall,
    QuantumEnhanced,
    DCC_GARCH,
    RollingCorrelation,
}

/// Copula models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CopulaModel {
    Gaussian,
    Student,
    Clayton,
    Gumbel,
    Frank,
    Joe,
    Archimedean,
}

/// GPU configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    /// GPU device index
    pub device_index: usize,
    /// GPU memory limit
    pub memory_limit: usize,
    /// Parallel batch size
    pub batch_size: usize,
    /// Workgroup size
    pub workgroup_size: usize,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            enable_gpu: true,
            device_index: 0,
            memory_limit: 2_000_000_000, // 2GB
            batch_size: 10_000,
            workgroup_size: 256,
        }
    }
}

impl RiskConfig {
    /// Load configuration from file
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let config_str = std::fs::read_to_string(path)?;
        if path.ends_with(".toml") {
            Ok(toml::from_str(&config_str)?)
        } else {
            Ok(serde_json::from_str(&config_str)?)
        }
    }
    
    /// Save configuration to file
    pub fn to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let config_str = if path.ends_with(".toml") {
            toml::to_string_pretty(self)?
        } else {
            serde_json::to_string_pretty(self)?
        };
        std::fs::write(path, config_str)?;
        Ok(())
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        // VaR validation
        if self.var_config.confidence_levels.is_empty() {
            return Err("At least one confidence level must be specified".to_string());
        }
        
        for &level in &self.var_config.confidence_levels {
            if level <= 0.0 || level >= 1.0 {
                return Err(format!("Confidence level {} must be between 0 and 1", level));
            }
        }
        
        // Position sizing validation
        if self.position_config.max_position_size <= 0.0 || self.position_config.max_position_size > 1.0 {
            return Err("Maximum position size must be between 0 and 1".to_string());
        }
        
        if self.position_config.max_leverage <= 0.0 {
            return Err("Maximum leverage must be positive".to_string());
        }
        
        // Portfolio validation
        if self.portfolio_config.min_weight < 0.0 || self.portfolio_config.max_weight > 1.0 {
            return Err("Portfolio weights must be between 0 and 1".to_string());
        }
        
        if self.portfolio_config.min_weight > self.portfolio_config.max_weight {
            return Err("Minimum weight cannot be greater than maximum weight".to_string());
        }
        
        // Risk limits validation
        if self.monitoring_config.risk_limits.max_drawdown <= 0.0 || self.monitoring_config.risk_limits.max_drawdown > 1.0 {
            return Err("Maximum drawdown must be between 0 and 1".to_string());
        }
        
        Ok(())
    }
}