//! # Tengri Market Readiness Sentinel
//! 
//! **Core infrastructure and utilities for SERRA sustainability platform**
//! 
//! This crate provides foundational types, traits, and utilities used throughout
//! the SERRA sustainability assessment platform. It implements the fundamental
//! building blocks that support all sustainability modeling and analysis operations.
//! 
//! ## Core Abstractions
//! 
//! ### Type System
//! ```text
//! Sustainability Types: Strongly typed domain modeling
//! Error Handling: Comprehensive error types with context
//! Configuration: Type-safe configuration management
//! Serialization: Efficient data interchange formats
//! ```
//! 
//! ## Design Principles
//! 
//! - **Type Safety**: Prevent errors at compile time
//! - **Performance**: Zero-cost abstractions where possible
//! - **Extensibility**: Trait-based design for modularity
//! - **Reliability**: Comprehensive error handling
//! 
//! ## Usage Example
//! 
//! ```rust
//! use tengri_market_readiness_sentinel::{Result, SustainabilityError};
//! 
//! fn sustainability_operation() -> Result<f64> {
//!     // Core operations with proper error handling
//!     Ok(42.0)
//! }
//! ```


use std::sync::Arc;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use tokio::sync::RwLock;
use tracing::{info, warn, error};

// Re-export all modules
pub mod config;
pub mod types;
pub mod validation;
pub mod market_validation;
pub mod regime_detection;
pub mod trading_hours;
pub mod volatility_assessment;
pub mod risk_limits;
pub mod regulatory_compliance;
pub mod market_impact;
pub mod error;
pub mod metrics;
pub mod monitoring;

// Core market readiness types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketReadinessReport {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub overall_status: MarketReadinessStatus,
    pub validations: HashMap<String, ValidationResult>,
    pub market_conditions: MarketConditions,
    pub risk_assessment: RiskAssessment,
    pub compliance_status: ComplianceStatus,
    pub recommendations: Vec<String>,
    pub next_validation: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketReadinessStatus {
    Ready,
    Warning,
    NotReady,
    Maintenance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditions {
    pub regime: MarketRegime,
    pub volatility_level: VolatilityLevel,
    pub liquidity_status: LiquidityStatus,
    pub trading_hours_status: TradingHoursStatus,
    pub market_impact_estimate: f64,
    pub current_spread: f64,
    pub volume_profile: VolumeProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketRegime {
    Trending,
    Ranging,
    Volatile,
    Calm,
    Crisis,
    Recovery,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VolatilityLevel {
    Low,
    Normal,
    High,
    Extreme,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LiquidityStatus {
    Abundant,
    Normal,
    Scarce,
    Dry,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradingHoursStatus {
    Open,
    Closed,
    PreMarket,
    PostMarket,
    Holiday,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeProfile {
    pub average_volume: f64,
    pub current_volume: f64,
    pub volume_ratio: f64,
    pub participation_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub var_95: f64,
    pub var_99: f64,
    pub expected_shortfall: f64,
    pub max_drawdown: f64,
    pub position_limits: PositionLimits,
    pub concentration_risk: f64,
    pub correlation_risk: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionLimits {
    pub max_position_size: f64,
    pub max_order_size: f64,
    pub max_daily_volume: f64,
    pub max_exposure: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceStatus {
    pub regulatory_checks: Vec<ComplianceCheck>,
    pub circuit_breakers: Vec<CircuitBreaker>,
    pub position_limits_check: bool,
    pub market_manipulation_check: bool,
    pub best_execution_check: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceCheck {
    pub rule_id: String,
    pub status: bool,
    pub description: String,
    pub last_check: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreaker {
    pub name: String,
    pub enabled: bool,
    pub threshold: f64,
    pub current_value: f64,
    pub time_window: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub status: ValidationStatus,
    pub message: String,
    pub details: Option<serde_json::Value>,
    pub timestamp: DateTime<Utc>,
    pub duration_ms: u64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationStatus {
    Passed,
    Warning,
    Failed,
    InProgress,
}

// Main market readiness sentinel
#[derive(Debug)]
pub struct MarketReadinessSentinel {
    config: Arc<config::MarketReadinessConfig>,
    validation_engine: Arc<validation::ValidationEngine>,
    market_validator: Arc<market_validation::MarketValidator>,
    regime_detector: Arc<regime_detection::RegimeDetector>,
    trading_hours_validator: Arc<trading_hours::TradingHoursValidator>,
    volatility_assessor: Arc<volatility_assessment::VolatilityAssessor>,
    risk_validator: Arc<risk_limits::RiskLimitsValidator>,
    compliance_checker: Arc<regulatory_compliance::ComplianceChecker>,
    market_impact_assessor: Arc<market_impact::MarketImpactAssessor>,
    metrics_collector: Arc<metrics::MetricsCollector>,
    monitoring_system: Arc<monitoring::MonitoringSystem>,
    state: Arc<RwLock<SentinelState>>,
}

#[derive(Debug, Clone)]
struct SentinelState {
    last_validation: Option<DateTime<Utc>>,
    current_status: MarketReadinessStatus,
    validation_count: u64,
    error_count: u64,
    uptime: DateTime<Utc>,
}

impl MarketReadinessSentinel {
    pub async fn new(config: Arc<config::MarketReadinessConfig>) -> Result<Self> {
        let validation_engine = Arc::new(validation::ValidationEngine::new(config.clone()).await?);
        let market_validator = Arc::new(market_validation::MarketValidator::new(config.clone()).await?);
        let regime_detector = Arc::new(regime_detection::RegimeDetector::new(config.clone()).await?);
        let trading_hours_validator = Arc::new(trading_hours::TradingHoursValidator::new(config.clone()).await?);
        let volatility_assessor = Arc::new(volatility_assessment::VolatilityAssessor::new(config.clone()).await?);
        let risk_validator = Arc::new(risk_limits::RiskLimitsValidator::new(config.clone()).await?);
        let compliance_checker = Arc::new(regulatory_compliance::ComplianceChecker::new(config.clone()).await?);
        let market_impact_assessor = Arc::new(market_impact::MarketImpactAssessor::new(config.clone()).await?);
        let metrics_collector = Arc::new(metrics::MetricsCollector::new(config.clone()).await?);
        let monitoring_system = Arc::new(monitoring::MonitoringSystem::new(config.clone()).await?);
        
        let state = Arc::new(RwLock::new(SentinelState {
            last_validation: None,
            current_status: MarketReadinessStatus::NotReady,
            validation_count: 0,
            error_count: 0,
            uptime: Utc::now(),
        }));

        Ok(Self {
            config,
            validation_engine,
            market_validator,
            regime_detector,
            trading_hours_validator,
            volatility_assessor,
            risk_validator,
            compliance_checker,
            market_impact_assessor,
            metrics_collector,
            monitoring_system,
            state,
        })
    }

    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing TENGRI Market Readiness Sentinel...");
        
        // Initialize all components
        self.validation_engine.initialize().await?;
        self.market_validator.initialize().await?;
        self.regime_detector.initialize().await?;
        self.trading_hours_validator.initialize().await?;
        self.volatility_assessor.initialize().await?;
        self.risk_validator.initialize().await?;
        self.compliance_checker.initialize().await?;
        self.market_impact_assessor.initialize().await?;
        self.metrics_collector.initialize().await?;
        self.monitoring_system.initialize().await?;
        
        let mut state = self.state.write().await;
        state.current_status = MarketReadinessStatus::Ready;
        
        info!("TENGRI Market Readiness Sentinel initialized successfully");
        Ok(())
    }

    pub async fn validate_market_readiness(&self) -> Result<MarketReadinessReport> {
        let start_time = std::time::Instant::now();
        let mut validations = HashMap::new();
        
        // Update state
        {
            let mut state = self.state.write().await;
            state.last_validation = Some(Utc::now());
            state.validation_count += 1;
        }
        
        // Run all validations in parallel
        let (market_data_result, regime_result, trading_hours_result, volatility_result, 
             risk_result, compliance_result, market_impact_result) = tokio::join!(
            self.market_validator.validate_market_data(),
            self.regime_detector.detect_regime(),
            self.trading_hours_validator.validate_trading_hours(),
            self.volatility_assessor.assess_volatility(),
            self.risk_validator.validate_risk_limits(),
            self.compliance_checker.check_compliance(),
            self.market_impact_assessor.assess_market_impact()
        );
        
        // Collect validation results
        validations.insert("market_data".to_string(), market_data_result?);
        validations.insert("regime_detection".to_string(), regime_result?);
        validations.insert("trading_hours".to_string(), trading_hours_result?);
        validations.insert("volatility".to_string(), volatility_result?);
        validations.insert("risk_limits".to_string(), risk_result?);
        validations.insert("compliance".to_string(), compliance_result?);
        validations.insert("market_impact".to_string(), market_impact_result?);
        
        // Determine overall status
        let overall_status = self.determine_overall_status(&validations);
        
        // Get market conditions
        let market_conditions = self.get_market_conditions().await?;
        
        // Get risk assessment
        let risk_assessment = self.get_risk_assessment().await?;
        
        // Get compliance status
        let compliance_status = self.get_compliance_status().await?;
        
        // Generate recommendations
        let recommendations = self.generate_recommendations(&validations, &market_conditions).await?;
        
        // Update metrics
        self.metrics_collector.record_validation(start_time.elapsed().as_millis() as u64).await?;
        
        Ok(MarketReadinessReport {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            overall_status,
            validations,
            market_conditions,
            risk_assessment,
            compliance_status,
            recommendations,
            next_validation: Utc::now() + chrono::Duration::minutes(self.config.validation_interval_minutes),
        })
    }
    
    fn determine_overall_status(&self, validations: &HashMap<String, ValidationResult>) -> MarketReadinessStatus {
        let failed_count = validations.values().filter(|v| v.status == ValidationStatus::Failed).count();
        let warning_count = validations.values().filter(|v| v.status == ValidationStatus::Warning).count();
        
        if failed_count > 0 {
            MarketReadinessStatus::NotReady
        } else if warning_count > 0 {
            MarketReadinessStatus::Warning
        } else {
            MarketReadinessStatus::Ready
        }
    }
    
    async fn get_market_conditions(&self) -> Result<MarketConditions> {
        let regime = self.regime_detector.get_current_regime().await?;
        let volatility_level = self.volatility_assessor.get_volatility_level().await?;
        let liquidity_status = self.market_validator.get_liquidity_status().await?;
        let trading_hours_status = self.trading_hours_validator.get_trading_status().await?;
        let market_impact_estimate = self.market_impact_assessor.get_current_impact().await?;
        let current_spread = self.market_validator.get_current_spread().await?;
        let volume_profile = self.market_validator.get_volume_profile().await?;
        
        Ok(MarketConditions {
            regime,
            volatility_level,
            liquidity_status,
            trading_hours_status,
            market_impact_estimate,
            current_spread,
            volume_profile,
        })
    }
    
    async fn get_risk_assessment(&self) -> Result<RiskAssessment> {
        self.risk_validator.get_risk_assessment().await
    }
    
    async fn get_compliance_status(&self) -> Result<ComplianceStatus> {
        self.compliance_checker.get_compliance_status().await
    }
    
    async fn generate_recommendations(&self, validations: &HashMap<String, ValidationResult>, 
                                   market_conditions: &MarketConditions) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();
        
        // Check for failed validations
        for (key, result) in validations {
            if result.status == ValidationStatus::Failed {
                recommendations.push(format!("Fix validation failure in {}: {}", key, result.message));
            }
        }
        
        // Market condition recommendations
        match market_conditions.regime {
            MarketRegime::Crisis => {
                recommendations.push("Consider reducing position sizes due to crisis conditions".to_string());
            },
            MarketRegime::Volatile => {
                recommendations.push("Increase stop-loss monitoring due to high volatility".to_string());
            },
            _ => {}
        }
        
        if market_conditions.volatility_level == VolatilityLevel::Extreme {
            recommendations.push("Consider temporary trading halt due to extreme volatility".to_string());
        }
        
        if market_conditions.liquidity_status == LiquidityStatus::Dry {
            recommendations.push("Reduce order sizes due to low liquidity".to_string());
        }
        
        Ok(recommendations)
    }
    
    pub async fn start_continuous_monitoring(&self) -> Result<()> {
        self.monitoring_system.start_monitoring().await
    }
    
    pub async fn stop_continuous_monitoring(&self) -> Result<()> {
        self.monitoring_system.stop_monitoring().await
    }
    
    pub async fn get_health_status(&self) -> Result<HealthStatus> {
        let state = self.state.read().await;
        Ok(HealthStatus {
            status: state.current_status.clone(),
            uptime: Utc::now().signed_duration_since(state.uptime),
            validation_count: state.validation_count,
            error_count: state.error_count,
            last_validation: state.last_validation,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: MarketReadinessStatus,
    pub uptime: chrono::Duration,
    pub validation_count: u64,
    pub error_count: u64,
    pub last_validation: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsReport {
    pub timestamp: DateTime<Utc>,
    pub validation_latency_ms: u64,
    pub success_rate: f64,
    pub error_rate: f64,
    pub throughput: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub disk_usage: f64,
    pub network_latency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub uptime: chrono::Duration,
    pub load_average: f64,
    pub thread_count: u64,
    pub connection_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingMetrics {
    pub orders_per_second: f64,
    pub fill_rate: f64,
    pub average_latency: f64,
    pub rejection_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    pub current_var: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub position_utilization: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    pub timestamp: DateTime<Utc>,
    pub overall_health: HealthStatus,
    pub performance: PerformanceMetrics,
    pub system: SystemMetrics,
    pub trading: TradingMetrics,
    pub risk: RiskMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub severity: AlertSeverity,
    pub category: String,
    pub message: String,
    pub details: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

// Export the main sentinel
pub use MarketReadinessSentinel as Sentinel;
