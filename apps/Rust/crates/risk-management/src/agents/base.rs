//! # Base Agent Traits and Types
//!
//! Core traits and data structures for RUV-swarm risk management agents.
//! This module defines the fundamental interfaces and types that all
//! specialized risk agents implement.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::types::*;

/// Base trait for all swarm agents
#[async_trait]
pub trait SwarmAgent: Send + Sync {
    /// Start the agent
    async fn start(&mut self) -> Result<()>;
    
    /// Stop the agent
    async fn stop(&mut self) -> Result<()>;
    
    /// Get agent ID
    async fn get_agent_id(&self) -> Uuid;
    
    /// Get agent type
    async fn get_agent_type(&self) -> AgentType;
    
    /// Get current status
    async fn get_status(&self) -> AgentStatus;
    
    /// Get performance metrics
    async fn get_performance_metrics(&self) -> Result<AgentPerformanceMetrics>;
    
    /// Handle coordination message
    async fn handle_coordination_message(&self, message: CoordinationMessage) -> Result<CoordinationResponse>;
}

/// Agent types in the risk management swarm
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentType {
    RiskManagement,
    PortfolioOptimization,
    StressTesting,
    CorrelationAnalysis,
    LiquidityRisk,
}

impl std::fmt::Display for AgentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentType::RiskManagement => write!(f, "RiskManagement"),
            AgentType::PortfolioOptimization => write!(f, "PortfolioOptimization"),
            AgentType::StressTesting => write!(f, "StressTesting"),
            AgentType::CorrelationAnalysis => write!(f, "CorrelationAnalysis"),
            AgentType::LiquidityRisk => write!(f, "LiquidityRisk"),
        }
    }
}

/// Agent status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentStatus {
    Initializing,
    Starting,
    Running,
    Stopping,
    Stopped,
    Error(String),
}

/// Health levels for agent monitoring
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthLevel {
    Healthy,
    Warning,
    Critical,
    Offline,
}

/// Agent health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentHealthStatus {
    pub agent_id: Uuid,
    pub agent_type: AgentType,
    pub health_level: HealthLevel,
    pub last_calculation_time: Duration,
    pub average_calculation_time: Duration,
    pub total_calculations: u64,
    pub error_count: u64,
    pub uptime: Duration,
}

/// Agent performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPerformanceMetrics {
    pub agent_id: Uuid,
    pub agent_type: AgentType,
    pub total_calculations: u64,
    pub successful_calculations: u64,
    pub error_count: u64,
    pub average_calculation_time: Duration,
    pub min_calculation_time: Duration,
    pub max_calculation_time: Duration,
    pub last_calculation_time: Duration,
    pub start_time: Instant,
    pub calculation_history: Vec<CalculationRecord>,
}

impl AgentPerformanceMetrics {
    pub fn new(agent_id: Uuid, agent_type: AgentType) -> Self {
        Self {
            agent_id,
            agent_type,
            total_calculations: 0,
            successful_calculations: 0,
            error_count: 0,
            average_calculation_time: Duration::from_nanos(0),
            min_calculation_time: Duration::from_secs(u64::MAX),
            max_calculation_time: Duration::from_nanos(0),
            last_calculation_time: Duration::from_nanos(0),
            start_time: Instant::now(),
            calculation_history: Vec::new(),
        }
    }
    
    pub fn start_calculation(&mut self) {
        self.total_calculations += 1;
    }
    
    pub fn end_calculation(&mut self, duration: Duration) {
        self.successful_calculations += 1;
        self.last_calculation_time = duration;
        
        // Update min/max
        if duration < self.min_calculation_time {
            self.min_calculation_time = duration;
        }
        if duration > self.max_calculation_time {
            self.max_calculation_time = duration;
        }
        
        // Update running average
        let total_time = self.average_calculation_time.as_nanos() as f64 * (self.successful_calculations - 1) as f64;
        let new_average = (total_time + duration.as_nanos() as f64) / self.successful_calculations as f64;
        self.average_calculation_time = Duration::from_nanos(new_average as u64);
        
        // Add to history (keep last 1000 records)
        let record = CalculationRecord {
            timestamp: Instant::now(),
            duration,
            success: true,
        };
        self.calculation_history.push(record);
        if self.calculation_history.len() > 1000 {
            self.calculation_history.remove(0);
        }
    }
    
    pub fn record_error(&mut self) {
        self.error_count += 1;
        
        let record = CalculationRecord {
            timestamp: Instant::now(),
            duration: Duration::from_nanos(0),
            success: false,
        };
        self.calculation_history.push(record);
        if self.calculation_history.len() > 1000 {
            self.calculation_history.remove(0);
        }
    }
    
    pub fn record_calculation(&mut self, operation: &str, duration: Duration) {
        self.end_calculation(duration);
    }
    
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }
    
    pub fn error_rate(&self) -> f64 {
        if self.total_calculations == 0 {
            0.0
        } else {
            self.error_count as f64 / self.total_calculations as f64
        }
    }
    
    pub fn success_rate(&self) -> f64 {
        if self.total_calculations == 0 {
            0.0
        } else {
            self.successful_calculations as f64 / self.total_calculations as f64
        }
    }
}

/// Individual calculation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalculationRecord {
    pub timestamp: Instant,
    pub duration: Duration,
    pub success: bool,
}

/// Message types for swarm communication
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageType {
    // Risk Management Messages
    VarCalculationRequest,
    VarCalculationResponse,
    RiskMonitoringRequest,
    RiskMonitoringResponse,
    RiskAlert,
    RiskLimitBreach,
    
    // Portfolio Optimization Messages
    PortfolioOptimizationRequest,
    PortfolioOptimizationResponse,
    RebalancingRequest,
    RebalancingResponse,
    RiskAllocationRequest,
    RiskAllocationResponse,
    
    // Stress Testing Messages
    StressTestRequest,
    StressTestResponse,
    TailRiskAnalysisRequest,
    TailRiskAnalysisResponse,
    SensitivityAnalysisRequest,
    SensitivityAnalysisResponse,
    ReverseStressTestRequest,
    ReverseStressTestResponse,
    
    // Correlation Analysis Messages
    CorrelationAnalysisRequest,
    CorrelationAnalysisResponse,
    RegimeDetectionRequest,
    RegimeDetectionResponse,
    CopulaAnalysisRequest,
    CopulaAnalysisResponse,
    DynamicCorrelationRequest,
    DynamicCorrelationResponse,
    CorrelationChangeAlert,
    
    // Liquidity Risk Messages
    LiquidityAssessmentRequest,
    LiquidityAssessmentResponse,
    MarketImpactAnalysisRequest,
    MarketImpactAnalysisResponse,
    ExecutionOptimizationRequest,
    ExecutionOptimizationResponse,
    FundingLiquidityAssessmentRequest,
    FundingLiquidityAssessmentResponse,
    LiquidityStressAlert,
    
    // System Messages
    HealthCheck,
    HealthCheckResponse,
    AgentRegistration,
    AgentDeregistration,
    SystemStatus,
    PerformanceReport,
    
    // Portfolio Updates
    PortfolioUpdate,
    RiskLimitUpdate,
    MarketDataUpdate,
}

/// Message priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MessagePriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Swarm message for inter-agent communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmMessage {
    pub id: Uuid,
    pub sender_id: Uuid,
    pub sender_type: AgentType,
    pub recipient_id: Option<Uuid>, // None for broadcast
    pub message_type: MessageType,
    pub content: MessageContent,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub priority: MessagePriority,
    pub requires_response: bool,
}

/// Message content enum for different types of data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageContent {
    // Risk Management
    VarCalculationRequest { portfolio: Portfolio, confidence_level: f64 },
    VarCalculationResponse(QuantumVarResult),
    RiskMonitoringRequest { portfolio: Portfolio },
    RiskMonitoringResponse(RealTimeRiskMetrics),
    RiskLimitBreach(RiskLimitBreach),
    
    // Portfolio Optimization
    PortfolioOptimizationRequest { 
        assets: Vec<Asset>, 
        constraints: PortfolioConstraints,
        objectives: Vec<OptimizationObjective>,
    },
    PortfolioOptimizationResponse(QuantumOptimizedPortfolio),
    RebalancingRequest { 
        portfolio: Portfolio, 
        market_update: MarketUpdate,
        constraints: RebalancingConstraints,
    },
    RebalancingResponse(RebalancedPortfolio),
    RiskAllocationRequest { 
        assets: Vec<Asset>, 
        market_regime: MarketRegime,
        risk_tolerance: f64,
    },
    RiskAllocationResponse(DynamicRiskAllocation),
    
    // Stress Testing
    StressTestRequest { 
        portfolio: Portfolio, 
        scenarios: Vec<StressScenario>,
        config: SimulationConfig,
    },
    StressTestResponse(QuantumStressTestResults),
    TailRiskAnalysisRequest { portfolio: Portfolio, confidence_levels: Vec<f64> },
    TailRiskAnalysisResponse(TailRiskAnalysis),
    SensitivityAnalysisRequest { 
        portfolio: Portfolio, 
        risk_factors: Vec<RiskFactor>,
        config: SensitivityConfig,
    },
    SensitivityAnalysisResponse(SensitivityAnalysisResults),
    ReverseStressTestRequest { portfolio: Portfolio, target_loss: f64 },
    ReverseStressTestResponse(ReverseStressTestResults),
    
    // Correlation Analysis
    CorrelationAnalysisRequest { 
        assets: Vec<Asset>, 
        market_data: MarketData,
        config: CorrelationAnalysisConfig,
    },
    CorrelationAnalysisResponse(QuantumCorrelationAnalysis),
    RegimeDetectionRequest { 
        assets: Vec<Asset>, 
        historical_data: HistoricalMarketData,
        config: RegimeDetectionConfig,
    },
    RegimeDetectionResponse(RegimeChangeAnalysis),
    CopulaAnalysisRequest { 
        assets: Vec<Asset>, 
        market_data: MarketData,
        config: CopulaConfig,
    },
    CopulaAnalysisResponse(CopulaDependencyAnalysis),
    DynamicCorrelationRequest { 
        assets: Vec<Asset>, 
        market_data: MarketData,
        config: DccConfig,
    },
    DynamicCorrelationResponse(DynamicCorrelationResults),
    CorrelationChangeAlert(RealTimeCorrelationUpdate),
    
    // Liquidity Risk
    LiquidityAssessmentRequest { 
        portfolio: Portfolio, 
        market_data: MarketData,
        config: LiquidityAssessmentConfig,
    },
    LiquidityAssessmentResponse(QuantumLiquidityRiskAssessment),
    MarketImpactAnalysisRequest { 
        trade_orders: Vec<TradeOrder>, 
        market_data: MarketData,
        config: MarketImpactConfig,
    },
    MarketImpactAnalysisResponse(MarketImpactAnalysis),
    ExecutionOptimizationRequest { 
        large_order: LargeOrder, 
        market_conditions: MarketConditions,
        constraints: ExecutionConstraints,
    },
    ExecutionOptimizationResponse(OptimalExecutionStrategy),
    FundingLiquidityAssessmentRequest { 
        portfolio: Portfolio, 
        funding_requirements: FundingRequirements,
        time_horizon: Duration,
    },
    FundingLiquidityAssessmentResponse(FundingLiquidityRiskAssessment),
    LiquidityStressAlert(RealTimeLiquidityUpdate),
    
    // System
    HealthCheckRequest,
    HealthCheckResponse(AgentHealthStatus),
    PerformanceReportRequest,
    PerformanceReportResponse(AgentPerformanceMetrics),
    
    // Portfolio Updates
    PortfolioUpdate { positions: Vec<Position> },
    RiskLimitUpdate { limits: RiskLimits },
    MarketDataUpdate(MarketData),
    
    // Generic
    Text(String),
    Error(String),
}

/// Configuration structures for different agent types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAgentConfig {
    pub quantum_var_config: QuantumVarConfig,
    pub monitoring_config: MonitoringConfig,
    pub limit_config: LimitConfig,
    pub tengri_config: TengriIntegrationConfig,
}

impl Default for RiskAgentConfig {
    fn default() -> Self {
        Self {
            quantum_var_config: QuantumVarConfig::default(),
            monitoring_config: MonitoringConfig::default(),
            limit_config: LimitConfig::default(),
            tengri_config: TengriIntegrationConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioAgentConfig {
    pub quantum_config: QuantumOptimizationConfig,
    pub optimization_config: OptimizationConfig,
    pub constraint_config: ConstraintConfig,
    pub analysis_config: AnalysisConfig,
    pub transaction_cost_rate: f64,
    pub market_impact_factor: f64,
    pub slippage_factor: f64,
    pub tengri_config: TengriIntegrationConfig,
}

impl Default for PortfolioAgentConfig {
    fn default() -> Self {
        Self {
            quantum_config: QuantumOptimizationConfig::default(),
            optimization_config: OptimizationConfig::default(),
            constraint_config: ConstraintConfig::default(),
            analysis_config: AnalysisConfig::default(),
            transaction_cost_rate: 0.001, // 0.1%
            market_impact_factor: 0.01,
            slippage_factor: 0.0005,
            tengri_config: TengriIntegrationConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressAgentConfig {
    pub quantum_config: QuantumMonteCarloConfig,
    pub scenario_config: ScenarioConfig,
    pub execution_config: ExecutionConfig,
    pub historical_config: HistoricalConfig,
    pub tengri_config: TengriIntegrationConfig,
}

impl Default for StressAgentConfig {
    fn default() -> Self {
        Self {
            quantum_config: QuantumMonteCarloConfig::default(),
            scenario_config: ScenarioConfig::default(),
            execution_config: ExecutionConfig::default(),
            historical_config: HistoricalConfig::default(),
            tengri_config: TengriIntegrationConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationAgentConfig {
    pub quantum_config: QuantumCorrelationConfig,
    pub regime_config: RegimeConfig,
    pub copula_config: CopulaConfig,
    pub dynamic_config: DynamicConfig,
    pub tengri_config: TengriIntegrationConfig,
}

impl Default for CorrelationAgentConfig {
    fn default() -> Self {
        Self {
            quantum_config: QuantumCorrelationConfig::default(),
            regime_config: RegimeConfig::default(),
            copula_config: CopulaConfig::default(),
            dynamic_config: DynamicConfig::default(),
            tengri_config: TengriIntegrationConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityAgentConfig {
    pub quantum_config: QuantumLiquidityConfig,
    pub impact_config: ImpactConfig,
    pub risk_config: LiquidityRiskConfig,
    pub monitoring_config: LiquidityMonitoringConfig,
    pub tengri_config: TengriIntegrationConfig,
}

impl Default for LiquidityAgentConfig {
    fn default() -> Self {
        Self {
            quantum_config: QuantumLiquidityConfig::default(),
            impact_config: ImpactConfig::default(),
            risk_config: LiquidityRiskConfig::default(),
            monitoring_config: LiquidityMonitoringConfig::default(),
            tengri_config: TengriIntegrationConfig::default(),
        }
    }
}

/// TENGRI oversight integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TengriIntegrationConfig {
    pub enabled: bool,
    pub endpoint: String,
    pub api_key: String,
    pub report_interval: Duration,
    pub alert_thresholds: TengriAlertThresholds,
}

impl Default for TengriIntegrationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            endpoint: "http://localhost:8080/tengri".to_string(),
            api_key: "default_key".to_string(),
            report_interval: Duration::from_secs(60),
            alert_thresholds: TengriAlertThresholds::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TengriAlertThresholds {
    pub calculation_time_threshold: Duration,
    pub error_rate_threshold: f64,
    pub quantum_advantage_threshold: f64,
}

impl Default for TengriAlertThresholds {
    fn default() -> Self {
        Self {
            calculation_time_threshold: Duration::from_micros(100),
            error_rate_threshold: 0.05, // 5%
            quantum_advantage_threshold: 0.1, // 10%
        }
    }
}

/// Agent contribution types for coordinated calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContributionType {
    Primary,
    Supporting,
    Validation,
    Fallback,
}

/// Stability levels for various metrics
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StabilityLevel {
    High,
    Medium,
    Low,
    Unstable,
}

/// Severity levels for stress scenarios
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SeverityLevel {
    Low,
    Medium,
    High,
    Extreme,
}

/// Execution strategy types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStrategyType {
    SlowExecution,
    SmallBlocks,
    OptimalTiming,
    LiquidityProvision,
}

/// Default implementations for configuration structures
macro_rules! impl_default_config {
    ($name:ident) => {
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct $name {
            pub enabled: bool,
        }
        
        impl Default for $name {
            fn default() -> Self {
                Self { enabled: true }
            }
        }
    };
}

impl_default_config!(QuantumVarConfig);
impl_default_config!(MonitoringConfig);
impl_default_config!(LimitConfig);
impl_default_config!(QuantumOptimizationConfig);
impl_default_config!(OptimizationConfig);
impl_default_config!(ConstraintConfig);
impl_default_config!(AnalysisConfig);
impl_default_config!(QuantumMonteCarloConfig);
impl_default_config!(ScenarioConfig);
impl_default_config!(ExecutionConfig);
impl_default_config!(HistoricalConfig);
impl_default_config!(QuantumCorrelationConfig);
impl_default_config!(RegimeConfig);
impl_default_config!(DynamicConfig);
impl_default_config!(QuantumLiquidityConfig);
impl_default_config!(ImpactConfig);
impl_default_config!(LiquidityRiskConfig);
impl_default_config!(LiquidityMonitoringConfig);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_performance_metrics() {
        let agent_id = Uuid::new_v4();
        let mut metrics = AgentPerformanceMetrics::new(agent_id, AgentType::RiskManagement);
        
        assert_eq!(metrics.total_calculations, 0);
        assert_eq!(metrics.successful_calculations, 0);
        assert_eq!(metrics.error_count, 0);
        
        metrics.start_calculation();
        assert_eq!(metrics.total_calculations, 1);
        
        let duration = Duration::from_micros(50);
        metrics.end_calculation(duration);
        assert_eq!(metrics.successful_calculations, 1);
        assert_eq!(metrics.last_calculation_time, duration);
        assert_eq!(metrics.average_calculation_time, duration);
    }

    #[test]
    fn test_agent_type_display() {
        assert_eq!(AgentType::RiskManagement.to_string(), "RiskManagement");
        assert_eq!(AgentType::PortfolioOptimization.to_string(), "PortfolioOptimization");
        assert_eq!(AgentType::StressTesting.to_string(), "StressTesting");
        assert_eq!(AgentType::CorrelationAnalysis.to_string(), "CorrelationAnalysis");
        assert_eq!(AgentType::LiquidityRisk.to_string(), "LiquidityRisk");
    }

    #[test]
    fn test_message_priority_ordering() {
        assert!(MessagePriority::Critical > MessagePriority::High);
        assert!(MessagePriority::High > MessagePriority::Normal);
        assert!(MessagePriority::Normal > MessagePriority::Low);
    }
}