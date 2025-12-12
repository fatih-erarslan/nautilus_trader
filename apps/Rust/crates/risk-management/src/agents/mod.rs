//! # RUV-Swarm Quantum Risk Management Agents
//!
//! This module contains specialized agents for quantum-enhanced risk modeling
//! and portfolio optimization, designed to work within the RUV-swarm framework
//! for ultra-high performance trading operations.
//!
//! ## Agents
//!
//! - `RiskManagementAgent`: Quantum-enhanced VaR calculations with real-time monitoring
//! - `PortfolioOptimizationAgent`: Multi-objective optimization using quantum annealing
//! - `StressTestingAgent`: Monte Carlo simulations with quantum random number generation
//! - `CorrelationAnalysisAgent`: Quantum correlation detection and regime change identification
//! - `LiquidityRiskAgent`: Real-time liquidity assessment with quantum uncertainty bounds
//!
//! ## Coordination Patterns
//!
//! All agents follow RUV-swarm coordination patterns:
//! - Real-time message passing with <10μs latency
//! - Distributed consensus for risk limit adjustments
//! - Quantum-enhanced uncertainty propagation
//! - TENGRI oversight integration for safety and compliance
//!
//! ## Performance Targets
//!
//! - Critical risk metrics: <100μs calculation time
//! - VaR calculations: <10μs for portfolio risk assessment
//! - Real-time monitoring: <1μs response time for limit breaches
//! - Agent coordination overhead: <5μs per message

pub mod risk_management;
pub mod portfolio_optimization;
pub mod stress_testing;
pub mod correlation_analysis;
pub mod liquidity_risk;
pub mod coordination;
pub mod base;

// Re-exports
pub use risk_management::RiskManagementAgent;
pub use portfolio_optimization::PortfolioOptimizationAgent;
pub use stress_testing::StressTestingAgent;
pub use correlation_analysis::CorrelationAnalysisAgent;
pub use liquidity_risk::LiquidityRiskAgent;
pub use coordination::*;
pub use base::*;

use std::sync::Arc;
use std::time::{Duration, Instant};
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info, warn, error};
use uuid::Uuid;

use crate::types::*;
use crate::quantum::*;

/// RUV-Swarm agent registry for risk management agents
#[derive(Debug)]
pub struct RiskSwarmRegistry {
    /// Risk management agent
    pub risk_agent: Arc<RwLock<RiskManagementAgent>>,
    /// Portfolio optimization agent
    pub portfolio_agent: Arc<RwLock<PortfolioOptimizationAgent>>,
    /// Stress testing agent
    pub stress_agent: Arc<RwLock<StressTestingAgent>>,
    /// Correlation analysis agent
    pub correlation_agent: Arc<RwLock<CorrelationAnalysisAgent>>,
    /// Liquidity risk agent
    pub liquidity_agent: Arc<RwLock<LiquidityRiskAgent>>,
    /// Agent coordination hub
    pub coordination_hub: Arc<RwLock<AgentCoordinationHub>>,
    /// Message routing
    pub message_router: Arc<RwLock<SwarmMessageRouter>>,
    /// Performance monitor
    pub performance_monitor: Arc<RwLock<SwarmPerformanceMonitor>>,
}

impl RiskSwarmRegistry {
    /// Create new risk swarm registry
    pub async fn new(config: RiskSwarmConfig) -> Result<Self> {
        info!("Initializing RUV-swarm risk management agents");
        
        // Create coordination infrastructure
        let coordination_hub = Arc::new(RwLock::new(
            AgentCoordinationHub::new(config.coordination_config.clone()).await?
        ));
        
        let message_router = Arc::new(RwLock::new(
            SwarmMessageRouter::new(config.routing_config.clone()).await?
        ));
        
        let performance_monitor = Arc::new(RwLock::new(
            SwarmPerformanceMonitor::new(config.performance_config.clone()).await?
        ));
        
        // Create specialized agents
        let risk_agent = Arc::new(RwLock::new(
            RiskManagementAgent::new(
                config.risk_agent_config.clone(),
                coordination_hub.clone(),
                message_router.clone(),
            ).await?
        ));
        
        let portfolio_agent = Arc::new(RwLock::new(
            PortfolioOptimizationAgent::new(
                config.portfolio_agent_config.clone(),
                coordination_hub.clone(),
                message_router.clone(),
            ).await?
        ));
        
        let stress_agent = Arc::new(RwLock::new(
            StressTestingAgent::new(
                config.stress_agent_config.clone(),
                coordination_hub.clone(),
                message_router.clone(),
            ).await?
        ));
        
        let correlation_agent = Arc::new(RwLock::new(
            CorrelationAnalysisAgent::new(
                config.correlation_agent_config.clone(),
                coordination_hub.clone(),
                message_router.clone(),
            ).await?
        ));
        
        let liquidity_agent = Arc::new(RwLock::new(
            LiquidityRiskAgent::new(
                config.liquidity_agent_config.clone(),
                coordination_hub.clone(),
                message_router.clone(),
            ).await?
        ));
        
        // Register agents with coordination hub
        {
            let mut hub = coordination_hub.write().await;
            hub.register_agent(risk_agent.clone()).await?;
            hub.register_agent(portfolio_agent.clone()).await?;
            hub.register_agent(stress_agent.clone()).await?;
            hub.register_agent(correlation_agent.clone()).await?;
            hub.register_agent(liquidity_agent.clone()).await?;
        }
        
        Ok(Self {
            risk_agent,
            portfolio_agent,
            stress_agent,
            correlation_agent,
            liquidity_agent,
            coordination_hub,
            message_router,
            performance_monitor,
        })
    }
    
    /// Start all agents
    pub async fn start_all_agents(&self) -> Result<()> {
        info!("Starting all RUV-swarm risk management agents");
        
        let start_futures = vec![
            self.risk_agent.write().await.start(),
            self.portfolio_agent.write().await.start(),
            self.stress_agent.write().await.start(),
            self.correlation_agent.write().await.start(),
            self.liquidity_agent.write().await.start(),
        ];
        
        futures::future::try_join_all(start_futures).await?;
        
        // Start coordination systems
        self.coordination_hub.write().await.start().await?;
        self.message_router.write().await.start().await?;
        self.performance_monitor.write().await.start().await?;
        
        info!("All agents started successfully");
        Ok(())
    }
    
    /// Stop all agents
    pub async fn stop_all_agents(&self) -> Result<()> {
        info!("Stopping all RUV-swarm risk management agents");
        
        // Stop coordination systems first
        self.performance_monitor.write().await.stop().await?;
        self.message_router.write().await.stop().await?;
        self.coordination_hub.write().await.stop().await?;
        
        let stop_futures = vec![
            self.risk_agent.write().await.stop(),
            self.portfolio_agent.write().await.stop(),
            self.stress_agent.write().await.stop(),
            self.correlation_agent.write().await.stop(),
            self.liquidity_agent.write().await.stop(),
        ];
        
        futures::future::try_join_all(stop_futures).await?;
        
        info!("All agents stopped successfully");
        Ok(())
    }
    
    /// Get swarm health status
    pub async fn get_swarm_health(&self) -> Result<SwarmHealthStatus> {
        let coordination_hub = self.coordination_hub.read().await;
        coordination_hub.get_swarm_health().await
    }
    
    /// Get swarm performance metrics
    pub async fn get_swarm_performance(&self) -> Result<SwarmPerformanceMetrics> {
        let performance_monitor = self.performance_monitor.read().await;
        performance_monitor.get_metrics().await
    }
    
    /// Execute coordinated risk calculation
    pub async fn execute_coordinated_risk_calculation(
        &self,
        portfolio: &Portfolio,
        calculation_type: RiskCalculationType,
    ) -> Result<CoordinatedRiskResult> {
        let start_time = Instant::now();
        
        // Coordinate calculation across agents
        let coordination_hub = self.coordination_hub.read().await;
        let result = coordination_hub.execute_coordinated_calculation(
            portfolio,
            calculation_type,
        ).await?;
        
        let elapsed = start_time.elapsed();
        if elapsed > Duration::from_micros(100) {
            warn!("Coordinated risk calculation took {:?}, exceeding 100μs target", elapsed);
        }
        
        Ok(result)
    }
}

/// Swarm configuration for risk management agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskSwarmConfig {
    pub coordination_config: CoordinationConfig,
    pub routing_config: RoutingConfig,
    pub performance_config: PerformanceConfig,
    pub risk_agent_config: RiskAgentConfig,
    pub portfolio_agent_config: PortfolioAgentConfig,
    pub stress_agent_config: StressAgentConfig,
    pub correlation_agent_config: CorrelationAgentConfig,
    pub liquidity_agent_config: LiquidityAgentConfig,
    pub tengri_integration: TengriIntegrationConfig,
}

impl Default for RiskSwarmConfig {
    fn default() -> Self {
        Self {
            coordination_config: CoordinationConfig::default(),
            routing_config: RoutingConfig::default(),
            performance_config: PerformanceConfig::default(),
            risk_agent_config: RiskAgentConfig::default(),
            portfolio_agent_config: PortfolioAgentConfig::default(),
            stress_agent_config: StressAgentConfig::default(),
            correlation_agent_config: CorrelationAgentConfig::default(),
            liquidity_agent_config: LiquidityAgentConfig::default(),
            tengri_integration: TengriIntegrationConfig::default(),
        }
    }
}

/// Risk calculation types for coordinated execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskCalculationType {
    VarCalculation { confidence_level: f64 },
    CvarCalculation { confidence_level: f64 },
    PortfolioOptimization { constraints: PortfolioConstraints },
    StressTest { scenarios: Vec<StressScenario> },
    CorrelationAnalysis { assets: Vec<Asset> },
    LiquidityAssessment { time_horizon: Duration },
    ComprehensiveRisk,
}

/// Coordinated risk calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatedRiskResult {
    pub calculation_type: RiskCalculationType,
    pub primary_result: RiskCalculationResult,
    pub supporting_results: Vec<RiskCalculationResult>,
    pub agent_contributions: Vec<AgentContribution>,
    pub calculation_time: Duration,
    pub quantum_advantage: f64,
    pub uncertainty_bounds: QuantumUncertaintyBounds,
}

/// Individual agent contribution to coordinated calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentContribution {
    pub agent_id: Uuid,
    pub agent_type: AgentType,
    pub contribution_type: ContributionType,
    pub result: RiskCalculationResult,
    pub calculation_time: Duration,
    pub confidence: f64,
}

/// Swarm health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmHealthStatus {
    pub overall_health: HealthLevel,
    pub agent_health: Vec<AgentHealthStatus>,
    pub coordination_health: HealthLevel,
    pub performance_health: HealthLevel,
    pub quantum_systems_health: HealthLevel,
    pub tengri_integration_health: HealthLevel,
}

/// Swarm performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmPerformanceMetrics {
    pub average_calculation_time: Duration,
    pub peak_calculation_time: Duration,
    pub throughput_per_second: f64,
    pub message_latency: Duration,
    pub coordination_overhead: Duration,
    pub quantum_advantage_ratio: f64,
    pub agent_performance: Vec<AgentPerformanceMetrics>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_risk_swarm_registry_creation() {
        let config = RiskSwarmConfig::default();
        let registry = RiskSwarmRegistry::new(config).await;
        assert!(registry.is_ok());
    }

    #[tokio::test]
    async fn test_swarm_agent_startup() {
        let config = RiskSwarmConfig::default();
        let registry = RiskSwarmRegistry::new(config).await.unwrap();
        
        let result = registry.start_all_agents().await;
        assert!(result.is_ok());
        
        let health = registry.get_swarm_health().await.unwrap();
        assert_eq!(health.overall_health, HealthLevel::Healthy);
        
        let _stop_result = registry.stop_all_agents().await;
    }

    #[tokio::test]
    async fn test_coordinated_risk_calculation_performance() {
        let config = RiskSwarmConfig::default();
        let registry = RiskSwarmRegistry::new(config).await.unwrap();
        registry.start_all_agents().await.unwrap();
        
        let portfolio = Portfolio::default();
        let calculation_type = RiskCalculationType::VarCalculation { confidence_level: 0.05 };
        
        let start_time = Instant::now();
        let result = registry.execute_coordinated_risk_calculation(&portfolio, calculation_type).await;
        let elapsed = start_time.elapsed();
        
        assert!(result.is_ok());
        assert!(elapsed < Duration::from_micros(100), "Coordinated calculation took {:?}", elapsed);
        
        let _stop_result = registry.stop_all_agents().await;
    }
}