//! # Risk Management Agent
//!
//! Quantum-enhanced portfolio optimization with real-time VaR calculations.
//! This agent implements ultra-fast risk assessment using quantum uncertainty
//! quantification and maintains sub-10μs calculation targets for critical metrics.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info, warn, error};
use uuid::Uuid;

use crate::types::*;
use crate::quantum::*;
use crate::var::*;
use super::base::*;
use super::coordination::*;

/// Risk Management Agent for quantum-enhanced VaR calculations
#[derive(Debug)]
pub struct RiskManagementAgent {
    /// Agent metadata
    pub agent_id: Uuid,
    pub agent_type: AgentType,
    pub status: AgentStatus,
    
    /// Configuration
    pub config: RiskAgentConfig,
    
    /// Quantum-enhanced VaR calculator
    pub quantum_var_calculator: Arc<RwLock<QuantumVarCalculator>>,
    
    /// Real-time risk monitor
    pub real_time_monitor: Arc<RwLock<RealTimeRiskMonitor>>,
    
    /// Risk limit manager
    pub limit_manager: Arc<RwLock<RiskLimitManager>>,
    
    /// Performance metrics
    pub performance_metrics: Arc<RwLock<AgentPerformanceMetrics>>,
    
    /// Coordination components
    pub coordination_hub: Arc<RwLock<AgentCoordinationHub>>,
    pub message_router: Arc<RwLock<SwarmMessageRouter>>,
    
    /// Message channels
    pub message_tx: mpsc::UnboundedSender<SwarmMessage>,
    pub message_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<SwarmMessage>>>>,
    
    /// TENGRI integration
    pub tengri_client: Arc<RwLock<TengriOversightClient>>,
}

impl RiskManagementAgent {
    /// Create new risk management agent
    pub async fn new(
        config: RiskAgentConfig,
        coordination_hub: Arc<RwLock<AgentCoordinationHub>>,
        message_router: Arc<RwLock<SwarmMessageRouter>>,
    ) -> Result<Self> {
        let agent_id = Uuid::new_v4();
        info!("Creating Risk Management Agent {}", agent_id);
        
        // Create quantum-enhanced VaR calculator
        let quantum_var_calculator = Arc::new(RwLock::new(
            QuantumVarCalculator::new(config.quantum_var_config.clone()).await?
        ));
        
        // Create real-time monitor
        let real_time_monitor = Arc::new(RwLock::new(
            RealTimeRiskMonitor::new(config.monitoring_config.clone()).await?
        ));
        
        // Create risk limit manager
        let limit_manager = Arc::new(RwLock::new(
            RiskLimitManager::new(config.limit_config.clone()).await?
        ));
        
        // Initialize performance metrics
        let performance_metrics = Arc::new(RwLock::new(
            AgentPerformanceMetrics::new(agent_id, AgentType::RiskManagement)
        ));
        
        // Create message channel
        let (message_tx, message_rx) = mpsc::unbounded_channel();
        let message_rx = Arc::new(RwLock::new(Some(message_rx)));
        
        // Initialize TENGRI client
        let tengri_client = Arc::new(RwLock::new(
            TengriOversightClient::new(config.tengri_config.clone()).await?
        ));
        
        Ok(Self {
            agent_id,
            agent_type: AgentType::RiskManagement,
            status: AgentStatus::Initializing,
            config,
            quantum_var_calculator,
            real_time_monitor,
            limit_manager,
            performance_metrics,
            coordination_hub,
            message_router,
            message_tx,
            message_rx,
            tengri_client,
        })
    }
    
    /// Calculate quantum-enhanced VaR
    pub async fn calculate_quantum_var(
        &self,
        portfolio: &Portfolio,
        confidence_level: f64,
    ) -> Result<QuantumVarResult> {
        let start_time = Instant::now();
        
        // Update performance metrics
        {
            let mut metrics = self.performance_metrics.write().await;
            metrics.start_calculation();
        }
        
        // Calculate VaR using quantum enhancement
        let quantum_var_calculator = self.quantum_var_calculator.read().await;
        let var_result = quantum_var_calculator.calculate_var(portfolio, confidence_level).await?;
        
        let calculation_time = start_time.elapsed();
        
        // Update performance metrics
        {
            let mut metrics = self.performance_metrics.write().await;
            metrics.end_calculation(calculation_time);
        }
        
        // Check performance target
        if calculation_time > Duration::from_micros(10) {
            warn!(
                "VaR calculation took {:?}, exceeding 10μs target for agent {}",
                calculation_time, self.agent_id
            );
        }
        
        // Report to TENGRI oversight
        {
            let tengri_client = self.tengri_client.read().await;
            tengri_client.report_calculation_metrics(
                self.agent_id,
                "quantum_var",
                calculation_time,
                var_result.confidence_score,
            ).await?;
        }
        
        Ok(var_result)
    }
    
    /// Calculate quantum-enhanced CVaR (Expected Shortfall)
    pub async fn calculate_quantum_cvar(
        &self,
        portfolio: &Portfolio,
        confidence_level: f64,
    ) -> Result<QuantumCvarResult> {
        let start_time = Instant::now();
        
        let quantum_var_calculator = self.quantum_var_calculator.read().await;
        let cvar_result = quantum_var_calculator.calculate_cvar(portfolio, confidence_level).await?;
        
        let calculation_time = start_time.elapsed();
        
        // Update performance metrics
        {
            let mut metrics = self.performance_metrics.write().await;
            metrics.record_calculation("quantum_cvar", calculation_time);
        }
        
        Ok(cvar_result)
    }
    
    /// Monitor real-time risk metrics
    pub async fn monitor_real_time_risk(&self, portfolio: &Portfolio) -> Result<RealTimeRiskMetrics> {
        let start_time = Instant::now();
        
        let real_time_monitor = self.real_time_monitor.read().await;
        let risk_metrics = real_time_monitor.calculate_real_time_metrics(portfolio).await?;
        
        let calculation_time = start_time.elapsed();
        
        // Check if risk limits are breached
        let limit_manager = self.limit_manager.read().await;
        let limit_breaches = limit_manager.check_limits(&risk_metrics).await?;
        
        // Send alerts for limit breaches
        if !limit_breaches.is_empty() {
            self.send_risk_limit_alerts(&limit_breaches).await?;
        }
        
        // Check performance target for real-time monitoring
        if calculation_time > Duration::from_micros(1) {
            warn!(
                "Real-time risk monitoring took {:?}, exceeding 1μs target for agent {}",
                calculation_time, self.agent_id
            );
        }
        
        Ok(risk_metrics)
    }
    
    /// Update portfolio positions for continuous monitoring
    pub async fn update_portfolio_positions(&self, positions: &[Position]) -> Result<()> {
        let mut real_time_monitor = self.real_time_monitor.write().await;
        real_time_monitor.update_positions(positions).await
    }
    
    /// Set risk limits
    pub async fn set_risk_limits(&self, limits: RiskLimits) -> Result<()> {
        let mut limit_manager = self.limit_manager.write().await;
        limit_manager.set_limits(limits).await
    }
    
    /// Send risk limit breach alerts
    async fn send_risk_limit_alerts(&self, breaches: &[RiskLimitBreach]) -> Result<()> {
        for breach in breaches {
            let alert_message = SwarmMessage {
                id: Uuid::new_v4(),
                sender_id: self.agent_id,
                sender_type: self.agent_type.clone(),
                recipient_id: None, // Broadcast to all agents
                message_type: MessageType::RiskAlert,
                content: MessageContent::RiskLimitBreach(breach.clone()),
                timestamp: chrono::Utc::now(),
                priority: MessagePriority::Critical,
                requires_response: true,
            };
            
            self.message_tx.send(alert_message)?;
        }
        
        // Also report to TENGRI oversight
        {
            let tengri_client = self.tengri_client.read().await;
            tengri_client.report_risk_limit_breaches(self.agent_id, breaches).await?;
        }
        
        Ok(())
    }
    
    /// Handle incoming swarm messages
    async fn handle_message(&self, message: SwarmMessage) -> Result<()> {
        match message.message_type {
            MessageType::VarCalculationRequest => {
                self.handle_var_calculation_request(message).await?;
            }
            MessageType::RiskMonitoringRequest => {
                self.handle_risk_monitoring_request(message).await?;
            }
            MessageType::PortfolioUpdate => {
                self.handle_portfolio_update(message).await?;
            }
            MessageType::RiskLimitUpdate => {
                self.handle_risk_limit_update(message).await?;
            }
            MessageType::HealthCheck => {
                self.handle_health_check(message).await?;
            }
            _ => {
                debug!("Received unhandled message type: {:?}", message.message_type);
            }
        }
        Ok(())
    }
    
    /// Handle VaR calculation request
    async fn handle_var_calculation_request(&self, message: SwarmMessage) -> Result<()> {
        if let MessageContent::VarCalculationRequest { portfolio, confidence_level } = message.content {
            let var_result = self.calculate_quantum_var(&portfolio, confidence_level).await?;
            
            let response = SwarmMessage {
                id: Uuid::new_v4(),
                sender_id: self.agent_id,
                sender_type: self.agent_type.clone(),
                recipient_id: Some(message.sender_id),
                message_type: MessageType::VarCalculationResponse,
                content: MessageContent::VarCalculationResponse(var_result),
                timestamp: chrono::Utc::now(),
                priority: MessagePriority::High,
                requires_response: false,
            };
            
            self.message_tx.send(response)?;
        }
        Ok(())
    }
    
    /// Handle risk monitoring request
    async fn handle_risk_monitoring_request(&self, message: SwarmMessage) -> Result<()> {
        if let MessageContent::RiskMonitoringRequest { portfolio } = message.content {
            let risk_metrics = self.monitor_real_time_risk(&portfolio).await?;
            
            let response = SwarmMessage {
                id: Uuid::new_v4(),
                sender_id: self.agent_id,
                sender_type: self.agent_type.clone(),
                recipient_id: Some(message.sender_id),
                message_type: MessageType::RiskMonitoringResponse,
                content: MessageContent::RiskMonitoringResponse(risk_metrics),
                timestamp: chrono::Utc::now(),
                priority: MessagePriority::Normal,
                requires_response: false,
            };
            
            self.message_tx.send(response)?;
        }
        Ok(())
    }
    
    /// Handle portfolio update
    async fn handle_portfolio_update(&self, message: SwarmMessage) -> Result<()> {
        if let MessageContent::PortfolioUpdate { positions } = message.content {
            self.update_portfolio_positions(&positions).await?;
        }
        Ok(())
    }
    
    /// Handle risk limit update
    async fn handle_risk_limit_update(&self, message: SwarmMessage) -> Result<()> {
        if let MessageContent::RiskLimitUpdate { limits } = message.content {
            self.set_risk_limits(limits).await?;
        }
        Ok(())
    }
    
    /// Handle health check
    async fn handle_health_check(&self, message: SwarmMessage) -> Result<()> {
        let health_status = self.get_health_status().await?;
        
        let response = SwarmMessage {
            id: Uuid::new_v4(),
            sender_id: self.agent_id,
            sender_type: self.agent_type.clone(),
            recipient_id: Some(message.sender_id),
            message_type: MessageType::HealthCheckResponse,
            content: MessageContent::HealthCheckResponse(health_status),
            timestamp: chrono::Utc::now(),
            priority: MessagePriority::Low,
            requires_response: false,
        };
        
        self.message_tx.send(response)?;
        Ok(())
    }
    
    /// Get agent health status
    pub async fn get_health_status(&self) -> Result<AgentHealthStatus> {
        let performance_metrics = self.performance_metrics.read().await;
        let health_level = if performance_metrics.average_calculation_time < Duration::from_micros(10) {
            HealthLevel::Healthy
        } else if performance_metrics.average_calculation_time < Duration::from_micros(50) {
            HealthLevel::Warning
        } else {
            HealthLevel::Critical
        };
        
        Ok(AgentHealthStatus {
            agent_id: self.agent_id,
            agent_type: self.agent_type.clone(),
            health_level,
            last_calculation_time: performance_metrics.last_calculation_time,
            average_calculation_time: performance_metrics.average_calculation_time,
            total_calculations: performance_metrics.total_calculations,
            error_count: performance_metrics.error_count,
            uptime: performance_metrics.uptime(),
        })
    }
}

#[async_trait]
impl SwarmAgent for RiskManagementAgent {
    async fn start(&mut self) -> Result<()> {
        info!("Starting Risk Management Agent {}", self.agent_id);
        self.status = AgentStatus::Starting;
        
        // Start message processing loop
        let mut message_rx = self.message_rx.write().await.take()
            .ok_or_else(|| anyhow!("Message receiver already taken"))?;
        
        let agent_clone = Arc::new(RwLock::new(self));
        tokio::spawn(async move {
            while let Some(message) = message_rx.recv().await {
                let agent = agent_clone.read().await;
                if let Err(e) = agent.handle_message(message).await {
                    error!("Error handling message: {}", e);
                }
            }
        });
        
        self.status = AgentStatus::Running;
        info!("Risk Management Agent {} started successfully", self.agent_id);
        Ok(())
    }
    
    async fn stop(&mut self) -> Result<()> {
        info!("Stopping Risk Management Agent {}", self.agent_id);
        self.status = AgentStatus::Stopping;
        
        // Cleanup and resource deallocation
        // The message processing loop will end when the sender is dropped
        
        self.status = AgentStatus::Stopped;
        info!("Risk Management Agent {} stopped successfully", self.agent_id);
        Ok(())
    }
    
    async fn get_agent_id(&self) -> Uuid {
        self.agent_id
    }
    
    async fn get_agent_type(&self) -> AgentType {
        self.agent_type.clone()
    }
    
    async fn get_status(&self) -> AgentStatus {
        self.status.clone()
    }
    
    async fn get_performance_metrics(&self) -> Result<AgentPerformanceMetrics> {
        let metrics = self.performance_metrics.read().await;
        Ok(metrics.clone())
    }
    
    async fn handle_coordination_message(&self, message: CoordinationMessage) -> Result<CoordinationResponse> {
        match message.message_type {
            CoordinationMessageType::VarCalculation => {
                if let Some(portfolio) = message.portfolio {
                    let confidence_level = message.parameters.get("confidence_level")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.05);
                    
                    let var_result = self.calculate_quantum_var(&portfolio, confidence_level).await?;
                    
                    Ok(CoordinationResponse {
                        agent_id: self.agent_id,
                        success: true,
                        result: Some(RiskCalculationResult::QuantumVar(var_result)),
                        error: None,
                        calculation_time: var_result.calculation_time,
                    })
                } else {
                    Ok(CoordinationResponse {
                        agent_id: self.agent_id,
                        success: false,
                        result: None,
                        error: Some("Portfolio data required for VaR calculation".to_string()),
                        calculation_time: Duration::from_nanos(0),
                    })
                }
            }
            _ => {
                Ok(CoordinationResponse {
                    agent_id: self.agent_id,
                    success: false,
                    result: None,
                    error: Some(format!("Unsupported coordination message type: {:?}", message.message_type)),
                    calculation_time: Duration::from_nanos(0),
                })
            }
        }
    }
}

/// Quantum-enhanced VaR calculator
#[derive(Debug)]
pub struct QuantumVarCalculator {
    config: QuantumVarConfig,
    quantum_engine: Arc<RwLock<QuantumUncertaintyEngine>>,
    cache: Arc<RwLock<VarCalculationCache>>,
}

impl QuantumVarCalculator {
    pub async fn new(config: QuantumVarConfig) -> Result<Self> {
        let quantum_engine = Arc::new(RwLock::new(
            QuantumUncertaintyEngine::new(config.quantum_config.clone()).await?
        ));
        
        let cache = Arc::new(RwLock::new(
            VarCalculationCache::new(config.cache_size)
        ));
        
        Ok(Self {
            config,
            quantum_engine,
            cache,
        })
    }
    
    pub async fn calculate_var(
        &self,
        portfolio: &Portfolio,
        confidence_level: f64,
    ) -> Result<QuantumVarResult> {
        let start_time = Instant::now();
        
        // Check cache first
        let cache_key = self.generate_cache_key(portfolio, confidence_level);
        {
            let cache = self.cache.read().await;
            if let Some(cached_result) = cache.get(&cache_key) {
                if cached_result.is_valid() {
                    return Ok(cached_result.clone());
                }
            }
        }
        
        // Convert portfolio to quantum format
        let quantum_data = self.portfolio_to_quantum_data(portfolio).await?;
        
        // Calculate uncertainty quantification
        let quantum_engine = self.quantum_engine.read().await;
        let uncertainty_quantification = quantum_engine.quantify_uncertainty(
            &quantum_data.returns,
            &quantum_data.targets,
        ).await?;
        
        // Calculate classical VaR for comparison
        let classical_var = self.calculate_classical_var(portfolio, confidence_level).await?;
        
        // Combine quantum and classical results
        let quantum_var = self.combine_quantum_classical_var(
            classical_var,
            &uncertainty_quantification,
            confidence_level,
        ).await?;
        
        let calculation_time = start_time.elapsed();
        
        let result = QuantumVarResult {
            classical_var,
            quantum_var,
            quantum_advantage: uncertainty_quantification.quantum_advantage,
            confidence_level,
            confidence_score: uncertainty_quantification.confidence_score(),
            uncertainty_bounds: uncertainty_quantification.conformal_intervals.clone(),
            calculation_time,
            timestamp: chrono::Utc::now(),
        };
        
        // Cache the result
        {
            let mut cache = self.cache.write().await;
            cache.insert(cache_key, result.clone());
        }
        
        Ok(result)
    }
    
    pub async fn calculate_cvar(
        &self,
        portfolio: &Portfolio,
        confidence_level: f64,
    ) -> Result<QuantumCvarResult> {
        let start_time = Instant::now();
        
        // Calculate VaR first
        let var_result = self.calculate_var(portfolio, confidence_level).await?;
        
        // Calculate CVaR using quantum enhancement
        let quantum_data = self.portfolio_to_quantum_data(portfolio).await?;
        let quantum_engine = self.quantum_engine.read().await;
        
        // Calculate expected shortfall with quantum uncertainty
        let cvar_calculation = quantum_engine.calculate_expected_shortfall(
            &quantum_data.returns,
            var_result.quantum_var,
            confidence_level,
        ).await?;
        
        let calculation_time = start_time.elapsed();
        
        Ok(QuantumCvarResult {
            var_result,
            quantum_cvar: cvar_calculation.expected_shortfall,
            tail_expectation: cvar_calculation.tail_expectation,
            uncertainty_bounds: cvar_calculation.uncertainty_bounds,
            calculation_time,
            timestamp: chrono::Utc::now(),
        })
    }
    
    async fn portfolio_to_quantum_data(&self, portfolio: &Portfolio) -> Result<QuantumPortfolioData> {
        // Implementation similar to the main risk manager
        let n_observations = portfolio.returns.len().max(1);
        let n_assets = portfolio.positions.len().max(1);
        
        let returns = if portfolio.returns.is_empty() {
            Array2::zeros((1, n_assets))
        } else if n_assets == 1 {
            Array2::from_shape_vec(
                (n_observations, 1),
                portfolio.returns.clone(),
            )?
        } else {
            let mut returns_matrix = Array2::zeros((n_observations, n_assets));
            for i in 0..n_observations {
                for j in 0..n_assets {
                    returns_matrix[[i, j]] = portfolio.returns[i];
                }
            }
            returns_matrix
        };
        
        let targets = if portfolio.targets.is_empty() {
            Array1::zeros(n_observations)
        } else {
            Array1::from_vec(portfolio.targets.clone())
        };
        
        Ok(QuantumPortfolioData {
            returns,
            targets,
            positions: portfolio.positions.clone(),
            market_data: portfolio.market_data.clone(),
        })
    }
    
    async fn calculate_classical_var(&self, portfolio: &Portfolio, confidence_level: f64) -> Result<f64> {
        // Classical VaR calculation using historical simulation
        if portfolio.returns.is_empty() {
            return Ok(0.0);
        }
        
        let mut sorted_returns = portfolio.returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let var_index = ((1.0 - confidence_level) * sorted_returns.len() as f64).floor() as usize;
        Ok(-sorted_returns[var_index.min(sorted_returns.len() - 1)])
    }
    
    async fn combine_quantum_classical_var(
        &self,
        classical_var: f64,
        uncertainty_quantification: &UncertaintyQuantification,
        confidence_level: f64,
    ) -> Result<f64> {
        // Combine classical and quantum estimates using confidence-weighted averaging
        let quantum_weight = uncertainty_quantification.quantum_advantage.min(1.0).max(0.0);
        let classical_weight = 1.0 - quantum_weight;
        
        let quantum_var_estimate = uncertainty_quantification.mean_uncertainty() * classical_var;
        
        Ok(classical_weight * classical_var + quantum_weight * quantum_var_estimate)
    }
    
    fn generate_cache_key(&self, portfolio: &Portfolio, confidence_level: f64) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        portfolio.positions.len().hash(&mut hasher);
        let conf_hash = (confidence_level * 1000000.0) as u64;
        conf_hash.hash(&mut hasher);
        if !portfolio.returns.is_empty() {
            let return_hash = (portfolio.returns[0] * 1000000.0) as i64;
            return_hash.hash(&mut hasher);
        }
        
        format!("var_{}_{}", hasher.finish(), confidence_level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_risk_management_agent_creation() {
        let config = RiskAgentConfig::default();
        let coordination_hub = Arc::new(RwLock::new(
            AgentCoordinationHub::new(CoordinationConfig::default()).await.unwrap()
        ));
        let message_router = Arc::new(RwLock::new(
            SwarmMessageRouter::new(RoutingConfig::default()).await.unwrap()
        ));
        
        let agent = RiskManagementAgent::new(config, coordination_hub, message_router).await;
        assert!(agent.is_ok());
    }

    #[tokio::test]
    async fn test_quantum_var_calculation_performance() {
        let config = RiskAgentConfig::default();
        let coordination_hub = Arc::new(RwLock::new(
            AgentCoordinationHub::new(CoordinationConfig::default()).await.unwrap()
        ));
        let message_router = Arc::new(RwLock::new(
            SwarmMessageRouter::new(RoutingConfig::default()).await.unwrap()
        ));
        
        let agent = RiskManagementAgent::new(config, coordination_hub, message_router).await.unwrap();
        
        let portfolio = Portfolio::default();
        let confidence_level = 0.05;
        
        let start_time = Instant::now();
        let var_result = agent.calculate_quantum_var(&portfolio, confidence_level).await;
        let elapsed = start_time.elapsed();
        
        assert!(var_result.is_ok());
        assert!(elapsed < Duration::from_micros(10), "VaR calculation took {:?}", elapsed);
    }

    #[tokio::test]
    async fn test_real_time_risk_monitoring_performance() {
        let config = RiskAgentConfig::default();
        let coordination_hub = Arc::new(RwLock::new(
            AgentCoordinationHub::new(CoordinationConfig::default()).await.unwrap()
        ));
        let message_router = Arc::new(RwLock::new(
            SwarmMessageRouter::new(RoutingConfig::default()).await.unwrap()
        ));
        
        let agent = RiskManagementAgent::new(config, coordination_hub, message_router).await.unwrap();
        
        let portfolio = Portfolio::default();
        
        let start_time = Instant::now();
        let risk_metrics = agent.monitor_real_time_risk(&portfolio).await;
        let elapsed = start_time.elapsed();
        
        assert!(risk_metrics.is_ok());
        assert!(elapsed < Duration::from_micros(1), "Real-time monitoring took {:?}", elapsed);
    }
}