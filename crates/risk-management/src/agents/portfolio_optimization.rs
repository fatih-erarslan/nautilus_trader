//! # Portfolio Optimization Agent
//!
//! Multi-objective optimization using quantum annealing techniques.
//! This agent implements quantum-enhanced portfolio optimization with
//! real-time constraint handling and sub-100μs optimization targets.

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
use crate::portfolio::*;
use super::base::*;
use super::coordination::*;

/// Portfolio Optimization Agent using quantum annealing
#[derive(Debug)]
pub struct PortfolioOptimizationAgent {
    /// Agent metadata
    pub agent_id: Uuid,
    pub agent_type: AgentType,
    pub status: AgentStatus,
    
    /// Configuration
    pub config: PortfolioAgentConfig,
    
    /// Quantum annealing optimizer
    pub quantum_optimizer: Arc<RwLock<QuantumAnnealingOptimizer>>,
    
    /// Multi-objective optimizer
    pub multi_objective_optimizer: Arc<RwLock<MultiObjectiveOptimizer>>,
    
    /// Constraint manager
    pub constraint_manager: Arc<RwLock<PortfolioConstraintManager>>,
    
    /// Risk-return analyzer
    pub risk_return_analyzer: Arc<RwLock<RiskReturnAnalyzer>>,
    
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

impl PortfolioOptimizationAgent {
    /// Create new portfolio optimization agent
    pub async fn new(
        config: PortfolioAgentConfig,
        coordination_hub: Arc<RwLock<AgentCoordinationHub>>,
        message_router: Arc<RwLock<SwarmMessageRouter>>,
    ) -> Result<Self> {
        let agent_id = Uuid::new_v4();
        info!("Creating Portfolio Optimization Agent {}", agent_id);
        
        // Create quantum annealing optimizer
        let quantum_optimizer = Arc::new(RwLock::new(
            QuantumAnnealingOptimizer::new(config.quantum_config.clone()).await?
        ));
        
        // Create multi-objective optimizer
        let multi_objective_optimizer = Arc::new(RwLock::new(
            MultiObjectiveOptimizer::new(config.optimization_config.clone()).await?
        ));
        
        // Create constraint manager
        let constraint_manager = Arc::new(RwLock::new(
            PortfolioConstraintManager::new(config.constraint_config.clone()).await?
        ));
        
        // Create risk-return analyzer
        let risk_return_analyzer = Arc::new(RwLock::new(
            RiskReturnAnalyzer::new(config.analysis_config.clone()).await?
        ));
        
        // Initialize performance metrics
        let performance_metrics = Arc::new(RwLock::new(
            AgentPerformanceMetrics::new(agent_id, AgentType::PortfolioOptimization)
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
            agent_type: AgentType::PortfolioOptimization,
            status: AgentStatus::Initializing,
            config,
            quantum_optimizer,
            multi_objective_optimizer,
            constraint_manager,
            risk_return_analyzer,
            performance_metrics,
            coordination_hub,
            message_router,
            message_tx,
            message_rx,
            tengri_client,
        })
    }
    
    /// Optimize portfolio using quantum annealing
    pub async fn optimize_portfolio_quantum(
        &self,
        assets: &[Asset],
        constraints: &PortfolioConstraints,
        objectives: &[OptimizationObjective],
    ) -> Result<QuantumOptimizedPortfolio> {
        let start_time = Instant::now();
        
        // Update performance metrics
        {
            let mut metrics = self.performance_metrics.write().await;
            metrics.start_calculation();
        }
        
        // Validate constraints
        let constraint_manager = self.constraint_manager.read().await;
        constraint_manager.validate_constraints(constraints).await?;
        
        // Analyze risk-return characteristics
        let risk_return_analyzer = self.risk_return_analyzer.read().await;
        let risk_return_analysis = risk_return_analyzer.analyze_assets(assets).await?;
        
        // Perform quantum annealing optimization
        let quantum_optimizer = self.quantum_optimizer.read().await;
        let quantum_result = quantum_optimizer.optimize(
            assets,
            constraints,
            objectives,
            &risk_return_analysis,
        ).await?;
        
        // Perform classical multi-objective optimization for comparison
        let multi_objective_optimizer = self.multi_objective_optimizer.read().await;
        let classical_result = multi_objective_optimizer.optimize(
            assets,
            constraints,
            objectives,
            &risk_return_analysis,
        ).await?;
        
        let calculation_time = start_time.elapsed();
        
        // Update performance metrics
        {
            let mut metrics = self.performance_metrics.write().await;
            metrics.end_calculation(calculation_time);
        }
        
        // Check performance target
        if calculation_time > Duration::from_micros(100) {
            warn!(
                "Portfolio optimization took {:?}, exceeding 100μs target for agent {}",
                calculation_time, self.agent_id
            );
        }
        
        // Combine quantum and classical results
        let optimized_portfolio = self.combine_optimization_results(
            quantum_result,
            classical_result,
            &risk_return_analysis,
        ).await?;
        
        // Report to TENGRI oversight
        {
            let tengri_client = self.tengri_client.read().await;
            tengri_client.report_optimization_metrics(
                self.agent_id,
                "quantum_portfolio_optimization",
                calculation_time,
                optimized_portfolio.optimization_quality,
            ).await?;
        }
        
        Ok(optimized_portfolio)
    }
    
    /// Optimize portfolio allocation with risk budgeting
    pub async fn optimize_with_risk_budgeting(
        &self,
        assets: &[Asset],
        risk_budget: &RiskBudget,
        target_return: f64,
    ) -> Result<RiskBudgetedPortfolio> {
        let start_time = Instant::now();
        
        // Convert risk budget to constraints
        let constraint_manager = self.constraint_manager.read().await;
        let constraints = constraint_manager.risk_budget_to_constraints(risk_budget).await?;
        
        // Define objectives for risk budgeting
        let objectives = vec![
            OptimizationObjective::MaximizeReturn { target: target_return },
            OptimizationObjective::MinimizeRisk,
            OptimizationObjective::RiskBudgetCompliance { budget: risk_budget.clone() },
        ];
        
        // Perform optimization
        let optimized_portfolio = self.optimize_portfolio_quantum(
            assets,
            &constraints,
            &objectives,
        ).await?;
        
        let calculation_time = start_time.elapsed();
        
        // Create risk budgeted portfolio
        let risk_budgeted_portfolio = RiskBudgetedPortfolio {
            base_portfolio: optimized_portfolio,
            risk_budget: risk_budget.clone(),
            risk_contribution: self.calculate_risk_contribution(&optimized_portfolio).await?,
            budget_utilization: self.calculate_budget_utilization(&optimized_portfolio, risk_budget).await?,
            calculation_time,
            timestamp: chrono::Utc::now(),
        };
        
        Ok(risk_budgeted_portfolio)
    }
    
    /// Rebalance portfolio maintaining optimization objectives
    pub async fn rebalance_portfolio(
        &self,
        current_portfolio: &Portfolio,
        market_update: &MarketUpdate,
        rebalancing_constraints: &RebalancingConstraints,
    ) -> Result<RebalancedPortfolio> {
        let start_time = Instant::now();
        
        // Analyze market impact of rebalancing
        let risk_return_analyzer = self.risk_return_analyzer.read().await;
        let market_impact = risk_return_analyzer.analyze_market_impact(
            current_portfolio,
            market_update,
        ).await?;
        
        // Calculate optimal rebalancing using quantum optimization
        let quantum_optimizer = self.quantum_optimizer.read().await;
        let rebalancing_plan = quantum_optimizer.calculate_rebalancing(
            current_portfolio,
            &market_impact,
            rebalancing_constraints,
        ).await?;
        
        let calculation_time = start_time.elapsed();
        
        Ok(RebalancedPortfolio {
            original_portfolio: current_portfolio.clone(),
            rebalancing_plan,
            market_impact,
            expected_performance: self.calculate_expected_performance(&rebalancing_plan).await?,
            execution_cost: self.calculate_execution_cost(&rebalancing_plan).await?,
            calculation_time,
            timestamp: chrono::Utc::now(),
        })
    }
    
    /// Calculate dynamic risk allocation
    pub async fn calculate_dynamic_risk_allocation(
        &self,
        assets: &[Asset],
        market_regime: &MarketRegime,
        risk_tolerance: f64,
    ) -> Result<DynamicRiskAllocation> {
        let start_time = Instant::now();
        
        // Analyze current market regime
        let risk_return_analyzer = self.risk_return_analyzer.read().await;
        let regime_analysis = risk_return_analyzer.analyze_market_regime(market_regime).await?;
        
        // Adjust risk allocation based on regime
        let quantum_optimizer = self.quantum_optimizer.read().await;
        let risk_allocation = quantum_optimizer.calculate_dynamic_allocation(
            assets,
            &regime_analysis,
            risk_tolerance,
        ).await?;
        
        let calculation_time = start_time.elapsed();
        
        Ok(DynamicRiskAllocation {
            asset_allocation: risk_allocation.weights,
            risk_contribution: risk_allocation.risk_contributions,
            market_regime: market_regime.clone(),
            regime_confidence: regime_analysis.confidence,
            expected_return: risk_allocation.expected_return,
            expected_risk: risk_allocation.expected_risk,
            calculation_time,
            timestamp: chrono::Utc::now(),
        })
    }
    
    /// Combine quantum and classical optimization results
    async fn combine_optimization_results(
        &self,
        quantum_result: QuantumOptimizationResult,
        classical_result: ClassicalOptimizationResult,
        risk_return_analysis: &RiskReturnAnalysis,
    ) -> Result<QuantumOptimizedPortfolio> {
        // Weight results based on quantum advantage and confidence
        let quantum_weight = quantum_result.quantum_advantage.min(1.0).max(0.0);
        let classical_weight = 1.0 - quantum_weight;
        
        // Combine weights
        let combined_weights = quantum_result.weights
            .iter()
            .zip(classical_result.weights.iter())
            .map(|(q_weight, c_weight)| {
                quantum_weight * q_weight + classical_weight * c_weight
            })
            .collect();
        
        // Calculate combined metrics
        let expected_return = quantum_weight * quantum_result.expected_return 
            + classical_weight * classical_result.expected_return;
        
        let expected_risk = quantum_weight * quantum_result.expected_risk 
            + classical_weight * classical_result.expected_risk;
        
        let sharpe_ratio = if expected_risk > 0.0 {
            expected_return / expected_risk
        } else {
            0.0
        };
        
        Ok(QuantumOptimizedPortfolio {
            weights: combined_weights,
            expected_return,
            expected_risk,
            sharpe_ratio,
            quantum_result,
            classical_result,
            quantum_advantage: quantum_result.quantum_advantage,
            optimization_quality: quantum_result.optimization_quality,
            constraint_satisfaction: self.check_constraint_satisfaction(&combined_weights).await?,
            calculation_time: quantum_result.calculation_time + classical_result.calculation_time,
            timestamp: chrono::Utc::now(),
        })
    }
    
    /// Calculate risk contribution for each asset
    async fn calculate_risk_contribution(
        &self,
        portfolio: &QuantumOptimizedPortfolio,
    ) -> Result<Vec<f64>> {
        let risk_return_analyzer = self.risk_return_analyzer.read().await;
        risk_return_analyzer.calculate_risk_contribution(&portfolio.weights).await
    }
    
    /// Calculate risk budget utilization
    async fn calculate_budget_utilization(
        &self,
        portfolio: &QuantumOptimizedPortfolio,
        risk_budget: &RiskBudget,
    ) -> Result<f64> {
        let risk_contribution = self.calculate_risk_contribution(portfolio).await?;
        
        let utilization = risk_contribution
            .iter()
            .zip(risk_budget.allocations.iter())
            .map(|(contribution, budget)| contribution / budget.max(1e-10))
            .sum::<f64>() / risk_contribution.len() as f64;
        
        Ok(utilization)
    }
    
    /// Calculate expected performance of rebalancing plan
    async fn calculate_expected_performance(
        &self,
        rebalancing_plan: &RebalancingPlan,
    ) -> Result<ExpectedPerformance> {
        let risk_return_analyzer = self.risk_return_analyzer.read().await;
        risk_return_analyzer.calculate_expected_performance(rebalancing_plan).await
    }
    
    /// Calculate execution cost of rebalancing
    async fn calculate_execution_cost(
        &self,
        rebalancing_plan: &RebalancingPlan,
    ) -> Result<ExecutionCost> {
        // Calculate transaction costs, market impact, and slippage
        let total_turnover = rebalancing_plan.trades
            .iter()
            .map(|trade| trade.quantity.abs())
            .sum::<f64>();
        
        let transaction_cost = total_turnover * self.config.transaction_cost_rate;
        let market_impact_cost = self.calculate_market_impact_cost(rebalancing_plan).await?;
        let slippage_cost = self.calculate_slippage_cost(rebalancing_plan).await?;
        
        Ok(ExecutionCost {
            transaction_cost,
            market_impact_cost,
            slippage_cost,
            total_cost: transaction_cost + market_impact_cost + slippage_cost,
        })
    }
    
    async fn calculate_market_impact_cost(&self, rebalancing_plan: &RebalancingPlan) -> Result<f64> {
        // Simplified market impact model
        let impact_factor = self.config.market_impact_factor;
        let total_impact = rebalancing_plan.trades
            .iter()
            .map(|trade| impact_factor * trade.quantity.abs().sqrt())
            .sum::<f64>();
        
        Ok(total_impact)
    }
    
    async fn calculate_slippage_cost(&self, rebalancing_plan: &RebalancingPlan) -> Result<f64> {
        // Simplified slippage model
        let slippage_factor = self.config.slippage_factor;
        let total_slippage = rebalancing_plan.trades
            .iter()
            .map(|trade| slippage_factor * trade.quantity.abs())
            .sum::<f64>();
        
        Ok(total_slippage)
    }
    
    /// Check constraint satisfaction
    async fn check_constraint_satisfaction(&self, weights: &[f64]) -> Result<ConstraintSatisfaction> {
        let constraint_manager = self.constraint_manager.read().await;
        constraint_manager.check_satisfaction(weights).await
    }
    
    /// Handle incoming swarm messages
    async fn handle_message(&self, message: SwarmMessage) -> Result<()> {
        match message.message_type {
            MessageType::PortfolioOptimizationRequest => {
                self.handle_optimization_request(message).await?;
            }
            MessageType::RebalancingRequest => {
                self.handle_rebalancing_request(message).await?;
            }
            MessageType::RiskAllocationRequest => {
                self.handle_risk_allocation_request(message).await?;
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
    
    async fn handle_optimization_request(&self, message: SwarmMessage) -> Result<()> {
        if let MessageContent::PortfolioOptimizationRequest { assets, constraints, objectives } = message.content {
            let optimized_portfolio = self.optimize_portfolio_quantum(&assets, &constraints, &objectives).await?;
            
            let response = SwarmMessage {
                id: Uuid::new_v4(),
                sender_id: self.agent_id,
                sender_type: self.agent_type.clone(),
                recipient_id: Some(message.sender_id),
                message_type: MessageType::PortfolioOptimizationResponse,
                content: MessageContent::PortfolioOptimizationResponse(optimized_portfolio),
                timestamp: chrono::Utc::now(),
                priority: MessagePriority::High,
                requires_response: false,
            };
            
            self.message_tx.send(response)?;
        }
        Ok(())
    }
    
    async fn handle_rebalancing_request(&self, message: SwarmMessage) -> Result<()> {
        if let MessageContent::RebalancingRequest { portfolio, market_update, constraints } = message.content {
            let rebalanced_portfolio = self.rebalance_portfolio(&portfolio, &market_update, &constraints).await?;
            
            let response = SwarmMessage {
                id: Uuid::new_v4(),
                sender_id: self.agent_id,
                sender_type: self.agent_type.clone(),
                recipient_id: Some(message.sender_id),
                message_type: MessageType::RebalancingResponse,
                content: MessageContent::RebalancingResponse(rebalanced_portfolio),
                timestamp: chrono::Utc::now(),
                priority: MessagePriority::High,
                requires_response: false,
            };
            
            self.message_tx.send(response)?;
        }
        Ok(())
    }
    
    async fn handle_risk_allocation_request(&self, message: SwarmMessage) -> Result<()> {
        if let MessageContent::RiskAllocationRequest { assets, market_regime, risk_tolerance } = message.content {
            let risk_allocation = self.calculate_dynamic_risk_allocation(&assets, &market_regime, risk_tolerance).await?;
            
            let response = SwarmMessage {
                id: Uuid::new_v4(),
                sender_id: self.agent_id,
                sender_type: self.agent_type.clone(),
                recipient_id: Some(message.sender_id),
                message_type: MessageType::RiskAllocationResponse,
                content: MessageContent::RiskAllocationResponse(risk_allocation),
                timestamp: chrono::Utc::now(),
                priority: MessagePriority::Normal,
                requires_response: false,
            };
            
            self.message_tx.send(response)?;
        }
        Ok(())
    }
    
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
    
    pub async fn get_health_status(&self) -> Result<AgentHealthStatus> {
        let performance_metrics = self.performance_metrics.read().await;
        let health_level = if performance_metrics.average_calculation_time < Duration::from_micros(100) {
            HealthLevel::Healthy
        } else if performance_metrics.average_calculation_time < Duration::from_micros(500) {
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
impl SwarmAgent for PortfolioOptimizationAgent {
    async fn start(&mut self) -> Result<()> {
        info!("Starting Portfolio Optimization Agent {}", self.agent_id);
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
        info!("Portfolio Optimization Agent {} started successfully", self.agent_id);
        Ok(())
    }
    
    async fn stop(&mut self) -> Result<()> {
        info!("Stopping Portfolio Optimization Agent {}", self.agent_id);
        self.status = AgentStatus::Stopping;
        
        self.status = AgentStatus::Stopped;
        info!("Portfolio Optimization Agent {} stopped successfully", self.agent_id);
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
            CoordinationMessageType::PortfolioOptimization => {
                if let (Some(assets), Some(constraints)) = (message.assets, message.constraints) {
                    let objectives = message.objectives.unwrap_or(vec![
                        OptimizationObjective::MaximizeReturn { target: 0.10 },
                        OptimizationObjective::MinimizeRisk,
                    ]);
                    
                    let optimized_portfolio = self.optimize_portfolio_quantum(&assets, &constraints, &objectives).await?;
                    
                    Ok(CoordinationResponse {
                        agent_id: self.agent_id,
                        success: true,
                        result: Some(RiskCalculationResult::PortfolioOptimization(optimized_portfolio)),
                        error: None,
                        calculation_time: optimized_portfolio.calculation_time,
                    })
                } else {
                    Ok(CoordinationResponse {
                        agent_id: self.agent_id,
                        success: false,
                        result: None,
                        error: Some("Assets and constraints required for portfolio optimization".to_string()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_portfolio_optimization_agent_creation() {
        let config = PortfolioAgentConfig::default();
        let coordination_hub = Arc::new(RwLock::new(
            AgentCoordinationHub::new(CoordinationConfig::default()).await.unwrap()
        ));
        let message_router = Arc::new(RwLock::new(
            SwarmMessageRouter::new(RoutingConfig::default()).await.unwrap()
        ));
        
        let agent = PortfolioOptimizationAgent::new(config, coordination_hub, message_router).await;
        assert!(agent.is_ok());
    }

    #[tokio::test]
    async fn test_quantum_portfolio_optimization_performance() {
        let config = PortfolioAgentConfig::default();
        let coordination_hub = Arc::new(RwLock::new(
            AgentCoordinationHub::new(CoordinationConfig::default()).await.unwrap()
        ));
        let message_router = Arc::new(RwLock::new(
            SwarmMessageRouter::new(RoutingConfig::default()).await.unwrap()
        ));
        
        let agent = PortfolioOptimizationAgent::new(config, coordination_hub, message_router).await.unwrap();
        
        let assets = vec![Asset::default(); 5];
        let constraints = PortfolioConstraints::default();
        let objectives = vec![
            OptimizationObjective::MaximizeReturn { target: 0.10 },
            OptimizationObjective::MinimizeRisk,
        ];
        
        let start_time = Instant::now();
        let result = agent.optimize_portfolio_quantum(&assets, &constraints, &objectives).await;
        let elapsed = start_time.elapsed();
        
        assert!(result.is_ok());
        assert!(elapsed < Duration::from_micros(100), "Portfolio optimization took {:?}", elapsed);
    }
}