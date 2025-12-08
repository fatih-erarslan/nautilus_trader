//! # Liquidity Risk Agent
//!
//! Real-time liquidity assessment with quantum uncertainty bounds.
//! This agent implements quantum-enhanced liquidity risk modeling with
//! ultra-fast market impact calculations and sub-100μs assessment targets.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2, Array3};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info, warn, error};
use uuid::Uuid;

use crate::types::*;
use crate::types::{ExecutionStrategy, ExecutionRiskMetrics};
use crate::quantum::*;
use super::base::*;
use super::coordination::*;

/// Liquidity Risk Agent using quantum uncertainty bounds
#[derive(Debug)]
pub struct LiquidityRiskAgent {
    /// Agent metadata
    pub agent_id: Uuid,
    pub agent_type: AgentType,
    pub status: AgentStatus,
    
    /// Configuration
    pub config: LiquidityAgentConfig,
    
    /// Quantum liquidity estimator
    pub quantum_liquidity_estimator: Arc<RwLock<QuantumLiquidityEstimator>>,
    
    /// Market impact analyzer
    pub market_impact_analyzer: Arc<RwLock<MarketImpactAnalyzer>>,
    
    /// Liquidity risk calculator
    pub liquidity_risk_calculator: Arc<RwLock<LiquidityRiskCalculator>>,
    
    /// Real-time liquidity monitor
    pub real_time_monitor: Arc<RwLock<RealTimeLiquidityMonitor>>,
    
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

impl LiquidityRiskAgent {
    /// Create new liquidity risk agent
    pub async fn new(
        config: LiquidityAgentConfig,
        coordination_hub: Arc<RwLock<AgentCoordinationHub>>,
        message_router: Arc<RwLock<SwarmMessageRouter>>,
    ) -> Result<Self> {
        let agent_id = Uuid::new_v4();
        info!("Creating Liquidity Risk Agent {}", agent_id);
        
        // Create quantum liquidity estimator
        let quantum_liquidity_estimator = Arc::new(RwLock::new(
            QuantumLiquidityEstimator::new(config.quantum_config.clone()).await?
        ));
        
        // Create market impact analyzer
        let market_impact_analyzer = Arc::new(RwLock::new(
            MarketImpactAnalyzer::new(config.impact_config.clone()).await?
        ));
        
        // Create liquidity risk calculator
        let liquidity_risk_calculator = Arc::new(RwLock::new(
            LiquidityRiskCalculator::new(config.risk_config.clone()).await?
        ));
        
        // Create real-time liquidity monitor
        let real_time_monitor = Arc::new(RwLock::new(
            RealTimeLiquidityMonitor::new(config.monitoring_config.clone()).await?
        ));
        
        // Initialize performance metrics
        let performance_metrics = Arc::new(RwLock::new(
            AgentPerformanceMetrics::new(agent_id, AgentType::LiquidityRisk)
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
            agent_type: AgentType::LiquidityRisk,
            status: AgentStatus::Initializing,
            config,
            quantum_liquidity_estimator,
            market_impact_analyzer,
            liquidity_risk_calculator,
            real_time_monitor,
            performance_metrics,
            coordination_hub,
            message_router,
            message_tx,
            message_rx,
            tengri_client,
        })
    }
    
    /// Assess quantum-enhanced liquidity risk
    pub async fn assess_quantum_liquidity_risk(
        &self,
        portfolio: &Portfolio,
        market_data: &MarketData,
        liquidity_config: &LiquidityAssessmentConfig,
    ) -> Result<QuantumLiquidityRiskAssessment> {
        let start_time = Instant::now();
        
        // Update performance metrics
        {
            let mut metrics = self.performance_metrics.write().await;
            metrics.start_calculation();
        }
        
        // Estimate liquidity for each asset using quantum enhancement
        let quantum_estimator = self.quantum_liquidity_estimator.read().await;
        let asset_liquidity = quantum_estimator.estimate_asset_liquidity(
            &portfolio.assets,
            market_data,
            liquidity_config,
        ).await?;
        
        // Calculate market impact with quantum uncertainty bounds
        let market_impact_analyzer = self.market_impact_analyzer.read().await;
        let market_impact = market_impact_analyzer.calculate_quantum_market_impact(
            portfolio,
            &asset_liquidity,
            liquidity_config,
        ).await?;
        
        // Assess liquidity risk metrics
        let liquidity_risk_calculator = self.liquidity_risk_calculator.read().await;
        let liquidity_risk = liquidity_risk_calculator.calculate_liquidity_risk(
            portfolio,
            &asset_liquidity,
            &market_impact,
        ).await?;
        
        // Calculate funding liquidity risk
        let funding_liquidity_risk = liquidity_risk_calculator.calculate_funding_liquidity_risk(
            portfolio,
            liquidity_config,
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
                "Liquidity risk assessment took {:?}, exceeding 100μs target for agent {}",
                calculation_time, self.agent_id
            );
        }
        
        let liquidity_assessment = QuantumLiquidityRiskAssessment {
            portfolio: portfolio.clone(),
            asset_liquidity,
            market_impact,
            liquidity_risk,
            funding_liquidity_risk,
            quantum_uncertainty_bounds: self.calculate_quantum_uncertainty_bounds(&liquidity_risk).await?,
            liquidity_at_risk: self.calculate_liquidity_at_risk(&liquidity_risk, liquidity_config).await?,
            time_to_liquidation: self.estimate_time_to_liquidation(portfolio, &liquidity_risk).await?,
            calculation_time,
            timestamp: chrono::Utc::now(),
        };
        
        // Report to TENGRI oversight
        {
            let tengri_client = self.tengri_client.read().await;
            tengri_client.report_liquidity_risk_metrics(
                self.agent_id,
                "quantum_liquidity_assessment",
                calculation_time,
                liquidity_assessment.liquidity_risk.overall_risk_score,
            ).await?;
        }
        
        Ok(liquidity_assessment)
    }
    
    /// Monitor real-time liquidity conditions
    pub async fn monitor_real_time_liquidity(
        &self,
        assets: &[Asset],
        streaming_data: &StreamingMarketData,
    ) -> Result<RealTimeLiquidityUpdate> {
        let start_time = Instant::now();
        
        // Update real-time liquidity monitor
        let mut real_time_monitor = self.real_time_monitor.write().await;
        real_time_monitor.update_with_streaming_data(streaming_data).await?;
        
        // Get current liquidity conditions
        let current_liquidity = real_time_monitor.get_current_liquidity_conditions(assets).await?;
        
        // Detect liquidity stress
        let liquidity_stress_signals = real_time_monitor.detect_liquidity_stress().await?;
        
        // Calculate bid-ask spread analysis
        let spread_analysis = real_time_monitor.analyze_bid_ask_spreads(assets).await?;
        
        // Estimate transaction costs
        let transaction_costs = real_time_monitor.estimate_real_time_transaction_costs(assets).await?;
        
        let calculation_time = start_time.elapsed();
        
        // Check real-time performance target (sub-microsecond)
        if calculation_time > Duration::from_micros(1) {
            warn!(
                "Real-time liquidity monitoring took {:?}, exceeding 1μs target for agent {}",
                calculation_time, self.agent_id
            );
        }
        
        let liquidity_update = RealTimeLiquidityUpdate {
            assets: assets.to_vec(),
            current_liquidity,
            liquidity_stress_signals,
            spread_analysis,
            transaction_costs,
            market_depth: real_time_monitor.get_market_depth_analysis().await?,
            update_timestamp: chrono::Utc::now(),
            calculation_time,
        };
        
        // Send alerts for liquidity stress
        if !liquidity_stress_signals.is_empty() {
            self.send_liquidity_stress_alert(&liquidity_update).await?;
        }
        
        Ok(liquidity_update)
    }
    
    /// Calculate market impact for large orders
    pub async fn calculate_market_impact(
        &self,
        trade_orders: &[TradeOrder],
        market_data: &MarketData,
        impact_config: &MarketImpactConfig,
    ) -> Result<MarketImpactAnalysis> {
        let start_time = Instant::now();
        
        // Analyze market microstructure
        let market_impact_analyzer = self.market_impact_analyzer.read().await;
        let microstructure_analysis = market_impact_analyzer.analyze_market_microstructure(
            market_data,
            impact_config,
        ).await?;
        
        // Calculate linear market impact
        let linear_impact = market_impact_analyzer.calculate_linear_impact(
            trade_orders,
            &microstructure_analysis,
        ).await?;
        
        // Calculate non-linear market impact with quantum enhancement
        let quantum_estimator = self.quantum_liquidity_estimator.read().await;
        let nonlinear_impact = quantum_estimator.calculate_nonlinear_quantum_impact(
            trade_orders,
            &microstructure_analysis,
            impact_config,
        ).await?;
        
        // Calculate permanent vs temporary impact
        let impact_decomposition = market_impact_analyzer.decompose_market_impact(
            &linear_impact,
            &nonlinear_impact,
        ).await?;
        
        let calculation_time = start_time.elapsed();
        
        Ok(MarketImpactAnalysis {
            trade_orders: trade_orders.to_vec(),
            microstructure_analysis,
            linear_impact,
            nonlinear_impact,
            impact_decomposition,
            total_impact_cost: self.calculate_total_impact_cost(&impact_decomposition).await?,
            execution_strategy_recommendations: self.generate_execution_recommendations(&impact_decomposition).await?,
            calculation_time,
            timestamp: chrono::Utc::now(),
        })
    }
    
    /// Optimize execution strategy for liquidity management
    pub async fn optimize_execution_strategy(
        &self,
        large_order: &LargeOrder,
        market_conditions: &MarketConditions,
        execution_constraints: &ExecutionConstraints,
    ) -> Result<OptimalExecutionStrategy> {
        let start_time = Instant::now();
        
        // Analyze optimal execution using quantum optimization
        let quantum_estimator = self.quantum_liquidity_estimator.read().await;
        let quantum_execution_plan = quantum_estimator.optimize_quantum_execution(
            large_order,
            market_conditions,
            execution_constraints,
        ).await?;
        
        // Calculate TWAP (Time-Weighted Average Price) strategy
        let market_impact_analyzer = self.market_impact_analyzer.read().await;
        let twap_strategy = market_impact_analyzer.calculate_twap_strategy(
            large_order,
            execution_constraints,
        ).await?;
        
        // Calculate VWAP (Volume-Weighted Average Price) strategy
        let vwap_strategy = market_impact_analyzer.calculate_vwap_strategy(
            large_order,
            market_conditions,
            execution_constraints,
        ).await?;
        
        // Compare strategies and select optimal
        let optimal_strategy = self.select_optimal_strategy(
            &quantum_execution_plan,
            &twap_strategy,
            &vwap_strategy,
            execution_constraints,
        ).await?;
        
        let calculation_time = start_time.elapsed();
        
        Ok(OptimalExecutionStrategy {
            large_order: large_order.clone(),
            quantum_execution_plan,
            twap_strategy,
            vwap_strategy,
            selected_strategy: optimal_strategy,
            expected_execution_cost: self.calculate_expected_execution_cost(&optimal_strategy).await?,
            risk_metrics: self.calculate_execution_risk_metrics(&optimal_strategy).await?,
            calculation_time,
            timestamp: chrono::Utc::now(),
        })
    }
    
    /// Assess funding liquidity risk
    pub async fn assess_funding_liquidity_risk(
        &self,
        portfolio: &Portfolio,
        funding_requirements: &FundingRequirements,
        time_horizon: Duration,
    ) -> Result<FundingLiquidityRiskAssessment> {
        let start_time = Instant::now();
        
        // Calculate cash flow projections
        let liquidity_risk_calculator = self.liquidity_risk_calculator.read().await;
        let cash_flow_projections = liquidity_risk_calculator.project_cash_flows(
            portfolio,
            funding_requirements,
            time_horizon,
        ).await?;
        
        // Assess funding gap analysis
        let funding_gap_analysis = liquidity_risk_calculator.analyze_funding_gaps(
            &cash_flow_projections,
            funding_requirements,
        ).await?;
        
        // Calculate liquidity buffer requirements
        let quantum_estimator = self.quantum_liquidity_estimator.read().await;
        let liquidity_buffer = quantum_estimator.calculate_quantum_liquidity_buffer(
            portfolio,
            &funding_gap_analysis,
            time_horizon,
        ).await?;
        
        // Assess contingent liquidity sources
        let contingent_liquidity = liquidity_risk_calculator.assess_contingent_liquidity(
            portfolio,
            funding_requirements,
        ).await?;
        
        let calculation_time = start_time.elapsed();
        
        Ok(FundingLiquidityRiskAssessment {
            portfolio: portfolio.clone(),
            funding_requirements: funding_requirements.clone(),
            cash_flow_projections,
            funding_gap_analysis,
            liquidity_buffer,
            contingent_liquidity,
            funding_liquidity_ratio: self.calculate_funding_liquidity_ratio(&cash_flow_projections, funding_requirements).await?,
            stress_test_results: self.run_funding_stress_tests(portfolio, funding_requirements).await?,
            calculation_time,
            timestamp: chrono::Utc::now(),
        })
    }
    
    /// Calculate quantum uncertainty bounds for liquidity risk
    async fn calculate_quantum_uncertainty_bounds(
        &self,
        liquidity_risk: &LiquidityRiskMetrics,
    ) -> Result<QuantumUncertaintyBounds> {
        let quantum_estimator = self.quantum_liquidity_estimator.read().await;
        quantum_estimator.calculate_uncertainty_bounds(liquidity_risk).await
    }
    
    /// Calculate Liquidity-at-Risk (LaR)
    async fn calculate_liquidity_at_risk(
        &self,
        liquidity_risk: &LiquidityRiskMetrics,
        config: &LiquidityAssessmentConfig,
    ) -> Result<LiquidityAtRisk> {
        let confidence_levels = &config.confidence_levels;
        let time_horizons = &config.time_horizons;
        
        let mut lar_estimates = HashMap::new();
        
        for &confidence_level in confidence_levels {
            for &time_horizon in time_horizons {
                let lar_value = liquidity_risk.illiquidity_cost * 
                    (1.0 - confidence_level).sqrt() * 
                    time_horizon.as_secs_f64().sqrt();
                
                lar_estimates.insert(
                    (confidence_level, time_horizon),
                    lar_value,
                );
            }
        }
        
        Ok(LiquidityAtRisk {
            estimates: lar_estimates,
            worst_case_scenario: liquidity_risk.worst_case_liquidity_cost,
            expected_liquidity_cost: liquidity_risk.expected_liquidity_cost,
        })
    }
    
    /// Estimate time to liquidation
    async fn estimate_time_to_liquidation(
        &self,
        portfolio: &Portfolio,
        liquidity_risk: &LiquidityRiskMetrics,
    ) -> Result<TimeToLiquidation> {
        let total_portfolio_value = portfolio.total_value();
        let average_daily_volume = liquidity_risk.average_daily_volume;
        
        // Estimate based on participation rate
        let participation_rate = 0.1; // 10% of daily volume
        let daily_liquidation_capacity = average_daily_volume * participation_rate;
        
        let normal_liquidation_days = total_portfolio_value / daily_liquidation_capacity;
        let stressed_liquidation_days = normal_liquidation_days * 2.0; // Assume liquidity halves in stress
        
        Ok(TimeToLiquidation {
            normal_conditions: Duration::from_secs_f64(normal_liquidation_days * 24.0 * 3600.0),
            stressed_conditions: Duration::from_secs_f64(stressed_liquidation_days * 24.0 * 3600.0),
            emergency_liquidation: Duration::from_secs_f64(stressed_liquidation_days * 24.0 * 3600.0 * 0.5),
            participation_rate,
        })
    }
    
    /// Calculate total impact cost
    async fn calculate_total_impact_cost(&self, impact_decomposition: &ImpactDecomposition) -> Result<f64> {
        Ok(impact_decomposition.permanent_impact + impact_decomposition.temporary_impact)
    }
    
    /// Generate execution recommendations
    async fn generate_execution_recommendations(
        &self,
        impact_decomposition: &ImpactDecomposition,
    ) -> Result<Vec<ExecutionRecommendation>> {
        let mut recommendations = Vec::new();
        
        if impact_decomposition.permanent_impact > 0.01 {
            recommendations.push(ExecutionRecommendation {
                strategy_type: ExecutionStrategyType::SlowExecution,
                reason: "High permanent impact detected".to_string(),
                expected_cost_reduction: impact_decomposition.permanent_impact * 0.3,
            });
        }
        
        if impact_decomposition.temporary_impact > 0.005 {
            recommendations.push(ExecutionRecommendation {
                strategy_type: ExecutionStrategyType::SmallBlocks,
                reason: "High temporary impact detected".to_string(),
                expected_cost_reduction: impact_decomposition.temporary_impact * 0.2,
            });
        }
        
        Ok(recommendations)
    }
    
    /// Select optimal execution strategy
    async fn select_optimal_strategy(
        &self,
        quantum_plan: &QuantumExecutionPlan,
        twap_strategy: &TwapStrategy,
        vwap_strategy: &VwapStrategy,
        constraints: &ExecutionConstraints,
    ) -> Result<ExecutionStrategy> {
        // Score each strategy based on cost and risk
        let quantum_score = self.score_execution_strategy(&quantum_plan.execution_schedule, &quantum_plan.expected_cost, constraints).await?;
        let twap_score = self.score_execution_strategy(&twap_strategy.execution_schedule, &twap_strategy.expected_cost, constraints).await?;
        let vwap_score = self.score_execution_strategy(&vwap_strategy.execution_schedule, &vwap_strategy.expected_cost, constraints).await?;
        
        if quantum_score >= twap_score && quantum_score >= vwap_score {
            Ok(ExecutionStrategy::Quantum(quantum_plan.clone()))
        } else if twap_score >= vwap_score {
            Ok(ExecutionStrategy::Twap(twap_strategy.clone()))
        } else {
            Ok(ExecutionStrategy::Vwap(vwap_strategy.clone()))
        }
    }
    
    /// Score execution strategy
    async fn score_execution_strategy(
        &self,
        schedule: &ExecutionSchedule,
        expected_cost: &ExecutionCost,
        constraints: &ExecutionConstraints,
    ) -> Result<f64> {
        let cost_score = 1.0 / (1.0 + expected_cost.total_cost);
        let time_score = if schedule.total_duration <= constraints.max_execution_time {
            1.0
        } else {
            constraints.max_execution_time.as_secs_f64() / schedule.total_duration.as_secs_f64()
        };
        
        Ok(cost_score * 0.7 + time_score * 0.3)
    }
    
    /// Calculate expected execution cost
    async fn calculate_expected_execution_cost(&self, strategy: &ExecutionStrategy) -> Result<ExecutionCost> {
        match strategy {
            ExecutionStrategy::Quantum(plan) => Ok(plan.expected_cost.clone()),
            ExecutionStrategy::Twap(strategy) => Ok(strategy.expected_cost.clone()),
            ExecutionStrategy::Vwap(strategy) => Ok(strategy.expected_cost.clone()),
        }
    }
    
    /// Calculate execution risk metrics
    async fn calculate_execution_risk_metrics(&self, strategy: &ExecutionStrategy) -> Result<ExecutionRiskMetrics> {
        // Simplified risk metrics calculation
        let cost_variance = match strategy {
            ExecutionStrategy::Quantum(plan) => plan.cost_variance,
            ExecutionStrategy::Twap(strategy) => strategy.cost_variance,
            ExecutionStrategy::Vwap(strategy) => strategy.cost_variance,
        };
        
        Ok(ExecutionRiskMetrics {
            cost_variance,
            execution_shortfall_risk: cost_variance.sqrt(),
            timing_risk: cost_variance * 0.5,
            market_impact_risk: cost_variance * 0.3,
        })
    }
    
    /// Calculate funding liquidity ratio
    async fn calculate_funding_liquidity_ratio(
        &self,
        cash_flows: &CashFlowProjections,
        requirements: &FundingRequirements,
    ) -> Result<f64> {
        let total_inflows = cash_flows.projected_inflows.iter().sum::<f64>();
        let total_outflows = cash_flows.projected_outflows.iter().sum::<f64>();
        let net_cash_flow = total_inflows - total_outflows;
        
        if requirements.total_funding_needed > 0.0 {
            Ok(net_cash_flow / requirements.total_funding_needed)
        } else {
            Ok(f64::INFINITY)
        }
    }
    
    /// Run funding stress tests
    async fn run_funding_stress_tests(
        &self,
        portfolio: &Portfolio,
        requirements: &FundingRequirements,
    ) -> Result<FundingStressTestResults> {
        // Simplified stress testing
        let base_funding_gap = requirements.total_funding_needed;
        
        let stress_scenarios = vec![
            ("Market Stress", base_funding_gap * 1.5),
            ("Credit Crunch", base_funding_gap * 2.0),
            ("Extreme Stress", base_funding_gap * 3.0),
        ];
        
        let stress_results: Vec<StressTestResult> = stress_scenarios
            .into_iter()
            .map(|(name, stressed_gap)| StressTestResult {
                scenario_name: name.to_string(),
                stressed_funding_gap: stressed_gap,
                survival_period: if stressed_gap > portfolio.cash_available() {
                    Duration::from_secs(0)
                } else {
                    Duration::from_secs(30 * 24 * 3600) // 30 days
                },
            })
            .collect();
        
        Ok(FundingStressTestResults {
            stress_results,
            worst_case_funding_gap: stress_results.iter()
                .map(|r| r.stressed_funding_gap)
                .fold(0.0f64, f64::max),
            minimum_survival_period: stress_results.iter()
                .map(|r| r.survival_period)
                .min()
                .unwrap_or(Duration::from_secs(0)),
        })
    }
    
    /// Send liquidity stress alert
    async fn send_liquidity_stress_alert(&self, update: &RealTimeLiquidityUpdate) -> Result<()> {
        let alert_message = SwarmMessage {
            id: Uuid::new_v4(),
            sender_id: self.agent_id,
            sender_type: self.agent_type.clone(),
            recipient_id: None, // Broadcast
            message_type: MessageType::LiquidityStressAlert,
            content: MessageContent::LiquidityStressAlert(update.clone()),
            timestamp: chrono::Utc::now(),
            priority: MessagePriority::Critical,
            requires_response: false,
        };
        
        self.message_tx.send(alert_message)?;
        
        // Also report to TENGRI oversight
        {
            let tengri_client = self.tengri_client.read().await;
            tengri_client.report_liquidity_stress_alert(self.agent_id, update).await?;
        }
        
        Ok(())
    }
    
    /// Handle incoming swarm messages
    async fn handle_message(&self, message: SwarmMessage) -> Result<()> {
        match message.message_type {
            MessageType::LiquidityAssessmentRequest => {
                self.handle_liquidity_assessment_request(message).await?;
            }
            MessageType::MarketImpactAnalysisRequest => {
                self.handle_market_impact_request(message).await?;
            }
            MessageType::ExecutionOptimizationRequest => {
                self.handle_execution_optimization_request(message).await?;
            }
            MessageType::FundingLiquidityAssessmentRequest => {
                self.handle_funding_liquidity_request(message).await?;
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
    
    async fn handle_liquidity_assessment_request(&self, message: SwarmMessage) -> Result<()> {
        if let MessageContent::LiquidityAssessmentRequest { portfolio, market_data, config } = message.content {
            let liquidity_assessment = self.assess_quantum_liquidity_risk(&portfolio, &market_data, &config).await?;
            
            let response = SwarmMessage {
                id: Uuid::new_v4(),
                sender_id: self.agent_id,
                sender_type: self.agent_type.clone(),
                recipient_id: Some(message.sender_id),
                message_type: MessageType::LiquidityAssessmentResponse,
                content: MessageContent::LiquidityAssessmentResponse(liquidity_assessment),
                timestamp: chrono::Utc::now(),
                priority: MessagePriority::High,
                requires_response: false,
            };
            
            self.message_tx.send(response)?;
        }
        Ok(())
    }
    
    async fn handle_market_impact_request(&self, message: SwarmMessage) -> Result<()> {
        if let MessageContent::MarketImpactAnalysisRequest { trade_orders, market_data, config } = message.content {
            let market_impact = self.calculate_market_impact(&trade_orders, &market_data, &config).await?;
            
            let response = SwarmMessage {
                id: Uuid::new_v4(),
                sender_id: self.agent_id,
                sender_type: self.agent_type.clone(),
                recipient_id: Some(message.sender_id),
                message_type: MessageType::MarketImpactAnalysisResponse,
                content: MessageContent::MarketImpactAnalysisResponse(market_impact),
                timestamp: chrono::Utc::now(),
                priority: MessagePriority::High,
                requires_response: false,
            };
            
            self.message_tx.send(response)?;
        }
        Ok(())
    }
    
    async fn handle_execution_optimization_request(&self, message: SwarmMessage) -> Result<()> {
        if let MessageContent::ExecutionOptimizationRequest { large_order, market_conditions, constraints } = message.content {
            let execution_strategy = self.optimize_execution_strategy(&large_order, &market_conditions, &constraints).await?;
            
            let response = SwarmMessage {
                id: Uuid::new_v4(),
                sender_id: self.agent_id,
                sender_type: self.agent_type.clone(),
                recipient_id: Some(message.sender_id),
                message_type: MessageType::ExecutionOptimizationResponse,
                content: MessageContent::ExecutionOptimizationResponse(execution_strategy),
                timestamp: chrono::Utc::now(),
                priority: MessagePriority::Normal,
                requires_response: false,
            };
            
            self.message_tx.send(response)?;
        }
        Ok(())
    }
    
    async fn handle_funding_liquidity_request(&self, message: SwarmMessage) -> Result<()> {
        if let MessageContent::FundingLiquidityAssessmentRequest { portfolio, funding_requirements, time_horizon } = message.content {
            let funding_assessment = self.assess_funding_liquidity_risk(&portfolio, &funding_requirements, time_horizon).await?;
            
            let response = SwarmMessage {
                id: Uuid::new_v4(),
                sender_id: self.agent_id,
                sender_type: self.agent_type.clone(),
                recipient_id: Some(message.sender_id),
                message_type: MessageType::FundingLiquidityAssessmentResponse,
                content: MessageContent::FundingLiquidityAssessmentResponse(funding_assessment),
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
impl SwarmAgent for LiquidityRiskAgent {
    async fn start(&mut self) -> Result<()> {
        info!("Starting Liquidity Risk Agent {}", self.agent_id);
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
        info!("Liquidity Risk Agent {} started successfully", self.agent_id);
        Ok(())
    }
    
    async fn stop(&mut self) -> Result<()> {
        info!("Stopping Liquidity Risk Agent {}", self.agent_id);
        self.status = AgentStatus::Stopping;
        
        self.status = AgentStatus::Stopped;
        info!("Liquidity Risk Agent {} stopped successfully", self.agent_id);
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
            CoordinationMessageType::LiquidityAssessment => {
                if let (Some(portfolio), Some(market_data)) = (message.portfolio, message.market_data) {
                    let config = message.liquidity_config.unwrap_or_default();
                    
                    let liquidity_assessment = self.assess_quantum_liquidity_risk(&portfolio, &market_data, &config).await?;
                    
                    Ok(CoordinationResponse {
                        agent_id: self.agent_id,
                        success: true,
                        result: Some(RiskCalculationResult::LiquidityAssessment(liquidity_assessment)),
                        error: None,
                        calculation_time: liquidity_assessment.calculation_time,
                    })
                } else {
                    Ok(CoordinationResponse {
                        agent_id: self.agent_id,
                        success: false,
                        result: None,
                        error: Some("Portfolio and market data required for liquidity assessment".to_string()),
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
    async fn test_liquidity_risk_agent_creation() {
        let config = LiquidityAgentConfig::default();
        let coordination_hub = Arc::new(RwLock::new(
            AgentCoordinationHub::new(CoordinationConfig::default()).await.unwrap()
        ));
        let message_router = Arc::new(RwLock::new(
            SwarmMessageRouter::new(RoutingConfig::default()).await.unwrap()
        ));
        
        let agent = LiquidityRiskAgent::new(config, coordination_hub, message_router).await;
        assert!(agent.is_ok());
    }

    #[tokio::test]
    async fn test_quantum_liquidity_assessment_performance() {
        let config = LiquidityAgentConfig::default();
        let coordination_hub = Arc::new(RwLock::new(
            AgentCoordinationHub::new(CoordinationConfig::default()).await.unwrap()
        ));
        let message_router = Arc::new(RwLock::new(
            SwarmMessageRouter::new(RoutingConfig::default()).await.unwrap()
        ));
        
        let agent = LiquidityRiskAgent::new(config, coordination_hub, message_router).await.unwrap();
        
        let portfolio = Portfolio::default();
        let market_data = MarketData::default();
        let liquidity_config = LiquidityAssessmentConfig::default();
        
        let start_time = Instant::now();
        let result = agent.assess_quantum_liquidity_risk(&portfolio, &market_data, &liquidity_config).await;
        let elapsed = start_time.elapsed();
        
        assert!(result.is_ok());
        assert!(elapsed < Duration::from_micros(100), "Liquidity assessment took {:?}", elapsed);
    }

    #[tokio::test]
    async fn test_real_time_liquidity_monitoring_performance() {
        let config = LiquidityAgentConfig::default();
        let coordination_hub = Arc::new(RwLock::new(
            AgentCoordinationHub::new(CoordinationConfig::default()).await.unwrap()
        ));
        let message_router = Arc::new(RwLock::new(
            SwarmMessageRouter::new(RoutingConfig::default()).await.unwrap()
        ));
        
        let agent = LiquidityRiskAgent::new(config, coordination_hub, message_router).await.unwrap();
        
        let assets = vec![Asset::default(); 3];
        let streaming_data = StreamingMarketData::default();
        
        let start_time = Instant::now();
        let result = agent.monitor_real_time_liquidity(&assets, &streaming_data).await;
        let elapsed = start_time.elapsed();
        
        assert!(result.is_ok());
        assert!(elapsed < Duration::from_micros(1), "Real-time liquidity monitoring took {:?}", elapsed);
    }
}