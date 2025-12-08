//! # Stress Testing Agent
//!
//! Monte Carlo simulations with quantum random number generation.
//! This agent implements quantum-enhanced stress testing with
//! ultra-fast scenario generation and sub-100μs simulation targets.

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
use crate::quantum::*;
use crate::stress::*;
use super::base::*;
use super::coordination::*;

/// Stress Testing Agent using quantum Monte Carlo simulations
#[derive(Debug)]
pub struct StressTestingAgent {
    /// Agent metadata
    pub agent_id: Uuid,
    pub agent_type: AgentType,
    pub status: AgentStatus,
    
    /// Configuration
    pub config: StressAgentConfig,
    
    /// Quantum Monte Carlo engine
    pub quantum_monte_carlo: Arc<RwLock<QuantumMonteCarloEngine>>,
    
    /// Scenario generator
    pub scenario_generator: Arc<RwLock<StressScenarioGenerator>>,
    
    /// Stress test executor
    pub stress_executor: Arc<RwLock<StressTestExecutor>>,
    
    /// Historical scenario analyzer
    pub historical_analyzer: Arc<RwLock<HistoricalScenarioAnalyzer>>,
    
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

impl StressTestingAgent {
    /// Create new stress testing agent
    pub async fn new(
        config: StressAgentConfig,
        coordination_hub: Arc<RwLock<AgentCoordinationHub>>,
        message_router: Arc<RwLock<SwarmMessageRouter>>,
    ) -> Result<Self> {
        let agent_id = Uuid::new_v4();
        info!("Creating Stress Testing Agent {}", agent_id);
        
        // Create quantum Monte Carlo engine
        let quantum_monte_carlo = Arc::new(RwLock::new(
            QuantumMonteCarloEngine::new(config.quantum_config.clone()).await?
        ));
        
        // Create scenario generator
        let scenario_generator = Arc::new(RwLock::new(
            StressScenarioGenerator::new(config.scenario_config.clone()).await?
        ));
        
        // Create stress test executor
        let stress_executor = Arc::new(RwLock::new(
            StressTestExecutor::new(config.execution_config.clone()).await?
        ));
        
        // Create historical scenario analyzer
        let historical_analyzer = Arc::new(RwLock::new(
            HistoricalScenarioAnalyzer::new(config.historical_config.clone()).await?
        ));
        
        // Initialize performance metrics
        let performance_metrics = Arc::new(RwLock::new(
            AgentPerformanceMetrics::new(agent_id, AgentType::StressTesting)
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
            agent_type: AgentType::StressTesting,
            status: AgentStatus::Initializing,
            config,
            quantum_monte_carlo,
            scenario_generator,
            stress_executor,
            historical_analyzer,
            performance_metrics,
            coordination_hub,
            message_router,
            message_tx,
            message_rx,
            tengri_client,
        })
    }
    
    /// Run quantum-enhanced stress tests
    pub async fn run_quantum_stress_tests(
        &self,
        portfolio: &Portfolio,
        stress_scenarios: &[StressScenario],
        simulation_config: &SimulationConfig,
    ) -> Result<QuantumStressTestResults> {
        let start_time = Instant::now();
        
        // Update performance metrics
        {
            let mut metrics = self.performance_metrics.write().await;
            metrics.start_calculation();
        }
        
        // Generate quantum random scenarios
        let quantum_monte_carlo = self.quantum_monte_carlo.read().await;
        let quantum_scenarios = quantum_monte_carlo.generate_quantum_scenarios(
            stress_scenarios,
            simulation_config.num_simulations,
        ).await?;
        
        // Execute stress tests
        let stress_executor = self.stress_executor.read().await;
        let stress_results = stress_executor.execute_quantum_stress_tests(
            portfolio,
            &quantum_scenarios,
            simulation_config,
        ).await?;
        
        // Analyze historical comparisons
        let historical_analyzer = self.historical_analyzer.read().await;
        let historical_comparison = historical_analyzer.compare_with_historical(
            &stress_results,
            stress_scenarios,
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
                "Stress testing took {:?}, exceeding 100μs target for agent {}",
                calculation_time, self.agent_id
            );
        }
        
        let quantum_stress_results = QuantumStressTestResults {
            classical_results: stress_results.classical_results,
            quantum_results: stress_results.quantum_results,
            quantum_advantage: stress_results.quantum_advantage,
            historical_comparison,
            scenario_coverage: self.calculate_scenario_coverage(&quantum_scenarios).await?,
            tail_risk_analysis: self.analyze_tail_risks(&stress_results).await?,
            confidence_intervals: stress_results.confidence_intervals,
            calculation_time,
            timestamp: chrono::Utc::now(),
        };
        
        // Report to TENGRI oversight
        {
            let tengri_client = self.tengri_client.read().await;
            tengri_client.report_stress_test_metrics(
                self.agent_id,
                "quantum_stress_testing",
                calculation_time,
                quantum_stress_results.quantum_advantage,
            ).await?;
        }
        
        Ok(quantum_stress_results)
    }
    
    /// Generate stress scenarios based on market conditions
    pub async fn generate_adaptive_scenarios(
        &self,
        market_conditions: &MarketConditions,
        portfolio: &Portfolio,
        scenario_count: u32,
    ) -> Result<Vec<AdaptiveStressScenario>> {
        let start_time = Instant::now();
        
        // Analyze current market regime
        let scenario_generator = self.scenario_generator.read().await;
        let market_regime = scenario_generator.analyze_market_regime(market_conditions).await?;
        
        // Generate quantum-enhanced scenarios
        let quantum_monte_carlo = self.quantum_monte_carlo.read().await;
        let quantum_scenarios = quantum_monte_carlo.generate_adaptive_scenarios(
            &market_regime,
            portfolio,
            scenario_count,
        ).await?;
        
        let calculation_time = start_time.elapsed();
        
        // Convert to adaptive scenarios with metadata
        let adaptive_scenarios: Vec<AdaptiveStressScenario> = quantum_scenarios
            .into_iter()
            .enumerate()
            .map(|(i, scenario)| AdaptiveStressScenario {
                base_scenario: scenario,
                adaptation_reason: format!("Market regime adaptation {}", i),
                probability: 1.0 / scenario_count as f64,
                severity_level: self.calculate_severity_level(&scenario).unwrap_or(SeverityLevel::Medium),
                market_regime: market_regime.clone(),
            })
            .collect();
        
        Ok(adaptive_scenarios)
    }
    
    /// Run tail risk analysis
    pub async fn analyze_tail_risk(
        &self,
        portfolio: &Portfolio,
        confidence_levels: &[f64],
    ) -> Result<TailRiskAnalysis> {
        let start_time = Instant::now();
        
        // Generate extreme tail scenarios
        let quantum_monte_carlo = self.quantum_monte_carlo.read().await;
        let tail_scenarios = quantum_monte_carlo.generate_tail_scenarios(
            portfolio,
            confidence_levels,
        ).await?;
        
        // Execute tail risk calculations
        let stress_executor = self.stress_executor.read().await;
        let tail_results = stress_executor.calculate_tail_risk(
            portfolio,
            &tail_scenarios,
            confidence_levels,
        ).await?;
        
        let calculation_time = start_time.elapsed();
        
        Ok(TailRiskAnalysis {
            tail_scenarios,
            var_estimates: tail_results.var_estimates,
            cvar_estimates: tail_results.cvar_estimates,
            extreme_loss_probabilities: tail_results.extreme_loss_probabilities,
            tail_dependence: tail_results.tail_dependence,
            quantum_tail_enhancement: tail_results.quantum_enhancement,
            calculation_time,
            timestamp: chrono::Utc::now(),
        })
    }
    
    /// Run reverse stress testing
    pub async fn run_reverse_stress_test(
        &self,
        portfolio: &Portfolio,
        target_loss: f64,
    ) -> Result<ReverseStressTestResults> {
        let start_time = Instant::now();
        
        // Find scenarios that lead to target loss
        let quantum_monte_carlo = self.quantum_monte_carlo.read().await;
        let reverse_scenarios = quantum_monte_carlo.find_scenarios_for_loss(
            portfolio,
            target_loss,
        ).await?;
        
        // Analyze scenario plausibility
        let scenario_generator = self.scenario_generator.read().await;
        let plausibility_analysis = scenario_generator.analyze_scenario_plausibility(
            &reverse_scenarios,
        ).await?;
        
        let calculation_time = start_time.elapsed();
        
        Ok(ReverseStressTestResults {
            target_loss,
            identified_scenarios: reverse_scenarios,
            plausibility_analysis,
            scenario_probabilities: self.calculate_scenario_probabilities(&reverse_scenarios).await?,
            risk_factors: self.identify_key_risk_factors(&reverse_scenarios).await?,
            calculation_time,
            timestamp: chrono::Utc::now(),
        })
    }
    
    /// Perform sensitivity analysis
    pub async fn perform_sensitivity_analysis(
        &self,
        portfolio: &Portfolio,
        risk_factors: &[RiskFactor],
        sensitivity_config: &SensitivityConfig,
    ) -> Result<SensitivityAnalysisResults> {
        let start_time = Instant::now();
        
        // Generate sensitivity scenarios
        let scenario_generator = self.scenario_generator.read().await;
        let sensitivity_scenarios = scenario_generator.generate_sensitivity_scenarios(
            risk_factors,
            sensitivity_config,
        ).await?;
        
        // Execute sensitivity tests
        let stress_executor = self.stress_executor.read().await;
        let sensitivity_results = stress_executor.execute_sensitivity_tests(
            portfolio,
            &sensitivity_scenarios,
        ).await?;
        
        let calculation_time = start_time.elapsed();
        
        Ok(SensitivityAnalysisResults {
            risk_factors: risk_factors.to_vec(),
            sensitivity_matrix: sensitivity_results.sensitivity_matrix,
            factor_importance: sensitivity_results.factor_importance,
            interaction_effects: sensitivity_results.interaction_effects,
            marginal_contributions: sensitivity_results.marginal_contributions,
            calculation_time,
            timestamp: chrono::Utc::now(),
        })
    }
    
    /// Calculate scenario coverage
    async fn calculate_scenario_coverage(&self, scenarios: &[QuantumScenario]) -> Result<ScenarioCoverage> {
        let historical_analyzer = self.historical_analyzer.read().await;
        historical_analyzer.calculate_coverage(scenarios).await
    }
    
    /// Analyze tail risks
    async fn analyze_tail_risks(&self, stress_results: &StressTestResults) -> Result<TailRiskMetrics> {
        let extreme_losses = stress_results.quantum_results
            .iter()
            .filter(|result| result.loss > stress_results.classical_results.percentile_95)
            .map(|result| result.loss)
            .collect::<Vec<_>>();
        
        let tail_mean = if extreme_losses.is_empty() {
            0.0
        } else {
            extreme_losses.iter().sum::<f64>() / extreme_losses.len() as f64
        };
        
        let tail_volatility = if extreme_losses.len() > 1 {
            let mean = tail_mean;
            let variance = extreme_losses.iter()
                .map(|loss| (loss - mean).powi(2))
                .sum::<f64>() / (extreme_losses.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };
        
        Ok(TailRiskMetrics {
            tail_mean,
            tail_volatility,
            extreme_loss_count: extreme_losses.len(),
            tail_probability: extreme_losses.len() as f64 / stress_results.quantum_results.len() as f64,
            maximum_loss: extreme_losses.iter().fold(0.0f64, |a, &b| a.max(b)),
        })
    }
    
    /// Calculate severity level for scenario
    fn calculate_severity_level(&self, scenario: &StressScenario) -> Result<SeverityLevel> {
        // Calculate composite severity based on scenario parameters
        let mut severity_score = 0.0;
        
        for shock in &scenario.market_shocks {
            severity_score += shock.magnitude.abs() * shock.probability;
        }
        
        if severity_score > 0.8 {
            Ok(SeverityLevel::Extreme)
        } else if severity_score > 0.5 {
            Ok(SeverityLevel::High)
        } else if severity_score > 0.2 {
            Ok(SeverityLevel::Medium)
        } else {
            Ok(SeverityLevel::Low)
        }
    }
    
    /// Calculate scenario probabilities
    async fn calculate_scenario_probabilities(&self, scenarios: &[StressScenario]) -> Result<Vec<f64>> {
        let historical_analyzer = self.historical_analyzer.read().await;
        historical_analyzer.estimate_probabilities(scenarios).await
    }
    
    /// Identify key risk factors
    async fn identify_key_risk_factors(&self, scenarios: &[StressScenario]) -> Result<Vec<KeyRiskFactor>> {
        let scenario_generator = self.scenario_generator.read().await;
        scenario_generator.identify_key_factors(scenarios).await
    }
    
    /// Handle incoming swarm messages
    async fn handle_message(&self, message: SwarmMessage) -> Result<()> {
        match message.message_type {
            MessageType::StressTestRequest => {
                self.handle_stress_test_request(message).await?;
            }
            MessageType::TailRiskAnalysisRequest => {
                self.handle_tail_risk_request(message).await?;
            }
            MessageType::SensitivityAnalysisRequest => {
                self.handle_sensitivity_analysis_request(message).await?;
            }
            MessageType::ReverseStressTestRequest => {
                self.handle_reverse_stress_test_request(message).await?;
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
    
    async fn handle_stress_test_request(&self, message: SwarmMessage) -> Result<()> {
        if let MessageContent::StressTestRequest { portfolio, scenarios, config } = message.content {
            let stress_results = self.run_quantum_stress_tests(&portfolio, &scenarios, &config).await?;
            
            let response = SwarmMessage {
                id: Uuid::new_v4(),
                sender_id: self.agent_id,
                sender_type: self.agent_type.clone(),
                recipient_id: Some(message.sender_id),
                message_type: MessageType::StressTestResponse,
                content: MessageContent::StressTestResponse(stress_results),
                timestamp: chrono::Utc::now(),
                priority: MessagePriority::High,
                requires_response: false,
            };
            
            self.message_tx.send(response)?;
        }
        Ok(())
    }
    
    async fn handle_tail_risk_request(&self, message: SwarmMessage) -> Result<()> {
        if let MessageContent::TailRiskAnalysisRequest { portfolio, confidence_levels } = message.content {
            let tail_analysis = self.analyze_tail_risk(&portfolio, &confidence_levels).await?;
            
            let response = SwarmMessage {
                id: Uuid::new_v4(),
                sender_id: self.agent_id,
                sender_type: self.agent_type.clone(),
                recipient_id: Some(message.sender_id),
                message_type: MessageType::TailRiskAnalysisResponse,
                content: MessageContent::TailRiskAnalysisResponse(tail_analysis),
                timestamp: chrono::Utc::now(),
                priority: MessagePriority::High,
                requires_response: false,
            };
            
            self.message_tx.send(response)?;
        }
        Ok(())
    }
    
    async fn handle_sensitivity_analysis_request(&self, message: SwarmMessage) -> Result<()> {
        if let MessageContent::SensitivityAnalysisRequest { portfolio, risk_factors, config } = message.content {
            let sensitivity_results = self.perform_sensitivity_analysis(&portfolio, &risk_factors, &config).await?;
            
            let response = SwarmMessage {
                id: Uuid::new_v4(),
                sender_id: self.agent_id,
                sender_type: self.agent_type.clone(),
                recipient_id: Some(message.sender_id),
                message_type: MessageType::SensitivityAnalysisResponse,
                content: MessageContent::SensitivityAnalysisResponse(sensitivity_results),
                timestamp: chrono::Utc::now(),
                priority: MessagePriority::Normal,
                requires_response: false,
            };
            
            self.message_tx.send(response)?;
        }
        Ok(())
    }
    
    async fn handle_reverse_stress_test_request(&self, message: SwarmMessage) -> Result<()> {
        if let MessageContent::ReverseStressTestRequest { portfolio, target_loss } = message.content {
            let reverse_results = self.run_reverse_stress_test(&portfolio, target_loss).await?;
            
            let response = SwarmMessage {
                id: Uuid::new_v4(),
                sender_id: self.agent_id,
                sender_type: self.agent_type.clone(),
                recipient_id: Some(message.sender_id),
                message_type: MessageType::ReverseStressTestResponse,
                content: MessageContent::ReverseStressTestResponse(reverse_results),
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
impl SwarmAgent for StressTestingAgent {
    async fn start(&mut self) -> Result<()> {
        info!("Starting Stress Testing Agent {}", self.agent_id);
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
        info!("Stress Testing Agent {} started successfully", self.agent_id);
        Ok(())
    }
    
    async fn stop(&mut self) -> Result<()> {
        info!("Stopping Stress Testing Agent {}", self.agent_id);
        self.status = AgentStatus::Stopping;
        
        self.status = AgentStatus::Stopped;
        info!("Stress Testing Agent {} stopped successfully", self.agent_id);
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
            CoordinationMessageType::StressTest => {
                if let (Some(portfolio), Some(scenarios)) = (message.portfolio, message.stress_scenarios) {
                    let config = message.simulation_config.unwrap_or_default();
                    
                    let stress_results = self.run_quantum_stress_tests(&portfolio, &scenarios, &config).await?;
                    
                    Ok(CoordinationResponse {
                        agent_id: self.agent_id,
                        success: true,
                        result: Some(RiskCalculationResult::StressTest(stress_results)),
                        error: None,
                        calculation_time: stress_results.calculation_time,
                    })
                } else {
                    Ok(CoordinationResponse {
                        agent_id: self.agent_id,
                        success: false,
                        result: None,
                        error: Some("Portfolio and stress scenarios required for stress testing".to_string()),
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
    async fn test_stress_testing_agent_creation() {
        let config = StressAgentConfig::default();
        let coordination_hub = Arc::new(RwLock::new(
            AgentCoordinationHub::new(CoordinationConfig::default()).await.unwrap()
        ));
        let message_router = Arc::new(RwLock::new(
            SwarmMessageRouter::new(RoutingConfig::default()).await.unwrap()
        ));
        
        let agent = StressTestingAgent::new(config, coordination_hub, message_router).await;
        assert!(agent.is_ok());
    }

    #[tokio::test]
    async fn test_quantum_stress_testing_performance() {
        let config = StressAgentConfig::default();
        let coordination_hub = Arc::new(RwLock::new(
            AgentCoordinationHub::new(CoordinationConfig::default()).await.unwrap()
        ));
        let message_router = Arc::new(RwLock::new(
            SwarmMessageRouter::new(RoutingConfig::default()).await.unwrap()
        ));
        
        let agent = StressTestingAgent::new(config, coordination_hub, message_router).await.unwrap();
        
        let portfolio = Portfolio::default();
        let scenarios = vec![StressScenario::default()];
        let simulation_config = SimulationConfig::default();
        
        let start_time = Instant::now();
        let result = agent.run_quantum_stress_tests(&portfolio, &scenarios, &simulation_config).await;
        let elapsed = start_time.elapsed();
        
        assert!(result.is_ok());
        assert!(elapsed < Duration::from_micros(100), "Stress testing took {:?}", elapsed);
    }

    #[tokio::test]
    async fn test_tail_risk_analysis_performance() {
        let config = StressAgentConfig::default();
        let coordination_hub = Arc::new(RwLock::new(
            AgentCoordinationHub::new(CoordinationConfig::default()).await.unwrap()
        ));
        let message_router = Arc::new(RwLock::new(
            SwarmMessageRouter::new(RoutingConfig::default()).await.unwrap()
        ));
        
        let agent = StressTestingAgent::new(config, coordination_hub, message_router).await.unwrap();
        
        let portfolio = Portfolio::default();
        let confidence_levels = vec![0.01, 0.05, 0.10];
        
        let start_time = Instant::now();
        let result = agent.analyze_tail_risk(&portfolio, &confidence_levels).await;
        let elapsed = start_time.elapsed();
        
        assert!(result.is_ok());
        assert!(elapsed < Duration::from_micros(100), "Tail risk analysis took {:?}", elapsed);
    }
}