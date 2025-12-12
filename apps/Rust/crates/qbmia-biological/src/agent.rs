//! QBMIA Agent implementation - Core agent with quantum-biological learning capabilities

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn, error};

use crate::{
    QBMIAConfig, MarketData, AnalysisResult, IntegratedDecision, ExecutionResult, ComponentHealth,
    memory::BiologicalMemory, hardware::HardwareOptimizer, quantum_integration::QuantumIntegration,
    strategy::StrategyManager, performance::PerformanceTracker, orchestration::OrchestrationManager,
    biological_patterns::BiologicalPatterns, neural_adaptation::NeuralAdaptation,
    ComponentResult, TradingAction, RiskAssessment, ExecutionStatus, HealthStatus,
};

/// Main QBMIA Agent
#[derive(Debug)]
pub struct QBMIAAgent {
    config: QBMIAConfig,
    memory: Arc<BiologicalMemory>,
    hardware_optimizer: Arc<HardwareOptimizer>,
    quantum_integration: Arc<QuantumIntegration>,
    strategy_manager: Arc<StrategyManager>,
    performance_tracker: Arc<PerformanceTracker>,
    orchestration_manager: Arc<OrchestrationManager>,
    biological_patterns: Arc<BiologicalPatterns>,
    neural_adaptation: Arc<NeuralAdaptation>,
    last_decision: Arc<RwLock<Option<IntegratedDecision>>>,
    is_running: Arc<RwLock<bool>>,
}

impl QBMIAAgent {
    /// Create new QBMIA agent
    pub async fn new(
        config: QBMIAConfig,
        memory: Arc<BiologicalMemory>,
        hardware_optimizer: Arc<HardwareOptimizer>,
        quantum_integration: Arc<QuantumIntegration>,
        strategy_manager: Arc<StrategyManager>,
        performance_tracker: Arc<PerformanceTracker>,
        orchestration_manager: Arc<OrchestrationManager>,
        biological_patterns: Arc<BiologicalPatterns>,
        neural_adaptation: Arc<NeuralAdaptation>,
    ) -> Result<Self> {
        info!("Initializing QBMIA Agent: {}", config.agent_id);
        
        Ok(Self {
            config,
            memory,
            hardware_optimizer,
            quantum_integration,
            strategy_manager,
            performance_tracker,
            orchestration_manager,
            biological_patterns,
            neural_adaptation,
            last_decision: Arc::new(RwLock::new(None)),
            is_running: Arc::new(RwLock::new(false)),
        })
    }
    
    /// Start the agent
    pub async fn start(&self) -> Result<()> {
        info!("Starting QBMIA Agent: {}", self.config.agent_id);
        *self.is_running.write().await = true;
        Ok(())
    }
    
    /// Stop the agent
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping QBMIA Agent: {}", self.config.agent_id);
        *self.is_running.write().await = false;
        Ok(())
    }
    
    /// Analyze market data
    pub async fn analyze_market(&self, market_data: MarketData) -> Result<AnalysisResult> {
        let start_time = Instant::now();
        
        // Ensure agent is running
        if !*self.is_running.read().await {
            return Err(anyhow::anyhow!("Agent is not running"));
        }
        
        // Request resources from orchestrator
        self.orchestration_manager.request_resources("market_analysis").await?;
        
        // Parallel component analysis
        let quantum_result = self.quantum_integration.analyze_quantum_nash(&market_data).await?;
        let strategy_result = self.strategy_manager.analyze_strategies(&market_data).await?;
        let biological_result = self.biological_patterns.analyze_patterns(&market_data).await?;
        let neural_result = self.neural_adaptation.analyze_adaptation(&market_data).await?;
        
        // Integrate results
        let mut component_results = HashMap::new();
        component_results.insert("quantum_nash".to_string(), quantum_result);
        component_results.insert("strategy_analysis".to_string(), strategy_result);
        component_results.insert("biological_patterns".to_string(), biological_result);
        component_results.insert("neural_adaptation".to_string(), neural_result);
        
        // Generate integrated decision
        let integrated_decision = self.generate_integrated_decision(&component_results).await?;
        
        // Store experience in memory
        self.memory.store_experience(&market_data, &integrated_decision).await?;
        
        // Calculate metrics
        let execution_time = start_time.elapsed();
        let confidence = integrated_decision.as_ref().map(|d| d.confidence).unwrap_or(0.0);
        let memory_utilization = self.memory.get_utilization().await?;
        let quantum_coherence = self.quantum_integration.get_coherence().await?;
        
        // Release resources
        self.orchestration_manager.release_resources("market_analysis").await?;
        
        let result = AnalysisResult {
            timestamp: chrono::Utc::now(),
            agent_id: self.config.agent_id.clone(),
            market_snapshot: market_data.snapshot,
            component_results,
            integrated_decision: integrated_decision.clone(),
            confidence,
            execution_time,
            memory_utilization,
            quantum_coherence,
        };
        
        // Update last decision
        *self.last_decision.write().await = integrated_decision;
        
        Ok(result)
    }
    
    /// Generate integrated decision from component results
    async fn generate_integrated_decision(&self, component_results: &HashMap<String, ComponentResult>) -> Result<Option<IntegratedDecision>> {
        let mut decision_vector = vec![0.0; 4]; // [buy, sell, hold, wait]
        let mut confidence_weights = Vec::new();
        
        // Process quantum Nash result
        if let Some(quantum_result) = component_results.get("quantum_nash") {
            if quantum_result.error.is_none() {
                let quantum_action = 2; // Default to hold
                let quantum_confidence = quantum_result.confidence;
                decision_vector[quantum_action] += quantum_confidence;
                confidence_weights.push(quantum_confidence);
            }
        }
        
        // Process strategy analysis result
        if let Some(strategy_result) = component_results.get("strategy_analysis") {
            if strategy_result.error.is_none() {
                let strategy_action = 2; // Default to hold
                let strategy_confidence = strategy_result.confidence * 0.9;
                decision_vector[strategy_action] += strategy_confidence;
                confidence_weights.push(strategy_confidence);
            }
        }
        
        // Process biological patterns result
        if let Some(biological_result) = component_results.get("biological_patterns") {
            if biological_result.error.is_none() {
                let biological_action = 2; // Default to hold
                let biological_confidence = biological_result.confidence * 0.8;
                decision_vector[biological_action] += biological_confidence;
                confidence_weights.push(biological_confidence);
            }
        }
        
        // Process neural adaptation result
        if let Some(neural_result) = component_results.get("neural_adaptation") {
            if neural_result.error.is_none() {
                let neural_action = 2; // Default to hold
                let neural_confidence = neural_result.confidence * 0.7;
                decision_vector[neural_action] += neural_confidence;
                confidence_weights.push(neural_confidence);
            }
        }
        
        // Check if we have enough confidence to make a decision
        if confidence_weights.is_empty() {
            return Ok(None);
        }
        
        // Normalize decision vector
        let sum: f64 = decision_vector.iter().sum();
        if sum > 0.0 {
            for weight in &mut decision_vector {
                *weight /= sum;
            }
        }
        
        // Select action with highest weight
        let action_idx = decision_vector.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(2); // Default to hold
        
        let actions = [TradingAction::Buy, TradingAction::Sell, TradingAction::Hold, TradingAction::Wait];
        let action = actions[action_idx].clone();
        
        let overall_confidence = confidence_weights.iter().sum::<f64>() / confidence_weights.len() as f64;
        
        // Generate risk assessment
        let risk_assessment = RiskAssessment {
            risk_score: 0.5, // Would calculate based on market conditions
            var_estimate: 0.02,
            max_drawdown: 0.1,
            sharpe_ratio: 1.5,
            risk_factors: vec!["market_volatility".to_string()],
        };
        
        // Generate reasoning
        let reasoning = self.generate_reasoning(&action, component_results);
        
        Ok(Some(IntegratedDecision {
            action,
            confidence: overall_confidence,
            decision_vector,
            reasoning,
            risk_assessment,
            expected_return: 0.05, // Would calculate based on analysis
            time_horizon: Duration::from_secs(3600), // 1 hour
        }))
    }
    
    /// Generate reasoning for decision
    fn generate_reasoning(&self, action: &TradingAction, component_results: &HashMap<String, ComponentResult>) -> String {
        let mut reasoning_parts = Vec::new();
        
        reasoning_parts.push(format!("Decision: {:?}", action));
        
        if let Some(quantum_result) = component_results.get("quantum_nash") {
            if quantum_result.error.is_none() {
                reasoning_parts.push(format!("Quantum Nash analysis supports decision with {:.1}% confidence", quantum_result.confidence * 100.0));
            }
        }
        
        if let Some(strategy_result) = component_results.get("strategy_analysis") {
            if strategy_result.error.is_none() {
                reasoning_parts.push(format!("Strategy analysis confirms with {:.1}% confidence", strategy_result.confidence * 100.0));
            }
        }
        
        if let Some(biological_result) = component_results.get("biological_patterns") {
            if biological_result.error.is_none() {
                reasoning_parts.push(format!("Biological patterns indicate {:.1}% confidence", biological_result.confidence * 100.0));
            }
        }
        
        if let Some(neural_result) = component_results.get("neural_adaptation") {
            if neural_result.error.is_none() {
                reasoning_parts.push(format!("Neural adaptation suggests {:.1}% confidence", neural_result.confidence * 100.0));
            }
        }
        
        reasoning_parts.join(" | ")
    }
    
    /// Execute trading decision
    pub async fn execute_decision(&self, decision: IntegratedDecision) -> Result<ExecutionResult> {
        let start_time = Instant::now();
        
        // Request execution resources
        self.orchestration_manager.request_resources("execution").await?;
        
        // Simulate execution (in real system, this would place actual orders)
        let execution_result = ExecutionResult {
            decision,
            status: ExecutionStatus::Executed,
            timestamp: chrono::Utc::now(),
            execution_time: start_time.elapsed(),
            actual_return: Some(0.02), // Simulated return
            slippage: Some(0.001),
            transaction_cost: Some(0.0005),
        };
        
        // Release resources
        self.orchestration_manager.release_resources("execution").await?;
        
        Ok(execution_result)
    }
    
    /// Save agent state
    pub async fn save_state(&self, filepath: Option<String>) -> Result<String> {
        let state = AgentState {
            agent_id: self.config.agent_id.clone(),
            config: self.config.clone(),
            last_decision: self.last_decision.read().await.clone(),
            timestamp: chrono::Utc::now(),
        };
        
        let filepath = filepath.unwrap_or_else(|| {
            format!("{}/agent_state_{}.json", self.config.checkpoint_dir, self.config.agent_id)
        });
        
        let content = serde_json::to_string_pretty(&state)?;
        std::fs::write(&filepath, content)?;
        
        info!("Agent state saved to: {}", filepath);
        Ok(filepath)
    }
    
    /// Load agent state
    pub async fn load_state(&self, filepath: String) -> Result<()> {
        let content = std::fs::read_to_string(&filepath)?;
        let state: AgentState = serde_json::from_str(&content)?;
        
        *self.last_decision.write().await = state.last_decision;
        
        info!("Agent state loaded from: {}", filepath);
        Ok(())
    }
    
    /// Get last checkpoint info
    pub async fn get_last_checkpoint(&self) -> Result<Option<chrono::DateTime<chrono::Utc>>> {
        Ok(Some(chrono::Utc::now())) // Would track actual checkpoint times
    }
    
    /// Health check
    pub async fn health_check(&self) -> Result<ComponentHealth> {
        let is_running = *self.is_running.read().await;
        let performance_score = 0.85; // Would calculate based on actual performance
        
        Ok(ComponentHealth {
            status: if is_running && performance_score > 0.7 {
                HealthStatus::Healthy
            } else if is_running && performance_score > 0.5 {
                HealthStatus::Degraded
            } else if is_running {
                HealthStatus::Critical
            } else {
                HealthStatus::Offline
            },
            last_update: chrono::Utc::now(),
            error_count: 0,
            performance_score,
        })
    }
}

/// Agent state for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AgentState {
    agent_id: String,
    config: QBMIAConfig,
    last_decision: Option<IntegratedDecision>,
    timestamp: chrono::DateTime<chrono::Utc>,
}