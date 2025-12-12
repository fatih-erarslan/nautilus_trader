//! Main QBMIA Agent implementation
//! 
//! High-performance Rust implementation integrating quantum Nash equilibrium,
//! Machiavellian strategic analysis, and biological memory systems.

use crate::{
    config::Config,
    error::{QBMIAError, Result},
    quantum::{QuantumNashEquilibrium, QuantumNashResult},
    strategy::{MachiavellianFramework, ManipulationDetectionResult, StrategyRecommendation},
    memory::{BiologicalMemory, MemoryStats},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::time::Instant;
use tracing::{info, warn, error, debug};

/// Market data structure for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    /// Market snapshot with current state
    pub snapshot: HashMap<String, serde_json::Value>,
    /// Order flow data
    pub order_flow: Vec<crate::strategy::OrderEvent>,
    /// Price history
    pub price_history: Vec<f64>,
    /// Time series data
    pub time_series: HashMap<String, Vec<f64>>,
    /// Market conditions
    pub conditions: HashMap<String, f64>,
    /// Participant information
    pub participants: Vec<String>,
    /// Competitor data
    pub competitors: HashMap<String, f64>,
}

/// Analysis result from QBMIA agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Timestamp of analysis
    pub timestamp: String,
    /// Agent ID
    pub agent_id: String,
    /// Market snapshot at time of analysis
    pub market_snapshot: HashMap<String, serde_json::Value>,
    /// Results from individual components
    pub component_results: ComponentResults,
    /// Integrated decision
    pub integrated_decision: Option<IntegratedDecision>,
    /// Overall confidence score
    pub confidence: f64,
    /// Execution time in milliseconds
    pub execution_time: f64,
}

/// Results from individual analysis components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentResults {
    /// Quantum Nash equilibrium analysis
    pub quantum_nash: Option<QuantumNashResult>,
    /// Machiavellian manipulation detection
    pub machiavellian: Option<ManipulationDetectionResult>,
    /// Strategic recommendations
    pub strategy: Option<StrategyRecommendation>,
}

/// Integrated decision from all components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegratedDecision {
    /// Recommended action
    pub action: String,
    /// Decision confidence [0, 1]
    pub confidence: f64,
    /// Decision vector weights
    pub decision_vector: Vec<f64>,
    /// Human-readable reasoning
    pub reasoning: String,
}

/// Agent status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStatus {
    /// Agent ID
    pub agent_id: String,
    /// Whether agent is currently running
    pub is_running: bool,
    /// Hardware information
    pub hardware: HashMap<String, String>,
    /// Memory usage statistics
    pub memory_usage: MemoryStats,
    /// Performance metrics
    pub performance: PerformanceMetrics,
    /// Last checkpoint information
    pub last_checkpoint: Option<String>,
}

/// Performance metrics for the agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total number of analyses performed
    pub total_analyses: usize,
    /// Average execution time in milliseconds
    pub average_execution_time: f64,
    /// Success rate [0, 1]
    pub success_rate: f64,
    /// Memory efficiency [0, 1]
    pub memory_efficiency: f64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_analyses: 0,
            average_execution_time: 0.0,
            success_rate: 1.0,
            memory_efficiency: 1.0,
        }
    }
}

/// Main QBMIA Agent
pub struct QBMIAAgent {
    /// Configuration
    config: Config,
    /// Agent ID
    agent_id: String,
    
    /// Core components
    quantum_nash: QuantumNashEquilibrium,
    machiavellian: MachiavellianFramework,
    memory: BiologicalMemory,
    
    /// Execution state
    is_running: bool,
    last_decision: Option<AnalysisResult>,
    
    /// Performance tracking
    performance_metrics: PerformanceMetrics,
    execution_times: Vec<f64>,
}

impl QBMIAAgent {
    /// Create a new QBMIA agent
    pub async fn new(config: Config) -> Result<Self> {
        // Validate configuration
        config.validate()?;
        
        let agent_id = config.agent_id.clone();
        
        info!("Initializing QBMIA Agent {}", agent_id);
        
        // Initialize quantum Nash equilibrium solver
        let quantum_nash = QuantumNashEquilibrium::new(config.quantum.clone()).await
            .map_err(|e| QBMIAError::quantum_simulation(format!("Failed to initialize quantum solver: {}", e)))?;
        
        // Initialize Machiavellian framework
        let machiavellian = MachiavellianFramework::new(config.hardware.clone(), config.strategy.machiavellian_sensitivity)
            .map_err(|e| QBMIAError::strategy(format!("Failed to initialize Machiavellian framework: {}", e)))?;
        
        // Initialize biological memory
        let memory_config = crate::memory::MemoryConfig {
            capacity: config.memory.capacity,
            short_term_size: config.memory.short_term_size,
            episodic_size: config.memory.episodic_size,
            consolidation_rate: config.memory.consolidation_rate,
            recall_threshold: config.memory.recall_threshold,
            attention_enabled: config.memory.attention_enabled,
        };
        let memory = BiologicalMemory::new(memory_config, config.hardware.clone())
            .map_err(|e| QBMIAError::memory(format!("Failed to initialize memory system: {}", e)))?;
        
        info!("QBMIA Agent {} initialized successfully", agent_id);
        
        Ok(Self {
            config,
            agent_id,
            quantum_nash,
            machiavellian,
            memory,
            is_running: false,
            last_decision: None,
            performance_metrics: PerformanceMetrics::default(),
            execution_times: Vec::new(),
        })
    }
    
    /// Perform comprehensive market analysis
    pub async fn analyze_market(&mut self, market_data: MarketData) -> Result<AnalysisResult> {
        let start_time = Instant::now();
        
        debug!("Starting market analysis for agent {}", self.agent_id);
        
        // Create experience data for memory storage
        let mut experience = HashMap::new();
        experience.insert("market_snapshot".to_string(), serde_json::to_value(&market_data.snapshot)?);
        experience.insert("timestamp".to_string(), serde_json::to_value(chrono::Utc::now().to_rfc3339())?);
        
        // Run component analyses in parallel
        let component_results = self.run_component_analyses(&market_data).await?;
        
        // Integrate results
        let integrated_decision = self.integrate_analyses(&component_results, &market_data)?;
        
        // Store experience in memory
        experience.insert("component_results".to_string(), serde_json::to_value(&component_results)?);
        experience.insert("integrated_decision".to_string(), serde_json::to_value(&integrated_decision)?);
        
        self.memory.store_experience(&experience)
            .map_err(|e| QBMIAError::memory(format!("Failed to store experience: {}", e)))?;
        
        // Calculate execution metrics
        let execution_time = start_time.elapsed().as_secs_f64() * 1000.0;
        self.update_performance_metrics(execution_time, true);
        
        let analysis_result = AnalysisResult {
            timestamp: chrono::Utc::now().to_rfc3339(),
            agent_id: self.agent_id.clone(),
            market_snapshot: market_data.snapshot,
            component_results,
            integrated_decision: integrated_decision.clone(),
            confidence: integrated_decision.as_ref().map(|d| d.confidence).unwrap_or(0.0),
            execution_time,
        };
        
        self.last_decision = Some(analysis_result.clone());
        
        info!("Market analysis completed in {:.2}ms", execution_time);
        
        Ok(analysis_result)
    }
    
    /// Run all component analyses in parallel
    async fn run_component_analyses(&mut self, market_data: &MarketData) -> Result<ComponentResults> {
        debug!("Running component analyses");
        
        // Execute analyses sequentially to avoid borrowing conflicts
        let quantum_result = self.run_quantum_nash_analysis(market_data).await?;
        let machiavellian_result = self.run_machiavellian_analysis(market_data).await?;
        
        // Generate strategic recommendation based on results
        let strategy_result = if let Some(ref manip_result) = machiavellian_result {
            Some(self.machiavellian.generate_strategy(manip_result, &market_data.competitors).await
                .map_err(|e| QBMIAError::strategy(format!("Strategy generation failed: {}", e)))?)
        } else {
            None
        };
        
        Ok(ComponentResults {
            quantum_nash: quantum_result,
            machiavellian: machiavellian_result,
            strategy: strategy_result,
        })
    }
    
    /// Run quantum Nash equilibrium analysis
    async fn run_quantum_nash_analysis(&mut self, market_data: &MarketData) -> Result<Option<QuantumNashResult>> {
        debug!("Running quantum Nash equilibrium analysis");
        
        // Extract payoff matrix from market data
        let payoff_matrix = self.extract_payoff_matrix(market_data)?;
        
        // Run quantum Nash equilibrium solver
        match self.quantum_nash.find_equilibrium(&payoff_matrix, Some(market_data.conditions.clone())).await {
            Ok(result) => {
                debug!("Quantum Nash analysis completed with convergence score: {:.3}", result.convergence_score);
                Ok(Some(result))
            }
            Err(e) => {
                warn!("Quantum Nash analysis failed: {}", e);
                Ok(None)
            }
        }
    }
    
    /// Run Machiavellian manipulation detection
    async fn run_machiavellian_analysis(&mut self, market_data: &MarketData) -> Result<Option<ManipulationDetectionResult>> {
        debug!("Running Machiavellian manipulation detection");
        
        match self.machiavellian.detect_manipulation(&market_data.order_flow, &market_data.price_history).await {
            Ok(result) => {
                if result.detected {
                    warn!("Market manipulation detected: {} (confidence: {:.3})", 
                          result.primary_pattern, result.confidence);
                } else {
                    debug!("No market manipulation detected");
                }
                Ok(Some(result))
            }
            Err(e) => {
                warn!("Machiavellian analysis failed: {}", e);
                Ok(None)
            }
        }
    }
    
    /// Extract payoff matrix from market data
    fn extract_payoff_matrix(&self, market_data: &MarketData) -> Result<crate::quantum::GameMatrix> {
        // Simplified payoff matrix extraction
        // In practice, this would be more sophisticated based on actual market structure
        
        let participants = &market_data.participants;
        let actions = ["buy", "sell", "hold", "wait"];
        
        let n_participants = participants.len().max(2);
        let n_actions = actions.len();
        
        // Create random payoff matrix biased by market conditions
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let mut payoff_data = Vec::new();
        let total_size = n_participants * n_participants * n_actions * n_actions;
        
        for _ in 0..total_size {
            payoff_data.push(rng.random_range(-1.0..1.0));
        }
        
        // Apply market trend bias
        if let Some(&trend) = market_data.conditions.get("trend") {
            for i in 0..payoff_data.len() {
                let action_type = (i / n_actions) % n_actions;
                if trend > 0.0 && action_type == 0 {
                    // Favor buying in bullish trend
                    payoff_data[i] += trend * 0.5;
                } else if trend < 0.0 && action_type == 1 {
                    // Favor selling in bearish trend
                    payoff_data[i] += trend.abs() * 0.5;
                }
            }
        }
        
        let shape = [n_participants, n_participants, n_actions, n_actions];
        let payoff_matrix = ndarray::Array4::from_shape_vec(shape, payoff_data)
            .map_err(|e| QBMIAError::numerical(format!("Failed to create payoff matrix: {}", e)))?;
        
        crate::quantum::GameMatrix::new(payoff_matrix)
    }
    
    /// Integrate results from all analysis components
    fn integrate_analyses(&self, results: &ComponentResults, market_data: &MarketData) -> Result<Option<IntegratedDecision>> {
        debug!("Integrating component analysis results");
        
        // Check if we have enough successful analyses
        let successful_components = [
            results.quantum_nash.is_some(),
            results.machiavellian.is_some(),
            results.strategy.is_some(),
        ].iter().filter(|&&x| x).count();
        
        if successful_components < 2 {
            warn!("Insufficient successful analyses for integration: {}", successful_components);
            return Ok(None);
        }
        
        // Generate integrated decision
        let decision = self.generate_decision(results)?;
        Ok(Some(decision))
    }
    
    /// Generate integrated decision from component analyses
    fn generate_decision(&self, results: &ComponentResults) -> Result<IntegratedDecision> {
        // Extract key signals from components
        let mut decision_vector = vec![0.0; 4]; // [buy, sell, hold, wait]
        let mut confidence_weights = Vec::new();
        let mut reasoning_parts = Vec::new();
        
        // Process quantum Nash signal
        if let Some(ref quantum_result) = results.quantum_nash {
            let q_action = quantum_result.optimal_action;
            let q_confidence = quantum_result.convergence_score;
            
            if q_action < decision_vector.len() {
                decision_vector[q_action] += q_confidence;
                confidence_weights.push(q_confidence);
                reasoning_parts.push(format!("Quantum Nash equilibrium suggests action {} with {:.1}% confidence", 
                                           q_action, q_confidence * 100.0));
            }
        }
        
        // Process Machiavellian signal
        if let Some(ref machiavellian_result) = results.machiavellian {
            let manipulation_confidence = machiavellian_result.confidence;
            
            if machiavellian_result.detected {
                // High manipulation detected - be defensive
                decision_vector[2] += manipulation_confidence * 0.8; // Hold
                decision_vector[3] += manipulation_confidence * 0.5; // Wait
                confidence_weights.push(manipulation_confidence * 0.8);
                reasoning_parts.push(format!("Market manipulation detected ({}), recommending defensive stance", 
                                           machiavellian_result.primary_pattern));
            } else {
                // No manipulation - normal confidence
                confidence_weights.push(0.7);
                reasoning_parts.push("No market manipulation detected".to_string());
            }
        }
        
        // Process strategic signal
        if let Some(ref strategy_result) = results.strategy {
            let strategy_confidence = strategy_result.confidence;
            
            match strategy_result.action {
                crate::strategy::StrategyAction::Buy => decision_vector[0] += strategy_confidence,
                crate::strategy::StrategyAction::Sell => decision_vector[1] += strategy_confidence,
                crate::strategy::StrategyAction::Hold => decision_vector[2] += strategy_confidence,
                crate::strategy::StrategyAction::Wait => decision_vector[3] += strategy_confidence,
            }
            
            confidence_weights.push(strategy_confidence);
            reasoning_parts.push(format!("Strategic analysis: {}", strategy_result.reasoning));
        }
        
        // Normalize decision vector
        let total_weight: f64 = decision_vector.iter().sum();
        if total_weight > 1e-8 {
            for weight in &mut decision_vector {
                *weight /= total_weight;
            }
        }
        
        // Select action with highest weight
        let action_idx = decision_vector
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(2); // Default to hold
        
        let actions = ["buy", "sell", "hold", "wait"];
        let action = actions[action_idx].to_string();
        
        // Calculate overall confidence
        let confidence = if confidence_weights.is_empty() {
            0.0
        } else {
            confidence_weights.iter().sum::<f64>() / confidence_weights.len() as f64
        };
        
        // Generate reasoning
        let reasoning = if reasoning_parts.is_empty() {
            "Insufficient data for decision reasoning".to_string()
        } else {
            reasoning_parts.join(" | ")
        };
        
        Ok(IntegratedDecision {
            action,
            confidence,
            decision_vector,
            reasoning,
        })
    }
    
    /// Update performance metrics
    fn update_performance_metrics(&mut self, execution_time: f64, success: bool) {
        self.execution_times.push(execution_time);
        
        // Keep only recent execution times (sliding window)
        if self.execution_times.len() > 1000 {
            self.execution_times.remove(0);
        }
        
        self.performance_metrics.total_analyses += 1;
        
        // Update average execution time
        let total_time: f64 = self.execution_times.iter().sum();
        self.performance_metrics.average_execution_time = total_time / self.execution_times.len() as f64;
        
        // Update success rate (simplified - would track actual outcomes)
        if success {
            let old_successes = self.performance_metrics.success_rate * (self.performance_metrics.total_analyses - 1) as f64;
            self.performance_metrics.success_rate = (old_successes + 1.0) / self.performance_metrics.total_analyses as f64;
        }
        
        // Update memory efficiency
        let memory_stats = self.memory.get_usage_stats();
        self.performance_metrics.memory_efficiency = if memory_stats.capacity_percentage < 80.0 {
            1.0 - (memory_stats.capacity_percentage / 100.0) * 0.5
        } else {
            0.6 - (memory_stats.capacity_percentage - 80.0) / 100.0
        };
    }
    
    /// Get current agent status
    pub fn get_status(&self) -> AgentStatus {
        let mut hardware = HashMap::new();
        hardware.insert("simd_enabled".to_string(), cfg!(feature = "simd").to_string());
        hardware.insert("parallel_enabled".to_string(), cfg!(feature = "parallel").to_string());
        hardware.insert("max_workers".to_string(), self.config.hardware.max_workers.to_string());
        
        AgentStatus {
            agent_id: self.agent_id.clone(),
            is_running: self.is_running,
            hardware,
            memory_usage: self.memory.get_usage_stats(),
            performance: self.performance_metrics.clone(),
            last_checkpoint: None, // Would implement checkpoint tracking
        }
    }
    
    /// Start the agent
    pub fn start(&mut self) {
        info!("Starting QBMIA Agent {}", self.agent_id);
        self.is_running = true;
    }
    
    /// Stop the agent
    pub fn stop(&mut self) {
        info!("Stopping QBMIA Agent {}", self.agent_id);
        self.is_running = false;
    }
    
    /// Get agent ID
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }
    
    /// Check if agent is running
    pub fn is_running(&self) -> bool {
        self.is_running
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    
    #[tokio::test]
    async fn test_agent_creation() {
        let config = Config::default();
        let agent = QBMIAAgent::new(config).await;
        assert!(agent.is_ok());
    }
    
    #[tokio::test]
    async fn test_agent_status() {
        let config = Config::default();
        let agent = QBMIAAgent::new(config).await.unwrap();
        
        let status = agent.get_status();
        assert_eq!(status.agent_id, "QBMIA_RUST_001");
        assert!(!status.is_running);
    }
    
    #[tokio::test]
    async fn test_market_analysis() {
        let config = Config {
            quantum: crate::config::QuantumConfig {
                num_qubits: 4,
                max_iterations: 10,
                ..Default::default()
            },
            ..Default::default()
        };
        
        let mut agent = QBMIAAgent::new(config).await.unwrap();
        
        let market_data = MarketData {
            snapshot: {
                let mut map = HashMap::new();
                map.insert("price".to_string(), serde_json::json!(50000.0));
                map.insert("volume".to_string(), serde_json::json!(1000000.0));
                map
            },
            order_flow: vec![],
            price_history: vec![49000.0, 49500.0, 50000.0],
            time_series: HashMap::new(),
            conditions: {
                let mut map = HashMap::new();
                map.insert("volatility".to_string(), 0.02);
                map.insert("trend".to_string(), 0.1);
                map
            },
            participants: vec!["trader1".to_string(), "trader2".to_string()],
            competitors: HashMap::new(),
        };
        
        let result = agent.analyze_market(market_data).await;
        assert!(result.is_ok());
        
        let analysis = result.unwrap();
        assert!(!analysis.agent_id.is_empty());
        assert!(analysis.execution_time > 0.0);
    }
}