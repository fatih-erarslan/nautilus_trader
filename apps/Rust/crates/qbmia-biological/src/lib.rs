//! QBMIA (Quantum-Biological Market Intuition Agent) Biological Systems
//!
//! This crate implements the biological components of the QBMIA system, including
//! biological memory patterns, neural adaptation, and quantum-biological learning
//! mechanisms for sophisticated market analysis and decision-making.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn, error};

pub mod agent;
pub mod memory;
pub mod hardware;
pub mod quantum_integration;
pub mod strategy;
pub mod performance;
pub mod orchestration;
pub mod biological_patterns;
pub mod neural_adaptation;

/// Main QBMIA biological system
#[derive(Debug)]
pub struct QBMIABiological {
    config: QBMIAConfig,
    agent: Arc<agent::QBMIAAgent>,
    memory: Arc<memory::BiologicalMemory>,
    hardware_optimizer: Arc<hardware::HardwareOptimizer>,
    quantum_integration: Arc<quantum_integration::QuantumIntegration>,
    strategy_manager: Arc<strategy::StrategyManager>,
    performance_tracker: Arc<performance::PerformanceTracker>,
    orchestration_manager: Arc<orchestration::OrchestrationManager>,
    biological_patterns: Arc<biological_patterns::BiologicalPatterns>,
    neural_adaptation: Arc<neural_adaptation::NeuralAdaptation>,
    is_running: Arc<RwLock<bool>>,
}

/// QBMIA configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QBMIAConfig {
    pub agent_id: String,
    pub checkpoint_dir: String,
    pub checkpoint_interval: Duration,
    pub num_qubits: u32,
    pub memory_capacity: usize,
    pub learning_rate: f64,
    pub force_cpu: bool,
    pub enable_profiling: bool,
    pub max_workers: usize,
    pub consolidation_rate: f64,
    pub recall_threshold: f64,
    pub volatility_threshold: f64,
    pub wealth_threshold: f64,
    pub memory_decay: f64,
}

impl Default for QBMIAConfig {
    fn default() -> Self {
        Self {
            agent_id: "QBMIA_001".to_string(),
            checkpoint_dir: "./checkpoints".to_string(),
            checkpoint_interval: Duration::from_secs(300),
            num_qubits: 16,
            memory_capacity: 10000,
            learning_rate: 0.001,
            force_cpu: false,
            enable_profiling: true,
            max_workers: 4,
            consolidation_rate: 0.1,
            recall_threshold: 0.7,
            volatility_threshold: 0.3,
            wealth_threshold: 0.8,
            memory_decay: 0.95,
        }
    }
}

/// Market data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub snapshot: MarketSnapshot,
    pub order_flow: Vec<OrderEvent>,
    pub price_history: Vec<f64>,
    pub conditions: MarketConditions,
    pub participants: Vec<String>,
    pub time_series: HashMap<String, Vec<f64>>,
    pub volatility: HashMap<String, f64>,
    pub crisis_indicators: HashMap<String, f64>,
    pub participant_wealth: HashMap<String, f64>,
    pub market_structure: HashMap<String, serde_json::Value>,
}

/// Market snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSnapshot {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub price: f64,
    pub volume: f64,
    pub volatility: f64,
    pub trend: f64,
    pub liquidity: f64,
    pub spread: f64,
}

/// Order event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderEvent {
    pub id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub side: OrderSide,
    pub price: f64,
    pub size: f64,
    pub cancelled: bool,
    pub order_type: OrderType,
}

/// Order side
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Order type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}

/// Market conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditions {
    pub regime: MarketRegime,
    pub volatility_state: VolatilityState,
    pub liquidity_state: LiquidityState,
    pub trend_strength: f64,
    pub market_stress: f64,
}

/// Market regime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketRegime {
    Bullish,
    Bearish,
    Sideways,
    Volatile,
    Calm,
}

/// Volatility state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VolatilityState {
    Low,
    Medium,
    High,
    Extreme,
}

/// Liquidity state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LiquidityState {
    Abundant,
    Normal,
    Scarce,
    Drought,
}

/// Analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub agent_id: String,
    pub market_snapshot: MarketSnapshot,
    pub component_results: HashMap<String, ComponentResult>,
    pub integrated_decision: Option<IntegratedDecision>,
    pub confidence: f64,
    pub execution_time: Duration,
    pub memory_utilization: f64,
    pub quantum_coherence: f64,
}

/// Component result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentResult {
    pub component_type: String,
    pub result: serde_json::Value,
    pub confidence: f64,
    pub execution_time: Duration,
    pub error: Option<String>,
}

/// Integrated decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegratedDecision {
    pub action: TradingAction,
    pub confidence: f64,
    pub decision_vector: Vec<f64>,
    pub reasoning: String,
    pub risk_assessment: RiskAssessment,
    pub expected_return: f64,
    pub time_horizon: Duration,
}

/// Trading action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradingAction {
    Buy,
    Sell,
    Hold,
    Wait,
}

/// Risk assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub risk_score: f64,
    pub var_estimate: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub risk_factors: Vec<String>,
}

/// Execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub decision: IntegratedDecision,
    pub status: ExecutionStatus,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub execution_time: Duration,
    pub actual_return: Option<f64>,
    pub slippage: Option<f64>,
    pub transaction_cost: Option<f64>,
}

/// Execution status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStatus {
    Pending,
    Executed,
    Failed,
    Cancelled,
}

/// System status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    pub agent_id: String,
    pub is_running: bool,
    pub uptime: Duration,
    pub hardware_info: HashMap<String, serde_json::Value>,
    pub memory_usage: MemoryUsage,
    pub performance_metrics: PerformanceMetrics,
    pub last_checkpoint: Option<chrono::DateTime<chrono::Utc>>,
    pub component_health: HashMap<String, ComponentHealth>,
}

/// Memory usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub short_term_size: usize,
    pub long_term_size: usize,
    pub episodic_size: usize,
    pub capacity_used: f64,
    pub consolidation_rate: f64,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_analyses: u64,
    pub successful_decisions: u64,
    pub failed_decisions: u64,
    pub average_execution_time: Duration,
    pub success_rate: f64,
    pub roi: f64,
    pub sharpe_ratio: f64,
}

/// Component health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub status: HealthStatus,
    pub last_update: chrono::DateTime<chrono::Utc>,
    pub error_count: u64,
    pub performance_score: f64,
}

/// Health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Critical,
    Offline,
}

impl QBMIABiological {
    /// Create new QBMIA biological system
    pub async fn new(config: QBMIAConfig) -> Result<Self> {
        info!("Initializing QBMIA Biological System");
        
        // Initialize hardware optimizer
        let hardware_optimizer = Arc::new(hardware::HardwareOptimizer::new(
            config.force_cpu,
            config.enable_profiling,
        )?);
        
        // Initialize biological memory
        let memory = Arc::new(memory::BiologicalMemory::new(
            config.memory_capacity,
            config.consolidation_rate,
            config.recall_threshold,
            Arc::clone(&hardware_optimizer),
        ).await?);
        
        // Initialize quantum integration
        let quantum_integration = Arc::new(quantum_integration::QuantumIntegration::new(
            config.num_qubits,
            Arc::clone(&hardware_optimizer),
        ).await?);
        
        // Initialize strategy manager
        let strategy_manager = Arc::new(strategy::StrategyManager::new(
            config.volatility_threshold,
            config.wealth_threshold,
            config.memory_decay,
            Arc::clone(&hardware_optimizer),
        ).await?);
        
        // Initialize performance tracker
        let performance_tracker = Arc::new(performance::PerformanceTracker::new().await?);
        
        // Initialize orchestration manager
        let orchestration_manager = Arc::new(orchestration::OrchestrationManager::new(
            config.agent_id.clone(),
            Arc::clone(&hardware_optimizer),
        ).await?);
        
        // Initialize biological patterns
        let biological_patterns = Arc::new(biological_patterns::BiologicalPatterns::new(
            Arc::clone(&memory),
            Arc::clone(&hardware_optimizer),
        ).await?);
        
        // Initialize neural adaptation
        let neural_adaptation = Arc::new(neural_adaptation::NeuralAdaptation::new(
            config.learning_rate,
            Arc::clone(&hardware_optimizer),
        ).await?);
        
        // Initialize main agent
        let agent = Arc::new(agent::QBMIAAgent::new(
            config.clone(),
            Arc::clone(&memory),
            Arc::clone(&hardware_optimizer),
            Arc::clone(&quantum_integration),
            Arc::clone(&strategy_manager),
            Arc::clone(&performance_tracker),
            Arc::clone(&orchestration_manager),
            Arc::clone(&biological_patterns),
            Arc::clone(&neural_adaptation),
        ).await?);
        
        let is_running = Arc::new(RwLock::new(false));
        
        Ok(Self {
            config,
            agent,
            memory,
            hardware_optimizer,
            quantum_integration,
            strategy_manager,
            performance_tracker,
            orchestration_manager,
            biological_patterns,
            neural_adaptation,
            is_running,
        })
    }
    
    /// Start the QBMIA biological system
    pub async fn start(&self) -> Result<()> {
        info!("Starting QBMIA Biological System");
        
        // Start all components
        self.hardware_optimizer.start().await?;
        self.memory.start().await?;
        self.quantum_integration.start().await?;
        self.strategy_manager.start().await?;
        self.performance_tracker.start().await?;
        self.orchestration_manager.start().await?;
        self.biological_patterns.start().await?;
        self.neural_adaptation.start().await?;
        self.agent.start().await?;
        
        // Mark as running
        *self.is_running.write().await = true;
        
        info!("QBMIA Biological System started successfully");
        Ok(())
    }
    
    /// Stop the QBMIA biological system
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping QBMIA Biological System");
        
        // Mark as not running
        *self.is_running.write().await = false;
        
        // Stop all components in reverse order
        self.agent.stop().await?;
        self.neural_adaptation.stop().await?;
        self.biological_patterns.stop().await?;
        self.orchestration_manager.stop().await?;
        self.performance_tracker.stop().await?;
        self.strategy_manager.stop().await?;
        self.quantum_integration.stop().await?;
        self.memory.stop().await?;
        self.hardware_optimizer.stop().await?;
        
        info!("QBMIA Biological System stopped successfully");
        Ok(())
    }
    
    /// Analyze market data
    pub async fn analyze_market(&self, market_data: MarketData) -> Result<AnalysisResult> {
        let start_time = Instant::now();
        
        // Ensure system is running
        if !*self.is_running.read().await {
            return Err(anyhow::anyhow!("QBMIA system is not running"));
        }
        
        // Delegate to agent
        let result = self.agent.analyze_market(market_data).await?;
        
        // Track performance
        self.performance_tracker.record_analysis(
            start_time.elapsed(),
            result.confidence,
            result.integrated_decision.is_some(),
        ).await?;
        
        Ok(result)
    }
    
    /// Execute trading decision
    pub async fn execute_decision(&self, decision: IntegratedDecision) -> Result<ExecutionResult> {
        let start_time = Instant::now();
        
        // Ensure system is running
        if !*self.is_running.read().await {
            return Err(anyhow::anyhow!("QBMIA system is not running"));
        }
        
        // Delegate to agent
        let result = self.agent.execute_decision(decision).await?;
        
        // Track execution
        self.performance_tracker.record_execution(
            start_time.elapsed(),
            matches!(result.status, ExecutionStatus::Executed),
        ).await?;
        
        Ok(result)
    }
    
    /// Get system status
    pub async fn get_status(&self) -> Result<SystemStatus> {
        let start_time = Instant::now();
        
        // Collect status from all components
        let mut component_health = HashMap::new();
        
        component_health.insert("agent".to_string(), self.agent.health_check().await?);
        component_health.insert("memory".to_string(), self.memory.health_check().await?);
        component_health.insert("hardware".to_string(), self.hardware_optimizer.health_check().await?);
        component_health.insert("quantum".to_string(), self.quantum_integration.health_check().await?);
        component_health.insert("strategy".to_string(), self.strategy_manager.health_check().await?);
        component_health.insert("performance".to_string(), self.performance_tracker.health_check().await?);
        component_health.insert("orchestration".to_string(), self.orchestration_manager.health_check().await?);
        component_health.insert("biological".to_string(), self.biological_patterns.health_check().await?);
        component_health.insert("neural".to_string(), self.neural_adaptation.health_check().await?);
        
        let status = SystemStatus {
            agent_id: self.config.agent_id.clone(),
            is_running: *self.is_running.read().await,
            uptime: start_time.elapsed(),
            hardware_info: self.hardware_optimizer.get_device_info().await?,
            memory_usage: self.memory.get_usage_stats().await?,
            performance_metrics: self.performance_tracker.get_metrics().await?,
            last_checkpoint: self.agent.get_last_checkpoint().await?,
            component_health,
        };
        
        Ok(status)
    }
    
    /// Save system state
    pub async fn save_state(&self, filepath: Option<String>) -> Result<String> {
        self.agent.save_state(filepath).await
    }
    
    /// Load system state
    pub async fn load_state(&self, filepath: String) -> Result<()> {
        self.agent.load_state(filepath).await
    }
    
    /// Get performance metrics
    pub async fn get_performance_metrics(&self) -> Result<PerformanceMetrics> {
        self.performance_tracker.get_metrics().await
    }
    
    /// Get memory usage
    pub async fn get_memory_usage(&self) -> Result<MemoryUsage> {
        self.memory.get_usage_stats().await
    }
    
    /// Configure biological patterns
    pub async fn configure_biological_patterns(&self, config: biological_patterns::BiologicalConfig) -> Result<()> {
        self.biological_patterns.configure(config).await
    }
    
    /// Configure neural adaptation
    pub async fn configure_neural_adaptation(&self, config: neural_adaptation::NeuralConfig) -> Result<()> {
        self.neural_adaptation.configure(config).await
    }
    
    /// Get quantum coherence
    pub async fn get_quantum_coherence(&self) -> Result<f64> {
        self.quantum_integration.get_coherence().await
    }
    
    /// Get strategy effectiveness
    pub async fn get_strategy_effectiveness(&self) -> Result<HashMap<String, f64>> {
        self.strategy_manager.get_effectiveness().await
    }
    
    /// Health check
    pub async fn health_check(&self) -> Result<ComponentHealth> {
        let is_healthy = *self.is_running.read().await;
        let performance_score = self.performance_tracker.get_overall_score().await?;
        
        Ok(ComponentHealth {
            status: if is_healthy && performance_score > 0.7 {
                HealthStatus::Healthy
            } else if is_healthy && performance_score > 0.5 {
                HealthStatus::Degraded
            } else if is_healthy {
                HealthStatus::Critical
            } else {
                HealthStatus::Offline
            },
            last_update: chrono::Utc::now(),
            error_count: 0, // Would track actual errors in production
            performance_score,
        })
    }
}

/// QBMIA biological trait
#[async_trait]
pub trait QBMIABiologicalTrait {
    async fn analyze_market(&self, market_data: MarketData) -> Result<AnalysisResult>;
    async fn execute_decision(&self, decision: IntegratedDecision) -> Result<ExecutionResult>;
    async fn get_status(&self) -> Result<SystemStatus>;
    async fn health_check(&self) -> Result<ComponentHealth>;
}

#[async_trait]
impl QBMIABiologicalTrait for QBMIABiological {
    async fn analyze_market(&self, market_data: MarketData) -> Result<AnalysisResult> {
        self.analyze_market(market_data).await
    }
    
    async fn execute_decision(&self, decision: IntegratedDecision) -> Result<ExecutionResult> {
        self.execute_decision(decision).await
    }
    
    async fn get_status(&self) -> Result<SystemStatus> {
        self.get_status().await
    }
    
    async fn health_check(&self) -> Result<ComponentHealth> {
        self.health_check().await
    }
}

/// Initialize QBMIA biological system from configuration
pub async fn initialize_qbmia_biological(config_path: &str) -> Result<QBMIABiological> {
    let config = QBMIAConfig::load_from_file(config_path)?;
    QBMIABiological::new(config).await
}

/// QBMIA biological builder
pub struct QBMIABiologicalBuilder {
    config: QBMIAConfig,
}

impl QBMIABiologicalBuilder {
    pub fn new() -> Self {
        Self {
            config: QBMIAConfig::default(),
        }
    }
    
    pub fn with_agent_id(mut self, agent_id: String) -> Self {
        self.config.agent_id = agent_id;
        self
    }
    
    pub fn with_memory_capacity(mut self, capacity: usize) -> Self {
        self.config.memory_capacity = capacity;
        self
    }
    
    pub fn with_num_qubits(mut self, num_qubits: u32) -> Self {
        self.config.num_qubits = num_qubits;
        self
    }
    
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }
    
    pub fn with_force_cpu(mut self, force_cpu: bool) -> Self {
        self.config.force_cpu = force_cpu;
        self
    }
    
    pub fn with_enable_profiling(mut self, enable_profiling: bool) -> Self {
        self.config.enable_profiling = enable_profiling;
        self
    }
    
    pub async fn build(self) -> Result<QBMIABiological> {
        QBMIABiological::new(self.config).await
    }
}

impl Default for QBMIABiologicalBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl QBMIAConfig {
    /// Load configuration from file
    pub fn load_from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn save_to_file(&self, path: &str) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::timeout;
    
    #[tokio::test]
    async fn test_qbmia_biological_initialization() {
        let config = QBMIAConfig::default();
        let result = QBMIABiological::new(config).await;
        
        assert!(result.is_ok());
        let qbmia = result.unwrap();
        assert!(!*qbmia.is_running.read().await);
    }
    
    #[tokio::test]
    async fn test_qbmia_biological_start_stop() {
        let config = QBMIAConfig::default();
        let qbmia = QBMIABiological::new(config).await.unwrap();
        
        // Test start
        let start_result = qbmia.start().await;
        assert!(start_result.is_ok());
        assert!(*qbmia.is_running.read().await);
        
        // Test stop
        let stop_result = qbmia.stop().await;
        assert!(stop_result.is_ok());
        assert!(!*qbmia.is_running.read().await);
    }
    
    #[tokio::test]
    async fn test_builder_pattern() {
        let builder = QBMIABiologicalBuilder::new()
            .with_agent_id("TEST_AGENT".to_string())
            .with_memory_capacity(5000)
            .with_num_qubits(8)
            .with_learning_rate(0.01)
            .with_force_cpu(true);
        
        let qbmia = builder.build().await.unwrap();
        assert_eq!(qbmia.config.agent_id, "TEST_AGENT");
        assert_eq!(qbmia.config.memory_capacity, 5000);
        assert_eq!(qbmia.config.num_qubits, 8);
        assert_eq!(qbmia.config.learning_rate, 0.01);
        assert!(qbmia.config.force_cpu);
    }
    
    #[tokio::test]
    async fn test_market_analysis_without_running() {
        let config = QBMIAConfig::default();
        let qbmia = QBMIABiological::new(config).await.unwrap();
        
        let market_data = create_test_market_data();
        let result = qbmia.analyze_market(market_data).await;
        
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not running"));
    }
    
    #[tokio::test]
    async fn test_health_check() {
        let config = QBMIAConfig::default();
        let qbmia = QBMIABiological::new(config).await.unwrap();
        
        let health = qbmia.health_check().await.unwrap();
        assert!(matches!(health.status, HealthStatus::Offline));
        
        qbmia.start().await.unwrap();
        let health = qbmia.health_check().await.unwrap();
        assert!(matches!(health.status, HealthStatus::Healthy | HealthStatus::Degraded | HealthStatus::Critical));
    }
    
    #[tokio::test]
    async fn test_config_serialization() {
        let config = QBMIAConfig {
            agent_id: "TEST_AGENT".to_string(),
            memory_capacity: 5000,
            num_qubits: 8,
            ..Default::default()
        };
        
        let temp_path = "/tmp/test_qbmia_config.toml";
        config.save_to_file(temp_path).unwrap();
        
        let loaded_config = QBMIAConfig::load_from_file(temp_path).unwrap();
        assert_eq!(loaded_config.agent_id, "TEST_AGENT");
        assert_eq!(loaded_config.memory_capacity, 5000);
        assert_eq!(loaded_config.num_qubits, 8);
        
        // Cleanup
        std::fs::remove_file(temp_path).ok();
    }
    
    fn create_test_market_data() -> MarketData {
        MarketData {
            snapshot: MarketSnapshot {
                timestamp: chrono::Utc::now(),
                price: 100.0,
                volume: 1000.0,
                volatility: 0.02,
                trend: 0.01,
                liquidity: 0.95,
                spread: 0.001,
            },
            order_flow: vec![],
            price_history: vec![99.0, 99.5, 100.0],
            conditions: MarketConditions {
                regime: MarketRegime::Sideways,
                volatility_state: VolatilityState::Low,
                liquidity_state: LiquidityState::Normal,
                trend_strength: 0.3,
                market_stress: 0.1,
            },
            participants: vec!["participant1".to_string()],
            time_series: HashMap::new(),
            volatility: HashMap::new(),
            crisis_indicators: HashMap::new(),
            participant_wealth: HashMap::new(),
            market_structure: HashMap::new(),
        }
    }
}