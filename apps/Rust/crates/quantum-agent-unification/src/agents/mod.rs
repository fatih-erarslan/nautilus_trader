//! Quantum Agent System - Unified Interface for 12 Specialized Agents
//!
//! This module provides the core `QuantumAgent` trait and implementations
//! for all 12 specialized quantum agents used in the trading system.

pub mod strategic;
pub mod tactical;
pub mod risk_management;
pub mod market_microstructure;
pub mod sentiment_analysis;
pub mod pattern_recognition;
pub mod arbitrage_detection;
pub mod volatility_forecasting;
pub mod portfolio_optimization;
pub mod execution_optimization;
pub mod regime_detection;
pub mod adaptive_learning;

// Re-exports
pub use strategic::StrategicQuantumAgent;
pub use tactical::TacticalQuantumAgent;
pub use risk_management::RiskManagementAgent;
pub use market_microstructure::MarketMicrostructureAgent;
pub use sentiment_analysis::SentimentAnalysisAgent;
pub use pattern_recognition::PatternRecognitionAgent;
pub use arbitrage_detection::ArbitrageDetectionAgent;
pub use volatility_forecasting::VolatilityForecastingAgent;
pub use portfolio_optimization::PortfolioOptimizationAgent;
pub use execution_optimization::ExecutionOptimizationAgent;
pub use regime_detection::RegimeDetectionAgent;
pub use adaptive_learning::AdaptiveLearningAgent;

use crate::error::{UnificationError, Result};
use crate::{Float, Vector, Matrix, Point};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use nalgebra::DVector;

/// Unique identifier for quantum agents
pub type AgentId = uuid::Uuid;

/// Core trait for all quantum agents in the unified system
#[async_trait]
pub trait QuantumAgent: Send + Sync + std::fmt::Debug {
    /// Get the unique identifier for this agent
    fn id(&self) -> AgentId;
    
    /// Get the agent type classification
    fn agent_type(&self) -> AgentType;
    
    /// Get the agent's specialization weight in PADS integration
    fn specialization_weight(&self) -> Float;
    
    /// Process market data and generate quantum decisions
    async fn process(&mut self, market_data: &MarketData) -> Result<AgentResult>;
    
    /// Update agent's internal state based on feedback
    async fn update_state(&mut self, feedback: &AgentFeedback) -> Result<()>;
    
    /// Get current agent configuration
    fn config(&self) -> &AgentConfig;
    
    /// Get agent performance metrics
    fn metrics(&self) -> AgentMetrics;
    
    /// Reset agent to initial state
    async fn reset(&mut self) -> Result<()>;
    
    /// Optimize agent parameters using hyperbolic geometry
    async fn optimize_hyperbolic(&mut self, optimization_data: &OptimizationData) -> Result<()>;
    
    /// Coordinate with other agents in the swarm
    async fn coordinate(&mut self, coordination_data: &CoordinationData) -> Result<CoordinationResponse>;
    
    /// Get agent's current quantum state vector
    fn quantum_state(&self) -> Vector;
    
    /// Apply quantum evolution operator
    async fn evolve_quantum_state(&mut self, evolution_operator: &Matrix) -> Result<()>;
    
    /// Check if agent is ready for processing
    fn is_ready(&self) -> bool;
    
    /// Get agent health status
    fn health_status(&self) -> HealthStatus;
}

/// Classification of quantum agent types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentType {
    /// Long-term strategic decision making
    Strategic,
    /// Short-term tactical execution
    Tactical,
    /// Dynamic risk assessment and control
    RiskManagement,
    /// Order book and liquidity analysis
    MarketMicrostructure,
    /// Market sentiment and news processing
    SentimentAnalysis,
    /// Technical pattern identification
    PatternRecognition,
    /// Cross-market opportunity identification
    ArbitrageDetection,
    /// Volatility prediction and modeling
    VolatilityForecasting,
    /// Multi-asset portfolio management
    PortfolioOptimization,
    /// Order execution optimization
    ExecutionOptimization,
    /// Market regime classification
    RegimeDetection,
    /// Continuous model improvement
    AdaptiveLearning,
}

impl AgentType {
    /// Get the default specialization weight for this agent type
    pub fn default_weight(&self) -> Float {
        match self {
            AgentType::Strategic => crate::STRATEGIC_WEIGHT,
            AgentType::Tactical => crate::TACTICAL_WEIGHT,
            AgentType::RiskManagement => crate::RISK_WEIGHT,
            AgentType::MarketMicrostructure => crate::MICROSTRUCTURE_WEIGHT,
            AgentType::SentimentAnalysis => crate::SENTIMENT_WEIGHT,
            AgentType::PatternRecognition => crate::PATTERN_WEIGHT,
            AgentType::ArbitrageDetection => crate::ARBITRAGE_WEIGHT,
            AgentType::VolatilityForecasting => crate::VOLATILITY_WEIGHT,
            AgentType::PortfolioOptimization => crate::PORTFOLIO_WEIGHT,
            AgentType::ExecutionOptimization => crate::EXECUTION_WEIGHT,
            AgentType::RegimeDetection => crate::REGIME_WEIGHT,
            AgentType::AdaptiveLearning => crate::LEARNING_WEIGHT,
        }
    }
    
    /// Get all agent types in order
    pub fn all() -> &'static [AgentType] {
        &[
            AgentType::Strategic,
            AgentType::Tactical,
            AgentType::RiskManagement,
            AgentType::MarketMicrostructure,
            AgentType::SentimentAnalysis,
            AgentType::PatternRecognition,
            AgentType::ArbitrageDetection,
            AgentType::VolatilityForecasting,
            AgentType::PortfolioOptimization,
            AgentType::ExecutionOptimization,
            AgentType::RegimeDetection,
            AgentType::AdaptiveLearning,
        ]
    }
}

impl std::fmt::Display for AgentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentType::Strategic => write!(f, "Strategic"),
            AgentType::Tactical => write!(f, "Tactical"),
            AgentType::RiskManagement => write!(f, "Risk Management"),
            AgentType::MarketMicrostructure => write!(f, "Market Microstructure"),
            AgentType::SentimentAnalysis => write!(f, "Sentiment Analysis"),
            AgentType::PatternRecognition => write!(f, "Pattern Recognition"),
            AgentType::ArbitrageDetection => write!(f, "Arbitrage Detection"),
            AgentType::VolatilityForecasting => write!(f, "Volatility Forecasting"),
            AgentType::PortfolioOptimization => write!(f, "Portfolio Optimization"),
            AgentType::ExecutionOptimization => write!(f, "Execution Optimization"),
            AgentType::RegimeDetection => write!(f, "Regime Detection"),
            AgentType::AdaptiveLearning => write!(f, "Adaptive Learning"),
        }
    }
}

/// Configuration for quantum agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Agent update interval
    pub update_interval: Duration,
    /// Learning rate for adaptive algorithms
    pub learning_rate: Float,
    /// Memory window size for historical data
    pub memory_window: usize,
    /// Confidence threshold for decisions
    pub confidence_threshold: Float,
    /// Enable hyperbolic optimization
    pub enable_hyperbolic_optimization: bool,
    /// Enable PADS integration
    pub enable_pads_integration: bool,
    /// SIMD batch size for vectorized operations
    pub simd_batch_size: usize,
    /// Maximum processing latency (milliseconds)
    pub max_processing_latency_ms: u64,
    /// Agent-specific parameters
    pub specific_params: HashMap<String, Float>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            update_interval: Duration::from_millis(100),
            learning_rate: 0.01,
            memory_window: 1000,
            confidence_threshold: 0.7,
            enable_hyperbolic_optimization: true,
            enable_pads_integration: true,
            simd_batch_size: 32,
            max_processing_latency_ms: 10,
            specific_params: HashMap::new(),
        }
    }
}

/// Market data input for agent processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    /// Timestamp of the data
    pub timestamp: Instant,
    /// Price data (OHLCV)
    pub price_data: PriceData,
    /// Order book snapshot
    pub order_book: OrderBook,
    /// Recent trade data
    pub trades: Vec<Trade>,
    /// Market indicators
    pub indicators: HashMap<String, Float>,
    /// News and sentiment data
    pub sentiment_data: Option<SentimentData>,
    /// Cross-market data for arbitrage
    pub cross_market_data: Option<HashMap<String, PriceData>>,
}

/// Price data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceData {
    pub open: Float,
    pub high: Float,
    pub low: Float,
    pub close: Float,
    pub volume: Float,
    pub timestamp: Instant,
}

/// Order book representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub bids: Vec<(Float, Float)>, // (price, size)
    pub asks: Vec<(Float, Float)>, // (price, size)
    pub timestamp: Instant,
}

/// Trade data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub price: Float,
    pub size: Float,
    pub side: TradeSide,
    pub timestamp: Instant,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Sentiment analysis data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentData {
    pub sentiment_score: Float, // -1.0 to 1.0
    pub confidence: Float,
    pub news_items: Vec<NewsItem>,
    pub social_sentiment: Option<Float>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsItem {
    pub title: String,
    pub content: String,
    pub sentiment: Float,
    pub relevance: Float,
    pub timestamp: Instant,
}

/// Result from agent processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    /// Agent ID
    pub agent_id: AgentId,
    /// Agent type
    pub agent_type: AgentType,
    /// Processing timestamp
    pub timestamp: Instant,
    /// Decision confidence (0.0 to 1.0)
    pub confidence: Float,
    /// Quantum state vector
    pub quantum_state: Vector,
    /// Agent-specific decision data
    pub decision_data: DecisionData,
    /// Performance metrics for this decision
    pub performance_metrics: ProcessingMetrics,
}

/// Agent-specific decision data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionData {
    /// Primary decision signal (-1.0 to 1.0)
    pub signal: Float,
    /// Signal strength (0.0 to 1.0)
    pub strength: Float,
    /// Risk assessment
    pub risk_level: Float,
    /// Expected return
    pub expected_return: Option<Float>,
    /// Time horizon for the decision
    pub time_horizon: Duration,
    /// Supporting evidence
    pub evidence: HashMap<String, Float>,
    /// Hyperbolic coordinates if available
    pub hyperbolic_coords: Option<Point>,
}

/// Processing performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetrics {
    /// Processing time in nanoseconds
    pub processing_time_ns: u64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Number of operations performed
    pub operations_count: u64,
    /// SIMD utilization (0.0 to 1.0)
    pub simd_utilization: Float,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: Float,
}

/// Feedback data for agent learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentFeedback {
    /// Agent ID this feedback is for
    pub target_agent_id: AgentId,
    /// Actual outcome vs prediction
    pub outcome_error: Float,
    /// Performance score (0.0 to 1.0)
    pub performance_score: Float,
    /// Market conditions during the decision
    pub market_conditions: MarketConditions,
    /// Reward signal for reinforcement learning
    pub reward_signal: Float,
    /// Timestamp of the feedback
    pub timestamp: Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditions {
    pub volatility: Float,
    pub liquidity: Float,
    pub trend_strength: Float,
    pub market_regime: String,
}

/// Agent performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetrics {
    /// Total decisions made
    pub total_decisions: u64,
    /// Average confidence
    pub avg_confidence: Float,
    /// Average processing time (nanoseconds)
    pub avg_processing_time_ns: Float,
    /// Success rate (0.0 to 1.0)
    pub success_rate: Float,
    /// Sharpe ratio of decisions
    pub sharpe_ratio: Option<Float>,
    /// Maximum drawdown
    pub max_drawdown: Option<Float>,
    /// Last update timestamp
    pub last_update: Instant,
    /// Agent health score (0.0 to 1.0)
    pub health_score: Float,
}

impl Default for AgentMetrics {
    fn default() -> Self {
        Self {
            total_decisions: 0,
            avg_confidence: 0.0,
            avg_processing_time_ns: 0.0,
            success_rate: 0.0,
            sharpe_ratio: None,
            max_drawdown: None,
            last_update: Instant::now(),
            health_score: 1.0,
        }
    }
}

/// Optimization data for hyperbolic geometry
#[derive(Debug, Clone)]
pub struct OptimizationData {
    /// Historical performance data
    pub performance_history: Vec<Float>,
    /// Market feature vectors
    pub feature_vectors: Vec<Vector>,
    /// Target optimization metric
    pub target_metric: OptimizationMetric,
    /// Optimization parameters
    pub parameters: HashMap<String, Float>,
}

#[derive(Debug, Clone, Copy)]
pub enum OptimizationMetric {
    SharpeRatio,
    MaxDrawdown,
    ProfitFactor,
    WinRate,
    Custom(Float),
}

/// Coordination data for swarm intelligence
#[derive(Debug, Clone)]
pub struct CoordinationData {
    /// Other agents' states
    pub agent_states: HashMap<AgentId, Vector>,
    /// Swarm consensus signal
    pub consensus_signal: Option<Float>,
    /// Coordination parameters
    pub coordination_params: HashMap<String, Float>,
    /// Network topology information
    pub network_topology: NetworkTopology,
}

#[derive(Debug, Clone)]
pub struct NetworkTopology {
    /// Adjacency matrix
    pub adjacency_matrix: Matrix,
    /// Node weights
    pub node_weights: Vector,
    /// Communication latencies
    pub latencies: HashMap<(AgentId, AgentId), Duration>,
}

/// Response to coordination request
#[derive(Debug, Clone)]
pub struct CoordinationResponse {
    /// Agent's updated state
    pub updated_state: Vector,
    /// Contribution to consensus
    pub consensus_contribution: Float,
    /// Communication messages to other agents
    pub messages: Vec<AgentMessage>,
}

#[derive(Debug, Clone)]
pub struct AgentMessage {
    pub from: AgentId,
    pub to: AgentId,
    pub message_type: MessageType,
    pub payload: Vec<u8>,
    pub timestamp: Instant,
}

#[derive(Debug, Clone, Copy)]
pub enum MessageType {
    StateUpdate,
    DecisionShare,
    CoordinationRequest,
    FeedbackShare,
    EmergencySignal,
}

/// Health status of an agent
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Critical,
    Offline,
}

impl HealthStatus {
    /// Get numeric score for health status
    pub fn score(&self) -> Float {
        match self {
            HealthStatus::Healthy => 1.0,
            HealthStatus::Degraded => 0.7,
            HealthStatus::Critical => 0.3,
            HealthStatus::Offline => 0.0,
        }
    }
}

/// Agent factory for creating specialized agents
pub struct AgentFactory;

impl AgentFactory {
    /// Create an agent of the specified type
    pub fn create_agent(agent_type: AgentType, config: AgentConfig) -> Result<Box<dyn QuantumAgent>> {
        match agent_type {
            AgentType::Strategic => Ok(Box::new(StrategicQuantumAgent::new(config)?)),
            AgentType::Tactical => Ok(Box::new(TacticalQuantumAgent::new(config)?)),
            AgentType::RiskManagement => Ok(Box::new(RiskManagementAgent::new(config)?)),
            AgentType::MarketMicrostructure => Ok(Box::new(MarketMicrostructureAgent::new(config)?)),
            AgentType::SentimentAnalysis => Ok(Box::new(SentimentAnalysisAgent::new(config)?)),
            AgentType::PatternRecognition => Ok(Box::new(PatternRecognitionAgent::new(config)?)),
            AgentType::ArbitrageDetection => Ok(Box::new(ArbitrageDetectionAgent::new(config)?)),
            AgentType::VolatilityForecasting => Ok(Box::new(VolatilityForecastingAgent::new(config)?)),
            AgentType::PortfolioOptimization => Ok(Box::new(PortfolioOptimizationAgent::new(config)?)),
            AgentType::ExecutionOptimization => Ok(Box::new(ExecutionOptimizationAgent::new(config)?)),
            AgentType::RegimeDetection => Ok(Box::new(RegimeDetectionAgent::new(config)?)),
            AgentType::AdaptiveLearning => Ok(Box::new(AdaptiveLearningAgent::new(config)?)),
        }
    }
    
    /// Create all 12 standard agents with default configuration
    pub fn create_full_agent_suite() -> Result<Vec<Box<dyn QuantumAgent>>> {
        let mut agents = Vec::new();
        
        for &agent_type in AgentType::all() {
            let config = AgentConfig::default();
            agents.push(Self::create_agent(agent_type, config)?);
        }
        
        Ok(agents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_type_weights() {
        // Verify all weights sum to 1.0
        let total_weight: Float = AgentType::all()
            .iter()
            .map(|t| t.default_weight())
            .sum();
        
        assert!((total_weight - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_agent_type_display() {
        for &agent_type in AgentType::all() {
            let display_str = format!("{}", agent_type);
            assert!(!display_str.is_empty());
        }
    }

    #[test]
    fn test_agent_config_default() {
        let config = AgentConfig::default();
        assert!(config.learning_rate > 0.0);
        assert!(config.memory_window > 0);
        assert!(config.confidence_threshold >= 0.0 && config.confidence_threshold <= 1.0);
    }

    #[test]
    fn test_health_status_scores() {
        assert_eq!(HealthStatus::Healthy.score(), 1.0);
        assert_eq!(HealthStatus::Degraded.score(), 0.7);
        assert_eq!(HealthStatus::Critical.score(), 0.3);
        assert_eq!(HealthStatus::Offline.score(), 0.0);
    }
}