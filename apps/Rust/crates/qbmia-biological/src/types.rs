//! Common types used throughout the QBMIA system

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::time::{Duration, Instant};

/// Market data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub snapshot: MarketSnapshot,
    pub order_flow: Vec<OrderEvent>,
    pub price_history: Vec<f64>,
    pub conditions: HashMap<String, f64>,
    pub participants: Vec<String>,
    pub participant_wealth: HashMap<String, f64>,
    pub competitors: HashMap<String, CompetitorInfo>,
    pub time_series: TimeSeriesData,
    pub volatility: HashMap<String, f64>,
    pub crisis_indicators: HashMap<String, f64>,
    pub market_structure: MarketStructure,
}

/// Market snapshot at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSnapshot {
    pub price: f64,
    pub volume: f64,
    pub volatility: f64,
    pub trend: f64,
    pub bid_ask_spread: f64,
    pub order_book_depth: u32,
    pub liquidity: f64,
}

/// Order event structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderEvent {
    pub timestamp: f64,
    pub price: f64,
    pub size: f64,
    pub side: String, // "buy" or "sell"
    pub order_id: String,
    pub cancelled: bool,
    pub executed: bool,
}

/// Competitor information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetitorInfo {
    pub id: String,
    pub strategy_type: String,
    pub recent_performance: f64,
    pub risk_profile: String,
    pub activity_level: f64,
}

/// Time series data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesData {
    pub prices: Vec<f64>,
    pub volumes: Vec<f64>,
    pub volatility: Vec<f64>,
    pub timestamps: Vec<f64>,
    pub returns: Vec<f64>,
}

/// Market structure information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketStructure {
    pub markets: HashMap<String, MarketInfo>,
    pub correlations: HashMap<String, f64>,
    pub liquidity_providers: Vec<String>,
    pub market_makers: Vec<String>,
}

/// Individual market information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketInfo {
    pub market_id: String,
    pub price: f64,
    pub volume: f64,
    pub bid_ask_spread: f64,
    pub order_book_depth: u32,
    pub volatility: f64,
    pub expected_ratio: f64,
}

/// Decision types for actions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionType {
    Buy = 0,
    Sell = 1,
    Hold = 2,
    Wait = 3,
}

impl From<usize> for ActionType {
    fn from(value: usize) -> Self {
        match value {
            0 => ActionType::Buy,
            1 => ActionType::Sell,
            2 => ActionType::Hold,
            _ => ActionType::Wait,
        }
    }
}

impl From<ActionType> for usize {
    fn from(action: ActionType) -> Self {
        action as usize
    }
}

/// Market decision with confidence and reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDecision {
    pub action: ActionType,
    pub confidence: f64,
    pub decision_vector: Vec<f64>,
    pub reasoning: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub component_contributions: HashMap<String, f64>,
}

/// Analysis results from individual components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentAnalysis {
    pub component_type: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub result: AnalysisResult,
    pub confidence: f64,
    pub execution_time: Duration,
}

/// Analysis result variants
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnalysisResult {
    QuantumNash {
        equilibrium: QuantumEquilibrium,
        convergence_score: f64,
    },
    Machiavellian {
        manipulation_detected: ManipulationDetection,
        strategy: MachiavellianStrategy,
    },
    RobinHood {
        wealth_distribution: WealthDistribution,
        interventions: Vec<MarketIntervention>,
    },
    TemporalNash {
        equilibrium: TemporalEquilibrium,
        learning_progress: f64,
    },
    Antifragile {
        coalitions: Vec<AntifragileCoalition>,
        volatility_benefit: f64,
    },
    Error {
        error_message: String,
    },
}

/// Quantum equilibrium state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumEquilibrium {
    pub strategies: HashMap<String, Vec<f64>>,
    pub payoffs: HashMap<String, f64>,
    pub convergence_score: f64,
    pub optimal_action: ActionType,
    pub stability_metrics: HashMap<String, f64>,
}

/// Manipulation detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManipulationDetection {
    pub detected: bool,
    pub confidence: f64,
    pub manipulation_scores: HashMap<String, f64>,
    pub primary_pattern: String,
    pub recommended_action: String,
}

/// Machiavellian strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MachiavellianStrategy {
    pub recommended_action: ActionType,
    pub confidence: f64,
    pub tactics: Vec<String>,
    pub deception_mode: bool,
}

/// Wealth distribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WealthDistribution {
    pub gini_coefficient: f64,
    pub top_10_percent_share: f64,
    pub concentration_ratio: f64,
    pub intervention_needed: bool,
    pub concentrated_participants: Vec<String>,
}

/// Market intervention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketIntervention {
    pub intervention_type: String,
    pub target_markets: Vec<String>,
    pub expected_impact: f64,
    pub resource_requirement: String,
    pub risk_level: String,
    pub priority: f64,
}

/// Temporal equilibrium
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalEquilibrium {
    pub strategies: HashMap<String, Vec<f64>>,
    pub payoffs: HashMap<String, f64>,
    pub convergence_rate: f64,
    pub learning_confidence: f64,
    pub predicted_action: ActionType,
}

/// Antifragile coalition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntifragileCoalition {
    pub coalition_id: String,
    pub members: Vec<String>,
    pub synergy: f64,
    pub volatility_benefit: f64,
    pub crisis_readiness: f64,
    pub expected_performance: HashMap<String, f64>,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_decisions: u64,
    pub successful_decisions: u64,
    pub average_confidence: f64,
    pub average_execution_time: Duration,
    pub component_performance: HashMap<String, ComponentPerformance>,
    pub recent_performance: Vec<DecisionOutcome>,
}

/// Component-specific performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentPerformance {
    pub component_name: String,
    pub total_executions: u64,
    pub successful_executions: u64,
    pub average_execution_time: Duration,
    pub error_rate: f64,
    pub confidence_trend: Vec<f64>,
}

/// Decision outcome tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionOutcome {
    pub decision: MarketDecision,
    pub outcome: f64, // Realized profit/loss
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub market_conditions: String,
}

/// Resource allocation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequest {
    pub request_id: String,
    pub operation: String,
    pub num_qubits: Option<usize>,
    pub estimated_memory: Option<usize>, // MB
    pub priority: String,
    pub timeout: Option<Duration>,
}

/// Resource allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub allocation_id: String,
    pub request_id: String,
    pub allocated_at: chrono::DateTime<chrono::Utc>,
    pub resources: AllocatedResources,
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Allocated resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocatedResources {
    pub cpu_cores: Option<usize>,
    pub memory_mb: Option<usize>,
    pub gpu_memory_mb: Option<usize>,
    pub gpu_index: Option<usize>,
    pub quantum_circuits: Option<usize>,
}

/// Agent status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStatus {
    pub agent_id: String,
    pub is_running: bool,
    pub uptime: Duration,
    pub hardware_info: HardwareInfo,
    pub memory_usage: MemoryUsage,
    pub performance_summary: PerformanceSummary,
    pub active_components: Vec<String>,
    pub last_checkpoint: Option<chrono::DateTime<chrono::Utc>>,
}

/// Hardware information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub device_type: String,
    pub cpu_cores: usize,
    pub total_memory_mb: usize,
    pub gpu_available: bool,
    pub gpu_memory_mb: Option<usize>,
    pub quantum_backend: Option<String>,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub short_term_memory_mb: f64,
    pub long_term_memory_mb: f64,
    pub total_memory_mb: f64,
    pub capacity_percentage: f64,
    pub consolidation_rate: f64,
}

/// Performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub total_analyses: u64,
    pub successful_analyses: u64,
    pub average_execution_time: Duration,
    pub success_rate: f64,
    pub recent_confidence: f64,
}

/// Biological constraints for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalConstraints {
    pub learning_rate: f64,
    pub memory_decay: f64,
    pub attention_span: usize,
    pub reaction_delay: usize,
    pub noise_level: f64,
    pub fatigue_rate: f64,
}

impl Default for BiologicalConstraints {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            memory_decay: 0.95,
            attention_span: 100,
            reaction_delay: 5,
            noise_level: 0.05,
            fatigue_rate: 0.001,
        }
    }
}

/// Utility trait for converting between different number types
pub trait NumCast {
    fn as_f64(&self) -> f64;
    fn as_usize(&self) -> usize;
}

impl NumCast for f64 {
    fn as_f64(&self) -> f64 {
        *self
    }
    
    fn as_usize(&self) -> usize {
        *self as usize
    }
}

impl NumCast for usize {
    fn as_f64(&self) -> f64 {
        *self as f64
    }
    
    fn as_usize(&self) -> usize {
        *self
    }
}

/// Validation trait for data structures
pub trait Validate {
    fn validate(&self) -> Result<(), String>;
}

impl Validate for MarketData {
    fn validate(&self) -> Result<(), String> {
        if self.price_history.is_empty() {
            return Err("Price history cannot be empty".to_string());
        }
        
        if self.snapshot.price <= 0.0 {
            return Err("Price must be positive".to_string());
        }
        
        if self.snapshot.volume < 0.0 {
            return Err("Volume must be non-negative".to_string());
        }
        
        Ok(())
    }
}

impl Validate for MarketDecision {
    fn validate(&self) -> Result<(), String> {
        if self.confidence < 0.0 || self.confidence > 1.0 {
            return Err("Confidence must be between 0 and 1".to_string());
        }
        
        if self.decision_vector.len() != 4 {
            return Err("Decision vector must have 4 elements".to_string());
        }
        
        let sum: f64 = self.decision_vector.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err("Decision vector must sum to 1.0".to_string());
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_action_type_conversion() {
        assert_eq!(ActionType::from(0), ActionType::Buy);
        assert_eq!(ActionType::from(1), ActionType::Sell);
        assert_eq!(ActionType::from(2), ActionType::Hold);
        assert_eq!(ActionType::from(3), ActionType::Wait);
        
        assert_eq!(usize::from(ActionType::Buy), 0);
        assert_eq!(usize::from(ActionType::Sell), 1);
        assert_eq!(usize::from(ActionType::Hold), 2);
        assert_eq!(usize::from(ActionType::Wait), 3);
    }
    
    #[test]
    fn test_market_decision_validation() {
        let decision = MarketDecision {
            action: ActionType::Buy,
            confidence: 0.8,
            decision_vector: vec![0.5, 0.2, 0.2, 0.1],
            reasoning: "Test decision".to_string(),
            timestamp: chrono::Utc::now(),
            component_contributions: HashMap::new(),
        };
        
        assert!(decision.validate().is_ok());
        
        let invalid_decision = MarketDecision {
            confidence: 1.5, // Invalid confidence
            ..decision
        };
        
        assert!(invalid_decision.validate().is_err());
    }
    
    #[test]
    fn test_biological_constraints_default() {
        let constraints = BiologicalConstraints::default();
        assert_eq!(constraints.learning_rate, 0.01);
        assert_eq!(constraints.memory_decay, 0.95);
        assert_eq!(constraints.attention_span, 100);
    }
}