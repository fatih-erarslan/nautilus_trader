//! Core types for Talebian Risk Management System
//!
//! Critical types for financial risk management

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Configuration for aggressive Machiavellian trading strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacchiavelianConfig {
    /// Antifragility threshold (0.0-1.0)
    pub antifragility_threshold: f64,
    /// Barbell strategy safe allocation ratio
    pub barbell_safe_ratio: f64,
    /// Barbell strategy risky allocation ratio
    pub barbell_risky_ratio: f64,
    /// Black swan detection threshold
    pub black_swan_threshold: f64,
    /// Kelly criterion fraction for position sizing
    pub kelly_fraction: f64,
    /// Maximum Kelly fraction allowed
    pub kelly_max_fraction: f64,
    /// Whale detection volume threshold multiplier
    pub whale_volume_threshold: f64,
    /// Maximum leverage allowed
    pub max_leverage: f64,
    /// Risk per trade
    pub risk_per_trade: f64,
    /// Stop loss percentage
    pub stop_loss: f64,
    /// Take profit percentage
    pub take_profit: f64,
    /// Enable quantum enhancement
    pub enable_quantum: bool,
    /// Tail risk percentile
    pub tail_risk_percentile: f64,
    /// Antifragility measurement window
    pub antifragility_window: usize,
    /// Number of qubits for quantum processing
    pub num_qubits: usize,
    /// Quantum circuit depth
    pub circuit_depth: usize,
    /// Whale detected multiplier
    pub whale_detected_multiplier: f64,
    /// Parasitic opportunity threshold
    pub parasitic_opportunity_threshold: f64,
    /// Destructive swan protection
    pub destructive_swan_protection: f64,
    /// Dynamic rebalance threshold
    pub dynamic_rebalance_threshold: f64,
}

impl MacchiavelianConfig {
    /// Create aggressive default configuration for maximum returns
    pub fn aggressive_defaults() -> Self {
        Self {
            antifragility_threshold: 0.3,
            barbell_safe_ratio: 0.65,
            barbell_risky_ratio: 0.35,
            black_swan_threshold: 0.05,
            kelly_fraction: 0.55, // Aggressive Kelly
            kelly_max_fraction: 0.75,
            whale_volume_threshold: 3.0,
            max_leverage: 2.0,
            risk_per_trade: 0.02,
            stop_loss: 0.05,
            take_profit: 0.15,
            enable_quantum: false,
            tail_risk_percentile: 0.01,
            antifragility_window: 100,
            num_qubits: 8,
            circuit_depth: 10,
            whale_detected_multiplier: 1.5,
            parasitic_opportunity_threshold: 0.6,
            destructive_swan_protection: 0.25,
            dynamic_rebalance_threshold: 0.1,
        }
    }

    /// Create conservative configuration
    pub fn conservative_defaults() -> Self {
        Self {
            antifragility_threshold: 0.5,
            barbell_safe_ratio: 0.80,
            barbell_risky_ratio: 0.20,
            black_swan_threshold: 0.01,
            kelly_fraction: 0.25,
            kelly_max_fraction: 0.40,
            whale_volume_threshold: 5.0,
            max_leverage: 1.0,
            risk_per_trade: 0.01,
            stop_loss: 0.02,
            take_profit: 0.10,
            enable_quantum: false,
            tail_risk_percentile: 0.01,
            antifragility_window: 200,
            num_qubits: 4,
            circuit_depth: 5,
            whale_detected_multiplier: 1.2,
            parasitic_opportunity_threshold: 0.7,
            destructive_swan_protection: 0.15,
            dynamic_rebalance_threshold: 0.05,
        }
    }

    /// Create conservative baseline configuration
    pub fn conservative_baseline() -> Self {
        Self::conservative_defaults()
    }
}

/// Market data structure for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    /// DateTime timestamp
    pub timestamp: DateTime<Utc>,
    /// Unix timestamp
    pub timestamp_unix: i64,
    /// Current price
    pub price: f64,
    /// Trading volume
    pub volume: f64,
    /// Bid price
    pub bid: f64,
    /// Ask price
    pub ask: f64,
    /// Bid volume
    pub bid_volume: f64,
    /// Ask volume
    pub ask_volume: f64,
    /// Current volatility
    pub volatility: f64,
    /// Recent returns
    pub returns: Vec<f64>,
    /// Volume history
    pub volume_history: Vec<f64>,
}

impl MarketData {
    /// Calculate spread
    pub fn spread(&self) -> f64 {
        self.ask - self.bid
    }

    /// Calculate mid price
    pub fn mid_price(&self) -> f64 {
        (self.bid + self.ask) / 2.0
    }

    /// Check if data is valid
    pub fn is_valid(&self) -> bool {
        self.price > 0.0
            && self.volume >= 0.0
            && self.bid > 0.0
            && self.ask > 0.0
            && self.bid <= self.ask
            && self.volatility >= 0.0
    }
}

/// Whale detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleDetection {
    /// Timestamp of detection
    pub timestamp: i64,
    /// Is whale activity detected
    pub detected: bool,
    /// Volume spike ratio
    pub volume_spike: f64,
    /// Whale direction
    pub direction: WhaleDirection,
    /// Confidence level (0.0-1.0)
    pub confidence: f64,
    /// Estimated whale size
    pub whale_size: f64,
    /// Market impact estimate
    pub impact: f64,
    /// Is whale detected (alias for detected)
    pub is_whale_detected: bool,
    /// Order book imbalance
    pub order_book_imbalance: f64,
    /// Price impact
    pub price_impact: f64,
}

/// Direction of whale activity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WhaleDirection {
    /// Whale is buying
    Buying,
    /// Whale is selling
    Selling,
    /// Direction unclear
    Neutral,
    /// Mixed signals
    Mixed,
}

/// Parasitic opportunity detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParasiticOpportunity {
    /// Opportunity identifier
    pub id: String,
    /// Expected return
    pub expected_return: f64,
    /// Risk level
    pub risk_level: f64,
    /// Time window in seconds
    pub time_window: u64,
    /// Confidence score
    pub confidence: f64,
    /// Entry price
    pub entry_price: f64,
    /// Exit price target
    pub exit_price: f64,
    /// Stop loss
    pub stop_loss: f64,
    /// Opportunity score (0.0 to 1.0)
    pub opportunity_score: f64,
    /// Momentum factor
    pub momentum_factor: f64,
    /// Volatility factor
    pub volatility_factor: f64,
    /// Whale alignment score
    pub whale_alignment: f64,
    /// Market regime factor
    pub regime_factor: f64,
    /// Recommended allocation
    pub recommended_allocation: f64,
}

/// Antifragility measurement result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntifragilityMeasurement {
    /// Antifragility score (-1.0 to 1.0)
    pub score: f64,
    /// Fragility index
    pub fragility_index: f64,
    /// Robustness measure
    pub robustness: f64,
    /// Volatility benefit
    pub volatility_benefit: f64,
    /// Stress response
    pub stress_response: f64,
}

/// Black swan event detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackSwanEvent {
    /// Event timestamp
    pub timestamp: i64,
    /// Event magnitude
    pub magnitude: f64,
    /// Event type
    pub event_type: String,
    /// Probability estimate
    pub probability: f64,
    /// Impact assessment
    pub impact: f64,
    /// Recovery time estimate
    pub recovery_time: u64,
}

/// Performance tracking
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceTracker {
    /// Total returns
    pub total_returns: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate
    pub win_rate: f64,
    /// Average win
    pub avg_win: f64,
    /// Average loss
    pub avg_loss: f64,
    /// Total trades
    pub total_trades: u64,
    /// Total observations
    pub total_observations: usize,
    /// Last update timestamp
    pub last_update: i64,
}

/// Return data for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReturnData {
    /// Return values
    pub returns: Vec<f64>,
    /// Timestamps
    pub timestamps: Vec<i64>,
    /// Cumulative returns
    pub cumulative_returns: Vec<f64>,
    /// Expected return
    pub expected_return: f64,
    /// Volatility
    pub volatility: f64,
    /// Unix timestamp for current data point
    pub timestamp_unix: i64,
    /// Single timestamp for compatibility
    pub timestamp: i64,
}

/// Talebian risk assessment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TalebianRiskAssessment {
    /// Overall risk score
    pub risk_score: f64,
    /// Antifragility assessment
    pub antifragility: AntifragilityMeasurement,
    /// Antifragility score
    pub antifragility_score: f64,
    /// Black swan probability
    pub black_swan_probability: f64,
    /// Barbell allocation (safe, risky)
    pub barbell_allocation: (f64, f64),
    /// Kelly fraction for position sizing
    pub kelly_fraction: f64,
    /// Whale detection results
    pub _whale_detection: Option<WhaleDetection>,
    /// Parasitic opportunity if detected
    pub parasitic_opportunity: Option<ParasiticOpportunity>,
    /// Recommended position size
    pub position_size: f64,
    /// Risk warnings
    pub warnings: Vec<String>,
    /// Overall risk score (alias for risk_score)
    pub overall_risk_score: f64,
    /// Recommended position size (alias for position_size)
    pub recommended_position_size: f64,
    /// Confidence level
    pub confidence: f64,
}

/// Risk category classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskCategory {
    /// Very low risk
    VeryLow,
    /// Low risk
    Low,
    /// Medium risk
    Medium,
    /// High risk
    High,
    /// Very high risk
    VeryHigh,
    /// Extreme risk
    Extreme,
}

impl RiskCategory {
    /// Get category from risk score
    pub fn from_score(score: f64) -> Self {
        match score {
            s if s < 0.1 => Self::VeryLow,
            s if s < 0.3 => Self::Low,
            s if s < 0.5 => Self::Medium,
            s if s < 0.7 => Self::High,
            s if s < 0.9 => Self::VeryHigh,
            _ => Self::Extreme,
        }
    }
}
