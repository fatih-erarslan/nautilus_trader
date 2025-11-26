//! Type definitions for the ReasoningBank self-learning engine

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A recorded pattern matching experience with predicted and actual outcomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternExperience {
    /// Unique identifier for this experience
    pub id: String,

    /// Type of pattern detected (e.g., "head_and_shoulders", "double_bottom")
    pub pattern_type: String,

    /// Vector embedding of the pattern for similarity search
    pub pattern_vector: Vec<f32>,

    /// Similarity score to the historical pattern (0-1)
    pub similarity: f64,

    /// Confidence in the prediction (0-1)
    pub confidence: f64,

    /// Expected return when pattern was detected
    pub predicted_outcome: f64,

    /// Actual return after pattern completion (filled later)
    pub actual_outcome: Option<f64>,

    /// Market context when pattern was detected
    pub market_context: MarketContext,

    /// When the experience was recorded
    pub timestamp: DateTime<Utc>,
}

/// Market conditions when a pattern was detected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketContext {
    /// Trading symbol (e.g., "BTC-USD")
    pub symbol: String,

    /// Timeframe (e.g., "1h", "4h", "1d")
    pub timeframe: String,

    /// Current market volatility (standard deviation)
    pub volatility: f64,

    /// Trading volume
    pub volume: f64,

    /// Market trend ("bullish", "bearish", "neutral")
    pub trend: String,

    /// Market sentiment score (-1 to 1)
    pub sentiment: f64,
}

/// Verdict on prediction quality after outcome is known
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternVerdict {
    /// Reference to the experience being judged
    pub experience_id: String,

    /// Overall quality score (0-1)
    pub quality_score: f64,

    /// Whether the predicted direction was correct
    pub direction_correct: bool,

    /// Magnitude prediction error (0-1)
    pub magnitude_error: f64,

    /// Whether this pattern should be learned from
    pub should_learn: bool,

    /// Whether thresholds should be adapted
    pub should_adapt: bool,

    /// Suggested parameter changes
    pub suggested_changes: Vec<Adaptation>,
}

/// Suggested adaptation to matching parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adaptation {
    /// Parameter name (e.g., "similarity_threshold")
    pub parameter: String,

    /// Current parameter value
    pub current_value: f64,

    /// Suggested new value
    pub suggested_value: f64,

    /// Explanation for the suggestion
    pub reason: String,
}

/// A successful pattern distilled into long-term memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistilledPattern {
    /// Type of pattern
    pub pattern_type: String,

    /// Vector embedding for similarity search
    pub pattern_vector: Vec<f32>,

    /// Historical success rate (0-1)
    pub success_rate: f64,

    /// Average return when pattern occurs
    pub avg_return: f64,

    /// Minimum confidence threshold for this pattern
    pub confidence_threshold: f64,

    /// Minimum similarity threshold for this pattern
    pub similarity_threshold: f64,

    /// Typical market conditions for this pattern
    pub market_conditions: MarketContext,

    /// Number of successful instances
    pub sample_count: usize,

    /// Last time pattern was updated
    pub last_updated: DateTime<Utc>,
}

/// Performance trajectory for a pattern type over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternTrajectory {
    /// Pattern type being tracked
    pub pattern_type: String,

    /// Total number of samples
    pub sample_count: usize,

    /// Success rate (profitable trades / total trades)
    pub success_rate: f64,

    /// Average return per trade
    pub avg_return: f64,

    /// Risk-adjusted return metric
    pub sharpe_ratio: f64,

    /// Best return achieved
    pub best_return: f64,

    /// Worst return (max loss)
    pub worst_return: f64,

    /// All experiences for this pattern
    pub experiences: Vec<PatternExperience>,

    /// Last trajectory update time
    pub last_updated: DateTime<Utc>,
}

/// Trading signal generated from pattern matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    /// Direction ("LONG", "SHORT", or "NEUTRAL")
    pub direction: String,

    /// Confidence in signal (0-1)
    pub confidence: f64,

    /// Expected return from this trade
    pub expected_return: f64,

    /// Experience ID for tracking outcomes
    pub experience_id: String,

    /// Pattern type that generated the signal
    pub pattern_type: String,
}

/// Matching thresholds for pattern detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchingThresholds {
    /// Minimum similarity score to match pattern (0-1)
    pub similarity_threshold: f64,

    /// Minimum confidence to generate signal (0-1)
    pub confidence_threshold: f64,

    /// Minimum number of historical samples required
    pub min_sample_size: usize,
}

impl Default for MatchingThresholds {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.80,
            confidence_threshold: 0.70,
            min_sample_size: 10,
        }
    }
}
