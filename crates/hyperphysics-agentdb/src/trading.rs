//! Trading-specific types and utilities for AgentDB
//!
//! All types are pair/symbol agnostic - designed to work with any trading instrument.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Trading action type (pair-agnostic)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeAction {
    /// Long position (buy)
    Long,
    /// Short position (sell)
    Short,
    /// Hold current position
    Hold,
    /// Close position
    Close,
    /// Scale in (add to position)
    ScaleIn,
    /// Scale out (reduce position)
    ScaleOut,
}

impl std::fmt::Display for TradeAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TradeAction::Long => write!(f, "LONG"),
            TradeAction::Short => write!(f, "SHORT"),
            TradeAction::Hold => write!(f, "HOLD"),
            TradeAction::Close => write!(f, "CLOSE"),
            TradeAction::ScaleIn => write!(f, "SCALE_IN"),
            TradeAction::ScaleOut => write!(f, "SCALE_OUT"),
        }
    }
}

impl Default for TradeAction {
    fn default() -> Self {
        Self::Hold
    }
}

/// Market regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketRegime {
    /// Strong upward trend
    BullTrend,
    /// Strong downward trend
    BearTrend,
    /// Low volatility sideways
    RangeBound,
    /// High volatility, no clear direction
    HighVolatility,
    /// Market transition period
    Transitioning,
}

/// Market context at time of trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketContext {
    /// Current market regime
    pub regime: MarketRegime,
    /// Volatility (e.g., realized vol, ATR)
    pub volatility: f64,
    /// Volume relative to average
    pub relative_volume: f64,
    /// Trend strength (0-1)
    pub trend_strength: f64,
    /// Order book imbalance (-1 to 1)
    pub order_imbalance: f64,
    /// Funding rate (for perpetuals)
    pub funding_rate: Option<f64>,
    /// Open interest change
    pub oi_change: Option<f64>,
    /// Additional indicators
    pub indicators: HashMap<String, f64>,
}

impl Default for MarketContext {
    fn default() -> Self {
        Self {
            regime: MarketRegime::RangeBound,
            volatility: 0.0,
            relative_volume: 1.0,
            trend_strength: 0.0,
            order_imbalance: 0.0,
            funding_rate: None,
            oi_change: None,
            indicators: HashMap::new(),
        }
    }
}

/// A complete trading episode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingEpisode {
    /// Trading symbol (e.g., "BTC-USD", "ETH-PERP")
    pub symbol: String,
    /// Action taken
    pub action: TradeAction,
    /// Entry price
    pub entry_price: f64,
    /// Exit price (if closed)
    pub exit_price: Option<f64>,
    /// Position size (as fraction of portfolio)
    pub position_size: f64,
    /// P&L as percentage
    pub pnl: f64,
    /// P&L in base currency
    pub pnl_absolute: f64,
    /// Trade duration in seconds
    pub duration_secs: u64,
    /// Market context at entry
    pub entry_context: MarketContext,
    /// Market context at exit (if closed)
    pub exit_context: Option<MarketContext>,
    /// Signals that triggered the trade
    pub signals: Vec<String>,
    /// Strategy that generated the trade
    pub strategy: String,
    /// Self-critique / lessons learned
    pub critique: String,
    /// Timestamp (Unix epoch)
    pub timestamp: i64,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Default for TradingEpisode {
    fn default() -> Self {
        Self {
            symbol: String::new(),
            action: TradeAction::default(),
            entry_price: 0.0,
            exit_price: None,
            position_size: 0.0,
            pnl: 0.0,
            pnl_absolute: 0.0,
            duration_secs: 0,
            entry_context: MarketContext::default(),
            exit_context: None,
            signals: Vec::new(),
            strategy: String::new(),
            critique: String::new(),
            timestamp: 0,
            metadata: HashMap::new(),
        }
    }
}

impl TradingEpisode {
    /// Create new episode for any trading pair
    pub fn new(symbol: &str, action: TradeAction, context: MarketContext) -> Self {
        Self {
            symbol: symbol.to_string(),
            action,
            entry_context: context,
            timestamp: chrono::Utc::now().timestamp(),
            ..Default::default()
        }
    }

    /// Generate task description (pair-agnostic)
    pub fn task_description(&self) -> String {
        format!(
            "{} {} on {} using {}",
            self.action,
            if self.pnl >= 0.0 { "profit" } else { "loss" },
            self.symbol,
            self.strategy
        )
    }

    /// Generate a self-critique from the trade outcome
    pub fn generate_critique(&self) -> String {
        let outcome = if self.pnl > 0.0 { "profitable" } else { "losing" };
        let size_assessment = if self.position_size > 0.1 {
            "Large position"
        } else if self.position_size < 0.02 {
            "Small position"
        } else {
            "Moderate position"
        };

        format!(
            "{} {} trade on {} with {} size. Entry: {:.2}, Exit: {:.2}, P&L: {:.2}%. {}. Signals: {}. {}",
            outcome.to_uppercase(),
            self.action,
            self.symbol,
            size_assessment.to_lowercase(),
            self.entry_price,
            self.exit_price.unwrap_or(self.entry_price),
            self.pnl * 100.0,
            self.entry_context.regime_description(),
            self.signals.join(", "),
            self.critique,
        )
    }

    /// Convert actions to string representation
    pub fn actions_as_strings(&self) -> Vec<String> {
        vec![
            format!("ACTION: {}", self.action),
            format!("SIZE: {:.2}%", self.position_size * 100.0),
            format!("STRATEGY: {}", self.strategy),
        ]
    }

    /// Convert observations to string representation
    pub fn observations_as_strings(&self) -> Vec<String> {
        let mut obs = vec![
            format!("ENTRY_PRICE: {:.2}", self.entry_price),
            format!("REGIME: {:?}", self.entry_context.regime),
            format!("VOLATILITY: {:.4}", self.entry_context.volatility),
            format!("VOLUME: {:.2}x", self.entry_context.relative_volume),
        ];

        if let Some(exit) = self.exit_price {
            obs.push(format!("EXIT_PRICE: {:.2}", exit));
        }

        obs.push(format!("PNL: {:.2}%", self.pnl * 100.0));
        obs.push(format!("DURATION: {}s", self.duration_secs));

        for signal in &self.signals {
            obs.push(format!("SIGNAL: {}", signal));
        }

        obs
    }
}

impl MarketContext {
    /// Get human-readable regime description
    pub fn regime_description(&self) -> String {
        match self.regime {
            MarketRegime::BullTrend => format!("Strong uptrend (strength: {:.2})", self.trend_strength),
            MarketRegime::BearTrend => format!("Strong downtrend (strength: {:.2})", self.trend_strength),
            MarketRegime::RangeBound => format!("Range-bound market (vol: {:.2}%)", self.volatility * 100.0),
            MarketRegime::HighVolatility => format!("High volatility environment ({:.2}%)", self.volatility * 100.0),
            MarketRegime::Transitioning => "Market in transition".to_string(),
        }
    }

    /// Create context from indicator values
    pub fn from_indicators(indicators: HashMap<String, f64>) -> Self {
        let volatility = *indicators.get("volatility").unwrap_or(&0.0);
        let trend_strength = *indicators.get("trend_strength").unwrap_or(&0.0);
        let relative_volume = *indicators.get("relative_volume").unwrap_or(&1.0);

        let regime = if trend_strength > 0.7 {
            MarketRegime::BullTrend
        } else if trend_strength < -0.7 {
            MarketRegime::BearTrend
        } else if volatility > 0.03 {
            MarketRegime::HighVolatility
        } else {
            MarketRegime::RangeBound
        };

        Self {
            regime,
            volatility,
            relative_volume,
            trend_strength: trend_strength.abs(),
            order_imbalance: *indicators.get("order_imbalance").unwrap_or(&0.0),
            funding_rate: indicators.get("funding_rate").copied(),
            oi_change: indicators.get("oi_change").copied(),
            indicators,
        }
    }
}

/// A trading strategy skill
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategySkill {
    /// Skill name
    pub name: String,
    /// Description of the strategy
    pub description: String,
    /// Parameters and their descriptions
    pub parameters: HashMap<String, String>,
    /// Example trades (as descriptions)
    pub example_trades: Vec<String>,
    /// Win rate
    pub win_rate: f64,
    /// Average P&L per trade
    pub avg_pnl: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Max drawdown
    pub max_drawdown: f64,
    /// Number of trades
    pub trade_count: usize,
}

impl StrategySkill {
    /// Create a new strategy skill from backtest results
    pub fn from_backtest(
        name: &str,
        description: &str,
        trades: &[TradingEpisode],
    ) -> Self {
        let trade_count = trades.len();
        let wins = trades.iter().filter(|t| t.pnl > 0.0).count();
        let win_rate = wins as f64 / trade_count as f64;
        
        let pnls: Vec<f64> = trades.iter().map(|t| t.pnl).collect();
        let avg_pnl = pnls.iter().sum::<f64>() / trade_count as f64;
        
        // Simple Sharpe calculation
        let mean = avg_pnl;
        let variance = pnls.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / trade_count as f64;
        let std_dev = variance.sqrt();
        let sharpe_ratio = if std_dev > 0.0 { mean / std_dev } else { 0.0 };
        
        // Max drawdown
        let mut peak = 0.0f64;
        let mut max_drawdown = 0.0f64;
        let mut cumulative = 0.0;
        for pnl in &pnls {
            cumulative += pnl;
            if cumulative > peak {
                peak = cumulative;
            }
            let drawdown = (peak - cumulative) / peak.max(0.0001);
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        let example_trades: Vec<String> = trades
            .iter()
            .take(5)
            .map(|t| t.generate_critique())
            .collect();

        Self {
            name: name.to_string(),
            description: description.to_string(),
            parameters: HashMap::new(),
            example_trades,
            win_rate,
            avg_pnl,
            sharpe_ratio,
            max_drawdown,
            trade_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SYMBOL: &str = "TEST-PAIR";

    #[test]
    fn test_trading_episode_new() {
        let episode = TradingEpisode::new(TEST_SYMBOL, TradeAction::Long, MarketContext::default());
        assert_eq!(episode.symbol, TEST_SYMBOL);
        assert_eq!(episode.action, TradeAction::Long);
    }

    #[test]
    fn test_trading_episode_critique() {
        let mut episode = TradingEpisode::new(TEST_SYMBOL, TradeAction::Long, MarketContext::default());
        episode.entry_price = 100.0;
        episode.exit_price = Some(105.0);
        episode.position_size = 0.05;
        episode.pnl = 0.05;
        episode.signals = vec!["signal_a".into(), "signal_b".into()];
        episode.strategy = "test_strategy".into();
        episode.critique = "Test critique".into();

        let critique = episode.generate_critique();
        assert!(critique.contains("PROFITABLE"));
        assert!(critique.contains(TEST_SYMBOL));
    }

    #[test]
    fn test_task_description() {
        let mut episode = TradingEpisode::new(TEST_SYMBOL, TradeAction::Short, MarketContext::default());
        episode.strategy = "mean_reversion".into();
        episode.pnl = -0.02;

        let desc = episode.task_description();
        assert!(desc.contains("SHORT"));
        assert!(desc.contains(TEST_SYMBOL));
        assert!(desc.contains("mean_reversion"));
    }

    #[test]
    fn test_market_context_from_indicators() {
        let mut indicators = HashMap::new();
        indicators.insert("volatility".into(), 0.025);
        indicators.insert("trend_strength".into(), 0.8);
        indicators.insert("relative_volume".into(), 1.5);

        let context = MarketContext::from_indicators(indicators);
        assert_eq!(context.regime, MarketRegime::BullTrend);
    }
}
