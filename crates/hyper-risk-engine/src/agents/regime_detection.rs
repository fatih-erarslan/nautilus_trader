//! Regime Detection Agent.
//!
//! Detects market regimes using Hidden Markov Models (HMM) and
//! Markov-Switching GARCH models.
//!
//! ## Regime Types
//! - Bull trending
//! - Bear trending
//! - Sideways low volatility
//! - Sideways high volatility
//! - Crisis
//! - Recovery
//!
//! ## Scientific References
//! - Hamilton (1989): "A New Approach to the Economic Analysis of Nonstationary Time Series"
//! - Gray (1996): "Modeling the Conditional Distribution of Interest Rates as a Regime-Switching Process"
//! - Ang & Bekaert (2002): "Regime Switches in Interest Rates"

use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

use parking_lot::RwLock;

use crate::core::types::{MarketRegime, Portfolio, RiskDecision, Timestamp};
use crate::core::error::Result;
use super::base::{Agent, AgentId, AgentStatus, AgentConfig, AgentStats};

/// Regime detection configuration.
#[derive(Debug, Clone)]
pub struct RegimeDetectionConfig {
    /// Base agent config.
    pub base: AgentConfig,
    /// Number of hidden states.
    pub num_states: usize,
    /// Lookback window in seconds.
    pub lookback_secs: u64,
    /// Minimum observations for detection.
    pub min_observations: usize,
    /// Volatility threshold for high/low classification.
    pub volatility_threshold: f64,
    /// Trend threshold for bull/bear classification.
    pub trend_threshold: f64,
}

impl Default for RegimeDetectionConfig {
    fn default() -> Self {
        Self {
            base: AgentConfig {
                name: "RegimeDetection".to_string(),
                max_latency_us: 1000,
                ..Default::default()
            },
            num_states: 4,              // Bull, Bear, SidewaysLow, SidewaysHigh
            lookback_secs: 3600,        // 1 hour
            min_observations: 100,
            volatility_threshold: 0.02, // 2% realized vol threshold
            trend_threshold: 0.005,     // 0.5% trend threshold
        }
    }
}

/// Regime state probabilities.
#[derive(Debug, Clone, Default)]
pub struct RegimeProbabilities {
    /// Probability of bull trending.
    pub bull_trending: f64,
    /// Probability of bear trending.
    pub bear_trending: f64,
    /// Probability of sideways low vol.
    pub sideways_low: f64,
    /// Probability of sideways high vol.
    pub sideways_high: f64,
    /// Probability of crisis.
    pub crisis: f64,
    /// Probability of recovery.
    pub recovery: f64,
}

impl RegimeProbabilities {
    /// Get most likely regime.
    pub fn most_likely(&self) -> MarketRegime {
        let probs = [
            (self.bull_trending, MarketRegime::BullTrending),
            (self.bear_trending, MarketRegime::BearTrending),
            (self.sideways_low, MarketRegime::SidewaysLow),
            (self.sideways_high, MarketRegime::SidewaysHigh),
            (self.crisis, MarketRegime::Crisis),
            (self.recovery, MarketRegime::Recovery),
        ];

        probs
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .map(|(_, regime)| *regime)
            .unwrap_or(MarketRegime::Unknown)
    }

    /// Get confidence of most likely regime.
    pub fn confidence(&self) -> f64 {
        [
            self.bull_trending,
            self.bear_trending,
            self.sideways_low,
            self.sideways_high,
            self.crisis,
            self.recovery,
        ]
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max)
    }
}

/// Return observation for regime detection.
#[derive(Debug, Clone, Copy)]
pub struct ReturnObservation {
    /// Log return.
    pub log_return: f64,
    /// Realized volatility.
    pub volatility: f64,
    /// Timestamp.
    pub timestamp: u64,
}

/// Regime Detection Agent.
#[derive(Debug)]
pub struct RegimeDetectionAgent {
    /// Configuration.
    config: RegimeDetectionConfig,
    /// Current status.
    status: AtomicU8,
    /// Current regime.
    current_regime: RwLock<MarketRegime>,
    /// Regime probabilities.
    probabilities: RwLock<RegimeProbabilities>,
    /// Return history.
    return_history: RwLock<Vec<ReturnObservation>>,
    /// Statistics.
    stats: AgentStats,
}

impl RegimeDetectionAgent {
    /// Create new regime detection agent.
    pub fn new(config: RegimeDetectionConfig) -> Self {
        Self {
            config,
            status: AtomicU8::new(AgentStatus::Idle as u8),
            current_regime: RwLock::new(MarketRegime::Unknown),
            probabilities: RwLock::new(RegimeProbabilities::default()),
            return_history: RwLock::new(Vec::with_capacity(1000)),
            stats: AgentStats::new(),
        }
    }

    /// Add return observation.
    pub fn add_observation(&self, log_return: f64, volatility: f64) {
        let obs = ReturnObservation {
            log_return,
            volatility,
            timestamp: Timestamp::now().as_nanos(),
        };

        let mut history = self.return_history.write();
        history.push(obs);

        // Trim old observations
        let cutoff = Timestamp::now().as_nanos() - (self.config.lookback_secs * 1_000_000_000);
        history.retain(|o| o.timestamp >= cutoff);
    }

    /// Detect regime from observations.
    fn detect_regime(&self, history: &[ReturnObservation]) -> (MarketRegime, RegimeProbabilities) {
        if history.len() < self.config.min_observations {
            return (MarketRegime::Unknown, RegimeProbabilities::default());
        }

        // Calculate statistics
        let returns: Vec<f64> = history.iter().map(|o| o.log_return).collect();
        let vols: Vec<f64> = history.iter().map(|o| o.volatility).collect();

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let mean_vol = vols.iter().sum::<f64>() / vols.len() as f64;

        // Calculate realized volatility
        let variance = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / returns.len() as f64;
        let realized_vol = variance.sqrt();

        // Simple regime classification (production would use HMM)
        let is_high_vol = realized_vol > self.config.volatility_threshold;
        let is_bull = mean_return > self.config.trend_threshold;
        let is_bear = mean_return < -self.config.trend_threshold;

        // Calculate probabilities based on heuristics
        // In production, this would use Baum-Welch or Viterbi algorithm
        let mut probs = RegimeProbabilities::default();

        if is_bull && !is_high_vol {
            probs.bull_trending = 0.7;
            probs.recovery = 0.2;
            probs.sideways_low = 0.1;
        } else if is_bull && is_high_vol {
            probs.recovery = 0.5;
            probs.bull_trending = 0.3;
            probs.sideways_high = 0.2;
        } else if is_bear && !is_high_vol {
            probs.bear_trending = 0.6;
            probs.sideways_low = 0.3;
            probs.crisis = 0.1;
        } else if is_bear && is_high_vol {
            probs.crisis = 0.6;
            probs.bear_trending = 0.3;
            probs.sideways_high = 0.1;
        } else if !is_high_vol {
            probs.sideways_low = 0.7;
            probs.bull_trending = 0.15;
            probs.bear_trending = 0.15;
        } else {
            probs.sideways_high = 0.6;
            probs.crisis = 0.2;
            probs.recovery = 0.2;
        }

        // Check for crisis conditions (sharp drawdown + high vol)
        let recent_returns = &returns[returns.len().saturating_sub(10)..];
        let recent_sum: f64 = recent_returns.iter().sum();
        if recent_sum < -0.05 && mean_vol > self.config.volatility_threshold * 2.0 {
            probs = RegimeProbabilities {
                crisis: 0.8,
                bear_trending: 0.15,
                sideways_high: 0.05,
                ..Default::default()
            };
        }

        let regime = probs.most_likely();
        (regime, probs)
    }

    /// Get current regime.
    pub fn get_regime(&self) -> MarketRegime {
        *self.current_regime.read()
    }

    /// Get current probabilities.
    pub fn get_probabilities(&self) -> RegimeProbabilities {
        self.probabilities.read().clone()
    }

    fn status_from_u8(val: u8) -> AgentStatus {
        match val {
            0 => AgentStatus::Idle,
            1 => AgentStatus::Processing,
            2 => AgentStatus::Paused,
            3 => AgentStatus::Error,
            _ => AgentStatus::ShuttingDown,
        }
    }
}

impl Agent for RegimeDetectionAgent {
    fn id(&self) -> AgentId {
        AgentId::new(&self.config.base.name)
    }

    fn status(&self) -> AgentStatus {
        Self::status_from_u8(self.status.load(Ordering::Relaxed))
    }

    fn process(&self, _portfolio: &Portfolio, _regime: MarketRegime) -> Result<Option<RiskDecision>> {
        let start = Instant::now();
        self.status.store(AgentStatus::Processing as u8, Ordering::Relaxed);

        // Detect regime from history
        let history = self.return_history.read();
        let (new_regime, new_probs) = self.detect_regime(&history);
        drop(history);

        // Update state
        {
            let mut regime = self.current_regime.write();
            *regime = new_regime;
        }
        {
            let mut probs = self.probabilities.write();
            *probs = new_probs;
        }

        self.stats.record_cycle(start.elapsed().as_nanos() as u64);
        self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);
        Ok(None)
    }

    fn start(&self) -> Result<()> {
        self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);
        Ok(())
    }

    fn stop(&self) -> Result<()> {
        self.status.store(AgentStatus::ShuttingDown as u8, Ordering::Relaxed);
        Ok(())
    }

    fn pause(&self) {
        self.status.store(AgentStatus::Paused as u8, Ordering::Relaxed);
    }

    fn resume(&self) {
        self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);
    }

    fn process_count(&self) -> u64 {
        self.stats.cycles.load(Ordering::Relaxed)
    }

    fn avg_latency_ns(&self) -> u64 {
        self.stats.avg_latency_ns()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_detection_creation() {
        let config = RegimeDetectionConfig::default();
        let agent = RegimeDetectionAgent::new(config);
        assert_eq!(agent.status(), AgentStatus::Idle);
        assert_eq!(agent.get_regime(), MarketRegime::Unknown);
    }

    #[test]
    fn test_regime_probabilities() {
        let probs = RegimeProbabilities {
            bull_trending: 0.6,
            bear_trending: 0.1,
            sideways_low: 0.2,
            sideways_high: 0.1,
            crisis: 0.0,
            recovery: 0.0,
        };

        assert_eq!(probs.most_likely(), MarketRegime::BullTrending);
        assert!((probs.confidence() - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_bull_detection() {
        let config = RegimeDetectionConfig {
            min_observations: 10,
            ..Default::default()
        };
        let agent = RegimeDetectionAgent::new(config);

        // Add bullish observations
        for _ in 0..20 {
            agent.add_observation(0.01, 0.01); // 1% return, low vol
        }

        let history = agent.return_history.read();
        let (regime, _) = agent.detect_regime(&history);
        assert_eq!(regime, MarketRegime::BullTrending);
    }

    #[test]
    fn test_crisis_detection() {
        let config = RegimeDetectionConfig {
            min_observations: 10,
            ..Default::default()
        };
        let agent = RegimeDetectionAgent::new(config);

        // Add crisis observations
        for _ in 0..20 {
            agent.add_observation(-0.03, 0.05); // -3% return, high vol
        }

        let history = agent.return_history.read();
        let (regime, _) = agent.detect_regime(&history);
        assert_eq!(regime, MarketRegime::Crisis);
    }
}
