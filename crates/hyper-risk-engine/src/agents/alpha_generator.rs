//! Alpha Generator Agent.
//!
//! Generates trading signals based on statistical models and
//! market microstructure analysis.
//!
//! ## Signal Types
//! - Momentum signals
//! - Mean reversion signals
//! - Cross-asset signals
//! - Microstructure signals
//!
//! ## Scientific References
//! - Avellaneda & Lee (2010): "Statistical Arbitrage in the U.S. Equities Market"
//! - Cartea et al. (2015): "Algorithmic and High-Frequency Trading"

use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

use parking_lot::RwLock;

use crate::core::types::{MarketRegime, Portfolio, RiskDecision, Symbol, Timestamp};
use crate::core::error::Result;
use super::base::{Agent, AgentId, AgentStatus, AgentConfig, AgentStats};

/// Alpha signal direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalDirection {
    /// Long signal (bullish).
    Long,
    /// Short signal (bearish).
    Short,
    /// Neutral (no signal).
    Neutral,
}

/// Alpha signal with strength.
#[derive(Debug, Clone)]
pub struct AlphaSignal {
    /// Symbol for the signal.
    pub symbol: Symbol,
    /// Signal direction.
    pub direction: SignalDirection,
    /// Signal strength (0.0 - 1.0).
    pub strength: f64,
    /// Confidence level (0.0 - 1.0).
    pub confidence: f64,
    /// Expected holding period in seconds.
    pub holding_period_secs: u64,
    /// Signal generation timestamp.
    pub timestamp: Timestamp,
}

/// Alpha generator configuration.
#[derive(Debug, Clone)]
pub struct AlphaGeneratorConfig {
    /// Base agent config.
    pub base: AgentConfig,
    /// Minimum signal strength to emit.
    pub min_signal_strength: f64,
    /// Minimum confidence level.
    pub min_confidence: f64,
    /// Enable momentum signals.
    pub enable_momentum: bool,
    /// Enable mean reversion signals.
    pub enable_mean_reversion: bool,
    /// Lookback window in seconds.
    pub lookback_secs: u64,
}

impl Default for AlphaGeneratorConfig {
    fn default() -> Self {
        Self {
            base: AgentConfig {
                name: "AlphaGenerator".to_string(),
                max_latency_us: 800,
                ..Default::default()
            },
            min_signal_strength: 0.3,
            min_confidence: 0.6,
            enable_momentum: true,
            enable_mean_reversion: true,
            lookback_secs: 300, // 5 minutes
        }
    }
}

/// Price observation for signal generation.
#[derive(Debug, Clone, Copy)]
pub struct PriceObservation {
    /// Price value.
    pub price: f64,
    /// Volume.
    pub volume: f64,
    /// Timestamp.
    pub timestamp: u64,
}

/// Alpha Generator Agent.
#[derive(Debug)]
pub struct AlphaGeneratorAgent {
    /// Configuration.
    config: AlphaGeneratorConfig,
    /// Current status.
    status: AtomicU8,
    /// Current signals.
    signals: RwLock<Vec<AlphaSignal>>,
    /// Price history for lookback.
    price_history: RwLock<Vec<PriceObservation>>,
    /// Statistics.
    stats: AgentStats,
}

impl AlphaGeneratorAgent {
    /// Create new alpha generator agent.
    pub fn new(config: AlphaGeneratorConfig) -> Self {
        Self {
            config,
            status: AtomicU8::new(AgentStatus::Idle as u8),
            signals: RwLock::new(Vec::new()),
            price_history: RwLock::new(Vec::with_capacity(1000)),
            stats: AgentStats::new(),
        }
    }

    /// Update price history.
    pub fn update_price(&self, symbol: Symbol, price: f64, volume: f64) {
        let obs = PriceObservation {
            price,
            volume,
            timestamp: Timestamp::now().as_nanos(),
        };

        let mut history = self.price_history.write();
        history.push(obs);

        // Trim old observations
        let cutoff = Timestamp::now().as_nanos() - (self.config.lookback_secs * 1_000_000_000);
        history.retain(|o| o.timestamp >= cutoff);
    }

    /// Generate momentum signal.
    fn generate_momentum_signal(&self, history: &[PriceObservation]) -> Option<(SignalDirection, f64)> {
        if history.len() < 10 {
            return None;
        }

        let n = history.len();
        let recent = history[n - 5..].iter().map(|o| o.price).sum::<f64>() / 5.0;
        let older = history[..n - 5].iter().map(|o| o.price).sum::<f64>() / (n - 5) as f64;

        if older <= 0.0 {
            return None;
        }

        let momentum = (recent - older) / older;

        if momentum > 0.01 {
            // 1% positive momentum
            Some((SignalDirection::Long, momentum.min(1.0)))
        } else if momentum < -0.01 {
            // 1% negative momentum
            Some((SignalDirection::Short, momentum.abs().min(1.0)))
        } else {
            Some((SignalDirection::Neutral, 0.0))
        }
    }

    /// Generate mean reversion signal.
    fn generate_mean_reversion_signal(&self, history: &[PriceObservation]) -> Option<(SignalDirection, f64)> {
        if history.len() < 20 {
            return None;
        }

        // Calculate z-score
        let prices: Vec<f64> = history.iter().map(|o| o.price).collect();
        let mean: f64 = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance: f64 = prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / prices.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev <= 0.0 {
            return None;
        }

        let current = prices[prices.len() - 1];
        let z_score = (current - mean) / std_dev;

        if z_score > 2.0 {
            // Overbought - short signal
            Some((SignalDirection::Short, (z_score - 2.0).min(1.0)))
        } else if z_score < -2.0 {
            // Oversold - long signal
            Some((SignalDirection::Long, (z_score.abs() - 2.0).min(1.0)))
        } else {
            Some((SignalDirection::Neutral, 0.0))
        }
    }

    /// Get current signals.
    pub fn get_signals(&self) -> Vec<AlphaSignal> {
        self.signals.read().clone()
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

impl Agent for AlphaGeneratorAgent {
    fn id(&self) -> AgentId {
        AgentId::new(&self.config.base.name)
    }

    fn status(&self) -> AgentStatus {
        Self::status_from_u8(self.status.load(Ordering::Relaxed))
    }

    fn process(&self, _portfolio: &Portfolio, regime: MarketRegime) -> Result<Option<RiskDecision>> {
        let start = Instant::now();
        self.status.store(AgentStatus::Processing as u8, Ordering::Relaxed);

        let history = self.price_history.read();
        let mut new_signals = Vec::new();

        // Generate momentum signals
        if self.config.enable_momentum {
            if let Some((direction, strength)) = self.generate_momentum_signal(&history) {
                if direction != SignalDirection::Neutral && strength >= self.config.min_signal_strength {
                    new_signals.push(AlphaSignal {
                        symbol: Symbol::new("DEFAULT"), // Would come from actual data
                        direction,
                        strength,
                        confidence: 0.7 * regime.risk_multiplier(), // Adjust by regime
                        holding_period_secs: 300,
                        timestamp: Timestamp::now(),
                    });
                }
            }
        }

        // Generate mean reversion signals
        if self.config.enable_mean_reversion {
            if let Some((direction, strength)) = self.generate_mean_reversion_signal(&history) {
                if direction != SignalDirection::Neutral && strength >= self.config.min_signal_strength {
                    new_signals.push(AlphaSignal {
                        symbol: Symbol::new("DEFAULT"),
                        direction,
                        strength,
                        confidence: 0.65 * regime.risk_multiplier(),
                        holding_period_secs: 600,
                        timestamp: Timestamp::now(),
                    });
                }
            }
        }

        // Update signals
        {
            let mut signals = self.signals.write();
            *signals = new_signals;
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
    fn test_alpha_generator_creation() {
        let config = AlphaGeneratorConfig::default();
        let agent = AlphaGeneratorAgent::new(config);
        assert_eq!(agent.status(), AgentStatus::Idle);
    }

    #[test]
    fn test_momentum_signal() {
        let config = AlphaGeneratorConfig::default();
        let agent = AlphaGeneratorAgent::new(config);

        // Create upward trending prices
        for i in 0..20 {
            agent.update_price(Symbol::new("TEST"), 100.0 + i as f64, 1000.0);
        }

        let history = agent.price_history.read();
        let signal = agent.generate_momentum_signal(&history);
        assert!(signal.is_some());
        assert_eq!(signal.unwrap().0, SignalDirection::Long);
    }
}
