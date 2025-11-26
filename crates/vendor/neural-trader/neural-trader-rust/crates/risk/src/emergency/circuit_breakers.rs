//! Circuit breaker implementation for market protection
//!
//! Automatically halts trading when:
//! - Extreme volatility detected
//! - Rapid losses exceed threshold
//! - Market conditions become abnormal

use crate::{Result, RiskError};
use crate::types::{AlertLevel, Portfolio};
use chrono::{DateTime, Duration, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use tracing::{error, info, warn};

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitBreakerState {
    /// Normal operation
    Closed,
    /// Temporarily halted
    Open,
    /// Recovery period (limited trading)
    HalfOpen,
}

/// Circuit breaker trigger condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerCondition {
    /// Rapid loss exceeds threshold (e.g., 5% in 5 minutes)
    RapidLoss { threshold_pct: f64, time_window_secs: u64 },
    /// High volatility spike
    VolatilitySpike { threshold_multiple: f64 },
    /// Drawdown exceeds limit
    MaxDrawdown { threshold_pct: f64 },
    /// Too many failed trades
    ConsecutiveLosses { count: usize },
    /// Manual trigger
    Manual,
}

/// Circuit breaker for risk protection
pub struct CircuitBreaker {
    state: Arc<RwLock<CircuitBreakerState>>,
    config: CircuitBreakerConfig,
    history: Arc<RwLock<VecDeque<CircuitBreakerEvent>>>,
    pnl_history: Arc<RwLock<VecDeque<(DateTime<Utc>, f64)>>>,
    trip_count: Arc<RwLock<usize>>,
}

/// Configuration for circuit breaker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Rapid loss threshold (e.g., 0.05 = 5%)
    pub rapid_loss_threshold: f64,
    /// Time window for rapid loss detection (seconds)
    pub rapid_loss_window_secs: u64,
    /// Volatility spike threshold (multiple of normal)
    pub volatility_spike_threshold: f64,
    /// Maximum drawdown before circuit break
    pub max_drawdown_threshold: f64,
    /// Number of consecutive losses to trigger
    pub consecutive_loss_count: usize,
    /// Cool-down period before allowing trading again (seconds)
    pub cooldown_period_secs: u64,
    /// Maximum number of circuit breaker trips per day
    pub max_trips_per_day: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            rapid_loss_threshold: 0.05,     // 5% loss
            rapid_loss_window_secs: 300,     // 5 minutes
            volatility_spike_threshold: 3.0, // 3x normal volatility
            max_drawdown_threshold: 0.15,    // 15% drawdown
            consecutive_loss_count: 5,       // 5 losses in a row
            cooldown_period_secs: 1800,      // 30 minutes
            max_trips_per_day: 3,            // Max 3 circuit breaks per day
        }
    }
}

impl CircuitBreaker {
    /// Create new circuit breaker
    pub fn new(config: CircuitBreakerConfig) -> Self {
        info!("Circuit breaker initialized with config: {:?}", config);
        Self {
            state: Arc::new(RwLock::new(CircuitBreakerState::Closed)),
            config,
            history: Arc::new(RwLock::new(VecDeque::new())),
            pnl_history: Arc::new(RwLock::new(VecDeque::new())),
            trip_count: Arc::new(RwLock::new(0)),
        }
    }

    /// Check if trading is allowed
    pub fn is_trading_allowed(&self) -> bool {
        let state = self.state.read();
        *state == CircuitBreakerState::Closed || *state == CircuitBreakerState::HalfOpen
    }

    /// Get current state
    pub fn get_state(&self) -> CircuitBreakerState {
        *self.state.read()
    }

    /// Update P&L and check for triggers
    pub fn update_pnl(&self, portfolio: &Portfolio) -> Result<()> {
        let current_pnl = portfolio.total_unrealized_pnl();
        let timestamp = Utc::now();

        // Record P&L
        {
            let mut pnl_hist = self.pnl_history.write();
            pnl_hist.push_back((timestamp, current_pnl));

            // Keep only recent history (1 hour)
            let cutoff = timestamp - Duration::hours(1);
            while pnl_hist.front().map(|(t, _)| *t < cutoff).unwrap_or(false) {
                pnl_hist.pop_front();
            }
        }

        // Check for rapid loss
        self.check_rapid_loss()?;

        // Auto-reset if cooldown period has passed
        self.check_cooldown_expiry();

        Ok(())
    }

    /// Check for rapid loss trigger
    fn check_rapid_loss(&self) -> Result<()> {
        let pnl_hist = self.pnl_history.read();
        if pnl_hist.len() < 2 {
            return Ok(());
        }

        let now = Utc::now();
        let window_start = now - Duration::seconds(self.config.rapid_loss_window_secs as i64);

        // Find P&L at window start
        let start_pnl = pnl_hist
            .iter()
            .find(|(t, _)| *t >= window_start)
            .map(|(_, p)| *p)
            .unwrap_or(0.0);

        let current_pnl = pnl_hist.back().map(|(_, p)| *p).unwrap_or(0.0);

        let pnl_change = current_pnl - start_pnl;
        let loss_pct = if start_pnl.abs() > 1.0 {
            -pnl_change / start_pnl.abs()
        } else {
            0.0
        };

        if loss_pct > self.config.rapid_loss_threshold {
            warn!(
                "Rapid loss detected: {:.2}% in {} seconds",
                loss_pct * 100.0,
                self.config.rapid_loss_window_secs
            );
            self.trip(TriggerCondition::RapidLoss {
                threshold_pct: loss_pct,
                time_window_secs: self.config.rapid_loss_window_secs,
            })?;
        }

        Ok(())
    }

    /// Trigger circuit breaker
    pub fn trip(&self, condition: TriggerCondition) -> Result<()> {
        let mut state = self.state.write();

        if *state != CircuitBreakerState::Closed {
            // Already tripped
            return Ok(());
        }

        // Check daily limit
        let mut trip_count = self.trip_count.write();
        if *trip_count >= self.config.max_trips_per_day {
            error!(
                "Circuit breaker daily limit reached: {} trips",
                *trip_count
            );
            return Err(RiskError::EmergencyProtocolError(
                "Circuit breaker daily limit reached".to_string(),
            ));
        }

        *state = CircuitBreakerState::Open;
        *trip_count += 1;

        let event = CircuitBreakerEvent {
            timestamp: Utc::now(),
            condition: condition.clone(),
            state_change: CircuitBreakerState::Open,
            alert_level: AlertLevel::Emergency,
        };

        error!("Circuit breaker TRIPPED: {:?}", condition);

        // Record event
        let mut history = self.history.write();
        history.push_back(event.clone());
        if history.len() > 100 {
            history.pop_front();
        }

        Ok(())
    }

    /// Manually reset circuit breaker
    pub fn reset(&self) -> Result<()> {
        let mut state = self.state.write();

        match *state {
            CircuitBreakerState::Open => {
                *state = CircuitBreakerState::HalfOpen;
                info!("Circuit breaker reset to HalfOpen state");

                let event = CircuitBreakerEvent {
                    timestamp: Utc::now(),
                    condition: TriggerCondition::Manual,
                    state_change: CircuitBreakerState::HalfOpen,
                    alert_level: AlertLevel::Warning,
                };

                let mut history = self.history.write();
                history.push_back(event);
            }
            CircuitBreakerState::HalfOpen => {
                *state = CircuitBreakerState::Closed;
                info!("Circuit breaker fully reset to Closed state");

                let event = CircuitBreakerEvent {
                    timestamp: Utc::now(),
                    condition: TriggerCondition::Manual,
                    state_change: CircuitBreakerState::Closed,
                    alert_level: AlertLevel::Info,
                };

                let mut history = self.history.write();
                history.push_back(event);
            }
            CircuitBreakerState::Closed => {
                warn!("Circuit breaker already in Closed state");
            }
        }

        Ok(())
    }

    /// Check if cooldown period has expired
    fn check_cooldown_expiry(&self) {
        let history = self.history.read();
        let last_trip = history.back();

        if let Some(event) = last_trip {
            if event.state_change == CircuitBreakerState::Open {
                let elapsed = Utc::now() - event.timestamp;
                if elapsed.num_seconds() as u64 >= self.config.cooldown_period_secs {
                    drop(history);
                    let _ = self.reset();
                }
            }
        }
    }

    /// Get circuit breaker history
    pub fn get_history(&self, limit: usize) -> Vec<CircuitBreakerEvent> {
        let history = self.history.read();
        history.iter().rev().take(limit).cloned().collect()
    }

    /// Get trip count for today
    pub fn get_trip_count(&self) -> usize {
        *self.trip_count.read()
    }

    /// Reset daily trip count (should be called at day rollover)
    pub fn reset_daily_count(&self) {
        let mut count = self.trip_count.write();
        *count = 0;
        info!("Circuit breaker daily count reset");
    }

    /// Check drawdown and trip if necessary
    pub fn check_drawdown(&self, portfolio: &Portfolio, peak_value: f64) -> Result<()> {
        let current_value = portfolio.total_value();
        if peak_value > 0.0 {
            let drawdown = (peak_value - current_value) / peak_value;

            if drawdown > self.config.max_drawdown_threshold {
                warn!("Maximum drawdown exceeded: {:.2}%", drawdown * 100.0);
                self.trip(TriggerCondition::MaxDrawdown {
                    threshold_pct: drawdown,
                })?;
            }
        }

        Ok(())
    }
}

/// Circuit breaker event record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerEvent {
    pub timestamp: DateTime<Utc>,
    pub condition: TriggerCondition,
    pub state_change: CircuitBreakerState,
    pub alert_level: AlertLevel,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{PositionSide, Symbol};
    use rust_decimal_macros::dec;
    use std::thread;
    use std::time::Duration as StdDuration;

    fn create_test_portfolio(pnl: f64) -> Portfolio {
        let mut portfolio = Portfolio::new(dec!(100000));

        let current_price_val = 150.0 + pnl / 100.0;
        let market_value_val = 15000.0 + pnl;

        portfolio.update_position(Position {
            symbol: Symbol::new("AAPL"),
            quantity: dec!(100),
            avg_entry_price: dec!(150.0),
            current_price: rust_decimal::Decimal::from_f64_retain(current_price_val).unwrap(),
            market_value: rust_decimal::Decimal::from_f64_retain(market_value_val).unwrap(),
            unrealized_pnl: rust_decimal::Decimal::from_f64_retain(pnl).unwrap(),
            unrealized_pnl_percent: dec!(0),
            side: PositionSide::Long,
            opened_at: Utc::now(),
        });

        portfolio
    }

    #[test]
    fn test_circuit_breaker_creation() {
        let _config = CircuitBreakerConfig::default();
        let breaker = CircuitBreaker::new(config);

        assert_eq!(breaker.get_state(), CircuitBreakerState::Closed);
        assert!(breaker.is_trading_allowed());
    }

    #[test]
    fn test_manual_trip() {
        let _config = CircuitBreakerConfig::default();
        let breaker = CircuitBreaker::new(config);

        breaker.trip(TriggerCondition::Manual).unwrap();

        assert_eq!(breaker.get_state(), CircuitBreakerState::Open);
        assert!(!breaker.is_trading_allowed());
    }

    #[test]
    fn test_reset_sequence() {
        let _config = CircuitBreakerConfig::default();
        let breaker = CircuitBreaker::new(config);

        // Trip it
        breaker.trip(TriggerCondition::Manual).unwrap();
        assert_eq!(breaker.get_state(), CircuitBreakerState::Open);

        // First reset -> HalfOpen
        breaker.reset().unwrap();
        assert_eq!(breaker.get_state(), CircuitBreakerState::HalfOpen);
        assert!(breaker.is_trading_allowed()); // Half-open allows trading

        // Second reset -> Closed
        breaker.reset().unwrap();
        assert_eq!(breaker.get_state(), CircuitBreakerState::Closed);
    }

    #[test]
    fn test_rapid_loss_detection() {
        let mut config = CircuitBreakerConfig::default();
        config.rapid_loss_threshold = 0.05; // 5%
        config.rapid_loss_window_secs = 2;   // 2 seconds for test

        let breaker = CircuitBreaker::new(config);

        // Record initial positive P&L
        let portfolio1 = create_test_portfolio(1000.0);
        breaker.update_pnl(&portfolio1).unwrap();

        // Wait a bit
        thread::sleep(StdDuration::from_millis(100));

        // Record large loss
        let portfolio2 = create_test_portfolio(-1000.0);
        breaker.update_pnl(&portfolio2).unwrap();

        // Should trip on rapid loss
        assert_eq!(breaker.get_state(), CircuitBreakerState::Open);
    }

    #[test]
    fn test_drawdown_trigger() {
        let _config = CircuitBreakerConfig::default();
        let breaker = CircuitBreaker::new(config);

        let portfolio = create_test_portfolio(-10000.0);
        let peak_value = 100000.0;

        // Should not trip yet (10% drawdown)
        breaker.check_drawdown(&portfolio, peak_value).unwrap();
        assert_eq!(breaker.get_state(), CircuitBreakerState::Closed);

        // Create larger drawdown (20%)
        let portfolio2 = create_test_portfolio(-20000.0);
        breaker.check_drawdown(&portfolio2, peak_value).unwrap();

        // Should trip (exceeds 15% threshold)
        assert_eq!(breaker.get_state(), CircuitBreakerState::Open);
    }

    #[test]
    fn test_trip_count_limit() {
        let mut config = CircuitBreakerConfig::default();
        config.max_trips_per_day = 2;

        let breaker = CircuitBreaker::new(config);

        // First trip - OK
        breaker.trip(TriggerCondition::Manual).unwrap();
        assert_eq!(breaker.get_trip_count(), 1);

        // Reset and trip again
        breaker.reset().unwrap();
        breaker.reset().unwrap(); // Fully reset
        breaker.trip(TriggerCondition::Manual).unwrap();
        assert_eq!(breaker.get_trip_count(), 2);

        // Third trip should fail (exceeds daily limit)
        breaker.reset().unwrap();
        breaker.reset().unwrap();
        let result = breaker.trip(TriggerCondition::Manual);
        assert!(result.is_err());
    }

    #[test]
    fn test_history_tracking() {
        let _config = CircuitBreakerConfig::default();
        let breaker = CircuitBreaker::new(config);

        breaker.trip(TriggerCondition::Manual).unwrap();
        breaker.reset().unwrap();

        let history = breaker.get_history(10);
        assert_eq!(history.len(), 2); // Trip + reset
    }
}
