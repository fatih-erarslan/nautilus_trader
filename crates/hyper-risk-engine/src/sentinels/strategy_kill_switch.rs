//! Strategy kill switch sentinel for emergency trading halt.
//!
//! Operates in the fast path (<20μs) to immediately halt trading
//! when critical conditions are detected.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::time::Instant;

use parking_lot::RwLock;

use crate::core::types::{Order, Portfolio, Symbol, Timestamp};
use crate::core::error::{Result, RiskError};

use super::base::{Sentinel, SentinelConfig, SentinelId, SentinelStats, SentinelStatus};

/// Configuration for the strategy kill switch sentinel.
#[derive(Debug, Clone)]
pub struct StrategyKillSwitchConfig {
    /// Base sentinel configuration.
    pub base: SentinelConfig,
    /// Maximum daily loss threshold (percentage of NAV).
    pub max_daily_loss_pct: f64,
    /// Maximum drawdown threshold (percentage).
    pub max_drawdown_pct: f64,
    /// Maximum loss per strategy (percentage of allocated capital).
    pub max_strategy_loss_pct: f64,
    /// Maximum consecutive losses before kill.
    pub max_consecutive_losses: u32,
    /// Cool-down period after kill in seconds.
    pub cooldown_secs: u64,
}

impl Default for StrategyKillSwitchConfig {
    fn default() -> Self {
        Self {
            base: SentinelConfig {
                name: "strategy_kill_switch_sentinel".to_string(),
                enabled: true,
                priority: 0, // Highest priority
                verbose: false,
            },
            max_daily_loss_pct: 5.0,
            max_drawdown_pct: 10.0,
            max_strategy_loss_pct: 20.0,
            max_consecutive_losses: 10,
            cooldown_secs: 3600, // 1 hour
        }
    }
}

/// Kill switch trigger reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KillReason {
    /// Daily loss limit exceeded.
    DailyLossLimit,
    /// Maximum drawdown exceeded.
    MaxDrawdown,
    /// Strategy loss limit exceeded.
    StrategyLossLimit,
    /// Too many consecutive losses.
    ConsecutiveLosses,
    /// Manual kill triggered.
    ManualKill,
    /// System error detected.
    SystemError,
}

/// Kill switch event record.
#[derive(Debug, Clone)]
pub struct KillEvent {
    /// Reason for kill.
    pub reason: KillReason,
    /// Symbol (if strategy-specific).
    pub symbol: Option<Symbol>,
    /// Strategy ID (if applicable).
    pub strategy_id: Option<String>,
    /// Trigger value.
    pub trigger_value: f64,
    /// Threshold that was exceeded.
    pub threshold: f64,
    /// Event timestamp.
    pub triggered_at: Timestamp,
    /// Cooldown end time.
    pub cooldown_until: Timestamp,
}

/// Strategy performance tracking.
#[derive(Debug, Clone)]
pub struct StrategyPerformance {
    /// Strategy identifier.
    pub strategy_id: String,
    /// Daily P&L.
    pub daily_pnl: f64,
    /// Peak value (for drawdown calculation).
    pub peak_value: f64,
    /// Current value.
    pub current_value: f64,
    /// Consecutive losses count.
    pub consecutive_losses: u32,
    /// Last update.
    pub last_update: Timestamp,
    /// Is strategy killed.
    pub killed: bool,
}

/// Strategy kill switch sentinel.
#[derive(Debug)]
pub struct StrategyKillSwitchSentinel {
    id: SentinelId,
    config: StrategyKillSwitchConfig,
    status: AtomicU8,
    stats: SentinelStats,
    /// Global kill switch (kills all trading).
    global_kill: AtomicBool,
    /// Strategy performance by ID.
    strategies: RwLock<HashMap<String, StrategyPerformance>>,
    /// Kill events history.
    kill_events: RwLock<Vec<KillEvent>>,
    /// Cooldown end timestamp (global).
    global_cooldown_until: RwLock<Option<Timestamp>>,
}

impl StrategyKillSwitchSentinel {
    /// Create a new strategy kill switch sentinel.
    pub fn new(config: StrategyKillSwitchConfig) -> Self {
        Self {
            id: SentinelId::new(&config.base.name),
            config,
            status: AtomicU8::new(SentinelStatus::Active as u8),
            stats: SentinelStats::new(),
            global_kill: AtomicBool::new(false),
            strategies: RwLock::new(HashMap::new()),
            kill_events: RwLock::new(Vec::new()),
            global_cooldown_until: RwLock::new(None),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(StrategyKillSwitchConfig::default())
    }

    /// Register a strategy for monitoring.
    pub fn register_strategy(&self, strategy_id: &str, initial_value: f64) {
        let mut strategies = self.strategies.write();
        strategies.insert(
            strategy_id.to_string(),
            StrategyPerformance {
                strategy_id: strategy_id.to_string(),
                daily_pnl: 0.0,
                peak_value: initial_value,
                current_value: initial_value,
                consecutive_losses: 0,
                last_update: Timestamp::now(),
                killed: false,
            },
        );
    }

    /// Update strategy P&L.
    pub fn update_strategy_pnl(&self, strategy_id: &str, pnl: f64, new_value: f64) {
        let mut strategies = self.strategies.write();
        if let Some(perf) = strategies.get_mut(strategy_id) {
            perf.daily_pnl += pnl;
            perf.current_value = new_value;
            perf.peak_value = perf.peak_value.max(new_value);
            perf.last_update = Timestamp::now();

            // Track consecutive losses
            if pnl < 0.0 {
                perf.consecutive_losses += 1;
            } else if pnl > 0.0 {
                perf.consecutive_losses = 0;
            }
        }
    }

    /// Trigger global kill switch.
    pub fn trigger_global_kill(&self, reason: KillReason) {
        self.global_kill.store(true, Ordering::SeqCst);
        self.status.store(SentinelStatus::Triggered as u8, Ordering::SeqCst);

        let now = Timestamp::now();
        let cooldown_ns = self.config.cooldown_secs * 1_000_000_000;
        let cooldown_until = Timestamp::from_nanos(now.as_nanos() + cooldown_ns);

        *self.global_cooldown_until.write() = Some(cooldown_until);

        self.kill_events.write().push(KillEvent {
            reason,
            symbol: None,
            strategy_id: None,
            trigger_value: 0.0,
            threshold: 0.0,
            triggered_at: now,
            cooldown_until,
        });
    }

    /// Trigger strategy-specific kill.
    pub fn trigger_strategy_kill(&self, strategy_id: &str, reason: KillReason, trigger_value: f64, threshold: f64) {
        let mut strategies = self.strategies.write();
        if let Some(perf) = strategies.get_mut(strategy_id) {
            perf.killed = true;
        }
        drop(strategies);

        let now = Timestamp::now();
        let cooldown_ns = self.config.cooldown_secs * 1_000_000_000;
        let cooldown_until = Timestamp::from_nanos(now.as_nanos() + cooldown_ns);

        self.kill_events.write().push(KillEvent {
            reason,
            symbol: None,
            strategy_id: Some(strategy_id.to_string()),
            trigger_value,
            threshold,
            triggered_at: now,
            cooldown_until,
        });
    }

    /// Check if global kill is active.
    pub fn is_global_kill_active(&self) -> bool {
        if !self.global_kill.load(Ordering::Relaxed) {
            return false;
        }

        // Check cooldown
        if let Some(cooldown_until) = *self.global_cooldown_until.read() {
            if Timestamp::now().as_nanos() >= cooldown_until.as_nanos() {
                // Cooldown expired, reset kill
                self.global_kill.store(false, Ordering::SeqCst);
                return false;
            }
        }

        true
    }

    /// Check if a strategy is killed.
    pub fn is_strategy_killed(&self, strategy_id: &str) -> bool {
        self.strategies
            .read()
            .get(strategy_id)
            .map(|p| p.killed)
            .unwrap_or(false)
    }

    /// Reset global kill switch.
    pub fn reset_global_kill(&self) {
        self.global_kill.store(false, Ordering::SeqCst);
        *self.global_cooldown_until.write() = None;
    }

    /// Reset strategy kill.
    pub fn reset_strategy_kill(&self, strategy_id: &str) {
        let mut strategies = self.strategies.write();
        if let Some(perf) = strategies.get_mut(strategy_id) {
            perf.killed = false;
            perf.consecutive_losses = 0;
        }
    }

    /// Get kill events history.
    pub fn get_kill_events(&self) -> Vec<KillEvent> {
        self.kill_events.read().clone()
    }

    /// Get strategy performance.
    pub fn get_strategy_performance(&self, strategy_id: &str) -> Option<StrategyPerformance> {
        self.strategies.read().get(strategy_id).cloned()
    }

    /// Check strategy limits.
    fn check_strategy_limits(&self, portfolio: &Portfolio) -> Result<()> {
        let strategies = self.strategies.read();
        let nav = portfolio.total_value;

        for (strategy_id, perf) in strategies.iter() {
            let strategy_id = strategy_id.clone(); // Clone to allow dropping lock

            if perf.killed {
                return Err(RiskError::InternalError(format!(
                    "Strategy {} is killed",
                    strategy_id
                )));
            }

            // Check daily loss
            if nav > 0.0 {
                let loss_pct = (-perf.daily_pnl / nav) * 100.0;
                if loss_pct > self.config.max_daily_loss_pct {
                    let _consecutive_losses = perf.consecutive_losses;
                    drop(strategies);
                    self.trigger_strategy_kill(
                        &strategy_id,
                        KillReason::DailyLossLimit,
                        loss_pct,
                        self.config.max_daily_loss_pct,
                    );
                    return Err(RiskError::InternalError(format!(
                        "Strategy {} daily loss {:.2}% exceeds limit {:.2}%",
                        strategy_id, loss_pct, self.config.max_daily_loss_pct
                    )));
                }
            }

            // Check drawdown
            if perf.peak_value > 0.0 {
                let drawdown_pct = ((perf.peak_value - perf.current_value) / perf.peak_value) * 100.0;
                if drawdown_pct > self.config.max_drawdown_pct {
                    drop(strategies);
                    self.trigger_strategy_kill(
                        &strategy_id,
                        KillReason::MaxDrawdown,
                        drawdown_pct,
                        self.config.max_drawdown_pct,
                    );
                    return Err(RiskError::InternalError(format!(
                        "Strategy {} drawdown {:.2}% exceeds limit {:.2}%",
                        strategy_id, drawdown_pct, self.config.max_drawdown_pct
                    )));
                }
            }

            // Check consecutive losses
            if perf.consecutive_losses >= self.config.max_consecutive_losses {
                let consecutive_losses = perf.consecutive_losses;
                drop(strategies);
                self.trigger_strategy_kill(
                    &strategy_id,
                    KillReason::ConsecutiveLosses,
                    consecutive_losses as f64,
                    self.config.max_consecutive_losses as f64,
                );
                return Err(RiskError::InternalError(format!(
                    "Strategy {} has {} consecutive losses (limit: {})",
                    strategy_id, consecutive_losses, self.config.max_consecutive_losses
                )));
            }
        }

        Ok(())
    }

    /// Convert u8 to SentinelStatus.
    fn status_from_u8(value: u8) -> SentinelStatus {
        match value {
            0 => SentinelStatus::Active,
            1 => SentinelStatus::Disabled,
            2 => SentinelStatus::Triggered,
            _ => SentinelStatus::Error,
        }
    }
}

impl Sentinel for StrategyKillSwitchSentinel {
    fn id(&self) -> SentinelId {
        self.id.clone()
    }

    fn status(&self) -> SentinelStatus {
        Self::status_from_u8(self.status.load(Ordering::Relaxed))
    }

    fn check(&self, _order: &Order, portfolio: &Portfolio) -> Result<()> {
        let start = Instant::now();

        // Check if disabled
        if self.status() == SentinelStatus::Disabled {
            return Ok(());
        }

        // Check global kill
        if self.is_global_kill_active() {
            self.stats.record_trigger();
            let latency_ns = start.elapsed().as_nanos() as u64;
            self.stats.record_check(latency_ns);
            return Err(RiskError::InternalError(
                "Global kill switch is active".to_string(),
            ));
        }

        // Check portfolio-level limits
        let nav = portfolio.total_value;
        let daily_pnl = portfolio.unrealized_pnl + portfolio.realized_pnl;

        if nav > 0.0 {
            let loss_pct = (-daily_pnl / nav) * 100.0;
            if loss_pct > self.config.max_daily_loss_pct {
                self.trigger_global_kill(KillReason::DailyLossLimit);
                self.stats.record_trigger();
                let latency_ns = start.elapsed().as_nanos() as u64;
                self.stats.record_check(latency_ns);
                return Err(RiskError::InternalError(format!(
                    "Portfolio daily loss {:.2}% exceeds limit {:.2}%",
                    loss_pct, self.config.max_daily_loss_pct
                )));
            }
        }

        // Check strategy-specific limits
        self.check_strategy_limits(portfolio)?;

        let latency_ns = start.elapsed().as_nanos() as u64;
        self.stats.record_check(latency_ns);
        Ok(())
    }

    fn reset(&self) {
        self.stats.reset();
        self.status.store(SentinelStatus::Active as u8, Ordering::SeqCst);
        self.reset_global_kill();

        // Reset all strategies
        let mut strategies = self.strategies.write();
        for perf in strategies.values_mut() {
            perf.killed = false;
            perf.consecutive_losses = 0;
            perf.daily_pnl = 0.0;
        }
    }

    fn enable(&self) {
        self.status.store(SentinelStatus::Active as u8, Ordering::SeqCst);
    }

    fn disable(&self) {
        self.status.store(SentinelStatus::Disabled as u8, Ordering::SeqCst);
    }

    fn check_count(&self) -> u64 {
        self.stats.checks.load(Ordering::Relaxed)
    }

    fn trigger_count(&self) -> u64 {
        self.stats.triggers.load(Ordering::Relaxed)
    }

    fn avg_latency_ns(&self) -> u64 {
        self.stats.avg_latency_ns()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{OrderSide, Price, Quantity};

    fn create_test_order() -> Order {
        Order {
            symbol: Symbol::new("AAPL"),
            side: OrderSide::Buy,
            quantity: Quantity::from_f64(100.0),
            limit_price: Some(Price::from_f64(100.0)),
            strategy_id: 1,
            timestamp: Timestamp::now(),
        }
    }

    #[test]
    fn test_kill_switch_sentinel_creation() {
        let sentinel = StrategyKillSwitchSentinel::with_defaults();
        assert_eq!(sentinel.status(), SentinelStatus::Active);
        assert_eq!(sentinel.check_count(), 0);
        assert!(!sentinel.is_global_kill_active());
    }

    #[test]
    fn test_strategy_registration() {
        let sentinel = StrategyKillSwitchSentinel::with_defaults();

        sentinel.register_strategy("momentum_strategy", 100000.0);

        let perf = sentinel.get_strategy_performance("momentum_strategy");
        assert!(perf.is_some());
        assert_eq!(perf.unwrap().current_value, 100000.0);
    }

    #[test]
    fn test_global_kill_trigger() {
        let sentinel = StrategyKillSwitchSentinel::with_defaults();

        sentinel.trigger_global_kill(KillReason::ManualKill);

        assert!(sentinel.is_global_kill_active());

        let order = create_test_order();
        let portfolio = Portfolio::default();

        let result = sentinel.check(&order, &portfolio);
        assert!(result.is_err());
    }

    #[test]
    fn test_strategy_kill_on_consecutive_losses() {
        let mut config = StrategyKillSwitchConfig::default();
        config.max_consecutive_losses = 3;

        let sentinel = StrategyKillSwitchSentinel::new(config);
        sentinel.register_strategy("test_strategy", 100000.0);

        // Simulate consecutive losses
        sentinel.update_strategy_pnl("test_strategy", -100.0, 99900.0);
        sentinel.update_strategy_pnl("test_strategy", -100.0, 99800.0);
        sentinel.update_strategy_pnl("test_strategy", -100.0, 99700.0);

        let perf = sentinel.get_strategy_performance("test_strategy").unwrap();
        assert_eq!(perf.consecutive_losses, 3);
    }

    #[test]
    fn test_consecutive_losses_reset_on_win() {
        let sentinel = StrategyKillSwitchSentinel::with_defaults();
        sentinel.register_strategy("test_strategy", 100000.0);

        // Simulate losses then a win
        sentinel.update_strategy_pnl("test_strategy", -100.0, 99900.0);
        sentinel.update_strategy_pnl("test_strategy", -100.0, 99800.0);
        sentinel.update_strategy_pnl("test_strategy", 200.0, 100000.0); // Win

        let perf = sentinel.get_strategy_performance("test_strategy").unwrap();
        assert_eq!(perf.consecutive_losses, 0);
    }

    #[test]
    fn test_reset_clears_kills() {
        let sentinel = StrategyKillSwitchSentinel::with_defaults();

        sentinel.trigger_global_kill(KillReason::ManualKill);
        assert!(sentinel.is_global_kill_active());

        sentinel.reset();
        assert!(!sentinel.is_global_kill_active());
    }

    #[test]
    fn test_check_passes_when_healthy() {
        let sentinel = StrategyKillSwitchSentinel::with_defaults();

        let order = create_test_order();
        let portfolio = Portfolio::default();

        let result = sentinel.check(&order, &portfolio);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sentinel_lifecycle() {
        let sentinel = StrategyKillSwitchSentinel::with_defaults();

        assert_eq!(sentinel.status(), SentinelStatus::Active);

        sentinel.disable();
        assert_eq!(sentinel.status(), SentinelStatus::Disabled);

        sentinel.enable();
        assert_eq!(sentinel.status(), SentinelStatus::Active);

        sentinel.reset();
        assert_eq!(sentinel.status(), SentinelStatus::Active);
        assert_eq!(sentinel.check_count(), 0);
    }
}
