//! Portfolio Manager Agent.
//!
//! Orchestrates position management, rebalancing, and risk allocation
//! across the portfolio.
//!
//! ## Responsibilities
//! - Position sizing and allocation
//! - Rebalancing triggers
//! - Risk budget management
//! - Correlation-aware hedging
//!
//! ## Scientific References
//! - Black & Litterman (1992): "Global Portfolio Optimization"
//! - Roncalli (2013): "Risk Parity and Budgeting"

use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

use parking_lot::RwLock;

use crate::core::types::{MarketRegime, Portfolio, RiskDecision, RiskLevel, Symbol, Timestamp};
use crate::core::error::Result;
use super::base::{Agent, AgentId, AgentStatus, AgentConfig, AgentStats};

/// Portfolio Manager configuration.
#[derive(Debug, Clone)]
pub struct PortfolioManagerConfig {
    /// Base agent config.
    pub base: AgentConfig,
    /// Maximum position concentration (fraction of portfolio).
    pub max_concentration: f64,
    /// Minimum cash reserve (fraction of portfolio).
    pub min_cash_reserve: f64,
    /// Rebalance threshold (deviation from target).
    pub rebalance_threshold: f64,
    /// Use risk parity allocation.
    pub risk_parity: bool,
}

impl Default for PortfolioManagerConfig {
    fn default() -> Self {
        Self {
            base: AgentConfig {
                name: "PortfolioManager".to_string(),
                max_latency_us: 500,
                ..Default::default()
            },
            max_concentration: 0.25,    // 25% max in single asset
            min_cash_reserve: 0.05,     // 5% cash minimum
            rebalance_threshold: 0.10,  // 10% deviation triggers rebalance
            risk_parity: true,
        }
    }
}

/// Target allocation for an asset.
#[derive(Debug, Clone)]
pub struct TargetAllocation {
    /// Symbol.
    pub symbol: Symbol,
    /// Target weight (0.0 - 1.0).
    pub target_weight: f64,
    /// Current weight.
    pub current_weight: f64,
    /// Risk contribution (for risk parity).
    pub risk_contribution: f64,
}

/// Portfolio Manager Agent.
#[derive(Debug)]
pub struct PortfolioManagerAgent {
    /// Configuration.
    config: PortfolioManagerConfig,
    /// Current status (atomic for lock-free reads).
    status: AtomicU8,
    /// Target allocations.
    targets: RwLock<Vec<TargetAllocation>>,
    /// Last rebalance timestamp.
    #[allow(dead_code)]
    last_rebalance: RwLock<Timestamp>,
    /// Statistics.
    stats: AgentStats,
}

impl PortfolioManagerAgent {
    /// Create new portfolio manager agent.
    pub fn new(config: PortfolioManagerConfig) -> Self {
        Self {
            config,
            status: AtomicU8::new(AgentStatus::Idle as u8),
            targets: RwLock::new(Vec::new()),
            last_rebalance: RwLock::new(Timestamp::now()),
            stats: AgentStats::new(),
        }
    }

    /// Set target allocations.
    pub fn set_targets(&self, targets: Vec<TargetAllocation>) {
        let mut t = self.targets.write();
        *t = targets;
    }

    /// Check if rebalance is needed.
    pub fn needs_rebalance(&self, portfolio: &Portfolio) -> bool {
        let targets = self.targets.read();
        let total_value = portfolio.total_value;

        if total_value <= 0.0 {
            return false;
        }

        for target in targets.iter() {
            if let Some(pos) = portfolio.get_position(&target.symbol) {
                let current_weight = pos.market_value() / total_value;
                let deviation = (current_weight - target.target_weight).abs();
                if deviation > self.config.rebalance_threshold {
                    return true;
                }
            }
        }

        false
    }

    /// Calculate rebalance orders.
    pub fn calculate_rebalance(&self, portfolio: &Portfolio) -> Vec<RebalanceOrder> {
        let targets = self.targets.read();
        let total_value = portfolio.total_value;
        let mut orders = Vec::new();

        if total_value <= 0.0 {
            return orders;
        }

        for target in targets.iter() {
            let target_value = total_value * target.target_weight;

            let current_value = portfolio
                .get_position(&target.symbol)
                .map(|p| p.market_value())
                .unwrap_or(0.0);

            let diff = target_value - current_value;

            // Only create order if significant deviation
            if diff.abs() > total_value * 0.01 {
                orders.push(RebalanceOrder {
                    symbol: target.symbol,
                    target_value,
                    current_value,
                    delta_value: diff,
                });
            }
        }

        orders
    }

    /// Check concentration limits.
    pub fn check_concentration(&self, portfolio: &Portfolio) -> Option<RiskDecision> {
        let total_value = portfolio.total_value;

        if total_value <= 0.0 {
            return None;
        }

        for pos in &portfolio.positions {
            let weight = pos.market_value().abs() / total_value;
            if weight > self.config.max_concentration {
                return Some(RiskDecision::reject(
                    format!(
                        "Position {} exceeds concentration limit ({:.1}% > {:.1}%)",
                        pos.symbol,
                        weight * 100.0,
                        self.config.max_concentration * 100.0
                    ),
                    RiskLevel::High,
                    0,
                ));
            }
        }

        None
    }

    /// Check cash reserve.
    pub fn check_cash_reserve(&self, portfolio: &Portfolio) -> Option<RiskDecision> {
        let total_value = portfolio.total_value;

        if total_value <= 0.0 {
            return None;
        }

        let cash_ratio = portfolio.cash / total_value;
        if cash_ratio < self.config.min_cash_reserve {
            return Some(RiskDecision::reject(
                format!(
                    "Cash reserve below minimum ({:.1}% < {:.1}%)",
                    cash_ratio * 100.0,
                    self.config.min_cash_reserve * 100.0
                ),
                RiskLevel::Elevated,
                0,
            ));
        }

        None
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

/// Rebalance order suggestion.
#[derive(Debug, Clone)]
pub struct RebalanceOrder {
    /// Symbol to trade.
    pub symbol: Symbol,
    /// Target value.
    pub target_value: f64,
    /// Current value.
    pub current_value: f64,
    /// Delta (positive = buy, negative = sell).
    pub delta_value: f64,
}

impl Agent for PortfolioManagerAgent {
    fn id(&self) -> AgentId {
        AgentId::new(&self.config.base.name)
    }

    fn status(&self) -> AgentStatus {
        Self::status_from_u8(self.status.load(Ordering::Relaxed))
    }

    fn process(&self, portfolio: &Portfolio, regime: MarketRegime) -> Result<Option<RiskDecision>> {
        let start = Instant::now();
        self.status.store(AgentStatus::Processing as u8, Ordering::Relaxed);

        // Check concentration limits
        if let Some(decision) = self.check_concentration(portfolio) {
            self.stats.record_cycle(start.elapsed().as_nanos() as u64);
            self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);
            return Ok(Some(decision));
        }

        // Check cash reserve
        if let Some(decision) = self.check_cash_reserve(portfolio) {
            self.stats.record_cycle(start.elapsed().as_nanos() as u64);
            self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);
            return Ok(Some(decision));
        }

        // Adjust risk based on regime
        let regime_adjustment = regime.risk_multiplier();
        if regime_adjustment < 0.5 {
            // In crisis/high vol, suggest reducing exposure
            let decision = RiskDecision {
                allowed: true,
                risk_level: RiskLevel::Elevated,
                reason: format!("Regime {} suggests reduced exposure", regime_adjustment),
                size_adjustment: regime_adjustment,
                timestamp: Timestamp::now(),
                latency_ns: start.elapsed().as_nanos() as u64,
            };
            self.stats.record_cycle(start.elapsed().as_nanos() as u64);
            self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);
            return Ok(Some(decision));
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
    use crate::core::types::Portfolio;

    #[test]
    fn test_portfolio_manager_creation() {
        let config = PortfolioManagerConfig::default();
        let agent = PortfolioManagerAgent::new(config);
        assert_eq!(agent.status(), AgentStatus::Idle);
    }

    #[test]
    fn test_concentration_check() {
        let config = PortfolioManagerConfig {
            max_concentration: 0.20, // 20% max
            ..Default::default()
        };
        let agent = PortfolioManagerAgent::new(config);

        let mut portfolio = Portfolio::new(100_000.0);
        // Simulate a position that's 30% of portfolio
        portfolio.positions.push(crate::core::types::Position {
            id: crate::core::types::PositionId::new(),
            symbol: Symbol::new("AAPL"),
            quantity: crate::core::types::Quantity::from_f64(100.0),
            avg_entry_price: crate::core::types::Price::from_f64(300.0),
            current_price: crate::core::types::Price::from_f64(300.0),
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            opened_at: Timestamp::now(),
            updated_at: Timestamp::now(),
        });
        portfolio.recalculate();

        let decision = agent.check_concentration(&portfolio);
        assert!(decision.is_some());
        assert!(!decision.unwrap().allowed);
    }
}
