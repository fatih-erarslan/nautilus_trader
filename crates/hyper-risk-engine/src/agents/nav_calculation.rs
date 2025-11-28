//! NAV (Net Asset Value) calculation agent.
//!
//! Operates in the medium path (<1ms) to compute accurate portfolio
//! valuations including all positions, cash, and adjustments.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

use parking_lot::RwLock;

use crate::core::types::{
    MarketRegime, Portfolio, Price, RiskDecision, Symbol, Timestamp,
};
use crate::core::error::Result;

use super::base::{Agent, AgentConfig, AgentId, AgentStats, AgentStatus};

/// Configuration for the NAV calculation agent.
#[derive(Debug, Clone)]
pub struct NAVCalculationConfig {
    /// Base agent configuration.
    pub base: AgentConfig,
    /// Base currency for NAV calculation.
    pub base_currency: String,
    /// Include accrued interest in NAV.
    pub include_accrued_interest: bool,
    /// Include pending settlements.
    pub include_pending_settlements: bool,
    /// Valuation precision (decimal places).
    pub precision: u32,
}

impl Default for NAVCalculationConfig {
    fn default() -> Self {
        Self {
            base: AgentConfig {
                name: "nav_calculation_agent".to_string(),
                enabled: true,
                priority: 2,
                max_latency_us: 1000, // 1ms
                verbose: false,
            },
            base_currency: "USD".to_string(),
            include_accrued_interest: true,
            include_pending_settlements: true,
            precision: 2,
        }
    }
}

/// NAV component breakdown.
#[derive(Debug, Clone)]
pub struct NAVComponent {
    /// Component name.
    pub name: String,
    /// Component value.
    pub value: f64,
    /// Percentage of total NAV.
    pub percentage: f64,
}

/// Position valuation detail.
#[derive(Debug, Clone)]
pub struct PositionValuation {
    /// Symbol.
    pub symbol: Symbol,
    /// Quantity.
    pub quantity: f64,
    /// Mark price used.
    pub mark_price: Price,
    /// Market value.
    pub market_value: f64,
    /// Unrealized P&L.
    pub unrealized_pnl: f64,
    /// Percentage of NAV.
    pub nav_percentage: f64,
}

/// Complete NAV calculation result.
#[derive(Debug, Clone)]
pub struct NAVResult {
    /// Total NAV.
    pub total_nav: f64,
    /// NAV per share (if applicable).
    pub nav_per_share: Option<f64>,
    /// Cash component.
    pub cash: f64,
    /// Securities value.
    pub securities_value: f64,
    /// Total unrealized P&L.
    pub total_unrealized_pnl: f64,
    /// Total realized P&L.
    pub total_realized_pnl: f64,
    /// Individual position valuations.
    pub positions: Vec<PositionValuation>,
    /// NAV components breakdown.
    pub components: Vec<NAVComponent>,
    /// Calculation timestamp.
    pub calculated_at: Timestamp,
    /// Base currency.
    pub currency: String,
}

/// NAV calculation agent.
#[derive(Debug)]
pub struct NAVCalculationAgent {
    config: NAVCalculationConfig,
    status: AtomicU8,
    stats: AgentStats,
    /// Total shares outstanding (for NAV per share).
    shares_outstanding: RwLock<Option<f64>>,
    /// Currency exchange rates.
    fx_rates: RwLock<HashMap<String, f64>>,
    /// Latest NAV result.
    latest_nav: RwLock<Option<NAVResult>>,
    /// Historical NAV values.
    nav_history: RwLock<Vec<(Timestamp, f64)>>,
}

impl NAVCalculationAgent {
    /// Create a new NAV calculation agent.
    pub fn new(config: NAVCalculationConfig) -> Self {
        Self {
            config,
            status: AtomicU8::new(AgentStatus::Idle as u8),
            stats: AgentStats::new(),
            shares_outstanding: RwLock::new(None),
            fx_rates: RwLock::new(HashMap::new()),
            latest_nav: RwLock::new(None),
            nav_history: RwLock::new(Vec::new()),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(NAVCalculationConfig::default())
    }

    /// Set shares outstanding for NAV per share calculation.
    pub fn set_shares_outstanding(&self, shares: f64) {
        *self.shares_outstanding.write() = Some(shares);
    }

    /// Set FX rate for currency conversion.
    pub fn set_fx_rate(&self, currency: String, rate: f64) {
        self.fx_rates.write().insert(currency, rate);
    }

    /// Get latest NAV result.
    pub fn get_nav(&self) -> Option<NAVResult> {
        self.latest_nav.read().clone()
    }

    /// Get NAV history.
    pub fn get_history(&self) -> Vec<(Timestamp, f64)> {
        self.nav_history.read().clone()
    }

    /// Calculate NAV from portfolio.
    fn calculate_nav(&self, portfolio: &Portfolio) -> NAVResult {
        let mut position_valuations = Vec::new();
        let mut total_securities_value = 0.0;
        let mut total_unrealized_pnl = 0.0;
        let mut total_realized_pnl = 0.0;

        // Calculate each position's value
        for position in &portfolio.positions {
            let market_value = position.market_value();
            let unrealized = position.unrealized_pnl;
            let realized = position.realized_pnl;

            total_securities_value += market_value;
            total_unrealized_pnl += unrealized;
            total_realized_pnl += realized;

            position_valuations.push(PositionValuation {
                symbol: position.symbol.clone(),
                quantity: position.quantity.as_f64(),
                mark_price: position.current_price,
                market_value,
                unrealized_pnl: unrealized,
                nav_percentage: 0.0, // Will be calculated after total NAV
            });
        }

        let cash = portfolio.cash;
        let total_nav = cash + total_securities_value;

        // Update NAV percentages
        for pv in &mut position_valuations {
            pv.nav_percentage = if total_nav > 0.0 {
                (pv.market_value / total_nav) * 100.0
            } else {
                0.0
            };
        }

        // Build components breakdown
        let mut components = vec![
            NAVComponent {
                name: "Cash".to_string(),
                value: cash,
                percentage: if total_nav > 0.0 { (cash / total_nav) * 100.0 } else { 0.0 },
            },
            NAVComponent {
                name: "Securities".to_string(),
                value: total_securities_value,
                percentage: if total_nav > 0.0 { (total_securities_value / total_nav) * 100.0 } else { 0.0 },
            },
        ];

        if total_unrealized_pnl.abs() > 0.0 {
            components.push(NAVComponent {
                name: "Unrealized P&L".to_string(),
                value: total_unrealized_pnl,
                percentage: if total_nav > 0.0 { (total_unrealized_pnl / total_nav) * 100.0 } else { 0.0 },
            });
        }

        // Calculate NAV per share if applicable
        let nav_per_share = self.shares_outstanding.read().map(|shares| {
            if shares > 0.0 { total_nav / shares } else { 0.0 }
        });

        // Round to configured precision
        let multiplier = 10_f64.powi(self.config.precision as i32);
        let rounded_nav = (total_nav * multiplier).round() / multiplier;

        NAVResult {
            total_nav: rounded_nav,
            nav_per_share,
            cash,
            securities_value: total_securities_value,
            total_unrealized_pnl,
            total_realized_pnl,
            positions: position_valuations,
            components,
            calculated_at: Timestamp::now(),
            currency: self.config.base_currency.clone(),
        }
    }

    /// Convert u8 to AgentStatus.
    fn status_from_u8(value: u8) -> AgentStatus {
        match value {
            0 => AgentStatus::Idle,
            1 => AgentStatus::Processing,
            2 => AgentStatus::Paused,
            3 => AgentStatus::Error,
            4 => AgentStatus::ShuttingDown,
            _ => AgentStatus::Error,
        }
    }
}

impl Agent for NAVCalculationAgent {
    fn id(&self) -> AgentId {
        AgentId::new(&self.config.base.name)
    }

    fn status(&self) -> AgentStatus {
        Self::status_from_u8(self.status.load(Ordering::Relaxed))
    }

    fn process(&self, portfolio: &Portfolio, _regime: MarketRegime) -> Result<Option<RiskDecision>> {
        let start = Instant::now();
        self.status.store(AgentStatus::Processing as u8, Ordering::Relaxed);

        // Calculate NAV
        let nav_result = self.calculate_nav(portfolio);

        // Store in history
        {
            let mut history = self.nav_history.write();
            history.push((nav_result.calculated_at, nav_result.total_nav));
            // Keep last 1000 entries
            if history.len() > 1000 {
                history.remove(0);
            }
        }

        *self.latest_nav.write() = Some(nav_result);

        let latency_ns = start.elapsed().as_nanos() as u64;
        self.stats.record_cycle(latency_ns);
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
    use crate::core::types::{Position, PositionId, Quantity};

    #[test]
    fn test_nav_calculation_agent_creation() {
        let agent = NAVCalculationAgent::with_defaults();
        assert_eq!(agent.status(), AgentStatus::Idle);
        assert_eq!(agent.process_count(), 0);
    }

    #[test]
    fn test_nav_calculation() {
        let agent = NAVCalculationAgent::with_defaults();
        agent.start().unwrap();

        let mut portfolio = Portfolio::new(10000.0);

        let symbol = Symbol::new("AAPL");
        portfolio.positions.push(Position {
            id: PositionId::new(),
            symbol,
            quantity: Quantity::from_f64(100.0),
            avg_entry_price: Price::from_f64(150.0),
            current_price: Price::from_f64(160.0),
            unrealized_pnl: 1000.0,
            realized_pnl: 0.0,
            opened_at: Timestamp::now(),
            updated_at: Timestamp::now(),
        });
        portfolio.recalculate();

        agent.process(&portfolio, MarketRegime::SidewaysLow).unwrap();

        let nav = agent.get_nav().unwrap();
        // Cash (10000) + Securities (100 * 160 = 16000) = 26000
        assert!((nav.total_nav - 26000.0).abs() < 0.01);
        assert!((nav.cash - 10000.0).abs() < 0.01);
        assert!((nav.securities_value - 16000.0).abs() < 0.01);
    }

    #[test]
    fn test_nav_per_share() {
        let agent = NAVCalculationAgent::with_defaults();
        agent.start().unwrap();
        agent.set_shares_outstanding(1000.0);

        let portfolio = Portfolio::new(100000.0);

        agent.process(&portfolio, MarketRegime::SidewaysLow).unwrap();

        let nav = agent.get_nav().unwrap();
        assert!(nav.nav_per_share.is_some());
        assert!((nav.nav_per_share.unwrap() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_nav_history() {
        let agent = NAVCalculationAgent::with_defaults();
        agent.start().unwrap();

        let portfolio = Portfolio::new(10000.0);

        // Process multiple times
        for _ in 0..5 {
            agent.process(&portfolio, MarketRegime::SidewaysLow).unwrap();
        }

        let history = agent.get_history();
        assert_eq!(history.len(), 5);
    }

    #[test]
    fn test_agent_lifecycle() {
        let agent = NAVCalculationAgent::with_defaults();

        agent.start().unwrap();
        assert_eq!(agent.status(), AgentStatus::Idle);

        agent.pause();
        assert_eq!(agent.status(), AgentStatus::Paused);

        agent.resume();
        assert_eq!(agent.status(), AgentStatus::Idle);

        agent.stop().unwrap();
        assert_eq!(agent.status(), AgentStatus::ShuttingDown);
    }
}
