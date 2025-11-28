//! Reconciliation agent for position and balance verification.
//!
//! Operates in the slow path to reconcile internal state with
//! external sources (exchanges, prime brokers, custodians).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

use parking_lot::RwLock;

use crate::core::types::{
    MarketRegime, Portfolio, Price, Quantity, RiskDecision, Symbol, Timestamp,
};
use crate::core::error::Result;

use super::base::{Agent, AgentConfig, AgentId, AgentStats, AgentStatus};

/// Configuration for the reconciliation agent.
#[derive(Debug, Clone)]
pub struct ReconciliationConfig {
    /// Base agent configuration.
    pub base: AgentConfig,
    /// Tolerance for quantity mismatches.
    pub quantity_tolerance: f64,
    /// Tolerance for price mismatches (as percentage).
    pub price_tolerance_pct: f64,
    /// Tolerance for cash balance mismatches.
    pub cash_tolerance: f64,
    /// Auto-correct minor discrepancies.
    pub auto_correct: bool,
}

impl Default for ReconciliationConfig {
    fn default() -> Self {
        Self {
            base: AgentConfig {
                name: "reconciliation_agent".to_string(),
                enabled: true,
                priority: 4,
                max_latency_us: 10_000, // 10ms (slow path)
                verbose: false,
            },
            quantity_tolerance: 0.0001,
            price_tolerance_pct: 0.01,
            cash_tolerance: 0.01,
            auto_correct: false,
        }
    }
}

/// External position record for reconciliation.
#[derive(Debug, Clone)]
pub struct ExternalPosition {
    /// Symbol.
    pub symbol: Symbol,
    /// Quantity from external source.
    pub quantity: Quantity,
    /// Average price from external source.
    pub avg_price: Price,
    /// Source identifier.
    pub source: String,
    /// Timestamp of external record.
    pub as_of: Timestamp,
}

/// External cash balance record.
#[derive(Debug, Clone)]
pub struct ExternalCashBalance {
    /// Currency.
    pub currency: String,
    /// Balance amount.
    pub balance: f64,
    /// Source identifier.
    pub source: String,
    /// Timestamp.
    pub as_of: Timestamp,
}

/// Type of reconciliation discrepancy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscrepancyType {
    /// Position quantity mismatch.
    QuantityMismatch,
    /// Position price mismatch.
    PriceMismatch,
    /// Missing internal position.
    MissingInternal,
    /// Missing external position.
    MissingExternal,
    /// Cash balance mismatch.
    CashMismatch,
}

/// Reconciliation discrepancy.
#[derive(Debug, Clone)]
pub struct Discrepancy {
    /// Type of discrepancy.
    pub discrepancy_type: DiscrepancyType,
    /// Symbol (if position-related).
    pub symbol: Option<Symbol>,
    /// Currency (if cash-related).
    pub currency: Option<String>,
    /// Internal value.
    pub internal_value: f64,
    /// External value.
    pub external_value: f64,
    /// Difference.
    pub difference: f64,
    /// External source.
    pub source: String,
    /// Detection timestamp.
    pub detected_at: Timestamp,
    /// Severity (0.0 to 1.0).
    pub severity: f64,
}

/// Reconciliation result.
#[derive(Debug, Clone)]
pub struct ReconciliationResult {
    /// Whether reconciliation passed.
    pub passed: bool,
    /// List of discrepancies found.
    pub discrepancies: Vec<Discrepancy>,
    /// Positions reconciled count.
    pub positions_checked: usize,
    /// Cash balances reconciled count.
    pub balances_checked: usize,
    /// Reconciliation timestamp.
    pub completed_at: Timestamp,
}

/// Reconciliation agent.
#[derive(Debug)]
pub struct ReconciliationAgent {
    config: ReconciliationConfig,
    status: AtomicU8,
    stats: AgentStats,
    /// External positions by source and symbol.
    external_positions: RwLock<HashMap<String, HashMap<Symbol, ExternalPosition>>>,
    /// External cash balances by source.
    external_balances: RwLock<HashMap<String, Vec<ExternalCashBalance>>>,
    /// Latest reconciliation result.
    latest_result: RwLock<Option<ReconciliationResult>>,
}

impl ReconciliationAgent {
    /// Create a new reconciliation agent.
    pub fn new(config: ReconciliationConfig) -> Self {
        Self {
            config,
            status: AtomicU8::new(AgentStatus::Idle as u8),
            stats: AgentStats::new(),
            external_positions: RwLock::new(HashMap::new()),
            external_balances: RwLock::new(HashMap::new()),
            latest_result: RwLock::new(None),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(ReconciliationConfig::default())
    }

    /// Load external position for reconciliation.
    pub fn load_external_position(&self, position: ExternalPosition) {
        let mut positions = self.external_positions.write();
        positions
            .entry(position.source.clone())
            .or_default()
            .insert(position.symbol.clone(), position);
    }

    /// Load external cash balance for reconciliation.
    pub fn load_external_balance(&self, balance: ExternalCashBalance) {
        self.external_balances
            .write()
            .entry(balance.source.clone())
            .or_default()
            .push(balance);
    }

    /// Get latest reconciliation result.
    pub fn get_result(&self) -> Option<ReconciliationResult> {
        self.latest_result.read().clone()
    }

    /// Perform reconciliation against portfolio.
    fn reconcile(&self, portfolio: &Portfolio) -> ReconciliationResult {
        let mut discrepancies = Vec::new();
        let mut positions_checked = 0;
        let mut balances_checked = 0;

        let external_positions = self.external_positions.read();
        let external_balances = self.external_balances.read();

        // Reconcile positions
        for (source, source_positions) in external_positions.iter() {
            for (symbol, external) in source_positions.iter() {
                positions_checked += 1;

                if let Some(internal) = portfolio.get_position(symbol) {
                    // Check quantity
                    let internal_qty = internal.quantity.as_f64();
                    let external_qty = external.quantity.as_f64();
                    let qty_diff = (internal_qty - external_qty).abs();

                    if qty_diff > self.config.quantity_tolerance {
                        discrepancies.push(Discrepancy {
                            discrepancy_type: DiscrepancyType::QuantityMismatch,
                            symbol: Some(symbol.clone()),
                            currency: None,
                            internal_value: internal_qty,
                            external_value: external_qty,
                            difference: qty_diff,
                            source: source.clone(),
                            detected_at: Timestamp::now(),
                            severity: (qty_diff / external_qty.abs().max(1.0)).min(1.0),
                        });
                    }

                    // Check price
                    let internal_price = internal.avg_entry_price.as_f64();
                    let external_price = external.avg_price.as_f64();
                    let price_diff_pct = if external_price > 0.0 {
                        ((internal_price - external_price) / external_price).abs() * 100.0
                    } else {
                        0.0
                    };

                    if price_diff_pct > self.config.price_tolerance_pct {
                        discrepancies.push(Discrepancy {
                            discrepancy_type: DiscrepancyType::PriceMismatch,
                            symbol: Some(symbol.clone()),
                            currency: None,
                            internal_value: internal_price,
                            external_value: external_price,
                            difference: price_diff_pct,
                            source: source.clone(),
                            detected_at: Timestamp::now(),
                            severity: (price_diff_pct / 10.0).min(1.0),
                        });
                    }
                } else {
                    // Missing internal position
                    discrepancies.push(Discrepancy {
                        discrepancy_type: DiscrepancyType::MissingInternal,
                        symbol: Some(symbol.clone()),
                        currency: None,
                        internal_value: 0.0,
                        external_value: external.quantity.as_f64(),
                        difference: external.quantity.as_f64(),
                        source: source.clone(),
                        detected_at: Timestamp::now(),
                        severity: 1.0,
                    });
                }
            }

            // Check for positions in portfolio but not in external
            for position in &portfolio.positions {
                if !source_positions.contains_key(&position.symbol) {
                    discrepancies.push(Discrepancy {
                        discrepancy_type: DiscrepancyType::MissingExternal,
                        symbol: Some(position.symbol.clone()),
                        currency: None,
                        internal_value: position.quantity.as_f64(),
                        external_value: 0.0,
                        difference: position.quantity.as_f64(),
                        source: source.clone(),
                        detected_at: Timestamp::now(),
                        severity: 1.0,
                    });
                }
            }
        }

        // Reconcile cash balances
        let internal_cash = portfolio.cash;
        for (_source, balances) in external_balances.iter() {
            for balance in balances.iter() {
                balances_checked += 1;
                let diff = (internal_cash - balance.balance).abs();

                if diff > self.config.cash_tolerance {
                    discrepancies.push(Discrepancy {
                        discrepancy_type: DiscrepancyType::CashMismatch,
                        symbol: None,
                        currency: Some(balance.currency.clone()),
                        internal_value: internal_cash,
                        external_value: balance.balance,
                        difference: diff,
                        source: balance.source.clone(),
                        detected_at: Timestamp::now(),
                        severity: (diff / internal_cash.abs().max(1.0)).min(1.0),
                    });
                }
            }
        }

        ReconciliationResult {
            passed: discrepancies.is_empty(),
            discrepancies,
            positions_checked,
            balances_checked,
            completed_at: Timestamp::now(),
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

impl Agent for ReconciliationAgent {
    fn id(&self) -> AgentId {
        AgentId::new(&self.config.base.name)
    }

    fn status(&self) -> AgentStatus {
        Self::status_from_u8(self.status.load(Ordering::Relaxed))
    }

    fn process(&self, portfolio: &Portfolio, _regime: MarketRegime) -> Result<Option<RiskDecision>> {
        let start = Instant::now();
        self.status.store(AgentStatus::Processing as u8, Ordering::Relaxed);

        // Perform reconciliation
        let result = self.reconcile(portfolio);
        *self.latest_result.write() = Some(result);

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
    use crate::core::types::{Position, PositionId};

    #[test]
    fn test_reconciliation_agent_creation() {
        let agent = ReconciliationAgent::with_defaults();
        assert_eq!(agent.status(), AgentStatus::Idle);
        assert_eq!(agent.process_count(), 0);
    }

    #[test]
    fn test_reconciliation_pass() {
        let agent = ReconciliationAgent::with_defaults();
        agent.start().unwrap();

        let symbol = Symbol::new("AAPL");

        // Create matching internal and external positions
        let mut portfolio = Portfolio::new(10000.0);
        portfolio.positions.push(Position {
            id: PositionId::new(),
            symbol: symbol.clone(),
            quantity: Quantity::from_f64(100.0),
            avg_entry_price: Price::from_f64(150.0),
            current_price: Price::from_f64(155.0),
            unrealized_pnl: 500.0,
            realized_pnl: 0.0,
            opened_at: Timestamp::now(),
            updated_at: Timestamp::now(),
        });

        agent.load_external_position(ExternalPosition {
            symbol: symbol.clone(),
            quantity: Quantity::from_f64(100.0),
            avg_price: Price::from_f64(150.0),
            source: "broker".to_string(),
            as_of: Timestamp::now(),
        });

        agent.process(&portfolio, MarketRegime::SidewaysLow).unwrap();

        let result = agent.get_result().unwrap();
        assert!(result.passed);
        assert!(result.discrepancies.is_empty());
    }

    #[test]
    fn test_reconciliation_quantity_mismatch() {
        let agent = ReconciliationAgent::with_defaults();
        agent.start().unwrap();

        let symbol = Symbol::new("AAPL");

        let mut portfolio = Portfolio::new(10000.0);
        portfolio.positions.push(Position {
            id: PositionId::new(),
            symbol: symbol.clone(),
            quantity: Quantity::from_f64(100.0),
            avg_entry_price: Price::from_f64(150.0),
            current_price: Price::from_f64(155.0),
            unrealized_pnl: 500.0,
            realized_pnl: 0.0,
            opened_at: Timestamp::now(),
            updated_at: Timestamp::now(),
        });

        // External shows different quantity
        agent.load_external_position(ExternalPosition {
            symbol: symbol.clone(),
            quantity: Quantity::from_f64(110.0),
            avg_price: Price::from_f64(150.0),
            source: "broker".to_string(),
            as_of: Timestamp::now(),
        });

        agent.process(&portfolio, MarketRegime::SidewaysLow).unwrap();

        let result = agent.get_result().unwrap();
        assert!(!result.passed);
        assert!(!result.discrepancies.is_empty());
        assert_eq!(result.discrepancies[0].discrepancy_type, DiscrepancyType::QuantityMismatch);
    }

    #[test]
    fn test_agent_lifecycle() {
        let agent = ReconciliationAgent::with_defaults();

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
