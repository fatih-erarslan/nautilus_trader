/// Finance system integration module

use crate::types::{L2Snapshot, FinanceError};
use crate::orderbook::{OrderBookState, OrderBookConfig};
use crate::risk::{RiskEngine, RiskConfig, RiskMetrics};

/// Main finance system configuration
#[derive(Debug, Clone, Default)]
pub struct FinanceConfig {
    pub orderbook: OrderBookConfig,
    pub risk: RiskConfig,
}

/// Integrated finance system combining order book and risk analytics
pub struct FinanceSystem {
    #[allow(dead_code)] // Reserved for future configuration updates
    config: FinanceConfig,
    orderbook_state: Option<OrderBookState>,
    risk_engine: RiskEngine,
}

impl FinanceSystem {
    /// Create new finance system
    pub fn new(config: FinanceConfig) -> Self {
        Self {
            risk_engine: RiskEngine::new(config.risk.clone()),
            config,
            orderbook_state: None,
        }
    }

    /// Create with default configuration
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(FinanceConfig::default())
    }
}

impl Default for FinanceSystem {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl FinanceSystem {

    /// Process order book snapshot
    ///
    /// This updates both order book analytics and risk metrics.
    pub fn process_snapshot(&mut self, snapshot: L2Snapshot) -> Result<(), FinanceError> {
        // Update order book state
        match &mut self.orderbook_state {
            Some(state) => state.update(snapshot.clone())?,
            None => {
                self.orderbook_state = Some(OrderBookState::from_snapshot(snapshot.clone())?);
            }
        }

        // Update risk engine
        self.risk_engine.update_from_snapshot(&snapshot)?;

        Ok(())
    }

    /// Get current order book state
    pub fn orderbook_state(&self) -> Option<&OrderBookState> {
        self.orderbook_state.as_ref()
    }

    /// Get risk engine
    pub fn risk_engine(&self) -> &RiskEngine {
        &self.risk_engine
    }

    /// Calculate current risk metrics
    pub fn calculate_risk_metrics(&self) -> Result<RiskMetrics, FinanceError> {
        self.risk_engine.calculate_metrics()
    }

    /// Get current mid-price
    pub fn current_price(&self) -> Option<f64> {
        self.orderbook_state
            .as_ref()
            .map(|state| state.mid_price())
    }

    /// Get current spread
    pub fn current_spread(&self) -> Option<f64> {
        self.orderbook_state
            .as_ref()
            .map(|state| state.spread())
    }

    /// Get order imbalance
    pub fn order_imbalance(&self) -> Option<f64> {
        self.orderbook_state
            .as_ref()
            .map(|state| state.order_imbalance())
    }

    /// Reset system state
    pub fn reset(&mut self) {
        self.orderbook_state = None;
        self.risk_engine.clear_history();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Price, Quantity};

    fn create_test_snapshot(mid_price: f64) -> L2Snapshot {
        L2Snapshot {
            symbol: "BTC-USD".to_string(),
            timestamp_us: 1000000,
            bids: vec![
                (Price::new(mid_price - 0.5).unwrap(), Quantity::new(1.0).unwrap()),
            ],
            asks: vec![
                (Price::new(mid_price + 0.5).unwrap(), Quantity::new(1.0).unwrap()),
            ],
        }
    }

    #[test]
    fn test_finance_system_creation() {
        let system = FinanceSystem::default();
        assert!(system.orderbook_state().is_none());
        assert!(system.current_price().is_none());
    }

    #[test]
    fn test_process_snapshot() {
        let mut system = FinanceSystem::with_defaults();
        let snapshot = create_test_snapshot(100.0);

        system.process_snapshot(snapshot).unwrap();

        assert!(system.orderbook_state().is_some());
        assert_eq!(system.current_price().unwrap(), 100.0);
        assert_eq!(system.current_spread().unwrap(), 1.0);
    }

    #[test]
    fn test_multiple_snapshots() {
        let mut system = FinanceSystem::with_defaults();

        for i in 0..50 {
            let price = 100.0 + (i as f64 * 0.1);
            let snapshot = create_test_snapshot(price);
            system.process_snapshot(snapshot).unwrap();
        }

        // Should have metrics after sufficient data
        assert!(system.current_price().is_some());
        assert!(system.risk_engine().history_size() == 50);
    }

    #[test]
    fn test_reset() {
        let mut system = FinanceSystem::with_defaults();
        let snapshot = create_test_snapshot(100.0);

        system.process_snapshot(snapshot).unwrap();
        assert!(system.orderbook_state().is_some());

        system.reset();
        assert!(system.orderbook_state().is_none());
        assert_eq!(system.risk_engine().history_size(), 0);
    }
}
