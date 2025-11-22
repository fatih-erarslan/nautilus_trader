/// HyperPhysics Finance - Production-grade financial modeling
///
/// This crate provides peer-reviewed implementations of financial models
/// used in quantitative finance and risk management.
///
/// # Modules
///
/// - `types`: Core financial data types (Price, Quantity, L2Snapshot)
/// - `risk`: Risk analytics (Black-Scholes Greeks, VaR models)
/// - `orderbook`: Order book analytics and market microstructure
/// - `system`: Integrated finance system
///
/// # Example
///
/// ```rust
/// use hyperphysics_finance::{FinanceSystem, types::*};
///
/// let mut system = FinanceSystem::default();
///
/// let snapshot = L2Snapshot {
///     symbol: "BTC-USD".to_string(),
///     timestamp_us: 1000000,
///     bids: vec![
///         (Price::new(100.0).unwrap(), Quantity::new(1.0).unwrap()),
///     ],
///     asks: vec![
///         (Price::new(101.0).unwrap(), Quantity::new(1.0).unwrap()),
///     ],
/// };
///
/// system.process_snapshot(snapshot).unwrap();
/// ```
pub mod types;
pub mod risk;
pub mod orderbook;
pub mod system;

// Re-export main types for convenience
pub use types::{Price, Quantity, L2Snapshot, FinanceError};
pub use risk::{
    OptionParams, Greeks, RiskMetrics, RiskEngine, RiskConfig,
    VarModel, GarchParams,
    calculate_black_scholes, calculate_put_greeks,
    calculate_var, historical_var, garch_var, ewma_var,
};
pub use orderbook::{OrderBookState, OrderBookConfig};
pub use system::{FinanceSystem, FinanceConfig};

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_system_integration() {
        let mut system = FinanceSystem::with_defaults();

        // Create test data with volatility (not just trending up)
        for i in 0..100 {
            let volatility = (i as f64 * 0.1).sin() * 2.0;  // Add some variation
            let price = 100.0 + (i as f64 * 0.05) + volatility;
            let snapshot = L2Snapshot {
                symbol: "BTC-USD".to_string(),
                timestamp_us: (1000000 + i * 1000) as u64,
                bids: vec![
                    (Price::new(price - 0.5).unwrap(), Quantity::new(1.0).unwrap()),
                ],
                asks: vec![
                    (Price::new(price + 0.5).unwrap(), Quantity::new(1.0).unwrap()),
                ],
            };

            system.process_snapshot(snapshot).unwrap();
        }

        // Verify system state
        assert!(system.current_price().is_some());
        assert!(system.current_spread().is_some());

        // Calculate risk metrics
        let metrics = system.calculate_risk_metrics().unwrap();
        assert!(metrics.volatility > 0.0);
        assert!(metrics.var_95 > 0.0);
        assert!(metrics.var_99 >= metrics.var_95);  // >= allows for equality with small samples
    }

    #[test]
    fn test_greeks_calculation() {
        let params = OptionParams {
            spot: 100.0,
            strike: 100.0,
            rate: 0.05,
            volatility: 0.20,
            time_to_maturity: 1.0,
        };

        let (call_price, greeks) = calculate_black_scholes(&params).unwrap();

        assert!(call_price > 0.0);
        assert!(greeks.delta > 0.0 && greeks.delta < 1.0);
        assert!(greeks.gamma > 0.0);
        assert!(greeks.vega > 0.0);
        assert!(greeks.theta < 0.0);  // Time decay
        assert!(greeks.rho > 0.0);
    }
}
