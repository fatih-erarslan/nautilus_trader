//! LMSR-RS: High-Performance Logarithmic Market Scoring Rule
//! 
//! This crate provides a numerically stable, thread-safe implementation of the
//! Logarithmic Market Scoring Rule (LMSR) for prediction markets and market making.
//! 
//! Key features:
//! - Numerical stability for extreme market conditions
//! - Thread-safe market state management
//! - PyO3 bindings for Python integration (optional)
//! - High-precision financial calculations
//! - Real-time market updates

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

pub mod market;
pub mod lmsr;
pub mod errors;
pub mod utils;

// Focus on core LMSR functionality only

#[cfg(feature = "simd")]
pub mod simd;

#[cfg(feature = "high-performance")]
pub mod performance;

pub mod performance_simple;

#[cfg(feature = "pyo3")]
pub mod python_bindings_simple;

pub use market::{Market, MarketState, Position};
pub use lmsr::{LMSRMarketMaker, LMSRCalculator};
pub use errors::{LMSRError, Result};

#[cfg(feature = "pyo3")]
use python_bindings_simple::*;

/// Initialize the Python module
#[cfg(feature = "pyo3")]
#[pymodule]
fn lmsr_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLMSRMarket>()?;
    m.add_class::<PyMarketSimulation>()?;
    m.add_function(wrap_pyfunction!(py_calculate_price, m)?)?;
    m.add_function(wrap_pyfunction!(py_calculate_cost, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_basic_functionality() {
        let market = Market::new(2, 1000.0).unwrap();
        let price = market.get_price(0).unwrap();
        assert!(price > 0.0 && price < 1.0);
    }

    #[test]
    fn test_numerical_stability() {
        // Test extreme cases that might cause numerical instability
        let mut market = Market::new(2, 1e6).unwrap();
        
        // Large quantity trade
        let cost = market.calculate_trade_cost(&[1e5, 0.0]).unwrap();
        assert!(cost.is_finite());
        
        // Very small quantity trade
        let cost = market.calculate_trade_cost(&[1e-10, 0.0]).unwrap();
        assert!(cost.is_finite());
    }
}