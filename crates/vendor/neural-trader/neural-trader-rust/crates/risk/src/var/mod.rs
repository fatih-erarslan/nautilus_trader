//! Value at Risk (VaR) and Conditional VaR (CVaR) calculations
//!
//! This module provides multiple methods for calculating VaR:
//! - Monte Carlo simulation (primary method, GPU-accelerated)
//! - Historical VaR
//! - Parametric VaR (variance-covariance)
//!
//! ## Example
//!
//! ```rust,no_run
//! use nt_risk::var::{MonteCarloVaR, VaRConfig};
//! use nt_risk::types::Portfolio;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let _config = VaRConfig {
//!         confidence_level: 0.95,
//!         time_horizon_days: 1,
//!         num_simulations: 10_000,
//!         use_gpu: false,
//!     };
//!
//!     let var_calculator = MonteCarloVaR::new(config);
//!     let portfolio = Portfolio::new(rust_decimal_macros::dec!(100000));
//!
//!     let result = var_calculator.calculate_portfolio(&portfolio).await?;
//!     println!("VaR (95%): ${:.2}", result.var_95);
//!     println!("CVaR (95%): ${:.2}", result.cvar_95);
//!
//!     Ok(())
//! }
//! ```

pub mod monte_carlo;
pub mod historical;
pub mod parametric;

pub use monte_carlo::{MonteCarloVaR, VaRConfig};
pub use historical::HistoricalVaR;
pub use parametric::ParametricVaR;

use async_trait::async_trait;
use crate::Result;
use crate::types::{Portfolio, VaRResult};

/// Trait for VaR calculation methods
#[async_trait]
pub trait VaRCalculator: Send + Sync {
    /// Calculate VaR and CVaR for a portfolio
    async fn calculate_portfolio(&self, portfolio: &Portfolio) -> Result<VaRResult>;

    /// Get the name of the calculation method
    fn method_name(&self) -> &str;

    /// Get configuration for this calculator
    fn config(&self) -> VaRConfigInfo;
}

/// VaR calculator configuration information
#[derive(Debug, Clone)]
pub struct VaRConfigInfo {
    pub confidence_level: f64,
    pub time_horizon_days: usize,
    pub method: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_var_config_info() {
        let _config = VaRConfigInfo {
            confidence_level: 0.95,
            time_horizon_days: 1,
            method: "Monte Carlo".to_string(),
        };

        assert_eq!(config.confidence_level, 0.95);
        assert_eq!(config.time_horizon_days, 1);
    }
}
