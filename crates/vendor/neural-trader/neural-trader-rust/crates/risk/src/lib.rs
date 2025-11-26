//! # Neural Trader Risk Management System
//!
//! Advanced risk management and portfolio tracking for algorithmic trading.
//!
//! ## Features
//!
//! - **VaR/CVaR Calculation**: Monte Carlo, Historical, and Parametric VaR
//! - **Stress Testing**: Historical scenario replay (2008, 2020) and custom scenarios
//! - **Kelly Criterion**: Optimal position sizing (single and multi-asset)
//! - **Portfolio Tracking**: Real-time P&L, exposure, and margin monitoring
//! - **Correlation Analysis**: Rolling correlations, copula-based dependence
//! - **Risk Limits**: Position size limits, VaR limits, drawdown thresholds
//! - **Emergency Protocols**: Circuit breakers, automated stop-loss, position flattening
//! - **GPU Acceleration**: Optional GPU-accelerated Monte Carlo simulations
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use nt_risk::var::{MonteCarloVaR, VaRConfig};
//! use nt_risk::portfolio::PortfolioTracker;
//! use rust_decimal_macros::dec;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Calculate VaR using Monte Carlo simulation
//!     let var_config = VaRConfig {
//!         confidence_level: 0.95,
//!         time_horizon_days: 1,
//!         num_simulations: 10_000,
//!         use_gpu: false,
//!     };
//!
//!     let positions = vec![/* your positions */];
//!     let var_calculator = MonteCarloVaR::new(var_config);
//!     let var_result = var_calculator.calculate(&positions).await?;
//!
//!     println!("VaR (95%): ${:.2}", var_result.var_95);
//!     println!("CVaR (95%): ${:.2}", var_result.cvar_95);
//!
//!     // Track portfolio in real-time
//!     let mut tracker = PortfolioTracker::new();
//!     tracker.update_position(/* position update */).await?;
//!
//!     let pnl = tracker.calculate_pnl().await?;
//!     println!("Unrealized P&L: ${:.2}", pnl.unrealized);
//!
//!     Ok(())
//! }
//! ```

pub mod var;
pub mod stress;
pub mod kelly;
pub mod portfolio;
pub mod correlation;
pub mod limits;
pub mod emergency;
pub mod types;
pub mod error;

// Re-export commonly used types
pub use error::{Result, RiskError};
pub use types::{
    Portfolio, Asset, RiskMetrics, VaRResult, StressTestResult,
    CorrelationMatrix, RiskLimit, AlertLevel, EmergencyAction,
};

use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Initialize the risk management system with logging
pub fn init() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "nt_risk=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
}

/// Check if GPU acceleration is available
#[cfg(feature = "gpu")]
pub fn is_gpu_available() -> bool {
    use cudarc::driver::CudaDevice;
    CudaDevice::new(0).is_ok()
}

#[cfg(not(feature = "gpu"))]
pub fn is_gpu_available() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        init();
        // Just verify it doesn't panic
    }

    #[test]
    fn test_gpu_availability() {
        let available = is_gpu_available();
        println!("GPU available: {}", available);
        // Test passes regardless of GPU availability
    }
}
