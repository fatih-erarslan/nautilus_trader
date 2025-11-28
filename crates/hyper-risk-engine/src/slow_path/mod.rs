//! Slow path calculations (>1ms, typically seconds to minutes).
//!
//! This module contains computationally intensive analyses that run
//! asynchronously or on scheduled intervals:
//!
//! - Monte Carlo VaR/ES
//! - Stress testing
//! - FRTB Expected Shortfall
//! - Portfolio optimization
//! - Historical scenario analysis
//!
//! ## Scheduling
//!
//! | Calculation | Frequency | Trigger |
//! |------------|-----------|---------|
//! | Monte Carlo VaR | Every 5 min | Timer + significant P&L change |
//! | Stress Tests | Every 30 min | Timer + regime change |
//! | FRTB ES | Daily | Market close |
//! | Portfolio Opt | Daily | Market close |
//! | Scenario Analysis | Weekly | Scheduled |

pub mod monte_carlo;
pub mod stress_test;
pub mod frtb;

pub use monte_carlo::{MonteCarloVaR, MonteCarloConfig, MonteCarloResult};
pub use stress_test::{StressTest, StressScenario, StressResult};
pub use frtb::{FRTBCalculator, FRTBResult, FRTBConfig};
