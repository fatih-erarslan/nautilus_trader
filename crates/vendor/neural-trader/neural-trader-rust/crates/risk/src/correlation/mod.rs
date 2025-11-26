//! Correlation analysis
//!
//! TODO: Implement correlation analysis
//! - Rolling correlations
//! - Copula-based dependence
//! - Regime detection

pub mod matrices;
pub mod copulas;

pub use matrices::CorrelationCalculator;
pub use copulas::CopulaAnalyzer;
