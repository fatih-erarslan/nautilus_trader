//! # HyperPhysics Risk
//!
//! Thermodynamic risk management for financial portfolios.
//!
//! This crate provides risk metrics grounded in statistical mechanics
//! and information theory:
//!
//! - **Portfolio Entropy**: Shannon entropy as diversification measure
//! - **Landauer Transaction Costs**: Fundamental limits on trading costs
//! - **Thermodynamic VaR**: Value-at-Risk with entropy constraints
//! - **Free Energy Optimization**: Risk-return balance via thermodynamics
//! - **Codependent Risk**: Network-based risk propagation (Pratītyasamutpāda)
//!
//! ## Example
//!
//! ```rust
//! use hyperphysics_risk::{PortfolioEntropy, Portfolio, Position};
//! use nalgebra::DVector;
//!
//! // Create portfolio
//! let mut portfolio = Portfolio::new(10000.0);
//! portfolio.add_position(Position::new("AAPL", 100.0, 150.0));
//! portfolio.add_position(Position::new("GOOGL", 50.0, 2800.0));
//!
//! // Calculate entropy (diversification)
//! let entropy_calc = PortfolioEntropy::new(1.0).unwrap();
//! let weights = DVector::from_vec(vec![0.4, 0.6]);
//! let entropy = entropy_calc.calculate_entropy(&weights).unwrap();
//! ```

pub mod error;
pub mod portfolio;
pub mod entropy;
pub mod landauer;
pub mod var;
pub mod codependent;

// Re-export main types
pub use error::{RiskError, Result};
pub use portfolio::{Portfolio, Position};
pub use entropy::PortfolioEntropy;
pub use landauer::TransactionCostModel;
pub use var::ThermodynamicVaR;
pub use codependent::{
    CodependentRiskModel, AssetNode, DependencyEdge, DependencyType,
    CodependentRisk, SystemicRisk, CodependentRiskError,
};
