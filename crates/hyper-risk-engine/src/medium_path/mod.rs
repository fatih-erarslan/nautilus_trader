//! Medium path calculations (100μs - 1ms).
//!
//! This module contains more sophisticated analyses that run
//! in the medium latency path, including:
//!
//! - VaR/CVaR calculation
//! - Correlation updates (DCC-GARCH style)
//! - Regime-aware risk adjustment
//! - Kelly position sizing
//!
//! ## Latency Budget
//!
//! | Component | Target | Implementation |
//! |-----------|--------|----------------|
//! | VaR calculation | 200μs | Parametric with EVT tail |
//! | Correlation update | 300μs | Exponential weighting |
//! | Regime adjustment | 100μs | Pre-computed multipliers |
//! | Position sizing | 200μs | Kelly with constraints |
//! | **Total** | **800μs** | |

pub mod risk_calculator;
pub mod correlation;
pub mod regime_adjustor;

pub use risk_calculator::{RiskCalculator, RiskMetrics};
pub use correlation::{CorrelationTracker, CorrelationMatrix};
pub use regime_adjustor::{RegimeAdjustor, RegimeAdjustment};
