//! Risk limits and enforcement
//!
//! Real-time risk limit validation and enforcement:
//! - Position size limits
//! - Portfolio VaR limits
//! - Drawdown thresholds
//! - Leverage constraints
//! - Concentration limits

pub mod rules;
pub mod enforcement;

pub use rules::{RiskLimitRule, RiskLimitRules, RiskMetrics};
pub use enforcement::{LimitEnforcer, EnforcementDecision, EnforcementRecord, LimitUtilization};
