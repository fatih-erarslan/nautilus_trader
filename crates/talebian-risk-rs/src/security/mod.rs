//! Security module for financial trading system
//! 
//! This module provides CRITICAL security utilities for the Talebian Risk Management system.
//! It implements bulletproof mathematical operations, comprehensive input validation,
//! and error handling patterns that prevent financial losses due to calculation errors.

use crate::TalebianRiskError;

pub mod safe_math;
pub mod validation;
pub mod error_handling;

pub use safe_math::{SafeMath, safe_divide, safe_multiply, safe_add, safe_subtract};
pub use validation::{validate_market_data, validate_percentage, validate_positive};
pub use error_handling::{FinancialError, SecurityResult};

/// Security constants for financial calculations
pub const EPSILON: f64 = 1e-10;
pub const MAX_POSITION_SIZE: f64 = 0.95; // Maximum 95% of capital
pub const MIN_POSITION_SIZE: f64 = 0.001; // Minimum 0.1% position
pub const MAX_PRICE: f64 = 1_000_000.0; // Maximum reasonable price
pub const MAX_VOLUME: f64 = 1_000_000_000.0; // Maximum reasonable volume
pub const MAX_PERCENTAGE: f64 = 1.0; // Maximum 100%

/// Security validation result
pub type SecurityValidationResult<T> = Result<T, TalebianRiskError>;

/// Initialize security framework - call once at startup
pub fn initialize_security_framework() -> SecurityValidationResult<()> {
    // Verify system integrity
    if !safe_math::verify_math_integrity() {
        return Err(TalebianRiskError::CalculationError("Math integrity check failed".to_string()));
    }
    
    Ok(())
}