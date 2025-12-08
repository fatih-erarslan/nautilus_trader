# Calculation Safety Framework
## Mathematical Integrity and Overflow Protection for Financial Trading

**Document Version**: 1.0  
**Date**: August 16, 2025  
**Classification**: TECHNICAL SPECIFICATION  

---

## OVERVIEW

This specification defines a comprehensive calculation safety framework that ensures mathematical integrity, prevents overflow/underflow conditions, and provides deterministic computation for financial trading operations. All financial calculations must be performed through this framework to prevent computational errors that could lead to significant financial losses.

## DESIGN PRINCIPLES

### Core Safety Principles

1. **Fail-Safe Mathematics**: All operations default to safe values on error
2. **Overflow Protection**: Detect and prevent numerical overflow/underflow
3. **Precision Preservation**: Maintain required precision for financial calculations
4. **Deterministic Results**: Ensure reproducible calculations across systems
5. **Comprehensive Validation**: Validate all inputs and outputs
6. **Error Transparency**: Provide detailed error context for debugging

## SAFE MATHEMATICAL OPERATIONS

### Core SafeMath Implementation

```rust
/// Safe mathematical operations with comprehensive error checking
pub struct SafeMath;

impl SafeMath {
    /// Safe division with comprehensive error checking
    pub fn safe_divide(
        numerator: f64, 
        denominator: f64, 
        context: &str
    ) -> Result<f64, TalebianSecureError> {
        // Input validation
        Self::validate_finite("numerator", numerator, context)?;
        Self::validate_finite("denominator", denominator, context)?;
        
        // Check for zero or near-zero denominator
        if denominator.abs() < f64::EPSILON * 1000.0 {
            return Err(TalebianSecureError::DivisionByZero {
                context: context.to_string(),
                numerator,
                denominator,
                operation_id: generate_operation_id(),
            });
        }
        
        // Check for potential overflow before division
        if numerator.abs() > f64::MAX / 2.0 && denominator.abs() < 1.0 {
            return Err(TalebianSecureError::NumericalOverflow {
                operation: format!("division: {} / {}", numerator, denominator),
                value: "overflow_risk".to_string(),
                max_allowed: Some(f64::MAX / 2.0),
                input_values: vec![numerator, denominator],
            });
        }
        
        // Perform division
        let result = numerator / denominator;
        
        // Validate result
        Self::validate_result(result, "division", context)?;
        
        // Check for underflow
        if result != 0.0 && result.abs() < f64::MIN_POSITIVE {
            return Err(TalebianSecureError::PrecisionLoss {
                operation: format!("division_underflow: {} / {}", numerator, denominator),
                loss_percentage: 100.0,
                original_precision: 64,
                resulting_precision: 0,
            });
        }
        
        Ok(result)
    }
    
    /// Safe multiplication with overflow checking
    pub fn safe_multiply(
        a: f64, 
        b: f64, 
        context: &str
    ) -> Result<f64, TalebianSecureError> {
        // Input validation
        Self::validate_finite("factor_a", a, context)?;
        Self::validate_finite("factor_b", b, context)?;
        
        // Check for potential overflow before multiplication
        // Use the fact that |a * b| = |a| * |b|
        let abs_a = a.abs();
        let abs_b = b.abs();
        
        if abs_a > 1.0 && abs_b > f64::MAX / abs_a {
            return Err(TalebianSecureError::NumericalOverflow {
                operation: format!("multiplication: {} * {}", a, b),
                value: "overflow_risk".to_string(),
                max_allowed: Some(f64::MAX),
                input_values: vec![a, b],
            });
        }
        
        let result = a * b;
        Self::validate_result(result, "multiplication", context)?;
        
        Ok(result)
    }
    
    /// Safe addition with overflow checking
    pub fn safe_add(
        a: f64, 
        b: f64, 
        context: &str
    ) -> Result<f64, TalebianSecureError> {
        Self::validate_finite("addend_a", a, context)?;
        Self::validate_finite("addend_b", b, context)?;
        
        // Check for overflow in addition
        if a > 0.0 && b > f64::MAX - a {
            return Err(TalebianSecureError::NumericalOverflow {
                operation: format!("addition: {} + {}", a, b),
                value: "positive_overflow".to_string(),
                max_allowed: Some(f64::MAX),
                input_values: vec![a, b],
            });
        }
        
        if a < 0.0 && b < f64::MIN - a {
            return Err(TalebianSecureError::NumericalOverflow {
                operation: format!("addition: {} + {}", a, b),
                value: "negative_overflow".to_string(),
                max_allowed: Some(f64::MIN),
                input_values: vec![a, b],
            });
        }
        
        let result = a + b;
        Self::validate_result(result, "addition", context)?;
        
        Ok(result)
    }
    
    /// Safe subtraction with overflow checking
    pub fn safe_subtract(
        a: f64, 
        b: f64, 
        context: &str
    ) -> Result<f64, TalebianSecureError> {
        Self::validate_finite("minuend", a, context)?;
        Self::validate_finite("subtrahend", b, context)?;
        
        // Subtraction: a - b = a + (-b)
        // Check for overflow when negating b
        if b == f64::MIN {
            return Err(TalebianSecureError::NumericalOverflow {
                operation: format!("subtraction: {} - {}", a, b),
                value: "negation_overflow".to_string(),
                max_allowed: Some(f64::MAX),
                input_values: vec![a, b],
            });
        }
        
        // Use safe addition with negated b
        Self::safe_add(a, -b, context)
    }
    
    /// Safe power calculation with overflow protection
    pub fn safe_pow(
        base: f64, 
        exponent: f64, 
        context: &str
    ) -> Result<f64, TalebianSecureError> {
        Self::validate_finite("base", base, context)?;
        Self::validate_finite("exponent", exponent, context)?;
        
        // Special cases
        if base == 0.0 && exponent <= 0.0 {
            return Err(TalebianSecureError::MathematicalComputation {
                details: format!("Invalid power: 0^{} in {}", exponent, context),
                computation_type: "power".to_string(),
                error_code: Some(1),
            });
        }
        
        if base < 0.0 && exponent.fract() != 0.0 {
            return Err(TalebianSecureError::MathematicalComputation {
                details: format!("Invalid power: negative base with fractional exponent in {}", context),
                computation_type: "power".to_string(),
                error_code: Some(2),
            });
        }
        
        // Check for potential overflow
        if base.abs() > 2.0 && exponent > 50.0 {
            return Err(TalebianSecureError::NumericalOverflow {
                operation: format!("power: {} ^ {}", base, exponent),
                value: "overflow_risk".to_string(),
                max_allowed: None,
                input_values: vec![base, exponent],
            });
        }
        
        // For large exponents with base > 1, estimate result size
        if base.abs() > 1.0 && exponent > 10.0 {
            let log_result = exponent * base.abs().ln();
            if log_result > f64::MAX.ln() {
                return Err(TalebianSecureError::NumericalOverflow {
                    operation: format!("power: {} ^ {}", base, exponent),
                    value: "estimated_overflow".to_string(),
                    max_allowed: Some(f64::MAX),
                    input_values: vec![base, exponent],
                });
            }
        }
        
        let result = base.powf(exponent);
        Self::validate_result(result, "power", context)?;
        
        Ok(result)
    }
    
    /// Safe logarithm calculation
    pub fn safe_log(value: f64, context: &str) -> Result<f64, TalebianSecureError> {
        Self::validate_finite("value", value, context)?;
        
        if value <= 0.0 {
            return Err(TalebianSecureError::MathematicalComputation {
                details: format!("Cannot take logarithm of non-positive value: {} in {}", value, context),
                computation_type: "logarithm".to_string(),
                error_code: Some(3),
            });
        }
        
        let result = value.ln();
        Self::validate_result(result, "logarithm", context)?;
        
        Ok(result)
    }
    
    /// Safe square root calculation
    pub fn safe_sqrt(value: f64, context: &str) -> Result<f64, TalebianSecureError> {
        Self::validate_finite("value", value, context)?;
        
        if value < 0.0 {
            return Err(TalebianSecureError::MathematicalComputation {
                details: format!("Cannot take square root of negative value: {} in {}", value, context),
                computation_type: "square_root".to_string(),
                error_code: Some(4),
            });
        }
        
        let result = value.sqrt();
        Self::validate_result(result, "square_root", context)?;
        
        Ok(result)
    }
    
    /// Safe percentage calculation
    pub fn safe_percentage(
        value: f64, 
        total: f64, 
        context: &str
    ) -> Result<f64, TalebianSecureError> {
        let fraction = Self::safe_divide(value, total, context)?;
        let percentage = Self::safe_multiply(fraction, 100.0, context)?;
        
        // Sanity check for percentage
        if percentage.abs() > 10000.0 {  // 10,000% seems unreasonable for most contexts
            return Err(TalebianSecureError::MathematicalComputation {
                details: format!("Unreasonable percentage: {:.2}% in {}", percentage, context),
                computation_type: "percentage".to_string(),
                error_code: Some(5),
            });
        }
        
        Ok(percentage)
    }
    
    /// Validate that a number is finite (not NaN or infinite)
    fn validate_finite(field_name: &str, value: f64, context: &str) -> Result<(), TalebianSecureError> {
        if value.is_nan() {
            return Err(TalebianSecureError::InvalidFloatingPoint {
                value: "NaN".to_string(),
                context: context.to_string(),
                operation_type: "input_validation".to_string(),
                input_summary: format!("Field: {} = NaN", field_name),
            });
        }
        
        if value.is_infinite() {
            return Err(TalebianSecureError::InvalidFloatingPoint {
                value: if value.is_sign_positive() { "Infinity" } else { "-Infinity" }.to_string(),
                context: context.to_string(),
                operation_type: "input_validation".to_string(),
                input_summary: format!("Field: {} = {}", field_name, value),
            });
        }
        
        Ok(())
    }
    
    /// Validate calculation result
    fn validate_result(result: f64, operation: &str, context: &str) -> Result<(), TalebianSecureError> {
        if result.is_nan() {
            return Err(TalebianSecureError::InvalidFloatingPoint {
                value: "NaN".to_string(),
                context: context.to_string(),
                operation_type: operation.to_string(),
                input_summary: "Result validation failed".to_string(),
            });
        }
        
        if result.is_infinite() {
            return Err(TalebianSecureError::InvalidFloatingPoint {
                value: if result.is_sign_positive() { "Infinity" } else { "-Infinity" }.to_string(),
                context: context.to_string(),
                operation_type: operation.to_string(),
                input_summary: "Result validation failed".to_string(),
            });
        }
        
        Ok(())
    }
}

/// Generate unique operation ID for error tracking
fn generate_operation_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    
    let id = COUNTER.fetch_add(1, Ordering::SeqCst);
    format!("op_{:08x}_{:016x}", 
           std::process::id(), 
           id)
}
```

## DETERMINISTIC CALCULATION PIPELINE

### Financial Calculation Engine

```rust
/// Deterministic calculation pipeline for financial operations
pub struct DeterministicCalculator {
    precision_config: PrecisionConfig,
    validation_config: ValidationConfig,
    calculation_cache: CalculationCache,
    audit_logger: CalculationAuditLogger,
}

impl DeterministicCalculator {
    /// Calculate Kelly fraction with full safety checks
    pub fn calculate_kelly_fraction(
        &self,
        win_probability: f64,
        win_amount: f64,
        loss_amount: f64,
    ) -> Result<f64, TalebianSecureError> {
        let context = "kelly_fraction_calculation";
        
        // Input validation with business logic
        self.validate_probability("win_probability", win_probability, context)?;
        self.validate_positive_amount("win_amount", win_amount, context)?;
        self.validate_positive_amount("loss_amount", loss_amount, context)?;
        
        // Log calculation start
        let calc_id = self.audit_logger.start_calculation(context, &[
            ("win_probability", win_probability),
            ("win_amount", win_amount),
            ("loss_amount", loss_amount),
        ])?;
        
        // Calculate odds with safety
        let odds = SafeMath::safe_divide(win_amount, loss_amount, context)?;
        
        // Kelly formula: f = (bp - q) / b
        // where b = odds, p = win_probability, q = loss_probability
        let loss_probability = SafeMath::safe_subtract(1.0, win_probability, context)?;
        let bp = SafeMath::safe_multiply(odds, win_probability, context)?;
        let numerator = SafeMath::safe_subtract(bp, loss_probability, context)?;
        
        let kelly_fraction = SafeMath::safe_divide(numerator, odds, context)?;
        
        // Apply safety bounds (Kelly can recommend more than 100% which is dangerous)
        let bounded_fraction = self.apply_kelly_bounds(kelly_fraction, context)?;
        
        // Validate final result
        self.validate_kelly_result(bounded_fraction, context)?;
        
        // Log calculation completion
        self.audit_logger.complete_calculation(calc_id, bounded_fraction)?;
        
        Ok(bounded_fraction)
    }
    
    /// Calculate position size with multiple safety layers
    pub fn calculate_position_size(
        &self,
        kelly_fraction: f64,
        confidence: f64,
        account_balance: f64,
        max_position_pct: f64,
        market_conditions: &MarketConditions,
    ) -> Result<PositionSizeResult, TalebianSecureError> {
        let context = "position_size_calculation";
        
        // Comprehensive input validation
        self.validate_kelly_fraction(kelly_fraction, context)?;
        self.validate_probability("confidence", confidence, context)?;
        self.validate_positive_amount("account_balance", account_balance, context)?;
        self.validate_percentage("max_position_pct", max_position_pct, context)?;
        
        let calc_id = self.audit_logger.start_calculation(context, &[
            ("kelly_fraction", kelly_fraction),
            ("confidence", confidence),
            ("account_balance", account_balance),
            ("max_position_pct", max_position_pct),
        ])?;
        
        // Step 1: Apply confidence adjustment
        let confidence_adjusted_kelly = SafeMath::safe_multiply(
            kelly_fraction, 
            confidence, 
            "confidence_adjustment"
        )?;
        
        // Step 2: Apply volatility adjustment
        let volatility_multiplier = self.calculate_volatility_multiplier(
            market_conditions.volatility, 
            context
        )?;
        let volatility_adjusted_size = SafeMath::safe_multiply(
            confidence_adjusted_kelly,
            volatility_multiplier,
            "volatility_adjustment"
        )?;
        
        // Step 3: Apply market regime adjustment
        let regime_multiplier = self.calculate_regime_multiplier(
            &market_conditions.regime,
            context
        )?;
        let regime_adjusted_size = SafeMath::safe_multiply(
            volatility_adjusted_size,
            regime_multiplier,
            "regime_adjustment"
        )?;
        
        // Step 4: Apply absolute maximum position limit
        let capped_fraction = regime_adjusted_size.min(max_position_pct);
        
        // Step 5: Apply additional safety bounds
        let safety_bounded_fraction = self.apply_position_safety_bounds(
            capped_fraction,
            market_conditions,
            context
        )?;
        
        // Step 6: Calculate monetary position size
        let position_size_monetary = SafeMath::safe_multiply(
            account_balance,
            safety_bounded_fraction,
            "monetary_conversion"
        )?;
        
        // Step 7: Final validation
        self.validate_position_size_result(
            position_size_monetary,
            account_balance,
            context
        )?;
        
        let result = PositionSizeResult {
            fraction: safety_bounded_fraction,
            monetary_amount: position_size_monetary,
            confidence_adjustment: confidence,
            volatility_adjustment: volatility_multiplier,
            regime_adjustment: regime_multiplier,
            safety_bounds_applied: safety_bounded_fraction < regime_adjusted_size,
            calculation_id: calc_id.clone(),
        };
        
        self.audit_logger.complete_calculation(calc_id, safety_bounded_fraction)?;
        
        Ok(result)
    }
    
    /// Calculate risk-adjusted return with safety checks
    pub fn calculate_risk_adjusted_return(
        &self,
        returns: &[f64],
        risk_free_rate: f64,
    ) -> Result<RiskAdjustedMetrics, TalebianSecureError> {
        let context = "risk_adjusted_return_calculation";
        
        if returns.is_empty() {
            return Err(TalebianSecureError::InputValidation {
                field: "returns".to_string(),
                reason: "Returns array cannot be empty".to_string(),
                source: None,
            });
        }
        
        // Validate all returns
        for (i, &return_val) in returns.iter().enumerate() {
            SafeMath::validate_finite(&format!("returns[{}]", i), return_val, context)?;
            
            // Check for extreme returns that might indicate data errors
            if return_val.abs() > self.validation_config.max_single_period_return {
                return Err(TalebianSecureError::InputValidation {
                    field: format!("returns[{}]", i),
                    reason: format!("Extreme return value: {:.4}", return_val),
                    source: None,
                });
            }
        }
        
        let calc_id = self.audit_logger.start_calculation(context, &[
            ("num_returns", returns.len() as f64),
            ("risk_free_rate", risk_free_rate),
        ])?;
        
        // Calculate mean return
        let mean_return = self.calculate_mean(returns, context)?;
        
        // Calculate standard deviation
        let std_dev = self.calculate_standard_deviation(returns, mean_return, context)?;
        
        // Calculate Sharpe ratio
        let excess_return = SafeMath::safe_subtract(mean_return, risk_free_rate, context)?;
        let sharpe_ratio = if std_dev > f64::EPSILON {
            SafeMath::safe_divide(excess_return, std_dev, "sharpe_ratio_calculation")?
        } else {
            return Err(TalebianSecureError::DivisionByZero {
                context: "sharpe_ratio_zero_volatility".to_string(),
                numerator: excess_return,
                denominator: std_dev,
                operation_id: generate_operation_id(),
            });
        };
        
        // Calculate downside deviation for Sortino ratio
        let downside_returns: Vec<f64> = returns.iter()
            .filter(|&&r| r < risk_free_rate)
            .map(|&r| SafeMath::safe_subtract(r, risk_free_rate, context))
            .collect::<Result<Vec<_>, _>>()?;
        
        let sortino_ratio = if !downside_returns.is_empty() {
            let downside_deviation = self.calculate_standard_deviation(
                &downside_returns, 
                0.0, 
                "downside_deviation"
            )?;
            
            if downside_deviation > f64::EPSILON {
                SafeMath::safe_divide(excess_return, downside_deviation, "sortino_ratio_calculation")?
            } else {
                f64::INFINITY  // Perfect Sortino ratio (no downside)
            }
        } else {
            f64::INFINITY  // No negative returns
        };
        
        // Calculate maximum drawdown
        let max_drawdown = self.calculate_maximum_drawdown(returns, context)?;
        
        let result = RiskAdjustedMetrics {
            mean_return,
            standard_deviation: std_dev,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            num_observations: returns.len(),
            calculation_id: calc_id.clone(),
        };
        
        self.audit_logger.complete_calculation(calc_id, sharpe_ratio)?;
        
        Ok(result)
    }
    
    /// Calculate mean with overflow protection
    fn calculate_mean(&self, values: &[f64], context: &str) -> Result<f64, TalebianSecureError> {
        if values.is_empty() {
            return Err(TalebianSecureError::InputValidation {
                field: "values".to_string(),
                reason: "Cannot calculate mean of empty array".to_string(),
                source: None,
            });
        }
        
        let mut sum = 0.0;
        for &value in values {
            sum = SafeMath::safe_add(sum, value, context)?;
        }
        
        SafeMath::safe_divide(sum, values.len() as f64, context)
    }
    
    /// Calculate standard deviation with numerical stability
    fn calculate_standard_deviation(
        &self, 
        values: &[f64], 
        mean: f64, 
        context: &str
    ) -> Result<f64, TalebianSecureError> {
        if values.len() < 2 {
            return Err(TalebianSecureError::MathematicalComputation {
                details: "Need at least 2 values for standard deviation".to_string(),
                computation_type: "standard_deviation".to_string(),
                error_code: Some(6),
            });
        }
        
        let mut sum_squared_deviations = 0.0;
        
        for &value in values {
            let deviation = SafeMath::safe_subtract(value, mean, context)?;
            let squared_deviation = SafeMath::safe_multiply(deviation, deviation, context)?;
            sum_squared_deviations = SafeMath::safe_add(sum_squared_deviations, squared_deviation, context)?;
        }
        
        let variance = SafeMath::safe_divide(
            sum_squared_deviations, 
            (values.len() - 1) as f64, 
            context
        )?;
        
        SafeMath::safe_sqrt(variance, context)
    }
    
    /// Calculate maximum drawdown
    fn calculate_maximum_drawdown(&self, returns: &[f64], context: &str) -> Result<f64, TalebianSecureError> {
        if returns.is_empty() {
            return Ok(0.0);
        }
        
        let mut cumulative_return = 1.0;
        let mut peak = 1.0;
        let mut max_drawdown = 0.0;
        
        for &return_val in returns {
            let one_plus_return = SafeMath::safe_add(1.0, return_val, context)?;
            cumulative_return = SafeMath::safe_multiply(cumulative_return, one_plus_return, context)?;
            
            if cumulative_return > peak {
                peak = cumulative_return;
            }
            
            let drawdown = SafeMath::safe_divide(
                SafeMath::safe_subtract(peak, cumulative_return, context)?,
                peak,
                context
            )?;
            
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }
        
        Ok(max_drawdown)
    }
    
    /// Apply Kelly fraction bounds for safety
    fn apply_kelly_bounds(&self, kelly_fraction: f64, context: &str) -> Result<f64, TalebianSecureError> {
        // Kelly can theoretically recommend > 100% position, which is dangerous
        let max_kelly = self.precision_config.max_kelly_fraction;
        let min_kelly = self.precision_config.min_kelly_fraction;
        
        let bounded = if kelly_fraction > max_kelly {
            max_kelly
        } else if kelly_fraction < min_kelly {
            min_kelly
        } else {
            kelly_fraction
        };
        
        // Log if bounds were applied
        if (bounded - kelly_fraction).abs() > f64::EPSILON {
            self.audit_logger.log_bounds_application(
                context,
                kelly_fraction,
                bounded,
                "kelly_bounds"
            )?;
        }
        
        Ok(bounded)
    }
    
    /// Validation methods
    fn validate_probability(&self, field: &str, value: f64, context: &str) -> Result<(), TalebianSecureError> {
        SafeMath::validate_finite(field, value, context)?;
        
        if value < 0.0 || value > 1.0 {
            return Err(TalebianSecureError::InputValidation {
                field: field.to_string(),
                reason: format!("Probability must be between 0.0 and 1.0, got {}", value),
                source: None,
            });
        }
        
        Ok(())
    }
    
    fn validate_positive_amount(&self, field: &str, value: f64, context: &str) -> Result<(), TalebianSecureError> {
        SafeMath::validate_finite(field, value, context)?;
        
        if value <= 0.0 {
            return Err(TalebianSecureError::InputValidation {
                field: field.to_string(),
                reason: format!("Amount must be positive, got {}", value),
                source: None,
            });
        }
        
        Ok(())
    }
    
    fn validate_percentage(&self, field: &str, value: f64, context: &str) -> Result<(), TalebianSecureError> {
        SafeMath::validate_finite(field, value, context)?;
        
        if value < 0.0 || value > 1.0 {
            return Err(TalebianSecureError::InputValidation {
                field: field.to_string(),
                reason: format!("Percentage must be between 0.0 and 1.0, got {}", value),
                source: None,
            });
        }
        
        Ok(())
    }
    
    fn validate_kelly_fraction(&self, value: f64, context: &str) -> Result<(), TalebianSecureError> {
        SafeMath::validate_finite("kelly_fraction", value, context)?;
        
        if value < -1.0 || value > 2.0 {
            return Err(TalebianSecureError::PositionSizing {
                reason: format!("Kelly fraction {} outside reasonable bounds [-1.0, 2.0]", value),
                calculated_size: Some(value),
                max_allowed_size: Some(2.0),
                account_balance: None,
            });
        }
        
        Ok(())
    }
}

/// Configuration for calculation precision
#[derive(Debug, Clone)]
pub struct PrecisionConfig {
    pub decimal_places: u8,
    pub rounding_mode: RoundingMode,
    pub max_kelly_fraction: f64,
    pub min_kelly_fraction: f64,
    pub epsilon_multiplier: f64,
}

impl Default for PrecisionConfig {
    fn default() -> Self {
        Self {
            decimal_places: 8,
            rounding_mode: RoundingMode::RoundHalfUp,
            max_kelly_fraction: 0.25,  // Maximum 25% Kelly
            min_kelly_fraction: 0.0,   // Minimum 0% (no negative positions)
            epsilon_multiplier: 1000.0,
        }
    }
}

#[derive(Debug, Clone)]
pub enum RoundingMode {
    RoundHalfUp,
    RoundHalfDown,
    RoundTowardsZero,
    RoundAwayFromZero,
}

/// Result structures
#[derive(Debug, Clone)]
pub struct PositionSizeResult {
    pub fraction: f64,
    pub monetary_amount: f64,
    pub confidence_adjustment: f64,
    pub volatility_adjustment: f64,
    pub regime_adjustment: f64,
    pub safety_bounds_applied: bool,
    pub calculation_id: String,
}

#[derive(Debug, Clone)]
pub struct RiskAdjustedMetrics {
    pub mean_return: f64,
    pub standard_deviation: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub num_observations: usize,
    pub calculation_id: String,
}

/// Market conditions for adjustments
#[derive(Debug, Clone)]
pub struct MarketConditions {
    pub volatility: f64,
    pub regime: MarketRegime,
    pub liquidity_score: f64,
    pub correlation_environment: f64,
}

#[derive(Debug, Clone)]
pub enum MarketRegime {
    Normal,
    HighVolatility,
    Crisis,
    Recovery,
    Bull,
    Bear,
}
```

This calculation safety framework provides comprehensive protection against computational errors, overflow conditions, and precision loss while maintaining deterministic behavior essential for financial trading operations.