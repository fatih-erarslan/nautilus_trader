//! Enhanced error handling patterns for financial calculations

use crate::error::TalebianError;
use std::fmt;

/// Enhanced error type specifically for safe math operations
#[derive(Debug, Clone)]
pub enum SafeMathError {
    /// Division by zero or near-zero value
    DivisionByZero {
        numerator: f64,
        denominator: f64,
        context: String,
    },
    /// Numerical overflow in calculation
    Overflow {
        operation: String,
        operands: Vec<f64>,
        result: f64,
    },
    /// Invalid input values (NaN, Infinity, etc.)
    InvalidInput {
        parameter: String,
        value: f64,
        expected: String,
    },
    /// Insufficient data for calculation
    InsufficientData {
        required: usize,
        available: usize,
        operation: String,
    },
    /// Domain error (e.g., negative value for sqrt)
    DomainError {
        function: String,
        input: f64,
        domain: String,
    },
    /// Convergence failure in iterative algorithms
    ConvergenceFailure {
        algorithm: String,
        iterations: usize,
        tolerance: f64,
        final_error: f64,
    },
    /// Matrix operation errors
    MatrixError {
        operation: String,
        dimensions: String,
        details: String,
    },
    /// Validation error
    ValidationError {
        field: String,
        value: String,
        constraint: String,
    },
}

impl fmt::Display for SafeMathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SafeMathError::DivisionByZero { numerator, denominator, context } => {
                write!(
                    f,
                    "Division by zero in {}: {} / {} (denominator too close to zero)",
                    context, numerator, denominator
                )
            }
            SafeMathError::Overflow { operation, operands, result } => {
                write!(
                    f,
                    "Numerical overflow in {}: operands {:?} resulted in {}",
                    operation, operands, result
                )
            }
            SafeMathError::InvalidInput { parameter, value, expected } => {
                write!(
                    f,
                    "Invalid input for parameter '{}': {} (expected {})",
                    parameter, value, expected
                )
            }
            SafeMathError::InsufficientData { required, available, operation } => {
                write!(
                    f,
                    "Insufficient data for {}: need {} but got {} data points",
                    operation, required, available
                )
            }
            SafeMathError::DomainError { function, input, domain } => {
                write!(
                    f,
                    "Domain error in {}: input {} not in valid domain {}",
                    function, input, domain
                )
            }
            SafeMathError::ConvergenceFailure { algorithm, iterations, tolerance, final_error } => {
                write!(
                    f,
                    "Convergence failure in {}: {} iterations, tolerance {}, final error {}",
                    algorithm, iterations, tolerance, final_error
                )
            }
            SafeMathError::MatrixError { operation, dimensions, details } => {
                write!(
                    f,
                    "Matrix error in {}: dimensions {}, details: {}",
                    operation, dimensions, details
                )
            }
            SafeMathError::ValidationError { field, value, constraint } => {
                write!(
                    f,
                    "Validation error for field '{}': value '{}' violates constraint '{}'",
                    field, value, constraint
                )
            }
        }
    }
}

impl std::error::Error for SafeMathError {}

impl From<SafeMathError> for TalebianError {
    fn from(error: SafeMathError) -> Self {
        TalebianError::math(error.to_string())
    }
}

/// Result type for safe math operations with enhanced error information
pub type SafeMathResult<T> = Result<T, SafeMathError>;

/// Error recovery strategies
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Return a default value
    UseDefault(f64),
    /// Use the last known good value
    UseLastGood,
    /// Interpolate from surrounding values
    Interpolate,
    /// Fail with error
    Fail,
    /// Use alternative calculation method
    UseAlternative,
}

/// Enhanced error context for better debugging
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub operation: String,
    pub inputs: Vec<f64>,
    pub expected_output: Option<f64>,
    pub recovery_strategy: RecoveryStrategy,
    pub source_location: String,
}

impl ErrorContext {
    pub fn new(operation: &str) -> Self {
        Self {
            operation: operation.to_string(),
            inputs: Vec::new(),
            expected_output: None,
            recovery_strategy: RecoveryStrategy::Fail,
            source_location: String::new(),
        }
    }
    
    pub fn with_inputs(mut self, inputs: Vec<f64>) -> Self {
        self.inputs = inputs;
        self
    }
    
    pub fn with_recovery(mut self, strategy: RecoveryStrategy) -> Self {
        self.recovery_strategy = strategy;
        self
    }
    
    pub fn with_location(mut self, location: &str) -> Self {
        self.source_location = location.to_string();
        self
    }
}

/// Error handler trait for custom error handling strategies
pub trait ErrorHandler {
    fn handle_division_by_zero(&self, ctx: &ErrorContext) -> SafeMathResult<f64>;
    fn handle_overflow(&self, ctx: &ErrorContext) -> SafeMathResult<f64>;
    fn handle_invalid_input(&self, ctx: &ErrorContext) -> SafeMathResult<f64>;
    fn handle_domain_error(&self, ctx: &ErrorContext) -> SafeMathResult<f64>;
}

/// Conservative error handler that prioritizes safety
pub struct ConservativeErrorHandler;

impl ErrorHandler for ConservativeErrorHandler {
    fn handle_division_by_zero(&self, ctx: &ErrorContext) -> SafeMathResult<f64> {
        match &ctx.recovery_strategy {
            RecoveryStrategy::UseDefault(value) => Ok(*value),
            RecoveryStrategy::UseLastGood => {
                // In a real implementation, this would use cached values
                Ok(0.0)
            }
            _ => Err(SafeMathError::DivisionByZero {
                numerator: ctx.inputs.get(0).copied().unwrap_or(0.0),
                denominator: ctx.inputs.get(1).copied().unwrap_or(0.0),
                context: ctx.operation.clone(),
            }),
        }
    }
    
    fn handle_overflow(&self, _ctx: &ErrorContext) -> SafeMathResult<f64> {
        match &_ctx.recovery_strategy {
            RecoveryStrategy::UseDefault(value) => Ok(*value),
            _ => Err(SafeMathError::Overflow {
                operation: _ctx.operation.clone(),
                operands: _ctx.inputs.clone(),
                result: f64::INFINITY,
            }),
        }
    }
    
    fn handle_invalid_input(&self, _ctx: &ErrorContext) -> SafeMathResult<f64> {
        match &_ctx.recovery_strategy {
            RecoveryStrategy::UseDefault(value) => Ok(*value),
            _ => Err(SafeMathError::InvalidInput {
                parameter: _ctx.operation.clone(),
                value: _ctx.inputs.get(0).copied().unwrap_or(f64::NAN),
                expected: "finite number".to_string(),
            }),
        }
    }
    
    fn handle_domain_error(&self, ctx: &ErrorContext) -> SafeMathResult<f64> {
        match &ctx.recovery_strategy {
            RecoveryStrategy::UseDefault(value) => Ok(*value),
            RecoveryStrategy::UseAlternative => {
                // Return a safe alternative (e.g., 0 for sqrt of negative)
                Ok(0.0)
            }
            _ => Err(SafeMathError::DomainError {
                function: ctx.operation.clone(),
                input: ctx.inputs.get(0).copied().unwrap_or(0.0),
                domain: "positive real numbers".to_string(),
            }),
        }
    }
}

/// Aggressive error handler that attempts recovery when possible
pub struct AggressiveErrorHandler;

impl ErrorHandler for AggressiveErrorHandler {
    fn handle_division_by_zero(&self, ctx: &ErrorContext) -> SafeMathResult<f64> {
        // Always try to recover with a sensible default
        if let Some(numerator) = ctx.inputs.get(0) {
            if *numerator == 0.0 {
                Ok(0.0) // 0/0 -> 0
            } else if *numerator > 0.0 {
                Ok(f64::MAX) // Positive/0 -> Large positive
            } else {
                Ok(f64::MIN) // Negative/0 -> Large negative
            }
        } else {
            Ok(0.0)
        }
    }
    
    fn handle_overflow(&self, ctx: &ErrorContext) -> SafeMathResult<f64> {
        // Use maximum safe value instead of infinity
        Ok(1e12)
    }
    
    fn handle_invalid_input(&self, ctx: &ErrorContext) -> SafeMathResult<f64> {
        // Replace NaN/Infinity with zero
        Ok(0.0)
    }
    
    fn handle_domain_error(&self, ctx: &ErrorContext) -> SafeMathResult<f64> {
        // Use absolute value or other transforms
        if let Some(input) = ctx.inputs.get(0) {
            if ctx.operation.contains("sqrt") && *input < 0.0 {
                return Ok(input.abs().sqrt()); // sqrt(|x|)
            } else if ctx.operation.contains("ln") && *input <= 0.0 {
                return Ok(f64::MIN.ln()); // ln(very small positive)
            }
        }
        Ok(0.0)
    }
}

/// Failsafe calculation wrapper with error recovery
pub struct FailsafeCalculator<H: ErrorHandler> {
    handler: H,
    last_good_values: std::collections::HashMap<String, f64>,
}

impl<H: ErrorHandler> FailsafeCalculator<H> {
    pub fn new(handler: H) -> Self {
        Self {
            handler,
            last_good_values: std::collections::HashMap::new(),
        }
    }
    
    /// Perform safe division with error recovery
    pub fn safe_divide_with_recovery(
        &mut self,
        numerator: f64,
        denominator: f64,
        operation_name: &str,
    ) -> SafeMathResult<f64> {
        let ctx = ErrorContext::new("division")
            .with_inputs(vec![numerator, denominator])
            .with_location(operation_name);
        
        // Check for invalid inputs
        if !numerator.is_finite() || !denominator.is_finite() {
            return self.handler.handle_invalid_input(&ctx);
        }
        
        // Check for division by zero
        if denominator.abs() < 1e-15 {
            return self.handler.handle_division_by_zero(&ctx);
        }
        
        let result = numerator / denominator;
        
        // Check for overflow
        if !result.is_finite() {
            return self.handler.handle_overflow(&ctx);
        }
        
        // Store last good value
        self.last_good_values.insert(operation_name.to_string(), result);
        
        Ok(result)
    }
    
    /// Get last good value for an operation
    pub fn get_last_good_value(&self, operation_name: &str) -> Option<f64> {
        self.last_good_values.get(operation_name).copied()
    }
    
    /// Clear cached values
    pub fn clear_cache(&mut self) {
        self.last_good_values.clear();
    }
}

/// Macro for safe mathematical operations with automatic error handling
#[macro_export]
macro_rules! safe_math {
    ($op:ident($($arg:expr),*) with $handler:expr => $default:expr) => {
        {
            let result = $op($($arg),*);
            match result {
                Ok(value) => value,
                Err(_) => $default,
            }
        }
    };
    
    ($op:ident($($arg:expr),*) with recovery $recovery:expr) => {
        {
            let result = $op($($arg),*);
            match result {
                Ok(value) => value,
                Err(_) => $recovery,
            }
        }
    };
}

/// Validation helpers for common financial constraints
pub mod validators {
    use super::*;
    
    pub fn validate_price(price: f64, name: &str) -> SafeMathResult<()> {
        if !price.is_finite() {
            return Err(SafeMathError::InvalidInput {
                parameter: name.to_string(),
                value: price,
                expected: "finite positive number".to_string(),
            });
        }
        
        if price <= 0.0 {
            return Err(SafeMathError::ValidationError {
                field: name.to_string(),
                value: price.to_string(),
                constraint: "must be positive".to_string(),
            });
        }
        
        if price > 1e12 {
            return Err(SafeMathError::ValidationError {
                field: name.to_string(),
                value: price.to_string(),
                constraint: "must be less than 1e12".to_string(),
            });
        }
        
        Ok(())
    }
    
    pub fn validate_ratio(ratio: f64, name: &str) -> SafeMathResult<()> {
        if !ratio.is_finite() {
            return Err(SafeMathError::InvalidInput {
                parameter: name.to_string(),
                value: ratio,
                expected: "finite number between 0 and 1".to_string(),
            });
        }
        
        if ratio < 0.0 || ratio > 1.0 {
            return Err(SafeMathError::ValidationError {
                field: name.to_string(),
                value: ratio.to_string(),
                constraint: "must be between 0 and 1".to_string(),
            });
        }
        
        Ok(())
    }
    
    pub fn validate_return(return_value: f64, name: &str) -> SafeMathResult<()> {
        if !return_value.is_finite() {
            return Err(SafeMathError::InvalidInput {
                parameter: name.to_string(),
                value: return_value,
                expected: "finite number >= -1".to_string(),
            });
        }
        
        if return_value < -1.0 {
            return Err(SafeMathError::ValidationError {
                field: name.to_string(),
                value: return_value.to_string(),
                constraint: "must be >= -1 (cannot lose more than 100%)".to_string(),
            });
        }
        
        Ok(())
    }
    
    pub fn validate_data_array(data: &[f64], name: &str, min_length: usize) -> SafeMathResult<()> {
        if data.len() < min_length {
            return Err(SafeMathError::InsufficientData {
                required: min_length,
                available: data.len(),
                operation: name.to_string(),
            });
        }
        
        for (i, &value) in data.iter().enumerate() {
            if !value.is_finite() {
                return Err(SafeMathError::InvalidInput {
                    parameter: format!("{}[{}]", name, i),
                    value,
                    expected: "finite number".to_string(),
                });
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conservative_error_handler() {
        let handler = ConservativeErrorHandler;
        let ctx = ErrorContext::new("test_division")
            .with_inputs(vec![10.0, 0.0])
            .with_recovery(RecoveryStrategy::UseDefault(1.0));
        
        let result = handler.handle_division_by_zero(&ctx);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1.0);
    }

    #[test]
    fn test_aggressive_error_handler() {
        let handler = AggressiveErrorHandler;
        let ctx = ErrorContext::new("test_division")
            .with_inputs(vec![10.0, 0.0]);
        
        let result = handler.handle_division_by_zero(&ctx);
        assert!(result.is_ok());
        assert!(result.unwrap() > 0.0);
    }

    #[test]
    fn test_failsafe_calculator() {
        let mut calculator = FailsafeCalculator::new(ConservativeErrorHandler);
        
        // Valid division
        let result = calculator.safe_divide_with_recovery(10.0, 2.0, "test_op");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 5.0);
        
        // Division by zero
        let result = calculator.safe_divide_with_recovery(10.0, 0.0, "test_op");
        assert!(result.is_err());
    }

    #[test]
    fn test_validators() {
        use validators::*;
        
        assert!(validate_price(100.0, "price").is_ok());
        assert!(validate_price(-10.0, "price").is_err());
        assert!(validate_price(f64::NAN, "price").is_err());
        
        assert!(validate_ratio(0.5, "ratio").is_ok());
        assert!(validate_ratio(-0.1, "ratio").is_err());
        assert!(validate_ratio(1.5, "ratio").is_err());
        
        assert!(validate_return(0.1, "return").is_ok());
        assert!(validate_return(-0.5, "return").is_ok());
        assert!(validate_return(-1.5, "return").is_err());
        
        let data = vec![1.0, 2.0, 3.0];
        assert!(validate_data_array(&data, "test_data", 3).is_ok());
        assert!(validate_data_array(&data, "test_data", 5).is_err());
    }

    #[test]
    fn test_error_display() {
        let error = SafeMathError::DivisionByZero {
            numerator: 10.0,
            denominator: 0.0,
            context: "test".to_string(),
        };
        
        let error_string = error.to_string();
        assert!(error_string.contains("Division by zero"));
        assert!(error_string.contains("test"));
    }
}