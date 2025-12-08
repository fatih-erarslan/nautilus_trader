//! # Error handling for Talebian Risk Management
//!
//! Comprehensive error handling for all components of the aggressive
//! Talebian risk management system.

use thiserror::Error;

/// Comprehensive error types for Talebian risk management
#[derive(Error, Debug, Clone)]
pub enum TalebianRiskError {
    #[error("Invalid input data: {0}")]
    InvalidInput(String),

    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    #[error("Calculation error: {0}")]
    CalculationError(String),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("Market data error: {0}")]
    MarketDataError(String),

    #[error("Whale detection error: {0}")]
    WhaleDetectionError(String),

    #[error("Antifragility calculation error: {0}")]
    AntifragilityError(String),

    #[error("Barbell strategy error: {0}")]
    BarbellError(String),

    #[error("Black swan assessment error: {0}")]
    BlackSwanError(String),

    #[error("Kelly criterion error: {0}")]
    KellyError(String),

    #[error("Opportunity analysis error: {0}")]
    OpportunityError(String),

    #[error("SIMD operation error: {0}")]
    SimdError(String),

    #[error("Memory allocation error: {0}")]
    MemoryError(String),

    #[error("Concurrency error: {0}")]
    ConcurrencyError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("External API error: {0}")]
    ExternalApiError(String),

    #[error("Runtime error: {0}")]
    RuntimeError(String),
}

impl TalebianRiskError {
    /// Create invalid input error
    pub fn invalid_input<S: Into<String>>(msg: S) -> Self {
        Self::InvalidInput(msg.into())
    }

    /// Create insufficient data error
    pub fn insufficient_data<S: Into<String>>(msg: S) -> Self {
        Self::InsufficientData(msg.into())
    }

    /// Create calculation error
    pub fn calculation_error<S: Into<String>>(msg: S) -> Self {
        Self::CalculationError(msg.into())
    }

    /// Create configuration error
    pub fn configuration_error<S: Into<String>>(msg: S) -> Self {
        Self::ConfigurationError(msg.into())
    }

    /// Create market data error
    pub fn market_data_error<S: Into<String>>(msg: S) -> Self {
        Self::MarketDataError(msg.into())
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::InvalidInput(_)
            | Self::InsufficientData(_)
            | Self::ConfigurationError(_)
            | Self::MemoryError(_) => false,

            Self::CalculationError(_)
            | Self::MarketDataError(_)
            | Self::WhaleDetectionError(_)
            | Self::AntifragilityError(_)
            | Self::BarbellError(_)
            | Self::BlackSwanError(_)
            | Self::KellyError(_)
            | Self::OpportunityError(_)
            | Self::SimdError(_)
            | Self::ConcurrencyError(_)
            | Self::SerializationError(_)
            | Self::ExternalApiError(_)
            | Self::RuntimeError(_) => true,
        }
    }

    /// Get error category
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::InvalidInput(_) | Self::InsufficientData(_) => ErrorCategory::Input,

            Self::CalculationError(_)
            | Self::AntifragilityError(_)
            | Self::BarbellError(_)
            | Self::BlackSwanError(_)
            | Self::KellyError(_)
            | Self::OpportunityError(_) => ErrorCategory::Calculation,

            Self::ConfigurationError(_) => ErrorCategory::Configuration,

            Self::MarketDataError(_) | Self::WhaleDetectionError(_) => ErrorCategory::Data,

            Self::SimdError(_) | Self::MemoryError(_) => ErrorCategory::Performance,

            Self::ConcurrencyError(_) => ErrorCategory::Concurrency,

            Self::SerializationError(_) => ErrorCategory::Serialization,

            Self::ExternalApiError(_) => ErrorCategory::External,

            Self::RuntimeError(_) => ErrorCategory::Runtime,
        }
    }

    /// Get severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::InvalidInput(_) | Self::ConfigurationError(_) | Self::MemoryError(_) => {
                ErrorSeverity::Critical
            }

            Self::InsufficientData(_)
            | Self::CalculationError(_)
            | Self::AntifragilityError(_)
            | Self::BarbellError(_)
            | Self::BlackSwanError(_)
            | Self::KellyError(_)
            | Self::OpportunityError(_) => ErrorSeverity::High,

            Self::MarketDataError(_)
            | Self::WhaleDetectionError(_)
            | Self::SimdError(_)
            | Self::SerializationError(_)
            | Self::RuntimeError(_) => ErrorSeverity::Medium,

            Self::ConcurrencyError(_) | Self::ExternalApiError(_) => ErrorSeverity::Low,
        }
    }
}

/// Error category for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    Input,
    Calculation,
    Configuration,
    Data,
    Performance,
    Concurrency,
    Serialization,
    External,
    Runtime,
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Result type alias for Talebian risk operations
pub type TalebianResult<T> = Result<T, TalebianRiskError>;

/// Error context for debugging and monitoring
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub error: TalebianRiskError,
    pub timestamp: i64,
    pub component: String,
    pub function: String,
    pub line: Option<u32>,
    pub additional_info: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    /// Create new error context
    pub fn new(error: TalebianRiskError, component: &str, function: &str) -> Self {
        Self {
            error,
            timestamp: chrono::Utc::now().timestamp(),
            component: component.to_string(),
            function: function.to_string(),
            line: None,
            additional_info: std::collections::HashMap::new(),
        }
    }

    /// Add line number information
    pub fn with_line(mut self, line: u32) -> Self {
        self.line = Some(line);
        self
    }

    /// Add additional context information
    pub fn with_info<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.additional_info.insert(key.into(), value.into());
        self
    }

    /// Format error for logging
    pub fn format_for_logging(&self) -> String {
        let mut log_msg = format!(
            "[{}] {} in {}::{}",
            self.timestamp, self.error, self.component, self.function
        );

        if let Some(line) = self.line {
            log_msg.push_str(&format!(" (line {})", line));
        }

        if !self.additional_info.is_empty() {
            log_msg.push_str(" - Additional info: ");
            for (key, value) in &self.additional_info {
                log_msg.push_str(&format!("{}={}, ", key, value));
            }
            log_msg.truncate(log_msg.len() - 2); // Remove trailing ", "
        }

        log_msg
    }
}

/// Macro for creating error context
#[macro_export]
macro_rules! error_context {
    ($error:expr, $component:expr, $function:expr) => {
        ErrorContext::new($error, $component, $function).with_line(line!())
    };
    ($error:expr, $component:expr, $function:expr, $($key:expr => $value:expr),*) => {
        {
            let mut ctx = ErrorContext::new($error, $component, $function).with_line(line!());
            $(
                ctx = ctx.with_info($key, $value);
            )*
            ctx
        }
    };
}

/// Validation utilities
pub mod validation {
    use super::*;

    /// Validate market data
    pub fn validate_market_data(market_data: &crate::MarketData) -> TalebianResult<()> {
        if market_data.price <= 0.0 {
            return Err(TalebianRiskError::invalid_input("Price must be positive"));
        }

        if market_data.volume < 0.0 {
            return Err(TalebianRiskError::invalid_input(
                "Volume cannot be negative",
            ));
        }

        if market_data.volatility < 0.0 {
            return Err(TalebianRiskError::invalid_input(
                "Volatility cannot be negative",
            ));
        }

        if market_data.bid > market_data.ask {
            return Err(TalebianRiskError::invalid_input(
                "Bid cannot be higher than ask",
            ));
        }

        if market_data.bid_volume < 0.0 || market_data.ask_volume < 0.0 {
            return Err(TalebianRiskError::invalid_input(
                "Order book volumes cannot be negative",
            ));
        }

        Ok(())
    }

    /// Validate configuration
    pub fn validate_config(config: &crate::MacchiavelianConfig) -> TalebianResult<()> {
        if config.antifragility_threshold < 0.0 || config.antifragility_threshold > 1.0 {
            return Err(TalebianRiskError::configuration_error(
                "Antifragility threshold must be between 0 and 1",
            ));
        }

        if config.barbell_safe_ratio + config.barbell_risky_ratio > 1.01 {
            // Allow small rounding error
            return Err(TalebianRiskError::configuration_error(
                "Barbell allocations must sum to 1.0 or less",
            ));
        }

        if config.black_swan_threshold < 0.0 || config.black_swan_threshold > 1.0 {
            return Err(TalebianRiskError::configuration_error(
                "Black swan threshold must be between 0 and 1",
            ));
        }

        if config.kelly_fraction < 0.0 || config.kelly_fraction > 1.0 {
            return Err(TalebianRiskError::configuration_error(
                "Kelly fraction must be between 0 and 1",
            ));
        }

        if config.kelly_max_fraction < config.kelly_fraction {
            return Err(TalebianRiskError::configuration_error(
                "Kelly max fraction must be >= Kelly fraction",
            ));
        }

        if config.whale_volume_threshold <= 1.0 {
            return Err(TalebianRiskError::configuration_error(
                "Whale volume threshold must be > 1.0",
            ));
        }

        Ok(())
    }

    /// Validate returns data
    pub fn validate_returns(returns: &[f64]) -> TalebianResult<()> {
        if returns.is_empty() {
            return Err(TalebianRiskError::insufficient_data(
                "Returns array is empty",
            ));
        }

        for (i, &ret) in returns.iter().enumerate() {
            if ret.is_nan() || ret.is_infinite() {
                return Err(TalebianRiskError::invalid_input(format!(
                    "Invalid return value at index {}: {}",
                    i, ret
                )));
            }

            if ret.abs() > 1.0 {
                // 100% return seems unrealistic for single period
                return Err(TalebianRiskError::invalid_input(format!(
                    "Suspicious return value at index {}: {}%",
                    i,
                    ret * 100.0
                )));
            }
        }

        Ok(())
    }

    /// Validate position size
    pub fn validate_position_size(size: f64) -> TalebianResult<()> {
        if size < 0.0 {
            return Err(TalebianRiskError::invalid_input(
                "Position size cannot be negative",
            ));
        }

        if size > 1.0 {
            return Err(TalebianRiskError::invalid_input(
                "Position size cannot exceed 100%",
            ));
        }

        if size.is_nan() || size.is_infinite() {
            return Err(TalebianRiskError::invalid_input(
                "Position size must be a valid number",
            ));
        }

        Ok(())
    }
}

/// Recovery strategies for different error types
pub mod recovery {
    use super::*;

    /// Recovery strategy for calculation errors
    pub fn recover_from_calculation_error(error: &TalebianRiskError) -> Option<RecoveryAction> {
        match error {
            TalebianRiskError::CalculationError(_)
            | TalebianRiskError::AntifragilityError(_)
            | TalebianRiskError::BarbellError(_)
            | TalebianRiskError::BlackSwanError(_)
            | TalebianRiskError::KellyError(_)
            | TalebianRiskError::OpportunityError(_) => Some(RecoveryAction::UseDefaultValues),

            TalebianRiskError::InsufficientData(_) => Some(RecoveryAction::RequestMoreData),

            TalebianRiskError::MarketDataError(_) => Some(RecoveryAction::RetryWithBackoff),

            _ => None,
        }
    }

    /// Get default values for failed calculations
    pub fn get_default_values() -> DefaultValues {
        DefaultValues {
            antifragility_score: 0.5,
            barbell_allocation: (0.65, 0.35), // Aggressive defaults
            black_swan_probability: 0.1,
            kelly_fraction: 0.55,
            opportunity_score: 0.3,
            confidence: 0.5,
            position_size: 0.1,
        }
    }
}

/// Recovery actions
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryAction {
    UseDefaultValues,
    RequestMoreData,
    RetryWithBackoff,
    FallbackToConservative,
    SkipCalculation,
    NotifyOperator,
}

/// Default values for recovery
#[derive(Debug, Clone)]
pub struct DefaultValues {
    pub antifragility_score: f64,
    pub barbell_allocation: (f64, f64),
    pub black_swan_probability: f64,
    pub kelly_fraction: f64,
    pub opportunity_score: f64,
    pub confidence: f64,
    pub position_size: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = TalebianRiskError::invalid_input("Test message");
        assert!(matches!(error, TalebianRiskError::InvalidInput(_)));
        assert!(!error.is_recoverable());
        assert_eq!(error.category(), ErrorCategory::Input);
        assert_eq!(error.severity(), ErrorSeverity::Critical);
    }

    #[test]
    fn test_error_context() {
        let error = TalebianRiskError::calculation_error("Division by zero");
        let context = ErrorContext::new(error, "kelly", "calculate_fraction")
            .with_line(42)
            .with_info("kelly_fraction", "0.55")
            .with_info("volatility", "0.0");

        let log_msg = context.format_for_logging();
        assert!(log_msg.contains("kelly"));
        assert!(log_msg.contains("calculate_fraction"));
        assert!(log_msg.contains("42"));
        assert!(log_msg.contains("kelly_fraction=0.55"));
    }

    #[test]
    fn test_validation() {
        let market_data = crate::MarketData {
            timestamp: 0,
            price: 100.0,
            volume: 1000.0,
            bid: 99.5,
            ask: 100.5,
            bid_volume: 500.0,
            ask_volume: 500.0,
            volatility: 0.02,
            returns: vec![0.01, -0.005, 0.02],
            volume_history: vec![900.0, 1100.0, 1000.0],
        };

        assert!(validation::validate_market_data(&market_data).is_ok());

        let invalid_data = crate::MarketData {
            price: -100.0, // Invalid negative price
            ..market_data
        };

        assert!(validation::validate_market_data(&invalid_data).is_err());
    }

    #[test]
    fn test_config_validation() {
        let config = crate::MacchiavelianConfig::aggressive_defaults();
        assert!(validation::validate_config(&config).is_ok());

        let invalid_config = crate::MacchiavelianConfig {
            antifragility_threshold: 1.5, // Invalid > 1.0
            ..config
        };

        assert!(validation::validate_config(&invalid_config).is_err());
    }

    #[test]
    fn test_recovery_strategies() {
        let calc_error = TalebianRiskError::calculation_error("Test");
        let recovery = recovery::recover_from_calculation_error(&calc_error);
        assert_eq!(recovery, Some(RecoveryAction::UseDefaultValues));

        let defaults = recovery::get_default_values();
        assert_eq!(defaults.antifragility_score, 0.5);
        assert_eq!(defaults.kelly_fraction, 0.55);
    }

    #[test]
    fn test_error_severity_ordering() {
        assert!(ErrorSeverity::Critical > ErrorSeverity::High);
        assert!(ErrorSeverity::High > ErrorSeverity::Medium);
        assert!(ErrorSeverity::Medium > ErrorSeverity::Low);
    }
}
