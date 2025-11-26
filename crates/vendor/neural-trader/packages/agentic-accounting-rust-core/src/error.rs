use thiserror::Error;

/// Custom error types for the Rust core
#[derive(Error, Debug)]
pub enum RustCoreError {
    #[error("Decimal arithmetic error: {0}")]
    DecimalError(String),

    #[error("Date/time parsing error: {0}")]
    DateTimeError(String),

    #[error("Invalid transaction type: {0}")]
    InvalidTransactionType(String),

    #[error("Invalid tax lot: {0}")]
    InvalidTaxLot(String),

    #[error("Calculation error: {0}")]
    CalculationError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("General error: {0}")]
    General(String),
}

/// Result type alias
pub type Result<T> = std::result::Result<T, RustCoreError>;

impl From<rust_decimal::Error> for RustCoreError {
    fn from(err: rust_decimal::Error) -> Self {
        RustCoreError::DecimalError(err.to_string())
    }
}

impl From<chrono::ParseError> for RustCoreError {
    fn from(err: chrono::ParseError) -> Self {
        RustCoreError::DateTimeError(err.to_string())
    }
}

impl From<serde_json::Error> for RustCoreError {
    fn from(err: serde_json::Error) -> Self {
        RustCoreError::SerializationError(err.to_string())
    }
}
