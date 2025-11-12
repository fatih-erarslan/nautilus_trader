use thiserror::Error;

#[derive(Error, Debug)]
pub enum RiskError {
    #[error("Invalid portfolio weights: {0}")]
    InvalidWeights(String),

    #[error("Insufficient data for calculation: {0}")]
    InsufficientData(String),

    #[error("Entropy constraint violation: {0}")]
    EntropyConstraintViolation(String),

    #[error("Invalid temperature: {0}")]
    InvalidTemperature(String),

    #[error("Calculation error: {0}")]
    CalculationError(String),
}

pub type Result<T> = std::result::Result<T, RiskError>;
