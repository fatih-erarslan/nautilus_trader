use thiserror::Error;
use hyperphysics_thermo::ThermoError;

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

    #[error("Thermodynamics error: {0}")]
    ThermoError(#[from] ThermoError),
}

pub type Result<T> = std::result::Result<T, RiskError>;
