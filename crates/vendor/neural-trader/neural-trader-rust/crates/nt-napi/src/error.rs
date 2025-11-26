//! Error handling utilities for NAPI bindings

use napi::bindgen_prelude::*;
use thiserror::Error;

/// Neural Trader NAPI Error types
#[derive(Error, Debug)]
pub enum NeuralTraderError {
    #[error("Trading error: {0}")]
    Trading(String),

    #[error("Neural network error: {0}")]
    Neural(String),

    #[error("Sports betting error: {0}")]
    Sports(String),

    #[error("Syndicate error: {0}")]
    Syndicate(String),

    #[error("Prediction market error: {0}")]
    Prediction(String),

    #[error("E2B deployment error: {0}")]
    E2B(String),

    #[error("Fantasy sports error: {0}")]
    Fantasy(String),

    #[error("News analysis error: {0}")]
    News(String),

    #[error("Portfolio management error: {0}")]
    Portfolio(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Convert NeuralTraderError to napi::Error
impl From<NeuralTraderError> for napi::Error {
    fn from(err: NeuralTraderError) -> Self {
        napi::Error::from_reason(err.to_string())
    }
}

/// Helper trait for converting Results to NAPI Results
pub trait IntoNapiResult<T> {
    fn into_napi_result(self) -> Result<T>;
}

impl<T, E> IntoNapiResult<T> for std::result::Result<T, E>
where
    E: std::fmt::Display,
{
    fn into_napi_result(self) -> Result<T> {
        self.map_err(|e| napi::Error::from_reason(e.to_string()))
    }
}

/// Create a NAPI error with a custom reason
pub fn napi_error<S: Into<String>>(reason: S) -> napi::Error {
    napi::Error::from_reason(reason.into())
}

/// Create a validation error
pub fn validation_error<S: Into<String>>(reason: S) -> napi::Error {
    NeuralTraderError::Validation(reason.into()).into()
}

/// Create an internal error
pub fn internal_error<S: Into<String>>(reason: S) -> napi::Error {
    NeuralTraderError::Internal(reason.into()).into()
}
