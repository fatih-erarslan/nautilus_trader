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

    #[error("Risk management error: {0}")]
    Risk(String),

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

    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    #[error("Forbidden: {0}")]
    Forbidden(String),

    #[error("Rate limit exceeded: {0}")]
    RateLimited(String),

    #[error("Authentication error: {0}")]
    Authentication(String),

    #[error("Authorization error: {0}")]
    Authorization(String),
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

/// Create an unauthorized error
pub fn unauthorized_error<S: Into<String>>(reason: S) -> napi::Error {
    NeuralTraderError::Unauthorized(reason.into()).into()
}

/// Create a forbidden error
pub fn forbidden_error<S: Into<String>>(reason: S) -> napi::Error {
    NeuralTraderError::Forbidden(reason.into()).into()
}

/// Create a rate limit error
pub fn rate_limit_error<S: Into<String>>(reason: S) -> napi::Error {
    NeuralTraderError::RateLimited(reason.into()).into()
}

/// Create an authentication error
pub fn authentication_error<S: Into<String>>(reason: S) -> napi::Error {
    NeuralTraderError::Authentication(reason.into()).into()
}

/// Create an authorization error
pub fn authorization_error<S: Into<String>>(reason: S) -> napi::Error {
    NeuralTraderError::Authorization(reason.into()).into()
}
