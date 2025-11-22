//! API module for the Autopoiesis trading system
//! 
//! Provides HTTP API interfaces for model serving and management

pub mod nhits_api;

// Re-export commonly used types
pub use nhits_api::{NHITSService, PredictionRequest, TrainingRequest};