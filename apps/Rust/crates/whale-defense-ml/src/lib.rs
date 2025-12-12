//! Whale Defense ML - Ultra-Fast Transformer-Based Whale Detection System
//! 
//! This crate implements high-performance ML models for whale detection
//! using Candle framework with sub-500μs inference targets.

#![warn(missing_docs)]
#![deny(unsafe_code)]

pub mod transformer;
pub mod features;
pub mod dataset;
pub mod ensemble;
pub mod metrics;
pub mod error;
pub mod integration;

pub use transformer::{TransformerWhaleDetector, TransformerConfig};
pub use features::{FeatureExtractor, MarketFeatures};
pub use dataset::{WhaleDataset, DataPreprocessor, WhaleEvent, WhaleEventType};
pub use ensemble::{EnsemblePredictor, PredictionResult};
pub use metrics::{PerformanceMetrics, InferenceTimer};
pub use error::{WhaleMLError, Result};
pub use integration::{WhaleDetector, WhaleDetectorBuilder, WhaleDetectorStream, MarketTick, WhaleAlert};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}