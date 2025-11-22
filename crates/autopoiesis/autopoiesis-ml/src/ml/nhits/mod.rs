//! NHITS: Neural Hierarchical Interpolation for Time Series
//! A consciousness-aware hierarchical neural architecture for time series forecasting
//!
//! This module implements a self-adapting NHITS model that integrates with the
//! autopoietic consciousness system. The architecture features:
//! 
//! - Hierarchical neural blocks with basis expansion
//! - Multi-scale time series decomposition
//! - Temporal attention mechanisms
//! - Adaptive structure evolution
//! - Consciousness field integration
//!
//! # Example
//! ```rust
//! use autopoiesis::ml::nhits::{NHITS, NHITSConfig};
//! use autopoiesis::consciousness::ConsciousnessField;
//! use autopoiesis::core::autopoiesis::AutopoieticSystem;
//! 
//! let config = NHITSConfig::default();
//! let consciousness = Arc::new(ConsciousnessField::new());
//! let autopoietic = Arc::new(AutopoieticSystem::new());
//! 
//! let mut model = NHITS::new(config, consciousness, autopoietic);
//! ```

pub mod core;
pub mod blocks;
pub mod attention;
pub mod pooling;
pub mod interpolation;
pub mod decomposition;
pub mod adaptation;
pub mod configs;
pub mod utils;
pub mod forecasting;
pub mod consciousness;
pub mod api;
pub mod financial;
pub mod tests;
pub mod optimization;
pub mod model;

// Re-export main types
pub use self::core::{NHITS, NHITSError, TrainingHistory, ModelState};
pub use self::model::{NHITSModel, NHITSModelTrait, NHITSConfig as ModelConfig, StackBlock, AttentionLayer};
pub use self::configs::{NHITSConfig, NHITSConfigBuilder, UseCase};
pub use self::blocks::BlockConfig;
pub use self::attention::{AttentionConfig, AttentionType};
pub use self::decomposition::{DecomposerConfig, DecompositionType};
pub use self::adaptation::{AdaptationConfig, AdaptationStrategy};
pub use self::forecasting::{
    ForecastingPipeline, ForecastingConfig, ForecastResult,
    PerformanceMetrics, ForecastingEvent, RetrainingConfig,
    PreprocessingConfig, PersistenceConfig,
};

/// Prelude module for convenient imports
pub mod prelude {
    pub use super::core::*;
    pub use super::core::model::*;
    pub use super::configs::*;
    pub use super::blocks::BlockConfig;
    pub use super::attention::AttentionConfig;
    pub use super::decomposition::DecomposerConfig;
    pub use super::adaptation::AdaptationConfig;
    pub use super::forecasting::{ForecastingPipeline, ForecastingConfig, ForecastResult};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    
    #[test]
    fn test_module_structure() {
        // Verify all submodules are accessible
        let _ = configs::NHITSConfig::default();
        let _ = blocks::BlockConfig {
            input_size: 128,
            hidden_size: 128,
            num_basis: 10,
            pooling_factor: 2,
            pooling_type: pooling::PoolingType::Average,
            interpolation_type: interpolation::InterpolationType::Linear,
            dropout_rate: 0.1,
            activation: blocks::ActivationType::GELU,
        };
    }
}