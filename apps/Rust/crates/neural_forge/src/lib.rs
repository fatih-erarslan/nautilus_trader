//! # Neural Forge
//! 
//! High-performance, modular neural network training framework for financial markets.
//! Integrated with the Nautilus Trader ecosystem for comprehensive trading intelligence.
//! 
//! ## Features
//! 
//! - **Ultra-fast training**: 10-100x faster than Python equivalents
//! - **Modular architecture**: Plug-and-play components
//! - **Multi-backend support**: CUDA, Metal, CPU, WASM
//! - **Advanced calibration**: Temperature scaling, conformal prediction
//! - **Distributed training**: Multi-GPU and multi-node support
//! - **Financial focus**: Built for time-series and trading applications
//! - **Nautilus Integration**: Seamless integration with ATS Core, CDFA, ML Ensemble, and Risk modules
//! 
//! ## Quick Start
//! 
//! ```rust
//! use neural_forge::prelude::*;
//! 
//! let config = TrainingConfig::default()
//!     .with_model(ModelConfig::transformer().with_layers(6))
//!     .with_optimizer(OptimizerConfig::adamw().with_lr(1e-3))
//!     .with_scheduler(SchedulerConfig::cosine_annealing())
//!     .with_calibration(CalibrationConfig::temperature_scaling());
//! 
//! let trainer = Trainer::new(config)?;
//! let results = trainer.train(dataset)?;
//! ```

pub mod prelude;

// Core modules
pub mod config;
pub mod data;
pub mod models;
pub mod training;
pub mod optimization;
pub mod calibration;
pub mod evaluation;
pub mod utils;

// Backend support
pub mod backends;
pub mod cuda;
pub mod distributed;

// Integration modules
pub mod python;
pub mod onnx;
pub mod tensorrt;

// Error handling
pub mod error;
pub use error::{Result, NeuralForgeError};

// Re-exports for convenience
pub use candle_core::{Device, Tensor, DType};
pub use config::*;
pub use training::Trainer;
pub use models::*;
pub use data::*;

// Integration with Nautilus Trader ecosystem
pub mod nautilus_integration {
    //! Integration module for seamless interaction with Nautilus Trader crates
    
    /// ATS Core integration for adaptive temperature scaling and conformal prediction
    pub use ats_core::{
        temperature::TemperatureScaling,
        conformal::ConformalPredictor,
        types::CalibrationConfig as ATSCalibrationConfig,
    };
    
    /// CDFA Core integration for complex financial analysis
    pub use cdfa_core::{
        traits::CDFAAlgorithm,
        types::TimeSeries,
        utils::FinancialMetrics,
    };
    
    /// ML Ensemble integration for ensemble learning
    pub use ml_ensemble::{
        ensemble::EnsembleModel,
        calibration::EnsembleCalibration,
        weights::WeightOptimizer,
    };
    
    /// Quantum Core integration for quantum-enhanced computing
    pub use quantum_core::{
        quantum_types::QuantumState,
        quantum_algorithms::QuantumOptimizer,
    };
    
    /// Risk management integration
    pub use nautilus_risk::{
        calculator::RiskCalculator,
        types::RiskMetrics,
    };
    
    /// Common utilities and core functionality
    pub use common::{
        timer::Timer,
        clock::Clock,
        testing::TestEnvironment,
    };
    
    /// Core Nautilus types and functionality
    pub use nautilus_core::{
        nanos::UnixNanos,
        uuid::UUID4,
        correctness::check_valid_string,
    };
    
    /// Hedge algorithms integration
    pub use hedge_algorithms::{
        experts::ExpertWeights,
        regret::RegretMinimization,
    };
    
    /// LMSR integration for logarithmic market scoring
    pub use lmsr::{
        core::LMSRMarketMaker,
        factors::RiskFactors,
    };
}

/// Re-export the Nautilus integration for convenience
pub use nautilus_integration::*;