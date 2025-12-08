//! ML Training Infrastructure for Neuro Trader
//!
//! This crate provides a unified training infrastructure for all machine learning models
//! including deep learning (Transformer, LSTM), gradient boosting (XGBoost, LightGBM),
//! and classical ML algorithms. It supports:
//!
//! - Unified training pipeline with common interface
//! - Experiment tracking and model versioning
//! - Cross-validation for time series
//! - Hyperparameter optimization
//! - Model persistence and deployment
//! - Real-time and batch training
//! - GPU acceleration
//! - Production-ready deployment

#![deny(unsafe_code)]
#![warn(missing_docs)]

/// Data loading and preprocessing
pub mod data;

/// Model implementations
pub mod models;

/// Training pipelines
pub mod training;

/// Model persistence and serialization
pub mod persistence;

/// Experiment tracking
pub mod experiments;

/// Cross-validation strategies
pub mod validation;

/// Hyperparameter optimization
pub mod optimization;

/// Model deployment and serving
pub mod deployment;

/// Performance monitoring
pub mod monitoring;

/// Utilities
pub mod utils;

/// Error types
pub mod error;

/// Configuration
pub mod config;

// Re-exports
pub use error::{Result, TrainingError};
pub use config::TrainingConfig;
pub use training::TrainingPipeline;
pub use models::{Model, ModelType};
pub use experiments::ExperimentTracker;
pub use deployment::ModelRegistry;

use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use std::sync::Arc;

/// Version of the training infrastructure
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize the training infrastructure
pub async fn initialize() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    
    // Initialize GPU if available
    #[cfg(feature = "gpu")]
    {
        crate::utils::gpu::initialize_gpu().await?;
    }
    
    // Initialize experiment tracking database
    crate::experiments::initialize_db().await?;
    
    // Initialize model registry
    crate::deployment::initialize_registry().await?;
    
    tracing::info!("ML Training Infrastructure v{} initialized", VERSION);
    Ok(())
}

/// Main entry point for training models
#[derive(Debug)]
pub struct MLInfrastructure {
    config: Arc<TrainingConfig>,
    pipeline: Arc<TrainingPipeline>,
    tracker: Arc<ExperimentTracker>,
    registry: Arc<ModelRegistry>,
}

impl MLInfrastructure {
    /// Create new infrastructure instance
    pub async fn new(config: TrainingConfig) -> Result<Self> {
        let config = Arc::new(config);
        let pipeline = Arc::new(TrainingPipeline::new(config.clone()).await?);
        let tracker = Arc::new(ExperimentTracker::new(config.clone()).await?);
        let registry = Arc::new(ModelRegistry::new(config.clone()).await?);
        
        Ok(Self {
            config,
            pipeline,
            tracker,
            registry,
        })
    }
    
    /// Train a model with full infrastructure support
    pub async fn train_model(
        &self,
        model_type: ModelType,
        data: data::TrainingData,
        experiment_name: &str,
    ) -> Result<String> {
        // Start experiment
        let experiment_id = self.tracker.start_experiment(experiment_name, model_type).await?;
        
        // Create model
        let mut model = models::create_model(model_type, &self.config)?;
        
        // Train with pipeline
        let metrics = self.pipeline.train(&mut model, data).await?;
        
        // Log results
        self.tracker.log_metrics(&experiment_id, &metrics).await?;
        
        // Save model
        let model_id = self.registry.save_model(&model, &experiment_id).await?;
        
        // Complete experiment
        self.tracker.complete_experiment(&experiment_id).await?;
        
        Ok(model_id)
    }
    
    /// Load and deploy a trained model
    pub async fn deploy_model(&self, model_id: &str) -> Result<Arc<dyn Model>> {
        self.registry.load_model(model_id).await
    }
    
    /// Get experiment history
    pub async fn get_experiments(&self) -> Result<Vec<experiments::Experiment>> {
        self.tracker.list_experiments().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_infrastructure_initialization() {
        assert!(initialize().await.is_ok());
    }
}