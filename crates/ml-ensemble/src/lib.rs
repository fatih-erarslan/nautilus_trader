//! ML Ensemble System
//! 
//! Unified machine learning ensemble that integrates multiple model types:
//! - Neural networks (Transformer, LSTM, GRU, N-BEATS, N-HiTS)
//! - Tree-based models (XGBoost, LightGBM)
//! - Anomaly detection (Isolation Forest)
//! 
//! Features:
//! - Dynamic model selection based on market conditions
//! - Weighted voting with adaptive weight adjustment
//! - Confidence calibration for reliable predictions
//! - Real-time inference with <500μs latency
//! - Feature engineering and interpretability

pub mod calibration;
pub mod ensemble;
pub mod features;
pub mod market_detector;
pub mod model_selector;
pub mod neural_models;
#[cfg(feature = "tree-models")]
pub mod tree_models;
pub mod types;
pub mod weights;

pub use ensemble::EnsemblePredictor;
pub use types::*;

use anyhow::Result;
use std::{collections::HashMap, sync::Arc};
use tracing::info;

/// Create default ensemble predictor with all models
pub async fn create_default_ensemble() -> Result<EnsemblePredictor> {
    info!("Creating default ML ensemble predictor");
    
    let config = EnsembleConfig::default();
    let mut models: HashMap<ModelType, Arc<dyn ensemble::ModelPredictor>> = HashMap::new();
    
    // Load neural models (dummy implementations for now)
    // In production, these would be loaded from saved model files
    
    // Tree models
    #[cfg(feature = "tree-models")]
    {
        use tree_models::{XGBoostModel, LightGBMModel, IsolationForestModel};
        
        models.insert(
            ModelType::XGBoost,
            Arc::new(XGBoostModel::new(Default::default())?),
        );
        
        models.insert(
            ModelType::LightGBM,
            Arc::new(LightGBMModel::new(Default::default())?),
        );
        
        models.insert(
            ModelType::IsolationForest,
            Arc::new(IsolationForestModel::new(100, 256)),
        );
    }
    
    // Note: Neural models would be added here when loaded
    // For now, we'll use only tree models for testing
    
    if models.is_empty() {
        return Err(anyhow::anyhow!("No models available for ensemble"));
    }
    
    info!("Created ensemble with {} models", models.len());
    
    EnsemblePredictor::new(config, models).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use ats_core::types::MarketData;
    
    #[tokio::test]
    async fn test_ensemble_creation() {
        let ensemble = create_default_ensemble().await;
        assert!(ensemble.is_ok());
    }
    
    #[tokio::test]
    async fn test_ensemble_prediction() {
        let ensemble = create_default_ensemble().await.unwrap();
        
        let market_data = MarketData {
            timestamp: 0,
            bid: 100.0,
            ask: 100.1,
            bid_size: 1000.0,
            ask_size: 1000.0,
        };
        
        let prediction = ensemble.predict(&market_data).await;
        assert!(prediction.is_ok());
        
        let pred = prediction.unwrap();
        assert!(pred.confidence >= 0.0 && pred.confidence <= 1.0);
        assert!(pred.latency_us > 0.0);
    }
}