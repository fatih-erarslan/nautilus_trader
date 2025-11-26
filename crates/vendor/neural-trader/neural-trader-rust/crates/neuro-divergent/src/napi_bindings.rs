// NAPI bindings for Neuro-Divergent neural forecasting models
//
// This module provides Node.js bindings for the 27+ neural models
// implemented in the neuro-divergent crate.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::{ModelConfig, TimeSeriesDataFrame, NeuralModel};
use crate::models::nhits::NHITSModel;

/// NAPI wrapper for Neuro-Divergent strategy
#[napi]
pub struct NeuroDivergent {
    strategy_name: String,
    parameters: String,
    model: Option<Box<dyn NeuralModel>>,
}

#[napi]
impl NeuroDivergent {
    /// Create a new Neuro-Divergent strategy instance
    #[napi(constructor)]
    pub fn new(strategy_name: String) -> Result<Self> {
        Ok(Self {
            strategy_name,
            parameters: "{}".to_string(),
            model: None,
        })
    }

    /// Get the strategy name
    #[napi]
    pub fn get_strategy_name(&self) -> String {
        self.strategy_name.clone()
    }

    /// Set strategy parameters (JSON)
    #[napi]
    pub fn set_parameters(&mut self, params: String) -> Result<()> {
        // Validate JSON
        serde_json::from_str::<serde_json::Value>(&params)
            .map_err(|e| Error::from_reason(format!("Invalid JSON: {}", e)))?;

        self.parameters = params;
        Ok(())
    }

    /// Get current parameters
    #[napi]
    pub fn get_parameters(&self) -> String {
        self.parameters.clone()
    }

    /// Initialize the neural model
    #[napi]
    pub fn initialize_model(&mut self, input_size: u32, horizon: u32) -> Result<()> {
        let config = ModelConfig::default()
            .with_input_size(input_size as usize)
            .with_horizon(horizon as usize);

        let model = NHITSModel::new(config)
            .map_err(|e| Error::from_reason(format!("Failed to create model: {}", e)))?;

        self.model = Some(Box::new(model));
        Ok(())
    }

    /// Analyze market data and generate signals
    #[napi]
    pub fn analyze(&self, market_data: String) -> Result<String> {
        // Parse market data
        let _data: serde_json::Value = serde_json::from_str(&market_data)
            .map_err(|e| Error::from_reason(format!("Invalid market data: {}", e)))?;

        // Placeholder analysis - will be implemented with actual strategy logic
        let result = serde_json::json!({
            "strategy": self.strategy_name,
            "signal": "HOLD",
            "confidence": 0.75,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "model": self.model.as_ref().map(|m| m.name()).unwrap_or("None")
        });

        Ok(result.to_string())
    }
}

/// Add two numbers (example function)
#[napi]
pub fn add(left: u32, right: u32) -> u32 {
    left + right
}

/// Get version information
#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Get platform information
#[napi]
pub fn platform_info() -> Result<String> {
    let info = serde_json::json!({
        "platform": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "version": env!("CARGO_PKG_VERSION"),
        "models_count": 27
    });

    Ok(info.to_string())
}

/// List all available neural models
#[napi]
pub fn list_models() -> Result<String> {
    let models = vec![
        "NHITS", "NBEATS", "NBEATSx", "TiDE",
        "LSTM", "GRU", "RNN",
        "TFT", "Informer", "AutoFormer", "FedFormer", "PatchTST", "ITransformer",
        "DeepAR", "DeepNPTS",
        "MLP", "DLinear", "NLinear", "MLPMultivariate",
        "TCN", "BiTCN", "TimesNet", "StemGNN", "TSMixer", "TimeLLM",
    ];

    let result = serde_json::json!({
        "models": models,
        "count": models.len()
    });

    Ok(result.to_string())
}
