//! # ruv-FANN Integration for Nautilus Trader
//!
//! This crate provides high-performance integration of ruv-FANN neural networks
//! with the Nautilus Trader platform, enabling real-time neural trading strategies
//! with GPU acceleration and SIMD optimizations.

pub mod config;
pub mod data_flow_bridge;
pub mod error;
pub mod ffi;
pub mod gpu_acceleration;
pub mod inference_engine;
pub mod memory_manager;
pub mod metrics;
pub mod neural_divergent;
pub mod neural_strategy;
pub mod parallel_processing;
pub mod performance_bridge;
pub mod quantum_ml_bridge;
pub mod real_time_inference;
pub mod trading_networks;
pub mod utils;

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;

// Re-exports
pub use config::*;
pub use data_flow_bridge::*;
pub use error::*;
pub use ffi::*;
pub use gpu_acceleration::*;
pub use inference_engine::*;
pub use memory_manager::*;
pub use metrics::*;
pub use neural_divergent::*;
pub use neural_strategy::*;
pub use parallel_processing::*;
pub use performance_bridge::*;
pub use quantum_ml_bridge::*;
pub use real_time_inference::*;
pub use trading_networks::*;
pub use utils::*;

/// Main integration manager for ruv-FANN with Nautilus Trader
#[derive(Debug, Clone)]
pub struct RuvFannIntegration {
    config: Arc<RwLock<IntegrationConfig>>,
    inference_engine: Arc<RwLock<InferenceEngine>>,
    memory_manager: Arc<RwLock<MemoryManager>>,
    metrics_collector: Arc<MetricsCollector>,
    gpu_context: Option<Arc<GpuContext>>,
}

impl RuvFannIntegration {
    /// Create a new ruv-FANN integration instance
    pub async fn new(config: IntegrationConfig) -> Result<Self> {
        let inference_engine = InferenceEngine::new(&config).await?;
        let memory_manager = MemoryManager::new(&config)?;
        let metrics_collector = Arc::new(MetricsCollector::new());
        
        let gpu_context = if config.enable_gpu {
            Some(Arc::new(GpuContext::new().await?))
        } else {
            None
        };

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            inference_engine: Arc::new(RwLock::new(inference_engine)),
            memory_manager: Arc::new(RwLock::new(memory_manager)),
            metrics_collector,
            gpu_context,
        })
    }

    /// Initialize the integration system
    pub async fn initialize(&self) -> Result<()> {
        // Initialize GPU context if enabled
        if let Some(gpu_context) = &self.gpu_context {
            gpu_context.initialize().await?;
        }

        // Initialize inference engine
        self.inference_engine.write().await.initialize().await?;
        
        // Initialize memory manager
        self.memory_manager.write().await.initialize()?;
        
        // Start metrics collection
        self.metrics_collector.start().await?;
        
        tracing::info!("ruv-FANN integration initialized successfully");
        Ok(())
    }

    /// Process trading data through neural networks
    pub async fn process_trading_data(&self, data: &TradingData) -> Result<NeuralPrediction> {
        let start = std::time::Instant::now();
        
        // Run inference
        let prediction = self.inference_engine
            .read()
            .await
            .predict(data)
            .await?;
            
        // Record metrics
        self.metrics_collector.record_inference_time(start.elapsed());
        self.metrics_collector.record_prediction(&prediction);
        
        Ok(prediction)
    }

    /// Shutdown the integration system
    pub async fn shutdown(&self) -> Result<()> {
        self.metrics_collector.stop().await?;
        self.inference_engine.write().await.shutdown().await?;
        
        if let Some(gpu_context) = &self.gpu_context {
            gpu_context.shutdown().await?;
        }
        
        tracing::info!("ruv-FANN integration shutdown completed");
        Ok(())
    }
}
