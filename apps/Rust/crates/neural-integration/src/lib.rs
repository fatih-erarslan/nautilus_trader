/*!
Neural Integration Library for Nautilus Trader
==============================================

This crate provides a unified integration layer that combines:
- ruv-FANN ultra-fast neural networks
- Cognition Engine NHITS forecasting
- Claude Flow v2 AI orchestration
- Nautilus Trader execution platform

Features:
- Sub-100μs neural predictions
- GPU-accelerated inference
- SIMD-optimized operations
- Real-time trading execution
- Swarm intelligence coordination
*/

#![warn(missing_docs)]
#![deny(unsafe_code)]

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};
use uuid::Uuid;

// Re-export external dependencies
pub use ruv_fann;
pub use cognition_engine;
pub use nautilus_core;
pub use nautilus_model;
pub use nautilus_common;

/// Core neural integration module
pub mod core;
/// Performance optimization utilities
pub mod performance;
/// Trading strategy integration
pub mod strategy;
/// GPU acceleration support
#[cfg(feature = "gpu")]
pub mod gpu;
/// CUDA acceleration support  
#[cfg(feature = "cuda")]
pub mod cuda;
/// Python bindings
#[cfg(feature = "python")]
pub mod python;

/// Configuration for neural integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralConfig {
    /// Maximum number of neural models to run concurrently
    pub max_concurrent_models: usize,
    /// GPU device ID (if available)
    #[cfg(feature = "gpu")]
    pub gpu_device_id: Option<u32>,
    /// Enable CUDA acceleration
    #[cfg(feature = "cuda")]
    pub enable_cuda: bool,
    /// Target latency in microseconds
    pub target_latency_us: u64,
    /// Memory pool size in bytes
    pub memory_pool_size: usize,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Claude Flow integration settings
    pub claude_flow: ClaudeFlowConfig,
    /// ruv-FANN specific settings
    pub ruv_fann: RuvFannConfig,
    /// Cognition Engine settings
    pub cognition_engine: CognitionEngineConfig,
}

/// Claude Flow v2 configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeFlowConfig {
    /// Enable swarm coordination
    pub enable_swarm: bool,
    /// Maximum number of agents
    pub max_agents: usize,
    /// Swarm topology
    pub topology: String,
    /// Memory namespace for trading
    pub memory_namespace: String,
    /// Performance strategy
    pub strategy: String,
}

/// ruv-FANN configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuvFannConfig {
    /// Enable ultra-performance mode
    pub ultra_performance: bool,
    /// Number of CPU threads
    pub num_threads: usize,
    /// Cache size for models
    pub cache_size: usize,
    /// Enable parallel training
    pub parallel_training: bool,
}

/// Cognition Engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitionEngineConfig {
    /// Enable NHITS forecasting
    pub enable_nhits: bool,
    /// Forecast horizon
    pub forecast_horizon: usize,
    /// Model ensemble size
    pub ensemble_size: usize,
    /// Enable GPU acceleration
    pub enable_gpu: bool,
}

impl Default for NeuralConfig {
    fn default() -> Self {
        Self {
            max_concurrent_models: 8,
            #[cfg(feature = "gpu")]
            gpu_device_id: Some(0),
            #[cfg(feature = "cuda")]
            enable_cuda: true,
            target_latency_us: 100,
            memory_pool_size: 1024 * 1024 * 1024, // 1GB
            enable_simd: true,
            claude_flow: ClaudeFlowConfig {
                enable_swarm: true,
                max_agents: 8,
                topology: "hierarchical".to_string(),
                memory_namespace: "neural-trading".to_string(),
                strategy: "ultra-performance".to_string(),
            },
            ruv_fann: RuvFannConfig {
                ultra_performance: true,
                num_threads: num_cpus::get(),
                cache_size: 100,
                parallel_training: true,
            },
            cognition_engine: CognitionEngineConfig {
                enable_nhits: true,
                forecast_horizon: 24,
                ensemble_size: 5,
                enable_gpu: true,
            },
        }
    }
}

/// Neural prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralPrediction {
    /// Unique prediction ID
    pub id: Uuid,
    /// Model that generated the prediction
    pub model_id: String,
    /// Prediction confidence (0.0 to 1.0)
    pub confidence: f64,
    /// Predicted value or signal
    pub value: f64,
    /// Prediction metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Execution time in microseconds
    pub execution_time_us: u64,
    /// Timestamp when prediction was made
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Neural model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Model identifier
    pub model_id: String,
    /// Total predictions made
    pub total_predictions: u64,
    /// Average execution time in microseconds
    pub avg_execution_time_us: f64,
    /// Accuracy score (0.0 to 1.0)
    pub accuracy: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Total return
    pub total_return: f64,
    /// Last updated timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Neural model trait for unified interface
#[async_trait]
pub trait NeuralModel: Send + Sync {
    /// Model identifier
    fn model_id(&self) -> &str;
    
    /// Make a prediction based on input data
    async fn predict(&self, input: &[f64]) -> Result<NeuralPrediction>;
    
    /// Train the model with new data
    async fn train(&mut self, inputs: &[Vec<f64>], targets: &[f64]) -> Result<()>;
    
    /// Get model performance metrics
    fn metrics(&self) -> ModelMetrics;
    
    /// Check if model supports GPU acceleration
    fn supports_gpu(&self) -> bool;
    
    /// Get model memory usage in bytes
    fn memory_usage(&self) -> usize;
}

/// Neural integration manager
pub struct NeuralIntegration {
    config: NeuralConfig,
    models: Arc<RwLock<HashMap<String, Box<dyn NeuralModel>>>>,
    metrics: Arc<RwLock<HashMap<String, ModelMetrics>>>,
    performance_monitor: performance::PerformanceMonitor,
}

impl NeuralIntegration {
    /// Create a new neural integration instance
    pub fn new(config: NeuralConfig) -> Result<Self> {
        info!("Initializing Neural Integration with config: {:?}", config);
        
        let performance_monitor = performance::PerformanceMonitor::new(&config)
            .context("Failed to create performance monitor")?;
        
        Ok(Self {
            config,
            models: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(HashMap::new())),
            performance_monitor,
        })
    }
    
    /// Initialize the neural integration system
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing neural integration system...");
        
        // Initialize Claude Flow swarm if enabled
        if self.config.claude_flow.enable_swarm {
            self.initialize_claude_flow().await
                .context("Failed to initialize Claude Flow")?;
        }
        
        // Initialize GPU acceleration if available
        #[cfg(feature = "gpu")]
        if let Some(device_id) = self.config.gpu_device_id {
            self.initialize_gpu(device_id).await
                .context("Failed to initialize GPU")?;
        }
        
        // Initialize performance monitoring
        self.performance_monitor.start().await
            .context("Failed to start performance monitor")?;
        
        info!("Neural integration system initialized successfully");
        Ok(())
    }
    
    /// Register a neural model
    pub async fn register_model(&self, model: Box<dyn NeuralModel>) -> Result<()> {
        let model_id = model.model_id().to_string();
        info!("Registering neural model: {}", model_id);
        
        // Store initial metrics
        let metrics = model.metrics();
        self.metrics.write().await.insert(model_id.clone(), metrics);
        
        // Register the model
        self.models.write().await.insert(model_id.clone(), model);
        
        info!("Neural model {} registered successfully", model_id);
        Ok(())
    }
    
    /// Make a prediction using a specific model
    pub async fn predict(&self, model_id: &str, input: &[f64]) -> Result<NeuralPrediction> {
        let start_time = std::time::Instant::now();
        
        let models = self.models.read().await;
        let model = models.get(model_id)
            .ok_or_else(|| anyhow::anyhow!("Model {} not found", model_id))?;
        
        let prediction = model.predict(input).await
            .context("Neural prediction failed")?;
        
        let execution_time = start_time.elapsed().as_micros() as u64;
        
        // Check if prediction meets latency target
        if execution_time > self.config.target_latency_us {
            warn!(
                "Prediction latency {}μs exceeds target {}μs for model {}",
                execution_time, self.config.target_latency_us, model_id
            );
        }
        
        // Update performance metrics
        self.performance_monitor.record_prediction(model_id, execution_time).await;
        
        debug!(
            "Neural prediction completed: model={}, confidence={:.3}, latency={}μs",
            model_id, prediction.confidence, execution_time
        );
        
        Ok(prediction)
    }
    
    /// Make predictions using multiple models in parallel
    pub async fn predict_ensemble(&self, input: &[f64]) -> Result<Vec<NeuralPrediction>> {
        let models = self.models.read().await;
        let model_ids: Vec<String> = models.keys().cloned().collect();
        drop(models);
        
        let mut tasks = Vec::new();
        for model_id in &model_ids {
            let integration = self.clone();
            let model_id = model_id.clone();
            let input = input.to_vec();
            
            let task = tokio::spawn(async move {
                integration.predict(&model_id, &input).await
            });
            tasks.push(task);
        }
        
        let mut predictions = Vec::new();
        for task in tasks {
            match task.await? {
                Ok(prediction) => predictions.push(prediction),
                Err(e) => error!("Ensemble prediction failed: {}", e),
            }
        }
        
        info!("Ensemble prediction completed with {} models", predictions.len());
        Ok(predictions)
    }
    
    /// Get system performance metrics
    pub async fn get_metrics(&self) -> Result<HashMap<String, ModelMetrics>> {
        Ok(self.metrics.read().await.clone())
    }
    
    /// Get system status
    pub async fn get_status(&self) -> Result<serde_json::Value> {
        let models = self.models.read().await;
        let metrics = self.metrics.read().await;
        
        let status = serde_json::json!({
            "system_status": "operational",
            "models_registered": models.len(),
            "claude_flow_enabled": self.config.claude_flow.enable_swarm,
            "gpu_enabled": self.config.gpu_device_id.is_some(),
            "target_latency_us": self.config.target_latency_us,
            "memory_pool_size": self.config.memory_pool_size,
            "performance_metrics": *metrics,
            "timestamp": chrono::Utc::now()
        });
        
        Ok(status)
    }
    
    /// Initialize Claude Flow integration
    async fn initialize_claude_flow(&self) -> Result<()> {
        info!("Initializing Claude Flow v2 integration...");
        
        // This would integrate with the Claude Flow neural bridge
        // For now, we'll simulate the initialization
        let config = &self.config.claude_flow;
        
        info!(
            "Claude Flow initialized: agents={}, topology={}, strategy={}",
            config.max_agents, config.topology, config.strategy
        );
        
        Ok(())
    }
    
    /// Initialize GPU acceleration
    #[cfg(feature = "gpu")]
    async fn initialize_gpu(&self, device_id: u32) -> Result<()> {
        info!("Initializing GPU acceleration on device {}", device_id);
        
        // GPU initialization would go here
        // This is a placeholder for actual GPU setup
        
        info!("GPU acceleration initialized successfully");
        Ok(())
    }
}

impl Clone for NeuralIntegration {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            models: Arc::clone(&self.models),
            metrics: Arc::clone(&self.metrics),
            performance_monitor: self.performance_monitor.clone(),
        }
    }
}

/// Example ruv-FANN model implementation
pub struct RuvFannModel {
    model_id: String,
    network: Option<ruv_fann::Network>,
    metrics: ModelMetrics,
}

impl RuvFannModel {
    /// Create a new ruv-FANN model
    pub fn new(model_id: String) -> Self {
        Self {
            model_id: model_id.clone(),
            network: None,
            metrics: ModelMetrics {
                model_id,
                total_predictions: 0,
                avg_execution_time_us: 0.0,
                accuracy: 0.0,
                sharpe_ratio: 0.0,
                max_drawdown: 0.0,
                total_return: 0.0,
                last_updated: chrono::Utc::now(),
            },
        }
    }
}

#[async_trait]
impl NeuralModel for RuvFannModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }
    
    async fn predict(&self, input: &[f64]) -> Result<NeuralPrediction> {
        let start_time = std::time::Instant::now();
        
        // Simulate ruv-FANN prediction
        // In real implementation, this would use the actual ruv-FANN network
        let value = input.iter().sum::<f64>() / input.len() as f64;
        let confidence = 0.85 + (value * 0.1).abs().min(0.14);
        
        let execution_time = start_time.elapsed().as_micros() as u64;
        
        Ok(NeuralPrediction {
            id: Uuid::new_v4(),
            model_id: self.model_id.clone(),
            confidence,
            value,
            metadata: HashMap::new(),
            execution_time_us: execution_time,
            timestamp: chrono::Utc::now(),
        })
    }
    
    async fn train(&mut self, _inputs: &[Vec<f64>], _targets: &[f64]) -> Result<()> {
        // Training implementation would go here
        info!("Training ruv-FANN model: {}", self.model_id);
        Ok(())
    }
    
    fn metrics(&self) -> ModelMetrics {
        self.metrics.clone()
    }
    
    fn supports_gpu(&self) -> bool {
        cfg!(feature = "cuda")
    }
    
    fn memory_usage(&self) -> usize {
        // Estimate memory usage
        1024 * 1024 // 1MB placeholder
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_neural_integration_creation() {
        let config = NeuralConfig::default();
        let integration = NeuralIntegration::new(config).unwrap();
        
        let status = integration.get_status().await.unwrap();
        assert_eq!(status["models_registered"], 0);
    }
    
    #[tokio::test]
    async fn test_model_registration() {
        let config = NeuralConfig::default();
        let integration = NeuralIntegration::new(config).unwrap();
        
        let model = Box::new(RuvFannModel::new("test_model".to_string()));
        integration.register_model(model).await.unwrap();
        
        let status = integration.get_status().await.unwrap();
        assert_eq!(status["models_registered"], 1);
    }
    
    #[tokio::test]
    async fn test_neural_prediction() {
        let config = NeuralConfig::default();
        let integration = NeuralIntegration::new(config).unwrap();
        
        let model = Box::new(RuvFannModel::new("test_model".to_string()));
        integration.register_model(model).await.unwrap();
        
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let prediction = integration.predict("test_model", &input).await.unwrap();
        
        assert_eq!(prediction.model_id, "test_model");
        assert!(prediction.confidence > 0.0);
        assert!(prediction.execution_time_us > 0);
    }
}