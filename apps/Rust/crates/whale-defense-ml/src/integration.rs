//! Integration helpers for using whale defense ML in trading systems
//! 
//! This module provides high-level APIs for integrating the whale detection
//! system into existing trading infrastructure.

use crate::{
    EnsemblePredictor, PredictionResult, FeatureExtractor,
    TransformerConfig, WhaleMLError, Result,
};
use candle_core::{Device, Tensor};
use std::sync::Arc;
use parking_lot::RwLock;
use tokio::sync::mpsc;

/// High-level whale detector for trading systems
pub struct WhaleDetector {
    /// The ensemble predictor
    predictor: Arc<EnsemblePredictor>,
    /// Feature extractor
    feature_extractor: Arc<RwLock<FeatureExtractor>>,
    /// Feature history buffer
    feature_history: Arc<RwLock<Vec<Vec<f32>>>>,
    /// Sequence length
    sequence_length: usize,
    /// Device for computation
    device: Device,
}

impl WhaleDetector {
    /// Create a new whale detector with default configuration
    pub fn new() -> Result<Self> {
        let device = if candle_core::utils::cuda_is_available() {
            Device::new_cuda(0)?
        } else {
            Device::Cpu
        };
        
        Self::with_device(device)
    }
    
    /// Create with specific device
    pub fn with_device(device: Device) -> Result<Self> {
        let config = TransformerConfig::default();
        let predictor = EnsemblePredictor::with_config(
            device.clone(),
            config.clone(),
            Default::default(),
        )?;
        
        Ok(Self {
            predictor: Arc::new(predictor),
            feature_extractor: Arc::new(RwLock::new(FeatureExtractor::new(20))),
            feature_history: Arc::new(RwLock::new(Vec::with_capacity(config.max_seq_length))),
            sequence_length: config.max_seq_length,
            device,
        })
    }
    
    /// Process a new market tick and detect whale activity
    pub async fn process_tick(
        &self,
        price: f32,
        volume: f32,
        bid: Option<f32>,
        ask: Option<f32>,
    ) -> Result<Option<PredictionResult>> {
        // Extract features
        let features = {
            let mut extractor = self.feature_extractor.write();
            extractor.extract_features(price, volume, bid, ask)?
        };
        
        let feature_array = FeatureExtractor::features_to_array(&features);
        
        // Update history
        {
            let mut history = self.feature_history.write();
            history.push(feature_array.to_vec());
            
            // Maintain sequence length
            if history.len() > self.sequence_length {
                history.remove(0);
            }
            
            // Need full sequence to make prediction
            if history.len() < self.sequence_length {
                return Ok(None);
            }
        }
        
        // Create tensor from history
        let tensor = {
            let history = self.feature_history.read();
            let flat_features: Vec<f32> = history
                .iter()
                .flat_map(|feat| feat.iter().copied())
                .collect();
            
            Tensor::from_vec(
                flat_features,
                (1, self.sequence_length, 19),
                &self.device,
            )?
        };
        
        // Make prediction
        let prediction = self.predictor.predict(&tensor)?;
        Ok(Some(prediction))
    }
    
    /// Get current performance metrics
    pub fn get_metrics(&self) -> crate::metrics::PerformanceMetrics {
        self.predictor.get_metrics()
    }
    
    /// Load pre-trained weights
    pub fn load_weights(&self, path: &str) -> Result<()> {
        self.predictor.load_weights(path)
    }
}

/// Async stream processor for real-time whale detection
pub struct WhaleDetectorStream {
    detector: Arc<WhaleDetector>,
    tick_receiver: mpsc::Receiver<MarketTick>,
    alert_sender: mpsc::Sender<WhaleAlert>,
}

/// Market tick data
#[derive(Debug, Clone)]
pub struct MarketTick {
    pub timestamp: i64,
    pub price: f32,
    pub volume: f32,
    pub bid: Option<f32>,
    pub ask: Option<f32>,
}

/// Whale alert
#[derive(Debug, Clone)]
pub struct WhaleAlert {
    pub timestamp: i64,
    pub whale_probability: f32,
    pub threat_level: u8,
    pub confidence: f32,
    pub inference_time_us: u64,
}

impl WhaleDetectorStream {
    /// Create a new stream processor
    pub fn new(
        detector: Arc<WhaleDetector>,
        tick_receiver: mpsc::Receiver<MarketTick>,
        alert_sender: mpsc::Sender<WhaleAlert>,
    ) -> Self {
        Self {
            detector,
            tick_receiver,
            alert_sender,
        }
    }
    
    /// Run the stream processor
    pub async fn run(mut self) -> Result<()> {
        while let Some(tick) = self.tick_receiver.recv().await {
            // Process tick
            if let Some(prediction) = self.detector
                .process_tick(tick.price, tick.volume, tick.bid, tick.ask)
                .await?
            {
                // Send alert if whale detected
                if prediction.whale_probability > 0.5 {
                    let alert = WhaleAlert {
                        timestamp: tick.timestamp,
                        whale_probability: prediction.whale_probability,
                        threat_level: prediction.threat_level,
                        confidence: prediction.confidence,
                        inference_time_us: prediction.inference_time_us,
                    };
                    
                    if let Err(e) = self.alert_sender.send(alert).await {
                        tracing::error!("Failed to send whale alert: {}", e);
                    }
                }
                
                // Log performance warnings
                if prediction.inference_time_us > 500 {
                    tracing::warn!(
                        "Inference time {}μs exceeds 500μs target",
                        prediction.inference_time_us
                    );
                }
            }
        }
        
        Ok(())
    }
}

/// Builder for configuring whale detection
pub struct WhaleDetectorBuilder {
    device: Option<Device>,
    transformer_config: Option<TransformerConfig>,
    sequence_length: Option<usize>,
    weights_path: Option<String>,
}

impl WhaleDetectorBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            device: None,
            transformer_config: None,
            sequence_length: None,
            weights_path: None,
        }
    }
    
    /// Set the device
    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }
    
    /// Set transformer configuration
    pub fn transformer_config(mut self, config: TransformerConfig) -> Self {
        self.transformer_config = Some(config);
        self
    }
    
    /// Set sequence length
    pub fn sequence_length(mut self, length: usize) -> Self {
        self.sequence_length = Some(length);
        self
    }
    
    /// Set weights path to load
    pub fn weights_path(mut self, path: impl Into<String>) -> Self {
        self.weights_path = Some(path.into());
        self
    }
    
    /// Build the whale detector
    pub fn build(self) -> Result<WhaleDetector> {
        let device = self.device.unwrap_or_else(|| {
            if candle_core::utils::cuda_is_available() {
                Device::new_cuda(0).unwrap_or(Device::Cpu)
            } else {
                Device::Cpu
            }
        });
        
        let mut config = self.transformer_config.unwrap_or_default();
        if let Some(seq_len) = self.sequence_length {
            config.max_seq_length = seq_len;
        }
        
        let predictor = EnsemblePredictor::with_config(
            device.clone(),
            config.clone(),
            Default::default(),
        )?;
        
        if let Some(weights_path) = &self.weights_path {
            predictor.load_weights(weights_path)?;
        }
        
        Ok(WhaleDetector {
            predictor: Arc::new(predictor),
            feature_extractor: Arc::new(RwLock::new(FeatureExtractor::new(20))),
            feature_history: Arc::new(RwLock::new(Vec::with_capacity(config.max_seq_length))),
            sequence_length: config.max_seq_length,
            device,
        })
    }
}

impl Default for WhaleDetectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_whale_detector() {
        let detector = WhaleDetector::new().unwrap();
        
        // Process some ticks
        for i in 0..100 {
            let price = 45000.0 + (i as f32 * 10.0);
            let volume = 1_000_000.0;
            
            let result = detector.process_tick(price, volume, None, None).await;
            assert!(result.is_ok());
            
            if i >= 59 {  // After sequence_length ticks
                assert!(result.unwrap().is_some());
            }
        }
    }
    
    #[test]
    fn test_builder() {
        let detector = WhaleDetectorBuilder::new()
            .device(Device::Cpu)
            .sequence_length(30)
            .build();
        
        assert!(detector.is_ok());
    }
}