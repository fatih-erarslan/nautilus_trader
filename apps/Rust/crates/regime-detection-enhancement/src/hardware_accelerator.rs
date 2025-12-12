//! Hardware acceleration for regime detection using GPU and specialized processors

use std::sync::Arc;
use std::time::{Duration, Instant};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

/// Hardware-accelerated regime classifier
#[derive(Debug)]
pub struct HardwareAccelerator {
    config: HardwareAcceleratorConfig,
    gpu_context: Option<Arc<GPUContext>>,
    fpga_context: Option<Arc<FPGAContext>>,
    neural_engine: Arc<RwLock<NeuralAccelerationEngine>>,
    performance_metrics: Arc<RwLock<HardwareMetrics>>,
}

impl HardwareAccelerator {
    /// Create new hardware accelerator
    pub async fn new(config: HardwareAcceleratorConfig) -> Result<Self> {
        // Initialize GPU context if available
        let gpu_context = if config.enable_gpu {
            Some(Arc::new(GPUContext::new(&config.gpu_config).await?))
        } else {
            None
        };
        
        // Initialize FPGA context if available
        let fpga_context = if config.enable_fpga {
            Some(Arc::new(FPGAContext::new(&config.fpga_config).await?))
        } else {
            None
        };
        
        // Initialize neural acceleration engine
        let neural_engine = Arc::new(RwLock::new(
            NeuralAccelerationEngine::new(&config.neural_config).await?
        ));
        
        // Initialize performance metrics
        let performance_metrics = Arc::new(RwLock::new(HardwareMetrics::new()));
        
        Ok(Self {
            config,
            gpu_context,
            fpga_context,
            neural_engine,
            performance_metrics,
        })
    }
    
    /// Classify regime using hardware acceleration
    pub async fn classify_regime_accelerated(
        &self,
        features: &[f64],
    ) -> Result<HardwareClassificationResult> {
        let start = Instant::now();
        
        // Choose best acceleration method based on data size and availability
        let result = if features.len() > 1000 && self.gpu_context.is_some() {
            self.classify_gpu(features).await?
        } else if features.len() > 100 && self.fpga_context.is_some() {
            self.classify_fpga(features).await?
        } else {
            self.classify_neural_engine(features).await?
        };
        
        let total_time = start.elapsed();
        
        // Update performance metrics
        {
            let mut metrics = self.performance_metrics.write().await;
            metrics.record_classification(total_time, result.acceleration_type.clone());
        }
        
        Ok(HardwareClassificationResult {
            regime_probabilities: result.regime_probabilities,
            dominant_regime: result.dominant_regime,
            confidence: result.confidence,
            processing_time: total_time,
            acceleration_type: result.acceleration_type,
            features_processed: features.len(),
            hardware_utilization: result.hardware_utilization,
        })
    }
    
    /// GPU-accelerated classification
    async fn classify_gpu(&self, features: &[f64]) -> Result<AccelerationResult> {
        let gpu = self.gpu_context.as_ref().unwrap();
        
        // Transfer data to GPU
        let gpu_features = gpu.transfer_to_device(features).await?;
        
        // Run parallel classification on GPU
        let gpu_probabilities = gpu.compute_regime_probabilities(&gpu_features).await?;
        
        // Transfer results back to host
        let probabilities = gpu.transfer_from_device(&gpu_probabilities).await?;
        
        // Find dominant regime
        let (regime, confidence) = self.find_dominant_regime(&probabilities);
        
        Ok(AccelerationResult {
            regime_probabilities: probabilities,
            dominant_regime: regime,
            confidence,
            acceleration_type: "GPU".to_string(),
            hardware_utilization: gpu.get_utilization().await?,
        })
    }
    
    /// FPGA-accelerated classification
    async fn classify_fpga(&self, features: &[f64]) -> Result<AccelerationResult> {
        let fpga = self.fpga_context.as_ref().unwrap();
        
        // Configure FPGA for regime classification
        fpga.configure_regime_classifier().await?;
        
        // Stream data to FPGA
        let fpga_result = fpga.process_streaming_data(features).await?;
        
        let (regime, confidence) = self.find_dominant_regime(&fpga_result.probabilities);
        
        Ok(AccelerationResult {
            regime_probabilities: fpga_result.probabilities,
            dominant_regime: regime,
            confidence,
            acceleration_type: "FPGA".to_string(),
            hardware_utilization: fpga_result.utilization,
        })
    }
    
    /// Neural engine acceleration
    async fn classify_neural_engine(&self, features: &[f64]) -> Result<AccelerationResult> {
        let neural_engine = self.neural_engine.read().await;
        
        // Use neural processing units for acceleration
        let neural_result = neural_engine.process_features(features).await?;
        
        let (regime, confidence) = self.find_dominant_regime(&neural_result.probabilities);
        
        Ok(AccelerationResult {
            regime_probabilities: neural_result.probabilities,
            dominant_regime: regime,
            confidence,
            acceleration_type: "Neural Engine".to_string(),
            hardware_utilization: neural_result.npu_utilization,
        })
    }
    
    /// Find dominant regime from probability distribution
    fn find_dominant_regime(&self, probabilities: &[f64]) -> (RegimeType, f64) {
        let regime_types = vec![
            RegimeType::BullTrending,
            RegimeType::BearTrending,
            RegimeType::SidewaysLow,
            RegimeType::SidewaysHigh,
            RegimeType::Crisis,
            RegimeType::Recovery,
            RegimeType::Unknown,
        ];
        
        let max_idx = probabilities.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        
        let regime = regime_types.get(max_idx).unwrap_or(&RegimeType::Unknown);
        let confidence = probabilities.get(max_idx).unwrap_or(&0.0);
        
        (*regime, *confidence)
    }
    
    /// Get hardware performance metrics
    pub async fn get_performance_metrics(&self) -> HardwareMetrics {
        self.performance_metrics.read().await.clone()
    }
    
    /// Optimize hardware configuration based on performance
    pub async fn optimize_configuration(&mut self) -> Result<()> {
        let metrics = self.get_performance_metrics().await;
        
        // Adjust configuration based on performance data
        if metrics.average_gpu_time < metrics.average_neural_time {
            self.config.prefer_gpu = true;
        } else {
            self.config.prefer_neural = true;
        }
        
        // Reconfigure hardware contexts
        if let Some(gpu) = &self.gpu_context {
            gpu.optimize_performance(&metrics).await?;
        }
        
        Ok(())
    }
}

/// GPU context for hardware acceleration
#[derive(Debug)]
pub struct GPUContext {
    device_id: u32,
    memory_pool: Vec<f32>,
    compute_streams: usize,
}

impl GPUContext {
    pub async fn new(config: &GPUConfig) -> Result<Self> {
        Ok(Self {
            device_id: config.device_id,
            memory_pool: vec![0.0; config.memory_pool_size],
            compute_streams: config.compute_streams,
        })
    }
    
    pub async fn transfer_to_device(&self, data: &[f64]) -> Result<Vec<f32>> {
        // Simulate GPU data transfer
        Ok(data.iter().map(|&x| x as f32).collect())
    }
    
    pub async fn compute_regime_probabilities(&self, _features: &[f32]) -> Result<Vec<f32>> {
        // Simulate GPU computation
        Ok(vec![0.1, 0.2, 0.3, 0.15, 0.1, 0.1, 0.05])
    }
    
    pub async fn transfer_from_device(&self, gpu_data: &[f32]) -> Result<Vec<f64>> {
        // Simulate GPU to host transfer
        Ok(gpu_data.iter().map(|&x| x as f64).collect())
    }
    
    pub async fn get_utilization(&self) -> Result<f64> {
        // Simulate GPU utilization
        Ok(0.85)
    }
    
    pub async fn optimize_performance(&self, _metrics: &HardwareMetrics) -> Result<()> {
        // Optimize GPU settings based on metrics
        Ok(())
    }
}

/// FPGA context for hardware acceleration
#[derive(Debug)]
pub struct FPGAContext {
    fpga_id: u32,
    pipeline_depth: usize,
}

impl FPGAContext {
    pub async fn new(config: &FPGAConfig) -> Result<Self> {
        Ok(Self {
            fpga_id: config.fpga_id,
            pipeline_depth: config.pipeline_depth,
        })
    }
    
    pub async fn configure_regime_classifier(&self) -> Result<()> {
        // Configure FPGA for regime classification
        Ok(())
    }
    
    pub async fn process_streaming_data(&self, _features: &[f64]) -> Result<FPGAResult> {
        // Simulate FPGA processing
        Ok(FPGAResult {
            probabilities: vec![0.15, 0.25, 0.2, 0.15, 0.1, 0.1, 0.05],
            utilization: 0.92,
            pipeline_latency: Duration::from_nanos(200),
        })
    }
}

/// Neural acceleration engine
#[derive(Debug)]
pub struct NeuralAccelerationEngine {
    npu_count: usize,
    model_cache: Vec<u8>,
}

impl NeuralAccelerationEngine {
    pub async fn new(config: &NeuralConfig) -> Result<Self> {
        Ok(Self {
            npu_count: config.npu_count,
            model_cache: vec![0; config.model_cache_size],
        })
    }
    
    pub async fn process_features(&self, _features: &[f64]) -> Result<NeuralResult> {
        // Simulate neural processing unit computation
        Ok(NeuralResult {
            probabilities: vec![0.2, 0.18, 0.25, 0.12, 0.08, 0.12, 0.05],
            npu_utilization: 0.78,
            inference_time: Duration::from_nanos(150),
        })
    }
}

/// Configuration types
#[derive(Debug, Clone)]
pub struct HardwareAcceleratorConfig {
    pub enable_gpu: bool,
    pub enable_fpga: bool,
    pub enable_neural: bool,
    pub prefer_gpu: bool,
    pub prefer_neural: bool,
    pub gpu_config: GPUConfig,
    pub fpga_config: FPGAConfig,
    pub neural_config: NeuralConfig,
}

#[derive(Debug, Clone)]
pub struct GPUConfig {
    pub device_id: u32,
    pub memory_pool_size: usize,
    pub compute_streams: usize,
}

#[derive(Debug, Clone)]
pub struct FPGAConfig {
    pub fpga_id: u32,
    pub pipeline_depth: usize,
    pub clock_frequency: u64,
}

#[derive(Debug, Clone)]
pub struct NeuralConfig {
    pub npu_count: usize,
    pub model_cache_size: usize,
    pub precision: String,
}

/// Result types
#[derive(Debug, Clone)]
pub struct HardwareClassificationResult {
    pub regime_probabilities: Vec<f64>,
    pub dominant_regime: RegimeType,
    pub confidence: f64,
    pub processing_time: Duration,
    pub acceleration_type: String,
    pub features_processed: usize,
    pub hardware_utilization: f64,
}

#[derive(Debug)]
struct AccelerationResult {
    pub regime_probabilities: Vec<f64>,
    pub dominant_regime: RegimeType,
    pub confidence: f64,
    pub acceleration_type: String,
    pub hardware_utilization: f64,
}

#[derive(Debug)]
struct FPGAResult {
    pub probabilities: Vec<f64>,
    pub utilization: f64,
    pub pipeline_latency: Duration,
}

#[derive(Debug)]
struct NeuralResult {
    pub probabilities: Vec<f64>,
    pub npu_utilization: f64,
    pub inference_time: Duration,
}

/// Hardware performance metrics
#[derive(Debug, Clone)]
pub struct HardwareMetrics {
    pub total_classifications: u64,
    pub gpu_classifications: u64,
    pub fpga_classifications: u64,
    pub neural_classifications: u64,
    pub average_gpu_time: Duration,
    pub average_fpga_time: Duration,
    pub average_neural_time: Duration,
    pub peak_gpu_utilization: f64,
    pub peak_fpga_utilization: f64,
    pub peak_neural_utilization: f64,
}

impl HardwareMetrics {
    pub fn new() -> Self {
        Self {
            total_classifications: 0,
            gpu_classifications: 0,
            fpga_classifications: 0,
            neural_classifications: 0,
            average_gpu_time: Duration::from_nanos(0),
            average_fpga_time: Duration::from_nanos(0),
            average_neural_time: Duration::from_nanos(0),
            peak_gpu_utilization: 0.0,
            peak_fpga_utilization: 0.0,
            peak_neural_utilization: 0.0,
        }
    }
    
    pub fn record_classification(&mut self, duration: Duration, acceleration_type: String) {
        self.total_classifications += 1;
        
        match acceleration_type.as_str() {
            "GPU" => {
                self.gpu_classifications += 1;
                self.update_average(&mut self.average_gpu_time, duration, self.gpu_classifications);
            }
            "FPGA" => {
                self.fpga_classifications += 1;
                self.update_average(&mut self.average_fpga_time, duration, self.fpga_classifications);
            }
            "Neural Engine" => {
                self.neural_classifications += 1;
                self.update_average(&mut self.average_neural_time, duration, self.neural_classifications);
            }
            _ => {}
        }
    }
    
    fn update_average(&self, average: &mut Duration, new_value: Duration, count: u64) {
        let avg_nanos = average.as_nanos() as u64;
        let new_nanos = new_value.as_nanos() as u64;
        let updated_avg = (avg_nanos * (count - 1) + new_nanos) / count;
        *average = Duration::from_nanos(updated_avg);
    }
}

// Re-export regime types
use crate::simd_optimizer::RegimeType;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_hardware_accelerator_creation() {
        let config = HardwareAcceleratorConfig {
            enable_gpu: false, // Disable for testing
            enable_fpga: false,
            enable_neural: true,
            prefer_gpu: false,
            prefer_neural: true,
            gpu_config: GPUConfig {
                device_id: 0,
                memory_pool_size: 1024,
                compute_streams: 4,
            },
            fpga_config: FPGAConfig {
                fpga_id: 0,
                pipeline_depth: 8,
                clock_frequency: 200_000_000,
            },
            neural_config: NeuralConfig {
                npu_count: 4,
                model_cache_size: 1024 * 1024,
                precision: "FP16".to_string(),
            },
        };
        
        let accelerator = HardwareAccelerator::new(config).await;
        assert!(accelerator.is_ok());
    }
    
    #[tokio::test]
    async fn test_hardware_classification() {
        let config = HardwareAcceleratorConfig {
            enable_gpu: false,
            enable_fpga: false,
            enable_neural: true,
            prefer_gpu: false,
            prefer_neural: true,
            gpu_config: GPUConfig {
                device_id: 0,
                memory_pool_size: 1024,
                compute_streams: 4,
            },
            fpga_config: FPGAConfig {
                fpga_id: 0,
                pipeline_depth: 8,
                clock_frequency: 200_000_000,
            },
            neural_config: NeuralConfig {
                npu_count: 4,
                model_cache_size: 1024 * 1024,
                precision: "FP16".to_string(),
            },
        };
        
        let accelerator = HardwareAccelerator::new(config).await.unwrap();
        
        let features = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let result = accelerator.classify_regime_accelerated(&features).await;
        
        assert!(result.is_ok());
        let classification = result.unwrap();
        assert!(classification.confidence > 0.0);
        assert_eq!(classification.features_processed, 5);
        println!("Classification: {:?} (confidence: {:.3})", 
                classification.dominant_regime, classification.confidence);
    }
}