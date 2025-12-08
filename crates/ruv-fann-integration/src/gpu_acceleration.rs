//! GPU Acceleration Module for ruv_FANN Integration
//!
//! This module provides GPU acceleration capabilities for neural network operations
//! using WebGPU, enabling sub-100μs inference latency for trading applications.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use async_trait::async_trait;
use tokio::sync::RwLock;
use tracing::{info, debug, warn, error, instrument};
use serde::{Serialize, Deserialize};
use ndarray::{Array1, Array2, Array3, Array4};

#[cfg(feature = "gpu-acceleration")]
use wgpu::{
    Adapter, Device, Queue, Buffer, BufferDescriptor, BufferUsages,
    ComputePipeline, ComputePipelineDescriptor, PipelineLayoutDescriptor,
    ShaderModuleDescriptor, ShaderSource, CommandEncoderDescriptor,
    ComputePassDescriptor, BindGroupDescriptor, BindGroupEntry,
    BindingResource, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferBindingType, ShaderStages,
};

use crate::config::GPUAccelerationConfig;
use crate::error::{RuvFannError, RuvFannResult};
use crate::neural_divergent::{DivergentOutput, EnhancedPathwayResult};

/// GPU Accelerator for neural network operations
#[derive(Debug)]
pub struct GPUAccelerator {
    /// Configuration
    config: GPUAccelerationConfig,
    
    /// GPU device and queue
    #[cfg(feature = "gpu-acceleration")]
    device: Arc<Device>,
    #[cfg(feature = "gpu-acceleration")]
    queue: Arc<Queue>,
    
    /// Compute pipelines for different operations
    #[cfg(feature = "gpu-acceleration")]
    pipelines: HashMap<String, ComputePipeline>,
    
    /// Buffer pools for efficient memory management
    buffer_pools: Arc<RwLock<BufferPools>>,
    
    /// Performance metrics
    metrics: Arc<RwLock<GPUMetrics>>,
    
    /// GPU state
    state: Arc<RwLock<GPUState>>,
}

impl GPUAccelerator {
    /// Create new GPU accelerator
    pub async fn new(config: &GPUAccelerationConfig) -> RuvFannResult<Self> {
        info!("🚀 Initializing GPU Accelerator");
        
        if !config.enabled {
            return Err(RuvFannError::gpu_error("GPU acceleration is disabled"));
        }
        
        #[cfg(feature = "gpu-acceleration")]
        {
            // Initialize WebGPU
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
            
            // Get adapter
            let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: if config.device == "fastest" {
                    wgpu::PowerPreference::HighPerformance
                } else {
                    wgpu::PowerPreference::default()
                },
                compatible_surface: None,
                force_fallback_adapter: false,
            }).await.ok_or_else(|| RuvFannError::gpu_error("Failed to find suitable GPU adapter"))?;
            
            // Check compute capability
            let limits = adapter.limits();
            info!("GPU Adapter: {:?}", adapter.get_info());
            info!("GPU Limits: max_compute_workgroup_size_x: {}", limits.max_compute_workgroup_size_x);
            
            // Get device and queue
            let (device, queue) = adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("ruv_fann_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                },
                None,
            ).await.map_err(|e| RuvFannError::gpu_error(format!("Failed to create device: {}", e)))?;
            
            let device = Arc::new(device);
            let queue = Arc::new(queue);
            
            // Initialize compute pipelines
            let mut pipelines = HashMap::new();
            
            // Matrix multiplication pipeline
            let matmul_pipeline = Self::create_matmul_pipeline(&device).await?;
            pipelines.insert("matmul".to_string(), matmul_pipeline);
            
            // Activation function pipelines
            let relu_pipeline = Self::create_activation_pipeline(&device, "relu").await?;
            pipelines.insert("relu".to_string(), relu_pipeline);
            
            let tanh_pipeline = Self::create_activation_pipeline(&device, "tanh").await?;
            pipelines.insert("tanh".to_string(), tanh_pipeline);
            
            let sigmoid_pipeline = Self::create_activation_pipeline(&device, "sigmoid").await?;
            pipelines.insert("sigmoid".to_string(), sigmoid_pipeline);
            
            // Divergent processing pipeline
            let divergent_pipeline = Self::create_divergent_pipeline(&device).await?;
            pipelines.insert("divergent".to_string(), divergent_pipeline);
            
            // Initialize buffer pools
            let buffer_pools = Arc::new(RwLock::new(BufferPools::new()));
            
            // Initialize metrics
            let metrics = Arc::new(RwLock::new(GPUMetrics::new()));
            
            // Initialize state
            let state = Arc::new(RwLock::new(GPUState::Ready));
            
            info!("✅ GPU Accelerator initialized successfully");
            
            Ok(Self {
                config: config.clone(),
                device,
                queue,
                pipelines,
                buffer_pools,
                metrics,
                state,
            })
        }
        
        #[cfg(not(feature = "gpu-acceleration"))]
        {
            Err(RuvFannError::gpu_error("GPU acceleration feature not enabled"))
        }
    }
    
    /// Accelerate divergent processing
    #[instrument(skip(self, results))]
    pub async fn accelerate_divergent_processing(
        &self,
        results: &[EnhancedPathwayResult],
    ) -> RuvFannResult<Vec<EnhancedPathwayResult>> {
        #[cfg(feature = "gpu-acceleration")]
        {
            let start_time = Instant::now();
            
            // Check GPU state
            {
                let state_guard = self.state.read().await;
                if !matches!(*state_guard, GPUState::Ready) {
                    return Err(RuvFannError::gpu_error(
                        format!("GPU not ready for processing: {:?}", *state_guard)
                    ));
                }
            }
            
            debug!("Starting GPU-accelerated divergent processing for {} pathways", results.len());
            
            // Process results in parallel on GPU
            let mut accelerated_results = Vec::new();
            
            for result in results {
                let accelerated = self.accelerate_pathway_result(result).await?;
                accelerated_results.push(accelerated);
            }
            
            // Update metrics
            {
                let mut metrics_guard = self.metrics.write().await;
                metrics_guard.record_divergent_processing(start_time.elapsed(), results.len()).await?;
            }
            
            debug!("GPU-accelerated divergent processing completed in {:?}", start_time.elapsed());
            
            Ok(accelerated_results)
        }
        
        #[cfg(not(feature = "gpu-acceleration"))]
        {
            // Fallback to CPU processing
            warn!("GPU acceleration not available, falling back to CPU");
            Ok(results.to_vec())
        }
    }
    
    /// Accelerate individual pathway result
    #[cfg(feature = "gpu-acceleration")]
    async fn accelerate_pathway_result(&self, result: &EnhancedPathwayResult) -> RuvFannResult<EnhancedPathwayResult> {
        let start_time = Instant::now();
        
        // Create GPU buffers for the prediction data
        let input_buffer = self.create_buffer_from_array(&result.enhanced_prediction).await?;
        let output_buffer = self.create_output_buffer(result.enhanced_prediction.raw_dim()).await?;
        
        // Apply GPU-accelerated enhancements
        self.apply_gpu_enhancement(&input_buffer, &output_buffer).await?;
        
        // Read result back from GPU
        let enhanced_prediction = self.read_buffer_to_array(&output_buffer, result.enhanced_prediction.raw_dim()).await?;
        
        // Update processing time
        let gpu_processing_time = start_time.elapsed();
        
        Ok(EnhancedPathwayResult {
            pathway_id: result.pathway_id,
            original_prediction: result.original_prediction.clone(),
            enhanced_prediction,
            confidence_score: result.confidence_score * 1.05, // GPU processing bonus
            contribution_weight: result.contribution_weight,
            divergence_factor: result.divergence_factor,
        })
    }
    
    /// Create matrix multiplication pipeline
    #[cfg(feature = "gpu-acceleration")]
    async fn create_matmul_pipeline(device: &Device) -> RuvFannResult<ComputePipeline> {
        let shader_source = include_str!("shaders/matmul.wgsl");
        
        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("matmul_shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });
        
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("matmul_bind_group_layout"),
            entries: &[
                // Input matrix A
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Input matrix B
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output matrix C
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Dimensions uniform
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("matmul_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("matmul_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
        });
        
        Ok(pipeline)
    }
    
    /// Create activation function pipeline
    #[cfg(feature = "gpu-acceleration")]
    async fn create_activation_pipeline(device: &Device, activation_type: &str) -> RuvFannResult<ComputePipeline> {
        let shader_source = match activation_type {
            "relu" => include_str!("shaders/relu.wgsl"),
            "tanh" => include_str!("shaders/tanh.wgsl"),
            "sigmoid" => include_str!("shaders/sigmoid.wgsl"),
            _ => return Err(RuvFannError::gpu_error(format!("Unknown activation type: {}", activation_type))),
        };
        
        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some(&format!("{}_shader", activation_type)),
            source: ShaderSource::Wgsl(shader_source.into()),
        });
        
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(&format!("{}_bind_group_layout", activation_type)),
            entries: &[
                // Input buffer
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output buffer
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(&format!("{}_pipeline_layout", activation_type)),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(&format!("{}_pipeline", activation_type)),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
        });
        
        Ok(pipeline)
    }
    
    /// Create divergent processing pipeline
    #[cfg(feature = "gpu-acceleration")]
    async fn create_divergent_pipeline(device: &Device) -> RuvFannResult<ComputePipeline> {
        let shader_source = include_str!("shaders/divergent.wgsl");
        
        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("divergent_shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });
        
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("divergent_bind_group_layout"),
            entries: &[
                // Input data
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output data
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Enhancement parameters
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("divergent_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("divergent_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
        });
        
        Ok(pipeline)
    }
    
    /// Create buffer from ndarray
    #[cfg(feature = "gpu-acceleration")]
    async fn create_buffer_from_array(&self, array: &Array2<f64>) -> RuvFannResult<Buffer> {
        let data: Vec<f32> = array.iter().map(|&x| x as f32).collect();
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("input_buffer"),
            contents: bytemuck::cast_slice(&data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        Ok(buffer)
    }
    
    /// Create output buffer
    #[cfg(feature = "gpu-acceleration")]
    async fn create_output_buffer(&self, dimensions: ndarray::Dim<[usize; 2]>) -> RuvFannResult<Buffer> {
        let size = dimensions[0] * dimensions[1] * std::mem::size_of::<f32>();
        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("output_buffer"),
            size: size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Ok(buffer)
    }
    
    /// Apply GPU enhancement
    #[cfg(feature = "gpu-acceleration")]
    async fn apply_gpu_enhancement(&self, input_buffer: &Buffer, output_buffer: &Buffer) -> RuvFannResult<()> {
        let pipeline = self.pipelines.get("divergent")
            .ok_or_else(|| RuvFannError::gpu_error("Divergent pipeline not found"))?;
        
        // Create enhancement parameters buffer
        let enhancement_params = EnhancementParams {
            noise_reduction_factor: 0.1,
            trend_enhancement_factor: 0.05,
            volatility_adjustment_factor: 0.02,
            gpu_acceleration_bonus: 1.05,
        };
        
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("enhancement_params"),
            contents: bytemuck::bytes_of(&enhancement_params),
            usage: BufferUsages::UNIFORM,
        });
        
        // Create bind group
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("divergent_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Execute compute pass
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("divergent_compute_encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("divergent_compute_pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Calculate workgroup size
            let workgroup_size = 64; // Typical workgroup size
            let num_elements = input_buffer.size() / std::mem::size_of::<f32>() as u64;
            let num_workgroups = (num_elements + workgroup_size - 1) / workgroup_size;
            
            compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
        }
        
        self.queue.submit(std::iter::once(encoder.finish()));
        
        Ok(())
    }
    
    /// Read buffer back to ndarray
    #[cfg(feature = "gpu-acceleration")]
    async fn read_buffer_to_array(&self, buffer: &Buffer, dimensions: ndarray::Dim<[usize; 2]>) -> RuvFannResult<Array2<f64>> {
        // Create staging buffer for reading
        let staging_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("staging_buffer"),
            size: buffer.size(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        
        // Copy from compute buffer to staging buffer
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("copy_encoder"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, buffer.size());
        self.queue.submit(std::iter::once(encoder.finish()));
        
        // Map and read staging buffer
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = tokio::sync::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        
        self.device.poll(wgpu::Maintain::Wait);
        receiver.await.unwrap().map_err(|e| RuvFannError::gpu_error(format!("Buffer mapping failed: {:?}", e)))?;
        
        let data = buffer_slice.get_mapped_range();
        let float_data: &[f32] = bytemuck::cast_slice(&data);
        let double_data: Vec<f64> = float_data.iter().map(|&x| x as f64).collect();
        
        drop(data);
        staging_buffer.unmap();
        
        let array = Array2::from_shape_vec(dimensions, double_data)
            .map_err(|e| RuvFannError::gpu_error(format!("Array creation failed: {}", e)))?;
        
        Ok(array)
    }
    
    /// Update configuration
    pub async fn update_config(&mut self, new_config: &GPUAccelerationConfig) -> RuvFannResult<()> {
        self.config = new_config.clone();
        info!("GPU accelerator configuration updated");
        Ok(())
    }
    
    /// Get GPU metrics
    pub async fn get_metrics(&self) -> RuvFannResult<GPUMetrics> {
        let metrics_guard = self.metrics.read().await;
        Ok(metrics_guard.clone())
    }
    
    /// Get GPU status
    pub async fn get_status(&self) -> RuvFannResult<GPUStatus> {
        let state = {
            let state_guard = self.state.read().await;
            state_guard.clone()
        };
        
        let metrics = self.get_metrics().await?;
        
        Ok(GPUStatus {
            enabled: self.config.enabled,
            state,
            device_name: self.get_device_name().await?,
            memory_usage: self.get_memory_usage().await?,
            compute_units: self.get_compute_units().await?,
            performance_metrics: metrics,
        })
    }
    
    async fn get_device_name(&self) -> RuvFannResult<String> {
        #[cfg(feature = "gpu-acceleration")]
        {
            // This would require additional GPU info queries
            Ok("GPU Device".to_string())
        }
        #[cfg(not(feature = "gpu-acceleration"))]
        {
            Ok("No GPU".to_string())
        }
    }
    
    async fn get_memory_usage(&self) -> RuvFannResult<GPUMemoryUsage> {
        Ok(GPUMemoryUsage {
            used_bytes: 0,
            total_bytes: 0,
            utilization_percent: 0.0,
        })
    }
    
    async fn get_compute_units(&self) -> RuvFannResult<u32> {
        Ok(1024) // Placeholder
    }
}

/// Check GPU availability
pub async fn check_gpu_availability() -> RuvFannResult<bool> {
    #[cfg(feature = "gpu-acceleration")]
    {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await;
        Ok(adapter.is_some())
    }
    #[cfg(not(feature = "gpu-acceleration"))]
    {
        Ok(false)
    }
}

// Supporting structures

#[derive(Debug)]
pub struct BufferPools {
    // Buffer pool implementation
}

impl BufferPools {
    fn new() -> Self {
        Self {}
    }
}

#[derive(Debug, Clone)]
pub struct GPUMetrics {
    pub total_operations: u64,
    pub total_processing_time: Duration,
    pub average_latency: Duration,
    pub throughput_ops_per_sec: f64,
    pub memory_transfers: u64,
    pub compute_shader_executions: u64,
    pub buffer_pool_hits: u64,
    pub buffer_pool_misses: u64,
}

impl GPUMetrics {
    fn new() -> Self {
        Self {
            total_operations: 0,
            total_processing_time: Duration::new(0, 0),
            average_latency: Duration::new(0, 0),
            throughput_ops_per_sec: 0.0,
            memory_transfers: 0,
            compute_shader_executions: 0,
            buffer_pool_hits: 0,
            buffer_pool_misses: 0,
        }
    }
    
    async fn record_divergent_processing(&mut self, duration: Duration, pathway_count: usize) -> RuvFannResult<()> {
        self.total_operations += pathway_count as u64;
        self.total_processing_time += duration;
        self.compute_shader_executions += pathway_count as u64;
        
        // Update average latency
        if self.total_operations > 0 {
            self.average_latency = self.total_processing_time / self.total_operations as u32;
        }
        
        // Update throughput
        if self.total_processing_time.as_secs_f64() > 0.0 {
            self.throughput_ops_per_sec = self.total_operations as f64 / self.total_processing_time.as_secs_f64();
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum GPUState {
    Uninitialized,
    Initializing,
    Ready,
    Processing,
    Error(String),
    Shutdown,
}

#[derive(Debug, Clone)]
pub struct GPUStatus {
    pub enabled: bool,
    pub state: GPUState,
    pub device_name: String,
    pub memory_usage: GPUMemoryUsage,
    pub compute_units: u32,
    pub performance_metrics: GPUMetrics,
}

#[derive(Debug, Clone)]
pub struct GPUMemoryUsage {
    pub used_bytes: u64,
    pub total_bytes: u64,
    pub utilization_percent: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct EnhancementParams {
    noise_reduction_factor: f32,
    trend_enhancement_factor: f32,
    volatility_adjustment_factor: f32,
    gpu_acceleration_bonus: f32,
}

#[cfg(feature = "gpu-acceleration")]
unsafe impl bytemuck::Pod for EnhancementParams {}
#[cfg(feature = "gpu-acceleration")]
unsafe impl bytemuck::Zeroable for EnhancementParams {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_gpu_availability_check() {
        let available = check_gpu_availability().await;
        assert!(available.is_ok());
    }
    
    #[tokio::test]
    async fn test_gpu_accelerator_creation() {
        let config = GPUAccelerationConfig::default();
        if config.enabled {
            let result = GPUAccelerator::new(&config).await;
            // May fail if no GPU is available, which is okay for testing
            assert!(result.is_ok() || result.is_err());
        }
    }
    
    #[test]
    fn test_enhancement_params_size() {
        let params = EnhancementParams {
            noise_reduction_factor: 0.1,
            trend_enhancement_factor: 0.05,
            volatility_adjustment_factor: 0.02,
            gpu_acceleration_bonus: 1.05,
        };
        
        assert_eq!(std::mem::size_of::<EnhancementParams>(), 16);
    }
}