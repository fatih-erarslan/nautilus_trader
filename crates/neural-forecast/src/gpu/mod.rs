//! GPU acceleration backend for neural forecasting models
//!
//! This module provides WebGPU-based acceleration for neural network inference,
//! optimized for sub-100μs latency requirements in financial trading applications.

#![cfg(feature = "gpu")]

use std::sync::Arc;
use tokio::sync::RwLock;
use wgpu::{
    Adapter, Device, Queue, Buffer, ComputePipeline, BindGroup, BindGroupLayout,
    CommandEncoder, ComputePass, BufferDescriptor, BufferUsages, BindGroupDescriptor,
    BindGroupEntry, BindingResource, ComputePassDescriptor, CommandEncoderDescriptor,
    util::DeviceExt,
};
use bytemuck::{Pod, Zeroable, cast_slice};
use crate::{Result, NeuralForecastError};
use crate::config::GPUConfig;

pub mod kernels;
pub mod memory;
pub mod pipeline;
pub mod buffers;
pub mod memory_pinning;
pub mod simd_fallback;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod cuda_kernels;

use kernels::*;
use memory::*;
use pipeline::*;
use buffers::*;
pub use memory_pinning::*;
pub use simd_fallback::*;

#[cfg(feature = "cuda")]
pub use cuda::*;
#[cfg(feature = "cuda")]
pub use cuda_kernels::*;

pub mod benchmarks;
pub mod streaming;
pub mod stream_multiplexing;
pub mod performance_benchmarks;

pub use stream_multiplexing::*;
pub use performance_benchmarks::*;

/// WebGPU backend for neural network acceleration
#[derive(Debug)]
pub struct GPUBackend {
    device: Arc<Device>,
    queue: Arc<Queue>,
    adapter: Arc<Adapter>,
    memory_manager: Arc<RwLock<GPUMemoryManager>>,
    pipeline_cache: Arc<RwLock<PipelineCache>>,
    config: GPUConfig,
}

/// GPU computation context
#[derive(Debug)]
pub struct GPUContext {
    backend: Arc<GPUBackend>,
    command_encoder: Option<CommandEncoder>,
    active_pass: Option<ComputePass<'static>>,
}

/// GPU tensor representation
#[derive(Debug, Clone)]
pub struct GPUTensor {
    buffer: Arc<Buffer>,
    shape: Vec<usize>,
    dtype: GPUDataType,
    device: Arc<Device>,
}

/// GPU data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GPUDataType {
    Float32,
    Float16,
    Int32,
    Int16,
    UInt32,
    UInt16,
}

/// GPU operation types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GPUOperation {
    MatMul,
    Add,
    Mul,
    Relu,
    Sigmoid,
    Tanh,
    Softmax,
    LayerNorm,
    BatchNorm,
    Dropout,
    Convolution,
    Pooling,
    Attention,
    Embedding,
    Custom(String),
}

/// GPU kernel execution parameters
#[derive(Debug, Clone)]
pub struct KernelParams {
    pub workgroup_size: [u32; 3],
    pub dispatch_size: [u32; 3],
    pub local_memory_size: u32,
    pub shared_memory_size: u32,
}

impl GPUBackend {
    /// Create new GPU backend
    pub async fn new(config: GPUConfig) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: wgpu::Dx12Compiler::default(),
            flags: wgpu::InstanceFlags::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| NeuralForecastError::GpuError("Failed to find adapter".to_string()))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Neural Forecast GPU Device"),
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| NeuralForecastError::GpuError(e.to_string()))?;

        let memory_manager = Arc::new(RwLock::new(GPUMemoryManager::new(&device, &config)));
        let pipeline_cache = Arc::new(RwLock::new(PipelineCache::new()));

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter: Arc::new(adapter),
            memory_manager,
            pipeline_cache,
            config,
        })
    }

    /// Create GPU context for computation
    pub fn create_context(&self) -> GPUContext {
        GPUContext {
            backend: Arc::new(self.clone()),
            command_encoder: None,
            active_pass: None,
        }
    }

    /// Get device info
    pub fn device_info(&self) -> GPUDeviceInfo {
        let info = self.adapter.get_info();
        GPUDeviceInfo {
            name: info.name,
            device_type: format!("{:?}", info.device_type),
            backend: format!("{:?}", info.backend),
            vendor: info.vendor,
            driver: info.driver,
            driver_info: info.driver_info,
        }
    }

    /// Create tensor from data
    pub fn create_tensor<T: Pod + Zeroable>(&self, data: &[T], shape: Vec<usize>) -> Result<GPUTensor> {
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Neural Forecast Tensor"),
            contents: cast_slice(data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });

        Ok(GPUTensor {
            buffer: Arc::new(buffer),
            shape,
            dtype: GPUDataType::Float32, // Default to Float32
            device: self.device.clone(),
        })
    }

    /// Create empty tensor
    pub fn create_empty_tensor(&self, shape: Vec<usize>, dtype: GPUDataType) -> Result<GPUTensor> {
        let size = shape.iter().product::<usize>() * dtype.size();
        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Neural Forecast Empty Tensor"),
            size: size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(GPUTensor {
            buffer: Arc::new(buffer),
            shape,
            dtype,
            device: self.device.clone(),
        })
    }

    /// Execute GPU kernel
    pub async fn execute_kernel(
        &self,
        operation: GPUOperation,
        inputs: &[&GPUTensor],
        outputs: &[&GPUTensor],
        params: KernelParams,
    ) -> Result<()> {
        let pipeline = self.get_or_create_pipeline(operation.clone()).await?;
        
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Neural Forecast Compute Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Neural Forecast Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            
            // Set bind groups for inputs and outputs
            let bind_group = self.create_bind_group(inputs, outputs, &pipeline)?;
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Dispatch workgroups
            compute_pass.dispatch_workgroups(
                params.dispatch_size[0],
                params.dispatch_size[1],
                params.dispatch_size[2],
            );
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        
        Ok(())
    }

    /// Get or create compute pipeline
    async fn get_or_create_pipeline(&self, operation: GPUOperation) -> Result<ComputePipeline> {
        let mut cache = self.pipeline_cache.write().await;
        
        if let Some(pipeline) = cache.get(&operation) {
            return Ok(pipeline.clone());
        }

        let shader_source = self.get_shader_source(&operation)?;
        let pipeline = self.create_compute_pipeline(&shader_source, &operation)?;
        cache.insert(operation, pipeline.clone());
        
        Ok(pipeline)
    }

    /// Create compute pipeline
    fn create_compute_pipeline(&self, shader_source: &str, operation: &GPUOperation) -> Result<ComputePipeline> {
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("Neural Forecast {} Shader", operation.name())),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = self.create_bind_group_layout(operation)?;
        
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("Neural Forecast {} Pipeline Layout", operation.name())),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("Neural Forecast {} Pipeline", operation.name())),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });

        Ok(pipeline)
    }

    /// Create bind group layout
    fn create_bind_group_layout(&self, operation: &GPUOperation) -> Result<BindGroupLayout> {
        let entries = match operation {
            GPUOperation::MatMul => vec![
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            _ => vec![], // Default layout
        };

        Ok(self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("Neural Forecast {} Bind Group Layout", operation.name())),
            entries: &entries,
        }))
    }

    /// Create bind group
    fn create_bind_group(
        &self,
        inputs: &[&GPUTensor],
        outputs: &[&GPUTensor],
        pipeline: &ComputePipeline,
    ) -> Result<BindGroup> {
        let layout = pipeline.get_bind_group_layout(0);
        let mut entries = Vec::new();

        // Add input tensors
        for (i, tensor) in inputs.iter().enumerate() {
            entries.push(BindGroupEntry {
                binding: i as u32,
                resource: BindingResource::Buffer(tensor.buffer.as_entire_buffer_binding()),
            });
        }

        // Add output tensors
        for (i, tensor) in outputs.iter().enumerate() {
            entries.push(BindGroupEntry {
                binding: (inputs.len() + i) as u32,
                resource: BindingResource::Buffer(tensor.buffer.as_entire_buffer_binding()),
            });
        }

        Ok(self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Neural Forecast Bind Group"),
            layout: &layout,
            entries: &entries,
        }))
    }

    /// Get shader source for operation
    fn get_shader_source(&self, operation: &GPUOperation) -> Result<String> {
        match operation {
            GPUOperation::MatMul => Ok(include_str!("shaders/matmul.wgsl").to_string()),
            GPUOperation::Add => Ok(include_str!("shaders/add.wgsl").to_string()),
            GPUOperation::Mul => Ok(include_str!("shaders/mul.wgsl").to_string()),
            GPUOperation::Relu => Ok(include_str!("shaders/relu.wgsl").to_string()),
            GPUOperation::Sigmoid => Ok(include_str!("shaders/sigmoid.wgsl").to_string()),
            GPUOperation::Tanh => Ok(include_str!("shaders/tanh.wgsl").to_string()),
            GPUOperation::Softmax => Ok(include_str!("shaders/softmax.wgsl").to_string()),
            GPUOperation::LayerNorm => Ok(include_str!("shaders/layer_norm.wgsl").to_string()),
            GPUOperation::BatchNorm => Ok(include_str!("shaders/batch_norm.wgsl").to_string()),
            GPUOperation::Attention => Ok(include_str!("shaders/attention.wgsl").to_string()),
            _ => Err(NeuralForecastError::GpuError(format!(
                "Shader not implemented for operation: {:?}",
                operation
            ))),
        }
    }

    /// Synchronize GPU operations
    pub async fn synchronize(&self) -> Result<()> {
        self.device.poll(wgpu::MaintainBase::Wait);
        Ok(())
    }
}

impl Clone for GPUBackend {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            queue: self.queue.clone(),
            adapter: self.adapter.clone(),
            memory_manager: self.memory_manager.clone(),
            pipeline_cache: self.pipeline_cache.clone(),
            config: self.config.clone(),
        }
    }
}

impl GPUTensor {
    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get tensor dtype
    pub fn dtype(&self) -> GPUDataType {
        self.dtype
    }

    /// Get tensor size in bytes
    pub fn size_bytes(&self) -> usize {
        self.shape.iter().product::<usize>() * self.dtype.size()
    }

    /// Get tensor element count
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// Read tensor data from GPU
    pub async fn read_data<T: Pod + Zeroable>(&self) -> Result<Vec<T>> {
        let buffer_slice = self.buffer.slice(..);
        let (sender, receiver) = tokio::sync::oneshot::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        
        self.device.poll(wgpu::MaintainBase::Wait);
        
        receiver.await
            .map_err(|e| NeuralForecastError::GpuError(e.to_string()))?
            .map_err(|e| NeuralForecastError::GpuError(e.to_string()))?;

        let data = buffer_slice.get_mapped_range();
        let result = cast_slice::<u8, T>(&data).to_vec();
        
        drop(data);
        self.buffer.unmap();
        
        Ok(result)
    }

    /// Write tensor data to GPU
    pub fn write_data<T: Pod + Zeroable>(&self, data: &[T], queue: &Queue) -> Result<()> {
        if data.len() != self.element_count() {
            return Err(NeuralForecastError::GpuError(
                "Data length does not match tensor size".to_string()
            ));
        }
        
        queue.write_buffer(&self.buffer, 0, cast_slice(data));
        Ok(())
    }

    /// Reshape tensor
    pub fn reshape(&mut self, new_shape: Vec<usize>) -> Result<()> {
        let old_size = self.shape.iter().product::<usize>();
        let new_size = new_shape.iter().product::<usize>();
        
        if old_size != new_size {
            return Err(NeuralForecastError::GpuError(
                "Cannot reshape tensor to different size".to_string()
            ));
        }
        
        self.shape = new_shape;
        Ok(())
    }
}

impl GPUDataType {
    /// Get size in bytes
    pub fn size(&self) -> usize {
        match self {
            GPUDataType::Float32 => 4,
            GPUDataType::Float16 => 2,
            GPUDataType::Int32 => 4,
            GPUDataType::Int16 => 2,
            GPUDataType::UInt32 => 4,
            GPUDataType::UInt16 => 2,
        }
    }
}

impl GPUOperation {
    /// Get operation name
    pub fn name(&self) -> &str {
        match self {
            GPUOperation::MatMul => "matmul",
            GPUOperation::Add => "add",
            GPUOperation::Mul => "mul",
            GPUOperation::Relu => "relu",
            GPUOperation::Sigmoid => "sigmoid",
            GPUOperation::Tanh => "tanh",
            GPUOperation::Softmax => "softmax",
            GPUOperation::LayerNorm => "layer_norm",
            GPUOperation::BatchNorm => "batch_norm",
            GPUOperation::Dropout => "dropout",
            GPUOperation::Convolution => "convolution",
            GPUOperation::Pooling => "pooling",
            GPUOperation::Attention => "attention",
            GPUOperation::Embedding => "embedding",
            GPUOperation::Custom(name) => name,
        }
    }
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GPUDeviceInfo {
    pub name: String,
    pub device_type: String,
    pub backend: String,
    pub vendor: u32,
    pub driver: String,
    pub driver_info: String,
}

/// Check GPU availability
pub fn check_gpu_availability() -> Result<()> {
    // This is a simplified check - in practice, you'd want to test actual GPU functionality
    if cfg!(feature = "gpu") {
        Ok(())
    } else {
        Err(NeuralForecastError::GpuError("GPU feature not enabled".to_string()))
    }
}

/// Get available GPU devices
pub async fn get_available_devices() -> Result<Vec<GPUDeviceInfo>> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        dx12_shader_compiler: wgpu::Dx12Compiler::default(),
        flags: wgpu::InstanceFlags::default(),
        gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
    });

    let mut devices = Vec::new();
    
    // This is a simplified implementation - in practice, you'd enumerate all available adapters
    if let Some(adapter) = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }).await {
        let info = adapter.get_info();
        devices.push(GPUDeviceInfo {
            name: info.name,
            device_type: format!("{:?}", info.device_type),
            backend: format!("{:?}", info.backend),
            vendor: info.vendor,
            driver: info.driver,
            driver_info: info.driver_info,
        });
    }
    
    Ok(devices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GPUConfig;
    
    #[tokio::test]
    async fn test_gpu_backend_creation() {
        let config = GPUConfig::default();
        let result = GPUBackend::new(config).await;
        
        // This test might fail if no GPU is available
        match result {
            Ok(_backend) => {
                // GPU backend created successfully
            }
            Err(e) => {
                // Expected if no GPU is available
                println!("GPU not available: {}", e);
            }
        }
    }
    
    #[tokio::test]
    async fn test_gpu_availability() {
        let result = check_gpu_availability();
        
        if cfg!(feature = "gpu") {
            assert!(result.is_ok());
        } else {
            assert!(result.is_err());
        }
    }
    
    #[test]
    fn test_gpu_data_type_size() {
        assert_eq!(GPUDataType::Float32.size(), 4);
        assert_eq!(GPUDataType::Float16.size(), 2);
        assert_eq!(GPUDataType::Int32.size(), 4);
        assert_eq!(GPUDataType::Int16.size(), 2);
    }
    
    #[test]
    fn test_gpu_operation_name() {
        assert_eq!(GPUOperation::MatMul.name(), "matmul");
        assert_eq!(GPUOperation::Add.name(), "add");
        assert_eq!(GPUOperation::Relu.name(), "relu");
    }
}