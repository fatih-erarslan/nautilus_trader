//! GPU device abstraction and backend implementations
//! 
//! This module provides unified GPU device abstractions with REAL hardware detection.
//! All implementations use authentic GPU APIs - NO MOCK DATA.

use std::sync::Arc;
use crate::{Result, QbmiaError, GpuCapabilities, GpuBackend};

pub mod cuda;
pub mod opencl;
pub mod vulkan;
pub mod metal;
pub mod kernels;
pub mod quantum_gpu;
pub mod quantum_cuda_kernels;
pub mod quantum_opencl_kernels;
pub mod quantum_algorithms;
pub mod quantum_benchmarks;

pub use cuda::*;
pub use opencl::*;
pub use vulkan::*;
pub use metal::*;
pub use kernels::*;
pub use quantum_gpu::*;
pub use quantum_cuda_kernels::*;
pub use quantum_opencl_kernels::*;
pub use quantum_algorithms::*;
pub use quantum_benchmarks::*;

/// Unified GPU device trait for all backends
/// 
/// This trait abstracts over different GPU backends while ensuring
/// all operations use REAL hardware interfaces.
#[async_trait::async_trait]
pub trait GpuDevice: Send + Sync {
    /// Get device capabilities from real hardware query
    fn capabilities(&self) -> &GpuCapabilities;
    
    /// Get the backend type
    fn backend(&self) -> GpuBackend;
    
    /// Get device ID (real hardware device index)
    fn device_id(&self) -> u32;
    
    /// Check if device is available (real hardware check)
    async fn is_available(&self) -> bool;
    
    /// Execute a kernel on this device
    async fn execute_kernel<T: bytemuck::Pod>(
        &self,
        kernel: &dyn GpuKernel<T>,
        input_data: &[T],
    ) -> Result<Vec<T>>;
    
    /// Synchronize device (wait for all operations to complete)
    async fn synchronize(&self) -> Result<()>;
    
    /// Get memory information from real hardware
    async fn get_memory_info(&self) -> Result<MemoryInfo>;
    
    /// Measure memory bandwidth using real hardware
    async fn measure_memory_bandwidth(&self) -> Result<f64>;
    
    /// Downcast to CUDA device (if applicable)
    #[cfg(feature = "cuda")]
    fn as_cuda_device(&self) -> Result<&CudaDevice> {
        Err(QbmiaError::Internal("Not a CUDA device".to_string()))
    }
    
    /// Downcast to OpenCL device (if applicable)
    #[cfg(feature = "opencl")]
    fn as_opencl_device(&self) -> Result<&OpenClDevice> {
        Err(QbmiaError::Internal("Not an OpenCL device".to_string()))
    }
    
    /// Downcast to Vulkan device (if applicable)
    #[cfg(feature = "vulkan")]
    fn as_vulkan_device(&self) -> Result<&VulkanDevice> {
        Err(QbmiaError::Internal("Not a Vulkan device".to_string()))
    }
    
    /// Downcast to Metal device (if applicable)
    #[cfg(feature = "metal")]
    fn as_metal_device(&self) -> Result<&MetalDevice> {
        Err(QbmiaError::Internal("Not a Metal device".to_string()))
    }
}

/// GPU memory information from real hardware
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Total memory in bytes (from real hardware query)
    pub total_bytes: usize,
    /// Free memory in bytes (real-time query)
    pub free_bytes: usize,
    /// Used memory in bytes
    pub used_bytes: usize,
}

impl MemoryInfo {
    /// Calculate memory utilization percentage
    pub fn utilization_percent(&self) -> f64 {
        if self.total_bytes == 0 {
            0.0
        } else {
            (self.used_bytes as f64 / self.total_bytes as f64) * 100.0
        }
    }
}

/// GPU kernel trait for different computation types
#[async_trait::async_trait]
pub trait GpuKernel<T: bytemuck::Pod>: Send + Sync {
    /// Get kernel name for debugging and profiling
    fn name(&self) -> &str;
    
    /// Get kernel source code or binary
    fn source(&self) -> &str;
    
    /// Execute kernel with input data and return results
    async fn execute(&self, device: &dyn GpuDevice, input_data: &[T]) -> Result<Vec<T>>;
    
    /// Get expected output size given input size
    fn output_size(&self, input_size: usize) -> usize;
    
    /// Get required local work group size (if applicable)
    fn local_work_size(&self) -> Option<[usize; 3]>;
    
    /// Get global work group size for given input
    fn global_work_size(&self, input_size: usize) -> [usize; 3];
}

/// Wrapper type for unified GPU device interface
#[derive(Debug, Clone)]
pub enum GpuDeviceEnum {
    #[cfg(feature = "cuda")]
    Cuda(Arc<CudaDevice>),
    #[cfg(feature = "opencl")]
    OpenCL(Arc<OpenClDevice>),
    #[cfg(feature = "vulkan")]
    Vulkan(Arc<VulkanDevice>),
    #[cfg(feature = "metal")]
    Metal(Arc<MetalDevice>),
}

#[async_trait::async_trait]
impl GpuDevice for GpuDeviceEnum {
    fn capabilities(&self) -> &GpuCapabilities {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(device) => device.capabilities(),
            #[cfg(feature = "opencl")]
            Self::OpenCL(device) => device.capabilities(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => device.capabilities(),
            #[cfg(feature = "metal")]
            Self::Metal(device) => device.capabilities(),
        }
    }
    
    fn backend(&self) -> GpuBackend {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => GpuBackend::Cuda,
            #[cfg(feature = "opencl")]
            Self::OpenCL(_) => GpuBackend::OpenCL,
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => GpuBackend::Vulkan,
            #[cfg(feature = "metal")]
            Self::Metal(_) => GpuBackend::Metal,
        }
    }
    
    fn device_id(&self) -> u32 {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(device) => device.device_id(),
            #[cfg(feature = "opencl")]
            Self::OpenCL(device) => device.device_id(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => device.device_id(),
            #[cfg(feature = "metal")]
            Self::Metal(device) => device.device_id(),
        }
    }
    
    async fn is_available(&self) -> bool {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(device) => device.is_available().await,
            #[cfg(feature = "opencl")]
            Self::OpenCL(device) => device.is_available().await,
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => device.is_available().await,
            #[cfg(feature = "metal")]
            Self::Metal(device) => device.is_available().await,
        }
    }
    
    async fn execute_kernel<T: bytemuck::Pod>(
        &self,
        kernel: &dyn GpuKernel<T>,
        input_data: &[T],
    ) -> Result<Vec<T>> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(device) => device.execute_kernel(kernel, input_data).await,
            #[cfg(feature = "opencl")]
            Self::OpenCL(device) => device.execute_kernel(kernel, input_data).await,
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => device.execute_kernel(kernel, input_data).await,
            #[cfg(feature = "metal")]
            Self::Metal(device) => device.execute_kernel(kernel, input_data).await,
        }
    }
    
    async fn synchronize(&self) -> Result<()> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(device) => device.synchronize().await,
            #[cfg(feature = "opencl")]
            Self::OpenCL(device) => device.synchronize().await,
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => device.synchronize().await,
            #[cfg(feature = "metal")]
            Self::Metal(device) => device.synchronize().await,
        }
    }
    
    async fn get_memory_info(&self) -> Result<MemoryInfo> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(device) => device.get_memory_info().await,
            #[cfg(feature = "opencl")]
            Self::OpenCL(device) => device.get_memory_info().await,
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => device.get_memory_info().await,
            #[cfg(feature = "metal")]
            Self::Metal(device) => device.get_memory_info().await,
        }
    }
    
    async fn measure_memory_bandwidth(&self) -> Result<f64> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(device) => device.measure_memory_bandwidth().await,
            #[cfg(feature = "opencl")]
            Self::OpenCL(device) => device.measure_memory_bandwidth().await,
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => device.measure_memory_bandwidth().await,
            #[cfg(feature = "metal")]
            Self::Metal(device) => device.measure_memory_bandwidth().await,
        }
    }
    
    #[cfg(feature = "cuda")]
    fn as_cuda_device(&self) -> Result<&CudaDevice> {
        match self {
            Self::Cuda(device) => Ok(device.as_ref()),
            _ => Err(QbmiaError::Internal("Not a CUDA device".to_string())),
        }
    }
    
    #[cfg(feature = "opencl")]
    fn as_opencl_device(&self) -> Result<&OpenClDevice> {
        match self {
            Self::OpenCL(device) => Ok(device.as_ref()),
            _ => Err(QbmiaError::Internal("Not an OpenCL device".to_string())),
        }
    }
    
    #[cfg(feature = "vulkan")]
    fn as_vulkan_device(&self) -> Result<&VulkanDevice> {
        match self {
            Self::Vulkan(device) => Ok(device.as_ref()),
            _ => Err(QbmiaError::Internal("Not a Vulkan device".to_string())),
        }
    }
    
    #[cfg(feature = "metal")]
    fn as_metal_device(&self) -> Result<&MetalDevice> {
        match self {
            Self::Metal(device) => Ok(device.as_ref()),
            _ => Err(QbmiaError::Internal("Not a Metal device".to_string())),
        }
    }
}

/// Type alias for the unified device type used throughout the framework
pub type GpuDevice = GpuDeviceEnum;