//! GPU acceleration module for CDFA operations
//!
//! This module provides a unified interface for GPU acceleration across different
//! hardware platforms including NVIDIA (CUDA), Apple Silicon (Metal), and 
//! cross-platform WebGPU support.

use crate::error::{CdfaError, CdfaResult};
use crate::types::{Float as CdfaFloat, FloatArray2 as CdfaMatrix, FloatArray1 as CdfaArray};
use std::sync::Arc;
use std::collections::HashMap;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "webgpu")]
pub mod webgpu;

pub mod memory;
pub mod kernels;
pub mod detection;

// Re-export key types
pub use memory::*;
pub use kernels::*;
pub use detection::*;

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    pub id: u32,
    pub name: String,
    pub backend: GpuBackend,
    pub memory_size: u64,
    pub compute_capability: String,
    pub max_work_group_size: u32,
    pub supports_double_precision: bool,
    pub supports_half_precision: bool,
}

/// GPU backend enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "metal")]
    Metal,
    #[cfg(feature = "webgpu")]
    WebGpu,
    Cpu, // Fallback
}

/// GPU context for managing device state
pub trait GpuContext: Send + Sync {
    /// Get device information
    fn device_info(&self) -> &GpuDeviceInfo;
    
    /// Allocate GPU memory buffer
    fn allocate_buffer(&self, size: usize) -> CdfaResult<Box<dyn GpuBuffer>>;
    
    /// Create compute kernel
    fn create_kernel(&self, source: &str, entry_point: &str) -> CdfaResult<Box<dyn GpuKernel>>;
    
    /// Synchronize all pending operations
    fn synchronize(&self) -> CdfaResult<()>;
    
    /// Get memory usage statistics
    fn memory_stats(&self) -> CdfaResult<MemoryStats>;
}

/// GPU memory buffer interface
pub trait GpuBuffer: Send + Sync {
    /// Get buffer size in bytes
    fn size(&self) -> usize;
    
    /// Copy data from host to device
    fn copy_from_host(&mut self, data: &[u8]) -> CdfaResult<()>;
    
    /// Copy data from device to host
    fn copy_to_host(&self, data: &mut [u8]) -> CdfaResult<()>;
    
    /// Map buffer for CPU access (if supported)
    fn map(&self) -> CdfaResult<*mut u8>;
    
    /// Unmap buffer
    fn unmap(&self) -> CdfaResult<()>;
}

/// GPU compute kernel interface
pub trait GpuKernel: Send + Sync {
    /// Set kernel argument
    fn set_arg(&mut self, index: u32, buffer: &dyn GpuBuffer) -> CdfaResult<()>;
    
    /// Set scalar argument
    fn set_scalar_arg<T: Copy>(&mut self, index: u32, value: T) -> CdfaResult<()>;
    
    /// Launch kernel with specified work group configuration
    fn launch(&self, global_size: &[u32], local_size: Option<&[u32]>) -> CdfaResult<()>;
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_memory: u64,
    pub used_memory: u64,
    pub free_memory: u64,
    pub allocated_buffers: usize,
}

/// GPU operation configuration
#[derive(Debug, Clone)]
pub struct GpuConfig {
    pub preferred_backend: Option<GpuBackend>,
    pub memory_pool_size: Option<u64>,
    pub enable_profiling: bool,
    pub fallback_to_cpu: bool,
    pub precision: GpuPrecision,
}

/// GPU precision mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPrecision {
    Half,    // f16
    Single,  // f32
    Double,  // f64
    Mixed,   // f16 compute, f32 accumulate
}

/// Main GPU manager for CDFA operations
pub struct GpuManager {
    contexts: HashMap<u32, Arc<dyn GpuContext>>,
    config: GpuConfig,
    memory_manager: Arc<GpuMemoryManager>,
    kernel_cache: HashMap<String, Arc<dyn GpuKernel>>,
}

impl GpuManager {
    /// Create new GPU manager with automatic device detection
    pub fn new(config: GpuConfig) -> CdfaResult<Self> {
        let mut contexts = HashMap::new();
        let devices = detect_gpu_devices()?;
        
        // Initialize available GPU contexts
        for device in devices {
            let context = Self::create_context_for_device(&device, &config)?;
            contexts.insert(device.id, context);
        }
        
        if contexts.is_empty() && !config.fallback_to_cpu {
            return Err(CdfaError::InvalidConfiguration(
                "No GPU devices available and CPU fallback disabled".to_string()
            ));
        }
        
        let memory_manager = Arc::new(GpuMemoryManager::new(
            config.memory_pool_size.unwrap_or(1024 * 1024 * 1024) // 1GB default
        ));
        
        Ok(Self {
            contexts,
            config,
            memory_manager,
            kernel_cache: HashMap::new(),
        })
    }
    
    /// Create GPU context for specific device
    fn create_context_for_device(
        device: &GpuDeviceInfo, 
        config: &GpuConfig
    ) -> CdfaResult<Arc<dyn GpuContext>> {
        match device.backend {
            #[cfg(feature = "cuda")]
            GpuBackend::Cuda => {
                Ok(Arc::new(cuda::CudaContext::new(device.id, config)?))
            }
            #[cfg(feature = "metal")]
            GpuBackend::Metal => {
                Ok(Arc::new(metal::MetalContext::new(device.id, config)?))
            }
            #[cfg(feature = "webgpu")]
            GpuBackend::WebGpu => {
                Ok(Arc::new(webgpu::WebGpuContext::new(device.id, config)?))
            }
            GpuBackend::Cpu => {
                Err(CdfaError::UnsupportedOperation(
                    "CPU context not implemented for GPU operations".to_string()
                ))
            }
        }
    }
    
    /// Get best available GPU context
    pub fn best_context(&self) -> CdfaResult<Arc<dyn GpuContext>> {
        // Prioritize based on performance and availability
        if let Some(preferred) = self.config.preferred_backend {
            for context in self.contexts.values() {
                if context.device_info().backend == preferred {
                    return Ok(context.clone());
                }
            }
        }
        
        // Fallback to best available context
        self.contexts.values()
            .next()
            .cloned()
            .ok_or_else(|| CdfaError::InvalidConfiguration(
                "No GPU contexts available".to_string()
            ))
    }
    
    /// Get context for specific device
    pub fn context(&self, device_id: u32) -> CdfaResult<Arc<dyn GpuContext>> {
        self.contexts.get(&device_id)
            .cloned()
            .ok_or_else(|| CdfaError::InvalidConfiguration(
                format!("No context found for device {}", device_id)
            ))
    }
    
    /// Get all available contexts
    pub fn all_contexts(&self) -> Vec<Arc<dyn GpuContext>> {
        self.contexts.values().cloned().collect()
    }
    
    /// Execute matrix multiplication on GPU
    pub async fn matrix_multiply(
        &self,
        a: &CdfaMatrix,
        b: &CdfaMatrix,
        device_id: Option<u32>
    ) -> CdfaResult<CdfaMatrix> {
        let context = match device_id {
            Some(id) => self.context(id)?,
            None => self.best_context()?,
        };
        
        kernels::matrix_ops::gpu_matrix_multiply(context, a, b).await
    }
    
    /// Execute element-wise operations on GPU
    pub async fn element_wise_op<F>(
        &self,
        a: &CdfaMatrix,
        b: &CdfaMatrix,
        op: F,
        device_id: Option<u32>
    ) -> CdfaResult<CdfaMatrix>
    where
        F: Fn(CdfaFloat, CdfaFloat) -> CdfaFloat + Send + Sync,
    {
        let context = match device_id {
            Some(id) => self.context(id)?,
            None => self.best_context()?,
        };
        
        kernels::element_ops::gpu_element_wise_op(context, a, b, op).await
    }
    
    /// Execute reduction operations on GPU
    pub async fn reduce_sum(
        &self,
        matrix: &CdfaMatrix,
        device_id: Option<u32>
    ) -> CdfaResult<CdfaFloat> {
        let context = match device_id {
            Some(id) => self.context(id)?,
            None => self.best_context()?,
        };
        
        kernels::reduction_ops::gpu_reduce_sum(context, matrix).await
    }
    
    /// Batch process multiple matrices
    pub async fn batch_process<F, R>(
        &self,
        matrices: &[CdfaMatrix],
        operation: F,
        device_id: Option<u32>
    ) -> CdfaResult<Vec<R>>
    where
        F: Fn(&CdfaMatrix) -> CdfaResult<R> + Send + Sync + Clone,
        R: Send + Sync,
    {
        let context = match device_id {
            Some(id) => self.context(id)?,
            None => self.best_context()?,
        };
        
        kernels::batch_ops::gpu_batch_process(context, matrices, operation).await
    }
    
    /// Get GPU memory statistics
    pub fn memory_stats(&self) -> CdfaResult<HashMap<u32, MemoryStats>> {
        let mut stats = HashMap::new();
        
        for (device_id, context) in &self.contexts {
            stats.insert(*device_id, context.memory_stats()?);
        }
        
        Ok(stats)
    }
    
    /// Cleanup GPU resources
    pub fn cleanup(&mut self) -> CdfaResult<()> {
        // Synchronize all contexts
        for context in self.contexts.values() {
            context.synchronize()?;
        }
        
        // Clear caches
        self.kernel_cache.clear();
        
        Ok(())
    }
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            preferred_backend: None,
            memory_pool_size: None,
            enable_profiling: false,
            fallback_to_cpu: true,
            precision: GpuPrecision::Single,
        }
    }
}

/// Utility functions for GPU operations
pub mod utils {
    use super::*;
    
    /// Convert CDFA matrix to GPU-compatible format
    pub fn matrix_to_gpu_format(matrix: &CdfaMatrix) -> Vec<f32> {
        matrix.iter().map(|&x| x as f32).collect()
    }
    
    /// Convert GPU result back to CDFA matrix
    pub fn gpu_format_to_matrix(data: Vec<f32>, shape: (usize, usize)) -> CdfaMatrix {
        CdfaMatrix::from_shape_vec(shape, data.into_iter().map(|x| x as CdfaFloat).collect())
            .expect("Invalid matrix shape")
    }
    
    /// Convert slice to bytes for GPU transfer
    pub fn to_bytes<T>(data: &[T]) -> &[u8] 
    where 
        T: bytemuck::Pod 
    {
        bytemuck::cast_slice(data)
    }
    
    /// Convert bytes back to typed slice
    pub fn from_bytes<T>(data: &[u8]) -> &[T] 
    where 
        T: bytemuck::Pod 
    {
        bytemuck::cast_slice(data)
    }
    
    /// Calculate optimal work group size for given problem size
    pub fn calculate_work_group_size(problem_size: usize, max_work_group_size: u32) -> u32 {
        let preferred_sizes = [256, 128, 64, 32, 16, 8, 4, 2, 1];
        
        for &size in &preferred_sizes {
            if size <= max_work_group_size && problem_size % size as usize == 0 {
                return size;
            }
        }
        
        // Fallback to maximum available size
        std::cmp::min(max_work_group_size, problem_size as u32)
    }
    
    /// Get GPU backend priority order
    pub fn get_backend_priority() -> Vec<GpuBackend> {
        vec![
            #[cfg(feature = "cuda")]
            GpuBackend::Cuda,
            #[cfg(feature = "metal")]
            GpuBackend::Metal,
            #[cfg(feature = "webgpu")]
            GpuBackend::WebGpu,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    
    #[test]
    fn test_gpu_config_default() {
        let config = GpuConfig::default();
        assert!(config.fallback_to_cpu);
        assert_eq!(config.precision, GpuPrecision::Single);
    }
    
    #[test]
    fn test_matrix_conversion() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let gpu_data = utils::matrix_to_gpu_format(&matrix);
        let converted_back = utils::gpu_format_to_matrix(gpu_data, (2, 2));
        
        assert_eq!(matrix, converted_back);
    }
    
    #[test]
    fn test_work_group_size_calculation() {
        assert_eq!(utils::calculate_work_group_size(1024, 256), 256);
        assert_eq!(utils::calculate_work_group_size(100, 256), 4);
        assert_eq!(utils::calculate_work_group_size(7, 256), 1);
    }
}