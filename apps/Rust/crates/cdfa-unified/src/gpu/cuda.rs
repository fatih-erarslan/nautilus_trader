//! CUDA GPU backend implementation for NVIDIA GPUs
//!
//! This module provides CUDA-based GPU acceleration for CDFA operations,
//! optimized for NVIDIA hardware with support for CUDA kernels.

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::Ptx;
#[cfg(feature = "cuda")]
use cuda_types::*;

use crate::error::{CdfaError, CdfaResult};
use crate::types::{CdfaFloat, CdfaMatrix};
use super::{GpuContext, GpuBuffer, GpuKernel, GpuDeviceInfo, GpuBackend, GpuConfig, MemoryStats, GpuPrecision};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

#[cfg(feature = "cuda")]
/// CUDA implementation of GPU context
pub struct CudaContext {
    device: Arc<CudaDevice>,
    device_info: GpuDeviceInfo,
    config: GpuConfig,
    memory_pool: Mutex<HashMap<usize, Vec<CudaBuffer>>>,
}

#[cfg(feature = "cuda")]
impl CudaContext {
    /// Create new CUDA context
    pub fn new(device_id: u32, config: &GpuConfig) -> CdfaResult<Self> {
        let device = CudaDevice::new(device_id as usize)
            .map_err(|e| CdfaError::GpuError(format!("Failed to create CUDA device: {}", e)))?;
        
        let device_info = Self::query_device_info(&device, device_id)?;
        
        Ok(Self {
            device: Arc::new(device),
            device_info,
            config: config.clone(),
            memory_pool: Mutex::new(HashMap::new()),
        })
    }
    
    /// Query device information
    fn query_device_info(device: &CudaDevice, device_id: u32) -> CdfaResult<GpuDeviceInfo> {
        let name = device.name()
            .map_err(|e| CdfaError::GpuError(format!("Failed to get device name: {}", e)))?;
        
        let total_memory = device.total_memory()
            .map_err(|e| CdfaError::GpuError(format!("Failed to get memory info: {}", e)))?;
        
        let (major, minor) = device.compute_capability()
            .map_err(|e| CdfaError::GpuError(format!("Failed to get compute capability: {}", e)))?;
        
        Ok(GpuDeviceInfo {
            id: device_id,
            name,
            backend: GpuBackend::Cuda,
            memory_size: total_memory,
            compute_capability: format!("{}.{}", major, minor),
            max_work_group_size: 1024, // CUDA max threads per block
            supports_double_precision: major >= 1 && minor >= 3,
            supports_half_precision: major >= 5 && minor >= 3,
        })
    }
    
    /// Get memory pool for specific size category
    fn get_memory_pool(&self, size_category: usize) -> Vec<CudaBuffer> {
        self.memory_pool.lock().unwrap()
            .get(&size_category)
            .cloned()
            .unwrap_or_default()
    }
    
    /// Return buffer to memory pool
    fn return_to_pool(&self, buffer: CudaBuffer, size_category: usize) {
        let mut pool = self.memory_pool.lock().unwrap();
        pool.entry(size_category).or_insert_with(Vec::new).push(buffer);
    }
}

#[cfg(feature = "cuda")]
impl GpuContext for CudaContext {
    fn device_info(&self) -> &GpuDeviceInfo {
        &self.device_info
    }
    
    fn allocate_buffer(&self, size: usize) -> CdfaResult<Box<dyn GpuBuffer>> {
        let size_category = (size / 1024 + 1) * 1024; // Round up to KB
        
        // Try to reuse from memory pool
        if let Some(buffer) = self.get_memory_pool(size_category).pop() {
            if buffer.size() >= size {
                return Ok(Box::new(buffer));
            }
        }
        
        // Allocate new buffer
        let cuda_slice = self.device.alloc_zeros::<u8>(size)
            .map_err(|e| CdfaError::GpuError(format!("CUDA allocation failed: {}", e)))?;
        
        Ok(Box::new(CudaBuffer {
            device: self.device.clone(),
            data: cuda_slice,
            size,
        }))
    }
    
    fn create_kernel(&self, source: &str, entry_point: &str) -> CdfaResult<Box<dyn GpuKernel>> {
        let ptx = self.device.compile_ptx_from_src(source)
            .map_err(|e| CdfaError::GpuError(format!("CUDA kernel compilation failed: {}", e)))?;
        
        self.device.load_ptx(ptx, "cdfa_kernel", &[entry_point])
            .map_err(|e| CdfaError::GpuError(format!("CUDA kernel loading failed: {}", e)))?;
        
        let function = self.device.get_func("cdfa_kernel", entry_point)
            .map_err(|e| CdfaError::GpuError(format!("CUDA function not found: {}", e)))?;
        
        Ok(Box::new(CudaKernel {
            device: self.device.clone(),
            function,
            args: Vec::new(),
        }))
    }
    
    fn synchronize(&self) -> CdfaResult<()> {
        self.device.synchronize()
            .map_err(|e| CdfaError::GpuError(format!("CUDA synchronization failed: {}", e)))
    }
    
    fn memory_stats(&self) -> CdfaResult<MemoryStats> {
        let (free, total) = self.device.memory_info()
            .map_err(|e| CdfaError::GpuError(format!("Failed to get memory info: {}", e)))?;
        
        let used = total - free;
        let pool_size = self.memory_pool.lock().unwrap().len();
        
        Ok(MemoryStats {
            total_memory: total,
            used_memory: used,
            free_memory: free,
            allocated_buffers: pool_size,
        })
    }
}

#[cfg(feature = "cuda")]
/// CUDA buffer implementation
pub struct CudaBuffer {
    device: Arc<CudaDevice>,
    data: CudaSlice<u8>,
    size: usize,
}

#[cfg(feature = "cuda")]
impl GpuBuffer for CudaBuffer {
    fn size(&self) -> usize {
        self.size
    }
    
    fn copy_from_host(&mut self, data: &[u8]) -> CdfaResult<()> {
        if data.len() > self.size {
            return Err(CdfaError::InvalidParameter(
                "Data size exceeds buffer capacity".to_string()
            ));
        }
        
        self.device.htod_copy_into(data, &mut self.data)
            .map_err(|e| CdfaError::GpuError(format!("CUDA host-to-device copy failed: {}", e)))
    }
    
    fn copy_to_host(&self, data: &mut [u8]) -> CdfaResult<()> {
        if data.len() > self.size {
            return Err(CdfaError::InvalidParameter(
                "Output buffer too small".to_string()
            ));
        }
        
        self.device.dtoh_sync_copy_into(&self.data, data)
            .map_err(|e| CdfaError::GpuError(format!("CUDA device-to-host copy failed: {}", e)))
    }
    
    fn map(&self) -> CdfaResult<*mut u8> {
        // CUDA doesn't support direct memory mapping like OpenGL
        // This would require unified memory or host-mapped memory
        Err(CdfaError::UnsupportedOperation(
            "Direct memory mapping not supported in CUDA backend".to_string()
        ))
    }
    
    fn unmap(&self) -> CdfaResult<()> {
        // No-op for CUDA
        Ok(())
    }
}

#[cfg(feature = "cuda")]
/// CUDA kernel implementation
pub struct CudaKernel {
    device: Arc<CudaDevice>,
    function: cudarc::driver::CudaFunction,
    args: Vec<*mut std::ffi::c_void>,
}

#[cfg(feature = "cuda")]
impl GpuKernel for CudaKernel {
    fn set_arg(&mut self, index: u32, buffer: &dyn GpuBuffer) -> CdfaResult<()> {
        // CUDA kernel arguments are handled differently
        // This is a simplified implementation
        Err(CdfaError::UnsupportedOperation(
            "CUDA kernel argument setting requires specialized implementation".to_string()
        ))
    }
    
    fn set_scalar_arg<T: Copy>(&mut self, index: u32, value: T) -> CdfaResult<()> {
        // CUDA scalar arguments
        let ptr = Box::into_raw(Box::new(value)) as *mut std::ffi::c_void;
        
        // Ensure args vector is large enough
        while self.args.len() <= index as usize {
            self.args.push(std::ptr::null_mut());
        }
        
        self.args[index as usize] = ptr;
        Ok(())
    }
    
    fn launch(&self, global_size: &[u32], local_size: Option<&[u32]>) -> CdfaResult<()> {
        let grid_size = match global_size.len() {
            1 => (global_size[0], 1, 1),
            2 => (global_size[0], global_size[1], 1),
            3 => (global_size[0], global_size[1], global_size[2]),
            _ => return Err(CdfaError::InvalidParameter(
                "Global size must be 1D, 2D, or 3D".to_string()
            )),
        };
        
        let block_size = match local_size {
            Some(local) => match local.len() {
                1 => (local[0], 1, 1),
                2 => (local[0], local[1], 1),
                3 => (local[0], local[1], local[2]),
                _ => return Err(CdfaError::InvalidParameter(
                    "Local size must be 1D, 2D, or 3D".to_string()
                )),
            },
            None => (256, 1, 1), // Default block size
        };
        
        let config = LaunchConfig {
            grid_dim: grid_size,
            block_dim: block_size,
            shared_mem_bytes: 0,
            stream: &self.device.default_stream(),
        };
        
        // Launch kernel (simplified - real implementation would handle arguments properly)
        unsafe {
            self.function.launch(config, ())
                .map_err(|e| CdfaError::GpuError(format!("CUDA kernel launch failed: {}", e)))?;
        }
        
        Ok(())
    }
}

/// CUDA-specific kernel sources
pub mod kernels {
    /// Matrix multiplication kernel
    pub const MATRIX_MULTIPLY_KERNEL: &str = r#"
        extern "C" __global__ void matrix_multiply(
            const float* a, const float* b, float* c,
            int m, int n, int k
        ) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (row < m && col < n) {
                float sum = 0.0f;
                for (int i = 0; i < k; i++) {
                    sum += a[row * k + i] * b[i * n + col];
                }
                c[row * n + col] = sum;
            }
        }
    "#;
    
    /// Element-wise operations kernel
    pub const ELEMENT_WISE_KERNEL: &str = r#"
        extern "C" __global__ void element_wise_op(
            const float* a, const float* b, float* c,
            int size, int op_type
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (idx < size) {
                switch (op_type) {
                    case 0: // Addition
                        c[idx] = a[idx] + b[idx];
                        break;
                    case 1: // Multiplication
                        c[idx] = a[idx] * b[idx];
                        break;
                    case 2: // Subtraction
                        c[idx] = a[idx] - b[idx];
                        break;
                    case 3: // Division
                        c[idx] = a[idx] / b[idx];
                        break;
                    default:
                        c[idx] = a[idx];
                }
            }
        }
    "#;
    
    /// Reduction sum kernel
    pub const REDUCE_SUM_KERNEL: &str = r#"
        extern "C" __global__ void reduce_sum(
            const float* input, float* output, int size
        ) {
            extern __shared__ float sdata[];
            
            int tid = threadIdx.x;
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            // Load data into shared memory
            sdata[tid] = (idx < size) ? input[idx] : 0.0f;
            __syncthreads();
            
            // Reduction in shared memory
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    sdata[tid] += sdata[tid + s];
                }
                __syncthreads();
            }
            
            // Write result for this block to global memory
            if (tid == 0) {
                output[blockIdx.x] = sdata[0];
            }
        }
    "#;
    
    /// Diversity calculation kernel (Pearson correlation)
    pub const PEARSON_DIVERSITY_KERNEL: &str = r#"
        extern "C" __global__ void pearson_diversity(
            const float* correlation_matrix, float* diversity_scores,
            int n
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (idx < n) {
                float sum = 0.0f;
                float sum_sq = 0.0f;
                int count = 0;
                
                for (int j = 0; j < n; j++) {
                    if (j != idx) {
                        float corr = correlation_matrix[idx * n + j];
                        sum += corr;
                        sum_sq += corr * corr;
                        count++;
                    }
                }
                
                if (count > 0) {
                    float mean = sum / count;
                    float variance = (sum_sq / count) - (mean * mean);
                    diversity_scores[idx] = sqrtf(variance);
                } else {
                    diversity_scores[idx] = 0.0f;
                }
            }
        }
    "#;
}

// Provide stub implementations when CUDA is not available
#[cfg(not(feature = "cuda"))]
pub struct CudaContext;

#[cfg(not(feature = "cuda"))]
impl CudaContext {
    pub fn new(_device_id: u32, _config: &GpuConfig) -> CdfaResult<Self> {
        Err(CdfaError::UnsupportedOperation(
            "CUDA support not compiled in".to_string()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cuda_availability() {
        #[cfg(feature = "cuda")]
        {
            // Test CUDA context creation (will fail if no CUDA device)
            let config = GpuConfig::default();
            match CudaContext::new(0, &config) {
                Ok(_) => println!("CUDA device available"),
                Err(_) => println!("No CUDA device available"),
            }
        }
        
        #[cfg(not(feature = "cuda"))]
        {
            let config = GpuConfig::default();
            assert!(CudaContext::new(0, &config).is_err());
        }
    }
    
    #[test]
    fn test_kernel_sources() {
        assert!(!kernels::MATRIX_MULTIPLY_KERNEL.is_empty());
        assert!(!kernels::ELEMENT_WISE_KERNEL.is_empty());
        assert!(!kernels::REDUCE_SUM_KERNEL.is_empty());
    }
}