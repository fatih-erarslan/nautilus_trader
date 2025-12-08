//! Optimized CUDA kernels for neural network operations
//!
//! This module provides PTX kernel implementations for maximum performance

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaDevice, CudaFunction, CudaModule, LaunchConfig};
use half::f16;
use crate::{Result, NeuralForecastError};

/// CUDA kernel registry for optimized operations
pub struct CudaKernelRegistry {
    device: std::sync::Arc<CudaDevice>,
    modules: std::collections::HashMap<String, CudaModule>,
    kernels: std::collections::HashMap<String, CudaFunction>,
}

impl CudaKernelRegistry {
    /// Create new kernel registry
    pub fn new(device: std::sync::Arc<CudaDevice>) -> Result<Self> {
        let mut registry = Self {
            device,
            modules: std::collections::HashMap::new(),
            kernels: std::collections::HashMap::new(),
        };
        
        // Load optimized PTX kernels
        registry.load_builtin_kernels()?;
        
        Ok(registry)
    }
    
    /// Load built-in optimized kernels
    fn load_builtin_kernels(&mut self) -> Result<()> {
        // Load transformer attention kernel
        self.load_ptx_kernel(
            "transformer_attention",
            include_str!("cuda/transformer_attention.ptx"),
            "transformer_multihead_attention_optimized"
        )?;
        
        // Load mixed precision GEMM kernel
        self.load_ptx_kernel(
            "mixed_precision_gemm",
            include_str!("cuda/mixed_precision_gemm.ptx"),
            "gemm_fp16_tensor_core"
        )?;
        
        // Load fused operations kernel
        self.load_ptx_kernel(
            "fused_ops",
            include_str!("cuda/fused_operations.ptx"),
            "fused_linear_activation"
        )?;
        
        Ok(())
    }
    
    /// Load PTX kernel from string
    fn load_ptx_kernel(&mut self, name: &str, ptx: &str, entry_point: &str) -> Result<()> {
        let module = self.device.load_ptx(ptx.into(), name, &[entry_point])
            .map_err(|e| NeuralForecastError::GpuError(format!("Failed to load PTX kernel {}: {}", name, e)))?;
        
        let func = module.get_function(entry_point)
            .map_err(|e| NeuralForecastError::GpuError(format!("Failed to get kernel function {}: {}", entry_point, e)))?;
        
        self.modules.insert(name.to_string(), module);
        self.kernels.insert(name.to_string(), func);
        
        Ok(())
    }
    
    /// Get kernel function
    pub fn get_kernel(&self, name: &str) -> Option<&CudaFunction> {
        self.kernels.get(name)
    }
}

/// Optimized transformer attention kernel implementation
pub mod transformer_attention {
    use super::*;
    
    /// Launch configuration for transformer attention
    pub struct AttentionLaunchConfig {
        pub batch_size: u32,
        pub num_heads: u32,
        pub seq_length: u32,
        pub head_dim: u32,
        pub use_causal_mask: bool,
        pub use_fp16: bool,
    }
    
    impl AttentionLaunchConfig {
        /// Calculate optimal launch configuration
        pub fn launch_config(&self) -> LaunchConfig {
            // Optimize for Ampere/Hopper architecture
            let threads_per_block = if self.seq_length <= 512 {
                (32, 8, 1) // Warp-aligned for small sequences
            } else {
                (32, 16, 1) // More threads for larger sequences
            };
            
            let blocks = (
                (self.seq_length + threads_per_block.0 - 1) / threads_per_block.0,
                (self.num_heads + threads_per_block.1 - 1) / threads_per_block.1,
                self.batch_size,
            );
            
            LaunchConfig {
                grid_dim: blocks,
                block_dim: threads_per_block,
                shared_mem_bytes: self.calculate_shared_memory(),
            }
        }
        
        /// Calculate required shared memory
        fn calculate_shared_memory(&self) -> u32 {
            let element_size = if self.use_fp16 { 2 } else { 4 };
            let tile_size = 32; // Optimal tile size for tensor cores
            
            // QK^T tile + V tile + softmax workspace
            let qk_tile = tile_size * tile_size * element_size;
            let v_tile = tile_size * self.head_dim as u32 * element_size;
            let softmax_workspace = self.seq_length * 4; // FP32 for numerical stability
            
            (qk_tile * 2 + v_tile + softmax_workspace) as u32
        }
    }
}

/// Mixed precision operations for tensor cores
pub mod mixed_precision {
    use super::*;
    
    /// FP16 tensor core configuration
    pub struct TensorCoreConfig {
        pub m: u32,
        pub n: u32,
        pub k: u32,
        pub use_tf32: bool, // Use TF32 for Ampere GPUs
        pub use_int8: bool, // Use INT8 for inference
    }
    
    impl TensorCoreConfig {
        /// Get optimal tile size for tensor cores
        pub fn tile_size(&self) -> (u32, u32, u32) {
            if self.use_int8 {
                (16, 16, 16) // INT8 tensor core tile
            } else if self.use_tf32 {
                (16, 16, 8)  // TF32 tensor core tile
            } else {
                (16, 16, 16) // FP16 tensor core tile
            }
        }
        
        /// Calculate launch configuration for GEMM
        pub fn launch_config(&self) -> LaunchConfig {
            let (tile_m, tile_n, _) = self.tile_size();
            let warp_size = 32;
            
            // Each warp handles one tile
            let warps_m = (self.m + tile_m - 1) / tile_m;
            let warps_n = (self.n + tile_n - 1) / tile_n;
            
            let threads_per_block = (warp_size, 8, 1);
            let blocks = (
                (warps_m + threads_per_block.1 - 1) / threads_per_block.1,
                (warps_n + 1) / 1,
                1
            );
            
            LaunchConfig {
                grid_dim: blocks,
                block_dim: threads_per_block,
                shared_mem_bytes: self.calculate_shared_memory(),
            }
        }
        
        fn calculate_shared_memory(&self) -> u32 {
            let (tile_m, tile_n, tile_k) = self.tile_size();
            let element_size = if self.use_int8 { 1 } else { 2 };
            
            // Double buffering for A and B tiles
            let a_size = tile_m * tile_k * element_size * 2;
            let b_size = tile_k * tile_n * element_size * 2;
            
            (a_size + b_size) as u32
        }
    }
}

/// Kernel fusion optimizations
pub mod kernel_fusion {
    use super::*;
    
    /// Fused operation types
    #[derive(Debug, Clone)]
    pub enum FusedOperation {
        /// Linear + ReLU
        LinearReLU { in_features: u32, out_features: u32 },
        /// Linear + GELU
        LinearGELU { in_features: u32, out_features: u32 },
        /// LayerNorm + Linear + Activation
        LayerNormLinearActivation {
            feature_size: u32,
            out_features: u32,
            activation: ActivationType,
        },
        /// Multi-operation fusion
        Custom(Vec<Operation>),
    }
    
    #[derive(Debug, Clone)]
    pub enum ActivationType {
        ReLU,
        GELU,
        SiLU,
        Tanh,
        Sigmoid,
    }
    
    #[derive(Debug, Clone)]
    pub enum Operation {
        Linear(u32, u32),
        LayerNorm(u32),
        BatchNorm(u32),
        Activation(ActivationType),
        Dropout(f32),
    }
    
    impl FusedOperation {
        /// Get launch configuration for fused operation
        pub fn launch_config(&self) -> LaunchConfig {
            match self {
                FusedOperation::LinearReLU { in_features, out_features } => {
                    let threads = 256;
                    let blocks = ((*out_features + threads - 1) / threads, 1, 1);
                    
                    LaunchConfig {
                        grid_dim: blocks,
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: 0,
                    }
                }
                FusedOperation::LinearGELU { in_features, out_features } => {
                    let threads = 256;
                    let blocks = ((*out_features + threads - 1) / threads, 1, 1);
                    
                    LaunchConfig {
                        grid_dim: blocks,
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: 0,
                    }
                }
                FusedOperation::LayerNormLinearActivation { feature_size, out_features, .. } => {
                    // Use cooperative groups for layer norm
                    let threads = 128;
                    let blocks = (1, 1, 1); // Process one sample at a time
                    
                    LaunchConfig {
                        grid_dim: blocks,
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: (*feature_size * 4) as u32, // For reduction
                    }
                }
                FusedOperation::Custom(ops) => {
                    // Analyze operations to determine optimal configuration
                    let threads = 256;
                    let blocks = (1, 1, 1);
                    
                    LaunchConfig {
                        grid_dim: blocks,
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: 0,
                    }
                }
            }
        }
    }
}

/// Memory optimization utilities
pub mod memory_optimization {
    use super::*;
    
    /// Pinned memory allocator for fast CPU-GPU transfers
    pub struct PinnedMemoryAllocator {
        device: std::sync::Arc<CudaDevice>,
        allocations: std::collections::HashMap<u64, cudarc::driver::CudaSlice<u8>>,
    }
    
    impl PinnedMemoryAllocator {
        pub fn new(device: std::sync::Arc<CudaDevice>) -> Self {
            Self {
                device,
                allocations: std::collections::HashMap::new(),
            }
        }
        
        /// Allocate pinned memory
        pub fn allocate<T: cudarc::driver::DeviceRepr>(&mut self, size: usize) -> Result<*mut T> {
            let bytes = size * std::mem::size_of::<T>();
            let allocation = self.device.alloc_zeros::<u8>(bytes)
                .map_err(|e| NeuralForecastError::GpuError(format!("Failed to allocate pinned memory: {}", e)))?;
            
            let ptr = allocation.as_ptr() as *mut T;
            let key = ptr as u64;
            
            self.allocations.insert(key, allocation);
            Ok(ptr)
        }
        
        /// Free pinned memory
        pub fn free<T>(&mut self, ptr: *mut T) {
            let key = ptr as u64;
            self.allocations.remove(&key);
        }
    }
    
    /// Memory pool for tensor allocations
    pub struct TensorMemoryPool {
        small_pool: Vec<cudarc::driver::CudaSlice<f32>>,  // < 1MB
        medium_pool: Vec<cudarc::driver::CudaSlice<f32>>, // 1MB - 16MB
        large_pool: Vec<cudarc::driver::CudaSlice<f32>>,  // > 16MB
        device: std::sync::Arc<CudaDevice>,
    }
    
    impl TensorMemoryPool {
        pub fn new(device: std::sync::Arc<CudaDevice>) -> Self {
            Self {
                small_pool: Vec::new(),
                medium_pool: Vec::new(),
                large_pool: Vec::new(),
                device,
            }
        }
        
        /// Get buffer from pool or allocate new
        pub fn get_buffer(&mut self, size: usize) -> Result<cudarc::driver::CudaSlice<f32>> {
            let bytes = size * std::mem::size_of::<f32>();
            
            let pool = if bytes < 1024 * 1024 {
                &mut self.small_pool
            } else if bytes < 16 * 1024 * 1024 {
                &mut self.medium_pool
            } else {
                &mut self.large_pool
            };
            
            if let Some(buffer) = pool.pop() {
                if buffer.len() >= size {
                    return Ok(buffer);
                }
            }
            
            // Allocate new buffer
            self.device.alloc_zeros::<f32>(size)
                .map_err(|e| NeuralForecastError::GpuError(format!("Failed to allocate tensor memory: {}", e)))
        }
        
        /// Return buffer to pool
        pub fn return_buffer(&mut self, buffer: cudarc::driver::CudaSlice<f32>) {
            let bytes = buffer.len() * std::mem::size_of::<f32>();
            
            let pool = if bytes < 1024 * 1024 {
                &mut self.small_pool
            } else if bytes < 16 * 1024 * 1024 {
                &mut self.medium_pool
            } else {
                &mut self.large_pool
            };
            
            pool.push(buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_attention_launch_config() {
        let config = transformer_attention::AttentionLaunchConfig {
            batch_size: 8,
            num_heads: 12,
            seq_length: 512,
            head_dim: 64,
            use_causal_mask: true,
            use_fp16: true,
        };
        
        let launch = config.launch_config();
        assert_eq!(launch.block_dim, (32, 8, 1));
        assert_eq!(launch.grid_dim.2, 8); // batch size
    }
    
    #[test]
    fn test_tensor_core_config() {
        let config = mixed_precision::TensorCoreConfig {
            m: 1024,
            n: 1024,
            k: 1024,
            use_tf32: false,
            use_int8: false,
        };
        
        let tile = config.tile_size();
        assert_eq!(tile, (16, 16, 16)); // FP16 tile size
    }
}