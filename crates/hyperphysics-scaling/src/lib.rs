//! Auto-Scaling Orchestrator for HyperPhysics
//!
//! Dynamically scales simulation configuration based on system capabilities.
//! Supports 48 nodes → 1 billion nodes with intelligent resource allocation.

pub mod config;
pub mod workload;
pub mod gpu_detect;

use hyperphysics_core::{EngineConfig, Result, EngineError};
use sysinfo::System;

/// Auto-scaling orchestrator with GPU detection
pub struct AutoScaler {
    system: System,
    gpu_info: Option<GPUInfo>,
}

/// GPU capability information
#[derive(Debug, Clone)]
pub struct GPUInfo {
    pub device_name: String,
    pub max_buffer_size: u64,
    pub max_workgroup_size: u32,
    pub available: bool,
}

impl AutoScaler {
    /// Create new auto-scaler with GPU detection
    pub fn new() -> Self {
        let gpu_info = Self::detect_gpu();
        Self {
            system: System::new_all(),
            gpu_info,
        }
    }

    /// Detect GPU capabilities asynchronously
    fn detect_gpu() -> Option<GPUInfo> {
        // Try to initialize GPU backend
        use pollster::FutureExt;
        use hyperphysics_gpu::backend::wgpu::WGPUBackend;
        use hyperphysics_gpu::backend::GPUBackend;

        match WGPUBackend::new().block_on() {
            Ok(backend) => {
                let caps = backend.capabilities();
                Some(GPUInfo {
                    device_name: caps.device_name.clone(),
                    max_buffer_size: caps.max_buffer_size,
                    max_workgroup_size: caps.max_workgroup_size,
                    available: caps.supports_compute,
                })
            }
            Err(_) => None,
        }
    }

    /// Get current system capabilities with GPU info
    pub fn system_capabilities(&mut self) -> SystemCapabilities {
        self.system.refresh_all();

        SystemCapabilities {
            cpu_count: num_cpus::get(),
            total_memory_gb: self.system.total_memory() as f64 / 1e9,
            available_memory_gb: self.system.available_memory() as f64 / 1e9,
            has_gpu: self.gpu_info.as_ref().map_or(false, |info| info.available),
            gpu_info: self.gpu_info.clone(),
        }
    }

    /// Recommend optimal configuration for target node count
    ///
    /// # Arguments
    /// * `target_nodes` - Desired number of pBits in lattice
    ///
    /// # Returns
    /// Optimal `EngineConfig` matching system capabilities
    pub fn recommend_config(&mut self, target_nodes: usize) -> Result<EngineConfig> {
        let caps = self.system_capabilities();

        // ROI 48: 48 nodes (default)
        if target_nodes <= 48 {
            return Ok(EngineConfig::roi_48(1.0));
        }

        // ROI 128×128: 16,384 nodes
        if target_nodes <= 16_384 {
            // Check memory requirements: ~500 MB for 16K nodes
            let required_memory_gb = 0.5;
            if caps.available_memory_gb < required_memory_gb {
                return Err(EngineError::Configuration {
                    message: format!(
                        "Insufficient memory: {} GB available, {} GB required",
                        caps.available_memory_gb, required_memory_gb
                    ),
                });
            }
            return Ok(EngineConfig::roi_128x128(300.0));
        }

        // ROI 1024×1024: 1,048,576 nodes
        if target_nodes <= 1_048_576 {
            let required_memory_gb = 8.0;
            if caps.available_memory_gb < required_memory_gb {
                return Err(EngineError::Configuration {
                    message: format!(
                        "Insufficient memory: {} GB available, {} GB required",
                        caps.available_memory_gb, required_memory_gb
                    ),
                });
            }
            if !caps.has_gpu {
                return Err(EngineError::Configuration {
                    message: "GPU required for 1M+ nodes".to_string(),
                });
            }
            return Ok(EngineConfig::roi_1024x1024(300.0));
        }

        // ROI 32K×32K: 1 billion nodes
        if target_nodes <= 1_073_741_824 {
            let required_memory_gb = 128.0;
            if caps.available_memory_gb < required_memory_gb {
                return Err(EngineError::Configuration {
                    message: format!(
                        "Insufficient memory: {} GB available, {} GB required",
                        caps.available_memory_gb, required_memory_gb
                    ),
                });
            }
            if !caps.has_gpu {
                return Err(EngineError::Configuration {
                    message: "High-end GPU required for 1B nodes".to_string(),
                });
            }
            return Ok(EngineConfig::roi_32kx32k(300.0));
        }

        Err(EngineError::Configuration {
            message: format!("Node count {} exceeds maximum supported (1B)", target_nodes),
        })
    }
}

impl Default for AutoScaler {
    fn default() -> Self {
        Self::new()
    }
}

/// System capability snapshot
#[derive(Debug, Clone)]
pub struct SystemCapabilities {
    /// Number of CPU cores
    pub cpu_count: usize,
    /// Total system memory in GB
    pub total_memory_gb: f64,
    /// Available system memory in GB
    pub available_memory_gb: f64,
    /// Whether GPU is available
    pub has_gpu: bool,
    /// Detailed GPU information
    pub gpu_info: Option<GPUInfo>,
}

impl SystemCapabilities {
    /// Calculate recommended backend based on workload
    pub fn recommend_backend(&self, node_count: usize) -> ComputeBackend {
        // Small workloads: CPU is fine
        if node_count < 1000 {
            return ComputeBackend::CPU;
        }

        // Medium workloads: GPU if available, else CPU with SIMD
        if node_count < 100_000 {
            if self.has_gpu {
                return ComputeBackend::GPU;
            }
            return ComputeBackend::CPUSIMD;
        }

        // Large workloads: Require GPU
        if node_count < 10_000_000 {
            if self.has_gpu {
                return ComputeBackend::GPU;
            }
            // Fallback to CPU but warn about performance
            return ComputeBackend::CPUSIMD;
        }

        // Massive workloads: Require high-end GPU
        if self.has_gpu {
            ComputeBackend::GPU
        } else {
            // This will likely fail, but let the engine handle it
            ComputeBackend::CPUSIMD
        }
    }

    /// Estimate memory usage for given configuration
    pub fn estimate_memory_usage(&self, node_count: usize, use_gpu: bool) -> f64 {
        // Base memory per node: 32 bytes for state
        let state_memory = node_count as f64 * 32.0 / 1e9; // GB

        // Coupling memory: assume average 6 neighbors per node
        let coupling_memory = node_count as f64 * 6.0 * 12.0 / 1e9; // GB

        // GPU requires double buffering
        let buffer_multiplier = if use_gpu { 2.0 } else { 1.0 };

        // Overhead: ~20% for auxiliary structures
        let total = (state_memory + coupling_memory) * buffer_multiplier * 1.2;

        total
    }

    /// Check if configuration is feasible
    pub fn can_handle(&self, node_count: usize, use_gpu: bool) -> Result<()> {
        let required_memory = self.estimate_memory_usage(node_count, use_gpu);

        if required_memory > self.available_memory_gb {
            return Err(EngineError::Configuration {
                message: format!(
                    "Insufficient memory: {:.2} GB required, {:.2} GB available",
                    required_memory, self.available_memory_gb
                ),
            });
        }

        if use_gpu && !self.has_gpu {
            return Err(EngineError::Configuration {
                message: "GPU required but not available".to_string(),
            });
        }

        // Check GPU buffer size limits
        if use_gpu {
            if let Some(gpu_info) = &self.gpu_info {
                let required_buffer = (node_count * 32) as u64;
                if required_buffer > gpu_info.max_buffer_size {
                    return Err(EngineError::Configuration {
                        message: format!(
                            "GPU buffer too small: {} bytes required, {} bytes max",
                            required_buffer, gpu_info.max_buffer_size
                        ),
                    });
                }
            }
        }

        Ok(())
    }
}

/// Compute backend recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackend {
    /// CPU-only (single-threaded or basic threading)
    CPU,
    /// CPU with SIMD vectorization
    CPUSIMD,
    /// GPU compute (WGPU/CUDA/Metal)
    GPU,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_capabilities() {
        let mut scaler = AutoScaler::new();
        let caps = scaler.system_capabilities();

        assert!(caps.cpu_count > 0);
        assert!(caps.total_memory_gb > 0.0);
        assert!(caps.available_memory_gb > 0.0);
        assert!(caps.available_memory_gb <= caps.total_memory_gb);
    }

    #[test]
    fn test_recommend_config_small() {
        let mut scaler = AutoScaler::new();
        let config = scaler.recommend_config(48);
        assert!(config.is_ok());
    }

    #[test]
    fn test_recommend_config_too_large() {
        let mut scaler = AutoScaler::new();
        let config = scaler.recommend_config(2_000_000_000);
        assert!(config.is_err());
    }
}
