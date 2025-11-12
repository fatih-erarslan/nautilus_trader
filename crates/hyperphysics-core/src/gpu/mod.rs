//! GPU acceleration for HyperPhysics simulations
//!
//! Provides wgpu-based GPU kernels for:
//! - pBit dynamics (Gillespie, Metropolis)
//! - Φ calculation (partition function)
//! - Coupling network construction
//!
//! # Performance Target
//! - <50μs message passing latency
//! - >100x speedup for large simulations (N>1000 pBits)
//! - Adaptive CPU/GPU scheduling based on problem size

pub mod device;
pub mod kernels;
pub mod buffers;

use wgpu;
use std::sync::Arc;

/// GPU computation context managing device, queue, and adapter
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter: wgpu::Adapter,
    pub features: wgpu::Features,
    pub limits: wgpu::Limits,
}

impl GpuContext {
    /// Initialize GPU context with optimal settings for HyperPhysics
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or("No GPU adapter found")?;

        let features = adapter.features();
        let limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("HyperPhysics GPU Device"),
                    required_features: features & wgpu::Features::TIMESTAMP_QUERY,
                    required_limits: limits.clone(),
                },
                None,
            )
            .await?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter,
            features,
            limits,
        })
    }

    /// Get GPU info for logging and diagnostics
    pub fn info(&self) -> GpuInfo {
        let adapter_info = self.adapter.get_info();
        GpuInfo {
            name: adapter_info.name,
            vendor: adapter_info.vendor,
            device_type: format!("{:?}", adapter_info.device_type),
            backend: format!("{:?}", adapter_info.backend),
            features: format!("{:?}", self.features),
            max_buffer_size: self.limits.max_buffer_size,
            max_compute_workgroup_size_x: self.limits.max_compute_workgroup_size_x,
            max_compute_invocations_per_workgroup: self.limits.max_compute_invocations_per_workgroup,
        }
    }
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub vendor: u32,
    pub device_type: String,
    pub backend: String,
    pub features: String,
    pub max_buffer_size: u64,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_invocations_per_workgroup: u32,
}

/// Determine if GPU acceleration should be used based on problem size
pub fn should_use_gpu(num_pbits: usize, num_timesteps: usize) -> bool {
    // GPU overhead is ~10-20μs for kernel launch
    // Break-even point is around 1000 pBits or 10000 timesteps
    const GPU_THRESHOLD_PBITS: usize = 1000;
    const GPU_THRESHOLD_TIMESTEPS: usize = 10000;

    num_pbits >= GPU_THRESHOLD_PBITS || num_timesteps >= GPU_THRESHOLD_TIMESTEPS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_threshold() {
        assert!(!should_use_gpu(100, 100));
        assert!(should_use_gpu(2000, 100));
        assert!(should_use_gpu(100, 20000));
        assert!(should_use_gpu(2000, 20000));
    }
}
