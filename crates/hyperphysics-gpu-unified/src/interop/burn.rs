//! Burn ML framework integration
//!
//! Provides a bridge between HyperPhysics GPU orchestration and Burn's
//! autodiff/neural network capabilities.

#[cfg(feature = "burn")]
use burn::tensor::backend::Backend;
#[cfg(feature = "burn")]
use burn_wgpu::{Wgpu, WgpuDevice};

use std::sync::Arc;
use wgpu::{Device, Queue};

use crate::{GpuError, GpuResult};

/// Configuration for Burn backend
#[derive(Debug, Clone)]
pub struct BurnConfig {
    /// Use the primary GPU (high performance)
    pub use_primary_gpu: bool,
    /// Enable autodiff for gradient computation
    pub enable_autodiff: bool,
    /// Memory limit for Burn operations (bytes)
    pub memory_limit_bytes: Option<u64>,
}

impl Default for BurnConfig {
    fn default() -> Self {
        Self {
            use_primary_gpu: true,
            enable_autodiff: true,
            memory_limit_bytes: None,
        }
    }
}

/// Bridge between HyperPhysics GPU orchestrator and Burn framework
pub struct BurnBridge {
    /// Configuration
    config: BurnConfig,
    /// Shared wgpu device (from orchestrator)
    #[allow(dead_code)]
    device: Arc<Device>,
    /// Shared wgpu queue
    #[allow(dead_code)]
    queue: Arc<Queue>,
    /// Burn device handle
    #[cfg(feature = "burn")]
    burn_device: WgpuDevice,
}

impl BurnBridge {
    /// Create a new Burn bridge using orchestrator's GPU resources
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        config: BurnConfig,
    ) -> GpuResult<Self> {
        #[cfg(feature = "burn")]
        {
            // Create Burn device using default wgpu device
            // Note: Burn manages its own wgpu instance, but we can coordinate
            let burn_device = if config.use_primary_gpu {
                WgpuDevice::DefaultDevice
            } else {
                WgpuDevice::DefaultDevice // Burn doesn't support multi-GPU yet
            };

            Ok(Self {
                config,
                device,
                queue,
                burn_device,
            })
        }

        #[cfg(not(feature = "burn"))]
        {
            let _ = (device, queue, config);
            Err(GpuError::InvalidConfig(
                "Burn feature not enabled. Recompile with --features burn".to_string(),
            ))
        }
    }

    /// Get the Burn device for tensor operations
    #[cfg(feature = "burn")]
    pub fn device(&self) -> &WgpuDevice {
        &self.burn_device
    }

    /// Check if autodiff is enabled
    pub fn autodiff_enabled(&self) -> bool {
        self.config.enable_autodiff
    }

    /// Get configuration
    pub fn config(&self) -> &BurnConfig {
        &self.config
    }
}

/// Trait for types that can be converted to Burn tensors
#[cfg(feature = "burn")]
pub trait ToBurnTensor<B: Backend> {
    /// Output tensor type
    type Output;

    /// Convert to Burn tensor
    fn to_burn_tensor(&self, device: &B::Device) -> Self::Output;
}

/// Trait for types that can be converted from Burn tensors
#[cfg(feature = "burn")]
pub trait FromBurnTensor<B: Backend> {
    /// Input tensor type
    type Input;

    /// Convert from Burn tensor
    fn from_burn_tensor(tensor: Self::Input) -> Self;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_burn_config_default() {
        let config = BurnConfig::default();
        assert!(config.use_primary_gpu);
        assert!(config.enable_autodiff);
        assert!(config.memory_limit_bytes.is_none());
    }
}
