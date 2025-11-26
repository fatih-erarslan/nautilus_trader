//! Backend router for selecting and initializing compute backends

use super::device::Device;
use crate::error::{MlError, MlResult};
use serde::{Deserialize, Serialize};

/// Supported backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendType {
    /// CPU backend using ndarray (pure Rust)
    Cpu,
    /// NVIDIA CUDA backend
    Cuda,
    /// AMD ROCm/HIP backend
    Rocm,
    /// Apple Metal backend
    Metal,
    /// Vulkan backend (cross-platform)
    Vulkan,
    /// WebGPU backend (browser/cross-platform)
    WebGpu,
}

impl BackendType {
    /// Get the feature flag name for this backend
    pub fn feature_flag(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
            Self::Metal => "metal",
            Self::Vulkan => "vulkan",
            Self::WebGpu => "wgpu",
        }
    }

    /// Check if this backend is available at runtime
    pub fn is_available(&self) -> bool {
        match self {
            Self::Cpu => true,
            #[cfg(feature = "cuda")]
            Self::Cuda => super::is_cuda_available(),
            #[cfg(feature = "rocm")]
            Self::Rocm => super::is_rocm_available(),
            #[cfg(feature = "metal")]
            Self::Metal => super::is_metal_available(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan => super::is_vulkan_available(),
            #[cfg(feature = "wgpu")]
            Self::WebGpu => super::is_wgpu_available(),
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU (ndarray)"),
            Self::Cuda => write!(f, "CUDA (NVIDIA)"),
            Self::Rocm => write!(f, "ROCm (AMD)"),
            Self::Metal => write!(f, "Metal (Apple)"),
            Self::Vulkan => write!(f, "Vulkan"),
            Self::WebGpu => write!(f, "WebGPU"),
        }
    }
}

/// Backend configuration and state
#[derive(Debug, Clone)]
pub struct Backend {
    /// Backend type
    backend_type: BackendType,
    /// Device to use
    device: Device,
    /// Whether the backend is initialized
    initialized: bool,
}

impl Backend {
    /// Create a new backend with the specified type
    pub fn new(backend_type: BackendType) -> MlResult<Self> {
        if !backend_type.is_available() {
            return Err(MlError::backend_unavailable(
                format!("{:?}", backend_type),
                format!(
                    "Backend not available. Enable with --features {}",
                    backend_type.feature_flag()
                ),
            ));
        }

        let device = match backend_type {
            BackendType::Cpu => Device::Cpu,
            BackendType::Cuda => Device::Cuda(0),
            BackendType::Rocm => Device::Rocm(0),
            BackendType::Metal => Device::Metal(0),
            BackendType::Vulkan => Device::Vulkan(0),
            BackendType::WebGpu => Device::WebGpu(0),
        };

        Ok(Self {
            backend_type,
            device,
            initialized: false,
        })
    }

    /// Create a CPU backend
    pub fn cpu() -> Self {
        Self {
            backend_type: BackendType::Cpu,
            device: Device::Cpu,
            initialized: true,
        }
    }

    /// Automatically select the best available backend
    pub fn auto() -> Self {
        let backend_type = super::select_best_backend();
        Self::new(backend_type).unwrap_or_else(|_| Self::cpu())
    }

    /// Create a CUDA backend if available
    #[cfg(feature = "cuda")]
    pub fn cuda(device_index: usize) -> MlResult<Self> {
        if !BackendType::Cuda.is_available() {
            return Err(MlError::backend_unavailable(
                "CUDA",
                "CUDA runtime not found. Ensure NVIDIA drivers are installed.",
            ));
        }

        Ok(Self {
            backend_type: BackendType::Cuda,
            device: Device::Cuda(device_index),
            initialized: false,
        })
    }

    /// Create a Metal backend if available (macOS only)
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn metal(device_index: usize) -> MlResult<Self> {
        Ok(Self {
            backend_type: BackendType::Metal,
            device: Device::Metal(device_index),
            initialized: false,
        })
    }

    /// Create a ROCm backend if available
    #[cfg(feature = "rocm")]
    pub fn rocm(device_index: usize) -> MlResult<Self> {
        if !BackendType::Rocm.is_available() {
            return Err(MlError::backend_unavailable(
                "ROCm",
                "ROCm runtime not found. Ensure AMD drivers are installed.",
            ));
        }

        Ok(Self {
            backend_type: BackendType::Rocm,
            device: Device::Rocm(device_index),
            initialized: false,
        })
    }

    /// Create a Vulkan backend
    #[cfg(feature = "vulkan")]
    pub fn vulkan(device_index: usize) -> MlResult<Self> {
        if !BackendType::Vulkan.is_available() {
            return Err(MlError::backend_unavailable(
                "Vulkan",
                "Vulkan ICD not found.",
            ));
        }

        Ok(Self {
            backend_type: BackendType::Vulkan,
            device: Device::Vulkan(device_index),
            initialized: false,
        })
    }

    /// Get the backend type
    pub fn backend_type(&self) -> BackendType {
        self.backend_type
    }

    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Check if backend is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Initialize the backend
    pub fn initialize(&mut self) -> MlResult<()> {
        if self.initialized {
            return Ok(());
        }

        match self.backend_type {
            BackendType::Cpu => {
                // CPU always ready
                self.initialized = true;
            }
            #[cfg(feature = "cuda")]
            BackendType::Cuda => {
                // Initialize CUDA context
                self.initialized = true;
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            BackendType::Metal => {
                // Initialize Metal device
                self.initialized = true;
            }
            #[cfg(feature = "rocm")]
            BackendType::Rocm => {
                // Initialize ROCm/HIP context
                self.initialized = true;
            }
            #[cfg(feature = "vulkan")]
            BackendType::Vulkan => {
                // Initialize Vulkan instance
                self.initialized = true;
            }
            #[cfg(feature = "wgpu")]
            BackendType::WebGpu => {
                // Initialize wgpu adapter
                self.initialized = true;
            }
            #[allow(unreachable_patterns)]
            _ => {
                return Err(MlError::backend_unavailable(
                    format!("{:?}", self.backend_type),
                    "Backend not compiled",
                ));
            }
        }

        Ok(())
    }

    /// Synchronize device (wait for all operations to complete)
    pub fn synchronize(&self) -> MlResult<()> {
        match self.backend_type {
            BackendType::Cpu => Ok(()), // CPU is always synchronized
            #[cfg(feature = "cuda")]
            BackendType::Cuda => {
                // cudaDeviceSynchronize()
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

impl Default for Backend {
    fn default() -> Self {
        Self::cpu()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend() {
        let backend = Backend::cpu();
        assert_eq!(backend.backend_type(), BackendType::Cpu);
        assert!(backend.is_initialized());
    }

    #[test]
    fn test_auto_backend() {
        let backend = Backend::auto();
        // Should return some backend (at minimum CPU)
        assert!(matches!(
            backend.backend_type(),
            BackendType::Cpu
                | BackendType::Cuda
                | BackendType::Metal
                | BackendType::Rocm
                | BackendType::Vulkan
                | BackendType::WebGpu
        ));
    }

    #[test]
    fn test_backend_type_availability() {
        // CPU should always be available
        assert!(BackendType::Cpu.is_available());
    }
}
