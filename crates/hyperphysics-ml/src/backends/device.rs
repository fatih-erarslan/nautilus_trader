//! Device abstraction for compute resources

use crate::error::{MlError, MlResult};
use serde::{Deserialize, Serialize};

/// Compute device identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Device {
    /// CPU device
    Cpu,
    /// CUDA GPU device with index
    Cuda(usize),
    /// Metal GPU device (macOS)
    Metal(usize),
    /// ROCm/HIP GPU device with index
    Rocm(usize),
    /// Vulkan GPU device with index
    Vulkan(usize),
    /// WebGPU device
    WebGpu(usize),
}

impl Device {
    /// Create a CPU device
    pub fn cpu() -> Self {
        Self::Cpu
    }

    /// Create a CUDA device with the given index
    pub fn cuda(index: usize) -> Self {
        Self::Cuda(index)
    }

    /// Create a Metal device with the given index
    pub fn metal(index: usize) -> Self {
        Self::Metal(index)
    }

    /// Create default GPU device for current platform
    pub fn gpu() -> Self {
        #[cfg(feature = "cuda")]
        return Self::Cuda(0);

        #[cfg(all(target_os = "macos", feature = "metal", not(feature = "cuda")))]
        return Self::Metal(0);

        #[cfg(all(feature = "rocm", not(feature = "cuda"), not(all(target_os = "macos", feature = "metal"))))]
        return Self::Rocm(0);

        #[cfg(all(feature = "vulkan", not(feature = "cuda"), not(feature = "rocm"), not(all(target_os = "macos", feature = "metal"))))]
        return Self::Vulkan(0);

        #[cfg(not(any(feature = "cuda", feature = "rocm", feature = "vulkan", all(target_os = "macos", feature = "metal"))))]
        Self::Cpu
    }

    /// Check if this is a GPU device
    pub fn is_gpu(&self) -> bool {
        !matches!(self, Self::Cpu)
    }

    /// Check if this is a CPU device
    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu)
    }

    /// Get device index (0 for CPU)
    pub fn index(&self) -> usize {
        match self {
            Self::Cpu => 0,
            Self::Cuda(i) | Self::Metal(i) | Self::Rocm(i) | Self::Vulkan(i) | Self::WebGpu(i) => *i,
        }
    }

    /// Get device name for display
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Cuda(_) => "CUDA",
            Self::Metal(_) => "Metal",
            Self::Rocm(_) => "ROCm",
            Self::Vulkan(_) => "Vulkan",
            Self::WebGpu(_) => "WebGPU",
        }
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::Cpu
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Cuda(i) => write!(f, "CUDA:{}", i),
            Self::Metal(i) => write!(f, "Metal:{}", i),
            Self::Rocm(i) => write!(f, "ROCm:{}", i),
            Self::Vulkan(i) => write!(f, "Vulkan:{}", i),
            Self::WebGpu(i) => write!(f, "WebGPU:{}", i),
        }
    }
}

/// Device information and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Device identifier
    pub device: Device,
    /// Device name/model
    pub name: String,
    /// Total memory in bytes
    pub total_memory: u64,
    /// Available memory in bytes
    pub available_memory: u64,
    /// Compute capability (for CUDA) or version
    pub compute_capability: String,
    /// Number of compute units (CUDA cores, Metal compute units, etc.)
    pub compute_units: u32,
    /// Maximum threads per block/workgroup
    pub max_threads_per_block: u32,
    /// Whether tensor cores are available
    pub has_tensor_cores: bool,
    /// Whether FP16 is supported
    pub supports_fp16: bool,
    /// Whether INT8 is supported
    pub supports_int8: bool,
}

impl DeviceInfo {
    /// Query device information
    pub fn query(device: &Device) -> MlResult<Self> {
        match device {
            Device::Cpu => Self::query_cpu(),
            #[cfg(feature = "cuda")]
            Device::Cuda(idx) => Self::query_cuda(*idx),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Device::Metal(idx) => Self::query_metal(*idx),
            _ => Err(MlError::backend_unavailable(
                device.name(),
                "Backend not compiled or not available",
            )),
        }
    }

    fn query_cpu() -> MlResult<Self> {
        Ok(Self {
            device: Device::Cpu,
            name: "CPU".to_string(),
            total_memory: sys_info_total_memory(),
            available_memory: sys_info_available_memory(),
            compute_capability: std::env::consts::ARCH.to_string(),
            compute_units: num_cpus(),
            max_threads_per_block: 1,
            has_tensor_cores: false,
            supports_fp16: cfg!(target_feature = "f16c"),
            supports_int8: true,
        })
    }

    #[cfg(feature = "cuda")]
    fn query_cuda(idx: usize) -> MlResult<Self> {
        // CUDA device query would go here
        // For now, return placeholder
        Ok(Self {
            device: Device::Cuda(idx),
            name: format!("CUDA Device {}", idx),
            total_memory: 0,
            available_memory: 0,
            compute_capability: "unknown".to_string(),
            compute_units: 0,
            max_threads_per_block: 1024,
            has_tensor_cores: false,
            supports_fp16: true,
            supports_int8: true,
        })
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn query_metal(idx: usize) -> MlResult<Self> {
        Ok(Self {
            device: Device::Metal(idx),
            name: format!("Metal Device {}", idx),
            total_memory: sys_info_total_memory(),
            available_memory: sys_info_available_memory(),
            compute_capability: "Metal 3".to_string(),
            compute_units: num_cpus() as u32, // Approximate
            max_threads_per_block: 1024,
            has_tensor_cores: true, // Apple Neural Engine
            supports_fp16: true,
            supports_int8: true,
        })
    }
}

// Helper functions for system info

fn sys_info_total_memory() -> u64 {
    // Platform-specific memory query
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("MemTotal:"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|v| v.parse::<u64>().ok())
                    .map(|kb| kb * 1024)
            })
            .unwrap_or(0)
    }

    #[cfg(target_os = "macos")]
    {
        // macOS: use sysctl
        8 * 1024 * 1024 * 1024 // Default 8GB placeholder
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        0
    }
}

fn sys_info_available_memory() -> u64 {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("MemAvailable:"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|v| v.parse::<u64>().ok())
                    .map(|kb| kb * 1024)
            })
            .unwrap_or(0)
    }

    #[cfg(not(target_os = "linux"))]
    {
        sys_info_total_memory() / 2 // Approximate
    }
}

fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|p| p.get() as u32)
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_creation() {
        let cpu = Device::cpu();
        assert!(cpu.is_cpu());
        assert!(!cpu.is_gpu());
        assert_eq!(cpu.index(), 0);
    }

    #[test]
    fn test_device_display() {
        assert_eq!(format!("{}", Device::Cpu), "CPU");
        assert_eq!(format!("{}", Device::Cuda(0)), "CUDA:0");
        assert_eq!(format!("{}", Device::Metal(1)), "Metal:1");
    }

    #[test]
    fn test_device_info_cpu() {
        let info = DeviceInfo::query(&Device::Cpu).unwrap();
        assert_eq!(info.device, Device::Cpu);
        assert!(info.compute_units > 0);
    }
}
