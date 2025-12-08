//! Device management for hardware-accelerated fusion operations
//!
//! This module provides automatic device detection and management for various
//! hardware acceleration backends including CUDA, Metal, and ROCm.

use crate::error::{FusionError, Result};
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Device manager for hardware acceleration
#[derive(Debug, Clone)]
pub struct DeviceManager {
    /// Primary device for computations
    pub primary_device: Device,
    /// Available devices
    pub available_devices: Vec<DeviceInfo>,
    /// Device capabilities
    pub capabilities: DeviceCapabilities,
}

/// Information about a compute device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Device type
    pub device_type: DeviceType,
    /// Device index
    pub index: usize,
    /// Device name
    pub name: String,
    /// Total memory in bytes
    pub memory_total: Option<u64>,
    /// Available memory in bytes
    pub memory_available: Option<u64>,
    /// Compute capability (for CUDA)
    pub compute_capability: Option<(u32, u32)>,
    /// Number of cores
    pub cores: Option<u32>,
    /// Clock speed in MHz
    pub clock_speed_mhz: Option<u32>,
    /// Whether the device is currently available
    pub is_available: bool,
}

/// Device type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceType {
    /// CPU device
    Cpu,
    /// NVIDIA CUDA device
    Cuda,
    /// Apple Metal device
    Metal,
    /// AMD ROCm device
    Rocm,
    /// Intel GPU device
    Intel,
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceType::Cpu => write!(f, "CPU"),
            DeviceType::Cuda => write!(f, "CUDA"),
            DeviceType::Metal => write!(f, "Metal"),
            DeviceType::Rocm => write!(f, "ROCm"),
            DeviceType::Intel => write!(f, "Intel"),
        }
    }
}

/// Device capabilities and features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// Supports half precision (FP16)
    pub supports_fp16: bool,
    /// Supports tensor cores
    pub supports_tensor_cores: bool,
    /// Supports unified memory
    pub supports_unified_memory: bool,
    /// Supports async operations
    pub supports_async: bool,
    /// Maximum tensor size
    pub max_tensor_size: Option<u64>,
    /// Maximum batch size
    pub max_batch_size: Option<usize>,
    /// Supports SIMD operations
    pub supports_simd: bool,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbs: Option<f32>,
    /// Peak performance in GFLOPS
    pub peak_gflops: Option<f32>,
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self {
            supports_fp16: false,
            supports_tensor_cores: false,
            supports_unified_memory: false,
            supports_async: false,
            max_tensor_size: None,
            max_batch_size: None,
            supports_simd: false,
            memory_bandwidth_gbs: None,
            peak_gflops: None,
        }
    }
}

impl DeviceManager {
    /// Create a new device manager with automatic device detection
    pub fn new() -> Result<Self> {
        let available_devices = Self::detect_devices()?;
        let primary_device = Self::select_best_device(&available_devices)?;
        let capabilities = Self::get_device_capabilities(&primary_device)?;

        Ok(Self {
            primary_device,
            available_devices,
            capabilities,
        })
    }

    /// Create a device manager with a specific device
    pub fn with_device(device: Device) -> Result<Self> {
        let available_devices = Self::detect_devices()?;
        let capabilities = Self::get_device_capabilities(&device)?;

        Ok(Self {
            primary_device: device,
            available_devices,
            capabilities,
        })
    }

    /// Detect all available devices
    pub fn detect_devices() -> Result<Vec<DeviceInfo>> {
        let mut devices = Vec::new();

        // Always add CPU
        devices.push(DeviceInfo {
            device_type: DeviceType::Cpu,
            index: 0,
            name: "CPU".to_string(),
            memory_total: Self::get_system_memory(),
            memory_available: Self::get_available_memory(),
            compute_capability: None,
            cores: Self::get_cpu_cores(),
            clock_speed_mhz: None,
            is_available: true,
        });

        // Detect CUDA devices
        #[cfg(feature = "cuda")]
        {
            if let Ok(cuda_devices) = Self::detect_cuda_devices() {
                devices.extend(cuda_devices);
            }
        }

        // Detect Metal devices
        #[cfg(feature = "metal")]
        {
            if let Ok(metal_devices) = Self::detect_metal_devices() {
                devices.extend(metal_devices);
            }
        }

        // Detect ROCm devices
        #[cfg(feature = "rocm")]
        {
            if let Ok(rocm_devices) = Self::detect_rocm_devices() {
                devices.extend(rocm_devices);
            }
        }

        Ok(devices)
    }

    /// Select the best available device
    pub fn select_best_device(devices: &[DeviceInfo]) -> Result<Device> {
        // Priority order: CUDA > Metal > ROCm > CPU
        for device in devices {
            if !device.is_available {
                continue;
            }

            match device.device_type {
                DeviceType::Cuda => {
                    #[cfg(feature = "cuda")]
                    {
                        if let Ok(cuda_device) = Device::new_cuda(device.index) {
                            return Ok(cuda_device);
                        }
                    }
                }
                DeviceType::Metal => {
                    #[cfg(feature = "metal")]
                    {
                        if let Ok(metal_device) = Device::new_metal(device.index) {
                            return Ok(metal_device);
                        }
                    }
                }
                DeviceType::Rocm => {
                    #[cfg(feature = "rocm")]
                    {
                        // ROCm support would be added here
                        // Currently not available in Candle
                    }
                }
                DeviceType::Cpu => {
                    return Ok(Device::Cpu);
                }
                _ => continue,
            }
        }

        // Fallback to CPU
        Ok(Device::Cpu)
    }

    /// Get device capabilities for a specific device
    pub fn get_device_capabilities(device: &Device) -> Result<DeviceCapabilities> {
        let mut capabilities = DeviceCapabilities::default();

        match device {
            Device::Cpu => {
                capabilities.supports_simd = true;
                capabilities.supports_async = true;
                capabilities.max_tensor_size = Some(1_000_000_000); // 1GB
                capabilities.max_batch_size = Some(10000);
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                {
                    capabilities.supports_fp16 = true;
                    capabilities.supports_tensor_cores = true;
                    capabilities.supports_unified_memory = true;
                    capabilities.supports_async = true;
                    capabilities.supports_simd = true;
                    capabilities.max_tensor_size = Some(10_000_000_000); // 10GB
                    capabilities.max_batch_size = Some(100000);
                    capabilities.memory_bandwidth_gbs = Some(900.0); // Estimate for modern GPUs
                    capabilities.peak_gflops = Some(30000.0); // Estimate for modern GPUs
                }
            }
            Device::Metal(_) => {
                #[cfg(feature = "metal")]
                {
                    capabilities.supports_fp16 = true;
                    capabilities.supports_unified_memory = true;
                    capabilities.supports_async = true;
                    capabilities.supports_simd = true;
                    capabilities.max_tensor_size = Some(8_000_000_000); // 8GB
                    capabilities.max_batch_size = Some(50000);
                    capabilities.memory_bandwidth_gbs = Some(400.0); // Estimate for Apple Silicon
                    capabilities.peak_gflops = Some(15000.0); // Estimate for Apple Silicon
                }
            }
        }

        Ok(capabilities)
    }

    /// Get system memory in bytes
    fn get_system_memory() -> Option<u64> {
        // This is a simplified implementation
        // In a real implementation, you would use system APIs
        #[cfg(target_os = "linux")]
        {
            if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
                for line in meminfo.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                return Some(kb * 1024);
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Get available memory in bytes
    fn get_available_memory() -> Option<u64> {
        // This is a simplified implementation
        #[cfg(target_os = "linux")]
        {
            if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
                for line in meminfo.lines() {
                    if line.starts_with("MemAvailable:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                return Some(kb * 1024);
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Get number of CPU cores
    fn get_cpu_cores() -> Option<u32> {
        Some(num_cpus::get() as u32)
    }

    /// Detect CUDA devices
    #[cfg(feature = "cuda")]
    fn detect_cuda_devices() -> Result<Vec<DeviceInfo>> {
        let mut devices = Vec::new();

        // Try to create CUDA devices
        for i in 0..8 {
            // Check up to 8 devices
            if let Ok(_device) = Device::new_cuda(i) {
                devices.push(DeviceInfo {
                    device_type: DeviceType::Cuda,
                    index: i,
                    name: format!("CUDA Device {}", i),
                    memory_total: None, // Would need CUDA API to get this
                    memory_available: None,
                    compute_capability: None,
                    cores: None,
                    clock_speed_mhz: None,
                    is_available: true,
                });
            } else {
                break;
            }
        }

        Ok(devices)
    }

    /// Detect Metal devices
    #[cfg(feature = "metal")]
    fn detect_metal_devices() -> Result<Vec<DeviceInfo>> {
        let mut devices = Vec::new();

        // Try to create Metal devices
        for i in 0..4 {
            // Check up to 4 devices
            if let Ok(_device) = Device::new_metal(i) {
                devices.push(DeviceInfo {
                    device_type: DeviceType::Metal,
                    index: i,
                    name: format!("Metal Device {}", i),
                    memory_total: None,
                    memory_available: None,
                    compute_capability: None,
                    cores: None,
                    clock_speed_mhz: None,
                    is_available: true,
                });
            } else {
                break;
            }
        }

        Ok(devices)
    }

    /// Detect ROCm devices
    #[cfg(feature = "rocm")]
    fn detect_rocm_devices() -> Result<Vec<DeviceInfo>> {
        let mut devices = Vec::new();

        // ROCm detection would be implemented here
        // Currently not supported in Candle

        Ok(devices)
    }

    /// Get the primary device
    pub fn primary_device(&self) -> &Device {
        &self.primary_device
    }

    /// Get device capabilities
    pub fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }

    /// Get all available devices
    pub fn available_devices(&self) -> &[DeviceInfo] {
        &self.available_devices
    }

    /// Check if a specific device type is available
    pub fn is_device_type_available(&self, device_type: DeviceType) -> bool {
        self.available_devices
            .iter()
            .any(|d| d.device_type == device_type && d.is_available)
    }

    /// Get device info for the primary device
    pub fn primary_device_info(&self) -> Option<&DeviceInfo> {
        let device_type = match &self.primary_device {
            Device::Cpu => DeviceType::Cpu,
            Device::Cuda(_) => DeviceType::Cuda,
            Device::Metal(_) => DeviceType::Metal,
        };

        self.available_devices
            .iter()
            .find(|d| d.device_type == device_type)
    }

    /// Switch to a different device
    pub fn switch_device(&mut self, device: Device) -> Result<()> {
        self.primary_device = device;
        self.capabilities = Self::get_device_capabilities(&self.primary_device)?;
        Ok(())
    }

    /// Get optimal chunk size for the current device
    pub fn optimal_chunk_size(&self, tensor_size: usize) -> usize {
        match &self.primary_device {
            Device::Cpu => {
                // For CPU, use smaller chunks to avoid memory issues
                std::cmp::min(tensor_size, 1000)
            }
            Device::Cuda(_) => {
                // For CUDA, use larger chunks for better GPU utilization
                std::cmp::min(tensor_size, 10000)
            }
            Device::Metal(_) => {
                // For Metal, use medium chunks
                std::cmp::min(tensor_size, 5000)
            }
        }
    }

    /// Check if the device supports sub-microsecond operations
    pub fn supports_sub_microsecond(&self) -> bool {
        match &self.primary_device {
            Device::Cpu => false,
            Device::Cuda(_) => true,
            Device::Metal(_) => true,
        }
    }

    /// Get device memory info
    pub fn memory_info(&self) -> Option<(u64, u64)> {
        if let Some(device_info) = self.primary_device_info() {
            if let (Some(total), Some(available)) = (device_info.memory_total, device_info.memory_available) {
                return Some((total, available));
            }
        }
        None
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            primary_device: Device::Cpu,
            available_devices: vec![DeviceInfo {
                device_type: DeviceType::Cpu,
                index: 0,
                name: "CPU".to_string(),
                memory_total: None,
                memory_available: None,
                compute_capability: None,
                cores: None,
                clock_speed_mhz: None,
                is_available: true,
            }],
            capabilities: DeviceCapabilities::default(),
        })
    }
}

// External dependency for CPU core count
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_manager_creation() {
        let device_manager = DeviceManager::new();
        assert!(device_manager.is_ok());
        
        let dm = device_manager.unwrap();
        assert!(!dm.available_devices.is_empty());
        
        // CPU should always be available
        assert!(dm.is_device_type_available(DeviceType::Cpu));
    }

    #[test]
    fn test_device_detection() {
        let devices = DeviceManager::detect_devices().unwrap();
        assert!(!devices.is_empty());
        
        // First device should be CPU
        assert_eq!(devices[0].device_type, DeviceType::Cpu);
        assert!(devices[0].is_available);
    }

    #[test]
    fn test_device_selection() {
        let devices = DeviceManager::detect_devices().unwrap();
        let best_device = DeviceManager::select_best_device(&devices);
        assert!(best_device.is_ok());
    }

    #[test]
    fn test_device_capabilities() {
        let device = Device::Cpu;
        let capabilities = DeviceManager::get_device_capabilities(&device).unwrap();
        assert!(capabilities.supports_simd);
        assert!(capabilities.supports_async);
    }

    #[test]
    fn test_device_type_display() {
        assert_eq!(DeviceType::Cpu.to_string(), "CPU");
        assert_eq!(DeviceType::Cuda.to_string(), "CUDA");
        assert_eq!(DeviceType::Metal.to_string(), "Metal");
    }

    #[test]
    fn test_optimal_chunk_size() {
        let dm = DeviceManager::default();
        
        let chunk_size = dm.optimal_chunk_size(500);
        assert_eq!(chunk_size, 500);
        
        let chunk_size = dm.optimal_chunk_size(5000);
        assert_eq!(chunk_size, 1000); // CPU limit
    }

    #[test]
    fn test_device_switch() {
        let mut dm = DeviceManager::default();
        let original_device = dm.primary_device.clone();
        
        // Switch to CPU (should always work)
        let result = dm.switch_device(Device::Cpu);
        assert!(result.is_ok());
        
        // Primary device should be CPU
        assert!(matches!(dm.primary_device, Device::Cpu));
    }

    #[test]
    fn test_device_info_serialization() {
        let device_info = DeviceInfo {
            device_type: DeviceType::Cpu,
            index: 0,
            name: "Test CPU".to_string(),
            memory_total: Some(8_000_000_000),
            memory_available: Some(4_000_000_000),
            compute_capability: None,
            cores: Some(8),
            clock_speed_mhz: Some(3000),
            is_available: true,
        };

        let serialized = serde_json::to_string(&device_info).unwrap();
        let deserialized: DeviceInfo = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(device_info.device_type, deserialized.device_type);
        assert_eq!(device_info.name, deserialized.name);
        assert_eq!(device_info.memory_total, deserialized.memory_total);
    }
}