//! GPU hardware detection and capability assessment
//!
//! This module provides runtime detection of available GPU hardware,
//! capability assessment, and automatic fallback mechanisms.

use crate::error::{CdfaError, CdfaResult};
use super::{GpuDeviceInfo, GpuBackend, GpuConfig};
use std::collections::HashMap;

/// Detect all available GPU devices
pub fn detect_gpu_devices() -> CdfaResult<Vec<GpuDeviceInfo>> {
    let mut devices = Vec::new();
    
    // Detect CUDA devices
    #[cfg(feature = "cuda")]
    {
        if let Ok(cuda_devices) = detect_cuda_devices() {
            devices.extend(cuda_devices);
        }
    }
    
    // Detect Metal devices (macOS)
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        if let Ok(metal_devices) = detect_metal_devices() {
            devices.extend(metal_devices);
        }
    }
    
    // Detect WebGPU devices
    #[cfg(feature = "webgpu")]
    {
        if let Ok(webgpu_devices) = detect_webgpu_devices() {
            devices.extend(webgpu_devices);
        }
    }
    
    // If no devices found, provide CPU fallback info
    if devices.is_empty() {
        devices.push(create_cpu_fallback_device());
    }
    
    Ok(devices)
}

/// Get the best available GPU device
pub fn get_best_gpu_device() -> CdfaResult<GpuDeviceInfo> {
    let devices = detect_gpu_devices()?;
    
    // Prioritize devices based on performance and capability
    let best_device = devices
        .into_iter()
        .max_by(|a, b| {
            // Priority order: CUDA > Metal > WebGPU > CPU
            let priority_a = get_backend_priority(a.backend);
            let priority_b = get_backend_priority(b.backend);
            
            priority_a.cmp(&priority_b)
                .then(a.memory_size.cmp(&b.memory_size))
                .then(a.max_work_group_size.cmp(&b.max_work_group_size))
        })
        .ok_or_else(|| CdfaError::GpuError("No GPU devices available".to_string()))?;
    
    Ok(best_device)
}

/// Get backend priority for device selection
fn get_backend_priority(backend: GpuBackend) -> u8 {
    match backend {
        #[cfg(feature = "cuda")]
        GpuBackend::Cuda => 4,
        #[cfg(feature = "metal")]
        GpuBackend::Metal => 3,
        #[cfg(feature = "webgpu")]
        GpuBackend::WebGpu => 2,
        GpuBackend::Cpu => 1,
    }
}

/// Detect CUDA devices
#[cfg(feature = "cuda")]
fn detect_cuda_devices() -> CdfaResult<Vec<GpuDeviceInfo>> {
    use cudarc::driver::CudaDevice;
    
    let mut devices = Vec::new();
    
    // Try to enumerate CUDA devices
    let device_count = match std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=count")
        .arg("--format=csv,noheader,nounits")
        .output()
    {
        Ok(output) => {
            let output_str = String::from_utf8_lossy(&output.stdout);
            output_str.trim().parse::<usize>().unwrap_or(0)
        }
        Err(_) => return Ok(devices), // No nvidia-smi, no CUDA devices
    };
    
    for device_id in 0..device_count {
        if let Ok(device) = CudaDevice::new(device_id) {
            let name = device.name().unwrap_or_else(|_| format!("CUDA Device {}", device_id));
            let total_memory = device.total_memory().unwrap_or(0);
            let (major, minor) = device.compute_capability().unwrap_or((0, 0));
            
            devices.push(GpuDeviceInfo {
                id: device_id as u32,
                name,
                backend: GpuBackend::Cuda,
                memory_size: total_memory,
                compute_capability: format!("{}.{}", major, minor),
                max_work_group_size: 1024, // CUDA max threads per block
                supports_double_precision: major >= 1 && minor >= 3,
                supports_half_precision: major >= 5 && minor >= 3,
            });
        }
    }
    
    Ok(devices)
}

/// Detect Metal devices (macOS only)
#[cfg(all(feature = "metal", target_os = "macos"))]
fn detect_metal_devices() -> CdfaResult<Vec<GpuDeviceInfo>> {
    use metal::Device;
    use objc::rc::autoreleasepool;
    
    let mut devices = Vec::new();
    
    autoreleasepool(|| {
        // System default device
        if let Some(device) = Device::system_default() {
            let name = device.name().to_string();
            let memory_size = if device.has_unified_memory() {
                device.recommended_max_working_set_size()
            } else {
                device.recommended_max_working_set_size()
            };
            
            devices.push(GpuDeviceInfo {
                id: 0,
                name,
                backend: GpuBackend::Metal,
                memory_size,
                compute_capability: "Metal".to_string(),
                max_work_group_size: device.max_threads_per_threadgroup().width as u32,
                supports_double_precision: true,
                supports_half_precision: true,
            });
        }
        
        // Try to enumerate all Metal devices
        let all_devices = Device::all();
        for (idx, device) in all_devices.iter().enumerate() {
            if idx == 0 {
                continue; // Skip system default which we already added
            }
            
            let name = device.name().to_string();
            let memory_size = device.recommended_max_working_set_size();
            
            devices.push(GpuDeviceInfo {
                id: idx as u32,
                name,
                backend: GpuBackend::Metal,
                memory_size,
                compute_capability: "Metal".to_string(),
                max_work_group_size: device.max_threads_per_threadgroup().width as u32,
                supports_double_precision: true,
                supports_half_precision: true,
            });
        }
    });
    
    Ok(devices)
}

/// Detect WebGPU devices
#[cfg(feature = "webgpu")]
fn detect_webgpu_devices() -> CdfaResult<Vec<GpuDeviceInfo>> {
    use wgpu::*;
    use pollster;
    
    let mut devices = Vec::new();
    
    pollster::block_on(async {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });
        
        // Try different power preferences to find all devices
        let power_preferences = [
            PowerPreference::HighPerformance,
            PowerPreference::LowPower,
        ];
        
        for (idx, power_pref) in power_preferences.iter().enumerate() {
            if let Some(adapter) = instance
                .request_adapter(&RequestAdapterOptions {
                    power_preference: *power_pref,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
            {
                let info = adapter.get_info();
                let limits = adapter.limits();
                
                devices.push(GpuDeviceInfo {
                    id: idx as u32,
                    name: info.name.clone(),
                    backend: GpuBackend::WebGpu,
                    memory_size: 0, // WebGPU doesn't expose memory info
                    compute_capability: format!("{:?}", info.backend),
                    max_work_group_size: limits.max_compute_workgroup_size_x,
                    supports_double_precision: false, // WebGPU typically doesn't support f64
                    supports_half_precision: true,
                });
            }
        }
    });
    
    Ok(devices)
}

/// Create CPU fallback device info
fn create_cpu_fallback_device() -> GpuDeviceInfo {
    let cpu_info = get_cpu_info();
    
    GpuDeviceInfo {
        id: u32::MAX, // Special ID for CPU fallback
        name: cpu_info.name,
        backend: GpuBackend::Cpu,
        memory_size: cpu_info.memory_size,
        compute_capability: cpu_info.features.join(", "),
        max_work_group_size: cpu_info.thread_count as u32,
        supports_double_precision: true,
        supports_half_precision: false, // CPU doesn't typically have hardware f16
    }
}

/// CPU information for fallback
struct CpuInfo {
    name: String,
    memory_size: u64,
    thread_count: usize,
    features: Vec<String>,
}

/// Get CPU information for fallback device
fn get_cpu_info() -> CpuInfo {
    let mut features = Vec::new();
    
    // Detect CPU features
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("sse2") {
            features.push("SSE2".to_string());
        }
        if std::arch::is_x86_feature_detected!("avx") {
            features.push("AVX".to_string());
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            features.push("AVX2".to_string());
        }
        if std::arch::is_x86_feature_detected!("fma") {
            features.push("FMA".to_string());
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            features.push("NEON".to_string());
        }
    }
    
    // Get CPU name and memory
    let name = get_cpu_name();
    let memory_size = get_system_memory();
    let thread_count = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    
    CpuInfo {
        name,
        memory_size,
        thread_count,
        features,
    }
}

/// Get CPU name from system
fn get_cpu_name() -> String {
    #[cfg(target_os = "linux")]
    {
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in cpuinfo.lines() {
                if line.starts_with("model name") {
                    if let Some(name) = line.split(':').nth(1) {
                        return name.trim().to_string();
                    }
                }
            }
        }
    }
    
    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("sysctl")
            .arg("-n")
            .arg("machdep.cpu.brand_string")
            .output()
        {
            if let Ok(name) = String::from_utf8(output.stdout) {
                return name.trim().to_string();
            }
        }
    }
    
    #[cfg(target_os = "windows")]
    {
        // Windows CPU detection would go here
        return "Unknown CPU".to_string();
    }
    
    "Unknown CPU".to_string()
}

/// Get system memory size
fn get_system_memory() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(size_str) = line.split_whitespace().nth(1) {
                        if let Ok(size_kb) = size_str.parse::<u64>() {
                            return size_kb * 1024; // Convert KB to bytes
                        }
                    }
                }
            }
        }
    }
    
    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("sysctl")
            .arg("-n")
            .arg("hw.memsize")
            .output()
        {
            if let Ok(size_str) = String::from_utf8(output.stdout) {
                if let Ok(size) = size_str.trim().parse::<u64>() {
                    return size;
                }
            }
        }
    }
    
    // Fallback to 8GB
    8 * 1024 * 1024 * 1024
}

/// Assess GPU capabilities for CDFA operations
pub fn assess_gpu_capabilities(device: &GpuDeviceInfo) -> GpuCapabilityAssessment {
    let mut assessment = GpuCapabilityAssessment::default();
    
    // Memory assessment
    assessment.memory_score = assess_memory_capability(device.memory_size);
    
    // Compute capability assessment
    assessment.compute_score = assess_compute_capability(device);
    
    // Backend-specific assessments
    assessment.backend_score = assess_backend_capability(device.backend);
    
    // Feature support
    assessment.feature_support = assess_feature_support(device);
    
    // Overall score
    assessment.overall_score = (assessment.memory_score + 
                              assessment.compute_score + 
                              assessment.backend_score) / 3.0;
    
    // Performance estimation
    assessment.estimated_performance = estimate_performance_tier(device);
    
    // Suitability for operations
    assessment.matrix_ops_suitable = assessment.overall_score > 0.6;
    assessment.ml_ops_suitable = assessment.overall_score > 0.7 && device.supports_half_precision;
    assessment.large_data_suitable = device.memory_size > 2 * 1024 * 1024 * 1024; // > 2GB
    
    assessment
}

/// GPU capability assessment results
#[derive(Debug, Clone)]
pub struct GpuCapabilityAssessment {
    pub memory_score: f32,
    pub compute_score: f32,
    pub backend_score: f32,
    pub overall_score: f32,
    pub feature_support: FeatureSupport,
    pub estimated_performance: PerformanceTier,
    pub matrix_ops_suitable: bool,
    pub ml_ops_suitable: bool,
    pub large_data_suitable: bool,
    pub recommended_batch_size: usize,
    pub recommended_work_group_size: u32,
}

/// Feature support assessment
#[derive(Debug, Clone)]
pub struct FeatureSupport {
    pub double_precision: bool,
    pub half_precision: bool,
    pub large_workgroups: bool,
    pub shared_memory: bool,
    pub atomic_operations: bool,
}

/// Performance tier estimation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceTier {
    Low,
    Medium,
    High,
    Extreme,
}

/// Assess memory capability
fn assess_memory_capability(memory_size: u64) -> f32 {
    match memory_size {
        0..=1_073_741_824 => 0.2,         // < 1GB
        1_073_741_824..=4_294_967_296 => 0.5,    // 1-4GB
        4_294_967_296..=8_589_934_592 => 0.7,    // 4-8GB
        8_589_934_592..=17_179_869_184 => 0.85,  // 8-16GB
        _ => 1.0,                         // > 16GB
    }
}

/// Assess compute capability
fn assess_compute_capability(device: &GpuDeviceInfo) -> f32 {
    let work_group_score = match device.max_work_group_size {
        0..=64 => 0.3,
        65..=256 => 0.6,
        257..=512 => 0.8,
        513..=1024 => 0.9,
        _ => 1.0,
    };
    
    let precision_score = match (device.supports_double_precision, device.supports_half_precision) {
        (true, true) => 1.0,
        (true, false) => 0.8,
        (false, true) => 0.6,
        (false, false) => 0.3,
    };
    
    (work_group_score + precision_score) / 2.0
}

/// Assess backend capability
fn assess_backend_capability(backend: GpuBackend) -> f32 {
    match backend {
        #[cfg(feature = "cuda")]
        GpuBackend::Cuda => 1.0,
        #[cfg(feature = "metal")]
        GpuBackend::Metal => 0.9,
        #[cfg(feature = "webgpu")]
        GpuBackend::WebGpu => 0.7,
        GpuBackend::Cpu => 0.3,
    }
}

/// Assess feature support
fn assess_feature_support(device: &GpuDeviceInfo) -> FeatureSupport {
    FeatureSupport {
        double_precision: device.supports_double_precision,
        half_precision: device.supports_half_precision,
        large_workgroups: device.max_work_group_size >= 512,
        shared_memory: match device.backend {
            #[cfg(feature = "cuda")]
            GpuBackend::Cuda => true,
            #[cfg(feature = "metal")]
            GpuBackend::Metal => true,
            #[cfg(feature = "webgpu")]
            GpuBackend::WebGpu => true,
            GpuBackend::Cpu => false,
        },
        atomic_operations: device.backend != GpuBackend::Cpu,
    }
}

/// Estimate performance tier
fn estimate_performance_tier(device: &GpuDeviceInfo) -> PerformanceTier {
    let memory_gb = device.memory_size / (1024 * 1024 * 1024);
    
    match device.backend {
        #[cfg(feature = "cuda")]
        GpuBackend::Cuda => {
            if memory_gb >= 16 && device.max_work_group_size >= 1024 {
                PerformanceTier::Extreme
            } else if memory_gb >= 8 {
                PerformanceTier::High
            } else if memory_gb >= 4 {
                PerformanceTier::Medium
            } else {
                PerformanceTier::Low
            }
        }
        #[cfg(feature = "metal")]
        GpuBackend::Metal => {
            if memory_gb >= 32 { // Unified memory systems
                PerformanceTier::High
            } else if memory_gb >= 16 {
                PerformanceTier::Medium
            } else {
                PerformanceTier::Low
            }
        }
        #[cfg(feature = "webgpu")]
        GpuBackend::WebGpu => PerformanceTier::Medium, // Conservative estimate
        GpuBackend::Cpu => PerformanceTier::Low,
    }
}

impl Default for GpuCapabilityAssessment {
    fn default() -> Self {
        Self {
            memory_score: 0.0,
            compute_score: 0.0,
            backend_score: 0.0,
            overall_score: 0.0,
            feature_support: FeatureSupport {
                double_precision: false,
                half_precision: false,
                large_workgroups: false,
                shared_memory: false,
                atomic_operations: false,
            },
            estimated_performance: PerformanceTier::Low,
            matrix_ops_suitable: false,
            ml_ops_suitable: false,
            large_data_suitable: false,
            recommended_batch_size: 100,
            recommended_work_group_size: 64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_device_detection() {
        let devices = detect_gpu_devices().unwrap();
        assert!(!devices.is_empty(), "Should always have at least CPU fallback");
        
        // Check that CPU fallback is present if no GPU devices
        let has_cpu_fallback = devices.iter().any(|d| d.backend == GpuBackend::Cpu);
        if devices.len() == 1 {
            assert!(has_cpu_fallback, "Should have CPU fallback if no GPU devices");
        }
    }
    
    #[test]
    fn test_cpu_fallback_device() {
        let cpu_device = create_cpu_fallback_device();
        assert_eq!(cpu_device.backend, GpuBackend::Cpu);
        assert_eq!(cpu_device.id, u32::MAX);
        assert!(!cpu_device.name.is_empty());
    }
    
    #[test]
    fn test_capability_assessment() {
        let device = GpuDeviceInfo {
            id: 0,
            name: "Test GPU".to_string(),
            backend: GpuBackend::Cpu,
            memory_size: 8 * 1024 * 1024 * 1024, // 8GB
            compute_capability: "Test".to_string(),
            max_work_group_size: 256,
            supports_double_precision: true,
            supports_half_precision: false,
        };
        
        let assessment = assess_gpu_capabilities(&device);
        assert!(assessment.overall_score > 0.0);
        assert!(assessment.memory_score > 0.5); // 8GB should get good score
    }
    
    #[test]
    fn test_performance_tier_estimation() {
        let high_end_device = GpuDeviceInfo {
            id: 0,
            name: "High-end GPU".to_string(),
            backend: GpuBackend::Cpu, // Using CPU for test
            memory_size: 32 * 1024 * 1024 * 1024, // 32GB
            compute_capability: "Test".to_string(),
            max_work_group_size: 1024,
            supports_double_precision: true,
            supports_half_precision: true,
        };
        
        let tier = estimate_performance_tier(&high_end_device);
        // Note: CPU backend always gets Low tier in current implementation
        assert_eq!(tier, PerformanceTier::Low);
    }
    
    #[test]
    fn test_backend_priority() {
        assert_eq!(get_backend_priority(GpuBackend::Cpu), 1);
        
        #[cfg(feature = "cuda")]
        assert_eq!(get_backend_priority(GpuBackend::Cuda), 4);
    }
}