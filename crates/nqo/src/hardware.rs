//! Hardware detection and optimization for NQO

use raw_cpuid::CpuId;
use std::sync::OnceLock;
use tracing::info;

/// Hardware capabilities
#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    /// AVX-512 support
    pub has_avx512: bool,
    /// AVX2 support
    pub has_avx2: bool,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// CUDA GPU available
    pub has_cuda: bool,
    /// OpenCL available
    pub has_opencl: bool,
    /// WebGPU available
    pub has_webgpu: bool,
    /// Apple Silicon GPU
    pub has_apple_gpu: bool,
    /// AMD GPU available
    pub has_amd_gpu: bool,
    /// Total system memory in bytes
    pub total_memory: u64,
}

static HARDWARE_CAPS: OnceLock<HardwareCapabilities> = OnceLock::new();

/// Detect hardware capabilities
pub fn detect_hardware() -> &'static HardwareCapabilities {
    HARDWARE_CAPS.get_or_init(|| {
        let cpuid = CpuId::new();
        
        // Detect CPU features
        let has_avx512 = cpuid.get_feature_info()
            .map(|_f| {
                cpuid.get_extended_feature_info()
                    .map(|ef| ef.has_avx512f())
                    .unwrap_or(false)
            })
            .unwrap_or(false);
            
        let has_avx2 = cpuid.get_feature_info()
            .map(|_f| {
                cpuid.get_extended_feature_info()
                    .map(|ef| ef.has_avx2())
                    .unwrap_or(false)
            })
            .unwrap_or(false);
        
        let cpu_cores = num_cpus::get();
        
        // Detect GPU capabilities
        let has_cuda = detect_cuda();
        let has_opencl = detect_opencl();
        let has_webgpu = cfg!(feature = "gpu-wgpu");
        let has_apple_gpu = detect_apple_gpu();
        let has_amd_gpu = detect_amd_gpu();
        
        // Get system memory
        let sys = sysinfo::System::new_all();
        let total_memory = sys.total_memory();
        
        let caps = HardwareCapabilities {
            has_avx512,
            has_avx2,
            cpu_cores,
            has_cuda,
            has_opencl,
            has_webgpu,
            has_apple_gpu,
            has_amd_gpu,
            total_memory,
        };
        
        info!("Hardware capabilities detected: {:?}", caps);
        caps
    })
}

/// Detect CUDA availability
#[cfg(feature = "gpu-cuda")]
fn detect_cuda() -> bool {
    use cuda_sys::cudart::{cudaGetDeviceCount, cudaError_t};
    
    unsafe {
        let mut device_count = 0;
        let result = cudaGetDeviceCount(&mut device_count);
        result == cudaError_t::Success && device_count > 0
    }
}

#[cfg(not(feature = "gpu-cuda"))]
fn detect_cuda() -> bool {
    false
}

/// Detect OpenCL availability
#[cfg(feature = "gpu-opencl")]
fn detect_opencl() -> bool {
    use opencl3::platform;
    
    match platform::get_platforms() {
        Ok(platforms) => !platforms.is_empty(),
        Err(_) => false,
    }
}

#[cfg(not(feature = "gpu-opencl"))]
fn detect_opencl() -> bool {
    false
}

/// Detect Apple GPU
fn detect_apple_gpu() -> bool {
    #[cfg(target_os = "macos")]
    {
        #[cfg(target_arch = "aarch64")]
        return true;
        #[cfg(not(target_arch = "aarch64"))]
        return false;
    }
    #[cfg(not(target_os = "macos"))]
    false
}

/// Detect AMD GPU
fn detect_amd_gpu() -> bool {
    let output = std::process::Command::new("lspci")
        .output()
        .ok();
        
    if let Some(output) = output {
        let text = String::from_utf8_lossy(&output.stdout);
        text.contains("AMD") && (text.contains("VGA") || text.contains("Display"))
    } else {
        false
    }
}

/// Get optimal device for quantum simulation
pub fn get_optimal_device() -> String {
    let caps = detect_hardware();
    
    if caps.has_cuda {
        info!("Using CUDA GPU for quantum optimization");
        "cuda".to_string()
    } else if caps.has_apple_gpu {
        info!("Using Apple Silicon GPU for quantum optimization");
        "metal".to_string()
    } else if caps.has_opencl {
        info!("Using OpenCL for quantum optimization");
        "opencl".to_string()
    } else if caps.has_webgpu {
        info!("Using WebGPU for quantum optimization");
        "webgpu".to_string()
    } else if caps.has_avx512 {
        info!("Using AVX-512 CPU for quantum optimization");
        "cpu-avx512".to_string()
    } else if caps.has_avx2 {
        info!("Using AVX2 CPU for quantum optimization");
        "cpu-avx2".to_string()
    } else {
        info!("Using standard CPU for quantum optimization");
        "cpu".to_string()
    }
}

/// SIMD feature detection for runtime dispatch
pub mod simd {
    use super::*;
    
    /// Check if AVX-512 is available
    #[inline]
    pub fn has_avx512() -> bool {
        detect_hardware().has_avx512
    }
    
    /// Check if AVX2 is available
    #[inline]
    pub fn has_avx2() -> bool {
        detect_hardware().has_avx2
    }
    
    /// Select optimal SIMD implementation
    #[cfg(feature = "simd")]
    pub fn select_simd_impl() -> &'static str {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx512f") {
                "avx512"
            } else if is_x86_feature_detected!("avx2") {
                "avx2"
            } else {
                "sse2"
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            "neon"
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            "scalar"
        }
    }
}