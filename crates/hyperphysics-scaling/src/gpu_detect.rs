//! GPU detection and capability probing
//!
//! Integrates with hyperphysics-gpu to detect available GPUs and their capabilities.

use crate::GPUInfo;
use hyperphysics_gpu::{GPUBackend, backend::wgpu::WGPUBackend};

/// Detect all available GPUs on the system
///
/// Returns a list of GPUs with their capabilities, sorted by performance (VRAM descending).
pub async fn detect_all_gpus() -> Vec<GPUInfo> {
    let mut gpus = Vec::new();

    // FIXED: Enumerate ALL adapters instead of just getting one
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        #[cfg(target_os = "macos")]
        backends: wgpu::Backends::METAL,
        #[cfg(not(target_os = "macos"))]
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapters: Vec<_> = instance.enumerate_adapters(
        #[cfg(target_os = "macos")]
        wgpu::Backends::METAL,
        #[cfg(not(target_os = "macos"))]
        wgpu::Backends::all(),
    );

    for adapter in adapters {
        let info = adapter.get_info();
        let limits = adapter.limits();

        gpus.push(GPUInfo {
            device_name: format!("{} ({:?})", info.name, info.backend),
            max_buffer_size: limits.max_buffer_size,
            max_workgroup_size: limits.max_compute_workgroup_size_x,
            available: true, // If enumerated, it's available
        });
    }

    // Sort by max_buffer_size (VRAM proxy) descending
    gpus.sort_by(|a, b| b.max_buffer_size.cmp(&a.max_buffer_size));

    // Fallback to WGPU backend if direct enumeration failed
    if gpus.is_empty() {
        if let Ok(backend) = WGPUBackend::new().await {
            let caps = backend.capabilities();
            gpus.push(GPUInfo {
                device_name: caps.device_name.clone(),
                max_buffer_size: caps.max_buffer_size,
                max_workgroup_size: caps.max_workgroup_size,
                available: caps.supports_compute,
            });
        }
    }

    gpus
}

/// Select best GPU for given workload
///
/// # Arguments
/// * `node_count` - Number of pBits in simulation
/// * `gpus` - Available GPUs
///
/// # Returns
/// Index of best GPU, or None if no suitable GPU found
pub fn select_best_gpu(node_count: usize, gpus: &[GPUInfo]) -> Option<usize> {
    if gpus.is_empty() {
        return None;
    }

    // Calculate required buffer size
    let required_buffer = (node_count * 32) as u64;

    // Find first GPU that can handle the workload
    for (i, gpu) in gpus.iter().enumerate() {
        if gpu.available && gpu.max_buffer_size >= required_buffer {
            return Some(i);
        }
    }

    None
}

/// Estimate GPU speedup factor vs CPU
///
/// # Arguments
/// * `node_count` - Number of pBits
/// * `gpu_info` - GPU capabilities
///
/// # Returns
/// Expected speedup factor (e.g., 100.0 means 100× faster)
pub fn estimate_speedup(node_count: usize, gpu_info: &GPUInfo) -> f64 {
    // Base speedup depends on problem size
    let base_speedup = if node_count < 1000 {
        // Small problems: overhead dominates
        2.0
    } else if node_count < 100_000 {
        // Medium problems: good parallelism
        50.0
    } else if node_count < 10_000_000 {
        // Large problems: excellent parallelism
        200.0
    } else {
        // Massive problems: optimal GPU utilization
        500.0
    };

    // Adjust for GPU capabilities
    let workgroup_factor = (gpu_info.max_workgroup_size as f64 / 256.0).min(2.0);
    let memory_factor = (gpu_info.max_buffer_size as f64 / 1e9).min(2.0);

    base_speedup * workgroup_factor * memory_factor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gpu_detection() {
        let gpus = detect_all_gpus().await;
        // May or may not have GPU depending on system
        println!("Detected {} GPU(s)", gpus.len());

        for (i, gpu) in gpus.iter().enumerate() {
            println!("GPU {}: {} (buffer: {} MB, workgroup: {})",
                i, gpu.device_name,
                gpu.max_buffer_size / 1_000_000,
                gpu.max_workgroup_size);
        }
    }

    #[test]
    fn test_gpu_selection() {
        let gpus = vec![
            GPUInfo {
                device_name: "Low-end GPU".to_string(),
                max_buffer_size: 1_000_000_000, // 1 GB
                max_workgroup_size: 128,
                available: true,
            },
            GPUInfo {
                device_name: "High-end GPU".to_string(),
                max_buffer_size: 8_000_000_000, // 8 GB
                max_workgroup_size: 1024,
                available: true,
            },
        ];

        // Small workload: first GPU is fine
        let selection = select_best_gpu(10_000, &gpus);
        assert_eq!(selection, Some(0));

        // Large workload: needs high-end GPU
        let selection = select_best_gpu(100_000_000, &gpus);
        assert_eq!(selection, Some(1));

        // Massive workload: exceeds both GPUs
        let selection = select_best_gpu(1_000_000_000, &gpus);
        assert!(selection.is_none());
    }

    #[test]
    fn test_speedup_estimation() {
        let gpu_info = GPUInfo {
            device_name: "Test GPU".to_string(),
            max_buffer_size: 4_000_000_000,
            max_workgroup_size: 256,
            available: true,
        };

        // Small workload: minimal speedup
        let speedup = estimate_speedup(100, &gpu_info);
        assert!(speedup < 10.0);

        // Large workload: significant speedup
        let speedup = estimate_speedup(1_000_000, &gpu_info);
        assert!(speedup > 100.0);
    }
}
