//! Backend abstraction layer for multi-platform GPU/CPU support
//!
//! Supports:
//! - CPU (ndarray) - All platforms
//! - CUDA - Linux with NVIDIA GPUs
//! - ROCm/HIP - Linux with AMD GPUs
//! - Metal - macOS with Apple Silicon
//! - Vulkan - Cross-platform GPU
//! - WebGPU - Browser/Edge deployment

mod device;
mod router;

pub use device::{Device, DeviceInfo};
pub use router::{Backend, BackendType};

use crate::error::{MlError, MlResult};

/// Detect available backends on the current platform
pub fn detect_backends() -> Vec<BackendType> {
    let mut backends = vec![BackendType::Cpu]; // CPU always available

    #[cfg(feature = "cuda")]
    if is_cuda_available() {
        backends.push(BackendType::Cuda);
    }

    #[cfg(feature = "rocm")]
    if is_rocm_available() {
        backends.push(BackendType::Rocm);
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    if is_metal_available() {
        backends.push(BackendType::Metal);
    }

    #[cfg(feature = "vulkan")]
    if is_vulkan_available() {
        backends.push(BackendType::Vulkan);
    }

    #[cfg(feature = "wgpu")]
    if is_wgpu_available() {
        backends.push(BackendType::WebGpu);
    }

    backends
}

/// Select the best available backend for the current platform
pub fn select_best_backend() -> BackendType {
    let backends = detect_backends();

    // Priority order: CUDA > Metal > ROCm > Vulkan > WebGPU > CPU
    #[cfg(feature = "cuda")]
    if backends.contains(&BackendType::Cuda) {
        return BackendType::Cuda;
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    if backends.contains(&BackendType::Metal) {
        return BackendType::Metal;
    }

    #[cfg(feature = "rocm")]
    if backends.contains(&BackendType::Rocm) {
        return BackendType::Rocm;
    }

    #[cfg(feature = "vulkan")]
    if backends.contains(&BackendType::Vulkan) {
        return BackendType::Vulkan;
    }

    #[cfg(feature = "wgpu")]
    if backends.contains(&BackendType::WebGpu) {
        return BackendType::WebGpu;
    }

    BackendType::Cpu
}

// Backend availability checks

#[cfg(feature = "cuda")]
fn is_cuda_available() -> bool {
    // Check for CUDA runtime
    std::process::Command::new("nvidia-smi")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[cfg(not(feature = "cuda"))]
fn is_cuda_available() -> bool {
    false
}

#[cfg(feature = "rocm")]
fn is_rocm_available() -> bool {
    // Check for ROCm runtime
    std::process::Command::new("rocm-smi")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[cfg(not(feature = "rocm"))]
fn is_rocm_available() -> bool {
    false
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn is_metal_available() -> bool {
    // Metal is always available on macOS 10.14+
    true
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn is_metal_available() -> bool {
    false
}

#[cfg(feature = "vulkan")]
fn is_vulkan_available() -> bool {
    // Check for Vulkan ICD
    std::env::var("VK_ICD_FILENAMES").is_ok()
        || std::path::Path::new("/usr/share/vulkan/icd.d").exists()
        || std::path::Path::new("/etc/vulkan/icd.d").exists()
}

#[cfg(not(feature = "vulkan"))]
fn is_vulkan_available() -> bool {
    false
}

#[cfg(feature = "wgpu")]
fn is_wgpu_available() -> bool {
    // WebGPU/wgpu is generally available if compiled
    true
}

#[cfg(not(feature = "wgpu"))]
fn is_wgpu_available() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_backends() {
        let backends = detect_backends();
        assert!(backends.contains(&BackendType::Cpu));
    }

    #[test]
    fn test_select_best_backend() {
        let backend = select_best_backend();
        // Should at least return CPU
        assert!(matches!(
            backend,
            BackendType::Cpu
                | BackendType::Cuda
                | BackendType::Metal
                | BackendType::Rocm
                | BackendType::Vulkan
                | BackendType::WebGpu
        ));
    }
}
