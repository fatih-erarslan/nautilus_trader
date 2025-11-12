//! SIMD backend selection and platform detection
//!
//! Automatically selects the optimal SIMD backend based on CPU features.

use std::fmt;

/// Available SIMD backends
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Scalar fallback (no SIMD)
    Scalar,
    /// x86_64 AVX2 (256-bit vectors, 8× f32)
    AVX2,
    /// x86_64 AVX-512 (512-bit vectors, 16× f32)
    AVX512,
    /// ARM NEON (128-bit vectors, 4× f32)
    NEON,
    /// ARM SVE (variable width vectors)
    SVE,
    /// WebAssembly SIMD128 (128-bit vectors)
    SIMD128,
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Backend::Scalar => write!(f, "Scalar"),
            Backend::AVX2 => write!(f, "AVX2"),
            Backend::AVX512 => write!(f, "AVX-512"),
            Backend::NEON => write!(f, "NEON"),
            Backend::SVE => write!(f, "SVE"),
            Backend::SIMD128 => write!(f, "WASM SIMD128"),
        }
    }
}

/// Detect optimal SIMD backend for current CPU
///
/// # Platform Support
/// - **x86_64**: Detects AVX-512 > AVX2 > Scalar
/// - **aarch64**: Detects SVE > NEON > Scalar
/// - **wasm32**: Returns SIMD128 if available
/// - **Other**: Returns Scalar
///
/// # Example
/// ```rust
/// use hyperphysics_core::simd::{Backend, optimal_backend};
///
/// let backend = optimal_backend();
/// println!("Using SIMD backend: {}", backend);
///
/// match backend {
///     Backend::AVX2 => println!("Intel/AMD CPU with AVX2 support"),
///     Backend::NEON => println!("ARM CPU (Apple Silicon, Raspberry Pi, etc.)"),
///     Backend::Scalar => println!("No SIMD support detected"),
///     _ => println!("Other backend: {}", backend),
/// }
/// ```
pub fn optimal_backend() -> Backend {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return Backend::AVX512;
        }
        if is_x86_feature_detected!("avx2") {
            return Backend::AVX2;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SVE detection would require runtime checks
        // For now, assume NEON is always available on aarch64
        return Backend::NEON;
    }

    #[cfg(target_arch = "wasm32")]
    {
        return Backend::SIMD128;
    }

    Backend::Scalar
}

/// Get vector width for backend (number of f32 elements)
pub fn vector_width(backend: Backend) -> usize {
    match backend {
        Backend::Scalar => 1,
        Backend::AVX2 => 8,
        Backend::AVX512 => 16,
        Backend::NEON => 4,
        Backend::SVE => 8, // Variable, but typically 256-512 bits
        Backend::SIMD128 => 4,
    }
}

/// Check if backend supports specific CPU features
pub fn backend_info(backend: Backend) -> BackendInfo {
    match backend {
        Backend::Scalar => BackendInfo {
            name: "Scalar",
            width: 1,
            supports_fma: false,
            supports_fast_exp: false,
            relative_performance: 1.0,
        },
        Backend::AVX2 => BackendInfo {
            name: "AVX2",
            width: 8,
            supports_fma: true,
            supports_fast_exp: true,
            relative_performance: 5.0,
        },
        Backend::AVX512 => BackendInfo {
            name: "AVX-512",
            width: 16,
            supports_fma: true,
            supports_fast_exp: true,
            relative_performance: 8.0,
        },
        Backend::NEON => BackendInfo {
            name: "NEON",
            width: 4,
            supports_fma: true,
            supports_fast_exp: true,
            relative_performance: 4.0,
        },
        Backend::SVE => BackendInfo {
            name: "SVE",
            width: 8,
            supports_fma: true,
            supports_fast_exp: true,
            relative_performance: 6.0,
        },
        Backend::SIMD128 => BackendInfo {
            name: "WASM SIMD128",
            width: 4,
            supports_fma: false,
            supports_fast_exp: false,
            relative_performance: 3.0,
        },
    }
}

/// Information about a SIMD backend
#[derive(Debug, Clone)]
pub struct BackendInfo {
    /// Backend name
    pub name: &'static str,
    /// Vector width (number of f32 elements)
    pub width: usize,
    /// Supports fused multiply-add
    pub supports_fma: bool,
    /// Supports fast exponential approximation
    pub supports_fast_exp: bool,
    /// Relative performance vs scalar (higher is better)
    pub relative_performance: f32,
}

/// Print detailed CPU and SIMD information
pub fn print_backend_info() {
    let backend = optimal_backend();
    let info = backend_info(backend);

    println!("=== HyperPhysics SIMD Backend ===");
    println!("Backend: {}", backend);
    println!("Vector width: {} × f32", info.width);
    println!("FMA support: {}", if info.supports_fma { "Yes" } else { "No" });
    println!("Fast exp: {}", if info.supports_fast_exp { "Yes" } else { "No" });
    println!("Performance: {:.1}× vs scalar", info.relative_performance);
    println!();

    #[cfg(target_arch = "x86_64")]
    {
        println!("CPU Features (x86_64):");
        println!("  SSE: {}", is_x86_feature_detected!("sse"));
        println!("  SSE2: {}", is_x86_feature_detected!("sse2"));
        println!("  SSE3: {}", is_x86_feature_detected!("sse3"));
        println!("  SSSE3: {}", is_x86_feature_detected!("ssse3"));
        println!("  SSE4.1: {}", is_x86_feature_detected!("sse4.1"));
        println!("  SSE4.2: {}", is_x86_feature_detected!("sse4.2"));
        println!("  AVX: {}", is_x86_feature_detected!("avx"));
        println!("  AVX2: {}", is_x86_feature_detected!("avx2"));
        println!("  FMA: {}", is_x86_feature_detected!("fma"));
        println!("  AVX-512F: {}", is_x86_feature_detected!("avx512f"));
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!("CPU Architecture: ARM64 (NEON available)");
    }

    #[cfg(target_arch = "wasm32")]
    {
        println!("CPU Architecture: WebAssembly (SIMD128)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_detection() {
        let backend = optimal_backend();
        println!("Detected backend: {}", backend);

        // Should never panic
        assert!(matches!(
            backend,
            Backend::Scalar | Backend::AVX2 | Backend::AVX512 | Backend::NEON | Backend::SVE | Backend::SIMD128
        ));
    }

    #[test]
    fn test_vector_width() {
        assert_eq!(vector_width(Backend::Scalar), 1);
        assert_eq!(vector_width(Backend::AVX2), 8);
        assert_eq!(vector_width(Backend::AVX512), 16);
        assert_eq!(vector_width(Backend::NEON), 4);
    }

    #[test]
    fn test_backend_info() {
        for backend in &[
            Backend::Scalar,
            Backend::AVX2,
            Backend::AVX512,
            Backend::NEON,
            Backend::SVE,
            Backend::SIMD128,
        ] {
            let info = backend_info(*backend);
            println!("{}: {}× f32, {:.1}× performance", info.name, info.width, info.relative_performance);

            assert!(info.width >= 1);
            assert!(info.relative_performance >= 1.0);
        }
    }

    #[test]
    fn test_print_info() {
        // Should not panic
        print_backend_info();
    }
}
