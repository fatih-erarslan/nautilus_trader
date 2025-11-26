//! SIMD Vectorization Module
//!
//! Provides 2-4x speedup for neural network operations using SIMD instructions.
//!
//! ## Supported Architectures
//!
//! - **x86_64**: AVX2, AVX-512
//! - **ARM**: NEON
//! - **Fallback**: Scalar implementations for unsupported platforms
//!
//! ## Features
//!
//! - Matrix multiplication (GEMM)
//! - Activation functions (ReLU, GELU, Tanh, Sigmoid, Softmax)
//! - Loss calculations (MSE, MAE, gradients)
//! - Automatic CPU feature detection
//! - Runtime fallback to scalar code
//!
//! ## Usage
//!
//! ```rust,no_run
//! use neuro_divergent::optimizations::simd::matmul;
//!
//! let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
//! let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
//! let result = matmul::gemm(&a, &b);
//! ```

pub mod cpu_features;
pub mod matmul;
pub mod activations;
pub mod losses;
pub mod utils;

pub use cpu_features::{CpuFeatures, detect_cpu_features};
pub use matmul::{gemm, gemv, dot_product};
pub use activations::{relu, gelu, tanh, sigmoid, softmax};
pub use losses::{mse, mae, mse_gradient, mae_gradient};

/// SIMD lane sizes for different types
pub const F32_LANES: usize = 8;  // 256-bit SIMD (AVX2)
pub const F64_LANES: usize = 4;  // 256-bit SIMD (AVX2)

/// Check if SIMD is available at runtime
pub fn is_simd_available() -> bool {
    detect_cpu_features().has_avx2 || detect_cpu_features().has_neon
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_feature_detection() {
        let features = detect_cpu_features();
        println!("CPU Features: {:?}", features);
        // Just verify it doesn't panic
    }

    #[test]
    fn test_simd_availability() {
        let available = is_simd_available();
        println!("SIMD available: {}", available);
        // Just verify it doesn't panic
    }
}
