//! SIMD-accelerated operations for HyperPhysics
//!
//! This module provides vectorized implementations of critical mathematical
//! operations, targeting 3-5× performance improvement over scalar code.
//!
//! # Supported Architectures
//!
//! - **x86_64**: AVX2 (256-bit vectors), AVX-512 (512-bit vectors)
//! - **aarch64**: NEON (128-bit vectors), SVE (variable width)
//! - **wasm32**: SIMD128 (128-bit vectors)
//!
//! # Feature Flags
//!
//! Enable SIMD optimizations with:
//! ```toml
//! [features]
//! simd = []
//! ```
//!
//! # Performance Targets
//!
//! | Operation | Scalar | SIMD Target | Speedup |
//! |-----------|--------|-------------|---------|
//! | Sigmoid   | 50 µs  | 10 µs       | 5×      |
//! | Entropy   | 100 µs | 20 µs       | 5×      |
//! | Energy    | 200 µs | 50 µs       | 4×      |
//! | Dot product | 10 µs | 2 µs      | 5×      |

pub mod math;
pub mod engine;
pub mod backend;

pub use math::{
    sigmoid_vectorized,
    shannon_entropy_vectorized,
    dot_product_vectorized,
    exp_vectorized,
};

pub use backend::{Backend, optimal_backend};

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sigmoid_vectorized() {
        let input: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0f32; 8];

        sigmoid_vectorized(&input, &mut output);

        // Verify sigmoid properties
        for (i, &val) in output.iter().enumerate() {
            assert!(val > 0.0 && val < 1.0, "Sigmoid must be in (0,1) at index {}", i);
        }

        // Verify sigmoid(0) ≈ 0.5
        assert_relative_eq!(output[2], 0.5, epsilon = 0.01);
    }

    #[test]
    fn test_entropy_basic() {
        // Uniform distribution: maximum entropy
        let uniform: Vec<f32> = vec![0.25; 4];
        let entropy_uniform = shannon_entropy_vectorized(&uniform);

        // Expected: -4 * (0.25 * ln(0.25)) = -4 * (0.25 * -1.386) ≈ 1.386
        assert_relative_eq!(entropy_uniform, 1.386, epsilon = 0.01);

        // Concentrated distribution: low entropy
        let concentrated: Vec<f32> = vec![0.97, 0.01, 0.01, 0.01];
        let entropy_concentrated = shannon_entropy_vectorized(&concentrated);

        assert!(entropy_concentrated < entropy_uniform);
    }

    #[test]
    fn test_dot_product() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b: Vec<f32> = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let result = dot_product_vectorized(&a, &b);

        // Expected: 1*8 + 2*7 + 3*6 + 4*5 + 5*4 + 6*3 + 7*2 + 8*1 = 120
        assert_relative_eq!(result, 120.0, epsilon = 0.001);
    }

    #[test]
    fn test_backend_detection() {
        let backend = optimal_backend();
        println!("Detected optimal backend: {:?}", backend);

        // Should detect something (never panic)
        assert!(matches!(
            backend,
            Backend::Scalar | Backend::AVX2 | Backend::NEON | Backend::SIMD128
        ));
    }
}
