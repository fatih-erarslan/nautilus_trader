//! Platform-specific SIMD optimizations for CDFA
//! 
//! This crate provides hardware-accelerated implementations using:
//! - AVX2 on x86_64 (stable Rust compatible)
//! - Portable SIMD via the `wide` crate
//! - Scalar fallbacks for all platforms
//! 
//! AVX-512 support is available only with nightly Rust and explicit feature flag

#[cfg(feature = "avx2")]
pub mod avx2;

#[cfg(all(feature = "avx512", feature = "nightly"))]
pub mod avx512;

pub mod portable;
pub mod scalar;
pub mod runtime;

pub use runtime::*;

/// Aligned vector type for SIMD operations
pub type AlignedVec = Vec<f64>;

/// Unified SIMD API for all algorithms
/// 
/// Automatically dispatches to the best available implementation
/// based on runtime CPU feature detection
pub mod unified {
    use super::*;
    
    /// Compute Pearson correlation coefficient
    /// 
    /// Performance targets:
    /// - AVX2: <100ns for 256 elements  
    /// - Portable SIMD: <150ns for 256 elements
    /// - Scalar: <300ns for 256 elements
    pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
        debug_assert_eq!(x.len(), y.len());
        
        match best_implementation() {
            #[cfg(all(feature = "avx512", feature = "nightly", target_arch = "x86_64"))]
            SimdImplementation::Avx512 => unsafe { avx512::correlation_avx512(x, y) },
            
            #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
            SimdImplementation::Avx2 => unsafe { avx2::correlation_avx2(x, y) },
            
            SimdImplementation::Portable => portable::correlation_portable(x, y),
            
            _ => scalar::correlation_scalar(x, y),
        }
    }
    
    /// Discrete Wavelet Transform (Haar wavelet)
    /// 
    /// Performance targets:
    /// - AVX2: <100ns for small transforms
    /// - Portable SIMD: <150ns for small transforms
    /// - Scalar: <300ns for small transforms
    pub fn dwt_haar(signal: &[f64], approx: &mut [f64], detail: &mut [f64]) {
        debug_assert_eq!(signal.len() % 2, 0);
        debug_assert_eq!(approx.len(), signal.len() / 2);
        debug_assert_eq!(detail.len(), signal.len() / 2);
        
        match best_implementation() {
            #[cfg(all(feature = "avx512", feature = "nightly", target_arch = "x86_64"))]
            SimdImplementation::Avx512 => unsafe { avx512::dwt_haar_avx512(signal, approx, detail) },
            
            #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
            SimdImplementation::Avx2 => unsafe { avx2::dwt_haar_avx2(signal, approx, detail) },
            
            SimdImplementation::Portable => portable::dwt_haar_portable(signal, approx, detail),
            
            _ => scalar::dwt_haar_scalar(signal, approx, detail),
        }
    }
    
    /// Euclidean distance calculation
    /// 
    /// Performance targets:
    /// - AVX2: <50ns for 256 elements
    /// - Portable SIMD: <100ns for 256 elements
    /// - Scalar: <200ns for 256 elements
    pub fn euclidean_distance(x: &[f64], y: &[f64]) -> f64 {
        debug_assert_eq!(x.len(), y.len());
        
        match best_implementation() {
            #[cfg(all(feature = "avx512", feature = "nightly", target_arch = "x86_64"))]
            SimdImplementation::Avx512 => unsafe { avx512::euclidean_distance_avx512(x, y) },
            
            #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
            SimdImplementation::Avx2 => unsafe { avx2::euclidean_distance_avx2(x, y) },
            
            SimdImplementation::Portable => portable::euclidean_distance_portable(x, y),
            
            _ => scalar::euclidean_distance_scalar(x, y),
        }
    }
    
    /// Matrix multiplication
    /// 
    /// Performance targets:
    /// - AVX2: <100ns for 64x64 matrices
    /// - Portable SIMD: <200ns for 64x64 matrices
    /// - Scalar: <500ns for 64x64 matrices
    pub fn matrix_multiply(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
        match best_implementation() {
            #[cfg(all(feature = "avx512", feature = "nightly", target_arch = "x86_64"))]
            SimdImplementation::Avx512 => unsafe { avx512::matrix_multiply_avx512(a, b, c, m, n, k) },
            
            #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
            SimdImplementation::Avx2 => unsafe { avx2::matrix_multiply_avx2(a, b, c, m, n, k) },
            
            SimdImplementation::Portable => portable::matrix_multiply_portable(a, b, c, m, n, k),
            
            _ => scalar::matrix_multiply_scalar(a, b, c, m, n, k),
        }
    }
    
    /// Softmax function
    /// 
    /// Performance targets:
    /// - AVX2: <50ns for 256 elements
    /// - Portable SIMD: <100ns for 256 elements
    /// - Scalar: <200ns for 256 elements
    pub fn softmax(input: &[f64], output: &mut [f64]) {
        debug_assert_eq!(input.len(), output.len());
        
        match best_implementation() {
            #[cfg(all(feature = "avx512", feature = "nightly", target_arch = "x86_64"))]
            SimdImplementation::Avx512 => unsafe { avx512::softmax_avx512(input, output) },
            
            #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
            SimdImplementation::Avx2 => unsafe { avx2::softmax_avx2(input, output) },
            
            SimdImplementation::Portable => portable::softmax_portable(input, output),
            
            _ => scalar::softmax_scalar(input, output),
        }
    }
}

/// Scalar implementations for fallback
mod scalar_impl {
    pub use super::scalar::*;
}

/// Portable SIMD implementations using the `wide` crate
mod portable_impl {
    pub use super::portable::*;
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        
        let result = unified::correlation(&x, &y);
        assert_relative_eq!(result, -1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_euclidean_distance() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![5.0, 6.0, 7.0, 8.0];
        
        let result = unified::euclidean_distance(&x, &y);
        let expected = 8.0; // sqrt(16 + 16 + 16 + 16) = sqrt(64) = 8
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }
}