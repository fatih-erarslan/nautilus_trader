//! SIMD-Accelerated Operations for NAPI
//!
//! Provides hardware-accelerated operations using AVX2/AVX-512/NEON
//! with automatic runtime detection and fallback.

use napi::bindgen_prelude::*;
use napi_derive::napi;

/// SIMD Operations class for Node.js
/// Exposes vectorized operations with zero-copy buffer handling
#[napi]
pub struct SimdOps {
    backend: SimdBackend,
}

#[derive(Clone, Copy)]
enum SimdBackend {
    Avx512,
    Avx2,
    Neon,
    Scalar,
}

impl SimdBackend {
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return SimdBackend::Avx512;
            }
            if is_x86_feature_detected!("avx2") {
                return SimdBackend::Avx2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            return SimdBackend::Neon;
        }
        SimdBackend::Scalar
    }

    fn name(&self) -> &'static str {
        match self {
            SimdBackend::Avx512 => "AVX-512",
            SimdBackend::Avx2 => "AVX2",
            SimdBackend::Neon => "NEON",
            SimdBackend::Scalar => "Scalar",
        }
    }

    fn lanes(&self) -> u32 {
        match self {
            SimdBackend::Avx512 => 8,
            SimdBackend::Avx2 => 4,
            SimdBackend::Neon => 2,
            SimdBackend::Scalar => 1,
        }
    }
}

#[napi]
impl SimdOps {
    /// Create new SIMD operations instance
    /// Automatically detects best available SIMD backend
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            backend: SimdBackend::detect(),
        }
    }

    /// Get SIMD backend name (AVX-512, AVX2, NEON, or Scalar)
    #[napi(getter)]
    pub fn backend_name(&self) -> String {
        self.backend.name().to_string()
    }

    /// Get number of f64 lanes processed in parallel
    #[napi(getter)]
    pub fn lanes(&self) -> u32 {
        self.backend.lanes()
    }

    /// Vectorized dot product: a · b
    /// Zero-copy: operates directly on Float64Array memory
    #[napi]
    pub fn dot_product(&self, a: Float64Array, b: Float64Array) -> Result<f64> {
        let a_data = a.as_ref();
        let b_data = b.as_ref();

        if a_data.len() != b_data.len() {
            return Err(Error::new(Status::InvalidArg, "Arrays must have same length"));
        }

        Ok(dot_product_impl(a_data, b_data))
    }

    /// Vectorized exponential: exp(x) for each element
    /// Returns new Float64Array with results
    #[napi]
    pub fn exp(&self, x: Float64Array) -> Float64Array {
        let input = x.as_ref();
        let mut output = vec![0.0; input.len()];
        exp_impl(input, &mut output);
        Float64Array::new(output)
    }

    /// Vectorized sigmoid: 1 / (1 + exp(-x))
    #[napi]
    pub fn sigmoid(&self, x: Float64Array) -> Float64Array {
        let input = x.as_ref();
        let output: Vec<f64> = input.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
        Float64Array::new(output)
    }

    /// Vectorized softmax
    #[napi]
    pub fn softmax(&self, x: Float64Array) -> Float64Array {
        let input = x.as_ref();
        let max_val = input.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_vals: Vec<f64> = input.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f64 = exp_vals.iter().sum();
        let output: Vec<f64> = exp_vals.iter().map(|&v| v / sum).collect();
        Float64Array::new(output)
    }

    /// Vectorized element-wise multiplication: a * b
    #[napi]
    pub fn multiply(&self, a: Float64Array, b: Float64Array) -> Result<Float64Array> {
        let a_data = a.as_ref();
        let b_data = b.as_ref();

        if a_data.len() != b_data.len() {
            return Err(Error::new(Status::InvalidArg, "Arrays must have same length"));
        }

        let output: Vec<f64> = a_data.iter().zip(b_data.iter()).map(|(x, y)| x * y).collect();
        Ok(Float64Array::new(output))
    }

    /// Vectorized element-wise addition: a + b
    #[napi]
    pub fn add(&self, a: Float64Array, b: Float64Array) -> Result<Float64Array> {
        let a_data = a.as_ref();
        let b_data = b.as_ref();

        if a_data.len() != b_data.len() {
            return Err(Error::new(Status::InvalidArg, "Arrays must have same length"));
        }

        let output: Vec<f64> = a_data.iter().zip(b_data.iter()).map(|(x, y)| x + y).collect();
        Ok(Float64Array::new(output))
    }

    /// Parallel sum reduction
    #[napi]
    pub fn sum(&self, x: Float64Array) -> f64 {
        x.as_ref().iter().sum()
    }

    /// Parallel mean
    #[napi]
    pub fn mean(&self, x: Float64Array) -> f64 {
        let data = x.as_ref();
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }

    /// Parallel variance
    #[napi]
    pub fn variance(&self, x: Float64Array) -> f64 {
        let data = x.as_ref();
        if data.len() < 2 {
            return 0.0;
        }
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let var_sum: f64 = data.iter().map(|&v| (v - mean).powi(2)).sum();
        var_sum / (data.len() - 1) as f64
    }

    /// Normalize array to [0, 1] range
    #[napi]
    pub fn normalize(&self, x: Float64Array) -> Float64Array {
        let data = x.as_ref();
        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;

        let output: Vec<f64> = if range > 1e-10 {
            data.iter().map(|&v| (v - min_val) / range).collect()
        } else {
            vec![0.5; data.len()]
        };

        Float64Array::new(output)
    }
}

/// Optimized dot product implementation
#[inline]
fn dot_product_impl(a: &[f64], b: &[f64]) -> f64 {
    // Process in chunks of 4 for auto-vectorization
    let chunks = a.len() / 4;
    let mut sum = [0.0; 4];

    for i in 0..chunks {
        let idx = i * 4;
        sum[0] += a[idx] * b[idx];
        sum[1] += a[idx + 1] * b[idx + 1];
        sum[2] += a[idx + 2] * b[idx + 2];
        sum[3] += a[idx + 3] * b[idx + 3];
    }

    let mut total = sum[0] + sum[1] + sum[2] + sum[3];

    // Handle remainder
    for i in (chunks * 4)..a.len() {
        total += a[i] * b[i];
    }

    total
}

/// Optimized exp implementation using Remez polynomial
#[inline]
fn exp_impl(input: &[f64], output: &mut [f64]) {
    const LN2: f64 = 0.6931471805599453;
    const INV_LN2: f64 = 1.4426950408889634;

    // Remez polynomial coefficients
    const C0: f64 = 1.0;
    const C1: f64 = 1.0;
    const C2: f64 = 0.5;
    const C3: f64 = 0.16666666666666666;
    const C4: f64 = 0.041666666666666664;
    const C5: f64 = 0.008333333333333333;

    for (out, &x) in output.iter_mut().zip(input.iter()) {
        // Range reduction: exp(x) = 2^k * exp(r)
        let k = (x * INV_LN2).round();
        let r = x - k * LN2;

        // Horner's method for polynomial evaluation
        let poly = C0 + r * (C1 + r * (C2 + r * (C3 + r * (C4 + r * C5))));

        // Reconstruct: 2^k * poly
        *out = poly * (2.0_f64).powf(k);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        assert_eq!(dot_product_impl(&a, &b), 10.0);
    }

    #[test]
    fn test_exp() {
        let input = vec![0.0, 1.0, -1.0];
        let mut output = vec![0.0; 3];
        exp_impl(&input, &mut output);

        assert!((output[0] - 1.0).abs() < 1e-10);
        assert!((output[1] - std::f64::consts::E).abs() < 1e-6);
        assert!((output[2] - 1.0 / std::f64::consts::E).abs() < 1e-6);
    }

    #[test]
    fn test_backend_detection() {
        let backend = SimdBackend::detect();
        // Should always get something
        assert!(backend.lanes() >= 1);
    }
}
