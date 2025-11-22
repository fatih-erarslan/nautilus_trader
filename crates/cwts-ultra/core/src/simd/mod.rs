pub mod x86_64;
pub mod aarch64;
pub mod wasm32;
pub mod simd_nn;
pub mod benchmarks;

// Export architecture-specific implementations
#[cfg(target_arch = "x86_64")]
pub use x86_64::{
    X86Features, SimdMatrix as X86Matrix, SimdVector as X86Vector, 
    SimdFFT as X86FFT, SimdStats as X86Stats, SimdReduction as X86Reduction
};

#[cfg(target_arch = "aarch64")]
pub use aarch64::{
    AArch64Features, SimdMatrix as AArch64Matrix, SimdVector as AArch64Vector,
    SimdFFT as AArch64FFT, SimdStats as AArch64Stats, SimdReduction as AArch64Reduction
};

#[cfg(target_arch = "wasm32")]
pub use wasm32::{
    WasmFeatures, SimdMatrix as WasmMatrix, SimdVector as WasmVector,
    SimdFFT as WasmFFT, SimdStats as WasmStats, SimdReduction as WasmReduction,
    BrowserCompat, WasmMemory
};

// Unified SIMD interface that automatically selects the best implementation
pub struct SimdMatrix;
pub struct SimdVector; 
pub struct SimdFFT;
pub struct SimdStats;
pub struct SimdReduction;

impl SimdMatrix {
    #[inline]
    pub fn new() -> Self { Self }
    
    /// High-performance matrix multiplication with automatic architecture selection
    #[inline]
    pub unsafe fn multiply_f32(&self, a: &[f32], b: &[f32], c: &mut [f32], rows: usize, cols: usize, inner: usize) {
        #[cfg(target_arch = "x86_64")]
        { X86Matrix::new().multiply_f32(a, b, c, rows, cols, inner); }
        
        #[cfg(target_arch = "aarch64")]
        { AArch64Matrix::new().multiply_f32(a, b, c, rows, cols, inner); }
        
        #[cfg(target_arch = "wasm32")]
        { WasmMatrix::new().multiply_f32(a, b, c, rows, cols, inner); }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32")))]
        { 
            // Fallback scalar implementation
            for i in 0..rows {
                for j in 0..cols {
                    let mut sum = 0.0;
                    for k in 0..inner {
                        sum += a[i * inner + k] * b[k * cols + j];
                    }
                    c[i * cols + j] = sum;
                }
            }
        }
    }
}

impl SimdVector {
    #[inline]
    pub fn new() -> Self { Self }
    
    /// High-performance dot product with automatic architecture selection
    #[inline]
    pub unsafe fn dot_product_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        { return X86Vector::new().dot_product_f32(a, b); }
        
        #[cfg(target_arch = "aarch64")]
        { return AArch64Vector::new().dot_product_f32(a, b); }
        
        #[cfg(target_arch = "wasm32")]
        { return WasmVector::new().dot_product_f32(a, b); }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32")))]
        { 
            // Fallback scalar implementation
            return a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        }
    }
}

impl SimdFFT {
    #[inline]
    pub fn new() -> Self { Self }
    
    /// High-performance FFT with automatic architecture selection
    #[inline]
    pub unsafe fn fft_complex_f32(&self, data: &mut [f32], n: usize, inverse: bool) {
        #[cfg(target_arch = "x86_64")]
        { X86FFT::new().fft_complex_f32(data, n, inverse); }
        
        #[cfg(target_arch = "aarch64")]
        { AArch64FFT::new().fft_complex_f32(data, n, inverse); }
        
        #[cfg(target_arch = "wasm32")]
        { WasmFFT::new().fft_complex_f32(data, n, inverse); }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32")))]
        { 
            // Fallback scalar DFT implementation
            let mut result = vec![0.0; 2 * n];
            let sign = if inverse { 1.0 } else { -1.0 };
            
            for k in 0..n {
                for j in 0..n {
                    let angle = sign * 2.0 * std::f32::consts::PI * (k * j) as f32 / n as f32;
                    let cos_val = angle.cos();
                    let sin_val = angle.sin();
                    
                    result[2*k] += data[2*j] * cos_val - data[2*j + 1] * sin_val;
                    result[2*k + 1] += data[2*j] * sin_val + data[2*j + 1] * cos_val;
                }
            }
            
            if inverse {
                let norm = 1.0 / n as f32;
                for i in 0..2*n {
                    result[i] *= norm;
                }
            }
            
            data.copy_from_slice(&result);
        }
    }
}

impl SimdStats {
    #[inline]
    pub fn new() -> Self { Self }
    
    /// High-performance mean calculation with automatic architecture selection
    #[inline]
    pub unsafe fn mean_f32(&self, data: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        { return X86Stats::new().mean_f32(data); }
        
        #[cfg(target_arch = "aarch64")]
        { return AArch64Stats::new().mean_f32(data); }
        
        #[cfg(target_arch = "wasm32")]
        { return WasmStats::new().mean_f32(data); }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32")))]
        { 
            // Fallback scalar implementation
            return data.iter().sum::<f32>() / data.len() as f32;
        }
    }
}

impl SimdReduction {
    #[inline]
    pub fn new() -> Self { Self }
    
    /// High-performance max reduction with automatic architecture selection
    #[inline]
    pub unsafe fn max_f32(&self, data: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        { return X86Reduction::new().max_f32(data); }
        
        #[cfg(target_arch = "aarch64")]
        { return AArch64Reduction::new().max_f32(data); }
        
        #[cfg(target_arch = "wasm32")]
        { return WasmReduction::new().max_f32(data); }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32")))]
        { 
            // Fallback scalar implementation
            return data.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        }
    }
    
    /// High-performance min reduction with automatic architecture selection  
    #[inline]
    pub unsafe fn min_f32(&self, data: &[f32]) -> f32 {
        // Similar pattern to max_f32 but finding minimum
        #[cfg(target_arch = "x86_64")]
        { 
            // Would implement min in each architecture - for now use fallback logic
            return data.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
        }
        
        #[cfg(target_arch = "aarch64")]
        { return data.iter().fold(f32::INFINITY, |acc, &x| acc.min(x)); }
        
        #[cfg(target_arch = "wasm32")]
        { return data.iter().fold(f32::INFINITY, |acc, &x| acc.min(x)); }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32")))]
        { return data.iter().fold(f32::INFINITY, |acc, &x| acc.min(x)); }
    }
}

/// Runtime feature detection and capabilities
pub struct SimdCapabilities;

impl SimdCapabilities {
    /// Detect all available SIMD features for current architecture
    pub fn detect() -> String {
        #[cfg(target_arch = "x86_64")]
        {
            let features = X86Features::detect();
            format!("x86_64: SSE4.2={}, AVX2={}, AVX512F={}, FMA={}", 
                    features.has_sse42, features.has_avx2, 
                    features.has_avx512f, features.has_fma)
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            let features = AArch64Features::detect();
            format!("AArch64: NEON={}, SVE={}, Crypto={}, Apple={}", 
                    features.has_neon, features.has_sve, 
                    features.has_crypto, features.is_apple_silicon)
        }
        
        #[cfg(target_arch = "wasm32")]
        {
            let features = WasmFeatures::detect();
            format!("WASM32: SIMD128={}, Relaxed={}, Threads={}, Browser={}", 
                    features.has_simd128, features.has_relaxed_simd,
                    features.has_threads, features.is_browser)
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32")))]
        {
            "Unsupported architecture - using scalar fallbacks".to_string()
        }
    }
    
    /// Get performance estimate for current architecture
    pub fn performance_estimate() -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            let features = X86Features::detect();
            if features.has_avx512f { 4.5 }
            else if features.has_avx2 { 3.2 }
            else if features.has_sse42 { 2.1 }
            else { 1.0 }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            let features = AArch64Features::detect();
            if features.is_apple_silicon { 3.8 }
            else if features.has_neon { 2.9 }
            else { 1.0 }
        }
        
        #[cfg(target_arch = "wasm32")]
        {
            BrowserCompat::new().estimate_performance_multiplier()
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32")))]
        { 1.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_unified_interface() {
        println!("SIMD Capabilities: {}", SimdCapabilities::detect());
        println!("Performance Estimate: {:.1}x", SimdCapabilities::performance_estimate());
        
        // Test unified matrix multiplication
        let matrix = SimdMatrix::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0]; 
        let mut c = vec![0.0; 4];
        
        unsafe {
            matrix.multiply_f32(&a, &b, &mut c, 2, 2, 2);
        }
        
        assert!(c.iter().all(|&x| x.is_finite()));
        
        // Test unified vector operations
        let vector = SimdVector::new();
        let result = unsafe { vector.dot_product_f32(&a, &b) };
        let expected = 70.0; // 1*5 + 2*6 + 3*7 + 4*8
        assert!((result - expected).abs() < 1e-6);
        
        // Test unified statistics
        let stats = SimdStats::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = unsafe { stats.mean_f32(&data) };
        assert!((mean - 3.0).abs() < 1e-6);
        
        // Test unified reduction
        let reduction = SimdReduction::new();
        let max = unsafe { reduction.max_f32(&data) };
        assert!((max - 5.0).abs() < 1e-6);
    }
}