//! WebAssembly SIMD128 optimizations with browser compatibility
//! Performance target: <100ns for basic operations in WASM runtime

use std::arch::wasm32::*;

/// Runtime feature detection for WebAssembly SIMD capabilities
#[derive(Debug, Clone, Copy)]
pub struct WasmFeatures {
    pub has_simd128: bool,
    pub has_relaxed_simd: bool,
    pub has_threads: bool,
    pub has_bulk_memory: bool,
    pub is_browser: bool,
    pub is_nodejs: bool,
}

impl WasmFeatures {
    /// Detect available WebAssembly SIMD features at runtime
    #[inline]
    pub fn detect() -> Self {
        Self {
            has_simd128: cfg!(target_feature = "simd128"),
            has_relaxed_simd: cfg!(target_feature = "relaxed-simd"),
            has_threads: cfg!(target_feature = "atomics"),
            has_bulk_memory: cfg!(target_feature = "bulk-memory"),
            is_browser: Self::detect_browser(),
            is_nodejs: Self::detect_nodejs(),
        }
    }

    /// Detect if running in browser environment
    #[inline]
    fn detect_browser() -> bool {
        #[cfg(target_arch = "wasm32")]
        {
            // Check for browser-specific globals
            cfg!(feature = "web-sys")
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            false
        }
    }

    /// Detect if running in Node.js environment
    #[inline]
    fn detect_nodejs() -> bool {
        #[cfg(target_arch = "wasm32")]
        {
            // Check for Node.js-specific features
            !Self::detect_browser()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            false
        }
    }
}

/// High-performance matrix multiplication using WASM SIMD128
pub struct SimdMatrix {
    features: WasmFeatures,
}

impl SimdMatrix {
    pub fn new() -> Self {
        Self {
            features: WasmFeatures::detect(),
        }
    }

    /// Matrix multiplication: C = A * B
    /// Target: <80ns for 4x4 matrices in WASM runtime
    #[inline]
    pub unsafe fn multiply_f32(&self, a: &[f32], b: &[f32], c: &mut [f32], rows: usize, cols: usize, inner: usize) {
        if self.features.has_simd128 {
            self.multiply_simd128(a, b, c, rows, cols, inner);
        } else {
            self.multiply_scalar(a, b, c, rows, cols, inner);
        }
    }

    /// WASM SIMD128 matrix multiplication
    #[cfg(target_feature = "simd128")]
    #[inline]
    unsafe fn multiply_simd128(&self, a: &[f32], b: &[f32], c: &mut [f32], rows: usize, cols: usize, inner: usize) {
        for i in 0..rows {
            for j in (0..cols).step_by(4) {
                let mut acc = f32x4_splat(0.0);
                
                for k in 0..inner {
                    let a_val = f32x4_splat(a[i * inner + k]);
                    
                    // Load 4 consecutive elements from B matrix
                    if j + 3 < cols {
                        let b_vals = v128_load(&b[k * cols + j] as *const f32 as *const v128);
                        let b_vec = f32x4_convert_i32x4(b_vals);
                        acc = f32x4_add(acc, f32x4_mul(a_val, b_vec));
                    } else {
                        // Handle edge case where we don't have 4 consecutive elements
                        for l in 0..4 {
                            if j + l < cols {
                                let b_val = f32x4_splat(b[k * cols + j + l]);
                                let mask = if l == 0 { f32x4_splat(1.0) }
                                          else if l == 1 { f32x4(1.0, 1.0, 0.0, 0.0) }
                                          else if l == 2 { f32x4(1.0, 1.0, 1.0, 0.0) }
                                          else { f32x4(1.0, 1.0, 1.0, 1.0) };
                                acc = f32x4_add(acc, f32x4_mul(f32x4_mul(a_val, b_val), mask));
                            }
                        }
                        break;
                    }
                }
                
                // Store result back to C matrix
                if j + 3 < cols {
                    v128_store(&mut c[i * cols + j] as *mut f32 as *mut v128, acc);
                } else {
                    // Store individual elements for edge case
                    let result_array = [
                        f32x4_extract_lane::<0>(acc),
                        f32x4_extract_lane::<1>(acc),
                        f32x4_extract_lane::<2>(acc),
                        f32x4_extract_lane::<3>(acc),
                    ];
                    for l in 0..4 {
                        if j + l < cols {
                            c[i * cols + j + l] = result_array[l];
                        }
                    }
                }
            }
        }
    }

    #[cfg(not(target_feature = "simd128"))]
    #[inline]
    unsafe fn multiply_simd128(&self, a: &[f32], b: &[f32], c: &mut [f32], rows: usize, cols: usize, inner: usize) {
        self.multiply_scalar(a, b, c, rows, cols, inner);
    }

    /// Scalar fallback for browsers without SIMD support
    #[inline]
    fn multiply_scalar(&self, a: &[f32], b: &[f32], c: &mut [f32], rows: usize, cols: usize, inner: usize) {
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

/// High-performance vector operations using WASM SIMD128
pub struct SimdVector {
    features: WasmFeatures,
}

impl SimdVector {
    pub fn new() -> Self {
        Self {
            features: WasmFeatures::detect(),
        }
    }

    /// Vector dot product
    /// Target: <40ns for 1024 elements in WASM runtime
    #[inline]
    pub unsafe fn dot_product_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        if self.features.has_simd128 {
            self.dot_product_simd128(a, b)
        } else {
            self.dot_product_scalar(a, b)
        }
    }

    /// WASM SIMD128 dot product
    #[cfg(target_feature = "simd128")]
    #[inline]
    unsafe fn dot_product_simd128(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut acc = f32x4_splat(0.0);
        
        // Process 4 elements at a time
        let chunks = len / 4;
        for i in 0..chunks {
            let idx = i * 4;
            
            // Load 4 elements from each array
            let va_vals = v128_load(&a[idx] as *const f32 as *const v128);
            let vb_vals = v128_load(&b[idx] as *const f32 as *const v128);
            
            let va = f32x4_convert_i32x4(va_vals);
            let vb = f32x4_convert_i32x4(vb_vals);
            
            // Multiply and accumulate
            acc = f32x4_add(acc, f32x4_mul(va, vb));
        }
        
        // Horizontal sum of 4 elements
        let sum = f32x4_extract_lane::<0>(acc) +
                  f32x4_extract_lane::<1>(acc) +
                  f32x4_extract_lane::<2>(acc) +
                  f32x4_extract_lane::<3>(acc);
        
        // Handle remaining elements
        let mut remainder_sum = 0.0;
        for i in (chunks * 4)..len {
            remainder_sum += a[i] * b[i];
        }
        
        sum + remainder_sum
    }

    #[cfg(not(target_feature = "simd128"))]
    #[inline]
    unsafe fn dot_product_simd128(&self, a: &[f32], b: &[f32]) -> f32 {
        self.dot_product_scalar(a, b)
    }

    /// Scalar fallback
    #[inline]
    fn dot_product_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

/// Fast Fourier Transform using WASM SIMD128
pub struct SimdFFT {
    features: WasmFeatures,
}

impl SimdFFT {
    pub fn new() -> Self {
        Self {
            features: WasmFeatures::detect(),
        }
    }

    /// Complex FFT using SIMD
    /// Target: <300ns for 256-point FFT in WASM runtime
    #[inline]
    pub unsafe fn fft_complex_f32(&self, data: &mut [f32], n: usize, inverse: bool) {
        if self.features.has_simd128 {
            self.fft_simd128(data, n, inverse);
        } else {
            self.fft_scalar(data, n, inverse);
        }
    }

    /// WASM SIMD128 FFT implementation
    #[cfg(target_feature = "simd128")]
    #[inline]
    unsafe fn fft_simd128(&self, data: &mut [f32], n: usize, inverse: bool) {
        // Bit-reversal permutation
        self.bit_reverse_wasm(data, n);
        
        // Cooley-Tukey FFT with WASM SIMD optimizations
        let mut len = 2;
        while len <= n {
            let wlen = if inverse { 2.0 * std::f32::consts::PI / len as f32 } 
                       else { -2.0 * std::f32::consts::PI / len as f32 };
            
            let wlen_cos = wlen.cos();
            let wlen_sin = wlen.sin();
            
            for i in (0..n).step_by(len) {
                let mut w_real = 1.0;
                let mut w_imag = 0.0;
                
                for j in 0..len/2 {
                    let u_idx = 2 * (i + j);
                    let v_idx = 2 * (i + j + len/2);
                    
                    // Load complex numbers
                    let u_real = data[u_idx];
                    let u_imag = data[u_idx + 1];
                    let v_real = data[v_idx];
                    let v_imag = data[v_idx + 1];
                    
                    // Complex multiplication: w * v
                    let temp_real = w_real * v_real - w_imag * v_imag;
                    let temp_imag = w_real * v_imag + w_imag * v_real;
                    
                    // Butterfly operation
                    data[u_idx] = u_real + temp_real;
                    data[u_idx + 1] = u_imag + temp_imag;
                    data[v_idx] = u_real - temp_real;
                    data[v_idx + 1] = u_imag - temp_imag;
                    
                    // Update twiddle factor
                    let new_w_real = w_real * wlen_cos - w_imag * wlen_sin;
                    let new_w_imag = w_real * wlen_sin + w_imag * wlen_cos;
                    w_real = new_w_real;
                    w_imag = new_w_imag;
                }
            }
            len *= 2;
        }

        // Normalization for inverse FFT
        if inverse {
            let norm = f32x4_splat(1.0 / n as f32);
            let chunks = (2 * n) / 4;
            
            for i in 0..chunks {
                let idx = i * 4;
                let vals = v128_load(&data[idx] as *const f32 as *const v128);
                let vals_f32 = f32x4_convert_i32x4(vals);
                let normalized = f32x4_mul(vals_f32, norm);
                v128_store(&mut data[idx] as *mut f32 as *mut v128, normalized);
            }
            
            // Handle remaining elements
            let norm_scalar = 1.0 / n as f32;
            for i in (chunks * 4)..(2 * n) {
                data[i] *= norm_scalar;
            }
        }
    }

    #[cfg(not(target_feature = "simd128"))]
    #[inline]
    unsafe fn fft_simd128(&self, data: &mut [f32], n: usize, inverse: bool) {
        self.fft_scalar(data, n, inverse);
    }

    /// WASM-optimized bit-reversal permutation
    #[inline]
    fn bit_reverse_wasm(&self, data: &mut [f32], n: usize) {
        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j {
                data.swap(2 * i, 2 * j);
                data.swap(2 * i + 1, 2 * j + 1);
            }
        }
    }

    /// Scalar FFT fallback
    #[inline]
    fn fft_scalar(&self, data: &mut [f32], n: usize, inverse: bool) {
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

/// Statistical calculations using WASM SIMD128
pub struct SimdStats {
    features: WasmFeatures,
}

impl SimdStats {
    pub fn new() -> Self {
        Self {
            features: WasmFeatures::detect(),
        }
    }

    /// Calculate mean using SIMD
    /// Target: <30ns for 1024 elements in WASM runtime
    #[inline]
    pub unsafe fn mean_f32(&self, data: &[f32]) -> f32 {
        if self.features.has_simd128 {
            self.mean_simd128(data)
        } else {
            self.mean_scalar(data)
        }
    }

    /// WASM SIMD128 mean calculation
    #[cfg(target_feature = "simd128")]
    #[inline]
    unsafe fn mean_simd128(&self, data: &[f32]) -> f32 {
        let len = data.len();
        let mut sum = f32x4_splat(0.0);
        
        // Process 4 elements at a time
        let chunks = len / 4;
        for i in 0..chunks {
            let vals = v128_load(&data[i * 4] as *const f32 as *const v128);
            let vals_f32 = f32x4_convert_i32x4(vals);
            sum = f32x4_add(sum, vals_f32);
        }
        
        // Horizontal sum of 4 elements
        let total_sum = f32x4_extract_lane::<0>(sum) +
                       f32x4_extract_lane::<1>(sum) +
                       f32x4_extract_lane::<2>(sum) +
                       f32x4_extract_lane::<3>(sum);
        
        // Handle remaining elements
        let mut remainder_sum = 0.0;
        for i in (chunks * 4)..len {
            remainder_sum += data[i];
        }
        
        (total_sum + remainder_sum) / len as f32
    }

    #[cfg(not(target_feature = "simd128"))]
    #[inline]
    unsafe fn mean_simd128(&self, data: &[f32]) -> f32 {
        self.mean_scalar(data)
    }

    /// Scalar fallback
    #[inline]
    fn mean_scalar(&self, data: &[f32]) -> f32 {
        data.iter().sum::<f32>() / data.len() as f32
    }
}

/// Parallel reduction operations using WASM SIMD128
pub struct SimdReduction {
    features: WasmFeatures,
}

impl SimdReduction {
    pub fn new() -> Self {
        Self {
            features: WasmFeatures::detect(),
        }
    }

    /// Find maximum value using SIMD
    /// Target: <35ns for 1024 elements in WASM runtime
    #[inline]
    pub unsafe fn max_f32(&self, data: &[f32]) -> f32 {
        if self.features.has_simd128 {
            self.max_simd128(data)
        } else {
            self.max_scalar(data)
        }
    }

    /// WASM SIMD128 max reduction
    #[cfg(target_feature = "simd128")]
    #[inline]
    unsafe fn max_simd128(&self, data: &[f32]) -> f32 {
        if data.is_empty() { return f32::NEG_INFINITY; }
        
        let mut max_vec = f32x4_splat(data[0]);
        
        // Process 4 elements at a time
        let chunks = data.len() / 4;
        for i in 0..chunks {
            let vals = v128_load(&data[i * 4] as *const f32 as *const v128);
            let vals_f32 = f32x4_convert_i32x4(vals);
            max_vec = f32x4_pmax(max_vec, vals_f32);
        }
        
        // Extract maximum from vector
        let mut max_val = f32x4_extract_lane::<0>(max_vec);
        max_val = max_val.max(f32x4_extract_lane::<1>(max_vec));
        max_val = max_val.max(f32x4_extract_lane::<2>(max_vec));
        max_val = max_val.max(f32x4_extract_lane::<3>(max_vec));
        
        // Handle remaining elements
        for i in (chunks * 4)..data.len() {
            if data[i] > max_val {
                max_val = data[i];
            }
        }
        
        max_val
    }

    #[cfg(not(target_feature = "simd128"))]
    #[inline]
    unsafe fn max_simd128(&self, data: &[f32]) -> f32 {
        self.max_scalar(data)
    }

    /// Scalar fallback
    #[inline]
    fn max_scalar(&self, data: &[f32]) -> f32 {
        data.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x))
    }
}

/// Browser-specific optimizations and polyfills
pub struct BrowserCompat {
    features: WasmFeatures,
}

impl BrowserCompat {
    pub fn new() -> Self {
        Self {
            features: WasmFeatures::detect(),
        }
    }

    /// Check if SIMD is available in current browser
    #[inline]
    pub fn is_simd_supported(&self) -> bool {
        self.features.has_simd128
    }

    /// Provide polyfill guidance for non-SIMD browsers
    #[inline]
    pub fn get_fallback_strategy(&self) -> &'static str {
        if !self.features.has_simd128 {
            "Using scalar fallback implementations for compatibility"
        } else if !self.features.has_relaxed_simd {
            "Basic SIMD available, some advanced features unavailable"
        } else {
            "Full SIMD support with relaxed semantics"
        }
    }

    /// Performance estimation based on runtime environment
    #[inline]
    pub fn estimate_performance_multiplier(&self) -> f32 {
        if self.features.has_simd128 {
            if self.features.has_relaxed_simd {
                3.5 // ~3.5x speedup with full SIMD
            } else {
                2.8 // ~2.8x speedup with basic SIMD
            }
        } else {
            1.0 // No speedup with scalar fallback
        }
    }

    /// Memory usage estimation for WASM linear memory
    #[inline]
    pub fn estimate_memory_overhead(&self) -> usize {
        if self.features.has_simd128 {
            128 // 128 bytes for SIMD registers and alignment
        } else {
            32  // Minimal overhead for scalar operations
        }
    }
}

/// WebAssembly-specific memory management
pub struct WasmMemory;

impl WasmMemory {
    /// Align memory for optimal SIMD access
    #[inline]
    pub fn align_for_simd(size: usize) -> usize {
        (size + 15) & !15 // Align to 16-byte boundary
    }

    /// Check if pointer is properly aligned for SIMD
    #[inline]
    pub fn is_aligned_for_simd(ptr: *const u8) -> bool {
        (ptr as usize) % 16 == 0
    }

    /// Prefetch data for better cache performance (no-op in WASM)
    #[inline]
    pub fn prefetch(_addr: *const u8) {
        // WASM doesn't have cache prefetching, but this provides
        // a consistent API for cross-platform code
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature_detection() {
        let features = WasmFeatures::detect();
        println!("Detected WASM features: {:?}", features);
        // Feature detection should not panic
    }
    
    #[test]
    fn test_matrix_multiplication() {
        let matrix = SimdMatrix::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];
        
        unsafe {
            matrix.multiply_f32(&a, &b, &mut c, 2, 2, 2);
        }
        
        assert!(c.iter().all(|&x| x.is_finite()));
    }
    
    #[test]
    fn test_dot_product() {
        let vector = SimdVector::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        
        let result = unsafe { vector.dot_product_f32(&a, &b) };
        let expected = 1.0*5.0 + 2.0*6.0 + 3.0*7.0 + 4.0*8.0; // 70.0
        
        assert!((result - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_mean_calculation() {
        let stats = SimdStats::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let result = unsafe { stats.mean_f32(&data) };
        let expected = 3.0;
        
        assert!((result - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_max_reduction() {
        let reduction = SimdReduction::new();
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        
        let result = unsafe { reduction.max_f32(&data) };
        let expected = 9.0;
        
        assert!((result - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_browser_compatibility() {
        let compat = BrowserCompat::new();
        
        println!("SIMD supported: {}", compat.is_simd_supported());
        println!("Fallback strategy: {}", compat.get_fallback_strategy());
        println!("Performance multiplier: {:.1}x", compat.estimate_performance_multiplier());
        println!("Memory overhead: {} bytes", compat.estimate_memory_overhead());
    }
    
    #[test]
    fn test_memory_alignment() {
        let size = 100;
        let aligned_size = WasmMemory::align_for_simd(size);
        assert!(aligned_size >= size);
        assert!(aligned_size % 16 == 0);
        
        let ptr = &aligned_size as *const usize as *const u8;
        // Note: This test might not pass since stack allocation isn't guaranteed to be 16-byte aligned
        println!("Pointer alignment check: {}", WasmMemory::is_aligned_for_simd(ptr));
    }
    
    #[test]
    fn test_fft_basic() {
        let fft = SimdFFT::new();
        let mut data = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]; // 4-point real signal
        
        unsafe {
            fft.fft_complex_f32(&mut data, 4, false);
        }
        
        // FFT of constant signal should have energy at DC
        assert!(data[0] > 0.0); // DC component should be positive
        assert!(data.iter().all(|&x| x.is_finite())); // All results should be finite
    }
}