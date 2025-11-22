use crate::Result;
use crate::utils::{has_avx512, has_avx2, has_sse41, has_fma};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;
// Removed unstable simd imports - using only intrinsics

/// SIMD vectorization configuration and optimization
#[derive(Debug, Clone)]
pub struct VectorizationConfig {
    pub enable_avx512: bool,
    pub enable_avx2: bool,
    pub enable_sse4: bool,
    pub enable_fma: bool,
    pub enable_auto_vectorization: bool,
    pub vector_width: VectorWidth,
    pub alignment: usize,
    pub unroll_factor: usize,
}

#[derive(Debug, Clone)]
pub enum VectorWidth {
    Auto,
    SSE128,    // 4 floats
    AVX256,    // 8 floats
    AVX512,    // 16 floats
}

/// High-performance SIMD vectorization engine
pub struct VectorizationEngine {
    config: VectorizationConfig,
    capabilities: CpuCapabilities,
    kernel_cache: std::collections::HashMap<String, VectorKernel>,
    performance_monitor: VectorizationStats,
}

/// CPU capabilities detection
#[derive(Debug, Clone)]
pub struct CpuCapabilities {
    pub has_sse4_1: bool,
    pub has_sse4_2: bool,
    pub has_avx: bool,
    pub has_avx2: bool,
    pub has_avx512f: bool,
    pub has_avx512bw: bool,
    pub has_avx512vl: bool,
    pub has_fma: bool,
    pub has_bmi1: bool,
    pub has_bmi2: bool,
    pub cache_line_size: usize,
    pub l1_cache_size: usize,
    pub l2_cache_size: usize,
    pub l3_cache_size: usize,
}

/// Vectorized kernel implementations
#[derive(Debug, Clone)]
pub struct VectorKernel {
    name: String,
    vector_width: VectorWidth,
    kernel_type: KernelType,
    optimized_for: Vec<CpuFeature>,
    performance_profile: KernelPerformance,
}

#[derive(Debug, Clone)]
pub enum KernelType {
    MatrixMultiply,
    Convolution1D,
    Activation,
    BatchNorm,
    LayerNorm,
    Attention,
    Reduction,
    ElementWise,
}

#[derive(Debug, Clone)]
pub enum CpuFeature {
    SSE41,
    SSE42,
    AVX,
    AVX2,
    AVX512F,
    AVX512BW,
    AVX512VL,
    FMA,
    BMI1,
    BMI2,
}

#[derive(Debug, Clone)]
pub struct KernelPerformance {
    pub throughput_gflops: f64,
    pub latency_ns: u64,
    pub cache_efficiency: f64,
    pub vectorization_ratio: f64,
    pub instruction_count: u64,
}

/// Performance monitoring for vectorization
#[derive(Debug, Clone)]
pub struct VectorizationStats {
    pub kernels_compiled: usize,
    pub total_operations: u64,
    pub vectorized_operations: u64,
    pub scalar_fallbacks: u64,
    pub average_vector_utilization: f64,
    pub peak_throughput: f64,
    pub cache_hit_rate: f64,
}

impl Default for VectorizationConfig {
    fn default() -> Self {
        Self {
            enable_avx512: true,
            enable_avx2: true,
            enable_sse4: true,
            enable_fma: true,
            enable_auto_vectorization: true,
            vector_width: VectorWidth::Auto,
            alignment: 64, // 64-byte alignment for AVX-512
            unroll_factor: 4,
        }
    }
}

impl VectorizationEngine {
    /// Create new vectorization engine with capability detection
    pub fn new(config: VectorizationConfig) -> Result<Self> {
        let capabilities = Self::detect_cpu_capabilities();
        let optimal_config = Self::optimize_config(config, &capabilities);

        Ok(Self {
            config: optimal_config,
            capabilities,
            kernel_cache: std::collections::HashMap::new(),
            performance_monitor: VectorizationStats::new(),
        })
    }

    /// Detect CPU capabilities at runtime
    fn detect_cpu_capabilities() -> CpuCapabilities {
        // Use runtime CPU feature detection
        CpuCapabilities {
            has_sse4_1: has_sse41(),
            has_sse4_2: Self::detect_sse42(),
            has_avx: Self::detect_avx(),
            has_avx2: has_avx2(),
            has_avx512f: has_avx512(),
            has_avx512bw: Self::detect_avx512bw(),
            has_avx512vl: Self::detect_avx512vl(),
            has_fma: has_fma(),
            has_bmi1: Self::detect_bmi1(),
            has_bmi2: Self::detect_bmi2(),
            cache_line_size: 64,
            l1_cache_size: 32 * 1024,
            l2_cache_size: 256 * 1024,
            l3_cache_size: 8 * 1024 * 1024,
        }
    }

    #[inline]
    fn detect_sse42() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            is_x86_feature_detected!("sse4.2")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }

    #[inline]
    fn detect_avx() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            is_x86_feature_detected!("avx")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }

    #[inline]
    fn detect_avx512bw() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            is_x86_feature_detected!("avx512bw")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }

    #[inline]
    fn detect_avx512vl() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            is_x86_feature_detected!("avx512vl")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }

    #[inline]
    fn detect_bmi1() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            is_x86_feature_detected!("bmi1")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }

    #[inline]
    fn detect_bmi2() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            is_x86_feature_detected!("bmi2")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }

    /// Optimize configuration based on CPU capabilities
    fn optimize_config(mut config: VectorizationConfig, caps: &CpuCapabilities) -> VectorizationConfig {
        // Auto-select optimal vector width
        if config.vector_width == VectorWidth::Auto {
            config.vector_width = if caps.has_avx512f && config.enable_avx512 {
                VectorWidth::AVX512
            } else if caps.has_avx2 && config.enable_avx2 {
                VectorWidth::AVX256
            } else if caps.has_sse4_1 && config.enable_sse4 {
                VectorWidth::SSE128
            } else {
                VectorWidth::SSE128
            };
        }

        // Adjust alignment based on vector width
        config.alignment = match config.vector_width {
            VectorWidth::AVX512 => 64,
            VectorWidth::AVX256 => 32,
            VectorWidth::SSE128 => 16,
            VectorWidth::Auto => 64,
        };

        config
    }

    /// Vectorized matrix multiplication using SIMD
    pub fn vectorized_matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();

        if k != k2 {
            return Err(crate::Error::Other(
                "Matrix dimensions don't match".to_string()
            ));
        }

        match self.config.vector_width {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            VectorWidth::AVX512 if self.capabilities.has_avx512f => {
                unsafe { self.avx512_matmul(a, b) }
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            VectorWidth::AVX256 if self.capabilities.has_avx2 => {
                unsafe { self.avx2_matmul(a, b) }
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            VectorWidth::SSE128 if self.capabilities.has_sse4_1 => {
                unsafe { self.sse_matmul(a, b) }
            }
            _ => self.scalar_matmul(a, b),
        }
    }

    /// AVX-512 optimized matrix multiplication
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
        let (m, k) = a.dim();
        let (_, n) = b.dim();
        let mut result = Array2::zeros((m, n));

        // Process 16 elements at a time with AVX-512
        for i in 0..m {
            for j in (0..n).step_by(16) {
                let remaining = (n - j).min(16);
                let mut sum = _mm512_setzero_ps();

                for l in 0..k {
                    let a_val = _mm512_set1_ps(a[[i, l]]);
                    
                    if remaining == 16 {
                        let b_row = _mm512_loadu_ps(b.as_ptr().add(l * n + j));
                        sum = _mm512_fmadd_ps(a_val, b_row, sum);
                    } else {
                        // Handle partial vectors
                        let mask = (1u16 << remaining) - 1;
                        let b_row = _mm512_maskz_loadu_ps(mask, b.as_ptr().add(l * n + j));
                        sum = _mm512_fmadd_ps(a_val, b_row, sum);
                    }
                }

                if remaining == 16 {
                    _mm512_storeu_ps(result.as_mut_ptr().add(i * n + j), sum);
                } else {
                    let mask = (1u16 << remaining) - 1;
                    _mm512_mask_storeu_ps(result.as_mut_ptr().add(i * n + j), mask, sum);
                }
            }
        }

        Ok(result)
    }

    /// AVX2 optimized matrix multiplication
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
        let (m, k) = a.dim();
        let (_, n) = b.dim();
        let mut result = Array2::zeros((m, n));

        // Process 8 elements at a time with AVX2
        for i in 0..m {
            for j in (0..n).step_by(8) {
                let remaining = (n - j).min(8);
                let mut sum = _mm256_setzero_ps();

                for l in 0..k {
                    let a_val = _mm256_set1_ps(a[[i, l]]);
                    
                    if remaining == 8 {
                        let b_row = _mm256_loadu_ps(b.as_ptr().add(l * n + j));
                        sum = _mm256_fmadd_ps(a_val, b_row, sum);
                    } else {
                        // Handle partial vectors with masking
                        let mut b_vals = [0.0f32; 8];
                        for idx in 0..remaining {
                            b_vals[idx] = b[[l, j + idx]];
                        }
                        let b_row = _mm256_loadu_ps(b_vals.as_ptr());
                        sum = _mm256_fmadd_ps(a_val, b_row, sum);
                    }
                }

                // Store results
                let mut result_vals = [0.0f32; 8];
                _mm256_storeu_ps(result_vals.as_mut_ptr(), sum);
                for idx in 0..remaining {
                    result[[i, j + idx]] = result_vals[idx];
                }
            }
        }

        Ok(result)
    }

    /// SSE4.1 optimized matrix multiplication
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse4.1")]
    unsafe fn sse_matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
        let (m, k) = a.dim();
        let (_, n) = b.dim();
        let mut result = Array2::zeros((m, n));

        // Process 4 elements at a time with SSE
        for i in 0..m {
            for j in (0..n).step_by(4) {
                let remaining = (n - j).min(4);
                let mut sum = _mm_setzero_ps();

                for l in 0..k {
                    let a_val = _mm_set1_ps(a[[i, l]]);
                    
                    if remaining == 4 {
                        let b_row = _mm_loadu_ps(b.as_ptr().add(l * n + j));
                        sum = _mm_add_ps(sum, _mm_mul_ps(a_val, b_row));
                    } else {
                        // Handle partial vectors
                        let mut b_vals = [0.0f32; 4];
                        for idx in 0..remaining {
                            b_vals[idx] = b[[l, j + idx]];
                        }
                        let b_row = _mm_loadu_ps(b_vals.as_ptr());
                        sum = _mm_add_ps(sum, _mm_mul_ps(a_val, b_row));
                    }
                }

                // Store results
                let mut result_vals = [0.0f32; 4];
                _mm_storeu_ps(result_vals.as_mut_ptr(), sum);
                for idx in 0..remaining {
                    result[[i, j + idx]] = result_vals[idx];
                }
            }
        }

        Ok(result)
    }

    // Fallback implementations for non-x86 architectures
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn avx512_matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
        self.scalar_matmul(a, b)
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn avx2_matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
        self.scalar_matmul(a, b)
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn sse_matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
        self.scalar_matmul(a, b)
    }

    /// Scalar fallback for unsupported architectures
    fn scalar_matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
        Ok(a.dot(b))
    }

    /// Vectorized 1D convolution
    pub fn vectorized_conv1d(
        &self,
        input: &Array3<f32>,
        kernel: &Array3<f32>,
    ) -> Result<Array3<f32>> {
        let (batch_size, seq_len, in_channels) = input.dim();
        let (out_channels, kernel_size, kernel_in_channels) = kernel.dim();

        if in_channels != kernel_in_channels {
            return Err(crate::Error::Other(
                "Channel dimensions don't match".to_string()
            ));
        }

        let output_len = seq_len - kernel_size + 1;
        let mut output = Array3::zeros((batch_size, output_len, out_channels));

        match self.config.vector_width {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            VectorWidth::AVX512 if self.capabilities.has_avx512f => {
                unsafe { self.avx512_conv1d(input, kernel, &mut output) }?;
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            VectorWidth::AVX256 if self.capabilities.has_avx2 => {
                unsafe { self.avx2_conv1d(input, kernel, &mut output) }?;
            }
            _ => {
                self.scalar_conv1d(input, kernel, &mut output)?;
            }
        }

        Ok(output)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_conv1d(
        &self,
        input: &Array3<f32>,
        kernel: &Array3<f32>,
        output: &mut Array3<f32>,
    ) -> Result<()> {
        let (batch_size, seq_len, in_channels) = input.dim();
        let (out_channels, kernel_size, _) = kernel.dim();
        let output_len = seq_len - kernel_size + 1;

        for batch in 0..batch_size {
            for out_ch in 0..out_channels {
                for pos in 0..output_len {
                    let mut sum = _mm512_setzero_ps();
                    
                    // Vectorize over input channels (16 at a time)
                    for in_ch in (0..in_channels).step_by(16) {
                        let remaining = (in_channels - in_ch).min(16);
                        
                        for k in 0..kernel_size {
                            if remaining == 16 {
                                let input_vals = _mm512_loadu_ps(
                                    input.as_ptr().add(
                                        batch * seq_len * in_channels +
                                        (pos + k) * in_channels + in_ch
                                    )
                                );
                                let kernel_vals = _mm512_loadu_ps(
                                    kernel.as_ptr().add(
                                        out_ch * kernel_size * in_channels +
                                        k * in_channels + in_ch
                                    )
                                );
                                sum = _mm512_fmadd_ps(input_vals, kernel_vals, sum);
                            } else {
                                // Handle partial vectors
                                let mask = (1u16 << remaining) - 1;
                                let input_vals = _mm512_maskz_loadu_ps(
                                    mask,
                                    input.as_ptr().add(
                                        batch * seq_len * in_channels +
                                        (pos + k) * in_channels + in_ch
                                    )
                                );
                                let kernel_vals = _mm512_maskz_loadu_ps(
                                    mask,
                                    kernel.as_ptr().add(
                                        out_ch * kernel_size * in_channels +
                                        k * in_channels + in_ch
                                    )
                                );
                                sum = _mm512_fmadd_ps(input_vals, kernel_vals, sum);
                            }
                        }
                    }
                    
                    // Horizontal sum of the vector
                    let result = self.horizontal_sum_avx512(sum);
                    output[[batch, pos, out_ch]] = result;
                }
            }
        }

        Ok(())
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_conv1d(
        &self,
        input: &Array3<f32>,
        kernel: &Array3<f32>,
        output: &mut Array3<f32>,
    ) -> Result<()> {
        let (batch_size, seq_len, in_channels) = input.dim();
        let (out_channels, kernel_size, _) = kernel.dim();
        let output_len = seq_len - kernel_size + 1;

        for batch in 0..batch_size {
            for out_ch in 0..out_channels {
                for pos in 0..output_len {
                    let mut sum = _mm256_setzero_ps();
                    
                    // Vectorize over input channels (8 at a time)
                    for in_ch in (0..in_channels).step_by(8) {
                        let remaining = (in_channels - in_ch).min(8);
                        
                        for k in 0..kernel_size {
                            if remaining == 8 {
                                let input_vals = _mm256_loadu_ps(
                                    input.as_ptr().add(
                                        batch * seq_len * in_channels +
                                        (pos + k) * in_channels + in_ch
                                    )
                                );
                                let kernel_vals = _mm256_loadu_ps(
                                    kernel.as_ptr().add(
                                        out_ch * kernel_size * in_channels +
                                        k * in_channels + in_ch
                                    )
                                );
                                sum = _mm256_fmadd_ps(input_vals, kernel_vals, sum);
                            }
                        }
                    }
                    
                    // Horizontal sum of the vector
                    let result = self.horizontal_sum_avx2(sum);
                    output[[batch, pos, out_ch]] = result;
                }
            }
        }

        Ok(())
    }

    // Fallback implementations for non-x86 architectures
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn avx512_conv1d(
        &self,
        input: &Array3<f32>,
        kernel: &Array3<f32>,
        output: &mut Array3<f32>,
    ) -> Result<()> {
        self.scalar_conv1d(input, kernel, output)
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn avx2_conv1d(
        &self,
        input: &Array3<f32>,
        kernel: &Array3<f32>,
        output: &mut Array3<f32>,
    ) -> Result<()> {
        self.scalar_conv1d(input, kernel, output)
    }

    fn scalar_conv1d(
        &self,
        input: &Array3<f32>,
        kernel: &Array3<f32>,
        output: &mut Array3<f32>,
    ) -> Result<()> {
        let (batch_size, seq_len, in_channels) = input.dim();
        let (out_channels, kernel_size, _) = kernel.dim();
        let output_len = seq_len - kernel_size + 1;

        for batch in 0..batch_size {
            for out_ch in 0..out_channels {
                for pos in 0..output_len {
                    let mut sum = 0.0;
                    
                    for k in 0..kernel_size {
                        for in_ch in 0..in_channels {
                            sum += input[[batch, pos + k, in_ch]] * kernel[[out_ch, k, in_ch]];
                        }
                    }
                    
                    output[[batch, pos, out_ch]] = sum;
                }
            }
        }

        Ok(())
    }

    /// Vectorized activation functions
    pub fn vectorized_relu(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let mut output = Array2::zeros(input.raw_dim());
        
        match self.config.vector_width {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            VectorWidth::AVX512 if self.capabilities.has_avx512f => {
                unsafe { self.avx512_relu(input, &mut output) }?;
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            VectorWidth::AVX256 if self.capabilities.has_avx2 => {
                unsafe { self.avx2_relu(input, &mut output) }?;
            }
            _ => {
                output.zip_mut_with(input, |out, &inp| *out = inp.max(0.0));
            }
        }

        Ok(output)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_relu(&self, input: &Array2<f32>, output: &mut Array2<f32>) -> Result<()> {
        let total_elements = input.len();
        let zeros = _mm512_setzero_ps();

        for i in (0..total_elements).step_by(16) {
            let remaining = (total_elements - i).min(16);
            
            if remaining == 16 {
                let vals = _mm512_loadu_ps(input.as_ptr().add(i));
                let result = _mm512_max_ps(vals, zeros);
                _mm512_storeu_ps(output.as_mut_ptr().add(i), result);
            } else {
                let mask = (1u16 << remaining) - 1;
                let vals = _mm512_maskz_loadu_ps(mask, input.as_ptr().add(i));
                let result = _mm512_max_ps(vals, zeros);
                _mm512_mask_storeu_ps(output.as_mut_ptr().add(i), mask, result);
            }
        }

        Ok(())
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_relu(&self, input: &Array2<f32>, output: &mut Array2<f32>) -> Result<()> {
        let total_elements = input.len();
        let zeros = _mm256_setzero_ps();

        for i in (0..total_elements).step_by(8) {
            let remaining = (total_elements - i).min(8);
            
            if remaining == 8 {
                let vals = _mm256_loadu_ps(input.as_ptr().add(i));
                let result = _mm256_max_ps(vals, zeros);
                _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
            } else {
                // Handle partial vectors
                let mut input_vals = [0.0f32; 8];
                for idx in 0..remaining {
                    input_vals[idx] = *input.as_ptr().add(i + idx);
                }
                let vals = _mm256_loadu_ps(input_vals.as_ptr());
                let result = _mm256_max_ps(vals, zeros);
                
                let mut result_vals = [0.0f32; 8];
                _mm256_storeu_ps(result_vals.as_mut_ptr(), result);
                for idx in 0..remaining {
                    *output.as_mut_ptr().add(i + idx) = result_vals[idx];
                }
            }
        }

        Ok(())
    }

    // Fallback implementations for non-x86 architectures
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn avx512_relu(&self, input: &Array2<f32>, output: &mut Array2<f32>) -> Result<()> {
        output.zip_mut_with(input, |out, &inp| *out = inp.max(0.0));
        Ok(())
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn avx2_relu(&self, input: &Array2<f32>, output: &mut Array2<f32>) -> Result<()> {
        output.zip_mut_with(input, |out, &inp| *out = inp.max(0.0));
        Ok(())
    }

    /// Vectorized GELU activation
    pub fn vectorized_gelu(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let mut output = Array2::zeros(input.raw_dim());
        
        match self.config.vector_width {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            VectorWidth::AVX512 if self.capabilities.has_avx512f => {
                unsafe { self.avx512_gelu(input, &mut output) }?;
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            VectorWidth::AVX256 if self.capabilities.has_avx2 => {
                unsafe { self.avx2_gelu(input, &mut output) }?;
            }
            _ => {
                output.zip_mut_with(input, |out, &inp| {
                    *out = 0.5 * inp * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * 
                            (inp + 0.044715 * inp.powi(3))).tanh());
                });
            }
        }

        Ok(output)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_gelu(&self, input: &Array2<f32>, output: &mut Array2<f32>) -> Result<()> {
        let total_elements = input.len();
        let half = _mm512_set1_ps(0.5);
        let one = _mm512_set1_ps(1.0);
        let coeff = _mm512_set1_ps(0.044715);
        let sqrt_2_pi = _mm512_set1_ps((2.0 / std::f32::consts::PI).sqrt());

        for i in (0..total_elements).step_by(16) {
            let remaining = (total_elements - i).min(16);
            
            if remaining == 16 {
                let x = _mm512_loadu_ps(input.as_ptr().add(i));
                
                // Calculate x^3
                let x2 = _mm512_mul_ps(x, x);
                let x3 = _mm512_mul_ps(x2, x);
                
                // Calculate tanh argument: sqrt(2/π) * (x + 0.044715 * x^3)
                let tanh_arg = _mm512_fmadd_ps(coeff, x3, x);
                let tanh_arg = _mm512_mul_ps(sqrt_2_pi, tanh_arg);
                
                // Approximate tanh using rational approximation
                let tanh_result = self.avx512_tanh_approx(tanh_arg);
                
                // Final GELU: 0.5 * x * (1 + tanh(...))
                let result = _mm512_add_ps(one, tanh_result);
                let result = _mm512_mul_ps(x, result);
                let result = _mm512_mul_ps(half, result);
                
                _mm512_storeu_ps(output.as_mut_ptr().add(i), result);
            }
        }

        Ok(())
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_gelu(&self, input: &Array2<f32>, output: &mut Array2<f32>) -> Result<()> {
        let total_elements = input.len();
        let half = _mm256_set1_ps(0.5);
        let one = _mm256_set1_ps(1.0);
        let coeff = _mm256_set1_ps(0.044715);
        let sqrt_2_pi = _mm256_set1_ps((2.0 / std::f32::consts::PI).sqrt());

        for i in (0..total_elements).step_by(8) {
            let remaining = (total_elements - i).min(8);
            
            if remaining == 8 {
                let x = _mm256_loadu_ps(input.as_ptr().add(i));
                
                // Calculate x^3
                let x2 = _mm256_mul_ps(x, x);
                let x3 = _mm256_mul_ps(x2, x);
                
                // Calculate tanh argument
                let tanh_arg = _mm256_fmadd_ps(coeff, x3, x);
                let tanh_arg = _mm256_mul_ps(sqrt_2_pi, tanh_arg);
                
                // Approximate tanh
                let tanh_result = self.avx2_tanh_approx(tanh_arg);
                
                // Final GELU calculation
                let result = _mm256_add_ps(one, tanh_result);
                let result = _mm256_mul_ps(x, result);
                let result = _mm256_mul_ps(half, result);
                
                _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
            }
        }

        Ok(())
    }

    // Fallback implementations for non-x86 architectures
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn avx512_gelu(&self, input: &Array2<f32>, output: &mut Array2<f32>) -> Result<()> {
        output.zip_mut_with(input, |out, &inp| {
            *out = 0.5 * inp * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * 
                    (inp + 0.044715 * inp.powi(3))).tanh());
        });
        Ok(())
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn avx2_gelu(&self, input: &Array2<f32>, output: &mut Array2<f32>) -> Result<()> {
        output.zip_mut_with(input, |out, &inp| {
            *out = 0.5 * inp * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * 
                    (inp + 0.044715 * inp.powi(3))).tanh());
        });
        Ok(())
    }

    /// Vectorized batch normalization
    pub fn vectorized_batch_norm(
        &self,
        input: &Array2<f32>,
        gamma: &Array1<f32>,
        beta: &Array1<f32>,
        mean: &Array1<f32>,
        variance: &Array1<f32>,
        epsilon: f32,
    ) -> Result<Array2<f32>> {
        let (batch_size, features) = input.dim();
        let mut output = Array2::zeros((batch_size, features));

        match self.config.vector_width {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            VectorWidth::AVX512 if self.capabilities.has_avx512f => {
                unsafe { self.avx512_batch_norm(input, gamma, beta, mean, variance, epsilon, &mut output) }?;
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            VectorWidth::AVX256 if self.capabilities.has_avx2 => {
                unsafe { self.avx2_batch_norm(input, gamma, beta, mean, variance, epsilon, &mut output) }?;
            }
            _ => {
                for i in 0..batch_size {
                    for j in 0..features {
                        let normalized = (input[[i, j]] - mean[j]) / (variance[j] + epsilon).sqrt();
                        output[[i, j]] = gamma[j] * normalized + beta[j];
                    }
                }
            }
        }

        Ok(output)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_batch_norm(
        &self,
        input: &Array2<f32>,
        gamma: &Array1<f32>,
        beta: &Array1<f32>,
        mean: &Array1<f32>,
        variance: &Array1<f32>,
        epsilon: f32,
        output: &mut Array2<f32>,
    ) -> Result<()> {
        let (batch_size, features) = input.dim();
        let eps = _mm512_set1_ps(epsilon);

        for i in 0..batch_size {
            for j in (0..features).step_by(16) {
                let remaining = (features - j).min(16);
                
                if remaining == 16 {
                    let x = _mm512_loadu_ps(input.as_ptr().add(i * features + j));
                    let m = _mm512_loadu_ps(mean.as_ptr().add(j));
                    let v = _mm512_loadu_ps(variance.as_ptr().add(j));
                    let g = _mm512_loadu_ps(gamma.as_ptr().add(j));
                    let b = _mm512_loadu_ps(beta.as_ptr().add(j));
                    
                    // (x - mean) / sqrt(var + eps)
                    let centered = _mm512_sub_ps(x, m);
                    let var_eps = _mm512_add_ps(v, eps);
                    let inv_std = _mm512_rsqrt14_ps(var_eps); // Fast reciprocal sqrt
                    let normalized = _mm512_mul_ps(centered, inv_std);
                    
                    // gamma * normalized + beta
                    let result = _mm512_fmadd_ps(g, normalized, b);
                    
                    _mm512_storeu_ps(output.as_mut_ptr().add(i * features + j), result);
                }
            }
        }

        Ok(())
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2_batch_norm(
        &self,
        input: &Array2<f32>,
        gamma: &Array1<f32>,
        beta: &Array1<f32>,
        mean: &Array1<f32>,
        variance: &Array1<f32>,
        epsilon: f32,
        output: &mut Array2<f32>,
    ) -> Result<()> {
        let (batch_size, features) = input.dim();
        let eps = _mm256_set1_ps(epsilon);

        for i in 0..batch_size {
            for j in (0..features).step_by(8) {
                let remaining = (features - j).min(8);
                
                if remaining == 8 {
                    let x = _mm256_loadu_ps(input.as_ptr().add(i * features + j));
                    let m = _mm256_loadu_ps(mean.as_ptr().add(j));
                    let v = _mm256_loadu_ps(variance.as_ptr().add(j));
                    let g = _mm256_loadu_ps(gamma.as_ptr().add(j));
                    let b = _mm256_loadu_ps(beta.as_ptr().add(j));
                    
                    // (x - mean) / sqrt(var + eps)
                    let centered = _mm256_sub_ps(x, m);
                    let var_eps = _mm256_add_ps(v, eps);
                    let inv_std = _mm256_rsqrt_ps(var_eps);
                    let normalized = _mm256_mul_ps(centered, inv_std);
                    
                    // gamma * normalized + beta
                    let result = _mm256_fmadd_ps(g, normalized, b);
                    
                    _mm256_storeu_ps(output.as_mut_ptr().add(i * features + j), result);
                }
            }
        }

        Ok(())
    }

    // Fallback implementations for non-x86 architectures
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn avx512_batch_norm(
        &self,
        input: &Array2<f32>,
        gamma: &Array1<f32>,
        beta: &Array1<f32>,
        mean: &Array1<f32>,
        variance: &Array1<f32>,
        epsilon: f32,
        output: &mut Array2<f32>,
    ) -> Result<()> {
        let (batch_size, features) = input.dim();
        for i in 0..batch_size {
            for j in 0..features {
                let normalized = (input[[i, j]] - mean[j]) / (variance[j] + epsilon).sqrt();
                output[[i, j]] = gamma[j] * normalized + beta[j];
            }
        }
        Ok(())
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn avx2_batch_norm(
        &self,
        input: &Array2<f32>,
        gamma: &Array1<f32>,
        beta: &Array1<f32>,
        mean: &Array1<f32>,
        variance: &Array1<f32>,
        epsilon: f32,
        output: &mut Array2<f32>,
    ) -> Result<()> {
        let (batch_size, features) = input.dim();
        for i in 0..batch_size {
            for j in 0..features {
                let normalized = (input[[i, j]] - mean[j]) / (variance[j] + epsilon).sqrt();
                output[[i, j]] = gamma[j] * normalized + beta[j];
            }
        }
        Ok(())
    }

    /// Helper functions for horizontal reductions
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx512f")]
    unsafe fn horizontal_sum_avx512(&self, v: __m512) -> f32 {
        let sum1 = _mm512_shuffle_f32x4(v, v, 0x4e);
        let sum2 = _mm512_add_ps(v, sum1);
        let sum3 = _mm512_shuffle_f32x4(sum2, sum2, 0xb1);
        let sum4 = _mm512_add_ps(sum2, sum3);
        let sum5 = _mm512_shuffle_ps(sum4, sum4, 0x4e);
        let sum6 = _mm512_add_ps(sum4, sum5);
        let sum7 = _mm512_shuffle_ps(sum6, sum6, 0xb1);
        let sum8 = _mm512_add_ps(sum6, sum7);
        _mm512_cvtss_f32(_mm512_castps512_ps128(sum8))
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn horizontal_sum_avx2(&self, v: __m256) -> f32 {
        let sum1 = _mm256_hadd_ps(v, v);
        let sum2 = _mm256_hadd_ps(sum1, sum1);
        let sum3 = _mm256_extractf128_ps(sum2, 1);
        let sum4 = _mm_add_ps(_mm256_castps256_ps128(sum2), sum3);
        _mm_cvtss_f32(sum4)
    }

    /// Fast tanh approximations for SIMD
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_tanh_approx(&self, x: __m512) -> __m512 {
        // Rational approximation: tanh(x) ≈ x * (27 + x²) / (27 + 9x²)
        let x2 = _mm512_mul_ps(x, x);
        let nine = _mm512_set1_ps(9.0);
        let twentyseven = _mm512_set1_ps(27.0);
        
        let numerator = _mm512_fmadd_ps(x2, x, _mm512_mul_ps(twentyseven, x));
        let denominator = _mm512_fmadd_ps(nine, x2, twentyseven);
        
        _mm512_div_ps(numerator, denominator)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_tanh_approx(&self, x: __m256) -> __m256 {
        // Same rational approximation for AVX2
        let x2 = _mm256_mul_ps(x, x);
        let nine = _mm256_set1_ps(9.0);
        let twentyseven = _mm256_set1_ps(27.0);
        
        let numerator = _mm256_fmadd_ps(x2, x, _mm256_mul_ps(twentyseven, x));
        let denominator = _mm256_fmadd_ps(nine, x2, twentyseven);
        
        _mm256_div_ps(numerator, denominator)
    }

    // Note: On non-x86 architectures, these functions won't be called due to the 
    // runtime checks in the public API methods, but we need stubs for compilation

    /// Get vectorization performance statistics
    pub fn get_performance_stats(&self) -> VectorizationStats {
        self.performance_monitor.clone()
    }

    /// Compile and optimize kernels for specific operations
    pub fn compile_kernel(&mut self, kernel_type: KernelType) -> Result<String> {
        let kernel_name = format!("{:?}_{:?}", kernel_type, self.config.vector_width);
        
        let kernel = VectorKernel {
            name: kernel_name.clone(),
            vector_width: self.config.vector_width.clone(),
            kernel_type,
            optimized_for: self.get_supported_features(),
            performance_profile: KernelPerformance::default(),
        };

        self.kernel_cache.insert(kernel_name.clone(), kernel);
        Ok(kernel_name)
    }

    fn get_supported_features(&self) -> Vec<CpuFeature> {
        let mut features = Vec::new();
        
        if self.capabilities.has_sse4_1 { features.push(CpuFeature::SSE41); }
        if self.capabilities.has_sse4_2 { features.push(CpuFeature::SSE42); }
        if self.capabilities.has_avx { features.push(CpuFeature::AVX); }
        if self.capabilities.has_avx2 { features.push(CpuFeature::AVX2); }
        if self.capabilities.has_avx512f { features.push(CpuFeature::AVX512F); }
        if self.capabilities.has_fma { features.push(CpuFeature::FMA); }
        
        features
    }
}

impl VectorizationStats {
    fn new() -> Self {
        Self {
            kernels_compiled: 0,
            total_operations: 0,
            vectorized_operations: 0,
            scalar_fallbacks: 0,
            average_vector_utilization: 0.0,
            peak_throughput: 0.0,
            cache_hit_rate: 0.0,
        }
    }
}

impl KernelPerformance {
    fn default() -> Self {
        Self {
            throughput_gflops: 0.0,
            latency_ns: 0,
            cache_efficiency: 0.0,
            vectorization_ratio: 0.0,
            instruction_count: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_vectorization_config() {
        let config = VectorizationConfig::default();
        assert!(config.enable_avx512);
        assert!(config.enable_avx2);
        assert!(config.enable_sse4);
    }

    #[test]
    fn test_cpu_capabilities_detection() {
        let caps = VectorizationEngine::detect_cpu_capabilities();
        // These tests will depend on the actual CPU running the tests
        assert!(caps.cache_line_size > 0);
        assert!(caps.l1_cache_size > 0);
    }

    #[test]
    fn test_vectorized_relu() {
        let config = VectorizationConfig::default();
        let engine = VectorizationEngine::new(config).unwrap();
        
        let input = Array2::from_shape_vec((2, 4), vec![
            -1.0, 2.0, -3.0, 4.0,
            5.0, -6.0, 7.0, -8.0,
        ]).unwrap();

        let output = engine.vectorized_relu(&input).unwrap();
        
        assert_eq!(output[[0, 0]], 0.0);
        assert_eq!(output[[0, 1]], 2.0);
        assert_eq!(output[[0, 2]], 0.0);
        assert_eq!(output[[0, 3]], 4.0);
        assert_eq!(output[[1, 0]], 5.0);
        assert_eq!(output[[1, 1]], 0.0);
        assert_eq!(output[[1, 2]], 7.0);
        assert_eq!(output[[1, 3]], 0.0);
    }

    #[test]
    fn test_matrix_multiplication() {
        let config = VectorizationConfig::default();
        let engine = VectorizationEngine::new(config).unwrap();

        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let result = engine.vectorized_matmul(&a, &b).unwrap();
        
        assert_eq!(result.dim(), (2, 2));
        assert_eq!(result[[0, 0]], 22.0);
        assert_eq!(result[[0, 1]], 28.0);
        assert_eq!(result[[1, 0]], 49.0);
        assert_eq!(result[[1, 1]], 64.0);
    }

    #[test]
    fn test_batch_normalization() {
        let config = VectorizationConfig::default();
        let engine = VectorizationEngine::new(config).unwrap();

        let input = Array2::from_shape_vec((2, 4), vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ]).unwrap();

        let gamma = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let beta = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
        let mean = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0]);
        let variance = Array1::from_vec(vec![4.0, 4.0, 4.0, 4.0]);

        let result = engine.vectorized_batch_norm(
            &input, &gamma, &beta, &mean, &variance, 1e-5
        ).unwrap();
        
        assert_eq!(result.dim(), (2, 4));
        // Results should be normalized values
        assert!((result[[0, 0]] - (-1.0)).abs() < 0.1);
    }
}