//! SIMD-optimized kernels for ultra-low latency neural computation
//! 
//! Implements AVX2/AVX-512 vectorized operations for LIF neuron dynamics,
//! batch processing, and memory-efficient tensor operations.

use wide::f32x8;
use std::arch::x86_64::*;
use anyhow::{Result, anyhow};
use candle_core::{Tensor, Device, DType};

/// SIMD configuration for different instruction sets
#[derive(Debug, Clone)]
pub struct SimdConfig {
    pub use_avx512: bool,
    pub use_avx2: bool,
    pub use_sse: bool,
    pub vector_width: usize,
    pub alignment: usize,
}

impl Default for SimdConfig {
    fn default() -> Self {
        Self {
            use_avx512: is_x86_feature_detected!("avx512f"),
            use_avx2: is_x86_feature_detected!("avx2"),
            use_sse: is_x86_feature_detected!("sse2"),
            vector_width: if is_x86_feature_detected!("avx512f") { 16 } 
                         else if is_x86_feature_detected!("avx2") { 8 }
                         else { 4 },
            alignment: 64, // Cache line alignment
        }
    }
}

/// SIMD-optimized LIF neuron processor
pub struct SimdLIFProcessor {
    config: SimdConfig,
    temp_buffers: SimdBuffers,
}

/// Aligned memory buffers for SIMD operations
struct SimdBuffers {
    v_mem_buffer: AlignedBuffer<f32>,
    i_syn_buffer: AlignedBuffer<f32>,
    spike_buffer: AlignedBuffer<u8>,
    temp_buffer: AlignedBuffer<f32>,
}

/// Cache-aligned memory buffer
#[repr(align(64))]
struct AlignedBuffer<T> {
    data: Vec<T>,
    capacity: usize,
}

impl<T: Clone + Default> AlignedBuffer<T> {
    fn new(capacity: usize) -> Self {
        let mut data = Vec::with_capacity(capacity);
        data.resize(capacity, T::default());
        Self { data, capacity }
    }
    
    fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    
    fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }
    
    fn len(&self) -> usize {
        self.data.len()
    }
}

impl SimdLIFProcessor {
    /// Create new SIMD processor with optimal configuration
    pub fn new(max_neurons: usize) -> Self {
        let config = SimdConfig::default();
        let temp_buffers = SimdBuffers {
            v_mem_buffer: AlignedBuffer::new(max_neurons),
            i_syn_buffer: AlignedBuffer::new(max_neurons),
            spike_buffer: AlignedBuffer::new(max_neurons),
            temp_buffer: AlignedBuffer::new(max_neurons),
        };
        
        Self { config, temp_buffers }
    }
    
    /// Process LIF neurons with SIMD optimization
    /// Target: <5ns per neuron with AVX-512
    pub fn process_lif_neurons_simd(
        &mut self,
        v_mem: &mut [f32],
        i_syn: &mut [f32],
        inputs: &[f32],
        spikes_out: &mut [bool],
        decay_mem: f32,
        decay_syn: f32,
        threshold: f32,
        reset_potential: f32,
    ) -> Result<()> {
        let n_neurons = v_mem.len();
        
        if n_neurons != i_syn.len() || n_neurons != inputs.len() || n_neurons != spikes_out.len() {
            return Err(anyhow!("Array length mismatch"));
        }
        
        if self.config.use_avx512 && n_neurons >= 16 {
            self.process_lif_avx512(
                v_mem, i_syn, inputs, spikes_out,
                decay_mem, decay_syn, threshold, reset_potential
            )?;
        } else if self.config.use_avx2 && n_neurons >= 8 {
            self.process_lif_avx2(
                v_mem, i_syn, inputs, spikes_out,
                decay_mem, decay_syn, threshold, reset_potential
            )?;
        } else {
            self.process_lif_scalar(
                v_mem, i_syn, inputs, spikes_out,
                decay_mem, decay_syn, threshold, reset_potential
            )?;
        }
        
        Ok(())
    }
    
    /// AVX-512 implementation (16 neurons per iteration)
    #[target_feature(enable = "avx512f")]
    unsafe fn process_lif_avx512(
        &mut self,
        v_mem: &mut [f32],
        i_syn: &mut [f32],
        inputs: &[f32],
        spikes_out: &mut [bool],
        decay_mem: f32,
        decay_syn: f32,
        threshold: f32,
        reset_potential: f32,
    ) -> Result<()> {
        let n_neurons = v_mem.len();
        let vec_size = 16;
        let n_vecs = n_neurons / vec_size;
        let remainder = n_neurons % vec_size;
        
        // Broadcast parameters to vectors
        let decay_mem_vec = _mm512_set1_ps(decay_mem);
        let decay_syn_vec = _mm512_set1_ps(decay_syn);
        let threshold_vec = _mm512_set1_ps(threshold);
        let reset_vec = _mm512_set1_ps(reset_potential);
        
        for i in 0..n_vecs {
            let base_idx = i * vec_size;
            
            // Load 16 neurons worth of data
            let v_mem_vec = _mm512_loadu_ps(v_mem.as_ptr().add(base_idx));
            let i_syn_vec = _mm512_loadu_ps(i_syn.as_ptr().add(base_idx));
            let input_vec = _mm512_loadu_ps(inputs.as_ptr().add(base_idx));
            
            // Update synaptic current: i_syn = i_syn * decay_syn + input
            let i_syn_new = _mm512_fmadd_ps(i_syn_vec, decay_syn_vec, input_vec);
            
            // Update membrane potential: v_mem = v_mem * decay_mem + i_syn
            let v_mem_new = _mm512_fmadd_ps(v_mem_vec, decay_mem_vec, i_syn_new);
            
            // Spike detection: spikes = v_mem >= threshold
            let spike_mask = _mm512_cmp_ps_mask(v_mem_new, threshold_vec, _CMP_GE_OQ);
            
            // Reset membrane potential where spikes occurred
            let v_mem_reset = _mm512_mask_blend_ps(spike_mask, v_mem_new, reset_vec);
            
            // Store results
            _mm512_storeu_ps(v_mem.as_mut_ptr().add(base_idx), v_mem_reset);
            _mm512_storeu_ps(i_syn.as_mut_ptr().add(base_idx), i_syn_new);
            
            // Store spike flags (convert mask to boolean array)
            for j in 0..vec_size {
                spikes_out[base_idx + j] = (spike_mask >> j) & 1 != 0;
            }
        }
        
        // Handle remainder with scalar operations
        if remainder > 0 {
            self.process_lif_scalar_range(
                v_mem, i_syn, inputs, spikes_out,
                n_vecs * vec_size, n_neurons,
                decay_mem, decay_syn, threshold, reset_potential
            );
        }
        
        Ok(())
    }
    
    /// AVX2 implementation (8 neurons per iteration)
    #[target_feature(enable = "avx2")]
    unsafe fn process_lif_avx2(
        &mut self,
        v_mem: &mut [f32],
        i_syn: &mut [f32],
        inputs: &[f32],
        spikes_out: &mut [bool],
        decay_mem: f32,
        decay_syn: f32,
        threshold: f32,
        reset_potential: f32,
    ) -> Result<()> {
        let n_neurons = v_mem.len();
        let vec_size = 8;
        let n_vecs = n_neurons / vec_size;
        let remainder = n_neurons % vec_size;
        
        // Broadcast parameters to vectors
        let decay_mem_vec = _mm256_set1_ps(decay_mem);
        let decay_syn_vec = _mm256_set1_ps(decay_syn);
        let threshold_vec = _mm256_set1_ps(threshold);
        let reset_vec = _mm256_set1_ps(reset_potential);
        
        for i in 0..n_vecs {
            let base_idx = i * vec_size;
            
            // Load 8 neurons worth of data
            let v_mem_vec = _mm256_loadu_ps(v_mem.as_ptr().add(base_idx));
            let i_syn_vec = _mm256_loadu_ps(i_syn.as_ptr().add(base_idx));
            let input_vec = _mm256_loadu_ps(inputs.as_ptr().add(base_idx));
            
            // Update synaptic current: i_syn = i_syn * decay_syn + input
            let i_syn_new = _mm256_fmadd_ps(i_syn_vec, decay_syn_vec, input_vec);
            
            // Update membrane potential: v_mem = v_mem * decay_mem + i_syn
            let v_mem_new = _mm256_fmadd_ps(v_mem_vec, decay_mem_vec, i_syn_new);
            
            // Spike detection: spikes = v_mem >= threshold
            let spike_mask = _mm256_cmp_ps(v_mem_new, threshold_vec, _CMP_GE_OQ);
            
            // Reset membrane potential where spikes occurred
            let v_mem_reset = _mm256_blendv_ps(v_mem_new, reset_vec, spike_mask);
            
            // Store results
            _mm256_storeu_ps(v_mem.as_mut_ptr().add(base_idx), v_mem_reset);
            _mm256_storeu_ps(i_syn.as_mut_ptr().add(base_idx), i_syn_new);
            
            // Store spike flags
            let mask_int = _mm256_movemask_ps(spike_mask);
            for j in 0..vec_size {
                spikes_out[base_idx + j] = (mask_int >> j) & 1 != 0;
            }
        }
        
        // Handle remainder
        if remainder > 0 {
            self.process_lif_scalar_range(
                v_mem, i_syn, inputs, spikes_out,
                n_vecs * vec_size, n_neurons,
                decay_mem, decay_syn, threshold, reset_potential
            );
        }
        
        Ok(())
    }
    
    /// Scalar fallback implementation
    fn process_lif_scalar(
        &mut self,
        v_mem: &mut [f32],
        i_syn: &mut [f32],
        inputs: &[f32],
        spikes_out: &mut [bool],
        decay_mem: f32,
        decay_syn: f32,
        threshold: f32,
        reset_potential: f32,
    ) -> Result<()> {
        self.process_lif_scalar_range(
            v_mem, i_syn, inputs, spikes_out,
            0, v_mem.len(),
            decay_mem, decay_syn, threshold, reset_potential
        );
        Ok(())
    }
    
    /// Scalar implementation for specified range
    fn process_lif_scalar_range(
        &mut self,
        v_mem: &mut [f32],
        i_syn: &mut [f32],
        inputs: &[f32],
        spikes_out: &mut [bool],
        start: usize,
        end: usize,
        decay_mem: f32,
        decay_syn: f32,
        threshold: f32,
        reset_potential: f32,
    ) {
        for i in start..end {
            // Update synaptic current
            i_syn[i] = i_syn[i] * decay_syn + inputs[i];
            
            // Update membrane potential
            v_mem[i] = v_mem[i] * decay_mem + i_syn[i];
            
            // Check for spike and reset
            if v_mem[i] >= threshold {
                spikes_out[i] = true;
                v_mem[i] = reset_potential;
            } else {
                spikes_out[i] = false;
            }
        }
    }
    
    /// SIMD-optimized batch matrix-vector multiplication
    pub fn simd_matvec(
        &mut self,
        matrix: &[f32],
        vector: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        if matrix.len() != rows * cols || vector.len() != cols || output.len() != rows {
            return Err(anyhow!("Matrix-vector dimension mismatch"));
        }
        
        if self.config.use_avx2 && cols >= 8 {
            unsafe { self.simd_matvec_avx2(matrix, vector, output, rows, cols) }
        } else {
            self.simd_matvec_scalar(matrix, vector, output, rows, cols)
        }
    }
    
    /// AVX2 matrix-vector multiplication
    #[target_feature(enable = "avx2")]
    unsafe fn simd_matvec_avx2(
        &mut self,
        matrix: &[f32],
        vector: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let vec_size = 8;
        let n_vecs = cols / vec_size;
        let remainder = cols % vec_size;
        
        for row in 0..rows {
            let mut sum_vec = _mm256_setzero_ps();
            let row_offset = row * cols;
            
            // Process 8 elements at a time
            for i in 0..n_vecs {
                let col_offset = i * vec_size;
                let matrix_vec = _mm256_loadu_ps(matrix.as_ptr().add(row_offset + col_offset));
                let vector_vec = _mm256_loadu_ps(vector.as_ptr().add(col_offset));
                sum_vec = _mm256_fmadd_ps(matrix_vec, vector_vec, sum_vec);
            }
            
            // Horizontal sum of the vector
            let mut result = self.horizontal_sum_avx2(sum_vec);
            
            // Handle remainder elements
            for i in (n_vecs * vec_size)..cols {
                result += matrix[row_offset + i] * vector[i];
            }
            
            output[row] = result;
        }
        
        Ok(())
    }
    
    /// Horizontal sum of AVX2 vector
    #[target_feature(enable = "avx2")]
    unsafe fn horizontal_sum_avx2(&self, vec: __m256) -> f32 {
        let hi = _mm256_extractf128_ps(vec, 1);
        let lo = _mm256_castps256_ps128(vec);
        let sum128 = _mm_add_ps(hi, lo);
        
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        
        _mm_cvtss_f32(sum32)
    }
    
    /// Scalar matrix-vector multiplication fallback
    fn simd_matvec_scalar(
        &mut self,
        matrix: &[f32],
        vector: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        for row in 0..rows {
            let mut sum = 0.0;
            let row_offset = row * cols;
            
            for col in 0..cols {
                sum += matrix[row_offset + col] * vector[col];
            }
            
            output[row] = sum;
        }
        
        Ok(())
    }
    
    /// SIMD-optimized element-wise operations
    pub fn simd_elementwise_ops(
        &mut self,
        a: &[f32],
        b: &[f32],
        output: &mut [f32],
        op: ElementwiseOp,
    ) -> Result<()> {
        if a.len() != b.len() || a.len() != output.len() {
            return Err(anyhow!("Array length mismatch"));
        }
        
        let n = a.len();
        
        if self.config.use_avx2 && n >= 8 {
            unsafe { self.simd_elementwise_avx2(a, b, output, op) }
        } else {
            self.simd_elementwise_scalar(a, b, output, op)
        }
    }
    
    /// AVX2 element-wise operations
    #[target_feature(enable = "avx2")]
    unsafe fn simd_elementwise_avx2(
        &mut self,
        a: &[f32],
        b: &[f32],
        output: &mut [f32],
        op: ElementwiseOp,
    ) -> Result<()> {
        let n = a.len();
        let vec_size = 8;
        let n_vecs = n / vec_size;
        let remainder = n % vec_size;
        
        for i in 0..n_vecs {
            let offset = i * vec_size;
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(offset));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(offset));
            
            let result_vec = match op {
                ElementwiseOp::Add => _mm256_add_ps(a_vec, b_vec),
                ElementwiseOp::Sub => _mm256_sub_ps(a_vec, b_vec),
                ElementwiseOp::Mul => _mm256_mul_ps(a_vec, b_vec),
                ElementwiseOp::Div => _mm256_div_ps(a_vec, b_vec),
                ElementwiseOp::Max => _mm256_max_ps(a_vec, b_vec),
                ElementwiseOp::Min => _mm256_min_ps(a_vec, b_vec),
            };
            
            _mm256_storeu_ps(output.as_mut_ptr().add(offset), result_vec);
        }
        
        // Handle remainder
        if remainder > 0 {
            let start = n_vecs * vec_size;
            self.simd_elementwise_scalar_range(a, b, output, op, start, n);
        }
        
        Ok(())
    }
    
    /// Scalar element-wise operations
    fn simd_elementwise_scalar(
        &mut self,
        a: &[f32],
        b: &[f32],
        output: &mut [f32],
        op: ElementwiseOp,
    ) -> Result<()> {
        self.simd_elementwise_scalar_range(a, b, output, op, 0, a.len());
        Ok(())
    }
    
    /// Scalar element-wise operations for range
    fn simd_elementwise_scalar_range(
        &mut self,
        a: &[f32],
        b: &[f32],
        output: &mut [f32],
        op: ElementwiseOp,
        start: usize,
        end: usize,
    ) {
        for i in start..end {
            output[i] = match op {
                ElementwiseOp::Add => a[i] + b[i],
                ElementwiseOp::Sub => a[i] - b[i],
                ElementwiseOp::Mul => a[i] * b[i],
                ElementwiseOp::Div => a[i] / b[i],
                ElementwiseOp::Max => a[i].max(b[i]),
                ElementwiseOp::Min => a[i].min(b[i]),
            };
        }
    }
    
    /// Get configuration info
    pub fn get_config(&self) -> &SimdConfig {
        &self.config
    }
    
    /// Benchmark SIMD performance
    pub fn benchmark_simd_performance(&mut self, size: usize) -> Result<SimdBenchmarkResults> {
        let mut v_mem = vec![0.5; size];
        let mut i_syn = vec![0.3; size];
        let inputs = vec![1.0; size];
        let mut spikes = vec![false; size];
        
        // Benchmark LIF processing
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            self.process_lif_neurons_simd(
                &mut v_mem, &mut i_syn, &inputs, &mut spikes,
                0.9, 0.8, 1.0, 0.0
            )?;
        }
        let lif_time = start.elapsed();
        
        // Benchmark matrix-vector multiplication
        let matrix = vec![0.1; size * size];
        let vector = vec![1.0; size];
        let mut output = vec![0.0; size];
        
        let start = std::time::Instant::now();
        for _ in 0..100 {
            self.simd_matvec(&matrix, &vector, &mut output, size, size)?;
        }
        let matvec_time = start.elapsed();
        
        Ok(SimdBenchmarkResults {
            lif_processing_time: lif_time,
            matvec_time,
            throughput_neurons_per_second: (size as f64 * 1000.0) / lif_time.as_secs_f64(),
            vector_width: self.config.vector_width,
            instruction_set: self.get_instruction_set(),
        })
    }
    
    fn get_instruction_set(&self) -> String {
        if self.config.use_avx512 {
            "AVX-512".to_string()
        } else if self.config.use_avx2 {
            "AVX2".to_string()
        } else if self.config.use_sse {
            "SSE2".to_string()
        } else {
            "Scalar".to_string()
        }
    }
}

/// Element-wise operation types
#[derive(Debug, Clone, Copy)]
pub enum ElementwiseOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
}

/// SIMD benchmark results
#[derive(Debug, Clone)]
pub struct SimdBenchmarkResults {
    pub lif_processing_time: std::time::Duration,
    pub matvec_time: std::time::Duration,
    pub throughput_neurons_per_second: f64,
    pub vector_width: usize,
    pub instruction_set: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_config() {
        let config = SimdConfig::default();
        assert!(config.vector_width >= 4);
        assert_eq!(config.alignment, 64);
    }
    
    #[test]
    fn test_aligned_buffer() {
        let buffer = AlignedBuffer::<f32>::new(100);
        assert_eq!(buffer.len(), 100);
        
        // Check alignment
        let ptr = buffer.as_ptr() as usize;
        assert_eq!(ptr % 64, 0);
    }
    
    #[test]
    fn test_lif_processing() {
        let mut processor = SimdLIFProcessor::new(100);
        let mut v_mem = vec![0.5; 10];
        let mut i_syn = vec![0.3; 10];
        let inputs = vec![1.0; 10];
        let mut spikes = vec![false; 10];
        
        let result = processor.process_lif_neurons_simd(
            &mut v_mem, &mut i_syn, &inputs, &mut spikes,
            0.9, 0.8, 1.0, 0.0
        );
        
        assert!(result.is_ok());
        
        // Check that some processing occurred
        assert!(v_mem.iter().any(|&x| x != 0.5));
        assert!(i_syn.iter().any(|&x| x != 0.3));
    }
    
    #[test]
    fn test_elementwise_operations() {
        let mut processor = SimdLIFProcessor::new(100);
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.5, 1.5, 2.5, 3.5];
        let mut output = vec![0.0; 4];
        
        processor.simd_elementwise_ops(&a, &b, &mut output, ElementwiseOp::Add).unwrap();
        
        assert_eq!(output, vec![1.5, 3.5, 5.5, 7.5]);
    }
    
    #[test]
    fn test_matrix_vector_multiplication() {
        let mut processor = SimdLIFProcessor::new(100);
        let matrix = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let vector = vec![1.0, 1.0];
        let mut output = vec![0.0; 2];
        
        processor.simd_matvec(&matrix, &vector, &mut output, 2, 2).unwrap();
        
        assert_eq!(output, vec![3.0, 7.0]); // [1+2, 3+4]
    }
}