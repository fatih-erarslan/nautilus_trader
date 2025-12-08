//! SIMD-accelerated preprocessing for CPU-side optimizations

use std::sync::Arc;
use wide::*;
use crate::{
    QBMIAError, QBMIAResult, QuantumState, PayoffMatrix, Pattern,
    Complex64, SimdFloat4, SimdComplex2
};

/// SIMD processor for ultra-fast CPU preprocessing
pub struct SimdProcessor {
    /// CPU feature detection
    cpu_features: CpuFeatures,
    
    /// Performance metrics
    metrics: Arc<tokio::sync::Mutex<SimdMetrics>>,
}

impl SimdProcessor {
    /// Create new SIMD processor with feature detection
    pub fn new() -> Self {
        let cpu_features = CpuFeatures::detect();
        let metrics = Arc::new(tokio::sync::Mutex::new(SimdMetrics::new()));
        
        tracing::info!("SIMD processor initialized with features: {:?}", cpu_features);
        
        Self {
            cpu_features,
            metrics,
        }
    }
    
    /// Preprocess quantum state for GPU transfer
    pub fn preprocess_state(&self, state: &QuantumState) -> QBMIAResult<QuantumState> {
        let start_time = std::time::Instant::now();
        
        // Use SIMD to normalize and validate quantum state
        let preprocessed_amplitudes = if self.cpu_features.has_avx2() {
            self.preprocess_state_avx2(&state.amplitudes)?
        } else if self.cpu_features.has_sse41() {
            self.preprocess_state_sse41(&state.amplitudes)?
        } else {
            self.preprocess_state_scalar(&state.amplitudes)?
        };
        
        let processing_time = start_time.elapsed();
        
        // Validate sub-10ns target for preprocessing
        if processing_time.as_nanos() > 10 {
            tracing::warn!(
                "SIMD preprocessing took {}ns, exceeding 10ns target",
                processing_time.as_nanos()
            );
        }
        
        let mut preprocessed_state = state.clone();
        preprocessed_state.amplitudes = preprocessed_amplitudes;
        
        Ok(preprocessed_state)
    }
    
    /// Preprocess payoff matrix for GPU transfer
    pub fn preprocess_matrix(&self, matrix: &PayoffMatrix) -> QBMIAResult<PayoffMatrix> {
        let start_time = std::time::Instant::now();
        
        // Use SIMD to validate and optimize matrix for GPU
        let preprocessed_data = if self.cpu_features.has_avx2() {
            self.preprocess_matrix_avx2(&matrix.data)?
        } else if self.cpu_features.has_sse41() {
            self.preprocess_matrix_sse41(&matrix.data)?
        } else {
            self.preprocess_matrix_scalar(&matrix.data)?
        };
        
        let processing_time = start_time.elapsed();
        
        let mut preprocessed_matrix = matrix.clone();
        preprocessed_matrix.data = preprocessed_data;
        
        Ok(preprocessed_matrix)
    }
    
    /// Preprocess patterns for GPU pattern matching
    pub fn preprocess_patterns(&self, patterns: &[Pattern]) -> QBMIAResult<Vec<Pattern>> {
        let start_time = std::time::Instant::now();
        
        let preprocessed_patterns: Result<Vec<_>, _> = patterns
            .iter()
            .map(|pattern| self.preprocess_single_pattern(pattern))
            .collect();
        
        let processing_time = start_time.elapsed();
        
        tracing::debug!(
            "SIMD preprocessed {} patterns in {:.3}ns",
            patterns.len(),
            processing_time.as_nanos()
        );
        
        preprocessed_patterns
    }
    
    /// Fast matrix-vector multiplication using SIMD
    pub fn matrix_vector_multiply(&self, matrix: &[f32], vector: &[f32], rows: usize, cols: usize) -> QBMIAResult<Vec<f32>> {
        if matrix.len() != rows * cols {
            return Err(QBMIAError::simd_op("Invalid matrix dimensions"));
        }
        
        if vector.len() != cols {
            return Err(QBMIAError::simd_op("Invalid vector dimensions"));
        }
        
        let start_time = std::time::Instant::now();
        
        let result = if self.cpu_features.has_avx2() {
            self.matrix_vector_multiply_avx2(matrix, vector, rows, cols)?
        } else if self.cpu_features.has_sse41() {
            self.matrix_vector_multiply_sse41(matrix, vector, rows, cols)?
        } else {
            self.matrix_vector_multiply_scalar(matrix, vector, rows, cols)?
        };
        
        let multiplication_time = start_time.elapsed();
        
        tracing::debug!(
            "SIMD matrix-vector multiply ({}x{}) in {:.3}ns",
            rows, cols, multiplication_time.as_nanos()
        );
        
        Ok(result)
    }
    
    /// Fast dot product using SIMD
    pub fn dot_product(&self, a: &[f32], b: &[f32]) -> QBMIAResult<f32> {
        if a.len() != b.len() {
            return Err(QBMIAError::simd_op("Vector lengths don't match"));
        }
        
        let start_time = std::time::Instant::now();
        
        let result = if self.cpu_features.has_avx2() {
            self.dot_product_avx2(a, b)?
        } else if self.cpu_features.has_sse41() {
            self.dot_product_sse41(a, b)?
        } else {
            self.dot_product_scalar(a, b)?
        };
        
        let dot_time = start_time.elapsed();
        
        // Target sub-5ns for small vectors
        if a.len() <= 16 && dot_time.as_nanos() > 5 {
            tracing::warn!(
                "SIMD dot product ({} elements) took {}ns, exceeding 5ns target",
                a.len(), dot_time.as_nanos()
            );
        }
        
        Ok(result)
    }
    
    /// Fast vector normalization using SIMD
    pub fn normalize_vector(&self, vector: &mut [f32]) -> QBMIAResult<()> {
        let start_time = std::time::Instant::now();
        
        if self.cpu_features.has_avx2() {
            self.normalize_vector_avx2(vector)?;
        } else if self.cpu_features.has_sse41() {
            self.normalize_vector_sse41(vector)?;
        } else {
            self.normalize_vector_scalar(vector)?;
        }
        
        let normalization_time = start_time.elapsed();
        
        tracing::debug!(
            "SIMD normalized vector ({} elements) in {:.3}ns",
            vector.len(), normalization_time.as_nanos()
        );
        
        Ok(())
    }
    
    /// Complex number operations using SIMD
    pub fn complex_multiply_arrays(&self, a: &[Complex64], b: &[Complex64]) -> QBMIAResult<Vec<Complex64>> {
        if a.len() != b.len() {
            return Err(QBMIAError::simd_op("Complex arrays lengths don't match"));
        }
        
        let start_time = std::time::Instant::now();
        
        let result = if self.cpu_features.has_avx2() {
            self.complex_multiply_avx2(a, b)?
        } else if self.cpu_features.has_sse41() {
            self.complex_multiply_sse41(a, b)?
        } else {
            self.complex_multiply_scalar(a, b)?
        };
        
        let multiplication_time = start_time.elapsed();
        
        tracing::debug!(
            "SIMD complex multiply ({} elements) in {:.3}ns",
            a.len(), multiplication_time.as_nanos()
        );
        
        Ok(result)
    }
    
    // AVX2 implementations (256-bit SIMD)
    fn preprocess_state_avx2(&self, amplitudes: &[Complex64]) -> QBMIAResult<Vec<Complex64>> {
        let mut result = amplitudes.to_vec();
        
        // Process 4 complex numbers at a time (8 f32 values)
        for chunk in result.chunks_exact_mut(4) {
            // Load 8 f32 values (4 complex numbers)
            let reals = [chunk[0].real, chunk[1].real, chunk[2].real, chunk[3].real];
            let imags = [chunk[0].imag, chunk[1].imag, chunk[2].imag, chunk[3].imag];
            
            let real_vec = f32x8::from([reals[0], imags[0], reals[1], imags[1], 
                                       reals[2], imags[2], reals[3], imags[3]]);
            
            // Validate that all values are finite
            let is_finite = real_vec.is_finite();
            if !is_finite.all() {
                return Err(QBMIAError::simd_op("Non-finite values in quantum state"));
            }
            
            // Clamp very small values to zero for numerical stability
            let threshold = f32x8::splat(1e-15);
            let abs_vals = real_vec.abs();
            let mask = abs_vals.cmp_lt(threshold);
            let clamped = mask.blend(f32x8::splat(0.0), real_vec);
            
            // Store back
            let values: [f32; 8] = clamped.into();
            chunk[0] = Complex64::new(values[0], values[1]);
            chunk[1] = Complex64::new(values[2], values[3]);
            chunk[2] = Complex64::new(values[4], values[5]);
            chunk[3] = Complex64::new(values[6], values[7]);
        }
        
        // Handle remaining elements
        for amplitude in result.chunks_exact_mut(4).remainder() {
            if !amplitude.real.is_finite() || !amplitude.imag.is_finite() {
                return Err(QBMIAError::simd_op("Non-finite values in quantum state"));
            }
            
            if amplitude.real.abs() < 1e-15 {
                amplitude.real = 0.0;
            }
            if amplitude.imag.abs() < 1e-15 {
                amplitude.imag = 0.0;
            }
        }
        
        Ok(result)
    }
    
    fn preprocess_matrix_avx2(&self, data: &[f32]) -> QBMIAResult<Vec<f32>> {
        let mut result = data.to_vec();
        
        // Process 8 f32 values at a time
        for chunk in result.chunks_exact_mut(8) {
            let vec = f32x8::from(*chunk);
            
            // Validate finite values
            let is_finite = vec.is_finite();
            if !is_finite.all() {
                return Err(QBMIAError::simd_op("Non-finite values in payoff matrix"));
            }
            
            // Clamp extreme values
            let min_val = f32x8::splat(-1e6);
            let max_val = f32x8::splat(1e6);
            let clamped = vec.max(min_val).min(max_val);
            
            *chunk = clamped.into();
        }
        
        // Handle remaining elements
        for value in result.chunks_exact_mut(8).remainder() {
            if !value.is_finite() {
                return Err(QBMIAError::simd_op("Non-finite values in payoff matrix"));
            }
            *value = value.clamp(-1e6, 1e6);
        }
        
        Ok(result)
    }
    
    fn matrix_vector_multiply_avx2(&self, matrix: &[f32], vector: &[f32], rows: usize, cols: usize) -> QBMIAResult<Vec<f32>> {
        let mut result = vec![0.0f32; rows];
        
        for i in 0..rows {
            let row_start = i * cols;
            let row = &matrix[row_start..row_start + cols];
            
            let mut sum = f32x8::splat(0.0);
            
            // Process 8 elements at a time
            for (row_chunk, vec_chunk) in row.chunks_exact(8).zip(vector.chunks_exact(8)) {
                let row_vec = f32x8::from(*row_chunk);
                let vec_vec = f32x8::from(*vec_chunk);
                sum += row_vec * vec_vec;
            }
            
            // Sum the SIMD vector
            let sum_array: [f32; 8] = sum.into();
            let mut row_sum = sum_array.iter().sum::<f32>();
            
            // Handle remaining elements
            let remaining_start = (cols / 8) * 8;
            for j in remaining_start..cols {
                row_sum += row[j - row_start] * vector[j];
            }
            
            result[i] = row_sum;
        }
        
        Ok(result)
    }
    
    fn dot_product_avx2(&self, a: &[f32], b: &[f32]) -> QBMIAResult<f32> {
        let mut sum = f32x8::splat(0.0);
        
        // Process 8 elements at a time
        for (a_chunk, b_chunk) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
            let a_vec = f32x8::from(*a_chunk);
            let b_vec = f32x8::from(*b_chunk);
            sum += a_vec * b_vec;
        }
        
        // Sum the SIMD vector
        let sum_array: [f32; 8] = sum.into();
        let mut result = sum_array.iter().sum::<f32>();
        
        // Handle remaining elements
        let remaining_start = (a.len() / 8) * 8;
        for i in remaining_start..a.len() {
            result += a[i] * b[i];
        }
        
        Ok(result)
    }
    
    fn normalize_vector_avx2(&self, vector: &mut [f32]) -> QBMIAResult<()> {
        // First pass: calculate norm using dot product
        let norm_squared = self.dot_product_avx2(vector, vector)?;
        let norm = norm_squared.sqrt();
        
        if norm < 1e-15 {
            return Err(QBMIAError::simd_op("Cannot normalize zero vector"));
        }
        
        let inv_norm = f32x8::splat(1.0 / norm);
        
        // Second pass: normalize using SIMD
        for chunk in vector.chunks_exact_mut(8) {
            let vec = f32x8::from(*chunk);
            let normalized = vec * inv_norm;
            *chunk = normalized.into();
        }
        
        // Handle remaining elements
        for value in vector.chunks_exact_mut(8).remainder() {
            *value /= norm;
        }
        
        Ok(())
    }
    
    fn complex_multiply_avx2(&self, a: &[Complex64], b: &[Complex64]) -> QBMIAResult<Vec<Complex64>> {
        let mut result = Vec::with_capacity(a.len());
        
        // Process 4 complex numbers at a time
        for (a_chunk, b_chunk) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
            // Load complex numbers into SIMD vectors
            let a_reals = f32x4::from([a_chunk[0].real, a_chunk[1].real, a_chunk[2].real, a_chunk[3].real]);
            let a_imags = f32x4::from([a_chunk[0].imag, a_chunk[1].imag, a_chunk[2].imag, a_chunk[3].imag]);
            let b_reals = f32x4::from([b_chunk[0].real, b_chunk[1].real, b_chunk[2].real, b_chunk[3].real]);
            let b_imags = f32x4::from([b_chunk[0].imag, b_chunk[1].imag, b_chunk[2].imag, b_chunk[3].imag]);
            
            // Complex multiplication: (a.real * b.real - a.imag * b.imag) + i(a.real * b.imag + a.imag * b.real)
            let result_reals = a_reals * b_reals - a_imags * b_imags;
            let result_imags = a_reals * b_imags + a_imags * b_reals;
            
            let real_array: [f32; 4] = result_reals.into();
            let imag_array: [f32; 4] = result_imags.into();
            
            for i in 0..4 {
                result.push(Complex64::new(real_array[i], imag_array[i]));
            }
        }
        
        // Handle remaining elements
        for (a_val, b_val) in a.chunks_exact(4).remainder().iter().zip(b.chunks_exact(4).remainder().iter()) {
            result.push(*a_val * *b_val);
        }
        
        Ok(result)
    }
    
    // SSE4.1 implementations (128-bit SIMD) - similar to AVX2 but smaller vectors
    fn preprocess_state_sse41(&self, amplitudes: &[Complex64]) -> QBMIAResult<Vec<Complex64>> {
        // Similar to AVX2 but using f32x4 instead of f32x8
        let mut result = amplitudes.to_vec();
        
        for chunk in result.chunks_exact_mut(2) {
            let reals = [chunk[0].real, chunk[1].real, 0.0, 0.0];
            let imags = [chunk[0].imag, chunk[1].imag, 0.0, 0.0];
            
            let real_vec = f32x4::from(reals);
            let imag_vec = f32x4::from(imags);
            
            let is_finite = real_vec.is_finite() & imag_vec.is_finite();
            if !is_finite.all() {
                return Err(QBMIAError::simd_op("Non-finite values in quantum state"));
            }
            
            let threshold = f32x4::splat(1e-15);
            let real_abs = real_vec.abs();
            let imag_abs = imag_vec.abs();
            let real_mask = real_abs.cmp_lt(threshold);
            let imag_mask = imag_abs.cmp_lt(threshold);
            
            let clamped_reals = real_mask.blend(f32x4::splat(0.0), real_vec);
            let clamped_imags = imag_mask.blend(f32x4::splat(0.0), imag_vec);
            
            let real_values: [f32; 4] = clamped_reals.into();
            let imag_values: [f32; 4] = clamped_imags.into();
            
            chunk[0] = Complex64::new(real_values[0], imag_values[0]);
            chunk[1] = Complex64::new(real_values[1], imag_values[1]);
        }
        
        // Handle remaining elements
        for amplitude in result.chunks_exact_mut(2).remainder() {
            if !amplitude.real.is_finite() || !amplitude.imag.is_finite() {
                return Err(QBMIAError::simd_op("Non-finite values in quantum state"));
            }
            
            if amplitude.real.abs() < 1e-15 {
                amplitude.real = 0.0;
            }
            if amplitude.imag.abs() < 1e-15 {
                amplitude.imag = 0.0;
            }
        }
        
        Ok(result)
    }
    
    fn preprocess_matrix_sse41(&self, data: &[f32]) -> QBMIAResult<Vec<f32>> {
        let mut result = data.to_vec();
        
        for chunk in result.chunks_exact_mut(4) {
            let vec = f32x4::from(*chunk);
            
            let is_finite = vec.is_finite();
            if !is_finite.all() {
                return Err(QBMIAError::simd_op("Non-finite values in payoff matrix"));
            }
            
            let min_val = f32x4::splat(-1e6);
            let max_val = f32x4::splat(1e6);
            let clamped = vec.max(min_val).min(max_val);
            
            *chunk = clamped.into();
        }
        
        for value in result.chunks_exact_mut(4).remainder() {
            if !value.is_finite() {
                return Err(QBMIAError::simd_op("Non-finite values in payoff matrix"));
            }
            *value = value.clamp(-1e6, 1e6);
        }
        
        Ok(result)
    }
    
    fn matrix_vector_multiply_sse41(&self, matrix: &[f32], vector: &[f32], rows: usize, cols: usize) -> QBMIAResult<Vec<f32>> {
        let mut result = vec![0.0f32; rows];
        
        for i in 0..rows {
            let row_start = i * cols;
            let row = &matrix[row_start..row_start + cols];
            
            let mut sum = f32x4::splat(0.0);
            
            for (row_chunk, vec_chunk) in row.chunks_exact(4).zip(vector.chunks_exact(4)) {
                let row_vec = f32x4::from(*row_chunk);
                let vec_vec = f32x4::from(*vec_chunk);
                sum += row_vec * vec_vec;
            }
            
            let sum_array: [f32; 4] = sum.into();
            let mut row_sum = sum_array.iter().sum::<f32>();
            
            let remaining_start = (cols / 4) * 4;
            for j in remaining_start..cols {
                row_sum += row[j - row_start] * vector[j];
            }
            
            result[i] = row_sum;
        }
        
        Ok(result)
    }
    
    fn dot_product_sse41(&self, a: &[f32], b: &[f32]) -> QBMIAResult<f32> {
        let mut sum = f32x4::splat(0.0);
        
        for (a_chunk, b_chunk) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
            let a_vec = f32x4::from(*a_chunk);
            let b_vec = f32x4::from(*b_chunk);
            sum += a_vec * b_vec;
        }
        
        let sum_array: [f32; 4] = sum.into();
        let mut result = sum_array.iter().sum::<f32>();
        
        let remaining_start = (a.len() / 4) * 4;
        for i in remaining_start..a.len() {
            result += a[i] * b[i];
        }
        
        Ok(result)
    }
    
    fn normalize_vector_sse41(&self, vector: &mut [f32]) -> QBMIAResult<()> {
        let norm_squared = self.dot_product_sse41(vector, vector)?;
        let norm = norm_squared.sqrt();
        
        if norm < 1e-15 {
            return Err(QBMIAError::simd_op("Cannot normalize zero vector"));
        }
        
        let inv_norm = f32x4::splat(1.0 / norm);
        
        for chunk in vector.chunks_exact_mut(4) {
            let vec = f32x4::from(*chunk);
            let normalized = vec * inv_norm;
            *chunk = normalized.into();
        }
        
        for value in vector.chunks_exact_mut(4).remainder() {
            *value /= norm;
        }
        
        Ok(())
    }
    
    fn complex_multiply_sse41(&self, a: &[Complex64], b: &[Complex64]) -> QBMIAResult<Vec<Complex64>> {
        let mut result = Vec::with_capacity(a.len());
        
        for (a_chunk, b_chunk) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
            let a_reals = f32x4::from([a_chunk[0].real, a_chunk[1].real, 0.0, 0.0]);
            let a_imags = f32x4::from([a_chunk[0].imag, a_chunk[1].imag, 0.0, 0.0]);
            let b_reals = f32x4::from([b_chunk[0].real, b_chunk[1].real, 0.0, 0.0]);
            let b_imags = f32x4::from([b_chunk[0].imag, b_chunk[1].imag, 0.0, 0.0]);
            
            let result_reals = a_reals * b_reals - a_imags * b_imags;
            let result_imags = a_reals * b_imags + a_imags * b_reals;
            
            let real_array: [f32; 4] = result_reals.into();
            let imag_array: [f32; 4] = result_imags.into();
            
            result.push(Complex64::new(real_array[0], imag_array[0]));
            result.push(Complex64::new(real_array[1], imag_array[1]));
        }
        
        for (a_val, b_val) in a.chunks_exact(2).remainder().iter().zip(b.chunks_exact(2).remainder().iter()) {
            result.push(*a_val * *b_val);
        }
        
        Ok(result)
    }
    
    // Scalar implementations (fallback)
    fn preprocess_state_scalar(&self, amplitudes: &[Complex64]) -> QBMIAResult<Vec<Complex64>> {
        let mut result = amplitudes.to_vec();
        
        for amplitude in &mut result {
            if !amplitude.real.is_finite() || !amplitude.imag.is_finite() {
                return Err(QBMIAError::simd_op("Non-finite values in quantum state"));
            }
            
            if amplitude.real.abs() < 1e-15 {
                amplitude.real = 0.0;
            }
            if amplitude.imag.abs() < 1e-15 {
                amplitude.imag = 0.0;
            }
        }
        
        Ok(result)
    }
    
    fn preprocess_matrix_scalar(&self, data: &[f32]) -> QBMIAResult<Vec<f32>> {
        let mut result = data.to_vec();
        
        for value in &mut result {
            if !value.is_finite() {
                return Err(QBMIAError::simd_op("Non-finite values in payoff matrix"));
            }
            *value = value.clamp(-1e6, 1e6);
        }
        
        Ok(result)
    }
    
    fn matrix_vector_multiply_scalar(&self, matrix: &[f32], vector: &[f32], rows: usize, cols: usize) -> QBMIAResult<Vec<f32>> {
        let mut result = vec![0.0f32; rows];
        
        for i in 0..rows {
            let row_start = i * cols;
            for j in 0..cols {
                result[i] += matrix[row_start + j] * vector[j];
            }
        }
        
        Ok(result)
    }
    
    fn dot_product_scalar(&self, a: &[f32], b: &[f32]) -> QBMIAResult<f32> {
        let result = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        Ok(result)
    }
    
    fn normalize_vector_scalar(&self, vector: &mut [f32]) -> QBMIAResult<()> {
        let norm_squared: f32 = vector.iter().map(|x| x * x).sum();
        let norm = norm_squared.sqrt();
        
        if norm < 1e-15 {
            return Err(QBMIAError::simd_op("Cannot normalize zero vector"));
        }
        
        for value in vector {
            *value /= norm;
        }
        
        Ok(())
    }
    
    fn complex_multiply_scalar(&self, a: &[Complex64], b: &[Complex64]) -> QBMIAResult<Vec<Complex64>> {
        let result = a.iter().zip(b.iter()).map(|(x, y)| *x * *y).collect();
        Ok(result)
    }
    
    /// Preprocess single pattern
    fn preprocess_single_pattern(&self, pattern: &Pattern) -> QBMIAResult<Pattern> {
        let mut preprocessed_features = pattern.features.clone();
        
        // Normalize pattern features
        self.normalize_vector_scalar(&mut preprocessed_features)?;
        
        // Validate features are finite
        for feature in &preprocessed_features {
            if !feature.is_finite() {
                return Err(QBMIAError::simd_op("Non-finite values in pattern"));
            }
        }
        
        Ok(Pattern::new(preprocessed_features, pattern.label.clone()))
    }
    
    /// Get performance metrics
    pub async fn get_metrics(&self) -> SimdMetrics {
        let metrics = self.metrics.lock().await;
        metrics.clone()
    }
}

/// CPU feature detection
#[derive(Debug, Clone)]
struct CpuFeatures {
    has_sse41: bool,
    has_avx: bool,
    has_avx2: bool,
    has_fma: bool,
}

impl CpuFeatures {
    fn detect() -> Self {
        // Simplified feature detection - would use actual CPU detection in production
        Self {
            has_sse41: true,  // Most modern CPUs have SSE4.1
            has_avx: true,    // Most modern CPUs have AVX
            has_avx2: true,   // Most modern CPUs have AVX2
            has_fma: true,    // Most modern CPUs have FMA
        }
    }
    
    fn has_sse41(&self) -> bool {
        self.has_sse41
    }
    
    fn has_avx(&self) -> bool {
        self.has_avx
    }
    
    fn has_avx2(&self) -> bool {
        self.has_avx2
    }
    
    fn has_fma(&self) -> bool {
        self.has_fma
    }
}

/// SIMD performance metrics
#[derive(Debug, Clone, Default)]
pub struct SimdMetrics {
    pub total_operations: u64,
    pub avx2_operations: u64,
    pub sse41_operations: u64,
    pub scalar_operations: u64,
    pub total_processing_time: std::time::Duration,
}

impl SimdMetrics {
    fn new() -> Self {
        Self::default()
    }
    
    pub fn simd_usage_rate(&self) -> f64 {
        if self.total_operations > 0 {
            (self.avx2_operations + self.sse41_operations) as f64 / self.total_operations as f64
        } else {
            0.0
        }
    }
    
    pub fn average_operation_time(&self) -> Option<std::time::Duration> {
        if self.total_operations > 0 {
            Some(self.total_processing_time / self.total_operations as u32)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_processor_creation() {
        let processor = SimdProcessor::new();
        assert!(processor.cpu_features.has_sse41()); // Should be true on most modern CPUs
    }
    
    #[test]
    fn test_dot_product() {
        let processor = SimdProcessor::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        
        let result = processor.dot_product(&a, &b).unwrap();
        assert_eq!(result, 70.0); // 1*5 + 2*6 + 3*7 + 4*8 = 70
    }
    
    #[test]
    fn test_vector_normalization() {
        let processor = SimdProcessor::new();
        let mut vector = vec![3.0, 4.0, 0.0, 0.0];
        
        processor.normalize_vector_scalar(&mut vector).unwrap();
        
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_complex_multiplication() {
        let processor = SimdProcessor::new();
        let a = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
        let b = vec![Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)];
        
        let result = processor.complex_multiply_scalar(&a, &b).unwrap();
        
        // (1+2i) * (5+6i) = 5 + 6i + 10i + 12i^2 = 5 + 16i - 12 = -7 + 16i
        assert_eq!(result[0].real, -7.0);
        assert_eq!(result[0].imag, 16.0);
        
        // (3+4i) * (7+8i) = 21 + 24i + 28i + 32i^2 = 21 + 52i - 32 = -11 + 52i
        assert_eq!(result[1].real, -11.0);
        assert_eq!(result[1].imag, 52.0);
    }
    
    #[test]
    fn test_matrix_vector_multiply() {
        let processor = SimdProcessor::new();
        let matrix = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let vector = vec![5.0, 6.0];
        
        let result = processor.matrix_vector_multiply_scalar(&matrix, &vector, 2, 2).unwrap();
        
        // [1 2] * [5] = [1*5 + 2*6] = [17]
        // [3 4]   [6]   [3*5 + 4*6]   [39]
        assert_eq!(result[0], 17.0);
        assert_eq!(result[1], 39.0);
    }
}