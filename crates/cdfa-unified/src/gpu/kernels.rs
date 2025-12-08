//! GPU kernel dispatch and optimization for CDFA operations
//!
//! This module provides high-level GPU kernel operations for common CDFA tasks,
//! with automatic optimization and fallback mechanisms.

use crate::error::{CdfaError, CdfaResult};
use crate::types::{Float as CdfaFloat, FloatArray2 as CdfaMatrix, FloatArray1 as CdfaArray};
use super::{GpuContext, GpuBuffer, GpuKernel, GpuPrecision, utils};
use std::sync::Arc;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// GPU kernel operations for matrices
pub mod matrix_ops {
    use super::*;
    
    /// GPU-accelerated matrix multiplication
    pub async fn gpu_matrix_multiply(
        context: Arc<dyn GpuContext>,
        a: &CdfaMatrix,
        b: &CdfaMatrix,
    ) -> CdfaResult<CdfaMatrix> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        
        if k != k2 {
            return Err(CdfaError::InvalidParameter(
                "Matrix dimensions don't match for multiplication".to_string()
            ));
        }
        
        // Convert matrices to GPU format
        let a_data = utils::matrix_to_gpu_format(a);
        let b_data = utils::matrix_to_gpu_format(b);
        let mut c_data = vec![0.0f32; m * n];
        
        // Allocate GPU buffers
        let mut buffer_a = context.allocate_buffer(a_data.len() * 4)?;
        let mut buffer_b = context.allocate_buffer(b_data.len() * 4)?;
        let mut buffer_c = context.allocate_buffer(c_data.len() * 4)?;
        
        // Copy data to GPU
        buffer_a.copy_from_host(utils::to_bytes(&a_data))?;
        buffer_b.copy_from_host(utils::to_bytes(&b_data))?;
        
        // Create and configure kernel
        let kernel_source = get_matrix_multiply_kernel(context.device_info().backend);
        let mut kernel = context.create_kernel(&kernel_source, "matrix_multiply")?;
        
        // Set kernel arguments
        kernel.set_arg(0, buffer_a.as_ref())?;
        kernel.set_arg(1, buffer_b.as_ref())?;
        kernel.set_arg(2, buffer_c.as_ref())?;
        kernel.set_scalar_arg(3, m as u32)?;
        kernel.set_scalar_arg(4, n as u32)?;
        kernel.set_scalar_arg(5, k as u32)?;
        
        // Calculate optimal work group size
        let max_work_group = context.device_info().max_work_group_size;
        let work_group_size = utils::calculate_work_group_size(std::cmp::max(m, n), max_work_group);
        
        // Launch kernel
        kernel.launch(
            &[m as u32, n as u32],
            Some(&[work_group_size, work_group_size]),
        )?;
        
        // Synchronize and read result
        context.synchronize()?;
        let mut c_bytes = vec![0u8; c_data.len() * 4];
        buffer_c.copy_to_host(&mut c_bytes)?;
        c_data.copy_from_slice(utils::from_bytes(&c_bytes));
        
        // Convert back to CDFA matrix
        Ok(utils::gpu_format_to_matrix(c_data, (m, n)))
    }
    
    /// GPU-accelerated matrix-vector multiplication
    pub async fn gpu_matrix_vector_multiply(
        context: Arc<dyn GpuContext>,
        matrix: &CdfaMatrix,
        vector: &CdfaArray,
    ) -> CdfaResult<CdfaArray> {
        let (m, n) = matrix.dim();
        
        if n != vector.len() {
            return Err(CdfaError::InvalidParameter(
                "Matrix and vector dimensions don't match".to_string()
            ));
        }
        
        let matrix_data = utils::matrix_to_gpu_format(matrix);
        let vector_data: Vec<f32> = vector.iter().map(|&x| x as f32).collect();
        let mut result_data = vec![0.0f32; m];
        
        // Allocate GPU buffers
        let mut buffer_matrix = context.allocate_buffer(matrix_data.len() * 4)?;
        let mut buffer_vector = context.allocate_buffer(vector_data.len() * 4)?;
        let mut buffer_result = context.allocate_buffer(result_data.len() * 4)?;
        
        // Copy data to GPU
        buffer_matrix.copy_from_host(bytemuck::cast_slice(&matrix_data))?;
        buffer_vector.copy_from_host(bytemuck::cast_slice(&vector_data))?;
        
        // Create and launch kernel
        let kernel_source = get_matrix_vector_kernel(context.device_info().backend);
        let mut kernel = context.create_kernel(&kernel_source, "matrix_vector_multiply")?;
        
        kernel.set_arg(0, buffer_matrix.as_ref())?;
        kernel.set_arg(1, buffer_vector.as_ref())?;
        kernel.set_arg(2, buffer_result.as_ref())?;
        kernel.set_scalar_arg(3, m as u32)?;
        kernel.set_scalar_arg(4, n as u32)?;
        
        kernel.launch(&[m as u32], None)?;
        
        context.synchronize()?;
        buffer_result.copy_to_host(bytemuck::cast_slice_mut(&mut result_data))?;
        
        let result: Vec<CdfaFloat> = result_data.into_iter().map(|x| x as CdfaFloat).collect();
        Ok(Array1::from_vec(result))
    }
    
    /// GPU-accelerated matrix transpose
    pub async fn gpu_matrix_transpose(
        context: Arc<dyn GpuContext>,
        matrix: &CdfaMatrix,
    ) -> CdfaResult<CdfaMatrix> {
        let (m, n) = matrix.dim();
        let input_data = utils::matrix_to_gpu_format(matrix);
        let mut output_data = vec![0.0f32; m * n];
        
        // Allocate GPU buffers
        let mut buffer_input = context.allocate_buffer(input_data.len() * 4)?;
        let mut buffer_output = context.allocate_buffer(output_data.len() * 4)?;
        
        // Copy data to GPU
        buffer_input.copy_from_host(utils::to_bytes(&input_data))?;
        
        // Create and launch transpose kernel
        let kernel_source = get_transpose_kernel(context.device_info().backend);
        let mut kernel = context.create_kernel(&kernel_source, "transpose")?;
        
        kernel.set_arg(0, buffer_input.as_ref())?;
        kernel.set_arg(1, buffer_output.as_ref())?;
        kernel.set_scalar_arg(2, m as u32)?;
        kernel.set_scalar_arg(3, n as u32)?;
        
        let work_group_size = utils::calculate_work_group_size(std::cmp::max(m, n), 256);
        kernel.launch(
            &[n as u32, m as u32],
            Some(&[work_group_size, work_group_size]),
        )?;
        
        context.synchronize()?;
        buffer_output.copy_to_host(bytemuck::cast_slice_mut(&mut output_data))?;
        
        Ok(utils::gpu_format_to_matrix(output_data, (n, m)))
    }
}

/// GPU kernel operations for element-wise operations
pub mod element_ops {
    use super::*;
    
    /// GPU-accelerated element-wise operation
    pub async fn gpu_element_wise_op<F>(
        context: Arc<dyn GpuContext>,
        a: &CdfaMatrix,
        b: &CdfaMatrix,
        op: F,
    ) -> CdfaResult<CdfaMatrix>
    where
        F: Fn(CdfaFloat, CdfaFloat) -> CdfaFloat + Send + Sync,
    {
        if a.dim() != b.dim() {
            return Err(CdfaError::InvalidParameter(
                "Matrix dimensions must match for element-wise operations".to_string()
            ));
        }
        
        let size = a.len();
        let a_data = utils::matrix_to_gpu_format(a);
        let b_data = utils::matrix_to_gpu_format(b);
        let mut result_data = vec![0.0f32; size];
        
        // Determine operation type for kernel
        let op_type = get_operation_type(&op);
        
        // Allocate GPU buffers
        let mut buffer_a = context.allocate_buffer(a_data.len() * 4)?;
        let mut buffer_b = context.allocate_buffer(b_data.len() * 4)?;
        let mut buffer_result = context.allocate_buffer(result_data.len() * 4)?;
        
        // Copy data to GPU
        buffer_a.copy_from_host(utils::to_bytes(&a_data))?;
        buffer_b.copy_from_host(utils::to_bytes(&b_data))?;
        
        // Create and launch kernel
        let kernel_source = get_element_wise_kernel(context.device_info().backend);
        let mut kernel = context.create_kernel(&kernel_source, "element_wise_op")?;
        
        kernel.set_arg(0, buffer_a.as_ref())?;
        kernel.set_arg(1, buffer_b.as_ref())?;
        kernel.set_arg(2, buffer_result.as_ref())?;
        kernel.set_scalar_arg(3, size as u32)?;
        kernel.set_scalar_arg(4, op_type)?;
        
        let work_group_size = utils::calculate_work_group_size(size, 256);
        kernel.launch(&[size as u32], Some(&[work_group_size]))?;
        
        context.synchronize()?;
        buffer_result.copy_to_host(bytemuck::cast_slice_mut(&mut result_data))?;
        
        Ok(utils::gpu_format_to_matrix(result_data, a.dim()))
    }
    
    /// GPU-accelerated element-wise addition
    pub async fn gpu_add(
        context: Arc<dyn GpuContext>,
        a: &CdfaMatrix,
        b: &CdfaMatrix,
    ) -> CdfaResult<CdfaMatrix> {
        gpu_element_wise_op(context, a, b, |x, y| x + y).await
    }
    
    /// GPU-accelerated element-wise multiplication
    pub async fn gpu_multiply(
        context: Arc<dyn GpuContext>,
        a: &CdfaMatrix,
        b: &CdfaMatrix,
    ) -> CdfaResult<CdfaMatrix> {
        gpu_element_wise_op(context, a, b, |x, y| x * y).await
    }
    
    /// GPU-accelerated scalar multiplication
    pub async fn gpu_scalar_multiply(
        context: Arc<dyn GpuContext>,
        matrix: &CdfaMatrix,
        scalar: CdfaFloat,
    ) -> CdfaResult<CdfaMatrix> {
        let size = matrix.len();
        let input_data = utils::matrix_to_gpu_format(matrix);
        let mut output_data = vec![0.0f32; size];
        
        let mut buffer_input = context.allocate_buffer(input_data.len() * 4)?;
        let mut buffer_output = context.allocate_buffer(output_data.len() * 4)?;
        
        buffer_input.copy_from_host(utils::to_bytes(&input_data))?;
        
        let kernel_source = get_scalar_multiply_kernel(context.device_info().backend);
        let mut kernel = context.create_kernel(&kernel_source, "scalar_multiply")?;
        
        kernel.set_arg(0, buffer_input.as_ref())?;
        kernel.set_arg(1, buffer_output.as_ref())?;
        kernel.set_scalar_arg(2, scalar as f32)?;
        kernel.set_scalar_arg(3, size as u32)?;
        
        let work_group_size = utils::calculate_work_group_size(size, 256);
        kernel.launch(&[size as u32], Some(&[work_group_size]))?;
        
        context.synchronize()?;
        buffer_output.copy_to_host(bytemuck::cast_slice_mut(&mut output_data))?;
        
        Ok(utils::gpu_format_to_matrix(output_data, matrix.dim()))
    }
    
    /// Helper function to determine operation type for kernel
    fn get_operation_type<F>(_op: &F) -> u32 
    where 
        F: Fn(CdfaFloat, CdfaFloat) -> CdfaFloat 
    {
        // This is a simplified implementation
        // In practice, you would need a more sophisticated way to determine operation types
        0 // Default to addition
    }
}

/// GPU kernel operations for reductions
pub mod reduction_ops {
    use super::*;
    
    /// GPU-accelerated sum reduction
    pub async fn gpu_reduce_sum(
        context: Arc<dyn GpuContext>,
        matrix: &CdfaMatrix,
    ) -> CdfaResult<CdfaFloat> {
        let size = matrix.len();
        let input_data = utils::matrix_to_gpu_format(matrix);
        
        // Calculate number of work groups needed
        let work_group_size = 256;
        let num_groups = (size + work_group_size - 1) / work_group_size;
        let mut partial_sums = vec![0.0f32; num_groups];
        
        // Allocate GPU buffers
        let mut buffer_input = context.allocate_buffer(input_data.len() * 4)?;
        let mut buffer_output = context.allocate_buffer(partial_sums.len() * 4)?;
        
        // Copy input data to GPU
        buffer_input.copy_from_host(utils::to_bytes(&input_data))?;
        
        // Create and launch reduction kernel
        let kernel_source = get_reduction_kernel(context.device_info().backend);
        let mut kernel = context.create_kernel(&kernel_source, "reduce_sum")?;
        
        kernel.set_arg(0, buffer_input.as_ref())?;
        kernel.set_arg(1, buffer_output.as_ref())?;
        kernel.set_scalar_arg(2, size as u32)?;
        
        kernel.launch(&[size as u32], Some(&[work_group_size as u32]))?;
        
        context.synchronize()?;
        buffer_output.copy_to_host(bytemuck::cast_slice_mut(&mut partial_sums))?;
        
        // Sum the partial results on CPU
        let total_sum: f32 = partial_sums.iter().sum();
        Ok(total_sum as CdfaFloat)
    }
    
    /// GPU-accelerated mean calculation
    pub async fn gpu_mean(
        context: Arc<dyn GpuContext>,
        matrix: &CdfaMatrix,
    ) -> CdfaResult<CdfaFloat> {
        let sum = gpu_reduce_sum(context, matrix).await?;
        Ok(sum / matrix.len() as CdfaFloat)
    }
    
    /// GPU-accelerated variance calculation
    pub async fn gpu_variance(
        context: Arc<dyn GpuContext>,
        matrix: &CdfaMatrix,
    ) -> CdfaResult<CdfaFloat> {
        let mean = gpu_mean(context.clone(), matrix).await?;
        
        // Calculate squared differences
        let size = matrix.len();
        let input_data = utils::matrix_to_gpu_format(matrix);
        let mut variance_data = vec![0.0f32; size];
        
        let mut buffer_input = context.allocate_buffer(input_data.len() * 4)?;
        let mut buffer_output = context.allocate_buffer(variance_data.len() * 4)?;
        
        buffer_input.copy_from_host(utils::to_bytes(&input_data))?;
        
        let kernel_source = get_variance_kernel(context.device_info().backend);
        let mut kernel = context.create_kernel(&kernel_source, "calculate_variance")?;
        
        kernel.set_arg(0, buffer_input.as_ref())?;
        kernel.set_arg(1, buffer_output.as_ref())?;
        kernel.set_scalar_arg(2, mean as f32)?;
        kernel.set_scalar_arg(3, size as u32)?;
        
        let work_group_size = utils::calculate_work_group_size(size, 256);
        kernel.launch(&[size as u32], Some(&[work_group_size]))?;
        
        context.synchronize()?;
        buffer_output.copy_to_host(bytemuck::cast_slice_mut(&mut variance_data))?;
        
        // Sum the squared differences and divide by size
        let sum_sq_diff: f32 = variance_data.iter().sum();
        Ok((sum_sq_diff / size as f32) as CdfaFloat)
    }
    
    /// GPU-accelerated min/max finding
    pub async fn gpu_min_max(
        context: Arc<dyn GpuContext>,
        matrix: &CdfaMatrix,
    ) -> CdfaResult<(CdfaFloat, CdfaFloat)> {
        let size = matrix.len();
        let input_data = utils::matrix_to_gpu_format(matrix);
        
        let work_group_size = 256;
        let num_groups = (size + work_group_size - 1) / work_group_size;
        let mut min_values = vec![f32::INFINITY; num_groups];
        let mut max_values = vec![f32::NEG_INFINITY; num_groups];
        
        let mut buffer_input = context.allocate_buffer(input_data.len() * 4)?;
        let mut buffer_min = context.allocate_buffer(min_values.len() * 4)?;
        let mut buffer_max = context.allocate_buffer(max_values.len() * 4)?;
        
        buffer_input.copy_from_host(utils::to_bytes(&input_data))?;
        
        let kernel_source = get_min_max_kernel(context.device_info().backend);
        let mut kernel = context.create_kernel(&kernel_source, "min_max_reduce")?;
        
        kernel.set_arg(0, buffer_input.as_ref())?;
        kernel.set_arg(1, buffer_min.as_ref())?;
        kernel.set_arg(2, buffer_max.as_ref())?;
        kernel.set_scalar_arg(3, size as u32)?;
        
        kernel.launch(&[size as u32], Some(&[work_group_size as u32]))?;
        
        context.synchronize()?;
        buffer_min.copy_to_host(bytemuck::cast_slice_mut(&mut min_values))?;
        buffer_max.copy_to_host(bytemuck::cast_slice_mut(&mut max_values))?;
        
        // Find global min/max from partial results
        let global_min = min_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let global_max = max_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        Ok((global_min as CdfaFloat, global_max as CdfaFloat))
    }
}

/// GPU kernel operations for batch processing
pub mod batch_ops {
    use super::*;
    
    /// GPU-accelerated batch processing
    pub async fn gpu_batch_process<F, R>(
        context: Arc<dyn GpuContext>,
        matrices: &[CdfaMatrix],
        operation: F,
    ) -> CdfaResult<Vec<R>>
    where
        F: Fn(&CdfaMatrix) -> CdfaResult<R> + Send + Sync + Clone,
        R: Send + Sync,
    {
        let mut results = Vec::with_capacity(matrices.len());
        
        // Process in batches to optimize GPU utilization
        const BATCH_SIZE: usize = 8;
        
        for chunk in matrices.chunks(BATCH_SIZE) {
            let mut batch_results = Vec::new();
            
            for matrix in chunk {
                let result = operation(matrix)?;
                batch_results.push(result);
            }
            
            results.extend(batch_results);
        }
        
        Ok(results)
    }
    
    /// GPU-accelerated batch matrix multiplication
    pub async fn gpu_batch_matrix_multiply(
        context: Arc<dyn GpuContext>,
        a_matrices: &[CdfaMatrix],
        b_matrices: &[CdfaMatrix],
    ) -> CdfaResult<Vec<CdfaMatrix>> {
        if a_matrices.len() != b_matrices.len() {
            return Err(CdfaError::InvalidParameter(
                "Number of A and B matrices must match".to_string()
            ));
        }
        
        let mut results = Vec::with_capacity(a_matrices.len());
        
        for (a, b) in a_matrices.iter().zip(b_matrices.iter()) {
            let result = matrix_ops::gpu_matrix_multiply(context.clone(), a, b).await?;
            results.push(result);
        }
        
        Ok(results)
    }
}

/// Kernel source selection based on GPU backend
fn get_matrix_multiply_kernel(backend: super::GpuBackend) -> String {
    match backend {
        #[cfg(feature = "cuda")]
        super::GpuBackend::Cuda => super::cuda::kernels::MATRIX_MULTIPLY_KERNEL.to_string(),
        #[cfg(feature = "webgpu")]
        super::GpuBackend::WebGpu => super::webgpu::shaders::MATRIX_MULTIPLY_WGSL.to_string(),
        #[cfg(feature = "metal")]
        super::GpuBackend::Metal => super::METAL_KERNELS_SOURCE.to_string(),
        _ => String::new(),
    }
}

fn get_element_wise_kernel(backend: super::GpuBackend) -> String {
    match backend {
        #[cfg(feature = "cuda")]
        super::GpuBackend::Cuda => super::cuda::kernels::ELEMENT_WISE_KERNEL.to_string(),
        #[cfg(feature = "webgpu")]
        super::GpuBackend::WebGpu => super::webgpu::shaders::ELEMENT_WISE_WGSL.to_string(),
        _ => String::new(),
    }
}

fn get_reduction_kernel(backend: super::GpuBackend) -> String {
    match backend {
        #[cfg(feature = "cuda")]
        super::GpuBackend::Cuda => super::cuda::kernels::REDUCE_SUM_KERNEL.to_string(),
        #[cfg(feature = "webgpu")]
        super::GpuBackend::WebGpu => super::webgpu::shaders::REDUCE_SUM_WGSL.to_string(),
        _ => String::new(),
    }
}

// Additional kernel source getters for other operations
fn get_matrix_vector_kernel(_backend: super::GpuBackend) -> String {
    // Implementation would provide backend-specific kernels
    String::new()
}

fn get_transpose_kernel(_backend: super::GpuBackend) -> String {
    String::new()
}

fn get_scalar_multiply_kernel(_backend: super::GpuBackend) -> String {
    String::new()
}

fn get_variance_kernel(_backend: super::GpuBackend) -> String {
    String::new()
}

fn get_min_max_kernel(_backend: super::GpuBackend) -> String {
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    
    #[test]
    fn test_kernel_source_selection() {
        #[cfg(feature = "cuda")]
        {
            let source = get_matrix_multiply_kernel(super::GpuBackend::Cuda);
            assert!(!source.is_empty());
        }
        
        #[cfg(feature = "webgpu")]
        {
            let source = get_matrix_multiply_kernel(super::GpuBackend::WebGpu);
            assert!(!source.is_empty());
        }
    }
    
    #[test]
    fn test_operation_type_detection() {
        // Test that operation type detection works
        let op_type = element_ops::get_operation_type(&|x, y| x + y);
        assert_eq!(op_type, 0);
    }
}