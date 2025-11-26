//! Tensor abstraction over multiple backends
//!
//! Provides a unified tensor API that routes to the appropriate backend
//! (CPU, CUDA, Metal, ROCm, Vulkan, WebGPU).

use crate::backends::{Backend, BackendType, Device};
use crate::error::{MlError, MlResult};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Tensor data type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
    /// 16-bit floating point (half precision)
    F16,
    /// 16-bit brain floating point
    BF16,
    /// 8-bit signed integer
    I8,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
}

impl DType {
    /// Size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
            Self::F16 | Self::BF16 => 2,
            Self::I8 => 1,
        }
    }
}

impl Default for DType {
    fn default() -> Self {
        Self::F32
    }
}

/// Tensor shape
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape(Vec<usize>);

impl Shape {
    /// Create a new shape
    pub fn new(dims: Vec<usize>) -> Self {
        Self(dims)
    }

    /// Get dimensions
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Number of dimensions
    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    /// Total number of elements
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    /// Check if shapes are broadcastable
    pub fn is_broadcastable_with(&self, other: &Shape) -> bool {
        let (longer, shorter) = if self.ndim() >= other.ndim() {
            (&self.0, &other.0)
        } else {
            (&other.0, &self.0)
        };

        let offset = longer.len() - shorter.len();
        for (i, &dim) in shorter.iter().enumerate() {
            let other_dim = longer[i + offset];
            if dim != other_dim && dim != 1 && other_dim != 1 {
                return false;
            }
        }
        true
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", d)?;
        }
        write!(f, "]")
    }
}

/// Tensor operations trait
pub trait TensorOps {
    /// Get the shape
    fn shape(&self) -> &Shape;

    /// Get the data type
    fn dtype(&self) -> DType;

    /// Get the device
    fn device(&self) -> &Device;

    /// Number of elements
    fn numel(&self) -> usize {
        self.shape().numel()
    }

    /// Number of dimensions
    fn ndim(&self) -> usize {
        self.shape().ndim()
    }

    /// Size in bytes
    fn size_bytes(&self) -> usize {
        self.numel() * self.dtype().size_bytes()
    }
}

/// Backend-agnostic tensor
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Tensor shape
    shape: Shape,
    /// Data type
    dtype: DType,
    /// Device where tensor resides
    device: Device,
    /// Raw data (CPU)
    #[cfg(feature = "cpu")]
    data_cpu: Option<Arc<Vec<f32>>>,
    /// Placeholder for GPU data pointers
    #[cfg(any(feature = "cuda", feature = "metal", feature = "rocm", feature = "vulkan", feature = "wgpu"))]
    data_gpu: Option<usize>, // Would be actual GPU pointer in real implementation
}

impl Tensor {
    /// Create a new tensor filled with zeros
    pub fn zeros(shape: impl Into<Shape>, dtype: DType, device: &Device) -> MlResult<Self> {
        let shape = shape.into();
        let numel = shape.numel();

        #[cfg(feature = "cpu")]
        {
            return Ok(Self {
                shape,
                dtype,
                device: device.clone(),
                data_cpu: Some(Arc::new(vec![0.0; numel])),
                #[cfg(any(feature = "cuda", feature = "metal", feature = "rocm", feature = "vulkan", feature = "wgpu"))]
                data_gpu: None,
            });
        }

        // GPU tensor creation would go here
        #[cfg(all(not(feature = "cpu"), any(feature = "cuda", feature = "metal", feature = "rocm", feature = "vulkan", feature = "wgpu")))]
        {
            Ok(Self {
                shape,
                dtype,
                device: device.clone(),
                data_cpu: None,
                data_gpu: Some(0), // Placeholder
            })
        }

        #[cfg(not(any(feature = "cpu", feature = "cuda", feature = "metal", feature = "rocm", feature = "vulkan", feature = "wgpu")))]
        Err(MlError::feature_not_enabled("cpu or gpu backend"))
    }

    /// Create a new tensor filled with ones
    pub fn ones(shape: impl Into<Shape>, dtype: DType, device: &Device) -> MlResult<Self> {
        let shape = shape.into();
        let numel = shape.numel();

        #[cfg(feature = "cpu")]
        if device.is_cpu() {
            return Ok(Self {
                shape,
                dtype,
                device: device.clone(),
                data_cpu: Some(Arc::new(vec![1.0; numel])),
                #[cfg(any(feature = "cuda", feature = "metal", feature = "rocm", feature = "vulkan", feature = "wgpu"))]
                data_gpu: None,
            });
        }

        Self::zeros(shape, dtype, device)
    }

    /// Create tensor from slice
    pub fn from_slice(data: &[f32], shape: impl Into<Shape>, device: &Device) -> MlResult<Self> {
        let shape = shape.into();
        if data.len() != shape.numel() {
            return Err(MlError::shape_mismatch(
                vec![shape.numel()],
                vec![data.len()],
            ));
        }

        #[cfg(feature = "cpu")]
        {
            return Ok(Self {
                shape,
                dtype: DType::F32,
                device: device.clone(),
                data_cpu: Some(Arc::new(data.to_vec())),
                #[cfg(any(feature = "cuda", feature = "metal", feature = "rocm", feature = "vulkan", feature = "wgpu"))]
                data_gpu: None,
            });
        }

        // GPU path: copy data to GPU
        #[cfg(all(not(feature = "cpu"), any(feature = "cuda", feature = "metal", feature = "rocm", feature = "vulkan", feature = "wgpu")))]
        {
            Ok(Self {
                shape,
                dtype: DType::F32,
                device: device.clone(),
                data_cpu: None,
                data_gpu: Some(0), // Would allocate and copy
            })
        }

        #[cfg(not(any(feature = "cpu", feature = "cuda", feature = "metal", feature = "rocm", feature = "vulkan", feature = "wgpu")))]
        Err(MlError::feature_not_enabled("cpu or gpu backend"))
    }

    /// Create tensor with random uniform values
    pub fn rand(shape: impl Into<Shape>, device: &Device) -> MlResult<Self> {
        use rand::Rng;
        let shape = shape.into();
        let numel = shape.numel();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..numel).map(|_| rng.gen()).collect();
        Self::from_slice(&data, shape, device)
    }

    /// Create tensor with random normal values
    pub fn randn(shape: impl Into<Shape>, device: &Device) -> MlResult<Self> {
        use rand::Rng;
        use rand_distr::StandardNormal;
        let shape = shape.into();
        let numel = shape.numel();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..numel).map(|_| rng.sample(StandardNormal)).collect();
        Self::from_slice(&data, shape, device)
    }

    /// Get data as slice (CPU only)
    #[cfg(feature = "cpu")]
    pub fn as_slice(&self) -> Option<&[f32]> {
        self.data_cpu.as_ref().map(|d| d.as_slice())
    }

    /// Get data as mutable slice (CPU only)
    /// Note: This creates a new mutable copy due to Arc
    #[cfg(feature = "cpu")]
    pub fn as_slice_mut(&mut self) -> Option<&mut [f32]> {
        self.data_cpu.as_mut().and_then(|arc| Arc::get_mut(arc).map(|v| v.as_mut_slice()))
    }

    /// Copy tensor to CPU
    pub fn to_cpu(&self) -> MlResult<Self> {
        if self.device.is_cpu() {
            return Ok(self.clone());
        }

        // GPU -> CPU copy would go here
        #[cfg(feature = "cpu")]
        {
            Ok(Self {
                shape: self.shape.clone(),
                dtype: self.dtype,
                device: Device::Cpu,
                data_cpu: Some(Arc::new(vec![0.0; self.numel()])), // Would copy from GPU
                #[cfg(any(feature = "cuda", feature = "metal", feature = "rocm", feature = "vulkan", feature = "wgpu"))]
                data_gpu: None,
            })
        }

        #[cfg(not(feature = "cpu"))]
        Err(MlError::feature_not_enabled("cpu"))
    }

    /// Copy tensor to specified device
    pub fn to_device(&self, device: &Device) -> MlResult<Self> {
        if &self.device == device {
            return Ok(self.clone());
        }

        // Cross-device copy would go here
        Self::zeros(self.shape.clone(), self.dtype, device)
    }

    /// Reshape tensor
    pub fn reshape(&self, new_shape: impl Into<Shape>) -> MlResult<Self> {
        let new_shape = new_shape.into();
        if new_shape.numel() != self.numel() {
            return Err(MlError::shape_mismatch(
                new_shape.dims().to_vec(),
                self.shape.dims().to_vec(),
            ));
        }

        Ok(Self {
            shape: new_shape,
            dtype: self.dtype,
            device: self.device.clone(),
            #[cfg(feature = "cpu")]
            data_cpu: self.data_cpu.clone(),
            #[cfg(any(feature = "cuda", feature = "metal", feature = "rocm", feature = "vulkan", feature = "wgpu"))]
            data_gpu: self.data_gpu,
        })
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &Tensor) -> MlResult<Self> {
        // Shape validation
        let a_shape = self.shape();
        let b_shape = other.shape();

        if a_shape.ndim() < 2 || b_shape.ndim() < 2 {
            return Err(MlError::InvalidConfig(
                "matmul requires at least 2D tensors".to_string(),
            ));
        }

        let a_dims = a_shape.dims();
        let b_dims = b_shape.dims();
        let m = a_dims[a_dims.len() - 2];
        let k = a_dims[a_dims.len() - 1];
        let k2 = b_dims[b_dims.len() - 2];
        let n = b_dims[b_dims.len() - 1];

        if k != k2 {
            return Err(MlError::shape_mismatch(vec![m, k], vec![k2, n]));
        }

        let result_shape = vec![m, n];
        Self::zeros(result_shape, self.dtype, &self.device)
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> MlResult<Self> {
        if !self.shape.is_broadcastable_with(&other.shape) {
            return Err(MlError::shape_mismatch(
                self.shape.dims().to_vec(),
                other.shape.dims().to_vec(),
            ));
        }
        Self::zeros(self.shape.clone(), self.dtype, &self.device)
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &Tensor) -> MlResult<Self> {
        if !self.shape.is_broadcastable_with(&other.shape) {
            return Err(MlError::shape_mismatch(
                self.shape.dims().to_vec(),
                other.shape.dims().to_vec(),
            ));
        }
        Self::zeros(self.shape.clone(), self.dtype, &self.device)
    }

    /// Apply ReLU activation
    pub fn relu(&self) -> MlResult<Self> {
        #[cfg(feature = "cpu")]
        if let Some(data) = &self.data_cpu {
            let result: Vec<f32> = data.iter().map(|&x| x.max(0.0)).collect();
            return Self::from_slice(&result, self.shape.clone(), &self.device);
        }
        Self::zeros(self.shape.clone(), self.dtype, &self.device)
    }

    /// Apply sigmoid activation
    pub fn sigmoid(&self) -> MlResult<Self> {
        #[cfg(feature = "cpu")]
        if let Some(data) = &self.data_cpu {
            let result: Vec<f32> = data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
            return Self::from_slice(&result, self.shape.clone(), &self.device);
        }
        Self::zeros(self.shape.clone(), self.dtype, &self.device)
    }

    /// Apply tanh activation
    pub fn tanh(&self) -> MlResult<Self> {
        #[cfg(feature = "cpu")]
        if let Some(data) = &self.data_cpu {
            let result: Vec<f32> = data.iter().map(|&x| x.tanh()).collect();
            return Self::from_slice(&result, self.shape.clone(), &self.device);
        }
        Self::zeros(self.shape.clone(), self.dtype, &self.device)
    }

    /// Apply softmax along last dimension
    pub fn softmax(&self) -> MlResult<Self> {
        #[cfg(feature = "cpu")]
        if let Some(data) = &self.data_cpu {
            let last_dim = *self.shape.dims().last().unwrap_or(&1);
            let mut result = vec![0.0f32; data.len()];

            for chunk_start in (0..data.len()).step_by(last_dim) {
                let chunk = &data[chunk_start..chunk_start + last_dim];
                let max_val = chunk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = chunk.iter().map(|&x| (x - max_val).exp()).sum();

                for (i, &x) in chunk.iter().enumerate() {
                    result[chunk_start + i] = (x - max_val).exp() / exp_sum;
                }
            }

            return Self::from_slice(&result, self.shape.clone(), &self.device);
        }
        Self::zeros(self.shape.clone(), self.dtype, &self.device)
    }

    /// Transpose last two dimensions
    pub fn transpose(&self) -> MlResult<Self> {
        let mut new_dims = self.shape.dims().to_vec();
        let ndim = new_dims.len();
        if ndim >= 2 {
            new_dims.swap(ndim - 1, ndim - 2);
        }
        Self::zeros(new_dims, self.dtype, &self.device)
    }

    /// Sum all elements
    pub fn sum(&self) -> MlResult<f32> {
        #[cfg(feature = "cpu")]
        if let Some(data) = &self.data_cpu {
            return Ok(data.iter().sum());
        }
        Ok(0.0)
    }

    /// Mean of all elements
    pub fn mean(&self) -> MlResult<f32> {
        let sum = self.sum()?;
        Ok(sum / self.numel() as f32)
    }

    /// Slice tensor along a dimension
    pub fn slice(&self, dim: usize, start: usize, end: usize) -> MlResult<Self> {
        if dim >= self.shape.ndim() {
            return Err(MlError::DimensionMismatch {
                expected: format!("dim < {}", self.shape.ndim()),
                got: format!("dim = {}", dim),
            });
        }

        let dims = self.shape.dims();
        let dim_size = dims[dim];
        if end > dim_size || start >= end {
            return Err(MlError::InvalidConfig(format!(
                "Invalid slice range [{}:{}] for dimension {} with size {}",
                start, end, dim, dim_size
            )));
        }

        // Calculate new shape
        let mut new_dims = dims.to_vec();
        new_dims[dim] = end - start;
        let new_shape = Shape::new(new_dims.clone());

        #[cfg(feature = "cpu")]
        if let Some(data) = &self.data_cpu {
            // Calculate strides
            let strides: Vec<usize> = dims.iter().rev()
                .scan(1, |acc, &d| {
                    let s = *acc;
                    *acc *= d;
                    Some(s)
                })
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();

            let stride = strides[dim];
            let outer_size: usize = dims[..dim].iter().product();
            let inner_size: usize = dims[dim + 1..].iter().product();
            let slice_len = (end - start) * inner_size;

            let mut result = Vec::with_capacity(new_shape.numel());

            for outer in 0..outer_size.max(1) {
                let base = outer * dim_size * inner_size;
                let slice_start = base + start * inner_size;
                result.extend_from_slice(&data[slice_start..slice_start + slice_len]);
            }

            return Self::from_slice(&result, new_shape, &self.device);
        }

        Self::zeros(new_shape, self.dtype, &self.device)
    }

    /// Transpose specific dimensions
    pub fn transpose_dims(&self, dim0: usize, dim1: usize) -> MlResult<Self> {
        let ndim = self.shape.ndim();
        if dim0 >= ndim || dim1 >= ndim {
            return Err(MlError::DimensionMismatch {
                expected: format!("dims < {}", ndim),
                got: format!("dim0={}, dim1={}", dim0, dim1),
            });
        }

        let mut new_dims = self.shape.dims().to_vec();
        new_dims.swap(dim0, dim1);

        // For CPU implementation, we need to actually transpose the data
        #[cfg(feature = "cpu")]
        if let Some(data) = &self.data_cpu {
            let old_dims = self.shape.dims();
            let mut result = vec![0.0f32; data.len()];

            // Calculate old and new strides
            let old_strides: Vec<usize> = old_dims.iter().rev()
                .scan(1, |acc, &d| { let s = *acc; *acc *= d; Some(s) })
                .collect::<Vec<_>>().into_iter().rev().collect();
            let new_strides: Vec<usize> = new_dims.iter().rev()
                .scan(1, |acc, &d| { let s = *acc; *acc *= d; Some(s) })
                .collect::<Vec<_>>().into_iter().rev().collect();

            // Transpose by iterating over all elements
            for old_idx in 0..data.len() {
                // Convert linear index to multi-dimensional
                let mut remaining = old_idx;
                let mut coords: Vec<usize> = vec![0; ndim];
                for d in 0..ndim {
                    coords[d] = remaining / old_strides[d];
                    remaining %= old_strides[d];
                }

                // Swap coordinates
                coords.swap(dim0, dim1);

                // Convert back to linear index with new strides
                let new_idx: usize = coords.iter().zip(new_strides.iter())
                    .map(|(&c, &s)| c * s).sum();

                result[new_idx] = data[old_idx];
            }

            return Self::from_slice(&result, new_dims, &self.device);
        }

        Self::zeros(new_dims, self.dtype, &self.device)
    }

    /// Stack multiple tensors along a new dimension
    pub fn stack(tensors: &[&Self], dim: usize) -> MlResult<Self> {
        if tensors.is_empty() {
            return Err(MlError::InsufficientData {
                required: 1,
                actual: 0,
            });
        }

        let first_shape = tensors[0].shape();
        let device = tensors[0].device();

        // Validate all tensors have the same shape
        for t in tensors.iter().skip(1) {
            if t.shape() != first_shape {
                return Err(MlError::ShapeMismatch {
                    expected: first_shape.dims().to_vec(),
                    actual: t.shape().dims().to_vec(),
                });
            }
        }

        // Calculate new shape
        let mut new_dims = first_shape.dims().to_vec();
        new_dims.insert(dim, tensors.len());

        #[cfg(feature = "cpu")]
        {
            let mut result = Vec::with_capacity(new_dims.iter().product());
            for t in tensors {
                if let Some(data) = t.as_slice() {
                    result.extend_from_slice(data);
                }
            }
            return Self::from_slice(&result, new_dims, device);
        }

        #[cfg(not(feature = "cpu"))]
        Self::zeros(new_dims, DType::F32, device)
    }

    /// Stack multiple tensors along a new dimension (convenience for Vec<Tensor>)
    pub fn stack_vec(tensors: &[Tensor], dim: usize) -> MlResult<Self> {
        let refs: Vec<&Self> = tensors.iter().collect();
        Self::stack(&refs, dim)
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Self) -> MlResult<Self> {
        if self.shape() != other.shape() {
            return Err(MlError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                actual: other.shape().dims().to_vec(),
            });
        }

        #[cfg(feature = "cpu")]
        if let (Some(a), Some(b)) = (&self.data_cpu, &other.data_cpu) {
            let result: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
            return Self::from_slice(&result, self.shape.clone(), &self.device);
        }

        Self::zeros(self.shape.clone(), self.dtype, &self.device)
    }

    /// Remove dimensions of size 1
    pub fn squeeze(&self) -> MlResult<Self> {
        let new_dims: Vec<usize> = self.shape.dims()
            .iter()
            .copied()
            .filter(|&d| d != 1)
            .collect();

        // If all dimensions are 1, keep at least one
        let new_dims = if new_dims.is_empty() { vec![1] } else { new_dims };

        #[cfg(feature = "cpu")]
        if let Some(data) = &self.data_cpu {
            return Self::from_slice(data, new_dims, &self.device);
        }

        Self::zeros(new_dims, self.dtype, &self.device)
    }

    /// Squeeze a specific dimension
    pub fn squeeze_dim(&self, dim: usize) -> MlResult<Self> {
        let dims = self.shape.dims();
        if dim >= dims.len() {
            return Err(MlError::DimensionMismatch {
                expected: format!("dim < {}", dims.len()),
                got: format!("dim = {}", dim),
            });
        }

        if dims[dim] != 1 {
            // Nothing to squeeze, return clone
            return Ok(self.clone());
        }

        let mut new_dims = dims.to_vec();
        new_dims.remove(dim);
        if new_dims.is_empty() {
            new_dims.push(1);
        }

        #[cfg(feature = "cpu")]
        if let Some(data) = &self.data_cpu {
            return Self::from_slice(data, new_dims, &self.device);
        }

        Self::zeros(new_dims, self.dtype, &self.device)
    }

    /// Multiply by a scalar
    pub fn mul_scalar(&self, scalar: f32) -> MlResult<Self> {
        #[cfg(feature = "cpu")]
        if let Some(data) = &self.data_cpu {
            let result: Vec<f32> = data.iter().map(|&x| x * scalar).collect();
            return Self::from_slice(&result, self.shape.clone(), &self.device);
        }
        Self::zeros(self.shape.clone(), self.dtype, &self.device)
    }

    /// Add a dimension at the specified position
    pub fn unsqueeze(&self, dim: usize) -> MlResult<Self> {
        let mut new_dims = self.shape.dims().to_vec();
        if dim > new_dims.len() {
            return Err(MlError::DimensionMismatch {
                expected: format!("dim <= {}", new_dims.len()),
                got: format!("dim = {}", dim),
            });
        }
        new_dims.insert(dim, 1);

        // Data doesn't change, just the shape
        #[cfg(feature = "cpu")]
        if let Some(data) = &self.data_cpu {
            return Self::from_slice(data, new_dims, &self.device);
        }

        Self::zeros(new_dims, self.dtype, &self.device)
    }

    /// Softplus activation: log(1 + exp(x))
    pub fn softplus(&self) -> MlResult<Self> {
        #[cfg(feature = "cpu")]
        if let Some(data) = &self.data_cpu {
            let result: Vec<f32> = data.iter().map(|&x| {
                if x > 20.0 {
                    x // Prevent overflow
                } else if x < -20.0 {
                    0.0
                } else {
                    (1.0 + x.exp()).ln()
                }
            }).collect();
            return Self::from_slice(&result, self.shape.clone(), &self.device);
        }
        Self::zeros(self.shape.clone(), self.dtype, &self.device)
    }

    /// 1D padding (for convolutions)
    pub fn pad_1d(&self, padding: usize) -> MlResult<Self> {
        let dims = self.shape.dims();
        if dims.is_empty() {
            return Err(MlError::InvalidConfig("Cannot pad empty tensor".to_string()));
        }

        let mut new_dims = dims.to_vec();
        let last_idx = new_dims.len() - 1;
        new_dims[last_idx] += 2 * padding;

        #[cfg(feature = "cpu")]
        if let Some(data) = &self.data_cpu {
            let old_last_dim = dims[last_idx];
            let new_last_dim = new_dims[last_idx];
            let num_sequences: usize = dims[..last_idx].iter().product::<usize>().max(1);

            let mut result = vec![0.0f32; new_dims.iter().product()];

            for seq in 0..num_sequences {
                let old_start = seq * old_last_dim;
                let new_start = seq * new_last_dim + padding;
                result[new_start..new_start + old_last_dim]
                    .copy_from_slice(&data[old_start..old_start + old_last_dim]);
            }

            return Self::from_slice(&result, new_dims, &self.device);
        }

        Self::zeros(new_dims, self.dtype, &self.device)
    }
}

impl TensorOps for Tensor {
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let device = Device::Cpu;
        let t = Tensor::zeros(vec![2, 3], DType::F32, &device).unwrap();
        assert_eq!(t.shape().dims(), &[2, 3]);
        assert_eq!(t.numel(), 6);
    }

    #[test]
    fn test_tensor_from_slice() {
        let device = Device::Cpu;
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_slice(&data, vec![2, 3], &device).unwrap();
        assert_eq!(t.shape().dims(), &[2, 3]);

        #[cfg(feature = "cpu")]
        {
            let slice = t.as_slice().unwrap();
            assert_eq!(slice, &data);
        }
    }

    #[test]
    fn test_tensor_reshape() {
        let device = Device::Cpu;
        let t = Tensor::zeros(vec![2, 3], DType::F32, &device).unwrap();
        let reshaped = t.reshape(vec![3, 2]).unwrap();
        assert_eq!(reshaped.shape().dims(), &[3, 2]);
    }

    #[test]
    fn test_shape_broadcast() {
        let s1 = Shape::new(vec![1, 3, 4]);
        let s2 = Shape::new(vec![2, 3, 4]);
        assert!(s1.is_broadcastable_with(&s2));

        let s3 = Shape::new(vec![2, 5, 4]);
        assert!(!s1.is_broadcastable_with(&s3));
    }
}
