//! Core tensor types for neural computations
//!
//! Provides efficient tensor operations optimized for HFT latency requirements.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

use crate::error::{NeuralError, NeuralResult};

/// Tensor shape descriptor
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorShape(pub Vec<usize>);

impl TensorShape {
    /// Create a new tensor shape
    pub fn new(dims: Vec<usize>) -> Self {
        Self(dims)
    }

    /// Create 1D shape
    pub fn d1(size: usize) -> Self {
        Self(vec![size])
    }

    /// Create 2D shape (batch, features)
    pub fn d2(rows: usize, cols: usize) -> Self {
        Self(vec![rows, cols])
    }

    /// Create 3D shape (batch, sequence, features)
    pub fn d3(batch: usize, seq: usize, features: usize) -> Self {
        Self(vec![batch, seq, features])
    }

    /// Total number of elements
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    /// Number of dimensions
    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    /// Get dimension at index
    pub fn dim(&self, idx: usize) -> Option<usize> {
        self.0.get(idx).copied()
    }

    /// Check if shapes are broadcastable
    pub fn broadcastable(&self, other: &TensorShape) -> bool {
        let a = &self.0;
        let b = &other.0;

        a.iter().rev().zip(b.iter().rev()).all(|(da, db)| {
            *da == *db || *da == 1 || *db == 1
        })
    }
}

impl From<Vec<usize>> for TensorShape {
    fn from(v: Vec<usize>) -> Self {
        Self(v)
    }
}

impl From<&[usize]> for TensorShape {
    fn from(s: &[usize]) -> Self {
        Self(s.to_vec())
    }
}

/// High-performance tensor for neural computations
///
/// Optimized for:
/// - SIMD operations via ndarray
/// - Cache-friendly memory layout
/// - Zero-copy views where possible
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    /// Underlying data storage
    data: Vec<f64>,
    /// Shape of the tensor
    shape: TensorShape,
    /// Stride for each dimension
    strides: Vec<usize>,
}

impl Tensor {
    /// Create tensor from data and shape
    pub fn new(data: Vec<f64>, shape: TensorShape) -> NeuralResult<Self> {
        let expected = shape.numel();
        if data.len() != expected {
            return Err(NeuralError::ShapeMismatch {
                expected: vec![expected],
                actual: vec![data.len()],
            });
        }

        let strides = Self::compute_strides(&shape);
        Ok(Self { data, shape, strides })
    }

    /// Create zeros tensor
    pub fn zeros(shape: TensorShape) -> Self {
        let n = shape.numel();
        let strides = Self::compute_strides(&shape);
        Self {
            data: vec![0.0; n],
            shape,
            strides,
        }
    }

    /// Create ones tensor
    pub fn ones(shape: TensorShape) -> Self {
        let n = shape.numel();
        let strides = Self::compute_strides(&shape);
        Self {
            data: vec![1.0; n],
            shape,
            strides,
        }
    }

    /// Create tensor filled with value
    pub fn full(shape: TensorShape, value: f64) -> Self {
        let n = shape.numel();
        let strides = Self::compute_strides(&shape);
        Self {
            data: vec![value; n],
            shape,
            strides,
        }
    }

    /// Create tensor with random values (uniform [0, 1))
    pub fn rand(shape: TensorShape, rng: &mut impl rand::Rng) -> Self {
        let n = shape.numel();
        let strides = Self::compute_strides(&shape);
        let data: Vec<f64> = (0..n).map(|_| rng.gen()).collect();
        Self { data, shape, strides }
    }

    /// Create tensor with random normal values
    pub fn randn(shape: TensorShape, rng: &mut impl rand::Rng) -> Self {
        use rand_distr::{Distribution, StandardNormal};
        let n = shape.numel();
        let strides = Self::compute_strides(&shape);
        let data: Vec<f64> = (0..n)
            .map(|_| StandardNormal.sample(rng))
            .collect();
        Self { data, shape, strides }
    }

    /// Xavier/Glorot initialization
    pub fn xavier(shape: TensorShape, rng: &mut impl rand::Rng) -> Self {
        use rand_distr::{Distribution, Normal};
        let fan_in = if shape.ndim() >= 2 { shape.dim(0).unwrap_or(1) } else { 1 };
        let fan_out = if shape.ndim() >= 2 { shape.dim(1).unwrap_or(1) } else { 1 };
        let std = (2.0 / (fan_in + fan_out) as f64).sqrt();

        let n = shape.numel();
        let strides = Self::compute_strides(&shape);
        let normal = Normal::new(0.0, std).unwrap();
        let data: Vec<f64> = (0..n).map(|_| normal.sample(rng)).collect();
        Self { data, shape, strides }
    }

    /// He initialization (for ReLU networks)
    pub fn he(shape: TensorShape, rng: &mut impl rand::Rng) -> Self {
        use rand_distr::{Distribution, Normal};
        let fan_in = if shape.ndim() >= 2 { shape.dim(0).unwrap_or(1) } else { 1 };
        let std = (2.0 / fan_in as f64).sqrt();

        let n = shape.numel();
        let strides = Self::compute_strides(&shape);
        let normal = Normal::new(0.0, std).unwrap();
        let data: Vec<f64> = (0..n).map(|_| normal.sample(rng)).collect();
        Self { data, shape, strides }
    }

    fn compute_strides(shape: &TensorShape) -> Vec<usize> {
        let mut strides = vec![1; shape.ndim()];
        for i in (0..shape.ndim().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape.0[i + 1];
        }
        strides
    }

    /// Get tensor shape
    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }

    /// Get raw data
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Get mutable raw data
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Total number of elements
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// View as 1D array
    pub fn as_array1(&self) -> NeuralResult<ArrayView1<'_, f64>> {
        if self.shape.ndim() != 1 {
            return Err(NeuralError::ShapeMismatch {
                expected: vec![self.numel()],
                actual: self.shape.0.clone(),
            });
        }
        Ok(ArrayView1::from(&self.data[..]))
    }

    /// View as 2D array
    pub fn as_array2(&self) -> NeuralResult<ArrayView2<'_, f64>> {
        if self.shape.ndim() != 2 {
            return Err(NeuralError::ShapeMismatch {
                expected: vec![0, 0],
                actual: self.shape.0.clone(),
            });
        }
        let rows = self.shape.0[0];
        let cols = self.shape.0[1];
        ArrayView2::from_shape((rows, cols), &self.data[..])
            .map_err(|_| NeuralError::ShapeMismatch {
                expected: vec![rows, cols],
                actual: vec![self.data.len()],
            })
    }

    /// Convert to owned 1D array
    pub fn to_array1(&self) -> Array1<f64> {
        Array1::from(self.data.clone())
    }

    /// Convert to owned 2D array
    pub fn to_array2(&self) -> NeuralResult<Array2<f64>> {
        if self.shape.ndim() != 2 {
            return Err(NeuralError::ShapeMismatch {
                expected: vec![0, 0],
                actual: self.shape.0.clone(),
            });
        }
        let rows = self.shape.0[0];
        let cols = self.shape.0[1];
        Array2::from_shape_vec((rows, cols), self.data.clone())
            .map_err(|_| NeuralError::ShapeMismatch {
                expected: vec![rows, cols],
                actual: vec![self.data.len()],
            })
    }

    /// Create from 1D array
    pub fn from_array1(arr: &Array1<f64>) -> Self {
        Self {
            data: arr.to_vec(),
            shape: TensorShape::d1(arr.len()),
            strides: vec![1],
        }
    }

    /// Create from 2D array
    pub fn from_array2(arr: &Array2<f64>) -> Self {
        let (rows, cols) = arr.dim();
        let shape = TensorShape::d2(rows, cols);
        let strides = Self::compute_strides(&shape);
        Self {
            data: arr.iter().copied().collect(),
            shape,
            strides,
        }
    }

    /// Matrix multiplication (2D tensors)
    pub fn matmul(&self, other: &Tensor) -> NeuralResult<Tensor> {
        let a = self.to_array2()?;
        let b = other.to_array2()?;

        if a.ncols() != b.nrows() {
            return Err(NeuralError::DimensionMismatch {
                input_dim: a.ncols(),
                expected_dim: b.nrows(),
            });
        }

        let result = a.dot(&b);
        Ok(Tensor::from_array2(&result))
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> NeuralResult<Tensor> {
        if self.shape != other.shape {
            return Err(NeuralError::ShapeMismatch {
                expected: self.shape.0.clone(),
                actual: other.shape.0.clone(),
            });
        }

        let data: Vec<f64> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Ok(Tensor { data, shape: self.shape.clone(), strides: self.strides.clone() })
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Tensor) -> NeuralResult<Tensor> {
        if self.shape != other.shape {
            return Err(NeuralError::ShapeMismatch {
                expected: self.shape.0.clone(),
                actual: other.shape.0.clone(),
            });
        }

        let data: Vec<f64> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();

        Ok(Tensor { data, shape: self.shape.clone(), strides: self.strides.clone() })
    }

    /// Element-wise multiplication (Hadamard product)
    pub fn mul(&self, other: &Tensor) -> NeuralResult<Tensor> {
        if self.shape != other.shape {
            return Err(NeuralError::ShapeMismatch {
                expected: self.shape.0.clone(),
                actual: other.shape.0.clone(),
            });
        }

        let data: Vec<f64> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();

        Ok(Tensor { data, shape: self.shape.clone(), strides: self.strides.clone() })
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: f64) -> Tensor {
        let data: Vec<f64> = self.data.iter().map(|x| x * scalar).collect();
        Tensor { data, shape: self.shape.clone(), strides: self.strides.clone() }
    }

    /// Transpose (2D only)
    pub fn transpose(&self) -> NeuralResult<Tensor> {
        if self.shape.ndim() != 2 {
            return Err(NeuralError::InvalidLayerConfig(
                "Transpose only supported for 2D tensors".into()
            ));
        }

        let arr = self.to_array2()?;
        let transposed = arr.t().to_owned();
        Ok(Tensor::from_array2(&transposed))
    }

    /// Sum all elements
    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    /// Mean of all elements
    pub fn mean(&self) -> f64 {
        self.sum() / self.numel() as f64
    }

    /// Sum along axis
    pub fn sum_axis(&self, axis: usize) -> NeuralResult<Tensor> {
        if axis >= self.shape.ndim() {
            return Err(NeuralError::InvalidLayerConfig(
                format!("Axis {} out of bounds for tensor with {} dims", axis, self.shape.ndim())
            ));
        }

        if self.shape.ndim() == 2 {
            let arr = self.to_array2()?;
            let result = arr.sum_axis(Axis(axis));
            Ok(Tensor::from_array1(&result))
        } else {
            // Fallback for other dimensions
            Ok(Tensor::full(TensorShape::d1(1), self.sum()))
        }
    }

    /// Apply function element-wise
    pub fn map<F: Fn(f64) -> f64>(&self, f: F) -> Tensor {
        let data: Vec<f64> = self.data.iter().map(|x| f(*x)).collect();
        Tensor { data, shape: self.shape.clone(), strides: self.strides.clone() }
    }

    /// Apply function element-wise in place
    pub fn map_inplace<F: Fn(f64) -> f64>(&mut self, f: F) {
        for x in &mut self.data {
            *x = f(*x);
        }
    }

    /// Reshape tensor (must have same number of elements)
    pub fn reshape(&self, new_shape: TensorShape) -> NeuralResult<Tensor> {
        if self.numel() != new_shape.numel() {
            return Err(NeuralError::ShapeMismatch {
                expected: new_shape.0.clone(),
                actual: vec![self.numel()],
            });
        }

        let strides = Self::compute_strides(&new_shape);
        Ok(Tensor {
            data: self.data.clone(),
            shape: new_shape,
            strides,
        })
    }

    /// Flatten to 1D
    pub fn flatten(&self) -> Tensor {
        Tensor {
            data: self.data.clone(),
            shape: TensorShape::d1(self.numel()),
            strides: vec![1],
        }
    }

    /// Clip values to range
    pub fn clip(&self, min: f64, max: f64) -> Tensor {
        self.map(|x| x.max(min).min(max))
    }

    /// Element-wise maximum
    pub fn max(&self) -> f64 {
        self.data.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Element-wise minimum
    pub fn min(&self) -> f64 {
        self.data.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// L2 norm
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::zeros(TensorShape::d2(3, 4));
        assert_eq!(t.shape().0, vec![3, 4]);
        assert_eq!(t.numel(), 12);
        assert!(t.data().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_tensor_matmul() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], TensorShape::d2(2, 2)).unwrap();
        let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], TensorShape::d2(2, 2)).unwrap();
        let c = a.matmul(&b).unwrap();

        // Identity multiplication
        assert_eq!(c.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tensor_ops() {
        let a = Tensor::ones(TensorShape::d1(5));
        let b = Tensor::full(TensorShape::d1(5), 2.0);

        let sum = a.add(&b).unwrap();
        assert!(sum.data().iter().all(|&x| (x - 3.0).abs() < 1e-10));

        let prod = a.mul(&b).unwrap();
        assert!(prod.data().iter().all(|&x| (x - 2.0).abs() < 1e-10));
    }

    #[test]
    fn test_xavier_init() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let t = Tensor::xavier(TensorShape::d2(100, 100), &mut rng);

        // Xavier should have mean ~0 and bounded variance
        let mean = t.mean();
        assert!(mean.abs() < 0.1, "Xavier mean should be near 0: {}", mean);
    }

    #[test]
    fn test_tensor_shape() {
        let s1 = TensorShape::d2(3, 4);
        let s2 = TensorShape::d2(3, 4);
        let s3 = TensorShape::d2(4, 3);

        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
        assert!(s1.broadcastable(&s2));
    }
}
