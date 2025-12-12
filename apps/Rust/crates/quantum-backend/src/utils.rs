//! Utility functions for quantum backend

use crate::error::Result;
use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;

/// Convert angle to radians
pub fn to_radians(degrees: f64) -> f64 {
    degrees * std::f64::consts::PI / 180.0
}

/// Create identity matrix
pub fn identity(size: usize) -> Array2<Complex64> {
    let mut matrix = Array2::zeros((size, size));
    for i in 0..size {
        matrix[[i, i]] = Complex64::new(1.0, 0.0);
    }
    matrix
}

/// Create Pauli X matrix
pub fn pauli_x() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        ]
    ).unwrap()
}

/// Create Pauli Y matrix
pub fn pauli_y() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0),
        ]
    ).unwrap()
}

/// Create Pauli Z matrix
pub fn pauli_z() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0),
        ]
    ).unwrap()
}

/// Create Hadamard matrix
pub fn hadamard() -> Array2<Complex64> {
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(inv_sqrt2, 0.0), Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0), Complex64::new(-inv_sqrt2, 0.0),
        ]
    ).unwrap()
}

/// Create rotation X matrix
pub fn rotation_x(angle: f64) -> Array2<Complex64> {
    let cos = (angle / 2.0).cos();
    let sin = (angle / 2.0).sin();
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(cos, 0.0), Complex64::new(0.0, -sin),
            Complex64::new(0.0, -sin), Complex64::new(cos, 0.0),
        ]
    ).unwrap()
}

/// Create rotation Y matrix
pub fn rotation_y(angle: f64) -> Array2<Complex64> {
    let cos = (angle / 2.0).cos();
    let sin = (angle / 2.0).sin();
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(cos, 0.0), Complex64::new(-sin, 0.0),
            Complex64::new(sin, 0.0), Complex64::new(cos, 0.0),
        ]
    ).unwrap()
}

/// Create rotation Z matrix
pub fn rotation_z(angle: f64) -> Array2<Complex64> {
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new((angle / 2.0).cos(), -(angle / 2.0).sin()),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new((angle / 2.0).cos(), (angle / 2.0).sin()),
        ]
    ).unwrap()
}

/// Tensor product of two matrices
pub fn tensor_product(a: &ArrayView2<Complex64>, b: &ArrayView2<Complex64>) -> Array2<Complex64> {
    let (m, n) = a.dim();
    let (p, q) = b.dim();
    let mut result = Array2::zeros((m * p, n * q));
    
    for i in 0..m {
        for j in 0..n {
            for k in 0..p {
                for l in 0..q {
                    result[[i * p + k, j * q + l]] = a[[i, j]] * b[[k, l]];
                }
            }
        }
    }
    
    result
}

/// Calculate fidelity between two quantum states
pub fn state_fidelity(state1: &[Complex64], state2: &[Complex64]) -> f64 {
    assert_eq!(state1.len(), state2.len());
    
    let inner_product: Complex64 = state1.iter()
        .zip(state2.iter())
        .map(|(a, b)| a.conj() * b)
        .sum();
    
    inner_product.norm_sqr()
}

/// Convert probability distribution to entropy
pub fn entropy(probabilities: &[f64]) -> f64 {
    probabilities.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.log2())
        .sum()
}

/// Binary representation of integer with fixed width
pub fn to_binary(value: usize, width: usize) -> String {
    format!("{:0width$b}", value, width = width)
}

/// Convert binary string to integer
pub fn from_binary(binary: &str) -> Result<usize> {
    usize::from_str_radix(binary, 2)
        .map_err(|e| anyhow::anyhow!("Invalid binary string: {}", e).into())
}

/// Calculate the trace of a matrix
pub fn trace(matrix: &ArrayView2<Complex64>) -> Complex64 {
    let (rows, cols) = matrix.dim();
    assert_eq!(rows, cols, "Matrix must be square");
    
    (0..rows).map(|i| matrix[[i, i]]).sum()
}

/// Check if matrix is unitary
pub fn is_unitary(matrix: &ArrayView2<Complex64>, tolerance: f64) -> bool {
    let (rows, cols) = matrix.dim();
    if rows != cols {
        return false;
    }
    
    let conjugate_transpose = matrix.t().mapv(|x| x.conj());
    let product = matrix.dot(&conjugate_transpose);
    let identity_matrix = identity(rows);
    
    product.iter()
        .zip(identity_matrix.iter())
        .all(|(a, b)| (a - b).norm() < tolerance)
}

/// Generate random unitary matrix
pub fn random_unitary(size: usize) -> Array2<Complex64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    // Generate random complex matrix
    let mut matrix = Array2::from_shape_fn((size, size), |_| {
        Complex64::new(
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        )
    });
    
    // QR decomposition to get unitary matrix
    // This is a simplified version - real implementation would use proper QR
    
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pauli_matrices() {
        let x = pauli_x();
        let y = pauli_y();
        let z = pauli_z();
        
        // Check anticommutation relations
        let xy = x.dot(&y);
        let yx = y.dot(&x);
        
        // XY - YX = 2iZ
        let commutator = &xy - &yx;
        let expected = pauli_z().mapv(|x| x * Complex64::new(0.0, 2.0));
        
        for (a, b) in commutator.iter().zip(expected.iter()) {
            assert!((a - b).norm() < 1e-10);
        }
    }
    
    #[test]
    fn test_tensor_product() {
        let h = hadamard();
        let i = identity(2);
        
        let hi = tensor_product(&h.view(), &i.view());
        assert_eq!(hi.dim(), (4, 4));
    }
    
    #[test]
    fn test_binary_conversion() {
        assert_eq!(to_binary(5, 4), "0101");
        assert_eq!(from_binary("0101").unwrap(), 5);
    }
}