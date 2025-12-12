//! Core types for QBMIA GPU acceleration

use std::fmt;
use serde::{Deserialize, Serialize};
use bytemuck::{Pod, Zeroable};

/// Complex number representation for quantum computations
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct Complex64 {
    pub real: f32,
    pub imag: f32,
}

impl Complex64 {
    pub fn new(real: f32, imag: f32) -> Self {
        Self { real, imag }
    }
    
    pub fn zero() -> Self {
        Self::new(0.0, 0.0)
    }
    
    pub fn one() -> Self {
        Self::new(1.0, 0.0)
    }
    
    pub fn i() -> Self {
        Self::new(0.0, 1.0)
    }
    
    pub fn magnitude(&self) -> f32 {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }
    
    pub fn magnitude_squared(&self) -> f32 {
        self.real * self.real + self.imag * self.imag
    }
    
    pub fn conjugate(&self) -> Self {
        Self::new(self.real, -self.imag)
    }
    
    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        if mag > 0.0 {
            Self::new(self.real / mag, self.imag / mag)
        } else {
            Self::zero()
        }
    }
}

impl std::ops::Add for Complex64 {
    type Output = Self;
    
    fn add(self, rhs: Self) -> Self {
        Self::new(self.real + rhs.real, self.imag + rhs.imag)
    }
}

impl std::ops::Mul for Complex64 {
    type Output = Self;
    
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.real * rhs.real - self.imag * rhs.imag,
            self.real * rhs.imag + self.imag * rhs.real,
        )
    }
}

impl std::ops::Mul<f32> for Complex64 {
    type Output = Self;
    
    fn mul(self, rhs: f32) -> Self {
        Self::new(self.real * rhs, self.imag * rhs)
    }
}

/// Quantum state representation for GPU acceleration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumState {
    /// Number of qubits
    pub n_qubits: usize,
    
    /// State vector amplitudes
    pub amplitudes: Vec<Complex64>,
    
    /// GPU buffer handle (optional)
    pub gpu_buffer: Option<u64>,
}

impl QuantumState {
    /// Create a new quantum state initialized to |0...0⟩
    pub fn new(n_qubits: usize) -> Result<Self, crate::QBMIAError> {
        if n_qubits > 32 {
            return Err(crate::QBMIAError::InvalidParameter("Too many qubits".to_string()));
        }
        
        let n_states = 1 << n_qubits;
        let mut amplitudes = vec![Complex64::zero(); n_states];
        amplitudes[0] = Complex64::one(); // |0...0⟩ state
        
        Ok(Self {
            n_qubits,
            amplitudes,
            gpu_buffer: None,
        })
    }
    
    /// Create a quantum state from amplitudes
    pub fn from_amplitudes(amplitudes: Vec<Complex64>) -> Result<Self, crate::QBMIAError> {
        let n_states = amplitudes.len();
        if !n_states.is_power_of_two() {
            return Err(crate::QBMIAError::InvalidParameter("Invalid state size".to_string()));
        }
        
        let n_qubits = n_states.trailing_zeros() as usize;
        
        // Normalize the state
        let norm_squared: f32 = amplitudes.iter().map(|a| a.magnitude_squared()).sum();
        let norm = norm_squared.sqrt();
        
        let normalized_amplitudes = if norm > 0.0 {
            amplitudes.iter().map(|a| *a * (1.0 / norm)).collect()
        } else {
            amplitudes
        };
        
        Ok(Self {
            n_qubits,
            amplitudes: normalized_amplitudes,
            gpu_buffer: None,
        })
    }
    
    /// Get the dimension of the state space
    pub fn dimension(&self) -> usize {
        self.amplitudes.len()
    }
    
    /// Calculate the fidelity with another quantum state
    pub fn fidelity(&self, other: &Self) -> f32 {
        if self.n_qubits != other.n_qubits {
            return 0.0;
        }
        
        let inner_product: Complex64 = self.amplitudes
            .iter()
            .zip(other.amplitudes.iter())
            .map(|(a, b)| a.conjugate() * *b)
            .fold(Complex64::zero(), |acc, x| acc + x);
        
        inner_product.magnitude()
    }
    
    /// Normalize the quantum state
    pub fn normalize(&mut self) {
        let norm_squared: f32 = self.amplitudes.iter().map(|a| a.magnitude_squared()).sum();
        let norm = norm_squared.sqrt();
        
        if norm > 0.0 {
            for amplitude in &mut self.amplitudes {
                *amplitude = *amplitude * (1.0 / norm);
            }
        }
    }
    
    /// Convert to raw bytes for GPU transfer
    pub fn to_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.amplitudes)
    }
    
    /// Create from raw bytes after GPU transfer
    pub fn from_bytes(bytes: &[u8], n_qubits: usize) -> Result<Self, crate::QBMIAError> {
        let amplitudes: Vec<Complex64> = bytemuck::cast_slice(bytes).to_vec();
        
        let expected_size = 1 << n_qubits;
        if amplitudes.len() != expected_size {
            return Err(crate::QBMIAError::InvalidParameter("Invalid state size".to_string()));
        }
        
        Ok(Self {
            n_qubits,
            amplitudes,
            gpu_buffer: None,
        })
    }
}

/// Unitary gate representation for quantum circuits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitaryGate {
    /// Gate name
    pub name: String,
    
    /// Gate matrix (row-major order)
    pub matrix: Vec<Complex64>,
    
    /// Number of qubits this gate acts on
    pub n_qubits: usize,
}

impl UnitaryGate {
    /// Create a new unitary gate
    pub fn new(name: String, matrix: Vec<Complex64>, n_qubits: usize) -> Result<Self, crate::QBMIAError> {
        let expected_size = 1 << (2 * n_qubits);
        if matrix.len() != expected_size {
            return Err(crate::QBMIAError::InvalidParameter("Invalid gate matrix size".to_string()));
        }
        
        Ok(Self {
            name,
            matrix,
            n_qubits,
        })
    }
    
    /// Create a Hadamard gate
    pub fn hadamard() -> Self {
        let inv_sqrt_2 = 1.0 / std::f32::consts::SQRT_2;
        let matrix = vec![
            Complex64::new(inv_sqrt_2, 0.0),  Complex64::new(inv_sqrt_2, 0.0),
            Complex64::new(inv_sqrt_2, 0.0),  Complex64::new(-inv_sqrt_2, 0.0),
        ];
        
        Self {
            name: "H".to_string(),
            matrix,
            n_qubits: 1,
        }
    }
    
    /// Create a CNOT gate
    pub fn cnot() -> Self {
        let matrix = vec![
            Complex64::one(),  Complex64::zero(), Complex64::zero(), Complex64::zero(),
            Complex64::zero(), Complex64::one(),  Complex64::zero(), Complex64::zero(),
            Complex64::zero(), Complex64::zero(), Complex64::zero(), Complex64::one(),
            Complex64::zero(), Complex64::zero(), Complex64::one(),  Complex64::zero(),
        ];
        
        Self {
            name: "CNOT".to_string(),
            matrix,
            n_qubits: 2,
        }
    }
    
    /// Create a Pauli-X gate
    pub fn pauli_x() -> Self {
        let matrix = vec![
            Complex64::zero(), Complex64::one(),
            Complex64::one(),  Complex64::zero(),
        ];
        
        Self {
            name: "X".to_string(),
            matrix,
            n_qubits: 1,
        }
    }
    
    /// Create a Pauli-Y gate
    pub fn pauli_y() -> Self {
        let matrix = vec![
            Complex64::zero(), Complex64::new(0.0, -1.0),
            Complex64::i(),    Complex64::zero(),
        ];
        
        Self {
            name: "Y".to_string(),
            matrix,
            n_qubits: 1,
        }
    }
    
    /// Create a Pauli-Z gate
    pub fn pauli_z() -> Self {
        let matrix = vec![
            Complex64::one(),  Complex64::zero(),
            Complex64::zero(), Complex64::new(-1.0, 0.0),
        ];
        
        Self {
            name: "Z".to_string(),
            matrix,
            n_qubits: 1,
        }
    }
    
    /// Create a rotation gate around Z axis
    pub fn rz(angle: f32) -> Self {
        let half_angle = angle / 2.0;
        let matrix = vec![
            Complex64::new(half_angle.cos(), -half_angle.sin()), Complex64::zero(),
            Complex64::zero(), Complex64::new(half_angle.cos(), half_angle.sin()),
        ];
        
        Self {
            name: format!("RZ({:.3})", angle),
            matrix,
            n_qubits: 1,
        }
    }
    
    /// Convert to raw bytes for GPU transfer
    pub fn to_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.matrix)
    }
}

/// Payoff matrix for game theory calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PayoffMatrix {
    /// Matrix dimensions
    pub rows: usize,
    pub cols: usize,
    
    /// Matrix data (row-major order)
    pub data: Vec<f32>,
    
    /// GPU buffer handle (optional)
    pub gpu_buffer: Option<u64>,
}

impl PayoffMatrix {
    /// Create a new payoff matrix
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self, crate::QBMIAError> {
        if data.len() != rows * cols {
            return Err(crate::QBMIAError::InvalidParameter("Invalid matrix dimensions".to_string()));
        }
        
        Ok(Self {
            rows,
            cols,
            data,
            gpu_buffer: None,
        })
    }
    
    /// Create a random payoff matrix
    pub fn random(rows: usize, cols: usize) -> Result<Self, crate::QBMIAError> {
        use rand::prelude::*;
        let mut rng = thread_rng();
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        
        Self::new(rows, cols, data)
    }
    
    /// Get element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> Option<f32> {
        if row < self.rows && col < self.cols {
            Some(self.data[row * self.cols + col])
        } else {
            None
        }
    }
    
    /// Set element at (row, col)
    pub fn set(&mut self, row: usize, col: usize, value: f32) -> Result<(), crate::QBMIAError> {
        if row < self.rows && col < self.cols {
            self.data[row * self.cols + col] = value;
            Ok(())
        } else {
            Err(crate::QBMIAError::InvalidParameter("Index out of bounds".to_string()))
        }
    }
    
    /// Convert to raw bytes for GPU transfer
    pub fn to_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.data)
    }
}

/// Strategy vector for game theory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyVector {
    /// Strategy probabilities
    pub probabilities: Vec<f32>,
    
    /// GPU buffer handle (optional)
    pub gpu_buffer: Option<u64>,
}

impl StrategyVector {
    /// Create a new strategy vector
    pub fn new(probabilities: Vec<f32>) -> Result<Self, crate::QBMIAError> {
        let sum: f32 = probabilities.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(crate::QBMIAError::InvalidParameter("Probabilities must sum to 1".to_string()));
        }
        
        Ok(Self {
            probabilities,
            gpu_buffer: None,
        })
    }
    
    /// Create a uniform strategy vector
    pub fn uniform(size: usize) -> Result<Self, crate::QBMIAError> {
        let prob = 1.0 / size as f32;
        let probabilities = vec![prob; size];
        Ok(Self {
            probabilities,
            gpu_buffer: None,
        })
    }
    
    /// Create a random strategy vector
    pub fn random(size: usize) -> Result<Self, crate::QBMIAError> {
        use rand::prelude::*;
        let mut rng = thread_rng();
        let mut probabilities: Vec<f32> = (0..size)
            .map(|_| rng.gen_range(0.0..1.0))
            .collect();
        
        // Normalize
        let sum: f32 = probabilities.iter().sum();
        if sum > 0.0 {
            for p in &mut probabilities {
                *p /= sum;
            }
        }
        
        Ok(Self {
            probabilities,
            gpu_buffer: None,
        })
    }
    
    /// Get the dimension of the strategy space
    pub fn dimension(&self) -> usize {
        self.probabilities.len()
    }
    
    /// Convert to raw bytes for GPU transfer
    pub fn to_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.probabilities)
    }
}

/// Nash equilibrium result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NashEquilibrium {
    /// Player strategies
    pub strategies: Vec<StrategyVector>,
    
    /// Convergence metric
    pub convergence: f32,
    
    /// Number of iterations taken
    pub iterations: usize,
    
    /// Expected payoffs
    pub payoffs: Vec<f32>,
}

impl NashEquilibrium {
    /// Create a new Nash equilibrium
    pub fn new(
        strategies: Vec<StrategyVector>,
        convergence: f32,
        iterations: usize,
        payoffs: Vec<f32>,
    ) -> Self {
        Self {
            strategies,
            convergence,
            iterations,
            payoffs,
        }
    }
    
    /// Check if the equilibrium has converged
    pub fn is_converged(&self, threshold: f32) -> bool {
        self.convergence < threshold
    }
}

/// Nash solver parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NashSolverParams {
    /// Learning rate for gradient updates
    pub learning_rate: f32,
    
    /// Maximum number of iterations
    pub max_iterations: usize,
    
    /// Convergence threshold
    pub convergence_threshold: f32,
    
    /// Regularization parameter
    pub regularization: f32,
}

impl Default for NashSolverParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            max_iterations: 1000,
            convergence_threshold: 1e-6,
            regularization: 1e-8,
        }
    }
}

/// Pattern for pattern matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    /// Pattern features
    pub features: Vec<f32>,
    
    /// Pattern label (optional)
    pub label: Option<String>,
    
    /// GPU buffer handle (optional)
    pub gpu_buffer: Option<u64>,
}

impl Pattern {
    /// Create a new pattern
    pub fn new(features: Vec<f32>, label: Option<String>) -> Self {
        Self {
            features,
            label,
            gpu_buffer: None,
        }
    }
    
    /// Create a random pattern
    pub fn random(size: usize) -> Result<Self, crate::QBMIAError> {
        use rand::prelude::*;
        let mut rng = thread_rng();
        let features: Vec<f32> = (0..size)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        
        Ok(Self {
            features,
            label: None,
            gpu_buffer: None,
        })
    }
    
    /// Get the dimension of the pattern
    pub fn dimension(&self) -> usize {
        self.features.len()
    }
    
    /// Calculate cosine similarity with another pattern
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        if self.features.len() != other.features.len() {
            return 0.0;
        }
        
        let dot_product: f32 = self.features
            .iter()
            .zip(other.features.iter())
            .map(|(a, b)| a * b)
            .sum();
        
        let norm_a: f32 = self.features.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.features.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a > 0.0 && norm_b > 0.0 {
            dot_product / (norm_a * norm_b)
        } else {
            0.0
        }
    }
    
    /// Convert to raw bytes for GPU transfer
    pub fn to_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.features)
    }
}

/// GPU buffer handle for memory management
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuBuffer {
    /// Unique buffer ID
    pub id: u64,
    
    /// Buffer size in bytes
    pub size: usize,
    
    /// Buffer usage type
    pub usage: GpuBufferUsage,
}

/// GPU buffer usage types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuBufferUsage {
    /// Storage buffer for general data
    Storage,
    /// Uniform buffer for constants
    Uniform,
    /// Vertex buffer for geometry
    Vertex,
    /// Index buffer for indices
    Index,
    /// Staging buffer for transfers
    Staging,
}

/// GPU compute pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuPipelineConfig {
    /// Shader source code
    pub shader_source: String,
    
    /// Workgroup size
    pub workgroup_size: [u32; 3],
    
    /// Buffer bindings
    pub buffer_bindings: Vec<GpuBufferBinding>,
    
    /// Pipeline name
    pub name: String,
}

/// GPU buffer binding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuBufferBinding {
    /// Binding index
    pub binding: u32,
    
    /// Buffer usage
    pub usage: GpuBufferUsage,
    
    /// Buffer size
    pub size: usize,
}

/// Kernel execution parameters
#[derive(Debug, Clone)]
pub struct KernelParams {
    /// Workgroup dispatch size
    pub dispatch_size: [u32; 3],
    
    /// Input buffers
    pub input_buffers: Vec<GpuBuffer>,
    
    /// Output buffers
    pub output_buffers: Vec<GpuBuffer>,
    
    /// Execution timeout (nanoseconds)
    pub timeout_ns: u64,
}

impl Default for KernelParams {
    fn default() -> Self {
        Self {
            dispatch_size: [1, 1, 1],
            input_buffers: Vec::new(),
            output_buffers: Vec::new(),
            timeout_ns: 1_000_000, // 1ms default timeout
        }
    }
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// Device name
    pub name: String,
    
    /// Device vendor
    pub vendor: String,
    
    /// Device type (discrete, integrated, etc.)
    pub device_type: String,
    
    /// Available memory (bytes)
    pub memory_bytes: u64,
    
    /// Max compute units
    pub compute_units: u32,
    
    /// Max workgroup size
    pub max_workgroup_size: [u32; 3],
    
    /// Supports CUDA
    pub supports_cuda: bool,
    
    /// Supports Metal
    pub supports_metal: bool,
    
    /// Supports Vulkan
    pub supports_vulkan: bool,
}

impl fmt::Display for GpuDeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} ({}, {} GB, {} CUs)",
            self.vendor,
            self.name,
            self.device_type,
            self.memory_bytes / (1024 * 1024 * 1024),
            self.compute_units
        )
    }
}

/// SIMD vector types for CPU preprocessing
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct SimdFloat4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl SimdFloat4 {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }
    
    pub fn splat(value: f32) -> Self {
        Self::new(value, value, value, value)
    }
    
    pub fn zero() -> Self {
        Self::splat(0.0)
    }
    
    pub fn one() -> Self {
        Self::splat(1.0)
    }
}

/// SIMD vector types for complex numbers
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct SimdComplex2 {
    pub re: [f32; 2],
    pub im: [f32; 2],
}

impl SimdComplex2 {
    pub fn new(z0: Complex64, z1: Complex64) -> Self {
        Self {
            re: [z0.real, z1.real],
            im: [z0.imag, z1.imag],
        }
    }
    
    pub fn zero() -> Self {
        Self {
            re: [0.0, 0.0],
            im: [0.0, 0.0],
        }
    }
}

use rand::prelude::*;

// Ensure the module compiles
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_complex64_operations() {
        let z1 = Complex64::new(1.0, 2.0);
        let z2 = Complex64::new(3.0, 4.0);
        let sum = z1 + z2;
        
        assert_eq!(sum.real, 4.0);
        assert_eq!(sum.imag, 6.0);
        
        let product = z1 * z2;
        assert_eq!(product.real, -5.0);
        assert_eq!(product.imag, 10.0);
    }
    
    #[test]
    fn test_quantum_state_creation() {
        let state = QuantumState::new(2).unwrap();
        assert_eq!(state.n_qubits, 2);
        assert_eq!(state.amplitudes.len(), 4);
        assert_eq!(state.amplitudes[0], Complex64::one());
        assert_eq!(state.amplitudes[1], Complex64::zero());
    }
    
    #[test]
    fn test_unitary_gates() {
        let h = UnitaryGate::hadamard();
        assert_eq!(h.n_qubits, 1);
        assert_eq!(h.matrix.len(), 4);
        
        let cnot = UnitaryGate::cnot();
        assert_eq!(cnot.n_qubits, 2);
        assert_eq!(cnot.matrix.len(), 16);
    }
    
    #[test]
    fn test_payoff_matrix() {
        let matrix = PayoffMatrix::random(3, 3).unwrap();
        assert_eq!(matrix.rows, 3);
        assert_eq!(matrix.cols, 3);
        assert_eq!(matrix.data.len(), 9);
    }
    
    #[test]
    fn test_strategy_vector() {
        let strategy = StrategyVector::uniform(4).unwrap();
        assert_eq!(strategy.dimension(), 4);
        assert!((strategy.probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_pattern_similarity() {
        let pattern1 = Pattern::new(vec![1.0, 0.0, 0.0], None);
        let pattern2 = Pattern::new(vec![0.0, 1.0, 0.0], None);
        let pattern3 = Pattern::new(vec![1.0, 0.0, 0.0], None);
        
        assert_eq!(pattern1.cosine_similarity(&pattern2), 0.0);
        assert_eq!(pattern1.cosine_similarity(&pattern3), 1.0);
    }
}