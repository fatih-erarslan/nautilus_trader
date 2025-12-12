//! Quantum Core Utilities and Helper Functions
//!
//! This module provides utility functions, helper types, and common functionality
//! used throughout the quantum-core crate.

use crate::quantum_state::{QuantumState, ComplexAmplitude};
use crate::error::{QuantumError, QuantumResult};
use num_complex::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;
use serde::{Deserialize, Serialize};
use rand::Rng;
use rayon::prelude::*;

/// Mathematical constants used in quantum computing
pub mod constants {
    use std::f64::consts::PI;
    
    /// Planck's constant (J⋅s)
    pub const PLANCK_CONSTANT: f64 = 6.62607015e-34;
    
    /// Reduced Planck's constant (ℏ)
    pub const HBAR: f64 = 1.0545718176461565e-34;
    
    /// Speed of light (m/s)
    pub const SPEED_OF_LIGHT: f64 = 299792458.0;
    
    /// Electron charge (C)
    pub const ELECTRON_CHARGE: f64 = 1.602176634e-19;
    
    /// Electron mass (kg)
    pub const ELECTRON_MASS: f64 = 9.1093837015e-31;
    
    /// Proton mass (kg)
    pub const PROTON_MASS: f64 = 1.67262192369e-27;
    
    /// Neutron mass (kg)
    pub const NEUTRON_MASS: f64 = 1.67492749804e-27;
    
    /// Bohr magneton (J/T)
    pub const BOHR_MAGNETON: f64 = 9.2740100783e-24;
    
    /// Common quantum gate angles
    pub const PI_2: f64 = PI / 2.0;
    pub const PI_4: f64 = PI / 4.0;
    pub const PI_8: f64 = PI / 8.0;
    pub const TWO_PI: f64 = 2.0 * PI;
    
    /// Quantum measurement precision
    pub const MEASUREMENT_PRECISION: f64 = 1e-12;
    
    /// Default quantum fidelity threshold
    pub const FIDELITY_THRESHOLD: f64 = 0.99;
    
    /// Default quantum error tolerance
    pub const ERROR_TOLERANCE: f64 = 1e-10;
}

/// Quantum utility functions
pub mod quantum_utils {
    use super::*;
    
    /// Generate random complex amplitude
    pub fn random_complex_amplitude() -> ComplexAmplitude {
        let mut rng = rand::thread_rng();
        let real = rng.gen_range(-1.0..1.0);
        let imag = rng.gen_range(-1.0..1.0);
        Complex64::new(real, imag)
    }
    
    /// Generate random quantum state
    pub fn random_quantum_state(num_qubits: usize) -> QuantumResult<QuantumState> {
        let mut state = QuantumState::new(num_qubits)?;
        let size = 1 << num_qubits;
        
        // Generate random amplitudes
        for i in 0..size {
            let amplitude = random_complex_amplitude();
            state.set_amplitude(i, amplitude)?;
        }
        
        // Normalize the state
        state.normalize()?;
        
        Ok(state)
    }
    
    /// Calculate quantum state norm
    pub fn state_norm(amplitudes: &[ComplexAmplitude]) -> f64 {
        amplitudes.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt()
    }
    
    /// Normalize quantum state amplitudes
    pub fn normalize_amplitudes(amplitudes: &mut [ComplexAmplitude]) -> QuantumResult<()> {
        let norm = state_norm(amplitudes);
        
        if norm < constants::ERROR_TOLERANCE {
            return Err(QuantumError::InvalidState { message: "Cannot normalize zero state".to_string() });
        }
        
        for amplitude in amplitudes.iter_mut() {
            *amplitude /= norm;
        }
        
        Ok(())
    }
    
    /// Calculate fidelity between two quantum states
    pub fn calculate_fidelity(state1: &QuantumState, state2: &QuantumState) -> QuantumResult<f64> {
        if state1.num_qubits() != state2.num_qubits() {
            return Err(QuantumError::InvalidOperation { 
                operation: "Fidelity calculation".to_string(), 
                message: "Cannot calculate fidelity between states with different qubit counts".to_string() 
            });
        }
        
        let size = 1 << state1.num_qubits();
        let mut inner_product = Complex64::new(0.0, 0.0);
        
        for i in 0..size {
            let amp1 = state1.get_amplitude(i)?;
            let amp2 = state2.get_amplitude(i)?;
            inner_product += amp1.conj() * amp2;
        }
        
        Ok(inner_product.norm_sqr())
    }
    
    /// Calculate quantum state entropy
    pub fn calculate_entropy(state: &QuantumState) -> QuantumResult<f64> {
        let size = 1 << state.num_qubits();
        let mut entropy = 0.0;
        
        for i in 0..size {
            let amplitude = state.get_amplitude(i)?;
            let probability = amplitude.norm_sqr();
            
            if probability > constants::ERROR_TOLERANCE {
                entropy -= probability * probability.ln();
            }
        }
        
        Ok(entropy)
    }
    
    /// Check if quantum state is normalized
    pub fn is_normalized(state: &QuantumState) -> QuantumResult<bool> {
        let size = 1 << state.num_qubits();
        let mut norm_squared = 0.0;
        
        for i in 0..size {
            let amplitude = state.get_amplitude(i)?;
            norm_squared += amplitude.norm_sqr();
        }
        
        Ok((norm_squared - 1.0).abs() < constants::ERROR_TOLERANCE)
    }
    
    /// Convert quantum state to density matrix
    pub fn to_density_matrix(state: &QuantumState) -> QuantumResult<Vec<Vec<ComplexAmplitude>>> {
        let size = 1 << state.num_qubits();
        let mut density_matrix = vec![vec![Complex64::new(0.0, 0.0); size]; size];
        
        for i in 0..size {
            for j in 0..size {
                let amp_i = state.get_amplitude(i)?;
                let amp_j = state.get_amplitude(j)?;
                density_matrix[i][j] = amp_i * amp_j.conj();
            }
        }
        
        Ok(density_matrix)
    }
    
    /// Calculate trace of density matrix
    pub fn trace_density_matrix(density_matrix: &[Vec<ComplexAmplitude>]) -> ComplexAmplitude {
        let mut trace = Complex64::new(0.0, 0.0);
        
        for i in 0..density_matrix.len() {
            trace += density_matrix[i][i];
        }
        
        trace
    }
    
    /// Generate maximally entangled state (Bell state)
    pub fn bell_state(num_qubits: usize) -> QuantumResult<QuantumState> {
        if num_qubits < 2 {
            return Err(QuantumError::InvalidOperation { operation: "Bell state creation".to_string(), message: "Bell state requires at least 2 qubits".to_string() });
        }
        
        let mut state = QuantumState::new(num_qubits)?;
        let size = 1 << num_qubits;
        
        // Create |00...0⟩ + |11...1⟩ state
        let amplitude = Complex64::new(1.0 / (2.0_f64).sqrt(), 0.0);
        state.set_amplitude(0, amplitude)?; // |00...0⟩
        state.set_amplitude(size - 1, amplitude)?; // |11...1⟩
        
        Ok(state)
    }
    
    /// Generate GHZ state (generalized Bell state)
    pub fn ghz_state(num_qubits: usize) -> QuantumResult<QuantumState> {
        if num_qubits < 2 {
            return Err(QuantumError::InvalidOperation { operation: "GHZ state creation".to_string(), message: "GHZ state requires at least 2 qubits".to_string() });
        }
        
        let mut state = QuantumState::new(num_qubits)?;
        let size = 1 << num_qubits;
        
        // Create (|00...0⟩ + |11...1⟩) / √2
        let amplitude = Complex64::new(1.0 / (2.0_f64).sqrt(), 0.0);
        state.set_amplitude(0, amplitude)?; // |00...0⟩
        state.set_amplitude(size - 1, amplitude)?; // |11...1⟩
        
        Ok(state)
    }
    
    /// Generate W state
    pub fn w_state(num_qubits: usize) -> QuantumResult<QuantumState> {
        if num_qubits < 2 {
            return Err(QuantumError::InvalidOperation { operation: "W state creation".to_string(), message: "W state requires at least 2 qubits".to_string() });
        }
        
        let mut state = QuantumState::new(num_qubits)?;
        let amplitude = Complex64::new(1.0 / (num_qubits as f64).sqrt(), 0.0);
        
        // Create equal superposition of all single-excitation states
        for i in 0..num_qubits {
            let basis_state = 1 << i;
            state.set_amplitude(basis_state, amplitude)?;
        }
        
        Ok(state)
    }
}

/// Mathematical utility functions
pub mod math_utils {
    use super::*;
    
    /// Calculate binomial coefficient
    pub fn binomial_coefficient(n: usize, k: usize) -> u64 {
        if k > n {
            return 0;
        }
        
        if k == 0 || k == n {
            return 1;
        }
        
        let k = k.min(n - k); // Take advantage of symmetry
        let mut result = 1u64;
        
        for i in 0..k {
            result = result * (n - i) as u64 / (i + 1) as u64;
        }
        
        result
    }
    
    /// Calculate factorial
    pub fn factorial(n: usize) -> u64 {
        if n <= 1 {
            1
        } else {
            (2..=n).map(|i| i as u64).product()
        }
    }
    
    /// Calculate nth Fibonacci number
    pub fn fibonacci(n: usize) -> u64 {
        match n {
            0 => 0,
            1 => 1,
            _ => {
                let mut a = 0u64;
                let mut b = 1u64;
                for _ in 2..=n {
                    let temp = a + b;
                    a = b;
                    b = temp;
                }
                b
            }
        }
    }
    
    /// Calculate greatest common divisor
    pub fn gcd(a: u64, b: u64) -> u64 {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    
    /// Calculate least common multiple
    pub fn lcm(a: u64, b: u64) -> u64 {
        (a * b) / gcd(a, b)
    }
    
    /// Check if number is prime
    pub fn is_prime(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }
        
        let sqrt_n = (n as f64).sqrt() as u64;
        for i in (3..=sqrt_n).step_by(2) {
            if n % i == 0 {
                return false;
            }
        }
        
        true
    }
    
    /// Find next prime number
    pub fn next_prime(n: u64) -> u64 {
        let mut candidate = n + 1;
        while !is_prime(candidate) {
            candidate += 1;
        }
        candidate
    }
    
    /// Calculate modular exponentiation
    pub fn mod_exp(base: u64, exponent: u64, modulus: u64) -> u64 {
        if modulus == 1 {
            return 0;
        }
        
        let mut result = 1;
        let mut base = base % modulus;
        let mut exp = exponent;
        
        while exp > 0 {
            if exp % 2 == 1 {
                result = (result * base) % modulus;
            }
            exp >>= 1;
            base = (base * base) % modulus;
        }
        
        result
    }
    
    /// Calculate matrix determinant (2x2)
    pub fn det_2x2(matrix: &[[f64; 2]; 2]) -> f64 {
        matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    }
    
    /// Calculate matrix inverse (2x2)
    pub fn inv_2x2(matrix: &[[f64; 2]; 2]) -> Option<[[f64; 2]; 2]> {
        let det = det_2x2(matrix);
        
        if det.abs() < constants::ERROR_TOLERANCE {
            return None;
        }
        
        let inv_det = 1.0 / det;
        
        Some([
            [matrix[1][1] * inv_det, -matrix[0][1] * inv_det],
            [-matrix[1][0] * inv_det, matrix[0][0] * inv_det],
        ])
    }
    
    /// Linear interpolation
    pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
        a + t * (b - a)
    }
    
    /// Clamp value between min and max
    pub fn clamp(value: f64, min: f64, max: f64) -> f64 {
        value.max(min).min(max)
    }
    
    /// Calculate Euclidean distance
    pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() {
            return f64::INFINITY;
        }
        
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// Performance optimization utilities
pub mod performance_utils {
    use super::*;
    
    /// Parallel computation of quantum state operations
    pub fn parallel_amplitude_operation<F>(
        amplitudes: &mut [ComplexAmplitude],
        operation: F,
    ) -> QuantumResult<()>
    where
        F: Fn(&mut ComplexAmplitude) + Send + Sync,
    {
        amplitudes.par_iter_mut().for_each(|amp| operation(amp));
        Ok(())
    }
    
    /// Parallel dot product calculation
    pub fn parallel_dot_product(a: &[ComplexAmplitude], b: &[ComplexAmplitude]) -> ComplexAmplitude {
        if a.len() != b.len() {
            return Complex64::new(0.0, 0.0);
        }
        
        a.par_iter()
            .zip(b.par_iter())
            .map(|(x, y)| x * y.conj())
            .sum()
    }
    
    /// Parallel matrix-vector multiplication
    pub fn parallel_matrix_vector_mult(
        matrix: &[Vec<ComplexAmplitude>],
        vector: &[ComplexAmplitude],
    ) -> Vec<ComplexAmplitude> {
        matrix
            .par_iter()
            .map(|row| {
                row.iter()
                    .zip(vector.iter())
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect()
    }
    
    /// Cache for memoization
    pub struct MemoCache<K, V> {
        cache: HashMap<K, V>,
        max_size: usize,
    }
    
    impl<K, V> MemoCache<K, V>
    where
        K: std::hash::Hash + Eq + Clone,
        V: Clone,
    {
        pub fn new(max_size: usize) -> Self {
            Self {
                cache: HashMap::new(),
                max_size,
            }
        }
        
        pub fn get_or_compute<F>(&mut self, key: K, compute: F) -> V
        where
            F: FnOnce() -> V,
        {
            if let Some(value) = self.cache.get(&key) {
                return value.clone();
            }
            
            let value = compute();
            
            if self.cache.len() >= self.max_size {
                // Simple eviction: clear half the cache
                let keys_to_remove: Vec<_> = self.cache.keys().take(self.max_size / 2).cloned().collect();
                for k in keys_to_remove {
                    self.cache.remove(&k);
                }
            }
            
            self.cache.insert(key, value.clone());
            value
        }
        
        pub fn clear(&mut self) {
            self.cache.clear();
        }
    }
    
    /// Timing utilities
    pub struct Timer {
        start: std::time::Instant,
    }
    
    impl Timer {
        pub fn start() -> Self {
            Self {
                start: std::time::Instant::now(),
            }
        }
        
        pub fn elapsed(&self) -> std::time::Duration {
            self.start.elapsed()
        }
        
        pub fn elapsed_ms(&self) -> f64 {
            self.elapsed().as_millis() as f64
        }
        
        pub fn elapsed_us(&self) -> f64 {
            self.elapsed().as_micros() as f64
        }
        
        pub fn elapsed_ns(&self) -> f64 {
            self.elapsed().as_nanos() as f64
        }
    }
}

/// Conversion utilities
pub mod conversion_utils {
    use super::*;
    
    /// Convert degrees to radians
    pub fn deg_to_rad(degrees: f64) -> f64 {
        degrees * PI / 180.0
    }
    
    /// Convert radians to degrees
    pub fn rad_to_deg(radians: f64) -> f64 {
        radians * 180.0 / PI
    }
    
    /// Convert complex number to polar form
    pub fn complex_to_polar(c: ComplexAmplitude) -> (f64, f64) {
        (c.norm(), c.arg())
    }
    
    /// Convert polar form to complex number
    pub fn polar_to_complex(magnitude: f64, phase: f64) -> ComplexAmplitude {
        Complex64::new(magnitude * phase.cos(), magnitude * phase.sin())
    }
    
    /// Convert quantum state to probability distribution
    pub fn state_to_probabilities(state: &QuantumState) -> QuantumResult<Vec<f64>> {
        let size = 1 << state.num_qubits();
        let mut probabilities = Vec::with_capacity(size);
        
        for i in 0..size {
            let amplitude = state.get_amplitude(i)?;
            probabilities.push(amplitude.norm_sqr());
        }
        
        Ok(probabilities)
    }
    
    /// Convert binary string to decimal
    pub fn binary_to_decimal(binary: &str) -> QuantumResult<usize> {
        usize::from_str_radix(binary, 2)
            .map_err(|_| QuantumError::InvalidOperation { operation: "Binary string parsing".to_string(), message: "Invalid binary string".to_string() })
    }
    
    /// Convert decimal to binary string
    pub fn decimal_to_binary(decimal: usize, width: usize) -> String {
        format!("{:0width$b}", decimal, width = width)
    }
    
    /// Convert qubit index to binary representation
    pub fn qubit_index_to_binary(index: usize, num_qubits: usize) -> String {
        decimal_to_binary(index, num_qubits)
    }
    
    /// Convert binary representation to qubit index
    pub fn binary_to_qubit_index(binary: &str) -> QuantumResult<usize> {
        binary_to_decimal(binary)
    }
}

/// Validation utilities
pub mod validation_utils {
    use super::*;
    
    /// Validate quantum state
    pub fn validate_quantum_state(state: &QuantumState) -> QuantumResult<()> {
        // Check if normalized
        if !quantum_utils::is_normalized(state)? {
            return Err(QuantumError::InvalidState { message: "Quantum state is not normalized".to_string() });
        }
        
        // Check for NaN or infinite values
        let size = 1 << state.num_qubits();
        for i in 0..size {
            let amplitude = state.get_amplitude(i)?;
            if !amplitude.re.is_finite() || !amplitude.im.is_finite() {
                return Err(QuantumError::InvalidState { message: "Quantum state contains invalid values".to_string() });
            }
        }
        
        Ok(())
    }
    
    /// Validate qubit index
    pub fn validate_qubit_index(index: usize, num_qubits: usize) -> QuantumResult<()> {
        if index >= num_qubits {
            return Err(QuantumError::invalid_qubit_index(index, num_qubits));
        }
        Ok(())
    }
    
    /// Validate probability
    pub fn validate_probability(probability: f64) -> QuantumResult<()> {
        if probability < 0.0 || probability > 1.0 {
            return Err(QuantumError::InvalidOperation { 
                operation: "Probability validation".to_string(), 
                message: format!("Invalid probability: {}", probability) 
            });
        }
        Ok(())
    }
    
    /// Validate angle
    pub fn validate_angle(angle: f64) -> QuantumResult<()> {
        if !angle.is_finite() {
            return Err(QuantumError::InvalidOperation { 
                operation: "Angle validation".to_string(), 
                message: format!("Invalid angle: {}", angle) 
            });
        }
        Ok(())
    }
    
    /// Validate matrix dimensions
    pub fn validate_matrix_dimensions(matrix: &[Vec<ComplexAmplitude>]) -> QuantumResult<()> {
        if matrix.is_empty() {
            return Err(QuantumError::InvalidOperation { operation: "Matrix validation".to_string(), message: "Matrix is empty".to_string() });
        }
        
        let cols = matrix[0].len();
        for row in matrix {
            if row.len() != cols {
                return Err(QuantumError::InvalidOperation { operation: "Matrix validation".to_string(), message: "Matrix has inconsistent dimensions".to_string() });
            }
        }
        
        Ok(())
    }
}

/// Testing utilities
pub mod testing_utils {
    use super::*;
    
    /// Create test quantum state
    pub fn create_test_state(num_qubits: usize) -> QuantumState {
        let mut state = QuantumState::new(num_qubits).unwrap();
        
        // Set |0⟩ state
        state.set_amplitude(0, Complex64::new(1.0, 0.0)).unwrap();
        
        state
    }
    
    /// Create test Bell state
    pub fn create_test_bell_state() -> QuantumState {
        quantum_utils::bell_state(2).unwrap()
    }
    
    /// Create test random state
    pub fn create_test_random_state(num_qubits: usize) -> QuantumState {
        quantum_utils::random_quantum_state(num_qubits).unwrap()
    }
    
    /// Assert states are approximately equal
    pub fn assert_states_approx_equal(
        state1: &QuantumState,
        state2: &QuantumState,
        tolerance: f64,
    ) -> QuantumResult<()> {
        if state1.num_qubits() != state2.num_qubits() {
            return Err(QuantumError::InvalidOperation { 
                operation: "State comparison".to_string(), 
                message: "States have different number of qubits".to_string() 
            });
        }
        
        let size = 1 << state1.num_qubits();
        for i in 0..size {
            let amp1 = state1.get_amplitude(i)?;
            let amp2 = state2.get_amplitude(i)?;
            
            if (amp1 - amp2).norm() > tolerance {
                return Err(QuantumError::InvalidOperation { 
                    operation: "State comparison".to_string(), 
                    message: format!("States differ at index {}: {} vs {}", i, amp1, amp2) 
                });
            }
        }
        
        Ok(())
    }
    
    /// Assert probability is approximately equal
    pub fn assert_prob_approx_equal(actual: f64, expected: f64, tolerance: f64) -> QuantumResult<()> {
        if (actual - expected).abs() > tolerance {
            return Err(QuantumError::InvalidOperation { 
                operation: "Probability comparison".to_string(), 
                message: format!("Probabilities differ: {} vs {} (tolerance: {})", actual, expected, tolerance) 
            });
        }
        Ok(())
    }
    
    /// Generate test parameters
    pub fn generate_test_angles(count: usize) -> Vec<f64> {
        let mut angles = Vec::with_capacity(count);
        for i in 0..count {
            angles.push(2.0 * PI * i as f64 / count as f64);
        }
        angles
    }
    
    /// Benchmark function execution
    pub fn benchmark_function<F, R>(func: F, iterations: usize) -> (R, std::time::Duration)
    where
        F: Fn() -> R,
    {
        let start = std::time::Instant::now();
        let mut result = None;
        
        for _ in 0..iterations {
            result = Some(func());
        }
        
        let duration = start.elapsed();
        (result.unwrap(), duration)
    }
}

/// Configuration utilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumConfig {
    pub precision: f64,
    pub max_qubits: usize,
    pub enable_parallel: bool,
    pub cache_size: usize,
    pub optimization_level: u8,
    pub measurement_shots: usize,
    pub random_seed: Option<u64>,
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            precision: constants::ERROR_TOLERANCE,
            max_qubits: 32,
            enable_parallel: true,
            cache_size: 1000,
            optimization_level: 2,
            measurement_shots: 1024,
            random_seed: None,
        }
    }
}

impl QuantumConfig {
    /// Create new configuration
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set precision
    pub fn with_precision(mut self, precision: f64) -> Self {
        self.precision = precision;
        self
    }
    
    /// Set maximum qubits
    pub fn with_max_qubits(mut self, max_qubits: usize) -> Self {
        self.max_qubits = max_qubits;
        self
    }
    
    /// Enable/disable parallel processing
    pub fn with_parallel(mut self, enable: bool) -> Self {
        self.enable_parallel = enable;
        self
    }
    
    /// Set cache size
    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.cache_size = size;
        self
    }
    
    /// Set optimization level
    pub fn with_optimization_level(mut self, level: u8) -> Self {
        self.optimization_level = level;
        self
    }
    
    /// Set measurement shots
    pub fn with_measurement_shots(mut self, shots: usize) -> Self {
        self.measurement_shots = shots;
        self
    }
    
    /// Set random seed
    pub fn with_random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }
    
    /// Validate configuration
    pub fn validate(&self) -> QuantumResult<()> {
        if self.precision <= 0.0 {
            return Err(QuantumError::InvalidOperation { operation: "Config validation".to_string(), message: "Precision must be positive".to_string() });
        }
        
        if self.max_qubits == 0 {
            return Err(QuantumError::InvalidOperation { operation: "Config validation".to_string(), message: "Max qubits must be positive".to_string() });
        }
        
        if self.optimization_level > 3 {
            return Err(QuantumError::InvalidOperation { operation: "Config validation".to_string(), message: "Optimization level must be 0-3".to_string() });
        }
        
        if self.measurement_shots == 0 {
            return Err(QuantumError::InvalidOperation { operation: "Config validation".to_string(), message: "Measurement shots must be positive".to_string() });
        }
        
        Ok(())
    }
}

/// Logging utilities
pub mod logging_utils {
    use tracing::{debug, info, warn, error};
    
    /// Log quantum state information
    pub fn log_state_info(state: &crate::quantum_state::QuantumState, label: &str) {
        info!("Quantum state '{}': {} qubits", label, state.num_qubits());
        
        if let Ok(is_norm) = super::quantum_utils::is_normalized(state) {
            debug!("State '{}' normalized: {}", label, is_norm);
        }
    }
    
    /// Log performance metrics
    pub fn log_performance(operation: &str, duration: std::time::Duration, qubits: usize) {
        info!("Operation '{}' on {} qubits took {:?}", operation, qubits, duration);
    }
    
    /// Log error with context
    pub fn log_error_with_context(error: &crate::error::QuantumError, context: &str) {
        error!("Quantum error in {}: {}", context, error);
    }
    
    /// Log warning with details
    pub fn log_warning_with_details(message: &str, details: &str) {
        warn!("{}: {}", message, details);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum_state::QuantumState;

    #[test]
    fn test_constants() {
        assert_eq!(constants::PI_2, std::f64::consts::PI / 2.0);
        assert_eq!(constants::PI_4, std::f64::consts::PI / 4.0);
        assert_eq!(constants::TWO_PI, 2.0 * std::f64::consts::PI);
        assert!(constants::PLANCK_CONSTANT > 0.0);
        assert!(constants::SPEED_OF_LIGHT > 0.0);
    }

    #[test]
    fn test_random_quantum_state() {
        let state = quantum_utils::random_quantum_state(3).unwrap();
        assert_eq!(state.num_qubits(), 3);
        assert!(quantum_utils::is_normalized(&state).unwrap());
    }

    #[test]
    fn test_bell_state() {
        let state = quantum_utils::bell_state(2).unwrap();
        assert_eq!(state.num_qubits(), 2);
        assert!(quantum_utils::is_normalized(&state).unwrap());
        
        // Check Bell state properties
        let amp_00 = state.get_amplitude(0).unwrap();
        let amp_11 = state.get_amplitude(3).unwrap();
        
        assert!((amp_00.norm() - 1.0 / 2.0_f64.sqrt()).abs() < 1e-10);
        assert!((amp_11.norm() - 1.0 / 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_ghz_state() {
        let state = quantum_utils::ghz_state(3).unwrap();
        assert_eq!(state.num_qubits(), 3);
        assert!(quantum_utils::is_normalized(&state).unwrap());
    }

    #[test]
    fn test_w_state() {
        let state = quantum_utils::w_state(3).unwrap();
        assert_eq!(state.num_qubits(), 3);
        assert!(quantum_utils::is_normalized(&state).unwrap());
        
        // Check W state properties - should have equal amplitudes for single excitation states
        let amp_001 = state.get_amplitude(1).unwrap();
        let amp_010 = state.get_amplitude(2).unwrap();
        let amp_100 = state.get_amplitude(4).unwrap();
        
        let expected_amp = 1.0 / 3.0_f64.sqrt();
        assert!((amp_001.norm() - expected_amp).abs() < 1e-10);
        assert!((amp_010.norm() - expected_amp).abs() < 1e-10);
        assert!((amp_100.norm() - expected_amp).abs() < 1e-10);
    }

    #[test]
    fn test_fidelity_calculation() {
        let state1 = quantum_utils::bell_state(2).unwrap();
        let state2 = quantum_utils::bell_state(2).unwrap();
        
        let fidelity = quantum_utils::calculate_fidelity(&state1, &state2).unwrap();
        assert!((fidelity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_calculation() {
        let state = quantum_utils::bell_state(2).unwrap();
        let entropy = quantum_utils::calculate_entropy(&state).unwrap();
        assert!(entropy > 0.0);
    }

    #[test]
    fn test_math_utils() {
        assert_eq!(math_utils::factorial(5), 120);
        assert_eq!(math_utils::binomial_coefficient(5, 2), 10);
        assert_eq!(math_utils::fibonacci(10), 55);
        assert_eq!(math_utils::gcd(48, 18), 6);
        assert_eq!(math_utils::lcm(12, 18), 36);
        assert!(math_utils::is_prime(17));
        assert!(!math_utils::is_prime(15));
    }

    #[test]
    fn test_conversion_utils() {
        let degrees = 90.0;
        let radians = conversion_utils::deg_to_rad(degrees);
        assert!((radians - std::f64::consts::PI / 2.0).abs() < 1e-10);
        
        let back_to_degrees = conversion_utils::rad_to_deg(radians);
        assert!((back_to_degrees - degrees).abs() < 1e-10);
    }

    #[test]
    fn test_binary_conversion() {
        let binary = "1010";
        let decimal = conversion_utils::binary_to_decimal(binary).unwrap();
        assert_eq!(decimal, 10);
        
        let back_to_binary = conversion_utils::decimal_to_binary(decimal, 4);
        assert_eq!(back_to_binary, binary);
    }

    #[test]
    fn test_validation_utils() {
        let state = testing_utils::create_test_state(2);
        assert!(validation_utils::validate_quantum_state(&state).is_ok());
        
        assert!(validation_utils::validate_qubit_index(0, 2).is_ok());
        assert!(validation_utils::validate_qubit_index(2, 2).is_err());
        
        assert!(validation_utils::validate_probability(0.5).is_ok());
        assert!(validation_utils::validate_probability(1.5).is_err());
    }

    #[test]
    fn test_performance_utils() {
        let timer = performance_utils::Timer::start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let duration = timer.elapsed();
        assert!(duration.as_millis() >= 10);
    }

    #[test]
    fn test_memo_cache() {
        let mut cache = performance_utils::MemoCache::new(100);
        
        let result1 = cache.get_or_compute("key1", || 42);
        assert_eq!(result1, 42);
        
        let result2 = cache.get_or_compute("key1", || 99);
        assert_eq!(result2, 42); // Should return cached value
    }

    #[test]
    fn test_quantum_config() {
        let config = QuantumConfig::new()
            .with_precision(1e-12)
            .with_max_qubits(16)
            .with_parallel(false)
            .with_cache_size(500)
            .with_optimization_level(1)
            .with_measurement_shots(2048)
            .with_random_seed(12345);
        
        assert_eq!(config.precision, 1e-12);
        assert_eq!(config.max_qubits, 16);
        assert!(!config.enable_parallel);
        assert_eq!(config.cache_size, 500);
        assert_eq!(config.optimization_level, 1);
        assert_eq!(config.measurement_shots, 2048);
        assert_eq!(config.random_seed, Some(12345));
        
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_testing_utils() {
        let state1 = testing_utils::create_test_state(2);
        let state2 = testing_utils::create_test_state(2);
        
        assert!(testing_utils::assert_states_approx_equal(&state1, &state2, 1e-10).is_ok());
        
        assert!(testing_utils::assert_prob_approx_equal(0.5, 0.5, 1e-10).is_ok());
        assert!(testing_utils::assert_prob_approx_equal(0.5, 0.6, 1e-10).is_err());
    }

    #[test]
    fn test_complex_polar_conversion() {
        let complex = Complex64::new(1.0, 1.0);
        let (magnitude, phase) = conversion_utils::complex_to_polar(complex);
        let back_to_complex = conversion_utils::polar_to_complex(magnitude, phase);
        
        assert!((complex - back_to_complex).norm() < 1e-10);
    }

    #[test]
    fn test_state_to_probabilities() {
        let state = testing_utils::create_test_bell_state();
        let probabilities = conversion_utils::state_to_probabilities(&state).unwrap();
        
        assert_eq!(probabilities.len(), 4);
        assert!((probabilities[0] - 0.5).abs() < 1e-10);
        assert!((probabilities[3] - 0.5).abs() < 1e-10);
        assert!((probabilities[1] - 0.0).abs() < 1e-10);
        assert!((probabilities[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_density_matrix() {
        let state = testing_utils::create_test_state(2);
        let density_matrix = quantum_utils::to_density_matrix(&state).unwrap();
        
        assert_eq!(density_matrix.len(), 4);
        assert_eq!(density_matrix[0].len(), 4);
        
        let trace = quantum_utils::trace_density_matrix(&density_matrix);
        assert!((trace.re - 1.0).abs() < 1e-10);
        assert!(trace.im.abs() < 1e-10);
    }

    #[test]
    fn test_matrix_operations() {
        let matrix = [[1.0, 2.0], [3.0, 4.0]];
        let det = math_utils::det_2x2(&matrix);
        assert_eq!(det, -2.0);
        
        let inv = math_utils::inv_2x2(&matrix);
        assert!(inv.is_some());
        
        let inv_matrix = inv.unwrap();
        assert!((inv_matrix[0][0] - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_operations() {
        let mut amplitudes = vec![Complex64::new(1.0, 0.0); 100];
        
        performance_utils::parallel_amplitude_operation(&mut amplitudes, |amp| {
            *amp *= 2.0;
        }).unwrap();
        
        for amp in &amplitudes {
            assert_eq!(amp.re, 2.0);
            assert_eq!(amp.im, 0.0);
        }
    }

    #[test]
    fn test_benchmark_function() {
        let (result, duration) = testing_utils::benchmark_function(|| {
            math_utils::factorial(10)
        }, 1000);
        
        assert_eq!(result, 3628800);
        assert!(duration.as_nanos() > 0);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let distance = math_utils::euclidean_distance(&a, &b);
        
        // Distance should be sqrt(9 + 9 + 9) = sqrt(27) ≈ 5.196
        assert!((distance - 27.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_modular_exponentiation() {
        let result = math_utils::mod_exp(2, 10, 1000);
        assert_eq!(result, 24); // 2^10 mod 1000 = 1024 mod 1000 = 24
    }

    #[test]
    fn test_linear_interpolation() {
        let result = math_utils::lerp(0.0, 10.0, 0.5);
        assert_eq!(result, 5.0);
        
        let result = math_utils::lerp(0.0, 10.0, 0.0);
        assert_eq!(result, 0.0);
        
        let result = math_utils::lerp(0.0, 10.0, 1.0);
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_clamp() {
        assert_eq!(math_utils::clamp(5.0, 0.0, 10.0), 5.0);
        assert_eq!(math_utils::clamp(-5.0, 0.0, 10.0), 0.0);
        assert_eq!(math_utils::clamp(15.0, 0.0, 10.0), 10.0);
    }

    #[test]
    fn test_normalize_amplitudes() {
        let mut amplitudes = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        
        quantum_utils::normalize_amplitudes(&mut amplitudes).unwrap();
        
        let norm = quantum_utils::state_norm(&amplitudes);
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_test_angles_generation() {
        let angles = testing_utils::generate_test_angles(4);
        assert_eq!(angles.len(), 4);
        assert_eq!(angles[0], 0.0);
        assert!((angles[1] - std::f64::consts::PI / 2.0).abs() < 1e-10);
        assert!((angles[2] - std::f64::consts::PI).abs() < 1e-10);
        assert!((angles[3] - 3.0 * std::f64::consts::PI / 2.0).abs() < 1e-10);
    }
}