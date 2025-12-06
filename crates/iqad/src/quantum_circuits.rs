//! Quantum circuits for IQAD using roqoqo
//!
//! This module implements real quantum circuits for anomaly detection,
//! including affinity calculation, detector generation, and scoring.

use crate::error::{IqadError, IqadResult};
use roqoqo::prelude::*;
use roqoqo::Circuit;
use roqoqo::operations::{
    RotateY, Hadamard, PhaseShiftState1, CNOT, MeasureQubit, 
    PragmaGetPauliProduct, RotateZ, ControlledPauliZ
};
use roqoqo::registers::Registers;
use roqoqo_quest::Backend;
use std::f64::consts::PI;

/// Quantum backend for circuit execution
pub struct QuantumBackend {
    backend: Backend,
    num_qubits: usize,
    shots: Option<usize>,
}

impl QuantumBackend {
    /// Create a new quantum backend
    pub fn new(num_qubits: usize, shots: Option<usize>) -> IqadResult<Self> {
        if num_qubits == 0 || num_qubits > 30 {
            return Err(IqadError::InvalidParameter(
                "Number of qubits must be between 1 and 30".to_string(),
            ));
        }
        
        let backend = Backend::new(num_qubits);
        
        Ok(Self {
            backend,
            num_qubits,
            shots,
        })
    }
    
    /// Execute a quantum circuit
    pub fn execute(&mut self, circuit: &Circuit) -> IqadResult<Vec<f64>> {
        // Run the circuit
        let result = self.backend.run_circuit(circuit)
            .map_err(|e| IqadError::QuantumError(format!("Circuit execution failed: {}", e)))?;
        
        // Extract expectation values
        let measurements = self.extract_measurements(&result)?;
        Ok(measurements)
    }
    
    /// Extract measurements from backend result
    fn extract_measurements(&self, result: &Registers) -> IqadResult<Vec<f64>> {
        // Extract float register values
        if let Some(float_output_reg) = result.1.get("measurements") {
            // float_output_reg is Vec<Vec<f64>> (FloatOutputRegister)
            // We take the first run's results (first Vec<f64>)
            if let Some(float_reg) = float_output_reg.first() {
                // float_reg is Vec<f64> (FloatRegister)
                let mut measurements = Vec::with_capacity(self.num_qubits);
                for i in 0..self.num_qubits {
                    if let Some(&value) = float_reg.get(i) {
                        measurements.push(value);
                    } else {
                        measurements.push(0.0);
                    }
                }
                Ok(measurements)
            } else {
                // No measurement results
                Ok(vec![0.0; self.num_qubits])
            }
        } else {
            // Fallback: return zeros if no measurements
            Ok(vec![0.0; self.num_qubits])
        }
    }
}

/// Quantum circuit builder for IQAD
pub struct QuantumCircuits {
    num_qubits: usize,
}

impl QuantumCircuits {
    /// Create a new quantum circuit builder
    pub fn new(num_qubits: usize) -> Self {
        Self { num_qubits }
    }
    
    /// Build quantum affinity circuit
    ///
    /// Calculates quantum affinity between pattern and detector
    pub fn build_affinity_circuit(&self, pattern: &[f64], detector: &[f64]) -> IqadResult<Circuit> {
        let mut circuit = Circuit::new();
        
        // Encode pattern into quantum state
        for i in 0..self.num_qubits {
            let angle = PI * pattern[i % pattern.len()];
            circuit += RotateY::new(i, angle.into());
        }
        
        // Apply Hadamard gates for superposition
        for i in 0..self.num_qubits {
            circuit += Hadamard::new(i);
        }
        
        // Apply phase based on detector
        for i in 0..self.num_qubits {
            let phase = detector[i % detector.len()] * PI;
            circuit += PhaseShiftState1::new(i, phase.into());
        }
        
        // Entangling operations
        for i in 0..self.num_qubits.saturating_sub(1) {
            circuit += CNOT::new(i, i + 1);
        }
        
        // Final Hadamard layer
        for i in 0..self.num_qubits {
            circuit += Hadamard::new(i);
        }
        
        // Measure all qubits
        for i in 0..self.num_qubits {
            circuit += MeasureQubit::new(i, "measurements".to_string(), i);
            circuit += PragmaGetPauliProduct::new(
                std::collections::HashMap::from([(i, 2)]), // Z measurement
                "measurements".to_string(),
                Circuit::new(),
            );
        }
        
        Ok(circuit)
    }
    
    /// Build detector generation circuit
    ///
    /// Generates quantum detector using negative selection
    pub fn build_detector_generation_circuit(
        &self,
        self_pattern: &[f64],
        random_seed: &[f64],
    ) -> IqadResult<Circuit> {
        let mut circuit = Circuit::new();
        
        // Encode self pattern to avoid
        for i in 0..self.num_qubits {
            let angle = PI * self_pattern[i % self_pattern.len()];
            circuit += RotateY::new(i, angle.into());
        }
        
        // Apply random rotations based on seed
        for i in 0..self.num_qubits {
            let angle = random_seed[i % random_seed.len()] * PI;
            circuit += RotateY::new(i, angle.into());
        }
        
        // Negative selection logic via entanglement
        for i in 0..self.num_qubits.saturating_sub(1) {
            circuit += CNOT::new(i, i + 1);
        }
        
        // Additional rotation layer
        for i in 0..self.num_qubits {
            let angle = random_seed[(i + self.num_qubits) % random_seed.len()] * PI;
            circuit += RotateZ::new(i, angle.into());
        }
        
        // Generate detector different from self
        for i in 0..self.num_qubits {
            circuit += Hadamard::new(i);
        }
        
        // Measurements
        for i in 0..self.num_qubits {
            circuit += MeasureQubit::new(i, "measurements".to_string(), i);
            circuit += PragmaGetPauliProduct::new(
                std::collections::HashMap::from([(i, 2)]), // Z measurement
                "measurements".to_string(),
                Circuit::new(),
            );
        }
        
        Ok(circuit)
    }
    
    /// Build anomaly scoring circuit
    ///
    /// Scores pattern against multiple detectors
    pub fn build_anomaly_scoring_circuit(
        &self,
        pattern: &[f64],
        detectors: &[Vec<f64>],
    ) -> IqadResult<Circuit> {
        let mut circuit = Circuit::new();
        
        // Encode pattern
        for i in 0..self.num_qubits {
            let angle = PI * pattern[i % pattern.len()];
            circuit += RotateY::new(i, angle.into());
        }
        
        // Process first few detectors (limit for efficiency)
        let max_detectors = 3.min(detectors.len());
        for (_d_idx, detector) in detectors.iter().take(max_detectors).enumerate() {
            // Calculate detector influence angle
            let detector_sum: f64 = detector.iter().take(4).sum();
            let angle = detector_sum * PI / 4.0;
            
            // Apply rotation based on detector
            circuit += RotateY::new(0, angle.into());
            
            // Controlled operations
            for j in 1..self.num_qubits {
                // let control_angle = detector[j % detector.len()] * PI;
                circuit += ControlledPauliZ::new(0, j);
            }
        }
        
        // Final measurement layer
        for i in 0..self.num_qubits {
            circuit += Hadamard::new(i);
        }
        
        // Measure anomaly score
        for i in 0..self.num_qubits {
            circuit += MeasureQubit::new(i, "measurements".to_string(), i);
            circuit += PragmaGetPauliProduct::new(
                std::collections::HashMap::from([(i, 2)]), // Z measurement
                "measurements".to_string(),
                Circuit::new(),
            );
        }
        
        Ok(circuit)
    }
    
    /// Build quantum distance circuit
    ///
    /// Calculates quantum distance between two vectors
    pub fn build_distance_circuit(&self, x: &[f64], y: &[f64]) -> IqadResult<Circuit> {
        let mut circuit = Circuit::new();
        
        // Amplitude encoding for first vector
        for i in 0..self.num_qubits {
            let angle = (x[i % x.len()].max(-1.0).min(1.0)).acos();
            circuit += RotateY::new(i, angle.into());
        }
        
        // Hadamard for superposition
        for i in 0..self.num_qubits {
            circuit += Hadamard::new(i);
        }
        
        // Controlled rotations based on second vector
        for i in 0..self.num_qubits {
            let angle = (y[i % y.len()].max(-1.0).min(1.0)).acos();
            circuit += RotateZ::new(i, angle.into());
        }
        
        // Entangling operations
        for i in 0..self.num_qubits.saturating_sub(1) {
            circuit += CNOT::new(i, i + 1);
        }
        
        // Measurements
        for i in 0..self.num_qubits {
            circuit += MeasureQubit::new(i, "measurements".to_string(), i);
            circuit += PragmaGetPauliProduct::new(
                std::collections::HashMap::from([(i, 2)]), // Z measurement
                "measurements".to_string(),
                Circuit::new(),
            );
        }
        
        Ok(circuit)
    }
}

/// SIMD-accelerated quantum state operations
#[cfg(feature = "simd")]
pub mod simd_ops {
    #[allow(unused_imports)]
    use super::*;
    use wide::*;
    
    /// Fast vector normalization using SIMD
    pub fn normalize_vector_simd(vec: &mut [f64]) {
        let mut norm_squared = 0.0;
        
        // Calculate norm squared using SIMD with wide crate
        let chunks = vec.chunks_exact(4);
        let remainder = chunks.remainder();
        
        // Process 4 elements at a time
        for chunk in chunks {
            let v = f64x4::from(chunk);
            let squared = v * v;
            let squared_arr: [f64; 4] = squared.into();
            norm_squared += squared_arr.iter().sum::<f64>();
        }
        
        // Process remainder
        for &val in remainder {
            norm_squared += val * val;
        }
        
        // Normalize vector
        let norm = norm_squared.sqrt();
        if norm > 1e-10 {
            let norm_vec = f64x4::splat(norm);
            
            // Process in chunks of 4
            let mut chunks = vec.chunks_exact_mut(4);
            
            for chunk in chunks.by_ref() {
                let v = f64x4::from(&chunk[..]);
                let normalized = v / norm_vec;
                let result: [f64; 4] = normalized.into();
                chunk.copy_from_slice(&result);
            }
            
            let remainder = chunks.into_remainder();
            
            // Process remainder
            for val in remainder {
                *val /= norm;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quantum_backend_creation() {
        let backend = QuantumBackend::new(4, None);
        assert!(backend.is_ok());
        
        let backend = QuantumBackend::new(0, None);
        assert!(backend.is_err());
        
        let backend = QuantumBackend::new(31, None);
        assert!(backend.is_err());
    }
    
    #[test]
    fn test_circuit_building() {
        let circuits = QuantumCircuits::new(4);
        
        let pattern = vec![0.1, 0.2, 0.3, 0.4];
        let detector = vec![0.5, 0.6, 0.7, 0.8];
        
        let circuit = circuits.build_affinity_circuit(&pattern, &detector);
        assert!(circuit.is_ok());
    }
}