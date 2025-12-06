//! Quantum circuits for NQO using roqoqo
//!
//! This module implements quantum optimization circuits including
//! QAOA, parameter exploration, and variational quantum algorithms.

use crate::error::{NqoError, NqoResult};
use roqoqo::prelude::*;
use roqoqo::Circuit;
use roqoqo::operations::{
    RotateX, RotateY, RotateZ, Hadamard, CNOT, MeasureQubit,
    PragmaGetPauliProduct, DefinitionBit, DefinitionFloat
};
use roqoqo::registers::Registers;
use roqoqo_quest::Backend;
use std::f64::consts::PI;
// use struqture::prelude::*; // Temporarily disabled due to dependency conflict

/// Quantum backend for circuit execution
pub struct QuantumBackend {
    backend: Backend,
    num_qubits: usize,
    shots: Option<usize>,
}

impl QuantumBackend {
    /// Create a new quantum backend
    pub fn new(num_qubits: usize, shots: Option<usize>) -> NqoResult<Self> {
        if num_qubits == 0 || num_qubits > 20 {
            return Err(NqoError::InvalidParameter(
                "Number of qubits must be between 1 and 20".to_string(),
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
    pub fn execute(&mut self, circuit: &Circuit) -> NqoResult<Vec<f64>> {
        let result = self.backend.run_circuit(circuit)
            .map_err(|e| NqoError::QuantumError(format!("Circuit execution failed: {}", e)))?;
        
        let measurements = self.extract_measurements(&result)?;
        Ok(measurements)
    }
    
    /// Extract measurements from backend result
    fn extract_measurements(&self, result: &Registers) -> NqoResult<Vec<f64>> {
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

/// Quantum circuit builder for NQO
pub struct QuantumOptimizationCircuits {
    num_qubits: usize,
}

impl QuantumOptimizationCircuits {
    /// Create a new quantum circuit builder
    pub fn new(num_qubits: usize) -> Self {
        Self { num_qubits }
    }
    
    /// Build parameter exploration circuit
    ///
    /// Explores parameter space using quantum superposition
    pub fn build_parameter_exploration_circuit(&self, parameters: &[f64]) -> NqoResult<Circuit> {
        let mut circuit = Circuit::new();
        
        // Encode parameters into quantum state
        for i in 0..self.num_qubits {
            let angle = PI * parameters[i % parameters.len()];
            circuit += RotateY::new(i, angle.into());
        }
        
        // Apply entangling layers
        for i in 0..self.num_qubits.saturating_sub(1) {
            circuit += CNOT::new(i, i + 1);
        }
        
        // Apply rotation gates
        for i in 0..self.num_qubits {
            let angle = parameters[(i + self.num_qubits) % parameters.len()];
            circuit += RotateX::new(i, (angle * PI).into());
            circuit += RotateZ::new(i, (angle * PI / 2.0).into());
        }
        
        // Additional entanglement
        for i in (1..self.num_qubits).rev() {
            circuit += CNOT::new(i, i - 1);
        }
        
        // Define registers for measurements
        circuit += DefinitionBit::new("measurements".to_string(), self.num_qubits, true);
        circuit += DefinitionFloat::new("measurements".to_string(), self.num_qubits, true);
        
        // Measurements
        for i in 0..self.num_qubits {
            circuit += MeasureQubit::new(i, "measurements".to_string(), i);
            circuit += PragmaGetPauliProduct::new(
                std::collections::HashMap::from([(i, 2)]),
                "measurements".to_string(),
                Circuit::new(),
            );
        }
        
        Ok(circuit)
    }
    
    /// Build QAOA-inspired optimization circuit
    ///
    /// Implements Quantum Approximate Optimization Algorithm
    pub fn build_qaoa_circuit(&self, params: &[f64], gradients: &[f64]) -> NqoResult<Circuit> {
        let mut circuit = Circuit::new();
        
        // Initial state preparation
        for i in 0..self.num_qubits {
            let angle = PI * params[i % params.len()];
            circuit += RotateY::new(i, angle.into());
        }
        
        // First layer - cost Hamiltonian
        for i in 0..self.num_qubits {
            let angle = PI * gradients[i % gradients.len()];
            circuit += RotateZ::new(i, angle.into());
        }
        
        // Mixing Hamiltonian
        for i in 0..self.num_qubits {
            circuit += RotateX::new(i, (PI / 2.0).into());
        }
        
        // Second layer - cost Hamiltonian with reduced angle
        for i in 0..self.num_qubits {
            let angle = PI * gradients[i % gradients.len()] / 2.0;
            circuit += RotateZ::new(i, angle.into());
        }
        
        // Final mixing for measurement
        for i in 0..self.num_qubits {
            circuit += Hadamard::new(i);
        }
        
        // Define registers for measurements
        circuit += DefinitionBit::new("measurements".to_string(), self.num_qubits, true);
        circuit += DefinitionFloat::new("measurements".to_string(), self.num_qubits, true);
        
        // Measurements
        for i in 0..self.num_qubits {
            circuit += MeasureQubit::new(i, "measurements".to_string(), i);
            circuit += PragmaGetPauliProduct::new(
                std::collections::HashMap::from([(i, 2)]),
                "measurements".to_string(),
                Circuit::new(),
            );
        }
        
        Ok(circuit)
    }
    
    /// Build optimization circuit with neural network weights
    pub fn build_neural_quantum_circuit(
        &self,
        parameters: &[f64],
        weights: &[f64],
    ) -> NqoResult<Circuit> {
        let mut circuit = Circuit::new();
        
        // Encode parameters
        for i in 0..self.num_qubits {
            let angle = PI * parameters[i % parameters.len()];
            circuit += RotateY::new(i, angle.into());
        }
        
        // Apply weighted transformation
        let weight_sum: f64 = weights.iter().sum();
        for i in 0..self.num_qubits {
            circuit += RotateZ::new(i, (weight_sum * 0.1).into());
        }
        
        // Entangling layers
        for i in 0..self.num_qubits.saturating_sub(1) {
            circuit += CNOT::new(i, i + 1);
        }
        
        // Variational layers
        for i in 0..self.num_qubits {
            let angle = parameters[(i + self.num_qubits) % parameters.len()];
            circuit += RotateX::new(i, (angle * PI).into());
        }
        
        // Final layer
        for i in 0..self.num_qubits {
            let angle = parameters[(i + 2 * self.num_qubits) % parameters.len()];
            circuit += RotateY::new(i, (angle * PI).into());
        }
        
        // Define registers for measurements
        circuit += DefinitionBit::new("measurements".to_string(), self.num_qubits, true);
        circuit += DefinitionFloat::new("measurements".to_string(), self.num_qubits, true);
        
        // Measurements
        for i in 0..self.num_qubits {
            circuit += MeasureQubit::new(i, "measurements".to_string(), i);
            circuit += PragmaGetPauliProduct::new(
                std::collections::HashMap::from([(i, 2)]),
                "measurements".to_string(),
                Circuit::new(),
            );
        }
        
        Ok(circuit)
    }
    
    /// Build parameter refinement circuit
    pub fn build_refinement_circuit(
        &self,
        parameters: &[f64],
        optimal_direction: &[f64],
    ) -> NqoResult<Circuit> {
        let mut circuit = Circuit::new();
        
        // Encode parameters
        for i in 0..self.num_qubits {
            let angle = PI * parameters[i % parameters.len()];
            circuit += RotateY::new(i, angle.into());
        }
        
        // Apply directional bias
        for i in 0..self.num_qubits {
            let angle = optimal_direction[i % optimal_direction.len()];
            circuit += RotateZ::new(i, angle.into());
        }
        
        // Entangling layer
        for i in 0..self.num_qubits.saturating_sub(1) {
            circuit += CNOT::new(i, i + 1);
        }
        
        // Apply variational layer in optimal direction
        for i in 0..self.num_qubits {
            let combined_angle = optimal_direction[i % optimal_direction.len()] 
                * parameters[i % parameters.len()];
            circuit += RotateY::new(i, combined_angle.into());
        }
        
        // Define registers for measurements
        circuit += DefinitionBit::new("measurements".to_string(), self.num_qubits, true);
        circuit += DefinitionFloat::new("measurements".to_string(), self.num_qubits, true);
        
        // Measurements
        for i in 0..self.num_qubits {
            circuit += MeasureQubit::new(i, "measurements".to_string(), i);
            circuit += PragmaGetPauliProduct::new(
                std::collections::HashMap::from([(i, 2)]),
                "measurements".to_string(),
                Circuit::new(),
            );
        }
        
        Ok(circuit)
    }
}

/// SIMD-accelerated quantum operations
#[cfg(feature = "simd")]
pub mod simd_ops {
    #[allow(unused_imports)]
    use super::*;
    use wide::*;
    
    /// Fast gradient computation using SIMD
    pub fn compute_gradient_simd(
        objective_values: &[f64],
        parameters: &[f64],
        epsilon: f64,
    ) -> Vec<f64> {
        let mut gradients = vec![0.0; parameters.len()];
        let eps_vec = f64x4::splat(epsilon);
        
        // Process 4 elements at a time using wide crate
        let chunks = objective_values.chunks_exact(4);
        let remainder = chunks.remainder();
        
        for (i, chunk) in chunks.enumerate() {
            if i * 4 + 4 < objective_values.len() {
                let vals = f64x4::from(&chunk[..]);
                let vals_plus = if (i * 4 + 5) <= objective_values.len() {
                    f64x4::from(&objective_values[(i * 4 + 1)..(i * 4 + 5)])
                } else {
                    continue;
                };
                
                let diff = vals_plus - vals;
                let grad = diff / eps_vec;
                
                let grad_arr: [f64; 4] = grad.into();
                if i * 4 < gradients.len() {
                    gradients[i * 4] = grad_arr[0];
                }
            }
        }
        
        // Process remainder
        for i in (objective_values.len() - remainder.len())..parameters.len() {
            if i + 1 < objective_values.len() {
                gradients[i] = (objective_values[i + 1] - objective_values[i]) / epsilon;
            }
        }
        
        gradients
    }
    
    /// Fast parameter update using SIMD
    pub fn update_parameters_simd(
        parameters: &mut [f64],
        gradients: &[f64],
        learning_rate: f64,
    ) {
        let lr_vec = f64x4::splat(learning_rate);
        let len = parameters.len().min(gradients.len());
        
        // Process 4 elements at a time
        for i in (0..len).step_by(4) {
            if i + 3 < len {
                let params = f64x4::from(&parameters[i..i+4]);
                let grads = f64x4::from(&gradients[i..i+4]);
                
                let update = grads * lr_vec;
                let new_params = params - update;
                
                let result: [f64; 4] = new_params.into();
                parameters[i..i+4].copy_from_slice(&result);
            }
        }
        
        // Handle remainder
        for i in (len / 4 * 4)..len {
            parameters[i] -= learning_rate * gradients[i];
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
        
        let backend = QuantumBackend::new(21, None);
        assert!(backend.is_err());
    }
    
    #[test]
    fn test_circuit_building() {
        let circuits = QuantumOptimizationCircuits::new(4);
        
        let params = vec![0.1, 0.2, 0.3, 0.4];
        let grads = vec![0.01, 0.02, 0.03, 0.04];
        
        let circuit = circuits.build_qaoa_circuit(&params, &grads);
        assert!(circuit.is_ok());
    }
}