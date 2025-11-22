//! Quantum state encoding methods

use crate::{error::Result, types::*, core::QuantumCircuit};
use ndarray::{Array1, s};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Quantum state encoder
pub struct QuantumStateEncoder {
    num_qubits: usize,
    encoding_type: EncodingType,
}

impl QuantumStateEncoder {
    /// Create a new encoder
    pub fn new(num_qubits: usize, encoding_type: EncodingType) -> Self {
        Self { num_qubits, encoding_type }
    }
    
    /// Encode classical data into quantum state
    pub fn encode(&self, data: &Array1<f64>) -> Result<QuantumState> {
        match self.encoding_type {
            EncodingType::Amplitude => self.amplitude_encoding(data),
            EncodingType::Angle => self.angle_encoding(data),
            EncodingType::Basis => self.basis_encoding(data),
            EncodingType::IQP => self.iqp_encoding(data),
            EncodingType::Hybrid => self.hybrid_encoding(data),
        }
    }
    
    /// Amplitude encoding
    fn amplitude_encoding(&self, data: &Array1<f64>) -> Result<QuantumState> {
        let n_amplitudes = 1 << self.num_qubits;
        let mut amplitudes = Array1::<Complex64>::zeros(n_amplitudes);
        
        // Normalize and encode data
        let data_norm = data.dot(data).sqrt();
        if data_norm < 1e-10 {
            return Err("Input data norm too small for amplitude encoding".into());
        }
        
        for (i, &val) in data.iter().enumerate() {
            if i < n_amplitudes {
                amplitudes[i] = Complex64::new(val / data_norm, 0.0);
            }
        }
        
        Ok(QuantumState {
            amplitudes,
            num_qubits: self.num_qubits,
            global_phase: None,
        })
    }
    
    /// Angle encoding
    fn angle_encoding(&self, data: &Array1<f64>) -> Result<QuantumState> {
        let mut circuit = QuantumCircuit::new(self.num_qubits);
        
        // Apply rotation gates based on data
        for (i, &val) in data.iter().enumerate() {
            if i < self.num_qubits {
                circuit.add_gate(GateType::RY(val * PI), &[i])?;
            }
        }
        
        // For now, return a placeholder state
        let n_amplitudes = 1 << self.num_qubits;
        let amplitudes = Array1::<Complex64>::from_elem(n_amplitudes, Complex64::new(1.0 / (n_amplitudes as f64).sqrt(), 0.0));
        
        Ok(QuantumState {
            amplitudes,
            num_qubits: self.num_qubits,
            global_phase: None,
        })
    }
    
    /// Basis encoding
    fn basis_encoding(&self, data: &Array1<f64>) -> Result<QuantumState> {
        let n_amplitudes = 1 << self.num_qubits;
        let mut amplitudes = Array1::<Complex64>::zeros(n_amplitudes);
        
        // Encode integer value in computational basis
        if let Some(&first_val) = data.get(0) {
            let index = (first_val.round() as usize) % n_amplitudes;
            amplitudes[index] = Complex64::new(1.0, 0.0);
        } else {
            amplitudes[0] = Complex64::new(1.0, 0.0);
        }
        
        Ok(QuantumState {
            amplitudes,
            num_qubits: self.num_qubits,
            global_phase: None,
        })
    }
    
    /// IQP encoding
    fn iqp_encoding(&self, data: &Array1<f64>) -> Result<QuantumState> {
        // Simplified IQP encoding
        self.angle_encoding(data)
    }
    
    /// Hybrid encoding
    fn hybrid_encoding(&self, data: &Array1<f64>) -> Result<QuantumState> {
        // Combine amplitude and angle encoding
        let amplitude_state = self.amplitude_encoding(&data.slice(s![..self.num_qubits/2]).to_owned())?;
        let angle_state = self.angle_encoding(&data.slice(s![self.num_qubits/2..]).to_owned())?;
        
        // Combine states (simplified)
        Ok(amplitude_state)
    }
}