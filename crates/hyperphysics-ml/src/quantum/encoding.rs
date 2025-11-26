//! Quantum-inspired state encoding schemes
//!
//! Provides methods to encode classical data into quantum-inspired state representations.
//! These encodings allow classical neural networks to leverage properties of quantum
//! state spaces without requiring quantum hardware.
//!
//! # Encoding Types
//!
//! - **Amplitude**: Data values become complex amplitudes (normalized)
//! - **Angle**: Data values become rotation angles
//! - **Phase**: Data values become phase factors
//! - **IQP**: Instantaneous Quantum Polynomial encoding
//! - **Hybrid**: Combination of amplitude and phase

use super::types::{Complex, StateVector, EncodingType};
use std::f32::consts::PI;

/// Quantum-inspired state encoder
/// Transforms classical vectors into quantum-inspired state representations
#[derive(Debug, Clone)]
pub struct StateEncoder {
    /// Encoding method
    encoding_type: EncodingType,
    /// Number of virtual qubits
    num_qubits: usize,
    /// Scaling factor for angle encoding
    angle_scale: f32,
    /// Whether to include entanglement-like correlations
    use_correlations: bool,
}

impl StateEncoder {
    /// Create new encoder
    pub fn new(encoding_type: EncodingType, num_qubits: usize) -> Self {
        Self {
            encoding_type,
            num_qubits,
            angle_scale: PI,
            use_correlations: true,
        }
    }

    /// Create amplitude encoder
    pub fn amplitude(num_qubits: usize) -> Self {
        Self::new(EncodingType::Amplitude, num_qubits)
    }

    /// Create angle encoder
    pub fn angle(num_qubits: usize) -> Self {
        Self::new(EncodingType::Angle, num_qubits)
    }

    /// Create phase encoder
    pub fn phase(num_qubits: usize) -> Self {
        Self::new(EncodingType::Phase, num_qubits)
    }

    /// Set angle scaling factor
    pub fn with_angle_scale(mut self, scale: f32) -> Self {
        self.angle_scale = scale;
        self
    }

    /// Enable/disable correlation encoding
    pub fn with_correlations(mut self, use_correlations: bool) -> Self {
        self.use_correlations = use_correlations;
        self
    }

    /// Get state dimension
    pub fn state_dim(&self) -> usize {
        1 << self.num_qubits
    }

    /// Encode a classical vector into quantum-inspired state
    pub fn encode(&self, data: &[f32]) -> StateVector {
        match self.encoding_type {
            EncodingType::Amplitude => self.amplitude_encoding(data),
            EncodingType::Angle => self.angle_encoding(data),
            EncodingType::Phase => self.phase_encoding(data),
            EncodingType::Hybrid => self.hybrid_encoding(data),
            EncodingType::IQP => self.iqp_encoding(data),
        }
    }

    /// Amplitude encoding: x → |ψ⟩ = Σᵢ xᵢ/‖x‖ |i⟩
    ///
    /// Data values become complex amplitudes, normalized to unit length.
    /// This is efficient for encoding 2^n values into n qubits.
    fn amplitude_encoding(&self, data: &[f32]) -> StateVector {
        let dim = self.state_dim();
        let mut amplitudes = vec![Complex::zero(); dim];

        // Compute norm for normalization
        let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm = if norm < 1e-10 { 1.0 } else { norm };

        // Encode data as amplitudes
        for (i, &val) in data.iter().enumerate() {
            if i < dim {
                amplitudes[i] = Complex::new(val / norm, 0.0);
            }
        }

        // If data is shorter than state dimension, normalize what we have
        if data.len() < dim {
            let partial_norm: f32 = amplitudes.iter()
                .map(|a| a.norm_sq())
                .sum::<f32>()
                .sqrt();
            if partial_norm > 1e-10 {
                for a in &mut amplitudes {
                    *a = *a * (1.0 / partial_norm);
                }
            }
        }

        StateVector { amplitudes, num_qubits: self.num_qubits }
    }

    /// Angle encoding: xᵢ → Rᵧ(θᵢ)|0⟩ where θᵢ = xᵢ * scale
    ///
    /// Each feature becomes a rotation angle, creating superpositions.
    /// Good for periodic/oscillatory data.
    fn angle_encoding(&self, data: &[f32]) -> StateVector {
        let dim = self.state_dim();
        let mut amplitudes = vec![Complex::zero(); dim];

        // Start with |0...0⟩ state
        amplitudes[0] = Complex::one();

        // Apply single-qubit Y-rotations
        for (qubit, &val) in data.iter().enumerate().take(self.num_qubits) {
            let theta = val * self.angle_scale;
            let cos_half = (theta / 2.0).cos();
            let sin_half = (theta / 2.0).sin();

            // Apply Ry rotation to qubit
            let mut new_amplitudes = vec![Complex::zero(); dim];
            for i in 0..dim {
                let bit = (i >> qubit) & 1;
                let partner = i ^ (1 << qubit);

                if bit == 0 {
                    // |0⟩ → cos(θ/2)|0⟩ + sin(θ/2)|1⟩
                    new_amplitudes[i] = new_amplitudes[i] + amplitudes[i] * cos_half;
                    new_amplitudes[partner] = new_amplitudes[partner] + amplitudes[i] * sin_half;
                } else {
                    // |1⟩ → -sin(θ/2)|0⟩ + cos(θ/2)|1⟩
                    new_amplitudes[partner] = new_amplitudes[partner] + amplitudes[i] * (-sin_half);
                    new_amplitudes[i] = new_amplitudes[i] + amplitudes[i] * cos_half;
                }
            }
            amplitudes = new_amplitudes;
        }

        // Add entanglement-like correlations
        if self.use_correlations && data.len() >= 2 {
            amplitudes = self.apply_correlations(&amplitudes, data);
        }

        StateVector { amplitudes, num_qubits: self.num_qubits }
    }

    /// Phase encoding: xᵢ → e^(i*xᵢ)|i⟩
    ///
    /// Data values become phases, preserving magnitude information
    /// in the phase relationships.
    fn phase_encoding(&self, data: &[f32]) -> StateVector {
        let dim = self.state_dim();
        let base_amp = 1.0 / (dim as f32).sqrt();

        // Start with uniform superposition
        let mut amplitudes: Vec<Complex> = (0..dim)
            .map(|i| {
                // Phase is sum of data[j] for each qubit j that is |1⟩ in basis state i
                let mut phase = 0.0;
                for (j, &val) in data.iter().enumerate().take(self.num_qubits) {
                    if (i >> j) & 1 == 1 {
                        phase += val * self.angle_scale;
                    }
                }
                Complex::from_polar(base_amp, phase)
            })
            .collect();

        // Normalize
        let norm: f32 = amplitudes.iter().map(|a| a.norm_sq()).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for a in &mut amplitudes {
                *a = *a * (1.0 / norm);
            }
        }

        StateVector { amplitudes, num_qubits: self.num_qubits }
    }

    /// Hybrid encoding: combines amplitude and phase
    ///
    /// First half of features → amplitudes
    /// Second half of features → phases
    fn hybrid_encoding(&self, data: &[f32]) -> StateVector {
        let dim = self.state_dim();
        let mid = data.len() / 2;

        // Amplitude part
        let amp_data = &data[..mid.min(dim)];
        let amp_norm: f32 = amp_data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let amp_norm = if amp_norm < 1e-10 { 1.0 } else { amp_norm };

        // Phase part
        let phase_data = &data[mid..];

        let mut amplitudes: Vec<Complex> = (0..dim)
            .map(|i| {
                // Amplitude from first half
                let amp = if i < amp_data.len() {
                    amp_data[i] / amp_norm
                } else {
                    0.0
                };

                // Phase from second half
                let mut phase = 0.0;
                for (j, &val) in phase_data.iter().enumerate().take(self.num_qubits) {
                    if (i >> j) & 1 == 1 {
                        phase += val * self.angle_scale;
                    }
                }

                Complex::from_polar(amp.abs(), if amp >= 0.0 { phase } else { phase + PI })
            })
            .collect();

        // Normalize
        let norm: f32 = amplitudes.iter().map(|a| a.norm_sq()).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for a in &mut amplitudes {
                *a = *a * (1.0 / norm);
            }
        }

        StateVector { amplitudes, num_qubits: self.num_qubits }
    }

    /// IQP (Instantaneous Quantum Polynomial) encoding
    ///
    /// Includes quadratic correlations: H^⊗n · Rz(x) · CZ · Rz(x²) · H^⊗n
    fn iqp_encoding(&self, data: &[f32]) -> StateVector {
        let dim = self.state_dim();

        // Start with Hadamard on all qubits (uniform superposition)
        let base_amp = 1.0 / (dim as f32).sqrt();

        // IQP: apply diagonal unitary with polynomial phases
        let amplitudes: Vec<Complex> = (0..dim)
            .map(|i| {
                let mut phase = 0.0;

                // Linear terms
                for (j, &val) in data.iter().enumerate().take(self.num_qubits) {
                    if (i >> j) & 1 == 1 {
                        phase += val;
                    }
                }

                // Quadratic terms (interactions)
                if self.use_correlations {
                    for j in 0..self.num_qubits.min(data.len()) {
                        for k in (j + 1)..self.num_qubits.min(data.len()) {
                            if ((i >> j) & 1 == 1) && ((i >> k) & 1 == 1) {
                                phase += data[j] * data[k];
                            }
                        }
                    }
                }

                Complex::from_polar(base_amp, phase)
            })
            .collect();

        StateVector { amplitudes, num_qubits: self.num_qubits }
    }

    /// Apply entanglement-like correlations via controlled rotations
    fn apply_correlations(&self, amplitudes: &[Complex], data: &[f32]) -> Vec<Complex> {
        let dim = amplitudes.len();
        let mut result = amplitudes.to_vec();

        // Apply CZ-like phases between adjacent qubits
        for i in 0..dim {
            for q in 0..(self.num_qubits - 1) {
                let bit_q = (i >> q) & 1;
                let bit_q1 = (i >> (q + 1)) & 1;

                if bit_q == 1 && bit_q1 == 1 && q < data.len() && q + 1 < data.len() {
                    // Apply correlation phase
                    let phase = data[q] * data[q + 1] * 0.5;
                    result[i] = result[i] * Complex::from_polar(1.0, phase);
                }
            }
        }

        result
    }

    /// Decode quantum-inspired state back to classical vector
    /// Uses amplitude extraction (real part of amplitudes)
    pub fn decode(&self, state: &StateVector) -> Vec<f32> {
        match self.encoding_type {
            EncodingType::Amplitude => {
                // Extract real parts of amplitudes
                state.amplitudes.iter().map(|a| a.re).collect()
            }
            EncodingType::Phase => {
                // Extract phases
                state.amplitudes.iter().map(|a| a.arg()).collect()
            }
            _ => {
                // For other encodings, use probabilities
                state.probabilities()
            }
        }
    }
}

/// Encoder for time series data
#[derive(Debug, Clone)]
pub struct TimeSeriesEncoder {
    /// Base encoder
    encoder: StateEncoder,
    /// Window size
    window_size: usize,
    /// Use differential encoding
    use_differential: bool,
}

impl TimeSeriesEncoder {
    /// Create new time series encoder
    pub fn new(encoding_type: EncodingType, window_size: usize) -> Self {
        // Calculate qubits needed for window
        let num_qubits = ((window_size as f32).log2().ceil() as usize).max(1);

        Self {
            encoder: StateEncoder::new(encoding_type, num_qubits),
            window_size,
            use_differential: false,
        }
    }

    /// Enable differential encoding (encode changes rather than values)
    pub fn with_differential(mut self, enable: bool) -> Self {
        self.use_differential = enable;
        self
    }

    /// Encode a time series window
    pub fn encode_window(&self, window: &[f32]) -> StateVector {
        let data = if self.use_differential && window.len() > 1 {
            // Compute differences
            window.windows(2)
                .map(|w| w[1] - w[0])
                .collect::<Vec<_>>()
        } else {
            window.to_vec()
        };

        self.encoder.encode(&data)
    }

    /// Encode multiple windows (batch)
    pub fn encode_batch(&self, windows: &[Vec<f32>]) -> Vec<StateVector> {
        windows.iter()
            .map(|w| self.encode_window(w))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amplitude_encoding() {
        let encoder = StateEncoder::amplitude(2);
        let data = vec![1.0, 0.0, 0.0, 0.0];
        let state = encoder.encode(&data);

        // Should be |00⟩ state
        assert!((state.amplitudes[0].abs() - 1.0).abs() < 1e-6);
        assert!(state.is_normalized(1e-6));
    }

    #[test]
    fn test_uniform_amplitude() {
        let encoder = StateEncoder::amplitude(2);
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let state = encoder.encode(&data);

        // Should be uniform superposition
        for p in state.probabilities() {
            assert!((p - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_angle_encoding() {
        let encoder = StateEncoder::angle(2);
        let data = vec![0.0, 0.0];
        let state = encoder.encode(&data);

        // Zero rotations should give |00⟩
        assert!((state.probability(0) - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_phase_encoding() {
        let encoder = StateEncoder::phase(2);
        let data = vec![0.0, 0.0];
        let state = encoder.encode(&data);

        // Should be uniform magnitude
        let probs = state.probabilities();
        for p in probs {
            assert!((p - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_normalization() {
        let encoder = StateEncoder::amplitude(3);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let state = encoder.encode(&data);

        assert!(state.is_normalized(1e-5));
    }

    #[test]
    fn test_time_series_encoder() {
        let encoder = TimeSeriesEncoder::new(EncodingType::Amplitude, 4);
        let window = vec![1.0, 2.0, 3.0, 4.0];
        let state = encoder.encode_window(&window);

        assert!(state.is_normalized(1e-5));
    }

    #[test]
    fn test_differential_encoding() {
        let encoder = TimeSeriesEncoder::new(EncodingType::Amplitude, 4)
            .with_differential(true);
        let window = vec![1.0, 2.0, 4.0, 7.0];
        let state = encoder.encode_window(&window);

        // Differences: [1, 2, 3]
        assert!(state.is_normalized(1e-5));
    }
}
