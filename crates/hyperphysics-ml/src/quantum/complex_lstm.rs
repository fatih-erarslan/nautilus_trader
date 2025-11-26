//! Complex-Valued LSTM Implementation
//!
//! This module implements an LSTM that operates on complex-valued data,
//! providing a richer representational space similar to quantum state spaces.
//!
//! # Mathematical Foundation
//!
//! Complex-valued networks can represent phase information and interference
//! effects that are useful for:
//! - Signal processing and time series
//! - Audio/speech processing
//! - Radar and communications
//! - Quantum-inspired computing
//!
//! # References
//!
//! - Trabelsi et al. (2018) "Deep Complex Networks"
//! - Arjovsky et al. (2016) "Unitary Evolution Recurrent Neural Networks"

use crate::error::MlResult;
use crate::tensor::{Tensor, TensorOps};
use crate::backends::Device;
use super::types::{Complex, QuantumHiddenState, QuantumLSTMOutput, CoherenceMetrics};
use super::config::QuantumLSTMConfig;
use super::encoding::{StateEncoder, TimeSeriesEncoder};
use std::f32::consts::PI;

/// Complex-Valued LSTM Cell
///
/// Processes complex-valued inputs and maintains complex hidden states.
/// Uses split-real representation for efficiency while preserving complex semantics.
#[derive(Debug)]
pub struct ComplexLSTM {
    /// Configuration
    config: QuantumLSTMConfig,
    /// Input weights (real part) [4*hidden, input]
    w_ih_re: Vec<f32>,
    /// Input weights (imaginary part)
    w_ih_im: Vec<f32>,
    /// Hidden weights (real part) [4*hidden, hidden]
    w_hh_re: Vec<f32>,
    /// Hidden weights (imaginary part)
    w_hh_im: Vec<f32>,
    /// Input bias (real)
    b_ih_re: Vec<f32>,
    /// Input bias (imaginary)
    b_ih_im: Vec<f32>,
    /// Hidden bias (real)
    b_hh_re: Vec<f32>,
    /// Hidden bias (imaginary)
    b_hh_im: Vec<f32>,
    /// State encoder for quantum-inspired encoding
    encoder: StateEncoder,
    /// Time step counter for coherence tracking
    time_step: usize,
    /// Coherence decay factor
    coherence_decay: f32,
}

impl ComplexLSTM {
    /// Create new Complex LSTM
    pub fn new(config: QuantumLSTMConfig, device: &Device) -> MlResult<Self> {
        config.validate().map_err(crate::error::MlError::ConfigError)?;

        let hidden_size = config.hidden_size;
        let input_size = config.input_size;
        let gate_size = 4 * hidden_size;

        // Xavier initialization scaled for complex values
        let scale_ih = (1.0 / (input_size + hidden_size) as f32).sqrt();
        let scale_hh = (1.0 / (hidden_size * 2) as f32).sqrt();

        // Initialize weights using LCG
        let w_ih_re = init_weights(gate_size * input_size, scale_ih, 42);
        let w_ih_im = init_weights(gate_size * input_size, scale_ih, 123);
        let w_hh_re = init_weights(gate_size * hidden_size, scale_hh, 456);
        let w_hh_im = init_weights(gate_size * hidden_size, scale_hh, 789);

        // Initialize biases
        let mut b_ih_re = vec![0.0; gate_size];
        let b_ih_im = vec![0.0; gate_size];
        let b_hh_re = vec![0.0; gate_size];
        let b_hh_im = vec![0.0; gate_size];

        // Forget gate bias = 1.0 for better gradient flow
        for i in hidden_size..(2 * hidden_size) {
            b_ih_re[i] = 1.0;
        }

        // Create encoder
        let num_qubits = ((input_size as f32).log2().ceil() as usize).max(2);
        let encoder = StateEncoder::new(config.encoding_type, num_qubits);

        Ok(Self {
            config,
            w_ih_re,
            w_ih_im,
            w_hh_re,
            w_hh_im,
            b_ih_re,
            b_ih_im,
            b_hh_re,
            b_hh_im,
            encoder,
            time_step: 0,
            coherence_decay: 0.0,
        })
    }

    /// Forward pass through one time step with complex values
    pub fn forward_step(
        &mut self,
        input_re: &[f32],
        input_im: &[f32],
        state: &mut QuantumHiddenState,
    ) -> MlResult<(Vec<f32>, Vec<f32>)> {
        let hidden_size = self.config.hidden_size;
        let input_size = self.config.input_size;

        // Compute gates using complex multiplication
        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        let mut gates_re = vec![0.0; 4 * hidden_size];
        let mut gates_im = vec![0.0; 4 * hidden_size];

        // W_ih @ input (complex multiplication)
        for i in 0..(4 * hidden_size) {
            let mut sum_re = self.b_ih_re[i] + self.b_hh_re[i];
            let mut sum_im = self.b_ih_im[i] + self.b_hh_im[i];

            // Input contribution
            for j in 0..input_size.min(input_re.len()) {
                let w_re = self.w_ih_re[i * input_size + j];
                let w_im = self.w_ih_im[i * input_size + j];
                let x_re = input_re[j];
                let x_im = if j < input_im.len() { input_im[j] } else { 0.0 };

                sum_re += w_re * x_re - w_im * x_im;
                sum_im += w_re * x_im + w_im * x_re;
            }

            // Hidden state contribution
            for j in 0..hidden_size {
                let w_re = self.w_hh_re[i * hidden_size + j];
                let w_im = self.w_hh_im[i * hidden_size + j];
                let h_re = state.h[j].re;
                let h_im = state.h[j].im;

                sum_re += w_re * h_re - w_im * h_im;
                sum_im += w_re * h_im + w_im * h_re;
            }

            gates_re[i] = sum_re;
            gates_im[i] = sum_im;
        }

        // Split gates: i, f, g, o
        let mut new_h_re = vec![0.0; hidden_size];
        let mut new_h_im = vec![0.0; hidden_size];

        for i in 0..hidden_size {
            // Input gate (use magnitude for gating)
            let i_mag = complex_sigmoid_magnitude(gates_re[i], gates_im[i]);

            // Forget gate
            let f_mag = complex_sigmoid_magnitude(
                gates_re[hidden_size + i],
                gates_im[hidden_size + i]
            );

            // Cell gate (complex tanh preserves phase)
            let (g_re, g_im) = complex_tanh(
                gates_re[2 * hidden_size + i],
                gates_im[2 * hidden_size + i]
            );

            // Output gate
            let o_mag = complex_sigmoid_magnitude(
                gates_re[3 * hidden_size + i],
                gates_im[3 * hidden_size + i]
            );

            // Update cell state: c = f * c + i * g
            let c_re = f_mag * state.c[i].re + i_mag * g_re;
            let c_im = f_mag * state.c[i].im + i_mag * g_im;
            state.c[i] = Complex::new(c_re, c_im);

            // Apply decoherence if enabled
            if self.config.use_biological_effects && self.config.decoherence_rate > 0.0 {
                let decay = 1.0 - self.config.decoherence_rate;
                // Decoherence primarily affects phase coherence
                let mag = state.c[i].abs();
                let phase = state.c[i].arg() * decay;
                state.c[i] = Complex::from_polar(mag, phase);
                self.coherence_decay += self.config.decoherence_rate;
            }

            // Update hidden state: h = o * tanh(c)
            let (c_tanh_re, c_tanh_im) = complex_tanh(state.c[i].re, state.c[i].im);
            new_h_re[i] = o_mag * c_tanh_re;
            new_h_im[i] = o_mag * c_tanh_im;

            state.h[i] = Complex::new(new_h_re[i], new_h_im[i]);
            state.phases[i] = state.h[i].arg();
        }

        self.time_step += 1;

        Ok((new_h_re, new_h_im))
    }

    /// Forward pass for real-valued input (converts to complex)
    pub fn forward_real(
        &mut self,
        input: &Tensor,
        initial_state: Option<QuantumHiddenState>,
    ) -> MlResult<QuantumLSTMOutput> {
        let input_data = input.as_slice().ok_or_else(||
            crate::error::MlError::ComputeError("Cannot access input".to_string()))?;

        let shape = input.shape();
        let dims = shape.dims();

        let (batch_size, seq_len, input_size) = match dims.len() {
            3 => (dims[0], dims[1], dims[2]),
            2 => (1, dims[0], dims[1]),
            _ => return Err(crate::error::MlError::DimensionMismatch {
                expected: "2D or 3D".to_string(),
                got: format!("{}D", dims.len()),
            }),
        };

        let mut state = initial_state.unwrap_or_else(||
            QuantumHiddenState::new(self.config.hidden_size));

        let mut outputs = Vec::with_capacity(batch_size * seq_len * self.config.hidden_size);
        let mut total_coherence = 0.0_f32;
        let mut total_fidelity = 0.0_f32;

        for b in 0..batch_size {
            state.reset();
            self.coherence_decay = 0.0;

            for t in 0..seq_len {
                let start = (b * seq_len + t) * input_size;
                let end = start + input_size;
                let x_re = &input_data[start..end];

                // Optionally encode input using quantum-inspired encoding
                let (input_re, input_im) = if self.config.use_complex {
                    // Use encoder to create complex representation
                    let encoded = self.encoder.encode(x_re);
                    let re: Vec<f32> = encoded.amplitudes.iter().map(|a| a.re).collect();
                    let im: Vec<f32> = encoded.amplitudes.iter().map(|a| a.im).collect();
                    (re, im)
                } else {
                    (x_re.to_vec(), vec![0.0; x_re.len()])
                };

                let (h_re, _h_im) = self.forward_step(&input_re, &input_im, &mut state)?;

                // Output magnitude only (real-valued output)
                outputs.extend_from_slice(&h_re);
            }

            // Compute coherence metrics for this batch
            let coherence = compute_phase_coherence(&state.phases);
            let fidelity = 1.0 - (self.coherence_decay / seq_len as f32);
            total_coherence += coherence;
            total_fidelity += fidelity;
        }

        let output_tensor = Tensor::from_slice(
            &outputs,
            vec![batch_size, seq_len, self.config.hidden_size],
            input.device(),
        )?;

        let metrics = CoherenceMetrics {
            phase_coherence: total_coherence / batch_size as f32,
            state_fidelity: total_fidelity / batch_size as f32,
            correlation_entropy: compute_entropy(&state.phases),
            tunneling_events: 0, // Not applicable for complex LSTM
        };

        Ok(QuantumLSTMOutput {
            output: output_tensor,
            hidden_state: state,
            attention_weights: None,
            coherence_metrics: Some(metrics),
        })
    }

    /// Forward with explicit complex input
    pub fn forward_complex(
        &mut self,
        input_re: &Tensor,
        input_im: &Tensor,
        initial_state: Option<QuantumHiddenState>,
    ) -> MlResult<(Tensor, Tensor, QuantumHiddenState)> {
        let re_data = input_re.as_slice().ok_or_else(||
            crate::error::MlError::ComputeError("Cannot access real input".to_string()))?;
        let im_data = input_im.as_slice().ok_or_else(||
            crate::error::MlError::ComputeError("Cannot access imag input".to_string()))?;

        let shape = input_re.shape();
        let dims = shape.dims();

        let (batch_size, seq_len, input_size) = match dims.len() {
            3 => (dims[0], dims[1], dims[2]),
            2 => (1, dims[0], dims[1]),
            _ => return Err(crate::error::MlError::DimensionMismatch {
                expected: "2D or 3D".to_string(),
                got: format!("{}D", dims.len()),
            }),
        };

        let mut state = initial_state.unwrap_or_else(||
            QuantumHiddenState::new(self.config.hidden_size));

        let mut outputs_re = Vec::with_capacity(batch_size * seq_len * self.config.hidden_size);
        let mut outputs_im = Vec::with_capacity(batch_size * seq_len * self.config.hidden_size);

        for b in 0..batch_size {
            state.reset();

            for t in 0..seq_len {
                let start = (b * seq_len + t) * input_size;
                let end = start + input_size;
                let x_re = &re_data[start..end];
                let x_im = &im_data[start..end];

                let (h_re, h_im) = self.forward_step(x_re, x_im, &mut state)?;

                outputs_re.extend_from_slice(&h_re);
                outputs_im.extend_from_slice(&h_im);
            }
        }

        let out_shape = vec![batch_size, seq_len, self.config.hidden_size];
        let output_re = Tensor::from_slice(&outputs_re, out_shape.clone(), input_re.device())?;
        let output_im = Tensor::from_slice(&outputs_im, out_shape, input_re.device())?;

        Ok((output_re, output_im, state))
    }

    /// Reset internal state
    pub fn reset(&mut self) {
        self.time_step = 0;
        self.coherence_decay = 0.0;
    }

    /// Get configuration
    pub fn config(&self) -> &QuantumLSTMConfig {
        &self.config
    }
}

// Complex activation functions

/// Sigmoid based on complex magnitude
/// σ(z) = 1 / (1 + exp(-|z|))
#[inline]
fn complex_sigmoid_magnitude(re: f32, im: f32) -> f32 {
    let mag = (re * re + im * im).sqrt();
    1.0 / (1.0 + (-mag).exp())
}

/// Complex tanh (applies tanh to real and imag separately, normalized)
/// Preserves phase while bounding magnitude
#[inline]
fn complex_tanh(re: f32, im: f32) -> (f32, f32) {
    let mag = (re * re + im * im).sqrt();
    if mag < 1e-10 {
        return (0.0, 0.0);
    }

    // Apply tanh to magnitude, preserve direction
    let tanh_mag = mag.tanh();
    let scale = tanh_mag / mag;

    (re * scale, im * scale)
}

/// Compute phase coherence across hidden units
fn compute_phase_coherence(phases: &[f32]) -> f32 {
    if phases.is_empty() {
        return 1.0;
    }

    // Coherence = |mean(e^(iφ))|
    let mut sum_re = 0.0_f32;
    let mut sum_im = 0.0_f32;

    for &phase in phases {
        sum_re += phase.cos();
        sum_im += phase.sin();
    }

    let n = phases.len() as f32;
    ((sum_re / n).powi(2) + (sum_im / n).powi(2)).sqrt()
}

/// Compute entropy of phase distribution
fn compute_entropy(phases: &[f32]) -> f32 {
    if phases.is_empty() {
        return 0.0;
    }

    // Discretize phases into bins
    const N_BINS: usize = 16;
    let mut bins = vec![0.0_f32; N_BINS];

    for &phase in phases {
        let normalized = (phase + PI) / (2.0 * PI);
        let bin = ((normalized * N_BINS as f32) as usize).min(N_BINS - 1);
        bins[bin] += 1.0;
    }

    // Normalize
    let total = phases.len() as f32;
    for b in &mut bins {
        *b /= total;
    }

    // Compute entropy
    let mut entropy = 0.0_f32;
    for &p in &bins {
        if p > 1e-10 {
            entropy -= p * p.ln();
        }
    }

    entropy / (N_BINS as f32).ln() // Normalize to [0, 1]
}

/// Initialize weights with LCG
fn init_weights(size: usize, scale: f32, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut data = Vec::with_capacity(size);

    for _ in 0..size {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((s >> 33) as f32) / (1u64 << 31) as f32 - 0.5;
        data.push(u * scale * 2.0);
    }

    data
}

/// Type alias for ComplexLSTM state - uses the same QuantumHiddenState
pub type ComplexLSTMState = QuantumHiddenState;

impl ComplexLSTM {
    /// Create new Complex LSTM with simple parameters
    ///
    /// This is a convenience constructor for integration with HFT ecosystem.
    pub fn with_dims(input_size: usize, hidden_size: usize) -> Self {
        let config = QuantumLSTMConfig::new(input_size, hidden_size)
            .with_complex(true);

        let hidden_size = config.hidden_size;
        let input_size = config.input_size;
        let gate_size = 4 * hidden_size;

        let scale_ih = (1.0 / (input_size + hidden_size) as f32).sqrt();
        let scale_hh = (1.0 / (hidden_size * 2) as f32).sqrt();

        let w_ih_re = init_weights(gate_size * input_size, scale_ih, 42);
        let w_ih_im = init_weights(gate_size * input_size, scale_ih, 123);
        let w_hh_re = init_weights(gate_size * hidden_size, scale_hh, 456);
        let w_hh_im = init_weights(gate_size * hidden_size, scale_hh, 789);

        let mut b_ih_re = vec![0.0; gate_size];
        let b_ih_im = vec![0.0; gate_size];
        let b_hh_re = vec![0.0; gate_size];
        let b_hh_im = vec![0.0; gate_size];

        for i in hidden_size..(2 * hidden_size) {
            b_ih_re[i] = 1.0;
        }

        let num_qubits = ((input_size as f32).log2().ceil() as usize).max(2);
        let encoder = StateEncoder::new(config.encoding_type, num_qubits);

        Self {
            config,
            w_ih_re,
            w_ih_im,
            w_hh_re,
            w_hh_im,
            b_ih_re,
            b_ih_im,
            b_hh_re,
            b_hh_im,
            encoder,
            time_step: 0,
            coherence_decay: 0.0,
        }
    }

    /// Simple forward pass for HFT integration
    ///
    /// Takes raw f32 slice and returns (output, new_state)
    pub fn forward(&mut self, input: &[f32], state: &ComplexLSTMState) -> (Vec<f32>, ComplexLSTMState) {
        let hidden_size = self.config.hidden_size;
        let input_size = self.config.input_size;

        // Clone state for modification
        let mut new_state = state.clone();

        // Pad/truncate input to expected size
        let padded_input: Vec<f32> = if input.len() >= input_size {
            input[input.len() - input_size..].to_vec()
        } else {
            let mut padded = vec![0.0f32; input_size - input.len()];
            padded.extend_from_slice(input);
            padded
        };

        // No imaginary part for simple interface
        let input_im = vec![0.0f32; input_size];

        // Run forward step
        let result = self.forward_step(&padded_input, &input_im, &mut new_state);

        match result {
            Ok((output_re, _output_im)) => (output_re, new_state),
            Err(_) => (vec![0.0; hidden_size], new_state),
        }
    }
}

impl ComplexLSTMState {
    /// Compute phase coherence from the state
    pub fn phase_coherence(&self) -> f32 {
        compute_phase_coherence(&self.phases)
    }
}

/// Quantum-Inspired LSTM wrapper
/// Combines encoding, complex processing, and biological effects
#[derive(Debug)]
pub struct QuantumInspiredLSTM {
    /// Complex LSTM layers
    layers: Vec<ComplexLSTM>,
    /// Configuration
    config: QuantumLSTMConfig,
    /// Time series encoder
    ts_encoder: TimeSeriesEncoder,
}

impl QuantumInspiredLSTM {
    /// Create new Quantum-Inspired LSTM
    pub fn new(config: QuantumLSTMConfig, device: &Device) -> MlResult<Self> {
        config.validate().map_err(crate::error::MlError::ConfigError)?;

        let mut layers = Vec::with_capacity(config.num_layers);

        // First layer takes input_size, subsequent layers take hidden_size
        for i in 0..config.num_layers {
            let layer_config = if i == 0 {
                config.clone()
            } else {
                let mut c = config.clone();
                c.input_size = config.hidden_size;
                c
            };
            layers.push(ComplexLSTM::new(layer_config, device)?);
        }

        let ts_encoder = TimeSeriesEncoder::new(config.encoding_type, config.input_size);

        Ok(Self {
            layers,
            config,
            ts_encoder,
        })
    }

    /// Forward pass through all layers
    pub fn forward(
        &mut self,
        input: &Tensor,
        initial_states: Option<Vec<QuantumHiddenState>>,
    ) -> MlResult<QuantumLSTMOutput> {
        let mut x = input.clone();
        let mut final_state = QuantumHiddenState::new(self.config.hidden_size);
        let mut cumulative_metrics = CoherenceMetrics::default();

        let states = initial_states.unwrap_or_else(||
            vec![QuantumHiddenState::new(self.config.hidden_size); self.layers.len()]);

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let state = states.get(i).cloned();
            let output = layer.forward_real(&x, state)?;

            x = output.output;
            final_state = output.hidden_state;

            if let Some(metrics) = output.coherence_metrics {
                cumulative_metrics.phase_coherence += metrics.phase_coherence;
                cumulative_metrics.state_fidelity += metrics.state_fidelity;
                cumulative_metrics.correlation_entropy += metrics.correlation_entropy;
            }
        }

        // Average metrics across layers
        let n_layers = self.layers.len() as f32;
        cumulative_metrics.phase_coherence /= n_layers;
        cumulative_metrics.state_fidelity /= n_layers;
        cumulative_metrics.correlation_entropy /= n_layers;

        Ok(QuantumLSTMOutput {
            output: x,
            hidden_state: final_state,
            attention_weights: None,
            coherence_metrics: Some(cumulative_metrics),
        })
    }

    /// Reset all layers
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    /// Get configuration
    pub fn config(&self) -> &QuantumLSTMConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_lstm_creation() {
        let config = QuantumLSTMConfig::new(10, 32);
        let device = Device::Cpu;
        let lstm = ComplexLSTM::new(config, &device);
        assert!(lstm.is_ok());
    }

    #[test]
    fn test_complex_forward() {
        let config = QuantumLSTMConfig::new(4, 8).with_complex(true);
        let device = Device::Cpu;
        let mut lstm = ComplexLSTM::new(config, &device).unwrap();

        let input_data: Vec<f32> = (0..20).map(|i| (i as f32) / 20.0).collect();
        let input = Tensor::from_slice(&input_data, vec![1, 5, 4], &device).unwrap();

        let output = lstm.forward_real(&input, None).unwrap();
        let out_shape = output.output.shape();
        assert_eq!(out_shape.dims(), &[1, 5, 8]);
    }

    #[test]
    fn test_complex_tanh() {
        let (re, im) = complex_tanh(1.0, 0.0);
        let expected = 1.0_f32.tanh();
        assert!((re - expected).abs() < 1e-6);
        assert!(im.abs() < 1e-6);
    }

    #[test]
    fn test_phase_coherence() {
        // All same phase = perfect coherence
        let phases = vec![0.5, 0.5, 0.5, 0.5];
        let coherence = compute_phase_coherence(&phases);
        assert!((coherence - 1.0).abs() < 1e-6);

        // Uniform phases = low coherence
        let phases: Vec<f32> = (0..8).map(|i| i as f32 * PI / 4.0).collect();
        let coherence = compute_phase_coherence(&phases);
        assert!(coherence < 0.5);
    }

    #[test]
    fn test_quantum_inspired_lstm() {
        let config = QuantumLSTMConfig::new(4, 8)
            .with_num_layers(2)
            .with_complex(true);
        let device = Device::Cpu;
        let mut lstm = QuantumInspiredLSTM::new(config, &device).unwrap();

        let input_data: Vec<f32> = (0..20).map(|i| (i as f32) / 20.0).collect();
        let input = Tensor::from_slice(&input_data, vec![1, 5, 4], &device).unwrap();

        let output = lstm.forward(&input, None).unwrap();
        let out_shape = output.output.shape();
        assert_eq!(out_shape.dims(), &[1, 5, 8]);

        // Should have coherence metrics
        assert!(output.coherence_metrics.is_some());
    }
}
