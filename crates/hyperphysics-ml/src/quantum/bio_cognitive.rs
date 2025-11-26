//! BioCognitive LSTM - LSTM with biologically-plausible neural dynamics
//!
//! This module implements an LSTM variant that incorporates biological neural
//! dynamics inspired by neuroscience research. Unlike the placeholder "quantum"
//! effects, these are REAL, implementable biophysical phenomena:
//!
//! # Biological Features
//!
//! 1. **Membrane Dynamics**: Leaky integrate-and-fire inspired gating
//! 2. **Neural Oscillations**: Gamma and theta rhythm modulation
//! 3. **Short-Term Plasticity**: Facilitation and depression
//! 4. **Spike-Timing**: Temporal coding through spike patterns
//! 5. **Adaptation**: Spike-frequency adaptation
//!
//! # References
//!
//! - Izhikevich (2003) "Simple Model of Spiking Neurons"
//! - Buzsáki & Draguhn (2004) "Neuronal Oscillations in Cortical Networks"
//! - Markram et al. (1998) "Differential signaling via the same axon"

use crate::error::MlResult;
use crate::tensor::{Tensor, TensorOps};
use crate::backends::Device;
use super::config::BioCognitiveConfig;
use std::f32::consts::PI;

/// BioCognitive LSTM Cell
///
/// Implements an LSTM cell with biologically-plausible dynamics:
/// - Membrane potential integration with leak
/// - Gamma and theta oscillation modulation
/// - Short-term synaptic plasticity
#[derive(Debug)]
pub struct BioCognitiveLSTM {
    /// Configuration
    config: BioCognitiveConfig,
    /// Input-to-hidden weights [4*hidden, input]
    w_ih: Tensor,
    /// Hidden-to-hidden weights [4*hidden, hidden]
    w_hh: Tensor,
    /// Input bias [4*hidden]
    b_ih: Tensor,
    /// Hidden bias [4*hidden]
    b_hh: Tensor,
    /// Internal time step counter
    time_step: usize,
    /// Refractory state per hidden unit
    refractory_counter: Vec<usize>,
    /// Adaptation variable per hidden unit
    adaptation: Vec<f32>,
    /// Facilitation variable (STP)
    facilitation: Vec<f32>,
    /// Depression variable (STP)
    depression: Vec<f32>,
}

/// Hidden state for BioCognitive LSTM
#[derive(Debug, Clone)]
pub struct BioCognitiveState {
    /// Hidden state
    pub h: Vec<f32>,
    /// Cell state
    pub c: Vec<f32>,
    /// Membrane potential (for dynamics)
    pub membrane: Vec<f32>,
    /// Spike history (recent spikes for timing)
    pub spike_history: Vec<Vec<usize>>,
}

impl BioCognitiveState {
    /// Create new state
    pub fn new(hidden_size: usize) -> Self {
        Self {
            h: vec![0.0; hidden_size],
            c: vec![0.0; hidden_size],
            membrane: vec![0.0; hidden_size],
            spike_history: vec![Vec::new(); hidden_size],
        }
    }

    /// Reset to initial state
    pub fn reset(&mut self) {
        for h in &mut self.h { *h = 0.0; }
        for c in &mut self.c { *c = 0.0; }
        for m in &mut self.membrane { *m = 0.0; }
        for hist in &mut self.spike_history { hist.clear(); }
    }
}

impl BioCognitiveLSTM {
    /// Create new BioCognitive LSTM
    pub fn new(config: BioCognitiveConfig, device: &Device) -> MlResult<Self> {
        config.validate().map_err(crate::error::MlError::ConfigError)?;

        let hidden_size = config.hidden_size;
        let input_size = config.input_size;

        // Xavier initialization for weights
        let scale_ih = (2.0 / (input_size + hidden_size) as f32).sqrt();
        let scale_hh = (2.0 / (hidden_size * 2) as f32).sqrt();

        // Initialize weights with proper scaling
        let w_ih = Self::init_weights(4 * hidden_size, input_size, scale_ih, device)?;
        let w_hh = Self::init_weights(4 * hidden_size, hidden_size, scale_hh, device)?;

        // Initialize biases (forget gate bias = 1.0 for gradient flow)
        let mut b_ih_data = vec![0.0; 4 * hidden_size];
        let mut b_hh_data = vec![0.0; 4 * hidden_size];

        // Set forget gate bias to 1.0 (indices hidden_size to 2*hidden_size)
        for i in hidden_size..(2 * hidden_size) {
            b_ih_data[i] = 1.0;
        }

        let b_ih = Tensor::from_slice(&b_ih_data, vec![4 * hidden_size], device)?;
        let b_hh = Tensor::from_slice(&b_hh_data, vec![4 * hidden_size], device)?;

        Ok(Self {
            config,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            time_step: 0,
            refractory_counter: vec![0; hidden_size],
            adaptation: vec![0.0; hidden_size],
            facilitation: vec![1.0; hidden_size],
            depression: vec![1.0; hidden_size],
        })
    }

    /// Create new BioCognitive LSTM (convenience constructor without device)
    ///
    /// Uses CPU device by default. For HFT integration.
    pub fn from_config(config: BioCognitiveConfig) -> Self {
        let device = Device::Cpu;
        // Try to create properly, fall back to minimal version if it fails
        Self::new(config.clone(), &device).unwrap_or_else(|_| {
            Self::minimal(config)
        })
    }

    /// Create minimal instance (for fallback)
    fn minimal(config: BioCognitiveConfig) -> Self {
        let hidden_size = config.hidden_size;
        let input_size = config.input_size;
        let device = Device::Cpu;

        let scale_ih = (2.0 / (input_size + hidden_size) as f32).sqrt();
        let scale_hh = (2.0 / (hidden_size * 2) as f32).sqrt();

        // Simple weight initialization
        let w_ih = Self::init_weights_simple(4 * hidden_size, input_size, scale_ih, &device);
        let w_hh = Self::init_weights_simple(4 * hidden_size, hidden_size, scale_hh, &device);

        let mut b_ih_data = vec![0.0; 4 * hidden_size];
        for i in hidden_size..(2 * hidden_size) {
            b_ih_data[i] = 1.0;
        }
        let b_ih = Tensor::from_slice(&b_ih_data, vec![4 * hidden_size], &device)
            .unwrap_or_else(|_| Tensor::zeros(vec![4 * hidden_size], crate::tensor::DType::F32, &device).unwrap());
        let b_hh = Tensor::zeros(vec![4 * hidden_size], crate::tensor::DType::F32, &device)
            .unwrap_or_else(|_| Tensor::zeros(vec![4 * hidden_size], crate::tensor::DType::F32, &device).unwrap());

        Self {
            config,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            time_step: 0,
            refractory_counter: vec![0; hidden_size],
            adaptation: vec![0.0; hidden_size],
            facilitation: vec![1.0; hidden_size],
            depression: vec![1.0; hidden_size],
        }
    }

    fn init_weights_simple(rows: usize, cols: usize, scale: f32, device: &Device) -> Tensor {
        let mut seed = 42u64;
        let mut data = Vec::with_capacity(rows * cols);

        for _ in 0..(rows * cols) {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((seed >> 33) as f32) / (1u64 << 31) as f32 - 0.5;
            data.push(u * scale * 2.0);
        }

        Tensor::from_slice(&data, vec![rows, cols], device)
            .unwrap_or_else(|_| Tensor::zeros(vec![rows, cols], crate::tensor::DType::F32, device).unwrap())
    }

    /// Simple forward pass for HFT integration
    ///
    /// Takes raw f32 slice and state reference, returns (output, new_state)
    pub fn forward_simple(&mut self, input: &[f32], state: &BioCognitiveState) -> (Vec<f32>, BioCognitiveState) {
        let mut new_state = state.clone();

        // Run single step
        match self.forward_step(input, &mut new_state) {
            Ok(output) => (output, new_state),
            Err(_) => (vec![0.0; self.config.hidden_size], new_state),
        }
    }

    fn init_weights(rows: usize, cols: usize, scale: f32, device: &Device) -> MlResult<Tensor> {
        // Simple LCG for reproducible initialization
        let mut seed = 42u64;
        let mut data = Vec::with_capacity(rows * cols);

        for _ in 0..(rows * cols) {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((seed >> 33) as f32) / (1u64 << 31) as f32 - 0.5;
            data.push(u * scale * 2.0);
        }

        Tensor::from_slice(&data, vec![rows, cols], device)
    }

    /// Forward pass through one time step
    pub fn forward_step(
        &mut self,
        input: &[f32],
        state: &mut BioCognitiveState,
    ) -> MlResult<Vec<f32>> {
        let hidden_size = self.config.hidden_size;

        // Get weight slices
        let w_ih_data = self.w_ih.as_slice().ok_or_else(||
            crate::error::MlError::ComputeError("Cannot access w_ih".to_string()))?;
        let w_hh_data = self.w_hh.as_slice().ok_or_else(||
            crate::error::MlError::ComputeError("Cannot access w_hh".to_string()))?;
        let b_ih_data = self.b_ih.as_slice().ok_or_else(||
            crate::error::MlError::ComputeError("Cannot access b_ih".to_string()))?;
        let b_hh_data = self.b_hh.as_slice().ok_or_else(||
            crate::error::MlError::ComputeError("Cannot access b_hh".to_string()))?;

        // Compute gates: [i, f, g, o] = W_ih @ x + W_hh @ h + b
        let mut gates = vec![0.0; 4 * hidden_size];

        // W_ih @ input
        for i in 0..(4 * hidden_size) {
            let mut sum = b_ih_data[i] + b_hh_data[i];
            for j in 0..input.len().min(self.config.input_size) {
                sum += w_ih_data[i * self.config.input_size + j] * input[j];
            }
            for j in 0..hidden_size {
                // Apply STP modulation to recurrent connections
                let stp_factor = self.facilitation[j] * self.depression[j];
                sum += w_hh_data[i * hidden_size + j] * state.h[j] * stp_factor;
            }
            gates[i] = sum;
        }

        // Apply oscillation modulation
        let osc_modulation = self.compute_oscillation_modulation();

        // Split gates and apply activations
        let mut i_gate = vec![0.0; hidden_size];
        let mut f_gate = vec![0.0; hidden_size];
        let mut g_gate = vec![0.0; hidden_size];
        let mut o_gate = vec![0.0; hidden_size];

        for i in 0..hidden_size {
            // Input gate with oscillation modulation
            i_gate[i] = sigmoid(gates[i] * osc_modulation[i % osc_modulation.len()]);

            // Forget gate
            f_gate[i] = sigmoid(gates[hidden_size + i]);

            // Cell gate (candidate) with adaptation
            let adapt_factor = 1.0 - self.adaptation[i] * self.config.adaptation_strength;
            g_gate[i] = fast_tanh(gates[2 * hidden_size + i]) * adapt_factor;

            // Output gate
            o_gate[i] = sigmoid(gates[3 * hidden_size + i]);
        }

        // Update cell state with membrane dynamics
        let tau = self.config.tau_membrane;
        let decay = (-1.0 / tau).exp();

        for i in 0..hidden_size {
            // Membrane potential integration (leaky)
            state.membrane[i] = decay * state.membrane[i] +
                (1.0 - decay) * (f_gate[i] * state.c[i] + i_gate[i] * g_gate[i]);

            // Check for "spike" (threshold crossing)
            if state.membrane[i] > self.config.spike_threshold &&
               self.refractory_counter[i] == 0 {
                // Record spike
                state.spike_history[i].push(self.time_step);
                if state.spike_history[i].len() > 100 {
                    state.spike_history[i].remove(0);
                }

                // Enter refractory period
                self.refractory_counter[i] = self.config.refractory_period;

                // Update adaptation
                self.adaptation[i] = (self.adaptation[i] + 0.1).min(1.0);
            }

            // Update refractory counter
            if self.refractory_counter[i] > 0 {
                self.refractory_counter[i] -= 1;
            }

            // Decay adaptation
            self.adaptation[i] *= 0.99;

            // Cell state update
            state.c[i] = state.membrane[i];
        }

        // Update hidden state
        for i in 0..hidden_size {
            state.h[i] = o_gate[i] * fast_tanh(state.c[i]);
        }

        // Update short-term plasticity
        self.update_stp(&state.h);

        self.time_step += 1;

        Ok(state.h.clone())
    }

    /// Compute oscillation modulation factors
    fn compute_oscillation_modulation(&self) -> Vec<f32> {
        let t = self.time_step as f32;
        let mut modulation = vec![1.0; self.config.hidden_size];

        if self.config.enable_gamma {
            let gamma = (2.0 * PI * self.config.gamma_frequency * t).cos();
            for (i, m) in modulation.iter_mut().enumerate() {
                // Different neurons phase-locked differently
                let phase_offset = (i as f32 / self.config.hidden_size as f32) * PI;
                let gamma_mod = (2.0 * PI * self.config.gamma_frequency * t + phase_offset).cos();
                *m *= 1.0 + 0.2 * gamma_mod; // 20% modulation depth
            }
        }

        if self.config.enable_theta {
            let theta = (2.0 * PI * self.config.theta_frequency * t).cos();
            for m in &mut modulation {
                *m *= 1.0 + 0.3 * theta; // 30% modulation depth
            }
        }

        modulation
    }

    /// Update short-term plasticity variables
    fn update_stp(&mut self, activity: &[f32]) {
        if !self.config.enable_stp {
            return;
        }

        let tau_f = self.config.tau_facilitation;
        let tau_d = self.config.tau_depression;

        for (i, &act) in activity.iter().enumerate() {
            let activity_level = act.abs().min(1.0);

            // Facilitation dynamics: increases with activity, decays to 1
            let decay_f = (-1.0 / tau_f).exp();
            self.facilitation[i] = decay_f * self.facilitation[i] +
                (1.0 - decay_f) * (1.0 + activity_level * 0.5);
            self.facilitation[i] = self.facilitation[i].clamp(1.0, 2.0);

            // Depression dynamics: decreases with activity, recovers to 1
            let decay_d = (-1.0 / tau_d).exp();
            self.depression[i] = decay_d * self.depression[i] +
                (1.0 - decay_d) * (1.0 - activity_level * 0.3);
            self.depression[i] = self.depression[i].clamp(0.2, 1.0);
        }
    }

    /// Forward pass through entire sequence
    pub fn forward(
        &mut self,
        input: &Tensor,
        initial_state: Option<BioCognitiveState>,
    ) -> MlResult<(Tensor, BioCognitiveState)> {
        let input_data = input.as_slice().ok_or_else(||
            crate::error::MlError::ComputeError("Cannot access input tensor".to_string()))?;

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
            BioCognitiveState::new(self.config.hidden_size));

        let mut outputs = Vec::with_capacity(batch_size * seq_len * self.config.hidden_size);

        // Process each batch item
        for b in 0..batch_size {
            state.reset();

            // Process sequence
            for t in 0..seq_len {
                let start = (b * seq_len + t) * input_size;
                let end = start + input_size;
                let x_t = &input_data[start..end];

                let h_t = self.forward_step(x_t, &mut state)?;
                outputs.extend_from_slice(&h_t);
            }
        }

        let output_tensor = Tensor::from_slice(
            &outputs,
            vec![batch_size, seq_len, self.config.hidden_size],
            input.device(),
        )?;

        Ok((output_tensor, state))
    }

    /// Reset internal state
    pub fn reset(&mut self) {
        self.time_step = 0;
        for r in &mut self.refractory_counter { *r = 0; }
        for a in &mut self.adaptation { *a = 0.0; }
        for f in &mut self.facilitation { *f = 1.0; }
        for d in &mut self.depression { *d = 1.0; }
    }

    /// Get configuration
    pub fn config(&self) -> &BioCognitiveConfig {
        &self.config
    }
}

// Fast activation functions

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn fast_tanh(x: f32) -> f32 {
    // Approximation: tanh(x) ≈ x / (1 + |x|) for small x
    // More accurate: use exp-based for larger values
    if x.abs() < 2.0 {
        x / (1.0 + x.abs())
    } else {
        x.tanh()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bio_cognitive_lstm_creation() {
        let config = BioCognitiveConfig::new(10, 32);
        let device = Device::Cpu;
        let lstm = BioCognitiveLSTM::new(config, &device);
        assert!(lstm.is_ok());
    }

    #[test]
    fn test_forward_step() {
        let config = BioCognitiveConfig::new(4, 8);
        let device = Device::Cpu;
        let mut lstm = BioCognitiveLSTM::new(config, &device).unwrap();

        let input = vec![1.0, 0.5, -0.5, 0.0];
        let mut state = BioCognitiveState::new(8);

        let output = lstm.forward_step(&input, &mut state);
        assert!(output.is_ok());
        assert_eq!(output.unwrap().len(), 8);
    }

    #[test]
    fn test_forward_sequence() {
        let config = BioCognitiveConfig::new(4, 8);
        let device = Device::Cpu;
        let mut lstm = BioCognitiveLSTM::new(config, &device).unwrap();

        // Create input tensor [batch=1, seq=5, features=4]
        let input_data: Vec<f32> = (0..20).map(|i| (i as f32) / 20.0).collect();
        let input = Tensor::from_slice(&input_data, vec![1, 5, 4], &device).unwrap();

        let (output, _state) = lstm.forward(&input, None).unwrap();
        let out_shape = output.shape();
        assert_eq!(out_shape.dims(), &[1, 5, 8]);
    }

    #[test]
    fn test_oscillation_modulation() {
        let config = BioCognitiveConfig::new(4, 8)
            .with_oscillations(true, true);
        let device = Device::Cpu;
        let lstm = BioCognitiveLSTM::new(config, &device).unwrap();

        let mod1 = lstm.compute_oscillation_modulation();
        assert_eq!(mod1.len(), 8);

        // All values should be positive (modulation around 1.0)
        for m in mod1 {
            assert!(m > 0.0);
        }
    }
}
