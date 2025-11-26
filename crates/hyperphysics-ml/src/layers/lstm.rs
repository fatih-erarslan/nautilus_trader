//! LSTM (Long Short-Term Memory) layer for time-series forecasting

use crate::backends::Device;
use crate::error::MlResult;
use crate::tensor::{DType, Tensor, TensorOps};
use super::{Layer, StatefulLayer};
use serde::{Deserialize, Serialize};

/// LSTM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LstmConfig {
    /// Input feature dimension
    pub input_size: usize,
    /// Hidden state dimension
    pub hidden_size: usize,
    /// Number of stacked LSTM layers
    pub num_layers: usize,
    /// Dropout probability between layers
    pub dropout: f32,
    /// Bidirectional LSTM
    pub bidirectional: bool,
    /// Use bias terms
    pub bias: bool,
    /// Batch first format (batch, seq, features)
    pub batch_first: bool,
}

impl LstmConfig {
    /// Create new LSTM config
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            num_layers: 1,
            dropout: 0.0,
            bidirectional: false,
            bias: true,
            batch_first: true,
        }
    }

    /// Set number of layers
    pub fn with_num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// Set dropout
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Set bidirectional
    pub fn with_bidirectional(mut self, bidirectional: bool) -> Self {
        self.bidirectional = bidirectional;
        self
    }

    /// Compute actual hidden size (doubled if bidirectional)
    pub fn effective_hidden_size(&self) -> usize {
        if self.bidirectional {
            self.hidden_size * 2
        } else {
            self.hidden_size
        }
    }
}

impl Default for LstmConfig {
    fn default() -> Self {
        Self::new(64, 128)
    }
}

/// LSTM hidden state: (h, c)
#[derive(Debug, Clone)]
pub struct LstmState {
    /// Hidden state h: [num_layers * directions, batch, hidden]
    pub h: Tensor,
    /// Cell state c: [num_layers * directions, batch, hidden]
    pub c: Tensor,
}

/// LSTM output containing output tensor and final state
#[derive(Debug, Clone)]
pub struct LstmOutput {
    /// Output sequence [batch, seq_len, hidden * directions] if batch_first
    pub output: Tensor,
    /// Final hidden state
    pub state: LstmState,
}

/// LSTM layer for sequential data
///
/// Implements the standard LSTM equations:
/// - i_t = σ(W_ii x_t + b_ii + W_hi h_{t-1} + b_hi)
/// - f_t = σ(W_if x_t + b_if + W_hf h_{t-1} + b_hf)
/// - g_t = tanh(W_ig x_t + b_ig + W_hg h_{t-1} + b_hg)
/// - o_t = σ(W_io x_t + b_io + W_ho h_{t-1} + b_ho)
/// - c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
/// - h_t = o_t ⊙ tanh(c_t)
#[derive(Debug, Clone)]
pub struct Lstm {
    config: LstmConfig,
    /// Weight matrices for each layer
    /// Each layer has: W_ih [4*hidden, input], W_hh [4*hidden, hidden]
    weight_ih: Vec<Tensor>,
    weight_hh: Vec<Tensor>,
    /// Bias vectors (if enabled)
    bias_ih: Vec<Option<Tensor>>,
    bias_hh: Vec<Option<Tensor>>,
    /// Device
    device: Device,
}

impl Lstm {
    /// Create a new LSTM layer
    pub fn new(config: LstmConfig, device: &Device) -> MlResult<Self> {
        let num_directions = if config.bidirectional { 2 } else { 1 };
        let mut weight_ih = Vec::with_capacity(config.num_layers * num_directions);
        let mut weight_hh = Vec::with_capacity(config.num_layers * num_directions);
        let mut bias_ih = Vec::with_capacity(config.num_layers * num_directions);
        let mut bias_hh = Vec::with_capacity(config.num_layers * num_directions);

        for layer in 0..config.num_layers {
            for _direction in 0..num_directions {
                let input_dim = if layer == 0 {
                    config.input_size
                } else {
                    config.hidden_size * num_directions
                };

                // Xavier initialization for gates
                let scale_ih = (2.0 / (input_dim + 4 * config.hidden_size) as f32).sqrt();
                let scale_hh = (2.0 / (config.hidden_size + 4 * config.hidden_size) as f32).sqrt();

                // W_ih: [4*hidden, input] - combined weights for i, f, g, o gates
                let w_ih = Tensor::randn(
                    vec![4 * config.hidden_size, input_dim],
                    device,
                )?;
                weight_ih.push(scale_weight(w_ih, scale_ih, device)?);

                // W_hh: [4*hidden, hidden]
                let w_hh = Tensor::randn(
                    vec![4 * config.hidden_size, config.hidden_size],
                    device,
                )?;
                weight_hh.push(scale_weight(w_hh, scale_hh, device)?);

                // Bias vectors
                if config.bias {
                    bias_ih.push(Some(Tensor::zeros(
                        vec![4 * config.hidden_size],
                        DType::F32,
                        device,
                    )?));
                    // Initialize forget gate bias to 1.0 for better gradient flow
                    let mut b_hh = Tensor::zeros(
                        vec![4 * config.hidden_size],
                        DType::F32,
                        device,
                    )?;
                    init_forget_gate_bias(&mut b_hh, config.hidden_size)?;
                    bias_hh.push(Some(b_hh));
                } else {
                    bias_ih.push(None);
                    bias_hh.push(None);
                }
            }
        }

        Ok(Self {
            config,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            device: device.clone(),
        })
    }

    /// Initialize hidden state to zeros
    pub fn init_state(&self, batch_size: usize) -> MlResult<LstmState> {
        let num_directions = if self.config.bidirectional { 2 } else { 1 };
        let num_layers = self.config.num_layers * num_directions;

        let h = Tensor::zeros(
            vec![num_layers, batch_size, self.config.hidden_size],
            DType::F32,
            &self.device,
        )?;
        let c = Tensor::zeros(
            vec![num_layers, batch_size, self.config.hidden_size],
            DType::F32,
            &self.device,
        )?;

        Ok(LstmState { h, c })
    }

    /// Get configuration
    pub fn config(&self) -> &LstmConfig {
        &self.config
    }

    /// Forward pass with optional initial state
    pub fn forward_with_state(
        &self,
        input: &Tensor,
        state: Option<LstmState>,
    ) -> MlResult<LstmOutput> {
        let input_shape = input.shape().dims();

        // Determine batch size and sequence length
        let (batch_size, seq_len, _input_size) = if self.config.batch_first {
            (input_shape[0], input_shape[1], input_shape[2])
        } else {
            (input_shape[1], input_shape[0], input_shape[2])
        };

        // Initialize state if not provided
        let state = match state {
            Some(s) => s,
            None => self.init_state(batch_size)?,
        };

        // For now, implement single-layer forward pass
        // Full implementation would iterate through layers and time steps
        let output = lstm_forward_single_layer(
            input,
            &state.h,
            &state.c,
            &self.weight_ih[0],
            &self.weight_hh[0],
            self.bias_ih[0].as_ref(),
            self.bias_hh[0].as_ref(),
            self.config.batch_first,
        )?;

        Ok(output)
    }
}

impl Layer for Lstm {
    fn forward(&self, input: &Tensor) -> MlResult<Tensor> {
        let output = self.forward_with_state(input, None)?;
        Ok(output.output)
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn num_parameters(&self) -> usize {
        let num_directions = if self.config.bidirectional { 2 } else { 1 };
        let mut count = 0;

        for layer in 0..self.config.num_layers {
            let input_dim = if layer == 0 {
                self.config.input_size
            } else {
                self.config.hidden_size * num_directions
            };

            for _direction in 0..num_directions {
                // W_ih: 4*hidden * input
                count += 4 * self.config.hidden_size * input_dim;
                // W_hh: 4*hidden * hidden
                count += 4 * self.config.hidden_size * self.config.hidden_size;
                // Biases
                if self.config.bias {
                    count += 8 * self.config.hidden_size;
                }
            }
        }

        count
    }

    fn to_device(&mut self, device: &Device) -> MlResult<()> {
        for w in &mut self.weight_ih {
            *w = w.to_device(device)?;
        }
        for w in &mut self.weight_hh {
            *w = w.to_device(device)?;
        }
        for b in &mut self.bias_ih {
            if let Some(bias) = b {
                *b = Some(bias.to_device(device)?);
            }
        }
        for b in &mut self.bias_hh {
            if let Some(bias) = b {
                *b = Some(bias.to_device(device)?);
            }
        }
        self.device = device.clone();
        Ok(())
    }
}

impl StatefulLayer for Lstm {
    type State = LstmState;

    fn forward_with_state(
        &self,
        input: &Tensor,
        state: Option<Self::State>,
    ) -> MlResult<(Tensor, Self::State)> {
        let output = Lstm::forward_with_state(self, input, state)?;
        Ok((output.output, output.state))
    }

    fn reset_state(&mut self) {
        // State is managed externally, nothing to reset
    }
}

// Helper functions

fn scale_weight(tensor: Tensor, scale: f32, device: &Device) -> MlResult<Tensor> {
    #[cfg(feature = "cpu")]
    {
        if let Some(data) = tensor.as_slice() {
            let scaled: Vec<f32> = data.iter().map(|&x| x * scale).collect();
            return Tensor::from_slice(&scaled, tensor.shape().dims().to_vec(), device);
        }
    }
    Ok(tensor)
}

fn init_forget_gate_bias(bias: &mut Tensor, hidden_size: usize) -> MlResult<()> {
    // Forget gate is at indices [hidden_size, 2*hidden_size)
    #[cfg(feature = "cpu")]
    {
        if let Some(data) = bias.as_slice_mut() {
            for i in hidden_size..(2 * hidden_size) {
                data[i] = 1.0;
            }
        }
    }
    Ok(())
}

/// Single-layer LSTM forward pass
fn lstm_forward_single_layer(
    input: &Tensor,
    h0: &Tensor,
    c0: &Tensor,
    weight_ih: &Tensor,
    weight_hh: &Tensor,
    bias_ih: Option<&Tensor>,
    bias_hh: Option<&Tensor>,
    batch_first: bool,
) -> MlResult<LstmOutput> {
    let input_shape = input.shape().dims();
    let (batch_size, seq_len, input_size) = if batch_first {
        (input_shape[0], input_shape[1], input_shape[2])
    } else {
        (input_shape[1], input_shape[0], input_shape[2])
    };

    let hidden_size = weight_hh.shape().dims()[1];
    let device = input.device();

    // Initialize output tensor
    let mut outputs = Vec::with_capacity(seq_len);

    // Get initial hidden state for batch
    let mut h = h0.slice(0, 0, 1)?.squeeze_dim(0)?;  // [batch, hidden]
    let mut c = c0.slice(0, 0, 1)?.squeeze_dim(0)?;  // [batch, hidden]

    // Process each time step
    for t in 0..seq_len {
        // Get input at time t: [batch, input_size]
        let x_t = if batch_first {
            input.slice(1, t, t + 1)?.squeeze_dim(1)?
        } else {
            input.slice(0, t, t + 1)?.squeeze_dim(0)?
        };

        // Compute gates: gates = x @ W_ih^T + h @ W_hh^T + b_ih + b_hh
        let gates_ih = x_t.matmul(&weight_ih.transpose()?)?;
        let gates_hh = h.matmul(&weight_hh.transpose()?)?;
        let mut gates = gates_ih.add(&gates_hh)?;

        if let Some(b_ih) = bias_ih {
            gates = gates.add(b_ih)?;
        }
        if let Some(b_hh) = bias_hh {
            gates = gates.add(b_hh)?;
        }

        // Split into i, f, g, o gates
        let i_gate = gates.slice(1, 0, hidden_size)?.sigmoid()?;
        let f_gate = gates.slice(1, hidden_size, 2 * hidden_size)?.sigmoid()?;
        let g_gate = gates.slice(1, 2 * hidden_size, 3 * hidden_size)?.tanh()?;
        let o_gate = gates.slice(1, 3 * hidden_size, 4 * hidden_size)?.sigmoid()?;

        // Update cell state: c = f * c + i * g
        let fc = f_gate.mul(&c)?;
        let ig = i_gate.mul(&g_gate)?;
        c = fc.add(&ig)?;

        // Update hidden state: h = o * tanh(c)
        h = o_gate.mul(&c.tanh()?)?;

        outputs.push(h.clone());
    }

    // Stack outputs along sequence dimension
    let output = Tensor::stack_vec(&outputs, if batch_first { 1 } else { 0 })?;

    // Final state
    let final_h = h.unsqueeze(0)?;
    let final_c = c.unsqueeze(0)?;

    Ok(LstmOutput {
        output,
        state: LstmState {
            h: final_h,
            c: final_c,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lstm_creation() {
        let config = LstmConfig::new(64, 128)
            .with_num_layers(2)
            .with_dropout(0.1);
        let device = Device::Cpu;
        let lstm = Lstm::new(config, &device).unwrap();

        assert_eq!(lstm.config().hidden_size, 128);
        assert_eq!(lstm.config().num_layers, 2);
    }

    #[test]
    fn test_lstm_parameter_count() {
        let config = LstmConfig::new(64, 128).with_num_layers(1);
        let device = Device::Cpu;
        let lstm = Lstm::new(config, &device).unwrap();

        // W_ih: 4*128*64 = 32768
        // W_hh: 4*128*128 = 65536
        // b_ih: 4*128 = 512
        // b_hh: 4*128 = 512
        // Total: 32768 + 65536 + 512 + 512 = 99328
        assert_eq!(lstm.num_parameters(), 99328);
    }
}
