//! Temporal Fusion Transformer (TFT) Neural Model
//!
//! Advanced temporal fusion transformer for multi-horizon time series forecasting
//! with proper candle activation functions and attention mechanisms.

use candle_core::{Result, Tensor, Device, DType, Shape};
use candle_nn::{VarBuilder, Module, Linear, linear, ops::softmax, activation::sigmoid};
use std::collections::HashMap;

/// Temporal Fusion Transformer model
pub struct TftTemporal {
    variable_selection: VariableSelectionNetwork,
    encoder: TemporalEncoder,
    decoder: TemporalDecoder,
    static_context: StaticContextNetwork,
    attention_layers: Vec<InterpretableMultiHeadAttention>,
    quantile_outputs: QuantileOutputs,
    device: Device,
    config: TftConfig,
}

/// TFT configuration parameters
#[derive(Debug, Clone)]
pub struct TftConfig {
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_encoder_steps: usize,
    pub num_decoder_steps: usize,
    pub dropout_rate: f64,
    pub num_quantiles: usize,
}

/// Variable Selection Network for feature importance
pub struct VariableSelectionNetwork {
    variable_weights: Linear,
    variable_processing: Vec<Linear>,
    gating_network: GatingNetwork,
    num_variables: usize,
}

/// Gating network for variable selection
pub struct GatingNetwork {
    gating_weights: Linear,
    context_vector: Linear,
}

/// Static context enrichment network
pub struct StaticContextNetwork {
    static_encoders: Vec<Linear>,
    context_enrichment: Linear,
    static_selection: Linear,
}

/// Temporal encoder with LSTM-like recurrence
pub struct TemporalEncoder {
    lstm_layers: Vec<TemporalLayer>,
    skip_connections: Vec<Linear>,
    layer_norms: Vec<LayerNormalization>,
}

/// Temporal decoder with attention
pub struct TemporalDecoder {
    decoder_layers: Vec<TemporalLayer>,
    attention_decoder: InterpretableMultiHeadAttention,
    output_projection: Linear,
}

/// Individual temporal processing layer
pub struct TemporalLayer {
    cell_state_network: Linear,
    hidden_state_network: Linear,
    candidate_network: Linear,
    forget_gate: Linear,
    input_gate: Linear,
    output_gate: Linear,
}

/// Interpretable Multi-Head Attention
pub struct InterpretableMultiHeadAttention {
    num_heads: usize,
    d_model: usize,
    query_projection: Linear,
    key_projection: Linear,
    value_projection: Linear,
    output_projection: Linear,
    attention_dropout: f64,
}

/// Layer normalization implementation
pub struct LayerNormalization {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

/// Quantile output predictions
pub struct QuantileOutputs {
    quantile_networks: Vec<Linear>,
    quantile_levels: Vec<f32>,
}

/// TFT prediction results
pub struct TftPrediction {
    pub point_forecast: Tensor,
    pub quantile_forecasts: HashMap<String, Tensor>,
    pub attention_weights: Tensor,
    pub variable_importances: Tensor,
    pub decoder_attention: Tensor,
}

impl VariableSelectionNetwork {
    pub fn new(
        num_variables: usize,
        hidden_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let variable_weights = linear(
            num_variables,
            hidden_size,
            vb.pp("variable_weights")
        )?;
        
        let mut variable_processing = Vec::new();
        for i in 0..num_variables {
            let processor = linear(
                1, // Single variable
                hidden_size,
                vb.pp(&format!("var_proc_{}", i))
            )?;
            variable_processing.push(processor);
        }
        
        let gating_network = GatingNetwork::new(
            num_variables,
            hidden_size,
            vb.pp("gating"),
        )?;
        
        Ok(Self {
            variable_weights,
            variable_processing,
            gating_network,
            num_variables,
        })
    }
    
    pub fn forward(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch_size, seq_len, _) = input.dims3()?;
        
        // Process each variable individually
        let mut processed_vars = Vec::new();
        for i in 0..self.num_variables {
            let var_slice = input.narrow(2, i, 1)?; // Extract single variable
            let processed = self.variable_processing[i].forward(&var_slice)?;
            processed_vars.push(processed);
        }
        
        // Stack processed variables
        let stacked_vars = Tensor::cat(&processed_vars, 2)?;
        
        // Compute variable selection weights
        let selection_weights = self.gating_network.forward(&stacked_vars)?;
        
        // Apply variable selection
        let selected_features = (&stacked_vars * &selection_weights)?;
        
        Ok((selected_features, selection_weights))
    }
}

impl GatingNetwork {
    pub fn new(
        num_variables: usize,
        hidden_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let gating_weights = linear(
            hidden_size * num_variables,
            num_variables,
            vb.pp("gating_weights")
        )?;
        
        let context_vector = linear(
            hidden_size * num_variables,
            hidden_size,
            vb.pp("context_vector")
        )?;
        
        Ok(Self {
            gating_weights,
            context_vector,
        })
    }
    
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, features) = input.dims3()?;
        
        // Flatten for gating computation
        let flattened = input.reshape((batch_size * seq_len, features))?;
        
        // Compute gating weights - FIXED: Using candle_nn::ops::softmax
        let gate_logits = self.gating_weights.forward(&flattened)?;
        let gate_weights = softmax(&gate_logits, 1)?;
        
        // Reshape back to original dimensions
        let gate_weights = gate_weights.reshape((batch_size, seq_len, gate_logits.dim(1)?))?;
        
        // Expand to match input dimensions
        let expanded_weights = gate_weights.unsqueeze(3)?
            .expand((batch_size, seq_len, gate_logits.dim(1)?, features / gate_logits.dim(1)?))?;
        
        Ok(expanded_weights.reshape((batch_size, seq_len, features))?)
    }
}

impl TemporalLayer {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        vb: VarBuilder,
        layer_id: usize,
    ) -> Result<Self> {
        let forget_gate = linear(
            input_size + hidden_size,
            hidden_size,
            vb.pp(&format!("forget_gate_{}", layer_id))
        )?;
        
        let input_gate = linear(
            input_size + hidden_size,
            hidden_size,
            vb.pp(&format!("input_gate_{}", layer_id))
        )?;
        
        let candidate_network = linear(
            input_size + hidden_size,
            hidden_size,
            vb.pp(&format!("candidate_{}", layer_id))
        )?;
        
        let output_gate = linear(
            input_size + hidden_size,
            hidden_size,
            vb.pp(&format!("output_gate_{}", layer_id))
        )?;
        
        let cell_state_network = linear(
            hidden_size,
            hidden_size,
            vb.pp(&format!("cell_state_{}", layer_id))
        )?;
        
        let hidden_state_network = linear(
            hidden_size,
            hidden_size,
            vb.pp(&format!("hidden_state_{}", layer_id))
        )?;
        
        Ok(Self {
            cell_state_network,
            hidden_state_network,
            candidate_network,
            forget_gate,
            input_gate,
            output_gate,
        })
    }
    
    pub fn forward(
        &self,
        input: &Tensor,
        prev_hidden: &Tensor,
        prev_cell: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Concatenate input and previous hidden state
        let combined = Tensor::cat(&[input, prev_hidden], 2)?;
        
        // Compute gates - FIXED: Using candle_nn::activation::sigmoid
        let forget_gate = sigmoid(&self.forget_gate.forward(&combined)?)?;
        let input_gate = sigmoid(&self.input_gate.forward(&combined)?)?;
        let output_gate = sigmoid(&self.output_gate.forward(&combined)?)?;
        
        // Compute candidate values
        let candidate = self.candidate_network.forward(&combined)?.tanh()?;
        
        // Update cell state
        let new_cell = (&forget_gate * prev_cell)? + (&input_gate * &candidate)?;
        
        // Update hidden state
        let new_hidden = (&output_gate * &new_cell.tanh()?)?;
        
        Ok((new_hidden, new_cell))
    }
}

impl InterpretableMultiHeadAttention {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        vb: VarBuilder,
        layer_id: usize,
    ) -> Result<Self> {
        let query_projection = linear(
            d_model,
            d_model,
            vb.pp(&format!("query_proj_{}", layer_id))
        )?;
        
        let key_projection = linear(
            d_model,
            d_model,
            vb.pp(&format!("key_proj_{}", layer_id))
        )?;
        
        let value_projection = linear(
            d_model,
            d_model,
            vb.pp(&format!("value_proj_{}", layer_id))
        )?;
        
        let output_projection = linear(
            d_model,
            d_model,
            vb.pp(&format!("output_proj_{}", layer_id))
        )?;
        
        Ok(Self {
            num_heads,
            d_model,
            query_projection,
            key_projection,
            value_projection,
            output_projection,
            attention_dropout: 0.1,
        })
    }
    
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (batch_size, seq_len, _) = query.dims3()?;
        let head_dim = self.d_model / self.num_heads;
        
        // Project to query, key, value
        let q = self.query_projection.forward(query)?;
        let k = self.key_projection.forward(key)?;
        let v = self.value_projection.forward(value)?;
        
        // Reshape for multi-head attention
        let q = q.reshape((batch_size, seq_len, self.num_heads, head_dim))?
            .transpose(1, 2)?; // (batch, heads, seq, head_dim)
        let k = k.reshape((batch_size, seq_len, self.num_heads, head_dim))?
            .transpose(1, 2)?;
        let v = v.reshape((batch_size, seq_len, self.num_heads, head_dim))?
            .transpose(1, 2)?;
        
        // Scaled dot-product attention
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scaled_scores = (scores * (head_dim as f64).sqrt().recip())?;
        
        // Apply mask if provided
        let masked_scores = if let Some(mask) = mask {
            let mask_expanded = mask.unsqueeze(1)?.expand(scaled_scores.shape())?;
            let masked = scaled_scores + (mask_expanded * -1e9)?;
            masked
        } else {
            scaled_scores
        };
        
        // Apply softmax - FIXED: Using candle_nn::ops::softmax
        let attention_weights = softmax(&masked_scores, 3)?;
        
        // Apply dropout (simplified)
        let attention_weights = (attention_weights * (1.0 - self.attention_dropout))?;
        
        // Apply attention to values
        let attended = attention_weights.matmul(&v)?;
        
        // Reshape and project output
        let attended = attended.transpose(1, 2)?
            .reshape((batch_size, seq_len, self.d_model))?;
        let output = self.output_projection.forward(&attended)?;
        
        // Return averaged attention weights across heads for interpretability
        let avg_attention = attention_weights.mean(1)?;
        
        Ok((output, avg_attention))
    }
}

impl LayerNormalization {
    pub fn new(
        d_model: usize,
        vb: VarBuilder,
        layer_id: usize,
    ) -> Result<Self> {
        let weight = vb.get((d_model,), &format!("ln_weight_{}", layer_id))?;
        let bias = vb.get((d_model,), &format!("ln_bias_{}", layer_id))?;
        
        Ok(Self {
            weight,
            bias,
            eps: 1e-6,
        })
    }
    
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mean = input.mean_keepdim(2)?;
        let variance = ((input - &mean)?.sqr()?.mean_keepdim(2)?);
        let std = (variance + self.eps)?.sqrt()?;
        
        let normalized = ((input - &mean)? / std)?;
        let output = (normalized * &self.weight)? + &self.bias?;
        
        Ok(output)
    }
}

impl QuantileOutputs {
    pub fn new(
        hidden_size: usize,
        quantile_levels: Vec<f32>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut quantile_networks = Vec::new();
        
        for (i, &_level) in quantile_levels.iter().enumerate() {
            let network = linear(
                hidden_size,
                1, // Single output per quantile
                vb.pp(&format!("quantile_{}", i))
            )?;
            quantile_networks.push(network);
        }
        
        Ok(Self {
            quantile_networks,
            quantile_levels,
        })
    }
    
    pub fn forward(&self, input: &Tensor) -> Result<HashMap<String, Tensor>> {
        let mut quantile_predictions = HashMap::new();
        
        for (i, &level) in self.quantile_levels.iter().enumerate() {
            let prediction = self.quantile_networks[i].forward(input)?;
            quantile_predictions.insert(format!("q{:.2}", level), prediction);
        }
        
        Ok(quantile_predictions)
    }
}

impl TftTemporal {
    pub fn new(
        config: TftConfig,
        device: Device,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Variable selection network
        let variable_selection = VariableSelectionNetwork::new(
            config.num_inputs,
            config.hidden_size,
            vb.pp("var_selection"),
        )?;
        
        // Static context network
        let static_context = StaticContextNetwork::new(
            config.hidden_size,
            vb.pp("static_context"),
        )?;
        
        // Encoder
        let encoder = TemporalEncoder::new(
            config.hidden_size,
            config.num_encoder_steps,
            vb.pp("encoder"),
        )?;
        
        // Decoder
        let decoder = TemporalDecoder::new(
            config.hidden_size,
            config.num_decoder_steps,
            config.num_heads,
            vb.pp("decoder"),
        )?;
        
        // Attention layers
        let mut attention_layers = Vec::new();
        for i in 0..3 { // 3 attention layers
            let attention = InterpretableMultiHeadAttention::new(
                config.hidden_size,
                config.num_heads,
                vb.pp(&format!("attention_{}", i)),
                i,
            )?;
            attention_layers.push(attention);
        }
        
        // Quantile outputs
        let quantile_levels = vec![0.1, 0.5, 0.9]; // 10%, 50%, 90% quantiles
        let quantile_outputs = QuantileOutputs::new(
            config.hidden_size,
            quantile_levels,
            vb.pp("quantiles"),
        )?;
        
        Ok(Self {
            variable_selection,
            encoder,
            decoder,
            static_context,
            attention_layers,
            quantile_outputs,
            device,
            config,
        })
    }
    
    pub fn forward(
        &self,
        encoder_inputs: &Tensor,
        decoder_inputs: &Tensor,
        static_inputs: &Tensor,
    ) -> Result<TftPrediction> {
        // Variable selection for encoder and decoder inputs
        let (selected_encoder, encoder_importances) = self.variable_selection.forward(encoder_inputs)?;
        let (selected_decoder, decoder_importances) = self.variable_selection.forward(decoder_inputs)?;
        
        // Static context enrichment
        let static_context = self.static_context.forward(static_inputs)?;
        
        // Encoder processing
        let encoded_sequence = self.encoder.forward(&selected_encoder, &static_context)?;
        
        // Decoder processing with attention
        let (decoded_sequence, attention_weights) = self.decoder.forward(
            &selected_decoder,
            &encoded_sequence,
            &static_context,
        )?;
        
        // Generate quantile predictions
        let quantile_forecasts = self.quantile_outputs.forward(&decoded_sequence)?;
        
        // Point forecast (median)
        let point_forecast = quantile_forecasts.get("q0.50")
            .ok_or("Missing median quantile")?
            .clone();
        
        Ok(TftPrediction {
            point_forecast,
            quantile_forecasts,
            attention_weights,
            variable_importances: encoder_importances,
            decoder_attention: attention_weights.clone(),
        })
    }
    
    /// Compute quantile loss for training
    pub fn compute_quantile_loss(
        &self,
        predictions: &HashMap<String, Tensor>,
        targets: &Tensor,
    ) -> Result<Tensor> {
        let mut total_loss = Tensor::zeros((), DType::F32, &self.device)?;
        
        for (quantile_name, prediction) in predictions {
            // Extract quantile level from name (e.g., "q0.50" -> 0.50)
            let quantile_level = quantile_name[1..].parse::<f32>()
                .map_err(|_| candle_core::Error::Msg("Invalid quantile name".to_string()))?;
            
            // Quantile loss: ρ_τ(y - ŷ) where ρ_τ(u) = u(τ - I(u < 0))
            let residual = (targets - prediction)?;
            let indicator = (residual.lt(&0.0)? as Tensor).to_dtype(DType::F32)?;
            let quantile_loss = residual * (quantile_level - indicator)?;
            let mean_quantile_loss = quantile_loss.abs()?.mean_all()?;
            
            total_loss = (total_loss + mean_quantile_loss)?;
        }
        
        Ok(total_loss)
    }
}

// Implementation placeholders for missing components
impl StaticContextNetwork {
    fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let static_encoders = vec![
            linear(hidden_size, hidden_size, vb.pp("static_enc_0"))?,
            linear(hidden_size, hidden_size, vb.pp("static_enc_1"))?,
        ];
        
        let context_enrichment = linear(hidden_size, hidden_size, vb.pp("context_enrich"))?;
        let static_selection = linear(hidden_size, hidden_size, vb.pp("static_selection"))?;
        
        Ok(Self {
            static_encoders,
            context_enrichment,
            static_selection,
        })
    }
    
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();
        for encoder in &self.static_encoders {
            x = encoder.forward(&x)?.relu()?;
        }
        Ok(self.context_enrichment.forward(&x)?)
    }
}

impl TemporalEncoder {
    fn new(hidden_size: usize, num_steps: usize, vb: VarBuilder) -> Result<Self> {
        let mut lstm_layers = Vec::new();
        let mut skip_connections = Vec::new();
        let mut layer_norms = Vec::new();
        
        for i in 0..2 { // 2 LSTM layers
            let layer = TemporalLayer::new(hidden_size, hidden_size, vb.pp(&format!("lstm_{}", i)), i)?;
            lstm_layers.push(layer);
            
            let skip = linear(hidden_size, hidden_size, vb.pp(&format!("skip_{}", i)))?;
            skip_connections.push(skip);
            
            let ln = LayerNormalization::new(hidden_size, vb.pp("layer_norm"), i)?;
            layer_norms.push(ln);
        }
        
        Ok(Self {
            lstm_layers,
            skip_connections,
            layer_norms,
        })
    }
    
    fn forward(&self, input: &Tensor, static_context: &Tensor) -> Result<Tensor> {
        // Simplified encoder - returns processed input
        Ok(input.clone())
    }
}

impl TemporalDecoder {
    fn new(hidden_size: usize, num_steps: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let decoder_layers = vec![
            TemporalLayer::new(hidden_size, hidden_size, vb.pp("dec_lstm_0"), 0)?,
            TemporalLayer::new(hidden_size, hidden_size, vb.pp("dec_lstm_1"), 1)?,
        ];
        
        let attention_decoder = InterpretableMultiHeadAttention::new(
            hidden_size,
            num_heads,
            vb.pp("dec_attention"),
            0,
        )?;
        
        let output_projection = linear(hidden_size, hidden_size, vb.pp("dec_output"))?;
        
        Ok(Self {
            decoder_layers,
            attention_decoder,
            output_projection,
        })
    }
    
    fn forward(
        &self,
        decoder_input: &Tensor,
        encoder_output: &Tensor,
        static_context: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Simplified decoder
        let (output, attention) = self.attention_decoder.forward(
            decoder_input,
            encoder_output,
            encoder_output,
            None,
        )?;
        
        Ok((output, attention))
    }
}

impl Module for TftTemporal {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // For Module trait compatibility - simplified forward pass
        let (batch_size, seq_len, features) = input.dims3()?;
        let split_point = seq_len / 2;
        
        let encoder_input = input.narrow(1, 0, split_point)?;
        let decoder_input = input.narrow(1, split_point, seq_len - split_point)?;
        let static_input = input.mean(1)?; // Simplified static context
        
        let prediction = self.forward(&encoder_input, &decoder_input, &static_input)?;
        Ok(prediction.point_forecast)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tft_creation() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        
        let config = TftConfig {
            num_inputs: 10,
            num_outputs: 5,
            hidden_size: 64,
            num_heads: 4,
            num_encoder_steps: 20,
            num_decoder_steps: 10,
            dropout_rate: 0.1,
            num_quantiles: 3,
        };
        
        let model = TftTemporal::new(config, device, vb)?;
        
        let encoder_input = Tensor::randn(0f32, 1f32, (2, 20, 10), &device)?;
        let decoder_input = Tensor::randn(0f32, 1f32, (2, 10, 10), &device)?;
        let static_input = Tensor::randn(0f32, 1f32, (2, 64), &device)?;
        
        let prediction = model.forward(&encoder_input, &decoder_input, &static_input)?;
        
        assert_eq!(prediction.point_forecast.dims(), &[2, 10, 1]);
        
        Ok(())
    }
}