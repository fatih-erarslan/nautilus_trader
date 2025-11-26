//! Multi-Head Attention layer for Transformer architectures

use crate::backends::Device;
use crate::error::MlResult;
use crate::tensor::{DType, Tensor, TensorOps};
use super::{Layer, Linear, LinearConfig};
use serde::{Deserialize, Serialize};

/// Multi-Head Attention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiHeadAttentionConfig {
    /// Model dimension (d_model)
    pub embed_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dropout probability
    pub dropout: f32,
    /// Whether to use bias in projections
    pub bias: bool,
    /// Key/Value dimension (defaults to embed_dim / num_heads)
    pub kdim: Option<usize>,
    /// Value dimension (defaults to embed_dim / num_heads)
    pub vdim: Option<usize>,
}

impl MultiHeadAttentionConfig {
    /// Create new config
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        assert!(
            embed_dim % num_heads == 0,
            "embed_dim must be divisible by num_heads"
        );
        Self {
            embed_dim,
            num_heads,
            dropout: 0.0,
            bias: true,
            kdim: None,
            vdim: None,
        }
    }

    /// Set dropout
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Set bias
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Head dimension
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }
}

impl Default for MultiHeadAttentionConfig {
    fn default() -> Self {
        Self::new(512, 8)
    }
}

/// Multi-Head Attention layer
///
/// Implements scaled dot-product attention:
/// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
///
/// With multiple heads, the computation is parallelized:
/// MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
/// where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    config: MultiHeadAttentionConfig,
    /// Query projection [embed_dim, embed_dim]
    q_proj: Linear,
    /// Key projection [kdim, embed_dim]
    k_proj: Linear,
    /// Value projection [vdim, embed_dim]
    v_proj: Linear,
    /// Output projection [embed_dim, embed_dim]
    out_proj: Linear,
    /// Scaling factor 1/sqrt(d_k)
    scale: f32,
    /// Device
    device: Device,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention layer
    pub fn new(config: MultiHeadAttentionConfig, device: &Device) -> MlResult<Self> {
        let kdim = config.kdim.unwrap_or(config.embed_dim);
        let vdim = config.vdim.unwrap_or(config.embed_dim);
        let head_dim = config.head_dim();

        // Query projection
        let q_proj = Linear::new(
            LinearConfig::new(config.embed_dim, config.embed_dim).with_bias(config.bias),
            device,
        )?;

        // Key projection
        let k_proj = Linear::new(
            LinearConfig::new(kdim, config.embed_dim).with_bias(config.bias),
            device,
        )?;

        // Value projection
        let v_proj = Linear::new(
            LinearConfig::new(vdim, config.embed_dim).with_bias(config.bias),
            device,
        )?;

        // Output projection
        let out_proj = Linear::new(
            LinearConfig::new(config.embed_dim, config.embed_dim).with_bias(config.bias),
            device,
        )?;

        let scale = 1.0 / (head_dim as f32).sqrt();

        Ok(Self {
            config,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            scale,
            device: device.clone(),
        })
    }

    /// Forward pass with separate query, key, value inputs
    ///
    /// Args:
    ///     query: [batch, seq_len, embed_dim]
    ///     key: [batch, src_len, kdim]
    ///     value: [batch, src_len, vdim]
    ///     mask: Optional attention mask [batch, seq_len, src_len]
    pub fn forward_qkv(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> MlResult<Tensor> {
        let query_shape = query.shape().dims();
        let batch_size = query_shape[0];
        let seq_len = query_shape[1];
        let src_len = key.shape().dims()[1];
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim();

        // Project Q, K, V
        let q = self.q_proj.forward(query)?;  // [batch, seq, embed]
        let k = self.k_proj.forward(key)?;    // [batch, src, embed]
        let v = self.v_proj.forward(value)?;  // [batch, src, embed]

        // Reshape for multi-head: [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
        let q = q.reshape(vec![batch_size, seq_len, num_heads, head_dim])?
            .transpose_dims(1, 2)?;
        let k = k.reshape(vec![batch_size, src_len, num_heads, head_dim])?
            .transpose_dims(1, 2)?;
        let v = v.reshape(vec![batch_size, src_len, num_heads, head_dim])?
            .transpose_dims(1, 2)?;

        // Compute attention scores: [batch, heads, seq, src]
        let scores = q.matmul(&k.transpose_dims(2, 3)?)?;
        let scores = scores.mul_scalar(self.scale)?;

        // Apply mask if provided
        let scores = if let Some(m) = mask {
            // Mask should be broadcastable to [batch, heads, seq, src]
            let mask_expanded = m.unsqueeze(1)?;  // [batch, 1, seq, src]
            apply_attention_mask(&scores, &mask_expanded)?
        } else {
            scores
        };

        // Softmax over source dimension (last dimension)
        let attn_weights = scores.softmax()?;

        // Apply attention to values: [batch, heads, seq, head_dim]
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, embed]
        let attn_output = attn_output
            .transpose_dims(1, 2)?
            .reshape(vec![batch_size, seq_len, self.config.embed_dim])?;

        // Output projection
        self.out_proj.forward(&attn_output)
    }

    /// Self-attention (query = key = value)
    pub fn forward_self_attention(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
    ) -> MlResult<Tensor> {
        self.forward_qkv(x, x, x, mask)
    }

    /// Get configuration
    pub fn config(&self) -> &MultiHeadAttentionConfig {
        &self.config
    }
}

impl Layer for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> MlResult<Tensor> {
        // Default to self-attention
        self.forward_self_attention(input, None)
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn num_parameters(&self) -> usize {
        self.q_proj.num_parameters()
            + self.k_proj.num_parameters()
            + self.v_proj.num_parameters()
            + self.out_proj.num_parameters()
    }

    fn to_device(&mut self, device: &Device) -> MlResult<()> {
        self.q_proj.to_device(device)?;
        self.k_proj.to_device(device)?;
        self.v_proj.to_device(device)?;
        self.out_proj.to_device(device)?;
        self.device = device.clone();
        Ok(())
    }
}

/// Apply attention mask (set masked positions to -inf before softmax)
fn apply_attention_mask(scores: &Tensor, mask: &Tensor) -> MlResult<Tensor> {
    // Where mask is 0, set score to -inf
    // scores + (1 - mask) * -1e9
    let neg_inf = -1e9_f32;
    let ones = Tensor::ones(mask.shape().dims().to_vec(), DType::F32, mask.device())?;
    let inv_mask = ones.sub(mask)?;
    let mask_addend = inv_mask.mul_scalar(neg_inf)?;
    scores.add(&mask_addend)
}

/// Causal attention mask for autoregressive models
pub fn create_causal_mask(seq_len: usize, device: &Device) -> MlResult<Tensor> {
    // Lower triangular matrix
    let mut mask_data = vec![0.0_f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..=i {
            mask_data[i * seq_len + j] = 1.0;
        }
    }
    Tensor::from_slice(&mask_data, vec![seq_len, seq_len], device)
}

/// Transformer encoder layer with self-attention and feed-forward
#[derive(Debug, Clone)]
pub struct TransformerEncoderLayer {
    /// Self-attention
    self_attn: MultiHeadAttention,
    /// Feed-forward network
    ff1: Linear,
    ff2: Linear,
    /// Layer normalization
    norm1: super::LayerNorm,
    norm2: super::LayerNorm,
    /// Dropout probability
    dropout: f32,
    /// Device
    device: Device,
}

impl TransformerEncoderLayer {
    /// Create a new transformer encoder layer
    pub fn new(
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        dropout: f32,
        device: &Device,
    ) -> MlResult<Self> {
        let self_attn = MultiHeadAttention::new(
            MultiHeadAttentionConfig::new(d_model, num_heads).with_dropout(dropout),
            device,
        )?;

        let ff1 = Linear::new(LinearConfig::new(d_model, d_ff), device)?;
        let ff2 = Linear::new(LinearConfig::new(d_ff, d_model), device)?;

        let norm1 = super::LayerNorm::new(
            super::LayerNormConfig::new(d_model),
            device,
        )?;
        let norm2 = super::LayerNorm::new(
            super::LayerNormConfig::new(d_model),
            device,
        )?;

        Ok(Self {
            self_attn,
            ff1,
            ff2,
            norm1,
            norm2,
            dropout,
            device: device.clone(),
        })
    }

    /// Forward pass
    pub fn forward_with_mask(&self, x: &Tensor, mask: Option<&Tensor>) -> MlResult<Tensor> {
        // Self-attention with residual
        let attn_out = self.self_attn.forward_self_attention(x, mask)?;
        let x = self.norm1.forward(&x.add(&attn_out)?)?;

        // Feed-forward with residual
        let ff_out = self.ff2.forward(&self.ff1.forward(&x)?.relu()?)?;
        let x = self.norm2.forward(&x.add(&ff_out)?)?;

        Ok(x)
    }
}

impl Layer for TransformerEncoderLayer {
    fn forward(&self, input: &Tensor) -> MlResult<Tensor> {
        self.forward_with_mask(input, None)
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn num_parameters(&self) -> usize {
        self.self_attn.num_parameters()
            + self.ff1.num_parameters()
            + self.ff2.num_parameters()
            + self.norm1.num_parameters()
            + self.norm2.num_parameters()
    }

    fn to_device(&mut self, device: &Device) -> MlResult<()> {
        self.self_attn.to_device(device)?;
        self.ff1.to_device(device)?;
        self.ff2.to_device(device)?;
        self.norm1.to_device(device)?;
        self.norm2.to_device(device)?;
        self.device = device.clone();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_creation() {
        let config = MultiHeadAttentionConfig::new(512, 8);
        let device = Device::Cpu;
        let attn = MultiHeadAttention::new(config, &device).unwrap();

        assert_eq!(attn.config().num_heads, 8);
        assert_eq!(attn.config().head_dim(), 64);
    }

    #[test]
    fn test_causal_mask() {
        let device = Device::Cpu;
        let mask = create_causal_mask(4, &device).unwrap();

        // Should be lower triangular
        assert_eq!(mask.shape().dims(), &[4, 4]);
    }

    #[test]
    fn test_transformer_encoder_layer() {
        let device = Device::Cpu;
        let layer = TransformerEncoderLayer::new(512, 8, 2048, 0.1, &device).unwrap();

        // Parameter count should be substantial
        assert!(layer.num_parameters() > 0);
    }
}
