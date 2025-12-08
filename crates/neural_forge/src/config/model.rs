//! Model configuration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::error::{Result, NeuralForgeError};

/// Model architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model architecture type
    pub architecture: ArchitectureConfig,
    
    /// Input dimension
    pub input_dim: usize,
    
    /// Output dimension
    pub output_dim: usize,
    
    /// Hidden dimensions
    pub hidden_dims: Vec<usize>,
    
    /// Activation function
    pub activation: ActivationConfig,
    
    /// Dropout rate
    pub dropout: f64,
    
    /// Batch normalization
    pub batch_norm: bool,
    
    /// Layer normalization
    pub layer_norm: bool,
    
    /// Weight initialization
    pub init: InitializationConfig,
    
    /// Custom model parameters
    pub custom_params: HashMap<String, serde_json::Value>,
}

/// Neural network architecture types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArchitectureConfig {
    /// Multi-layer perceptron
    Mlp {
        layers: Vec<usize>,
        skip_connections: bool,
    },
    
    /// Convolutional neural network
    Cnn {
        conv_layers: Vec<ConvLayerConfig>,
        fc_layers: Vec<usize>,
        global_pooling: PoolingConfig,
    },
    
    /// Recurrent neural network
    Rnn {
        rnn_type: RnnType,
        hidden_size: usize,
        num_layers: usize,
        bidirectional: bool,
        dropout: f64,
    },
    
    /// Transformer architecture
    Transformer {
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        d_ff: usize,
        max_seq_length: usize,
        positional_encoding: PositionalEncodingConfig,
        attention_config: AttentionConfig,
    },
    
    /// Mixture of Experts
    MoE {
        num_experts: usize,
        expert_config: Box<ArchitectureConfig>,
        gating_config: GatingConfig,
        top_k: usize,
    },
    
    /// Vision Transformer
    ViT {
        patch_size: (usize, usize),
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        mlp_ratio: f64,
    },
    
    /// Residual Network
    ResNet {
        blocks: Vec<ResidualBlockConfig>,
        stem_config: StemConfig,
    },
    
    /// Custom architecture
    Custom {
        name: String,
        config: HashMap<String, serde_json::Value>,
    },
}

/// Convolutional layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvLayerConfig {
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
    pub groups: usize,
}

/// Pooling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PoolingConfig {
    None,
    Max { kernel_size: (usize, usize) },
    Average { kernel_size: (usize, usize) },
    Adaptive { output_size: (usize, usize) },
    Global,
}

/// RNN types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RnnType {
    Vanilla,
    Lstm,
    Gru,
}

/// Positional encoding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PositionalEncodingConfig {
    Sinusoidal,
    Learned,
    Rotary,
    None,
}

/// Attention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    pub attention_type: AttentionType,
    pub dropout: f64,
    pub bias: bool,
    pub scale: Option<f64>,
}

/// Attention types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttentionType {
    MultiHead,
    LocalWindow { window_size: usize },
    Sparse { sparsity_pattern: SparsityPattern },
    Linear,
    FlashAttention,
}

/// Sparsity patterns for sparse attention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SparsityPattern {
    Fixed { pattern: Vec<Vec<bool>> },
    Random { sparsity: f64 },
    Strided { stride: usize },
    BlockSparse { block_size: usize },
}

/// Gating configuration for MoE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatingConfig {
    pub gating_type: GatingType,
    pub noise_epsilon: f64,
    pub capacity_factor: f64,
}

/// Gating types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GatingType {
    TopK,
    SwitchGating,
    ExpertChoice,
}

/// Residual block configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualBlockConfig {
    pub block_type: ResidualBlockType,
    pub channels: usize,
    pub stride: usize,
}

/// Residual block types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResidualBlockType {
    Basic,
    Bottleneck,
    PreActivation,
}

/// Stem configuration for ResNet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StemConfig {
    pub conv_size: usize,
    pub pool_size: usize,
}

/// Activation function configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationConfig {
    ReLU,
    LeakyReLU { negative_slope: f64 },
    ELU { alpha: f64 },
    SELU,
    Swish,
    GELU,
    Mish,
    Tanh,
    Sigmoid,
    Softmax { dim: i64 },
    LogSoftmax { dim: i64 },
    PReLU,
    Custom { name: String },
}

/// Weight initialization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InitializationConfig {
    Xavier {
        uniform: bool,
        gain: f64,
    },
    Kaiming {
        uniform: bool,
        mode: KaimingMode,
        nonlinearity: String,
    },
    Normal {
        mean: f64,
        std: f64,
    },
    Uniform {
        low: f64,
        high: f64,
    },
    Constant {
        value: f64,
    },
    Identity,
    Orthogonal {
        gain: f64,
    },
    Custom {
        name: String,
        params: HashMap<String, f64>,
    },
}

/// Kaiming initialization modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KaimingMode {
    FanIn,
    FanOut,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architecture: ArchitectureConfig::Transformer {
                d_model: 256,
                num_heads: 8,
                num_layers: 6,
                d_ff: 1024,
                max_seq_length: 512,
                positional_encoding: PositionalEncodingConfig::Sinusoidal,
                attention_config: AttentionConfig {
                    attention_type: AttentionType::MultiHead,
                    dropout: 0.1,
                    bias: true,
                    scale: None,
                },
            },
            input_dim: 128,
            output_dim: 1,
            hidden_dims: vec![256, 128],
            activation: ActivationConfig::GELU,
            dropout: 0.1,
            batch_norm: false,
            layer_norm: true,
            init: InitializationConfig::Xavier {
                uniform: false,
                gain: 1.0,
            },
            custom_params: HashMap::new(),
        }
    }
}

impl ModelConfig {
    /// Validate model configuration
    pub fn validate(&self) -> Result<()> {
        if self.input_dim == 0 {
            return Err(NeuralForgeError::config("Input dimension must be > 0"));
        }
        
        if self.output_dim == 0 {
            return Err(NeuralForgeError::config("Output dimension must be > 0"));
        }
        
        if self.dropout < 0.0 || self.dropout >= 1.0 {
            return Err(NeuralForgeError::config("Dropout must be in [0, 1)"));
        }
        
        // Validate architecture-specific parameters
        match &self.architecture {
            ArchitectureConfig::Transformer { num_heads, d_model, .. } => {
                if d_model % num_heads != 0 {
                    return Err(NeuralForgeError::config(
                        "d_model must be divisible by num_heads"
                    ));
                }
            }
            ArchitectureConfig::MoE { num_experts, top_k, .. } => {
                if *top_k > *num_experts {
                    return Err(NeuralForgeError::config(
                        "top_k cannot be greater than num_experts"
                    ));
                }
            }
            _ => {}
        }
        
        Ok(())
    }
    
    /// Create transformer configuration
    pub fn transformer() -> Self {
        Self {
            architecture: ArchitectureConfig::Transformer {
                d_model: 256,
                num_heads: 8,
                num_layers: 6,
                d_ff: 1024,
                max_seq_length: 512,
                positional_encoding: PositionalEncodingConfig::Sinusoidal,
                attention_config: AttentionConfig {
                    attention_type: AttentionType::MultiHead,
                    dropout: 0.1,
                    bias: true,
                    scale: None,
                },
            },
            ..Default::default()
        }
    }
    
    /// Create MLP configuration
    pub fn mlp(layers: Vec<usize>) -> Self {
        Self {
            architecture: ArchitectureConfig::Mlp {
                layers,
                skip_connections: false,
            },
            ..Default::default()
        }
    }
    
    /// Create CNN configuration
    pub fn cnn() -> Self {
        Self {
            architecture: ArchitectureConfig::Cnn {
                conv_layers: vec![
                    ConvLayerConfig {
                        out_channels: 32,
                        kernel_size: (3, 3),
                        stride: (1, 1),
                        padding: (1, 1),
                        dilation: (1, 1),
                        groups: 1,
                    },
                ],
                fc_layers: vec![128],
                global_pooling: PoolingConfig::Global,
            },
            ..Default::default()
        }
    }
    
    /// Create LSTM configuration
    pub fn lstm(hidden_size: usize, num_layers: usize) -> Self {
        Self {
            architecture: ArchitectureConfig::Rnn {
                rnn_type: RnnType::Lstm,
                hidden_size,
                num_layers,
                bidirectional: false,
                dropout: 0.1,
            },
            ..Default::default()
        }
    }
    
    /// Builder methods
    pub fn with_layers(mut self, num_layers: usize) -> Self {
        match &mut self.architecture {
            ArchitectureConfig::Transformer { num_layers: layers, .. } => {
                *layers = num_layers;
            }
            ArchitectureConfig::Rnn { num_layers: layers, .. } => {
                *layers = num_layers;
            }
            _ => {}
        }
        self
    }
    
    pub fn with_hidden_size(mut self, hidden_size: usize) -> Self {
        match &mut self.architecture {
            ArchitectureConfig::Transformer { d_model, .. } => {
                *d_model = hidden_size;
            }
            ArchitectureConfig::Rnn { hidden_size: size, .. } => {
                *size = hidden_size;
            }
            _ => {}
        }
        self
    }
    
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }
    
    pub fn with_activation(mut self, activation: ActivationConfig) -> Self {
        self.activation = activation;
        self
    }
}