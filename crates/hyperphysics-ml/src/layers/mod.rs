//! Neural network layers
//!
//! Provides common layer implementations that work across all backends.

mod linear;
mod lstm;
mod attention;
mod conv;
mod normalization;

pub use linear::{Linear, LinearConfig};
pub use lstm::{Lstm, LstmConfig, LstmOutput};
pub use attention::{MultiHeadAttention, MultiHeadAttentionConfig};
pub use conv::{Conv1d, Conv1dConfig, TemporalBlock, TcnConfig};
pub use normalization::{LayerNorm, LayerNormConfig, BatchNorm1d};

use crate::backends::Device;
use crate::error::MlResult;
use crate::tensor::{Tensor, TensorOps};

/// Layer trait for neural network modules
pub trait Layer {
    /// Forward pass
    fn forward(&self, input: &Tensor) -> MlResult<Tensor>;

    /// Get the device this layer is on
    fn device(&self) -> &Device;

    /// Number of trainable parameters
    fn num_parameters(&self) -> usize;

    /// Move layer to device
    fn to_device(&mut self, device: &Device) -> MlResult<()>;
}

/// Trait for layers with hidden state (RNNs)
pub trait StatefulLayer: Layer {
    /// Hidden state type
    type State;

    /// Forward with state
    fn forward_with_state(
        &self,
        input: &Tensor,
        state: Option<Self::State>,
    ) -> MlResult<(Tensor, Self::State)>;

    /// Reset hidden state
    fn reset_state(&mut self);
}
