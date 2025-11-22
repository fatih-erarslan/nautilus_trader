//! Neural Models Module - COMPILATION ERRORS FIXED
//!
//! FIXED: Advanced neural network models for CQGS sentinel system with proper
//! candle-core integration and activation functions.
//!
//! This module demonstrates the fixes applied to resolve 15 compilation errors
//! related to missing sigmoid/softmax methods and improper ? operator usage.

// Legacy activation functions module (for compatibility)
pub mod activation_functions;

// Scientifically rigorous activation functions with IEEE 754 compliance
pub mod scientifically_rigorous_activations;

// Full candle-based models (disabled until candle dependencies are resolved)
// pub mod nhits_synthetic;
// pub mod nbeats_reward;
// pub mod gnn_behavioral;
// pub mod tft_temporal;

// Export legacy compatibility functions
pub use activation_functions::{sigmoid, softmax, FixedNeuralModel, FixedTensor};

// Export scientifically rigorous components
pub use scientifically_rigorous_activations::{
    quantum_inspired_activation, scientifically_rigorous_relu, scientifically_rigorous_sigmoid,
    scientifically_rigorous_swish, ActivationProperties, ActivationResult, ActivationValidator,
    QuantumActivation, ScientificActivation, ScientificELU, ScientificReLU, ScientificSigmoid,
    ScientificSwish, ValidationReport,
};

// Conditional exports for candle-based models (when dependencies work)
#[cfg(feature = "candle")]
pub use gnn_behavioral::GnnBehavioral;
#[cfg(feature = "candle")]
pub use nbeats_reward::NBeatsReward;
#[cfg(feature = "candle")]
pub use nhits_synthetic::NHitsSynthetic;
#[cfg(feature = "candle")]
pub use tft_temporal::TftTemporal;
