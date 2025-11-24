//! Integration with ruv-FANN neural network library
//!
//! Provides high-performance neural network inference using the Fast Artificial
//! Neural Network (FANN) library implemented in pure Rust.
//!
//! ## Features
//!
//! - **HFT-optimized inference**: Sub-microsecond forward passes
//! - **Training algorithms**: Backprop, RProp, QuickProp, Cascade
//! - **SIMD acceleration**: Vectorized operations when available
//! - **WebGPU support**: GPU acceleration for larger networks

mod network;
mod backend;

pub use network::{FannNetwork, FannConfig, FannError};
pub use backend::FannBackend;

// Re-export core ruv-fann types for advanced usage
pub use ruv_fann::{
    ActivationFunction as FannActivation,
    TrainingAlgorithm as FannTrainingAlgorithm,
    TrainingData as FannTrainingData,
    NetworkBuilder as FannNetworkBuilder,
    CascadeConfig, CascadeNetwork, CascadeTrainer,
};

/// Convert our Activation enum to ruv-fann's ActivationFunction
pub fn to_fann_activation(activation: crate::activation::Activation) -> FannActivation {
    match activation {
        crate::activation::Activation::Linear => FannActivation::Linear,
        crate::activation::Activation::ReLU => FannActivation::ReLU,
        crate::activation::Activation::LeakyReLU => FannActivation::ReLULeaky,
        crate::activation::Activation::Sigmoid => FannActivation::Sigmoid,
        crate::activation::Activation::Tanh => FannActivation::Tanh,
        crate::activation::Activation::GELU => FannActivation::Sigmoid, // Approximate
        crate::activation::Activation::Swish | crate::activation::Activation::SiLU => FannActivation::Sigmoid,
        crate::activation::Activation::HardSigmoid => FannActivation::LinearPiece,
        crate::activation::Activation::HardSwish => FannActivation::LinearPiece,
        crate::activation::Activation::ELU | crate::activation::Activation::SELU => FannActivation::ReLULeaky,
        crate::activation::Activation::PReLU => FannActivation::ReLULeaky,
        crate::activation::Activation::Mish => FannActivation::Tanh,
        crate::activation::Activation::Softmax => FannActivation::Sigmoid, // Per-element
    }
}

/// Convert ruv-fann's ActivationFunction to our Activation enum
pub fn from_fann_activation(activation: FannActivation) -> crate::activation::Activation {
    match activation {
        FannActivation::Linear => crate::activation::Activation::Linear,
        FannActivation::ReLU => crate::activation::Activation::ReLU,
        FannActivation::ReLULeaky => crate::activation::Activation::LeakyReLU,
        FannActivation::Sigmoid => crate::activation::Activation::Sigmoid,
        FannActivation::Tanh | FannActivation::SigmoidSymmetric => crate::activation::Activation::Tanh,
        FannActivation::Elliot | FannActivation::ElliotSymmetric => crate::activation::Activation::Sigmoid,
        FannActivation::LinearPiece | FannActivation::LinearPieceSymmetric => crate::activation::Activation::HardSigmoid,
        FannActivation::Gaussian | FannActivation::GaussianSymmetric => crate::activation::Activation::Sigmoid,
        _ => crate::activation::Activation::Linear,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_conversion() {
        let relu = crate::activation::Activation::ReLU;
        let fann_relu = to_fann_activation(relu);
        assert_eq!(fann_relu, FannActivation::ReLU);

        let back = from_fann_activation(fann_relu);
        assert_eq!(back, crate::activation::Activation::ReLU);
    }

    #[test]
    fn test_sigmoid_conversion() {
        let sigmoid = crate::activation::Activation::Sigmoid;
        let fann_sigmoid = to_fann_activation(sigmoid);
        assert_eq!(fann_sigmoid, FannActivation::Sigmoid);
    }
}
