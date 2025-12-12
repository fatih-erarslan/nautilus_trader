//! Custom activation functions for financial neural networks

use ndarray::Array3;
use crate::Result;

/// Custom activation functions optimized for financial data
pub struct FinancialActivations;

impl FinancialActivations {
    /// Modified ReLU that preserves small negative values (for price changes)
    pub fn leaky_relu(input: &mut Array3<f32>, alpha: f32) {
        input.mapv_inplace(|x| if x > 0.0 { x } else { alpha * x });
    }
    
    /// Swish activation with learnable beta parameter
    pub fn swish_beta(input: &mut Array3<f32>, beta: f32) {
        input.mapv_inplace(|x| x / (1.0 + (-beta * x).exp()));
    }
    
    /// Mish activation for better gradient flow
    pub fn mish(input: &mut Array3<f32>) {
        input.mapv_inplace(|x| x * (1.0 + x.exp()).ln().tanh());
    }
    
    /// Custom activation for price momentum
    pub fn momentum_activation(input: &mut Array3<f32>, threshold: f32) {
        input.mapv_inplace(|x| {
            if x.abs() < threshold {
                x * 0.1 // Dampen small movements
            } else {
                x.signum() * (x.abs() - threshold + threshold * 0.1)
            }
        });
    }
    
    /// Tanh with scaling for financial returns
    pub fn scaled_tanh(input: &mut Array3<f32>, scale: f32) {
        input.mapv_inplace(|x| (x * scale).tanh() / scale);
    }
}