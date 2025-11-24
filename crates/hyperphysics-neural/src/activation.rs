//! Activation functions for neural networks
//!
//! Includes standard activations optimized for HFT inference latency.

use serde::{Deserialize, Serialize};
use crate::core::Tensor;

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Activation {
    /// Linear (identity)
    Linear,
    /// Rectified Linear Unit: max(0, x)
    ReLU,
    /// Leaky ReLU: max(α*x, x)
    LeakyReLU,
    /// Parametric ReLU (learnable α)
    PReLU,
    /// Exponential Linear Unit
    ELU,
    /// Scaled ELU (self-normalizing)
    SELU,
    /// Gaussian Error Linear Unit
    GELU,
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// Softmax (for output layer)
    Softmax,
    /// Swish: x * sigmoid(x)
    Swish,
    /// Mish: x * tanh(softplus(x))
    Mish,
    /// Hard Sigmoid (faster approximation)
    HardSigmoid,
    /// Hard Swish (faster approximation)
    HardSwish,
    /// SiLU (Sigmoid Linear Unit, same as Swish)
    SiLU,
}

impl Default for Activation {
    fn default() -> Self {
        Activation::ReLU
    }
}

impl Activation {
    /// Apply activation function to a single value
    #[inline(always)]
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::Linear => x,
            Activation::ReLU => x.max(0.0),
            Activation::LeakyReLU => if x > 0.0 { x } else { 0.01 * x },
            Activation::PReLU => if x > 0.0 { x } else { 0.25 * x }, // Default α=0.25
            Activation::ELU => if x > 0.0 { x } else { x.exp() - 1.0 },
            Activation::SELU => {
                const ALPHA: f64 = 1.6732632423543772;
                const LAMBDA: f64 = 1.0507009873554805;
                if x > 0.0 { LAMBDA * x } else { LAMBDA * ALPHA * (x.exp() - 1.0) }
            },
            Activation::GELU => {
                // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
                let coeff = (2.0 / std::f64::consts::PI).sqrt();
                0.5 * x * (1.0 + (coeff * (x + 0.044715 * x.powi(3))).tanh())
            },
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::Softmax => x.exp(), // Actual softmax needs normalization across vector
            Activation::Swish | Activation::SiLU => x / (1.0 + (-x).exp()),
            Activation::Mish => x * (1.0 + x.exp()).ln().tanh(),
            Activation::HardSigmoid => ((x + 3.0) / 6.0).max(0.0).min(1.0),
            Activation::HardSwish => x * ((x + 3.0) / 6.0).max(0.0).min(1.0),
        }
    }

    /// Apply derivative of activation function
    #[inline(always)]
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::Linear => 1.0,
            Activation::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            Activation::LeakyReLU => if x > 0.0 { 1.0 } else { 0.01 },
            Activation::PReLU => if x > 0.0 { 1.0 } else { 0.25 },
            Activation::ELU => if x > 0.0 { 1.0 } else { x.exp() },
            Activation::SELU => {
                const ALPHA: f64 = 1.6732632423543772;
                const LAMBDA: f64 = 1.0507009873554805;
                if x > 0.0 { LAMBDA } else { LAMBDA * ALPHA * x.exp() }
            },
            Activation::GELU => {
                // Numerical approximation
                let coeff = (2.0 / std::f64::consts::PI).sqrt();
                let inner = coeff * (x + 0.044715 * x.powi(3));
                let tanh_inner = inner.tanh();
                let sech2 = 1.0 - tanh_inner * tanh_inner;
                0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * coeff * (1.0 + 3.0 * 0.044715 * x * x)
            },
            Activation::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            },
            Activation::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            },
            Activation::Softmax => {
                // Softmax derivative is Jacobian, not scalar - return 1 for now
                1.0
            },
            Activation::Swish | Activation::SiLU => {
                let s = 1.0 / (1.0 + (-x).exp());
                s + x * s * (1.0 - s)
            },
            Activation::Mish => {
                let sp = (1.0 + x.exp()).ln();
                let tanh_sp = sp.tanh();
                // Simplified Mish derivative
                tanh_sp + x * (1.0 - tanh_sp * tanh_sp) * (x.exp() / (1.0 + x.exp()))
            },
            Activation::HardSigmoid => {
                if x < -3.0 || x > 3.0 { 0.0 } else { 1.0 / 6.0 }
            },
            Activation::HardSwish => {
                if x <= -3.0 { 0.0 }
                else if x >= 3.0 { 1.0 }
                else { (2.0 * x + 3.0) / 6.0 }
            },
        }
    }

    /// Apply activation to tensor (element-wise)
    pub fn forward(&self, input: &Tensor) -> Tensor {
        match self {
            Activation::Softmax => {
                // Softmax needs special handling - normalize across last dimension
                let data = input.data();
                let max_val = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exp_data: Vec<f64> = data.iter().map(|x| (x - max_val).exp()).collect();
                let sum: f64 = exp_data.iter().sum();
                let softmax_data: Vec<f64> = exp_data.iter().map(|x| x / sum).collect();
                Tensor::new(softmax_data, input.shape().clone()).unwrap()
            },
            _ => input.map(|x| self.apply(x)),
        }
    }

    /// Apply activation derivative to tensor
    pub fn backward(&self, input: &Tensor) -> Tensor {
        input.map(|x| self.derivative(x))
    }

    /// Check if activation is suitable for output layer
    pub fn is_output_activation(&self) -> bool {
        matches!(self,
            Activation::Sigmoid |
            Activation::Softmax |
            Activation::Tanh |
            Activation::Linear
        )
    }

    /// Get recommended initialization for this activation
    pub fn recommended_init(&self) -> &'static str {
        match self {
            Activation::ReLU | Activation::LeakyReLU | Activation::PReLU |
            Activation::ELU | Activation::GELU | Activation::Swish |
            Activation::Mish | Activation::SiLU => "he",
            Activation::SELU => "lecun",
            _ => "xavier",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let relu = Activation::ReLU;
        assert_eq!(relu.apply(-1.0), 0.0);
        assert_eq!(relu.apply(0.0), 0.0);
        assert_eq!(relu.apply(1.0), 1.0);
        assert_eq!(relu.derivative(-1.0), 0.0);
        assert_eq!(relu.derivative(1.0), 1.0);
    }

    #[test]
    fn test_sigmoid() {
        let sigmoid = Activation::Sigmoid;
        assert!((sigmoid.apply(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid.apply(10.0) > 0.99);
        assert!(sigmoid.apply(-10.0) < 0.01);
    }

    #[test]
    fn test_tanh() {
        let tanh = Activation::Tanh;
        assert!((tanh.apply(0.0)).abs() < 1e-10);
        assert!(tanh.apply(10.0) > 0.99);
        assert!(tanh.apply(-10.0) < -0.99);
    }

    #[test]
    fn test_softmax() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], crate::core::TensorShape::d1(3)).unwrap();
        let softmax = Activation::Softmax.forward(&t);

        // Should sum to 1
        let sum: f64 = softmax.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Should be monotonically increasing with input
        assert!(softmax.data()[0] < softmax.data()[1]);
        assert!(softmax.data()[1] < softmax.data()[2]);
    }

    #[test]
    fn test_gelu() {
        let gelu = Activation::GELU;
        // GELU(0) ≈ 0
        assert!(gelu.apply(0.0).abs() < 1e-10);
        // GELU is approximately linear for large positive x
        assert!((gelu.apply(3.0) - 3.0).abs() < 0.1);
        // GELU(-x) ≈ 0 for large negative x
        assert!(gelu.apply(-3.0).abs() < 0.1);
    }

    #[test]
    fn test_tensor_activation() {
        let t = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], crate::core::TensorShape::d1(5)).unwrap();
        let result = Activation::ReLU.forward(&t);
        assert_eq!(result.data(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
    }
}
