//! Backpropagation engine with automatic differentiation
//!
//! This module provides a gradient tape-based automatic differentiation system
//! for neural network training. It supports:
//! - Automatic gradient computation via chain rule
//! - Gradient accumulation and clipping
//! - All common activation functions
//! - Efficient memory management

use ndarray::{Array1, Array2, Axis};
use crate::{Result, NeuroDivergentError};
use std::collections::HashMap;

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    ReLU,
    GELU,
    Tanh,
    Sigmoid,
    Swish,
    Linear,
}

impl Activation {
    /// Forward pass through activation function
    #[inline]
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        match self {
            Activation::ReLU => x.mapv(|v| v.max(0.0)),
            Activation::GELU => {
                // GELU(x) = x * Φ(x) where Φ is Gaussian CDF
                // Approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
                x.mapv(|v| {
                    0.5 * v * (1.0 + ((2.0_f64 / std::f64::consts::PI).sqrt()
                        * (v + 0.044715 * v.powi(3))).tanh())
                })
            },
            Activation::Tanh => x.mapv(|v| v.tanh()),
            Activation::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            Activation::Swish => x.mapv(|v| v / (1.0 + (-v).exp())),
            Activation::Linear => x.clone(),
        }
    }

    /// Backward pass - compute derivative
    #[inline]
    pub fn backward(&self, x: &Array2<f64>, grad_output: &Array2<f64>) -> Array2<f64> {
        match self {
            Activation::ReLU => {
                grad_output * &x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
            },
            Activation::GELU => {
                // d/dx GELU(x) approximation
                x.iter().zip(grad_output.iter())
                    .map(|(x_val, grad_val)| {
                        let x = *x_val;
                        let cdf = 0.5 * (1.0 + ((2.0_f64 / std::f64::consts::PI).sqrt()
                            * (x + 0.044715 * x.powi(3))).tanh());
                        let pdf = (2.0_f64 / std::f64::consts::PI).sqrt()
                            * (-(x.powi(2)) / 2.0).exp();
                        grad_val * (cdf + x * pdf)
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .collect::<Array1<_>>()
                    .into_shape(grad_output.dim())
                    .unwrap()
            },
            Activation::Tanh => {
                let tanh_x = x.mapv(|v| v.tanh());
                grad_output * &tanh_x.mapv(|t| 1.0 - t.powi(2))
            },
            Activation::Sigmoid => {
                let sigmoid_x = self.forward(x);
                grad_output * &(&sigmoid_x * &sigmoid_x.mapv(|s| 1.0 - s))
            },
            Activation::Swish => {
                let sigmoid_x = x.mapv(|v| 1.0 / (1.0 + (-v).exp()));
                grad_output * &(x.mapv(|_| 0.0) + &sigmoid_x + x * &sigmoid_x.mapv(|s| 1.0 - s))
            },
            Activation::Linear => grad_output.clone(),
        }
    }
}

/// Gradient tape for automatic differentiation
pub struct GradientTape {
    /// Operations recorded during forward pass
    operations: Vec<Operation>,
    /// Gradients accumulated during backward pass
    gradients: HashMap<String, Array2<f64>>,
    /// Whether to record operations
    recording: bool,
}

/// An operation in the computational graph
#[derive(Debug, Clone)]
struct Operation {
    /// Operation type
    op_type: OpType,
    /// Input tensor IDs
    inputs: Vec<String>,
    /// Output tensor ID
    output: String,
    /// Cached values needed for backward pass
    cache: OpCache,
}

#[derive(Debug, Clone)]
enum OpType {
    MatMul,
    Add,
    Activation(Activation),
    Dropout(f64),
}

#[derive(Debug, Clone)]
struct OpCache {
    /// For MatMul: left and right matrices
    matmul_left: Option<Array2<f64>>,
    matmul_right: Option<Array2<f64>>,
    /// For Activation: input before activation
    activation_input: Option<Array2<f64>>,
    /// For Dropout: mask used
    dropout_mask: Option<Array2<f64>>,
}

impl Default for OpCache {
    fn default() -> Self {
        Self {
            matmul_left: None,
            matmul_right: None,
            activation_input: None,
            dropout_mask: None,
        }
    }
}

impl GradientTape {
    /// Create a new gradient tape
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            gradients: HashMap::new(),
            recording: false,
        }
    }

    /// Start recording operations
    pub fn record(&mut self) {
        self.recording = true;
        self.operations.clear();
        self.gradients.clear();
    }

    /// Stop recording operations
    pub fn stop(&mut self) {
        self.recording = false;
    }

    /// Record matrix multiplication: C = A @ B
    pub fn matmul(
        &mut self,
        a: &Array2<f64>,
        b: &Array2<f64>,
        input_ids: (String, String),
        output_id: String,
    ) -> Array2<f64> {
        let result = a.dot(b);

        if self.recording {
            self.operations.push(Operation {
                op_type: OpType::MatMul,
                inputs: vec![input_ids.0, input_ids.1],
                output: output_id,
                cache: OpCache {
                    matmul_left: Some(a.clone()),
                    matmul_right: Some(b.clone()),
                    ..Default::default()
                },
            });
        }

        result
    }

    /// Record addition: C = A + B
    pub fn add(
        &mut self,
        a: &Array2<f64>,
        b: &Array2<f64>,
        input_ids: (String, String),
        output_id: String,
    ) -> Array2<f64> {
        let result = a + b;

        if self.recording {
            self.operations.push(Operation {
                op_type: OpType::Add,
                inputs: vec![input_ids.0, input_ids.1],
                output: output_id,
                cache: OpCache::default(),
            });
        }

        result
    }

    /// Record activation function
    pub fn activation(
        &mut self,
        x: &Array2<f64>,
        activation: Activation,
        input_id: String,
        output_id: String,
    ) -> Array2<f64> {
        let result = activation.forward(x);

        if self.recording {
            self.operations.push(Operation {
                op_type: OpType::Activation(activation),
                inputs: vec![input_id],
                output: output_id,
                cache: OpCache {
                    activation_input: Some(x.clone()),
                    ..Default::default()
                },
            });
        }

        result
    }

    /// Record dropout
    pub fn dropout(
        &mut self,
        x: &Array2<f64>,
        p: f64,
        training: bool,
        input_id: String,
        output_id: String,
    ) -> Array2<f64> {
        if !training || p == 0.0 {
            return x.clone();
        }

        use rand::Rng;
        let mut rng = rand::thread_rng();
        let keep_prob = 1.0 - p;

        let mask = Array2::from_shape_fn(x.dim(), |_| {
            if rng.gen::<f64>() < keep_prob {
                1.0 / keep_prob  // Inverted dropout
            } else {
                0.0
            }
        });

        let result = x * &mask;

        if self.recording {
            self.operations.push(Operation {
                op_type: OpType::Dropout(p),
                inputs: vec![input_id],
                output: output_id,
                cache: OpCache {
                    dropout_mask: Some(mask),
                    ..Default::default()
                },
            });
        }

        result
    }

    /// Perform backward pass and compute gradients
    pub fn backward(&mut self, loss_grad: Array2<f64>, output_id: String) -> Result<()> {
        // Initialize gradient for output
        self.gradients.insert(output_id.clone(), loss_grad);

        // Clone operations to avoid borrow conflicts
        let operations = self.operations.clone();

        // Backpropagate through operations in reverse order
        for op in operations.iter().rev() {
            let grad_output = self.gradients.get(&op.output)
                .ok_or_else(|| NeuroDivergentError::TrainingError(
                    format!("Missing gradient for {}", op.output)
                ))?
                .clone();

            match &op.op_type {
                OpType::MatMul => {
                    // grad_A = grad_output @ B^T
                    // grad_B = A^T @ grad_output
                    let a = op.cache.matmul_left.as_ref().unwrap();
                    let b = op.cache.matmul_right.as_ref().unwrap();

                    let grad_a = grad_output.dot(&b.t());
                    let grad_b = a.t().dot(&grad_output);

                    self.accumulate_gradient(&op.inputs[0], grad_a);
                    self.accumulate_gradient(&op.inputs[1], grad_b);
                },
                OpType::Add => {
                    // Gradient flows equally to both inputs
                    self.accumulate_gradient(&op.inputs[0], grad_output.clone());
                    self.accumulate_gradient(&op.inputs[1], grad_output);
                },
                OpType::Activation(activation) => {
                    let x = op.cache.activation_input.as_ref().unwrap();
                    let grad_input = activation.backward(x, &grad_output);
                    self.accumulate_gradient(&op.inputs[0], grad_input);
                },
                OpType::Dropout(_) => {
                    let mask = op.cache.dropout_mask.as_ref().unwrap();
                    let grad_input = &grad_output * mask;
                    self.accumulate_gradient(&op.inputs[0], grad_input);
                },
            }
        }

        Ok(())
    }

    /// Accumulate gradient for a tensor
    fn accumulate_gradient(&mut self, tensor_id: &str, grad: Array2<f64>) {
        self.gradients
            .entry(tensor_id.to_string())
            .and_modify(|existing| *existing = &*existing + &grad)
            .or_insert(grad);
    }

    /// Get gradient for a tensor
    pub fn get_gradient(&self, tensor_id: &str) -> Option<&Array2<f64>> {
        self.gradients.get(tensor_id)
    }

    /// Clear all gradients
    pub fn zero_grad(&mut self) {
        self.gradients.clear();
    }
}

/// Gradient clipping strategies
pub enum GradientClipping {
    /// Clip by value: clip each gradient element to [-threshold, threshold]
    ByValue(f64),
    /// Clip by norm: scale gradients if total norm exceeds threshold
    ByNorm(f64),
    /// No clipping
    None,
}

impl GradientClipping {
    /// Apply clipping to gradients
    pub fn clip(&self, gradients: &mut [Array2<f64>]) {
        match self {
            GradientClipping::ByValue(threshold) => {
                for grad in gradients.iter_mut() {
                    grad.mapv_inplace(|g| g.clamp(-threshold, *threshold));
                }
            },
            GradientClipping::ByNorm(max_norm) => {
                // Compute total gradient norm
                let total_norm: f64 = gradients
                    .iter()
                    .map(|g| g.iter().map(|x| x.powi(2)).sum::<f64>())
                    .sum::<f64>()
                    .sqrt();

                if total_norm > *max_norm {
                    let scale = max_norm / total_norm;
                    for grad in gradients.iter_mut() {
                        grad.mapv_inplace(|g| g * scale);
                    }
                }
            },
            GradientClipping::None => {},
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    /// Helper function for array comparison with tolerance
    fn arrays_close(a: &ndarray::Array2<f64>, b: &ndarray::Array2<f64>, epsilon: f64) -> bool {
        if a.shape() != b.shape() {
            return false;
        }
        (a - b).mapv(|v| v.abs()).iter().all(|&v| v < epsilon)
    }

    #[test]
    fn test_relu_forward() {
        let x = arr2(&[[1.0, -2.0], [3.0, -4.0]]);
        let result = Activation::ReLU.forward(&x);
        let expected = arr2(&[[1.0, 0.0], [3.0, 0.0]]);
        assert!(arrays_close(&result, &expected, 1e-10), "ReLU forward failed");
    }

    #[test]
    fn test_relu_backward() {
        let x = arr2(&[[1.0, -2.0], [3.0, -4.0]]);
        let grad_out = arr2(&[[1.0, 1.0], [1.0, 1.0]]);
        let result = Activation::ReLU.backward(&x, &grad_out);
        let expected = arr2(&[[1.0, 0.0], [1.0, 0.0]]);
        assert!(arrays_close(&result, &expected, 1e-10), "ReLU backward failed");
    }

    #[test]
    fn test_sigmoid_forward_backward() {
        let x = arr2(&[[0.0, 1.0], [-1.0, 2.0]]);
        let forward = Activation::Sigmoid.forward(&x);

        // Check forward pass values
        assert!((forward[[0, 0]] - 0.5).abs() < 1e-10, "Sigmoid(0) should be 0.5");
        let expected_val = 1.0 / (1.0 + (-1.0_f64).exp());
        assert!((forward[[0, 1]] - expected_val).abs() < 1e-10, "Sigmoid(1) mismatch");

        // Check backward pass (numerical gradient)
        let grad_out = arr2(&[[1.0, 1.0], [1.0, 1.0]]);
        let backward = Activation::Sigmoid.backward(&x, &grad_out);

        // Sigmoid derivative at x=0 should be 0.25
        assert!((backward[[0, 0]] - 0.25).abs() < 1e-10, "Sigmoid derivative at 0 should be 0.25");
    }

    #[test]
    fn test_gradient_tape_matmul() {
        let mut tape = GradientTape::new();
        tape.record();

        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = arr2(&[[5.0, 6.0], [7.0, 8.0]]);

        let c = tape.matmul(
            &a,
            &b,
            ("a".to_string(), "b".to_string()),
            "c".to_string(),
        );

        // Forward pass check
        let expected = arr2(&[[19.0, 22.0], [43.0, 50.0]]);
        assert!(arrays_close(&c, &expected, 1e-10), "MatMul forward failed");

        // Backward pass
        let grad_c = arr2(&[[1.0, 1.0], [1.0, 1.0]]);
        tape.backward(grad_c, "c".to_string()).unwrap();

        // Check gradients
        let grad_a = tape.get_gradient("a").unwrap();
        let grad_b = tape.get_gradient("b").unwrap();

        let expected_grad_a = arr2(&[[11.0, 15.0], [11.0, 15.0]]);
        let expected_grad_b = arr2(&[[4.0, 4.0], [6.0, 6.0]]);

        assert!(arrays_close(grad_a, &expected_grad_a, 1e-10), "Gradient for 'a' failed");
        assert!(arrays_close(grad_b, &expected_grad_b, 1e-10), "Gradient for 'b' failed");
    }

    #[test]
    fn test_gradient_clipping_by_value() {
        let mut grads = vec![arr2(&[[5.0, -10.0], [15.0, -20.0]])];
        GradientClipping::ByValue(8.0).clip(&mut grads);

        let expected = arr2(&[[5.0, -8.0], [8.0, -8.0]]);
        assert!(arrays_close(&grads[0], &expected, 1e-10), "Gradient clipping by value failed");
    }

    #[test]
    fn test_gradient_clipping_by_norm() {
        let mut grads = vec![
            arr2(&[[3.0, 4.0]]),  // norm = 5
        ];

        GradientClipping::ByNorm(2.5).clip(&mut grads);

        // Should scale by 2.5/5.0 = 0.5
        let expected = arr2(&[[1.5, 2.0]]);
        assert!(arrays_close(&grads[0], &expected, 1e-10), "Gradient clipping by norm failed");
    }
}
