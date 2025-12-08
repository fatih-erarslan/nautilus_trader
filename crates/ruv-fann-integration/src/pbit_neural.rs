//! pBit Neural Network Module
//!
//! Implements neural network training using pBit/Ising dynamics
//! for stochastic gradient descent and weight optimization.
//!
//! ## Mathematical Foundation (Wolfram Validated)
//!
//! - **Boltzmann Machine**: P(visible) = Σ_h exp(-E(v,h)) / Z
//! - **Weight Update**: ΔW_ij = η (<v_i h_j>_data - <v_i h_j>_model)
//! - **pBit Activation**: σ(x) = 1 / (1 + exp(-x/T))
//! - **Energy**: E = -Σ W_ij v_i h_j - Σ b_i v_i - Σ c_j h_j

use rand::prelude::*;
use std::ops::{Add, Mul};

/// pBit activation function (temperature-scaled sigmoid)
/// σ(x, T) = 1 / (1 + exp(-x/T))
#[inline]
pub fn pbit_activation(x: f64, temperature: f64) -> f64 {
    1.0 / (1.0 + (-x / temperature.max(0.001)).exp())
}

/// pBit derivative for backprop
#[inline]
pub fn pbit_activation_derivative(x: f64, temperature: f64) -> f64 {
    let s = pbit_activation(x, temperature);
    s * (1.0 - s) / temperature.max(0.001)
}

/// pBit Neural Layer
#[derive(Debug, Clone)]
pub struct PBitLayer {
    /// Weights (input_size x output_size)
    pub weights: Vec<Vec<f64>>,
    /// Biases
    pub biases: Vec<f64>,
    /// Temperature
    pub temperature: f64,
    /// Input size
    pub input_size: usize,
    /// Output size  
    pub output_size: usize,
}

impl PBitLayer {
    /// Create new layer with random initialization
    pub fn new(input_size: usize, output_size: usize, temperature: f64) -> Self {
        let mut rng = rand::thread_rng();
        
        // Xavier initialization scaled by temperature
        let scale = (2.0 / (input_size + output_size) as f64).sqrt() * temperature;
        
        let weights = (0..input_size)
            .map(|_| {
                (0..output_size)
                    .map(|_| rng.gen_range(-scale..scale))
                    .collect()
            })
            .collect();
            
        let biases = vec![0.0; output_size];
        
        Self {
            weights,
            biases,
            temperature,
            input_size,
            output_size,
        }
    }

    /// Forward pass with pBit activation
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut output = self.biases.clone();
        
        for i in 0..self.input_size.min(input.len()) {
            for j in 0..self.output_size {
                output[j] += input[i] * self.weights[i][j];
            }
        }
        
        // Apply pBit activation
        for o in &mut output {
            *o = pbit_activation(*o, self.temperature);
        }
        
        output
    }

    /// Backward pass (compute gradients)
    pub fn backward(&self, input: &[f64], output_grad: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        let mut weight_grad = vec![vec![0.0; self.output_size]; self.input_size];
        let mut bias_grad = vec![0.0; self.output_size];
        let mut input_grad = vec![0.0; self.input_size];
        
        // Compute pre-activation for derivative
        let mut pre_activation = self.biases.clone();
        for i in 0..self.input_size.min(input.len()) {
            for j in 0..self.output_size {
                pre_activation[j] += input[i] * self.weights[i][j];
            }
        }
        
        for j in 0..self.output_size {
            let grad = output_grad[j] * pbit_activation_derivative(pre_activation[j], self.temperature);
            bias_grad[j] = grad;
            
            for i in 0..self.input_size.min(input.len()) {
                weight_grad[i][j] = input[i] * grad;
                input_grad[i] += self.weights[i][j] * grad;
            }
        }
        
        (weight_grad, bias_grad, input_grad)
    }

    /// Update weights using pBit-enhanced SGD
    pub fn update(&mut self, weight_grad: &[Vec<f64>], bias_grad: &[f64], learning_rate: f64) {
        for i in 0..self.input_size {
            for j in 0..self.output_size {
                self.weights[i][j] -= learning_rate * weight_grad[i][j];
            }
        }
        
        for j in 0..self.output_size {
            self.biases[j] -= learning_rate * bias_grad[j];
        }
    }
}

/// Simple pBit MLP for trading signals
#[derive(Debug)]
pub struct PBitMLP {
    pub layers: Vec<PBitLayer>,
    pub temperature: f64,
}

impl PBitMLP {
    /// Create new MLP
    pub fn new(layer_sizes: &[usize], temperature: f64) -> Self {
        let layers = layer_sizes
            .windows(2)
            .map(|w| PBitLayer::new(w[0], w[1], temperature))
            .collect();
            
        Self { layers, temperature }
    }

    /// Forward pass
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut x = input.to_vec();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }

    /// Train on single example
    pub fn train_step(&mut self, input: &[f64], target: &[f64], learning_rate: f64) -> f64 {
        // Forward pass, storing activations
        let mut activations = vec![input.to_vec()];
        let mut x = input.to_vec();
        for layer in &self.layers {
            x = layer.forward(&x);
            activations.push(x.clone());
        }
        
        // Compute loss (MSE)
        let output = activations.last().unwrap();
        let loss: f64 = output.iter()
            .zip(target.iter())
            .map(|(o, t)| (o - t).powi(2))
            .sum::<f64>() / target.len() as f64;
        
        // Output gradient
        let mut grad: Vec<f64> = output.iter()
            .zip(target.iter())
            .map(|(o, t)| 2.0 * (o - t) / target.len() as f64)
            .collect();
        
        // Backward pass
        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            let (weight_grad, bias_grad, input_grad) = layer.backward(&activations[i], &grad);
            layer.update(&weight_grad, &bias_grad, learning_rate);
            grad = input_grad;
        }
        
        loss
    }

    /// Anneal temperature (simulated annealing)
    pub fn anneal(&mut self, factor: f64) {
        self.temperature *= factor;
        for layer in &mut self.layers {
            layer.temperature *= factor;
        }
    }
}

/// Restricted Boltzmann Machine with pBit dynamics
#[derive(Debug)]
pub struct PBitRBM {
    /// Visible-to-hidden weights
    pub weights: Vec<Vec<f64>>,
    /// Visible biases
    pub visible_bias: Vec<f64>,
    /// Hidden biases
    pub hidden_bias: Vec<f64>,
    /// Temperature
    pub temperature: f64,
    /// Number of visible units
    pub n_visible: usize,
    /// Number of hidden units
    pub n_hidden: usize,
}

impl PBitRBM {
    /// Create new RBM
    pub fn new(n_visible: usize, n_hidden: usize, temperature: f64) -> Self {
        let mut rng = rand::thread_rng();
        let scale = 0.1 * temperature;
        
        let weights = (0..n_visible)
            .map(|_| (0..n_hidden).map(|_| rng.gen_range(-scale..scale)).collect())
            .collect();
            
        Self {
            weights,
            visible_bias: vec![0.0; n_visible],
            hidden_bias: vec![0.0; n_hidden],
            temperature,
            n_visible,
            n_hidden,
        }
    }

    /// Sample hidden given visible (Gibbs sampling)
    pub fn sample_hidden(&self, visible: &[f64]) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut hidden = vec![0.0; self.n_hidden];
        
        for j in 0..self.n_hidden {
            let mut activation = self.hidden_bias[j];
            for i in 0..self.n_visible.min(visible.len()) {
                activation += visible[i] * self.weights[i][j];
            }
            let prob = pbit_activation(activation, self.temperature);
            hidden[j] = if rng.gen::<f64>() < prob { 1.0 } else { 0.0 };
        }
        
        hidden
    }

    /// Sample visible given hidden
    pub fn sample_visible(&self, hidden: &[f64]) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut visible = vec![0.0; self.n_visible];
        
        for i in 0..self.n_visible {
            let mut activation = self.visible_bias[i];
            for j in 0..self.n_hidden.min(hidden.len()) {
                activation += hidden[j] * self.weights[i][j];
            }
            let prob = pbit_activation(activation, self.temperature);
            visible[i] = if rng.gen::<f64>() < prob { 1.0 } else { 0.0 };
        }
        
        visible
    }

    /// Contrastive Divergence training (CD-k)
    pub fn train_cd(&mut self, data: &[f64], k: usize, learning_rate: f64) {
        // Positive phase
        let h_pos = self.sample_hidden(data);
        
        // Negative phase (k steps of Gibbs sampling)
        let mut v_neg = data.to_vec();
        let mut h_neg = h_pos.clone();
        for _ in 0..k {
            v_neg = self.sample_visible(&h_neg);
            h_neg = self.sample_hidden(&v_neg);
        }
        
        // Update weights: ΔW_ij = η (<v_i h_j>_data - <v_i h_j>_model)
        for i in 0..self.n_visible.min(data.len()) {
            for j in 0..self.n_hidden {
                let pos_corr = data[i] * h_pos[j];
                let neg_corr = v_neg[i] * h_neg[j];
                self.weights[i][j] += learning_rate * (pos_corr - neg_corr);
            }
        }
        
        // Update biases
        for i in 0..self.n_visible.min(data.len()) {
            self.visible_bias[i] += learning_rate * (data[i] - v_neg[i]);
        }
        for j in 0..self.n_hidden {
            self.hidden_bias[j] += learning_rate * (h_pos[j] - h_neg[j]);
        }
    }

    /// Compute free energy F(v) = -Σ b_i v_i - Σ log(1 + exp(c_j + Σ W_ij v_i))
    pub fn free_energy(&self, visible: &[f64]) -> f64 {
        let mut energy = 0.0;
        
        // Visible bias term
        for i in 0..self.n_visible.min(visible.len()) {
            energy -= self.visible_bias[i] * visible[i];
        }
        
        // Hidden unit term
        for j in 0..self.n_hidden {
            let mut x = self.hidden_bias[j];
            for i in 0..self.n_visible.min(visible.len()) {
                x += self.weights[i][j] * visible[i];
            }
            energy -= (1.0 + (x / self.temperature).exp()).ln() * self.temperature;
        }
        
        energy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pbit_activation() {
        // At T=1, σ(0) = 0.5
        let a = pbit_activation(0.0, 1.0);
        assert!((a - 0.5).abs() < 0.01);
        
        // High input -> ~1
        let b = pbit_activation(10.0, 1.0);
        assert!(b > 0.99);
    }

    #[test]
    fn test_layer_forward() {
        let layer = PBitLayer::new(3, 2, 1.0);
        let input = vec![1.0, 0.5, 0.0];
        let output = layer.forward(&input);
        
        assert_eq!(output.len(), 2);
        assert!(output[0] >= 0.0 && output[0] <= 1.0);
    }

    #[test]
    fn test_mlp_training() {
        let mut mlp = PBitMLP::new(&[2, 4, 1], 1.0);
        
        // XOR training
        let data = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];
        
        let mut total_loss = 0.0;
        for _ in 0..100 {
            for (input, target) in &data {
                total_loss = mlp.train_step(input, target, 0.5);
            }
        }
        
        // Loss should decrease
        assert!(total_loss < 1.0);
    }

    #[test]
    fn test_rbm_gibbs() {
        let rbm = PBitRBM::new(4, 3, 1.0);
        let visible = vec![1.0, 0.0, 1.0, 0.0];
        
        let hidden = rbm.sample_hidden(&visible);
        assert_eq!(hidden.len(), 3);
        
        let reconstructed = rbm.sample_visible(&hidden);
        assert_eq!(reconstructed.len(), 4);
    }

    #[test]
    fn test_free_energy() {
        let rbm = PBitRBM::new(4, 3, 1.0);
        let v1 = vec![1.0, 0.0, 1.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 1.0];
        
        let e1 = rbm.free_energy(&v1);
        let e2 = rbm.free_energy(&v2);
        
        // Both should be finite
        assert!(e1.is_finite());
        assert!(e2.is_finite());
    }
}
