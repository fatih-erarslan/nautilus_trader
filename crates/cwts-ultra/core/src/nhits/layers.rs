// Hierarchical neural network layers for NHITS

use super::model::{ActivationType, BasisType};

pub struct HierarchicalBlock {
    pub layers: Vec<NeuralLayer>,
    pub skip_connection: bool,
    pub layer_norm: Option<LayerNorm>,
}

pub struct BasisExpansion {
    pub basis_type: BasisType,
    pub n_bases: usize,
    pub learnable: bool,
    pub coefficients: Vec<f32>,
}

pub struct NeuralLayer {
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
    activation: ActivationType,
    use_bias: bool,
}

pub struct LayerNorm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
    eps: f32,
}

impl HierarchicalBlock {
    pub fn new(n_layers: usize, hidden_size: usize, activation: ActivationType) -> Self {
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(NeuralLayer::new(hidden_size, hidden_size, activation));
        }
        
        Self {
            layers,
            skip_connection: true,
            layer_norm: Some(LayerNorm::new(hidden_size)),
        }
    }
    
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut output = x.to_vec();
        let input = x.to_vec();
        
        for layer in &self.layers {
            output = layer.forward(&output);
        }
        
        if let Some(ref norm) = self.layer_norm {
            output = norm.forward(&output);
        }
        
        if self.skip_connection {
            for (i, val) in output.iter_mut().enumerate() {
                *val += input[i];
            }
        }
        
        output
    }
}

impl NeuralLayer {
    pub fn new(input_size: usize, output_size: usize, activation: ActivationType) -> Self {
        let scale = (2.0 / input_size as f32).sqrt();
        let mut weights = vec![vec![0.0; input_size]; output_size];
        
        // Xavier initialization
        for row in weights.iter_mut() {
            for val in row.iter_mut() {
                *val = (rand::random::<f32>() - 0.5) * 2.0 * scale;
            }
        }
        
        Self {
            weights,
            bias: vec![0.0; output_size],
            activation,
            use_bias: true,
        }
    }
    
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut output = if self.use_bias {
            self.bias.clone()
        } else {
            vec![0.0; self.bias.len()]
        };
        
        // Matrix multiplication
        for (i, row) in self.weights.iter().enumerate() {
            for (j, &val) in x.iter().enumerate() {
                output[i] += row[j] * val;
            }
        }
        
        // Apply activation
        self.apply_activation(&mut output);
        output
    }
    
    fn apply_activation(&self, output: &mut [f32]) {
        match self.activation {
            ActivationType::ReLU => {
                for val in output.iter_mut() {
                    *val = val.max(0.0);
                }
            }
            ActivationType::GELU => {
                for val in output.iter_mut() {
                    *val = 0.5 * *val * (1.0 + (*val * std::f32::consts::FRAC_2_PI.sqrt() * 
                           (1.0 + 0.044715 * val.powi(3))).tanh());
                }
            }
            ActivationType::SiLU => {
                for val in output.iter_mut() {
                    *val = *val / (1.0 + (-*val).exp());
                }
            }
            ActivationType::Tanh => {
                for val in output.iter_mut() {
                    *val = val.tanh();
                }
            }
        }
    }
}

impl LayerNorm {
    pub fn new(size: usize) -> Self {
        Self {
            gamma: vec![1.0; size],
            beta: vec![0.0; size],
            eps: 1e-5,
        }
    }
    
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mean = x.iter().sum::<f32>() / x.len() as f32;
        let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32;
        
        x.iter().enumerate().map(|(i, &val)| {
            let normalized = (val - mean) / (variance + self.eps).sqrt();
            self.gamma[i % self.gamma.len()] * normalized + self.beta[i % self.beta.len()]
        }).collect()
    }
}

impl BasisExpansion {
    pub fn new(basis_type: BasisType, n_bases: usize) -> Self {
        Self {
            basis_type,
            n_bases,
            learnable: true,
            coefficients: vec![1.0 / n_bases as f32; n_bases],
        }
    }
    
    pub fn expand(&self, x: &[f32], horizon: usize) -> Vec<f32> {
        match self.basis_type {
            BasisType::Polynomial => self.polynomial_expansion(x, horizon),
            BasisType::Fourier => self.fourier_expansion(x, horizon),
            BasisType::Wavelet => self.wavelet_expansion(x, horizon),
            BasisType::LearnedBasis => self.learned_expansion(x, horizon),
        }
    }
    
    fn polynomial_expansion(&self, x: &[f32], horizon: usize) -> Vec<f32> {
        let mut expanded = Vec::with_capacity(horizon);
        
        for h in 0..horizon {
            let t = h as f32 / horizon as f32;
            let mut value = 0.0;
            
            for (i, &coef) in self.coefficients.iter().enumerate() {
                value += coef * t.powi(i as i32);
            }
            
            expanded.push(value);
        }
        
        expanded
    }
    
    fn fourier_expansion(&self, x: &[f32], horizon: usize) -> Vec<f32> {
        let mut expanded = Vec::with_capacity(horizon);
        
        for h in 0..horizon {
            let t = h as f32 / horizon as f32;
            let mut value = self.coefficients[0];
            
            for i in 1..self.n_bases {
                let freq = 2.0 * std::f32::consts::PI * i as f32;
                value += self.coefficients[i] * (freq * t).cos();
                if i < self.n_bases - 1 {
                    value += self.coefficients[i] * (freq * t).sin();
                }
            }
            
            expanded.push(value);
        }
        
        expanded
    }
    
    fn wavelet_expansion(&self, x: &[f32], horizon: usize) -> Vec<f32> {
        // Simplified Haar wavelet
        let mut expanded = Vec::with_capacity(horizon);
        
        for h in 0..horizon {
            let t = h as f32 / horizon as f32;
            let mut value = 0.0;
            
            for (level, &coef) in self.coefficients.iter().enumerate() {
                let scale = 2.0_f32.powi(level as i32);
                let wavelet = self.haar_wavelet(t * scale);
                value += coef * wavelet;
            }
            
            expanded.push(value);
        }
        
        expanded
    }
    
    fn haar_wavelet(&self, t: f32) -> f32 {
        let t_mod = t % 1.0;
        if t_mod < 0.5 {
            1.0
        } else {
            -1.0
        }
    }
    
    fn learned_expansion(&self, x: &[f32], horizon: usize) -> Vec<f32> {
        // Use neural network to learn basis
        let mut expanded = Vec::with_capacity(horizon);
        
        for h in 0..horizon {
            let mut value = 0.0;
            for (i, &coef) in self.coefficients.iter().enumerate() {
                if i < x.len() {
                    value += coef * x[i];
                }
            }
            expanded.push(value);
        }
        
        expanded
    }
}

use rand;