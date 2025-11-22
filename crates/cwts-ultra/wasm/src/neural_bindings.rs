//! Neural Network Bindings for WASM
//! Lightweight neural network implementation for trading decisions

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct JSTradingNN {
    weights: Vec<f64>,
    biases: Vec<f64>,
    layers: Vec<usize>,
}

#[wasm_bindgen]
impl JSTradingNN {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            weights: vec![0.5, 0.3, -0.2, 0.8, -0.1, 0.4],
            biases: vec![0.1, -0.05, 0.2],
            layers: vec![4, 8, 1], // 4 inputs, 8 hidden, 1 output
        }
    }
    
    pub fn predict(&self, inputs: &[f64]) -> f64 {
        if inputs.len() < 4 {
            return 0.5; // Neutral prediction
        }
        
        // Simple neural network forward pass
        let mut activation = inputs[0] * self.weights[0] + 
                          inputs[1] * self.weights[1] + 
                          inputs[2] * self.weights[2] + 
                          inputs[3] * self.weights[3] + 
                          self.biases[0];
        
        // Apply sigmoid activation
        activation = 1.0 / (1.0 + (-activation).exp());
        
        // Second layer
        activation = activation * self.weights[4] + self.biases[1];
        activation = 1.0 / (1.0 + (-activation).exp());
        
        // Output layer
        activation = activation * self.weights[5] + self.biases[2];
        1.0 / (1.0 + (-activation).exp())
    }
    
    pub fn train(&mut self, _inputs: &[f64], _target: f64) {
        // Simplified training - in production would implement backpropagation
        for weight in &mut self.weights {
            *weight += (js_sys::Math::random() - 0.5) * 0.01;
        }
    }
}

impl Default for JSTradingNN {
    fn default() -> Self {
        Self::new()
    }
}