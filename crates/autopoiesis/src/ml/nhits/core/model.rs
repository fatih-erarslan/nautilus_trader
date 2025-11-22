//! Core NHITS Model Implementation
//! 
//! This module provides the main NHITSModel struct and its implementation
//! for neural hierarchical interpolation time series forecasting.

use ndarray::{Array1, Array2, Array3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Core NHITS model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NHITSConfig {
    pub input_size: usize,
    pub output_size: usize,
    pub num_stacks: usize,
    pub num_blocks: usize,
    pub hidden_size: usize,
    pub num_basis: usize,
    pub forecast_horizon: usize,
    pub learning_rate: f32,
    pub dropout_rate: f32,
    pub pooling_factor: usize,
}

impl Default for NHITSConfig {
    fn default() -> Self {
        Self {
            input_size: 128,
            output_size: 24,
            num_stacks: 3,
            num_blocks: 2,
            hidden_size: 256,
            num_basis: 10,
            forecast_horizon: 24,
            learning_rate: 0.001,
            dropout_rate: 0.1,
            pooling_factor: 2,
        }
    }
}

/// Main NHITS model struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NHITSModel {
    pub config: NHITSConfig,
    pub stacks: Vec<StackBlock>,
    pub is_trained: bool,
    pub training_history: TrainingHistory,
    pub weights: HashMap<String, Array2<f32>>,
}

/// Stack block in NHITS architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackBlock {
    pub id: usize,
    pub blocks: Vec<Block>,
    pub pooling_factor: usize,
    pub basis_expansion: BasisExpansion,
}

/// Individual block within a stack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub id: usize,
    pub hidden_size: usize,
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
    pub activation: ActivationType,
}

/// Basis expansion for hierarchical decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasisExpansion {
    pub num_basis: usize,
    pub coefficients: Array2<f32>,
    pub basis_functions: Vec<BasisFunction>,
}

/// Basis function types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BasisFunction {
    Polynomial { degree: usize },
    Fourier { frequency: f32 },
    Wavelet { scale: f32 },
}

/// Activation function types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationType {
    ReLU,
    GELU,
    Tanh,
    Sigmoid,
    Swish,
}

/// Training history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHistory {
    pub losses: Vec<f32>,
    pub val_losses: Vec<f32>,
    pub epochs: usize,
    pub best_epoch: usize,
    pub metrics: HashMap<String, Vec<f32>>,
}

impl Default for TrainingHistory {
    fn default() -> Self {
        Self {
            losses: Vec::new(),
            val_losses: Vec::new(),
            epochs: 0,
            best_epoch: 0,
            metrics: HashMap::new(),
        }
    }
}

impl NHITSModel {
    /// Create a new NHITS model with given configuration
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        num_stacks: usize,
        num_blocks: usize,
        forecast_horizon: usize,
    ) -> Self {
        let config = NHITSConfig {
            input_size: input_dim,
            hidden_size: hidden_dim,
            num_stacks,
            num_blocks,
            forecast_horizon,
            output_size: forecast_horizon,
            ..Default::default()
        };
        
        Self::from_config(config)
    }
    
    /// Create model from configuration
    pub fn from_config(config: NHITSConfig) -> Self {
        let stacks = (0..config.num_stacks)
            .map(|i| Self::create_stack(i, &config))
            .collect();
        
        Self {
            config,
            stacks,
            is_trained: false,
            training_history: TrainingHistory::default(),
            weights: HashMap::new(),
        }
    }
    
    /// Create a stack with blocks
    fn create_stack(stack_id: usize, config: &NHITSConfig) -> StackBlock {
        let blocks = (0..config.num_blocks)
            .map(|i| Self::create_block(i, config))
            .collect();
        
        let basis_expansion = BasisExpansion {
            num_basis: config.num_basis,
            coefficients: Array2::zeros((config.num_basis, config.output_size)),
            basis_functions: (0..config.num_basis)
                .map(|i| match i % 3 {
                    0 => BasisFunction::Polynomial { degree: i + 1 },
                    1 => BasisFunction::Fourier { frequency: (i + 1) as f32 },
                    _ => BasisFunction::Wavelet { scale: (i + 1) as f32 },
                })
                .collect(),
        };
        
        StackBlock {
            id: stack_id,
            blocks,
            pooling_factor: config.pooling_factor.saturating_pow(stack_id as u32 + 1),
            basis_expansion,
        }
    }
    
    /// Create an individual block
    fn create_block(block_id: usize, config: &NHITSConfig) -> Block {
        let weights = Array2::zeros((config.hidden_size, config.hidden_size));
        let bias = Array1::zeros(config.hidden_size);
        
        Block {
            id: block_id,
            hidden_size: config.hidden_size,
            weights,
            bias,
            activation: ActivationType::GELU,
        }
    }
    
    /// Forward pass through the model
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let batch_size = input.shape()[0];
        let mut output = Array2::zeros((batch_size, self.config.output_size));
        
        // Process through each stack
        for stack in &self.stacks {
            let stack_output = self.forward_stack(stack, input);
            output = output + stack_output;
        }
        
        output
    }
    
    /// Forward pass through a single stack
    fn forward_stack(&self, stack: &StackBlock, input: &Array2<f32>) -> Array2<f32> {
        let mut hidden = input.clone();
        
        // Pass through blocks in the stack
        for block in &stack.blocks {
            hidden = self.forward_block(block, &hidden);
        }
        
        // Apply basis expansion
        self.apply_basis_expansion(&stack.basis_expansion, &hidden)
    }
    
    /// Forward pass through a single block
    fn forward_block(&self, block: &Block, input: &Array2<f32>) -> Array2<f32> {
        // Simplified linear transformation + activation
        let mut output = input.clone();
        
        // Apply activation function
        match block.activation {
            ActivationType::ReLU => {
                output.mapv_inplace(|x| x.max(0.0));
            },
            ActivationType::GELU => {
                output.mapv_inplace(|x| x * 0.5 * (1.0 + (x * std::f32::consts::FRAC_2_SQRT_PI * 0.5).tanh()));
            },
            ActivationType::Tanh => {
                output.mapv_inplace(|x| x.tanh());
            },
            ActivationType::Sigmoid => {
                output.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
            },
            ActivationType::Swish => {
                output.mapv_inplace(|x| x / (1.0 + (-x).exp()));
            },
        }
        
        output
    }
    
    /// Apply basis expansion for hierarchical decomposition
    fn apply_basis_expansion(&self, basis: &BasisExpansion, input: &Array2<f32>) -> Array2<f32> {
        let batch_size = input.shape()[0];
        let mut output = Array2::zeros((batch_size, self.config.output_size));
        
        // Simplified basis expansion - in practice this would be more complex
        for i in 0..batch_size {
            for j in 0..self.config.output_size {
                output[[i, j]] = input[[i, j.min(input.shape()[1] - 1)]];
            }
        }
        
        output
    }
    
    /// Train the model (simplified implementation)
    pub fn train(&mut self, x_train: &Array3<f32>, y_train: &Array2<f32>, epochs: usize) {
        println!("Training NHITS model for {} epochs...", epochs);
        
        for epoch in 0..epochs {
            let loss = self.compute_loss(x_train, y_train);
            self.training_history.losses.push(loss);
            self.training_history.epochs = epoch + 1;
            
            if epoch % 10 == 0 {
                println!("Epoch {}: Loss = {:.6}", epoch, loss);
            }
        }
        
        self.is_trained = true;
        println!("Training completed!");
    }
    
    /// Compute training loss
    fn compute_loss(&self, x_train: &Array3<f32>, y_train: &Array2<f32>) -> f32 {
        // Convert 3D input to 2D for forward pass
        let batch_size = x_train.shape()[0];
        let seq_len = x_train.shape()[1];
        let features = x_train.shape()[2];
        
        let x_2d = x_train.clone().into_shape((batch_size, seq_len * features)).unwrap();
        let predictions = self.forward(&x_2d);
        
        // Mean squared error
        let diff = &predictions - y_train;
        diff.mapv(|x| x * x).mean().unwrap_or(1.0)
    }
    
    /// Predict using the trained model
    pub fn predict(&self, input: &Array2<f32>) -> Array2<f32> {
        if !self.is_trained {
            println!("Warning: Model not trained yet, predictions may be unreliable");
        }
        
        self.forward(input)
    }
    
    /// Save model state
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let serialized = serde_json::to_string_pretty(self)?;
        std::fs::write(path, serialized)?;
        Ok(())
    }
    
    /// Load model state
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        let model: Self = serde_json::from_str(&data)?;
        Ok(model)
    }
    
    /// Serialize model for persistence
    pub fn serialize(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        Ok(serde_json::to_vec(self)?)
    }
    
    /// Deserialize model from bytes
    pub fn deserialize(data: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(serde_json::from_slice(data)?)
    }
    
    /// Get model summary
    pub fn summary(&self) -> String {
        format!(
            "NHITS Model Summary:\n\
             - Input size: {}\n\
             - Output size: {}\n\
             - Stacks: {}\n\
             - Blocks per stack: {}\n\
             - Hidden size: {}\n\
             - Trained: {}\n\
             - Training epochs: {}",
            self.config.input_size,
            self.config.output_size,
            self.config.num_stacks,
            self.config.num_blocks,
            self.config.hidden_size,
            self.is_trained,
            self.training_history.epochs
        )
    }
}

impl Default for NHITSModel {
    fn default() -> Self {
        Self::from_config(NHITSConfig::default())
    }
}

// Trait for NHITS model interface
pub trait NHITSModelTrait {
    fn forward(&self, input: &Array2<f32>) -> Array2<f32>;
    fn train(&mut self, x_train: &Array3<f32>, y_train: &Array2<f32>, epochs: usize);
    fn predict(&self, input: &Array2<f32>) -> Array2<f32>;
    fn is_trained(&self) -> bool;
    fn get_config(&self) -> &NHITSConfig;
}

impl NHITSModelTrait for NHITSModel {
    fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        self.forward(input)
    }
    
    fn train(&mut self, x_train: &Array3<f32>, y_train: &Array2<f32>, epochs: usize) {
        self.train(x_train, y_train, epochs)
    }
    
    fn predict(&self, input: &Array2<f32>) -> Array2<f32> {
        self.predict(input)
    }
    
    fn is_trained(&self) -> bool {
        self.is_trained
    }
    
    fn get_config(&self) -> &NHITSConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_creation() {
        let model = NHITSModel::new(128, 256, 3, 2, 24);
        
        assert_eq!(model.config.input_size, 128);
        assert_eq!(model.config.hidden_size, 256);
        assert_eq!(model.config.num_stacks, 3);
        assert_eq!(model.config.num_blocks, 2);
        assert_eq!(model.config.forecast_horizon, 24);
        assert!(!model.is_trained);
    }
    
    #[test]
    fn test_forward_pass() {
        let model = NHITSModel::default();
        let input = Array2::zeros((1, model.config.input_size));
        let output = model.forward(&input);
        
        assert_eq!(output.shape()[0], 1);
        assert_eq!(output.shape()[1], model.config.output_size);
    }
    
    #[test]
    fn test_model_serialization() {
        let model = NHITSModel::default();
        let serialized = model.serialize().unwrap();
        let deserialized = NHITSModel::deserialize(&serialized).unwrap();
        
        assert_eq!(model.config.input_size, deserialized.config.input_size);
        assert_eq!(model.config.output_size, deserialized.config.output_size);
    }
}