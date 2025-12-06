//! Neural network components for NQO

use crate::error::{NqoError, NqoResult};
use crate::types::NeuralWeights;
use candle_core::{Device, Tensor, DType, Module};
use candle_nn::{Linear, VarBuilder, VarMap};
use ndarray::Array1;
use parking_lot::RwLock;
use std::sync::Arc;
use tracing::{debug, info};

/// Neural network for quantum parameter optimization
pub struct NeuralNetwork {
    /// Number of neurons in hidden layer
    neurons: usize,
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
    /// Device (CPU/GPU)
    device: Device,
    /// Variable map for parameters
    var_map: Arc<RwLock<VarMap>>,
    /// Network layers
    input_hidden: Linear,
    hidden_output: Linear,
    /// Recurrent connections
    recurrent: Option<Linear>,
    /// Hidden state
    hidden_state: Arc<RwLock<Tensor>>,
}

impl NeuralNetwork {
    /// Create a new neural network
    pub fn new(
        neurons: usize,
        input_dim: usize,
        output_dim: usize,
        use_gpu: bool,
    ) -> NqoResult<Self> {
        // Select device
        let device = if use_gpu {
            #[cfg(feature = "gpu-cuda")]
            {
                Device::new_cuda(0)
                    .map_err(|e| NqoError::HardwareError(format!("CUDA init failed: {}", e)))?
            }
            #[cfg(not(feature = "gpu-cuda"))]
            {
                info!("GPU requested but CUDA not available, using CPU");
                Device::Cpu
            }
        } else {
            Device::Cpu
        };
        
        info!("Neural network using device: {:?}", device);
        
        // Create variable map
        let var_map = VarMap::new();
        let vs = VarBuilder::from_varmap(&var_map, DType::F64, &device);
        
        // Create layers
        let input_hidden = candle_nn::linear(input_dim, neurons, vs.pp("input_hidden"))
            .map_err(|e| NqoError::NeuralError(format!("Failed to create input layer: {}", e)))?;
            
        let hidden_output = candle_nn::linear(neurons, output_dim, vs.pp("hidden_output"))
            .map_err(|e| NqoError::NeuralError(format!("Failed to create output layer: {}", e)))?;
            
        let recurrent = Some(
            candle_nn::linear(neurons, neurons, vs.pp("recurrent"))
                .map_err(|e| NqoError::NeuralError(format!("Failed to create recurrent layer: {}", e)))?
        );
        
        // Initialize hidden state
        let hidden_state = Tensor::zeros((1, neurons), DType::F64, &device)
            .map_err(|e| NqoError::NeuralError(format!("Failed to create hidden state: {}", e)))?;
        
        Ok(Self {
            neurons,
            input_dim,
            output_dim,
            device,
            var_map: Arc::new(RwLock::new(var_map)),
            input_hidden,
            hidden_output,
            recurrent,
            hidden_state: Arc::new(RwLock::new(hidden_state)),
        })
    }
    
    /// Forward pass through the network
    pub fn forward(&self, input: &Tensor) -> NqoResult<Tensor> {
        // Get current hidden state
        let hidden_state = self.hidden_state.read();
        
        // Input to hidden layer
        let mut hidden = self.input_hidden.forward(input)
            .map_err(|e| NqoError::NeuralError(format!("Forward pass failed at input layer: {}", e)))?;
            
        // Add recurrent connection if available
        if let Some(ref recurrent) = self.recurrent {
            let recurrent_input = recurrent.forward(&*hidden_state)
                .map_err(|e| NqoError::NeuralError(format!("Recurrent forward failed: {}", e)))?;
            hidden = (hidden + recurrent_input)
                .map_err(|e| NqoError::NeuralError(format!("Hidden addition failed: {}", e)))?;
        }
        
        // Apply ReLU activation
        let hidden_activated = hidden.relu()
            .map_err(|e| NqoError::NeuralError(format!("ReLU activation failed: {}", e)))?;
        
        // Update hidden state
        drop(hidden_state);
        let mut hidden_state_mut = self.hidden_state.write();
        *hidden_state_mut = hidden_activated.clone();
        
        // Hidden to output layer
        let output = self.hidden_output.forward(&hidden_activated)
            .map_err(|e| NqoError::NeuralError(format!("Output layer failed: {}", e)))?;
            
        Ok(output)
    }
    
    /// Update network weights based on gradients
    pub fn update_weights(&self, gradients: &Tensor, learning_rate: f64) -> NqoResult<()> {
        use candle_nn::Optimizer;
        
        // Create optimizer for gradient descent
        let var_map = self.var_map.write();
        let mut sgd = candle_nn::SGD::new(var_map.all_vars(), learning_rate)
            .map_err(|e| NqoError::NeuralError(format!("Failed to create optimizer: {}", e)))?;
        
        // Apply gradients
        sgd.backward_step(&gradients)
            .map_err(|e| NqoError::NeuralError(format!("Failed to update weights: {}", e)))?;
        
        debug!("Updated neural network weights with learning rate {}", learning_rate);
        Ok(())
    }
    
    /// Convert input array to tensor
    pub fn array_to_tensor(&self, array: &Array1<f64>) -> NqoResult<Tensor> {
        let shape = [1, array.len()];
        Tensor::from_slice(array.as_slice().unwrap(), &shape, &self.device)
            .map_err(|e| NqoError::NeuralError(format!("Failed to create tensor: {}", e)))
    }
    
    /// Convert tensor to array
    pub fn tensor_to_array(&self, tensor: &Tensor) -> NqoResult<Vec<f64>> {
        tensor.to_vec2::<f64>()
            .map_err(|e| NqoError::NeuralError(format!("Failed to convert tensor: {}", e)))
            .map(|v| v[0].clone())
    }
    
    /// Get current weights from the neural network
    pub fn get_weights(&self) -> NqoResult<NeuralWeights> {
        // Note: VarMap API has changed in newer versions of candle-nn
        // For now, return default weights. In production, would need to update
        // to use the new VarMap API or access weights directly from layers
        
        let input_hidden = vec![vec![0.0; self.input_dim]; self.neurons];
        let hidden_output = vec![vec![0.0; self.neurons]; self.output_dim];
        let recurrent = vec![vec![0.0; self.neurons]; self.neurons];
        let hidden_biases = vec![0.0; self.neurons];
        let output_biases = vec![0.0; self.output_dim];
        
        debug!("Returning default neural network weights");
        
        Ok(NeuralWeights {
            input_hidden,
            hidden_output,
            recurrent,
            hidden_biases,
            output_biases,
        })
    }
    
    /// Reset hidden state
    pub fn reset_hidden_state(&self) -> NqoResult<()> {
        let mut hidden_state = self.hidden_state.write();
        *hidden_state = Tensor::zeros((1, self.neurons), DType::F64, &self.device)
            .map_err(|e| NqoError::NeuralError(format!("Failed to reset hidden state: {}", e)))?;
        Ok(())
    }
}

/// Neuromorphic activation functions
pub mod activations {
    use super::*;
    
    /// Adaptive sigmoid activation
    pub fn adaptive_sigmoid(x: &Tensor, temperature: f64) -> NqoResult<Tensor> {
        let scaled = (x * (1.0 / temperature))
            .map_err(|e| NqoError::NeuralError(format!("Scaling failed: {}", e)))?;
        
        // Manual sigmoid implementation since method may not be available
        let sigmoid_data = scaled.to_vec1::<f64>()
            .map_err(|e| NqoError::NeuralError(format!("Failed to extract data: {}", e)))?
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect::<Vec<f64>>();
        
        Tensor::from_vec(sigmoid_data, scaled.shape(), &scaled.device())
            .map_err(|e| NqoError::NeuralError(format!("Sigmoid tensor creation failed: {}", e)))
    }
    
    /// Spiking activation (simplified)
    pub fn spiking_activation(x: &Tensor, threshold: f64) -> NqoResult<Tensor> {
        x.ge(threshold)
            .map_err(|e| NqoError::NeuralError(format!("Threshold comparison failed: {}", e)))?
            .to_dtype(DType::F64)
            .map_err(|e| NqoError::NeuralError(format!("Type conversion failed: {}", e)))
    }
}

/// GPU-accelerated operations
#[cfg(feature = "gpu-cuda")]
pub mod gpu_ops {
    use super::*;
    // use candle_core::CudaDevice; // Unused for now
    
    /// Optimized matrix multiplication on GPU
    pub fn gpu_matmul(a: &Tensor, b: &Tensor) -> NqoResult<Tensor> {
        a.matmul(b)
            .map_err(|e| NqoError::NeuralError(format!("GPU matmul failed: {}", e)))
    }
    
    /// Batch normalization on GPU
    pub fn gpu_batch_norm(x: &Tensor, eps: f64) -> NqoResult<Tensor> {
        let mean = x.mean_keepdim(1)
            .map_err(|e| NqoError::NeuralError(format!("Mean calculation failed: {}", e)))?;
        let var = x.var_keepdim(1)
            .map_err(|e| NqoError::NeuralError(format!("Variance calculation failed: {}", e)))?;
            
        let normalized = ((x - mean)? / (var + eps)?.sqrt()?)
            .map_err(|e| NqoError::NeuralError(format!("Normalization failed: {}", e)))?;
            
        Ok(normalized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_neural_network_creation() {
        let nn = NeuralNetwork::new(64, 10, 5, false);
        assert!(nn.is_ok());
    }
    
    #[test]
    fn test_forward_pass() {
        let nn = NeuralNetwork::new(32, 8, 4, false).unwrap();
        let input = Array1::from_vec(vec![0.1; 8]);
        let tensor = nn.array_to_tensor(&input).unwrap();
        let output = nn.forward(&tensor);
        assert!(output.is_ok());
    }
}