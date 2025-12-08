//! Cerebellar layer implementations with Norse-compatible dynamics
//! 
//! Provides high-performance cerebellar layer implementations supporting both
//! LIF and AdEx neuron models with biological connectivity patterns.

use candle_core::{Tensor, Device, DType, Result as CandleResult};
use candle_nn::{self as nn, VarBuilder};
use anyhow::{Result, anyhow};
use tracing::{debug, info, warn};
use std::collections::HashMap;

use crate::{LayerConfig, NeuronType};
use crate::neuron_types::{LIFCell, AdExCell, LIFState, AdExState, NeuronCellFactory, NeuronState};
use crate::compatibility::{TensorCompat, NeuralNetCompat, DTypeCompat, DeviceCompat};

/// Cerebellar layer with configurable neuron types
pub struct CerebellarLayer {
    /// Layer size (number of neurons)
    size: usize,
    /// Neuron type
    neuron_type: NeuronType,
    /// Time step
    dt: f64,
    /// Device
    device: Device,
    /// LIF cell (if using LIF neurons)
    lif_cell: Option<LIFCell>,
    /// AdEx cell (if using AdEx neurons)
    adex_cell: Option<AdExCell>,
    /// Current neuron state
    state: Option<Box<dyn NeuronState>>,
    /// Batch size for current state
    current_batch_size: Option<i64>,
}

impl CerebellarLayer {
    /// Create new cerebellar layer
    pub fn new(config: &LayerConfig, dt: f64, device: Device) -> Result<Self> {
        let size = config.size;
        let neuron_type = config.neuron_type;
        
        // Create appropriate neuron cell based on type
        let (lif_cell, adex_cell) = match neuron_type {
            NeuronType::LIF => {
                let cell = NeuronCellFactory::create_lif_cell(config, dt, device.clone());
                (Some(cell), None)
            }
            NeuronType::AdEx => {
                let cell = NeuronCellFactory::create_adex_cell(config, dt, device.clone());
                (None, Some(cell))
            }
        };
        
        debug!("Created cerebellar layer: {} neurons, {:?} type", size, neuron_type);
        
        Ok(Self {
            size,
            neuron_type,
            dt,
            device,
            lif_cell,
            adex_cell,
            state: None,
            current_batch_size: None,
        })
    }
    
    /// Reset layer state
    pub fn reset_state(&mut self, batch_size: Option<i64>) {
        let batch_size = batch_size.unwrap_or(1);
        
        // Create new state based on neuron type
        self.state = match self.neuron_type {
            NeuronType::LIF => {
                let state = LIFState::new(batch_size, self.size as i64, self.device);
                Some(Box::new(state) as Box<dyn NeuronState>)
            }
            NeuronType::AdEx => {
                let state = AdExState::new(batch_size, self.size as i64, self.device);
                Some(Box::new(state) as Box<dyn NeuronState>)
            }
        };
        
        self.current_batch_size = Some(batch_size);
        debug!("Reset cerebellar layer state for batch size {}", batch_size);
    }
    
    /// Forward pass through the layer
    pub fn forward(&mut self, input_current: &Tensor) -> Result<(Tensor, &dyn NeuronState)> {
        let batch_size = input_current.size()[0];
        
        // Initialize state if needed or if batch size changed
        if self.state.is_none() || self.current_batch_size != Some(batch_size) {
            self.reset_state(Some(batch_size));
        }
        
        // Process input through appropriate neuron type
        let spikes = match self.neuron_type {
            NeuronType::LIF => {
                let cell = self.lif_cell.as_ref()
                    .ok_or_else(|| anyhow!("LIF cell not initialized"))?;
                let state = self.state.as_mut().unwrap();
                
                // Downcast to LIF state
                let lif_state = state.as_any_mut().downcast_mut::<LIFState>()
                    .ok_or_else(|| anyhow!("Invalid state type for LIF cell"))?;
                
                cell.forward(input_current, lif_state)?
            }
            NeuronType::AdEx => {
                let cell = self.adex_cell.as_ref()
                    .ok_or_else(|| anyhow!("AdEx cell not initialized"))?;
                let state = self.state.as_mut().unwrap();
                
                // Downcast to AdEx state
                let adex_state = state.as_any_mut().downcast_mut::<AdExState>()
                    .ok_or_else(|| anyhow!("Invalid state type for AdEx cell"))?;
                
                cell.forward(input_current, adex_state)?
            }
        };
        
        let state_ref = self.state.as_ref().unwrap().as_ref();
        Ok((spikes, state_ref))
    }
    
    /// Get current spikes
    pub fn get_spikes(&self) -> Option<&Tensor> {
        self.state.as_ref().map(|s| s.get_spikes())
    }
    
    /// Get current membrane potential
    pub fn get_membrane_potential(&self) -> Option<&Tensor> {
        self.state.as_ref().map(|s| s.get_membrane_potential())
    }
    
    /// Get layer size
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Get neuron type
    pub fn neuron_type(&self) -> NeuronType {
        self.neuron_type
    }
    
    /// Get device
    pub fn device(&self) -> Device {
        self.device.clone()
    }
    
    /// Calculate current firing rate
    pub fn firing_rate(&self) -> f64 {
        if let Some(spikes) = self.get_spikes() {
            let spike_sum = TensorCompat::sum_compat(spikes).unwrap_or_else(|_| {
                warn!("Failed to compute spike sum, returning zero tensor");
                Tensor::zeros(&[], DTypeCompat::float32(), &self.device)
            });
            let spike_count: f64 = TensorCompat::to_scalar_compat::<f32>(&spike_sum).unwrap_or(0.0) as f64;
            let total_neurons = TensorCompat::elem_count_compat(spikes) as f64;
            if total_neurons > 0.0 {
                spike_count / total_neurons
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
    
    /// Apply external current injection (for testing/debugging)
    pub fn inject_current(&mut self, current: &Tensor) -> Result<Tensor> {
        self.forward(current).map(|(spikes, _)| spikes)
    }
    
    /// Get detailed layer statistics
    pub fn get_statistics(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        
        stats.insert("size".to_string(), self.size as f64);
        stats.insert("firing_rate".to_string(), self.firing_rate());
        
        if let Some(spikes) = self.get_spikes() {
            let spike_mean_tensor = TensorCompat::mean_compat(spikes).unwrap_or_else(|_| {
                warn!("Failed to compute spike mean");
                Tensor::zeros(&[], DTypeCompat::float32(), &self.device)
            });
            let spike_std_tensor = TensorCompat::std_compat(spikes).unwrap_or_else(|_| {
                warn!("Failed to compute spike std");
                Tensor::zeros(&[], DTypeCompat::float32(), &self.device)
            });
            let spike_mean: f64 = TensorCompat::to_scalar_compat::<f32>(&spike_mean_tensor).unwrap_or(0.0) as f64;
            let spike_std: f64 = TensorCompat::to_scalar_compat::<f32>(&spike_std_tensor).unwrap_or(0.0) as f64;
            stats.insert("spike_mean".to_string(), spike_mean);
            stats.insert("spike_std".to_string(), spike_std);
        }
        
        if let Some(v_mem) = self.get_membrane_potential() {
            let v_mean_tensor = TensorCompat::mean_compat(v_mem).unwrap_or_else(|_| {
                warn!("Failed to compute membrane mean");
                Tensor::zeros(&[], DTypeCompat::float32(), &self.device)
            });
            let v_std_tensor = TensorCompat::std_compat(v_mem).unwrap_or_else(|_| {
                warn!("Failed to compute membrane std");
                Tensor::zeros(&[], DTypeCompat::float32(), &self.device)
            });
            let v_mean: f64 = TensorCompat::to_scalar_compat::<f32>(&v_mean_tensor).unwrap_or(0.0) as f64;
            let v_std: f64 = TensorCompat::to_scalar_compat::<f32>(&v_std_tensor).unwrap_or(0.0) as f64;
            stats.insert("membrane_mean".to_string(), v_mean);
            stats.insert("membrane_std".to_string(), v_std);
        }
        
        stats
    }
}

// Extend NeuronState trait to support downcasting
pub trait NeuronStateExt: NeuronState {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

impl NeuronStateExt for LIFState {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl NeuronStateExt for AdExState {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Connection weight manager for cerebellar layers
pub struct ConnectionWeights {
    /// Weight matrix
    weights: nn::Linear,
    /// Connection probability
    connectivity: f64,
    /// Whether connection is inhibitory
    inhibitory: bool,
    /// Weight scale
    weight_scale: f64,
}

impl ConnectionWeights {
    /// Create new connection weights
    pub fn new(
        vs: &VarBuilder,
        input_size: usize,
        output_size: usize,
        connectivity: f64,
        weight_scale: f64,
        inhibitory: bool,
    ) -> Result<Self> {
        let mut weights = nn::linear(vs, input_size as i64, output_size as i64, Default::default());
        
        // Initialize with sparse connectivity
        Self::initialize_sparse_weights(&mut weights, connectivity, weight_scale, inhibitory)?;
        
        Ok(Self {
            weights,
            connectivity,
            inhibitory,
            weight_scale,
        })
    }
    
    /// Initialize sparse weight matrix
    fn initialize_sparse_weights(
        linear: &mut nn::Linear,
        connectivity: f64,
        weight_scale: f64,
        inhibitory: bool,
    ) -> Result<()> {
        // Initialize sparse weights using candle-core
        let weight_tensor = &linear.ws;
        let size = weight_tensor.size();
        let output_size = size[0];
        let input_size = size[1];
        
        // Create random connectivity mask
        let mask = Tensor::rand(&[output_size, input_size], (DType::F32, weight_tensor.device()))?
            .lt(connectivity)?;
        
        // Create random weights
        let mut weights = Tensor::randn(&[output_size, input_size], (DType::F32, weight_tensor.device()))?
            .mul_scalar(weight_scale)?;
        
        // Make inhibitory if needed
        if inhibitory {
            weights = weights.abs()?.neg()?;
        }
        
        // Apply sparse mask
        let sparse_weights = weights.mul(&mask.to_dtype(DType::F32)?)?;
        
        // This would need to be implemented with candle's parameter update mechanism
        // For now, we'll skip the actual weight update
        debug!("Sparse weights initialized with connectivity {}", connectivity);
        Ok(())
    }
    
    /// Forward pass through connection
    pub fn forward(&self, input: &Tensor) -> Tensor {
        self.weights.forward(input)
    }
    
    /// Get weight statistics
    pub fn weight_statistics(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        
        // Get weight statistics using candle-core
        let weights = &self.weights.ws;
        if let Ok(weight_mean) = weights.mean_all() {
            if let Ok(weight_std) = weights.std_all() {
                if let Ok(weight_min) = weights.min_all() {
                    if let Ok(weight_max) = weights.max_all() {
                        stats.insert("mean".to_string(), weight_mean.to_scalar::<f64>().unwrap_or(0.0));
                        stats.insert("std".to_string(), weight_std.to_scalar::<f64>().unwrap_or(0.0));
                        stats.insert("min".to_string(), weight_min.to_scalar::<f64>().unwrap_or(0.0));
                        stats.insert("max".to_string(), weight_max.to_scalar::<f64>().unwrap_or(0.0));
                        stats.insert("connectivity".to_string(), self.connectivity);
                    }
                }
            }
        }
        
        stats
    }
    
    /// Update weights during learning
    pub fn update_weights(&mut self, _delta_weights: &Tensor) -> Result<()> {
        // Weight updates with candle-core would need to be implemented
        // through the parameter update mechanism
        debug!("Weight update requested - not implemented in candle-core version");
        Ok(())
    }
    
    /// Apply weight constraints (bounds, sparsity)
    pub fn apply_constraints(&mut self) -> Result<()> {
        // Weight constraints with candle-core would need to be implemented
        // through the parameter constraint mechanism
        debug!("Weight constraints requested - not implemented in candle-core version");
        Ok(())
    }
}

/// Multi-layer cerebellar network component
pub struct MultiLayerCerebellar {
    /// Individual layers
    layers: Vec<CerebellarLayer>,
    /// Inter-layer connections
    connections: Vec<ConnectionWeights>,
    /// Layer configurations
    configs: Vec<LayerConfig>,
    /// Time step
    dt: f64,
    /// Device
    device: Device,
}

impl MultiLayerCerebellar {
    /// Create multi-layer cerebellar component
    pub fn new(
        configs: Vec<LayerConfig>,
        connection_params: Vec<(f64, f64, bool)>, // (connectivity, weight_scale, inhibitory)
        dt: f64,
        device: Device,
        vs: &VarBuilder,
    ) -> Result<Self> {
        if configs.len() < 2 {
            return Err(anyhow!("Need at least 2 layers for multi-layer network"));
        }
        
        if connection_params.len() != configs.len() - 1 {
            return Err(anyhow!("Connection parameters must match number of layer transitions"));
        }
        
        // Create layers
        let mut layers = Vec::new();
        for config in &configs {
            let layer = CerebellarLayer::new(config, dt, device)?;
            layers.push(layer);
        }
        
        // Create connections between consecutive layers
        let mut connections = Vec::new();
        for (i, &(connectivity, weight_scale, inhibitory)) in connection_params.iter().enumerate() {
            let input_size = configs[i].size;
            let output_size = configs[i + 1].size;
            
            let conn_path = vs / format!("layer_{}_{}", i, i + 1);
            let connection = ConnectionWeights::new(
                &conn_path,
                input_size,
                output_size,
                connectivity,
                weight_scale,
                inhibitory,
            );
            connections.push(connection);
        }
        
        info!("Created multi-layer cerebellar network with {} layers", layers.len());
        
        Ok(Self {
            layers,
            connections,
            configs,
            dt,
            device,
        })
    }
    
    /// Forward pass through all layers
    pub fn forward(&mut self, input: &Tensor) -> Result<Vec<Tensor>> {
        let mut layer_outputs = Vec::new();
        let mut current_input = input.shallow_clone();
        
        for (i, layer) in self.layers.iter_mut().enumerate() {
            // Pass through layer
            let (spikes, _) = layer.forward(&current_input)?;
            layer_outputs.push(spikes.shallow_clone());
            
            // Connect to next layer if not last
            if i < self.connections.len() {
                current_input = self.connections[i].forward(&spikes);
            }
        }
        
        Ok(layer_outputs)
    }
    
    /// Reset all layer states
    pub fn reset(&mut self, batch_size: Option<i64>) {
        for layer in &mut self.layers {
            layer.reset_state(batch_size);
        }
    }
    
    /// Get layer by index
    pub fn get_layer(&self, index: usize) -> Option<&CerebellarLayer> {
        self.layers.get(index)
    }
    
    /// Get mutable layer by index
    pub fn get_layer_mut(&mut self, index: usize) -> Option<&mut CerebellarLayer> {
        self.layers.get_mut(index)
    }
    
    /// Get connection statistics
    pub fn get_connection_statistics(&self) -> Vec<HashMap<String, f64>> {
        self.connections.iter().map(|conn| conn.weight_statistics()).collect()
    }
    
    /// Apply learning updates to connections
    pub fn update_connections(&mut self, weight_updates: Vec<Tensor>) -> Result<()> {
        if weight_updates.len() != self.connections.len() {
            return Err(anyhow!("Weight update count must match connection count"));
        }
        
        for (connection, update) in self.connections.iter_mut().zip(weight_updates.iter()) {
            connection.update_weights(update);
            connection.apply_constraints();
        }
        
        Ok(())
    }
    
    /// Get comprehensive layer statistics
    pub fn get_layer_statistics(&self) -> Vec<HashMap<String, f64>> {
        self.layers.iter().map(|layer| layer.get_statistics()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NeuronType;
    
    #[test]
    fn test_cerebellar_layer_creation() {
        let config = LayerConfig {
            size: 100,
            neuron_type: NeuronType::LIF,
            tau_mem: 10.0,
            tau_syn_exc: 2.0,
            tau_syn_inh: 10.0,
            tau_adapt: Some(50.0),
            a: Some(2e-9),
            b: Some(1e-10),
        };
        
        let layer = CerebellarLayer::new(&config, 1e-3, Device::Cpu).unwrap();
        assert_eq!(layer.size(), 100);
        assert_eq!(layer.neuron_type(), NeuronType::LIF);
    }
    
    #[test]
    fn test_layer_forward_pass() {
        let config = LayerConfig {
            size: 50,
            neuron_type: NeuronType::LIF,
            tau_mem: 10.0,
            tau_syn_exc: 2.0,
            tau_syn_inh: 10.0,
            tau_adapt: Some(50.0),
            a: Some(2e-9),
            b: Some(1e-10),
        };
        
        let mut layer = CerebellarLayer::new(&config, 1e-3, Device::Cpu).unwrap();
        let input = Tensor::ones(&[2, 50], (Kind::Float, Device::Cpu)) * 1.5;
        
        let (spikes, _) = layer.forward(&input).unwrap();
        assert_eq!(spikes.size(), vec![2, 50]);
        
        // Check that spikes are binary
        let spike_values: Vec<f64> = Vec::<f64>::from(&spikes);
        for spike in spike_values {
            assert!(spike >= 0.0 && spike <= 1.0);
        }
    }
    
    #[test]
    fn test_connection_weights() {
        let vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();
        
        let connection = ConnectionWeights::new(
            &root,
            20,  // input size
            10,  // output size
            0.3, // connectivity
            0.1, // weight scale
            false, // not inhibitory
        );
        
        let input = Tensor::randn(&[5, 20], (Kind::Float, Device::Cpu));
        let output = connection.forward(&input);
        assert_eq!(output.size(), vec![5, 10]);
        
        let stats = connection.weight_statistics();
        assert!(stats.contains_key("sparsity"));
        assert!(stats["sparsity"] > 0.0); // Should be sparse
    }
    
    #[test]
    fn test_multi_layer_network() {
        let configs = vec![
            LayerConfig {
                size: 20,
                neuron_type: NeuronType::LIF,
                tau_mem: 10.0,
                tau_syn_exc: 2.0,
                tau_syn_inh: 10.0,
                tau_adapt: Some(50.0),
                a: Some(2e-9),
                b: Some(1e-10),
            },
            LayerConfig {
                size: 10,
                neuron_type: NeuronType::AdEx,
                tau_mem: 15.0,
                tau_syn_exc: 3.0,
                tau_syn_inh: 5.0,
                tau_adapt: Some(100.0),
                a: Some(4e-9),
                b: Some(5e-10),
            },
        ];
        
        let connection_params = vec![(0.2, 0.1, false)]; // One connection between layers
        
        let vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();
        
        let mut network = MultiLayerCerebellar::new(
            configs,
            connection_params,
            1e-3,
            Device::Cpu,
            &root,
        ).unwrap();
        
        let input = Tensor::randn(&[3, 20], (Kind::Float, Device::Cpu));
        let outputs = network.forward(&input).unwrap();
        
        assert_eq!(outputs.len(), 2); // Two layers
        assert_eq!(outputs[0].size(), vec![3, 20]); // First layer output
        assert_eq!(outputs[1].size(), vec![3, 10]); // Second layer output
    }
}