//! Neuron type implementations for cerebellar layers
//! 
//! Pure Rust implementations of LIF and AdEx neurons optimized for trading systems.

use candle_core::{Tensor, Device, DType, Result as CandleResult};
use candle_nn as nn;
use nalgebra::{DVector, DMatrix};
use anyhow::{Result, anyhow};
use tracing::{debug, warn};
use serde::{Serialize, Deserialize};

use crate::{LIFNeuron, LayerType};
use crate::compatibility::{TensorCompat, NeuralNetCompat, DTypeCompat, DeviceCompat, ErrorHandling};

/// LIF neuron parameters optimized for trading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LIFParameters {
    /// Membrane time constant (ms)
    pub tau_mem: f64,
    /// Synaptic time constant (ms)
    pub tau_syn: f64,
    /// Threshold voltage (mV)
    pub v_th: f64,
    /// Reset voltage (mV)
    pub v_reset: f64,
    /// Leak voltage (mV)
    pub v_leak: f64,
    /// Refractory period (ms)
    pub refractory_period: f64,
}

impl Default for LIFParameters {
    fn default() -> Self {
        Self {
            tau_mem: 10.0,
            tau_syn: 5.0,
            v_th: 1.0,
            v_reset: 0.0,
            v_leak: 0.0,
            refractory_period: 2.0,
        }
    }
}

/// AdEx neuron parameters for adaptive behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdExParameters {
    /// Membrane time constant (ms)
    pub tau_mem: f64,
    /// Synaptic time constant (ms)
    pub tau_syn: f64,
    /// Threshold voltage (mV)
    pub v_th: f64,
    /// Reset voltage (mV)
    pub v_reset: f64,
    /// Leak voltage (mV)
    pub v_leak: f64,
    /// Adaptation time constant (ms)
    pub tau_adapt: f64,
    /// Adaptation increment (nA)
    pub alpha: f64,
    /// Spike slope factor (mV)
    pub delta_th: f64,
}

impl Default for AdExParameters {
    fn default() -> Self {
        Self {
            tau_mem: 10.0,
            tau_syn: 5.0,
            v_th: 1.0,
            v_reset: 0.0,
            v_leak: 0.0,
            tau_adapt: 100.0,
            alpha: 0.1,
            delta_th: 0.1,
        }
    }
}

/// Neuron state trait for different neuron types
pub trait NeuronState: std::fmt::Debug {
    /// Get membrane potential
    fn get_membrane_potential(&self) -> &Tensor;
    /// Get synaptic current
    fn get_synaptic_current(&self) -> &Tensor;
    /// Update state with input
    fn update(&mut self, input: &Tensor, dt: f64) -> CandleResult<Tensor>;
    /// Reset state
    fn reset(&mut self) -> CandleResult<()>;
}

/// LIF neuron state implementation
#[derive(Debug)]
pub struct LIFState {
    pub v_mem: Tensor,
    pub i_syn: Tensor,
    pub refractory: Tensor,
    pub params: LIFParameters,
}

impl LIFState {
    pub fn new(size: usize, params: LIFParameters, device: Device) -> CandleResult<Self> {
        let v_mem = Tensor::zeros(&[size], DType::F32, &device)?;
        let i_syn = Tensor::zeros(&[size], DType::F32, &device)?;
        let refractory = Tensor::zeros(&[size], DType::F32, &device)?;
        
        Ok(Self {
            v_mem,
            i_syn,
            refractory,
            params,
        })
    }
}

impl NeuronState for LIFState {
    fn get_membrane_potential(&self) -> &Tensor {
        &self.v_mem
    }
    
    fn get_synaptic_current(&self) -> &Tensor {
        &self.i_syn
    }
    
    fn update(&mut self, input: &Tensor, dt: f64) -> CandleResult<Tensor> {
        // Update synaptic current
        let syn_decay = (-dt / self.params.tau_syn).exp() as f32;
        self.i_syn = (self.i_syn.mul_scalar(syn_decay)? + input)?;
        
        // Update membrane potential
        let mem_decay = (-dt / self.params.tau_mem).exp() as f32;
        self.v_mem = (self.v_mem.mul_scalar(mem_decay)? + &self.i_syn)?;
        
        // Check for spikes
        let threshold = self.params.v_th as f32;
        let spikes = self.v_mem.gt(threshold)?;
        
        // Reset spiked neurons
        let reset_mask = spikes.to_dtype(DType::F32)?;
        let reset_value = self.params.v_reset as f32;
        self.v_mem = self.v_mem.mul(&reset_mask.neg()?.add_scalar(1.0)?)?.add_scalar(reset_value)?;
        
        Ok(spikes)
    }
    
    fn reset(&mut self) -> CandleResult<()> {
        let device = self.v_mem.device().clone();
        let shape_dims = self.v_mem.shape().dims().to_vec();
        
        self.v_mem = Tensor::zeros(&shape_dims, DType::F32, &device)?;
        self.i_syn = Tensor::zeros(&shape_dims, DType::F32, &device)?;
        self.refractory = Tensor::zeros(&shape_dims, DType::F32, &device)?;
        
        Ok(())
    }
}

/// AdEx neuron state implementation
#[derive(Debug)]
pub struct AdExState {
    pub v_mem: Tensor,
    pub i_syn: Tensor,
    pub adaptation: Tensor,
    pub refractory: Tensor,
    pub params: AdExParameters,
}

impl AdExState {
    pub fn new(size: usize, params: AdExParameters, device: Device) -> CandleResult<Self> {
        let v_mem = Tensor::zeros(&[size], DType::F32, &device)?;
        let i_syn = Tensor::zeros(&[size], DType::F32, &device)?;
        let adaptation = Tensor::zeros(&[size], DType::F32, &device)?;
        let refractory = Tensor::zeros(&[size], DType::F32, &device)?;
        
        Ok(Self {
            v_mem,
            i_syn,
            adaptation,
            refractory,
            params,
        })
    }
}

impl NeuronState for AdExState {
    fn get_membrane_potential(&self) -> &Tensor {
        &self.v_mem
    }
    
    fn get_synaptic_current(&self) -> &Tensor {
        &self.i_syn
    }
    
    fn update(&mut self, input: &Tensor, dt: f64) -> CandleResult<Tensor> {
        // AdEx differential equations implementation
        // dv/dt = (-(v - E_L) + delta_T * exp((v - v_T)/delta_T) - w + I) / tau_m
        // dw/dt = (a * (v - E_L) - w) / tau_w
        
        let dt_f32 = dt as f32;
        
        // Update synaptic current with exponential decay
        let syn_decay = (-dt / self.params.tau_syn).exp() as f32;
        self.i_syn = (self.i_syn.mul_scalar(syn_decay)? + input)?;
        
        // Update adaptation current with subthreshold and spike-triggered components
        let adapt_decay = (-dt / self.params.tau_adapt).exp() as f32;
        
        // Subthreshold adaptation: a * (v - E_L)
        let v_leak_tensor = Tensor::full(&self.v_mem.shape(), self.params.v_leak as f32, self.v_mem.device())?;
        let subthreshold_adapt = (&self.v_mem - &v_leak_tensor)?.mul_scalar(self.params.alpha as f32)?;
        
        // Apply adaptation dynamics: dw/dt = (a*(v-E_L) - w) / tau_w
        let adapt_increment = &subthreshold_adapt.sub(&self.adaptation)?.mul_scalar(dt_f32 / self.params.tau_adapt as f32)?;
        self.adaptation = (&self.adaptation + &adapt_increment)?;
        
        // Exponential spike mechanism: delta_T * exp((v - v_T) / delta_T)
        let v_th_tensor = Tensor::full(&self.v_mem.shape(), self.params.v_th as f32, self.v_mem.device())?;
        let spike_term = (&self.v_mem - &v_th_tensor)?.div_scalar(self.params.delta_th as f32)?;
        
        // Safely compute exponential (clamp to prevent overflow)
        let spike_term_clamped = spike_term.clamp(-10.0, 2.0)?; // Prevent numerical overflow
        let exp_spike = spike_term_clamped.exp()?.mul_scalar(self.params.delta_th as f32)?;
        
        // Membrane potential dynamics: dv/dt = (-(v-E_L) + delta_T*exp(...) - w + I) / tau_m
        let leak_current = (&v_leak_tensor - &self.v_mem)?; // -(v - E_L)
        let total_current = (&leak_current + &exp_spike)?.sub(&self.adaptation)?.add(&self.i_syn)?;
        let dv_dt = total_current.div_scalar(self.params.tau_mem as f32)?;
        
        // Update membrane potential
        self.v_mem = (&self.v_mem + &dv_dt.mul_scalar(dt_f32)?)?;
        
        // Spike detection with adaptive threshold
        let effective_threshold = self.params.v_th as f32 + 0.1; // Slight margin for numerical stability
        let spikes = self.v_mem.gt(effective_threshold)?;
        
        // Apply spike-triggered reset and adaptation
        let reset_mask = spikes.to_dtype(DType::F32)?;
        let inv_reset_mask = reset_mask.neg()?.add_scalar(1.0)?;
        
        // Reset membrane potential for spiked neurons
        let reset_value = self.params.v_reset as f32;
        self.v_mem = self.v_mem.mul(&inv_reset_mask)?.add(&reset_mask.mul_scalar(reset_value)?)?;
        
        // Spike-triggered adaptation increment
        let spike_adapt_increment = self.params.alpha as f32 * 5.0; // Stronger spike-triggered component
        self.adaptation = (&self.adaptation + &reset_mask.mul_scalar(spike_adapt_increment)?)?;
        
        // Handle refractory period by clamping voltage
        let refractory_clamp = Tensor::full(&self.v_mem.shape(), self.params.v_reset as f32, self.v_mem.device())?;
        self.refractory = self.refractory.mul_scalar(0.9)?; // Decay refractory
        let refractory_active = self.refractory.gt(0.1)?;
        let refractory_mask = refractory_active.to_dtype(DType::F32)?;
        let normal_mask = refractory_mask.neg()?.add_scalar(1.0)?;
        
        self.v_mem = self.v_mem.mul(&normal_mask)?.add(&refractory_clamp.mul(&refractory_mask)?)?;
        
        // Set refractory period for spiked neurons
        let new_refractory = Tensor::full(&self.refractory.shape(), 2.0, self.refractory.device())?; // 2ms refractory
        self.refractory = self.refractory.mul(&inv_reset_mask)?.add(&new_refractory.mul(&reset_mask)?)?;
        
        Ok(spikes)
    }
    
    fn reset(&mut self) -> CandleResult<()> {
        let device = self.v_mem.device().clone();
        let shape_dims = self.v_mem.shape().dims().to_vec();
        
        self.v_mem = Tensor::zeros(&shape_dims, DType::F32, &device)?;
        self.i_syn = Tensor::zeros(&shape_dims, DType::F32, &device)?;
        self.adaptation = Tensor::zeros(&shape_dims, DType::F32, &device)?;
        self.refractory = Tensor::zeros(&shape_dims, DType::F32, &device)?;
        
        Ok(())
    }
}

/// LIF neuron cell implementation
#[derive(Debug)]
pub struct LIFCell {
    pub params: LIFParameters,
    pub size: usize,
    pub device: Device,
}

impl LIFCell {
    pub fn new(params: LIFParameters, size: usize, device: Device) -> Self {
        Self {
            params,
            size,
            device,
        }
    }
    
    pub fn create_state(&self) -> CandleResult<LIFState> {
        LIFState::new(self.size, self.params.clone(), self.device.clone())
    }
}

/// AdEx neuron cell implementation
#[derive(Debug)]
pub struct AdExCell {
    pub params: AdExParameters,
    pub size: usize,
    pub device: Device,
}

impl AdExCell {
    pub fn new(params: AdExParameters, size: usize, device: Device) -> Self {
        Self {
            params,
            size,
            device,
        }
    }
    
    pub fn create_state(&self) -> CandleResult<AdExState> {
        AdExState::new(self.size, self.params.clone(), self.device.clone())
    }
}

/// Factory for creating neuron cells
pub struct NeuronCellFactory;

impl NeuronCellFactory {
    /// Create LIF cell with configuration
    pub fn create_lif_cell(config: &LayerConfig, _dt: f64, device: Device) -> LIFCell {
        let mut params = LIFParameters::default();
        
        // Optimize parameters based on layer type
        match config.layer_type {
            LayerType::GranuleCell => {
                params.tau_mem = 8.0;
                params.tau_syn = 3.0;
                params.v_th = 0.8;
            }
            LayerType::PurkinjeCell => {
                params.tau_mem = 12.0;
                params.tau_syn = 5.0;
                params.v_th = 1.2;
            }
            LayerType::GolgiCell => {
                params.tau_mem = 15.0;
                params.tau_syn = 8.0;
                params.v_th = 1.0;
            }
            LayerType::DeepCerebellarNucleus => {
                params.tau_mem = 10.0;
                params.tau_syn = 4.0;
                params.v_th = 1.5;
            }
        }
        
        LIFCell::new(params, config.size, device)
    }
    
    /// Create AdEx cell with configuration
    pub fn create_adex_cell(config: &LayerConfig, _dt: f64, device: Device) -> AdExCell {
        let mut params = AdExParameters::default();
        
        // Optimize parameters based on layer type
        match config.layer_type {
            LayerType::GranuleCell => {
                params.tau_mem = 8.0;
                params.tau_syn = 3.0;
                params.v_th = 0.8;
                params.tau_adapt = 50.0;
                params.alpha = 0.05;
            }
            LayerType::PurkinjeCell => {
                params.tau_mem = 12.0;
                params.tau_syn = 5.0;
                params.v_th = 1.2;
                params.tau_adapt = 100.0;
                params.alpha = 0.1;
            }
            LayerType::GolgiCell => {
                params.tau_mem = 15.0;
                params.tau_syn = 8.0;
                params.v_th = 1.0;
                params.tau_adapt = 80.0;
                params.alpha = 0.08;
            }
            LayerType::DeepCerebellarNucleus => {
                params.tau_mem = 10.0;
                params.tau_syn = 4.0;
                params.v_th = 1.5;
                params.tau_adapt = 120.0;
                params.alpha = 0.12;
            }
        }
        
        AdExCell::new(params, config.size, device)
    }
}

/// Factory for creating optimized neurons
pub struct NeuronFactory;

impl NeuronFactory {
    /// Create LIF neuron optimized for granule cells
    pub fn create_granule_cell() -> LIFNeuron {
        let mut neuron = LIFNeuron::new_trading_optimized();
        neuron.decay_mem = 0.95;  // Slower decay for integration
        neuron.decay_syn = 0.7;   // Faster synaptic decay
        neuron.threshold = 0.8;   // Lower threshold for sensitivity
        neuron
    }

    /// Create LIF neuron optimized for Purkinje cells
    pub fn create_purkinje_cell() -> LIFNeuron {
        let mut neuron = LIFNeuron::new_trading_optimized();
        neuron.decay_mem = 0.9;   // Standard decay
        neuron.decay_syn = 0.8;   // Standard synaptic decay
        neuron.threshold = 1.2;   // Higher threshold for selectivity
        neuron
    }

    /// Create LIF neuron optimized for Golgi cells (inhibitory)
    pub fn create_golgi_cell() -> LIFNeuron {
        let mut neuron = LIFNeuron::new_trading_optimized();
        neuron.decay_mem = 0.85;  // Faster decay for inhibition
        neuron.decay_syn = 0.9;   // Slower synaptic for sustained inhibition
        neuron.threshold = 1.0;   // Standard threshold
        neuron
    }

    /// Create LIF neuron optimized for DCN (output)
    pub fn create_dcn_cell() -> LIFNeuron {
        let mut neuron = LIFNeuron::new_trading_optimized();
        neuron.decay_mem = 0.92;  // Balanced decay
        neuron.decay_syn = 0.85;  // Moderate synaptic decay
        neuron.threshold = 1.5;   // High threshold for output certainty
        neuron
    }
}

/// Batch neuron processing for performance
pub struct BatchNeuronProcessor {
    neurons: Vec<LIFNeuron>,
    batch_size: usize,
}

impl BatchNeuronProcessor {
    /// Create new batch processor
    pub fn new(neurons: Vec<LIFNeuron>) -> Self {
        let batch_size = neurons.len();
        Self {
            neurons,
            batch_size,
        }
    }

    /// Process batch of neurons with SIMD optimization
    pub fn process_batch(&mut self, inputs: &[f32]) -> Vec<bool> {
        self.neurons.iter_mut()
            .zip(inputs.iter())
            .map(|(neuron, &input)| neuron.step(input))
            .collect()
    }

    /// Get batch statistics
    pub fn get_batch_stats(&self) -> BatchStats {
        let active_count = self.neurons.iter()
            .filter(|n| n.v_mem > 0.1)
            .count();
        
        let avg_membrane = self.neurons.iter()
            .map(|n| n.v_mem)
            .sum::<f32>() / self.batch_size as f32;
            
        BatchStats {
            total_neurons: self.batch_size,
            active_neurons: active_count,
            average_membrane_potential: avg_membrane,
        }
    }
}

/// Batch processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStats {
    pub total_neurons: usize,
    pub active_neurons: usize,
    pub average_membrane_potential: f32,
}

/// Layer configuration for neuron creation
#[derive(Debug, Clone)]
pub struct LayerConfig {
    pub size: usize,
    pub layer_type: LayerType,
    pub neuron_type: NeuronType,
    pub tau_mem: f64,
    pub tau_syn_exc: f64,
    pub tau_syn_inh: f64,
    pub tau_adapt: Option<f64>,
    pub a: Option<f64>,
    pub b: Option<f64>,
}

/// Neuron type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeuronType {
    LIF,
    AdEx,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_neuron_factory() {
        let granule = NeuronFactory::create_granule_cell();
        let purkinje = NeuronFactory::create_purkinje_cell();
        let golgi = NeuronFactory::create_golgi_cell();
        let dcn = NeuronFactory::create_dcn_cell();
        
        // Test different thresholds
        assert!(granule.threshold < purkinje.threshold);
        assert!(purkinje.threshold < dcn.threshold);
    }

    #[test]
    fn test_batch_processor() {
        let neurons = vec![LIFNeuron::new_trading_optimized(); 100];
        let mut processor = BatchNeuronProcessor::new(neurons);
        
        let inputs = vec![1.0; 100];
        let spikes = processor.process_batch(&inputs);
        
        assert_eq!(spikes.len(), 100);
        
        let stats = processor.get_batch_stats();
        assert_eq!(stats.total_neurons, 100);
    }
    
    #[test]
    fn test_lif_state() {
        let device = Device::Cpu;
        let params = LIFParameters::default();
        let mut state = LIFState::new(10, params, device).unwrap();
        
        let input = Tensor::ones(&[10], DType::F32, &device).unwrap();
        let spikes = state.update(&input, 0.1).unwrap();
        
        assert_eq!(spikes.shape(), &[10]);
    }
    
    #[test]
    fn test_adex_state() {
        let device = Device::Cpu;
        let params = AdExParameters::default();
        let mut state = AdExState::new(10, params, device).unwrap();
        
        let input = Tensor::ones(&[10], DType::F32, &device).unwrap();
        let spikes = state.update(&input, 0.1).unwrap();
        
        assert_eq!(spikes.shape(), &[10]);
    }
    
    #[test]
    fn test_neuron_cell_factory() {
        let device = Device::Cpu;
        let config = LayerConfig {
            size: 100,
            layer_type: LayerType::GranuleCell,
            neuron_type: NeuronType::LIF,
        };
        
        let lif_cell = NeuronCellFactory::create_lif_cell(&config, 0.1, device.clone());
        let adex_cell = NeuronCellFactory::create_adex_cell(&config, 0.1, device);
        
        assert_eq!(lif_cell.size, 100);
        assert_eq!(adex_cell.size, 100);
    }
}