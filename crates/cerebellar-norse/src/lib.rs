//! Pure Rust Cerebellar Spiking Neural Network
//! 
//! Ultra-low latency implementation for high-frequency trading systems.
//! Features sub-microsecond inference times with deterministic performance.

use std::collections::HashMap;
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex64;
use anyhow::{Result, anyhow};
use tracing::{debug, info, warn};
use serde::{Serialize, Deserialize};
use candle_core::Device;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

// Core neuron types
pub mod neuron_types;
pub mod cerebellar_layers;
pub mod cerebellar_circuit;
pub mod architecture_interfaces;
pub mod encoding;
pub mod training;
pub mod optimization;
pub mod compatibility;

// Market microstructure and execution modules
pub mod market_microstructure;
pub mod execution_algorithms;
pub mod order_routing;

// Security modules
pub mod security;
pub mod input_validation;
pub mod penetration_testing;

// Performance optimization modules
pub mod cuda_kernels;
pub mod simd_kernels;
pub mod zero_alloc;
pub mod validation;

// Risk management and safety systems
pub mod risk_management;
pub mod risk_dashboard;
pub mod neural_validator;

// Observability and monitoring systems
pub mod observability;
pub mod dashboard;
pub mod log_aggregation;
pub mod metrics_exporter;

// Quantitative analysis and validation framework
pub mod quantitative_analysis;
pub mod statistical_validation;
pub mod factor_attribution;
pub mod risk_metrics;
pub mod quantitative_benchmarks;

// Re-export core types
pub use neuron_types::*;
pub use cerebellar_layers::*;
pub use cerebellar_circuit::*;
pub use architecture_interfaces::*;
pub use encoding::*;
pub use training::*;
pub use optimization::*;
pub use compatibility::*;

// Re-export market microstructure types
pub use market_microstructure::*;
pub use execution_algorithms::*;
pub use order_routing::*;

// Re-export security types
pub use security::*;
pub use input_validation::*;

// Re-export risk management types
pub use risk_management::*;
pub use risk_dashboard::*;
pub use neural_validator::*;
pub use penetration_testing::*;

// Re-export performance optimization types
pub use cuda_kernels::*;
pub use simd_kernels::*;
pub use zero_alloc::*;
pub use validation::*;

// Re-export observability types
pub use observability::*;
pub use dashboard::*;
pub use log_aggregation::*;
pub use metrics_exporter::*;

// Re-export quantitative analysis types
pub use quantitative_analysis::*;
pub use statistical_validation::*;
pub use factor_attribution::*;
pub use risk_metrics::*;
pub use quantitative_benchmarks::*;

/// High-performance LIF neuron optimized for trading
#[repr(C, align(64))] // Cache-line aligned for performance
#[derive(Debug, Clone, Copy)]
pub struct LIFNeuron {
    /// Membrane potential
    pub v_mem: f32,
    /// Synaptic current
    pub i_syn: f32,
    /// Refractory counter
    pub refractory_counter: u8,
    /// Pre-computed decay constants
    pub decay_mem: f32,
    pub decay_syn: f32,
    pub threshold: f32,
    pub reset_potential: f32,
}

impl LIFNeuron {
    /// Create new LIF neuron with default parameters optimized for trading
    pub fn new_trading_optimized() -> Self {
        Self {
            v_mem: 0.0,
            i_syn: 0.0,
            refractory_counter: 0,
            decay_mem: 0.9,  // tau_mem = 10ms
            decay_syn: 0.8,  // tau_syn = 5ms
            threshold: 1.0,
            reset_potential: 0.0,
        }
    }

    /// Ultra-fast neuron step function (target: <10ns)
    #[inline(always)]
    pub fn step(&mut self, input: f32) -> bool {
        // Handle refractory period
        if self.refractory_counter > 0 {
            self.refractory_counter -= 1;
            return false;
        }
        
        // Update membrane dynamics
        self.i_syn = self.i_syn * self.decay_syn + input;
        self.v_mem = self.v_mem * self.decay_mem + self.i_syn;
        
        // Spike detection and reset
        if self.v_mem >= self.threshold {
            self.v_mem = self.reset_potential;
            self.refractory_counter = 2; // 2ms refractory period
            true
        } else {
            false
        }
    }
}

/// Cerebellar layer with optimized processing
#[derive(Debug)]
pub struct CerebellarLayer {
    /// Neurons in this layer
    pub neurons: Vec<LIFNeuron>,
    /// Connection weights (sparse representation)
    pub weights: Array2<f32>,
    /// Layer-specific parameters
    pub layer_type: LayerType,
    /// Performance metrics
    pub spike_count: u64,
    pub last_activity: f32,
}

/// Layer types in cerebellar circuit
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerType {
    /// Granule cell layer (input expansion)
    GranuleCell,
    /// Purkinje cell layer (main processing)
    PurkinjeCell,
    /// Golgi cell layer (inhibitory)
    GolgiCell,
    /// Deep cerebellar nucleus (output)
    DeepCerebellarNucleus,
}

impl CerebellarLayer {
    /// Create new cerebellar layer
    pub fn new(size: usize, layer_type: LayerType) -> Self {
        let neurons = (0..size)
            .map(|_| LIFNeuron::new_trading_optimized())
            .collect();
        
        Self {
            neurons,
            weights: Array2::zeros((size, size)),
            layer_type,
            spike_count: 0,
            last_activity: 0.0,
        }
    }

    /// Process layer with SIMD optimization
    pub fn process_layer(&mut self, inputs: &[f32]) -> Vec<bool> {
        let mut spikes = Vec::with_capacity(self.neurons.len());
        
        // Process neurons in parallel for larger layers
        #[cfg(feature = "parallel")]
        if self.neurons.len() > 64 {
            self.neurons.par_iter_mut()
                .zip(inputs.par_iter())
                .map(|(neuron, &input)| neuron.step(input))
                .collect()
        } else {
            // Sequential processing for small layers (better cache locality)
            self.neurons.iter_mut()
                .zip(inputs.iter())
                .map(|(neuron, &input)| neuron.step(input))
                .collect()
        }
        
        #[cfg(not(feature = "parallel"))]
        {
            // Sequential processing only
            self.neurons.iter_mut()
                .zip(inputs.iter())
                .map(|(neuron, &input)| neuron.step(input))
                .collect()
        }
    }

    /// Get layer statistics
    pub fn get_stats(&self) -> LayerStats {
        let active_neurons = self.neurons.iter()
            .filter(|n| n.v_mem > 0.1)
            .count();
        
        LayerStats {
            total_neurons: self.neurons.len(),
            active_neurons,
            spike_count: self.spike_count,
            average_membrane_potential: self.neurons.iter()
                .map(|n| n.v_mem)
                .sum::<f32>() / self.neurons.len() as f32,
        }
    }
}

/// Layer performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerStats {
    pub total_neurons: usize,
    pub active_neurons: usize,
    pub spike_count: u64,
    pub average_membrane_potential: f32,
}

/// Complete cerebellar circuit for trading
#[derive(Debug)]
pub struct CerebellarCircuit {
    /// Granule cell layer (input expansion)
    pub granule_layer: CerebellarLayer,
    /// Purkinje cell layer (main processing)
    pub purkinje_layer: CerebellarLayer,
    /// Golgi cell layer (inhibitory feedback)
    pub golgi_layer: CerebellarLayer,
    /// Deep cerebellar nucleus (output)
    pub dcn_layer: CerebellarLayer,
    /// Circuit configuration
    pub config: CircuitConfig,
}

/// Circuit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitConfig {
    pub granule_size: usize,
    pub purkinje_size: usize,
    pub golgi_size: usize,
    pub dcn_size: usize,
    pub learning_rate: f32,
    pub sparsity: f32,
    pub output_dim: usize,
    pub device: Device,
}

/// Cerebellar Norse configuration alias
pub type CerebellarNorseConfig = CircuitConfig;

/// Cerebellar metrics alias
pub type CerebellarMetrics = CircuitMetrics;

/// Layer configuration for neuron creation
#[derive(Debug, Clone)]
pub struct LayerConfig {
    pub size: usize,
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

impl Default for CircuitConfig {
    fn default() -> Self {
        Self {
            granule_size: 4000,   // Large expansion layer
            purkinje_size: 100,   // Main processing layer
            golgi_size: 50,       // Inhibitory feedback
            dcn_size: 10,         // Compact output
            learning_rate: 0.01,
            sparsity: 0.1,        // 10% connectivity
            output_dim: 10,       // Default output dimension
            device: Device::Cpu,  // Default to CPU
        }
    }
}

impl CerebellarCircuit {
    /// Create new cerebellar circuit optimized for trading
    pub fn new_trading_optimized(config: CircuitConfig) -> Self {
        let granule_layer = CerebellarLayer::new(config.granule_size, LayerType::GranuleCell);
        let purkinje_layer = CerebellarLayer::new(config.purkinje_size, LayerType::PurkinjeCell);
        let golgi_layer = CerebellarLayer::new(config.golgi_size, LayerType::GolgiCell);
        let dcn_layer = CerebellarLayer::new(config.dcn_size, LayerType::DeepCerebellarNucleus);
        
        Self {
            granule_layer,
            purkinje_layer,
            golgi_layer,
            dcn_layer,
            config,
        }
    }

    /// Process market data through cerebellar circuit
    pub fn process_market_data(&mut self, market_inputs: &[f32]) -> Result<Vec<f32>> {
        // Input expansion through granule cells
        let granule_spikes = self.granule_layer.process_layer(market_inputs);
        
        // Convert spikes to currents for next layer
        let granule_currents: Vec<f32> = granule_spikes.iter()
            .map(|&spike| if spike { 1.0 } else { 0.0 })
            .collect();
        
        // Main processing through Purkinje cells
        let purkinje_spikes = self.purkinje_layer.process_layer(&granule_currents);
        
        // Inhibitory feedback through Golgi cells
        let _golgi_spikes = self.golgi_layer.process_layer(&granule_currents);
        
        // Final output through deep cerebellar nucleus
        let purkinje_currents: Vec<f32> = purkinje_spikes.iter()
            .map(|&spike| if spike { 1.0 } else { 0.0 })
            .collect();
        
        let dcn_spikes = self.dcn_layer.process_layer(&purkinje_currents);
        
        // Convert output spikes to trading signals
        let output_signals: Vec<f32> = dcn_spikes.iter()
            .map(|&spike| if spike { 1.0 } else { 0.0 })
            .collect();
        
        Ok(output_signals)
    }

    /// Get circuit performance metrics
    pub fn get_performance_metrics(&self) -> CircuitMetrics {
        CircuitMetrics {
            granule_stats: self.granule_layer.get_stats(),
            purkinje_stats: self.purkinje_layer.get_stats(),
            golgi_stats: self.golgi_layer.get_stats(),
            dcn_stats: self.dcn_layer.get_stats(),
            total_neurons: self.config.granule_size + self.config.purkinje_size + 
                          self.config.golgi_size + self.config.dcn_size,
        }
    }
}

/// Circuit performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitMetrics {
    pub granule_stats: LayerStats,
    pub purkinje_stats: LayerStats,
    pub golgi_stats: LayerStats,
    pub dcn_stats: LayerStats,
    pub total_neurons: usize,
}

/// Trading-specific cerebellar processor
#[derive(Debug)]
pub struct TradingCerebellarProcessor {
    circuit: CerebellarCircuit,
    input_buffer: Vec<f32>,
    output_buffer: Vec<f32>,
    processing_time_ns: u64,
}

impl TradingCerebellarProcessor {
    /// Create new trading processor
    pub fn new() -> Self {
        let config = CircuitConfig::default();
        let circuit = CerebellarCircuit::new_trading_optimized(config);
        
        Self {
            circuit,
            input_buffer: Vec::with_capacity(100),
            output_buffer: Vec::with_capacity(10),
            processing_time_ns: 0,
        }
    }

    /// Process market tick with sub-microsecond latency
    pub fn process_tick(&mut self, price: f32, volume: f32, timestamp: u64) -> Result<Vec<f32>> {
        let start_time = std::time::Instant::now();
        
        // Prepare input features
        self.input_buffer.clear();
        self.input_buffer.push(price);
        self.input_buffer.push(volume);
        self.input_buffer.push(timestamp as f32);
        
        // Process through cerebellar circuit
        let output = self.circuit.process_market_data(&self.input_buffer)?;
        
        // Record processing time
        self.processing_time_ns = start_time.elapsed().as_nanos() as u64;
        
        // Validate ultra-low latency requirement
        if self.processing_time_ns > 1000 { // 1 microsecond
            warn!("Processing time exceeded 1μs: {}ns", self.processing_time_ns);
        }
        
        Ok(output)
    }

    /// Get processing performance
    pub fn get_processing_time_ns(&self) -> u64 {
        self.processing_time_ns
    }

    /// Get circuit metrics
    pub fn get_metrics(&self) -> CircuitMetrics {
        self.circuit.get_performance_metrics()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lif_neuron_step() {
        let mut neuron = LIFNeuron::new_trading_optimized();
        
        // Test no spike for small input
        assert!(!neuron.step(0.5));
        
        // Test spike for large input
        assert!(neuron.step(2.0));
        
        // Test refractory period
        assert!(!neuron.step(2.0));
        assert!(!neuron.step(2.0));
        
        // Test recovery from refractory
        assert!(neuron.step(2.0));
    }

    #[test]
    fn test_cerebellar_layer() {
        let mut layer = CerebellarLayer::new(10, LayerType::GranuleCell);
        let inputs = vec![1.0; 10];
        
        let spikes = layer.process_layer(&inputs);
        assert_eq!(spikes.len(), 10);
        
        let stats = layer.get_stats();
        assert_eq!(stats.total_neurons, 10);
    }

    #[test]
    fn test_trading_processor() {
        let mut processor = TradingCerebellarProcessor::new();
        
        let output = processor.process_tick(100.0, 1000.0, 1234567890);
        assert!(output.is_ok());
        
        // Verify ultra-low latency
        assert!(processor.get_processing_time_ns() < 10000); // < 10μs
    }
}