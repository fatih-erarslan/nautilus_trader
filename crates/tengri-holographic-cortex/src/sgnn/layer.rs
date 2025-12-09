//! # SGNN Layer Module
//!
//! Implements layers of spiking neurons with:
//! - Parallel neuron updates
//! - Spike event routing
//! - Multi-scale temporal dynamics
//!
//! ## Multi-Scale Architecture
//!
//! The multi-scale SGNN implements three temporal scales inspired by
//! biological cortical hierarchy:
//!
//! - **Fast Layer** (5ms): Sensory processing, rapid responses
//! - **Medium Layer** (20ms): Integration, pattern recognition
//! - **Slow Layer** (100ms): Context, long-term dependencies
//!
//! ## Wolfram-Verified Dynamics
//!
//! ```wolfram
//! (* Multi-scale integration *)
//! fastResponse[t_] := Exp[-t/5];    (* τ=5ms *)
//! medResponse[t_] := Exp[-t/20];    (* τ=20ms *)
//! slowResponse[t_] := Exp[-t/100];  (* τ=100ms *)
//!
//! (* Verified decay rates at t=10ms: *)
//! N[Exp[-10/5]] = 0.1353      (* fast *)
//! N[Exp[-10/20]] = 0.6065     (* medium *)
//! N[Exp[-10/100]] = 0.9048    (* slow *)
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::sgnn::{LIFNeuron, LIFConfig, SpikeEvent, Synapse, SynapseConfig};

/// Configuration for an SGNN layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    /// Number of neurons in the layer
    pub num_neurons: usize,

    /// Layer identifier
    pub layer_id: u8,

    /// LIF neuron configuration
    pub neuron_config: LIFConfig,

    /// Default synapse configuration
    pub synapse_config: SynapseConfig,
}

impl Default for LayerConfig {
    fn default() -> Self {
        Self {
            num_neurons: 100,
            layer_id: 0,
            neuron_config: LIFConfig::default(),
            synapse_config: SynapseConfig::default(),
        }
    }
}

impl LayerConfig {
    /// Create a new layer configuration
    pub fn new(num_neurons: usize, layer_id: u8, neuron_config: LIFConfig) -> Self {
        Self {
            num_neurons,
            layer_id,
            neuron_config,
            synapse_config: SynapseConfig::default(),
        }
    }

    /// Create fast layer configuration
    pub fn fast(num_neurons: usize, layer_id: u8) -> Self {
        Self::new(num_neurons, layer_id, LIFConfig::fast())
    }

    /// Create medium layer configuration
    pub fn medium(num_neurons: usize, layer_id: u8) -> Self {
        Self::new(num_neurons, layer_id, LIFConfig::medium())
    }

    /// Create slow layer configuration
    pub fn slow(num_neurons: usize, layer_id: u8) -> Self {
        Self::new(num_neurons, layer_id, LIFConfig::slow())
    }
}

/// SGNN Layer: Collection of LIF neurons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SGNNLayer {
    /// Layer configuration
    config: LayerConfig,

    /// Neurons in the layer
    neurons: Vec<LIFNeuron>,

    /// Synapses (target_id -> source synapses)
    synapses: HashMap<u32, Vec<Synapse>>,

    /// Current simulation time (μs)
    current_time: u64,
}

impl SGNNLayer {
    /// Create a new SGNN layer
    pub fn new(config: LayerConfig) -> Self {
        let neurons: Vec<LIFNeuron> = (0..config.num_neurons)
            .map(|i| LIFNeuron::new(i as u32, config.layer_id, config.neuron_config.clone()))
            .collect();

        Self {
            config,
            neurons,
            synapses: HashMap::new(),
            current_time: 0,
        }
    }

    /// Add a synapse to the layer
    pub fn add_synapse(&mut self, synapse: Synapse) {
        self.synapses
            .entry(synapse.target_id)
            .or_insert_with(Vec::new)
            .push(synapse);
    }

    /// Process spike events and update all neurons
    ///
    /// # Arguments
    /// * `input_spikes` - Spike events from previous layers
    ///
    /// # Returns
    /// Vector of output spike events
    pub fn step(&mut self, input_spikes: &[SpikeEvent]) -> Vec<SpikeEvent> {
        let dt_us = (self.config.neuron_config.dt * 1000.0) as u64;
        self.current_time += dt_us;

        // Route input spikes to synapses
        for spike in input_spikes {
            if let Some(synapses) = self.synapses.get_mut(&spike.neuron_id) {
                for synapse in synapses.iter_mut() {
                    if synapse.source_id == spike.neuron_id {
                        synapse.receive_spike(spike.timestamp);
                    }
                }
            }
        }

        // Compute input currents for each neuron
        let mut input_currents = vec![0.0; self.neurons.len()];
        for (target_id, synapses) in self.synapses.iter_mut() {
            let idx = *target_id as usize;
            if idx < input_currents.len() {
                for synapse in synapses.iter_mut() {
                    input_currents[idx] += synapse.deliver_spikes(self.current_time);
                }
            }
        }

        // Update all neurons
        let mut output_spikes = Vec::new();
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            if let Some(spike) = neuron.step(input_currents[i]) {
                output_spikes.push(spike);
            }
        }

        output_spikes
    }

    /// Get neuron by index
    pub fn neuron(&self, idx: usize) -> Option<&LIFNeuron> {
        self.neurons.get(idx)
    }

    /// Get mutable neuron by index
    pub fn neuron_mut(&mut self, idx: usize) -> Option<&mut LIFNeuron> {
        self.neurons.get_mut(idx)
    }

    /// Get all neurons
    pub fn neurons(&self) -> &[LIFNeuron] {
        &self.neurons
    }

    /// Get synapses for a target neuron
    pub fn synapses_for(&self, target_id: u32) -> Option<&Vec<Synapse>> {
        self.synapses.get(&target_id)
    }

    /// Get mutable synapses for a target neuron
    pub fn synapses_for_mut(&mut self, target_id: u32) -> Option<&mut Vec<Synapse>> {
        self.synapses.get_mut(&target_id)
    }

    /// Get current time (μs)
    pub fn current_time(&self) -> u64 {
        self.current_time
    }

    /// Reset layer state
    pub fn reset(&mut self) {
        for neuron in &mut self.neurons {
            neuron.reset();
        }
        for synapses in self.synapses.values_mut() {
            for synapse in synapses {
                synapse.clear_buffer();
            }
        }
        self.current_time = 0;
    }

    /// Get layer statistics
    pub fn stats(&self) -> LayerStats {
        let total_spikes: u64 = self.neurons.iter().map(|n| n.spike_count()).sum();
        let active_neurons = self.neurons.iter().filter(|n| n.spike_count() > 0).count();
        let avg_membrane = self.neurons.iter().map(|n| n.membrane_potential()).sum::<f64>()
            / self.neurons.len() as f64;

        LayerStats {
            num_neurons: self.neurons.len(),
            total_spikes,
            active_neurons,
            avg_membrane_potential: avg_membrane,
            num_synapses: self.synapses.values().map(|v| v.len()).sum(),
        }
    }
}

/// Layer statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerStats {
    pub num_neurons: usize,
    pub total_spikes: u64,
    pub active_neurons: usize,
    pub avg_membrane_potential: f64,
    pub num_synapses: usize,
}

/// Multi-scale SGNN configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiScaleConfig {
    /// Number of neurons per layer
    pub neurons_per_layer: usize,

    /// Enable fast layer (5ms)
    pub enable_fast: bool,

    /// Enable medium layer (20ms)
    pub enable_medium: bool,

    /// Enable slow layer (100ms)
    pub enable_slow: bool,
}

impl Default for MultiScaleConfig {
    fn default() -> Self {
        Self {
            neurons_per_layer: 100,
            enable_fast: true,
            enable_medium: true,
            enable_slow: true,
        }
    }
}

/// Multi-scale SGNN with fast, medium, and slow layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiScaleSGNN {
    /// Configuration
    config: MultiScaleConfig,

    /// Fast layer (5ms time constant)
    fast_layer: Option<SGNNLayer>,

    /// Medium layer (20ms time constant)
    medium_layer: Option<SGNNLayer>,

    /// Slow layer (100ms time constant)
    slow_layer: Option<SGNNLayer>,
}

impl MultiScaleSGNN {
    /// Create a new multi-scale SGNN
    pub fn new(config: MultiScaleConfig) -> Self {
        let fast_layer = if config.enable_fast {
            Some(SGNNLayer::new(LayerConfig::fast(config.neurons_per_layer, 0)))
        } else {
            None
        };

        let medium_layer = if config.enable_medium {
            Some(SGNNLayer::new(LayerConfig::medium(config.neurons_per_layer, 1)))
        } else {
            None
        };

        let slow_layer = if config.enable_slow {
            Some(SGNNLayer::new(LayerConfig::slow(config.neurons_per_layer, 2)))
        } else {
            None
        };

        Self {
            config,
            fast_layer,
            medium_layer,
            slow_layer,
        }
    }

    /// Step all layers forward
    ///
    /// # Arguments
    /// * `input_spikes` - External input spikes
    ///
    /// # Returns
    /// Tuple of (fast_spikes, medium_spikes, slow_spikes)
    pub fn step(&mut self, input_spikes: &[SpikeEvent]) -> (Vec<SpikeEvent>, Vec<SpikeEvent>, Vec<SpikeEvent>) {
        let fast_spikes = if let Some(layer) = &mut self.fast_layer {
            layer.step(input_spikes)
        } else {
            Vec::new()
        };

        let medium_spikes = if let Some(layer) = &mut self.medium_layer {
            layer.step(input_spikes)
        } else {
            Vec::new()
        };

        let slow_spikes = if let Some(layer) = &mut self.slow_layer {
            layer.step(input_spikes)
        } else {
            Vec::new()
        };

        (fast_spikes, medium_spikes, slow_spikes)
    }

    /// Get fast layer
    pub fn fast_layer(&self) -> Option<&SGNNLayer> {
        self.fast_layer.as_ref()
    }

    /// Get medium layer
    pub fn medium_layer(&self) -> Option<&SGNNLayer> {
        self.medium_layer.as_ref()
    }

    /// Get slow layer
    pub fn slow_layer(&self) -> Option<&SGNNLayer> {
        self.slow_layer.as_ref()
    }

    /// Get mutable fast layer
    pub fn fast_layer_mut(&mut self) -> Option<&mut SGNNLayer> {
        self.fast_layer.as_mut()
    }

    /// Get mutable medium layer
    pub fn medium_layer_mut(&mut self) -> Option<&mut SGNNLayer> {
        self.medium_layer.as_mut()
    }

    /// Get mutable slow layer
    pub fn slow_layer_mut(&mut self) -> Option<&mut SGNNLayer> {
        self.slow_layer.as_mut()
    }

    /// Reset all layers
    pub fn reset(&mut self) {
        if let Some(layer) = &mut self.fast_layer {
            layer.reset();
        }
        if let Some(layer) = &mut self.medium_layer {
            layer.reset();
        }
        if let Some(layer) = &mut self.slow_layer {
            layer.reset();
        }
    }

    /// Get statistics for all layers
    pub fn stats(&self) -> MultiScaleStats {
        MultiScaleStats {
            fast: self.fast_layer.as_ref().map(|l| l.stats()),
            medium: self.medium_layer.as_ref().map(|l| l.stats()),
            slow: self.slow_layer.as_ref().map(|l| l.stats()),
        }
    }
}

/// Multi-scale statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiScaleStats {
    pub fast: Option<LayerStats>,
    pub medium: Option<LayerStats>,
    pub slow: Option<LayerStats>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_layer_config_timescales() {
        let fast = LayerConfig::fast(100, 0);
        let medium = LayerConfig::medium(100, 1);
        let slow = LayerConfig::slow(100, 2);

        assert!(fast.neuron_config.tau_membrane < medium.neuron_config.tau_membrane);
        assert!(medium.neuron_config.tau_membrane < slow.neuron_config.tau_membrane);
    }

    #[test]
    fn test_sgnn_layer_creation() {
        let config = LayerConfig::default();
        let layer = SGNNLayer::new(config);

        assert_eq!(layer.neurons().len(), 100);
        assert_eq!(layer.current_time(), 0);
    }

    #[test]
    fn test_sgnn_layer_step() {
        let config = LayerConfig::fast(10, 0);
        let mut layer = SGNNLayer::new(config);

        // No input spikes
        let spikes = layer.step(&[]);
        assert!(spikes.is_empty());

        // Time should advance
        assert!(layer.current_time() > 0);
    }

    #[test]
    fn test_sgnn_layer_with_synapses() {
        let config = LayerConfig::fast(3, 0);
        let mut layer = SGNNLayer::new(config);

        // Add excitatory synapse from neuron 0 to neuron 1
        let synapse = Synapse::excitatory(0, 1, 1.0);
        layer.add_synapse(synapse);

        // Inject spike from neuron 0
        let input_spike = SpikeEvent::new(0, 0, 0);
        layer.step(&[input_spike]);

        // Check synapses exist
        assert!(layer.synapses_for(1).is_some());
    }

    #[test]
    fn test_sgnn_layer_stats() {
        let config = LayerConfig::fast(100, 0);
        let layer = SGNNLayer::new(config);

        let stats = layer.stats();
        assert_eq!(stats.num_neurons, 100);
        assert_eq!(stats.total_spikes, 0);
        assert_eq!(stats.active_neurons, 0);
    }

    #[test]
    fn test_sgnn_layer_reset() {
        let config = LayerConfig::fast(10, 0);
        let mut layer = SGNNLayer::new(config);

        // Run simulation
        for _ in 0..10 {
            layer.step(&[]);
        }

        assert!(layer.current_time() > 0);

        // Reset
        layer.reset();
        assert_eq!(layer.current_time(), 0);

        // All neurons reset
        for neuron in layer.neurons() {
            assert_abs_diff_eq!(neuron.time_ms(), 0.0);
        }
    }

    #[test]
    fn test_multi_scale_sgnn_creation() {
        let config = MultiScaleConfig::default();
        let sgnn = MultiScaleSGNN::new(config);

        assert!(sgnn.fast_layer().is_some());
        assert!(sgnn.medium_layer().is_some());
        assert!(sgnn.slow_layer().is_some());
    }

    #[test]
    fn test_multi_scale_sgnn_selective_layers() {
        let config = MultiScaleConfig {
            neurons_per_layer: 50,
            enable_fast: true,
            enable_medium: false,
            enable_slow: true,
        };

        let sgnn = MultiScaleSGNN::new(config);
        assert!(sgnn.fast_layer().is_some());
        assert!(sgnn.medium_layer().is_none());
        assert!(sgnn.slow_layer().is_some());
    }

    #[test]
    fn test_multi_scale_sgnn_step() {
        let config = MultiScaleConfig {
            neurons_per_layer: 10,
            enable_fast: true,
            enable_medium: true,
            enable_slow: false,
        };

        let mut sgnn = MultiScaleSGNN::new(config);

        // Step with no input
        let (fast, medium, slow) = sgnn.step(&[]);
        assert!(fast.is_empty());
        assert!(medium.is_empty());
        assert!(slow.is_empty());
    }

    #[test]
    fn test_multi_scale_sgnn_stats() {
        let config = MultiScaleConfig::default();
        let sgnn = MultiScaleSGNN::new(config);

        let stats = sgnn.stats();
        assert!(stats.fast.is_some());
        assert!(stats.medium.is_some());
        assert!(stats.slow.is_some());
    }

    #[test]
    fn test_multi_scale_sgnn_reset() {
        let config = MultiScaleConfig::default();
        let mut sgnn = MultiScaleSGNN::new(config);

        // Run simulation
        for _ in 0..10 {
            sgnn.step(&[]);
        }

        // Reset
        sgnn.reset();

        // Check all layers reset
        if let Some(layer) = sgnn.fast_layer() {
            assert_eq!(layer.current_time(), 0);
        }
        if let Some(layer) = sgnn.medium_layer() {
            assert_eq!(layer.current_time(), 0);
        }
        if let Some(layer) = sgnn.slow_layer() {
            assert_eq!(layer.current_time(), 0);
        }
    }

    #[test]
    fn test_multi_scale_temporal_dynamics() {
        // Test that different layers have different time constants
        let config = MultiScaleConfig {
            neurons_per_layer: 1,
            enable_fast: true,
            enable_medium: true,
            enable_slow: true,
        };

        let sgnn = MultiScaleSGNN::new(config);

        let fast_tau = sgnn.fast_layer().unwrap().config.neuron_config.tau_membrane;
        let med_tau = sgnn.medium_layer().unwrap().config.neuron_config.tau_membrane;
        let slow_tau = sgnn.slow_layer().unwrap().config.neuron_config.tau_membrane;

        assert_abs_diff_eq!(fast_tau, 5.0);
        assert_abs_diff_eq!(med_tau, 20.0);
        assert_abs_diff_eq!(slow_tau, 100.0);

        // Verify decay rates
        let fast_leak = sgnn.fast_layer().unwrap().config.neuron_config.leak;
        let med_leak = sgnn.medium_layer().unwrap().config.neuron_config.leak;
        let slow_leak = sgnn.slow_layer().unwrap().config.neuron_config.leak;

        // Slower layers should have less leak (closer to 1)
        assert!(fast_leak < med_leak);
        assert!(med_leak < slow_leak);
    }
}
