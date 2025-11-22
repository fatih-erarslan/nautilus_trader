//! # Spike-Timing-Dependent Plasticity (STDP) Synapses
//!
//! This module implements synapses with STDP learning rules that modify synaptic weights
//! based on the relative timing of pre- and post-synaptic spikes.
//!
//! ## STDP Learning Rule
//!
//! **Weight Update:**
//! - If t_post - t_pre > 0: Δw = A_+ * exp(-Δt/τ_+) (LTP - Long Term Potentiation)  
//! - If t_post - t_pre < 0: Δw = -A_- * exp(Δt/τ_-) (LTD - Long Term Depression)
//!
//! **Synaptic Transmission:**
//! I_syn(t) = w * g(t) * (V_post - E_rev)
//!
//! ## Features
//!
//! - Exponential decay traces for efficient computation
//! - Weight bounds and normalization
//! - Multiple plasticity rules (Hebbian, anti-Hebbian, etc.)
//! - Synaptic delays for realistic timing
//! - Homeostatic scaling mechanisms

use crate::neuromorphic::spiking_neuron::SpikeEvent;
use crate::{TengriError, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Instant;

/// STDP learning rule configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningRule {
    /// LTP amplitude (A_+)
    pub a_plus: f64,
    
    /// LTD amplitude (A_-)  
    pub a_minus: f64,
    
    /// LTP time constant in ms (τ_+)
    pub tau_plus_ms: f64,
    
    /// LTD time constant in ms (τ_-)
    pub tau_minus_ms: f64,
    
    /// Learning rate scaling factor
    pub learning_rate: f64,
    
    /// Minimum synaptic weight
    pub w_min: f64,
    
    /// Maximum synaptic weight
    pub w_max: f64,
    
    /// Weight normalization target
    pub normalization_target: Option<f64>,
    
    /// Enable homeostatic scaling
    pub homeostatic_scaling: bool,
    
    /// Homeostatic time constant in ms
    pub homeostatic_tau_ms: f64,
}

impl Default for LearningRule {
    fn default() -> Self {
        Self {
            a_plus: 0.01,           // 1% weight change for coincident spikes
            a_minus: 0.0105,        // Slightly larger LTD (balances LTP)
            tau_plus_ms: 20.0,      // 20ms LTP window
            tau_minus_ms: 20.0,     // 20ms LTD window  
            learning_rate: 1.0,     // Full learning rate
            w_min: 0.0,             // No negative weights
            w_max: 1.0,             // Maximum weight
            normalization_target: Some(0.1), // Target average weight
            homeostatic_scaling: true,
            homeostatic_tau_ms: 10000.0, // 10 second homeostatic time constant
        }
    }
}

impl LearningRule {
    /// Create anti-Hebbian learning rule (reversed plasticity)
    pub fn anti_hebbian() -> Self {
        Self {
            a_plus: -0.01,  // LTP becomes depression
            a_minus: 0.0105, // LTD becomes potentiation  
            ..Default::default()
        }
    }
    
    /// Create symmetric STDP rule
    pub fn symmetric() -> Self {
        Self {
            a_plus: 0.01,
            a_minus: 0.01,   // Equal LTP and LTD
            ..Default::default()
        }
    }
    
    /// Create fast learning rule
    pub fn fast_learning() -> Self {
        Self {
            learning_rate: 5.0,  // 5x faster learning
            tau_plus_ms: 10.0,   // Shorter time constants
            tau_minus_ms: 10.0,
            ..Default::default()
        }
    }
    
    /// Create stabilized rule with strong homeostasis
    pub fn stabilized() -> Self {
        Self {
            homeostatic_scaling: true,
            homeostatic_tau_ms: 1000.0, // Faster homeostasis
            normalization_target: Some(0.05), // Lower target weight
            ..Default::default()
        }
    }
    
    /// Validate learning rule parameters
    pub fn validate(&self) -> Result<()> {
        if self.tau_plus_ms <= 0.0 || self.tau_minus_ms <= 0.0 {
            return Err(TengriError::Config("Time constants must be positive".to_string()));
        }
        
        if self.w_min >= self.w_max {
            return Err(TengriError::Config("Weight minimum must be less than maximum".to_string()));
        }
        
        if self.learning_rate < 0.0 {
            return Err(TengriError::Config("Learning rate cannot be negative".to_string()));
        }
        
        if self.homeostatic_tau_ms <= 0.0 {
            return Err(TengriError::Config("Homeostatic time constant must be positive".to_string()));
        }
        
        Ok(())
    }
}

/// Synaptic configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapseConfig {
    /// Synaptic delay in ms
    pub delay_ms: f64,
    
    /// Reversal potential in mV
    pub reversal_potential_mv: f64,
    
    /// Peak synaptic conductance in nS
    pub peak_conductance_ns: f64,
    
    /// Synaptic time constant in ms (for conductance decay)
    pub synaptic_tau_ms: f64,
    
    /// Initial synaptic weight (0.0 to 1.0)
    pub initial_weight: f64,
    
    /// Learning rule configuration
    pub learning_rule: LearningRule,
    
    /// Enable synaptic plasticity
    pub plasticity_enabled: bool,
    
    /// Synaptic failure probability (0.0 = reliable, 1.0 = always fails)
    pub failure_probability: f64,
}

impl Default for SynapseConfig {
    fn default() -> Self {
        Self {
            delay_ms: 1.0,               // 1ms synaptic delay
            reversal_potential_mv: 0.0,   // Excitatory synapse
            peak_conductance_ns: 0.1,     // 0.1 nS peak conductance
            synaptic_tau_ms: 2.0,        // 2ms synaptic decay
            initial_weight: 0.1,          // 10% initial weight
            learning_rule: LearningRule::default(),
            plasticity_enabled: true,
            failure_probability: 0.0,     // Reliable transmission
        }
    }
}

impl SynapseConfig {
    /// Create inhibitory synapse configuration
    pub fn inhibitory() -> Self {
        Self {
            reversal_potential_mv: -80.0, // Inhibitory reversal potential
            peak_conductance_ns: 0.2,     // Larger inhibitory conductance
            initial_weight: 0.15,
            ..Default::default()
        }
    }
    
    /// Create slow synapse (NMDA-like)
    pub fn slow_excitatory() -> Self {
        Self {
            synaptic_tau_ms: 10.0,       // Slow decay
            peak_conductance_ns: 0.05,   // Smaller peak conductance
            delay_ms: 2.0,               // Longer delay
            ..Default::default()
        }
    }
    
    /// Create fast synapse (AMPA-like)
    pub fn fast_excitatory() -> Self {
        Self {
            synaptic_tau_ms: 0.5,        // Very fast decay
            peak_conductance_ns: 0.2,    // Large peak conductance  
            delay_ms: 0.5,               // Short delay
            ..Default::default()
        }
    }
    
    /// Create unreliable synapse
    pub fn unreliable() -> Self {
        Self {
            failure_probability: 0.3,     // 30% failure rate
            ..Default::default()
        }
    }
}

/// Synaptic trace for STDP computation
#[derive(Debug, Clone)]
struct SynapticTrace {
    /// Current trace value
    value: f64,
    
    /// Decay time constant in ms
    tau_ms: f64,
    
    /// Last update time in ms
    last_update_ms: f64,
}

impl SynapticTrace {
    fn new(tau_ms: f64) -> Self {
        Self {
            value: 0.0,
            tau_ms,
            last_update_ms: 0.0,
        }
    }
    
    /// Update trace with exponential decay
    fn update(&mut self, current_time_ms: f64) {
        if current_time_ms > self.last_update_ms {
            let dt = current_time_ms - self.last_update_ms;
            let decay_factor = (-dt / self.tau_ms).exp();
            self.value *= decay_factor;
            self.last_update_ms = current_time_ms;
        }
    }
    
    /// Add spike to trace
    fn add_spike(&mut self, current_time_ms: f64) {
        self.update(current_time_ms);
        self.value += 1.0;
    }
    
    /// Get current trace value
    fn get_value(&mut self, current_time_ms: f64) -> f64 {
        self.update(current_time_ms);
        self.value
    }
    
    /// Reset trace
    fn reset(&mut self) {
        self.value = 0.0;
        self.last_update_ms = 0.0;
    }
}

/// STDP synapse connecting two neurons
#[derive(Debug, Clone)]
pub struct STDPSynapse {
    /// Unique synapse identifier
    pub id: usize,
    
    /// Pre-synaptic neuron ID
    pub pre_neuron_id: usize,
    
    /// Post-synaptic neuron ID  
    pub post_neuron_id: usize,
    
    /// Synapse configuration
    config: SynapseConfig,
    
    /// Current synaptic weight (0.0 to 1.0)
    weight: f64,
    
    /// Current synaptic conductance in nS
    conductance_ns: f64,
    
    /// Pre-synaptic trace for STDP
    pre_trace: SynapticTrace,
    
    /// Post-synaptic trace for STDP  
    post_trace: SynapticTrace,
    
    /// Delayed spike queue
    spike_queue: VecDeque<DelayedSpike>,
    
    /// Homeostatic scaling factor
    homeostatic_factor: f64,
    
    /// Activity-dependent scaling
    activity_scaling: f64,
    
    /// Total number of pre-synaptic spikes processed
    pre_spike_count: u64,
    
    /// Total number of post-synaptic spikes processed
    post_spike_count: u64,
    
    /// Total weight changes applied
    total_weight_change: f64,
    
    /// Last synaptic transmission time
    last_transmission_ms: f64,
    
    /// Random number generator for stochastic failures
    rng: fastrand::Rng,
}

/// Spike event with synaptic delay
#[derive(Debug, Clone)]
struct DelayedSpike {
    /// Original spike event
    spike: SpikeEvent,
    
    /// Time when spike should be delivered (after delay)
    delivery_time_ms: f64,
}

impl STDPSynapse {
    /// Create a new STDP synapse
    pub fn new(id: usize, pre_neuron_id: usize, post_neuron_id: usize, 
               config: SynapseConfig) -> Result<Self> {
        config.learning_rule.validate()?;
        
        let weight = config.initial_weight.clamp(config.learning_rule.w_min, 
                                               config.learning_rule.w_max);
        
        Ok(Self {
            id,
            pre_neuron_id,
            post_neuron_id,
            pre_trace: SynapticTrace::new(config.learning_rule.tau_plus_ms),
            post_trace: SynapticTrace::new(config.learning_rule.tau_minus_ms),
            spike_queue: VecDeque::new(),
            weight,
            conductance_ns: 0.0,
            homeostatic_factor: 1.0,
            activity_scaling: 1.0,
            pre_spike_count: 0,
            post_spike_count: 0,
            total_weight_change: 0.0,
            last_transmission_ms: 0.0,
            config,
            rng: fastrand::Rng::new(),
        })
    }
    
    /// Create synapse with custom seed
    pub fn with_seed(id: usize, pre_neuron_id: usize, post_neuron_id: usize,
                     config: SynapseConfig, seed: u64) -> Result<Self> {
        let mut synapse = Self::new(id, pre_neuron_id, post_neuron_id, config)?;
        synapse.rng = fastrand::Rng::with_seed(seed);
        Ok(synapse)
    }
    
    /// Process pre-synaptic spike
    pub fn process_pre_spike(&mut self, spike: &SpikeEvent, current_time_ms: f64) {
        if spike.neuron_id != self.pre_neuron_id {
            return; // Wrong neuron
        }
        
        self.pre_spike_count += 1;
        
        // Add to delayed spike queue
        let delayed_spike = DelayedSpike {
            spike: spike.clone(),
            delivery_time_ms: current_time_ms + self.config.delay_ms,
        };
        self.spike_queue.push_back(delayed_spike);
        
        // Update pre-synaptic trace for STDP
        self.pre_trace.add_spike(current_time_ms);
        
        // Apply STDP learning (LTD component)
        if self.config.plasticity_enabled {
            let post_trace_value = self.post_trace.get_value(current_time_ms);
            if post_trace_value > 0.0 {
                let delta_w = -self.config.learning_rule.a_minus * post_trace_value *
                             self.config.learning_rule.learning_rate * self.homeostatic_factor;
                self.update_weight(delta_w);
            }
        }
    }
    
    /// Process post-synaptic spike
    pub fn process_post_spike(&mut self, spike: &SpikeEvent, current_time_ms: f64) {
        if spike.neuron_id != self.post_neuron_id {
            return; // Wrong neuron
        }
        
        self.post_spike_count += 1;
        
        // Update post-synaptic trace for STDP
        self.post_trace.add_spike(current_time_ms);
        
        // Apply STDP learning (LTP component)
        if self.config.plasticity_enabled {
            let pre_trace_value = self.pre_trace.get_value(current_time_ms);
            if pre_trace_value > 0.0 {
                let delta_w = self.config.learning_rule.a_plus * pre_trace_value *
                             self.config.learning_rule.learning_rate * self.homeostatic_factor;
                self.update_weight(delta_w);
            }
        }
    }
    
    /// Update synapse state and return synaptic current
    pub fn update(&mut self, current_time_ms: f64, post_voltage_mv: f64) -> f64 {
        // Process delayed spikes
        self.process_delayed_spikes(current_time_ms, post_voltage_mv);
        
        // Update synaptic conductance (exponential decay)
        if current_time_ms > self.last_transmission_ms {
            let dt = current_time_ms - self.last_transmission_ms;
            let decay_factor = (-dt / self.config.synaptic_tau_ms).exp();
            self.conductance_ns *= decay_factor;
        }
        
        // Update homeostatic scaling
        if self.config.learning_rule.homeostatic_scaling {
            self.update_homeostatic_scaling(current_time_ms);
        }
        
        // Calculate synaptic current: I = g * (V_post - E_rev)
        let current_pa = self.conductance_ns * 1000.0 * // Convert nS to pS
                        (post_voltage_mv - self.config.reversal_potential_mv);
        
        current_pa
    }
    
    /// Process spikes that have reached their delivery time
    fn process_delayed_spikes(&mut self, current_time_ms: f64, post_voltage_mv: f64) {
        while let Some(delayed_spike) = self.spike_queue.front() {
            if delayed_spike.delivery_time_ms <= current_time_ms {
                let delayed_spike = self.spike_queue.pop_front().unwrap();
                
                // Check for synaptic failure
                if self.rng.f64() < self.config.failure_probability {
                    continue; // Transmission failed
                }
                
                // Generate synaptic conductance
                let conductance_increment = self.config.peak_conductance_ns * 
                                          self.weight * self.activity_scaling;
                self.conductance_ns += conductance_increment;
                self.last_transmission_ms = current_time_ms;
            } else {
                break; // No more ready spikes
            }
        }
    }
    
    /// Update synaptic weight with bounds checking
    fn update_weight(&mut self, delta_w: f64) {
        let old_weight = self.weight;
        self.weight = (self.weight + delta_w).clamp(self.config.learning_rule.w_min,
                                                   self.config.learning_rule.w_max);
        self.total_weight_change += self.weight - old_weight;
        
        // Apply weight normalization if enabled
        if let Some(target) = self.config.learning_rule.normalization_target {
            if self.weight > target * 2.0 {
                self.weight *= 0.95; // Gentle normalization
            }
        }
    }
    
    /// Update homeostatic scaling based on activity levels
    fn update_homeostatic_scaling(&mut self, current_time_ms: f64) {
        // Simple activity-based homeostatic scaling
        let time_window_ms = 1000.0; // 1 second window
        let target_rate_hz = 10.0;   // Target 10 Hz firing rate
        
        if current_time_ms > time_window_ms {
            let actual_pre_rate = self.pre_spike_count as f64 / (current_time_ms / 1000.0);
            let actual_post_rate = self.post_spike_count as f64 / (current_time_ms / 1000.0);
            
            let avg_rate = (actual_pre_rate + actual_post_rate) / 2.0;
            let rate_ratio = target_rate_hz / (avg_rate + 0.1); // Avoid division by zero
            
            // Slowly adjust homeostatic factor
            let tau = self.config.learning_rule.homeostatic_tau_ms;
            let alpha = 1.0 / tau;
            self.homeostatic_factor = self.homeostatic_factor * (1.0 - alpha) + 
                                     rate_ratio * alpha;
            
            // Clamp to reasonable bounds
            self.homeostatic_factor = self.homeostatic_factor.clamp(0.1, 10.0);
        }
    }
    
    /// Get current synaptic weight
    pub fn weight(&self) -> f64 {
        self.weight
    }
    
    /// Set synaptic weight directly
    pub fn set_weight(&mut self, weight: f64) {
        self.weight = weight.clamp(self.config.learning_rule.w_min,
                                  self.config.learning_rule.w_max);
    }
    
    /// Get current synaptic conductance
    pub fn conductance_ns(&self) -> f64 {
        self.conductance_ns
    }
    
    /// Get synapse configuration
    pub fn config(&self) -> &SynapseConfig {
        &self.config
    }
    
    /// Get pre-synaptic trace value
    pub fn pre_trace_value(&mut self, current_time_ms: f64) -> f64 {
        self.pre_trace.get_value(current_time_ms)
    }
    
    /// Get post-synaptic trace value  
    pub fn post_trace_value(&mut self, current_time_ms: f64) -> f64 {
        self.post_trace.get_value(current_time_ms)
    }
    
    /// Reset synapse to initial state
    pub fn reset(&mut self) {
        self.weight = self.config.initial_weight;
        self.conductance_ns = 0.0;
        self.pre_trace.reset();
        self.post_trace.reset();
        self.spike_queue.clear();
        self.homeostatic_factor = 1.0;
        self.activity_scaling = 1.0;
        self.pre_spike_count = 0;
        self.post_spike_count = 0;
        self.total_weight_change = 0.0;
        self.last_transmission_ms = 0.0;
    }
    
    /// Get learning statistics
    pub fn learning_stats(&self) -> (u64, u64, f64, f64) {
        (self.pre_spike_count, self.post_spike_count, 
         self.total_weight_change, self.homeostatic_factor)
    }
    
    /// Enable or disable plasticity
    pub fn set_plasticity_enabled(&mut self, enabled: bool) {
        // Note: This creates a new config which might be expensive
        // In a real implementation, you'd want config to be mutable
        let mut new_config = self.config.clone();
        new_config.plasticity_enabled = enabled;
        self.config = new_config;
    }
    
    /// Update learning rate
    pub fn set_learning_rate(&mut self, rate: f64) {
        let mut new_config = self.config.clone();
        new_config.learning_rule.learning_rate = rate.max(0.0);
        self.config = new_config;
    }
    
    /// Calculate effective synaptic strength
    pub fn effective_strength(&self) -> f64 {
        self.weight * self.homeostatic_factor * self.activity_scaling
    }
    
    /// Check if synapse is potentiated (weight increased)
    pub fn is_potentiated(&self) -> bool {
        self.total_weight_change > 0.0
    }
    
    /// Check if synapse is depressed (weight decreased)
    pub fn is_depressed(&self) -> bool {
        self.total_weight_change < 0.0
    }
}

/// Collection of synapses for efficient batch processing
#[derive(Debug)]
pub struct SynapsePool {
    /// Vector of synapses
    synapses: Vec<STDPSynapse>,
    
    /// Mapping from (pre_id, post_id) to synapse indices
    connection_map: std::collections::HashMap<(usize, usize), Vec<usize>>,
    
    /// Total number of synapses
    total_synapses: usize,
}

impl SynapsePool {
    /// Create a new synapse pool
    pub fn new() -> Self {
        Self {
            synapses: Vec::new(),
            connection_map: std::collections::HashMap::new(),
            total_synapses: 0,
        }
    }
    
    /// Add a synapse to the pool
    pub fn add_synapse(&mut self, synapse: STDPSynapse) {
        let connection_key = (synapse.pre_neuron_id, synapse.post_neuron_id);
        
        self.connection_map
            .entry(connection_key)
            .or_insert_with(Vec::new)
            .push(self.synapses.len());
        
        self.synapses.push(synapse);
        self.total_synapses += 1;
    }
    
    /// Process pre-synaptic spike for all relevant synapses
    pub fn process_pre_spike(&mut self, spike: &SpikeEvent, current_time_ms: f64) {
        for synapse in &mut self.synapses {
            if synapse.pre_neuron_id == spike.neuron_id {
                synapse.process_pre_spike(spike, current_time_ms);
            }
        }
    }
    
    /// Process post-synaptic spike for all relevant synapses
    pub fn process_post_spike(&mut self, spike: &SpikeEvent, current_time_ms: f64) {
        for synapse in &mut self.synapses {
            if synapse.post_neuron_id == spike.neuron_id {
                synapse.process_post_spike(spike, current_time_ms);
            }
        }
    }
    
    /// Update all synapses and collect currents for each post-synaptic neuron
    pub fn update_all(&mut self, current_time_ms: f64, 
                      post_voltages: &[f64]) -> std::collections::HashMap<usize, f64> {
        let mut currents = std::collections::HashMap::new();
        
        for synapse in &mut self.synapses {
            if let Some(&post_voltage) = post_voltages.get(synapse.post_neuron_id) {
                let current = synapse.update(current_time_ms, post_voltage);
                *currents.entry(synapse.post_neuron_id).or_insert(0.0) += current;
            }
        }
        
        currents
    }
    
    /// Get synapse by ID
    pub fn get_synapse(&self, id: usize) -> Option<&STDPSynapse> {
        self.synapses.iter().find(|s| s.id == id)
    }
    
    /// Get mutable synapse by ID
    pub fn get_synapse_mut(&mut self, id: usize) -> Option<&mut STDPSynapse> {
        self.synapses.iter_mut().find(|s| s.id == id)
    }
    
    /// Get synapses connecting specific neurons
    pub fn get_connections(&self, pre_id: usize, post_id: usize) -> Vec<&STDPSynapse> {
        if let Some(indices) = self.connection_map.get(&(pre_id, post_id)) {
            indices.iter()
                .filter_map(|&idx| self.synapses.get(idx))
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Get total number of synapses
    pub fn size(&self) -> usize {
        self.total_synapses
    }
    
    /// Reset all synapses
    pub fn reset_all(&mut self) {
        for synapse in &mut self.synapses {
            synapse.reset();
        }
    }
    
    /// Get average synaptic weight
    pub fn average_weight(&self) -> f64 {
        if self.synapses.is_empty() {
            return 0.0;
        }
        
        let total_weight: f64 = self.synapses.iter().map(|s| s.weight()).sum();
        total_weight / self.synapses.len() as f64
    }
    
    /// Get weight distribution statistics
    pub fn weight_statistics(&self) -> (f64, f64, f64, f64) { // (min, max, mean, std)
        if self.synapses.is_empty() {
            return (0.0, 0.0, 0.0, 0.0);
        }
        
        let weights: Vec<f64> = self.synapses.iter().map(|s| s.weight()).collect();
        
        let min_weight = weights.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_weight = weights.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mean_weight = weights.iter().sum::<f64>() / weights.len() as f64;
        
        let variance = weights.iter()
            .map(|w| (w - mean_weight).powi(2))
            .sum::<f64>() / weights.len() as f64;
        let std_weight = variance.sqrt();
        
        (min_weight, max_weight, mean_weight, std_weight)
    }
}

impl Default for SynapsePool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuromorphic::spiking_neuron::SpikeEvent;
    
    #[test]
    fn test_learning_rule_validation() {
        let mut rule = LearningRule::default();
        assert!(rule.validate().is_ok());
        
        // Test invalid time constants
        rule.tau_plus_ms = -1.0;
        assert!(rule.validate().is_err());
        
        // Test invalid weight bounds
        rule = LearningRule::default();
        rule.w_min = 1.0;
        rule.w_max = 0.0;
        assert!(rule.validate().is_err());
    }
    
    #[test]
    fn test_synapse_creation() {
        let config = SynapseConfig::default();
        let synapse = STDPSynapse::new(0, 1, 2, config).unwrap();
        
        assert_eq!(synapse.id, 0);
        assert_eq!(synapse.pre_neuron_id, 1);
        assert_eq!(synapse.post_neuron_id, 2);
        assert_eq!(synapse.weight(), 0.1); // Default initial weight
    }
    
    #[test]
    fn test_synaptic_transmission() {
        let config = SynapseConfig::fast_excitatory();
        let mut synapse = STDPSynapse::new(0, 1, 2, config).unwrap();
        
        // Create pre-synaptic spike
        let spike = SpikeEvent::new(1, 10.0, -55.0, 45.0);
        
        // Process spike
        synapse.process_pre_spike(&spike, 10.0);
        
        // Update synapse after delay
        let post_voltage = -60.0; // mV
        let current_time = 10.0 + synapse.config.delay_ms + 0.1;
        let synaptic_current = synapse.update(current_time, post_voltage);
        
        // Should have generated synaptic current
        assert!(synaptic_current != 0.0);
        assert!(synapse.conductance_ns() > 0.0);
    }
    
    #[test]
    fn test_stdp_ltp() {
        let mut config = SynapseConfig::default();
        config.learning_rule.learning_rate = 10.0; // High learning rate for testing
        let mut synapse = STDPSynapse::new(0, 1, 2, config).unwrap();
        
        let initial_weight = synapse.weight();
        
        // Pre-spike followed by post-spike (should cause LTP)
        let pre_spike = SpikeEvent::new(1, 10.0, -55.0, 45.0);
        let post_spike = SpikeEvent::new(2, 15.0, -55.0, 45.0); // 5ms after pre
        
        synapse.process_pre_spike(&pre_spike, 10.0);
        synapse.process_post_spike(&post_spike, 15.0);
        
        // Weight should increase (LTP)
        assert!(synapse.weight() > initial_weight, 
               "Weight should increase due to LTP: {} -> {}", 
               initial_weight, synapse.weight());
    }
    
    #[test]
    fn test_stdp_ltd() {
        let mut config = SynapseConfig::default();
        config.learning_rule.learning_rate = 10.0; // High learning rate for testing
        let mut synapse = STDPSynapse::new(0, 1, 2, config).unwrap();
        
        let initial_weight = synapse.weight();
        
        // Post-spike followed by pre-spike (should cause LTD)
        let post_spike = SpikeEvent::new(2, 10.0, -55.0, 45.0);
        let pre_spike = SpikeEvent::new(1, 15.0, -55.0, 45.0); // 5ms after post
        
        synapse.process_post_spike(&post_spike, 10.0);
        synapse.process_pre_spike(&pre_spike, 15.0);
        
        // Weight should decrease (LTD)
        assert!(synapse.weight() < initial_weight,
               "Weight should decrease due to LTD: {} -> {}",
               initial_weight, synapse.weight());
    }
    
    #[test]
    fn test_synaptic_delay() {
        let mut config = SynapseConfig::default();
        config.delay_ms = 5.0; // 5ms delay
        let mut synapse = STDPSynapse::new(0, 1, 2, config).unwrap();
        
        let spike = SpikeEvent::new(1, 10.0, -55.0, 45.0);
        synapse.process_pre_spike(&spike, 10.0);
        
        // Check that spike is not delivered immediately
        let current_early = synapse.update(10.1, -60.0);
        assert_eq!(current_early, 0.0);
        
        // Check that spike is delivered after delay
        let current_late = synapse.update(15.1, -60.0);
        assert!(current_late != 0.0 || synapse.conductance_ns() > 0.0);
    }
    
    #[test]
    fn test_synaptic_failure() {
        let mut config = SynapseConfig::default();
        config.failure_probability = 1.0; // Always fail
        let mut synapse = STDPSynapse::new(0, 1, 2, config).unwrap();
        
        let spike = SpikeEvent::new(1, 10.0, -55.0, 45.0);
        synapse.process_pre_spike(&spike, 10.0);
        
        // Update after delay - should not generate current due to failure
        let current_time = 10.0 + synapse.config.delay_ms + 0.1;
        let synaptic_current = synapse.update(current_time, -60.0);
        
        // Should have no current due to failure
        assert_eq!(synaptic_current, 0.0);
        assert_eq!(synapse.conductance_ns(), 0.0);
    }
    
    #[test]
    fn test_weight_bounds() {
        let mut config = SynapseConfig::default();
        config.learning_rule.w_max = 0.5;
        config.learning_rule.w_min = 0.0;
        let mut synapse = STDPSynapse::new(0, 1, 2, config).unwrap();
        
        // Try to set weight beyond bounds
        synapse.set_weight(1.0); // Above max
        assert!(synapse.weight() <= 0.5);
        
        synapse.set_weight(-0.1); // Below min
        assert!(synapse.weight() >= 0.0);
    }
    
    #[test]
    fn test_synapse_pool() {
        let mut pool = SynapsePool::new();
        
        // Add some synapses
        for i in 0..5 {
            let config = SynapseConfig::default();
            let synapse = STDPSynapse::new(i, 0, 1, config).unwrap(); // All connect 0->1
            pool.add_synapse(synapse);
        }
        
        assert_eq!(pool.size(), 5);
        
        // Test spike processing
        let spike = SpikeEvent::new(0, 10.0, -55.0, 45.0);
        pool.process_pre_spike(&spike, 10.0);
        
        // All synapses should have processed the spike
        let connections = pool.get_connections(0, 1);
        assert_eq!(connections.len(), 5);
    }
    
    #[test]
    fn test_homeostatic_scaling() {
        let mut config = SynapseConfig::default();
        config.learning_rule.homeostatic_scaling = true;
        config.learning_rule.homeostatic_tau_ms = 100.0; // Fast homeostasis
        
        let mut synapse = STDPSynapse::new(0, 1, 2, config).unwrap();
        
        // Generate many spikes to trigger homeostatic scaling
        for i in 0..100 {
            let spike = SpikeEvent::new(1, i as f64 * 10.0, -55.0, 45.0);
            synapse.process_pre_spike(&spike, i as f64 * 10.0);
            synapse.update(i as f64 * 10.0, -60.0);
        }
        
        let (_, _, _, homeostatic_factor) = synapse.learning_stats();
        
        // Homeostatic factor should have changed from initial value
        assert!((homeostatic_factor - 1.0).abs() > 0.01);
    }
    
    #[test]
    fn test_anti_hebbian_learning() {
        let config = SynapseConfig {
            learning_rule: LearningRule::anti_hebbian(),
            ..Default::default()
        };
        
        let mut synapse = STDPSynapse::new(0, 1, 2, config).unwrap();
        let initial_weight = synapse.weight();
        
        // Pre-spike followed by post-spike (normally LTP, but anti-Hebbian)
        let pre_spike = SpikeEvent::new(1, 10.0, -55.0, 45.0);
        let post_spike = SpikeEvent::new(2, 15.0, -55.0, 45.0);
        
        synapse.process_pre_spike(&pre_spike, 10.0);
        synapse.process_post_spike(&post_spike, 15.0);
        
        // With anti-Hebbian rule, weight should decrease instead of increase
        assert!(synapse.weight() < initial_weight);
    }
    
    #[test]
    fn test_trace_dynamics() {
        let config = SynapseConfig::default();
        let mut synapse = STDPSynapse::new(0, 1, 2, config).unwrap();
        
        // Add pre-synaptic spike
        let spike = SpikeEvent::new(1, 10.0, -55.0, 45.0);
        synapse.process_pre_spike(&spike, 10.0);
        
        // Check trace immediately after spike
        let trace_immediate = synapse.pre_trace_value(10.0);
        assert!(trace_immediate > 0.0);
        
        // Check trace decay over time
        let trace_later = synapse.pre_trace_value(50.0); // 40ms later
        assert!(trace_later < trace_immediate);
        assert!(trace_later > 0.0); // Should still be positive
    }
}