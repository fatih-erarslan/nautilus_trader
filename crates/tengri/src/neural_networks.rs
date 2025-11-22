//! Neural Network Components for TENGRI Trading System
//!
//! Implementation of biologically-inspired neural network components including
//! Spike-Timing-Dependent Plasticity (STDP) synapses for adaptive trading strategies.

use crate::TengriError;
use std::collections::VecDeque;

/// STDP (Spike-Timing-Dependent Plasticity) Synapse
///
/// Implements the biological learning rule where synaptic strength changes
/// based on the relative timing of pre- and post-synaptic spikes.
#[derive(Debug, Clone)]
pub struct STDPSynapse {
    /// Pre-synaptic neuron ID
    pub pre_neuron_id: usize,
    /// Post-synaptic neuron ID  
    pub post_neuron_id: usize,
    /// Current synaptic weight
    pub weight: f64,
    /// Maximum weight bound
    pub w_max: f64,
    /// Minimum weight bound (typically 0)
    pub w_min: f64,
    
    // STDP Learning Parameters
    /// LTP amplitude (A+)
    pub a_plus: f64,
    /// LTD amplitude (A-)
    pub a_minus: f64,
    /// LTP time constant (τ+) in milliseconds
    pub tau_plus: f64,
    /// LTD time constant (τ-) in milliseconds
    pub tau_minus: f64,
    
    // Synaptic Traces
    /// Pre-synaptic trace
    pub pre_trace: f64,
    /// Post-synaptic trace  
    pub post_trace: f64,
    
    // Synaptic Delay
    /// Synaptic transmission delay in milliseconds
    pub delay: f64,
    /// Spike buffer for delayed transmission
    spike_buffer: VecDeque<(f64, f64)>, // (spike_time, spike_value)
    
    // Validation metrics
    pub total_potentiation: f64,
    pub total_depression: f64,
    pub spike_count: u64,
}

impl STDPSynapse {
    /// Create a new STDP synapse with biological parameters
    ///
    /// # Arguments
    /// * `pre_neuron_id` - Pre-synaptic neuron identifier
    /// * `post_neuron_id` - Post-synaptic neuron identifier  
    /// * `initial_weight` - Initial synaptic weight
    ///
    /// # Returns
    /// * New STDP synapse with biologically realistic parameters
    pub fn new(pre_neuron_id: usize, post_neuron_id: usize, initial_weight: f64) -> Self {
        Self {
            pre_neuron_id,
            post_neuron_id,
            weight: initial_weight.max(0.0).min(1.0), // Clamp to valid range
            w_max: 1.0,
            w_min: 0.0,
            
            // Biological STDP parameters from literature
            a_plus: 0.005,    // LTP amplitude
            a_minus: 0.00525, // LTD amplitude (slightly larger for stability)
            tau_plus: 20.0,   // LTP time constant (ms)
            tau_minus: 20.0,  // LTD time constant (ms)
            
            pre_trace: 0.0,
            post_trace: 0.0,
            
            delay: 1.0, // 1ms default synaptic delay
            spike_buffer: VecDeque::with_capacity(100),
            
            total_potentiation: 0.0,
            total_depression: 0.0,
            spike_count: 0,
        }
    }

    /// Create STDP synapse with custom parameters
    pub fn with_parameters(
        pre_neuron_id: usize,
        post_neuron_id: usize,
        initial_weight: f64,
        a_plus: f64,
        a_minus: f64,
        tau_plus: f64,
        tau_minus: f64,
        w_max: f64,
        delay: f64,
    ) -> Self {
        Self {
            pre_neuron_id,
            post_neuron_id,
            weight: initial_weight.max(0.0).min(w_max),
            w_max,
            w_min: 0.0,
            a_plus,
            a_minus,
            tau_plus,
            tau_minus,
            pre_trace: 0.0,
            post_trace: 0.0,
            delay,
            spike_buffer: VecDeque::with_capacity(100),
            total_potentiation: 0.0,
            total_depression: 0.0,
            spike_count: 0,
        }
    }

    /// Update synapse using STDP learning rule
    ///
    /// # Arguments
    /// * `pre_spike` - True if pre-synaptic neuron spiked
    /// * `post_spike` - True if post-synaptic neuron spiked
    /// * `dt` - Time step in milliseconds
    /// * `current_time` - Current simulation time in milliseconds
    ///
    /// # Returns
    /// * Result indicating success or failure
    pub fn update_stdp(
        &mut self,
        pre_spike: bool,
        post_spike: bool,
        dt: f64,
        current_time: f64,
    ) -> Result<(), TengriError> {
        // Validate input parameters
        if dt <= 0.0 {
            return Err(TengriError::Unknown("Time step must be positive".to_string()));
        }

        // Decay traces with exponential decay
        self.pre_trace *= (-dt / self.tau_plus).exp();
        self.post_trace *= (-dt / self.tau_minus).exp();

        let mut weight_change = 0.0;

        // Handle pre-synaptic spike
        if pre_spike {
            self.spike_count += 1;
            
            // Add spike to delay buffer
            self.spike_buffer.push_back((current_time, 1.0));
            
            // Increment pre-synaptic trace
            self.pre_trace += 1.0;
            
            // Depression: post-before-pre (LTD)
            // Δw = -A_- * exp(Δt/τ_-)
            if self.post_trace > 0.0 {
                let depression = -self.a_minus * self.post_trace;
                weight_change += depression;
                self.total_depression += depression.abs();
            }
        }

        // Handle post-synaptic spike  
        if post_spike {
            // Potentiation: pre-before-post (LTP)
            // Δw = A_+ * exp(-Δt/τ_+)
            if self.pre_trace > 0.0 {
                let potentiation = self.a_plus * self.pre_trace;
                weight_change += potentiation;
                self.total_potentiation += potentiation;
            }
            
            // Increment post-synaptic trace
            self.post_trace += 1.0;
        }

        // Apply weight change with bounds
        self.weight += weight_change;
        self.weight = self.weight.max(self.w_min).min(self.w_max);

        // Clean old spikes from buffer (older than 5 * tau_plus)
        let cutoff_time = current_time - 5.0 * self.tau_plus.max(self.tau_minus);
        while let Some(&(spike_time, _)) = self.spike_buffer.front() {
            if spike_time < cutoff_time {
                self.spike_buffer.pop_front();
            } else {
                break;
            }
        }

        Ok(())
    }

    /// Get delayed synaptic output
    ///
    /// # Arguments  
    /// * `current_time` - Current simulation time in milliseconds
    ///
    /// # Returns
    /// * Synaptic current considering transmission delay
    pub fn get_delayed_output(&self, current_time: f64) -> f64 {
        let mut output = 0.0;
        
        // Check for spikes that should arrive now
        for &(spike_time, spike_value) in &self.spike_buffer {
            let arrival_time = spike_time + self.delay;
            if (arrival_time - current_time).abs() < 0.5 { // Within 0.5ms tolerance
                output += spike_value * self.weight;
            }
        }
        
        output
    }

    /// Reset synapse state
    pub fn reset(&mut self) {
        self.pre_trace = 0.0;
        self.post_trace = 0.0;
        self.spike_buffer.clear();
        self.total_potentiation = 0.0;
        self.total_depression = 0.0;
        self.spike_count = 0;
    }

    /// Get STDP curve for validation
    ///
    /// # Arguments
    /// * `delta_t_range` - Range of Δt values to test (ms)
    /// * `dt` - Time step for simulation
    ///
    /// # Returns
    /// * Vector of (Δt, Δw) pairs representing STDP curve
    pub fn get_stdp_curve(&self, delta_t_range: &[f64]) -> Vec<(f64, f64)> {
        let mut curve = Vec::new();
        
        for &delta_t in delta_t_range {
            let weight_change = if delta_t > 0.0 {
                // Pre before post (LTP)
                self.a_plus * (-delta_t / self.tau_plus).exp()
            } else if delta_t < 0.0 {
                // Post before pre (LTD)  
                -self.a_minus * (delta_t / self.tau_minus).exp()
            } else {
                0.0
            };
            
            curve.push((delta_t, weight_change));
        }
        
        curve
    }

    /// Validate STDP implementation against biological data
    pub fn validate_stdp(&self) -> Result<STDPValidation, TengriError> {
        let delta_t_range: Vec<f64> = (-100..=100).map(|x| x as f64).collect();
        let curve = self.get_stdp_curve(&delta_t_range);
        
        // Find maximum potentiation and depression
        let max_potentiation = curve.iter()
            .filter(|(dt, _)| *dt > 0.0)
            .map(|(_, dw)| *dw)
            .fold(0.0f64, |a, b| a.max(b));
        
        let max_depression = curve.iter()
            .filter(|(dt, _)| *dt < 0.0)
            .map(|(_, dw)| dw.abs())
            .fold(0.0f64, |a, b| a.max(b));
        
        // Check for proper STDP curve shape
        let has_ltp = max_potentiation > 0.0;
        let has_ltd = max_depression > 0.0;
        let ltp_ltd_ratio = max_potentiation / max_depression;
        
        Ok(STDPValidation {
            has_ltp,
            has_ltd,
            max_potentiation,
            max_depression,
            ltp_ltd_ratio,
            curve_points: curve.len(),
            is_valid: has_ltp && has_ltd && ltp_ltd_ratio > 0.5 && ltp_ltd_ratio < 2.0,
        })
    }

    /// Get synapse statistics
    pub fn get_statistics(&self) -> SynapseStatistics {
        let potentiation_ratio = if self.spike_count > 0 {
            self.total_potentiation / self.spike_count as f64
        } else {
            0.0
        };
        
        let depression_ratio = if self.spike_count > 0 {
            self.total_depression / self.spike_count as f64  
        } else {
            0.0
        };
        
        SynapseStatistics {
            weight: self.weight,
            total_potentiation: self.total_potentiation,
            total_depression: self.total_depression,
            spike_count: self.spike_count,
            potentiation_ratio,
            depression_ratio,
            trace_values: (self.pre_trace, self.post_trace),
        }
    }
}

/// STDP validation results
#[derive(Debug, Clone)]
pub struct STDPValidation {
    pub has_ltp: bool,
    pub has_ltd: bool,
    pub max_potentiation: f64,
    pub max_depression: f64,
    pub ltp_ltd_ratio: f64,
    pub curve_points: usize,
    pub is_valid: bool,
}

/// Synapse statistics for monitoring
#[derive(Debug, Clone)]
pub struct SynapseStatistics {
    pub weight: f64,
    pub total_potentiation: f64,
    pub total_depression: f64,
    pub spike_count: u64,
    pub potentiation_ratio: f64,
    pub depression_ratio: f64,
    pub trace_values: (f64, f64), // (pre_trace, post_trace)
}

/// Simple spiking neuron for testing STDP
#[derive(Debug, Clone)]
pub struct SpikingNeuron {
    pub id: usize,
    pub membrane_potential: f64,
    pub threshold: f64,
    pub reset_potential: f64,
    pub leak_rate: f64,
    pub refractory_period: f64,
    pub refractory_timer: f64,
    pub spike_times: Vec<f64>,
}

impl SpikingNeuron {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            membrane_potential: 0.0,
            threshold: 1.0,
            reset_potential: 0.0,
            leak_rate: 0.1,
            refractory_period: 5.0, // ms
            refractory_timer: 0.0,
            spike_times: Vec::new(),
        }
    }

    pub fn update(&mut self, input_current: f64, dt: f64, current_time: f64) -> bool {
        // Update refractory timer
        if self.refractory_timer > 0.0 {
            self.refractory_timer -= dt;
            return false;
        }

        // Leaky integrate-and-fire dynamics
        let leak = self.leak_rate * self.membrane_potential;
        self.membrane_potential += (input_current - leak) * dt;

        // Check for spike
        if self.membrane_potential >= self.threshold {
            self.membrane_potential = self.reset_potential;
            self.refractory_timer = self.refractory_period;
            self.spike_times.push(current_time);
            return true;
        }

        false
    }

    pub fn reset(&mut self) {
        self.membrane_potential = self.reset_potential;
        self.refractory_timer = 0.0;
        self.spike_times.clear();
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod basic_tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_stdp_synapse_creation() {
        let synapse = STDPSynapse::new(0, 1, 0.5);
        
        assert_eq!(synapse.pre_neuron_id, 0);
        assert_eq!(synapse.post_neuron_id, 1);
        assert_abs_diff_eq!(synapse.weight, 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(synapse.a_plus, 0.005, epsilon = 1e-10);
        assert_abs_diff_eq!(synapse.a_minus, 0.00525, epsilon = 1e-10);
        assert_abs_diff_eq!(synapse.tau_plus, 20.0, epsilon = 1e-10);
        assert_abs_diff_eq!(synapse.tau_minus, 20.0, epsilon = 1e-10);
    }

    #[test]
    fn test_stdp_learning_ltp() {
        let mut synapse = STDPSynapse::new(0, 1, 0.5);
        let dt = 0.1;
        let current_time = 0.0;
        
        // Pre-synaptic spike first (should increase pre_trace)
        synapse.update_stdp(true, false, dt, current_time).unwrap();
        assert!(synapse.pre_trace > 0.0);
        
        // Post-synaptic spike after delay (should cause LTP)
        let initial_weight = synapse.weight;
        synapse.update_stdp(false, true, dt, current_time + dt).unwrap();
        
        // Weight should increase (LTP)
        assert!(synapse.weight > initial_weight);
        assert!(synapse.total_potentiation > 0.0);
    }

    #[test]
    fn test_stdp_learning_ltd() {
        let mut synapse = STDPSynapse::new(0, 1, 0.5);
        let dt = 0.1;
        let current_time = 0.0;
        
        // Post-synaptic spike first (should increase post_trace)
        synapse.update_stdp(false, true, dt, current_time).unwrap();
        assert!(synapse.post_trace > 0.0);
        
        // Pre-synaptic spike after delay (should cause LTD)
        let initial_weight = synapse.weight;
        synapse.update_stdp(true, false, dt, current_time + dt).unwrap();
        
        // Weight should decrease (LTD)
        assert!(synapse.weight < initial_weight);
        assert!(synapse.total_depression > 0.0);
    }

    #[test]
    fn test_trace_decay() {
        let mut synapse = STDPSynapse::new(0, 1, 0.5);
        let dt = 1.0; // 1ms
        
        // Set initial traces
        synapse.pre_trace = 1.0;
        synapse.post_trace = 1.0;
        
        // Update without spikes (should decay traces)
        synapse.update_stdp(false, false, dt, 0.0).unwrap();
        
        let expected_pre_decay = (-dt / synapse.tau_plus).exp();
        let expected_post_decay = (-dt / synapse.tau_minus).exp();
        
        assert_abs_diff_eq!(synapse.pre_trace, expected_pre_decay, epsilon = 1e-10);
        assert_abs_diff_eq!(synapse.post_trace, expected_post_decay, epsilon = 1e-10);
    }

    #[test]
    fn test_weight_bounds() {
        let mut synapse = STDPSynapse::new(0, 1, 0.0); // Start at minimum
        
        // Try to decrease weight below minimum
        synapse.weight = -0.1;
        synapse.update_stdp(false, false, 0.1, 0.0).unwrap();
        assert!(synapse.weight >= synapse.w_min);
        
        // Set weight to maximum
        synapse.weight = synapse.w_max;
        
        // Apply large potentiation
        synapse.pre_trace = 10.0;
        synapse.update_stdp(false, true, 0.1, 0.0).unwrap();
        
        // Weight should be bounded by w_max
        assert!(synapse.weight <= synapse.w_max);
    }

    #[test]
    fn test_stdp_curve_validation() {
        let synapse = STDPSynapse::new(0, 1, 0.5);
        let validation = synapse.validate_stdp().unwrap();
        
        assert!(validation.has_ltp);
        assert!(validation.has_ltd);
        assert!(validation.max_potentiation > 0.0);
        assert!(validation.max_depression > 0.0);
        assert!(validation.is_valid);
        
        // Check that LTP occurs for positive Δt
        let curve = synapse.get_stdp_curve(&[10.0]);
        assert!(curve[0].1 > 0.0); // Positive weight change for Δt > 0
        
        // Check that LTD occurs for negative Δt
        let curve = synapse.get_stdp_curve(&[-10.0]);
        assert!(curve[0].1 < 0.0); // Negative weight change for Δt < 0
    }

    #[test]
    fn test_synaptic_delay() {
        let mut synapse = STDPSynapse::with_parameters(0, 1, 0.5, 0.005, 0.00525, 20.0, 20.0, 1.0, 2.0);
        
        // Send spike at t=0
        synapse.update_stdp(true, false, 0.1, 0.0).unwrap();
        
        // Check output at t=0 (should be 0 due to delay)
        let output_t0 = synapse.get_delayed_output(0.0);
        assert_abs_diff_eq!(output_t0, 0.0, epsilon = 1e-10);
        
        // Check output at t=2ms (spike should arrive)
        let output_t2 = synapse.get_delayed_output(2.0);
        assert!(output_t2 > 0.0);
    }

    #[test]
    fn test_spiking_neuron() {
        let mut neuron = SpikingNeuron::new(0);
        
        // Apply strong input current
        let spiked = neuron.update(2.0, 0.1, 0.0);
        
        // Should eventually spike with sufficient input
        let mut spike_occurred = spiked;
        for i in 1..100 {
            if neuron.update(2.0, 0.1, i as f64 * 0.1) {
                spike_occurred = true;
                break;
            }
        }
        
        assert!(spike_occurred);
        assert!(neuron.spike_times.len() > 0);
    }

    #[test]
    fn test_hebbian_learning_validation() {
        let mut synapse = STDPSynapse::new(0, 1, 0.5);
        let dt = 0.1;
        
        // Simulate correlated pre-post spiking (Hebbian)
        let initial_weight = synapse.weight;
        
        for i in 0..100 {
            let time = i as f64 * dt;
            // Pre spike followed by post spike (positive correlation)
            synapse.update_stdp(true, false, dt, time).unwrap();
            synapse.update_stdp(false, true, dt, time + dt).unwrap();
        }
        
        // Weight should increase due to positive correlation
        assert!(synapse.weight > initial_weight, 
               "Weight should increase with positive correlation (Hebbian learning)");
        
        let stats = synapse.get_statistics();
        assert!(stats.potentiation_ratio > 0.0);
    }
}