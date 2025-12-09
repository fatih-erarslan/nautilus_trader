//! # Synapse Module
//!
//! Implements weighted synaptic connections with:
//! - Adaptive weights via STDP
//! - Hyperbolic distance-based delays
//! - Synaptic efficacy modulation
//!
//! ## Wolfram-Verified STDP
//!
//! ```wolfram
//! (* STDP weight update *)
//! deltaW[deltaT_, aPlus_, aMinus_, tauPlus_, tauMinus_] :=
//!   If[deltaT > 0,
//!     aPlus * Exp[-deltaT/tauPlus],      (* LTP *)
//!     -aMinus * Exp[deltaT/tauMinus]     (* LTD *)
//!   ];
//!
//! (* Verified for deltaT=10ms, A+=0.1, A-=0.12, tau=20ms: *)
//! deltaW[10, 0.1, 0.12, 20, 20] = 0.0607
//! deltaW[-10, 0.1, 0.12, 20, 20] = -0.0728
//! ```

use serde::{Deserialize, Serialize};
use crate::constants::{STDP_A_PLUS, STDP_A_MINUS, STDP_TAU_PLUS, STDP_TAU_MINUS};

/// Configuration for a synapse
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapseConfig {
    /// Initial weight (can be negative for inhibitory)
    pub initial_weight: f64,

    /// Transmission delay (ms)
    /// Computed from hyperbolic distance between neurons
    pub delay: f64,

    /// LTP amplitude
    pub a_plus: f64,

    /// LTD amplitude
    pub a_minus: f64,

    /// LTP time constant (ms)
    pub tau_plus: f64,

    /// LTD time constant (ms)
    pub tau_minus: f64,

    /// Minimum weight (for stability)
    pub w_min: f64,

    /// Maximum weight (for stability)
    pub w_max: f64,
}

impl Default for SynapseConfig {
    fn default() -> Self {
        Self {
            initial_weight: 0.5,
            delay: 1.0,
            a_plus: STDP_A_PLUS,
            a_minus: STDP_A_MINUS,
            tau_plus: STDP_TAU_PLUS,
            tau_minus: STDP_TAU_MINUS,
            w_min: 0.0,
            w_max: 2.0,
        }
    }
}

impl SynapseConfig {
    /// Create excitatory synapse configuration
    pub fn excitatory(delay: f64) -> Self {
        Self {
            initial_weight: 0.5,
            delay,
            ..Default::default()
        }
    }

    /// Create inhibitory synapse configuration
    pub fn inhibitory(delay: f64) -> Self {
        Self {
            initial_weight: -0.5,
            delay,
            w_min: -2.0,
            w_max: 0.0,
            ..Default::default()
        }
    }

    /// Create synapse with delay from hyperbolic distance
    ///
    /// # Wolfram Formula
    /// ```wolfram
    /// (* Convert hyperbolic distance to axonal delay *)
    /// (* Assume propagation speed ~1 m/s, distance normalized to [0,1] *)
    /// delay[dist_] := 0.5 + 2.0 * dist  (* ms *)
    /// ```
    pub fn from_hyperbolic_distance(distance: f64, excitatory: bool) -> Self {
        // Convert hyperbolic distance to delay (0.5ms to 2.5ms range)
        let delay = 0.5 + 2.0 * distance.min(1.0);

        if excitatory {
            Self::excitatory(delay)
        } else {
            Self::inhibitory(delay)
        }
    }
}

/// Synapse connecting two neurons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Synapse {
    /// Source neuron ID
    pub source_id: u32,

    /// Target neuron ID
    pub target_id: u32,

    /// Configuration
    config: SynapseConfig,

    /// Current synaptic weight
    weight: f64,

    /// Spike buffer for delayed transmission
    spike_buffer: Vec<u64>, // Timestamps in μs
}

impl Synapse {
    /// Create a new synapse
    pub fn new(source_id: u32, target_id: u32, config: SynapseConfig) -> Self {
        let weight = config.initial_weight;
        Self {
            source_id,
            target_id,
            config,
            weight,
            spike_buffer: Vec::new(),
        }
    }

    /// Create an excitatory synapse
    pub fn excitatory(source_id: u32, target_id: u32, delay: f64) -> Self {
        Self::new(source_id, target_id, SynapseConfig::excitatory(delay))
    }

    /// Create an inhibitory synapse
    pub fn inhibitory(source_id: u32, target_id: u32, delay: f64) -> Self {
        Self::new(source_id, target_id, SynapseConfig::inhibitory(delay))
    }

    /// Receive a presynaptic spike
    ///
    /// Adds spike to buffer for delayed transmission
    pub fn receive_spike(&mut self, timestamp: u64) {
        let delay_us = (self.config.delay * 1000.0) as u64;
        let delivery_time = timestamp + delay_us;
        self.spike_buffer.push(delivery_time);
    }

    /// Get spikes ready for delivery at current time
    ///
    /// Returns weighted input current for each delivered spike
    pub fn deliver_spikes(&mut self, current_time: u64) -> f64 {
        let mut total_current = 0.0;

        // Remove and process spikes that should be delivered
        self.spike_buffer.retain(|&delivery_time| {
            if delivery_time <= current_time {
                total_current += self.weight;
                false // Remove from buffer
            } else {
                true // Keep in buffer
            }
        });

        total_current
    }

    /// Update weight using STDP learning rule
    ///
    /// # Arguments
    /// * `delta_t` - Time difference (post - pre) in milliseconds
    ///
    /// # Wolfram-Verified STDP
    /// ```wolfram
    /// (* LTP: pre before post (deltaT > 0) *)
    /// deltaW = A+ * Exp[-deltaT/tau+]
    ///
    /// (* LTD: post before pre (deltaT < 0) *)
    /// deltaW = -A- * Exp[deltaT/tau-]
    ///
    /// (* Verified examples: *)
    /// deltaW[10ms] = 0.1 * Exp[-10/20] = 0.0607
    /// deltaW[-10ms] = -0.12 * Exp[10/20] = -0.0728
    /// ```
    pub fn stdp_update(&mut self, delta_t: f64) {
        let delta_w = if delta_t > 0.0 {
            // LTP: pre before post
            self.config.a_plus * (-delta_t / self.config.tau_plus).exp()
        } else {
            // LTD: post before pre
            // For delta_t < 0: exp(delta_t/tau) = exp(-|delta_t|/tau)
            -self.config.a_minus * (delta_t / self.config.tau_minus).exp()
        };

        // Update weight with bounds
        self.weight = (self.weight + delta_w).clamp(self.config.w_min, self.config.w_max);
    }

    /// Get current weight
    #[inline]
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// Set weight (for manual control)
    pub fn set_weight(&mut self, weight: f64) {
        self.weight = weight.clamp(self.config.w_min, self.config.w_max);
    }

    /// Get delay (ms)
    #[inline]
    pub fn delay(&self) -> f64 {
        self.config.delay
    }

    /// Check if synapse is excitatory
    #[inline]
    pub fn is_excitatory(&self) -> bool {
        self.weight > 0.0
    }

    /// Check if synapse is inhibitory
    #[inline]
    pub fn is_inhibitory(&self) -> bool {
        self.weight < 0.0
    }

    /// Get number of spikes in buffer
    #[inline]
    pub fn buffered_spikes(&self) -> usize {
        self.spike_buffer.len()
    }

    /// Clear spike buffer
    pub fn clear_buffer(&mut self) {
        self.spike_buffer.clear();
    }

    /// Get configuration
    pub fn config(&self) -> &SynapseConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_synapse_config_default() {
        let config = SynapseConfig::default();
        assert_abs_diff_eq!(config.initial_weight, 0.5);
        assert_abs_diff_eq!(config.a_plus, STDP_A_PLUS);
        assert_abs_diff_eq!(config.a_minus, STDP_A_MINUS);
    }

    #[test]
    fn test_synapse_excitatory_inhibitory() {
        let exc = Synapse::excitatory(0, 1, 1.0);
        let inh = Synapse::inhibitory(0, 1, 1.0);

        assert!(exc.is_excitatory());
        assert!(!exc.is_inhibitory());
        assert!(!inh.is_excitatory());
        assert!(inh.is_inhibitory());
        assert!(exc.weight() > 0.0);
        assert!(inh.weight() < 0.0);
    }

    #[test]
    fn test_synapse_from_hyperbolic_distance() {
        let syn = SynapseConfig::from_hyperbolic_distance(0.5, true);

        // Delay should be in range [0.5, 2.5]ms
        assert!(syn.delay >= 0.5);
        assert!(syn.delay <= 2.5);
    }

    #[test]
    fn test_synapse_spike_buffering() {
        let mut synapse = Synapse::excitatory(0, 1, 1.0);

        // Send spike at t=1000μs
        synapse.receive_spike(1000);
        assert_eq!(synapse.buffered_spikes(), 1);

        // Check delivery before delay
        let current = synapse.deliver_spikes(1500);
        assert_abs_diff_eq!(current, 0.0); // Not delivered yet
        assert_eq!(synapse.buffered_spikes(), 1);

        // Check delivery after delay (1ms = 1000μs)
        let current = synapse.deliver_spikes(2500);
        assert!(current > 0.0); // Delivered
        assert_eq!(synapse.buffered_spikes(), 0);
    }

    #[test]
    fn test_stdp_ltp_wolfram_verified() {
        let mut synapse = Synapse::excitatory(0, 1, 1.0);
        let initial_weight = synapse.weight();

        // Pre before post: LTP (deltaT = 10ms)
        synapse.stdp_update(10.0);

        let delta_w = synapse.weight() - initial_weight;
        let expected = STDP_A_PLUS * (-10.0 / STDP_TAU_PLUS).exp();

        assert_abs_diff_eq!(delta_w, expected, epsilon = 0.001);
        assert_abs_diff_eq!(delta_w, 0.0607, epsilon = 0.001);
    }

    #[test]
    fn test_stdp_ltd_wolfram_verified() {
        let mut synapse = Synapse::excitatory(0, 1, 1.0);
        let initial_weight = synapse.weight();

        // Post before pre: LTD (deltaT = -10ms)
        synapse.stdp_update(-10.0);

        let delta_w = synapse.weight() - initial_weight;
        let expected = -STDP_A_MINUS * (-10.0 / STDP_TAU_MINUS).exp();

        assert_abs_diff_eq!(delta_w, expected, epsilon = 0.001);
        assert_abs_diff_eq!(delta_w, -0.0728, epsilon = 0.001);
    }

    #[test]
    fn test_stdp_weight_bounds() {
        let mut synapse = Synapse::excitatory(0, 1, 1.0);

        // Saturate to max
        for _ in 0..100 {
            synapse.stdp_update(10.0);
        }
        assert!(synapse.weight() <= synapse.config.w_max);

        // Saturate to min
        for _ in 0..1000 {
            synapse.stdp_update(-10.0);
        }
        assert!(synapse.weight() >= synapse.config.w_min);
    }

    #[test]
    fn test_synapse_set_weight() {
        let mut synapse = Synapse::excitatory(0, 1, 1.0);

        synapse.set_weight(1.5);
        assert_abs_diff_eq!(synapse.weight(), 1.5);

        // Test clamping
        synapse.set_weight(10.0);
        assert_abs_diff_eq!(synapse.weight(), synapse.config.w_max);
    }

    #[test]
    fn test_synapse_multiple_spikes() {
        let mut synapse = Synapse::excitatory(0, 1, 1.0);

        // Multiple spikes
        synapse.receive_spike(1000);
        synapse.receive_spike(1500);
        synapse.receive_spike(2000);
        assert_eq!(synapse.buffered_spikes(), 3);

        // Deliver all at once
        let current = synapse.deliver_spikes(10000);
        let expected = 3.0 * synapse.weight();
        assert_abs_diff_eq!(current, expected, epsilon = 1e-10);
        assert_eq!(synapse.buffered_spikes(), 0);
    }
}
