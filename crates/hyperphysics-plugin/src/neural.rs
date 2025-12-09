//! # Neural Learning Module
//!
//! Implements Spike-Timing Dependent Plasticity (STDP) with Fibonacci time constants
//! and multi-scale learning for the HyperPhysics plugin.
//!
//! ## Mathematical Foundation
//!
//! ### STDP Learning Rule
//! ```text
//! ΔW(Δt) = Σᵢ aᵢ × exp(-|Δt| / τᵢ) × sign(Δt)
//! ```
//!
//! ### Fibonacci Time Constants
//! - τ ∈ {13, 21, 34, 55, 89} ms
//! - Amplitude decay: aᵢ = a₀ × φ⁻ⁱ
//!
//! ### Eligibility Traces
//! - e(t+1) = λ × e(t) + δ(spike)
//! - Enables three-factor learning with reward signals
//!
//! ## References
//!
//! - Bi, G., & Poo, M. (1998). "Synaptic modifications in cultured hippocampal neurons"
//! - Izhikevich, E. M. (2007). "Solving the distal reward problem through linkage of STDP
//!   and dopamine signaling"

use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// ============================================================================
// Constants
// ============================================================================

/// Golden ratio
pub const PHI: f64 = 1.618033988749895;

/// Inverse golden ratio
pub const PHI_INV: f64 = 0.618033988749895;

/// Fibonacci time constants (ms)
pub const FIBONACCI_TAU: [f64; 5] = [13.0, 21.0, 34.0, 55.0, 89.0];

/// Default LTP amplitude
pub const STDP_A_PLUS: f64 = 0.1;

/// Default LTD amplitude
pub const STDP_A_MINUS: f64 = 0.12;

/// Default time constant (ms)
pub const STDP_TAU: f64 = 20.0;

/// Default learning rate
pub const DEFAULT_LEARNING_RATE: f64 = 0.01;

/// Default eligibility trace decay
pub const DEFAULT_LAMBDA: f64 = 0.9;

/// Weight bounds
pub const DEFAULT_WEIGHT_BOUNDS: (f64, f64) = (-1.0, 1.0);

// ============================================================================
// STDP Learning Rule
// ============================================================================

/// Standard STDP weight change
///
/// # Arguments
/// * `delta_t` - Time difference (post - pre) in ms
/// * `a_plus` - LTP amplitude
/// * `a_minus` - LTD amplitude
/// * `tau` - Time constant (ms)
///
/// # Returns
/// Weight change ΔW
#[inline]
pub fn stdp_weight_change(delta_t: f64, a_plus: f64, a_minus: f64, tau: f64) -> f64 {
    if delta_t > 0.0 {
        a_plus * (-delta_t / tau).exp()
    } else {
        -a_minus * (delta_t / tau).exp()
    }
}

/// Fibonacci multi-scale STDP weight change
///
/// Uses 5 Fibonacci time constants with golden ratio amplitude decay.
///
/// # Arguments
/// * `delta_t` - Time difference (post - pre) in ms
///
/// # Returns
/// Multi-scale weight change
pub fn fibonacci_stdp_weight_change(delta_t: f64) -> f64 {
    let is_ltp = delta_t > 0.0;
    let abs_dt = delta_t.abs();

    let mut dw = 0.0;

    for (i, &tau) in FIBONACCI_TAU.iter().enumerate() {
        let amplitude = if is_ltp {
            STDP_A_PLUS * PHI_INV.powi(i as i32)
        } else {
            STDP_A_MINUS * PHI_INV.powi(i as i32)
        };

        let decay = (-abs_dt / tau).exp();
        dw += amplitude * decay;
    }

    if is_ltp {
        dw.min(1.0)
    } else {
        -dw.min(1.0)
    }
}

/// Compute effective time constant for Fibonacci STDP
pub fn effective_time_constant() -> f64 {
    // Weighted average based on golden ratio amplitudes
    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;

    for (i, &tau) in FIBONACCI_TAU.iter().enumerate() {
        let weight = PHI_INV.powi(i as i32);
        weighted_sum += tau * weight;
        total_weight += weight;
    }

    weighted_sum / total_weight
}

/// Compute STDP balance (ratio of LTP to LTD)
pub fn compute_stdp_balance(a_plus: f64, a_minus: f64, tau_plus: f64, tau_minus: f64) -> f64 {
    // Integral of LTP / Integral of LTD
    let ltp_integral = a_plus * tau_plus;
    let ltd_integral = a_minus * tau_minus;
    ltp_integral / ltd_integral.max(1e-10)
}

// ============================================================================
// Fibonacci STDP
// ============================================================================

/// Multi-scale Fibonacci STDP learning rule
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FibonacciSTDP {
    /// Fibonacci time constants
    pub tau: [f64; 5],
    /// Amplitude weights
    pub amplitudes: [f64; 5],
    /// Learning rate
    pub learning_rate: f64,
    /// Weight bounds
    pub weight_bounds: (f64, f64),
}

impl Default for FibonacciSTDP {
    fn default() -> Self {
        Self::new()
    }
}

impl FibonacciSTDP {
    /// Create with default parameters
    pub fn new() -> Self {
        Self::with_learning_rate(DEFAULT_LEARNING_RATE)
    }

    /// Create with custom learning rate
    pub fn with_learning_rate(learning_rate: f64) -> Self {
        let mut amplitudes = [0.0; 5];
        for i in 0..5 {
            amplitudes[i] = PHI_INV.powi(i as i32);
        }

        Self {
            tau: FIBONACCI_TAU,
            amplitudes,
            learning_rate,
            weight_bounds: DEFAULT_WEIGHT_BOUNDS,
        }
    }

    /// Compute weight change for spike timing
    pub fn compute_weight_change(&self, delta_t: f64) -> f64 {
        let raw_dw = fibonacci_stdp_weight_change(delta_t);
        self.learning_rate * raw_dw
    }

    /// Update weight with bounds
    pub fn update_weight(&self, current: f64, delta_t: f64) -> f64 {
        let dw = self.compute_weight_change(delta_t);
        let new = current + dw;
        new.clamp(self.weight_bounds.0, self.weight_bounds.1)
    }

    /// Compute learning window
    pub fn learning_window(&self, dt_range: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut ltp = Vec::with_capacity(dt_range.len());
        let mut ltd = Vec::with_capacity(dt_range.len());

        for &dt in dt_range {
            let dw = self.compute_weight_change(dt);
            if dt > 0.0 {
                ltp.push(dw);
                ltd.push(0.0);
            } else {
                ltp.push(0.0);
                ltd.push(dw.abs());
            }
        }

        (ltp, ltd)
    }

    /// Get scale contributions at given delta_t
    pub fn scale_contributions(&self, delta_t: f64) -> [f64; 5] {
        let abs_dt = delta_t.abs();
        let mut contributions = [0.0; 5];

        for (i, &tau) in self.tau.iter().enumerate() {
            contributions[i] = self.amplitudes[i] * (-abs_dt / tau).exp();
        }

        contributions
    }
}

// ============================================================================
// Eligibility Traces
// ============================================================================

/// Eligibility trace for three-factor learning
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EligibilityTrace {
    /// Trace value
    value: f64,
    /// Decay rate (λ)
    lambda: f64,
    /// Last update tick
    last_tick: u64,
}

impl Default for EligibilityTrace {
    fn default() -> Self {
        Self::new(DEFAULT_LAMBDA)
    }
}

impl EligibilityTrace {
    /// Create new trace
    pub fn new(lambda: f64) -> Self {
        Self {
            value: 0.0,
            lambda: lambda.clamp(0.0, 1.0),
            last_tick: 0,
        }
    }

    /// Get current value with lazy decay
    pub fn get(&self, current_tick: u64) -> f64 {
        let elapsed = current_tick.saturating_sub(self.last_tick);
        self.value * self.lambda.powi(elapsed as i32)
    }

    /// Accumulate spike contribution
    pub fn accumulate(&mut self, delta: f64, current_tick: u64) {
        // Apply lazy decay
        let decayed = self.get(current_tick);
        self.value = decayed + delta;
        self.last_tick = current_tick;
    }

    /// Apply modulation (reward signal)
    pub fn modulate(&self, reward: f64, current_tick: u64) -> f64 {
        self.get(current_tick) * reward
    }

    /// Reset trace
    pub fn reset(&mut self) {
        self.value = 0.0;
        self.last_tick = 0;
    }
}

/// Manager for multiple eligibility traces
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EligibilityTraceManager {
    /// Traces by synapse ID
    traces: HashMap<(usize, usize), EligibilityTrace>,
    /// Default decay rate
    lambda: f64,
}

impl Default for EligibilityTraceManager {
    fn default() -> Self {
        Self::new(DEFAULT_LAMBDA)
    }
}

impl EligibilityTraceManager {
    /// Create new manager
    pub fn new(lambda: f64) -> Self {
        Self {
            traces: HashMap::new(),
            lambda,
        }
    }

    /// Get trace for synapse
    pub fn get(&self, pre: usize, post: usize, tick: u64) -> f64 {
        self.traces
            .get(&(pre, post))
            .map(|t| t.get(tick))
            .unwrap_or(0.0)
    }

    /// Accumulate to trace
    pub fn accumulate(&mut self, pre: usize, post: usize, delta: f64, tick: u64) {
        self.traces
            .entry((pre, post))
            .or_insert_with(|| EligibilityTrace::new(self.lambda))
            .accumulate(delta, tick);
    }

    /// Apply reward to all traces
    pub fn apply_reward(&mut self, reward: f64, tick: u64) -> Vec<((usize, usize), f64)> {
        self.traces
            .iter()
            .map(|(&key, trace)| (key, trace.modulate(reward, tick)))
            .collect()
    }

    /// Clear all traces
    pub fn clear(&mut self) {
        self.traces.clear();
    }

    /// Get all active traces
    pub fn active_traces(&self, tick: u64, threshold: f64) -> Vec<(usize, usize)> {
        self.traces
            .iter()
            .filter(|(_, t)| t.get(tick).abs() > threshold)
            .map(|(&key, _)| key)
            .collect()
    }
}

// ============================================================================
// Spiking Neuron Model
// ============================================================================

/// Simple spiking neuron (LIF-like)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpikingNeuron {
    /// Membrane potential
    pub potential: f64,
    /// Threshold
    pub threshold: f64,
    /// Reset potential
    pub reset: f64,
    /// Decay rate
    pub decay: f64,
    /// Refractory period (ticks)
    pub refractory: usize,
    /// Current refractory counter
    refractory_counter: usize,
    /// Last spike time
    last_spike: Option<u64>,
}

impl Default for SpikingNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl SpikingNeuron {
    /// Create with default parameters
    pub fn new() -> Self {
        Self {
            potential: 0.0,
            threshold: 1.0,
            reset: 0.0,
            decay: 0.9,
            refractory: 2,
            refractory_counter: 0,
            last_spike: None,
        }
    }

    /// Create with custom threshold
    pub fn with_threshold(threshold: f64) -> Self {
        Self {
            threshold,
            ..Default::default()
        }
    }

    /// Integrate input current
    pub fn integrate(&mut self, input: f64, tick: u64) -> bool {
        // Check refractory
        if self.refractory_counter > 0 {
            self.refractory_counter -= 1;
            return false;
        }

        // Decay and integrate
        self.potential = self.potential * self.decay + input;

        // Check threshold
        if self.potential >= self.threshold {
            self.potential = self.reset;
            self.refractory_counter = self.refractory;
            self.last_spike = Some(tick);
            true
        } else {
            false
        }
    }

    /// Get last spike time
    pub fn last_spike_time(&self) -> Option<u64> {
        self.last_spike
    }

    /// Is in refractory period
    pub fn is_refractory(&self) -> bool {
        self.refractory_counter > 0
    }

    /// Reset state
    pub fn reset_state(&mut self) {
        self.potential = 0.0;
        self.refractory_counter = 0;
        self.last_spike = None;
    }
}

// ============================================================================
// Neural Network Layer
// ============================================================================

/// Simple spiking neural network layer
#[derive(Debug, Clone)]
pub struct SpikingLayer {
    /// Neurons
    neurons: Vec<SpikingNeuron>,
    /// Weight matrix (row: pre, col: post)
    weights: Vec<Vec<f64>>,
    /// STDP rule
    stdp: FibonacciSTDP,
    /// Eligibility traces
    traces: EligibilityTraceManager,
    /// Current tick
    tick: u64,
}

impl SpikingLayer {
    /// Create new layer
    pub fn new(n_pre: usize, n_post: usize) -> Self {
        let neurons: Vec<SpikingNeuron> = (0..n_post)
            .map(|_| SpikingNeuron::new())
            .collect();

        // Initialize weights with small random values
        let weights: Vec<Vec<f64>> = (0..n_pre)
            .map(|_| {
                (0..n_post)
                    .map(|_| 0.1)
                    .collect()
            })
            .collect();

        Self {
            neurons,
            weights,
            stdp: FibonacciSTDP::new(),
            traces: EligibilityTraceManager::default(),
            tick: 0,
        }
    }

    /// Forward pass
    pub fn forward(&mut self, inputs: &[bool]) -> Vec<bool> {
        self.tick += 1;
        let tick = self.tick;

        let mut outputs = vec![false; self.neurons.len()];

        for (post_idx, neuron) in self.neurons.iter_mut().enumerate() {
            // Sum weighted inputs
            let mut total_input = 0.0;
            for (pre_idx, &spiked) in inputs.iter().enumerate() {
                if spiked {
                    total_input += self.weights[pre_idx][post_idx];
                }
            }

            // Integrate
            outputs[post_idx] = neuron.integrate(total_input, tick);
        }

        outputs
    }

    /// Apply STDP learning
    pub fn learn(&mut self, pre_spikes: &[bool], post_spikes: &[bool]) {
        let tick = self.tick;

        for (pre_idx, &pre_spiked) in pre_spikes.iter().enumerate() {
            for (post_idx, &post_spiked) in post_spikes.iter().enumerate() {
                if pre_spiked || post_spiked {
                    // Compute delta_t
                    let delta_t = if pre_spiked && post_spiked {
                        0.0
                    } else if post_spiked {
                        1.0 // Post after pre → LTP
                    } else {
                        -1.0 // Pre after post → LTD
                    };

                    // Update weight
                    let dw = self.stdp.compute_weight_change(delta_t);
                    self.weights[pre_idx][post_idx] += dw;

                    // Clamp weights
                    self.weights[pre_idx][post_idx] = self.weights[pre_idx][post_idx]
                        .clamp(self.stdp.weight_bounds.0, self.stdp.weight_bounds.1);

                    // Update eligibility trace
                    self.traces.accumulate(pre_idx, post_idx, dw, tick);
                }
            }
        }
    }

    /// Apply reward signal
    pub fn apply_reward(&mut self, reward: f64) -> f64 {
        let updates = self.traces.apply_reward(reward, self.tick);
        let mut total_update = 0.0;

        for ((pre, post), delta) in updates {
            self.weights[pre][post] += delta;
            self.weights[pre][post] = self.weights[pre][post]
                .clamp(self.stdp.weight_bounds.0, self.stdp.weight_bounds.1);
            total_update += delta.abs();
        }

        total_update
    }

    /// Get weight matrix
    pub fn weights(&self) -> &Vec<Vec<f64>> {
        &self.weights
    }

    /// Get neurons
    pub fn neurons(&self) -> &[SpikingNeuron] {
        &self.neurons
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stdp_ltp() {
        let dw = stdp_weight_change(10.0, 0.1, 0.12, 20.0);
        assert!(dw > 0.0, "LTP should be positive");
    }

    #[test]
    fn test_stdp_ltd() {
        let dw = stdp_weight_change(-10.0, 0.1, 0.12, 20.0);
        assert!(dw < 0.0, "LTD should be negative");
    }

    #[test]
    fn test_fibonacci_stdp() {
        let stdp = FibonacciSTDP::new();
        let ltp = stdp.compute_weight_change(10.0);
        let ltd = stdp.compute_weight_change(-10.0);

        assert!(ltp > 0.0);
        assert!(ltd < 0.0);
    }

    #[test]
    fn test_fibonacci_stdp_symmetry() {
        let ltp = fibonacci_stdp_weight_change(10.0);
        let ltd = fibonacci_stdp_weight_change(-10.0);

        // LTD should be slightly stronger (homeostatic)
        assert!(ltd.abs() >= ltp.abs() * 0.9);
    }

    #[test]
    fn test_eligibility_trace() {
        let mut trace = EligibilityTrace::new(0.9);

        trace.accumulate(1.0, 0);
        assert!((trace.get(0) - 1.0).abs() < 0.01);

        // Should decay
        assert!(trace.get(10) < trace.get(0));
    }

    #[test]
    fn test_spiking_neuron() {
        let mut neuron = SpikingNeuron::new();

        // Sub-threshold input
        assert!(!neuron.integrate(0.5, 0));

        // Accumulate to threshold
        for i in 1..10 {
            if neuron.integrate(0.3, i) {
                assert!(i > 1, "Should take multiple inputs to spike");
                break;
            }
        }
    }

    #[test]
    fn test_refractory_period() {
        let mut neuron = SpikingNeuron::with_threshold(0.5);

        // Force spike
        neuron.integrate(1.0, 0);

        // Should be in refractory
        assert!(neuron.is_refractory());
        assert!(!neuron.integrate(1.0, 1));
    }

    #[test]
    fn test_spiking_layer() {
        let mut layer = SpikingLayer::new(3, 2);

        let inputs = [true, false, true];
        let outputs = layer.forward(&inputs);

        assert_eq!(outputs.len(), 2);
    }

    #[test]
    fn test_effective_time_constant() {
        let tau_eff = effective_time_constant();
        assert!(tau_eff > 13.0 && tau_eff < 89.0);
    }

    #[test]
    fn test_scale_contributions() {
        let stdp = FibonacciSTDP::new();
        let contributions = stdp.scale_contributions(20.0);

        // Fast scale should contribute more at short dt
        let short = stdp.scale_contributions(5.0);
        let long = stdp.scale_contributions(100.0);

        assert!(short[0] > long[0], "Fast scale should dominate at short dt");
    }
}
