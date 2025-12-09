//! # Multi-Scale Fibonacci STDP Learning
//!
//! This module implements Spike-Timing Dependent Plasticity (STDP) with Fibonacci
//! time constants for the pentagonal cortex topology.
//!
//! ## Mathematical Foundation
//!
//! ### Multi-Scale STDP Rule
//!
//! The weight change is computed as a sum over 5 Fibonacci time scales:
//!
//! ```text
//! ΔW(Δt) = Σᵢ aᵢ × exp(-|Δt| / τᵢ) × sign(Δt)
//! ```
//!
//! where:
//! - τᵢ ∈ {13, 21, 34, 55, 89} ms (Fibonacci sequence)
//! - aᵢ = a₀ × φ⁻ⁱ (golden ratio amplitude decay)
//! - Δt = t_post - t_pre (spike timing difference)
//!
//! ### Properties (Wolfram-Verified)
//!
//! 1. **Multi-Scale Integration**:
//!    - Fast scale (13ms): Captures immediate correlations
//!    - Slow scale (89ms): Captures long-range dependencies
//!
//! 2. **Golden Ratio Decay**:
//!    - Amplitude weights follow φ⁻ⁱ geometric decay
//!    - Ensures balanced contribution across scales
//!
//! 3. **Homeostatic Balance**:
//!    - Net LTP/LTD slightly favors depression
//!    - Weight changes bounded to [-1, 1]
//!
//! 4. **Asymmetric Learning Window**:
//!    - LTP (Δt > 0): Potentiation when pre→post
//!    - LTD (Δt < 0): Depression when post→pre
//!
//! ## References
//!
//! - Bi, G., & Poo, M. (1998). "Synaptic modifications in cultured hippocampal
//!   neurons: dependence on spike timing, synaptic strength, and postsynaptic
//!   cell type." Journal of Neuroscience, 18(24), 10464-10472.
//! - Song, S., Miller, K. D., & Abbott, L. F. (2000). "Competitive Hebbian
//!   learning through spike-timing-dependent synaptic plasticity." Nature
//!   Neuroscience, 3(9), 919-926.

use super::constants::{PHI_INV, FIBONACCI_TAU};
use crate::constants::{STDP_A_PLUS, STDP_A_MINUS};

// ============================================================================
// Constants
// ============================================================================

/// Number of Fibonacci time scales
const NUM_SCALES: usize = 5;

/// Default learning rate for Fibonacci STDP
const DEFAULT_LEARNING_RATE: f64 = 0.01;

/// Default weight bounds
const DEFAULT_WEIGHT_BOUNDS: (f64, f64) = (-1.0, 1.0);

/// Epsilon for numerical comparisons
const EPSILON: f64 = 1e-10;

// ============================================================================
// Multi-Scale Fibonacci STDP
// ============================================================================

/// Compute multi-scale Fibonacci STDP weight change
///
/// Combines contributions from all 5 Fibonacci time scales:
/// ```text
/// ΔW = Σᵢ aᵢ × exp(-|Δt| / τᵢ) × sign(Δt)
/// ```
///
/// # Arguments
/// * `delta_t` - Spike timing difference (ms): Δt = t_post - t_pre
///
/// # Returns
/// Weight change in range [-1, 1]
///
/// # Examples
/// ```
/// use tengri_holographic_cortex::fibonacci::fibonacci_stdp::fibonacci_stdp_weight_change;
///
/// // LTP: pre before post (Δt > 0)
/// let dw_ltp = fibonacci_stdp_weight_change(10.0);
/// assert!(dw_ltp > 0.0);
///
/// // LTD: post before pre (Δt < 0)
/// let dw_ltd = fibonacci_stdp_weight_change(-10.0);
/// assert!(dw_ltd < 0.0);
/// ```
pub fn fibonacci_stdp_weight_change(delta_t: f64) -> f64 {
    // Special case: simultaneous spikes produce no change
    if delta_t.abs() < EPSILON {
        return 0.0;
    }

    let mut dw = 0.0;
    let is_ltp = delta_t > 0.0;
    let abs_dt = delta_t.abs();

    for i in 0..NUM_SCALES {
        // Golden ratio amplitude decay: aᵢ = a₀ × φ⁻ⁱ
        let amplitude = if is_ltp {
            STDP_A_PLUS * PHI_INV.powi(i as i32)
        } else {
            STDP_A_MINUS * PHI_INV.powi(i as i32)
        };

        // Exponential decay with Fibonacci time constant
        let decay = (-abs_dt / FIBONACCI_TAU[i]).exp();

        // Accumulate contribution
        dw += amplitude * decay;
    }

    // Apply sign based on causality
    if is_ltp {
        dw.min(1.0) // Bound LTP
    } else {
        -dw.min(1.0) // Bound LTD
    }
}

// ============================================================================
// FibonacciSTDP Struct
// ============================================================================

/// Multi-scale Fibonacci STDP learning rule
///
/// Implements spike-timing dependent plasticity with 5 Fibonacci time constants
/// and golden ratio amplitude decay.
#[derive(Debug, Clone)]
pub struct FibonacciSTDP {
    /// Fibonacci time constants: [13, 21, 34, 55, 89] ms
    pub tau: [f64; NUM_SCALES],

    /// Amplitude weights following φ⁻ⁱ decay
    pub amplitudes: [f64; NUM_SCALES],

    /// Learning rate (scales overall plasticity)
    pub learning_rate: f64,

    /// Weight bounds (min, max)
    pub weight_bounds: (f64, f64),
}

impl FibonacciSTDP {
    /// Create new Fibonacci STDP with default parameters
    ///
    /// # Returns
    /// FibonacciSTDP instance with:
    /// - τ = [13, 21, 34, 55, 89] ms
    /// - Amplitudes following φ⁻ⁱ decay
    /// - Learning rate = 0.01
    /// - Weight bounds = [-1, 1]
    pub fn new() -> Self {
        Self::with_learning_rate(DEFAULT_LEARNING_RATE)
    }

    /// Create Fibonacci STDP with custom learning rate
    ///
    /// # Arguments
    /// * `learning_rate` - Global learning rate multiplier
    pub fn with_learning_rate(learning_rate: f64) -> Self {
        let tau = FIBONACCI_TAU;

        // Compute golden ratio amplitude decay
        let mut amplitudes = [0.0; NUM_SCALES];
        for i in 0..NUM_SCALES {
            amplitudes[i] = PHI_INV.powi(i as i32);
        }

        Self {
            tau,
            amplitudes,
            learning_rate,
            weight_bounds: DEFAULT_WEIGHT_BOUNDS,
        }
    }

    /// Compute weight change for given spike timing
    ///
    /// # Arguments
    /// * `delta_t` - Spike timing difference (ms)
    ///
    /// # Returns
    /// Scaled weight change: learning_rate × ΔW(Δt)
    pub fn compute_weight_change(&self, delta_t: f64) -> f64 {
        let raw_dw = fibonacci_stdp_weight_change(delta_t);
        self.learning_rate * raw_dw
    }

    /// Apply STDP update to a weight
    ///
    /// # Arguments
    /// * `current_weight` - Current synaptic weight
    /// * `delta_t` - Spike timing difference (ms)
    ///
    /// # Returns
    /// Updated weight (bounded to weight_bounds)
    pub fn update_weight(&self, current_weight: f64, delta_t: f64) -> f64 {
        let dw = self.compute_weight_change(delta_t);
        let new_weight = current_weight + dw;

        // Clamp to bounds
        new_weight.max(self.weight_bounds.0).min(self.weight_bounds.1)
    }

    /// Compute the complete STDP learning window
    ///
    /// Returns separate LTP and LTD curves for plotting/analysis.
    ///
    /// # Arguments
    /// * `dt_range` - Array of Δt values to evaluate (ms)
    ///
    /// # Returns
    /// Tuple of (ltp_curve, ltd_curve) where each is a Vec<f64>
    ///
    /// # Examples
    /// ```
    /// use tengri_holographic_cortex::fibonacci::fibonacci_stdp::FibonacciSTDP;
    ///
    /// let stdp = FibonacciSTDP::new();
    /// let dt_range: Vec<f64> = (-100..=100).map(|x| x as f64).collect();
    /// let (ltp, ltd) = stdp.learning_window(&dt_range);
    /// ```
    pub fn learning_window(&self, dt_range: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut ltp_curve = Vec::with_capacity(dt_range.len());
        let mut ltd_curve = Vec::with_capacity(dt_range.len());

        for &dt in dt_range {
            let dw = self.compute_weight_change(dt);

            if dt >= 0.0 {
                ltp_curve.push(dw);
                ltd_curve.push(0.0);
            } else {
                ltp_curve.push(0.0);
                ltd_curve.push(dw.abs()); // Store magnitude for LTD
            }
        }

        (ltp_curve, ltd_curve)
    }

    /// Apply Fibonacci STDP to pentagonal inter-engine couplings
    ///
    /// Updates the 5×5 coupling matrix based on cross-engine spike correlations.
    ///
    /// # Arguments
    /// * `spike_times` - Recent spike times for each engine [engine][spike_idx]
    ///   - Each engine maintains a buffer of recent spike times (μs)
    /// * `couplings` - Mutable 5×5 coupling matrix to update
    ///
    /// # Algorithm
    /// 1. For each pair of engines (i, j):
    ///    - Compare all recent spike pairs
    ///    - Compute Δt = t_j - t_i (convert μs → ms)
    ///    - Accumulate STDP weight change
    ///    - Update coupling[i][j]
    ///
    /// # Examples
    /// ```
    /// use tengri_holographic_cortex::fibonacci::fibonacci_stdp::FibonacciSTDP;
    /// use tengri_holographic_cortex::fibonacci::constants::FIBONACCI_COUPLING;
    ///
    /// let stdp = FibonacciSTDP::new();
    /// let mut couplings = FIBONACCI_COUPLING.clone();
    ///
    /// // Simulate spike times (μs)
    /// let spike_times = [
    ///     [1000, 2000, 3000, 0, 0],
    ///     [1500, 2500, 3500, 0, 0],
    ///     [1200, 2200, 3200, 0, 0],
    ///     [1800, 2800, 3800, 0, 0],
    ///     [1100, 2100, 3100, 0, 0],
    /// ];
    ///
    /// stdp.update_pentagon_couplings(&spike_times, &mut couplings);
    /// ```
    pub fn update_pentagon_couplings(
        &self,
        spike_times: &[[u64; 5]; 5],
        couplings: &mut [[f64; 5]; 5],
    ) {
        const NUM_ENGINES: usize = 5;

        // Iterate over all engine pairs
        for i in 0..NUM_ENGINES {
            for j in 0..NUM_ENGINES {
                if i == j {
                    // Skip self-coupling
                    continue;
                }

                let mut total_dw = 0.0;
                let mut num_pairs = 0;

                // Compare all spike pairs between engines i and j
                for &t_pre in &spike_times[i] {
                    if t_pre == 0 {
                        break; // No more valid spikes
                    }

                    for &t_post in &spike_times[j] {
                        if t_post == 0 {
                            break;
                        }

                        // Compute timing difference (μs → ms)
                        let delta_t = (t_post as f64 - t_pre as f64) / 1000.0;

                        // Only consider spikes within reasonable window
                        if delta_t.abs() < 200.0 {
                            total_dw += self.compute_weight_change(delta_t);
                            num_pairs += 1;
                        }
                    }
                }

                // Apply averaged weight change
                if num_pairs > 0 {
                    let avg_dw = total_dw / num_pairs as f64;
                    let new_coupling = couplings[i][j] + avg_dw;

                    // Clamp to bounds
                    couplings[i][j] = new_coupling
                        .max(self.weight_bounds.0)
                        .min(self.weight_bounds.1);
                }
            }
        }
    }
}

impl Default for FibonacciSTDP {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute the net LTP/LTD balance over a time window
///
/// Integrates the STDP learning window to measure homeostatic balance.
///
/// # Arguments
/// * `window_ms` - Time window to integrate (ms)
///
/// # Returns
/// Net weight change (positive = LTP-dominated, negative = LTD-dominated)
pub fn compute_stdp_balance(window_ms: f64) -> f64 {
    let num_samples = 1000;
    let dt_step = window_ms / num_samples as f64;

    let mut balance = 0.0;

    // Integrate LTP side
    for i in 1..=num_samples {
        let dt = i as f64 * dt_step;
        balance += fibonacci_stdp_weight_change(dt) * dt_step;
    }

    // Integrate LTD side
    for i in 1..=num_samples {
        let dt = -(i as f64 * dt_step);
        balance += fibonacci_stdp_weight_change(dt) * dt_step;
    }

    balance
}

/// Get the effective time constant for multi-scale STDP
///
/// Computes the weighted average time constant based on amplitude decay.
///
/// # Returns
/// Effective τ_eff in milliseconds
pub fn effective_time_constant() -> f64 {
    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;

    for i in 0..NUM_SCALES {
        let weight = PHI_INV.powi(i as i32);
        weighted_sum += weight * FIBONACCI_TAU[i];
        total_weight += weight;
    }

    weighted_sum / total_weight
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_EPSILON: f64 = 1e-6;

    // ------------------------------------------------------------------------
    // Basic STDP Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_fibonacci_stdp_ltp() {
        // LTP for pre→post (Δt > 0)
        let dw = fibonacci_stdp_weight_change(10.0);

        assert!(
            dw > 0.0,
            "LTP should produce positive weight change, got {}",
            dw
        );

        // Check it's bounded
        assert!(
            dw <= 1.0,
            "Weight change should be ≤ 1.0, got {}",
            dw
        );
    }

    #[test]
    fn test_fibonacci_stdp_ltd() {
        // LTD for post→pre (Δt < 0)
        let dw = fibonacci_stdp_weight_change(-10.0);

        assert!(
            dw < 0.0,
            "LTD should produce negative weight change, got {}",
            dw
        );

        // Check it's bounded
        assert!(
            dw >= -1.0,
            "Weight change should be ≥ -1.0, got {}",
            dw
        );
    }

    #[test]
    fn test_fibonacci_stdp_symmetry() {
        // STDP window should be asymmetric (LTD typically stronger)
        let ltp = fibonacci_stdp_weight_change(10.0);
        let ltd = fibonacci_stdp_weight_change(-10.0);

        // Both should have non-zero magnitude
        assert!(ltp.abs() > TEST_EPSILON);
        assert!(ltd.abs() > TEST_EPSILON);

        // LTD amplitude should be slightly larger (homeostatic)
        // |LTD| > |LTP| due to STDP_A_MINUS > STDP_A_PLUS
        assert!(
            ltd.abs() > ltp.abs(),
            "LTD magnitude should exceed LTP for homeostasis"
        );
    }

    #[test]
    fn test_multi_scale_contribution() {
        // All 5 time scales should contribute
        // Test by comparing single-scale vs multi-scale

        let delta_t = 20.0;
        let multi_scale_dw = fibonacci_stdp_weight_change(delta_t);

        // Single-scale contribution (just fastest)
        let single_scale_dw = STDP_A_PLUS * (-delta_t / FIBONACCI_TAU[0]).exp();

        // Multi-scale should be larger (accumulates contributions)
        assert!(
            multi_scale_dw > single_scale_dw,
            "Multi-scale should accumulate contributions from all time scales"
        );
    }

    #[test]
    fn test_amplitude_golden_decay() {
        // Verify amplitudes follow φ⁻ⁱ decay
        let stdp = FibonacciSTDP::new();

        for i in 0..NUM_SCALES - 1 {
            let ratio = stdp.amplitudes[i + 1] / stdp.amplitudes[i];

            assert!(
                (ratio - PHI_INV).abs() < TEST_EPSILON,
                "Amplitude ratio should equal φ⁻¹, got {} at scale {}",
                ratio,
                i
            );
        }
    }

    #[test]
    fn test_weight_bounds() {
        // Weights should stay in [-1, 1]
        let stdp = FibonacciSTDP::new();

        // Test extreme timing differences
        let extreme_times = [-200.0, -100.0, -50.0, 0.0, 50.0, 100.0, 200.0];

        for &dt in &extreme_times {
            let dw = stdp.compute_weight_change(dt);

            assert!(
                dw >= -1.0 && dw <= 1.0,
                "Weight change {} out of bounds for Δt = {}",
                dw,
                dt
            );
        }
    }

    #[test]
    fn test_pentagon_coupling_update() {
        // Test inter-engine STDP learning
        let stdp = FibonacciSTDP::new();

        // Initial coupling matrix (copy from constants)
        use super::super::constants::FIBONACCI_COUPLING;
        let mut couplings = FIBONACCI_COUPLING.clone();

        // Simulate correlated spike times (μs)
        // Engine 0 spikes at t=1000, 2000, 3000
        // Engine 1 spikes at t=1050, 2050, 3050 (50μs = 0.05ms after)
        let spike_times = [
            [1000, 2000, 3000, 0, 0],
            [1050, 2050, 3050, 0, 0],
            [1100, 2100, 3100, 0, 0],
            [1200, 2200, 3200, 0, 0],
            [1300, 2300, 3300, 0, 0],
        ];

        let initial_coupling = couplings[0][1];

        stdp.update_pentagon_couplings(&spike_times, &mut couplings);

        let updated_coupling = couplings[0][1];

        // Coupling should change due to consistent timing
        assert!(
            (updated_coupling - initial_coupling).abs() > TEST_EPSILON,
            "Pentagon coupling should update based on spike correlations"
        );

        // Updated coupling should still be in bounds
        assert!(
            updated_coupling >= stdp.weight_bounds.0
            && updated_coupling <= stdp.weight_bounds.1,
            "Updated coupling out of bounds: {}",
            updated_coupling
        );
    }

    #[test]
    fn test_learning_window_shape() {
        // Test that learning window has correct shape
        let stdp = FibonacciSTDP::new();

        let dt_range: Vec<f64> = (-100..=100).map(|x| x as f64).collect();
        let (ltp_curve, ltd_curve) = stdp.learning_window(&dt_range);

        // Check lengths
        assert_eq!(ltp_curve.len(), dt_range.len());
        assert_eq!(ltd_curve.len(), dt_range.len());

        // LTP should be non-zero for positive Δt
        let ltp_positive: Vec<_> = dt_range.iter()
            .zip(&ltp_curve)
            .filter(|(&dt, &dw)| dt > 0.0 && dw > TEST_EPSILON)
            .collect();

        assert!(
            ltp_positive.len() > 0,
            "LTP curve should have positive values for Δt > 0"
        );

        // LTD should be non-zero for negative Δt
        let ltd_negative: Vec<_> = dt_range.iter()
            .zip(&ltd_curve)
            .filter(|(&dt, &dw)| dt < 0.0 && dw > TEST_EPSILON)
            .collect();

        assert!(
            ltd_negative.len() > 0,
            "LTD curve should have positive values for Δt < 0"
        );
    }

    // ------------------------------------------------------------------------
    // Advanced Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_stdp_decay_rates() {
        // Verify that weight change decays with |Δt|
        let times = [5.0, 10.0, 20.0, 40.0, 80.0];
        let mut prev_dw = f64::INFINITY;

        for &dt in &times {
            let dw = fibonacci_stdp_weight_change(dt);

            assert!(
                dw < prev_dw,
                "Weight change should decay with increasing |Δt|"
            );

            prev_dw = dw;
        }
    }

    #[test]
    fn test_homeostatic_balance() {
        // Net LTP/LTD balance should slightly favor depression
        let balance = compute_stdp_balance(100.0);

        assert!(
            balance < 0.0,
            "STDP balance should favor LTD for homeostasis, got {}",
            balance
        );
    }

    #[test]
    fn test_effective_time_constant() {
        // Effective time constant should be between min and max Fibonacci tau
        let tau_eff = effective_time_constant();

        let min_tau = FIBONACCI_TAU.iter().copied().fold(f64::INFINITY, f64::min);
        let max_tau = FIBONACCI_TAU.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        assert!(
            tau_eff >= min_tau && tau_eff <= max_tau,
            "Effective τ = {} should be in range [{}, {}]",
            tau_eff,
            min_tau,
            max_tau
        );
    }

    #[test]
    fn test_custom_learning_rate() {
        // Test custom learning rate scaling
        let lr = 0.05;
        let stdp = FibonacciSTDP::with_learning_rate(lr);

        assert_eq!(stdp.learning_rate, lr);

        // Weight change should scale with learning rate
        let dt = 10.0;
        let raw_dw = fibonacci_stdp_weight_change(dt);
        let scaled_dw = stdp.compute_weight_change(dt);

        assert!(
            (scaled_dw - lr * raw_dw).abs() < TEST_EPSILON,
            "Weight change should scale with learning rate"
        );
    }

    #[test]
    fn test_weight_update_bounds() {
        // Test that weight updates respect bounds
        let stdp = FibonacciSTDP::new();

        // Try to push weight beyond upper bound
        let high_weight = 0.95;
        let updated = stdp.update_weight(high_weight, 10.0);

        assert!(
            updated <= stdp.weight_bounds.1,
            "Weight should not exceed upper bound"
        );

        // Try to push weight beyond lower bound
        let low_weight = -0.95;
        let updated = stdp.update_weight(low_weight, -10.0);

        assert!(
            updated >= stdp.weight_bounds.0,
            "Weight should not exceed lower bound"
        );
    }

    #[test]
    fn test_zero_delta_t() {
        // Simultaneous spikes should produce no change
        let dw = fibonacci_stdp_weight_change(0.0);

        // Should be exactly zero
        assert_eq!(
            dw, 0.0,
            "Simultaneous spikes should produce zero weight change, got {}",
            dw
        );
    }

    #[test]
    fn test_fibonacci_stdp_default() {
        // Test default constructor
        let stdp1 = FibonacciSTDP::new();
        let stdp2 = FibonacciSTDP::default();

        assert_eq!(stdp1.learning_rate, stdp2.learning_rate);
        assert_eq!(stdp1.weight_bounds, stdp2.weight_bounds);

        for i in 0..NUM_SCALES {
            assert_eq!(stdp1.tau[i], stdp2.tau[i]);
            assert_eq!(stdp1.amplitudes[i], stdp2.amplitudes[i]);
        }
    }
}
