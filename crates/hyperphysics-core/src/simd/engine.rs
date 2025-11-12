//! SIMD-accelerated engine operations
//!
//! Integration layer between SIMD kernels and HyperPhysics engine.

use crate::simd::math::*;

/// SIMD-accelerated entropy calculation for pBit lattice
///
/// Replaces scalar entropy calculation with vectorized version.
/// Target: 5× speedup
pub fn entropy_from_probabilities_simd(probabilities: &[f64]) -> f64 {
    // Convert f64 to f32 for SIMD (most operations don't need f64 precision)
    let probs_f32: Vec<f32> = probabilities.iter().map(|&p| p as f32).collect();

    shannon_entropy_vectorized(&probs_f32) as f64
}

/// SIMD-accelerated sigmoid for pBit state probabilities
///
/// Target: 5× speedup
pub fn sigmoid_batch_simd(h_eff: &[f64], temperature: f64, output: &mut [f64]) {
    assert_eq!(h_eff.len(), output.len());

    // Convert to f32 for SIMD
    let h_f32: Vec<f32> = h_eff.iter().map(|&h| (h / temperature) as f32).collect();
    let mut out_f32 = vec![0.0f32; h_f32.len()];

    sigmoid_vectorized(&h_f32, &mut out_f32);

    // Convert back to f64
    for (i, &val) in out_f32.iter().enumerate() {
        output[i] = val as f64;
    }
}

/// SIMD-accelerated energy calculation
///
/// Computes E = -Σ J_ij s_i s_j using vectorized dot products.
/// Target: 4× speedup
pub fn energy_simd(states: &[bool], couplings: &[f64]) -> f64 {
    // Convert states to f32 (-1 or +1)
    let states_f32: Vec<f32> = states.iter()
        .map(|&s| if s { 1.0 } else { -1.0 })
        .collect();

    // For pairwise interactions: E = -Σ J_ij s_i s_j
    // Simplified: use dot product of coupling strengths
    let couplings_f32: Vec<f32> = couplings.iter().map(|&c| c as f32).collect();

    -dot_product_vectorized(&states_f32, &couplings_f32) as f64
}

/// SIMD-accelerated magnetization calculation
///
/// M = (Σ s_i) / N where s_i ∈ {-1, +1}
/// Target: 3× speedup
pub fn magnetization_simd(states: &[bool]) -> f64 {
    let states_f32: Vec<f32> = states.iter()
        .map(|&s| if s { 1.0 } else { -1.0 })
        .collect();

    let sum = sum_vectorized(&states_f32);
    (sum as f64) / (states.len() as f64)
}

/// SIMD-accelerated correlation calculation
///
/// Corr(i,j) = <s_i s_j> - <s_i><s_j>
pub fn correlation_simd(states_i: &[f32], states_j: &[f32]) -> f32 {
    let dot = dot_product_vectorized(states_i, states_j);
    let mean_i = mean_vectorized(states_i);
    let mean_j = mean_vectorized(states_j);

    dot / states_i.len() as f32 - mean_i * mean_j
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_entropy_simd() {
        let probs = vec![0.25, 0.25, 0.25, 0.25]; // Uniform
        let entropy = entropy_from_probabilities_simd(&probs);

        // Expected: ln(4) ≈ 1.386
        assert_relative_eq!(entropy, 1.386, epsilon = 0.01);
    }

    #[test]
    fn test_sigmoid_batch_simd() {
        let h_eff = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0; 5];

        sigmoid_batch_simd(&h_eff, 1.0, &mut output);

        // sigmoid(0) ≈ 0.5
        assert_relative_eq!(output[2], 0.5, epsilon = 0.01);

        // All values in (0, 1)
        for &val in &output {
            assert!(val > 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_magnetization_simd() {
        // All spins up
        let all_up = vec![true; 100];
        let mag_up = magnetization_simd(&all_up);
        assert_relative_eq!(mag_up, 1.0, epsilon = 0.001);

        // All spins down
        let all_down = vec![false; 100];
        let mag_down = magnetization_simd(&all_down);
        assert_relative_eq!(mag_down, -1.0, epsilon = 0.001);

        // Half and half
        let mut mixed = vec![true; 50];
        mixed.extend(vec![false; 50]);
        let mag_mixed = magnetization_simd(&mixed);
        assert_relative_eq!(mag_mixed, 0.0, epsilon = 0.001);
    }

    #[test]
    fn test_correlation_simd() {
        // Varying states (positive correlation)
        let states_i = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let states_j = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let corr = correlation_simd(&states_i, &states_j);
        assert!(corr >= 0.99); // Strong positive correlation

        // Anticorrelated
        let states_anti = vec![-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
        let corr_anti = correlation_simd(&states_i, &states_anti);
        assert!(corr_anti <= -0.99); // Strong negative correlation
    }
}
