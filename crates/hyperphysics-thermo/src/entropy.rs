//! Gibbs entropy and negentropy calculations

use crate::{BOLTZMANN_CONSTANT, LN_2};
use hyperphysics_pbit::PBitLattice;

/// Entropy calculator using Gibbs formulation
///
/// S = -k_B Σ P(s) ln P(s)
pub struct EntropyCalculator {
    boltzmann_constant: f64,
}

impl EntropyCalculator {
    /// Create new entropy calculator
    pub fn new() -> Self {
        Self {
            boltzmann_constant: BOLTZMANN_CONSTANT,
        }
    }

    /// Calculate Gibbs entropy from probability distribution
    ///
    /// S = -k_B Σ P(s) ln P(s)
    pub fn gibbs_entropy(&self, probabilities: &[f64]) -> f64 {
        -self.boltzmann_constant
            * probabilities
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| p * p.ln())
                .sum::<f64>()
    }

    /// Calculate maximum entropy for N pBits
    ///
    /// S_max = k_B N ln(2)
    pub fn max_entropy(&self, num_pbits: usize) -> f64 {
        self.boltzmann_constant * (num_pbits as f64) * LN_2
    }

    /// Calculate negentropy (information content)
    ///
    /// S_neg = S_max - S
    pub fn negentropy(&self, entropy: f64, num_pbits: usize) -> f64 {
        self.max_entropy(num_pbits) - entropy
    }

    /// Estimate entropy from pBit probabilities (independent approximation)
    ///
    /// Assumes pBits are independent: S ≈ -k_B Σ_i [p_i ln p_i + (1-p_i) ln(1-p_i)]
    pub fn entropy_from_pbits(&self, lattice: &PBitLattice) -> f64 {
        let entropy_per_bit: f64 = lattice
            .pbits()
            .iter()
            .map(|pbit| {
                let p1 = pbit.prob_one();
                let p0 = 1.0 - p1;
                let mut s = 0.0;
                if p1 > 0.0 && p1 < 1.0 {
                    s -= p1 * p1.ln();
                }
                if p0 > 0.0 && p0 < 1.0 {
                    s -= p0 * p0.ln();
                }
                s
            })
            .sum();

        self.boltzmann_constant * entropy_per_bit
    }

    /// Calculate Shannon entropy (dimensionless, in bits)
    ///
    /// H = -Σ p_i log_2(p_i)
    pub fn shannon_entropy(&self, probabilities: &[f64]) -> f64 {
        -probabilities
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.log2())
            .sum::<f64>()
    }

    /// Estimate entropy rate (change per time step)
    pub fn entropy_rate(
        &self,
        current_entropy: f64,
        previous_entropy: f64,
        dt: f64,
    ) -> f64 {
        if dt > 0.0 {
            (current_entropy - previous_entropy) / dt
        } else {
            0.0
        }
    }

    /// Calculate entropy production rate (always non-negative by second law)
    pub fn entropy_production(&self, delta_s: f64, dt: f64) -> f64 {
        if dt > 0.0 {
            delta_s / dt
        } else {
            0.0
        }
    }

    /// Verify second law: ΔS ≥ 0 for isolated system
    pub fn verify_second_law(&self, delta_s: f64, tolerance: f64) -> bool {
        delta_s >= -tolerance
    }
}

impl Default for EntropyCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_entropy() {
        let calc = EntropyCalculator::new();

        // One bit: S_max = k_B ln(2)
        let s_max_1 = calc.max_entropy(1);
        assert!((s_max_1 - BOLTZMANN_CONSTANT * LN_2).abs() < 1e-30);

        // N bits: S_max = k_B N ln(2)
        let s_max_48 = calc.max_entropy(48);
        assert!((s_max_48 - 48.0 * BOLTZMANN_CONSTANT * LN_2).abs() < 1e-30);
    }

    #[test]
    fn test_uniform_distribution_max_entropy() {
        let calc = EntropyCalculator::new();

        // Uniform distribution has maximum entropy
        let probs = vec![0.5, 0.5];
        let s = calc.gibbs_entropy(&probs);
        let s_max = calc.max_entropy(1);

        assert!((s - s_max).abs() < 1e-25);
    }

    #[test]
    fn test_deterministic_zero_entropy() {
        let calc = EntropyCalculator::new();

        // Deterministic: p=1 or p=0 gives S=0
        let probs_certain = vec![1.0, 0.0];
        let s = calc.gibbs_entropy(&probs_certain);

        assert!(s.abs() < 1e-25);
    }

    #[test]
    fn test_shannon_entropy() {
        let calc = EntropyCalculator::new();

        // Fair coin: H = 1 bit
        let probs = vec![0.5, 0.5];
        let h = calc.shannon_entropy(&probs);

        assert!((h - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_negentropy() {
        let calc = EntropyCalculator::new();

        let s_max = calc.max_entropy(10);
        let s = s_max * 0.5; // Half of maximum

        let neg = calc.negentropy(s, 10);
        assert!((neg - s_max * 0.5).abs() < 1e-25);
    }

    #[test]
    fn test_entropy_from_lattice() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let calc = EntropyCalculator::new();

        let s = calc.entropy_from_pbits(&lattice);

        // Should be close to maximum for uniform probabilities
        let s_max = calc.max_entropy(lattice.size());
        assert!(s > 0.0);
        assert!(s <= s_max + 1e-20); // Allow tiny numerical error
    }

    #[test]
    fn test_second_law_verification() {
        let calc = EntropyCalculator::new();

        assert!(calc.verify_second_law(0.1, 1e-10)); // ΔS > 0: OK
        assert!(calc.verify_second_law(0.0, 1e-10)); // ΔS = 0: OK
        assert!(!calc.verify_second_law(-0.1, 1e-10)); // ΔS < 0: Violation
    }
}
