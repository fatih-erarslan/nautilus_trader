//! Mathematical properties to verify

use crate::VerificationResult;

/// Verify probability is in [0,1]
pub fn verify_probability_bounds_pure(p: f64) -> VerificationResult {
    if (0.0..=1.0).contains(&p) && p.is_finite() {
        VerificationResult::Verified
    } else {
        VerificationResult::Violated(format!(
            "Probability {} outside valid range [0,1]",
            p
        ))
    }
}

/// Verify second law: entropy never decreases
pub fn verify_second_law(delta_s: f64, tolerance: f64) -> VerificationResult {
    if delta_s >= -tolerance {
        VerificationResult::Verified
    } else {
        VerificationResult::Violated(format!(
            "Second law violation: ΔS = {} < 0 (tolerance: {})",
            delta_s, tolerance
        ))
    }
}

/// Verify Landauer bound: E ≥ k_B T ln(2) per bit erasure
pub fn verify_landauer_bound(
    energy: f64,
    erasures: usize,
    temperature: f64,
) -> VerificationResult {
    const K_B: f64 = 1.380649e-23; // Boltzmann constant
    let min_energy = erasures as f64 * K_B * temperature * 2.0_f64.ln();

    if energy >= min_energy {
        VerificationResult::Verified
    } else {
        VerificationResult::Violated(format!(
            "Landauer bound violation: E = {} < {} (min for {} erasures at T={}K)",
            energy, min_energy, erasures, temperature
        ))
    }
}

/// Verify energy is finite and real
pub fn verify_energy_bounds(energy: f64) -> VerificationResult {
    if energy.is_finite() {
        VerificationResult::Verified
    } else {
        VerificationResult::Violated(format!("Energy {} is not finite", energy))
    }
}

/// Verify Φ (integrated information) is non-negative
pub fn verify_phi_nonnegative(phi: f64) -> VerificationResult {
    if phi >= 0.0 && phi.is_finite() {
        VerificationResult::Verified
    } else {
        VerificationResult::Violated(format!(
            "Φ = {} violates non-negativity",
            phi
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probability_bounds() {
        assert!(verify_probability_bounds_pure(0.5).is_verified());
        assert!(verify_probability_bounds_pure(0.0).is_verified());
        assert!(verify_probability_bounds_pure(1.0).is_verified());

        assert!(!verify_probability_bounds_pure(-0.1).is_verified());
        assert!(!verify_probability_bounds_pure(1.1).is_verified());
        assert!(!verify_probability_bounds_pure(f64::NAN).is_verified());
    }

    #[test]
    fn test_second_law() {
        assert!(verify_second_law(0.1, 1e-10).is_verified());
        assert!(verify_second_law(0.0, 1e-10).is_verified());
        assert!(verify_second_law(-1e-11, 1e-10).is_verified()); // Within tolerance

        assert!(!verify_second_law(-0.1, 1e-10).is_verified());
    }

    #[test]
    fn test_landauer_bound() {
        const K_B: f64 = 1.380649e-23;
        let temp = 300.0;
        let min_energy = K_B * temp * 2.0_f64.ln();

        assert!(verify_landauer_bound(min_energy, 1, temp).is_verified());
        assert!(verify_landauer_bound(min_energy * 2.0, 1, temp).is_verified());

        assert!(!verify_landauer_bound(min_energy * 0.5, 1, temp).is_verified());
    }

    #[test]
    fn test_phi_nonnegative() {
        assert!(verify_phi_nonnegative(0.0).is_verified());
        assert!(verify_phi_nonnegative(1.5).is_verified());

        assert!(!verify_phi_nonnegative(-0.1).is_verified());
        assert!(!verify_phi_nonnegative(f64::INFINITY).is_verified());
    }
}
