//! Comprehensive tests for Φ non-negativity invariant
//!
//! Reference: Tononi et al. (2016) "Integrated information theory" Nat Rev Neurosci 17:450
//! IIT 3.0 Axiom: Φ ≥ 0 (integrated information is always non-negative)

use hyperphysics_consciousness::PhiCalculator;
use hyperphysics_pbit::PBitLattice;

/// Helper to create small test lattice with {3,7,1} tessellation (7 nodes)
fn create_test_lattice(temperature: f64) -> PBitLattice {
    // Using {3,7} hyperbolic tessellation with depth 1 (7 nodes)
    PBitLattice::new(3, 7, 1, temperature)
        .expect("Failed to create test lattice")
}

/// Helper to create larger test lattice with {3,7,2} tessellation (48 nodes)
fn create_large_lattice(temperature: f64) -> PBitLattice {
    PBitLattice::roi_48(temperature)
        .expect("Failed to create large lattice")
}

#[test]
fn test_phi_nonnegative_disconnected_system() {
    // Very high temperature = nearly disconnected (random) system, should have low Φ
    let lattice = create_test_lattice(100.0); // High temp = low coupling
    let calculator = PhiCalculator::exact();

    let result = calculator.calculate(&lattice).unwrap();

    assert!(result.phi >= 0.0, "Φ must be non-negative for high-temp system, got {}", result.phi);
    assert!(result.phi.is_finite(), "Φ must be finite");
}

#[test]
fn test_phi_nonnegative_weakly_coupled() {
    // Medium temperature = weakly coupled system
    let lattice = create_test_lattice(10.0);
    let calculator = PhiCalculator::greedy();
    let result = calculator.calculate(&lattice).unwrap();

    assert!(result.phi >= 0.0, "Φ must be non-negative for weakly coupled system, got {}", result.phi);
    assert!(result.phi.is_finite(), "Φ must be finite");
}

#[test]
fn test_phi_nonnegative_strongly_coupled() {
    // Low temperature = strongly coupled system
    let lattice = create_test_lattice(0.1);
    let calculator = PhiCalculator::greedy();
    let result = calculator.calculate(&lattice).unwrap();

    assert!(result.phi >= 0.0, "Φ must be non-negative for strongly coupled system, got {}", result.phi);
    assert!(result.phi.is_finite(), "Φ must be finite");
}

#[test]
fn test_phi_nonnegative_roi_48_lattice() {
    // Test ROI-48 configuration at different temperatures
    let temperatures = [0.1, 1.0, 10.0];
    let calculator = PhiCalculator::greedy();

    for &temp in &temperatures {
        let lattice = create_large_lattice(temp);
        let result = calculator.calculate(&lattice).unwrap();

        assert!(
            result.phi >= 0.0,
            "Φ must be non-negative for ROI-48 lattice at temp={}, got {}",
            temp, result.phi
        );
        assert!(result.phi.is_finite(), "Φ must be finite");
    }
}

#[test]
fn test_phi_partition_enumeration() {
    // Test that partition enumeration produces valid results
    let lattice = create_test_lattice(1.0);
    let calculator = PhiCalculator::exact();

    let result = calculator.calculate(&lattice).unwrap();

    assert!(result.phi >= 0.0, "Φ must be non-negative, got {}", result.phi);
    assert!(result.mip.is_some(), "Exact calculation should return MIP");

    let mip = result.mip.unwrap();
    assert!(!mip.subset_a.is_empty(), "MIP subset A should not be empty");
    assert!(!mip.subset_b.is_empty(), "MIP subset B should not be empty");
}

#[test]
fn test_phi_monte_carlo_convergence() {
    // Test that Monte Carlo approximation converges to non-negative values
    let lattice = create_test_lattice(1.0);
    let samples_list = [100, 500, 1000];

    for samples in samples_list {
        let calculator = PhiCalculator::monte_carlo(samples);
        let result = calculator.calculate(&lattice).unwrap();

        assert!(
            result.phi >= 0.0,
            "Monte Carlo Φ with {} samples must be non-negative, got {}",
            samples, result.phi
        );
        assert!(result.phi.is_finite());
    }
}

#[test]
fn test_phi_hierarchical_method() {
    // Test hierarchical method on ROI-48 system
    let lattice = create_large_lattice(1.0);
    let calculator = PhiCalculator::hierarchical(3);
    let result = calculator.calculate(&lattice).unwrap();

    assert!(result.phi >= 0.0, "Hierarchical Φ must be non-negative, got {}", result.phi);
    assert!(result.phi.is_finite());
}

#[test]
fn test_phi_mutual_information_bounds() {
    // Test that mutual information calculation is bounded
    let lattice = create_test_lattice(0.5);
    let calculator = PhiCalculator::exact();
    let result = calculator.calculate(&lattice).unwrap();

    assert!(result.phi >= 0.0, "Φ must be non-negative, got {}", result.phi);

    // Mutual information should be bounded by system entropy
    let n = lattice.size() as f64;
    let max_entropy = n * (2.0_f64).ln();
    assert!(
        result.phi <= max_entropy,
        "Φ ({}) should be bounded by system entropy ({})",
        result.phi, max_entropy
    );
}

#[test]
fn test_phi_effective_information_properties() {
    // Test effective information properties
    let lattice = create_test_lattice(1.0);
    let calculator = PhiCalculator::greedy();
    let result = calculator.calculate(&lattice).unwrap();

    assert!(result.phi >= 0.0, "Φ must be non-negative, got {}", result.phi);

    if let Some(mip) = result.mip {
        // Effective information should equal Φ
        assert!(
            (mip.effective_info - result.phi).abs() < 1e-10,
            "Effective info ({}) should equal Φ ({})",
            mip.effective_info, result.phi
        );

        // Partitions should be non-empty
        assert!(!mip.subset_a.is_empty(), "MIP subset A should not be empty");
        assert!(!mip.subset_b.is_empty(), "MIP subset B should not be empty");

        // Partitions should be disjoint
        for &a in &mip.subset_a {
            assert!(!mip.subset_b.contains(&a), "Partitions must be disjoint");
        }
    }
}

#[test]
fn test_phi_temperature_dependence() {
    // Verify that Φ behaves sensibly with temperature
    // Lower temperature = stronger coupling = higher Φ (typically)
    let calculator = PhiCalculator::greedy();

    let low_temp = create_test_lattice(0.1);
    let high_temp = create_test_lattice(10.0);

    let phi_low = calculator.calculate(&low_temp).unwrap().phi;
    let phi_high = calculator.calculate(&high_temp).unwrap().phi;

    assert!(phi_low >= 0.0, "Low-temp Φ must be non-negative");
    assert!(phi_high >= 0.0, "High-temp Φ must be non-negative");

    // Both should be non-negative (main invariant)
    // Note: we don't assert phi_low > phi_high because
    // the relationship depends on system specifics
}
