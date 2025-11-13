//! Comprehensive tests for Φ non-negativity invariant
//!
//! Reference: Tononi et al. (2016) "Integrated information theory" Nat Rev Neurosci 17:450
//! IIT 3.0 Axiom: Φ ≥ 0 (integrated information is always non-negative)

use hyperphysics_consciousness::{PhiCalculator, PhiApproximation};
use hyperphysics_pbit::PBitLattice;

#[test]
fn test_phi_nonnegative_disconnected_system() {
    // Disconnected system should have Φ = 0
    let lattice = PBitLattice::new(4);
    let calculator = PhiCalculator::exact();

    let result = calculator.calculate(&lattice).unwrap();

    assert!(result.phi >= 0.0, "Φ must be non-negative for disconnected system");
    assert!(result.phi.abs() < 1e-10, "Disconnected system should have Φ ≈ 0");
}

#[test]
fn test_phi_nonnegative_weakly_coupled() {
    // Weakly coupled system
    let mut lattice = PBitLattice::new(6);
    lattice.set_uniform_coupling(0.1);

    let calculator = PhiCalculator::greedy();
    let result = calculator.calculate(&lattice).unwrap();

    assert!(result.phi >= 0.0, "Φ must be non-negative for weakly coupled system");
    assert!(result.phi.is_finite(), "Φ must be finite");
}

#[test]
fn test_phi_nonnegative_strongly_coupled() {
    // Strongly coupled system
    let mut lattice = PBitLattice::new(8);
    lattice.set_uniform_coupling(1.0);

    let calculator = PhiCalculator::greedy();
    let result = calculator.calculate(&lattice).unwrap();

    assert!(result.phi >= 0.0, "Φ must be non-negative for strongly coupled system");
    assert!(result.phi.is_finite(), "Φ must be finite");
}

#[test]
fn test_phi_nonnegative_roi_lattices() {
    // Test all ROI configurations
    let configurations = vec![
        (12, 0.5),
        (24, 1.0),
        (48, 1.0),
    ];

    let calculator = PhiCalculator::greedy();

    for (size, coupling) in configurations {
        let lattice = match size {
            12 => PBitLattice::roi_12(coupling),
            24 => PBitLattice::roi_24(coupling),
            48 => PBitLattice::roi_48(coupling),
            _ => panic!("Invalid size"),
        }.unwrap();

        let result = calculator.calculate(&lattice).unwrap();

        assert!(
            result.phi >= 0.0,
            "Φ must be non-negative for ROI-{} lattice with coupling {}",
            size, coupling
        );
        assert!(result.phi.is_finite(), "Φ must be finite");
    }
}

#[test]
fn test_phi_partition_enumeration() {
    // Test that partition enumeration produces valid results
    let lattice = PBitLattice::new(4);
    let calculator = PhiCalculator::exact();

    let result = calculator.calculate(&lattice).unwrap();

    assert!(result.phi >= 0.0);
    assert!(result.mip.is_some(), "Exact calculation should return MIP");

    let mip = result.mip.unwrap();
    assert!(!mip.subset_a.is_empty(), "MIP subset A should not be empty");
    assert!(!mip.subset_b.is_empty(), "MIP subset B should not be empty");
}

#[test]
fn test_phi_monte_carlo_convergence() {
    // Test that Monte Carlo approximation converges to non-negative values
    let mut lattice = PBitLattice::new(10);
    lattice.set_uniform_coupling(0.5);

    let samples_list = [100, 500, 1000];

    for samples in samples_list {
        let calculator = PhiCalculator::monte_carlo(samples);
        let result = calculator.calculate(&lattice).unwrap();

        assert!(
            result.phi >= 0.0,
            "Monte Carlo Φ with {} samples must be non-negative",
            samples
        );
        assert!(result.phi.is_finite());
    }
}

#[test]
fn test_phi_hierarchical_method() {
    // Test hierarchical method on larger system
    let mut lattice = PBitLattice::new(50);
    lattice.set_uniform_coupling(0.5);

    let calculator = PhiCalculator::hierarchical(3);
    let result = calculator.calculate(&lattice).unwrap();

    assert!(result.phi >= 0.0, "Hierarchical Φ must be non-negative");
    assert!(result.phi.is_finite());
}

#[test]
fn test_phi_mutual_information_bounds() {
    // Test that mutual information calculation is bounded
    use hyperphysics_consciousness::phi::PhiCalculator;

    let mut lattice = PBitLattice::new(6);
    lattice.set_uniform_coupling(1.0);

    // Set deterministic state
    for i in 0..6 {
        lattice.set_state(i, i % 2 == 0);
    }

    let calculator = PhiCalculator::exact();
    let result = calculator.calculate(&lattice).unwrap();

    assert!(result.phi >= 0.0);
    // Mutual information should be bounded by min entropy
    assert!(result.phi <= 6.0 * (2.0_f64).ln(), "Φ should be bounded by system entropy");
}

#[test]
fn test_phi_effective_information_properties() {
    // Test effective information properties
    let mut lattice = PBitLattice::new(8);
    lattice.set_uniform_coupling(0.5);

    let calculator = PhiCalculator::greedy();
    let result = calculator.calculate(&lattice).unwrap();

    assert!(result.phi >= 0.0);

    if let Some(mip) = result.mip {
        // Effective information should equal Φ
        assert!((mip.effective_info - result.phi).abs() < 1e-10);

        // Partitions should be non-empty
        assert!(!mip.subset_a.is_empty());
        assert!(!mip.subset_b.is_empty());

        // Partitions should be disjoint
        for &a in &mip.subset_a {
            assert!(!mip.subset_b.contains(&a), "Partitions must be disjoint");
        }
    }
}
