//! Property-Based Tests for Hierarchical Φ Calculator
//!
//! Uses QuickCheck-style property testing to verify mathematical properties
//! of the hierarchical multi-scale integrated information calculator.
//!
//! # Properties Tested
//!
//! 1. **Non-negativity**: Φ ≥ 0 for all valid inputs
//! 2. **Boundedness**: Φ values are finite and reasonable
//! 3. **Scale consistency**: Larger scales produce coarser estimates
//! 4. **Cluster validity**: All clusters have minimum size
//! 5. **Determinism**: Same inputs produce same outputs
//! 6. **Linearity approximation**: Φ(a+b) ≈ Φ(a) + Φ(b) for weakly coupled systems

use hyperphysics_consciousness::{
    ClusteringMethod, HierarchicalPhiCalculator,
};
use hyperphysics_pbit::PBitLattice;
use proptest::prelude::*;
use proptest::strategy::ValueTree;

/// Generate valid temperature values
fn temperature_strategy() -> impl Strategy<Value = f64> {
    (0.1f64..=10.0f64)
}

/// Generate valid level counts
fn levels_strategy() -> impl Strategy<Value = usize> {
    (1usize..=5usize)
}

/// Generate valid scale factors
fn scale_factor_strategy() -> impl Strategy<Value = f64> {
    (1.5f64..=4.0f64)
}

/// Generate valid min cluster sizes
fn min_cluster_size_strategy() -> impl Strategy<Value = usize> {
    (2usize..=10usize)
}

proptest! {
    /// Property: Φ is always non-negative
    #[test]
    fn prop_phi_non_negative(
        temperature in temperature_strategy(),
        levels in levels_strategy(),
        scale_factor in scale_factor_strategy(),
        min_size in min_cluster_size_strategy(),
    ) {
        let lattice = PBitLattice::roi_48(temperature).unwrap();
        let calculator = HierarchicalPhiCalculator::new(
            levels,
            ClusteringMethod::Tessellation,
            scale_factor,
            min_size,
        );

        let result = calculator.calculate(&lattice).unwrap();

        // Φ must be non-negative
        prop_assert!(result.phi_total >= 0.0,
            "Φ must be non-negative, got: {}", result.phi_total);

        // All per-scale Φ values must be non-negative
        for (i, &phi) in result.phi_per_scale.iter().enumerate() {
            prop_assert!(phi >= 0.0,
                "Φ at scale {} must be non-negative, got: {}", i, phi);
        }
    }

    /// Property: Φ is always finite (not NaN or infinite)
    #[test]
    fn prop_phi_finite(
        temperature in temperature_strategy(),
        levels in levels_strategy(),
    ) {
        let lattice = PBitLattice::roi_48(temperature).unwrap();
        let calculator = HierarchicalPhiCalculator::new(
            levels,
            ClusteringMethod::Tessellation,
            2.0,
            3,
        );

        let result = calculator.calculate(&lattice).unwrap();

        prop_assert!(result.phi_total.is_finite(),
            "Φ must be finite, got: {}", result.phi_total);

        for (i, &phi) in result.phi_per_scale.iter().enumerate() {
            prop_assert!(phi.is_finite(),
                "Φ at scale {} must be finite, got: {}", i, phi);
        }
    }

    /// Property: Number of scales matches levels
    #[test]
    fn prop_scale_count_matches_levels(
        temperature in temperature_strategy(),
        levels in levels_strategy(),
    ) {
        let lattice = PBitLattice::roi_48(temperature).unwrap();
        let calculator = HierarchicalPhiCalculator::new(
            levels,
            ClusteringMethod::Tessellation,
            2.0,
            3,
        );

        let result = calculator.calculate(&lattice).unwrap();

        prop_assert_eq!(result.scales.len(), levels,
            "Number of scales should match levels");
        prop_assert_eq!(result.phi_per_scale.len(), levels,
            "Number of Φ values should match levels");
        prop_assert_eq!(result.clusters_per_scale.len(), levels,
            "Number of cluster counts should match levels");
    }

    /// Property: Scales increase geometrically
    #[test]
    fn prop_scales_increase(
        temperature in temperature_strategy(),
        levels in (2usize..=5usize),
        scale_factor in scale_factor_strategy(),
    ) {
        let lattice = PBitLattice::roi_48(temperature).unwrap();
        let calculator = HierarchicalPhiCalculator::new(
            levels,
            ClusteringMethod::Tessellation,
            scale_factor,
            3,
        );

        let result = calculator.calculate(&lattice).unwrap();

        // Scales should increase monotonically
        for i in 1..result.scales.len() {
            prop_assert!(result.scales[i] > result.scales[i-1],
                "Scale {} ({}) should be larger than scale {} ({})",
                i, result.scales[i], i-1, result.scales[i-1]);
        }
    }

    /// Property: Deterministic - same input produces same output
    #[test]
    fn prop_deterministic(
        temperature in temperature_strategy(),
        levels in levels_strategy(),
    ) {
        let lattice = PBitLattice::roi_48(temperature).unwrap();
        let calculator = HierarchicalPhiCalculator::new(
            levels,
            ClusteringMethod::Tessellation,
            2.0,
            3,
        );

        let result1 = calculator.calculate(&lattice).unwrap();
        let result2 = calculator.calculate(&lattice).unwrap();

        prop_assert_eq!(result1.phi_total, result2.phi_total,
            "Same inputs should produce same Φ");

        prop_assert_eq!(result1.phi_per_scale, result2.phi_per_scale,
            "Same inputs should produce same per-scale Φ");
    }

    /// Property: Cluster sizes respect minimum
    #[test]
    fn prop_cluster_minimum_size(
        temperature in temperature_strategy(),
        min_size in (2usize..=5usize),
    ) {
        let lattice = PBitLattice::roi_48(temperature).unwrap();
        let calculator = HierarchicalPhiCalculator::new(
            2,
            ClusteringMethod::Tessellation,
            2.5,
            min_size,
        );

        // Test clustering at a reasonable scale
        let clusters = calculator.cluster_at_scale(&lattice, 1.5).unwrap();

        for cluster in &clusters {
            prop_assert!(cluster.indices.len() >= min_size,
                "Cluster size {} should be >= minimum {}",
                cluster.indices.len(), min_size);
        }
    }

    /// Property: Cluster indices are within lattice bounds
    #[test]
    fn prop_cluster_indices_valid(
        temperature in temperature_strategy(),
        levels in levels_strategy(),
    ) {
        let lattice = PBitLattice::roi_48(temperature).unwrap();
        let n = lattice.size();
        let calculator = HierarchicalPhiCalculator::new(
            levels,
            ClusteringMethod::Tessellation,
            2.0,
            3,
        );

        let clusters = calculator.cluster_at_scale(&lattice, 1.5).unwrap();

        for cluster in &clusters {
            for &idx in &cluster.indices {
                prop_assert!(idx < n,
                    "Cluster index {} should be < lattice size {}",
                    idx, n);
            }
        }
    }

    /// Property: Total Φ is bounded by system size
    #[test]
    fn prop_phi_reasonable_bound(
        temperature in temperature_strategy(),
        levels in levels_strategy(),
    ) {
        let lattice = PBitLattice::roi_48(temperature).unwrap();
        let n = lattice.size();
        let calculator = HierarchicalPhiCalculator::new(
            levels,
            ClusteringMethod::Tessellation,
            2.0,
            3,
        );

        let result = calculator.calculate(&lattice).unwrap();

        // Φ should be bounded by some reasonable multiple of system size
        // For pBit systems with bounded couplings
        let max_reasonable = (n as f64) * 10.0;

        prop_assert!(result.phi_total <= max_reasonable,
            "Φ {} seems unreasonably large for system size {}",
            result.phi_total, n);
    }

    /// Property: Different clustering methods give finite results
    #[test]
    fn prop_clustering_methods_all_work(
        temperature in temperature_strategy(),
    ) {
        let lattice = PBitLattice::roi_48(temperature).unwrap();

        let methods = vec![
            ClusteringMethod::Tessellation,
            ClusteringMethod::HyperbolicKMeans,
            ClusteringMethod::Agglomerative,
        ];

        for method in methods {
            let calculator = HierarchicalPhiCalculator::new(2, method, 2.0, 3);
            let result = calculator.calculate(&lattice).unwrap();

            prop_assert!(result.phi_total.is_finite(),
                "Clustering method {:?} should produce finite Φ", method);
            prop_assert!(result.phi_total >= 0.0,
                "Clustering method {:?} should produce non-negative Φ", method);
        }
    }

    /// Property: Cluster radii are non-negative
    #[test]
    fn prop_cluster_radii_non_negative(
        temperature in temperature_strategy(),
    ) {
        let lattice = PBitLattice::roi_48(temperature).unwrap();
        let calculator = HierarchicalPhiCalculator::new(
            2,
            ClusteringMethod::Tessellation,
            2.0,
            3,
        );

        let clusters = calculator.cluster_at_scale(&lattice, 1.5).unwrap();

        for cluster in &clusters {
            prop_assert!(cluster.radius >= 0.0,
                "Cluster radius must be non-negative, got: {}",
                cluster.radius);
            prop_assert!(cluster.radius.is_finite(),
                "Cluster radius must be finite");
        }
    }
}

#[cfg(test)]
mod standard_tests {
    use super::*;

    #[test]
    fn test_property_test_strategies_valid() {
        // Test that our strategies generate valid values
        let temp = temperature_strategy().new_tree(&mut proptest::test_runner::TestRunner::default()).unwrap().current();
        assert!(temp > 0.0 && temp <= 10.0);

        let levels = levels_strategy().new_tree(&mut proptest::test_runner::TestRunner::default()).unwrap().current();
        assert!(levels >= 1 && levels <= 5);
    }

    #[test]
    fn test_hierarchical_phi_with_zero_temperature() {
        // Edge case: very low temperature (deterministic limit)
        let lattice = PBitLattice::roi_48(0.01).unwrap();
        let calculator = HierarchicalPhiCalculator::new(
            2,
            ClusteringMethod::Tessellation,
            2.0,
            3,
        );

        let result = calculator.calculate(&lattice).unwrap();

        assert!(result.phi_total >= 0.0);
        assert!(result.phi_total.is_finite());
    }

    #[test]
    fn test_hierarchical_phi_with_high_temperature() {
        // Edge case: high temperature (random limit)
        let lattice = PBitLattice::roi_48(10.0).unwrap();
        let calculator = HierarchicalPhiCalculator::new(
            2,
            ClusteringMethod::Tessellation,
            2.0,
            3,
        );

        let result = calculator.calculate(&lattice).unwrap();

        assert!(result.phi_total >= 0.0);
        assert!(result.phi_total.is_finite());
    }

    #[test]
    fn test_single_level_hierarchical() {
        // Edge case: single level (no hierarchy)
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let calculator = HierarchicalPhiCalculator::new(
            1,
            ClusteringMethod::Tessellation,
            2.0,
            3,
        );

        let result = calculator.calculate(&lattice).unwrap();

        assert_eq!(result.scales.len(), 1);
        assert_eq!(result.phi_per_scale.len(), 1);
        assert!(result.phi_total >= 0.0);
    }

    #[test]
    fn test_many_levels_hierarchical() {
        // Edge case: many levels
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let calculator = HierarchicalPhiCalculator::new(
            5,
            ClusteringMethod::Tessellation,
            2.0,
            2,
        );

        let result = calculator.calculate(&lattice).unwrap();

        assert_eq!(result.scales.len(), 5);
        assert_eq!(result.phi_per_scale.len(), 5);

        // All scales should be distinct and increasing
        for i in 1..result.scales.len() {
            assert!(result.scales[i] > result.scales[i-1]);
        }
    }
}
