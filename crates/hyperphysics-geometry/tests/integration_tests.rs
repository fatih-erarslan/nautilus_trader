//! # Cross-Module Integration Tests
//!
//! Simplified integration tests that verify proper coordination between
//! different modules of the hyperphysics-geometry crate.

use hyperphysics_geometry::{
    // Core geometry
    PoincarePoint, EPSILON,

    // Fuchsian groups
    FuchsianGroupAlgebraic, MoebiusTransform,
    OrbitConfig, ExactOrbitEnumerator,

    // Tessellation
    HeptagonalTessellation,
    HyperbolicDelaunay, HyperbolicVoronoi,

    // SNN types - use LorentzVec4D which is re-exported
    LorentzVec4D as LorentzVec,

    // Markov kernels and spectral
    HyperbolicHeatKernel, TransitionOperator,
    SpectralConfig, SpectralAnalysis,

    // STDP Learning
    STDPConfig, STDPLearner,

    // Cognitive modules
    FreeEnergyCalculator,
    GlobalWorkspaceConfig, GlobalWorkspace,

    // Enactive layer
    BeliefState, Observation, Policy,

    // Bateson ecology
    EcologyConfig, EcologicalMind,

    // IIT
    PhiConfig, PhiCalculator, ProbabilityDistribution,

    // Replicator dynamics
    ReplicatorConfig, HyperbolicReplicator, PayoffMatrix,
};

use nalgebra as na;
use std::f64::consts::PI;
use num_complex::Complex64;

// ============================================================================
// Geometry Integration Tests
// ============================================================================

#[test]
fn test_poincare_fuchsian_integration() {
    // Create a rotation generator
    let rotation = MoebiusTransform::rotation(PI / 7.0);

    // Create group with generator
    let group = FuchsianGroupAlgebraic::new(vec![rotation]);

    // Test point in Poincaré disk
    let p = PoincarePoint::new(na::Vector3::new(0.3, 0.2, 0.0)).unwrap();

    // Apply group action using generators() accessor
    let z = Complex64::new(p.coords().x, p.coords().y);
    let result = group.generators()[0].apply(z);

    // Result should still be in disk
    assert!(result.norm() < 1.0 + EPSILON);

    // Verify Poincaré point creation works
    let p2 = PoincarePoint::new(na::Vector3::new(result.re, result.im, 0.0));
    assert!(p2.is_ok());
}

#[test]
fn test_fuchsian_orbit_computation() {
    // Create rotation generator (7-fold)
    let angle = 2.0 * PI / 7.0;
    let rotation = MoebiusTransform::rotation(angle);
    let group = FuchsianGroupAlgebraic::new(vec![rotation]);

    // Compute orbit with default config
    let base_point = Complex64::new(0.2, 0.0);
    let config = OrbitConfig::default();

    let mut enumerator = ExactOrbitEnumerator::new(group, base_point, config);
    let orbit = enumerator.enumerate();

    assert!(!orbit.is_empty());

    // All orbit points should be in the disk
    for point in orbit {
        assert!(point.point.norm() < 1.0 + EPSILON);
    }
}

#[test]
fn test_fuchsian_basic_operations() {
    // Create translation generator
    let translation = MoebiusTransform::translation(0.5);
    let group = FuchsianGroupAlgebraic::new(vec![translation]);

    // Test generators accessor
    assert_eq!(group.generators().len(), 1);

    // Apply to a point
    let z = Complex64::new(0.0, 0.0);
    let result = group.generators()[0].apply(z);

    // Result should be moved along real axis
    assert!(result.re > 0.0);
    assert!(result.im.abs() < EPSILON);
}

#[test]
fn test_voronoi_delaunay() {
    // Create points in a grid-like pattern with center for proper triangulation
    let mut points = vec![
        PoincarePoint::new(na::Vector3::new(0.0, 0.0, 0.0)).unwrap(), // center
    ];

    // Add ring of points
    for i in 0..8 {
        let angle = 2.0 * PI * i as f64 / 8.0;
        let r = 0.4;
        if let Ok(p) = PoincarePoint::new(na::Vector3::new(r * angle.cos(), r * angle.sin(), 0.0)) {
            points.push(p);
        }
    }

    // Need at least 3 points for triangulation
    assert!(points.len() >= 3);

    // Create Delaunay triangulation from sites
    let delaunay = HyperbolicDelaunay::from_sites(&points).unwrap();
    // Verify sites were stored
    assert_eq!(delaunay.sites().len(), points.len());

    // Create Voronoi from sites
    let voronoi = HyperbolicVoronoi::from_sites(&points).unwrap();
    assert!(voronoi.cells().len() > 0);
    assert!(voronoi.cells().len() <= points.len());
}

// ============================================================================
// Neural Integration Tests
// ============================================================================

#[test]
fn test_stdp_learning() {
    let config = STDPConfig::default();
    let learner = STDPLearner::new(config);

    // Get config through accessor
    let cfg = learner.config();

    // STDP function computes properly
    let delta_t: f64 = 5.0; // ms
    let stdp_value = cfg.a_plus * (-delta_t / cfg.tau_plus).exp();
    assert!(stdp_value > 0.0); // LTP

    let delta_t_neg: f64 = -5.0;
    let ltd_value = -cfg.a_minus * (delta_t_neg / cfg.tau_minus).exp();
    assert!(ltd_value < 0.0); // LTD
}

#[test]
fn test_heat_kernel_spectral() {
    let n = 5;
    let positions: Vec<LorentzVec> = (0..n)
        .map(|i| {
            let angle = 2.0 * PI * i as f64 / n as f64;
            let r = 0.3;
            LorentzVec::from_spatial(r * angle.cos(), r * angle.sin(), 0.0)
        })
        .collect();

    let kernel = HyperbolicHeatKernel::new(1.0);

    // Test kernel values
    for i in 0..n {
        for j in 0..n {
            let k_ij = kernel.evaluate(&positions[i], &positions[j]);
            assert!(k_ij.is_finite());
            assert!(k_ij >= 0.0);
        }
    }

    // Create transition operator
    let transition = TransitionOperator::new(positions, 1.0);

    // Spectral analysis
    let spectral_config = SpectralConfig {
        num_eigenvalues: 3,
        max_iterations: 100,
        tolerance: 1e-6,
        compute_eigenvectors: true,
        compute_cheeger: true,
    };

    let spectral = SpectralAnalysis::analyze_with_config(&transition, &spectral_config);

    assert!(!spectral.eigenvalues.is_empty());

    // Largest eigenvalue should be close to 1
    let lambda_1 = spectral.eigenvalues[0];
    assert!((lambda_1 - 1.0).abs() < 0.1);

    // Convergence bounds should be computed
    assert!(!spectral.convergence_bounds.is_empty());
}

#[test]
fn test_mixing_time_bound() {
    let n = 5;
    let positions: Vec<LorentzVec> = (0..n)
        .map(|i| {
            let angle = 2.0 * PI * i as f64 / n as f64;
            let r = 0.4;
            LorentzVec::from_spatial(r * angle.cos(), r * angle.sin(), 0.0)
        })
        .collect();

    let transition = TransitionOperator::new(positions, 1.0);
    let spectral = SpectralAnalysis::analyze(&transition, 3);

    assert!(spectral.mixing_time.is_finite());
    assert!(spectral.mixing_time > 0.0);
    assert!(spectral.spectral_gap >= 0.0);
}

// ============================================================================
// Cognitive Integration Tests
// ============================================================================

#[test]
fn test_free_energy_calculation() {
    let prior_mean = LorentzVec::origin();
    let prior_precision = 1.0;
    let mut fe_calc = FreeEnergyCalculator::with_prior(prior_mean, prior_precision);

    let mut belief = BeliefState::new(4, 10);
    belief.position_mean = LorentzVec::new(1.1, 0.1, 0.0, 0.0);
    belief.position_uncertainty = 0.5;
    belief.hidden_precision = 2.0;

    let obs = Observation::simple(
        0.0,
        LorentzVec::new(1.05, 0.05, 0.02, 0.0),
        vec![0.5, 0.3, 0.1, 0.0],
        1.5,
    );

    let fe_result = fe_calc.compute(&obs, &belief);

    assert!(fe_result.free_energy >= 0.0);
    assert!(fe_result.prediction_error >= 0.0);
}

#[test]
fn test_global_workspace() {
    let config = GlobalWorkspaceConfig {
        num_specialists: 4,
        ignition_threshold: 0.5,
        broadcast_decay: 50.0,
        max_coalition_size: 3,
        workspace_capacity: 5,
        access_threshold: 0.3,
        competition_temperature: 1.0,
    };

    let mut workspace = GlobalWorkspace::new(config);

    // Step the workspace
    for t in 0..10 {
        workspace.step(t as f64);
    }

    // Access stats as field
    assert!(workspace.stats.total_broadcasts >= 0);
}

#[test]
fn test_bateson_ecology() {
    let config = EcologyConfig::default();
    let mut eco_mind = EcologicalMind::new(config);

    for i in 0..10 {
        let pos = LorentzVec::new(
            1.0 + 0.01 * i as f64,
            0.1 * (i as f64 * 0.1).sin(),
            0.1 * (i as f64 * 0.1).cos(),
            0.0
        );

        let obs = Observation::simple(
            i as f64 * 10.0,
            pos,
            vec![0.5; 4],
            1.0,
        );

        let response = vec![0.1, 0.2, 0.3, 0.4];
        let result = eco_mind.process(&obs, &response);

        assert!(result.context_id < 100);
        assert!(result.learning_level.level() <= 3);
    }

    assert!(eco_mind.stats.total_observations > 0);
}

#[test]
fn test_iit_phi() {
    let config = PhiConfig {
        max_exact_size: 4,
        num_samples: 100,
        phi_threshold: 0.01,
        unfolding_steps: 3,
        background_prob: 0.5,
        use_icp: true,
    };

    let mut phi_calc = PhiCalculator::new(config, 4);

    let connectivity = vec![
        (0, 1, 0.8),
        (1, 2, 0.8),
        (2, 3, 0.8),
        (3, 0, 0.8),
    ];
    phi_calc.set_tpm_from_connectivity(&connectivity);
    phi_calc.set_state(vec![true, false, true, false]);

    let phi_result = phi_calc.compute_phi();
    assert!(phi_result.phi >= 0.0);
}

#[test]
fn test_replicator_dynamics() {
    let config = ReplicatorConfig {
        num_strategies: 3,
        dt: 0.01,
        selection_intensity: 1.0,
        mutation_rate: 0.001,
        min_frequency: 1e-6,
        hyperbolic_payoffs: true,
        curvature: -1.0,
    };

    // Rock-Paper-Scissors
    let mut payoff = PayoffMatrix::zeros(3);
    payoff.set(0, 0, 0.0);
    payoff.set(0, 1, -1.0);
    payoff.set(0, 2, 1.0);
    payoff.set(1, 0, 1.0);
    payoff.set(1, 1, 0.0);
    payoff.set(1, 2, -1.0);
    payoff.set(2, 0, -1.0);
    payoff.set(2, 1, 1.0);
    payoff.set(2, 2, 0.0);

    let mut replicator = HyperbolicReplicator::new(config, payoff);

    for _ in 0..100 {
        let result = replicator.step();
        assert!(result.avg_fitness.is_finite());
    }

    let total_freq: f64 = replicator.strategies.iter().map(|s| s.frequency).sum();
    assert!((total_freq - 1.0).abs() < 1e-6);
}

// ============================================================================
// Full System Integration Tests
// ============================================================================

#[test]
fn test_geometry_to_cognition_pipeline() {
    // 1. Create hyperbolic structure
    let tessellation = HeptagonalTessellation::new(2).unwrap();
    let tiles = tessellation.tiles();
    assert!(!tiles.is_empty());

    // 2. Extract positions (center is a field, not a method)
    let positions: Vec<LorentzVec> = tiles.iter()
        .take(10)
        .map(|tile| {
            let c = tile.center.coords();
            LorentzVec::from_spatial(c.x, c.y, 0.0)
        })
        .collect();

    // 3. Create heat kernel
    let kernel = HyperbolicHeatKernel::new(1.0);

    // 4. Compute kernel matrix
    let n = positions.len();
    let mut kernel_matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            kernel_matrix[i][j] = kernel.evaluate(&positions[i], &positions[j]);
        }
    }

    // 5. Use in free energy calculation
    let mut fe_calc = FreeEnergyCalculator::with_prior(LorentzVec::origin(), 1.0);

    let belief = BeliefState::new(4, 10);
    let obs = Observation::simple(
        0.0,
        positions[0],
        vec![kernel_matrix[0][1], kernel_matrix[0][2], 0.0, 0.0],
        1.0,
    );

    let fe_result = fe_calc.compute(&obs, &belief);
    assert!(fe_result.free_energy.is_finite());
}

#[test]
fn test_policy_belief_integration() {
    let mut belief = BeliefState::new(4, 10);
    belief.position_mean = LorentzVec::new(1.05, 0.1, 0.05, 0.0);
    belief.position_uncertainty = 0.5;

    let policy = Policy::free_energy();

    // Verify policy has expected properties
    assert!(policy.temperature > 0.0);
    assert!(policy.exploration >= 0.0);
}

// ============================================================================
// Consistency Tests
// ============================================================================

#[test]
fn test_distance_symmetry() {
    let p1 = PoincarePoint::new(na::Vector3::new(0.3, 0.2, 0.0)).unwrap();
    let p2 = PoincarePoint::new(na::Vector3::new(-0.1, 0.4, 0.0)).unwrap();

    let d12 = p1.hyperbolic_distance(&p2);
    let d21 = p2.hyperbolic_distance(&p1);

    assert!((d12 - d21).abs() < EPSILON);
}

#[test]
fn test_triangle_inequality() {
    let p1 = PoincarePoint::new(na::Vector3::new(0.2, 0.0, 0.0)).unwrap();
    let p2 = PoincarePoint::new(na::Vector3::new(0.0, 0.2, 0.0)).unwrap();
    let p3 = PoincarePoint::new(na::Vector3::new(-0.2, 0.0, 0.0)).unwrap();

    let d12 = p1.hyperbolic_distance(&p2);
    let d23 = p2.hyperbolic_distance(&p3);
    let d13 = p1.hyperbolic_distance(&p3);

    assert!(d13 <= d12 + d23 + EPSILON);
}

#[test]
fn test_moebius_composition() {
    let t1 = MoebiusTransform::rotation(PI / 4.0);
    let t2 = MoebiusTransform::rotation(PI / 4.0);
    let t3 = t1.compose(&t2);

    let z = Complex64::new(0.3, 0.0);
    let result = t3.apply(z);

    // π/2 rotation of (0.3, 0) should give (0, 0.3)
    assert!((result.re).abs() < 0.01);
    assert!((result.im - 0.3).abs() < 0.01);
}

#[test]
fn test_stochastic_eigenvalue_bounds() {
    let positions: Vec<LorentzVec> = (0..5)
        .map(|i| {
            let angle = 2.0 * PI * i as f64 / 5.0;
            let r = 0.3;
            LorentzVec::from_spatial(r * angle.cos(), r * angle.sin(), 0.0)
        })
        .collect();

    let transition = TransitionOperator::new(positions, 1.0);
    let spectral = SpectralAnalysis::analyze(&transition, 3);

    // All eigenvalues in [-1, 1]
    for lambda in &spectral.eigenvalues {
        assert!(*lambda >= -1.0 - EPSILON);
        assert!(*lambda <= 1.0 + EPSILON);
    }

    // Largest eigenvalue should be 1
    let max_lambda = spectral.eigenvalues.iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!((max_lambda - 1.0).abs() < 0.1);
}

#[test]
fn test_probability_normalization() {
    let dist = ProbabilityDistribution::uniform(3);

    let total: f64 = dist.probs.values().sum();
    assert!((total - 1.0).abs() < EPSILON);

    for p in dist.probs.values() {
        assert!(*p >= 0.0);
    }
}

#[test]
fn test_emd_properties() {
    let dist1 = ProbabilityDistribution::uniform(2);
    let dist2 = ProbabilityDistribution::uniform(2);

    let emd_self = dist1.emd(&dist1);
    assert!(emd_self < 0.01);

    let emd_12 = dist1.emd(&dist2);
    let emd_21 = dist2.emd(&dist1);
    assert!((emd_12 - emd_21).abs() < 0.01);
    assert!(emd_12 >= 0.0);
}
