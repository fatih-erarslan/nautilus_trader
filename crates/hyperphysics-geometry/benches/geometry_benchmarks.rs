//! Criterion.rs benchmarks for hyperphysics-geometry
//!
//! Benchmarks core operations across geometry, neural, and cognitive modules.
//! Run with: cargo bench -p hyperphysics-geometry

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use hyperphysics_geometry::{
    // Core geometry
    PoincarePoint, MoebiusTransform,
    FuchsianGroupAlgebraic, OrbitConfig, ExactOrbitEnumerator,
    HeptagonalTessellation,
    HyperbolicDelaunay, HyperbolicVoronoi,

    // SNN
    LorentzVec4D as LorentzVec,

    // Markov kernels
    HyperbolicHeatKernel, TransitionOperator,
    SpectralAnalysis,

    // STDP
    STDPConfig, STDPLearner,

    // Cognitive
    FreeEnergyCalculator,
    GlobalWorkspaceConfig, GlobalWorkspace,
    BeliefState, Observation,
    PhiConfig, PhiCalculator,
    EcologyConfig, EcologicalMind,
    ReplicatorConfig, HyperbolicReplicator, PayoffMatrix,
};

use nalgebra as na;
use num_complex::Complex64;
use std::f64::consts::PI;

// ============================================================================
// Geometry Benchmarks
// ============================================================================

fn bench_poincare_distance(c: &mut Criterion) {
    let p1 = PoincarePoint::new(na::Vector3::new(0.3, 0.2, 0.0)).unwrap();
    let p2 = PoincarePoint::new(na::Vector3::new(-0.2, 0.4, 0.0)).unwrap();

    c.bench_function("poincare_distance", |b| {
        b.iter(|| black_box(p1.hyperbolic_distance(&p2)))
    });
}

fn bench_moebius_apply(c: &mut Criterion) {
    let transform = MoebiusTransform::rotation(PI / 7.0);
    let z = Complex64::new(0.3, 0.2);

    c.bench_function("moebius_apply", |b| {
        b.iter(|| black_box(transform.apply(z)))
    });
}

fn bench_moebius_compose(c: &mut Criterion) {
    let t1 = MoebiusTransform::rotation(PI / 4.0);
    let t2 = MoebiusTransform::translation(0.5);

    c.bench_function("moebius_compose", |b| {
        b.iter(|| black_box(t1.compose(&t2)))
    });
}

fn bench_fuchsian_orbit(c: &mut Criterion) {
    let rotation = MoebiusTransform::rotation(2.0 * PI / 7.0);
    let group = FuchsianGroupAlgebraic::new(vec![rotation]);
    let base_point = Complex64::new(0.2, 0.0);

    let mut group_c = c.benchmark_group("fuchsian_orbit");
    for max_length in [3, 5, 7] {
        let config = OrbitConfig {
            max_word_length: max_length,
            ..OrbitConfig::default()
        };
        group_c.bench_with_input(
            BenchmarkId::from_parameter(max_length),
            &max_length,
            |b, _| {
                b.iter(|| {
                    let mut enumerator = ExactOrbitEnumerator::new(
                        group.clone(),
                        base_point,
                        config.clone(),
                    );
                    let orbit = enumerator.enumerate();
                    black_box(orbit.len())
                })
            },
        );
    }
    group_c.finish();
}

fn bench_tessellation_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tessellation_generation");
    for depth in [1, 2, 3] {
        group.bench_with_input(
            BenchmarkId::from_parameter(depth),
            &depth,
            |b, &d| {
                b.iter(|| black_box(HeptagonalTessellation::new(d).unwrap()))
            },
        );
    }
    group.finish();
}

fn bench_delaunay_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("delaunay_construction");
    for n_points in [5, 10, 20] {
        let points: Vec<PoincarePoint> = (0..n_points)
            .filter_map(|i| {
                let angle = 2.0 * PI * i as f64 / n_points as f64;
                let r = 0.3 + 0.1 * (i as f64 / n_points as f64);
                PoincarePoint::new(na::Vector3::new(r * angle.cos(), r * angle.sin(), 0.0)).ok()
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(n_points),
            &points,
            |b, pts| {
                b.iter(|| black_box(HyperbolicDelaunay::from_sites(pts).unwrap()))
            },
        );
    }
    group.finish();
}

fn bench_voronoi_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("voronoi_construction");
    for n_points in [5, 10, 20] {
        let points: Vec<PoincarePoint> = (0..n_points)
            .filter_map(|i| {
                let angle = 2.0 * PI * i as f64 / n_points as f64;
                let r = 0.3 + 0.1 * (i as f64 / n_points as f64);
                PoincarePoint::new(na::Vector3::new(r * angle.cos(), r * angle.sin(), 0.0)).ok()
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(n_points),
            &points,
            |b, pts| {
                b.iter(|| black_box(HyperbolicVoronoi::from_sites(pts).unwrap()))
            },
        );
    }
    group.finish();
}

// ============================================================================
// Neural/Markov Benchmarks
// ============================================================================

fn bench_heat_kernel_evaluate(c: &mut Criterion) {
    let kernel = HyperbolicHeatKernel::new(1.0);
    let p1 = LorentzVec::from_spatial(0.3, 0.2, 0.0);
    let p2 = LorentzVec::from_spatial(-0.1, 0.4, 0.0);

    c.bench_function("heat_kernel_evaluate", |b| {
        b.iter(|| black_box(kernel.evaluate(&p1, &p2)))
    });
}

fn bench_transition_operator_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("transition_operator");
    for n in [5, 10, 20] {
        let positions: Vec<LorentzVec> = (0..n)
            .map(|i| {
                let angle = 2.0 * PI * i as f64 / n as f64;
                let r = 0.3;
                LorentzVec::from_spatial(r * angle.cos(), r * angle.sin(), 0.0)
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &positions,
            |b, pos| {
                b.iter(|| black_box(TransitionOperator::new(pos.clone(), 1.0)))
            },
        );
    }
    group.finish();
}

fn bench_spectral_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_analysis");
    for n in [5, 10, 15] {
        let positions: Vec<LorentzVec> = (0..n)
            .map(|i| {
                let angle = 2.0 * PI * i as f64 / n as f64;
                let r = 0.4;
                LorentzVec::from_spatial(r * angle.cos(), r * angle.sin(), 0.0)
            })
            .collect();
        let transition = TransitionOperator::new(positions, 1.0);

        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &transition,
            |b, t| {
                b.iter(|| black_box(SpectralAnalysis::analyze(t, 3)))
            },
        );
    }
    group.finish();
}

fn bench_stdp_update(c: &mut Criterion) {
    let config = STDPConfig::default();

    c.bench_function("stdp_weight_update", |b| {
        let learner = STDPLearner::new(config.clone());
        let cfg = learner.config();

        b.iter(|| {
            let delta_t = 5.0;
            let stdp = cfg.a_plus * (-delta_t / cfg.tau_plus).exp();
            black_box(stdp)
        })
    });
}

// ============================================================================
// Cognitive Benchmarks
// ============================================================================

fn bench_free_energy_computation(c: &mut Criterion) {
    let mut fe_calc = FreeEnergyCalculator::with_prior(LorentzVec::origin(), 1.0);
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

    c.bench_function("free_energy_compute", |b| {
        b.iter(|| black_box(fe_calc.compute(&obs, &belief)))
    });
}

fn bench_global_workspace_step(c: &mut Criterion) {
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

    c.bench_function("global_workspace_step", |b| {
        let mut t = 0.0;
        b.iter(|| {
            t += 1.0;
            black_box(workspace.step(t))
        })
    });
}

fn bench_phi_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("phi_computation");

    for n in [3, 4, 5] {
        let config = PhiConfig {
            max_exact_size: n,
            num_samples: 50,
            phi_threshold: 0.01,
            unfolding_steps: 2,
            background_prob: 0.5,
            use_icp: true,
        };

        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &n,
            |b, &size| {
                b.iter(|| {
                    let mut phi_calc = PhiCalculator::new(config.clone(), size);

                    // Create ring connectivity
                    let connectivity: Vec<(usize, usize, f64)> = (0..size)
                        .map(|i| (i, (i + 1) % size, 0.8))
                        .collect();
                    phi_calc.set_tpm_from_connectivity(&connectivity);

                    let state: Vec<bool> = (0..size).map(|i| i % 2 == 0).collect();
                    phi_calc.set_state(state);

                    black_box(phi_calc.compute_phi())
                })
            },
        );
    }
    group.finish();
}

fn bench_ecology_process(c: &mut Criterion) {
    let config = EcologyConfig::default();
    let mut eco_mind = EcologicalMind::new(config);

    c.bench_function("ecology_process", |b| {
        let mut i = 0;
        b.iter(|| {
            i += 1;
            let pos = LorentzVec::new(
                1.0 + 0.01 * (i as f64 % 100.0),
                0.1 * ((i as f64 * 0.1) % (2.0 * PI)).sin(),
                0.1 * ((i as f64 * 0.1) % (2.0 * PI)).cos(),
                0.0
            );
            let obs = Observation::simple(i as f64 * 10.0, pos, vec![0.5; 4], 1.0);
            let response = vec![0.1, 0.2, 0.3, 0.4];
            black_box(eco_mind.process(&obs, &response))
        })
    });
}

fn bench_replicator_step(c: &mut Criterion) {
    let config = ReplicatorConfig {
        num_strategies: 3,
        dt: 0.01,
        selection_intensity: 1.0,
        mutation_rate: 0.001,
        min_frequency: 1e-6,
        hyperbolic_payoffs: true,
        curvature: -1.0,
    };

    let mut payoff = PayoffMatrix::zeros(3);
    payoff.set(0, 0, 0.0); payoff.set(0, 1, -1.0); payoff.set(0, 2, 1.0);
    payoff.set(1, 0, 1.0); payoff.set(1, 1, 0.0); payoff.set(1, 2, -1.0);
    payoff.set(2, 0, -1.0); payoff.set(2, 1, 1.0); payoff.set(2, 2, 0.0);

    let mut replicator = HyperbolicReplicator::new(config, payoff);

    c.bench_function("replicator_step", |b| {
        b.iter(|| black_box(replicator.step()))
    });
}

// ============================================================================
// Benchmark Groups
// ============================================================================

criterion_group!(
    geometry_benches,
    bench_poincare_distance,
    bench_moebius_apply,
    bench_moebius_compose,
    bench_fuchsian_orbit,
    bench_tessellation_generation,
    bench_delaunay_construction,
    bench_voronoi_construction,
);

criterion_group!(
    neural_benches,
    bench_heat_kernel_evaluate,
    bench_transition_operator_construction,
    bench_spectral_analysis,
    bench_stdp_update,
);

criterion_group!(
    cognitive_benches,
    bench_free_energy_computation,
    bench_global_workspace_step,
    bench_phi_computation,
    bench_ecology_process,
    bench_replicator_step,
);

criterion_main!(geometry_benches, neural_benches, cognitive_benches);
