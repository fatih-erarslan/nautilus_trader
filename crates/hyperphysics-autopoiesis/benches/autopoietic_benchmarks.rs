//! Benchmarks for hyperphysics-autopoiesis bridge components
//!
//! These benchmarks measure the performance of the autopoietic system
//! components to ensure they meet HFT latency requirements.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use nalgebra::{DMatrix, DVector};

use hyperphysics_autopoiesis::adapters::{
    ConsciousnessAdapter, NetworkAdapter, SyncAdapter, ThermoAdapter,
};
use hyperphysics_autopoiesis::bridges::{AutopoieticBridge, DissipativeBridge, SyntergicBridge};
use hyperphysics_autopoiesis::dynamics::{AutopoieticDynamics, BifurcationDetector, DynamicsConfig, EmergenceMonitor};
use hyperphysics_autopoiesis::emergence::EmergenceAnalyzer;

/// Benchmark thermodynamic adapter entropy production
fn bench_thermo_adapter(c: &mut Criterion) {
    let mut group = c.benchmark_group("ThermoAdapter");

    for size in [10, 100, 1000].iter() {
        let fluxes: Vec<f64> = (0..*size).map(|i| (i as f64) * 0.01).collect();
        let forces: Vec<f64> = vec![1.0; *size];

        group.bench_with_input(BenchmarkId::new("entropy_production", size), size, |b, _| {
            let mut adapter = ThermoAdapter::new(300.0);
            b.iter(|| {
                adapter.compute_entropy_production(black_box(&fluxes), black_box(&forces))
            });
        });
    }

    group.finish();
}

/// Benchmark synchronization adapter order parameter
fn bench_sync_adapter(c: &mut Criterion) {
    let mut group = c.benchmark_group("SyncAdapter");

    for size in [10, 100, 1000, 10000].iter() {
        let phases: Vec<f64> = (0..*size)
            .map(|i| (i as f64) * 0.01 * std::f64::consts::PI)
            .collect();

        group.bench_with_input(BenchmarkId::new("order_parameter", size), size, |b, _| {
            let mut adapter = SyncAdapter::new(1.0);
            b.iter(|| adapter.compute_order_parameter(black_box(&phases)));
        });
    }

    group.finish();
}

/// Benchmark network adapter metrics computation
fn bench_network_adapter(c: &mut Criterion) {
    let mut group = c.benchmark_group("NetworkAdapter");

    for size in [10, 50, 100].iter() {
        let mut adj = DMatrix::zeros(*size, *size);
        // Create random-ish adjacency
        for i in 0..*size {
            for j in 0..*size {
                if i != j && (i + j) % 3 == 0 {
                    adj[(i, j)] = 1.0;
                }
            }
        }

        group.bench_with_input(BenchmarkId::new("set_adjacency", size), &adj, |b, adj| {
            let mut adapter = NetworkAdapter::new(*size);
            b.iter(|| adapter.set_adjacency(black_box(adj.clone())));
        });
    }

    group.finish();
}

/// Benchmark autopoietic bridge cycle
fn bench_autopoietic_bridge(c: &mut Criterion) {
    let mut group = c.benchmark_group("AutopoieticBridge");

    group.bench_function("execute_cycle", |b| {
        let mut bridge = AutopoieticBridge::new(0.8);
        bridge.register_component("A", 1.0, 0.5);
        bridge.register_component("B", 0.8, 0.3);
        bridge.register_component("C", 0.6, 0.4);

        b.iter(|| bridge.execute_cycle());
    });

    group.finish();
}

/// Benchmark dissipative bridge regime detection
fn bench_dissipative_bridge(c: &mut Criterion) {
    let mut group = c.benchmark_group("DissipativeBridge");

    group.bench_function("update_control_parameter", |b| {
        let mut bridge = DissipativeBridge::new(1.0);
        let mut param = 0.0;

        b.iter(|| {
            param = (param + 0.01) % 1.0;
            bridge.update_control_parameter(black_box(param))
        });
    });

    group.finish();
}

/// Benchmark syntergic bridge coherence update
fn bench_syntergic_bridge(c: &mut Criterion) {
    let mut group = c.benchmark_group("SyntergicBridge");

    for size in [10, 100, 1000].iter() {
        let phases: Vec<f64> = (0..*size)
            .map(|i| (i as f64) * 0.01)
            .collect();

        group.bench_with_input(BenchmarkId::new("update_from_phases", size), size, |b, _| {
            let mut bridge = SyntergicBridge::new(0.9);
            b.iter(|| bridge.update_from_phases(black_box(&phases)));
        });
    }

    group.finish();
}

/// Benchmark dynamics state recording
fn bench_dynamics(c: &mut Criterion) {
    let mut group = c.benchmark_group("AutopoieticDynamics");

    let config = DynamicsConfig::default();

    group.bench_function("record_state", |b| {
        let mut dynamics = AutopoieticDynamics::new(config.clone(), 5);
        let state = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        b.iter(|| dynamics.record_state(black_box(state.clone()), black_box(0.5)));
    });

    group.finish();
}

/// Benchmark bifurcation detection
fn bench_bifurcation_detector(c: &mut Criterion) {
    let mut group = c.benchmark_group("BifurcationDetector");

    for size in [3, 5, 10].iter() {
        let jacobian = DMatrix::from_diagonal(&DVector::from_vec(
            (0..*size).map(|i| -(i as f64 + 1.0)).collect(),
        ));

        group.bench_with_input(BenchmarkId::new("record_jacobian", size), &jacobian, |b, j| {
            let mut detector = BifurcationDetector::default();
            b.iter(|| detector.record_jacobian(black_box(j.clone())));
        });
    }

    group.finish();
}

/// Benchmark emergence detection
fn bench_emergence_analyzer(c: &mut Criterion) {
    let mut group = c.benchmark_group("EmergenceAnalyzer");

    for size in [4, 8, 16].iter() {
        let mut cov = DMatrix::zeros(*size, *size);
        cov[(0, 0)] = 10.0;
        for i in 1..*size {
            cov[(i, i)] = 1.0;
        }

        group.bench_with_input(BenchmarkId::new("detect_from_covariance", size), &cov, |b, c| {
            let mut analyzer = EmergenceAnalyzer::new(0.1, 10000);
            b.iter(|| analyzer.detect_from_covariance(black_box(c)));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_thermo_adapter,
    bench_sync_adapter,
    bench_network_adapter,
    bench_autopoietic_bridge,
    bench_dissipative_bridge,
    bench_syntergic_bridge,
    bench_dynamics,
    bench_bifurcation_detector,
    bench_emergence_analyzer,
);

criterion_main!(benches);
