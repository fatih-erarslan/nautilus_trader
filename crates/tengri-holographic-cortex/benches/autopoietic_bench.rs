//! Benchmarks for Phase 6: Autopoietic pBit Networks with SOC
//!
//! Wolfram-verified performance targets:
//! - Network step: <100μs for 1000 nodes
//! - Avalanche tracking: <1μs overhead
//! - Homeostatic update: <10ns
//! - Phi computation: <10ms for 12 nodes (NP-hard)

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use tengri_holographic_cortex::{
    AvalancheTracker, HomeostaticRegulator, PhiComputer, AutopoieticTopology,
    AutopoieticNetwork, AutopoieticConfig, SOC_CRITICAL_TEMP,
};

fn bench_avalanche_tracker(c: &mut Criterion) {
    let mut group = c.benchmark_group("avalanche_tracker");

    group.bench_function("record_activations", |b| {
        let mut tracker = AvalancheTracker::new();
        b.iter(|| {
            tracker.record_activations(black_box(10));
        })
    });

    group.bench_function("branching_ratio", |b| {
        let mut tracker = AvalancheTracker::new();
        for _ in 0..100 {
            tracker.record_activations(10);
        }
        b.iter(|| {
            black_box(tracker.branching_ratio())
        })
    });

    group.bench_function("power_law_estimate", |b| {
        let mut tracker = AvalancheTracker::new();
        for size in [1, 2, 1, 3, 1, 1, 2, 4, 1, 2, 1, 1, 5, 2, 1, 3, 1, 1, 2, 1] {
            tracker.record_activations(size);
            tracker.record_activations(0);
        }
        b.iter(|| {
            black_box(tracker.estimate_power_law_exponent())
        })
    });

    group.finish();
}

fn bench_homeostatic_regulator(c: &mut Criterion) {
    let mut group = c.benchmark_group("homeostatic_regulator");

    group.bench_function("update", |b| {
        let mut regulator = HomeostaticRegulator::default();
        b.iter(|| {
            regulator.update(black_box(1.1), black_box(0.1));
        })
    });

    group.bench_function("temperature_variance", |b| {
        let mut regulator = HomeostaticRegulator::default();
        for _ in 0..50 {
            regulator.update(1.0, 0.1);
        }
        b.iter(|| {
            black_box(regulator.temperature_variance())
        })
    });

    group.finish();
}

fn bench_phi_computer(c: &mut Criterion) {
    let mut group = c.benchmark_group("phi_computer");

    for num_nodes in [4, 8, 12].iter() {
        group.bench_with_input(BenchmarkId::new("compute_phi", num_nodes), num_nodes, |b, &n| {
            let phi = PhiComputer::new(n);
            b.iter(|| {
                black_box(phi.compute_phi())
            })
        });
    }

    group.finish();
}

fn bench_autopoietic_topology(c: &mut Criterion) {
    let mut group = c.benchmark_group("autopoietic_topology");

    for num_nodes in [50, 100, 200].iter() {
        group.bench_with_input(BenchmarkId::new("record_activations", num_nodes), num_nodes, |b, &n| {
            let mut topology = AutopoieticTopology::new(n);
            let activations: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
            b.iter(|| {
                topology.record_activations(black_box(&activations))
            })
        });
    }

    group.bench_function("evolve_100_nodes", |b| {
        let mut topology = AutopoieticTopology::new(100);
        let mut rng = SmallRng::seed_from_u64(42);

        // Warm up with some activations
        for i in 0..20 {
            let activations: Vec<bool> = (0..100).map(|j| (i + j) % 3 == 0).collect();
            topology.record_activations(&activations);
        }

        b.iter(|| {
            topology.evolve(black_box(&mut rng))
        })
    });

    group.finish();
}

fn bench_autopoietic_network(c: &mut Criterion) {
    let mut group = c.benchmark_group("autopoietic_network");

    for num_nodes in [100, 500, 1000].iter() {
        let config = AutopoieticConfig {
            num_nodes: *num_nodes,
            initial_temp: SOC_CRITICAL_TEMP,
            homeostatic: true,
            autopoietic_topology: false, // Disable for speed
            compute_phi: false,
            activity_setpoint: 0.1,
        };

        group.bench_with_input(BenchmarkId::new("step", num_nodes), num_nodes, |b, _| {
            let mut network = AutopoieticNetwork::new(config.clone());
            let mut rng = SmallRng::seed_from_u64(42);

            b.iter(|| {
                network.step(black_box(&mut rng))
            })
        });
    }

    // Benchmark with small-world topology
    group.bench_function("step_small_world_100", |b| {
        let config = AutopoieticConfig {
            num_nodes: 100,
            initial_temp: SOC_CRITICAL_TEMP,
            homeostatic: true,
            autopoietic_topology: false,
            compute_phi: false,
            activity_setpoint: 0.1,
        };

        let mut network = AutopoieticNetwork::new(config);
        let mut rng = SmallRng::seed_from_u64(42);
        network.init_small_world(6, 0.1, &mut rng);

        b.iter(|| {
            network.step(black_box(&mut rng))
        })
    });

    group.finish();
}

fn bench_full_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_simulation");

    group.bench_function("run_100_steps", |b| {
        let config = AutopoieticConfig {
            num_nodes: 100,
            initial_temp: SOC_CRITICAL_TEMP,
            homeostatic: true,
            autopoietic_topology: false,
            compute_phi: false,
            activity_setpoint: 0.1,
        };

        b.iter(|| {
            let mut network = AutopoieticNetwork::new(config.clone());
            let mut rng = SmallRng::seed_from_u64(42);
            network.run(black_box(100), &mut rng);
            black_box(network.stats())
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_avalanche_tracker,
    bench_homeostatic_regulator,
    bench_phi_computer,
    bench_autopoietic_topology,
    bench_autopoietic_network,
    bench_full_simulation,
);

criterion_main!(benches);
