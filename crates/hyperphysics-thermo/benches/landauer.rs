//! Landauer Principle Benchmarks
//!
//! Validates minimum erasure energy calculations and bound verification.
//!
//! ## Scientific Foundation
//! E_min = k_B T ln(2) per bit erased
//!
//! References:
//! - Landauer (1961) "Irreversibility and heat generation" IBM J. Res. Dev.
//! - Berut et al. (2012) "Experimental verification" Nature 483:187

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use hyperphysics_thermo::LandauerEnforcer;

fn bench_minimum_erasure_energy(c: &mut Criterion) {
    let mut group = c.benchmark_group("landauer_min_energy");

    // Test at different temperatures
    for &temp_k in &[77.0, 300.0, 1000.0, 4000.0] {
        let enforcer = LandauerEnforcer::new(temp_k).unwrap();

        group.bench_with_input(
            BenchmarkId::new("temp_K", temp_k as u64),
            &temp_k,
            |b, _| {
                b.iter(|| {
                    black_box(enforcer.minimum_erasure_energy())
                });
            },
        );
    }

    group.finish();
}

fn bench_n_bit_erasure(c: &mut Criterion) {
    let mut group = c.benchmark_group("landauer_n_bits");

    let enforcer = LandauerEnforcer::new(300.0).unwrap();

    for &bits in &[1, 10, 100, 1000, 10_000] {
        group.throughput(Throughput::Elements(bits as u64));
        group.bench_with_input(
            BenchmarkId::new("bits", bits),
            &bits,
            |b, _| {
                b.iter(|| {
                    black_box(enforcer.minimum_erasure_energy_n(black_box(bits)))
                });
            },
        );
    }

    group.finish();
}

fn bench_verify_bound(c: &mut Criterion) {
    let mut group = c.benchmark_group("landauer_verify");

    let enforcer = LandauerEnforcer::new(300.0).unwrap();
    let e_min = enforcer.minimum_erasure_energy();

    // Test bound verification with varying energy margins
    for &margin_factor in &[0.5, 1.0, 1.5, 2.0, 10.0] {
        let energy = e_min * margin_factor;

        group.bench_with_input(
            BenchmarkId::new("margin", format!("{:.1}x", margin_factor)),
            &margin_factor,
            |b, _| {
                b.iter(|| {
                    black_box(enforcer.verify_bound(
                        black_box(energy),
                        black_box(1)
                    ))
                });
            },
        );
    }

    group.finish();
}

fn bench_energy_entropy_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("landauer_conversion");

    let enforcer = LandauerEnforcer::new(300.0).unwrap();

    // Deterministic entropy values (NOT random)
    let entropy_values: Vec<f64> = (1..=10)
        .map(|i| i as f64 * 1e-23)
        .collect();

    group.bench_function("energy_from_entropy", |b| {
        b.iter(|| {
            for &ds in &entropy_values {
                black_box(enforcer.energy_from_entropy(black_box(ds)));
            }
        });
    });

    group.bench_function("entropy_from_energy", |b| {
        let energies: Vec<f64> = entropy_values.iter()
            .map(|&ds| enforcer.energy_from_entropy(ds))
            .collect();

        b.iter(|| {
            for &e in &energies {
                black_box(enforcer.entropy_from_energy(black_box(e)));
            }
        });
    });

    group.finish();
}

fn bench_track_dissipation(c: &mut Criterion) {
    let mut group = c.benchmark_group("landauer_dissipation");

    let enforcer = LandauerEnforcer::new(300.0).unwrap();

    // Deterministic entropy sequences (NOT random)
    let entropy_sequence: Vec<(f64, f64)> = (0..100)
        .map(|i| {
            let s0 = i as f64 * 1e-23;
            let s1 = (i as f64 + 0.1) * 1e-23;  // Small increase (second law compliant)
            (s0, s1)
        })
        .collect();

    group.throughput(Throughput::Elements(entropy_sequence.len() as u64));
    group.bench_function("track_100_steps", |b| {
        b.iter(|| {
            for &(s0, s1) in &entropy_sequence {
                black_box(enforcer.track_dissipation(
                    black_box(s0),
                    black_box(s1)
                ));
            }
        });
    });

    group.finish();
}

fn bench_bits_erasable(c: &mut Criterion) {
    let mut group = c.benchmark_group("landauer_bits_erasable");

    // Test at different temperatures
    for &temp_k in &[77.0, 300.0, 1000.0] {
        let enforcer = LandauerEnforcer::new(temp_k).unwrap();
        let energy = 1e-18;  // 1 attojoule budget

        group.bench_with_input(
            BenchmarkId::new("temp_K", temp_k as u64),
            &temp_k,
            |b, _| {
                b.iter(|| {
                    black_box(enforcer.bits_erasable(black_box(energy)))
                });
            },
        );
    }

    group.finish();
}

fn bench_room_temperature_bound(c: &mut Criterion) {
    let mut group = c.benchmark_group("landauer_room_temp");

    group.bench_function("bound_at_300K", |b| {
        b.iter(|| {
            black_box(LandauerEnforcer::bound_at_room_temperature())
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_minimum_erasure_energy,
    bench_n_bit_erasure,
    bench_verify_bound,
    bench_energy_entropy_conversion,
    bench_track_dissipation,
    bench_bits_erasable,
    bench_room_temperature_bound,
);
criterion_main!(benches);
