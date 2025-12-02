//! Entropy Calculation Benchmarks
//!
//! Validates Gibbs entropy and Boltzmann entropy computations.
//!
//! ## Scientific Foundation
//! - Boltzmann: S = k_B ln(Ω)
//! - Gibbs: S = -k_B Σ P(s) ln P(s)
//! - NIST-JANAF: Tabulated reference data with <0.1% accuracy
//!
//! References:
//! - Gibbs (1902) "Elementary Principles in Statistical Mechanics"
//! - Chase (1998) "NIST-JANAF Thermochemical Tables" 4th Ed.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use hyperphysics_thermo::{EntropyCalculator, Temperature, entropy::constants};
use hyperphysics_pbit::PBitLattice;

fn bench_gibbs_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_gibbs");

    // Test with different distribution sizes
    for &size in &[10, 100, 1000, 10_000] {
        // Create uniform distribution (deterministic, NOT random)
        let probs: Vec<f64> = vec![1.0 / size as f64; size];

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("states", size),
            &size,
            |b, _| {
                let calc = EntropyCalculator::new();
                b.iter(|| {
                    black_box(calc.gibbs_entropy(black_box(&probs)))
                });
            },
        );
    }

    group.finish();
}

fn bench_boltzmann_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_boltzmann");

    // Create energy levels (deterministic spectrum)
    let calc = EntropyCalculator::new();

    for &num_levels in &[10, 50, 100, 500] {
        // Harmonic oscillator spectrum: E_n = (n + 0.5) * ℏω
        let energy_levels: Vec<(f64, usize)> = (0..num_levels)
            .map(|n| ((n as f64 + 0.5) * 0.01, 1))  // Energy in arbitrary units, degeneracy 1
            .collect();

        let temp = Temperature::from_kelvin(300.0).unwrap();

        group.throughput(Throughput::Elements(num_levels as u64));
        group.bench_with_input(
            BenchmarkId::new("levels", num_levels),
            &num_levels,
            |b, _| {
                b.iter(|| {
                    black_box(calc.boltzmann_entropy(
                        black_box(&temp),
                        black_box(&energy_levels)
                    ))
                });
            },
        );
    }

    group.finish();
}

fn bench_entropy_from_pbits(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_pbit");

    let calc = EntropyCalculator::new();

    let configs = [
        ("roi_48", 3, 7, 2),
        ("medium_108", 3, 7, 3),
    ];

    for (name, p, q, depth) in configs {
        let lattice = PBitLattice::new(p, q, depth, 1.0).expect("Failed to create lattice");
        let size = lattice.size();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("lattice", name),
            &name,
            |b, _| {
                b.iter(|| {
                    black_box(calc.entropy_from_pbits(black_box(&lattice)))
                });
            },
        );
    }

    group.finish();
}

fn bench_entropy_temperature_dependence(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_temperature");

    let calc = EntropyCalculator::new();
    let lattice = PBitLattice::roi_48(1.0).expect("Failed to create lattice");

    // Test at different temperatures
    for &temp_k in &[10.0, 100.0, 300.0, 1000.0] {
        let temp = Temperature::from_kelvin(temp_k).unwrap();

        group.bench_with_input(
            BenchmarkId::new("temp_K", temp_k as u64),
            &temp_k,
            |b, _| {
                b.iter(|| {
                    black_box(calc.entropy_from_pbits_with_temperature(
                        black_box(&lattice),
                        black_box(&temp)
                    ))
                });
            },
        );
    }

    group.finish();
}

fn bench_sackur_tetrode(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_sackur_tetrode");

    let calc = EntropyCalculator::new();

    // Test with different gas types (identified by mass)
    let gases = [
        ("Argon", 39.948 * constants::AMU),
        ("Helium", 4.003 * constants::AMU),
        ("Nitrogen", 28.014 * constants::AMU),
    ];

    for (name, mass) in gases {
        let temp = Temperature::from_kelvin(298.15).unwrap();
        let volume = constants::GAS_CONSTANT * 298.15 / 100000.0;  // 1 bar pressure
        let num_particles = constants::AVOGADRO;

        group.bench_with_input(
            BenchmarkId::new("gas", name),
            &name,
            |b, _| {
                b.iter(|| {
                    black_box(calc.sackur_tetrode_entropy(
                        black_box(&temp),
                        black_box(volume),
                        black_box(num_particles),
                        black_box(mass)
                    ))
                });
            },
        );
    }

    group.finish();
}

fn bench_shannon_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_shannon");

    let calc = EntropyCalculator::new();

    for &size in &[10, 100, 1000] {
        // Deterministic probability distribution (NOT random)
        let probs: Vec<f64> = (0..size)
            .map(|i| {
                // Zipf-like distribution
                let rank = i + 1;
                1.0 / (rank as f64)
            })
            .collect();

        // Normalize
        let sum: f64 = probs.iter().sum();
        let probs: Vec<f64> = probs.iter().map(|p| p / sum).collect();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("states", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(calc.shannon_entropy(black_box(&probs)))
                });
            },
        );
    }

    group.finish();
}

fn bench_max_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_max");

    let calc = EntropyCalculator::new();

    for &num_pbits in &[48, 100, 500, 1000] {
        group.bench_with_input(
            BenchmarkId::new("pbits", num_pbits),
            &num_pbits,
            |b, _| {
                b.iter(|| {
                    black_box(calc.max_entropy(black_box(num_pbits)))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_gibbs_entropy,
    bench_boltzmann_entropy,
    bench_entropy_from_pbits,
    bench_entropy_temperature_dependence,
    bench_sackur_tetrode,
    bench_shannon_entropy,
    bench_max_entropy,
);
criterion_main!(benches);
