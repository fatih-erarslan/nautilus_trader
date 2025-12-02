//! Hamiltonian Calculation Benchmarks
//!
//! Validates Ising model energy calculations.
//!
//! ## Mathematical Foundation
//! H = -Σ_i h_i s_i - Σ_{i<j} J_ij s_i s_j
//!
//! Reference: Mezard et al. (1987) "Spin Glass Theory and Beyond"

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use hyperphysics_thermo::HamiltonianCalculator;
use hyperphysics_pbit::{PBitLattice, CouplingNetwork};

fn create_coupled_lattice(p: usize, q: usize, depth: usize, coupling_strength: f64) -> PBitLattice {
    let mut lattice = PBitLattice::new(p, q, depth, 1.0).expect("Failed to create lattice");

    let network = CouplingNetwork::new(coupling_strength, 1.0, 1e-6);
    network.build_couplings(&mut lattice).expect("Failed to build couplings");

    lattice
}

fn bench_total_energy(c: &mut Criterion) {
    let mut group = c.benchmark_group("hamiltonian_total_energy");

    // Test with different lattice sizes via tessellation parameters
    let configs = [
        ("roi_48", 3, 7, 2),      // 48 pBits
        ("large_108", 3, 7, 3),   // ~108 pBits
        ("huge_300", 5, 5, 3),    // ~300 pBits
    ];

    for (name, p, q, depth) in configs {
        let lattice = create_coupled_lattice(p, q, depth, 1.0);
        let size = lattice.size();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("lattice", name),
            &name,
            |b, _| {
                b.iter(|| {
                    black_box(HamiltonianCalculator::energy(black_box(&lattice)))
                });
            },
        );
    }

    group.finish();
}

fn bench_energy_difference(c: &mut Criterion) {
    let mut group = c.benchmark_group("hamiltonian_energy_diff");

    let lattice = create_coupled_lattice(3, 7, 2, 1.0);
    let size = lattice.size();

    // Benchmark energy difference calculation for single spin flip
    group.bench_function("single_flip", |b| {
        b.iter(|| {
            // Test flip at middle index
            black_box(HamiltonianCalculator::energy_difference(
                black_box(&lattice),
                black_box(size / 2)
            ))
        });
    });

    // Benchmark for all spins (useful for Metropolis)
    group.bench_function("all_flips", |b| {
        b.iter(|| {
            for i in 0..size {
                black_box(HamiltonianCalculator::energy_difference(
                    black_box(&lattice),
                    black_box(i)
                ));
            }
        });
    });

    group.finish();
}

fn bench_local_energy(c: &mut Criterion) {
    let mut group = c.benchmark_group("hamiltonian_local_energy");

    let lattice = create_coupled_lattice(3, 7, 2, 1.0);
    let size = lattice.size();

    group.bench_function("single_site", |b| {
        b.iter(|| {
            black_box(HamiltonianCalculator::local_energy(
                black_box(&lattice),
                black_box(size / 2)
            ))
        });
    });

    group.throughput(Throughput::Elements(size as u64));
    group.bench_function("all_sites", |b| {
        b.iter(|| {
            for i in 0..size {
                black_box(HamiltonianCalculator::local_energy(
                    black_box(&lattice),
                    black_box(i)
                ));
            }
        });
    });

    group.finish();
}

fn bench_coupling_strength_calc(c: &mut Criterion) {
    let mut group = c.benchmark_group("hamiltonian_coupling");

    let lattice = create_coupled_lattice(3, 7, 2, 1.0);

    group.bench_function("avg_coupling", |b| {
        b.iter(|| {
            black_box(HamiltonianCalculator::average_coupling_strength(
                black_box(&lattice)
            ))
        });
    });

    group.finish();
}

fn bench_energy_per_pbit(c: &mut Criterion) {
    let mut group = c.benchmark_group("hamiltonian_energy_per_pbit");

    let configs = [
        ("small", 3, 5, 2),
        ("medium", 3, 7, 2),
        ("large", 3, 7, 3),
    ];

    for (name, p, q, depth) in configs {
        let lattice = create_coupled_lattice(p, q, depth, 1.0);

        group.bench_with_input(
            BenchmarkId::new("lattice", name),
            &name,
            |b, _| {
                b.iter(|| {
                    black_box(HamiltonianCalculator::energy_per_pbit(
                        black_box(&lattice)
                    ))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_total_energy,
    bench_energy_difference,
    bench_local_energy,
    bench_coupling_strength_calc,
    bench_energy_per_pbit,
);
criterion_main!(benches);
