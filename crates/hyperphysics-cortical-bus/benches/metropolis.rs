//! Metropolis sweep benchmark.
//!
//! Benchmarks pBit dynamics including:
//! - Full Metropolis sweeps at various scales
//! - Single pBit updates
//! - Ising energy computation

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use hyperphysics_cortical_bus::pbit::{PBitArrayImpl, CouplingMatrix, MetropolisSweep};

fn bench_metropolis_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("metropolis_sweep");

    for num_pbits in [100, 1000, 10000, 65536].iter() {
        let pbits = PBitArrayImpl::new(*num_pbits);
        let couplings = CouplingMatrix::random_sparse(*num_pbits, 10, 1.0);
        let mut rng = fastrand::Rng::new();
        pbits.randomize(&mut rng);

        let sweep = MetropolisSweep::new(1.0);
        
        group.throughput(Throughput::Elements(*num_pbits as u64));
        group.bench_with_input(BenchmarkId::from_parameter(num_pbits), num_pbits, |b, _| {
            b.iter(|| {
                let flips = sweep.sweep(black_box(&pbits), black_box(&couplings));
                black_box(flips);
            })
        });
    }

    group.finish();
}

fn bench_single_update(c: &mut Criterion) {
    let pbits = PBitArrayImpl::new(1000);
    let couplings = CouplingMatrix::random_sparse(1000, 10, 1.0);
    let mut rng = fastrand::Rng::new();
    pbits.randomize(&mut rng);

    c.bench_function("pbit_single_update", |b| {
        let mut rng = fastrand::Rng::new();
        b.iter(|| {
            let flipped = pbits.metropolis_update(
                black_box(500), 
                black_box(&couplings), 
                1.0, 
                &mut rng
            );
            black_box(flipped);
        })
    });
}

fn bench_energy_compute(c: &mut Criterion) {
    let pbits = PBitArrayImpl::new(1000);
    let couplings = CouplingMatrix::random_sparse(1000, 10, 1.0);
    let mut rng = fastrand::Rng::new();
    pbits.randomize(&mut rng);

    c.bench_function("pbit_compute_energy", |b| {
        b.iter(|| {
            let energy = pbits.compute_energy(black_box(&couplings));
            black_box(energy);
        })
    });
}

fn bench_temperature_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("temperature_scaling");
    
    let pbits = PBitArrayImpl::new(1000);
    let couplings = CouplingMatrix::lattice_2d(32, 32, 1.0); // 2D Ising model
    let mut rng = fastrand::Rng::new();
    pbits.randomize(&mut rng);

    for temp in [0.1, 1.0, 2.27, 10.0].iter() { // 2.27 ≈ critical temperature for 2D Ising
        let sweep = MetropolisSweep::new(*temp);
        
        group.bench_with_input(BenchmarkId::from_parameter(temp), temp, |b, _| {
            b.iter(|| {
                let flips = sweep.sweep(black_box(&pbits), black_box(&couplings));
                black_box(flips);
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches, 
    bench_metropolis_sweep, 
    bench_single_update, 
    bench_energy_compute,
    bench_temperature_scaling
);
criterion_main!(benches);
