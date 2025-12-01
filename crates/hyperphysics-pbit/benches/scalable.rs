//! Benchmarks for scalable pBit implementation
//!
//! Run with: cargo bench -p hyperphysics-pbit --bench scalable

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hyperphysics_pbit::scalable::{MetropolisSweep, ScalableCouplings, ScalablePBitArray};

fn create_sparse_system(n: usize, avg_degree: usize, seed: u64) -> (ScalablePBitArray, ScalableCouplings, Vec<f32>) {
    let states = ScalablePBitArray::random(n, seed);
    let mut couplings = ScalableCouplings::with_capacity(n, n * avg_degree);
    
    let mut rng = fastrand::Rng::with_seed(seed + 1);
    let target_edges = n * avg_degree / 2;
    
    for _ in 0..target_edges {
        let i = rng.usize(0..n);
        let j = rng.usize(0..n);
        if i != j {
            couplings.add_symmetric(i, j, rng.f32() * 2.0 - 1.0);
        }
    }
    couplings.finalize();
    
    let biases = vec![0.0f32; n];
    (states, couplings, biases)
}

fn bench_sweep_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalable_sweep");
    
    for size in [64, 256, 1024, 4096, 16384, 65536] {
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let (mut states, couplings, biases) = create_sparse_system(size, 8, 42);
            let mut sweep = MetropolisSweep::new(1.0, 42);
            
            b.iter(|| {
                let stats = sweep.execute(black_box(&mut states), &couplings, &biases);
                black_box(stats.flips)
            });
        });
    }
    
    group.finish();
}

fn bench_vs_old(c: &mut Criterion) {
    use hyperphysics_pbit::{CouplingNetwork, MetropolisSimulator, PBitLattice};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    
    let mut group = c.benchmark_group("scalable_vs_geometric");
    
    // Small system where both can run
    let size = 29; // {3,7,2} tessellation size
    
    // Geometric lattice
    group.bench_function("geometric_29", |b| {
        let mut lattice = PBitLattice::roi_48(1.0).unwrap();
        let coupling = CouplingNetwork::new(1.0, 1.0, 0.01);
        coupling.build_couplings(&mut lattice).unwrap();
        let mut sim = MetropolisSimulator::new(lattice, 1.0);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        
        b.iter(|| {
            sim.step(&mut rng).unwrap()
        });
    });
    
    // Scalable (same size)
    group.bench_function("scalable_29", |b| {
        let (mut states, couplings, biases) = create_sparse_system(29, 8, 42);
        let mut sweep = MetropolisSweep::new(1.0, 42);
        
        b.iter(|| {
            // Single step (not full sweep) for fair comparison
            let delta_e = couplings.delta_energy(0, &states, biases[0]);
            if delta_e <= 0.0 {
                states.flip(0);
            }
            black_box(delta_e)
        });
    });
    
    // Full sweep comparison
    group.bench_function("geometric_29_sweep", |b| {
        let mut lattice = PBitLattice::roi_48(1.0).unwrap();
        let coupling = CouplingNetwork::new(1.0, 1.0, 0.01);
        coupling.build_couplings(&mut lattice).unwrap();
        let n = lattice.size();
        let mut sim = MetropolisSimulator::new(lattice, 1.0);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        
        b.iter(|| {
            for _ in 0..n {
                sim.step(&mut rng).unwrap();
            }
        });
    });
    
    group.bench_function("scalable_29_sweep", |b| {
        let (mut states, couplings, biases) = create_sparse_system(29, 8, 42);
        let mut sweep = MetropolisSweep::new(1.0, 42);
        
        b.iter(|| {
            sweep.execute(&mut states, &couplings, &biases)
        });
    });
    
    group.finish();
}

fn bench_large_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_scale");
    group.sample_size(20); // Fewer samples for large systems
    
    // 100K pBits
    group.bench_function("100k_sweep", |b| {
        let (mut states, couplings, biases) = create_sparse_system(100_000, 10, 42);
        let mut sweep = MetropolisSweep::new(1.0, 42);
        
        b.iter(|| {
            sweep.execute(black_box(&mut states), &couplings, &biases)
        });
    });
    
    group.finish();
}

criterion_group!(benches, bench_sweep_scaling, bench_vs_old, bench_large_scale);
criterion_main!(benches);
