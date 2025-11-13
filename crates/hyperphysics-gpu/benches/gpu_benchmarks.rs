//! Performance benchmarks for GPU vs CPU implementations
//!
//! Validates expected 10-800× speedup claims from blueprint.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use hyperphysics_gpu::GPUExecutor;
use pollster::FutureExt;

/// Create nearest-neighbor couplings for benchmark lattices
fn create_benchmark_couplings(size: usize) -> Vec<(usize, usize, f64)> {
    let mut couplings = Vec::new();

    // 1D chain with periodic boundary conditions
    for i in 0..size {
        let next = (i + 1) % size;
        couplings.push((i, next, 1.0));
        couplings.push((next, i, 1.0));
    }

    couplings
}

/// CPU reference: pBit update step
fn cpu_pbit_update(
    states: &mut [u32],
    biases: &[f32],
    couplings: &[(usize, usize, f64)],
    temperature: f32,
) {
    let beta = 1.0 / temperature;

    for i in 0..states.len() {
        // Compute effective field
        let mut field = biases[i] as f64;

        for &(src, dst, strength) in couplings.iter() {
            if src == i {
                let spin_j = (states[dst] as f64) * 2.0 - 1.0;
                field += strength * spin_j;
            }
        }

        // Gillespie update
        let p_flip = 1.0 / (1.0 + (-beta as f64 * field).exp());
        let random: f64 = rand::random();

        states[i] = if random < p_flip { 1 } else { 0 };
    }
}

/// CPU reference: energy calculation
fn cpu_compute_energy(states: &[u32], couplings: &[(usize, usize, f64)]) -> f64 {
    let mut energy = 0.0;

    for &(i, j, strength) in couplings.iter() {
        if i < j {
            let spin_i = (states[i] as f64) * 2.0 - 1.0;
            let spin_j = (states[j] as f64) * 2.0 - 1.0;
            energy -= strength * spin_i * spin_j;
        }
    }

    energy
}

/// CPU reference: entropy calculation
fn cpu_compute_entropy(states: &[u32]) -> f64 {
    let mut entropy = 0.0;

    for &state in states.iter() {
        let p = state as f64;
        let q = 1.0 - p;

        if p > 1e-10 {
            entropy -= p * p.ln();
        }
        if q > 1e-10 {
            entropy -= q * q.ln();
        }
    }

    entropy
}

fn bench_pbit_update_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("pbit_update_cpu");

    for size in [48, 1000, 10_000].iter() {
        let couplings = create_benchmark_couplings(*size);
        let mut states = vec![0u32; *size];
        let biases = vec![0.0f32; *size];

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &_size| {
                b.iter(|| {
                    cpu_pbit_update(
                        black_box(&mut states),
                        black_box(&biases),
                        black_box(&couplings),
                        black_box(1.0),
                    );
                });
            },
        );
    }

    group.finish();
}

fn bench_pbit_update_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("pbit_update_gpu");

    for size in [48, 1000, 10_000].iter() {
        let couplings = create_benchmark_couplings(*size);

        // Initialize GPU executor
        let mut executor = GPUExecutor::new(*size, &couplings)
            .block_on()
            .expect("GPU initialization failed");

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &_size| {
                b.iter(|| {
                    executor.step(black_box(1.0), black_box(0.01))
                        .block_on()
                        .expect("GPU step failed");
                });
            },
        );
    }

    group.finish();
}

fn bench_energy_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("energy_cpu");

    for size in [48, 1000, 10_000].iter() {
        let couplings = create_benchmark_couplings(*size);
        let states = vec![1u32; *size];

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &_size| {
                b.iter(|| {
                    cpu_compute_energy(
                        black_box(&states),
                        black_box(&couplings),
                    )
                });
            },
        );
    }

    group.finish();
}

fn bench_energy_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("energy_gpu");

    for size in [48, 1000, 10_000].iter() {
        let couplings = create_benchmark_couplings(*size);

        let mut executor = GPUExecutor::new(*size, &couplings)
            .block_on()
            .expect("GPU initialization failed");

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &_size| {
                b.iter(|| {
                    executor.compute_energy()
                        .block_on()
                        .expect("GPU energy failed")
                });
            },
        );
    }

    group.finish();
}

fn bench_entropy_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_cpu");

    for size in [48, 1000, 10_000].iter() {
        let states = vec![1u32; *size];

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &_size| {
                b.iter(|| {
                    cpu_compute_entropy(black_box(&states))
                });
            },
        );
    }

    group.finish();
}

fn bench_entropy_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_gpu");

    for size in [48, 1000, 10_000].iter() {
        let couplings = create_benchmark_couplings(*size);

        let mut executor = GPUExecutor::new(*size, &couplings)
            .block_on()
            .expect("GPU initialization failed");

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &_size| {
                b.iter(|| {
                    executor.compute_entropy()
                        .block_on()
                        .expect("GPU entropy failed")
                });
            },
        );
    }

    group.finish();
}

fn bench_end_to_end_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end_simulation");
    group.sample_size(10); // Fewer samples for long-running benchmarks

    for size in [1000, 10_000].iter() {
        let couplings = create_benchmark_couplings(*size);

        let mut executor = GPUExecutor::new(*size, &couplings)
            .block_on()
            .expect("GPU initialization failed");

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &_size| {
                b.iter(|| {
                    // Run 10 simulation steps + observables
                    for _ in 0..10 {
                        executor.step(black_box(1.0), black_box(0.01))
                            .block_on()
                            .expect("GPU step failed");
                    }

                    let _energy = executor.compute_energy()
                        .block_on()
                        .expect("GPU energy failed");

                    let _entropy = executor.compute_entropy()
                        .block_on()
                        .expect("GPU entropy failed");
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_pbit_update_cpu,
    bench_pbit_update_gpu,
    bench_energy_cpu,
    bench_energy_gpu,
    bench_entropy_cpu,
    bench_entropy_gpu,
    bench_end_to_end_simulation,
);

criterion_main!(benches);
