//! SIMD Performance Benchmarks
//!
//! Run with: cargo bench --bench simd_benchmarks
//!
//! Expected speedups:
//! - Matrix multiplication: 2-3x (AVX2), 3-4x (AVX-512)
//! - Activation functions: 2-3x
//! - Loss calculations: 2-4x

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use neuro_divergent::optimizations::simd::{matmul, activations, losses, utils};

fn benchmark_matrix_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    for size in [64, 128, 256, 512].iter() {
        let a: Vec<Vec<f32>> = (0..*size)
            .map(|_| (0..*size).map(|_| rand::random::<f32>()).collect())
            .collect();
        let b: Vec<Vec<f32>> = (0..*size)
            .map(|_| (0..*size).map(|_| rand::random::<f32>()).collect())
            .collect();

        group.throughput(Throughput::Elements((size * size * size) as u64));

        group.bench_with_input(BenchmarkId::new("gemm", size), size, |bencher, _| {
            bencher.iter(|| {
                matmul::gemm(black_box(&a), black_box(&b))
            });
        });
    }

    group.finish();
}

fn benchmark_matrix_vector(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemv");

    for size in [128, 256, 512, 1024].iter() {
        let a: Vec<Vec<f32>> = (0..*size)
            .map(|_| (0..*size).map(|_| rand::random::<f32>()).collect())
            .collect();
        let x: Vec<f32> = (0..*size).map(|_| rand::random::<f32>()).collect();

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(BenchmarkId::new("gemv", size), size, |bencher, _| {
            bencher.iter(|| {
                matmul::gemv(black_box(&a), black_box(&x))
            });
        });
    }

    group.finish();
}

fn benchmark_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");

    for size in [128, 256, 512, 1024, 2048].iter() {
        let a: Vec<f32> = (0..*size).map(|_| rand::random::<f32>()).collect();
        let b: Vec<f32> = (0..*size).map(|_| rand::random::<f32>()).collect();

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("dot", size), size, |bencher, _| {
            bencher.iter(|| {
                matmul::dot_product(black_box(&a), black_box(&b))
            });
        });
    }

    group.finish();
}

fn benchmark_activations(c: &mut Criterion) {
    let mut group = c.benchmark_group("activations");

    for size in [1024, 4096, 16384].iter() {
        let x: Vec<f32> = (0..*size).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect();

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("relu", size), &x, |bencher, x| {
            bencher.iter(|| activations::relu(black_box(x)));
        });

        group.bench_with_input(BenchmarkId::new("gelu", size), &x, |bencher, x| {
            bencher.iter(|| activations::gelu(black_box(x)));
        });

        group.bench_with_input(BenchmarkId::new("tanh", size), &x, |bencher, x| {
            bencher.iter(|| activations::tanh(black_box(x)));
        });

        group.bench_with_input(BenchmarkId::new("sigmoid", size), &x, |bencher, x| {
            bencher.iter(|| activations::sigmoid(black_box(x)));
        });

        group.bench_with_input(BenchmarkId::new("softmax", size), &x, |bencher, x| {
            bencher.iter(|| activations::softmax(black_box(x)));
        });

        group.bench_with_input(BenchmarkId::new("leaky_relu", size), &x, |bencher, x| {
            bencher.iter(|| activations::leaky_relu(black_box(x), 0.01));
        });
    }

    group.finish();
}

fn benchmark_losses(c: &mut Criterion) {
    let mut group = c.benchmark_group("losses");

    for size in [1024, 4096, 16384].iter() {
        let pred: Vec<f32> = (0..*size).map(|_| rand::random::<f32>()).collect();
        let target: Vec<f32> = (0..*size).map(|_| rand::random::<f32>()).collect();

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("mse", size), &(&pred, &target), |bencher, (p, t)| {
            bencher.iter(|| losses::mse(black_box(p), black_box(t)));
        });

        group.bench_with_input(BenchmarkId::new("mae", size), &(&pred, &target), |bencher, (p, t)| {
            bencher.iter(|| losses::mae(black_box(p), black_box(t)));
        });

        group.bench_with_input(BenchmarkId::new("mse_gradient", size), &(&pred, &target), |bencher, (p, t)| {
            bencher.iter(|| losses::mse_gradient(black_box(p), black_box(t)));
        });

        group.bench_with_input(BenchmarkId::new("mae_gradient", size), &(&pred, &target), |bencher, (p, t)| {
            bencher.iter(|| losses::mae_gradient(black_box(p), black_box(t)));
        });

        group.bench_with_input(BenchmarkId::new("huber_loss", size), &(&pred, &target), |bencher, (p, t)| {
            bencher.iter(|| losses::huber_loss(black_box(p), black_box(t), 1.0));
        });
    }

    group.finish();
}

fn benchmark_utils(c: &mut Criterion) {
    let mut group = c.benchmark_group("utils");

    for size in [1024, 4096, 16384].iter() {
        let x: Vec<f32> = (0..*size).map(|_| rand::random::<f32>()).collect();

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("reduce_sum", size), &x, |bencher, x| {
            bencher.iter(|| utils::reduce_sum(black_box(x)));
        });

        group.bench_with_input(BenchmarkId::new("reduce_max", size), &x, |bencher, x| {
            bencher.iter(|| utils::reduce_max(black_box(x)));
        });

        group.bench_with_input(BenchmarkId::new("scalar_mul", size), &x, |bencher, x| {
            bencher.iter(|| utils::scalar_mul(black_box(x), 2.0));
        });

        group.bench_with_input(BenchmarkId::new("norm_l2", size), &x, |bencher, x| {
            bencher.iter(|| utils::norm_l2(black_box(x)));
        });

        group.bench_with_input(BenchmarkId::new("clamp", size), &x, |bencher, x| {
            bencher.iter(|| utils::clamp(black_box(x), -1.0, 1.0));
        });
    }

    group.finish();
}

fn benchmark_vector_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_ops");

    for size in [1024, 4096, 16384].iter() {
        let a: Vec<f32> = (0..*size).map(|_| rand::random::<f32>()).collect();
        let b: Vec<f32> = (0..*size).map(|_| rand::random::<f32>()).collect();

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("vec_add", size), &(&a, &b), |bencher, (a, b)| {
            bencher.iter(|| matmul::vec_add(black_box(a), black_box(b)));
        });

        group.bench_with_input(BenchmarkId::new("vec_mul", size), &(&a, &b), |bencher, (a, b)| {
            bencher.iter(|| matmul::vec_mul(black_box(a), black_box(b)));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_matrix_multiplication,
    benchmark_matrix_vector,
    benchmark_dot_product,
    benchmark_activations,
    benchmark_losses,
    benchmark_utils,
    benchmark_vector_ops,
);

criterion_main!(benches);
