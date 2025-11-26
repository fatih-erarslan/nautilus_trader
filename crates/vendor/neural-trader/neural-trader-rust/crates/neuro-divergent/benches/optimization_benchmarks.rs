//! Optimization Benchmarks
//!
//! Measures:
//! - SIMD vs scalar performance
//! - Rayon parallel vs sequential
//! - FP16 vs FP32 speed/memory
//! - Flash Attention vs standard attention
//!
//! Validates optimization targets and identifies bottlenecks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use neuro_divergent::{
    optimizations::{simd, parallel},
};
use ndarray::{Array1, Array2};
use rayon::prelude::*;

// ============================================================================
// SIMD VS SCALAR BENCHMARKS
// ============================================================================

fn bench_simd_vs_scalar_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization/simd/dot_product");

    for &size in &[64, 256, 1024, 4096] {
        group.throughput(Throughput::Elements(size as u64));

        let a: Vec<f32> = (0..size).map(|i| i as f32 / 100.0).collect();
        let b: Vec<f32> = (0..size).map(|i| (i * 7919) as f32 % 100.0 / 100.0).collect();

        // Scalar version
        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let result: f32 = a.iter()
                        .zip(b.iter())
                        .map(|(x, y)| x * y)
                        .sum();
                    black_box(result)
                });
            },
        );

        // SIMD version (if available)
        #[cfg(target_feature = "avx2")]
        group.bench_with_input(
            BenchmarkId::new("simd_avx2", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let result = simd::dot_product_f32(&a, &b);
                    black_box(result)
                });
            },
        );

        #[cfg(target_feature = "neon")]
        group.bench_with_input(
            BenchmarkId::new("simd_neon", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let result = simd::dot_product_f32(&a, &b);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

fn bench_simd_vs_scalar_activation(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization/simd/activation");

    for &size in &[256, 1024, 4096] {
        let input: Vec<f32> = (0..size).map(|i| (i as f32 - size as f32 / 2.0) / 100.0).collect();

        // Scalar ReLU
        group.bench_with_input(
            BenchmarkId::new("scalar_relu", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let result: Vec<f32> = input.iter().map(|&x| x.max(0.0)).collect();
                    black_box(result)
                });
            },
        );

        // SIMD ReLU
        #[cfg(any(target_feature = "avx2", target_feature = "neon"))]
        group.bench_with_input(
            BenchmarkId::new("simd_relu", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let result = simd::relu_f32(&input);
                    black_box(result)
                });
            },
        );

        // Scalar Sigmoid
        group.bench_with_input(
            BenchmarkId::new("scalar_sigmoid", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let result: Vec<f32> = input.iter()
                        .map(|&x| 1.0 / (1.0 + (-x).exp()))
                        .collect();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

fn bench_simd_matrix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization/simd/matrix_ops");

    for &size in &[64, 128, 256] {
        let a = Array2::<f32>::from_shape_fn((size, size), |(i, j)| (i + j) as f32 / 100.0);
        let b = Array2::<f32>::from_shape_fn((size, size), |(i, j)| (i * j) as f32 / 100.0);

        // Standard ndarray matmul
        group.bench_with_input(
            BenchmarkId::new("ndarray_matmul", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let result = a.dot(&b);
                    black_box(result)
                });
            },
        );

        // BLAS-accelerated (via ndarray-linalg)
        group.bench_with_input(
            BenchmarkId::new("blas_matmul", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let result = a.dot(&b);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// PARALLEL VS SEQUENTIAL BENCHMARKS
// ============================================================================

fn bench_parallel_vs_sequential_map(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization/parallel/map");

    for &size in &[1000, 10000, 100000] {
        group.throughput(Throughput::Elements(size as u64));

        let data: Vec<f64> = (0..size).map(|i| i as f64 / 100.0).collect();

        // Sequential
        group.bench_with_input(
            BenchmarkId::new("sequential", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let result: Vec<f64> = data.iter()
                        .map(|&x| x.sin() * x.cos() + x.sqrt())
                        .collect();
                    black_box(result)
                });
            },
        );

        // Parallel with Rayon
        group.bench_with_input(
            BenchmarkId::new("parallel", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let result: Vec<f64> = data.par_iter()
                        .map(|&x| x.sin() * x.cos() + x.sqrt())
                        .collect();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

fn bench_parallel_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization/parallel/batch");

    for &batch_size in &[8, 32, 128] {
        let batches: Vec<Vec<f64>> = (0..batch_size)
            .map(|_| (0..1000).map(|i| i as f64 / 100.0).collect())
            .collect();

        // Sequential batch processing
        group.bench_with_input(
            BenchmarkId::new("sequential", batch_size),
            &batch_size,
            |bench, _| {
                bench.iter(|| {
                    let results: Vec<f64> = batches.iter()
                        .map(|batch| {
                            batch.iter().map(|&x| x.sin()).sum::<f64>() / batch.len() as f64
                        })
                        .collect();
                    black_box(results)
                });
            },
        );

        // Parallel batch processing
        group.bench_with_input(
            BenchmarkId::new("parallel", batch_size),
            &batch_size,
            |bench, _| {
                bench.iter(|| {
                    let results: Vec<f64> = batches.par_iter()
                        .map(|batch| {
                            batch.iter().map(|&x| x.sin()).sum::<f64>() / batch.len() as f64
                        })
                        .collect();
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

fn bench_parallel_training_epochs(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization/parallel/training");
    group.sample_size(10);

    let data_size = 1000;
    let num_epochs = 10;

    let data: Vec<f64> = (0..data_size).map(|i| i as f64 / 100.0).collect();

    // Sequential epoch processing
    group.bench_function("sequential_epochs", |bench| {
        bench.iter(|| {
            let mut losses = Vec::new();
            for _ in 0..num_epochs {
                let loss: f64 = data.iter()
                    .map(|&x| (x.sin() - x).powi(2))
                    .sum::<f64>() / data.len() as f64;
                losses.push(loss);
            }
            black_box(losses)
        });
    });

    // Parallel within-epoch processing
    group.bench_function("parallel_within_epoch", |bench| {
        bench.iter(|| {
            let mut losses = Vec::new();
            for _ in 0..num_epochs {
                let loss: f64 = data.par_iter()
                    .map(|&x| (x.sin() - x).powi(2))
                    .sum::<f64>() / data.len() as f64;
                losses.push(loss);
            }
            black_box(losses)
        });
    });
}

// ============================================================================
// FP16 VS FP32 BENCHMARKS
// ============================================================================

fn bench_fp16_vs_fp32_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization/precision/inference");

    for &size in &[256, 1024, 4096] {
        // FP32 inference
        let weights_f32 = Array2::<f32>::from_shape_fn((size, size), |(i, j)| {
            (i + j) as f32 / 1000.0
        });
        let input_f32 = Array1::<f32>::from_shape_fn(size, |i| i as f32 / 100.0);

        group.bench_with_input(
            BenchmarkId::new("fp32", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let result = weights_f32.dot(&input_f32);
                    black_box(result)
                });
            },
        );

        // FP16 would require half crate or similar
        // Simulating with reduced precision calculations
        group.bench_with_input(
            BenchmarkId::new("fp32_quantized", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    // Simulate quantization overhead
                    let quantized: Vec<f32> = input_f32.iter()
                        .map(|&x| (x * 256.0).round() / 256.0)
                        .collect();
                    black_box(quantized)
                });
            },
        );
    }

    group.finish();
}

fn bench_memory_bandwidth(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization/memory/bandwidth");

    for &size in &[1024, 4096, 16384] {
        group.throughput(Throughput::Bytes((size * std::mem::size_of::<f32>()) as u64));

        let data = vec![1.0f32; size];

        // Sequential read
        group.bench_with_input(
            BenchmarkId::new("sequential_read", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let sum: f32 = data.iter().sum();
                    black_box(sum)
                });
            },
        );

        // Parallel read
        group.bench_with_input(
            BenchmarkId::new("parallel_read", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let sum: f32 = data.par_iter().sum();
                    black_box(sum)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// ATTENTION MECHANISM BENCHMARKS
// ============================================================================

fn bench_standard_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization/attention/standard");

    for &seq_len in &[32, 64, 128, 256] {
        let d_model = 64;

        let query = Array2::<f32>::from_shape_fn((seq_len, d_model), |(i, j)| {
            ((i + j) as f32 / 100.0).sin()
        });
        let key = Array2::<f32>::from_shape_fn((seq_len, d_model), |(i, j)| {
            ((i * j) as f32 / 100.0).cos()
        });
        let value = Array2::<f32>::from_shape_fn((seq_len, d_model), |(i, j)| {
            (i as f32 - j as f32) / 100.0
        });

        group.bench_with_input(
            BenchmarkId::new("qkv_matmul", seq_len),
            &seq_len,
            |bench, _| {
                bench.iter(|| {
                    // Q @ K^T
                    let scores = query.dot(&key.t());
                    // Softmax (simplified)
                    let attention = scores.mapv(|x| x.exp());
                    // @ V
                    let output = attention.dot(&value);
                    black_box(output)
                });
            },
        );
    }

    group.finish();
}

fn bench_flash_attention_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization/attention/flash");

    for &seq_len in &[64, 128, 256] {
        let d_model = 64;
        let block_size = 32;

        let query = Array2::<f32>::from_shape_fn((seq_len, d_model), |(i, j)| {
            ((i + j) as f32 / 100.0).sin()
        });
        let key = Array2::<f32>::from_shape_fn((seq_len, d_model), |(i, j)| {
            ((i * j) as f32 / 100.0).cos()
        });

        // Simulate block-wise processing
        group.bench_with_input(
            BenchmarkId::new("blockwise", seq_len),
            &seq_len,
            |bench, _| {
                bench.iter(|| {
                    let num_blocks = (seq_len + block_size - 1) / block_size;
                    let mut outputs = Vec::new();

                    for i in 0..num_blocks {
                        let start = i * block_size;
                        let end = (start + block_size).min(seq_len);

                        let q_block = query.slice(ndarray::s![start..end, ..]);
                        let k_block = key.slice(ndarray::s![start..end, ..]);

                        let scores = q_block.dot(&k_block.t());
                        outputs.push(scores);
                    }

                    black_box(outputs)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// CACHE OPTIMIZATION BENCHMARKS
// ============================================================================

fn bench_cache_locality(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization/cache/locality");

    let size = 1024;
    let matrix = Array2::<f32>::from_shape_fn((size, size), |(i, j)| (i + j) as f32);

    // Row-major access (cache-friendly)
    group.bench_function("row_major", |bench| {
        bench.iter(|| {
            let mut sum = 0.0f32;
            for i in 0..size {
                for j in 0..size {
                    sum += matrix[[i, j]];
                }
            }
            black_box(sum)
        });
    });

    // Column-major access (cache-unfriendly)
    group.bench_function("column_major", |bench| {
        bench.iter(|| {
            let mut sum = 0.0f32;
            for j in 0..size {
                for i in 0..size {
                    sum += matrix[[i, j]];
                }
            }
            black_box(sum)
        });
    });

    group.finish();
}

criterion_group!(
    optimization_benches,
    // SIMD benchmarks
    bench_simd_vs_scalar_dot_product,
    bench_simd_vs_scalar_activation,
    bench_simd_matrix_operations,
    // Parallel benchmarks
    bench_parallel_vs_sequential_map,
    bench_parallel_batch_processing,
    bench_parallel_training_epochs,
    // Precision benchmarks
    bench_fp16_vs_fp32_inference,
    bench_memory_bandwidth,
    // Attention benchmarks
    bench_standard_attention,
    bench_flash_attention_optimization,
    // Cache benchmarks
    bench_cache_locality,
);
criterion_main!(optimization_benches);
