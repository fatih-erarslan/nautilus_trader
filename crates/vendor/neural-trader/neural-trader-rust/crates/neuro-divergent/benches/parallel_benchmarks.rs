//! Comprehensive parallel processing benchmarks
//!
//! Benchmarks parallel vs sequential performance for:
//! - Batch inference (1, 4, 8, 16 threads)
//! - Data preprocessing
//! - Gradient computation
//! - Cross-validation
//!
//! Run with: cargo bench --bench parallel_benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, PlotConfiguration, AxisScale};
use neuro_divergent::optimizations::parallel::*;
use neuro_divergent::optimizations::parallel::benchmark::*;
use ndarray::{Array1, Array2, Array};
use rand::Rng;

/// Generate random time series data
fn generate_random_data(n_samples: usize, n_features: usize) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    Array::from_shape_fn((n_samples, n_features), |_| rng.gen_range(-1.0..1.0))
}

/// Generate random batches
fn generate_batches(n_batches: usize, batch_size: usize, n_features: usize) -> Vec<Array2<f64>> {
    (0..n_batches)
        .map(|_| generate_random_data(batch_size, n_features))
        .collect()
}

/// Simple mock inference function
fn mock_inference(batch: &Array2<f64>) -> neuro_divergent::Result<Vec<f64>> {
    // Simulate some computation
    let mut result = vec![0.0; batch.nrows()];
    for i in 0..batch.nrows() {
        result[i] = batch.row(i).sum();
        // Add some computational cost
        for _ in 0..10 {
            result[i] = (result[i] * 1.01).sin();
        }
    }
    Ok(result)
}

/// Sequential batch inference (baseline)
fn sequential_batch_inference(batches: &[Array2<f64>]) -> Vec<Vec<f64>> {
    batches
        .iter()
        .map(|batch| mock_inference(batch).unwrap())
        .collect()
}

/// Benchmark batch inference with different thread counts
fn benchmark_batch_inference_threads(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_inference_threads");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    let n_batches = 100;
    let batch_size = 32;
    let n_features = 128;
    let batches = generate_batches(n_batches, batch_size, n_features);

    // Sequential baseline
    group.bench_function("sequential", |b| {
        b.iter(|| sequential_batch_inference(black_box(&batches)))
    });

    // Parallel with different thread counts
    for num_threads in [1, 2, 4, 8, 16] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .ok();

        group.bench_with_input(
            BenchmarkId::new("parallel", num_threads),
            &num_threads,
            |b, _| {
                b.iter(|| {
                    parallel_batch_inference(black_box(&batches), mock_inference).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark batch inference scaling
fn benchmark_batch_inference_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_inference_scaling");

    let batch_size = 32;
    let n_features = 128;

    // Test different numbers of batches
    for n_batches in [10, 50, 100, 200, 500] {
        let batches = generate_batches(n_batches, batch_size, n_features);

        group.bench_with_input(
            BenchmarkId::new("parallel_8_threads", n_batches),
            &n_batches,
            |b, _| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(8)
                    .build_global()
                    .ok();

                b.iter(|| {
                    parallel_batch_inference(black_box(&batches), mock_inference).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark data preprocessing
fn benchmark_parallel_preprocess(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_preprocessing");

    let n_chunks = 50;
    let chunk_size = 1000;
    let n_features = 64;
    let chunks: Vec<Array2<f64>> = (0..n_chunks)
        .map(|_| generate_random_data(chunk_size, n_features))
        .collect();

    // Preprocessing function: normalization
    let preprocess_fn = |chunk: &Array2<f64>| -> neuro_divergent::Result<Array2<f64>> {
        let mut normalized = chunk.clone();
        for col in 0..chunk.ncols() {
            let column = chunk.column(col);
            let mean = column.mean().unwrap_or(0.0);
            let std = column.std(0.0);
            if std > 0.0 {
                for row in 0..chunk.nrows() {
                    normalized[[row, col]] = (chunk[[row, col]] - mean) / std;
                }
            }
        }
        Ok(normalized)
    };

    // Sequential
    group.bench_function("sequential", |b| {
        b.iter(|| {
            let _: Vec<_> = chunks.iter()
                .map(|chunk| preprocess_fn(black_box(chunk)).unwrap())
                .collect();
        })
    });

    // Parallel
    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()
        .ok();

    group.bench_function("parallel_8_threads", |b| {
        b.iter(|| {
            parallel_preprocess(black_box(&chunks), preprocess_fn).unwrap()
        })
    });

    group.finish();
}

/// Benchmark gradient computation
fn benchmark_parallel_gradients(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_computation");

    let n_batches = 50;
    let batch_size = 32;
    let n_features = 64;

    let batches: Vec<(Array2<f64>, Array1<f64>)> = (0..n_batches)
        .map(|_| {
            let x = generate_random_data(batch_size, n_features);
            let y = Array1::from_vec(vec![0.0; batch_size]);
            (x, y)
        })
        .collect();

    // Mock gradient function
    let gradient_fn = |x: &Array2<f64>, _y: &Array1<f64>| -> neuro_divergent::Result<Vec<Array2<f64>>> {
        // Simulate gradient computation
        let grad = x * 0.01;
        Ok(vec![grad])
    };

    // Sequential
    group.bench_function("sequential", |b| {
        b.iter(|| {
            let _: Vec<_> = batches.iter()
                .map(|(x, y)| gradient_fn(black_box(x), black_box(y)).unwrap())
                .collect();
        })
    });

    // Parallel
    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()
        .ok();

    group.bench_function("parallel_8_threads", |b| {
        b.iter(|| {
            parallel_gradient_computation(black_box(&batches), gradient_fn).unwrap()
        })
    });

    group.finish();
}

/// Benchmark cross-validation
fn benchmark_parallel_cross_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_validation");
    group.sample_size(10); // Reduce sample size for expensive operation

    let n_samples = 1000;
    let n_features = 32;
    let data = generate_random_data(n_samples, n_features);
    let labels = Array1::from_vec(vec![0.0; n_samples]);

    // Mock train and evaluate function
    let train_eval_fn = |train_x: &Array2<f64>,
                         _train_y: &Array1<f64>,
                         val_x: &Array2<f64>,
                         _val_y: &Array1<f64>|
     -> neuro_divergent::Result<f64> {
        // Simulate training
        let mut loss = 0.0;
        for i in 0..10 {
            loss += (train_x.sum() * 0.001).sin() * (i as f64);
        }
        // Simulate validation
        loss += val_x.sum() * 0.0001;
        Ok(loss)
    };

    // 5-fold CV sequential
    group.bench_function("5_fold_sequential", |b| {
        b.iter(|| {
            let k_folds = 5;
            let fold_size = n_samples / k_folds;
            let mut scores = Vec::new();

            for fold_idx in 0..k_folds {
                let val_start = fold_idx * fold_size;
                let val_end = if fold_idx == k_folds - 1 {
                    n_samples
                } else {
                    (fold_idx + 1) * fold_size
                };

                let val_data = data.slice(ndarray::s![val_start..val_end, ..]).to_owned();
                let val_labels = labels.slice(ndarray::s![val_start..val_end]).to_owned();

                let mut train_indices = Vec::new();
                train_indices.extend(0..val_start);
                train_indices.extend(val_end..n_samples);

                let train_data = data.select(ndarray::Axis(0), &train_indices);
                let train_labels = labels.select(ndarray::Axis(0), &train_indices);

                let score = train_eval_fn(
                    black_box(&train_data),
                    black_box(&train_labels),
                    black_box(&val_data),
                    black_box(&val_labels),
                ).unwrap();
                scores.push(score);
            }
        })
    });

    // 5-fold CV parallel
    rayon::ThreadPoolBuilder::new()
        .num_threads(5)
        .build_global()
        .ok();

    group.bench_function("5_fold_parallel", |b| {
        b.iter(|| {
            parallel_cross_validation(
                black_box(&data),
                black_box(&labels),
                5,
                train_eval_fn,
            ).unwrap()
        })
    });

    group.finish();
}

/// Scalability analysis benchmark
fn benchmark_scalability_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability_analysis");
    group.sample_size(20);

    let n_batches = 200;
    let batch_size = 32;
    let n_features = 128;
    let batches = generate_batches(n_batches, batch_size, n_features);

    // Create workload
    let workload = || -> neuro_divergent::Result<()> {
        parallel_batch_inference(&batches, mock_inference)?;
        Ok(())
    };

    group.bench_function("full_scalability_test", |b| {
        b.iter(|| {
            let thread_counts = vec![1, 2, 4, 8, 16];
            let results = scalability_benchmark(black_box(&thread_counts), &workload).unwrap();

            // Verify we got results
            assert_eq!(results.len(), 5);

            // Print results (only in test mode)
            if false {
                print_benchmark_results(&results);
                let parallel_fraction = amdahl_analysis(&results);
                println!("Estimated parallel fraction: {:.2}%", parallel_fraction * 100.0);
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_batch_inference_threads,
    benchmark_batch_inference_scaling,
    benchmark_parallel_preprocess,
    benchmark_parallel_gradients,
    benchmark_parallel_cross_validation,
    benchmark_scalability_analysis,
);
criterion_main!(benches);
