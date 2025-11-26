//! Performance Benchmarks for Conformal Prediction
//!
//! Measures:
//! - CPD generation performance
//! - PCP clustering overhead
//! - Streaming update latency
//! - Memory usage profiling
//! - Scalability tests

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use conformal_prediction::{
    ConformalPredictor, KNNNonconformity, NormalizedNonconformity,
};

// ====================================================================================
// Helper Functions
// ====================================================================================

fn generate_data(n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let x: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
    let y: Vec<f64> = (0..n).map(|i| (i as f64) * 2.0 + 1.0).collect();
    (x, y)
}

fn generate_multidim_data(n: usize, dims: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let x: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..dims).map(|j| (i + j) as f64).collect())
        .collect();
    let y: Vec<f64> = (0..n).map(|i| (i as f64) * 2.0).collect();
    (x, y)
}

// ====================================================================================
// Benchmark 1: CPD Generation Performance
// ====================================================================================

fn bench_cpd_calibration(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpd_calibration");

    for size in [100, 500, 1000, 2000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let (cal_x, cal_y) = generate_data(size);
            let mut measure = KNNNonconformity::new(5);
            measure.fit(&cal_x, &cal_y);

            b.iter(|| {
                let mut predictor = ConformalPredictor::new(0.1, measure.clone()).unwrap();
                predictor.calibrate(black_box(&cal_x), black_box(&cal_y)).unwrap();
                predictor
            });
        });
    }

    group.finish();
}

fn bench_cpd_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpd_prediction");

    let (cal_x, cal_y) = generate_data(500);
    let mut measure = KNNNonconformity::new(5);
    measure.fit(&cal_x, &cal_y);

    let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();
    predictor.calibrate(&cal_x, &cal_y).unwrap();

    group.bench_function("single_prediction", |b| {
        b.iter(|| {
            predictor.predict_interval(black_box(&[250.0]), black_box(500.0)).unwrap()
        });
    });

    group.bench_function("batch_prediction_100", |b| {
        let test_x: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64]).collect();

        b.iter(|| {
            for x in &test_x {
                predictor.predict_interval(black_box(x), black_box(x[0] * 2.0)).unwrap();
            }
        });
    });

    group.finish();
}

// ====================================================================================
// Benchmark 2: PCP Clustering Overhead
// ====================================================================================

fn bench_pcp_clustering(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcp_clustering");

    // Standard measure (baseline)
    let (cal_x, cal_y) = generate_data(500);

    group.bench_function("knn_nonconformity", |b| {
        let mut measure = KNNNonconformity::new(5);

        b.iter(|| {
            measure.fit(black_box(&cal_x), black_box(&cal_y));
            let mut predictor = ConformalPredictor::new(0.1, measure.clone()).unwrap();
            predictor.calibrate(&cal_x, &cal_y).unwrap();
            predictor
        });
    });

    // Normalized measure (with clustering)
    group.bench_function("normalized_nonconformity", |b| {
        let mut measure = NormalizedNonconformity::new(5);

        b.iter(|| {
            measure.fit(black_box(&cal_x), black_box(&cal_y));
            let mut predictor = ConformalPredictor::new(0.1, measure.clone()).unwrap();
            predictor.calibrate(&cal_x, &cal_y).unwrap();
            predictor
        });
    });

    group.finish();
}

fn bench_pcp_prediction_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcp_prediction_overhead");

    let (cal_x, cal_y) = generate_data(500);

    // Setup KNN predictor
    let mut knn_measure = KNNNonconformity::new(5);
    knn_measure.fit(&cal_x, &cal_y);
    let mut knn_predictor = ConformalPredictor::new(0.1, knn_measure).unwrap();
    knn_predictor.calibrate(&cal_x, &cal_y).unwrap();

    // Setup Normalized predictor
    let mut norm_measure = NormalizedNonconformity::new(5);
    norm_measure.fit(&cal_x, &cal_y);
    let mut norm_predictor = ConformalPredictor::new(0.1, norm_measure).unwrap();
    norm_predictor.calibrate(&cal_x, &cal_y).unwrap();

    group.bench_function("knn_prediction", |b| {
        b.iter(|| {
            knn_predictor.predict_interval(black_box(&[250.0]), black_box(500.0)).unwrap()
        });
    });

    group.bench_function("normalized_prediction", |b| {
        b.iter(|| {
            norm_predictor.predict_interval(black_box(&[250.0]), black_box(500.0)).unwrap()
        });
    });

    group.finish();
}

// ====================================================================================
// Benchmark 3: Streaming Update Latency
// ====================================================================================

fn bench_streaming_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_update");

    let (initial_x, initial_y) = generate_data(500);
    let (new_x, new_y) = generate_data(100);

    group.bench_function("recalibrate_500_samples", |b| {
        b.iter(|| {
            let mut measure = KNNNonconformity::new(5);
            measure.fit(black_box(&initial_x), black_box(&initial_y));

            let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();
            predictor.calibrate(&initial_x, &initial_y).unwrap();
            predictor
        });
    });

    group.bench_function("incremental_update_100_samples", |b| {
        let mut measure = KNNNonconformity::new(5);
        measure.fit(&initial_x, &initial_y);

        b.iter(|| {
            // Simulate incremental update by combining data
            let mut combined_x = initial_x.clone();
            combined_x.extend(new_x.iter().cloned());

            let mut combined_y = initial_y.clone();
            combined_y.extend(new_y.iter().cloned());

            measure.fit(black_box(&combined_x), black_box(&combined_y));

            let mut predictor = ConformalPredictor::new(0.1, measure.clone()).unwrap();
            predictor.calibrate(&combined_x, &combined_y).unwrap();
            predictor
        });
    });

    group.finish();
}

fn bench_sliding_window_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("sliding_window");

    for window_size in [100, 500, 1000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(window_size),
            &window_size,
            |b, &size| {
                let (cal_x, cal_y) = generate_data(size);
                let (new_x, new_y) = generate_data(50);

                b.iter(|| {
                    // Simulate sliding window: drop old, add new
                    let drop_count = new_x.len();
                    let mut window_x = cal_x[drop_count..].to_vec();
                    window_x.extend(new_x.iter().cloned());

                    let mut window_y = cal_y[drop_count..].to_vec();
                    window_y.extend(new_y.iter().cloned());

                    let mut measure = KNNNonconformity::new(5);
                    measure.fit(black_box(&window_x), black_box(&window_y));

                    let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();
                    predictor.calibrate(&window_x, &window_y).unwrap();
                    predictor
                });
            },
        );
    }

    group.finish();
}

// ====================================================================================
// Benchmark 4: Scalability Tests
// ====================================================================================

fn bench_scalability_data_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability_data_size");

    for size in [100, 500, 1000, 5000, 10000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let (cal_x, cal_y) = generate_data(size);

            b.iter(|| {
                let mut measure = KNNNonconformity::new(5);
                measure.fit(black_box(&cal_x), black_box(&cal_y));

                let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();
                predictor.calibrate(&cal_x, &cal_y).unwrap();

                // Make a prediction
                predictor.predict_interval(&[size as f64 / 2.0], (size as f64)).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_scalability_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability_dimensions");

    let n_samples = 500;

    for dims in [1, 5, 10, 20, 50] {
        group.bench_with_input(BenchmarkId::from_parameter(dims), &dims, |b, &dims| {
            let (cal_x, cal_y) = generate_multidim_data(n_samples, dims);
            let test_x: Vec<f64> = (0..dims).map(|i| i as f64).collect();

            b.iter(|| {
                let mut measure = KNNNonconformity::new(5);
                measure.fit(black_box(&cal_x), black_box(&cal_y));

                let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();
                predictor.calibrate(&cal_x, &cal_y).unwrap();

                predictor.predict_interval(black_box(&test_x), black_box(500.0)).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_scalability_k_neighbors(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability_k_neighbors");

    let (cal_x, cal_y) = generate_data(1000);

    for k in [1, 3, 5, 10, 20, 50] {
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            b.iter(|| {
                let mut measure = KNNNonconformity::new(k);
                measure.fit(black_box(&cal_x), black_box(&cal_y));

                let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();
                predictor.calibrate(&cal_x, &cal_y).unwrap();

                predictor.predict_interval(&[500.0], 1000.0).unwrap();
            });
        });
    }

    group.finish();
}

// ====================================================================================
// Benchmark 5: Prediction Set Performance
// ====================================================================================

fn bench_prediction_set_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("prediction_set");

    let (cal_x, cal_y) = generate_data(500);
    let mut measure = KNNNonconformity::new(5);
    measure.fit(&cal_x, &cal_y);

    let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();
    predictor.calibrate(&cal_x, &cal_y).unwrap();

    for n_candidates in [10, 50, 100, 500, 1000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(n_candidates),
            &n_candidates,
            |b, &n| {
                let candidates: Vec<f64> = (0..n).map(|i| i as f64).collect();

                b.iter(|| {
                    predictor.predict(black_box(&[250.0]), black_box(&candidates)).unwrap()
                });
            },
        );
    }

    group.finish();
}

// ====================================================================================
// Benchmark 6: Memory Usage Profiling
// ====================================================================================

fn bench_memory_footprint(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_footprint");

    // Measure memory overhead of different components

    group.bench_function("create_predictor_100", |b| {
        let (cal_x, cal_y) = generate_data(100);
        let mut measure = KNNNonconformity::new(5);
        measure.fit(&cal_x, &cal_y);

        b.iter(|| {
            let mut predictor = ConformalPredictor::new(0.1, measure.clone()).unwrap();
            predictor.calibrate(&cal_x, &cal_y).unwrap();
            black_box(predictor)
        });
    });

    group.bench_function("create_predictor_1000", |b| {
        let (cal_x, cal_y) = generate_data(1000);
        let mut measure = KNNNonconformity::new(5);
        measure.fit(&cal_x, &cal_y);

        b.iter(|| {
            let mut predictor = ConformalPredictor::new(0.1, measure.clone()).unwrap();
            predictor.calibrate(&cal_x, &cal_y).unwrap();
            black_box(predictor)
        });
    });

    group.bench_function("create_predictor_10000", |b| {
        let (cal_x, cal_y) = generate_data(10000);
        let mut measure = KNNNonconformity::new(5);
        measure.fit(&cal_x, &cal_y);

        b.iter(|| {
            let mut predictor = ConformalPredictor::new(0.1, measure.clone()).unwrap();
            predictor.calibrate(&cal_x, &cal_y).unwrap();
            black_box(predictor)
        });
    });

    group.finish();
}

// ====================================================================================
// Benchmark 7: Concurrent Prediction Throughput
// ====================================================================================

fn bench_concurrent_predictions(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_throughput");

    let (cal_x, cal_y) = generate_data(500);
    let mut measure = KNNNonconformity::new(5);
    measure.fit(&cal_x, &cal_y);

    let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();
    predictor.calibrate(&cal_x, &cal_y).unwrap();

    for n_predictions in [10, 50, 100, 500] {
        group.bench_with_input(
            BenchmarkId::from_parameter(n_predictions),
            &n_predictions,
            |b, &n| {
                let test_points: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();

                b.iter(|| {
                    for x in &test_points {
                        predictor.predict_interval(black_box(x), black_box(x[0] * 2.0)).unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

// ====================================================================================
// Benchmark Groups
// ====================================================================================

criterion_group!(
    cpd_benches,
    bench_cpd_calibration,
    bench_cpd_prediction,
);

criterion_group!(
    pcp_benches,
    bench_pcp_clustering,
    bench_pcp_prediction_overhead,
);

criterion_group!(
    streaming_benches,
    bench_streaming_update,
    bench_sliding_window_update,
);

criterion_group!(
    scalability_benches,
    bench_scalability_data_size,
    bench_scalability_dimensions,
    bench_scalability_k_neighbors,
);

criterion_group!(
    prediction_benches,
    bench_prediction_set_size,
    bench_concurrent_predictions,
);

criterion_group!(
    memory_benches,
    bench_memory_footprint,
);

criterion_main!(
    cpd_benches,
    pcp_benches,
    streaming_benches,
    scalability_benches,
    prediction_benches,
    memory_benches,
);
