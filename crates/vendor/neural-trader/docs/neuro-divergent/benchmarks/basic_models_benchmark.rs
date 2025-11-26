// Comprehensive Benchmark Suite for Basic Models
// Place in: neural-trader-rust/crates/neuro-divergent/benches/basic_models.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, PlotConfiguration, AxisScale};
use neuro_divergent::{
    models::basic::{MLP, DLinear, NLinear, MLPMultivariate},
    NeuralModel, ModelConfig, TimeSeriesDataFrame,
};
use std::time::Duration;

/// Generate synthetic time series data
fn generate_data(n_samples: usize, with_noise: bool) -> Vec<f64> {
    (0..n_samples)
        .map(|i| {
            let trend = i as f64 * 0.1;
            let seasonal = (i as f64 * 0.05).sin() * 10.0;
            let noise = if with_noise {
                rand::random::<f64>() * 2.0 - 1.0
            } else {
                0.0
            };
            100.0 + trend + seasonal + noise
        })
        .collect()
}

/// Benchmark: Training Time vs Dataset Size
fn bench_training_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("training_time");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    let dataset_sizes = vec![100, 500, 1000, 5000, 10000];
    let config = ModelConfig::default()
        .with_input_size(24)
        .with_horizon(12)
        .with_hidden_size(128);

    for size in dataset_sizes.iter() {
        // MLP Training
        group.bench_with_input(
            BenchmarkId::new("MLP", size),
            size,
            |b, &size| {
                let data = generate_data(size, true);
                let df = TimeSeriesDataFrame::from_values(data, None).unwrap();

                b.iter(|| {
                    let mut model = MLP::new(config.clone());
                    model.fit(&df).unwrap();
                    black_box(model);
                });
            },
        );

        // DLinear Training (current naive implementation)
        group.bench_with_input(
            BenchmarkId::new("DLinear", size),
            size,
            |b, &size| {
                let data = generate_data(size, true);
                let df = TimeSeriesDataFrame::from_values(data, None).unwrap();

                b.iter(|| {
                    let mut model = DLinear::new(config.clone());
                    model.fit(&df).unwrap();
                    black_box(model);
                });
            },
        );

        // NLinear Training
        group.bench_with_input(
            BenchmarkId::new("NLinear", size),
            size,
            |b, &size| {
                let data = generate_data(size, true);
                let df = TimeSeriesDataFrame::from_values(data, None).unwrap();

                b.iter(|| {
                    let mut model = NLinear::new(config.clone());
                    model.fit(&df).unwrap();
                    black_box(model);
                });
            },
        );

        // MLPMultivariate Training
        group.bench_with_input(
            BenchmarkId::new("MLPMultivariate", size),
            size,
            |b, &size| {
                let data = generate_data(size, true);
                let df = TimeSeriesDataFrame::from_values(data, None).unwrap();

                b.iter(|| {
                    let mut model = MLPMultivariate::new(config.clone());
                    model.fit(&df).unwrap();
                    black_box(model);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Inference Latency vs Horizon
fn bench_inference_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference_latency");
    group.measurement_time(Duration::from_secs(10));

    let horizons = vec![1, 6, 12, 24, 48, 96];

    // Pre-train models
    let data = generate_data(1000, true);
    let df = TimeSeriesDataFrame::from_values(data, None).unwrap();

    let base_config = ModelConfig::default()
        .with_input_size(24)
        .with_hidden_size(128);

    for horizon in horizons.iter() {
        let config = base_config.clone().with_horizon(*horizon);

        // MLP Inference
        {
            let mut model = MLP::new(config.clone());
            model.fit(&df).unwrap();

            group.bench_with_input(
                BenchmarkId::new("MLP", horizon),
                horizon,
                |b, &h| {
                    b.iter(|| {
                        let pred = model.predict(h).unwrap();
                        black_box(pred);
                    });
                },
            );
        }

        // DLinear Inference
        {
            let mut model = DLinear::new(config.clone());
            model.fit(&df).unwrap();

            group.bench_with_input(
                BenchmarkId::new("DLinear", horizon),
                horizon,
                |b, &h| {
                    b.iter(|| {
                        let pred = model.predict(h).unwrap();
                        black_box(pred);
                    });
                },
            );
        }

        // NLinear Inference
        {
            let mut model = NLinear::new(config.clone());
            model.fit(&df).unwrap();

            group.bench_with_input(
                BenchmarkId::new("NLinear", horizon),
                horizon,
                |b, &h| {
                    b.iter(|| {
                        let pred = model.predict(h).unwrap();
                        black_box(pred);
                    });
                },
            );
        }

        // MLPMultivariate Inference
        {
            let mut model = MLPMultivariate::new(config.clone());
            model.fit(&df).unwrap();

            group.bench_with_input(
                BenchmarkId::new("MLPMultivariate", horizon),
                horizon,
                |b, &h| {
                    b.iter(|| {
                        let pred = model.predict(h).unwrap();
                        black_box(pred);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark: Memory Usage (Indirect - measure serialized size)
fn bench_model_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_size");

    let config = ModelConfig::default()
        .with_input_size(168)
        .with_horizon(24)
        .with_hidden_size(512);

    let data = generate_data(1000, true);
    let df = TimeSeriesDataFrame::from_values(data, None).unwrap();

    // MLP Size
    group.bench_function("MLP_serialize", |b| {
        let mut model = MLP::new(config.clone());
        model.fit(&df).unwrap();

        b.iter(|| {
            let serialized = bincode::serialize(&model).unwrap();
            black_box(serialized.len());
        });
    });

    // DLinear Size
    group.bench_function("DLinear_serialize", |b| {
        let mut model = DLinear::new(config.clone());
        model.fit(&df).unwrap();

        b.iter(|| {
            let serialized = bincode::serialize(&model).unwrap();
            black_box(serialized.len());
        });
    });

    group.finish();
}

/// Benchmark: Throughput (predictions per second)
fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");
    group.measurement_time(Duration::from_secs(15));

    let config = ModelConfig::default()
        .with_input_size(24)
        .with_horizon(12);

    let data = generate_data(1000, true);
    let df = TimeSeriesDataFrame::from_values(data, None).unwrap();

    // Pre-train models
    let mut mlp = MLP::new(config.clone());
    mlp.fit(&df).unwrap();

    let mut dlinear = DLinear::new(config.clone());
    dlinear.fit(&df).unwrap();

    // MLP Throughput
    group.bench_function("MLP_throughput", |b| {
        b.iter(|| {
            for _ in 0..100 {
                let pred = mlp.predict(12).unwrap();
                black_box(pred);
            }
        });
    });

    // DLinear Throughput
    group.bench_function("DLinear_throughput", |b| {
        b.iter(|| {
            for _ in 0..100 {
                let pred = dlinear.predict(12).unwrap();
                black_box(pred);
            }
        });
    });

    group.finish();
}

/// Benchmark: Scaling with Hidden Size (MLP only)
fn bench_hidden_size_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("hidden_size_scaling");

    let hidden_sizes = vec![64, 128, 256, 512, 1024];
    let data = generate_data(500, true);
    let df = TimeSeriesDataFrame::from_values(data, None).unwrap();

    for hidden_size in hidden_sizes.iter() {
        let config = ModelConfig::default()
            .with_input_size(24)
            .with_horizon(12)
            .with_hidden_size(*hidden_size);

        group.bench_with_input(
            BenchmarkId::new("MLP_training", hidden_size),
            hidden_size,
            |b, _| {
                b.iter(|| {
                    let mut model = MLP::new(config.clone());
                    model.fit(&df).unwrap();
                    black_box(model);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("MLP_inference", hidden_size),
            hidden_size,
            |b, _| {
                let mut model = MLP::new(config.clone());
                model.fit(&df).unwrap();

                b.iter(|| {
                    let pred = model.predict(12).unwrap();
                    black_box(pred);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Batch Processing
fn bench_batch_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_prediction");

    let batch_sizes = vec![1, 8, 32, 128, 512];
    let config = ModelConfig::default()
        .with_input_size(24)
        .with_horizon(12);

    let data = generate_data(1000, true);
    let df = TimeSeriesDataFrame::from_values(data, None).unwrap();

    let mut mlp = MLP::new(config.clone());
    mlp.fit(&df).unwrap();

    for batch_size in batch_sizes.iter() {
        group.bench_with_input(
            BenchmarkId::new("MLP_batch", batch_size),
            batch_size,
            |b, &size| {
                b.iter(|| {
                    for _ in 0..size {
                        let pred = mlp.predict(12).unwrap();
                        black_box(pred);
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_training_time,
    bench_inference_latency,
    bench_model_size,
    bench_throughput,
    bench_hidden_size_scaling,
    bench_batch_prediction,
);

criterion_main!(benches);

// ============================================================
// EXPECTED BENCHMARK RESULTS (Estimated)
// ============================================================
//
// Training Time (ms):
// +--------+--------+----------+----------+-----------------+
// | Size   | MLP    | DLinear  | NLinear  | MLPMultivariate |
// +--------+--------+----------+----------+-----------------+
// | 100    | 450    | <1       | <1       | <1              |
// | 500    | 1,200  | <1       | <1       | <1              |
// | 1,000  | 2,500  | 1        | 1        | 1               |
// | 5,000  | 15,000 | 2        | 2        | 2               |
// | 10,000 | 35,000 | 3        | 3        | 3               |
// +--------+--------+----------+----------+-----------------+
//
// Inference Latency (μs):
// +--------+------+---------+---------+-----------------+
// | Horizon| MLP  | DLinear | NLinear | MLPMultivariate |
// +--------+------+---------+---------+-----------------+
// | 1      | 850  | 2       | 2       | 3               |
// | 6      | 920  | 3       | 3       | 4               |
// | 12     | 1,050| 5       | 5       | 6               |
// | 24     | 1,280| 8       | 8       | 10              |
// | 48     | 1,650| 12      | 12      | 15              |
// | 96     | 2,300| 18      | 18      | 22              |
// +--------+------+---------+---------+-----------------+
//
// Model Size (bytes):
// - MLP (h=512):     ~1,800,000
// - DLinear:         ~500
// - NLinear:         ~500
// - MLPMultivariate: ~600
//
// Throughput (predictions/sec):
// - MLP:     ~1,000
// - DLinear: ~500,000
// - NLinear: ~500,000
//
// Scaling with Hidden Size (MLP):
// Training time grows O(d²):
// - h=64:    ~150 ms
// - h=128:   ~450 ms
// - h=256:   ~1,200 ms
// - h=512:   ~2,500 ms
// - h=1024:  ~6,000 ms
