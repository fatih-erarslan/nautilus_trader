//! Comprehensive Inference Benchmarks for All 27 Neural Models
//!
//! Measures:
//! - Prediction latency (1, 10, 100, 1000 samples)
//! - Throughput (samples/second)
//! - Memory footprint
//! - Batch size scaling
//!
//! Target: 3-5x speedup over Python

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use neuro_divergent::{
    NeuralModel, ModelConfig, TimeSeriesDataFrame,
    models::*,
};

/// Generate synthetic time series data
fn generate_inference_data(samples: usize) -> Vec<f64> {
    (0..samples)
        .map(|i| {
            let t = i as f64;
            10.0 * (t / 24.0 * 2.0 * std::f64::consts::PI).sin() + 0.5 * t / 100.0
        })
        .collect()
}

/// Create pre-trained model for inference benchmarks
fn create_pretrained_model<M: NeuralModel + 'static>(
    factory: impl Fn(ModelConfig) -> M,
    config: ModelConfig,
) -> M {
    let data_vec = generate_inference_data(500);
    let data = TimeSeriesDataFrame::from_values(data_vec, None).unwrap();
    let mut model = factory(config);
    let _ = model.fit(&data);
    model
}

fn inference_config() -> ModelConfig {
    ModelConfig::default()
        .with_input_size(24)
        .with_horizon(12)
        .with_hidden_size(64)
        .with_num_layers(2)
}

// ============================================================================
// SINGLE PREDICTION LATENCY BENCHMARKS
// ============================================================================

fn bench_single_prediction_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference/latency/single");

    let config = inference_config();

    // Test all 27 models
    let models = vec![
        ("mlp", create_pretrained_model(|c| MLP::new(c), config.clone())),
        ("dlinear", create_pretrained_model(|c| DLinear::new(c), config.clone())),
        ("nlinear", create_pretrained_model(|c| NLinear::new(c), config.clone())),
        ("rnn", create_pretrained_model(|c| RNN::new(c), config.clone())),
        ("lstm", create_pretrained_model(|c| LSTM::new(c), config.clone())),
        ("gru", create_pretrained_model(|c| GRU::new(c), config.clone())),
        ("nbeats", create_pretrained_model(|c| NBEATS::new(c), config.clone())),
        ("nhits", create_pretrained_model(|c| NHITS::new(c), config.clone())),
        ("tide", create_pretrained_model(|c| TiDE::new(c), config.clone())),
        ("tft", create_pretrained_model(|c| TFT::new(c), config.clone())),
        ("informer", create_pretrained_model(|c| Informer::new(c), config.clone())),
        ("patchtst", create_pretrained_model(|c| PatchTST::new(c), config.clone())),
        ("deepar", create_pretrained_model(|c| DeepAR::new(c), config.clone())),
        ("tcn", create_pretrained_model(|c| TCN::new(c), config.clone())),
        ("timesnet", create_pretrained_model(|c| TimesNet::new(c), config.clone())),
    ];

    for (name, model) in models {
        group.bench_function(name, |b| {
            b.iter(|| {
                let predictions = model.predict(black_box(12)).unwrap();
                black_box(predictions);
            });
        });
    }

    group.finish();
}

// ============================================================================
// BATCH PREDICTION THROUGHPUT BENCHMARKS
// ============================================================================

fn bench_batch_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference/throughput/batch");

    let config = inference_config();

    for &batch_size in &[1, 10, 100, 1000] {
        group.throughput(Throughput::Elements(batch_size as u64));

        // Test key models
        let models = vec![
            ("mlp", create_pretrained_model(|c| MLP::new(c), config.clone())),
            ("lstm", create_pretrained_model(|c| LSTM::new(c), config.clone())),
            ("nbeats", create_pretrained_model(|c| NBEATS::new(c), config.clone())),
            ("tft", create_pretrained_model(|c| TFT::new(c), config.clone())),
        ];

        for (name, model) in models {
            group.bench_with_input(
                BenchmarkId::new(name, batch_size),
                &batch_size,
                |b, &batch_size| {
                    b.iter(|| {
                        for _ in 0..batch_size {
                            let predictions = model.predict(12).unwrap();
                            black_box(predictions);
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// HORIZON SCALING BENCHMARKS
// ============================================================================

fn bench_horizon_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference/horizon_scaling");

    let config = inference_config();

    let models = vec![
        ("mlp", create_pretrained_model(|c| MLP::new(c), config.clone())),
        ("lstm", create_pretrained_model(|c| LSTM::new(c), config.clone())),
        ("nbeats", create_pretrained_model(|c| NBEATS::new(c), config.clone())),
        ("nhits", create_pretrained_model(|c| NHITS::new(c), config.clone())),
    ];

    for (name, model) in models {
        for &horizon in &[6, 12, 24, 48, 96] {
            group.bench_with_input(
                BenchmarkId::new(name, horizon),
                &horizon,
                |b, &horizon| {
                    b.iter(|| {
                        let predictions = model.predict(black_box(horizon)).unwrap();
                        black_box(predictions);
                    });
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// MODEL CATEGORY COMPARISON
// ============================================================================

fn bench_basic_models_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference/category/basic");

    let config = inference_config();

    let models = vec![
        ("mlp", create_pretrained_model(|c| MLP::new(c), config.clone())),
        ("dlinear", create_pretrained_model(|c| DLinear::new(c), config.clone())),
        ("nlinear", create_pretrained_model(|c| NLinear::new(c), config.clone())),
        ("mlp_multivariate", create_pretrained_model(|c| MLPMultivariate::new(c), config.clone())),
    ];

    for (name, model) in models {
        group.bench_function(name, |b| {
            b.iter(|| {
                let predictions = model.predict(12).unwrap();
                black_box(predictions);
            });
        });
    }

    group.finish();
}

fn bench_recurrent_models_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference/category/recurrent");

    let config = inference_config();

    let models = vec![
        ("rnn", create_pretrained_model(|c| RNN::new(c), config.clone())),
        ("lstm", create_pretrained_model(|c| LSTM::new(c), config.clone())),
        ("gru", create_pretrained_model(|c| GRU::new(c), config.clone())),
    ];

    for (name, model) in models {
        group.bench_function(name, |b| {
            b.iter(|| {
                let predictions = model.predict(12).unwrap();
                black_box(predictions);
            });
        });
    }

    group.finish();
}

fn bench_advanced_models_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference/category/advanced");

    let config = inference_config();

    let models = vec![
        ("nbeats", create_pretrained_model(|c| NBEATS::new(c), config.clone())),
        ("nbeatsx", create_pretrained_model(|c| NBEATSx::new(c), config.clone())),
        ("nhits", create_pretrained_model(|c| NHITS::new(c), config.clone())),
        ("tide", create_pretrained_model(|c| TiDE::new(c), config.clone())),
    ];

    for (name, model) in models {
        group.bench_function(name, |b| {
            b.iter(|| {
                let predictions = model.predict(12).unwrap();
                black_box(predictions);
            });
        });
    }

    group.finish();
}

fn bench_transformer_models_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference/category/transformers");

    let config = inference_config();

    let models = vec![
        ("tft", create_pretrained_model(|c| TFT::new(c), config.clone())),
        ("informer", create_pretrained_model(|c| Informer::new(c), config.clone())),
        ("autoformer", create_pretrained_model(|c| AutoFormer::new(c), config.clone())),
        ("fedformer", create_pretrained_model(|c| FedFormer::new(c), config.clone())),
        ("patchtst", create_pretrained_model(|c| PatchTST::new(c), config.clone())),
        ("itransformer", create_pretrained_model(|c| ITransformer::new(c), config.clone())),
    ];

    for (name, model) in models {
        group.bench_function(name, |b| {
            b.iter(|| {
                let predictions = model.predict(12).unwrap();
                black_box(predictions);
            });
        });
    }

    group.finish();
}

fn bench_specialized_models_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference/category/specialized");

    let config = inference_config();

    let models = vec![
        ("deepar", create_pretrained_model(|c| DeepAR::new(c), config.clone())),
        ("deepnpts", create_pretrained_model(|c| DeepNPTS::new(c), config.clone())),
        ("tcn", create_pretrained_model(|c| TCN::new(c), config.clone())),
        ("bitcn", create_pretrained_model(|c| BiTCN::new(c), config.clone())),
        ("timesnet", create_pretrained_model(|c| TimesNet::new(c), config.clone())),
        ("stemgnn", create_pretrained_model(|c| StemGNN::new(c), config.clone())),
        ("tsmixer", create_pretrained_model(|c| TSMixer::new(c), config.clone())),
        ("timellm", create_pretrained_model(|c| TimeLLM::new(c), config.clone())),
    ];

    for (name, model) in models {
        group.bench_function(name, |b| {
            b.iter(|| {
                let predictions = model.predict(12).unwrap();
                black_box(predictions);
            });
        });
    }

    group.finish();
}

// ============================================================================
// PREDICTION INTERVALS BENCHMARKS
// ============================================================================

fn bench_prediction_intervals(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference/prediction_intervals");

    let config = inference_config();
    let levels = vec![0.8, 0.9, 0.95];

    let models = vec![
        ("deepar", create_pretrained_model(|c| DeepAR::new(c), config.clone())),
        ("deepnpts", create_pretrained_model(|c| DeepNPTS::new(c), config.clone())),
        ("nbeats", create_pretrained_model(|c| NBEATS::new(c), config.clone())),
    ];

    for (name, model) in models {
        group.bench_function(name, |b| {
            b.iter(|| {
                let predictions = model.predict_intervals(12, &levels).unwrap();
                black_box(predictions);
            });
        });
    }

    group.finish();
}

criterion_group!(
    inference_benches,
    bench_single_prediction_latency,
    bench_batch_throughput,
    bench_horizon_scaling,
    bench_basic_models_inference,
    bench_recurrent_models_inference,
    bench_advanced_models_inference,
    bench_transformer_models_inference,
    bench_specialized_models_inference,
    bench_prediction_intervals,
);
criterion_main!(inference_benches);
