//! Benchmarks comparing RNN, LSTM, and GRU performance
//!
//! Measures:
//! - Training time
//! - Inference time
//! - Memory usage
//! - Gradient computation time

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use neuro_divergent::{
    NeuralModel, ModelConfig, TimeSeriesDataFrame,
    models::recurrent::{RNN, LSTM, GRU},
};

fn benchmark_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("recurrent_training");

    let config = ModelConfig::default()
        .with_input_size(20)
        .with_horizon(5)
        .with_hidden_size(64);

    let values: Vec<f64> = (0..200).map(|i| (i as f64 * 0.05).sin()).collect();
    let data = TimeSeriesDataFrame::from_values(values, None).unwrap();

    group.bench_function("RNN", |b| {
        b.iter(|| {
            let mut model = RNN::new(config.clone());
            model.fit(black_box(&data)).unwrap();
        });
    });

    group.bench_function("LSTM", |b| {
        b.iter(|| {
            let mut model = LSTM::new(config.clone());
            model.fit(black_box(&data)).unwrap();
        });
    });

    group.bench_function("GRU", |b| {
        b.iter(|| {
            let mut model = GRU::new(config.clone());
            model.fit(black_box(&data)).unwrap();
        });
    });

    group.finish();
}

fn benchmark_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("recurrent_inference");

    let config = ModelConfig::default()
        .with_input_size(20)
        .with_horizon(5)
        .with_hidden_size(64);

    let values: Vec<f64> = (0..200).map(|i| (i as f64 * 0.05).sin()).collect();
    let data = TimeSeriesDataFrame::from_values(values, None).unwrap();

    // Pre-train models
    let mut rnn = RNN::new(config.clone());
    rnn.fit(&data).unwrap();

    let mut lstm = LSTM::new(config.clone());
    lstm.fit(&data).unwrap();

    let mut gru = GRU::new(config.clone());
    gru.fit(&data).unwrap();

    group.bench_function("RNN_predict", |b| {
        b.iter(|| {
            rnn.predict(black_box(5)).unwrap();
        });
    });

    group.bench_function("LSTM_predict", |b| {
        b.iter(|| {
            lstm.predict(black_box(5)).unwrap();
        });
    });

    group.bench_function("GRU_predict", |b| {
        b.iter(|| {
            gru.predict(black_box(5)).unwrap();
        });
    });

    group.finish();
}

fn benchmark_hidden_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("recurrent_hidden_sizes");

    let hidden_sizes = vec![32, 64, 128, 256];
    let values: Vec<f64> = (0..150).map(|i| (i as f64 * 0.05).sin()).collect();

    for &hidden_size in &hidden_sizes {
        let config = ModelConfig::default()
            .with_input_size(20)
            .with_horizon(5)
            .with_hidden_size(hidden_size);

        let data = TimeSeriesDataFrame::from_values(values.clone(), None).unwrap();

        group.bench_with_input(
            BenchmarkId::new("GRU", hidden_size),
            &(config.clone(), data.clone()),
            |b, (cfg, d)| {
                b.iter(|| {
                    let mut model = GRU::new(cfg.clone());
                    model.fit(black_box(d)).unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("LSTM", hidden_size),
            &(config.clone(), data.clone()),
            |b, (cfg, d)| {
                b.iter(|| {
                    let mut model = LSTM::new(cfg.clone());
                    model.fit(black_box(d)).unwrap();
                });
            },
        );
    }

    group.finish();
}

fn benchmark_sequence_lengths(c: &mut Criterion) {
    let mut group = c.benchmark_group("recurrent_sequence_lengths");

    let seq_lengths = vec![10, 20, 50, 100];
    let config_base = ModelConfig::default()
        .with_horizon(5)
        .with_hidden_size(64);

    for &seq_len in &seq_lengths {
        let config = config_base.clone().with_input_size(seq_len);
        let values: Vec<f64> = (0..(seq_len + 50)).map(|i| (i as f64 * 0.1).sin()).collect();
        let data = TimeSeriesDataFrame::from_values(values, None).unwrap();

        group.bench_with_input(
            BenchmarkId::new("RNN", seq_len),
            &(config.clone(), data.clone()),
            |b, (cfg, d)| {
                b.iter(|| {
                    let mut model = RNN::new(cfg.clone());
                    model.fit(black_box(d)).unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_training,
    benchmark_inference,
    benchmark_hidden_sizes,
    benchmark_sequence_lengths
);
criterion_main!(benches);
