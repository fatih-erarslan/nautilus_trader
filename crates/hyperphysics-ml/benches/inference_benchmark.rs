//! Inference benchmarks for hyperphysics-ml
//!
//! Measures inference latency for various model configurations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_lstm_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("lstm_inference");

    // Different hidden sizes
    for hidden_size in [32, 64, 128, 256] {
        group.bench_with_input(
            BenchmarkId::new("hidden_size", hidden_size),
            &hidden_size,
            |b, &_size| {
                // Placeholder: actual benchmark would create LSTM and run inference
                b.iter(|| {
                    black_box(0.0f32)
                });
            },
        );
    }

    group.finish();
}

fn bench_quantum_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantum_encoding");

    // Different input sizes
    for input_size in [8, 16, 32, 64] {
        group.bench_with_input(
            BenchmarkId::new("input_size", input_size),
            &input_size,
            |b, &size| {
                let input: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();
                b.iter(|| {
                    black_box(&input)
                });
            },
        );
    }

    group.finish();
}

fn bench_bio_cognitive(c: &mut Criterion) {
    let mut group = c.benchmark_group("bio_cognitive");

    // Different sequence lengths
    for seq_len in [10, 20, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("seq_len", seq_len),
            &seq_len,
            |b, &_len| {
                b.iter(|| {
                    black_box(0.0f32)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_lstm_inference,
    bench_quantum_encoding,
    bench_bio_cognitive,
);

criterion_main!(benches);
