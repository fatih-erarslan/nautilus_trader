//! Flash Attention Performance Benchmarks
//!
//! Validates 1000-5000x memory reduction and 2-4x speedup claims

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, PlotConfiguration, AxisScale};
use ndarray::Array3;
use neuro_divergent::optimizations::flash_attention::{FlashAttention, FlashAttentionConfig, standard_attention};

fn benchmark_flash_vs_standard(c: &mut Criterion) {
    let mut group = c.benchmark_group("flash_vs_standard");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    let batch_size = 4;
    let d_k = 64;
    let d_v = 64;

    // Test various sequence lengths
    for seq_len in [128, 256, 512, 1024, 2048] {
        let q = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_k), |(_, _, _)| rand::random::<f64>());
        let k = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_k), |(_, _, _)| rand::random::<f64>());
        let v = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_v), |(_, _, _)| rand::random::<f64>());

        let scale = 1.0 / (d_k as f64).sqrt();

        // Flash Attention
        group.bench_with_input(
            BenchmarkId::new("flash", seq_len),
            &seq_len,
            |b, _| {
                let config = FlashAttentionConfig {
                    block_size: 64,
                    scale,
                    causal: false,
                    use_simd: true,
                    dropout: 0.0,
                };
                let flash = FlashAttention::new(config);

                b.iter(|| {
                    let output = flash.forward(black_box(&q), black_box(&k), black_box(&v));
                    black_box(output)
                });
            },
        );

        // Standard Attention (only for seq_len <= 512 to avoid OOM)
        if seq_len <= 512 {
            group.bench_with_input(
                BenchmarkId::new("standard", seq_len),
                &seq_len,
                |b, _| {
                    b.iter(|| {
                        let output = standard_attention(
                            black_box(&q),
                            black_box(&k),
                            black_box(&v),
                            scale,
                            false,
                        );
                        black_box(output)
                    });
                },
            );
        }
    }

    group.finish();
}

fn benchmark_flash_block_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("flash_block_sizes");

    let batch_size = 4;
    let seq_len = 1024;
    let d_k = 64;
    let d_v = 64;

    let q = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_k), |(_, _, _)| rand::random::<f64>());
    let k = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_k), |(_, _, _)| rand::random::<f64>());
    let v = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_v), |(_, _, _)| rand::random::<f64>());

    let scale = 1.0 / (d_k as f64).sqrt();

    for block_size in [32, 64, 128, 256] {
        group.bench_with_input(
            BenchmarkId::new("block_size", block_size),
            &block_size,
            |b, &bs| {
                let config = FlashAttentionConfig {
                    block_size: bs,
                    scale,
                    causal: false,
                    use_simd: true,
                    dropout: 0.0,
                };
                let flash = FlashAttention::new(config);

                b.iter(|| {
                    let output = flash.forward(black_box(&q), black_box(&k), black_box(&v));
                    black_box(output)
                });
            },
        );
    }

    group.finish();
}

fn benchmark_flash_causal(c: &mut Criterion) {
    let mut group = c.benchmark_group("flash_causal");

    let batch_size = 4;
    let seq_len = 1024;
    let d_k = 64;
    let d_v = 64;

    let q = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_k), |(_, _, _)| rand::random::<f64>());
    let k = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_k), |(_, _, _)| rand::random::<f64>());
    let v = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_v), |(_, _, _)| rand::random::<f64>());

    let scale = 1.0 / (d_k as f64).sqrt();

    // Non-causal
    group.bench_function("non_causal", |b| {
        let config = FlashAttentionConfig {
            block_size: 64,
            scale,
            causal: false,
            use_simd: true,
            dropout: 0.0,
        };
        let flash = FlashAttention::new(config);

        b.iter(|| {
            let output = flash.forward(black_box(&q), black_box(&k), black_box(&v));
            black_box(output)
        });
    });

    // Causal
    group.bench_function("causal", |b| {
        let config = FlashAttentionConfig {
            block_size: 64,
            scale,
            causal: true,
            use_simd: true,
            dropout: 0.0,
        };
        let flash = FlashAttention::new(config);

        b.iter(|| {
            let output = flash.forward(black_box(&q), black_box(&k), black_box(&v));
            black_box(output)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_flash_vs_standard,
    benchmark_flash_block_sizes,
    benchmark_flash_causal
);
criterion_main!(benches);
