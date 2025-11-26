//! Performance benchmarks for the integration layer.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use neural_trader_integration::Config;

fn bench_config_creation(c: &mut Criterion) {
    c.bench_function("config_creation", |b| {
        b.iter(|| {
            black_box(Config::default())
        })
    });
}

fn bench_config_validation(c: &mut Criterion) {
    let _config = Config::default();

    c.bench_function("config_validation", |b| {
        b.iter(|| {
            black_box(config.validate())
        })
    });
}

criterion_group!(benches, bench_config_creation, bench_config_validation);
criterion_main!(benches);
