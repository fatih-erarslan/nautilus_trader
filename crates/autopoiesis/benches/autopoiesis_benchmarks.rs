#[cfg(feature = "benchmarks")]
use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[cfg(feature = "benchmarks")]
fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 1,
        1 => 1,
        n => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

#[cfg(feature = "benchmarks")]
fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("fibonacci 20", |b| b.iter(|| fibonacci(black_box(20))));
}

#[cfg(feature = "benchmarks")]
criterion_group!(benches, criterion_benchmark);

#[cfg(feature = "benchmarks")]
criterion_main!(benches);