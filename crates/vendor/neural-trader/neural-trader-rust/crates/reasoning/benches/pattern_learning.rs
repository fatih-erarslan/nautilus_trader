use criterion::{black_box, criterion_group, criterion_main, Criterion};
use reasoning::metrics::{
    calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown,
    calculate_win_rate, calculate_profit_factor,
};

fn generate_returns(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| ((i as f64 * 0.1).sin() * 0.02))
        .collect()
}

fn benchmark_sharpe_ratio(c: &mut Criterion) {
    let returns = generate_returns(1000);

    c.bench_function("sharpe_ratio_1000", |b| {
        b.iter(|| calculate_sharpe_ratio(black_box(&returns)))
    });
}

fn benchmark_sortino_ratio(c: &mut Criterion) {
    let returns = generate_returns(1000);

    c.bench_function("sortino_ratio_1000", |b| {
        b.iter(|| calculate_sortino_ratio(black_box(&returns)))
    });
}

fn benchmark_max_drawdown(c: &mut Criterion) {
    let returns = generate_returns(1000);

    c.bench_function("max_drawdown_1000", |b| {
        b.iter(|| calculate_max_drawdown(black_box(&returns)))
    });
}

fn benchmark_win_rate(c: &mut Criterion) {
    let returns = generate_returns(1000);

    c.bench_function("win_rate_1000", |b| {
        b.iter(|| calculate_win_rate(black_box(&returns)))
    });
}

fn benchmark_profit_factor(c: &mut Criterion) {
    let returns = generate_returns(1000);

    c.bench_function("profit_factor_1000", |b| {
        b.iter(|| calculate_profit_factor(black_box(&returns)))
    });
}

criterion_group!(
    benches,
    benchmark_sharpe_ratio,
    benchmark_sortino_ratio,
    benchmark_max_drawdown,
    benchmark_win_rate,
    benchmark_profit_factor
);

criterion_main!(benches);
