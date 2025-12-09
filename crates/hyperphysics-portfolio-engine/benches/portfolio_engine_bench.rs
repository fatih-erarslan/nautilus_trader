//! Benchmark for Thermodynamic Portfolio Engine

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hyperphysics_portfolio_engine::{
    ThermodynamicPortfolioEngine, EngineFactory, BacktestRunner,
    SGNNEvent, EventType,
};

fn bench_engine_creation(c: &mut Criterion) {
    c.bench_function("engine_creation", |b| {
        b.iter(|| {
            black_box(ThermodynamicPortfolioEngine::hyperphysics_default())
        })
    });
}

fn bench_event_processing(c: &mut Criterion) {
    let mut engine = ThermodynamicPortfolioEngine::hyperphysics_default();
    let event = SGNNEvent {
        timestamp: 1000,
        asset_id: 0,
        event_type: EventType::Trade,
        price: 100.0,
        volume: 100.0,
    };

    c.bench_function("event_processing", |b| {
        b.iter(|| {
            black_box(engine.process_event(event.clone()))
        })
    });
}

fn bench_diagnostics(c: &mut Criterion) {
    let engine = ThermodynamicPortfolioEngine::hyperphysics_default();

    c.bench_function("diagnostics", |b| {
        b.iter(|| {
            black_box(engine.diagnostics())
        })
    });
}

fn bench_backtest(c: &mut Criterion) {
    c.bench_function("backtest_100_events", |b| {
        b.iter(|| {
            let engine = EngineFactory::hft_engine(5);
            let mut runner = BacktestRunner::new(engine);
            runner.run_synthetic(100);
            black_box(runner.statistics())
        })
    });
}

criterion_group!(benches, bench_engine_creation, bench_event_processing, bench_diagnostics, bench_backtest);
criterion_main!(benches);
