//! Benchmark for Market Microstructure Simulator

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hyperphysics_market_microstructure::{MarketMicrostructureSimulator, Order, Side, MarketEvent};

fn bench_simulator_step(c: &mut Criterion) {
    let mut sim = MarketMicrostructureSimulator::hyperphysics_default(100.0);

    c.bench_function("simulator_step_no_events", |b| {
        b.iter(|| {
            black_box(sim.step(&[]))
        })
    });
}

fn bench_order_processing(c: &mut Criterion) {
    let mut sim = MarketMicrostructureSimulator::hyperphysics_default(100.0);
    let order = Order {
        side: Side::Bid,
        price: 100.0,
        quantity: 100.0,
        timestamp: 0,
    };
    let events = vec![MarketEvent::LimitOrder(order)];

    c.bench_function("order_processing", |b| {
        b.iter(|| {
            black_box(sim.step(&events))
        })
    });
}

fn bench_imbalance_computation(c: &mut Criterion) {
    let sim = MarketMicrostructureSimulator::hyperphysics_default(100.0);

    c.bench_function("imbalance_computation", |b| {
        b.iter(|| {
            black_box(sim.compute_imbalance())
        })
    });
}

criterion_group!(benches, bench_simulator_step, bench_order_processing, bench_imbalance_computation);
criterion_main!(benches);
