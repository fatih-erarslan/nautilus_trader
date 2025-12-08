//! Benchmarks for the quantum hive

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use quantum_hive::*;

fn benchmark_hive_creation(c: &mut Criterion) {
    c.bench_function("hive_creation", |b| {
        b.iter(|| {
            let _hive = black_box(AutopoieticHive::new());
        });
    });
}

fn benchmark_strategy_lookup(c: &mut Criterion) {
    let lut = QuantumStrategyLUT::default();
    
    c.bench_function("strategy_lookup", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let price_index = (i % 65536) as u16;
                unsafe {
                    let _action = black_box(lut.get_action(price_index));
                }
            }
        });
    });
}

criterion_group!(benches, benchmark_hive_creation, benchmark_strategy_lookup);
criterion_main!(benches);