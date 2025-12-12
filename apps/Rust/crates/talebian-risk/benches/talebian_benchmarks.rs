//! Simplified benchmarks for the Talebian Risk Management library

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use talebian_risk::{TalebianRisk, RiskConfig, ReturnData};
use chrono::Utc;

fn benchmark_risk_system_creation(c: &mut Criterion) {
    c.bench_function("risk_system_creation", |b| {
        b.iter(|| {
            let config = RiskConfig::default();
            TalebianRisk::new(black_box(config)).unwrap()
        })
    });
}

fn benchmark_market_data_update(c: &mut Criterion) {
    let config = RiskConfig::default();
    let mut risk_system = TalebianRisk::new(config).unwrap();
    
    c.bench_function("market_data_update", |b| {
        b.iter(|| {
            let return_data = ReturnData {
                expected_return: black_box(0.01),
                volatility: black_box(0.15),
                timestamp: Utc::now(),
            };
            risk_system.update_market_data(&return_data).unwrap()
        })
    });
}

criterion_group!(benches, benchmark_risk_system_creation, benchmark_market_data_update);
criterion_main!(benches);