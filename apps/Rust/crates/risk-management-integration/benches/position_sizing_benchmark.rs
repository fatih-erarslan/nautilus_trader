use criterion::{black_box, criterion_group, criterion_main, Criterion};
use risk_management_integration::position::PositionSizer;
use risk_management_integration::risk::RiskCalculator;
use risk_management_integration::config::RiskConfig;

fn position_sizing_benchmark(c: &mut Criterion) {
    let config = RiskConfig::default();
    let sizer = PositionSizer::new(config);
    let account_equity = 100000.0;
    let risk_per_trade = 0.02;
    let stop_loss = 0.05;
    
    c.bench_function("position_sizing", |b| {
        b.iter(|| {
            sizer.calculate_position_size(
                black_box(account_equity),
                black_box(risk_per_trade),
                black_box(stop_loss)
            )
        })
    });
}

fn risk_calculation_benchmark(c: &mut Criterion) {
    let calculator = RiskCalculator::new();
    let position_size = 1000.0;
    let entry_price = 100.0;
    let stop_loss = 95.0;
    
    c.bench_function("risk_calculation", |b| {
        b.iter(|| {
            calculator.calculate_risk(
                black_box(position_size),
                black_box(entry_price),
                black_box(stop_loss)
            )
        })
    });
}

criterion_group!(benches, position_sizing_benchmark, risk_calculation_benchmark);
criterion_main!(benches);