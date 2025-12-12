use criterion::{black_box, criterion_group, criterion_main, Criterion};
use risk_management_integration::var::VarCalculator;
use risk_management_integration::portfolio::PortfolioRisk;
use risk_management_integration::config::RiskConfig;

fn var_calculation_benchmark(c: &mut Criterion) {
    let config = RiskConfig::default();
    let calculator = VarCalculator::new(config);
    let returns = vec![0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, -0.01];
    
    c.bench_function("var_calculation", |b| {
        b.iter(|| {
            calculator.calculate_var(black_box(&returns), black_box(0.05))
        })
    });
}

fn portfolio_risk_benchmark(c: &mut Criterion) {
    let portfolio = PortfolioRisk::new();
    let positions = vec![
        (100.0, 0.02),
        (200.0, 0.03),
        (150.0, 0.01),
        (300.0, 0.04),
    ];
    
    c.bench_function("portfolio_risk", |b| {
        b.iter(|| {
            portfolio.calculate_portfolio_risk(black_box(&positions))
        })
    });
}

criterion_group!(benches, var_calculation_benchmark, portfolio_risk_benchmark);
criterion_main!(benches);