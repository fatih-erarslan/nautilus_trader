use criterion::{black_box, criterion_group, criterion_main, Criterion};
use risk_management_integration::stress::StressTester;
use risk_management_integration::scenario::ScenarioGenerator;
use risk_management_integration::config::RiskConfig;

fn stress_testing_benchmark(c: &mut Criterion) {
    let config = RiskConfig::default();
    let tester = StressTester::new(config);
    let portfolio = vec![
        (1000.0, 0.02),
        (2000.0, 0.03),
        (1500.0, 0.01),
    ];
    
    c.bench_function("stress_testing", |b| {
        b.iter(|| {
            tester.run_stress_test(black_box(&portfolio))
        })
    });
}

fn scenario_generation_benchmark(c: &mut Criterion) {
    let generator = ScenarioGenerator::new();
    let market_conditions = vec![0.5, 0.3, 0.2, 0.8];
    
    c.bench_function("scenario_generation", |b| {
        b.iter(|| {
            generator.generate_scenarios(black_box(&market_conditions), black_box(100))
        })
    });
}

criterion_group!(benches, stress_testing_benchmark, scenario_generation_benchmark);
criterion_main!(benches);