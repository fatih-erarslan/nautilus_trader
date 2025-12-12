use criterion::{black_box, criterion_group, criterion_main, Criterion};
use quantum_early_warning_system::warning::QuantumWarningSystem;
use quantum_early_warning_system::predictor::ThreatPredictor;
use quantum_early_warning_system::config::QuantumEarlyWarningConfig;

fn quantum_warning_benchmark(c: &mut Criterion) {
    let config = QuantumEarlyWarningConfig::default();
    let warning_system = QuantumWarningSystem::new(config);
    let market_indicators = vec![0.3, 0.7, 0.2, 0.9, 0.1];
    
    c.bench_function("quantum_warning", |b| {
        b.iter(|| {
            warning_system.assess_quantum_threat(black_box(&market_indicators))
        })
    });
}

fn threat_prediction_benchmark(c: &mut Criterion) {
    let predictor = ThreatPredictor::new();
    let historical_patterns = vec![
        (0.1, 0.2),
        (0.3, 0.4),
        (0.5, 0.6),
        (0.7, 0.8),
    ];
    
    c.bench_function("threat_prediction", |b| {
        b.iter(|| {
            predictor.predict_future_threats(black_box(&historical_patterns))
        })
    });
}

criterion_group!(benches, quantum_warning_benchmark, threat_prediction_benchmark);
criterion_main!(benches);