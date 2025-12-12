use criterion::{black_box, criterion_group, criterion_main, Criterion};
use quantum_early_warning_system::detector::ThreatDetector;
use quantum_early_warning_system::analyzer::QuantumAnalyzer;
use quantum_early_warning_system::config::QuantumEarlyWarningConfig;

fn threat_detection_benchmark(c: &mut Criterion) {
    let detector = ThreatDetector::new();
    let market_data = vec![
        (100.0, 1000.0, 0.01),
        (101.0, 1500.0, 0.02),
        (99.0, 2000.0, 0.03),
        (102.0, 1800.0, 0.02),
    ];
    
    c.bench_function("threat_detection", |b| {
        b.iter(|| {
            detector.detect_threats(black_box(&market_data))
        })
    });
}

fn quantum_analysis_benchmark(c: &mut Criterion) {
    let config = QuantumEarlyWarningConfig::default();
    let analyzer = QuantumAnalyzer::new(config);
    let quantum_state = vec![0.5, 0.3, 0.2, 0.8, 0.1];
    
    c.bench_function("quantum_analysis", |b| {
        b.iter(|| {
            analyzer.analyze_quantum_state(black_box(&quantum_state))
        })
    });
}

criterion_group!(benches, threat_detection_benchmark, quantum_analysis_benchmark);
criterion_main!(benches);