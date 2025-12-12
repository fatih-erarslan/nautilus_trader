use criterion::{black_box, criterion_group, criterion_main, Criterion};
use quantum_pair_analyzer::correlation::CorrelationAnalyzer;
use quantum_pair_analyzer::analyzer::QuantumPairAnalyzer;
use quantum_pair_analyzer::config::QuantumPairConfig;

fn correlation_benchmark(c: &mut Criterion) {
    let config = QuantumPairConfig::default();
    let analyzer = CorrelationAnalyzer::new(config);
    let asset1_prices = vec![100.0, 101.0, 102.0, 101.5, 103.0, 102.5];
    let asset2_prices = vec![50.0, 50.5, 51.0, 50.8, 51.5, 51.2];
    
    c.bench_function("correlation_analysis", |b| {
        b.iter(|| {
            analyzer.analyze_correlation(
                black_box(&asset1_prices),
                black_box(&asset2_prices)
            )
        })
    });
}

fn quantum_pair_analysis_benchmark(c: &mut Criterion) {
    let config = QuantumPairConfig::default();
    let analyzer = QuantumPairAnalyzer::new(config);
    let pair_data = vec![
        (100.0, 50.0),
        (101.0, 50.5),
        (102.0, 51.0),
        (101.5, 50.8),
        (103.0, 51.5),
    ];
    
    c.bench_function("quantum_pair_analysis", |b| {
        b.iter(|| {
            analyzer.analyze_quantum_pair(black_box(&pair_data))
        })
    });
}

criterion_group!(benches, correlation_benchmark, quantum_pair_analysis_benchmark);
criterion_main!(benches);