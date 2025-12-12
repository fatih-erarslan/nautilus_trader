use criterion::{black_box, criterion_group, criterion_main, Criterion};
use quantum_pair_analyzer::optimization::QuantumOptimizer;
use quantum_pair_analyzer::portfolio::QuantumPortfolio;
use quantum_pair_analyzer::config::QuantumPairConfig;

fn quantum_optimization_benchmark(c: &mut Criterion) {
    let config = QuantumPairConfig::default();
    let optimizer = QuantumOptimizer::new(config);
    let parameters = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    
    c.bench_function("quantum_optimization", |b| {
        b.iter(|| {
            optimizer.optimize_quantum_parameters(black_box(&parameters))
        })
    });
}

fn quantum_portfolio_benchmark(c: &mut Criterion) {
    let config = QuantumPairConfig::default();
    let portfolio = QuantumPortfolio::new(config);
    let assets = vec![
        ("AAPL", 0.3),
        ("GOOGL", 0.2),
        ("MSFT", 0.25),
        ("AMZN", 0.15),
        ("TSLA", 0.1),
    ];
    
    c.bench_function("quantum_portfolio", |b| {
        b.iter(|| {
            portfolio.optimize_quantum_allocation(black_box(&assets))
        })
    });
}

criterion_group!(benches, quantum_optimization_benchmark, quantum_portfolio_benchmark);
criterion_main!(benches);