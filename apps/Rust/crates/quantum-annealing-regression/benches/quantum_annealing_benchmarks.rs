//! Benchmarks for Quantum Annealing Regression

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use quantum_annealing_regression::*;

fn benchmark_linear_regression(c: &mut Criterion) {
    let config = QuantumAnnealingConfig {
        num_steps: 100,
        num_chains: 2,
        seed: Some(42),
        ..Default::default()
    };
    
    // Create test data
    let features = vec![
        vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]
    ];
    let targets = vec![3.0, 5.0, 7.0, 9.0, 11.0];
    let problem = RegressionProblem::new(features, targets).unwrap();
    
    c.bench_function("linear_regression_fit", |b| {
        b.iter(|| {
            let mut model = QuantumLinearRegression::new(config.clone());
            model.fit(&problem).unwrap()
        })
    });
}

fn benchmark_annealer_optimization(c: &mut Criterion) {
    let config = QuantumAnnealingConfig {
        num_steps: 50,
        num_chains: 1,
        seed: Some(42),
        ..Default::default()
    };
    
    c.bench_function("annealer_simple_optimization", |b| {
        b.iter(|| {
            let mut annealer = QuantumAnnealer::new(config.clone()).unwrap();
            let cost_function = |params: &[f64]| (params[0] - 2.0).powi(2);
            annealer.optimize(cost_function, vec![0.0]).unwrap()
        })
    });
}

criterion_group!(benches, benchmark_linear_regression, benchmark_annealer_optimization);
criterion_main!(benches);