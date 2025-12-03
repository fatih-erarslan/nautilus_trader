//! Performance benchmarks for prospect theory implementation

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use prospect_theory_rs::*;

fn benchmark_value_function(c: &mut Criterion) {
    let vf = ValueFunction::default_kt();
    
    // Single value calculation
    c.bench_function("value_function_single", |b| {
        b.iter(|| vf.value(black_box(100.0)).unwrap())
    });
    
    // Vectorized calculation with different sizes
    let sizes = [100, 1_000, 10_000, 100_000];
    for size in sizes {
        let outcomes: Vec<f64> = (0..size).map(|i| (i as f64) - (size as f64) / 2.0).collect();
        
        c.bench_with_input(
            BenchmarkId::new("value_function_vectorized", size),
            &outcomes,
            |b, outcomes| {
                b.iter(|| vf.values(black_box(outcomes)).unwrap())
            },
        );
        
        c.bench_with_input(
            BenchmarkId::new("value_function_parallel", size),
            &outcomes,
            |b, outcomes| {
                b.iter(|| vf.values_parallel(black_box(outcomes)).unwrap())
            },
        );
    }
}

fn benchmark_probability_weighting(c: &mut Criterion) {
    let pw = ProbabilityWeighting::default_tk();
    
    // Single weight calculation
    c.bench_function("probability_weight_single", |b| {
        b.iter(|| pw.weight_gains(black_box(0.3)).unwrap())
    });
    
    // Vectorized calculation with different sizes
    let sizes = [100, 1_000, 10_000, 100_000];
    for size in sizes {
        let probs: Vec<f64> = (1..=size).map(|i| (i as f64) / (size as f64 + 1.0)).collect();
        
        c.bench_with_input(
            BenchmarkId::new("probability_weight_vectorized", size),
            &probs,
            |b, probs| {
                b.iter(|| pw.weights_gains(black_box(probs)).unwrap())
            },
        );
        
        c.bench_with_input(
            BenchmarkId::new("probability_weight_parallel", size),
            &probs,
            |b, probs| {
                b.iter(|| pw.weights_gains_parallel(black_box(probs)).unwrap())
            },
        );
    }
}

fn benchmark_decision_weights(c: &mut Criterion) {
    let pw = ProbabilityWeighting::default_tk();

    let sizes = [10, 50, 100, 500];
    for size in sizes {
        // Create probabilities that sum to slightly less than 1.0 to avoid
        // cumulative floating-point errors exceeding 1.0 inside decision_weights
        let total = 0.9999; // Slightly under 1.0 to leave room for FP errors
        let prob = total / size as f64;
        let probs: Vec<f64> = vec![prob; size];
        let outcomes: Vec<f64> = (0..size).map(|i| (i as f64) - (size as f64) / 2.0).collect();

        c.bench_with_input(
            BenchmarkId::new("decision_weights", size),
            &(probs, outcomes),
            |b, (probs, outcomes)| {
                b.iter(|| pw.decision_weights(black_box(probs), black_box(outcomes)).unwrap())
            },
        );
    }
}

fn benchmark_marginal_value(c: &mut Criterion) {
    let vf = ValueFunction::default_kt();
    
    c.bench_function("marginal_value_single", |b| {
        b.iter(|| vf.marginal_value(black_box(100.0)).unwrap())
    });
    
    // Batch marginal value calculation
    let outcomes: Vec<f64> = (1..1000).map(|i| i as f64).collect();
    c.bench_function("marginal_value_batch", |b| {
        b.iter(|| {
            outcomes
                .iter()
                .map(|&outcome| vf.marginal_value(black_box(outcome)).unwrap())
                .collect::<Vec<_>>()
        })
    });
}

fn benchmark_risk_premium(c: &mut Criterion) {
    let vf = ValueFunction::default_kt();
    
    let outcomes = vec![100.0, 50.0, 0.0, -25.0, -50.0];
    let probabilities = vec![0.2, 0.2, 0.2, 0.2, 0.2];
    
    c.bench_function("risk_premium", |b| {
        b.iter(|| {
            vf.risk_premium(black_box(&outcomes), black_box(&probabilities))
                .unwrap()
        })
    });
}

fn benchmark_inverse_value(c: &mut Criterion) {
    let vf = ValueFunction::default_kt();
    
    c.bench_function("inverse_value_single", |b| {
        b.iter(|| vf.inverse_value(black_box(50.0)).unwrap())
    });
    
    // Batch inverse value calculation
    let values: Vec<f64> = (-100..100).map(|i| i as f64).collect();
    c.bench_function("inverse_value_batch", |b| {
        b.iter(|| {
            values
                .iter()
                .map(|&value| vf.inverse_value(black_box(value)).unwrap())
                .collect::<Vec<_>>()
        })
    });
}

fn benchmark_complete_prospect_calculation(c: &mut Criterion) {
    let vf = ValueFunction::default_kt();
    let pw = ProbabilityWeighting::default_tk();

    let sizes = [5, 10, 20, 50];
    for size in sizes {
        let outcomes: Vec<f64> = (0..size).map(|i| (i as f64) - (size as f64) / 2.0).collect();
        // Use 0.9999 total to avoid cumulative FP errors exceeding 1.0 inside decision_weights
        let probabilities: Vec<f64> = vec![0.9999 / size as f64; size];
        
        c.bench_with_input(
            BenchmarkId::new("complete_prospect_calculation", size),
            &(outcomes, probabilities),
            |b, (outcomes, probabilities)| {
                b.iter(|| {
                    let values = vf.values(black_box(outcomes)).unwrap();
                    let decision_weights = pw
                        .decision_weights(black_box(probabilities), black_box(outcomes))
                        .unwrap();
                    
                    let prospect_value: f64 = values
                        .iter()
                        .zip(decision_weights.iter())
                        .map(|(&value, &weight)| value * weight)
                        .sum();
                    
                    black_box(prospect_value)
                })
            },
        );
    }
}

fn benchmark_weighting_functions(c: &mut Criterion) {
    let params = WeightingParams::default();
    let tk_weighting = ProbabilityWeighting::new(params.clone(), WeightingFunction::TverskyKahneman).unwrap();
    let prelec_weighting = ProbabilityWeighting::prelec(params.clone()).unwrap();
    let linear_weighting = ProbabilityWeighting::linear();
    
    let prob = 0.3;
    
    c.bench_function("tversky_kahneman_weighting", |b| {
        b.iter(|| tk_weighting.weight_gains(black_box(prob)).unwrap())
    });
    
    c.bench_function("prelec_weighting", |b| {
        b.iter(|| prelec_weighting.weight_gains(black_box(prob)).unwrap())
    });
    
    c.bench_function("linear_weighting", |b| {
        b.iter(|| linear_weighting.weight_gains(black_box(prob)).unwrap())
    });
}

fn benchmark_parameter_variations(c: &mut Criterion) {
    let alphas = [0.5, 0.7, 0.88, 0.95];
    let lambdas = [1.5, 2.0, 2.25, 3.0];
    
    for &alpha in &alphas {
        for &lambda in &lambdas {
            let params = ValueFunctionParams::new(alpha, 0.88, lambda, 0.0).unwrap();
            let vf = ValueFunction::new(params).unwrap();
            
            c.bench_function(&format!("value_function_alpha_{}_lambda_{}", alpha, lambda), |b| {
                b.iter(|| vf.value(black_box(100.0)).unwrap())
            });
        }
    }
}

fn benchmark_memory_intensive(c: &mut Criterion) {
    let vf = ValueFunction::default_kt();
    let pw = ProbabilityWeighting::default_tk();
    
    // Large dataset benchmark
    let large_outcomes: Vec<f64> = (0..1_000_000).map(|i| (i as f64) - 500_000.0).collect();
    let large_probs: Vec<f64> = (1..1_000_000).map(|i| (i as f64) / 1_000_000.0).collect();
    
    c.bench_function("value_function_1M_parallel", |b| {
        b.iter(|| vf.values_parallel(black_box(&large_outcomes)).unwrap())
    });
    
    c.bench_function("probability_weight_1M_parallel", |b| {
        b.iter(|| pw.weights_gains_parallel(black_box(&large_probs)).unwrap())
    });
}

fn benchmark_financial_precision(c: &mut Criterion) {
    let vf = ValueFunction::default_kt();
    
    // Test precision near zero
    let small_values: Vec<f64> = (1..1000)
        .map(|i| FINANCIAL_PRECISION * (i as f64))
        .collect();
    
    c.bench_function("financial_precision_near_zero", |b| {
        b.iter(|| vf.values(black_box(&small_values)).unwrap())
    });
    
    // Test precision at extremes
    let extreme_values = vec![
        MAX_INPUT_VALUE - 1.0,
        MIN_INPUT_VALUE + 1.0,
        -FINANCIAL_PRECISION,
        FINANCIAL_PRECISION,
    ];
    
    c.bench_function("financial_precision_extremes", |b| {
        b.iter(|| vf.values(black_box(&extreme_values)).unwrap())
    });
}

criterion_group!(
    benches,
    benchmark_value_function,
    benchmark_probability_weighting,
    benchmark_decision_weights,
    benchmark_marginal_value,
    benchmark_risk_premium,
    benchmark_inverse_value,
    benchmark_complete_prospect_calculation,
    benchmark_weighting_functions,
    benchmark_parameter_variations,
    benchmark_memory_intensive,
    benchmark_financial_precision
);
criterion_main!(benches);