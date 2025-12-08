//! ML Performance showcase benchmarks
//!
//! Demonstrates CDFA-ML integration performance

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cdfa_core::prelude::*;
use cdfa_ml::prelude::*;
use std::time::Duration;

/// Benchmark ML-enhanced CDFA pipeline
fn bench_ml_enhanced_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("ml_enhanced_pipeline");
    group.measurement_time(Duration::from_secs(20));
    
    // Generate trading-like data
    let signals: Vec<Vec<f64>> = (0..10)
        .map(|i| {
            (0..500)
                .map(|j| ((i + j) as f64 * 0.1).sin() + (i as f64 * 0.01))
                .collect()
        })
        .collect();
    
    group.bench_function("ml_enhanced_fusion", |b| {
        b.iter(|| {
            // Use ML to determine optimal fusion weights
            let features: Vec<Vec<f64>> = signals.iter()
                .map(|signal| {
                    vec![
                        signal.iter().sum::<f64>() / signal.len() as f64, // mean
                        signal.iter().map(|x| x * x).sum::<f64>() / signal.len() as f64, // variance
                        signal.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>(), // volatility
                    ]
                })
                .collect();
            
            let targets = vec![0.1, 0.15, 0.08, 0.12, 0.09, 0.11, 0.14, 0.13, 0.07, 0.16];
            
            let mut model = BasicMLP::new(3, vec![5], 1);
            model.train(&features, &targets, 10);
            
            let predicted_weights: Vec<f64> = features.iter()
                .map(|f| model.predict(f).max(0.01))
                .collect();
            
            let weight_sum: f64 = predicted_weights.iter().sum();
            let normalized_weights: Vec<f64> = predicted_weights.iter()
                .map(|w| w / weight_sum)
                .collect();
            
            let fusion = ScoreBasedFusion::new(normalized_weights);
            black_box(fusion.fuse(&signals))
        })
    });
    
    group.finish();
}

/// Benchmark adaptive learning
fn bench_adaptive_learning(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_learning");
    group.measurement_time(Duration::from_secs(15));
    
    group.bench_function("online_weight_adaptation", |b| {
        b.iter(|| {
            let mut learner = OnlineSGD::new(5, 0.01);
            
            for i in 0..1000 {
                let features = vec![
                    (i as f64 * 0.1).sin(),
                    (i as f64 * 0.2).cos(),
                    (i as f64).sqrt() % 10.0,
                    (i as f64 * 0.05).tan(),
                    i as f64 % 7.0,
                ];
                let target = if i % 3 == 0 { 1.0 } else { 0.0 };
                learner.update(&features, target);
            }
            
            black_box(learner.get_weights())
        })
    });
    
    group.finish();
}

criterion_group!(benches, bench_ml_enhanced_pipeline, bench_adaptive_learning);
criterion_main!(benches);