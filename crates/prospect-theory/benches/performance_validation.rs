//! Performance validation benchmarks demonstrating speedup achievements
//! 
//! This benchmark validates the performance improvements achievable
//! with the optimizations implemented in the crates.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use prospect_theory_rs::*;
use prospect_theory_rs::performance_simple::*;
use prospect_theory_rs::probability_weighting::WeightingFunction;

use std::time::Instant;

/// Benchmark single-threaded vs multi-threaded processing
fn benchmark_threading_speedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("threading_speedup");
    
    let value_params = ValueFunctionParams::default();
    let weighting_params = WeightingParams::default();
    
    // Create processors with different thread counts
    let single_thread = SimpleBatchProcessor::new(value_params.clone(), weighting_params.clone(), Some(1)).unwrap();
    let multi_thread = SimpleBatchProcessor::new(value_params, weighting_params, Some(4)).unwrap();
    
    // Large batch for demonstrating parallelism benefits
    let large_batch_size = 1000;
    let outcomes_batch: Vec<Vec<f64>> = (0..large_batch_size)
        .map(|i| vec![100.0 + i as f64, 0.0, -100.0 - i as f64, 50.0, -50.0])
        .collect();
    let probabilities_batch: Vec<Vec<f64>> = (0..large_batch_size)
        .map(|_| vec![0.2, 0.2, 0.2, 0.2, 0.2])
        .collect();
    
    group.throughput(Throughput::Elements(large_batch_size as u64));
    
    group.bench_function("single_thread_1000", |b| {
        b.iter(|| {
            single_thread.process_batch(
                black_box(&outcomes_batch),
                black_box(&probabilities_batch)
            ).unwrap()
        })
    });
    
    group.bench_function("multi_thread_1000", |b| {
        b.iter(|| {
            multi_thread.process_batch(
                black_box(&outcomes_batch),
                black_box(&probabilities_batch)
            ).unwrap()
        })
    });
    
    group.finish();
}

/// Benchmark caching vs non-caching performance
fn benchmark_caching_speedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("caching_speedup");
    
    let cache = SimpleCache::new(1000);
    
    // Common power calculations in financial computations
    let bases: Vec<f64> = vec![1.1, 1.2, 1.5, 2.0, 2.5, 3.0, 0.8, 0.9, 0.7, 0.6];
    let exponents: Vec<f64> = vec![0.88, 0.89, 0.87, 0.86, 0.85];
    
    group.bench_function("without_cache", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                for &base in &bases {
                    for &exp in &exponents {
                        black_box(base.powf(exp));
                    }
                }
            }
        })
    });
    
    group.bench_function("with_cache", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                for &base in &bases {
                    for &exp in &exponents {
                        black_box(cache.cached_pow(base, exp));
                    }
                }
            }
        })
    });
    
    group.finish();
}

/// Benchmark realistic trading scenarios showing speedup
fn benchmark_trading_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("trading_scenarios");
    
    let value_params = ValueFunctionParams::default();
    let weighting_params = WeightingParams::default();
    
    let processor = SimpleBatchProcessor::new(value_params.clone(), weighting_params.clone(), Some(4)).unwrap();
    let standard_vf = ValueFunction::new(value_params).unwrap();
    let standard_pw = ProbabilityWeighting::new(weighting_params.clone(), WeightingFunction::TverskyKahneman).unwrap();
    
    // High-frequency trading scenario: 1000 rapid calculations
    let hft_outcomes = vec![100.0, -100.0]; // Binary outcome
    let hft_probabilities = vec![0.6, 0.4];
    
    group.bench_function("hft_standard_1000", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                let values = standard_vf.values(black_box(&hft_outcomes)).unwrap();
                let decision_weights = standard_pw
                    .decision_weights(black_box(&hft_probabilities), black_box(&hft_outcomes))
                    .unwrap();
                
                let prospect_value: f64 = values
                    .iter()
                    .zip(decision_weights.iter())
                    .map(|(&value, &weight)| value * weight)
                    .sum();
                
                black_box(prospect_value);
            }
        })
    });
    
    // Batch processing the same 1000 calculations
    let hft_batch_outcomes: Vec<Vec<f64>> = (0..1000).map(|_| hft_outcomes.clone()).collect();
    let hft_batch_probabilities: Vec<Vec<f64>> = (0..1000).map(|_| hft_probabilities.clone()).collect();
    
    group.bench_function("hft_batch_1000", |b| {
        b.iter(|| {
            processor.process_batch(
                black_box(&hft_batch_outcomes),
                black_box(&hft_batch_probabilities)
            ).unwrap()
        })
    });
    
    // Market making scenario: varying outcomes
    let market_outcomes: Vec<Vec<f64>> = (0..500)
        .map(|i| vec![100.0 + i as f64, 50.0, 0.0, -50.0, -100.0 - i as f64])
        .collect();
    let market_probabilities: Vec<Vec<f64>> = (0..500)
        .map(|_| vec![0.25, 0.2, 0.1, 0.2, 0.25])
        .collect();
    
    group.throughput(Throughput::Elements(500));
    
    group.bench_function("market_making_standard", |b| {
        b.iter(|| {
            let mut results = Vec::with_capacity(500);
            for (outcomes, probabilities) in market_outcomes.iter().zip(market_probabilities.iter()) {
                let values = standard_vf.values(black_box(outcomes)).unwrap();
                let decision_weights = standard_pw
                    .decision_weights(black_box(probabilities), black_box(outcomes))
                    .unwrap();
                
                let prospect_value: f64 = values
                    .iter()
                    .zip(decision_weights.iter())
                    .map(|(&value, &weight)| value * weight)
                    .sum();
                
                results.push(prospect_value);
            }
            black_box(results)
        })
    });
    
    group.bench_function("market_making_batch", |b| {
        b.iter(|| {
            processor.process_batch(
                black_box(&market_outcomes),
                black_box(&market_probabilities)
            ).unwrap()
        })
    });
    
    group.finish();
}

/// Benchmark memory efficiency improvements
fn benchmark_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    
    let value_params = ValueFunctionParams::default();
    let standard_vf = ValueFunction::new(value_params).unwrap();
    
    // Test with different data sizes to show memory scaling
    for size in [1_000, 10_000, 50_000].iter() {
        let outcomes: Vec<f64> = (0..*size).map(|i| (i as f64) - (*size as f64) / 2.0).collect();
        
        group.throughput(Throughput::Elements(*size as u64));
        
        // Standard calculation
        group.bench_with_input(
            BenchmarkId::new("standard_calculation", size),
            &outcomes,
            |b, outcomes| {
                b.iter(|| standard_vf.values(black_box(outcomes)).unwrap())
            },
        );
        
        // Parallel calculation (should use same memory more efficiently)
        group.bench_with_input(
            BenchmarkId::new("parallel_calculation", size),
            &outcomes,
            |b, outcomes| {
                b.iter(|| standard_vf.values_parallel(black_box(outcomes)).unwrap())
            },
        );
    }
    
    group.finish();
}

/// Demonstrate speedup ratios achieved
fn benchmark_speedup_demonstration(c: &mut Criterion) {
    let mut group = c.benchmark_group("speedup_demonstration");
    
    let value_params = ValueFunctionParams::default();
    let weighting_params = WeightingParams::default();
    
    let standard_vf = ValueFunction::new(value_params.clone()).unwrap();
    let standard_pw = ProbabilityWeighting::new(weighting_params.clone(), WeightingFunction::TverskyKahneman).unwrap();
    let batch_processor = SimpleBatchProcessor::new(value_params, weighting_params, Some(8)).unwrap();
    
    // Large dataset for maximum speedup demonstration
    let large_size = 10_000;
    let outcomes_batch: Vec<Vec<f64>> = (0..large_size)
        .map(|i| vec![(i as f64) * 0.1, (i as f64) * -0.1, 0.0])
        .collect();
    let probabilities_batch: Vec<Vec<f64>> = (0..large_size)
        .map(|_| vec![0.4, 0.3, 0.3])
        .collect();
    
    group.sample_size(10); // Fewer samples for large datasets
    group.throughput(Throughput::Elements(large_size as u64));
    
    // Baseline: naive sequential processing
    group.bench_function("baseline_sequential_10k", |b| {
        b.iter(|| {
            let start = Instant::now();
            let mut results = Vec::with_capacity(large_size);
            
            for (outcomes, probabilities) in outcomes_batch.iter().zip(probabilities_batch.iter()) {
                let values = standard_vf.values(black_box(outcomes)).unwrap();
                let decision_weights = standard_pw
                    .decision_weights(black_box(probabilities), black_box(outcomes))
                    .unwrap();
                
                let prospect_value: f64 = values
                    .iter()
                    .zip(decision_weights.iter())
                    .map(|(&value, &weight)| value * weight)
                    .sum();
                
                results.push(prospect_value);
            }
            
            let duration = start.elapsed();
            println!("Sequential execution time: {:?}", duration);
            black_box(results)
        })
    });
    
    // Optimized: batch processing with threading
    group.bench_function("optimized_batch_10k", |b| {
        b.iter(|| {
            let start = Instant::now();
            let results = batch_processor.process_batch(
                black_box(&outcomes_batch),
                black_box(&probabilities_batch)
            ).unwrap();
            let duration = start.elapsed();
            println!("Batch execution time: {:?}", duration);
            black_box(results)
        })
    });
    
    group.finish();
}

/// Performance measurement utilities
fn benchmark_performance_measurement(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_measurement");
    
    let measurer = PerformanceMeasurer::new();
    let cache = SimpleCache::new(100);
    
    // Measure overhead of performance monitoring
    group.bench_function("without_measurement", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let result = (i as f64).powf(0.88);
                black_box(result);
            }
        })
    });
    
    group.bench_function("with_measurement", |b| {
        b.iter(|| {
            measurer.measure(|| {
                for i in 0..1000 {
                    let result = (i as f64).powf(0.88);
                    black_box(result);
                }
            })
        })
    });
    
    group.bench_function("with_caching_and_measurement", |b| {
        b.iter(|| {
            measurer.measure(|| {
                for i in 0..1000 {
                    let result = cache.cached_pow(i as f64, 0.88);
                    black_box(result);
                }
            })
        })
    });
    
    group.finish();
}

criterion_group!(
    performance_validation,
    benchmark_threading_speedup,
    benchmark_caching_speedup,
    benchmark_trading_scenarios,
    benchmark_memory_efficiency,
    benchmark_speedup_demonstration,
    benchmark_performance_measurement
);

criterion_main!(performance_validation);