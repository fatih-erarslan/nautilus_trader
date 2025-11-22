//! SIMD performance benchmarks targeting 50-100x speedup
//! 
//! These benchmarks validate the performance improvements from SIMD vectorization,
//! parallel processing, and memory optimizations for financial trading systems.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use prospect_theory_rs::*;

#[cfg(feature = "simd")]
use prospect_theory_rs::simd::*;
use prospect_theory_rs::performance::*;

use std::time::Instant;

/// Benchmark SIMD vs scalar implementations
fn benchmark_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_scalar");
    
    let params = ValueFunctionParams::default();
    let standard_vf = ValueFunction::new(params.clone()).unwrap();
    
    #[cfg(feature = "simd")]
    let simd_vf = SIMDValueFunction::new(params).unwrap();
    
    // Test different data sizes to show scaling benefits
    for size in [100, 1_000, 10_000, 100_000].iter() {
        let outcomes: Vec<f64> = (0..*size).map(|i| (i as f64) - (*size as f64) / 2.0).collect();
        
        group.throughput(Throughput::Elements(*size as u64));
        
        // Scalar baseline
        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &outcomes,
            |b, outcomes| {
                b.iter(|| standard_vf.values(black_box(outcomes)).unwrap())
            },
        );
        
        // SIMD implementation
        #[cfg(feature = "simd")]
        group.bench_with_input(
            BenchmarkId::new("simd", size),
            &outcomes,
            |b, outcomes| {
                b.iter(|| simd_vf.values_simd(black_box(outcomes)).unwrap())
            },
        );
        
        // Parallel SIMD implementation
        #[cfg(feature = "simd")]
        group.bench_with_input(
            BenchmarkId::new("parallel_simd", size),
            &outcomes,
            |b, outcomes| {
                b.iter(|| simd_vf.values_parallel_simd(black_box(outcomes)).unwrap())
            },
        );
    }
    
    group.finish();
}

/// Benchmark probability weighting SIMD vs scalar
fn benchmark_probability_weighting_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("probability_weighting_simd");
    
    let params = WeightingParams::default();
    let standard_pw = ProbabilityWeighting::default_tk();
    
    #[cfg(feature = "simd")]
    let simd_pw = SIMDProbabilityWeighting::new(params).unwrap();
    
    for size in [100, 1_000, 10_000, 100_000].iter() {
        let probabilities: Vec<f64> = (1..=*size).map(|i| (i as f64) / (*size as f64 + 1.0)).collect();
        
        group.throughput(Throughput::Elements(*size as u64));
        
        // Scalar baseline
        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &probabilities,
            |b, probs| {
                b.iter(|| standard_pw.weights_gains(black_box(probs)).unwrap())
            },
        );
        
        // SIMD implementation
        #[cfg(feature = "simd")]
        group.bench_with_input(
            BenchmarkId::new("simd", size),
            &probabilities,
            |b, probs| {
                b.iter(|| simd_pw.weights_gains_simd(black_box(probs)).unwrap())
            },
        );
        
        // Parallel SIMD implementation
        #[cfg(feature = "simd")]
        group.bench_with_input(
            BenchmarkId::new("parallel_simd", size),
            &probabilities,
            |b, probs| {
                b.iter(|| simd_pw.weights_gains_parallel_simd(black_box(probs)).unwrap())
            },
        );
    }
    
    group.finish();
}

/// Benchmark complete prospect theory calculation with SIMD
fn benchmark_complete_prospect_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("complete_prospect_simd");
    
    let value_params = ValueFunctionParams::default();
    let weighting_params = WeightingParams::default();
    
    #[cfg(feature = "simd")]
    let mut simd_calculator = HighPerformanceProspectCalculator::new(
        value_params.clone(),
        weighting_params.clone()
    ).unwrap();
    
    // Standard calculation
    let standard_vf = ValueFunction::new(value_params).unwrap();
    let standard_pw = ProbabilityWeighting::default_tk();
    
    for size in [10, 50, 100, 500].iter() {
        let outcomes: Vec<f64> = (0..*size).map(|i| (i as f64) - (*size as f64) / 2.0).collect();
        let probabilities: Vec<f64> = (0..*size).map(|_| 1.0 / *size as f64).collect();
        
        group.throughput(Throughput::Elements(*size as u64));
        
        // Standard implementation
        group.bench_with_input(
            BenchmarkId::new("standard", size),
            &(outcomes.clone(), probabilities.clone()),
            |b, (outcomes, probabilities)| {
                b.iter(|| {
                    let values = standard_vf.values(black_box(outcomes)).unwrap();
                    let decision_weights = standard_pw
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
        
        // SIMD optimized implementation
        #[cfg(feature = "simd")]
        group.bench_with_input(
            BenchmarkId::new("simd_optimized", size),
            &(outcomes.clone(), probabilities.clone()),
            |b, (outcomes, probabilities)| {
                b.iter(|| {
                    simd_calculator.calculate_prospect_value(
                        black_box(outcomes), 
                        black_box(probabilities)
                    ).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark batch processing performance
fn benchmark_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");
    
    let value_params = ValueFunctionParams::default();
    let weighting_params = WeightingParams::default();
    
    let batch_processor = BatchProcessor::new(
        value_params,
        weighting_params,
        Some(4) // Use 4 threads
    ).unwrap();
    
    // Create large batches for testing
    for batch_size in [100, 1_000, 10_000].iter() {
        let outcomes_batch: Vec<Vec<f64>> = (0..*batch_size)
            .map(|_| vec![100.0, 0.0, -100.0, 50.0, -50.0])
            .collect();
        let probabilities_batch: Vec<Vec<f64>> = (0..*batch_size)
            .map(|_| vec![0.2, 0.2, 0.2, 0.2, 0.2])
            .collect();
        
        group.throughput(Throughput::Elements(*batch_size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("batch_processor", batch_size),
            &(outcomes_batch, probabilities_batch),
            |b, (outcomes_batch, probabilities_batch)| {
                b.iter(|| {
                    batch_processor.process_batch(
                        black_box(outcomes_batch),
                        black_box(probabilities_batch)
                    ).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory allocation optimizations
fn benchmark_memory_optimizations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_optimizations");
    
    let allocator = FinancialAllocator::new();
    let object_pool = ObjectPool::new(|| Vec::<f64>::with_capacity(1000), 16);
    
    // Standard allocation
    group.bench_function("standard_allocation", |b| {
        b.iter(|| {
            let mut vectors = Vec::new();
            for _ in 0..100 {
                vectors.push(black_box(vec![0.0; 1000]));
            }
            black_box(vectors)
        })
    });
    
    // Object pool allocation
    group.bench_function("pool_allocation", |b| {
        b.iter(|| {
            let mut vectors = Vec::new();
            for _ in 0..100 {
                let mut vec = object_pool.get();
                vec.resize(1000, 0.0);
                vectors.push(black_box(vec));
            }
            // Return to pool
            for vec in vectors {
                object_pool.put(vec);
            }
        })
    });
    
    group.finish();
}

/// Benchmark cache optimization
fn benchmark_cache_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_optimization");
    
    let cache_calc = CacheOptimizedCalculator::new();
    
    // Test cache hit performance
    group.bench_function("cached_power_calculation", |b| {
        b.iter(|| {
            // These will hit cache after first calculation
            for i in 0..100 {
                let base = 2.0 + (i % 10) as f64;
                let exp = 0.88;
                black_box(cache_calc.cached_pow(base, exp));
            }
        })
    });
    
    // Test standard power calculation
    group.bench_function("standard_power_calculation", |b| {
        b.iter(|| {
            for i in 0..100 {
                let base = 2.0 + (i % 10) as f64;
                let exp = 0.88;
                black_box(base.powf(exp));
            }
        })
    });
    
    group.finish();
}

/// Real-world trading scenario benchmarks
fn benchmark_trading_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("trading_scenarios");
    
    // High-frequency trading scenario: 1000 trades per second
    let hft_outcomes = vec![100.0, -100.0]; // Binary outcome
    let hft_probabilities = vec![0.6, 0.4];
    
    let value_params = ValueFunctionParams::default();
    let weighting_params = WeightingParams::default();
    
    #[cfg(feature = "simd")]
    let mut hft_calculator = HighPerformanceProspectCalculator::new(
        value_params.clone(),
        weighting_params.clone()
    ).unwrap();
    
    let standard_vf = ValueFunction::new(value_params).unwrap();
    let standard_pw = ProbabilityWeighting::default_tk();
    
    group.bench_function("hft_single_calculation_standard", |b| {
        b.iter(|| {
            let values = standard_vf.values(black_box(&hft_outcomes)).unwrap();
            let decision_weights = standard_pw
                .decision_weights(black_box(&hft_probabilities), black_box(&hft_outcomes))
                .unwrap();
            
            let prospect_value: f64 = values
                .iter()
                .zip(decision_weights.iter())
                .map(|(&value, &weight)| value * weight)
                .sum();
            
            black_box(prospect_value)
        })
    });
    
    #[cfg(feature = "simd")]
    group.bench_function("hft_single_calculation_simd", |b| {
        b.iter(|| {
            hft_calculator.calculate_prospect_value(
                black_box(&hft_outcomes),
                black_box(&hft_probabilities)
            ).unwrap()
        })
    });
    
    // Market making scenario: 10,000 price calculations per second
    let market_quantities = vec![1000.0, 2000.0, 1500.0, 3000.0, 2500.0];
    
    group.bench_function("market_making_scenario", |b| {
        b.iter(|| {
            for _ in 0..100 { // Simulate 100 rapid calculations
                let values = standard_vf.values(black_box(&market_quantities)).unwrap();
                black_box(values);
            }
        })
    });
    
    group.finish();
}

/// Measure actual speedup ratios
fn benchmark_speedup_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("speedup_validation");
    
    let value_params = ValueFunctionParams::default();
    let weighting_params = WeightingParams::default();
    
    // Large dataset for maximum speedup demonstration
    let large_outcomes: Vec<f64> = (0..100_000).map(|i| (i as f64) - 50_000.0).collect();
    let large_probabilities: Vec<f64> = (1..=100_000).map(|i| (i as f64) / 100_001.0).collect();
    
    let standard_vf = ValueFunction::new(value_params.clone()).unwrap();
    
    #[cfg(feature = "simd")]
    let simd_vf = SIMDValueFunction::new(value_params.clone()).unwrap();
    
    #[cfg(feature = "simd")]
    let mut hft_calculator = HighPerformanceProspectCalculator::new(
        value_params,
        weighting_params
    ).unwrap();
    
    group.sample_size(10); // Fewer samples for large datasets
    
    // Baseline: single-threaded scalar
    group.bench_function("baseline_scalar_100k", |b| {
        b.iter(|| {
            standard_vf.values(black_box(&large_outcomes)).unwrap()
        })
    });
    
    // SIMD vectorized
    #[cfg(feature = "simd")]
    group.bench_function("simd_vectorized_100k", |b| {
        b.iter(|| {
            simd_vf.values_simd(black_box(&large_outcomes)).unwrap()
        })
    });
    
    // Parallel SIMD
    #[cfg(feature = "simd")]
    group.bench_function("parallel_simd_100k", |b| {
        b.iter(|| {
            simd_vf.values_parallel_simd(black_box(&large_outcomes)).unwrap()
        })
    });
    
    // Complete optimized prospect calculation
    #[cfg(feature = "simd")]
    group.bench_function("complete_optimized_100k", |b| {
        b.iter(|| {
            hft_calculator.calculate_prospect_value(
                black_box(&large_outcomes[..1000]), // Smaller subset for complete calculation
                black_box(&large_probabilities[..1000])
            ).unwrap()
        })
    });
    
    group.finish();
}

/// Memory bandwidth and cache efficiency benchmarks
fn benchmark_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    
    let optimizer = MemoryBandwidthOptimizer::new();
    
    // Test different data sizes and access patterns
    for size in [1_000, 10_000, 100_000].iter() {
        let data: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        
        group.throughput(Throughput::Bytes((*size * std::mem::size_of::<f64>()) as u64));
        
        // Sequential access
        group.bench_with_input(
            BenchmarkId::new("sequential_access", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut sum = 0.0;
                    for &value in data {
                        sum += black_box(value);
                    }
                    black_box(sum)
                })
            },
        );
        
        // Optimized access pattern
        group.bench_with_input(
            BenchmarkId::new("optimized_access", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let indices = optimizer.optimize_access_pattern(data);
                    let mut sum = 0.0;
                    for &index in &indices {
                        sum += black_box(data[index]);
                    }
                    black_box(sum)
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    simd_benches,
    benchmark_simd_vs_scalar,
    benchmark_probability_weighting_simd,
    benchmark_complete_prospect_simd,
    benchmark_batch_processing,
    benchmark_memory_optimizations,
    benchmark_cache_optimization,
    benchmark_trading_scenarios,
    benchmark_speedup_validation,
    benchmark_memory_efficiency
);

criterion_main!(simd_benches);