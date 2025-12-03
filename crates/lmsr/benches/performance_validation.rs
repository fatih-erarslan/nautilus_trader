//! Performance validation benchmarks for LMSR demonstrating speedup achievements
//! 
//! This benchmark validates the performance improvements achievable
//! with the optimizations implemented in the LMSR crate.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use lmsr_rs::*;
use lmsr_rs::performance_simple::*;

use std::sync::Arc;
use std::thread;
use std::time::Instant;

/// Benchmark single-threaded vs multi-threaded LMSR calculations
fn benchmark_lmsr_threading_speedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("lmsr_threading_speedup");
    
    let single_calc = SimpleBatchCalculator::new(10, 5000.0, Some(1)).unwrap();
    let multi_calc = SimpleBatchCalculator::new(10, 5000.0, Some(4)).unwrap();
    
    // Large batch for demonstrating parallelism benefits
    let batch_size = 1000;
    let quantities_batch: Vec<Vec<f64>> = (0..batch_size)
        .map(|i| (0..10).map(|j| (i * j + 100) as f64).collect())
        .collect();
    
    group.throughput(Throughput::Elements(batch_size as u64));
    
    group.bench_function("single_thread_1000", |b| {
        b.iter(|| {
            single_calc.calculate_batch_prices(black_box(&quantities_batch)).unwrap()
        })
    });
    
    group.bench_function("multi_thread_1000", |b| {
        b.iter(|| {
            multi_calc.calculate_batch_prices(black_box(&quantities_batch)).unwrap()
        })
    });
    
    group.finish();
}

/// Benchmark HFT performance with optimizations
fn benchmark_hft_performance_speedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("hft_performance");
    
    let standard_market = Market::new(5, 10000.0).unwrap();
    let hft_market = SimpleHFTMarketMaker::new(5, 10000.0, Some(4)).unwrap();
    
    let trade_quantities = vec![1.0, 0.5, 0.3, 0.2, 0.1];
    
    group.bench_function("standard_market_trade", |b| {
        let mut counter = 0;
        b.iter(|| {
            counter += 1;
            let trader_id = format!("trader_{}", counter);
            black_box(standard_market.execute_trade(trader_id, black_box(&trade_quantities)).unwrap())
        })
    });
    
    group.bench_function("hft_optimized_trade", |b| {
        let mut counter = 0;
        b.iter(|| {
            counter += 1;
            let trader_id = format!("trader_{}", counter);
            black_box(hft_market.execute_trade(trader_id, black_box(&trade_quantities)).unwrap())
        })
    });
    
    group.finish();
}

/// Benchmark concurrent market access optimizations
fn benchmark_concurrent_access_speedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_access");
    
    let standard_market = Arc::new(Market::new(8, 5000.0).unwrap());
    let hft_market = Arc::new(SimpleHFTMarketMaker::new(8, 5000.0, Some(8)).unwrap());
    
    let trade_quantities = vec![1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0];
    
    // Test with different thread counts
    for num_threads in [2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("standard_concurrent", num_threads),
            num_threads,
            |b, &threads| {
                b.iter(|| {
                    let handles: Vec<_> = (0..threads).map(|i| {
                        let market = Arc::clone(&standard_market);
                        let quantities = trade_quantities.clone();
                        thread::spawn(move || {
                            for j in 0..50 {
                                let trader_id = format!("trader_{}_{}", i, j);
                                let _ = market.execute_trade(trader_id, &quantities);
                            }
                        })
                    }).collect();
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("hft_concurrent", num_threads),
            num_threads,
            |b, &threads| {
                b.iter(|| {
                    let handles: Vec<_> = (0..threads).map(|i| {
                        let market = Arc::clone(&hft_market);
                        let quantities = trade_quantities.clone();
                        thread::spawn(move || {
                            for j in 0..50 {
                                let trader_id = format!("trader_{}_{}", i, j);
                                let _ = market.execute_trade(trader_id, &quantities);
                            }
                        })
                    }).collect();
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark streaming processing optimizations
fn benchmark_streaming_speedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_processing");
    
    let calculator = LMSRCalculator::new(15, 3000.0).unwrap();
    let streaming_processor = SimpleStreamingProcessor::new(calculator.clone(), 100);
    
    // Generate test data
    let test_quantities: Vec<Vec<f64>> = (0..1000)
        .map(|i| (0..15).map(|j| ((i + j) * 10) as f64).collect())
        .collect();
    
    group.throughput(Throughput::Elements(1000));
    
    // Standard sequential processing
    group.bench_function("sequential_processing", |b| {
        b.iter(|| {
            let results: Vec<Vec<f64>> = test_quantities.iter()
                .map(|quantities| calculator.all_marginal_prices(quantities).unwrap())
                .collect();
            black_box(results)
        })
    });
    
    // Streaming processing
    group.bench_function("streaming_processing", |b| {
        b.iter(|| {
            let mut all_results = Vec::new();
            for quantities in &test_quantities {
                if let Some(results) = streaming_processor.add_quantities(quantities.clone()).unwrap() {
                    all_results.extend(results);
                }
            }
            // Flush remaining
            let final_results = streaming_processor.flush().unwrap();
            all_results.extend(final_results);
            black_box(all_results)
        })
    });
    
    group.finish();
}

/// Benchmark real-world trading scenarios
fn benchmark_real_world_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_world_scenarios");
    
    // Market making scenario: frequent price calculations
    let market_calc = LMSRCalculator::new(10, 50000.0).unwrap();
    let batch_calc = SimpleBatchCalculator::new(10, 50000.0, Some(4)).unwrap();
    
    let market_quantities: Vec<Vec<f64>> = (0..1000)
        .map(|i| (0..10).map(|j| ((i + j) * 100) as f64).collect())
        .collect();
    
    group.throughput(Throughput::Elements(1000));
    
    group.bench_function("market_making_sequential", |b| {
        b.iter(|| {
            let results: Vec<Vec<f64>> = market_quantities.iter()
                .map(|quantities| market_calc.all_marginal_prices(quantities).unwrap())
                .collect();
            black_box(results)
        })
    });
    
    group.bench_function("market_making_batch", |b| {
        b.iter(|| {
            batch_calc.calculate_batch_prices(black_box(&market_quantities)).unwrap()
        })
    });
    
    // Arbitrage detection scenario
    let arbitrage_calc = LMSRCalculator::new(5, 25000.0).unwrap();
    let current_quantities = vec![1000.0, 2000.0, 1500.0, 3000.0, 2500.0];
    // External prices must sum to 1.0 for valid arbitrage detection
    // Base prices: [0.2, 0.25, 0.15, 0.25, 0.15] = 1.0
    let external_prices_batch: Vec<Vec<f64>> = (0..500)
        .map(|_i| {
            // Fixed valid probability distribution summing to 1.0
            vec![0.20, 0.25, 0.15, 0.25, 0.15]
        })
        .collect();
    
    group.throughput(Throughput::Elements(500));
    
    group.bench_function("arbitrage_detection_sequential", |b| {
        b.iter(|| {
            let results: Vec<Vec<f64>> = external_prices_batch.iter()
                .map(|external_prices| {
                    arbitrage_calc.optimal_arbitrage_trade(
                        black_box(&current_quantities),
                        black_box(external_prices)
                    ).unwrap()
                })
                .collect();
            black_box(results)
        })
    });
    
    // Binary trading scenario (high frequency)
    let binary_calc = LMSRCalculator::new(2, 100000.0).unwrap();
    let binary_quantities = vec![50000.0, 45000.0];
    
    group.bench_function("binary_hft_pricing", |b| {
        b.iter(|| {
            for _ in 0..10000 {
                let price = binary_calc.marginal_price(black_box(&binary_quantities), 0).unwrap();
                black_box(price);
            }
        })
    });
    
    group.finish();
}

/// Benchmark memory efficiency improvements
fn benchmark_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    
    // Test with different data sizes
    for num_outcomes in [10, 50, 100].iter() {
        let calc = LMSRCalculator::new(*num_outcomes, 10000.0).unwrap();
        let batch_calc = SimpleBatchCalculator::new(*num_outcomes, 10000.0, Some(4)).unwrap();
        
        let quantities_batch: Vec<Vec<f64>> = (0..1000)
            .map(|i| (0..*num_outcomes).map(|j| ((i + j) * 50) as f64).collect())
            .collect();
        
        group.throughput(Throughput::Bytes((1000 * num_outcomes * std::mem::size_of::<f64>()) as u64));
        
        group.bench_with_input(
            BenchmarkId::new("sequential_memory", num_outcomes),
            &quantities_batch,
            |b, batch| {
                b.iter(|| {
                    let results: Vec<Vec<f64>> = batch.iter()
                        .map(|quantities| calc.all_marginal_prices(quantities).unwrap())
                        .collect();
                    black_box(results)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("optimized_memory", num_outcomes),
            &quantities_batch,
            |b, batch| {
                b.iter(|| {
                    batch_calc.calculate_batch_prices(black_box(batch)).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

/// Demonstrate maximum speedup achievements
fn benchmark_maximum_speedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("maximum_speedup");
    
    // Large-scale calculation for maximum speedup demonstration
    let large_outcomes = 50;
    let large_calc = LMSRCalculator::new(large_outcomes, 100000.0).unwrap();
    let large_batch_calc = SimpleBatchCalculator::new(large_outcomes, 100000.0, Some(8)).unwrap();
    
    let large_batch_size = 5000;
    let large_quantities_batch: Vec<Vec<f64>> = (0..large_batch_size)
        .map(|i| (0..large_outcomes).map(|j| ((i + j) * 200) as f64).collect())
        .collect();
    
    group.sample_size(10); // Fewer samples for large datasets
    group.throughput(Throughput::Elements(large_batch_size as u64));
    
    // Baseline: single-threaded calculation
    group.bench_function("baseline_sequential_5k", |b| {
        b.iter(|| {
            let start = Instant::now();
            let results: Vec<Vec<f64>> = large_quantities_batch.iter()
                .map(|quantities| large_calc.all_marginal_prices(quantities).unwrap())
                .collect();
            let duration = start.elapsed();
            println!("Sequential execution time: {:?}", duration);
            black_box(results)
        })
    });
    
    // Optimized: multi-threaded batch processing
    group.bench_function("optimized_parallel_5k", |b| {
        b.iter(|| {
            let start = Instant::now();
            let results = large_batch_calc.calculate_batch_prices(black_box(&large_quantities_batch)).unwrap();
            let duration = start.elapsed();
            println!("Parallel execution time: {:?}", duration);
            black_box(results)
        })
    });
    
    group.finish();
}

/// Numerical stability under optimizations
fn benchmark_numerical_stability(c: &mut Criterion) {
    let mut group = c.benchmark_group("numerical_stability");
    
    // Use larger liquidity parameter to handle wider range of quantities
    let calc = LMSRCalculator::new(20, 100000.0).unwrap();
    let batch_calc = SimpleBatchCalculator::new(20, 100000.0, Some(4)).unwrap();

    // Test values that stress numerical precision but stay within valid bounds
    // For LMSR: exp(q/b) must remain finite, so q/b must be < ~700 for f64
    // With b=100000, quantities up to ~70 million are safe
    let extreme_batch: Vec<Vec<f64>> = vec![
        // Mix of varying magnitudes (staying within bounds for b=100000)
        vec![1000.0, 0.1, 10000.0, 0.01, 50000.0, 0.001, 100000.0, 0.0001, 500000.0, 1e-5,
             1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 500000.0, 1000000.0, 2000000.0, 5000000.0],
        // Uniform moderate values
        vec![100000.0; 20],
        // Uniform small values
        vec![0.001; 20],
        // Linear progression of reasonable values
        vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0,
             1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0, 2000.0],
    ];
    
    group.bench_function("standard_extreme_values", |b| {
        b.iter(|| {
            let results: Vec<Vec<f64>> = extreme_batch.iter()
                .map(|quantities| calc.all_marginal_prices(quantities).unwrap())
                .collect();
            black_box(results)
        })
    });
    
    group.bench_function("optimized_extreme_values", |b| {
        b.iter(|| {
            batch_calc.calculate_batch_prices(black_box(&extreme_batch)).unwrap()
        })
    });
    
    group.finish();
}

criterion_group!(
    lmsr_performance_validation,
    benchmark_lmsr_threading_speedup,
    benchmark_hft_performance_speedup,
    benchmark_concurrent_access_speedup,
    benchmark_streaming_speedup,
    benchmark_real_world_scenarios,
    benchmark_memory_efficiency,
    benchmark_maximum_speedup,
    benchmark_numerical_stability
);

criterion_main!(lmsr_performance_validation);