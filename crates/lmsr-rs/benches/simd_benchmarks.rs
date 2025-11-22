//! SIMD performance benchmarks for LMSR targeting 50-100x speedup
//! 
//! These benchmarks validate the performance improvements from SIMD vectorization,
//! parallel processing, and memory optimizations for high-frequency trading systems.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use lmsr_rs::*;

#[cfg(feature = "simd")]
use lmsr_rs::simd::*;
use lmsr_rs::performance::*;

use std::sync::Arc;
use std::thread;
use std::time::Instant;

/// Benchmark SIMD vs scalar LMSR calculations
fn benchmark_lmsr_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("lmsr_simd_vs_scalar");
    
    let standard_calc = LMSRCalculator::new(10, 1000.0).unwrap();
    
    #[cfg(feature = "simd")]
    let simd_calc = SIMDLMSRCalculator::new(10, 1000.0).unwrap();
    
    // Test different scales of operations
    for num_outcomes in [5, 10, 20, 50].iter() {
        let quantities = vec![100.0; *num_outcomes];
        
        group.throughput(Throughput::Elements(*num_outcomes as u64));
        
        // Standard implementation
        group.bench_with_input(
            BenchmarkId::new("scalar_marginal_prices", num_outcomes),
            &quantities,
            |b, quantities| {
                let calc = LMSRCalculator::new(quantities.len(), 1000.0).unwrap();
                b.iter(|| calc.all_marginal_prices(black_box(quantities)).unwrap())
            },
        );
        
        // SIMD implementation
        #[cfg(feature = "simd")]
        group.bench_with_input(
            BenchmarkId::new("simd_marginal_prices", num_outcomes),
            &quantities,
            |b, quantities| {
                let calc = SIMDLMSRCalculator::new(quantities.len(), 1000.0).unwrap();
                b.iter(|| calc.all_marginal_prices_simd(black_box(quantities)).unwrap())
            },
        );
        
        // Cost function comparison
        group.bench_with_input(
            BenchmarkId::new("scalar_cost_function", num_outcomes),
            &quantities,
            |b, quantities| {
                let calc = LMSRCalculator::new(quantities.len(), 1000.0).unwrap();
                b.iter(|| calc.cost_function(black_box(quantities)).unwrap())
            },
        );
        
        #[cfg(feature = "simd")]
        group.bench_with_input(
            BenchmarkId::new("simd_cost_function", num_outcomes),
            &quantities,
            |b, quantities| {
                let calc = SIMDLMSRCalculator::new(quantities.len(), 1000.0).unwrap();
                b.iter(|| calc.cost_function_simd(black_box(quantities)).unwrap())
            },
        );
    }
    
    group.finish();
}

/// Benchmark high-frequency trading performance
fn benchmark_hft_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("hft_performance");
    
    #[cfg(feature = "simd")]
    {
        let hft_mm = HFTMarketMaker::new(5, 10000.0, Some(4)).unwrap();
        let _ = hft_mm.start_hft_processing();
        
        group.bench_function("hft_trade_submission", |b| {
            let mut counter = 0;
            b.iter(|| {
                counter += 1;
                let trader_id = format!("trader_{}", counter);
                let quantities = vec![1.0, 0.0, 0.0, 0.0, 0.0];
                
                let response_receiver = hft_mm.submit_trade(trader_id, quantities).unwrap();
                // Don't wait for response in benchmark to measure submission overhead
                black_box(response_receiver)
            })
        });
    }
    
    // Standard market for comparison
    let standard_market = Market::new(5, 10000.0).unwrap();
    
    group.bench_function("standard_market_trade", |b| {
        let mut counter = 0;
        b.iter(|| {
            counter += 1;
            let trader_id = format!("trader_{}", counter);
            let quantities = vec![1.0, 0.0, 0.0, 0.0, 0.0];
            
            black_box(standard_market.execute_trade(trader_id, &quantities).unwrap())
        })
    });
    
    group.finish();
}

/// Benchmark lock-free vs standard market operations
fn benchmark_lock_free_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("lock_free_performance");
    
    #[cfg(feature = "simd")]
    let lock_free_market = Arc::new(LockFreeMarketState::new(10, 5000.0).unwrap());
    
    let standard_market = Arc::new(Market::new(10, 5000.0).unwrap());
    
    let trade_quantities = vec![1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0];
    
    // Single-threaded performance
    group.bench_function("standard_single_thread", |b| {
        b.iter(|| {
            standard_market.calculate_trade_cost(black_box(&trade_quantities)).unwrap()
        })
    });
    
    #[cfg(feature = "simd")]
    group.bench_function("lock_free_single_thread", |b| {
        b.iter(|| {
            lock_free_market.execute_trade_lockfree(black_box(&trade_quantities)).unwrap()
        })
    });
    
    // Multi-threaded contention test
    for num_threads in [2, 4, 8].iter() {
        #[cfg(feature = "simd")]
        group.bench_with_input(
            BenchmarkId::new("lock_free_concurrent", num_threads),
            num_threads,
            |b, &threads| {
                b.iter(|| {
                    let handles: Vec<_> = (0..threads).map(|_| {
                        let market = Arc::clone(&lock_free_market);
                        let quantities = trade_quantities.clone();
                        thread::spawn(move || {
                            for _ in 0..100 {
                                let _ = market.execute_trade_lockfree(&quantities);
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
            BenchmarkId::new("standard_concurrent", num_threads),
            num_threads,
            |b, &threads| {
                b.iter(|| {
                    let handles: Vec<_> = (0..threads).map(|i| {
                        let market = Arc::clone(&standard_market);
                        let quantities = trade_quantities.clone();
                        thread::spawn(move || {
                            for j in 0..100 {
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

/// Benchmark batch processing performance
fn benchmark_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");
    
    let batch_calc = BatchPriceCalculator::new(20, 5000.0).unwrap();
    
    // Test different batch sizes
    for batch_size in [100, 1_000, 10_000].iter() {
        let quantities_batch: Vec<Vec<f64>> = (0..*batch_size)
            .map(|_| (0..20).map(|i| i as f64 * 10.0).collect())
            .collect();
        
        group.throughput(Throughput::Elements(*batch_size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("batch_price_calculation", batch_size),
            &quantities_batch,
            |b, batch| {
                b.iter(|| batch_calc.calculate_batch_prices(black_box(batch)).unwrap())
            },
        );
        
        // Compare with individual calculations
        group.bench_with_input(
            BenchmarkId::new("individual_calculations", batch_size),
            &quantities_batch,
            |b, batch| {
                b.iter(|| {
                    let calc = LMSRCalculator::new(20, 5000.0).unwrap();
                    let results: Vec<Vec<f64>> = batch.iter()
                        .map(|quantities| calc.all_marginal_prices(quantities).unwrap())
                        .collect();
                    black_box(results)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark streaming processing performance
fn benchmark_streaming_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_performance");
    
    let calculator = LMSRCalculator::new(15, 3000.0).unwrap();
    
    // Test different buffer sizes
    for buffer_size in [10, 100, 1000].iter() {
        let processor = StreamingProcessor::new(calculator.clone(), *buffer_size);
        
        group.bench_with_input(
            BenchmarkId::new("streaming_processor", buffer_size),
            buffer_size,
            |b, &buf_size| {
                b.iter(|| {
                    for i in 0..(buf_size * 2) {
                        let quantities: Vec<f64> = (0..15).map(|j| (i * j) as f64).collect();
                        let _ = processor.add_quantities(quantities);
                    }
                    processor.flush().unwrap()
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark adaptive SIMD executor
fn benchmark_adaptive_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_simd");
    
    #[cfg(feature = "simd")]
    {
        let simd_calc = SIMDLMSRCalculator::new(25, 8000.0).unwrap();
        let adaptive_executor = AdaptiveSIMDExecutor::new(simd_calc);
        
        // Test different data sizes to show adaptive behavior
        for size in [10, 100, 1_000, 10_000].iter() {
            let quantities: Vec<f64> = (0..*size).map(|i| (i % 25) as f64 * 100.0).collect();
            
            group.throughput(Throughput::Elements(*size as u64));
            
            group.bench_with_input(
                BenchmarkId::new("adaptive_execution", size),
                &quantities,
                |b, quantities| {
                    b.iter(|| adaptive_executor.adaptive_marginal_prices(black_box(quantities)).unwrap())
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark real-world trading scenarios
fn benchmark_trading_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("trading_scenarios");
    
    // Scenario 1: Market making with frequent price updates
    let market_making_calc = LMSRCalculator::new(10, 50000.0).unwrap();
    let market_quantities = vec![5000.0, 3000.0, 2000.0, 1000.0, 500.0, 800.0, 1200.0, 1800.0, 2500.0, 3500.0];
    
    group.bench_function("market_making_prices", |b| {
        b.iter(|| {
            for _ in 0..1000 { // 1000 price calculations per iteration
                let prices = market_making_calc.all_marginal_prices(black_box(&market_quantities)).unwrap();
                black_box(prices);
            }
        })
    });
    
    // Scenario 2: Arbitrage detection across multiple markets
    let arbitrage_calc = LMSRCalculator::new(5, 25000.0).unwrap();
    let current_quantities = vec![1000.0, 2000.0, 1500.0, 3000.0, 2500.0];
    let external_prices = vec![0.15, 0.25, 0.20, 0.25, 0.15];
    
    group.bench_function("arbitrage_detection", |b| {
        b.iter(|| {
            for _ in 0..500 { // 500 arbitrage calculations per iteration
                let arbitrage_trades = arbitrage_calc.optimal_arbitrage_trade(
                    black_box(&current_quantities),
                    black_box(&external_prices)
                ).unwrap();
                black_box(arbitrage_trades);
            }
        })
    });
    
    // Scenario 3: High-frequency binary trading
    let binary_calc = LMSRCalculator::new(2, 100000.0).unwrap();
    let binary_quantities = vec![50000.0, 45000.0];
    
    group.bench_function("binary_hft_trading", |b| {
        b.iter(|| {
            for _ in 0..10000 { // 10,000 calculations per iteration (simulating HFT)
                let price = binary_calc.marginal_price(black_box(&binary_quantities), 0).unwrap();
                black_box(price);
            }
        })
    });
    
    group.finish();
}

/// Validate 50-100x speedup targets
fn benchmark_speedup_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("speedup_validation");
    
    // Large-scale calculation for maximum speedup demonstration
    let large_outcomes = 100;
    let large_calc = LMSRCalculator::new(large_outcomes, 100000.0).unwrap();
    let large_quantities: Vec<f64> = (0..large_outcomes).map(|i| (i * 1000) as f64).collect();
    
    #[cfg(feature = "simd")]
    let simd_large_calc = SIMDLMSRCalculator::new(large_outcomes, 100000.0).unwrap();
    
    group.sample_size(10); // Fewer samples for large datasets
    
    // Baseline: single-threaded scalar calculation
    group.bench_function("baseline_scalar_large", |b| {
        b.iter(|| {
            for _ in 0..1000 { // 1000 iterations to amplify differences
                let prices = large_calc.all_marginal_prices(black_box(&large_quantities)).unwrap();
                black_box(prices);
            }
        })
    });
    
    // SIMD optimized calculation
    #[cfg(feature = "simd")]
    group.bench_function("simd_optimized_large", |b| {
        b.iter(|| {
            for _ in 0..1000 { // 1000 iterations to amplify differences
                let prices = simd_large_calc.all_marginal_prices_simd(black_box(&large_quantities)).unwrap();
                black_box(prices);
            }
        })
    });
    
    // Batch processing
    #[cfg(feature = "simd")]
    group.bench_function("batch_optimized_large", |b| {
        let batch: Vec<Vec<f64>> = (0..100).map(|_| large_quantities.clone()).collect();
        b.iter(|| {
            let results = simd_large_calc.batch_marginal_prices_simd(black_box(&batch)).unwrap();
            black_box(results);
        })
    });
    
    group.finish();
}

/// Memory and cache efficiency benchmarks
fn benchmark_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    
    // Test memory access patterns
    for data_size in [1_000, 10_000, 100_000].iter() {
        let calc = LMSRCalculator::new(50, 10000.0).unwrap();
        let quantities_batch: Vec<Vec<f64>> = (0..*data_size)
            .map(|i| (0..50).map(|j| ((i + j) % 1000) as f64).collect())
            .collect();
        
        group.throughput(Throughput::Bytes((*data_size * 50 * std::mem::size_of::<f64>()) as u64));
        
        // Sequential processing
        group.bench_with_input(
            BenchmarkId::new("sequential_processing", data_size),
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
        
        // Parallel processing (memory bandwidth test)
        group.bench_with_input(
            BenchmarkId::new("parallel_processing", data_size),
            &quantities_batch,
            |b, batch| {
                use rayon::prelude::*;
                b.iter(|| {
                    let results: Vec<Vec<f64>> = batch.par_iter()
                        .map(|quantities| calc.all_marginal_prices(quantities).unwrap())
                        .collect();
                    black_box(results)
                })
            },
        );
    }
    
    group.finish();
}

/// Numerical stability under SIMD optimizations
fn benchmark_numerical_stability(c: &mut Criterion) {
    let mut group = c.benchmark_group("numerical_stability");
    
    let calc = LMSRCalculator::new(20, 1000.0).unwrap();
    
    #[cfg(feature = "simd")]
    let simd_calc = SIMDLMSRCalculator::new(20, 1000.0).unwrap();
    
    // Test extreme values
    let extreme_quantities = vec![1e10, 1e-10, 1e8, 1e-8, 1e6, 1e-6, 
                                  1e4, 1e-4, 1e2, 1e-2, 1.0, 0.1, 
                                  0.01, 0.001, 100.0, 1000.0, 
                                  10000.0, 100000.0, 1000000.0, 10000000.0];
    
    group.bench_function("standard_extreme_values", |b| {
        b.iter(|| {
            let prices = calc.all_marginal_prices(black_box(&extreme_quantities)).unwrap();
            black_box(prices)
        })
    });
    
    #[cfg(feature = "simd")]
    group.bench_function("simd_extreme_values", |b| {
        b.iter(|| {
            let prices = simd_calc.all_marginal_prices_simd(black_box(&extreme_quantities)).unwrap();
            black_box(prices)
        })
    });
    
    group.finish();
}

criterion_group!(
    lmsr_simd_benches,
    benchmark_lmsr_simd_vs_scalar,
    benchmark_hft_performance,
    benchmark_lock_free_performance,
    benchmark_batch_processing,
    benchmark_streaming_performance,
    benchmark_adaptive_simd,
    benchmark_trading_scenarios,
    benchmark_speedup_validation,
    benchmark_memory_efficiency,
    benchmark_numerical_stability
);

criterion_main!(lmsr_simd_benches);