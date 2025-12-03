//! Performance benchmarks for LMSR-RS
//! 
//! These benchmarks measure the performance of critical operations
//! to ensure the system meets financial trading requirements.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use lmsr_rs::*;
use std::sync::Arc;
use std::thread;

/// Benchmark basic LMSR calculations
fn bench_lmsr_calculations(c: &mut Criterion) {
    let mut group = c.benchmark_group("lmsr_calculations");
    
    // Test different numbers of outcomes
    for num_outcomes in [2, 5, 10, 20, 50].iter() {
        let calculator = LMSRCalculator::new(*num_outcomes, 1000.0).unwrap();
        let quantities = vec![10.0; *num_outcomes];
        
        group.bench_with_input(
            BenchmarkId::new("marginal_prices", num_outcomes),
            num_outcomes,
            |b, &size| {
                b.iter(|| {
                    black_box(calculator.all_marginal_prices(black_box(&quantities)).unwrap())
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("cost_function", num_outcomes),
            num_outcomes,
            |b, &size| {
                b.iter(|| {
                    black_box(calculator.cost_function(black_box(&quantities)).unwrap())
                })
            },
        );
        
        let buy_amounts = vec![1.0; *num_outcomes];
        group.bench_with_input(
            BenchmarkId::new("trade_cost", num_outcomes),
            num_outcomes,
            |b, &size| {
                b.iter(|| {
                    black_box(calculator.calculate_buy_cost(
                        black_box(&quantities), 
                        black_box(&buy_amounts)
                    ).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark market operations
fn bench_market_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("market_operations");
    
    for num_outcomes in [2, 5, 10].iter() {
        let market = Market::new(*num_outcomes, 1000.0).unwrap();
        let quantities = vec![1.0; *num_outcomes];
        
        group.bench_with_input(
            BenchmarkId::new("get_prices", num_outcomes),
            num_outcomes,
            |b, &size| {
                b.iter(|| {
                    black_box(market.get_prices().unwrap())
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("calculate_trade_cost", num_outcomes),
            num_outcomes,
            |b, &size| {
                b.iter(|| {
                    black_box(market.calculate_trade_cost(black_box(&quantities)).unwrap())
                })
            },
        );
        
        // Benchmark actual trade execution
        let mut trade_counter = 0;
        group.bench_with_input(
            BenchmarkId::new("execute_trade", num_outcomes),
            num_outcomes,
            |b, &size| {
                b.iter(|| {
                    trade_counter += 1;
                    let trader_id = format!("trader_{}", trade_counter);
                    black_box(market.execute_trade(trader_id, black_box(&quantities)).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark concurrent market access
fn bench_concurrent_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_access");
    
    for num_threads in [1, 2, 4, 8].iter() {
        let market = Arc::new(Market::new(5, 1000.0).unwrap());
        
        group.bench_with_input(
            BenchmarkId::new("concurrent_reads", num_threads),
            num_threads,
            |b, &threads| {
                b.iter(|| {
                    let handles: Vec<_> = (0..threads).map(|_| {
                        let market = Arc::clone(&market);
                        thread::spawn(move || {
                            for _ in 0..100 {
                                black_box(market.get_prices().unwrap());
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
            BenchmarkId::new("concurrent_trades", num_threads),
            num_threads,
            |b, &threads| {
                b.iter(|| {
                    let handles: Vec<_> = (0..threads).map(|i| {
                        let market = Arc::clone(&market);
                        thread::spawn(move || {
                            for j in 0..10 {
                                let trader_id = format!("trader_{}_{}", i, j);
                                let quantities = vec![1.0, 0.0, 0.0, 0.0, 0.0];
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

/// Benchmark numerical stability edge cases
fn bench_numerical_stability(c: &mut Criterion) {
    let mut group = c.benchmark_group("numerical_stability");
    
    let calculator = LMSRCalculator::new(10, 1000.0).unwrap();
    
    // Large quantities
    let large_quantities = vec![1e6; 10];
    group.bench_function("large_quantities", |b| {
        b.iter(|| {
            black_box(calculator.all_marginal_prices(black_box(&large_quantities)).unwrap())
        })
    });
    
    // Small quantities
    let small_quantities = vec![1e-6; 10];
    group.bench_function("small_quantities", |b| {
        b.iter(|| {
            black_box(calculator.all_marginal_prices(black_box(&small_quantities)).unwrap())
        })
    });
    
    // Mixed scales
    let mixed_quantities = vec![1e6, 1e-6, 1.0, 1e3, 1e-3, 100.0, 0.01, 1e4, 1e-4, 10.0];
    group.bench_function("mixed_scales", |b| {
        b.iter(|| {
            black_box(calculator.all_marginal_prices(black_box(&mixed_quantities)).unwrap())
        })
    });
    
    // Very small liquidity parameter
    let small_liquidity_calc = LMSRCalculator::new(10, 0.001).unwrap();
    let normal_quantities = vec![1.0; 10];
    group.bench_function("small_liquidity", |b| {
        b.iter(|| {
            black_box(small_liquidity_calc.all_marginal_prices(black_box(&normal_quantities)).unwrap())
        })
    });
    
    group.finish();
}

/// Benchmark utility functions
fn bench_utility_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("utility_functions");
    
    // Test different vector sizes
    for size in [10, 100, 1000].iter() {
        let values: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        
        group.bench_with_input(
            BenchmarkId::new("log_sum_exp", size),
            size,
            |b, &s| {
                b.iter(|| {
                    black_box(lmsr_rs::utils::log_sum_exp(black_box(&values)).unwrap())
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("softmax", size),
            size,
            |b, &s| {
                b.iter(|| {
                    black_box(lmsr_rs::utils::softmax(black_box(&values)).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark position tracking
fn bench_position_tracking(c: &mut Criterion) {
    let mut group = c.benchmark_group("position_tracking");
    
    let position_manager = crate::market::PositionManager::new();
    
    // Benchmark position updates
    group.bench_function("update_position", |b| {
        let mut counter = 0;
        b.iter(|| {
            counter += 1;
            let trader_id = format!("trader_{}", counter);
            let quantities = vec![1.0, 2.0, 3.0];
            black_box(position_manager.update_position(
                trader_id, 
                black_box(&quantities), 
                black_box(10.0)
            ).unwrap())
        })
    });
    
    // Create many positions for retrieval benchmarks
    for i in 0..1000 {
        let trader_id = format!("trader_{}", i);
        let quantities = vec![1.0, 2.0, 3.0];
        position_manager.update_position(trader_id, &quantities, 10.0).unwrap();
    }
    
    group.bench_function("get_position", |b| {
        let mut counter = 0;
        b.iter(|| {
            counter = (counter + 1) % 1000;
            let trader_id = format!("trader_{}", counter);
            black_box(position_manager.get_position(black_box(&trader_id)))
        })
    });
    
    group.bench_function("get_all_positions", |b| {
        b.iter(|| {
            black_box(position_manager.get_all_positions())
        })
    });
    
    let prices = vec![0.3, 0.4, 0.3];
    group.bench_function("calculate_position_value", |b| {
        let mut counter = 0;
        b.iter(|| {
            counter = (counter + 1) % 1000;
            let trader_id = format!("trader_{}", counter);
            black_box(position_manager.calculate_position_value(
                black_box(&trader_id), 
                black_box(&prices)
            ).unwrap())
        })
    });
    
    group.finish();
}

/// Benchmark memory usage patterns
fn bench_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");
    
    // Benchmark market creation/destruction
    group.bench_function("market_lifecycle", |b| {
        b.iter(|| {
            let market = Market::new(5, 1000.0).unwrap();
            for i in 0..10 {
                let trader_id = format!("trader_{}", i);
                let quantities = vec![1.0, 0.0, 0.0, 0.0, 0.0];
                let _ = market.execute_trade(trader_id, &quantities);
            }
            // Market is dropped here
            black_box(market)
        })
    });
    
    // Benchmark large number of small trades
    let market = Market::new(10, 5000.0).unwrap();
    group.bench_function("many_small_trades", |b| {
        let mut counter = 0;
        b.iter(|| {
            for _ in 0..100 {
                counter += 1;
                let trader_id = format!("trader_{}", counter);
                let quantities = vec![0.1; 10];
                black_box(market.execute_trade(trader_id, &quantities).unwrap());
            }
        })
    });
    
    group.finish();
}

/// Benchmark serialization performance
fn bench_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");
    
    let market = Market::new(10, 1000.0).unwrap();
    
    // Execute many trades to create complex state
    for i in 0..1000 {
        let trader_id = format!("trader_{}", i);
        let quantities = vec![1.0; 10];
        let _ = market.execute_trade(trader_id, &quantities);
    }
    
    let state = market.get_state().unwrap();
    
    group.bench_function("serialize_market_state", |b| {
        b.iter(|| {
            black_box(serde_json::to_string(black_box(&state)).unwrap())
        })
    });
    
    let serialized = serde_json::to_string(&state).unwrap();
    group.bench_function("deserialize_market_state", |b| {
        b.iter(|| {
            black_box(serde_json::from_str::<MarketState>(black_box(&serialized)).unwrap())
        })
    });
    
    group.finish();
}

/// Comparison with naive Python-like implementation
fn bench_comparison_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison");
    
    // Our optimized implementation
    let calculator = LMSRCalculator::new(10, 1000.0).unwrap();
    let quantities = vec![10.0; 10];
    
    group.bench_function("rust_optimized", |b| {
        b.iter(|| {
            black_box(calculator.all_marginal_prices(black_box(&quantities)).unwrap())
        })
    });
    
    // Naive implementation mimicking Python behavior
    fn naive_marginal_prices(quantities: &[f64], liquidity: f64) -> Vec<f64> {
        let n = quantities.len();
        let mut exp_values = Vec::with_capacity(n);
        
        // Calculate exp(q_i / b) for each outcome
        for &q in quantities {
            exp_values.push((q / liquidity).exp());
        }
        
        // Calculate sum
        let sum: f64 = exp_values.iter().sum();
        
        // Calculate probabilities
        exp_values.into_iter().map(|exp_val| exp_val / sum).collect()
    }
    
    group.bench_function("naive_implementation", |b| {
        b.iter(|| {
            black_box(naive_marginal_prices(black_box(&quantities), black_box(1000.0)))
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_lmsr_calculations,
    bench_market_operations,
    bench_concurrent_access,
    bench_numerical_stability,
    bench_utility_functions,
    bench_position_tracking,
    bench_memory_patterns,
    bench_serialization,
    bench_comparison_baseline
);

criterion_main!(benches);