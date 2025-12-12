//! Comprehensive benchmarks for whale defense performance validation
//! 
//! Validates sub-microsecond performance targets across all components.

use whale_defense_core::{
    WhaleDefenseEngine, QuantumGameTheoryEngine, SteganographicOrderManager,
    MarketOrder, ThreatLevel, DefenseConfig, timing::Timestamp,
    config::*,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;

/// Benchmark whale detection performance
/// Target: <500 nanoseconds
fn bench_whale_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("whale_detection");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10000);
    
    // Setup test data
    let config = DefenseConfig::default();
    let engine = unsafe { WhaleDefenseEngine::new(config).unwrap() };
    unsafe { engine.start().unwrap() };
    
    let test_orders = vec![
        MarketOrder::new(100.0, 1000.0, 1, 1, 0),   // Small order
        MarketOrder::new(100.0, 10000.0, 1, 1, 0),  // Medium order
        MarketOrder::new(100.0, 100000.0, 1, 1, 0), // Large order (whale)
        MarketOrder::new(100.0, 1000000.0, 1, 1, 0), // Huge order (mega whale)
    ];
    
    for (i, order) in test_orders.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("detection", format!("order_{}", i)),
            order,
            |b, order| {
                b.iter(|| unsafe {
                    let result = engine.process_market_order(black_box(order.clone()));
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark quantum game theory strategy calculation
/// Target: <100 nanoseconds
fn bench_quantum_strategy(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantum_strategy");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50000);
    
    let engine = unsafe { QuantumGameTheoryEngine::new().unwrap() };
    
    let test_strategies = vec![
        ([0.5, 0.3, 0.1, 0.1], 1000000.0, ThreatLevel::Low),
        ([0.3, 0.4, 0.2, 0.1], 5000000.0, ThreatLevel::Medium),
        ([0.7, 0.2, 0.05, 0.05], 10000000.0, ThreatLevel::High),
        ([0.9, 0.05, 0.03, 0.02], 50000000.0, ThreatLevel::Critical),
    ];
    
    for (i, (whale_strategy, size, threat)) in test_strategies.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("strategy_calc", format!("threat_{:?}", threat)),
            &(whale_strategy, size, threat),
            |b, (strategy, size, threat)| {
                b.iter(|| unsafe {
                    let result = engine.calculate_optimal_strategy(
                        black_box(*strategy),
                        black_box(**size),
                        black_box(**threat),
                    );
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark steganographic order generation
/// Target: <100 nanoseconds
fn bench_steganographic_orders(c: &mut Criterion) {
    let mut group = c.benchmark_group("steganographic_orders");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(25000);
    
    let manager = unsafe { SteganographicOrderManager::new().unwrap() };
    
    let stealth_levels = vec![0, 1, 2, 3];
    let threat_levels = vec![
        ThreatLevel::Low,
        ThreatLevel::Medium,
        ThreatLevel::High,
        ThreatLevel::Critical,
    ];
    
    for stealth_level in stealth_levels {
        for threat_level in &threat_levels {
            group.bench_with_input(
                BenchmarkId::new("order_generation", format!("stealth_{}_threat_{:?}", stealth_level, threat_level)),
                &(stealth_level, threat_level),
                |b, (stealth, threat)| {
                    b.iter(|| unsafe {
                        let mut manager_mut = &manager;
                        let result = manager_mut.create_steganographic_order(
                            black_box(100.0),
                            black_box(1000.0),
                            black_box(*stealth),
                            black_box(**threat),
                        );
                        black_box(result)
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark end-to-end whale defense performance
/// Target: <1 microsecond total
fn bench_end_to_end_defense(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end_defense");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(5000);
    
    let config = DefenseConfig::default();
    let engine = unsafe { WhaleDefenseEngine::new(config).unwrap() };
    unsafe { engine.start().unwrap() };
    
    // Test various whale scenarios
    let whale_scenarios = vec![
        ("small_whale", MarketOrder::new(100.0, 50000.0, 1, 1, 0)),
        ("medium_whale", MarketOrder::new(100.0, 500000.0, 1, 1, 0)),
        ("large_whale", MarketOrder::new(100.0, 5000000.0, 1, 1, 0)),
        ("mega_whale", MarketOrder::new(100.0, 50000000.0, 1, 1, 0)),
    ];
    
    for (scenario_name, order) in whale_scenarios {
        group.bench_with_input(
            BenchmarkId::new("full_defense", scenario_name),
            &order,
            |b, order| {
                b.iter(|| unsafe {
                    let start_time = Timestamp::now();
                    let result = engine.process_market_order(black_box(order.clone()));
                    let end_time = Timestamp::now();
                    
                    let elapsed_ns = end_time.as_tsc() - start_time.as_tsc();
                    
                    // Validate performance target
                    if elapsed_ns > TARGET_DETECTION_LATENCY_NS + TARGET_DEFENSE_EXECUTION_NS {
                        panic!("Performance target exceeded: {}ns > {}ns", 
                               elapsed_ns, TARGET_DETECTION_LATENCY_NS + TARGET_DEFENSE_EXECUTION_NS);
                    }
                    
                    black_box((result, elapsed_ns))
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark lock-free data structures performance
fn bench_lockfree_structures(c: &mut Criterion) {
    use whale_defense_core::lockfree::{LockFreeRingBuffer, LockFreeQueue, LockFreeStack};
    
    let mut group = c.benchmark_group("lockfree_structures");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(100000);
    
    // Ring buffer benchmarks
    group.bench_function("ring_buffer_write_read", |b| {
        let mut buffer = unsafe { LockFreeRingBuffer::<u64>::new(1024).unwrap() };
        b.iter(|| unsafe {
            let value = black_box(42u64);
            buffer.try_write(value).unwrap();
            let result = buffer.try_read().unwrap();
            black_box(result)
        });
        unsafe { buffer.destroy() };
    });
    
    // Queue benchmarks
    group.bench_function("queue_enqueue_dequeue", |b| {
        let queue = unsafe { LockFreeQueue::<u64>::new() };
        b.iter(|| unsafe {
            let value = black_box(42u64);
            queue.enqueue(value).unwrap();
            let result = queue.dequeue().unwrap();
            black_box(result)
        });
    });
    
    // Stack benchmarks
    group.bench_function("stack_push_pop", |b| {
        let stack = unsafe { LockFreeStack::<u64>::new() };
        b.iter(|| unsafe {
            let value = black_box(42u64);
            stack.push(value).unwrap();
            let result = stack.pop().unwrap();
            black_box(result)
        });
    });
    
    group.finish();
}

/// Benchmark SIMD operations performance
#[cfg(feature = "simd")]
fn bench_simd_operations(c: &mut Criterion) {
    use whale_defense_core::simd::*;
    
    let mut group = c.benchmark_group("simd_operations");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50000);
    
    let dispatcher = SimdDispatcher::new();
    
    // Whale pattern matching
    group.bench_function("whale_pattern_match_simd", |b| {
        let prices = vec![100.0; 32];
        let volumes = vec![1000.0; 32];
        let thresholds = [500.0, 99.0, 50000.0, 0.05];
        
        b.iter(|| {
            let result = dispatcher.dispatch_whale_pattern_match(
                black_box(&prices),
                black_box(&volumes),
                black_box(&thresholds),
            );
            black_box(result)
        });
    });
    
    // Volume analysis
    group.bench_function("volume_analysis_simd", |b| {
        let volumes = vec![1000.0; 64];
        
        b.iter(|| unsafe {
            let result = simd_volume_analysis(black_box(&volumes), black_box(16));
            black_box(result)
        });
    });
    
    // Moving average
    group.bench_function("moving_average_simd", |b| {
        let data = vec![100.0; 128];
        
        b.iter(|| unsafe {
            let result = simd_moving_average(black_box(&data), black_box(8));
            black_box(result)
        });
    });
    
    group.finish();
}

/// Benchmark memory allocation and cache performance
fn bench_memory_performance(c: &mut Criterion) {
    use whale_defense_core::cache::*;
    
    let mut group = c.benchmark_group("memory_performance");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10000);
    
    // Cache warm-up benchmark
    group.bench_function("cache_warmup", |b| {
        b.iter(|| {
            warm_up_caches();
            black_box(())
        });
    });
    
    // Cache-optimized memcpy
    group.bench_function("cache_optimized_memcpy", |b| {
        let src = vec![0u8; 1024];
        let mut dst = vec![0u8; 1024];
        
        b.iter(|| unsafe {
            cache_optimized_memcpy(
                black_box(dst.as_mut_ptr()),
                black_box(src.as_ptr()),
                black_box(1024),
            );
            black_box(&dst)
        });
    });
    
    // Prefetch operations
    group.bench_function("prefetch_operations", |b| {
        let data = vec![0u64; 256];
        
        b.iter(|| {
            for chunk in data.chunks(8) {
                prefetch_data(black_box(chunk.as_ptr()), black_box(Locality::High));
            }
            black_box(&data)
        });
    });
    
    group.finish();
}

/// Benchmark timing precision and overhead
fn bench_timing_performance(c: &mut Criterion) {
    use whale_defense_core::timing::*;
    
    let mut group = c.benchmark_group("timing_performance");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(1000000);
    
    // Timestamp creation overhead
    group.bench_function("timestamp_now", |b| {
        b.iter(|| {
            let ts = Timestamp::now();
            black_box(ts)
        });
    });
    
    // Precise timestamp overhead
    group.bench_function("timestamp_now_precise", |b| {
        b.iter(|| {
            let ts = Timestamp::now_precise();
            black_box(ts)
        });
    });
    
    // Elapsed time calculation
    group.bench_function("elapsed_calculation", |b| {
        let start = Timestamp::now();
        b.iter(|| {
            let elapsed = start.elapsed_nanos();
            black_box(elapsed)
        });
    });
    
    group.finish();
}

/// Comprehensive performance validation test
/// 
/// This test validates that all components meet their performance targets
/// under realistic load conditions.
fn validate_performance_targets(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_validation");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(1000);
    
    let config = DefenseConfig::default();
    let engine = unsafe { WhaleDefenseEngine::new(config).unwrap() };
    unsafe { engine.start().unwrap() };
    
    // Simulate realistic trading load
    group.bench_function("realistic_trading_load", |b| {
        b.iter(|| {
            let start_time = Timestamp::now();
            
            // Process 100 market orders in sequence (realistic burst)
            for i in 0..100 {
                let order = MarketOrder::new(
                    100.0 + (i as f64 * 0.01),  // Slight price variation
                    1000.0 + (i as f64 * 100.0), // Volume variation
                    1,
                    (i % 10) as u16,
                    0,
                );
                
                let result = unsafe { engine.process_market_order(black_box(order)) };
                black_box(result);
            }
            
            let total_time = start_time.elapsed_nanos();
            let avg_time_per_order = total_time / 100;
            
            // Validate average time per order is within targets
            assert!(
                avg_time_per_order <= TARGET_DETECTION_LATENCY_NS + TARGET_DEFENSE_EXECUTION_NS,
                "Average processing time {} ns exceeds target {} ns",
                avg_time_per_order,
                TARGET_DETECTION_LATENCY_NS + TARGET_DEFENSE_EXECUTION_NS
            );
            
            black_box((total_time, avg_time_per_order))
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_whale_detection,
    bench_quantum_strategy,
    bench_steganographic_orders,
    bench_end_to_end_defense,
    bench_lockfree_structures,
    #[cfg(feature = "simd")]
    bench_simd_operations,
    bench_memory_performance,
    bench_timing_performance,
    validate_performance_targets,
);

criterion_main!(benches);