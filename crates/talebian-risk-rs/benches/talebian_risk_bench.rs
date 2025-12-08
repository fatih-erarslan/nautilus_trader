//! # Performance Benchmarks for Talebian Risk Management
//!
//! Comprehensive benchmarks measuring the performance of the aggressive
//! Machiavellian risk management system under various load conditions.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use talebian_risk_rs::{MacchiavelianConfig, MarketData, TalebianRiskEngine, WhaleDirection};

fn create_benchmark_market_data(i: usize) -> MarketData {
    MarketData {
        timestamp: 1640995200 + i as i64,
        price: 50000.0 + (i as f64 * 100.0),
        volume: 1000.0 + (i as f64 * 10.0),
        bid: 49995.0 + (i as f64 * 100.0),
        ask: 50005.0 + (i as f64 * 100.0),
        bid_volume: 500.0 + (i as f64 * 5.0),
        ask_volume: 500.0 + (i as f64 * 5.0),
        volatility: 0.02 + (i as f64 * 0.001),
        returns: vec![0.01, -0.005, 0.015, -0.008, 0.02],
        volume_history: vec![900.0, 1100.0, 950.0, 1050.0, 1000.0],
    }
}

fn create_whale_market_data(i: usize) -> MarketData {
    MarketData {
        timestamp: 1640995200 + i as i64,
        price: 50000.0,
        volume: 5000.0, // High whale volume
        bid: 49990.0,
        ask: 50010.0,
        bid_volume: 2000.0, // Large whale orders
        ask_volume: 500.0,
        volatility: 0.04,
        returns: vec![0.02, 0.025, 0.015, 0.03, 0.01],
        volume_history: vec![1000.0, 1200.0, 1100.0, 1050.0, 1000.0],
    }
}

fn bench_single_risk_assessment(c: &mut Criterion) {
    let config = MacchiavelianConfig::aggressive_defaults();
    let mut engine = TalebianRiskEngine::new(config);
    let market_data = create_benchmark_market_data(0);

    c.bench_function("single_risk_assessment", |b| {
        b.iter(|| {
            black_box(engine.assess_risk(black_box(&market_data)).unwrap());
        });
    });
}

fn bench_bulk_risk_assessment(c: &mut Criterion) {
    let config = MacchiavelianConfig::aggressive_defaults();
    let mut engine = TalebianRiskEngine::new(config);

    let mut group = c.benchmark_group("bulk_risk_assessment");

    for size in [10, 50, 100, 500, 1000].iter() {
        let market_data_batch: Vec<MarketData> =
            (0..*size).map(create_benchmark_market_data).collect();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("batch_size", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    engine
                        .assess_bulk_risks(black_box(&market_data_batch))
                        .unwrap(),
                );
            });
        });
    }
    group.finish();
}

fn bench_whale_detection_performance(c: &mut Criterion) {
    let config = MacchiavelianConfig::aggressive_defaults();
    let mut engine = TalebianRiskEngine::new(config);

    // Test normal vs whale data performance
    let normal_data = create_benchmark_market_data(0);
    let whale_data = create_whale_market_data(0);

    let mut group = c.benchmark_group("whale_detection");

    group.bench_function("normal_market", |b| {
        b.iter(|| {
            black_box(engine.assess_risk(black_box(&normal_data)).unwrap());
        });
    });

    group.bench_function("whale_market", |b| {
        b.iter(|| {
            black_box(engine.assess_risk(black_box(&whale_data)).unwrap());
        });
    });

    group.finish();
}

fn bench_configuration_variants(c: &mut Criterion) {
    let aggressive_config = MacchiavelianConfig::aggressive_defaults();
    let conservative_config = MacchiavelianConfig::conservative_baseline();
    let extreme_config = MacchiavelianConfig::extreme_machiavellian();

    let market_data = create_benchmark_market_data(0);

    let mut group = c.benchmark_group("configuration_variants");

    group.bench_function("aggressive", |b| {
        let mut engine = TalebianRiskEngine::new(aggressive_config.clone());
        b.iter(|| {
            black_box(engine.assess_risk(black_box(&market_data)).unwrap());
        });
    });

    group.bench_function("conservative", |b| {
        let mut engine = TalebianRiskEngine::new(conservative_config.clone());
        b.iter(|| {
            black_box(engine.assess_risk(black_box(&market_data)).unwrap());
        });
    });

    group.bench_function("extreme", |b| {
        let mut engine = TalebianRiskEngine::new(extreme_config.clone());
        b.iter(|| {
            black_box(engine.assess_risk(black_box(&market_data)).unwrap());
        });
    });

    group.finish();
}

fn bench_recommendation_generation(c: &mut Criterion) {
    let config = MacchiavelianConfig::aggressive_defaults();
    let mut engine = TalebianRiskEngine::new(config);
    let market_data = create_benchmark_market_data(0);

    c.bench_function("recommendation_generation", |b| {
        b.iter(|| {
            black_box(
                engine
                    .generate_recommendations(black_box(&market_data))
                    .unwrap(),
            );
        });
    });
}

fn bench_learning_updates(c: &mut Criterion) {
    let config = MacchiavelianConfig::aggressive_defaults();
    let mut engine = TalebianRiskEngine::new(config);

    c.bench_function("learning_updates", |b| {
        b.iter(|| {
            black_box(
                engine
                    .record_trade_outcome(black_box(0.02), black_box(true), black_box(0.8))
                    .unwrap(),
            );
        });
    });
}

fn bench_memory_efficiency(c: &mut Criterion) {
    let config = MacchiavelianConfig::aggressive_defaults();

    let mut group = c.benchmark_group("memory_efficiency");

    // Test memory usage with different history sizes
    for history_size in [100, 500, 1000, 5000].iter() {
        group.bench_with_input(
            BenchmarkId::new("history_size", history_size),
            history_size,
            |b, &size| {
                let mut engine = TalebianRiskEngine::new(config.clone());

                // Pre-populate with history
                for i in 0..size {
                    let market_data = create_benchmark_market_data(i);
                    engine.assess_risk(&market_data).unwrap();
                }

                let market_data = create_benchmark_market_data(size);

                b.iter(|| {
                    black_box(engine.assess_risk(black_box(&market_data)).unwrap());
                });
            },
        );
    }
    group.finish();
}

fn bench_high_frequency_updates(c: &mut Criterion) {
    let config = MacchiavelianConfig::aggressive_defaults();
    let mut engine = TalebianRiskEngine::new(config);

    let market_data_stream: Vec<MarketData> = (0..1000).map(create_benchmark_market_data).collect();

    c.bench_function("high_frequency_stream", |b| {
        b.iter(|| {
            for data in &market_data_stream {
                black_box(engine.assess_risk(black_box(data)).unwrap());
            }
        });
    });
}

fn bench_concurrent_access(c: &mut Criterion) {
    use std::sync::Arc;
    use std::thread;

    let config = MacchiavelianConfig::aggressive_defaults();

    c.bench_function("concurrent_engines", |b| {
        b.iter(|| {
            let handles: Vec<_> = (0..4)
                .map(|i| {
                    let config = config.clone();
                    thread::spawn(move || {
                        let mut engine = TalebianRiskEngine::new(config);
                        let market_data = create_benchmark_market_data(i);
                        engine.assess_risk(&market_data)
                    })
                })
                .collect();

            for handle in handles {
                black_box(handle.join().unwrap().unwrap());
            }
        });
    });
}

#[cfg(feature = "simd")]
fn bench_simd_operations(c: &mut Criterion) {
    use talebian_risk_rs::performance::SimdMath;

    let mut group = c.benchmark_group("simd_operations");

    for size in [100, 500, 1000, 5000].iter() {
        let expected_returns: Vec<f64> = (0..*size).map(|i| 0.01 + i as f64 * 0.001).collect();
        let variances: Vec<f64> = (0..*size).map(|i| 0.001 + i as f64 * 0.0001).collect();
        let whale_multipliers: Vec<f64> = (0..*size).map(|i| 1.0 + (i % 2) as f64 * 0.5).collect();

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("kelly_simd", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    SimdMath::kelly_fraction_simd_x4(
                        black_box(&expected_returns),
                        black_box(&variances),
                        black_box(&whale_multipliers),
                    )
                    .unwrap(),
                );
            });
        });

        // Volatility calculation benchmark
        let returns: Vec<f64> = (0..*size).map(|i| (i as f64 * 0.01) % 0.1 - 0.05).collect();

        group.bench_with_input(BenchmarkId::new("volatility_simd", size), size, |b, _| {
            b.iter(|| {
                black_box(SimdMath::volatility_simd_x8(black_box(&returns)).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_cache_performance(c: &mut Criterion) {
    use talebian_risk_rs::performance::CalculationCache;

    let mut cache = CalculationCache::new(1000);

    let mut group = c.benchmark_group("cache_performance");

    // Benchmark cache hits
    group.bench_function("cache_hit", |b| {
        cache.set_kelly(12345, 0.5);
        b.iter(|| {
            black_box(cache.get_kelly(black_box(12345)));
        });
    });

    // Benchmark cache misses
    group.bench_function("cache_miss", |b| {
        b.iter(|| {
            black_box(cache.get_kelly(black_box(99999)));
        });
    });

    // Benchmark cache updates
    group.bench_function("cache_update", |b| {
        let mut key = 0u64;
        b.iter(|| {
            cache.set_kelly(black_box(key), black_box(0.5));
            key += 1;
        });
    });

    group.finish();
}

fn bench_error_handling_overhead(c: &mut Criterion) {
    let config = MacchiavelianConfig::aggressive_defaults();
    let mut engine = TalebianRiskEngine::new(config);

    let valid_data = create_benchmark_market_data(0);
    let invalid_data = MarketData {
        price: -100.0, // Invalid negative price
        ..valid_data.clone()
    };

    let mut group = c.benchmark_group("error_handling");

    group.bench_function("valid_data", |b| {
        b.iter(|| {
            black_box(engine.assess_risk(black_box(&valid_data)).unwrap());
        });
    });

    group.bench_function("invalid_data", |b| {
        b.iter(|| {
            // This might succeed with data sanitization or fail gracefully
            let _ = black_box(engine.assess_risk(black_box(&invalid_data)));
        });
    });

    group.finish();
}

fn bench_real_world_scenario(c: &mut Criterion) {
    let config = MacchiavelianConfig::aggressive_defaults();
    let mut engine = TalebianRiskEngine::new(config);

    // Simulate a realistic trading day with mixed scenarios
    let market_scenarios = vec![
        create_benchmark_market_data(0), // Normal market
        create_whale_market_data(1),     // Whale activity
        create_benchmark_market_data(2), // Normal market
        create_whale_market_data(3),     // More whale activity
        MarketData {
            // High volatility
            volatility: 0.08,
            ..create_benchmark_market_data(4)
        },
        create_benchmark_market_data(5), // Back to normal
    ];

    c.bench_function("real_world_scenario", |b| {
        b.iter(|| {
            for data in &market_scenarios {
                let assessment = black_box(engine.assess_risk(black_box(data)).unwrap());

                // Simulate trade execution decision
                if assessment.recommended_position_size > 0.1 {
                    // Simulate recording trade outcome
                    black_box(
                        engine
                            .record_trade_outcome(
                                0.015,
                                assessment.whale_detection.is_whale_detected,
                                0.7,
                            )
                            .unwrap(),
                    );
                }
            }
        });
    });
}

fn bench_latency_requirements(c: &mut Criterion) {
    let config = MacchiavelianConfig::aggressive_defaults();
    let mut engine = TalebianRiskEngine::new(config);
    let market_data = create_benchmark_market_data(0);

    // Test sub-millisecond latency requirement
    let mut group = c.benchmark_group("latency_requirements");
    group.measurement_time(std::time::Duration::from_secs(10));

    group.bench_function("sub_millisecond", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                black_box(engine.assess_risk(black_box(&market_data)).unwrap());
            }
            start.elapsed()
        });
    });

    group.finish();
}

// Conditional SIMD benchmarks
#[cfg(feature = "simd")]
criterion_group!(simd_benches, bench_simd_operations,);

#[cfg(not(feature = "simd"))]
criterion_group!(simd_benches,);

criterion_group!(
    benches,
    bench_single_risk_assessment,
    bench_bulk_risk_assessment,
    bench_whale_detection_performance,
    bench_configuration_variants,
    bench_recommendation_generation,
    bench_learning_updates,
    bench_memory_efficiency,
    bench_high_frequency_updates,
    bench_concurrent_access,
    bench_cache_performance,
    bench_error_handling_overhead,
    bench_real_world_scenario,
    bench_latency_requirements,
);

criterion_main!(benches, simd_benches);
