//! Performance Benchmarks for Parasitic System
//! 
//! Comprehensive benchmarking suite to validate sub-millisecond performance
//! across all organism operations and system components.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use parasitic::*;
use parasitic::organisms::{PlatypusElectroreceptor, OctopusCamouflage};
use parasitic::traits::*;
use std::time::Duration;

/// Benchmark configuration
pub struct BenchmarkConfig {
    pub sample_size: usize,
    pub measurement_time: Duration,
    pub warm_up_time: Duration,
    pub target_time_ns: u64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            sample_size: 1000,
            measurement_time: Duration::from_secs(10),
            warm_up_time: Duration::from_secs(2),
            target_time_ns: 500_000, // 0.5ms target
        }
    }
}

/// Create benchmark market data
fn create_benchmark_market_data() -> MarketData {
    MarketData {
        symbol: "BTC_USD".to_string(),
        timestamp: chrono::Utc::now(),
        price: 50000.0,
        volume: 1000.0,
        volatility: 0.15,
        bid: 49975.0,
        ask: 50025.0,
        spread_percent: 0.1,
        market_cap: Some(1_000_000_000_000.0),
        liquidity_score: 0.8,
    }
}

/// Benchmark system initialization
fn benchmark_system_initialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("system_initialization");
    group.sample_size(100);
    
    group.bench_function("initialize", |b| {
        b.iter(|| {
            let result = black_box(initialize());
            assert!(result.is_ok());
            result.unwrap()
        })
    });
    
    group.finish();
}

/// Benchmark Platypus Electroreceptor operations
fn benchmark_platypus_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("platypus_operations");
    group.sample_size(1000);
    group.measurement_time(Duration::from_secs(10));
    
    let platypus = PlatypusElectroreceptor::new().expect("Failed to create Platypus");
    let market_data = create_benchmark_market_data();
    
    group.bench_function("creation", |b| {
        b.iter(|| {
            let result = black_box(PlatypusElectroreceptor::new());
            assert!(result.is_ok());
            result.unwrap()
        })
    });
    
    group.bench_function("wound_detection", |b| {
        b.iter(|| {
            let result = black_box(platypus.detect_wound(&market_data));
            assert!(result.is_ok());
            result.unwrap()
        })
    });
    
    group.bench_function("get_metrics", |b| {
        b.iter(|| {
            black_box(platypus.get_metrics())
        })
    });
    
    group.finish();
}

/// Benchmark Octopus Camouflage operations
fn benchmark_octopus_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("octopus_operations");
    group.sample_size(1000);
    group.measurement_time(Duration::from_secs(10));
    
    let octopus = OctopusCamouflage::new().expect("Failed to create Octopus");
    let market_data = create_benchmark_market_data();
    
    group.bench_function("creation", |b| {
        b.iter(|| {
            let result = black_box(OctopusCamouflage::new());
            assert!(result.is_ok());
            result.unwrap()
        })
    });
    
    group.bench_function("adaptation", |b| {
        b.iter(|| {
            let result = black_box(octopus.adapt(&market_data));
            assert!(result.is_ok());
            result.unwrap()
        })
    });
    
    group.bench_function("get_metrics", |b| {
        b.iter(|| {
            black_box(octopus.get_metrics())
        })
    });
    
    group.finish();
}

/// Benchmark multi-organism coordination
fn benchmark_coordination(c: &mut Criterion) {
    let mut group = c.benchmark_group("coordination");
    group.sample_size(500);
    group.measurement_time(Duration::from_secs(15));
    
    let platypus = PlatypusElectroreceptor::new().expect("Failed to create Platypus");
    let octopus = OctopusCamouflage::new().expect("Failed to create Octopus");
    let market_data = create_benchmark_market_data();
    
    group.bench_function("parallel_processing", |b| {
        b.iter(|| {
            let platypus_result = black_box(platypus.detect_wound(&market_data));
            let octopus_result = black_box(octopus.adapt(&market_data));
            assert!(platypus_result.is_ok());
            assert!(octopus_result.is_ok());
            (platypus_result.unwrap(), octopus_result.unwrap())
        })
    });
    
    group.finish();
}

/// Benchmark throughput under different loads
fn benchmark_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");
    
    let platypus = PlatypusElectroreceptor::new().expect("Failed to create Platypus");
    let market_data = create_benchmark_market_data();
    
    for operations in [100, 500, 1000, 5000].iter() {
        group.bench_with_input(
            BenchmarkId::new("wound_detection", operations),
            operations,
            |b, &operations| {
                b.iter(|| {
                    for _ in 0..operations {
                        let result = black_box(platypus.detect_wound(&market_data));
                        assert!(result.is_ok());
                    }
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory allocation patterns
fn benchmark_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");
    group.sample_size(200);
    
    group.bench_function("organism_creation_cleanup", |b| {
        b.iter(|| {
            let platypus = black_box(PlatypusElectroreceptor::new().unwrap());
            let octopus = black_box(OctopusCamouflage::new().unwrap());
            drop(platypus);
            drop(octopus);
        })
    });
    
    let platypus = PlatypusElectroreceptor::new().expect("Failed to create Platypus");
    let market_data = create_benchmark_market_data();
    
    group.bench_function("repeated_operations", |b| {
        b.iter(|| {
            for _ in 0..100 {
                let _result = black_box(platypus.detect_wound(&market_data));
            }
        })
    });
    
    group.finish();
}

/// Benchmark different market conditions
fn benchmark_market_conditions(c: &mut Criterion) {
    let mut group = c.benchmark_group("market_conditions");
    group.sample_size(500);
    
    let platypus = PlatypusElectroreceptor::new().expect("Failed to create Platypus");
    
    let market_scenarios = vec![
        ("normal", create_normal_market_data()),
        ("volatile", create_volatile_market_data()),
        ("low_liquidity", create_low_liquidity_market_data()),
        ("high_spread", create_high_spread_market_data()),
    ];
    
    for (scenario_name, market_data) in market_scenarios {
        group.bench_with_input(
            BenchmarkId::new("wound_detection", scenario_name),
            &market_data,
            |b, market_data| {
                b.iter(|| {
                    let result = black_box(platypus.detect_wound(market_data));
                    assert!(result.is_ok());
                    result.unwrap()
                })
            },
        );
    }
    
    group.finish();
}

/// Create different market condition test data
fn create_normal_market_data() -> MarketData {
    MarketData {
        symbol: "BTC_USD".to_string(),
        timestamp: chrono::Utc::now(),
        price: 50000.0,
        volume: 1000.0,
        volatility: 0.1,
        bid: 49990.0,
        ask: 50010.0,
        spread_percent: 0.04,
        market_cap: Some(1_000_000_000_000.0),
        liquidity_score: 0.9,
    }
}

fn create_volatile_market_data() -> MarketData {
    MarketData {
        symbol: "ETH_USD".to_string(),
        timestamp: chrono::Utc::now(),
        price: 3000.0,
        volume: 2000.0,
        volatility: 0.3,
        bid: 2970.0,
        ask: 3030.0,
        spread_percent: 2.0,
        market_cap: Some(500_000_000_000.0),
        liquidity_score: 0.7,
    }
}

fn create_low_liquidity_market_data() -> MarketData {
    MarketData {
        symbol: "ALT_USD".to_string(),
        timestamp: chrono::Utc::now(),
        price: 100.0,
        volume: 10.0,
        volatility: 0.4,
        bid: 90.0,
        ask: 110.0,
        spread_percent: 20.0,
        market_cap: Some(1_000_000.0),
        liquidity_score: 0.1,
    }
}

fn create_high_spread_market_data() -> MarketData {
    MarketData {
        symbol: "ILLIQUID_USD".to_string(),
        timestamp: chrono::Utc::now(),
        price: 1000.0,
        volume: 5.0,
        volatility: 0.5,
        bid: 900.0,
        ask: 1100.0,
        spread_percent: 22.0,
        market_cap: Some(500_000.0),
        liquidity_score: 0.05,
    }
}

/// Benchmark SIMD operations if available
fn benchmark_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_operations");
    group.sample_size(1000);
    
    // This would test SIMD-specific operations if implemented
    group.bench_function("simd_wound_detection", |b| {
        let platypus = PlatypusElectroreceptor::new().expect("Failed to create Platypus");
        let market_data = create_benchmark_market_data();
        
        b.iter(|| {
            // In real implementation, this would specifically test SIMD code paths
            let result = black_box(platypus.detect_wound(&market_data));
            assert!(result.is_ok());
            result.unwrap()
        })
    });
    
    group.finish();
}

/// Custom measurement for sub-millisecond validation
fn validate_sub_millisecond_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("sub_millisecond_validation");
    group.sample_size(1000);
    
    let platypus = PlatypusElectroreceptor::new().expect("Failed to create Platypus");
    let octopus = OctopusCamouflage::new().expect("Failed to create Octopus");
    let market_data = create_benchmark_market_data();
    
    group.bench_function("platypus_sub_ms", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let result = black_box(platypus.detect_wound(&market_data));
                assert!(result.is_ok());
            }
            let duration = start.elapsed();
            
            // Validate each operation was sub-millisecond
            let avg_time_per_op = duration / iters as u32;
            assert!(avg_time_per_op.as_nanos() < 1_000_000, 
                "Average operation time {}ns exceeds 1ms threshold", 
                avg_time_per_op.as_nanos());
            
            duration
        })
    });
    
    group.bench_function("octopus_sub_ms", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let result = black_box(octopus.adapt(&market_data));
                assert!(result.is_ok());
            }
            let duration = start.elapsed();
            
            // Validate each operation was sub-millisecond
            let avg_time_per_op = duration / iters as u32;
            assert!(avg_time_per_op.as_nanos() < 1_000_000, 
                "Average operation time {}ns exceeds 1ms threshold", 
                avg_time_per_op.as_nanos());
            
            duration
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_system_initialization,
    benchmark_platypus_operations,
    benchmark_octopus_operations,
    benchmark_coordination,
    benchmark_throughput,
    benchmark_memory_patterns,
    benchmark_market_conditions,
    benchmark_simd_operations,
    validate_sub_millisecond_performance
);

criterion_main!(benches);