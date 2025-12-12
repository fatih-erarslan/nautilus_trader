// Stealth Execution Benchmark for Whale Hunting Strategy
// Copyright (c) 2025 TENGRI Trading Swarm

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use whale_hunting_strategy::{
    WhaleHunter, StealthExecutor, WhaleDetector, OrderFragmentation, MarketImpactAnalyzer,
    VolumeProfileAnalyzer, TimingOptimizer, ExecutionAlgorithm, WhaleSignature,
    StealthParameters, ExecutionStrategy, LiquidityAnalyzer
};
use std::time::Instant;

fn create_market_data(size: usize) -> Vec<MarketData> {
    (0..size).map(|i| MarketData {
        timestamp: chrono::Utc::now(),
        symbol: "BTCUSDT".to_string(),
        price: 50000.0 + (i as f64 * 0.1),
        volume: 1000.0 + (i as f64 * 100.0),
        bid: 49995.0,
        ask: 50005.0,
        bid_size: 500.0 + (i as f64 * 10.0),
        ask_size: 500.0 + (i as f64 * 10.0),
        large_orders: i % 10 == 0,
        unusual_volume: i % 15 == 0,
        price_impact: 0.001 * (i as f64 / 100.0),
    }).collect()
}

use whale_hunting_strategy::MarketData;

fn benchmark_whale_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("whale_detection");
    
    for size in [100, 500, 1000, 5000].iter() {
        let market_data = create_market_data(*size);
        let detector = WhaleDetector::new();
        
        group.bench_with_input(BenchmarkId::new("detect_whales", size), size, |b, _| {
            b.iter(|| {
                detector.detect_whale_activity(&market_data)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("analyze_signatures", size), size, |b, _| {
            b.iter(|| {
                detector.analyze_whale_signatures(&market_data)
            })
        });
    }
    group.finish();
}

fn benchmark_stealth_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("stealth_execution");
    
    let market_data = create_market_data(1000);
    let executor = StealthExecutor::new();
    
    for order_size in [10000.0, 50000.0, 100000.0, 500000.0].iter() {
        group.bench_with_input(BenchmarkId::new("execute_stealth_order", format!("{:.0}", order_size)), order_size, |b, &size| {
            b.iter(|| {
                executor.execute_stealth_order(&market_data, size)
            })
        });
    }
    
    group.bench_function("optimize_execution_timing", |b| {
        b.iter(|| {
            executor.optimize_execution_timing(&market_data)
        })
    });
    
    group.bench_function("calculate_slippage", |b| {
        b.iter(|| {
            executor.calculate_expected_slippage(&market_data, 100000.0)
        })
    });
    
    group.finish();
}

fn benchmark_order_fragmentation(c: &mut Criterion) {
    let mut group = c.benchmark_group("order_fragmentation");
    
    let market_data = create_market_data(1000);
    let fragmenter = OrderFragmentation::new();
    
    for total_size in [50000.0, 100000.0, 500000.0, 1000000.0].iter() {
        group.bench_with_input(BenchmarkId::new("fragment_order", format!("{:.0}", total_size)), total_size, |b, &size| {
            b.iter(|| {
                fragmenter.fragment_large_order(&market_data, size)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("optimize_fragments", format!("{:.0}", total_size)), total_size, |b, &size| {
            b.iter(|| {
                fragmenter.optimize_fragment_sizes(&market_data, size)
            })
        });
    }
    
    group.finish();
}

fn benchmark_market_impact_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("market_impact_analysis");
    
    let market_data = create_market_data(1000);
    let analyzer = MarketImpactAnalyzer::new();
    
    for order_size in [10000.0, 50000.0, 100000.0, 250000.0].iter() {
        group.bench_with_input(BenchmarkId::new("predict_impact", format!("{:.0}", order_size)), order_size, |b, &size| {
            b.iter(|| {
                analyzer.predict_market_impact(&market_data, size)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("calculate_permanent_impact", format!("{:.0}", order_size)), order_size, |b, &size| {
            b.iter(|| {
                analyzer.calculate_permanent_impact(&market_data, size)
            })
        });
    }
    
    group.bench_function("analyze_liquidity_curve", |b| {
        b.iter(|| {
            analyzer.analyze_liquidity_curve(&market_data)
        })
    });
    
    group.finish();
}

fn benchmark_volume_profile_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("volume_profile_analysis");
    
    let market_data = create_market_data(2000);
    let analyzer = VolumeProfileAnalyzer::new();
    
    group.bench_function("analyze_volume_profile", |b| {
        b.iter(|| {
            analyzer.analyze_volume_profile(&market_data)
        })
    });
    
    group.bench_function("identify_poc", |b| {
        b.iter(|| {
            analyzer.identify_point_of_control(&market_data)
        })
    });
    
    group.bench_function("calculate_vwap", |b| {
        b.iter(|| {
            analyzer.calculate_volume_weighted_average_price(&market_data)
        })
    });
    
    group.bench_function("detect_volume_anomalies", |b| {
        b.iter(|| {
            analyzer.detect_volume_anomalies(&market_data)
        })
    });
    
    group.finish();
}

fn benchmark_timing_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("timing_optimization");
    
    let market_data = create_market_data(1500);
    let optimizer = TimingOptimizer::new();
    
    group.bench_function("optimize_entry_timing", |b| {
        b.iter(|| {
            optimizer.optimize_entry_timing(&market_data)
        })
    });
    
    group.bench_function("predict_optimal_windows", |b| {
        b.iter(|| {
            optimizer.predict_optimal_execution_windows(&market_data)
        })
    });
    
    group.bench_function("analyze_market_microstructure", |b| {
        b.iter(|| {
            optimizer.analyze_market_microstructure(&market_data)
        })
    });
    
    group.finish();
}

fn benchmark_execution_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("execution_algorithms");
    
    let market_data = create_market_data(1000);
    
    let algorithms = vec![
        ("TWAP", ExecutionAlgorithm::TWAP),
        ("VWAP", ExecutionAlgorithm::VWAP),
        ("POV", ExecutionAlgorithm::ParticipationOfVolume),
        ("Implementation Shortfall", ExecutionAlgorithm::ImplementationShortfall),
        ("Stealth", ExecutionAlgorithm::Stealth),
    ];
    
    for (name, algorithm) in algorithms.iter() {
        group.bench_with_input(BenchmarkId::new("execute_algorithm", name), algorithm, |b, algo| {
            b.iter(|| {
                algo.execute(&market_data, 100000.0)
            })
        });
    }
    
    group.finish();
}

fn benchmark_liquidity_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("liquidity_analysis");
    
    let market_data = create_market_data(1000);
    let analyzer = LiquidityAnalyzer::new();
    
    group.bench_function("analyze_order_book_depth", |b| {
        b.iter(|| {
            analyzer.analyze_order_book_depth(&market_data)
        })
    });
    
    group.bench_function("calculate_spread_cost", |b| {
        b.iter(|| {
            analyzer.calculate_spread_cost(&market_data)
        })
    });
    
    group.bench_function("estimate_available_liquidity", |b| {
        b.iter(|| {
            analyzer.estimate_available_liquidity(&market_data)
        })
    });
    
    group.bench_function("predict_liquidity_recovery", |b| {
        b.iter(|| {
            analyzer.predict_liquidity_recovery_time(&market_data)
        })
    });
    
    group.finish();
}

fn benchmark_whale_hunting_strategy(c: &mut Criterion) {
    let mut group = c.benchmark_group("whale_hunting_strategy");
    
    let market_data = create_market_data(2000);
    let hunter = WhaleHunter::new();
    
    group.bench_function("full_whale_hunt", |b| {
        b.iter(|| {
            hunter.execute_whale_hunting_strategy(&market_data, 500000.0)
        })
    });
    
    group.bench_function("adaptive_strategy", |b| {
        b.iter(|| {
            hunter.adapt_strategy_to_market_conditions(&market_data)
        })
    });
    
    group.bench_function("risk_management", |b| {
        b.iter(|| {
            hunter.apply_risk_management_controls(&market_data, 500000.0)
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_whale_detection,
    benchmark_stealth_execution,
    benchmark_order_fragmentation,
    benchmark_market_impact_analysis,
    benchmark_volume_profile_analysis,
    benchmark_timing_optimization,
    benchmark_execution_algorithms,
    benchmark_liquidity_analysis,
    benchmark_whale_hunting_strategy
);
criterion_main!(benches);