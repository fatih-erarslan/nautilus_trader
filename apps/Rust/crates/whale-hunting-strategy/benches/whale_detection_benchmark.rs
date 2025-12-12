// Whale Detection Benchmark for Whale Hunting Strategy
// Copyright (c) 2025 TENGRI Trading Swarm

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use whale_hunting_strategy::{
    WhaleDetector, WhaleSignature, WhaleActivityAnalyzer, LargeOrderDetector,
    MarketImpactAnalyzer, VolumeAnomalyDetector, PriceMovementAnalyzer,
    OrderFlowAnalyzer, WhaleClassification, WhaleTracking, MarketData
};

fn create_whale_market_data(size: usize, whale_activity: bool) -> Vec<MarketData> {
    (0..size).map(|i| MarketData {
        timestamp: chrono::Utc::now(),
        symbol: "BTCUSDT".to_string(),
        price: 50000.0 + (i as f64 * if whale_activity { 5.0 } else { 0.1 }),
        volume: if whale_activity && i % 20 == 0 { 50000.0 } else { 1000.0 },
        bid: 49995.0,
        ask: 50005.0,
        bid_size: if whale_activity && i % 15 == 0 { 5000.0 } else { 500.0 },
        ask_size: if whale_activity && i % 15 == 0 { 5000.0 } else { 500.0 },
        large_orders: whale_activity && i % 10 == 0,
        unusual_volume: whale_activity && i % 12 == 0,
        price_impact: if whale_activity { 0.01 } else { 0.001 },
        order_flow_imbalance: if whale_activity { 0.3 } else { 0.05 },
        market_depth: if whale_activity { 0.5 } else { 1.0 },
        volatility: if whale_activity { 0.05 } else { 0.02 },
        momentum: if whale_activity { 0.1 } else { 0.01 },
    }).collect()
}

fn benchmark_whale_detection_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("whale_detection_algorithms");
    
    for size in [100, 500, 1000, 5000].iter() {
        let whale_data = create_whale_market_data(*size, true);
        let normal_data = create_whale_market_data(*size, false);
        let detector = WhaleDetector::new();
        
        group.bench_with_input(BenchmarkId::new("detect_whale_activity", size), size, |b, _| {
            b.iter(|| {
                detector.detect_whale_activity(&whale_data)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("detect_normal_activity", size), size, |b, _| {
            b.iter(|| {
                detector.detect_whale_activity(&normal_data)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("classify_whale_type", size), size, |b, _| {
            b.iter(|| {
                detector.classify_whale_type(&whale_data)
            })
        });
    }
    group.finish();
}

fn benchmark_whale_signature_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("whale_signature_analysis");
    
    let whale_data = create_whale_market_data(2000, true);
    let signature_analyzer = WhaleSignature::new();
    
    group.bench_function("extract_whale_signatures", |b| {
        b.iter(|| {
            signature_analyzer.extract_whale_signatures(&whale_data)
        })
    });
    
    group.bench_function("analyze_signature_patterns", |b| {
        b.iter(|| {
            signature_analyzer.analyze_signature_patterns(&whale_data)
        })
    });
    
    group.bench_function("match_historical_signatures", |b| {
        b.iter(|| {
            signature_analyzer.match_historical_signatures(&whale_data)
        })
    });
    
    group.bench_function("predict_whale_behavior", |b| {
        b.iter(|| {
            signature_analyzer.predict_whale_behavior(&whale_data)
        })
    });
    
    group.finish();
}

fn benchmark_large_order_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_order_detection");
    
    let market_data = create_whale_market_data(1500, true);
    let order_detector = LargeOrderDetector::new();
    
    group.bench_function("detect_iceberg_orders", |b| {
        b.iter(|| {
            order_detector.detect_iceberg_orders(&market_data)
        })
    });
    
    group.bench_function("detect_hidden_orders", |b| {
        b.iter(|| {
            order_detector.detect_hidden_orders(&market_data)
        })
    });
    
    group.bench_function("detect_block_trades", |b| {
        b.iter(|| {
            order_detector.detect_block_trades(&market_data)
        })
    });
    
    group.bench_function("analyze_order_fragmentation", |b| {
        b.iter(|| {
            order_detector.analyze_order_fragmentation(&market_data)
        })
    });
    
    group.finish();
}

fn benchmark_whale_activity_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("whale_activity_analysis");
    
    let whale_data = create_whale_market_data(1000, true);
    let activity_analyzer = WhaleActivityAnalyzer::new();
    
    group.bench_function("analyze_accumulation_patterns", |b| {
        b.iter(|| {
            activity_analyzer.analyze_accumulation_patterns(&whale_data)
        })
    });
    
    group.bench_function("analyze_distribution_patterns", |b| {
        b.iter(|| {
            activity_analyzer.analyze_distribution_patterns(&whale_data)
        })
    });
    
    group.bench_function("detect_coordinated_activity", |b| {
        b.iter(|| {
            activity_analyzer.detect_coordinated_whale_activity(&whale_data)
        })
    });
    
    group.bench_function("analyze_market_manipulation", |b| {
        b.iter(|| {
            activity_analyzer.analyze_potential_manipulation(&whale_data)
        })
    });
    
    group.finish();
}

fn benchmark_market_impact_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("market_impact_analysis");
    
    let whale_data = create_whale_market_data(1200, true);
    let impact_analyzer = MarketImpactAnalyzer::new();
    
    group.bench_function("measure_immediate_impact", |b| {
        b.iter(|| {
            impact_analyzer.measure_immediate_impact(&whale_data)
        })
    });
    
    group.bench_function("measure_permanent_impact", |b| {
        b.iter(|| {
            impact_analyzer.measure_permanent_impact(&whale_data)
        })
    });
    
    group.bench_function("analyze_impact_decay", |b| {
        b.iter(|| {
            impact_analyzer.analyze_impact_decay(&whale_data)
        })
    });
    
    group.bench_function("predict_future_impact", |b| {
        b.iter(|| {
            impact_analyzer.predict_future_impact(&whale_data)
        })
    });
    
    group.finish();
}

fn benchmark_volume_anomaly_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("volume_anomaly_detection");
    
    let market_data = create_whale_market_data(1800, true);
    let volume_detector = VolumeAnomalyDetector::new();
    
    group.bench_function("detect_volume_spikes", |b| {
        b.iter(|| {
            volume_detector.detect_volume_spikes(&market_data)
        })
    });
    
    group.bench_function("analyze_volume_patterns", |b| {
        b.iter(|| {
            volume_detector.analyze_volume_patterns(&market_data)
        })
    });
    
    group.bench_function("detect_unusual_volume_clusters", |b| {
        b.iter(|| {
            volume_detector.detect_unusual_volume_clusters(&market_data)
        })
    });
    
    group.bench_function("calculate_volume_profile", |b| {
        b.iter(|| {
            volume_detector.calculate_volume_profile(&market_data)
        })
    });
    
    group.finish();
}

fn benchmark_price_movement_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("price_movement_analysis");
    
    let whale_data = create_whale_market_data(1000, true);
    let price_analyzer = PriceMovementAnalyzer::new();
    
    group.bench_function("analyze_price_momentum", |b| {
        b.iter(|| {
            price_analyzer.analyze_price_momentum(&whale_data)
        })
    });
    
    group.bench_function("detect_price_manipulation", |b| {
        b.iter(|| {
            price_analyzer.detect_price_manipulation(&whale_data)
        })
    });
    
    group.bench_function("analyze_volatility_patterns", |b| {
        b.iter(|| {
            price_analyzer.analyze_volatility_patterns(&whale_data)
        })
    });
    
    group.bench_function("predict_price_direction", |b| {
        b.iter(|| {
            price_analyzer.predict_price_direction(&whale_data)
        })
    });
    
    group.finish();
}

fn benchmark_order_flow_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("order_flow_analysis");
    
    let market_data = create_whale_market_data(1500, true);
    let flow_analyzer = OrderFlowAnalyzer::new();
    
    group.bench_function("analyze_order_flow_imbalance", |b| {
        b.iter(|| {
            flow_analyzer.analyze_order_flow_imbalance(&market_data)
        })
    });
    
    group.bench_function("detect_aggressive_trading", |b| {
        b.iter(|| {
            flow_analyzer.detect_aggressive_trading(&market_data)
        })
    });
    
    group.bench_function("analyze_market_depth_changes", |b| {
        b.iter(|| {
            flow_analyzer.analyze_market_depth_changes(&market_data)
        })
    });
    
    group.bench_function("calculate_order_flow_metrics", |b| {
        b.iter(|| {
            flow_analyzer.calculate_order_flow_metrics(&market_data)
        })
    });
    
    group.finish();
}

fn benchmark_whale_classification(c: &mut Criterion) {
    let mut group = c.benchmark_group("whale_classification");
    
    let whale_data = create_whale_market_data(1000, true);
    let classifier = WhaleClassification::new();
    
    group.bench_function("classify_institutional_whales", |b| {
        b.iter(|| {
            classifier.classify_institutional_whales(&whale_data)
        })
    });
    
    group.bench_function("classify_retail_whales", |b| {
        b.iter(|| {
            classifier.classify_retail_whales(&whale_data)
        })
    });
    
    group.bench_function("classify_algorithmic_whales", |b| {
        b.iter(|| {
            classifier.classify_algorithmic_whales(&whale_data)
        })
    });
    
    group.bench_function("analyze_whale_behavior_patterns", |b| {
        b.iter(|| {
            classifier.analyze_whale_behavior_patterns(&whale_data)
        })
    });
    
    group.finish();
}

fn benchmark_whale_tracking(c: &mut Criterion) {
    let mut group = c.benchmark_group("whale_tracking");
    
    let whale_data = create_whale_market_data(2000, true);
    let tracker = WhaleTracking::new();
    
    group.bench_function("track_whale_movements", |b| {
        b.iter(|| {
            tracker.track_whale_movements(&whale_data)
        })
    });
    
    group.bench_function("maintain_whale_database", |b| {
        b.iter(|| {
            tracker.maintain_whale_database(&whale_data)
        })
    });
    
    group.bench_function("predict_whale_targets", |b| {
        b.iter(|| {
            tracker.predict_whale_targets(&whale_data)
        })
    });
    
    group.bench_function("analyze_whale_network", |b| {
        b.iter(|| {
            tracker.analyze_whale_network(&whale_data)
        })
    });
    
    group.finish();
}

fn benchmark_real_time_whale_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_time_whale_detection");
    
    let detector = WhaleDetector::new();
    
    for batch_size in [10, 50, 100, 500].iter() {
        let whale_data = create_whale_market_data(*batch_size, true);
        
        group.bench_with_input(BenchmarkId::new("streaming_detection", batch_size), batch_size, |b, _| {
            b.iter(|| {
                detector.streaming_whale_detection(&whale_data)
            })
        });
    }
    
    group.bench_function("low_latency_detection", |b| {
        b.iter(|| {
            let single_tick = create_whale_market_data(1, true);
            detector.low_latency_whale_detection(&single_tick[0])
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_whale_detection_algorithms,
    benchmark_whale_signature_analysis,
    benchmark_large_order_detection,
    benchmark_whale_activity_analysis,
    benchmark_market_impact_analysis,
    benchmark_volume_anomaly_detection,
    benchmark_price_movement_analysis,
    benchmark_order_flow_analysis,
    benchmark_whale_classification,
    benchmark_whale_tracking,
    benchmark_real_time_whale_detection
);
criterion_main!(benches);