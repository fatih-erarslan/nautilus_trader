use criterion::{black_box, criterion_group, criterion_main, Criterion};
use whale_hunting_strategy::impact::MarketImpactAnalyzer;
use whale_hunting_strategy::detector::WhaleDetector;
use whale_hunting_strategy::config::WhaleHuntingConfig;

fn market_impact_benchmark(c: &mut Criterion) {
    let config = WhaleHuntingConfig::default();
    let analyzer = MarketImpactAnalyzer::new(config);
    let order_flow = vec![
        (1000.0, 100.0),
        (2000.0, 101.0),
        (3000.0, 102.0),
        (4000.0, 103.0),
    ];
    
    c.bench_function("market_impact", |b| {
        b.iter(|| {
            analyzer.analyze_impact(black_box(&order_flow))
        })
    });
}

fn whale_detection_benchmark(c: &mut Criterion) {
    let detector = WhaleDetector::new();
    let trade_data = vec![
        (10000.0, 100.0),
        (15000.0, 101.0),
        (20000.0, 102.0),
        (25000.0, 103.0),
    ];
    
    c.bench_function("whale_detection", |b| {
        b.iter(|| {
            detector.detect_whales(black_box(&trade_data))
        })
    });
}

criterion_group!(benches, market_impact_benchmark, whale_detection_benchmark);
criterion_main!(benches);