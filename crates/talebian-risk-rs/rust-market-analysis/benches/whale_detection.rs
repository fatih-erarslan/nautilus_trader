use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rust_market_analysis::{Config, MarketAnalyzer, MarketData, Timeframe};

fn whale_detection_benchmark(c: &mut Criterion) {
    let config = Config::default();
    let analyzer = MarketAnalyzer::new(config).unwrap();
    
    // Create sample market data
    let mut market_data = MarketData::new("BTCUSDT".to_string(), Timeframe::OneMinute);
    market_data.prices = (0..1000).map(|i| 50000.0 + (i as f64 * 10.0)).collect();
    market_data.volumes = (0..1000).map(|i| 100.0 + (i as f64 * 5.0)).collect();
    
    c.bench_function("whale_detection", |b| {
        b.iter(|| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                analyzer.analyze_market(black_box(&market_data)).await.unwrap()
            })
        })
    });
}

criterion_group!(benches, whale_detection_benchmark);
criterion_main!(benches);