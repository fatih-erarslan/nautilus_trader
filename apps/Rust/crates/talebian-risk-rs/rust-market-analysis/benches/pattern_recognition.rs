use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rust_market_analysis::{Config, MarketAnalyzer, MarketData, Timeframe};

fn pattern_recognition_benchmark(c: &mut Criterion) {
    let config = Config::default();
    let analyzer = MarketAnalyzer::new(config).unwrap();
    
    // Create sample market data with pattern-like structure
    let mut market_data = MarketData::new("BTCUSDT".to_string(), Timeframe::OneMinute);
    
    // Create head and shoulders pattern
    let mut prices = Vec::new();
    for i in 0..100 {
        let price = match i {
            0..=20 => 50000.0 + (i as f64 * 50.0), // Left shoulder
            21..=40 => 51000.0 - ((i - 20) as f64 * 25.0), // Valley
            41..=60 => 50500.0 + ((i - 40) as f64 * 75.0), // Head
            61..=80 => 52000.0 - ((i - 60) as f64 * 25.0), // Valley
            _ => 51500.0 + ((i - 80) as f64 * 25.0), // Right shoulder
        };
        prices.push(price);
    }
    
    market_data.prices = prices;
    market_data.volumes = (0..100).map(|i| 100.0 + (i as f64 * 2.0)).collect();
    
    c.bench_function("pattern_recognition", |b| {
        b.iter(|| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                analyzer.analyze_market(black_box(&market_data)).await.unwrap()
            })
        })
    });
}

criterion_group!(benches, pattern_recognition_benchmark);
criterion_main!(benches);