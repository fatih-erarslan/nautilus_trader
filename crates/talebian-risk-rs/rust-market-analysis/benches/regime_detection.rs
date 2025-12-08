use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rust_market_analysis::{Config, MarketAnalyzer, MarketData, Timeframe};

fn regime_detection_benchmark(c: &mut Criterion) {
    let config = Config::default();
    let analyzer = MarketAnalyzer::new(config).unwrap();
    
    // Create sample market data with regime changes
    let mut market_data = MarketData::new("BTCUSDT".to_string(), Timeframe::OneMinute);
    
    let mut prices = Vec::new();
    let mut volumes = Vec::new();
    
    // Bull market regime
    for i in 0..200 {
        let price = 50000.0 + (i as f64 * 25.0) + (i as f64 % 10.0 - 5.0) * 20.0;
        let volume = 100.0 + (i as f64 % 5.0) * 10.0;
        prices.push(price);
        volumes.push(volume);
    }
    
    // Bear market regime
    for i in 0..200 {
        let price = 55000.0 - (i as f64 * 15.0) + (i as f64 % 7.0 - 3.5) * 30.0;
        let volume = 120.0 + (i as f64 % 6.0) * 15.0;
        prices.push(price);
        volumes.push(volume);
    }
    
    // Sideways regime
    for i in 0..200 {
        let price = 52000.0 + ((i as f64 * 0.1).sin() * 200.0) + (i as f64 % 8.0 - 4.0) * 25.0;
        let volume = 90.0 + (i as f64 % 4.0) * 10.0;
        prices.push(price);
        volumes.push(volume);
    }
    
    market_data.prices = prices;
    market_data.volumes = volumes;
    
    c.bench_function("regime_detection", |b| {
        b.iter(|| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                analyzer.analyze_market(black_box(&market_data)).await.unwrap()
            })
        })
    });
}

criterion_group!(benches, regime_detection_benchmark);
criterion_main!(benches);