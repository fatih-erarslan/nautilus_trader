//! Performance benchmarks for Machiavellian strategic framework

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use qbmia_core::{
    strategy::{MachiavellianFramework, OrderEvent},
    config::HardwareConfig,
};
use tokio::runtime::Runtime;

fn generate_test_orders(count: usize) -> Vec<OrderEvent> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    (0..count)
        .map(|i| OrderEvent {
            timestamp: i as f64,
            side: if rng.gen_bool(0.5) { "buy".to_string() } else { "sell".to_string() },
            size: rng.gen_range(1.0..1000.0),
            price: rng.gen_range(49000.0..51000.0),
            cancelled: rng.gen_bool(0.1), // 10% cancellation rate
        })
        .collect()
}

fn generate_test_prices(count: usize) -> Vec<f64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut prices = Vec::with_capacity(count);
    let mut current_price = 50000.0;
    
    for _ in 0..count {
        current_price += rng.gen_range(-100.0..100.0);
        prices.push(current_price);
    }
    
    prices
}

fn bench_manipulation_detection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("manipulation_detection");
    
    for order_count in [100, 500, 1000, 5000].iter() {
        group.bench_with_input(
            BenchmarkId::new("detect_manipulation", order_count),
            order_count,
            |b, &order_count| {
                let orders = generate_test_orders(order_count);
                let prices = generate_test_prices(order_count / 10);
                
                b.to_async(&rt).iter(|| async {
                    let config = HardwareConfig::default();
                    let mut framework = MachiavellianFramework::new(config, 0.7).unwrap();
                    
                    let result = framework.detect_manipulation(&orders, &prices).await.unwrap();
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_spoofing_detection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("spoofing_detection");
    
    for order_count in [100, 500, 1000, 5000].iter() {
        group.bench_with_input(
            BenchmarkId::new("spoofing_detection", order_count),
            order_count,
            |b, &order_count| {
                let orders = generate_test_orders(order_count);
                let prices = generate_test_prices(order_count / 10);
                
                b.to_async(&rt).iter(|| async {
                    let config = HardwareConfig::default();
                    let mut framework = MachiavellianFramework::new(config, 0.7).unwrap();
                    
                    let result = framework.detect_manipulation(&orders, &prices).await.unwrap();
                    black_box(result.manipulation_scores.get("spoofing").unwrap_or(&0.0))
                });
            },
        );
    }
    
    group.finish();
}

fn bench_layering_detection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("layering_detection");
    
    for order_count in [100, 500, 1000, 2000].iter() {
        group.bench_with_input(
            BenchmarkId::new("layering_detection", order_count),
            order_count,
            |b, &order_count| {
                let orders = generate_test_orders(order_count);
                let prices = generate_test_prices(order_count / 10);
                
                b.to_async(&rt).iter(|| async {
                    let config = HardwareConfig::default();
                    let mut framework = MachiavellianFramework::new(config, 0.7).unwrap();
                    
                    let result = framework.detect_manipulation(&orders, &prices).await.unwrap();
                    black_box(result.manipulation_scores.get("layering").unwrap_or(&0.0))
                });
            },
        );
    }
    
    group.finish();
}

fn bench_strategy_generation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("strategy_generation", |b| {
        b.to_async(&rt).iter(|| async {
            let config = HardwareConfig::default();
            let framework = MachiavellianFramework::new(config, 0.7).unwrap();
            
            let manipulation_result = qbmia_core::strategy::ManipulationDetectionResult {
                detected: true,
                confidence: 0.8,
                manipulation_scores: std::collections::HashMap::new(),
                primary_pattern: "spoofing".to_string(),
                execution_time: 1.0,
                recommended_action: "DEFENSIVE_TRADING".to_string(),
            };
            
            let competitors = std::collections::HashMap::new();
            let strategy = framework.generate_strategy(&manipulation_result, &competitors).await.unwrap();
            black_box(strategy)
        });
    });
}

criterion_group!(
    benches,
    bench_manipulation_detection,
    bench_spoofing_detection,
    bench_layering_detection,
    bench_strategy_generation
);
criterion_main!(benches);