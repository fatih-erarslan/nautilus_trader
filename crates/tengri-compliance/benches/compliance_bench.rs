//! Performance benchmarks for TENGRI compliance engine

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use tokio::runtime::Runtime;
use std::collections::HashMap;
use uuid::Uuid;
use chrono::Utc;
use rust_decimal::Decimal;

use tengri_compliance::{
    ComplianceEngine, ComplianceConfig,
    rules::{TradingContext, OrderSide, OrderType, Position},
};

fn create_test_context(trade_size: f64) -> TradingContext {
    let mut positions = HashMap::new();
    positions.insert("BTCUSD".to_string(), Position {
        symbol: "BTCUSD".to_string(),
        quantity: Decimal::from(10),
        average_price: Decimal::from(50000),
        unrealized_pnl: Decimal::from(1000),
        market_value: Decimal::from(500000),
    });

    TradingContext {
        order_id: Uuid::new_v4(),
        symbol: "BTCUSD".to_string(),
        side: OrderSide::Buy,
        quantity: Decimal::from_f64(trade_size).unwrap(),
        price: Some(Decimal::from(50000)),
        order_type: OrderType::Limit,
        trader_id: "trader_001".to_string(),
        timestamp: Utc::now(),
        portfolio_value: Decimal::from(1_000_000),
        current_positions: positions,
        daily_pnl: Decimal::from(5000),
        metadata: HashMap::new(),
    }
}

async fn setup_engine() -> ComplianceEngine {
    let config = ComplianceConfig::default();
    ComplianceEngine::new(config).await.unwrap()
}

fn bench_trade_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let engine = rt.block_on(setup_engine());
    
    let mut group = c.benchmark_group("trade_processing");
    
    for trade_size in [0.1, 1.0, 10.0, 100.0].iter() {
        group.bench_with_input(
            BenchmarkId::new("process_trade", trade_size),
            trade_size,
            |b, &size| {
                b.to_async(&rt).iter(|| {
                    let context = create_test_context(size);
                    async {
                        let result = engine.process_trade(black_box(context)).await;
                        black_box(result)
                    }
                });
            },
        );
    }
    
    group.finish();
}

fn bench_rule_evaluation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let engine = rt.block_on(setup_engine());
    
    c.bench_function("rule_evaluation", |b| {
        b.to_async(&rt).iter(|| {
            let context = create_test_context(1.0);
            async {
                let rule_engine = engine.get_rule_engine();
                let result = rule_engine.evaluate_all(black_box(&context)).await;
                black_box(result)
            }
        });
    });
}

fn bench_audit_recording(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let engine = rt.block_on(setup_engine());
    
    c.bench_function("audit_recording", |b| {
        b.to_async(&rt).iter(|| {
            async {
                let audit_trail = engine.get_audit_trail();
                let result = audit_trail.record(
                    black_box(tengri_compliance::audit::AuditEventType::TradeSubmitted { 
                        order_id: Uuid::new_v4() 
                    }),
                    black_box("trader_001".to_string()),
                    black_box(serde_json::json!({"test": "data"})),
                ).await;
                black_box(result)
            }
        });
    });
}

fn bench_surveillance_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let engine = rt.block_on(setup_engine());
    
    c.bench_function("surveillance_analysis", |b| {
        b.to_async(&rt).iter(|| {
            async {
                let surveillance = engine.get_surveillance_engine();
                let result = surveillance.analyze_patterns().await;
                black_box(result)
            }
        });
    });
}

fn bench_concurrent_trades(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let engine = rt.block_on(setup_engine());
    
    let mut group = c.benchmark_group("concurrent_trades");
    
    for concurrent_count in [1, 10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent_processing", concurrent_count),
            concurrent_count,
            |b, &count| {
                b.to_async(&rt).iter(|| {
                    async {
                        let futures: Vec<_> = (0..count)
                            .map(|_| {
                                let context = create_test_context(1.0);
                                engine.process_trade(context)
                            })
                            .collect();
                        
                        let results = futures::future::join_all(futures).await;
                        black_box(results)
                    }
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_trade_processing,
    bench_rule_evaluation,
    bench_audit_recording,
    bench_surveillance_analysis,
    bench_concurrent_trades
);

criterion_main!(benches);