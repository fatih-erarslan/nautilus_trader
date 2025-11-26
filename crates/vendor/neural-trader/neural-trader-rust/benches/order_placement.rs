// Order Placement Benchmark
//
// Performance targets:
// - Order creation: <1ms
// - Order validation: <2ms
// - Order placement: <10ms
// - End-to-end: <61ms (including external broker API)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nt_core::types::{Direction, OrderSide, OrderType, Symbol, TimeInForce};
use nt_execution::{broker::BrokerClient, order_manager::OrderManager, router::OrderRouter};
use nt_portfolio::Portfolio;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::sync::Arc;
use std::time::Duration;

// ============================================================================
// Mock Broker for Benchmarking
// ============================================================================

#[derive(Clone)]
struct MockBroker {
    latency_ms: u64,
}

impl MockBroker {
    fn new(latency_ms: u64) -> Self {
        Self { latency_ms }
    }
}

#[async_trait::async_trait]
impl BrokerClient for MockBroker {
    async fn place_order(
        &self,
        request: nt_execution::broker::OrderRequest,
    ) -> Result<nt_execution::broker::OrderResponse, nt_execution::broker::BrokerError> {
        // Simulate network latency
        tokio::time::sleep(Duration::from_millis(self.latency_ms)).await;

        Ok(nt_execution::broker::OrderResponse {
            order_id: uuid::Uuid::new_v4().to_string(),
            symbol: request.symbol,
            side: request.side,
            qty: request.qty,
            price: request.price,
            status: nt_execution::broker::OrderStatus::Filled,
            filled_qty: request.qty,
            filled_avg_price: request.price,
            timestamp: chrono::Utc::now(),
        })
    }

    async fn cancel_order(
        &self,
        _order_id: &str,
    ) -> Result<(), nt_execution::broker::BrokerError> {
        tokio::time::sleep(Duration::from_millis(self.latency_ms)).await;
        Ok(())
    }

    async fn get_order_status(
        &self,
        _order_id: &str,
    ) -> Result<nt_execution::broker::OrderStatus, nt_execution::broker::BrokerError> {
        tokio::time::sleep(Duration::from_millis(self.latency_ms / 2)).await;
        Ok(nt_execution::broker::OrderStatus::Filled)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn create_order_request(symbol: &str, side: OrderSide, qty: Decimal) -> nt_execution::broker::OrderRequest {
    nt_execution::broker::OrderRequest {
        symbol: Symbol::new(symbol).unwrap(),
        side,
        qty,
        order_type: OrderType::Market,
        time_in_force: TimeInForce::Day,
        limit_price: None,
        stop_price: None,
        client_order_id: Some(uuid::Uuid::new_v4().to_string()),
    }
}

// ============================================================================
// Benchmarks - Order Creation
// ============================================================================

fn bench_order_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("order_creation");

    group.bench_function("create_market_order", |b| {
        b.iter(|| {
            black_box(create_order_request("AAPL", OrderSide::Buy, dec!(100)))
        });
    });

    group.bench_function("create_limit_order", |b| {
        b.iter(|| {
            black_box(nt_execution::broker::OrderRequest {
                symbol: Symbol::new("AAPL").unwrap(),
                side: OrderSide::Buy,
                qty: dec!(100),
                order_type: OrderType::Limit,
                time_in_force: TimeInForce::Day,
                limit_price: Some(dec!(150.50)),
                stop_price: None,
                client_order_id: Some(uuid::Uuid::new_v4().to_string()),
            })
        });
    });

    group.bench_function("create_stop_loss_order", |b| {
        b.iter(|| {
            black_box(nt_execution::broker::OrderRequest {
                symbol: Symbol::new("AAPL").unwrap(),
                side: OrderSide::Sell,
                qty: dec!(100),
                order_type: OrderType::StopLoss,
                time_in_force: TimeInForce::Day,
                limit_price: None,
                stop_price: Some(dec!(145.00)),
                client_order_id: Some(uuid::Uuid::new_v4().to_string()),
            })
        });
    });

    group.finish();
}

// ============================================================================
// Benchmarks - Order Validation
// ============================================================================

fn bench_order_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("order_validation");

    let rt = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("validate_order_request", |b| {
        let order = create_order_request("AAPL", OrderSide::Buy, dec!(100));

        b.iter(|| {
            // Simulate validation logic
            let is_valid = order.qty > Decimal::ZERO
                && !order.symbol.as_str().is_empty()
                && matches!(order.side, OrderSide::Buy | OrderSide::Sell);

            black_box(is_valid)
        });
    });

    group.bench_function("validate_with_portfolio_check", |b| {
        let order = create_order_request("AAPL", OrderSide::Buy, dec!(100));
        let portfolio = Portfolio::new(dec!(100000));

        b.iter(|| {
            // Check buying power
            let total_cost = order.qty * dec!(150); // Assume $150/share
            let has_funds = portfolio.cash() >= total_cost;

            black_box(has_funds)
        });
    });

    group.finish();
}

// ============================================================================
// Benchmarks - Order Placement (Mock Broker)
// ============================================================================

fn bench_order_placement(c: &mut Criterion) {
    let mut group = c.benchmark_group("order_placement");
    group.sample_size(50);

    let rt = tokio::runtime::Runtime::new().unwrap();

    // Test with different simulated latencies
    for latency_ms in [5, 10, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("latency_ms", latency_ms),
            latency_ms,
            |b, &latency_ms| {
                let broker = Arc::new(MockBroker::new(latency_ms));
                let order = create_order_request("AAPL", OrderSide::Buy, dec!(100));

                b.to_async(&rt).iter(|| {
                    let broker = broker.clone();
                    let order = order.clone();

                    async move {
                        broker.place_order(black_box(order)).await
                    }
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Batch Order Placement
// ============================================================================

fn bench_batch_order_placement(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_order_placement");
    group.sample_size(30);

    let rt = tokio::runtime::Runtime::new().unwrap();

    for batch_size in [5, 10, 20].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &batch_size| {
                let broker = Arc::new(MockBroker::new(10));

                b.to_async(&rt).iter(|| {
                    let broker = broker.clone();

                    async move {
                        let mut handles = vec![];

                        for i in 0..batch_size {
                            let symbol = format!("SYM{}", i);
                            let order = create_order_request(&symbol, OrderSide::Buy, dec!(100));
                            let broker = broker.clone();

                            let handle = tokio::spawn(async move {
                                broker.place_order(order).await
                            });

                            handles.push(handle);
                        }

                        // Wait for all orders
                        for handle in handles {
                            let _ = handle.await;
                        }
                    }
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Order Router
// ============================================================================

fn bench_order_router(c: &mut Criterion) {
    let mut group = c.benchmark_group("order_router");

    let rt = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("route_to_single_broker", |b| {
        let broker = Arc::new(MockBroker::new(10));
        let router = OrderRouter::new(vec![broker]);
        let order = create_order_request("AAPL", OrderSide::Buy, dec!(100));

        b.to_async(&rt).iter(|| {
            let router = router.clone();
            let order = order.clone();

            async move {
                router.route_order(black_box(order)).await
            }
        });
    });

    group.finish();
}

// ============================================================================
// Benchmarks - Order Manager
// ============================================================================

fn bench_order_manager(c: &mut Criterion) {
    let mut group = c.benchmark_group("order_manager");
    group.sample_size(50);

    let rt = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("submit_and_track_order", |b| {
        let broker = Arc::new(MockBroker::new(10));
        let manager = OrderManager::new(broker);
        let order = create_order_request("AAPL", OrderSide::Buy, dec!(100));

        b.to_async(&rt).iter(|| {
            let manager = manager.clone();
            let order = order.clone();

            async move {
                manager.submit_order(black_box(order)).await
            }
        });
    });

    group.finish();
}

// ============================================================================
// Benchmarks - Cancel Order
// ============================================================================

fn bench_order_cancellation(c: &mut Criterion) {
    let mut group = c.benchmark_group("order_cancellation");
    group.sample_size(50);

    let rt = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("cancel_single_order", |b| {
        let broker = Arc::new(MockBroker::new(10));
        let order_id = "test-order-123";

        b.to_async(&rt).iter(|| {
            let broker = broker.clone();

            async move {
                broker.cancel_order(black_box(order_id)).await
            }
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark Configuration
// ============================================================================

fn configure_criterion() -> Criterion {
    Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::from_secs(5))
        .warm_up_time(Duration::from_secs(2))
        .configure_from_args()
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets =
        bench_order_creation,
        bench_order_validation,
        bench_order_placement,
        bench_batch_order_placement,
        bench_order_router,
        bench_order_manager,
        bench_order_cancellation
}

criterion_main!(benches);
