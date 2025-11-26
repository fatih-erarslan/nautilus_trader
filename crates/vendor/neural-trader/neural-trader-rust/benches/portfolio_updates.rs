// Portfolio Updates Benchmark
//
// Performance targets:
// - Position update: <100μs
// - P&L calculation: <100μs
// - Portfolio sync: <1ms
// - Throughput: >10,000 operations/sec

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nt_core::types::{Direction, Symbol};
use nt_portfolio::{Portfolio, Position, Trade};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::time::Duration;

// ============================================================================
// Helper Functions - Generate Test Data
// ============================================================================

fn create_position(symbol: &str, quantity: Decimal, price: Decimal) -> Position {
    Position {
        symbol: Symbol::new(symbol).unwrap(),
        quantity,
        avg_entry_price: price,
        current_price: price,
        direction: Direction::Long,
        unrealized_pnl: Decimal::ZERO,
        realized_pnl: Decimal::ZERO,
    }
}

fn create_trade(symbol: &str, quantity: Decimal, price: Decimal, direction: Direction) -> Trade {
    Trade {
        id: uuid::Uuid::new_v4().to_string(),
        symbol: Symbol::new(symbol).unwrap(),
        quantity,
        price,
        direction,
        timestamp: chrono::Utc::now(),
        commission: dec!(1.0),
    }
}

// ============================================================================
// Benchmarks - Position Updates
// ============================================================================

fn bench_position_updates(c: &mut Criterion) {
    let mut group = c.benchmark_group("position_updates");

    group.bench_function("add_new_position", |b| {
        let mut portfolio = Portfolio::new(dec!(100000));
        let position = create_position("AAPL", dec!(100), dec!(150));

        b.iter(|| {
            portfolio.add_position(black_box(position.clone()));
        });
    });

    group.bench_function("update_existing_position", |b| {
        let mut portfolio = Portfolio::new(dec!(100000));
        let position = create_position("AAPL", dec!(100), dec!(150));
        portfolio.add_position(position.clone());

        b.iter(|| {
            let mut updated_position = position.clone();
            updated_position.current_price = dec!(155);
            portfolio.update_position(black_box(updated_position));
        });
    });

    group.bench_function("remove_position", |b| {
        let mut portfolio = Portfolio::new(dec!(100000));
        let symbol = Symbol::new("AAPL").unwrap();
        let position = create_position("AAPL", dec!(100), dec!(150));
        portfolio.add_position(position);

        b.iter(|| {
            portfolio.remove_position(black_box(&symbol));
        });
    });

    group.finish();
}

// ============================================================================
// Benchmarks - P&L Calculations
// ============================================================================

fn bench_pnl_calculations(c: &mut Criterion) {
    let mut group = c.benchmark_group("pnl_calculations");

    group.bench_function("calculate_unrealized_pnl", |b| {
        let position = Position {
            symbol: Symbol::new("AAPL").unwrap(),
            quantity: dec!(100),
            avg_entry_price: dec!(150),
            current_price: dec!(155),
            direction: Direction::Long,
            unrealized_pnl: Decimal::ZERO,
            realized_pnl: Decimal::ZERO,
        };

        b.iter(|| {
            let pnl = (position.current_price - position.avg_entry_price) * position.quantity;
            black_box(pnl)
        });
    });

    group.bench_function("calculate_realized_pnl", |b| {
        let entry_price = dec!(150);
        let exit_price = dec!(155);
        let quantity = dec!(100);
        let commission = dec!(2.0); // $1 entry + $1 exit

        b.iter(|| {
            let gross_pnl = (exit_price - entry_price) * quantity;
            let net_pnl = gross_pnl - commission;
            black_box(net_pnl)
        });
    });

    group.bench_function("calculate_total_portfolio_pnl", |b| {
        let mut portfolio = Portfolio::new(dec!(100000));

        for i in 0..50 {
            let symbol = format!("SYM{}", i);
            let position = create_position(&symbol, dec!(100), dec!(100) + Decimal::from(i));
            portfolio.add_position(position);
        }

        b.iter(|| {
            portfolio.calculate_total_pnl()
        });
    });

    group.finish();
}

// ============================================================================
// Benchmarks - Portfolio Metrics
// ============================================================================

fn bench_portfolio_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("portfolio_metrics");

    let mut portfolio = Portfolio::new(dec!(100000));
    for i in 0..50 {
        let symbol = format!("SYM{}", i);
        let position = create_position(&symbol, dec!(100), dec!(100) + Decimal::from(i));
        portfolio.add_position(position);
    }

    group.bench_function("calculate_total_value", |b| {
        b.iter(|| {
            portfolio.total_value()
        });
    });

    group.bench_function("calculate_buying_power", |b| {
        b.iter(|| {
            portfolio.buying_power()
        });
    });

    group.bench_function("calculate_leverage", |b| {
        b.iter(|| {
            portfolio.leverage()
        });
    });

    group.bench_function("count_positions", |b| {
        b.iter(|| {
            portfolio.position_count()
        });
    });

    group.finish();
}

// ============================================================================
// Benchmarks - Trade Processing
// ============================================================================

fn bench_trade_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("trade_processing");

    group.bench_function("process_single_trade", |b| {
        let mut portfolio = Portfolio::new(dec!(100000));
        let trade = create_trade("AAPL", dec!(100), dec!(150), Direction::Long);

        b.iter(|| {
            portfolio.process_trade(black_box(trade.clone()));
        });
    });

    for batch_size in [10, 50, 100].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("batch_trades", batch_size),
            batch_size,
            |b, &batch_size| {
                let mut portfolio = Portfolio::new(dec!(100000));
                let trades: Vec<_> = (0..batch_size)
                    .map(|i| {
                        let symbol = format!("SYM{}", i % 10);
                        create_trade(&symbol, dec!(100), dec!(100) + Decimal::from(i), Direction::Long)
                    })
                    .collect();

                b.iter(|| {
                    for trade in black_box(&trades) {
                        portfolio.process_trade(trade.clone());
                    }
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Position Size Calculations
// ============================================================================

fn bench_position_sizing(c: &mut Criterion) {
    let mut group = c.benchmark_group("position_sizing");

    group.bench_function("calculate_position_value", |b| {
        let position = create_position("AAPL", dec!(100), dec!(150));

        b.iter(|| {
            let value = position.quantity * position.current_price;
            black_box(value)
        });
    });

    group.bench_function("calculate_position_weight", |b| {
        let portfolio_value = dec!(100000);
        let position_value = dec!(15000);

        b.iter(|| {
            let weight = position_value / portfolio_value;
            black_box(weight)
        });
    });

    group.bench_function("calculate_max_position_size", |b| {
        let portfolio_value = dec!(100000);
        let max_weight = dec!(0.1); // 10% max per position
        let price = dec!(150);

        b.iter(|| {
            let max_value = portfolio_value * max_weight;
            let max_shares = max_value / price;
            black_box(max_shares)
        });
    });

    group.finish();
}

// ============================================================================
// Benchmarks - Concurrent Portfolio Updates
// ============================================================================

fn bench_concurrent_updates(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_updates");
    group.sample_size(50);

    let rt = tokio::runtime::Runtime::new().unwrap();

    for thread_count in [2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(thread_count),
            thread_count,
            |b, &thread_count| {
                b.to_async(&rt).iter(|| async move {
                    let mut handles = vec![];

                    for i in 0..thread_count {
                        let handle = tokio::spawn(async move {
                            let mut portfolio = Portfolio::new(dec!(100000));
                            let symbol = format!("SYM{}", i);
                            let position = create_position(&symbol, dec!(100), dec!(150));

                            for _ in 0..100 {
                                portfolio.add_position(position.clone());
                                let _ = portfolio.total_value();
                            }
                        });

                        handles.push(handle);
                    }

                    for handle in handles {
                        let _ = handle.await;
                    }
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Portfolio Snapshots
// ============================================================================

fn bench_portfolio_snapshots(c: &mut Criterion) {
    let mut group = c.benchmark_group("portfolio_snapshots");

    for position_count in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(position_count),
            position_count,
            |b, &position_count| {
                let mut portfolio = Portfolio::new(dec!(100000));

                for i in 0..position_count {
                    let symbol = format!("SYM{}", i);
                    let position = create_position(&symbol, dec!(100), dec!(100) + Decimal::from(i));
                    portfolio.add_position(position);
                }

                b.iter(|| {
                    portfolio.create_snapshot()
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Commission Calculations
// ============================================================================

fn bench_commission_calculations(c: &mut Criterion) {
    let mut group = c.benchmark_group("commission_calculations");

    group.bench_function("fixed_commission", |b| {
        let commission_per_trade = dec!(1.0);

        b.iter(|| {
            black_box(commission_per_trade)
        });
    });

    group.bench_function("percentage_commission", |b| {
        let trade_value = dec!(15000);
        let commission_rate = dec!(0.001); // 0.1%

        b.iter(|| {
            let commission = trade_value * commission_rate;
            black_box(commission)
        });
    });

    group.bench_function("tiered_commission", |b| {
        let shares = dec!(1000);

        b.iter(|| {
            let commission = if shares <= dec!(100) {
                dec!(1.0)
            } else if shares <= dec!(500) {
                dec!(0.75)
            } else {
                dec!(0.50)
            };

            black_box(commission)
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
        bench_position_updates,
        bench_pnl_calculations,
        bench_portfolio_metrics,
        bench_trade_processing,
        bench_position_sizing,
        bench_concurrent_updates,
        bench_portfolio_snapshots,
        bench_commission_calculations
}

criterion_main!(benches);
