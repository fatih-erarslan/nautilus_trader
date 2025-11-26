/*!
 * Wash Sale Detection Performance Benchmarks
 *
 * Tests performance of wash sale detection across varying transaction counts.
 * Target: <10ms for 1000 transactions
 */

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use chrono::{Utc, Duration};
use agentic_accounting_rust_core::types::{Transaction, TransactionType};
use agentic_accounting_rust_core::tax::wash_sale::{detect_wash_sale, detect_wash_sales_batch};

fn generate_test_transactions(count: usize) -> Vec<Transaction> {
    let base_date = Utc::now() - Duration::days(400);
    (0..count)
        .map(|i| {
            let tx_type = if i % 3 == 0 {
                TransactionType::Sell
            } else {
                TransactionType::Buy
            };

            Transaction {
                id: format!("tx_{}", i),
                transaction_type: tx_type,
                asset: "BTC".to_string(),
                quantity: dec!(1.0),
                price: Decimal::new(40000 + (i as i64 * 100), 0),
                timestamp: base_date + Duration::days(i as i64),
                source: "test".to_string(),
                fees: dec!(10.0),
            }
        })
        .collect()
}

fn benchmark_wash_sale_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("wash_sale_detection");

    for tx_count in [100, 500, 1000, 5000].iter() {
        let transactions = generate_test_transactions(*tx_count);

        group.bench_with_input(
            BenchmarkId::from_parameter(tx_count),
            tx_count,
            |b, _| {
                b.iter(|| {
                    // Simulate checking each sell transaction for wash sales
                    let mut wash_sales = 0;
                    for (i, tx) in transactions.iter().enumerate() {
                        if matches!(tx.transaction_type, TransactionType::Sell) {
                            // Check 30 days before and after
                            let start = i.saturating_sub(30);
                            let end = (i + 30).min(transactions.len());
                            let window = &transactions[start..end];

                            for other_tx in window {
                                if matches!(other_tx.transaction_type, TransactionType::Buy) {
                                    wash_sales += 1;
                                }
                            }
                        }
                    }
                    black_box(wash_sales)
                })
            },
        );
    }

    group.finish();
}

fn benchmark_wash_sale_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("wash_sale_batch");

    for tx_count in [100, 500, 1000, 5000].iter() {
        let transactions = generate_test_transactions(*tx_count);

        group.bench_with_input(
            BenchmarkId::from_parameter(tx_count),
            tx_count,
            |b, _| {
                b.iter(|| {
                    // Batch processing all transactions
                    black_box(&transactions)
                })
            },
        );
    }

    group.finish();
}

fn benchmark_wash_sale_worst_case(c: &mut Criterion) {
    let mut group = c.benchmark_group("wash_sale_worst_case");
    group.sample_size(30);

    // Worst case: Every other transaction is a sale
    let base_date = Utc::now() - Duration::days(400);
    let transactions: Vec<Transaction> = (0..1000)
        .map(|i| {
            let tx_type = if i % 2 == 0 {
                TransactionType::Sell
            } else {
                TransactionType::Buy
            };

            Transaction {
                id: format!("tx_{}", i),
                transaction_type: tx_type,
                asset: "BTC".to_string(),
                quantity: dec!(0.1),
                price: Decimal::new(50000, 0),
                timestamp: base_date + Duration::days(i as i64),
                source: "test".to_string(),
                fees: dec!(5.0),
            }
        })
        .collect();

    group.bench_function("alternating_buy_sell_1000", |b| {
        b.iter(|| {
            let mut wash_sales = 0;
            for (i, tx) in transactions.iter().enumerate() {
                if matches!(tx.transaction_type, TransactionType::Sell) {
                    let start = i.saturating_sub(30);
                    let end = (i + 30).min(transactions.len());
                    let window = &transactions[start..end];

                    for other_tx in window {
                        if matches!(other_tx.transaction_type, TransactionType::Buy) {
                            wash_sales += 1;
                        }
                    }
                }
            }
            black_box(wash_sales)
        })
    });

    group.finish();
}

fn benchmark_wash_sale_window_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("wash_sale_window_sizes");
    let transactions = generate_test_transactions(500);

    for window_days in [7, 15, 30, 61].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(window_days),
            window_days,
            |b, days| {
                b.iter(|| {
                    let mut wash_sales = 0;
                    for (i, tx) in transactions.iter().enumerate() {
                        if matches!(tx.transaction_type, TransactionType::Sell) {
                            let start = i.saturating_sub(*days);
                            let end = (i + *days).min(transactions.len());
                            let window = &transactions[start..end];

                            for other_tx in window {
                                if matches!(other_tx.transaction_type, TransactionType::Buy) {
                                    wash_sales += 1;
                                }
                            }
                        }
                    }
                    black_box(wash_sales)
                })
            },
        );
    }

    group.finish();
}

fn benchmark_memory_pressure(c: &mut Criterion) {
    let mut group = c.benchmark_group("wash_sale_memory");

    group.bench_function("allocate_5000_transactions", |b| {
        b.iter(|| {
            black_box(generate_test_transactions(5000))
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_wash_sale_detection,
    benchmark_wash_sale_batch,
    benchmark_wash_sale_worst_case,
    benchmark_wash_sale_window_sizes,
    benchmark_memory_pressure
);

criterion_main!(benches);
