/*!
 * Performance benchmarks for FIFO tax calculation algorithm
 *
 * Measures performance across various scenarios:
 * - Different numbers of tax lots (10, 100, 1000)
 * - Different disposal quantities
 * - Worst-case scenarios (many small lots)
 */

use agentic_accounting_rust_core::{
    calculate_fifo_disposal,
    Transaction, TaxLot, TransactionType,
};
use chrono::{TimeZone, Utc};
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rust_decimal::Decimal;
use std::str::FromStr;

fn create_test_sale(quantity: &str) -> Transaction {
    Transaction {
        id: "sale1".to_string(),
        transaction_type: TransactionType::Sell,
        asset: "BTC".to_string(),
        quantity: Decimal::from_str(quantity).unwrap(),
        price: Decimal::from_str("60000").unwrap(),
        timestamp: Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap(),
        source: "test".to_string(),
        fees: Decimal::ZERO,
    }
}

fn create_test_lots(count: usize) -> Vec<TaxLot> {
    (0..count)
        .map(|i| {
            let qty = Decimal::from_str("1.0").unwrap();
            TaxLot {
                id: format!("lot{}", i),
                transaction_id: format!("buy{}", i),
                asset: "BTC".to_string(),
                quantity: qty,
                remaining_quantity: qty,
                cost_basis: Decimal::from_str(&format!("{}", 40000 + i * 100)).unwrap(),
                acquisition_date: Utc.with_ymd_and_hms(2023, 1, (i % 28) + 1, 0, 0, 0).unwrap(),
            }
        })
        .collect()
}

fn bench_fifo_small_dataset(c: &mut Criterion) {
    let mut group = c.benchmark_group("fifo_small");

    // 10 lots, selling 5
    let sale = create_test_sale("5.0");
    let lots = create_test_lots(10);

    group.bench_function("10_lots_sell_5", |b| {
        b.iter(|| {
            let lots_clone = lots.clone();
            black_box(calculate_fifo_disposal(&sale, lots_clone).unwrap())
        });
    });

    group.finish();
}

fn bench_fifo_medium_dataset(c: &mut Criterion) {
    let mut group = c.benchmark_group("fifo_medium");

    // 100 lots, selling 50
    let sale = create_test_sale("50.0");
    let lots = create_test_lots(100);

    group.bench_function("100_lots_sell_50", |b| {
        b.iter(|| {
            let lots_clone = lots.clone();
            black_box(calculate_fifo_disposal(&sale, lots_clone).unwrap())
        });
    });

    group.finish();
}

fn bench_fifo_large_dataset(c: &mut Criterion) {
    let mut group = c.benchmark_group("fifo_large");

    // 1000 lots, selling 500
    let sale = create_test_sale("500.0");
    let lots = create_test_lots(1000);

    group.bench_function("1000_lots_sell_500", |b| {
        b.iter(|| {
            let lots_clone = lots.clone();
            black_box(calculate_fifo_disposal(&sale, lots_clone).unwrap())
        });
    });

    group.finish();
}

fn bench_fifo_full_disposal(c: &mut Criterion) {
    let mut group = c.benchmark_group("fifo_full_disposal");

    // Test disposing all lots
    for size in [10, 50, 100, 200].iter() {
        let sale = create_test_sale(&format!("{}.0", size));
        let lots = create_test_lots(*size);

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                b.iter(|| {
                    let lots_clone = lots.clone();
                    black_box(calculate_fifo_disposal(&sale, lots_clone).unwrap())
                });
            },
        );
    }

    group.finish();
}

fn bench_fifo_partial_disposal(c: &mut Criterion) {
    let mut group = c.benchmark_group("fifo_partial_disposal");

    // Test disposing 10% of lots
    for size in [100, 200, 500, 1000].iter() {
        let sale_qty = *size as f64 * 0.1;
        let sale = create_test_sale(&format!("{:.1}", sale_qty));
        let lots = create_test_lots(*size);

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                b.iter(|| {
                    let lots_clone = lots.clone();
                    black_box(calculate_fifo_disposal(&sale, lots_clone).unwrap())
                });
            },
        );
    }

    group.finish();
}

fn bench_fifo_single_lot_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("fifo_single_lot");

    // Single lot, full disposal
    let sale = create_test_sale("1.0");
    let lots = create_test_lots(1);

    group.bench_function("single_lot_full", |b| {
        b.iter(|| {
            let lots_clone = lots.clone();
            black_box(calculate_fifo_disposal(&sale, lots_clone).unwrap())
        });
    });

    // Single lot, partial disposal
    let sale_partial = create_test_sale("0.5");
    group.bench_function("single_lot_partial", |b| {
        b.iter(|| {
            let lots_clone = lots.clone();
            black_box(calculate_fifo_disposal(&sale_partial, lots_clone).unwrap())
        });
    });

    group.finish();
}

fn bench_fifo_worst_case(c: &mut Criterion) {
    let mut group = c.benchmark_group("fifo_worst_case");
    group.sample_size(50); // Reduce sample size for expensive benchmarks

    // Worst case: many lots, each with very small quantity
    // Simulates a user who made many small purchases
    let lots: Vec<TaxLot> = (0..1000)
        .map(|i| {
            let qty = Decimal::from_str("0.001").unwrap(); // Small amounts
            TaxLot {
                id: format!("lot{}", i),
                transaction_id: format!("buy{}", i),
                asset: "BTC".to_string(),
                quantity: qty,
                remaining_quantity: qty,
                cost_basis: Decimal::from_str(&format!("{}", 50 + i)).unwrap(),
                acquisition_date: Utc.with_ymd_and_hms(2023, 1, (i % 28) + 1, 0, 0, 0).unwrap(),
            }
        })
        .collect();

    let sale = create_test_sale("0.5"); // Selling 0.5 BTC from 1000 tiny lots

    group.bench_function("1000_tiny_lots", |b| {
        b.iter(|| {
            let lots_clone = lots.clone();
            black_box(calculate_fifo_disposal(&sale, lots_clone).unwrap())
        });
    });

    group.finish();
}

fn bench_fifo_sorting_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("fifo_sorting");

    // Test with unsorted lots (reverse chronological order)
    let mut lots = create_test_lots(500);
    lots.reverse(); // Worst case for sorting

    let sale = create_test_sale("250.0");

    group.bench_function("500_lots_reverse_sorted", |b| {
        b.iter(|| {
            let lots_clone = lots.clone();
            black_box(calculate_fifo_disposal(&sale, lots_clone).unwrap())
        });
    });

    group.finish();
}

fn bench_fifo_realistic_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("fifo_realistic");

    // Scenario 1: Typical retail investor
    // 50 purchases over 2 years, selling 25%
    let sale1 = create_test_sale("12.5");
    let lots1 = create_test_lots(50);

    group.bench_function("retail_investor_50_lots", |b| {
        b.iter(|| {
            let lots_clone = lots1.clone();
            black_box(calculate_fifo_disposal(&sale1, lots_clone).unwrap())
        });
    });

    // Scenario 2: Active trader
    // 200 purchases, selling 10%
    let sale2 = create_test_sale("20.0");
    let lots2 = create_test_lots(200);

    group.bench_function("active_trader_200_lots", |b| {
        b.iter(|| {
            let lots_clone = lots2.clone();
            black_box(calculate_fifo_disposal(&sale2, lots_clone).unwrap())
        });
    });

    // Scenario 3: DCA investor
    // 100 regular purchases, selling half
    let sale3 = create_test_sale("50.0");
    let lots3 = create_test_lots(100);

    group.bench_function("dca_investor_100_lots", |b| {
        b.iter(|| {
            let lots_clone = lots3.clone();
            black_box(calculate_fifo_disposal(&sale3, lots_clone).unwrap())
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_fifo_small_dataset,
    bench_fifo_medium_dataset,
    bench_fifo_large_dataset,
    bench_fifo_full_disposal,
    bench_fifo_partial_disposal,
    bench_fifo_single_lot_scenarios,
    bench_fifo_worst_case,
    bench_fifo_sorting_overhead,
    bench_fifo_realistic_scenarios,
);

criterion_main!(benches);
