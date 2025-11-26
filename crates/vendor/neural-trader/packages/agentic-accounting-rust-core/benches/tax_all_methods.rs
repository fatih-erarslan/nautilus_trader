/*!
 * Comprehensive benchmarks for all tax calculation methods
 *
 * Tests performance across varying lot counts: 10, 100, 1000, 10000
 * Target: <10ms for 1000 lots
 */

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use chrono::{Utc, Duration};
use agentic_accounting_rust_core::types::TaxLot;
use agentic_accounting_rust_core::tax::{fifo, lifo, hifo, average_cost};

fn generate_test_lots(count: usize) -> Vec<TaxLot> {
    let base_date = Utc::now() - Duration::days(400);
    (0..count)
        .map(|i| {
            let cost = 40000 + (i as i64 * 100);
            TaxLot {
                id: format!("lot_{}", i),
                transaction_id: format!("tx_{}", i),
                asset: "BTC".to_string(),
                quantity: dec!(1.0),
                remaining_quantity: dec!(1.0),
                cost_basis: Decimal::new(cost, 0),
                acquisition_date: base_date + Duration::days(i as i64),
            }
        })
        .collect()
}

fn benchmark_fifo(c: &mut Criterion) {
    let mut group = c.benchmark_group("fifo");
    
    for lot_count in [10, 100, 1000, 10000].iter() {
        let lots = generate_test_lots(*lot_count);
        let sale_quantity = dec!(100.0);
        
        group.bench_with_input(
            BenchmarkId::from_parameter(lot_count),
            lot_count,
            |b, _| {
                b.iter(|| {
                    fifo::calculate_fifo(
                        black_box(&lots),
                        black_box(sale_quantity),
                        black_box(dec!(70000)),
                        black_box(Utc::now()),
                        black_box("sale1"),
                        black_box("BTC"),
                    )
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_lifo(c: &mut Criterion) {
    let mut group = c.benchmark_group("lifo");
    
    for lot_count in [10, 100, 1000, 10000].iter() {
        let lots = generate_test_lots(*lot_count);
        let sale_quantity = dec!(100.0);
        
        group.bench_with_input(
            BenchmarkId::from_parameter(lot_count),
            lot_count,
            |b, _| {
                b.iter(|| {
                    lifo::calculate_lifo(
                        black_box(&lots),
                        black_box(sale_quantity),
                        black_box(dec!(70000)),
                        black_box(Utc::now()),
                        black_box("sale1"),
                        black_box("BTC"),
                    )
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_hifo(c: &mut Criterion) {
    let mut group = c.benchmark_group("hifo");
    
    for lot_count in [10, 100, 1000, 10000].iter() {
        let lots = generate_test_lots(*lot_count);
        let sale_quantity = dec!(100.0);
        
        group.bench_with_input(
            BenchmarkId::from_parameter(lot_count),
            lot_count,
            |b, _| {
                b.iter(|| {
                    hifo::calculate_hifo(
                        black_box(&lots),
                        black_box(sale_quantity),
                        black_box(dec!(70000)),
                        black_box(Utc::now()),
                        black_box("sale1"),
                        black_box("BTC"),
                    )
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_average_cost(c: &mut Criterion) {
    let mut group = c.benchmark_group("average_cost");
    
    for lot_count in [10, 100, 1000, 10000].iter() {
        let lots = generate_test_lots(*lot_count);
        let sale_quantity = dec!(100.0);
        
        group.bench_with_input(
            BenchmarkId::from_parameter(lot_count),
            lot_count,
            |b, _| {
                b.iter(|| {
                    average_cost::calculate_average_cost(
                        black_box(&lots),
                        black_box(sale_quantity),
                        black_box(dec!(70000)),
                        black_box(Utc::now()),
                        black_box("sale1"),
                        black_box("BTC"),
                    )
                })
            },
        );
    }
    
    group.finish();
}

// Benchmark memory allocation patterns
fn benchmark_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");
    
    group.bench_function("create_1000_lots", |b| {
        b.iter(|| {
            black_box(generate_test_lots(1000))
        })
    });
    
    group.finish();
}

// Benchmark decimal operations (critical path)
fn benchmark_decimal_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("decimal_operations");
    
    let a = dec!(50000.123456789);
    let b = dec!(1.234567890);
    
    group.bench_function("multiply", |b| {
        b.iter(|| {
            black_box(a) * black_box(b)
        })
    });
    
    group.bench_function("divide", |b| {
        b.iter(|| {
            black_box(a) / black_box(b)
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_fifo,
    benchmark_lifo,
    benchmark_hifo,
    benchmark_average_cost,
    benchmark_memory_allocation,
    benchmark_decimal_operations
);

criterion_main!(benches);
