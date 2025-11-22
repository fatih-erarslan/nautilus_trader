//! Basic Order Matching Engine Tests
//!
//! Focused tests for order matching engine core functionality.

use crossbeam::queue::SegQueue;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::algorithms::order_matching::*;

/// Test 1: Basic Order Creation and Atomic Operations
#[test]
fn test_basic_order_creation() {
    let order = Order::new(
        1,
        "BTCUSD".to_string(),
        Side::Buy,
        OrderType::Limit,
        TimeInForce::GoodTillCancel,
        50000_000000, // $50,000
        1_000000,     // 1 BTC
        1000000000,   // timestamp
    );

    assert_eq!(order.order_id, 1);
    assert_eq!(order.side, Side::Buy);
    assert_eq!(order.price, 50000_000000);
    assert_eq!(order.original_quantity, 1_000000);
    assert_eq!(order.remaining_quantity.load(Ordering::Acquire), 1_000000);
    assert_eq!(order.filled_quantity.load(Ordering::Acquire), 0);
    assert_eq!(order.get_status(), OrderStatus::New);
}

/// Test 2: Atomic Order Fill Operations
#[test]
fn test_atomic_order_fill() {
    let order = Order::new(
        1,
        "BTCUSD".to_string(),
        Side::Sell,
        OrderType::Limit,
        TimeInForce::GoodTillCancel,
        50000_000000,
        5_000000, // 5 BTC
        1000000000,
    );

    // Fill 2 BTC
    let fill_result = order.try_fill(2_000000, 50000_000000);
    assert!(fill_result.is_ok());
    assert_eq!(fill_result.unwrap(), 2_000000);

    // Check remaining quantity
    assert_eq!(order.remaining_quantity.load(Ordering::Acquire), 3_000000);
    assert_eq!(order.filled_quantity.load(Ordering::Acquire), 2_000000);
    assert_eq!(order.get_status(), OrderStatus::PartiallyFilled);

    // Fill remaining quantity
    let fill_result2 = order.try_fill(3_000000, 50001_000000);
    assert!(fill_result2.is_ok());
    assert_eq!(fill_result2.unwrap(), 3_000000);

    // Check final state
    assert_eq!(order.remaining_quantity.load(Ordering::Acquire), 0);
    assert_eq!(order.filled_quantity.load(Ordering::Acquire), 5_000000);
    assert_eq!(order.get_status(), OrderStatus::Filled);
}

/// Test 3: Price Level Operations
#[test]
fn test_price_level_operations() {
    let price_level = PriceLevel::new(50000_000000);

    // Create test orders
    let order1 = Box::into_raw(Box::new(Order::new(
        1,
        "BTCUSD".to_string(),
        Side::Buy,
        OrderType::Limit,
        TimeInForce::GoodTillCancel,
        50000_000000,
        1_000000,
        1000000000,
    )));

    let order2 = Box::into_raw(Box::new(Order::new(
        2,
        "BTCUSD".to_string(),
        Side::Buy,
        OrderType::Limit,
        TimeInForce::GoodTillCancel,
        50000_000000,
        2_000000,
        1000000001,
    )));

    // Insert orders
    assert!(price_level.insert_order(order1));
    assert!(price_level.insert_order(order2));

    // Check quantities
    assert_eq!(price_level.get_total_quantity(), 3_000000);
    assert_eq!(price_level.get_displayed_quantity(), 3_000000);
    assert_eq!(price_level.get_order_count(), 2);

    // Clean up
    unsafe {
        let _ = Box::from_raw(order1);
        let _ = Box::from_raw(order2);
    }
}

/// Test 4: Concurrent Order Fill Test
#[test]
fn test_concurrent_order_fill() {
    let order = Arc::new(Order::new(
        1,
        "BTCUSD".to_string(),
        Side::Sell,
        OrderType::Limit,
        TimeInForce::GoodTillCancel,
        50000_000000,
        1000_000000, // 1000 BTC - large enough for concurrent access
        1000000000,
    ));

    let num_threads = 10;
    let fills_per_thread = 100;
    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = vec![];

    let successful_fills = Arc::new(AtomicU64::new(0));

    for thread_id in 0..num_threads {
        let order = order.clone();
        let barrier = barrier.clone();
        let successful_fills = successful_fills.clone();

        let handle = thread::spawn(move || {
            barrier.wait(); // Synchronize start

            for i in 0..fills_per_thread {
                let fill_qty = 1_000000; // 1 BTC per fill
                let fill_price = 50000_000000 + (thread_id * 1000) + i;

                match order.try_fill(fill_qty, fill_price) {
                    Ok(filled) => {
                        assert_eq!(filled, fill_qty);
                        successful_fills.fetch_add(1, Ordering::AcqRel);
                    }
                    Err(_) => {
                        // Order exhausted - expected for some threads
                        break;
                    }
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    let total_fills = successful_fills.load(Ordering::Acquire);
    assert!(total_fills > 0);
    assert!(total_fills <= 1000); // Cannot fill more than original quantity

    // Verify final state consistency
    let remaining = order.remaining_quantity.load(Ordering::Acquire);
    let filled = order.filled_quantity.load(Ordering::Acquire);
    assert_eq!(order.original_quantity, remaining + filled);

    println!("Concurrent fill test completed:");
    println!("  Successful fills: {}", total_fills);
    println!("  Final remaining: {}", remaining);
    println!("  Final filled: {}", filled);
}

/// Test 5: Lock-Free Queue Operations
#[test]
fn test_lockfree_queue() {
    let queue = Arc::new(SegQueue::new());
    let num_producers = 4;
    let num_consumers = 2;
    let items_per_producer = 1000;

    let barrier = Arc::new(Barrier::new(num_producers + num_consumers));
    let mut handles = vec![];

    let produced_count = Arc::new(AtomicU64::new(0));
    let consumed_count = Arc::new(AtomicU64::new(0));

    // Producer threads
    for producer_id in 0..num_producers {
        let queue = queue.clone();
        let barrier = barrier.clone();
        let produced_count = produced_count.clone();

        let handle = thread::spawn(move || {
            barrier.wait();

            for i in 0..items_per_producer {
                let trade = Trade {
                    trade_id: (producer_id as u64 * 10000) + i as u64,
                    buy_order_id: 1,
                    sell_order_id: 2,
                    symbol: format!("SYM{}", producer_id),
                    price: 50000_000000 + i as u64,
                    quantity: 1_000000,
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_nanos() as u64,
                    aggressor_side: Side::Buy,
                };

                queue.push(trade);
                produced_count.fetch_add(1, Ordering::AcqRel);
            }
        });

        handles.push(handle);
    }

    // Consumer threads
    for _consumer_id in 0..num_consumers {
        let queue = queue.clone();
        let barrier = barrier.clone();
        let consumed_count = consumed_count.clone();
        let produced_count = produced_count.clone();

        let handle = thread::spawn(move || {
            barrier.wait();

            let total_expected = (num_producers * items_per_producer) as u64;

            loop {
                match queue.pop() {
                    Some(trade) => {
                        consumed_count.fetch_add(1, Ordering::AcqRel);

                        // Validate trade
                        assert!(trade.trade_id > 0);
                        assert!(trade.price > 0);
                        assert!(trade.quantity > 0);
                        assert!(!trade.symbol.is_empty());
                    }
                    None => {
                        let total_produced = produced_count.load(Ordering::Acquire);
                        let total_consumed = consumed_count.load(Ordering::Acquire);

                        if total_produced >= total_expected && total_consumed >= total_expected {
                            break;
                        }

                        thread::sleep(Duration::from_micros(1));
                    }
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    let final_produced = produced_count.load(Ordering::Acquire);
    let final_consumed = consumed_count.load(Ordering::Acquire);

    assert_eq!(final_produced, (num_producers * items_per_producer) as u64);
    assert_eq!(final_consumed, final_produced);
    assert!(queue.is_empty());

    println!("Lock-free queue test completed:");
    println!("  Items produced: {}", final_produced);
    println!("  Items consumed: {}", final_consumed);
}

/// Test 6: Performance Benchmark
#[test]
fn test_performance_benchmark() {
    let num_operations = 10000;
    let mut latencies = Vec::with_capacity(num_operations);

    for i in 0..num_operations {
        let order = Order::new(
            i as u64,
            "BTCUSD".to_string(),
            Side::Buy,
            OrderType::Limit,
            TimeInForce::GoodTillCancel,
            50000_000000,
            1_000000,
            1000000000 + i as u64,
        );

        let start = Instant::now();

        // Simulate order processing work
        let _remaining = order.remaining_quantity.load(Ordering::Acquire);
        let _filled = order.filled_quantity.load(Ordering::Acquire);
        let _status = order.get_status();

        let latency = start.elapsed().as_nanos() as u64;
        latencies.push(latency);
    }

    // Calculate statistics
    latencies.sort_unstable();
    let min_latency = latencies[0];
    let max_latency = latencies[latencies.len() - 1];
    let median_latency = latencies[latencies.len() / 2];
    let p99_latency = latencies[(latencies.len() * 99) / 100];
    let avg_latency = latencies.iter().sum::<u64>() / latencies.len() as u64;

    // Performance assertions - should be very fast for basic operations
    assert!(
        avg_latency < 10_000,
        "Average latency too high: {} ns",
        avg_latency
    );
    assert!(
        median_latency < 5_000,
        "Median latency too high: {} ns",
        median_latency
    );

    println!("Performance benchmark completed:");
    println!("  Operations: {}", num_operations);
    println!("  Min latency: {} ns", min_latency);
    println!("  Median latency: {} ns", median_latency);
    println!("  Average latency: {} ns", avg_latency);
    println!("  P99 latency: {} ns", p99_latency);
    println!("  Max latency: {} ns", max_latency);
}

/// Test 7: Memory Ordering Validation
#[test]
fn test_memory_ordering_validation() {
    let order = Arc::new(Order::new(
        1,
        "BTCUSD".to_string(),
        Side::Sell,
        OrderType::Limit,
        TimeInForce::GoodTillCancel,
        50000_000000,
        1000_000000, // Large quantity
        1000000000,
    ));

    let num_threads = 8;
    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = vec![];

    let memory_errors = Arc::new(AtomicU64::new(0));

    for thread_id in 0..num_threads {
        let order = order.clone();
        let barrier = barrier.clone();
        let memory_errors = memory_errors.clone();

        let handle = thread::spawn(move || {
            barrier.wait();

            for i in 0..1000 {
                let fill_qty = 1000; // Small fills
                let fill_price = 50000_000000 + (thread_id * 1000) + i;

                if let Ok(_filled) = order.try_fill(fill_qty, fill_price) {
                    // Immediately check consistency with acquire ordering
                    let remaining = order.remaining_quantity.load(Ordering::Acquire);
                    let total_filled = order.filled_quantity.load(Ordering::Acquire);

                    // Memory fence to ensure visibility
                    std::sync::atomic::fence(Ordering::SeqCst);

                    // Re-read with acquire to check for memory ordering issues
                    let remaining_check = order.remaining_quantity.load(Ordering::Acquire);
                    let filled_check = order.filled_quantity.load(Ordering::Acquire);

                    // Values should be consistent
                    if remaining != remaining_check || total_filled != filled_check {
                        memory_errors.fetch_add(1, Ordering::AcqRel);
                    }

                    // Check invariant
                    if order.original_quantity != remaining + total_filled {
                        memory_errors.fetch_add(1, Ordering::AcqRel);
                    }
                }
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let errors = memory_errors.load(Ordering::Acquire);
    assert_eq!(errors, 0, "Memory ordering violations detected: {}", errors);

    // Final consistency check
    let final_remaining = order.remaining_quantity.load(Ordering::Acquire);
    let final_filled = order.filled_quantity.load(Ordering::Acquire);
    assert_eq!(order.original_quantity, final_remaining + final_filled);

    println!("Memory ordering validation completed:");
    println!("  Final remaining: {}", final_remaining);
    println!("  Final filled: {}", final_filled);
    println!("  Memory errors: {}", errors);
}

/// Test 8: Iceberg Order Behavior
#[test]
fn test_iceberg_order_basic() {
    let iceberg_order = Order::new(
        1,
        "BTCUSD".to_string(),
        Side::Sell,
        OrderType::Iceberg,
        TimeInForce::GoodTillCancel,
        50000_000000,
        10_000000, // 10 BTC total
        1000000000,
    );

    // Iceberg should display only part of the quantity initially
    let displayed = iceberg_order.displayed_quantity.load(Ordering::Acquire);
    assert!(displayed < iceberg_order.original_quantity);
    assert!(displayed > 0);
    assert_eq!(
        iceberg_order.hidden_quantity,
        iceberg_order.original_quantity - displayed
    );

    println!("Iceberg order test:");
    println!("  Total quantity: {}", iceberg_order.original_quantity);
    println!("  Displayed quantity: {}", displayed);
    println!("  Hidden quantity: {}", iceberg_order.hidden_quantity);
}

/// Test 9: Order Status Transitions
#[test]
fn test_order_status_transitions() {
    let order = Order::new(
        1,
        "BTCUSD".to_string(),
        Side::Buy,
        OrderType::Limit,
        TimeInForce::GoodTillCancel,
        50000_000000,
        5_000000,
        1000000000,
    );

    // Initial status
    assert_eq!(order.get_status(), OrderStatus::New);

    // Partial fill
    let _ = order.try_fill(2_000000, 50000_000000);
    assert_eq!(order.get_status(), OrderStatus::PartiallyFilled);

    // Complete fill
    let _ = order.try_fill(3_000000, 50000_000000);
    assert_eq!(order.get_status(), OrderStatus::Filled);

    // Try to cancel filled order
    let cancelled = order.set_status(OrderStatus::Cancelled);
    assert!(cancelled); // Status change should succeed
    assert_eq!(order.get_status(), OrderStatus::Cancelled);
}

/// Test 10: High-Frequency Operations
#[test]
fn test_high_frequency_operations() {
    let order = Arc::new(Order::new(
        1,
        "BTCUSD".to_string(),
        Side::Sell,
        OrderType::Limit,
        TimeInForce::GoodTillCancel,
        50000_000000,
        100_000_000, // Very large quantity
        1000000000,
    ));

    let num_threads = 16;
    let operations_per_thread = 10000;
    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = vec![];

    let total_operations = Arc::new(AtomicU64::new(0));
    let start_time = Instant::now();

    for thread_id in 0..num_threads {
        let order = order.clone();
        let barrier = barrier.clone();
        let total_operations = total_operations.clone();

        let handle = thread::spawn(move || {
            barrier.wait();

            for i in 0..operations_per_thread {
                // Mix of operations
                match i % 4 {
                    0 => {
                        // Try fill
                        let _ = order.try_fill(1, 50000_000000 + (thread_id * 1000) + i);
                    }
                    1 => {
                        // Read remaining quantity
                        let _ = order.remaining_quantity.load(Ordering::Acquire);
                    }
                    2 => {
                        // Read filled quantity
                        let _ = order.filled_quantity.load(Ordering::Acquire);
                    }
                    3 => {
                        // Check status
                        let _ = order.get_status();
                    }
                    _ => unreachable!(),
                }

                total_operations.fetch_add(1, Ordering::AcqRel);
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let elapsed = start_time.elapsed();
    let ops_count = total_operations.load(Ordering::Acquire);
    let ops_per_second = (ops_count as f64) / elapsed.as_secs_f64();

    println!("High-frequency operations test:");
    println!("  Total operations: {}", ops_count);
    println!("  Elapsed time: {:?}", elapsed);
    println!("  Operations per second: {:.0}", ops_per_second);

    assert!(
        ops_per_second > 1_000_000.0,
        "Operations per second too low: {}",
        ops_per_second
    );
}
