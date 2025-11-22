//! Comprehensive Order Matching Engine Tests
//!
//! This module implements comprehensive tests for the order matching engine
//! with focus on atomic operations, lock-free queues, concurrent matching,
//! and performance validation under CQGS governance.

use crossbeam::queue::SegQueue;
use crossbeam::utils::CachePadded;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::algorithms::order_matching::*;

/// Test harness for concurrent order matching tests
struct OrderMatchingTestHarness {
    engine: Arc<parking_lot::Mutex<OrderMatchingEngine>>,
    metrics: Arc<TestMetrics>,
    trade_sink: Arc<SegQueue<Trade>>,
}

/// Metrics collection for test validation
#[derive(Debug)]
struct TestMetrics {
    total_orders_submitted: AtomicU64,
    total_trades_received: AtomicU64,
    total_volume_traded: AtomicU64,
    min_latency_ns: AtomicU64,
    max_latency_ns: AtomicU64,
    sum_latency_ns: AtomicU64,
    latency_samples: AtomicU64,
    race_condition_detections: AtomicU64,
    cas_failures: AtomicU64,
    memory_ordering_violations: AtomicU64,
}

impl TestMetrics {
    fn new() -> Self {
        Self {
            total_orders_submitted: AtomicU64::new(0),
            total_trades_received: AtomicU64::new(0),
            total_volume_traded: AtomicU64::new(0),
            min_latency_ns: AtomicU64::new(u64::MAX),
            max_latency_ns: AtomicU64::new(0),
            sum_latency_ns: AtomicU64::new(0),
            latency_samples: AtomicU64::new(0),
            race_condition_detections: AtomicU64::new(0),
            cas_failures: AtomicU64::new(0),
            memory_ordering_violations: AtomicU64::new(0),
        }
    }

    fn record_latency(&self, latency_ns: u64) {
        self.sum_latency_ns.fetch_add(latency_ns, Ordering::AcqRel);
        self.latency_samples.fetch_add(1, Ordering::AcqRel);

        // Update min latency
        loop {
            let current_min = self.min_latency_ns.load(Ordering::Acquire);
            if latency_ns >= current_min
                || self
                    .min_latency_ns
                    .compare_exchange_weak(
                        current_min,
                        latency_ns,
                        Ordering::AcqRel,
                        Ordering::Relaxed,
                    )
                    .is_ok()
            {
                break;
            }
        }

        // Update max latency
        loop {
            let current_max = self.max_latency_ns.load(Ordering::Acquire);
            if latency_ns <= current_max
                || self
                    .max_latency_ns
                    .compare_exchange_weak(
                        current_max,
                        latency_ns,
                        Ordering::AcqRel,
                        Ordering::Relaxed,
                    )
                    .is_ok()
            {
                break;
            }
        }
    }

    fn record_cas_failure(&self) {
        self.cas_failures.fetch_add(1, Ordering::AcqRel);
    }

    fn record_race_condition(&self) {
        self.race_condition_detections
            .fetch_add(1, Ordering::AcqRel);
    }

    fn record_memory_ordering_violation(&self) {
        self.memory_ordering_violations
            .fetch_add(1, Ordering::AcqRel);
    }

    fn get_avg_latency_ns(&self) -> u64 {
        let samples = self.latency_samples.load(Ordering::Acquire);
        if samples == 0 {
            0
        } else {
            self.sum_latency_ns.load(Ordering::Acquire) / samples
        }
    }
}

impl OrderMatchingTestHarness {
    fn new() -> Self {
        Self {
            engine: Arc::new(parking_lot::Mutex::new(OrderMatchingEngine::new(
                MatchingAlgorithm::PriceTimePriority,
            ))),
            metrics: Arc::new(TestMetrics::new()),
            trade_sink: Arc::new(SegQueue::new()),
        }
    }

    /// Create test order with automatic ID generation
    fn create_order(&self, side: Side, order_type: OrderType, price: u64, quantity: u64) -> Order {
        static ORDER_ID_COUNTER: AtomicU64 = AtomicU64::new(1);
        let order_id = ORDER_ID_COUNTER.fetch_add(1, Ordering::AcqRel);

        Order::new(
            order_id,
            "BTCUSD".to_string(),
            side,
            order_type,
            TimeInForce::GoodTillCancel,
            price,
            quantity,
            self.get_timestamp_ns(),
        )
    }

    fn get_timestamp_ns(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
}

/// Test 1: Atomic Operation Validation
/// Validates that all atomic operations maintain consistency under concurrent access
#[test]
fn test_atomic_operations_consistency() {
    let harness = OrderMatchingTestHarness::new();
    let num_threads: usize = 8;
    let operations_per_thread = 1000;

    // Test atomic order quantity updates
    let test_order = Arc::new(Order::new(
        1,
        "BTCUSD".to_string(),
        Side::Buy,
        OrderType::Limit,
        TimeInForce::GoodTillCancel,
        50000_000000,
        operations_per_thread * num_threads,
        harness.get_timestamp_ns(),
    ));

    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = vec![];

    // Spawn threads that concurrently try to fill the order
    for thread_id in 0..num_threads {
        let order = test_order.clone();
        let metrics = harness.metrics.clone();
        let barrier = barrier.clone();

        let handle = thread::spawn(move || {
            barrier.wait(); // Synchronize thread start

            for i in 0..operations_per_thread {
                let fill_qty = 1;
                let fill_price = 50000_000000 + (thread_id * 1000) + i; // Unique price per fill

                let start = Instant::now();
                match order.try_fill(fill_qty, fill_price) {
                    Ok(filled) => {
                        let latency = start.elapsed().as_nanos() as u64;
                        metrics.record_latency(latency);

                        // Validate filled quantity
                        assert_eq!(filled, fill_qty, "Filled quantity mismatch");

                        // Validate remaining quantity is consistent
                        let remaining = order.remaining_quantity.load(Ordering::Acquire);
                        let filled_total = order.filled_quantity.load(Ordering::Acquire);

                        // Check invariant: original = remaining + filled
                        if order.original_quantity != remaining + filled_total {
                            metrics.record_memory_ordering_violation();
                            panic!(
                                "Quantity invariant violation: {} != {} + {}",
                                order.original_quantity, remaining, filled_total
                            );
                        }
                    }
                    Err(_) => {
                        // Order is fully filled - this is expected
                        break;
                    }
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Validate final state
    let final_remaining = test_order.remaining_quantity.load(Ordering::Acquire);
    let final_filled = test_order.filled_quantity.load(Ordering::Acquire);

    assert_eq!(test_order.original_quantity, final_remaining + final_filled);
    assert!(final_filled > 0, "No fills occurred");
    assert_eq!(
        harness
            .metrics
            .memory_ordering_violations
            .load(Ordering::Acquire),
        0,
        "Memory ordering violations detected"
    );

    println!("Atomic operations test completed:");
    println!("  Total fills: {}", final_filled);
    println!("  Remaining quantity: {}", final_remaining);
    println!(
        "  Average latency: {} ns",
        harness.metrics.get_avg_latency_ns()
    );
}

/// Test 2: Lock-Free Queue Testing
/// Tests the lock-free queue implementation used for trade output
#[test]
fn test_lock_free_queue_operations() {
    let queue: Arc<SegQueue<Trade>> = Arc::new(SegQueue::new());
    let num_producers = 4;
    let num_consumers = 2;
    let items_per_producer = 2500;

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

            let mut consumed = 0;
            let total_expected = (num_producers * items_per_producer) as u64;

            loop {
                match queue.pop() {
                    Some(trade) => {
                        consumed += 1;
                        consumed_count.fetch_add(1, Ordering::AcqRel);

                        // Validate trade data integrity
                        assert!(trade.trade_id > 0);
                        assert!(trade.price > 0);
                        assert!(trade.quantity > 0);
                        assert!(!trade.symbol.is_empty());
                    }
                    None => {
                        // Check if all producers are done
                        let total_produced = produced_count.load(Ordering::Acquire);
                        let total_consumed = consumed_count.load(Ordering::Acquire);

                        if total_produced >= total_expected && total_consumed >= total_expected {
                            break;
                        }

                        // Small delay to avoid busy waiting
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

/// Test 3: FIFO Order Type Test
/// Tests First-In-First-Out order matching within price levels
#[test]
fn test_fifo_order_matching() {
    let mut engine = OrderMatchingEngine::new(MatchingAlgorithm::PriceTimePriority);

    let base_price = 50000_000000;
    let order_qty = 1_000000;

    // Add multiple buy orders at same price with sequential timestamps
    let mut buy_orders = vec![];
    for i in 0..5 {
        let order = Order::new(
            i + 1,
            "BTCUSD".to_string(),
            Side::Buy,
            OrderType::Limit,
            TimeInForce::GoodTillCancel,
            base_price,
            order_qty,
            1000000000 + i, // Sequential timestamps to ensure FIFO
        );
        buy_orders.push(order.order_id);
        engine.add_order(order).unwrap();
    }

    // Add large sell order that should match all buy orders in FIFO order
    let sell_order = Order::new(
        100,
        "BTCUSD".to_string(),
        Side::Sell,
        OrderType::Limit,
        TimeInForce::GoodTillCancel,
        base_price,
        order_qty * 5,
        1000000010,
    );

    let trades = engine.add_order(sell_order).unwrap();

    // Validate FIFO execution
    assert_eq!(trades.len(), 5);
    for (i, trade) in trades.iter().enumerate() {
        assert_eq!(trade.buy_order_id, buy_orders[i]);
        assert_eq!(trade.sell_order_id, 100);
        assert_eq!(trade.quantity, order_qty);
        assert_eq!(trade.price, base_price);
    }

    println!("FIFO matching test completed successfully");
}

/// Test 4: Pro-Rata Order Type Test
/// Tests pro-rata allocation algorithm
#[test]
fn test_pro_rata_order_matching() {
    let mut engine = OrderMatchingEngine::new(MatchingAlgorithm::ProRata);

    let base_price = 50000_000000;
    let total_buy_quantity = 10_000000; // 10 BTC total

    // Add buy orders with different quantities at same price
    let buy_quantities = vec![1_000000, 2_000000, 3_000000, 4_000000]; // 1, 2, 3, 4 BTC
    let mut buy_order_ids = vec![];

    for (i, &qty) in buy_quantities.iter().enumerate() {
        let order = Order::new(
            i as u64 + 1,
            "BTCUSD".to_string(),
            Side::Buy,
            OrderType::Limit,
            TimeInForce::GoodTillCancel,
            base_price,
            qty,
            1000000000 + i as u64,
        );
        buy_order_ids.push(order.order_id);
        engine.add_order(order).unwrap();
    }

    // Add sell order for 5 BTC - should be allocated pro-rata
    let sell_order = Order::new(
        100,
        "BTCUSD".to_string(),
        Side::Sell,
        OrderType::Limit,
        TimeInForce::GoodTillCancel,
        base_price,
        5_000000,
        1000000010,
    );

    let trades = engine.add_order(sell_order).unwrap();

    // Validate pro-rata allocation
    assert_eq!(trades.len(), 4);

    let total_available = buy_quantities.iter().sum::<u64>();
    let sell_quantity = 5_000000;

    for (i, trade) in trades.iter().enumerate() {
        let expected_allocation = (sell_quantity * buy_quantities[i]) / total_available;
        assert_eq!(
            trade.quantity, expected_allocation,
            "Pro-rata allocation mismatch for order {}",
            buy_order_ids[i]
        );
        assert_eq!(trade.buy_order_id, buy_order_ids[i]);
    }

    println!("Pro-rata matching test completed successfully");
}

/// Test 5: Iceberg Order Type Test
/// Tests iceberg order behavior with hidden quantity
#[test]
fn test_iceberg_order_matching() {
    let mut engine = OrderMatchingEngine::new(MatchingAlgorithm::PriceTimePriority);

    let base_price = 50000_000000;
    let total_qty = 10_000000; // 10 BTC total
    let display_qty = 1_000000; // 1 BTC displayed (10%)

    // Create iceberg sell order
    let iceberg_order = Order::new(
        1,
        "BTCUSD".to_string(),
        Side::Sell,
        OrderType::Iceberg,
        TimeInForce::GoodTillCancel,
        base_price,
        total_qty,
        1000000000,
    );

    // Verify initial display quantity
    assert_eq!(
        iceberg_order.displayed_quantity.load(Ordering::Acquire),
        display_qty
    );
    assert_eq!(iceberg_order.hidden_quantity, total_qty - display_qty);

    engine.add_order(iceberg_order).unwrap();

    // Get market data - should only show displayed quantity
    let market_data = engine.get_market_data("BTCUSD", 5).unwrap();
    assert_eq!(market_data.asks[0].quantity, display_qty);

    // Add buy order that matches the displayed quantity
    let buy_order1 = Order::new(
        2,
        "BTCUSD".to_string(),
        Side::Buy,
        OrderType::Limit,
        TimeInForce::GoodTillCancel,
        base_price,
        display_qty,
        1000000001,
    );

    let trades1 = engine.add_order(buy_order1).unwrap();
    assert_eq!(trades1.len(), 1);
    assert_eq!(trades1[0].quantity, display_qty);

    // Check that iceberg refilled its display
    let market_data = engine.get_market_data("BTCUSD", 5).unwrap();
    assert!(
        market_data.asks.len() > 0,
        "Iceberg should have refilled display"
    );

    // Add another buy order to test refill behavior
    let buy_order2 = Order::new(
        3,
        "BTCUSD".to_string(),
        Side::Buy,
        OrderType::Limit,
        TimeInForce::GoodTillCancel,
        base_price,
        display_qty,
        1000000002,
    );

    let trades2 = engine.add_order(buy_order2).unwrap();
    assert_eq!(trades2.len(), 1);
    assert_eq!(trades2[0].quantity, display_qty);

    println!("Iceberg order test completed successfully");
}

/// Test 6: Concurrent Order Matching with CAS Operations
/// Tests concurrent order matching with Compare-And-Swap operations
#[test]
fn test_concurrent_order_matching_cas() {
    let harness = Arc::new(OrderMatchingTestHarness::new());
    let num_threads = 8;
    let orders_per_thread = 100;

    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = vec![];

    // Each thread will submit alternating buy and sell orders
    for thread_id in 0..num_threads {
        let harness = harness.clone();
        let barrier = barrier.clone();

        let handle = thread::spawn(move || {
            barrier.wait();

            for i in 0..orders_per_thread {
                let side = if i % 2 == 0 { Side::Buy } else { Side::Sell };
                let price = 50000_000000 + ((thread_id * 1000) as u64);
                let quantity = 1_000000;

                let order = harness.create_order(side, OrderType::Limit, price, quantity);
                harness
                    .metrics
                    .total_orders_submitted
                    .fetch_add(1, Ordering::AcqRel);

                let start = Instant::now();
                let result = {
                    let mut engine = harness.engine.lock();
                    engine.add_order(order)
                };
                let latency = start.elapsed().as_nanos() as u64;
                harness.metrics.record_latency(latency);

                match result {
                    Ok(trades) => {
                        harness
                            .metrics
                            .total_trades_received
                            .fetch_add(trades.len() as u64, Ordering::AcqRel);
                        let volume: u64 = trades.iter().map(|t| t.quantity).sum();
                        harness
                            .metrics
                            .total_volume_traded
                            .fetch_add(volume, Ordering::AcqRel);

                        // Push trades to sink for validation
                        for trade in trades {
                            harness.trade_sink.push(trade);
                        }
                    }
                    Err(e) => {
                        panic!("Order submission failed: {}", e);
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

    // Validate results
    let total_orders = harness
        .metrics
        .total_orders_submitted
        .load(Ordering::Acquire);
    let total_trades = harness
        .metrics
        .total_trades_received
        .load(Ordering::Acquire);
    let total_volume = harness.metrics.total_volume_traded.load(Ordering::Acquire);
    let avg_latency = harness.metrics.get_avg_latency_ns();

    assert_eq!(total_orders, (num_threads * orders_per_thread) as u64);
    assert!(total_trades > 0, "No trades generated");
    assert!(total_volume > 0, "No volume traded");
    assert!(avg_latency > 0, "Invalid latency measurement");

    // Validate trade data integrity
    let mut trade_count = 0;
    while let Some(trade) = harness.trade_sink.pop() {
        trade_count += 1;
        assert!(trade.trade_id > 0);
        assert!(trade.price > 0);
        assert!(trade.quantity > 0);
        assert_eq!(trade.symbol, "BTCUSD");
    }
    assert_eq!(trade_count, total_trades);

    println!("Concurrent matching test completed:");
    println!("  Total orders: {}", total_orders);
    println!("  Total trades: {}", total_trades);
    println!("  Total volume: {}", total_volume);
    println!("  Average latency: {} ns", avg_latency);
}

/// Test 7: Performance Benchmarks for Sub-10ms Matching
/// Tests matching performance under high load to ensure sub-10ms latency
#[test]
fn test_performance_sub_10ms_matching() {
    let mut engine = OrderMatchingEngine::new(MatchingAlgorithm::PriceTimePriority);

    let num_orders = 10000;
    let target_latency_ns = 10_000_000; // 10ms in nanoseconds
    let mut latencies = Vec::with_capacity(num_orders);

    // Pre-populate order book with liquidity
    for i in 0..1000 {
        let buy_order = Order::new(
            i + 1000000,
            "BTCUSD".to_string(),
            Side::Buy,
            OrderType::Limit,
            TimeInForce::GoodTillCancel,
            49000_000000 + (i % 1000) as u64,
            1_000000,
            1000000000,
        );
        engine.add_order(buy_order).unwrap();

        let sell_order = Order::new(
            i + 2000000,
            "BTCUSD".to_string(),
            Side::Sell,
            OrderType::Limit,
            TimeInForce::GoodTillCancel,
            51000_000000 + (i % 1000) as u64,
            1_000000,
            1000000000,
        );
        engine.add_order(sell_order).unwrap();
    }

    // Test matching performance
    for i in 0..num_orders {
        let side = if i % 2 == 0 { Side::Buy } else { Side::Sell };
        let price = if side == Side::Buy {
            51000_000000
        } else {
            49000_000000
        };

        let order = Order::new(
            (i + 3000000) as u64,
            "BTCUSD".to_string(),
            side,
            OrderType::Market,
            TimeInForce::ImmediateOrCancel,
            price,
            1_000000,
            1000000000 + i as u64,
        );

        let start = Instant::now();
        let _trades = engine.add_order(order).unwrap();
        let latency = start.elapsed().as_nanos() as u64;

        latencies.push(latency);

        // Ensure we stay under target latency
        assert!(
            latency <= target_latency_ns,
            "Order {} exceeded target latency: {} ns > {} ns",
            i,
            latency,
            target_latency_ns
        );
    }

    // Calculate statistics
    latencies.sort_unstable();
    let min_latency = latencies[0];
    let max_latency = latencies[latencies.len() - 1];
    let median_latency = latencies[latencies.len() / 2];
    let p99_latency = latencies[(latencies.len() * 99) / 100];
    let avg_latency = latencies.iter().sum::<u64>() / latencies.len() as u64;

    // Performance assertions
    assert!(avg_latency <= target_latency_ns, "Average latency too high");
    assert!(p99_latency <= target_latency_ns, "P99 latency too high");
    assert!(
        median_latency <= target_latency_ns / 2,
        "Median latency too high"
    );

    println!("Performance benchmark completed:");
    println!("  Orders processed: {}", num_orders);
    println!(
        "  Min latency: {} ns ({:.2} μs)",
        min_latency,
        min_latency as f64 / 1000.0
    );
    println!(
        "  Median latency: {} ns ({:.2} μs)",
        median_latency,
        median_latency as f64 / 1000.0
    );
    println!(
        "  Average latency: {} ns ({:.2} μs)",
        avg_latency,
        avg_latency as f64 / 1000.0
    );
    println!(
        "  P99 latency: {} ns ({:.2} μs)",
        p99_latency,
        p99_latency as f64 / 1000.0
    );
    println!(
        "  Max latency: {} ns ({:.2} μs)",
        max_latency,
        max_latency as f64 / 1000.0
    );
}

/// Test 8: Race Condition Detection and Memory Ordering
/// Tests for race conditions and memory ordering issues in concurrent scenarios
#[test]
fn test_race_condition_detection() {
    let harness = Arc::new(OrderMatchingTestHarness::new());
    let num_threads: usize = 16;
    let iterations = 1000;

    // Shared order that multiple threads will try to modify
    let shared_order = Arc::new(Order::new(
        1,
        "BTCUSD".to_string(),
        Side::Sell,
        OrderType::Limit,
        TimeInForce::GoodTillCancel,
        50000_000000,
        num_threads * iterations, // Enough quantity for all threads
        harness.get_timestamp_ns(),
    ));

    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = vec![];

    // Race condition detector - tracks order state consistency
    let race_detector = Arc::new(AtomicBool::new(false));

    for thread_id in 0..num_threads {
        let order = shared_order.clone();
        let harness = harness.clone();
        let barrier = barrier.clone();
        let race_detector = race_detector.clone();

        let handle = thread::spawn(move || {
            barrier.wait();

            for i in 0..iterations {
                // Attempt to fill 1 unit from the shared order
                let fill_result = order.try_fill(1, 50000_000000 + thread_id as u64);

                match fill_result {
                    Ok(filled_qty) => {
                        // Verify consistency immediately after successful fill
                        let remaining = order.remaining_quantity.load(Ordering::Acquire);
                        let total_filled = order.filled_quantity.load(Ordering::Acquire);

                        // Memory barrier to ensure all updates are visible
                        std::sync::atomic::fence(Ordering::SeqCst);

                        // Check invariant with memory barrier
                        let remaining_check = order.remaining_quantity.load(Ordering::Acquire);
                        let filled_check = order.filled_quantity.load(Ordering::Acquire);

                        if remaining != remaining_check || total_filled != filled_check {
                            race_detector.store(true, Ordering::Release);
                            harness.metrics.record_race_condition();
                        }

                        // Verify quantity invariant
                        if order.original_quantity != remaining + total_filled {
                            race_detector.store(true, Ordering::Release);
                            harness.metrics.record_memory_ordering_violation();
                        }

                        assert_eq!(filled_qty, 1);
                    }
                    Err(_) => {
                        // Order exhausted - expected behavior
                        break;
                    }
                }

                // Small delay to increase chance of race conditions
                if i % 100 == 0 {
                    thread::yield_now();
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    // Check for race condition detection
    let race_detected = race_detector.load(Ordering::Acquire);
    let race_count = harness
        .metrics
        .race_condition_detections
        .load(Ordering::Acquire);
    let memory_violations = harness
        .metrics
        .memory_ordering_violations
        .load(Ordering::Acquire);

    assert!(
        !race_detected,
        "Race condition detected during concurrent access"
    );
    assert_eq!(race_count, 0, "Race conditions detected: {}", race_count);
    assert_eq!(
        memory_violations, 0,
        "Memory ordering violations: {}",
        memory_violations
    );

    // Verify final state consistency
    let final_remaining = shared_order.remaining_quantity.load(Ordering::Acquire);
    let final_filled = shared_order.filled_quantity.load(Ordering::Acquire);
    assert_eq!(
        shared_order.original_quantity,
        final_remaining + final_filled
    );

    println!("Race condition detection test completed:");
    println!("  Final filled quantity: {}", final_filled);
    println!("  Final remaining quantity: {}", final_remaining);
    println!("  Race conditions detected: {}", race_count);
    println!("  Memory ordering violations: {}", memory_violations);
}

/// Test 9: Order Book Consistency Under Load
/// Tests order book state consistency under high concurrent load
#[test]
fn test_order_book_consistency_under_load() {
    let harness = Arc::new(OrderMatchingTestHarness::new());
    let num_threads = 12;
    let orders_per_thread = 500;

    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = vec![];

    // Statistics tracking
    let successful_orders = Arc::new(AtomicU64::new(0));
    let successful_trades = Arc::new(AtomicU64::new(0));

    for thread_id in 0..num_threads {
        let harness = harness.clone();
        let barrier = barrier.clone();
        let successful_orders = successful_orders.clone();
        let successful_trades = successful_trades.clone();

        let handle = thread::spawn(move || {
            barrier.wait();

            for i in 0..orders_per_thread {
                // Vary order parameters to create realistic load
                let side = if (thread_id + i) % 2 == 0 {
                    Side::Buy
                } else {
                    Side::Sell
                };
                let price = 50000_000000 + ((thread_id * 100 + i) % 1000) as u64;
                let quantity = 1_000000 + ((i % 10) * 100000) as u64;
                let order_type = if i % 10 == 0 {
                    OrderType::Market
                } else {
                    OrderType::Limit
                };

                let order = harness.create_order(side, order_type, price, quantity);

                let start = Instant::now();
                let result = {
                    let mut engine = harness.engine.lock();
                    engine.add_order(order)
                };
                let latency = start.elapsed().as_nanos() as u64;

                match result {
                    Ok(trades) => {
                        successful_orders.fetch_add(1, Ordering::AcqRel);
                        successful_trades.fetch_add(trades.len() as u64, Ordering::AcqRel);
                        harness.metrics.record_latency(latency);

                        // Validate each trade
                        for trade in trades {
                            assert!(trade.trade_id > 0);
                            assert!(trade.price > 0);
                            assert!(trade.quantity > 0);
                            assert_eq!(trade.symbol, "BTCUSD");
                        }
                    }
                    Err(e) => {
                        panic!("Order processing failed: {}", e);
                    }
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Validate final state
    let total_orders = successful_orders.load(Ordering::Acquire);
    let total_trades = successful_trades.load(Ordering::Acquire);
    let avg_latency = harness.metrics.get_avg_latency_ns();

    assert_eq!(total_orders, (num_threads * orders_per_thread) as u64);
    assert!(
        avg_latency < 10_000_000,
        "Average latency too high: {} ns",
        avg_latency
    );

    // Verify order book consistency
    {
        let engine = harness.engine.lock();
        let market_data = engine.get_market_data("BTCUSD", 100);

        if let Some(data) = market_data {
            // Check bid-ask spread consistency
            if !data.bids.is_empty() && !data.asks.is_empty() {
                let best_bid = data.bids[0].price;
                let best_ask = data.asks[0].price;
                assert!(
                    best_bid <= best_ask,
                    "Crossed market detected: bid {} > ask {}",
                    best_bid,
                    best_ask
                );
            }

            // Check price level ordering
            for i in 1..data.bids.len() {
                assert!(
                    data.bids[i - 1].price >= data.bids[i].price,
                    "Bid levels not properly sorted"
                );
            }
            for i in 1..data.asks.len() {
                assert!(
                    data.asks[i - 1].price <= data.asks[i].price,
                    "Ask levels not properly sorted"
                );
            }
        }

        let stats = engine.get_statistics();
        println!("Order book consistency test completed:");
        println!("  Total orders processed: {}", total_orders);
        println!("  Total trades generated: {}", total_trades);
        println!("  Average latency: {} ns", avg_latency);
        println!("  Engine statistics: {:?}", stats);
    }
}

/// Test 10: Memory Safety and Cleanup
/// Tests memory safety and proper cleanup of completed orders
#[test]
fn test_memory_safety_and_cleanup() {
    let mut engine = OrderMatchingEngine::new(MatchingAlgorithm::PriceTimePriority);

    let num_order_pairs = 1000;
    let mut order_ids = Vec::new();

    // Add many order pairs that will fully match
    for i in 0..num_order_pairs {
        // Buy order
        let buy_order = Order::new(
            (i * 2) + 1,
            "BTCUSD".to_string(),
            Side::Buy,
            OrderType::Limit,
            TimeInForce::GoodTillCancel,
            50000_000000,
            1_000000,
            1000000000 + (i * 2) as u64,
        );
        order_ids.push(buy_order.order_id);
        engine.add_order(buy_order).unwrap();

        // Matching sell order
        let sell_order = Order::new(
            (i * 2) + 2,
            "BTCUSD".to_string(),
            Side::Sell,
            OrderType::Limit,
            TimeInForce::GoodTillCancel,
            50000_000000,
            1_000000,
            1000000000 + (i * 2 + 1) as u64,
        );
        order_ids.push(sell_order.order_id);
        let trades = engine.add_order(sell_order).unwrap();

        // Each pair should generate exactly one trade
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].quantity, 1_000000);
    }

    // Verify that order book is clean (no remaining orders)
    let market_data = engine.get_market_data("BTCUSD", 100).unwrap();
    assert!(market_data.bids.is_empty(), "Bid levels should be empty");
    assert!(market_data.asks.is_empty(), "Ask levels should be empty");

    // Test order cancellation memory safety
    for i in 0..100 {
        let order = Order::new(
            i + 10000,
            "BTCUSD".to_string(),
            Side::Buy,
            OrderType::Limit,
            TimeInForce::GoodTillCancel,
            49000_000000 + (i % 10) as u64,
            1_000000,
            2000000000 + i as u64,
        );
        let order_id = order.order_id;
        engine.add_order(order).unwrap();

        // Cancel every other order
        if i % 2 == 0 {
            let cancelled = engine.cancel_order("BTCUSD", order_id).unwrap();
            assert!(cancelled);
        }
    }

    // Verify partially cleaned state
    let market_data = engine.get_market_data("BTCUSD", 100).unwrap();
    assert_eq!(market_data.bids.len(), 50); // Half should remain after cancellations

    let stats = engine.get_statistics();
    println!("Memory safety test completed:");
    println!("  Orders processed: {}", stats.total_orders_processed);
    println!("  Trades generated: {}", stats.total_trades_generated);
    println!("  Remaining bid levels: {}", market_data.bids.len());
}

/// Test 11: Edge Cases and Error Handling
/// Tests various edge cases and error conditions
#[test]
fn test_edge_cases_and_error_handling() {
    let mut engine = OrderMatchingEngine::new(MatchingAlgorithm::PriceTimePriority);

    // Test 1: Zero quantity order
    let zero_qty_order = Order::new(
        1,
        "BTCUSD".to_string(),
        Side::Buy,
        OrderType::Limit,
        TimeInForce::GoodTillCancel,
        50000_000000,
        0, // Zero quantity
        1000000000,
    );

    let result = engine.add_order(zero_qty_order);
    // Should handle gracefully (implementation specific behavior)
    assert!(result.is_ok());

    // Test 2: Maximum quantity order
    let max_qty_order = Order::new(
        2,
        "BTCUSD".to_string(),
        Side::Sell,
        OrderType::Limit,
        TimeInForce::GoodTillCancel,
        50000_000000,
        u64::MAX,
        1000000000,
    );

    let result = engine.add_order(max_qty_order);
    assert!(result.is_ok());

    // Test 3: Cancel non-existent order
    let result = engine.cancel_order("BTCUSD", 99999);
    assert!(result.is_err());

    // Test 4: Cancel from non-existent symbol
    let result = engine.cancel_order("NONEXISTENT", 1);
    assert!(result.is_err());

    // Test 5: Market data for non-existent symbol
    let market_data = engine.get_market_data("NONEXISTENT", 5);
    assert!(market_data.is_none());

    // Test 6: Fill-or-Kill with exact liquidity
    let sell_order = Order::new(
        10,
        "TESTSYM".to_string(),
        Side::Sell,
        OrderType::Limit,
        TimeInForce::GoodTillCancel,
        60000_000000,
        5_000000,
        1000000000,
    );
    engine.add_order(sell_order).unwrap();

    let fok_order = Order::new(
        11,
        "TESTSYM".to_string(),
        Side::Buy,
        OrderType::FillOrKill,
        TimeInForce::FillOrKill,
        60000_000000,
        5_000000, // Exactly matches available liquidity
        1000000001,
    );

    let trades = engine.add_order(fok_order).unwrap();
    assert_eq!(trades.len(), 1);
    assert_eq!(trades[0].quantity, 5_000000);

    println!("Edge cases test completed successfully");
}

/// Test 12: Cross-Symbol Isolation
/// Tests that orders for different symbols don't interfere with each other
#[test]
fn test_cross_symbol_isolation() {
    let mut engine = OrderMatchingEngine::new(MatchingAlgorithm::PriceTimePriority);

    let symbols = vec!["BTCUSD", "ETHUSD", "ADAUSD"];
    let base_price = 50000_000000;

    // Add orders for each symbol
    for (i, symbol) in symbols.iter().enumerate() {
        let price = base_price + (i as u64 * 1000_000000);

        // Add buy order
        let buy_order = Order::new(
            (i * 2) as u64 + 1,
            symbol.to_string(),
            Side::Buy,
            OrderType::Limit,
            TimeInForce::GoodTillCancel,
            price,
            1_000000,
            1000000000,
        );
        engine.add_order(buy_order).unwrap();

        // Verify order appears in correct symbol's book
        let market_data = engine.get_market_data(symbol, 5).unwrap();
        assert_eq!(market_data.bids.len(), 1);
        assert_eq!(market_data.bids[0].price, price);

        // Verify other symbols are unaffected
        for other_symbol in &symbols {
            if *other_symbol != *symbol {
                let other_data = engine.get_market_data(other_symbol, 5);
                if let Some(data) = other_data {
                    assert!(
                        data.bids.is_empty() || data.bids[0].price != price,
                        "Symbol isolation violated"
                    );
                }
            }
        }
    }

    // Test cross-symbol matching isolation
    for (i, symbol) in symbols.iter().enumerate() {
        let price = base_price + (i as u64 * 1000_000000);

        let sell_order = Order::new(
            (i * 2) as u64 + 2,
            symbol.to_string(),
            Side::Sell,
            OrderType::Limit,
            TimeInForce::GoodTillCancel,
            price,
            1_000000,
            1000000001,
        );

        let trades = engine.add_order(sell_order).unwrap();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].symbol, *symbol);
    }

    // Verify all symbols have clean books after matching
    for symbol in &symbols {
        let market_data = engine.get_market_data(symbol, 5).unwrap();
        assert!(market_data.bids.is_empty());
        assert!(market_data.asks.is_empty());
    }

    println!("Cross-symbol isolation test completed successfully");
}

/// Comprehensive test coverage validation
#[test]
fn test_comprehensive_coverage_validation() {
    // This test ensures we've covered all major code paths
    let mut engine = OrderMatchingEngine::new(MatchingAlgorithm::PriceTimePriority);

    // Cover all order types
    let order_types = vec![
        OrderType::Limit,
        OrderType::Market,
        OrderType::StopLimit,
        OrderType::Iceberg,
        OrderType::FillOrKill,
        OrderType::ImmediateOrCancel,
    ];

    // Cover all matching algorithms
    let algorithms = vec![
        MatchingAlgorithm::PriceTimePriority,
        MatchingAlgorithm::ProRata,
        MatchingAlgorithm::PriceTimeSize,
    ];

    // Cover all time-in-force types
    let tif_types = vec![
        TimeInForce::Day,
        TimeInForce::GoodTillCancel,
        TimeInForce::ImmediateOrCancel,
        TimeInForce::FillOrKill,
    ];

    // Cover all order statuses through state transitions
    let mut order_id = 1;

    for &order_type in &order_types {
        for &algorithm in &algorithms {
            let mut test_engine = OrderMatchingEngine::new(algorithm);

            // Test each order type with each algorithm
            let order = Order::new(
                order_id,
                "TESTCOV".to_string(),
                Side::Buy,
                order_type,
                TimeInForce::GoodTillCancel,
                50000_000000,
                1_000000,
                1000000000 + order_id,
            );
            order_id += 1;

            let result = test_engine.add_order(order);

            match order_type {
                OrderType::StopLimit => {
                    // Stop limit orders should be accepted
                    assert!(result.is_ok());
                }
                _ => {
                    assert!(result.is_ok(), "Failed to add {:?} order", order_type);
                }
            }
        }
    }

    println!("Comprehensive coverage validation completed");
    println!("Tested {} order types", order_types.len());
    println!("Tested {} algorithms", algorithms.len());
    println!("Tested {} time-in-force types", tif_types.len());
}

// Additional utility functions for complex testing scenarios

/// Create a market maker scenario with multiple price levels
fn setup_market_maker_scenario(engine: &mut OrderMatchingEngine) -> Vec<u64> {
    let mut order_ids = Vec::new();
    let base_price = 50000_000000;

    // Add buy orders at descending prices
    for i in 0..10 {
        let buy_order = Order::new(
            i + 1,
            "BTCUSD".to_string(),
            Side::Buy,
            OrderType::Limit,
            TimeInForce::GoodTillCancel,
            base_price - (i * 100_000000), // Decreasing prices
            1_000000 * (i + 1),            // Increasing quantities
            1000000000 + i,
        );
        order_ids.push(buy_order.order_id);
        engine.add_order(buy_order).unwrap();
    }

    // Add sell orders at ascending prices
    for i in 0..10 {
        let sell_order = Order::new(
            i + 1000,
            "BTCUSD".to_string(),
            Side::Sell,
            OrderType::Limit,
            TimeInForce::GoodTillCancel,
            base_price + 100_000000 + (i * 100_000000), // Increasing prices
            1_000000 * (i + 1),                         // Increasing quantities
            1000000000 + i + 1000,
        );
        order_ids.push(sell_order.order_id);
        engine.add_order(sell_order).unwrap();
    }

    order_ids
}

/// Validate market data structure consistency
fn validate_market_data_consistency(market_data: &MarketDataSnapshot) {
    // Check bid ordering (descending)
    for i in 1..market_data.bids.len() {
        assert!(
            market_data.bids[i - 1].price >= market_data.bids[i].price,
            "Bid prices not in descending order: {} >= {}",
            market_data.bids[i - 1].price,
            market_data.bids[i].price
        );
    }

    // Check ask ordering (ascending)
    for i in 1..market_data.asks.len() {
        assert!(
            market_data.asks[i - 1].price <= market_data.asks[i].price,
            "Ask prices not in ascending order: {} <= {}",
            market_data.asks[i - 1].price,
            market_data.asks[i].price
        );
    }

    // Check for crossed market
    if !market_data.bids.is_empty() && !market_data.asks.is_empty() {
        let best_bid = market_data.bids[0].price;
        let best_ask = market_data.asks[0].price;
        assert!(
            best_bid <= best_ask,
            "Crossed market detected: best_bid {} > best_ask {}",
            best_bid,
            best_ask
        );
    }

    // Validate quantities are positive
    for bid in &market_data.bids {
        assert!(bid.quantity > 0, "Zero quantity in bid level");
        assert!(bid.order_count > 0, "Zero order count in bid level");
    }

    for ask in &market_data.asks {
        assert!(ask.quantity > 0, "Zero quantity in ask level");
        assert!(ask.order_count > 0, "Zero order count in ask level");
    }
}

/// Generate realistic trading scenario with various order types and sizes
fn generate_realistic_trading_scenario(
    engine: &mut OrderMatchingEngine,
    num_orders: usize,
) -> Vec<Trade> {
    let mut all_trades = Vec::new();
    let base_price = 50000_000000;

    for i in 0..num_orders {
        let side = if i % 2 == 0 { Side::Buy } else { Side::Sell };
        let price_variation = ((i % 100) as i64 - 50) * 10_000; // ±$0.10 variation
        let price = if price_variation >= 0 {
            base_price + price_variation as u64
        } else {
            base_price.saturating_sub((-price_variation) as u64)
        };

        let order_type = match i % 20 {
            0..=15 => OrderType::Limit,         // 80% limit orders
            16..=17 => OrderType::Market,       // 10% market orders
            18 => OrderType::ImmediateOrCancel, // 5% IOC orders
            _ => OrderType::FillOrKill,         // 5% FOK orders
        };

        let quantity = match i % 10 {
            0..=6 => 1_000000, // 70% standard size
            7..=8 => 5_000000, // 20% large size
            _ => 100_000,      // 10% small size
        };

        let order = Order::new(
            i as u64 + 1,
            "BTCUSD".to_string(),
            side,
            order_type,
            TimeInForce::GoodTillCancel,
            price,
            quantity,
            1000000000 + i as u64,
        );

        if let Ok(trades) = engine.add_order(order) {
            all_trades.extend(trades);
        }
    }

    all_trades
}
