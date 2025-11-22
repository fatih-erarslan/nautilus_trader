//! Order Matching Engine - Ultra-high performance atomic order matching
//!
//! This module implements a lock-free order matching engine with sub-microsecond
//! matching latency, supporting multiple matching algorithms and order types.

use std::collections::{BTreeMap, HashMap};
use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
// Removed unused CachePadded import
use crossbeam::queue::SegQueue;

/// Maximum supported price levels
#[allow(dead_code)]
const MAX_PRICE_LEVELS: usize = 100000;
/// Maximum orders per level
#[allow(dead_code)]
const ORDERS_PER_LEVEL: usize = 10000;
/// Nanosecond precision for timestamps
#[allow(dead_code)]
const NANOS_PER_SECOND: u64 = 1_000_000_000;

/// Order matching algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatchingAlgorithm {
    PriceTimePriority, // Standard FIFO within price levels
    ProRata,           // Pro-rata allocation at same price
    PriceTimeSize,     // Price-Time-Size priority
    MarketMaker,       // Market maker priority
    Iceberg,           // Iceberg order handling
}

/// Order types supported by the engine
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderType {
    Limit,             // Standard limit order
    Market,            // Market order
    StopLimit,         // Stop-limit order
    Iceberg,           // Iceberg order with hidden quantity
    PeggedLimit,       // Pegged to best bid/offer
    AllOrNone,         // All-or-none execution
    FillOrKill,        // Fill-or-kill order
    ImmediateOrCancel, // Immediate-or-cancel
}

/// Order side
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Side {
    Buy,
    Sell,
}

/// Order time-in-force
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeInForce {
    Day,               // Valid until end of day
    GoodTillCancel,    // Valid until cancelled
    ImmediateOrCancel, // Execute immediately or cancel
    FillOrKill,        // Fill completely or cancel
}

/// Order status
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderStatus {
    New,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

/// Trade execution result
#[derive(Debug, Clone)]
pub struct Trade {
    pub trade_id: u64,
    pub buy_order_id: u64,
    pub sell_order_id: u64,
    pub symbol: String,
    pub price: u64,     // Price in micro-units
    pub quantity: u64,  // Quantity in micro-units
    pub timestamp: u64, // Nanosecond timestamp
    pub aggressor_side: Side,
}

/// Order structure optimized for atomic operations
#[repr(C, align(64))] // Cache line aligned
pub struct Order {
    pub order_id: u64,
    pub symbol: String,
    pub side: Side,
    pub order_type: OrderType,
    pub time_in_force: TimeInForce,
    pub price: u64,                    // Price in micro-units (1/1M of base unit)
    pub original_quantity: u64,        // Original quantity in micro-units
    pub remaining_quantity: AtomicU64, // Remaining quantity (atomic for concurrent access)
    pub hidden_quantity: u64,          // For iceberg orders
    pub displayed_quantity: AtomicU64, // Displayed quantity for iceberg
    pub timestamp: u64,                // Order creation timestamp (nanoseconds)
    pub last_update: AtomicU64,        // Last update timestamp
    pub status: AtomicU64,             // Order status (as u64 for atomic ops)
    pub filled_quantity: AtomicU64,    // Total filled quantity
    pub average_fill_price: AtomicU64, // Volume-weighted average fill price
    pub next: AtomicPtr<Order>,        // Next order in linked list
    pub prev: AtomicPtr<Order>,        // Previous order for doubly-linked list
}

impl Order {
    /// Create new order
    pub fn new(
        order_id: u64,
        symbol: String,
        side: Side,
        order_type: OrderType,
        time_in_force: TimeInForce,
        price: u64,
        quantity: u64,
        timestamp: u64,
    ) -> Self {
        let displayed_qty = match order_type {
            OrderType::Iceberg => quantity.min(quantity / 10), // Display 10% for iceberg
            _ => quantity,
        };

        Self {
            order_id,
            symbol,
            side,
            order_type,
            time_in_force,
            price,
            original_quantity: quantity,
            remaining_quantity: AtomicU64::new(quantity),
            hidden_quantity: if order_type == OrderType::Iceberg {
                quantity - displayed_qty
            } else {
                0
            },
            displayed_quantity: AtomicU64::new(displayed_qty),
            timestamp,
            last_update: AtomicU64::new(timestamp),
            status: AtomicU64::new(OrderStatus::New as u64),
            filled_quantity: AtomicU64::new(0),
            average_fill_price: AtomicU64::new(0),
            next: AtomicPtr::new(ptr::null_mut()),
            prev: AtomicPtr::new(ptr::null_mut()),
        }
    }

    /// Get current status - safe implementation without unsafe transmute
    pub fn get_status(&self) -> OrderStatus {
        let status_val = self.status.load(Ordering::Acquire);
        match status_val {
            0 => OrderStatus::New,
            1 => OrderStatus::PartiallyFilled,
            2 => OrderStatus::Filled,
            3 => OrderStatus::Cancelled,
            4 => OrderStatus::Rejected,
            5 => OrderStatus::Expired,
            _ => OrderStatus::Rejected, // Safe fallback for invalid states
        }
    }

    /// Update order status atomically
    pub fn set_status(&self, status: OrderStatus) -> bool {
        let old_status = self.status.swap(status as u64, Ordering::AcqRel);
        old_status != (status as u64)
    }

    /// Try to fill quantity from this order
    pub fn try_fill(&self, requested_qty: u64, fill_price: u64) -> Result<u64, &'static str> {
        loop {
            let current_remaining = self.remaining_quantity.load(Ordering::Acquire);

            if current_remaining == 0 {
                return Err("Order already filled");
            }

            let fill_qty = requested_qty.min(current_remaining);
            let new_remaining = current_remaining - fill_qty;

            match self.remaining_quantity.compare_exchange_weak(
                current_remaining,
                new_remaining,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // Update filled quantity
                    self.filled_quantity.fetch_add(fill_qty, Ordering::AcqRel);

                    // Update average fill price
                    self.update_average_fill_price(fill_qty, fill_price);

                    // Update status
                    if new_remaining == 0 {
                        self.set_status(OrderStatus::Filled);
                    } else if fill_qty > 0 {
                        self.set_status(OrderStatus::PartiallyFilled);

                        // Handle iceberg order refill
                        if self.order_type == OrderType::Iceberg {
                            self.refill_iceberg_display();
                        }
                    }

                    // Update timestamp
                    self.last_update
                        .store(self.get_timestamp_ns(), Ordering::Release);

                    return Ok(fill_qty);
                }
                Err(_) => continue, // Retry on CAS failure
            }
        }
    }

    /// Update volume-weighted average fill price
    fn update_average_fill_price(&self, fill_qty: u64, fill_price: u64) {
        loop {
            let current_filled = self.filled_quantity.load(Ordering::Acquire) - fill_qty; // Before this fill
            let current_avg_price = self.average_fill_price.load(Ordering::Acquire);

            let new_avg_price = if current_filled == 0 {
                fill_price
            } else {
                ((current_avg_price * current_filled) + (fill_price * fill_qty))
                    / (current_filled + fill_qty)
            };

            match self.average_fill_price.compare_exchange_weak(
                current_avg_price,
                new_avg_price,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(_) => continue,
            }
        }
    }

    /// Refill iceberg order display quantity
    fn refill_iceberg_display(&self) {
        if self.order_type != OrderType::Iceberg {
            return;
        }

        let remaining = self.remaining_quantity.load(Ordering::Acquire);
        let current_displayed = self.displayed_quantity.load(Ordering::Acquire);

        if current_displayed == 0 && remaining > 0 {
            let new_display = remaining.min(self.original_quantity / 10); // 10% display size
            self.displayed_quantity
                .store(new_display, Ordering::Release);
        }
    }

    fn get_timestamp_ns(&self) -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
}

/// Lock-free price level with atomic operations
#[repr(C, align(64))]
pub struct PriceLevel {
    pub price: u64,
    pub total_quantity: AtomicU64,
    pub displayed_quantity: AtomicU64, // For iceberg orders
    pub order_count: AtomicU64,
    pub head: AtomicPtr<Order>,
    pub tail: AtomicPtr<Order>,
    pub last_update: AtomicU64,
}

impl PriceLevel {
    pub fn new(price: u64) -> Self {
        Self {
            price,
            total_quantity: AtomicU64::new(0),
            displayed_quantity: AtomicU64::new(0),
            order_count: AtomicU64::new(0),
            head: AtomicPtr::new(ptr::null_mut()),
            tail: AtomicPtr::new(ptr::null_mut()),
            last_update: AtomicU64::new(0),
        }
    }

    /// Insert order at end of queue (price-time priority)
    pub fn insert_order(&self, order: *mut Order) -> bool {
        unsafe {
            let order_ref = &*order;
            let order_qty = if order_ref.order_type == OrderType::Iceberg {
                order_ref.displayed_quantity.load(Ordering::Acquire)
            } else {
                order_ref.remaining_quantity.load(Ordering::Acquire)
            };

            // Update price level quantities
            self.total_quantity.fetch_add(
                order_ref.remaining_quantity.load(Ordering::Acquire),
                Ordering::AcqRel,
            );
            self.displayed_quantity
                .fetch_add(order_qty, Ordering::AcqRel);

            loop {
                let tail = self.tail.load(Ordering::Acquire);

                if tail.is_null() {
                    // Empty level
                    match self.head.compare_exchange_weak(
                        ptr::null_mut(),
                        order,
                        Ordering::Release,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => {
                            self.tail.store(order, Ordering::Release);
                            self.order_count.fetch_add(1, Ordering::AcqRel);
                            self.last_update
                                .store(order_ref.timestamp, Ordering::Release);
                            return true;
                        }
                        Err(_) => continue,
                    }
                } else {
                    // Non-empty level - append to tail
                    let tail_ref = &*tail;

                    match tail_ref.next.compare_exchange_weak(
                        ptr::null_mut(),
                        order,
                        Ordering::Release,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => {
                            order_ref.prev.store(tail, Ordering::Release);

                            // Try to update tail
                            let _ = self.tail.compare_exchange_weak(
                                tail,
                                order,
                                Ordering::Release,
                                Ordering::Relaxed,
                            );

                            self.order_count.fetch_add(1, Ordering::AcqRel);
                            self.last_update
                                .store(order_ref.timestamp, Ordering::Release);
                            return true;
                        }
                        Err(_) => {
                            // Help advance tail if needed
                            let next = tail_ref.next.load(Ordering::Acquire);
                            if !next.is_null() {
                                let _ = self.tail.compare_exchange_weak(
                                    tail,
                                    next,
                                    Ordering::Release,
                                    Ordering::Relaxed,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Match orders at this price level using specified algorithm
    pub fn match_orders(
        &self,
        aggressor_qty: u64,
        aggressor_price: u64,
        algorithm: MatchingAlgorithm,
    ) -> Vec<(u64, u64, *mut Order)> {
        // (filled_qty, price, order_ptr)

        match algorithm {
            MatchingAlgorithm::PriceTimePriority => self.match_fifo(aggressor_qty, aggressor_price),
            MatchingAlgorithm::ProRata => self.match_pro_rata(aggressor_qty, aggressor_price),
            MatchingAlgorithm::PriceTimeSize => {
                self.match_price_time_size(aggressor_qty, aggressor_price)
            }
            _ => self.match_fifo(aggressor_qty, aggressor_price), // Default to FIFO
        }
    }

    /// FIFO matching within price level
    fn match_fifo(
        &self,
        mut remaining_qty: u64,
        aggressor_price: u64,
    ) -> Vec<(u64, u64, *mut Order)> {
        let mut matches = Vec::new();
        let mut current = self.head.load(Ordering::Acquire);

        while !current.is_null() && remaining_qty > 0 {
            unsafe {
                let order = &*current;

                // Skip cancelled orders
                if order.get_status() == OrderStatus::Cancelled {
                    current = order.next.load(Ordering::Acquire);
                    continue;
                }

                let available_qty = if order.order_type == OrderType::Iceberg {
                    order.displayed_quantity.load(Ordering::Acquire)
                } else {
                    order.remaining_quantity.load(Ordering::Acquire)
                };

                if available_qty > 0 {
                    let match_qty = remaining_qty.min(available_qty);

                    match order.try_fill(match_qty, aggressor_price) {
                        Ok(filled_qty) => {
                            matches.push((filled_qty, aggressor_price, current));
                            remaining_qty -= filled_qty;

                            // Update level quantities
                            self.total_quantity.fetch_sub(filled_qty, Ordering::AcqRel);
                            self.displayed_quantity
                                .fetch_sub(filled_qty, Ordering::AcqRel);

                            // Remove order if fully filled
                            if order.remaining_quantity.load(Ordering::Acquire) == 0 {
                                self.remove_order(current);
                            }
                        }
                        Err(_) => {
                            // Order couldn't be filled, move to next
                        }
                    }
                }

                current = order.next.load(Ordering::Acquire);
            }
        }

        matches
    }

    /// Pro-rata matching algorithm
    fn match_pro_rata(
        &self,
        aggressor_qty: u64,
        aggressor_price: u64,
    ) -> Vec<(u64, u64, *mut Order)> {
        let mut matches = Vec::new();

        // First pass: collect all eligible orders and total quantity
        let mut eligible_orders = Vec::new();
        let mut total_eligible_qty = 0u64;
        let mut current = self.head.load(Ordering::Acquire);

        while !current.is_null() {
            unsafe {
                let order = &*current;

                if order.get_status() != OrderStatus::Cancelled {
                    let available_qty = if order.order_type == OrderType::Iceberg {
                        order.displayed_quantity.load(Ordering::Acquire)
                    } else {
                        order.remaining_quantity.load(Ordering::Acquire)
                    };

                    if available_qty > 0 {
                        eligible_orders.push((current, available_qty));
                        total_eligible_qty += available_qty;
                    }
                }

                current = order.next.load(Ordering::Acquire);
            }
        }

        // Second pass: allocate pro-rata
        let mut remaining_qty = aggressor_qty.min(total_eligible_qty);

        for (order_ptr, available_qty) in eligible_orders {
            if remaining_qty == 0 {
                break;
            }

            unsafe {
                let order = &*order_ptr;

                // Calculate pro-rata allocation
                let allocation = if total_eligible_qty > 0 {
                    (aggressor_qty * available_qty) / total_eligible_qty
                } else {
                    0
                };

                let match_qty = allocation.min(available_qty).min(remaining_qty);

                if match_qty > 0 {
                    if let Ok(filled_qty) = order.try_fill(match_qty, aggressor_price) {
                        matches.push((filled_qty, aggressor_price, order_ptr));
                        remaining_qty -= filled_qty;

                        // Update level quantities
                        self.total_quantity.fetch_sub(filled_qty, Ordering::AcqRel);
                        self.displayed_quantity
                            .fetch_sub(filled_qty, Ordering::AcqRel);

                        if order.remaining_quantity.load(Ordering::Acquire) == 0 {
                            self.remove_order(order_ptr);
                        }
                    }
                }
            }
        }

        matches
    }

    /// Price-Time-Size priority matching
    fn match_price_time_size(
        &self,
        aggressor_qty: u64,
        aggressor_price: u64,
    ) -> Vec<(u64, u64, *mut Order)> {
        // For simplicity, this implements size priority within time priority
        // In practice, this would be more sophisticated

        let mut large_orders = Vec::new();
        let mut small_orders = Vec::new();
        let size_threshold = 10000; // Orders >= 10k units are "large"

        let mut current = self.head.load(Ordering::Acquire);

        while !current.is_null() {
            unsafe {
                let order = &*current;

                if order.get_status() != OrderStatus::Cancelled {
                    let available_qty = order.remaining_quantity.load(Ordering::Acquire);

                    if available_qty > 0 {
                        if available_qty >= size_threshold {
                            large_orders.push(current);
                        } else {
                            small_orders.push(current);
                        }
                    }
                }

                current = order.next.load(Ordering::Acquire);
            }
        }

        // Match large orders first, then small orders
        let mut matches = Vec::new();
        let mut remaining_qty = aggressor_qty;

        for order_list in [&large_orders, &small_orders] {
            for &order_ptr in order_list {
                if remaining_qty == 0 {
                    break;
                }

                unsafe {
                    let order = &*order_ptr;
                    let available_qty = order.remaining_quantity.load(Ordering::Acquire);
                    let match_qty = remaining_qty.min(available_qty);

                    if let Ok(filled_qty) = order.try_fill(match_qty, aggressor_price) {
                        matches.push((filled_qty, aggressor_price, order_ptr));
                        remaining_qty -= filled_qty;

                        self.total_quantity.fetch_sub(filled_qty, Ordering::AcqRel);
                        self.displayed_quantity
                            .fetch_sub(filled_qty, Ordering::AcqRel);

                        if order.remaining_quantity.load(Ordering::Acquire) == 0 {
                            self.remove_order(order_ptr);
                        }
                    }
                }
            }

            if remaining_qty == 0 {
                break;
            }
        }

        matches
    }

    /// Remove order from price level
    fn remove_order(&self, order_ptr: *mut Order) {
        unsafe {
            let order = &*order_ptr;

            let prev = order.prev.load(Ordering::Acquire);
            let next = order.next.load(Ordering::Acquire);

            // Update links
            if prev.is_null() {
                // Removing head
                self.head.store(next, Ordering::Release);
            } else {
                (*prev).next.store(next, Ordering::Release);
            }

            if next.is_null() {
                // Removing tail
                self.tail.store(prev, Ordering::Release);
            } else {
                (*next).prev.store(prev, Ordering::Release);
            }

            // Update counters
            self.order_count.fetch_sub(1, Ordering::AcqRel);

            // Update quantities
            let remaining_qty = order.remaining_quantity.load(Ordering::Acquire);
            let displayed_qty = if order.order_type == OrderType::Iceberg {
                order.displayed_quantity.load(Ordering::Acquire)
            } else {
                remaining_qty
            };

            self.total_quantity
                .fetch_sub(remaining_qty, Ordering::AcqRel);
            self.displayed_quantity
                .fetch_sub(displayed_qty, Ordering::AcqRel);
        }
    }

    /// Get total quantity at this level
    pub fn get_total_quantity(&self) -> u64 {
        self.total_quantity.load(Ordering::Acquire)
    }

    /// Get displayed quantity (for market data)
    pub fn get_displayed_quantity(&self) -> u64 {
        self.displayed_quantity.load(Ordering::Acquire)
    }

    /// Get order count
    pub fn get_order_count(&self) -> u64 {
        self.order_count.load(Ordering::Acquire)
    }
}

/// High-performance order matching engine
pub struct OrderMatchingEngine {
    // Order books by symbol
    order_books: HashMap<String, OrderBook>,

    // Trade output queue
    trade_queue: SegQueue<Trade>,

    // Configuration
    default_matching_algorithm: MatchingAlgorithm,

    // Statistics
    total_orders_processed: AtomicU64,
    total_trades_generated: AtomicU64,
    total_volume_matched: AtomicU64,

    // Performance tracking
    avg_matching_time_ns: AtomicU64,
    max_matching_time_ns: AtomicU64,

    // Order ID generator
    #[allow(dead_code)]
    next_trade_id: AtomicU64,
}

/// Order book for a single symbol
pub struct OrderBook {
    pub symbol: String,

    // Price levels (using BTreeMap for ordered prices)
    pub bid_levels: BTreeMap<u64, Box<PriceLevel>>, // Descending order
    pub ask_levels: BTreeMap<u64, Box<PriceLevel>>, // Ascending order

    // Best prices
    pub best_bid: AtomicU64,
    pub best_ask: AtomicU64,

    // Order lookup
    pub orders: HashMap<u64, *mut Order>,

    // Matching algorithm for this symbol
    pub matching_algorithm: MatchingAlgorithm,

    // Statistics
    pub last_trade_price: AtomicU64,
    pub total_volume: AtomicU64,
    pub trade_count: AtomicU64,
}

impl OrderBook {
    pub fn new(symbol: String, matching_algorithm: MatchingAlgorithm) -> Self {
        Self {
            symbol,
            bid_levels: BTreeMap::new(),
            ask_levels: BTreeMap::new(),
            best_bid: AtomicU64::new(0),
            best_ask: AtomicU64::new(u64::MAX),
            orders: HashMap::new(),
            matching_algorithm,
            last_trade_price: AtomicU64::new(0),
            total_volume: AtomicU64::new(0),
            trade_count: AtomicU64::new(0),
        }
    }

    /// Add order to book
    pub fn add_order(&mut self, order: Order) -> Result<Vec<Trade>, &'static str> {
        let order_ptr = Box::into_raw(Box::new(order));
        let order_ref = unsafe { &*order_ptr };

        // Store order for lookup
        self.orders.insert(order_ref.order_id, order_ptr);

        let mut trades = Vec::new();

        match order_ref.order_type {
            OrderType::Market => {
                // Market order - match immediately
                trades = self.match_market_order(order_ptr)?;
            }
            OrderType::Limit => {
                // Limit order - try to match, then add to book if not fully filled
                trades = self.match_limit_order(order_ptr)?;
            }
            OrderType::StopLimit => {
                // Stop-limit order - add to book (stop logic would be handled elsewhere)
                self.add_limit_order_to_book(order_ptr);
            }
            OrderType::Iceberg => {
                // Iceberg order - treat like limit order but with hidden quantity
                trades = self.match_limit_order(order_ptr)?;
            }
            OrderType::FillOrKill => {
                trades = self.match_fill_or_kill(order_ptr)?;
            }
            OrderType::ImmediateOrCancel => {
                trades = self.match_immediate_or_cancel(order_ptr)?;
            }
            _ => {
                return Err("Unsupported order type");
            }
        }

        Ok(trades)
    }

    /// Match market order
    fn match_market_order(&mut self, order_ptr: *mut Order) -> Result<Vec<Trade>, &'static str> {
        let order = unsafe { &*order_ptr };
        let mut trades = Vec::new();
        let mut remaining_qty = order.remaining_quantity.load(Ordering::Acquire);
        let order_side = order.side;
        let order_id = order.order_id;
        let original_qty = order.original_quantity;
        let symbol = self.symbol.clone();
        let matching_algorithm = self.matching_algorithm;

        let levels = match order_side {
            Side::Buy => &mut self.ask_levels,  // Buy matches against asks
            Side::Sell => &mut self.bid_levels, // Sell matches against bids
        };

        // Sort levels by best price first
        let mut level_prices: Vec<u64> = levels.keys().cloned().collect();
        match order_side {
            Side::Buy => level_prices.sort(), // Buy wants lowest ask prices first
            Side::Sell => level_prices.sort_by(|a, b| b.cmp(a)), // Sell wants highest bid prices first
        }

        let mut prices_to_remove = Vec::new();

        for price in level_prices {
            if remaining_qty == 0 {
                break;
            }

            if let Some(level) = levels.get(&price) {
                let matches = level.match_orders(remaining_qty, price, matching_algorithm);

                for (filled_qty, fill_price, matched_order_ptr) in matches {
                    let matched_order = unsafe { &*matched_order_ptr };

                    // Generate trade ID first
                    static TRADE_ID_COUNTER: std::sync::atomic::AtomicU64 =
                        std::sync::atomic::AtomicU64::new(1);
                    let trade_id = TRADE_ID_COUNTER.fetch_add(1, Ordering::AcqRel);

                    // Get timestamp
                    let timestamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos() as u64;

                    let trade = Trade {
                        trade_id,
                        buy_order_id: if order_side == Side::Buy {
                            order_id
                        } else {
                            matched_order.order_id
                        },
                        sell_order_id: if order_side == Side::Sell {
                            order_id
                        } else {
                            matched_order.order_id
                        },
                        symbol: symbol.clone(),
                        price: fill_price,
                        quantity: filled_qty,
                        timestamp,
                        aggressor_side: order_side,
                    };

                    trades.push(trade);
                    remaining_qty -= filled_qty;

                    // Update statistics
                    self.last_trade_price.store(fill_price, Ordering::Release);
                    self.total_volume.fetch_add(filled_qty, Ordering::AcqRel);
                    self.trade_count.fetch_add(1, Ordering::AcqRel);
                }

                // Mark empty levels for removal
                if level.get_total_quantity() == 0 {
                    prices_to_remove.push(price);
                }
            }
        }

        // Remove empty levels
        for price in prices_to_remove {
            levels.remove(&price);
        }

        // Update order status
        if remaining_qty == 0 {
            unsafe { &*order_ptr }.set_status(OrderStatus::Filled);
        } else if remaining_qty < original_qty {
            unsafe { &*order_ptr }.set_status(OrderStatus::PartiallyFilled);
        }

        // Market orders don't stay in the book
        if remaining_qty > 0 {
            unsafe { &*order_ptr }.set_status(OrderStatus::Cancelled); // Partially filled market order remainder is cancelled
        }

        self.update_best_prices();
        Ok(trades)
    }

    /// Match limit order
    fn match_limit_order(&mut self, order_ptr: *mut Order) -> Result<Vec<Trade>, &'static str> {
        let order = unsafe { &*order_ptr };
        let mut trades = Vec::new();
        let mut remaining_qty = order.remaining_quantity.load(Ordering::Acquire);
        let order_side = order.side;
        let order_id = order.order_id;
        let order_price = order.price;
        let symbol = self.symbol.clone();
        let matching_algorithm = self.matching_algorithm;

        // Check if limit order can match
        let can_match = match order_side {
            Side::Buy => {
                let best_ask = self.best_ask.load(Ordering::Acquire);
                best_ask != u64::MAX && order_price >= best_ask
            }
            Side::Sell => {
                let best_bid = self.best_bid.load(Ordering::Acquire);
                best_bid != 0 && order_price <= best_bid
            }
        };

        if can_match {
            // Match against existing orders
            let levels = match order_side {
                Side::Buy => &mut self.ask_levels,
                Side::Sell => &mut self.bid_levels,
            };

            let mut level_prices: Vec<u64> = levels.keys().cloned().collect();
            match order_side {
                Side::Buy => level_prices.sort(),
                Side::Sell => level_prices.sort_by(|a, b| b.cmp(a)),
            }

            let mut prices_to_remove = Vec::new();

            for price in level_prices {
                if remaining_qty == 0 {
                    break;
                }

                // Check if we can still match at this price
                let price_match = match order_side {
                    Side::Buy => order_price >= price,
                    Side::Sell => order_price <= price,
                };

                if !price_match {
                    break;
                }

                if let Some(level) = levels.get(&price) {
                    let matches = level.match_orders(remaining_qty, price, matching_algorithm);

                    for (filled_qty, fill_price, matched_order_ptr) in matches {
                        let matched_order = unsafe { &*matched_order_ptr };

                        // Generate trade ID first
                        static TRADE_ID_COUNTER: std::sync::atomic::AtomicU64 =
                            std::sync::atomic::AtomicU64::new(1);
                        let trade_id = TRADE_ID_COUNTER.fetch_add(1, Ordering::AcqRel);

                        // Get timestamp
                        let timestamp = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_nanos() as u64;

                        let trade = Trade {
                            trade_id,
                            buy_order_id: if order_side == Side::Buy {
                                order_id
                            } else {
                                matched_order.order_id
                            },
                            sell_order_id: if order_side == Side::Sell {
                                order_id
                            } else {
                                matched_order.order_id
                            },
                            symbol: symbol.clone(),
                            price: fill_price,
                            quantity: filled_qty,
                            timestamp,
                            aggressor_side: order_side,
                        };

                        trades.push(trade);
                        remaining_qty -= filled_qty;

                        // Update statistics
                        self.last_trade_price.store(fill_price, Ordering::Release);
                        self.total_volume.fetch_add(filled_qty, Ordering::AcqRel);
                        self.trade_count.fetch_add(1, Ordering::AcqRel);
                    }

                    // Mark empty levels for removal
                    if level.get_total_quantity() == 0 {
                        prices_to_remove.push(price);
                    }
                }
            }

            // Remove empty levels
            for price in prices_to_remove {
                levels.remove(&price);
            }
        }

        // Add remaining quantity to book if any
        if remaining_qty > 0 {
            self.add_limit_order_to_book(order_ptr);
        }

        self.update_best_prices();
        Ok(trades)
    }

    /// Add limit order to appropriate price level
    fn add_limit_order_to_book(&mut self, order_ptr: *mut Order) {
        let order = unsafe { &*order_ptr };

        let levels = match order.side {
            Side::Buy => &mut self.bid_levels,
            Side::Sell => &mut self.ask_levels,
        };

        // Get or create price level
        let level = levels
            .entry(order.price)
            .or_insert_with(|| Box::new(PriceLevel::new(order.price)));

        level.insert_order(order_ptr);

        // Update order status
        if order.filled_quantity.load(Ordering::Acquire) > 0 {
            order.set_status(OrderStatus::PartiallyFilled);
        }
    }

    /// Match fill-or-kill order
    fn match_fill_or_kill(&mut self, order_ptr: *mut Order) -> Result<Vec<Trade>, &'static str> {
        let order = unsafe { &*order_ptr };

        // Check if full quantity can be matched
        let available_qty = self.get_available_quantity(order.side, order.price);

        if available_qty >= order.original_quantity {
            // Can fill completely - proceed with matching
            self.match_limit_order(order_ptr)
        } else {
            // Cannot fill completely - cancel order
            order.set_status(OrderStatus::Cancelled);
            Ok(Vec::new())
        }
    }

    /// Match immediate-or-cancel order
    fn match_immediate_or_cancel(
        &mut self,
        order_ptr: *mut Order,
    ) -> Result<Vec<Trade>, &'static str> {
        // IOC orders match what they can immediately, then cancel remainder
        let trades = self.match_limit_order(order_ptr)?;

        let order = unsafe { &*order_ptr };
        if order.remaining_quantity.load(Ordering::Acquire) > 0 {
            order.set_status(OrderStatus::Cancelled);
        }

        Ok(trades)
    }

    /// Get available quantity for matching at or better than specified price
    fn get_available_quantity(&self, side: Side, limit_price: u64) -> u64 {
        let levels = match side {
            Side::Buy => &self.ask_levels,
            Side::Sell => &self.bid_levels,
        };

        let mut total_qty = 0;

        for (&price, level) in levels {
            let price_ok = match side {
                Side::Buy => price <= limit_price,
                Side::Sell => price >= limit_price,
            };

            if price_ok {
                total_qty += level.get_displayed_quantity();
            } else {
                break; // Prices are sorted, so we can stop here
            }
        }

        total_qty
    }

    /// Update best bid and ask prices
    fn update_best_prices(&mut self) {
        // Update best bid
        if let Some((&best_bid_price, _)) = self.bid_levels.iter().next_back() {
            self.best_bid.store(best_bid_price, Ordering::Release);
        } else {
            self.best_bid.store(0, Ordering::Release);
        }

        // Update best ask
        if let Some((&best_ask_price, _)) = self.ask_levels.iter().next() {
            self.best_ask.store(best_ask_price, Ordering::Release);
        } else {
            self.best_ask.store(u64::MAX, Ordering::Release);
        }
    }

    /// Cancel order
    pub fn cancel_order(&mut self, order_id: u64) -> Result<bool, &'static str> {
        if let Some(&order_ptr) = self.orders.get(&order_id) {
            let order = unsafe { &*order_ptr };

            if order.set_status(OrderStatus::Cancelled) {
                // Remove from appropriate price level
                let levels = match order.side {
                    Side::Buy => &mut self.bid_levels,
                    Side::Sell => &mut self.ask_levels,
                };

                if let Some(level) = levels.get(&order.price) {
                    level.remove_order(order_ptr);

                    // Remove empty level
                    if level.get_total_quantity() == 0 {
                        levels.remove(&order.price);
                    }
                }

                self.orders.remove(&order_id);
                self.update_best_prices();

                Ok(true)
            } else {
                Err("Order cannot be cancelled")
            }
        } else {
            Err("Order not found")
        }
    }

    /// Generate unique trade ID
    #[allow(dead_code)]
    fn generate_trade_id(&self) -> u64 {
        static TRADE_ID_COUNTER: AtomicU64 = AtomicU64::new(1);
        TRADE_ID_COUNTER.fetch_add(1, Ordering::AcqRel)
    }

    fn get_timestamp_ns(&self) -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }

    /// Get market data snapshot
    pub fn get_market_data(&self, depth: usize) -> MarketDataSnapshot {
        let mut bids = Vec::new();
        let mut asks = Vec::new();

        // Collect bid levels (highest to lowest)
        for (&price, level) in self.bid_levels.iter().rev().take(depth) {
            if level.get_displayed_quantity() > 0 {
                bids.push(PriceLevelData {
                    price,
                    quantity: level.get_displayed_quantity(),
                    order_count: level.get_order_count(),
                });
            }
        }

        // Collect ask levels (lowest to highest)
        for (&price, level) in self.ask_levels.iter().take(depth) {
            if level.get_displayed_quantity() > 0 {
                asks.push(PriceLevelData {
                    price,
                    quantity: level.get_displayed_quantity(),
                    order_count: level.get_order_count(),
                });
            }
        }

        MarketDataSnapshot {
            symbol: self.symbol.clone(),
            timestamp: self.get_timestamp_ns(),
            bids,
            asks,
            last_trade_price: self.last_trade_price.load(Ordering::Acquire),
            total_volume: self.total_volume.load(Ordering::Acquire),
            trade_count: self.trade_count.load(Ordering::Acquire),
        }
    }
}

/// Market data snapshot
#[derive(Debug, Clone)]
pub struct MarketDataSnapshot {
    pub symbol: String,
    pub timestamp: u64,
    pub bids: Vec<PriceLevelData>,
    pub asks: Vec<PriceLevelData>,
    pub last_trade_price: u64,
    pub total_volume: u64,
    pub trade_count: u64,
}

/// Price level data for market data
#[derive(Debug, Clone)]
pub struct PriceLevelData {
    pub price: u64,
    pub quantity: u64,
    pub order_count: u64,
}

impl OrderMatchingEngine {
    pub fn new(default_algorithm: MatchingAlgorithm) -> Self {
        Self {
            order_books: HashMap::new(),
            trade_queue: SegQueue::new(),
            default_matching_algorithm: default_algorithm,
            total_orders_processed: AtomicU64::new(0),
            total_trades_generated: AtomicU64::new(0),
            total_volume_matched: AtomicU64::new(0),
            avg_matching_time_ns: AtomicU64::new(0),
            max_matching_time_ns: AtomicU64::new(0),
            next_trade_id: AtomicU64::new(1),
        }
    }

    /// Add order to engine
    pub fn add_order(&mut self, order: Order) -> Result<Vec<Trade>, &'static str> {
        let start_time = self.get_timestamp_ns();

        // Get or create order book
        let symbol = order.symbol.clone();
        let order_book = self
            .order_books
            .entry(symbol.clone())
            .or_insert_with(|| OrderBook::new(symbol, self.default_matching_algorithm));

        let trades = order_book.add_order(order)?;

        // Enqueue trades
        for trade in &trades {
            self.trade_queue.push(trade.clone());
        }

        // Update statistics
        self.total_orders_processed.fetch_add(1, Ordering::AcqRel);
        self.total_trades_generated
            .fetch_add(trades.len() as u64, Ordering::AcqRel);

        let total_volume: u64 = trades.iter().map(|t| t.quantity).sum();
        self.total_volume_matched
            .fetch_add(total_volume, Ordering::AcqRel);

        // Update timing statistics
        let execution_time = self.get_timestamp_ns() - start_time;
        let current_max = self.max_matching_time_ns.load(Ordering::Acquire);
        if execution_time > current_max {
            self.max_matching_time_ns
                .store(execution_time, Ordering::Release);
        }

        // Update average (simplified)
        let current_avg = self.avg_matching_time_ns.load(Ordering::Acquire);
        let orders_processed = self.total_orders_processed.load(Ordering::Acquire);
        let new_avg = (current_avg * (orders_processed - 1) + execution_time) / orders_processed;
        self.avg_matching_time_ns.store(new_avg, Ordering::Release);

        Ok(trades)
    }

    /// Cancel order
    pub fn cancel_order(&mut self, symbol: &str, order_id: u64) -> Result<bool, &'static str> {
        if let Some(order_book) = self.order_books.get_mut(symbol) {
            order_book.cancel_order(order_id)
        } else {
            Err("Order book not found")
        }
    }

    /// Get market data for symbol
    pub fn get_market_data(&self, symbol: &str, depth: usize) -> Option<MarketDataSnapshot> {
        self.order_books
            .get(symbol)
            .map(|book| book.get_market_data(depth))
    }

    /// Get next trade from queue
    pub fn get_next_trade(&self) -> Option<Trade> {
        self.trade_queue.pop()
    }

    /// Get engine statistics
    pub fn get_statistics(&self) -> EngineStatistics {
        EngineStatistics {
            total_orders_processed: self.total_orders_processed.load(Ordering::Acquire),
            total_trades_generated: self.total_trades_generated.load(Ordering::Acquire),
            total_volume_matched: self.total_volume_matched.load(Ordering::Acquire),
            avg_matching_time_ns: self.avg_matching_time_ns.load(Ordering::Acquire),
            max_matching_time_ns: self.max_matching_time_ns.load(Ordering::Acquire),
            active_symbols: self.order_books.len(),
            pending_trades: self.trade_queue.len(),
        }
    }

    fn get_timestamp_ns(&self) -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
}

/// Engine performance statistics
#[derive(Debug, Clone)]
pub struct EngineStatistics {
    pub total_orders_processed: u64,
    pub total_trades_generated: u64,
    pub total_volume_matched: u64,
    pub avg_matching_time_ns: u64,
    pub max_matching_time_ns: u64,
    pub active_symbols: usize,
    pub pending_trades: usize,
}

// Thread safety
unsafe impl Send for OrderMatchingEngine {}
unsafe impl Sync for OrderMatchingEngine {}
unsafe impl Send for Order {}
unsafe impl Sync for Order {}
unsafe impl Send for PriceLevel {}
unsafe impl Sync for PriceLevel {}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_order(id: u64, side: Side, price: u64, qty: u64) -> Order {
        Order::new(
            id,
            "BTCUSD".to_string(),
            side,
            OrderType::Limit,
            TimeInForce::GoodTillCancel,
            price,
            qty,
            1000000000, // 1 second timestamp
        )
    }

    #[test]
    fn test_order_creation() {
        let order = create_test_order(1, Side::Buy, 50000_000000, 1_000000); // $50,000, 1 BTC
        assert_eq!(order.order_id, 1);
        assert_eq!(order.side, Side::Buy);
        assert_eq!(order.price, 50000_000000);
        assert_eq!(order.original_quantity, 1_000000);
    }

    #[test]
    fn test_order_matching_engine() {
        let mut engine = OrderMatchingEngine::new(MatchingAlgorithm::PriceTimePriority);

        // Add buy order
        let buy_order = create_test_order(1, Side::Buy, 50000_000000, 1_000000);
        let trades = engine.add_order(buy_order).unwrap();
        assert!(trades.is_empty()); // No matching order yet

        // Add sell order at same price
        let sell_order = create_test_order(2, Side::Sell, 50000_000000, 1_000000);
        let trades = engine.add_order(sell_order).unwrap();
        assert_eq!(trades.len(), 1); // Should match

        let trade = &trades[0];
        assert_eq!(trade.buy_order_id, 1);
        assert_eq!(trade.sell_order_id, 2);
        assert_eq!(trade.price, 50000_000000);
        assert_eq!(trade.quantity, 1_000000);
    }

    #[test]
    fn test_partial_fill() {
        let mut engine = OrderMatchingEngine::new(MatchingAlgorithm::PriceTimePriority);

        // Add large buy order
        let buy_order = create_test_order(1, Side::Buy, 50000_000000, 5_000000);
        engine.add_order(buy_order).unwrap();

        // Add smaller sell order
        let sell_order = create_test_order(2, Side::Sell, 50000_000000, 2_000000);
        let trades = engine.add_order(sell_order).unwrap();

        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].quantity, 2_000000);

        // The buy order should still have 3 BTC remaining
        let market_data = engine.get_market_data("BTCUSD", 5).unwrap();
        assert_eq!(market_data.bids[0].quantity, 3_000000);
    }

    #[test]
    fn test_market_order() {
        let mut engine = OrderMatchingEngine::new(MatchingAlgorithm::PriceTimePriority);

        // Add limit sell orders at different prices
        let sell1 = Order::new(
            1,
            "BTCUSD".to_string(),
            Side::Sell,
            OrderType::Limit,
            TimeInForce::GoodTillCancel,
            50000_000000,
            1_000000,
            1000000000,
        );
        let sell2 = Order::new(
            2,
            "BTCUSD".to_string(),
            Side::Sell,
            OrderType::Limit,
            TimeInForce::GoodTillCancel,
            50001_000000,
            1_000000,
            1000000001,
        );

        engine.add_order(sell1).unwrap();
        engine.add_order(sell2).unwrap();

        // Add market buy order
        let market_buy = Order::new(
            3,
            "BTCUSD".to_string(),
            Side::Buy,
            OrderType::Market,
            TimeInForce::ImmediateOrCancel,
            0,
            1_500000,
            1000000002,
        );

        let trades = engine.add_order(market_buy).unwrap();

        assert_eq!(trades.len(), 2); // Should match both sell orders
        assert_eq!(trades[0].price, 50000_000000); // First trade at better price
        assert_eq!(trades[0].quantity, 1_000000);
        assert_eq!(trades[1].price, 50001_000000); // Second trade at worse price
        assert_eq!(trades[1].quantity, 500000);
    }

    #[test]
    fn test_fill_or_kill() {
        let mut engine = OrderMatchingEngine::new(MatchingAlgorithm::PriceTimePriority);

        // Add sell order with insufficient quantity
        let sell_order = create_test_order(1, Side::Sell, 50000_000000, 500000);
        engine.add_order(sell_order).unwrap();

        // Add FOK buy order for more than available
        let fok_order = Order::new(
            2,
            "BTCUSD".to_string(),
            Side::Buy,
            OrderType::FillOrKill,
            TimeInForce::FillOrKill,
            50000_000000,
            1_000000,
            1000000000,
        );

        let trades = engine.add_order(fok_order).unwrap();
        assert!(trades.is_empty()); // Should be cancelled due to insufficient liquidity
    }

    #[test]
    fn test_iceberg_order() {
        let iceberg = Order::new(
            1,
            "BTCUSD".to_string(),
            Side::Sell,
            OrderType::Iceberg,
            TimeInForce::GoodTillCancel,
            50000_000000,
            10_000000,
            1000000000,
        );

        // Iceberg should only display 10% initially
        assert_eq!(iceberg.displayed_quantity.load(Ordering::Acquire), 1_000000);
        assert_eq!(iceberg.hidden_quantity, 9_000000);
    }

    #[test]
    fn test_order_cancellation() {
        let mut engine = OrderMatchingEngine::new(MatchingAlgorithm::PriceTimePriority);

        // Add order
        let order = create_test_order(1, Side::Buy, 50000_000000, 1_000000);
        engine.add_order(order).unwrap();

        // Verify order is in book
        let market_data = engine.get_market_data("BTCUSD", 5).unwrap();
        assert_eq!(market_data.bids.len(), 1);

        // Cancel order
        let result = engine.cancel_order("BTCUSD", 1).unwrap();
        assert!(result);

        // Verify order is removed
        let market_data = engine.get_market_data("BTCUSD", 5).unwrap();
        assert_eq!(market_data.bids.len(), 0);
    }

    #[test]
    fn test_engine_statistics() {
        let mut engine = OrderMatchingEngine::new(MatchingAlgorithm::PriceTimePriority);

        // Add some orders
        let buy_order = create_test_order(1, Side::Buy, 50000_000000, 1_000000);
        engine.add_order(buy_order).unwrap();

        let sell_order = create_test_order(2, Side::Sell, 50000_000000, 1_000000);
        engine.add_order(sell_order).unwrap();

        let stats = engine.get_statistics();
        assert_eq!(stats.total_orders_processed, 2);
        assert_eq!(stats.total_trades_generated, 1);
        assert_eq!(stats.total_volume_matched, 1_000000);
        assert!(stats.avg_matching_time_ns > 0);
    }
}
