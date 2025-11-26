// Lock-free Order Book - REAL IMPLEMENTATION with atomic operations
use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};
// Removed unused std::mem import
use crossbeam::utils::CachePadded;
use std::alloc::{alloc, dealloc, Layout};

const MAX_PRICE_LEVELS: usize = 10000;
const ORDERS_PER_LEVEL: usize = 1024;

/// Atomic order structure - cache-aligned
#[repr(C, align(64))]
pub struct AtomicOrder {
    pub price: AtomicU64,             // Price in micropips
    pub quantity: AtomicU64,          // Quantity in micro-units
    pub order_id: AtomicU64,          // Unique order ID
    pub timestamp: AtomicU64,         // Nanosecond timestamp
    pub next: AtomicPtr<AtomicOrder>, // Next order in level
}

impl AtomicOrder {
    pub fn new(price: u64, quantity: u64, order_id: u64, timestamp: u64) -> Self {
        Self {
            price: AtomicU64::new(price),
            quantity: AtomicU64::new(quantity),
            order_id: AtomicU64::new(order_id),
            timestamp: AtomicU64::new(timestamp),
            next: AtomicPtr::new(ptr::null_mut()),
        }
    }
}

/// Lock-free price level
#[repr(C, align(64))]
pub struct PriceLevel {
    pub price: AtomicU64,
    pub total_quantity: AtomicU64,
    pub order_count: AtomicUsize,
    pub head: AtomicPtr<AtomicOrder>,
    pub tail: AtomicPtr<AtomicOrder>,
}

impl PriceLevel {
    pub fn new(price: u64) -> Self {
        Self {
            price: AtomicU64::new(price),
            total_quantity: AtomicU64::new(0),
            order_count: AtomicUsize::new(0),
            head: AtomicPtr::new(ptr::null_mut()),
            tail: AtomicPtr::new(ptr::null_mut()),
        }
    }

    /// Lock-free insertion at tail
    pub fn insert(&self, order: *mut AtomicOrder) -> bool {
        unsafe {
            let order_ref = &*order;
            let quantity = order_ref.quantity.load(Ordering::Acquire);

            loop {
                let tail = self.tail.load(Ordering::Acquire);

                if tail.is_null() {
                    // Empty level - try to set both head and tail
                    let null = ptr::null_mut();

                    match self.head.compare_exchange_weak(
                        null,
                        order,
                        Ordering::Release,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => {
                            self.tail.store(order, Ordering::Release);
                            self.total_quantity.fetch_add(quantity, Ordering::AcqRel);
                            self.order_count.fetch_add(1, Ordering::AcqRel);
                            return true;
                        }
                        Err(_) => continue, // Retry
                    }
                } else {
                    // Non-empty level - append to tail
                    let tail_ref = &*tail;
                    let null = ptr::null_mut();

                    match tail_ref.next.compare_exchange_weak(
                        null,
                        order,
                        Ordering::Release,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => {
                            // Try to update tail pointer
                            let _ = self.tail.compare_exchange_weak(
                                tail,
                                order,
                                Ordering::Release,
                                Ordering::Relaxed,
                            );

                            self.total_quantity.fetch_add(quantity, Ordering::AcqRel);
                            self.order_count.fetch_add(1, Ordering::AcqRel);
                            return true;
                        }
                        Err(_) => {
                            // Help update tail if needed
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

    /// Lock-free removal from head (FIFO)
    pub fn remove(&self) -> Option<*mut AtomicOrder> {
        loop {
            let head = self.head.load(Ordering::Acquire);

            if head.is_null() {
                return None;
            }

            unsafe {
                let head_ref = &*head;
                let next = head_ref.next.load(Ordering::Acquire);

                match self.head.compare_exchange_weak(
                    head,
                    next,
                    Ordering::Release,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // If we removed the last element, update tail
                        if next.is_null() {
                            let _ = self.tail.compare_exchange(
                                head,
                                ptr::null_mut(),
                                Ordering::Release,
                                Ordering::Relaxed,
                            );
                        }

                        let quantity = head_ref.quantity.load(Ordering::Acquire);
                        self.total_quantity.fetch_sub(quantity, Ordering::AcqRel);
                        self.order_count.fetch_sub(1, Ordering::AcqRel);

                        return Some(head);
                    }
                    Err(_) => continue, // Retry
                }
            }
        }
    }

    /// Match orders at this price level
    pub fn match_quantity(&self, requested_qty: u64) -> Vec<(u64, u64, u64)> {
        let mut matched = Vec::new();
        let mut remaining = requested_qty;

        let mut current = self.head.load(Ordering::Acquire);

        while !current.is_null() && remaining > 0 {
            unsafe {
                let order = &*current;
                let order_qty = order.quantity.load(Ordering::Acquire);
                let order_id = order.order_id.load(Ordering::Acquire);
                let price = order.price.load(Ordering::Acquire);

                if order_qty > 0 {
                    let match_qty = order_qty.min(remaining);

                    // Try to update order quantity atomically
                    let new_qty = order_qty - match_qty;

                    match order.quantity.compare_exchange(
                        order_qty,
                        new_qty,
                        Ordering::AcqRel,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => {
                            matched.push((order_id, match_qty, price));
                            remaining -= match_qty;

                            // Update total quantity
                            self.total_quantity.fetch_sub(match_qty, Ordering::AcqRel);

                            if new_qty == 0 {
                                // Order fully matched, remove it
                                self.remove();
                            }
                        }
                        Err(_) => {
                            // Another thread modified the order, retry
                            continue;
                        }
                    }
                }

                current = order.next.load(Ordering::Acquire);
            }
        }

        matched
    }
}

/// Lock-free order book
pub struct LockFreeOrderBook {
    bid_levels: Vec<CachePadded<PriceLevel>>,
    ask_levels: Vec<CachePadded<PriceLevel>>,
    best_bid: AtomicU64,
    best_ask: AtomicU64,
    order_pool: OrderPool,
}

/// Memory pool for orders
struct OrderPool {
    allocated: std::sync::Mutex<Vec<*mut AtomicOrder>>, // Track all allocations for cleanup
    free_list: AtomicPtr<AtomicOrder>,
}

impl OrderPool {
    fn new(_capacity: usize) -> Self {
        // Lazy allocation - don't pre-allocate to avoid 655MB+ memory usage
        // Orders are allocated on-demand in allocate()
        Self {
            allocated: std::sync::Mutex::new(Vec::new()),
            free_list: AtomicPtr::new(ptr::null_mut()),
        }
    }

    fn allocate(&self) -> *mut AtomicOrder {
        // Try to get from free list first
        loop {
            let head = self.free_list.load(Ordering::Acquire);

            if head.is_null() {
                // Allocate new order and track it
                let layout = Layout::new::<AtomicOrder>();
                let ptr = unsafe { alloc(layout) as *mut AtomicOrder };
                if let Ok(mut allocated) = self.allocated.lock() {
                    allocated.push(ptr);
                }
                return ptr;
            }

            unsafe {
                let next = (*head).next.load(Ordering::Acquire);

                match self.free_list.compare_exchange_weak(
                    head,
                    next,
                    Ordering::Release,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return head,
                    Err(_) => continue,
                }
            }
        }
    }

    #[allow(dead_code)]
    fn deallocate(&self, order: *mut AtomicOrder) {
        unsafe {
            // Add to free list for reuse
            loop {
                let head = self.free_list.load(Ordering::Acquire);
                (*order).next.store(head, Ordering::Release);

                match self.free_list.compare_exchange_weak(
                    head,
                    order,
                    Ordering::Release,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return,
                    Err(_) => continue,
                }
            }
        }
    }
}

impl Drop for OrderPool {
    fn drop(&mut self) {
        let layout = Layout::new::<AtomicOrder>();

        // Deallocate all tracked allocations
        if let Ok(allocated) = self.allocated.lock() {
            for &ptr in allocated.iter() {
                if !ptr.is_null() {
                    unsafe { dealloc(ptr as *mut u8, layout); }
                }
            }
        }
    }
}

impl Default for LockFreeOrderBook {
    fn default() -> Self {
        Self::new()
    }
}

impl LockFreeOrderBook {
    pub fn new() -> Self {
        let mut bid_levels = Vec::with_capacity(MAX_PRICE_LEVELS);
        let mut ask_levels = Vec::with_capacity(MAX_PRICE_LEVELS);

        // Initialize price levels
        for i in 0..MAX_PRICE_LEVELS {
            bid_levels.push(CachePadded::new(PriceLevel::new(i as u64)));
            ask_levels.push(CachePadded::new(PriceLevel::new(i as u64)));
        }

        Self {
            bid_levels,
            ask_levels,
            best_bid: AtomicU64::new(0),
            best_ask: AtomicU64::new(u64::MAX),
            order_pool: OrderPool::new(ORDERS_PER_LEVEL * MAX_PRICE_LEVELS),
        }
    }

    /// Add buy order
    pub fn add_bid(&self, price: u64, quantity: u64, order_id: u64) -> bool {
        let timestamp = self.get_timestamp_ns();
        let order = self.order_pool.allocate();

        unsafe {
            ptr::write(
                order,
                AtomicOrder::new(price, quantity, order_id, timestamp),
            );
        }

        let level_idx = (price as usize).min(MAX_PRICE_LEVELS - 1);
        let success = self.bid_levels[level_idx].insert(order);

        if success {
            // Update best bid if necessary
            loop {
                let current_best = self.best_bid.load(Ordering::Acquire);
                if price <= current_best {
                    break;
                }

                match self.best_bid.compare_exchange_weak(
                    current_best,
                    price,
                    Ordering::Release,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(_) => continue,
                }
            }
        }

        success
    }

    /// Add sell order
    pub fn add_ask(&self, price: u64, quantity: u64, order_id: u64) -> bool {
        let timestamp = self.get_timestamp_ns();
        let order = self.order_pool.allocate();

        unsafe {
            ptr::write(
                order,
                AtomicOrder::new(price, quantity, order_id, timestamp),
            );
        }

        let level_idx = (price as usize).min(MAX_PRICE_LEVELS - 1);
        let success = self.ask_levels[level_idx].insert(order);

        if success {
            // Update best ask if necessary
            loop {
                let current_best = self.best_ask.load(Ordering::Acquire);
                if price >= current_best {
                    break;
                }

                match self.best_ask.compare_exchange_weak(
                    current_best,
                    price,
                    Ordering::Release,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(_) => continue,
                }
            }
        }

        success
    }

    /// Execute market order
    pub fn execute_market_order(&self, is_buy: bool, quantity: u64) -> Vec<(u64, u64, u64)> {
        if is_buy {
            // Buy from asks
            let mut remaining = quantity;
            let mut executions = Vec::new();

            for level in &self.ask_levels {
                if remaining == 0 {
                    break;
                }

                let matched = level.match_quantity(remaining);
                for (order_id, qty, price) in matched {
                    executions.push((order_id, qty, price));
                    remaining = remaining.saturating_sub(qty);
                }
            }

            executions
        } else {
            // Sell to bids
            let mut remaining = quantity;
            let mut executions = Vec::new();

            for level in self.bid_levels.iter().rev() {
                if remaining == 0 {
                    break;
                }

                let matched = level.match_quantity(remaining);
                for (order_id, qty, price) in matched {
                    executions.push((order_id, qty, price));
                    remaining = remaining.saturating_sub(qty);
                }
            }

            executions
        }
    }

    /// Get current spread
    pub fn get_spread(&self) -> (u64, u64) {
        let bid = self.best_bid.load(Ordering::Acquire);
        let ask = self.best_ask.load(Ordering::Acquire);
        (bid, ask)
    }

    /// Get order book depth
    pub fn get_depth(&self, levels: usize) -> (Vec<(u64, u64)>, Vec<(u64, u64)>) {
        let mut bids = Vec::new();
        let mut asks = Vec::new();

        // Collect bid levels
        for level in self.bid_levels.iter().rev().take(levels) {
            let price = level.price.load(Ordering::Acquire);
            let quantity = level.total_quantity.load(Ordering::Acquire);
            if quantity > 0 {
                bids.push((price, quantity));
            }
        }

        // Collect ask levels
        for level in self.ask_levels.iter().take(levels) {
            let price = level.price.load(Ordering::Acquire);
            let quantity = level.total_quantity.load(Ordering::Acquire);
            if quantity > 0 {
                asks.push((price, quantity));
            }
        }

        (bids, asks)
    }

    fn get_timestamp_ns(&self) -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
}

unsafe impl Send for LockFreeOrderBook {}
unsafe impl Sync for LockFreeOrderBook {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_order_book_creation() {
        let ob = LockFreeOrderBook::new();
        let (bid, ask) = ob.get_spread();
        assert_eq!(bid, 0);
        assert_eq!(ask, u64::MAX);
    }

    #[test]
    fn test_add_orders() {
        let ob = LockFreeOrderBook::new();

        // Add buy orders
        assert!(ob.add_bid(100, 1000, 1));
        assert!(ob.add_bid(99, 2000, 2));
        assert!(ob.add_bid(101, 500, 3));

        // Add sell orders
        assert!(ob.add_ask(102, 1500, 4));
        assert!(ob.add_ask(103, 1000, 5));
        assert!(ob.add_ask(101, 2000, 6));

        let (bid, ask) = ob.get_spread();
        assert_eq!(bid, 101);
        assert_eq!(ask, 101);
    }

    #[test]
    fn test_market_order_execution() {
        let ob = LockFreeOrderBook::new();

        // Add liquidity
        ob.add_ask(100, 1000, 1);
        ob.add_ask(101, 2000, 2);
        ob.add_ask(102, 3000, 3);

        // Execute buy market order
        let executions = ob.execute_market_order(true, 2500);

        assert!(!executions.is_empty());
        let total_executed: u64 = executions.iter().map(|(_, qty, _)| qty).sum();
        assert_eq!(total_executed, 2500);
    }

    #[test]
    fn test_concurrent_operations() {
        let ob = Arc::new(LockFreeOrderBook::new());
        let mut handles = vec![];

        // Spawn multiple threads adding orders
        for i in 0..10 {
            let ob_clone = ob.clone();
            let handle = thread::spawn(move || {
                for j in 0..100 {
                    let order_id = (i * 100 + j) as u64;
                    if i % 2 == 0 {
                        ob_clone.add_bid(100 + (j % 10), 100, order_id);
                    } else {
                        ob_clone.add_ask(100 + (j % 10), 100, order_id);
                    }
                }
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify order book has orders using get_spread()
        // best_bid and best_ask atomics are updated when orders are added
        let (best_bid, best_ask) = ob.get_spread();

        // After concurrent adds: bids at 100-109 should set best_bid to 109
        // asks at 100-109 should set best_ask to 100
        assert!(best_bid >= 100 && best_bid <= 109, "best_bid should be in range 100-109, got {}", best_bid);
        assert!(best_ask >= 100 && best_ask <= 109, "best_ask should be in range 100-109, got {}", best_ask);
    }
}
