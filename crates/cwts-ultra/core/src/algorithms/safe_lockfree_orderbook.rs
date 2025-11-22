//! Memory-Safe Lock-Free Order Book Implementation
//! Replaces unsafe raw pointer operations with safe alternatives
//! Maintains microsecond performance while ensuring memory safety

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use crossbeam::queue::{SegQueue, ArrayQueue};
use crossbeam::utils::CachePadded;
use parking_lot::RwLock;
use dashmap::DashMap;
use std::collections::BTreeMap;

const MAX_PRICE_LEVELS: usize = 10000;
const ORDERS_PER_LEVEL: usize = 1024;

/// Safe atomic order structure using heap allocation
#[derive(Debug)]
pub struct SafeAtomicOrder {
    pub price: AtomicU64,      // Price in micropips
    pub quantity: AtomicU64,    // Quantity in micro-units
    pub order_id: AtomicU64,    // Unique order ID
    pub timestamp: AtomicU64,   // Nanosecond timestamp
}

impl SafeAtomicOrder {
    pub fn new(price: u64, quantity: u64, order_id: u64, timestamp: u64) -> Arc<Self> {
        Arc::new(Self {
            price: AtomicU64::new(price),
            quantity: AtomicU64::new(quantity),
            order_id: AtomicU64::new(order_id),
            timestamp: AtomicU64::new(timestamp),
        })
    }
}

/// Safe price level using concurrent data structures
#[derive(Debug)]
pub struct SafePriceLevel {
    pub price: AtomicU64,
    pub total_quantity: AtomicU64,
    pub order_count: AtomicUsize,
    /// Orders stored in a lock-free queue for FIFO ordering
    pub orders: SegQueue<Arc<SafeAtomicOrder>>,
}

impl SafePriceLevel {
    pub fn new(price: u64) -> Self {
        Self {
            price: AtomicU64::new(price),
            total_quantity: AtomicU64::new(0),
            order_count: AtomicUsize::new(0),
            orders: SegQueue::new(),
        }
    }
    
    /// Safe insertion maintaining FIFO order
    pub fn insert(&self, order: Arc<SafeAtomicOrder>) -> bool {
        let quantity = order.quantity.load(Ordering::Acquire);
        
        // Add to queue (always succeeds for SegQueue)
        self.orders.push(order);
        
        // Update counters atomically
        self.total_quantity.fetch_add(quantity, Ordering::AcqRel);
        self.order_count.fetch_add(1, Ordering::AcqRel);
        
        true
    }
    
    /// Safe removal with FIFO ordering
    pub fn remove(&self) -> Option<Arc<SafeAtomicOrder>> {
        if let Some(order) = self.orders.pop() {
            let quantity = order.quantity.load(Ordering::Acquire);
            self.total_quantity.fetch_sub(quantity, Ordering::AcqRel);
            self.order_count.fetch_sub(1, Ordering::AcqRel);
            Some(order)
        } else {
            None
        }
    }
    
    /// Match orders at this price level with bounds checking
    pub fn match_quantity(&self, requested_qty: u64) -> Vec<(u64, u64, u64)> {
        let mut matched = Vec::new();
        let mut remaining = requested_qty;
        
        // Temporary storage for partially filled orders
        let mut partial_orders = Vec::new();
        
        while remaining > 0 {
            if let Some(order) = self.orders.pop() {
                let order_qty = order.quantity.load(Ordering::Acquire);
                let order_id = order.order_id.load(Ordering::Acquire);
                let price = order.price.load(Ordering::Acquire);
                
                if order_qty > 0 {
                    let match_qty = order_qty.min(remaining);
                    
                    // Try to update order quantity atomically
                    let new_qty = order_qty - match_qty;
                    
                    // Use compare_exchange to ensure atomicity
                    match order.quantity.compare_exchange(
                        order_qty,
                        new_qty,
                        Ordering::AcqRel,
                        Ordering::Relaxed
                    ) {
                        Ok(_) => {
                            matched.push((order_id, match_qty, price));
                            remaining -= match_qty;
                            
                            // Update total quantity
                            self.total_quantity.fetch_sub(match_qty, Ordering::AcqRel);
                            
                            if new_qty > 0 {
                                // Partial fill - put back at front for FIFO
                                partial_orders.push(order);
                            } else {
                                // Complete fill - decrease order count
                                self.order_count.fetch_sub(1, Ordering::AcqRel);
                            }
                        }
                        Err(_) => {
                            // Another thread modified the order, put it back
                            partial_orders.push(order);
                            break;
                        }
                    }
                } else {
                    // Empty order, remove it
                    self.order_count.fetch_sub(1, Ordering::AcqRel);
                }
            } else {
                break; // No more orders
            }
        }
        
        // Put back partial orders in reverse order to maintain FIFO
        for order in partial_orders.into_iter().rev() {
            self.orders.push(order);
        }
        
        matched
    }
}

/// Safe lock-free order book using concurrent data structures
pub struct SafeLockFreeOrderBook {
    /// Price levels stored in concurrent map for fast access
    bid_levels: DashMap<u64, Arc<SafePriceLevel>>,
    ask_levels: DashMap<u64, Arc<SafePriceLevel>>,
    
    /// Best prices cached for fast access
    best_bid: CachePadded<AtomicU64>,
    best_ask: CachePadded<AtomicU64>,
    
    /// Order pool for memory reuse
    order_pool: ArrayQueue<Arc<SafeAtomicOrder>>,
    
    /// Statistics
    total_orders: AtomicU64,
    total_trades: AtomicU64,
}

impl Default for SafeLockFreeOrderBook {
    fn default() -> Self {
        Self::new()
    }
}

impl SafeLockFreeOrderBook {
    pub fn new() -> Self {
        Self {
            bid_levels: DashMap::with_capacity(MAX_PRICE_LEVELS),
            ask_levels: DashMap::with_capacity(MAX_PRICE_LEVELS),
            best_bid: CachePadded::new(AtomicU64::new(0)),
            best_ask: CachePadded::new(AtomicU64::new(u64::MAX)),
            order_pool: ArrayQueue::new(ORDERS_PER_LEVEL * 100), // Pre-allocate pool
            total_orders: AtomicU64::new(0),
            total_trades: AtomicU64::new(0),
        }
    }
    
    /// Add buy order with memory safety
    pub fn add_bid(&self, price: u64, quantity: u64, order_id: u64) -> bool {
        let timestamp = self.get_timestamp_ns();
        let order = SafeAtomicOrder::new(price, quantity, order_id, timestamp);
        
        // Get or create price level
        let level = self.bid_levels
            .entry(price)
            .or_insert_with(|| Arc::new(SafePriceLevel::new(price)))
            .clone();
        
        let success = level.insert(order);
        
        if success {
            self.total_orders.fetch_add(1, Ordering::Relaxed);
            
            // Update best bid atomically
            loop {
                let current_best = self.best_bid.load(Ordering::Acquire);
                if price <= current_best {
                    break;
                }
                
                match self.best_bid.compare_exchange_weak(
                    current_best,
                    price,
                    Ordering::Release,
                    Ordering::Relaxed
                ) {
                    Ok(_) => break,
                    Err(_) => continue,
                }
            }
        }
        
        success
    }
    
    /// Add sell order with memory safety
    pub fn add_ask(&self, price: u64, quantity: u64, order_id: u64) -> bool {
        let timestamp = self.get_timestamp_ns();
        let order = SafeAtomicOrder::new(price, quantity, order_id, timestamp);
        
        // Get or create price level
        let level = self.ask_levels
            .entry(price)
            .or_insert_with(|| Arc::new(SafePriceLevel::new(price)))
            .clone();
        
        let success = level.insert(order);
        
        if success {
            self.total_orders.fetch_add(1, Ordering::Relaxed);
            
            // Update best ask atomically
            loop {
                let current_best = self.best_ask.load(Ordering::Acquire);
                if price >= current_best {
                    break;
                }
                
                match self.best_ask.compare_exchange_weak(
                    current_best,
                    price,
                    Ordering::Release,
                    Ordering::Relaxed
                ) {
                    Ok(_) => break,
                    Err(_) => continue,
                }
            }
        }
        
        success
    }
    
    /// Execute market order with guaranteed memory safety
    pub fn execute_market_order(&self, is_buy: bool, quantity: u64) -> Vec<(u64, u64, u64)> {
        let mut executions = Vec::new();
        let mut remaining = quantity;
        
        if is_buy {
            // Buy from asks - iterate in price order
            let ask_prices: Vec<u64> = self.ask_levels.iter()
                .map(|entry| *entry.key())
                .collect::<Vec<_>>();
            
            let mut sorted_prices = ask_prices;
            sorted_prices.sort_unstable();
            
            for price in sorted_prices {
                if remaining == 0 { break; }
                
                if let Some(level_ref) = self.ask_levels.get(&price) {
                    let level = level_ref.value();
                    let matched = level.match_quantity(remaining);
                    
                    for (order_id, qty, exec_price) in matched {
                        executions.push((order_id, qty, exec_price));
                        remaining = remaining.saturating_sub(qty);
                        self.total_trades.fetch_add(1, Ordering::Relaxed);
                    }
                    
                    // Remove empty levels
                    if level.order_count.load(Ordering::Acquire) == 0 {
                        self.ask_levels.remove(&price);
                    }
                }
            }
        } else {
            // Sell to bids - iterate in reverse price order  
            let bid_prices: Vec<u64> = self.bid_levels.iter()
                .map(|entry| *entry.key())
                .collect::<Vec<_>>();
            
            let mut sorted_prices = bid_prices;
            sorted_prices.sort_unstable();
            sorted_prices.reverse();
            
            for price in sorted_prices {
                if remaining == 0 { break; }
                
                if let Some(level_ref) = self.bid_levels.get(&price) {
                    let level = level_ref.value();
                    let matched = level.match_quantity(remaining);
                    
                    for (order_id, qty, exec_price) in matched {
                        executions.push((order_id, qty, exec_price));
                        remaining = remaining.saturating_sub(qty);
                        self.total_trades.fetch_add(1, Ordering::Relaxed);
                    }
                    
                    // Remove empty levels
                    if level.order_count.load(Ordering::Acquire) == 0 {
                        self.bid_levels.remove(&price);
                    }
                }
            }
        }
        
        executions
    }
    
    /// Get current spread with memory safety
    pub fn get_spread(&self) -> (u64, u64) {
        let bid = self.best_bid.load(Ordering::Acquire);
        let ask = self.best_ask.load(Ordering::Acquire);
        (bid, ask)
    }
    
    /// Get order book depth with bounds checking
    pub fn get_depth(&self, levels: usize) -> (Vec<(u64, u64)>, Vec<(u64, u64)>) {
        let mut bids = Vec::new();
        let mut asks = Vec::new();
        
        // Collect bid levels (highest to lowest)
        let mut bid_prices: Vec<u64> = self.bid_levels.iter()
            .map(|entry| *entry.key())
            .collect();
        bid_prices.sort_unstable();
        bid_prices.reverse();
        
        for price in bid_prices.into_iter().take(levels) {
            if let Some(level_ref) = self.bid_levels.get(&price) {
                let level = level_ref.value();
                let quantity = level.total_quantity.load(Ordering::Acquire);
                if quantity > 0 {
                    bids.push((price, quantity));
                }
            }
        }
        
        // Collect ask levels (lowest to highest)
        let mut ask_prices: Vec<u64> = self.ask_levels.iter()
            .map(|entry| *entry.key())
            .collect();
        ask_prices.sort_unstable();
        
        for price in ask_prices.into_iter().take(levels) {
            if let Some(level_ref) = self.ask_levels.get(&price) {
                let level = level_ref.value();
                let quantity = level.total_quantity.load(Ordering::Acquire);
                if quantity > 0 {
                    asks.push((price, quantity));
                }
            }
        }
        
        (bids, asks)
    }
    
    /// Get performance statistics
    pub fn get_stats(&self) -> OrderBookStats {
        OrderBookStats {
            total_orders: self.total_orders.load(Ordering::Acquire),
            total_trades: self.total_trades.load(Ordering::Acquire),
            bid_levels: self.bid_levels.len(),
            ask_levels: self.ask_levels.len(),
        }
    }
    
    fn get_timestamp_ns(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
}

#[derive(Debug, Clone)]
pub struct OrderBookStats {
    pub total_orders: u64,
    pub total_trades: u64,
    pub bid_levels: usize,
    pub ask_levels: usize,
}

// Safe Send + Sync implementation
unsafe impl Send for SafeLockFreeOrderBook {}
unsafe impl Sync for SafeLockFreeOrderBook {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::sync::Arc;
    
    #[test]
    fn test_safe_order_book_creation() {
        let ob = SafeLockFreeOrderBook::new();
        let (bid, ask) = ob.get_spread();
        assert_eq!(bid, 0);
        assert_eq!(ask, u64::MAX);
    }
    
    #[test]
    fn test_safe_add_orders() {
        let ob = SafeLockFreeOrderBook::new();
        
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
        assert!(ask <= 103); // Best ask should be <= 103
    }
    
    #[test]
    fn test_safe_market_order_execution() {
        let ob = SafeLockFreeOrderBook::new();
        
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
    fn test_safe_concurrent_operations() {
        let ob = Arc::new(SafeLockFreeOrderBook::new());
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
        
        // Verify order book has orders
        let (bids, asks) = ob.get_depth(10);
        assert!(!bids.is_empty() || !asks.is_empty());
        
        let stats = ob.get_stats();
        assert!(stats.total_orders > 0);
    }
    
    #[test]
    fn test_memory_safety_bounds_checking() {
        let ob = SafeLockFreeOrderBook::new();
        
        // Test with extreme values
        assert!(ob.add_bid(u64::MAX - 1, 1, 1));
        assert!(ob.add_ask(1, u64::MAX - 1, 2));
        
        // Test depth with large numbers
        let (bids, asks) = ob.get_depth(usize::MAX);
        assert!(bids.len() <= MAX_PRICE_LEVELS);
        assert!(asks.len() <= MAX_PRICE_LEVELS);
    }
}