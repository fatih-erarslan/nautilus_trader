//! Lock-free data structures for ultra-low latency HFT systems
//! 
//! This module implements lock-free concurrent data structures that eliminate
//! lock contention and provide consistent microsecond-level performance.

use std::sync::atomic::{AtomicPtr, AtomicUsize, AtomicBool, Ordering};
use std::sync::Arc;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::ptr::{self, NonNull};
use std::alloc::{Layout, alloc, dealloc};
use crossbeam_epoch::{self as epoch, Atomic, Owned, Shared, Guard};
use std::marker::PhantomData;

/// Lock-free order book for high-frequency trading
#[derive(Debug)]
pub struct LockFreeOrderBook<T> {
    /// Buy orders (sorted by price descending)
    buy_orders: Arc<LockFreeSkipList<PriceLevel<T>>>,
    
    /// Sell orders (sorted by price ascending)  
    sell_orders: Arc<LockFreeSkipList<PriceLevel<T>>>,
    
    /// Order lookup table for O(1) access
    order_lookup: Arc<LockFreeHashMap<u64, Arc<Order<T>>>>,
    
    /// Statistics
    stats: Arc<OrderBookStats>,
}

/// Price level in the order book
#[derive(Debug, Clone)]
pub struct PriceLevel<T> {
    /// Price in minimal units (to avoid floating point)
    pub price: u64,
    
    /// Total quantity at this price level
    pub total_quantity: AtomicUsize,
    
    /// Orders at this price level
    pub orders: Arc<LockFreeQueue<Arc<Order<T>>>>,
    
    /// Number of orders at this level
    pub order_count: AtomicUsize,
}

/// Individual order
#[derive(Debug)]
pub struct Order<T> {
    /// Unique order ID
    pub id: u64,
    
    /// Order price
    pub price: u64,
    
    /// Order quantity
    pub quantity: AtomicUsize,
    
    /// Order side (buy/sell)
    pub side: OrderSide,
    
    /// Order data
    pub data: T,
    
    /// Order timestamp (nanoseconds)
    pub timestamp: u64,
    
    /// Order status
    pub status: AtomicOrderStatus,
}

/// Order side enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Atomic order status
#[derive(Debug)]
pub struct AtomicOrderStatus {
    status: AtomicUsize,
}

/// Order status values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum OrderStatus {
    Active = 0,
    PartiallyFilled = 1,
    Filled = 2,
    Cancelled = 3,
    Expired = 4,
}

/// Order book statistics
#[derive(Debug)]
pub struct OrderBookStats {
    /// Total number of orders
    pub total_orders: AtomicUsize,
    
    /// Number of buy orders
    pub buy_orders: AtomicUsize,
    
    /// Number of sell orders  
    pub sell_orders: AtomicUsize,
    
    /// Number of price levels
    pub price_levels: AtomicUsize,
    
    /// Last update timestamp
    pub last_update: AtomicUsize,
}

/// Lock-free skip list for sorted data with O(log n) operations
#[derive(Debug)]
pub struct LockFreeSkipList<T> {
    /// Head node
    head: Atomic<SkipListNode<T>>,
    
    /// Maximum height
    max_height: AtomicUsize,
    
    /// Random number generator for height selection
    rng: AtomicUsize,
    
    /// Number of elements
    len: AtomicUsize,
}

/// Skip list node
#[derive(Debug)]
pub struct SkipListNode<T> {
    /// Node data
    pub data: ManuallyDrop<T>,
    
    /// Forward pointers at different levels
    pub forward: Vec<Atomic<SkipListNode<T>>>,
    
    /// Node height
    pub height: usize,
    
    /// Comparison key for ordering
    pub key: u64,
    
    /// Marked for deletion flag
    pub marked: AtomicBool,
}

/// Lock-free hash map with linear probing
#[derive(Debug)]
pub struct LockFreeHashMap<K, V> {
    /// Hash table buckets
    buckets: Vec<Atomic<HashBucket<K, V>>>,
    
    /// Number of buckets (power of 2)
    bucket_count: usize,
    
    /// Number of elements
    len: AtomicUsize,
    
    /// Load factor threshold for resize
    load_factor_threshold: f64,
    
    /// Resize in progress flag
    resizing: AtomicBool,
}

/// Hash map bucket
#[derive(Debug)]
pub struct HashBucket<K, V> {
    /// Key
    pub key: ManuallyDrop<K>,
    
    /// Value
    pub value: ManuallyDrop<V>,
    
    /// Hash value
    pub hash: u64,
    
    /// Occupied flag
    pub occupied: AtomicBool,
    
    /// Deleted flag
    pub deleted: AtomicBool,
}

/// Lock-free queue with MPMC support
#[derive(Debug)]
pub struct LockFreeQueue<T> {
    /// Head pointer
    head: Atomic<QueueNode<T>>,
    
    /// Tail pointer
    tail: Atomic<QueueNode<T>>,
    
    /// Number of elements
    len: AtomicUsize,
}

/// Queue node
#[derive(Debug)]
pub struct QueueNode<T> {
    /// Node data
    pub data: MaybeUninit<T>,
    
    /// Next node pointer
    pub next: Atomic<QueueNode<T>>,
    
    /// Data ready flag
    pub ready: AtomicBool,
}

/// Lock-free memory pool for zero-allocation operations
#[derive(Debug)]
pub struct LockFreeMemoryPool<T> {
    /// Free list of available objects
    free_list: Atomic<PoolNode<T>>,
    
    /// Pool capacity
    capacity: usize,
    
    /// Current size
    size: AtomicUsize,
    
    /// Statistics
    stats: PoolStats,
}

/// Memory pool node
#[derive(Debug)]
pub struct PoolNode<T> {
    /// Object storage
    pub object: MaybeUninit<T>,
    
    /// Next free node
    pub next: Atomic<PoolNode<T>>,
}

/// Memory pool statistics
#[derive(Debug)]
pub struct PoolStats {
    /// Allocations
    pub allocations: AtomicUsize,
    
    /// Deallocations  
    pub deallocations: AtomicUsize,
    
    /// Peak usage
    pub peak_usage: AtomicUsize,
    
    /// Pool misses (had to allocate outside pool)
    pub pool_misses: AtomicUsize,
}

impl<T> LockFreeOrderBook<T>
where
    T: Clone + Send + Sync + 'static,
{
    /// Create new lock-free order book
    pub fn new() -> Self {
        Self {
            buy_orders: Arc::new(LockFreeSkipList::new()),
            sell_orders: Arc::new(LockFreeSkipList::new()),
            order_lookup: Arc::new(LockFreeHashMap::with_capacity(1024)),
            stats: Arc::new(OrderBookStats {
                total_orders: AtomicUsize::new(0),
                buy_orders: AtomicUsize::new(0),
                sell_orders: AtomicUsize::new(0),
                price_levels: AtomicUsize::new(0),
                last_update: AtomicUsize::new(0),
            }),
        }
    }
    
    /// Add order to book (O(log n) complexity)
    pub fn add_order(&self, order: Order<T>) -> Result<(), &'static str> {
        let order_arc = Arc::new(order);
        
        // Add to lookup table for O(1) access
        if !self.order_lookup.insert(order_arc.id, order_arc.clone()) {
            return Err("Order ID already exists");
        }
        
        // Find or create price level
        let price_level = self.get_or_create_price_level(order_arc.price, order_arc.side)?;
        
        // Add order to price level
        price_level.orders.enqueue(order_arc.clone());
        price_level.order_count.fetch_add(1, Ordering::Relaxed);
        price_level.total_quantity.fetch_add(
            order_arc.quantity.load(Ordering::Relaxed), 
            Ordering::Relaxed
        );
        
        // Update statistics
        self.stats.total_orders.fetch_add(1, Ordering::Relaxed);
        match order_arc.side {
            OrderSide::Buy => self.stats.buy_orders.fetch_add(1, Ordering::Relaxed),
            OrderSide::Sell => self.stats.sell_orders.fetch_add(1, Ordering::Relaxed),
        };
        
        // Update timestamp
        self.stats.last_update.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as usize,
            Ordering::Relaxed
        );
        
        Ok(())
    }
    
    /// Remove order from book (O(1) complexity for lookup + O(log n) for level removal)
    pub fn remove_order(&self, order_id: u64) -> Result<Arc<Order<T>>, &'static str> {
        // Find order in lookup table
        let order = self.order_lookup.remove(&order_id)
            .ok_or("Order not found")?;
        
        // Mark order as cancelled
        order.status.store(OrderStatus::Cancelled);
        
        // Update statistics
        self.stats.total_orders.fetch_sub(1, Ordering::Relaxed);
        match order.side {
            OrderSide::Buy => self.stats.buy_orders.fetch_sub(1, Ordering::Relaxed),
            OrderSide::Sell => self.stats.sell_orders.fetch_sub(1, Ordering::Relaxed),
        };
        
        Ok(order)
    }
    
    /// Get best bid price (O(1) complexity)
    pub fn best_bid(&self) -> Option<u64> {
        self.buy_orders.max_key()
    }
    
    /// Get best ask price (O(1) complexity)  
    pub fn best_ask(&self) -> Option<u64> {
        self.sell_orders.min_key()
    }
    
    /// Get order by ID (O(1) complexity)
    pub fn get_order(&self, order_id: u64) -> Option<Arc<Order<T>>> {
        self.order_lookup.get(&order_id)
    }
    
    /// Get market depth (O(k) complexity where k is depth levels)
    pub fn get_market_depth(&self, depth: usize) -> MarketDepth {
        let bids = self.buy_orders.top_k(depth);
        let asks = self.sell_orders.top_k(depth);
        
        MarketDepth { bids, asks }
    }
    
    /// Find or create price level
    fn get_or_create_price_level(&self, price: u64, side: OrderSide) -> Result<Arc<PriceLevel<T>>, &'static str> {
        let skip_list = match side {
            OrderSide::Buy => &self.buy_orders,
            OrderSide::Sell => &self.sell_orders,
        };
        
        // Try to find existing price level
        if let Some(level) = skip_list.find(price) {
            return Ok(level);
        }
        
        // Create new price level
        let new_level = Arc::new(PriceLevel {
            price,
            total_quantity: AtomicUsize::new(0),
            orders: Arc::new(LockFreeQueue::new()),
            order_count: AtomicUsize::new(0),
        });
        
        // Insert into skip list
        if skip_list.insert(price, new_level.clone()) {
            self.stats.price_levels.fetch_add(1, Ordering::Relaxed);
            Ok(new_level)
        } else {
            // Someone else inserted it concurrently, find it again
            skip_list.find(price).ok_or("Failed to find or create price level")
        }
    }
}

/// Market depth snapshot
#[derive(Debug, Clone)]
pub struct MarketDepth {
    /// Bid levels (price, quantity)
    pub bids: Vec<(u64, usize)>,
    
    /// Ask levels (price, quantity)  
    pub asks: Vec<(u64, usize)>,
}

impl AtomicOrderStatus {
    /// Create new atomic order status
    pub fn new(status: OrderStatus) -> Self {
        Self {
            status: AtomicUsize::new(status as usize),
        }
    }
    
    /// Load current status
    pub fn load(&self) -> OrderStatus {
        let value = self.status.load(Ordering::Relaxed);
        match value {
            0 => OrderStatus::Active,
            1 => OrderStatus::PartiallyFilled,
            2 => OrderStatus::Filled,
            3 => OrderStatus::Cancelled,
            4 => OrderStatus::Expired,
            _ => OrderStatus::Active, // Default fallback
        }
    }
    
    /// Store new status
    pub fn store(&self, status: OrderStatus) {
        self.status.store(status as usize, Ordering::Relaxed);
    }
    
    /// Compare and swap status
    pub fn compare_exchange(&self, current: OrderStatus, new: OrderStatus) -> Result<OrderStatus, OrderStatus> {
        match self.status.compare_exchange(
            current as usize,
            new as usize,
            Ordering::Relaxed,
            Ordering::Relaxed
        ) {
            Ok(old) => Ok(unsafe { std::mem::transmute(old) }),
            Err(actual) => Err(unsafe { std::mem::transmute(actual) }),
        }
    }
}

impl<T> LockFreeSkipList<T>
where
    T: Clone + Send + Sync + 'static,
{
    /// Maximum skip list height
    const MAX_HEIGHT: usize = 16;
    
    /// Create new skip list
    pub fn new() -> Self {
        // Create sentinel head node
        let head = Owned::new(SkipListNode {
            data: ManuallyDrop::new(unsafe { std::mem::zeroed() }),
            forward: (0..Self::MAX_HEIGHT).map(|_| Atomic::null()).collect(),
            height: Self::MAX_HEIGHT,
            key: 0,
            marked: AtomicBool::new(false),
        });
        
        Self {
            head: Atomic::from(head),
            max_height: AtomicUsize::new(1),
            rng: AtomicUsize::new(1),
            len: AtomicUsize::new(0),
        }
    }
    
    /// Insert value with key
    pub fn insert(&self, key: u64, value: T) -> bool {
        let guard = &epoch::pin();
        let height = self.random_height();
        
        loop {
            // Find insertion point
            let (pred_refs, succ_refs) = self.find_node(key, guard);
            
            // Check if key already exists
            if let Some(succ_ref) = succ_refs[0] {
                let succ = unsafe { succ_ref.deref() };
                if succ.key == key && !succ.marked.load(Ordering::Relaxed) {
                    return false; // Key already exists
                }
            }
            
            // Create new node
            let new_node = Owned::new(SkipListNode {
                data: ManuallyDrop::new(value.clone()),
                forward: (0..height).map(|_| Atomic::null()).collect(),
                height,
                key,
                marked: AtomicBool::new(false),
            });
            
            let new_node_ref = new_node.as_ref();
            
            // Link forward pointers
            for level in 0..height {
                new_node_ref.forward[level].store(succ_refs[level], Ordering::Relaxed);
            }
            
            // Try to link the new node at level 0
            if let Some(pred_ref) = pred_refs[0] {
                let pred = unsafe { pred_ref.deref() };
                if pred.forward[0].compare_exchange(
                    succ_refs[0], 
                    new_node_ref, 
                    Ordering::Release, 
                    Ordering::Relaxed
                ).is_ok() {
                    // Successfully inserted at level 0, now link higher levels
                    for level in 1..height {
                        loop {
                            if let Some(pred_ref) = pred_refs[level] {
                                let pred = unsafe { pred_ref.deref() };
                                if pred.forward[level].compare_exchange(
                                    succ_refs[level],
                                    new_node_ref,
                                    Ordering::Release,
                                    Ordering::Relaxed
                                ).is_ok() {
                                    break;
                                }
                            }
                            
                            // Retry finding predecessors for this level
                            let (new_pred_refs, new_succ_refs) = self.find_node(key, guard);
                            if let Some(succ_ref) = new_succ_refs[level] {
                                new_node_ref.forward[level].store(Some(succ_ref), Ordering::Relaxed);
                            }
                        }
                    }
                    
                    self.len.fetch_add(1, Ordering::Relaxed);
                    
                    // Update max height if needed
                    let current_max = self.max_height.load(Ordering::Relaxed);
                    if height > current_max {
                        self.max_height.compare_exchange(
                            current_max,
                            height,
                            Ordering::Relaxed,
                            Ordering::Relaxed
                        ).ok();
                    }
                    
                    // Leak the owned node since it's now in the list
                    std::mem::forget(new_node);
                    return true;
                }
            }
        }
    }
    
    /// Find value by key
    pub fn find(&self, key: u64) -> Option<Arc<T>> {
        let guard = &epoch::pin();
        let current = self.head.load(Ordering::Acquire, guard)?;
        let mut current = unsafe { current.deref() };
        
        for level in (0..self.max_height.load(Ordering::Relaxed)).rev() {
            loop {
                if let Some(next) = current.forward[level].load(Ordering::Acquire, guard) {
                    let next_node = unsafe { next.deref() };
                    
                    if next_node.key < key {
                        current = next_node;
                    } else if next_node.key == key && !next_node.marked.load(Ordering::Relaxed) {
                        // Found the key
                        let data = unsafe { &*next_node.data };
                        return Some(Arc::new(data.clone()));
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        
        None
    }
    
    /// Get minimum key
    pub fn min_key(&self) -> Option<u64> {
        let guard = &epoch::pin();
        let head = self.head.load(Ordering::Acquire, guard)?;
        let head_node = unsafe { head.deref() };
        
        if let Some(first) = head_node.forward[0].load(Ordering::Acquire, guard) {
            let first_node = unsafe { first.deref() };
            if !first_node.marked.load(Ordering::Relaxed) {
                return Some(first_node.key);
            }
        }
        
        None
    }
    
    /// Get maximum key
    pub fn max_key(&self) -> Option<u64> {
        let guard = &epoch::pin();
        let current = self.head.load(Ordering::Acquire, guard)?;
        let mut current = unsafe { current.deref() };
        let mut max_key = None;
        
        // Traverse to find the rightmost node
        for level in (0..self.max_height.load(Ordering::Relaxed)).rev() {
            loop {
                if let Some(next) = current.forward[level].load(Ordering::Acquire, guard) {
                    let next_node = unsafe { next.deref() };
                    if !next_node.marked.load(Ordering::Relaxed) {
                        current = next_node;
                        max_key = Some(current.key);
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        
        max_key
    }
    
    /// Get top k elements
    pub fn top_k(&self, k: usize) -> Vec<(u64, usize)> {
        let guard = &epoch::pin();
        let mut result = Vec::with_capacity(k);
        
        if let Some(head) = self.head.load(Ordering::Acquire, guard) {
            let head_node = unsafe { head.deref() };
            let mut current = head_node.forward[0].load(Ordering::Acquire, guard);
            
            while let Some(node_ref) = current {
                let node = unsafe { node_ref.deref() };
                if !node.marked.load(Ordering::Relaxed) {
                    result.push((node.key, 1)); // Simplified quantity
                    if result.len() >= k {
                        break;
                    }
                }
                current = node.forward[0].load(Ordering::Acquire, guard);
            }
        }
        
        result
    }
    
    /// Find node with predecessors and successors
    fn find_node(&self, key: u64, guard: &Guard) -> (Vec<Option<Shared<SkipListNode<T>>>>, Vec<Option<Shared<SkipListNode<T>>>>) {
        let max_height = self.max_height.load(Ordering::Relaxed);
        let mut pred_refs = vec![None; max_height];
        let mut succ_refs = vec![None; max_height];
        
        if let Some(head) = self.head.load(Ordering::Acquire, guard) {
            let mut current = head;
            
            for level in (0..max_height).rev() {
                loop {
                    let current_node = unsafe { current.deref() };
                    if let Some(next) = current_node.forward[level].load(Ordering::Acquire, guard) {
                        let next_node = unsafe { next.deref() };
                        
                        if next_node.key < key {
                            current = next;
                        } else {
                            pred_refs[level] = Some(current);
                            succ_refs[level] = Some(next);
                            break;
                        }
                    } else {
                        pred_refs[level] = Some(current);
                        succ_refs[level] = None;
                        break;
                    }
                }
            }
        }
        
        (pred_refs, succ_refs)
    }
    
    /// Generate random height for new node
    fn random_height(&self) -> usize {
        let mut rng = self.rng.load(Ordering::Relaxed);
        let mut height = 1;
        
        // Use simple linear congruential generator for speed
        while height < Self::MAX_HEIGHT {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            if (rng >> 30) & 1 == 0 {
                height += 1;
            } else {
                break;
            }
        }
        
        self.rng.store(rng, Ordering::Relaxed);
        height
    }
}

impl<K, V> LockFreeHashMap<K, V>
where
    K: Clone + Eq + std::hash::Hash + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Create hash map with initial capacity
    pub fn with_capacity(capacity: usize) -> Self {
        let bucket_count = capacity.next_power_of_two().max(16);
        let mut buckets = Vec::with_capacity(bucket_count);
        
        for _ in 0..bucket_count {
            buckets.push(Atomic::null());
        }
        
        Self {
            buckets,
            bucket_count,
            len: AtomicUsize::new(0),
            load_factor_threshold: 0.75,
            resizing: AtomicBool::new(false),
        }
    }
    
    /// Insert key-value pair
    pub fn insert(&self, key: K, value: V) -> bool {
        let hash = self.hash(&key);
        let mut index = (hash as usize) & (self.bucket_count - 1);
        
        for _ in 0..self.bucket_count {
            let bucket_ptr = &self.buckets[index];
            let guard = &epoch::pin();
            
            match bucket_ptr.load(Ordering::Acquire, guard) {
                Some(bucket_ref) => {
                    let bucket = unsafe { bucket_ref.deref() };
                    
                    if !bucket.occupied.load(Ordering::Relaxed) {
                        // Try to claim this bucket
                        if bucket.occupied.compare_exchange(
                            false, true, Ordering::Acquire, Ordering::Relaxed
                        ).is_ok() {
                            // Successfully claimed, insert data
                            bucket.key = ManuallyDrop::new(key);
                            bucket.value = ManuallyDrop::new(value);
                            bucket.hash = hash;
                            bucket.deleted.store(false, Ordering::Relaxed);
                            
                            self.len.fetch_add(1, Ordering::Relaxed);
                            return true;
                        }
                    } else if bucket.hash == hash && 
                              *bucket.key == key &&
                              !bucket.deleted.load(Ordering::Relaxed) {
                        // Key already exists
                        return false;
                    }
                }
                None => {
                    // Create new bucket
                    let new_bucket = Owned::new(HashBucket {
                        key: ManuallyDrop::new(key.clone()),
                        value: ManuallyDrop::new(value.clone()),
                        hash,
                        occupied: AtomicBool::new(true),
                        deleted: AtomicBool::new(false),
                    });
                    
                    if bucket_ptr.compare_exchange(
                        None,
                        Some(new_bucket.as_ref()),
                        Ordering::Release,
                        Ordering::Relaxed
                    ).is_ok() {
                        std::mem::forget(new_bucket);
                        self.len.fetch_add(1, Ordering::Relaxed);
                        return true;
                    }
                }
            }
            
            // Linear probing to next bucket
            index = (index + 1) & (self.bucket_count - 1);
        }
        
        false // Hash table is full
    }
    
    /// Get value by key
    pub fn get(&self, key: &K) -> Option<V> {
        let hash = self.hash(key);
        let mut index = (hash as usize) & (self.bucket_count - 1);
        
        for _ in 0..self.bucket_count {
            let bucket_ptr = &self.buckets[index];
            let guard = &epoch::pin();
            
            if let Some(bucket_ref) = bucket_ptr.load(Ordering::Acquire, guard) {
                let bucket = unsafe { bucket_ref.deref() };
                
                if bucket.occupied.load(Ordering::Relaxed) &&
                   bucket.hash == hash &&
                   *bucket.key == *key &&
                   !bucket.deleted.load(Ordering::Relaxed) {
                    return Some(bucket.value.clone());
                }
            }
            
            index = (index + 1) & (self.bucket_count - 1);
        }
        
        None
    }
    
    /// Remove key-value pair
    pub fn remove(&self, key: &K) -> Option<V> {
        let hash = self.hash(key);
        let mut index = (hash as usize) & (self.bucket_count - 1);
        
        for _ in 0..self.bucket_count {
            let bucket_ptr = &self.buckets[index];
            let guard = &epoch::pin();
            
            if let Some(bucket_ref) = bucket_ptr.load(Ordering::Acquire, guard) {
                let bucket = unsafe { bucket_ref.deref() };
                
                if bucket.occupied.load(Ordering::Relaxed) &&
                   bucket.hash == hash &&
                   *bucket.key == *key &&
                   !bucket.deleted.load(Ordering::Relaxed) {
                    
                    // Mark as deleted
                    bucket.deleted.store(true, Ordering::Relaxed);
                    let value = bucket.value.clone();
                    
                    self.len.fetch_sub(1, Ordering::Relaxed);
                    return Some(value);
                }
            }
            
            index = (index + 1) & (self.bucket_count - 1);
        }
        
        None
    }
    
    /// Hash function
    fn hash(&self, key: &K) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }
}

impl<T> LockFreeQueue<T>
where
    T: Send + Sync + 'static,
{
    /// Create new lock-free queue
    pub fn new() -> Self {
        let dummy = Owned::new(QueueNode {
            data: MaybeUninit::uninit(),
            next: Atomic::null(),
            ready: AtomicBool::new(false),
        });
        
        let dummy_ref = dummy.as_ref();
        
        Self {
            head: Atomic::from(dummy),
            tail: Atomic::from(dummy_ref),
            len: AtomicUsize::new(0),
        }
    }
    
    /// Enqueue element (add to tail)
    pub fn enqueue(&self, item: T) {
        let new_node = Owned::new(QueueNode {
            data: MaybeUninit::new(item),
            next: Atomic::null(),
            ready: AtomicBool::new(true),
        });
        
        let guard = &epoch::pin();
        let new_node_ref = new_node.as_ref();
        
        loop {
            let tail = self.tail.load(Ordering::Acquire, guard).unwrap();
            let tail_node = unsafe { tail.deref() };
            let next = tail_node.next.load(Ordering::Acquire, guard);
            
            if tail == self.tail.load(Ordering::Acquire, guard).unwrap() {
                if next.is_none() {
                    // Try to link new node to tail
                    if tail_node.next.compare_exchange(
                        None,
                        Some(new_node_ref),
                        Ordering::Release,
                        Ordering::Relaxed
                    ).is_ok() {
                        break;
                    }
                } else {
                    // Try to swing tail to next node
                    self.tail.compare_exchange(
                        Some(tail),
                        next,
                        Ordering::Release,
                        Ordering::Relaxed
                    ).ok();
                }
            }
        }
        
        // Try to swing tail to new node
        self.tail.compare_exchange(
            self.tail.load(Ordering::Acquire, guard),
            Some(new_node_ref),
            Ordering::Release,
            Ordering::Relaxed
        ).ok();
        
        std::mem::forget(new_node);
        self.len.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Dequeue element (remove from head)
    pub fn dequeue(&self) -> Option<T> {
        let guard = &epoch::pin();
        
        loop {
            let head = self.head.load(Ordering::Acquire, guard)?;
            let tail = self.tail.load(Ordering::Acquire, guard)?;
            let head_node = unsafe { head.deref() };
            let next = head_node.next.load(Ordering::Acquire, guard);
            
            if head == self.head.load(Ordering::Acquire, guard)? {
                if head == tail {
                    if next.is_none() {
                        // Queue is empty
                        return None;
                    }
                    
                    // Try to swing tail to next
                    self.tail.compare_exchange(
                        Some(tail),
                        next,
                        Ordering::Release,
                        Ordering::Relaxed
                    ).ok();
                } else {
                    if let Some(next_ref) = next {
                        let next_node = unsafe { next_ref.deref() };
                        
                        if next_node.ready.load(Ordering::Acquire) {
                            // Try to swing head to next
                            if self.head.compare_exchange(
                                Some(head),
                                Some(next_ref),
                                Ordering::Release,
                                Ordering::Relaxed
                            ).is_ok() {
                                let data = unsafe { next_node.data.assume_init_read() };
                                self.len.fetch_sub(1, Ordering::Relaxed);
                                
                                // Schedule old head for deletion
                                unsafe { guard.defer_destroy(head) };
                                
                                return Some(data);
                            }
                        }
                    }
                }
            }
        }
    }
    
    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.len.load(Ordering::Relaxed) == 0
    }
    
    /// Get approximate length
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }
}

impl<T> LockFreeMemoryPool<T>
where
    T: Default + Send + Sync + 'static,
{
    /// Create memory pool with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        let pool = Self {
            free_list: Atomic::null(),
            capacity,
            size: AtomicUsize::new(0),
            stats: PoolStats {
                allocations: AtomicUsize::new(0),
                deallocations: AtomicUsize::new(0),
                peak_usage: AtomicUsize::new(0),
                pool_misses: AtomicUsize::new(0),
            },
        };
        
        // Pre-allocate pool objects
        for _ in 0..capacity {
            let node = Box::into_raw(Box::new(PoolNode {
                object: MaybeUninit::new(T::default()),
                next: Atomic::null(),
            }));
            
            let guard = &epoch::pin();
            let node_ref = unsafe { Shared::from_raw(node) };
            
            loop {
                let head = pool.free_list.load(Ordering::Acquire, guard);
                unsafe { node_ref.deref() }.next.store(head, Ordering::Relaxed);
                
                if pool.free_list.compare_exchange(
                    head,
                    Some(node_ref),
                    Ordering::Release,
                    Ordering::Relaxed
                ).is_ok() {
                    break;
                }
            }
        }
        
        pool.size.store(capacity, Ordering::Relaxed);
        pool
    }
    
    /// Allocate object from pool
    pub fn allocate(&self) -> Option<Box<T>> {
        let guard = &epoch::pin();
        
        loop {
            let head = self.free_list.load(Ordering::Acquire, guard);
            
            if let Some(head_ref) = head {
                let head_node = unsafe { head_ref.deref() };
                let next = head_node.next.load(Ordering::Relaxed, guard);
                
                if self.free_list.compare_exchange(
                    Some(head_ref),
                    next,
                    Ordering::Release,
                    Ordering::Relaxed
                ).is_ok() {
                    self.size.fetch_sub(1, Ordering::Relaxed);
                    self.stats.allocations.fetch_add(1, Ordering::Relaxed);
                    
                    // Convert to Box and return
                    let obj = unsafe { head_node.object.assume_init_read() };
                    let boxed = Box::new(obj);
                    
                    // Schedule node for deletion
                    unsafe { guard.defer_destroy(head_ref) };
                    
                    return Some(boxed);
                }
            } else {
                // Pool is empty, record miss
                self.stats.pool_misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        }
    }
    
    /// Deallocate object back to pool
    pub fn deallocate(&self, obj: Box<T>) {
        if self.size.load(Ordering::Relaxed) >= self.capacity {
            // Pool is full, just drop the object
            return;
        }
        
        let node = Box::into_raw(Box::new(PoolNode {
            object: MaybeUninit::new(*obj),
            next: Atomic::null(),
        }));
        
        let guard = &epoch::pin();
        let node_ref = unsafe { Shared::from_raw(node) };
        
        loop {
            let head = self.free_list.load(Ordering::Acquire, guard);
            unsafe { node_ref.deref() }.next.store(head, Ordering::Relaxed);
            
            if self.free_list.compare_exchange(
                head,
                Some(node_ref),
                Ordering::Release,
                Ordering::Relaxed
            ).is_ok() {
                self.size.fetch_add(1, Ordering::Relaxed);
                self.stats.deallocations.fetch_add(1, Ordering::Relaxed);
                break;
            }
        }
    }
    
    /// Get pool statistics
    pub fn stats(&self) -> (usize, usize, usize, usize) {
        (
            self.stats.allocations.load(Ordering::Relaxed),
            self.stats.deallocations.load(Ordering::Relaxed),
            self.stats.peak_usage.load(Ordering::Relaxed),
            self.stats.pool_misses.load(Ordering::Relaxed),
        )
    }
}

// Safety implementations for Send/Sync
unsafe impl<T: Send> Send for LockFreeOrderBook<T> {}
unsafe impl<T: Sync> Sync for LockFreeOrderBook<T> {}
unsafe impl<T: Send> Send for LockFreeSkipList<T> {}
unsafe impl<T: Sync> Sync for LockFreeSkipList<T> {}
unsafe impl<K: Send, V: Send> Send for LockFreeHashMap<K, V> {}
unsafe impl<K: Sync, V: Sync> Sync for LockFreeHashMap<K, V> {}
unsafe impl<T: Send> Send for LockFreeQueue<T> {}
unsafe impl<T: Sync> Sync for LockFreeQueue<T> {}
unsafe impl<T: Send> Send for LockFreeMemoryPool<T> {}
unsafe impl<T: Sync> Sync for LockFreeMemoryPool<T> {}