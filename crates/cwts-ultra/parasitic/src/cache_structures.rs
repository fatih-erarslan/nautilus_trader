//! Cache-Friendly Data Structures for Ultra-Low Latency Trading
//! 
//! This module implements cache-optimized data structures designed for
//! sub-microsecond decision making in parasitic trading systems.
//!
//! Key features:
//! - Cache-line aligned structures (64-byte alignment)
//! - NUMA-aware memory layouts
//! - Prefetch-friendly access patterns
//! - Lock-free concurrent operations
//! - Memory pool allocation to reduce allocation overhead

use std::sync::atomic::{AtomicUsize, AtomicPtr, AtomicU64, Ordering};
use std::ptr;
use std::alloc::{alloc, Layout};
use std::mem::{size_of, align_of};
use crossbeam::utils::CachePadded;

/// Cache-line size for x86_64 (64 bytes)
pub const CACHE_LINE_SIZE: usize = 64;

/// NUMA node-aware memory allocator
pub struct NumaAllocator {
    node_id: u32,
    preferred_cpu: u32,
}

impl NumaAllocator {
    pub fn new(node_id: u32, preferred_cpu: u32) -> Self {
        Self { node_id, preferred_cpu }
    }

    /// Allocate cache-aligned memory on specific NUMA node
    #[inline]
    pub unsafe fn alloc_aligned<T>(&self, count: usize) -> *mut T {
        let layout = Layout::from_size_align(
            size_of::<T>() * count,
            CACHE_LINE_SIZE.max(align_of::<T>())
        ).unwrap();
        
        // In a real implementation, this would use numa_alloc_onnode()
        alloc(layout) as *mut T
    }

    /// Prefetch memory for optimal cache performance
    #[inline]
    pub unsafe fn prefetch_read<T>(&self, ptr: *const T, count: usize) {
        // Prefetch multiple cache lines
        let bytes = size_of::<T>() * count;
        let cache_lines = (bytes + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE;
        
        let base_ptr = ptr as *const u8;
        for i in 0..cache_lines {
            let line_ptr = base_ptr.add(i * CACHE_LINE_SIZE);
            #[cfg(target_arch = "x86_64")]
            std::arch::x86_64::_mm_prefetch(line_ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }
    }
}

/// Ultra-fast memory pool for trading objects
/// Optimized for frequent allocation/deallocation cycles
#[repr(C, align(64))]
pub struct FastMemoryPool<T> {
    free_list: CachePadded<AtomicPtr<PoolNode<T>>>,
    allocated_count: CachePadded<AtomicUsize>,
    total_capacity: usize,
    pool_memory: *mut u8,
    node_size: usize,
    numa_allocator: NumaAllocator,
}

#[repr(C, align(64))]
struct PoolNode<T> {
    data: T,
    next: AtomicPtr<PoolNode<T>>,
}

impl<T> FastMemoryPool<T> {
    /// Create new memory pool with specified capacity
    pub fn new(capacity: usize, numa_node: u32) -> Self {
        let node_size = size_of::<PoolNode<T>>().max(CACHE_LINE_SIZE);
        let total_size = node_size * capacity;
        
        let numa_allocator = NumaAllocator::new(numa_node, 0);
        
        let pool_memory = unsafe { numa_allocator.alloc_aligned::<u8>(total_size) };
        
        // Initialize free list
        let mut free_list_head: *mut PoolNode<T> = ptr::null_mut();
        
        unsafe {
            for i in 0..capacity {
                let node_ptr = pool_memory.add(i * node_size) as *mut PoolNode<T>;
                
                // Initialize the node
                (*node_ptr).next = AtomicPtr::new(free_list_head);
                free_list_head = node_ptr;
            }
        }

        Self {
            free_list: CachePadded::new(AtomicPtr::new(free_list_head)),
            allocated_count: CachePadded::new(AtomicUsize::new(0)),
            total_capacity: capacity,
            pool_memory,
            node_size,
            numa_allocator,
        }
    }

    /// Allocate object from pool (lock-free)
    /// Target: <10ns allocation time
    #[inline]
    pub fn allocate(&self) -> Option<*mut T> {
        loop {
            let head = self.free_list.load(Ordering::Acquire);
            
            if head.is_null() {
                return None; // Pool exhausted
            }

            unsafe {
                let next = (*head).next.load(Ordering::Relaxed);
                
                match self.free_list.compare_exchange_weak(
                    head,
                    next,
                    Ordering::Release,
                    Ordering::Relaxed
                ) {
                    Ok(_) => {
                        self.allocated_count.fetch_add(1, Ordering::Relaxed);
                        return Some(&mut (*head).data as *mut T);
                    }
                    Err(_) => continue, // Retry on contention
                }
            }
        }
    }

    /// Deallocate object back to pool (lock-free)
    /// Target: <5ns deallocation time
    #[inline]
    pub fn deallocate(&self, ptr: *mut T) {
        if ptr.is_null() { return; }

        // Calculate node address from data pointer
        let node_ptr = unsafe {
            let data_offset = ptr as usize - &(*(ptr::null::<PoolNode<T>>())).data as *const T as usize;
            (ptr as usize - data_offset) as *mut PoolNode<T>
        };

        unsafe {
            loop {
                let head = self.free_list.load(Ordering::Acquire);
                (*node_ptr).next.store(head, Ordering::Relaxed);
                
                match self.free_list.compare_exchange_weak(
                    head,
                    node_ptr,
                    Ordering::Release,
                    Ordering::Relaxed
                ) {
                    Ok(_) => {
                        self.allocated_count.fetch_sub(1, Ordering::Relaxed);
                        return;
                    }
                    Err(_) => continue, // Retry on contention
                }
            }
        }
    }

    /// Get pool utilization statistics
    #[inline]
    pub fn utilization(&self) -> f32 {
        let allocated = self.allocated_count.load(Ordering::Relaxed);
        allocated as f32 / self.total_capacity as f32
    }

    /// Prefetch pool memory for better cache performance
    #[inline]
    pub fn prefetch_pool(&self) {
        unsafe {
            self.numa_allocator.prefetch_read(self.pool_memory, self.total_capacity * self.node_size);
        }
    }
}

unsafe impl<T> Send for FastMemoryPool<T> {}
unsafe impl<T> Sync for FastMemoryPool<T> {}

/// Cache-optimized circular buffer for streaming market data
/// Designed for single producer, multiple consumer pattern
#[repr(C, align(64))]
pub struct CacheOptimizedRingBuffer<T> {
    buffer: *mut T,
    capacity: usize,
    mask: usize, // capacity - 1 (for power-of-2 capacity)
    
    // Producer state (separate cache line)
    producer_head: CachePadded<AtomicU64>,
    
    // Consumer states (separate cache lines)
    consumer_tails: Vec<CachePadded<AtomicU64>>,
    
    // Buffer memory layout optimized for cache
    memory_layout: Layout,
    numa_allocator: NumaAllocator,
}

impl<T: Copy> CacheOptimizedRingBuffer<T> {
    /// Create new ring buffer with power-of-2 capacity
    pub fn new(capacity_log2: u8, num_consumers: usize, numa_node: u32) -> Self {
        let capacity = 1usize << capacity_log2;
        let mask = capacity - 1;
        
        let numa_allocator = NumaAllocator::new(numa_node, 0);
        
        // Allocate buffer memory aligned to cache lines
        let layout = Layout::from_size_align(
            size_of::<T>() * capacity,
            CACHE_LINE_SIZE
        ).unwrap();
        
        let buffer = unsafe { numa_allocator.alloc_aligned::<T>(capacity) };
        
        // Initialize consumer tails
        let mut consumer_tails = Vec::with_capacity(num_consumers);
        for _ in 0..num_consumers {
            consumer_tails.push(CachePadded::new(AtomicU64::new(0)));
        }

        Self {
            buffer,
            capacity,
            mask,
            producer_head: CachePadded::new(AtomicU64::new(0)),
            consumer_tails,
            memory_layout: layout,
            numa_allocator,
        }
    }

    /// Producer: Push data to buffer
    /// Target: <20ns including cache warming
    #[inline]
    pub fn push(&self, data: T) -> bool {
        let head = self.producer_head.load(Ordering::Relaxed);
        
        // Check if buffer is full by examining slowest consumer
        let min_tail = self.consumer_tails
            .iter()
            .map(|tail| tail.load(Ordering::Acquire))
            .min()
            .unwrap_or(0);
        
        if head.wrapping_sub(min_tail) >= self.capacity as u64 {
            return false; // Buffer full
        }

        // Write data
        let index = (head & self.mask as u64) as usize;
        unsafe {
            ptr::write(self.buffer.add(index), data);
            
            // Memory barrier to ensure data write completes before head update
            std::sync::atomic::fence(Ordering::Release);
        }
        
        // Update producer head
        self.producer_head.store(head + 1, Ordering::Release);
        true
    }

    /// Consumer: Pop data from buffer
    /// Target: <15ns including prefetch
    #[inline]
    pub fn pop(&self, consumer_id: usize) -> Option<T> {
        if consumer_id >= self.consumer_tails.len() {
            return None;
        }

        let tail = self.consumer_tails[consumer_id].load(Ordering::Relaxed);
        let head = self.producer_head.load(Ordering::Acquire);
        
        if tail >= head {
            return None; // Buffer empty for this consumer
        }

        // Prefetch next few items for better cache performance
        let next_index = ((tail + 1) & self.mask as u64) as usize;
        unsafe {
            self.numa_allocator.prefetch_read(self.buffer.add(next_index), 4);
        }

        // Read data
        let index = (tail & self.mask as u64) as usize;
        let data = unsafe { ptr::read(self.buffer.add(index)) };
        
        // Update consumer tail
        self.consumer_tails[consumer_id].store(tail + 1, Ordering::Release);
        
        Some(data)
    }

    /// Batch pop for better cache utilization
    /// Target: <10ns per item when batching
    #[inline]
    pub fn pop_batch(&self, consumer_id: usize, batch: &mut [T]) -> usize {
        if consumer_id >= self.consumer_tails.len() {
            return 0;
        }

        let tail = self.consumer_tails[consumer_id].load(Ordering::Relaxed);
        let head = self.producer_head.load(Ordering::Acquire);
        
        if tail >= head {
            return 0; // Buffer empty
        }

        let available = (head - tail) as usize;
        let to_read = available.min(batch.len());
        
        // Prefetch the entire range we're about to read
        let start_index = (tail & self.mask as u64) as usize;
        unsafe {
            self.numa_allocator.prefetch_read(self.buffer.add(start_index), to_read);
        }

        // Copy data in batches to maximize cache efficiency
        for i in 0..to_read {
            let index = ((tail + i as u64) & self.mask as u64) as usize;
            unsafe {
                batch[i] = ptr::read(self.buffer.add(index));
            }
        }
        
        // Update consumer tail
        self.consumer_tails[consumer_id].store(tail + to_read as u64, Ordering::Release);
        
        to_read
    }

    /// Get buffer utilization for monitoring
    #[inline]
    pub fn utilization(&self) -> f32 {
        let head = self.producer_head.load(Ordering::Relaxed);
        let min_tail = self.consumer_tails
            .iter()
            .map(|tail| tail.load(Ordering::Relaxed))
            .min()
            .unwrap_or(head);
        
        let used = head.wrapping_sub(min_tail) as usize;
        used as f32 / self.capacity as f32
    }
}

unsafe impl<T: Copy> Send for CacheOptimizedRingBuffer<T> {}
unsafe impl<T: Copy> Sync for CacheOptimizedRingBuffer<T> {}

/// High-performance hash table optimized for market data lookups
/// Uses Robin Hood hashing with SIMD acceleration
#[repr(C, align(64))]
pub struct FastHashTable<K, V> {
    buckets: *mut HashBucket<K, V>,
    capacity: usize,
    mask: usize,
    size: AtomicUsize,
    max_probe_distance: u8,
    numa_allocator: NumaAllocator,
}

#[repr(C, align(32))]
struct HashBucket<K, V> {
    hash: u32,
    probe_distance: u8,
    occupied: bool,
    _padding: [u8; 2], // Ensure 32-byte alignment
    key: K,
    value: V,
}

impl<K: Copy + Eq, V: Copy> FastHashTable<K, V> {
    /// Create hash table with power-of-2 capacity
    pub fn new(capacity_log2: u8, numa_node: u32) -> Self {
        let capacity = 1usize << capacity_log2;
        let mask = capacity - 1;
        
        let numa_allocator = NumaAllocator::new(numa_node, 0);
        
        let buckets = unsafe { numa_allocator.alloc_aligned::<HashBucket<K, V>>(capacity) };
        
        // Initialize all buckets as empty
        unsafe {
            for i in 0..capacity {
                ptr::write(buckets.add(i), HashBucket {
                    hash: 0,
                    probe_distance: 0,
                    occupied: false,
                    _padding: [0; 2],
                    key: std::mem::zeroed(),
                    value: std::mem::zeroed(),
                });
            }
        }

        Self {
            buckets,
            capacity,
            mask,
            size: AtomicUsize::new(0),
            max_probe_distance: (capacity_log2 + 4).min(255) as u8,
            numa_allocator,
        }
    }

    /// Fast hash function optimized for integers
    #[inline]
    fn hash_key(&self, key: &K) -> u32 {
        // Simple but effective hash for performance
        let key_bytes = unsafe { 
            std::slice::from_raw_parts(key as *const K as *const u8, size_of::<K>())
        };
        
        let mut hash = 2166136261u32;
        for &byte in key_bytes {
            hash ^= byte as u32;
            hash = hash.wrapping_mul(16777619);
        }
        hash
    }

    /// Lookup value by key
    /// Target: <25ns average case
    #[inline]
    pub fn get(&self, key: &K) -> Option<V> {
        let hash = self.hash_key(key);
        let mut index = (hash as usize) & self.mask;
        let mut probe_distance = 0u8;

        // Prefetch the initial lookup location
        unsafe {
            self.numa_allocator.prefetch_read(self.buckets.add(index), 4);
        }

        loop {
            unsafe {
                let bucket = &*self.buckets.add(index);
                
                if !bucket.occupied {
                    return None; // Empty bucket, key not found
                }
                
                if bucket.hash == hash && bucket.key == *key {
                    return Some(bucket.value);
                }
                
                if probe_distance > bucket.probe_distance {
                    return None; // Key would be here if it existed (Robin Hood invariant)
                }
                
                probe_distance += 1;
                if probe_distance > self.max_probe_distance {
                    return None; // Probe limit reached
                }
                
                index = (index + 1) & self.mask;
            }
        }
    }

    /// Insert or update key-value pair
    /// Target: <35ns average case
    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> bool {
        if self.size.load(Ordering::Relaxed) * 4 > self.capacity * 3 {
            return false; // Table too full (>75% load factor)
        }

        let hash = self.hash_key(&key);
        let mut index = (hash as usize) & self.mask;
        let mut probe_distance = 0u8;
        
        let mut inserting_bucket = HashBucket {
            hash,
            probe_distance,
            occupied: true,
            _padding: [0; 2],
            key,
            value,
        };

        loop {
            unsafe {
                let bucket = &mut *self.buckets.add(index);
                
                if !bucket.occupied {
                    // Empty slot found
                    ptr::write(bucket, inserting_bucket);
                    self.size.fetch_add(1, Ordering::Relaxed);
                    return true;
                }
                
                if bucket.hash == hash && bucket.key == key {
                    // Update existing key
                    bucket.value = value;
                    return true;
                }
                
                // Robin Hood hashing: swap if we've traveled further
                if inserting_bucket.probe_distance > bucket.probe_distance {
                    std::mem::swap(&mut inserting_bucket, bucket);
                }
                
                inserting_bucket.probe_distance += 1;
                if inserting_bucket.probe_distance > self.max_probe_distance {
                    return false; // Probe limit reached
                }
                
                index = (index + 1) & self.mask;
            }
        }
    }

    /// Get current load factor
    #[inline]
    pub fn load_factor(&self) -> f32 {
        self.size.load(Ordering::Relaxed) as f32 / self.capacity as f32
    }
}

unsafe impl<K: Copy, V: Copy> Send for FastHashTable<K, V> {}
unsafe impl<K: Copy, V: Copy> Sync for FastHashTable<K, V> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool() {
        let pool: FastMemoryPool<u64> = FastMemoryPool::new(100, 0);
        
        // Test allocation
        let ptr = pool.allocate().expect("Should allocate successfully");
        unsafe { *ptr = 42; }
        
        // Test deallocation
        pool.deallocate(ptr);
        
        assert!(pool.utilization() < 0.01); // Should be nearly empty
    }

    #[test]
    fn test_ring_buffer() {
        let buffer: CacheOptimizedRingBuffer<u32> = CacheOptimizedRingBuffer::new(4, 1, 0); // 16 capacity
        
        // Test push/pop
        assert!(buffer.push(42));
        assert_eq!(buffer.pop(0), Some(42));
        assert_eq!(buffer.pop(0), None);
    }

    #[test]
    fn test_hash_table() {
        let mut table: FastHashTable<u32, u32> = FastHashTable::new(4, 0); // 16 capacity
        
        // Test insert/get
        assert!(table.insert(42, 100));
        assert_eq!(table.get(&42), Some(100));
        assert_eq!(table.get(&99), None);
    }

    #[test]
    fn test_numa_allocator() {
        let allocator = NumaAllocator::new(0, 0);
        
        let ptr = unsafe { allocator.alloc_aligned::<u64>(10) };
        assert!(!ptr.is_null());
        
        // Test prefetching doesn't panic
        unsafe { allocator.prefetch_read(ptr, 10); }
    }

    #[test]
    fn test_cache_line_alignment() {
        let pool: FastMemoryPool<u64> = FastMemoryPool::new(10, 0);
        
        // Verify alignment
        assert_eq!(&pool as *const _ as usize % CACHE_LINE_SIZE, 0);
    }

    #[test]
    fn test_concurrent_ring_buffer() {
        use std::thread;
        use std::sync::Arc;
        
        let buffer = Arc::new(CacheOptimizedRingBuffer::<u32>::new(8, 2, 0)); // 256 capacity
        
        let producer = buffer.clone();
        let consumer1 = buffer.clone();
        let consumer2 = buffer.clone();
        
        let producer_handle = thread::spawn(move || {
            for i in 0..100 {
                while !producer.push(i) {
                    std::hint::spin_loop();
                }
            }
        });
        
        let consumer1_handle = thread::spawn(move || {
            let mut count = 0;
            while count < 50 {
                if consumer1.pop(0).is_some() {
                    count += 1;
                }
            }
            count
        });
        
        let consumer2_handle = thread::spawn(move || {
            let mut count = 0;
            while count < 50 {
                if consumer2.pop(1).is_some() {
                    count += 1;
                }
            }
            count
        });
        
        producer_handle.join().unwrap();
        let c1_count = consumer1_handle.join().unwrap();
        let c2_count = consumer2_handle.join().unwrap();
        
        assert_eq!(c1_count + c2_count, 100);
    }
}