//! Lock-free data structures for high-performance parallel processing
//!
//! This module provides wait-free and lock-free primitives optimized for 
//! CDFA's parallel processing requirements.

use atomic_float::AtomicF64;
use crossbeam::queue::{ArrayQueue, SegQueue};
use crossbeam::utils::CachePadded;
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::hint;
use std::mem::MaybeUninit;

/// Branch prediction hint for likely condition
#[inline(always)]
fn likely(b: bool) -> bool {
    #[cold]
    fn cold() {}
    
    if !b {
        cold();
    }
    b
}

/// Branch prediction hint for unlikely condition
#[inline(always)]
fn unlikely(b: bool) -> bool {
    #[cold]
    fn cold() {}
    
    if b {
        cold();
    }
    b
}

use cdfa_core::types::{AnalysisResult, DiversityMatrix, Signal};

/// Lock-free signal buffer for streaming data
///
/// Designed for single-producer, multiple-consumer scenarios
/// with minimal contention and cache-coherent access patterns.
pub struct LockFreeSignalBuffer {
    /// Ring buffer for signals
    buffer: Arc<ArrayQueue<Signal>>,
    
    /// Current write position
    write_pos: CachePadded<AtomicU64>,
    
    /// Number of signals processed
    processed_count: CachePadded<AtomicU64>,
    
    /// Buffer capacity
    capacity: usize,
    
    /// Overflow flag
    overflow: AtomicBool,
}

impl LockFreeSignalBuffer {
    /// Creates a new lock-free signal buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Arc::new(ArrayQueue::new(capacity)),
            write_pos: CachePadded::new(AtomicU64::new(0)),
            processed_count: CachePadded::new(AtomicU64::new(0)),
            capacity,
            overflow: AtomicBool::new(false),
        }
    }
    
    /// Pushes a signal into the buffer (wait-free)
    #[inline(always)]
    pub fn push(&self, signal: Signal) -> Result<(), Signal> {
        // Prefetch buffer metadata for faster access
        unsafe {
            hint::black_box(&self.buffer);
        }
        
        match self.buffer.push(signal) {
            Ok(()) => {
                // Use relaxed ordering for write position - order doesn't matter for perf counters
                self.write_pos.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(signal) => {
                self.overflow.store(true, Ordering::Relaxed);
                Err(signal)
            }
        }
    }
    
    /// Tries to pop a signal from the buffer (wait-free)
    #[inline(always)]
    pub fn try_pop(&self) -> Option<Signal> {
        // Branch prediction hint: most calls succeed
        let signal = self.buffer.pop();
        if likely(signal.is_some()) {
            self.processed_count.fetch_add(1, Ordering::Relaxed);
        }
        signal
    }
    
    /// Returns the number of signals currently in the buffer
    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    
    /// Returns true if the buffer is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
    
    /// Returns true if overflow occurred
    #[inline]
    pub fn has_overflowed(&self) -> bool {
        self.overflow.load(Ordering::Relaxed)
    }
    
    /// Resets the overflow flag
    pub fn reset_overflow(&self) {
        self.overflow.store(false, Ordering::Relaxed);
    }
    
    /// Returns processing statistics
    pub fn stats(&self) -> BufferStats {
        BufferStats {
            total_written: self.write_pos.load(Ordering::Acquire),
            total_processed: self.processed_count.load(Ordering::Acquire),
            current_size: self.len(),
            capacity: self.capacity,
            has_overflowed: self.has_overflowed(),
        }
    }
}

/// Statistics for the lock-free buffer
#[derive(Debug, Clone)]
pub struct BufferStats {
    pub total_written: u64,
    pub total_processed: u64,
    pub current_size: usize,
    pub capacity: usize,
    pub has_overflowed: bool,
}

/// Wait-free correlation matrix for diversity calculations
///
/// Uses atomic operations for lock-free updates across threads
pub struct WaitFreeCorrelationMatrix {
    /// Matrix data stored as atomic floats
    data: Vec<CachePadded<AtomicF64>>,
    
    /// Matrix dimension
    dimension: usize,
    
    /// Update counter for each cell
    update_counts: Vec<AtomicU64>,
    
    /// Global version for consistency checks
    version: AtomicU64,
}

impl WaitFreeCorrelationMatrix {
    /// Creates a new wait-free correlation matrix
    pub fn new(dimension: usize) -> Self {
        let size = dimension * dimension;
        let mut data = Vec::with_capacity(size);
        let mut update_counts = Vec::with_capacity(size);
        
        for _ in 0..size {
            data.push(CachePadded::new(AtomicF64::new(0.0)));
            update_counts.push(AtomicU64::new(0));
        }
        
        Self {
            data,
            dimension,
            update_counts,
            version: AtomicU64::new(0),
        }
    }
    
    /// Updates a correlation value atomically
    #[inline(always)]
    pub fn update(&self, row: usize, col: usize, value: f64) {
        assert!(row < self.dimension && col < self.dimension);
        let idx = row * self.dimension + col;
        
        // Update value
        self.data[idx].store(value, Ordering::Release);
        
        // Increment update counter
        self.update_counts[idx].fetch_add(1, Ordering::Relaxed);
        
        // Update global version
        self.version.fetch_add(1, Ordering::Release);
    }
    
    /// Gets a correlation value atomically
    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        assert!(row < self.dimension && col < self.dimension);
        let idx = row * self.dimension + col;
        self.data[idx].load(Ordering::Acquire)
    }
    
    /// Performs an atomic compare-and-swap update
    #[inline]
    pub fn compare_and_swap(&self, row: usize, col: usize, current: f64, new: f64) -> f64 {
        assert!(row < self.dimension && col < self.dimension);
        let idx = row * self.dimension + col;
        self.data[idx].compare_and_swap(current, new, Ordering::AcqRel)
    }
    
    /// Gets the number of updates for a specific cell
    #[inline]
    pub fn update_count(&self, row: usize, col: usize) -> u64 {
        assert!(row < self.dimension && col < self.dimension);
        let idx = row * self.dimension + col;
        self.update_counts[idx].load(Ordering::Relaxed)
    }
    
    /// Gets the current version number
    #[inline]
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }
    
    /// Converts to a regular DiversityMatrix (snapshot)
    pub fn to_diversity_matrix(&self) -> DiversityMatrix {
        let mut matrix = DiversityMatrix::zeros(self.dimension);
        
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                matrix.set(i, j, self.get(i, j));
            }
        }
        
        matrix
    }
}

/// Lock-free result aggregator for parallel analysis
///
/// Efficiently collects results from multiple analysis threads
pub struct LockFreeResultAggregator {
    /// Results queue (unbounded for performance)
    results: SegQueue<AnalysisResult>,
    
    /// Result count
    count: AtomicUsize,
    
    /// Aggregation statistics
    total_latency_ns: AtomicU64,
    min_latency_ns: AtomicU64,
    max_latency_ns: AtomicU64,
    
    /// Confidence accumulator
    confidence_sum: AtomicF64,
}

impl LockFreeResultAggregator {
    /// Creates a new result aggregator
    pub fn new() -> Self {
        Self {
            results: SegQueue::new(),
            count: AtomicUsize::new(0),
            total_latency_ns: AtomicU64::new(0),
            min_latency_ns: AtomicU64::new(u64::MAX),
            max_latency_ns: AtomicU64::new(0),
            confidence_sum: AtomicF64::new(0.0),
        }
    }
    
    /// Adds a result to the aggregator (wait-free)
    pub fn add_result(&self, result: AnalysisResult) {
        // Update statistics atomically
        self.total_latency_ns.fetch_add(result.latency_ns, Ordering::Relaxed);
        self.confidence_sum.fetch_add(result.confidence, Ordering::Relaxed);
        
        // Update min/max latency with CAS loop
        loop {
            let current_min = self.min_latency_ns.load(Ordering::Relaxed);
            if result.latency_ns >= current_min || 
               self.min_latency_ns.compare_exchange_weak(
                   current_min,
                   result.latency_ns,
                   Ordering::Relaxed,
                   Ordering::Relaxed
               ).is_ok() {
                break;
            }
        }
        
        loop {
            let current_max = self.max_latency_ns.load(Ordering::Relaxed);
            if result.latency_ns <= current_max ||
               self.max_latency_ns.compare_exchange_weak(
                   current_max,
                   result.latency_ns,
                   Ordering::Relaxed,
                   Ordering::Relaxed
               ).is_ok() {
                break;
            }
        }
        
        // Add to queue and increment count
        self.results.push(result);
        self.count.fetch_add(1, Ordering::Release);
    }
    
    /// Collects all results (drains the aggregator)
    pub fn collect_results(&self) -> Vec<AnalysisResult> {
        let mut results = Vec::with_capacity(self.count.load(Ordering::Acquire));
        
        while let Some(result) = self.results.pop() {
            results.push(result);
        }
        
        // Reset count
        self.count.store(0, Ordering::Release);
        
        results
    }
    
    /// Gets aggregation statistics without draining
    pub fn stats(&self) -> AggregatorStats {
        let count = self.count.load(Ordering::Acquire);
        let total_latency = self.total_latency_ns.load(Ordering::Relaxed);
        
        AggregatorStats {
            count,
            avg_latency_ns: if count > 0 { total_latency / count as u64 } else { 0 },
            min_latency_ns: self.min_latency_ns.load(Ordering::Relaxed),
            max_latency_ns: self.max_latency_ns.load(Ordering::Relaxed),
            avg_confidence: if count > 0 { 
                self.confidence_sum.load(Ordering::Relaxed) / count as f64 
            } else { 
                0.0 
            },
        }
    }
}

/// Statistics for the result aggregator
#[derive(Debug, Clone)]
pub struct AggregatorStats {
    pub count: usize,
    pub avg_latency_ns: u64,
    pub min_latency_ns: u64,
    pub max_latency_ns: u64,
    pub avg_confidence: f64,
}

/// Lock-free cache for computed features
///
/// Uses DashMap for concurrent access with minimal contention
pub struct LockFreeFeatureCache {
    /// Feature cache with TTL support
    cache: DashMap<u64, CachedFeature>,
    
    /// Maximum cache size
    max_size: usize,
    
    /// Cache statistics
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

/// Cached feature with metadata
#[derive(Clone)]
struct CachedFeature {
    data: Arc<Vec<f64>>,
    timestamp_ns: u64,
    access_count: Arc<AtomicU64>,
}

impl LockFreeFeatureCache {
    /// Creates a new feature cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: DashMap::with_capacity(max_size),
            max_size,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }
    
    /// Gets a cached feature if available
    pub fn get(&self, key: u64) -> Option<Arc<Vec<f64>>> {
        if let Some(entry) = self.cache.get(&key) {
            entry.access_count.fetch_add(1, Ordering::Relaxed);
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(Arc::clone(&entry.data))
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }
    
    /// Inserts a feature into the cache
    pub fn insert(&self, key: u64, data: Vec<f64>, timestamp_ns: u64) {
        // Check if eviction is needed
        if self.cache.len() >= self.max_size {
            self.evict_lru();
        }
        
        let feature = CachedFeature {
            data: Arc::new(data),
            timestamp_ns,
            access_count: Arc::new(AtomicU64::new(0)),
        };
        
        self.cache.insert(key, feature);
    }
    
    /// Evicts least recently used entry
    fn evict_lru(&self) {
        if let Some((key, _)) = self.cache
            .iter()
            .min_by_key(|entry| entry.value().access_count.load(Ordering::Relaxed))
            .map(|entry| (*entry.key(), entry.value().clone()))
        {
            self.cache.remove(&key);
            self.evictions.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    /// Returns cache statistics
    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        
        CacheStats {
            size: self.cache.len(),
            capacity: self.max_size,
            hits,
            misses,
            evictions: self.evictions.load(Ordering::Relaxed),
            hit_rate: if total > 0 { hits as f64 / total as f64 } else { 0.0 },
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub size: usize,
    pub capacity: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use cdfa_core::types::SignalId;
    
    #[test]
    fn test_lock_free_signal_buffer() {
        let buffer = LockFreeSignalBuffer::new(100);
        
        // Test push and pop
        let signal = Signal::new(SignalId(1), 1000, vec![1.0, 2.0, 3.0]);
        assert!(buffer.push(signal.clone()).is_ok());
        assert_eq!(buffer.len(), 1);
        
        let popped = buffer.try_pop().unwrap();
        assert_eq!(popped.id, signal.id);
        assert!(buffer.is_empty());
        
        // Test stats
        let stats = buffer.stats();
        assert_eq!(stats.total_written, 1);
        assert_eq!(stats.total_processed, 1);
    }
    
    #[test]
    fn test_wait_free_correlation_matrix() {
        let matrix = WaitFreeCorrelationMatrix::new(3);
        
        // Test update and get
        matrix.update(0, 1, 0.5);
        assert_eq!(matrix.get(0, 1), 0.5);
        
        // Test update count
        assert_eq!(matrix.update_count(0, 1), 1);
        
        // Test CAS
        let old = matrix.compare_and_swap(0, 1, 0.5, 0.7);
        assert_eq!(old, 0.5);
        assert_eq!(matrix.get(0, 1), 0.7);
    }
    
    #[test]
    fn test_lock_free_result_aggregator() {
        let aggregator = LockFreeResultAggregator::new();
        
        // Add results
        let mut result1 = AnalysisResult::new("test1".to_string(), 0.8, 0.9);
        result1.latency_ns = 1000;
        aggregator.add_result(result1);
        
        let mut result2 = AnalysisResult::new("test2".to_string(), 0.7, 0.85);
        result2.latency_ns = 1500;
        aggregator.add_result(result2);
        
        // Check stats
        let stats = aggregator.stats();
        assert_eq!(stats.count, 2);
        assert_eq!(stats.avg_latency_ns, 1250);
        assert_eq!(stats.min_latency_ns, 1000);
        assert_eq!(stats.max_latency_ns, 1500);
        assert!((stats.avg_confidence - 0.875).abs() < 0.001);
        
        // Collect results
        let results = aggregator.collect_results();
        assert_eq!(results.len(), 2);
    }
    
    #[test]
    fn test_lock_free_feature_cache() {
        let cache = LockFreeFeatureCache::new(10);
        
        // Test insert and get
        cache.insert(1, vec![1.0, 2.0, 3.0], 1000);
        let data = cache.get(1).unwrap();
        assert_eq!(data.len(), 3);
        
        // Test cache miss
        assert!(cache.get(2).is_none());
        
        // Check stats
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate, 0.5);
    }
}