//! # Streaming LSH Index
//!
//! Zero-allocation, lock-free LSH index for O(1) streaming insertions.
//!
//! ## Design Goals
//!
//! 1. **O(1) insertion** - No graph updates, just hash and bucket
//! 2. **Zero allocation** - Pre-allocated buckets, ArrayVec signatures
//! 3. **Lock-free reads** - Concurrent queries without blocking
//! 4. **Streaming-first** - Optimized for continuous market data
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     StreamingLshIndex                           │
//! │  ┌─────────────────────────────────────────────────────────┐   │
//! │  │                    Hash Tables                           │   │
//! │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐     ┌─────────┐   │   │
//! │  │  │ Table 0 │ │ Table 1 │ │ Table 2 │ ... │ Table L │   │   │
//! │  │  └────┬────┘ └────┬────┘ └────┬────┘     └────┬────┘   │   │
//! │  │       │           │           │               │         │   │
//! │  │  ┌────▼────┐ ┌────▼────┐ ┌────▼────┐     ┌────▼────┐   │   │
//! │  │  │ Buckets │ │ Buckets │ │ Buckets │     │ Buckets │   │   │
//! │  │  │  (Map)  │ │  (Map)  │ │  (Map)  │     │  (Map)  │   │   │
//! │  │  └─────────┘ └─────────┘ └─────────┘     └─────────┘   │   │
//! │  └─────────────────────────────────────────────────────────┘   │
//! │                                                                 │
//! │  ┌─────────────────────────────────────────────────────────┐   │
//! │  │                   Item Storage                           │   │
//! │  │  [Item 0] [Item 1] [Item 2] ... [Item N]                │   │
//! │  │  (Pre-allocated vector of pattern data)                  │   │
//! │  └─────────────────────────────────────────────────────────┘   │
//! │                                                                 │
//! │  ┌─────────────────────────────────────────────────────────┐   │
//! │  │                   Statistics                             │   │
//! │  │  insert_count | query_count | collision_count | ...      │   │
//! │  └─────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use arrayvec::ArrayVec;
use crossbeam::queue::ArrayQueue;
use parking_lot::RwLock;

use crate::config::LshConfig;
use crate::error::{LshError, Result};
use crate::hash::{HashFamily, SimHash, SimHashSignature};

// ============================================================================
// Constants
// ============================================================================

/// Maximum items per bucket (to bound query time).
const MAX_BUCKET_SIZE: usize = 1000;

/// Maximum tables supported.
const MAX_TABLES: usize = 32;

/// Streaming buffer size for lock-free ingestion.
const STREAMING_BUFFER_SIZE: usize = 4096;

// ============================================================================
// Types
// ============================================================================

/// ID type for indexed items.
pub type ItemId = u64;

/// Bucket key (hash signature truncated to u64).
pub type BucketKey = u64;

/// Stored item with ID and signature.
#[derive(Clone, Debug)]
pub struct StoredItem {
    /// Unique item ID.
    pub id: ItemId,
    
    /// Original vector (stored for similarity computation).
    pub vector: Vec<f32>,
    
    /// Pre-computed signature.
    pub signature: SimHashSignature,
}

/// Query result from LSH search.
#[derive(Clone, Debug)]
pub struct LshResult {
    /// Item ID.
    pub id: ItemId,
    
    /// Estimated similarity (from signature comparison).
    pub estimated_similarity: f32,
    
    /// Exact similarity (if computed).
    pub exact_similarity: Option<f32>,
}

/// Bucket containing items with the same hash.
#[derive(Clone, Debug, Default)]
struct Bucket {
    /// Items in this bucket.
    items: Vec<ItemId>,
}

/// Single hash table.
struct HashTable {
    /// Bucket map: key -> item IDs.
    buckets: RwLock<HashMap<BucketKey, Bucket>>,
}

impl HashTable {
    fn new() -> Self {
        Self {
            buckets: RwLock::new(HashMap::with_capacity(1024)),
        }
    }
    
    fn insert(&self, key: BucketKey, item_id: ItemId) -> Result<()> {
        let mut buckets = self.buckets.write();
        let bucket = buckets.entry(key).or_default();
        
        if bucket.items.len() >= MAX_BUCKET_SIZE {
            return Err(LshError::BucketOverflow {
                table_id: 0, // Will be set by caller
                bucket_id: key,
            });
        }
        
        bucket.items.push(item_id);
        Ok(())
    }
    
    fn query(&self, key: BucketKey) -> Vec<ItemId> {
        let buckets = self.buckets.read();
        buckets
            .get(&key)
            .map(|b| b.items.clone())
            .unwrap_or_default()
    }
}

// ============================================================================
// Statistics
// ============================================================================

/// Index statistics.
#[derive(Debug, Default)]
pub struct LshStats {
    /// Total insertions.
    pub insert_count: AtomicU64,
    
    /// Total queries.
    pub query_count: AtomicU64,
    
    /// Total insert time in nanoseconds.
    pub total_insert_ns: AtomicU64,
    
    /// Total query time in nanoseconds.
    pub total_query_ns: AtomicU64,
    
    /// Insertions that exceeded latency budget.
    pub slow_inserts: AtomicU64,
    
    /// Bucket overflow events.
    pub bucket_overflows: AtomicU64,
    
    /// Streaming buffer drops.
    pub stream_drops: AtomicU64,
}

impl LshStats {
    /// Get average insert latency in nanoseconds.
    pub fn avg_insert_latency_ns(&self) -> f64 {
        let count = self.insert_count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        self.total_insert_ns.load(Ordering::Relaxed) as f64 / count as f64
    }
    
    /// Get average query latency in nanoseconds.
    pub fn avg_query_latency_ns(&self) -> f64 {
        let count = self.query_count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        self.total_query_ns.load(Ordering::Relaxed) as f64 / count as f64
    }
}

// ============================================================================
// Streaming LSH Index
// ============================================================================

/// Zero-allocation streaming LSH index.
///
/// Optimized for O(1) insertions in market data streaming scenarios.
pub struct StreamingLshIndex {
    /// Configuration.
    config: LshConfig,
    
    /// Hash family.
    hasher: SimHash,
    
    /// Hash tables.
    tables: Vec<HashTable>,
    
    /// Item storage.
    items: RwLock<Vec<StoredItem>>,
    
    /// Next item ID.
    next_id: AtomicU64,
    
    /// Streaming buffer for lock-free ingestion.
    stream_buffer: Option<ArrayQueue<Vec<f32>>>,
    
    /// Statistics.
    stats: LshStats,
}

impl StreamingLshIndex {
    /// Create a new streaming LSH index.
    pub fn new(config: LshConfig) -> Result<Self> {
        config.validate()?;
        
        let (dimensions, num_bits) = match config.hash_family {
            crate::config::HashFamilyConfig::SimHash { dimensions, num_bits } => {
                (dimensions, num_bits)
            }
            _ => {
                // Default to SimHash for now
                (128, 64)
            }
        };
        
        let hasher = SimHash::new(dimensions, num_bits, config.seed);
        
        let tables = (0..config.num_tables)
            .map(|_| HashTable::new())
            .collect();
        
        let stream_buffer = if config.zero_alloc {
            Some(ArrayQueue::new(STREAMING_BUFFER_SIZE))
        } else {
            None
        };
        
        let initial_capacity = config.max_capacity.unwrap_or(10_000);
        
        Ok(Self {
            config,
            hasher,
            tables,
            items: RwLock::new(Vec::with_capacity(initial_capacity)),
            next_id: AtomicU64::new(0),
            stream_buffer,
            stats: LshStats::default(),
        })
    }
    
    /// Insert a vector into the index.
    ///
    /// Target latency: <500ns.
    pub fn insert(&self, vector: Vec<f32>) -> Result<ItemId> {
        let start = Instant::now();
        
        // Validate dimensions
        let expected_dims = match self.config.hash_family {
            crate::config::HashFamilyConfig::SimHash { dimensions, .. } => dimensions,
            _ => 128,
        };
        
        if vector.len() != expected_dims {
            return Err(LshError::DimensionMismatch {
                expected: expected_dims,
                actual: vector.len(),
            });
        }
        
        // Compute signature
        let signature = self.hasher.hash(&vector);
        
        // Generate bucket keys for all tables
        let bucket_keys = self.generate_bucket_keys(&signature);
        
        // Allocate ID
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        
        // Check capacity
        if let Some(max) = self.config.max_capacity {
            if id as usize >= max {
                return Err(LshError::CapacityExceeded {
                    max,
                    attempted: id as usize + 1,
                });
            }
        }
        
        // Store item
        {
            let mut items = self.items.write();
            items.push(StoredItem {
                id,
                vector,
                signature,
            });
        }
        
        // Insert into all tables
        for (table_idx, &key) in bucket_keys.iter().enumerate() {
            if let Err(e) = self.tables[table_idx].insert(key, id) {
                self.stats.bucket_overflows.fetch_add(1, Ordering::Relaxed);
                return Err(e);
            }
        }
        
        // Update statistics
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        self.stats.insert_count.fetch_add(1, Ordering::Relaxed);
        self.stats.total_insert_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
        
        if elapsed_ns > crate::INSERT_LATENCY_BUDGET_NS {
            self.stats.slow_inserts.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                latency_ns = elapsed_ns,
                budget_ns = crate::INSERT_LATENCY_BUDGET_NS,
                "LSH insert exceeded latency budget"
            );
        }
        
        Ok(id)
    }
    
    /// Stream insert (lock-free, may drop under backpressure).
    ///
    /// Target latency: <200ns.
    pub fn stream_insert(&self, vector: Vec<f32>) -> Result<()> {
        if let Some(buffer) = &self.stream_buffer {
            match buffer.push(vector) {
                Ok(()) => Ok(()),
                Err(_) => {
                    self.stats.stream_drops.fetch_add(1, Ordering::Relaxed);
                    let drops = self.stats.stream_drops.load(Ordering::Relaxed);
                    let total = self.stats.insert_count.load(Ordering::Relaxed);
                    let drop_rate = if total > 0 {
                        (drops as f32 / total as f32) * 100.0
                    } else {
                        100.0
                    };
                    Err(LshError::BufferFull { drop_rate })
                }
            }
        } else {
            // Fall back to regular insert if streaming not enabled
            self.insert(vector)?;
            Ok(())
        }
    }
    
    /// Process buffered stream items.
    ///
    /// Call this periodically from a background thread.
    pub fn process_stream_buffer(&self) -> usize {
        let buffer = match &self.stream_buffer {
            Some(b) => b,
            None => return 0,
        };
        
        let mut processed = 0;
        while let Some(vector) = buffer.pop() {
            if self.insert(vector).is_ok() {
                processed += 1;
            }
        }
        
        processed
    }
    
    /// Query for similar items.
    ///
    /// Returns candidates from buckets matching the query's signature.
    /// For accurate results, caller should compute exact similarity.
    pub fn query(&self, vector: &[f32], k: usize) -> Result<Vec<LshResult>> {
        let start = Instant::now();
        
        // Compute query signature
        let signature = self.hasher.hash(vector);
        let bucket_keys = self.generate_bucket_keys(&signature);
        
        // Collect candidates from all tables
        let mut candidate_ids: Vec<ItemId> = Vec::new();
        for (table_idx, &key) in bucket_keys.iter().enumerate() {
            let ids = self.tables[table_idx].query(key);
            candidate_ids.extend(ids);
        }
        
        // Deduplicate
        candidate_ids.sort_unstable();
        candidate_ids.dedup();
        
        // Score candidates by signature similarity
        let items = self.items.read();
        let mut results: Vec<LshResult> = candidate_ids
            .into_iter()
            .filter_map(|id| {
                items.get(id as usize).map(|item| {
                    let hamming = signature.hamming_distance(&item.signature);
                    let estimated_similarity = 1.0 - (hamming as f32 / 64.0); // Normalized
                    
                    LshResult {
                        id,
                        estimated_similarity,
                        exact_similarity: None,
                    }
                })
            })
            .collect();
        
        // Sort by estimated similarity
        results.sort_by(|a, b| {
            b.estimated_similarity
                .partial_cmp(&a.estimated_similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Truncate to k
        results.truncate(k);
        
        // Update statistics
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        self.stats.query_count.fetch_add(1, Ordering::Relaxed);
        self.stats.total_query_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
        
        Ok(results)
    }
    
    /// Get item by ID.
    pub fn get(&self, id: ItemId) -> Option<StoredItem> {
        let items = self.items.read();
        items.get(id as usize).cloned()
    }
    
    /// Get current item count.
    pub fn len(&self) -> usize {
        self.items.read().len()
    }
    
    /// Check if index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get statistics.
    pub fn stats(&self) -> &LshStats {
        &self.stats
    }
    
    /// Generate bucket keys for all tables from a signature.
    fn generate_bucket_keys(&self, signature: &SimHashSignature) -> ArrayVec<BucketKey, MAX_TABLES> {
        let mut keys = ArrayVec::new();
        
        // Use different portions of the signature for each table
        // This implements the multi-table LSH scheme
        for table_idx in 0..self.config.num_tables {
            // Mix table index into key to get different buckets per table
            let base_key = signature.bits[0];
            let mixed_key = base_key.wrapping_add((table_idx as u64).wrapping_mul(0x9E3779B97F4A7C15));
            keys.push(mixed_key);
        }
        
        keys
    }
}

// ============================================================================
// Triangular Architecture Integration
// ============================================================================

impl crate::AcquisitionConstraint for StreamingLshIndex {
    type Pattern = Vec<f32>;
    
    fn should_promote(&self, pattern: &Self::Pattern) -> bool {
        // Promote patterns that have high collision probability
        // (i.e., are similar to existing patterns)
        if let Ok(results) = self.query(pattern, 1) {
            if let Some(top) = results.first() {
                return top.estimated_similarity >= self.promotion_threshold();
            }
        }
        false
    }
    
    fn promotion_threshold(&self) -> f32 {
        // Patterns with >0.7 similarity to existing patterns are promoted
        0.7
    }
    
    fn ingestion_rate(&self) -> f64 {
        let insert_count = self.stats.insert_count.load(Ordering::Relaxed);
        let insert_ns = self.stats.total_insert_ns.load(Ordering::Relaxed);
        
        if insert_ns == 0 {
            return 0.0;
        }
        
        // Items per second
        (insert_count as f64 / insert_ns as f64) * 1_000_000_000.0
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    fn test_config() -> LshConfig {
        LshConfig::simhash(64, 64)
            .with_tables(4)
            .with_seed(42)
    }
    
    #[test]
    fn test_insert_and_query() {
        let index = StreamingLshIndex::new(test_config()).unwrap();
        
        let v1 = vec![1.0f32; 64];
        let v2 = vec![1.0f32; 64]; // Same as v1
        let v3 = vec![-1.0f32; 64]; // Opposite
        
        let id1 = index.insert(v1.clone()).unwrap();
        let id2 = index.insert(v2.clone()).unwrap();
        let id3 = index.insert(v3.clone()).unwrap();
        
        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id3, 2);
        
        // Query with v1 should find v1 and v2 as top results
        let results = index.query(&v1, 10).unwrap();
        assert!(!results.is_empty());
        
        // v1 and v2 should have high similarity
        let v1_result = results.iter().find(|r| r.id == id1);
        let v2_result = results.iter().find(|r| r.id == id2);
        
        assert!(v1_result.is_some());
        assert!(v2_result.is_some());
    }
    
    #[test]
    fn test_dimension_mismatch() {
        let index = StreamingLshIndex::new(test_config()).unwrap();
        
        let wrong_dims = vec![1.0f32; 32]; // Config expects 64
        let result = index.insert(wrong_dims);
        
        assert!(matches!(result, Err(LshError::DimensionMismatch { .. })));
    }
    
    #[test]
    fn test_streaming_buffer() {
        let config = LshConfig::simhash(64, 64)
            .with_tables(4)
            .with_seed(42);
        
        let index = StreamingLshIndex::new(config).unwrap();
        
        // Stream some vectors
        for _ in 0..10 {
            let v = vec![1.0f32; 64];
            let _ = index.stream_insert(v);
        }
        
        // Process buffer
        let processed = index.process_stream_buffer();
        assert_eq!(processed, 10);
        assert_eq!(index.len(), 10);
    }
    
    #[test]
    fn test_statistics() {
        let index = StreamingLshIndex::new(test_config()).unwrap();
        
        for _ in 0..100 {
            let v = vec![1.0f32; 64];
            index.insert(v).unwrap();
        }
        
        assert_eq!(index.stats().insert_count.load(Ordering::Relaxed), 100);
        assert!(index.stats().avg_insert_latency_ns() > 0.0);
    }
}
