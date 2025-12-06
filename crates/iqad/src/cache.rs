//! High-performance caching for IQAD

use dashmap::DashMap;
use moka::future::Cache;
use parking_lot::RwLock;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;

/// Cache key for quantum circuit results
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct CacheKey {
    circuit_name: String,
    param_hash: u64,
}

impl CacheKey {
    /// Create a new cache key
    pub fn new(circuit_name: &str, params: &[f64]) -> Self {
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        for &p in params {
            // Convert to bits for consistent hashing
            p.to_bits().hash(&mut hasher);
        }
        
        Self {
            circuit_name: circuit_name.to_string(),
            param_hash: hasher.finish(),
        }
    }
}

/// Thread-safe cache for quantum circuit results
pub struct QuantumCache {
    /// Async cache using moka
    cache: Cache<CacheKey, Vec<f64>>,
    /// Hit/miss statistics
    stats: Arc<RwLock<CacheStats>>,
}

/// Cache statistics
#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl QuantumCache {
    /// Create a new quantum cache
    pub fn new(max_capacity: u64) -> Self {
        let cache = Cache::builder()
            .max_capacity(max_capacity)
            .time_to_live(Duration::from_secs(900)) // 15 minutes TTL
            .time_to_idle(Duration::from_secs(300)) // 5 minutes idle
            .eviction_listener(|_key, _value, cause| {
                tracing::debug!("Cache eviction: {:?}", cause);
            })
            .build();
            
        Self {
            cache,
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }
    
    /// Get a value from cache
    pub async fn get(&self, key: &CacheKey) -> Option<Vec<f64>> {
        let result = self.cache.get(key).await;
        
        // Update stats
        let mut stats = self.stats.write();
        if result.is_some() {
            stats.hits += 1;
        } else {
            stats.misses += 1;
        }
        
        result
    }
    
    /// Insert a value into cache
    pub async fn insert(&self, key: CacheKey, value: Vec<f64>) {
        self.cache.insert(key, value).await;
    }
    
    /// Clear the cache
    pub async fn clear(&self) {
        self.cache.invalidate_all();
        self.cache.run_pending_tasks().await;
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.read().clone()
    }
    
    /// Get hit rate
    pub fn hit_rate(&self) -> f64 {
        let stats = self.stats.read();
        let total = stats.hits + stats.misses;
        if total == 0 {
            0.0
        } else {
            stats.hits as f64 / total as f64
        }
    }
}

/// Circuit result cache using DashMap for sync access
pub struct CircuitCache {
    cache: DashMap<CacheKey, Vec<f64>>,
    max_size: usize,
}

impl CircuitCache {
    /// Create a new circuit cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: DashMap::with_capacity(max_size),
            max_size,
        }
    }
    
    /// Get from cache
    pub fn get(&self, key: &CacheKey) -> Option<Vec<f64>> {
        self.cache.get(key).map(|v| v.clone())
    }
    
    /// Insert into cache
    pub fn insert(&self, key: CacheKey, value: Vec<f64>) {
        // Simple LRU eviction if at capacity
        if self.cache.len() >= self.max_size {
            // Remove a random entry (not true LRU but fast)
            if let Some(entry) = self.cache.iter().next() {
                let k = entry.key().clone();
                drop(entry);
                self.cache.remove(&k);
            }
        }
        self.cache.insert(key, value);
    }
    
    /// Clear cache
    pub fn clear(&self) {
        self.cache.clear();
    }
}