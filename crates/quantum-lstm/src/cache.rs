//! Cache implementation for quantum operations

use crate::types::*;
use lru::LruCache;
use std::sync::Mutex;
use std::num::NonZeroUsize;

/// Thread-safe cache for quantum states
pub struct QuantumCache {
    cache: Mutex<LruCache<CacheKey, QuantumState>>,
}

impl QuantumCache {
    /// Create new cache with given capacity
    pub fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(100).unwrap());
        Self {
            cache: Mutex::new(LruCache::new(cap)),
        }
    }
    
    /// Get cached state
    pub fn get(&self, key: CacheKey) -> Option<QuantumState> {
        self.cache.lock().ok()?.get(&key).cloned()
    }
    
    /// Put state in cache
    pub fn put(&self, key: CacheKey, state: QuantumState) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.put(key, state);
        }
    }
    
    /// Clear cache
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }
}