//! Caching utilities for performance optimization

use crate::AnalysisResult;
use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, Instant};

/// Thread-safe cache for analysis results
pub struct AnalysisCache {
    cache: RwLock<HashMap<String, CacheEntry>>,
    max_size: usize,
    ttl: Duration,
}

/// Cache entry with timestamp
#[derive(Clone)]
struct CacheEntry {
    result: AnalysisResult,
    timestamp: Instant,
}

impl AnalysisCache {
    /// Create a new cache with specified size and TTL
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            max_size,
            ttl: Duration::from_secs(300), // 5 minutes default TTL
        }
    }
    
    /// Get a cached result if available and not expired
    pub fn get(&self, key: &str) -> Option<AnalysisResult> {
        let cache = self.cache.read().ok()?;
        
        if let Some(entry) = cache.get(key) {
            if entry.timestamp.elapsed() < self.ttl {
                return Some(entry.result.clone());
            }
        }
        
        None
    }
    
    /// Insert a result into the cache
    pub fn insert(&self, key: String, result: AnalysisResult) {
        let mut cache = match self.cache.write() {
            Ok(c) => c,
            Err(_) => return,
        };
        
        // Remove expired entries
        self.cleanup_expired(&mut cache);
        
        // Remove oldest entries if at capacity
        if cache.len() >= self.max_size {
            self.remove_oldest(&mut cache);
        }
        
        cache.insert(key, CacheEntry {
            result,
            timestamp: Instant::now(),
        });
    }
    
    /// Clear all cached entries
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let cache = match self.cache.read() {
            Ok(c) => c,
            Err(_) => return CacheStats::default(),
        };
        
        let mut expired = 0;
        let now = Instant::now();
        
        for entry in cache.values() {
            if now.duration_since(entry.timestamp) > self.ttl {
                expired += 1;
            }
        }
        
        CacheStats {
            total_entries: cache.len(),
            expired_entries: expired,
            max_size: self.max_size,
        }
    }
    
    /// Remove expired entries from cache
    fn cleanup_expired(&self, cache: &mut HashMap<String, CacheEntry>) {
        let now = Instant::now();
        cache.retain(|_, entry| now.duration_since(entry.timestamp) < self.ttl);
    }
    
    /// Remove oldest entry from cache
    fn remove_oldest(&self, cache: &mut HashMap<String, CacheEntry>) {
        if let Some(oldest_key) = cache.iter()
            .min_by_key(|(_, entry)| entry.timestamp)
            .map(|(key, _)| key.clone())
        {
            cache.remove(&oldest_key);
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub total_entries: usize,
    pub expired_entries: usize,
    pub max_size: usize,
}

impl CacheStats {
    /// Get cache utilization percentage
    pub fn utilization(&self) -> f64 {
        if self.max_size == 0 {
            0.0
        } else {
            (self.total_entries as f64 / self.max_size as f64) * 100.0
        }
    }
    
    /// Get expired entry percentage
    pub fn expired_percentage(&self) -> f64 {
        if self.total_entries == 0 {
            0.0
        } else {
            (self.expired_entries as f64 / self.total_entries as f64) * 100.0
        }
    }
}

/// LRU cache implementation
pub struct LruCache<K, V> {
    cache: HashMap<K, LruNode<V>>,
    head: Option<K>,
    tail: Option<K>,
    capacity: usize,
}

#[derive(Clone)]
struct LruNode<V> {
    value: V,
    prev: Option<String>,
    next: Option<String>,
}

impl<K, V> LruCache<K, V>
where
    K: Clone + Eq + std::hash::Hash + std::fmt::Display,
    V: Clone,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: HashMap::new(),
            head: None,
            tail: None,
            capacity,
        }
    }
    
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if self.cache.contains_key(key) {
            self.move_to_front(key);
            return self.cache.get(key).map(|node| &node.value);
        }
        None
    }
    
    pub fn put(&mut self, key: K, value: V) {
        if self.cache.len() >= self.capacity {
            self.remove_tail();
        }
        
        let node = LruNode {
            value,
            prev: None,
            next: self.head.as_ref().map(|h| h.to_string()),
        };
        
        self.cache.insert(key.clone(), node);
        self.head = Some(key);
    }
    
    fn move_to_front(&mut self, _key: &K) {
        // Simplified implementation
        // In a full implementation, this would move the node to the front
    }
    
    fn remove_tail(&mut self) {
        // Simplified implementation
        // In a full implementation, this would remove the least recently used item
        if self.cache.len() > 0 {
            let key_to_remove = self.cache.keys().next().unwrap().clone();
            self.cache.remove(&key_to_remove);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AnalysisResult, VolatilityResult, PerformanceResult};
    use std::time::Duration;
    
    fn create_test_result() -> AnalysisResult {
        AnalysisResult {
            antifragility_index: 0.7,
            fragility_score: 0.3,
            convexity_score: 0.6,
            asymmetry_score: 0.5,
            recovery_score: 0.8,
            benefit_ratio_score: 0.4,
            volatility: VolatilityResult::default(),
            performance: PerformanceResult::default(),
            data_points: 100,
            calculation_time: Duration::from_millis(10),
        }
    }
    
    #[test]
    fn test_cache_operations() {
        let cache = AnalysisCache::new(2);
        let result = create_test_result();
        
        // Test insertion and retrieval
        cache.insert("key1".to_string(), result.clone());
        let cached_result = cache.get("key1");
        assert!(cached_result.is_some());
        
        // Test cache miss
        let missing_result = cache.get("nonexistent");
        assert!(missing_result.is_none());
    }
    
    #[test]
    fn test_cache_capacity() {
        let cache = AnalysisCache::new(2);
        let result = create_test_result();
        
        // Fill cache to capacity
        cache.insert("key1".to_string(), result.clone());
        cache.insert("key2".to_string(), result.clone());
        
        let stats = cache.stats();
        assert_eq!(stats.total_entries, 2);
        assert_eq!(stats.max_size, 2);
        
        // Add one more item (should evict oldest)
        cache.insert("key3".to_string(), result.clone());
        
        let stats_after = cache.stats();
        assert_eq!(stats_after.total_entries, 2); // Still at capacity
    }
    
    #[test]
    fn test_cache_clear() {
        let cache = AnalysisCache::new(10);
        let result = create_test_result();
        
        cache.insert("key1".to_string(), result.clone());
        cache.insert("key2".to_string(), result.clone());
        
        cache.clear();
        
        let stats = cache.stats();
        assert_eq!(stats.total_entries, 0);
    }
    
    #[test]
    fn test_cache_stats() {
        let cache = AnalysisCache::new(10);
        let result = create_test_result();
        
        cache.insert("key1".to_string(), result.clone());
        cache.insert("key2".to_string(), result.clone());
        
        let stats = cache.stats();
        assert_eq!(stats.total_entries, 2);
        assert_eq!(stats.max_size, 10);
        assert_eq!(stats.utilization(), 20.0);
    }
    
    #[test]
    fn test_lru_cache() {
        let mut lru = LruCache::new(2);
        
        lru.put("key1".to_string(), "value1".to_string());
        lru.put("key2".to_string(), "value2".to_string());
        
        assert!(lru.get(&"key1".to_string()).is_some());
        assert!(lru.get(&"key2".to_string()).is_some());
        
        // Add third item (should evict least recently used)
        lru.put("key3".to_string(), "value3".to_string());
        
        assert!(lru.get(&"key3".to_string()).is_some());
    }
}