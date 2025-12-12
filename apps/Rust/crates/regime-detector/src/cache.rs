//! High-performance cache for regime detection results

use crate::types::RegimeDetectionResult;
use ahash::AHashMap;
use parking_lot::RwLock;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

/// Cache key based on price/volume fingerprint
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CacheKey {
    price_hash: u64,
    volume_hash: u64,
    window_size: usize,
}

/// Cache entry with TTL and access tracking
#[derive(Debug)]
struct CacheEntry {
    result: RegimeDetectionResult,
    created_at: u64,
    access_count: AtomicU64,
}

impl Clone for CacheEntry {
    fn clone(&self) -> Self {
        Self {
            result: self.result.clone(),
            created_at: self.created_at,
            access_count: AtomicU64::new(self.access_count.load(Ordering::Relaxed)),
        }
    }
}

/// High-performance cache for regime detection
pub struct RegimeCache {
    cache: RwLock<AHashMap<CacheKey, CacheEntry>>,
    max_entries: usize,
    ttl_ns: u64,
}

impl RegimeCache {
    pub fn new(max_entries: usize, ttl_seconds: u64) -> Self {
        Self {
            cache: RwLock::new(AHashMap::with_capacity(max_entries)),
            max_entries,
            ttl_ns: ttl_seconds * 1_000_000_000,
        }
    }
    
    /// Generate cache key from price and volume data
    #[inline(always)]
    fn generate_key(&self, prices: &[f32], volumes: &[f32], window_size: usize) -> CacheKey {
        let price_hash = self.fast_hash(prices);
        let volume_hash = self.fast_hash(volumes);
        
        CacheKey {
            price_hash,
            volume_hash,
            window_size,
        }
    }
    
    /// Fast hash function for numerical data
    #[inline(always)]
    fn fast_hash(&self, data: &[f32]) -> u64 {
        let mut hasher = DefaultHasher::new();
        
        // Hash key statistics instead of all data for speed
        if !data.is_empty() {
            let len = data.len();
            
            // Sample key points for hash
            let indices = [
                0,
                len / 4,
                len / 2,
                3 * len / 4,
                len - 1,
            ];
            
            for &idx in &indices {
                if idx < len {
                    data[idx].to_bits().hash(&mut hasher);
                }
            }
            
            len.hash(&mut hasher);
        }
        
        hasher.finish()
    }
    
    /// Get cached result if available and valid
    #[inline(always)]
    pub fn get(&self, prices: &[f32], volumes: &[f32], window_size: usize) -> Option<RegimeDetectionResult> {
        let key = self.generate_key(prices, volumes, window_size);
        
        let cache = self.cache.read();
        if let Some(entry) = cache.get(&key) {
            let now = current_time_ns();
            
            // Check TTL
            if now - entry.created_at < self.ttl_ns {
                entry.access_count.fetch_add(1, Ordering::Relaxed);
                return Some(entry.result.clone());
            }
        }
        
        None
    }
    
    /// Store result in cache
    #[inline(always)]
    pub fn put(&self, prices: &[f32], volumes: &[f32], window_size: usize, result: RegimeDetectionResult) {
        let key = self.generate_key(prices, volumes, window_size);
        let now = current_time_ns();
        
        let entry = CacheEntry {
            result,
            created_at: now,
            access_count: AtomicU64::new(1),
        };
        
        let mut cache = self.cache.write();
        
        // Evict if at capacity
        if cache.len() >= self.max_entries {
            self.evict_lru(&mut cache);
        }
        
        cache.insert(key, entry);
    }
    
    /// Evict least recently used entry
    fn evict_lru(&self, cache: &mut AHashMap<CacheKey, CacheEntry>) {
        let now = current_time_ns();
        
        // Find oldest or least accessed entry
        let mut oldest_key = None;
        let mut oldest_score = f64::MAX;
        
        for (key, entry) in cache.iter() {
            let age = now - entry.created_at;
            let access_count = entry.access_count.load(Ordering::Relaxed);
            
            // Score combines age and access frequency
            let score = age as f64 / (access_count as f64 + 1.0);
            
            if score < oldest_score {
                oldest_score = score;
                oldest_key = Some(*key);
            }
        }
        
        if let Some(key) = oldest_key {
            cache.remove(&key);
        }
    }
    
    /// Clear expired entries
    pub fn cleanup(&self) {
        let now = current_time_ns();
        let mut cache = self.cache.write();
        
        cache.retain(|_, entry| now - entry.created_at < self.ttl_ns);
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let cache = self.cache.read();
        let size = cache.len();
        let capacity = self.max_entries;
        
        let total_accesses: u64 = cache.values()
            .map(|entry| entry.access_count.load(Ordering::Relaxed))
            .sum();
        
        CacheStats {
            size,
            capacity,
            total_accesses,
            hit_rate: if total_accesses > 0 {
                size as f64 / total_accesses as f64
            } else {
                0.0
            },
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub size: usize,
    pub capacity: usize,
    pub total_accesses: u64,
    pub hit_rate: f64,
}

/// Get current time in nanoseconds
#[inline(always)]
fn current_time_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{RegimeFeatures, MarketRegime};
    
    #[test]
    fn test_cache_basic_operations() {
        let cache = RegimeCache::new(10, 1);
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let volumes = vec![100.0, 200.0, 300.0, 400.0, 500.0];
        
        // Should be empty initially
        assert!(cache.get(&prices, &volumes, 5).is_none());
        
        // Add entry
        let result = RegimeDetectionResult {
            regime: MarketRegime::TrendingBull,
            confidence: 0.8,
            features: RegimeFeatures::default(),
            transition_probs: vec![],
            latency_ns: 100,
        };
        
        cache.put(&prices, &volumes, 5, result.clone());
        
        // Should retrieve the same result
        let cached = cache.get(&prices, &volumes, 5).unwrap();
        assert_eq!(cached.regime, MarketRegime::TrendingBull);
        assert_eq!(cached.confidence, 0.8);
    }
    
    #[test]
    fn test_cache_performance() {
        let cache = RegimeCache::new(1000, 1);
        let prices: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let volumes: Vec<f32> = (0..100).map(|i| (i * 10) as f32).collect();
        
        let start = std::time::Instant::now();
        
        // Test key generation speed
        for _ in 0..1000 {
            let _ = cache.generate_key(&prices, &volumes, 50);
        }
        
        let elapsed = start.elapsed();
        assert!(elapsed.as_micros() < 100, "Key generation too slow: {}μs", elapsed.as_micros());
    }
}