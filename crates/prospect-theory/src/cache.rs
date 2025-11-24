//! Cache implementation for Prospect Theory
//! High-performance caching with LRU eviction and metrics collection

use std::collections::HashMap;
use std::time::Instant;

pub use crate::metrics::ProspectTheoryCache;

#[derive(Debug, Clone)]
pub struct CacheKey {
    pub operation: String,
    pub parameters: Vec<String>,
}

impl CacheKey {
    pub fn new(operation: &str, params: &[f64]) -> Self {
        Self {
            operation: operation.to_string(),
            parameters: params.iter().map(|p| format!("{:.6}", p)).collect(),
        }
    }
    
    pub fn to_string(&self) -> String {
        format!("{}:{}", self.operation, self.parameters.join(","))
    }
}

#[derive(Debug)]
pub struct ProspectTheoryValueCache {
    value_cache: ProspectTheoryCache,
    probability_cache: ProspectTheoryCache,
}

impl ProspectTheoryValueCache {
    pub fn new(size: usize) -> Self {
        Self {
            value_cache: ProspectTheoryCache::new(size / 2),
            probability_cache: ProspectTheoryCache::new(size / 2),
        }
    }
    
    pub fn get_value(&mut self, key: &CacheKey) -> Option<f64> {
        self.value_cache.get(&key.to_string())
    }
    
    pub fn set_value(&mut self, key: CacheKey, value: f64) {
        self.value_cache.insert(key.to_string(), value);
    }
    
    pub fn get_probability(&mut self, key: &CacheKey) -> Option<f64> {
        self.probability_cache.get(&key.to_string())
    }
    
    pub fn set_probability(&mut self, key: CacheKey, probability: f64) {
        self.probability_cache.insert(key.to_string(), probability);
    }
    
    pub fn clear_all(&mut self) {
        self.value_cache.clear();
        self.probability_cache.clear();
    }
    
    pub fn value_hit_rate(&self) -> f64 {
        self.value_cache.hit_rate()
    }
    
    pub fn probability_hit_rate(&self) -> f64 {
        self.probability_cache.hit_rate()
    }
    
    pub fn overall_hit_rate(&self) -> f64 {
        (self.value_hit_rate() + self.probability_hit_rate()) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_creation() {
        let key = CacheKey::new("value_function", &[1.5, -2.3, 0.8]);
        assert_eq!(key.operation, "value_function");
        assert_eq!(key.parameters.len(), 3);
        assert!(key.to_string().contains("value_function"));
    }

    #[test]
    fn test_cache_operations() {
        let mut cache = ProspectTheoryValueCache::new(100);
        let key = CacheKey::new("test", &[1.0]);
        
        assert!(cache.get_value(&key).is_none());
        
        cache.set_value(key.clone(), 42.0);
        assert_eq!(cache.get_value(&key), Some(42.0));
    }

    #[test]
    fn test_cache_hit_rates() {
        let mut cache = ProspectTheoryValueCache::new(100);
        let key1 = CacheKey::new("test1", &[1.0]);
        let key2 = CacheKey::new("test2", &[2.0]);
        
        // Miss
        cache.get_value(&key1);
        
        // Hit
        cache.set_value(key1.clone(), 42.0);
        cache.get_value(&key1);
        
        // Another miss
        cache.get_value(&key2);
        
        assert!(cache.value_hit_rate() > 0.0);
        assert!(cache.value_hit_rate() < 1.0);
    }
}