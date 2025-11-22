//! Metrics collection and caching for Prospect Theory
use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use crate::performance::PerformanceMetrics;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProspectTheoryMetrics {
    pub value_function_calls: u64,
    pub probability_weighting_calls: u64,
    pub decision_count: u64,
    pub cache_hit_rate: f64,
    pub avg_computation_time_ns: u64,
    pub quantum_enhancement_usage: f64,
    pub behavioral_factor_impact: f64,
    pub loss_aversion_activations: u64,
    pub framing_effect_triggers: u64,
    pub mental_accounting_adjustments: u64,
}

impl Default for ProspectTheoryMetrics {
    fn default() -> Self {
        Self {
            value_function_calls: 0,
            probability_weighting_calls: 0,
            decision_count: 0,
            cache_hit_rate: 0.0,
            avg_computation_time_ns: 0,
            quantum_enhancement_usage: 0.0,
            behavioral_factor_impact: 0.0,
            loss_aversion_activations: 0,
            framing_effect_triggers: 0,
            mental_accounting_adjustments: 0,
        }
    }
}

#[derive(Debug)]
pub struct ProspectTheoryCache {
    cache: HashMap<String, CacheEntry>,
    max_size: usize,
    hits: u64,
    misses: u64,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    value: f64,
    timestamp: Instant,
    access_count: u32,
}

impl ProspectTheoryCache {
    pub fn new(size: usize) -> Self {
        Self { 
            cache: HashMap::with_capacity(size),
            max_size: size,
            hits: 0,
            misses: 0,
        }
    }
    
    pub fn get(&mut self, key: &str) -> Option<f64> {
        if let Some(entry) = self.cache.get_mut(key) {
            entry.access_count += 1;
            entry.timestamp = Instant::now();
            self.hits += 1;
            Some(entry.value)
        } else {
            self.misses += 1;
            None
        }
    }
    
    pub fn insert(&mut self, key: String, value: f64) {
        if self.cache.len() >= self.max_size {
            self.evict_lru();
        }
        
        let entry = CacheEntry {
            value,
            timestamp: Instant::now(),
            access_count: 1,
        };
        
        self.cache.insert(key, entry);
    }
    
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
    
    pub fn clear(&mut self) {
        self.cache.clear();
        self.hits = 0;
        self.misses = 0;
    }
    
    fn evict_lru(&mut self) {
        if let Some((oldest_key, _)) = self.cache.iter()
            .min_by_key(|(_, entry)| entry.timestamp)
            .map(|(k, v)| (k.clone(), v.clone())) {
            self.cache.remove(&oldest_key);
        }
    }
}

#[derive(Debug)]
pub struct MetricsCollector {
    metrics: ProspectTheoryMetrics,
    cache: ProspectTheoryCache,
    start_time: Instant,
}

impl MetricsCollector {
    pub fn new(cache_size: usize) -> Self {
        Self {
            metrics: ProspectTheoryMetrics::default(),
            cache: ProspectTheoryCache::new(cache_size),
            start_time: Instant::now(),
        }
    }
    
    pub fn record_value_function_call(&mut self, duration: Duration) {
        self.metrics.value_function_calls += 1;
        self.update_avg_time(duration);
    }
    
    pub fn record_probability_weighting_call(&mut self, duration: Duration) {
        self.metrics.probability_weighting_calls += 1;
        self.update_avg_time(duration);
    }
    
    pub fn record_decision(&mut self) {
        self.metrics.decision_count += 1;
    }
    
    pub fn record_behavioral_impact(&mut self, impact: f64) {
        let count = self.metrics.decision_count.max(1) as f64;
        self.metrics.behavioral_factor_impact = 
            (self.metrics.behavioral_factor_impact * (count - 1.0) + impact) / count;
    }
    
    pub fn record_quantum_enhancement(&mut self, enhancement: f64) {
        let count = self.metrics.decision_count.max(1) as f64;
        self.metrics.quantum_enhancement_usage = 
            (self.metrics.quantum_enhancement_usage * (count - 1.0) + enhancement) / count;
    }
    
    pub fn record_loss_aversion(&mut self) {
        self.metrics.loss_aversion_activations += 1;
    }
    
    pub fn record_framing_effect(&mut self) {
        self.metrics.framing_effect_triggers += 1;
    }
    
    pub fn record_mental_accounting(&mut self) {
        self.metrics.mental_accounting_adjustments += 1;
    }
    
    pub fn get_metrics(&mut self) -> ProspectTheoryMetrics {
        self.metrics.cache_hit_rate = self.cache.hit_rate();
        self.metrics.clone()
    }
    
    pub fn get_cache_mut(&mut self) -> &mut ProspectTheoryCache {
        &mut self.cache
    }
    
    pub fn reset(&mut self) {
        self.metrics = ProspectTheoryMetrics::default();
        self.cache.clear();
        self.start_time = Instant::now();
    }
    
    fn update_avg_time(&mut self, duration: Duration) {
        let duration_ns = duration.as_nanos() as u64;
        let total_calls = self.metrics.value_function_calls + self.metrics.probability_weighting_calls;
        
        if total_calls > 0 {
            self.metrics.avg_computation_time_ns = 
                (self.metrics.avg_computation_time_ns * (total_calls - 1) + duration_ns) / total_calls;
        } else {
            self.metrics.avg_computation_time_ns = duration_ns;
        }
    }
}