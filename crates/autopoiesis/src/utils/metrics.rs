use crate::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Comprehensive metrics collection and reporting system
#[derive(Debug, Clone)]
pub struct MetricsCollector {
    counters: Arc<RwLock<HashMap<String, u64>>>,
    gauges: Arc<RwLock<HashMap<String, f64>>>,
    tags: HashMap<String, String>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            counters: Arc::new(RwLock::new(HashMap::new())),
            gauges: Arc::new(RwLock::new(HashMap::new())),
            tags: HashMap::new(),
        }
    }
    
    pub fn with_tags(mut self, tags: HashMap<String, String>) -> Self {
        self.tags = tags;
        self
    }
    
    /// Increment a counter metric
    pub fn increment_counter(&self, name: &str, value: u64) {
        let mut counters = self.counters.write().unwrap();
        *counters.entry(name.to_string()).or_insert(0) += value;
    }
    
    /// Set a gauge metric
    pub fn set_gauge(&self, name: &str, value: f64) {
        let mut gauges = self.gauges.write().unwrap();
        gauges.insert(name.to_string(), value);
    }
    
    /// Get counter value
    pub fn get_counter(&self, name: &str) -> u64 {
        self.counters.read().unwrap().get(name).copied().unwrap_or(0)
    }
    
    /// Get gauge value
    pub fn get_gauge(&self, name: &str) -> Option<f64> {
        self.gauges.read().unwrap().get(name).copied()
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}