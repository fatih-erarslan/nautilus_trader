//! Real-time Performance Monitoring for ATS-Core

use crate::{
    config::AtsCpConfig,
    error::{Result},
    types::{PerformanceStats, TimeNs},
};
use instant::Instant;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Performance monitoring system
pub struct PerformanceMonitor {
    /// Configuration
    #[allow(dead_code)]
    config: AtsCpConfig,
    
    /// Performance statistics
    stats: Arc<Mutex<PerformanceStats>>,
    
    /// Operation timers
    #[allow(dead_code)]
    timers: Arc<Mutex<HashMap<String, Vec<TimeNs>>>>,
}

impl PerformanceMonitor {
    /// Creates a new performance monitor
    pub fn new(config: &AtsCpConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            stats: Arc::new(Mutex::new(PerformanceStats::new())),
            timers: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Starts a timer for an operation
    pub fn start_timer(&self, operation: &str) -> Timer {
        Timer::new(operation.to_string(), self.stats.clone())
    }

    /// Gets current performance statistics  
    pub fn get_stats(&self) -> crate::types::PerformanceStats {
        self.stats.lock().unwrap().clone()
    }
}

/// Timer for measuring operation latency
pub struct Timer {
    #[allow(dead_code)]
    operation: String,
    start_time: Instant,
    stats: Arc<Mutex<PerformanceStats>>,
}

impl Timer {
    fn new(operation: String, stats: Arc<Mutex<PerformanceStats>>) -> Self {
        Self {
            operation,
            start_time: Instant::now(),
            stats,
        }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        let elapsed_ns = self.start_time.elapsed().as_nanos() as u64;
        let mut stats = self.stats.lock().unwrap();
        stats.update(elapsed_ns);
    }
}