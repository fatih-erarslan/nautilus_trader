//! Performance monitoring and optimization

use crate::core::{QercError, QercResult, PerformanceMetrics};
use std::time::{Duration, Instant};

/// Performance monitor
#[derive(Debug, Clone)]
pub struct PerformanceMonitor {
    /// Metrics
    pub metrics: PerformanceMetrics,
    /// Start time
    pub start_time: Instant,
    /// Enabled flag
    pub enabled: bool,
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub async fn new() -> QercResult<Self> {
        Ok(Self {
            metrics: PerformanceMetrics::default(),
            start_time: Instant::now(),
            enabled: true,
        })
    }
    
    /// Create disabled performance monitor
    pub async fn disabled() -> QercResult<Self> {
        Ok(Self {
            metrics: PerformanceMetrics::default(),
            start_time: Instant::now(),
            enabled: false,
        })
    }
    
    /// Record error detection
    pub fn record_error_detection(&mut self, duration: Duration, has_error: bool) {
        if !self.enabled {
            return;
        }
        
        self.metrics.latency_ms = duration.as_millis() as f64;
        if has_error {
            self.metrics.error_detection_rate += 0.1;
        }
    }
    
    /// Record error correction
    pub fn record_error_correction(&mut self, duration: Duration, success: bool) {
        if !self.enabled {
            return;
        }
        
        self.metrics.latency_ms = duration.as_millis() as f64;
        if success {
            self.metrics.error_correction_rate += 0.1;
        }
    }
    
    /// Get current metrics
    pub fn get_metrics(&self) -> PerformanceMetrics {
        self.metrics.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::new().await.unwrap();
        monitor.record_error_detection(Duration::from_millis(1), false);
        let metrics = monitor.get_metrics();
        assert_eq!(metrics.latency_ms, 1.0);
    }
}