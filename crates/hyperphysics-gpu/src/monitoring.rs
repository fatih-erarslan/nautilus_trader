//! Performance monitoring and profiling for GPU operations
//!
//! Tracks execution times, throughput, memory usage, and GPU utilization.

use std::time::{Duration, Instant};
use std::collections::VecDeque;

/// Performance metrics for a single operation
#[derive(Debug, Clone)]
pub struct OperationMetrics {
    /// Operation name
    pub name: String,
    /// Execution duration
    pub duration: Duration,
    /// Number of elements processed
    pub element_count: usize,
    /// Timestamp when operation completed
    pub timestamp: Instant,
}

impl OperationMetrics {
    /// Calculate throughput in elements/second
    pub fn throughput(&self) -> f64 {
        if self.duration.as_secs_f64() > 0.0 {
            self.element_count as f64 / self.duration.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Throughput in millions of elements per second
    pub fn throughput_millions(&self) -> f64 {
        self.throughput() / 1_000_000.0
    }

    /// Average time per element in microseconds
    pub fn time_per_element_us(&self) -> f64 {
        if self.element_count > 0 {
            self.duration.as_micros() as f64 / self.element_count as f64
        } else {
            0.0
        }
    }
}

/// Performance monitor that tracks GPU operations
pub struct PerformanceMonitor {
    /// Operation history (last N operations)
    history: VecDeque<OperationMetrics>,
    /// Maximum history size
    max_history: usize,
    /// Total operations tracked
    total_operations: usize,
    /// Start time for session tracking
    session_start: Instant,
}

impl PerformanceMonitor {
    /// Create new performance monitor
    ///
    /// # Arguments
    /// * `max_history` - Maximum number of operations to keep in history
    pub fn new(max_history: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_history),
            max_history,
            total_operations: 0,
            session_start: Instant::now(),
        }
    }

    /// Record a new operation
    pub fn record(&mut self, metrics: OperationMetrics) {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(metrics);
        self.total_operations += 1;
    }

    /// Record operation by name with timing
    pub fn record_timed(
        &mut self,
        name: impl Into<String>,
        duration: Duration,
        element_count: usize,
    ) {
        let metrics = OperationMetrics {
            name: name.into(),
            duration,
            element_count,
            timestamp: Instant::now(),
        };
        self.record(metrics);
    }

    /// Get statistics for specific operation type
    pub fn operation_stats(&self, operation_name: &str) -> Option<OperationStats> {
        let ops: Vec<&OperationMetrics> = self.history
            .iter()
            .filter(|m| m.name == operation_name)
            .collect();

        if ops.is_empty() {
            return None;
        }

        let durations: Vec<Duration> = ops.iter().map(|m| m.duration).collect();
        let throughputs: Vec<f64> = ops.iter().map(|m| m.throughput()).collect();

        Some(OperationStats {
            operation_name: operation_name.to_string(),
            count: ops.len(),
            avg_duration: avg_duration(&durations),
            min_duration: *durations.iter().min().unwrap(),
            max_duration: *durations.iter().max().unwrap(),
            avg_throughput: avg_f64(&throughputs),
            total_elements: ops.iter().map(|m| m.element_count).sum(),
        })
    }

    /// Get overall statistics across all operations
    pub fn overall_stats(&self) -> OverallStats {
        let total_duration: Duration = self.history.iter().map(|m| m.duration).sum();
        let total_elements: usize = self.history.iter().map(|m| m.element_count).sum();
        let session_duration = self.session_start.elapsed();

        OverallStats {
            total_operations: self.total_operations,
            recent_operations: self.history.len(),
            total_compute_time: total_duration,
            total_elements_processed: total_elements,
            session_duration,
            avg_ops_per_second: self.total_operations as f64 / session_duration.as_secs_f64(),
        }
    }

    /// Get recent throughput trend (last N operations)
    pub fn throughput_trend(&self, n: usize) -> Vec<f64> {
        self.history
            .iter()
            .rev()
            .take(n)
            .map(|m| m.throughput_millions())
            .collect()
    }

    /// Clear all history
    pub fn clear(&mut self) {
        self.history.clear();
        self.total_operations = 0;
        self.session_start = Instant::now();
    }

    /// Generate human-readable report
    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== GPU Performance Report ===\n\n");

        // Overall statistics
        let overall = self.overall_stats();
        report.push_str(&format!("Session Duration: {:.2}s\n", overall.session_duration.as_secs_f64()));
        report.push_str(&format!("Total Operations: {}\n", overall.total_operations));
        report.push_str(&format!("Avg Operations/sec: {:.2}\n", overall.avg_ops_per_second));
        report.push_str(&format!("Total Compute Time: {:.2}s\n", overall.total_compute_time.as_secs_f64()));
        report.push_str(&format!("Total Elements: {}\n\n", overall.total_elements_processed));

        // Per-operation statistics
        let mut operation_names: Vec<String> = self.history
            .iter()
            .map(|m| m.name.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        operation_names.sort();

        for op_name in operation_names {
            if let Some(stats) = self.operation_stats(&op_name) {
                report.push_str(&format!("\n--- {} ---\n", op_name));
                report.push_str(&format!("Count: {}\n", stats.count));
                report.push_str(&format!("Avg Duration: {:.2}ms\n", stats.avg_duration.as_secs_f64() * 1000.0));
                report.push_str(&format!("Min/Max: {:.2}ms / {:.2}ms\n",
                    stats.min_duration.as_secs_f64() * 1000.0,
                    stats.max_duration.as_secs_f64() * 1000.0));
                report.push_str(&format!("Avg Throughput: {:.2} M elements/sec\n",
                    stats.avg_throughput / 1_000_000.0));
                report.push_str(&format!("Total Elements: {}\n", stats.total_elements));
            }
        }

        report
    }
}

/// Statistics for a specific operation type
#[derive(Debug, Clone)]
pub struct OperationStats {
    pub operation_name: String,
    pub count: usize,
    pub avg_duration: Duration,
    pub min_duration: Duration,
    pub max_duration: Duration,
    pub avg_throughput: f64,
    pub total_elements: usize,
}

/// Overall session statistics
#[derive(Debug, Clone)]
pub struct OverallStats {
    pub total_operations: usize,
    pub recent_operations: usize,
    pub total_compute_time: Duration,
    pub total_elements_processed: usize,
    pub session_duration: Duration,
    pub avg_ops_per_second: f64,
}

/// Helper: Average of durations
fn avg_duration(durations: &[Duration]) -> Duration {
    if durations.is_empty() {
        return Duration::ZERO;
    }
    let total: Duration = durations.iter().copied().sum();
    total / durations.len() as u32
}

/// Helper: Average of f64 values
fn avg_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Scoped timer that automatically records metrics when dropped
pub struct ScopedTimer<'a> {
    monitor: &'a mut PerformanceMonitor,
    operation_name: String,
    element_count: usize,
    start: Instant,
}

impl<'a> ScopedTimer<'a> {
    /// Create new scoped timer
    pub fn new(
        monitor: &'a mut PerformanceMonitor,
        operation_name: impl Into<String>,
        element_count: usize,
    ) -> Self {
        Self {
            monitor,
            operation_name: operation_name.into(),
            element_count,
            start: Instant::now(),
        }
    }
}

impl<'a> Drop for ScopedTimer<'a> {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        self.monitor.record_timed(
            self.operation_name.clone(),
            duration,
            self.element_count,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_operation_metrics() {
        let metrics = OperationMetrics {
            name: "test_op".to_string(),
            duration: Duration::from_millis(100),
            element_count: 10_000,
            timestamp: Instant::now(),
        };

        assert_eq!(metrics.throughput(), 100_000.0); // 10k / 0.1s = 100k/s
        assert_eq!(metrics.throughput_millions(), 0.1); // 0.1 M/s
        assert_eq!(metrics.time_per_element_us(), 10.0); // 100ms / 10k = 10µs/element
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::new(10);

        // Record some operations
        monitor.record_timed("pbit_update", Duration::from_millis(50), 1000);
        monitor.record_timed("pbit_update", Duration::from_millis(45), 1000);
        monitor.record_timed("energy", Duration::from_millis(20), 1000);

        assert_eq!(monitor.total_operations, 3);

        // Check stats for pbit_update
        let stats = monitor.operation_stats("pbit_update").unwrap();
        assert_eq!(stats.count, 2);
        assert_eq!(stats.total_elements, 2000);

        // Check overall stats
        let overall = monitor.overall_stats();
        assert_eq!(overall.total_operations, 3);
        assert_eq!(overall.total_elements_processed, 3000);
    }

    #[test]
    fn test_history_limit() {
        let mut monitor = PerformanceMonitor::new(3);

        // Add 5 operations
        for i in 0..5 {
            monitor.record_timed(
                format!("op_{}", i),
                Duration::from_millis(10),
                100,
            );
        }

        // Should only keep last 3
        assert_eq!(monitor.history.len(), 3);
        assert_eq!(monitor.total_operations, 5);
    }

    #[test]
    fn test_scoped_timer() {
        let mut monitor = PerformanceMonitor::new(10);

        {
            let _timer = ScopedTimer::new(&mut monitor, "test_scope", 1000);
            thread::sleep(Duration::from_millis(10));
            // Timer records when dropped here
        }

        assert_eq!(monitor.total_operations, 1);
        let stats = monitor.operation_stats("test_scope").unwrap();
        assert!(stats.avg_duration >= Duration::from_millis(10));
    }

    #[test]
    fn test_throughput_trend() {
        let mut monitor = PerformanceMonitor::new(10);

        for i in 1..=5 {
            monitor.record_timed(
                "varying_op",
                Duration::from_millis(i * 10),
                10_000,
            );
        }

        let trend = monitor.throughput_trend(3);
        assert_eq!(trend.len(), 3);
        // Most recent should be slowest (longest duration)
        assert!(trend[0] < trend[2]);
    }

    #[test]
    fn test_report_generation() {
        let mut monitor = PerformanceMonitor::new(10);

        monitor.record_timed("pbit_update", Duration::from_millis(100), 10_000);
        monitor.record_timed("energy", Duration::from_millis(50), 10_000);

        let report = monitor.report();
        assert!(report.contains("GPU Performance Report"));
        assert!(report.contains("pbit_update"));
        assert!(report.contains("energy"));
        assert!(report.contains("Total Operations: 2"));
    }

    #[test]
    fn test_clear() {
        let mut monitor = PerformanceMonitor::new(10);

        monitor.record_timed("op1", Duration::from_millis(10), 100);
        monitor.record_timed("op2", Duration::from_millis(20), 200);

        assert_eq!(monitor.total_operations, 2);

        monitor.clear();

        assert_eq!(monitor.total_operations, 0);
        assert_eq!(monitor.history.len(), 0);
    }
}
