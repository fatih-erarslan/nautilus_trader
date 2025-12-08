//! Performance metrics and monitoring for Black Swan detection
//!
//! This module provides comprehensive performance monitoring, benchmarking,
//! and optimization capabilities for the Black Swan detector.

use crate::error::*;
use crate::types::*;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// Real-time performance metrics tracker
#[derive(Debug)]
pub struct MetricsCollector {
    /// Total number of detections performed
    detection_count: AtomicUsize,
    
    /// Total latency accumulator in nanoseconds
    total_latency_ns: AtomicU64,
    
    /// Peak latency in nanoseconds
    peak_latency_ns: AtomicU64,
    
    /// Memory usage tracker
    memory_usage: AtomicUsize,
    
    /// SIMD operations counter
    simd_ops: AtomicUsize,
    
    /// GPU utilization tracker
    gpu_utilization: std::sync::Mutex<f32>,
    
    /// Cache metrics
    cache_hits: AtomicUsize,
    cache_misses: AtomicUsize,
    
    /// Latency histogram
    latency_histogram: std::sync::Mutex<VecDeque<u64>>,
    
    /// Throughput tracker
    throughput_tracker: std::sync::Mutex<VecDeque<(Instant, usize)>>,
    
    /// Error tracker
    error_counts: std::sync::Mutex<HashMap<String, usize>>,
    
    /// Accuracy metrics
    accuracy_tracker: std::sync::Mutex<AccuracyTracker>,
    
    /// Resource utilization
    resource_monitor: std::sync::Mutex<ResourceMonitor>,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            detection_count: AtomicUsize::new(0),
            total_latency_ns: AtomicU64::new(0),
            peak_latency_ns: AtomicU64::new(0),
            memory_usage: AtomicUsize::new(0),
            simd_ops: AtomicUsize::new(0),
            gpu_utilization: std::sync::Mutex::new(0.0),
            cache_hits: AtomicUsize::new(0),
            cache_misses: AtomicUsize::new(0),
            latency_histogram: std::sync::Mutex::new(VecDeque::with_capacity(10000)),
            throughput_tracker: std::sync::Mutex::new(VecDeque::with_capacity(1000)),
            error_counts: std::sync::Mutex::new(HashMap::new()),
            accuracy_tracker: std::sync::Mutex::new(AccuracyTracker::new()),
            resource_monitor: std::sync::Mutex::new(ResourceMonitor::new()),
        }
    }
    
    /// Record a detection operation
    pub fn record_detection(&self, latency_ns: u64, memory_bytes: usize, simd_ops: usize) {
        self.detection_count.fetch_add(1, Ordering::Relaxed);
        self.total_latency_ns.fetch_add(latency_ns, Ordering::Relaxed);
        self.memory_usage.store(memory_bytes, Ordering::Relaxed);
        self.simd_ops.fetch_add(simd_ops, Ordering::Relaxed);
        
        // Update peak latency
        let mut current_peak = self.peak_latency_ns.load(Ordering::Relaxed);
        while latency_ns > current_peak {
            match self.peak_latency_ns.compare_exchange_weak(
                current_peak, latency_ns, Ordering::Relaxed, Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(x) => current_peak = x,
            }
        }
        
        // Update histogram
        {
            let mut histogram = self.latency_histogram.lock().unwrap();
            if histogram.len() >= 10000 {
                histogram.pop_front();
            }
            histogram.push_back(latency_ns);
        }
        
        // Update throughput
        {
            let mut throughput = self.throughput_tracker.lock().unwrap();
            if throughput.len() >= 1000 {
                throughput.pop_front();
            }
            throughput.push_back((Instant::now(), 1));
        }
    }
    
    /// Record cache hit
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record cache miss
    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record GPU utilization
    pub fn record_gpu_utilization(&self, utilization: f32) {
        let mut gpu_util = self.gpu_utilization.lock().unwrap();
        *gpu_util = utilization;
    }
    
    /// Record an error
    pub fn record_error(&self, error_type: &str) {
        let mut errors = self.error_counts.lock().unwrap();
        *errors.entry(error_type.to_string()).or_insert(0) += 1;
    }
    
    /// Record accuracy metrics
    pub fn record_accuracy(&self, predicted: f64, actual: f64) {
        let mut accuracy = self.accuracy_tracker.lock().unwrap();
        accuracy.record_prediction(predicted, actual);
    }
    
    /// Get current metrics snapshot
    pub fn get_metrics(&self) -> MetricsSnapshot {
        let detection_count = self.detection_count.load(Ordering::Relaxed);
        let total_latency = self.total_latency_ns.load(Ordering::Relaxed);
        let peak_latency = self.peak_latency_ns.load(Ordering::Relaxed);
        let memory_usage = self.memory_usage.load(Ordering::Relaxed);
        let simd_ops = self.simd_ops.load(Ordering::Relaxed);
        let gpu_utilization = *self.gpu_utilization.lock().unwrap();
        let cache_hits = self.cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.cache_misses.load(Ordering::Relaxed);
        
        let avg_latency = if detection_count > 0 {
            total_latency / detection_count as u64
        } else {
            0
        };
        
        let cache_hit_ratio = if cache_hits + cache_misses > 0 {
            cache_hits as f32 / (cache_hits + cache_misses) as f32
        } else {
            0.0
        };
        
        // Calculate throughput (detections per second)
        let throughput = {
            let tracker = self.throughput_tracker.lock().unwrap();
            if tracker.len() < 2 {
                0.0
            } else {
                let first = tracker.front().unwrap();
                let last = tracker.back().unwrap();
                let duration = last.0.duration_since(first.0).as_secs_f64();
                if duration > 0.0 {
                    tracker.len() as f64 / duration
                } else {
                    0.0
                }
            }
        };
        
        // Get latency percentiles
        let latency_percentiles = {
            let mut histogram = self.latency_histogram.lock().unwrap();
            if histogram.is_empty() {
                LatencyPercentiles::default()
            } else {
                let mut sorted: Vec<u64> = histogram.iter().cloned().collect();
                sorted.sort_unstable();
                
                let len = sorted.len();
                LatencyPercentiles {
                    p50: sorted[len / 2],
                    p95: sorted[len * 95 / 100],
                    p99: sorted[len * 99 / 100],
                    p999: sorted[len * 999 / 1000],
                }
            }
        };
        
        // Get error summary
        let error_summary = {
            let errors = self.error_counts.lock().unwrap();
            errors.clone()
        };
        
        // Get accuracy metrics
        let accuracy_metrics = {
            let accuracy = self.accuracy_tracker.lock().unwrap();
            accuracy.get_metrics()
        };
        
        // Get resource utilization
        let resource_utilization = {
            let monitor = self.resource_monitor.lock().unwrap();
            monitor.get_current_usage()
        };
        
        MetricsSnapshot {
            detection_count,
            avg_latency_ns: avg_latency,
            peak_latency_ns: peak_latency,
            throughput_per_sec: throughput,
            memory_usage_bytes: memory_usage,
            simd_operations: simd_ops,
            gpu_utilization,
            cache_hit_ratio,
            latency_percentiles,
            error_summary,
            accuracy_metrics,
            resource_utilization,
            timestamp: Instant::now(),
        }
    }
    
    /// Reset all metrics
    pub fn reset(&self) {
        self.detection_count.store(0, Ordering::Relaxed);
        self.total_latency_ns.store(0, Ordering::Relaxed);
        self.peak_latency_ns.store(0, Ordering::Relaxed);
        self.memory_usage.store(0, Ordering::Relaxed);
        self.simd_ops.store(0, Ordering::Relaxed);
        *self.gpu_utilization.lock().unwrap() = 0.0;
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
        self.latency_histogram.lock().unwrap().clear();
        self.throughput_tracker.lock().unwrap().clear();
        self.error_counts.lock().unwrap().clear();
        self.accuracy_tracker.lock().unwrap().reset();
        self.resource_monitor.lock().unwrap().reset();
    }
    
    /// Check if performance targets are met
    pub fn check_performance_targets(&self, targets: &PerformanceTargets) -> PerformanceReport {
        let metrics = self.get_metrics();
        
        let latency_target_met = metrics.avg_latency_ns <= targets.max_latency_ns;
        let throughput_target_met = metrics.throughput_per_sec >= targets.min_throughput_per_sec;
        let memory_target_met = metrics.memory_usage_bytes <= targets.max_memory_bytes;
        let cache_target_met = metrics.cache_hit_ratio >= targets.min_cache_hit_ratio;
        let accuracy_target_met = metrics.accuracy_metrics.mean_absolute_error <= targets.max_mae;
        
        PerformanceReport {
            targets_met: latency_target_met && throughput_target_met && memory_target_met && cache_target_met && accuracy_target_met,
            latency_target_met,
            throughput_target_met,
            memory_target_met,
            cache_target_met,
            accuracy_target_met,
            metrics,
            recommendations: generate_recommendations(&metrics, targets),
        }
    }
}

/// Snapshot of current metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub detection_count: usize,
    pub avg_latency_ns: u64,
    pub peak_latency_ns: u64,
    pub throughput_per_sec: f64,
    pub memory_usage_bytes: usize,
    pub simd_operations: usize,
    pub gpu_utilization: f32,
    pub cache_hit_ratio: f32,
    pub latency_percentiles: LatencyPercentiles,
    pub error_summary: HashMap<String, usize>,
    pub accuracy_metrics: AccuracyMetrics,
    pub resource_utilization: ResourceUtilization,
    pub timestamp: Instant,
}

/// Latency percentiles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyPercentiles {
    pub p50: u64,
    pub p95: u64,
    pub p99: u64,
    pub p999: u64,
}

impl Default for LatencyPercentiles {
    fn default() -> Self {
        Self {
            p50: 0,
            p95: 0,
            p99: 0,
            p999: 0,
        }
    }
}

/// Accuracy tracking
#[derive(Debug)]
pub struct AccuracyTracker {
    predictions: VecDeque<(f64, f64)>, // (predicted, actual)
    max_history: usize,
}

impl AccuracyTracker {
    pub fn new() -> Self {
        Self {
            predictions: VecDeque::with_capacity(10000),
            max_history: 10000,
        }
    }
    
    pub fn record_prediction(&mut self, predicted: f64, actual: f64) {
        if self.predictions.len() >= self.max_history {
            self.predictions.pop_front();
        }
        self.predictions.push_back((predicted, actual));
    }
    
    pub fn get_metrics(&self) -> AccuracyMetrics {
        if self.predictions.is_empty() {
            return AccuracyMetrics::default();
        }
        
        let mut total_absolute_error = 0.0;
        let mut total_squared_error = 0.0;
        let mut max_error = 0.0;
        
        for &(predicted, actual) in &self.predictions {
            let absolute_error = (predicted - actual).abs();
            let squared_error = (predicted - actual).powi(2);
            
            total_absolute_error += absolute_error;
            total_squared_error += squared_error;
            max_error = max_error.max(absolute_error);
        }
        
        let count = self.predictions.len() as f64;
        let mean_absolute_error = total_absolute_error / count;
        let mean_squared_error = total_squared_error / count;
        let root_mean_squared_error = mean_squared_error.sqrt();
        
        AccuracyMetrics {
            mean_absolute_error,
            mean_squared_error,
            root_mean_squared_error,
            max_error,
            sample_count: self.predictions.len(),
        }
    }
    
    pub fn reset(&mut self) {
        self.predictions.clear();
    }
}

/// Accuracy metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    pub mean_absolute_error: f64,
    pub mean_squared_error: f64,
    pub root_mean_squared_error: f64,
    pub max_error: f64,
    pub sample_count: usize,
}

impl Default for AccuracyMetrics {
    fn default() -> Self {
        Self {
            mean_absolute_error: 0.0,
            mean_squared_error: 0.0,
            root_mean_squared_error: 0.0,
            max_error: 0.0,
            sample_count: 0,
        }
    }
}

/// Resource utilization monitor
#[derive(Debug)]
pub struct ResourceMonitor {
    cpu_usage: f32,
    memory_usage: usize,
    disk_io: u64,
    network_io: u64,
    last_update: Instant,
}

impl ResourceMonitor {
    pub fn new() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0,
            disk_io: 0,
            network_io: 0,
            last_update: Instant::now(),
        }
    }
    
    pub fn update(&mut self) {
        // Platform-specific resource monitoring would go here
        // For now, we'll use placeholder values
        self.cpu_usage = 0.0;
        self.memory_usage = 0;
        self.disk_io = 0;
        self.network_io = 0;
        self.last_update = Instant::now();
    }
    
    pub fn get_current_usage(&self) -> ResourceUtilization {
        ResourceUtilization {
            cpu_usage: self.cpu_usage,
            memory_usage: self.memory_usage,
            disk_io: self.disk_io,
            network_io: self.network_io,
        }
    }
    
    pub fn reset(&mut self) {
        self.cpu_usage = 0.0;
        self.memory_usage = 0;
        self.disk_io = 0;
        self.network_io = 0;
        self.last_update = Instant::now();
    }
}

/// Current resource utilization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    pub cpu_usage: f32,
    pub memory_usage: usize,
    pub disk_io: u64,
    pub network_io: u64,
}

/// Performance targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    pub max_latency_ns: u64,
    pub min_throughput_per_sec: f64,
    pub max_memory_bytes: usize,
    pub min_cache_hit_ratio: f32,
    pub max_mae: f64,
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            max_latency_ns: 1_000_000, // 1ms
            min_throughput_per_sec: 1000.0,
            max_memory_bytes: 64 * 1024 * 1024, // 64MB
            min_cache_hit_ratio: 0.8,
            max_mae: 0.1,
        }
    }
}

/// Performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub targets_met: bool,
    pub latency_target_met: bool,
    pub throughput_target_met: bool,
    pub memory_target_met: bool,
    pub cache_target_met: bool,
    pub accuracy_target_met: bool,
    pub metrics: MetricsSnapshot,
    pub recommendations: Vec<String>,
}

/// Generate optimization recommendations
fn generate_recommendations(metrics: &MetricsSnapshot, targets: &PerformanceTargets) -> Vec<String> {
    let mut recommendations = Vec::new();
    
    if metrics.avg_latency_ns > targets.max_latency_ns {
        recommendations.push("Consider enabling more aggressive caching".to_string());
        recommendations.push("Optimize SIMD operations".to_string());
        recommendations.push("Reduce memory allocations".to_string());
    }
    
    if metrics.throughput_per_sec < targets.min_throughput_per_sec {
        recommendations.push("Enable parallel processing".to_string());
        recommendations.push("Optimize critical path algorithms".to_string());
    }
    
    if metrics.memory_usage_bytes > targets.max_memory_bytes {
        recommendations.push("Implement memory pooling".to_string());
        recommendations.push("Reduce window sizes".to_string());
    }
    
    if metrics.cache_hit_ratio < targets.min_cache_hit_ratio {
        recommendations.push("Increase cache size".to_string());
        recommendations.push("Improve cache eviction policy".to_string());
    }
    
    if metrics.accuracy_metrics.mean_absolute_error > targets.max_mae {
        recommendations.push("Retrain models with more data".to_string());
        recommendations.push("Adjust detection thresholds".to_string());
    }
    
    if metrics.gpu_utilization < 0.5 {
        recommendations.push("Enable GPU acceleration".to_string());
        recommendations.push("Optimize GPU memory usage".to_string());
    }
    
    recommendations
}

/// Benchmarking suite for performance testing
pub struct BenchmarkSuite {
    metrics_collector: MetricsCollector,
    test_cases: Vec<BenchmarkCase>,
}

impl BenchmarkSuite {
    pub fn new() -> Self {
        Self {
            metrics_collector: MetricsCollector::new(),
            test_cases: Vec::new(),
        }
    }
    
    pub fn add_test_case(&mut self, case: BenchmarkCase) {
        self.test_cases.push(case);
    }
    
    pub fn run_benchmarks(&mut self) -> BSResult<Vec<BenchmarkResult>> {
        let mut results = Vec::new();
        
        for test_case in &self.test_cases {
            let result = self.run_single_benchmark(test_case)?;
            results.push(result);
        }
        
        Ok(results)
    }
    
    fn run_single_benchmark(&mut self, test_case: &BenchmarkCase) -> BSResult<BenchmarkResult> {
        self.metrics_collector.reset();
        
        let start_time = Instant::now();
        
        // Run the benchmark iterations
        for _ in 0..test_case.iterations {
            let iter_start = Instant::now();
            
            // Simulate detection operation
            std::thread::sleep(Duration::from_nanos(test_case.expected_latency_ns));
            
            let iter_latency = iter_start.elapsed().as_nanos() as u64;
            self.metrics_collector.record_detection(iter_latency, test_case.memory_usage, test_case.simd_ops);
        }
        
        let total_duration = start_time.elapsed();
        let metrics = self.metrics_collector.get_metrics();
        
        Ok(BenchmarkResult {
            test_case: test_case.clone(),
            total_duration,
            metrics,
            success: metrics.avg_latency_ns <= test_case.expected_latency_ns,
        })
    }
}

/// Benchmark test case
#[derive(Debug, Clone)]
pub struct BenchmarkCase {
    pub name: String,
    pub iterations: usize,
    pub expected_latency_ns: u64,
    pub memory_usage: usize,
    pub simd_ops: usize,
}

/// Benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub test_case: BenchmarkCase,
    pub total_duration: Duration,
    pub metrics: MetricsSnapshot,
    pub success: bool,
}

/// Continuous performance monitoring
pub struct PerformanceMonitor {
    metrics_collector: MetricsCollector,
    targets: PerformanceTargets,
    monitoring_interval: Duration,
    last_report_time: Instant,
    alert_threshold: f64,
}

impl PerformanceMonitor {
    pub fn new(targets: PerformanceTargets, monitoring_interval: Duration) -> Self {
        Self {
            metrics_collector: MetricsCollector::new(),
            targets,
            monitoring_interval,
            last_report_time: Instant::now(),
            alert_threshold: 0.8, // Alert when 80% of targets are not met
        }
    }
    
    pub fn record_detection(&self, latency_ns: u64, memory_bytes: usize, simd_ops: usize) {
        self.metrics_collector.record_detection(latency_ns, memory_bytes, simd_ops);
    }
    
    pub fn check_performance(&mut self) -> Option<PerformanceReport> {
        if self.last_report_time.elapsed() >= self.monitoring_interval {
            let report = self.metrics_collector.check_performance_targets(&self.targets);
            self.last_report_time = Instant::now();
            
            if !report.targets_met {
                log::warn!("Performance targets not met: {:?}", report.recommendations);
            }
            
            Some(report)
        } else {
            None
        }
    }
    
    pub fn get_current_metrics(&self) -> MetricsSnapshot {
        self.metrics_collector.get_metrics()
    }
    
    pub fn reset(&self) {
        self.metrics_collector.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;
    
    #[test]
    fn test_metrics_collector() {
        let collector = MetricsCollector::new();
        
        // Record some metrics
        collector.record_detection(1000, 1024, 8);
        collector.record_detection(2000, 2048, 16);
        collector.record_cache_hit();
        collector.record_cache_miss();
        
        let metrics = collector.get_metrics();
        
        assert_eq!(metrics.detection_count, 2);
        assert_eq!(metrics.avg_latency_ns, 1500);
        assert_eq!(metrics.peak_latency_ns, 2000);
        assert_eq!(metrics.cache_hit_ratio, 0.5);
    }
    
    #[test]
    fn test_accuracy_tracker() {
        let mut tracker = AccuracyTracker::new();
        
        tracker.record_prediction(0.8, 0.9);
        tracker.record_prediction(0.7, 0.6);
        tracker.record_prediction(0.9, 0.8);
        
        let metrics = tracker.get_metrics();
        
        assert_eq!(metrics.sample_count, 3);
        assert!(metrics.mean_absolute_error > 0.0);
        assert!(metrics.root_mean_squared_error > 0.0);
    }
    
    #[test]
    fn test_performance_targets() {
        let collector = MetricsCollector::new();
        let targets = PerformanceTargets {
            max_latency_ns: 1000,
            min_throughput_per_sec: 10.0,
            max_memory_bytes: 1024,
            min_cache_hit_ratio: 0.8,
            max_mae: 0.1,
        };
        
        // Record metrics that exceed targets
        collector.record_detection(2000, 2048, 8);
        
        let report = collector.check_performance_targets(&targets);
        
        assert!(!report.targets_met);
        assert!(!report.latency_target_met);
        assert!(!report.memory_target_met);
        assert!(!report.recommendations.is_empty());
    }
    
    #[test]
    fn test_benchmark_suite() {
        let mut suite = BenchmarkSuite::new();
        
        suite.add_test_case(BenchmarkCase {
            name: "Low Latency Test".to_string(),
            iterations: 100,
            expected_latency_ns: 1000,
            memory_usage: 1024,
            simd_ops: 8,
        });
        
        let results = suite.run_benchmarks().unwrap();
        
        assert_eq!(results.len(), 1);
        assert!(results[0].metrics.detection_count > 0);
    }
    
    #[test]
    fn test_performance_monitor() {
        let targets = PerformanceTargets::default();
        let mut monitor = PerformanceMonitor::new(targets, Duration::from_millis(100));
        
        monitor.record_detection(1000, 1024, 8);
        
        // Should not generate report yet
        assert!(monitor.check_performance().is_none());
        
        // Wait for monitoring interval
        thread::sleep(Duration::from_millis(150));
        
        // Should generate report now
        let report = monitor.check_performance();
        assert!(report.is_some());
    }
}