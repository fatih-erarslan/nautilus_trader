//! Performance monitoring for whale defense operations
//! 
//! Ultra-low overhead performance tracking and metrics collection.

use crate::{
    error::{WhaleDefenseError, Result},
    timing::{Timestamp, TimingStats},
    config::*,
    AtomicU64, AtomicBool, Ordering,
};
use cache_padded::CachePadded;
use serde::{Deserialize, Serialize};
use core::sync::atomic::compiler_fence;

/// Performance monitor for whale defense operations
#[repr(C, align(64))]
pub struct PerformanceMonitor {
    /// Detection performance counters (cache-aligned)
    detection_metrics: CachePadded<DetectionMetrics>,
    
    /// Defense execution metrics (cache-aligned)
    defense_metrics: CachePadded<DefenseMetrics>,
    
    /// System performance metrics (cache-aligned)
    system_metrics: CachePadded<SystemMetrics>,
    
    /// Monitoring state
    is_monitoring: AtomicBool,
    
    /// Performance thresholds
    thresholds: PerformanceThresholds,
}

/// Detection performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C, align(64))]
struct DetectionMetrics {
    /// Total detections performed
    total_detections: AtomicU64,
    /// Failed detections
    failed_detections: AtomicU64,
    /// Total detection time (nanoseconds)
    total_detection_time_ns: AtomicU64,
    /// Minimum detection time (nanoseconds)
    min_detection_time_ns: AtomicU64,
    /// Maximum detection time (nanoseconds)
    max_detection_time_ns: AtomicU64,
    /// Threshold violations
    threshold_violations: AtomicU64,
}

/// Defense execution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C, align(64))]
struct DefenseMetrics {
    /// Total defenses executed
    total_defenses: AtomicU64,
    /// Failed defenses
    failed_defenses: AtomicU64,
    /// Total defense execution time (nanoseconds)
    total_defense_time_ns: AtomicU64,
    /// Minimum defense time (nanoseconds)
    min_defense_time_ns: AtomicU64,
    /// Maximum defense time (nanoseconds)
    max_defense_time_ns: AtomicU64,
    /// Strategy calculation time (nanoseconds)
    strategy_calc_time_ns: AtomicU64,
    /// Order generation time (nanoseconds)
    order_gen_time_ns: AtomicU64,
}

/// System performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C, align(64))]
struct SystemMetrics {
    /// Memory usage (bytes)
    memory_usage_bytes: AtomicU64,
    /// CPU cycles consumed
    cpu_cycles: AtomicU64,
    /// Cache misses
    cache_misses: AtomicU64,
    /// Lock-free operation failures
    lockfree_failures: AtomicU64,
    /// Queue overflows
    queue_overflows: AtomicU64,
    /// Queue underflows
    queue_underflows: AtomicU64,
}

/// Performance thresholds for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    /// Maximum detection latency (nanoseconds)
    pub max_detection_latency_ns: u64,
    /// Maximum defense execution latency (nanoseconds)
    pub max_defense_latency_ns: u64,
    /// Maximum memory usage (bytes)
    pub max_memory_usage_bytes: u64,
    /// Maximum acceptable failure rate (0.0-1.0)
    pub max_failure_rate: f64,
    /// Performance reporting interval (nanoseconds)
    pub reporting_interval_ns: u64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_detection_latency_ns: TARGET_DETECTION_LATENCY_NS,
            max_defense_latency_ns: TARGET_DEFENSE_EXECUTION_NS,
            max_memory_usage_bytes: 100 * 1024 * 1024, // 100MB
            max_failure_rate: 0.01, // 1%
            reporting_interval_ns: 1_000_000_000, // 1 second
        }
    }
}

/// Comprehensive performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    /// Detection metrics
    pub detection: DetectionSummary,
    /// Defense metrics
    pub defense: DefenseSummary,
    /// System metrics
    pub system: SystemSummary,
    /// Timestamp of metrics collection
    pub timestamp: u64,
}

/// Detection performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionSummary {
    /// Total detections
    pub total: u64,
    /// Failed detections
    pub failed: u64,
    /// Success rate (0.0-1.0)
    pub success_rate: f64,
    /// Average detection time (nanoseconds)
    pub avg_time_ns: f64,
    /// Minimum detection time (nanoseconds)
    pub min_time_ns: u64,
    /// Maximum detection time (nanoseconds)
    pub max_time_ns: u64,
    /// Threshold violations
    pub threshold_violations: u64,
}

/// Defense performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefenseSummary {
    /// Total defenses
    pub total: u64,
    /// Failed defenses
    pub failed: u64,
    /// Success rate (0.0-1.0)
    pub success_rate: f64,
    /// Average defense time (nanoseconds)
    pub avg_time_ns: f64,
    /// Minimum defense time (nanoseconds)
    pub min_time_ns: u64,
    /// Maximum defense time (nanoseconds)
    pub max_time_ns: u64,
    /// Average strategy calculation time (nanoseconds)
    pub avg_strategy_time_ns: f64,
    /// Average order generation time (nanoseconds)
    pub avg_order_gen_time_ns: f64,
}

/// System performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemSummary {
    /// Current memory usage (bytes)
    pub memory_usage_bytes: u64,
    /// Total CPU cycles consumed
    pub cpu_cycles: u64,
    /// Cache miss rate (0.0-1.0)
    pub cache_miss_rate: f64,
    /// Lock-free failure rate (0.0-1.0)
    pub lockfree_failure_rate: f64,
    /// Queue overflow count
    pub queue_overflows: u64,
    /// Queue underflow count
    pub queue_underflows: u64,
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub unsafe fn new() -> Result<Self> {
        Ok(Self {
            detection_metrics: CachePadded::new(DetectionMetrics::new()),
            defense_metrics: CachePadded::new(DefenseMetrics::new()),
            system_metrics: CachePadded::new(SystemMetrics::new()),
            is_monitoring: AtomicBool::new(false),
            thresholds: PerformanceThresholds::default(),
        })
    }
    
    /// Start performance monitoring
    pub fn start(&self) -> Result<()> {
        self.is_monitoring.store(true, Ordering::Release);
        Ok(())
    }
    
    /// Record detection performance
    /// 
    /// # Performance
    /// Target overhead: <10 nanoseconds
    #[inline(always)]
    pub fn record_detection(&self, latency_ns: u64, success: bool) {
        if likely(self.is_monitoring.load(Ordering::Relaxed)) {
            compiler_fence(Ordering::Acquire);
            
            self.detection_metrics.total_detections.fetch_add(1, Ordering::Relaxed);
            self.detection_metrics.total_detection_time_ns.fetch_add(latency_ns, Ordering::Relaxed);
            
            // Update min/max with relaxed ordering for speed
            self.update_min_max(
                &self.detection_metrics.min_detection_time_ns,
                &self.detection_metrics.max_detection_time_ns,
                latency_ns,
            );
            
            if !success {
                self.detection_metrics.failed_detections.fetch_add(1, Ordering::Relaxed);
            }
            
            // Check threshold violation
            if latency_ns > self.thresholds.max_detection_latency_ns {
                self.detection_metrics.threshold_violations.fetch_add(1, Ordering::Relaxed);
            }
            
            compiler_fence(Ordering::Release);
        }
    }
    
    /// Record defense performance
    /// 
    /// # Performance
    /// Target overhead: <10 nanoseconds
    #[inline(always)]
    pub fn record_defense(
        &self,
        latency_ns: u64,
        strategy_time_ns: u64,
        order_gen_time_ns: u64,
        success: bool,
    ) {
        if likely(self.is_monitoring.load(Ordering::Relaxed)) {
            compiler_fence(Ordering::Acquire);
            
            self.defense_metrics.total_defenses.fetch_add(1, Ordering::Relaxed);
            self.defense_metrics.total_defense_time_ns.fetch_add(latency_ns, Ordering::Relaxed);
            self.defense_metrics.strategy_calc_time_ns.fetch_add(strategy_time_ns, Ordering::Relaxed);
            self.defense_metrics.order_gen_time_ns.fetch_add(order_gen_time_ns, Ordering::Relaxed);
            
            self.update_min_max(
                &self.defense_metrics.min_defense_time_ns,
                &self.defense_metrics.max_defense_time_ns,
                latency_ns,
            );
            
            if !success {
                self.defense_metrics.failed_defenses.fetch_add(1, Ordering::Relaxed);
            }
            
            compiler_fence(Ordering::Release);
        }
    }
    
    /// Record system metrics
    #[inline(always)]
    pub fn record_system_metrics(
        &self,
        memory_usage: u64,
        cpu_cycles: u64,
        cache_misses: u64,
    ) {
        if likely(self.is_monitoring.load(Ordering::Relaxed)) {
            self.system_metrics.memory_usage_bytes.store(memory_usage, Ordering::Relaxed);
            self.system_metrics.cpu_cycles.fetch_add(cpu_cycles, Ordering::Relaxed);
            self.system_metrics.cache_misses.fetch_add(cache_misses, Ordering::Relaxed);
        }
    }
    
    /// Record lock-free operation failure
    #[inline(always)]
    pub fn record_lockfree_failure(&self) {
        if likely(self.is_monitoring.load(Ordering::Relaxed)) {
            self.system_metrics.lockfree_failures.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    /// Record queue overflow
    #[inline(always)]
    pub fn record_queue_overflow(&self) {
        if likely(self.is_monitoring.load(Ordering::Relaxed)) {
            self.system_metrics.queue_overflows.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    /// Record queue underflow
    #[inline(always)]
    pub fn record_queue_underflow(&self) {
        if likely(self.is_monitoring.load(Ordering::Relaxed)) {
            self.system_metrics.queue_underflows.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    /// Get comprehensive metrics
    pub fn get_metrics(&self) -> Metrics {
        let timestamp = Timestamp::now().as_nanos();
        
        Metrics {
            detection: self.get_detection_summary(),
            defense: self.get_defense_summary(),
            system: self.get_system_summary(),
            timestamp,
        }
    }
    
    /// Get detection performance summary
    fn get_detection_summary(&self) -> DetectionSummary {
        let total = self.detection_metrics.total_detections.load(Ordering::Acquire);
        let failed = self.detection_metrics.failed_detections.load(Ordering::Acquire);
        let total_time = self.detection_metrics.total_detection_time_ns.load(Ordering::Acquire);
        let min_time = self.detection_metrics.min_detection_time_ns.load(Ordering::Acquire);
        let max_time = self.detection_metrics.max_detection_time_ns.load(Ordering::Acquire);
        let violations = self.detection_metrics.threshold_violations.load(Ordering::Acquire);
        
        let success_rate = if total > 0 {
            (total - failed) as f64 / total as f64
        } else {
            1.0
        };
        
        let avg_time_ns = if total > 0 {
            total_time as f64 / total as f64
        } else {
            0.0
        };
        
        DetectionSummary {
            total,
            failed,
            success_rate,
            avg_time_ns,
            min_time_ns: if min_time == u64::MAX { 0 } else { min_time },
            max_time_ns: max_time,
            threshold_violations: violations,
        }
    }
    
    /// Get defense performance summary
    fn get_defense_summary(&self) -> DefenseSummary {
        let total = self.defense_metrics.total_defenses.load(Ordering::Acquire);
        let failed = self.defense_metrics.failed_defenses.load(Ordering::Acquire);
        let total_time = self.defense_metrics.total_defense_time_ns.load(Ordering::Acquire);
        let min_time = self.defense_metrics.min_defense_time_ns.load(Ordering::Acquire);
        let max_time = self.defense_metrics.max_defense_time_ns.load(Ordering::Acquire);
        let strategy_time = self.defense_metrics.strategy_calc_time_ns.load(Ordering::Acquire);
        let order_gen_time = self.defense_metrics.order_gen_time_ns.load(Ordering::Acquire);
        
        let success_rate = if total > 0 {
            (total - failed) as f64 / total as f64
        } else {
            1.0
        };
        
        let avg_time_ns = if total > 0 {
            total_time as f64 / total as f64
        } else {
            0.0
        };
        
        let avg_strategy_time_ns = if total > 0 {
            strategy_time as f64 / total as f64
        } else {
            0.0
        };
        
        let avg_order_gen_time_ns = if total > 0 {
            order_gen_time as f64 / total as f64
        } else {
            0.0
        };
        
        DefenseSummary {
            total,
            failed,
            success_rate,
            avg_time_ns,
            min_time_ns: if min_time == u64::MAX { 0 } else { min_time },
            max_time_ns: max_time,
            avg_strategy_time_ns,
            avg_order_gen_time_ns,
        }
    }
    
    /// Get system performance summary
    fn get_system_summary(&self) -> SystemSummary {
        let memory_usage = self.system_metrics.memory_usage_bytes.load(Ordering::Acquire);
        let cpu_cycles = self.system_metrics.cpu_cycles.load(Ordering::Acquire);
        let cache_misses = self.system_metrics.cache_misses.load(Ordering::Acquire);
        let lockfree_failures = self.system_metrics.lockfree_failures.load(Ordering::Acquire);
        let queue_overflows = self.system_metrics.queue_overflows.load(Ordering::Acquire);
        let queue_underflows = self.system_metrics.queue_underflows.load(Ordering::Acquire);
        
        // Calculate rates (simplified approximations)
        let total_operations = self.detection_metrics.total_detections.load(Ordering::Acquire) +
                              self.defense_metrics.total_defenses.load(Ordering::Acquire);
        
        let cache_miss_rate = if total_operations > 0 {
            cache_misses as f64 / (total_operations * 100) as f64 // Approximate
        } else {
            0.0
        };
        
        let lockfree_failure_rate = if total_operations > 0 {
            lockfree_failures as f64 / total_operations as f64
        } else {
            0.0
        };
        
        SystemSummary {
            memory_usage_bytes: memory_usage,
            cpu_cycles,
            cache_miss_rate,
            lockfree_failure_rate,
            queue_overflows,
            queue_underflows,
        }
    }
    
    /// Check if performance thresholds are violated
    pub fn check_thresholds(&self) -> Vec<String> {
        let mut violations = Vec::new();
        let metrics = self.get_metrics();
        
        if metrics.detection.avg_time_ns > self.thresholds.max_detection_latency_ns as f64 {
            violations.push(format!(
                "Detection latency exceeded: {:.0}ns > {}ns",
                metrics.detection.avg_time_ns,
                self.thresholds.max_detection_latency_ns
            ));
        }
        
        if metrics.defense.avg_time_ns > self.thresholds.max_defense_latency_ns as f64 {
            violations.push(format!(
                "Defense latency exceeded: {:.0}ns > {}ns",
                metrics.defense.avg_time_ns,
                self.thresholds.max_defense_latency_ns
            ));
        }
        
        if metrics.system.memory_usage_bytes > self.thresholds.max_memory_usage_bytes {
            violations.push(format!(
                "Memory usage exceeded: {} bytes > {} bytes",
                metrics.system.memory_usage_bytes,
                self.thresholds.max_memory_usage_bytes
            ));
        }
        
        if metrics.detection.success_rate < (1.0 - self.thresholds.max_failure_rate) {
            violations.push(format!(
                "Detection failure rate exceeded: {:.2}% > {:.2}%",
                (1.0 - metrics.detection.success_rate) * 100.0,
                self.thresholds.max_failure_rate * 100.0
            ));
        }
        
        violations
    }
    
    /// Reset all performance metrics
    pub fn reset_metrics(&self) {
        // Reset detection metrics
        self.detection_metrics.total_detections.store(0, Ordering::Relaxed);
        self.detection_metrics.failed_detections.store(0, Ordering::Relaxed);
        self.detection_metrics.total_detection_time_ns.store(0, Ordering::Relaxed);
        self.detection_metrics.min_detection_time_ns.store(u64::MAX, Ordering::Relaxed);
        self.detection_metrics.max_detection_time_ns.store(0, Ordering::Relaxed);
        self.detection_metrics.threshold_violations.store(0, Ordering::Relaxed);
        
        // Reset defense metrics
        self.defense_metrics.total_defenses.store(0, Ordering::Relaxed);
        self.defense_metrics.failed_defenses.store(0, Ordering::Relaxed);
        self.defense_metrics.total_defense_time_ns.store(0, Ordering::Relaxed);
        self.defense_metrics.min_defense_time_ns.store(u64::MAX, Ordering::Relaxed);
        self.defense_metrics.max_defense_time_ns.store(0, Ordering::Relaxed);
        self.defense_metrics.strategy_calc_time_ns.store(0, Ordering::Relaxed);
        self.defense_metrics.order_gen_time_ns.store(0, Ordering::Relaxed);
        
        // Reset system metrics
        self.system_metrics.cpu_cycles.store(0, Ordering::Relaxed);
        self.system_metrics.cache_misses.store(0, Ordering::Relaxed);
        self.system_metrics.lockfree_failures.store(0, Ordering::Relaxed);
        self.system_metrics.queue_overflows.store(0, Ordering::Relaxed);
        self.system_metrics.queue_underflows.store(0, Ordering::Relaxed);
    }
    
    /// Stop performance monitoring
    pub fn stop(&self) {
        self.is_monitoring.store(false, Ordering::Release);
    }
    
    /// Shutdown performance monitor
    pub unsafe fn shutdown(&self) -> Result<()> {
        self.stop();
        Ok(())
    }
    
    /// Update min/max values atomically
    #[inline(always)]
    fn update_min_max(&self, min_atomic: &AtomicU64, max_atomic: &AtomicU64, value: u64) {
        // Update minimum
        let mut current_min = min_atomic.load(Ordering::Relaxed);
        while value < current_min {
            match min_atomic.compare_exchange_weak(
                current_min,
                value,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_min = actual,
            }
        }
        
        // Update maximum
        let mut current_max = max_atomic.load(Ordering::Relaxed);
        while value > current_max {
            match max_atomic.compare_exchange_weak(
                current_max,
                value,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_max = actual,
            }
        }
    }
}

impl DetectionMetrics {
    fn new() -> Self {
        Self {
            total_detections: AtomicU64::new(0),
            failed_detections: AtomicU64::new(0),
            total_detection_time_ns: AtomicU64::new(0),
            min_detection_time_ns: AtomicU64::new(u64::MAX),
            max_detection_time_ns: AtomicU64::new(0),
            threshold_violations: AtomicU64::new(0),
        }
    }
}

impl DefenseMetrics {
    fn new() -> Self {
        Self {
            total_defenses: AtomicU64::new(0),
            failed_defenses: AtomicU64::new(0),
            total_defense_time_ns: AtomicU64::new(0),
            min_defense_time_ns: AtomicU64::new(u64::MAX),
            max_defense_time_ns: AtomicU64::new(0),
            strategy_calc_time_ns: AtomicU64::new(0),
            order_gen_time_ns: AtomicU64::new(0),
        }
    }
}

impl SystemMetrics {
    fn new() -> Self {
        Self {
            memory_usage_bytes: AtomicU64::new(0),
            cpu_cycles: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            lockfree_failures: AtomicU64::new(0),
            queue_overflows: AtomicU64::new(0),
            queue_underflows: AtomicU64::new(0),
        }
    }
}

/// Initialize performance counters
pub unsafe fn init_performance_counters() -> Result<()> {
    // Initialize hardware performance counters if available
    Ok(())
}

/// Shutdown performance counters
pub unsafe fn shutdown_performance_counters() {
    // Cleanup performance counter resources
}

/// Utility function for likely branch prediction
#[inline(always)]
fn likely(b: bool) -> bool {
    core::intrinsics::likely(b)
}

unsafe impl Send for PerformanceMonitor {}
unsafe impl Sync for PerformanceMonitor {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_performance_monitor_creation() {
        unsafe {
            let monitor = PerformanceMonitor::new().unwrap();
            assert!(!monitor.is_monitoring.load(Ordering::Acquire));
        }
    }
    
    #[test]
    fn test_detection_recording() {
        unsafe {
            let monitor = PerformanceMonitor::new().unwrap();
            monitor.start().unwrap();
            
            monitor.record_detection(100, true);
            monitor.record_detection(200, false);
            monitor.record_detection(150, true);
            
            let summary = monitor.get_detection_summary();
            assert_eq!(summary.total, 3);
            assert_eq!(summary.failed, 1);
            assert_eq!(summary.success_rate, 2.0 / 3.0);
            assert_eq!(summary.min_time_ns, 100);
            assert_eq!(summary.max_time_ns, 200);
        }
    }
    
    #[test]
    fn test_defense_recording() {
        unsafe {
            let monitor = PerformanceMonitor::new().unwrap();
            monitor.start().unwrap();
            
            monitor.record_defense(300, 100, 50, true);
            monitor.record_defense(400, 150, 75, false);
            
            let summary = monitor.get_defense_summary();
            assert_eq!(summary.total, 2);
            assert_eq!(summary.failed, 1);
            assert_eq!(summary.success_rate, 0.5);
            assert_eq!(summary.avg_time_ns, 350.0);
        }
    }
    
    #[test]
    fn test_threshold_checking() {
        unsafe {
            let monitor = PerformanceMonitor::new().unwrap();
            monitor.start().unwrap();
            
            // Record detection that exceeds threshold
            monitor.record_detection(1000, true); // Exceeds 500ns threshold
            
            let violations = monitor.check_thresholds();
            assert!(!violations.is_empty());
        }
    }
    
    #[test]
    fn test_metrics_reset() {
        unsafe {
            let monitor = PerformanceMonitor::new().unwrap();
            monitor.start().unwrap();
            
            monitor.record_detection(100, true);
            monitor.record_defense(200, 50, 25, true);
            
            let metrics_before = monitor.get_metrics();
            assert!(metrics_before.detection.total > 0);
            assert!(metrics_before.defense.total > 0);
            
            monitor.reset_metrics();
            
            let metrics_after = monitor.get_metrics();
            assert_eq!(metrics_after.detection.total, 0);
            assert_eq!(metrics_after.defense.total, 0);
        }
    }
}