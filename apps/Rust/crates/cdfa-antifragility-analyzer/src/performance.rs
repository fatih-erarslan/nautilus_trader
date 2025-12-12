//! Performance monitoring and metrics

use std::time::{Duration, Instant};
use std::collections::VecDeque;
use serde::{Deserialize, Serialize};

/// Performance metrics for the antifragility analyzer
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Total number of analyses performed
    pub total_analyses: u64,
    /// Total time spent on analyses
    pub total_analysis_time: Duration,
    /// Average analysis time
    pub average_analysis_time: Duration,
    /// Recent analysis times (for trend analysis)
    pub recent_times: VecDeque<Duration>,
    /// Total data points processed
    pub total_data_points: u64,
    /// Average data points per analysis
    pub average_data_points: f64,
    /// Peak analysis time
    pub peak_analysis_time: Duration,
    /// Minimum analysis time
    pub min_analysis_time: Duration,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Error count
    pub error_count: u64,
    /// Success count
    pub success_count: u64,
    /// Throughput (analyses per second)
    pub throughput: f64,
    /// Last analysis timestamp
    pub last_analysis_time: Option<Instant>,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_analyses: 0,
            total_analysis_time: Duration::from_nanos(0),
            average_analysis_time: Duration::from_nanos(0),
            recent_times: VecDeque::with_capacity(100),
            total_data_points: 0,
            average_data_points: 0.0,
            peak_analysis_time: Duration::from_nanos(0),
            min_analysis_time: Duration::from_secs(u64::MAX),
            cache_hit_rate: 0.0,
            cache_hits: 0,
            cache_misses: 0,
            error_count: 0,
            success_count: 0,
            throughput: 0.0,
            last_analysis_time: None,
        }
    }
}

impl PerformanceMetrics {
    /// Create new performance metrics
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Record a successful analysis
    pub fn record_analysis(&mut self, duration: Duration, data_points: usize) {
        self.total_analyses += 1;
        self.success_count += 1;
        self.total_analysis_time += duration;
        self.total_data_points += data_points as u64;
        
        // Update averages
        self.average_analysis_time = self.total_analysis_time / self.total_analyses as u32;
        self.average_data_points = self.total_data_points as f64 / self.total_analyses as f64;
        
        // Update peak and min times
        if duration > self.peak_analysis_time {
            self.peak_analysis_time = duration;
        }
        if duration < self.min_analysis_time {
            self.min_analysis_time = duration;
        }
        
        // Update recent times (keep last 100)
        if self.recent_times.len() >= 100 {
            self.recent_times.pop_front();
        }
        self.recent_times.push_back(duration);
        
        // Update throughput
        self.update_throughput();
        
        self.last_analysis_time = Some(Instant::now());
    }
    
    /// Record a cache hit
    pub fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
        self.update_cache_hit_rate();
    }
    
    /// Record a cache miss
    pub fn record_cache_miss(&mut self) {
        self.cache_misses += 1;
        self.update_cache_hit_rate();
    }
    
    /// Record an error
    pub fn record_error(&mut self) {
        self.error_count += 1;
    }
    
    /// Update cache hit rate
    fn update_cache_hit_rate(&mut self) {
        let total_requests = self.cache_hits + self.cache_misses;
        if total_requests > 0 {
            self.cache_hit_rate = self.cache_hits as f64 / total_requests as f64;
        }
    }
    
    /// Update throughput calculation
    fn update_throughput(&mut self) {
        if let Some(last_time) = self.last_analysis_time {
            let now = Instant::now();
            let time_diff = now.duration_since(last_time).as_secs_f64();
            
            if time_diff > 0.0 {
                self.throughput = 1.0 / time_diff;
            }
        }
    }
    
    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        let total_attempts = self.success_count + self.error_count;
        if total_attempts > 0 {
            self.success_count as f64 / total_attempts as f64
        } else {
            0.0
        }
    }
    
    /// Get recent average analysis time
    pub fn recent_average_time(&self) -> Duration {
        if self.recent_times.is_empty() {
            return Duration::from_nanos(0);
        }
        
        let sum: Duration = self.recent_times.iter().sum();
        sum / self.recent_times.len() as u32
    }
    
    /// Get analysis time trend (positive = getting slower, negative = getting faster)
    pub fn time_trend(&self) -> f64 {
        if self.recent_times.len() < 10 {
            return 0.0;
        }
        
        let mid_point = self.recent_times.len() / 2;
        let first_half: Duration = self.recent_times.iter().take(mid_point).sum();
        let second_half: Duration = self.recent_times.iter().skip(mid_point).sum();
        
        let first_avg = first_half.as_nanos() as f64 / mid_point as f64;
        let second_avg = second_half.as_nanos() as f64 / (self.recent_times.len() - mid_point) as f64;
        
        (second_avg - first_avg) / first_avg
    }
    
    /// Get performance summary
    pub fn summary(&self) -> String {
        format!(
            "Performance Summary:\n\
             - Total Analyses: {}\n\
             - Success Rate: {:.2}%\n\
             - Average Time: {:?}\n\
             - Recent Average: {:?}\n\
             - Peak Time: {:?}\n\
             - Min Time: {:?}\n\
             - Cache Hit Rate: {:.2}%\n\
             - Throughput: {:.2} analyses/sec\n\
             - Time Trend: {:.2}%\n\
             - Avg Data Points: {:.0}",
            self.total_analyses,
            self.success_rate() * 100.0,
            self.average_analysis_time,
            self.recent_average_time(),
            self.peak_analysis_time,
            self.min_analysis_time,
            self.cache_hit_rate * 100.0,
            self.throughput,
            self.time_trend() * 100.0,
            self.average_data_points
        )
    }
    
    /// Check if performance is degrading
    pub fn is_performance_degrading(&self) -> bool {
        self.time_trend() > 0.1 // 10% slower
    }
    
    /// Get percentile analysis time
    pub fn percentile_time(&self, percentile: f64) -> Duration {
        if self.recent_times.is_empty() {
            return Duration::from_nanos(0);
        }
        
        let mut times: Vec<Duration> = self.recent_times.iter().copied().collect();
        times.sort();
        
        let index = ((percentile / 100.0) * times.len() as f64) as usize;
        let index = index.min(times.len() - 1);
        
        times[index]
    }
    
    /// Reset metrics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Benchmark runner for performance testing
pub struct BenchmarkRunner {
    name: String,
    iterations: usize,
    warmup_iterations: usize,
    results: Vec<Duration>,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner
    pub fn new(name: &str, iterations: usize) -> Self {
        Self {
            name: name.to_string(),
            iterations,
            warmup_iterations: iterations / 10, // 10% warmup
            results: Vec::with_capacity(iterations),
        }
    }
    
    /// Run a benchmark
    pub fn run<F>(&mut self, mut benchmark_fn: F) -> BenchmarkResult
    where
        F: FnMut() -> Result<(), Box<dyn std::error::Error>>,
    {
        // Warmup
        for _ in 0..self.warmup_iterations {
            let _ = benchmark_fn();
        }
        
        // Actual benchmark
        let mut errors = 0;
        for _ in 0..self.iterations {
            let start = Instant::now();
            
            match benchmark_fn() {
                Ok(()) => {
                    let duration = start.elapsed();
                    self.results.push(duration);
                }
                Err(_) => {
                    errors += 1;
                }
            }
        }
        
        self.calculate_results(errors)
    }
    
    /// Calculate benchmark results
    fn calculate_results(&self, errors: usize) -> BenchmarkResult {
        if self.results.is_empty() {
            return BenchmarkResult::default();
        }
        
        let mut sorted_results = self.results.clone();
        sorted_results.sort();
        
        let total_time: Duration = self.results.iter().sum();
        let mean_time = total_time / self.results.len() as u32;
        
        let median_time = sorted_results[sorted_results.len() / 2];
        let p95_time = sorted_results[(sorted_results.len() as f64 * 0.95) as usize];
        let p99_time = sorted_results[(sorted_results.len() as f64 * 0.99) as usize];
        
        let min_time = sorted_results[0];
        let max_time = sorted_results[sorted_results.len() - 1];
        
        BenchmarkResult {
            name: self.name.clone(),
            iterations: self.iterations,
            successful_iterations: self.results.len(),
            errors,
            mean_time,
            median_time,
            min_time,
            max_time,
            p95_time,
            p99_time,
        }
    }
}

/// Benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub iterations: usize,
    pub successful_iterations: usize,
    pub errors: usize,
    pub mean_time: Duration,
    pub median_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub p95_time: Duration,
    pub p99_time: Duration,
}

impl Default for BenchmarkResult {
    fn default() -> Self {
        Self {
            name: String::new(),
            iterations: 0,
            successful_iterations: 0,
            errors: 0,
            mean_time: Duration::from_nanos(0),
            median_time: Duration::from_nanos(0),
            min_time: Duration::from_nanos(0),
            max_time: Duration::from_nanos(0),
            p95_time: Duration::from_nanos(0),
            p99_time: Duration::from_nanos(0),
        }
    }
}

impl BenchmarkResult {
    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.iterations == 0 {
            0.0
        } else {
            self.successful_iterations as f64 / self.iterations as f64
        }
    }
    
    /// Get throughput (operations per second)
    pub fn throughput(&self) -> f64 {
        if self.mean_time.as_nanos() == 0 {
            0.0
        } else {
            1_000_000_000.0 / self.mean_time.as_nanos() as f64
        }
    }
    
    /// Format result as string
    pub fn format(&self) -> String {
        format!(
            "Benchmark: {}\n\
             Iterations: {} (Success: {}, Errors: {})\n\
             Success Rate: {:.2}%\n\
             Mean Time: {:?}\n\
             Median Time: {:?}\n\
             Min Time: {:?}\n\
             Max Time: {:?}\n\
             P95 Time: {:?}\n\
             P99 Time: {:?}\n\
             Throughput: {:.2} ops/sec",
            self.name,
            self.iterations,
            self.successful_iterations,
            self.errors,
            self.success_rate() * 100.0,
            self.mean_time,
            self.median_time,
            self.min_time,
            self.max_time,
            self.p95_time,
            self.p99_time,
            self.throughput()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    
    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::new();
        
        // Record some analyses
        metrics.record_analysis(Duration::from_millis(10), 100);
        metrics.record_analysis(Duration::from_millis(20), 200);
        metrics.record_analysis(Duration::from_millis(15), 150);
        
        assert_eq!(metrics.total_analyses, 3);
        assert_eq!(metrics.success_count, 3);
        assert_eq!(metrics.total_data_points, 450);
        assert_eq!(metrics.average_data_points, 150.0);
        assert_eq!(metrics.peak_analysis_time, Duration::from_millis(20));
        assert_eq!(metrics.min_analysis_time, Duration::from_millis(10));
    }
    
    #[test]
    fn test_cache_metrics() {
        let mut metrics = PerformanceMetrics::new();
        
        // Record cache operations
        metrics.record_cache_hit();
        metrics.record_cache_hit();
        metrics.record_cache_miss();
        
        assert_eq!(metrics.cache_hits, 2);
        assert_eq!(metrics.cache_misses, 1);
        assert_eq!(metrics.cache_hit_rate, 2.0 / 3.0);
    }
    
    #[test]
    fn test_error_tracking() {
        let mut metrics = PerformanceMetrics::new();
        
        metrics.record_analysis(Duration::from_millis(10), 100);
        metrics.record_error();
        metrics.record_error();
        
        assert_eq!(metrics.success_count, 1);
        assert_eq!(metrics.error_count, 2);
        assert_eq!(metrics.success_rate(), 1.0 / 3.0);
    }
    
    #[test]
    fn test_time_trend() {
        let mut metrics = PerformanceMetrics::new();
        
        // Add times that get progressively slower
        for i in 1..=20 {
            metrics.record_analysis(Duration::from_millis(i * 10), 100);
        }
        
        let trend = metrics.time_trend();
        assert!(trend > 0.0); // Should be positive (getting slower)
    }
    
    #[test]
    fn test_percentile_time() {
        let mut metrics = PerformanceMetrics::new();
        
        // Add various times
        for i in 1..=100 {
            metrics.record_analysis(Duration::from_millis(i), 100);
        }
        
        let p50 = metrics.percentile_time(50.0);
        let p95 = metrics.percentile_time(95.0);
        
        assert!(p95 > p50);
    }
    
    #[test]
    fn test_benchmark_runner() {
        let mut runner = BenchmarkRunner::new("test_benchmark", 10);
        
        let result = runner.run(|| {
            // Simulate some work
            thread::sleep(Duration::from_millis(1));
            Ok(())
        });
        
        assert_eq!(result.name, "test_benchmark");
        assert_eq!(result.iterations, 10);
        assert!(result.successful_iterations > 0);
        assert!(result.mean_time > Duration::from_nanos(0));
        assert!(result.success_rate() > 0.0);
        assert!(result.throughput() > 0.0);
    }
    
    #[test]
    fn test_benchmark_with_errors() {
        let mut runner = BenchmarkRunner::new("error_test", 5);
        
        let mut call_count = 0;
        let result = runner.run(|| {
            call_count += 1;
            if call_count % 2 == 0 {
                Err("Test error".into())
            } else {
                Ok(())
            }
        });
        
        assert!(result.errors > 0);
        assert!(result.success_rate() < 1.0);
    }
}