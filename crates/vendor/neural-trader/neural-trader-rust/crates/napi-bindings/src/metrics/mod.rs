use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::info;

/// System-wide metrics for connection pool and neural memory
#[derive(Clone)]
pub struct SystemMetrics {
    data: Arc<RwLock<MetricsData>>,
}

#[derive(Debug, Clone)]
struct MetricsData {
    // Connection pool metrics
    pub pool_gets: u64,
    pub pool_timeouts: u64,
    pub pool_errors: u64,

    // Neural memory metrics
    pub neural_allocations: u64,
    pub neural_deallocations: u64,
    pub neural_memory_bytes: usize,
    pub neural_cache_hits: u64,
    pub neural_cache_misses: u64,

    // General metrics
    pub start_time: Instant,
    pub last_reset: Instant,
}

impl Default for MetricsData {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            pool_gets: 0,
            pool_timeouts: 0,
            pool_errors: 0,
            neural_allocations: 0,
            neural_deallocations: 0,
            neural_memory_bytes: 0,
            neural_cache_hits: 0,
            neural_cache_misses: 0,
            start_time: now,
            last_reset: now,
        }
    }
}

impl SystemMetrics {
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(MetricsData::default())),
        }
    }

    // Connection pool metrics
    pub fn record_pool_get(&self) {
        let mut data = self.data.write();
        data.pool_gets += 1;
    }

    pub fn record_pool_timeout(&self) {
        let mut data = self.data.write();
        data.pool_timeouts += 1;
    }

    pub fn record_pool_error(&self) {
        let mut data = self.data.write();
        data.pool_errors += 1;
    }

    // Neural memory metrics
    pub fn record_neural_allocation(&self, bytes: usize) {
        let mut data = self.data.write();
        data.neural_allocations += 1;
        data.neural_memory_bytes += bytes;
    }

    pub fn record_neural_deallocation(&self, bytes: usize) {
        let mut data = self.data.write();
        data.neural_deallocations += 1;
        data.neural_memory_bytes = data.neural_memory_bytes.saturating_sub(bytes);
    }

    pub fn record_neural_cache_hit(&self) {
        let mut data = self.data.write();
        data.neural_cache_hits += 1;
    }

    pub fn record_neural_cache_miss(&self) {
        let mut data = self.data.write();
        data.neural_cache_misses += 1;
    }

    // Get snapshot
    pub fn snapshot(&self) -> MetricsSnapshot {
        let data = self.data.read();
        let uptime = data.start_time.elapsed();
        let since_reset = data.last_reset.elapsed();

        let pool_success_rate = if data.pool_gets > 0 {
            let successful = data.pool_gets - data.pool_timeouts - data.pool_errors;
            (successful as f64 / data.pool_gets as f64) * 100.0
        } else {
            100.0
        };

        let cache_hit_rate = if data.neural_cache_hits + data.neural_cache_misses > 0 {
            (data.neural_cache_hits as f64
                / (data.neural_cache_hits + data.neural_cache_misses) as f64)
                * 100.0
        } else {
            0.0
        };

        MetricsSnapshot {
            // Connection pool
            pool_gets: data.pool_gets,
            pool_timeouts: data.pool_timeouts,
            pool_errors: data.pool_errors,
            pool_success_rate,

            // Neural memory
            neural_allocations: data.neural_allocations,
            neural_deallocations: data.neural_deallocations,
            neural_active_allocations: data
                .neural_allocations
                .saturating_sub(data.neural_deallocations),
            neural_memory_bytes: data.neural_memory_bytes,
            neural_memory_mb: data.neural_memory_bytes as f64 / (1024.0 * 1024.0),
            neural_cache_hits: data.neural_cache_hits,
            neural_cache_misses: data.neural_cache_misses,
            neural_cache_hit_rate: cache_hit_rate,

            // General
            uptime_seconds: uptime.as_secs(),
            time_since_reset_seconds: since_reset.as_secs(),
        }
    }

    // Reset counters
    pub fn reset(&self) {
        let mut data = self.data.write();
        let start_time = data.start_time; // Preserve original start time
        *data = MetricsData {
            start_time,
            ..Default::default()
        };
        info!("System metrics reset");
    }

    // Get rates (per second)
    pub fn rates(&self) -> MetricsRates {
        let snapshot = self.snapshot();
        let time_window = snapshot.time_since_reset_seconds.max(1) as f64;

        MetricsRates {
            pool_gets_per_sec: snapshot.pool_gets as f64 / time_window,
            pool_timeouts_per_sec: snapshot.pool_timeouts as f64 / time_window,
            pool_errors_per_sec: snapshot.pool_errors as f64 / time_window,
            neural_allocations_per_sec: snapshot.neural_allocations as f64 / time_window,
            neural_deallocations_per_sec: snapshot.neural_deallocations as f64 / time_window,
        }
    }
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    // Connection pool
    pub pool_gets: u64,
    pub pool_timeouts: u64,
    pub pool_errors: u64,
    pub pool_success_rate: f64,

    // Neural memory
    pub neural_allocations: u64,
    pub neural_deallocations: u64,
    pub neural_active_allocations: u64,
    pub neural_memory_bytes: usize,
    pub neural_memory_mb: f64,
    pub neural_cache_hits: u64,
    pub neural_cache_misses: u64,
    pub neural_cache_hit_rate: f64,

    // General
    pub uptime_seconds: u64,
    pub time_since_reset_seconds: u64,
}

#[derive(Debug, Clone)]
pub struct MetricsRates {
    pub pool_gets_per_sec: f64,
    pub pool_timeouts_per_sec: f64,
    pub pool_errors_per_sec: f64,
    pub neural_allocations_per_sec: f64,
    pub neural_deallocations_per_sec: f64,
}

/// Periodic metrics reporting
pub async fn metrics_reporter(metrics: Arc<SystemMetrics>, interval_secs: u64) {
    let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));

    loop {
        interval.tick().await;

        let snapshot = metrics.snapshot();
        let rates = metrics.rates();

        info!(
            "=== System Metrics Report ===\n\
             Pool: {} gets ({:.1}% success, {:.2}/s rate)\n\
             Pool Timeouts: {} ({:.2}/s)\n\
             Pool Errors: {} ({:.2}/s)\n\
             Neural Memory: {:.2} MB ({} active allocations)\n\
             Neural Cache: {:.1}% hit rate ({} hits, {} misses)\n\
             Uptime: {}s\n\
             ============================",
            snapshot.pool_gets,
            snapshot.pool_success_rate,
            rates.pool_gets_per_sec,
            snapshot.pool_timeouts,
            rates.pool_timeouts_per_sec,
            snapshot.pool_errors,
            rates.pool_errors_per_sec,
            snapshot.neural_memory_mb,
            snapshot.neural_active_allocations,
            snapshot.neural_cache_hit_rate,
            snapshot.neural_cache_hits,
            snapshot.neural_cache_misses,
            snapshot.uptime_seconds
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = SystemMetrics::new();
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.pool_gets, 0);
        assert_eq!(snapshot.neural_allocations, 0);
    }

    #[test]
    fn test_pool_metrics() {
        let metrics = SystemMetrics::new();
        metrics.record_pool_get();
        metrics.record_pool_get();
        metrics.record_pool_timeout();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.pool_gets, 2);
        assert_eq!(snapshot.pool_timeouts, 1);
        assert_eq!(snapshot.pool_success_rate, 50.0);
    }

    #[test]
    fn test_neural_metrics() {
        let metrics = SystemMetrics::new();
        metrics.record_neural_allocation(1024);
        metrics.record_neural_allocation(2048);
        metrics.record_neural_deallocation(1024);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.neural_allocations, 2);
        assert_eq!(snapshot.neural_deallocations, 1);
        assert_eq!(snapshot.neural_memory_bytes, 2048);
    }

    #[test]
    fn test_cache_metrics() {
        let metrics = SystemMetrics::new();
        metrics.record_neural_cache_hit();
        metrics.record_neural_cache_hit();
        metrics.record_neural_cache_hit();
        metrics.record_neural_cache_miss();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.neural_cache_hits, 3);
        assert_eq!(snapshot.neural_cache_misses, 1);
        assert_eq!(snapshot.neural_cache_hit_rate, 75.0);
    }

    #[test]
    fn test_metrics_reset() {
        let metrics = SystemMetrics::new();
        metrics.record_pool_get();
        metrics.record_neural_allocation(1024);

        metrics.reset();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.pool_gets, 0);
        assert_eq!(snapshot.neural_allocations, 0);
    }
}
