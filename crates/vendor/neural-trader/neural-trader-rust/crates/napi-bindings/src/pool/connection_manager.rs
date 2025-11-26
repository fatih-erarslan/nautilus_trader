use anyhow::{anyhow, Result};
use deadpool::managed::{Manager, Object, Pool, RecycleResult};
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

/// Connection object that can be pooled
#[derive(Clone)]
pub struct Connection {
    pub id: String,
    pub created_at: Instant,
    pub last_used: Instant,
}

impl Connection {
    pub fn new(id: String) -> Self {
        let now = Instant::now();
        Self {
            id,
            created_at: now,
            last_used: now,
        }
    }

    pub fn update_last_used(&mut self) {
        self.last_used = Instant::now();
    }

    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

/// Internal connection manager for deadpool
pub struct ConnectionManagerInner {
    connection_counter: Arc<RwLock<usize>>,
}

impl ConnectionManagerInner {
    pub fn new() -> Self {
        Self {
            connection_counter: Arc::new(RwLock::new(0)),
        }
    }
}

impl Manager for ConnectionManagerInner {
    type Type = Connection;
    type Error = anyhow::Error;

    async fn create(&self) -> Result<Connection> {
        let mut counter = self.connection_counter.write();
        *counter += 1;
        let id = format!("conn-{}", *counter);
        debug!("Creating new connection: {}", id);
        Ok(Connection::new(id))
    }

    async fn recycle(&self, conn: &mut Connection, _metrics: &deadpool::managed::Metrics) -> RecycleResult<anyhow::Error> {
        conn.update_last_used();

        // Check if connection is too old (max age: 1 hour)
        if conn.age() > Duration::from_secs(3600) {
            warn!("Connection {} is too old, marking for recreation", conn.id);
            return Err(deadpool::managed::RecycleError::Backend(
                anyhow!("Connection too old"),
            ));
        }

        debug!("Recycling connection: {}", conn.id);
        Ok(())
    }
}

/// High-performance connection pool manager
pub struct ConnectionManager {
    pool: Pool<ConnectionManagerInner>,
    max_size: usize,
    timeout: Duration,
    metrics: Arc<RwLock<PoolMetricsData>>,
}

#[derive(Debug, Clone)]
struct PoolMetricsData {
    total_gets: u64,
    successful_gets: u64,
    timeouts: u64,
    errors: u64,
    last_reset: Instant,
}

impl Default for PoolMetricsData {
    fn default() -> Self {
        Self {
            total_gets: 0,
            successful_gets: 0,
            timeouts: 0,
            errors: 0,
            last_reset: Instant::now(),
        }
    }
}

impl ConnectionManager {
    /// Create a new connection pool manager
    pub fn new(max_size: usize, timeout_secs: u64) -> Result<Self> {
        info!(
            "Initializing connection pool with max_size={}, timeout={}s",
            max_size, timeout_secs
        );

        let manager = ConnectionManagerInner::new();
        let pool = Pool::builder(manager)
            .max_size(max_size)
            .wait_timeout(Some(Duration::from_secs(timeout_secs)))
            .recycle_timeout(Some(Duration::from_secs(30)))
            .build()
            .map_err(|e| anyhow!("Failed to create connection pool: {}", e))?;

        Ok(Self {
            pool,
            max_size,
            timeout: Duration::from_secs(timeout_secs),
            metrics: Arc::new(RwLock::new(PoolMetricsData::default())),
        })
    }

    /// Get a connection from the pool
    pub async fn get_connection(&self) -> Result<Object<ConnectionManagerInner>> {
        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.total_gets += 1;
        }

        match self.pool.get().await {
            Ok(conn) => {
                let mut metrics = self.metrics.write();
                metrics.successful_gets += 1;
                debug!("Connection acquired successfully");
                Ok(conn)
            }
            Err(e) => {
                let mut metrics = self.metrics.write();
                if e.to_string().contains("timeout") {
                    metrics.timeouts += 1;
                    error!("Connection pool timeout");
                } else {
                    metrics.errors += 1;
                    error!("Connection pool error: {}", e);
                }
                Err(anyhow!("Connection pool exhausted: {}", e))
            }
        }
    }

    /// Get current pool metrics
    pub fn metrics(&self) -> PoolMetrics {
        let status = self.pool.status();
        let metrics_data = self.metrics.read();

        PoolMetrics {
            max_size: self.max_size,
            current_size: status.size,
            available: status.available,
            waiting: status.waiting,
            total_gets: metrics_data.total_gets,
            successful_gets: metrics_data.successful_gets,
            timeouts: metrics_data.timeouts,
            errors: metrics_data.errors,
            success_rate: if metrics_data.total_gets > 0 {
                (metrics_data.successful_gets as f64 / metrics_data.total_gets as f64) * 100.0
            } else {
                100.0
            },
            uptime_seconds: metrics_data.last_reset.elapsed().as_secs(),
        }
    }

    /// Reset metrics counters
    pub fn reset_metrics(&self) {
        let mut metrics = self.metrics.write();
        *metrics = PoolMetricsData::default();
        info!("Pool metrics reset");
    }

    /// Get pool health status
    pub fn health_check(&self) -> PoolHealth {
        let metrics = self.metrics();

        let health_score = if metrics.total_gets == 0 {
            100.0
        } else {
            let timeout_rate = (metrics.timeouts as f64 / metrics.total_gets as f64) * 100.0;
            let error_rate = (metrics.errors as f64 / metrics.total_gets as f64) * 100.0;
            100.0 - timeout_rate - error_rate
        };

        let status = if health_score >= 95.0 {
            HealthStatus::Healthy
        } else if health_score >= 80.0 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Unhealthy
        };

        PoolHealth {
            status,
            health_score,
            available_connections: metrics.available,
            waiting_requests: metrics.waiting,
            utilization_percent: if metrics.max_size > 0 {
                ((metrics.current_size - metrics.available) as f64 / metrics.max_size as f64) * 100.0
            } else {
                0.0
            },
        }
    }
}

/// Pool metrics snapshot
#[derive(Debug, Clone)]
pub struct PoolMetrics {
    pub max_size: usize,
    pub current_size: usize,
    pub available: usize,
    pub waiting: usize,
    pub total_gets: u64,
    pub successful_gets: u64,
    pub timeouts: u64,
    pub errors: u64,
    pub success_rate: f64,
    pub uptime_seconds: u64,
}

/// Pool health information
#[derive(Debug, Clone)]
pub struct PoolHealth {
    pub status: HealthStatus,
    pub health_score: f64,
    pub available_connections: usize,
    pub waiting_requests: usize,
    pub utilization_percent: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

// Configuration constants
pub const DEFAULT_POOL_SIZE: usize = 2000; // Increased from 100 for high concurrency
pub const DEFAULT_TIMEOUT_SECS: u64 = 5;
pub const MAX_POOL_SIZE: usize = 10000;
pub const MIN_POOL_SIZE: usize = 10;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_connection_manager_creation() {
        let manager = ConnectionManager::new(10, 5).unwrap();
        let metrics = manager.metrics();
        assert_eq!(metrics.max_size, 10);
        assert_eq!(metrics.total_gets, 0);
    }

    #[tokio::test]
    async fn test_connection_acquisition() {
        let manager = ConnectionManager::new(10, 5).unwrap();
        let conn = manager.get_connection().await.unwrap();
        assert!(conn.id.starts_with("conn-"));
    }

    #[tokio::test]
    async fn test_metrics_tracking() {
        let manager = ConnectionManager::new(10, 5).unwrap();

        for _ in 0..5 {
            let _conn = manager.get_connection().await.unwrap();
        }

        let metrics = manager.metrics();
        assert_eq!(metrics.total_gets, 5);
        assert_eq!(metrics.successful_gets, 5);
        assert_eq!(metrics.success_rate, 100.0);
    }

    #[tokio::test]
    async fn test_health_check() {
        let manager = ConnectionManager::new(10, 5).unwrap();
        let health = manager.health_check();
        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.health_score, 100.0);
    }
}
