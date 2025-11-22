use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::net::TcpStream;
use tokio::sync::Semaphore;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};
use url::Url;

/// Connection pool for efficient WebSocket connection management
///
/// Manages a pool of WebSocket connections with automatic reconnection,
/// health checking, and load balancing capabilities.
///
/// References:
/// - Fowler, M. "Patterns of Enterprise Application Architecture" (2002)
/// - Apache Commons Pool - Connection pooling patterns
#[derive(Debug)]
pub struct ConnectionPool {
    connections: Arc<Mutex<VecDeque<PooledConnection>>>,
    semaphore: Arc<Semaphore>,
    config: PoolConfig,
}

#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub max_connections: usize,
    pub min_connections: usize,
    pub connection_timeout: Duration,
    pub idle_timeout: Duration,
    pub health_check_interval: Duration,
    pub max_retries: usize,
    pub retry_delay: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_connections: 10,
            min_connections: 2,
            connection_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(300), // 5 minutes
            health_check_interval: Duration::from_secs(60),
            max_retries: 3,
            retry_delay: Duration::from_millis(1000),
        }
    }
}

/// Pooled WebSocket connection with metadata
#[derive(Debug)]
pub struct PooledConnection {
    pub stream: Option<WebSocketStream<MaybeTlsStream<TcpStream>>>,
    pub created_at: Instant,
    pub last_used: Instant,
    pub usage_count: u64,
    pub is_healthy: bool,
    pub endpoint: String,
}

impl ConnectionPool {
    /// Create new connection pool with default configuration
    pub fn new(max_connections: usize) -> Self {
        let config = PoolConfig {
            max_connections,
            ..Default::default()
        };

        Self::with_config(config)
    }

    /// Create connection pool with custom configuration
    pub fn with_config(config: PoolConfig) -> Self {
        Self {
            connections: Arc::new(Mutex::new(VecDeque::new())),
            semaphore: Arc::new(Semaphore::new(config.max_connections)),
            config,
        }
    }

    /// Get connection from pool or create new one
    pub async fn get_connection(&self, endpoint: &str) -> Result<PooledConnection, PoolError> {
        // Acquire semaphore permit to respect max connections limit
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| PoolError::SemaphoreError)?;

        // Try to get existing healthy connection
        if let Some(mut connection) = self.try_get_existing_connection(endpoint) {
            if self.is_connection_healthy(&connection).await {
                connection.last_used = Instant::now();
                connection.usage_count += 1;
                return Ok(connection);
            }
        }

        // Create new connection with retries
        self.create_new_connection(endpoint).await
    }

    /// Return connection to pool
    pub async fn return_connection(
        &self,
        mut connection: PooledConnection,
    ) -> Result<(), PoolError> {
        // Update connection metadata
        connection.last_used = Instant::now();

        // Check if connection should be kept in pool
        if self.should_keep_connection(&connection) {
            let mut connections = self.connections.lock();

            // Limit pool size
            while connections.len() >= self.config.max_connections {
                if let Some(old_conn) = connections.pop_front() {
                    self.close_connection(old_conn).await;
                }
            }

            connections.push_back(connection);
        } else {
            // Close connection if it shouldn't be pooled
            self.close_connection(connection).await;
        }

        Ok(())
    }

    /// Try to get existing connection from pool
    fn try_get_existing_connection(&self, endpoint: &str) -> Option<PooledConnection> {
        let mut connections = self.connections.lock();

        // Find connection for the same endpoint that isn't too old
        for i in 0..connections.len() {
            if let Some(connection) = connections.get(i) {
                if connection.endpoint == endpoint
                    && connection.last_used.elapsed() < self.config.idle_timeout
                {
                    return connections.remove(i);
                }
            }
        }

        None
    }

    /// Create new WebSocket connection with exponential backoff
    async fn create_new_connection(&self, endpoint: &str) -> Result<PooledConnection, PoolError> {
        let mut retry_delay = self.config.retry_delay;

        for attempt in 0..self.config.max_retries {
            match self.establish_connection(endpoint).await {
                Ok(stream) => {
                    return Ok(PooledConnection {
                        stream: Some(stream),
                        created_at: Instant::now(),
                        last_used: Instant::now(),
                        usage_count: 0,
                        is_healthy: true,
                        endpoint: endpoint.to_string(),
                    });
                }
                Err(e) => {
                    if attempt < self.config.max_retries - 1 {
                        tokio::time::sleep(retry_delay).await;
                        retry_delay = std::cmp::min(retry_delay * 2, Duration::from_secs(30));
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Err(PoolError::MaxRetriesExceeded)
    }

    /// Establish WebSocket connection
    async fn establish_connection(
        &self,
        endpoint: &str,
    ) -> Result<WebSocketStream<MaybeTlsStream<TcpStream>>, PoolError> {
        let url = Url::parse(endpoint).map_err(|e| PoolError::InvalidUrl(e.to_string()))?;

        let (ws_stream, _) =
            tokio::time::timeout(self.config.connection_timeout, connect_async(url.as_str()))
                .await
                .map_err(|_| PoolError::ConnectionTimeout)?
                .map_err(|e| PoolError::ConnectionFailed(e.to_string()))?;

        Ok(ws_stream)
    }

    /// Check if connection is healthy
    async fn is_connection_healthy(&self, connection: &PooledConnection) -> bool {
        // Check if connection exists and isn't too old
        connection.stream.is_some() && 
        connection.created_at.elapsed() < Duration::from_secs(3600) && // 1 hour max age
        connection.is_healthy
    }

    /// Determine if connection should be kept in pool
    fn should_keep_connection(&self, connection: &PooledConnection) -> bool {
        connection.is_healthy
            && connection.created_at.elapsed() < Duration::from_secs(3600)
            && connection.usage_count < 1000 // Limit connection reuse
    }

    /// Close connection and clean up resources
    async fn close_connection(&self, mut connection: PooledConnection) {
        if let Some(mut stream) = connection.stream.take() {
            let _ = stream.close(None).await;
        }
    }

    /// Perform health check on all pooled connections
    pub async fn health_check(&self) -> Result<HealthCheckResult, PoolError> {
        let mut connections = self.connections.lock();
        let mut healthy_count = 0;
        let mut unhealthy_connections = Vec::new();

        for (index, connection) in connections.iter_mut().enumerate() {
            if self.is_connection_healthy(connection).await {
                healthy_count += 1;
            } else {
                connection.is_healthy = false;
                unhealthy_connections.push(index);
            }
        }

        // Remove unhealthy connections
        for &index in unhealthy_connections.iter().rev() {
            if let Some(connection) = connections.remove(index) {
                tokio::spawn(async move {
                    // Close connection in background
                    if let Some(mut stream) = connection.stream {
                        let _ = stream.close(None).await;
                    }
                });
            }
        }

        Ok(HealthCheckResult {
            total_connections: connections.len(),
            healthy_connections: healthy_count,
            unhealthy_connections: unhealthy_connections.len(),
        })
    }

    /// Clean up expired connections
    pub async fn cleanup_expired(&self) -> Result<usize, PoolError> {
        let mut connections = self.connections.lock();
        let initial_count = connections.len();

        let mut expired_connections = Vec::new();

        // Identify expired connections
        for (index, connection) in connections.iter().enumerate() {
            if connection.last_used.elapsed() > self.config.idle_timeout {
                expired_connections.push(index);
            }
        }

        // Remove expired connections
        for &index in expired_connections.iter().rev() {
            if let Some(connection) = connections.remove(index) {
                tokio::spawn(async move {
                    // Close connection in background
                    if let Some(mut stream) = connection.stream {
                        let _ = stream.close(None).await;
                    }
                });
            }
        }

        Ok(initial_count - connections.len())
    }

    /// Get pool statistics
    pub fn get_statistics(&self) -> PoolStatistics {
        let connections = self.connections.lock();

        let active_connections = connections.len();
        let total_usage: u64 = connections.iter().map(|c| c.usage_count).sum();

        PoolStatistics {
            active_connections,
            max_connections: self.config.max_connections,
            min_connections: self.config.min_connections,
            total_usage_count: total_usage,
            average_connection_age: if active_connections > 0 {
                connections
                    .iter()
                    .map(|c| c.created_at.elapsed().as_secs())
                    .sum::<u64>()
                    / active_connections as u64
            } else {
                0
            },
        }
    }
}

/// Health check results
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    pub total_connections: usize,
    pub healthy_connections: usize,
    pub unhealthy_connections: usize,
}

/// Pool statistics for monitoring
#[derive(Debug, Clone)]
pub struct PoolStatistics {
    pub active_connections: usize,
    pub max_connections: usize,
    pub min_connections: usize,
    pub total_usage_count: u64,
    pub average_connection_age: u64,
}

/// Pool errors
#[derive(Debug, Error)]
pub enum PoolError {
    #[error("Invalid URL: {0}")]
    InvalidUrl(String),

    #[error("Connection timeout")]
    ConnectionTimeout,

    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Max retries exceeded")]
    MaxRetriesExceeded,

    #[error("Semaphore error")]
    SemaphoreError,

    #[error("Pool exhausted")]
    PoolExhausted,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let pool = ConnectionPool::new(5);
        let stats = pool.get_statistics();
        assert_eq!(stats.max_connections, 5);
        assert_eq!(stats.active_connections, 0);
    }

    #[test]
    fn test_pool_config() {
        let config = PoolConfig {
            max_connections: 20,
            min_connections: 5,
            connection_timeout: Duration::from_secs(60),
            ..Default::default()
        };

        let pool = ConnectionPool::with_config(config);
        let stats = pool.get_statistics();
        assert_eq!(stats.max_connections, 20);
        assert_eq!(stats.min_connections, 5);
    }

    #[tokio::test]
    async fn test_cleanup_operations() {
        let pool = ConnectionPool::new(10);

        // Test cleanup of expired connections
        let cleaned_count = pool.cleanup_expired().await.unwrap();
        assert_eq!(cleaned_count, 0); // No connections to clean initially

        // Test health check
        let health_result = pool.health_check().await.unwrap();
        assert_eq!(health_result.total_connections, 0);
        assert_eq!(health_result.healthy_connections, 0);
    }
}
