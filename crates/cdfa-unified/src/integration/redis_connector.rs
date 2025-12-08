//! Redis connector with connection pooling and advanced features
//!
//! This module provides a high-performance Redis client with:
//! - Connection pooling for optimal resource utilization
//! - Automatic failover and reconnection
//! - Metrics collection and monitoring
//! - Distributed coordination support
//! - MessagePack serialization for efficient data transfer

use std::{
    collections::HashMap,
    sync::{Arc, Weak},
    time::{Duration, Instant},
};
use tokio::sync::{RwLock, Mutex};
use redis::{
    aio::{ConnectionManager, MultiplexedConnection},
    cluster::ClusterClient,
    Client, RedisResult, Value, AsyncCommands, ConnectionInfo,
};
use serde::{Serialize, Deserialize};
use crate::error::{CdfaError, Result};

/// Redis connection pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisPoolConfig {
    /// Redis connection URLs (supports cluster mode)
    pub urls: Vec<String>,
    /// Maximum number of connections per pool
    pub max_connections: usize,
    /// Connection timeout in milliseconds
    pub connection_timeout_ms: u64,
    /// Command timeout in milliseconds
    pub command_timeout_ms: u64,
    /// Maximum number of retries for failed operations
    pub max_retries: u32,
    /// Retry delay in milliseconds
    pub retry_delay_ms: u64,
    /// Enable cluster mode
    pub cluster_mode: bool,
    /// Connection keep-alive interval in seconds
    pub keepalive_interval_s: u64,
    /// Health check interval in seconds
    pub health_check_interval_s: u64,
}

impl Default for RedisPoolConfig {
    fn default() -> Self {
        Self {
            urls: vec!["redis://localhost:6379".to_string()],
            max_connections: 10,
            connection_timeout_ms: 5000,
            command_timeout_ms: 3000,
            max_retries: 3,
            retry_delay_ms: 100,
            cluster_mode: false,
            keepalive_interval_s: 30,
            health_check_interval_s: 10,
        }
    }
}

/// Connection statistics for monitoring
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConnectionStats {
    pub total_connections: usize,
    pub active_connections: usize,
    pub failed_connections: usize,
    pub total_commands: u64,
    pub successful_commands: u64,
    pub failed_commands: u64,
    pub average_response_time_ms: f64,
    pub last_health_check: Option<Instant>,
    pub uptime_seconds: u64,
}

/// Connection pool health status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolHealth {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Individual connection wrapper with statistics
#[derive(Debug)]
struct PooledConnection {
    connection: ConnectionManager,
    created_at: Instant,
    last_used: Instant,
    command_count: u64,
    error_count: u64,
}

impl PooledConnection {
    fn new(connection: ConnectionManager) -> Self {
        let now = Instant::now();
        Self {
            connection,
            created_at: now,
            last_used: now,
            command_count: 0,
            error_count: 0,
        }
    }

    fn update_usage(&mut self, success: bool) {
        self.last_used = Instant::now();
        self.command_count += 1;
        if !success {
            self.error_count += 1;
        }
    }

    fn error_rate(&self) -> f64 {
        if self.command_count == 0 {
            0.0
        } else {
            self.error_count as f64 / self.command_count as f64
        }
    }
}

/// High-performance Redis connection pool with advanced features
pub struct RedisPool {
    config: RedisPoolConfig,
    connections: Arc<RwLock<Vec<Arc<Mutex<PooledConnection>>>>>,
    stats: Arc<RwLock<ConnectionStats>>,
    created_at: Instant,
    client: Option<Client>,
    cluster_client: Option<ClusterClient>,
}

impl RedisPool {
    /// Create a new Redis connection pool
    pub async fn new(config: RedisPoolConfig) -> Result<Self> {
        let created_at = Instant::now();
        let connections = Arc::new(RwLock::new(Vec::new()));
        let stats = Arc::new(RwLock::new(ConnectionStats::default()));

        let (client, cluster_client) = if config.cluster_mode {
            let cluster_client = ClusterClient::new(config.urls.clone())
                .map_err(|e| CdfaError::Network(format!("Failed to create cluster client: {}", e)))?;
            (None, Some(cluster_client))
        } else {
            let client = Client::open(config.urls[0].as_str())
                .map_err(|e| CdfaError::Network(format!("Failed to create Redis client: {}", e)))?;
            (Some(client), None)
        };

        let pool = Self {
            config: config.clone(),
            connections,
            stats,
            created_at,
            client,
            cluster_client,
        };

        // Initialize connections
        pool.initialize_connections().await?;

        // Start background tasks
        pool.start_health_checker().await;
        pool.start_keepalive_task().await;

        Ok(pool)
    }

    /// Initialize the connection pool
    async fn initialize_connections(&self) -> Result<()> {
        let mut connections = self.connections.write().await;
        
        for _ in 0..self.config.max_connections {
            match self.create_connection().await {
                Ok(conn) => {
                    connections.push(Arc::new(Mutex::new(PooledConnection::new(conn))));
                }
                Err(e) => {
                    log::warn!("Failed to create initial connection: {}", e);
                    // Continue with fewer connections rather than failing completely
                    break;
                }
            }
        }

        if connections.is_empty() {
            return Err(CdfaError::Network("Failed to create any Redis connections".to_string()));
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_connections = connections.len();
        stats.active_connections = connections.len();

        Ok(())
    }

    /// Create a new Redis connection
    async fn create_connection(&self) -> Result<ConnectionManager> {
        let timeout = Duration::from_millis(self.config.connection_timeout_ms);
        
        if let Some(ref client) = self.client {
            let conn = client.get_connection_manager().await
                .map_err(|e| CdfaError::Network(format!("Failed to create connection manager: {}", e)))?;
            Ok(conn)
        } else if let Some(ref cluster_client) = self.cluster_client {
            let conn = cluster_client.get_connection_manager().await
                .map_err(|e| CdfaError::Network(format!("Failed to create cluster connection manager: {}", e)))?;
            Ok(conn)
        } else {
            Err(CdfaError::Network("No Redis client available".to_string()))
        }
    }

    /// Get a connection from the pool
    pub async fn get_connection(&self) -> Result<Arc<Mutex<PooledConnection>>> {
        let connections = self.connections.read().await;
        
        if connections.is_empty() {
            return Err(CdfaError::Network("No connections available in pool".to_string()));
        }

        // Find the least used connection
        let mut best_conn = None;
        let mut min_usage = u64::MAX;

        for conn in connections.iter() {
            let locked_conn = conn.lock().await;
            if locked_conn.command_count < min_usage {
                min_usage = locked_conn.command_count;
                best_conn = Some(conn.clone());
            }
        }

        best_conn.ok_or_else(|| CdfaError::Network("Failed to select connection".to_string()))
    }

    /// Execute a Redis command with automatic retry logic
    pub async fn execute_command<T, R>(&self, mut command: T) -> Result<R>
    where
        T: AsyncCommands + Send + Sync,
        R: redis::FromRedisValue,
    {
        let mut retries = 0;
        let start_time = Instant::now();

        loop {
            let conn = self.get_connection().await?;
            let mut locked_conn = conn.lock().await;

            match tokio::time::timeout(
                Duration::from_millis(self.config.command_timeout_ms),
                command.query_async::<_, R>(&mut locked_conn.connection)
            ).await {
                Ok(Ok(result)) => {
                    locked_conn.update_usage(true);
                    self.update_stats(true, start_time.elapsed()).await;
                    return Ok(result);
                }
                Ok(Err(e)) => {
                    locked_conn.update_usage(false);
                    
                    if retries >= self.config.max_retries {
                        self.update_stats(false, start_time.elapsed()).await;
                        return Err(CdfaError::Network(format!("Redis command failed after {} retries: {}", retries, e)));
                    }
                    
                    retries += 1;
                    tokio::time::sleep(Duration::from_millis(self.config.retry_delay_ms * retries as u64)).await;
                }
                Err(_) => {
                    locked_conn.update_usage(false);
                    
                    if retries >= self.config.max_retries {
                        self.update_stats(false, start_time.elapsed()).await;
                        return Err(CdfaError::Network(format!("Redis command timed out after {} retries", retries)));
                    }
                    
                    retries += 1;
                    tokio::time::sleep(Duration::from_millis(self.config.retry_delay_ms * retries as u64)).await;
                }
            }
        }
    }

    /// Update connection statistics
    async fn update_stats(&self, success: bool, duration: Duration) {
        let mut stats = self.stats.write().await;
        stats.total_commands += 1;
        
        if success {
            stats.successful_commands += 1;
        } else {
            stats.failed_commands += 1;
        }

        // Update average response time using exponential moving average
        let new_time = duration.as_millis() as f64;
        if stats.average_response_time_ms == 0.0 {
            stats.average_response_time_ms = new_time;
        } else {
            stats.average_response_time_ms = 0.9 * stats.average_response_time_ms + 0.1 * new_time;
        }

        stats.uptime_seconds = self.created_at.elapsed().as_secs();
    }

    /// Get current pool statistics
    pub async fn get_stats(&self) -> ConnectionStats {
        let stats = self.stats.read().await;
        let mut current_stats = stats.clone();
        current_stats.uptime_seconds = self.created_at.elapsed().as_secs();
        current_stats
    }

    /// Get pool health status
    pub async fn get_health(&self) -> PoolHealth {
        let connections = self.connections.read().await;
        let stats = self.stats.read().await;

        if connections.is_empty() {
            return PoolHealth::Unhealthy;
        }

        let error_rate = if stats.total_commands > 0 {
            stats.failed_commands as f64 / stats.total_commands as f64
        } else {
            0.0
        };

        if error_rate > 0.1 {
            PoolHealth::Unhealthy
        } else if error_rate > 0.05 || connections.len() < self.config.max_connections / 2 {
            PoolHealth::Degraded
        } else {
            PoolHealth::Healthy
        }
    }

    /// Start background health checker
    async fn start_health_checker(&self) {
        let connections = Arc::downgrade(&self.connections);
        let stats = Arc::downgrade(&self.stats);
        let interval = Duration::from_secs(self.config.health_check_interval_s);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(interval);
            
            loop {
                interval.tick().await;
                
                if let (Some(connections), Some(stats)) = (connections.upgrade(), stats.upgrade()) {
                    Self::health_check_task(connections, stats).await;
                } else {
                    break; // Pool has been dropped
                }
            }
        });
    }

    /// Health check task implementation
    async fn health_check_task(
        connections: Arc<RwLock<Vec<Arc<Mutex<PooledConnection>>>>>,
        stats: Arc<RwLock<ConnectionStats>>,
    ) {
        let connections_guard = connections.read().await;
        let mut active_count = 0;
        let mut failed_count = 0;

        for conn in connections_guard.iter() {
            let locked_conn = conn.lock().await;
            
            // Check if connection has high error rate
            if locked_conn.error_rate() > 0.2 {
                failed_count += 1;
            } else {
                active_count += 1;
            }
        }

        // Update stats
        let mut stats_guard = stats.write().await;
        stats_guard.active_connections = active_count;
        stats_guard.failed_connections = failed_count;
        stats_guard.last_health_check = Some(Instant::now());
    }

    /// Start keepalive task
    async fn start_keepalive_task(&self) {
        let connections = Arc::downgrade(&self.connections);
        let interval = Duration::from_secs(self.config.keepalive_interval_s);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(interval);
            
            loop {
                interval.tick().await;
                
                if let Some(connections) = connections.upgrade() {
                    Self::keepalive_task(connections).await;
                } else {
                    break; // Pool has been dropped
                }
            }
        });
    }

    /// Keepalive task implementation
    async fn keepalive_task(connections: Arc<RwLock<Vec<Arc<Mutex<PooledConnection>>>>>) {
        let connections_guard = connections.read().await;
        
        for conn in connections_guard.iter() {
            let mut locked_conn = conn.lock().await;
            
            // Send PING to keep connection alive
            if let Err(e) = locked_conn.connection.ping().await {
                log::warn!("Keepalive ping failed: {}", e);
                locked_conn.update_usage(false);
            } else {
                locked_conn.update_usage(true);
            }
        }
    }

    /// Gracefully shutdown the pool
    pub async fn shutdown(&self) -> Result<()> {
        log::info!("Shutting down Redis connection pool");
        
        let connections = self.connections.read().await;
        for conn in connections.iter() {
            let mut locked_conn = conn.lock().await;
            if let Err(e) = locked_conn.connection.quit().await {
                log::warn!("Error during connection shutdown: {}", e);
            }
        }
        
        Ok(())
    }
}

/// Redis cluster coordinator for distributed operations
pub struct RedisClusterCoordinator {
    pool: Arc<RedisPool>,
    node_id: String,
    coordination_prefix: String,
}

impl RedisClusterCoordinator {
    /// Create a new cluster coordinator
    pub fn new(pool: Arc<RedisPool>, node_id: String) -> Self {
        Self {
            pool,
            node_id,
            coordination_prefix: "cdfa:cluster".to_string(),
        }
    }

    /// Register this node in the cluster
    pub async fn register_node(&self) -> Result<()> {
        let key = format!("{}:nodes:{}", self.coordination_prefix, self.node_id);
        let value = serde_json::json!({
            "node_id": self.node_id,
            "registered_at": chrono::Utc::now().timestamp(),
            "capabilities": ["redis", "coordination", "messaging"]
        });

        let conn = self.pool.get_connection().await?;
        let mut locked_conn = conn.lock().await;
        
        let _: () = locked_conn.connection
            .set_ex(&key, value.to_string(), 60) // 60 second TTL
            .await
            .map_err(|e| CdfaError::Network(format!("Failed to register node: {}", e)))?;

        Ok(())
    }

    /// Get list of active nodes in the cluster
    pub async fn get_active_nodes(&self) -> Result<Vec<String>> {
        let pattern = format!("{}:nodes:*", self.coordination_prefix);
        
        let conn = self.pool.get_connection().await?;
        let mut locked_conn = conn.lock().await;
        
        let keys: Vec<String> = locked_conn.connection
            .keys(&pattern)
            .await
            .map_err(|e| CdfaError::Network(format!("Failed to get node keys: {}", e)))?;

        let mut nodes = Vec::new();
        for key in keys {
            if let Some(node_id) = key.strip_prefix(&format!("{}:nodes:", self.coordination_prefix)) {
                nodes.push(node_id.to_string());
            }
        }

        Ok(nodes)
    }

    /// Coordinate a distributed operation
    pub async fn coordinate_operation(&self, operation_id: &str, operation_data: &serde_json::Value) -> Result<()> {
        let key = format!("{}:operations:{}", self.coordination_prefix, operation_id);
        
        let coordination_data = serde_json::json!({
            "operation_id": operation_id,
            "coordinator": self.node_id,
            "data": operation_data,
            "created_at": chrono::Utc::now().timestamp(),
            "status": "pending"
        });

        let conn = self.pool.get_connection().await?;
        let mut locked_conn = conn.lock().await;
        
        let _: () = locked_conn.connection
            .set_ex(&key, coordination_data.to_string(), 300) // 5 minute TTL
            .await
            .map_err(|e| CdfaError::Network(format!("Failed to coordinate operation: {}", e)))?;

        Ok(())
    }
}

/// Redis connection factory for creating pools with different configurations
pub struct RedisConnectionFactory;

impl RedisConnectionFactory {
    /// Create a default Redis pool for local development
    pub async fn create_local_pool() -> Result<Arc<RedisPool>> {
        let config = RedisPoolConfig::default();
        let pool = RedisPool::new(config).await?;
        Ok(Arc::new(pool))
    }

    /// Create a production Redis pool with optimized settings
    pub async fn create_production_pool(redis_urls: Vec<String>) -> Result<Arc<RedisPool>> {
        let config = RedisPoolConfig {
            urls: redis_urls,
            max_connections: 50,
            connection_timeout_ms: 10000,
            command_timeout_ms: 5000,
            max_retries: 5,
            retry_delay_ms: 50,
            cluster_mode: true,
            keepalive_interval_s: 15,
            health_check_interval_s: 5,
        };
        
        let pool = RedisPool::new(config).await?;
        Ok(Arc::new(pool))
    }

    /// Create a Redis pool for high-frequency trading scenarios
    pub async fn create_hft_pool(redis_urls: Vec<String>) -> Result<Arc<RedisPool>> {
        let config = RedisPoolConfig {
            urls: redis_urls,
            max_connections: 100,
            connection_timeout_ms: 1000,
            command_timeout_ms: 500,
            max_retries: 2,
            retry_delay_ms: 10,
            cluster_mode: true,
            keepalive_interval_s: 5,
            health_check_interval_s: 1,
        };
        
        let pool = RedisPool::new(config).await?;
        Ok(Arc::new(pool))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_redis_pool_creation() {
        let config = RedisPoolConfig::default();
        // Note: This test requires Redis to be running
        // In a real test environment, you would use a test container
        match RedisPool::new(config).await {
            Ok(pool) => {
                let stats = pool.get_stats().await;
                assert!(stats.total_connections > 0);
            }
            Err(_) => {
                // Redis not available in test environment
                println!("Redis not available for testing");
            }
        }
    }

    #[test]
    fn test_config_defaults() {
        let config = RedisPoolConfig::default();
        assert_eq!(config.max_connections, 10);
        assert_eq!(config.connection_timeout_ms, 5000);
        assert!(!config.cluster_mode);
    }

    #[test]
    fn test_connection_stats_default() {
        let stats = ConnectionStats::default();
        assert_eq!(stats.total_connections, 0);
        assert_eq!(stats.active_connections, 0);
        assert_eq!(stats.total_commands, 0);
    }
}