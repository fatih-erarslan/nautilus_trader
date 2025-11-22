//! Connection pooling for efficient resource management
//!
//! Provides connection pooling capabilities for WebSocket connections
//! with automatic health checking, reconnection, and load balancing.

pub mod connection_pool;

pub use connection_pool::{
    ConnectionPool, HealthCheckResult, PoolConfig, PoolError, PoolStatistics, PooledConnection,
};
