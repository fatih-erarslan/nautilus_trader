//! Circuit breaker implementation for fault tolerance
//!
//! Provides production-grade circuit breaker patterns for handling
//! failures in distributed systems and external service integrations.

pub mod breaker;

pub use breaker::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError, CircuitBreakerMetrics, CircuitState,
};
