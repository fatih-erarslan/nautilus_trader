//! Utility functions and helpers for MCP orchestration.

use crate::error::{OrchestrationError, Result};
use crate::types::{AgentId, TaskId, Timestamp};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// Performance measurement utilities
pub mod performance {
    use super::*;
    use std::time::Instant;
    use tracing::{debug, warn};
    
    /// Simple performance timer
    #[derive(Debug)]
    pub struct Timer {
        name: String,
        start: Instant,
    }
    
    impl Timer {
        /// Start a new timer with the given name
        pub fn new(name: impl Into<String>) -> Self {
            Self {
                name: name.into(),
                start: Instant::now(),
            }
        }
        
        /// Get elapsed time without stopping the timer
        pub fn elapsed(&self) -> Duration {
            self.start.elapsed()
        }
        
        /// Stop the timer and return elapsed time
        pub fn stop(self) -> Duration {
            let elapsed = self.elapsed();
            debug!("Timer '{}' elapsed: {:?}", self.name, elapsed);
            elapsed
        }
        
        /// Stop the timer and warn if it exceeds the threshold
        pub fn stop_with_threshold(self, threshold: Duration) -> Duration {
            let elapsed = self.stop();
            if elapsed > threshold {
                warn!("Timer '{}' exceeded threshold {:?}: {:?}", self.name, threshold, elapsed);
            }
            elapsed
        }
    }
    
    /// Performance statistics collector
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PerfStats {
        /// Operation name
        pub name: String,
        /// Total operations
        pub count: u64,
        /// Total time spent
        pub total_time: Duration,
        /// Minimum time
        pub min_time: Duration,
        /// Maximum time
        pub max_time: Duration,
        /// Average time
        pub avg_time: Duration,
    }
    
    impl PerfStats {
        /// Create new performance statistics
        pub fn new(name: impl Into<String>) -> Self {
            Self {
                name: name.into(),
                count: 0,
                total_time: Duration::ZERO,
                min_time: Duration::MAX,
                max_time: Duration::ZERO,
                avg_time: Duration::ZERO,
            }
        }
        
        /// Record a measurement
        pub fn record(&mut self, duration: Duration) {
            self.count += 1;
            self.total_time += duration;
            self.min_time = self.min_time.min(duration);
            self.max_time = self.max_time.max(duration);
            self.avg_time = self.total_time / self.count as u32;
        }
        
        /// Get operations per second
        pub fn ops_per_second(&self) -> f64 {
            if self.total_time.is_zero() {
                0.0
            } else {
                self.count as f64 / self.total_time.as_secs_f64()
            }
        }
    }
    
    /// Performance statistics collector for multiple operations
    #[derive(Debug, Default)]
    pub struct PerfCollector {
        stats: HashMap<String, PerfStats>,
    }
    
    impl PerfCollector {
        /// Create a new performance collector
        pub fn new() -> Self {
            Self::default()
        }
        
        /// Record a measurement for the given operation
        pub fn record(&mut self, operation: impl Into<String>, duration: Duration) {
            let operation = operation.into();
            let stats = self.stats.entry(operation.clone()).or_insert_with(|| PerfStats::new(operation));
            stats.record(duration);
        }
        
        /// Get statistics for an operation
        pub fn get_stats(&self, operation: &str) -> Option<&PerfStats> {
            self.stats.get(operation)
        }
        
        /// Get all statistics
        pub fn get_all_stats(&self) -> &HashMap<String, PerfStats> {
            &self.stats
        }
        
        /// Clear all statistics
        pub fn clear(&mut self) {
            self.stats.clear();
        }
    }
}

/// Retry utilities for handling transient failures
pub mod retry {
    use super::*;
    use std::future::Future;
    use tokio::time::{sleep, Duration};
    use tracing::{debug, warn};
    
    /// Retry configuration
    #[derive(Debug, Clone)]
    pub struct RetryConfig {
        /// Maximum number of attempts
        pub max_attempts: u32,
        /// Initial delay between attempts
        pub initial_delay: Duration,
        /// Maximum delay between attempts
        pub max_delay: Duration,
        /// Backoff multiplier
        pub backoff_multiplier: f64,
        /// Jitter factor (0.0 to 1.0)
        pub jitter_factor: f64,
    }
    
    impl Default for RetryConfig {
        fn default() -> Self {
            Self {
                max_attempts: 3,
                initial_delay: Duration::from_millis(100),
                max_delay: Duration::from_secs(30),
                backoff_multiplier: 2.0,
                jitter_factor: 0.1,
            }
        }
    }
    
    impl RetryConfig {
        /// Calculate delay for the given attempt
        pub fn calculate_delay(&self, attempt: u32) -> Duration {
            let base_delay = self.initial_delay.as_millis() as f64 
                * self.backoff_multiplier.powi(attempt as i32);
            let delay = base_delay.min(self.max_delay.as_millis() as f64);
            
            // Add jitter
            let jitter = delay * self.jitter_factor * (rand::random::<f64>() - 0.5) * 2.0;
            let final_delay = (delay + jitter).max(0.0) as u64;
            
            Duration::from_millis(final_delay)
        }
    }
    
    /// Retry a future operation with the given configuration
    pub async fn retry_with_config<F, Fut, T, E>(
        config: RetryConfig,
        mut operation: F,
    ) -> std::result::Result<T, E>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = std::result::Result<T, E>>,
        E: std::fmt::Display,
    {
        let mut last_error = None;
        
        for attempt in 0..config.max_attempts {
            match operation().await {
                Ok(result) => {
                    if attempt > 0 {
                        debug!("Operation succeeded on attempt {}", attempt + 1);
                    }
                    return Ok(result);
                }
                Err(error) => {
                    last_error = Some(error);
                    
                    if attempt + 1 < config.max_attempts {
                        let delay = config.calculate_delay(attempt);
                        warn!("Operation failed on attempt {}, retrying in {:?}", attempt + 1, delay);
                        sleep(delay).await;
                    }
                }
            }
        }
        
        Err(last_error.unwrap())
    }
    
    /// Retry an operation with default configuration
    pub async fn retry<F, Fut, T, E>(operation: F) -> std::result::Result<T, E>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = std::result::Result<T, E>>,
        E: std::fmt::Display,
    {
        retry_with_config(RetryConfig::default(), operation).await
    }
}

/// Rate limiting utilities
pub mod rate_limit {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use tokio::sync::Semaphore;
    use tokio::time::{sleep, Duration, Instant};
    
    /// Token bucket rate limiter
    #[derive(Debug)]
    pub struct TokenBucket {
        /// Maximum number of tokens
        capacity: u64,
        /// Current number of tokens
        tokens: Arc<AtomicU64>,
        /// Token refill rate (tokens per second)
        refill_rate: f64,
        /// Last refill time
        last_refill: Arc<AtomicU64>,
    }
    
    impl TokenBucket {
        /// Create a new token bucket
        pub fn new(capacity: u64, refill_rate: f64) -> Self {
            Self {
                capacity,
                tokens: Arc::new(AtomicU64::new(capacity)),
                refill_rate,
                last_refill: Arc::new(AtomicU64::new(timestamp_millis())),
            }
        }
        
        /// Try to acquire tokens without blocking
        pub fn try_acquire(&self, tokens: u64) -> bool {
            self.refill();
            
            let current_tokens = self.tokens.load(Ordering::Relaxed);
            if current_tokens >= tokens {
                self.tokens.fetch_sub(tokens, Ordering::Relaxed);
                true
            } else {
                false
            }
        }
        
        /// Acquire tokens, blocking if necessary
        pub async fn acquire(&self, tokens: u64) -> Result<()> {
            while !self.try_acquire(tokens) {
                let wait_time = self.calculate_wait_time(tokens);
                sleep(wait_time).await;
            }
            Ok(())
        }
        
        /// Calculate time to wait for tokens
        fn calculate_wait_time(&self, tokens: u64) -> Duration {
            let current_tokens = self.tokens.load(Ordering::Relaxed);
            if current_tokens >= tokens {
                Duration::ZERO
            } else {
                let needed_tokens = tokens - current_tokens;
                let wait_seconds = needed_tokens as f64 / self.refill_rate;
                Duration::from_secs_f64(wait_seconds)
            }
        }
        
        /// Refill tokens based on elapsed time
        fn refill(&self) {
            let now = timestamp_millis();
            let last_refill = self.last_refill.load(Ordering::Relaxed);
            let elapsed_ms = now.saturating_sub(last_refill);
            
            if elapsed_ms > 0 {
                let tokens_to_add = (elapsed_ms as f64 / 1000.0 * self.refill_rate) as u64;
                if tokens_to_add > 0 {
                    let current_tokens = self.tokens.load(Ordering::Relaxed);
                    let new_tokens = (current_tokens + tokens_to_add).min(self.capacity);
                    self.tokens.store(new_tokens, Ordering::Relaxed);
                    self.last_refill.store(now, Ordering::Relaxed);
                }
            }
        }
    }
    
    /// Concurrency limiter using semaphore
    #[derive(Debug)]
    pub struct ConcurrencyLimiter {
        semaphore: Arc<Semaphore>,
    }
    
    impl ConcurrencyLimiter {
        /// Create a new concurrency limiter
        pub fn new(max_concurrent: usize) -> Self {
            Self {
                semaphore: Arc::new(Semaphore::new(max_concurrent)),
            }
        }
        
        /// Execute a future with concurrency limiting
        pub async fn execute<F, Fut, T>(&self, future: F) -> T
        where
            F: FnOnce() -> Fut,
            Fut: Future<Output = T>,
        {
            let _permit = self.semaphore.acquire().await.unwrap();
            future().await
        }
    }
}

/// Circuit breaker utilities
pub mod circuit_breaker {
    use super::*;
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
    use std::sync::Arc;
    use tokio::time::{Duration, Instant};
    
    /// Circuit breaker state
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum State {
        Closed,
        Open,
        HalfOpen,
    }
    
    /// Simple circuit breaker
    #[derive(Debug)]
    pub struct CircuitBreaker {
        /// Failure threshold
        failure_threshold: u32,
        /// Recovery timeout
        recovery_timeout: Duration,
        /// Current failure count
        failure_count: Arc<AtomicU32>,
        /// Last failure time
        last_failure_time: Arc<AtomicU64>,
        /// Current state
        state: Arc<AtomicU32>, // 0 = Closed, 1 = Open, 2 = HalfOpen
    }
    
    impl CircuitBreaker {
        /// Create a new circuit breaker
        pub fn new(failure_threshold: u32, recovery_timeout: Duration) -> Self {
            Self {
                failure_threshold,
                recovery_timeout,
                failure_count: Arc::new(AtomicU32::new(0)),
                last_failure_time: Arc::new(AtomicU64::new(0)),
                state: Arc::new(AtomicU32::new(0)), // Closed
            }
        }
        
        /// Get current state
        pub fn state(&self) -> State {
            match self.state.load(Ordering::Relaxed) {
                0 => State::Closed,
                1 => State::Open,
                2 => State::HalfOpen,
                _ => State::Closed,
            }
        }
        
        /// Check if operation is allowed
        pub fn is_call_allowed(&self) -> bool {
            match self.state() {
                State::Closed => true,
                State::Open => {
                    // Check if recovery timeout has elapsed
                    let now = timestamp_millis();
                    let last_failure = self.last_failure_time.load(Ordering::Relaxed);
                    if now - last_failure >= self.recovery_timeout.as_millis() as u64 {
                        // Transition to half-open
                        self.state.store(2, Ordering::Relaxed);
                        true
                    } else {
                        false
                    }
                }
                State::HalfOpen => true,
            }
        }
        
        /// Record a successful operation
        pub fn record_success(&self) {
            match self.state() {
                State::HalfOpen => {
                    // Success in half-open state closes the circuit
                    self.state.store(0, Ordering::Relaxed);
                    self.failure_count.store(0, Ordering::Relaxed);
                }
                State::Closed => {
                    // Reset failure count on success
                    self.failure_count.store(0, Ordering::Relaxed);
                }
                State::Open => {
                    // Ignore success in open state
                }
            }
        }
        
        /// Record a failed operation
        pub fn record_failure(&self) {
            let failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
            self.last_failure_time.store(timestamp_millis(), Ordering::Relaxed);
            
            match self.state() {
                State::Closed => {
                    if failures >= self.failure_threshold {
                        self.state.store(1, Ordering::Relaxed); // Open
                    }
                }
                State::HalfOpen => {
                    // Failure in half-open state reopens the circuit
                    self.state.store(1, Ordering::Relaxed); // Open
                }
                State::Open => {
                    // Already open, nothing to do
                }
            }
        }
        
        /// Execute an operation with circuit breaker protection
        pub async fn execute<F, Fut, T, E>(&self, operation: F) -> std::result::Result<T, CircuitBreakerError<E>>
        where
            F: FnOnce() -> Fut,
            Fut: Future<Output = std::result::Result<T, E>>,
        {
            if !self.is_call_allowed() {
                return Err(CircuitBreakerError::CircuitOpen);
            }
            
            match operation().await {
                Ok(result) => {
                    self.record_success();
                    Ok(result)
                }
                Err(error) => {
                    self.record_failure();
                    Err(CircuitBreakerError::OperationFailed(error))
                }
            }
        }
    }
    
    /// Circuit breaker error
    #[derive(Debug, thiserror::Error)]
    pub enum CircuitBreakerError<E> {
        #[error("Circuit breaker is open")]
        CircuitOpen,
        #[error("Operation failed: {0}")]
        OperationFailed(E),
    }
}

/// Validation utilities
pub mod validation {
    use super::*;
    
    /// Validate agent ID
    pub fn validate_agent_id(agent_id: &AgentId) -> Result<()> {
        // Check if UUID is valid (it should be by construction)
        if agent_id.uuid().is_nil() {
            return Err(OrchestrationError::invalid_state("Invalid agent ID: nil UUID"));
        }
        Ok(())
    }
    
    /// Validate task ID
    pub fn validate_task_id(task_id: &TaskId) -> Result<()> {
        // Check if UUID is valid (it should be by construction)
        if task_id.uuid().is_nil() {
            return Err(OrchestrationError::invalid_state("Invalid task ID: nil UUID"));
        }
        Ok(())
    }
    
    /// Validate timestamp
    pub fn validate_timestamp(timestamp: &Timestamp) -> Result<()> {
        let now = Timestamp::now();
        
        // Check if timestamp is too far in the future (more than 1 hour)
        if timestamp.as_millis() > now.as_millis() + 3600000 {
            return Err(OrchestrationError::invalid_state("Timestamp too far in the future"));
        }
        
        // Check if timestamp is too far in the past (more than 1 year)
        if timestamp.as_millis() + 31536000000 < now.as_millis() {
            return Err(OrchestrationError::invalid_state("Timestamp too far in the past"));
        }
        
        Ok(())
    }
    
    /// Validate string is not empty and within length limits
    pub fn validate_string(value: &str, field_name: &str, max_length: usize) -> Result<()> {
        if value.is_empty() {
            return Err(OrchestrationError::invalid_state(format!("{} cannot be empty", field_name)));
        }
        
        if value.len() > max_length {
            return Err(OrchestrationError::invalid_state(
                format!("{} cannot exceed {} characters", field_name, max_length)
            ));
        }
        
        Ok(())
    }
}

/// Formatting utilities
pub mod format {
    use super::*;
    
    /// Format duration in human-readable form
    pub fn format_duration(duration: Duration) -> String {
        let total_seconds = duration.as_secs();
        let days = total_seconds / 86400;
        let hours = (total_seconds % 86400) / 3600;
        let minutes = (total_seconds % 3600) / 60;
        let seconds = total_seconds % 60;
        let millis = duration.subsec_millis();
        
        if days > 0 {
            format!("{}d {}h {}m {}s", days, hours, minutes, seconds)
        } else if hours > 0 {
            format!("{}h {}m {}s", hours, minutes, seconds)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, seconds)
        } else if seconds > 0 {
            format!("{}.{:03}s", seconds, millis)
        } else {
            format!("{}ms", millis)
        }
    }
    
    /// Format bytes in human-readable form
    pub fn format_bytes(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        const THRESHOLD: f64 = 1024.0;
        
        if bytes == 0 {
            return "0 B".to_string();
        }
        
        let mut size = bytes as f64;
        let mut unit_index = 0;
        
        while size >= THRESHOLD && unit_index < UNITS.len() - 1 {
            size /= THRESHOLD;
            unit_index += 1;
        }
        
        if unit_index == 0 {
            format!("{} {}", bytes, UNITS[unit_index])
        } else {
            format!("{:.2} {}", size, UNITS[unit_index])
        }
    }
    
    /// Format percentage
    pub fn format_percentage(value: f64) -> String {
        format!("{:.1}%", value * 100.0)
    }
}

/// Helper function to get current timestamp in milliseconds
fn timestamp_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};
    
    #[test]
    fn test_performance_timer() {
        let timer = performance::Timer::new("test");
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = timer.stop();
        assert!(elapsed.as_millis() >= 10);
    }
    
    #[test]
    fn test_performance_stats() {
        let mut stats = performance::PerfStats::new("test_op");
        
        stats.record(Duration::from_millis(100));
        stats.record(Duration::from_millis(200));
        stats.record(Duration::from_millis(300));
        
        assert_eq!(stats.count, 3);
        assert_eq!(stats.min_time, Duration::from_millis(100));
        assert_eq!(stats.max_time, Duration::from_millis(300));
        assert_eq!(stats.avg_time, Duration::from_millis(200));
    }
    
    #[tokio::test]
    async fn test_retry_success() {
        let mut attempt = 0;
        let result = retry::retry(|| async {
            attempt += 1;
            if attempt < 3 {
                Err("not yet")
            } else {
                Ok("success")
            }
        }).await;
        
        assert_eq!(result.unwrap(), "success");
        assert_eq!(attempt, 3);
    }
    
    #[tokio::test]
    async fn test_retry_failure() {
        let result = retry::retry(|| async {
            Err("always fails")
        }).await;
        
        assert!(result.is_err());
    }
    
    #[test]
    fn test_token_bucket() {
        let bucket = rate_limit::TokenBucket::new(10, 1.0); // 10 tokens, 1 token/second
        
        // Should be able to acquire up to capacity
        assert!(bucket.try_acquire(5));
        assert!(bucket.try_acquire(5));
        
        // Should fail to acquire more
        assert!(!bucket.try_acquire(1));
    }
    
    #[tokio::test]
    async fn test_concurrency_limiter() {
        let limiter = rate_limit::ConcurrencyLimiter::new(2);
        let start = std::time::Instant::now();
        
        // Start 3 concurrent operations that take 100ms each
        let futures = (0..3).map(|_| {
            let limiter = &limiter;
            async move {
                limiter.execute(|| async {
                    sleep(Duration::from_millis(100)).await;
                }).await
            }
        });
        
        futures::future::join_all(futures).await;
        
        // Should take at least 200ms (2 batches of 100ms each)
        assert!(start.elapsed() >= Duration::from_millis(150));
    }
    
    #[test]
    fn test_circuit_breaker() {
        let cb = circuit_breaker::CircuitBreaker::new(3, Duration::from_millis(100));
        
        // Initially closed
        assert_eq!(cb.state(), circuit_breaker::State::Closed);
        assert!(cb.is_call_allowed());
        
        // Record failures
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), circuit_breaker::State::Closed);
        
        cb.record_failure();
        assert_eq!(cb.state(), circuit_breaker::State::Open);
        assert!(!cb.is_call_allowed());
        
        // Record success in closed state
        let cb2 = circuit_breaker::CircuitBreaker::new(3, Duration::from_millis(100));
        cb2.record_success();
        assert_eq!(cb2.state(), circuit_breaker::State::Closed);
    }
    
    #[test]
    fn test_validation() {
        // Test agent ID validation
        let agent_id = AgentId::new();
        assert!(validation::validate_agent_id(&agent_id).is_ok());
        
        // Test task ID validation
        let task_id = TaskId::new();
        assert!(validation::validate_task_id(&task_id).is_ok());
        
        // Test timestamp validation
        let timestamp = Timestamp::now();
        assert!(validation::validate_timestamp(&timestamp).is_ok());
        
        // Test string validation
        assert!(validation::validate_string("valid", "test", 10).is_ok());
        assert!(validation::validate_string("", "test", 10).is_err());
        assert!(validation::validate_string("too_long_string", "test", 5).is_err());
    }
    
    #[test]
    fn test_formatting() {
        // Test duration formatting
        assert_eq!(format::format_duration(Duration::from_millis(500)), "500ms");
        assert_eq!(format::format_duration(Duration::from_secs(65)), "1m 5s");
        assert_eq!(format::format_duration(Duration::from_secs(3665)), "1h 1m 5s");
        
        // Test bytes formatting
        assert_eq!(format::format_bytes(0), "0 B");
        assert_eq!(format::format_bytes(1024), "1.00 KB");
        assert_eq!(format::format_bytes(1048576), "1.00 MB");
        
        // Test percentage formatting
        assert_eq!(format::format_percentage(0.75), "75.0%");
    }
}