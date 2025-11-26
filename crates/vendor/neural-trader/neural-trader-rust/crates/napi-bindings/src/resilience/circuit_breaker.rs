use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};
use anyhow::{anyhow, Result};
use std::future::Future;
use serde::{Deserialize, Serialize};

/// Circuit breaker for resilient operation execution
#[derive(Clone)]
pub struct CircuitBreaker {
    name: String,
    state: Arc<RwLock<CircuitState>>,
    config: CircuitBreakerConfig,
    metrics: Arc<RwLock<CircuitBreakerMetrics>>,
}

/// Configuration for circuit breaker behavior
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures before opening circuit
    pub failure_threshold: u32,
    /// Number of consecutive successes needed to close from half-open
    pub success_threshold: u32,
    /// Timeout for individual operations
    pub timeout: Duration,
    /// Time to wait before transitioning from open to half-open
    pub reset_timeout: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout: Duration::from_secs(30),
            reset_timeout: Duration::from_secs(60),
        }
    }
}

/// Current state of the circuit breaker
#[derive(Clone, Debug)]
enum CircuitState {
    /// Normal operation, tracking consecutive failures
    Closed { failures: u32 },
    /// Circuit is open, rejecting all requests
    Open { opened_at: Instant },
    /// Testing if service has recovered, tracking consecutive successes
    HalfOpen { successes: u32 },
}

/// Metrics tracking for circuit breaker performance
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CircuitBreakerMetrics {
    pub total_calls: u64,
    pub successful_calls: u64,
    pub failed_calls: u64,
    pub rejected_calls: u64,
    pub total_open_time_ms: u64,
    pub times_opened: u32,
    pub times_half_opened: u32,
}

impl CircuitBreaker {
    /// Create a new circuit breaker with the given name and configuration
    pub fn new(name: String, config: CircuitBreakerConfig) -> Self {
        log::info!(
            "Creating circuit breaker '{}' with failure_threshold={}, reset_timeout={:?}",
            name,
            config.failure_threshold,
            config.reset_timeout
        );

        Self {
            name,
            state: Arc::new(RwLock::new(CircuitState::Closed { failures: 0 })),
            config,
            metrics: Arc::new(RwLock::new(CircuitBreakerMetrics::default())),
        }
    }

    /// Execute an operation through the circuit breaker
    pub async fn call<F, T>(&self, operation: F) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        // Increment total calls
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_calls += 1;
        }

        let state = self.state.read().await;

        match *state {
            CircuitState::Open { opened_at } => {
                // Check if reset timeout has elapsed
                if opened_at.elapsed() > self.config.reset_timeout {
                    drop(state);
                    self.transition_to_half_open().await;
                    self.execute_half_open(operation).await
                } else {
                    // Reject the call
                    let mut metrics = self.metrics.write().await;
                    metrics.rejected_calls += 1;
                    drop(metrics);
                    drop(state);

                    Err(anyhow!(
                        "Circuit breaker '{}' is OPEN. Rejecting request. Time until retry: {:?}",
                        self.name,
                        self.config.reset_timeout - opened_at.elapsed()
                    ))
                }
            }
            CircuitState::HalfOpen { .. } => {
                drop(state);
                self.execute_half_open(operation).await
            }
            CircuitState::Closed { .. } => {
                drop(state);
                self.execute_closed(operation).await
            }
        }
    }

    /// Execute operation in closed state
    async fn execute_closed<F, T>(&self, operation: F) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        match tokio::time::timeout(self.config.timeout, operation).await {
            Ok(Ok(result)) => {
                // Success - reset failure counter
                self.reset_failures().await;

                let mut metrics = self.metrics.write().await;
                metrics.successful_calls += 1;
                drop(metrics);

                Ok(result)
            }
            Ok(Err(e)) => {
                // Operation failed
                self.record_failure().await;

                let mut metrics = self.metrics.write().await;
                metrics.failed_calls += 1;
                drop(metrics);

                Err(anyhow!(
                    "Operation failed in circuit breaker '{}': {}",
                    self.name,
                    e
                ))
            }
            Err(_) => {
                // Timeout
                self.record_failure().await;

                let mut metrics = self.metrics.write().await;
                metrics.failed_calls += 1;
                drop(metrics);

                Err(anyhow!(
                    "Operation timed out in circuit breaker '{}' after {:?}",
                    self.name,
                    self.config.timeout
                ))
            }
        }
    }

    /// Execute operation in half-open state
    async fn execute_half_open<F, T>(&self, operation: F) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        match tokio::time::timeout(self.config.timeout, operation).await {
            Ok(Ok(result)) => {
                // Success in half-open state
                self.record_success().await;

                let mut metrics = self.metrics.write().await;
                metrics.successful_calls += 1;
                drop(metrics);

                Ok(result)
            }
            Ok(Err(e)) => {
                // Failure in half-open state - trip back to open
                self.trip_breaker().await;

                let mut metrics = self.metrics.write().await;
                metrics.failed_calls += 1;
                drop(metrics);

                Err(anyhow!(
                    "Operation failed in half-open state for '{}': {}",
                    self.name,
                    e
                ))
            }
            Err(_) => {
                // Timeout in half-open state - trip back to open
                self.trip_breaker().await;

                let mut metrics = self.metrics.write().await;
                metrics.failed_calls += 1;
                drop(metrics);

                Err(anyhow!(
                    "Operation timed out in half-open state for '{}' after {:?}",
                    self.name,
                    self.config.timeout
                ))
            }
        }
    }

    /// Transition from open to half-open state
    async fn transition_to_half_open(&self) {
        let mut state = self.state.write().await;
        *state = CircuitState::HalfOpen { successes: 0 };

        let mut metrics = self.metrics.write().await;
        metrics.times_half_opened += 1;
        drop(metrics);

        log::info!("Circuit breaker '{}' transitioned to HALF_OPEN", self.name);
    }

    /// Record a failure in closed state
    async fn record_failure(&self) {
        let mut state = self.state.write().await;
        if let CircuitState::Closed { failures } = *state {
            let new_failures = failures + 1;
            if new_failures >= self.config.failure_threshold {
                *state = CircuitState::Open { opened_at: Instant::now() };

                let mut metrics = self.metrics.write().await;
                metrics.times_opened += 1;
                drop(metrics);

                log::warn!(
                    "Circuit breaker '{}' OPENED after {} consecutive failures",
                    self.name,
                    new_failures
                );
            } else {
                *state = CircuitState::Closed { failures: new_failures };
                log::debug!(
                    "Circuit breaker '{}' recorded failure {}/{}",
                    self.name,
                    new_failures,
                    self.config.failure_threshold
                );
            }
        }
    }

    /// Record a success in half-open state
    async fn record_success(&self) {
        let mut state = self.state.write().await;
        if let CircuitState::HalfOpen { successes } = *state {
            let new_successes = successes + 1;
            if new_successes >= self.config.success_threshold {
                *state = CircuitState::Closed { failures: 0 };
                log::info!(
                    "Circuit breaker '{}' CLOSED after {} consecutive successes",
                    self.name,
                    new_successes
                );
            } else {
                *state = CircuitState::HalfOpen { successes: new_successes };
                log::debug!(
                    "Circuit breaker '{}' recorded success {}/{}",
                    self.name,
                    new_successes,
                    self.config.success_threshold
                );
            }
        }
    }

    /// Reset failure counter
    async fn reset_failures(&self) {
        let mut state = self.state.write().await;
        if let CircuitState::Closed { .. } = *state {
            *state = CircuitState::Closed { failures: 0 };
        }
    }

    /// Trip the breaker back to open from half-open
    async fn trip_breaker(&self) {
        let mut state = self.state.write().await;
        *state = CircuitState::Open { opened_at: Instant::now() };

        let mut metrics = self.metrics.write().await;
        metrics.times_opened += 1;
        drop(metrics);

        log::warn!("Circuit breaker '{}' tripped to OPEN from half-open state", self.name);
    }

    /// Get current state as string
    pub async fn get_state(&self) -> String {
        let state = self.state.read().await;
        match *state {
            CircuitState::Closed { failures } => format!("CLOSED (failures: {})", failures),
            CircuitState::Open { opened_at } => {
                format!("OPEN (elapsed: {:?})", opened_at.elapsed())
            }
            CircuitState::HalfOpen { successes } => format!("HALF_OPEN (successes: {})", successes),
        }
    }

    /// Get circuit breaker metrics
    pub async fn get_metrics(&self) -> CircuitBreakerMetrics {
        let metrics = self.metrics.read().await;
        metrics.clone()
    }

    /// Reset all metrics
    pub async fn reset_metrics(&self) {
        let mut metrics = self.metrics.write().await;
        *metrics = CircuitBreakerMetrics::default();
        log::info!("Reset metrics for circuit breaker '{}'", self.name);
    }

    /// Get success rate as percentage
    pub async fn get_success_rate(&self) -> f64 {
        let metrics = self.metrics.read().await;
        if metrics.total_calls == 0 {
            return 100.0;
        }
        (metrics.successful_calls as f64 / metrics.total_calls as f64) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[tokio::test]
    async fn test_circuit_breaker_closed_success() {
        let cb = CircuitBreaker::new(
            "test".to_string(),
            CircuitBreakerConfig {
                failure_threshold: 3,
                success_threshold: 2,
                timeout: Duration::from_secs(1),
                reset_timeout: Duration::from_secs(5),
            },
        );

        let result = cb.call(async { Ok::<_, anyhow::Error>(42) }).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);

        let state = cb.get_state().await;
        assert!(state.starts_with("CLOSED"));
    }

    #[tokio::test]
    async fn test_circuit_breaker_opens_after_failures() {
        let cb = CircuitBreaker::new(
            "test".to_string(),
            CircuitBreakerConfig {
                failure_threshold: 3,
                success_threshold: 2,
                timeout: Duration::from_secs(1),
                reset_timeout: Duration::from_secs(5),
            },
        );

        // Trigger 3 failures
        for _ in 0..3 {
            let _ = cb
                .call(async { Err::<(), _>(anyhow!("test failure")) })
                .await;
        }

        let state = cb.get_state().await;
        assert!(state.starts_with("OPEN"));

        // Next call should be rejected
        let result = cb.call(async { Ok::<_, anyhow::Error>(42) }).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("is OPEN"));
    }

    #[tokio::test]
    async fn test_circuit_breaker_half_open_transition() {
        let cb = CircuitBreaker::new(
            "test".to_string(),
            CircuitBreakerConfig {
                failure_threshold: 2,
                success_threshold: 2,
                timeout: Duration::from_secs(1),
                reset_timeout: Duration::from_millis(100),
            },
        );

        // Open the circuit
        for _ in 0..2 {
            let _ = cb
                .call(async { Err::<(), _>(anyhow!("test failure")) })
                .await;
        }

        assert!(cb.get_state().await.starts_with("OPEN"));

        // Wait for reset timeout
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Next call should transition to half-open
        let _ = cb.call(async { Ok::<_, anyhow::Error>(42) }).await;

        let state = cb.get_state().await;
        assert!(state.starts_with("HALF_OPEN") || state.starts_with("CLOSED"));
    }

    #[tokio::test]
    async fn test_circuit_breaker_closes_after_successes() {
        let cb = CircuitBreaker::new(
            "test".to_string(),
            CircuitBreakerConfig {
                failure_threshold: 2,
                success_threshold: 2,
                timeout: Duration::from_secs(1),
                reset_timeout: Duration::from_millis(100),
            },
        );

        // Open the circuit
        for _ in 0..2 {
            let _ = cb
                .call(async { Err::<(), _>(anyhow!("test failure")) })
                .await;
        }

        // Wait for reset timeout
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Execute 2 successful operations to close circuit
        for _ in 0..2 {
            let result = cb.call(async { Ok::<_, anyhow::Error>(42) }).await;
            assert!(result.is_ok());
        }

        let state = cb.get_state().await;
        assert!(state.starts_with("CLOSED"));
    }

    #[tokio::test]
    async fn test_circuit_breaker_timeout() {
        let cb = CircuitBreaker::new(
            "test".to_string(),
            CircuitBreakerConfig {
                failure_threshold: 1,
                success_threshold: 2,
                timeout: Duration::from_millis(50),
                reset_timeout: Duration::from_secs(5),
            },
        );

        let result = cb
            .call(async {
                tokio::time::sleep(Duration::from_millis(200)).await;
                Ok::<_, anyhow::Error>(42)
            })
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("timed out"));

        let state = cb.get_state().await;
        assert!(state.starts_with("OPEN"));
    }

    #[tokio::test]
    async fn test_circuit_breaker_metrics() {
        let cb = CircuitBreaker::new(
            "test".to_string(),
            CircuitBreakerConfig::default(),
        );

        // Successful call
        let _ = cb.call(async { Ok::<_, anyhow::Error>(42) }).await;

        // Failed call
        let _ = cb
            .call(async { Err::<(), _>(anyhow!("test failure")) })
            .await;

        let metrics = cb.get_metrics().await;
        assert_eq!(metrics.total_calls, 2);
        assert_eq!(metrics.successful_calls, 1);
        assert_eq!(metrics.failed_calls, 1);

        let success_rate = cb.get_success_rate().await;
        assert_eq!(success_rate, 50.0);
    }

    #[tokio::test]
    async fn test_circuit_breaker_concurrent_calls() {
        let cb = Arc::new(CircuitBreaker::new(
            "test".to_string(),
            CircuitBreakerConfig {
                failure_threshold: 5,
                success_threshold: 2,
                timeout: Duration::from_secs(1),
                reset_timeout: Duration::from_secs(5),
            },
        ));

        let counter = Arc::new(AtomicU32::new(0));
        let mut handles = vec![];

        for _ in 0..10 {
            let cb_clone = cb.clone();
            let counter_clone = counter.clone();
            let handle = tokio::spawn(async move {
                let result = cb_clone
                    .call(async {
                        counter_clone.fetch_add(1, Ordering::SeqCst);
                        Ok::<_, anyhow::Error>(42)
                    })
                    .await;
                result.is_ok()
            });
            handles.push(handle);
        }

        let results: Vec<bool> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        assert_eq!(results.iter().filter(|&&r| r).count(), 10);
        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }
}
