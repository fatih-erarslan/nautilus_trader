use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: usize,
    pub recovery_timeout: Duration,
    pub half_open_max_calls: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(60),
            half_open_max_calls: 3,
        }
    }
}

/// Production-grade circuit breaker for fault tolerance
///
/// References:
/// - Nygard, M. "Release It!" (2018) - Circuit Breaker Pattern
/// - Fowler, M. "Circuit Breaker" (2014) - Microservices patterns
#[derive(Debug)]
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: Arc<RwLock<CircuitBreakerState>>,
}

#[derive(Debug)]
struct CircuitBreakerState {
    current_state: CircuitState,
    failure_count: usize,
    last_failure_time: Option<Instant>,
    half_open_calls: usize,
    success_count: usize,
}

impl CircuitBreaker {
    /// Create a new circuit breaker with configuration
    pub fn new(failure_threshold: usize, recovery_timeout_secs: u64) -> Self {
        let config = CircuitBreakerConfig {
            failure_threshold,
            recovery_timeout: Duration::from_secs(recovery_timeout_secs),
            half_open_max_calls: 3,
        };

        Self {
            config,
            state: Arc::new(RwLock::new(CircuitBreakerState {
                current_state: CircuitState::Closed,
                failure_count: 0,
                last_failure_time: None,
                half_open_calls: 0,
                success_count: 0,
            })),
        }
    }

    /// Check if the circuit breaker allows the call
    pub fn check_health(&self) -> Result<(), CircuitBreakerError> {
        let mut state = self.state.write();

        match state.current_state {
            CircuitState::Closed => {
                // Normal operation - allow all calls
                Ok(())
            }
            CircuitState::Open => {
                // Check if recovery timeout has passed
                if let Some(last_failure) = state.last_failure_time {
                    if last_failure.elapsed() >= self.config.recovery_timeout {
                        // Transition to half-open state
                        state.current_state = CircuitState::HalfOpen;
                        state.half_open_calls = 0;
                        state.success_count = 0;
                        Ok(())
                    } else {
                        Err(CircuitBreakerError::CircuitOpen {
                            remaining_timeout: self.config.recovery_timeout
                                - last_failure.elapsed(),
                        })
                    }
                } else {
                    // No last failure time recorded, transition to half-open
                    state.current_state = CircuitState::HalfOpen;
                    state.half_open_calls = 0;
                    state.success_count = 0;
                    Ok(())
                }
            }
            CircuitState::HalfOpen => {
                // Limited calls allowed to test recovery
                if state.half_open_calls < self.config.half_open_max_calls {
                    state.half_open_calls += 1;
                    Ok(())
                } else {
                    Err(CircuitBreakerError::HalfOpenLimitExceeded)
                }
            }
        }
    }

    /// Record a successful operation
    pub fn record_success(&self) {
        let mut state = self.state.write();

        match state.current_state {
            CircuitState::Closed => {
                // Reset failure count on success
                state.failure_count = 0;
            }
            CircuitState::HalfOpen => {
                state.success_count += 1;

                // If enough successes in half-open state, close the circuit
                if state.success_count >= self.config.half_open_max_calls {
                    state.current_state = CircuitState::Closed;
                    state.failure_count = 0;
                    state.last_failure_time = None;
                    state.half_open_calls = 0;
                    state.success_count = 0;
                }
            }
            CircuitState::Open => {
                // Should not happen, but reset to closed if it does
                state.current_state = CircuitState::Closed;
                state.failure_count = 0;
                state.last_failure_time = None;
            }
        }
    }

    /// Record a failed operation
    pub fn record_failure(&self) {
        let mut state = self.state.write();

        state.failure_count += 1;
        state.last_failure_time = Some(Instant::now());

        match state.current_state {
            CircuitState::Closed => {
                if state.failure_count >= self.config.failure_threshold {
                    state.current_state = CircuitState::Open;
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in half-open state opens the circuit
                state.current_state = CircuitState::Open;
                state.half_open_calls = 0;
                state.success_count = 0;
            }
            CircuitState::Open => {
                // Already open, just update failure time
            }
        }
    }

    /// Get current circuit breaker state for monitoring
    pub fn get_state(&self) -> CircuitState {
        self.state.read().current_state
    }

    /// Get failure count for monitoring
    pub fn get_failure_count(&self) -> usize {
        self.state.read().failure_count
    }

    /// Get metrics for monitoring and alerting
    pub fn get_metrics(&self) -> CircuitBreakerMetrics {
        let state = self.state.read();
        CircuitBreakerMetrics {
            current_state: state.current_state,
            failure_count: state.failure_count,
            success_count: state.success_count,
            half_open_calls: state.half_open_calls,
            last_failure_time: state.last_failure_time,
        }
    }
}

/// Circuit breaker metrics for monitoring
#[derive(Debug, Clone)]
pub struct CircuitBreakerMetrics {
    pub current_state: CircuitState,
    pub failure_count: usize,
    pub success_count: usize,
    pub half_open_calls: usize,
    pub last_failure_time: Option<Instant>,
}

/// Circuit breaker errors
#[derive(Debug, Error)]
pub enum CircuitBreakerError {
    #[error("Circuit breaker is open, remaining timeout: {remaining_timeout:?}")]
    CircuitOpen { remaining_timeout: Duration },

    #[error("Half-open state call limit exceeded")]
    HalfOpenLimitExceeded,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_circuit_breaker_closed_state() {
        let breaker = CircuitBreaker::new(3, 5);

        // Should allow calls when closed
        assert!(breaker.check_health().is_ok());
        assert_eq!(breaker.get_state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_opens_after_failures() {
        let breaker = CircuitBreaker::new(2, 5);

        // Record failures to open circuit
        breaker.record_failure();
        assert_eq!(breaker.get_state(), CircuitState::Closed);

        breaker.record_failure();
        assert_eq!(breaker.get_state(), CircuitState::Open);

        // Should reject calls when open
        assert!(breaker.check_health().is_err());
    }

    #[test]
    fn test_circuit_breaker_half_open_recovery() {
        let breaker = CircuitBreaker::new(1, 0); // Immediate recovery

        // Open circuit
        breaker.record_failure();
        assert_eq!(breaker.get_state(), CircuitState::Open);

        // Sleep to allow recovery timeout
        thread::sleep(Duration::from_millis(10));

        // Should transition to half-open on next check
        assert!(breaker.check_health().is_ok());
        assert_eq!(breaker.get_state(), CircuitState::HalfOpen);

        // Record success to close circuit
        breaker.record_success();
        breaker.record_success();
        breaker.record_success();
        assert_eq!(breaker.get_state(), CircuitState::Closed);
    }
}
