/// Comprehensive circuit breaker tests
///
/// These tests verify circuit breaker functionality including:
/// - State transitions
/// - Failure detection and recovery
/// - Timeout handling
/// - Concurrent operations
/// - Integration scenarios

use neural_trader_napi::resilience::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerRegistry,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;
use anyhow::{anyhow, Result};

#[tokio::test]
async fn test_basic_success_flow() {
    let cb = CircuitBreaker::new(
        "test_basic".to_string(),
        CircuitBreakerConfig::default(),
    );

    // Execute successful operation
    let result = cb.call(async { Ok::<i32, anyhow::Error>(42) }).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 42);

    // Verify metrics
    let metrics = cb.get_metrics().await;
    assert_eq!(metrics.total_calls, 1);
    assert_eq!(metrics.successful_calls, 1);
    assert_eq!(metrics.failed_calls, 0);
}

#[tokio::test]
async fn test_circuit_opens_after_threshold() {
    let cb = CircuitBreaker::new(
        "test_opens".to_string(),
        CircuitBreakerConfig {
            failure_threshold: 3,
            success_threshold: 2,
            timeout: Duration::from_secs(1),
            reset_timeout: Duration::from_secs(10),
        },
    );

    // Trigger failures up to threshold
    for i in 0..3 {
        let result = cb.call(async { Err::<(), _>(anyhow!("failure {}", i)) }).await;
        assert!(result.is_err());
    }

    // Verify circuit is open
    let state = cb.get_state().await;
    assert!(state.starts_with("OPEN"), "Expected OPEN state, got: {}", state);

    // Verify metrics
    let metrics = cb.get_metrics().await;
    assert_eq!(metrics.times_opened, 1);
    assert_eq!(metrics.failed_calls, 3);
}

#[tokio::test]
async fn test_open_circuit_rejects_calls() {
    let cb = CircuitBreaker::new(
        "test_rejects".to_string(),
        CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout: Duration::from_secs(1),
            reset_timeout: Duration::from_secs(10),
        },
    );

    // Open the circuit
    for _ in 0..2 {
        let _ = cb.call(async { Err::<(), _>(anyhow!("failure")) }).await;
    }

    // Attempt operation on open circuit
    let result = cb.call(async { Ok::<i32, anyhow::Error>(42) }).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("is OPEN"));

    // Verify rejection was tracked
    let metrics = cb.get_metrics().await;
    assert_eq!(metrics.rejected_calls, 1);
}

#[tokio::test]
async fn test_half_open_transition() {
    let cb = CircuitBreaker::new(
        "test_half_open".to_string(),
        CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout: Duration::from_secs(1),
            reset_timeout: Duration::from_millis(100),
        },
    );

    // Open the circuit
    for _ in 0..2 {
        let _ = cb.call(async { Err::<(), _>(anyhow!("failure")) }).await;
    }

    assert!(cb.get_state().await.starts_with("OPEN"));

    // Wait for reset timeout
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Next call should transition to half-open
    let result = cb.call(async { Ok::<i32, anyhow::Error>(1) }).await;

    // State should be half-open or closed (if success threshold is 1)
    let state = cb.get_state().await;
    assert!(
        state.starts_with("HALF_OPEN") || state.starts_with("CLOSED"),
        "Unexpected state: {}",
        state
    );

    // Verify metrics
    let metrics = cb.get_metrics().await;
    assert_eq!(metrics.times_half_opened, 1);
}

#[tokio::test]
async fn test_half_open_to_closed() {
    let cb = CircuitBreaker::new(
        "test_recovery".to_string(),
        CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout: Duration::from_secs(1),
            reset_timeout: Duration::from_millis(100),
        },
    );

    // Open the circuit
    for _ in 0..2 {
        let _ = cb.call(async { Err::<(), _>(anyhow!("failure")) }).await;
    }

    // Wait for reset timeout
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Execute successful operations to close circuit
    for _ in 0..2 {
        let result = cb.call(async { Ok::<i32, anyhow::Error>(42) }).await;
        assert!(result.is_ok());
    }

    // Verify circuit is closed
    let state = cb.get_state().await;
    assert!(state.starts_with("CLOSED"), "Expected CLOSED state, got: {}", state);
}

#[tokio::test]
async fn test_half_open_failure_reopens() {
    let cb = CircuitBreaker::new(
        "test_reopen".to_string(),
        CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout: Duration::from_secs(1),
            reset_timeout: Duration::from_millis(100),
        },
    );

    // Open the circuit
    for _ in 0..2 {
        let _ = cb.call(async { Err::<(), _>(anyhow!("failure")) }).await;
    }

    // Wait for reset timeout
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Fail in half-open state
    let result = cb.call(async { Err::<(), _>(anyhow!("failure in half-open")) }).await;
    assert!(result.is_err());

    // Verify circuit reopened
    let state = cb.get_state().await;
    assert!(state.starts_with("OPEN"), "Expected OPEN state, got: {}", state);

    // Verify it opened twice
    let metrics = cb.get_metrics().await;
    assert_eq!(metrics.times_opened, 2);
}

#[tokio::test]
async fn test_timeout_handling() {
    let cb = CircuitBreaker::new(
        "test_timeout".to_string(),
        CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 2,
            timeout: Duration::from_millis(50),
            reset_timeout: Duration::from_secs(10),
        },
    );

    // Execute operation that exceeds timeout
    let result = cb
        .call(async {
            tokio::time::sleep(Duration::from_millis(200)).await;
            Ok::<i32, anyhow::Error>(42)
        })
        .await;

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("timed out"));

    // Verify circuit opened due to timeout
    let state = cb.get_state().await;
    assert!(state.starts_with("OPEN"));
}

#[tokio::test]
async fn test_concurrent_operations() {
    let cb = Arc::new(CircuitBreaker::new(
        "test_concurrent".to_string(),
        CircuitBreakerConfig {
            failure_threshold: 10,
            success_threshold: 2,
            timeout: Duration::from_secs(1),
            reset_timeout: Duration::from_secs(5),
        },
    ));

    let counter = Arc::new(AtomicU32::new(0));
    let mut handles = vec![];

    // Spawn 20 concurrent operations
    for i in 0..20 {
        let cb_clone = cb.clone();
        let counter_clone = counter.clone();
        let handle = tokio::spawn(async move {
            let result = cb_clone
                .call(async move {
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    Ok::<i32, anyhow::Error>(i)
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

    // All should succeed
    assert_eq!(results.iter().filter(|&&r| r).count(), 20);
    assert_eq!(counter.load(Ordering::SeqCst), 20);

    // Verify metrics
    let metrics = cb.get_metrics().await;
    assert_eq!(metrics.total_calls, 20);
    assert_eq!(metrics.successful_calls, 20);
}

#[tokio::test]
async fn test_mixed_success_failure() {
    let cb = CircuitBreaker::new(
        "test_mixed".to_string(),
        CircuitBreakerConfig {
            failure_threshold: 5,
            success_threshold: 2,
            timeout: Duration::from_secs(1),
            reset_timeout: Duration::from_secs(5),
        },
    );

    // Mix of successes and failures
    let _ = cb.call(async { Ok::<(), _>(()) }).await; // Success
    let _ = cb.call(async { Err::<(), _>(anyhow!("fail")) }).await; // Fail
    let _ = cb.call(async { Ok::<(), _>(()) }).await; // Success
    let _ = cb.call(async { Err::<(), _>(anyhow!("fail")) }).await; // Fail
    let _ = cb.call(async { Ok::<(), _>(()) }).await; // Success

    // Circuit should still be closed
    let state = cb.get_state().await;
    assert!(state.starts_with("CLOSED"));

    // Verify metrics
    let metrics = cb.get_metrics().await;
    assert_eq!(metrics.total_calls, 5);
    assert_eq!(metrics.successful_calls, 3);
    assert_eq!(metrics.failed_calls, 2);
    assert_eq!(cb.get_success_rate().await, 60.0);
}

#[tokio::test]
async fn test_registry_operations() {
    let registry = CircuitBreakerRegistry::new();

    // Register circuit breaker
    let config = CircuitBreakerConfig::default();
    let cb = registry.register("test".to_string(), config).await;

    // Get circuit breaker
    let retrieved = registry.get("test").await;
    assert!(retrieved.is_some());

    // List all breakers
    let names = registry.list_names().await;
    assert!(names.contains(&"test".to_string()));

    // Get all metrics
    let all_metrics = registry.get_all_metrics().await;
    assert!(all_metrics.contains_key("test"));

    // Remove circuit breaker
    assert!(registry.remove("test").await);
    assert!(registry.get("test").await.is_none());
}

#[tokio::test]
async fn test_registry_get_or_create() {
    let registry = CircuitBreakerRegistry::new();
    let config = CircuitBreakerConfig::default();

    // First call creates
    let cb1 = registry.get_or_create("test".to_string(), config.clone()).await;

    // Execute operation to change state
    let _ = cb1.call(async { Ok::<(), _>(()) }).await;

    // Second call retrieves same instance
    let cb2 = registry.get_or_create("test".to_string(), config).await;

    // States should match
    assert_eq!(cb1.get_state().await, cb2.get_state().await);
}

#[tokio::test]
async fn test_metrics_tracking() {
    let cb = CircuitBreaker::new(
        "test_metrics".to_string(),
        CircuitBreakerConfig {
            failure_threshold: 10,
            success_threshold: 2,
            timeout: Duration::from_secs(1),
            reset_timeout: Duration::from_secs(5),
        },
    );

    // Execute various operations
    for i in 0..10 {
        if i % 2 == 0 {
            let _ = cb.call(async { Ok::<(), _>(()) }).await;
        } else {
            let _ = cb.call(async { Err::<(), _>(anyhow!("fail")) }).await;
        }
    }

    let metrics = cb.get_metrics().await;
    assert_eq!(metrics.total_calls, 10);
    assert_eq!(metrics.successful_calls, 5);
    assert_eq!(metrics.failed_calls, 5);
    assert_eq!(cb.get_success_rate().await, 50.0);

    // Reset metrics
    cb.reset_metrics().await;
    let metrics = cb.get_metrics().await;
    assert_eq!(metrics.total_calls, 0);
}

#[tokio::test]
async fn test_rapid_state_changes() {
    let cb = CircuitBreaker::new(
        "test_rapid".to_string(),
        CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout: Duration::from_secs(1),
            reset_timeout: Duration::from_millis(50),
        },
    );

    // Open circuit
    for _ in 0..2 {
        let _ = cb.call(async { Err::<(), _>(anyhow!("fail")) }).await;
    }
    assert!(cb.get_state().await.starts_with("OPEN"));

    // Wait and recover
    tokio::time::sleep(Duration::from_millis(100)).await;
    for _ in 0..2 {
        let _ = cb.call(async { Ok::<(), _>(()) }).await;
    }
    assert!(cb.get_state().await.starts_with("CLOSED"));

    // Open again
    for _ in 0..2 {
        let _ = cb.call(async { Err::<(), _>(anyhow!("fail")) }).await;
    }
    assert!(cb.get_state().await.starts_with("OPEN"));

    let metrics = cb.get_metrics().await;
    assert_eq!(metrics.times_opened, 2);
}

#[tokio::test]
async fn test_error_message_preservation() {
    let cb = CircuitBreaker::new(
        "test_errors".to_string(),
        CircuitBreakerConfig::default(),
    );

    let custom_error = "Custom error message";
    let result = cb.call(async { Err::<(), _>(anyhow!(custom_error)) }).await;

    assert!(result.is_err());
    let error_string = result.unwrap_err().to_string();
    assert!(error_string.contains(custom_error) || error_string.contains("test_errors"));
}
