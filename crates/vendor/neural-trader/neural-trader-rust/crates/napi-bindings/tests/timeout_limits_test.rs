use neural_trader_backend::utils::{
    with_timeout, validate_json_size, validate_array_length, validate_string_length,
    validate_swarm_agents, validate_neural_epochs, validate_positive, validate_percentage,
    MAX_JSON_SIZE, MAX_ARRAY_LENGTH, MAX_STRING_LENGTH, MAX_SWARM_AGENTS, MAX_NEURAL_EPOCHS,
};
use tokio::time::{sleep, Duration};
use anyhow::{anyhow, Result};

#[tokio::test]
async fn test_timeout_success() {
    let result = with_timeout(
        async {
            sleep(Duration::from_millis(100)).await;
            Ok::<_, anyhow::Error>(42)
        },
        1,
        "test operation"
    ).await;

    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 42);
}

#[tokio::test]
async fn test_timeout_failure() {
    let result = with_timeout(
        async {
            sleep(Duration::from_secs(3)).await;
            Ok::<_, anyhow::Error>(42)
        },
        1,
        "slow operation"
    ).await;

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("timed out"));
}

#[tokio::test]
async fn test_timeout_propagates_error() {
    let result = with_timeout(
        async {
            Err::<i32, _>(anyhow!("original error"))
        },
        1,
        "error operation"
    ).await;

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("original error"));
}

#[test]
fn test_validate_json_size_ok() {
    let json = r#"{"key": "value"}"#;
    assert!(validate_json_size(json, "test_json").is_ok());
}

#[test]
fn test_validate_json_size_too_large() {
    let json = "x".repeat(MAX_JSON_SIZE + 1);
    let result = validate_json_size(&json, "large_json");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("exceeds maximum size"));
}

#[test]
fn test_validate_array_length_ok() {
    assert!(validate_array_length(100, "test_array").is_ok());
}

#[test]
fn test_validate_array_length_too_large() {
    let result = validate_array_length(MAX_ARRAY_LENGTH + 1, "large_array");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("exceeds maximum length"));
}

#[test]
fn test_validate_string_length_ok() {
    let s = "hello world";
    assert!(validate_string_length(s, "test_string").is_ok());
}

#[test]
fn test_validate_string_length_too_long() {
    let s = "x".repeat(MAX_STRING_LENGTH + 1);
    let result = validate_string_length(&s, "long_string");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("exceeds maximum length"));
}

#[test]
fn test_validate_swarm_agents_ok() {
    assert!(validate_swarm_agents(10, "test_swarm").is_ok());
}

#[test]
fn test_validate_swarm_agents_too_many() {
    let result = validate_swarm_agents(MAX_SWARM_AGENTS + 1, "large_swarm");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("exceeds maximum"));
}

#[test]
fn test_validate_neural_epochs_ok() {
    assert!(validate_neural_epochs(100, "test_epochs").is_ok());
}

#[test]
fn test_validate_neural_epochs_too_many() {
    let result = validate_neural_epochs(MAX_NEURAL_EPOCHS + 1, "many_epochs");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("exceeds maximum"));
}

#[test]
fn test_validate_positive_ok() {
    assert!(validate_positive(1.0, "test_positive").is_ok());
    assert!(validate_positive(0.001, "small_positive").is_ok());
}

#[test]
fn test_validate_positive_invalid() {
    assert!(validate_positive(0.0, "zero").is_err());
    assert!(validate_positive(-1.0, "negative").is_err());
}

#[test]
fn test_validate_percentage_ok() {
    assert!(validate_percentage(0.0, "test_zero").is_ok());
    assert!(validate_percentage(0.5, "test_half").is_ok());
    assert!(validate_percentage(1.0, "test_one").is_ok());
}

#[test]
fn test_validate_percentage_invalid() {
    assert!(validate_percentage(-0.1, "negative_pct").is_err());
    assert!(validate_percentage(1.1, "over_one").is_err());
}

#[tokio::test]
async fn test_multiple_concurrent_timeouts() {
    let results = tokio::join!(
        with_timeout(
            async {
                sleep(Duration::from_millis(100)).await;
                Ok::<_, anyhow::Error>("first")
            },
            1,
            "op1"
        ),
        with_timeout(
            async {
                sleep(Duration::from_millis(200)).await;
                Ok::<_, anyhow::Error>("second")
            },
            1,
            "op2"
        ),
        with_timeout(
            async {
                sleep(Duration::from_secs(2)).await;
                Ok::<_, anyhow::Error>("third")
            },
            1,
            "op3"
        )
    );

    assert!(results.0.is_ok());
    assert!(results.1.is_ok());
    assert!(results.2.is_err()); // This one times out
}
