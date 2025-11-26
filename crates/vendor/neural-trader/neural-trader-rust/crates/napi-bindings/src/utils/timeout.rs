use tokio::time::{timeout, Duration};
use anyhow::{anyhow, Result};
use std::future::Future;

/// Wraps a future with a timeout, returning an error if the operation takes too long
pub async fn with_timeout<F, T>(
    future: F,
    seconds: u64,
    operation: &str
) -> Result<T>
where
    F: Future<Output = Result<T>>,
{
    match timeout(Duration::from_secs(seconds), future).await {
        Ok(result) => result,
        Err(_) => {
            log::error!("{} timed out after {}s", operation, seconds);
            Err(anyhow!("{} timed out after {}s", operation, seconds))
        }
    }
}

// Timeout constants (in seconds)
/// Timeout for general API calls
pub const TIMEOUT_API_CALL: u64 = 10;

/// Timeout for trading operations (execution, portfolio operations)
pub const TIMEOUT_TRADING_OP: u64 = 30;

/// Timeout for neural network training
pub const TIMEOUT_NEURAL_TRAIN: u64 = 300;

/// Timeout for backtesting operations
pub const TIMEOUT_BACKTEST: u64 = 120;

/// Timeout for E2B sandbox operations
pub const TIMEOUT_E2B_OPERATION: u64 = 60;

/// Timeout for sports betting operations
pub const TIMEOUT_SPORTS_BETTING: u64 = 30;

/// Timeout for syndicate operations
pub const TIMEOUT_SYNDICATE_OP: u64 = 30;

/// Timeout for prediction market operations
pub const TIMEOUT_PREDICTION_MARKET: u64 = 30;

/// Timeout for risk analysis operations
pub const TIMEOUT_RISK_ANALYSIS: u64 = 60;

/// Timeout for news fetching operations
pub const TIMEOUT_NEWS_FETCH: u64 = 20;

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

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
                sleep(Duration::from_secs(2)).await;
                Ok::<_, anyhow::Error>(42)
            },
            1,
            "slow operation"
        ).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("timed out"));
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
}
