# Timeout and Resource Limits - Implementation Guide

## Quick Reference

### 1. Import Required Utilities

```rust
use crate::utils::{
    with_timeout,
    TIMEOUT_NEURAL_TRAIN,
    TIMEOUT_TRADING_OP,
    validate_json_size,
    validate_positive,
};
```

### 2. Basic Timeout Pattern

```rust
#[napi]
pub async fn my_async_function(param: String) -> Result<String> {
    // Wrap async logic with timeout
    let result = with_timeout(
        async {
            // Your async logic here
            do_something().await?;
            Ok(result_value)
        },
        TIMEOUT_TRADING_OP,  // Choose appropriate timeout constant
        "my_async_function"   // Operation name for logging
    ).await.map_err(|e| napi::Error::from_reason(e.to_string()))?;

    Ok(result)
}
```

### 3. Input Validation Pattern

```rust
#[napi]
pub async fn validated_function(
    json_data: String,
    amount: f64,
    percentage: f64,
) -> Result<String> {
    // Validate inputs BEFORE processing
    validate_json_size(&json_data, "json_data")
        .map_err(|e| napi::Error::from_reason(e.to_string()))?;

    validate_positive(amount, "amount")
        .map_err(|e| napi::Error::from_reason(e.to_string()))?;

    validate_percentage(percentage, "percentage")
        .map_err(|e| napi::Error::from_reason(e.to_string()))?;

    // Now safe to proceed with validated inputs
    // ...
}
```

### 4. Complete Example

```rust
use crate::utils::{
    with_timeout, TIMEOUT_NEURAL_TRAIN,
    validate_neural_epochs, validate_json_size
};

#[napi]
pub async fn neural_train(
    training_data: String,
    epochs: Option<u32>,
) -> Result<String> {
    let epochs = epochs.unwrap_or(50);

    // Step 1: Validate all inputs
    validate_json_size(&training_data, "training_data")
        .map_err(|e| napi::Error::from_reason(e.to_string()))?;

    validate_neural_epochs(epochs, "epochs")
        .map_err(|e| napi::Error::from_reason(e.to_string()))?;

    // Step 2: Execute with timeout
    let result = with_timeout(
        async {
            // Your training logic
            train_model(&training_data, epochs).await?;
            Ok(json!({
                "status": "success",
                "epochs": epochs,
                "duration_ms": 1234
            }).to_string())
        },
        TIMEOUT_NEURAL_TRAIN,
        "neural_train"
    ).await.map_err(|e| napi::Error::from_reason(e.to_string()))?;

    Ok(result)
}
```

## Available Timeout Constants

```rust
TIMEOUT_API_CALL           // 10s  - General API calls
TIMEOUT_TRADING_OP         // 30s  - Trading operations
TIMEOUT_NEURAL_TRAIN       // 300s - Neural network training
TIMEOUT_BACKTEST           // 120s - Backtesting
TIMEOUT_E2B_OPERATION      // 60s  - E2B sandbox ops
TIMEOUT_SPORTS_BETTING     // 30s  - Sports betting
TIMEOUT_SYNDICATE_OP       // 30s  - Syndicate operations
TIMEOUT_PREDICTION_MARKET  // 30s  - Prediction markets
TIMEOUT_RISK_ANALYSIS      // 60s  - Risk analysis
TIMEOUT_NEWS_FETCH         // 20s  - News fetching
```

## Available Validation Functions

### Size and Length
```rust
validate_json_size(json: &str, field: &str) -> Result<()>
validate_array_length(len: usize, field: &str) -> Result<()>
validate_string_length(s: &str, field: &str) -> Result<()>
```

### Numeric Values
```rust
validate_positive(value: f64, field: &str) -> Result<()>
validate_non_negative(value: f64, field: &str) -> Result<()>
validate_percentage(value: f64, field: &str) -> Result<()>
```

### Domain-Specific
```rust
validate_swarm_agents(count: u32, field: &str) -> Result<()>
validate_portfolio_positions(count: usize, field: &str) -> Result<()>
validate_backtest_days(days: u32, field: &str) -> Result<()>
validate_neural_epochs(epochs: u32, field: &str) -> Result<()>
validate_syndicate_members(count: u32, field: &str) -> Result<()>
validate_batch_size(size: usize, field: &str) -> Result<()>
validate_symbols_count(count: usize, field: &str) -> Result<()>
validate_trades_count(count: usize, field: &str) -> Result<()>
validate_betting_opportunities(count: usize, field: &str) -> Result<()>
```

## Resource Limits Reference

| Constant | Value | Purpose |
|----------|-------|---------|
| MAX_JSON_SIZE | 1,000,000 | 1MB JSON input limit |
| MAX_ARRAY_LENGTH | 10,000 | Maximum array elements |
| MAX_STRING_LENGTH | 100,000 | Maximum string characters |
| MAX_SWARM_AGENTS | 100 | Maximum swarm agents |
| MAX_CONCURRENT_REQUESTS | 1,000 | Concurrent request limit |
| MAX_PORTFOLIO_POSITIONS | 10,000 | Portfolio position limit |
| MAX_BACKTEST_DAYS | 3,650 | 10 years of backtest data |
| MAX_NEURAL_EPOCHS | 10,000 | Neural training epochs |
| MAX_SYNDICATE_MEMBERS | 1,000 | Syndicate member limit |
| MAX_BATCH_SIZE | 1,000 | Batch operation size |
| MAX_SYMBOLS | 100 | Symbols per request |
| MAX_TRADES_PER_REQUEST | 50 | Trades per request |
| MAX_BETTING_OPPORTUNITIES | 100 | Betting opportunities |

## Migration Checklist

When adding timeout/validation to an existing function:

- [ ] Import timeout wrapper and relevant validators
- [ ] Add validation for all string inputs
- [ ] Add validation for all numeric inputs
- [ ] Add validation for all array/collection inputs
- [ ] Wrap async logic with `with_timeout`
- [ ] Choose appropriate timeout constant
- [ ] Convert errors to NAPI errors
- [ ] Add tests for validation
- [ ] Add tests for timeout
- [ ] Update documentation

## Testing Your Implementation

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_function_success() {
        let result = my_function("valid_input".to_string()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_function_timeout() {
        // Test with operation that will timeout
        // ...
    }

    #[test]
    fn test_input_validation() {
        // Test validation logic
        // ...
    }
}
```

## Common Patterns

### Pattern 1: Multiple Validations
```rust
// Validate all inputs before processing
validate_json_size(&config, "config")?;
validate_positive(amount, "amount")?;
validate_percentage(threshold, "threshold")?;
validate_symbols_count(symbols.len(), "symbols")?;
```

### Pattern 2: Optional Parameters with Defaults
```rust
let epochs = epochs.unwrap_or(50);
let use_gpu = use_gpu.unwrap_or(true);

validate_neural_epochs(epochs, "epochs")?;
```

### Pattern 3: Nested Async Operations
```rust
with_timeout(
    async {
        let step1 = do_step1().await?;
        let step2 = do_step2(step1).await?;
        let step3 = do_step3(step2).await?;
        Ok(step3)
    },
    TIMEOUT_TRADING_OP,
    "multi_step_operation"
).await?
```

### Pattern 4: Conditional Timeout
```rust
let timeout_duration = if use_gpu {
    TIMEOUT_API_CALL  // GPU is fast
} else {
    TIMEOUT_NEURAL_TRAIN  // CPU is slow
};

with_timeout(operation, timeout_duration, "neural_operation").await?
```

## Error Messages

Good error messages include:
- What failed
- What the limit is
- What was provided

Examples:
```
"training_data exceeds maximum size of 1000000 bytes (got 1500000)"
"epochs exceeds maximum of 10000 epochs (got 15000)"
"price must be positive (got -10.5)"
"allocation must be between 0 and 1 (got 1.5)"
"neural_train timed out after 300s"
```

## Performance Impact

- Validation: < 1ms overhead (negligible)
- Timeout wrapper: < 1ms overhead (negligible)
- Total impact: < 1% for most operations
- Benefits: Protection from 100% CPU/memory exhaustion

## Next Steps

1. Review all async functions in your module
2. Add timeout wrappers where missing
3. Add input validation where missing
4. Write tests for edge cases
5. Update documentation

## Need Help?

- Check `/workspaces/neural-trader/neural-trader-rust/crates/napi-bindings/src/utils/`
- See tests in `/workspaces/neural-trader/neural-trader-rust/crates/napi-bindings/tests/timeout_limits_test.rs`
- Run example: `cargo run --example timeout_usage`
