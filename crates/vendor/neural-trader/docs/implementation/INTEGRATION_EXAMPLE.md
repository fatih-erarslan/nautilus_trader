# Timeout and Resource Limits - Integration Example

## Real-World Integration Example

Here's how to apply timeout and resource limits to an existing async NAPI function:

### Before (No Protection)

```rust
#[napi]
pub async fn neural_train(
    training_data: String,
    epochs: Option<u32>,
) -> Result<String> {
    let epochs = epochs.unwrap_or(50);
    
    // No validation - vulnerable to:
    // - Huge JSON inputs (memory exhaustion)
    // - Excessive epochs (infinite training)
    // - Hung operations (no timeout)
    
    train_model(&training_data, epochs).await?;
    
    Ok(json!({
        "status": "success",
        "epochs": epochs
    }).to_string())
}
```

**Vulnerabilities:**
- ❌ No input size limits → Memory exhaustion
- ❌ No parameter validation → Invalid epochs
- ❌ No timeout → Hung operations
- ❌ Poor error messages → Hard to debug

### After (Full Protection)

```rust
use crate::utils::{
    with_timeout,
    TIMEOUT_NEURAL_TRAIN,
    validate_neural_epochs,
    validate_json_size
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

    // Step 2: Execute with timeout protection
    let result = with_timeout(
        async {
            train_model(&training_data, epochs).await?;
            Ok(json!({
                "status": "success",
                "epochs": epochs
            }).to_string())
        },
        TIMEOUT_NEURAL_TRAIN,  // 300 seconds max
        "neural_train"          // For logging
    ).await.map_err(|e| napi::Error::from_reason(e.to_string()))?;

    Ok(result)
}
```

**Protections:**
- ✅ JSON size limited to 1MB
- ✅ Epochs limited to 10,000
- ✅ Operations timeout after 5 minutes
- ✅ Clear error messages

**Error Examples:**
```
"training_data exceeds maximum size of 1000000 bytes (got 1500000)"
"epochs exceeds maximum of 10000 epochs (got 15000)"
"neural_train timed out after 300s"
```

## More Examples

### Trading Operation

```rust
use crate::utils::{with_timeout, TIMEOUT_TRADING_OP, validate_positive};

#[napi]
pub async fn execute_trade(
    symbol: String,
    quantity: f64,
    price: f64,
) -> Result<String> {
    // Validate inputs
    validate_positive(quantity, "quantity")
        .map_err(|e| napi::Error::from_reason(e.to_string()))?;
    validate_positive(price, "price")
        .map_err(|e| napi::Error::from_reason(e.to_string()))?;

    // Execute with timeout
    with_timeout(
        async {
            place_order(&symbol, quantity, price).await
        },
        TIMEOUT_TRADING_OP,  // 30 seconds
        "execute_trade"
    ).await.map_err(|e| napi::Error::from_reason(e.to_string()))
}
```

### Risk Analysis

```rust
use crate::utils::{
    with_timeout, TIMEOUT_RISK_ANALYSIS,
    validate_portfolio_positions, validate_percentage
};

#[napi]
pub async fn risk_analysis(
    portfolio: String,
    var_confidence: f64,
) -> Result<String> {
    // Parse and validate
    let positions: Vec<Position> = serde_json::from_str(&portfolio)
        .map_err(|e| napi::Error::from_reason(e.to_string()))?;
    
    validate_portfolio_positions(positions.len(), "portfolio")
        .map_err(|e| napi::Error::from_reason(e.to_string()))?;
    validate_percentage(var_confidence, "var_confidence")
        .map_err(|e| napi::Error::from_reason(e.to_string()))?;

    // Execute with timeout
    with_timeout(
        async {
            calculate_var(&positions, var_confidence).await
        },
        TIMEOUT_RISK_ANALYSIS,  // 60 seconds
        "risk_analysis"
    ).await.map_err(|e| napi::Error::from_reason(e.to_string()))
}
```

### Sports Betting

```rust
use crate::utils::{
    with_timeout, TIMEOUT_SPORTS_BETTING,
    validate_positive, validate_percentage
};

#[napi]
pub async fn execute_bet(
    market_id: String,
    stake: f64,
    odds: f64,
) -> Result<String> {
    // Validate inputs
    validate_positive(stake, "stake")
        .map_err(|e| napi::Error::from_reason(e.to_string()))?;
    validate_positive(odds, "odds")
        .map_err(|e| napi::Error::from_reason(e.to_string()))?;

    // Execute with timeout
    with_timeout(
        async {
            place_bet(&market_id, stake, odds).await
        },
        TIMEOUT_SPORTS_BETTING,  // 30 seconds
        "execute_bet"
    ).await.map_err(|e| napi::Error::from_reason(e.to_string()))
}
```

## Performance Comparison

### Without Protection
```
Input: 10MB JSON
Result: ❌ Out of memory crash (10GB+ consumed)
Time: N/A (crashed)
```

### With Protection
```
Input: 10MB JSON
Result: ✅ "training_data exceeds maximum size of 1000000 bytes (got 10485760)"
Time: < 1ms (immediate rejection)
```

### Without Timeout
```
Input: Slow API call
Result: ❌ Hung forever, blocked thread
Time: ∞
```

### With Timeout
```
Input: Slow API call
Result: ✅ "execute_trade timed out after 30s"
Time: 30s (clean failure)
```

## Summary

Adding timeout and validation is straightforward:

1. Import utilities
2. Add validation before processing
3. Wrap async logic with timeout
4. Convert errors to NAPI errors

The overhead is negligible (< 1ms) but the protection is invaluable.
