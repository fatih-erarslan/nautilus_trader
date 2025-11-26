# Circuit Breaker Pattern Implementation

## Overview

This document describes the comprehensive circuit breaker implementation for building resilient, fault-tolerant operations in the Neural Trader system.

## Architecture

### Circuit States

The circuit breaker operates in three states:

1. **CLOSED (Normal Operation)**
   - All requests pass through
   - Failures are tracked
   - Opens when failure threshold is reached

2. **OPEN (Fault Detected)**
   - All requests are immediately rejected
   - System waits for reset timeout
   - Transitions to half-open after timeout

3. **HALF_OPEN (Testing Recovery)**
   - Limited requests allowed through
   - Successful requests close the circuit
   - Any failure reopens the circuit

### State Transitions

```
CLOSED --[failures >= threshold]--> OPEN
OPEN --[reset timeout elapsed]--> HALF_OPEN
HALF_OPEN --[success threshold met]--> CLOSED
HALF_OPEN --[any failure]--> OPEN
```

## Configuration

### CircuitBreakerConfig

```rust
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
```

### Default Configuration

```rust
CircuitBreakerConfig {
    failure_threshold: 5,
    success_threshold: 2,
    timeout: Duration::from_secs(30),
    reset_timeout: Duration::from_secs(60),
}
```

## Usage Examples

### Basic Usage

```rust
use neural_trader_napi::resilience::{CircuitBreaker, CircuitBreakerConfig};
use std::time::Duration;

#[tokio::main]
async fn main() {
    let cb = CircuitBreaker::new(
        "my_service".to_string(),
        CircuitBreakerConfig::default(),
    );

    // Execute operation through circuit breaker
    let result = cb.call(async {
        // Your operation here
        Ok::<i32, anyhow::Error>(42)
    }).await;

    match result {
        Ok(value) => println!("Success: {}", value),
        Err(e) => println!("Failed: {}", e),
    }
}
```

### External API Calls

```rust
use neural_trader_napi::resilience::ApiCircuitBreaker;

let api_cb = ApiCircuitBreaker::new("market_data_api");

// GET request
let data: MarketData = api_cb
    .get("https://api.example.com/market/data")
    .await?;

// POST request
let response: OrderResponse = api_cb
    .post("https://api.example.com/orders", &order_request)
    .await?;

// Check circuit state
println!("API Circuit State: {}", api_cb.get_state().await);
```

### E2B Sandbox Operations

```rust
use neural_trader_napi::resilience::E2BSandboxCircuitBreaker;

let sandbox_cb = E2BSandboxCircuitBreaker::new();

// Create sandbox
let sandbox_id = sandbox_cb
    .create_sandbox("base")
    .await?;

// Execute code
let result = sandbox_cb
    .execute_code(&sandbox_id, "print('hello')")
    .await?;

// Stop sandbox
sandbox_cb.stop_sandbox(&sandbox_id).await?;
```

### Neural Network Operations

```rust
use neural_trader_napi::resilience::NeuralCircuitBreaker;

let neural_cb = NeuralCircuitBreaker::new("price_predictor");

// Perform inference
let prediction = neural_cb
    .predict(vec![1.0, 2.0, 3.0])
    .await?;

// Train model
neural_cb
    .train(training_data, labels)
    .await?;
```

### Database Operations

```rust
use neural_trader_napi::resilience::DatabaseCircuitBreaker;

let db_cb = DatabaseCircuitBreaker::new("trading_db");

// Execute query
let users: Vec<User> = db_cb
    .query("SELECT * FROM users")
    .await?;

// Execute transaction
db_cb.transaction(|| {
    // Transaction logic
    Ok(())
}).await?;
```

### Complete Trading System

```rust
use neural_trader_napi::resilience::TradingSystemCircuitBreakers;

let system = TradingSystemCircuitBreakers::new();

// Use different circuit breakers for different services
let market_data = system.market_data_api
    .get("/v1/quotes/AAPL")
    .await?;

let prediction = system.neural_predictor
    .predict(feature_vector)
    .await?;

let order_result = system.order_execution_api
    .post("/v1/orders", &order)
    .await?;

// Check overall system health
let health = system.health_status().await;
for (service, state) in health {
    println!("{}: {}", service, state);
}
```

## Circuit Breaker Registry

For managing multiple circuit breakers across your application:

```rust
use neural_trader_napi::resilience::CircuitBreakerRegistry;

let registry = CircuitBreakerRegistry::new();

// Register circuit breakers
let api_cb = registry.register(
    "api".to_string(),
    CircuitBreakerConfig::default(),
).await;

let db_cb = registry.register(
    "database".to_string(),
    CircuitBreakerConfig {
        failure_threshold: 3,
        ..Default::default()
    },
).await;

// Get existing circuit breaker
if let Some(cb) = registry.get("api").await {
    let result = cb.call(async { /* ... */ }).await;
}

// Get or create
let cb = registry.get_or_create(
    "cache".to_string(),
    CircuitBreakerConfig::default(),
).await;

// List all circuit breakers
let names = registry.list_names().await;

// Get all metrics
let all_metrics = registry.get_all_metrics().await;
```

## Metrics and Monitoring

### Available Metrics

```rust
pub struct CircuitBreakerMetrics {
    pub total_calls: u64,
    pub successful_calls: u64,
    pub failed_calls: u64,
    pub rejected_calls: u64,
    pub total_open_time_ms: u64,
    pub times_opened: u32,
    pub times_half_opened: u32,
}
```

### Accessing Metrics

```rust
// Get current metrics
let metrics = cb.get_metrics().await;
println!("Total calls: {}", metrics.total_calls);
println!("Success rate: {}%",
    (metrics.successful_calls as f64 / metrics.total_calls as f64) * 100.0
);

// Get success rate directly
let success_rate = cb.get_success_rate().await;
println!("Success rate: {}%", success_rate);

// Reset metrics
cb.reset_metrics().await;
```

## Best Practices

### 1. Choose Appropriate Thresholds

```rust
// For critical services - more tolerant
CircuitBreakerConfig {
    failure_threshold: 10,
    success_threshold: 3,
    timeout: Duration::from_secs(5),
    reset_timeout: Duration::from_secs(30),
}

// For non-critical services - fail faster
CircuitBreakerConfig {
    failure_threshold: 3,
    success_threshold: 2,
    timeout: Duration::from_secs(2),
    reset_timeout: Duration::from_secs(60),
}
```

### 2. Set Timeouts Appropriately

- **Fast operations** (local calls, cache): 1-5 seconds
- **Medium operations** (API calls): 10-30 seconds
- **Slow operations** (ML inference, batch jobs): 30-120 seconds

### 3. Monitor Circuit State

```rust
// Log state changes
let state = cb.get_state().await;
log::info!("Circuit breaker '{}' state: {}", name, state);

// Alert on open circuits
if state.starts_with("OPEN") {
    alert_ops_team(&format!("Circuit {} is OPEN", name));
}
```

### 4. Use Registry for Global Management

```rust
// Centralized registry
static CIRCUIT_BREAKERS: Lazy<CircuitBreakerRegistry> =
    Lazy::new(|| CircuitBreakerRegistry::new());

// Access from anywhere
async fn make_api_call() -> Result<Response> {
    let cb = CIRCUIT_BREAKERS.get_or_create(
        "api".to_string(),
        CircuitBreakerConfig::default(),
    ).await;

    cb.call(async { /* API call */ }).await
}
```

### 5. Handle Failures Gracefully

```rust
match cb.call(operation).await {
    Ok(result) => Ok(result),
    Err(e) if e.to_string().contains("is OPEN") => {
        // Circuit is open - use fallback
        log::warn!("Circuit open, using cached data");
        Ok(get_cached_data()?)
    }
    Err(e) => {
        // Actual operation failure
        log::error!("Operation failed: {}", e);
        Err(e)
    }
}
```

## Testing

### Unit Tests

The implementation includes comprehensive tests:

```bash
# Run circuit breaker tests
cargo test -p neural-trader-napi circuit_breaker

# Run with output
cargo test -p neural-trader-napi circuit_breaker -- --nocapture
```

### Integration Tests

```rust
#[tokio::test]
async fn test_end_to_end_scenario() {
    let system = TradingSystemCircuitBreakers::new();

    // Simulate failures
    for _ in 0..5 {
        let _ = system.market_data_api
            .get("/v1/quotes/FAIL")
            .await;
    }

    // Verify circuit opened
    let health = system.health_status().await;
    assert!(health.iter()
        .find(|(name, _)| name == "market_data_api")
        .unwrap()
        .1
        .starts_with("OPEN"));
}
```

## Performance Considerations

### Overhead

- **Closed state**: ~1-2µs per call
- **Open state**: ~100ns per call (immediate rejection)
- **Half-open state**: ~1-2µs per call

### Memory Usage

- Each circuit breaker: ~200 bytes
- Thread-safe with Arc and RwLock
- Minimal allocation during operation

### Concurrency

- Fully thread-safe
- Lock-free reads for state checks
- Write locks only on state transitions

## Troubleshooting

### Circuit Opens Too Frequently

```rust
// Increase failure threshold
CircuitBreakerConfig {
    failure_threshold: 10,  // Was 5
    ..Default::default()
}

// Increase timeout
CircuitBreakerConfig {
    timeout: Duration::from_secs(60),  // Was 30
    ..Default::default()
}
```

### Circuit Never Opens

```rust
// Check if failures are being reported
let metrics = cb.get_metrics().await;
println!("Failed calls: {}", metrics.failed_calls);

// Reduce failure threshold
CircuitBreakerConfig {
    failure_threshold: 3,  // Was 5
    ..Default::default()
}
```

### Circuit Doesn't Close After Recovery

```rust
// Check half-open transition
let metrics = cb.get_metrics().await;
println!("Times half-opened: {}", metrics.times_half_opened);

// Reduce success threshold
CircuitBreakerConfig {
    success_threshold: 1,  // Was 2
    ..Default::default()
}

// Check reset timeout
CircuitBreakerConfig {
    reset_timeout: Duration::from_secs(30),  // Was 60
    ..Default::default()
}
```

## Future Enhancements

Potential improvements for future versions:

1. **Adaptive Thresholds**: Automatically adjust thresholds based on historical patterns
2. **Exponential Backoff**: Increase reset timeout on repeated failures
3. **Bulkhead Pattern**: Combine with resource pooling
4. **Metrics Export**: Prometheus/StatsD integration
5. **Distributed Coordination**: Share circuit state across instances

## References

- [Release It!](https://pragprog.com/titles/mnee2/release-it-second-edition/) by Michael Nygard
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html) by Martin Fowler
- [Netflix Hystrix](https://github.com/Netflix/Hystrix) (inspiration)

## License

Part of the Neural Trader project. See LICENSE file for details.
