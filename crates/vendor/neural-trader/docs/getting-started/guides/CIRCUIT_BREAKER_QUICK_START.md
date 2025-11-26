# Circuit Breaker Quick Start Guide

## 🚀 5-Minute Integration Guide

### 1. Import the Module

```rust
use neural_trader_napi::resilience::{
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry
};
use std::time::Duration;
```

### 2. Create a Circuit Breaker

```rust
// Default configuration (recommended for most use cases)
let cb = CircuitBreaker::new(
    "my_service".to_string(),
    CircuitBreakerConfig::default()
);

// Custom configuration
let cb = CircuitBreaker::new(
    "my_service".to_string(),
    CircuitBreakerConfig {
        failure_threshold: 3,           // Open after 3 failures
        success_threshold: 2,           // Close after 2 successes
        timeout: Duration::from_secs(5),    // 5 second timeout
        reset_timeout: Duration::from_secs(30), // Try again after 30 seconds
    }
);
```

### 3. Wrap Your Operations

```rust
// Wrap any async operation
let result = cb.call(async {
    // Your code here
    make_api_call().await
}).await;

match result {
    Ok(data) => println!("Success: {:?}", data),
    Err(e) => println!("Failed: {}", e),
}
```

## 📊 Common Patterns

### Pattern 1: External API Calls

```rust
use neural_trader_napi::resilience::ApiCircuitBreaker;

let api_cb = ApiCircuitBreaker::new("market_data");

// GET request
let quotes = api_cb.get::<MarketData>("https://api.example.com/quotes/AAPL").await?;

// POST request
let order = api_cb.post::<OrderResponse, _>(
    "https://api.example.com/orders",
    &order_request
).await?;
```

### Pattern 2: E2B Sandbox Operations

```rust
use neural_trader_napi::resilience::E2BSandboxCircuitBreaker;

let sandbox_cb = E2BSandboxCircuitBreaker::new();

let sandbox_id = sandbox_cb.create_sandbox("base").await?;
let result = sandbox_cb.execute_code(&sandbox_id, "print('hello')").await?;
sandbox_cb.stop_sandbox(&sandbox_id).await?;
```

### Pattern 3: Neural Network Operations

```rust
use neural_trader_napi::resilience::NeuralCircuitBreaker;

let neural_cb = NeuralCircuitBreaker::new("price_predictor");

let prediction = neural_cb.predict(vec![1.0, 2.0, 3.0]).await?;
```

### Pattern 4: Database Operations

```rust
use neural_trader_napi::resilience::DatabaseCircuitBreaker;

let db_cb = DatabaseCircuitBreaker::new("trading_db");

let users: Vec<User> = db_cb.query("SELECT * FROM users").await?;
```

### Pattern 5: Complete Trading System

```rust
use neural_trader_napi::resilience::TradingSystemCircuitBreakers;

let system = TradingSystemCircuitBreakers::new();

// All services automatically protected
let market_data = system.market_data_api.get("/v1/quotes/AAPL").await?;
let prediction = system.neural_predictor.predict(features).await?;
let order = system.order_execution_api.post("/v1/orders", &order).await?;

// Check health of all services
let health = system.health_status().await;
for (service, state) in health {
    println!("{}: {}", service, state);
}
```

## 🎯 Configuration Examples

### Conservative (High Availability)

```rust
CircuitBreakerConfig {
    failure_threshold: 10,              // Tolerate more failures
    success_threshold: 3,               // Need more successes
    timeout: Duration::from_secs(60),   // Longer timeout
    reset_timeout: Duration::from_secs(30), // Try recovery sooner
}
```

### Aggressive (Fail Fast)

```rust
CircuitBreakerConfig {
    failure_threshold: 3,               // Open quickly
    success_threshold: 2,               // Close quickly
    timeout: Duration::from_secs(5),    // Short timeout
    reset_timeout: Duration::from_secs(120), // Wait longer
}
```

### Balanced (Recommended Default)

```rust
CircuitBreakerConfig {
    failure_threshold: 5,
    success_threshold: 2,
    timeout: Duration::from_secs(30),
    reset_timeout: Duration::from_secs(60),
}
```

## 📈 Monitoring

### Check Circuit State

```rust
let state = cb.get_state().await;
println!("Circuit state: {}", state); // "CLOSED (failures: 0)"
```

### Get Metrics

```rust
let metrics = cb.get_metrics().await;
println!("Total calls: {}", metrics.total_calls);
println!("Success rate: {:.2}%", cb.get_success_rate().await);
```

### Monitor in Logs

```rust
if cb.get_state().await.starts_with("OPEN") {
    log::error!("Circuit breaker '{}' is OPEN!", name);
    // Alert ops team
}
```

## 🔧 Global Registry Pattern

### Setup (once per application)

```rust
use once_cell::sync::Lazy;
use neural_trader_napi::resilience::CircuitBreakerRegistry;

static BREAKERS: Lazy<CircuitBreakerRegistry> = Lazy::new(|| {
    CircuitBreakerRegistry::new()
});
```

### Use Anywhere

```rust
async fn make_api_call() -> Result<Data> {
    let cb = BREAKERS.get_or_create(
        "api".to_string(),
        CircuitBreakerConfig::default()
    ).await;

    cb.call(async { /* API call */ }).await
}
```

### Health Check Endpoint

```rust
async fn health_check() -> HashMap<String, String> {
    let mut status = HashMap::new();

    for name in BREAKERS.list_names().await {
        if let Some(cb) = BREAKERS.get(&name).await {
            status.insert(name, cb.get_state().await);
        }
    }

    status
}
```

## 🛡️ Error Handling Patterns

### With Fallback

```rust
let result = cb.call(fetch_latest_data()).await;

match result {
    Ok(data) => data,
    Err(e) if e.to_string().contains("is OPEN") => {
        log::warn!("Circuit open, using cached data");
        get_cached_data()?
    }
    Err(e) => return Err(e),
}
```

### With Retry Logic

```rust
for attempt in 1..=3 {
    match cb.call(risky_operation()).await {
        Ok(result) => return Ok(result),
        Err(e) if e.to_string().contains("is OPEN") => {
            log::warn!("Circuit open, waiting...");
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
        Err(e) => return Err(e),
    }
}
```

### With Graceful Degradation

```rust
async fn get_market_data(symbol: &str) -> Result<MarketData> {
    let cb = BREAKERS.get_or_create(
        "market_data".to_string(),
        CircuitBreakerConfig::default()
    ).await;

    match cb.call(fetch_live_data(symbol)).await {
        Ok(data) => Ok(data),
        Err(e) if e.to_string().contains("is OPEN") => {
            // Degraded mode: use delayed data
            Ok(fetch_delayed_data(symbol).await?)
        }
        Err(e) => Err(e),
    }
}
```

## 📋 Checklist for Production Use

- [ ] Choose appropriate thresholds for your service
- [ ] Set timeout based on expected operation duration
- [ ] Add logging for circuit state changes
- [ ] Implement fallback strategy for open circuit
- [ ] Add metrics to monitoring dashboard
- [ ] Set up alerts for circuit opens
- [ ] Document expected behavior in runbook
- [ ] Test circuit behavior under load
- [ ] Verify recovery after failures

## 🧪 Testing Your Circuit Breaker

```rust
#[tokio::test]
async fn test_my_service_circuit_breaker() {
    let cb = CircuitBreaker::new(
        "test".to_string(),
        CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout: Duration::from_secs(1),
            reset_timeout: Duration::from_millis(100),
        }
    );

    // Test failures open circuit
    for _ in 0..2 {
        let _ = cb.call(async { Err::<(), _>(anyhow!("fail")) }).await;
    }
    assert!(cb.get_state().await.starts_with("OPEN"));

    // Test recovery
    tokio::time::sleep(Duration::from_millis(150)).await;
    for _ in 0..2 {
        let _ = cb.call(async { Ok::<(), _>(()) }).await;
    }
    assert!(cb.get_state().await.starts_with("CLOSED"));
}
```

## 🚨 Troubleshooting

### Circuit Opens Too Frequently

```rust
// Increase failure threshold
failure_threshold: 10,  // was 5

// Increase timeout
timeout: Duration::from_secs(60),  // was 30
```

### Circuit Never Opens

```rust
// Check metrics
let metrics = cb.get_metrics().await;
println!("Failed calls: {}", metrics.failed_calls);

// Reduce threshold
failure_threshold: 3,  // was 5
```

### Circuit Won't Close

```rust
// Reduce success threshold
success_threshold: 1,  // was 2

// Reduce reset timeout
reset_timeout: Duration::from_secs(30),  // was 60
```

## 📚 Additional Resources

- **Full Documentation**: `/neural-trader-rust/crates/napi-bindings/docs/CIRCUIT_BREAKER.md`
- **Implementation Summary**: `/docs/CIRCUIT_BREAKER_IMPLEMENTATION.md`
- **Test Examples**: `/neural-trader-rust/crates/napi-bindings/tests/circuit_breaker_tests.rs`
- **Integration Examples**: `/neural-trader-rust/crates/napi-bindings/src/resilience/integration.rs`

## 🎓 Learning Path

1. ✅ Read this quick start guide
2. ✅ Try the basic example
3. ✅ Integrate with one service
4. ✅ Add monitoring
5. ✅ Test failure scenarios
6. ✅ Configure for production
7. ✅ Set up alerts
8. ✅ Document runbook procedures

---

**Need help?** Check the full documentation or review the test cases for more examples!
