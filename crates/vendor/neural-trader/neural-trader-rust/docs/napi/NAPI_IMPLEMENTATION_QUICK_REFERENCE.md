# NAPI Implementation Quick Reference

**Quick lookup for implementing real Rust functions**

---

## 🎯 Priority Matrix

| Priority | Count | Timeline | Functions |
|----------|-------|----------|-----------|
| **P0** | 6 | Week 1 | `ping`, `list_strategies`, `get_strategy_info`, `get_portfolio_status`, `get_health_status`, `initialize_neural_trader` |
| **P1** | 22 | Weeks 1-3 | Core trading, neural basics, risk management |
| **P2** | 50 | Weeks 4-9 | Advanced features, sports betting, syndicates |
| **P3** | 25 | Weeks 10-12 | E2B, monitoring, optimization |

---

## 📦 Crate Usage Map

```
FUNCTION CATEGORY          → TARGET CRATE(S)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Core Trading (23)          → nt-strategies, nt-execution, nt-portfolio
Neural Networks (7)        → nt-neural (with candle-core)
News Trading (8)           → nt-news-trading
Portfolio & Risk (5)       → nt-risk, nt-portfolio
Sports Betting (13)        → nt-sports-betting
Odds API (9)              → nt-sports-betting (OddsAPI module)
Prediction Markets (5)     → nt-prediction-markets
Syndicates (15)           → nt-syndicate
E2B Cloud (9)             → nt-e2b-integration
System Monitoring (5)      → nt-core, nt-utils
```

---

## 🚀 Quick Start Implementation

### Step 1: Initialize Services (Week 1)

```rust
// File: napi-bindings/src/services/mod.rs

pub struct ServiceContainer {
    pub broker_client: Arc<dyn BrokerClient>,
    pub portfolio_tracker: Arc<RwLock<PortfolioTracker>>,
    pub risk_manager: Arc<RiskManager>,
    pub neural_engine: Arc<NeuralEngine>,
    pub market_data_provider: Arc<dyn MarketDataProvider>,
}

#[napi]
pub async fn initialize_neural_trader(config_json: String) -> Result<String> {
    let config: SystemConfig = serde_json::from_str(&config_json)?;
    services::init_services(config).await?;
    Ok(json!({"status": "initialized"}).to_string())
}
```

### Step 2: Implement First Function (Example)

```rust
#[napi]
pub async fn execute_trade(
    strategy: String,
    symbol: String,
    action: String,
    quantity: i32,
) -> Result<String> {
    napi_result! {
        let services = services();
        let order = OrderRequest {
            symbol,
            side: parse_side(&action)?,
            quantity: quantity as u32,
            order_type: OrderType::Market,
            time_in_force: TimeInForce::Day,
        };

        services.broker_client.execute_order(order).await
    }
}
```

### Step 3: Add Error Handling

```rust
macro_rules! napi_result {
    ($expr:expr) => {
        match $expr {
            Ok(val) => Ok(serde_json::to_string(&val)?),
            Err(e) => Ok(to_json_error(e.into())),
        }
    };
}
```

---

## 🔥 GPU-Accelerated Functions (Top Priority)

| Function | Speedup | Crate | Feature Flag |
|----------|---------|-------|--------------|
| `neural_forecast()` | 50-100x | `nt-neural` | `cuda` or `metal` |
| `neural_train()` | 50-100x | `nt-neural` | `cuda` or `metal` |
| `risk_analysis()` | 10-50x | `nt-risk` | `gpu` |
| `monte_carlo_simulation()` | 10-50x | `nt-risk` | `gpu` |
| `correlation_analysis()` | 5-20x | `nt-risk` | `gpu` |
| `run_backtest()` | 3-10x | `nt-strategies` | `gpu` |

---

## 📋 Implementation Checklist (Per Function)

- [ ] Read existing simulation code
- [ ] Identify target crate and module
- [ ] Write service container method
- [ ] Implement NAPI function with error handling
- [ ] Add JSON serialization
- [ ] Write unit tests
- [ ] Write integration test
- [ ] Add performance benchmark (if critical path)
- [ ] Update documentation

---

## 🛠️ Common Patterns

### Pattern 1: Simple Query (No External Calls)

```rust
#[napi]
pub async fn list_strategies() -> Result<String> {
    napi_result! {
        let services = services();
        let strategies = services.strategy_orchestrator.list_strategies();
        json!({
            "strategies": strategies,
            "timestamp": Utc::now().to_rfc3339()
        })
    }
}
```

### Pattern 2: Broker Interaction

```rust
#[napi]
pub async fn execute_trade(...) -> Result<String> {
    napi_result! {
        let services = services();

        // 1. Validate with risk manager
        services.risk_manager.validate_order(&order).await?;

        // 2. Execute via broker
        let response = services.broker_client.execute_order(order).await?;

        // 3. Update portfolio
        services.portfolio_tracker.write().record_order(&response).await?;

        // 4. Return result
        json!({"order_id": response.order_id, ...})
    }
}
```

### Pattern 3: GPU-Accelerated Computation

```rust
#[napi]
pub async fn neural_forecast(use_gpu: Option<bool>, ...) -> Result<String> {
    napi_result! {
        let services = services();

        // Select device
        let device = if use_gpu.unwrap_or(true) {
            &services.gpu_manager.neural_device
        } else {
            &Device::Cpu
        };

        // Run computation
        let result = services.neural_engine
            .forecast(&model, &data, device)
            .await?;

        json!({
            "predictions": result,
            "gpu_accelerated": device.is_cuda(),
            ...
        })
    }
}
```

---

## 📊 Testing Strategy

### Unit Test Template

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_execute_trade_success() {
        // Setup mock services
        let config = test_config();
        init_services(config).await.unwrap();

        // Execute function
        let result = execute_trade(
            "momentum".into(),
            "AAPL".into(),
            "buy".into(),
            10,
            None,
            None,
        ).await;

        // Verify
        assert!(result.is_ok());
        let json: serde_json::Value =
            serde_json::from_str(&result.unwrap()).unwrap();
        assert_eq!(json["status"], "filled");
    }
}
```

### Integration Test Template

```rust
#[tokio::test]
async fn test_end_to_end_workflow() {
    init_services(paper_trading_config()).await.unwrap();

    // 1. Get strategy info
    let strategies = list_strategies().await.unwrap();

    // 2. Execute trade
    let trade_result = execute_trade(...).await.unwrap();

    // 3. Check portfolio
    let portfolio = get_portfolio_status(Some(true)).await.unwrap();

    // Verify full workflow
    // ...
}
```

---

## 🎯 Phase 1 Implementation Order (Weeks 1-3)

### Week 1: Foundation
1. ✅ `initialize_neural_trader()` - System setup
2. ✅ `ping()` - Health check
3. ✅ `get_health_status()` - System status
4. ✅ `list_strategies()` - Strategy listing
5. ✅ `get_strategy_info()` - Strategy details
6. ✅ `get_portfolio_status()` - Portfolio tracking

### Week 2: Trading Core
7. ✅ `execute_trade()` - Order execution
8. ✅ `simulate_trade()` - Trade simulation
9. ✅ `quick_analysis()` - Technical analysis
10. ✅ `run_backtest()` - Backtesting
11. ✅ `risk_analysis()` - Risk metrics

### Week 3: Neural & Advanced
12. ✅ `neural_forecast()` - Price prediction
13. ✅ `neural_train()` - Model training
14. ✅ `neural_predict()` - Generic prediction
15. ✅ `calculate_kelly_criterion()` - Position sizing

---

## 🔧 Configuration Example

```json
{
  "broker": {
    "provider": "alpaca",
    "api_key": "YOUR_KEY",
    "api_secret": "YOUR_SECRET",
    "base_url": "https://paper-api.alpaca.markets"
  },
  "neural": {
    "use_gpu": true,
    "device": "cuda",
    "model_cache_dir": "./models"
  },
  "risk": {
    "max_position_size": 10000.0,
    "var_confidence": 0.95,
    "use_gpu": true
  },
  "system": {
    "log_level": "info",
    "enable_metrics": true
  }
}
```

---

## 📈 Performance Targets

| Function | Target Latency (p50) | Target Latency (p95) |
|----------|---------------------|---------------------|
| Order execution | < 50ms | < 100ms |
| Portfolio queries | < 10ms | < 20ms |
| Risk (CPU) | < 500ms | < 1s |
| Risk (GPU) | < 50ms | < 100ms |
| Neural forecast (CPU) | < 2s | < 5s |
| Neural forecast (GPU) | < 200ms | < 500ms |

---

## 🚨 Common Pitfalls & Solutions

### Pitfall 1: Not Initializing Services
**Problem**: Functions fail with "services not initialized"
**Solution**: Always call `initialize_neural_trader()` before other functions

### Pitfall 2: GPU Not Available
**Problem**: GPU functions fail
**Solution**: Always check `gpu_available` and fallback to CPU

### Pitfall 3: Invalid Configuration
**Problem**: Services fail to initialize
**Solution**: Use schema validation for configuration

### Pitfall 4: Memory Leaks
**Problem**: Long-running processes consume memory
**Solution**: Use Arc/RwLock properly, test with stress tests

---

## 📚 Key Dependencies

```toml
# Add to napi-bindings/Cargo.toml
[dependencies]
nt-strategies = { version = "2.0.0", path = "../strategies" }
nt-execution = { version = "2.0.0", path = "../execution" }
nt-portfolio = { version = "2.0.0", path = "../portfolio" }
nt-risk = { version = "2.0.0", path = "../risk" }
nt-neural = { version = "2.0.0", path = "../neural" }

[features]
gpu = ["nt-neural/cuda", "nt-risk/gpu"]
```

---

## 🔗 Related Documentation

- **Full Architecture**: `NAPI_REAL_IMPLEMENTATION_ARCHITECTURE.md` (1153 lines)
- **Crate Docs**: `../*/README.md` for each crate
- **MCP Protocol**: `../mcp-protocol/README.md`
- **Testing Guide**: `TESTING.md`

---

**Last Updated**: 2025-11-14
**Total Functions**: 103
**Estimated Timeline**: 12 weeks
