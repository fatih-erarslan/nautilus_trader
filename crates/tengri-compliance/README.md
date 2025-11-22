# TENGRI Compliance Engine

**Trading Engine with No-compromise Governance, Risk, and Intelligence**

> ⚠️ **CRITICAL WARNING**: This is a production-grade compliance engine with **ZERO TOLERANCE** for violations. Every trade must pass TENGRI checks or it will be rejected immediately. This is not a toy - it's designed for real trading systems where compliance failures can result in regulatory violations and financial losses.

## 🎯 Overview

TENGRI is a comprehensive compliance engine designed for high-frequency trading systems that require:

- **Zero-tolerance compliance** - No trade executes without passing all checks
- **Real-time risk management** - Millisecond-level decision making
- **Comprehensive audit trails** - Every action is logged immutably
- **Market manipulation detection** - Advanced surveillance algorithms
- **Circuit breakers & kill switches** - Emergency protection mechanisms
- **Regulatory compliance** - Built for institutional trading requirements

## 🚨 Key Features

### Core Compliance
- **Position Limits** - Per-symbol and global exposure limits
- **Leverage Constraints** - Maximum leverage enforcement
- **Risk Limits** - Daily loss limits, concentration risk, volatility limits
- **Regulatory Rules** - Configurable compliance frameworks

### Surveillance & Detection
- **Wash Trading Detection** - Pattern recognition for offsetting trades
- **Spoofing Detection** - Large order cancellation analysis
- **Market Manipulation** - Advanced pattern detection algorithms
- **Volume Anomalies** - Statistical analysis of unusual activity

### Safety Mechanisms
- **Circuit Breakers** - Automatic trading halts on failures
- **Kill Switch** - Emergency shutdown capabilities
- **Audit Trail** - Immutable record of all decisions
- **Real-time Monitoring** - Continuous health checks

### Performance
- **Sub-millisecond latency** - Optimized for HFT requirements
- **Concurrent processing** - Handle thousands of trades/second
- **Memory efficient** - Ring buffers and optimized data structures
- **SIMD acceleration** - Vectorized calculations where possible

## 🔧 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TENGRI Compliance Engine                │
├─────────────────────────────────────────────────────────────┤
│  Trade Context  →  Rules Engine  →  Surveillance  →  Decision │
│                 ↓                 ↓               ↓          │
│             Audit Trail      Circuit Breakers   Metrics     │
└─────────────────────────────────────────────────────────────┘
```

### Components

1. **Rules Engine** - Evaluates compliance rules in priority order
2. **Surveillance Engine** - Detects suspicious trading patterns
3. **Circuit Breaker Manager** - Emergency protection mechanisms
4. **Audit Trail** - Immutable logging system
5. **Metrics System** - Real-time monitoring and reporting

## 📚 Usage

### Basic Setup

```rust
use tengri_compliance::{ComplianceEngine, ComplianceConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create engine with default configuration
    let config = ComplianceConfig::default();
    let engine = ComplianceEngine::new(config).await?;
    
    // Process a trade
    let context = create_trading_context();
    match engine.process_trade(context).await? {
        ComplianceDecision::Approved { order_id, .. } => {
            println!("Trade approved: {}", order_id);
            // Execute the trade
        },
        ComplianceDecision::Rejected { reason, .. } => {
            println!("Trade rejected: {}", reason);
            // Handle rejection
        }
    }
    
    Ok(())
}
```

### Advanced Configuration

```rust
use tengri_compliance::{
    ComplianceEngine, ComplianceConfig, StrictnessLevel,
    rules::{RuleCategories, RuleSet},
};
use std::time::Duration;

let config = ComplianceConfig {
    max_audit_records: 1_000_000,
    strictness_level: StrictnessLevel::Conservative,
    enabled_rule_categories: RuleCategories::ALL,
    auto_circuit_breakers: true,
    kill_switch_threshold: 0.05, // 5% error rate triggers kill switch
    monitoring_frequency: Duration::from_secs(5),
    audit_wal_path: Some("/var/log/tengri/audit.wal".to_string()),
    ..Default::default()
};

let engine = ComplianceEngine::new(config).await?;

// Add custom rules
let custom_rules = RuleSet::conservative();
for rule in custom_rules {
    engine.add_rule(rule);
}
```

### Trading Context

```rust
use tengri_compliance::rules::{TradingContext, OrderSide, OrderType};
use rust_decimal::Decimal;
use std::collections::HashMap;

let context = TradingContext {
    order_id: uuid::Uuid::new_v4(),
    symbol: "BTCUSD".to_string(),
    side: OrderSide::Buy,
    quantity: Decimal::from_str("0.5").unwrap(),
    price: Some(Decimal::from(50000)),
    order_type: OrderType::Limit,
    trader_id: "trader_001".to_string(),
    timestamp: chrono::Utc::now(),
    portfolio_value: Decimal::from(1_000_000),
    current_positions: HashMap::new(),
    daily_pnl: Decimal::from(5000),
    metadata: HashMap::new(),
};
```

## 🛡️ Safety Mechanisms

### Circuit Breakers

```rust
// Automatic circuit breakers are enabled by default
// They trigger on:
// - Consecutive failures (5+ in a row)
// - High failure rate (>20% in 1 minute)
// - Loss thresholds ($100K in 1 hour)

// Manual circuit breaker activation
engine.emergency_shutdown("Market volatility spike detected".to_string()).await?;
```

### Kill Switch

```rust
// The kill switch activates automatically when:
// - Error rate exceeds threshold (default 10%)
// - Critical system failures occur
// - Manual activation

// Check if kill switch is active
if engine.get_status().await == EngineStatus::Emergency {
    // All trading is halted
    println!("EMERGENCY: Trading halted by kill switch");
}
```

## 📊 Monitoring & Metrics

### Real-time Dashboard

```rust
let metrics = engine.get_metrics();
let dashboard = metrics.get_dashboard_data();

println!("System Health: {:.1}%", dashboard.system_health_score);
println!("Trades Processed: {}", dashboard.total_trades_processed);
println!("Rejection Rate: {:.2}%", dashboard.rejection_rate);
println!("Active Circuit Breakers: {}", dashboard.circuit_breakers_active);
```

### Prometheus Metrics

```rust
// Export metrics in Prometheus format
let prometheus_metrics = metrics.export_metrics()?;
// Send to monitoring system
```

### Audit Trail

```rust
let audit_trail = engine.get_audit_trail();

// Get recent events
let recent_events = audit_trail.get_recent_events(100);

// Search by trader
let trader_events = audit_trail.search_by_actor("trader_001");

// Get critical events
let critical_events = audit_trail.get_critical_events();
```

## 🔍 Surveillance

### Pattern Detection

```rust
let surveillance = engine.get_surveillance_engine();

// Analyze patterns (runs automatically)
let patterns = surveillance.analyze_patterns().await?;

for pattern in patterns {
    match pattern.pattern_type {
        PatternType::WashTrading => {
            println!("ALERT: Wash trading detected with {:.1}% confidence", 
                    pattern.confidence);
        },
        PatternType::Spoofing => {
            println!("ALERT: Spoofing detected with {:.1}% confidence", 
                    pattern.confidence);
        },
        _ => {}
    }
}
```

## ⚡ Performance

### Benchmarks

```bash
# Run performance benchmarks
cargo bench

# Example results:
# trade_processing/0.1     time: [245.2 μs 247.8 μs 250.6 μs]
# trade_processing/1.0     time: [248.1 μs 251.2 μs 254.8 μs]
# rule_evaluation          time: [45.2 μs 46.1 μs 47.3 μs]
# audit_recording          time: [12.1 μs 12.4 μs 12.8 μs]
# concurrent_trades/100    time: [2.45 ms 2.52 ms 2.61 ms]
```

### Optimization Tips

1. **Use Conservative Rules** - Start with conservative rule sets
2. **Monitor Memory** - Check audit trail buffer utilization
3. **Circuit Breaker Tuning** - Adjust thresholds based on your risk tolerance
4. **Concurrent Processing** - TENGRI is designed for high concurrency
5. **Metrics Collection** - Use metrics to identify bottlenecks

## 🚨 Production Deployment

### Prerequisites

- Rust 1.70+ with `async` support
- Sufficient memory for audit trails (recommend 8GB+)
- Fast SSD storage for audit WAL
- Low-latency network for real-time monitoring

### Configuration Checklist

- [ ] Set appropriate position limits
- [ ] Configure leverage constraints
- [ ] Set up audit WAL persistence
- [ ] Configure circuit breaker thresholds
- [ ] Set up monitoring/alerting
- [ ] Test emergency procedures
- [ ] Validate rule configurations
- [ ] Performance test under load

### Security Considerations

- Store audit logs immutably
- Use encrypted connections for monitoring
- Implement proper access controls
- Regular compliance rule updates
- Backup audit trails regularly

## 🧪 Testing

```bash
# Run all tests
cargo test

# Run integration tests
cargo test --test integration_tests

# Run benchmarks
cargo bench

# Test with different strictness levels
TENGRI_STRICTNESS=conservative cargo test
TENGRI_STRICTNESS=aggressive cargo test
```

## 📝 Configuration Files

### Environment Variables

```bash
# Compliance configuration
export TENGRI_STRICTNESS=conservative
export TENGRI_MAX_POSITION=1000000
export TENGRI_MAX_LEVERAGE=5.0
export TENGRI_AUDIT_WAL=/var/log/tengri/audit.wal

# Monitoring
export TENGRI_METRICS_PORT=9090
export TENGRI_HEALTH_CHECK_INTERVAL=10
```

## 🆘 Emergency Procedures

### Kill Switch Activation

```bash
# Manual kill switch activation
tengri-cli emergency-shutdown "Market manipulation detected"

# Or via API
curl -X POST http://localhost:8080/api/v1/emergency-shutdown \
  -H "Content-Type: application/json" \
  -d '{"reason": "Regulatory investigation", "authorized_by": "compliance_officer"}'
```

### Recovery Procedures

1. **Identify Root Cause** - Check audit logs and metrics
2. **Fix Issues** - Address compliance violations or system problems
3. **Reset Circuit Breakers** - Clear any triggered breakers
4. **Deactivate Kill Switch** - Only after thorough investigation
5. **Gradual Restart** - Start with reduced position limits
6. **Monitor Closely** - Watch for any recurring issues

## 📜 Compliance Standards

TENGRI is designed to meet requirements for:

- **MiFID II** - European markets regulation
- **FINRA** - US securities regulation  
- **CFTC** - US derivatives regulation
- **Basel III** - International banking standards
- **SOX** - Sarbanes-Oxley audit requirements

## 🤝 Contributing

This is a critical production system. All contributions must:

1. Include comprehensive tests
2. Pass all benchmarks
3. Maintain backward compatibility
4. Include security review
5. Update documentation

## 📄 License

Licensed under MIT OR Apache-2.0

---

> ⚠️ **DISCLAIMER**: This compliance engine is designed for professional trading environments. Improper configuration or use may result in financial losses or regulatory violations. Always test thoroughly in non-production environments and consult with compliance professionals before deployment.