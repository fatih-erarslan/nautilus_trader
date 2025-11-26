# Neural Trader Verification Rubric
## Real Data Validation Framework

### 1. DATA SOURCE VERIFICATION (Truth Score: 0.95 Threshold)

#### Primary Verification Points:
- [ ] **Live API Connection** - Verify real-time data feed
- [ ] **Timestamp Validation** - Ensure current market timestamps
- [ ] **Price Movement Correlation** - Cross-reference with multiple sources
- [ ] **Volume Authenticity** - Validate trading volumes are realistic
- [ ] **Latency Checks** - Measure data feed latency (<100ms)

### 2. VERIFICATION COMMANDS

```bash
# Initialize truth verification system
npx claude-flow@alpha verify init --threshold 0.95

# Check current truth scores
npx claude-flow@alpha truth --metrics

# Start pair verification mode (collaborative real-time checking)
npx claude-flow@alpha pair --start --verify-mode

# Verify specific data source
npx claude-flow@alpha verify source --api "alpaca" --live
npx claude-flow@alpha verify source --api "polygon" --live
```

### 3. AUTOMATED VERIFICATION CHECKS

```python
# Real Data Verification Module
class DataVerification:
    def __init__(self):
        self.truth_threshold = 0.95
        self.verified_sources = []
        
    def verify_live_data(self, data_source):
        """Verify data is from live API, not simulation"""
        checks = {
            'timestamp_current': self.check_timestamp(data_source),
            'price_realistic': self.check_price_movements(data_source),
            'volume_authentic': self.check_volume_patterns(data_source),
            'api_authenticated': self.check_api_credentials(data_source),
            'latency_acceptable': self.check_latency(data_source)
        }
        
        truth_score = sum(checks.values()) / len(checks)
        return truth_score >= self.truth_threshold
        
    def check_timestamp(self, data):
        """Ensure timestamp is within 1 second of current time"""
        import time
        current_time = time.time()
        data_time = data.get('timestamp', 0)
        return abs(current_time - data_time) < 1.0
        
    def check_price_movements(self, data):
        """Verify price changes are within realistic bounds"""
        price_change = abs(data.get('price_change_percent', 0))
        return price_change < 20  # Less than 20% change (circuit breaker limit)
        
    def check_volume_patterns(self, data):
        """Validate volume matches market hours and patterns"""
        volume = data.get('volume', 0)
        return volume > 0 and volume < 1e12  # Realistic volume range
        
    def check_api_credentials(self, data_source):
        """Verify API keys are valid and not in demo mode"""
        return (
            data_source.api_key and 
            not data_source.api_key.startswith('demo_') and
            not data_source.api_key.startswith('test_')
        )
        
    def check_latency(self, data_source):
        """Ensure data latency is under threshold"""
        return data_source.latency_ms < 100
```

### 4. RUBRIC SCORING SYSTEM

| Component | Weight | Pass Criteria | Verification Method |
|-----------|--------|--------------|-------------------|
| **Data Source** | 30% | Live API authenticated | `verify source --api` |
| **Timestamps** | 25% | Within 1s of current | `verify timestamp --live` |
| **Price Data** | 20% | Realistic movements | `verify price --bounds` |
| **Volume** | 15% | Authentic patterns | `verify volume --pattern` |
| **Latency** | 10% | <100ms feed delay | `verify latency --threshold` |

### 5. CONTINUOUS VERIFICATION WORKFLOW

```yaml
# .claude/verification-workflow.yml
verification:
  mode: continuous
  threshold: 0.95
  
  pre_trade_checks:
    - verify_api_connection
    - validate_credentials
    - check_market_hours
    - confirm_live_data
    
  during_trade_checks:
    - monitor_data_feed
    - validate_prices
    - check_execution_latency
    - verify_order_fills
    
  post_trade_checks:
    - audit_transactions
    - verify_positions
    - validate_pnl
    - check_compliance
    
  alerts:
    - truth_score_below_threshold
    - simulation_mode_detected
    - demo_credentials_found
    - data_anomaly_detected
```

### 6. PAIR VERIFICATION MODE

```bash
# Start collaborative verification with real-time checks
npx claude-flow@alpha pair --start \
  --verify-real-data \
  --threshold 0.95 \
  --alert-on-simulation

# Configuration for pair mode
pair_config:
  verification:
    enabled: true
    real_time_validation: true
    cross_reference_sources: ["alpaca", "polygon", "yahoo"]
    alert_channels: ["console", "log", "webhook"]
    fail_on_simulation: true
```

### 7. NEURAL TRADER SPECIFIC CHECKS

```python
# Ensure neural-trader uses real data
class NeuralTraderVerification:
    @staticmethod
    def verify_not_demo_mode():
        """Check neural-trader is NOT in demo mode"""
        from mcp__neural_trader import get_config
        config = get_config()
        
        checks = {
            'not_demo_mode': not config.get('demo_mode', False),
            'real_api_keys': all([
                config.get('ALPACA_KEY') and not 'demo' in config.get('ALPACA_KEY', ''),
                config.get('POLYGON_KEY') and not 'test' in config.get('POLYGON_KEY', '')
            ]),
            'live_trading_enabled': config.get('live_trading', False),
            'paper_trading_disabled': not config.get('paper_trading', True)
        }
        
        return all(checks.values())
        
    @staticmethod
    def verify_data_sources():
        """Verify all data sources are live"""
        sources = [
            ('alpaca', 'wss://stream.data.alpaca.markets'),
            ('polygon', 'wss://socket.polygon.io'),
            ('news', 'https://api.polygon.io/v2/reference/news')
        ]
        
        for name, endpoint in sources:
            # Verify WebSocket connection is live
            # Verify data timestamps are current
            # Verify no "sandbox" or "demo" in URLs
            pass
```

### 8. TRUTH SCORE MONITORING

```bash
# Monitor truth scores in real-time
npx claude-flow@alpha truth --monitor --interval 5s

# Output format:
# ┌─────────────────────────────────────┐
# │ TRUTH VERIFICATION SYSTEM           │
# ├─────────────────────────────────────┤
# │ Overall Truth Score: 0.97 ✓         │
# │ Data Source: LIVE ✓                 │
# │ API Status: AUTHENTICATED ✓         │
# │ Timestamp Delta: 0.3s ✓             │
# │ Price Validity: CONFIRMED ✓         │
# │ Volume Pattern: NORMAL ✓            │
# │ Latency: 45ms ✓                     │
# └─────────────────────────────────────┘
```

### 9. AUTOMATED ENFORCEMENT

```python
# Auto-fail trades if verification fails
def execute_trade_with_verification(trade_params):
    verifier = DataVerification()
    
    # Pre-trade verification
    if not verifier.verify_live_data(trade_params.data_source):
        raise ValueError("VERIFICATION FAILED: Data source not live")
    
    # Execute trade only if verified
    result = execute_trade(trade_params)
    
    # Post-trade verification
    if not verifier.verify_execution(result):
        rollback_trade(result)
        raise ValueError("VERIFICATION FAILED: Execution anomaly detected")
    
    return result
```

### 10. COMPLIANCE REPORTING

```bash
# Generate verification report
npx claude-flow@alpha verify report --format json > verification_report.json

# Continuous compliance monitoring
npx claude-flow@alpha verify compliance --continuous \
  --log-file /var/log/neural-trader-verification.log \
  --alert-webhook $WEBHOOK_URL
```

## USAGE EXAMPLES

```bash
# 1. Initialize verification system
npx claude-flow@alpha verify init --threshold 0.95

# 2. Start pair programming with verification
npx claude-flow@alpha pair --start --verify-mode

# 3. Run truth checks
npx claude-flow@alpha truth --metrics

# 4. Verify specific trade
npx claude-flow@alpha verify trade --id "trade_123" --deep-check

# 5. Monitor in real-time
npx claude-flow@alpha verify monitor --real-time --alert-threshold 0.90
```

## VERIFICATION RUBRIC SUMMARY

✅ **PASS CRITERIA (Truth Score ≥ 0.95)**
- Live API connections verified
- Real-time timestamps confirmed
- Price data validated against multiple sources
- Volume patterns authentic
- No demo/simulation mode detected
- Latency within acceptable range

❌ **FAIL CONDITIONS**
- Demo API keys detected
- Simulation mode active
- Timestamps not current
- Price anomalies detected
- Volume patterns unrealistic
- Truth score < 0.95

## ENFORCEMENT
All trades MUST pass verification or they will be:
1. Blocked from execution
2. Logged as verification failure
3. Trigger immediate alert
4. Require manual override with justification