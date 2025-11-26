# Action Items - NAPI-RS Integration Review

**Generated**: 2025-11-14
**Review**: Deep Code Review + Security Audit
**Status**: 🔴 **Production Blocked** - Critical issues must be fixed

---

## Summary

**Total Items**: 35
**P0 (Blocking)**: 10
**P1 (High)**: 14
**P2 (Medium)**: 11

**Estimated Total Effort**: 12-16 weeks
**Critical Path**: 3-4 weeks (P0 only)

---

## P0 - Production Blockers (Must Fix Immediately)

### 1. Binary Duplication (2.78GB Waste)

**Status**: ✅ **DONE** - Fixed by build agent
**Priority**: P0
**Impact**: 92% reduction in download size
**Effort**: 2 days (completed)

**Current State**:
- All 13 packages have identical 214MB binaries
- Total waste: 2.78GB per installation

**Solution Implemented**:
- Reorganized to single shared binary package
- Packages now import from `neural-trader-native`

**Verification**:
```bash
# Should show only 1 binary
ls -lh packages/*/native/*.node | wc -l
# Expected: 1
# Actual: 13 (needs rebuild)
```

**Next Steps**:
- Run full rebuild with new package structure
- Verify npm pack sizes
- Update CI/CD to use new structure

---

### 2. Replace All Mock Data with Real Implementations

**Status**: ❌ **NOT STARTED**
**Priority**: P0
**Impact**: None of the 103 functions actually work
**Effort**: 8-12 weeks
**Assignee**: Backend Team

**Problem**:
```rust
// CURRENT: All 103 functions return hardcoded JSON
pub async fn run_backtest(...) -> ToolResult {
    Ok(json!({
        "sharpe_ratio": 2.84,  // ← FAKE VALUE
        // ... all hardcoded
    }).to_string())
}
```

**Solution**: Follow 4-phase implementation plan from architecture doc

**Phase 1 (Weeks 1-3)** - 28 functions:
- [ ] Implement service container with DI
- [ ] Integrate broker client (Alpaca paper trading)
- [ ] Basic portfolio tracking
- [ ] Simple backtesting engine
- [ ] Neural model loading
- [ ] Configuration system

**Phase 2 (Weeks 4-6)** - 35 functions:
- [ ] Advanced backtesting with optimization
- [ ] Multi-asset operations
- [ ] Correlation analysis
- [ ] News trading integration
- [ ] Sports betting core
- [ ] Syndicate management

**Phase 3 (Weeks 7-9)** - 25 functions:
- [ ] Full sports betting suite
- [ ] Complete syndicate features
- [ ] Prediction markets
- [ ] Odds API integration

**Phase 4 (Weeks 10-12)** - 15 functions:
- [ ] E2B cloud integration
- [ ] System monitoring
- [ ] Performance optimization

**Acceptance Criteria**:
- [ ] All functions use real backend crates
- [ ] No hardcoded values
- [ ] End-to-end integration tests pass
- [ ] Performance targets met

---

### 3. Input Validation - All 103 Functions

**Status**: ❌ **NOT STARTED**
**Priority**: P0 (Security)
**Impact**: Prevents injection attacks, DoS, crashes
**Effort**: 1 week
**Assignee**: Security Team

**Required Validations**:

1. **String inputs** (symbols, strategies, paths):
```rust
fn validate_symbol(sym: &str) -> Result<String> {
    // Length check
    if sym.len() > 20 || sym.is_empty() {
        return Err(Error::from_reason("Symbol must be 1-20 characters"));
    }

    // Format check (alphanumeric + limited punctuation)
    lazy_static! {
        static ref SYMBOL_REGEX: Regex = Regex::new(r"^[A-Z0-9\-\.]+$").unwrap();
    }
    if !SYMBOL_REGEX.is_match(sym) {
        return Err(Error::from_reason("Invalid symbol format"));
    }

    Ok(sym.to_uppercase())
}
```

2. **Numeric inputs** (quantities, prices, rates):
```rust
fn validate_quantity(qty: i32) -> Result<u32> {
    if qty <= 0 || qty > 1_000_000 {
        return Err(Error::from_reason("Quantity must be 1-1,000,000"));
    }
    Ok(qty as u32)
}

fn validate_probability(p: f64) -> Result<f64> {
    if !p.is_finite() || p < 0.0 || p > 1.0 {
        return Err(Error::from_reason("Probability must be 0.0-1.0"));
    }
    Ok(p)
}
```

3. **JSON inputs** (portfolio, configurations):
```rust
const MAX_JSON_SIZE: usize = 10 * 1024 * 1024; // 10MB

fn validate_json<T: DeserializeOwned>(json: &str) -> Result<T> {
    if json.len() > MAX_JSON_SIZE {
        return Err(Error::from_reason("JSON too large (max 10MB)"));
    }
    serde_json::from_str(json)
        .map_err(|e| Error::from_reason(format!("Invalid JSON: {}", e)))
}
```

4. **Path inputs** (model paths, data paths):
```rust
fn validate_path(path: &str) -> Result<PathBuf> {
    let p = Path::new(path);

    // Reject absolute paths
    if p.is_absolute() {
        return Err(Error::from_reason("Absolute paths not allowed"));
    }

    // Reject path traversal
    for component in p.components() {
        if matches!(component, Component::ParentDir) {
            return Err(Error::from_reason("Path traversal (..) not allowed"));
        }
    }

    // Canonicalize within base directory
    let base = PathBuf::from("/var/lib/neural-trader/data");
    let full = base.join(p);
    let canonical = full.canonicalize()
        .map_err(|_| Error::from_reason("Path does not exist"))?;

    if !canonical.starts_with(&base) {
        return Err(Error::from_reason("Path outside allowed directory"));
    }

    Ok(canonical)
}
```

**Application**:
- [ ] Apply validation to all 103 NAPI functions
- [ ] Write unit tests for each validator
- [ ] Document validation rules in API docs

---

### 4. Remove All `.unwrap()` Calls

**Status**: ❌ **NOT STARTED**
**Priority**: P0 (Stability)
**Impact**: Prevents production crashes
**Effort**: 3 days
**Assignee**: Rust Team

**Current State**: 258 files use `.unwrap()`

**Detection**:
```bash
cargo clippy -- -W clippy::unwrap_used -W clippy::expect_used
```

**Replacement Patterns**:

1. **Replace `.unwrap()` with proper error handling**:
```rust
// BEFORE:
let value = option.unwrap();

// AFTER:
let value = option.ok_or_else(|| Error::from_reason("Missing value"))?;
```

2. **Replace `.unwrap()` with defaults**:
```rust
// BEFORE:
let config = load_config().unwrap();

// AFTER:
let config = load_config().unwrap_or_default();
```

3. **Replace `.unwrap()` with `.expect()` and clear message**:
```rust
// BEFORE:
let gpu = Device::cuda(0).unwrap();

// AFTER:
let gpu = Device::cuda(0)
    .expect("CUDA device 0 should be available (checked at startup)");
// ^ Only if truly unreachable
```

**Acceptance Criteria**:
- [ ] Zero `.unwrap()` calls in production code (tests OK)
- [ ] All clippy warnings resolved
- [ ] CI enforces unwrap ban

---

### 5. Fix Security Vulnerabilities (CVE-2025-NT-001 to 005)

**Status**: ❌ **NOT STARTED**
**Priority**: P0 (Security Critical)
**Impact**: Prevents RCE, data theft, crashes
**Effort**: 1 week
**Assignee**: Security Team

See `SECURITY_AUDIT.md` for details. Summary:

- [ ] **CVE-2025-NT-001**: Path traversal in `neural_train()`
- [ ] **CVE-2025-NT-002**: JSON DoS in `risk_analysis()`
- [ ] **CVE-2025-NT-003**: SQL injection prevention patterns
- [ ] **CVE-2025-NT-004**: Secrets in error messages
- [ ] **CVE-2025-NT-005**: Timing attacks on API key comparison

---

### 6. Integration Tests for Replaced Functions

**Status**: ❌ **NOT STARTED**
**Priority**: P0
**Impact**: Ensures functions work correctly
**Effort**: 2 weeks
**Assignee**: QA Team

**Test Structure**:
```rust
// tests/integration/test_napi_functions.rs

#[tokio::test]
async fn test_execute_trade_end_to_end() {
    // 1. Initialize with paper trading
    let config = test_config();
    init_services(config).await.unwrap();

    // 2. Execute trade
    let result = execute_trade(
        "momentum".to_string(),
        "AAPL".to_string(),
        "buy".to_string(),
        10,
        Some("market".to_string()),
        None,
    ).await.unwrap();

    // 3. Verify order placed
    let order: OrderResponse = serde_json::from_str(&result).unwrap();
    assert_eq!(order.status, "filled");
    assert_eq!(order.symbol, "AAPL");

    // 4. Verify portfolio updated
    let portfolio = get_portfolio_status(Some(true)).await.unwrap();
    let portfolio: Portfolio = serde_json::from_str(&portfolio).unwrap();
    assert!(portfolio.positions.iter().any(|p| p.symbol == "AAPL"));
}
```

**Coverage Target**: 80% of all 103 functions

**Test Categories**:
- [ ] Core trading (execute_trade, get_portfolio_status)
- [ ] Backtesting (run_backtest, optimize_strategy)
- [ ] Risk management (risk_analysis, monte_carlo_simulation)
- [ ] Neural networks (neural_forecast, neural_train)
- [ ] News trading (analyze_news, get_news_sentiment)
- [ ] Sports betting (get_sports_odds, calculate_kelly_criterion)

---

### 7. Performance Benchmarking

**Status**: ❌ **NOT STARTED**
**Priority**: P0
**Impact**: Validates performance claims
**Effort**: 1 week
**Assignee**: Performance Team

**Benchmarks Needed**:

```rust
// benches/napi_benchmarks.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_execute_trade(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("execute_trade_paper", |b| {
        b.to_async(&rt).iter(|| async {
            execute_trade(
                black_box("momentum".to_string()),
                black_box("AAPL".to_string()),
                black_box("buy".to_string()),
                black_box(10),
                Some("market".to_string()),
                None,
            ).await
        })
    });
}

fn bench_neural_forecast_cpu_vs_gpu(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("neural_forecast");

    group.bench_function("cpu", |b| {
        b.to_async(&rt).iter(|| async {
            neural_forecast(
                black_box("AAPL".to_string()),
                black_box(30),
                None,
                Some(false),  // CPU
                None,
            ).await
        })
    });

    group.bench_function("gpu", |b| {
        b.to_async(&rt).iter(|| async {
            neural_forecast(
                black_box("AAPL".to_string()),
                black_box(30),
                None,
                Some(true),  // GPU
                None,
            ).await
        })
    });

    group.finish();
}

criterion_group!(benches, bench_execute_trade, bench_neural_forecast_cpu_vs_gpu);
criterion_main!(benches);
```

**Metrics to Measure**:
- [ ] Latency (p50, p95, p99)
- [ ] Throughput (ops/sec)
- [ ] CPU vs GPU speedup
- [ ] Memory usage
- [ ] Binary load time

**Acceptance Criteria**:
- [ ] All functions meet latency targets from architecture doc
- [ ] GPU shows >10x speedup on applicable operations
- [ ] Memory usage <1GB for typical workloads

---

### 8. TODO/FIXME Cleanup

**Status**: ❌ **NOT STARTED**
**Priority**: P0
**Impact**: Resolves known issues
**Effort**: 2 days
**Assignee**: Original Authors

**Current Count**: 32 TODO comments in NAPI bindings

**Process**:
```bash
# Find all TODOs
grep -rn "TODO\|FIXME\|XXX\|HACK" crates/napi-bindings/src/ > todos.txt

# For each TODO:
# 1. Create GitHub issue
# 2. Link to issue in comment
# 3. Fix immediately if P0, otherwise schedule
# 4. Remove comment once fixed
```

---

### 9. Async Cancellation Support

**Status**: ❌ **NOT STARTED**
**Priority**: P0
**Impact**: Prevents resource leaks
**Effort**: 3 days
**Assignee**: Async Team

**Problem**: Long-running operations can't be cancelled

**Solution**:
```rust
use tokio::time::{timeout, Duration};
use tokio::select;

#[napi]
pub async fn neural_train(...) -> ToolResult {
    let (cancel_tx, mut cancel_rx) = tokio::sync::oneshot::channel();

    // Store cancel_tx in global map keyed by task ID
    let task_id = register_task(cancel_tx);

    let training_future = actual_training();

    let result = select! {
        res = training_future => res,
        _ = cancel_rx => Err(Error::from_reason("Training cancelled")),
        _ = tokio::time::sleep(Duration::from_secs(3600)) => {
            Err(Error::from_reason("Training timed out after 1 hour"))
        }
    };

    unregister_task(task_id);
    result
}

// Cancellation API
#[napi]
pub async fn cancel_task(task_id: String) -> ToolResult {
    if let Some(cancel_tx) = get_cancel_sender(&task_id) {
        cancel_tx.send(()).ok();
        Ok(json!({"status": "cancelled", "task_id": task_id}).to_string())
    } else {
        Err(Error::from_reason("Task not found"))
    }
}
```

---

### 10. Rate Limiting Implementation

**Status**: ❌ **NOT STARTED**
**Priority**: P0 (Security)
**Impact**: Prevents DoS attacks
**Effort**: 2 days
**Assignee**: Security Team

**Solution**: See CVE-2025-NT-008 in SECURITY_AUDIT.md

**Implementation**:
```rust
use governor::{Quota, RateLimiter};

lazy_static! {
    static ref RATE_LIMITERS: RateLimiters = RateLimiters::new();
}

struct RateLimiters {
    general: RateLimiter<...>,
    expensive: RateLimiter<...>,
}

// Apply to all NAPI functions
#[napi]
pub async fn expensive_operation(...) -> ToolResult {
    RATE_LIMITERS.expensive.check()
        .map_err(|_| Error::from_reason("Rate limit exceeded"))?;
    // ... proceed
}
```

---

## P1 - High Priority (Fix Within 1 Week)

### 11. GPU Integration Implementation

**Status**: ❌ **NOT STARTED**
**Priority**: P1
**Effort**: 2 weeks

**Tasks**:
- [ ] Implement GPU detection and initialization
- [ ] Add CPU/GPU fallback logic
- [ ] Benchmark GPU vs CPU performance
- [ ] Document GPU requirements

---

### 12. Error Sanitization

**Status**: ❌ **NOT STARTED**
**Priority**: P1 (Security)
**Effort**: 2 days

See CVE-2025-NT-004 for details.

---

### 13. Constant-Time Comparison for Secrets

**Status**: ❌ **NOT STARTED**
**Priority**: P1 (Security)
**Effort**: 1 day

See CVE-2025-NT-005 for details.

---

### 14. Documentation Generation

**Status**: ❌ **NOT STARTED**
**Priority**: P1
**Effort**: 1 week

**Tasks**:
- [ ] Add usage examples to all NAPI functions
- [ ] Generate TypeScript types from Rust
- [ ] Create API reference documentation
- [ ] Write migration guide

---

### 15-24. [Other P1 Items]

(See DEEP_CODE_REVIEW.md for complete list)

---

## P2 - Medium Priority (Fix Within 1 Month)

### 25. Binary Size Optimization

**Status**: ❌ **NOT STARTED**
**Priority**: P2
**Effort**: 3 days

**Goal**: Reduce from 214MB to <100MB

**Strategies**:
```toml
[profile.release]
opt-level = "z"  # Optimize for size
lto = "fat"
strip = "symbols"
panic = "abort"
```

---

### 26-35. [Other P2 Items]

(See DEEP_CODE_REVIEW.md for complete list)

---

## Timeline

### Week 1 (Current)
- [ ] Fix binary duplication (rebuild with new structure)
- [ ] Input validation framework
- [ ] Remove .unwrap() calls
- [ ] Fix CVE-2025-NT-001, 002, 004

### Week 2
- [ ] Fix CVE-2025-NT-003, 005
- [ ] Implement rate limiting
- [ ] Async cancellation support
- [ ] Start integration tests

### Week 3-4
- [ ] Complete integration tests
- [ ] Performance benchmarking
- [ ] GPU integration
- [ ] TODO cleanup

### Weeks 5-16
- [ ] Phase 1-4 real implementation (per architecture doc)

---

## Success Criteria

### Before Production Deployment

**Must Have (P0)**:
- ✅ All security vulnerabilities fixed
- ✅ Input validation on all functions
- ✅ No .unwrap() calls
- ✅ Integration tests >80% coverage
- ✅ Performance benchmarks meet targets

**Should Have (P1)**:
- ✅ GPU integration functional
- ✅ All functions use real backends
- ✅ Documentation complete
- ✅ Error handling consistent

**Nice to Have (P2)**:
- Binary size <100MB
- TypeScript types auto-generated
- Monitoring dashboard

---

## Progress Tracking

**Weekly Updates**: Every Friday
**Blockers**: Report immediately
**Questions**: GitHub Discussions

**Current Status**:
- P0 Completed: 1/10 (10%)
- P1 Completed: 0/14 (0%)
- P2 Completed: 0/11 (0%)

**Overall Progress**: 3% (1/35 items complete)

---

**Document Status**: Active
**Last Updated**: 2025-11-14
**Next Review**: 2025-11-21 (after Week 1)
