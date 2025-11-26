# Risk Register and Fallback Procedures

## Document Purpose

This document provides a **comprehensive risk register, mitigation strategies, and fallback procedures** for the Neural Rust port. It ensures the project can handle failures gracefully and recover from disasters.

## Table of Contents

1. [Risk Register](#risk-register)
2. [Fallback Procedures](#fallback-procedures)
3. [Rollback Steps](#rollback-steps)
4. [Disaster Recovery](#disaster-recovery)
5. [Circuit Breakers](#circuit-breakers)
6. [Graceful Degradation](#graceful-degradation)
7. [Incident Response](#incident-response)
8. [Post-Mortem Template](#post-mortem-template)
9. [Business Continuity](#business-continuity)

---

## Risk Register

### Risk Assessment Matrix

**Probability Scale:**
- Low (L): 0-30% chance
- Medium (M): 31-60% chance
- High (H): 61-100% chance

**Impact Scale:**
- Low: Minor delays, <1 week
- Medium: Significant delays, 1-4 weeks
- High: Critical delays, >4 weeks or project failure

---

### R001: Rust-Node Interop Failure

**Category:** Technical
**Probability:** Medium (40%)
**Impact:** High (4 weeks delay)
**Risk Score:** 16/25

**Description:**
napi-rs bindings fail to properly bridge Rust async code to Node.js Promises, causing crashes or memory leaks when calling Rust functions from Node.

**Indicators:**
- Segmentation faults in Node.js process
- Memory leaks detected by Valgrind
- Promise rejections without error messages
- Garbage collection triggering Rust panics

**Mitigation:**
1. **Early Validation** (Week 3)
   - Prototype napi-rs bindings with async Tokio code
   - Test with memory leak detectors (Valgrind, AddressSanitizer)
   - Verify Promise lifecycle aligns with Rust Future lifecycle

2. **Fallback Approach:**
   - Use IPC (Inter-Process Communication) instead of FFI
   - Run Rust as separate process, communicate via stdin/stdout
   - Use JSON-RPC protocol for type safety

3. **Reference Implementation:**
   ```rust
   // Safe async bridge pattern
   #[napi]
   pub async fn process_signal(signal: JsSignal) -> Result<JsOrder> {
       // Convert JS types to Rust
       let rust_signal = signal.try_into()?;

       // Spawn Tokio task (detached from JS runtime)
       let result = tokio::spawn(async move {
           process_signal_internal(rust_signal).await
       }).await??;

       // Convert back to JS
       Ok(result.into())
   }
   ```

**Contingency Plan:**
- Budget: +2 weeks if IPC fallback needed
- Cost: $4,000 additional development time
- Decision Point: Week 5 (after prototype testing)

**Owner:** Backend Developer

---

### R002: Performance Target Shortfall

**Category:** Technical
**Probability:** Medium (50%)
**Impact:** Medium (2 weeks)
**Risk Score:** 12/25

**Description:**
Rust implementation fails to achieve 10x performance improvement target, potentially due to suboptimal algorithms, excessive allocations, or poor async coordination.

**Indicators:**
- Benchmark results <5x Python baseline
- p95 latency >500ms (target: 200ms)
- Memory usage >2GB (target: 1GB)
- Throughput <50K events/sec (target: 100K)

**Mitigation:**
1. **Continuous Benchmarking** (Weekly)
   - Run criterion benchmarks on every PR
   - Compare against Python baseline
   - Profile with `perf` and `flamegraph`

2. **Performance Optimization Checklist:**
   ```rust
   // ✅ Use zero-copy where possible
   fn process_data(data: &[u8]) -> Result<Output> {
       // Avoid: let owned = data.to_vec();
       // Instead: work with slice directly
   }

   // ✅ Pre-allocate collections
   let mut results = Vec::with_capacity(expected_size);

   // ✅ Use parallel iterators for CPU-bound work
   use rayon::prelude::*;
   let signals: Vec<_> = strategies.par_iter()
       .map(|s| s.process(&data))
       .collect();

   // ✅ Cache expensive computations
   use moka::sync::Cache;
   let cache = Cache::new(10_000);
   ```

3. **Fallback Targets:**
   - Minimum Acceptable: 5x improvement (p95 < 400ms)
   - Stretch Goal: 20x improvement (p95 < 100ms)

**Contingency Plan:**
- Budget: +1 week for profiling and optimization
- Tools: perf, valgrind, cachegrind, flamegraph
- Decision Point: Week 12 (end of Phase 2)

**Owner:** Performance Engineer

---

### R003: Data Provider Outage

**Category:** External Dependency
**Probability:** High (70%)
**Impact:** Low (hours)
**Risk Score:** 14/25

**Description:**
Primary data provider (Alpaca, Polygon, Yahoo Finance) experiences downtime, causing trading strategies to fail due to lack of market data.

**Indicators:**
- WebSocket disconnections lasting >1 minute
- HTTP 503/504 errors from REST API
- Stale data (no updates for >5 minutes)
- Ping timeouts

**Mitigation:**
1. **Multi-Provider Architecture:**
   ```rust
   pub struct MarketDataAggregator {
       primary: Box<dyn MarketDataProvider>,
       fallbacks: Vec<Box<dyn MarketDataProvider>>,
       health_checker: HealthChecker,
   }

   impl MarketDataAggregator {
       pub async fn get_quote(&self, symbol: &str) -> Result<Quote> {
           // Try primary
           match self.primary.get_quote(symbol).await {
               Ok(quote) => return Ok(quote),
               Err(e) => {
                   tracing::warn!("Primary provider failed: {}, trying fallbacks", e);
               }
           }

           // Try fallbacks in order
           for provider in &self.fallbacks {
               if let Ok(quote) = provider.get_quote(symbol).await {
                   return Ok(quote);
               }
           }

           Err(MarketDataError::AllProvidersFailed)
       }
   }
   ```

2. **Circuit Breaker Pattern:**
   ```rust
   use failsafe::{CircuitBreaker, Config};

   let breaker = CircuitBreaker::new(
       Config::new()
           .failure_threshold(5)
           .success_threshold(2)
           .timeout(Duration::from_secs(30))
   );

   let result = breaker.call(|| {
       provider.get_quote(symbol)
   }).await;
   ```

3. **Data Caching:**
   - Cache last known good quotes (TTL: 5 minutes)
   - Use cached data if all providers down
   - Warn user of stale data

**Fallback Order:**
1. Alpaca (primary)
2. Polygon (real-time fallback)
3. Yahoo Finance (delayed fallback)
4. Cached data (last resort)

**Contingency Plan:**
- Monitor provider health every 10 seconds
- Auto-switch to fallback on 3 consecutive failures
- Alert user when using degraded data sources

**Owner:** Data Engineer

---

### R004: Security Breach

**Category:** Security
**Probability:** Low (20%)
**Impact:** High (Critical)
**Risk Score:** 20/25

**Description:**
API keys, secrets, or database credentials are compromised through code leaks, supply chain attacks, or runtime exploitation.

**Indicators:**
- Unexpected API usage patterns
- Unauthorized trades detected
- Secrets leaked in logs or error messages
- Dependency audit warnings

**Mitigation:**
1. **Secrets Management:**
   ```rust
   // ❌ NEVER DO THIS
   const API_KEY: &str = "sk_live_abc123";

   // ✅ ALWAYS DO THIS
   use std::env;

   fn load_api_key() -> Result<String, ConfigError> {
       env::var("ALPACA_API_KEY")
           .map_err(|_| ConfigError::MissingSecret("ALPACA_API_KEY"))
   }
   ```

2. **Secret Scanning:**
   ```yaml
   # .github/workflows/security.yml
   - name: Secret Scanning
     run: |
       pip install detect-secrets
       detect-secrets scan --all-files --force-use-all-plugins
   ```

3. **Dependency Auditing:**
   ```bash
   # Run daily
   cargo audit --deny warnings
   cargo deny check licenses
   cargo deny check bans
   cargo deny check sources
   ```

4. **Principle of Least Privilege:**
   - Use read-only API keys for market data
   - Use paper trading keys for testing
   - Rotate production keys every 90 days

**Incident Response Plan:**
1. **Immediate** (0-15 minutes):
   - Revoke compromised credentials
   - Disable affected services
   - Close all open positions

2. **Short-term** (15 minutes - 4 hours):
   - Generate new credentials
   - Update configuration
   - Restart services with new keys
   - Audit trade history for unauthorized activity

3. **Long-term** (4 hours - 1 week):
   - Root cause analysis
   - Implement additional safeguards
   - Report to stakeholders
   - Update security procedures

**Contingency Plan:**
- Emergency credential rotation runbook (see below)
- Pre-configured backup API keys
- Automated shutdown script

**Owner:** Security Engineer

---

### R005: Database Corruption

**Category:** Data Integrity
**Probability:** Low (15%)
**Impact:** High (3 days)
**Risk Score:** 15/25

**Description:**
PostgreSQL or SQLite database becomes corrupted due to disk failure, power loss, or buggy queries, causing data loss or system crashes.

**Indicators:**
- Database connection errors
- Query timeouts or deadlocks
- Checksum validation failures
- Unexpected constraint violations

**Mitigation:**
1. **Write-Ahead Logging (WAL):**
   ```sql
   -- Enable WAL mode (PostgreSQL)
   ALTER SYSTEM SET wal_level = 'replica';
   ALTER SYSTEM SET archive_mode = 'on';

   -- Enable WAL mode (SQLite)
   PRAGMA journal_mode = WAL;
   ```

2. **Automated Backups:**
   ```bash
   #!/bin/bash
   # scripts/backup_database.sh

   DATE=$(date +%Y%m%d_%H%M%S)
   BACKUP_FILE="backups/neural_trader_${DATE}.sql.gz"

   # PostgreSQL backup
   pg_dump neural_trader | gzip > "$BACKUP_FILE"

   # Verify backup integrity
   gunzip -t "$BACKUP_FILE"

   # Upload to S3
   aws s3 cp "$BACKUP_FILE" s3://neural-trader-backups/

   # Retain last 30 days
   find backups/ -mtime +30 -delete
   ```

3. **Connection Pooling:**
   ```rust
   use sqlx::postgres::PgPoolOptions;

   let pool = PgPoolOptions::new()
       .max_connections(20)
       .acquire_timeout(Duration::from_secs(5))
       .idle_timeout(Duration::from_secs(600))
       .connect(&database_url).await?;
   ```

**Recovery Procedure:**
1. Stop all services writing to database
2. Attempt repair:
   ```bash
   # PostgreSQL
   pg_dump neural_trader > backup_before_repair.sql
   psql neural_trader -c "REINDEX DATABASE neural_trader;"

   # SQLite
   sqlite3 neural_trader.db ".recover" | sqlite3 recovered.db
   ```
3. If repair fails, restore from backup:
   ```bash
   dropdb neural_trader
   createdb neural_trader
   gunzip -c latest_backup.sql.gz | psql neural_trader
   ```
4. Validate data integrity
5. Resume services

**Contingency Plan:**
- Hourly automated backups
- Point-in-time recovery enabled
- Replica database for failover
- RPO: 1 hour (max data loss)
- RTO: 30 minutes (max downtime)

**Owner:** Database Administrator

---

### R006: Dependency Supply Chain Attack

**Category:** Security
**Probability:** Low (10%)
**Impact:** High (Critical)
**Risk Score:** 10/25

**Description:**
A malicious actor compromises a Rust crate or npm package in the dependency tree, injecting backdoors or malware.

**Indicators:**
- Unexpected network requests
- Cargo audit warnings
- Changed binary checksums
- Suspicious dependency updates

**Mitigation:**
1. **Dependency Pinning:**
   ```toml
   # Cargo.toml
   [dependencies]
   tokio = "=1.35.0"  # Exact version, not "^1.35"
   reqwest = "=0.11.23"

   # Use Cargo.lock in version control
   ```

2. **Cargo-deny Configuration:**
   ```toml
   # deny.toml
   [advisories]
   vulnerability = "deny"
   unmaintained = "warn"
   notice = "warn"

   [licenses]
   allow = ["MIT", "Apache-2.0", "BSD-3-Clause"]
   deny = ["GPL", "AGPL"]

   [bans]
   multiple-versions = "deny"
   wildcards = "deny"
   ```

3. **SBOM Generation:**
   ```bash
   # Generate Software Bill of Materials
   cargo install cargo-sbom
   cargo sbom > sbom.json

   # Audit SBOM
   syft sbom.json -o table
   ```

**Response Plan:**
1. Immediate: Freeze dependency updates
2. Audit: Review recent changes with `cargo tree`
3. Isolate: Remove compromised dependency
4. Replace: Find secure alternative or vendor code
5. Validate: Re-run all security scans

**Contingency Plan:**
- Vendored dependencies for critical crates
- Internal crate mirror
- Binary reproducibility verification

**Owner:** Security Engineer

---

### R007: Node.js Version Incompatibility

**Category:** Compatibility
**Probability:** Medium (30%)
**Impact:** Low (1 week)
**Risk Score:** 6/25

**Description:**
napi-rs bindings fail on certain Node.js versions (18, 20, 22) or operating systems due to ABI changes or missing APIs.

**Indicators:**
- "Cannot find module" errors
- Native module load failures
- Crashes on specific Node versions
- Windows-specific compilation errors

**Mitigation:**
1. **Comprehensive CI Matrix:**
   - Test all Node.js LTS versions (18, 20, 22)
   - Test all platforms (Linux, macOS, Windows)
   - Test all architectures (x64, arm64)

2. **N-API Version Compatibility:**
   ```toml
   # Cargo.toml
   [dependencies]
   napi = "2"
   napi-derive = "2"

   [build-dependencies]
   napi-build = "2"
   ```

3. **Platform-Specific Packages:**
   ```json
   {
     "optionalDependencies": {
       "@neural-trader/linux-x64-gnu": "1.0.0",
       "@neural-trader/darwin-x64": "1.0.0",
       "@neural-trader/darwin-arm64": "1.0.0",
       "@neural-trader/win32-x64-msvc": "1.0.0"
     }
   }
   ```

**Fallback Strategy:**
- Ship pre-compiled binaries for all platforms
- Use Node-API version 6 (compatible with Node 16+)
- Provide WASM fallback for unsupported platforms

**Contingency Plan:**
- Pre-release beta testing on all platforms
- Community testing program
- Rollback to last stable version

**Owner:** Release Engineer

---

### R008: Regulatory Compliance Violation

**Category:** Legal/Compliance
**Probability:** Low (25%)
**Impact:** High (Project halt)
**Risk Score:** 12.5/25

**Description:**
Trading system violates financial regulations (SEC, FINRA, etc.) due to insufficient audit trails, improper order handling, or unauthorized trading.

**Indicators:**
- Missing trade logs
- Incorrect order timestamps
- Best execution violations
- Pattern day trading violations

**Mitigation:**
1. **Comprehensive Audit Logging:**
   ```rust
   use tracing::{info, warn};

   #[tracing::instrument(skip(order))]
   async fn place_order(order: &Order) -> Result<OrderResponse> {
       info!(
           order_id = %order.id,
           symbol = %order.symbol,
           qty = %order.qty,
           side = ?order.side,
           "Placing order"
       );

       let response = broker.place_order(order).await?;

       info!(
           order_id = %response.id,
           status = ?response.status,
           filled_qty = %response.filled_qty,
           "Order placed successfully"
       );

       Ok(response)
   }
   ```

2. **Regulatory Checks:**
   ```rust
   pub struct ComplianceEngine {
       pdt_checker: PatternDayTradeChecker,
       wash_sale_detector: WashSaleDetector,
       position_limits: PositionLimits,
   }

   impl ComplianceEngine {
       pub async fn validate_order(&self, order: &Order) -> Result<(), ComplianceError> {
           // Check pattern day trading rules
           if self.pdt_checker.would_violate(order)? {
               return Err(ComplianceError::PatternDayTradingViolation);
           }

           // Check wash sale rules
           if self.wash_sale_detector.detect(order)? {
               warn!("Potential wash sale detected");
           }

           // Check position limits
           self.position_limits.validate(order)?;

           Ok(())
       }
   }
   ```

3. **Legal Review:**
   - Consult securities attorney before launch
   - Document all regulatory requirements
   - Implement required disclosures

**Contingency Plan:**
- Disable live trading until compliance verified
- Switch to paper trading mode
- Engage legal counsel immediately

**Owner:** Compliance Officer

---

### R009: Memory Leak in Long-Running Process

**Category:** Technical
**Probability:** Medium (40%)
**Impact:** Medium (2 weeks)
**Risk Score:** 16/25

**Description:**
Rust process leaks memory over time due to circular references (Arc<Mutex<>>), forgotten channels, or unbounded caches.

**Indicators:**
- RSS memory growth over days
- OOM (Out of Memory) kills
- Slow garbage collection in Node.js side
- File descriptor exhaustion

**Mitigation:**
1. **Memory Leak Detection:**
   ```bash
   # Run with AddressSanitizer
   RUSTFLAGS="-Z sanitizer=address" cargo +nightly run

   # Run with Valgrind
   valgrind --leak-check=full --show-leak-kinds=all ./target/debug/neural-trader

   # Profile memory usage
   heaptrack ./target/debug/neural-trader
   ```

2. **Bounded Collections:**
   ```rust
   // ❌ BAD: Unbounded cache
   use std::collections::HashMap;
   let mut cache: HashMap<String, Data> = HashMap::new();

   // ✅ GOOD: LRU cache with size limit
   use lru::LruCache;
   let mut cache: LruCache<String, Data> = LruCache::new(10_000);
   ```

3. **Resource Cleanup:**
   ```rust
   use tokio::sync::mpsc;

   pub struct MarketDataManager {
       tx: mpsc::Sender<MarketTick>,
       rx: mpsc::Receiver<MarketTick>,
       tasks: Vec<JoinHandle<()>>,
   }

   impl Drop for MarketDataManager {
       fn drop(&mut self) {
           // Cancel all background tasks
           for task in &self.tasks {
               task.abort();
           }
       }
   }
   ```

**Monitoring:**
```rust
// Expose memory metrics
use prometheus::{Gauge, register_gauge};

lazy_static! {
    static ref MEMORY_USAGE: Gauge = register_gauge!(
        "process_memory_bytes",
        "Current memory usage in bytes"
    ).unwrap();
}

// Update periodically
tokio::spawn(async {
    loop {
        let usage = get_memory_usage();
        MEMORY_USAGE.set(usage as f64);
        tokio::time::sleep(Duration::from_secs(60)).await;
    }
});
```

**Contingency Plan:**
- Restart service every 24 hours as preventive measure
- Set memory limit (e.g., 2GB) and graceful shutdown on approach
- Rollback to previous version if leak persists

**Owner:** Backend Developer

---

## Fallback Procedures

### Fallback 1: Switch to Paper Trading

**Trigger:** Production issues detected, capital at risk

**Procedure:**
```bash
# 1. Pause live trading immediately
neural-trader pause --mode live

# 2. Cancel all open orders
neural-trader cancel-all --confirm

# 3. Close all positions (optional, use with caution)
neural-trader close-all --confirm

# 4. Switch to paper trading
neural-trader config set execution_mode paper

# 5. Restart in paper mode
neural-trader restart --mode paper

# 6. Verify paper mode active
neural-trader status | grep "Execution Mode: Paper"
```

**Rollback:** Resume live trading after issue resolved
```bash
neural-trader config set execution_mode live
neural-trader restart --mode live
```

---

### Fallback 2: Use Cached Market Data

**Trigger:** All market data providers down

**Procedure:**
```rust
// Automatic fallback in code
impl MarketDataManager {
    async fn get_quote(&self, symbol: &str) -> Result<Quote> {
        // Try live providers
        for provider in &self.providers {
            if let Ok(quote) = provider.get_quote(symbol).await {
                // Cache for future use
                self.cache.insert(symbol, quote.clone()).await;
                return Ok(quote);
            }
        }

        // Fallback to cached data
        if let Some(quote) = self.cache.get(symbol).await {
            warn!("Using cached data for {}, age: {:?}", symbol, quote.age());
            return Ok(quote);
        }

        Err(MarketDataError::NoDataAvailable)
    }
}
```

**User Warning:**
```
⚠️  WARNING: All market data providers unavailable.
Using cached data (last updated: 2 minutes ago).
Trading signals may be stale. Consider pausing trading.
```

---

### Fallback 3: IPC Instead of FFI

**Trigger:** napi-rs bindings fail (segfaults, memory leaks)

**Procedure:**
1. Build Rust as standalone binary
2. Use Node.js child_process to spawn
3. Communicate via JSON over stdin/stdout

```javascript
// Node.js side
const { spawn } = require('child_process');

class RustBridge {
  constructor() {
    this.process = spawn('./target/release/neural-trader', ['--ipc-mode']);
    this.requestId = 0;
    this.pending = new Map();
  }

  async call(method, params) {
    const id = this.requestId++;
    const request = JSON.stringify({ id, method, params });

    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.process.stdin.write(request + '\n');
    });
  }
}
```

```rust
// Rust side
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

#[tokio::main]
async fn main() -> Result<()> {
    let stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    let mut lines = tokio::io::BufReader::new(stdin).lines();

    while let Some(line) = lines.next_line().await? {
        let request: RpcRequest = serde_json::from_str(&line)?;
        let response = handle_request(request).await?;

        stdout.write_all(serde_json::to_string(&response)?.as_bytes()).await?;
        stdout.write_all(b"\n").await?;
        stdout.flush().await?;
    }

    Ok(())
}
```

**Trade-offs:**
- ✅ More stable (no FFI segfaults)
- ✅ Better isolation (crash isolation)
- ❌ Slower (~1ms overhead per call)
- ❌ More complex deployment

---

## Rollback Steps

### Rollback 1: NPM Package

**Scenario:** New release has critical bug

**Steps:**
```bash
# 1. Deprecate buggy version
npm deprecate neural-trader@1.2.0 "Critical bug, use 1.1.0 instead"

# 2. Publish fixed version or rollback
npm publish --tag latest

# 3. Users can pin to last good version
npm install neural-trader@1.1.0
```

### Rollback 2: Database Schema

**Scenario:** Migration causes issues

**Steps:**
```bash
# 1. Stop all services
systemctl stop neural-trader

# 2. Rollback migration
sqlx migrate revert

# 3. Restore from backup if needed
gunzip -c backups/before_migration.sql.gz | psql neural_trader

# 4. Deploy previous code version
git checkout v1.1.0
cargo build --release

# 5. Restart services
systemctl start neural-trader
```

### Rollback 3: Configuration Change

**Scenario:** Config change breaks system

**Steps:**
```bash
# 1. Revert config file
git checkout HEAD~1 config/production.toml

# 2. Restart with old config
neural-trader restart --config config/production.toml

# 3. Validate system healthy
neural-trader health-check
```

---

## Disaster Recovery

### Scenario 1: Complete Data Center Failure

**RTO:** 4 hours
**RPO:** 1 hour

**Procedure:**
1. **Immediate (0-15 min):**
   - Activate backup data center
   - Restore database from latest backup
   - Deploy application from container registry

2. **Short-term (15 min - 1 hour):**
   - Restore configuration from Git
   - Restore secrets from vault
   - Verify data integrity

3. **Resume Operations (1-4 hours):**
   - Start in paper trading mode
   - Run health checks
   - Switch to live trading after validation

```bash
#!/bin/bash
# scripts/disaster_recovery.sh

# Restore database
aws s3 cp s3://neural-trader-backups/latest.sql.gz .
gunzip latest.sql.gz
psql -h backup-db.example.com -f latest.sql

# Deploy application
docker pull neural-trader:latest
docker run -d --env-file .env neural-trader:latest

# Validate
curl http://localhost:8080/health
```

---

### Scenario 2: Ransomware Attack

**RTO:** 8 hours
**RPO:** 24 hours

**Procedure:**
1. **Isolate (0-30 min):**
   - Disconnect affected systems from network
   - Preserve forensic evidence
   - Notify security team and stakeholders

2. **Assess (30 min - 2 hours):**
   - Identify extent of compromise
   - Determine if backups are clean
   - Evaluate ransom vs. recovery cost

3. **Recover (2-8 hours):**
   - Restore from clean backups
   - Rebuild compromised systems from scratch
   - Change all credentials
   - Apply security patches

4. **Validate (8+ hours):**
   - Run malware scans
   - Verify data integrity
   - Monitor for re-infection

**Do NOT:**
- Pay ransom without consulting legal/security
- Reconnect systems before full cleaning
- Use potentially compromised backups

---

## Circuit Breakers

### Implementation

```rust
use failsafe::{CircuitBreaker, Config, failure_policy};

pub struct ResilientBrokerClient {
    client: AlpacaClient,
    breaker: CircuitBreaker,
}

impl ResilientBrokerClient {
    pub fn new(client: AlpacaClient) -> Self {
        let breaker = CircuitBreaker::new(
            Config::new()
                .failure_threshold(5)      // Open after 5 failures
                .success_threshold(2)      // Close after 2 successes
                .timeout(Duration::from_secs(60))  // Stay open for 60s
        );

        Self { client, breaker }
    }

    pub async fn place_order(&self, order: &Order) -> Result<OrderResponse> {
        self.breaker.call(|| async {
            self.client.place_order(order).await
        }).await
    }
}
```

### Circuit States

**CLOSED (Normal):**
- All requests pass through
- Failures increment counter
- Open on threshold

**OPEN (Failing):**
- All requests immediately fail
- No requests to downstream service
- Periodic health check

**HALF-OPEN (Testing):**
- Limited requests pass through
- Success → CLOSED
- Failure → OPEN

---

## Graceful Degradation

### Feature Prioritization

**Tier 1 (Critical - Always On):**
- Market data ingestion
- Risk management
- Position tracking

**Tier 2 (Important - Degrade on Load):**
- Signal generation
- Order placement
- Portfolio analytics

**Tier 3 (Optional - Disable on Load):**
- Historical backtesting
- Neural forecasting
- Advanced analytics

### Load Shedding

```rust
use governor::{Quota, RateLimiter};

pub struct LoadShedder {
    rate_limiter: RateLimiter<
        governor::state::direct::NotKeyed,
        governor::state::InMemoryState,
        governor::clock::DefaultClock,
    >,
}

impl LoadShedder {
    pub fn new(requests_per_sec: u32) -> Self {
        let quota = Quota::per_second(requests_per_sec);
        Self {
            rate_limiter: RateLimiter::direct(quota),
        }
    }

    pub async fn acquire(&self) -> Result<(), LoadSheddingError> {
        match self.rate_limiter.check() {
            Ok(_) => Ok(()),
            Err(_) => Err(LoadSheddingError::RateLimitExceeded),
        }
    }
}

// Usage
#[axum::debug_handler]
async fn handle_request(
    Extension(load_shedder): Extension<Arc<LoadShedder>>,
    Json(request): Json<Request>,
) -> Result<Json<Response>, StatusCode> {
    // Shed load if over capacity
    load_shedder.acquire().await
        .map_err(|_| StatusCode::TOO_MANY_REQUESTS)?;

    // Process request normally
    Ok(Json(process(request)))
}
```

---

## Incident Response

### Incident Severity Levels

**P0 (Critical):**
- Production system down
- Financial loss occurring
- Security breach
- **Response Time:** 15 minutes
- **Escalation:** Immediate

**P1 (High):**
- Major feature broken
- Performance degradation
- Data inconsistency
- **Response Time:** 1 hour
- **Escalation:** 2 hours

**P2 (Medium):**
- Minor feature broken
- Non-critical bug
- **Response Time:** 4 hours
- **Escalation:** 1 day

**P3 (Low):**
- Cosmetic issue
- Feature request
- **Response Time:** 1 week
- **Escalation:** None

### Incident Response Runbook

```markdown
# INCIDENT: [TITLE]
**Severity:** P0 | P1 | P2 | P3
**Detected:** YYYY-MM-DD HH:MM UTC
**Incident Commander:** [Name]
**Status:** Investigating | Mitigating | Resolved

## Timeline
- **HH:MM UTC:** Incident detected by [monitoring/user report]
- **HH:MM UTC:** Incident commander assigned
- **HH:MM UTC:** Root cause identified
- **HH:MM UTC:** Mitigation deployed
- **HH:MM UTC:** Incident resolved

## Impact
- **Users Affected:** [number/percentage]
- **Duration:** [minutes/hours]
- **Financial Impact:** $[amount]
- **Data Loss:** [yes/no, details]

## Root Cause
[Detailed explanation of what went wrong]

## Resolution
[What was done to fix the issue]

## Action Items
- [ ] Deploy permanent fix (Owner: [Name], Due: [Date])
- [ ] Update monitoring (Owner: [Name], Due: [Date])
- [ ] Update documentation (Owner: [Name], Due: [Date])
- [ ] Conduct post-mortem (Owner: [Name], Due: [Date])
```

---

## Post-Mortem Template

```markdown
# Post-Mortem: [Incident Title]

**Date:** YYYY-MM-DD
**Authors:** [Names]
**Reviewers:** [Names]
**Status:** Draft | In Review | Final

## Summary
[2-3 sentence summary of what happened]

## Impact
- **Duration:** [start time] to [end time] ([total duration])
- **Users Affected:** [number/percentage]
- **Financial Impact:** $[amount]
- **Reputation Impact:** [high/medium/low]

## Timeline (all times UTC)
| Time | Event |
|------|-------|
| HH:MM | Deployment of version X.Y.Z |
| HH:MM | First error alert triggered |
| HH:MM | Incident declared, IC assigned |
| HH:MM | Root cause identified |
| HH:MM | Rollback initiated |
| HH:MM | Service restored |
| HH:MM | All-clear declared |

## Root Cause
**Immediate Cause:**
[What directly caused the incident]

**Underlying Cause:**
[Systemic issues that allowed the incident to occur]

## What Went Wrong
1. [Problem 1]
2. [Problem 2]
3. [Problem 3]

## What Went Right
1. [Success 1]
2. [Success 2]

## Action Items
| Action | Owner | Priority | Due Date | Status |
|--------|-------|----------|----------|--------|
| Add integration test for scenario X | [Name] | P0 | YYYY-MM-DD | ✅ Done |
| Implement circuit breaker for API Y | [Name] | P0 | YYYY-MM-DD | 🔄 In Progress |
| Update runbook with new procedure | [Name] | P1 | YYYY-MM-DD | ⏳ TODO |

## Lessons Learned
1. [Lesson 1]
2. [Lesson 2]
3. [Lesson 3]

## Prevention
**Short-term:**
- [Action 1]
- [Action 2]

**Long-term:**
- [Systemic improvement 1]
- [Systemic improvement 2]
```

---

## Business Continuity

### Critical Functions

1. **Market Data Ingestion:** RTO 5 minutes
2. **Risk Management:** RTO 5 minutes
3. **Order Execution:** RTO 15 minutes
4. **Position Tracking:** RTO 30 minutes

### Backup Personnel

| Role | Primary | Backup 1 | Backup 2 |
|------|---------|----------|----------|
| Incident Commander | [Name] | [Name] | [Name] |
| Backend Engineer | [Name] | [Name] | [Name] |
| Database Admin | [Name] | [Name] | [Name] |
| Security Engineer | [Name] | [Name] | [Name] |

### Communication Plan

**Internal:**
- Slack #incidents channel
- PagerDuty escalation
- Status page updates

**External:**
- Status page (status.neural-trader.io)
- Email notifications to affected users
- Social media updates (if needed)

---

## Acceptance Criteria

- [ ] All risks documented with probability, impact, mitigation
- [ ] Fallback procedures tested in staging
- [ ] Rollback scripts automated and tested
- [ ] Disaster recovery plan validated
- [ ] Circuit breakers implemented in critical paths
- [ ] Graceful degradation tested under load
- [ ] Incident response runbooks created
- [ ] Post-mortem template approved
- [ ] Business continuity plan validated

---

## Cross-References

- **Testing:** [13_Tests_Benchmarks_CI.md](./13_Tests_Benchmarks_CI.md) - Test failure handling
- **Security:** [08_Security_Governance_AIDefence_Lean.md](./08_Security_Governance_AIDefence_Lean.md) - Security measures
- **Roadmap:** [15_Roadmap_Phases_and_Milestones.md](./15_Roadmap_Phases_and_Milestones.md) - Timeline impact
- **Exchange Adapters:** [17_Exchange_Adapters_and_Data_Pipeline.md](./17_Exchange_Adapters_and_Data_Pipeline.md) - Provider failover

---

**Document Version:** 1.0.0
**Last Updated:** 2025-11-12
**Owner:** Risk Manager
**Status:** Complete
**Next Review:** 2025-11-19
