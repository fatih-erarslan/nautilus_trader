# Neural Trading Rust Port - Quick Reference Guide

**Fast lookup for tasks, dependencies, and critical paths**

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Duration** | 24 weeks (6 months) |
| **Total Person-Hours** | 1,960 hours |
| **Number of Tasks** | 30 GOAP tasks |
| **Critical Path Length** | 20 tasks (longest chain) |
| **Team Size** | 4-6 specialists (rotating) |
| **Total Investment** | ~$690 (research) + team costs |

---

## Phase Timeline at a Glance

```
Phase 0: Research          [██]                                 Weeks 1-2
Phase 1: MVP Core          [════████]                          Weeks 3-6
Phase 2: Full Parity       [════════████████]                  Weeks 7-12
Phase 3: Performance       [════════════████]                  Weeks 13-16
Phase 4: Distributed       [════════════════████]              Weeks 17-20
Phase 5: Release           [════════════════════████]          Weeks 21-24
```

---

## Critical Path (Longest Dependency Chain)

**20 tasks on critical path - optimize these to reduce timeline!**

```
GOAL-0-00-01 (Research)
    ↓ (1 week)
GOAL-0-00-02 (Analysis)
    ↓ (1 week)
GOAL-1-01-01 (Project Structure)
    ↓ (3 days)
GOAL-1-02-01 (Core Types)
    ↓ (5 days)
GOAL-1-02-02 (Error Handling)
    ↓ (2 days)
GOAL-1-03-01 (Configuration)
    ↓ (2 days)
GOAL-1-05-01 (Alpaca Client)
    ↓ (10 days)
GOAL-1-09-01 (Basic Strategy)
    ↓ (6 days)
GOAL-2-09-01 (All Strategies)
    ↓ (20 days)
GOAL-2-10-01 (Portfolio Management)
    ↓ (6 days)
GOAL-2-11-01 (Risk Management)
    ↓ (7 days)
GOAL-2-18-01 (Backtesting)
    ↓ (9 days)
GOAL-3-16-01 (GPU Acceleration)
    ↓ (12 days)
GOAL-3-17-01 (Performance Optimization)
    ↓ (10 days)
GOAL-5-17-01 (Deployment)
    ↓ (9 days)
GOAL-5-22-01 (Benchmarking)
    ↓ (5 days)
GOAL-5-23-01 (Security Audit)
    ↓ (7 days)
GOAL-5-24-01 (Production Release)
    ↓ (6 days)
✓ COMPLETE (Week 24)
```

**Total Critical Path:** 120 days (24 weeks)

---

## Parallel Execution Opportunities

### Week 3-6 (MVP Phase)
Can run in parallel:
- Track A: Core Types (GOAL-1-02-01) + Error Handling (GOAL-1-02-02)
- Track B: Configuration (GOAL-1-03-01)
- Track C: HTTP Server setup (GOAL-1-13-01 start)

### Week 7-12 (Full Parity)
Can run in parallel:
- Track A: News Collection (GOAL-2-06-01) → Sentiment (GOAL-2-08-01)
- Track B: Database (GOAL-2-14-01)
- Track C: Authentication (GOAL-2-12-01) → Complete API (GOAL-2-13-01)
- Track D: Portfolio (GOAL-2-10-01) → Risk (GOAL-2-11-01)

### Week 13-16 (Performance)
Can run in parallel:
- Track A: GPU Acceleration (GOAL-3-16-01)
- Track B: CPU Profiling (GOAL-3-17-01)
- Track C: Backtesting Optimization (GOAL-3-18-02)

### Week 21-24 (Release)
Can run in parallel:
- Track A: Testing (GOAL-5-15-01)
- Track B: Documentation (GOAL-5-21-01)
- Track C: Deployment prep (GOAL-5-17-01)

**Parallelization can reduce timeline from 24 weeks to ~16 weeks with 6-person team!**

---

## Task Risk Heatmap

### High Risk (8 tasks - 584 hours)
🔴 **GOAL-2-08-01** - Sentiment Analysis (ML inference speed risk)
🔴 **GOAL-2-09-01** - All Strategies (trading logic bugs)
🔴 **GOAL-2-11-01** - Risk Management (calculation errors)
🔴 **GOAL-2-12-01** - Authentication (security vulnerabilities)
🔴 **GOAL-3-16-01** - GPU Acceleration (complexity, portability)
🔴 **GOAL-4-19-01** - Multi-Node Architecture (distributed systems complexity)
🔴 **GOAL-4-20-01** - Multi-Tenant (data leakage risk)
🔴 **GOAL-5-23-01** - Security Audit (critical vulns found late)

### Medium Risk (12 tasks - 736 hours)
🟡 **GOAL-0-00-01** - Research (wrong tech choices)
🟡 **GOAL-1-02-01** - Core Types (type design errors cascade)
🟡 **GOAL-1-05-01** - Alpaca Client (API changes, rate limits)
🟡 **GOAL-1-09-01** - Basic Strategy (logic bugs)
🟡 **GOAL-1-11-01** - Portfolio Management (P&L calculation errors)
🟡 **GOAL-2-06-01** - News Collection (source failures, rate limits)
🟡 **GOAL-2-10-01** - Advanced Portfolio (optimization too slow)
🟡 **GOAL-2-13-01** - Complete API (compatibility breaks)
🟡 **GOAL-2-14-01** - Database (performance bottlenecks)
🟡 **GOAL-2-18-01** - Backtesting (results don't match production)
🟡 **GOAL-3-17-01** - Performance Optimization (premature optimization bugs)
🟡 **GOAL-5-17-01** - Deployment (deployment issues)

### Low Risk (10 tasks - 420 hours)
🟢 All other tasks

---

## Quick Task Lookup

### By Phase

#### Phase 0 (Weeks 1-2)
| Task | Duration | Owner |
|------|----------|-------|
| GOAL-0-00-01 | 40h | System Architect |
| GOAL-0-00-02 | 60h | Code Analyst |

#### Phase 1 (Weeks 3-6)
| Task | Duration | Owner |
|------|----------|-------|
| GOAL-1-01-01 | 24h | DevOps + Architect |
| GOAL-1-02-01 | 40h | Rust Developer |
| GOAL-1-02-02 | 20h | Rust Developer |
| GOAL-1-03-01 | 16h | Backend Developer |
| GOAL-1-05-01 | 80h | Backend + API Specialist |
| GOAL-1-09-01 | 48h | Quant + Rust Developer |
| GOAL-1-11-01 | 32h | Backend Developer |
| GOAL-1-13-01 | 40h | Backend Developer |

#### Phase 2 (Weeks 7-12)
| Task | Duration | Owner |
|------|----------|-------|
| GOAL-2-06-01 | 64h | Backend + Data Engineer |
| GOAL-2-08-01 | 80h | ML + Rust Engineer |
| GOAL-2-09-01 | 160h | Quant Developers (2) |
| GOAL-2-10-01 | 48h | Quant Developer |
| GOAL-2-11-01 | 56h | Risk + Rust Developer |
| GOAL-2-12-01 | 40h | Security + Backend |
| GOAL-2-13-01 | 64h | Backend Developer |
| GOAL-2-14-01 | 56h | Database + Backend |
| GOAL-2-18-01 | 72h | Quant + Performance Engineer |

#### Phase 3 (Weeks 13-16)
| Task | Duration | Owner |
|------|----------|-------|
| GOAL-3-16-01 | 96h | GPU/CUDA + ML Engineer |
| GOAL-3-17-01 | 80h | Performance Engineer |
| GOAL-3-18-02 | 48h | Performance + Quant |

#### Phase 4 (Weeks 17-20)
| Task | Duration | Owner |
|------|----------|-------|
| GOAL-4-19-01 | 64h | Architect + DevOps |
| GOAL-4-20-01 | 56h | Backend + Security |

#### Phase 5 (Weeks 21-24)
| Task | Duration | Owner |
|------|----------|-------|
| GOAL-5-15-01 | 80h | QA + Test Automation |
| GOAL-5-17-01 | 72h | DevOps + SRE |
| GOAL-5-21-01 | 64h | Technical Writer |
| GOAL-5-22-01 | 40h | Performance Engineer |
| GOAL-5-23-01 | 56h | Security + External Auditor |
| GOAL-5-24-01 | 48h | Project Manager + Team |

---

## By Module (18 Modules)

| Module | Primary Tasks | Duration |
|--------|---------------|----------|
| **00: README** | GOAL-0-00-01 | Week 1 |
| **01: Project Structure** | GOAL-1-01-01 | Week 3 |
| **02: Core Types** | GOAL-1-02-01, GOAL-1-02-02 | Week 3-4 |
| **03: Error Handling** | GOAL-1-02-02 | Week 4 |
| **04: Configuration** | GOAL-1-03-01 | Week 4 |
| **05: Trading API** | GOAL-1-05-01 | Week 4-5 |
| **06: News Collection** | GOAL-2-06-01 | Week 7-8 |
| **07: News Integration** | Part of GOAL-2-06-01 | Week 8 |
| **08: Sentiment** | GOAL-2-08-01 | Week 8-9 |
| **09: Strategies** | GOAL-1-09-01, GOAL-2-09-01 | Week 5-6, 9-10 |
| **10: Portfolio** | GOAL-1-11-01, GOAL-2-10-01 | Week 5, 10 |
| **11: Risk** | GOAL-2-11-01 | Week 10-11 |
| **12: Authentication** | GOAL-2-12-01 | Week 11 |
| **13: API Server** | GOAL-1-13-01, GOAL-2-13-01 | Week 5, 11-12 |
| **14: Database** | GOAL-2-14-01 | Week 11 |
| **15: Testing** | GOAL-5-15-01 | Week 21-22 |
| **16: Performance/GPU** | GOAL-3-16-01, GOAL-3-17-01 | Week 13-16 |
| **17: Deployment** | GOAL-5-17-01 | Week 21-22 |
| **18: Backtesting** | GOAL-2-18-01, GOAL-3-18-02 | Week 11-12, 16 |

---

## Dependency Matrix

| Task | Depends On | Blocks |
|------|------------|--------|
| GOAL-0-00-01 | - | GOAL-0-00-02 |
| GOAL-0-00-02 | GOAL-0-00-01 | GOAL-1-01-01 |
| GOAL-1-01-01 | GOAL-0-00-02 | GOAL-1-02-01 |
| GOAL-1-02-01 | GOAL-1-01-01 | GOAL-1-02-02, GOAL-1-05-01, etc. |
| GOAL-1-02-02 | GOAL-1-02-01 | GOAL-1-03-01 |
| GOAL-1-03-01 | GOAL-1-02-02 | GOAL-1-05-01, GOAL-2-06-01 |
| GOAL-1-05-01 | GOAL-1-03-01 | GOAL-1-09-01, GOAL-1-11-01 |
| GOAL-1-09-01 | GOAL-1-05-01 | GOAL-2-09-01 |
| GOAL-1-11-01 | GOAL-1-05-01 | GOAL-1-13-01, GOAL-2-10-01 |
| GOAL-1-13-01 | GOAL-1-11-01 | GOAL-2-12-01, GOAL-2-13-01 |
| GOAL-2-06-01 | GOAL-1-03-01 | GOAL-2-08-01 |
| GOAL-2-08-01 | GOAL-2-06-01 | GOAL-2-09-01 |
| GOAL-2-09-01 | GOAL-1-09-01, GOAL-2-08-01 | GOAL-2-10-01, GOAL-2-18-01 |
| GOAL-2-10-01 | GOAL-2-09-01 | GOAL-2-11-01 |
| GOAL-2-11-01 | GOAL-2-10-01 | GOAL-2-18-01 |
| GOAL-2-12-01 | GOAL-1-13-01 | GOAL-2-13-01 |
| GOAL-2-13-01 | GOAL-2-12-01 | Phase 2 Complete |
| GOAL-2-14-01 | GOAL-1-03-01 | - |
| GOAL-2-18-01 | GOAL-2-09-01, GOAL-2-11-01 | GOAL-3-18-02 |
| GOAL-3-16-01 | Phase 2 Complete | GOAL-3-17-01 |
| GOAL-3-17-01 | GOAL-3-16-01 | GOAL-3-18-02 |
| GOAL-3-18-02 | GOAL-2-18-01, GOAL-3-17-01 | Phase 3 Complete |
| GOAL-4-19-01 | Phase 3 Complete | GOAL-4-20-01 |
| GOAL-4-20-01 | GOAL-4-19-01 | Phase 4 Complete |
| GOAL-5-15-01 | Phase 4 Complete | GOAL-5-17-01 |
| GOAL-5-17-01 | GOAL-5-15-01 | GOAL-5-22-01 |
| GOAL-5-21-01 | Phase 4 Complete | GOAL-5-24-01 |
| GOAL-5-22-01 | GOAL-5-17-01 | GOAL-5-23-01 |
| GOAL-5-23-01 | GOAL-5-22-01 | GOAL-5-24-01 |
| GOAL-5-24-01 | GOAL-5-23-01, GOAL-5-21-01 | PROJECT COMPLETE |

---

## Resource Loading by Week

```
Week  1: ████░░ (2 people - Research)
Week  2: ████░░ (2 people - Analysis)
Week  3: ██████ (3 people - Project setup, Types, Config)
Week  4: ██████ (3 people - Types, Config, Alpaca client)
Week  5: ████████ (4 people - Alpaca, Strategy, Portfolio, API)
Week  6: ████████ (4 people - Strategy, Portfolio, API)
Week  7: ████████ (4 people - News collection, Database)
Week  8: ██████████ (5 people - News, Sentiment, Strategies)
Week  9: ██████████ (5 people - Sentiment, Strategies)
Week 10: ██████████ (5 people - Strategies, Portfolio, Risk)
Week 11: ████████████ (6 people - Risk, Auth, API, Database, Backtest)
Week 12: ████████████ (6 people - API, Backtesting)
Week 13: ████████ (4 people - GPU acceleration, Performance)
Week 14: ████████ (4 people - GPU, Performance)
Week 15: ████████ (4 people - Performance optimization)
Week 16: ██████ (3 people - Performance, Backtest optimization)
Week 17: ██████ (3 people - Multi-node architecture)
Week 18: ██████ (3 people - Multi-node, State replication)
Week 19: ██████ (3 people - Load balancing)
Week 20: ████████ (4 people - Multi-tenant)
Week 21: ██████████ (5 people - Testing, Docs, Deployment)
Week 22: ██████████ (5 people - Testing, Docs, Deployment)
Week 23: ████████ (4 people - Benchmarking, Security)
Week 24: ██████ (3 people - Production release)
```

**Peak Resource Usage:** Week 11-12 (6 people)
**Average Resource Usage:** 4.3 people

---

## Quick Wins (Low-Hanging Fruit)

These tasks can be completed quickly to show progress:

### Week 3
✅ **GOAL-1-01-01** - Project Structure (24h)
  - Quick win: Get CI/CD green

### Week 4
✅ **GOAL-1-02-02** - Error Handling (20h)
  - Quick win: Unified error types

✅ **GOAL-1-03-01** - Configuration (16h)
  - Quick win: Config loading working

### Week 5
✅ **GOAL-1-13-01** - Basic HTTP API (40h)
  - Quick win: `/health` endpoint responding

---

## Bottleneck Analysis

### Potential Bottlenecks

1. **GOAL-2-09-01** - All Strategies (160h, Week 9-10)
   - **Impact:** Blocks portfolio, risk, backtesting
   - **Mitigation:** Parallelize across 2 quant developers
   - **Alternative:** Complete MVP with 3 strategies, defer others

2. **GOAL-3-16-01** - GPU Acceleration (96h, Week 13-14)
   - **Impact:** Blocks performance optimization
   - **Mitigation:** Have CPU fallback, make GPU optional
   - **Alternative:** Ship without GPU, add later

3. **GOAL-5-23-01** - Security Audit (56h, Week 23)
   - **Impact:** Blocks production release
   - **Mitigation:** Start security review in Phase 2
   - **Alternative:** Launch to internal users first

### Critical Resources

**Most in-demand personas:**
1. **Rust Developer** - Needed for 12+ tasks
2. **Backend Developer** - Needed for 8+ tasks
3. **Quant Developer** - Needed for 5 tasks (but high-hour tasks)

**Recommendation:** Hire 2 full-time Rust developers + rotating specialists

---

## Success Checkpoints by Week

| Week | Checkpoint | Validation |
|------|------------|------------|
| 2 | Architecture approved | Document reviewed by 3+ engineers |
| 4 | First API call succeeds | Can fetch account info from Alpaca |
| 6 | MVP strategy running | 1 strategy executing in paper trading |
| 8 | News sentiment working | Sentiment scores match Python ±5% |
| 10 | 5/8 strategies complete | Backtests match Python ±10% |
| 12 | Full feature parity | All Python features working |
| 14 | GPU acceleration working | 10x speedup on GPU vs CPU |
| 16 | 3x performance improvement | API < 50ms, backtests 10x faster |
| 18 | 3-node cluster working | Leader election in < 5 seconds |
| 20 | Multi-tenant operational | 10 test tenants isolated |
| 22 | 95% test coverage | All tests passing in CI |
| 24 | Production deployment live | 100% traffic on Rust system |

---

## Command-Line Cheat Sheet

### Daily Research
```bash
# Start research for a topic
./scripts/research.sh --day 1 --topic "async_runtime" --expert "Performance Engineer"

# View research results
cat docs/research/day1_async_runtime_analysis.md
```

### Build & Test
```bash
# Build entire workspace
cargo build --workspace --release

# Run all tests
cargo test --workspace --all-features

# Check code coverage
cargo tarpaulin --out Html --output-dir coverage/

# Run clippy (linter)
cargo clippy --workspace -- -D warnings

# Format code
cargo fmt --all
```

### Benchmarking
```bash
# Run all benchmarks
cargo bench --workspace

# Specific benchmark
cargo bench --bench trading_strategy_bench

# With flamegraph
cargo flamegraph --bench trading_strategy_bench
```

### Documentation
```bash
# Generate documentation
cargo doc --workspace --no-deps --open

# Check doc tests
cargo test --doc
```

### Database
```bash
# Run migrations
sqlx migrate run

# Revert last migration
sqlx migrate revert

# Create new migration
sqlx migrate add create_orders_table
```

### Docker
```bash
# Build image
docker build -t neural-trader:latest .

# Run container
docker run -p 8080:8080 --env-file .env neural-trader:latest

# Multi-stage build (optimized)
docker build -f Dockerfile.optimized -t neural-trader:slim .
```

### Deployment
```bash
# Deploy to staging
kubectl apply -f k8s/staging/

# Deploy to production (with approval)
kubectl apply -f k8s/production/

# Rollback
kubectl rollout undo deployment/neural-trader

# Check status
kubectl rollout status deployment/neural-trader
```

---

## Key Decisions Log

| Decision | Date | Rationale | Alternatives Considered |
|----------|------|-----------|------------------------|
| Async Runtime: **Tokio** | Week 1 | Best ecosystem, mature, most crates compatible | async-std, smol |
| Web Framework: **Axum** | Week 1 | Type-safe, fast, Tokio-native | Actix-web, Rocket |
| ORM: **SQLx** | Week 1 | Compile-time checked, async | Diesel, SeaORM |
| ML Framework: **tch-rs** | Week 1 | PyTorch compatibility, GPU support | tract, rust-bert |
| Serialization: **serde** | Week 1 | Industry standard, best support | - |
| Error Handling: **thiserror + anyhow** | Week 1 | Best practice pattern | - |

*(Populate as decisions are made)*

---

## Emergency Contacts & Escalation

| Issue Type | Contact | Response Time |
|------------|---------|---------------|
| **Technical Blocker** | Tech Lead | < 4 hours |
| **Security Vulnerability** | Security Lead | < 1 hour |
| **Production Incident** | On-Call Engineer | < 15 minutes |
| **Resource Conflict** | Project Manager | < 1 day |
| **Architecture Decision** | System Architect | < 2 days |

**Escalation Path:**
1. Team Member → Team Lead (same day)
2. Team Lead → Tech Lead (next day)
3. Tech Lead → Project Manager (within 2 days)
4. Project Manager → Executive Sponsor (within 1 week)

---

## Useful Links

- **Main Taskboard:** [RUST_PORT_GOAP_TASKBOARD.md](./RUST_PORT_GOAP_TASKBOARD.md)
- **Module Breakdown:** [RUST_PORT_MODULE_BREAKDOWN.md](./RUST_PORT_MODULE_BREAKDOWN.md)
- **Research Protocol:** [RUST_PORT_RESEARCH_PROTOCOL.md](./RUST_PORT_RESEARCH_PROTOCOL.md)
- **GitHub Project Board:** https://github.com/your-org/neural-trader-rust/projects/1
- **CI/CD Dashboard:** https://github.com/your-org/neural-trader-rust/actions
- **Documentation:** https://your-org.github.io/neural-trader-rust/
- **Monitoring:** https://grafana.your-org.com/d/neural-trader

---

## Glossary

| Term | Definition |
|------|------------|
| **GOAP** | Goal-Oriented Action Planning - AI planning methodology |
| **PoC** | Proof of Concept - minimal implementation to validate approach |
| **E2B** | Execute to Build - cloud sandbox platform |
| **VaR** | Value at Risk - risk management metric |
| **CVaR** | Conditional Value at Risk (Expected Shortfall) |
| **P&L** | Profit and Loss |
| **Slippage** | Difference between expected and actual trade execution price |
| **Sharpe Ratio** | Risk-adjusted return metric |
| **CUDA** | Compute Unified Device Architecture - NVIDIA GPU programming |
| **JWT** | JSON Web Token - authentication standard |
| **ORM** | Object-Relational Mapping - database abstraction |
| **tch-rs** | PyTorch bindings for Rust |
| **SQLx** | Async SQL toolkit for Rust |
| **Tokio** | Async runtime for Rust |
| **Axum** | Web framework for Rust |

---

**Document Version:** 1.0.0
**Last Updated:** 2025-11-12
**Print-Friendly:** Yes (remove this section before printing)

---

## PDF Export Instructions

```bash
# Convert to PDF using pandoc
pandoc RUST_PORT_QUICK_REFERENCE.md \
  -o RUST_PORT_QUICK_REFERENCE.pdf \
  --toc \
  --toc-depth=2 \
  -V geometry:margin=1in

# Or use markdown-pdf
markdown-pdf RUST_PORT_QUICK_REFERENCE.md
```
