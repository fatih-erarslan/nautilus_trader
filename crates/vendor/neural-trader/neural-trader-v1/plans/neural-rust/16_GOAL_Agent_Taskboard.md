# GOAP Agent Taskboard

## Document Purpose

This document defines **Goal-Oriented Action Planning (GOAP) tasks** for the Neural Rust port. Each task has objectives, preconditions, effects, cost estimates, risk levels, and checkpoints for autonomous agent coordination.

## Table of Contents

1. [GOAP Overview](#goap-overview)
2. [Task Templates](#task-templates)
3. [Planning Document Tasks](#planning-document-tasks)
4. [Implementation Tasks](#implementation-tasks)
5. [Daily Research Cadence](#daily-research-cadence)
6. [Task Chains](#task-chains)
7. [State Transitions](#state-transitions)
8. [Monitoring](#monitoring)

---

## GOAP Overview

### What is GOAP?

**Goal-Oriented Action Planning** is an AI planning algorithm that:
- Defines **world state** (current conditions)
- Specifies **goals** (desired outcomes)
- Finds **action sequences** that transform state → goal
- Optimizes for **lowest cost** path

### GOAP in Neural Rust Port

Each task is a GOAP action with:

```rust
struct GoalTask {
    id: String,
    objective: String,           // What to achieve
    preconditions: Vec<String>,  // What must be true before
    effects: Vec<String>,        // What becomes true after
    cost_estimate: u32,          // Hours or story points
    risk_level: RiskLevel,       // Low | Medium | High
    owner_persona: Persona,      // researcher | coder | tester | etc.
    checkpoints: Vec<Checkpoint>, // Validation steps
    dependencies: Vec<String>,   // Other task IDs
}
```

### Task States

```
[Pending] → [Ready] → [In Progress] → [Complete]
    ↑                       ↓
    └───────[Blocked]←──────┘
```

---

## Task Templates

### Template: Research Task

```yaml
id: RESEARCH-001
objective: "Research {topic} and document findings"
preconditions:
  - Access to research resources
  - Clear research questions defined
effects:
  - "{topic} architecture documented"
  - "Technology choices validated"
  - "Risks identified and mitigated"
cost_estimate: 8 hours
risk_level: Low
owner_persona: researcher
checkpoints:
  - name: "Research questions answered"
    validation: "Document includes answers to all questions"
  - name: "Alternatives compared"
    validation: "Decision matrix with ≥3 options"
  - name: "Peer review complete"
    validation: "At least 2 reviewers approved"
dependencies: []
```

### Template: Implementation Task

```yaml
id: IMPL-001
objective: "Implement {feature} with tests"
preconditions:
  - "{feature} design approved"
  - "Development environment ready"
  - "Dependencies available"
effects:
  - "{feature} implemented in Rust"
  - "Unit tests pass (≥90% coverage)"
  - "Integration tests pass"
  - "Documentation updated"
cost_estimate: 16 hours
risk_level: Medium
owner_persona: coder
checkpoints:
  - name: "Tests written first (TDD)"
    validation: "Tests exist before implementation"
  - name: "Implementation complete"
    validation: "cargo test --all passes"
  - name: "Code reviewed"
    validation: "PR approved by ≥1 reviewer"
  - name: "Benchmarks passing"
    validation: "Performance within 5% of target"
dependencies: ["DESIGN-001"]
```

### Template: Testing Task

```yaml
id: TEST-001
objective: "Create {test_type} tests for {feature}"
preconditions:
  - "{feature} implemented"
  - "Test data available"
effects:
  - "{test_type} test suite complete"
  - "Coverage ≥90%"
  - "All tests passing"
cost_estimate: 8 hours
risk_level: Low
owner_persona: tester
checkpoints:
  - name: "Test cases defined"
    validation: "Test plan with ≥5 scenarios"
  - name: "Tests implemented"
    validation: "All test cases automated"
  - name: "Coverage verified"
    validation: "cargo tarpaulin shows ≥90%"
dependencies: ["IMPL-001"]
```

---

## Planning Document Tasks

### Task Group: Phase 0 Documentation

#### T-001: Python Architecture Analysis

```yaml
id: T-001
objective: "Analyze Python codebase and document architecture"
preconditions:
  - Access to Python repository
  - Access to live demo
owner_persona: researcher
cost_estimate: 16 hours
risk_level: Low

effects:
  - "Python architecture documented"
  - "All 58+ MCP tools enumerated"
  - "8 trading strategies cataloged"
  - "Data flows mapped"

checkpoints:
  - name: "Feature inventory complete"
    validation: "Spreadsheet with all features, 100% complete"
  - name: "API surface documented"
    validation: "All MCP tools with signatures and descriptions"
  - name: "Dependencies mapped"
    validation: "Dependency graph with Alpaca, Polygon, Yahoo, NewsAPI"

dependencies: []
```

#### T-002: Parity Requirements Matrix

```yaml
id: T-002
objective: "Define feature parity requirements and priorities"
preconditions:
  - "Python architecture documented (T-001)"
  - "Stakeholder availability"
owner_persona: code-goal-planner
cost_estimate: 12 hours
risk_level: Low

effects:
  - "02_Parity_Requirements.md complete"
  - "Features prioritized (P0/P1/P2)"
  - "Acceptance criteria defined"
  - "Stakeholder sign-off obtained"

checkpoints:
  - name: "P0 features identified"
    validation: "≥15 must-have features listed"
  - name: "Acceptance tests defined"
    validation: "Each feature has ≥1 acceptance test"
  - name: "Stakeholder approval"
    validation: "Sign-off email or meeting notes"

dependencies: ["T-001"]
```

#### T-003: Architecture Design

```yaml
id: T-003
objective: "Design Rust system architecture and module boundaries"
preconditions:
  - "Parity requirements approved (T-002)"
  - "Architecture team available"
owner_persona: system-architect
cost_estimate: 24 hours
risk_level: Medium

effects:
  - "03_Architecture.md complete"
  - "15 crate modules defined"
  - "Trait boundaries specified"
  - "Data flows documented"

checkpoints:
  - name: "Module structure defined"
    validation: "Crate dependency graph with no cycles"
  - name: "Interfaces designed"
    validation: "All public traits defined"
  - name: "Technology stack chosen"
    validation: "ADRs for tokio, polars, napi-rs, etc."
  - name: "Architecture review passed"
    validation: "Approved by ≥2 senior engineers"

dependencies: ["T-002"]
```

#### T-004: Rust Crates Research

```yaml
id: T-004
objective: "Research and validate Rust crates for Node.js interop"
preconditions:
  - "Architecture design complete (T-003)"
owner_persona: backend-dev
cost_estimate: 16 hours
risk_level: Medium

effects:
  - "04_Rust_Crates_and_Node_Interop.md complete"
  - "napi-rs validated for async"
  - "FFI vs IPC trade-offs documented"
  - "Prototype working"

checkpoints:
  - name: "napi-rs prototype working"
    validation: "Async function called from Node.js"
  - name: "Memory safety validated"
    validation: "No segfaults in 1000 test runs"
  - name: "Performance measured"
    validation: "FFI overhead <1ms"

dependencies: ["T-003"]
```

#### T-005: AgentDB Memory Design

```yaml
id: T-005
objective: "Design AgentDB integration for persistent memory"
preconditions:
  - "Architecture design complete (T-003)"
owner_persona: ml-developer
cost_estimate: 16 hours
risk_level: Low

effects:
  - "05_Memory_and_AgentDB.md complete"
  - "Vector storage schema designed"
  - "Memory patterns defined"
  - "API integration planned"

checkpoints:
  - name: "Schema designed"
    validation: "Vector dimensions and metadata defined"
  - name: "Integration patterns documented"
    validation: "Code examples for store/retrieve"
  - name: "Performance estimated"
    validation: "Latency <10ms for retrieval"

dependencies: ["T-003"]
```

#### T-006: Strategy Algorithms Design

```yaml
id: T-006
objective: "Design trading strategy algorithms in pseudocode"
preconditions:
  - "Parity requirements approved (T-002)"
owner_persona: researcher
cost_estimate: 20 hours
risk_level: Medium

effects:
  - "06_Strategy_and_Sublinear_Solvers.md complete"
  - "8 strategies in pseudocode"
  - "Complexity analysis done (Big O)"
  - "Sublinear solver integration planned"

checkpoints:
  - name: "All strategies in pseudocode"
    validation: "8 algorithms with clear logic"
  - name: "Complexity analyzed"
    validation: "Big O notation for each"
  - name: "Edge cases documented"
    validation: "≥5 edge cases per strategy"

dependencies: ["T-002"]
```

#### T-007: Streaming Architecture Design

```yaml
id: T-007
objective: "Design event streaming architecture with midstreamer"
preconditions:
  - "Architecture design complete (T-003)"
owner_persona: backend-dev
cost_estimate: 16 hours
risk_level: Medium

effects:
  - "07_Streaming_and_Midstreamer.md complete"
  - "Event flow patterns defined"
  - "Backpressure handling designed"
  - "Midstreamer integration planned"

checkpoints:
  - name: "Event flow diagrams complete"
    validation: "ASCII diagrams for all flows"
  - name: "Backpressure strategy defined"
    validation: "Drop, buffer, or block policies"
  - name: "Integration tested"
    validation: "Prototype with midstreamer"

dependencies: ["T-003"]
```

#### T-008: Security & Governance Design

```yaml
id: T-008
objective: "Design security architecture with AIDefence and Lean"
preconditions:
  - "Architecture design complete (T-003)"
owner_persona: security-manager
cost_estimate: 16 hours
risk_level: High

effects:
  - "08_Security_Governance_AIDefence_Lean.md complete"
  - "Threat model documented"
  - "AIDefence guardrails designed"
  - "Lean formal verification planned"

checkpoints:
  - name: "Threat model complete"
    validation: "STRIDE analysis for all components"
  - name: "Security controls defined"
    validation: "Input validation, output sanitization"
  - name: "Formal proofs identified"
    validation: "≥3 critical invariants to prove"

dependencies: ["T-003"]
```

#### T-009: E2B Sandboxes & SBOM

```yaml
id: T-009
objective: "Design E2B sandbox architecture and supply chain security"
preconditions:
  - "Architecture design complete (T-003)"
owner_persona: cicd-engineer
cost_estimate: 12 hours
risk_level: Low

effects:
  - "09_E2B_Sandboxes_and_Supply_Chain.md complete"
  - "Sandbox templates defined"
  - "SBOM generation plan"
  - "Dependency auditing automated"

checkpoints:
  - name: "E2B templates designed"
    validation: "Template.yaml for strategy execution"
  - name: "SBOM tooling chosen"
    validation: "cargo-sbom or syft integrated"
  - name: "Audit automation planned"
    validation: "CI job for cargo-audit daily"

dependencies: ["T-003"]
```

#### T-010: Federation & Payments Design

```yaml
id: T-010
objective: "Design multi-strategy federation with agentic payments"
preconditions:
  - "Architecture design complete (T-003)"
owner_persona: system-architect
cost_estimate: 16 hours
risk_level: Medium

effects:
  - "10_Federations_and_Agentic_Payments.md complete"
  - "Federation topology defined"
  - "Payment tracking designed"
  - "Cost allocation algorithm"

checkpoints:
  - name: "Topology designed"
    validation: "Mesh, hierarchical, or star topology chosen"
  - name: "Payment API defined"
    validation: "Cost tracking per strategy"
  - name: "Budget limits designed"
    validation: "Hard and soft limits with alerts"

dependencies: ["T-003"]
```

#### T-011: CLI & NPM Release Plan

```yaml
id: T-011
objective: "Design CLI interface and NPM packaging strategy"
preconditions:
  - "Architecture design complete (T-003)"
owner_persona: cicd-engineer
cost_estimate: 12 hours
risk_level: Low

effects:
  - "11_CLI_and_NPM_Release.md complete"
  - "CLI commands defined"
  - "Cross-platform packaging planned"
  - "Release automation designed"

checkpoints:
  - name: "CLI commands specified"
    validation: "≥8 commands with help text"
  - name: "Platform packages defined"
    validation: "linux-x64, darwin-x64, darwin-arm64, win32-x64"
  - name: "Release workflow designed"
    validation: "GitHub Actions for publish"

dependencies: ["T-003"]
```

#### T-012: Secrets Management Plan

```yaml
id: T-012
objective: "Design secrets management and environment configuration"
preconditions:
  - "Architecture design complete (T-003)"
owner_persona: security-manager
cost_estimate: 8 hours
risk_level: High

effects:
  - "12_Secrets_and_Environments.md complete"
  - "Secret storage strategy defined"
  - "Environment config designed"
  - "Rotation procedures documented"

checkpoints:
  - name: "Secret storage chosen"
    validation: "Environment vars, vault, or keychain"
  - name: "Rotation procedure defined"
    validation: "Step-by-step runbook"
  - name: "Zero hardcoded secrets"
    validation: "Grep for 'sk_', 'pk_' returns nothing"

dependencies: ["T-003"]
```

#### T-013: Tests, Benchmarks, CI

```yaml
id: T-013
objective: "Design comprehensive testing and CI/CD strategy"
preconditions:
  - "Architecture design complete (T-003)"
owner_persona: tester
cost_estimate: 16 hours
risk_level: Low

effects:
  - "13_Tests_Benchmarks_CI.md complete"
  - "Test hierarchy defined (unit, integration, E2E)"
  - "Benchmark targets specified"
  - "CI matrix configured"

checkpoints:
  - name: "Test pyramid defined"
    validation: "70% unit, 15% integration, 10% property, 5% other"
  - name: "Performance targets set"
    validation: "p50/p95/p99 latency, throughput"
  - name: "CI matrix defined"
    validation: "Linux/macOS/Windows × Node 18/20/22"

dependencies: ["T-003"]
```

#### T-014: Risk Register

```yaml
id: T-014
objective: "Create risk register with mitigation strategies"
preconditions:
  - "Architecture design complete (T-003)"
owner_persona: researcher
cost_estimate: 12 hours
risk_level: Low

effects:
  - "14_Risk_and_Fallbacks.md complete"
  - "≥10 risks documented"
  - "Mitigation strategies defined"
  - "Fallback procedures documented"

checkpoints:
  - name: "Risks identified"
    validation: "≥10 risks with probability and impact"
  - name: "Mitigations planned"
    validation: "Each risk has ≥1 mitigation"
  - name: "Rollback procedures"
    validation: "Step-by-step for each critical system"

dependencies: ["T-003"]
```

#### T-015: Roadmap & Milestones

```yaml
id: T-015
objective: "Create detailed 24-week project roadmap"
preconditions:
  - "Parity requirements approved (T-002)"
  - "Architecture design complete (T-003)"
owner_persona: code-goal-planner
cost_estimate: 16 hours
risk_level: Low

effects:
  - "15_Roadmap_Phases_and_Milestones.md complete"
  - "6 phases defined with milestones"
  - "Gantt chart created"
  - "Resource allocation planned"

checkpoints:
  - name: "All phases defined"
    validation: "Phase 0-5 with entry/exit criteria"
  - name: "Milestones set"
    validation: "≥15 milestones with dates"
  - name: "Resource plan"
    validation: "FTE allocation per phase"

dependencies: ["T-002", "T-003"]
```

#### T-016: GOAP Taskboard

```yaml
id: T-016
objective: "Create GOAP agent taskboard with all implementation tasks"
preconditions:
  - "Roadmap complete (T-015)"
owner_persona: code-goal-planner
cost_estimate: 20 hours
risk_level: Low

effects:
  - "16_GOAL_Agent_Taskboard.md complete"
  - "All implementation tasks defined"
  - "Task dependencies mapped"
  - "Daily research cadence defined"

checkpoints:
  - name: "All tasks defined"
    validation: "≥50 tasks with GOAP structure"
  - name: "Dependencies mapped"
    validation: "Dependency graph with critical path"
  - name: "Agent personas assigned"
    validation: "Each task has owner persona"

dependencies: ["T-015"]
```

#### T-017: Exchange Adapters Design

```yaml
id: T-017
objective: "Design exchange adapter architecture and data pipeline"
preconditions:
  - "Architecture design complete (T-003)"
owner_persona: backend-dev
cost_estimate: 16 hours
risk_level: Medium

effects:
  - "17_Exchange_Adapters_and_Data_Pipeline.md complete"
  - "Abstract adapter trait defined"
  - "Alpaca adapter designed"
  - "Multi-exchange aggregation planned"

checkpoints:
  - name: "Adapter trait defined"
    validation: "Rust trait with all methods"
  - name: "Alpaca integration designed"
    validation: "REST and WebSocket patterns"
  - name: "Failover strategy"
    validation: "Primary → Polygon → Yahoo fallback"

dependencies: ["T-003"]
```

#### T-018: Backtesting Engine Design

```yaml
id: T-018
objective: "Design deterministic backtesting engine"
preconditions:
  - "Strategy algorithms designed (T-006)"
owner_persona: researcher
cost_estimate: 16 hours
risk_level: Medium

effects:
  - "18_Simulation_Backtesting.md complete"
  - "Event-sourced backtest designed"
  - "Slippage/fee models defined"
  - "Statistical validation planned"

checkpoints:
  - name: "Backtest engine designed"
    validation: "Deterministic with seeded RNG"
  - name: "Market tape format defined"
    validation: "Event-sourced CSV/Parquet format"
  - name: "Metrics defined"
    validation: "Sharpe, Sortino, max drawdown, etc."

dependencies: ["T-006"]
```

---

## Implementation Tasks

### Task Group: Phase 1 MVP Core

#### T-101: Core Types Implementation

```yaml
id: T-101
objective: "Implement core Rust types and traits"
preconditions:
  - "Architecture design complete (T-003)"
  - "Rust toolchain installed"
owner_persona: coder
cost_estimate: 12 hours
risk_level: Low

effects:
  - "crates/core/ implemented"
  - "MarketTick, Signal, Order types defined"
  - "Strategy, BrokerClient traits defined"
  - "Unit tests pass (95% coverage)"

checkpoints:
  - name: "Types defined"
    validation: "cargo build succeeds"
  - name: "Tests written"
    validation: "≥20 unit tests"
  - name: "Docs complete"
    validation: "cargo doc --open shows all types"

dependencies: ["T-003"]
```

#### T-102: Market Data Ingestion

```yaml
id: T-102
objective: "Implement Alpaca WebSocket market data ingestion"
preconditions:
  - "Core types implemented (T-101)"
  - "Alpaca paper trading account"
owner_persona: coder
cost_estimate: 16 hours
risk_level: Medium

effects:
  - "crates/market-data/ implemented"
  - "WebSocket client connects to Alpaca"
  - "Market ticks parsed and buffered"
  - "Integration test passes"

checkpoints:
  - name: "WebSocket connection works"
    validation: "Receives ticks from Alpaca sandbox"
  - name: "Parsing correct"
    validation: "All tick fields populated"
  - name: "Performance target met"
    validation: "<100μs per tick (benchmark)"

dependencies: ["T-101"]
```

#### T-103: napi-rs Bindings

```yaml
id: T-103
objective: "Implement Node.js FFI bindings with napi-rs"
preconditions:
  - "Core types implemented (T-101)"
  - "Rust crates research complete (T-004)"
owner_persona: coder
cost_estimate: 20 hours
risk_level: High

effects:
  - "crates/napi-bindings/ implemented"
  - "Async Rust functions callable from Node.js"
  - "TypeScript types generated"
  - "Memory safe (no leaks in 1000 runs)"

checkpoints:
  - name: "Async bridge working"
    validation: "Promise resolves from Rust Future"
  - name: "Type safety"
    validation: "TypeScript types match Rust"
  - name: "No memory leaks"
    validation: "Valgrind clean after 1000 calls"

dependencies: ["T-101", "T-004"]
```

#### T-104: Momentum Strategy

```yaml
id: T-104
objective: "Implement Momentum trading strategy"
preconditions:
  - "Core types implemented (T-101)"
  - "Strategy algorithms designed (T-006)"
owner_persona: coder
cost_estimate: 16 hours
risk_level: Medium

effects:
  - "crates/strategies/momentum.rs implemented"
  - "Z-score momentum calculation"
  - "Long/short signal generation"
  - "Parity test vs Python passes"

checkpoints:
  - name: "Algorithm implemented"
    validation: "Unit tests pass"
  - name: "Parity validated"
    validation: "Results match Python within 1e-6"
  - name: "Performance target met"
    validation: "<5ms per signal (benchmark)"

dependencies: ["T-101", "T-006"]
```

#### T-105: Order Execution

```yaml
id: T-105
objective: "Implement Alpaca order execution"
preconditions:
  - "Core types implemented (T-101)"
  - "Exchange adapters designed (T-017)"
owner_persona: coder
cost_estimate: 16 hours
risk_level: Medium

effects:
  - "crates/execution/ implemented"
  - "Alpaca REST client"
  - "Market order placement"
  - "Order status tracking"

checkpoints:
  - name: "Orders placed successfully"
    validation: "Paper trading order confirmed"
  - name: "Error handling"
    validation: "Retries on 429, circuit breaker on 500"
  - name: "Performance target met"
    validation: "<10ms API call (benchmark)"

dependencies: ["T-101", "T-017"]
```

#### T-106: Risk Management

```yaml
id: T-106
objective: "Implement Kelly criterion risk management"
preconditions:
  - "Core types implemented (T-101)"
owner_persona: coder
cost_estimate: 12 hours
risk_level: Low

effects:
  - "crates/risk/ implemented"
  - "Kelly fraction calculation"
  - "Position size validation"
  - "Portfolio constraints enforced"

checkpoints:
  - name: "Position sizing works"
    validation: "Property tests pass"
  - name: "Constraints enforced"
    validation: "Max position, cash buffer respected"
  - name: "Performance target met"
    validation: "<500μs per check (benchmark)"

dependencies: ["T-101"]
```

#### T-107: MVP Integration

```yaml
id: T-107
objective: "Integrate all MVP components end-to-end"
preconditions:
  - "All MVP components implemented (T-101 to T-106)"
owner_persona: coder
cost_estimate: 16 hours
risk_level: Medium

effects:
  - "crates/neural-trader/ main binary"
  - "Event loop implemented"
  - "Graceful shutdown"
  - "E2E test passes"

checkpoints:
  - name: "End-to-end flow works"
    validation: "Market data → signal → order"
  - name: "Graceful shutdown"
    validation: "Ctrl+C closes cleanly"
  - name: "Observability added"
    validation: "Tracing logs all events"

dependencies: ["T-101", "T-102", "T-103", "T-104", "T-105", "T-106"]
```

---

## Daily Research Cadence

### E2B Agent Research Pattern

**Frequency:** Daily (weekdays)
**Duration:** 2 hours per day
**Platform:** E2B sandboxes with OpenRouter/Kimi

**Daily Workflow:**

```yaml
daily_research:
  morning:
    - name: "Identify knowledge gaps"
      duration: 15 min
      output: "Research questions list"

    - name: "Spawn E2B agent"
      duration: 5 min
      command: |
        e2b sandbox create --template rust-research
        e2b sandbox run research-agent --task "Research {topic}"

    - name: "Agent research execution"
      duration: 60 min
      activities:
        - "Search documentation (docs.rs, crates.io)"
        - "Analyze example code"
        - "Run benchmarks"
        - "Compare alternatives"

  afternoon:
    - name: "Review agent findings"
      duration: 20 min
      output: "Findings summary markdown"

    - name: "Make decisions"
      duration: 15 min
      output: "ADR (Architecture Decision Record)"

    - name: "Update planning docs"
      duration: 20 min
      output: "Commit to planning docs"

  evening:
    - name: "Prep tomorrow's research"
      duration: 5 min
      output: "Tomorrow's research agenda"
```

### Research Topics by Phase

**Phase 0 (Weeks 1-2):**
- Day 1: Rust async runtime comparison (tokio vs async-std)
- Day 2: DataFrame libraries (polars vs datafusion)
- Day 3: napi-rs async patterns
- Day 4: WebSocket libraries (tungstenite vs tokio-tungstenite)
- Day 5: HTTP client comparison (reqwest vs hyper)
- Day 6: Testing frameworks (built-in vs proptest)
- Day 7: Benchmarking tools (criterion vs built-in)
- Day 8: Observability (tracing vs log)
- Day 9: Database options (sqlx vs diesel)
- Day 10: Error handling patterns (anyhow vs thiserror)

**Phase 1 (Weeks 3-6):**
- Market data parsing optimization
- Async channel performance
- Memory allocation profiling
- FFI boundary optimization

**Phase 2 (Weeks 7-12):**
- Neural network inference (candle vs tract)
- ONNX runtime integration
- GPU acceleration patterns
- AgentDB integration patterns

---

## Task Chains

### Critical Path (24 weeks)

```
T-001 (Python Analysis)
  ↓
T-002 (Parity Requirements)
  ↓
T-003 (Architecture Design)
  ↓
T-101 (Core Types)
  ↓
T-104 (Momentum Strategy)
  ↓
T-107 (MVP Integration)
  ↓
[Phase 2 Implementation: T-201 to T-208]
  ↓
[Phase 3 Optimization: T-301 to T-304]
  ↓
[Phase 4 Federation: T-401 to T-404]
  ↓
[Phase 5 Release: T-501 to T-505]
```

### Parallel Work Streams

**Stream 1: Core Infrastructure (Weeks 3-6)**
```
T-101 (Core Types) → T-102 (Market Data) → T-107 (Integration)
```

**Stream 2: Strategies (Weeks 3-12)**
```
T-104 (Momentum) → T-201 (Mean Reversion) → T-202 (Mirror) → ...
```

**Stream 3: Execution (Weeks 5-14)**
```
T-105 (Execution) → T-106 (Risk) → T-205 (Portfolio Tracking)
```

**Stream 4: Node Interop (Weeks 4-18)**
```
T-103 (napi-rs) → T-206 (Type Definitions) → T-504 (NPM Package)
```

---

## State Transitions

### Task State Machine

```rust
enum TaskState {
    Pending,       // Not yet started
    Ready,         // Preconditions met, can start
    InProgress,    // Currently being worked on
    Blocked,       // Waiting on dependency or blocker
    Complete,      // All checkpoints validated
}

impl Task {
    fn can_start(&self, world_state: &WorldState) -> bool {
        self.preconditions.iter().all(|cond| world_state.is_true(cond))
    }

    fn effects_apply(&self, world_state: &mut WorldState) {
        for effect in &self.effects {
            world_state.set_true(effect);
        }
    }
}
```

### World State Example

```rust
let mut world_state = WorldState::new();

// Initial state
world_state.set_true("Python architecture documented");
world_state.set_true("Parity requirements approved");

// Task T-003 can start because preconditions met
assert!(task_t003.can_start(&world_state));

// Complete task T-003
task_t003.execute()?;
task_t003.effects_apply(&mut world_state);

// Now these are true:
assert!(world_state.is_true("03_Architecture.md complete"));
assert!(world_state.is_true("15 crate modules defined"));
```

---

## Monitoring

### Task Progress Tracking

```rust
struct TaskProgress {
    total_tasks: usize,
    completed: usize,
    in_progress: usize,
    blocked: usize,
    pending: usize,
}

impl TaskProgress {
    fn completion_percentage(&self) -> f64 {
        (self.completed as f64 / self.total_tasks as f64) * 100.0
    }

    fn estimated_remaining_hours(&self) -> u32 {
        // Sum cost_estimate for incomplete tasks
        self.pending_tasks.iter()
            .map(|t| t.cost_estimate)
            .sum()
    }
}
```

### Daily Standup Report

```markdown
# Daily Standup - 2025-11-12

## Completed Today
- [T-101] Core Types Implementation ✅
- [T-102] Market Data Ingestion ✅

## In Progress
- [T-103] napi-rs Bindings (60% complete)
- [T-104] Momentum Strategy (30% complete)

## Blocked
- [T-105] Order Execution (waiting on Alpaca API keys)

## Starting Tomorrow
- [T-106] Risk Management
- [T-104] Momentum Strategy (continued)

## Metrics
- Total Tasks: 50
- Completed: 12 (24%)
- In Progress: 4 (8%)
- Blocked: 1 (2%)
- Pending: 33 (66%)
- Estimated Remaining: 380 hours (9.5 weeks at 40 hrs/week)
```

### Burndown Chart (ASCII)

```
Tasks
 50 │●
    │ ●
 40 │  ●
    │   ●
 30 │    ●●
    │      ●●
 20 │        ●●●
    │           ●●●
 10 │              ●●●●
    │                  ●●●●
  0 │______________________●
    Week 1  5  10 15 20 24

Ideal: ● (linear)
Actual: (TBD)
```

---

## Acceptance Criteria

### GOAP Task Completeness

- [ ] All 18 planning documents have GOAP tasks
- [ ] All implementation phases have GOAP tasks
- [ ] Every task has preconditions, effects, cost, risk
- [ ] Dependencies mapped for critical path
- [ ] Daily research cadence defined
- [ ] Task state machine implemented
- [ ] Monitoring dashboard operational

### Task Validation

Each task must:
- [ ] Have clear, measurable objective
- [ ] List all preconditions (dependencies)
- [ ] Define observable effects (outcomes)
- [ ] Include cost estimate (hours or story points)
- [ ] Assign risk level (Low/Medium/High)
- [ ] Specify owner persona
- [ ] Provide ≥3 validation checkpoints

---

## Cross-References

- **Roadmap:** [15_Roadmap_Phases_and_Milestones.md](./15_Roadmap_Phases_and_Milestones.md) - Phase timeline
- **SPARC:** [01_SPARC_Plan.md](./01_SPARC_Plan.md) - Development methodology
- **Testing:** [13_Tests_Benchmarks_CI.md](./13_Tests_Benchmarks_CI.md) - Validation criteria
- **Architecture:** [03_Architecture.md](./03_Architecture.md) - Technical design

---

**Document Version:** 1.0.0
**Last Updated:** 2025-11-12
**Owner:** Planning Agent
**Status:** Complete
**Next Review:** Weekly during execution
