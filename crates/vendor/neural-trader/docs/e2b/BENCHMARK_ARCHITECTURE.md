# E2B Swarm Benchmark Architecture

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Benchmark Test Suite                        │
│                  (swarm-benchmarks.test.js)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Executes
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  TradingSwarmBenchmark Class                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Swarm Mgmt  │  │  Metrics     │  │  Report Gen  │         │
│  │              │  │  Collection  │  │              │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Creates/Manages
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        E2B Sandboxes                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Agent 1  │  │ Agent 2  │  │ Agent 3  │  │ Agent N  │       │
│  │          │  │          │  │          │  │          │       │
│  │ Python   │  │ Python   │  │ Python   │  │ Python   │       │
│  │ Trading  │  │ Trading  │  │ Trading  │  │ Trading  │       │
│  │ Logic    │  │ Logic    │  │ Logic    │  │ Logic    │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Generates
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Benchmark Report (MD)                        │
│  • Executive Summary    • Resource Analysis                     │
│  • Performance Metrics  • Cost Analysis                         │
│  • Scalability Data     • Recommendations                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Benchmark Flow Diagram

```
START
  │
  ├─► Setup Phase
  │   ├─ Initialize TradingSwarmBenchmark
  │   ├─ Set performance targets
  │   ├─ Initialize metrics collection
  │   └─ Verify E2B connection
  │
  ├─► Creation Performance Tests
  │   ├─ Test 1: Swarm Initialization
  │   │   ├─ Create swarm (5 runs)
  │   │   ├─ Measure time
  │   │   ├─ Calculate statistics
  │   │   └─ Validate against target (<5s)
  │   │
  │   ├─ Test 2: Agent Deployment
  │   │   ├─ Deploy agents
  │   │   ├─ Measure per-agent latency
  │   │   └─ Validate against target (<3s)
  │   │
  │   └─ Test 3: Parallel vs Sequential
  │       ├─ Deploy in parallel (mesh)
  │       ├─ Deploy sequentially (hierarchical)
  │       ├─ Calculate speedup
  │       └─ Validate speedup (>1.5x)
  │
  ├─► Scalability Tests
  │   ├─ Test 1: Agent Count Scaling
  │   │   ├─ Test 1 agent
  │   │   ├─ Test 5 agents
  │   │   ├─ Test 10 agents
  │   │   └─ Analyze scaling pattern
  │   │
  │   ├─ Test 2: Linear Scaling (2-20)
  │   │   ├─ Test 2, 4, 8, 12, 16, 20 agents
  │   │   ├─ Measure resources
  │   │   └─ Validate sub-linear growth
  │   │
  │   └─ Test 3: Topology Comparison
  │       ├─ Test mesh topology
  │       ├─ Test hierarchical topology
  │       ├─ Test ring topology
  │       └─ Compare performance
  │
  ├─► Trading Operations Tests
  │   ├─ Test 1: Strategy Execution
  │   │   ├─ Execute 50 strategies
  │   │   ├─ Measure throughput
  │   │   └─ Validate latency (<100ms)
  │   │
  │   ├─ Test 2: Task Distribution
  │   │   ├─ Distribute 100 tasks
  │   │   ├─ Measure efficiency
  │   │   └─ Calculate throughput
  │   │
  │   └─ Test 3: Consensus Latency
  │       ├─ Run 20 consensus rounds
  │       ├─ Measure decision time
  │       └─ Validate speed (<200ms)
  │
  ├─► Communication Tests
  │   ├─ Test 1: Inter-Agent Latency
  │   │   ├─ Send 10 messages
  │   │   ├─ Measure round-trip time
  │   │   └─ Validate latency (<50ms)
  │   │
  │   ├─ Test 2: State Synchronization
  │   │   ├─ Sync 10KB state
  │   │   ├─ Measure overhead
  │   │   └─ Validate performance
  │   │
  │   └─ Test 3: Message Throughput
  │       ├─ Pass 100 messages
  │       ├─ Calculate msg/sec
  │       └─ Validate throughput (>50)
  │
  ├─► Resource Usage Tests
  │   ├─ Test 1: Memory Usage
  │   │   ├─ Measure per agent
  │   │   ├─ Test 1, 5, 10 agents
  │   │   └─ Validate efficiency
  │   │
  │   ├─ Test 2: CPU Utilization
  │   │   ├─ Measure per topology
  │   │   ├─ Monitor during work
  │   │   └─ Compare efficiency
  │   │
  │   └─ Test 3: Network Bandwidth
  │       ├─ Track bytes transferred
  │       ├─ Count messages
  │       └─ Calculate avg/operation
  │
  ├─► Cost Analysis Tests
  │   ├─ Test 1: Cost per Operation
  │   │   ├─ Run 100 operations
  │   │   ├─ Calculate total cost
  │   │   └─ Compute cost/op
  │   │
  │   ├─ Test 2: Topology Cost Compare
  │   │   ├─ Test each topology
  │   │   ├─ Run for 1 minute
  │   │   └─ Compare costs
  │   │
  │   └─ Test 3: Scaling Cost Efficiency
  │       ├─ Test 2, 5, 10, 15 agents
  │       ├─ Calculate cost/operation
  │       └─ Validate sub-linear cost
  │
  ├─► Report Generation
  │   ├─ Aggregate all results
  │   ├─ Calculate statistics
  │   ├─ Generate charts
  │   ├─ Create recommendations
  │   └─ Write SWARM_BENCHMARKS_REPORT.md
  │
  └─► Cleanup
      ├─ Destroy all sandboxes
      ├─ Save metrics
      └─ Display summary
        │
        END
```

---

## Component Architecture

### 1. TradingSwarmBenchmark Class

```
TradingSwarmBenchmark
│
├─ Properties
│  ├─ sandboxes: Map<swarmId, agents[]>
│  ├─ agents: Map<agentKey, agent>
│  └─ metrics: { messages, decisions, strategies, bytes }
│
├─ Swarm Management
│  ├─ createSwarm(topology, agentCount)
│  ├─ deployAgent(swarmId, agentId, topology)
│  ├─ establishConnections(agents, topology)
│  └─ cleanup(swarmId)
│
├─ Performance Testing
│  ├─ executeStrategy(swarmId, marketData)
│  ├─ measureInterAgentLatency(swarmId)
│  ├─ reachConsensus(swarmId, proposals)
│  └─ measureResourceUsage(swarmId)
│
└─ Metrics Collection
   ├─ getMetrics()
   └─ updateMetrics(type, value)
```

### 2. Agent Architecture (Python in E2B)

```
TradingAgent (Python)
│
├─ Properties
│  ├─ agent_id: int
│  ├─ topology: str
│  ├─ portfolio: dict
│  ├─ strategy: str
│  ├─ messages: list
│  └─ decisions: list
│
├─ Trading Methods
│  ├─ execute_strategy(market_data)
│  ├─ send_message(target, message)
│  ├─ reach_consensus(proposals)
│  └─ calculate_portfolio_value()
│
└─ Metrics
   └─ get_metrics()
```

### 3. Benchmark Categories

```
Benchmark Categories
│
├─ Creation (3 tests)
│  ├─ Swarm init time
│  ├─ Agent deployment
│  └─ Parallel efficiency
│
├─ Scalability (3 tests)
│  ├─ Agent count scaling
│  ├─ Linear scaling test
│  └─ Topology comparison
│
├─ Trading (3 tests)
│  ├─ Strategy throughput
│  ├─ Task distribution
│  └─ Consensus latency
│
├─ Communication (3 tests)
│  ├─ Inter-agent latency
│  ├─ State sync overhead
│  └─ Message throughput
│
├─ Resources (3 tests)
│  ├─ Memory usage
│  ├─ CPU utilization
│  └─ Network bandwidth
│
└─ Costs (3 tests)
   ├─ Cost per operation
   ├─ Topology costs
   └─ Scaling costs
```

---

## Data Flow

```
Input (Market Data)
        │
        ▼
┌─────────────────┐
│  Test Executor  │
└─────────────────┘
        │
        ├─► Create Swarm
        │   └─► Deploy Agents (E2B Sandboxes)
        │
        ├─► Execute Operations
        │   ├─► Strategy Execution
        │   ├─► Consensus
        │   └─► Communication
        │
        ├─► Measure Performance
        │   ├─► Timing (performance.now)
        │   ├─► Resources (psutil in Python)
        │   └─► Costs (E2B pricing)
        │
        ├─► Collect Metrics
        │   ├─► Latency measurements
        │   ├─► Throughput calculations
        │   └─► Resource snapshots
        │
        ├─► Analyze Results
        │   ├─► Calculate statistics
        │   ├─► Compare to targets
        │   └─► Identify bottlenecks
        │
        └─► Generate Report
            ├─► Markdown formatting
            ├─► Tables and charts
            └─► Recommendations
                │
                ▼
        SWARM_BENCHMARKS_REPORT.md
```

---

## Topology Architectures

### Mesh Topology
```
Agent Network: Full Mesh (6 agents)

    A1 ─────── A2
    │╲       ╱│
    │  ╲   ╱  │
    │    ╳    │
    │  ╱   ╲  │
    │╱       ╲│
    A3 ─────── A4
    │╲       ╱│
    │  ╲   ╱  │
    │    ╳    │
    │  ╱   ╲  │
    │╱       ╲│
    A5 ─────── A6

Connections: n(n-1)/2 = 15
Max Hops: 1
Latency: Lowest
Cost: Highest at scale
```

### Hierarchical Topology
```
Agent Network: Hierarchical (7 agents)

         A1 (Root)
        ╱  ╲
       ╱    ╲
      A2    A3
     ╱ ╲   ╱ ╲
    A4 A5 A6 A7

Connections: n-1 = 6
Max Hops: log₂(n) = 3
Latency: Moderate
Cost: Best at scale
```

### Ring Topology
```
Agent Network: Ring (6 agents)

    A1 → A2
     ↑    ↓
    A6    A3
     ↑    ↓
    A5 ← A4

Connections: n = 6
Max Hops: n/2 = 3
Latency: Moderate
Cost: Balanced
```

---

## Metrics Collection Flow

```
Continuous Monitoring
│
├─► Time Metrics
│   ├─ performance.now() → Start
│   ├─ Execute operation
│   ├─ performance.now() → End
│   └─ Duration = End - Start
│
├─► Resource Metrics
│   ├─ psutil.cpu_percent()
│   ├─ psutil.memory_info()
│   ├─ psutil.net_io_counters()
│   └─ Store snapshot
│
├─► Cost Metrics
│   ├─ Sandbox count
│   ├─ Duration (ms)
│   ├─ CPU/memory usage
│   └─ Calculate: sandbox + cpu + memory costs
│
└─► Aggregation
    ├─ Collect all measurements
    ├─ Calculate statistics
    │   ├─ Mean
    │   ├─ Median
    │   ├─ Standard deviation
    │   └─ Percentiles (P50, P95, P99)
    │
    └─ Store in benchmarkResults
```

---

## Report Generation Pipeline

```
Benchmark Results
        │
        ▼
┌────────────────────┐
│  Result Processor  │
└────────────────────┘
        │
        ├─► Format Statistics
        │   └─ Create tables
        │
        ├─► Generate Charts
        │   └─ ASCII visualizations
        │
        ├─► Analyze Performance
        │   └─ Compare to targets
        │
        ├─► Cost Analysis
        │   └─ Calculate ROI
        │
        ├─► Recommendations
        │   └─ Prioritize improvements
        │
        └─► Write Markdown
            └─ SWARM_BENCHMARKS_REPORT.md
                │
                ├─ Executive Summary
                ├─ Detailed Metrics
                ├─ Comparison Charts
                ├─ Recommendations
                └─ Conclusions
```

---

## Error Handling

```
Benchmark Execution
        │
        ├─ Try: Run test
        │   ├─ Success → Continue
        │   └─ Error → Handle
        │       ├─ Log error
        │       ├─ Cleanup resources
        │       ├─ Mark test failed
        │       └─ Continue to next test
        │
        └─ Finally: Always cleanup
            ├─ Destroy sandboxes
            ├─ Save partial results
            └─ Generate report
```

---

## Performance Optimization Points

```
Optimization Opportunities
│
├─ Parallel Execution
│  ├─ Deploy agents concurrently
│  ├─ Execute strategies in parallel
│  └─ Measure multiple metrics simultaneously
│
├─ Connection Pooling
│  ├─ Reuse sandbox connections
│  ├─ Minimize initialization overhead
│  └─ Batch operations
│
├─ Caching
│  ├─ Cache market data
│  ├─ Cache strategy results
│  └─ Reuse calculations
│
└─ Batch Processing
   ├─ Group similar operations
   ├─ Reduce network calls
   └─ Optimize resource usage
```

---

## CI/CD Integration Architecture

```
GitHub Actions Workflow
│
├─ Trigger
│  ├─ Schedule (weekly)
│  ├─ Manual dispatch
│  └─ Pull request
│
├─ Setup
│  ├─ Checkout code
│  ├─ Setup Node.js
│  ├─ Install dependencies
│  └─ Set E2B_API_KEY
│
├─ Execute
│  └─ npm run bench:swarm:full
│      │
│      ├─ Run all benchmarks
│      ├─ Generate report
│      └─ Exit with status
│
├─ Analyze
│  ├─ Check for failures (❌)
│  ├─ Compare to previous
│  └─ Detect regressions
│
└─ Artifact
   ├─ Upload report
   ├─ Send notifications
   └─ Update dashboard
```

---

## Scalability Patterns

```
Agent Count Scaling
│
├─ 1 Agent (Baseline)
│  └─ Minimal overhead
│
├─ 2-5 Agents (Small)
│  └─ Linear scaling
│
├─ 6-10 Agents (Medium)
│  └─ Sub-linear scaling
│
├─ 11-20 Agents (Large)
│  └─ Logarithmic scaling
│
└─ 20+ Agents (Very Large)
   └─ Hierarchical recommended
```

---

This architecture document provides a comprehensive view of how the benchmark system works, from high-level flow to detailed component interactions.

For implementation details, see:
- `/tests/e2b/swarm-benchmarks.test.js`
- `/docs/e2b/BENCHMARK_GUIDE.md`
- `/docs/e2b/BENCHMARK_QUICK_START.md`
