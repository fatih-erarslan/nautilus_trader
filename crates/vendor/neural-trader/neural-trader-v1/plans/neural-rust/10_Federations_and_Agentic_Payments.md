# Federations & Agentic Payments

## Document Overview

**Status**: Production-Ready Planning
**Last Updated**: 2025-11-12
**Owner**: Architecture & Finance Team
**Related Docs**: `09_E2B_Sandboxes_and_Supply_Chain.md`, `12_Secrets_and_Environments.md`

## Executive Summary

This document covers:
- **Agentic Flow Federations**: Multi-agent coordination at scale (mesh vs hierarchical topologies)
- **Multi-Strategy Coordination**: Orchestrating multiple trading strategies concurrently
- **Quota & Cost Controls**: Budget enforcement and resource management
- **Agentic Payments**: Usage-based billing, credit systems, and cost tracking
- **Cost Optimization**: Strategies for minimizing LLM, compute, and infrastructure costs

---

## 1. Agentic Flow Federation Architecture

### 1.1 Overview

**Federation** enables multiple trading agents to coordinate across:
- Multiple exchanges
- Multiple strategies
- Multiple regions
- Multiple time horizons

**Key Benefits:**
- **Horizontal Scaling**: Add agents without bottlenecks
- **Fault Tolerance**: Agent failures don't cascade
- **Geographic Distribution**: Low-latency execution globally
- **Strategy Isolation**: Strategies don't interfere with each other

### 1.2 Federation Topologies

#### Topology 1: Mesh (Peer-to-Peer)

```
┌─────────────────────────────────────────────────────────┐
│                    Mesh Federation                       │
│                                                          │
│     Agent A ←──────────→ Agent B                        │
│        ↕                    ↕                           │
│        ↕                    ↕                           │
│     Agent C ←──────────→ Agent D                        │
│        ↕                    ↕                           │
│        ↕                    ↕                           │
│     Agent E ←──────────→ Agent F                        │
│                                                          │
│  • Each agent knows about all others                    │
│  • Direct communication (low latency)                   │
│  • No single point of failure                           │
│  • Best for: <10 agents, low-latency trading           │
└─────────────────────────────────────────────────────────┘
```

**Pros:**
- Low latency (direct communication)
- No central coordinator bottleneck
- High fault tolerance

**Cons:**
- Complex coordination (O(n²) connections)
- Difficult to maintain consistency
- Higher network overhead

**Use Cases:**
- High-frequency trading (HFT)
- Market making with <10 agents
- Real-time arbitrage

#### Topology 2: Hierarchical (Leader-Based)

```
┌─────────────────────────────────────────────────────────┐
│                 Hierarchical Federation                  │
│                                                          │
│                  ┌─────────────┐                        │
│                  │ Coordinator │                        │
│                  │   (Leader)  │                        │
│                  └──────┬──────┘                        │
│                         │                               │
│        ┌────────────────┼────────────────┐             │
│        ▼                ▼                ▼             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│  │ Agent A  │    │ Agent B  │    │ Agent C  │         │
│  │(Strategy1)│    │(Strategy2)│    │(Strategy3)│         │
│  └──────────┘    └──────────┘    └──────────┘         │
│                                                          │
│  • Coordinator assigns tasks                            │
│  • Agents report back to coordinator                    │
│  • Simple coordination logic                            │
│  • Best for: 10-100 agents, complex strategies         │
└─────────────────────────────────────────────────────────┘
```

**Pros:**
- Simple coordination (O(n) connections)
- Centralized decision-making
- Easy to implement and debug

**Cons:**
- Coordinator is single point of failure
- Coordinator can become bottleneck
- Higher latency (two hops for agent-to-agent communication)

**Use Cases:**
- Multi-strategy portfolios
- Risk-managed trading (coordinator enforces limits)
- Backtesting orchestration

#### Topology 3: Hybrid (Regional Coordinators)

```
┌─────────────────────────────────────────────────────────┐
│                   Hybrid Federation                      │
│                                                          │
│             ┌─────────────────────┐                     │
│             │  Global Coordinator │                     │
│             └──────────┬──────────┘                     │
│                        │                                │
│        ┌───────────────┼───────────────┐               │
│        ▼               ▼               ▼               │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐          │
│  │ Regional  │  │ Regional  │  │ Regional  │          │
│  │   (US)    │  │   (EU)    │  │  (APAC)   │          │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘          │
│        │              │              │                 │
│    ┌───┴───┐      ┌───┴───┐      ┌───┴───┐            │
│    ▼       ▼      ▼       ▼      ▼       ▼            │
│  Agent   Agent  Agent   Agent  Agent   Agent           │
│    A       B      C       D      E       F             │
│                                                          │
│  • Regional coordinators for low latency                │
│  • Global coordinator for risk/capital allocation       │
│  • Best for: 100+ agents, global operations            │
└─────────────────────────────────────────────────────────┘
```

**Pros:**
- Scales to 100+ agents
- Low latency within regions
- Fault tolerance (regional coordinators)

**Cons:**
- Most complex to implement
- Requires careful state synchronization
- Higher operational overhead

**Use Cases:**
- Global trading operations
- Multi-asset class portfolios
- Institutional-grade systems

---

## 2. Federation Implementation

### 2.1 Mesh Federation (Rust)

```rust
// src/federation/mesh.rs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    pub id: String,
    pub name: String,
    pub role: String,
    pub endpoint: String,
    pub status: AgentStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgentStatus {
    Active,
    Idle,
    Offline,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub from: String,
    pub to: String,
    pub msg_type: MessageType,
    pub payload: serde_json::Value,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    TradeSignal,
    PositionUpdate,
    RiskAlert,
    Heartbeat,
}

pub struct MeshFederation {
    agents: Arc<RwLock<HashMap<String, AgentInfo>>>,
    message_queue: Arc<RwLock<Vec<Message>>>,
}

impl MeshFederation {
    pub fn new() -> Self {
        Self {
            agents: Arc::new(RwLock::new(HashMap::new())),
            message_queue: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a new agent in the federation
    pub async fn register_agent(&self, agent: AgentInfo) -> Result<(), String> {
        let mut agents = self.agents.write().await;

        if agents.contains_key(&agent.id) {
            return Err(format!("Agent {} already registered", agent.id));
        }

        agents.insert(agent.id.clone(), agent);
        Ok(())
    }

    /// Broadcast message to all agents
    pub async fn broadcast(&self, message: Message) -> Result<(), String> {
        let agents = self.agents.read().await;

        for (agent_id, agent) in agents.iter() {
            if agent_id != &message.from && agent.status == AgentStatus::Active {
                let mut msg = message.clone();
                msg.to = agent_id.clone();
                self.send_message(msg).await?;
            }
        }

        Ok(())
    }

    /// Send message to specific agent
    pub async fn send_message(&self, message: Message) -> Result<(), String> {
        let agents = self.agents.read().await;

        if !agents.contains_key(&message.to) {
            return Err(format!("Agent {} not found", message.to));
        }

        let mut queue = self.message_queue.write().await;
        queue.push(message);

        Ok(())
    }

    /// Get messages for specific agent
    pub async fn get_messages(&self, agent_id: &str) -> Vec<Message> {
        let mut queue = self.message_queue.write().await;
        let (mine, others): (Vec<_>, Vec<_>) = queue
            .drain(..)
            .partition(|msg| msg.to == agent_id);

        *queue = others;
        mine
    }

    /// Update agent status
    pub async fn update_agent_status(
        &self,
        agent_id: &str,
        status: AgentStatus,
    ) -> Result<(), String> {
        let mut agents = self.agents.write().await;

        let agent = agents
            .get_mut(agent_id)
            .ok_or_else(|| format!("Agent {} not found", agent_id))?;

        agent.status = status;
        Ok(())
    }

    /// List all active agents
    pub async fn list_active_agents(&self) -> Vec<AgentInfo> {
        let agents = self.agents.read().await;
        agents
            .values()
            .filter(|a| a.status == AgentStatus::Active)
            .cloned()
            .collect()
    }
}
```

### 2.2 Hierarchical Federation (Rust)

```rust
// src/federation/hierarchical.rs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub task_type: TaskType,
    pub assigned_to: Option<String>,
    pub status: TaskStatus,
    pub payload: serde_json::Value,
    pub priority: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    Backtest,
    LiveTrade,
    RiskCheck,
    DataFetch,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskStatus {
    Pending,
    Assigned,
    InProgress,
    Completed,
    Failed,
}

pub struct HierarchicalFederation {
    agents: Arc<RwLock<HashMap<String, AgentInfo>>>,
    tasks: Arc<RwLock<HashMap<String, Task>>>,
}

impl HierarchicalFederation {
    pub fn new() -> Self {
        Self {
            agents: Arc::new(RwLock::new(HashMap::new())),
            tasks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Coordinator assigns task to agent
    pub async fn assign_task(
        &self,
        task_id: &str,
        agent_id: &str,
    ) -> Result<(), String> {
        let agents = self.agents.read().await;
        if !agents.contains_key(agent_id) {
            return Err(format!("Agent {} not found", agent_id));
        }

        let mut tasks = self.tasks.write().await;
        let task = tasks
            .get_mut(task_id)
            .ok_or_else(|| format!("Task {} not found", task_id))?;

        if task.status != TaskStatus::Pending {
            return Err(format!("Task {} not in pending state", task_id));
        }

        task.assigned_to = Some(agent_id.to_string());
        task.status = TaskStatus::Assigned;

        Ok(())
    }

    /// Create new task
    pub async fn create_task(&self, task: Task) -> Result<(), String> {
        let mut tasks = self.tasks.write().await;

        if tasks.contains_key(&task.id) {
            return Err(format!("Task {} already exists", task.id));
        }

        tasks.insert(task.id.clone(), task);
        Ok(())
    }

    /// Agent reports task completion
    pub async fn complete_task(
        &self,
        task_id: &str,
        result: serde_json::Value,
    ) -> Result<(), String> {
        let mut tasks = self.tasks.write().await;
        let task = tasks
            .get_mut(task_id)
            .ok_or_else(|| format!("Task {} not found", task_id))?;

        task.status = TaskStatus::Completed;
        task.payload = result;

        Ok(())
    }

    /// Get pending tasks (for assignment)
    pub async fn get_pending_tasks(&self) -> Vec<Task> {
        let tasks = self.tasks.read().await;
        tasks
            .values()
            .filter(|t| t.status == TaskStatus::Pending)
            .cloned()
            .collect()
    }

    /// Get tasks for specific agent
    pub async fn get_agent_tasks(&self, agent_id: &str) -> Vec<Task> {
        let tasks = self.tasks.read().await;
        tasks
            .values()
            .filter(|t| {
                t.assigned_to.as_ref().map(|id| id == agent_id).unwrap_or(false)
                    && t.status != TaskStatus::Completed
            })
            .cloned()
            .collect()
    }
}
```

---

## 3. Multi-Strategy Coordination

### 3.1 Strategy Coordination Patterns

**Pattern 1: Portfolio Allocation**

```rust
// src/federation/portfolio_allocator.rs

use rust_decimal::Decimal;
use std::collections::HashMap;

pub struct PortfolioAllocator {
    total_capital: Decimal,
    strategy_allocations: HashMap<String, Decimal>,
}

impl PortfolioAllocator {
    pub fn new(total_capital: Decimal) -> Self {
        Self {
            total_capital,
            strategy_allocations: HashMap::new(),
        }
    }

    /// Allocate capital to strategies based on performance
    pub fn allocate_capital(
        &mut self,
        strategies: &HashMap<String, StrategyMetrics>,
    ) -> HashMap<String, Decimal> {
        let total_sharpe: Decimal = strategies
            .values()
            .map(|m| m.sharpe_ratio.max(Decimal::ZERO))
            .sum();

        let mut allocations = HashMap::new();

        for (strategy_id, metrics) in strategies {
            let weight = if total_sharpe > Decimal::ZERO {
                metrics.sharpe_ratio.max(Decimal::ZERO) / total_sharpe
            } else {
                Decimal::from(1) / Decimal::from(strategies.len() as u64)
            };

            let allocation = self.total_capital * weight;
            allocations.insert(strategy_id.clone(), allocation);
        }

        self.strategy_allocations = allocations.clone();
        allocations
    }

    /// Rebalance based on drift
    pub fn rebalance_if_needed(
        &self,
        current_values: &HashMap<String, Decimal>,
        threshold: Decimal,
    ) -> Option<HashMap<String, Decimal>> {
        let total_value: Decimal = current_values.values().sum();

        let mut needs_rebalance = false;
        for (strategy_id, target_allocation) in &self.strategy_allocations {
            let current_value = current_values.get(strategy_id).copied().unwrap_or(Decimal::ZERO);
            let current_weight = current_value / total_value;
            let target_weight = target_allocation / self.total_capital;

            if (current_weight - target_weight).abs() > threshold {
                needs_rebalance = true;
                break;
            }
        }

        if needs_rebalance {
            Some(self.strategy_allocations.clone())
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct StrategyMetrics {
    pub sharpe_ratio: Decimal,
    pub returns: Decimal,
    pub volatility: Decimal,
    pub max_drawdown: Decimal,
}
```

**Pattern 2: Risk Aggregation**

```rust
// src/federation/risk_aggregator.rs

use rust_decimal::Decimal;
use std::collections::HashMap;

pub struct RiskAggregator {
    max_total_var: Decimal,
    max_correlation: Decimal,
}

impl RiskAggregator {
    pub fn new(max_total_var: Decimal, max_correlation: Decimal) -> Self {
        Self {
            max_total_var,
            max_correlation,
        }
    }

    /// Check if new trade violates aggregate risk limits
    pub fn check_aggregate_risk(
        &self,
        current_positions: &HashMap<String, Position>,
        new_trade: &Trade,
    ) -> Result<(), String> {
        // 1. Calculate total VaR
        let total_var = self.calculate_total_var(current_positions, Some(new_trade));

        if total_var > self.max_total_var {
            return Err(format!(
                "Total VaR {} exceeds limit {}",
                total_var, self.max_total_var
            ));
        }

        // 2. Check correlation limits
        let correlation = self.calculate_correlation(current_positions, new_trade);

        if correlation > self.max_correlation {
            return Err(format!(
                "Correlation {} exceeds limit {}",
                correlation, self.max_correlation
            ));
        }

        Ok(())
    }

    fn calculate_total_var(
        &self,
        positions: &HashMap<String, Position>,
        new_trade: Option<&Trade>,
    ) -> Decimal {
        // Simplified VaR calculation
        let mut total_exposure = Decimal::ZERO;

        for position in positions.values() {
            total_exposure += position.value * position.volatility;
        }

        if let Some(trade) = new_trade {
            total_exposure += trade.notional_value * Decimal::from_str_exact("0.20").unwrap(); // Assume 20% vol
        }

        // VaR = 1.65 * std_dev (95% confidence)
        total_exposure * Decimal::from_str_exact("1.65").unwrap()
    }

    fn calculate_correlation(
        &self,
        positions: &HashMap<String, Position>,
        new_trade: &Trade,
    ) -> Decimal {
        // Simplified correlation check
        for position in positions.values() {
            if position.symbol == new_trade.symbol {
                return Decimal::ONE;
            }

            // Check asset class correlation (simplified)
            if self.get_asset_class(&position.symbol) == self.get_asset_class(&new_trade.symbol) {
                return Decimal::from_str_exact("0.70").unwrap(); // High correlation
            }
        }

        Decimal::from_str_exact("0.30").unwrap() // Low correlation
    }

    fn get_asset_class(&self, symbol: &str) -> &str {
        if symbol.ends_with("USD") {
            "crypto"
        } else {
            "equity"
        }
    }
}

#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub value: Decimal,
    pub volatility: Decimal,
}

#[derive(Debug, Clone)]
pub struct Trade {
    pub symbol: String,
    pub notional_value: Decimal,
}
```

---

## 4. Quota & Cost Controls

### 4.1 Resource Quotas

**Configuration:**

```toml
# config/quotas.toml

[agents]
max_concurrent_agents = 50
max_agents_per_user = 10
max_lifetime_hours = 24

[llm]
max_tokens_per_day = 1_000_000
max_requests_per_minute = 100
max_context_length = 128_000

[sandboxes]
max_concurrent_sandboxes = 20
max_cpu_cores_total = 100
max_memory_gb_total = 500
max_disk_gb_total = 2000

[api]
max_requests_per_minute = 1000
max_requests_per_day = 100_000
max_concurrent_connections = 100

[storage]
max_total_storage_gb = 1000
max_file_size_mb = 500
```

### 4.2 Quota Enforcement

```rust
// src/quotas/enforcer.rs

use rust_decimal::Decimal;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
pub struct QuotaConfig {
    pub max_tokens_per_day: u64,
    pub max_requests_per_minute: u32,
    pub max_concurrent_agents: u32,
    pub max_cost_per_day: Decimal,
}

pub struct QuotaEnforcer {
    config: QuotaConfig,
    usage: Arc<RwLock<UsageMetrics>>,
}

#[derive(Debug, Default)]
struct UsageMetrics {
    tokens_today: u64,
    requests_this_minute: u32,
    concurrent_agents: u32,
    cost_today: Decimal,
    reset_time: Option<Instant>,
}

impl QuotaEnforcer {
    pub fn new(config: QuotaConfig) -> Self {
        Self {
            config,
            usage: Arc::new(RwLock::new(UsageMetrics::default())),
        }
    }

    /// Check if request is within quota
    pub async fn check_quota(
        &self,
        tokens: u64,
        cost: Decimal,
    ) -> Result<(), String> {
        let mut usage = self.usage.write().await;

        // Reset daily counters if needed
        if let Some(reset_time) = usage.reset_time {
            if reset_time.elapsed() >= Duration::from_secs(86400) {
                usage.tokens_today = 0;
                usage.cost_today = Decimal::ZERO;
                usage.reset_time = Some(Instant::now());
            }
        } else {
            usage.reset_time = Some(Instant::now());
        }

        // Check token quota
        if usage.tokens_today + tokens > self.config.max_tokens_per_day {
            return Err(format!(
                "Daily token quota exceeded: {} / {}",
                usage.tokens_today, self.config.max_tokens_per_day
            ));
        }

        // Check cost quota
        if usage.cost_today + cost > self.config.max_cost_per_day {
            return Err(format!(
                "Daily cost quota exceeded: {} / {}",
                usage.cost_today, self.config.max_cost_per_day
            ));
        }

        // Check request rate
        if usage.requests_this_minute >= self.config.max_requests_per_minute {
            return Err(format!(
                "Rate limit exceeded: {} req/min",
                self.config.max_requests_per_minute
            ));
        }

        // Update counters
        usage.tokens_today += tokens;
        usage.cost_today += cost;
        usage.requests_this_minute += 1;

        Ok(())
    }

    /// Check agent quota
    pub async fn check_agent_quota(&self) -> Result<(), String> {
        let usage = self.usage.read().await;

        if usage.concurrent_agents >= self.config.max_concurrent_agents {
            return Err(format!(
                "Max concurrent agents reached: {}",
                self.config.max_concurrent_agents
            ));
        }

        Ok(())
    }

    /// Increment agent count
    pub async fn increment_agent_count(&self) {
        let mut usage = self.usage.write().await;
        usage.concurrent_agents += 1;
    }

    /// Decrement agent count
    pub async fn decrement_agent_count(&self) {
        let mut usage = self.usage.write().await;
        usage.concurrent_agents = usage.concurrent_agents.saturating_sub(1);
    }

    /// Get current usage
    pub async fn get_usage(&self) -> UsageSnapshot {
        let usage = self.usage.read().await;
        UsageSnapshot {
            tokens_today: usage.tokens_today,
            cost_today: usage.cost_today,
            concurrent_agents: usage.concurrent_agents,
            requests_this_minute: usage.requests_this_minute,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct UsageSnapshot {
    pub tokens_today: u64,
    pub cost_today: Decimal,
    pub concurrent_agents: u32,
    pub requests_this_minute: u32,
}
```

---

## 5. Agentic Payments Integration

### 5.1 Cost Formulas

**LLM Costs:**

| Model | Input ($/1M tokens) | Output ($/1M tokens) |
|-------|---------------------|----------------------|
| GPT-4 | $10.00 | $30.00 |
| GPT-4-Turbo | $1.00 | $3.00 |
| Claude 3.5 Sonnet | $3.00 | $15.00 |
| Claude 3 Haiku | $0.25 | $1.25 |

**Sandbox Costs:**

| Resource | Cost |
|----------|------|
| CPU (per core-hour) | $0.10 |
| Memory (per GB-hour) | $0.01 |
| Disk (per GB-hour) | $0.0001 |
| GPU A100 (per hour) | $2.50 |
| GPU V100 (per hour) | $1.25 |

**Data Transfer Costs:**

| Transfer Type | Cost |
|---------------|------|
| Ingress | Free |
| Egress (first 100GB/month) | Free |
| Egress (>100GB/month) | $0.09/GB |

### 5.2 Cost Calculation

```rust
// src/payments/cost_calculator.rs

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct CostBreakdown {
    pub llm_cost: Decimal,
    pub sandbox_cost: Decimal,
    pub storage_cost: Decimal,
    pub data_transfer_cost: Decimal,
    pub total_cost: Decimal,
}

pub struct CostCalculator {
    llm_rates: LLMRates,
    sandbox_rates: SandboxRates,
}

#[derive(Debug, Clone)]
struct LLMRates {
    gpt4_input: Decimal,
    gpt4_output: Decimal,
    gpt4_turbo_input: Decimal,
    gpt4_turbo_output: Decimal,
    claude_sonnet_input: Decimal,
    claude_sonnet_output: Decimal,
}

impl Default for LLMRates {
    fn default() -> Self {
        Self {
            gpt4_input: Decimal::from_str_exact("0.00001").unwrap(), // $10/1M
            gpt4_output: Decimal::from_str_exact("0.00003").unwrap(), // $30/1M
            gpt4_turbo_input: Decimal::from_str_exact("0.000001").unwrap(),
            gpt4_turbo_output: Decimal::from_str_exact("0.000003").unwrap(),
            claude_sonnet_input: Decimal::from_str_exact("0.000003").unwrap(),
            claude_sonnet_output: Decimal::from_str_exact("0.000015").unwrap(),
        }
    }
}

#[derive(Debug, Clone)]
struct SandboxRates {
    cpu_per_core_hour: Decimal,
    memory_per_gb_hour: Decimal,
    disk_per_gb_hour: Decimal,
    gpu_a100_per_hour: Decimal,
}

impl Default for SandboxRates {
    fn default() -> Self {
        Self {
            cpu_per_core_hour: Decimal::from_str_exact("0.10").unwrap(),
            memory_per_gb_hour: Decimal::from_str_exact("0.01").unwrap(),
            disk_per_gb_hour: Decimal::from_str_exact("0.0001").unwrap(),
            gpu_a100_per_hour: Decimal::from_str_exact("2.50").unwrap(),
        }
    }
}

impl CostCalculator {
    pub fn new() -> Self {
        Self {
            llm_rates: LLMRates::default(),
            sandbox_rates: SandboxRates::default(),
        }
    }

    pub fn calculate_llm_cost(
        &self,
        model: &str,
        input_tokens: u64,
        output_tokens: u64,
    ) -> Decimal {
        let (input_rate, output_rate) = match model {
            "gpt-4" => (self.llm_rates.gpt4_input, self.llm_rates.gpt4_output),
            "gpt-4-turbo" => (self.llm_rates.gpt4_turbo_input, self.llm_rates.gpt4_turbo_output),
            "claude-3.5-sonnet" => (self.llm_rates.claude_sonnet_input, self.llm_rates.claude_sonnet_output),
            _ => (Decimal::ZERO, Decimal::ZERO),
        };

        let input_cost = Decimal::from(input_tokens) * input_rate;
        let output_cost = Decimal::from(output_tokens) * output_rate;

        input_cost + output_cost
    }

    pub fn calculate_sandbox_cost(
        &self,
        cpu_cores: u32,
        memory_gb: u64,
        disk_gb: u64,
        gpu_count: u32,
        duration_hours: f64,
    ) -> Decimal {
        let cpu_cost = Decimal::from(cpu_cores)
            * self.sandbox_rates.cpu_per_core_hour
            * Decimal::from_f64_retain(duration_hours).unwrap();

        let memory_cost = Decimal::from(memory_gb)
            * self.sandbox_rates.memory_per_gb_hour
            * Decimal::from_f64_retain(duration_hours).unwrap();

        let disk_cost = Decimal::from(disk_gb)
            * self.sandbox_rates.disk_per_gb_hour
            * Decimal::from_f64_retain(duration_hours).unwrap();

        let gpu_cost = Decimal::from(gpu_count)
            * self.sandbox_rates.gpu_a100_per_hour
            * Decimal::from_f64_retain(duration_hours).unwrap();

        cpu_cost + memory_cost + disk_cost + gpu_cost
    }

    pub fn calculate_total_cost(
        &self,
        llm_usage: &[(String, u64, u64)], // (model, input_tokens, output_tokens)
        sandbox_usage: &[(u32, u64, u64, u32, f64)], // (cpu, mem, disk, gpu, hours)
    ) -> CostBreakdown {
        let llm_cost: Decimal = llm_usage
            .iter()
            .map(|(model, input, output)| self.calculate_llm_cost(model, *input, *output))
            .sum();

        let sandbox_cost: Decimal = sandbox_usage
            .iter()
            .map(|(cpu, mem, disk, gpu, hours)| {
                self.calculate_sandbox_cost(*cpu, *mem, *disk, *gpu, *hours)
            })
            .sum();

        let total_cost = llm_cost + sandbox_cost;

        CostBreakdown {
            llm_cost,
            sandbox_cost,
            storage_cost: Decimal::ZERO,
            data_transfer_cost: Decimal::ZERO,
            total_cost,
        }
    }
}
```

### 5.3 Monthly Budget Tracking

```rust
// src/payments/budget_tracker.rs

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Budget {
    pub monthly_limit: Decimal,
    pub current_spend: Decimal,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
}

impl Budget {
    pub fn new(monthly_limit: Decimal) -> Self {
        let now = Utc::now();
        let period_start = DateTime::from_utc(
            chrono::NaiveDate::from_ymd_opt(now.year(), now.month(), 1)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            Utc,
        );

        let period_end = if now.month() == 12 {
            DateTime::from_utc(
                chrono::NaiveDate::from_ymd_opt(now.year() + 1, 1, 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap(),
                Utc,
            )
        } else {
            DateTime::from_utc(
                chrono::NaiveDate::from_ymd_opt(now.year(), now.month() + 1, 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap(),
                Utc,
            )
        };

        Self {
            monthly_limit,
            current_spend: Decimal::ZERO,
            period_start,
            period_end,
        }
    }

    pub fn add_cost(&mut self, cost: Decimal) -> Result<(), String> {
        // Check if period has reset
        if Utc::now() > self.period_end {
            self.reset_period();
        }

        // Check budget limit
        if self.current_spend + cost > self.monthly_limit {
            return Err(format!(
                "Budget exceeded: ${} / ${} monthly limit",
                self.current_spend + cost,
                self.monthly_limit
            ));
        }

        self.current_spend += cost;
        Ok(())
    }

    pub fn remaining_budget(&self) -> Decimal {
        self.monthly_limit - self.current_spend
    }

    pub fn utilization_percent(&self) -> Decimal {
        if self.monthly_limit == Decimal::ZERO {
            Decimal::ZERO
        } else {
            (self.current_spend / self.monthly_limit) * Decimal::from(100)
        }
    }

    fn reset_period(&mut self) {
        self.current_spend = Decimal::ZERO;
        let now = Utc::now();

        self.period_start = DateTime::from_utc(
            chrono::NaiveDate::from_ymd_opt(now.year(), now.month(), 1)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            Utc,
        );

        self.period_end = if now.month() == 12 {
            DateTime::from_utc(
                chrono::NaiveDate::from_ymd_opt(now.year() + 1, 1, 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap(),
                Utc,
            )
        } else {
            DateTime::from_utc(
                chrono::NaiveDate::from_ymd_opt(now.year(), now.month() + 1, 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap(),
                Utc,
            )
        };
    }
}
```

### 5.4 Payment Webhook Handling

```rust
// src/payments/webhooks.rs

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Deserialize)]
struct WebhookPayload {
    event_type: String,
    user_id: String,
    amount: String,
    currency: String,
    timestamp: i64,
    signature: String,
}

#[derive(Debug, Serialize)]
struct WebhookResponse {
    status: String,
    message: String,
}

struct AppState {
    webhook_secret: String,
}

pub fn create_webhook_router(webhook_secret: String) -> Router {
    let state = Arc::new(AppState { webhook_secret });

    Router::new()
        .route("/webhooks/payment", post(handle_payment_webhook))
        .with_state(state)
}

async fn handle_payment_webhook(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<WebhookPayload>,
) -> impl IntoResponse {
    // 1. Verify signature
    if !verify_signature(&payload, &state.webhook_secret) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(WebhookResponse {
                status: "error".to_string(),
                message: "Invalid signature".to_string(),
            }),
        );
    }

    // 2. Handle event
    match payload.event_type.as_str() {
        "payment.succeeded" => {
            // Credit user account
            println!("Payment succeeded for user {}: {}", payload.user_id, payload.amount);
        }
        "payment.failed" => {
            // Handle failure
            println!("Payment failed for user {}", payload.user_id);
        }
        "subscription.cancelled" => {
            // Handle cancellation
            println!("Subscription cancelled for user {}", payload.user_id);
        }
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(WebhookResponse {
                    status: "error".to_string(),
                    message: format!("Unknown event type: {}", payload.event_type),
                }),
            );
        }
    }

    (
        StatusCode::OK,
        Json(WebhookResponse {
            status: "success".to_string(),
            message: "Webhook processed".to_string(),
        }),
    )
}

fn verify_signature(payload: &WebhookPayload, secret: &str) -> bool {
    use sha2::{Digest, Sha256};

    let data = format!(
        "{}:{}:{}:{}:{}",
        payload.event_type, payload.user_id, payload.amount, payload.currency, payload.timestamp
    );

    let mut hasher = Sha256::new();
    hasher.update(data.as_bytes());
    hasher.update(secret.as_bytes());

    let computed_signature = format!("{:x}", hasher.finalize());

    computed_signature == payload.signature
}
```

---

## 6. Cost Optimization Strategies

### 6.1 Model Selection

**Strategy: Use cheapest model that meets requirements**

```rust
// src/optimization/model_selector.rs

pub enum TaskComplexity {
    Simple,    // Use Haiku/GPT-3.5
    Medium,    // Use Sonnet/GPT-4-Turbo
    Complex,   // Use Opus/GPT-4
}

pub fn select_model(complexity: TaskComplexity) -> &'static str {
    match complexity {
        TaskComplexity::Simple => "claude-3-haiku",    // $0.25/$1.25 per 1M
        TaskComplexity::Medium => "gpt-4-turbo",       // $1/$3 per 1M
        TaskComplexity::Complex => "claude-3.5-sonnet", // $3/$15 per 1M
    }
}
```

### 6.2 Prompt Caching

**Strategy: Cache system prompts to reduce input tokens**

```rust
// System prompt cached (billed once)
let cached_system_prompt = "You are a trading assistant..."; // 1000 tokens

// User query (new tokens)
let user_query = "What's the current BTC price?"; // 10 tokens

// Total cost: 1000 tokens (first call) + 10 tokens (subsequent calls)
// Savings: 99% on system prompt after first call
```

### 6.3 Batch Processing

**Strategy: Batch multiple requests to amortize overhead**

```rust
// Bad: Sequential requests
for symbol in symbols {
    let price = fetch_price(symbol).await;  // N requests
}

// Good: Batch request
let prices = fetch_prices_batch(symbols).await;  // 1 request
```

### 6.4 Smart Caching

```rust
// src/optimization/cache.rs

use std::collections::HashMap;
use std::time::{Duration, Instant};

pub struct SmartCache<K, V> {
    cache: HashMap<K, CachedValue<V>>,
    ttl: Duration,
}

struct CachedValue<V> {
    value: V,
    inserted_at: Instant,
}

impl<K: std::hash::Hash + Eq + Clone, V: Clone> SmartCache<K, V> {
    pub fn new(ttl_seconds: u64) -> Self {
        Self {
            cache: HashMap::new(),
            ttl: Duration::from_secs(ttl_seconds),
        }
    }

    pub fn get(&mut self, key: &K) -> Option<V> {
        if let Some(cached) = self.cache.get(key) {
            if cached.inserted_at.elapsed() < self.ttl {
                return Some(cached.value.clone());
            } else {
                self.cache.remove(key);
            }
        }
        None
    }

    pub fn insert(&mut self, key: K, value: V) {
        self.cache.insert(
            key,
            CachedValue {
                value,
                inserted_at: Instant::now(),
            },
        );
    }
}
```

---

## 7. Usage Metering & Billing

### 7.1 Metering Implementation

```rust
// src/payments/metering.rs

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;

#[derive(Debug, Serialize, Deserialize)]
pub struct UsageEvent {
    pub user_id: String,
    pub resource_type: ResourceType,
    pub quantity: Decimal,
    pub unit_cost: Decimal,
    pub total_cost: Decimal,
    pub timestamp: DateTime<Utc>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ResourceType {
    LLMTokens,
    SandboxCPU,
    SandboxMemory,
    SandboxGPU,
    Storage,
    DataTransfer,
}

pub struct MeteringService {
    db: PgPool,
}

impl MeteringService {
    pub fn new(db: PgPool) -> Self {
        Self { db }
    }

    pub async fn record_usage(&self, event: UsageEvent) -> Result<(), String> {
        sqlx::query!(
            r#"
            INSERT INTO usage_events (user_id, resource_type, quantity, unit_cost, total_cost, timestamp, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            "#,
            event.user_id,
            format!("{:?}", event.resource_type),
            event.quantity,
            event.unit_cost,
            event.total_cost,
            event.timestamp,
            event.metadata,
        )
        .execute(&self.db)
        .await
        .map_err(|e| e.to_string())?;

        Ok(())
    }

    pub async fn get_usage_for_period(
        &self,
        user_id: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<UsageEvent>, String> {
        let rows = sqlx::query_as!(
            UsageEventRow,
            r#"
            SELECT user_id, resource_type, quantity, unit_cost, total_cost, timestamp, metadata
            FROM usage_events
            WHERE user_id = $1 AND timestamp >= $2 AND timestamp < $3
            ORDER BY timestamp DESC
            "#,
            user_id,
            start,
            end,
        )
        .fetch_all(&self.db)
        .await
        .map_err(|e| e.to_string())?;

        Ok(rows.into_iter().map(|r| r.into()).collect())
    }
}

#[derive(Debug)]
struct UsageEventRow {
    user_id: String,
    resource_type: String,
    quantity: Decimal,
    unit_cost: Decimal,
    total_cost: Decimal,
    timestamp: DateTime<Utc>,
    metadata: serde_json::Value,
}

impl From<UsageEventRow> for UsageEvent {
    fn from(row: UsageEventRow) -> Self {
        let resource_type = match row.resource_type.as_str() {
            "LLMTokens" => ResourceType::LLMTokens,
            "SandboxCPU" => ResourceType::SandboxCPU,
            "SandboxMemory" => ResourceType::SandboxMemory,
            "SandboxGPU" => ResourceType::SandboxGPU,
            "Storage" => ResourceType::Storage,
            "DataTransfer" => ResourceType::DataTransfer,
            _ => ResourceType::LLMTokens,
        };

        Self {
            user_id: row.user_id,
            resource_type,
            quantity: row.quantity,
            unit_cost: row.unit_cost,
            total_cost: row.total_cost,
            timestamp: row.timestamp,
            metadata: row.metadata,
        }
    }
}
```

---

## 8. Troubleshooting

### 8.1 Common Issues

**Issue: Budget exceeded mid-month**

```bash
# Symptoms:
Error: Budget exceeded: $1,050 / $1,000 monthly limit

# Solutions:
1. Increase monthly budget:
   neural-trader config set budget.monthly_limit 2000

2. Optimize model usage:
   - Use cheaper models for simple tasks
   - Enable prompt caching
   - Batch requests

3. Check for runaway agents:
   neural-trader status --verbose
```

**Issue: Federation coordination failures**

```bash
# Symptoms:
Error: Agent timeout in mesh federation

# Solutions:
1. Check network connectivity between agents
2. Increase timeout values in config
3. Switch to hierarchical topology (more fault-tolerant)
4. Enable health checks and auto-restart
```

---

## 9. References & Resources

### Federation Patterns
- **Microservices Patterns**: https://microservices.io/patterns/
- **Distributed Systems**: "Designing Data-Intensive Applications" by Martin Kleppmann

### Cost Optimization
- **OpenAI Pricing**: https://openai.com/pricing
- **Anthropic Pricing**: https://www.anthropic.com/pricing
- **AWS Cost Optimization**: https://aws.amazon.com/pricing/cost-optimization/

### Payment Systems
- **Stripe Webhooks**: https://stripe.com/docs/webhooks
- **Payment Security**: PCI DSS compliance guide

---

**Document Status**: ✅ Production-Ready
**Next Review**: 2026-02-12
**Contact**: finance@neural-trader.io
