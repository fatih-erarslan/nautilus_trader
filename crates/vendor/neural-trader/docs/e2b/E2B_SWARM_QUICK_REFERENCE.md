# E2B Trading Swarm - Quick Reference

## Installation

```bash
npm install @neural-trader/mcp
```

## Import

```javascript
const { E2bSwarmToolHandler, registerE2bSwarmTools } = require('@neural-trader/mcp');
```

## Tool Summary

| Tool | Purpose | Cost | Latency | GPU |
|------|---------|------|---------|-----|
| `init_e2b_swarm` | Initialize swarm | High | Medium | Yes |
| `deploy_trading_agent` | Deploy agent | Medium | Medium | Yes |
| `get_swarm_status` | Get status | Low | Fast | No |
| `scale_swarm` | Scale agents | Medium | Medium | No |
| `execute_swarm_strategy` | Execute strategy | High | Slow | Yes |
| `monitor_swarm_health` | Monitor health | Low | Fast | No |
| `get_swarm_metrics` | Get metrics | Low | Fast | No |
| `shutdown_swarm` | Shutdown | Low | Medium | No |

## Quick Examples

### Initialize Swarm

```javascript
const swarm = await handler.initE2bSwarm({
  topology: 'mesh',        // mesh|hierarchical|ring|star
  maxAgents: 5,
  strategy: 'balanced',    // balanced|aggressive|conservative|adaptive
  sharedMemory: true,
  autoScale: false
});
// Returns: { swarm_id, topology, status, created_at }
```

### Deploy Agent

```javascript
const agent = await handler.deployTradingAgent({
  swarm_id: 'swarm_123',
  agent_type: 'market_maker',  // market_maker|trend_follower|arbitrage|risk_manager|coordinator
  symbols: ['AAPL', 'MSFT'],
  resources: {
    memory_mb: 512,
    cpu_count: 1,
    gpu_enabled: false
  }
});
// Returns: { agent_id, sandbox_id, status, deployed_at }
```

### Get Status

```javascript
const status = await handler.getSwarmStatus({
  swarm_id: 'swarm_123',
  include_metrics: true,
  include_agents: true
});
// Returns: { status, topology, agent_count, agents[], metrics{} }
```

### Scale Swarm

```javascript
const scaling = await handler.scaleSwarm({
  swarm_id: 'swarm_123',
  target_agents: 10,
  scale_mode: 'gradual',  // immediate|gradual|adaptive
  preserve_state: true
});
// Returns: { previous_agents, target_agents, status }
```

### Execute Strategy

```javascript
const execution = await handler.executeSwarmStrategy({
  swarm_id: 'swarm_123',
  strategy: 'arbitrage',
  parameters: { min_spread: 0.5 },
  coordination: 'parallel',  // parallel|sequential|adaptive
  timeout: 300
});
// Returns: { execution_id, status, total_trades, total_pnl }
```

### Monitor Health

```javascript
const health = await handler.monitorSwarmHealth({
  swarm_id: 'swarm_123',
  interval: 60,
  alerts: {
    failure_threshold: 0.2,
    latency_threshold: 1000,
    error_rate_threshold: 0.05
  },
  include_system_metrics: true
});
// Returns: { health_status, metrics{}, alerts[], agent_health[] }
```

### Get Metrics

```javascript
const metrics = await handler.getSwarmMetrics({
  swarm_id: 'swarm_123',
  time_range: '24h',  // 1h|6h|24h|7d|30d
  metrics: ['all'],   // latency|throughput|error_rate|success_rate|pnl|trades|all
  aggregation: 'avg'  // avg|min|max|sum|p50|p95|p99
});
// Returns: { metrics{}, per_agent_metrics[] }
```

### Shutdown Swarm

```javascript
const shutdown = await handler.shutdownSwarm({
  swarm_id: 'swarm_123',
  grace_period: 60,
  save_state: true,
  force: false
});
// Returns: { status, agents_stopped, final_metrics{} }
```

## Topology Options

- **mesh**: Peer-to-peer, high resilience, best for 5-20 agents
- **hierarchical**: Tree structure, coordinator-agent model, scales to 50+ agents
- **ring**: Circular communication, balanced load, good for 10-30 agents
- **star**: Centralized coordinator, efficient for 3-10 agents

## Agent Types

- **market_maker**: Provides liquidity, maintains spreads
- **trend_follower**: Identifies and follows trends
- **arbitrage**: Exploits price differences
- **risk_manager**: Monitors and manages risk
- **coordinator**: Orchestrates multi-agent strategies

## Strategy Options

- **balanced**: Balanced risk/reward
- **aggressive**: Higher risk/reward
- **conservative**: Lower risk/reward
- **adaptive**: Adjusts based on conditions

## Metrics Available

- **latency_ms**: Average latency
- **throughput_tps**: Transactions per second
- **error_rate**: Error percentage
- **success_rate**: Success percentage
- **total_pnl**: Total profit/loss
- **total_trades**: Trade count
- **avg_trade_size**: Average position size
- **win_rate**: Win percentage

## Error Handling

```javascript
try {
  const swarm = await handler.initE2bSwarm({ topology: 'mesh' });
} catch (error) {
  console.error('Swarm initialization failed:', error.message);
  // Handle error
}
```

## File Paths

- Tool Handler: `/workspaces/neural-trader/neural-trader-rust/packages/mcp/src/tools/e2b-swarm.js`
- Rust Implementation: `/workspaces/neural-trader/neural-trader-rust/crates/napi-bindings/src/e2b_monitoring_impl.rs`
- Schemas: `/workspaces/neural-trader/neural-trader-rust/packages/mcp/tools/*swarm*.json`

## MCP Protocol

All tools follow MCP 2025-11 specification:
- JSON Schema 2020-12 format
- Proper input/output validation
- ETag support for caching
- Category-based discovery

## Status Codes

- **active**: Swarm/agent is running
- **degraded**: Some agents failing
- **stopped**: Swarm/agent is stopped
- **error**: Critical failure
- **initializing**: Starting up
- **scaling**: Scaling in progress
- **executing**: Strategy running

## Next Steps

1. See `/workspaces/neural-trader/docs/E2B_SWARM_INTEGRATION_COMPLETE.md` for full documentation
2. Test tools: `node scripts/test-e2b-swarm.js`
3. Deploy to production: Enable real E2B API integration

---

**Version**: 2.1.1
**Last Updated**: 2025-11-14
**Total Tools**: 103 (8 E2B swarm + 95 other)
