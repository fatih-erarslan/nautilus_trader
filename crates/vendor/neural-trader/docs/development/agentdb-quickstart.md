# AgentDB Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies

```bash
npm install agentdb
```

### 2. Run Setup Script

```bash
node /workspaces/neural-trader/scripts/agentdb-setup.js
```

**Output**:
```
🚀 Initializing AgentDB Distributed Memory...
📦 Deployment: neural-trader-1763096012878
🔗 Topology: mesh
✅ VectorDB collection created: agent_states
✅ A3C RL Agent initialized for coordination optimization
🌐 Starting QUIC synchronization server...
✅ QUIC server listening on 0.0.0.0:8443
⚡ Sync interval: 5000ms (20x faster than WebSocket)
📊 Creating performance indexes...
  ✓ Index created: sandbox_id (hash)
  ✓ Index created: strategy_type (btree)
  ✓ Index created: timestamp (btree)
  ✓ Index created: coordination_score (btree)
  ✓ Index created: pnl_current (btree)
✨ AgentDB initialization complete!
```

### 3. Generate TLS Certificates (Production)

```bash
npx agentdb generate-cert --output /workspaces/neural-trader/certs
```

### 4. Test with Example Agent

```bash
node /workspaces/neural-trader/src/coordination/agentdb-client.js
```

## Integration Examples

### Basic Agent Integration

```javascript
const { AgentDBClient } = require('./src/coordination/agentdb-client');

// Initialize client
const client = new AgentDBClient({
  quicUrl: 'quic://localhost:8443',
  sandboxId: process.env.SANDBOX_ID,
  strategyType: 'momentum'
});

// Connect and start updates
await client.connect();
client.startStateUpdates(5000);

// Query coordination
const action = await client.getCoordinationAction();
console.log('Execute action:', action);

// Find collaborators
const similar = await client.findSimilarAgents(3);
console.log('Collaborate with:', similar);
```

### E2B Sandbox Integration

```javascript
const { Sandbox } = require('@e2b/sdk');
const { AgentDBClient } = require('./src/coordination/agentdb-client');

async function runTradingAgent() {
  // Create E2B sandbox
  const sandbox = await Sandbox.create({
    template: 'neural-trader',
    envVars: {
      AGENTDB_QUIC_URL: 'quic://agentdb.example.com:8443'
    }
  });

  // Initialize AgentDB client
  const client = new AgentDBClient({
    quicUrl: process.env.AGENTDB_QUIC_URL,
    sandboxId: sandbox.id,
    strategyType: 'momentum'
  });

  await client.connect();
  client.startStateUpdates(5000);

  // Trading loop
  while (true) {
    const action = await client.getCoordinationAction();

    // Execute trade in sandbox
    const result = await sandbox.process.start({
      cmd: 'python',
      args: ['trade.py', '--action', action.action]
    });

    // Report outcome
    await client.reportTradeOutcome(action, result.pnl);

    await new Promise(resolve => setTimeout(resolve, 60000));
  }
}
```

## Common Operations

### Store Agent State

```javascript
const state = {
  sandbox_id: 'sb_trader_1',
  cpu_usage: 45.2,
  memory_usage_mb: 1024,
  active_trades: 5,
  pnl_current: 1250.00,
  strategy_type: 'momentum',
  // ... other fields
};

await client.updateState(state);
```

### Query Similar Agents

```javascript
const similar = await client.findSimilarAgents(5);
/*
[
  { sandbox_id: 'sb_trader_2', similarity: 0.923 },
  { sandbox_id: 'sb_trader_4', similarity: 0.867 },
  { sandbox_id: 'sb_trader_3', similarity: 0.845 }
]
*/
```

### Get Coordination Action

```javascript
const action = await client.getCoordinationAction();
/*
{
  action: 'increase_position_sizes',
  confidence: 0.78,
  exploration: 0.15
}
*/
```

### Report Trade Outcome

```javascript
await client.reportTradeOutcome({
  action: 'buy',
  symbol: 'AAPL',
  quantity: 100
}, 350.25); // Profit/loss
```

## Environment Variables

```bash
# QUIC Server
AGENTDB_QUIC_URL=quic://localhost:8443
AGENTDB_QUIC_CERT=/workspaces/neural-trader/certs/server.crt
AGENTDB_QUIC_KEY=/workspaces/neural-trader/certs/server.key

# Agent Configuration
SANDBOX_ID=sb_trader_1
STRATEGY_TYPE=momentum

# Sync Settings
SYNC_INTERVAL_MS=5000
```

## Monitoring

### View Swarm Stats

```javascript
const stats = await memory.getSwarmStats();
console.log(stats);
```

### Test QUIC Connection

```bash
npx agentdb test-quic --server localhost:8443
```

### Monitor Real-time Updates

```bash
npx agentdb monitor --port 8443 --verbose
```

## Troubleshooting

### Connection Issues

```javascript
// Enable debug logging
process.env.DEBUG = 'agentdb:*';
await client.connect();
```

### Memory Issues

```javascript
// Check collection stats
const stats = await memory.db.getCollectionStats('agent_states');
console.log('Vectors:', stats.vectorCount);
console.log('Memory:', stats.memoryBytes / 1024 / 1024, 'MB');
```

### RL Training Issues

```javascript
// Check training progress
const rlStats = memory.rlAgent.getTrainingStats();
console.log('Episodes:', rlStats.episodes);
console.log('Avg Reward:', rlStats.avgReward);
```

## Production Checklist

- [ ] Generate production TLS certificates
- [ ] Configure firewall for port 8443
- [ ] Set up monitoring and alerting
- [ ] Enable authentication for QUIC connections
- [ ] Configure backup and disaster recovery
- [ ] Implement rate limiting
- [ ] Set up logging and metrics
- [ ] Test failover scenarios

## Next Steps

1. Read full [Architecture Documentation](./agentdb-architecture.md)
2. Integrate with your trading strategies
3. Configure production deployment
4. Set up monitoring dashboards
5. Train RL agent with historical data

## Support

- GitHub Issues: https://github.com/ruvnet/agentdb/issues
- Documentation: https://github.com/ruvnet/agentdb
- Neural Trader: /workspaces/neural-trader

---

**Ready to scale your trading swarm!** 🚀
