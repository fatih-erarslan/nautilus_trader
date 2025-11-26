# E2B Deployment Patterns Test Suite

Comprehensive test suite for validating 8 production E2B swarm deployment patterns.

## Quick Start

### Prerequisites

```bash
# Install dependencies
npm install

# Set E2B API key (optional - runs in mock mode without it)
export E2B_API_KEY="your-e2b-api-key"
```

### Running Tests

```bash
# Run all tests (mock mode if no API key)
npm test

# Run with live E2B API
npm run test:live

# Run specific pattern
npm run test:pattern "Mesh Topology"

# Watch mode for development
npm run test:watch

# Generate coverage report
npm run test:coverage
```

## Test Patterns

### 1. Mesh Topology (Peer-to-Peer)
- 5 momentum traders with equal coordination
- Consensus trading with 3 agents
- Failover and redundancy

### 2. Hierarchical Topology (Leader-Worker)
- 1 coordinator + 4 workers
- Multi-strategy coordination
- Load balancing across workers

### 3. Ring Topology (Sequential Processing)
- Pipeline processing with 4 agents
- Data flow optimization
- Circuit breaker on failure

### 4. Star Topology (Centralized Hub)
- Central hub with 6 specialized agents
- Hub failover recovery

### 5. Auto-Scaling Deployment
- Start with 2, scale to 10 based on load
- Scale down during low activity
- VIX-based scaling (volatility-driven)

### 6. Multi-Strategy Deployment
- 2 momentum + 2 pairs + 1 arbitrage
- Strategy rotation based on performance

### 7. Blue-Green Deployment
- Deploy new swarm, gradual traffic shift
- Rollback on error rate spike

### 8. Canary Deployment
- Deploy 1 new agent, monitor, then full rollout

## Documentation

- **Detailed Results:** See `/workspaces/neural-trader/docs/e2b/DEPLOYMENT_PATTERNS_RESULTS.md`
- **Test Code:** See `deployment-patterns.test.js`
- **E2B Modules:** See `/workspaces/neural-trader/src/e2b/`

## License

MIT License
