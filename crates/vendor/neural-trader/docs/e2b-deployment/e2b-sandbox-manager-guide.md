# E2B Sandbox Manager Guide

## Overview

The E2B Sandbox Manager provides robust lifecycle management for trading agent deployment in isolated E2B cloud sandboxes. It includes connection pooling, health monitoring, auto-recovery, and resource optimization.

## Features

### Core Capabilities

1. **Full Lifecycle Control**
   - Create and configure sandboxes
   - Deploy trading agents
   - Execute trading strategies
   - Monitor health and performance
   - Automatic cleanup and termination

2. **Connection Pooling**
   - Reuse existing sandboxes
   - Automatic pool management
   - Configurable pool size limits
   - Efficient resource allocation

3. **Health Monitoring**
   - Continuous health checks
   - Performance metrics tracking
   - Error rate monitoring
   - Uptime tracking

4. **Auto-Recovery**
   - Automatic detection of unhealthy sandboxes
   - Self-healing with automatic replacement
   - Graceful degradation
   - Event-driven notifications

5. **Resource Optimization**
   - Automatic cleanup of inactive sandboxes
   - Memory and CPU tracking
   - Configurable timeout policies
   - Resource usage reporting

## Installation

```bash
# Ensure E2B API key is configured
export E2B_API_KEY=your_api_key_here

# Install dependencies (already included in neural-trader)
npm install
```

## Quick Start

### Basic Usage

```javascript
const { SandboxManager } = require('./src/e2b');

// Create manager instance
const manager = new SandboxManager();

// Create a trading sandbox
const sandbox = await manager.createTradingSandbox({
  template: 'base',
  timeout: 3600,
  memoryMb: 512,
  cpuCount: 1
});

console.log('Sandbox created:', sandbox.sandboxId);

// Deploy a trading agent
const deployment = await manager.deployAgent(
  sandbox.sandboxId,
  'momentum',
  ['AAPL', 'TSLA', 'GOOGL'],
  {
    period: 20,
    threshold: 0.02,
    useGpu: false
  }
);

console.log('Agent deployed:', deployment.agentType);

// Execute a trading strategy
const execution = await manager.executeStrategy(
  sandbox.sandboxId,
  'mean-reversion',
  {
    windowSize: 20,
    stddevs: 2,
    timeout: 300
  }
);

console.log('Strategy executed:', execution.strategy);

// Monitor health
const health = await manager.monitorHealth();
console.log('Health report:', health.health);

// Cleanup when done
await manager.shutdown();
```

## API Reference

### SandboxManager

#### Constructor

```javascript
const manager = new SandboxManager();
```

Creates a new SandboxManager instance with default configuration:
- `maxPoolSize`: 10 sandboxes
- `healthCheckInterval`: 30 seconds
- `autoRecovery`: true
- `retryAttempts`: 3
- `cleanupInterval`: 5 minutes

#### Methods

##### createTradingSandbox(config)

Creates a new trading sandbox.

**Parameters:**
- `config` (Object):
  - `template` (string): Sandbox template (default: 'base')
  - `timeout` (number): Timeout in seconds (default: 3600)
  - `memoryMb` (number): Memory allocation in MB (default: 512)
  - `cpuCount` (number): Number of CPU cores (default: 1)
  - `env` (Object): Environment variables

**Returns:** Promise<Object>
```javascript
{
  success: true,
  sandboxId: 'sb_xxx',
  sandbox: { /* sandbox details */ },
  message: 'Trading sandbox created successfully'
}
```

**Example:**
```javascript
const result = await manager.createTradingSandbox({
  template: 'node',
  timeout: 7200,
  memoryMb: 1024,
  cpuCount: 2
});
```

##### deployAgent(sandboxId, agentType, symbols, strategyParams)

Deploys a trading agent to a sandbox.

**Parameters:**
- `sandboxId` (string): Target sandbox ID
- `agentType` (string): Type of trading agent
- `symbols` (Array<string>): Trading symbols
- `strategyParams` (Object): Strategy configuration

**Returns:** Promise<Object>

**Example:**
```javascript
const result = await manager.deployAgent(
  'sb_xxx',
  'momentum',
  ['AAPL', 'TSLA'],
  { period: 20, threshold: 0.02 }
);
```

##### executeStrategy(sandboxId, strategy, params)

Executes a trading strategy in a sandbox.

**Parameters:**
- `sandboxId` (string): Target sandbox ID
- `strategy` (string): Strategy name
- `params` (Object): Strategy parameters

**Returns:** Promise<Object>

**Example:**
```javascript
const result = await manager.executeStrategy(
  'sb_xxx',
  'mean-reversion',
  { windowSize: 20, stddevs: 2, timeout: 300 }
);
```

##### monitorHealth()

Monitors health of all active sandboxes.

**Returns:** Promise<Object>
```javascript
{
  success: true,
  health: {
    timestamp: Date,
    totalSandboxes: 5,
    healthy: 4,
    degraded: 1,
    unhealthy: 0,
    sandboxes: { /* per-sandbox health data */ }
  }
}
```

##### scaleUp(targetCount)

Scales up the sandbox pool to target count.

**Parameters:**
- `targetCount` (number): Target number of sandboxes

**Returns:** Promise<Object>

**Example:**
```javascript
const result = await manager.scaleUp(10);
console.log(`Created ${result.created} new sandboxes`);
```

##### cleanup()

Cleans up inactive and unhealthy sandboxes.

**Returns:** Promise<Object>
```javascript
{
  success: true,
  cleanup: {
    timestamp: Date,
    checked: 10,
    terminated: 3,
    failed: 0,
    freed: { memory: 1536, cpu: 3 }
  }
}
```

##### getOrCreateSandbox(config)

Gets an available sandbox from pool or creates a new one.

**Parameters:**
- `config` (Object): Sandbox configuration (optional)

**Returns:** Promise<Object>
```javascript
{
  success: true,
  sandboxId: 'sb_xxx',
  sandbox: { /* sandbox details */ },
  reused: true, // true if from pool, false if newly created
  message: 'Sandbox retrieved from pool'
}
```

##### getSandboxStatus(sandboxId)

Gets detailed status of a sandbox.

**Parameters:**
- `sandboxId` (string): Sandbox ID

**Returns:** Promise<Object>

##### listSandboxes()

Lists all active sandboxes.

**Returns:** Object
```javascript
{
  success: true,
  count: 5,
  sandboxes: [
    {
      id: 'sb_xxx',
      name: 'trading-sandbox-xxx',
      template: 'base',
      status: 'active',
      created: Date,
      lastUsed: Date,
      agentCount: 2,
      executionCount: 15,
      health: { /* health data */ }
    }
  ],
  stats: { /* manager statistics */ }
}
```

##### getStats()

Gets manager statistics.

**Returns:** Object
```javascript
{
  success: true,
  stats: {
    sandboxesCreated: 25,
    sandboxesDestroyed: 10,
    agentsDeployed: 50,
    strategiesExecuted: 200,
    healthChecks: 100,
    recoveryAttempts: 2,
    errors: 5,
    activeSandboxes: 5,
    poolSize: 8,
    healthyCount: 4
  },
  config: { /* manager configuration */ }
}
```

##### shutdown()

Shuts down the manager and cleans up all resources.

**Returns:** Promise<Object>

**Example:**
```javascript
await manager.shutdown();
console.log('Manager shut down successfully');
```

### Events

The SandboxManager extends EventEmitter and emits the following events:

#### sandbox:created
Emitted when a new sandbox is created.
```javascript
manager.on('sandbox:created', (sandbox) => {
  console.log('Sandbox created:', sandbox.id);
});
```

#### agent:deployed
Emitted when an agent is deployed to a sandbox.
```javascript
manager.on('agent:deployed', ({ sandboxId, agentType, symbols }) => {
  console.log(`${agentType} agent deployed to ${sandboxId}`);
});
```

#### strategy:executed
Emitted when a strategy is executed.
```javascript
manager.on('strategy:executed', ({ sandboxId, strategy, params }) => {
  console.log(`Strategy ${strategy} executed in ${sandboxId}`);
});
```

#### health:checked
Emitted after health monitoring completes.
```javascript
manager.on('health:checked', (healthReport) => {
  console.log(`Health check: ${healthReport.healthy} healthy sandboxes`);
});
```

#### scale:up
Emitted when scaling up completes.
```javascript
manager.on('scale:up', ({ targetCount, created }) => {
  console.log(`Scaled up: ${created} sandboxes created`);
});
```

#### cleanup:complete
Emitted when cleanup completes.
```javascript
manager.on('cleanup:complete', (cleanupResults) => {
  console.log(`Cleanup: ${cleanupResults.terminated} sandboxes terminated`);
});
```

#### recovery:success
Emitted when auto-recovery succeeds.
```javascript
manager.on('recovery:success', ({ oldId, newId }) => {
  console.log(`Recovered: ${oldId} -> ${newId}`);
});
```

#### recovery:failed
Emitted when auto-recovery fails.
```javascript
manager.on('recovery:failed', ({ sandboxId, error }) => {
  console.error(`Recovery failed for ${sandboxId}:`, error);
});
```

#### error
Emitted when an error occurs.
```javascript
manager.on('error', ({ type, error }) => {
  console.error(`Error (${type}):`, error);
});
```

#### shutdown
Emitted when manager shuts down.
```javascript
manager.on('shutdown', () => {
  console.log('Manager shut down');
});
```

## Advanced Usage

### Custom Configuration

```javascript
const manager = new SandboxManager();

// Override configuration
manager.config = {
  ...manager.config,
  maxPoolSize: 20,
  healthCheckInterval: 15000, // 15 seconds
  cleanupInterval: 600000, // 10 minutes
  autoRecovery: true,
  retryAttempts: 5,
  retryDelay: 2000
};
```

### Event-Driven Architecture

```javascript
const manager = new SandboxManager();

// Set up event handlers
manager.on('sandbox:created', handleSandboxCreated);
manager.on('agent:deployed', handleAgentDeployed);
manager.on('health:checked', handleHealthCheck);
manager.on('error', handleError);

function handleSandboxCreated(sandbox) {
  // Store sandbox ID in database
  // Send notification
  // Update monitoring dashboard
}

function handleAgentDeployed({ sandboxId, agentType }) {
  // Log deployment
  // Update agent registry
  // Trigger initial strategy execution
}

function handleHealthCheck(report) {
  // Update monitoring dashboard
  // Send alerts for unhealthy sandboxes
  // Trigger manual intervention if needed
}

function handleError({ type, error }) {
  // Log error
  // Send alert
  // Trigger incident response
}
```

### Pool Management Strategy

```javascript
// Pre-create sandbox pool
async function initializePool(manager, count) {
  console.log(`Initializing pool with ${count} sandboxes...`);

  const results = await manager.scaleUp(count);

  console.log(`Pool initialized: ${results.created} sandboxes ready`);
  return results;
}

// Use pool for agent deployment
async function deployWithPooling(manager, agentType, symbols, params) {
  // Get sandbox from pool
  const sandboxResult = await manager.getOrCreateSandbox();

  if (!sandboxResult.success) {
    throw new Error('Failed to get sandbox from pool');
  }

  const { sandboxId, reused } = sandboxResult;

  console.log(`Using ${reused ? 'pooled' : 'new'} sandbox: ${sandboxId}`);

  // Deploy agent
  return await manager.deployAgent(sandboxId, agentType, symbols, params);
}

// Example usage
const manager = new SandboxManager();
await initializePool(manager, 5);
await deployWithPooling(manager, 'momentum', ['AAPL'], {});
```

### Health Monitoring Dashboard

```javascript
// Periodic health reporting
setInterval(async () => {
  const health = await manager.monitorHealth();

  if (!health.success) {
    console.error('Health monitoring failed');
    return;
  }

  const { totalSandboxes, healthy, degraded, unhealthy } = health.health;

  console.log(`
=== Sandbox Health Report ===
Total: ${totalSandboxes}
Healthy: ${healthy}
Degraded: ${degraded}
Unhealthy: ${unhealthy}
===========================
  `);

  // Alert on unhealthy sandboxes
  if (unhealthy > 0) {
    console.warn(`⚠️  ${unhealthy} unhealthy sandboxes detected!`);
  }
}, 60000); // Every minute
```

### Resource Optimization

```javascript
// Aggressive cleanup strategy
manager.config.cleanupInterval = 180000; // 3 minutes

// Monitor resource usage
setInterval(() => {
  const stats = manager.getStats();

  const { activeSandboxes, poolSize } = stats.stats;

  if (poolSize > manager.config.maxPoolSize * 0.8) {
    console.warn('⚠️  Pool utilization high, triggering cleanup');
    manager.cleanup();
  }
}, 60000);
```

## Best Practices

1. **Always Configure E2B_API_KEY**
   ```bash
   export E2B_API_KEY=your_api_key_here
   ```

2. **Handle Errors Gracefully**
   ```javascript
   try {
     const result = await manager.createTradingSandbox();
     if (!result.success) {
       console.error('Failed:', result.error);
       // Implement fallback logic
     }
   } catch (error) {
     console.error('Exception:', error);
     // Implement recovery logic
   }
   ```

3. **Use Connection Pooling**
   ```javascript
   // Reuse sandboxes for multiple agents
   const sandbox = await manager.getOrCreateSandbox();
   await manager.deployAgent(sandbox.sandboxId, 'agent1', ['AAPL'], {});
   await manager.deployAgent(sandbox.sandboxId, 'agent2', ['TSLA'], {});
   ```

4. **Monitor Health Regularly**
   ```javascript
   // Set up periodic health checks
   setInterval(() => manager.monitorHealth(), 30000);
   ```

5. **Clean Up Resources**
   ```javascript
   // Always shutdown when done
   process.on('SIGINT', async () => {
     await manager.shutdown();
     process.exit(0);
   });
   ```

6. **Listen to Events**
   ```javascript
   // Set up event handlers for monitoring
   manager.on('error', (error) => {
     // Log to monitoring system
     // Send alerts
   });

   manager.on('recovery:success', (data) => {
     // Log successful recovery
   });
   ```

## Troubleshooting

### E2B_API_KEY Not Set
```
Error: E2B_API_KEY not configured
```
**Solution:** Set the environment variable:
```bash
export E2B_API_KEY=your_api_key_here
```

### NAPI Bindings Not Found
```
Error: Failed to load NAPI bindings
```
**Solution:** Ensure NAPI bindings are built:
```bash
cd neural-trader-rust/crates/napi-bindings
npm run build
```

### Sandbox Creation Fails
```
Error: Failed to create trading sandbox
```
**Solution:**
1. Check E2B_API_KEY is valid
2. Check network connectivity
3. Check E2B service status
4. Review error logs for details

### Health Check Timeouts
**Solution:**
1. Increase health check interval
2. Reduce number of active sandboxes
3. Check sandbox responsiveness

## Performance Tuning

### Configuration Recommendations

**For High-Throughput Trading:**
```javascript
manager.config = {
  maxPoolSize: 20,
  healthCheckInterval: 60000, // 1 minute
  cleanupInterval: 600000, // 10 minutes
  autoRecovery: true
};
```

**For Development/Testing:**
```javascript
manager.config = {
  maxPoolSize: 5,
  healthCheckInterval: 30000, // 30 seconds
  cleanupInterval: 300000, // 5 minutes
  autoRecovery: true
};
```

**For Resource-Constrained Environments:**
```javascript
manager.config = {
  maxPoolSize: 3,
  healthCheckInterval: 120000, // 2 minutes
  cleanupInterval: 180000, // 3 minutes
  autoRecovery: false // Manual recovery
};
```

## Integration Examples

### Express.js API

```javascript
const express = require('express');
const { SandboxManager } = require('./src/e2b');

const app = express();
const manager = new SandboxManager();

app.post('/sandbox/create', async (req, res) => {
  const result = await manager.createTradingSandbox(req.body);
  res.json(result);
});

app.post('/agent/deploy', async (req, res) => {
  const { sandboxId, agentType, symbols, params } = req.body;
  const result = await manager.deployAgent(sandboxId, agentType, symbols, params);
  res.json(result);
});

app.get('/health', async (req, res) => {
  const result = await manager.monitorHealth();
  res.json(result);
});

app.listen(3000, () => {
  console.log('API server running on port 3000');
});
```

## License

MIT License - See LICENSE file for details.
