# E2B Trading Strategy Deployment Guide

## Overview

This guide covers deployment of five autonomous trading strategies to E2B sandboxes for isolated execution with comprehensive risk management.

## Available Strategies

### 1. Momentum Trading (`/e2b-strategies/momentum`)
- **Symbols**: SPY, QQQ, IWM
- **Logic**: Buy when 5-minute momentum > 2%, sell when < -2%
- **Position Size**: Fixed 10 shares per trade
- **Port**: 3000

### 2. Neural Forecasting (`/e2b-strategies/neural-forecast`)
- **Symbols**: AAPL, TSLA, NVDA
- **Logic**: LSTM price prediction with confidence-based sizing
- **Position Size**: Dynamic 5-50 shares based on confidence
- **Port**: 3001

### 3. Mean Reversion (`/e2b-strategies/mean-reversion`)
- **Symbols**: GLD, SLV, TLT
- **Logic**: Z-score based entry/exit (buy at -2σ, sell at +2σ)
- **Position Size**: Max 100 shares per symbol
- **Port**: 3002

### 4. Risk Manager (`/e2b-strategies/risk-manager`)
- **Function**: Portfolio risk monitoring and enforcement
- **Metrics**: VaR, CVaR, drawdown, Sharpe ratio
- **Actions**: Auto stop-loss at 2% per trade
- **Port**: 3003

### 5. Portfolio Optimizer (`/e2b-strategies/portfolio-optimizer`)
- **Symbols**: SPY, QQQ, IWM, GLD, TLT, AAPL, TSLA
- **Methods**: Sharpe optimization, risk parity
- **Rebalance**: Triggered on 5% deviation
- **Port**: 3004

## Prerequisites

### 1. Environment Variables

All strategies require Alpaca API credentials:

```bash
export ALPACA_API_KEY="your_alpaca_key"
export ALPACA_SECRET_KEY="your_alpaca_secret"
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"  # Optional
```

### 2. Node.js Requirements

- Node.js >= 18.0.0
- npm or yarn

## Deployment Options

### Option 1: Using Neural Trader MCP Tools

```javascript
// Create E2B sandbox for momentum strategy
mcp__neural-trader__create_e2b_sandbox({
  name: "momentum-strategy",
  template: "node",
  env_vars: {
    ALPACA_API_KEY: process.env.ALPACA_API_KEY,
    ALPACA_SECRET_KEY: process.env.ALPACA_SECRET_KEY,
    PORT: "3000"
  },
  install_packages: ["@alpacahq/alpaca-trade-api", "express"]
})

// Upload strategy code
mcp__neural-trader__sandbox_upload({
  sandbox_id: "momentum-strategy",
  file_path: "/app/index.js",
  content: fs.readFileSync("e2b-strategies/momentum/index.js", "utf-8")
})

// Upload package.json
mcp__neural-trader__sandbox_upload({
  sandbox_id: "momentum-strategy",
  file_path: "/app/package.json",
  content: fs.readFileSync("e2b-strategies/momentum/package.json", "utf-8")
})

// Execute strategy
mcp__neural-trader__sandbox_execute({
  sandbox_id: "momentum-strategy",
  code: "npm install && npm start",
  capture_output: true
})
```

### Option 2: Using Flow-Nexus Platform

```javascript
// Create sandbox with neural trader configuration
mcp__flow-nexus__sandbox_create({
  template: "node",
  name: "neural-forecast-strategy",
  env_vars: {
    ALPACA_API_KEY: process.env.ALPACA_API_KEY,
    ALPACA_SECRET_KEY: process.env.ALPACA_SECRET_KEY,
    PORT: "3001"
  },
  install_packages: [
    "@alpacahq/alpaca-trade-api",
    "@tensorflow/tfjs-node",
    "express"
  ]
})

// Configure and start
mcp__flow-nexus__sandbox_configure({
  sandbox_id: "neural-forecast-strategy",
  run_commands: [
    "cd /app",
    "npm install",
    "npm start"
  ]
})
```

### Option 3: Manual E2B Deployment

```bash
# Install E2B CLI
npm install -g @e2b/cli

# Create sandbox
e2b sandbox create --template node --name momentum-strategy

# Upload files
e2b sandbox upload momentum-strategy ./e2b-strategies/momentum/

# Execute in sandbox
e2b sandbox exec momentum-strategy "cd /app && npm install && npm start"
```

## Multi-Strategy Orchestration

Deploy all strategies together for comprehensive trading:

```javascript
// Deploy all 5 strategies in parallel
const strategies = [
  { name: "momentum", port: 3000 },
  { name: "neural-forecast", port: 3001 },
  { name: "mean-reversion", port: 3002 },
  { name: "risk-manager", port: 3003 },
  { name: "portfolio-optimizer", port: 3004 }
];

for (const strategy of strategies) {
  await mcp__neural-trader__create_e2b_sandbox({
    name: `${strategy.name}-strategy`,
    template: "node",
    env_vars: {
      ALPACA_API_KEY: process.env.ALPACA_API_KEY,
      ALPACA_SECRET_KEY: process.env.ALPACA_SECRET_KEY,
      PORT: strategy.port.toString()
    }
  });

  // Upload code and start
  await deployStrategy(strategy.name);
}
```

## API Endpoints

Each strategy exposes these endpoints:

### Health Check
```bash
GET /health
Response: { status: "healthy", strategy: "...", timestamp: "..." }
```

### Status
```bash
GET /status
Response: {
  account: { equity, cash, buyingPower },
  positions: [...],
  ...strategy-specific data
}
```

### Manual Execution
```bash
POST /execute
Response: { success: true, message: "Strategy executed" }
```

### Strategy-Specific Endpoints

**Neural Forecast**:
```bash
POST /retrain/:symbol
# Retrain LSTM model for specific symbol
```

**Risk Manager**:
```bash
GET /metrics
# Get current risk metrics (VaR, CVaR, drawdown)

GET /alerts
# Get active risk alerts
```

**Portfolio Optimizer**:
```bash
POST /optimize
# Run portfolio optimization

POST /rebalance
# Execute rebalancing trades
```

**Mean Reversion**:
```bash
GET /statistics/:symbol
# Get current z-score and statistics
```

## Monitoring and Logs

All strategies use structured JSON logging:

```json
{
  "level": "INFO|ERROR|TRADE|RISK|OPTIMIZE",
  "msg": "message",
  "timestamp": "ISO-8601",
  ...additional data
}
```

### Log Levels

- **INFO**: General information
- **ERROR**: Errors and exceptions
- **TRADE**: Trade execution
- **RISK**: Risk metrics and alerts
- **OPTIMIZE**: Optimization results
- **SIGNAL**: Trading signals

### Retrieving Logs

```javascript
// Using MCP
const logs = await mcp__neural-trader__sandbox_logs({
  sandbox_id: "momentum-strategy",
  lines: 100
});

// Using Flow-Nexus
const logs = await mcp__flow-nexus__sandbox_logs({
  sandbox_id: "neural-forecast-strategy",
  lines: 100
});
```

## Risk Management Integration

The Risk Manager should run alongside all trading strategies:

```javascript
// Monitor all strategies
setInterval(async () => {
  const metrics = await fetch("http://risk-manager:3003/metrics");
  const data = await metrics.json();

  if (data.alerts.length > 0) {
    console.error("Risk alerts:", data.alerts);
    // Take corrective action
  }
}, 60000); // Check every minute
```

## Scaling Deployments

### Using Flow-Nexus Scaling

```javascript
mcp__flow-nexus__scale_e2b_deployment({
  deployment_id: "momentum-strategy",
  instance_count: 3,  // Scale to 3 instances
  auto_scale: true    // Enable auto-scaling
})
```

### Load Balancing

Distribute symbols across multiple instances:

```javascript
// Instance 1: SPY only
env_vars: { SYMBOLS: "SPY" }

// Instance 2: QQQ only
env_vars: { SYMBOLS: "QQQ" }

// Instance 3: IWM only
env_vars: { SYMBOLS: "IWM" }
```

## Performance Optimization

### 1. Memory Management

For neural forecasting strategy:
```javascript
// Add to environment
env_vars: {
  NODE_OPTIONS: "--max-old-space-size=2048"
}
```

### 2. Execution Intervals

Adjust strategy execution frequency:
```javascript
// In index.js, modify:
setInterval(runStrategy, 5 * 60 * 1000);  // 5 minutes
// to:
setInterval(runStrategy, 1 * 60 * 1000);  // 1 minute
```

### 3. Concurrent Execution

Strategies are already optimized for concurrent symbol processing using `Promise.all()`.

## Troubleshooting

### Common Issues

**1. API Rate Limits**
```javascript
// Add exponential backoff
async function executeWithRetry(fn, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (error.message.includes('rate limit')) {
        await new Promise(resolve => setTimeout(resolve, 2 ** i * 1000));
        continue;
      }
      throw error;
    }
  }
}
```

**2. Market Closed**
All strategies check market hours automatically. During closed hours, they log and skip execution.

**3. Insufficient Data**
Strategies require historical data. First execution may fail - this is normal. Wait for next interval.

**4. Memory Leaks (TensorFlow)**
Neural forecast strategy properly disposes tensors. Monitor with:
```javascript
console.log(tf.memory());
```

## Security Best Practices

1. **Never commit API keys** - Use environment variables
2. **Use paper trading** initially - Set `ALPACA_BASE_URL` to paper API
3. **Monitor sandbox resources** - Check CPU/memory usage
4. **Implement circuit breakers** - Auto-shutdown on excessive losses
5. **Log all trades** - Maintain audit trail

## Cost Management

### E2B Sandbox Costs

- **Compute**: ~$0.10/hour per sandbox
- **Total for 5 strategies**: ~$0.50/hour or ~$360/month
- **Optimization**: Use Flow-Nexus auto-scaling to reduce costs during off-hours

### Alpaca Costs

- Paper trading: Free
- Live trading: Commission-free stocks

## Next Steps

1. **Test individually**: Deploy and test each strategy separately
2. **Integrate risk manager**: Ensure risk limits are enforced
3. **Run portfolio optimizer**: Establish optimal allocations
4. **Monitor performance**: Track metrics and logs
5. **Scale gradually**: Start with small positions, increase as confidence grows

## Support

For issues or questions:
- Neural Trader Documentation: `/docs`
- E2B Documentation: https://e2b.dev/docs
- Alpaca API Docs: https://alpaca.markets/docs
- Flow-Nexus Platform: https://flow-nexus.ruv.io
