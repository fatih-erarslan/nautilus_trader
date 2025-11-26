# E2B Trading Swarm CLI Guide

Comprehensive command-line interface for managing E2B trading swarms with sandbox orchestration, agent deployment, and real-time monitoring.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Commands](#commands)
- [Examples](#examples)
- [Configuration](#configuration)
- [Best Practices](#best-practices)

## Installation

### Prerequisites

```bash
# Required environment variables in .env
E2B_API_KEY=your-e2b-api-key
E2B_ACCESS_TOKEN=your-e2b-access-token
ALPACA_API_KEY=your-alpaca-key
ALPACA_SECRET_KEY=your-alpaca-secret
ANTHROPIC_API_KEY=your-anthropic-key
```

### Install Dependencies

```bash
cd scripts
npm install
```

### Make CLI Executable

```bash
chmod +x e2b-swarm-cli.js
# Optionally, add to PATH or use npm link
```

## Quick Start

### 1. Create Sandboxes

```bash
# Create a single sandbox
node e2b-swarm-cli.js create --template trading-bot --name my-sandbox

# Create multiple sandboxes
node e2b-swarm-cli.js create --template base --count 3 --name swarm
```

### 2. Deploy Trading Agent

```bash
# Deploy momentum trading agent
node e2b-swarm-cli.js deploy --agent momentum --symbols AAPL,MSFT,GOOGL

# Deploy pairs trading agent
node e2b-swarm-cli.js deploy --agent pairs --symbols AAPL,MSFT
```

### 3. Monitor Swarm

```bash
# Real-time monitoring (updates every 5 seconds)
node e2b-swarm-cli.js monitor --interval 5s

# Monitor for specific duration
node e2b-swarm-cli.js monitor --interval 10s --duration 5m
```

## Commands

### Sandbox Management

#### `create` - Create E2B Sandboxes

Create one or more E2B sandboxes for isolated agent execution.

```bash
node e2b-swarm-cli.js create [options]

Options:
  -t, --template <template>  Sandbox template (default: "base")
  -c, --count <count>        Number of sandboxes (default: "1")
  -n, --name <name>          Sandbox name prefix
  -j, --json                 Output in JSON format
```

**Examples:**

```bash
# Create single sandbox
node e2b-swarm-cli.js create

# Create 5 sandboxes with custom template
node e2b-swarm-cli.js create --template trading-bot --count 5 --name trader

# Create and output as JSON
node e2b-swarm-cli.js create --count 3 --json
```

#### `list` - List All Sandboxes

Display all sandboxes with their current status.

```bash
node e2b-swarm-cli.js list [options]

Options:
  -s, --status <status>  Filter by status (running, stopped, failed, pending)
  -j, --json            Output in JSON format
```

**Examples:**

```bash
# List all sandboxes
node e2b-swarm-cli.js list

# List only running sandboxes
node e2b-swarm-cli.js list --status running

# Get JSON output
node e2b-swarm-cli.js list --json
```

#### `status` - Get Sandbox Status

Get detailed information about a specific sandbox.

```bash
node e2b-swarm-cli.js status <sandbox-id> [options]

Options:
  -j, --json  Output in JSON format
```

**Examples:**

```bash
# Get sandbox status
node e2b-swarm-cli.js status sb-1234567890

# Get JSON output
node e2b-swarm-cli.js status sb-1234567890 --json
```

#### `destroy` - Destroy Sandbox

Terminate and remove a sandbox.

```bash
node e2b-swarm-cli.js destroy <sandbox-id> [options]

Options:
  -f, --force  Skip confirmation
  -j, --json   Output in JSON format
```

**Examples:**

```bash
# Destroy with confirmation
node e2b-swarm-cli.js destroy sb-1234567890

# Force destroy without confirmation
node e2b-swarm-cli.js destroy sb-1234567890 --force
```

### Agent Deployment

#### `deploy` - Deploy Trading Agent

Deploy a trading agent with specified strategy and symbols.

```bash
node e2b-swarm-cli.js deploy [options]

Options:
  -a, --agent <type>       Agent type (required)
  -s, --symbols <symbols>  Comma-separated symbol list (default: "SPY")
  --sandbox <id>           Target sandbox ID
  -j, --json              Output in JSON format
```

**Agent Types:**
- `momentum` - Momentum Trading
- `pairs` - Pairs Trading
- `neural` - Neural Forecasting
- `mean_reversion` - Mean Reversion
- `arbitrage` - Statistical Arbitrage

**Examples:**

```bash
# Deploy momentum trader for tech stocks
node e2b-swarm-cli.js deploy --agent momentum --symbols AAPL,MSFT,GOOGL,NVDA

# Deploy pairs trader to specific sandbox
node e2b-swarm-cli.js deploy --agent pairs --symbols AAPL,MSFT --sandbox sb-1234567890

# Deploy neural forecaster
node e2b-swarm-cli.js deploy --agent neural --symbols TSLA,AAPL,NVDA --json
```

#### `agents` - List Deployed Agents

Display all deployed agents and their status.

```bash
node e2b-swarm-cli.js agents [options]

Options:
  -j, --json  Output in JSON format
```

**Examples:**

```bash
# List all agents
node e2b-swarm-cli.js agents

# Get JSON output
node e2b-swarm-cli.js agents --json
```

### Swarm Operations

#### `scale` - Scale Swarm Size

Scale the swarm up or down to a target sandbox count.

```bash
node e2b-swarm-cli.js scale [options]

Options:
  -c, --count <count>  Target sandbox count (required)
  -j, --json          Output in JSON format
```

**Examples:**

```bash
# Scale up to 10 sandboxes
node e2b-swarm-cli.js scale --count 10

# Scale down to 3 sandboxes
node e2b-swarm-cli.js scale --count 3

# Scale and get JSON output
node e2b-swarm-cli.js scale --count 5 --json
```

#### `monitor` - Real-Time Monitoring

Monitor swarm status with automatic updates.

```bash
node e2b-swarm-cli.js monitor [options]

Options:
  -i, --interval <interval>  Update interval (default: "5s")
  -d, --duration <duration>  Monitoring duration (optional)
  -j, --json                Output in JSON format
```

**Interval Formats:**
- `5s` - 5 seconds
- `1m` - 1 minute
- `500ms` - 500 milliseconds
- `1h` - 1 hour

**Examples:**

```bash
# Monitor with default 5-second interval
node e2b-swarm-cli.js monitor

# Monitor every 10 seconds for 5 minutes
node e2b-swarm-cli.js monitor --interval 10s --duration 5m

# Monitor with fast updates
node e2b-swarm-cli.js monitor --interval 1s

# JSON stream output
node e2b-swarm-cli.js monitor --interval 5s --json
```

#### `health` - Health Check

Check overall swarm health and resource utilization.

```bash
node e2b-swarm-cli.js health [options]

Options:
  --detailed  Show detailed health information
  -j, --json  Output in JSON format
```

**Examples:**

```bash
# Basic health check
node e2b-swarm-cli.js health

# Detailed health check with sandbox list
node e2b-swarm-cli.js health --detailed

# Get JSON output
node e2b-swarm-cli.js health --json
```

### Strategy Execution

#### `execute` - Execute Strategy

Start strategy execution on deployed agents.

```bash
node e2b-swarm-cli.js execute [options]

Options:
  -s, --strategy <strategy>  Strategy type (required)
  --symbols <symbols>        Comma-separated symbol list (default: "SPY")
  --sandbox <id>            Target sandbox ID
  -j, --json                Output in JSON format
```

**Examples:**

```bash
# Execute momentum strategy
node e2b-swarm-cli.js execute --strategy momentum --symbols AAPL,MSFT

# Execute on specific sandbox
node e2b-swarm-cli.js execute --strategy pairs --symbols AAPL,MSFT --sandbox sb-1234567890

# Execute and get JSON output
node e2b-swarm-cli.js execute --strategy neural --symbols NVDA --json
```

#### `backtest` - Run Backtest

Run historical backtest for a strategy.

```bash
node e2b-swarm-cli.js backtest [options]

Options:
  -s, --strategy <strategy>  Strategy type (required)
  --start <date>            Start date YYYY-MM-DD (required)
  --end <date>              End date YYYY-MM-DD (default: today)
  --symbols <symbols>        Comma-separated symbol list (default: "SPY")
  -j, --json                Output in JSON format
```

**Examples:**

```bash
# Backtest momentum strategy for 2024
node e2b-swarm-cli.js backtest --strategy momentum --start 2024-01-01 --end 2024-12-31

# Backtest pairs trading with multiple symbols
node e2b-swarm-cli.js backtest --strategy pairs --start 2024-01-01 --symbols AAPL,MSFT,GOOGL

# Backtest and get JSON output
node e2b-swarm-cli.js backtest --strategy neural --start 2024-06-01 --symbols NVDA --json
```

## Examples

### Complete Workflow

```bash
# 1. Create sandboxes for swarm
node e2b-swarm-cli.js create --template trading-bot --count 3 --name swarm

# 2. List created sandboxes
node e2b-swarm-cli.js list

# 3. Deploy different agents
node e2b-swarm-cli.js deploy --agent momentum --symbols AAPL,MSFT,GOOGL
node e2b-swarm-cli.js deploy --agent pairs --symbols AAPL,MSFT
node e2b-swarm-cli.js deploy --agent neural --symbols NVDA,TSLA

# 4. Check swarm health
node e2b-swarm-cli.js health --detailed

# 5. Monitor in real-time
node e2b-swarm-cli.js monitor --interval 5s

# 6. Scale up if needed
node e2b-swarm-cli.js scale --count 5

# 7. Execute strategies
node e2b-swarm-cli.js execute --strategy momentum --symbols AAPL,MSFT,GOOGL
```

### Production Deployment

```bash
#!/bin/bash
# production-deploy.sh

# Create production swarm
node e2b-swarm-cli.js create --template trading-bot --count 5 --name production --json > /tmp/sandboxes.json

# Deploy multiple strategies in parallel
node e2b-swarm-cli.js deploy --agent momentum --symbols AAPL,MSFT,GOOGL &
node e2b-swarm-cli.js deploy --agent pairs --symbols AAPL,MSFT &
node e2b-swarm-cli.js deploy --agent neural --symbols NVDA,TSLA &
node e2b-swarm-cli.js deploy --agent mean_reversion --symbols SPY,QQQ &
wait

# Start monitoring in background
node e2b-swarm-cli.js monitor --interval 10s --json >> /var/log/swarm-monitor.log &

# Health check every 5 minutes
while true; do
  node e2b-swarm-cli.js health --json >> /var/log/health-check.log
  sleep 300
done
```

### Automated Backtesting

```bash
#!/bin/bash
# backtest-strategies.sh

STRATEGIES=("momentum" "pairs" "neural" "mean_reversion")
SYMBOLS="AAPL,MSFT,GOOGL,NVDA,TSLA"
START="2024-01-01"
END="2024-12-31"

for strategy in "${STRATEGIES[@]}"; do
  echo "Backtesting $strategy..."
  node e2b-swarm-cli.js backtest \
    --strategy "$strategy" \
    --start "$START" \
    --end "$END" \
    --symbols "$SYMBOLS" \
    --json > "results/${strategy}_backtest.json"
done

echo "All backtests complete!"
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# E2B Configuration
E2B_API_KEY=your-e2b-api-key
E2B_ACCESS_TOKEN=your-e2b-access-token

# Trading API
ALPACA_API_KEY=your-alpaca-api-key
ALPACA_SECRET_KEY=your-alpaca-secret-key
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2

# AI Services
ANTHROPIC_API_KEY=your-anthropic-key
```

### CLI State

The CLI maintains state in `.swarm/cli-state.json`:

```json
{
  "sandboxes": [...],
  "agents": [...],
  "deployments": [...],
  "lastUpdate": "2025-11-14T12:00:00.000Z",
  "version": "2.1.1"
}
```

### Logs

CLI operations are logged to `.swarm/cli.log`:

```
[2025-11-14T12:00:00.000Z] [INFO] Created sandbox: sb-1234567890
[2025-11-14T12:00:05.000Z] [INFO] Deployed agent: agent-0987654321 (momentum)
[2025-11-14T12:00:10.000Z] [INFO] Health check: healthy
```

## Best Practices

### 1. Resource Management

```bash
# Check health before scaling
node e2b-swarm-cli.js health --detailed

# Scale gradually
node e2b-swarm-cli.js scale --count 3
sleep 30
node e2b-swarm-cli.js health
node e2b-swarm-cli.js scale --count 5
```

### 2. Error Handling

```bash
# Use JSON mode for scripting
if ! output=$(node e2b-swarm-cli.js create --count 3 --json); then
  echo "Failed to create sandboxes"
  exit 1
fi

# Parse and validate
echo "$output" | jq '.sandboxes[] | .id'
```

### 3. Monitoring

```bash
# Run health checks periodically
watch -n 60 "node e2b-swarm-cli.js health"

# Monitor logs
tail -f .swarm/cli.log | grep ERROR
```

### 4. Cleanup

```bash
# List all sandboxes
node e2b-swarm-cli.js list --json > sandboxes.json

# Destroy each sandbox
jq -r '.sandboxes[] | .id' sandboxes.json | while read id; do
  node e2b-swarm-cli.js destroy "$id" --force
done
```

### 5. Deployment Strategies

```bash
# Blue-green deployment
node e2b-swarm-cli.js create --count 3 --name green
node e2b-swarm-cli.js deploy --agent momentum --symbols AAPL,MSFT
# Test green deployment
# Switch traffic
# Destroy blue deployment
```

## Troubleshooting

### Common Issues

**Issue: "Missing required environment variables"**
```bash
# Solution: Check .env file
cat .env | grep E2B_API_KEY
```

**Issue: "Sandbox creation failed"**
```bash
# Solution: Check E2B API key validity
echo $E2B_API_KEY
# Verify API key on E2B dashboard
```

**Issue: "Agent deployment failed"**
```bash
# Solution: Check sandbox status
node e2b-swarm-cli.js list --status failed
# Check logs
cat .swarm/cli.log | grep ERROR
```

### Getting Help

```bash
# Show all commands
node e2b-swarm-cli.js --help

# Show command-specific help
node e2b-swarm-cli.js create --help
node e2b-swarm-cli.js deploy --help
```

## Integration with Claude-Flow

```bash
# Use with hooks for coordination
npx claude-flow@alpha hooks pre-task --description "Deploying E2B swarm"

# Deploy with CLI
node e2b-swarm-cli.js create --count 5 --json

# Report completion
npx claude-flow@alpha hooks post-task --task-id "e2b-deployment"
```

## API Usage (Programmatic)

```javascript
const { createCLI, CLIStateManager } = require('./e2b-swarm-cli');

// Use state manager
const state = new CLIStateManager();
const sandboxes = state.getSandboxes('running');

// Use CLI programmatically
const program = createCLI();
program.parse(['node', 'cli', 'create', '--count', '3', '--json']);
```

## Performance Tips

1. **Batch Operations**: Create multiple sandboxes in one command
2. **Parallel Deployment**: Deploy multiple agents simultaneously
3. **JSON Mode**: Use `--json` for faster parsing in scripts
4. **Rate Limiting**: Add delays between operations to avoid API limits

## Security Considerations

1. **Never commit** `.env` files with real credentials
2. **Use environment variables** for sensitive data
3. **Rotate API keys** regularly
4. **Monitor logs** for suspicious activity
5. **Use force flag carefully** when destroying resources

## Support

For issues and questions:
- GitHub Issues: https://github.com/ruvnet/neural-trader/issues
- Documentation: https://github.com/ruvnet/neural-trader
- E2B Documentation: https://e2b.dev/docs

---

**Version**: 2.1.1
**Last Updated**: 2025-11-14
