# Command Specification Reference
## Neural Trader CLI v3.0

**Version:** 1.0.0
**Date:** 2025-11-17

---

## Command Syntax

```
neural-trader <command> [subcommand] [options] [arguments]
```

### Global Options

```
--help, -h          Show help
--version, -v       Show version
--config <path>     Config file path
--debug             Enable debug logging
--quiet, -q         Quiet mode (minimal output)
--json              JSON output format
--no-color          Disable colors
--profile           Enable profiling
```

---

## Core Commands

### version

Show version and system information.

```bash
neural-trader version

# Output:
# Neural Trader v3.0.0
# Node: v18.17.0
# Platform: linux-x64
# NAPI Bindings: Available
# MCP Server: v2.1.0
# Packages: 17 available
```

**Options:**
- `--json` - JSON output

---

### help

Show help for commands.

```bash
neural-trader help [command]

# Examples:
neural-trader help                    # General help
neural-trader help agent              # Help for agent commands
neural-trader help agent spawn        # Help for specific subcommand
```

---

### interactive

Start interactive shell mode.

```bash
neural-trader interactive

# Features:
# - Command auto-completion
# - Command history (up/down arrows)
# - Inline help (Tab key)
# - Multi-line input (\ at end of line)
# - Exit: Ctrl+D or 'exit'
```

**Options:**
- `--history-size <n>` - History size (default: 1000)
- `--no-autocomplete` - Disable auto-completion

**Example Session:**
```
neural-trader> package list trading
┌─────────────────┬────────────────────────────────┐
│ Package         │ Description                    │
├─────────────────┼────────────────────────────────┤
│ trading         │ Algorithmic trading system     │
│ backtesting     │ Backtesting engine             │
│ portfolio       │ Portfolio management           │
│ news-trading    │ News-driven trading            │
└─────────────────┴────────────────────────────────┘

neural-trader> agent spawn momentum --symbols AAPL
✓ Agent spawned: agent-a3d8f1 (momentum)

neural-trader> agent status agent-a3d8f1
Status: running
Strategy: momentum
Symbols: [AAPL]
Positions: 0
PnL: $0.00

neural-trader> exit
Goodbye!
```

---

### configure

Interactive configuration wizard.

```bash
neural-trader configure [--reset]

# Wizard prompts:
# 1. MCP Server settings
# 2. Agent coordination settings
# 3. Monitoring settings
# 4. Deployment settings
# 5. API keys and credentials
# 6. Logging and output preferences
```

**Options:**
- `--reset` - Reset to defaults
- `--show` - Show current config
- `--edit` - Open config in editor
- `--validate` - Validate config file

---

### doctor

System health check and diagnostics.

```bash
neural-trader doctor [--fix] [--verbose]

# Checks:
# ✓ Node.js version
# ✓ NAPI bindings
# ✓ Package installations
# ✓ Configuration files
# ✓ MCP server
# ✓ API key validation
# ✓ Network connectivity
# ⚠ Disk space
# ✗ Missing dependencies
```

**Options:**
- `--fix` - Auto-fix issues
- `--verbose` - Detailed output
- `--json` - JSON output

---

## Package Commands

### package list

List available packages.

```bash
neural-trader package list [category] [options]

# Examples:
neural-trader package list                    # All packages
neural-trader package list trading            # Trading packages
neural-trader package list --examples         # Example packages
neural-trader package list --installed        # Installed packages
```

**Options:**
- `--category <cat>` - Filter by category
- `--examples` - Only example packages
- `--installed` - Only installed packages
- `--json` - JSON output

**Output:**
```
Available Packages (17 total):

TRADING
  trading         Algorithmic trading system
  backtesting     High-performance backtesting
  portfolio       Portfolio management
  news-trading    Sentiment-driven trading

BETTING
  sports-betting  Sports betting with Kelly criterion

ACCOUNTING
  accounting      Tax-aware portfolio accounting

PREDICTION
  predictor       Conformal prediction with WASM

EXAMPLES (10)
  Use 'neural-trader package list --examples' to see all
```

---

### package info

Show detailed package information.

```bash
neural-trader package info <package>

# Example:
neural-trader package info backtesting
```

**Output:**
```
Backtesting Engine
Category: trading

Description:
  High-performance backtesting with walk-forward optimization
  and Monte Carlo simulation

Features:
  • Multi-threaded execution
  • Walk-forward analysis
  • Monte Carlo simulation
  • Performance metrics

NPM Packages:
  • @neural-trader/backtesting
  • @neural-trader/market-data

Examples: Yes
Documentation: https://neural-trader.io/docs/backtesting

Initialize:
  neural-trader init backtesting
```

---

### package install

Install a package or dependency.

```bash
neural-trader package install <package> [options]

# Examples:
neural-trader package install backtesting
neural-trader package install @neural-trader/risk --save-dev
```

**Options:**
- `--save-dev` - Install as dev dependency
- `--global` - Global installation
- `--version <ver>` - Specific version

---

### package update

Update packages.

```bash
neural-trader package update [package] [options]

# Examples:
neural-trader package update              # Update all
neural-trader package update backtesting  # Update specific
neural-trader package update --check      # Check for updates
```

**Options:**
- `--check` - Check for updates only
- `--major` - Include major versions
- `--dry-run` - Show what would be updated

---

### package remove

Remove a package.

```bash
neural-trader package remove <package>

# Example:
neural-trader package remove backtesting
```

**Options:**
- `--force` - Force removal

---

## MCP Commands

### mcp start

Start MCP server.

```bash
neural-trader mcp start [options]

# Examples:
neural-trader mcp start                      # Default port 3000
neural-trader mcp start --port 3001          # Custom port
neural-trader mcp start --detach             # Background mode
```

**Options:**
- `--port <n>` - Port number (default: 3000)
- `--host <addr>` - Host address (default: localhost)
- `--detach, -d` - Run in background
- `--log <file>` - Log file path

**Output:**
```
Starting MCP server...
✓ Server started on http://localhost:3000
✓ Tools loaded: 99
✓ Ready for connections

Server PID: 12345
Log file: ~/.neural-trader/logs/mcp-server.log

To stop: neural-trader mcp stop
```

---

### mcp stop

Stop MCP server.

```bash
neural-trader mcp stop
```

**Output:**
```
Stopping MCP server...
✓ Server stopped (PID: 12345)
```

---

### mcp restart

Restart MCP server.

```bash
neural-trader mcp restart
```

---

### mcp status

Show MCP server status.

```bash
neural-trader mcp status [--json]

# Output:
# MCP Server Status
#
# Status: running
# PID: 12345
# Uptime: 2h 34m 15s
# Port: 3000
# Tools: 99 loaded
# Requests: 1,234 total (5.2/sec avg)
# Memory: 84.3 MB
# CPU: 2.1%
```

**Options:**
- `--json` - JSON output

---

### mcp tools

List available MCP tools.

```bash
neural-trader mcp tools [options]

# Examples:
neural-trader mcp tools                      # All tools
neural-trader mcp tools --filter trading     # Filter by name
neural-trader mcp tools --category execution # Filter by category
```

**Options:**
- `--filter <pattern>` - Filter by name
- `--category <cat>` - Filter by category
- `--json` - JSON output

**Output:**
```
MCP Tools (99 available):

MARKET DATA (12)
  market_data.fetch                Fetch market data
  market_data.subscribe            Subscribe to real-time data
  market_data.historical           Get historical data
  ...

TRADING (18)
  trading.execute_order            Execute market/limit order
  trading.cancel_order             Cancel pending order
  trading.get_positions            Get current positions
  ...

ANALYSIS (24)
  analysis.technical               Technical analysis
  analysis.sentiment               Sentiment analysis
  analysis.risk                    Risk analysis
  ...

Use 'neural-trader mcp test <tool>' to test a tool
```

---

### mcp test

Test an MCP tool.

```bash
neural-trader mcp test <tool> [args] [options]

# Examples:
neural-trader mcp test trading.get_positions
neural-trader mcp test market_data.fetch '{"symbol":"AAPL"}'
neural-trader mcp test analysis.technical --file params.json
```

**Options:**
- `--file <path>` - Read args from JSON file
- `--dry-run` - Simulate without executing
- `--verbose` - Show detailed output

**Output:**
```
Testing tool: trading.get_positions

Request:
{
  "tool": "trading.get_positions",
  "args": {}
}

Response (142ms):
{
  "positions": [
    {
      "symbol": "AAPL",
      "quantity": 100,
      "entry_price": 150.25,
      "current_price": 155.80,
      "pnl": 555.00,
      "pnl_pct": 3.69
    }
  ],
  "total_pnl": 555.00
}

✓ Test passed
```

---

### mcp configure

Configure Claude Desktop integration.

```bash
neural-trader mcp configure [options]

# Examples:
neural-trader mcp configure --show          # Show current config
neural-trader mcp configure --add           # Add to Claude Desktop
neural-trader mcp configure --remove        # Remove from Claude Desktop
neural-trader mcp configure --verify        # Verify configuration
```

**Options:**
- `--show` - Show current configuration
- `--add` - Add neural-trader MCP server
- `--remove` - Remove from configuration
- `--verify` - Verify configuration
- `--path <path>` - Claude Desktop config path

**Output (--add):**
```
Configuring Claude Desktop...

Config file: ~/.config/claude/config.json
✓ Backup created: config.json.backup
✓ Configuration updated
✓ neural-trader MCP server added

Configuration:
{
  "mcpServers": {
    "neural-trader": {
      "command": "neural-trader",
      "args": ["mcp", "start"],
      "env": {}
    }
  }
}

Please restart Claude Desktop for changes to take effect.
```

---

## Agent Commands

### agent spawn

Spawn a trading agent.

```bash
neural-trader agent spawn <strategy> [options]

# Examples:
neural-trader agent spawn momentum --symbols AAPL,MSFT
neural-trader agent spawn pairs-trading --config ./config.yaml
neural-trader agent spawn mean-reversion --symbols SPY --dry-run
neural-trader agent spawn portfolio --file strategies.json
```

**Options:**
- `--symbols <list>` - Trading symbols (comma-separated)
- `--config <file>` - Configuration file
- `--dry-run` - Simulation mode (no real trades)
- `--capital <amount>` - Starting capital
- `--max-position <size>` - Max position size
- `--risk <pct>` - Max portfolio risk (%)
- `--name <name>` - Agent name
- `--detach, -d` - Run in background

**Strategies:**
- `momentum` - Momentum trading
- `mean-reversion` - Mean reversion
- `pairs-trading` - Statistical arbitrage
- `market-making` - Market making
- `portfolio` - Portfolio optimization
- `custom` - Custom strategy (requires --config)

**Output:**
```
Spawning agent...

Agent: agent-a3d8f1 (momentum)
Strategy: momentum
Symbols: [AAPL, MSFT]
Capital: $100,000
Risk: 2% max
Mode: dry-run

✓ Agent spawned and running
✓ Strategy initialized
✓ Market data connected

Monitor: neural-trader agent status agent-a3d8f1
Logs: neural-trader agent logs agent-a3d8f1 --follow
Stop: neural-trader agent stop agent-a3d8f1
```

---

### agent list

List active agents.

```bash
neural-trader agent list [options]

# Examples:
neural-trader agent list                  # All agents
neural-trader agent list --status running # Running agents only
neural-trader agent list --strategy momentum # By strategy
```

**Options:**
- `--status <status>` - Filter by status (running, stopped, error)
- `--strategy <name>` - Filter by strategy
- `--json` - JSON output

**Output:**
```
Active Agents (3):

┌─────────────┬─────────────┬───────┬────────────┬──────────┬─────────┐
│ ID          │ Strategy    │ Status│ Symbols    │ Positions│ PnL     │
├─────────────┼─────────────┼───────┼────────────┼──────────┼─────────┤
│ agent-a3d8f1│ momentum    │ ✓ Run │ AAPL, MSFT │ 2        │ +$1,234 │
│ agent-b7e2c4│ mean-rev    │ ✓ Run │ SPY, QQQ   │ 1        │ -$156   │
│ agent-f9a1d8│ portfolio   │ ⚠ Idle│ 10 symbols │ 8        │ +$3,456 │
└─────────────┴─────────────┴───────┴────────────┴──────────┴─────────┘

Total PnL: +$4,534
```

---

### agent status

Show agent status details.

```bash
neural-trader agent status <agent-id> [options]

# Example:
neural-trader agent status agent-a3d8f1 --detailed
```

**Options:**
- `--detailed` - Show detailed information
- `--json` - JSON output

**Output:**
```
Agent: agent-a3d8f1

GENERAL
  Status: running
  Strategy: momentum
  Name: momentum-trader-1
  Started: 2h 34m ago
  Uptime: 99.8%

CONFIGURATION
  Symbols: [AAPL, MSFT]
  Capital: $100,000
  Mode: dry-run
  Risk: 2% max
  Max position: $10,000

PERFORMANCE
  Positions: 2 active
  Orders: 5 pending
  Trades: 23 total (15 win, 8 loss)
  Win rate: 65.2%
  PnL: +$1,234.56 (1.23%)
  Sharpe ratio: 1.85
  Max drawdown: -2.1%

RESOURCES
  Memory: 42.3 MB
  CPU: 3.2%
  Network: 1.2 KB/s

RECENT ACTIVITY
  2m ago  BUY AAPL 50 @ $155.80
  5m ago  SELL MSFT 30 @ $320.45
  12m ago SIGNAL: momentum strong on AAPL
```

---

### agent stop

Stop a running agent.

```bash
neural-trader agent stop <agent-id> [options]

# Example:
neural-trader agent stop agent-a3d8f1 --graceful
```

**Options:**
- `--graceful` - Graceful shutdown (close positions first)
- `--force` - Force stop immediately
- `--keep-positions` - Keep positions open

**Output:**
```
Stopping agent agent-a3d8f1...

✓ Stop signal sent
✓ Pending orders cancelled (3)
✓ Positions closed (2)
  - Sold 50 AAPL @ $155.90 (+$5.00)
  - Sold 100 MSFT @ $320.50 (+$15.00)
✓ Agent stopped

Final PnL: +$1,234.56 (1.23%)
Trades: 23 total (15 win, 8 loss)
```

---

### agent logs

Show agent logs.

```bash
neural-trader agent logs <agent-id> [options]

# Examples:
neural-trader agent logs agent-a3d8f1           # Last 100 lines
neural-trader agent logs agent-a3d8f1 --follow  # Live tail
neural-trader agent logs agent-a3d8f1 --tail 50 --level error
```

**Options:**
- `--follow, -f` - Follow log output
- `--tail <n>` - Last n lines (default: 100)
- `--level <level>` - Filter by level (debug, info, warn, error)
- `--since <time>` - Since timestamp
- `--json` - JSON output

**Output:**
```
[2025-11-17 10:15:23] INFO  Strategy initialized: momentum
[2025-11-17 10:15:24] INFO  Connected to market data
[2025-11-17 10:15:25] INFO  Subscribed to: AAPL, MSFT
[2025-11-17 10:16:12] INFO  Signal: momentum strong on AAPL
[2025-11-17 10:16:13] INFO  Placing order: BUY AAPL 50 @ market
[2025-11-17 10:16:14] INFO  Order filled: BUY AAPL 50 @ $155.80
[2025-11-17 10:20:45] WARN  High volatility detected on MSFT
[2025-11-17 10:25:30] INFO  Position update: AAPL +$127.50 (+1.63%)
```

---

### agent coordinate

Multi-agent coordination.

```bash
neural-trader agent coordinate [options]

# Examples:
neural-trader agent coordinate --strategy multi-strategy --agents agent-1,agent-2
neural-trader agent coordinate --topology mesh --config coordination.yaml
```

**Options:**
- `--strategy <name>` - Coordination strategy
- `--agents <list>` - Agent IDs (comma-separated)
- `--topology <type>` - Network topology (mesh, hierarchical, star)
- `--config <file>` - Configuration file

**Strategies:**
- `multi-strategy` - Combine multiple strategies
- `risk-parity` - Risk parity allocation
- `portfolio-rebalance` - Dynamic rebalancing
- `custom` - Custom coordination

---

## Monitor Commands

### monitor dashboard

Launch real-time monitoring dashboard.

```bash
neural-trader monitor dashboard [options]

# Examples:
neural-trader monitor dashboard                       # All agents
neural-trader monitor dashboard --agent agent-a3d8f1  # Specific agent
neural-trader monitor dashboard --layout compact      # Compact layout
```

**Options:**
- `--agent <id>` - Monitor specific agent
- `--refresh <ms>` - Refresh interval (default: 1000)
- `--layout <name>` - Dashboard layout (default, compact, detailed)
- `--widgets <list>` - Active widgets (comma-separated)

**Dashboard Widgets:**
- Positions table
- PnL chart
- Orders list
- Market data ticker
- Performance metrics
- System resources

**Keyboard Controls:**
- `q` - Quit
- `r` - Refresh now
- `p` - Pause/resume
- `1-9` - Switch layouts
- `↑↓` - Scroll
- `Tab` - Switch widgets

---

### monitor positions

Show current positions.

```bash
neural-trader monitor positions [options]

# Examples:
neural-trader monitor positions
neural-trader monitor positions --agent agent-a3d8f1
neural-trader monitor positions --symbol AAPL
```

**Options:**
- `--agent <id>` - Filter by agent
- `--symbol <sym>` - Filter by symbol
- `--json` - JSON output

**Output:**
```
Current Positions:

┌────────┬──────────┬─────────────┬───────────────┬──────────┬──────────┐
│ Symbol │ Quantity │ Entry Price │ Current Price │ PnL      │ PnL %    │
├────────┼──────────┼─────────────┼───────────────┼──────────┼──────────┤
│ AAPL   │ 50       │ $155.80     │ $156.90       │ +$55.00  │ +0.71%   │
│ MSFT   │ 100      │ $320.45     │ $320.50       │ +$5.00   │ +0.02%   │
│ GOOGL  │ 30       │ $140.20     │ $139.80       │ -$12.00  │ -0.29%   │
└────────┴──────────┴─────────────┴───────────────┴──────────┴──────────┘

Total: 3 positions
Total Value: $51,144.00
Total PnL: +$48.00 (+0.09%)
```

---

### monitor pnl

Show profit & loss breakdown.

```bash
neural-trader monitor pnl [options]

# Examples:
neural-trader monitor pnl
neural-trader monitor pnl --period 1w --breakdown
neural-trader monitor pnl --agent agent-a3d8f1 --export pnl.json
```

**Options:**
- `--agent <id>` - Filter by agent
- `--period <time>` - Time period (1d, 1w, 1m, ytd, all)
- `--breakdown` - Show by symbol/strategy
- `--export <file>` - Export to file
- `--format <fmt>` - Export format (json, csv)

**Output:**
```
Profit & Loss (Last 7 days):

SUMMARY
  Total PnL: +$1,234.56 (+1.23%)
  Realized: +$987.65
  Unrealized: +$246.91
  Trades: 23 (15 win, 8 loss)
  Win rate: 65.2%

BY SYMBOL
  AAPL:  +$555.00 (45.0%)
  MSFT:  +$432.10 (35.0%)
  GOOGL: +$247.46 (20.0%)

BY STRATEGY
  momentum:      +$789.12 (63.9%)
  mean-reversion: +$445.44 (36.1%)

METRICS
  Sharpe Ratio: 1.85
  Sortino Ratio: 2.34
  Max Drawdown: -2.1%
  Win/Loss Ratio: 1.88
```

---

### monitor metrics

Show performance metrics.

```bash
neural-trader monitor metrics [options]

# Examples:
neural-trader monitor metrics
neural-trader monitor metrics --agent agent-a3d8f1
neural-trader monitor metrics --export metrics.json --format json
```

**Options:**
- `--agent <id>` - Filter by agent
- `--export <file>` - Export to file
- `--format <fmt>` - Export format (json, csv)
- `--json` - JSON output

**Output:**
```
Performance Metrics:

RETURNS
  Total Return: +1.23%
  Daily Avg: +0.18%
  Best Day: +2.1%
  Worst Day: -1.3%

RISK METRICS
  Volatility: 12.3% (annualized)
  Sharpe Ratio: 1.85
  Sortino Ratio: 2.34
  Beta: 0.95
  Max Drawdown: -2.1%
  Value at Risk (95%): -1.5%

TRADING METRICS
  Total Trades: 23
  Win Rate: 65.2%
  Profit Factor: 2.15
  Avg Win: +$87.34
  Avg Loss: -$40.62
  Win/Loss Ratio: 2.15

EFFICIENCY
  Order Fill Rate: 98.5%
  Avg Execution Time: 145ms
  Slippage: 0.03%
  Commission/Trade: $1.00
```

---

## Deploy Commands

### deploy e2b

Deploy to E2B sandbox.

```bash
neural-trader deploy e2b create <strategy> [options]

# Examples:
neural-trader deploy e2b create momentum --template advanced
neural-trader deploy e2b create custom --file ./strategy.js --env API_KEY=xxx
```

**Options:**
- `--template <name>` - Template to use
- `--file <path>` - Strategy file
- `--env <vars>` - Environment variables (KEY=value)
- `--config <file>` - Configuration file
- `--name <name>` - Deployment name

**Output:**
```
Creating E2B deployment...

✓ Sandbox created: sb-a3d8f1
✓ Files uploaded: strategy.js, config.yaml
✓ Environment configured
✓ Dependencies installed
✓ Strategy started

Deployment: deploy-e2b-a3d8f1
Status: running
URL: https://sb-a3d8f1.e2b.dev
Logs: neural-trader deploy logs deploy-e2b-a3d8f1

To stop: neural-trader deploy stop deploy-e2b-a3d8f1
```

---

### deploy flow-nexus

Deploy to Flow Nexus platform.

```bash
neural-trader deploy flow-nexus create <strategy> [options]

# Examples:
neural-trader deploy flow-nexus create momentum --scale 3
neural-trader deploy flow-nexus create portfolio --region us-east --scale 5
```

**Options:**
- `--scale <n>` - Number of instances
- `--region <name>` - Deployment region
- `--config <file>` - Configuration file
- `--env <vars>` - Environment variables

**Output:**
```
Creating Flow Nexus deployment...

✓ Deployment created: deploy-fn-b7e2c4
✓ Instances: 3 (us-east)
✓ Load balancer configured
✓ Auto-scaling enabled (1-5 instances)
✓ All instances healthy

Deployment: deploy-fn-b7e2c4
Status: running
Endpoint: https://deploy-fn-b7e2c4.flow-nexus.io
Instances: 3/3 healthy

Monitoring: neural-trader deploy status deploy-fn-b7e2c4
Logs: neural-trader deploy logs deploy-fn-b7e2c4
```

---

### deploy list

List all deployments.

```bash
neural-trader deploy list [options]

# Examples:
neural-trader deploy list
neural-trader deploy list --platform e2b
neural-trader deploy list --status running
```

**Options:**
- `--platform <name>` - Filter by platform (e2b, flow-nexus)
- `--status <status>` - Filter by status
- `--json` - JSON output

**Output:**
```
Active Deployments (5):

┌──────────────────┬────────────┬─────────┬────────┬─────────┬────────────┐
│ ID               │ Platform   │ Strategy│ Status │ Uptime  │ Cost/Day   │
├──────────────────┼────────────┼─────────┼────────┼─────────┼────────────┤
│ deploy-e2b-a3d8f1│ E2B        │ momentum│ ✓ Run  │ 2d 3h   │ $2.40      │
│ deploy-fn-b7e2c4 │ Flow Nexus │ portfolio│ ✓ Run │ 5h 23m  │ $7.20      │
│ deploy-e2b-c1f9a2│ E2B        │ pairs   │ ⚠ Idle │ 12h 45m │ $0.60      │
└──────────────────┴────────────┴─────────┴────────┴─────────┴────────────┘

Total cost: $10.20/day
```

---

### deploy status

Show deployment status.

```bash
neural-trader deploy status <deployment-id> [options]

# Example:
neural-trader deploy status deploy-fn-b7e2c4 --detailed
```

**Options:**
- `--detailed` - Detailed information
- `--json` - JSON output

---

### deploy logs

Show deployment logs.

```bash
neural-trader deploy logs <deployment-id> [options]

# Examples:
neural-trader deploy logs deploy-e2b-a3d8f1 --follow
neural-trader deploy logs deploy-fn-b7e2c4 --tail 50 --level error
```

**Options:**
- `--follow, -f` - Follow log output
- `--tail <n>` - Last n lines
- `--level <level>` - Filter by level
- `--since <time>` - Since timestamp

---

### deploy stop

Stop a deployment.

```bash
neural-trader deploy stop <deployment-id> [options]

# Example:
neural-trader deploy stop deploy-e2b-a3d8f1 --graceful
```

**Options:**
- `--graceful` - Graceful shutdown
- `--force` - Force stop

---

## Profile Commands

### profile start

Start performance profiling.

```bash
neural-trader profile start [options]

# Example:
neural-trader profile start --output ./profile-data
```

**Options:**
- `--output <dir>` - Output directory
- `--sample-rate <ms>` - Sample rate (default: 100)
- `--cpu` - CPU profiling
- `--memory` - Memory profiling
- `--all` - All profiling

---

### profile stop

Stop profiling.

```bash
neural-trader profile stop
```

---

### profile report

Generate profiling report.

```bash
neural-trader profile report [options]

# Examples:
neural-trader profile report --html
neural-trader profile report --json --output report.json
```

**Options:**
- `--html` - HTML report
- `--json` - JSON report
- `--output <file>` - Output file
- `--open` - Open report in browser

---

## Template Commands

### template list

List available templates.

```bash
neural-trader template list [options]

# Examples:
neural-trader template list
neural-trader template list --category trading
```

**Options:**
- `--category <cat>` - Filter by category
- `--json` - JSON output

---

### template info

Show template information.

```bash
neural-trader template info <template>

# Example:
neural-trader template info advanced-momentum
```

---

### template use

Use a template to create project.

```bash
neural-trader template use <template> [options]

# Examples:
neural-trader template use advanced-momentum --output ./my-strategy
neural-trader template use custom-backtest --params params.json
```

**Options:**
- `--output <dir>` - Output directory
- `--params <file>` - Template parameters

---

### template create

Create a template from existing project.

```bash
neural-trader template create <name> [options]

# Example:
neural-trader template create my-template --from ./my-strategy
```

**Options:**
- `--from <dir>` - Source directory
- `--description <text>` - Template description
- `--category <cat>` - Template category

---

## Enhanced Init Command

### init

Initialize a new project (enhanced).

```bash
neural-trader init [type] [options]

# Examples:
neural-trader init                                    # Interactive mode
neural-trader init trading                            # Basic trading
neural-trader init trading --template advanced        # With template
neural-trader init backtesting --examples             # With examples
neural-trader init --interactive                      # Wizard mode
```

**Options:**
- `--template <name>` - Use template
- `--interactive` - Interactive wizard
- `--examples` - Include examples
- `--output <dir>` - Output directory
- `--git` - Initialize git repo
- `--install` - Run npm install

---

## Enhanced Test Command

### test

Run tests (enhanced).

```bash
neural-trader test [options]

# Examples:
neural-trader test                    # Run all tests
neural-trader test --unit             # Unit tests only
neural-trader test --integration      # Integration tests
neural-trader test --e2e              # End-to-end tests
neural-trader test --package backtesting  # Specific package
```

**Options:**
- `--unit` - Unit tests
- `--integration` - Integration tests
- `--e2e` - End-to-end tests
- `--package <name>` - Test specific package
- `--watch` - Watch mode
- `--coverage` - Coverage report

---

## Conclusion

This specification provides a complete reference for all neural-trader CLI commands. Each command is designed for consistency, discoverability, and ease of use.

**Key Principles:**
- Consistent syntax and options
- Helpful error messages
- Rich, formatted output
- JSON output for scripting
- Interactive and batch modes
- Comprehensive help text

For implementation details, see:
- CLI_ENHANCEMENT_PLAN.md
- PLUGIN_SYSTEM_DESIGN.md
