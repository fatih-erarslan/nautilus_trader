# System Orchestration Commands

## Overview
Claude-Flow orchestration commands for managing the AI trading platform infrastructure, agent coordination, and system monitoring.

## Core System Commands

### claude-flow start
Start the trading platform orchestration system.

**Usage:**
```bash
./claude-flow start [OPTIONS]
```

**Examples:**
```bash
# Basic startup
./claude-flow start

# Start with web UI dashboard
./claude-flow start --ui --port 3000

# Start with custom host and port
./claude-flow start --ui --port 8080 --host 0.0.0.0

# Start with monitoring enabled
./claude-flow start --ui --monitor --port 3000
```

**Features:**
- Launches trading dashboard interface
- Initializes agent management system
- Starts memory management system
- Enables real-time monitoring
- Activates task orchestration

### claude-flow status
Show comprehensive trading platform status.

**Usage:**
```bash
./claude-flow status
```

**Status Information:**
- System running state
- Active agents count
- Task queue status
- Memory usage statistics
- MCP server status
- Terminal pool status
- Neural forecasting availability
- GPU acceleration status

### claude-flow monitor
Real-time system monitoring dashboard.

**Usage:**
```bash
./claude-flow monitor
```

**Monitoring Features:**
- Live agent activity
- Task execution progress
- Memory usage trends
- Trading performance metrics
- Neural forecasting accuracy
- System resource utilization

## Agent Management Commands

### claude-flow agent spawn
Create specialized AI agents for trading tasks.

**Usage:**
```bash
./claude-flow agent spawn <type> [--name <name>]
```

**Trading Agent Types:**
```bash
# Market research agents
./claude-flow agent spawn researcher --name "market_analyst"
./claude-flow agent spawn analyzer --name "tech_analyst"

# Strategy development agents
./claude-flow agent spawn coder --name "strategy_developer"
./claude-flow agent spawn architect --name "system_designer"

# Testing and optimization agents
./claude-flow agent spawn tester --name "backtest_validator"
./claude-flow agent spawn optimizer --name "param_optimizer"

# Risk management agents
./claude-flow agent spawn reviewer --name "risk_manager"
./claude-flow agent spawn debugger --name "error_handler"
```

### claude-flow agent list
List all active trading agents.

**Usage:**
```bash
./claude-flow agent list
```

**Agent Information:**
- Agent ID and name
- Agent type and capabilities
- Current task assignment
- Performance metrics
- Resource usage

### claude-flow spawn
Quick agent spawning (alias for agent spawn).

**Usage:**
```bash
./claude-flow spawn <type>
```

**Quick Examples:**
```bash
# Quick strategy development
./claude-flow spawn coder

# Quick market analysis
./claude-flow spawn researcher

# Quick performance testing
./claude-flow spawn tester
```

## SPARC Development Modes

### claude-flow sparc modes
List all 17 available SPARC development modes.

**Usage:**
```bash
./claude-flow sparc modes
```

**SPARC Mode Categories:**
- **Core Orchestration**: orchestrator, swarm-coordinator, workflow-manager, batch-executor
- **Development**: coder, architect, reviewer, tdd
- **Analysis & Research**: researcher, analyzer, optimizer
- **Creative & Support**: designer, innovator, documenter, debugger
- **Testing & Quality**: tester, memory-manager

### claude-flow sparc run
Execute specific SPARC modes for trading tasks.

**Usage:**
```bash
./claude-flow sparc run <mode> "<task>" [OPTIONS]
```

**Trading-Focused Examples:**
```bash
# Strategy development
./claude-flow sparc run coder "Build neural-enhanced momentum trading strategy"

# Market research
./claude-flow sparc run researcher "Research correlation between news sentiment and price movements"

# System architecture
./claude-flow sparc run architect "Design scalable neural forecasting pipeline with GPU acceleration"

# Performance analysis
./claude-flow sparc run analyzer "Analyze strategy performance across different market regimes"

# Risk assessment
./claude-flow sparc run reviewer "Review trading strategy for risk management compliance"

# Testing framework
./claude-flow sparc run tester "Create comprehensive backtesting framework"

# Parameter optimization
./claude-flow sparc run optimizer "Optimize neural model hyperparameters for maximum accuracy"
```

**Advanced Options:**
```bash
# Parallel execution
./claude-flow sparc run researcher "Market analysis" --parallel

# Memory integration
./claude-flow sparc run coder "Strategy implementation" --memory-key strategy_params

# Monitoring enabled
./claude-flow sparc run optimizer "Parameter tuning" --monitor

# Custom timeout
./claude-flow sparc run analyzer "Deep analysis" --timeout 120
```

### claude-flow sparc tdd
Test-driven development mode for trading algorithms.

**Usage:**
```bash
./claude-flow sparc tdd "<feature_description>"
```

**TDD Examples:**
```bash
# Neural forecasting integration
./claude-flow sparc tdd "Neural forecasting integration with error handling and fallback"

# Risk management system
./claude-flow sparc tdd "Position sizing algorithm with volatility adjustment"

# Order execution system
./claude-flow sparc tdd "Order execution with slippage modeling and cost analysis"

# Portfolio rebalancing
./claude-flow sparc tdd "Automated portfolio rebalancing with correlation constraints"

# Real-time data processing
./claude-flow sparc tdd "Real-time market data processing with neural forecasting pipeline"
```

## Swarm Coordination

### claude-flow swarm
Multi-agent swarm coordination for complex trading projects.

**Usage:**
```bash
./claude-flow swarm "<objective>" [OPTIONS]
```

**Trading Swarm Examples:**
```bash
# Strategy development swarm
./claude-flow swarm "Build complete momentum trading system with neural forecasting" \
  --strategy development --max-agents 5 --parallel --monitor

# Portfolio optimization swarm
./claude-flow swarm "Optimize multi-asset portfolio with risk constraints" \
  --strategy optimization --mode hierarchical --output portfolio_results.json

# Market analysis swarm
./claude-flow swarm "Comprehensive market analysis for Q3 trading strategy" \
  --strategy research --mode distributed --parallel --monitor

# System maintenance swarm
./claude-flow swarm "Update neural models and validate performance" \
  --strategy maintenance --mode centralized --monitor

# Risk assessment swarm
./claude-flow swarm "Comprehensive risk analysis across all trading strategies" \
  --strategy analysis --mode mesh --parallel --output risk_report.json
```

**Swarm Strategies:**
- `research`: Market analysis and data gathering
- `development`: Strategy implementation and coding
- `analysis`: Performance and risk analysis
- `testing`: Validation and quality assurance
- `optimization`: Parameter tuning and improvement
- `maintenance`: System updates and monitoring

**Coordination Modes:**
- `centralized`: Single coordinator manages all agents
- `distributed`: Agents work independently and sync
- `hierarchical`: Tree structure with lead agents
- `mesh`: Full peer-to-peer communication
- `hybrid`: Combination of coordination patterns

## Task Orchestration

### claude-flow task create
Create and manage trading-related tasks.

**Usage:**
```bash
./claude-flow task create <type> [description]
```

**Trading Task Examples:**
```bash
# Neural model training task
./claude-flow task create neural_training "Train NHITS model on FAANG stocks data"

# Strategy backtesting task
./claude-flow task create backtest "Validate momentum strategy on 2024 data"

# Performance analysis task
./claude-flow task create analysis "Compare strategy performance vs benchmarks"

# Risk assessment task
./claude-flow task create risk_analysis "Calculate portfolio VaR and stress test scenarios"
```

### claude-flow task list
View active task queue for trading operations.

**Usage:**
```bash
./claude-flow task list
```

**Task Information:**
- Task ID and type
- Description and priority
- Assigned agent
- Progress status
- Estimated completion time

### claude-flow workflow
Execute automated trading workflows.

**Usage:**
```bash
./claude-flow workflow <file>
```

**Workflow Examples:**
```bash
# Daily trading workflow
./claude-flow workflow config/workflows/daily_trading.yaml

# Weekly rebalancing workflow
./claude-flow workflow config/workflows/weekly_rebalance.yaml

# Model training workflow
./claude-flow workflow config/workflows/neural_training.yaml

# Risk monitoring workflow
./claude-flow workflow config/workflows/risk_monitoring.yaml
```

## Configuration Management

### claude-flow config
Manage trading platform configuration.

**Usage:**
```bash
./claude-flow config <subcommand>
```

**Configuration Subcommands:**
```bash
# Show current configuration
./claude-flow config show

# Get specific setting
./claude-flow config get neural.gpu_enabled

# Set configuration value
./claude-flow config set neural.gpu_enabled true

# Initialize default configuration
./claude-flow config init

# Validate configuration
./claude-flow config validate
```

**Trading Configuration Categories:**
- `neural`: Neural forecasting settings
- `trading`: Strategy parameters
- `risk`: Risk management settings
- `data`: Market data configuration
- `performance`: System optimization
- `monitoring`: Logging and alerts

## Integration with MCP Tools

### claude-flow mcp start
Start MCP server for enhanced tool integration.

**Usage:**
```bash
./claude-flow mcp start [--port 3000] [--host localhost]
```

### claude-flow mcp status
Show MCP server status and available tools.

**Usage:**
```bash
./claude-flow mcp status
```

### claude-flow mcp tools
List all 21 available MCP trading tools.

**Usage:**
```bash
./claude-flow mcp tools
```

**Tool Categories:**
- Neural Forecasting (6 tools)
- Trading Strategies (4 tools)
- Advanced Analytics (7 tools)
- News & Sentiment (2 tools)
- System Tools (2 tools)

## Advanced Orchestration Patterns

### Memory-Driven Coordination
```bash
# Store system state
./claude-flow memory store "system_state" "Production ready, neural models validated, strategies optimized"

# Coordinate using memory
./claude-flow sparc run coder "Implement feature using system_state from memory"
```

### Multi-Stage Trading Operations
```bash
# Stage 1: Data collection and analysis
./claude-flow swarm "Collect and analyze market data for strategy development" --strategy research

# Stage 2: Strategy development
./claude-flow sparc run coder "Build strategy based on research findings"

# Stage 3: Testing and validation
./claude-flow sparc tdd "Comprehensive strategy testing framework"

# Stage 4: Deployment and monitoring
./claude-flow workflow config/workflows/production_deployment.yaml
```

### Continuous Integration for Trading
```bash
# Automated testing workflow
./claude-flow swarm "Run comprehensive test suite on all trading strategies" --strategy testing

# Performance monitoring
./claude-flow monitor --alerts --threshold "sharpe_ratio < 1.5"

# Automated optimization
./claude-flow sparc run optimizer "Continuously optimize strategy parameters based on performance"
```

## Best Practices

### System Orchestration
1. Always start with `./claude-flow status` to check system health
2. Use `--monitor` flag for long-running operations
3. Enable parallel execution for independent tasks
4. Store important configurations in memory
5. Regular workflow automation for routine tasks

### Agent Management
1. Spawn specialized agents for specific trading tasks
2. Monitor agent performance and resource usage
3. Use descriptive names for agent identification
4. Balance agent load across available resources
5. Regular cleanup of completed agent tasks

### SPARC Mode Selection
1. Use `researcher` for market analysis and data gathering
2. Use `coder` for strategy implementation
3. Use `analyzer` for performance evaluation
4. Use `optimizer` for parameter tuning
5. Use `tester` for validation and quality assurance

### Swarm Coordination
1. Choose appropriate coordination mode for task complexity
2. Use parallel execution for independent workstreams
3. Monitor swarm progress with real-time dashboards
4. Store swarm results for future reference
5. Regular swarm performance optimization

### Performance Monitoring
1. Continuous monitoring of system resources
2. Alert setup for critical performance thresholds
3. Regular performance benchmarking
4. Automated scaling based on load
5. Historical performance trend analysis