# CLI Reference

This document provides comprehensive reference for all command-line interfaces available in the AI News Trading Platform with Neural Forecasting integration.

## Overview

The platform provides multiple CLI tools:

- **claude-flow**: Main orchestration and development CLI
- **benchmark CLI**: Performance testing and validation
- **MCP CLI**: Model Context Protocol server management
- **Neural Forecast CLI**: Direct neural forecasting operations

## claude-flow CLI

The main CLI for orchestration, agent management, and development workflows.

### Core System Commands

#### start
Start the orchestration system with optional web UI.

```bash
./claude-flow start [OPTIONS]
```

**Options:**
- `--ui`: Launch web interface
- `--port PORT`: Port number (default: 3000)
- `--host HOST`: Host address (default: localhost)
- `--gpu`: Enable GPU acceleration
- `--neural-forecast`: Enable neural forecasting features

**Examples:**
```bash
# Start with web UI and neural forecasting
./claude-flow start --ui --port 3000 --neural-forecast

# Start with GPU acceleration
./claude-flow start --gpu --host 0.0.0.0
```

#### status
Show comprehensive system status including neural forecasting services.

```bash
./claude-flow status [OPTIONS]
```

**Options:**
- `--verbose, -v`: Detailed status information
- `--json`: Output in JSON format
- `--check-gpu`: Include GPU status
- `--check-models`: Check neural model status

**Example Output:**
```json
{
  "system": {
    "status": "running",
    "uptime": "2h 34m",
    "version": "2.1.0"
  },
  "neural_forecasting": {
    "status": "active",
    "models_loaded": 4,
    "gpu_acceleration": true,
    "memory_usage": "3.2GB"
  },
  "mcp_server": {
    "status": "running",
    "port": 3000,
    "active_connections": 2
  },
  "agents": {
    "active": 3,
    "idle": 1
  }
}
```

#### monitor
Real-time system monitoring dashboard with neural forecasting metrics.

```bash
./claude-flow monitor [OPTIONS]
```

**Options:**
- `--refresh SECONDS`: Refresh interval (default: 5)
- `--neural-metrics`: Include neural forecasting metrics
- `--gpu-metrics`: Include GPU utilization
- `--export-metrics`: Export metrics to file

### Neural Forecasting Commands

#### neural forecast
Generate neural forecasts for trading symbols.

```bash
./claude-flow neural forecast SYMBOL [OPTIONS]
```

**Options:**
- `--horizon DAYS`: Forecast horizon in days (default: 30)
- `--model MODEL`: Model to use (nhits, nbeats, autoformer)
- `--confidence LEVELS`: Confidence levels (default: 80,95)
- `--output FORMAT`: Output format (json, csv, plot)
- `--gpu`: Use GPU acceleration

**Examples:**
```bash
# Generate 30-day forecast for AAPL
./claude-flow neural forecast AAPL --horizon 30 --gpu

# Multi-symbol forecast with plotting
./claude-flow neural forecast AAPL,GOOGL,MSFT --output plot

# High-confidence forecast
./claude-flow neural forecast TSLA --confidence 90,95,99
```

#### neural train
Train neural forecasting models.

```bash
./claude-flow neural train [OPTIONS]
```

**Options:**
- `--data PATH`: Training data path
- `--model MODEL`: Model architecture
- `--epochs EPOCHS`: Training epochs (default: 100)
- `--batch-size SIZE`: Batch size (default: 32)
- `--gpu`: Use GPU acceleration
- `--save-path PATH`: Model save location

**Examples:**
```bash
# Train NHITS model with GPU
./claude-flow neural train --model nhits --epochs 200 --gpu

# Train with custom data
./claude-flow neural train --data /data/custom.csv --model nbeats
```

#### neural optimize
Optimize neural model hyperparameters.

```bash
./claude-flow neural optimize MODEL [OPTIONS]
```

**Options:**
- `--trials TRIALS`: Number of optimization trials (default: 100)
- `--metric METRIC`: Optimization metric (mape, rmse, mase)
- `--gpu`: Use GPU acceleration
- `--parallel JOBS`: Parallel jobs (default: 4)

### Agent Management

#### agent spawn
Create AI agents with neural forecasting capabilities.

```bash
./claude-flow agent spawn TYPE [OPTIONS]
```

**Types:**
- `researcher`: Market research and analysis
- `coder`: Code development and testing
- `analyst`: Data analysis and forecasting
- `trader`: Trading strategy execution
- `neural-forecaster`: Specialized neural forecasting agent

**Options:**
- `--name NAME`: Agent name
- `--neural-enabled`: Enable neural forecasting
- `--gpu-access`: Grant GPU access
- `--strategy STRATEGY`: Trading strategy to use

**Examples:**
```bash
# Spawn neural forecasting analyst
./claude-flow agent spawn analyst --name "MarketAnalyst" --neural-enabled

# Spawn GPU-enabled trader
./claude-flow agent spawn trader --gpu-access --strategy momentum
```

### SPARC Development Modes

#### sparc neural
Run neural forecasting development tasks.

```bash
./claude-flow sparc neural "TASK" [OPTIONS]
```

**Options:**
- `--mode MODE`: Development mode (research, implement, test, optimize)
- `--model MODEL`: Neural model to focus on
- `--gpu`: Enable GPU acceleration

**Examples:**
```bash
# Research new forecasting techniques
./claude-flow sparc neural "Research transformer-based forecasting models" --mode research

# Implement NHITS optimization
./claude-flow sparc neural "Optimize NHITS model performance" --mode optimize --gpu
```

### Memory Management with Neural Context

#### memory store-forecast
Store neural forecast results and model states.

```bash
./claude-flow memory store-forecast KEY DATA [OPTIONS]
```

**Options:**
- `--model-state`: Include model state
- `--metadata`: Include forecast metadata
- `--compress`: Compress large forecasts

#### memory get-forecast
Retrieve stored forecasts.

```bash
./claude-flow memory get-forecast KEY [OPTIONS]
```

**Options:**
- `--include-metadata`: Include forecast metadata
- `--format FORMAT`: Output format (json, csv)

## Benchmark CLI

Performance testing and validation for neural forecasting.

### benchmark neural
Run neural forecasting benchmarks.

```bash
cd benchmark/
python benchmark_cli.py neural [OPTIONS]
```

**Options:**
- `--models MODELS`: Models to benchmark (nhits,nbeats,autoformer)
- `--symbols SYMBOLS`: Symbols to test
- `--gpu`: Use GPU acceleration
- `--export PATH`: Export results path

**Example:**
```bash
python benchmark_cli.py neural --models nhits,nbeats --symbols AAPL,GOOGL --gpu
```

### benchmark compare
Compare neural model performance.

```bash
python benchmark_cli.py compare --baseline MODEL --candidates MODELS
```

### benchmark gpu
Test GPU acceleration performance.

```bash
python benchmark_cli.py gpu [OPTIONS]
```

**Options:**
- `--memory-test`: Test memory usage
- `--throughput-test`: Test inference throughput
- `--latency-test`: Test inference latency

## MCP Server CLI

Model Context Protocol server management.

### Start MCP Server

```bash
# Enhanced MCP server with neural forecasting
python mcp_server_enhanced.py

# Start with specific configuration
python mcp_server_enhanced.py --config config.json --port 3001
```

### MCP Tools via claude-flow

```bash
# Start MCP server through claude-flow
./claude-flow mcp start --port 3000

# Check MCP server status
./claude-flow mcp status

# List available MCP tools
./claude-flow mcp tools
```

## Direct Neural Forecasting CLI

Direct access to neural forecasting without orchestration.

### Forecast Generation

```bash
# Generate forecast
python -m src.forecasting.cli forecast SYMBOL [OPTIONS]
```

**Options:**
- `--horizon DAYS`: Forecast horizon
- `--model MODEL`: Model to use
- `--confidence LEVELS`: Confidence intervals
- `--output PATH`: Output file path

### Model Management

```bash
# List available models
python -m src.forecasting.cli models list

# Train new model
python -m src.forecasting.cli models train --data DATA --model MODEL

# Evaluate model
python -m src.forecasting.cli models evaluate --model MODEL --test-data DATA
```

### Batch Processing

```bash
# Batch forecast multiple symbols
python -m src.forecasting.cli batch-forecast --symbols-file symbols.txt --output-dir forecasts/

# Batch model training
python -m src.forecasting.cli batch-train --config batch_config.yaml
```

## Configuration Files

### claude-flow Configuration

Location: `.claude/config.yaml`

```yaml
system:
  neural_forecasting:
    enabled: true
    default_model: "nhits"
    gpu_acceleration: true
    max_memory_gb: 8
    
  mcp_server:
    auto_start: true
    port: 3000
    timeout: 30
    
agents:
  max_concurrent: 8
  neural_forecaster:
    enabled: true
    gpu_access: true
    
memory:
  neural_context:
    enabled: true
    max_forecasts: 1000
    compression: true
```

### Neural Forecasting Configuration

Location: `config/neural_forecast.yaml`

```yaml
models:
  nhits:
    input_size: 168
    horizon: 30
    n_freq_downsample: [168, 24, 1]
    stack_types: ["trend", "seasonality"]
    n_blocks: [1, 1]
    mlp_units: [[512, 512], [512, 512]]
    
  training:
    max_epochs: 100
    learning_rate: 0.001
    batch_size: 32
    early_stopping_patience: 10
    
gpu:
  enabled: true
  memory_fraction: 0.8
  allow_growth: true
```

### MCP Server Configuration

Location: `mcp_config.json`

```json
{
  "server": {
    "name": "ai-news-trader-gpu",
    "version": "1.0.0",
    "neural_forecasting": true,
    "gpu_acceleration": true
  },
  "tools": {
    "neural_forecast": {
      "enabled": true,
      "default_horizon": 30,
      "max_symbols": 50
    },
    "quick_analysis": {
      "neural_enhancement": true,
      "gpu_acceleration": true
    }
  }
}
```

## Environment Variables

### System Configuration

```bash
# Neural forecasting
export NEURAL_FORECAST_GPU=true
export NEURAL_FORECAST_DEVICE=cuda
export NEURAL_FORECAST_BATCH_SIZE=32
export NEURAL_FORECAST_MAX_MEMORY=8

# MCP server
export MCP_SERVER_PORT=3000
export MCP_SERVER_HOST=localhost
export MCP_NEURAL_ENABLED=true

# Claude-flow
export CLAUDE_WORKING_DIR=/workspaces/ai-news-trader
export CLAUDE_GPU_ENABLED=true
export CLAUDE_NEURAL_FORECASTING=true
```

### Development Environment

```bash
# Development mode
export CLAUDE_DEV_MODE=true
export NEURAL_FORECAST_DEBUG=true
export MCP_DEBUG_LOGGING=true

# Testing
export CLAUDE_TEST_MODE=true
export NEURAL_FORECAST_MOCK_GPU=false
export MCP_TEST_TIMEOUT=60
```

## Common Usage Patterns

### Development Workflow

```bash
# 1. Start system with neural forecasting
./claude-flow start --ui --neural-forecast --gpu

# 2. Check system status
./claude-flow status --verbose --check-models

# 3. Generate forecasts for portfolio
./claude-flow neural forecast AAPL,GOOGL,MSFT --horizon 30 --output json

# 4. Run analysis swarm
./claude-flow swarm "Analyze market conditions with neural forecasts" --strategy analysis --mode distributed
```

### Research Workflow

```bash
# 1. Spawn research agents
./claude-flow agent spawn researcher --neural-enabled --name "ForecastResearcher"

# 2. Research new techniques
./claude-flow sparc neural "Research attention mechanisms for financial forecasting" --mode research

# 3. Store findings
./claude-flow memory store "forecast_research" "Attention mechanisms show 15% improvement"

# 4. Generate benchmarks
python benchmark_cli.py neural --models nhits,attention --export research_results.json
```

### Production Deployment

```bash
# 1. Optimize models for production
./claude-flow neural optimize nhits --trials 500 --gpu --parallel 8

# 2. Run comprehensive benchmarks
python benchmark_cli.py neural --gpu --export production_benchmarks.json

# 3. Deploy with monitoring
./claude-flow start --neural-forecast --gpu --monitor --port 8080

# 4. Set up automated forecasting
./claude-flow memory store "daily_forecast_symbols" "AAPL,GOOGL,MSFT,TSLA,NVDA"
```

## Troubleshooting Commands

### Diagnostic Commands

```bash
# Check GPU availability
./claude-flow status --check-gpu

# Validate neural models
./claude-flow neural validate --all-models

# Test MCP connectivity
./claude-flow mcp status --test-connection

# Memory usage analysis
./claude-flow memory stats --neural-context
```

### Performance Analysis

```bash
# Profile neural forecasting performance
python benchmark_cli.py profile --model nhits --symbols AAPL

# Analyze memory usage
python benchmark_cli.py memory --gpu-analysis

# Test latency targets
python benchmark_cli.py latency --target-ms 10
```

### Error Recovery

```bash
# Reset system state
./claude-flow system reset --preserve-memory

# Rebuild neural models
./claude-flow neural rebuild --all-models --gpu

# Restart MCP server
./claude-flow mcp restart --port 3000
```

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | General error |
| 2 | Configuration error |
| 3 | Neural model error |
| 4 | GPU error |
| 5 | MCP server error |
| 6 | Memory error |
| 7 | Network/connectivity error |

## See Also

- [Neural Forecast API](neural_forecast.md)
- [MCP Tools API](mcp_tools.md)
- [Quick Start Guide](../guides/quickstart.md)
- [GPU Optimization Tutorial](../tutorials/gpu_optimization.md)
- [Troubleshooting Guide](../guides/troubleshooting.md)