# Claude Code MCP Integration Quick Start

## Overview
This guide shows how to use the AI News Trading Platform's MCP tools directly through Claude Code, leveraging the configured MCP server for neural forecasting and trading operations.

## MCP Server Configuration
The MCP server is already configured in `.roo/mcp.json`:
```json
{
  "mcpServers": {
    "ai-news-trader": {
      "type": "stdio",
      "command": "python",
      "args": ["src/mcp/mcp_server_enhanced.py"],
      "cwd": "/workspaces/ai-news-trader",
      "env": {
        "MCP_SERVER_NAME": "AI News Trading Platform",
        "MCP_SERVER_VERSION": "1.0.0",
        "PYTHONPATH": "/workspaces/ai-news-trader",
        "PYTHONUNBUFFERED": "1",
        "MCP_TIMEOUT": "30000"
      }
    }
  },
  "globalShortcut": "Ctrl+Shift+M"
}
```

## Using MCP Tools in Claude Code

### Direct Tool Access
When using Claude Code, you can directly request MCP tool operations:

```
"Use the neural_forecast tool to predict AAPL price for the next 24 hours"
"Run a backtest on the momentum strategy for TSLA from 2024-01-01 to 2024-06-01"
"Analyze current market sentiment for tech stocks using analyze_news"
```

### Available MCP Tools (21 Total)

#### Neural Forecasting Tools
- `neural_forecast` - Generate AI price predictions
- `neural_train` - Train custom models
- `neural_evaluate` - Evaluate model performance
- `neural_backtest` - Historical validation
- `neural_model_status` - Check model health
- `neural_optimize` - Hyperparameter tuning

#### Trading Strategy Tools
- `quick_analysis` - Real-time market analysis
- `simulate_trade` - Trade simulation
- `execute_trade` - Live trading (demo mode)
- `get_portfolio_status` - Portfolio analytics

#### Advanced Analytics
- `run_backtest` - Strategy backtesting
- `optimize_strategy` - Parameter optimization
- `performance_report` - Performance metrics
- `correlation_analysis` - Asset correlations
- `run_benchmark` - System benchmarks
- `risk_analysis` - Risk assessment

#### News & Sentiment
- `analyze_news` - AI news analysis
- `get_news_sentiment` - Sentiment scores

#### System Tools
- `ping` - Connectivity test
- `list_strategies` - Available strategies
- `get_strategy_info` - Strategy details

## Quick Start Examples

### 1. Neural Price Prediction
```
"Generate a neural forecast for AAPL with 24-hour horizon using GPU acceleration"
```
Claude Code will use: `mcp__ai-news-trader__neural_forecast` with parameters:
- symbol: "AAPL"
- horizon: 24
- use_gpu: true

### 2. Strategy Backtesting
```
"Backtest the momentum strategy on TSLA for Q1 2024 with transaction costs"
```
Claude Code will use: `mcp__ai-news-trader__run_backtest` with parameters:
- strategy: "momentum"
- symbol: "TSLA"
- start_date: "2024-01-01"
- end_date: "2024-03-31"
- include_costs: true

### 3. Real-time Market Analysis
```
"Analyze current market conditions for SPY and generate a quick trading analysis"
```
Claude Code will use: `mcp__ai-news-trader__quick_analysis` with parameters:
- symbol: "SPY"
- use_gpu: true

### 4. Portfolio Risk Assessment
```
"Run a comprehensive risk analysis on my portfolio: 40% AAPL, 30% GOOGL, 30% MSFT"
```
Claude Code will use: `mcp__ai-news-trader__risk_analysis` with parameters:
- portfolio: [{"symbol": "AAPL", "weight": 0.4}, {"symbol": "GOOGL", "weight": 0.3}, {"symbol": "MSFT", "weight": 0.3}]
- use_gpu: true

## Complex Workflow Example

### Neural-Enhanced Trading Decision
```
"Help me make a trading decision for NVDA: analyze recent news, generate a 48-hour forecast, and assess the risk"
```

Claude Code will orchestrate:
1. `mcp__ai-news-trader__analyze_news` - Get sentiment
2. `mcp__ai-news-trader__neural_forecast` - Price prediction
3. `mcp__ai-news-trader__quick_analysis` - Technical analysis
4. `mcp__ai-news-trader__risk_analysis` - Risk assessment

## Memory Integration

### Storing Results
```
"Store the neural forecast results in memory as 'nvda_forecast_2024'"
```
Claude Code will use both MCP tools and memory commands.

### Using Stored Data
```
"Use the stored momentum_params to run a new backtest on AAPL"
```
Claude Code will retrieve parameters from memory and execute MCP tools.

## Best Practices

### 1. GPU Acceleration
Always specify `use_gpu: true` for neural operations:
```
"Train a neural model on my dataset with GPU acceleration"
```

### 2. Error Handling
Claude Code handles MCP errors gracefully:
```
"If the neural forecast fails, fall back to technical analysis"
```

### 3. Parallel Operations
Request multiple analyses simultaneously:
```
"Analyze AAPL, TSLA, and GOOGL in parallel using quick_analysis"
```

### 4. Chained Operations
Combine tools for comprehensive analysis:
```
"First check market sentiment, then run neural forecast, finally optimize strategy parameters"
```

## Monitoring MCP Server

### Check Server Status
```
"Ping the ai-news-trader MCP server to verify it's running"
```

### View Available Tools
```
"List all available MCP tools from the ai-news-trader server"
```

### Get Tool Details
```
"Show me the parameters for the neural_forecast tool"
```

## Troubleshooting

### If MCP Tools Don't Respond
1. Verify server is running: `"Check if the MCP server process is active"`
2. Check configuration: `"Show the MCP configuration in .roo/mcp.json"`
3. Restart server: `"Help me restart the MCP server"`

### Common Issues
- **Timeout**: Increase MCP_TIMEOUT in configuration
- **GPU errors**: Ensure CUDA is properly configured
- **Data errors**: Verify symbol names and date formats

## Advanced Usage

### Custom Neural Models
```
"Train a new NHITS model on my custom dataset using the neural_train tool"
```

### Strategy Optimization
```
"Optimize the swing trading strategy for maximum Sharpe ratio over the past year"
```

### Real-time Monitoring
```
"Set up continuous monitoring of my portfolio using MCP tools"
```

## Integration with Claude Flow

While using MCP tools directly through Claude Code, you can still leverage Claude Flow features:

### Memory System
```
"Store the backtest results in Claude Flow memory"
./claude-flow memory store "backtest_results" "..."
```

### Swarm Coordination
```
"Use Claude Flow swarm to coordinate multiple MCP analyses"
./claude-flow swarm "Analyze portfolio" --strategy analysis
```

## Next Steps

1. **Test Connection**: `"Ping the MCP server"`
2. **Run First Analysis**: `"Quick analysis of AAPL"`
3. **Generate Forecast**: `"Neural forecast for SPY"`
4. **Explore Strategies**: `"List all available trading strategies"`

Remember: Claude Code automatically handles MCP tool prefixing (`mcp__ai-news-trader__`) and parameter formatting. Just describe what you want to accomplish!