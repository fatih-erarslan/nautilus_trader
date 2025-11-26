# MCP Tools API Reference

The Model Context Protocol (MCP) server provides a comprehensive suite of tools for AI-powered trading operations with advanced analytics, GPU acceleration, and real-time market analysis.

## Overview

The MCP server exposes 49 advanced tools through JSON-RPC 2.0 protocol across multiple categories:

### Core Trading Tools (15 tools)
- **Trading Operations**: Simulation, execution, portfolio management
- **Market Analysis**: News sentiment, technical analysis, correlation analysis
- **Performance Analytics**: Backtesting, optimization, benchmarking
- **Risk Management**: Portfolio risk analysis, Monte Carlo simulations

### Extended Tool Categories (34 additional tools)
- **Neural Forecasting** (6 tools): AI model training, prediction, optimization
- **Prediction Markets** (6 tools): Market analysis, order placement, position tracking
- **News Collection** (4 tools): Real-time news monitoring, sentiment analysis
- **Strategy Management** (4 tools): Strategy selection, comparison, adaptation
- **System Monitoring** (3 tools): Performance metrics, health checks
- **Multi-Asset Trading** (3 tools): Portfolio rebalancing, correlation analysis
- **Sports Betting** (10 tools): Event odds, arbitrage, Kelly calculations
- **Syndicate Management** (17 tools): Member management, fund allocation, profit distribution

For detailed documentation on specific tool categories:
- [Syndicate Tools Guide](../mcp/SYNDICATE_TOOLS.md) - Collaborative investment management
- [Neural Forecast Guide](neural_forecast.md) - AI forecasting capabilities
- [Sports Betting Examples](../sports_betting_api_examples.md) - Betting platform integration

## Server Configuration

### Connection Details

```json
{
  "server": {
    "name": "ai-news-trader-gpu",
    "version": "1.0.0",
    "transport": "stdio",
    "capabilities": {
      "tools": true,
      "resources": true,
      "prompts": true
    }
  }
}
```

### Starting the Server

```bash
# Start MCP server with GPU acceleration
python mcp_server_enhanced.py

# Or via Claude Code MCP integration
./claude-flow mcp start --port 3000 --host localhost
```

## Core Tools

### 1. ping

Health check and server connectivity verification.

**Parameters:** None

**Response:**
```json
{
  "status": "ok",
  "server": "ai-news-trader-gpu",
  "version": "1.0.0",
  "gpu_available": true,
  "capabilities": ["trading", "analytics", "forecasting", "gpu_acceleration"]
}
```

**Example:**
```python
# Claude Code integration
result = mcp__ai_news_trader__ping()
print(f"Server status: {result['status']}")
```

### 2. list_strategies

List all available trading strategies with GPU capabilities.

**Parameters:** None

**Response:**
```json
{
  "strategies": [
    {
      "name": "momentum_trading_optimized",
      "type": "momentum",
      "gpu_accelerated": true,
      "speedup": "5000x",
      "description": "Dual momentum strategy with optimized parameters"
    },
    {
      "name": "mean_reversion_optimized", 
      "type": "mean_reversion",
      "gpu_accelerated": true,
      "speedup": "6000x",
      "description": "Mean reversion with dynamic thresholds"
    }
  ]
}
```

### 3. get_strategy_info

Get detailed information about a specific trading strategy.

**Parameters:**
- `strategy` (string, required): Strategy name

**Response:**
```json
{
  "name": "momentum_trading_optimized",
  "type": "momentum",
  "parameters": {
    "lookback_periods": [3, 11, 33],
    "momentum_thresholds": {
      "strong": 0.40,
      "moderate": 0.35,
      "weak": 0.12
    },
    "max_position_pct": 0.20,
    "stop_loss_pct": 0.132
  },
  "performance": {
    "sharpe_ratio": 0.592,
    "max_drawdown": 0.18,
    "win_rate": 0.64
  },
  "gpu_acceleration": {
    "enabled": true,
    "speedup": "5000x",
    "memory_usage": "2.1GB"
  }
}
```

### 4. quick_analysis

Get rapid market analysis for a symbol with optional GPU acceleration.

**Parameters:**
- `symbol` (string, required): Trading symbol (e.g., "AAPL")
- `use_gpu` (boolean, optional): Enable GPU acceleration (default: false)

**Response:**
```json
{
  "symbol": "AAPL",
  "analysis": {
    "trend": "bullish",
    "momentum": 0.67,
    "volatility": 0.24,
    "support_level": 185.50,
    "resistance_level": 195.20,
    "rsi": 58.3,
    "macd_signal": "buy"
  },
  "neural_forecast": {
    "next_day": 192.45,
    "confidence": 0.82,
    "trend_direction": "up"
  },
  "processing_time_ms": 8.2,
  "gpu_accelerated": true
}
```

### 5. simulate_trade

Simulate a trading operation with performance tracking.

**Parameters:**
- `strategy` (string, required): Strategy name
- `symbol` (string, required): Trading symbol
- `action` (string, required): "buy", "sell", or "hold"
- `use_gpu` (boolean, optional): Enable GPU acceleration (default: false)

**Response:**
```json
{
  "simulation_id": "sim_20240626_001",
  "symbol": "AAPL",
  "strategy": "momentum_trading_optimized",
  "action": "buy",
  "entry_price": 190.25,
  "position_size": 0.15,
  "expected_return": 0.078,
  "risk_metrics": {
    "var_95": -0.032,
    "expected_drawdown": 0.024,
    "sharpe_ratio": 1.24
  },
  "neural_forecast_support": {
    "forecast_alignment": 0.89,
    "confidence_boost": 0.12
  },
  "processing_time_ms": 12.5
}
```

### 6. get_portfolio_status

Get current portfolio status with advanced analytics.

**Parameters:**
- `include_analytics` (boolean, optional): Include advanced analytics (default: true)

**Response:**
```json
{
  "portfolio": {
    "total_value": 125000.00,
    "cash": 25000.00,
    "positions": [
      {
        "symbol": "AAPL",
        "quantity": 50,
        "market_value": 9512.50,
        "unrealized_pnl": 412.50,
        "weight": 0.076
      }
    ]
  },
  "analytics": {
    "total_return": 0.25,
    "sharpe_ratio": 1.45,
    "max_drawdown": 0.087,
    "beta": 0.94,
    "alpha": 0.032,
    "volatility": 0.156
  },
  "risk_metrics": {
    "var_95": -0.028,
    "var_99": -0.045,
    "expected_shortfall": -0.052
  },
  "neural_forecasts": {
    "portfolio_forecast": {
      "next_day_return": 0.008,
      "confidence": 0.76
    }
  }
}
```

### 7. analyze_news

AI sentiment analysis of market news for a symbol.

**Parameters:**
- `symbol` (string, required): Trading symbol
- `lookback_hours` (integer, optional): Hours to look back (default: 24)
- `sentiment_model` (string, optional): Model to use (default: "enhanced")
- `use_gpu` (boolean, optional): Enable GPU acceleration (default: false)

**Response:**
```json
{
  "symbol": "AAPL",
  "sentiment_analysis": {
    "overall_sentiment": 0.72,
    "sentiment_score": "bullish",
    "confidence": 0.85,
    "news_count": 47,
    "sentiment_breakdown": {
      "positive": 0.64,
      "neutral": 0.23,
      "negative": 0.13
    }
  },
  "key_themes": [
    "earnings_beat",
    "product_launch",
    "market_expansion"
  ],
  "impact_prediction": {
    "price_impact": 0.028,
    "confidence": 0.79,
    "time_horizon": "2-3 days"
  },
  "processing_time_ms": 145.2
}
```

### 8. get_news_sentiment

Get real-time news sentiment for a symbol.

**Parameters:**
- `symbol` (string, required): Trading symbol
- `sources` (array[string], optional): News sources to analyze

**Response:**
```json
{
  "symbol": "AAPL",
  "real_time_sentiment": {
    "current_sentiment": 0.68,
    "sentiment_trend": "improving",
    "momentum": 0.12,
    "last_updated": "2024-06-26T14:30:00Z"
  },
  "source_breakdown": {
    "financial_news": 0.71,
    "social_media": 0.65,
    "analyst_reports": 0.72
  },
  "alerts": [
    {
      "type": "sentiment_spike",
      "description": "Positive sentiment increased 15% in last hour",
      "severity": "medium"
    }
  ]
}
```

### 9. run_backtest

Run comprehensive historical backtest with GPU acceleration.

**Parameters:**
- `strategy` (string, required): Strategy name
- `symbol` (string, required): Trading symbol
- `start_date` (string, required): Start date (YYYY-MM-DD)
- `end_date` (string, required): End date (YYYY-MM-DD)
- `benchmark` (string, optional): Benchmark (default: "sp500")
- `include_costs` (boolean, optional): Include transaction costs (default: true)
- `use_gpu` (boolean, optional): Enable GPU acceleration (default: true)

**Response:**
```json
{
  "backtest_id": "bt_20240626_001",
  "strategy": "momentum_trading_optimized",
  "symbol": "AAPL",
  "period": {
    "start_date": "2023-01-01",
    "end_date": "2024-06-26",
    "trading_days": 376
  },
  "performance": {
    "total_return": 0.347,
    "annual_return": 0.285,
    "sharpe_ratio": 1.42,
    "max_drawdown": 0.087,
    "win_rate": 0.64,
    "profit_factor": 1.89
  },
  "benchmark_comparison": {
    "benchmark": "sp500",
    "benchmark_return": 0.198,
    "excess_return": 0.149,
    "beta": 0.94,
    "alpha": 0.048
  },
  "trade_statistics": {
    "total_trades": 89,
    "winning_trades": 57,
    "avg_trade_return": 0.0038,
    "largest_win": 0.089,
    "largest_loss": -0.045
  },
  "neural_forecast_contribution": {
    "accuracy_improvement": 0.23,
    "return_enhancement": 0.067,
    "risk_reduction": 0.031
  },
  "processing_time_ms": 2340.5,
  "gpu_acceleration": {
    "speedup": "4200x",
    "gpu_utilization": 0.78
  }
}
```

### 10. optimize_strategy

Optimize strategy parameters using GPU acceleration.

**Parameters:**
- `strategy` (string, required): Strategy name
- `symbol` (string, required): Trading symbol
- `parameter_ranges` (object, required): Parameter ranges to optimize
- `optimization_metric` (string, optional): Metric to optimize (default: "sharpe_ratio")
- `max_iterations` (integer, optional): Maximum iterations (default: 1000)
- `use_gpu` (boolean, optional): Enable GPU acceleration (default: true)

**Example Request:**
```json
{
  "strategy": "momentum_trading_optimized",
  "symbol": "AAPL",
  "parameter_ranges": {
    "momentum_threshold": [0.1, 0.8],
    "lookback_period": [5, 60],
    "position_size": [0.05, 0.25]
  },
  "optimization_metric": "sharpe_ratio",
  "max_iterations": 500
}
```

**Response:**
```json
{
  "optimization_id": "opt_20240626_001",
  "strategy": "momentum_trading_optimized",
  "symbol": "AAPL",
  "best_parameters": {
    "momentum_threshold": 0.387,
    "lookback_period": 22,
    "position_size": 0.185
  },
  "optimization_results": {
    "best_sharpe": 1.67,
    "best_return": 0.412,
    "best_drawdown": 0.069,
    "improvement_vs_baseline": {
      "sharpe_improvement": 0.25,
      "return_improvement": 0.089,
      "risk_reduction": 0.043
    }
  },
  "convergence": {
    "iterations": 347,
    "converged": true,
    "stability_score": 0.94
  },
  "processing_time_ms": 8750.2,
  "gpu_acceleration": {
    "speedup": "3800x",
    "trials_per_second": 127.5
  }
}
```

### 11. risk_analysis

Comprehensive portfolio risk analysis with GPU acceleration.

**Parameters:**
- `portfolio` (array[object], required): Portfolio positions
- `time_horizon` (integer, optional): Analysis time horizon in days (default: 1)
- `var_confidence` (number, optional): VaR confidence level (default: 0.05)
- `use_monte_carlo` (boolean, optional): Use Monte Carlo simulation (default: true)
- `use_gpu` (boolean, optional): Enable GPU acceleration (default: true)

**Response:**
```json
{
  "risk_analysis_id": "risk_20240626_001",
  "portfolio_summary": {
    "total_value": 125000.00,
    "position_count": 8,
    "concentration_risk": "medium"
  },
  "var_analysis": {
    "var_95": -0.028,
    "var_99": -0.045,
    "expected_shortfall_95": -0.039,
    "expected_shortfall_99": -0.062
  },
  "monte_carlo_simulation": {
    "simulations": 100000,
    "worst_case_loss": -0.087,
    "probability_of_loss": 0.42,
    "expected_return": 0.012
  },
  "correlation_analysis": {
    "max_correlation": 0.78,
    "avg_correlation": 0.34,
    "diversification_ratio": 0.67
  },
  "neural_risk_modeling": {
    "tail_risk_forecast": -0.051,
    "volatility_forecast": 0.189,
    "confidence": 0.83
  },
  "processing_time_ms": 1890.3
}
```

### 12. execute_trade

Execute live trade with advanced order management.

**Parameters:**
- `strategy` (string, required): Strategy name
- `symbol` (string, required): Trading symbol
- `action` (string, required): "buy" or "sell"
- `quantity` (integer, required): Number of shares
- `order_type` (string, optional): Order type (default: "market")
- `limit_price` (number, optional): Limit price for limit orders

**Response:**
```json
{
  "trade_id": "trade_20240626_001",
  "status": "executed",
  "symbol": "AAPL",
  "action": "buy",
  "quantity": 50,
  "execution_price": 190.45,
  "total_value": 9522.50,
  "commission": 2.50,
  "execution_time": "2024-06-26T14:35:22Z",
  "strategy_context": {
    "strategy": "momentum_trading_optimized",
    "signal_strength": 0.78,
    "neural_forecast_support": 0.85
  },
  "risk_checks": {
    "position_size_check": "passed",
    "concentration_check": "passed",
    "risk_limit_check": "passed"
  }
}
```

### 13. performance_report

Generate detailed performance analytics report.

**Parameters:**
- `strategy` (string, required): Strategy name
- `period_days` (integer, optional): Analysis period in days (default: 30)
- `include_benchmark` (boolean, optional): Include benchmark comparison (default: true)
- `use_gpu` (boolean, optional): Enable GPU acceleration (default: false)

**Response:**
```json
{
  "report_id": "perf_20240626_001",
  "strategy": "momentum_trading_optimized",
  "period": {
    "start_date": "2024-05-27",
    "end_date": "2024-06-26",
    "trading_days": 30
  },
  "performance_metrics": {
    "total_return": 0.089,
    "daily_return_avg": 0.0029,
    "volatility": 0.147,
    "sharpe_ratio": 1.23,
    "max_drawdown": 0.034,
    "calmar_ratio": 2.62
  },
  "trade_analysis": {
    "total_trades": 12,
    "win_rate": 0.67,
    "avg_trade_duration": 2.3,
    "profit_factor": 1.94
  },
  "neural_forecast_impact": {
    "forecast_accuracy": 0.78,
    "signal_enhancement": 0.15,
    "risk_reduction": 0.023
  },
  "benchmark_comparison": {
    "benchmark": "sp500",
    "excess_return": 0.034,
    "information_ratio": 1.45,
    "tracking_error": 0.023
  }
}
```

### 14. correlation_analysis

Analyze asset correlations with GPU acceleration.

**Parameters:**
- `symbols` (array[string], required): List of symbols to analyze
- `period_days` (integer, optional): Analysis period in days (default: 90)
- `use_gpu` (boolean, optional): Enable GPU acceleration (default: true)

**Response:**
```json
{
  "analysis_id": "corr_20240626_001",
  "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA"],
  "correlation_matrix": {
    "AAPL": {"AAPL": 1.00, "GOOGL": 0.67, "MSFT": 0.72, "TSLA": 0.34},
    "GOOGL": {"AAPL": 0.67, "GOOGL": 1.00, "MSFT": 0.78, "TSLA": 0.42},
    "MSFT": {"AAPL": 0.72, "GOOGL": 0.78, "MSFT": 1.00, "TSLA": 0.38},
    "TSLA": {"AAPL": 0.34, "GOOGL": 0.42, "MSFT": 0.38, "TSLA": 1.00}
  },
  "rolling_correlations": {
    "avg_correlation": 0.57,
    "max_correlation": 0.89,
    "min_correlation": 0.12,
    "correlation_stability": 0.78
  },
  "neural_correlation_forecast": {
    "next_period_correlations": {
      "AAPL_GOOGL": 0.71,
      "AAPL_MSFT": 0.75,
      "confidence": 0.82
    }
  },
  "processing_time_ms": 456.7
}
```

### 15. run_benchmark

Run comprehensive benchmarks for strategy performance and system capabilities.

**Parameters:**
- `strategy` (string, required): Strategy name
- `benchmark_type` (string, optional): Type of benchmark (default: "performance")
- `use_gpu` (boolean, optional): Enable GPU acceleration (default: true)

**Response:**
```json
{
  "benchmark_id": "bench_20240626_001",
  "strategy": "momentum_trading_optimized",
  "benchmark_type": "performance",
  "system_performance": {
    "cpu_usage": 0.45,
    "memory_usage": 0.67,
    "gpu_utilization": 0.89,
    "inference_latency_ms": 7.8,
    "throughput_ops_sec": 1247.5
  },
  "strategy_benchmarks": {
    "sharpe_ratio_rank": 2,
    "return_rank": 3,
    "risk_rank": 1,
    "overall_score": 8.7
  },
  "neural_forecast_benchmarks": {
    "accuracy_score": 0.82,
    "latency_ms": 5.2,
    "confidence_calibration": 0.91
  },
  "gpu_acceleration_metrics": {
    "speedup_factor": 5200,
    "memory_efficiency": 0.78,
    "power_efficiency": 0.85
  }
}
```

## Error Handling

### Error Response Format

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32001,
    "message": "Model not found",
    "data": {
      "strategy": "invalid_strategy",
      "available_strategies": ["momentum_trading_optimized", "mean_reversion_optimized"]
    }
  },
  "id": null
}
```

### Error Codes

| Code | Name | Description |
|------|------|-------------|
| -32700 | Parse Error | Invalid JSON |
| -32600 | Invalid Request | Invalid request object |
| -32601 | Method Not Found | Method does not exist |
| -32602 | Invalid Params | Invalid method parameters |
| -32603 | Internal Error | Internal JSON-RPC error |
| -32001 | Model Not Found | Trading strategy not found |
| -32002 | Strategy Error | Strategy execution error |
| -32003 | Data Error | Market data error |
| -32004 | Auth Error | Authentication failed |

## Rate Limits

| Tool Category | Rate Limit | Burst Limit |
|---------------|------------|-------------|
| Market Data | 100/min | 20/sec |
| Trading Operations | 50/min | 10/sec |
| Analytics | 200/min | 30/sec |
| Optimization | 10/min | 2/sec |

## Authentication

MCP server supports multiple authentication methods:

```python
# API Key authentication
headers = {"Authorization": "Bearer your-api-key"}

# Session-based authentication
session = authenticate_session("username", "password")
```

## Integration Examples

### Claude Code Integration

```python
# Using Claude Code MCP integration
import asyncio

async def analyze_portfolio():
    # Get portfolio status
    portfolio = await mcp__ai_news_trader__get_portfolio_status(
        include_analytics=True
    )
    
    # Analyze news sentiment for each position
    for position in portfolio['positions']:
        sentiment = await mcp__ai_news_trader__analyze_news(
            symbol=position['symbol'],
            use_gpu=True
        )
        print(f"{position['symbol']}: {sentiment['sentiment_score']}")
    
    # Run risk analysis
    risk = await mcp__ai_news_trader__risk_analysis(
        portfolio=portfolio['positions'],
        use_gpu=True
    )
    
    return {
        'portfolio': portfolio,
        'risk': risk
    }

# Run analysis
results = asyncio.run(analyze_portfolio())
```

### Direct JSON-RPC Usage

```python
import json
import asyncio
import websockets

async def call_mcp_tool(method, params):
    request = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1
    }
    
    async with websockets.connect("ws://localhost:3000/mcp") as websocket:
        await websocket.send(json.dumps(request))
        response = await websocket.recv()
        return json.loads(response)

# Example usage
result = await call_mcp_tool("quick_analysis", {
    "symbol": "AAPL",
    "use_gpu": True
})
```

## Syndicate Management Tools (NEW)

The platform now includes 17 comprehensive syndicate management tools for collaborative investment:

### Member Management
- `syndicate_create` - Create new investment syndicate
- `syndicate_add_member` - Add members with roles and permissions
- `syndicate_update_member` - Update member roles or status
- `syndicate_get_members` - List all syndicate members

### Fund Management
- `syndicate_allocate_funds` - AI-driven fund allocation
- `syndicate_execute_bet` - Execute syndicate bets with risk checks
- `syndicate_get_positions` - Track active positions

### Profit Distribution
- `syndicate_calculate_distribution` - Calculate fair profit sharing
- `syndicate_process_distribution` - Execute distributions
- `syndicate_request_withdrawal` - Member withdrawal management

### Governance
- `syndicate_create_proposal` - Democratic voting proposals
- `syndicate_cast_vote` - Member voting system
- `syndicate_get_proposal_results` - Voting results

### Analytics
- `syndicate_member_performance` - Individual performance tracking
- `syndicate_performance_report` - Comprehensive syndicate analytics
- `syndicate_get_info` - Syndicate information
- `syndicate_list` - List available syndicates

For complete syndicate tools documentation, see [Syndicate Tools Guide](../mcp/SYNDICATE_TOOLS.md).

## See Also

- [Syndicate Tools Guide](../mcp/SYNDICATE_TOOLS.md) - Complete syndicate management
- [Neural Forecast API](neural_forecast.md) - AI forecasting tools
- [CLI Reference](cli_reference.md) - Command-line interface
- [MCP Integration Guide](../guides/mcp_integration.md) - Integration patterns
- [GPU Optimization Tutorial](../tutorials/gpu_optimization.md) - Performance optimization