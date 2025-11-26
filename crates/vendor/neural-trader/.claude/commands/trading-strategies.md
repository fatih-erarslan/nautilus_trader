# Trading Strategies with Native MCP Tools

## Overview
Develop, test, and optimize AI-powered trading strategies using Claude Code's native MCP integration. All strategy tools are available through natural language requests.

## Core Strategy MCP Tools

### Strategy Analysis & Simulation

#### mcp__ai-news-trader__quick_analysis
Real-time market analysis for trading decisions.

**Natural Language Examples:**
```
"Analyze AAPL market conditions"
"Quick technical analysis of TSLA with GPU"
"Check if MSFT is good to buy right now"
"Market analysis for my watchlist: AAPL, GOOGL, NVDA"
```

**Tool Parameters:**
- `symbol` (string): Trading symbol to analyze
- `use_gpu` (bool): Enable GPU acceleration

**Expected Return:**
```json
{
  "symbol": "AAPL",
  "recommendation": "hold",
  "technical_indicators": {
    "rsi": 58.3,
    "macd": "bullish",
    "support": 182.50,
    "resistance": 188.75
  },
  "sentiment": "positive",
  "volatility": "moderate"
}
```

#### mcp__ai-news-trader__simulate_trade
Simulate trades before execution.

**Natural Language Examples:**
```
"Simulate buying AAPL with momentum strategy"
"Test a sell trade for TSLA using mean reversion"
"Simulate portfolio rebalancing trades"
```

**Tool Parameters:**
- `strategy` (string): Trading strategy name
- `symbol` (string): Trading symbol
- `action` (string): "buy" or "sell"
- `use_gpu` (bool): GPU acceleration

### Strategy Testing & Optimization

#### mcp__ai-news-trader__run_backtest
Historical strategy validation.

**Natural Language Examples:**
```
"Backtest momentum strategy on AAPL for 2024"
"Test my trading strategy on Q1-Q2 data"
"Run comprehensive backtest with transaction costs"
"Compare strategy performance against S&P 500"
```

**Tool Parameters:**
- `strategy` (string): Strategy name
- `symbol` (string): Trading symbol
- `start_date` (string): Start date (YYYY-MM-DD)
- `end_date` (string): End date (YYYY-MM-DD)
- `benchmark` (string): Benchmark comparison
- `include_costs` (bool): Include transaction costs
- `use_gpu` (bool): GPU acceleration

**Expected Return:**
```json
{
  "strategy": "momentum",
  "performance": {
    "total_return": 0.342,
    "sharpe_ratio": 2.15,
    "max_drawdown": -0.087,
    "win_rate": 0.64
  },
  "vs_benchmark": {
    "alpha": 0.18,
    "beta": 0.92
  }
}
```

#### mcp__ai-news-trader__optimize_strategy
Optimize strategy parameters.

**Natural Language Examples:**
```
"Optimize momentum strategy parameters for best Sharpe ratio"
"Find optimal settings for mean reversion on TSLA"
"Optimize my strategy for minimum drawdown"
```

**Tool Parameters:**
- `strategy` (string): Strategy to optimize
- `symbol` (string): Trading symbol
- `parameter_ranges` (dict): Parameters to optimize
- `optimization_metric` (string): Metric to optimize
- `max_iterations` (int): Maximum iterations
- `use_gpu` (bool): GPU acceleration

### Performance Analysis Tools

#### mcp__ai-news-trader__performance_report
Generate detailed performance analytics.

**Natural Language Examples:**
```
"Generate performance report for momentum strategy"
"Show me last 30 days performance with benchmarks"
"Detailed analytics for all my strategies"
"Performance breakdown by market conditions"
```

**Tool Parameters:**
- `strategy` (string): Strategy name
- `period_days` (int): Analysis period (default: 30)
- `include_benchmark` (bool): Include benchmark comparison
- `use_gpu` (bool): GPU acceleration

**Expected Return:**
```json
{
  "strategy": "momentum",
  "period": "30_days",
  "metrics": {
    "total_return": 0.0523,
    "sharpe_ratio": 2.34,
    "sortino_ratio": 3.12,
    "max_drawdown": -0.0312,
    "trades_executed": 47,
    "win_rate": 0.66
  },
  "daily_returns": [...],
  "benchmark_comparison": {
    "excess_return": 0.0234,
    "information_ratio": 1.85
  }
}
```

#### mcp__ai-news-trader__risk_analysis
Comprehensive portfolio risk assessment.

**Natural Language Examples:**
```
"Analyze risk for my current portfolio"
"Calculate VaR with 95% confidence"
"Run Monte Carlo risk simulation"
"Show me all risk metrics with GPU acceleration"
```

**Tool Parameters:**
- `portfolio` (list): Portfolio positions
- `var_confidence` (float): VaR confidence level
- `time_horizon` (int): Time horizon in days
- `use_monte_carlo` (bool): Use Monte Carlo simulation
- `use_gpu` (bool): GPU acceleration

### Strategy Benchmarking

#### mcp__ai-news-trader__run_benchmark
Benchmark strategy performance and capabilities.

**Natural Language Examples:**
```
"Benchmark all my trading strategies"
"Compare momentum strategy against market"
"Run performance benchmarks with GPU"
"Test strategy speed and accuracy"
```

**Tool Parameters:**
- `strategy` (string): Strategy to benchmark
- `benchmark_type` (string): "performance" or "system"
- `use_gpu` (bool): GPU acceleration

**Available Strategies:**
- **momentum**: Trend-following strategies
- **mean_reversion**: Contrarian strategies
- **neural_enhanced**: AI-driven strategies
- **arbitrage**: Market neutral strategies
- **swing**: Multi-timeframe trading
- **all**: Test all strategies

#### mcp__ai-news-trader__correlation_analysis
Analyze correlations between assets.

**Natural Language Examples:**
```
"Analyze correlation between AAPL and MSFT"
"Check my portfolio correlations"
"Find uncorrelated assets to TSLA"
"Correlation matrix for tech stocks"
```

**Tool Parameters:**
- `symbols` (list): List of symbols to analyze
- `period_days` (int): Analysis period (default: 90)
- `use_gpu` (bool): GPU acceleration

## Trading Execution Tools

### mcp__ai-news-trader__execute_trade
Execute trades with advanced order management.

**Natural Language Examples:**
```
"Buy 100 shares of AAPL using momentum strategy"
"Place a limit order for TSLA at $240"
"Execute portfolio rebalancing trades"
"Sell half my NVDA position"
```

**Tool Parameters:**
- `strategy` (string): Trading strategy
- `symbol` (string): Trading symbol
- `action` (string): "buy" or "sell"
- `quantity` (int): Number of shares
- `order_type` (string): "market" or "limit"
- `limit_price` (float, optional): Limit price

**Example Return:**
```json
{
  "order_id": "ORD_20241215_1234",
  "status": "executed",
  "filled_price": 185.75,
  "filled_quantity": 100,
  "commission": 0.65,
  "timestamp": "2024-12-15T10:30:45Z"
}
```

### mcp__ai-news-trader__get_portfolio_status
Get current portfolio with analytics.

**Natural Language Examples:**
```
"Show my current portfolio"
"Portfolio status with risk metrics"
"What's my portfolio performance?"
"Detailed portfolio analytics"
```

## Practical Trading Workflows

### Daily Trading Routine
```
"Good morning! Please help with my daily trading:
1. Check portfolio status with analytics
2. Run quick analysis on my watchlist
3. Generate neural forecasts for top movers
4. Simulate any recommended trades
5. Show risk metrics for proposed changes"
```

### Strategy Development Workflow
```
"I want to develop a new momentum strategy:
1. Analyze historical momentum patterns in tech stocks
2. Backtest on 2024 data with AAPL, MSFT, GOOGL
3. Optimize parameters for best Sharpe ratio
4. Compare performance against buy-and-hold
5. Generate detailed performance report"
```

### Risk Management Workflow
```
"Analyze my portfolio risk:
1. Calculate current portfolio correlations
2. Run Monte Carlo risk simulation
3. Identify concentration risks
4. Suggest diversification options
5. Show VaR at 95% confidence"
```

### Multi-Strategy Comparison
```
"Compare trading strategies for TSLA:
1. Backtest momentum vs mean reversion
2. Include transaction costs
3. Test across different market conditions
4. Show risk-adjusted returns
5. Recommend best strategy"
```

## Strategy Performance Metrics

### Key Performance Indicators
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Total Return**: Overall profitability
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of winning to losing trades
- **Calmar Ratio**: Annual return divided by maximum drawdown

### Risk Metrics
- **Value at Risk (VaR)**: Potential loss at confidence level
- **Expected Shortfall**: Average loss beyond VaR
- **Beta**: Correlation with market movements
- **Volatility**: Standard deviation of returns
- **Correlation**: Relationship between assets

## Complete MCP Trading Tools Reference

### Strategy Tools (11 tools)
1. **quick_analysis** - Real-time market analysis
2. **simulate_trade** - Trade simulation
3. **execute_trade** - Order execution
4. **get_portfolio_status** - Portfolio analytics
5. **run_backtest** - Historical testing
6. **optimize_strategy** - Parameter optimization
7. **performance_report** - Performance analytics
8. **risk_analysis** - Risk assessment
9. **correlation_analysis** - Asset correlations
10. **run_benchmark** - Strategy benchmarking
11. **list_strategies** - Available strategies

### Integration Examples
```
# Morning routine
"Start my trading day:
1. Use get_portfolio_status for current positions
2. Use quick_analysis on watchlist stocks
3. Use analyze_news for market sentiment
4. Use neural_forecast for price predictions
5. Use simulate_trade for any opportunities"

# Strategy testing
"Test a new strategy:
1. Use run_backtest on historical data
2. Use optimize_strategy for best parameters
3. Use risk_analysis on results
4. Use performance_report for metrics
5. Use run_benchmark against market"
```

## Best Practices for MCP Trading Tools

### Natural Language Best Practices
1. **Be specific about goals:**
   - ✅ "Optimize for maximum Sharpe ratio"
   - ❌ "Optimize my strategy"

2. **Include timeframes:**
   - ✅ "Backtest on Q1 2024 data"
   - ❌ "Backtest my strategy"

3. **Specify risk preferences:**
   - ✅ "Minimize drawdown while maintaining 15% returns"
   - ❌ "Make it profitable"

### Common Trading Patterns

#### Research Before Trading
```
"Before trading NVDA:
1. Analyze recent news sentiment
2. Check technical indicators
3. Generate neural forecast
4. Review correlation with my portfolio
5. Simulate the trade first"
```

#### Progressive Strategy Testing
```
"Test momentum strategy progressively:
1. Start with single stock (AAPL)
2. If profitable, test on sector (tech)
3. Optimize parameters on full dataset
4. Validate on out-of-sample period
5. Compare with other strategies"
```

#### Risk-First Approach
```
"Evaluate new position:
1. Check portfolio correlation impact
2. Calculate position size based on volatility
3. Set stop-loss levels
4. Assess total portfolio risk
5. Only then execute if acceptable"
```

### Tips for Success
1. **Start simple:** Test one strategy on one stock first
2. **Use simulations:** Always simulate before executing
3. **Check correlations:** Avoid concentration risk
4. **Monitor continuously:** Set up regular performance reviews
5. **Document decisions:** Keep track of strategy changes

### Error Prevention
- Always specify dates for backtests
- Include transaction costs in simulations
- Validate data quality before training
- Set reasonable position limits
- Use multiple confirmation signals