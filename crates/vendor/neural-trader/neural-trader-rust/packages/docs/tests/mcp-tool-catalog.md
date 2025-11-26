# MCP Tool Catalog - Neural Trader

**Last Updated**: 2025-11-14
**Total Tools**: 102+

## Tool Categories

### 1. Core System (5 tools)
- `ping` - Server health check
- `list_strategies` - List all trading strategies
- `get_strategy_info` - Get detailed strategy information
- `features_detect` - Detect runtime capabilities (WASM, SIMD)
- `benchmark_run` - Execute performance benchmarks

### 2. Market Data & Analysis (8 tools)
- `quick_analysis` - Quick market analysis for symbols
- `analyze_news` - AI sentiment analysis of market news
- `get_news_sentiment` - Real-time news sentiment
- `correlation_analysis` - Asset correlation analysis
- `cross_asset_correlation_matrix` - Multi-asset correlation matrix
- `control_news_collection` - Control news fetching
- `get_news_provider_status` - News provider status
- `fetch_filtered_news` - Advanced news filtering
- `get_news_trends` - Multi-interval trend analysis

### 3. Trading Execution (7 tools)
- `simulate_trade` - Simulate trading operations
- `execute_trade` - Execute live trades
- `execute_multi_asset_trade` - Multi-asset execution
- `get_portfolio_status` - Portfolio status and analytics
- `portfolio_rebalance` - Calculate rebalancing
- `switch_active_strategy` - Switch between strategies
- `adaptive_strategy_selection` - Auto-select best strategy

### 4. Risk Management (4 tools)
- `risk_analysis` - Comprehensive portfolio risk analysis
- `calculate_kelly_criterion` - Kelly Criterion bet sizing
- `calculate_expected_value_tool` - Expected value calculations
- `monitor_strategy_health` - Strategy health monitoring

### 5. Backtesting & Optimization (3 tools)
- `run_backtest` - Historical backtesting with GPU
- `optimize_strategy` - Parameter optimization
- `performance_report` - Performance analytics

### 6. Neural Networks (9 tools)
- `neural_forecast` - Generate forecasts
- `neural_train` - Train models
- `neural_evaluate` - Evaluate models
- `neural_backtest` - Backtest neural models
- `neural_model_status` - Model status
- `neural_optimize` - Optimize hyperparameters
- `neural_list_templates` - List templates
- `neural_deploy_template` - Deploy templates
- `neural_validation_workflow` - Validation workflows

### 7. Sports Betting (17 tools)
- `get_sports_events` - Upcoming sports events
- `get_sports_odds` - Real-time betting odds
- `find_sports_arbitrage` - Arbitrage opportunities
- `analyze_betting_market_depth` - Market liquidity
- `calculate_kelly_criterion` - Bet sizing
- `simulate_betting_strategy` - Strategy simulation
- `get_betting_portfolio_status` - Portfolio status
- `execute_sports_bet` - Execute bets
- `get_sports_betting_performance` - Performance analytics
- `compare_betting_providers` - Provider comparison
- `odds_api_get_sports` - List sports
- `odds_api_get_live_odds` - Live odds
- `odds_api_get_event_odds` - Event details
- `odds_api_find_arbitrage` - Find arbitrage
- `odds_api_get_bookmaker_odds` - Bookmaker odds
- `odds_api_analyze_movement` - Odds movement
- `odds_api_compare_margins` - Margin comparison

### 8. Betting Syndicates (16 tools)
- `create_syndicate_tool` - Create syndicate
- `add_syndicate_member` - Add members
- `get_syndicate_status_tool` - Syndicate status
- `allocate_syndicate_funds` - Fund allocation
- `distribute_syndicate_profits` - Profit distribution
- `process_syndicate_withdrawal` - Process withdrawals
- `get_syndicate_member_performance` - Member metrics
- `create_syndicate_vote` - Create votes
- `cast_syndicate_vote` - Cast votes
- `get_syndicate_allocation_limits` - Allocation limits
- `update_syndicate_member_contribution` - Update contributions
- `get_syndicate_profit_history` - Profit history
- `simulate_syndicate_allocation` - Simulate allocation
- `get_syndicate_withdrawal_history` - Withdrawal history
- `update_syndicate_allocation_strategy` - Update strategy
- `get_syndicate_member_list` - List members
- `calculate_syndicate_tax_liability` - Tax calculations

### 9. Prediction Markets (6 tools)
- `get_prediction_markets_tool` - List markets
- `analyze_market_sentiment_tool` - Market sentiment
- `get_market_orderbook_tool` - Orderbook data
- `place_prediction_order_tool` - Place orders
- `get_prediction_positions_tool` - Current positions
- `calculate_expected_value_tool` - EV calculations

### 10. E2B Sandbox Integration (10 tools)
- `create_e2b_sandbox` - Create sandbox
- `run_e2b_agent` - Run trading agents
- `execute_e2b_process` - Execute processes
- `list_e2b_sandboxes` - List sandboxes
- `terminate_e2b_sandbox` - Terminate sandbox
- `get_e2b_sandbox_status` - Sandbox status
- `deploy_e2b_template` - Deploy templates
- `scale_e2b_deployment` - Scale deployment
- `monitor_e2b_health` - Health monitoring
- `export_e2b_template` - Export templates

### 11. Strategy Management (5 tools)
- `recommend_strategy` - Strategy recommendations
- `switch_active_strategy` - Switch strategies
- `get_strategy_comparison` - Compare strategies
- `adaptive_strategy_selection` - Auto-selection
- `monitor_strategy_health` - Health monitoring

### 12. System Monitoring (5 tools)
- `get_system_metrics` - System performance
- `monitor_strategy_health` - Strategy health
- `get_execution_analytics` - Execution analytics
- `memory_usage` - Memory statistics
- `performance_report` - Performance reports

## Tool Usage Examples

### Quick Market Analysis
\`\`\`javascript
{
  "name": "quick_analysis",
  "arguments": {
    "symbol": "AAPL",
    "use_gpu": false
  }
}
\`\`\`

### Execute Trade
\`\`\`javascript
{
  "name": "execute_trade",
  "arguments": {
    "strategy": "momentum",
    "symbol": "AAPL",
    "action": "buy",
    "quantity": 10,
    "order_type": "market"
  }
}
\`\`\`

### Sports Betting Arbitrage
\`\`\`javascript
{
  "name": "find_sports_arbitrage",
  "arguments": {
    "sport": "americanfootball_nfl",
    "min_profit_margin": 0.01
  }
}
\`\`\`

### Neural Forecast
\`\`\`javascript
{
  "name": "neural_forecast",
  "arguments": {
    "symbol": "AAPL",
    "horizon": 5,
    "use_gpu": true
  }
}
\`\`\`

### Kelly Criterion
\`\`\`javascript
{
  "name": "calculate_kelly_criterion",
  "arguments": {
    "probability": 0.55,
    "odds": 2.0,
    "bankroll": 1000
  }
}
\`\`\`

### Risk Analysis
\`\`\`javascript
{
  "name": "risk_analysis",
  "arguments": {
    "portfolio": [
      { "symbol": "AAPL", "weight": 0.3 },
      { "symbol": "GOOGL", "weight": 0.3 },
      { "symbol": "MSFT", "weight": 0.4 }
    ],
    "use_gpu": true
  }
}
\`\`\`

## Performance Characteristics

### Fast Tools (<100ms)
- `ping`
- `list_strategies`
- `features_detect`
- `calculate_kelly_criterion`
- `get_portfolio_status`

### Medium Tools (100ms-1s)
- `quick_analysis`
- `simulate_trade`
- `get_news_sentiment`
- `neural_model_status`
- `get_sports_odds`

### Intensive Tools (1s-5s)
- `run_backtest`
- `risk_analysis`
- `neural_train`
- `optimize_strategy`
- `correlation_analysis`

## GPU Acceleration Support

The following tools support GPU acceleration via the `use_gpu` parameter:

- `quick_analysis`
- `simulate_trade`
- `run_backtest`
- `optimize_strategy`
- `risk_analysis`
- `neural_train`
- `neural_forecast`
- `neural_backtest`
- `correlation_analysis`
- `get_sports_events`
- `find_sports_arbitrage`

## Error Handling

All tools return errors in the following format:

\`\`\`json
{
  "content": [{
    "type": "text",
    "text": "{\\"error\\": \\"Error message\\", \\"code\\": \\"ERROR_CODE\\"}"
  }]
}
\`\`\`

## Rate Limits

### External APIs
- **The Odds API**: 500 requests/month (free tier)
- **Market Data**: Varies by provider
- **News APIs**: Varies by provider

### Internal Operations
- **MCP Server**: No hard limits (system resources)
- **GPU Operations**: Limited by hardware
- **Concurrent Requests**: Supports 50+ concurrent calls

## Authentication

Most tools require no authentication when using the MCP server in development mode. For production use:

1. Set environment variables in `.env`
2. Configure API keys for external services
3. Enable authentication in MCP server settings

## Tool Discovery

To list all available tools via MCP:

\`\`\`javascript
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}
\`\`\`

## Support

For issues or feature requests:
- GitHub: https://github.com/ruvnet/neural-trader
- Documentation: https://neural-trader.ruv.io

---

*Last updated: 2025-11-14*
*Neural Trader v1.0.6*
