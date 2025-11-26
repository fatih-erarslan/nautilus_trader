# MCP Tools Quick Reference

**Total: 97 Tools | 17 Categories | MCP 2025-11 Compliant**

## All Tools by Name (Alphabetical)

### Trading & Strategy (9 tools)
1. `adaptive_strategy_selection`
2. `execute_trade`
3. `get_strategy_comparison`
4. `get_strategy_info`
5. `list_strategies`
6. `optimize_strategy`
7. `recommend_strategy`
8. `run_backtest`
9. `simulate_trade`

### Analysis & Risk (5 tools)
1. `correlation_analysis`
2. `cross_asset_correlation_matrix`
3. `quick_analysis`
4. `risk_analysis`
5. `run_backtest` (also in trading)

### Neural Network Tools (6 tools)
1. `neural_backtest`
2. `neural_evaluate`
3. `neural_forecast`
4. `neural_model_status`
5. `neural_optimize`
6. `neural_train`

### News & Sentiment (6 tools)
1. `analyze_news`
2. `control_news_collection`
3. `fetch_filtered_news`
4. `get_news_provider_status`
5. `get_news_sentiment`
6. `get_news_trends`

### Sports Betting (10 tools)
1. `analyze_betting_market_depth`
2. `calculate_expected_value_tool`
3. `calculate_kelly_criterion`
4. `compare_betting_providers`
5. `execute_sports_bet`
6. `find_sports_arbitrage`
7. `get_betting_portfolio_status`
8. `get_sports_betting_performance`
9. `get_sports_events`
10. `get_sports_odds`
11. `simulate_betting_strategy`

### Odds API Tools (9 tools)
1. `odds_api_analyze_movement`
2. `odds_api_calculate_probability`
3. `odds_api_compare_margins`
4. `odds_api_find_arbitrage`
5. `odds_api_get_bookmaker_odds`
6. `odds_api_get_event_odds`
7. `odds_api_get_live_odds`
8. `odds_api_get_sports`
9. `odds_api_get_upcoming`

### Prediction Markets (6 tools)
1. `analyze_market_sentiment_tool`
2. `calculate_expected_value_tool`
3. `get_market_orderbook_tool`
4. `get_prediction_markets_tool`
5. `get_prediction_positions_tool`
6. `place_prediction_order_tool`

### Syndicate Management (19 tools)
1. `add_syndicate_member`
2. `allocate_syndicate_funds`
3. `calculate_syndicate_tax_liability`
4. `cast_syndicate_vote`
5. `create_syndicate`
6. `create_syndicate_tool`
7. `create_syndicate_vote`
8. `distribute_syndicate_profits`
9. `get_syndicate_allocation_limits`
10. `get_syndicate_member_list`
11. `get_syndicate_member_performance`
12. `get_syndicate_profit_history`
13. `get_syndicate_status`
14. `get_syndicate_status_tool`
15. `get_syndicate_withdrawal_history`
16. `process_syndicate_withdrawal`
17. `simulate_syndicate_allocation`
18. `update_syndicate_allocation_strategy`
19. `update_syndicate_member_contribution`

### E2B Cloud (10 tools)
1. `create_e2b_sandbox`
2. `deploy_e2b_template`
3. `execute_e2b_process`
4. `export_e2b_template`
5. `get_e2b_sandbox_status`
6. `list_e2b_sandboxes`
7. `monitor_e2b_health`
8. `run_e2b_agent`
9. `scale_e2b_deployment`
10. `terminate_e2b_sandbox`

### E2B Swarm (8 tools)
1. `deploy_trading_agent`
2. `execute_swarm_strategy`
3. `get_swarm_metrics`
4. `get_swarm_status`
5. `init_e2b_swarm`
6. `monitor_swarm_health`
7. `scale_swarm`
8. `shutdown_swarm`

### Multi-Asset Trading (1 tool)
1. `execute_multi_asset_trade`

### Portfolio Management (2 tools)
1. `get_portfolio_status`
2. `portfolio_rebalance`

### Analytics & Metrics (2 tools)
1. `get_execution_analytics`
2. `performance_report`

### System & Monitoring (4 tools)
1. `get_system_metrics`
2. `monitor_strategy_health`
3. `ping`
4. `switch_active_strategy`

---

## Tools by Performance Category

### GPU-Accelerated (10 tools)
- neural_train
- neural_optimize
- neural_forecast
- neural_backtest
- neural_evaluate
- correlation_analysis
- cross_asset_correlation_matrix
- quick_analysis
- get_sports_odds
- run_backtest

### High-Performance (25 tools)
- All neural tools (6)
- Trading tools (9)
- Swarm tools (8)
- Sports betting arbitrage (2)

### Standard Performance (62 tools)
- All remaining tools

---

## Tools by Cost Level

### Very High Cost
- neural_train
- neural_optimize

### High Cost
- execute_trade
- execute_multi_asset_trade
- run_backtest
- deploy_trading_agent

### Medium Cost
- Most odds and betting tools
- Syndicate management
- E2B deployment

### Low Cost
- All query/list tools
- Status checks
- ping

---

## Quick Search Index

### Tools Starting with...
- **a**: adaptive_strategy_selection, add_syndicate_member, allocate_syndicate_funds, analyze_betting_market_depth, analyze_market_sentiment_tool, analyze_news
- **c**: calculate_expected_value_tool, calculate_kelly_criterion, calculate_syndicate_tax_liability, cast_syndicate_vote, compare_betting_providers, control_news_collection, correlation_analysis, create_e2b_sandbox, create_syndicate (and variants), cross_asset_correlation_matrix
- **d**: deploy_e2b_template, deploy_trading_agent, distribute_syndicate_profits
- **e**: execute_e2b_process, execute_multi_asset_trade, execute_sports_bet, execute_swarm_strategy, execute_trade, export_e2b_template
- **f**: fetch_filtered_news, find_sports_arbitrage
- **g**: get_betting_portfolio_status, get_e2b_sandbox_status, get_execution_analytics, get_market_orderbook_tool, get_news_provider_status, get_news_sentiment, get_news_trends, get_portfolio_status, get_prediction_markets_tool, get_prediction_positions_tool, get_sports_betting_performance, get_sports_events, get_sports_odds, get_strategy_comparison, get_strategy_info, get_swarm_metrics, get_swarm_status, get_syndicate_* (7 tools), get_system_metrics
- **i**: init_e2b_swarm
- **l**: list_e2b_sandboxes, list_strategies
- **m**: monitor_e2b_health, monitor_strategy_health, monitor_swarm_health
- **n**: neural_backtest, neural_evaluate, neural_forecast, neural_model_status, neural_optimize, neural_train
- **o**: odds_api_* (9 tools), optimize_strategy
- **p**: performance_report, ping, place_prediction_order_tool, portfolio_rebalance, process_syndicate_withdrawal
- **q**: quick_analysis
- **r**: recommend_strategy, risk_analysis, run_backtest, run_benchmark, run_e2b_agent
- **s**: scale_e2b_deployment, scale_swarm, shutdown_swarm, simulate_betting_strategy, simulate_syndicate_allocation, simulate_trade, switch_active_strategy
- **t**: terminate_e2b_sandbox
- **u**: update_syndicate_allocation_strategy, update_syndicate_member_contribution

---

## MCP Methods

### Core Methods
- `initialize` - Initialize MCP protocol
- `tools/list` - List all available tools
- `tools/call` - Execute a tool with arguments
- `tools/schema` - Get tool JSON schema
- `tools/search` - Search tools by query
- `tools/categories` - List tool categories

### Protocol Version
- MCP 2025-11

### Transport
- stdio (default)

---

## Configuration

### Start Server
```bash
npx neural-trader mcp
```

### Options
```
--transport <type>    Transport: stdio (default)
--port <number>       Port: 3000 (default)
--host <address>      Host: localhost (default)
--stub               Run in stub/test mode
--no-rust            Disable Rust NAPI bridge
--no-audit           Disable audit logging
--help               Show help
```

### Environment Variables
```
NEURAL_TRADER_API_KEY    Trader API key
ALPACA_API_KEY          Trading platform key
ALPACA_SECRET_KEY       Trading platform secret
```

---

## Integration with Claude Desktop

Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "neural-trader": {
      "command": "npx",
      "args": ["neural-trader", "mcp"]
    }
  }
}
```

---

## Tool Statistics

- **Total Tools**: 97
- **Categories**: 17
- **GPU-Capable**: 10
- **Avg Parameters**: 4-6 per tool
- **Compliance**: MCP 2025-11
- **Schema Format**: JSON Schema draft 2020-12

---

Generated: 2025-11-17
Status: Production Ready ✅
