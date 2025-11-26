# MCP Tools Documentation

## Overview

The AI News Trader platform provides a comprehensive suite of Model Context Protocol (MCP) tools for AI-powered trading, sports betting, and syndicate investment management. The platform currently offers **49 total MCP tools** across multiple categories.

## Tool Categories

### 1. Core Trading Tools (15 tools)
- Trading operations and portfolio management
- Market analysis and sentiment evaluation
- Backtesting and strategy optimization
- Risk analysis and performance reporting

### 2. Neural Forecasting Tools (6 tools)
- Neural network training and evaluation
- Time-series forecasting
- Model optimization and backtesting
- Multi-model ensemble creation

### 3. Prediction Markets Tools (6 tools)
- Market listing and analysis
- Order placement and management
- Expected value calculations
- Position tracking

### 4. News Collection Tools (4 tools)
- News provider management
- Filtered news fetching
- Trend analysis
- Real-time sentiment monitoring

### 5. Strategy Management Tools (4 tools)
- Strategy recommendations
- Performance comparison
- Adaptive selection
- Strategy switching

### 6. System Monitoring Tools (3 tools)
- Performance metrics
- Health monitoring
- Execution analytics

### 7. Multi-Asset Trading Tools (3 tools)
- Portfolio rebalancing
- Cross-asset correlation
- Multi-asset execution

### 8. Sports Betting Tools (10 tools)
- Event and odds retrieval
- Arbitrage detection
- Kelly Criterion calculations
- Betting portfolio management

### 9. Syndicate Management Tools (17 tools) - NEW
- Member management and roles
- Capital allocation and fund management
- Profit distribution systems
- Democratic voting
- Performance tracking
- Risk controls

## Quick Start

```python
# Using Claude Code integration
import asyncio

# Example: Create a syndicate
result = await mcp__ai_news_trader__syndicate_create(
    name="Elite Traders",
    initial_capital=100000,
    allocation_strategy="kelly_criterion"
)

# Example: Add member to syndicate
member = await mcp__ai_news_trader__syndicate_add_member(
    syndicate_id=result['syndicate_id'],
    name="John Smith",
    email="john@example.com",
    role="senior_analyst",
    contribution=50000
)
```

## Documentation Structure

- [Syndicate Tools Guide](SYNDICATE_TOOLS.md) - Detailed syndicate management tools
- [Trading Tools Reference](../api/mcp_tools.md) - Core trading operations
- [Neural Forecast Guide](../api/neural_forecast.md) - AI forecasting tools
- [Sports Betting Tools](../sports_betting_api_examples.md) - Betting platform integration

## Security Considerations

All MCP tools implement:
- Authentication and authorization
- Rate limiting
- Input validation
- Audit logging
- Secure credential management

## Integration Examples

See [MCP Syndicate Examples](../examples/mcp_syndicate_examples.py) for comprehensive usage examples.

## Syndicate Tools Summary

The 17 new syndicate management tools enable:

### 🏦 Investment Pool Management
- Create and manage investment syndicates
- Pool funds from multiple investors
- Automated fund allocation with AI optimization
- Real-time position and exposure tracking

### 👥 Member Management
- 5 distinct member roles with permissions
- Performance tracking and skill assessment
- Investment tier classification (Bronze to Platinum)
- Individual member analytics and reporting

### 💰 Capital & Risk Management
- Kelly Criterion and other allocation strategies
- Automated risk controls and position limits
- Stop-loss and exposure management
- Multi-level approval workflows

### 📊 Profit Distribution
- Multiple distribution models (hybrid, proportional, performance-based)
- Automated tax withholding by jurisdiction
- Scheduled and emergency withdrawals
- Transparent calculation breakdowns

### 🗳️ Democratic Governance
- Weighted voting based on contribution and performance
- Proposal management for strategy changes
- Voting history and audit trails
- Supermajority requirements for major decisions

### 📈 Performance Analytics
- Individual member performance metrics
- Syndicate-wide reporting and analysis
- Sport and strategy breakdowns
- Risk-adjusted return calculations

## Support

For questions or issues:
- Check the [Troubleshooting Guide](../guides/troubleshooting.md)
- Review [Best Practices](MCP_BEST_PRACTICES_SECURITY.md)
- See [Syndicate Examples](../examples/mcp_syndicate_examples.py)
- Contact the development team