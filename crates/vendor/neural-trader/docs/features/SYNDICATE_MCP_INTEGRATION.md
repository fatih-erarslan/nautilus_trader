# Syndicate MCP Integration Guide

## Overview

The AI News Trader Syndicate System has been successfully integrated into the MCP (Model Context Protocol) server, exposing 17 syndicate investment tools for collaborative trading and investment management.

## Integration Details

### Files Modified
1. **`src/mcp/mcp_server_enhanced.py`**
   - Added syndicate tools import section
   - Integrated 17 syndicate MCP tools
   - Updated tool count to 62 total tools

2. **`src/syndicate/syndicate_tools.py`** (created)
   - Implements all syndicate functionality as callable tools
   - Handles syndicate creation, member management, fund allocation, and profit distribution
   - Provides risk management and performance tracking

### Available Syndicate Tools

#### Core Management Tools
1. **`create_syndicate_tool`** - Create a new investment syndicate
2. **`add_syndicate_member`** - Add members with roles and contributions
3. **`get_syndicate_status_tool`** - Get current syndicate statistics
4. **`get_syndicate_member_list`** - List all syndicate members

#### Capital Management Tools
5. **`allocate_syndicate_funds`** - Allocate funds using advanced strategies (Kelly Criterion, etc.)
6. **`distribute_syndicate_profits`** - Distribute profits using various models (hybrid, proportional, etc.)
7. **`process_syndicate_withdrawal`** - Handle member withdrawal requests
8. **`update_syndicate_member_contribution`** - Update member capital contributions

#### Performance & Analytics Tools
9. **`get_syndicate_member_performance`** - Get detailed member performance metrics
10. **`get_syndicate_profit_history`** - View profit distribution history
11. **`get_syndicate_withdrawal_history`** - Track withdrawal history
12. **`calculate_syndicate_tax_liability`** - Estimate tax obligations

#### Risk Management Tools
13. **`get_syndicate_allocation_limits`** - View current risk limits and constraints
14. **`update_syndicate_allocation_strategy`** - Modify allocation parameters
15. **`simulate_syndicate_allocation`** - Test allocation strategies before execution

#### Voting System Tools (Placeholder)
16. **`create_syndicate_vote`** - Create member votes (future implementation)
17. **`cast_syndicate_vote`** - Cast votes on proposals (future implementation)

## Usage Examples

### Creating a Syndicate
```python
# MCP tool call
{
  "tool": "create_syndicate_tool",
  "arguments": {
    "syndicate_id": "elite-traders-001",
    "name": "Elite Trading Syndicate",
    "description": "Professional sports betting syndicate"
  }
}
```

### Adding Members
```python
# MCP tool call
{
  "tool": "add_syndicate_member",
  "arguments": {
    "syndicate_id": "elite-traders-001",
    "name": "Alice Johnson",
    "email": "alice@syndicate.com",
    "role": "lead_investor",
    "initial_contribution": 100000
  }
}
```

### Allocating Funds
```python
# MCP tool call
{
  "tool": "allocate_syndicate_funds",
  "arguments": {
    "syndicate_id": "elite-traders-001",
    "opportunities": [
      {
        "sport": "NBA",
        "event": "Lakers vs Celtics",
        "bet_type": "spread",
        "selection": "Lakers -3.5",
        "odds": 1.91,
        "probability": 0.58,
        "edge": 0.05,
        "confidence": 0.75,
        "model_agreement": 0.82,
        "hours_until_event": 24,
        "liquidity": 50000
      }
    ],
    "strategy": "kelly_criterion"
  }
}
```

## Member Roles

- **`lead_investor`** - Full control, veto power, profit distribution
- **`senior_analyst`** - Strategy development, model creation
- **`junior_analyst`** - Analysis support, bet proposals
- **`contributing_member`** - Regular investor, voting rights
- **`observer`** - View-only access, no voting

## Allocation Strategies

- **`kelly_criterion`** - Mathematical optimal sizing (fractional Kelly)
- **`fixed_percentage`** - Consistent 2% base allocation
- **`dynamic_confidence`** - Tiered based on confidence levels
- **`risk_parity`** - Equal risk contribution across bets
- **`martingale`** - Progressive betting (use with caution)

## Distribution Models

- **`hybrid`** - 50% capital, 30% performance, 20% equal (default)
- **`proportional`** - Based on capital contribution
- **`performance_weighted`** - Based on ROI and win rate
- **`tiered`** - Based on member tier status

## Risk Limits

- **Single Bet**: Max 5% of bankroll (2% for parlays)
- **Daily Exposure**: Max 20% of total capital
- **Sport Concentration**: Max 40% in one sport
- **Cash Reserve**: Minimum 30% liquidity
- **Stop Losses**: 10% daily, 20% weekly

## Error Handling

All syndicate tools include comprehensive error handling:
- Invalid syndicate IDs return error with "syndicate not found"
- Invalid member roles return error with available roles
- Insufficient funds return error with available capital
- Risk limit violations return error with specific limit exceeded

## Testing

To test the syndicate integration:

1. Start the MCP server:
   ```bash
   python src/mcp/mcp_server_enhanced.py
   ```

2. Use Claude or another MCP client to call syndicate tools

3. Monitor logs for any errors or warnings

## Future Enhancements

1. **Voting System**: Full implementation of democratic decision-making
2. **Performance Analytics**: Advanced ML-based performance prediction
3. **Tax Integration**: Automated tax form generation and filing
4. **Multi-Currency**: Support for international syndicates
5. **Audit Trail**: Blockchain-based immutable transaction history

## Troubleshooting

### Common Issues

1. **"Syndicate tools not available"**
   - Ensure syndicate modules are properly installed
   - Check Python path includes src directory

2. **"Module not found" errors**
   - Verify all dependencies are installed
   - Check for missing enum definitions (VoteType, MarketType)

3. **Allocation failures**
   - Verify sufficient capital in syndicate
   - Check risk limits aren't exceeded
   - Ensure valid betting opportunities provided

## Support

For issues or questions about the syndicate integration:
1. Check the main syndicate documentation: `/docs/SYNDICATE_SYSTEM.md`
2. Review error logs in the MCP server output
3. Ensure all syndicate members have valid roles and contributions