# Syndicate MCP Tools Documentation

This document provides comprehensive documentation for all 17 syndicate management tools available through the MCP (Model Context Protocol) server.

## Overview

The syndicate MCP tools provide a complete system for managing professional sports betting syndicates, including:
- Member management and roles
- Capital allocation and bankroll management
- Profit distribution
- Democratic governance through voting
- Performance analytics and reporting
- Risk management and compliance

## Tool Categories

### 1. Member Management Tools (5 tools)
- `syndicate_create` - Create a new trading syndicate
- `syndicate_add_member` - Add a member to a syndicate
- `syndicate_update_member` - Update member details or status
- `syndicate_member_performance` - Get member performance metrics
- `syndicate_list_members` - List all syndicate members

### 2. Capital Management Tools (5 tools)
- `syndicate_allocate_funds` - Allocate funds for a betting opportunity
- `syndicate_get_exposure` - Get current capital exposure
- `syndicate_distribute_profits` - Distribute profits to members
- `syndicate_process_withdrawal` - Process member withdrawal request
- `syndicate_get_balance` - Get syndicate balance and financial status

### 3. Governance Tools (4 tools)
- `syndicate_create_vote` - Create a governance vote
- `syndicate_cast_vote` - Cast a vote on a proposal
- `syndicate_vote_results` - Get voting results
- `syndicate_get_rules` - Get syndicate governance rules

### 4. Analytics Tools (3 tools)
- `syndicate_performance_report` - Generate comprehensive performance report
- `syndicate_risk_analysis` - Perform risk analysis
- `syndicate_tax_report` - Generate tax report for syndicate

## Detailed Tool Documentation

### Member Management Tools

#### syndicate_create
Creates a new trading syndicate with initial configuration.

**Parameters:**
- `name` (string, required): Syndicate name
- `description` (string, required): Syndicate description
- `initial_capital` (number, required): Initial capital amount
- `bankroll_rules` (object, optional): Custom bankroll management rules
- `distribution_model` (string, optional): Profit distribution model (default: "hybrid")

**Response:**
```json
{
  "syndicate_id": "uuid-string",
  "name": "Alpha Trading Syndicate",
  "status": "active",
  "created_at": "2024-01-20T10:00:00Z",
  "initial_capital": "100000.00",
  "current_capital": "100000.00",
  "distribution_model": "hybrid",
  "bankroll_rules": {}
}
```

#### syndicate_add_member
Adds a new member to an existing syndicate.

**Parameters:**
- `syndicate_id` (string, required): Syndicate ID
- `name` (string, required): Member name
- `email` (string, required): Member email
- `role` (string, required): Member role (lead_investor, senior_analyst, junior_analyst, contributing_member, observer)
- `initial_contribution` (number, required): Initial capital contribution

**Response:**
```json
{
  "member_id": "uuid-string",
  "name": "John Doe",
  "role": "senior_analyst",
  "tier": "gold",
  "permissions": {
    "propose_bets": true,
    "vote_on_strategy": true,
    "manage_members": false
  },
  "capital_contribution": "25000.00",
  "joined_date": "2024-01-20T10:00:00Z"
}
```

### Capital Management Tools

#### syndicate_allocate_funds
Allocates funds for a betting opportunity using advanced algorithms.

**Parameters:**
- `syndicate_id` (string, required): Syndicate ID
- `sport` (string, required): Sport type
- `event` (string, required): Event description
- `bet_type` (string, required): Bet type
- `selection` (string, required): Selection
- `odds` (number, required): Betting odds
- `probability` (number, required): Win probability (0-1)
- `edge` (number, required): Expected edge
- `confidence` (number, required): Confidence level (0-1)
- `strategy` (string, optional): Allocation strategy (default: "kelly_criterion")

**Response:**
```json
{
  "allocation_id": "uuid-string",
  "amount": "2500.00",
  "percentage_of_bankroll": 0.025,
  "reasoning": {
    "strategy_used": "kelly_criterion",
    "base_calculation": {...}
  },
  "risk_metrics": {
    "expected_value": 112.50,
    "value_at_risk_95": 2375.00
  },
  "approval_required": false,
  "warnings": [],
  "recommended_stake_sizing": {
    "recommended": "2500.00",
    "conservative": "1250.00",
    "aggressive": "3750.00"
  }
}
```

#### syndicate_distribute_profits
Distributes profits to members based on the configured model.

**Parameters:**
- `syndicate_id` (string, required): Syndicate ID
- `total_profit` (number, required): Total profit to distribute
- `distribution_model` (string, optional): Distribution model (default: "hybrid")
- `authorized_by` (string, required): ID of authorizing member

**Response:**
```json
{
  "distribution_id": "uuid-string",
  "total_profit": "50000.00",
  "operational_reserve": "2500.00",
  "distributed_amount": "47500.00",
  "distributions": {
    "member-uuid": {
      "gross_amount": "10000.00",
      "tax_withheld": "2400.00",
      "net_amount": "7600.00",
      "payment_method": "bank_transfer"
    }
  },
  "distribution_date": "2024-01-20T10:00:00Z"
}
```

### Governance Tools

#### syndicate_create_vote
Creates a governance vote for syndicate decisions.

**Parameters:**
- `syndicate_id` (string, required): Syndicate ID
- `proposal_type` (string, required): Type of proposal
- `proposal_details` (object, required): Proposal details
- `proposed_by` (string, required): ID of proposing member
- `voting_period_hours` (integer, optional): Voting period in hours (default: 24)

**Response:**
```json
{
  "vote_id": "uuid-string",
  "proposal_type": "strategy_change",
  "proposal_details": {...},
  "proposed_by": "member-uuid",
  "created_at": "2024-01-20T10:00:00Z",
  "expires_at": "2024-01-22T10:00:00Z",
  "status": "active"
}
```

### Analytics Tools

#### syndicate_performance_report
Generates comprehensive performance report for the syndicate.

**Parameters:**
- `syndicate_id` (string, required): Syndicate ID
- `period_days` (integer, optional): Report period in days (default: 30)

**Response:**
```json
{
  "syndicate_id": "uuid-string",
  "report_period": {
    "start_date": "2023-12-21T00:00:00Z",
    "end_date": "2024-01-20T23:59:59Z",
    "days": 30
  },
  "financial_performance": {
    "total_return": "11.11",
    "roi": "4.0",
    "sharpe_ratio": 1.85,
    "max_drawdown": -0.08
  },
  "betting_statistics": {
    "total_bets_placed": 156,
    "win_rate": 0.545,
    "by_sport": {...}
  },
  "member_rankings": [...],
  "risk_metrics": {...}
}
```

## Member Roles and Permissions

### Lead Investor
- Full syndicate management capabilities
- Can approve large bets and distribute profits
- Has veto power on major decisions
- Can manage other members

### Senior Analyst
- Can propose bets and create models
- Access to advanced analytics
- Can vote on strategy changes
- Can manage junior analysts

### Junior Analyst
- Can propose bets
- Access to advanced analytics
- Limited voting rights

### Contributing Member
- Can vote on major decisions
- Access to basic analytics
- Can propose ideas
- Can withdraw own funds

### Observer
- View-only access to bets
- No voting rights
- Cannot propose ideas or withdraw funds

## Investment Tiers

- **Bronze**: $1,000 - $5,000
- **Silver**: $5,000 - $25,000
- **Gold**: $25,000 - $100,000
- **Platinum**: $100,000+

Higher tiers receive better profit sharing rates and increased voting weights.

## Bankroll Management Rules

Default bankroll rules include:
- Max single bet: 5% of total bankroll
- Max daily exposure: 20% of bankroll
- Max sport concentration: 40% in one sport
- Minimum reserve: 30% cash reserve
- Daily stop loss: 10%
- Weekly stop loss: 20%

## Allocation Strategies

- **Kelly Criterion**: Mathematical optimization based on edge and probability
- **Fixed Percentage**: Fixed percentage of bankroll per bet
- **Dynamic Confidence**: Allocation based on confidence levels
- **Risk Parity**: Equal risk contribution across bets

## Distribution Models

- **Hybrid**: 50% capital-based, 30% performance-based, 20% equal share
- **Proportional**: Pure capital contribution based
- **Performance Weighted**: Based on member performance metrics
- **Tiered**: Based on member investment tiers

## Error Handling

All tools return consistent error responses:
```json
{
  "error": "Error message describing the issue"
}
```

Common errors:
- Syndicate not found
- Member not found
- Insufficient permissions
- Invalid parameters
- Insufficient funds

## Best Practices

1. **Always verify syndicate exists** before performing operations
2. **Check member permissions** for actions requiring authorization
3. **Monitor exposure levels** to maintain responsible bankroll management
4. **Regular performance reviews** to optimize member contributions
5. **Maintain proper documentation** for tax and compliance purposes

## Integration Example

```python
# Example of using syndicate tools through MCP
import asyncio
from mcp_client import MCPClient

async def manage_syndicate():
    client = MCPClient("http://localhost:8080")
    
    # Create syndicate
    syndicate = await client.call_tool("syndicate_create", {
        "name": "Pro Sports Syndicate",
        "description": "Professional sports betting syndicate",
        "initial_capital": 250000.0
    })
    
    # Add members
    member = await client.call_tool("syndicate_add_member", {
        "syndicate_id": syndicate["syndicate_id"],
        "name": "Expert Analyst",
        "email": "analyst@example.com",
        "role": "senior_analyst",
        "initial_contribution": 50000.0
    })
    
    # Allocate funds for bet
    allocation = await client.call_tool("syndicate_allocate_funds", {
        "syndicate_id": syndicate["syndicate_id"],
        "sport": "NFL",
        "event": "Super Bowl",
        "bet_type": "moneyline",
        "selection": "Chiefs",
        "odds": 2.20,
        "probability": 0.52,
        "edge": 0.055,
        "confidence": 0.90
    })
    
    print(f"Allocated ${allocation['amount']} for bet")
```

## Security Considerations

1. **Authentication**: All requests should be authenticated
2. **Authorization**: Role-based access control enforced
3. **Audit Trail**: All actions are logged for compliance
4. **Data Encryption**: Sensitive financial data should be encrypted
5. **Rate Limiting**: Prevent abuse through rate limiting

## Support

For issues or questions regarding syndicate tools:
- Check error messages for specific issues
- Review logs for detailed error information
- Ensure all required parameters are provided
- Verify member permissions for restricted operations