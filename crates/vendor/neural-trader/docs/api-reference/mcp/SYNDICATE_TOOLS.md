# Syndicate Management MCP Tools Guide

## Overview

The Syndicate Management tools provide comprehensive functionality for creating and managing investment syndicates, enabling collaborative trading and sports betting with advanced risk management, democratic governance, and fair profit distribution.

## Table of Contents

1. [Syndicate Creation and Management](#syndicate-creation-and-management)
2. [Member Management](#member-management)
3. [Capital and Fund Management](#capital-and-fund-management)
4. [Profit Distribution](#profit-distribution)
5. [Voting and Governance](#voting-and-governance)
6. [Performance Tracking](#performance-tracking)
7. [Risk Management](#risk-management)
8. [Integration Examples](#integration-examples)
9. [Error Handling](#error-handling)
10. [Best Practices](#best-practices)

## Syndicate Creation and Management

### 1. syndicate_create

Create a new investment syndicate with initial configuration.

**Parameters:**
- `name` (string, required): Syndicate name
- `description` (string, optional): Syndicate description
- `initial_capital` (number, required): Initial capital amount
- `allocation_strategy` (string, optional): Default allocation strategy (default: "kelly_criterion")
- `risk_limits` (object, optional): Risk management limits

**Response:**
```json
{
  "syndicate_id": "syn_20240626_001",
  "name": "Elite Traders",
  "created_at": "2024-06-26T14:30:00Z",
  "initial_capital": 100000,
  "allocation_strategy": "kelly_criterion",
  "risk_limits": {
    "max_single_bet": 0.05,
    "daily_exposure": 0.20,
    "stop_loss": 0.10
  },
  "status": "active"
}
```

**Example:**
```python
syndicate = await mcp__ai_news_trader__syndicate_create(
    name="Pro Sports Investors",
    initial_capital=250000,
    allocation_strategy="kelly_criterion",
    risk_limits={
        "max_single_bet": 0.03,
        "daily_exposure": 0.15,
        "stop_loss": 0.08
    }
)
```

### 2. syndicate_get_info

Get detailed information about a syndicate.

**Parameters:**
- `syndicate_id` (string, required): Syndicate identifier

**Response:**
```json
{
  "syndicate_id": "syn_20240626_001",
  "name": "Elite Traders",
  "total_capital": 425000,
  "member_count": 12,
  "active_positions": 5,
  "total_return": 0.285,
  "creation_date": "2024-01-15",
  "status": "active",
  "performance": {
    "total_return": 0.285,
    "monthly_return": 0.042,
    "sharpe_ratio": 1.45,
    "max_drawdown": 0.087
  }
}
```

### 3. syndicate_list

List all available syndicates with filtering options.

**Parameters:**
- `status` (string, optional): Filter by status ("active", "suspended", "closed")
- `min_capital` (number, optional): Minimum capital requirement
- `sort_by` (string, optional): Sort criteria (default: "total_return")

**Response:**
```json
{
  "syndicates": [
    {
      "syndicate_id": "syn_20240626_001",
      "name": "Elite Traders",
      "total_capital": 425000,
      "member_count": 12,
      "total_return": 0.285,
      "status": "active"
    }
  ],
  "total_count": 5,
  "active_count": 4
}
```

## Member Management

### 4. syndicate_add_member

Add a new member to the syndicate.

**Parameters:**
- `syndicate_id` (string, required): Syndicate identifier
- `name` (string, required): Member name
- `email` (string, required): Member email
- `role` (string, required): Member role ("lead_investor", "senior_analyst", "junior_analyst", "contributing_member", "observer")
- `contribution` (number, required): Initial capital contribution

**Response:**
```json
{
  "member_id": "mem_20240626_001",
  "syndicate_id": "syn_20240626_001",
  "name": "John Smith",
  "role": "senior_analyst",
  "contribution": 50000,
  "investment_tier": "gold",
  "voting_weight": 0.15,
  "joined_at": "2024-06-26T14:35:00Z"
}
```

### 5. syndicate_update_member

Update member role or status.

**Parameters:**
- `syndicate_id` (string, required): Syndicate identifier
- `member_id` (string, required): Member identifier
- `new_role` (string, optional): New role
- `status` (string, optional): Member status ("active", "suspended")
- `authorized_by` (string, required): ID of authorizing member

**Response:**
```json
{
  "member_id": "mem_20240626_001",
  "previous_role": "junior_analyst",
  "new_role": "senior_analyst",
  "status": "active",
  "updated_at": "2024-06-26T14:40:00Z",
  "updated_by": "mem_20240626_000"
}
```

### 6. syndicate_get_members

Get all members of a syndicate with their details.

**Parameters:**
- `syndicate_id` (string, required): Syndicate identifier
- `include_performance` (boolean, optional): Include performance metrics (default: true)

**Response:**
```json
{
  "syndicate_id": "syn_20240626_001",
  "members": [
    {
      "member_id": "mem_20240626_000",
      "name": "Alice Johnson",
      "role": "lead_investor",
      "contribution": 100000,
      "current_value": 128500,
      "roi": 0.285,
      "voting_weight": 0.35,
      "performance": {
        "win_rate": 0.68,
        "avg_return": 0.045,
        "alpha": 0.032
      }
    }
  ],
  "total_members": 12,
  "total_capital": 425000
}
```

## Capital and Fund Management

### 7. syndicate_allocate_funds

Get fund allocation recommendation for a betting opportunity.

**Parameters:**
- `syndicate_id` (string, required): Syndicate identifier
- `opportunity` (object, required): Betting opportunity details
  - `sport` (string): Sport type
  - `event` (string): Event description
  - `odds` (number): Betting odds
  - `probability` (number): Win probability
  - `edge` (number): Expected edge
  - `confidence` (number): Confidence level
- `strategy` (string, optional): Allocation strategy override

**Response:**
```json
{
  "allocation_id": "alloc_20240626_001",
  "recommended_amount": 8500,
  "percentage_of_bankroll": 0.02,
  "strategy_used": "kelly_criterion",
  "risk_assessment": {
    "risk_score": 0.35,
    "exposure_after": 0.12,
    "within_limits": true
  },
  "approval_required": false,
  "warnings": []
}
```

### 8. syndicate_execute_bet

Execute a syndicate bet with risk checks.

**Parameters:**
- `syndicate_id` (string, required): Syndicate identifier
- `allocation_id` (string, required): Allocation recommendation ID
- `final_amount` (number, optional): Override amount
- `approved_by` (string, required): Approving member ID

**Response:**
```json
{
  "bet_id": "bet_20240626_001",
  "syndicate_id": "syn_20240626_001",
  "sport": "NBA",
  "event": "Lakers vs Celtics",
  "selection": "Lakers -3.5",
  "odds": 1.91,
  "amount": 8500,
  "potential_return": 16235,
  "executed_at": "2024-06-26T14:45:00Z",
  "risk_checks": {
    "position_limit": "passed",
    "daily_exposure": "passed",
    "concentration": "passed"
  }
}
```

### 9. syndicate_get_positions

Get all active positions for a syndicate.

**Parameters:**
- `syndicate_id` (string, required): Syndicate identifier
- `include_settled` (boolean, optional): Include settled bets (default: false)

**Response:**
```json
{
  "syndicate_id": "syn_20240626_001",
  "active_positions": [
    {
      "bet_id": "bet_20240626_001",
      "sport": "NBA",
      "event": "Lakers vs Celtics",
      "selection": "Lakers -3.5",
      "amount": 8500,
      "current_value": 8500,
      "potential_return": 16235,
      "status": "pending",
      "placed_at": "2024-06-26T14:45:00Z"
    }
  ],
  "total_exposure": 42500,
  "total_potential_return": 81200,
  "exposure_percentage": 0.10
}
```

## Profit Distribution

### 10. syndicate_calculate_distribution

Calculate profit distribution for syndicate members.

**Parameters:**
- `syndicate_id` (string, required): Syndicate identifier
- `total_profit` (number, required): Total profit to distribute
- `distribution_model` (string, optional): Distribution model (default: "hybrid")
- `period` (string, optional): Distribution period

**Response:**
```json
{
  "distribution_id": "dist_20240626_001",
  "total_profit": 25000,
  "distribution_model": "hybrid",
  "distributions": [
    {
      "member_id": "mem_20240626_000",
      "member_name": "Alice Johnson",
      "gross_amount": 8750,
      "tax_withheld": 2625,
      "net_amount": 6125,
      "calculation_breakdown": {
        "capital_based": 5000,
        "performance_based": 2500,
        "equal_share": 1250
      }
    }
  ],
  "total_distributed": 25000,
  "tax_withheld_total": 7500
}
```

### 11. syndicate_process_distribution

Execute profit distribution to members.

**Parameters:**
- `syndicate_id` (string, required): Syndicate identifier
- `distribution_id` (string, required): Distribution calculation ID
- `authorized_by` (string, required): Authorizing member ID

**Response:**
```json
{
  "distribution_id": "dist_20240626_001",
  "status": "completed",
  "processed_at": "2024-06-26T15:00:00Z",
  "transactions": [
    {
      "member_id": "mem_20240626_000",
      "amount": 6125,
      "transaction_id": "txn_20240626_001",
      "status": "completed"
    }
  ],
  "summary": {
    "total_distributed": 17500,
    "total_tax_withheld": 7500,
    "successful_transactions": 12,
    "failed_transactions": 0
  }
}
```

### 12. syndicate_request_withdrawal

Request withdrawal from syndicate.

**Parameters:**
- `syndicate_id` (string, required): Syndicate identifier
- `member_id` (string, required): Member identifier
- `amount` (number, required): Withdrawal amount
- `is_emergency` (boolean, optional): Emergency withdrawal flag (default: false)

**Response:**
```json
{
  "withdrawal_id": "with_20240626_001",
  "member_id": "mem_20240626_001",
  "requested_amount": 25000,
  "available_balance": 75000,
  "is_emergency": false,
  "penalty": 0,
  "net_amount": 25000,
  "scheduled_date": "2024-07-03",
  "status": "scheduled"
}
```

## Voting and Governance

### 13. syndicate_create_proposal

Create a voting proposal for syndicate decisions.

**Parameters:**
- `syndicate_id` (string, required): Syndicate identifier
- `proposal_type` (string, required): Type of proposal
- `title` (string, required): Proposal title
- `details` (object, required): Proposal details
- `proposed_by` (string, required): Proposing member ID
- `voting_period_hours` (number, optional): Voting period (default: 48)

**Response:**
```json
{
  "proposal_id": "prop_20240626_001",
  "syndicate_id": "syn_20240626_001",
  "title": "Increase NBA exposure limit",
  "proposal_type": "strategy_change",
  "status": "active",
  "voting_ends": "2024-06-28T14:30:00Z",
  "current_votes": {
    "approve": 0,
    "reject": 0,
    "abstain": 0
  }
}
```

### 14. syndicate_cast_vote

Cast a vote on a syndicate proposal.

**Parameters:**
- `syndicate_id` (string, required): Syndicate identifier
- `proposal_id` (string, required): Proposal identifier
- `member_id` (string, required): Voting member ID
- `vote` (string, required): Vote decision ("approve", "reject", "abstain")

**Response:**
```json
{
  "vote_id": "vote_20240626_001",
  "proposal_id": "prop_20240626_001",
  "member_id": "mem_20240626_000",
  "vote": "approve",
  "voting_weight": 0.35,
  "timestamp": "2024-06-26T15:30:00Z",
  "current_results": {
    "approval_percentage": 35,
    "participation_rate": 25,
    "votes_cast": 3
  }
}
```

### 15. syndicate_get_proposal_results

Get voting results for a proposal.

**Parameters:**
- `syndicate_id` (string, required): Syndicate identifier
- `proposal_id` (string, required): Proposal identifier

**Response:**
```json
{
  "proposal_id": "prop_20240626_001",
  "title": "Increase NBA exposure limit",
  "status": "passed",
  "final_results": {
    "total_votes": 10,
    "approval_percentage": 72.5,
    "rejection_percentage": 20.0,
    "abstention_percentage": 7.5,
    "participation_rate": 83.3
  },
  "vote_breakdown": [
    {
      "member_name": "Alice Johnson",
      "vote": "approve",
      "weight": 0.35
    }
  ],
  "implementation_date": "2024-06-29"
}
```

## Performance Tracking

### 16. syndicate_member_performance

Get detailed performance metrics for a syndicate member.

**Parameters:**
- `syndicate_id` (string, required): Syndicate identifier
- `member_id` (string, required): Member identifier
- `period_days` (number, optional): Analysis period (default: 30)

**Response:**
```json
{
  "member_id": "mem_20240626_001",
  "member_name": "Bob Smith",
  "performance_period": "30_days",
  "betting_performance": {
    "total_bets": 45,
    "winning_bets": 31,
    "win_rate": 0.689,
    "avg_odds": 2.15,
    "roi": 0.285,
    "profit_loss": 14250
  },
  "skill_metrics": {
    "edge_capture": 0.82,
    "consistency_score": 0.75,
    "risk_adjusted_return": 1.45,
    "alpha": 0.048
  },
  "contribution_analysis": {
    "profit_contribution": 0.165,
    "idea_quality_score": 0.88,
    "collaboration_score": 0.92
  },
  "ranking": {
    "overall_rank": 2,
    "roi_rank": 1,
    "consistency_rank": 3
  }
}
```

### 17. syndicate_performance_report

Generate comprehensive syndicate performance report.

**Parameters:**
- `syndicate_id` (string, required): Syndicate identifier
- `period` (string, optional): Report period ("weekly", "monthly", "quarterly")
- `include_member_details` (boolean, optional): Include individual member performance (default: true)

**Response:**
```json
{
  "report_id": "report_20240626_001",
  "syndicate_id": "syn_20240626_001",
  "period": "monthly",
  "summary": {
    "total_return": 0.142,
    "total_profit": 60350,
    "win_rate": 0.645,
    "sharpe_ratio": 1.38,
    "max_drawdown": 0.065
  },
  "sport_breakdown": {
    "NBA": {
      "profit": 28500,
      "roi": 0.19,
      "win_rate": 0.68
    },
    "NFL": {
      "profit": 22300,
      "roi": 0.15,
      "win_rate": 0.62
    }
  },
  "member_performance": [
    {
      "member_name": "Alice Johnson",
      "roi": 0.285,
      "profit_contribution": 0.35,
      "skill_score": 0.92
    }
  ],
  "risk_metrics": {
    "var_95": 0.045,
    "expected_shortfall": 0.062,
    "correlation_risk": "low"
  },
  "recommendations": [
    "Consider increasing NBA exposure based on strong performance",
    "Review risk limits for NFL betting",
    "Schedule quarterly member performance reviews"
  ]
}
```

## Integration Examples

### Complete Syndicate Workflow

```python
import asyncio
from datetime import datetime, timedelta

async def syndicate_workflow_example():
    # 1. Create syndicate
    syndicate = await mcp__ai_news_trader__syndicate_create(
        name="Professional Sports Investors",
        initial_capital=500000,
        allocation_strategy="kelly_criterion"
    )
    
    # 2. Add members
    members = []
    for name, email, role, amount in [
        ("Alice Johnson", "alice@example.com", "lead_investor", 200000),
        ("Bob Smith", "bob@example.com", "senior_analyst", 150000),
        ("Charlie Brown", "charlie@example.com", "junior_analyst", 100000),
        ("Diana Prince", "diana@example.com", "contributing_member", 50000)
    ]:
        member = await mcp__ai_news_trader__syndicate_add_member(
            syndicate_id=syndicate['syndicate_id'],
            name=name,
            email=email,
            role=role,
            contribution=amount
        )
        members.append(member)
    
    # 3. Create betting opportunity
    opportunity = {
        "sport": "NBA",
        "event": "Lakers vs Warriors",
        "bet_type": "spread",
        "selection": "Lakers -3.5",
        "odds": 1.91,
        "probability": 0.58,
        "edge": 0.05,
        "confidence": 0.75
    }
    
    # 4. Get allocation recommendation
    allocation = await mcp__ai_news_trader__syndicate_allocate_funds(
        syndicate_id=syndicate['syndicate_id'],
        opportunity=opportunity
    )
    
    # 5. Execute bet if approved
    if not allocation['approval_required']:
        bet = await mcp__ai_news_trader__syndicate_execute_bet(
            syndicate_id=syndicate['syndicate_id'],
            allocation_id=allocation['allocation_id'],
            approved_by=members[0]['member_id']  # Lead investor
        )
        print(f"Bet placed: ${bet['amount']} on {bet['selection']}")
    
    # 6. Create proposal for strategy change
    proposal = await mcp__ai_news_trader__syndicate_create_proposal(
        syndicate_id=syndicate['syndicate_id'],
        proposal_type="strategy_change",
        title="Increase single bet limit to 7%",
        details={
            "current_limit": "5%",
            "proposed_limit": "7%",
            "rationale": "Strong performance and increased capital base"
        },
        proposed_by=members[1]['member_id'],  # Senior analyst
        voting_period_hours=48
    )
    
    # 7. Cast votes
    for member in members[:3]:  # First 3 members vote
        await mcp__ai_news_trader__syndicate_cast_vote(
            syndicate_id=syndicate['syndicate_id'],
            proposal_id=proposal['proposal_id'],
            member_id=member['member_id'],
            vote="approve"
        )
    
    # 8. Check member performance
    performance = await mcp__ai_news_trader__syndicate_member_performance(
        syndicate_id=syndicate['syndicate_id'],
        member_id=members[1]['member_id'],
        period_days=30
    )
    
    print(f"Member ROI: {performance['betting_performance']['roi']:.2%}")
    
    # 9. Generate syndicate report
    report = await mcp__ai_news_trader__syndicate_performance_report(
        syndicate_id=syndicate['syndicate_id'],
        period="monthly"
    )
    
    print(f"Syndicate monthly return: {report['summary']['total_return']:.2%}")
    
    return syndicate

# Run the workflow
syndicate = asyncio.run(syndicate_workflow_example())
```

### Risk-Managed Betting Example

```python
async def risk_managed_betting():
    # Check current exposure before placing bet
    positions = await mcp__ai_news_trader__syndicate_get_positions(
        syndicate_id="syn_20240626_001"
    )
    
    current_exposure = positions['exposure_percentage']
    
    # Only proceed if within risk limits
    if current_exposure < 0.15:  # 15% limit
        opportunity = {
            "sport": "NFL",
            "event": "Chiefs vs Bills",
            "odds": 2.10,
            "probability": 0.52,
            "edge": 0.042,
            "confidence": 0.70
        }
        
        allocation = await mcp__ai_news_trader__syndicate_allocate_funds(
            syndicate_id="syn_20240626_001",
            opportunity=opportunity,
            strategy="kelly_criterion"
        )
        
        if allocation['risk_assessment']['within_limits']:
            bet = await mcp__ai_news_trader__syndicate_execute_bet(
                syndicate_id="syn_20240626_001",
                allocation_id=allocation['allocation_id'],
                approved_by="mem_20240626_000"
            )
            print(f"Bet placed within risk limits: ${bet['amount']}")
        else:
            print(f"Bet rejected - Risk limits exceeded")
    else:
        print(f"Current exposure too high: {current_exposure:.1%}")
```

### Profit Distribution Example

```python
async def distribute_weekly_profits():
    # Calculate distribution
    distribution = await mcp__ai_news_trader__syndicate_calculate_distribution(
        syndicate_id="syn_20240626_001",
        total_profit=50000,
        distribution_model="hybrid"  # 50% capital, 30% performance, 20% equal
    )
    
    print(f"Total profit to distribute: ${distribution['total_profit']}")
    print(f"Tax withheld: ${distribution['tax_withheld_total']}")
    
    # Show distribution breakdown
    for dist in distribution['distributions']:
        print(f"\n{dist['member_name']}:")
        print(f"  Capital-based: ${dist['calculation_breakdown']['capital_based']}")
        print(f"  Performance-based: ${dist['calculation_breakdown']['performance_based']}")
        print(f"  Equal share: ${dist['calculation_breakdown']['equal_share']}")
        print(f"  Net amount: ${dist['net_amount']}")
    
    # Process distribution
    result = await mcp__ai_news_trader__syndicate_process_distribution(
        syndicate_id="syn_20240626_001",
        distribution_id=distribution['distribution_id'],
        authorized_by="mem_20240626_000"
    )
    
    print(f"\nDistribution completed: {result['summary']['successful_transactions']} transactions")
```

## Error Handling

### Common Error Responses

```json
{
  "error": {
    "code": "INSUFFICIENT_FUNDS",
    "message": "Syndicate has insufficient funds for this allocation",
    "details": {
      "available": 42500,
      "requested": 50000
    }
  }
}
```

### Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `SYNDICATE_NOT_FOUND` | Syndicate ID doesn't exist | Verify syndicate ID |
| `UNAUTHORIZED` | Member lacks permission | Check member role/permissions |
| `INSUFFICIENT_FUNDS` | Not enough capital | Reduce bet size or add funds |
| `RISK_LIMIT_EXCEEDED` | Risk limits breached | Wait or adjust limits |
| `MEMBER_SUSPENDED` | Member is suspended | Contact lead investor |
| `VOTE_CLOSED` | Voting period ended | Wait for next proposal |
| `WITHDRAWAL_LIMIT` | Withdrawal limit exceeded | Reduce amount or wait |

## Best Practices

### 1. Risk Management
- Always check current exposure before placing bets
- Use fractional Kelly Criterion (25% or less)
- Maintain minimum 30% cash reserves
- Set strict stop-loss limits

### 2. Member Management
- Regular performance reviews (monthly)
- Clear role definitions and permissions
- Transparent communication
- Fair voting procedures

### 3. Capital Management
- Diversify across sports and bet types
- Regular profit distributions (weekly/monthly)
- Clear withdrawal policies
- Maintain detailed transaction logs

### 4. Governance
- Document all major decisions
- Allow adequate voting periods (48+ hours)
- Require supermajority for major changes
- Regular strategy reviews

### 5. Security
- Multi-signature approval for large bets
- Regular audits of fund movements
- Secure credential management
- Comprehensive activity logging

## Performance Optimization

### Caching Strategy
- Cache member data for 5 minutes
- Cache performance metrics for 15 minutes
- Real-time updates for positions and balances

### Batch Operations
```python
# Batch member additions
members_to_add = [
    {"name": "John", "email": "john@example.com", "role": "analyst", "contribution": 25000},
    {"name": "Jane", "email": "jane@example.com", "role": "member", "contribution": 10000}
]

for member_data in members_to_add:
    await mcp__ai_news_trader__syndicate_add_member(
        syndicate_id="syn_20240626_001",
        **member_data
    )
```

## Compliance and Regulations

### Tax Considerations
- Automatic tax withholding by jurisdiction
- Support for tax treaty benefits
- Generation of tax forms (1099, W-8BEN)
- Multi-currency support

### Audit Trail
- All transactions logged with timestamps
- Member action history
- Vote records maintained
- Distribution calculations stored

## Troubleshooting

### Common Issues

1. **Allocation Rejected**
   - Check current exposure levels
   - Verify member has approval rights
   - Ensure within single bet limits

2. **Vote Not Counted**
   - Verify voting period still active
   - Check member has voting rights
   - Ensure not duplicate vote

3. **Distribution Failed**
   - Verify all members have valid payment info
   - Check tax withholding calculations
   - Ensure sufficient profits to distribute

## Security Considerations

### Authentication and Authorization
- All syndicate tools require proper authentication
- Role-based access control (RBAC) enforced
- Multi-signature approval for critical operations
- Session management with timeout controls

### Data Protection
- Encrypted storage of sensitive member data
- Secure credential management
- Audit logging of all transactions
- GDPR compliance for member data

### Financial Security
- Segregated fund management
- Daily reconciliation processes
- Automated fraud detection
- Insurance coverage options

### Operational Security
- Rate limiting on all endpoints
- DDoS protection
- Regular security audits
- Incident response procedures

## Quick Reference Card

### Essential Tools
```python
# Create syndicate
mcp__ai_news_trader__syndicate_create(name, initial_capital, allocation_strategy)

# Add member
mcp__ai_news_trader__syndicate_add_member(syndicate_id, name, email, role, contribution)

# Allocate funds
mcp__ai_news_trader__syndicate_allocate_funds(syndicate_id, opportunity, strategy)

# Execute bet
mcp__ai_news_trader__syndicate_execute_bet(syndicate_id, allocation_id, approved_by)

# Calculate distribution
mcp__ai_news_trader__syndicate_calculate_distribution(syndicate_id, total_profit, model)

# Get performance
mcp__ai_news_trader__syndicate_performance_report(syndicate_id, period)
```

### Member Roles
- `lead_investor` - Full control
- `senior_analyst` - Strategy management
- `junior_analyst` - Analysis support
- `contributing_member` - Regular investor
- `observer` - View only

### Distribution Models
- `hybrid` - 50% capital, 30% performance, 20% equal
- `proportional` - Based on capital contribution
- `performance_weighted` - Based on returns
- `tiered` - Based on member tier

## Conclusion

The Syndicate Management MCP tools provide institutional-grade infrastructure for collaborative investment in sports betting and trading. With comprehensive risk management, transparent governance, and fair profit distribution, these tools enable groups to operate professionally and efficiently.

For additional support, refer to:
- [Syndicate System Documentation](../SYNDICATE_SYSTEM.md)
- [MCP Best Practices](MCP_BEST_PRACTICES_SECURITY.md)
- [API Reference](../api/mcp_tools.md)
- [Example Implementations](../examples/mcp_syndicate_examples.py)